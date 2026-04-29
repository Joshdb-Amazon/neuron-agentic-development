# Cumsum Kernel Optimization — Findings

## Baseline Analysis

**Kernel**: `cumsum_kernel` — computes `torch.cumsum(x, dim=-1)` on a `(32768, 32768)` float32 tensor using `nisa.tensor_tensor_scan`.

**Baseline metrics** (from `neuron-profile` + `neuron-explorer`):

| Metric | Value |
|--------|-------|
| total_time | 21.521 ms |
| vector_engine_active_time_percent | 85.5% |
| dma_active_time | 21.121 ms (98.1%) |
| tensor_engine_active_time_percent | 0.015% (unused) |
| hbm_read_bytes | 4.29 GB |
| hbm_write_bytes | 4.29 GB |
| dma_transfer_count | 8192 |

**Bottleneck classification**: **DMA-BOUND**. DMA active time is 98.1% of total time. The vector engine (running `tensor_tensor_scan`) is at 85.5%, indicating good but not perfect overlap with DMA. No tensor engine usage (no matmul operations).

**Instruction breakdown** (from neuron-explorer):
- TENSOR_TENSOR_SCAN: 4096 instructions, 18.37ms total (= 256 p_tiles × 16 f_tiles)
- Vector EVENT_SEMAPHORE: 98.2ms total wait (vector waiting for DMA)
- GpSimd EVENT_SEMAPHORE: 76.4ms total wait (DMA waiting for vector)
- DMA_DIRECT2D: 8192 instructions (4096 loads + 4096 stores)
- Vector COPY: 254 carry-copy instructions, 39µs

The mutual waiting between Vector and GpSimd engines indicated a DMA/compute overlap opportunity.

---

## V1: Double Buffering (SUCCESS — -4.7%)

**Target bottleneck**: DMA/compute sync stalls (Vector and GpSimd engines waiting on each other).

**Change**: Added two sets of data/result SBUF buffers (ping-pong). Prefetch the next tile's data while the vector engine scans the current tile. This overlaps DMA latency with compute.

**Results**:

| Metric | Baseline | V1 | Change |
|--------|----------|-----|--------|
| total_time | 21.521 ms | 20.507 ms | **-4.7%** |
| vector_engine | 85.5% | 89.6% | +4.1pp |
| dma_active_time | 21.121 ms | 20.498 ms (100%) | -2.9% |
| dma_transfer_count | 8192 | 8192 | — |

**Instruction-level impact**:
- Vector SCAN wait: 37µs → 1µs (data always pre-fetched)
- Vector COPY: 254 → 0 (carry copies eliminated by compiler)
- DMA utilization: 98.1% → 100%

**Verdict**: SUCCESS. Double buffering eliminated DMA/compute sync stalls. The vector engine no longer waits for data loads. However, the improvement is limited because the kernel is fundamentally DMA-bandwidth-limited.

---

## V2: Deferred Store Overlap (MARGINAL)

**Target bottleneck**: Store latency overlapping with next scan.

**Change**: Reordered DMA operations to issue the store of the previous tile's result before scanning the current tile, attempting to overlap store DMA with vector compute.

**Results**:

| Metric | V1 | V2 | Change |
|--------|-----|-----|--------|
| total_time | 20.507 ms | 20.598 ms | +0.4% (no improvement) |
| vector_engine | 89.6% | 89.2% | -0.4pp |

**Verdict**: MARGINAL. The compiler already handles store scheduling optimally with V1's double buffering. Explicit store reordering adds no benefit.

---

## V3: Larger Tiles + Double Buffering (REGRESSION)

**Target bottleneck**: Per-tile overhead (hypothesis: fewer, larger tiles reduce synchronization).

**Change**: Increased F_TILE_SIZE from 2048 to 4096 with double buffering. Reduces sequential iterations from 16 to 8 per partition tile.

**Results**:

| Metric | V1 | V3 | Change |
|--------|-----|-----|--------|
| total_time | 20.507 ms | 22.298 ms | **+8.7% REGRESSION** |
| vector_engine | 89.6% | 80.4% | -9.2pp |
| dma_active_time | 20.498 ms | 21.598 ms | +5.4% |
| dma_transfer_count | 8192 | 4096 | -50% |

**Verdict**: REGRESSION. Despite fewer DMA transfers, larger 2MB tiles are less efficient than 1MB tiles for this access pattern (128 rows × 4096 cols vs 128 rows × 2048 cols). The DMA engine handles smaller contiguous transfers more efficiently, and the vector engine gets worse pipeline utilization with longer scan operations.

**Lesson**: For 2D tiled DMA, there's an optimal transfer size. Bigger is not always better.

---

## V4: Smaller Tiles + Double Buffering (SLIGHT REGRESSION)

**Target bottleneck**: Pipeline granularity (hypothesis: smaller tiles allow more frequent DMA/compute interleaving).

**Change**: Reduced F_TILE_SIZE to 1024 with double buffering. Creates 32 sequential iterations per partition tile (vs 16 baseline).

**Results**:

| Metric | V1 | V4 | Change |
|--------|-----|-----|--------|
| total_time | 20.507 ms | 20.778 ms | +1.3% |
| vector_engine | 89.6% | 92.6% | +3.0pp |
| dma_transfer_count | 8192 | 16384 | +100% |

**Verdict**: SLIGHT REGRESSION. Higher vector utilization (92.6%, best of all versions) but more DMA overhead from 16K transfers vs 8K. The per-transfer overhead of DMA_DIRECT2D becomes significant with smaller tiles. F_TILE_SIZE=2048 remains the sweet spot.

---

## V5: affine_range + Double Buffering (NO CHANGE)

**Target bottleneck**: Compiler optimization — TiledRange unrolls 256 iterations at trace time, producing a large NEFF. affine_range creates a single compact loop body.

**Change**: Replaced TiledRange Python loop with nl.affine_range for the outer partition loop. Combined with V1's double buffering.

**Results**:

| Metric | V1 | V5 | Change |
|--------|-----|-----|--------|
| total_time | 20.507 ms | 20.507 ms | 0% |
| vector_engine | 89.6% | 89.6% | — |

**Verdict**: NO CHANGE. The compiler generates identical runtime behavior whether the loop is Python-unrolled (TiledRange) or compiler-managed (affine_range). The NEFF may be smaller, but execution performance is identical.

---

## Why 10% Improvement Was Not Achievable

The cumsum kernel is **fundamentally DMA-bandwidth-limited**:

1. **Irreducible DMA volume**: The computation requires reading 4GB (input) and writing 4GB (output) = 8GB total. No redundant reads/writes exist.

2. **Near-peak bandwidth**: At V1's 20.5ms for 8GB, the effective bandwidth is ~390 GB/s, which is approximately the per-NeuronCore HBM bandwidth on trn2.

3. **Compute is not the bottleneck**: The 4096 TENSOR_TENSOR_SCAN operations take only 18.37ms (vector engine), which fits within the 20.5ms DMA time.

4. **No algorithmic alternatives**: A "scan-then-correct" approach (independent per-tile scans + correction pass) would require 12-16GB of DMA (1.5-2× more), making it slower.

5. **Tile size sensitivity**: DMA efficiency peaks at F_TILE_SIZE=2048 (128×2048×4 = 1MB transfers). Both smaller and larger tiles degrade performance.

The 4.7% improvement from double buffering represents the gap between "sequential DMA/compute" and "overlapped DMA/compute". Further improvement would require higher HBM bandwidth hardware or a fundamentally different algorithm that reduces DMA volume.
