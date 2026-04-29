# GELU Kernel Optimization — Findings

## Baseline Analysis

**Kernel**: Element-wise GELU activation on (2048, 8192) float32 tensor.
**Pattern**: Pure elementwise (read → compute → write). Memory-bound by nature.

**Baseline Metrics**:
- total_time: **332.17 µs**
- DMA active: 96.90% (321.88 µs)
- ScalarE active: 35.53% (118.00 µs)
- MBU: 56.43%
- HBM traffic: 128 MB (64 MB read + 64 MB write)
- DMA transfers: 32 × 4 MB each
- DMA parallelism: ~15.7× across 16 engines

**Bottleneck Classification**: **DMA-BOUND**

The kernel is fundamentally bandwidth-limited. DMA consumes 97% of execution
time while the ScalarE GELU computation takes only 35.5%. The MBU of 56%
reflects the practical bandwidth limit for a single-NeuronCore read-compute-write
pattern where 16 DMA engines alternate between loads and stores.

---

## Version 1: 2D Tiling (128×2048 tiles)

**Hypothesis**: Smaller tiles (4× more) create more concurrent DMA operations,
enabling better DMA/ScalarE interleaving and pipeline overlap.

**Implementation**: Split 8192-wide free dimension into 4 tiles of 2048. Nested
`affine_range` loops for rows × columns.

**Results**:

| Metric | Baseline | V1 | Change |
|--------|----------|------|--------|
| total_time (µs) | 332.17 | 332.94 | +0.2% |
| DMA active (%) | 96.90 | 97.98 | +1.1pp |
| MBU (%) | 56.43 | 56.30 | -0.1pp |
| DMA xfer count | 32 | 128 | 4× more |
| DMA avg bytes (KB) | 4096 | 1024 | 4× smaller |

**Outcome**: FAILED — No improvement. More but smaller DMA transfers didn't
improve bandwidth utilization. The baseline's 4MB transfers are already at the
DMA throughput saturation point (32KB+ per partition).

---

## Version 2: Double Buffering with sequential_range

**Hypothesis**: Explicit ping-pong buffers with `sequential_range` overlap
load(N+1) with compute(N), hiding ScalarE latency behind DMA.

**Implementation**: Two sets of in/out buffers. Prefetch first tile, then
loop: prefetch next + compute current + store current.

**Results**:

| Metric | Baseline | V2 | Change |
|--------|----------|------|--------|
| total_time (µs) | 332.17 | 330.97 | -0.4% |
| DMA active (%) | 96.90 | 97.21 | +0.3pp |
| MBU (%) | 56.43 | 56.64 | +0.2pp |
| ScalarE (%) | 35.53 | 35.60 | +0.1pp |

**Outcome**: MARGINAL — 0.4% improvement. DMA dominates at 97% and ScalarE
(35.5%) is already well-hidden behind DMA. Overlapping 35% compute with 97%
DMA yields minimal benefit.

---

## Version 3: In-Place Activation

**Hypothesis**: Using same buffer for activation input and output halves SBUF
footprint (4MB vs 8MB per tile), allowing more tiles in-flight simultaneously.

**Implementation**: `nisa.activation(dst=tile, op=nl.gelu, data=tile)` — write
GELU result to the same SBUF buffer as the input.

**Results**:

| Metric | Baseline | V3 | Change |
|--------|----------|------|--------|
| total_time (µs) | 332.17 | 330.13 | -0.6% |
| DMA active (%) | 96.90 | 97.11 | +0.2pp |
| MBU (%) | 56.43 | 56.78 | +0.4pp |

**Outcome**: MARGINAL — 0.6% improvement. SBUF savings don't translate to
meaningful DMA improvement because the bandwidth limit, not SBUF capacity,
is the bottleneck.

---

## Version 4: Hardware DGE + In-Place Activation

**Hypothesis**: Hardware DMA Gather Engine (hwdge) generates DMA descriptors
on-demand without consuming SBUF or GpSimdE cycles. Combined with in-place
activation, this removes software DGE overhead entirely.

**Implementation**: Added `dge_mode=nisa.dge_mode.hwdge` to all `nisa.dma_copy`
calls. In-place activation.

**Results**:

| Metric | Baseline | V4 | Change |
|--------|----------|------|--------|
| total_time (µs) | 332.17 | 329.37 | -0.8% |
| DMA active (%) | 96.90 | 96.94 | +0.0pp |
| MBU (%) | 56.43 | 56.91 | +0.5pp |
| HW DGE (%) | 0.00 | 98.31 | new |
| SW DGE (%) | 98.60 | 0.00 | eliminated |
| GpSimdE instr | 70 | 27 | -61% |

**Outcome**: MARGINAL — 0.8% improvement. Hardware DGE successfully eliminated
software DGE overhead and reduced GpSimdE instructions by 61%, but the DMA
bandwidth itself (not descriptor generation) is the true bottleneck.

**Note**: An earlier attempt at v4 using phased load/compute/store with
128×1024 column tiles REGRESSED to 361.18 µs (+8.7%) due to 256 smaller
(512KB) DMA transfers. This confirmed that the baseline's 4MB tile size is
near-optimal for DMA throughput.

---

## Version 5: Split affine_range Blocks + HW DGE + In-Place

**Hypothesis**: Splitting 16 iterations across 4 separate `affine_range` blocks
of 4 reduces compiler unrolling pressure while maintaining large-tile DMA
efficiency. Combined with hw_dge and in-place activation.

**Implementation**: 4 explicit `affine_range(4)` blocks processing rows 0-3,
4-7, 8-11, 12-15 respectively. Hardware DGE + in-place activation.

**Results**:

| Metric | Baseline | V5 | Change |
|--------|----------|------|--------|
| total_time (µs) | 332.17 | 328.85 | **-1.0%** |
| DMA active (%) | 96.90 | 96.83 | -0.1pp |
| MBU (%) | 56.43 | 57.00 | +0.6pp |
| DMA active time (µs) | 321.88 | 318.43 | -1.1% |

**Outcome**: BEST RESULT — 1.0% improvement (3.32 µs). Combines all effective
techniques. MBU increased from 56.43% to 57.00%.

**Note**: An alternative v5 using column-stripe traversal (128×512 tiles) with
hw_dge REGRESSED to 407.82 µs (+23%) due to 511 smaller DMA transfers and
MBU dropping to 45.97%.

---

## Summary: Why 10% Improvement Is Not Achievable

This kernel is **fundamentally DMA-bandwidth limited** for a standalone
elementwise operation on a single NeuronCore. The evidence:

1. **Arithmetic intensity = 0** (no matmul operations). The kernel sits far
   below the Roofline knee. Per the NKI Performance Guide: element-wise kernels
   "can only improve significantly via operator fusion to increase arithmetic
   intensity."

2. **DMA dominates at 97%**. ScalarE compute (35%) is already well-hidden.
   Even making compute instant would only save ~3% of total time.

3. **MBU at 56-57%** reflects the practical limit for read-compute-write on
   one NeuronCore. The 16 DMA engines achieve 15.7× parallelism (near-perfect),
   but alternating between loads and stores creates inherent inefficiency.

4. **DMA transfers are already optimally sized**: 4MB (32KB per partition),
   well above the 4KB saturation threshold. Smaller tiles consistently
   degraded performance.

5. **No fusion opportunity**: GELU is a standalone operation. Without adjacent
   operations to fuse (e.g., preceding matmul, following normalization), the
   kernel must read and write the full 128MB.

**Theoretical minimum**: 128MB / peak_bandwidth ≈ 188 µs (MBU=100%).
The 57% MBU at 329 µs represents about 1.75× overhead, which is typical
for single-pass elementwise kernels competing for memory bandwidth between
reads and writes.

**To achieve 10%+ improvement would require**:
- Operator fusion with adjacent operations (fused_gelu_add, etc.)
- Multi-core SPMD distribution (4 cores = ~4× throughput)
- Using reduced precision types (bf16 halves memory traffic)
All of which change the kernel's interface or precision, which is prohibited.
