# RMS Fused QKV — Optimization Findings

## Baseline Analysis

- **total_time: 275.96 µs**
- Bottleneck: **Vector Engine at 81.4%** (1067 instructions)
- Tensor Engine: 49.3% (128 matmul instructions, 322 total)
- DMA: 15.4% (underutilized)
- MFU: 19.8%
- HBM Read: 10 MB, Write: 1 MB

Root cause: The manual 32x32 `nc_transpose` tiling generates ~1024 vector engine
instructions (16 sub-tiles × 16 k_tiles × 2 n_tiles × 2 s_tiles). These dominate
the vector engine, which is the critical path.

---

## V1: DMA Transpose (SUCCESS — 59.7% reduction)

**Bottleneck targeted:** Vector engine overload from nc_transpose
**Strategy:** Replace 16× nc_transpose sub-tile calls with `nisa.dma_transpose` via
4D reshape. The input chunk is loaded from HBM with a 4D view `[S, 1, 1, K]` and
`dma_transpose` applies permutation [3,1,2,0] to produce `[K, 1, 1, S]` (transposed).

**Code change:** Removed the 4×4 nc_transpose loop. Added `hidden_4d = hidden_hbm.reshape((BS, 1, 1, D))` and used `nisa.dma_transpose(dst=stat_4d, src=hidden_4d[...])`.

| Metric | Baseline | V1 | Change |
|--------|----------|----|--------|
| total_time (µs) | 275.96 | 111.24 | **-59.7%** |
| Vector Engine % | 81.4 | 15.7 | -65.7pp |
| Vector Instr | 1067 | 43 | -96% |
| DMA Active % | 15.4 | 63.2 | +47.8pp |
| HBM Read (KB) | 10240 | 12288 | +20% |
| MFU % | 19.8 | 49.1 | +29.3pp |

**Result:** Massive success. Eliminated ~1024 vector instructions. Shifted transpose work from overloaded vector engine to underutilized DMA engine. The 2 MB extra HBM reads (DMA transpose loads vs SBUF-resident data) are a minor cost for a 60% total_time reduction. Bottleneck shifted to DMA (63%).

---

## V2: Incremental RMSNorm (SUCCESS — 7.2% reduction over V1)

**Bottleneck targeted:** SBUF pressure from V1's full input_sb [128, 2048] + sq_sb [128, 2048] buffers (2 MB SBUF) limiting compiler scheduling.
**Strategy:** Compute inv_rms incrementally per K-chunk instead of loading the full input. Each chunk is [128, 128] (64 KB). Eliminates both the input_sb and sq_sb large buffers.

**Code change:** Replaced full-width load + square + reduce with a loop: load [128,128] chunk, square, reduce, accumulate sum_sq.

| Metric | V1 | V2 | Change |
|--------|----|----|--------|
| total_time (µs) | 111.24 | 103.27 | **-7.2%** |
| Vector Instr | 43 | 137 | +219% |
| MFU % | 49.1 | 52.9 | +3.8pp |

**Result:** Improved despite more vector instructions. The SBUF pressure reduction enabled better compiler scheduling and DMA-compute overlap. DMA remains the bottleneck.

---

## V3: Fused PSUM Multiply (NEUTRAL)

**Bottleneck targeted:** Vector engine overhead from separate tensor_copy + tensor_tensor per n_tile.
**Strategy:** Use `nisa.tensor_tensor` directly with PSUM as data1 operand, fusing the copy and multiply into a single operation.

| Metric | V2 | V3 | Change |
|--------|----|----|--------|
| total_time (µs) | 103.27 | 104.24 | **+0.9%** |
| Vector Instr | 137 | 133 | -3% |

**Result:** Within measurement noise. The fused operation saves 4 vector instructions per s_tile but the DMA bottleneck dominates total_time. Vector engine is no longer on the critical path.

---

## V4: In-Place Squaring + Fused PSUM (NEUTRAL)

**Bottleneck targeted:** Further SBUF pressure reduction.
**Strategy:** Square the input chunk in-place (no separate chunk_sq buffer). Combined with v3's fused PSUM multiply.

| Metric | V2 | V4 | Change |
|--------|----|----|--------|
| total_time (µs) | 103.27 | 104.16 | **+0.9%** |

**Result:** No improvement. SBUF microoptimizations don't help when DMA is the bottleneck at 65%. These are secondary effects below the measurement threshold.

---

## V5: Manually Unrolled N-Tiles (SUCCESS — 7.3% reduction over V2)

**Bottleneck targeted:** DMA serialization from processing n_tiles sequentially.
**Strategy:** Manually unroll the n_tile loop: allocate separate PSUM accumulators (psum_n0, psum_n1) outside the k_tile loop, process both n_tiles in a single k_tile pass. Each `dma_transpose` is shared by both n_tile matmuls, and weight loads for both n_tiles interleave with compute.

**Code change:** Replaced `for n_idx in affine_range(2): for k_idx: ...` with `for k_idx: matmul_n0; matmul_n1`. PSUM accumulators allocated at s_tile scope.

| Metric | V2 | V5 | Change |
|--------|----|----|--------|
| total_time (µs) | 103.27 | 95.78 | **-7.3%** |
| Tensor Engine % | 59.3 | 62.5 | +3.2pp |
| DMA Active % | 63.0 | 70.5 | +7.5pp |
| MFU % | 52.9 | 57.0 | +4.1pp |

**Result:** Better DMA-compute pipelining from processing both n_tiles per k_tile iteration. The DMA transpose and two weight loads interleave with the two matmuls, keeping both DMA and tensor engines busy. DMA utilization increases to 70.5% indicating better overall pipeline saturation.

---

## Failed Approaches (documented for completeness)

1. **Loop reorder (k outer, n inner)** — Failed with PSUM scope error. PSUM allocated in inner loop can't be read from a separate Phase 3 outside the loop. NKI compiler treats separate allocations as distinct tensors.

2. **nl.load_transpose2d** — Not supported in current SDK release.

3. **nl.transpose** — Not supported in current SDK release.

4. **nl.matmul** — Not supported in current SDK release.

5. **nl.loop_reduce** — Not supported in current SDK release.

6. **bf16 matmul** — Compiled successfully but failed correctness (max_diff 9.6e-3 > tolerance). The bf16 precision loss on the input/weight data exceeded the test's atol=1e-3, rtol=1e-2 thresholds.

7. **Pre-normalize + HBM buffer** — V3 variant that wrote normalized input to HBM then read via dma_transpose. Regressed to 128 µs due to extra 2 MB HBM writes.

---

## Summary

The optimization progression achieved a **65.3% reduction** in total_time (275.96 µs → 95.78 µs). The key breakthrough was V1's `dma_transpose` via 4D reshape, which eliminated ~1024 vector engine instructions by shifting transpose work to the DMA engine. Subsequent optimizations (V2's incremental RMSNorm for SBUF pressure reduction, V5's manual n_tile unrolling for better pipelining) provided additional 14% improvement. The kernel is now DMA-bound at 70.5%, with MFU at 57.0%.
