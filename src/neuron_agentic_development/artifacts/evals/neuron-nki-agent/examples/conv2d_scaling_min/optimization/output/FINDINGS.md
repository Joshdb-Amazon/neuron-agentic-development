# Conv2D + Scale + Min — Optimization Findings

## Baseline Analysis

The baseline kernel (`conv2d_scale_min_direct`) performs direct convolution without
im2col, processing output row-by-row. For each of 62 output rows, it loads 9 weight
tiles from HBM and extracts 9 input slices via tensor_copy, accumulating matmul
results in PSUM.

**Baseline Metrics (B=8, C_in=64, C_out=128, H=W=64, K=3):**
- total_time: 13.438 ms
- HBM read: 75,520 KB (17.8× reload factor — weights reloaded every row)
- TensorE: 7.97%, GpSimd: 18.60%, total_active: 40.05%
- 60% of time was **idle** (engines waiting for DMA to complete weight loads)
- No spilling (spill_save_bytes = 0)

**Root cause:** Weights [9, 64, 128] reloaded from HBM for every output row. With
62 rows × 8 batches × 9 positions = 4,464 weight loads, this dominated execution time.

---

## Version 1: Hoist Weight Loading

**Bottleneck targeted:** Massive HBM weight reloading (17.8× reload factor)

**Code change:** Load all 9 weight tiles [C_in, C_out] once per batch before the
output row loop, storing them in SBUF. Reuse across all 62 row iterations.

**Before/After:**

| Metric | Baseline | v1 | Change |
|--------|----------|-----|--------|
| total_time (ms) | 13.438 | 3.074 | **-77.1%** |
| HBM read (KB) | 75,520 | 4,352 | -94.2% |
| reload_factor | 17.8× | 1.0× | Eliminated |
| TensorE% | 7.97 | 10.75 | +2.78pp |
| total_active% | 40.05 | 98.19 | +58.14pp |

**Verdict: SUCCESS** — Massive improvement. Eliminated idle wait time. SBUF can easily
hold 9 × [64,128] × 2B = 144KB of weights alongside the input.

---

## Version 2: Fuse Scale and Negate Operations

**Bottleneck targeted:** ScalarE at 17% from 3 tensor_scalar ops per row (scale, negate, negate-back)

**Code change:** Fuse scale (×s) and first negate (×-1) into single multiply by
-scale_factor. Reduces 3 tensor_scalar ops to 2 per output row.

**Before/After:**

| Metric | v1 | v2 | Change |
|--------|-----|-----|--------|
| total_time (ms) | 3.074 | 2.879 | **-6.3%** |
| ScalarE% | 17.00 | 12.14 | -4.86pp |

**Verdict: PARTIAL SUCCESS** — Measurable improvement but below 10% target.
ScalarE overhead reduced. The GpSimd-bound nature limits further gains from
post-processing optimizations.

---

## Version 3: Batch Output DMA Stores

**Bottleneck targeted:** 496 tiny output DMA stores (124 bytes each, far below 32KB target)

**Code change:** Accumulate all row results into a contiguous SBUF buffer, then
perform a single large DMA store per batch instead of 62 tiny stores.

**Before/After:**

| Metric | v2 | v3 | Change |
|--------|-----|-----|--------|
| total_time (ms) | 2.879 | 2.910 | +1.1% |
| DMA count | 520 | 80 | -84.6% |
| DMA avg bytes | 8,688 | 66,305 | +663% |
| HBM read (KB) | 4,352 | 5,120 | +17.6% |

**Verdict: FAILED** — DMA efficiency improved dramatically (fewer, larger transfers)
but total time slightly regressed. The large output buffer [1, 3844] in SBUF increased
memory pressure, causing 17.6% more HBM reads (compiler evicted/reloaded data).
The output stores were NOT the bottleneck — the GpSimd time of ~2.45ms is dominated
by input loading DMAs, which are constant regardless of output store strategy.

**Key learning:** GpSimd absolute time is constant at ~2.4ms across all versions.
The bottleneck is the software-dynamic DMA engine processing the 8 input loads of
[64, 4096] = 512KB each. This is a fundamental bandwidth/instruction-overhead limitation.

---

## Version 4: Multi-Row PSUM Tiling

**Bottleneck targeted:** Per-row post-processing overhead (scalar 12%, vector 5.5%)

**Code change:** Process 2 output rows together by packing them into PSUM free
dimension (2×62=124 ≤ 512). This batches PSUM→SBUF copy, scale, reduce, negate,
and DMA store operations to process 2 rows at once, halving post-processing
instruction count.

**Before/After:**

| Metric | v2 | v4 | Change |
|--------|-----|-----|--------|
| total_time (ms) | 2.879 | **2.661** | **-7.6%** |
| ScalarE% | 12.14 | 7.76 | -4.38pp |
| VectorE% | 5.50 | 3.63 | -1.87pp |
| total_active% | 98.37 | 99.03 | +0.66pp |

**Verdict: PARTIAL SUCCESS** — Best total_time achieved. Post-processing overhead
reduced by batching. Below 10% target for incremental improvement but significant
cumulative gain (80.2% from baseline).

---

## Version 5: Hoist Weights Outside Batch Loop

**Bottleneck targeted:** Weight loading redundancy across batches

**Code change:** Move weight tile loading before the batch affine_range loop so
weights are loaded exactly once for all 8 batches (vs once per batch in v4).

**Before/After:**

| Metric | v4 | v5 | Change |
|--------|-----|-----|--------|
| total_time (ms) | 2.661 | 2.771 | +4.1% |
| HBM read (KB) | 4,624 | 4,240 | -8.3% |
| VectorE% | 3.63 | 17.85 | +14.22pp |

**Verdict: FAILED** — Achieved theoretical minimum HBM reads (4,240KB =
exact inputs+weights size with zero reloading), but total time regressed.
Keeping 9 weight tiles alive across all batch iterations increased SBUF pressure
and forced the compiler to generate different (worse) scheduling. VectorE
jumped from 3.6% to 17.9%, indicating the compiler routed operations differently.

**Key learning:** Minimum HBM traffic ≠ minimum execution time. The compiler's
ability to schedule overlapping operations matters more than raw DMA reduction
when the kernel is instruction-overhead-bound.

---

## Why Further Optimization Is Difficult

The kernel is **GpSimd-bound** (software-dynamic DMA engine). The GpSimd absolute
time is ~2.4ms across all versions, constituting 85-91% of total time. This time
is spent executing DMA instructions to load the 8 input batches ([64, 4096] × 2B =
512KB each).

The effective HBM bandwidth is ~1.8 GB/s vs theoretical ~50 GB/s per core (3.6%
utilization). This low utilization is caused by **DMA instruction overhead** —
each software-dynamic DMA instruction has setup/teardown cycles that dominate
the actual data transfer time for these moderately-sized transfers.

Potential approaches beyond what was attempted:
1. **SPMD parallelism** — Distribute batches across 4 available cores
2. **Hardware-dynamic DMA** — Use patterns the compiler can map to hardware DMA
3. **Larger data granularity** — Process bigger problem sizes where DMA overhead
   is better amortized (the 64×64 spatial dims are relatively small)

---

## Summary

| Version | total_time | Δ from baseline | Status |
|---------|-----------|----------------|--------|
| baseline | 13.438 ms | — | BASELINE |
| v1 (weight hoist) | 3.074 ms | **-77.1%** | SUCCESS |
| v2 (fuse ops) | 2.879 ms | -78.6% | IMPROVED |
| v3 (batch stores) | 2.910 ms | -78.3% | REGRESSED |
| **v4 (multi-row tile)** | **2.661 ms** | **-80.2%** | **BEST** |
| v5 (weight outside batch) | 2.771 ms | -79.4% | REGRESSED |

**Best overall improvement: 80.2% total_time reduction (13.438ms → 2.661ms).**
v1 alone accounts for 77.1% of the improvement by eliminating weight reloading.
