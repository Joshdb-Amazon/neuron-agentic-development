# Matmul + Subtract + Multiply + ReLU — Optimization Findings

## Kernel Overview

Computes: `relu((x @ weight_t + bias - subtract_value) * multiply_value)`
- Dimensions: M=256 (batch), K=4096 (contraction), N=4096 (output features)
- Tile sizes: TILE_M=128 (P_MAX), TILE_K=128, TILE_N=512
- Tiling: 2 M tiles, 32 K tiles, 8 N tiles

---

## Baseline Profile

| Metric | Value |
|--------|-------|
| total_time | 649.9 us |
| tensor_engine | 48.9% |
| dma_active | 90.7% |
| hbm_read | 83.9 MB |
| hbm_write | 2.1 MB |
| mfu | 16.81% |
| spill | 0 |
| theoretical min HBM read | 35.7 MB |
| read amplification | 2.35x |

**Bottleneck**: DMA-bound. The M->N->K loop structure causes x_t tiles (which depend only
on m,k) to be loaded 8x redundantly (once per N tile). Weight tiles loaded 2x (once per M tile).

---

## V1: x_t Load Hoisting

**Bottleneck targeted**: Redundant x_t loads (16 MB actual vs 2 MB minimum)

**Code change**: Pre-load all K tiles of x_t into SBUF list before the N loop.
x_t tiles are allocated during `affine_range(num_k_tiles)` and appended to a Python list.
The N loop reuses these pre-loaded tiles via list indexing with `affine_range` variable.

```python
# Before (baseline): x_t loaded inside N loop (8x redundant)
for m_idx in affine_range(num_m_tiles):
    for n_idx in affine_range(num_n_tiles):
        for k_idx in affine_range(num_k_tiles):
            lhs = dma_copy(x_t[k, m])  # loaded 2*8*32 = 512 times

# After (V1): x_t hoisted outside N loop
for m_idx in affine_range(num_m_tiles):
    lhsT_tiles = [dma_copy(x_t[k, m]) for k in affine_range]  # loaded 2*32 = 64 times
    for n_idx in affine_range(num_n_tiles):
        rhs_tiles = [dma_copy(weight[k, n]) for k in affine_range]
        for k_idx in affine_range(num_k_tiles):
            nc_matmul(psum, lhsT_tiles[k], rhs_tiles[k])
```

**SBUF budget**: lhsT_tiles = 32 * 32KB = 1MB, rhs_tiles = 32 * 128KB = 4MB = 5MB total (24MB available)

| Metric | Baseline | V1 | Change |
|--------|----------|----|--------|
| total_time (us) | 649.9 | 392.0 | **-39.7%** |
| tensor_engine | 48.9% | 68.0% | +19.1pp |
| hbm_read (MB) | 83.9 | 69.2 | -17.5% |
| mfu | 16.81% | 27.86% | +11.05pp |

**Result**: SUCCESS. Major reduction from eliminating 8x redundant x_t loads.

---

## V2: N-outer/M-inner with Weight Hoisting

**Bottleneck targeted**: Weight loaded 2x (once per M tile) = 64 MB. With N-outer, weight
loads once per N tile = 32 MB.

**Code change**: Swap M and N loop order. N becomes outer, M becomes inner. Weight tiles
pre-loaded per N iteration and reused across M iterations. Bias also hoisted per-N.

Trade-off: x_t now loaded per (N, M) pair = 16 MB (up from V1's 2 MB), but weight drops
from 64 MB to 32 MB. Net: ~48 MB vs 66 MB.

| Metric | V1 | V2 | Change |
|--------|----|----|--------|
| total_time (us) | 392.0 | 306.5 | **-21.8%** |
| tensor_engine | 68.0% | 63.6% | -4.4pp |
| hbm_read (MB) | 69.2 | 39.9 | -42.3% |
| mfu | 27.86% | 35.64% | +7.78pp |

**Result**: SUCCESS. Weight hoisting + N-outer reduced total HBM reads by 42%.

---

## V3: Fused Activation (subtract + multiply + relu)

**Bottleneck targeted**: Epilogue instruction count. Fuse `tensor_scalar(sub, mul)` +
`activation(relu)` into single `activation(relu, scale=mul, bias=-sub*mul)`.

**Code change**: Replace two epilogue instructions with one:
- Removed: `nisa.tensor_scalar(op0=add, operand0=-sub, op1=mul, operand1=mul_val)`
- Replaced: `nisa.activation(op=relu, scale=mul_val, bias=-sub*mul_val)`

Note: `activation.bias` requires a tensor, not a scalar. Created `(P_MAX, 1)` SBUF tile
with `nisa.memset`.

| Metric | V2 | V3 | Change |
|--------|----|----|--------|
| total_time (us) | 306.5 | 305.6 | -0.3% |
| scalar_engine | 3.8% | 4.8% | +1.0pp |
| vector_engine | 19.5% | 15.3% | -4.2pp |

**Result**: MARGINAL. Epilogue is not the bottleneck — DMA dominates at 87.1%. The instruction
fusion saved negligible time. Documented as a valid optimization that didn't apply here.

---

## V4: Full x_t Hoisting (Outside All Loops)

**Bottleneck targeted**: x_t reloaded per (N, M) = 8*2*32 = 512 loads = 16 MB.
By hoisting x_t outside ALL loops, x_t loaded only 64 times = 2 MB.

**Code change**: Pre-load ALL x_t tiles into a 2D Python list `all_lhsT[m][k]` before the
main N-outer loop. Weight hoisting unchanged (per-N).

```python
all_lhsT = []
for m_idx in affine_range(num_m_tiles):
    m_tiles = []
    for k_idx in affine_range(num_k_tiles):
        tile = dma_copy(x_t[k, m])
        m_tiles.append(tile)
    all_lhsT.append(m_tiles)

for n_idx in affine_range(num_n_tiles):
    rhs_tiles = [load weight per N]
    for m_idx in affine_range(num_m_tiles):
        for k_idx in affine_range(num_k_tiles):
            nc_matmul(psum, all_lhsT[m_idx][k_idx], rhs_tiles[k_idx])
```

**SBUF budget**: x_t = 2*32*32KB = 2MB, weight = 32*128KB = 4MB = 6MB total

| Metric | V3 | V4 | Change |
|--------|----|----|--------|
| total_time (us) | 305.6 | 226.4 | **-25.9%** |
| hbm_read (MB) | 39.9 | 35.7 | -10.5% |
| mfu | 35.74% | 48.25% | +12.51pp |
| dma_transfer_count | — | 344 | — |

**Result**: SUCCESS. HBM reads now at theoretical minimum (35.7 MB = inputs + weights).
No more redundant reads of any operand.

---

## V5: Fused PSUM Copy + Bias Add

**Bottleneck targeted**: Epilogue instruction count. Replace separate `tensor_copy(psum->sbuf)`
+ `tensor_tensor(sbuf+bias)` with single `tensor_tensor(psum+bias->sbuf)`.

**Code change**:
- Removed: `nisa.tensor_copy(dst=result_sb, src=psum_tile)`
- Changed: `nisa.tensor_tensor(dst=result_sb, data1=psum_tile, data2=bias_all, op=nl.add)`
  reads directly from PSUM (data1) and SBUF (data2).

| Metric | V4 | V5 | Change |
|--------|----|----|--------|
| total_time (us) | 226.4 | 225.9 | -0.2% |
| tensor_engine | 68.4% | 68.9% | +0.5pp |
| vector_engine | 20.5% | 15.9% | -4.6pp |

**Result**: MARGINAL. Same as V3 — epilogue optimizations don't move the needle when DMA
is at 90.8% active. The kernel is fundamentally DMA-latency-bound.

---

## Why Further Optimization is Difficult

At V5, the kernel has:
1. **HBM reads at theoretical minimum** (35.7 MB = exact input + weight size)
2. **Zero spill/reload**
3. **All operands hoisted to maximum reuse** (x_t loaded once, weight once per N tile)

The remaining bottleneck is DMA latency/overhead:
- 344 DMA transfers averaging 110 KB each
- DMA active 90.8% of total time
- Effective bandwidth ~174 GB/s vs ~716 GB/s peak per NC (24% utilization)

To improve further would require:
- **K-dimension blocking** (interleave DMA + compute) — blocked by inability to index
  pre-loaded lists with `sequential_range` traced values
- **FP8 precision** — would double TensorE throughput but requires data format changes
- **Larger problem size** — larger M dimension would better amortize DMA overhead

---

## Final Summary

| Version | total_time (us) | vs Baseline | Status |
|---------|----------------|-------------|--------|
| Baseline | 649.9 | — | BASELINE |
| V1 | 392.0 | -39.7% | IMPROVED |
| V2 | 306.5 | -52.8% | IMPROVED |
| V3 | 305.6 | -53.0% | MARGINAL |
| V4 | 226.4 | -65.2% | IMPROVED |
| V5 | 225.9 | -65.3% | CURRENT (best) |
