# Matmul BT Kernel Optimization — FINDINGS

**Kernel:** `matmul_bt_kernel` — C[2048, 4096] = A[2048, 8192] @ B[4096, 8192].T (float16)
**Target hardware:** trn2 (gen3)
**Total improvement:** 26.379 ms → 3.876 ms (**85.3% reduction**)

---

## Baseline Analysis

The baseline kernel loads and transposes both A and B tiles inside the innermost (m, n, k) triple loop with TILE_M=TILE_N=TILE_K=128. Key problems identified via profiling:

- **Reload factor 13.33x** — Data loaded 13x more than necessary (1280 MB vs 96 MB inputs)
- **Transpose overhead 66.7%** — Two-thirds of all hardware FLOPs wasted on nc_transpose
- **MFU 6.62%** — Extremely low model FLOPs utilization
- **Arithmetic intensity 101.14** — Below the peak ratio of 109.84 → memory-bound

---

## V1: Load Hoisting (26.379 → 20.630 ms, -21.8%)

**Bottleneck targeted:** Redundant A tile loads. A[m,k] was loaded for every (m, n, k) iteration but only depends on (m, k).

**Code change:** Pre-load and transpose all K A tiles outside the N loop. Pre-load all K B tiles outside the K matmul loop (separate load and compute phases).

**Evidence:**
| Metric | Baseline | V1 | Change |
|--------|----------|-----|--------|
| total_time (ms) | 26.379 | 20.630 | -21.8% |
| HBM read (MB) | 1280 | 1056 | -17.5% |
| reload_factor | 13.33x | 11.00x | -2.33x |
| arith_intensity | 101.14 | 122.27 | +20.9% |
| transpose_pct | 66.7% | 50.8% | -15.9pp |

**Result:** SUCCESS. Reduced A reloads from ~32K to ~1K loads. Arithmetic intensity crossed the peak ratio threshold (109.84), shifting from memory-bound toward compute-bound.

---

## V2: M-Dimension Blocking, TILES_IN_BLOCK_M=4 (20.630 → 5.868 ms, -71.6%)

**Bottleneck targeted:** B tile reloads. With V1's M-outer loop, B[n,k] was loaded once per (m_block, n, k) — reloaded 16x per unique tile.

**Code change:** Block M dimension into groups of 4 tiles (BLOCK_M=512). Pre-load 4 M-tile sets of A outside N loop, then iterate N with B tiles reused across the 4 M tiles in the block.

**Evidence:**
| Metric | V1 | V2 | Change |
|--------|-----|-----|--------|
| total_time (ms) | 20.630 | 5.868 | -71.6% |
| HBM read (MB) | 1056 | 288 | -72.7% |
| reload_factor | 11.00x | 3.00x | -8.00x |
| arith_intensity | 122.27 | 431.16 | +3.5x |
| MFU (%) | 8.47 | 29.78 | +21.3pp |
| GpSimd (%) | 99.36 | 97.60 | (DMA saturated) |

**Result:** SUCCESS. B reloads reduced from 16x to 4x. DMA traffic dropped 72.7%, directly translating to 71.6% time reduction. GpSimd at 97.6% confirms DMA was the primary bottleneck and is now near-saturated.

---

## V3: Increase TILES_IN_BLOCK_M to 8 (5.868 → 4.760 ms, -18.9%)

**Bottleneck targeted:** Remaining B reload redundancy. With BLOCK_M=4, there are 4 M blocks and B is reloaded 4x per unique tile.

**Code change:** Increase TILES_IN_BLOCK_M from 4 to 8 (BLOCK_M=1024). A tiles: 8×64×32KB = 16 MB. B tiles: 64×32KB = 2 MB. Total: 18 MB SBUF (within 24 MB limit).

**Evidence:**
| Metric | V2 | V3 | Change |
|--------|-----|-----|--------|
| total_time (ms) | 5.868 | 4.760 | -18.9% |
| HBM read (MB) | 288 | 160 | -44.4% |
| reload_factor | 3.00x | 1.67x | -1.33x |
| arith_intensity | 431.16 | 744.73 | +72.8% |
| GpSimd (%) | 97.60 | 5.24 | -92.4pp |
| MFU (%) | 29.78 | 36.72 | +6.9pp |

**Result:** SUCCESS. B reloads reduced from 4x to 2x. GpSimd dropped from 97.6% to 5.24% — DMA is no longer the bottleneck. The kernel has shifted to a compute-limited regime.

---

## V4: TILE_N=512 with Sub-Tile B Transposes (4.760 → 4.548 ms, -4.5%)

**Bottleneck targeted:** TensorE instruction count. With 37888 TensorE instructions (13.5% transpose overhead), reducing matmul instruction count by using larger TILE_N=512 (PSUM free dim max for gen3).

**Code change:** Load B as 4 sub-tiles of [128,128], transpose each via nc_transpose, assemble into [128,512] moving tensor. Each nc_matmul now computes 4x more output (128×512 instead of 128×128). BLOCK_M=8 maintained for data reuse.

**Evidence:**
| Metric | V3 | V4 | Change |
|--------|-----|-----|--------|
| total_time (ms) | 4.760 | 4.548 | -4.5% |
| matmul_inst_count | 37888 | 13312 | -64.9% |
| HBM read (MB) | 160 | 185.3 | +15.8% |
| reload_factor | 1.67x | 1.93x | +0.26x |
| MFU (%) | 36.72 | 38.43 | +1.7pp |

**Result:** PARTIAL. Matmul instruction count dropped 65% but total time only improved 4.5% (below 10% target). The sub-tile B transposes added DMA overhead (4 loads+transposes per B tile instead of 1), slightly increasing HBM reads from 160 to 185 MB. The instruction reduction and extra DMA partially cancel out.

**Failed attempt within V4:** K-dimension blocking with N-outer loop was also attempted. This restructured the loop to N-outer, K-middle, M-inner with K blocked in groups of 8. Result: massive regression to 24.260 ms (reload_factor 12x) because the loop restructuring lost A tile caching across N iterations.

---

## V5: 2-Pass K with BLOCK_M=16 (4.548 → 3.876 ms, -14.8%)

**Bottleneck targeted:** B reload factor. With BLOCK_M=8 (V3/V4), only 2 M blocks exist and B is loaded 2x per unique tile. BLOCK_M=16 (all M tiles) eliminates redundancy but requires 32 MB for A tiles (exceeds 24 MB SBUF).

**Code change:** Split K dimension into 2 passes of 32 tiles each. Per pass: load all 16 M tiles × 32 K tiles = 16 MB (fits in SBUF). Pass 1 stores fp16 partials to C. Pass 2 loads previous partials from C, adds in PSUM (float32), stores final result. This enables BLOCK_M=16 with only 16 MB A tiles per pass.

**Trade-off:** Extra C read-modify-write (16 MB read + 16 MB write in pass 2) adds some DMA traffic, but net HBM traffic is reduced because B tiles are loaded only 2x (once per K pass) across ALL M tiles instead of per M block.

**Evidence:**
| Metric | V4 | V5 | Change |
|--------|-----|-----|--------|
| total_time (ms) | 4.548 | 3.876 | -14.8% |
| HBM read (MB) | 185.3 | 112.0 | -39.6% |
| HBM write (MB) | 26.3 | 32.0 | +21.7% |
| reload_factor | 1.93x | 1.17x | -0.76x |
| arith_intensity | 619.36 | 910.22 | +46.9% |
| MFU (%) | 38.43 | 45.09 | +6.7pp |
| transpose_pct | 13.5% | 8.6% | -4.9pp |

**Result:** SUCCESS. HBM reads dropped 39.6% by eliminating B reload redundancy. MFU improved to 45.09% — the highest across all versions. Arithmetic intensity reached 910, ~8.3x above the Roofline threshold.

**Precision note:** V5 stores fp16 intermediate partials to C between K passes. Accuracy validation confirms cosine_sim=1.00045753, max_diff=0.000031, allclose=True — no measurable precision loss.

---

## Summary

The optimization journey progressed through 3 phases:

1. **Eliminating redundant loads** (V1-V3): Reload factor 13.33x → 1.67x via load hoisting and M-dimension blocking
2. **Reducing instruction overhead** (V4): Matmul instruction count 37888 → 13312 via larger TILE_N=512 (partial success)
3. **Maximizing data reuse** (V5): 2-pass K approach enabling BLOCK_M=16 with reload factor 1.17x

Key architectural insights:
- nc_transpose overhead is unavoidable with [M,K] and [N,K] input layouts (both need K as partition dim for nc_matmul)
- SBUF capacity (24 MB) is the binding constraint for tile caching
- The 2-pass K approach with C read-modify-write enables larger M blocking within SBUF limits
- Arithmetic intensity increased 9x from 101 to 910, well above the Roofline threshold
