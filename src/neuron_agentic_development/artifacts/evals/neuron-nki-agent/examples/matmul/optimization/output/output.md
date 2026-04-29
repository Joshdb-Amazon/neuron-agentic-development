# Matmul BT Optimization — Summary

| Metric | Baseline | V1 | V2 | V3 | V4 | V5 |
|--------|----------|-----|-----|-----|-----|-----|
| **total_time (ms)** | 26.379 | 20.630 | 5.868 | 4.760 | 4.548 | **3.876** |
| vs previous | — | -21.8% | -71.6% | -18.9% | -4.5% | -14.8% |
| vs baseline | — | -21.8% | -77.8% | -82.0% | -82.8% | **-85.3%** |
| TensorE (%) | 61.93 | 54.86 | 69.99 | 67.28 | 69.66 | 66.82 |
| VectorE (%) | 51.00 | 36.06 | 33.15 | 25.71 | 25.48 | 25.39 |
| DMA (%) | 63.55 | 66.83 | 66.25 | 47.21 | 54.30 | 44.75 |
| GpSimd (%) | 26.45 | 99.36 | 97.60 | 5.24 | 15.43 | 6.90 |
| HBM read (MB) | 1280 | 1056 | 288 | 160 | 185 | **112** |
| HBM write (MB) | 16 | 16 | 16 | 16 | 26 | 32 |
| MFU (%) | 6.62 | 8.47 | 29.78 | 36.72 | 38.43 | **45.09** |
| Arith intensity | 101 | 122 | 431 | 745 | 619 | **910** |
| Reload factor | 13.33x | 11.00x | 3.00x | 1.67x | 1.93x | **1.17x** |
| Transpose (%) | 66.7 | 50.8 | 22.0 | 13.5 | 13.5 | **8.6** |
| Matmul insts | 98304 | 66560 | 41984 | 37888 | 13312 | 35840 |
| SBUF spill | 0 | 0 | 0 | 0 | 0 | 0 |

**Key changes per version:**
- **V1:** Load hoisting — pre-load A tiles outside N loop
- **V2:** M-dimension blocking (BLOCK_M=4) — reduce B reloads from 16x to 4x
- **V3:** Increase BLOCK_M to 8 — reduce B reloads to 2x, DMA no longer bottleneck
- **V4:** TILE_N=512 with sub-tile transposes — 65% fewer matmul instructions (only 4.5% time gain)
- **V5:** 2-pass K with BLOCK_M=16 — all M tiles cached, B loaded once per K pass, near-optimal data reuse

**Versions meeting ≥10% improvement target:** V1 (-21.8%), V2 (-71.6%), V3 (-18.9%), V5 (-14.8%)
**V4 did not meet 10% target** (-4.5%): instruction reduction offset by sub-tile transpose DMA overhead
