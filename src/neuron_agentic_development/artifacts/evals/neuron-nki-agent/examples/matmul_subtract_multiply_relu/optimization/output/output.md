# Matmul + Subtract + Multiply + ReLU — Optimization Summary

## Progression Table

| Version | Change | total_time (us) | Delta vs prev | TensorE (%) | DMA (%) | HBM Read (MB) | MFU (%) | Spill |
|---------|--------|----------------|---------------|-------------|---------|---------------|---------|-------|
| Baseline | Original kernel | 649.9 | — | 48.9 | 90.7 | 83.9 | 16.81 | 0 |
| V1 | x_t load hoisting | 392.0 | **-39.7%** | 68.0 | 93.3 | 69.2 | 27.86 | 0 |
| V2 | N-outer + weight hoisting | 306.5 | **-21.8%** | 63.6 | 86.9 | 39.9 | 35.64 | 0 |
| V3 | Fused activation (sub+mul+relu) | 305.6 | -0.3% | 63.5 | 87.1 | 39.9 | 35.74 | 0 |
| V4 | Full x_t hoisting (outside all loops) | 226.4 | **-25.9%** | 68.4 | 90.5 | 35.7 | 48.25 | 0 |
| V5 | Fused PSUM+bias copy | 225.9 | -0.2% | 68.9 | 90.8 | 35.7 | 48.34 | 0 |

## Key Results

- **Total improvement: 65.2%** (649.9 us -> 225.9 us)
- **3 versions achieved >10% reduction**: V1 (-39.7%), V2 (-21.8%), V4 (-25.9%)
- **HBM reads reduced from 2.35x to 1.0x theoretical minimum** (83.9 MB -> 35.7 MB)
- **MFU improved 2.9x** (16.81% -> 48.34%)
- **Zero spill throughout all versions**

## Bottleneck Analysis

The kernel remains DMA-bound (90.8%) at V5. With HBM reads at theoretical minimum (35.7 MB),
further gains require either DMA bandwidth optimization (K-dimension blocking with interleaved
load/compute) or algorithmic changes (FP8 precision on gen3+).
