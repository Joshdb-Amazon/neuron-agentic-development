# SwiGLU MLP Kernel Optimization Summary

## Test Configuration
- Shape: tokens=4096, input_size=4096, hidden_size=8192
- dtype: float16
- Hardware: trn2 (gen3), cores 20-23

## Progression Table

| Version | total_time (ms) | TensorE (%) | VectorE (%) | DMA (%) | MFU (%) | HBM Read (MB) | HBM Write (MB) | Reload Factor | Matmul Instrs | Status |
|---------|-----------------|-------------|-------------|---------|---------|----------------|-----------------|---------------|---------------|--------|
| baseline | 36.704 | 80.48 | 25.92 | 93.25 | 28.57 | 7143 | 101 | 30.41x | 81920 | BASELINE |
| v1 | 33.558 | 82.92 | 15.83 | 96.24 | 31.25 | 6908 | 101 | 29.41x | 66560 | IMPROVED (8.6%) |
| v2 | 30.357 | 82.01 | 39.33 | 99.27 | 34.54 | 6476 | 34 | 27.57x | 52224 | IMPROVED (17.3%) |
| v3 | 32.050 | 73.63 | 69.16 | 42.86 | 32.72 | 1275 | 571 | 5.43x | 81920 | REGRESSED |
| v4 | 20.434 | 77.15 | 57.08 | 79.06 | 51.32 | 3255 | 34 | 13.86x | 52224 | IMPROVED (44.3%) |
| **v5** | **18.711** | **78.74** | **62.41** | **44.43** | **56.04** | **1644** | **34** | **7.00x** | **52224** | **BEST (49.0%)** |

All percentages are relative to baseline total_time.

## Key Metrics Evolution

- **total_time**: 36.704 ms → 18.711 ms (**49.0% reduction**)
- **HBM reads**: 7143 MB → 1644 MB (**77.0% reduction**)
- **MFU**: 28.57% → 56.04% (**1.96x improvement**)
- **DMA utilization**: 93.25% → 44.43% (shifted from DMA-bound to balanced)
- **Arithmetic intensity**: 113.84 → 491.52 (**4.3x increase**)
