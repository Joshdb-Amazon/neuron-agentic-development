# Softmax Kernel Optimization — Summary

## Progression Table

| Version | total_time (μs) | Δ from prev | Δ from baseline | VectorE (%) | ScalarE (%) | DMA (%) | Vector Instr | Scalar Instr | HBM R/W (MB) | Status |
|---------|----------------|-------------|-----------------|-------------|-------------|---------|-------------|-------------|---------------|--------|
| Baseline | 348.78 | — | — | 51.31 | 33.94 | 47.74 | 73 | 45 | 32/32 | BASELINE |
| v1 (fuse sub+exp) | 285.00 | -18.3% | -18.3% | 49.76 | 42.38 | 58.34 | 58 | 59 | 32/32 | IMPROVED |
| v2 (buffer reuse) | 275.25 | -3.4% | -21.1% | 51.24 | 43.64 | 60.24 | 51 | 52 | 32/32 | MARGINAL |
| **v3 (fuse exp+sum)** | **182.17** | **-33.8%** | **-47.8%** | 40.02 | 66.62 | 90.59 | 43 | 60 | 32/32 | **BEST** |
| v4 (double buffer) | 233.03 | +27.9% | -33.2% | 31.48 | 52.41 | 73.73 | 44 | 60 | 32/32 | FAILED |
| v5 (P=64 tiles) | 739.12 | +306% | +112% | 19.51 | 32.79 | 46.11 | 59 | 92 | 32/32 | FAILED |

## Key Optimizations That Worked

1. **v1: Fuse subtract+exp** (-18.3%): `nisa.activation(op=nl.exp, bias=-max)` combines subtraction and exponentiation into a single ScalarE instruction pass, eliminating one full [P, C] tile data pass.

2. **v3: Fuse exp+sum_reduce** (-33.8%): `nisa.activation(..., reduce_op=nl.add, reduce_cmd=reset_reduce)` computes exp AND sum in a single instruction, eliminating the separate `tensor_reduce(sum)` call entirely.

## Best Result

**Baseline → v3: 348.78 μs → 182.17 μs = 47.8% total_time reduction**

The optimized kernel is DMA-bound (90.59% DMA utilization) with 64MB unavoidable HBM traffic. Further optimization is constrained by memory bandwidth for this inherently low-arithmetic-intensity operation.
