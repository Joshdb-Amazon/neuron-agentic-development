# GELU Optimization — Summary

## Progression Table

| Version | Strategy | total_time (µs) | Δ vs baseline | DMA active (%) | ScalarE (%) | MBU (%) | DMA xfers | DMA avg (KB) | Status |
|---------|----------|-----------------|---------------|----------------|-------------|---------|-----------|--------------|--------|
| Baseline | affine_range, dma_copy, activation | 332.17 | — | 96.90 | 35.53 | 56.43 | 32 | 4096 | BASELINE |
| V1 | 2D tiling (128×2048) | 332.94 | +0.2% | 97.98 | 39.56 | 56.30 | 128 | 1024 | FAILED |
| V2 | Double buffering, sequential_range | 330.97 | -0.4% | 97.21 | 35.60 | 56.64 | 32 | 4096 | MARGINAL |
| V3 | In-place activation | 330.13 | -0.6% | 97.11 | 35.65 | 56.78 | 32 | 4096 | MARGINAL |
| V4 | HW DGE + in-place | 329.37 | -0.8% | 96.94 | 35.74 | 56.91 | 32 | 4096 | MARGINAL |
| **V5** | **Split blocks + HW DGE + in-place** | **328.85** | **-1.0%** | **96.83** | **35.79** | **57.00** | **32** | **4096** | **BEST** |

## Failed Approaches (documented for completeness)

| Attempt | Strategy | total_time (µs) | Δ | Why it failed |
|---------|----------|-----------------|---|---------------|
| V4-alt | Phased load/compute/store, 128×1024 tiles | 361.18 | +8.7% | 256 small DMA transfers (512KB) — overhead dominated |
| V5-alt | Column-stripe traversal, 128×512 tiles | 407.82 | +22.8% | 511 tiny DMA transfers (256KB) — MBU dropped to 46% |

## Key Finding

GELU is a standalone elementwise kernel that is **fundamentally DMA-bandwidth
limited** at 57% MBU. With zero arithmetic intensity and no adjacent operations
to fuse, the kernel cannot be significantly optimized beyond the current memory
bandwidth ceiling. The 1% improvement from V5 combines hardware DGE, in-place
activation, and split affine_range blocks — extracting every available micro-
optimization from the DMA/ScalarE scheduling.
