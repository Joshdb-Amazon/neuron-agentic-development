# Cumsum Kernel Optimization — Summary

## Progression Table

| Version | Optimization | total_time (ms) | vs Baseline | VE% | DMA% | HBM Read (GB) | HBM Write (GB) | DMA Transfers | Status |
|---------|-------------|-----------------|-------------|-----|------|---------------|----------------|---------------|--------|
| Baseline | None | 21.521 | — | 85.5% | 98.1% | 4.29 | 4.29 | 8192 | BASELINE |
| V1 | Double buffering | **20.507** | **-4.7%** | 89.6% | 100% | 4.29 | 4.29 | 8192 | **IMPROVED** |
| V2 | Deferred store overlap | 20.598 | -4.3% | 89.2% | 100% | 4.29 | 4.29 | 8192 | MARGINAL |
| V3 | F=4096 + double buffer | 22.298 | +3.6% | 80.4% | 96.9% | 4.29 | 4.29 | 4096 | REGRESSED |
| V4 | F=1024 + double buffer | 20.778 | -3.5% | 92.6% | 100% | 4.29 | 4.29 | 16384 | MARGINAL |
| V5 | affine_range + double buffer | 20.507 | -4.7% | 89.6% | 100% | 4.29 | 4.29 | 8192 | IMPROVED |

## Best Result

**V1 / V5**: 20.507 ms (4.7% improvement from 21.521 ms baseline)

## Key Finding

The cumsum kernel is **DMA-bandwidth-limited**. The computation reads 4GB and writes 4GB — the absolute minimum for this operation. At the best configuration (V1), the effective HBM bandwidth is ~390 GB/s, near the per-NeuronCore hardware limit on trn2. The 4.7% gain came from double buffering to eliminate DMA/compute sync stalls. Further improvement requires either reduced data volume (impossible for cumsum) or higher hardware bandwidth.

## MFU / Spill

- MFU: 0% (no matrix multiply operations — cumsum uses tensor_tensor_scan on Vector Engine)
- SBUF spill: Not observed in any version
