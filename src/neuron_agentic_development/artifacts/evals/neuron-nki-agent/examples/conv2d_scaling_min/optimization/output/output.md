# Conv2D + Scale + Min — Optimization Summary

## Progression Table

| Version | total_time (ms) | Δ vs prev | TensorE% | VectorE% | ScalarE% | GpSimdE% | HBM Read (KB) | MFU% | DMA Count | DMA Avg (B) | Status |
|---------|----------------|-----------|----------|----------|----------|----------|---------------|------|-----------|-------------|--------|
| baseline | 13.438 | — | 7.97 | 1.18 | 3.88 | 18.60 | 75,520 | 0.0043 | 4,968 | 15,578 | BASELINE |
| v1 | 3.074 | **-77.1%** | 10.75 | 5.14 | 17.00 | 81.16 | 4,352 | 0.0188 | 520 | 8,688 | IMPROVED |
| v2 | 2.879 | -6.3% | 11.51 | 5.50 | 12.14 | 86.63 | 4,352 | 0.0200 | 520 | 8,688 | IMPROVED |
| v3 | 2.910 | +1.1% | 11.55 | 5.44 | 12.02 | 84.18 | 5,120 | 0.0198 | 80 | 66,305 | REGRESSED |
| v4 | **2.661** | **-7.6%** | 10.74 | 3.63 | 7.76 | 90.74 | 4,624 | 0.0217 | 288 | 16,654 | **BEST** |
| v5 | 2.771 | +4.1% | 12.05 | 17.85 | 7.44 | 87.15 | 4,240 | 0.0208 | 265 | 16,616 | REGRESSED |

## Best Result

**v4: 2.661ms** (80.2% reduction from baseline 13.438ms)

## Test Configuration

- Input: [8, 64, 64×64] (batch=8, C_in=64, H=W=64)
- Weight: [9, 64, 128] (3×3 kernel, C_out=128)
- Scale factor: 2.0
- Hardware: trn2, gen3, NEURON_RT_VISIBLE_CORES=0-3
