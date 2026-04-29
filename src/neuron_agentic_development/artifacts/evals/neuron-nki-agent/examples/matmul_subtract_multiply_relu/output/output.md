# Matmul + Subtract + Multiply + ReLU NKI Kernel - Accuracy Results

## Kernel Configuration
- **Operations**: Linear (matmul + bias) -> Subtract -> Multiply -> ReLU
- **Platform**: trn2 (gen3)
- **Input dtype**: bfloat16
- **Epilogue dtype**: float32 (PSUM accumulation + all post-matmul ops)

## Accuracy Results

| Batch | In_Features | Out_Features | subtract | multiply | Cosine Sim | Mean Abs Diff | Max Abs Diff | Status |
|-------|-------------|--------------|----------|----------|------------|---------------|--------------|--------|
| 256 | 4096 | 4096 | 2.0 | 1.5 | 0.99999 | 0.051 | 0.999 | PASS |

## Tolerance Checks

| Tolerance Level | atol | rtol | Status |
|-----------------|------|------|--------|
| Standard | 1.0 | 0.1 | PASS |
| Tight | 0.5 | 0.05 | PASS |

## Key Observations

1. **Cosine similarity of 0.99999** confirms the kernel computes the correct mathematical operation
2. **Max abs diff ~1.0** is expected for bfloat16 matmul with 4096-dimensional contraction
   - bfloat16 has ~7 bits of mantissa (vs 23 for float32)
   - Accumulating 4096 products introduces rounding errors proportional to sqrt(4096) * ulp
3. **Mean abs diff of 0.051** indicates errors are small and uniformly distributed
4. **Both tolerance levels pass**, confirming production-quality accuracy

## Compilation Details

| Metric | Value |
|--------|-------|
| Compiler status | PASS |
| Tracing time | ~2s |
| Compilation time | ~3s |
| Total tiles per output | 2 (M) x 8 (N) x 32 (K) = 512 matmul tiles |
