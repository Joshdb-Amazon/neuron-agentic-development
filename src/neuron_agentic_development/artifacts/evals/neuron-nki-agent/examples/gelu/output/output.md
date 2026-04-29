# GELU NKI Kernel - Accuracy Results

## Kernel Configuration
- **Operation**: Element-wise GELU activation
- **Platform**: trn2 (gen3)
- **Input**: 2D tensor [rows, cols], float32
- **Compute**: `nisa.activation(op=nl.gelu)` on Scalar Engine

## Accuracy Results by Input Shape

| Test | Rows | Cols | Cosine Sim | Mean Rel Diff | Max Abs Diff | Mean Abs Diff | Status |
|------|------|------|------------|---------------|--------------|---------------|--------|
| 1 | 2048 | 8192 | 1.000000 | 0.000001 | 1.91e-06 | 1.94e-08 | PASS |
| 2 | 128 | 512 | 1.000000 | 0.000000 | 7.15e-07 | 1.96e-08 | PASS |
| 3 | 256 | 4096 | 1.000000 | 0.000001 | 1.91e-06 | 1.93e-08 | PASS |
| 4 | 4096 | 8192 | 1.000000 | 0.000001 | 1.91e-06 | 1.94e-08 | PASS |

## Compilation

| Input Shape | Compile Time | Status |
|-------------|--------------|--------|
| 2048 x 8192 | ~1s | PASS |
| 128 x 512 | ~1s | PASS |
| 256 x 4096 | ~1s | PASS |
| 4096 x 8192 | ~1s | PASS |

## Key Observations

1. **Excellent accuracy**: Max absolute difference is ~1.91e-06 across all shapes, well within float32 tolerance
2. **Fast compilation**: All shapes compile in ~1 second due to the simple tiling structure (no unrolling of complex patterns)
3. **Scalable**: Kernel handles any (rows, cols) where rows is a multiple of 128

## Running Tests

```bash
source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate
cd examples/gelu/output
python test_gelu.py
```
