# Softmax NKI Kernel - Development Findings

## Summary

Successfully implemented a numerically stable NKI softmax kernel that computes
`softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))` along the last dimension
of a 2D tensor.

## Implementation

### Algorithm
1. **Find row max** - `nisa.tensor_reduce(op=nl.maximum, axis=1)` for numerical stability
2. **Subtract max** - `nisa.tensor_scalar(op0=nl.subtract)` broadcasts [P,1] across [P,C]
3. **Exp** - `nisa.activation(op=nl.exp)`
4. **Sum exp** - `nisa.tensor_reduce(op=nl.add, axis=1)`
5. **Reciprocal** - `nisa.reciprocal()` computes 1/sum
6. **Multiply** - `nisa.tensor_scalar(op0=nl.multiply)` broadcasts reciprocal

### Key NKI APIs Used
- `nisa.dma_copy`: Load/store between HBM and SBUF
- `nisa.tensor_reduce`: Row-wise max and sum reductions
- `nisa.tensor_scalar`: Broadcast subtract and multiply operations
- `nisa.activation`: Element-wise exp function
- `nisa.reciprocal`: Element-wise reciprocal

### Hardware Constraints

| Constraint | Limit | Impact |
|------------|-------|--------|
| P_MAX (partition dim) | 128 | Rows tiled into chunks of 128 |
| SBUF free dim | 32767 | Cols=8192 fits without tiling |

### Tiling Strategy
- **Partition dimension (rows=1024):** 8 tiles of 128 rows each
- **Free dimension (cols=8192):** No tiling needed (fits in SBUF)
- All intermediates kept in SBUF (no unnecessary HBM round-trips)

## Accuracy Results

| Metric | Value |
|--------|-------|
| Max Absolute Difference | 5.96e-08 |
| Norm of Difference | 1.75e-06 |
| Cosine Similarity | ~1.0 |
| torch.allclose(rtol=1e-5, atol=1e-5) | True |
| Row sums | All 1.0000 |
| All values >= 0 | True |
| All values <= 1 | True |

## Challenges Solved

### 1. `tensor_reduce` axis parameter format
- **Error:** `not a valid dim` when using `axis=(1,)` (tuple format)
- **Fix:** Use `axis=1` (plain integer). All production NKI code uses integer format.

### 2. Deprecated `NEURON_PLATFORM_TARGET_OVERRIDE`
- **Warning:** Environment variable is deprecated
- **Fix:** Pass `platform_target="trn2"` directly to the `@nki.jit()` decorator.

## Files

- `softmax_nki_kernel.py`: NKI kernel implementation
- `test_softmax.py`: Test harness with PyTorch CPU reference comparison
- `FINDINGS.md`: This document
- `output.md`: Accuracy results summary

## Running Tests

```bash
source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate
cd examples/softmax/output
python test_softmax.py
```
