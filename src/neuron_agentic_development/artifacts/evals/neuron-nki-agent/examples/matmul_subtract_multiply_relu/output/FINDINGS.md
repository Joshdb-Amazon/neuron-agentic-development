# Matmul + Subtract + Multiply + ReLU NKI Kernel - Findings

## Operation
```
relu((x @ W^T + bias - subtract_value) * multiply_value)
```
- x: (256, 4096) bfloat16
- W: (4096, 4096) bfloat16 (nn.Linear weight)
- bias: (4096,) float32
- subtract_value: 2.0
- multiply_value: 1.5

## Architecture
- **Hardware**: trn2.48xlarge (gen3 / NeuronCore-v3)
- **Venv**: `/opt/aws_neuronx_venv_pytorch_2_8/`
- **Target**: `--target trn2 --lnc 1`

## Kernel Design

### Tiling Strategy
| Dimension | Total | Tile Size | Num Tiles | Notes |
|-----------|-------|-----------|-----------|-------|
| M (batch) | 256 | 128 (P_MAX) | 2 | Partition dim of output |
| K (contraction) | 4096 | 128 (TILE_K) | 32 | Partition dim of matmul operands |
| N (output features) | 4096 | 512 (TILE_N) | 8 | PSUM free dim max for gen3 |

### Data Flow per Output Tile [128, 512]
1. **MatMul accumulation** (32 K-tiles):
   - Load `x_t[K, M]` tile (stationary, [128, 128]) via `dma_copy`
   - Load `weight_t[K, N]` tile (moving, [128, 512]) via `dma_copy`
   - `nc_matmul` accumulates into PSUM [128, 512] (float32)
2. **PSUM to SBUF copy**: `tensor_copy` -> result_sb [128, 512] (float32)
3. **Bias broadcast**: Load bias[1, 512] -> `stream_shuffle_broadcast` to [128, 512]
4. **Bias addition**: `tensor_tensor` add
5. **Subtract + multiply**: `tensor_scalar` with chained ops `(x + (-sub)) * mul`
6. **ReLU**: `activation` with `op=nl.relu`
7. **Store**: `dma_copy` result to HBM output

### NKI Operations Used
| Step | Operation | NKI API |
|------|-----------|---------|
| Load tiles | HBM -> SBUF | `nisa.dma_copy` |
| Matrix multiply | Accumulate K-dim | `nisa.nc_matmul` |
| PSUM to SBUF | Copy accumulator | `nisa.tensor_copy` |
| Bias broadcast | Partition 0 -> all | `nisa.nc_stream_shuffle` |
| Add bias | Element-wise | `nisa.tensor_tensor(op=nl.add)` |
| Subtract + Multiply | Chained scalar ops | `nisa.tensor_scalar(op0=nl.add, op1=nl.multiply)` |
| ReLU | Activation | `nisa.activation(op=nl.relu)` |
| Store result | SBUF -> HBM | `nisa.dma_copy` |

### Key Decisions
- **Pre-transposed inputs**: Both `x` and `weight` are transposed on the host before
  passing to the kernel, following the matmul tutorial pattern. This avoids `dma_transpose`
  which requires 4D tensors on the current runtime version.
- **Bias broadcast**: Uses `nc_stream_shuffle` with zero mask to replicate partition-0
  bias values across all 128 partitions.
- **Fused scalar ops**: `tensor_scalar` chains subtract + multiply in a single instruction
  (same latency as one operation).
- **Float32 epilogue**: All post-matmul operations (bias add, subtract, multiply, relu)
  are performed in float32 for numerical precision.

## Accuracy Results (bf16 matmul, float32 epilogue)
| Metric | Value |
|--------|-------|
| Max absolute diff | 0.999 |
| Mean absolute diff | 0.051 |
| Cosine similarity | 0.99999 |
| allclose(atol=1.0, rtol=0.1) | PASS |
| allclose(atol=0.5, rtol=0.05) | PASS |

The differences are expected for bfloat16 matrix multiplication over a 4096-dimensional
contraction. The cosine similarity of 0.99999 confirms the kernel is computing the
correct operation.

## Issues Encountered

### 1. `dma_transpose` requires 4D source
- **Error**: `source tensor must have 4 dimmensions`
- **Docs**: Suggest 2D is supported with permutation [1, 0], but the runtime requires 4D.
- **Fix**: Pre-transpose on host, pass `x_t = x.T.contiguous()` to the kernel.

### 2. `NEURON_PLATFORM_TARGET_OVERRIDE` deprecated
- **Warning**: Use `platform_target="trn2"` in `@nki.jit()` decorator instead.


## Files
- `matmul_subtract_multiply_relu_kernel.py` - NKI kernel implementation
- `test_matmul_subtract_multiply_relu.py` - Test script with device execution and validation
- `FINDINGS.md` - This file
- `output.md` - Accuracy results summary

## Running Tests
```bash
source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate
cd examples/matmul_subtract_multiply_relu/output
python test_matmul_subtract_multiply_relu.py
```
