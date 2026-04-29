# RMS Fused QKV NKI Kernel - Development Findings

## Summary

Successfully implemented an NKI kernel that performs:
1. **RMSNorm**: `x / sqrt(mean(x^2, dim=-1) + eps)`
2. **Linear Projection**: Matrix multiply with weight matrix

The kernel fuses both operations on a single NeuronCore, computing
`(input @ weight) * inv_rms` instead of `(input * inv_rms) @ weight`
to avoid materializing a large normalized intermediate in SBUF.

## Implementation Details

### Algorithm
- **Phase 1**: Compute `inv_rms = rsqrt(mean(x^2) + eps)` per row using fused `nisa.activation(op=rsqrt, scale=1/dim, bias=eps)`
- **Phase 2**: Tiled matrix multiply `input @ weight` via `nc_matmul` with PSUM accumulation, then scale by `inv_rms`

### Tiling Strategy

| Dimension | Size | Tile Size | Tiles | Hardware Limit |
|-----------|------|-----------|-------|----------------|
| batch*seqlen (P) | 256 | 128 | 2 | P_MAX=128 |
| dim (K contraction) | 2048 | 128 | 16 | nc_matmul K<=128 per tile |
| head_dim (N output) | 1024 | 512 | 2 | PSUM_FMAX=512 (gen3) |

### Key NKI APIs Used
- `nisa.dma_copy`: Load/store between HBM and SBUF
- `nisa.nc_transpose`: Transpose input tiles (tiled 32x32) for matmul stationary operand
- `nisa.nc_matmul`: Matrix multiplication with PSUM accumulation
- `nisa.tensor_tensor`: Element-wise multiply (with `.ap()` broadcast)
- `nisa.tensor_reduce`: Sum reduction along free dimension
- `nisa.activation`: Fused scale + bias + rsqrt in single instruction
- `nisa.memset`: Initialize epsilon buffer

### Hardware Constraints

| Constraint | Limit | Impact |
|------------|-------|--------|
| P_MAX (partition dim) | 128 | Tile seqlen dimension |
| PSUM_F_MAX (free dim) | 512 | Tile head_dim dimension |
| Transpose tile size | 32x32 | Tile nc_transpose operations |
| MatMul K per tile | 128 | 16 accumulation steps for dim=2048 |

## Challenges Solved

1. **nc_transpose limited to 32x32 on Scalar Engine**: The documented 128x128 limit applies to the Tensor Engine, but on gen3 `nc_transpose` routes through the Scalar Engine with a 32x32 max tile. Solved by tiling into a 4x4 grid of 32x32 sub-tiles.

2. **dma_transpose requires 4D source**: `nisa.dma_transpose` does not accept 2D tensor slices. Solved by using nc_transpose with tiling instead.

3. **tensor_tensor does NOT auto-broadcast**: Attempting `tensor_tensor` with shapes `[128, 512]` and `[128, 1]` produces a BIR verification error ("Expect AP same number of elements"). Solved by using `.ap(pattern=[[1, P], [0, F]])` with stride=0 to create a broadcast view.

4. **Matmul layout mismatch**: RMSNorm reduces along the hidden dimension (free dim), but nc_matmul needs the contraction dimension as the partition dim. Solved by transposing input tiles via nc_transpose before feeding to nc_matmul.

5. **Platform detection**: Environment did not auto-detect the Neuron platform. Set `NEURON_PLATFORM_TARGET_OVERRIDE=gen3`.

## Accuracy Results

| Metric | Result |
|--------|--------|
| Cosine Similarity | 1.00000310 |
| Max Absolute Diff | 1.618862e-04 |
| Mean Absolute Diff | 7.282228e-06 |
| Relative Norm Diff | 1.567028e-05 |
| Allclose (atol=1e-4, rtol=1e-3) | PASS |

## Files

- `rms_fused_qkv_kernel.py`: NKI kernel implementation
- `test_rms_fused_qkv.py`: Test harness with PyTorch CPU reference
- `output.md`: Accuracy results table
- `FINDINGS.md`: This document

## Running Tests

```bash
source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate
cd examples/rms_fused_qkv/output
NEURON_PLATFORM_TARGET_OVERRIDE=gen3 python test_rms_fused_qkv.py
```
