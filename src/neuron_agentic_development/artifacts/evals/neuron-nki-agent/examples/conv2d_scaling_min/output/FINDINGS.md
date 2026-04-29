# Conv2d + Scale + Min NKI Kernel - Development Findings

## Summary

Successfully implemented an NKI kernel that performs:
1. **Conv2d**: 2D convolution using matmul
2. **Scale**: Multiply by scale factor
3. **Min**: Reduce along channel dimension

## Implementation Versions

### Version 1: With im2col preprocessing (Fast)
- **Input**: im2col_input [B, weight_k, spatial_out] (preprocessed on host)
- **Weight**: [C_out, weight_k] (flattened)
- **Pros**: Fast kernel execution, efficient DMA
- **Cons**: Requires host-side im2col transformation
- **Status**: Fully working for all tested sizes

### Version 2: Without im2col (Current)
- **Input**: x_input [B, C_in, H, W] (original layout)
- **Weight**: [C_out, C_in * k * k] (flattened on host - cheap)
- **Pros**: No large host-side preprocessing
- **Cons**: Element-by-element patch extraction is extremely slow for large inputs
- **Status**: Correct but impractical for large spatial dimensions

## Key Technical Details

### Algorithm
- **MatMul**: weight.T @ patches = [C_out, spatial_out] per batch
- **Min reduction**: Use `tensor_partition_reduce` with `max` on negated data (`min(x) = -max(-x)`)

### Key NKI APIs Used
- `nisa.dma_copy`: Load/store between HBM and SBUF
- `nisa.nc_transpose`: Weight transpose (tiled 32x32)
- `nisa.nc_matmul`: Matrix multiplication with PSUM accumulation
- `nisa.tensor_tensor`: Element-wise operations (add)
- `nisa.tensor_scalar`: Scalar broadcast operations (scale, negate)
- `nisa.tensor_partition_reduce`: Reduce across partition dimension

### Hardware Constraints

| Constraint | Limit | Impact |
|------------|-------|--------|
| P_MAX (partition dim) | 128 | Requires tiling for weight_k > 128 |
| PSUM_F_MAX (free dim) | 512 | Tiled for larger spatial dims |
| Transpose tile size | 32x32 | Requires tiled transpose |

### Challenges Solved

1. **"cannot reshape a complex access pattern"**: NKI can't reshape 4D HBM tensors with complex indexing. Solved by:
   - Flattening weights to 2D on host (cheap)
   - Using 2D view of input for DMA

2. **"TensorCopy has invalid memory location type: DRAM"**: `tensor_copy` only works for SBUF. Use `dma_copy` for HBM access.

3. **Transpose size limit**: `nc_transpose` has 32x32 max constraint. Tiled the transpose operation.

4. **Min reduction on partition dim**: `tensor_reduce` only works on free dimension. Used `tensor_partition_reduce` with `-max(-x)` trick.

5. **weight_k > P_MAX**: Tile over weight_k dimension, accumulating matmul results.

## Accuracy Results

All tests pass with excellent accuracy:

| Metric | Result |
|--------|--------|
| Cosine Similarity | 1.000000 |
| Mean Relative Diff | < 0.0002 |
| Max Absolute Diff | ~0.0005 (fp16 limit) |

## Test Results

| Test | Dimensions | Status |
|------|------------|--------|
| 1 | batch=1, in_ch=2, out_ch=4, HW=8 | PASS |
| 2 | batch=2, in_ch=4, out_ch=8, HW=16 | PASS |
| 3 | batch=4, in_ch=32, out_ch=32, HW=32 | PASS |
| 4 | batch=2, in_ch=64, out_ch=64, HW=32 | PASS |

## Performance Considerations

### Im2col on Host (Recommended for production)
- PyTorch's `unfold` is highly optimized
- Produces contiguous memory layout
- Enables efficient bulk DMA loads
- Kernel compilation is fast

### No Im2col (Current implementation)
- Avoids host-side preprocessing
- Element-by-element DMA is extremely slow
- For HW=64, in_ch=64, k=3: ~2.2 million DMA operations per spatial tile
- Kernel compilation/tracing time explodes for large inputs

### HW=256 Test Results (Full Scale)
Attempted to run with batch=8, in_ch=64, out_ch=128, HW=256, k=3:
- **spatial_out = 254 × 254 = 64,516**
- **Total DMA operations: ~37 million**
- **Tracing time: >65 minutes (did not complete)**
- **Memory usage: >193GB (and still growing)**
- **Conclusion: Impractical for large inputs**

The element-by-element patch extraction generates an IR with tens of millions of operations, causing:
1. Extreme memory consumption during tracing
2. Unacceptable compilation times
3. Likely large compiled kernel size

## Recommendation

For production use with large inputs (H, W > 64):
- Use the im2col approach (Version 1)
- The host-side preprocessing is much faster than element-by-element kernel DMA

For small inputs or when host preprocessing is not available:
- Current kernel (Version 2) works correctly
- Best suited for small spatial dimensions

## Files

- `conv2d_scale_min_kernel.py`: NKI kernel implementation (Version 2 - no im2col)
- `test_conv2d_scale_min.py`: Test harness with PyTorch reference
- `test_full_scale.py`: Full-scale dimension test
- `FINDINGS.md`: This document

## Running Tests

```bash
source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate
cd /home/ubuntu/silverhand/nki-dev-suite/tmp/conv2d_scaling_min
python test_conv2d_scale_min.py
```
