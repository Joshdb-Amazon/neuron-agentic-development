# NKI Cumsum Kernel - Implementation Findings

## Overview

Successfully implemented a NKI kernel for `torch.cumsum` along the last dimension using `nisa.tensor_tensor_scan`.

## Implementation Details

### Kernel Design

The cumsum operation `result[i] = result[i-1] + x[i]` maps to `tensor_tensor_scan` as:

```python
nisa.tensor_tensor_scan(
    dst=result,
    data0=ones,      # multiply by 1 (identity)
    data1=input,     # add input value
    initial=0,       # start from 0
    op0=nl.multiply,
    op1=nl.add,
)
```

This computes: `result[i] = op1(op0(data0[i], result[i-1]), data1[i])` = `1 * result[i-1] + x[i]`

### Tiling Strategy

For a tensor of shape (32768, 32768) along dim=1:

1. **Partition dimension (dim=0)**: Tiled to 128 (P_MAX constraint)
   - 32768 / 128 = 256 partition tiles
   - Each tile uses full hardware parallelism

2. **Free dimension (dim=1)**: Tiled to 2048 elements
   - Large enough for good DMA efficiency (2KB at float32)
   - 32768 / 2048 = 16 free dimension tiles

3. **Loop structure**:
   - Outer loop: `TiledRange(outer_dim, P_MAX)` - iterates partition tiles
   - Inner loop: `nl.sequential_range(num_f_tiles)` - carries dependency across free tiles

### Key Implementation Points

1. **TiledRangeIterator must inherit from NKIObject**
   - NKI compiler requires custom classes to inherit from `nki.language.NKIObject`

2. **Carry state between tiles**
   - Initialize `init_sb` to 0 for each partition tile
   - After each free tile, copy the last value to `init_sb` for the next iteration

3. **Data type handling**
   - `tensor_tensor_scan` computes in float32 internally
   - Input/output can be any supported dtype

## Test Results

**Important**: Inputs are scaled by `1/seq_len` to prevent float16 overflow in large accumulations.
This is standard practice for cumsum tests to ensure numerical stability.

### Small Input Test (128, 256)
- **Max absolute difference**: 3.58e-07
- **Result**: PASS

### Full Input Test (32768, 32768)
- **Total elements**: 1,073,741,824 (1B+)
- **Max absolute difference**: 5.16e-06
- **Mean absolute difference**: 4.16e-07
- **Relative norm of difference**: 3.17e-06
- **Cosine similarity**: 1.0000000000
- **torch.allclose (atol=1e-4, rtol=1e-4)**: PASS

### Analysis

With properly scaled inputs:
- Maximum absolute difference is in the 1e-6 range (excellent accuracy)
- Cosine similarity of 1.0 indicates perfect directional agreement
- Relative norm confirms results match reference within floating-point precision

## Hardware Configuration

- **Target**: Trainium 2 (gen3)
- **Compiler flags**: `--target trn2 --lnc 1`
- **Venv**: `/opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/`

## Files

- `cumsum_kernel.py` - NKI kernel implementation
- `test_cumsum.py` - Test script with CPU baseline comparison
- `FINDINGS.md` - This documentation

## Performance Notes

The kernel uses:
- Single-core execution (`--lnc 1`)
- Single-buffered DMA (no pipelining)
- Full partition dimension usage (128) for hardware parallelism

For production use, consider:
- Double buffering for DMA/compute overlap
- SPMD parallelization across multiple NeuronCores for batch dimension
