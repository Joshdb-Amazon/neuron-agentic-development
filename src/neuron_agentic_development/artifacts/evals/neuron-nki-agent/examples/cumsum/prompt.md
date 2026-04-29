Write a NKI kernel that runs on a single neuron core for the nki-dev-suite/cumsum.py, make sure to run it on device and compare with torch CPU baseline to ensure correctness. Use the neuron-nki-writing skill. Also, use the neuron-nki-docs skill to look up NKI docs for API usage, architecture constraints, etc.
- When testing the kernel, reference the neuron-nki-debugging skill and use the venv at `/opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/`  default_nc_version is gen3.
Please create a ./tmp/cumsum folder to save all the any intermediate test file or script generated.
Also document necessary finding during the process.

## Implementation Hints

### Algorithm
Use `nisa.tensor_tensor_scan` for cumsum. The operation `result[i] = result[i-1] + x[i]` maps to:
```python
nisa.tensor_tensor_scan(
    dst=result, data0=ones, data1=input, initial=0,
    op0=nl.multiply, op1=nl.add
)
# Computes: result[i] = op1(op0(data0[i], result[i-1]), data1[i]) = 1 * result[i-1] + x[i]
```

### Tiling
- Partition dim: tile to 128 (P_MAX) using `TiledRange`
- Free dim: tile with `nl.sequential_range` for carry dependency between tiles
- Carry state: copy last element of each tile to `initial` for next tile

### Critical Details
1. `TiledRangeIterator` class must inherit from `nki.language.NKIObject`
2. Scale test inputs by `1/seq_len` to prevent float16 overflow in large accumulations
3. Use `--target trn2 --lnc 1` compiler flags for gen3
