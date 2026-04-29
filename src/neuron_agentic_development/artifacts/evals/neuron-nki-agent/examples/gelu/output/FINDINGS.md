# GELU NKI Kernel - Development Findings

## Summary

Successfully implemented an NKI kernel that performs element-wise GELU activation
using the Scalar Engine's built-in piecewise-polynomial GELU approximation (`nl.gelu`).

## Implementation Details

### Algorithm
- GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
- Hardware Scalar Engine approximates this with piecewise polynomials via `nisa.activation(op=nl.gelu)`
- Valid input range: [-inf, inf] (no clamping required)
- Internal math performed in float32 precision

### Tiling Strategy
- Partition dimension (P): 128 (`nl.tile_size.pmax`)
- Free dimension (F): full column width per tile (e.g., 8192)
- Number of row tiles: rows / 128 (e.g., 16 for 2048 rows)
- Loop type: `nl.affine_range` (no loop-carried dependencies, fully parallel)

### Key NKI APIs Used
- `nisa.dma_copy`: Load/store between HBM and SBUF
- `nisa.activation(op=nl.gelu)`: Apply GELU activation on Scalar Engine
- `nl.ndarray(..., buffer=nl.shared_hbm)`: Allocate output in HBM (return-based pattern)
- `nl.affine_range`: Parallel loop over row tiles

### Hardware Constraints

| Constraint | Limit | Used |
|------------|-------|------|
| P_MAX (partition dim) | 128 | 128 |
| SBUF free dim max | 32767 | 8192 |

## Challenges Solved

### 1. Mutable output parameter pattern produced all zeros
- **Problem**: Initially used `def gelu_kernel(input_hbm, output_hbm)` with a pre-allocated
  output tensor passed as a parameter. The kernel compiled and ran but the output was all zeros
  (cosine similarity = 0.0, max diff = 5.2).
- **Root Cause**: The mutable output parameter pattern requires `neuronxcc.nki.typing` annotations
  for proper in-place mutation. Without proper annotation, writes to the output parameter are not
  reflected in the caller's tensor.
- **Resolution**: Switched to the return-based pattern where the kernel allocates output via
  `nl.ndarray(..., buffer=nl.shared_hbm)` and returns it. This is the recommended Quick Start
  pattern and works without special annotations.

### 2. Platform target identification
- **Problem**: Setting only `NEURON_CC_FLAGS="--target trn2"` without `NEURON_PLATFORM_TARGET_OVERRIDE`
  caused a `ValueError: Could not identify a target platform`.
- **Root Cause**: The 2.9 venv deprecated `NEURON_PLATFORM_TARGET_OVERRIDE` env var but still needs
  platform identification.
- **Resolution**: Use `@nki.jit(platform_target="trn2")` in the decorator. This is the recommended
  approach and avoids deprecation warnings.

### 3. Neuron core allocation (LNC=2 on trn2)
- **Problem**: `NEURON_RT_VISIBLE_CORES=0` failed with "Logical Neuron Core(s) not available -
  Requested:lnc0-lnc0 Available:0 Logical Core size:2".
- **Root Cause**: trn2 uses Logical NeuronCore (LNC) size of 2, meaning each logical core
  consists of 2 physical cores. Pinning to a single physical core is insufficient.
- **Resolution**: Set `NEURON_RT_VISIBLE_CORES=0-1` to allocate a pair of physical cores for
  one logical NeuronCore.


## Accuracy Results

All tests pass with excellent accuracy:

| Metric | Result |
|--------|--------|
| Cosine Similarity | ~1.000000 |
| Mean Relative Diff | ~0.000001 |
| Max Absolute Diff | ~1.91e-06 |

## Test Results

| Test | Dimensions | Status |
|------|------------|--------|
| 1 | 2048 x 8192 | PASS |
| 2 | 128 x 512 | PASS |
| 3 | 256 x 4096 | PASS |
| 4 | 4096 x 8192 | PASS |

## Files

- `gelu_nki_kernels.py`: NKI kernel implementation
- `test_gelu.py`: Test harness with PyTorch reference
- `output.md`: Accuracy results summary
- `FINDINGS.md`: This document

## Running Tests

```bash
source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate
cd examples/gelu/output
python test_gelu.py
```
