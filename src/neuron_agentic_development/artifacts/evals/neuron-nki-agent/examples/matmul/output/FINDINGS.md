# NKI Matmul with B Transposed - Development Findings

## Task Summary
Implemented an NKI kernel for `C = A @ B.T` where:
- A: (M=2048, K=8192)
- B: (N=4096, K=8192)
- C: (M=2048, N=4096)

## Key Findings

### 1. nc_matmul Input Format
For `nisa.nc_matmul(dst, stationary, moving)`:
- **stationary**: Shape `[K_tile, M_tile]` where K is partition dim (<=128)
- **moving**: Shape `[K_tile, N_tile]` where K is partition dim (<=128), N can be up to 512
- **output**: Shape `[M_tile, N_tile]` in PSUM (float32 accumulation)

Since our inputs have K as the free dimension (A is M×K, B is N×K), we need to transpose them.

### 2. Transpose Approaches Evaluated

#### a. nisa.dma_transpose (Not Used)
- **Limitation**: Requires 4D source tensors on this version
- Error: "source tensor must have 4 dimensions"
- Would have been ideal for HBM→SBUF transpose during DMA

#### b. nisa.nc_transpose (Used)
- Works with 2D tensors
- Uses TensorEngine to swap partition and free dimensions
- Flow: SBUF [P, F] → PSUM [F, P] → copy to SBUF
- **Critical constraint for gen3+**: PSUM output dtype must match input dtype (not float32)

### 3. Gen3 (NeuronCore-v3) Specific Constraints

```python
# WRONG: float32 PSUM for transpose (works on gen2, fails on gen3+)
A_transposed_psum = nl.ndarray((K, M), dtype=nl.float32, buffer=nl.psum)

# CORRECT: Same dtype as input
A_transposed_psum = nl.ndarray((K, M), dtype=A.dtype, buffer=nl.psum)  # A.dtype is float16
```

The error code `NCC_IBIR031` indicates: "For CoreV3+, Matmult in transpose mode must have same input and output dtype"

### 4. Tile Size Selection

Due to nc_transpose partition dimension limit (P <= 128):
- TILE_K = 128 (partition for matmul operands)
- TILE_M = 128 (stationary free dimension)
- TILE_N = 128 (reduced from optimal 512 due to transpose constraint)

**Optimization opportunity**: Could use TILE_N=512 with strided DMA access instead of nc_transpose for B tiles.

### 5. Accumulation Pattern
Multiple `nc_matmul` writes to the same PSUM buffer with `affine_range` loop triggers hardware accumulation:

```python
result_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
for k_idx in nl.affine_range(num_k_tiles):  # NOT sequential_range
    # ... load and transpose tiles ...
    nisa.nc_matmul(dst=result_psum, stationary=A_tile, moving=B_tile)
```

### 6. Accuracy Results

With input scaling by `1/sqrt(K)` to prevent float16 overflow:
- Cosine Similarity: 1.00046 (target >= 0.999)
- Max Absolute Diff: 3.05e-05 (target <= 0.1)
- Mean Relative Diff: 5.45e-06
- `torch.allclose`: True

## Files Created

1. `matmul_bt_kernel.py` - NKI kernel implementation
2. `test_matmul_bt.py` - Test script with accuracy validation
3. `FINDINGS.md` - This document

## Potential Optimizations (Not Implemented)

1. **Increase TILE_N to 512**: Use strided DMA or array patterns (.ap()) to load B tiles already transposed
2. **Load hoisting**: Pre-load A tiles for the inner N loop (reuse across n_idx iterations)
3. **Block tiling**: Load larger blocks of K tiles to improve DMA efficiency
4. **Double buffering**: Overlap DMA with compute

## References

- `/skills/neuron-nki-writing/SKILL.md` - NKI kernel writing guide
- `/skills/neuron-nki-docs/references/architecture/trainium2_arch.md` - gen3 architecture details
- `/skills/neuron-nki-writing/references/transpose-and-layout.md` - Transpose techniques
- `/skills/neuron-nki-docs/references/downloads/matrix_multiplication_nki_kernels.py` - Reference matmul examples
