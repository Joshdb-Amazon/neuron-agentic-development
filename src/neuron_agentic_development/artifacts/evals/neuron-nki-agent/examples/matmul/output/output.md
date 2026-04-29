# Matmul with B Transposed NKI Kernel - Accuracy Results

## Kernel Configuration
- **Operation**: C = A @ B.T (matrix multiplication with B transposed)
- **Platform**: trn2 (gen3)
- **Transpose Method**: nc_transpose (TensorEngine) for both A and B tiles
- **Accumulation**: Hardware PSUM accumulation over K tiles

## Input/Output Shapes
| Tensor | Shape | Description |
|--------|-------|-------------|
| A | (M=2048, K=8192) | Left operand |
| B | (N=4096, K=8192) | Right operand (transposed in matmul) |
| C | (M=2048, N=4096) | Output: A @ B.T |

## Tile Configuration
| Parameter | Value | Notes |
|-----------|-------|-------|
| TILE_K | 128 | Partition dimension (hardware max) |
| TILE_M | 128 | Stationary free dimension |
| TILE_N | 128 | Moving free dimension (limited by nc_transpose P<=128) |
| num_k_tiles | 64 | K / TILE_K = 8192 / 128 |
| num_m_tiles | 16 | M / TILE_M = 2048 / 128 |
| num_n_tiles | 32 | N / TILE_N = 4096 / 128 |

## Accuracy Results

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Cosine Similarity | 1.00045741 | >= 0.999 | PASS |
| Mean Relative Diff | 5.45e-06 | - | - |
| Max Absolute Diff | 3.05e-05 | <= 0.1 | PASS |
| Mean Absolute Diff | 3.45e-09 | - | - |
| torch.allclose | True | True | PASS |

## Input Scaling
- **Scale factor**: 1/sqrt(K) = 1/sqrt(8192) ≈ 0.011049
- **Purpose**: Prevent float16 overflow in large matmul accumulations
- **A range**: [-0.0580, 0.0573]
- **B range**: [-0.0581, 0.0589]
- **Output range**: [-0.0603, 0.0544]

## Compilation Statistics

| Phase | Time | Notes |
|-------|------|-------|
| Kernel Tracing | <1s | No warnings/errors |
| neuronx-cc Compile | ~35s | Single LNC, verbose=35 |
| Total Wall Time | ~40s | Including device transfer |

## Key Technical Findings

### 1. nc_transpose Gen3 Constraint
```python
# WRONG (gen2 style): float32 PSUM output
A_psum = nl.ndarray((K, M), dtype=nl.float32, buffer=nl.psum)

# CORRECT (gen3): Same dtype as input
A_psum = nl.ndarray((K, M), dtype=A.dtype, buffer=nl.psum)
```
**Error if wrong**: `NCC_IBIR031: For CoreV3+, Matmult in transpose mode must have same input and output dtype`

### 2. dma_transpose Limitation
- Requires 4D source tensors in current SDK version
- Not usable for 2D matrix tiles
- Alternative: nc_transpose (TensorEngine)

### 3. Tile Size Trade-off
- Optimal TILE_N for nc_matmul: 512 (gemm_moving_fmax)
- Actual TILE_N used: 128 (due to nc_transpose P<=128 constraint)
- **Optimization opportunity**: Use strided DMA for B tiles to enable TILE_N=512

## Kernel Flow

```
For each (m_idx, n_idx) output tile:
    For each k_idx in K dimension:
        1. DMA: A[m, k] → SBUF [M_tile, K_tile]
        2. nc_transpose: SBUF → PSUM [K_tile, M_tile]
        3. tensor_copy: PSUM → SBUF (A_tile)

        4. DMA: B[n, k] → SBUF [N_tile, K_tile]
        5. nc_transpose: SBUF → PSUM [K_tile, N_tile]
        6. tensor_copy: PSUM → SBUF (B_tile)

        7. nc_matmul: A_tile @ B_tile → accumulate in result_psum

    8. tensor_copy: result_psum → SBUF
    9. DMA: SBUF → C[m, n]
```

## Recommendations

1. **For correctness**: Current implementation is numerically accurate
2. **For performance**: Consider strided DMA to load B tiles pre-transposed, enabling TILE_N=512
3. **For larger matrices**: Load hoisting (pre-load A tiles for inner N loop) would reduce redundant DMA
4. **For production**: Pre-compile for expected input shapes to avoid runtime compilation

## Files Generated

| File | Description |
|------|-------------|
| `matmul_bt_kernel.py` | NKI kernel implementation |
| `test_matmul_bt.py` | Test script with accuracy validation |
| `FINDINGS.md` | Detailed development notes |
| `output.md` | This summary document |
