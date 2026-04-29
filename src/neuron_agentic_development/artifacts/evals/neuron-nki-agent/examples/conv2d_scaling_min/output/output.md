# Conv2d + Scale + Min NKI Kernel - Accuracy Results

## Kernel Configuration
- **Operations**: Conv2d -> Scale -> Min(dim=1)
- **Platform**: trn2 (gen3)
- **Input**: Original tensor [B, C_in, H, W] (no im2col preprocessing)
- **Weight**: Pre-flattened [C_out, C_in * k * k] on host

## Accuracy Results by Input Shape

| Test | Batch | In_Ch | Out_Ch | H | W | k | spatial_out | weight_k | Cosine Sim | Mean Rel Diff | Max Abs Diff | Status |
|------|-------|-------|--------|---|---|---|-------------|----------|------------|---------------|--------------|--------|
| 1 | 1 | 2 | 4 | 8 | 8 | 3 | 36 | 18 | 1.000000 | 0.008516 | 0.000488 | PASS |
| 2 | 2 | 4 | 8 | 16 | 16 | 3 | 196 | 36 | 1.000000 | 0.000136 | 0.000488 | PASS |
| 3 | 4 | 32 | 32 | 32 | 32 | 3 | 900 | 288 | 1.000000 | 0.000113 | 0.000244 | PASS |
| 4 | 2 | 64 | 64 | 32 | 32 | 3 | 900 | 576 | 1.000000 | 0.000112 | 0.000122 | PASS |
| 5 | 8 | 64 | 128 | 64 | 64 | 3 | 3844 | 576 | 0.999998 | 0.000102 | 0.000244 | PASS |

## Compilation Time Comparison

### Vectorized Kernel (Current)
| Input Shape | spatial_out | Tracing Time | Compile Time | Memory Usage |
|-------------|-------------|--------------|--------------|--------------|
| HW=8 | 36 | <1s | ~1s | minimal |
| HW=16 | 196 | <1s | ~1s | minimal |
| HW=32 | 900 | ~1s | ~1s | minimal |
| HW=64 | 3844 | ~50s | ~120s | ~6GB |
| HW=256 | 64516 | >30min (stopped) | N/A | ~8GB (stable) |

### Element-by-Element Kernel (Previous)
| Input Shape | spatial_out | Tracing Time | Memory Usage | Status |
|-------------|-------------|--------------|--------------|--------|
| HW=32 | 900 | ~6 min | ~few GB | Completed |
| HW=256 | 64516 | >65 min | >193GB (growing) | Failed/Stopped |

## Key Improvements

1. **Memory Usage**: Reduced from 193GB+ to ~8GB for HW=256
2. **Tracing Stability**: Memory now stable instead of growing unbounded
3. **Vectorization**: Row-based DMA loads instead of element-by-element

## Remaining Bottlenecks

1. **Bias broadcast loop**: Still has per-column tensor_copy operations
2. **Tracing time**: Scales with h_out * k_tile_size loops
3. **Compilation time**: Still significant for large spatial dimensions

## Recommendations

1. **For HW <= 64**: Current kernel works with acceptable compile time (~2 min)
2. **For HW > 64**: Use im2col preprocessing on host for faster compilation
3. **For production**: Pre-compile kernels for expected input shapes
