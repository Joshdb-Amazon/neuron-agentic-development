# NKI SwiGLU MLP Kernel - Test Results

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Operation | `output = down_proj(silu(gate_proj(x)) * up_proj(x))` |
| Target Platform | trn2 (gen3) |
| Compiler Flags | `--target trn2 --lnc 1` |
| Token Tile Size | 128 (F_STAT_MAX) |
| K Tile Size | 128 (P_MAX) |
| Hidden/Output Tile Size | 512 (F_MOV_MAX) |

## Small Tests (Unit Test Suite)

| Test (tokens, input, hidden) | Cosine Similarity | Mean Rel. Diff | Status |
|------------------------------|-------------------|----------------|--------|
| (64, 128, 256) | 0.999999 | 0.0043% | PASS |
| (64, 256, 512) | 0.999998 | 0.0032% | PASS |
| (128, 128, 512) | 0.999999 | 0.0028% | PASS |
| (128, 256, 512) | 0.999998 | 0.0035% | PASS |
| (64, 512, 1024) | 0.999998 | 0.0094% | PASS |

## Large Test (4096 tokens)

| Parameter | Value |
|-----------|-------|
| Tokens | 4096 |
| Input Size | 4096 |
| Hidden Size | 8192 |
| Token Tiles | 32 (4096 / 128) |
| Input Scaling | 1/sqrt(input_size) (overflow prevention) |

| Metric | Value | Status |
|--------|-------|--------|
| Cosine similarity | 1.001356 | PASS |
| Mean relative diff | 0.0098% | PASS |
| Max absolute diff | 0.000000 | - |

## Sample Output Comparison

```
NKI sample: [ 2.3329e-04,  8.6665e-05, -1.5616e-05, -9.8348e-06, -1.1098e-04]
Ref sample: [ 2.3317e-04,  8.6665e-05, -1.5616e-05, -9.8348e-06, -1.1098e-04]
```

## Overall Result

**PASS** - All 6 test cases passed with cosine similarity > 0.9999 and mean relative diff < 1%.
