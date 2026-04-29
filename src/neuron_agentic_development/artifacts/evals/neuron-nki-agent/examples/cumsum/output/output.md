# NKI Cumsum Kernel - Test Results

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Input Shape | (32768, 32768) |
| Cumsum Dimension | 1 (last dim) |
| Total Elements | 1,073,741,824 |
| Input Scaling | 1/seq_len (overflow prevention) |
| Target Platform | trn2 (gen3) |
| Compiler Flags | `--target trn2 --lnc 1` |

## Quick Test (128, 256)

| Metric | Value | Status |
|--------|-------|--------|
| torch.allclose | - | PASS |
| Max absolute diff | 3.576279e-07 | - |

## Full Test (32768, 32768)

| Metric | Value | Status |
|--------|-------|--------|
| torch.allclose (atol=1e-4, rtol=1e-4) | - | PASS |
| Maximum absolute difference | 5.185604e-06 | - |
| Mean absolute difference | 4.169021e-07 | - |
| Relative norm of difference | 3.186058e-06 | - |
| Cosine similarity | 1.0000000000 | - |

## Overall Result

**PASS** - Kernel produces correct results within floating-point precision tolerance.
