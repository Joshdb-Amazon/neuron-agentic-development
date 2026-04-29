# Softmax NKI Kernel - Accuracy Results

## Kernel Configuration
- **Operation**: Softmax along last dimension (dim=-1)
- **Platform**: trn2 (gen3)
- **Input**: 2D tensor [rows, cols] float32
- **Algorithm**: Numerically stable: exp(x - max(x)) / sum(exp(x - max(x)))

## Accuracy Results by Input Shape

| Test | Rows | Cols | Cosine Sim | Max Abs Diff | Norm Diff | Row Sums=1 | Status |
|------|------|------|------------|--------------|-----------|------------|--------|
| 1 | 1024 | 8192 | 1.000000 | 5.96e-08 | 1.75e-06 | Yes | PASS |

## Softmax Property Checks

| Property | Result |
|----------|--------|
| All values >= 0 | True |
| All values <= 1 | True |
| Row sums close to 1.0 | True |
| torch.allclose(rtol=1e-5, atol=1e-5) | True |

## Key Design Decisions

1. **Numerical stability**: Subtract row-wise max before exp to prevent overflow
2. **Tiling**: Partition dim tiled at 128 (P_MAX); free dim (8192) fits in SBUF
3. **All intermediates in SBUF**: No unnecessary HBM round-trips
4. **Broadcast via tensor_scalar**: Reduction results [P,1] broadcast across [P,C] efficiently

## Compilation

| Metric | Value |
|--------|-------|
| Compiler status | PASS |
| Tracing time | < 1s |
| Compilation time | ~ 1s |
