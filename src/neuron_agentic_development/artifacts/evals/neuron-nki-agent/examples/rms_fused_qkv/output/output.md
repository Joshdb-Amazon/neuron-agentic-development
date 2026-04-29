# RMS Fused QKV NKI Kernel - Accuracy Results

## Kernel Configuration
- **Operations**: RMSNorm -> Linear Projection (QKV)
- **Platform**: trn2 (gen3)
- **Input**: [batch*seqlen, dim] float32
- **Weight**: [dim, head_dim] float32

## Accuracy Results

| Test | Batch | SeqLen | Dim | Head_Dim | Cosine Sim | Mean Abs Diff | Max Abs Diff | Rel Norm Diff | Status |
|------|-------|--------|-----|----------|------------|---------------|--------------|---------------|--------|
| 1 | 1 | 256 | 2048 | 1024 | 1.00000310 | 7.282228e-06 | 1.618862e-04 | 1.567028e-05 | PASS |

## Tolerance Checks

| Tolerance | atol | rtol | Result |
|-----------|------|------|--------|
| Strict | 1e-4 | 1e-3 | PASS |
| Loose | 1e-3 | 1e-2 | PASS |

## Output Range Comparison

| Metric | CPU Reference | NKI Kernel |
|--------|---------------|------------|
| Min | -4.0512 | -4.0512 |
| Max | 4.2939 | 4.2938 |

## Key Observations

1. **Excellent accuracy**: Cosine similarity ~1.0, max abs diff < 2e-4
2. **Float32 precision**: Small differences from different accumulation order in tiled matmul
3. **Compilation**: Single-pass compilation with neuronx-cc, ~2s compile time

## Running Tests

```bash
source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate
cd examples/rms_fused_qkv/output
NEURON_PLATFORM_TARGET_OVERRIDE=gen3 python test_rms_fused_qkv.py
```
