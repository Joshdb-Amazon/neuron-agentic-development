# Compiler Artifacts Mode

Advanced debugging mode that preserves intermediate compiler outputs for inspection. Use when you need to debug internal compiler passes or inspect the generated IR.

## When to Use

- "compiler artifacts" - Need to see what the compiler generated
- "debug internal compiler passes" - Investigating compiler transformations
- "inspect IR" - Looking at intermediate representation
- "SaveTemps" - Preserve temporary files
- Kernel compiles differently than expected
- Performance issues requiring compiler-level analysis

## Full Debug Flags

```python
import os

os.environ["NEURON_CC_FLAGS"] = (
    "--target trn2 "
    "--lnc 1 "
    "--verbose=info "
    "--pipeline compile SaveTemps "
    "--internal-compiler-debug-mode=all"
)
```

| Flag | Purpose |
|------|---------|
| `--target trn2` | Target platform |
| `--lnc 1` | Single NeuronCore |
| `--verbose=info` | Progress messages |
| `--pipeline compile SaveTemps` | Preserve intermediate files |
| `--internal-compiler-debug-mode=all` | Full debug output |

## Finding the Compiler Temp Folder

After compilation, the last ~50 lines of output contain the temp directory path. Look for:

```
/tmp/<username>/neuroncc_compile_workdir/<uuid>/
```

To find recent compilation directories:

```bash
ls -lt /tmp/$USER/neuroncc_compile_workdir/ | head -5
```

The most recent directory (by modification time) contains your compilation artifacts.

## Generated Artifacts

| File | Description |
|------|-------------|
| `penguin.py` | Traced kernel intermediate representation |
| `*.neff` | Compiled Neuron Executable File Format (binary) |
| `*.bir` | Backend IR (low-level representation) |
| `log-neuron-cc.txt` | Detailed compiler log |
| `*.hlo` | HLO (High Level Operations) graphs |

### penguin.py

The traced Python representation of your kernel after NKI transformation. Useful for understanding how NKI rewrites your kernel code.

### *.neff

The compiled binary executed on Neuron hardware. Use `neuron-profile` tools to analyze.

### *.bir

Backend Intermediate Representation. Low-level view of operations mapped to hardware.

### log-neuron-cc.txt

Complete compiler log including:
- Compilation phases and timing
- Warnings and diagnostics
- Memory allocation decisions
- Optimization passes applied

## BIRSIM Validation (Optional)

For CPU-based numerical validation during compilation, add BIRSIM flags:

```python
os.environ["NEURON_CC_FLAGS"] = (
    "--target trn2 "
    "--lnc 1 "
    "--verbose=info "
    "--pipeline compile SaveTemps "
    "--internal-compiler-debug-mode=all "
    "--internal-backend-options='"
    "--enable-birsim=True "
    "--enable-birsim-at-begin=False "
    "--enable-birsim-after-all=False "
    "--enable-birsim-at-end=True "
    "--birsim-output-tolerance 0.001,1e-5'"
)
```

BIRSIM validates kernel correctness on CPU before hardware execution:

| BIRSIM Flag | Purpose |
|-------------|---------|
| `--enable-birsim=True` | Enable CPU simulation |
| `--enable-birsim-at-end=True` | Validate at compilation end |
| `--birsim-output-tolerance rtol,atol` | Set tolerance thresholds |

## IR Dump Options

To dump IR after specific compiler passes:

```python
os.environ["NEURON_CC_FLAGS"] = (
    "--target trn2 "
    "--lnc 1 "
    "--verbose=info "
    "--pipeline compile SaveTemps "
    "--internal-backend-options='"
    "--print-format=condensed "
    "--print-after=translate_nki_ast_to_bir,lower_klir_kernel'"
)
```

| Pass | Description |
|------|-------------|
| `translate_nki_ast_to_bir` | NKI AST to BIR translation |
| `lower_klir_kernel` | KLIR kernel lowering |

## Complete Debug Script

```python
import os
import torch
from torch_xla.core import xla_model as xm
import nki
import nki.language as nl
import nki.isa as nisa

# Full debug configuration
os.environ["NEURON_CC_FLAGS"] = (
    "--target trn2 "
    "--lnc 1 "
    "--verbose=info "
    "--pipeline compile SaveTemps "
    "--internal-compiler-debug-mode=all"
)

# Optional: Capture runtime profiles
os.environ['NEURON_RT_INSPECT_ENABLE'] = '1'
os.environ['NEURON_RT_INSPECT_DEVICE_PROFILE'] = '1'
os.environ['NEURON_RT_INSPECT_OUTPUT_DIR'] = './output'
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

@nki.jit
def my_kernel(input_tensor):
    # Your kernel implementation
    tile = nl.ndarray(input_tensor.shape, dtype=input_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=input_tensor)
    result = nl.ndarray(input_tensor.shape, dtype=input_tensor.dtype, buffer=nl.sbuf)
    nisa.activation(dst=result, data=tile, op=nl.exp)
    output = nl.ndarray(input_tensor.shape, dtype=input_tensor.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=output, src=result)
    return output

device = xm.xla_device()
x = torch.randn((64, 128), dtype=torch.float32).to(device=device)

y = my_kernel(x)
print(y)  # Triggers compilation with full debug output

# After execution, find artifacts:
# ls -lt /tmp/$USER/neuroncc_compile_workdir/ | head -5
```

## Artifact Analysis Workflow

1. Run kernel with full debug flags
2. Find the temp directory from output or `ls -lt` command
3. Examine `log-neuron-cc.txt` for errors and warnings
4. Check `penguin.py` to understand kernel transformation done by the compiler frontend
5. Inspect `*.bir` files for low-level operation mapping done by the compiler backend
6. Use `neuron-profile` on `*.neff` for performance analysis

## Notes

- Debug flags significantly increase compilation time
- Use standard flags (`--target --lnc`) for regular development
- Only enable full artifacts mode when investigating compiler issues
- Temp directories are not automatically cleaned; manage disk space
- BIRSIM validation adds CPU simulation overhead
