# NeuronX Compiler Debugging

When all component tests pass and CPU E2E passes but device E2E still diverges, the issue may be in the NeuronX compiler (neuronxcc). This reference covers the compiler debugging tools and workflow.

## When to Escalate Here

Escalate to compiler debugging when:
- All patches verified correct on CPU
- Code paths confirmed identical between CPU and device modes
- Weight loading confirmed correct (pre/post shard checksums match)
- Device intermediate tensors show divergence unexplainable by code differences

## Compilation Pipeline

```
XLA/HLO → [hlo2penguin] → Penguin IR → [Tensorizer] → BIR → [Walrus] → .neff
```

| Stage | Component | Description |
|-------|-----------|-------------|
| Frontend | `hlo2penguin` | Converts XLA HLO to Penguin IR |
| Middle-end | Tensorizer | Lowers Penguin IR to BIR with optimizations |
| Backend | Walrus | Generates final Trainium executables (.neff) |

## Debugging Artifacts

When compilation produces numerical issues, check the compiler artifacts directory.

| Artifact | Description |
|----------|-------------|
| `log-neuron-cc.txt` | Complete error messages and stack traces |
| `penguin.py` | Original Penguin IR input (Python format, for simulation) |
| `penguin-sg*/` | Git repositories with optimization pass history |

### Pass History

The `penguin-sg*` directories are git repos where each commit = one optimization pass:

```bash
cd {artifacts}/penguin-sg0
git log --oneline          # View pass history
git checkout <hash>        # Examine IR after a specific pass
cat penguin.py             # Inspect the IR
```

## Debugging Workflow

### Level 1: XLA/HLO Simulation

Use HLO Bugpoint to isolate which HLO operation introduces the numerical error:

```bash
hlo_bugpoint.py \
  --search-strategy bisect \
  --hlo-modify-mode cut \
  --input model.hlo \
  --input-data ./data/ \
  --tolerance 5 1e-5
```

Search strategies: `bisect` (fast), `linear` (thorough), `topo` (topological), `oneshot`.

### Level 2: Penguin IR Simulation

Verify transformations at the Penguin IR level:

```bash
simulator.py penguin.py --rtol=0.01 --atol=1e-5
simulator.py penguin.py --save-all --save-interm-tensors
simulator.py penguin.py --track-rw
```

### Level 3: NKI Kernel Simulation

For NKI kernel issues, simulate on CPU:

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import numpy as np

@nki.jit
def my_kernel(a_tensor):
    a = nl.load(a_tensor)
    nl.device_print("value of a:", a)
    b = nl.empty_like(a_tensor, buffer=nl.hbm)
    nl.store(b, value=a)
    return b

result = nki.simulate_kernel(my_kernel, np.random.randn(128, 256).astype(np.float32))
```

### Level 4: BIR Simulation

Low-level hardware simulation:

```bash
neuronxcc ... --pass bir_sim --mem-mode=physical --sync-mode=ON \
  --enable-check-outputs --check-inst-output-NaN
```

### Test Case Reduction

```bash
bugpoint.py failing_penguin.py --passes Pass1 Pass2 SimulatorPass --save-weights
```

## Tool Paths

### Source (KaenaCompiler)

| Tool | Path |
|------|------|
| HLO Bugpoint | `KaenaCompiler/neuronxcc/starfish/util/hlo_bugpoint/hlo_bugpoint.py` |
| Penguin Simulator | `KaenaCompiler/neuronxcc/starfish/penguin/tools/simulator.py` |
| Penguin Bugpoint | `KaenaCompiler/neuronxcc/starfish/penguin/tools/bugpoint.py` |
| Penguin opt.py | `KaenaCompiler/neuronxcc/starfish/penguin/tools/opt.py` |
| NKI compile | `KaenaCompiler/neuronxcc/nki/compile.py` |
| KLR Simulator | `KaenaCompiler/neuronxcc/starfish/penguin/tools/klr_simulator.py` |
| NEFF Analyzer | `KaenaCompiler/neuronxcc/starfish/bin/analyze_neff_artifacts.py` |

### Docker Container (when wheel installed)

Base: `/opt/conda/lib/python3.11/site-packages/neuronxcc/`

| Tool | Container Path |
|------|---------------|
| HLO Bugpoint | `{base}/starfish/util/hlo_bugpoint/hlo_bugpoint.py` |
| Penguin Simulator | `{base}/starfish/penguin/tools/simulator.py` |
| Penguin Bugpoint | `{base}/starfish/penguin/tools/bugpoint.py` |
| NKI compile | `{base}/nki/compile.py` |
| Kernel Sim CLI | `{base}/kernel_sim/cli.py` |

## Relationship to Equivalence Stages

| Situation | Action |
|-----------|--------|
| Stage 2 component fails | NOT a compiler issue — debug in Stage 4 |
| Stage 5 E2E fails, Stage 2 all pass | Possible compiler issue — but first check if device class differs from CPU class (run `detect_class_divergence.py`) |
| Stage 5 fails, class divergence confirmed | NOT a compiler issue — the device uses a different algorithm. Reimplement and test in Stage 2 |
| Stage 5 fails, no class divergence, CPU E2E passes | Likely compiler issue — escalate here |

---

Based on: Claude Code neuronxcc-debugging skill
