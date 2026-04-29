---
name: neuron-framework-equivalence
description: Verifies functional equivalence between two implementations of the same model using a hierarchical 8-stage algorithm. Orchestrates model tree comparison, component-level R-ratio testing, E2E logit comparison, and distributional checks. Use when porting models between frameworks, hardware targets, or precision regimes — e.g., HuggingFace to NxDI, PyTorch to ONNX, GPU to Neuron, FP32 to BF16, or any source-target pair.
---

# Model Equivalence Framework

Verify and diagnose functional equivalence between a **source** (reference) and a **target** (ported) implementation of the same model.

## Required Inputs

Before starting, collect these from the user. Ask for any missing ones.

| Input | Description | Example |
|-------|-------------|---------|
| `SOURCE_MODEL_PATH` | Path to source model weights (HF format) | `/path/to/hf_models/Qwen3-0.6B` |
| `COMPILED_MODEL_PATH` | Path to compiled target model | `/path/to/neuron_models/Qwen3-0.6B` |
| `TARGET_MODELING_FILE` | Path to target's modeling .py file | `/path/to/modeling_qwen3.py` |
| `TARGET_INNER_CLASS` | Inner model class (extends NeuronBaseModel) | `NeuronQwen3Model` |
| `TARGET_CAUSAL_CLASS` | ForCausalLM wrapper class | `NeuronQwen3ForCausalLM` |
| `TARGET_CONFIG_CLASS` | InferenceConfig class | `Qwen3InferenceConfig` |
| `VENV` | Path to Python venv with torch + neuronx | `/opt/aws_neuronx_venv_pytorch_2_8_nxd_inference` |
| `MODEL_VALIDATION_DIR` | Path to model_validation package | `/path/to/NeuroborosFoundations/model_validation` |
| `EXP_DIR` | Experiment output directory | `agent_artifacts/equiv_qwen3` |

Set `SCRIPTS_DIR` to the absolute path of this skill's `scripts/` directory.

## CRITICAL RULES

1. **Do NOT write your own scripts.** Run the bundled scripts in `scripts/` directly.
2. **The ONLY files you create are:** `component_mapping.json` (Step 1) and `test_NN_*.py` test files (Step 3).
3. **Do NOT write wrapper scripts, helper scripts, or any .py files** other than the test files.
4. **Follow stage order strictly.** Do NOT skip ahead or reorder.
5. **Show full output** from every script run. Do not summarize or truncate.

## Workflow

Execute in this exact order. Each step has a single command to run.

### Step 1: Stage 0 — Build Model Trees

```bash
source {VENV}/bin/activate
PYTHONPATH={SCRIPTS_DIR} python3 {SCRIPTS_DIR}/run_stage0.py \
  --source-model-path {SOURCE_MODEL_PATH} \
  --target-model-path {SOURCE_MODEL_PATH} \
  --target-module-file {TARGET_MODELING_FILE} \
  --target-inner-class {TARGET_INNER_CLASS} \
  --target-config-class {TARGET_CONFIG_CLASS} \
  --output-dir {EXP_DIR}/model_tree
```

Then compare the two printed trees and build `{EXP_DIR}/component_mapping.json`.
See [STAGE0.md](STAGE0.md) for mapping instructions and [references/mapping_example.json](references/mapping_example.json) for format.

Then run class divergence detection:

```bash
python3 {SCRIPTS_DIR}/detect_class_divergence.py \
  --target-module-file {TARGET_MODELING_FILE} \
  --output {EXP_DIR}/class_divergence_report.json
```

This identifies components that use different classes on CPU vs device (e.g., `LlamaRMSNorm` on CPU, `CustomRMSNorm` on device). These require dual tests in Stage 2.

### Step 2: Stage 1 — Smoke Test

```bash
PYTHONPATH={MODEL_VALIDATION_DIR} python3 {SCRIPTS_DIR}/run_stage1.py \
  --model-path {SOURCE_MODEL_PATH} \
  --compiled-model-path {COMPILED_MODEL_PATH} \
  --model-class {TARGET_MODELING_FILE}:{TARGET_CAUSAL_CLASS} \
  --config-class {TARGET_MODELING_FILE}:{TARGET_CONFIG_CLASS} \
  --num-tokens 32 \
  --output {EXP_DIR}/results/stage1.json
```

**Decision:** If token match < 50% → catastrophic failure, proceed to Step 3 for localization. Otherwise continue.

### Step 3: Stage 2 — Component Tests (YOU WRITE THESE)

1. Copy `{SCRIPTS_DIR}/tensor_compare.py` into `{EXP_DIR}/tests/`
2. Create `{EXP_DIR}/tests/conftest.py` from [templates/conftest_template.py](templates/conftest_template.py) — fill in model constants from `config.json`
3. Write `test_NN_*.py` files for each mapped component. See [STAGE2.md](STAGE2.md) for the pattern and [templates/test_template.py](templates/test_template.py) for the template.
4. Run:

```bash
NXD_CPU_MODE=1 python3 {SCRIPTS_DIR}/run_stage2.py \
  --tests-dir {EXP_DIR}/tests \
  --tau-r 1.2 \
  --output {EXP_DIR}/results/stage2.json
```

**Decision:** If all R < 1.2 → proceed to Step 4. If any fail → run Stage 3:

```bash
python3 {SCRIPTS_DIR}/run_stage3.py \
  --stage2-output {EXP_DIR}/results/stage2.json \
  --output {EXP_DIR}/results/stage3.json
```

Then debug and patch (see [STAGE4.md](STAGE4.md)). Re-run Stage 2 until all pass.

### Step 4: Stages 5+6 — Teacher-Forced E2E + Distributional + Semantic

This single script covers Stage 5 (E2E R-ratio), Stage 6 Condition B (cosine), and Stage 6 Condition C (KL) — all under proper teacher forcing.

```bash
PYTHONPATH={MODEL_VALIDATION_DIR}:{SCRIPTS_DIR} python3 {SCRIPTS_DIR}/run_teacher_forced_comparison.py \
  --model-path {SOURCE_MODEL_PATH} \
  --compiled-model-path {COMPILED_MODEL_PATH} \
  --model-class {TARGET_MODELING_FILE}:{TARGET_CAUSAL_CLASS} \
  --config-class {TARGET_MODELING_FILE}:{TARGET_CONFIG_CLASS} \
  --num-tokens 32 \
  --output {EXP_DIR}/results/teacher_forced.json
```

### Step 5: Stage 7 — Downstream Eval (optional)

```bash
python3 {SCRIPTS_DIR}/run_stage7.py \
  --bench-config {EXP_DIR}/bench_config.yaml \
  --output-dir {EXP_DIR}/results/stage7
```

See [STAGE7.md](STAGE7.md) for bench config format.

## R-Ratio

```
R = ||target - source_fp32||_F / (||source_lowprec - source_fp32||_F + ε)
```

| R | Meaning |
|---|---------|
| ≈ 1.0 | Healthy — matches precision baseline |
| > 1.2 | Bug — excess divergence |
| < 1.0 | Over-precision — extra `.float()` calls |

## Verdict

```
PASS ⟺ Stage 1 acceptable ∧ Stage 2 all R < 1.2
      ∧ Stages 5+6 E2E R consistent ∧ Condition B(θ) ∧ Condition C(δ)
      ∧ Stage 7 no regressions
```

## Resources

- [STAGE0.md](STAGE0.md) — Tree building + component mapping
- [STAGE2.md](STAGE2.md) — How to write component tests
- [STAGE4.md](STAGE4.md) — Debug/patch workflow
- [STAGE5.md](STAGE5.md) — E2E comparison details
- [STAGE6.md](STAGE6.md) — Condition B/C details
- [templates/](templates/) — conftest and test templates
- [references/](references/) — structural diffs, mapping example, equiv-concept, QQ plots
- [references/debugging-case-study-gptoss.md](references/debugging-case-study-gptoss.md) — Complete worked example from GPT-OSS 20B
- [references/device-component-debugging.md](references/device-component-debugging.md) — XLA-compatible patch patterns for device debugging
- [references/device-e2e-debugging.md](references/device-e2e-debugging.md) — 1-layer isolation and device E2E fix-compile-verify cycle
- [references/cpu-e2e-debugging.md](references/cpu-e2e-debugging.md) — CPU E2E with mp.spawn, TP>1, bias restoration
- [references/dump-tensors.md](references/dump-tensors.md) — Intermediate tensor capture methodology
- [references/neuronxcc-debugging.md](references/neuronxcc-debugging.md) — NeuronX compiler debugging tools and escalation workflow
