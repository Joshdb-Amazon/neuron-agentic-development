# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Orchestration layer: model creation, HF golden loading, the three
``test_*`` entry points, result persistence, and reporting.

This is the module that ``run.py`` (the CLI) imports.  Everything here
is a top-level function — no classes.  The pattern for each ``test_*``
function is:

    1. ``create_model`` → load compiled weights
    2. Run the actual check (accuracy / performance / smoke)
    3. ``save_*_results`` in a ``finally`` block (so results are persisted
       even on crash)
    4. ``del model`` to free Neuron device memory
    5. Return ``(passed: bool, details: dict)``

Key design decisions
--------------------
* The HF golden model is always loaded in **FP32** (not BF16).  Comparing
  BF16-vs-BF16 masks real precision issues — the whole point of validation
  is to measure how much the Neuron BF16 model drifts from the FP32 truth.

* ``create_model`` reads ``neuron_config.json`` from the compiled model
  directory to reconstruct the exact ``NeuronConfig`` / ``MoENeuronConfig``
  that was used at compile time.  This avoids mismatches between the
  validation config and what the model was actually compiled with.

* Dynamic class importing (``import_class``) supports two formats:
  ``path/to/file.py:ClassName`` (standalone ported models) and
  ``package.module.ClassName`` (models integrated into NxDI).

* Results are written as timestamped JSON files under
  ``model_validation/results/{smoke_tests,accuracy,benchmarks}/{model_name}/``.
  Filenames encode batch_size, seq_len, pass/fail, and timestamp so you
  can track regressions over time.
"""

import importlib
import importlib.util
import json
import os
import sys
import traceback
import torch
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)

from neuronx_distributed_inference.models.config import NeuronConfig, MoENeuronConfig
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

from .patches import (
    ensure_generation_config_version,
    ensure_pad_token,
    patch_load_hf_model,
    patch_generation_mixin,
)
from .accuracy import (
    _load_nxdi_accuracy_utils,
    check_accuracy_with_hf_golden,
    run_logit_matching_v2,
    run_inference_only_validation,
    format_prompt_with_chat_template,
)
from .enhanced_metrics import compute_enhanced_metrics
from .benchmark import run_custom_benchmark, print_benchmark_report
from .constants import PERFORMANCE_TOLERANCE, THROUGHPUT_TOLERANCE


# ---------------------------------------------------------------------------
# Neuron config loading
# ---------------------------------------------------------------------------

def load_neuron_config_from_compiled(compiled_path: str) -> Dict[str, Any]:
    """Read ``neuron_config.json`` from a compiled model directory.

    The file is expected at ``compiled_path/neuron_config.json``.  If it's
    not there (some compilation flows nest it in a subdirectory), we do a
    recursive glob and use the first match.

    The JSON may have the config nested under a ``"neuron_config"`` key or
    at the top level — we handle both.

    Raises:
        FileNotFoundError: If no ``neuron_config.json`` exists anywhere
            under ``compiled_path``.
    """
    config_path = Path(compiled_path) / "neuron_config.json"
    if not config_path.exists():
        candidates = list(Path(compiled_path).rglob("neuron_config.json"))
        if candidates:
            config_path = candidates[0]
            print(f"⚠ neuron_config.json not at root, using: {config_path}")
        else:
            raise FileNotFoundError(
                f"neuron_config.json not found under {compiled_path}. "
                "Ensure compilation completed successfully."
            )

    with open(config_path) as f:
        config_data = json.load(f)

    neuron_config = config_data.get("neuron_config", config_data)
    return neuron_config


# ---------------------------------------------------------------------------
# Class importing
# ---------------------------------------------------------------------------

def import_class_from_file(file_path: str, class_name: str):
    """Dynamically import a single class from an arbitrary ``.py`` file.

    Adds the file's parent directory to ``sys.path`` (if not already there)
    and uses ``importlib.util.spec_from_file_location`` to load the module.
    The module is registered in ``sys.modules`` under a synthetic name to
    avoid collisions.

    Args:
        file_path:  Absolute or CWD-relative path to the ``.py`` file.
        class_name: Name of the class to pull from the loaded module.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ImportError:       If the module can't be loaded.
        AttributeError:    If the class doesn't exist in the module.
    """
    file_path_obj = Path(file_path)
    if not file_path_obj.is_absolute():
        file_path_obj = Path.cwd() / file_path

    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path_obj}")

    parent_dir = str(file_path_obj.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    module_name = f"_module_{file_path_obj.stem}_{class_name}"
    spec = importlib.util.spec_from_file_location(module_name, str(file_path_obj))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path_obj}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def import_class(class_path: str):
    """Import a class from either a file path or a dotted module path.

    Two formats are supported (detected by the presence of ``.py:``):
      * ``path/to/file.py:ClassName`` — standalone ported model files
      * ``package.module.ClassName``  — classes installed in a package

    Returns the class object.
    """
    if ".py:" in class_path:
        file_path, class_name = class_path.split(":")
        return import_class_from_file(file_path, class_name)
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

def create_model(config: dict, batch_size: int, seq_len: int):
    """Build a Neuron model, tokenizer, and generation config from a
    validation config dict.

    Steps:
      1. Dynamically import ``ModelClass`` and ``ConfigClass`` from the
         paths in ``config["model_class"]`` / ``config["config_class"]``.
      2. Read ``neuron_config.json`` from the compiled model directory to
         get the exact compilation parameters (tp_degree, dtype, etc.).
      3. Auto-detect MoE models (presence of ``moe_tp_degree`` or
         ``router_config``) and use ``MoENeuronConfig`` accordingly.
      4. Load ``GenerationConfig`` from the HF model path (greedy,
         ``do_sample=False``).
      5. Build the model config via ``ConfigClass.from_pretrained()``
         (with a two-arg constructor fallback for older NxDI).
      6. Set up the tokenizer with right-padding and a pad-token fallback
         chain: eos → bos → ``[PAD]``.
      7. Instantiate the model (``from_pretrained`` or direct constructor).

    Note: this does NOT call ``model.load()`` — the caller is responsible
    for loading the compiled weights.

    Args:
        config:     Validation config dict (from ``configs/*.json``).
        batch_size: Batch size for this test run.
        seq_len:    Sequence length for this test run.

    Returns:
        ``(model, tokenizer, generation_config)``
    """
    ModelClass = import_class(config["model_class"])
    ConfigClass = import_class(config["config_class"])

    neuron_config_dict = load_neuron_config_from_compiled(config["compiled_model_path"])

    # Resolve dtype
    dtype_str = neuron_config_dict.get("torch_dtype", "torch.bfloat16")
    if isinstance(dtype_str, str):
        if dtype_str.startswith("torch."):
            dtype = getattr(torch, dtype_str.split(".")[1])
        elif dtype_str in ("float32", "float16", "bfloat16"):
            dtype = getattr(torch, dtype_str)
        else:
            dtype = torch.bfloat16
    else:
        dtype = dtype_str

    is_moe = "moe_tp_degree" in neuron_config_dict or "router_config" in neuron_config_dict
    NeuronConfigClass = MoENeuronConfig if is_moe else NeuronConfig

    kwargs = {
        "tp_degree": neuron_config_dict.get("tp_degree", config.get("tp_degree", 1)),
        "batch_size": neuron_config_dict.get("batch_size", batch_size),
        "seq_len": neuron_config_dict.get("seq_len", seq_len),
        "torch_dtype": dtype,
        "save_sharded_checkpoint": neuron_config_dict.get("save_sharded_checkpoint", True),
        "on_cpu": neuron_config_dict.get("on_cpu", False),
    }
    for p in [
        "world_size", "max_context_length", "enable_bucketing",
        "enable_cte_modular_flow", "ep_degree", "moe_ep_degree",
    ]:
        if p in neuron_config_dict:
            kwargs[p] = neuron_config_dict[p]
    if "max_context_length" not in kwargs:
        kwargs["max_context_length"] = kwargs["seq_len"]

    neuron_config = NeuronConfigClass(**kwargs)

    generation_config = GenerationConfig.from_pretrained(
        config["model_path"], do_sample=False, top_k=1, trust_remote_code=True,
    )

    try:
        model_config = ConfigClass.from_pretrained(
            config["model_path"], neuron_config=neuron_config,
        )
    except (TypeError, AttributeError):
        model_config = ConfigClass(
            neuron_config, load_config=load_pretrained_config(config["model_path"]),
        )

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"], padding_side="right", trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.bos_token is not None:
            tokenizer.pad_token = tokenizer.bos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    try:
        if hasattr(ModelClass, "from_pretrained"):
            model = ModelClass.from_pretrained(
                config["compiled_model_path"], config=model_config,
            )
        else:
            raise AttributeError("No from_pretrained method")
    except (TypeError, AttributeError, Exception):
        model = ModelClass(config["model_path"], model_config)

    ensure_generation_config_version(model)
    return model, tokenizer, generation_config


# ---------------------------------------------------------------------------
# HF golden model loading
# ---------------------------------------------------------------------------

def _import_hf_reference_class(class_path: str):
    """Import a HuggingFace model class for use as the golden reference.

    Supports ``module.path:ClassName`` and ``module.path.ClassName``.
    Used when the config specifies ``hf_reference_class`` to override
    ``AutoModelForCausalLM`` (e.g. for vision-language models or custom
    architectures that ``Auto`` classes don't recognise).
    """
    if ":" in class_path:
        module_path, class_name = class_path.rsplit(":", 1)
    elif "." in class_path:
        module_path, class_name = class_path.rsplit(".", 1)
    else:
        raise ValueError(f"Invalid hf_reference_class: {class_path}")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _auto_resolve_model_class(model_path: str, load_kwargs: dict, original_error: Exception):
    """Last-resort model class resolution when ``AutoModelForCausalLM`` fails.

    Reads ``config.json`` to get ``model_type``, then tries to import
    ``{ModelType}ForCausalLM``, ``{ModelType}ForConditionalGeneration``,
    and ``{ModelType}Model`` from the corresponding transformers submodule.
    If all fail, falls back to ``AutoModel``.

    Raises ``ValueError`` (chained from ``original_error``) if nothing works.
    """
    from transformers import AutoConfig, AutoModel

    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type = getattr(cfg, "model_type", None)

        if model_type:
            module_name = f"transformers.models.{model_type}.modeling_{model_type}"
            class_base = "".join(w.capitalize() for w in model_type.split("_"))
            for suffix in ["ForCausalLM", "ForConditionalGeneration", "Model"]:
                try:
                    mod = importlib.import_module(module_name)
                    ModelClass = getattr(mod, f"{class_base}{suffix}")
                    return ModelClass.from_pretrained(model_path, **load_kwargs)
                except (ImportError, AttributeError):
                    continue

        return AutoModel.from_pretrained(model_path, **load_kwargs)
    except Exception as fallback_e:
        raise ValueError(
            f"{original_error}\nAuto-resolution also failed: {fallback_e}\n"
            f"Set hf_reference_class in config to fix."
        ) from original_error


def load_hf_golden_model(
    model_path: str,
    config: Optional[dict] = None,
    neuron_dtype: torch.dtype = None,
    hf_dtype_override: Optional[str] = None,
):
    """Load the HuggingFace model that serves as the accuracy ground truth.

    By default loads in **FP32** — this is intentional.  The Neuron model
    runs in BF16, and comparing BF16-vs-BF16 would mask the very precision
    issues we're trying to detect.  Use ``hf_dtype_override="bfloat16"``
    only if you know what you're doing.

    Loading strategy:
      1. If ``config["hf_reference_class"]`` is set, import that class
         directly (for non-standard architectures).
      2. Otherwise try ``AutoModelForCausalLM.from_pretrained``.
      3. If that raises "Unrecognized configuration class", fall back to
         ``_auto_resolve_model_class`` which tries several class suffixes.

    Always uses ``attn_implementation="eager"`` to avoid flash-attention
    issues on CPU, and ``low_cpu_mem_usage=True`` to keep peak RAM down.

    Args:
        model_path:       Path to the HF model directory (with config.json,
                          tokenizer, and weights).
        config:           Validation config dict (optional — checked for
                          ``hf_reference_class`` and ``hf_reference_model_path``).
        neuron_dtype:     Unused (kept for API compat).
        hf_dtype_override: Force a specific dtype string ("float32",
                          "bfloat16", "float16").

    Returns:
        HuggingFace model in eval mode.
    """
    if hf_dtype_override and hf_dtype_override != "auto":
        dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
        hf_dtype = dtype_map.get(hf_dtype_override, torch.float32)
    else:
        hf_dtype = torch.float32

    load_kwargs = dict(
        torch_dtype=hf_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    hf_reference_class = config.get("hf_reference_class") if config else None
    if hf_reference_class:
        ModelClass = _import_hf_reference_class(hf_reference_class)
        hf_ref_path = config.get("hf_reference_model_path", model_path) if config else model_path
        hf_model = ModelClass.from_pretrained(hf_ref_path, **load_kwargs)
        hf_model.eval()
        return hf_model

    try:
        hf_model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        hf_model.eval()
        return hf_model
    except ValueError as e:
        if "Unrecognized configuration class" in str(e):
            hf_model = _auto_resolve_model_class(model_path, load_kwargs, e)
            hf_model.eval()
            return hf_model
        raise RuntimeError(f"Failed to load HF model: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load HF model: {e}")


# ---------------------------------------------------------------------------
# Comprehensive report printer
# ---------------------------------------------------------------------------

def _print_comprehensive_accuracy_report(
    token_details: Optional[dict],
    logit_details: Optional[dict],
    enhanced_details: Optional[dict] = None,
):
    """Print a unified accuracy report to stdout (used in ``--comprehensive`` mode).

    Combines token-matching results, NxDI logit-matching v2 results, and
    enhanced distribution metrics into a single formatted block.  Only
    prints sections for which data is available.
    """
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ACCURACY REPORT")
    print(f"{'='*80}")

    if token_details:
        rate = token_details.get("match_rate", 0)
        total = token_details.get("total_tokens_compared", 0)
        matching = token_details.get("matching_tokens", 0)
        print(f"\n  TOKEN MATCHING: {rate*100:.2f}% ({matching}/{total})")

    if logit_details:
        status = "PASSED" if logit_details.get("logit_matching_passed") else "FAILED"
        print(f"\n  LOGIT MATCHING (v2): {status}")
        print(f"    Tokens validated:  {logit_details.get('total_tokens_validated', 0)}")
        print(f"    Divergence count:  {logit_details.get('divergence_count', 0)}")
        print(f"    Divergence rate:   {logit_details.get('divergence_rate', 0)*100:.2f}%")

        md = logit_details.get("max_divergence_difference", {})
        if md.get("error", -1) >= 0:
            print(f"    Max divergence:    {md['error']:.6f} "
                  f"(batch {md.get('batch')}, token {md.get('token')})")
        print(f"    Mean divergence:   {logit_details.get('mean_divergence_difference', 0):.6f}")

        for label_prefix, key in [("Max", "max_top_k_errors"), ("Avg", "avg_top_k_errors")]:
            errs = logit_details.get(key, {})
            if errs:
                print(f"    Top-K {label_prefix} Errors:")
                for k in sorted(errs.keys(), key=lambda x: (x == "None", x)):
                    val = errs[k]
                    tag = f"k={k}" if k != "None" else "full"
                    if isinstance(val, dict):
                        print(f"      {tag:>10}: {val['error']:.6f} "
                              f"(batch {val.get('batch')}, token {val.get('token')})")
                    else:
                        print(f"      {tag:>10}: {val:.6f}")

        for label, key in [
            ("Normalized MAE", "avg_normalized_mae_by_k"),
            ("Normalized MSE", "avg_normalized_mse_by_k"),
        ]:
            vals = logit_details.get(key, {})
            if vals:
                print(f"    {label} (avg over tokens):")
                for k in sorted(vals.keys(), key=lambda x: (x == "None", x)):
                    tag = f"k={k}" if k != "None" else "full"
                    print(f"      {tag:>10}: {vals[k]:.8f}")

        print(f"    Mean logit shift:  {logit_details.get('mean_shift', 0):.6f}")
        print(f"    Max |shift|:       {logit_details.get('max_shift', 0):.6f}")
        print(f"    Confidence gap (expected top1-top2): "
              f"{logit_details.get('mean_expected_top1_top2_diff', 0):.4f}")
        print(f"    Confidence gap (actual top1-top2):   "
              f"{logit_details.get('mean_actual_top1_top2_diff', 0):.4f}")
        print(f"    Mean top-1 relative error: "
              f"{logit_details.get('mean_top1_relative_error', 0):.6f}")
        print(f"    Max  top-1 relative error: "
              f"{logit_details.get('max_top1_relative_error', 0):.6f}")
        print(f"    Mean top-2 relative error: "
              f"{logit_details.get('mean_top2_relative_error', 0):.6f}")
        print(f"    Max  top-2 relative error: "
              f"{logit_details.get('max_top2_relative_error', 0):.6f}")

    if enhanced_details and enhanced_details.get("enhanced_metrics_available"):
        print(f"\n  DISTRIBUTION METRICS:")
        print(f"    Cosine similarity (mean): "
              f"{enhanced_details.get('logit_cosine_similarity_mean', 0):.6f}")
        print(f"    Cosine similarity (min):  "
              f"{enhanced_details.get('logit_cosine_similarity_min', 0):.6f}")
        print(f"    Cosine similarity (std):  "
              f"{enhanced_details.get('logit_cosine_similarity_std', 0):.6f}")
        if "top5_agreement" in enhanced_details:
            print(f"    Top-5 agreement:          "
                  f"{enhanced_details['top5_agreement']*100:.2f}%")
        print(f"    KL divergence:            "
              f"{enhanced_details.get('kl_divergence', 0):.6f}")
        print(f"    Max logit diff:           "
              f"{enhanced_details.get('max_logit_diff', 0):.4f}")
        print(f"    Mean logit diff:          "
              f"{enhanced_details.get('mean_logit_diff', 0):.4f}")
        if "mean_prob_of_hf_token" in enhanced_details:
            print(f"    Mean P(HF token):         "
                  f"{enhanced_details['mean_prob_of_hf_token']:.6f}")

    print(f"\n{'='*80}")


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_accuracy_results(
    config: dict,
    test_params: dict,
    passed: bool,
    error_details: Optional[dict] = None,
    token_details: Optional[dict] = None,
    logit_details: Optional[dict] = None,
    enhanced_details: Optional[dict] = None,
):
    """Persist accuracy test results as a timestamped JSON file.

    Written to ``results/accuracy/{model_name}/accuracy_bs{B}_seq{S}_{pass|fail}_{timestamp}.json``.
    Includes whichever detail dicts are non-None (token matching, logit
    matching v2, enhanced metrics, error details).

    Returns the ``Path`` to the written file.
    """
    results_dir = Path(config.get("results_dir", "agent_artifacts/results")) / "accuracy" / config["model_name"]
    results_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "timestamp": datetime.now().isoformat(),
        "model_name": config["model_name"],
        "model_path": config["model_path"],
        "compiled_model_path": config["compiled_model_path"],
        "test_parameters": test_params,
        "test_status": "PASSED" if passed else "FAILED",
        "num_tokens_checked": config.get("num_tokens_to_check"),
        "error_details": error_details,
    }
    if token_details:
        result["token_matching"] = token_details
    if logit_details:
        result["logit_matching_v2"] = logit_details
    if enhanced_details:
        result["enhanced_metrics"] = enhanced_details
    status = "pass" if passed else "fail"
    fp = results_dir / (
        f"accuracy_bs{test_params['batch_size']}_seq{test_params['seq_len']}"
        f"_{status}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(fp, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved accuracy results to: {fp}")
    return fp


# ---------------------------------------------------------------------------
# Public test functions
# ---------------------------------------------------------------------------

def test_model_load(config: dict, test_params: dict) -> bool:
    """Smoke test: can the compiled model load without crashing?

    Creates the model via ``create_model``, then calls ``model.load()``
    with the compiled checkpoint path.  If anything raises, writes a
    failure JSON to ``results/smoke_tests/{model_name}/`` and returns False.

    This is the first stage of progressive validation — if the model can't
    even load, there's no point running accuracy or performance tests.
    """
    bs, sl = test_params["batch_size"], test_params["seq_len"]
    try:
        model, tokenizer, gc = create_model(config, bs, sl)
        if not os.path.exists(config["compiled_model_path"]):
            raise FileNotFoundError(f"Compiled model not found: {config['compiled_model_path']}")
        model.load(config["compiled_model_path"])
        print("✓ Model loaded successfully")
        del model
        return True
    except Exception as e:
        print(f"✗ SMOKE TEST FAILED: {e}")
        results_dir = Path(config.get("results_dir", "agent_artifacts/results")) / "smoke_tests" / config["model_name"]
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / f"smoke_test_fail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "model_name": config["model_name"],
                "test_status": "FAILED",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }, f, indent=2)
        return False


def test_accuracy(
    config: dict,
    test_params: dict,
    use_token_matching: bool = True,
    use_chat_template: bool = False,
    comprehensive: bool = False,
    hf_dtype_override: Optional[str] = None,
    skip_hf_comparison: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """Run accuracy validation — the second stage of progressive validation.

    Supports four modes (controlled by flags):

      * **Token matching** (default, ``use_token_matching=True``):
        Greedy-decodes 10 factual prompts on both models, compares tokens.
        Also computes enhanced distribution metrics as a bonus.

      * **Logit matching v2** (``use_token_matching=False``):
        Delegates to NxDI's ``check_accuracy_logits_v2`` for stricter
        distribution-level comparison.

      * **Comprehensive** (``comprehensive=True``):
        Runs both token matching AND logit matching.  Both must pass.
        Prints a unified report at the end.

      * **Inference-only** (``skip_hf_comparison=True``):
        No HF model loaded.  Just checks the Neuron model can generate
        non-degenerate text.

    The HF golden model is loaded in FP32 (see ``load_hf_golden_model``).
    Results are always persisted via ``save_accuracy_results`` in a
    ``finally`` block, even if the test crashes.

    Args:
        config:              Validation config dict.
        test_params:         ``{"batch_size": int, "seq_len": int, ...}``
        use_token_matching:  Use multi-prompt token matching (default True).
        use_chat_template:   Wrap prompts with the tokenizer's chat template.
        comprehensive:       Run both token + logit matching.
        hf_dtype_override:   Force HF model dtype (e.g. "bfloat16").
        skip_hf_comparison:  Inference-only mode (no HF model needed).

    Returns:
        ``(passed, details)`` — ``details`` contains whichever sub-dicts
        are relevant: ``token_matching``, ``logit_matching_v2``,
        ``enhanced_metrics``, ``error_details``.
    """
    bs, sl = test_params["batch_size"], test_params["seq_len"]
    hf_model = None
    token_details = None
    logit_details = None
    enhanced_details = None
    error_details = None
    passed = False

    try:
        model, tokenizer, generation_config = create_model(config, bs, sl)
        if not os.path.exists(config["compiled_model_path"]):
            raise FileNotFoundError(f"Compiled model not found: {config['compiled_model_path']}")
        model.load(config["compiled_model_path"])

        neuron_dtype = model.config.neuron_config.torch_dtype
        ensure_generation_config_version(model)
        ensure_pad_token(generation_config, tokenizer)

        # --- Inference-only mode ---
        if skip_hf_comparison:
            passed, details = run_inference_only_validation(model, tokenizer)
            return passed, details

        # Setup for NxDI logit matching
        _load_nxdi_accuracy_utils()
        patch_load_hf_model(
            model,
            lambda path: load_hf_golden_model(
                path, config, neuron_dtype=neuron_dtype, hf_dtype_override=hf_dtype_override,
            ),
        )

        chat = use_chat_template or config.get("use_chat_template", False)

        # --- Token Matching ---
        if use_token_matching or comprehensive:
            hf_model = load_hf_golden_model(
                config["model_path"], config,
                neuron_dtype=neuron_dtype, hf_dtype_override=hf_dtype_override,
            )
            token_passed, token_details = check_accuracy_with_hf_golden(
                neuron_model=model, hf_model=hf_model, tokenizer=tokenizer,
                generation_config=generation_config,
                num_tokens_to_check=config.get("num_tokens_to_check", 64),
                use_chat_template=chat,
            )

            # Enhanced distribution metrics (both models are loaded)
            enhanced_details = compute_enhanced_metrics(
                model, hf_model, tokenizer,
            )

            if comprehensive:
                passed = token_passed
            else:
                passed = token_passed
                if not passed:
                    error_details = {
                        "error_type": "AccuracyMismatch",
                        "error_message": (
                            f"Token match rate {token_details['match_rate']*100:.2f}% "
                            f"below threshold"
                        ),
                        "accuracy_details": token_details,
                    }

        # --- Logit Matching v2 ---
        if (not use_token_matching) or comprehensive:
            logit_passed, logit_details = run_logit_matching_v2(
                model, tokenizer, generation_config, config,
            )
            if comprehensive:
                passed = passed and logit_passed
            else:
                passed = logit_passed
                if not passed and error_details is None:
                    error_details = {
                        "error_type": "LogitMatchingFailed",
                        "error_message": logit_details.get(
                            "error_message", "Logit validation failed"
                        ),
                    }

        # --- Report ---
        if comprehensive:
            _print_comprehensive_accuracy_report(token_details, logit_details, enhanced_details)

        if not passed and error_details is None:
            error_details = {
                "error_type": "AccuracyFailed",
                "error_message": "One or more accuracy checks failed",
            }

    except Exception as e:
        if error_details is None:
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "full_traceback": traceback.format_exc(),
            }
        passed = False

    finally:
        save_accuracy_results(
            config, test_params, passed, error_details,
            token_details=token_details, logit_details=logit_details,
            enhanced_details=enhanced_details,
        )
        if "model" in dir() and model is not None:
            del model
        if hf_model is not None:
            del hf_model

    all_details = {}
    if token_details:
        all_details["token_matching"] = token_details
    if logit_details:
        all_details["logit_matching_v2"] = logit_details
    if enhanced_details:
        all_details["enhanced_metrics"] = enhanced_details
    if error_details:
        all_details["error_details"] = error_details
    return passed, all_details


def save_benchmark_results(
    config: dict,
    benchmark_report: Optional[dict],
    test_params: dict,
    passed: bool,
    error_details: Optional[dict] = None,
):
    results_dir = Path(config.get("results_dir", "agent_artifacts/results")) / "benchmarks" / config["model_name"]
    results_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "timestamp": datetime.now().isoformat(),
        "model_name": config["model_name"],
        "model_path": config["model_path"],
        "compiled_model_path": config["compiled_model_path"],
        "test_parameters": test_params,
        "test_status": "PASSED" if passed else "FAILED",
        "benchmark_report": benchmark_report,
        "error_details": error_details,
    }
    status = "pass" if passed else "fail"
    fp = results_dir / (
        f"benchmark_bs{test_params['batch_size']}_seq{test_params['seq_len']}"
        f"_{status}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(fp, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved benchmark results to: {fp}")
    return fp


def test_performance(
    config: dict,
    test_params: dict,
) -> Tuple[bool, Dict[str, Any]]:
    bs, sl = test_params["batch_size"], test_params["seq_len"]
    ttft_thresh = test_params["ttft_threshold"]
    tp_thresh = test_params["throughput_threshold"]

    error_details = None
    passed = False
    report = None

    try:
        model, tokenizer, generation_config = create_model(config, bs, sl)
        model.load(config["compiled_model_path"])

        ensure_generation_config_version(model)
        ensure_pad_token(generation_config, tokenizer)

        try:
            report = benchmark_sampling(model, generation_config=generation_config)
        except (TypeError, AttributeError, RuntimeError) as e:
            print(f"benchmark_sampling failed ({e}), falling back to custom benchmark...")
            report = run_custom_benchmark(model, tokenizer, bs, sl)

        print_benchmark_report(report)

        ttft = report["context_encoding_model"]["latency_ms_p50"]
        throughput = report["token_generation_model"]["throughput"]
        ttft_limit = ttft_thresh * PERFORMANCE_TOLERANCE
        tp_limit = tp_thresh * THROUGHPUT_TOLERANCE

        if ttft >= ttft_limit:
            error_details = {"error_type": "TTFTExceeded", "actual": ttft, "limit": ttft_limit}
            raise AssertionError(f"TTFT {ttft:.2f}ms >= {ttft_limit:.2f}ms")
        if throughput <= tp_limit:
            error_details = {"error_type": "ThroughputLow", "actual": throughput, "limit": tp_limit}
            raise AssertionError(f"Throughput {throughput:.2f} <= {tp_limit:.2f}")

        passed = True

    except Exception as e:
        if error_details is None:
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "full_traceback": traceback.format_exc(),
            }
        passed = False

    finally:
        save_benchmark_results(config, report, test_params, passed, error_details)
        if "model" in dir() and model is not None:
            del model

    return passed, report or {}
