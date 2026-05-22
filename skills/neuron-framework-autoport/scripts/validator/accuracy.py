# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Accuracy validation methods for comparing Neuron model outputs against
a HuggingFace golden reference.

Three accuracy strategies live here, from lightest to heaviest:

1. **Inference-only** (``run_inference_only_validation``)
   No HF model needed.  Generates 20 tokens from 3 prompts and fails if
   the output is empty or degenerate (all-same-character).  Useful as a
   quick smoke-check on boxes without enough RAM for the HF reference.

2. **Token matching** (``check_accuracy_with_hf_golden``)
   Greedy-decodes both models on the 10 factual prompts from
   ``constants.DEFAULT_VALIDATION_PROMPTS``, compares token-by-token, and
   reports an overall match rate.  Passes when rate ≥ 95%.

3. **NxDI logit matching v2** (``run_logit_matching_v2``)
   Delegates to ``neuronx_distributed_inference.utils.accuracy
   .check_accuracy_logits_v2``, which compares full logit distributions
   position-by-position with model-type-aware tolerances.  Stricter than
   token matching — catches subtle numerical drift that doesn't flip the
   argmax.

Also contains:

* ``extract_logit_validation_summary`` — flattens the raw per-token dicts
  returned by NxDI's logit validation into a single JSON-friendly summary.
* ``format_prompt_with_chat_template`` / ``generate_with_neuron_model`` —
  small helpers shared across the strategies.
"""

import torch
from typing import Any, Dict, List, Optional, Tuple

from transformers import GenerationConfig

from .constants import (
    DEFAULT_VALIDATION_PROMPTS,
    TOKEN_MATCH_THRESHOLD,
    get_logit_thresholds,
)
from .patches import (
    ensure_generation_config_version,
    patch_generation_mixin,
)


# ---------------------------------------------------------------------------
# Lazy NxDI accuracy imports (avoids pulling in torchvision at module load)
# ---------------------------------------------------------------------------
from typing import Callable

_nxdi_check_accuracy_logits_v2: Optional[Callable] = None
_nxdi_generate_expected_logits: Optional[Callable] = None
_nxdi_prepare_inputs_from_prompt: Optional[Callable] = None


def _load_nxdi_accuracy_utils():
    """Lazily import the three NxDI accuracy helpers on first use.

    Deferred because ``neuronx_distributed_inference.utils.accuracy``
    transitively pulls in torchvision and other heavy deps that slow down
    module load and aren't needed for smoke tests or benchmarks.

    Also re-runs ``patch_generation_mixin`` after the import, since the
    ``HuggingFaceGenerationAdapter`` class may not have existed yet when
    patches.py first ran.
    """
    global _nxdi_check_accuracy_logits_v2
    global _nxdi_generate_expected_logits
    global _nxdi_prepare_inputs_from_prompt
    if _nxdi_check_accuracy_logits_v2 is None:
        from neuronx_distributed_inference.utils.accuracy import (
            check_accuracy_logits_v2,
            generate_expected_logits,
            prepare_inputs_from_prompt,
        )
        _nxdi_check_accuracy_logits_v2 = check_accuracy_logits_v2
        _nxdi_generate_expected_logits = generate_expected_logits
        _nxdi_prepare_inputs_from_prompt = prepare_inputs_from_prompt
        patch_generation_mixin()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_prompt_with_chat_template(tokenizer, prompt: str) -> str:
    """Wrap ``prompt`` in the tokenizer's chat template (if one exists).

    For instruct / chat models the raw prompt needs to be wrapped in the
    model's expected turn format (e.g. ``<|user|>\\n...``).  If the
    tokenizer has no ``chat_template``, returns the prompt unchanged.
    """
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_with_neuron_model(neuron_model, input_ids, max_new_tokens: int):
    """Manual greedy-decode loop for Neuron models that lack ``.generate()``.

    Some ported models only expose a raw ``__call__`` (returns logits) and
    don't implement the HuggingFace ``GenerationMixin`` interface.  This
    function does a simple argmax loop as a fallback.

    Args:
        neuron_model: Compiled Neuron model (must accept ``input_ids`` and
                      ``position_ids``).
        input_ids:    ``[batch, seq]`` tensor — the prompt tokens.
        max_new_tokens: How many tokens to generate.

    Returns:
        ``[batch, seq + max_new_tokens]`` tensor of input + generated ids.
    """
    generated_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        seq_len = generated_ids.shape[1]
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(generated_ids.shape[0], -1)
        outputs = neuron_model(generated_ids, position_ids=position_ids)
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        if isinstance(logits, list):
            logits = logits[0]
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
    return generated_ids


# ---------------------------------------------------------------------------
# NxDI logit-validation metric extraction
# ---------------------------------------------------------------------------

def extract_logit_validation_summary(results: list) -> dict:
    """Flatten NxDI's raw per-token logit-validation output into one dict.

    ``check_accuracy_logits_v2`` returns a 2-D list
    ``results[batch_idx][token_idx]`` where each element is a dict of
    per-position metrics (divergence flag, error_map, shift, etc.).
    This function walks the full grid and aggregates into a single
    JSON-serialisable summary with counts, means, maxes, and locations
    of worst-case values.

    Key fields in the returned dict:
        total_tokens_validated, divergence_count, divergence_rate,
        max_divergence_difference  (with batch + token location),
        mean_divergence_difference,
        max_top_k_errors / avg_top_k_errors  (keyed by k),
        avg_normalized_mae_by_k / avg_normalized_mse_by_k,
        mean_shift, max_shift,
        mean/max top1/top2 relative errors,
        confidence-gap stats (expected vs actual top1−top2 diff).

    Args:
        results: 2-D list ``[batch_size][output_length]`` of per-token dicts.

    Returns:
        Flat dict of aggregated metrics (all values are plain Python floats
        or dicts of floats — no tensors).
    """
    if not results or not results[0]:
        return {}

    batch_size = len(results)
    num_tokens = len(results[0])

    divergence_count = 0
    divergence_diffs: List[float] = []
    top_k_errors: Dict[Any, List[float]] = {}
    top_k_total_mae: Dict[Any, List[float]] = {}
    top_k_total_mse: Dict[Any, List[float]] = {}
    shifts: List[float] = []
    top1_top2_diffs_expected: List[float] = []
    top1_top2_diffs_actual: List[float] = []
    top1_top2_rel_diffs_expected: List[float] = []
    top1_top2_rel_diffs_actual: List[float] = []
    actual_with_expected_rel_diffs: List[float] = []
    top1_rel_errors: List[float] = []
    top2_rel_errors: List[float] = []

    max_divergence = {"error": -1, "batch": -1, "token": -1}
    max_top_k_errors: Dict[Any, dict] = {}

    for b in range(batch_size):
        for t in range(len(results[b])):
            r = results[b][t]

            if r.get("divergence", False):
                divergence_count += 1
                dd = r.get("divergence_difference", 0)
                if hasattr(dd, "item"):
                    dd = dd.item()
                divergence_diffs.append(dd)
                if dd > max_divergence["error"]:
                    max_divergence = {"error": dd, "batch": b, "token": t}

            for k, err in r.get("error_map", {}).items():
                top_k_errors.setdefault(k, []).append(err)
                if k not in max_top_k_errors or abs(err) > abs(max_top_k_errors[k]["error"]):
                    max_top_k_errors[k] = {"error": err, "batch": b, "token": t}

            for k, errs in r.get("total_errors", {}).items():
                if "mean_abs_error" in errs:
                    top_k_total_mae.setdefault(k, []).append(errs["mean_abs_error"])
                if "mean_squared_error" in errs:
                    top_k_total_mse.setdefault(k, []).append(errs["mean_squared_error"])

            s = r.get("shift", 0)
            if hasattr(s, "item"):
                s = s.item()
            shifts.append(s)

            top1_top2_diffs_expected.append(r.get("expected_top1_top2_diff", 0))
            top1_top2_diffs_actual.append(r.get("actual_top1_top2_diff", 0))
            top1_top2_rel_diffs_expected.append(r.get("expected_top1_top2_relative_diff", 0))
            top1_top2_rel_diffs_actual.append(r.get("actual_top1_top2_relative_diff", 0))
            actual_with_expected_rel_diffs.append(
                r.get("actual_with_expected_top1_top2_relative_diff", 0)
            )

            t1re = r.get("top1_relative_errors", 0)
            t2re = r.get("top2_relative_errors", 0)
            if hasattr(t1re, "item"):
                t1re = t1re.item()
            if hasattr(t2re, "item"):
                t2re = t2re.item()
            top1_rel_errors.append(t1re)
            top2_rel_errors.append(t2re)

    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    def _max_abs(lst):
        return max(abs(v) for v in lst) if lst else 0.0

    return {
        "total_tokens_validated": batch_size * num_tokens,
        "batch_size": batch_size,
        "output_length": num_tokens,
        "divergence_count": divergence_count,
        "divergence_rate": divergence_count / (batch_size * num_tokens) if num_tokens else 0,
        "max_divergence_difference": max_divergence,
        "mean_divergence_difference": _mean(divergence_diffs),
        "max_top_k_errors": {str(k): v for k, v in max_top_k_errors.items()},
        "avg_top_k_errors": {str(k): _mean(e) for k, e in top_k_errors.items()},
        "avg_normalized_mae_by_k": {str(k): _mean(v) for k, v in top_k_total_mae.items()},
        "avg_normalized_mse_by_k": {str(k): _mean(v) for k, v in top_k_total_mse.items()},
        "mean_shift": _mean(shifts),
        "max_shift": _max_abs(shifts),
        "mean_expected_top1_top2_diff": _mean(top1_top2_diffs_expected),
        "mean_actual_top1_top2_diff": _mean(top1_top2_diffs_actual),
        "mean_expected_top1_top2_relative_diff": _mean(top1_top2_rel_diffs_expected),
        "mean_actual_top1_top2_relative_diff": _mean(top1_top2_rel_diffs_actual),
        "mean_actual_with_expected_top1_top2_relative_diff": _mean(actual_with_expected_rel_diffs),
        "mean_top1_relative_error": _mean(top1_rel_errors),
        "mean_top2_relative_error": _mean(top2_rel_errors),
        "max_top1_relative_error": _max_abs(top1_rel_errors),
        "max_top2_relative_error": _max_abs(top2_rel_errors),
    }


# ---------------------------------------------------------------------------
# Token matching (multi-prompt)
# ---------------------------------------------------------------------------

def check_accuracy_with_hf_golden(
    neuron_model,
    hf_model,
    tokenizer,
    generation_config,
    num_tokens_to_check: int = 256,
    use_chat_template: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """Multi-prompt greedy token matching: Neuron vs HF golden.

    Iterates over ``DEFAULT_VALIDATION_PROMPTS`` (10 factual prompts),
    greedy-decodes ``num_tokens_to_check`` tokens from each model, and
    compares the generated token ids position-by-position.

    The overall match rate is the total matching tokens across all prompts
    divided by total tokens compared.  Passes when that rate is ≥
    ``TOKEN_MATCH_THRESHOLD`` (currently 95%).

    Per-prompt details (match rate, first divergence index, decoded text
    samples) are included in the returned dict for debugging.

    Args:
        neuron_model:        Loaded + compiled Neuron model.
        hf_model:            HuggingFace reference model (FP32, eval mode).
        tokenizer:           Shared tokenizer (must have pad_token set).
        generation_config:   ``GenerationConfig`` (greedy, do_sample=False).
        num_tokens_to_check: Max new tokens per prompt (capped by seq_len).
        use_chat_template:   Wrap prompts with the tokenizer's chat template.

    Returns:
        (passed, details) — ``passed`` is True when overall match rate ≥ threshold.
    """
    prompts_to_test = list(DEFAULT_VALIDATION_PROMPTS)
    # Automatically apply chat template if tokenizer has one
    prompts_to_test = [format_prompt_with_chat_template(tokenizer, p) for p in prompts_to_test]

    batch_size = neuron_model.config.neuron_config.batch_size
    seq_len = neuron_model.config.neuron_config.seq_len

    total_matching = 0
    total_compared = 0
    per_prompt_results = []

    for idx, test_prompt in enumerate(prompts_to_test):
        prompts_batch = [test_prompt] * batch_size
        inputs = tokenizer(prompts_batch, padding=True, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        input_len = input_ids.shape[1]

        max_safe = seq_len - input_len - 1
        if max_safe < 1:
            continue
        actual_tokens = min(num_tokens_to_check, max_safe)

        # HF generation
        with torch.no_grad():
            try:
                hf_out = hf_model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=actual_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True, output_scores=False,
                )
            except (AttributeError, TypeError) as e:
                if any(k in str(e) for k in ("seen_tokens", "DynamicCache", "NoneType", "shape", "past_key_values")):
                    hf_out = hf_model.generate(
                        input_ids=input_ids, attention_mask=attention_mask,
                        max_new_tokens=actual_tokens, do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        return_dict_in_generate=True, output_scores=False,
                        use_cache=False,
                    )
                else:
                    raise

        hf_ids = hf_out.sequences

        # Neuron generation
        with torch.no_grad():
            try:
                neuron_out = neuron_model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=actual_tokens, generation_config=generation_config,
                )
                neuron_ids = neuron_out.sequences if hasattr(neuron_out, "sequences") else neuron_out
            except (AttributeError, TypeError):
                neuron_ids = generate_with_neuron_model(neuron_model, input_ids, actual_tokens)

        # Compare
        hf_gen = hf_ids[:, input_len:]
        neuron_gen = neuron_ids[:, input_len:]
        min_len = min(hf_gen.shape[1], neuron_gen.shape[1], actual_tokens)
        hf_gen, neuron_gen = hf_gen[:, :min_len], neuron_gen[:, :min_len]

        matches = (hf_gen == neuron_gen).float()
        prompt_total = matches.numel()
        prompt_matching = int(matches.sum().item())
        prompt_rate = prompt_matching / prompt_total if prompt_total > 0 else 0

        first_div = None
        for i in range(min_len):
            if not torch.all(hf_gen[:, i] == neuron_gen[:, i]):
                first_div = i
                break

        total_matching += prompt_matching
        total_compared += prompt_total
        per_prompt_results.append({
            "prompt": test_prompt,
            "match_rate": prompt_rate,
            "matching_tokens": prompt_matching,
            "total_tokens": prompt_total,
            "first_divergence_idx": first_div,
            "hf_output": tokenizer.batch_decode(hf_ids, skip_special_tokens=True)[0][:300],
            "neuron_output": tokenizer.batch_decode(neuron_ids, skip_special_tokens=True)[0][:300],
        })

    overall = total_matching / total_compared if total_compared > 0 else 0
    details = {
        "total_tokens_compared": total_compared,
        "matching_tokens": total_matching,
        "match_rate": overall,
        "num_prompts_tested": len(prompts_to_test),
        "per_prompt_results": per_prompt_results,
    }
    passed = overall >= TOKEN_MATCH_THRESHOLD
    return passed, details


# ---------------------------------------------------------------------------
# Logit matching v2 (NxDI)
# ---------------------------------------------------------------------------

def run_logit_matching_v2(
    model, tokenizer, generation_config, config,
) -> Tuple[bool, dict]:
    """Run NxDI's ``check_accuracy_logits_v2`` and return a flat summary.

    This is the stricter accuracy check.  It generates expected logits from
    the HF golden (loaded internally by NxDI via the patched
    ``load_hf_model``), then compares full logit distributions at every
    generated position against model-type-aware tolerances (see
    ``constants.get_logit_thresholds``).

    If the check raises ``LogitMatchingValidationError``, we catch it,
    extract the partial results from the exception, and still return a
    summary — the caller decides whether to treat it as fatal.

    Args:
        model:             Loaded Neuron model (with ``load_hf_model`` patched).
        tokenizer:         Shared tokenizer.
        generation_config: Greedy ``GenerationConfig``.
        config:            Validation config dict (needs ``model_name``,
                           ``num_tokens_to_check``, optionally
                           ``use_chat_template``).

    Returns:
        (passed, summary) — ``summary`` is the output of
        ``extract_logit_validation_summary`` plus tolerance metadata.
    """
    _load_nxdi_accuracy_utils()

    tol_map, div_tol = get_logit_thresholds(config["model_name"])
    num_tokens = config.get("num_tokens_to_check", 256)

    # Automatically apply chat template if tokenizer has one
    prompt = format_prompt_with_chat_template(tokenizer, "Hello, I am a language model")

    inputs = _nxdi_prepare_inputs_from_prompt(model, tokenizer, prompt=prompt)  # type: ignore[misc]
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    expected_logits = _nxdi_generate_expected_logits(  # type: ignore[misc]
        model, input_ids, attention_mask, generation_config,
        num_tokens=num_tokens, tokenizer=tokenizer,
    )

    from neuronx_distributed_inference.utils.exceptions import LogitMatchingValidationError

    passed = True
    logit_results = None
    error_msg = None
    try:
        logit_results = _nxdi_check_accuracy_logits_v2(  # type: ignore[misc]
            neuron_model=model,
            expected_logits=expected_logits,
            inputs_input_ids=input_ids,
            inputs_attention_mask=attention_mask,
            generation_config=generation_config,
            divergence_difference_tol=div_tol,
            tol_map=tol_map,
            num_tokens_to_check=num_tokens,
            tokenizer=tokenizer,
        )
    except LogitMatchingValidationError as e:
        passed = False
        logit_results = e.results if hasattr(e, "results") else None
        error_msg = str(e)

    summary = {}
    if logit_results is not None:
        summary = extract_logit_validation_summary(logit_results)

    summary["logit_matching_passed"] = passed
    summary["tol_map"] = {str(k): list(v) for k, v in tol_map.items()}
    summary["divergence_difference_tol"] = div_tol
    if error_msg:
        summary["error_message"] = error_msg

    return passed, summary


# ---------------------------------------------------------------------------
# Inference-only validation (no HF comparison)
# ---------------------------------------------------------------------------

def run_inference_only_validation(model, tokenizer) -> Tuple[bool, Dict[str, Any]]:
    """Lightweight sanity check — no HF golden model required.

    Generates 20 tokens from 3 diverse prompts (factual, ML, code) and
    fails if any output is:
      * empty (model produced nothing)
      * degenerate (every character is the same — stuck in a repeat loop)

    This is the ``--skip-hf-comparison`` mode, useful when you just want
    to confirm the compiled model can produce coherent text without
    downloading a multi-GB HF checkpoint.

    Returns:
        (passed, details) — ``details["prompt_results"]`` has per-prompt
        status and a 100-char sample of the generated text.
    """
    test_prompts = [
        "The capital of France is",
        "In machine learning, neural networks",
        "def fibonacci(n):",
    ]
    passed = True
    prompt_results = []

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]
        generated_ids = input_ids.clone()

        with torch.no_grad():
            for _ in range(20):
                pos = torch.arange(generated_ids.shape[1], dtype=torch.int32).unsqueeze(0)
                outputs = model(generated_ids, position_ids=pos)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                if tokenizer.eos_token_id and next_tok.item() == tokenizer.eos_token_id:
                    break
                generated_ids = torch.cat([generated_ids, next_tok], dim=-1)

        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        new_text = text[len(prompt):]
        result = {"prompt": prompt, "generated": new_text[:100]}

        if len(new_text.strip()) == 0:
            result["status"] = "FAIL - Empty"
            passed = False
        elif all(c == new_text[0] for c in new_text.strip()):
            result["status"] = "FAIL - Repetitive"
            passed = False
        else:
            result["status"] = "OK"
        prompt_results.append(result)

    return passed, {"mode": "inference_only", "prompt_results": prompt_results}
