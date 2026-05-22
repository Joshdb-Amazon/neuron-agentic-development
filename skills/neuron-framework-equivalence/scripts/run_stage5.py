#!/usr/bin/env python3
"""
Stage 5: E2E Three-Tensor Comparison.

Compares last-position logits from HF FP32, HF BF16, and compiled Neuron model.
Requires model_validation on PYTHONPATH for model loading.

Usage:
    python3 scripts/run_stage5.py \
        --model-path /path/to/hf_model \
        --compiled-model-path /path/to/compiled_model \
        --model-class path/to/modeling.py:NeuronXxxForCausalLM \
        --config-class path/to/modeling.py:XxxInferenceConfig \
        --prompts "The capital of France is" "Water freezes at"
"""

import argparse
import json
import sys
import os

import torch
import torch.nn.functional as F
import numpy as np


def get_last_logits(model, input_ids, attention_mask=None):
    """Get last-position logits, handling both HF and NxDI models."""
    seq_len = input_ids.shape[1]
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        try:
            out = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        except (TypeError, AttributeError) as e:
            if "DynamicCache" in str(e) or "get_usable_length" in str(e):
                out = model(input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
            else:
                try:
                    out = model(input_ids, attention_mask=attention_mask)
                except TypeError:
                    out = model(input_ids)

        logits = out.logits if hasattr(out, "logits") else out
        if isinstance(logits, (list, tuple)):
            logits = logits[0]

    last = logits[:, -1, :].float()
    return torch.nan_to_num(last, nan=0.0, posinf=1e6, neginf=-1e6)


def main():
    parser = argparse.ArgumentParser(description="Stage 5: E2E Three-Tensor")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-model-path", required=True)
    parser.add_argument("--model-class", required=True)
    parser.add_argument("--config-class", required=True)
    parser.add_argument("--prompts", nargs="+", default=[
        "The capital of France is", "Water freezes at", "The speed of light is approximately",
    ])
    parser.add_argument("--tau-r", type=float, default=1.2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # Import tensor_compare from same scripts/ directory
    sys.path.insert(0, os.path.dirname(__file__))
    from tensor_compare import compare_3tensors

    from validator.main import create_model, load_hf_golden_model
    from validator.patches import ensure_generation_config_version, patch_generation_mixin
    from transformers import AutoTokenizer

    config = {
        "model_name": os.path.basename(args.model_path),
        "model_path": args.model_path,
        "compiled_model_path": args.compiled_model_path,
        "model_class": args.model_class,
        "config_class": args.config_class,
        "test_parameters": [{"batch_size": 1, "seq_len": 128}],
    }

    # Load all three models
    print("Loading HF FP32...")
    fp32_model = load_hf_golden_model(args.model_path, config=config, hf_dtype_override="float32")

    print("Loading HF BF16...")
    bf16_model = load_hf_golden_model(args.model_path, config=config, hf_dtype_override="bfloat16")

    print("Loading compiled Neuron model...")
    port_model, tokenizer, _ = create_model(config, batch_size=1, seq_len=128)
    port_model.load(args.compiled_model_path)
    ensure_generation_config_version(port_model)
    patch_generation_mixin()

    # Compare
    results = []
    all_r = []

    for prompt in args.prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        fp32_logits = get_last_logits(fp32_model, inputs.input_ids, inputs.attention_mask)
        bf16_logits = get_last_logits(bf16_model, inputs.input_ids, inputs.attention_mask)
        port_logits = get_last_logits(port_model, inputs.input_ids, inputs.attention_mask)

        min_v = min(fp32_logits.shape[-1], bf16_logits.shape[-1], port_logits.shape[-1])
        metrics = compare_3tensors(
            fp32_logits[0, :min_v].numpy(),
            bf16_logits[0, :min_v].numpy(),
            port_logits[0, :min_v].numpy(),
        )
        r = metrics["r_ratio"]
        all_r.append(r)

        # Top-k agreement
        topk = {}
        for k in [1, 5, 10, 50, 100]:
            if k > min_v:
                continue
            fp32_topk = set(torch.topk(fp32_logits[0, :min_v], k).indices.tolist())
            port_topk = set(torch.topk(port_logits[0, :min_v], k).indices.tolist())
            topk[k] = len(fp32_topk & port_topk) / k

        print(f"\n  \"{prompt[:50]}\" → R={r:.4f}, top-1={'match' if topk.get(1,0)==1 else 'MISMATCH'}")
        results.append({"prompt": prompt, "r_ratio": r, "top_k": topk})

    r_arr = np.array(all_r)
    p95 = float(np.percentile(r_arr, 95)) if len(r_arr) > 1 else float(r_arr[0])
    passed = p95 < args.tau_r

    print(f"\n{'=' * 60}")
    print(f"  Stage 5: {'PASS' if passed else 'FAIL'}")
    print(f"  R-ratio: mean={np.mean(r_arr):.4f}, p95={p95:.4f}, max={np.max(r_arr):.4f}")
    print(f"{'=' * 60}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"results": results, "r_ratio_p95": p95, "passed": passed}, f, indent=2, default=str)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
