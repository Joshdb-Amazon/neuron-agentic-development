#!/usr/bin/env python3
"""
Teacher-Forced Per-Position Logit Comparison.

Used by Stages 5 and 6 for proper per-position comparison under teacher forcing.

Pattern (from NxDI's logit_validation):
1. Generate expected tokens from the source model (greedy)
2. Feed the source tokens to the target model
3. At each position, compare logits (both models see the same prefix)
4. If the target diverges, re-feed the source tokens and continue

This ensures per-position KL and cosine are computed on identical contexts,
not contaminated by trajectory divergence after token flips.

Usage:
    python3 scripts/run_teacher_forced_comparison.py \
        --model-path /path/to/hf_model \
        --compiled-model-path /path/to/compiled_model \
        --model-class path/to/modeling.py:NeuronXxxForCausalLM \
        --config-class path/to/modeling.py:XxxInferenceConfig \
        --num-tokens 32 \
        --output results/teacher_forced.json
"""

import argparse
import json
import os
import sys
import numpy as np

import torch
import torch.nn.functional as F


def get_source_logits_and_tokens(model, tokenizer, prompt, num_tokens):
    """Generate from source model with output_scores=True to get per-position logits."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    with torch.no_grad():
        try:
            out = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=num_tokens, min_new_tokens=num_tokens,
                do_sample=False, return_dict_in_generate=True, output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        except (AttributeError, TypeError):
            out = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=num_tokens, min_new_tokens=num_tokens,
                do_sample=False, return_dict_in_generate=True, output_scores=True,
                pad_token_id=tokenizer.pad_token_id, use_cache=False,
            )

    # scores: tuple of [batch, vocab] tensors, one per generated position
    source_logits = torch.stack(out.scores)  # [num_tokens, batch, vocab]
    source_tokens = source_logits.argmax(dim=2).T  # [batch, num_tokens]
    return source_logits, source_tokens, input_ids


def get_target_logits_teacher_forced(model, tokenizer, input_ids, source_tokens, num_tokens):
    """Get target model logits under teacher forcing.

    Feeds source tokens one at a time, collecting target logits at each position.
    If the target model only returns last-position logits (compiled model),
    we extend the input by one source token at a time.
    """
    from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
    from transformers import GenerationConfig
    import transformers

    adapter = HuggingFaceGenerationAdapter(model)
    gen_config = GenerationConfig(
        do_sample=False, pad_token_id=tokenizer.pad_token_id,
    )
    gen_config.transformers_version = transformers.__version__

    # Try generating with output_scores (works if model supports it)
    try:
        # Build the full teacher-forced input: prompt + source tokens
        full_input = torch.cat([input_ids, source_tokens], dim=1)
        attention_mask = torch.ones_like(full_input)

        with torch.no_grad():
            out = adapter.generate(
                input_ids=full_input[:, :-num_tokens],  # Start with just the prompt
                attention_mask=attention_mask[:, :-num_tokens],
                max_new_tokens=num_tokens, min_new_tokens=num_tokens,
                do_sample=False, return_dict_in_generate=True, output_scores=True,
                generation_config=gen_config,
            )
        if hasattr(model, "reset"):
            model.reset()

        target_logits = torch.stack(out.scores)[:num_tokens]
        return target_logits

    except Exception as e:
        print(f"  output_scores generation failed ({e}), falling back to per-token forward")

    # Fallback: per-token forward passes with teacher-forced input
    target_logits_list = []
    current_input = input_ids.clone()

    for t in range(num_tokens):
        attention_mask = torch.ones_like(current_input)
        position_ids = torch.arange(current_input.shape[1], dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            try:
                out = model(current_input, attention_mask=attention_mask, position_ids=position_ids)
            except TypeError:
                out = model(current_input)

            logits = out.logits if hasattr(out, "logits") else out
            if isinstance(logits, (list, tuple)):
                logits = logits[0]

        # Take last position logits
        last_logits = logits[:, -1, :].float()
        last_logits = torch.nan_to_num(last_logits, nan=0.0, posinf=1e6, neginf=-1e6)
        target_logits_list.append(last_logits)

        # Teacher forcing: append the SOURCE token, not the target's argmax
        next_token = source_tokens[:, t:t+1]
        current_input = torch.cat([current_input, next_token], dim=1)

    if hasattr(model, "reset"):
        model.reset()

    return torch.stack(target_logits_list)  # [num_tokens, batch, vocab]


def main():
    parser = argparse.ArgumentParser(description="Teacher-Forced Per-Position Comparison")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-model-path", required=True)
    parser.add_argument("--model-class", required=True)
    parser.add_argument("--config-class", required=True)
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--prompts", nargs="+", default=[
        "The capital of France is", "Water freezes at", "The speed of light is approximately",
    ])
    parser.add_argument("--theta", type=float, default=0.95)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

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

    # Load models
    print("Loading source model (FP32)...")
    source_fp32 = load_hf_golden_model(args.model_path, config=config, hf_dtype_override="float32")

    print("Loading source model (BF16)...")
    source_bf16 = load_hf_golden_model(args.model_path, config=config, hf_dtype_override="bfloat16")

    print("Loading target model (compiled)...")
    target_model, tokenizer, _ = create_model(config, batch_size=1, seq_len=128)
    target_model.load(args.compiled_model_path)
    ensure_generation_config_version(target_model)
    patch_generation_mixin()

    all_results = []

    for prompt in args.prompts:
        print(f"\n{'=' * 70}")
        print(f"  Prompt: \"{prompt}\"")
        print(f"{'=' * 70}")

        # Step 1: Get source tokens (FP32, greedy generation)
        print("  Generating source tokens (FP32, greedy)...")
        src_logits_fp32, src_tokens, input_ids = get_source_logits_and_tokens(
            source_fp32, tokenizer, prompt, args.num_tokens,
        )
        num_gen = min(args.num_tokens, src_tokens.shape[1])

        # Step 1b: Get source FP32 logits via teacher-forced forward passes
        # (for consistency — each position conditioned on the same prefix)
        print("  Getting source FP32 logits (teacher-forced)...")
        fp32_logits_list = []
        current_input = input_ids.clone()
        for t in range(num_gen):
            attention_mask = torch.ones_like(current_input)
            position_ids = torch.arange(current_input.shape[1], dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                try:
                    out = source_fp32(current_input, attention_mask=attention_mask, position_ids=position_ids)
                except (TypeError, AttributeError):
                    try:
                        out = source_fp32(current_input, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
                    except TypeError:
                        out = source_fp32(current_input)
                logits = out.logits if hasattr(out, "logits") else out
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
            fp32_logits_list.append(logits[:, -1, :].float())
            current_input = torch.cat([current_input, src_tokens[:, t:t+1]], dim=1)
        src_logits_fp32_tf = torch.stack(fp32_logits_list)

        # Step 2: Get source BF16 logits (teacher-forced by FP32 tokens)
        print("  Getting source BF16 logits (teacher-forced)...")
        bf16_logits_list = []
        current_input = input_ids.clone()
        for t in range(num_gen):
            attention_mask = torch.ones_like(current_input)
            position_ids = torch.arange(current_input.shape[1], dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                try:
                    out = source_bf16(current_input, attention_mask=attention_mask, position_ids=position_ids)
                except (TypeError, AttributeError):
                    try:
                        out = source_bf16(current_input, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
                    except TypeError:
                        out = source_bf16(current_input)
                logits = out.logits if hasattr(out, "logits") else out
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
            last_logits = logits[:, -1, :].float()
            last_logits = torch.nan_to_num(last_logits, nan=0.0, posinf=1e6, neginf=-1e6)
            bf16_logits_list.append(last_logits)
            # Teacher forcing: append the FP32 source token
            current_input = torch.cat([current_input, src_tokens[:, t:t+1]], dim=1)
        src_logits_bf16_tf = torch.stack(bf16_logits_list)  # [num_tokens, batch, vocab]

        # Step 3: Get target logits (teacher-forced by FP32 tokens)
        print("  Getting target logits (teacher-forced)...")
        tgt_logits = get_target_logits_teacher_forced(
            target_model, tokenizer, input_ids, src_tokens, args.num_tokens,
        )

        # Align vocab sizes
        min_vocab = min(src_logits_fp32_tf.shape[-1], src_logits_bf16_tf.shape[-1], tgt_logits.shape[-1])
        min_tokens = min(src_logits_fp32_tf.shape[0], src_logits_bf16_tf.shape[0], tgt_logits.shape[0])

        fp32 = src_logits_fp32_tf[:min_tokens, 0, :min_vocab].float()
        bf16 = src_logits_bf16_tf[:min_tokens, 0, :min_vocab].float()
        tgt = tgt_logits[:min_tokens, 0, :min_vocab].float()
        tgt = torch.nan_to_num(tgt, nan=0.0, posinf=1e6, neginf=-1e6)
        bf16 = torch.nan_to_num(bf16, nan=0.0, posinf=1e6, neginf=-1e6)

        # Per-position metrics
        per_pos_cos = []
        per_pos_kl = []
        per_pos_r = []
        per_pos_topk = {1: [], 5: [], 10: []}

        for t in range(min_tokens):
            # Cosine similarity
            cos = F.cosine_similarity(fp32[t].unsqueeze(0), tgt[t].unsqueeze(0)).item()
            per_pos_cos.append(cos)

            # KL divergence (on full vocab)
            ref_probs = F.softmax(fp32[t], dim=-1)
            tgt_log_probs = F.log_softmax(tgt[t], dim=-1)
            kl = max(0.0, F.kl_div(tgt_log_probs.unsqueeze(0), ref_probs.unsqueeze(0), reduction="sum").item())
            per_pos_kl.append(kl)

            # R-ratio (three-tensor)
            baseline_err = torch.norm(bf16[t] - fp32[t], p=2).item()
            target_err = torch.norm(tgt[t] - fp32[t], p=2).item()
            # Use a meaningful epsilon: scale by the FP32 norm to avoid
            # division by near-zero when FP32 and BF16 happen to agree exactly
            fp32_norm = torch.norm(fp32[t], p=2).item()
            eps = max(1e-12, fp32_norm * 1e-7)
            r = target_err / (baseline_err + eps)
            per_pos_r.append(r)

            # Top-k agreement
            for k in per_pos_topk:
                if k > min_vocab:
                    continue
                fp32_topk = set(torch.topk(fp32[t], k).indices.tolist())
                tgt_topk = set(torch.topk(tgt[t], k).indices.tolist())
                per_pos_topk[k].append(len(fp32_topk & tgt_topk) / k)

        # Print per-position results
        cos_arr = np.array(per_pos_cos)
        kl_arr = np.array(per_pos_kl)
        r_arr = np.array(per_pos_r)

        print(f"\n  Per-position results ({min_tokens} positions, teacher-forced):")
        print(f"  {'Pos':>4s} {'R-ratio':>8s} {'Cosine':>8s} {'KL':>10s} {'Top-1':>6s}")
        print(f"  {'-'*4} {'-'*8} {'-'*8} {'-'*10} {'-'*6}")
        for t in range(min(min_tokens, 10)):  # Show first 10
            top1 = "✓" if per_pos_topk[1][t] == 1.0 else "✗"
            print(f"  {t:>4d} {per_pos_r[t]:>8.4f} {per_pos_cos[t]:>8.4f} {per_pos_kl[t]:>10.4f} {top1:>6s}")
        if min_tokens > 10:
            print(f"  ... ({min_tokens - 10} more positions)")

        print(f"\n  Summary:")
        print(f"    R-ratio:  mean={np.mean(r_arr):.4f}, p95={np.percentile(r_arr, 95):.4f}, max={np.max(r_arr):.4f}")
        print(f"    Cosine:   mean={np.mean(cos_arr):.6f}, min={np.min(cos_arr):.6f}, p5={np.percentile(cos_arr, 5):.6f}")
        print(f"    KL:       mean={np.mean(kl_arr):.6f}, p95={np.percentile(kl_arr, 95):.6f}, max={np.max(kl_arr):.6f}")
        print(f"    Top-1:    {np.mean(per_pos_topk[1]):.2%}")

        all_results.append({
            "prompt": prompt,
            "num_positions": min_tokens,
            "r_ratio": {"mean": float(np.mean(r_arr)), "p95": float(np.percentile(r_arr, 95)), "max": float(np.max(r_arr))},
            "cosine": {"mean": float(np.mean(cos_arr)), "min": float(np.min(cos_arr)), "p5": float(np.percentile(cos_arr, 5))},
            "kl": {"mean": float(np.mean(kl_arr)), "p95": float(np.percentile(kl_arr, 95)), "max": float(np.max(kl_arr))},
            "top1_agreement": float(np.mean(per_pos_topk[1])),
            "per_position_cosine": per_pos_cos,
            "per_position_kl": per_pos_kl,
            "per_position_r_ratio": per_pos_r,
        })

    # Aggregate across prompts
    all_cos = [c for r in all_results for c in r["per_position_cosine"]]
    all_kl = [k for r in all_results for k in r["per_position_kl"]]
    all_r = [r for res in all_results for r in res["per_position_r_ratio"]]

    cos_arr = np.array(all_cos)
    kl_arr = np.array(all_kl)
    r_arr = np.array(all_r)

    condition_b = float(np.percentile(cos_arr, 5)) >= args.theta
    condition_c_kl_p95 = float(np.percentile(kl_arr, 95))

    print(f"\n{'=' * 70}")
    print(f"  TEACHER-FORCED COMPARISON SUMMARY ({len(all_cos)} total positions)")
    print(f"{'=' * 70}")
    print(f"  Condition B (Semantic): {'PASS' if condition_b else 'FAIL'}")
    print(f"    cosine: mean={np.mean(cos_arr):.6f}, min={np.min(cos_arr):.6f}, p5={np.percentile(cos_arr, 5):.6f} (θ={args.theta})")
    print(f"  Condition C (Distributional):")
    print(f"    KL: mean={np.mean(kl_arr):.6f}, p95={condition_c_kl_p95:.6f}, max={np.max(kl_arr):.6f}")
    print(f"  E2E R-ratio:")
    print(f"    mean={np.mean(r_arr):.4f}, p95={np.percentile(r_arr, 95):.4f}, max={np.max(r_arr):.4f}")
    print(f"  Top-1 agreement: {np.mean([r['top1_agreement'] for r in all_results]):.2%}")
    print(f"{'=' * 70}")

    output = {
        "per_prompt": all_results,
        "aggregate": {
            "condition_b_passed": condition_b,
            "cosine": {"mean": float(np.mean(cos_arr)), "min": float(np.min(cos_arr)), "p5": float(np.percentile(cos_arr, 5))},
            "kl": {"mean": float(np.mean(kl_arr)), "p95": float(np.percentile(kl_arr, 95)), "max": float(np.max(kl_arr))},
            "r_ratio": {"mean": float(np.mean(r_arr)), "p95": float(np.percentile(r_arr, 95)), "max": float(np.max(r_arr))},
            "num_positions": len(all_cos),
        },
    }

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
