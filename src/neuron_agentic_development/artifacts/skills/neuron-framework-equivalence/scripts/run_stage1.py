#!/usr/bin/env python3
"""
Stage 1: Smoke Test Runner.

Delegates to model_validation's check_accuracy_with_hf_golden and
compute_enhanced_metrics. Requires model_validation on PYTHONPATH.

Usage:
    python3 scripts/run_stage1.py \
        --model-path /path/to/hf_model \
        --compiled-model-path /path/to/compiled_model \
        --model-class path/to/modeling.py:NeuronXxxForCausalLM \
        --config-class path/to/modeling.py:XxxInferenceConfig
"""

import argparse
import json
import sys
import os


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Smoke Test")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-model-path", required=True)
    parser.add_argument("--model-class", required=True)
    parser.add_argument("--config-class", required=True)
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    # Build a config dict compatible with model_validation
    config = {
        "model_name": os.path.basename(args.model_path),
        "model_path": args.model_path,
        "compiled_model_path": args.compiled_model_path,
        "model_class": args.model_class,
        "config_class": args.config_class,
        "num_tokens_to_check": args.num_tokens,
        "test_parameters": [{"batch_size": 1, "seq_len": 128}],
    }

    from validator.main import (
        create_model, load_hf_golden_model,
    )
    from validator.patches import (
        ensure_generation_config_version, ensure_pad_token, patch_generation_mixin,
    )
    from validator.accuracy import check_accuracy_with_hf_golden
    from validator.enhanced_metrics import compute_enhanced_metrics
    from transformers import AutoTokenizer, GenerationConfig
    import transformers

    # Load port model
    print("Loading compiled Neuron model...")
    model, tokenizer, gen_config = create_model(config, batch_size=1, seq_len=128)
    model.load(args.compiled_model_path)
    ensure_generation_config_version(model)
    patch_generation_mixin()

    gen_config = GenerationConfig(do_sample=False, top_k=1, pad_token_id=tokenizer.pad_token_id)
    gen_config.transformers_version = transformers.__version__
    ensure_pad_token(gen_config, tokenizer)

    # Load HF reference (FP32)
    print("Loading HF reference (FP32)...")
    hf_model = load_hf_golden_model(args.model_path, config=config)

    # Token matching
    print("\n" + "=" * 70)
    print("  Step 1.1: Token Matching (10 prompts, greedy decoding)")
    print("=" * 70)
    token_passed, token_details = check_accuracy_with_hf_golden(
        neuron_model=model, hf_model=hf_model, tokenizer=tokenizer,
        generation_config=gen_config, num_tokens_to_check=args.num_tokens,
    )

    rate = token_details['match_rate']
    total = token_details.get('total_tokens_compared', 0)
    matching = token_details.get('matching_tokens', 0)
    print(f"\n  Overall: {rate:.2%} ({matching}/{total} tokens)")
    print(f"  {'Prompt':<55s} {'Match':>6s} {'Diverge':>10s}")
    print(f"  {'-'*55} {'-'*6} {'-'*10}")
    for pp in token_details.get("per_prompt_results", []):
        div = pp.get("first_divergence_idx")
        div_str = f"token {div}" if div is not None else "—"
        print(f"  {pp['prompt'][:55]:<55s} {pp['match_rate']:>5.0%} {div_str:>10s}")

    # Enhanced metrics
    print("\n" + "=" * 70)
    print("  Step 1.2: Enhanced Distribution Metrics (FP32 vs BF16)")
    print("=" * 70)
    enhanced = compute_enhanced_metrics(neuron_model=model, hf_model=hf_model, tokenizer=tokenizer)

    if enhanced.get("enhanced_metrics_available"):
        print(f"\n  Cosine similarity:  mean={enhanced.get('logit_cosine_similarity_mean', 0):.6f}  "
              f"min={enhanced.get('logit_cosine_similarity_min', 0):.6f}  "
              f"std={enhanced.get('logit_cosine_similarity_std', 0):.6f}")
        print(f"  KL divergence:      {enhanced.get('kl_divergence', 0):.6f}")
        print(f"  Top-5 agreement:    {enhanced.get('top5_agreement', 0):.2%}")
        print(f"  Relative L2 error:  mean={enhanced.get('relative_l2_error_mean', 0):.6f}  "
              f"max={enhanced.get('relative_l2_error_max', 0):.6f}")
        if enhanced.get("mean_prob_of_hf_token"):
            print(f"  P(source token):    {enhanced['mean_prob_of_hf_token']:.6f}")
        for k in [5, 50, 1000]:
            key = f"topk{k}_normwise_error_mean"
            if key in enhanced:
                print(f"  Top-{k:<4d} norm err:  mean={enhanced[key]:.6f}  "
                      f"max={enhanced.get(f'topk{k}_normwise_error_max', 0):.6f}")
        print(f"\n  Note: These metrics compare FP32 vs BF16 (NOT precision-normalized).")
        print(f"  Use Stage 2 R-ratio for precision-normalized comparison.")
    else:
        print(f"  Enhanced metrics not available: {enhanced.get('enhanced_metrics_error', 'unknown')}")

    result = {"token_matching": token_details, "enhanced_metrics": enhanced}
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    print(f"\n{'=' * 70}")
    print(f"  STAGE 1 VERDICT: {'PASS' if token_passed else 'FAIL'}")
    print(f"  Token match rate: {rate:.2%}")
    if not token_passed:
        print(f"  Action: Proceed to Stage 2 for component-level localization")
    else:
        print(f"  Action: Proceed to Stage 2 for component-level verification")
    print(f"{'=' * 70}")
    return 0 if token_passed else 1


if __name__ == "__main__":
    sys.exit(main())
