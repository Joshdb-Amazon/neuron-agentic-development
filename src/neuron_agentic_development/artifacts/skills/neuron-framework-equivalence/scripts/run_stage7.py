#!/usr/bin/env python3
"""
Stage 7: Downstream Task Evaluation.

Wraps neuron_bench to run lm_eval benchmarks on both HF and Neuron models,
then compares scores with per-task tolerance bands.

Usage:
    python3 scripts/run_stage7.py \
        --bench-config ${EXP_DIR}/bench_config.yaml \
        --output-dir ${EXP_DIR}/results/stage7 \
        --tolerance 0.02

    Or run neuron_bench directly:
    python -m neuron_bench.run --config bench_config.yaml --hf-baseline
"""

import argparse
import json
import sys
import os


def main():
    parser = argparse.ArgumentParser(description="Stage 7: Downstream Task Evaluation")
    parser.add_argument("--bench-config", required=True, help="Path to neuron_bench YAML config")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tolerance", type=float, default=0.02, help="Max regression in percentage points")
    parser.add_argument("--hf-baseline", action="store_true", default=True,
                        help="Run HF baseline for comparison (default: True)")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Override tasks to run (default: from config)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit samples per task (for faster testing)")
    args = parser.parse_args()

    # Delegate to neuron_bench
    from neuron_bench.config import parse_config
    from neuron_bench.run import (
        run_lm_eval_scenarios,
        run_hf_lm_eval_scenarios,
        run_longbench_scenarios,
    )
    from neuron_bench.compare import compare_results, print_comparison_report
    from neuron_bench.model_loader import load_model

    config = parse_config(args.bench_config)

    output_dir = args.output_dir or f"results/stage7"
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    # HF baseline
    hf_results = None
    if args.hf_baseline and config.lm_eval:
        print("=" * 60)
        print("Phase 1: HF baseline (FP32 CPU)")
        print("=" * 60)
        hf_results = run_hf_lm_eval_scenarios(
            config.model["model_path"], config.lm_eval, output_dir,
        )
        all_results["lm_eval_hf"] = hf_results

    # Neuron model
    neuron_results = None
    if config.lm_eval:
        print("=" * 60)
        print("Phase 2: Neuron model")
        print("=" * 60)
        model, tokenizer, generation_config = load_model(config.model)
        neuron_results = run_lm_eval_scenarios(
            model, tokenizer, generation_config, config.lm_eval, output_dir,
        )
        all_results["lm_eval"] = neuron_results

    # Compare
    passed = True
    if hf_results and neuron_results:
        print("=" * 60)
        print("Phase 3: Comparing HF vs Neuron")
        print("=" * 60)
        comparison = compare_results(hf_results, neuron_results, tolerances=config.tolerance)
        all_results["comparison"] = comparison
        print_comparison_report(comparison)
        passed = comparison["overall_pass"]

    # Save
    summary_file = os.path.join(output_dir, "stage7_summary.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"  Stage 7: {'PASS' if passed else 'FAIL'}")
    print(f"  Results saved to {output_dir}")
    print(f"{'=' * 60}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
