#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI entry point for model validation.

Usage:
    python scripts/validate_model.py --config configs/my_model.json [--mode token|logit|comprehensive]
    python scripts/validate_model.py --config configs/my_model.json --performance-only
    python scripts/validate_model.py --config configs/my_model.json --accuracy-only --skip-hf-comparison

Reads batch_size and seq_len from config's test_parameters (first entry).
CLI --batch-size/--seq-len override config values if provided.

Success criteria: >= 95% greedy token match rate.
Exit code 0 = passed, 1 = failed.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from validator import apply_all_patches, create_model, test_model_load, test_accuracy
from validator.main import test_performance


def main():
    parser = argparse.ArgumentParser(description="Validate a Neuron-compiled model against HuggingFace reference")
    parser.add_argument("--config", required=True, help="Path to validation config JSON")
    parser.add_argument("--mode", choices=["token", "logit", "comprehensive"], default="token")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size from config")
    parser.add_argument("--seq-len", type=int, default=None, help="Override seq_len from config")
    parser.add_argument("--num-tokens", type=int, default=64)
    parser.add_argument("--hf-dtype", default=None)
    parser.add_argument("--skip-hf-comparison", action="store_true")
    parser.add_argument("--accuracy-only", action="store_true", help="Run only accuracy tests")
    parser.add_argument("--performance-only", action="store_true", help="Run only performance tests")
    parser.add_argument("--skip-smoke-test", action="store_true", help="Skip the smoke test")
    parser.add_argument("--continue-on-failure", action="store_true", help="Continue through all test_parameters on failure")
    parser.add_argument("--results-dir", default=None, help="Directory to save results (default: agent_artifacts/results)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    config.setdefault("num_tokens_to_check", args.num_tokens)
    if args.results_dir:
        config["results_dir"] = args.results_dir

    # Build test_parameters list: from config or CLI overrides
    if "test_parameters" in config and config["test_parameters"]:
        test_params_list = config["test_parameters"]
    else:
        test_params_list = [{"batch_size": 1, "seq_len": 2048}]

    # CLI overrides apply to all entries
    if args.batch_size is not None or args.seq_len is not None:
        for tp in test_params_list:
            if args.batch_size is not None:
                tp["batch_size"] = args.batch_size
            if args.seq_len is not None:
                tp["seq_len"] = args.seq_len

    print(f"\nModel: {config.get('model_name', 'unknown')}")
    print(f"HF Model: {config.get('model_path', 'unknown')}")
    print(f"Compiled Model: {config.get('compiled_model_path', 'unknown')}")

    total = passed_count = failed_count = 0

    for i, test_params in enumerate(test_params_list, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Test params {i}/{len(test_params_list)}: "
              f"bs={test_params['batch_size']} seq={test_params['seq_len']}")
        print(f"{'#'*80}")

        # --- Smoke test ---
        if not args.skip_smoke_test and not args.accuracy_only and not args.performance_only:
            total += 1
            print("\n" + "=" * 60)
            print("STAGE 1: Smoke test (model load)")
            print("=" * 60)
            if not test_model_load(config, test_params):
                print("\nFAILED: Model could not load.")
                failed_count += 1
                if not args.continue_on_failure:
                    sys.exit(1)
                continue
            passed_count += 1

        # --- Accuracy ---
        if not args.performance_only:
            total += 1
            print("\n" + "=" * 60)
            print(f"STAGE 2: Accuracy validation (mode={args.mode})")
            print("=" * 60)

            ok, details = test_accuracy(
                config, test_params,
                use_token_matching=args.mode in ("token", "comprehensive"),
                comprehensive=args.mode == "comprehensive",
                hf_dtype_override=args.hf_dtype,
                skip_hf_comparison=args.skip_hf_comparison,
            )

            if "token_matching" in details:
                rate = details["token_matching"]["match_rate"]
                print(f"Token match rate: {rate*100:.2f}%  (threshold: 95%)")
            if "error_details" in details:
                print(f"Error: {details['error_details'].get('error_message', 'unknown')}")

            if ok:
                passed_count += 1
            else:
                failed_count += 1
                if not args.continue_on_failure:
                    print(f"\nRESULT: FAILED")
                    sys.exit(1)

        # --- Performance ---
        if not args.accuracy_only:
            total += 1
            print("\n" + "=" * 60)
            print("STAGE 3: Performance validation")
            print("=" * 60)

            if "ttft_threshold" not in test_params or "throughput_threshold" not in test_params:
                print("Skipping performance test — no ttft_threshold/throughput_threshold in test_parameters")
                total -= 1
            else:
                ok, report = test_performance(config, test_params)
                if ok:
                    passed_count += 1
                else:
                    failed_count += 1
                    if not args.continue_on_failure:
                        print(f"\nRESULT: FAILED")
                        sys.exit(1)

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: {config.get('model_name', 'unknown')} — "
          f"{passed_count}/{total} passed, {failed_count} failed")
    print(f"{'='*80}")
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
