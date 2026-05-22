# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Custom performance benchmarking — fallback for when NxDI's
``benchmark_sampling`` doesn't work.

NxDI ships ``benchmark_sampling()`` which measures TTFT and throughput
using its own internal harness.  But it can fail on models with
non-standard signatures or when the NxDI version is too old.  When that
happens, ``test_performance`` in ``main.py`` falls back to the functions
here.

``run_custom_benchmark`` does a manual measurement:

  1. **Warmup** — 3 forward passes (not timed) to fill caches and let
     the Neuron runtime settle.
  2. **Context encoding** — time a single forward pass (prompt only)
     over ``num_iterations`` runs.  This gives TTFT.
  3. **Token generation** — time a full 64-token greedy-decode loop
     over ``num_iterations`` runs.  Per-token latency → throughput.

Results are returned in the same dict shape as ``benchmark_sampling``
(keys ``context_encoding_model`` and ``token_generation_model``, each
with ``latency_ms_p50/p90/p95/p99/p100/avg`` and ``throughput``), so
the caller doesn't need to know which path was taken.

``print_benchmark_report`` pretty-prints any benchmark dict to stdout.
"""

import time
import torch
from typing import Dict, Optional


def run_custom_benchmark(
    model, tokenizer, batch_size: int, seq_len: int, num_iterations: int = 20,
) -> dict:
    """Measure TTFT and throughput with a manual token-by-token loop.

    Used as a fallback when ``neuronx_distributed_inference.utils.benchmark
    .benchmark_sampling`` raises (TypeError, AttributeError, RuntimeError).

    Procedure:
      * 3 warmup forward passes (untimed).
      * ``num_iterations`` context-encoding passes (single forward, timed).
      * ``num_iterations`` full 64-token decode loops (timed per-token).

    Args:
        model:          Loaded Neuron model.
        tokenizer:      Tokenizer (used to encode the fixed prompt).
        batch_size:     Batch size the model was compiled with.
        seq_len:        Sequence length the model was compiled with (unused
                        directly, but documents the compilation context).
        num_iterations: How many timed runs for each phase (default 20).

    Returns:
        Dict with ``context_encoding_model`` and ``token_generation_model``
        sub-dicts, each containing ``latency_ms_p50/p90/p95/p99/p100/avg``
        and (for token gen) ``throughput`` in tokens/s.
    """
    import numpy as np

    prompt = "Hello, I am a language model"
    num_new_tokens = 64
    inputs = tokenizer([prompt] * batch_size, padding=True, return_tensors="pt")
    input_ids = inputs.input_ids

    # Warmup
    for _ in range(3):
        ids = input_ids.clone()
        pos = torch.arange(ids.shape[1]).unsqueeze(0).expand(batch_size, -1)
        with torch.no_grad():
            model(ids, position_ids=pos)

    # Context encoding
    context_times = []
    for _ in range(num_iterations):
        ids = input_ids.clone()
        pos = torch.arange(ids.shape[1]).unsqueeze(0).expand(batch_size, -1)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(ids, position_ids=pos)
        context_times.append((time.perf_counter() - t0) * 1000)

    # Token generation
    token_times = []
    for _ in range(num_iterations):
        ids = input_ids.clone()
        t0 = time.perf_counter()
        for _ in range(num_new_tokens):
            pos = torch.arange(ids.shape[1]).unsqueeze(0).expand(ids.shape[0], -1)
            with torch.no_grad():
                out = model(ids, position_ids=pos)
            logits = out.logits if hasattr(out, "logits") else (
                out[0] if isinstance(out, tuple) else out
            )
            ids = torch.cat([ids, torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)], dim=-1)
        token_times.append((time.perf_counter() - t0) * 1000 / num_new_tokens)

    def _stats(times):
        return {
            "latency_ms_p50": float(np.percentile(times, 50)),
            "latency_ms_p90": float(np.percentile(times, 90)),
            "latency_ms_p95": float(np.percentile(times, 95)),
            "latency_ms_p99": float(np.percentile(times, 99)),
            "latency_ms_p100": float(np.max(times)),
            "latency_ms_avg": float(np.mean(times)),
        }

    ctx = _stats(context_times)
    tok = _stats(token_times)
    tok["throughput"] = (1000 / tok["latency_ms_avg"]) * batch_size if tok["latency_ms_avg"] > 0 else 0
    return {"context_encoding_model": ctx, "token_generation_model": tok}


def print_benchmark_report(report: dict):
    """Pretty-print a benchmark report dict to stdout.

    Handles both NxDI's ``benchmark_sampling`` output and our own
    ``run_custom_benchmark`` output (same key structure).  Prints
    latency percentiles and throughput for each sub-component
    (context encoding, token generation, e2e if present).
    """
    print("\n" + "=" * 80 + "\nBENCHMARK RESULTS\n" + "=" * 80)
    for name, metrics in report.items():
        if metrics is None:
            continue
        print(f"\n{name.upper().replace('_', ' ')}:")
        if "latency_ms_p50" in metrics:
            for p in ["p50", "p90", "p95", "p99", "p100", "avg"]:
                k = f"latency_ms_{p}"
                if k in metrics:
                    print(f"  {p.upper():>5}: {metrics[k]:.2f} ms")
        if "throughput" in metrics:
            print(f"  Throughput: {metrics['throughput']:.2f} tokens/s")
    print("=" * 80)
