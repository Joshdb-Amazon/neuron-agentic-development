# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
validator — end-to-end accuracy and performance validation for Neuron-compiled models.

Compares a compiled Neuron model against its HuggingFace golden reference
across three progressive stages:

    1. Smoke test   — can the compiled model load at all?
    2. Accuracy     — do greedy-decoded tokens / logit distributions match
                      the FP32 HuggingFace reference?
    3. Performance  — does TTFT and throughput meet the configured thresholds?

Orchestration lives in ``run.py``; this package is the library it calls.
All functions return structured dicts (never call ``sys.exit``), and every
test run persists a timestamped JSON under ``results/``.

Submodules
----------
patches          Monkey-patches that bridge transformers ↔ NxDI version gaps.
constants        Thresholds, tolerance maps, prompt lists, model-type detection.
accuracy         Token matching, NxDI logit matching v2, inference-only checks.
enhanced_metrics Distribution-level metrics (cosine sim, KL div, top-k overlap).
main             Model creation, HF golden loading, the three test_* entry points,
                 and result persistence.
"""

from .patches import apply_all_patches  # noqa: F401 – side-effect import
from .main import (  # noqa: F401
    create_model,
    import_class,
    load_hf_golden_model,
    load_neuron_config_from_compiled,
    test_model_load,
    test_accuracy,
    save_accuracy_results,
)
from .accuracy import check_accuracy_with_hf_golden  # noqa: F401
