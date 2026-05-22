# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Thresholds, tolerance maps, validation prompts, and model-type detection.

This module is pure data + one small helper (``get_logit_thresholds``).
Nothing here imports heavy dependencies — it's safe to import anywhere.

Key values
----------
TOKEN_MATCH_THRESHOLD       Minimum overall token-match rate (across all 10
                            prompts) for the token-matching accuracy test to
                            pass.  Currently 0.95 (95%).

LOGIT_TOL_MAP_*             Per-top-k error tolerance maps fed to NxDI's
                            ``check_accuracy_logits_v2``.  Three profiles:
                            standard (dense transformers), MoE (expert-routed
                            models — looser because routing adds variance),
                            and encoder (BERT-family — tighter because they're
                            smaller and more deterministic).

DEFAULT_VALIDATION_PROMPTS  10 short factual prompts used by the multi-prompt
                            token-matching test.  Chosen to be deterministic
                            under greedy decoding and easy to verify by eye.
"""

import torch
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Global seed for reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Token matching
# ---------------------------------------------------------------------------
TOKEN_MATCH_THRESHOLD = 0.95

# ---------------------------------------------------------------------------
# Logit matching thresholds by model type
# ---------------------------------------------------------------------------
# Each tol_map entry: top_k -> (min_error_threshold, max_error_threshold)

LOGIT_TOL_MAP_STANDARD = {
    None: (1e-5, 0.40), 1000: (1e-5, 0.35), 50: (1e-5, 0.30), 5: (1e-5, 0.25),
}
LOGIT_DIVERGENCE_TOL_STANDARD = 1.5

LOGIT_TOL_MAP_MOE = {
    None: (1e-5, 0.50), 1000: (1e-5, 0.45), 50: (1e-5, 0.40), 5: (1e-5, 0.35),
}
LOGIT_DIVERGENCE_TOL_MOE = 2.0

LOGIT_TOL_MAP_ENCODER = {
    None: (1e-5, 0.20), 1000: (1e-5, 0.15), 50: (1e-5, 0.10), 5: (1e-5, 0.05),
}
LOGIT_DIVERGENCE_TOL_ENCODER = 0.5

# ---------------------------------------------------------------------------
# Model-type keyword lists
# ---------------------------------------------------------------------------
MOE_KEYWORDS = ['mixtral', 'moe', 'deepseek_v3', 'qwen3_coder_30b', 'phi3_5_moe']
ENCODER_KEYWORDS = ['bert', 'modernbert', 't5', 'xlnet', 'codet5']

# Performance tolerances
PERFORMANCE_TOLERANCE = 1.1   # TTFT may exceed threshold by up to 10%
THROUGHPUT_TOLERANCE = 0.9    # Throughput may fall 10% below threshold

# ---------------------------------------------------------------------------
# Default validation prompts (factual, deterministic)
# ---------------------------------------------------------------------------
DEFAULT_VALIDATION_PROMPTS = [
    "The capital of France is",
    "The color of the sky is",
    "Water freezes at",
    "The largest planet in our solar system is",
    "The speed of light is approximately",
    "The chemical symbol for gold is",
    "The square root of 144 is",
    "The first president of the United States was",
    "The boiling point of water is",
    "The number of continents on Earth is",
]


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def get_logit_thresholds(model_name: str) -> Tuple[dict, float]:
    """Pick the right ``(tol_map, divergence_difference_tol)`` for a model.

    Scans ``model_name`` (case-insensitive) for keywords that indicate
    MoE or encoder architectures.  Falls back to the standard dense-
    transformer profile.

    Returns:
        tol_map: ``{top_k: (min_err, max_err)}`` dict for
                 ``check_accuracy_logits_v2``.
        divergence_difference_tol: scalar tolerance for the overall
                 divergence-difference metric.
    """
    name_lower = model_name.lower()
    for kw in MOE_KEYWORDS:
        if kw in name_lower:
            return LOGIT_TOL_MAP_MOE, LOGIT_DIVERGENCE_TOL_MOE
    for kw in ENCODER_KEYWORDS:
        if kw in name_lower:
            return LOGIT_TOL_MAP_ENCODER, LOGIT_DIVERGENCE_TOL_ENCODER
    return LOGIT_TOL_MAP_STANDARD, LOGIT_DIVERGENCE_TOL_STANDARD
