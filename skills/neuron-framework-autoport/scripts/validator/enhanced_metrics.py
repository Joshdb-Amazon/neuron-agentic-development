# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Distribution-level accuracy metrics that go beyond token matching.

While token matching tells you "did the argmax agree?", these metrics
answer "how similar are the full probability distributions?".  They're
computed from a single short generation (32 tokens) on a fixed prompt,
comparing the raw logit tensors from both models position-by-position.

Metrics produced (all stored in the returned dict):

  logit_cosine_similarity_{mean,min,std}
      Cosine similarity of the full vocab-sized logit vectors at each
      position.  Values near 1.0 mean the distributions point in the
      same direction; < 0.9 is a red flag.

  top{k}_agreement
      Fraction of top-k token sets that overlap between the two models.
      Measures whether the models agree on the "likely next tokens" even
      if the exact argmax differs.

  kl_divergence
      KL(HF || Neuron) averaged over positions.  Measures how much
      information is lost if you use the Neuron distribution as a proxy
      for the HF distribution.

  max_logit_diff / mean_logit_diff
      Raw absolute differences in the logit tensors.

  topk{k}_normwise_error_{mean,max,std}
      For k ∈ {5, 50, 1000}: L2 norm of the logit difference restricted
      to HF's top-k indices, normalised by HF's L2 norm.  Focuses the
      error measurement on the tokens that actually matter.

  per_position_mse_{mean,max} / relative_l2_error_{mean,max}
      Full-vector MSE and relative L2 error across all vocab entries.

  mean_prob_of_hf_token
      Average probability the Neuron model assigns to the token that HF
      picked.  Even when the argmax differs, a high value here means the
      Neuron model "almost" agreed.

  per_token_cosine_similarity
      Raw list of per-position cosine similarities (for plotting).

These metrics are always computed alongside token matching (when an HF
model is loaded) and saved in the ``enhanced_metrics`` section of the
accuracy results JSON.
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional

from transformers import GenerationConfig

from .patches import ensure_generation_config_version


def compute_enhanced_metrics(
    neuron_model,
    hf_model,
    tokenizer,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Generate 32 tokens from both models and compare full logit tensors.

    Uses the fixed prompt ``"Hello, I am a language model"`` replicated to
    the model's compiled batch size.  Both models are called with
    ``output_scores=True`` so we get per-position logit tensors.

    The Neuron model is wrapped in ``HuggingFaceGenerationAdapter`` to get
    the ``.generate()`` interface with score output.  After generation,
    ``neuron_model.reset()`` is called to clear KV-cache state.

    Shapes are aligned (min of seq lengths, min of vocab sizes) before
    any comparison, so this works even when the Neuron model pads the
    vocab dimension for TP divisibility.

    On any exception (e.g. the model doesn't support ``output_scores``),
    returns ``{"enhanced_metrics_available": False, "enhanced_metrics_error": ...}``
    instead of crashing the whole validation run.

    Args:
        neuron_model: Loaded + compiled Neuron model.
        hf_model:     HuggingFace reference model (FP32, eval mode).
        tokenizer:    Shared tokenizer (must have pad_token set).
        top_k:        K for the top-k agreement metric (default 5).

    Returns:
        Dict of metric name → value.  Check ``enhanced_metrics_available``
        to know if the computation succeeded.
    """
    prompt = "Hello, I am a language model"
    batch_size = neuron_model.config.neuron_config.batch_size
    seq_len = neuron_model.config.neuron_config.seq_len
    inputs = tokenizer([prompt] * batch_size, padding=True, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    input_len = input_ids.shape[1]

    max_new = min(32, seq_len - input_len - 1)
    if max_new < 1:
        return {"enhanced_metrics_available": False, "reason": "prompt too long for seq_len"}

    result: Dict[str, Any] = {"enhanced_metrics_available": False}

    try:
        # HF generation with logits
        with torch.no_grad():
            try:
                hf_out = hf_model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=max_new, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True, output_scores=True,
                )
            except (AttributeError, TypeError):
                hf_out = hf_model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=max_new, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True, output_scores=True,
                    use_cache=False,
                )

        hf_logits = torch.stack(hf_out.scores)   # [seq, batch, vocab]
        hf_tokens = hf_logits.argmax(dim=2).T     # [batch, seq]

        # Neuron generation with logits
        from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
        gen_model = HuggingFaceGenerationAdapter(neuron_model)
        ensure_generation_config_version(gen_model)
        ensure_generation_config_version(neuron_model)
        neuron_gen_config = GenerationConfig(
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
        import transformers as _tf
        neuron_gen_config.transformers_version = _tf.__version__
        with torch.no_grad():
            neuron_out = gen_model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
                generation_config=neuron_gen_config,
            )
        neuron_model.reset()

        neuron_logits = torch.stack(neuron_out.scores)  # [seq, batch, vocab]
        neuron_tokens = neuron_logits.argmax(dim=2).T

        # Align shapes
        min_seq = min(neuron_logits.shape[0], hf_logits.shape[0])
        min_vocab = min(neuron_logits.shape[2], hf_logits.shape[2])
        n_logits = neuron_logits[:min_seq, :, :min_vocab].float()
        h_logits = hf_logits[:min_seq, :, :min_vocab].float()
        n_tokens = neuron_tokens[:, :min_seq]
        h_tokens = hf_tokens[:, :min_seq]

        # Flatten to [batch*seq, vocab]
        n_flat = n_logits.reshape(-1, min_vocab)
        h_flat = h_logits.reshape(-1, min_vocab)

        # 1. Cosine similarity
        cos_sim = F.cosine_similarity(n_flat, h_flat, dim=-1)
        result["logit_cosine_similarity_mean"] = cos_sim.mean().item()
        result["logit_cosine_similarity_min"] = cos_sim.min().item()
        result["logit_cosine_similarity_std"] = cos_sim.std().item()

        # 2. Top-K agreement
        n_topk = torch.topk(n_logits, top_k, dim=-1).indices  # [seq, batch, k]
        h_topk = torch.topk(h_logits, top_k, dim=-1).indices
        agreement_sum: float = 0
        total_pos = n_topk.shape[0] * n_topk.shape[1]
        for s in range(n_topk.shape[0]):
            for b in range(n_topk.shape[1]):
                n_set = set(n_topk[s, b].tolist())
                h_set = set(h_topk[s, b].tolist())
                agreement_sum += len(n_set & h_set) / top_k
        result[f"top{top_k}_agreement"] = agreement_sum / total_pos if total_pos else 0

        # 3. KL divergence
        n_log_probs = F.log_softmax(n_flat, dim=-1)
        h_probs = F.softmax(h_flat, dim=-1)
        kl = F.kl_div(n_log_probs, h_probs, reduction="batchmean")
        result["kl_divergence"] = kl.item()

        # 4. Logit differences
        diff = (n_logits - h_logits).abs()
        result["max_logit_diff"] = diff.max().item()
        result["mean_logit_diff"] = diff.mean().item()

        # 5. Normwise error for top-k logits
        for k in [5, 50, 1000]:
            if k > min_vocab:
                continue
            h_topk_idx = torch.topk(h_logits, k, dim=-1).indices  # [seq, batch, k]
            per_pos_l2 = []
            for s in range(min_seq):
                for b in range(h_topk_idx.shape[1]):
                    idx = h_topk_idx[s, b]  # [k]
                    h_vals = h_logits[s, b][idx]
                    n_vals = n_logits[s, b][idx]
                    l2 = torch.norm(n_vals - h_vals, p=2).item()
                    h_norm = torch.norm(h_vals, p=2).item()
                    per_pos_l2.append(l2 / h_norm if h_norm > 1e-8 else l2)
            t = torch.tensor(per_pos_l2)
            result[f"topk{k}_normwise_error_mean"] = t.mean().item()
            result[f"topk{k}_normwise_error_max"] = t.max().item()
            result[f"topk{k}_normwise_error_std"] = t.std().item()

        # 6. Full-vector normwise error (MSE and relative L2)
        per_pos_mse = ((n_flat - h_flat) ** 2).mean(dim=-1)
        result["per_position_mse_mean"] = per_pos_mse.mean().item()
        result["per_position_mse_max"] = per_pos_mse.max().item()
        per_pos_rel_l2 = torch.norm(n_flat - h_flat, p=2, dim=-1) / (torch.norm(h_flat, p=2, dim=-1) + 1e-8)
        result["relative_l2_error_mean"] = per_pos_rel_l2.mean().item()
        result["relative_l2_error_max"] = per_pos_rel_l2.max().item()

        # 7. Probability of HF token under Neuron's distribution
        n_probs = F.softmax(n_flat, dim=-1)
        h_tokens_flat = h_tokens.reshape(-1)
        probs_of_hf = []
        for i in range(n_probs.shape[0]):
            tok = h_tokens_flat[i].item()
            if tok < n_probs.shape[1]:
                probs_of_hf.append(n_probs[i, tok].item())
        if probs_of_hf:
            result["mean_prob_of_hf_token"] = sum(probs_of_hf) / len(probs_of_hf)

        # 8. Per-token cosine similarity series (for plotting)
        result["per_token_cosine_similarity"] = cos_sim.tolist()

        result["enhanced_metrics_available"] = True
        result["num_positions_compared"] = total_pos

    except Exception as e:
        result["enhanced_metrics_error"] = str(e)

    return result
