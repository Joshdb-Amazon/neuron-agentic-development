# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Monkey-patches that fix compatibility issues between the installed versions
of ``transformers`` and ``neuronx_distributed_inference`` (NxDI).

Why these exist
---------------
NxDI's ``HuggingFaceGenerationAdapter`` and its config serialisation helpers
were written against an older transformers API.  Newer transformers versions:

  * reject read-only ``PretrainedConfig`` properties passed to ``__init__``
  * require ``generation_config.transformers_version`` to be a real string
  * expect ``GenerationMixin`` in the MRO of anything that calls ``.generate()``

Rather than forking NxDI, we patch at import time so the validator works with
whatever SDK version is on the box.

Patch inventory
---------------
1. ``_patched_to_pretrained_config``  — strips ~35 read-only attrs before
   constructing a ``PretrainedConfig`` from an NxDI config dict.
2. ``_apply_generation_config_patch`` — wraps
   ``GenerationMixin._prepare_generation_config`` to coerce raw dicts and
   ``None`` versions into valid ``GenerationConfig`` objects.
3. ``patch_generation_mixin``         — injects ``GenerationMixin`` into
   ``HuggingFaceGenerationAdapter.__bases__`` so ``.generate()`` works.
4. ``ensure_generation_config_version`` — walks ``model`` / ``model.config``
   and forces ``generation_config.transformers_version`` to a string.
5. ``ensure_pad_token``               — copies ``tokenizer.pad_token_id``
   onto ``generation_config`` to silence repeated HF warnings.
6. ``patch_load_hf_model``            — monkey-patches
   ``model.__class__.load_hf_model`` so NxDI's logit-matching code can load
   the HF golden reference through the model class, with a ``DynamicCache``
   retry fallback.

All patches are applied once via ``apply_all_patches()`` at the bottom of
this module (i.e. on first import of the ``validator`` package).
"""

import importlib

from transformers import GenerationConfig, PretrainedConfig

import neuronx_distributed_inference.utils.hf_adapter as _hf_adapter_module

# ---------------------------------------------------------------------------
# 1. Filter read-only PretrainedConfig attributes
# ---------------------------------------------------------------------------

_READONLY_ATTRS = {
    'use_return_dict', 'output_hidden_states', 'output_attentions',
    'torchscript', 'use_bfloat16', 'pruned_heads', 'tie_word_embeddings',
    'is_encoder_decoder', 'is_decoder', 'cross_attention_hidden_size',
    'add_cross_attention', 'tie_encoder_decoder', 'max_length',
    'min_length', 'do_sample', 'early_stopping', 'num_beams',
    'num_beam_groups', 'diversity_penalty', 'temperature',
    'top_k', 'top_p', 'typical_p', 'repetition_penalty',
    'length_penalty', 'no_repeat_ngram_size', 'bad_words_ids',
    'force_words_ids', 'renormalize_logits', 'constraints',
    'forced_bos_token_id', 'forced_eos_token_id', 'remove_invalid_values',
    'exponential_decay_length_penalty', 'suppress_tokens',
    'begin_suppress_tokens', 'forced_decoder_ids', 'num_return_sequences',
    'chunk_size_feed_forward', 'output_scores', 'return_dict_in_generate',
    'pad_token_id', 'bos_token_id', 'eos_token_id',
    'encoder_no_repeat_ngram_size', 'decoder_start_token_id', 'use_cache',
}


def _patched_to_pretrained_config(config):
    """Replace ``hf_adapter.to_pretrained_config`` with a version that strips
    read-only properties (``use_return_dict``, ``pad_token_id``, etc.) before
    calling ``PretrainedConfig(**filtered)``.

    Without this, newer transformers raises ``AttributeError`` because those
    properties are computed from other fields and can't be set via __init__.
    """
    config_dict = _hf_adapter_module.to_dict(config)
    filtered = {k: v for k, v in config_dict.items() if k not in _READONLY_ATTRS}
    return PretrainedConfig(**filtered)


# ---------------------------------------------------------------------------
# 2. Patch GenerationMixin._prepare_generation_config
# ---------------------------------------------------------------------------

def _apply_generation_config_patch():
    """Wrap ``GenerationMixin._prepare_generation_config`` so it tolerates:

    * ``generation_config`` being a plain dict (common with NxDI adapters)
    * ``transformers_version`` being ``None`` instead of a string

    The wrapper normalises both cases before delegating to the original
    method.  A ``_is_safe_patched`` sentinel prevents double-application.
    """
    try:
        from transformers.generation.utils import GenerationMixin
        import transformers as _tf

        _orig = GenerationMixin._prepare_generation_config

        if getattr(_orig, "_is_safe_patched", False):
            return

        def _safe_prepare(self, generation_config, *args, **kwargs):
            for obj in [self, getattr(self, "config", None)]:
                if obj is None:
                    continue
                gc = getattr(obj, "generation_config", None)
                if gc is None:
                    continue
                if isinstance(gc, dict):
                    obj.generation_config = GenerationConfig(**{
                        k: v for k, v in gc.items()
                        if k not in ("transformers_version",)
                    })
                    obj.generation_config.transformers_version = _tf.__version__
                elif not isinstance(getattr(gc, "transformers_version", None), str):
                    gc.transformers_version = _tf.__version__
            if generation_config is not None:
                if isinstance(generation_config, dict):
                    generation_config = GenerationConfig(**{
                        k: v for k, v in generation_config.items()
                        if k not in ("transformers_version",)
                    })
                    generation_config.transformers_version = _tf.__version__
                elif not isinstance(getattr(generation_config, "transformers_version", None), str):
                    generation_config.transformers_version = _tf.__version__
            return _orig(self, generation_config, *args, **kwargs)

        _safe_prepare._is_safe_patched = True  # type: ignore[attr-defined]
        GenerationMixin._prepare_generation_config = _safe_prepare
    except (ImportError, AttributeError):
        pass


# ---------------------------------------------------------------------------
# 3. Inject GenerationMixin into HuggingFaceGenerationAdapter
# ---------------------------------------------------------------------------

def patch_generation_mixin():
    """Ensure ``HuggingFaceGenerationAdapter`` inherits from
    ``transformers.GenerationMixin``.

    NxDI's adapter doesn't always include ``GenerationMixin`` in its MRO,
    but the logit-matching code (and our own enhanced-metrics code) calls
    ``.generate()`` on it, which lives on ``GenerationMixin``.  This patch
    appends it to ``__bases__`` if missing.

    Safe to call multiple times — checks before mutating.
    """
    from transformers import GenerationMixin
    for mod_path in (
        "neuronx_distributed_inference.utils.accuracy",
        "neuronx_distributed_inference.utils.hf_adapter",
    ):
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, "HuggingFaceGenerationAdapter", None)
            if cls and GenerationMixin not in cls.__bases__:
                cls.__bases__ = cls.__bases__ + (GenerationMixin,)
        except (ImportError, AttributeError):
            pass


# ---------------------------------------------------------------------------
# 4. Ensure generation_config.transformers_version is valid
# ---------------------------------------------------------------------------

def ensure_generation_config_version(model):
    """Walk ``model`` and ``model.config``, find every ``generation_config``
    attribute, and make sure its ``transformers_version`` is a real string.

    Handles three cases:
      * ``generation_config`` is missing entirely → create a default one
      * it's a raw dict → convert to ``GenerationConfig`` and stamp version
      * ``transformers_version`` is ``None`` → set to current version

    Called before every ``.generate()`` call to prevent the
    ``TypeError: '<' not supported between instances of 'NoneType' and 'str'``
    crash inside transformers' version-check logic.
    """
    import transformers
    for obj in [model, getattr(model, "config", None)]:
        if obj is None:
            continue
        gc = getattr(obj, "generation_config", None)
        if gc is None:
            obj.generation_config = GenerationConfig()
            gc = obj.generation_config
        if isinstance(gc, dict):
            obj.generation_config = GenerationConfig(**{
                k: v for k, v in gc.items() if k != "transformers_version"
            })
            obj.generation_config.transformers_version = transformers.__version__
        elif not isinstance(getattr(gc, "transformers_version", None), str):
            gc.transformers_version = transformers.__version__


# ---------------------------------------------------------------------------
# 5. Set pad_token_id on generation_config
# ---------------------------------------------------------------------------

def ensure_pad_token(generation_config, tokenizer):
    """Copy ``tokenizer.pad_token_id`` onto ``generation_config`` if the
    latter is ``None``.

    Without this, every ``.generate()`` call emits a noisy warning about
    defaulting pad_token_id to eos_token_id.
    """
    if generation_config.pad_token_id is None and tokenizer.pad_token_id is not None:
        generation_config.pad_token_id = tokenizer.pad_token_id


# ---------------------------------------------------------------------------
# 6. Monkey-patch load_hf_model for NxDI logit matching
# ---------------------------------------------------------------------------

def patch_load_hf_model(model, loader_fn):
    """Install a ``load_hf_model`` staticmethod on ``model.__class__`` so
    NxDI's ``check_accuracy_logits_v2`` can load the HF golden reference
    through the Neuron model class.

    The installed loader wraps ``loader_fn`` (typically
    ``load_hf_golden_model``) and adds a retry path: if ``.generate()``
    fails with a ``DynamicCache`` / ``seen_tokens`` ``AttributeError``
    (a known transformers version mismatch), it retries with
    ``use_cache=False``.

    Args:
        model:     The instantiated Neuron model whose *class* gets patched.
        loader_fn: ``Callable[[str], PreTrainedModel]`` — the function that
                   actually loads the HF checkpoint (receives model_path).
    """
    def _wrapped(path):
        hf_model = loader_fn(path)
        _orig_gen = hf_model.generate

        def _gen_fallback(*a, **kw):
            try:
                return _orig_gen(*a, **kw)
            except AttributeError as e:
                if "DynamicCache" in str(e) or "seen_tokens" in str(e):
                    kw["use_cache"] = False
                    return _orig_gen(*a, **kw)
                raise
        hf_model.generate = _gen_fallback
        return hf_model
    model.__class__.load_hf_model = staticmethod(_wrapped)


# ---------------------------------------------------------------------------
# Apply all patches eagerly on import
# ---------------------------------------------------------------------------

def apply_all_patches():
    """Run every compatibility patch once.

    Called automatically at the bottom of this module (and therefore on
    first import of the ``validator`` package via ``__init__.py``).
    Idempotent — safe to call again, though there's no reason to.
    """
    _hf_adapter_module.to_pretrained_config = _patched_to_pretrained_config
    _apply_generation_config_patch()
    patch_generation_mixin()


apply_all_patches()
