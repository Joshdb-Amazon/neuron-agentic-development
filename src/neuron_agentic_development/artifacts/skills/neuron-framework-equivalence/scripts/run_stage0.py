#!/usr/bin/env python3
"""
Stage 0: Build Model Trees for source and target.

Generates compressed trees, pretty-printed ASCII, and flat module paths
for both models. The target model is instantiated in CPU mode (TP=1).

Usage:
    python3 scripts/run_stage0.py \
        --source-model-path /path/to/hf_model \
        --target-model-path /path/to/hf_model \
        --target-module-file /path/to/modeling_xxx.py \
        --target-inner-class NeuronXxxModel \
        --target-config-class XxxInferenceConfig \
        --output-dir experiments/model_tree
"""

import argparse
import importlib
import importlib.util
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Stage 0: Build Model Trees")
    parser.add_argument("--source-model-path", required=True, help="Path to source (HF) model weights")
    parser.add_argument("--target-model-path", required=True, help="Path to HF-compatible weights for target config")
    parser.add_argument("--target-module-file", required=True, help="Path to target modeling .py file")
    parser.add_argument("--target-inner-class", required=True, help="Inner model class name (NeuronXxxModel)")
    parser.add_argument("--target-config-class", required=True, help="Config class name (XxxInferenceConfig)")
    parser.add_argument("--output-dir", required=True, help="Output directory for tree artifacts")
    args = parser.parse_args()

    # Import tree utilities from same directory
    sys.path.insert(0, os.path.dirname(__file__))
    from stage0_scaffolding import build_model_tree, save_model_tree, compressed_tree_to_pretty_string

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Source Model Tree ──
    print("=" * 70)
    print("  Building Source Model Tree")
    print("=" * 70)

    import torch
    from transformers import AutoModelForCausalLM

    print(f"  Loading source model from: {args.source_model_path}")
    source_model = AutoModelForCausalLM.from_pretrained(
        args.source_model_path, torch_dtype="auto",
        trust_remote_code=True, attn_implementation="eager",
    )
    source_model.eval()

    tree, full, paths = build_model_tree(source_model, "model")
    pretty = save_model_tree(tree, full, paths, args.output_dir, "model_tree_source")
    print(f"\n  Source tree ({len(paths)} module paths):")
    print(pretty)

    del source_model
    import gc; gc.collect()

    # ── Target Model Tree (CPU mode, inner model only) ──
    print("\n" + "=" * 70)
    print("  Building Target Model Tree (CPU mode)")
    print("=" * 70)

    os.environ["NXD_CPU_MODE"] = "1"
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")

    # Import target classes
    module_file = os.path.abspath(args.target_module_file)
    module_dir = os.path.dirname(module_file)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    mod_name = f"_target_{os.path.basename(module_file).replace('.py', '')}"
    spec = importlib.util.spec_from_file_location(mod_name, module_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)

    InnerClass = getattr(module, args.target_inner_class)
    ConfigClass = getattr(module, args.target_config_class)
    print(f"  Inner class: {InnerClass.__name__}")
    print(f"  Config class: {ConfigClass.__name__}")

    # Get correct NeuronConfig subclass
    from neuronx_distributed_inference.models.config import NeuronConfig
    NeuronConfigCls = NeuronConfig
    if hasattr(ConfigClass, "get_neuron_config_cls"):
        NeuronConfigCls = ConfigClass.get_neuron_config_cls()
        print(f"  NeuronConfig class: {NeuronConfigCls.__name__}")

    neuron_config = NeuronConfigCls(
        tp_degree=1, world_size=1, batch_size=1, seq_len=128,
        torch_dtype=torch.bfloat16, save_sharded_checkpoint=True, on_cpu=True,
    )

    # Load config
    try:
        config = ConfigClass.from_pretrained(args.target_model_path, neuron_config=neuron_config)
    except (TypeError, AttributeError):
        from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
        config = ConfigClass(neuron_config, load_config=load_pretrained_config(args.target_model_path))
    if hasattr(config, "add_derived_config"):
        config.add_derived_config()

    # Init distributed
    from neuronx_distributed.parallel_layers.parallel_state import (
        initialize_model_parallel, model_parallel_is_initialized,
    )
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo", world_size=1, rank=0)
    if not model_parallel_is_initialized():
        initialize_model_parallel(tensor_model_parallel_size=1)

    # Instantiate
    print(f"  Instantiating {InnerClass.__name__} (CPU mode, structure only)...")
    target_model = InnerClass(config)

    tree, full, paths = build_model_tree(target_model, "model")
    pretty = save_model_tree(tree, full, paths, args.output_dir, "model_tree_target")
    print(f"\n  Target tree ({len(paths)} module paths):")
    print(pretty)

    print("\n" + "=" * 70)
    print("  Stage 0 complete. Trees saved to:", args.output_dir)
    print("  Next: Compare the trees and build component_mapping.json")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
