#!/usr/bin/env python3
"""
Direct Class Model Inference Runner for NeuronX

A clean inference utility that accepts model classes directly, providing
maximum flexibility and control over the inference process.

Architecture:
- Direct model class imports
- Simple configuration
- Core inference engine
- No enum or CLI dependencies
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Type, Tuple

import torch

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def setup_inference_environment() -> None:
    """Setup environment for inference."""
    print("🔧 Setting up inference environment...")
    
    # Common settings
    os.environ["NEURON_RT_LOG_LEVEL"] = "ERROR"
    os.environ["NEURON_FUSE_SOFTMAX"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    
    # XLA settings
    os.environ["XLA_IR_DEBUG"] = "0"
    os.environ["XLA_HLO_DEBUG"] = "0"
    os.environ["XLA_FLAGS"] = "--xla_hlo_profile=false"
    
    print("✅ Inference environment configured")


def load_neuron_config(compiled_path: str) -> Dict[str, Any]:
    """Load neuron configuration from compiled model."""
    print("🔄 Loading neuron configuration...")
    
    config_path = Path(compiled_path) / "neuron_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Neuron config not found: {config_path}")
    
    with open(config_path) as f:
        config_data = json.load(f)
    
    # Extract neuron config
    if "neuron_config" in config_data:
        neuron_config = config_data["neuron_config"]
    else:
        neuron_config = config_data
    
    print(f"✅ NeuronConfig loaded:")
    print(f"   TP Degree: {neuron_config.get('tp_degree', 1)}")
    print(f"   Batch Size: {neuron_config.get('batch_size', 1)}")
    print(f"   Sequence Length: {neuron_config.get('seq_len', 128)}")
    print(f"   Data Type: {neuron_config.get('torch_dtype', 'torch.bfloat16')}")
    print(f"   On CPU: {neuron_config.get('on_cpu', False)}")
    print(f"   Modular Flow: {neuron_config.get('enable_cte_modular_flow', False)}")
    
    return neuron_config


def load_tokenizer(model_path: str):
    """Load tokenizer from model path."""
    print("📝 Loading tokenizer...")
    print(f"   Tokenizer path: {model_path}")
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("   ✅ Tokenizer loaded")
        return tokenizer
        
    except Exception as e:
        print(f"   ❌ Failed to load tokenizer: {e}")
        raise


def load_model_config(model_path: str, config_class: Type, neuron_config_obj: Optional[Any] = None):
    """Load model configuration.

    Args:
        model_path: Path to the model directory (for model architecture config)
        config_class: Configuration class to instantiate
        neuron_config_obj: Pre-loaded neuron config object from compiled_path (optional)
    """
    print("🔄 Loading model configuration...")
    print(f"   Model config path: {model_path}")
    if neuron_config_obj:
        print(f"   Using neuron_config from compiled model")

    try:
        # Pass neuron_config object to from_pretrained to avoid loading from model_path
        model_config = config_class.from_pretrained(model_path, neuron_config=neuron_config_obj)
        print("   ✅ Model configuration loaded")
        return model_config

    except Exception as e:
        print(f"   ❌ Failed to load model configuration: {e}")
        raise


def resolve_modules_for_inference(modules_to_capture: Optional[Union[str, List[str]]], compiled_path: str) -> Optional[List[str]]:
    """
    Resolve modules_to_capture specification for inference.

    Args:
        modules_to_capture: Module specification:
            - "all": Load all modules from saved tensor_capture_config.json
            - Single module name as string
            - List of module names
            - None: Load from saved config (backward compatible)

        compiled_path: Path to compiled model directory

    Returns:
        List of module names to capture, or None if no tensor capture
    """
    if modules_to_capture is None:
        # Backward compatible: load from saved config
        tensor_capture_config_path = Path(compiled_path) / "tensor_capture_config.json"
        if tensor_capture_config_path.exists():
            print("📋 Loading modules from saved tensor_capture_config.json")
            with open(tensor_capture_config_path) as f:
                tc_config = json.load(f)
            return tc_config.get("modules_to_capture", None)
        return None

    # Handle "all" mode - load from saved config
    if isinstance(modules_to_capture, str) and modules_to_capture.lower() == "all":
        print("🔍 Loading all modules from saved tensor_capture_config.json...")
        tensor_capture_config_path = Path(compiled_path) / "tensor_capture_config.json"
        if tensor_capture_config_path.exists():
            with open(tensor_capture_config_path) as f:
                tc_config = json.load(f)
            all_modules = tc_config.get("modules_to_capture", [])
            print(f"   Found {len(all_modules)} modules in saved config")
            return all_modules
        else:
            print("   ⚠️  tensor_capture_config.json not found")
            print("   Model may not have been compiled with tensor capture enabled")
            return None

    # Handle single module name
    if isinstance(modules_to_capture, str):
        print(f"📌 Capturing single module: {modules_to_capture}")
        return [modules_to_capture]

    # Handle list of modules
    if isinstance(modules_to_capture, list):
        print(f"📋 Capturing {len(modules_to_capture)} specified modules")
        return modules_to_capture

    raise ValueError(f"Invalid modules_to_capture type: {type(modules_to_capture)}")


def load_compiled_model(model_class: Type, config_class: Type, model_path: str, compiled_path: str,
                       enable_tensor_capture: bool = False, modules_to_capture: Optional[Union[str, List[str]]] = None):
    """Load compiled model with proper configuration."""
    print("🏗️  Loading compiled model...")

    try:
        # Load neuron config
        neuron_config_dict = load_neuron_config(compiled_path)

        # Setup distributed environment
        tp_degree = neuron_config_dict.get('tp_degree', 1)

        # Create neuron config object from compiled_path dict
        # Use MoENeuronConfig for MoE models, otherwise use NeuronConfig
        from neuronx_distributed_inference.models.config import NeuronConfig, MoENeuronConfig
        # Check if this is an MoE model by looking for MoE-specific keys
        is_moe = 'moe_tp_degree' in neuron_config_dict or 'router_config' in neuron_config_dict
        neuron_config_class = MoENeuronConfig if is_moe else NeuronConfig
        neuron_config_obj = neuron_config_class(**neuron_config_dict)

        # Load model configuration, passing neuron_config object from compiled_path
        model_config = load_model_config(model_path, config_class, neuron_config_obj)

        # Convert torch_dtype string to actual torch dtype
        dtype_str = neuron_config_dict.get('torch_dtype', 'torch.bfloat16')
        if isinstance(dtype_str, str):
            if dtype_str.startswith('torch.'):
                dtype = getattr(torch, dtype_str.split('.')[1])
            else:
                dtype = torch.bfloat16
        else:
            dtype = dtype_str

        neuron_config_kwargs = {
            'tp_degree': neuron_config_dict.get('tp_degree', 1),
            'world_size': neuron_config_dict.get('world_size', 1),
            'batch_size': neuron_config_dict.get('batch_size', 1),
            'seq_len': neuron_config_dict.get('seq_len', 128),
            'torch_dtype': dtype,
            'save_sharded_checkpoint': neuron_config_dict.get('save_sharded_checkpoint', True),
            'enable_cte_modular_flow': neuron_config_dict.get('enable_cte_modular_flow', False),
            'on_cpu': neuron_config_dict.get('on_cpu', False)
        }

        # Resolve modules for tensor capture if enabled
        if enable_tensor_capture:
            print("🎯 Setting up tensor capture configuration...")

            # Resolve which modules to capture
            resolved_modules = resolve_modules_for_inference(modules_to_capture, compiled_path)

            if resolved_modules:
                from neuronx_distributed_inference.models.config import (
                    TensorCaptureConfig,
                    OnDeviceSamplingConfig,
                )

                # Load additional tensor capture settings from saved config
                tensor_capture_config_path = Path(compiled_path) / "tensor_capture_config.json"
                capture_inputs = True
                max_intermediate_tensors = 10

                if tensor_capture_config_path.exists():
                    with open(tensor_capture_config_path) as f:
                        tc_config = json.load(f)
                    capture_inputs = tc_config.get("capture_inputs", True)
                    max_intermediate_tensors = tc_config.get("max_intermediate_tensors", 10)

                # Create TensorCaptureConfig with resolved modules
                tensor_capture_config = TensorCaptureConfig(
                    modules_to_capture=resolved_modules,
                    capture_inputs=capture_inputs,
                    max_intermediate_tensors=max_intermediate_tensors,
                )

                # Tensor capture requires OnDeviceSamplingConfig
                on_device_sampling_config = OnDeviceSamplingConfig()

                neuron_config_kwargs["tensor_capture_config"] = tensor_capture_config
                neuron_config_kwargs["on_device_sampling_config"] = on_device_sampling_config

                print(f"   ✅ Tensor capture enabled for {len(resolved_modules)} modules")
                print(f"   Capture inputs: {capture_inputs}")
                print(f"   Max intermediate tensors: {max_intermediate_tensors}")
            else:
                print(f"   ⚠️  No modules resolved for tensor capture")
                print(f"   Tensor capture will be disabled")

        neuron_config = NeuronConfig(**neuron_config_kwargs)

        # Add neuron_config to model_config
        model_config.neuron_config = neuron_config

        print("   ✅ Model initialized successfully")

        # Create model instance
        if hasattr(model_class, 'from_pretrained'):
            # For models that support from_pretrained
            model = model_class.from_pretrained(compiled_path, config=model_config)
        else:
            # For models that need explicit initialization
            model = model_class(model_path=model_path, config=model_config)

        print("   Loading model weights...")

        # Load the compiled model
        if hasattr(model, 'load'):
            model.load(compiled_path)
        else:
            # Alternative loading method
            model_file = Path(compiled_path) / "model.pt"
            if model_file.exists():
                model.load_state_dict(torch.load(model_file, map_location='cpu'))

        print("   ✅ Model weights loaded and ready for inference")

        return model, neuron_config

    except Exception as e:
        print(f"   ❌ Failed to load compiled model: {e}")
        import traceback
        traceback.print_exc()
        raise


def run_inference_with_classes(
    model_class: Type,
    config_class: Type,
    model_path: str,
    compiled_path: Optional[str] = None,
    prompt: str = "",
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_p: float = 0.9,
    cpu_mode: bool = False,
    num_layers: Optional[int] = None,
    seq_len: int = 512,
    batch_size: int = 1,
    # Special token suppression
    suppress_special_tokens: bool = True,
    # Tensor capture parameters
    enable_tensor_capture: bool = False,
    modules_to_capture: Optional[Union[str, List[str]]] = None,
    capture_indices: Optional[List[int]] = None,
    tensor_capture_output_dir: Optional[str] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Run inference using provided model and config classes.

    Args:
        model_class: The model class to use
        config_class: The config class to use
        model_path: Path to the original model
        compiled_path: Path to the compiled model (not used in CPU mode)
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        cpu_mode: If True, run in CPU mode without loading compiled model
        num_layers: Number of layers to load (for CPU testing with reduced layers)
        seq_len: Sequence length for CPU mode neuron_config
        batch_size: Batch size for CPU mode neuron_config
        suppress_special_tokens: If True, prevent generation of special tokens (BOS, EOS, PAD, etc.)
            by setting their logits to -inf during generation. This is critical for coherent output
            from compiled models. Default: True.
        enable_tensor_capture: If True, enable tensor capture during inference
        modules_to_capture: Module selection for tensor capture:
            - "all": Use all modules from compiled model's tensor_capture_config.json
            - "module_name": Capture single module by name
            - ["module1", "module2"]: Capture specific list of modules
            - None: Use compiled model's saved config (backward compatible)
        capture_indices: List of generation steps at which to capture tensors (e.g., [1, 5, 10])
        tensor_capture_output_dir: Directory to save captured tensors (default: agent_artifacts/tmp/captured_tensors)

    Returns:
        Tuple of (success, response, metrics)
    """
    print("🚀 Initializing Generic Model Inference")
    print("=" * 60)
    print(f"Model Class: {model_class.__name__}")
    print(f"Config Class: {config_class.__name__}")
    print(f"Model Path: {model_path}")
    print(f"CPU Mode: {cpu_mode}")
    if not cpu_mode:
        print(f"Compiled Path: {compiled_path}")
    print("=" * 60)

    try:
        # Setup environment
        if not cpu_mode:
            setup_inference_environment()

        # Import model classes
        print("📦 Importing model classes...")
        print(f"   ✅ Model class imported: {model_class.__name__}")
        print(f"   ✅ Config class imported: {config_class.__name__}")

        # Load tokenizer
        tokenizer = load_tokenizer(model_path)

        # Load special tokens for suppression
        special_token_ids_list: List[int] = []
        if suppress_special_tokens:
            print("📝 Loading special tokens for suppression...")
            special_token_ids = []
            if hasattr(tokenizer, 'all_special_ids'):
                special_token_ids = tokenizer.all_special_ids
            elif hasattr(tokenizer, 'additional_special_tokens_ids'):
                special_token_ids = tokenizer.additional_special_tokens_ids
                # Manually add core special tokens if not in additional_special_tokens
                if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
                    special_token_ids.append(tokenizer.bos_token_id)
                if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                    special_token_ids.append(tokenizer.eos_token_id)
                if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                    special_token_ids.append(tokenizer.pad_token_id)

            # Remove duplicates
            special_token_ids_list = list(set(special_token_ids))
            print(f"   Found {len(special_token_ids_list)} special tokens")
        else:
            print("⚠️  Special token suppression disabled - may produce gibberish output")

        # Load model based on mode
        if cpu_mode:
            print("🖥️  Loading model in CPU mode...")
            # Import necessary classes for CPU mode
            from neuronx_distributed_inference.models.config import MoENeuronConfig

            # Create CPU neuron_config
            neuron_config = MoENeuronConfig(
                tp_degree=1,
                batch_size=batch_size,
                seq_len=seq_len,
                max_context_length=seq_len,
                torch_dtype=torch.float32,
                on_cpu=True,
                attn_kernel_enabled=False,
            )

            # Load model config
            config = config_class.from_pretrained(model_path, neuron_config=neuron_config)

            # Filter and store special tokens on config
            if suppress_special_tokens and special_token_ids_list:
                config.special_token_ids = [tid for tid in special_token_ids_list if tid < config.vocab_size]
                print(f"   ✅ Loaded {len(config.special_token_ids)} special tokens for runtime suppression")

            # Optionally reduce layers for faster testing
            if num_layers is not None:
                print(f"   🔧 Reducing model to {num_layers} layers for testing")
                config.num_hidden_layers = num_layers

            # Load model
            print(f"   Loading {config.num_hidden_layers} layers...")
            model = model_class(model_path, config)
            model.to_cpu()
            print(f"   ✅ Model loaded on CPU")
        else:
            # Load compiled model for Neuron hardware
            if compiled_path is None:
                raise ValueError("compiled_path is required for non-CPU mode")
            
            neuron_config_dict = load_neuron_config(compiled_path)
            model, neuron_config = load_compiled_model(
                model_class,
                config_class,
                model_path,
                compiled_path,
                enable_tensor_capture=enable_tensor_capture,
                modules_to_capture=modules_to_capture
            )

            # Get config from model and add special tokens
            if suppress_special_tokens and special_token_ids_list:
                config = model.config
                config.special_token_ids = [tid for tid in special_token_ids_list if tid < config.vocab_size]
                print(f"   ✅ Loaded {len(config.special_token_ids)} special tokens for runtime suppression")

        print("✅ All components initialized successfully")

        # Create special token mask for runtime suppression
        if suppress_special_tokens:
            if hasattr(model.config, 'special_token_ids') and model.config.special_token_ids:
                special_token_mask = torch.tensor(model.config.special_token_ids, dtype=torch.long)
                model.special_token_mask = special_token_mask
                print(f"✅ Created special token mask with {len(special_token_mask)} tokens for runtime suppression")
            else:
                model.special_token_mask = None
                print("⚠️  No special tokens found - proceeding without suppression")
        else:
            model.special_token_mask = None

        # Setup tensor capture hook if enabled
        tensor_capture_hook = None
        if enable_tensor_capture and not cpu_mode:
            print("\n🎯 Setting up tensor capture...")

            # Set default output directory if not provided
            if tensor_capture_output_dir is None:
                tensor_capture_output_dir = "agent_artifacts/tmp/captured_tensors"

            # Set default capture indices if not provided
            if capture_indices is None:
                capture_indices = [1]

            print(f"   Output directory: {tensor_capture_output_dir}")
            print(f"   Capture indices: {capture_indices}")

            # Create tensor capture hook
            from neuronx_distributed_inference.utils.tensor_capture_utils import get_tensor_capture_hook

            tensor_capture_hook = get_tensor_capture_hook(
                capture_indices=capture_indices,
                tensor_capture_save_dir=tensor_capture_output_dir,
            )

            print(f"   ✅ Tensor capture hook created")

        # Run inference
        print("\n🧪 Running inference test...")
        print("=" * 50)
        print(f"📝 Prompt: '{prompt}'")
        
        # Tokenize input
        print("🔄 Tokenizing input...")
        # Include special tokens as the model expects them (BOS/EOS at start)
        # Note: BOS (100352) will be filtered if > vocab_size, EOS (2) is expected
        inputs = tokenizer([prompt], padding=True, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs.input_ids
        input_length = input_ids.shape[1]
        
        print(f"   Input tokens: {input_length}")
        print(f"   Input shape: {input_ids.shape}")
        
        # Generate response
        print("⚡ Generating response...")
        start_time = time.time()
        
        # Determine if we should use greedy decoding
        use_greedy = temperature < 0.1
        if use_greedy:
            print("   Using GREEDY decoding (argmax)")
        else:
            print(f"   Using SAMPLING (temperature={temperature}, top_p={top_p})")
        
        with torch.no_grad():
            # Always use manual generation loop for better control and consistency
            # The model.generate() method may have different behavior with sampling
            if False:  # Disabled: hasattr(model, 'generate'):
                # Use generate method if available
                generate_kwargs = {
                    'input_ids': input_ids,
                    'max_new_tokens': max_new_tokens,
                    'temperature': temperature,
                    'top_p': top_p,
                    'do_sample': not use_greedy,
                    'pad_token_id': tokenizer.pad_token_id,
                    'eos_token_id': tokenizer.eos_token_id,
                    'use_cache': True
                }
                if use_greedy:
                    generate_kwargs.pop('temperature', None)
                    generate_kwargs.pop('top_p', None)

                # Add tensor_capture_hook if enabled
                if tensor_capture_hook is not None:
                    generate_kwargs['tensor_capture_hook'] = tensor_capture_hook

                outputs = model.generate(**generate_kwargs)
                generated_ids = outputs
            else:
                # Manual generation loop
                generated_ids = input_ids.clone()
                
                for _ in range(max_new_tokens):
                    seq_len = generated_ids.shape[1]
                    
                    # Create position_ids if needed
                    position_ids = torch.arange(seq_len).unsqueeze(0)
                    
                    # Forward pass
                    outputs = model(generated_ids, position_ids=position_ids)
                    
                    # Get logits
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    elif isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs

                    # Apply special token suppression BEFORE temperature/sampling
                    if suppress_special_tokens and hasattr(model, 'special_token_mask') and model.special_token_mask is not None:
                        vocab_size = logits.shape[-1]
                        # Filter mask to valid indices only
                        valid_mask = model.special_token_mask[model.special_token_mask < vocab_size]
                        if len(valid_mask) > 0:
                            # Set special token logits to -inf to prevent selection
                            logits[:, :, valid_mask] = float('-inf')

                    # Get logits for last position only
                    next_token_logits = logits[:, -1, :]
                    
                    if use_greedy:
                        # GREEDY DECODING: Use argmax (deterministic, like original script)
                        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                    else:
                        # SAMPLING: Apply temperature and top-p
                        next_token_logits = next_token_logits / temperature
                        
                        # Apply top-p sampling
                        if top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                            next_token_logits[indices_to_remove] = float('-inf')
                        
                        # Sample from distribution
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Append to sequence (dynamic growth with torch.cat)
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    
                    # Check for EOS
                    if tokenizer.eos_token_id and next_token.item() == tokenizer.eos_token_id:
                        break
        
        inference_time = time.time() - start_time
        
        # Decode output
        print("🔄 Decoding output...")
        full_response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        generated_text = full_response[len(prompt):].strip()
        tokens_generated = generated_ids.shape[1] - input_length
        tokens_per_second = tokens_generated / inference_time if inference_time > 0 else 0
        
        # Display results
        print(f"\n{'=' * 50}")
        print("🎯 INFERENCE RESULTS")
        print("=" * 50)
        print(f"📝 Prompt: {prompt}")
        print(f"🤖 Response: {generated_text}")
        print(f"📊 Performance Metrics:")
        print(f"   ⏱️  Inference time: {inference_time:.2f} seconds")
        print(f"   🔢 Input tokens: {input_length}")
        print(f"   🔢 Generated tokens: {tokens_generated}")
        print(f"   🚀 Tokens/second: {tokens_per_second:.1f}")
        print(f"   🔧 Tensor Parallel degree: {neuron_config.tp_degree}")
        
        # Check if modular flow attribute exists
        if hasattr(neuron_config, 'enable_cte_modular_flow'):
            print(f"   🚀 Modular flow: {'Enabled' if neuron_config.enable_cte_modular_flow else 'Disabled'}")
        
        # Report on captured tensors if tensor capture was enabled
        if enable_tensor_capture and tensor_capture_output_dir:
            print(f"\n📊 Tensor Capture Results:")
            output_path = Path(tensor_capture_output_dir)
            if output_path.exists():
                tensor_files = list(output_path.glob("*.pt"))
                print(f"   Captured {len(tensor_files)} tensor files:")
                for f in sorted(tensor_files)[:10]:  # Show first 10
                    print(f"      - {f.name}")
                if len(tensor_files) > 10:
                    print(f"      ... and {len(tensor_files) - 10} more")
                print(f"   📁 Location: {tensor_capture_output_dir}")
            else:
                print(f"   ⚠️  No tensors captured")

        print(f"\n🎉 INFERENCE COMPLETED SUCCESSFULLY!")
        print(f"✅ The compiled {model_class.__name__} model is working correctly")

        # Prepare metrics
        metrics = {
            "inference_time": inference_time,
            "input_tokens": input_length,
            "generated_tokens": tokens_generated,
            "tokens_per_second": tokens_per_second,
            "tp_degree": neuron_config.tp_degree,
            "modular_flow": getattr(neuron_config, 'enable_cte_modular_flow', False)
        }

        # Add tensor capture metrics if enabled
        if enable_tensor_capture and tensor_capture_output_dir:
            output_path = Path(tensor_capture_output_dir)
            if output_path.exists():
                tensor_files = list(output_path.glob("*.pt"))
                metrics["tensor_capture_enabled"] = True
                metrics["tensors_captured"] = len(tensor_files)
                metrics["tensor_capture_dir"] = str(tensor_capture_output_dir)
            else:
                metrics["tensor_capture_enabled"] = True
                metrics["tensors_captured"] = 0

        return True, generated_text, metrics
        
    except Exception as e:
        print(f"\n❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False, "", {}


# No CLI interface - use Python API only