#!/usr/bin/env python3
"""
Direct Class Model Compiler for NeuronX

A clean compilation utility that accepts model classes directly, providing
maximum flexibility and control over the compilation process.

Architecture:
- Direct model class imports
- Simple configuration dataclass
- Core compilation engine
- No enum or CLI dependencies
"""

import os
import json
import shutil
import torch
from pathlib import Path
from typing import Optional, Type, List, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime

if TYPE_CHECKING:
    from neuronx_distributed_inference.models.config import NeuronConfig

# No weights downloading - models should be prepared externally

@dataclass
class CompilationConfig:
    """Configuration for direct class-based model compilation.

    Tensor Capture Module Selection:
    - modules_to_capture="all": Automatically discover and capture all model modules
    - modules_to_capture="module_name": Capture a single module by name
    - modules_to_capture=["module1", "module2"]: Capture specific list of modules
    - modules_to_capture=None: No tensor capture (default)
    """
    model_class: Type
    config_class: Type
    neuron_config_class: Type
    model_path: str
    output_path: str
    batch_size: int = 1
    seq_len: int = 128
    tp_degree: int = 1
    ep_degree: int = 1
    use_fp16: bool = True
    reduce_layers: Optional[int] = None
    on_cpu: bool = False
    # Tensor capture configuration - supports "all", single module, or list
    modules_to_capture: Optional[Union[str, List[str]]] = None
    capture_inputs: bool = True
    max_intermediate_tensors: int = 10
    # Additional attributes for example usage
    hf_model_id: Optional[str] = None
    quantize: bool = False
    # MoE-specific optimizations
    blockwise_matmul_config: Optional[dict] = None


class DirectModelCompiler:
    """Direct class-based model compiler for NeuronX hardware."""

    def __init__(self, config: CompilationConfig):
        self.config = config
        self.setup_environment()

    @staticmethod
    def get_all_modules_from_model(model_instance) -> List[str]:
        """
        Automatically discover all capturable internal module names for tensor capture.

        This generates internal module names within the compiled XLA graphs based on the
        model architecture, not top-level PyTorch modules.

        For PhiMoE: Generates 162 internal module names (32 layers × 5 submodules + 2 top-level)
        For other models: Uses model-specific logic or config-based generation

        Returns:
            List of internal module names that can be used for tensor capture
        """
        # Check if model class provides a method to get all capturable modules
        if hasattr(model_instance, 'get_all_capturable_modules'):
            return model_instance.get_all_capturable_modules()

        # Otherwise, generate based on model config
        if not hasattr(model_instance, 'config'):
            raise ValueError(
                "Model must have 'config' attribute or 'get_all_capturable_modules()' method "
                "for automatic module discovery"
            )

        config = model_instance.config
        modules = []

        # Get model architecture info from config
        num_layers = getattr(config, 'num_hidden_layers', None)
        if num_layers is None:
            raise ValueError(
                "Model config must have 'num_hidden_layers' for automatic module discovery"
            )

        # Model-specific module generation based on architecture
        model_type = getattr(config, 'model_type', 'unknown').lower()

        if model_type == 'phimoe':
            # PhiMoE architecture: 32 layers with specific submodules
            for layer_idx in range(num_layers):
                # Capture entire layer
                modules.append(f"layers.{layer_idx}")

                # Capture layer components individually
                modules.append(f"layers.{layer_idx}.input_layernorm")
                modules.append(f"layers.{layer_idx}.self_attn")
                modules.append(f"layers.{layer_idx}.post_attention_layernorm")
                modules.append(f"layers.{layer_idx}.block_sparse_moe")

            # Top-level modules
            modules.append("norm")  # Final layer norm
            modules.append("lm_head")  # Output projection

        else:
            # Generic transformer architecture (fallback)
            # This may need customization for specific model types
            for layer_idx in range(num_layers):
                modules.append(f"layers.{layer_idx}")
                modules.append(f"layers.{layer_idx}.self_attn")
                modules.append(f"layers.{layer_idx}.mlp")

            # Common top-level modules
            if hasattr(config, 'vocab_size'):
                modules.append("lm_head")

        return modules

    def resolve_modules_to_capture(self, model_instance=None) -> Optional[List[str]]:
        """
        Resolve modules_to_capture configuration into a list of module names.

        Supports three modes:
        1. "all" - automatically discover all modules (requires model_instance)
        2. Single module name as string - convert to list
        3. List of module names - use as-is
        4. None - no tensor capture

        Args:
            model_instance: Optional model instance for "all" mode

        Returns:
            List of module names to capture, or None if no tensor capture
        """
        if self.config.modules_to_capture is None:
            return None

        # Handle "all" mode
        if isinstance(self.config.modules_to_capture, str) and self.config.modules_to_capture.lower() == "all":
            if model_instance is None:
                raise ValueError("Model instance required for modules_to_capture='all' mode")

            print(f"🔍 Discovering all capturable modules...")
            all_modules = self.get_all_modules_from_model(model_instance)
            print(f"   Found {len(all_modules)} capturable modules")
            return all_modules

        # Handle single module name
        if isinstance(self.config.modules_to_capture, str):
            print(f"📌 Capturing single module: {self.config.modules_to_capture}")
            return [self.config.modules_to_capture]

        # Handle list of modules
        if isinstance(self.config.modules_to_capture, list):
            print(f"📋 Capturing {len(self.config.modules_to_capture)} specified modules")
            return self.config.modules_to_capture

        raise ValueError(f"Invalid modules_to_capture type: {type(self.config.modules_to_capture)}")

    def setup_environment(self):
        """Set up environment variables for optimal compilation."""
        print("🔧 Setting up compilation environment...")

        # Enable XLA debugging for better visibility
        os.environ["XLA_IR_DEBUG"] = "1"
        os.environ["XLA_HLO_DEBUG"] = "1"
        os.environ["XLA_FALLBACK_CPU"] = "0"

        # Neuron runtime debugging
        os.environ["NEURON_RT_LOG_LEVEL"] = "INFO"
        os.environ["NEURON_FUSE_SOFTMAX"] = "1"

        # Enable HLO dumps (even on failure)
        os.environ["NEURON_DUMP_HLO_SNAPSHOT"] = "1"
        os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"

        # set path to compile directory with timestamp to avoid collisions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.environ["BASE_COMPILE_WORK_DIR"] = f"./agent_artifacts/data/neff_output_{timestamp}"

        # Critical workaround: Disable HLO verifier for all models
        # This prevents exit code 70 errors (HLO verifier failures) with complex models
        # See knowledge_base/Category1_Scripts_Compilation_Config_Summary.md for details
        os.environ['NEURON_CC_FLAGS'] = '--internal-hlo2tensorizer-options=--verify-hlo=false'
        print(f"✅ HLO verifier disabled (workaround for complex models)")

        # Model-specific optimizations based on model class name
        model_name = self.config.model_class.__name__.lower()
        if "gptoss" in model_name:
            # Force modular flow for GPT-OSS
            neuron_cc_flags = [
                "--model-type=transformer",
                "-O1",  # Critical: Use O1 instead of O2 to enable modular flow
                "--internal-hlo2tensorizer-options='--modular-flow-mac-threshold=10 --verify-hlo=false'",
                "--verbose=35",
                "--enable-internal-neff-wrapper"
            ]
            os.environ["NEURON_CC_FLAGS"] = " ".join(neuron_cc_flags)
            print(f"✅ Modular flow enabled for GPT-OSS: {os.environ['NEURON_CC_FLAGS']}")

        print("✅ Environment configured")
    
    def validate_model_config(self, model_config_path: str):
        """Validate model configuration for compatibility with TP degree."""
        try:
            with open(os.path.join(model_config_path, 'config.json'), 'r') as f:
                config = json.load(f)
            
            vocab_size = config.get('vocab_size', 0)
            if vocab_size > 0 and vocab_size % self.config.tp_degree != 0:
                print(f"⚠️  Warning: Vocabulary size ({vocab_size}) is not divisible by TP degree ({self.config.tp_degree})")
                print(f"   This may cause compilation errors. Consider using TP degree 1 or a divisor of {vocab_size}")
                
                # Auto-adjust TP degree if possible
                if self.config.tp_degree > 1:
                    for tp in [1, 2, 4, 8, 16, 32]:
                        if vocab_size % tp == 0 and tp <= self.config.tp_degree:
                            print(f"   💡 Suggestion: Use --tp_degree {tp} for compatibility")
                            break
        except Exception as e:
            print(f"⚠️  Could not validate model config: {e}")
    
    def create_neuron_config(self, resolved_modules: Optional[List[str]] = None) -> 'NeuronConfig':
        """
        Create NeuronConfig using the provided neuron_config_class.

        Args:
            resolved_modules: Optional resolved list of modules for tensor capture.
                            If None, will use modules_to_capture from config (if it's a list).
        """
        dtype = torch.bfloat16 if self.config.use_fp16 else torch.float32

        # Calculate world_size as TP × EP
        world_size = self.config.tp_degree * self.config.ep_degree

        neuron_config_kwargs = {
            "tp_degree": self.config.tp_degree,
            "ep_degree": self.config.ep_degree,
            "world_size": world_size,
            "batch_size": self.config.batch_size,
            "seq_len": self.config.seq_len,
            "torch_dtype": dtype,
            "save_sharded_checkpoint": True,
            "on_cpu": self.config.on_cpu,
        }

        # Add model-specific optimizations
        model_name = self.config.model_class.__name__.lower()
        if "gptoss" in model_name:
            neuron_config_kwargs["enable_cte_modular_flow"] = True
        elif "moe" in model_name:
            # For MoE models, also set moe_ep_degree to match ep_degree
            neuron_config_kwargs["moe_ep_degree"] = self.config.ep_degree
            
            # Add blockwise_matmul_config if provided
            if self.config.blockwise_matmul_config:
                neuron_config_kwargs["blockwise_matmul_config"] = self.config.blockwise_matmul_config
                print(f"   ✅ Using blockwise_matmul_config: {self.config.blockwise_matmul_config}")

        # Critical workaround: Disable HLO verifier for all models
        # Pass compiler flags to NeuronConfig to override default --verify-hlo=true
        neuron_config_kwargs["compiler_args"] = "--internal-hlo2tensorizer-options=--verify-hlo=false"

        # Determine modules to capture
        # Priority: resolved_modules > config.modules_to_capture (if list) > None
        modules_list = None
        if resolved_modules is not None:
            modules_list = resolved_modules
        elif isinstance(self.config.modules_to_capture, list):
            modules_list = self.config.modules_to_capture

        # Add tensor capture configuration if modules list is available
        if modules_list:
            print(f"🎯 Tensor Capture Enabled:")
            print(f"   Modules to capture: {len(modules_list)} modules")
            print(f"   Capture inputs: {self.config.capture_inputs}")
            print(f"   Max intermediate tensors: {self.config.max_intermediate_tensors}")

            # Import TensorCaptureConfig and OnDeviceSamplingConfig
            from neuronx_distributed_inference.models.config import (
                TensorCaptureConfig,
                OnDeviceSamplingConfig,
            )

            # Create TensorCaptureConfig
            tensor_capture_config = TensorCaptureConfig(
                modules_to_capture=modules_list,
                capture_inputs=self.config.capture_inputs,
                max_intermediate_tensors=self.config.max_intermediate_tensors,
            )

            # Tensor capture requires OnDeviceSamplingConfig
            on_device_sampling_config = OnDeviceSamplingConfig()

            # Add to neuron_config_kwargs
            neuron_config_kwargs["tensor_capture_config"] = tensor_capture_config
            neuron_config_kwargs["on_device_sampling_config"] = on_device_sampling_config

            print(f"   ✅ Tensor capture configuration created")
            print(f"   Expected tensors per step: {len(modules_list) * (2 if self.config.capture_inputs else 1)}")

        return self.config.neuron_config_class(**neuron_config_kwargs)
    
    def compile(self) -> bool:
        """Main compilation method."""
        print("🚀 Starting Direct Model Compilation")
        print("=" * 60)
        
        # Validate model path
        if not Path(self.config.model_path).exists():
            print(f"❌ Model path does not exist: {self.config.model_path}")
            return False
        
        print(f"✅ Using model path: {self.config.model_path}")
        
        # Validate model configuration
        self.validate_model_config(self.config.model_path)
        
        # Create output directory
        Path(self.config.output_path).mkdir(parents=True, exist_ok=True)
        
        # Create neuron config
        neuron_config = self.create_neuron_config()
        
        print(f"✅ Created NeuronConfig for {self.config.model_class.__name__}:")
        print(f"  TP degree: {neuron_config.tp_degree}")
        print(f"  Batch size: {neuron_config.batch_size}")
        print(f"  Sequence length: {neuron_config.seq_len}")
        print(f"  Data type: {neuron_config.torch_dtype}")
        
        # Apply layer reduction if requested
        if self.config.reduce_layers is not None:
            print(f"🔧 Layer reduction requested: {self.config.reduce_layers} layers")
        
        # Compile using direct model classes
        print(f"⚙️  Starting compilation for {self.config.model_class.__name__}...")
        success = self.compile_model(neuron_config)
        
        if success:
            self.verify_compilation()
            
            # Save neuron_config.json to all MODULE_{HASH} directories in compile cache
            self.save_neuron_config_to_module_dirs()
            
            print(f"\n🎉 SUCCESS: {self.config.model_class.__name__} model compilation completed!")
            print(f"📁 Output: {self.config.output_path}")
            print(f"🚀 Ready for NeuronX distributed inference!")
        else:
            # Compilation failed - save HLO artifacts for debugging
            print(f"\n❌ FAILURE: {self.config.model_class.__name__} model compilation failed!")
            self.save_hlo_artifacts_on_failure()
        
        return success
    
    def to_dict(self, obj, _seen=None):
        """
        Convert an object to a dictionary for JSON serialization.
        This follows the same pattern as NeuronxDistributedInference's to_dict function.
        """
        import inspect
        
        # Track seen objects to avoid circular references
        if _seen is None:
            _seen = set()
        
        # Check for circular references
        obj_id = id(obj)
        if obj_id in _seen:
            return None
        
        if type(obj) is dict:
            _seen.add(obj_id)
            result = {k: self.to_dict(v, _seen) for k, v in obj.items()}
            _seen.remove(obj_id)
            return result
        elif type(obj) is list:
            return [self.to_dict(v, _seen) for v in obj]
        elif inspect.isclass(obj):
            return {
                "__module__": obj.__module__,
                "__name__": obj.__name__,
            }
        elif hasattr(obj, '__dict__'):
            _seen.add(obj_id)
            result = {k: self.to_dict(v, _seen) for k, v in obj.__dict__.items()}
            _seen.remove(obj_id)
            return result
        elif type(obj) is torch.dtype:
            return str(obj).split(".")[1]
        else:
            return obj
    
    def save_neuron_config(self, neuron_config):
        """Save neuron_config to JSON file in output directory and HLO/NEFF directory."""
        try:
            # Convert neuron_config to dictionary using the same pattern as NeuronxDistributedInference
            config_dict = self.to_dict(neuron_config)

            # Save to output directory (for inference)
            output_path = Path(self.config.output_path)
            output_config_file = output_path / "neuron_config.json"
            with open(output_config_file, 'w') as f:
                json.dump(config_dict, f, indent=2, sort_keys=True)
            print(f"✅ Saved neuron_config.json to {output_config_file}")

            # Save to HLO/NEFF directory (for traceability with compilation artifacts)
            compile_work_dir = Path(os.environ.get("BASE_COMPILE_WORK_DIR", "./agent_artifacts/data/neff_output"))
            compile_work_dir.mkdir(parents=True, exist_ok=True)
            compile_config_file = compile_work_dir / "neuron_config.json"
            with open(compile_config_file, 'w') as f:
                json.dump(config_dict, f, indent=2, sort_keys=True)
            print(f"✅ Saved neuron_config.json to {compile_config_file} (alongside HLO/NEFF artifacts)")

            # Store config_dict for later use in save_neuron_config_to_module_dirs
            self._neuron_config_dict = config_dict

        except Exception as e:
            print(f"⚠️  Warning: Could not save neuron_config.json: {e}")
            import traceback
            traceback.print_exc()

    def save_neuron_config_to_module_dirs(self):
        """
        Save neuron_config.json to all MODULE_{HASH} directories in the compile cache.
        
        This should be called after compilation completes, when all MODULE_{HASH} directories
        have been created by the compiler.
        """
        if not hasattr(self, '_neuron_config_dict'):
            print("⚠️  Warning: neuron_config not available for saving to module directories")
            return

        try:
            # Get compile cache directory
            compile_cache_dir = Path(os.environ.get("NEURON_COMPILE_CACHE_URL", "/var/tmp/neuron-compile-cache"))
            
            if not compile_cache_dir.exists():
                print(f"⚠️  Compile cache directory does not exist: {compile_cache_dir}")
                return

            # Find all MODULE_{HASH} directories
            # Pattern: neuron-compile-cache/neuronxcc-{version}/MODULE_{hash}/
            module_dirs = []
            for version_dir in compile_cache_dir.iterdir():
                if version_dir.is_dir() and version_dir.name.startswith('neuronxcc-'):
                    for module_dir in version_dir.iterdir():
                        if module_dir.is_dir() and module_dir.name.startswith('MODULE_'):
                            module_dirs.append(module_dir)

            if not module_dirs:
                print(f"⚠️  No MODULE_{{HASH}} directories found in {compile_cache_dir}")
                return

            print(f"\n🔍 Found {len(module_dirs)} MODULE_{{HASH}} directories")
            
            # Save neuron_config.json to each MODULE_{HASH} directory
            saved_count = 0
            for module_dir in module_dirs:
                config_file = module_dir / "neuron_config.json"
                with open(config_file, 'w') as f:
                    json.dump(self._neuron_config_dict, f, indent=2, sort_keys=True)
                saved_count += 1
                print(f"   ✅ Saved to {module_dir.name}/neuron_config.json")

            print(f"✅ Saved neuron_config.json to {saved_count} MODULE_{{HASH}} directories")

        except Exception as e:
            print(f"⚠️  Warning: Could not save neuron_config.json to module directories: {e}")
            import traceback
            traceback.print_exc()

    def save_hlo_artifacts_on_failure(self):
        """Save HLO artifacts and metadata when compilation fails."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.config.model_class.__name__
            failure_dir = Path(f"./agent_artifacts/data/compilation_failures/{model_name}_{timestamp}")
            failure_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy HLO artifacts from BASE_COMPILE_WORK_DIR
            compile_work_dir = Path(os.environ.get("BASE_COMPILE_WORK_DIR", "./agent_artifacts/data/neff_output"))
            if compile_work_dir.exists():
                shutil.copytree(compile_work_dir, failure_dir / "hlo_artifacts", dirs_exist_ok=True)
                print(f"💾 Saved HLO artifacts to: {failure_dir / 'hlo_artifacts'}")
            else:
                print(f"⚠️  Compile work directory not found: {compile_work_dir}")
            
            # Save compilation config for debugging
            config_info = {
                "timestamp": timestamp,
                "model_name": model_name,
                "model_path": self.config.model_path,
                "output_path": self.config.output_path,
                "tp_degree": self.config.tp_degree,
                "ep_degree": self.config.ep_degree,
                "batch_size": self.config.batch_size,
                "seq_len": self.config.seq_len,
                "use_fp16": self.config.use_fp16,
                "reduce_layers": self.config.reduce_layers,
                "on_cpu": self.config.on_cpu,
                "modules_to_capture": self.config.modules_to_capture,
            }
            
            with open(failure_dir / "compilation_config.json", 'w') as f:
                json.dump(config_info, f, indent=2)
            
            print(f"📋 Saved compilation config to: {failure_dir / 'compilation_config.json'}")
            print(f"🔍 Review HLO files for debugging at: {failure_dir}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not save HLO artifacts on failure: {e}")
            import traceback
            traceback.print_exc()

    def save_tensor_capture_config(self, resolved_modules: List[str]):
        """
        Save tensor capture configuration to JSON file for inference.

        Args:
            resolved_modules: The actual list of modules that will be captured
        """
        output_path = Path(self.config.output_path)
        config_file = output_path / "tensor_capture_config.json"

        try:
            tensor_capture_config = {
                "modules_to_capture": resolved_modules,
                "capture_inputs": self.config.capture_inputs,
                "max_intermediate_tensors": self.config.max_intermediate_tensors,
            }

            with open(config_file, 'w') as f:
                json.dump(tensor_capture_config, f, indent=2)

            print(f"✅ Saved tensor_capture_config.json to {config_file}")
            print(f"   This file will be loaded during inference to enable tensor capture")

        except Exception as e:
            print(f"⚠️  Warning: Could not save tensor_capture_config.json: {e}")
            import traceback
            traceback.print_exc()
    
    def compile_model(self, neuron_config) -> bool:
        """Compile the model using the provided classes."""
        print(f"🚀 Compiling {self.config.model_class.__name__}...")

        try:
            # Load model configuration
            model_config = self.config.config_class.from_pretrained(
                self.config.model_path,
                neuron_config=neuron_config
            )

            # Apply layer reduction if requested
            if self.config.reduce_layers is not None:
                original_layers = model_config.num_hidden_layers
                model_config.num_hidden_layers = self.config.reduce_layers
                print(f"🔧 Reduced layers from {original_layers} to {model_config.num_hidden_layers}")

            print(f"✅ Model configuration loaded:")
            print(f"  Hidden size: {model_config.hidden_size}")
            print(f"  Attention heads: {model_config.num_attention_heads}")
            print(f"  Hidden layers: {model_config.num_hidden_layers}")
            print(f"  Vocabulary size: {model_config.vocab_size}")
            if hasattr(model_config, 'sliding_window'):
                print(f"  Sliding window: {model_config.sliding_window}")
            if hasattr(model_config, 'num_local_experts'):
                print(f"  Local experts: {model_config.num_local_experts}")

            # Create model instance first (allows model to configure neuron_config)
            model = self.config.model_class(
                model_path=self.config.model_path,
                config=model_config
            )

            # Check if we need to resolve modules for tensor capture
            # This is needed for "all" mode or single module string
            resolved_modules = None
            if self.config.modules_to_capture:
                if isinstance(self.config.modules_to_capture, str):
                    # Need to resolve - either "all" or single module name
                    resolved_modules = self.resolve_modules_to_capture(model)

                    # Recreate neuron_config with resolved modules
                    print(f"\n🔄 Updating neuron config with resolved modules...")
                    updated_neuron_config = self.create_neuron_config(resolved_modules)

                    # Update model_config.neuron_config
                    model_config.neuron_config = updated_neuron_config
                    print(f"   ✅ Neuron config updated with {len(resolved_modules) if resolved_modules else 0} modules")
                elif isinstance(self.config.modules_to_capture, list):
                    # Already a list, use it directly
                    resolved_modules = self.config.modules_to_capture

            # Save neuron_config.json AFTER model initialization and any updates
            # This ensures any model-specific config modifications (like router_config)
            # and tensor capture config are captured in the saved config
            # Use model_config.neuron_config to ensure we save the modified version
            self.save_neuron_config(model_config.neuron_config)

            # Compile model
            model.compile(self.config.output_path)

            # Save tensor capture configuration if enabled
            if resolved_modules:
                self.save_tensor_capture_config(resolved_modules)

            return True

        except Exception as e:
            print(f"❌ Model compilation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def verify_compilation(self):
        """Verify compilation artifacts."""
        output_path = Path(self.config.output_path)
        
        if (output_path / "model.pt").exists():
            print("✅ Compilation artifacts verified")
            
            # Check for sharded checkpoints
            weights_dir = output_path / "weights"
            if weights_dir.exists():
                sharded_files = list(weights_dir.glob("*_sharded_checkpoint.safetensors"))
                if sharded_files:
                    print(f"✅ Weight sharding completed: {len(sharded_files)} files")
                else:
                    print("⚠️  No sharded checkpoint files found")
        else:
            print("⚠️  Warning: Expected compilation artifacts not found")


# No CLI interface - use Python API only