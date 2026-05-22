#!/usr/bin/env python3
"""
Example showing how to use the refactored model compiler for Phi3
Includes complete workflow: download, compile, and test.
"""

import sys
import os

# Add the source directory to Python path
sys.path.insert(0, './NeuroborosFoundations/src')

def example_phi3_full_workflow():
    """Example: Complete Phi3 workflow - download and compile."""
    print("📋 Example: Complete Phi3 Workflow (Download + Compile)")
    print("=" * 70)

    # Import the required classes and utilities
    
    from amzn.neuron.neuroboros.models.phi3.modeling_phi3 import NeuronPhi3ForCausalLM, Phi3InferenceConfig
    from neuronx_distributed_inference.models.config import NeuronConfig
    from amzn.neuron.neuroboros.utils import DirectModelCompiler, CompilationConfig, download_model_weights
    
    # Configuration
    current_dir = os.getcwd()
    hf_model_id = "microsoft/Phi-3-mini-4k-instruct"
    model_path = os.path.join(current_dir, "agent_artifacts/data/Phi-3-mini-4k-instruct")
    compiled_output_path = os.path.join(current_dir, "agent_artifacts/data/phi3-mini-4k-compiled")
    
    print("🎯 Workflow Configuration:")
    print(f"  HuggingFace Model: {hf_model_id}")
    print(f"  Download Path: {model_path}")
    print(f"  Compiled Output: {compiled_output_path}")
    print()
    
    try:
        # Step 1: Download model weights
        print("📥 Step 1: Download Model Weights")
        print("-" * 40)
        downloaded_path = download_model_weights(hf_model_id, model_path)
        print(f"✅ Model downloaded to: {downloaded_path}")
        print()
        
        # Step 2: Create compilation configuration
        print("⚙️  Step 2: Configure Compilation")
        print("-" * 40)
        config = CompilationConfig(
            model_class=NeuronPhi3ForCausalLM,
            config_class=Phi3InferenceConfig,
            neuron_config_class=NeuronConfig,
            model_path=downloaded_path,
            output_path=compiled_output_path,
            batch_size=1,
            seq_len=128,
            tp_degree=1,  # Start with TP=1 for Phi3
            use_fp16=True,
        )
        
        print("Configuration:")
        print(f"  Model Class: {config.model_class.__name__}")
        print(f"  Config Class: {config.config_class.__name__}")
        print(f"  Model Path: {config.model_path}")
        print(f"  Output Path: {config.output_path}")
        print(f"  TP Degree: {config.tp_degree}")
        print(f"  Sequence Length: {config.seq_len}")
        print(f"  Use FP16: {config.use_fp16}")
        print()
        
        # Step 3: Compile model
        print("🚀 Step 3: Compile Model")
        print("-" * 40)
        compiler = DirectModelCompiler(config)
        success = compiler.compile()
        
        if success:
            print("\n🎉 COMPLETE WORKFLOW SUCCESS!")
            print("=" * 70)
            print("✅ All steps completed successfully:")
            print("  1. ✅ Model downloaded from HuggingFace")
            print("  2. ✅ Model compiled for NeuronX")
            print()
            print(f"📁 Compiled model ready at: {compiled_output_path}")
            print()
            print("🚀 Ready for inference! Use:")
            print(f"python src/amzn/neuron/neuroboros/utils/run_inference.py \\")
            print(f"  --model_class NeuronPhi3ForCausalLM \\")
            print(f"  --config_class Phi3InferenceConfig \\")
            print(f"  --model_path {downloaded_path} \\")
            print(f"  --compiled_path {compiled_output_path} \\")
            print(f"  --prompt 'The meaning of life is'")
            
            return True
        else:
            print("\n❌ Compilation failed!")
            return False
            
    except Exception as e:
        print(f"\n💥 Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def example_phi3_download_and_compile():
    """Example: Download Phi3 model and compile it using direct class imports."""
    print("📋 Example: Download and Compile Phi3 Model (Direct Classes)")
    print("=" * 60)
    
    from amzn.neuron.neuroboros.models.phi3.modeling_phi3 import NeuronPhi3ForCausalLM, Phi3InferenceConfig
    from neuronx_distributed_inference.models.config import NeuronConfig
    from amzn.neuron.neuroboros.utils import DirectModelCompiler, CompilationConfig, download_model_weights
    
    # Step 1: Download model first
    current_dir = os.getcwd()
    hf_model_id = "microsoft/Phi-3-mini-4k-instruct"
    model_path = os.path.join(current_dir, "agent_artifacts/data/Phi-3-mini-4k-instruct")

    print(f"📥 Downloading model from: {hf_model_id}")
    downloaded_path = download_model_weights(hf_model_id, model_path)
    print(f"✅ Model downloaded to: {downloaded_path}")

    # Step 2: Create configuration with direct class references
    config = CompilationConfig(
        model_class=NeuronPhi3ForCausalLM,       # Direct model class
        config_class=Phi3InferenceConfig,        # Direct config class
        neuron_config_class=NeuronConfig,        # Direct neuron config class
        model_path=downloaded_path,              # Use downloaded model path
        output_path=os.path.join(current_dir, "agent_artifacts/data/phi3-mini-4k-compiled"),  # Where to save compiled artifacts
        batch_size=1,
        seq_len=128,
        tp_degree=1,                             # Start with TP=1 for simplicity
        use_fp16=True,                           # Use bfloat16 for memory efficiency
    )
    
    print("Configuration:")
    print(f"  Model class: {config.model_class.__name__}")
    print(f"  Config class: {config.config_class.__name__}")
    print(f"  NeuronConfig class: {config.neuron_config_class.__name__}")
    print(f"  Model path: {config.model_path}")
    print(f"  Save model to: {config.model_path}")
    print(f"  Save compiled to: {config.output_path}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Sequence length: {config.seq_len}")
    print(f"  TP degree: {config.tp_degree}")
    print(f"  Use FP16: {config.use_fp16}")
    
    # Step 2: Create compiler
    print("\n🔧 Creating compiler...")
    compiler = DirectModelCompiler(config)
    
    # Step 3: Run compilation
    print("\n🚀 Starting compilation...")
    print("This will:")
    print("  1. Download Phi3 model from HuggingFace")
    print("  2. Load model using NeuronPhi3ForCausalLM")
    print("  3. Configure using Phi3InferenceConfig")
    print("  4. Create NeuronConfig for compilation")
    print("  5. Compile for NeuronX hardware")
    print("  6. Save compiled artifacts")
    
    success = compiler.compile()
    
    if success:
        print("\n🎉 SUCCESS!")
        print(f"Model compiled and ready at: {config.output_path}")
        
        # Show how to run inference
        print("\n📝 To run inference:")
        print(f"python3 src/amzn/neuron/neuroboros/models/phi3/run_inference.py \\")
        print(f"  --model_path {config.output_path} \\")
        print(f"  --hf_model_path {config.model_path} \\")
        print(f"  --prompt 'The meaning of life is' \\")
        print(f"  --max_new_tokens 50")
    else:
        print("\n❌ Compilation failed!")
    
    return success

def example_phi3_compile_existing():
    """Example: Compile existing Phi3 model (no download) using direct classes."""
    print("📋 Example: Compile Existing Phi3 Model (Direct Classes)")
    print("=" * 60)
    
    from amzn.neuron.neuroboros.models.phi3.modeling_phi3 import NeuronPhi3ForCausalLM, Phi3InferenceConfig
    from neuronx_distributed_inference.models.config import NeuronConfig
    from amzn.neuron.neuroboros.utils import DirectModelCompiler, CompilationConfig
    
    # Check if model exists
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, "agent_artifacts/data/Phi-3-mini-4k-instruct")
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Run example_phi3_download_and_compile() first")
        return False

    # Configuration for existing model with direct classes
    config = CompilationConfig(
        model_class=NeuronPhi3ForCausalLM,       # Direct model class
        config_class=Phi3InferenceConfig,        # Direct config class
        neuron_config_class=NeuronConfig,        # Direct neuron config class
        model_path=model_path,
        output_path=os.path.join(current_dir, "agent_artifacts/data/phi3-mini-4k-compiled-256"),
        batch_size=1,
        seq_len=256,                    # Longer sequence
        tp_degree=1,
        use_fp16=True,
    )
    
    print("Configuration:")
    print(f"  Model class: {config.model_class.__name__}")
    print(f"  Config class: {config.config_class.__name__}")
    print(f"  Model path: {config.model_path}")
    print(f"  Output path: {config.output_path}")
    print(f"  Sequence length: {config.seq_len}")
    
    # Create and run compiler
    compiler = DirectModelCompiler(config)
    success = compiler.compile()
    
    return success

def show_python_api_usage():
    """Show how to use the Python API with direct classes."""
    print("📋 Python API Usage Examples (Direct Classes)")
    print("=" * 60)
    
    print("1. Complete workflow - download and compile Phi3:")
    print("""
from amzn.neuron.neuroboros.models.phi3.modeling_phi3 import NeuronPhi3ForCausalLM, Phi3InferenceConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from amzn.neuron.neuroboros.utils import DirectModelCompiler, CompilationConfig, download_model_weights

# Step 1: Download model
model_path = download_model_weights("microsoft/Phi-3-mini-4k-instruct", "./Phi-3-mini-4k-instruct")

# Step 2: Configure compilation
config = CompilationConfig(
    model_class=NeuronPhi3ForCausalLM,
    config_class=Phi3InferenceConfig,
    neuron_config_class=NeuronConfig,
    model_path=model_path,
    output_path="./phi3_compiled",
    batch_size=1,
    seq_len=128,
    tp_degree=1,
    use_fp16=True
)

# Step 3: Compile
compiler = DirectModelCompiler(config)
success = compiler.compile()
""")
    
    print("\n2. Compile existing model:")
    print("""
config = CompilationConfig(
    model_class=NeuronPhi3ForCausalLM,
    config_class=Phi3InferenceConfig,
    neuron_config_class=NeuronConfig,
    model_path="./Phi-3-mini-4k-instruct",
    output_path="./phi3_compiled_existing",
    batch_size=1,
    seq_len=256,
    tp_degree=1,
    use_fp16=True
)
""")
    
    print("\n3. Run inference:")
    print("""
# Use the generic inference utility
python src/amzn/neuron/neuroboros/utils/run_inference.py \\
  --model_class NeuronPhi3ForCausalLM \\
  --config_class Phi3InferenceConfig \\
  --model_path ./Phi-3-mini-4k-instruct \\
  --compiled_path ./phi3_compiled \\
  --prompt 'The meaning of life is' \\
  --max_new_tokens 50
""")
    
    print("\n📝 Note: Uses refactored architecture with separated concerns")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phi3 Model Compiler Examples")
    parser.add_argument("--example", choices=["full", "download", "existing", "api"], 
                       default="api", help="Which example to run")
    
    args = parser.parse_args()
    
    if args.example == "full":
        print("🚀 Running complete workflow example...")
        success = example_phi3_full_workflow()
    elif args.example == "download":
        print("🚀 Running download and compile example...")
        success = example_phi3_download_and_compile()
    elif args.example == "existing":
        print("�  Running compile existing model example...")
        success = example_phi3_compile_existing()
    else:
        print("📋 Showing Python API usage examples...")
        show_python_api_usage()
        success = True
    
    if success:
        print("\n✅ Example completed successfully!")
    else:
        print("\n❌ Example failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)