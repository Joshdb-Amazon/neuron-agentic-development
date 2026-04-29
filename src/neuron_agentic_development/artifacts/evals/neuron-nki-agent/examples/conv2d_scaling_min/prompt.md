Write a NKI kernel that runs on a single neuron core for the nki-dev-suite/conv2d_scaling_min.py, make sure to run it on device and compare with torch CPU baseline to ensure correctness. Use the neuron-nki-writing skill. Also, use the neuron-nki-docs skill to look up NKI docs for API usage, architecture constraints, etc.
- When testing the kernel, reference the neuron-nki-debugging skill and use the venv at `/opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/` and following the `## Testing Workflow` section. 
default_nc_version is gen3. 
Please create a ./tmp/conv2d_scaling_min folder to save all the any intermediate test file or script generated.
Also document necessary finding during the process. 


## Testing Workflow
1. Create PyTorch reference implementation
2. Create test with small dimensions first (fits in single tiles). Test inputs are scaled by 1/sqrt(input_size) to prevent float16 overflow in large matmul accumulations.
3. Validate the accuracy with metrics (cosine sim, mean rel diff, max abs diff)
4. Scale up dimensions to test tiling for the full size in the original reference pytorch
