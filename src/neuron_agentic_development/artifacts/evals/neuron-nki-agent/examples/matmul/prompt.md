Write a NKI kernel that runs on a single neuron core for the matmul_with_BT.py, make sure to run it on device and compare with torch CPU baseline to ensure correctness. Use the neuron-nki-writing skill. Also, use the neuron-nki-docs skill to look up NKI docs for API usage, architecture constraints, etc.
- When testing the kernel, reference the neuron-nki-debugging skill and use the venv at `/opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/`.
default_nc_version is gen3. 
- Please create a ./tmp/matmul folder to save all the any intermediate test file or script generated.
- Validate the accuracy with metrics (cosine sim, mean rel diff, max abs diff). Test inputs are scaled by 1/sqrt(input_size) to prevent float16 overflow in large matmul accumulations.
- Also document necessary finding during the process.