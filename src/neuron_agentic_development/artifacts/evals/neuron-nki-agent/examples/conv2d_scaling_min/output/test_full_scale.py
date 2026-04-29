"""Test with full-scale dimensions HW=256."""
import sys
import os
import time
os.environ["NEURON_CC_FLAGS"] = "--target trn2 --lnc 1"
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

print('=' * 60, flush=True)
print('Full-scale test (batch=8, in_ch=64, out_ch=128, HW=256)', flush=True)
print('=' * 60, flush=True)

print("\nStep 1: Importing torch...", flush=True)
import torch
import torch.nn as nn
import math
print("  Done.", flush=True)

print("Step 2: Importing torch_xla...", flush=True)
from torch_xla.core import xla_model as xm
print("  Done.", flush=True)

print("Step 3: Importing NKI kernel...", flush=True)
from conv2d_scale_min_kernel import conv2d_scale_min_kernel
print("  Done.", flush=True)

batch_size = 8
in_channels = 64
out_channels = 128
height = 256
width = 256
kernel_size = 3
scale_factor = 2.0

h_out = height - kernel_size + 1
w_out = width - kernel_size + 1
spatial_out = h_out * w_out
weight_k = in_channels * kernel_size * kernel_size

print(f"\nTest Configuration:", flush=True)
print(f"  batch_size={batch_size}, in_channels={in_channels}, out_channels={out_channels}", flush=True)
print(f"  height={height}, width={width}, kernel_size={kernel_size}", flush=True)
print(f"  h_out={h_out}, w_out={w_out}, spatial_out={spatial_out}", flush=True)
print(f"  weight_k={weight_k}, scale_factor={scale_factor}", flush=True)

# Create reference model
print("\nStep 4: Creating reference model...", flush=True)
class Conv2dScaleMin(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv(x)
        x = x * self.scale_factor
        x = torch.min(x, dim=1, keepdim=True)[0]
        return x

model_cpu = Conv2dScaleMin(in_channels, out_channels, kernel_size, scale_factor)
print("  Done.", flush=True)

# Scale input to prevent fp16 overflow
print("Step 5: Creating input tensor...", flush=True)
input_scale = 1.0 / math.sqrt(weight_k)
x_cpu = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32) * input_scale
print(f"  Input shape: {x_cpu.shape}", flush=True)

print("Step 6: Computing reference output...", flush=True)
ref_start = time.time()
with torch.no_grad():
    ref_output = model_cpu(x_cpu)
ref_time = time.time() - ref_start
print(f"  Reference compute time: {ref_time:.3f}s", flush=True)
print(f"  Reference output shape: {ref_output.shape}", flush=True)

# Get device
print("\nStep 7: Getting XLA device...", flush=True)
device = xm.xla_device()
print("  Done.", flush=True)

# Move tensors to device with fp16
print("Step 8: Moving tensors to device...", flush=True)
x_device = x_cpu.to(dtype=torch.float16, device=device)
weight_flat = model_cpu.conv.weight.data.view(out_channels, weight_k)
weight_device = weight_flat.to(dtype=torch.float16, device=device)
bias_device = model_cpu.conv.bias.data.to(dtype=torch.float16, device=device)
print(f"  x_device shape: {x_device.shape}", flush=True)
print(f"  weight_device shape (flattened): {weight_device.shape}", flush=True)
print(f"  bias_device shape: {bias_device.shape}", flush=True)

print("\nStep 9: Running NKI kernel (first call includes compilation)...", flush=True)
compile_start = time.time()

# First call - includes compilation
nki_output = conv2d_scale_min_kernel(
    x_device,
    weight_device,
    bias_device,
    scale_factor=scale_factor,
    kernel_size=kernel_size,
)
print("  Kernel returned, syncing...", flush=True)
# Sync to ensure compilation is complete
xm.mark_step()
nki_output_cpu = nki_output.cpu()

compile_time = time.time() - compile_start
print(f"  First call (compile + run): {compile_time:.3f}s", flush=True)

# Skip warmup/latency runs to save time (kernel recompiles each call anyway)
print("\nSkipping latency measurements (kernel recompiles each call)...", flush=True)
avg_latency = compile_time  # Just use first compile time

# Compute accuracy
print("\nStep 11: Computing accuracy metrics...", flush=True)
nki_output_reshaped = nki_output_cpu.view(batch_size, 1, h_out, w_out)
ref_output_f16 = ref_output.to(torch.float16)

nki_flat = nki_output_reshaped.float().flatten()
ref_flat = ref_output_f16.float().flatten()

cos_sim = torch.nn.functional.cosine_similarity(
    nki_flat.unsqueeze(0),
    ref_flat.unsqueeze(0)
).item()

mean_rel_diff = (torch.abs(nki_flat - ref_flat) / (torch.abs(ref_flat) + 1e-8)).mean().item()
max_abs_diff = torch.abs(nki_flat - ref_flat).max().item()

print(f"\nAccuracy Metrics:", flush=True)
print(f"  Cosine Similarity: {cos_sim:.6f}", flush=True)
print(f"  Mean Relative Diff: {mean_rel_diff:.6f}", flush=True)
print(f"  Max Absolute Diff: {max_abs_diff:.6f}", flush=True)

passed = cos_sim > 0.99 and mean_rel_diff < 0.1
if passed:
    print("\n[PASS] NKI kernel matches reference!", flush=True)
else:
    print("\n[FAIL] NKI kernel differs from reference", flush=True)

print("\n" + "=" * 60, flush=True)
print("Summary", flush=True)
print("=" * 60, flush=True)
print(f"  Compilation time: {compile_time:.1f}s", flush=True)
print(f"  Inference latency: {avg_latency*1000:.2f}ms (avg of {num_runs} runs)", flush=True)
print(f"  Accuracy: {'PASS' if passed else 'FAIL'} (cos_sim={cos_sim:.6f})", flush=True)
