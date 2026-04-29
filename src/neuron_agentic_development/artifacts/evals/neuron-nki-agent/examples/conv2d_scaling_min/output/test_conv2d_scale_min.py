"""
Test harness for conv2d + scale + min NKI kernel.

Workflow:
1. Create PyTorch reference implementation
2. Test with small dimensions first
3. Validate accuracy with multiple metrics
4. Scale up to full dimensions
"""

import os
import torch
import torch.nn as nn
import math

# Set environment variables for gen3 (trn2)
os.environ["NEURON_CC_FLAGS"] = "--target trn2 --lnc 1"
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

from torch_xla.core import xla_model as xm

# Import the NKI kernel
from conv2d_scale_min_kernel import conv2d_scale_min_kernel


# ============================================================================
# Host-side Utilities
# ============================================================================

def compute_metrics(output_nki, output_ref):
    """
    Compute accuracy metrics between NKI output and reference.

    Returns dict with:
        - cosine_similarity: angle between vectors (1.0 = identical direction)
        - mean_relative_diff: average relative error
        - max_absolute_diff: worst-case absolute error
        - diff_norm: L2 norm of difference
    """
    nki_cpu = output_nki.cpu().float()
    ref_cpu = output_ref.cpu().float()

    nki_flat = nki_cpu.flatten()
    ref_flat = ref_cpu.flatten()

    cos_sim = torch.nn.functional.cosine_similarity(
        nki_flat.unsqueeze(0),
        ref_flat.unsqueeze(0)
    ).item()

    mean_rel_diff = (torch.abs(nki_flat - ref_flat) / (torch.abs(ref_flat) + 1e-8)).mean().item()
    max_abs_diff = torch.abs(nki_flat - ref_flat).max().item()
    diff_norm = torch.norm(nki_flat - ref_flat).item()

    return {
        'cosine_similarity': cos_sim,
        'mean_relative_diff': mean_rel_diff,
        'max_absolute_diff': max_abs_diff,
        'diff_norm': diff_norm
    }


# ============================================================================
# Reference Implementation
# ============================================================================

class Conv2dScaleMin(nn.Module):
    """PyTorch reference: Conv2d -> Scale -> Min(dim=1)"""

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv(x)
        x = x * self.scale_factor
        x = torch.min(x, dim=1, keepdim=True)[0]
        return x


# ============================================================================
# Test Functions
# ============================================================================

def run_test(batch_size, in_channels, out_channels, height, width, kernel_size, scale_factor):
    """
    Run a single test case.

    Returns True if test passes, False otherwise.
    """
    h_out = height - kernel_size + 1
    w_out = width - kernel_size + 1
    spatial_out = h_out * w_out
    weight_k = in_channels * kernel_size * kernel_size

    print(f"\nTest Configuration:")
    print(f"  batch_size={batch_size}, in_channels={in_channels}, out_channels={out_channels}")
    print(f"  height={height}, width={width}, kernel_size={kernel_size}")
    print(f"  h_out={h_out}, w_out={w_out}, spatial_out={spatial_out}")
    print(f"  weight_k={weight_k}, scale_factor={scale_factor}")

    # Create reference model
    model_cpu = Conv2dScaleMin(in_channels, out_channels, kernel_size, scale_factor)

    # Scale input to prevent fp16 overflow
    input_scale = 1.0 / math.sqrt(weight_k)
    x_cpu = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32) * input_scale

    # Compute reference output
    with torch.no_grad():
        ref_output = model_cpu(x_cpu)

    print(f"  Reference output shape: {ref_output.shape}")

    # Get device
    device = xm.xla_device()

    # Move tensors to device with fp16
    # x_input: [B, C_in, H, W] - original layout
    # weight_flat: [C_out, C_in * k * k] - flattened on host (cheap)
    # bias: [C_out]
    x_device = x_cpu.to(dtype=torch.float16, device=device)

    # Flatten weight on host: [C_out, C_in, k, k] -> [C_out, C_in * k * k]
    weight_flat = model_cpu.conv.weight.data.view(out_channels, weight_k)
    weight_device = weight_flat.to(dtype=torch.float16, device=device)

    bias_device = model_cpu.conv.bias.data.to(dtype=torch.float16, device=device)

    print(f"  x_device shape: {x_device.shape}")
    print(f"  weight_device shape (flattened): {weight_device.shape}")
    print(f"  bias_device shape: {bias_device.shape}")

    print("Running NKI kernel...")

    # Run NKI kernel (no im2col - kernel handles convolution internally)
    nki_output = conv2d_scale_min_kernel(
        x_device,
        weight_device,
        bias_device,
        scale_factor=scale_factor,
        kernel_size=kernel_size,
    )

    print(f"  NKI output shape: {nki_output.shape}")

    # Reshape NKI output from [B, spatial_out] to [B, 1, H_out, W_out]
    nki_output_reshaped = nki_output.view(batch_size, 1, h_out, w_out)
    print(f"  NKI output reshaped: {nki_output_reshaped.shape}")

    # Compute metrics
    ref_output_f16 = ref_output.to(torch.float16)
    metrics = compute_metrics(nki_output_reshaped.cpu(), ref_output_f16)

    print("\nAccuracy Metrics:")
    print(f"  Cosine Similarity: {metrics['cosine_similarity']:.6f}")
    print(f"  Mean Relative Diff: {metrics['mean_relative_diff']:.6f}")
    print(f"  Max Absolute Diff: {metrics['max_absolute_diff']:.6f}")
    print(f"  Diff Norm: {metrics['diff_norm']:.6f}")

    # Determine pass/fail
    passed = metrics['cosine_similarity'] > 0.99 and metrics['mean_relative_diff'] < 0.1

    if passed:
        print("\n[PASS] NKI kernel matches reference!")
    else:
        print("\n[FAIL] NKI kernel differs from reference")
        print("NKI output (first 5):", nki_output_reshaped.cpu().flatten()[:5])
        print("Reference (first 5):", ref_output_f16.flatten()[:5])

    return passed


def main():
    """Run test suite."""
    print("=" * 60)
    print("Conv2d + Scale + Min NKI Kernel Test Suite")
    print("(No im2col - kernel handles convolution internally)")
    print("=" * 60)

    # Test 1: Small dimensions (single tile)
    print("\n" + "=" * 60)
    print("Test 1: Small dimensions (single tile)")
    print("=" * 60)
    test1_passed = run_test(
        batch_size=1,
        in_channels=2,
        out_channels=4,
        height=8,
        width=8,
        kernel_size=3,
        scale_factor=2.0
    )

    # Test 2: Medium dimensions (multiple batches)
    print("\n" + "=" * 60)
    print("Test 2: Medium dimensions")
    print("=" * 60)
    test2_passed = run_test(
        batch_size=2,
        in_channels=4,
        out_channels=8,
        height=16,
        width=16,
        kernel_size=3,
        scale_factor=2.0
    )

    # Test 3: Larger dimensions with channel tiling
    # Now we can support larger in_channels due to tiling
    print("\n" + "=" * 60)
    print("Test 3: Larger dimensions with channel tiling")
    print("=" * 60)
    test3_passed = run_test(
        batch_size=4,
        in_channels=32,  # weight_k = 32*9 = 288 > P_MAX, requires tiling
        out_channels=32,
        height=32,
        width=32,
        kernel_size=3,
        scale_factor=2.0
    )

    # Test 4: Full in_channels=64 test
    print("\n" + "=" * 60)
    print("Test 4: Full in_channels=64 test")
    print("=" * 60)
    test4_passed = run_test(
        batch_size=2,
        in_channels=64,  # weight_k = 64*9 = 576 > P_MAX, requires multiple tiles
        out_channels=64,
        height=32,
        width=32,
        kernel_size=3,
        scale_factor=2.0
    )

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"  Test 1 (small): {'PASS' if test1_passed else 'FAIL'}")
    print(f"  Test 2 (medium): {'PASS' if test2_passed else 'FAIL'}")
    print(f"  Test 3 (larger, channel tiling): {'PASS' if test3_passed else 'FAIL'}")
    print(f"  Test 4 (full in_ch=64): {'PASS' if test4_passed else 'FAIL'}")

    all_passed = test1_passed and test2_passed and test3_passed and test4_passed
    if all_passed:
        print("\nAll tests passed!")
        return True
    else:
        print("\nSome tests failed.")
        return False


if __name__ == "__main__":
    main()
