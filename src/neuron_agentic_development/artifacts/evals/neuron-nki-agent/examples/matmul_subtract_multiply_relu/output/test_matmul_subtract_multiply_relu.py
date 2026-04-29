"""
Test script for the fused matmul + subtract + multiply + ReLU NKI kernel.

Compares NKI kernel output (on Neuron device) against PyTorch CPU baseline.
"""

import os

# Must be set before any neuron/NKI imports
os.environ["NEURON_CC_FLAGS"] = "--target trn2 --lnc 1"
os.environ["NEURON_RT_VISIBLE_CORES"] = "0"

import torch
import torch.nn as nn

# Import the NKI kernel
from matmul_subtract_multiply_relu_kernel import matmul_subtract_multiply_relu_kernel


def torch_reference(x, weight, bias, subtract_value, multiply_value):
    """CPU reference: relu((x @ weight.T + bias - subtract_value) * multiply_value)"""
    out = x @ weight.T + bias
    out = (out - subtract_value) * multiply_value
    out = torch.relu(out)
    return out


def main():
    # --- Configuration ---
    batch_size = 256
    in_features = 4096
    out_features = 4096
    subtract_value = 2.0
    multiply_value = 1.5

    # --- Setup device ---
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()

    # --- Generate inputs (bfloat16 for TensorEngine compatibility) ---
    torch.manual_seed(42)
    x = torch.randn(batch_size, in_features, dtype=torch.bfloat16)
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)
    bias = torch.randn(out_features, dtype=torch.bfloat16)

    # --- CPU reference (compute in float32 for accuracy) ---
    ref_output = torch_reference(
        x.float(), weight.float(), bias.float(),
        subtract_value, multiply_value
    )

    # --- Prepare kernel inputs ---
    # Transpose x for stationary operand layout: (in_features, batch_size)
    x_t = x.T.contiguous()
    # Transpose weight for the kernel: (in_features, out_features)
    weight_t = weight.T.contiguous()
    # Reshape bias to 2D for DMA loading: (1, out_features)
    bias_2d = bias.float().unsqueeze(0).contiguous()

    # Move to device
    x_t_dev = x_t.to(device)
    weight_t_dev = weight_t.to(device)
    bias_2d_dev = bias_2d.to(device)

    # --- Run NKI kernel ---
    print("Running NKI kernel on device...")
    output_dev = matmul_subtract_multiply_relu_kernel(
        x_t_dev, weight_t_dev, bias_2d_dev,
        subtract_value, multiply_value,
    )

    # --- Compare results ---
    output_cpu = output_dev.cpu().float()
    print(f"Output shape: {output_cpu.shape}")
    print(f"Reference shape: {ref_output.shape}")

    # Multiple accuracy checks
    abs_diff = (ref_output - output_cpu).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    # Cosine similarity
    flat_ref = ref_output.flatten()
    flat_out = output_cpu.flatten()
    cosine_sim = torch.nn.functional.cosine_similarity(
        flat_ref.unsqueeze(0), flat_out.unsqueeze(0)
    ).item()

    print(f"\n--- Accuracy Metrics ---")
    print(f"Max absolute diff:  {max_diff:.6f}")
    print(f"Mean absolute diff: {mean_diff:.6f}")
    print(f"Cosine similarity:  {cosine_sim:.8f}")

    # Correctness check with tolerances appropriate for bfloat16 matmul
    passed = torch.allclose(ref_output, output_cpu, atol=1.0, rtol=1e-1)
    if passed:
        print(f"\nPASS: NKI kernel matches PyTorch reference (atol=1.0, rtol=0.1)")
    else:
        print(f"\nFAIL: NKI kernel does NOT match PyTorch reference")
        # Show distribution of diffs
        print(f"Diff percentiles: "
              f"p50={abs_diff.median().item():.4f}, "
              f"p90={abs_diff.quantile(0.9).item():.4f}, "
              f"p99={abs_diff.quantile(0.99).item():.4f}")

    # Tighter check
    tight_pass = torch.allclose(ref_output, output_cpu, atol=0.5, rtol=5e-2)
    if tight_pass:
        print(f"PASS (tight): atol=0.5, rtol=0.05")
    else:
        print(f"INFO: Did not pass tight tolerance (atol=0.5, rtol=0.05)")

    return passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
