"""
Test script for SwiGLU MLP NKI Kernel.

This script tests the NKI kernel against a PyTorch reference implementation
following the neuron-nki-debugging workflow.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate
    python test_swiglu_mlp.py
"""

import os
import sys

# === Environment Configuration ===
# Standard debugging flags for Trainium 2 (gen3)
os.environ["NEURON_CC_FLAGS"] = "--target trn2 --lnc 1"
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

import torch
import torch.nn.functional as F
from torch_xla.core import xla_model as xm

# Import the kernel
from swiglu_mlp_kernel import swiglu_mlp_kernel


def pytorch_swiglu_mlp(x, gate_w, up_w, down_w):
    """
    PyTorch reference implementation of SwiGLU MLP.

    Args:
        x: Input tensor [tokens, input_size]
        gate_w: Gate weight [hidden_size, input_size] (PyTorch format)
        up_w: Up weight [hidden_size, input_size]
        down_w: Down weight [input_size, hidden_size]

    Returns:
        Output tensor [tokens, input_size]
    """
    gate = x @ gate_w.T
    up = x @ up_w.T
    hidden = F.silu(gate) * up
    output = hidden @ down_w.T
    return output


def validate_outputs(nki_output, ref_output):
    """
    Validate NKI kernel output against reference using multiple metrics.

    For float16 deep learning workloads, we use:
    - Cosine similarity (directional accuracy) - primary metric
    - Mean relative error (accounting for value magnitude)
    - Max absolute error (informational)

    Returns:
        dict with validation metrics and pass/fail status
    """
    abs_diff = torch.abs(nki_output - ref_output)
    ref_abs = torch.abs(ref_output).clamp(min=1e-6)

    # Compute relative diff, replacing NaN/Inf with 0 for mean calculation
    rel_diff = abs_diff / ref_abs
    rel_diff = torch.where(torch.isfinite(rel_diff), rel_diff, torch.zeros_like(rel_diff))

    metrics = {
        'max_abs_diff': abs_diff.max().item(),
        'mean_abs_diff': abs_diff.mean().item(),
        'norm_diff': torch.norm(nki_output - ref_output).item(),
        'cosine_sim': F.cosine_similarity(
            nki_output.flatten().unsqueeze(0).float(),
            ref_output.flatten().unsqueeze(0).float()
        ).item(),
        'mean_rel_diff': rel_diff.mean().item(),
    }

    # For float16 deep learning workloads, pass criteria:
    # 1. Cosine similarity > 0.9999 (very high directional match)
    # 2. Mean relative difference < 1% (accounting for value magnitudes)
    metrics['passed'] = (
        metrics['cosine_sim'] > 0.9999 and
        metrics['mean_rel_diff'] < 0.01
    )

    return metrics


def run_test(tokens, input_size, hidden_size, dtype=torch.float16):
    """
    Run a single test case.

    Args:
        tokens: Number of tokens
        input_size: Input dimension
        hidden_size: Hidden dimension
        dtype: Data type

    Returns:
        dict with test results
    """
    print(f"\n--- Test: tokens={tokens}, input={input_size}, hidden={hidden_size} ---")

    # Create inputs
    torch.manual_seed(42)

    # Scale factor to prevent float16 overflow for large matmuls
    # Standard randn has std=1, matmul accumulates input_size values
    # Use sqrt(input_size) scaling to keep variance ~1 after matmul
    scale = 1.0 / (input_size ** 0.5)

    x = (torch.randn((tokens, input_size), dtype=torch.float32) * scale).to(dtype)
    gate_w = (torch.randn((hidden_size, input_size), dtype=torch.float32) * scale).to(dtype)
    up_w = (torch.randn((hidden_size, input_size), dtype=torch.float32) * scale).to(dtype)
    down_w = (torch.randn((input_size, hidden_size), dtype=torch.float32) * scale).to(dtype)

    # Compute reference
    print("Computing PyTorch reference...")
    ref_output = pytorch_swiglu_mlp(x, gate_w, up_w, down_w)

    # Get XLA device
    device = xm.xla_device()

    # Move to device (weights in transposed form for kernel)
    x_dev = x.to(device=device)
    gate_w_t = gate_w.T.contiguous().to(device=device)
    up_w_t = up_w.T.contiguous().to(device=device)
    down_w_t = down_w.T.contiguous().to(device=device)

    # Run kernel
    print("Running NKI kernel...")
    try:
        nki_output = swiglu_mlp_kernel(x_dev, gate_w_t, up_w_t, down_w_t)
        nki_output_cpu = nki_output.cpu()

        # Validate
        metrics = validate_outputs(nki_output_cpu, ref_output)

        print(f"  Cosine similarity:  {metrics['cosine_sim']:.6f}")
        print(f"  Mean relative diff: {metrics['mean_rel_diff']*100:.4f}%")
        print(f"  Max absolute diff:  {metrics['max_abs_diff']:.6f}")
        print(f"  Mean absolute diff: {metrics['mean_abs_diff']:.6f}")

        if metrics['passed']:
            print("  [PASS]")
        else:
            print("  [FAIL]")

        # Always show sample comparison
        print(f"    NKI sample: {nki_output_cpu[0, :5]}")
        print(f"    Ref sample: {ref_output[0, :5]}")

        return {'success': True, 'metrics': metrics}

    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def main():
    print("=" * 70)
    print("SwiGLU MLP NKI Kernel Test Suite")
    print("=" * 70)

    # Check device
    device = xm.xla_device()
    print(f"Device: {device}")

    # Test cases with various dimensions
    test_cases = [
        # (tokens, input_size, hidden_size)
        (64, 128, 256),    # Minimal - all fit in single tiles
        (64, 256, 512),    # Input tiling
        (128, 128, 512),   # Max tokens, single input tile
        (128, 256, 512),   # Max tokens, input tiling
        (64, 512, 1024),   # Larger dimensions
        (4096, 4096, 8192),  # Large test - full token tiling
    ]

    results = []
    for tokens, input_size, hidden_size in test_cases:
        result = run_test(tokens, input_size, hidden_size)
        results.append({
            'config': (tokens, input_size, hidden_size),
            **result
        })

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    passed = sum(1 for r in results if r.get('success') and r.get('metrics', {}).get('passed'))
    total = len(results)

    for r in results:
        config = r['config']
        if r.get('success'):
            m = r.get('metrics', {})
            cos_sim = m.get('cosine_sim', 0)
            max_abs = m.get('max_abs_diff', 0)
            status = "PASS" if m.get('passed') else "FAIL"
            print(f"  {config}: {status} (cosine_sim={cos_sim:.6f}, max_abs_diff={max_abs:.6f})")
        else:
            print(f"  {config}: ERROR")

    print(f"\nTotal: {passed}/{total} passed")

    if passed == total:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
