"""
Test harness for GELU NKI kernel.

Workflow:
1. Create PyTorch reference implementation (CPU)
2. Run NKI kernel on device
3. Validate accuracy with multiple metrics
"""

import os
import sys

# Set environment variables for gen3 (trn2)
os.environ["NEURON_CC_FLAGS"] = "--target trn2 --lnc 1"
os.environ["NEURON_RT_VISIBLE_CORES"] = "0-1"

import torch
import torch.nn.functional as F
from torch_xla.core import xla_model as xm

# Import the NKI kernel
from gelu_nki_kernels import gelu_kernel


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
        - mean_absolute_diff: average absolute error
        - diff_norm: L2 norm of difference
        - ref_norm: L2 norm of reference
    """
    nki_flat = output_nki.flatten().float()
    ref_flat = output_ref.flatten().float()

    cos_sim = F.cosine_similarity(
        nki_flat.unsqueeze(0),
        ref_flat.unsqueeze(0)
    ).item()

    abs_diff = torch.abs(nki_flat - ref_flat)
    mean_rel_diff = (abs_diff / (torch.abs(ref_flat) + 1e-8)).mean().item()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    diff_norm = torch.norm(nki_flat - ref_flat).item()
    ref_norm = torch.norm(ref_flat).item()

    return {
        'cosine_similarity': cos_sim,
        'mean_relative_diff': mean_rel_diff,
        'max_absolute_diff': max_abs_diff,
        'mean_absolute_diff': mean_abs_diff,
        'diff_norm': diff_norm,
        'ref_norm': ref_norm,
    }


# ============================================================================
# Test Functions
# ============================================================================

def run_test(rows, cols, dtype=torch.float32):
    """
    Run a single test case.

    Returns True if test passes, False otherwise.
    """
    print(f"\nTest Configuration:")
    print(f"  rows={rows}, cols={cols}, dtype={dtype}")

    # Create input on CPU for reference
    cpu_input = torch.randn(rows, cols, dtype=dtype)

    # Compute CPU reference
    cpu_reference = F.gelu(cpu_input)

    print(f"  Reference output shape: {cpu_reference.shape}")

    # Move input to XLA device and run NKI kernel
    device = xm.xla_device()
    device_input = cpu_input.to(device=device)

    print("Running NKI kernel...")
    device_output = gelu_kernel(device_input)

    # Force XLA execution and bring result back to CPU
    device_result = device_output.cpu()

    print(f"  NKI output shape: {device_result.shape}")

    # Compute metrics
    metrics = compute_metrics(device_result, cpu_reference)

    print("\nAccuracy Metrics:")
    print(f"  Cosine Similarity:    {metrics['cosine_similarity']:.6f}")
    print(f"  Mean Relative Diff:   {metrics['mean_relative_diff']:.6f}")
    print(f"  Max Absolute Diff:    {metrics['max_absolute_diff']:.2e}")
    print(f"  Mean Absolute Diff:   {metrics['mean_absolute_diff']:.2e}")
    print(f"  Diff Norm:            {metrics['diff_norm']:.2e}")
    print(f"  Ref Norm:             {metrics['ref_norm']:.2e}")

    # Determine pass/fail
    allclose = torch.allclose(device_result, cpu_reference, rtol=1e-4, atol=1e-5)
    passed = allclose and metrics['cosine_similarity'] > 0.9999

    if passed:
        print("\n[PASS] NKI kernel matches reference!")
    else:
        print("\n[FAIL] NKI kernel differs from reference")
        print("NKI output (first 5):", device_result.flatten()[:5])
        print("Reference (first 5):", cpu_reference.flatten()[:5])

    return passed, metrics


def main():
    """Run test suite."""
    print("=" * 60)
    print("GELU NKI Kernel Test Suite")
    print("=" * 60)

    results = []

    # Test 1: Target dimensions from spec
    print("\n" + "=" * 60)
    print("Test 1: Target dimensions (2048 x 8192)")
    print("=" * 60)
    test1_passed, test1_metrics = run_test(rows=2048, cols=8192)
    results.append(("2048x8192", test1_passed, test1_metrics))

    # Test 2: Smaller dimensions
    print("\n" + "=" * 60)
    print("Test 2: Small dimensions (128 x 512)")
    print("=" * 60)
    test2_passed, test2_metrics = run_test(rows=128, cols=512)
    results.append(("128x512", test2_passed, test2_metrics))

    # Test 3: Non-square dimensions
    print("\n" + "=" * 60)
    print("Test 3: Non-square dimensions (256 x 4096)")
    print("=" * 60)
    test3_passed, test3_metrics = run_test(rows=256, cols=4096)
    results.append(("256x4096", test3_passed, test3_metrics))

    # Test 4: Large dimensions
    print("\n" + "=" * 60)
    print("Test 4: Large dimensions (4096 x 8192)")
    print("=" * 60)
    test4_passed, test4_metrics = run_test(rows=4096, cols=8192)
    results.append(("4096x8192", test4_passed, test4_metrics))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    all_passed = True
    for name, passed, metrics in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status} (cos_sim={metrics['cosine_similarity']:.6f}, "
              f"max_abs_diff={metrics['max_absolute_diff']:.2e})")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
