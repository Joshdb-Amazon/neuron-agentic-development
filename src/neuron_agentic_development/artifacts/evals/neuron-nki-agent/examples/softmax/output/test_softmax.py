"""
Test harness for NKI softmax kernel.

Workflow:
1. Generate random input tensor
2. Compute PyTorch CPU reference (torch.softmax)
3. Run NKI kernel on Neuron device
4. Validate accuracy with multiple metrics
"""

import os
import torch

# Set environment variables for gen3 (trn2)
os.environ["NEURON_CC_FLAGS"] = "--target trn2 --lnc 1"
os.environ["NEURON_RT_VISIBLE_CORES"] = "0"

from torch_xla.core import xla_model as xm

# Import the NKI kernel
from softmax_nki_kernel import softmax_kernel


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
    nki_flat = output_nki.flatten()
    ref_flat = output_ref.flatten()

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
# Test Functions
# ============================================================================

def run_test(rows, cols, test_name=""):
    """
    Run a single softmax test case.

    Returns True if test passes, False otherwise.
    """
    print(f"\nTest Configuration: {test_name}")
    print(f"  rows={rows}, cols={cols}")

    # Generate random input
    cpu_input = torch.randn(rows, cols, dtype=torch.float32)

    # Compute CPU reference
    cpu_output = torch.softmax(cpu_input, dim=-1)

    # Run on device
    device = xm.xla_device()
    device_input = cpu_input.to(device=device)

    print("  Running NKI kernel...")
    device_output = softmax_kernel(device_input)
    device_result = device_output.cpu()

    print(f"  Output shape: {device_result.shape}")

    # Compute metrics
    metrics = compute_metrics(device_result, cpu_output)

    print("\n  Accuracy Metrics:")
    print(f"    Cosine Similarity:  {metrics['cosine_similarity']:.10f}")
    print(f"    Mean Relative Diff: {metrics['mean_relative_diff']:.6e}")
    print(f"    Max Absolute Diff:  {metrics['max_absolute_diff']:.6e}")
    print(f"    Diff Norm:          {metrics['diff_norm']:.6e}")

    # Softmax-specific checks
    row_sums = device_result.sum(dim=-1)
    all_positive = (device_result >= 0).all().item()
    all_le_one = (device_result <= 1).all().item()
    sums_close_to_one = torch.allclose(row_sums, torch.ones(rows), atol=1e-4)

    print(f"\n  Softmax Properties:")
    print(f"    All values >= 0:      {all_positive}")
    print(f"    All values <= 1:      {all_le_one}")
    print(f"    Row sums close to 1:  {sums_close_to_one}")

    # Determine pass/fail
    allclose = torch.allclose(device_result, cpu_output, rtol=1e-5, atol=1e-5)
    passed = allclose and all_positive and sums_close_to_one

    if passed:
        print(f"\n  [PASS] {test_name}")
    else:
        print(f"\n  [FAIL] {test_name}")
        if not allclose:
            # Try relaxed tolerances
            for test_atol in [1e-4, 1e-3, 1e-2]:
                relaxed = torch.allclose(device_result, cpu_output, rtol=1e-4, atol=test_atol)
                print(f"    torch.allclose(rtol=1e-4, atol={test_atol}): {relaxed}")
                if relaxed:
                    break

    return passed, metrics


def main():
    """Run test suite."""
    print("=" * 60)
    print("Softmax NKI Kernel Test Suite")
    print("=" * 60)

    results = []

    # Test 1: Original spec dimensions
    print("\n" + "=" * 60)
    print("Test 1: Original spec (1024 x 8192)")
    print("=" * 60)
    passed, metrics = run_test(1024, 8192, "1024x8192 (original spec)")
    results.append(("1024x8192", passed, metrics))

    # Test 2: Small dimensions (single tile)
    print("\n" + "=" * 60)
    print("Test 2: Small dimensions (single tile)")
    print("=" * 60)
    passed, metrics = run_test(128, 512, "128x512 (single tile)")
    results.append(("128x512", passed, metrics))

    # Test 3: Non-aligned rows (tests remainder handling)
    print("\n" + "=" * 60)
    print("Test 3: Non-aligned rows")
    print("=" * 60)
    passed, metrics = run_test(300, 4096, "300x4096 (non-aligned rows)")
    results.append(("300x4096", passed, metrics))

    # Test 4: Large columns
    print("\n" + "=" * 60)
    print("Test 4: Large columns")
    print("=" * 60)
    passed, metrics = run_test(256, 16384, "256x16384 (large cols)")
    results.append(("256x16384", passed, metrics))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"\n{'Test':<20} {'Cosine Sim':<14} {'Max Abs Diff':<14} {'Status':<8}")
    print("-" * 56)
    for name, passed, metrics in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name:<20} {metrics['cosine_similarity']:<14.10f} {metrics['max_absolute_diff']:<14.2e} {status:<8}")

    all_passed = all(p for _, p, _ in results)
    if all_passed:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed.")

    return all_passed


if __name__ == "__main__":
    main()
