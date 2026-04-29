"""
Test script for NKI cumsum kernel.

Validates correctness by comparing against torch.cumsum on CPU.
Uses gen3 (trn2) as target platform.
"""

import os
import torch
from torch_xla.core import xla_model as xm

# Configure for gen3 (trn2)
os.environ["NEURON_CC_FLAGS"] = "--target trn2 --lnc 1"
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

# Import the kernel
from cumsum_kernel import cumsum_kernel


def test_cumsum_correctness():
    """Test cumsum kernel correctness against torch.cumsum."""
    print("=" * 60)
    print("Testing NKI Cumsum Kernel")
    print("=" * 60)

    # Test configuration (from cumsum.py)
    batch_size = 32768
    seq_len = 32768
    dim = 1  # cumsum along last dimension

    print(f"\nTest configuration:")
    print(f"  Shape: ({batch_size}, {seq_len})")
    print(f"  Cumsum dim: {dim}")
    print(f"  Total elements: {batch_size * seq_len:,}")

    # Generate random input on CPU
    # Scale by 1/seq_len to prevent float16 overflow in large accumulations
    print("\nGenerating random input (scaled by 1/seq_len)...")
    cpu_input = torch.rand((batch_size, seq_len), dtype=torch.float32) / seq_len

    # Compute reference on CPU
    print("Computing CPU reference (torch.cumsum)...")
    cpu_reference = torch.cumsum(cpu_input, dim=dim)

    # Get XLA device
    print("\nInitializing XLA device...")
    device = xm.xla_device()

    # Move input to device
    print("Moving input to device...")
    device_input = cpu_input.to(device=device)

    # Run NKI kernel
    print("Running NKI cumsum kernel...")
    device_output = cumsum_kernel(device_input)

    # Force execution and get result back to CPU
    print("Executing and fetching results...")
    device_output_cpu = device_output.cpu()

    # Validate correctness
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)

    # Multiple complementary checks
    # 1. torch.allclose with appropriate tolerances
    atol = 1e-4
    rtol = 1e-4
    close = torch.allclose(device_output_cpu, cpu_reference, atol=atol, rtol=rtol)
    print(f"\n1. torch.allclose (atol={atol}, rtol={rtol}): {'PASS' if close else 'FAIL'}")

    # 2. Maximum absolute difference
    max_abs_diff = torch.max(torch.abs(device_output_cpu - cpu_reference)).item()
    print(f"2. Maximum absolute difference: {max_abs_diff:.6e}")

    # 3. Mean absolute difference
    mean_abs_diff = torch.mean(torch.abs(device_output_cpu - cpu_reference)).item()
    print(f"3. Mean absolute difference: {mean_abs_diff:.6e}")

    # 4. Norm of difference tensor
    diff_norm = torch.norm(device_output_cpu - cpu_reference).item()
    ref_norm = torch.norm(cpu_reference).item()
    relative_norm = diff_norm / ref_norm if ref_norm > 0 else diff_norm
    print(f"4. Relative norm of difference: {relative_norm:.6e}")

    # 5. Cosine similarity (compute manually to avoid edge cases)
    flat_device = device_output_cpu.flatten().double()
    flat_ref = cpu_reference.flatten().double()
    cos_sim = (torch.dot(flat_device, flat_ref) /
               (torch.norm(flat_device) * torch.norm(flat_ref))).item()
    print(f"5. Cosine similarity: {cos_sim:.10f}")

    # Final verdict
    print("\n" + "=" * 60)
    if close and cos_sim > 0.9999:
        print("OVERALL: PASS - Kernel produces correct results!")
    else:
        print("OVERALL: FAIL - Results do not match reference")
        # Print some sample values for debugging
        print("\nSample values (first 5 elements of row 0):")
        print(f"  Device: {device_output_cpu[0, :5].tolist()}")
        print(f"  Reference: {cpu_reference[0, :5].tolist()}")
    print("=" * 60)

    return close


def test_small_input():
    """Quick test with small input for faster iteration."""
    print("\n" + "=" * 60)
    print("Quick Test with Small Input")
    print("=" * 60)

    # Small test case
    batch_size = 128
    seq_len = 256
    dim = 1

    print(f"\nTest configuration:")
    print(f"  Shape: ({batch_size}, {seq_len})")

    # Generate input (scaled by 1/seq_len to prevent overflow)
    cpu_input = torch.rand((batch_size, seq_len), dtype=torch.float32) / seq_len
    cpu_reference = torch.cumsum(cpu_input, dim=dim)

    # Run on device
    device = xm.xla_device()
    device_input = cpu_input.to(device=device)

    print("Running kernel...")
    device_output = cumsum_kernel(device_input)
    device_output_cpu = device_output.cpu()

    # Check
    close = torch.allclose(device_output_cpu, cpu_reference, atol=1e-4, rtol=1e-4)
    max_diff = torch.max(torch.abs(device_output_cpu - cpu_reference)).item()

    print(f"\nResults:")
    print(f"  Close: {'PASS' if close else 'FAIL'}")
    print(f"  Max diff: {max_diff:.6e}")

    if not close:
        print("\nSample comparison:")
        print(f"  Device[0, :10]: {device_output_cpu[0, :10].tolist()}")
        print(f"  Reference[0, :10]: {cpu_reference[0, :10].tolist()}")

    return close


if __name__ == "__main__":
    # Run quick test first
    small_pass = test_small_input()

    if small_pass:
        # Run full test if small test passes
        full_pass = test_cumsum_correctness()
    else:
        print("\nSkipping full test due to small test failure.")
