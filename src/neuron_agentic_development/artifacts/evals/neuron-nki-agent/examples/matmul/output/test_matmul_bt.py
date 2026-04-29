"""
Test script for matmul_bt_kernel on Neuron hardware.

Compares NKI kernel output against PyTorch CPU reference.
Test inputs are scaled by 1/sqrt(input_size) to prevent float16 overflow.
"""

import os
import torch
import numpy as np
import torch_xla

# Set Neuron compiler flags for gen3 (Trainium2)
os.environ["NEURON_CC_FLAGS"] = "--target trn2 --lnc 1"
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

from matmul_bt_kernel import matmul_bt_kernel


def compute_accuracy_metrics(output: torch.Tensor, reference: torch.Tensor):
    """
    Compute multiple accuracy metrics for numerical validation.

    Args:
        output: Tensor from NKI kernel
        reference: Reference tensor from PyTorch

    Returns:
        dict with accuracy metrics
    """
    output_flat = output.flatten().float()
    reference_flat = reference.flatten().float()

    # 1. Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        output_flat.unsqueeze(0),
        reference_flat.unsqueeze(0)
    ).item()

    # 2. Mean relative difference
    rel_diff = torch.abs(output_flat - reference_flat) / (torch.abs(reference_flat) + 1e-8)
    mean_rel_diff = rel_diff.mean().item()

    # 3. Max absolute difference
    max_abs_diff = torch.abs(output_flat - reference_flat).max().item()

    # 4. Mean absolute difference
    mean_abs_diff = torch.abs(output_flat - reference_flat).mean().item()

    # 5. torch.allclose check
    allclose = torch.allclose(output.float(), reference.float(), rtol=1e-2, atol=1e-2)

    return {
        "cosine_similarity": cos_sim,
        "mean_rel_diff": mean_rel_diff,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "allclose": allclose
    }


def test_matmul_bt():
    """Test matmul_bt_kernel against PyTorch reference."""

    # Dimensions from matmul_with_BT.py
    M = 1024 * 2  # 2048
    K = 4096 * 2  # 8192
    N = 2048 * 2  # 4096

    print(f"Testing matmul: C[{M}, {N}] = A[{M}, {K}] @ B[{N}, {K}].T")
    print(f"Using dtype: float16")

    # Create test inputs scaled by 1/sqrt(input_size) to prevent overflow
    # For matmul, scale by 1/sqrt(K) since K elements are summed
    scale = 1.0 / np.sqrt(K)

    torch.manual_seed(42)
    A_cpu = (torch.randn(M, K) * scale).to(torch.float16)
    B_cpu = (torch.randn(N, K) * scale).to(torch.float16)

    print(f"Input scale factor: {scale:.6f}")
    print(f"A range: [{A_cpu.min().item():.4f}, {A_cpu.max().item():.4f}]")
    print(f"B range: [{B_cpu.min().item():.4f}, {B_cpu.max().item():.4f}]")

    # Compute reference on CPU (using float32 for accuracy)
    print("\nComputing CPU reference...")
    reference = torch.matmul(A_cpu.float(), B_cpu.float().T).to(torch.float16)
    print(f"Reference range: [{reference.min().item():.4f}, {reference.max().item():.4f}]")

    # Move inputs to Neuron device
    print("\nMoving inputs to Neuron device...")
    device = torch_xla.device()
    A_device = A_cpu.to(device=device)
    B_device = B_cpu.to(device=device)

    # Run NKI kernel
    print("Running NKI kernel...")
    C_device = matmul_bt_kernel(A_device, B_device)

    # Force compilation and execution
    print("Triggering compilation (this may take a moment)...")
    C_cpu = C_device.cpu()

    print(f"Output range: [{C_cpu.min().item():.4f}, {C_cpu.max().item():.4f}]")
    print(f"Output shape: {C_cpu.shape}")

    # Compute accuracy metrics
    print("\n" + "="*60)
    print("Accuracy Metrics:")
    print("="*60)

    metrics = compute_accuracy_metrics(C_cpu, reference)

    print(f"  Cosine Similarity:     {metrics['cosine_similarity']:.8f}")
    print(f"  Mean Relative Diff:    {metrics['mean_rel_diff']:.8e}")
    print(f"  Max Absolute Diff:     {metrics['max_abs_diff']:.8e}")
    print(f"  Mean Absolute Diff:    {metrics['mean_abs_diff']:.8e}")
    print(f"  torch.allclose:        {metrics['allclose']}")

    # Validation thresholds
    COSINE_THRESHOLD = 0.999
    MAX_DIFF_THRESHOLD = 0.1  # Relaxed for float16

    print("\n" + "="*60)
    print("Validation Results:")
    print("="*60)

    passed = True

    if metrics['cosine_similarity'] >= COSINE_THRESHOLD:
        print(f"  [PASS] Cosine similarity {metrics['cosine_similarity']:.6f} >= {COSINE_THRESHOLD}")
    else:
        print(f"  [FAIL] Cosine similarity {metrics['cosine_similarity']:.6f} < {COSINE_THRESHOLD}")
        passed = False

    if metrics['max_abs_diff'] <= MAX_DIFF_THRESHOLD:
        print(f"  [PASS] Max abs diff {metrics['max_abs_diff']:.6f} <= {MAX_DIFF_THRESHOLD}")
    else:
        print(f"  [FAIL] Max abs diff {metrics['max_abs_diff']:.6f} > {MAX_DIFF_THRESHOLD}")
        passed = False

    if passed:
        print("\n*** ALL TESTS PASSED ***")
    else:
        print("\n*** SOME TESTS FAILED ***")

    return passed, metrics


if __name__ == "__main__":
    passed, metrics = test_matmul_bt()
    exit(0 if passed else 1)
