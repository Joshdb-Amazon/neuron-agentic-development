"""
Test script for RMS Fused QKV NKI kernel.

Compares NKI kernel output against CPU PyTorch reference.
Uses the neuron-nki-debugging skill pattern: CPU reference, device execution, numerical comparison.
"""

import os
import sys
import torch

# Import NKI kernel
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rms_fused_qkv_kernel import rms_fused_qkv_kernel


def cpu_reference(hidden, weight, eps=1e-6):
    """CPU reference: RMSNorm(hidden) @ weight."""
    h = hidden.float()
    rms = torch.sqrt(torch.mean(h ** 2, dim=-1, keepdim=True) + eps)
    h_norm = h / rms
    return (h_norm @ weight.float()).to(hidden.dtype)


def main():
    # Parameters matching the original spec
    batch = 1
    seqlen = 256
    dim = 2048
    head_dim = 1024
    eps = 1e-6

    # Create deterministic inputs
    torch.manual_seed(42)
    hidden = torch.randn(batch, seqlen, dim, dtype=torch.float32)
    weight = torch.randn(dim, head_dim, dtype=torch.float32) * 0.02

    # CPU reference (compute on CPU, not XLA device)
    ref_output = cpu_reference(hidden, weight, eps)  # [batch, seqlen, head_dim]
    ref_flat = ref_output.reshape(batch * seqlen, head_dim)

    print(f"Input shape: {hidden.shape}")
    print(f"Weight shape: {weight.shape}")
    print(f"Reference output shape: {ref_output.shape}")
    print(f"Ref output range: [{ref_flat.min():.4f}, {ref_flat.max():.4f}]")
    print()

    # Run NKI kernel on device
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    hidden_2d = hidden.reshape(batch * seqlen, dim).to(device)
    weight_dev = weight.to(device)

    print("Running NKI kernel on device...")
    nki_output_2d = rms_fused_qkv_kernel(hidden_2d, weight_dev, eps=eps)
    nki_output = nki_output_2d.cpu()  # [batch*seqlen, head_dim]

    print(f"NKI output shape: {nki_output.shape}")
    print(f"NKI output range: [{nki_output.min():.4f}, {nki_output.max():.4f}]")
    print()

    # Numerical comparison
    max_diff = (ref_flat - nki_output).abs().max().item()
    mean_diff = (ref_flat - nki_output).abs().mean().item()

    # Cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(
        ref_flat.flatten().unsqueeze(0),
        nki_output.flatten().unsqueeze(0),
    ).item()

    # Norm of difference
    diff_norm = torch.norm(ref_flat - nki_output).item()
    ref_norm = torch.norm(ref_flat).item()
    relative_norm = diff_norm / ref_norm if ref_norm > 0 else float("inf")

    print("=== Accuracy Metrics ===")
    print(f"Max absolute diff:    {max_diff:.6e}")
    print(f"Mean absolute diff:   {mean_diff:.6e}")
    print(f"Cosine similarity:    {cosine_sim:.8f}")
    print(f"Relative norm diff:   {relative_norm:.6e}")
    print()

    # Pass/fail checks
    allclose_strict = torch.allclose(ref_flat, nki_output, atol=1e-4, rtol=1e-3)
    allclose_loose = torch.allclose(ref_flat, nki_output, atol=1e-3, rtol=1e-2)

    print(f"Allclose (atol=1e-4, rtol=1e-3): {allclose_strict}")
    print(f"Allclose (atol=1e-3, rtol=1e-2): {allclose_loose}")
    print()

    if allclose_loose:
        print("PASSED: NKI kernel matches CPU reference within tolerance.")
    else:
        print("FAILED: NKI kernel output does NOT match CPU reference.")
        sys.exit(1)


if __name__ == "__main__":
    main()
