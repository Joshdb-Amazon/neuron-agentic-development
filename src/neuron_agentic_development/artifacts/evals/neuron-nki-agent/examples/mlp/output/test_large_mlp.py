"""
Large MLP Test - Tests with tokens=4096, input_size=4096, hidden_size=8192

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate
    python test_large_mlp.py
"""

import os
import sys
import time

# Environment setup
os.environ["NEURON_CC_FLAGS"] = "--target trn2 --lnc 1"
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

import torch
import torch.nn.functional as F
from torch_xla.core import xla_model as xm

from swiglu_mlp_kernel import swiglu_mlp_kernel


def pytorch_swiglu_mlp(x, gate_w, up_w, down_w):
    """PyTorch reference implementation."""
    gate = x @ gate_w.T
    up = x @ up_w.T
    hidden = F.silu(gate) * up
    output = hidden @ down_w.T
    return output


def run_nki_batched(x, gate_w_t, up_w_t, down_w_t, device, batch_size=128):
    """
    Run NKI kernel in batches of 128 tokens.
    """
    total_tokens = x.shape[0]
    num_batches = (total_tokens + batch_size - 1) // batch_size

    outputs = []
    for i, start in enumerate(range(0, total_tokens, batch_size)):
        end = min(start + batch_size, total_tokens)
        x_batch = x[start:end]

        if i % 8 == 0:
            print(f"    Processing batch {i+1}/{num_batches} (tokens {start}-{end})...")

        out_batch = swiglu_mlp_kernel(x_batch, gate_w_t, up_w_t, down_w_t)
        outputs.append(out_batch)

    result = torch.cat(outputs, dim=0)
    return result


def main():
    print("=" * 70)
    print("Large MLP Test")
    print("  tokens=4096, input_size=4096, hidden_size=8192")
    print("=" * 70)

    device = xm.xla_device()
    print(f"Device: {device}")

    # Target dimensions
    tokens = 4096
    input_size = 4096
    hidden_size = 8192

    print(f"\nTest dimensions:")
    print(f"  Tokens: {tokens}")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Batches: {tokens // 128} x 128 tokens")

    # Memory estimate
    param_size = (input_size * hidden_size * 2 + hidden_size * input_size) * 2  # bytes (float16)
    print(f"  Weight memory: {param_size / 1e9:.2f} GB")

    # Create inputs
    print("\nCreating test tensors...")
    torch.manual_seed(42)
    x = torch.randn((tokens, input_size), dtype=torch.float16)
    gate_w = torch.randn((hidden_size, input_size), dtype=torch.float16)
    up_w = torch.randn((hidden_size, input_size), dtype=torch.float16)
    down_w = torch.randn((input_size, hidden_size), dtype=torch.float16)

    # PyTorch reference
    print("\nComputing PyTorch reference...")
    t0 = time.time()
    ref_output = pytorch_swiglu_mlp(x, gate_w, up_w, down_w)
    ref_time = time.time() - t0
    print(f"  PyTorch CPU time: {ref_time*1000:.2f} ms")

    # Move to device
    print("\nMoving tensors to device...")
    x_dev = x.to(device=device)
    gate_w_t = gate_w.T.contiguous().to(device=device)
    up_w_t = up_w.T.contiguous().to(device=device)
    down_w_t = down_w.T.contiguous().to(device=device)

    # Run NKI kernel (batched)
    print("\nRunning NKI kernel (32 batches of 128 tokens)...")
    t0 = time.time()
    nki_output = run_nki_batched(x_dev, gate_w_t, up_w_t, down_w_t, device, batch_size=128)
    nki_output_cpu = nki_output.cpu()
    nki_time = time.time() - t0
    print(f"\n  NKI total time: {nki_time*1000:.2f} ms")

    # Validation
    print("\n" + "=" * 70)
    print("Validation Results")
    print("=" * 70)

    abs_diff = torch.abs(nki_output_cpu - ref_output)
    ref_abs = torch.abs(ref_output).clamp(min=1e-6)

    cos_sim = F.cosine_similarity(
        nki_output_cpu.flatten().unsqueeze(0).float(),
        ref_output.flatten().unsqueeze(0).float()
    ).item()

    mean_rel_diff = (abs_diff / ref_abs).mean().item()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    print(f"  Cosine similarity:   {cos_sim:.6f}")
    print(f"  Mean relative diff:  {mean_rel_diff*100:.4f}%")
    print(f"  Max absolute diff:   {max_abs_diff:.2f}")
    print(f"  Mean absolute diff:  {mean_abs_diff:.4f}")

    # Pass criteria
    passed = cos_sim > 0.9999 and mean_rel_diff < 0.01

    print("\n" + "-" * 70)
    if passed:
        print("[PASS] Large MLP test passed!")
    else:
        print("[FAIL] Large MLP test failed!")
    print("-" * 70)

    # Show samples from different positions
    print("\nSample comparisons:")
    for idx in [0, tokens//2, tokens-1]:
        print(f"\n  Row {idx}:")
        print(f"    NKI: {nki_output_cpu[idx, :4]}")
        print(f"    Ref: {ref_output[idx, :4]}")

    print("\n" + "=" * 70)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
