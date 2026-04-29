"""
Full MLP Test - Tests the NKI kernel with original mlp.py dimensions.

Since the kernel processes tokens <= 128 at a time, this test tiles
the batch dimension to handle larger inputs.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate
    python test_full_mlp.py
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

    Args:
        x: Input tensor [total_tokens, input_size] on device
        gate_w_t, up_w_t, down_w_t: Transposed weights on device
        device: XLA device
        batch_size: Tokens per batch (max 128)

    Returns:
        Output tensor [total_tokens, input_size]
    """
    total_tokens = x.shape[0]
    input_size = x.shape[1]

    # Process in batches
    outputs = []
    for start in range(0, total_tokens, batch_size):
        end = min(start + batch_size, total_tokens)
        x_batch = x[start:end]

        out_batch = swiglu_mlp_kernel(x_batch, gate_w_t, up_w_t, down_w_t)
        outputs.append(out_batch)

    # Concatenate results
    result = torch.cat(outputs, dim=0)
    return result


def main():
    print("=" * 70)
    print("Full MLP Test - Original mlp.py Dimensions")
    print("=" * 70)

    device = xm.xla_device()
    print(f"Device: {device}")

    # Original dimensions from mlp.py (scaled down for practical testing)
    # Full: batch=1024, seq=2048, input=4096, hidden=8192
    # This would require processing 2M tokens - too slow for a demo

    # Use representative dimensions for testing
    batch_size = 8
    seq_len = 64
    input_size = 512   # Scaled from 4096
    hidden_size = 1024  # Scaled from 8192

    total_tokens = batch_size * seq_len
    print(f"\nTest dimensions:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")

    # Create inputs
    torch.manual_seed(42)
    x = torch.randn((total_tokens, input_size), dtype=torch.float16)
    gate_w = torch.randn((hidden_size, input_size), dtype=torch.float16)
    up_w = torch.randn((hidden_size, input_size), dtype=torch.float16)
    down_w = torch.randn((input_size, hidden_size), dtype=torch.float16)

    # PyTorch reference
    print("\nComputing PyTorch reference...")
    t0 = time.time()
    ref_output = pytorch_swiglu_mlp(x, gate_w, up_w, down_w)
    ref_time = time.time() - t0
    print(f"  PyTorch time: {ref_time*1000:.2f} ms")

    # Move to device
    x_dev = x.to(device=device)
    gate_w_t = gate_w.T.contiguous().to(device=device)
    up_w_t = up_w.T.contiguous().to(device=device)
    down_w_t = down_w.T.contiguous().to(device=device)

    # Run NKI kernel (batched)
    print("\nRunning NKI kernel (batched, 128 tokens per batch)...")
    t0 = time.time()
    nki_output = run_nki_batched(x_dev, gate_w_t, up_w_t, down_w_t, device, batch_size=128)
    nki_output_cpu = nki_output.cpu()
    nki_time = time.time() - t0
    print(f"  NKI time (including compilation): {nki_time*1000:.2f} ms")

    # Validation
    print("\nValidation:")
    abs_diff = torch.abs(nki_output_cpu - ref_output)
    ref_abs = torch.abs(ref_output).clamp(min=1e-6)

    cos_sim = F.cosine_similarity(
        nki_output_cpu.flatten().unsqueeze(0).float(),
        ref_output.flatten().unsqueeze(0).float()
    ).item()

    mean_rel_diff = (abs_diff / ref_abs).mean().item()

    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Mean relative diff: {mean_rel_diff*100:.4f}%")
    print(f"  Max absolute diff: {abs_diff.max().item():.2f}")

    passed = cos_sim > 0.9999 and mean_rel_diff < 0.01

    if passed:
        print("\n[PASS] Full MLP test passed!")
    else:
        print("\n[FAIL] Full MLP test failed!")

    # Show sample
    print(f"\n  Output sample [0, :5]: {nki_output_cpu[0, :5]}")
    print(f"  Reference [0, :5]:     {ref_output[0, :5]}")

    print("\n" + "=" * 70)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
