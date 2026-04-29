"""
NKI Softmax Kernel - Numerically stable softmax along the last dimension of a 2D tensor.

Algorithm: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

Tiling strategy:
  - Partition dimension (rows): tiled into chunks of 128 (P_MAX)
  - Free dimension (cols): 8192, fits in SBUF (max 32767)

Tested on gen3 (trn2) hardware.
"""

import nki
import nki.isa as nisa
import nki.language as nl


# === Self-contained utilities ===

def kernel_assert(condition: bool, error_text: str):
    """Assert with NKI-formatted error message."""
    assert condition, f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"


def div_ceil(n: int, d: int) -> int:
    """Ceiling division: smallest integer >= n/d."""
    return (n + d - 1) // d


# === Hardware constants ===
P_MAX = 128  # Partition dimension max


@nki.jit(platform_target="trn2")
def softmax_kernel(input_tensor: nl.ndarray) -> nl.ndarray:
    """
    Numerically stable softmax along the last dimension.

    Dimensions:
        R: Number of rows (first dimension)
        C: Number of columns (last dimension, softmax applied here)

    Args:
        input_tensor (nl.ndarray): [R, C] @ HBM, input 2D tensor

    Returns:
        nl.ndarray: [R, C] @ HBM, softmax output

    Pseudocode:
        for row_tile in tiles(R, 128):
            tile = load(input[row_tile, :])
            max_val = reduce_max(tile, axis=1)
            shifted = tile - max_val
            exp_val = exp(shifted)
            sum_exp = reduce_sum(exp_val, axis=1)
            result = exp_val / sum_exp
            store(output[row_tile, :], result)
    """
    # === Input Validation ===
    kernel_assert(len(input_tensor.shape) == 2, "Input must be 2D [rows, cols]")

    # === Extract Dimensions ===
    rows, cols = input_tensor.shape

    # === Allocate Output ===
    output = nl.ndarray(input_tensor.shape, dtype=input_tensor.dtype, buffer=nl.shared_hbm)

    # === Calculate Tiling ===
    num_row_tiles = div_ceil(rows, P_MAX)

    # === Main Processing Loop ===
    for r_idx in nl.affine_range(num_row_tiles):
        r_start = r_idx * P_MAX
        r_end = min(r_start + P_MAX, rows)
        r_size = r_end - r_start

        # --- Load Input Tile [P, C] ---
        input_sb = nl.ndarray((r_size, cols), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=input_sb, src=input_tensor[r_start:r_end, 0:cols])

        # --- Step 1: Find max along cols (axis=1) for numerical stability ---
        # Result shape: [P, 1]
        max_sb = nl.ndarray((r_size, 1), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=max_sb, data=input_sb, op=nl.maximum, axis=1)

        # --- Step 2: Subtract max from input (broadcast [P,1] across [P,C]) ---
        shifted_sb = nl.ndarray((r_size, cols), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=shifted_sb, data=input_sb, op0=nl.subtract, operand0=max_sb)

        # --- Step 3: Compute exp(shifted) ---
        exp_sb = nl.ndarray((r_size, cols), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.activation(dst=exp_sb, data=shifted_sb, op=nl.exp)

        # --- Step 4: Sum exp values along cols (axis=1) ---
        sum_sb = nl.ndarray((r_size, 1), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sum_sb, data=exp_sb, op=nl.add, axis=1)

        # --- Step 5: Compute reciprocal of sum ---
        recip_sb = nl.ndarray((r_size, 1), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.reciprocal(dst=recip_sb, data=sum_sb)

        # --- Step 6: Multiply exp values by reciprocal of sum (broadcast) ---
        result_sb = nl.ndarray((r_size, cols), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=result_sb, data=exp_sb, op0=nl.multiply, operand0=recip_sb)

        # --- Store Output Tile ---
        nisa.dma_copy(dst=output[r_start:r_end, 0:cols], src=result_sb)

    return output
