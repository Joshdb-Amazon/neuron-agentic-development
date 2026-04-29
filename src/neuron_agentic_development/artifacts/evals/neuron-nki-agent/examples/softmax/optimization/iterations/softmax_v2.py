"""
NKI Softmax Kernel v2 - In-place buffer reuse + reduced SBUF allocations.

Optimization over v1:
  - Reuse input_sb buffer for exp result (input no longer needed after activation)
  - Reuse exp buffer for final result (exp no longer needed after multiply)
  - Reduces SBUF pressure and may improve compiler scheduling

Tiling strategy:
  - Partition dimension (rows): tiled into chunks of 128 (P_MAX)
  - Free dimension (cols): 8192, fits in SBUF (max 32767)
"""

import nki
import nki.isa as nisa
import nki.language as nl


def kernel_assert(condition: bool, error_text: str):
    assert condition, f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"


def div_ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


P_MAX = 128


@nki.jit(platform_target="trn2")
def softmax_kernel(input_tensor: nl.ndarray) -> nl.ndarray:
    """
    Numerically stable softmax along the last dimension.

    Args:
        input_tensor (nl.ndarray): [R, C] @ HBM, input 2D tensor

    Returns:
        nl.ndarray: [R, C] @ HBM, softmax output
    """
    kernel_assert(len(input_tensor.shape) == 2, "Input must be 2D [rows, cols]")

    rows, cols = input_tensor.shape
    output = nl.ndarray(input_tensor.shape, dtype=input_tensor.dtype, buffer=nl.shared_hbm)
    num_row_tiles = div_ceil(rows, P_MAX)

    for r_idx in nl.affine_range(num_row_tiles):
        r_start = r_idx * P_MAX
        r_end = min(r_start + P_MAX, rows)
        r_size = r_end - r_start

        # --- Load Input Tile [P, C] ---
        tile = nl.ndarray((r_size, cols), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile, src=input_tensor[r_start:r_end, 0:cols])

        # --- Step 1: Find max along cols ---
        max_sb = nl.ndarray((r_size, 1), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=max_sb, data=tile, op=nl.maximum, axis=1)

        # --- Step 2: Negate max for bias ---
        neg_max_sb = nl.ndarray((r_size, 1), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=neg_max_sb, data=max_sb, op0=nl.multiply, operand0=-1.0)

        # --- Step 3 (FUSED): exp(input - max) in-place into tile ---
        nisa.activation(dst=tile, data=tile, op=nl.exp, scale=1.0, bias=neg_max_sb)

        # --- Step 4: Sum exp values ---
        sum_sb = nl.ndarray((r_size, 1), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sum_sb, data=tile, op=nl.add, axis=1)

        # --- Step 5: Compute reciprocal of sum ---
        recip_sb = nl.ndarray((r_size, 1), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.reciprocal(dst=recip_sb, data=sum_sb)

        # --- Step 6: Multiply exp by reciprocal in-place ---
        nisa.tensor_scalar(dst=tile, data=tile, op0=nl.multiply, operand0=recip_sb)

        # --- Store Output Tile ---
        nisa.dma_copy(dst=output[r_start:r_end, 0:cols], src=tile)

    return output
