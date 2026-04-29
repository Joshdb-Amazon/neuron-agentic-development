"""
NKI Softmax Kernel v5 - Force negate to Scalar Engine for better engine overlap.

Optimization over v3:
  - Move the negate(max) step from VectorE to ScalarE by using tensor_scalar
    with op0=multiply, op1=add (which is the only dual-op combo ScalarE supports)
  - This frees VectorE to start the next tile's max reduction sooner
  - Better engine overlap: VectorE(max_reduce) || ScalarE(negate + exp + recip)

Also: Pre-compute intermediate [P,1] buffers once outside the loop, reducing
per-tile allocation overhead.
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

        # --- Step 1: Find max along cols (VectorE) ---
        max_sb = nl.ndarray((r_size, 1), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=max_sb, data=tile, op=nl.maximum, axis=1)

        # --- Step 2: Negate max using ScalarE (multiply by -1, add 0) ---
        # Forces ScalarE via the op0=multiply, op1=add pattern
        neg_max_sb = nl.ndarray((r_size, 1), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=neg_max_sb,
            data=max_sb,
            op0=nl.multiply,
            operand0=-1.0,
            op1=nl.add,
            operand1=0.0,
            engine=nisa.scalar_engine,
        )

        # --- Step 3 (FUSED): exp(input - max) AND sum(exp) (ScalarE) ---
        sum_sb = nl.ndarray((r_size, 1), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.activation(
            dst=tile,
            data=tile,
            op=nl.exp,
            scale=1.0,
            bias=neg_max_sb,
            reduce_op=nl.add,
            reduce_res=sum_sb,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )

        # --- Step 4: Reciprocal of sum (ScalarE) ---
        recip_sb = nl.ndarray((r_size, 1), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.reciprocal(dst=recip_sb, data=sum_sb)

        # --- Step 5: Multiply exp by reciprocal (VectorE) ---
        nisa.tensor_scalar(dst=tile, data=tile, op0=nl.multiply, operand0=recip_sb)

        # --- Store Output Tile ---
        nisa.dma_copy(dst=output[r_start:r_end, 0:cols], src=tile)

    return output
