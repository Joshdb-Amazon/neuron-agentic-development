"""
NKI Softmax Kernel v4 - Preload all tiles then batch-process.

Optimization over v3:
  - Load all input tiles into SBUF first, then process them.
  - This allows DMA engines to work continuously without waiting for compute,
    and compute engines to work without waiting for DMA.
  - With 8 tiles of 128x8192xfp32 = 4MB each, total = 32MB (fits in 24MB SBUF? No!)
  - Actually 128x8192x4 = 4MB per tile, 8 tiles = 32MB > 24MB SBUF limit.
  - So preload 4 tiles at a time (16MB fits in 24MB SBUF with room for intermediates)
  - Two-phase: preload batch of tiles, process batch, preload next batch, process next batch

Actually, let's try a simpler approach: preload tiles in pairs and overlap.
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

    # Process tiles in pairs - preload both, then compute both
    num_pairs = div_ceil(num_row_tiles, 2)

    for pair_idx in nl.affine_range(num_pairs):
        # Tile A (always exists)
        r_start_a = (pair_idx * 2) * P_MAX
        r_end_a = min(r_start_a + P_MAX, rows)
        r_size_a = r_end_a - r_start_a

        # Preload tile A
        tile_a = nl.ndarray((r_size_a, cols), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile_a, src=input_tensor[r_start_a:r_end_a, 0:cols])

        # Preload tile B if it exists
        r_start_b = (pair_idx * 2 + 1) * P_MAX
        has_tile_b = r_start_b < rows
        if has_tile_b:
            r_end_b = min(r_start_b + P_MAX, rows)
            r_size_b = r_end_b - r_start_b
            tile_b = nl.ndarray((r_size_b, cols), dtype=input_tensor.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=tile_b, src=input_tensor[r_start_b:r_end_b, 0:cols])

        # Process tile A
        max_a = nl.ndarray((r_size_a, 1), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=max_a, data=tile_a, op=nl.maximum, axis=1)
        neg_max_a = nl.ndarray((r_size_a, 1), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=neg_max_a, data=max_a, op0=nl.multiply, operand0=-1.0)
        sum_a = nl.ndarray((r_size_a, 1), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.activation(dst=tile_a, data=tile_a, op=nl.exp, scale=1.0, bias=neg_max_a,
                        reduce_op=nl.add, reduce_res=sum_a,
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
        recip_a = nl.ndarray((r_size_a, 1), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.reciprocal(dst=recip_a, data=sum_a)
        nisa.tensor_scalar(dst=tile_a, data=tile_a, op0=nl.multiply, operand0=recip_a)
        nisa.dma_copy(dst=output[r_start_a:r_end_a, 0:cols], src=tile_a)

        # Process tile B
        if has_tile_b:
            max_b = nl.ndarray((r_size_b, 1), dtype=input_tensor.dtype, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=max_b, data=tile_b, op=nl.maximum, axis=1)
            neg_max_b = nl.ndarray((r_size_b, 1), dtype=input_tensor.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=neg_max_b, data=max_b, op0=nl.multiply, operand0=-1.0)
            sum_b = nl.ndarray((r_size_b, 1), dtype=input_tensor.dtype, buffer=nl.sbuf)
            nisa.activation(dst=tile_b, data=tile_b, op=nl.exp, scale=1.0, bias=neg_max_b,
                            reduce_op=nl.add, reduce_res=sum_b,
                            reduce_cmd=nisa.reduce_cmd.reset_reduce)
            recip_b = nl.ndarray((r_size_b, 1), dtype=input_tensor.dtype, buffer=nl.sbuf)
            nisa.reciprocal(dst=recip_b, data=sum_b)
            nisa.tensor_scalar(dst=tile_b, data=tile_b, op0=nl.multiply, operand0=recip_b)
            nisa.dma_copy(dst=output[r_start_b:r_end_b, 0:cols], src=tile_b)

    return output
