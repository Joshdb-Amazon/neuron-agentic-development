"""GELU v1: 2D tiling for better DMA pipelining.

Optimization: Tile along both rows AND columns. Smaller tiles (128x2048 = 1MB
vs baseline 128x8192 = 4MB) create 4x more DMA operations. With affine_range,
the compiler can better interleave loads/stores/compute across tiles, improving
DMA/ScalarE overlap and overall bandwidth utilization.
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit(platform_target="trn2")
def gelu_kernel(input_hbm):
    rows, cols = input_hbm.shape

    TILE_P = nl.tile_size.pmax  # 128
    TILE_F = 2048  # Tile free dim: 4 col tiles instead of 1

    num_row_tiles = rows // TILE_P
    num_col_tiles = cols // TILE_F

    output_hbm = nl.ndarray(input_hbm.shape, dtype=input_hbm.dtype, buffer=nl.shared_hbm)

    for i_row in nl.affine_range(num_row_tiles):
        for i_col in nl.affine_range(num_col_tiles):
            tile_in = nl.ndarray((TILE_P, TILE_F), dtype=input_hbm.dtype, buffer=nl.sbuf)
            tile_out = nl.ndarray((TILE_P, TILE_F), dtype=input_hbm.dtype, buffer=nl.sbuf)

            nisa.dma_copy(
                dst=tile_in,
                src=input_hbm[i_row * TILE_P : (i_row + 1) * TILE_P,
                              i_col * TILE_F : (i_col + 1) * TILE_F]
            )

            nisa.activation(dst=tile_out, op=nl.gelu, data=tile_in)

            nisa.dma_copy(
                dst=output_hbm[i_row * TILE_P : (i_row + 1) * TILE_P,
                               i_col * TILE_F : (i_col + 1) * TILE_F],
                src=tile_out
            )

    return output_hbm
