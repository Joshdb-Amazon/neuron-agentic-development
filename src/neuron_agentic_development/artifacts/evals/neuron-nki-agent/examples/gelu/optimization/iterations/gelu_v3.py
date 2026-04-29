"""GELU v3: In-place activation to halve SBUF footprint per tile.

Optimization: Apply GELU in-place (dst=src) to use only one SBUF buffer per
tile instead of two. This halves SBUF pressure from 8MB to 4MB per tile,
allowing the compiler to keep more tiles in-flight simultaneously and better
overlap DMA load/store operations.
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit(platform_target="trn2")
def gelu_kernel(input_hbm):
    rows, cols = input_hbm.shape

    TILE_P = nl.tile_size.pmax  # 128

    num_row_tiles = rows // TILE_P

    output_hbm = nl.ndarray(input_hbm.shape, dtype=input_hbm.dtype, buffer=nl.shared_hbm)

    for i_row in nl.affine_range(num_row_tiles):
        # Single buffer - apply GELU in-place
        tile = nl.ndarray((TILE_P, cols), dtype=input_hbm.dtype, buffer=nl.sbuf)

        # Load
        nisa.dma_copy(dst=tile, src=input_hbm[i_row * TILE_P : (i_row + 1) * TILE_P, 0:cols])

        # In-place GELU
        nisa.activation(dst=tile, op=nl.gelu, data=tile)

        # Store
        nisa.dma_copy(dst=output_hbm[i_row * TILE_P : (i_row + 1) * TILE_P, 0:cols], src=tile)

    return output_hbm
