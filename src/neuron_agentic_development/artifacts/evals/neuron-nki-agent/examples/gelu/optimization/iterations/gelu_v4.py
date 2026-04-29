"""GELU v4: Hardware DGE mode for more efficient DMA descriptor generation.

Optimization: Use hardware-based DMA Gather Engine (hw_dge) instead of the
default software DGE. On Trainium2, hardware DGE generates DMA descriptors
on-demand without consuming SBUF space or GpSimdE cycles. This frees up
GpSimdE and potentially reduces DMA initiation overhead.

Combined with in-place activation to minimize SBUF footprint.
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
        tile = nl.ndarray((TILE_P, cols), dtype=input_hbm.dtype, buffer=nl.sbuf)

        # Load with hardware DGE
        nisa.dma_copy(
            dst=tile,
            src=input_hbm[i_row * TILE_P : (i_row + 1) * TILE_P, 0:cols],
            dge_mode=nisa.dge_mode.hwdge
        )

        # In-place GELU via ScalarEngine
        nisa.activation(dst=tile, op=nl.gelu, data=tile)

        # Store with hardware DGE
        nisa.dma_copy(
            dst=output_hbm[i_row * TILE_P : (i_row + 1) * TILE_P, 0:cols],
            src=tile,
            dge_mode=nisa.dge_mode.hwdge
        )

    return output_hbm
