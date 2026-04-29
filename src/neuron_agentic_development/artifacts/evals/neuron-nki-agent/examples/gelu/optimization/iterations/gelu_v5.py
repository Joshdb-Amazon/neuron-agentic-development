"""GELU v5: Split affine_range + hw_dge + in-place + interleaved rows.

Optimization: Combines all best techniques:
1. Hardware DGE (hw_dge) — efficient DMA descriptor generation
2. In-place activation — halves SBUF footprint
3. Split into 4 affine_range blocks of 4 rows each — reduces compiler
   unrolling pressure while maintaining large tile DMA efficiency
4. Interleaved row processing within each block for HBM bank diversity

SBUF per block: 4 tiles × 4MB = 16MB (fits in 24MB SBUF)
DMA: 32 transfers of 4MB each (same as baseline)
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit(platform_target="trn2")
def gelu_kernel(input_hbm):
    rows, cols = input_hbm.shape

    TILE_P = nl.tile_size.pmax  # 128

    num_row_tiles = rows // TILE_P  # 16
    BLOCK = 4  # Process 4 tiles per affine_range block

    output_hbm = nl.ndarray(input_hbm.shape, dtype=input_hbm.dtype, buffer=nl.shared_hbm)

    # Block 0: rows 0-3
    for i in nl.affine_range(BLOCK):
        tile = nl.ndarray((TILE_P, cols), dtype=input_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile, src=input_hbm[i * TILE_P:(i + 1) * TILE_P, 0:cols],
                      dge_mode=nisa.dge_mode.hwdge)
        nisa.activation(dst=tile, op=nl.gelu, data=tile)
        nisa.dma_copy(dst=output_hbm[i * TILE_P:(i + 1) * TILE_P, 0:cols], src=tile,
                      dge_mode=nisa.dge_mode.hwdge)

    # Block 1: rows 4-7
    for i in nl.affine_range(BLOCK):
        row = i + BLOCK
        tile = nl.ndarray((TILE_P, cols), dtype=input_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile, src=input_hbm[row * TILE_P:(row + 1) * TILE_P, 0:cols],
                      dge_mode=nisa.dge_mode.hwdge)
        nisa.activation(dst=tile, op=nl.gelu, data=tile)
        nisa.dma_copy(dst=output_hbm[row * TILE_P:(row + 1) * TILE_P, 0:cols], src=tile,
                      dge_mode=nisa.dge_mode.hwdge)

    # Block 2: rows 8-11
    for i in nl.affine_range(BLOCK):
        row = i + 2 * BLOCK
        tile = nl.ndarray((TILE_P, cols), dtype=input_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile, src=input_hbm[row * TILE_P:(row + 1) * TILE_P, 0:cols],
                      dge_mode=nisa.dge_mode.hwdge)
        nisa.activation(dst=tile, op=nl.gelu, data=tile)
        nisa.dma_copy(dst=output_hbm[row * TILE_P:(row + 1) * TILE_P, 0:cols], src=tile,
                      dge_mode=nisa.dge_mode.hwdge)

    # Block 3: rows 12-15
    for i in nl.affine_range(BLOCK):
        row = i + 3 * BLOCK
        tile = nl.ndarray((TILE_P, cols), dtype=input_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile, src=input_hbm[row * TILE_P:(row + 1) * TILE_P, 0:cols],
                      dge_mode=nisa.dge_mode.hwdge)
        nisa.activation(dst=tile, op=nl.gelu, data=tile)
        nisa.dma_copy(dst=output_hbm[row * TILE_P:(row + 1) * TILE_P, 0:cols], src=tile,
                      dge_mode=nisa.dge_mode.hwdge)

    return output_hbm
