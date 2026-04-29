"""NKI kernel for element-wise GELU activation.

Implements GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2))) using the hardware
Scalar Engine's built-in GELU activation function (nl.gelu).

Input:  (rows, cols) float32 tensor in HBM
Output: (rows, cols) float32 tensor in HBM
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit(platform_target="trn2")
def gelu_kernel(input_hbm):
    """Apply GELU activation element-wise to a 2D tensor.

    Args:
        input_hbm: Input tensor of shape (rows, cols) in HBM.

    Returns:
        Output tensor of shape (rows, cols) with GELU applied.
    """
    rows, cols = input_hbm.shape

    TILE_P = nl.tile_size.pmax  # 128

    num_row_tiles = rows // TILE_P

    # Allocate output in HBM
    output_hbm = nl.ndarray(input_hbm.shape, dtype=input_hbm.dtype, buffer=nl.shared_hbm)

    for i_row in nl.affine_range(num_row_tiles):
        # Allocate SBUF tiles for input and output
        tile_in = nl.ndarray((TILE_P, cols), dtype=input_hbm.dtype, buffer=nl.sbuf)
        tile_out = nl.ndarray((TILE_P, cols), dtype=input_hbm.dtype, buffer=nl.sbuf)

        # Load tile from HBM to SBUF
        nisa.dma_copy(dst=tile_in, src=input_hbm[i_row * TILE_P : (i_row + 1) * TILE_P, 0:cols])

        # Apply GELU activation using Scalar Engine
        nisa.activation(dst=tile_out, op=nl.gelu, data=tile_in)

        # Store result from SBUF back to HBM
        nisa.dma_copy(dst=output_hbm[i_row * TILE_P : (i_row + 1) * TILE_P, 0:cols], src=tile_out)

    return output_hbm
