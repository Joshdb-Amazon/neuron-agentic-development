"""GELU v2: Double buffering with sequential_range for DMA/compute overlap.

Optimization: Manually overlap DMA loads with ScalarE GELU computation using
ping-pong buffers. While the compiler can pipeline with affine_range, explicit
double buffering with sequential_range ensures load(N+1) overlaps with
compute(N), hiding scalar engine latency behind DMA.
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

    # Double buffers for ping-pong
    buf_in_a = nl.ndarray((TILE_P, cols), dtype=input_hbm.dtype, buffer=nl.sbuf)
    buf_in_b = nl.ndarray((TILE_P, cols), dtype=input_hbm.dtype, buffer=nl.sbuf)
    buf_out_a = nl.ndarray((TILE_P, cols), dtype=input_hbm.dtype, buffer=nl.sbuf)
    buf_out_b = nl.ndarray((TILE_P, cols), dtype=input_hbm.dtype, buffer=nl.sbuf)

    # Prefetch first tile
    nisa.dma_copy(dst=buf_in_a, src=input_hbm[0:TILE_P, 0:cols])

    for i_row in nl.sequential_range(num_row_tiles - 1):
        # Determine current and next buffers based on iteration parity
        # Even iterations: compute on A, prefetch into B
        # Odd iterations: compute on B, prefetch into A
        if i_row % 2 == 0:
            cur_in = buf_in_a
            cur_out = buf_out_a
            nxt_in = buf_in_b
        else:
            cur_in = buf_in_b
            cur_out = buf_out_b
            nxt_in = buf_in_a

        # Prefetch next tile (overlaps with compute below)
        nisa.dma_copy(
            dst=nxt_in,
            src=input_hbm[(i_row + 1) * TILE_P : (i_row + 2) * TILE_P, 0:cols]
        )

        # Compute GELU on current tile
        nisa.activation(dst=cur_out, op=nl.gelu, data=cur_in)

        # Store current result
        nisa.dma_copy(
            dst=output_hbm[i_row * TILE_P : (i_row + 1) * TILE_P, 0:cols],
            src=cur_out
        )

    # Process last tile
    last = num_row_tiles - 1
    if last % 2 == 0:
        last_in = buf_in_a
        last_out = buf_out_a
    else:
        last_in = buf_in_b
        last_out = buf_out_b

    nisa.activation(dst=last_out, op=nl.gelu, data=last_in)
    nisa.dma_copy(
        dst=output_hbm[last * TILE_P : (last + 1) * TILE_P, 0:cols],
        src=last_out
    )

    return output_hbm
