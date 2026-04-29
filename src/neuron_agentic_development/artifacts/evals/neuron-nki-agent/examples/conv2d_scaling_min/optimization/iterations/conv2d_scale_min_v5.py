"""
Conv2D + Scale + Min NKI Kernel - v5: Weights outside batch loop + all optimizations

Optimizations over v4:
1. Load weight tiles ONCE outside the batch loop (not per batch).
   With affine_range on batch, compiler may partially optimize this,
   but explicit hoisting ensures weights are loaded exactly once.
   Saves up to (batch_size-1) * 9 * 16KB = 7 * 144KB = 1MB HBM traffic.
2. Multi-row PSUM tiling from v4 (2 rows packed in free dim)
3. Fused scale+negate from v2
4. Batched output stores per tile
"""

import nki
import nki.isa as nisa
import nki.language as nl


def div_ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


@nki.jit(platform_target="trn2")
def conv2d_scale_min_v5(
    input_hbm,              # [B, C_in, H_in * W_in]
    weight_hbm,             # [K*K, C_in, C_out]
    scale_factor,           # scalar
    batch_size,
    c_in,
    c_out,
    h_in,
    w_in,
    kernel_size,
) -> nl.ndarray:
    h_out = h_in - kernel_size + 1
    w_out = w_in - kernel_size + 1
    n_out = h_out * w_out
    ksq = kernel_size * kernel_size
    hw_in = h_in * w_in

    P_MAX = 128
    PSUM_F_MAX = 512

    ROWS_PER_TILE = 2
    n_tiles = h_out // ROWS_PER_TILE  # 31
    tile_free = ROWS_PER_TILE * w_out  # 124

    output_hbm = nl.ndarray(
        (batch_size, n_out),
        dtype=input_hbm.dtype,
        buffer=nl.shared_hbm
    )

    neg_scale = -1.0 * scale_factor

    # OPTIMIZATION: Load ALL weight tiles ONCE (outside batch loop)
    weight_tiles = []
    for k_pos in nl.affine_range(ksq):
        wt = nl.ndarray((c_in, c_out), dtype=weight_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=wt,
            src=weight_hbm[k_pos, 0:c_in, 0:c_out]
        )
        weight_tiles.append(wt)

    for b_idx in nl.affine_range(batch_size):
        # Load entire input for this batch
        input_batch = nl.ndarray((c_in, hw_in), dtype=input_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=input_batch,
            src=input_hbm[b_idx, 0:c_in, 0:hw_in]
        )

        # Working buffers
        input_slices = []
        for r in range(ROWS_PER_TILE):
            s = nl.ndarray((c_in, w_out), dtype=input_hbm.dtype, buffer=nl.sbuf)
            input_slices.append(s)
        psum_tile = nl.ndarray((P_MAX, PSUM_F_MAX), dtype=nl.float32, buffer=nl.psum)

        for tile_idx in nl.affine_range(n_tiles):
            nisa.memset(dst=psum_tile[0:c_out, 0:tile_free], value=0.0)

            for kh in nl.static_range(kernel_size):
                for kw in nl.static_range(kernel_size):
                    k_pos = kh * kernel_size + kw

                    for r in nl.static_range(ROWS_PER_TILE):
                        h_o = tile_idx * ROWS_PER_TILE + r
                        input_row = h_o + kh
                        input_col_start = input_row * w_in + kw

                        nisa.tensor_copy(
                            dst=input_slices[r][0:c_in, 0:w_out],
                            src=input_batch[0:c_in, input_col_start:input_col_start+w_out]
                        )

                        nisa.nc_matmul(
                            dst=psum_tile[0:c_out, r*w_out:(r+1)*w_out],
                            stationary=weight_tiles[k_pos][0:c_in, 0:c_out],
                            moving=input_slices[r][0:c_in, 0:w_out]
                        )

            # Batched post-processing
            conv_tile = nl.ndarray((P_MAX, PSUM_F_MAX), dtype=input_hbm.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=conv_tile[0:c_out, 0:tile_free],
                src=psum_tile[0:c_out, 0:tile_free]
            )

            neg_scaled = nl.ndarray((P_MAX, PSUM_F_MAX), dtype=input_hbm.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=neg_scaled[0:c_out, 0:tile_free],
                data=conv_tile[0:c_out, 0:tile_free],
                op0=nl.multiply,
                operand0=neg_scale
            )

            max_neg = nl.ndarray((1, PSUM_F_MAX), dtype=input_hbm.dtype, buffer=nl.sbuf)
            nisa.tensor_partition_reduce(
                dst=max_neg[0:1, 0:tile_free],
                op=nl.maximum,
                data=neg_scaled[0:c_out, 0:tile_free]
            )

            min_tile = nl.ndarray((1, PSUM_F_MAX), dtype=input_hbm.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=min_tile[0:1, 0:tile_free],
                data=max_neg[0:1, 0:tile_free],
                op0=nl.multiply,
                operand0=-1.0
            )

            out_start = tile_idx * tile_free
            nisa.dma_copy(
                dst=output_hbm[b_idx, out_start:out_start+tile_free],
                src=min_tile[0, 0:tile_free]
            )

    return output_hbm
