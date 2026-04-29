"""
Conv2D + Scale + Min NKI Kernel - v1: Hoist weight loading

Optimization: Load all 9 weight tiles once per batch instead of
reloading them for every output row. This eliminates the 17.8x
HBM reload factor observed in baseline profiling.
"""

import nki
import nki.isa as nisa
import nki.language as nl


def div_ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


@nki.jit(platform_target="trn2")
def conv2d_scale_min_v1(
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

    output_hbm = nl.ndarray(
        (batch_size, n_out),
        dtype=input_hbm.dtype,
        buffer=nl.shared_hbm
    )

    for b_idx in nl.affine_range(batch_size):
        # Load entire input for this batch [C_in, H_in * W_in]
        input_batch = nl.ndarray((c_in, hw_in), dtype=input_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=input_batch,
            src=input_hbm[b_idx, 0:c_in, 0:hw_in]
        )

        # OPTIMIZATION: Load ALL weight tiles ONCE per batch (not per row!)
        # Weight shape: [K*K, C_in, C_out] -> load each [C_in, C_out] tile
        weight_tiles = []
        for k_pos in nl.affine_range(ksq):
            wt = nl.ndarray((c_in, c_out), dtype=weight_hbm.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=wt,
                src=weight_hbm[k_pos, 0:c_in, 0:c_out]
            )
            weight_tiles.append(wt)

        # Working buffers
        input_slice = nl.ndarray((c_in, w_out), dtype=input_hbm.dtype, buffer=nl.sbuf)
        psum_row = nl.ndarray((P_MAX, PSUM_F_MAX), dtype=nl.float32, buffer=nl.psum)

        for h_o in nl.affine_range(h_out):
            # Clear PSUM
            nisa.memset(dst=psum_row[0:c_out, 0:w_out], value=0.0)

            # Accumulate over kernel positions using pre-loaded weights
            for kh in nl.static_range(kernel_size):
                for kw in nl.static_range(kernel_size):
                    k_pos = kh * kernel_size + kw

                    input_row = h_o + kh
                    input_col_start = input_row * w_in + kw

                    nisa.tensor_copy(
                        dst=input_slice[0:c_in, 0:w_out],
                        src=input_batch[0:c_in, input_col_start:input_col_start+w_out]
                    )

                    nisa.nc_matmul(
                        dst=psum_row[0:c_out, 0:w_out],
                        stationary=weight_tiles[k_pos][0:c_in, 0:c_out],
                        moving=input_slice[0:c_in, 0:w_out]
                    )

            # PSUM -> SBUF
            conv_row = nl.ndarray((P_MAX, PSUM_F_MAX), dtype=input_hbm.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=conv_row[0:c_out, 0:w_out],
                src=psum_row[0:c_out, 0:w_out]
            )

            # Scale
            scaled = nl.ndarray((P_MAX, PSUM_F_MAX), dtype=input_hbm.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=scaled[0:c_out, 0:w_out],
                data=conv_row[0:c_out, 0:w_out],
                op0=nl.multiply,
                operand0=scale_factor
            )

            # Min = -max(-x)
            neg_scaled = nl.ndarray((P_MAX, PSUM_F_MAX), dtype=input_hbm.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=neg_scaled[0:c_out, 0:w_out],
                data=scaled[0:c_out, 0:w_out],
                op0=nl.multiply,
                operand0=-1.0
            )

            max_neg = nl.ndarray((1, PSUM_F_MAX), dtype=input_hbm.dtype, buffer=nl.sbuf)
            nisa.tensor_partition_reduce(
                dst=max_neg[0:1, 0:w_out],
                op=nl.maximum,
                data=neg_scaled[0:c_out, 0:w_out]
            )

            min_row = nl.ndarray((1, PSUM_F_MAX), dtype=input_hbm.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=min_row[0:1, 0:w_out],
                data=max_neg[0:1, 0:w_out],
                op0=nl.multiply,
                operand0=-1.0
            )

            out_row_start = h_o * w_out
            nisa.dma_copy(
                dst=output_hbm[b_idx, out_row_start:out_row_start+w_out],
                src=min_row[0, 0:w_out]
            )

    return output_hbm
