"""
SwiGLU MLP NKI Kernel - v1: Hoist x tile loading out of h_idx loop.

Optimization: In Phase 1, x tiles are loaded and transposed for every (h_idx, k_idx).
Since x doesn't depend on h_idx, we hoist x load+transpose before the h_idx loop,
caching all K tiles of x_transposed in SBUF. This eliminates (num_hidden_tiles - 1)
redundant x loads per token tile.

Target: gen3 (Trainium 2)
"""

import nki
import nki.isa as nisa
import nki.language as nl


def div_ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


P_MAX = 128
F_STAT_MAX = 128
F_MOV_MAX = 512


@nki.jit
def swiglu_mlp_kernel(
    x: nl.ndarray,
    gate_weight: nl.ndarray,
    up_weight: nl.ndarray,
    down_weight: nl.ndarray,
) -> nl.ndarray:
    """
    SwiGLU MLP kernel: output = down_proj(silu(gate_proj(x)) * up_proj(x))
    v1: x tiles hoisted out of h_idx loop
    """
    tokens, input_size = x.shape
    _, hidden_size = gate_weight.shape

    output = nl.ndarray((tokens, input_size), dtype=x.dtype, buffer=nl.shared_hbm)

    num_token_tiles = div_ceil(tokens, F_STAT_MAX)
    num_k_in_tiles = div_ceil(input_size, P_MAX)
    num_hidden_tiles = div_ceil(hidden_size, F_MOV_MAX)
    num_k_hidden_tiles = div_ceil(hidden_size, P_MAX)
    num_out_tiles = div_ceil(input_size, F_MOV_MAX)

    hidden_hbm = nl.ndarray((tokens, hidden_size), dtype=x.dtype, buffer=nl.shared_hbm)

    for t_idx in nl.affine_range(num_token_tiles):
        t_start = t_idx * F_STAT_MAX
        t_end = min(t_start + F_STAT_MAX, tokens)
        t_size = t_end - t_start

        # === OPTIMIZATION: Pre-load and transpose all x tiles for this token tile ===
        # These are reused across all h_idx iterations
        x_t_tiles = []
        for k_idx in nl.affine_range(num_k_in_tiles):
            k_start = k_idx * P_MAX
            k_end = min(k_start + P_MAX, input_size)
            k_size = k_end - k_start

            x_tile_sb = nl.ndarray((F_STAT_MAX, P_MAX), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=x_tile_sb[0:t_size, 0:k_size],
                src=x[t_start:t_end, k_start:k_end],
            )

            x_tile_f32 = nl.ndarray((F_STAT_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=x_tile_f32[0:t_size, 0:k_size],
                data=x_tile_sb[0:t_size, 0:k_size],
                op0=nl.multiply,
                operand0=1.0,
            )

            x_t_psum = nl.ndarray((P_MAX, F_STAT_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(
                dst=x_t_psum[0:k_size, 0:t_size],
                data=x_tile_f32[0:t_size, 0:k_size],
            )

            x_t_sb = nl.ndarray((P_MAX, F_STAT_MAX), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=x_t_sb[0:k_size, 0:t_size],
                src=x_t_psum[0:k_size, 0:t_size],
            )
            x_t_tiles.append(x_t_sb)

        # === Phase 1: Gate and Up Projections + SiLU + Multiply ===
        for h_idx in nl.affine_range(num_hidden_tiles):
            h_start = h_idx * F_MOV_MAX
            h_end = min(h_start + F_MOV_MAX, hidden_size)
            h_size = h_end - h_start

            gate_psum = nl.ndarray((F_STAT_MAX, F_MOV_MAX), dtype=nl.float32, buffer=nl.psum)
            up_psum = nl.ndarray((F_STAT_MAX, F_MOV_MAX), dtype=nl.float32, buffer=nl.psum)

            for k_idx in nl.affine_range(num_k_in_tiles):
                k_start = k_idx * P_MAX
                k_end = min(k_start + P_MAX, input_size)
                k_size = k_end - k_start

                # Reuse pre-loaded x_t tiles
                x_t_sb = x_t_tiles[k_idx]

                gate_w_sb = nl.ndarray((P_MAX, F_MOV_MAX), dtype=gate_weight.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=gate_w_sb[0:k_size, 0:h_size],
                    src=gate_weight[k_start:k_end, h_start:h_end],
                )

                up_w_sb = nl.ndarray((P_MAX, F_MOV_MAX), dtype=up_weight.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=up_w_sb[0:k_size, 0:h_size],
                    src=up_weight[k_start:k_end, h_start:h_end],
                )

                nisa.nc_matmul(
                    dst=gate_psum[0:t_size, 0:h_size],
                    stationary=x_t_sb[0:k_size, 0:t_size],
                    moving=gate_w_sb[0:k_size, 0:h_size],
                )

                nisa.nc_matmul(
                    dst=up_psum[0:t_size, 0:h_size],
                    stationary=x_t_sb[0:k_size, 0:t_size],
                    moving=up_w_sb[0:k_size, 0:h_size],
                )

            gate_sb = nl.ndarray((F_STAT_MAX, F_MOV_MAX), dtype=x.dtype, buffer=nl.sbuf)
            up_sb = nl.ndarray((F_STAT_MAX, F_MOV_MAX), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gate_sb[0:t_size, 0:h_size], src=gate_psum[0:t_size, 0:h_size])
            nisa.tensor_copy(dst=up_sb[0:t_size, 0:h_size], src=up_psum[0:t_size, 0:h_size])

            silu_sb = nl.ndarray((F_STAT_MAX, F_MOV_MAX), dtype=x.dtype, buffer=nl.sbuf)
            nisa.activation(
                dst=silu_sb[0:t_size, 0:h_size],
                data=gate_sb[0:t_size, 0:h_size],
                op=nl.silu,
            )

            hidden_sb = nl.ndarray((F_STAT_MAX, F_MOV_MAX), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=hidden_sb[0:t_size, 0:h_size],
                data1=silu_sb[0:t_size, 0:h_size],
                data2=up_sb[0:t_size, 0:h_size],
                op=nl.multiply,
            )

            nisa.dma_copy(
                dst=hidden_hbm[t_start:t_end, h_start:h_end],
                src=hidden_sb[0:t_size, 0:h_size],
            )

        # === Phase 2: Down Projection (unchanged) ===
        for o_idx in nl.affine_range(num_out_tiles):
            o_start = o_idx * F_MOV_MAX
            o_end = min(o_start + F_MOV_MAX, input_size)
            o_size = o_end - o_start

            out_psum = nl.ndarray((F_STAT_MAX, F_MOV_MAX), dtype=nl.float32, buffer=nl.psum)

            for k_idx in nl.affine_range(num_k_hidden_tiles):
                k_start = k_idx * P_MAX
                k_end = min(k_start + P_MAX, hidden_size)
                k_size = k_end - k_start

                hidden_tile_sb = nl.ndarray((F_STAT_MAX, P_MAX), dtype=x.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=hidden_tile_sb[0:t_size, 0:k_size],
                    src=hidden_hbm[t_start:t_end, k_start:k_end],
                )

                hidden_tile_f32 = nl.ndarray((F_STAT_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=hidden_tile_f32[0:t_size, 0:k_size],
                    data=hidden_tile_sb[0:t_size, 0:k_size],
                    op0=nl.multiply,
                    operand0=1.0,
                )

                hidden_t_psum = nl.ndarray((P_MAX, F_STAT_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(
                    dst=hidden_t_psum[0:k_size, 0:t_size],
                    data=hidden_tile_f32[0:t_size, 0:k_size],
                )

                hidden_t_sb = nl.ndarray((P_MAX, F_STAT_MAX), dtype=x.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=hidden_t_sb[0:k_size, 0:t_size],
                    src=hidden_t_psum[0:k_size, 0:t_size],
                )

                down_w_sb = nl.ndarray((P_MAX, F_MOV_MAX), dtype=down_weight.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=down_w_sb[0:k_size, 0:o_size],
                    src=down_weight[k_start:k_end, o_start:o_end],
                )

                nisa.nc_matmul(
                    dst=out_psum[0:t_size, 0:o_size],
                    stationary=hidden_t_sb[0:k_size, 0:t_size],
                    moving=down_w_sb[0:k_size, 0:o_size],
                )

            out_sb = nl.ndarray((F_STAT_MAX, F_MOV_MAX), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out_sb[0:t_size, 0:o_size], src=out_psum[0:t_size, 0:o_size])

            nisa.dma_copy(
                dst=output[t_start:t_end, o_start:o_end],
                src=out_sb[0:t_size, 0:o_size],
            )

    return output
