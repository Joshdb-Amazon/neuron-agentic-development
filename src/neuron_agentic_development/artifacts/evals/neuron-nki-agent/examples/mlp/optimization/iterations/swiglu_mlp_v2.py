"""
SwiGLU MLP NKI Kernel - v2: Fuse gate/up projections with down projection.

Optimization: Eliminate hidden HBM roundtrip by processing each hidden tile's
down projection immediately after computing it. Instead of:
  Phase 1: Compute all hidden tiles -> write to HBM
  Phase 2: Read hidden tiles from HBM -> down projection
We now do:
  For each hidden tile: compute hidden -> immediately use for down projection -> accumulate

This eliminates writing/reading hidden_hbm (tokens * hidden_size * 2 bytes).
Also keeps x tile hoisting from v1.

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
    v2: Fused phases, no hidden HBM roundtrip
    """
    tokens, input_size = x.shape
    _, hidden_size = gate_weight.shape

    output = nl.ndarray((tokens, input_size), dtype=x.dtype, buffer=nl.shared_hbm)

    num_token_tiles = div_ceil(tokens, F_STAT_MAX)
    num_k_in_tiles = div_ceil(input_size, P_MAX)
    num_hidden_tiles = div_ceil(hidden_size, F_MOV_MAX)
    num_out_tiles = div_ceil(input_size, F_MOV_MAX)
    # Number of K sub-tiles within one hidden tile for down projection
    num_k_sub = div_ceil(F_MOV_MAX, P_MAX)  # 512/128 = 4

    for t_idx in nl.affine_range(num_token_tiles):
        t_start = t_idx * F_STAT_MAX
        t_end = min(t_start + F_STAT_MAX, tokens)
        t_size = t_end - t_start

        # === Pre-load and transpose all x tiles (from v1) ===
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

        # === Initialize output accumulators in SBUF (one per output tile) ===
        out_accum = []
        for o_idx in nl.affine_range(num_out_tiles):
            o_start = o_idx * F_MOV_MAX
            o_end = min(o_start + F_MOV_MAX, input_size)
            o_size = o_end - o_start

            acc = nl.ndarray((F_STAT_MAX, F_MOV_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=acc, value=0.0)
            out_accum.append(acc)

        # === Fused: For each hidden tile, compute gate/up -> silu*up -> down proj ===
        for h_idx in nl.sequential_range(num_hidden_tiles):
            h_start = h_idx * F_MOV_MAX
            h_end = min(h_start + F_MOV_MAX, hidden_size)
            h_size = h_end - h_start

            # --- Gate and Up projections ---
            gate_psum = nl.ndarray((F_STAT_MAX, F_MOV_MAX), dtype=nl.float32, buffer=nl.psum)
            up_psum = nl.ndarray((F_STAT_MAX, F_MOV_MAX), dtype=nl.float32, buffer=nl.psum)

            for k_idx in nl.affine_range(num_k_in_tiles):
                k_start = k_idx * P_MAX
                k_end = min(k_start + P_MAX, input_size)
                k_size = k_end - k_start

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

            # --- SiLU + Multiply ---
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

            # --- Down projection: accumulate hidden_sb @ down_weight[h_start:h_end, :] ---
            # hidden_sb is [t_size, h_size] where h_size <= 512
            # We need it as [k, t_size] for nc_matmul stationary
            # Tile the h_size dimension into P_MAX=128 chunks for K dimension

            for k_sub_idx in nl.affine_range(num_k_sub):
                ks_start = k_sub_idx * P_MAX
                ks_end = min(ks_start + P_MAX, h_size)
                ks_size = ks_end - ks_start

                if ks_size <= 0:
                    continue

                # Extract and transpose the K-sub-tile of hidden
                # hidden_sb[:, ks_start:ks_end] is [t_size, ks_size]
                hidden_sub_f32 = nl.ndarray((F_STAT_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=hidden_sub_f32[0:t_size, 0:ks_size],
                    data=hidden_sb[0:t_size, ks_start:ks_end],
                    op0=nl.multiply,
                    operand0=1.0,
                )

                hidden_sub_t_psum = nl.ndarray((P_MAX, F_STAT_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(
                    dst=hidden_sub_t_psum[0:ks_size, 0:t_size],
                    data=hidden_sub_f32[0:t_size, 0:ks_size],
                )

                hidden_sub_t_sb = nl.ndarray((P_MAX, F_STAT_MAX), dtype=x.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=hidden_sub_t_sb[0:ks_size, 0:t_size],
                    src=hidden_sub_t_psum[0:ks_size, 0:t_size],
                )

                # For each output tile, load down weight and matmul
                for o_idx in nl.affine_range(num_out_tiles):
                    o_start = o_idx * F_MOV_MAX
                    o_end = min(o_start + F_MOV_MAX, input_size)
                    o_size = o_end - o_start

                    down_w_sb = nl.ndarray((P_MAX, F_MOV_MAX), dtype=down_weight.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=down_w_sb[0:ks_size, 0:o_size],
                        src=down_weight[h_start + ks_start:h_start + ks_end, o_start:o_end],
                    )

                    down_psum = nl.ndarray((F_STAT_MAX, F_MOV_MAX), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(
                        dst=down_psum[0:t_size, 0:o_size],
                        stationary=hidden_sub_t_sb[0:ks_size, 0:t_size],
                        moving=down_w_sb[0:ks_size, 0:o_size],
                    )

                    # Accumulate into SBUF
                    nisa.tensor_tensor(
                        dst=out_accum[o_idx][0:t_size, 0:o_size],
                        data1=out_accum[o_idx][0:t_size, 0:o_size],
                        data2=down_psum[0:t_size, 0:o_size],
                        op=nl.add,
                    )

        # === Write accumulated output to HBM ===
        for o_idx in nl.affine_range(num_out_tiles):
            o_start = o_idx * F_MOV_MAX
            o_end = min(o_start + F_MOV_MAX, input_size)
            o_size = o_end - o_start

            out_sb = nl.ndarray((F_STAT_MAX, F_MOV_MAX), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out_sb[0:t_size, 0:o_size], src=out_accum[o_idx][0:t_size, 0:o_size])

            nisa.dma_copy(
                dst=output[t_start:t_end, o_start:o_end],
                src=out_sb[0:t_size, 0:o_size],
            )

    return output
