"""
SwiGLU MLP NKI Kernel for AWS Trainium.

Implements the SwiGLU MLP operation used in LLaMA-style models:
    output = down_proj(silu(gate_proj(x)) * up_proj(x))

This kernel runs on a single NeuronCore and uses tiled matrix multiplication
to handle dimensions that exceed hardware limits.

Target: gen3 (Trainium 2)
"""

import nki
import nki.isa as nisa
import nki.language as nl


# === Self-contained utilities ===

def kernel_assert(condition: bool, error_text: str):
    """Kernel-safe assertion that raises informative errors."""
    assert condition, f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"


def div_ceil(n: int, d: int) -> int:
    """Ceiling division without floating point."""
    return (n + d - 1) // d


# === Hardware constants ===
P_MAX = 128        # Partition dimension max (contraction dim for matmul)
F_STAT_MAX = 128   # Stationary free dimension max (token tile size)
F_MOV_MAX = 512    # Moving free dimension max (gen2/gen3)


@nki.jit
def swiglu_mlp_kernel(
    x: nl.ndarray,
    gate_weight: nl.ndarray,
    up_weight: nl.ndarray,
    down_weight: nl.ndarray,
) -> nl.ndarray:
    """
    SwiGLU MLP kernel: output = down_proj(silu(gate_proj(x)) * up_proj(x))

    This kernel processes tokens in tiles, computing gate and up projections,
    applying SiLU activation and element-wise multiplication, then computing
    the down projection.

    Args:
        x (nl.ndarray): Input [tokens, input_size] @ HBM
        gate_weight (nl.ndarray): Gate weights [input_size, hidden_size] @ HBM (W.T format)
        up_weight (nl.ndarray): Up weights [input_size, hidden_size] @ HBM (W.T format)
        down_weight (nl.ndarray): Down weights [hidden_size, input_size] @ HBM (W.T format)

    Returns:
        nl.ndarray: Output [tokens, input_size] @ HBM

    Note:
        - Supports any token count via internal tiling (tiles of 128 tokens)
        - Weights must be in transposed format (W.T)
    """
    tokens, input_size = x.shape
    _, hidden_size = gate_weight.shape

    # Allocate output
    output = nl.ndarray((tokens, input_size), dtype=x.dtype, buffer=nl.shared_hbm)

    # Calculate tile counts
    num_token_tiles = div_ceil(tokens, F_STAT_MAX)        # Token tiles
    num_k_in_tiles = div_ceil(input_size, P_MAX)          # K tiles for input projections
    num_hidden_tiles = div_ceil(hidden_size, F_MOV_MAX)   # Hidden dimension tiles
    num_k_hidden_tiles = div_ceil(hidden_size, P_MAX)     # K tiles for down projection
    num_out_tiles = div_ceil(input_size, F_MOV_MAX)       # Output dimension tiles

    # Temporary HBM for hidden activations [tokens, hidden_size]
    hidden_hbm = nl.ndarray((tokens, hidden_size), dtype=x.dtype, buffer=nl.shared_hbm)

    # === Process each token tile ===
    for t_idx in nl.affine_range(num_token_tiles):
        t_start = t_idx * F_STAT_MAX
        t_end = min(t_start + F_STAT_MAX, tokens)
        t_size = t_end - t_start

        # === Phase 1: Gate and Up Projections + SiLU + Multiply ===
        for h_idx in nl.affine_range(num_hidden_tiles):
            h_start = h_idx * F_MOV_MAX
            h_end = min(h_start + F_MOV_MAX, hidden_size)
            h_size = h_end - h_start

            # PSUM accumulators for gate and up projections
            gate_psum = nl.ndarray((F_STAT_MAX, F_MOV_MAX), dtype=nl.float32, buffer=nl.psum)
            up_psum = nl.ndarray((F_STAT_MAX, F_MOV_MAX), dtype=nl.float32, buffer=nl.psum)

            # Accumulate over input dimension (K)
            for k_idx in nl.affine_range(num_k_in_tiles):
                k_start = k_idx * P_MAX
                k_end = min(k_start + P_MAX, input_size)
                k_size = k_end - k_start

                # === Load and transpose x tile ===
                x_tile_sb = nl.ndarray((F_STAT_MAX, P_MAX), dtype=x.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=x_tile_sb[0:t_size, 0:k_size],
                    src=x[t_start:t_end, k_start:k_end],
                )

                # Cast to float32 for transpose
                x_tile_f32 = nl.ndarray((F_STAT_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=x_tile_f32[0:t_size, 0:k_size],
                    data=x_tile_sb[0:t_size, 0:k_size],
                    op0=nl.multiply,
                    operand0=1.0,
                )

                # Transpose: [t_size, k_size] -> [k_size, t_size]
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

                # Load weight tiles
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

                # Gate and Up matmuls
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

            # Copy PSUM to SBUF
            gate_sb = nl.ndarray((F_STAT_MAX, F_MOV_MAX), dtype=x.dtype, buffer=nl.sbuf)
            up_sb = nl.ndarray((F_STAT_MAX, F_MOV_MAX), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gate_sb[0:t_size, 0:h_size], src=gate_psum[0:t_size, 0:h_size])
            nisa.tensor_copy(dst=up_sb[0:t_size, 0:h_size], src=up_psum[0:t_size, 0:h_size])

            # Apply SiLU to gate
            silu_sb = nl.ndarray((F_STAT_MAX, F_MOV_MAX), dtype=x.dtype, buffer=nl.sbuf)
            nisa.activation(
                dst=silu_sb[0:t_size, 0:h_size],
                data=gate_sb[0:t_size, 0:h_size],
                op=nl.silu,
            )

            # Element-wise multiply: hidden = silu(gate) * up
            hidden_sb = nl.ndarray((F_STAT_MAX, F_MOV_MAX), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=hidden_sb[0:t_size, 0:h_size],
                data1=silu_sb[0:t_size, 0:h_size],
                data2=up_sb[0:t_size, 0:h_size],
                op=nl.multiply,
            )

            # Store hidden tile to HBM
            nisa.dma_copy(
                dst=hidden_hbm[t_start:t_end, h_start:h_end],
                src=hidden_sb[0:t_size, 0:h_size],
            )

        # === Phase 2: Down Projection ===
        for o_idx in nl.affine_range(num_out_tiles):
            o_start = o_idx * F_MOV_MAX
            o_end = min(o_start + F_MOV_MAX, input_size)
            o_size = o_end - o_start

            out_psum = nl.ndarray((F_STAT_MAX, F_MOV_MAX), dtype=nl.float32, buffer=nl.psum)

            # Accumulate over hidden dimension (K)
            for k_idx in nl.affine_range(num_k_hidden_tiles):
                k_start = k_idx * P_MAX
                k_end = min(k_start + P_MAX, hidden_size)
                k_size = k_end - k_start

                # Load hidden tile
                hidden_tile_sb = nl.ndarray((F_STAT_MAX, P_MAX), dtype=x.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=hidden_tile_sb[0:t_size, 0:k_size],
                    src=hidden_hbm[t_start:t_end, k_start:k_end],
                )

                # Cast and transpose
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

                # Load down weight tile
                down_w_sb = nl.ndarray((P_MAX, F_MOV_MAX), dtype=down_weight.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=down_w_sb[0:k_size, 0:o_size],
                    src=down_weight[k_start:k_end, o_start:o_end],
                )

                # Down projection matmul
                nisa.nc_matmul(
                    dst=out_psum[0:t_size, 0:o_size],
                    stationary=hidden_t_sb[0:k_size, 0:t_size],
                    moving=down_w_sb[0:k_size, 0:o_size],
                )

            # Copy to SBUF and store to output
            out_sb = nl.ndarray((F_STAT_MAX, F_MOV_MAX), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out_sb[0:t_size, 0:o_size], src=out_psum[0:t_size, 0:o_size])

            nisa.dma_copy(
                dst=output[t_start:t_end, o_start:o_end],
                src=out_sb[0:t_size, 0:o_size],
            )

    return output
