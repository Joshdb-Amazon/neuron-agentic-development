"""
Fused matmul + subtract + multiply + ReLU NKI kernel.

Computes: relu((x @ weight_t + bias - subtract_value) * multiply_value)

Dimensions:
    M: batch_size (tiled by 128)
    K: in_features (contraction, tiled by 128)
    N: out_features (tiled by 512 for gen3 PSUM limit)
"""

import nki
import nki.isa as nisa
import nki.language as nl


# === Self-contained utilities ===

def kernel_assert(condition: bool, error_text: str):
    """Assert with NKI-formatted error message."""
    assert condition, f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"


def div_ceil(n: int, d: int) -> int:
    """Ceiling division: smallest integer >= n/d."""
    return (n + d - 1) // d


def stream_shuffle_broadcast(src, dst):
    """Broadcast partition 0 of src to all partitions of dst using nc_stream_shuffle."""
    dst_npar = dst.shape[0]
    shuffle_mask = [0] * 32
    for i in range((dst_npar + 31) // 32):
        cur_npar = min(32, dst_npar - i * 32)
        nisa.nc_stream_shuffle(
            src=src[0:1, :],
            dst=dst[i * 32 : i * 32 + cur_npar, 0 : dst.shape[1]],
            shuffle_mask=shuffle_mask,
        )


# === Hardware constants ===
P_MAX = 128       # Partition dimension max
TILE_K = 128      # Contraction dimension tile size
TILE_N = 512      # PSUM free dimension max for gen3


@nki.jit(platform_target="trn2")
def matmul_subtract_multiply_relu_kernel(
    x_t,          # (in_features, batch_size) @ HBM - pre-transposed input
    weight_t,     # (in_features, out_features) @ HBM - pre-transposed weight
    bias_2d,      # (1, out_features) @ HBM - bias reshaped to 2D
    subtract_value,   # float scalar
    multiply_value,   # float scalar
) -> nl.ndarray:
    """Fused linear + subtract + multiply + ReLU kernel.

    Computes: relu((x_t.T @ weight_t + bias - subtract_value) * multiply_value)

    Args:
        x_t: [in_features, batch_size] @ HBM, transposed input activations
        weight_t: [in_features, out_features] @ HBM, transposed weight matrix
        bias_2d: [1, out_features] @ HBM, bias vector reshaped to 2D
        subtract_value: scalar subtracted from linear output
        multiply_value: scalar multiplied after subtraction

    Returns:
        [batch_size, out_features] @ HBM, output tensor
    """
    # === Input Validation ===
    in_features, batch_size = x_t.shape
    K, out_features = weight_t.shape
    kernel_assert(K == in_features, "Weight contraction dim must match in_features")
    kernel_assert(in_features % TILE_K == 0, "in_features must be divisible by 128")
    kernel_assert(out_features % TILE_N == 0, "out_features must be divisible by 512")
    kernel_assert(batch_size % P_MAX == 0, "batch_size must be divisible by 128")

    # === Calculate Tiling ===
    num_m_tiles = batch_size // P_MAX
    num_k_tiles = in_features // TILE_K
    num_n_tiles = out_features // TILE_N

    # === Allocate Output ===
    output = nl.ndarray((batch_size, out_features), dtype=x_t.dtype, buffer=nl.shared_hbm)

    # === Main Processing Loop ===
    for m_idx in nl.affine_range(num_m_tiles):
        m_start = m_idx * P_MAX

        for n_idx in nl.affine_range(num_n_tiles):
            n_start = n_idx * TILE_N

            # --- MatMul: accumulate x_t.T @ weight_t over K dimension ---
            psum_tile = nl.ndarray((P_MAX, TILE_N), dtype=nl.float32, buffer=nl.psum)

            for k_idx in nl.affine_range(num_k_tiles):
                k_start = k_idx * TILE_K

                # Load x_t tile: x_t[K, M] already in stationary layout [K, M]
                lhs_tile = nl.ndarray((TILE_K, P_MAX), dtype=x_t.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=lhs_tile,
                    src=x_t[k_start:k_start + TILE_K, m_start:m_start + P_MAX]
                )

                # Load weight_t tile: [K, N] directly (moving operand)
                rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=weight_t.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=rhs_tile,
                    src=weight_t[k_start:k_start + TILE_K, n_start:n_start + TILE_N]
                )

                # Accumulate into PSUM: psum += lhs_tile.T @ rhs_tile
                nisa.nc_matmul(dst=psum_tile, stationary=lhs_tile, moving=rhs_tile)

            # --- Copy PSUM -> SBUF (float32 for epilogue precision) ---
            result_sb = nl.ndarray((P_MAX, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=result_sb, src=psum_tile)

            # --- Add bias: broadcast partition 0 to all 128 partitions ---
            bias_p0 = nl.ndarray((1, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=bias_p0, src=bias_2d[0:1, n_start:n_start + TILE_N])

            bias_all = nl.ndarray((P_MAX, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            stream_shuffle_broadcast(src=bias_p0, dst=bias_all)

            nisa.tensor_tensor(dst=result_sb, data1=result_sb, data2=bias_all, op=nl.add)

            # --- Subtract + Multiply: (result - subtract_value) * multiply_value ---
            nisa.tensor_scalar(
                dst=result_sb, data=result_sb,
                op0=nl.add, operand0=-subtract_value,
                op1=nl.multiply, operand1=multiply_value,
            )

            # --- ReLU activation ---
            nisa.activation(dst=result_sb, data=result_sb, op=nl.relu)

            # --- Store to HBM ---
            nisa.dma_copy(
                dst=output[m_start:m_start + P_MAX, n_start:n_start + TILE_N],
                src=result_sb
            )

    return output
