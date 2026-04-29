"""
Fused matmul + subtract + multiply + ReLU NKI kernel — V1 optimization.

V1: Load Hoisting — pre-load all K tiles of x_t into SBUF before the N loop.
Eliminates redundant x_t reloads (8x reduction: from 16MB to 2MB HBM reads for x_t).
Also pre-loads weight K tiles before matmul for better DMA/compute overlap.

Computes: relu((x @ weight_t + bias - subtract_value) * multiply_value)
"""

import nki
import nki.isa as nisa
import nki.language as nl


def kernel_assert(condition: bool, error_text: str):
    assert condition, f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"


def stream_shuffle_broadcast(src, dst):
    dst_npar = dst.shape[0]
    shuffle_mask = [0] * 32
    for i in range((dst_npar + 31) // 32):
        cur_npar = min(32, dst_npar - i * 32)
        nisa.nc_stream_shuffle(
            src=src[0:1, :],
            dst=dst[i * 32 : i * 32 + cur_npar, 0 : dst.shape[1]],
            shuffle_mask=shuffle_mask,
        )


P_MAX = 128
TILE_K = 128
TILE_N = 512


@nki.jit(platform_target="trn2")
def matmul_subtract_multiply_relu_kernel(
    x_t,
    weight_t,
    bias_2d,
    subtract_value,
    multiply_value,
) -> nl.ndarray:
    """V1: Load hoisting — pre-load x_t K tiles into SBUF, reuse across N tiles."""
    in_features, batch_size = x_t.shape
    K, out_features = weight_t.shape
    kernel_assert(K == in_features, "Weight contraction dim must match in_features")
    kernel_assert(in_features % TILE_K == 0, "in_features must be divisible by 128")
    kernel_assert(out_features % TILE_N == 0, "out_features must be divisible by 512")
    kernel_assert(batch_size % P_MAX == 0, "batch_size must be divisible by 128")

    num_m_tiles = batch_size // P_MAX
    num_k_tiles = in_features // TILE_K
    num_n_tiles = out_features // TILE_N

    output = nl.ndarray((batch_size, out_features), dtype=x_t.dtype, buffer=nl.shared_hbm)

    for m_idx in nl.affine_range(num_m_tiles):
        m_start = m_idx * P_MAX

        # === LOAD HOISTING: Pre-load ALL K tiles of x_t for this m_tile ===
        # These tiles are reused across all N iterations (hoisted out of N loop)
        # SBUF usage: 32 tiles * 128*128*2 = 1MB (fits in 24MB SBUF)
        lhsT_tiles = []
        for k_idx in nl.affine_range(num_k_tiles):
            k_start = k_idx * TILE_K
            lhsT_tile = nl.ndarray((TILE_K, P_MAX), dtype=x_t.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=lhsT_tile,
                src=x_t[k_start:k_start + TILE_K, m_start:m_start + P_MAX]
            )
            lhsT_tiles.append(lhsT_tile)

        for n_idx in nl.affine_range(num_n_tiles):
            n_start = n_idx * TILE_N

            # Pre-load ALL K tiles of weight for this n_tile
            # SBUF usage: 32 tiles * 128*512*2 = 4MB
            rhs_tiles = []
            for k_idx in nl.affine_range(num_k_tiles):
                k_start = k_idx * TILE_K
                rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=weight_t.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=rhs_tile,
                    src=weight_t[k_start:k_start + TILE_K, n_start:n_start + TILE_N]
                )
                rhs_tiles.append(rhs_tile)

            # MatMul using pre-loaded tiles — no DMA stalls
            psum_tile = nl.ndarray((P_MAX, TILE_N), dtype=nl.float32, buffer=nl.psum)
            for k_idx in nl.affine_range(num_k_tiles):
                nisa.nc_matmul(dst=psum_tile, stationary=lhsT_tiles[k_idx], moving=rhs_tiles[k_idx])

            # --- Epilogue: bias + subtract + multiply + relu ---
            result_sb = nl.ndarray((P_MAX, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=result_sb, src=psum_tile)

            bias_p0 = nl.ndarray((1, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=bias_p0, src=bias_2d[0:1, n_start:n_start + TILE_N])

            bias_all = nl.ndarray((P_MAX, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            stream_shuffle_broadcast(src=bias_p0, dst=bias_all)

            nisa.tensor_tensor(dst=result_sb, data1=result_sb, data2=bias_all, op=nl.add)

            nisa.tensor_scalar(
                dst=result_sb, data=result_sb,
                op0=nl.add, operand0=-subtract_value,
                op1=nl.multiply, operand1=multiply_value,
            )

            nisa.activation(dst=result_sb, data=result_sb, op=nl.relu)

            nisa.dma_copy(
                dst=output[m_start:m_start + P_MAX, n_start:n_start + TILE_N],
                src=result_sb
            )

    return output
