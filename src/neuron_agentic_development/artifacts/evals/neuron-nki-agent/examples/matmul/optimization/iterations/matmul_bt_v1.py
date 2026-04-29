"""
NKI kernel for matrix multiplication with B transposed: C = A @ B.T
V1: Load hoisting — pre-load and transpose all A tiles outside the N loop,
    and pre-load/transpose all B tiles outside the K matmul loop.
    This eliminates redundant HBM reads (baseline reloads 13x).
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit(platform_target="trn2")
def matmul_bt_kernel(A, B):
    """
    NKI kernel to compute C = A @ B.T
    V1: Load hoisting optimization
    """
    M, K = A.shape
    N, K_ = B.shape
    assert K == K_

    TILE_K = 128  # partition dimension
    TILE_M = 128  # stationary free dimension
    TILE_N = 128  # moving free dimension

    assert M % TILE_M == 0
    assert N % TILE_N == 0
    assert K % TILE_K == 0

    num_m_tiles = M // TILE_M
    num_n_tiles = N // TILE_N
    num_k_tiles = K // TILE_K

    C = nl.ndarray((M, N), dtype=A.dtype, buffer=nl.shared_hbm)

    for m_idx in nl.affine_range(num_m_tiles):
        # === Pre-load and transpose ALL A tiles for this m_idx ===
        # A[m_idx] depends only on m_idx, not n_idx.
        # Hoisting saves num_n_tiles redundant loads per tile.
        A_tiles = []
        for k_idx in nl.affine_range(num_k_tiles):
            # Load A tile [TILE_M, TILE_K] from HBM
            A_loaded = nl.ndarray((TILE_M, TILE_K), dtype=A.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=A_loaded,
                src=A[m_idx * TILE_M:(m_idx + 1) * TILE_M,
                     k_idx * TILE_K:(k_idx + 1) * TILE_K]
            )
            # Transpose: [TILE_M, TILE_K] -> [TILE_K, TILE_M]
            A_T_psum = nl.ndarray((TILE_K, TILE_M), dtype=A.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=A_T_psum, data=A_loaded)
            A_tile = nl.ndarray((TILE_K, TILE_M), dtype=A.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=A_tile, src=A_T_psum)
            A_tiles.append(A_tile)

        for n_idx in nl.affine_range(num_n_tiles):
            # === Pre-load and transpose ALL B tiles for this n_idx ===
            B_tiles = []
            for k_idx in nl.affine_range(num_k_tiles):
                B_loaded = nl.ndarray((TILE_N, TILE_K), dtype=B.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=B_loaded,
                    src=B[n_idx * TILE_N:(n_idx + 1) * TILE_N,
                         k_idx * TILE_K:(k_idx + 1) * TILE_K]
                )
                B_T_psum = nl.ndarray((TILE_K, TILE_N), dtype=B.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=B_T_psum, data=B_loaded)
                B_tile = nl.ndarray((TILE_K, TILE_N), dtype=B.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=B_tile, src=B_T_psum)
                B_tiles.append(B_tile)

            # === Compute matmul with pre-loaded tiles ===
            result_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            for k_idx in nl.affine_range(num_k_tiles):
                nisa.nc_matmul(dst=result_psum, stationary=A_tiles[k_idx], moving=B_tiles[k_idx])

            # Store result
            result_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=A.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=result_sbuf, src=result_psum, dtype=A.dtype)
            nisa.dma_copy(
                dst=C[m_idx * TILE_M:(m_idx + 1) * TILE_M,
                      n_idx * TILE_N:(n_idx + 1) * TILE_N],
                src=result_sbuf
            )

    return C
