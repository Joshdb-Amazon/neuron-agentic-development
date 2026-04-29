"""
NKI kernel for matrix multiplication with B transposed: C = A @ B.T
V2: Free dimension blocking — block M dimension with TILES_IN_BLOCK_M=4.
    Pre-load 4 M-tile sets of A, then iterate over N reusing B tiles across
    the M tiles in the block. Reduces B reloads from 16x to 4x.
    Expected HBM reads: ~288 MB vs V1's 1056 MB.
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit(platform_target="trn2")
def matmul_bt_kernel(A, B):
    """
    NKI kernel to compute C = A @ B.T
    V2: M-dimension blocking
    """
    M, K = A.shape
    N, K_ = B.shape
    assert K == K_

    TILE_K = 128
    TILE_M = 128
    TILE_N = 128
    TILES_IN_BLOCK_M = 4
    BLOCK_M = TILE_M * TILES_IN_BLOCK_M  # 512

    assert M % BLOCK_M == 0
    assert N % TILE_N == 0
    assert K % TILE_K == 0

    num_m_blocks = M // BLOCK_M  # 2048/512 = 4
    num_n_tiles = N // TILE_N    # 4096/128 = 32
    num_k_tiles = K // TILE_K    # 8192/128 = 64

    C = nl.ndarray((M, N), dtype=A.dtype, buffer=nl.shared_hbm)

    for m_blk in nl.affine_range(num_m_blocks):
        # === Pre-load and transpose ALL A tiles for this M block ===
        # 4 M tiles × 64 K tiles = 256 tiles × 32KB = 8 MB SBUF
        A_tiles = []
        for bm in nl.affine_range(TILES_IN_BLOCK_M):
            A_row = []
            m_idx = m_blk * TILES_IN_BLOCK_M + bm
            for k_idx in nl.affine_range(num_k_tiles):
                A_loaded = nl.ndarray((TILE_M, TILE_K), dtype=A.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=A_loaded,
                    src=A[m_idx * TILE_M:(m_idx + 1) * TILE_M,
                         k_idx * TILE_K:(k_idx + 1) * TILE_K]
                )
                A_T_psum = nl.ndarray((TILE_K, TILE_M), dtype=A.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=A_T_psum, data=A_loaded)
                A_tile = nl.ndarray((TILE_K, TILE_M), dtype=A.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=A_tile, src=A_T_psum)
                A_row.append(A_tile)
            A_tiles.append(A_row)

        for n_idx in nl.affine_range(num_n_tiles):
            # === Pre-load and transpose B tiles for this N tile ===
            # 64 K tiles × 32KB = 2 MB SBUF
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

            # === Compute matmul for all M tiles in block × this N tile ===
            # B tiles are reused across all 4 M tiles
            for bm in nl.affine_range(TILES_IN_BLOCK_M):
                m_idx = m_blk * TILES_IN_BLOCK_M + bm
                result_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                for k_idx in nl.affine_range(num_k_tiles):
                    nisa.nc_matmul(dst=result_psum, stationary=A_tiles[bm][k_idx],
                                  moving=B_tiles[k_idx])

                result_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=A.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=result_sbuf, src=result_psum, dtype=A.dtype)
                nisa.dma_copy(
                    dst=C[m_idx * TILE_M:(m_idx + 1) * TILE_M,
                          n_idx * TILE_N:(n_idx + 1) * TILE_N],
                    src=result_sbuf
                )

    return C
