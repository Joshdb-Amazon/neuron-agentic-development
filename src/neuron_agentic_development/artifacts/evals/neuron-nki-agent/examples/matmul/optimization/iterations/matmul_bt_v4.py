"""
NKI kernel for matrix multiplication with B transposed: C = A @ B.T
V4: BLOCK_M=8 + TILE_N=512 — combines V3's data reuse with larger matmul tiles.
    A tiles: 8×64×32KB = 16 MB. B tiles: 64×128KB = 8 MB. Total: 24 MB (SBUF limit).
    Same DMA traffic as V3 (~160 MB) but ~3.5x fewer matmul instructions.
    Uses sub-tile transposes: each B [512,128] transposed as 4× [128,128] sub-tiles.
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit(platform_target="trn2")
def matmul_bt_kernel(A, B):
    M, K = A.shape
    N, K_ = B.shape
    assert K == K_

    TILE_K = 128
    TILE_M = 128
    TILE_N = 512
    SUB_N = 128
    N_SUBS = TILE_N // SUB_N  # 4

    TILES_IN_BLOCK_M = 8
    BLOCK_M = TILE_M * TILES_IN_BLOCK_M  # 1024

    assert M % BLOCK_M == 0
    assert N % TILE_N == 0
    assert K % TILE_K == 0

    num_m_blocks = M // BLOCK_M  # 2
    num_n_tiles = N // TILE_N    # 8
    num_k_tiles = K // TILE_K    # 64

    C = nl.ndarray((M, N), dtype=A.dtype, buffer=nl.shared_hbm)

    for m_blk in nl.affine_range(num_m_blocks):
        # Pre-load and transpose A tiles: 8 M × 64 K × 32KB = 16 MB
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
            # Pre-load B tiles: 64 K × [128, 512] via 4 sub-tile transposes = 8 MB
            B_tiles = []
            for k_idx in nl.affine_range(num_k_tiles):
                B_tile = nl.ndarray((TILE_K, TILE_N), dtype=B.dtype, buffer=nl.sbuf)
                for s in nl.affine_range(N_SUBS):
                    n_off = n_idx * TILE_N + s * SUB_N
                    B_sub = nl.ndarray((SUB_N, TILE_K), dtype=B.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=B_sub,
                        src=B[n_off:n_off + SUB_N,
                             k_idx * TILE_K:(k_idx + 1) * TILE_K]
                    )
                    B_sub_T_psum = nl.ndarray((TILE_K, SUB_N), dtype=B.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=B_sub_T_psum, data=B_sub)
                    nisa.tensor_copy(
                        dst=B_tile[0:TILE_K, s * SUB_N:(s + 1) * SUB_N],
                        src=B_sub_T_psum
                    )
                B_tiles.append(B_tile)

            # Compute: stationary [128, 128] × moving [128, 512] → PSUM [128, 512]
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
