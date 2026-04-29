"""
NKI kernel for matrix multiplication with B transposed: C = A @ B.T
V5: 2-pass K approach with BLOCK_M=16 (all M tiles at once).
    K dimension split into 2 halves of 32 tiles each.
    Per pass: load all 16 M × 32 K A tiles (16 MB), then stream B tiles.
    Pass 1: store fp16 partial to C. Pass 2: load prev partial, add, store.
    This eliminates B reload redundancy: each unique B tile loaded exactly once
    per K pass = 2x total (vs V3's 2x from BLOCK_M=8).
    Trade-off: extra C read-modify-write (16 MB) but simpler scheduling.
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
    TILE_N = 128

    # All M tiles at once, K split into 2 passes
    num_m_tiles = M // TILE_M     # 16
    num_n_tiles = N // TILE_N     # 32
    num_k_tiles = K // TILE_K     # 64
    K_PASSES = 2
    K_TILES_PER_PASS = num_k_tiles // K_PASSES  # 32

    assert M % TILE_M == 0
    assert N % TILE_N == 0
    assert K % TILE_K == 0
    assert num_k_tiles % K_PASSES == 0

    C = nl.ndarray((M, N), dtype=A.dtype, buffer=nl.shared_hbm)

    for k_pass in nl.sequential_range(K_PASSES):
        # Pre-load and transpose ALL M tiles × 32 K tiles = 16 × 32 × 32KB = 16 MB
        A_tiles = []
        for m_idx in nl.affine_range(num_m_tiles):
            A_row = []
            for bk in nl.affine_range(K_TILES_PER_PASS):
                k_idx = k_pass * K_TILES_PER_PASS + bk
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
            # Load and transpose B tiles for this N tile and K pass
            # 32 K tiles × 32KB = 1 MB
            B_tiles = []
            for bk in nl.affine_range(K_TILES_PER_PASS):
                k_idx = k_pass * K_TILES_PER_PASS + bk
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

            # Compute matmul for ALL M tiles with this N tile
            for m_idx in nl.affine_range(num_m_tiles):
                # Matmul over 32 K tiles
                result_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                for bk in nl.affine_range(K_TILES_PER_PASS):
                    nisa.nc_matmul(dst=result_psum, stationary=A_tiles[m_idx][bk],
                                  moving=B_tiles[bk])

                if k_pass > 0:
                    # Load previous partial from C and add to PSUM result
                    prev_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=A.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=prev_sbuf,
                        src=C[m_idx * TILE_M:(m_idx + 1) * TILE_M,
                              n_idx * TILE_N:(n_idx + 1) * TILE_N]
                    )
                    # Add prev (float16 in SBUF) to result (float32 in PSUM)
                    nisa.tensor_tensor(dst=result_psum,
                                      data1=prev_sbuf,
                                      data2=result_psum,
                                      op=nl.add)

                # Store result to C
                result_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=A.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=result_sbuf, src=result_psum, dtype=A.dtype)
                nisa.dma_copy(
                    dst=C[m_idx * TILE_M:(m_idx + 1) * TILE_M,
                          n_idx * TILE_N:(n_idx + 1) * TILE_N],
                    src=result_sbuf
                )

    return C
