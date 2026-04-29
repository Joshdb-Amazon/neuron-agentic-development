"""
NKI kernel for matrix multiplication with B transposed: C = A @ B.T

This kernel computes C[M, N] = A[M, K] @ B[N, K].T = A[M, K] @ B.T[K, N]

For nisa.nc_matmul:
- stationary: (K_tile, M_tile) - transposed A tile
- moving: (K_tile, N_tile) - transposed B tile
- output: (M_tile, N_tile) in PSUM

Approach:
1. Load A tile [M_tile, K_tile] from HBM to SBUF
2. nc_transpose to get [K_tile, M_tile] (swaps partition and free dims)
3. Load B tile [N_tile, K_tile] from HBM to SBUF
4. nc_transpose to get [K_tile, N_tile]
5. nc_matmul with transposed tiles

Hardware constraints (gen3):
- TILE_K = 128 (partition dimension max)
- TILE_M = 128 (stationary free dimension max)
- TILE_N = 128 (limited by nc_transpose partition max, could use 512 with strided DMA)
"""

import nki
import nki.isa as nisa
import nki.language as nl


def div_ceil(n: int, d: int) -> int:
    """Ceiling division: ceil(n/d)"""
    return (n + d - 1) // d


@nki.jit(platform_target="trn2")
def matmul_bt_kernel(A, B):
    """
    NKI kernel to compute C = A @ B.T

    Args:
        A: Input tensor of shape (M, K), dtype float16/bfloat16
        B: Input tensor of shape (N, K), dtype float16/bfloat16

    Returns:
        C: Output tensor of shape (M, N)
    """
    M, K = A.shape
    N, K_ = B.shape

    assert K == K_, f"Contraction dimension mismatch: A has K={K}, B has K={K_}"

    # Hardware tile sizes
    TILE_K = nl.tile_size.pmax  # 128
    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    # Using 128 for N since nc_transpose requires P <= 128
    # (the input partition dimension before transpose)
    TILE_N = 128

    # Verify dimensions are multiples of tile sizes
    assert M % TILE_M == 0, f"M={M} must be multiple of {TILE_M}"
    assert N % TILE_N == 0, f"N={N} must be multiple of {TILE_N}"
    assert K % TILE_K == 0, f"K={K} must be multiple of {TILE_K}"

    # Number of tiles in each dimension
    num_m_tiles = M // TILE_M
    num_n_tiles = N // TILE_N
    num_k_tiles = K // TILE_K

    # Allocate output in HBM
    C = nl.ndarray((M, N), dtype=A.dtype, buffer=nl.shared_hbm)

    # Loop over M and N tiles
    for m_idx in nl.affine_range(num_m_tiles):
        for n_idx in nl.affine_range(num_n_tiles):
            # Allocate PSUM for matmul accumulation
            result_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

            # Accumulate over K tiles
            for k_idx in nl.affine_range(num_k_tiles):
                # --- Transpose A tile ---
                # Load A tile [M_tile, K_tile] from HBM to SBUF
                A_loaded = nl.ndarray((TILE_M, TILE_K), dtype=A.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=A_loaded,
                    src=A[m_idx * TILE_M:(m_idx + 1) * TILE_M,
                         k_idx * TILE_K:(k_idx + 1) * TILE_K]
                )

                # nc_transpose: [M_tile, K_tile] -> [K_tile, M_tile] in PSUM
                # For gen3+, transpose mode requires same input/output dtype
                A_transposed_psum = nl.ndarray((TILE_K, TILE_M), dtype=A.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=A_transposed_psum, data=A_loaded)

                # Copy transposed A from PSUM to SBUF
                A_tile = nl.ndarray((TILE_K, TILE_M), dtype=A.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=A_tile, src=A_transposed_psum)

                # --- Transpose B tile ---
                # Load B tile [N_tile, K_tile] from HBM to SBUF
                B_loaded = nl.ndarray((TILE_N, TILE_K), dtype=B.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=B_loaded,
                    src=B[n_idx * TILE_N:(n_idx + 1) * TILE_N,
                         k_idx * TILE_K:(k_idx + 1) * TILE_K]
                )

                # nc_transpose: [N_tile, K_tile] -> [K_tile, N_tile] in PSUM
                # For gen3+, transpose mode requires same input/output dtype
                B_transposed_psum = nl.ndarray((TILE_K, TILE_N), dtype=B.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=B_transposed_psum, data=B_loaded)

                # Copy transposed B from PSUM to SBUF
                B_tile = nl.ndarray((TILE_K, TILE_N), dtype=B.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=B_tile, src=B_transposed_psum)

                # --- Matrix multiplication ---
                # stationary [K, M] @ moving [K, N] -> output [M, N]
                # Multiple writes to same PSUM buffer triggers hardware accumulation
                nisa.nc_matmul(dst=result_psum, stationary=A_tile, moving=B_tile)

            # Copy result from PSUM to SBUF with dtype cast
            result_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=A.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=result_sbuf, src=result_psum, dtype=A.dtype)

            # Store result tile to HBM
            nisa.dma_copy(
                dst=C[m_idx * TILE_M:(m_idx + 1) * TILE_M,
                      n_idx * TILE_N:(n_idx + 1) * TILE_N],
                src=result_sbuf
            )

    return C
