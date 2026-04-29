"""
Fused RMSNorm + QKV projection NKI kernel — v5 optimization.

v5 changes vs v2:
- Manually unroll the n_tile loop: process BOTH n_tiles in a single k_tile
  pass. This means each dma_transpose is done ONCE per k_tile (shared by both
  n_tile matmuls), halving the dma_transpose reads from HBM.
- Since num_n_tiles=2 is small, manual unrolling is practical.
- Expected: reduce dma_transpose HBM reads from 2 MB to 1 MB (shared across n_tiles).
  Total HBM read: 12 MB → 10 MB.
"""

import nki
import nki.isa as nisa
import nki.language as nl


def div_ceil(n, d):
    return (n + d - 1) // d


P_MAX = 128
PSUM_FMAX = 512


@nki.jit
def rms_fused_qkv_kernel(hidden_hbm, weight_hbm, eps=1e-6):
    batch_seqlen = hidden_hbm.shape[0]
    dim = hidden_hbm.shape[1]
    head_dim = weight_hbm.shape[1]

    assert dim % P_MAX == 0
    assert head_dim % PSUM_FMAX == 0

    output_hbm = nl.ndarray(
        (batch_seqlen, head_dim), dtype=hidden_hbm.dtype, buffer=nl.shared_hbm
    )

    hidden_4d = hidden_hbm.reshape((batch_seqlen, 1, 1, dim))

    num_seq_tiles = div_ceil(batch_seqlen, P_MAX)
    num_k_tiles = div_ceil(dim, P_MAX)
    num_n_tiles = div_ceil(head_dim, PSUM_FMAX)

    for s_idx in nl.affine_range(num_seq_tiles):
        s_start = s_idx * P_MAX
        s_end = min(s_start + P_MAX, batch_seqlen)
        s_size = s_end - s_start

        # ========== Phase 1: Compute inv_rms incrementally ==========
        sum_sq = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=sum_sq, value=0.0)

        for k_idx in nl.affine_range(num_k_tiles):
            k_start = k_idx * P_MAX
            k_end = min(k_start + P_MAX, dim)

            chunk_sb = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=chunk_sb[0:s_size, 0:P_MAX],
                src=hidden_hbm[s_start:s_end, k_start:k_end],
            )

            nisa.tensor_tensor(
                dst=chunk_sb[0:s_size, 0:P_MAX],
                data1=chunk_sb[0:s_size, 0:P_MAX],
                data2=chunk_sb[0:s_size, 0:P_MAX],
                op=nl.multiply,
            )

            partial_sum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(
                dst=partial_sum[0:s_size, 0:1],
                op=nl.add,
                data=chunk_sb[0:s_size, 0:P_MAX],
                axis=1,
            )

            nisa.tensor_tensor(
                dst=sum_sq[0:s_size, 0:1],
                data1=sum_sq[0:s_size, 0:1],
                data2=partial_sum[0:s_size, 0:1],
                op=nl.add,
            )

        eps_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=eps_sb, value=eps)

        inv_rms = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=inv_rms[0:s_size, 0:1],
            op=nl.rsqrt,
            data=sum_sq[0:s_size, 0:1],
            scale=1.0 / dim,
            bias=eps_sb[0:s_size, 0:1],
        )

        # ========== Phase 2: MatMul — manually unrolled n_tiles ==========
        # Both n_tile PSUM accumulators
        psum_n0 = nl.ndarray(
            (P_MAX, PSUM_FMAX), dtype=nl.float32, buffer=nl.psum
        )
        psum_n1 = nl.ndarray(
            (P_MAX, PSUM_FMAX), dtype=nl.float32, buffer=nl.psum
        )

        n0_start = 0
        n0_end = min(PSUM_FMAX, head_dim)
        n0_size = n0_end - n0_start
        n1_start = PSUM_FMAX
        n1_end = min(2 * PSUM_FMAX, head_dim)
        n1_size = n1_end - n1_start

        # Single k_tile loop: dma_transpose once, matmul for both n_tiles
        for k_idx in nl.affine_range(num_k_tiles):
            k_start = k_idx * P_MAX
            k_end = min(k_start + P_MAX, dim)
            k_size = k_end - k_start

            # DMA transpose ONCE per k_tile (shared by both n_tiles)
            stat_4d = nl.ndarray(
                (P_MAX, 1, 1, P_MAX), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.dma_transpose(
                dst=stat_4d[0:k_size, 0:1, 0:1, 0:s_size],
                src=hidden_4d[s_start:s_end, 0:1, 0:1, k_start:k_end],
            )
            stat_sb = stat_4d.reshape((P_MAX, P_MAX))

            # Weight for n_tile 0
            weight_n0 = nl.ndarray(
                (P_MAX, PSUM_FMAX), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.dma_copy(
                dst=weight_n0[0:k_size, 0:n0_size],
                src=weight_hbm[k_start:k_end, n0_start:n0_end],
            )
            nisa.nc_matmul(
                dst=psum_n0[0:s_size, 0:n0_size],
                stationary=stat_sb[0:k_size, 0:s_size],
                moving=weight_n0[0:k_size, 0:n0_size],
            )

            # Weight for n_tile 1
            weight_n1 = nl.ndarray(
                (P_MAX, PSUM_FMAX), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.dma_copy(
                dst=weight_n1[0:k_size, 0:n1_size],
                src=weight_hbm[k_start:k_end, n1_start:n1_end],
            )
            nisa.nc_matmul(
                dst=psum_n1[0:s_size, 0:n1_size],
                stationary=stat_sb[0:k_size, 0:s_size],
                moving=weight_n1[0:k_size, 0:n1_size],
            )

        # ========== Phase 3: Scale and store both n_tiles ==========
        inv_rms_bc0 = inv_rms.ap(pattern=[[1, s_size], [0, n0_size]])
        matmul_n0 = nl.ndarray(
            (P_MAX, PSUM_FMAX), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=matmul_n0[0:s_size, 0:n0_size],
            data1=psum_n0[0:s_size, 0:n0_size],
            data2=inv_rms_bc0,
            op=nl.multiply,
        )
        nisa.dma_copy(
            dst=output_hbm[s_start:s_end, n0_start:n0_end],
            src=matmul_n0[0:s_size, 0:n0_size],
        )

        inv_rms_bc1 = inv_rms.ap(pattern=[[1, s_size], [0, n1_size]])
        matmul_n1 = nl.ndarray(
            (P_MAX, PSUM_FMAX), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=matmul_n1[0:s_size, 0:n1_size],
            data1=psum_n1[0:s_size, 0:n1_size],
            data2=inv_rms_bc1,
            op=nl.multiply,
        )
        nisa.dma_copy(
            dst=output_hbm[s_start:s_end, n1_start:n1_end],
            src=matmul_n1[0:s_size, 0:n1_size],
        )

    return output_hbm
