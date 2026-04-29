"""
Fused RMSNorm + QKV projection NKI kernel — v3 optimization.

v3 changes vs v2:
- Try to fuse tensor_copy + tensor_tensor by using tensor_tensor directly
  with PSUM as data1 operand. If PSUM can be read by tensor_tensor, this
  saves one tensor_copy instruction per n_tile (4 vector instructions each).
- If that doesn't work, fallback: use nisa.tensor_scalar to scale the PSUM
  copy with a compile-time known pattern.
- Also optimize: use nisa.tensor_copy with dtype conversion to combine
  operations where possible.
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

            chunk_sq = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=chunk_sq[0:s_size, 0:P_MAX],
                data1=chunk_sb[0:s_size, 0:P_MAX],
                data2=chunk_sb[0:s_size, 0:P_MAX],
                op=nl.multiply,
            )

            partial_sum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(
                dst=partial_sum[0:s_size, 0:1],
                op=nl.add,
                data=chunk_sq[0:s_size, 0:P_MAX],
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

        # ========== Phase 2: MatMul with DMA transpose ==========
        for n_idx in nl.affine_range(num_n_tiles):
            n_start = n_idx * PSUM_FMAX
            n_end = min(n_start + PSUM_FMAX, head_dim)
            n_size = n_end - n_start

            psum_result = nl.ndarray(
                (P_MAX, PSUM_FMAX), dtype=nl.float32, buffer=nl.psum
            )

            for k_idx in nl.affine_range(num_k_tiles):
                k_start = k_idx * P_MAX
                k_end = min(k_start + P_MAX, dim)
                k_size = k_end - k_start

                stat_4d = nl.ndarray(
                    (P_MAX, 1, 1, P_MAX), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.dma_transpose(
                    dst=stat_4d[0:k_size, 0:1, 0:1, 0:s_size],
                    src=hidden_4d[s_start:s_end, 0:1, 0:1, k_start:k_end],
                )
                stat_sb = stat_4d.reshape((P_MAX, P_MAX))

                weight_sb = nl.ndarray(
                    (P_MAX, PSUM_FMAX), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.dma_copy(
                    dst=weight_sb[0:k_size, 0:n_size],
                    src=weight_hbm[k_start:k_end, n_start:n_end],
                )

                nisa.nc_matmul(
                    dst=psum_result[0:s_size, 0:n_size],
                    stationary=stat_sb[0:k_size, 0:s_size],
                    moving=weight_sb[0:k_size, 0:n_size],
                )

            # Try fused: tensor_tensor directly on PSUM × inv_rms → SBUF
            matmul_sb = nl.ndarray(
                (P_MAX, PSUM_FMAX), dtype=nl.float32, buffer=nl.sbuf
            )
            inv_rms_broadcast = inv_rms.ap(
                pattern=[[1, s_size], [0, n_size]]
            )
            nisa.tensor_tensor(
                dst=matmul_sb[0:s_size, 0:n_size],
                data1=psum_result[0:s_size, 0:n_size],
                data2=inv_rms_broadcast,
                op=nl.multiply,
            )

            nisa.dma_copy(
                dst=output_hbm[s_start:s_end, n_start:n_end],
                src=matmul_sb[0:s_size, 0:n_size],
            )

    return output_hbm
