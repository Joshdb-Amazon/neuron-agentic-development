"""
Fused RMSNorm + QKV projection NKI kernel — v1 optimization.

v1 changes vs baseline:
- Replace 16× nc_transpose sub-tile calls with nisa.dma_transpose from HBM.
  Reshape hidden_hbm to 4D [BS, 1, 1, D] and use dma_transpose perm [3,1,2,0]
  to load transposed directly from HBM → SBUF.
  This shifts ALL transpose work from Vector Engine to DMA Engine.
- Trades: extra HBM reads (input loaded twice: once for RMSNorm, once per n_tile
  for matmul transpose) but DMA engine has headroom (was 15% utilized).
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

    # 4D view for DMA transpose: [BS, 1, 1, D]
    hidden_4d = hidden_hbm.reshape((batch_seqlen, 1, 1, dim))

    num_seq_tiles = div_ceil(batch_seqlen, P_MAX)
    num_k_tiles = div_ceil(dim, P_MAX)
    num_n_tiles = div_ceil(head_dim, PSUM_FMAX)

    for s_idx in nl.affine_range(num_seq_tiles):
        s_start = s_idx * P_MAX
        s_end = min(s_start + P_MAX, batch_seqlen)
        s_size = s_end - s_start

        # ========== Phase 1: Compute inv_rms ==========
        input_sb = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=input_sb[0:s_size, 0:dim],
            src=hidden_hbm[s_start:s_end, 0:dim],
        )

        sq_sb = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=sq_sb[0:s_size, 0:dim],
            data1=input_sb[0:s_size, 0:dim],
            data2=input_sb[0:s_size, 0:dim],
            op=nl.multiply,
        )

        sum_sq = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=sum_sq[0:s_size, 0:1],
            op=nl.add,
            data=sq_sb[0:s_size, 0:dim],
            axis=1,
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

        # ========== Phase 2: MatMul with DMA transpose from HBM ==========
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

                # DMA transpose from HBM 4D → SBUF 4D
                # src: [s_size, 1, 1, k_size], perm [3,1,2,0] → [k_size, 1, 1, s_size]
                stat_4d = nl.ndarray(
                    (P_MAX, 1, 1, P_MAX), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.dma_transpose(
                    dst=stat_4d[0:k_size, 0:1, 0:1, 0:s_size],
                    src=hidden_4d[s_start:s_end, 0:1, 0:1, k_start:k_end],
                )
                stat_sb = stat_4d.reshape((P_MAX, P_MAX))

                # Load weight tile
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

            # Copy PSUM → SBUF, scale by inv_rms, store
            matmul_sb = nl.ndarray(
                (P_MAX, PSUM_FMAX), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(
                dst=matmul_sb[0:s_size, 0:n_size],
                src=psum_result[0:s_size, 0:n_size],
            )

            inv_rms_broadcast = inv_rms.ap(
                pattern=[[1, s_size], [0, n_size]]
            )
            nisa.tensor_tensor(
                dst=matmul_sb[0:s_size, 0:n_size],
                data1=matmul_sb[0:s_size, 0:n_size],
                data2=inv_rms_broadcast,
                op=nl.multiply,
            )

            nisa.dma_copy(
                dst=output_hbm[s_start:s_end, n_start:n_end],
                src=matmul_sb[0:s_size, 0:n_size],
            )

    return output_hbm
