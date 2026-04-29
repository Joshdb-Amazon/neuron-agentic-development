"""
Fused RMSNorm + QKV projection NKI kernel.

Computes: output = RMSNorm(hidden) @ weight
where RMSNorm(x) = x / sqrt(mean(x^2, dim=-1) + eps)

Optimization: we compute (input @ weight) * inv_rms instead of (input * inv_rms) @ weight,
which is mathematically equivalent (scalar per row commutes with matmul) and avoids
materializing the full normalized tensor in SBUF.

Target: gen3 (Trainium2), single NeuronCore.
"""

import nki
import nki.isa as nisa
import nki.language as nl


def div_ceil(n, d):
    """Ceiling division: smallest integer >= n/d."""
    return (n + d - 1) // d


def kernel_assert(condition, error_text):
    """Assert with NKI-formatted error message."""
    assert condition, f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"


# Hardware constants
P_MAX = 128       # Max partition dimension
PSUM_FMAX = 512   # PSUM free dim limit (gen2/gen3)


@nki.jit
def rms_fused_qkv_kernel(hidden_hbm, weight_hbm, eps=1e-6):
    """
    Fused RMSNorm + linear projection on a single NeuronCore.

    Dimensions:
        BS: batch * seqlen (flattened)
        D:  hidden dimension (input)
        HD: head dimension (output / projection)

    Args:
        hidden_hbm (nl.ndarray): [BS, D] @ HBM, float32 input tensor
        weight_hbm (nl.ndarray): [D, HD] @ HBM, float32 weight matrix
        eps (float): Epsilon for RMSNorm numerical stability. Default 1e-6

    Returns:
        nl.ndarray: [BS, HD] @ HBM, float32 output tensor

    Notes:
        - D must be divisible by 128 (partition dimension)
        - HD must be divisible by 512 (PSUM free dimension on gen3)
        - Uses float32 throughout for numerical accuracy

    Pseudocode:
        for s_tile in tiles(BS, 128):
            inv_rms = rsqrt(mean(input^2, axis=-1) + eps)
            for n_tile in tiles(HD, 512):
                psum = 0
                for k_tile in tiles(D, 128):
                    stat = transpose(input[:, k_tile])
                    psum += stat.T @ weight[k_tile, n_tile]
                output = psum * inv_rms
    """
    batch_seqlen = hidden_hbm.shape[0]
    dim = hidden_hbm.shape[1]
    head_dim = weight_hbm.shape[1]

    kernel_assert(dim % P_MAX == 0, f"dim ({dim}) must be divisible by {P_MAX}")
    kernel_assert(head_dim % PSUM_FMAX == 0, f"head_dim ({head_dim}) must be divisible by {PSUM_FMAX}")

    output_hbm = nl.ndarray(
        (batch_seqlen, head_dim), dtype=hidden_hbm.dtype, buffer=nl.shared_hbm
    )

    num_seq_tiles = div_ceil(batch_seqlen, P_MAX)   # 256/128 = 2
    num_k_tiles = div_ceil(dim, P_MAX)               # 2048/128 = 16
    num_n_tiles = div_ceil(head_dim, PSUM_FMAX)      # 1024/512 = 2

    for s_idx in nl.affine_range(num_seq_tiles):
        s_start = s_idx * P_MAX
        s_end = min(s_start + P_MAX, batch_seqlen)
        s_size = s_end - s_start

        # ========== Phase 1: Compute inv_rms per row ==========
        # Load input tile [s_size, dim]
        input_sb = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=input_sb[0:s_size, 0:dim],
            src=hidden_hbm[s_start:s_end, 0:dim],
        )

        # x^2
        sq_sb = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=sq_sb[0:s_size, 0:dim],
            data1=input_sb[0:s_size, 0:dim],
            data2=input_sb[0:s_size, 0:dim],
            op=nl.multiply,
        )

        # sum(x^2) along free dim (axis=1)
        sum_sq = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=sum_sq[0:s_size, 0:1],
            op=nl.add,
            data=sq_sb[0:s_size, 0:dim],
            axis=1,
        )

        # inv_rms = rsqrt(sum_sq / dim + eps) via fused activation
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

        # ========== Phase 2: MatMul (input @ weight) then scale ==========
        # nc_matmul: dst = stationary.T @ moving
        #   stationary [K, M]: K=dim_chunk (partition <=128), M=seqlen_chunk (free <=128)
        #   moving [K, N]: K=dim_chunk (partition <=128), N=head_dim_chunk (free <=512)
        #   result [M, N] in PSUM

        for n_idx in nl.affine_range(num_n_tiles):
            n_start = n_idx * PSUM_FMAX
            n_end = min(n_start + PSUM_FMAX, head_dim)
            n_size = n_end - n_start

            psum_result = nl.ndarray(
                (P_MAX, PSUM_FMAX), dtype=nl.float32, buffer=nl.psum
            )

            # Accumulate over K (contraction dim) tiles
            for k_idx in nl.affine_range(num_k_tiles):
                k_start = k_idx * P_MAX
                k_end = min(k_start + P_MAX, dim)
                k_size = k_end - k_start

                # Transpose input chunk: [s_size, k_size] -> [k_size, s_size]
                # nc_transpose limited to 32x32 on Scalar Engine, so tile it
                # This makes K the partition dim for nc_matmul stationary operand
                stat_sb = nl.ndarray(
                    (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
                )
                TRANSPOSE_TILE = 32
                num_tp = P_MAX // TRANSPOSE_TILE  # 128/32 = 4
                for tp_i in nl.affine_range(num_tp):
                    for tp_j in nl.affine_range(num_tp):
                        p_off = tp_i * TRANSPOSE_TILE
                        f_off = tp_j * TRANSPOSE_TILE
                        nisa.nc_transpose(
                            dst=stat_sb[f_off:f_off + TRANSPOSE_TILE,
                                        p_off:p_off + TRANSPOSE_TILE],
                            data=input_sb[p_off:p_off + TRANSPOSE_TILE,
                                          k_start + f_off:k_start + f_off + TRANSPOSE_TILE],
                        )

                # Load weight tile [k_size, n_size]
                weight_sb = nl.ndarray(
                    (P_MAX, PSUM_FMAX), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.dma_copy(
                    dst=weight_sb[0:k_size, 0:n_size],
                    src=weight_hbm[k_start:k_end, n_start:n_end],
                )

                # Matmul: stationary.T @ moving -> [s_size, n_size] in PSUM
                nisa.nc_matmul(
                    dst=psum_result[0:s_size, 0:n_size],
                    stationary=stat_sb[0:k_size, 0:s_size],
                    moving=weight_sb[0:k_size, 0:n_size],
                )

            # Copy PSUM -> SBUF for element-wise scaling
            matmul_sb = nl.ndarray(
                (P_MAX, PSUM_FMAX), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(
                dst=matmul_sb[0:s_size, 0:n_size],
                src=psum_result[0:s_size, 0:n_size],
            )

            # Scale by inv_rms: output = (input @ weight) * inv_rms
            # Use .ap() with stride=0 to broadcast inv_rms [P,1] to [P, n_size]
            inv_rms_broadcast = inv_rms.ap(
                pattern=[[1, s_size], [0, n_size]]
            )
            nisa.tensor_tensor(
                dst=matmul_sb[0:s_size, 0:n_size],
                data1=matmul_sb[0:s_size, 0:n_size],
                data2=inv_rms_broadcast,
                op=nl.multiply,
            )

            # Store result to HBM
            nisa.dma_copy(
                dst=output_hbm[s_start:s_end, n_start:n_end],
                src=matmul_sb[0:s_size, 0:n_size],
            )

    return output_hbm
