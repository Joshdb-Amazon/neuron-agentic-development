"""
Conv2D + Scale + Min NKI Kernel - Direct Convolution (No CPU im2col)

Implements the fused operation:
    x = conv2d(x)        # [B, C_out, H_out, W_out]
    x = x * scale        # Scale by scale_factor
    x = min(x, dim=1)    # Min along channel dim -> [B, 1, H_out, W_out]

Key implementation notes:
- Direct convolution without im2col transformation on CPU
- Weight and input loaded with contiguous DMAs
- Convolution computed as accumulated matmuls over kernel positions
- Processes output by rows to enable efficient strided input access

Hardware limits (gen3):
- P_MAX: 128 (partition dim)
- PSUM_F_MAX: 512 (free dim for matmul)
"""

import nki
import nki.isa as nisa
import nki.language as nl


def div_ceil(n: int, d: int) -> int:
    """Ceiling division."""
    return (n + d - 1) // d


@nki.jit(platform_target="trn2")
def conv2d_scale_min_direct(
    input_hbm,              # [B, C_in, H_in * W_in] input tensor (spatial flattened)
    weight_hbm,             # [K*K, C_in, C_out] weight for kernel positions
    scale_factor,           # scalar scale factor
    batch_size,             # B
    c_in,                   # input channels
    c_out,                  # output channels
    h_in,                   # input height
    w_in,                   # input width
    kernel_size,            # K
) -> nl.ndarray:
    """
    Conv2D + Scale + Min kernel with direct on-device convolution.

    Expects input shaped as [B, C_in, H_in * W_in] - spatial dimensions flattened.
    Weight reshaped to [K*K, C_in, C_out] for efficient kernel-position access.

    Args:
        input_hbm: Input tensor [B, C_in, H_in * W_in]
        weight_hbm: Weight [K*K, C_in, C_out]
        scale_factor: Scalar to multiply conv output
        batch_size, c_in, c_out, h_in, w_in, kernel_size: Dimension parameters

    Returns:
        Output tensor [B, H_out * W_out] flattened
    """
    h_out = h_in - kernel_size + 1
    w_out = w_in - kernel_size + 1
    n_out = h_out * w_out
    ksq = kernel_size * kernel_size
    hw_in = h_in * w_in

    # Hardware limits
    P_MAX = 128
    PSUM_F_MAX = 512

    # Allocate output in HBM [B, H_out * W_out]
    output_hbm = nl.ndarray(
        (batch_size, n_out),
        dtype=input_hbm.dtype,
        buffer=nl.shared_hbm
    )

    # Process each batch
    for b_idx in nl.affine_range(batch_size):
        # Load entire input for this batch [C_in, H_in * W_in]
        input_batch = nl.ndarray((c_in, hw_in), dtype=input_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=input_batch,
            src=input_hbm[b_idx, 0:c_in, 0:hw_in]
        )

        # Allocate working buffers outside the row loop
        weight_tile = nl.ndarray((c_in, c_out), dtype=weight_hbm.dtype, buffer=nl.sbuf)
        input_slice = nl.ndarray((c_in, w_out), dtype=input_hbm.dtype, buffer=nl.sbuf)

        # PSUM buffer - allocated once, cleared each iteration
        psum_row = nl.ndarray((P_MAX, PSUM_F_MAX), dtype=nl.float32, buffer=nl.psum)

        # Process output row by row with affine_range
        for h_o in nl.affine_range(h_out):
            # Clear PSUM using memset
            nisa.memset(dst=psum_row[0:c_out, 0:w_out], value=0.0)

            # Accumulate over kernel positions
            for kh in nl.static_range(kernel_size):
                for kw in nl.static_range(kernel_size):
                    k_pos = kh * kernel_size + kw

                    # Load weight for this kernel position [C_in, C_out]
                    nisa.dma_copy(
                        dst=weight_tile,
                        src=weight_hbm[k_pos, 0:c_in, 0:c_out]
                    )

                    # Extract input slice for this kernel position
                    # For output row h_o, we need input row (h_o + kh)
                    input_row = h_o + kh
                    input_col_start = input_row * w_in + kw

                    # Copy [c_in, w_out] slice
                    nisa.tensor_copy(
                        dst=input_slice[0:c_in, 0:w_out],
                        src=input_batch[0:c_in, input_col_start:input_col_start+w_out]
                    )

                    # Matmul: weight[C_in, C_out] @ input_slice[C_in, w_out] = [C_out, w_out]
                    nisa.nc_matmul(
                        dst=psum_row[0:c_out, 0:w_out],
                        stationary=weight_tile[0:c_in, 0:c_out],
                        moving=input_slice[0:c_in, 0:w_out]
                    )

            # PSUM -> SBUF
            conv_row = nl.ndarray((P_MAX, PSUM_F_MAX), dtype=input_hbm.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=conv_row[0:c_out, 0:w_out],
                src=psum_row[0:c_out, 0:w_out]
            )

            # Scale
            scaled = nl.ndarray((P_MAX, PSUM_F_MAX), dtype=input_hbm.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=scaled[0:c_out, 0:w_out],
                data=conv_row[0:c_out, 0:w_out],
                op0=nl.multiply,
                operand0=scale_factor
            )

            # Min = -max(-x)
            neg_scaled = nl.ndarray((P_MAX, PSUM_F_MAX), dtype=input_hbm.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=neg_scaled[0:c_out, 0:w_out],
                data=scaled[0:c_out, 0:w_out],
                op0=nl.multiply,
                operand0=-1.0
            )

            max_neg = nl.ndarray((1, PSUM_F_MAX), dtype=input_hbm.dtype, buffer=nl.sbuf)
            nisa.tensor_partition_reduce(
                dst=max_neg[0:1, 0:w_out],
                op=nl.maximum,
                data=neg_scaled[0:c_out, 0:w_out]
            )

            min_row = nl.ndarray((1, PSUM_F_MAX), dtype=input_hbm.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=min_row[0:1, 0:w_out],
                data=max_neg[0:1, 0:w_out],
                op0=nl.multiply,
                operand0=-1.0
            )

            # Store this output row
            out_row_start = h_o * w_out
            nisa.dma_copy(
                dst=output_hbm[b_idx, out_row_start:out_row_start+w_out],
                src=min_row[0, 0:w_out]
            )

    return output_hbm


@nki.jit(platform_target="trn2")
def conv2d_scale_min_direct_v2(
    input_hbm,              # [B, C_in, H_in * W_in] input tensor (spatial flattened)
    weight_hbm,             # [C_in * K * K, C_out] weight (im2col style, pre-transposed)
    scale_factor,           # scalar scale factor
    batch_size,             # B
    c_in,                   # input channels
    c_out,                  # output channels
    h_in,                   # input height
    w_in,                   # input width
    kernel_size,            # K
) -> nl.ndarray:
    """
    Conv2D + Scale + Min kernel - Version 2 with row-based on-device im2col.

    For each output row, builds im2col on-device by gathering input values.
    Weight should be pre-transposed to [C_in * K * K, C_out] on CPU.
    """
    h_out = h_in - kernel_size + 1
    w_out = w_in - kernel_size + 1
    n_out = h_out * w_out
    k_flat = c_in * kernel_size * kernel_size
    hw_in = h_in * w_in

    # Hardware limits
    P_MAX = 128
    K_MAX = 128
    PSUM_F_MAX = 512

    # Output
    output_hbm = nl.ndarray(
        (batch_size, n_out),
        dtype=input_hbm.dtype,
        buffer=nl.shared_hbm
    )

    # Tile parameters
    k_tile = K_MAX
    num_k_tiles = div_ceil(k_flat, k_tile)

    # Process each batch
    for b_idx in nl.affine_range(batch_size):
        # Load entire input for this batch [C_in, H_in * W_in]
        input_batch = nl.ndarray((c_in, hw_in), dtype=input_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=input_batch,
            src=input_hbm[b_idx, 0:c_in, 0:hw_in]
        )

        # Allocate all buffers outside inner loops
        weight_tile = nl.ndarray((K_MAX, P_MAX), dtype=weight_hbm.dtype, buffer=nl.sbuf)
        im2col_patch = nl.ndarray((K_MAX, PSUM_F_MAX), dtype=input_hbm.dtype, buffer=nl.sbuf)
        psum_row = nl.ndarray((P_MAX, PSUM_F_MAX), dtype=nl.float32, buffer=nl.psum)
        conv_row = nl.ndarray((P_MAX, PSUM_F_MAX), dtype=input_hbm.dtype, buffer=nl.sbuf)
        scaled = nl.ndarray((P_MAX, PSUM_F_MAX), dtype=input_hbm.dtype, buffer=nl.sbuf)
        neg_scaled = nl.ndarray((P_MAX, PSUM_F_MAX), dtype=input_hbm.dtype, buffer=nl.sbuf)
        max_neg = nl.ndarray((1, PSUM_F_MAX), dtype=input_hbm.dtype, buffer=nl.sbuf)
        min_row = nl.ndarray((1, PSUM_F_MAX), dtype=input_hbm.dtype, buffer=nl.sbuf)

        # Process output row by row with affine_range + memset
        for h_o in nl.affine_range(h_out):
            # Clear PSUM using memset
            nisa.memset(dst=psum_row[0:c_out, 0:w_out], value=0.0)

            # Accumulate over K tiles
            for k_tile_idx in nl.affine_range(num_k_tiles):
                k_start = k_tile_idx * k_tile
                k_end_raw = k_start + k_tile
                k_end = k_end_raw if k_end_raw <= k_flat else k_flat
                k_size = k_end - k_start

                # Load weight tile [k_size, C_out]
                nisa.dma_copy(
                    dst=weight_tile[0:k_size, 0:c_out],
                    src=weight_hbm[k_start:k_end, 0:c_out]
                )

                # Build im2col patch for this row [k_size, w_out]
                # Each k corresponds to (c, kh, kw)
                for k_local in nl.affine_range(k_size):
                    k_global = k_start + k_local
                    # Decode (c, kh, kw) from k_global
                    ksq = kernel_size * kernel_size
                    c_idx = k_global // ksq
                    k_rem = k_global % ksq
                    kh = k_rem // kernel_size
                    kw = k_rem % kernel_size

                    # Input row for this (kh, h_o)
                    input_row = h_o + kh
                    input_row_start = input_row * w_in + kw

                    # Copy w_out contiguous elements
                    nisa.tensor_copy(
                        dst=im2col_patch[k_local:k_local+1, 0:w_out],
                        src=input_batch[c_idx:c_idx+1, input_row_start:input_row_start+w_out]
                    )

                # Matmul
                nisa.nc_matmul(
                    dst=psum_row[0:c_out, 0:w_out],
                    stationary=weight_tile[0:k_size, 0:c_out],
                    moving=im2col_patch[0:k_size, 0:w_out]
                )

            # PSUM -> SBUF
            nisa.tensor_copy(
                dst=conv_row[0:c_out, 0:w_out],
                src=psum_row[0:c_out, 0:w_out]
            )

            # Scale
            nisa.tensor_scalar(
                dst=scaled[0:c_out, 0:w_out],
                data=conv_row[0:c_out, 0:w_out],
                op0=nl.multiply,
                operand0=scale_factor
            )

            # Min = -max(-x)
            nisa.tensor_scalar(
                dst=neg_scaled[0:c_out, 0:w_out],
                data=scaled[0:c_out, 0:w_out],
                op0=nl.multiply,
                operand0=-1.0
            )

            nisa.tensor_partition_reduce(
                dst=max_neg[0:1, 0:w_out],
                op=nl.maximum,
                data=neg_scaled[0:c_out, 0:w_out]
            )

            nisa.tensor_scalar(
                dst=min_row[0:1, 0:w_out],
                data=max_neg[0:1, 0:w_out],
                op0=nl.multiply,
                operand0=-1.0
            )

            # Store
            out_row_start = h_o * w_out
            nisa.dma_copy(
                dst=output_hbm[b_idx, out_row_start:out_row_start+w_out],
                src=min_row[0, 0:w_out]
            )

    return output_hbm
