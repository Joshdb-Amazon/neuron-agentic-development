"""
NKI kernel for conv2d + scaling + min operation.

Operations: Conv2d -> Scale -> Min(dim=1)

This kernel takes original input tensor (no host-side im2col) and handles
the convolution patch extraction internally with proper tiling.

Architecture:
- Weight is pre-flattened to [C_out, C_in * k * k] on host (cheap)
- Input remains [B, C_in, H, W] - patches built in kernel via row loads
- Tiles over input channels (C_in) to handle weight_k > P_MAX
- Uses matmul with PSUM accumulation
- tensor_partition_reduce with -max(-x) trick for min reduction

VECTORIZED: Loads w_out elements per output row instead of single elements.
VECTORIZED: Bias add uses nl.add with broadcasting.

Tested on gen3 (trn2) hardware.
"""

import nki
import nki.isa as nisa
import nki.language as nl


# ============================================================================
# Utility Functions
# ============================================================================

def div_ceil(n: int, d: int) -> int:
    """Ceiling division: smallest integer >= n/d."""
    return (n + d - 1) // d


# Hardware constraints
P_MAX = 128        # Max partition dimension
PSUM_F_MAX = 512   # Max PSUM free dimension (gen3)
TRANSPOSE_MAX = 32 # Max size for scalar engine transpose


# ============================================================================
# NKI Kernel
# ============================================================================

@nki.jit(platform_target="trn2")
def conv2d_scale_min_kernel(
    x_input,       # [B, C_in, H, W] - original input tensor
    weight_flat,   # [C_out, C_in * k * k] - flattened weight tensor
    bias,          # [C_out] - bias vector
    scale_factor: float,
    kernel_size: int,
):
    """
    Fused conv2d + scale + min kernel with internal patch extraction.

    Takes original input tensor (no im2col preprocessing needed).
    Weight should be pre-flattened to [C_out, C_in * k * k] on host.
    Tiles over input channels to handle large C_in * k * k.

    Args:
        x_input: [B, C_in, H, W] input tensor
        weight_flat: [C_out, C_in * k * k] flattened convolution weights
        bias: [C_out] bias vector
        scale_factor: scalar multiplier for conv output
        kernel_size: convolution kernel size (k)

    Returns:
        output: [B, spatial_out] min-reduced result (reshape to [B, 1, H_out, W_out] on host)
    """
    # Extract dimensions
    batch_size = x_input.shape[0]
    in_channels = x_input.shape[1]
    height = x_input.shape[2]
    width = x_input.shape[3]
    out_channels = weight_flat.shape[0]
    weight_k = weight_flat.shape[1]  # = in_channels * k * k

    h_out = height - kernel_size + 1
    w_out = width - kernel_size + 1
    spatial_out = h_out * w_out

    k_sq = kernel_size * kernel_size

    # Calculate tiling for weight dimension
    max_k_per_tile = P_MAX
    num_k_tiles = div_ceil(weight_k, max_k_per_tile)

    # Allocate output in HBM [B, spatial_out]
    output = nl.ndarray(
        (batch_size, spatial_out),
        dtype=x_input.dtype,
        buffer=nl.shared_hbm
    )

    # Load bias [C_out] to SBUF as [C_out, 1]
    bias_sb = nl.ndarray(
        (out_channels, 1),
        dtype=bias.dtype,
        buffer=nl.sbuf
    )
    nisa.dma_copy(
        dst=bias_sb,
        src=bias.reshape((out_channels, 1))[0:out_channels, 0:1]
    )

    # Process each batch
    for b_idx in nl.affine_range(batch_size):

        # Allocate accumulated conv result [C_out, spatial_out]
        conv_accum = nl.ndarray(
            (out_channels, spatial_out),
            dtype=nl.float32,
            buffer=nl.sbuf
        )
        nisa.memset(dst=conv_accum, value=0.0)

        # Tile over weight_k dimension (input channels * k^2)
        for k_tile_idx in nl.affine_range(num_k_tiles):
            k_start = k_tile_idx * max_k_per_tile
            k_end_val = k_start + max_k_per_tile
            k_tile_size = max_k_per_tile
            if k_end_val > weight_k:
                k_tile_size = weight_k - k_start

            # Load weight tile [C_out, k_tile_size]
            weight_tile_sb = nl.ndarray(
                (out_channels, P_MAX),
                dtype=weight_flat.dtype,
                buffer=nl.sbuf
            )
            nisa.dma_copy(
                dst=weight_tile_sb[0:out_channels, 0:k_tile_size],
                src=weight_flat[0:out_channels, k_start:k_start+k_tile_size]
            )

            # Transpose weight [C_out, k_tile_size] -> [k_tile_size, C_out]
            weight_T_sb = nl.ndarray(
                (P_MAX, out_channels),
                dtype=weight_flat.dtype,
                buffer=nl.sbuf
            )

            num_p_tiles = div_ceil(k_tile_size, TRANSPOSE_MAX)
            num_f_tiles = div_ceil(out_channels, TRANSPOSE_MAX)

            for p_tile in nl.affine_range(num_p_tiles):
                p_start_t = p_tile * TRANSPOSE_MAX
                p_size_t = TRANSPOSE_MAX
                if p_start_t + p_size_t > k_tile_size:
                    p_size_t = k_tile_size - p_start_t

                for f_tile in nl.affine_range(num_f_tiles):
                    f_start_t = f_tile * TRANSPOSE_MAX
                    f_size_t = TRANSPOSE_MAX
                    if f_start_t + f_size_t > out_channels:
                        f_size_t = out_channels - f_start_t

                    nisa.nc_transpose(
                        dst=weight_T_sb[p_start_t:p_start_t+p_size_t, f_start_t:f_start_t+f_size_t],
                        data=weight_tile_sb[f_start_t:f_start_t+f_size_t, p_start_t:p_start_t+p_size_t]
                    )

            # Tile over spatial output dimension
            num_spatial_tiles = div_ceil(spatial_out, PSUM_F_MAX)

            for s_tile_idx in nl.affine_range(num_spatial_tiles):
                s_start = s_tile_idx * PSUM_F_MAX
                s_size = PSUM_F_MAX
                if s_start + s_size > spatial_out:
                    s_size = spatial_out - s_start

                # Build patches [k_tile_size, s_size]
                patches_sb = nl.ndarray(
                    (P_MAX, PSUM_F_MAX),
                    dtype=x_input.dtype,
                    buffer=nl.sbuf
                )
                nisa.memset(dst=patches_sb, value=0.0)

                # VECTORIZED patch loading: load contiguous row segments
                # For each k_rel (c_in, kh, kw), oh: load x[b, c_in, ih, iw_start:iw_end]

                for k_rel in nl.affine_range(k_tile_size):
                    k_abs = k_start + k_rel
                    c_in = k_abs // k_sq
                    remainder = k_abs - c_in * k_sq
                    kh = remainder // kernel_size
                    kw = remainder - kh * kernel_size

                    # For each output row, check overlap with spatial tile and load
                    for oh in nl.affine_range(h_out):
                        ih = oh + kh

                        # This row covers spatial positions [oh * w_out, (oh+1) * w_out)
                        row_s_start = oh * w_out
                        row_s_end = row_s_start + w_out

                        # Compute overlap with [s_start, s_start + s_size)
                        overlap_start = row_s_start
                        if s_start > overlap_start:
                            overlap_start = s_start
                        overlap_end = row_s_end
                        if s_start + s_size < overlap_end:
                            overlap_end = s_start + s_size

                        overlap_len = overlap_end - overlap_start
                        if overlap_len > 0:
                            # Load overlapping portion
                            iw_start = (overlap_start - row_s_start) + kw
                            dst_col = overlap_start - s_start

                            input_idx = c_in * (height * width) + ih * width + iw_start

                            nisa.dma_copy(
                                dst=patches_sb[k_rel:k_rel+1, dst_col:dst_col+overlap_len],
                                src=x_input.reshape((batch_size, in_channels * height * width))[b_idx:b_idx+1, input_idx:input_idx+overlap_len]
                            )

                # MatMul: weight_T [k_tile_size, C_out] @ patches [k_tile_size, s_size]
                matmul_psum = nl.ndarray(
                    (out_channels, PSUM_F_MAX),
                    dtype=nl.float32,
                    buffer=nl.psum
                )

                nisa.nc_matmul(
                    dst=matmul_psum[0:out_channels, 0:s_size],
                    stationary=weight_T_sb[0:k_tile_size, 0:out_channels],
                    moving=patches_sb[0:k_tile_size, 0:s_size]
                )

                # Copy and accumulate
                tile_result = nl.ndarray(
                    (out_channels, PSUM_F_MAX),
                    dtype=nl.float32,
                    buffer=nl.sbuf
                )
                nisa.tensor_copy(
                    dst=tile_result[0:out_channels, 0:s_size],
                    src=matmul_psum[0:out_channels, 0:s_size]
                )

                nisa.tensor_tensor(
                    dst=conv_accum[0:out_channels, s_start:s_start+s_size],
                    data1=conv_accum[0:out_channels, s_start:s_start+s_size],
                    data2=tile_result[0:out_channels, 0:s_size],
                    op=nl.add
                )

        # Add bias - explicitly expand bias to match spatial dimensions
        # bias_sb is [C_out, 1], conv_accum is [C_out, spatial_out]
        bias_f32 = nl.ndarray((out_channels, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=bias_f32, src=bias_sb)

        biased = nl.ndarray(
            (out_channels, spatial_out),
            dtype=nl.float32,
            buffer=nl.sbuf
        )

        # First copy conv_accum to biased
        nisa.tensor_copy(dst=biased, src=conv_accum)

        # Then add bias per output channel row
        # Each row gets bias[c] added to all spatial positions
        # Process in spatial tiles for efficiency
        num_bias_tiles = div_ceil(spatial_out, PSUM_F_MAX)
        for bt_idx in nl.affine_range(num_bias_tiles):
            bt_start = bt_idx * PSUM_F_MAX
            bt_size = PSUM_F_MAX
            if bt_start + bt_size > spatial_out:
                bt_size = spatial_out - bt_start

            # Create expanded bias tile [out_channels, bt_size]
            bias_tile = nl.ndarray((out_channels, PSUM_F_MAX), dtype=nl.float32, buffer=nl.sbuf)

            # Replicate bias_f32 [out_channels, 1] to [out_channels, bt_size]
            # Use tensor_copy to copy each column
            # But to vectorize, copy in chunks
            for col_chunk in nl.affine_range(div_ceil(bt_size, TRANSPOSE_MAX)):
                chunk_start = col_chunk * TRANSPOSE_MAX
                chunk_end = chunk_start + TRANSPOSE_MAX
                if chunk_end > bt_size:
                    chunk_end = bt_size
                chunk_len = chunk_end - chunk_start

                # For this chunk, copy bias to each column
                for col in nl.affine_range(chunk_len):
                    nisa.tensor_copy(
                        dst=bias_tile[0:out_channels, chunk_start+col:chunk_start+col+1],
                        src=bias_f32[0:out_channels, 0:1]
                    )

            # Add expanded bias to biased
            nisa.tensor_tensor(
                dst=biased[0:out_channels, bt_start:bt_start+bt_size],
                data1=biased[0:out_channels, bt_start:bt_start+bt_size],
                data2=bias_tile[0:out_channels, 0:bt_size],
                op=nl.add
            )

        # Scale
        scaled = nl.ndarray(
            (out_channels, spatial_out),
            dtype=nl.float32,
            buffer=nl.sbuf
        )
        nisa.tensor_scalar(
            dst=scaled,
            data=biased,
            op0=nl.multiply,
            operand0=scale_factor
        )

        # Min reduction: min(x) = -max(-x)
        negated = nl.ndarray(
            (out_channels, spatial_out),
            dtype=nl.float32,
            buffer=nl.sbuf
        )
        nisa.tensor_scalar(
            dst=negated,
            data=scaled,
            op0=nl.multiply,
            operand0=-1.0
        )

        max_negated = nl.ndarray(
            (1, spatial_out),
            dtype=nl.float32,
            buffer=nl.sbuf
        )
        nisa.tensor_partition_reduce(
            dst=max_negated,
            op=nl.maximum,
            data=negated
        )

        min_result = nl.ndarray(
            (1, spatial_out),
            dtype=nl.float32,
            buffer=nl.sbuf
        )
        nisa.tensor_scalar(
            dst=min_result,
            data=max_negated,
            op0=nl.multiply,
            operand0=-1.0
        )

        # Cast to output dtype
        if x_input.dtype != nl.float32:
            final_result = nl.ndarray((1, spatial_out), dtype=x_input.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=final_result, src=min_result)
        else:
            final_result = min_result

        # Store output
        nisa.dma_copy(
            dst=output[b_idx:b_idx+1, 0:spatial_out],
            src=final_result
        )

    return output
