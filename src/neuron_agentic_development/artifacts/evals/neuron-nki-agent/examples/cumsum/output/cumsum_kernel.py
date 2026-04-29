"""
NKI Cumsum Kernel

Implements torch.cumsum along the last dimension using nisa.tensor_tensor_scan.

For a tensor of shape (batch_size, seq_len), cumsum along dim=1 computes:
    out[i, j] = sum(x[i, 0:j+1]) for all i, j

This kernel uses the associative scan pattern:
    result[i] = 1 * result[i-1] + x[i]  (starting from result[-1] = 0)
"""

import math
from typing import Tuple

import nki
import nki.isa as nisa
import nki.language as nl


# ============================================================================
# TiledRange utility (self-contained)
# ============================================================================

from nki.language import NKIObject


class TiledRangeIterator(NKIObject):
    """Represents a single tile in a tiled range."""

    def __init__(self, tile_size: int, tile_index: int, start_offset: int, end_offset: int):
        self.size = tile_size
        self.index = tile_index
        self.start_offset = start_offset
        self.end_offset = end_offset

    def __repr__(self) -> str:
        return f"TiledRangeIterator(size={self.size}, index={self.index}, start_offset={self.start_offset})"


def TiledRange(size, tile_size: int) -> Tuple[TiledRangeIterator, ...]:
    """Divides a dimension into tiles and returns a tuple of TiledRangeIterators."""
    if isinstance(size, TiledRangeIterator):
        total_size = size.size
        base_offset = size.start_offset
    else:
        total_size = size
        base_offset = 0

    num_tiles = math.ceil(total_size / tile_size)

    iterators = []
    for i in range(num_tiles):
        relative_offset = i * tile_size
        current_tile_size = min(tile_size, total_size - relative_offset)
        start_offset = base_offset + relative_offset
        end_offset = start_offset + current_tile_size
        iterators.append(TiledRangeIterator(current_tile_size, i, start_offset, end_offset))

    return tuple(iterators)


def div_ceil(n: int, d: int) -> int:
    """Ceiling division."""
    return (n + d - 1) // d


# ============================================================================
# Cumsum Kernel
# ============================================================================

# Hardware constraints
P_MAX = 128  # Maximum partition dimension


@nki.jit
def cumsum_kernel(x: nl.ndarray) -> nl.ndarray:
    """
    Compute cumulative sum along the last dimension.

    Args:
        x: Input tensor of shape (outer_dim, last_dim) where outer_dim can be
           the product of batch dimensions.

    Returns:
        Output tensor of same shape as x with cumsum applied along dim=-1.

    Notes:
        - Uses tensor_tensor_scan for efficient sequential scan operation
        - Tiles partition dimension (≤128) and free dimension for carry dependency
        - Supports any 2D input shape
    """
    # Get input shape
    outer_dim = x.shape[0]
    last_dim = x.shape[1]

    # Tiling parameters
    F_TILE_SIZE = min(last_dim, 2048)  # Free dimension tile size (2KB for good DMA efficiency)
    num_f_tiles = div_ceil(last_dim, F_TILE_SIZE)

    # Allocate output in HBM
    y = nl.ndarray((outer_dim, last_dim), dtype=x.dtype, buffer=nl.shared_hbm)

    # Create a ones buffer for the scan (multiply by 1)
    ones_sb = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_sb, value=1.0)

    # Iterate over partition tiles (outer dimension)
    for p_tile in TiledRange(outer_dim, P_MAX):
        # Initialize carry state for this partition tile
        init_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=init_sb, value=0.0)

        # Iterate over free dimension tiles (with sequential dependency)
        for f_tile_idx in nl.sequential_range(num_f_tiles):
            f_start = f_tile_idx * F_TILE_SIZE
            f_end = min(f_start + F_TILE_SIZE, last_dim)
            f_size = f_end - f_start

            # Load input tile from HBM to SBUF
            data_sb = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=data_sb[0:p_tile.size, 0:f_size],
                src=x[p_tile.start_offset:p_tile.start_offset + p_tile.size, f_start:f_end],
            )

            # Perform cumsum scan: result[i] = 1 * result[i-1] + data[i]
            result_sb = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor_scan(
                dst=result_sb[0:p_tile.size, 0:f_size],
                data0=ones_sb[0:p_tile.size, 0:f_size],   # multiply by 1 (identity for sum)
                data1=data_sb[0:p_tile.size, 0:f_size],   # add input value
                initial=init_sb[0:p_tile.size, 0:1],      # carry from previous tile
                op0=nl.multiply,
                op1=nl.add,
            )

            # Store result to HBM
            nisa.dma_copy(
                dst=y[p_tile.start_offset:p_tile.start_offset + p_tile.size, f_start:f_end],
                src=result_sb[0:p_tile.size, 0:f_size],
            )

            # Update carry for next tile: last value of current result
            # Only update if there are more tiles to process
            if f_tile_idx + 1 < num_f_tiles:
                nisa.tensor_copy(
                    dst=init_sb[0:p_tile.size, 0:1],
                    src=result_sb[0:p_tile.size, f_size-1:f_size],
                )

    return y
