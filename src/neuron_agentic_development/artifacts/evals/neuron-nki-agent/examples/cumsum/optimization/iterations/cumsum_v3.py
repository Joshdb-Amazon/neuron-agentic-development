"""
NKI Cumsum Kernel - V3: Double buffering + larger tiles (F_TILE_SIZE=4096)

Hypothesis: Larger tiles without double buffering regressed (30.8ms) because
of poor DMA/compute overlap. With double buffering, larger tiles should
benefit from fewer sequential iterations (8 vs 16), fewer carry copies,
and potentially better DMA efficiency from larger transfers.
"""

import nki
import nki.isa as nisa
import nki.language as nl


from nki.language import NKIObject
from typing import Tuple
import math


class TiledRangeIterator(NKIObject):
    def __init__(self, tile_size: int, tile_index: int, start_offset: int, end_offset: int):
        self.size = tile_size
        self.index = tile_index
        self.start_offset = start_offset
        self.end_offset = end_offset


def TiledRange(size, tile_size: int) -> Tuple[TiledRangeIterator, ...]:
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
    return (n + d - 1) // d


P_MAX = 128


@nki.jit
def cumsum_kernel(x: nl.ndarray) -> nl.ndarray:
    outer_dim = x.shape[0]
    last_dim = x.shape[1]

    # V3: Larger tiles (4096 vs 2048) with double buffering
    F_TILE_SIZE = min(last_dim, 4096)
    num_f_tiles = div_ceil(last_dim, F_TILE_SIZE)

    y = nl.ndarray((outer_dim, last_dim), dtype=x.dtype, buffer=nl.shared_hbm)

    ones_sb = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_sb, value=1.0)

    for p_tile in TiledRange(outer_dim, P_MAX):
        init_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=init_sb, value=0.0)

        # Double-buffer
        data_sb_0 = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=x.dtype, buffer=nl.sbuf)
        data_sb_1 = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=x.dtype, buffer=nl.sbuf)
        result_sb_0 = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        result_sb_1 = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=nl.float32, buffer=nl.sbuf)

        # Prefetch first tile
        nisa.dma_copy(
            dst=data_sb_0[0:p_tile.size, 0:F_TILE_SIZE],
            src=x[p_tile.start_offset:p_tile.start_offset + p_tile.size, 0:F_TILE_SIZE],
        )

        for f_tile_idx in nl.sequential_range(num_f_tiles):
            f_start = f_tile_idx * F_TILE_SIZE
            f_end = min(f_start + F_TILE_SIZE, last_dim)
            f_size = f_end - f_start

            if f_tile_idx % 2 == 0:
                cur_data = data_sb_0
                cur_result = result_sb_0
                nxt_data = data_sb_1
            else:
                cur_data = data_sb_1
                cur_result = result_sb_1
                nxt_data = data_sb_0

            # Prefetch next tile
            if f_tile_idx + 1 < num_f_tiles:
                next_f_start = (f_tile_idx + 1) * F_TILE_SIZE
                next_f_end = min(next_f_start + F_TILE_SIZE, last_dim)
                next_f_size = next_f_end - next_f_start
                nisa.dma_copy(
                    dst=nxt_data[0:p_tile.size, 0:next_f_size],
                    src=x[p_tile.start_offset:p_tile.start_offset + p_tile.size, next_f_start:next_f_end],
                )

            # Scan current tile
            nisa.tensor_tensor_scan(
                dst=cur_result[0:p_tile.size, 0:f_size],
                data0=ones_sb[0:p_tile.size, 0:f_size],
                data1=cur_data[0:p_tile.size, 0:f_size],
                initial=init_sb[0:p_tile.size, 0:1],
                op0=nl.multiply,
                op1=nl.add,
            )

            # Store result
            nisa.dma_copy(
                dst=y[p_tile.start_offset:p_tile.start_offset + p_tile.size, f_start:f_end],
                src=cur_result[0:p_tile.size, 0:f_size],
            )

            # Update carry
            if f_tile_idx + 1 < num_f_tiles:
                nisa.tensor_copy(
                    dst=init_sb[0:p_tile.size, 0:1],
                    src=cur_result[0:p_tile.size, f_size-1:f_size],
                )

    return y
