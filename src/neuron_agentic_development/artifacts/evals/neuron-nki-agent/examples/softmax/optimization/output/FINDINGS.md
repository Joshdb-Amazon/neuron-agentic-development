# Softmax Kernel Optimization - Findings

## Baseline Analysis

**Kernel**: Numerically stable softmax along last dimension of 2D tensor [1024, 8192] float32.

**Algorithm**: `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`

**Baseline profile** (348.78 μs):
- VectorE: 51.31% (reductions, element-wise ops)
- ScalarE: 33.94% (activation/exp)
- DMA: 47.74%
- TensorE: 0.89% (no matmul in softmax)
- MBU: 26.87%
- HBM read/write: 32MB each (64MB total)
- 73 vector + 45 scalar instructions

**Classification**: Mixed bottleneck — VectorE, ScalarE, and DMA all significant. Element-wise kernel with arithmetic intensity ~1 (memory-bound by nature).

---

## v1: Fuse subtract+exp into single activation instruction

### Bottleneck targeted
The baseline uses separate `nisa.tensor_scalar(subtract)` + `nisa.activation(exp)` instructions. The intermediate `shifted_sb` buffer requires a full data pass over the [P, C] tile just for subtraction.

### Profiling evidence
- 73 vector instructions, 45 scalar instructions in baseline
- Two separate data passes over the large tile for subtract and exp

### Code change
Replaced:
```python
nisa.tensor_scalar(dst=shifted_sb, data=input_sb, op0=nl.subtract, operand0=max_sb)
nisa.activation(dst=exp_sb, data=shifted_sb, op=nl.exp)
```
With:
```python
neg_max_sb = nisa.tensor_scalar(data=max_sb, op0=nl.multiply, operand0=-1.0)
nisa.activation(dst=exp_sb, data=input_sb, op=nl.exp, scale=1.0, bias=neg_max_sb)
```

The `nisa.activation` API computes `op(data * scale + bias)`, so `exp(input * 1.0 + (-max))` = `exp(input - max)` in a single ScalarE instruction. The negate of max is a cheap [P, 1] operation instead of a full [P, C] tile subtract.

### Before/after metrics

| Metric | Baseline | v1 | Change |
|--------|----------|-----|--------|
| total_time (μs) | 348.78 | 285.00 | **-18.3%** |
| vector_engine (%) | 51.31 | 49.76 | -1.55pp |
| scalar_engine (%) | 33.94 | 42.38 | +8.44pp |
| dma_active (%) | 47.74 | 58.34 | +10.6pp |
| vector_instr | 73 | 58 | -15 |
| scalar_instr | 45 | 59 | +14 |

### Result: SUCCESS — 18.3% improvement

---

## v2: In-place buffer reuse

### Bottleneck targeted
v1 allocates separate SBUF buffers for each intermediate (input_sb, exp_sb, result_sb). Reusing buffers reduces SBUF pressure and allocation overhead.

### Code change
- Reuse `tile` buffer for exp result (overwrite input after activation)
- Reuse `tile` buffer for final multiply result (overwrite exp after multiply)
- Eliminated 2 separate buffer allocations per tile

### Before/after metrics

| Metric | v1 | v2 | Change |
|--------|-----|-----|--------|
| total_time (μs) | 285.00 | 275.25 | **-3.4%** |
| vector_engine (%) | 49.76 | 51.24 | +1.48pp |
| scalar_engine (%) | 42.38 | 43.64 | +1.26pp |
| dma_active (%) | 58.34 | 60.24 | +1.9pp |
| vector_instr | 58 | 51 | -7 |
| scalar_instr | 59 | 52 | -7 |

### Result: MARGINAL — 3.4% improvement (below 10% target)

---

## v3: Fuse exp+sum_reduce into single activation instruction

### Bottleneck targeted
v2 uses separate `nisa.activation(exp)` and `nisa.tensor_reduce(sum)` instructions. The `nisa.activation` API supports `reduce_op` and `reduce_cmd` parameters that can compute the activation AND accumulate a free-axis reduction in the same hardware pass.

### Profiling evidence
- Two separate ScalarE/VectorE passes over the [P, C] tile for exp and sum
- ScalarE and VectorE both ~50% utilized — room to combine work

### Code change
Replaced:
```python
nisa.activation(dst=tile, data=tile, op=nl.exp, scale=1.0, bias=neg_max_sb)
nisa.tensor_reduce(dst=sum_sb, data=tile, op=nl.add, axis=1)
```
With:
```python
nisa.activation(
    dst=tile, data=tile, op=nl.exp, scale=1.0, bias=neg_max_sb,
    reduce_op=nl.add, reduce_res=sum_sb,
    reduce_cmd=nisa.reduce_cmd.reset_reduce,
)
```

This computes `exp(tile - max)` AND `sum(exp(tile - max))` in a single ScalarE instruction pass, eliminating the separate tensor_reduce call entirely.

### Before/after metrics

| Metric | v2 | v3 | Change |
|--------|-----|-----|--------|
| total_time (μs) | 275.25 | 182.17 | **-33.8%** |
| vector_engine (%) | 51.24 | 40.02 | -11.22pp |
| scalar_engine (%) | 43.64 | 66.62 | +22.98pp |
| dma_active (%) | 60.24 | 90.59 | +30.35pp |
| vector_instr | 51 | 43 | -8 |
| scalar_instr | 52 | 60 | +8 |

### Result: SUCCESS — 33.8% improvement

ScalarE now dominant (66.62%) and DMA nearly saturated (90.59%). Kernel has become DMA-bound.

---

## v4: Double buffering (FAILED)

### Bottleneck targeted
DMA at 90.59% — attempted to overlap DMA load of next tile with compute of current tile using ping-pong buffers.

### Code change
Pre-allocated two SBUF buffers. Prefetched next tile while computing current. Used `sequential_range` to control buffer alternation.

### Before/after metrics

| Metric | v3 | v4 (double buf) | Change |
|--------|-----|-----------------|--------|
| total_time (μs) | 182.17 | 233.03 | **+27.9%** |

### Result: FAILED — 28% regression

**Root cause**: `sequential_range` prevents the compiler from parallelizing across tile iterations. With only 8 tiles (1024/128), the pipelining benefit was outweighed by forced sequential execution overhead. The compiler with `affine_range` was already doing good cross-tile pipelining.

**Also tried**: Pair-processing (181.47 μs, -0.4%) and nl.* high-level API (not supported in this release).

---

## v5: Smaller tile size P=64 (FAILED)

### Bottleneck targeted
DMA-bound kernel — tried doubling tile count (8→16) for more pipelining opportunities with half-sized tiles (64 rows instead of 128).

### Before/after metrics

| Metric | v3 | v5 (P=64) | Change |
|--------|-----|-----------|--------|
| total_time (μs) | 182.17 | 739.12 | **+306%** |

### Result: FAILED — severe regression

**Root cause**: Using only 64 partitions wastes half the hardware parallelism (128 parallel SBUF/PSUM lanes). Instruction count increased significantly.

**Also tried**: Forcing negate to ScalarE for engine overlap (185.70 μs, +1.9%). No improvement — compiler already making good engine assignment decisions.

---

## Why further optimization is difficult

The optimized kernel (v3) is DMA-bound at 90.59% DMA utilization. Key constraints:

1. **Minimum HBM traffic**: 32MB read + 32MB write = 64MB is unavoidable (must read input, must write output)
2. **Arithmetic intensity ≈ 1**: Softmax performs ~6 ops per element but reads and writes each element once. This is inherently memory-bound.
3. **No data reuse**: Unlike matmul, softmax doesn't reuse data across dimensions
4. **Compiler already pipelining**: `affine_range` enables cross-tile overlap that manual approaches couldn't beat
5. **P=128 is the hardware maximum**: Can't increase parallelism further

The kernel achieves effective bandwidth of ~351 GB/s (64MB / 182μs), which is a reasonable fraction of per-core HBM bandwidth on trn2.
