# NKI Language - Dimensions

> **Module**: nki.language
> **Total Functions**: 7

## Overview

Dimension and range management functions.

## Functions

### nki.language.affine_range {#nki-language-affine_range}

# nki.language.affine_range

nki.language.affine_range

nki.language.affine_range(*start*, *stop=None*, *step=1*)[[source]](../../../_modules/nki/language.html#affine_range)
Create a sequence of numbers for use as **parallel** loop iterators in NKI. `affine_range` should be the default
loop iterator choice, when there is **no** loop carried dependency. Note, associative reductions are **not** considered
loop carried dependencies in this context. A concrete example of associative reduction
is multiple nl.matmul
or [nisa.nc_matmul](nki.isa.nc_matmul.md) calls accumulating into the same
output buffer defined outside of this loop level (see code example #2 below).

When the above conditions are not met, we recommend using [sequential_range](nki.language.sequential_range.md)
instead.

Notes:

* Using `affine_range` prevents Neuron compiler from unrolling the loops until entering compiler backend,
which typically results in better compilation time compared to the fully unrolled iterator
[static_range](nki.language.static_range.md).

* Using `affine_range` also allows Neuron compiler to perform additional loop-level optimizations, such as
loop vectorization in current release. The exact type of loop-level optimizations applied is subject
to changes in future releases.

* Since each kernel instance only runs on a single NeuronCore, affine_range does **not** parallelize
different loop iterations across multiple NeuronCores. However, different iterations could be parallelized/pipelined
on different compute engines within a NeuronCore depending on the invoked instructions (engines) and data dependency
in the loop body.


```python
import nki.language as nl

#######################################################################
# Example 1: No loop carried dependency
# Input/Output tensor shape: [128, 2048]
# Load one tile ([128, 512]) at a time, square the tensor element-wise,
# and store it into output tile
#######################################################################

# Every loop instance works on an independent input/output tile.
# No data dependency between loop instances.
for i_input in nl.affine_range(input.shape[1] // 512):
  offset = i_input * 512
  input_sb = nl.ndarray((input.shape[0], 512), dtype=input.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=input_sb, src=input[0:input.shape[0], offset:offset+512])
  result = nl.multiply(input_sb, input_sb)
  nisa.dma_copy(dst=output[0:input.shape[0], offset:offset+512], src=result)

#######################################################################
# Example 2: Matmul output buffer accumulation, a type of associative reduction
# Input tensor shapes for nl.matmul: xT[K=2048, M=128] and y[K=2048, N=128]
# Load one tile ([128, 128]) from both xT and y at a time, matmul and
# accumulate into the same output buffer
#######################################################################

result_psum = nl.zeros((128, 128), dtype=nl.float32, buffer=nl.psum)
for i_K in nl.affine_range(xT.shape[0] // 128):
  offset = i_K * 128
  xT_sbuf = nl.ndarray((128, xT.shape[1]), dtype=xT.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=xT_sbuf, src=xT[offset:offset+128, 0:xT.shape[1]])
  y_sbuf = nl.ndarray((128, y.shape[1]), dtype=y.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=y_sbuf, src=y[offset:offset+128, 0:y.shape[1]])

  result_psum += nl.matmul(xT_sbuf, y_sbuf, transpose_x=True)
```

---

### nki.language.num_programs {#nki-language-num_programs}

# nki.language.num_programs

nki.language.num_programs

nki.language.num_programs(*axes=None*)[[source]](../../../_modules/nki/language.html#num_programs)
Number of SPMD programs along the given axes in the launch grid. If `axes` is not provided,
returns the total number of programs.

Parameters:
**axes** – The axes of the ND launch grid. If not provided, returns the total number of programs along the entire launch grid.

Returns:
The number of SPMD(single process multiple data) programs along `axes` in the launch grid

---

### nki.language.program_id {#nki-language-program_id}

# nki.language.program_id

nki.language.program_id

nki.language.program_id(*axis*)[[source]](../../../_modules/nki/language.html#program_id)
Index of the current SPMD program along the given axis in the launch grid.

Parameters:
**axis** – The axis of the ND launch grid.

Returns:
The program id along `axis` in the launch grid

---

### nki.language.program_ndim {#nki-language-program_ndim}

# nki.language.program_ndim

nki.language.program_ndim

nki.language.program_ndim()[[source]](../../../_modules/nki/language.html#program_ndim)
Number of dimensions in the SPMD launch grid.

Returns:
The number of dimensions in the launch grid, i.e. the number of axes

---

### nki.language.sequential_range {#nki-language-sequential_range}

# nki.language.sequential_range

nki.language.sequential_range

nki.language.sequential_range(*start*, *stop*, *step*)[[source]](../../../_modules/nki/language.html#sequential_range)
Create a sequence of numbers for use as **sequential** loop iterators in NKI. `sequential_range`
should be used when there is a loop carried dependency. Note, associative reductions are **not** considered
loop carried dependencies in this context. See [affine_range](nki.language.affine_range.md) for
an example of such associative reduction.

Notes:

* Inside a NKI kernel, any use of Python `range(...)` will be replaced with `sequential_range(...)`
by Neuron compiler.

* Using `sequential_range` prevents Neuron compiler from unrolling the loops until entering compiler backend,
which typically results in better compilation time compared to the fully unrolled iterator
[static_range](nki.language.static_range.md).

* Using `sequential_range` informs Neuron compiler to respect inter-loop dependency and perform
much more conservative loop-level optimizations compared to `affine_range`.

* Using `affine_range` instead of `sequential_range` in case of loop carried dependency
incorrectly is considered unsafe and could lead to numerical errors.


```python
import nki.language as nl

#######################################################################
# Example 1: Loop carried dependency from tiling tensor_tensor_scan
# Both sbuf tensor input0 and input1 shapes: [128, 2048]
# Perform a scan operation between the two inputs using a tile size of [128, 512]
# Store the scan output to another [128, 2048] tensor
#######################################################################

# Loop iterations communicate through this init tensor
init = nl.zeros((128, 1), dtype=input0.dtype)

# This loop will only produce correct results if the iterations are performed in order
for i_input in nl.sequential_range(input0.shape[1] // 512):
  offset = i_input * 512

  # Depends on scan result from the previous loop iteration
  result = nisa.tensor_tensor_scan(input0[:, offset:offset+512],
                                   input1[:, offset:offset+512],
                                   initial=init,
                                   op0=nl.multiply, op1=nl.add)

  nl.store(output[0:input0.shape[0], offset:offset+512], result)

  # Prepare initial result for scan in the next loop iteration
  init[:, :] = result[:, 511]
```

---

### nki.language.static_range {#nki-language-static_range}

# nki.language.static_range

nki.language.static_range

nki.language.static_range(*start*, *stop=None*, *step=1*)[[source]](../../../_modules/nki/language.html#static_range)
Create a sequence of numbers for use as loop iterators in NKI, resulting in a fully unrolled loop.
Unlike [affine_range](nki.language.affine_range.md) or [sequential_range](nki.language.sequential_range.md),
Neuron compiler will fully unroll the loop during NKI kernel tracing.

Notes:

* Due to loop unrolling, compilation time may go up significantly compared to
[affine_range](nki.language.affine_range.md) or [sequential_range](nki.language.sequential_range.md).

* On-chip memory (SBUF) usage may also go up significantly compared to
[affine_range](nki.language.affine_range.md) or [sequential_range](nki.language.sequential_range.md).

* No loop-level optimizations will be performed in the compiler.

* `static_range` should only be used as a fall-back option for debugging purposes when
[affine_range](nki.language.affine_range.md) or [sequential_range](nki.language.sequential_range.md)
is giving functionally incorrect results or undesirable performance characteristics.

---

### nki.language.tile_size {#nki-language-tile_size}

# nki.language.tile_size

nki.language.tile_size

*class *nki.language.tile_size[[source]](../../../_modules/nki/language.html#tile_size)
Tile size constants.

Attributes


| bn_stats_fmax | Maximum free dimension of BN_STATS |
| --- | --- |
| gemm_moving_fmax | Maximum free dimension of the moving operand of General Matrix Multiplication on Tensor Engine |
| gemm_stationary_fmax | Maximum free dimension of the stationary operand of General Matrix Multiplication on Tensor Engine |
| pmax | Maximum partition dimension of a tile |
| psum_fmax | Maximum free dimension of a tile on PSUM buffer |
| psum_min_align | Minimum byte alignment requirement for PSUM free dimension address |
| sbuf_min_align | Minimum byte alignment requirement for SBUF free dimension address |
| total_available_sbuf_size | Total SBUF available size |

---
