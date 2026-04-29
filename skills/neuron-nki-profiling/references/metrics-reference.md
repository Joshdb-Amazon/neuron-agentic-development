# NKI Profiling Metrics Reference

Comprehensive reference for interpreting Neuron Profile metrics through the lens of NeuronCore architecture.

## Quick Reference

| Metric | Description | Target |
|--------|-------------|--------|
| `latency` | Total kernel execution time (ms) | Lower is better |
| `mfu_estimated_percent` | Model FLOPs Utilization | >90% for compute-bound matmul kernels |
| `mbu_estimated_percent` | Memory Bandwidth Utilization | >60% for memory-bound kernels |
| `tensor_engine_active_time_percent` | TensorE utilization | >90% for matmul-heavy workloads |
| `hbm_read_bytes` / `hbm_write_bytes` | HBM traffic | Minimize; compare to inputs_outputs_weights_size_bytes |
| `sbuf_spill_bytes` / `psum_spill_bytes` | Spilled data | Should be 0; indicates SBUF/PSUM capacity exceeded |
| `transpose_flops` / `hardware_flops` | Transpose overhead ratio | <5% is excellent, <15% is acceptable |
| `mm_arithmetic_intensity` | FLOPs per byte of HBM traffic | Compare to peak_flops_bandwidth_ratio |
| `dynamic_dma_packet_percent` | Dynamic DMA efficiency | >80% indicates well-optimized DMAs |
| `dma_active_time_percent` | DMA utilization | Context-dependent; >60% for memory-bound |

---

## Understanding NeuronCore Architecture for Profiling

Before diving into metrics, understanding NeuronCore's architecture helps interpret profiling data. Metrics reveal how your kernel interacts with specific hardware components.

### Key Architectural Components

**TensorE (Tensor Engine)**
- 128×128 systolic array for matrix multiplication
- Primary source of FLOPs (>90% of hardware compute capacity)
- Performs both useful matmuls and PF-transposes (partition ↔ free dimension layout adjustments)
- Metrics: `tensor_engine_active_time_percent`, `hardware_flops`, `mfu_estimated_percent`, `hfu_estimated_percent`

**VectorE (Vector Engine)**
- 128 parallel lanes for element-wise operations
- Handles reductions, activations, normalization operations
- Can operate in parallel with TensorE
- Metrics: `vector_engine_active_time_percent`

**ScalarE (Scalar Engine)**
- Element-wise scalar operations per partition (exponentiation, scaling, biasing)
- Often pipelined with VectorE for complex operations
- Metrics: `scalar_engine_active_time_percent`

**GpSimdE (General-Purpose SIMD Engine)**
- Custom SIMD operations not mappable to other engines
- Metrics: `gpsimd_engine_active_time_percent`

**DMA Engines**
- 16 independent engines for moving data between HBM and SBUF
- Ideal transfer size: **32 KiB per engine** for good bandwidth utilization
- Can generate transposes on-the-fly (with significant bandwidth penalty)
- Metrics: `dma_active_time_percent`, `dma_transfer_count`, `dynamic_dma_packet_percent`

### Memory Hierarchy

**HBM (High Bandwidth Memory)**
- Off-chip device memory with limited bandwidth
- All kernel inputs/outputs reside here
- Metrics: `hbm_read_bytes`, `hbm_write_bytes`, `mbu_estimated_percent`

**SBUF (State Buffer)**
- **24 MB** on-chip memory organized into **128 partitions**
- Primary working memory for all compute engines
- Spilling occurs when working set exceeds capacity
- Metrics: `sbuf_read_bytes`, `sbuf_write_bytes`, `sbuf_spill_bytes`

**PSUM (Partial Sum Buffer)**
- On-chip accumulator for TensorE matmul outputs
- Free dimension limits: **512 elements (gen2/3)**, **4096 elements (gen4)**
- Exceeding limit triggers spilling
- Metrics: `psum_read_bytes`, `psum_write_bytes`, `psum_spill_bytes`

### The Roofline Model

Performance is fundamentally bounded by either:
1. **Compute throughput** (TensorE max ops/sec) → **compute-bound**
2. **Memory bandwidth** (HBM bandwidth) → **memory-bound**

The crossover point depends on **arithmetic intensity**: FLOPs performed per byte of HBM traffic.
- High arithmetic intensity (> `peak_flops_bandwidth_ratio`) → compute-bound
- Low arithmetic intensity (< `peak_flops_bandwidth_ratio`) → memory-bound

**Optimization goal**: Maximize achieved performance given the algorithmic arithmetic intensity constraint.

See [NKI Performance Guide](../../neuron-nki-docs/references/optimization/nki_perf_guide.md) for optimization strategies.

---

## Complete Metrics Catalog

### 1. Overall Performance Indicators

#### `total_time` / `latency`
- **What it measures**: End-to-end kernel execution time on device (seconds for total_time, milliseconds for latency)
- **Architectural context**: Sum of all engine activities + DMA transfers + synchronization overhead
- **Related metrics**: All engine utilization and DMA metrics contribute to total time
- **Usage**: Primary performance metric; lower is better

#### `throughput`
- **What it measures**: Operations per second (derived from latency and workload size)
- **Architectural context**: How efficiently NeuronCore processes the workload
- **Usage**: Comparing different kernel implementations or tile sizes for the same operation

---

### 2. Arithmetic Intensity and Roofline Metrics

#### `mm_arithmetic_intensity`
- **What it measures**: Ratio of (hardware_flops - transpose_flops) / (hbm_write_bytes + hbm_read_bytes)
- **Formula**: Useful compute FLOPs ÷ total HBM traffic in bytes
- **Architectural context**: FLOPs per byte of HBM traffic (excluding transpose overhead)
- **Interpretation**:
  - Higher is better (more compute reuse per data fetch from HBM)
  - Compare to `peak_flops_bandwidth_ratio` to determine if workload should be compute or memory bound
- **Optimization**: [Opt #1](../../neuron-nki-docs/references/optimization/nki_perf_guide.md#opt-1-exploit-temporal-locality-to-minimize-input-data-reloading) (temporal locality) and [Opt #2](../../neuron-nki-docs/references/optimization/nki_perf_guide.md#opt-2-fuse-operations-to-minimize-intermediate-data-spilling) (operator fusion) increase this
- **Related metrics**: `peak_flops_bandwidth_ratio`, `hardware_flops`, `transpose_flops`, `hbm_read_bytes`, `hbm_write_bytes`

#### `peak_flops_bandwidth_ratio`
- **What it measures**: Hardware's theoretical max FLOP rate ÷ max HBM bandwidth
- **Architectural context**: The arithmetic intensity threshold for compute-bound execution (the "knee" of the Roofline curve)
- **Interpretation**:
  - If `mm_arithmetic_intensity` > this value → workload should be compute-bound
  - If `mm_arithmetic_intensity` < this value → workload should be memory-bound
  - Gap between them indicates optimization opportunity
- **Usage**: Fundamental for understanding performance bottleneck type

#### `mfu_max_achievable_estimated_percent`
- **What it measures**: Best possible MFU given the current arithmetic intensity (Roofline ceiling)
- **Formula**: (model_flops / hbm_bytes) / (max_ops / max_bandwidth)
- **Interpretation**:
  - Large gap from `mfu_estimated_percent` → data movement inefficiency ([Opt #4](../../neuron-nki-docs/references/optimization/nki_perf_guide.md#opt-4-overlap-data-loading-with-computation), [#9](../../neuron-nki-docs/references/optimization/nki_perf_guide.md#opt-9-perform-sufficiently-large-dma-transfers), [#10](../../neuron-nki-docs/references/optimization/nki_perf_guide.md#opt-10-minimize-use-of-dma-transposes))
  - Small gap → hitting fundamental Roofline limit, need algorithmic changes to increase arithmetic intensity
- **Variants**: `mfu_hlo_max_achievable_estimated_percent` (HLO-based), `mfu_inst_max_achievable_estimated_percent` (instruction-based)

---

### 3. Compute Efficiency Metrics

#### `hardware_flops`
- **What it measures**: Total floating-point operations from ALL TensorE instructions (including transposes)
- **Formula**: Sum of (2 × MAC_count × active_rows × active_cols × elements) for each matmul
- **Architectural context**: Includes both useful matmuls and transpose-induced matmuls
- **Note**: Each multiply-add operation counts as 2 FLOPs
- **Related metrics**: `transpose_flops` (subset), `model_flops` (excludes compiler-inserted overhead)

#### `model_flops`
- **What it measures**: Floating-point operations from the algorithm definition (HLO stats)
- **Architectural context**: Represents useful compute only; excludes compiler-inserted transposes
- **Availability**: Only for compiler-generated code; **NOT available for raw NKI kernels**
- **Comparison**: `hardware_flops` ≥ `model_flops` (equality means no transpose overhead)

#### `transpose_flops`
- **What it measures**: FLOPs spent on transpose-induced matmul instructions (subset of `hardware_flops`)
- **Architectural context**: TensorE performs PF-transposes (partition ↔ free dimension swaps) when producer/consumer operations have layout mismatches
- **Types**:
  - **IO transposes**: Kernel input/output layout vs compute operation requirements
  - **Intermediate transposes**: Producer/consumer operation layout mismatches
- **Interpretation**: High `transpose_flops` / `hardware_flops` ratio indicates layout inefficiency
- **Optimization**: [Opt #8](../../neuron-nki-docs/references/optimization/nki_perf_guide.md#opt-8-tensorengine-only-mitigating-overhead-from-tensor-transposes) explains mitigation strategies

#### `mfu_estimated_percent` (Model FLOPs Utilization)
- **What it measures**: Achieved TensorE utilization for **useful** algorithmic compute
- **Formula (HLO available)**: `model_flops` / (tensor_engine_max_ops_per_sec × total_time)
- **Formula (HLO unavailable, e.g., NKI kernels)**: (`hardware_flops` - `transpose_flops`) / (tensor_engine_max_ops_per_sec × total_time)
- **Architectural context**: How efficiently TensorE is used for useful work (not overhead)
- **Targets from perf guide**: >90% for matmul-heavy compute-bound kernels
- **Related metrics**: `hfu_estimated_percent` (includes transposes), `mfu_hlo_estimated_percent`, `mfu_inst_estimated_percent`

#### `mfu_hlo_estimated_percent`
- **What it measures**: MFU calculated from HLO stats (if available)
- **Architectural context**: Based on model definition FLOPs, excludes all compiler-inserted overhead
- **Note**: Will be 0 for NKI kernels (no HLO stats)

#### `mfu_inst_estimated_percent`
- **What it measures**: MFU calculated from instruction trace (excludes transposes)
- **Formula**: (`hardware_flops` - `transpose_flops`) / (tensor_engine_max_ops_per_sec × total_time)
- **Architectural context**: Based on actual executed instructions, not model definition
- **Usage**: Available for all kernels including NKI; represents useful compute from trace

#### `hfu_estimated_percent` (Hardware FLOPs Utilization)
- **What it measures**: Achieved TensorE utilization including ALL work (useful + transposes)
- **Formula**: `hardware_flops` / (tensor_engine_max_ops_per_sec × total_time)
- **Architectural context**: Total TensorE activity relative to theoretical maximum
- **Interpretation**: `hfu_estimated_percent` - `mfu_estimated_percent` = transpose overhead percentage
- **Related metrics**: `mfu_estimated_percent`, `transpose_flops`

#### `matmul_instruction_count`
- **What it measures**: Total number of MATMUL instructions executed on TensorE
- **Architectural context**: More instructions = smaller tiles (higher instruction overhead per tile)
- **Optimization**: [Opt #5a](../../neuron-nki-docs/references/optimization/nki_perf_guide.md#opt-5a-use-sufficiently-large-input-tiles-in-free-dimension) (free dimension sizing), [Opt #5b](../../neuron-nki-docs/references/optimization/nki_perf_guide.md#opt-5b-use-sufficiently-large-input-tiles-in-partition-dimension) (partition dimension sizing)
- **Trade-off**: Larger tiles (fewer instructions, lower overhead) vs SBUF pressure (spilling risk)

---

### 4. Memory Hierarchy Metrics

#### `sbuf_read_bytes` / `sbuf_write_bytes`
- **What it measures**: Total bytes read from / written to SBUF by all engines and DMAs
- **Architectural context**: SBUF is 24 MB on-chip memory with 128 partitions
  - All compute engines read operands from SBUF
  - DMAs write loaded data to SBUF, read output data from SBUF
- **Interpretation**: High traffic is expected; imbalance may indicate unnecessary loads
- **Related metrics**: `sbuf_spill_bytes` (subset indicating spills)

#### `psum_read_bytes` / `psum_write_bytes`
- **What it measures**: Total bytes read from / written to PSUM accumulator
- **Architectural context**: PSUM stores TensorE matmul outputs before moving to SBUF
  - Free dimension limit: 512 elements (gen2/3), 4096 elements (gen4)
  - Exceeding limit triggers spilling
- **Typical workloads**: High for matmul-heavy kernels
- **Related metrics**: `psum_spill_bytes`

#### `sbuf_spill_bytes` / `psum_spill_bytes`
- **What it measures**: Bytes spilled from SBUF/PSUM to HBM when working set exceeds on-chip capacity
- **Architectural context**: Compiler automatically inserts spill/reload when:
  - SBUF usage exceeds 24 MB capacity
  - PSUM free dimension exceeds generation-specific limit (512 gen2/3, 4096 gen4)
- **Performance impact**: Creates extra HBM traffic, reduces arithmetic intensity
- **Optimization**: [Opt #2](../../neuron-nki-docs/references/optimization/nki_perf_guide.md#opt-2-fuse-operations-to-minimize-intermediate-data-spilling) (operator fusion) reduces intermediate data size
- **Recommended threshold from perf guide**: `spill_save_bytes` / `sbuf_read_bytes` < 30%
- **Related metrics**: `spill_save_bytes`, `spill_reload_bytes`

#### `spill_save_bytes` / `spill_reload_bytes`
- **What it measures**: Total bytes saved to HBM / reloaded from HBM due to spilling
- **Architectural context**: `spill_save_bytes` ≤ `spill_reload_bytes` (tensors can be reloaded multiple times)
- **Diagnosis**: Compare `spill_save_bytes` against `sbuf_read_bytes` + `psum_read_bytes` to assess severity
- **Related metrics**: `sbuf_spill_bytes`, `psum_spill_bytes`, `hbm_read_bytes` (spill reloads increase this)

#### `hbm_read_bytes` / `hbm_write_bytes`
- **What it measures**: Total bytes moved between HBM (device memory) and SBUF via DMA engines
- **Architectural context**: HBM is off-chip with limited bandwidth
- **Ideal minimum**: `inputs_outputs_weights_size_bytes` (no redundant loads or spills)
- **Excess traffic sources**:
  - Input reloading ([Opt #1](../../neuron-nki-docs/references/optimization/nki_perf_guide.md#opt-1-exploit-temporal-locality-to-minimize-input-data-reloading) symptom)
  - Spilling ([Opt #2](../../neuron-nki-docs/references/optimization/nki_perf_guide.md#opt-2-fuse-operations-to-minimize-intermediate-data-spilling) symptom)
  - Inefficient tiling
- **Related metrics**: `inputs_outputs_weights_size_bytes`, `mbu_estimated_percent`, `spill_save_bytes`

#### `inputs_outputs_weights_size_bytes`
- **What it measures**: Theoretical minimum HBM traffic (inputs + outputs + weights, each counted once)
- **Architectural context**: Best case if no reloading or spilling occurs
- **Usage**: Baseline for evaluating `hbm_read_bytes` + `hbm_write_bytes` efficiency
- **Comparison**: (hbm_read + hbm_write) / inputs_outputs_weights_size_bytes → overhead ratio

#### `inputs_and_weights_size_bytes`
- **What it measures**: Size of input tensors and weight tensors (subset of inputs_outputs_weights)
- **Usage**: For calculating `mbu_min_read_util_percent` (minimum read bandwidth needs)

#### `input_queue_bytes` / `output_queue_bytes` / `weight_queue_bytes`
- **What it measures**: Total transfer size per DMA queue type
- **Architectural context**: Different queue types for different data movement patterns
  - Input queues: Load activations from HBM to SBUF
  - Weight queues: Load model weights from HBM to SBUF
  - Output queues: Store results from SBUF to HBM
- **Usage**: Identifying which data movement dominates traffic

---

### 5. Data Movement (DMA) Metrics

#### `dma_active_time` / `dma_active_time_percent`
- **What it measures**: Duration when at least one DMA packet is transferring data
- **Architectural context**: 16 DMA engines can operate in parallel
  - Good utilization: Multiple engines busy simultaneously
  - Poor utilization: Engines mostly idle or sequential transfers
- **Interpretation**: High percentage (>60%) in memory-bound kernels indicates DMA keeping up with demand
- **Optimization**: [Opt #4](../../neuron-nki-docs/references/optimization/nki_perf_guide.md#opt-4-overlap-data-loading-with-computation) (overlap DMA with compute), [Opt #9](../../neuron-nki-docs/references/optimization/nki_perf_guide.md#opt-9-perform-sufficiently-large-dma-transfers) (large DMA transfers)

#### `dma_active_cycles`
- **What it measures**: `dma_active_time` multiplied by NeuronCore clock speed
- **Usage**: Alternative representation of DMA active time in cycles

#### `dma_transfer_count`
- **What it measures**: Number of high-level DMA transfers
- **Architectural context**: DMA "transfer" (compiler construct for moving one or more tensors) ≠ DMA "packet" (hardware unit)
  - One transfer can spawn multiple packets across DMA engines
- **Related metrics**: `dma_transfer_total_bytes`, `dma_transfer_average_bytes`

#### `dma_transfer_total_bytes`
- **What it measures**: Total transfer size of all DMA transfers
- **Architectural context**: Sum of all high-level DMA transfer sizes

#### `dma_transfer_average_bytes`
- **What it measures**: `dma_transfer_total_bytes` / `dma_transfer_count`
- **Optimization**: [Opt #9](../../neuron-nki-docs/references/optimization/nki_perf_guide.md#opt-9-perform-sufficiently-large-dma-transfers) explains ideal size: **32 KiB per DMA engine** for good bandwidth
- **Diagnosis**: Small average indicates overhead-dominated transfers

#### `dma_transfer_time`
- **What it measures**: Total duration of all DMA transfers (overlapping transfers summed)
- **Architectural context**: High-level compiler construct timing

#### `dma_packet_time`
- **What it measures**: Total duration of all DMA packets (overlapping packets summed)
- **Architectural context**: Packet = smallest unit of DMA data movement
  - Timeline view can show packets with "Show expanded DMA" option
  - Summing overlapping packets shows total DMA engine workload
- **Related metrics**: `dma_active_time` (considers parallel execution)

#### `dma_queue_count`
- **What it measures**: Total number of DMA queues in NEFF (input/output/weight queues)
- **Architectural context**: Each queue maps to a DMA engine group
- **Usage**: Understanding DMA parallelization structure

#### `dynamic_dma_packet_percent`
- **What it measures**: Percentage of DMA packets dynamically generated during execution
- **Architectural context**: Dynamic DMA (generated at runtime) is more efficient than static DMA (compile-time generated)
  - Higher percentage indicates better compiler optimization
  - Hardware-generated dynamic DMA is fastest (see `hardware_dynamic_dma_packet_percent`)
- **Targets from perf guide**: >80% for well-optimized kernels
- **Interpretation**:
  - Low (<60%) → Static DMA overhead dominates
  - High (>80%) → Well-optimized DMA generation
- **Related metrics**: `dynamic_dma_packet_count`, `hardware_dynamic_dma_packet_percent`

#### `dynamic_dma_packet_count`
- **What it measures**: Number of dynamically generated DMA packets
- **Related metrics**: Total packet count (not directly exposed) = dynamic + static

#### `dynamic_dma_size_percent`
- **What it measures**: Percentage of total transfer size for dynamically generated DMA packets
- **Usage**: Understanding if dynamic DMAs handle bulk of data movement

#### `dynamic_dma_active_time_percent`
- **What it measures**: Percentage of time when at least one dynamically generated DMA packet is transferring
- **Usage**: Temporal efficiency of dynamic DMAs

#### `hardware_dynamic_dma_packet_percent`
- **What it measures**: Percentage of DMA packets generated by dedicated hardware (subset of dynamic)
- **Architectural context**: Hardware DMA generation has lowest overhead (highest tier of efficiency)
- **Related metrics**: `dynamic_dma_packet_percent` (superset)

#### `hardware_dynamic_dma_packet_count` / `hardware_dynamic_dma_size` / `hardware_dynamic_dma_size_percent` / `hardware_dynamic_dma_active_time` / `hardware_dynamic_dma_active_time_percent`
- **What they measure**: Various statistics for hardware-generated dynamic DMAs
- **Usage**: Detailed breakdown of most efficient DMA type

---

### 6. Engine Utilization Metrics

#### `tensor_engine_active_time_percent`
- **What it measures**: Percentage of time TensorE is processing instructions (excluding semaphore waits)
- **Architectural context**: TensorE provides >90% of hardware FLOPs via 128×128 systolic array
  - Should be high (>90%) for matmul-heavy workloads
  - Idle gaps indicate waiting for data or other engines
- **Targets from perf guide**: >90% for compute-bound kernels
- **Optimization**: [Opt #3](../../neuron-nki-docs/references/optimization/nki_perf_guide.md#opt-3-overlap-execution-across-compute-engines-through-pipelining) (engine pipelining), [Opt #4](../../neuron-nki-docs/references/optimization/nki_perf_guide.md#opt-4-overlap-data-loading-with-computation) (DMA overlap)
- **Related metrics**: `mfu_estimated_percent` (considers instruction efficiency, not just active time)

#### `vector_engine_active_time_percent`
- **What it measures**: Percentage of time VectorE is processing instructions
- **Architectural context**: VectorE has 128 parallel lanes for element-wise ops, reductions, normalization
  - High for element-wise workloads (expected)
  - Can run in parallel with TensorE
- **Typical patterns**:
  - Matmul kernels: Low VectorE usage (<10%)
  - Element-wise kernels: High VectorE usage (>70%)
- **Optimization**: [Opt #5b](../../neuron-nki-docs/references/optimization/nki_perf_guide.md#opt-5b-use-sufficiently-large-input-tiles-in-partition-dimension) (partition vectorization for efficiency)

#### `scalar_engine_active_time_percent`
- **What it measures**: Percentage of time ScalarE is processing instructions
- **Architectural context**: ScalarE performs per-partition scalar operations (exponentiation, scaling, biasing)
  - Often pipelined with VectorE ([Opt #6](../../neuron-nki-docs/references/optimization/nki_perf_guide.md#opt-6-combine-instructions): combine instructions)
  - High usage may indicate loop overhead or inefficient instruction patterns
- **Optimization**: [Opt #5a](../../neuron-nki-docs/references/optimization/nki_perf_guide.md#opt-5a-use-sufficiently-large-input-tiles-in-free-dimension) (increase free dimension to amortize instruction overhead)

#### `gpsimd_engine_active_time_percent`
- **What it measures**: Percentage of time GpSimdE is processing custom SIMD instructions
- **Architectural context**: Used for specialized operations not mappable to other engines
- **Typical usage**: Low in most kernels unless using custom SIMD operations

#### `gpsimd_engine_instruction_count` / `gpsimd_engine_instruction_time`
- **What they measure**: Number and duration of GpSimd Engine instructions
- **Usage**: Understanding GpSimdE workload

---

### 7. Bandwidth Utilization Metrics

#### `mbu_estimated_percent` (Memory Bandwidth Utilization)
- **What it measures**: Achieved HBM bandwidth ÷ theoretical max HBM bandwidth
- **Formula**: (hbm_read_bytes + hbm_write_bytes) / total_time / max_hbm_bandwidth
- **Architectural context**: How efficiently DMA engines are using available HBM bandwidth
- **Targets from perf guide**: >60% for memory-bound kernels
- **Related metrics**: `dma_active_time_percent`, `hbm_read_bytes`, `hbm_write_bytes`

#### `mbu_min_read_util_percent`
- **What it measures**: Minimum possible MBU assuming inputs/weights read only once (no reloading)
- **Formula**: `inputs_and_weights_size_bytes` / total_time / max_hbm_bandwidth
- **Usage**: Lower bound for memory bandwidth needs
  - If `mbu_estimated` >> `mbu_min_read_util` → redundant loads or spilling
- **Optimization**: [Opt #1](../../neuron-nki-docs/references/optimization/nki_perf_guide.md#opt-1-exploit-temporal-locality-to-minimize-input-data-reloading) (minimize reloading), [Opt #2](../../neuron-nki-docs/references/optimization/nki_perf_guide.md#opt-2-fuse-operations-to-minimize-intermediate-data-spilling) (minimize spilling)

---

### 8. Collective Communication Metrics (Multi-NeuronCore)

These metrics are relevant for distributed training, tensor parallelism, and multi-chip workloads.

#### `cc_op_count` / `cc_op_time` / `cc_op_active_time` / `cc_op_active_time_percent`
- **What they measure**: Collective communication operations (AllReduce, AllGather, etc.) across NeuronCores
- **Architectural context**: Multi-chip communication operations
- **When relevant**: Distributed training, tensor parallelism, large model inference
- **Interpretation**: High percentage indicates communication-bound workload
- **Optimization**: Reduce cross-chip data movement, increase local computation

#### `cc_cores_instruction_count` / `cc_cores_instruction_time` / `cc_cores_instruction_active_time` / `cc_cores_instruction_active_time_percent`
- **What they measure**: Instructions and activity on all Collective Communication cores
- **Architectural context**: Dedicated cores handle multi-chip communication
- **Usage**: Understanding CC workload distribution and utilization

---

### 9. Specialized Metrics

#### `activate_instruction_count` / `activate_instruction_time`
- **What they measure**: Number and duration of ACTIVATE/ACTIVATE_QUANTIZE instructions
- **When relevant**: Quantization-heavy kernels
- **Architectural context**: Special activation functions with quantization support

#### `event_count`
- **What it measures**: Total number of event notifications (semaphore updates, status notifications)
- **Architectural context**: Events are synchronization points between engines and DMAs
- **Note**: `event_count` < `trace_count` (events don't include instructions)

#### `trace_count`
- **What it measures**: Total trace entries captured (events + instructions)
- **Usage**: Understanding profiling overhead and trace granularity
- **Related metrics**: `event_count` (subset)

---

### 10. System Information Metrics

#### `compiler_version`
- **What it measures**: Neuron Compiler version that built the NEFF (from info.json tool_version field)
- **Usage**: Version compatibility checking, reproducing results

#### `driver_version`
- **What it measures**: Neuron driver version used during profiling
- **Usage**: Version compatibility checking

#### `collectives_version`
- **What it measures**: Neuron Collectives library version used during profiling
- **Usage**: For multi-chip workloads

#### `instance_type`
- **What it measures**: AWS instance type (e.g., trn1.32xlarge, inf2.48xlarge)
- **Usage**: Hardware generation determination, comparing results across instances

---

## Hardware Generation Specifications

Understanding hardware limits helps interpret metric values and explains why certain optimizations are needed.

| Specification | gen2 (Trn1/Inf2) | gen3 (Trn2) | gen4 (Trn3) |
|---------------|------------------|-------------|-------------|
| **PSUM free dim limit** | 512 elements | 512 elements | **4096 elements** |
| **SBUF capacity** | 24 MB | 24 MB | 24 MB |
| **Max partition dim (P)** | 128 | 128 | 128 |
| **TensorE systolic array** | 128×128 | 128×128 | 128×128 |
| **DMA engines** | 16 | 16 | 16 |
| **Supported precisions** | BF16, FP16, FP32 | **+ FP8** | **+ MXFP8, MXFP4** |
| **Ideal DMA transfer size** | 32 KiB/engine | 32 KiB/engine | 32 KiB/engine |
| **Instruction overhead** | ~100 cycles | ~100 cycles | ~100 cycles |

**Key Differences:**
- **gen4 PSUM limit increase**: 8× larger (512 → 4096 elements) enables larger tile sizes without PSUM spilling
- **FP8 support (gen3+)**: Affects `hardware_flops` calculation, enables precision optimizations
- **MXFP8/MXFP4 (gen4)**: Microscaling formats for advanced quantization strategies

---

## Optimization Guidance

For comprehensive optimization strategies, metric-to-fix mappings, and detailed case studies, see [../../neuron-nki-optimizing/references/optimization-insights.md](../../neuron-nki-optimizing/references/optimization-insights.md).

**Quick optimization mapping:**
- **Low TensorE utilization** → [Opt #4, #9](../../neuron-nki-optimizing/references/optimization-insights.md#opt-4-overlap-data-loading-with-computation) (DMA overlap, sizing)
- **Spilling (sbuf_spill_bytes >0)** → [Opt #2](../../neuron-nki-optimizing/references/optimization-insights.md#opt-2-fuse-operations-minimize-spilling) (fusion, tiling)
- **High transpose ratio** → [Opt #8](../../neuron-nki-optimizing/references/optimization-insights.md#opt-8-mitigate-transpose-overhead) (layout choice)
- **Input reloading** → [Opt #1](../../neuron-nki-optimizing/references/optimization-insights.md#opt-1-temporal-locality) (load hoisting)

For the 10 core optimization strategies from the NKI Performance Guide, see [../../neuron-nki-optimizing/references/optimization-insights.md](../../neuron-nki-optimizing/references/optimization-insights.md#connecting-metrics-to-optimizations).

---

## Using This Reference

### For Initial Profiling

1. **Start with overall indicators**: `latency`, `mfu_estimated_percent`, `mbu_estimated_percent`
2. **Use Roofline metrics**: Compare `mm_arithmetic_intensity` to `peak_flops_bandwidth_ratio` to understand if workload should be compute or memory bound
3. **Drill into relevant category**:
   - Compute-bound → Check compute efficiency metrics (`hardware_flops`, `transpose_flops`, `hfu vs mfu`), engine utilization
   - Memory-bound → Check DMA metrics (`dma_active_time`, `dynamic_dma_percent`, `transfer_average_bytes`), memory hierarchy metrics (`hbm traffic`, `spills`)
   - Mixed or uncertain → Check for spilling, engine pipelining opportunities, examine timeline view

### For Optimization

See [../../neuron-nki-optimizing/references/optimization-insights.md](../../neuron-nki-optimizing/references/optimization-insights.md) for comprehensive optimization guidance including:
- Profiling-driven workflow and bottleneck identification
- Metric-to-optimization mapping (which metrics indicate which fixes)
- Tiling and blocking strategies with concrete examples
- Memory optimization techniques
- Case studies with real performance improvements

**Quick workflow:**
1. **Identify symptoms in metrics**: Use [../../neuron-nki-optimizing/references/optimization-insights.md - Profiling-Driven Workflow](../../neuron-nki-optimizing/references/optimization-insights.md#profiling-driven-optimization-workflow)
2. **Map metrics to optimizations**: See [../../neuron-nki-optimizing/references/optimization-insights.md - Connecting Metrics](../../neuron-nki-optimizing/references/optimization-insights.md#connecting-metrics-to-optimizations)
3. **Apply targeted optimization**: Follow [NKI Performance Guide Opt #1-10](../../neuron-nki-docs/references/optimization/nki_perf_guide.md)
4. **Re-profile and verify**: Check if targeted metrics improved

### For Debugging

1. **Use timeline view** in neuron-explorer GUI to visualize engine/DMA activities
2. **Inspect idle gaps and semaphore wait conditions**: What is each engine waiting for?
3. **Cross-reference with metrics to quantify issues**: How severe is the problem?
4. **Use architecture understanding to reason about root causes**: Why is this happening given the hardware constraints?
5. **Match to representative patterns**: See [../../neuron-nki-optimizing/references/optimization-insights.md - Representative Patterns](../../neuron-nki-optimizing/references/optimization-insights.md#representative-kernel-patterns)

---

## Representative Kernel Patterns

For detailed descriptions of expected metric patterns for common kernel types (well-optimized matmul, element-wise, spilling-induced degradation, DMA-bound, transpose-heavy, poor pipelining), see [../../neuron-nki-optimizing/references/optimization-insights.md - Representative Kernel Patterns](../../neuron-nki-optimizing/references/optimization-insights.md#representative-kernel-patterns).

**Quick pattern recognition:**
- **TensorE >90%, MFU >90%, no spills** → Well-optimized compute-bound kernel
- **VectorE >70%, MBU >60%, TensorE <10%** → Expected memory-bound element-wise
- **sbuf_spill_bytes >0, excess HBM traffic** → Spilling degradation
- **dynamic_dma_percent <60%, small transfers** → DMA inefficiency
- **transpose_flops/hardware_flops >15%** → Transpose overhead
- **All engines <60%** → Pipelining issues

---

## Advanced Topics

### Understanding Max Achievable Metrics
- `mfu_max_achievable_estimated_percent`: Roofline model ceiling
- Formula: (flops / hbm_bytes) / (max_ops / max_bandwidth)
- If actual `mfu_estimated_percent` approaches this value, you've hit the fundamental Roofline limit for your workload
- Further improvement requires increasing arithmetic intensity (algorithmic changes: fusion, reuse)

### Trace Collection Details
- `--enable-dge-notifs` flag: Enables DMA granular events in trace
- Impact: More detailed DMA breakdown in timeline
- When to use: Debugging DMA inefficiencies with [Opt #9](../../neuron-nki-docs/references/optimization/nki_perf_guide.md#opt-9-perform-sufficiently-large-dma-transfers), [Opt #10](../../neuron-nki-docs/references/optimization/nki_perf_guide.md#opt-10-minimize-use-of-dma-transposes)

### Event Count vs Trace Count
- `event_count`: Notifications (semaphores, status updates)
- `trace_count`: Total trace entries (events + instructions)
- High `event_count` / `trace_count` ratio: More synchronization overhead

---

## Summary

This reference provides comprehensive coverage of 50+ NKI profiling metrics with architectural context. Key takeaways:

1. **Understand the architecture**: Metrics reveal how your kernel interacts with NeuronCore components (TensorE, VectorE, ScalarE, DMA, SBUF, PSUM, HBM)
2. **Use Roofline Model**: `mm_arithmetic_intensity` vs `peak_flops_bandwidth_ratio` determines fundamental bottleneck type
3. **Check for spilling**: Non-zero `sbuf_spill_bytes` / `psum_spill_bytes` indicates SBUF (24 MB) or PSUM (512/4096 elements) capacity exceeded
4. **Know your hardware limits**: See [Hardware Generation Specifications](#hardware-generation-specifications)
5. **Cross-reference metrics**: Related metrics (e.g., `hfu` vs `mfu`, `spill_save` vs `sbuf_read`) provide deeper insights

**For optimization:** See [../../neuron-nki-optimizing/references/optimization-insights.md](../../neuron-nki-optimizing/references/optimization-insights.md) for:
- Profiling-driven optimization workflow
- Metric-to-fix mappings (Opt #1-10)
- Tiling and blocking strategies
- Representative kernel patterns
- Case studies with real performance improvements

**For detailed techniques:** See [NKI Performance Guide](../../neuron-nki-docs/references/optimization/nki_perf_guide.md) for Opt #1-10 implementation details.
