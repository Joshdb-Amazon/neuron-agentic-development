# SwiGLU MLP Kernel Optimization Findings

## Baseline Analysis

**Kernel**: SwiGLU MLP — `output = down_proj(silu(gate_proj(x)) * up_proj(x))`
**Shape**: tokens=4096, input_size=4096, hidden_size=8192, dtype=float16
**Hardware**: trn2 (gen3)

**Baseline metrics** (36.704 ms):
- DMA-bound: DMA active 93.25%, TensorE 80.48%
- Massive weight reloading: HBM reads 7143 MB vs inputs+weights 235 MB = **30.4x reload factor**
- MFU only 28.57% despite high TensorE utilization
- 81,920 matmul instructions

**Primary bottleneck**: Weight tiles are reloaded for every token tile. With 32 token tiles, each weight tile is loaded 32 times. The 3 weight matrices (gate, up, down) dominate HBM traffic at ~6 GB.

---

## V1: Hoist x Tile Loading Out of h_idx Loop

**Bottleneck targeted**: Redundant x tile loads in Phase 1 (profiling evidence: 30.4x reload factor)

**Code change**: Pre-load and transpose all x tiles before the hidden dimension loop (h_idx), instead of reloading per (h_idx, k_idx). Since x doesn't depend on h_idx, this eliminates 15/16 = 93.75% of x loads in Phase 1.

| Metric | Baseline | v1 | Change |
|--------|----------|-----|--------|
| total_time (ms) | 36.704 | 33.558 | -8.6% |
| HBM read (MB) | 7143 | 6908 | -3.3% |
| matmul_instructions | 81920 | 66560 | -18.8% |
| transpose_flops (GF) | 68.72 | 36.51 | -46.9% |

**Result**: Improved but below 10% target. The x tile loading was a minor contributor; weight reloading dominates.

---

## V2: Fuse Phase 1 and Phase 2 to Eliminate Hidden HBM Roundtrip

**Bottleneck targeted**: Hidden activation HBM roundtrip (profiling evidence: 101 MB HBM writes for hidden)

**Code change**: Instead of the two-phase approach (compute all hidden → write to HBM → read back for down projection), fuse the phases. For each hidden tile, immediately compute the down projection and accumulate into output SBUF accumulators. This eliminates the `hidden_hbm` intermediate entirely.

| Metric | v1 | v2 | Change |
|--------|-----|-----|--------|
| total_time (ms) | 33.558 | 30.357 | -9.5% |
| HBM write (MB) | 101 | 34 | -66.3% |
| HBM read (MB) | 6908 | 6476 | -6.3% |
| transpose_flops (GF) | 36.51 | 6.44 | -82.4% |

**Result**: 17.3% improvement over baseline. Eliminated hidden HBM roundtrip and dramatically reduced transpose overhead. DMA still saturated at 99.27% due to weight reloading.

---

## V3: h_idx Outer Loop for Weight Reuse (FAILED)

**Bottleneck targeted**: Weight reloading across token tiles (profiling evidence: 27.57x reload factor in v2)

**Code change**: Restructured loop order to put hidden dimension (h_idx) as the OUTER loop and token tiles (t_idx) as INNER. This loads each weight tile once per h_idx instead of once per (t_idx, h_idx). Zero-initialized output in HBM, then read-modify-write for each h_idx contribution.

| Metric | v2 | v3 | Change |
|--------|-----|-----|--------|
| total_time (ms) | 30.357 | 32.050 | **+5.6% (regressed)** |
| HBM read (MB) | 6476 | 1275 | -80.3% |
| HBM write (MB) | 34 | 571 | +1600% |
| DMA active (%) | 99.27 | 42.86 | -56.4pp |
| matmul_instructions | 52224 | 81920 | +56.9% |

**Result**: REGRESSED. While HBM reads dropped 5x, the output read-modify-write overhead (570 MB writes), x-tile reloading (16x instead of 1x), and doubled matmul instruction count negated the savings. The total time increased.

**Lesson**: Reducing HBM traffic is necessary but not sufficient. The overhead of HBM-based accumulation (read-add-write per h_idx) and loss of x-tile caching outweighed the weight loading savings.

---

## V4: Batch Token Tiles for Weight Reuse (BATCH_T=2)

**Bottleneck targeted**: Weight reloading across token tiles (profiling evidence: v2 reload factor 27.57x)

**Code change**: Hybrid approach combining v2's SBUF accumulation with v3's weight caching. Process token tiles in **batches of 2**: load x_t tiles for both into SBUF (cached across all h_idx), initialize SBUF accumulators for both, then for each h_idx load weights once and compute for both token tiles. Output is written to HBM only at the end of the batch.

**Key insight**: With BATCH_T=2, weights are loaded once per batch (16 batches) instead of once per token tile (32 tiles), halving weight traffic. x_t tiles stay cached in SBUF. Output accumulates in float32 SBUF with no HBM R/W per h_idx.

**SBUF budget**: x_t(2MB) + accum(4MB) + weights(12MB) + work(3MB) = 21MB < 24MB

| Metric | v2 | v4 | Change |
|--------|-----|-----|--------|
| total_time (ms) | 30.357 | 20.434 | -32.7% |
| HBM read (MB) | 6476 | 3255 | -49.7% |
| DMA active (%) | 99.27 | 79.06 | -20.2pp |
| MFU (%) | 34.54 | 51.32 | +16.8pp |
| reload_factor | 27.57x | 13.86x | -49.8% |

**Result**: **44.3% improvement over baseline**. Halved HBM reads, shifted DMA from saturated to moderate utilization, nearly doubled MFU.

---

## V5: Larger Batch Size (BATCH_T=4)

**Bottleneck targeted**: Remaining weight reloading (profiling evidence: v4 reload factor 13.86x)

**Code change**: Increased BATCH_T from 2 to 4. This reduces weight loading by 4x vs v2 (8 batches × 192MB vs 32 × 192MB).

**SBUF budget**: x_t(4MB) + accum(8MB) + weights(12MB) + work(1MB) = 25MB. Tight but no spilling observed.

| Metric | v4 | v5 | Change |
|--------|-----|-----|--------|
| total_time (ms) | 20.434 | 18.711 | -8.4% |
| HBM read (MB) | 3255 | 1644 | -49.5% |
| DMA active (%) | 79.06 | 44.43 | -34.6pp |
| MFU (%) | 51.32 | 56.04 | +4.7pp |
| arithmetic_intensity | 250.78 | 491.52 | +96.0% |
| reload_factor | 13.86x | 7.00x | -49.5% |
| sbuf_spill_bytes | 0 | 0 | — |

**Result**: **49.0% improvement over baseline**. DMA now at 44%, no longer the bottleneck. Kernel has transitioned from heavily DMA-bound to a more balanced profile with room for further compute optimization.

---

## Summary

| Version | total_time (ms) | vs Baseline | Key Optimization |
|---------|-----------------|-------------|------------------|
| Baseline | 36.704 | — | — |
| v1 | 33.558 | -8.6% | x tile hoisting |
| v2 | 30.357 | -17.3% | Phase fusion (eliminate hidden HBM) |
| v3 | 32.050 | -12.7% | h_idx outer (REGRESSED from v2) |
| v4 | 20.434 | -44.3% | Batch token tiles (BATCH_T=2) |
| **v5** | **18.711** | **-49.0%** | **Larger batch (BATCH_T=4)** |

**Best result: v5 at 18.711 ms (49.0% faster than baseline)**

The dominant optimization was reducing weight reloading through token tile batching (v4/v5). The fused phase approach (v2) provided the foundation by eliminating intermediate HBM traffic. The failed v3 attempt revealed that naively swapping loop order without maintaining SBUF accumulation can regress performance despite lower HBM traffic.
