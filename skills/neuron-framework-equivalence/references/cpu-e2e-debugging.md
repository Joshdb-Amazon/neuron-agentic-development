# CPU End-to-End Equivalence Debugging

After individual components pass their equivalence tests, this reference covers validating the full assembled model end-to-end on CPU. Five phases: TP=1 FP32 baseline, TP=1 BF16 3-tensor comparison, TP>1 validation, TP>1 debugging, and full-model final validation.

**Core principle:** CPU-mode testing (`NXD_CPU_MODE=1`, `gloo` backend) isolates software implementation differences from Neuron hardware effects. Fix all bugs here before moving to on-device validation.

---

## Phase 1: TP=1 FP32 Baseline

Verify the Neuron model's forward-pass logic is correct at FP32 with TP=1.

**Pass criterion:** `rel_fro_norm(HF_FP32, Neuron_FP32) < 1e-5`

**If this fails:** Do NOT proceed. Return to component debugging.

---

## Phase 2: TP=1 BF16 3-Tensor Comparison

**Pass criterion:** `error_ratio < 1.2`

| Tensor | Role |
|--------|------|
| HF FP32 | Ground truth |
| HF BF16 | Baseline dtype error |
| Neuron BF16 | Target |

```
error_ratio = rel_fro_norm(Neuron_BF16, HF_FP32) / rel_fro_norm(HF_BF16, HF_FP32)
```

---

## Phase 3: TP>1 Validation

**Pass criteria:**
- Fast test: `rel_fro(TP=1, TP=N) < 1e-2`
- Real weights: 3-tensor `error_ratio < 1.2` at each TP degree

### CRITICAL: mp.spawn and Monkey Patches

> `torch.multiprocessing.spawn()` ALWAYS uses `start_method="spawn"`, creating **brand new child processes** that do NOT inherit the parent's Python state. All monkey patches MUST be re-applied inside each worker function BEFORE model instantiation.

This was the **primary root cause** of TP>1 divergence for GPT-OSS 20B.

**Detection:** Capture a `_patched` flag in the diagnostic forward. If `True` in TP=1 but `False` in TP=N workers, patches aren't being re-applied.

### Weight Sharding Pipeline

```
1. get_state_dict(model_path, config)                        # Load HF weights + convert keys
2. fix_<layout>(model_sd, ...)                                # Fix weight layouts BEFORE sharding
3. get_sharded_checkpoint(model_sd, model, rank, tp_degree)   # Shard per rank
4. model.load_state_dict(model_sd, strict=False)              # Load sharded weights
5. Restore attention biases                                    # Removed by step 3
6. Restore per-expert biases (MoE models)                      # Lost in standard weight copy
```

**Step 2 is critical.** If paired ColumnParallelLinear and RowParallelLinear layers use different layouts, sharding gives each rank incompatible slices.

---

## Phase 4: TP>1 Debugging (When Phase 3 Fails)

### Fast Random-Weight Test

Use a tiny 1-layer model with random weights for 5-10 second iteration cycles.

**TINY_CONFIG design rules:**
- `num_key_value_heads` must be divisible by all TP degrees you test
- `num_local_experts` must be divisible by all TP degrees
- `num_hidden_layers=1` keeps it fast

### Common TP>1 Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| **Patches not applied in worker** | `patched_flag=0` in TP>1; all intermediates diverge | Add `apply_all_patches()` as first action in `_tp_worker` |
| **Weight layout mismatch** | MoE output FAIL; all later stages FAIL | `fix_<layout>()` before `get_sharded_checkpoint` |
| **Biases removed by sharding** | Attention output diverges; biases are zero | Manual bias restoration with TP-aware slicing |
| **CONVERT_TO_MHA** | KV bias shape mismatch when `tp_degree % kv_heads != 0` | Replicate bias via `repeat_interleave` then shard |
| **Parameter not TP-sharded** | Full-size parameter vs local head count → shape error | Manual TP slicing: `param[rank*local:(rank+1)*local]` |
| **Missing all-reduce** | MoE output is 1/N of correct value | Add `torch.distributed.all_reduce()` after partial computation |
| **Port conflict** | `EADDRINUSE` error | Use different `MASTER_PORT` for TP=1 (8080) and TP>1 (29501) |

### Bias Restoration (TP-Aware)

`get_sharded_checkpoint` removes biases it considers "redundant". Three cases for restoration:

| Bias size vs local heads | Action |
|-------------------------|--------|
| Equal | Use as-is |
| Greater (full-size bias) | Chunk and shard: `bias.chunk(tp_degree)[rank]` |
| Less (CONVERT_TO_MHA) | Replicate via `repeat_interleave(repeats)` then chunk-shard |

---

## Phase 5: Full Model Validation

| Test | Metric | Threshold |
|------|--------|-----------|
| FP32 Direct TP=1 | rel_fro_norm | < 1e-5 |
| 3-Tensor BF16 TP=1 | error_ratio | < 1.2 |
| 3-Tensor BF16 TP=4 | error_ratio | < 1.2 |
| 3-Tensor BF16 TP=8 | error_ratio | < 1.2 |
| Token coherence | All same next token | Match HF FP32 |

### Key Insight from GPT-OSS

The dequantization script had a bug producing incorrect weights for ALL configurations. Both models matched each other but produced gibberish. Always verify output token coherence — numerical equivalence is meaningless if both models are wrong.

---

Based on: GPT-OSS 20B CPU E2E debugging (February-March 2026)
