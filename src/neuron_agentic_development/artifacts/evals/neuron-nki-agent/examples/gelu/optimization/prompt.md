# GELU — Optimization

You are optimizing the `gelu` kernel through 5 versioned improvements.

- **Baseline kernel**: `evals/neuron-nki-agent/examples/gelu/output/gelu_nki_kernels.py`
- **Baseline test**: `evals/neuron-nki-agent/examples/gelu/output/test_gelu.py`
- **Working directory**: `evals/neuron-nki-agent/examples/gelu/optimization/tmp/`
- **Final output**: `evals/neuron-nki-agent/examples/gelu/optimization/output/`

The kernel is correct and compiles. Your job is to produce 5 progressively
optimized versions, each targeting at least a 10% improvement over the
previous version on total_time. You must achieve at least one successful
improvement across the 5 versions.

## Core Assignment

You are assigned neuron cores **8–11** (4 cores).
Set `NEURON_RT_VISIBLE_CORES=8-11` in every compile and run command.
Do not use cores outside this range.

## Skills

You have 5 skills at your disposal:

- `/neuron-nki-writing` — NKI kernel authoring patterns, API usage, buffer
  management, tiling strategies. Consult when writing or restructuring kernel code.
- `/neuron-nki-debugging` — Diagnosing compilation failures, runtime errors,
  correctness issues. Consult when a version breaks.
- `/neuron-nki-optimizing` — Optimization strategies: blocking, double-buffering,
  engine overlap, DMA coalescing, SBUF reuse, loop restructuring. Consult when
  designing each optimization.
- `/neuron-nki-profiling` — Capturing profiles (env vars, neuron-explorer capture).
  Consult when you need to capture a new profile.
- `/neuron-nki-profile-querying` — Querying the neuron-explorer HTTP API for
  metrics (Summary, Instruction, DmaPacketAggregated tables). Consult when you
  need evidence to identify bottlenecks or measure improvement.

The workflow is: use `/neuron-nki-optimizing` and `/neuron-nki-writing` to reason
about and implement improvements, `/neuron-nki-debugging` when things break, and
`/neuron-nki-profiling` + `/neuron-nki-profile-querying` to measure before/after
and validate your gains. Profiling supports the optimization loop — it is not
the loop.

## CRITICAL: Shared neuron-explorer server

A single `neuron-explorer` server is running and shared across all 8 kernel
sessions. **DO NOT kill, restart, or stop the neuron-explorer process.**

Only use:
- `neuron-explorer capture` to capture profiles
- `curl` to query the HTTP API

If you encounter a port conflict or connection error, wait and retry — do NOT
attempt to restart the server.

## Optimization Loop (repeat 5 times)

For each version v1 through v5:

1. **Profile the current version** using `/neuron-nki-profiling` to capture, then
   `/neuron-nki-profile-querying` to extract key metrics (total_time, engine
   utilization, HBM bytes, MFU, spill). This is your baseline for this iteration.
2. **Identify the top bottleneck** — use `/neuron-nki-profile-querying` evidence to
   pinpoint what's slow. Don't guess.
3. **Design the optimization** — consult `/neuron-nki-optimizing` for strategies that
   address the identified bottleneck. Use `/neuron-nki-writing` for implementation
   patterns.
4. **Implement** as `gelu_v<N>.py` in the working directory.
5. **Verify correctness** — run the test against both the baseline and the new
   version, comparing against the PyTorch reference.
6. **Profile the new version** — capture and query. Record the before/after
   delta. Each version must target at least **10% reduction in total_time**
   over the previous version.
7. **If it broke or regressed**:
   - Use `/neuron-nki-debugging` to diagnose.
   - Document what you tried, why it failed, and what you learned.
   - Revert to the last working version and try a different optimization
     direction. This still counts as a version attempt.
   - Failed versions are valuable data — record them in FINDINGS.md.

**Minimum bar**: at least one of the 5 versions must achieve a measurable
improvement. If after 5 honest attempts none improve, document why the kernel
is near-optimal with profiling evidence.

## Output Structure

Only write to `evals/neuron-nki-agent/examples/gelu/optimization/output/` when completely done:

- `FINDINGS.md` — running optimization journal, one section per version:
  - What bottleneck was targeted and the profiling evidence
  - What code change was made
  - Before/after metrics table
  - Whether it succeeded or failed and why
- `gelu_v5.py` — your best optimized kernel (or latest working version)
- `test_gelu.py` — correctness test covering baseline and optimized
- `output.md` — single summary table showing v1->v5 progression across key
  metrics (total_time, engine utilization %, HBM read/write bytes, MFU, spill)

## Environment

- Activate the venv: `source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate`
- default_nc_version is gen3
- `NEURON_RT_VISIBLE_CORES=8-11` on every compile/run
- Use `evals/neuron-nki-agent/examples/gelu/optimization/tmp/` for ALL intermediate work
- Only write final deliverables to `evals/neuron-nki-agent/examples/gelu/optimization/output/`
- Do NOT modify the original kernel or test files in `evals/neuron-nki-agent/examples/gelu/output/`

## Tool usage — avoid unnecessary Bash

Use your built-in tools instead of Bash commands wherever possible.

- **Read files**: use the `Read` tool, NOT `cat`, `head`, `tail`
- **Search files**: use `Glob` / `Grep` — NOT `find`, `ls`, `grep`
- **Write/edit files**: use `Write` / `Edit` — NOT `sed`, `awk`, `echo >`

Reserve Bash ONLY for:
- `neuron-explorer capture` / `curl` (profiling)
- `python3` (running kernels and tests)
- `source` (activating the venv)
- `rm -rf` / `mkdir` (managing tmp artifacts)

## What success looks like

- 5 versioned optimization attempts, each grounded in `/neuron-nki-optimizing`
  strategy and `/neuron-nki-profile-querying` evidence
- At least one version achieves >=10% total_time reduction
- Failed attempts documented honestly, not hidden
- Correctness preserved across all working versions
- FINDINGS.md reads as a progression, not a single shot

## What failure looks like

- Skipping profiling and guessing at optimizations
- A single optimization attempt with no iteration
- Silently reverting failures without documenting them
- No measurable improvement and no evidence explaining why
- Killing or restarting the shared neuron-explorer server
