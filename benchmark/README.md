# NAD Skill Benchmark Gate

A CI gate that evaluates this repo's agent skills by running them through the
`kiro-cli` agent in Docker and scoring the output against per-scenario rubrics.
A PR that changes `skills/`, `agents/`, `tests/`, or this `benchmark/` grader is
scored against the repo's own assets.

## Layout (single-repo)

Everything the gate needs lives in this repo:

- **Inputs** (at the repo root, scored by the gate):
  - `skills/` — the skills under test
  - `agents/` — agent specs (`*.agent-spec.json`) listing each agent's `skillNames`
  - `tests/` — scenario definitions per subset + `tests/registry.json` (the source
    of truth for which scenarios are *active/gated*)
- **Grader** (this `benchmark/` directory):
  - `run_skill_tests.py` — loads scenarios, runs the agent in Docker, scores, gates
  - `shell/{docker,common}.sh` — build the image + run `kiro-cli` in Docker
  - `environment/Dockerfile` — the run image (`python:3.11-slim` + kiro-cli)

## How it runs

`.github/workflows/benchmark.yml` checks the repo out and runs
`benchmark/run_skill_tests.py --nad-root=.`, so the grader scores the repo's own
`skills/` + `agents/` + `tests/`. No cross-repo references or tokens are needed.

- **Trigger:** `pull_request` touching `skills/**`, `agents/**`, `tests/**`,
  `benchmark/**`, or the workflow file; plus manual `workflow_dispatch`
  (inputs: `subset`, `count`, `gate`).
- **Requires:** a `KIRO_API_KEY` Actions secret on the repo (authenticates both the
  agent and the LLM judge). Fork PRs from external contributors do not receive
  secrets, so gating those requires a maintainer-triggered run.

## Scoring

Per scenario:

- `pattern_score` (75%) — proportional match of `expectedStrings` (present) and
  `forbiddenStrings` (absent), matched against the output file + the agent's response.
- `judge_score` (25%) — an LLM judge (`kiro-cli`) verdict against the scenario's
  rubric: `1.0` pass / `0.0` fail (or `null` if no rubric).
- `combined_score = 0.75*pattern + 0.25*judge` (pattern-only if the judge is off).
- `passed = combined_score >= 0.90` (per-scenario override via `metadata.baseline_k`).

The gate fails if the overall pass rate is below `--gate` (default 90%).

> **Note on the judge:** the LLM judge is non-deterministic, so a scenario with a
> perfect `pattern_score=1.00` can still score `judge=0` on an unlucky verdict,
> pulling `combined` to 0.75. The overall pass rate can therefore vary run-to-run
> near the threshold independent of any code change.

## Reliability handling

- **Infra-error retry:** a scenario that hits an infrastructure error (Docker/agent
  timeout, etc.) is retried up to 2 more times before counting. Genuine skill
  failures are never retried.
- **Salvage-on-timeout:** if the agent times out but has already written a non-empty
  `output.py`, that output is scored instead of being discarded (the agent often
  finishes the task and then hangs on a trivial final step). Only a timeout that
  produced no output counts as an infrastructure error.

## Running it

```bash
# In CI: Actions -> NAD Benchmark -> Run workflow (workflow_dispatch), or on a PR.

# Manually via CLI:
gh workflow run benchmark.yml -f subset=all -f count=1 -f gate=90

# Locally (needs Docker + KIRO_API_KEY):
python benchmark/run_skill_tests.py --nad-root=. --subset=all --count=1 --judge --gate=90
```

`--subset` accepts `all`, a single subset name (e.g. `neuron-nki-agent`), or a
comma-separated list. Results are written to `benchmark/build/nad_benchmark_results.json`.
