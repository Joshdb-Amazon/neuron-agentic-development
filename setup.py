"""
NeuronAgenticDevelopment setup.py

Build pipeline (brazil-build release):
  1. aim-build release    — validate AIM capabilities, populate build/
  2. aim-build benchmark  — run agent scenarios, fail if Pass@1 < 100%
  3. setup()              — build wheel with bundled artifacts + run pytest
  4. cleanup              — remove ephemeral artifacts on clean

AIM integration follows:
  https://docs.hub.amazon.dev/aim/user-guide/build-system-integration/
"""
import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path
from setuptools import setup, Command

root = Path(__file__).parent
pkg_artifacts = root / "src" / "neuron_agentic_development" / "artifacts"

# Build targets that require AIM integration
_AIM_TARGETS = ("release", "build", "build_py", "bdist_wheel", "benchmark", "brazil")
# Build targets that should run benchmarks
_BENCHMARK_TARGETS = ("release", "build", "bdist_wheel")
# AIM-produced directories to clean (preserve BrazilPython's private/)
_CLEAN_DIRS = ("agents", "agent-sops", "context", "skills", "output", "dist", "roda", "pip")
# Artifact directories to bundle into the wheel
_ARTIFACT_DIRS = ("agent-sops", "agents", "context", "evals", "hooks", "skills", "tools")

def _get_benchmark_setup_skills(agent_name):
    """Read an agent spec and return skill names that have a setup_dry_run.sh script."""
    spec = root / "agents" / f"{agent_name}.agent-spec.json"
    if not spec.exists():
        return []
    with open(spec) as f:
        data = json.load(f)
    skill_names = data.get("dependencies", {}).get("skills", {}).get("skillNames", [])
    return [s for s in skill_names if (root / "skills" / s / "scripts" / "setup_dry_run.sh").exists()]


def _run_benchmark_test_setup(agent=None, clean=False):
    """Run setup_dry_run.sh (or with --clean) for each skill mapped to the target agent(s)."""
    if agent:
        agents = [agent]
    else:
        agents = [p.stem.replace(".agent-spec", "") for p in root.glob("agents/*.agent-spec.json")]
    for a in agents:
        for skill in _get_benchmark_setup_skills(a):
            script = root / "skills" / skill / "scripts" / "setup_dry_run.sh"
            cmd = ["bash", str(script)] + (["--clean"] if clean else [])
            if clean:
                print(f"Cleaning up test resources for agent '{a}' (skill: {skill})...", flush=True)
                subprocess.run(cmd, check=False)
            else:
                print(f"Setting up test resources for agent '{a}' (skill: {skill})...", flush=True)
                result = subprocess.run(cmd, check=False)
                if result.returncode != 0:
                    raise SystemExit(f"Test setup failed for agent '{a}' (skill: {skill})")


def _check_benchmark_results(agent=None):
    """Read benchmark results and raise if any registered subset has Pass@1 < 100%."""
    registry_path = root / "tests" / "registry.json"
    with open(registry_path) as f:
        registry = json.load(f)
    expected = {s["name"] for s in registry.get("subsets", [])}
    if agent:
        if agent not in expected:
            raise SystemExit(f"Agent {agent} not found in registry")
        expected = {agent}

    results = sorted(glob.glob(str(root / "build" / "output" / "BenchmarkingOutput_*.json")))
    if not results:
        raise SystemExit("aim-build benchmark produced no results file")

    # Find latest result file per registered agent
    latest = {}
    for result_file in results:
        with open(result_file) as f:
            data = json.load(f)
        name = data.get("metadata", {}).get("customAgentName", "")
        if name in expected:
            latest[name] = data

    missing = expected - latest.keys()
    if missing:
        raise SystemExit(f"No benchmark results for: {', '.join(sorted(missing))}")

    failures = []
    for name, data in latest.items():
        score = data.get("resultSummary", {}).get("passAtKMetrics", {}).get("kToResult", {}).get("1", 0)
        if score < 1.0:
            failures.append(f"{name}: Pass@1 = {score:.1%}")
    if failures:
        raise SystemExit("Benchmark failures:\n  " + "\n  ".join(failures))


def _clean(build_root):
    """Remove AIM artifacts and egg-info."""
    build_dir = build_root / "build"
    if build_dir.is_symlink():
        build_dir = build_dir.resolve()
    if build_dir.is_dir():
        for name in _CLEAN_DIRS:
            p = build_dir / name
            if p.is_dir():
                shutil.rmtree(p)
        for f in build_dir.glob("aim-build-*.json"):
            f.unlink()
    egg = build_root / "src" / "neuron_agentic_development.egg-info"
    if egg.exists():
        shutil.rmtree(egg)


def _run_aim_validation():
    """Run aim-build release; skip if aim-build not on PATH."""
    if shutil.which("aim-build"):
        result = subprocess.run(["aim-build", "release"], check=False)
        if result.returncode != 0:
            raise SystemExit("aim-build release failed")
    else:
        print("WARN: Skipping AIM validation (aim-build not on PATH)")


def _run_benchmarks():
    """Run aim-build benchmark; skip if aim/kiro-cli not on PATH."""
    if shutil.which("aim-build") and shutil.which("aim") and shutil.which("kiro-cli"):
        setup_failed = set()
        try:
            _run_benchmark_test_setup()
            result = subprocess.run(["aim-build", "benchmark"], check=False)
            if result.returncode != 0:
                raise SystemExit("aim-build benchmark failed")
            _check_benchmark_results()
        finally:
            _run_benchmark_test_setup(clean=True)
    else:
        print("WARN: Skipping benchmarks (aim or kiro-cli not on PATH)")


def _bundle_artifacts():
    """Copy AIM artifacts into the Python package for wheel inclusion."""
    pkg_artifacts.mkdir(exist_ok=True)
    for d in _ARTIFACT_DIRS:
        src = root / "build" / d
        if not src.is_dir():
            src = root / d
        dst = pkg_artifacts / d
        if src.is_dir() and not dst.exists():
            shutil.copytree(src, dst)


# --- Module-level build orchestration ---
target = sys.argv[1] if len(sys.argv) > 1 else ""
is_clean = "clean" in sys.argv

if is_clean:
    _clean(root)
elif target in _AIM_TARGETS:
    _run_aim_validation()
    if target in _BENCHMARK_TARGETS:
        _run_benchmarks()
    _bundle_artifacts()


class Benchmark(Command):
    """Custom setuptools command: brazil-build benchmark."""
    description = 'run AIM agent benchmark tests from tests/registry.json'
    user_options = [
        ('agent=', None, 'run benchmarks only for this agent'),
        ('existing', None, 'use existing installed agent'),
        ('output=', None, 'copy results to this directory'),
        ('debug', None, 'enable debug tracing'),
    ]

    def initialize_options(self):
        self.agent = None
        self.existing = False
        self.output = None
        self.debug = False

    def finalize_options(self):
        pass

    def run(self):
        try:
            _run_benchmark_test_setup(agent=self.agent)
            cmd = ["aim-build", "benchmark"]
            if self.agent:
                cmd.append(self.agent)
            if self.existing:
                cmd.append("--existing")
            if self.output:
                cmd.extend(["--output", self.output])
            if self.debug:
                cmd.append("--debug")
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                raise Exception("Benchmark failed")
            _check_benchmark_results(agent=self.agent)
        finally:
            _run_benchmark_test_setup(agent=self.agent, clean=True)


def _publish_wheel():
    """Copy wheel from private/dist/ to pip/public/neuron-agentic-development/."""
    private_dist = root / "build" / "private" / "dist"
    if private_dist.is_symlink():
        private_dist = private_dist.resolve()
    wheels = list(private_dist.glob("*.whl")) if private_dist.is_dir() else []
    if not wheels:
        return
    public_dir = root / "build" / "pip" / "public" / "neuron-agentic-development"
    public_dir.mkdir(parents=True, exist_ok=True)
    for whl in wheels:
        shutil.copy2(whl, public_dir / whl.name)
        print(f"Published {whl.name} -> {public_dir}")


try:
    setup(
        cmdclass={
            'benchmark': Benchmark,
        }
    )
    if target in _AIM_TARGETS and not is_clean:
        _publish_wheel()
finally:
    # Ephemeral artifacts cleanup — only during clean.
    # During release/test, pytest needs the artifacts to remain in src/ for test_deploy.
    if is_clean and pkg_artifacts.exists():
        shutil.rmtree(pkg_artifacts)
