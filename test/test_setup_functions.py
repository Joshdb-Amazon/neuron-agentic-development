"""Tests for setup.py build helper functions."""
import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


def _import_setup():
    import importlib, sys
    orig_argv = sys.argv[:]
    sys.argv = ["setup.py", "egg_info"]
    try:
        sys.modules.pop("setup", None)
        sys.path.insert(0, str(Path(__file__).parent.parent))
        return importlib.import_module("setup")
    finally:
        sys.argv = orig_argv


_setup = _import_setup()


def _make_registry(tmp_path, subsets):
    registry = tmp_path / "tests" / "registry.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(json.dumps({"subsets": [{"name": s} for s in subsets]}))


def _make_result(tmp_path, agent, score, timestamp):
    # aim-build writes 13-digit millisecond Unix timestamps; match that width
    # so lexicographic filename sort matches chronological order.
    out_dir = tmp_path / "build" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"BenchmarkingOutput_{timestamp:013d}.json").write_text(json.dumps({
        "metadata": {"customAgentName": agent},
        "resultSummary": {"passAtKMetrics": {"kToResult": {"1": score}}},
    }))


def test_check_benchmark_results_raises_on_low_score(tmp_path):
    _make_registry(tmp_path, ["test-agent"])
    _make_result(tmp_path, "test-agent", 0.5, 1)
    with patch.object(_setup, "root", tmp_path):
        try:
            _setup._check_benchmark_results()
            assert False, "Should have raised"
        except SystemExit as e:
            assert "50.0%" in str(e)


def test_check_benchmark_results_raises_on_no_results(tmp_path):
    _make_registry(tmp_path, ["test-agent"])
    (tmp_path / "build" / "output").mkdir(parents=True)
    with patch.object(_setup, "root", tmp_path):
        try:
            _setup._check_benchmark_results()
            assert False, "Should have raised"
        except SystemExit as e:
            assert "no results" in str(e).lower()


def test_check_benchmark_results_catches_one_of_two_failing(tmp_path):
    _make_registry(tmp_path, ["agent-a", "agent-b"])
    _make_result(tmp_path, "agent-a", 0.8, 1)
    _make_result(tmp_path, "agent-b", 1.0, 2)
    with patch.object(_setup, "root", tmp_path):
        try:
            _setup._check_benchmark_results()
            assert False, "Should have raised"
        except SystemExit as e:
            assert "agent-a" in str(e)
            assert "80.0%" in str(e)


def test_check_benchmark_results_passes_when_all_pass(tmp_path):
    _make_registry(tmp_path, ["agent-a", "agent-b"])
    _make_result(tmp_path, "agent-a", 1.0, 1)
    _make_result(tmp_path, "agent-b", 1.0, 2)
    with patch.object(_setup, "root", tmp_path):
        _setup._check_benchmark_results()  # should not raise


def test_check_benchmark_results_fails_on_missing_subset(tmp_path):
    _make_registry(tmp_path, ["agent-a", "agent-b"])
    _make_result(tmp_path, "agent-a", 1.0, 1)
    # agent-b has no result file
    with patch.object(_setup, "root", tmp_path):
        try:
            _setup._check_benchmark_results()
            assert False, "Should have raised"
        except SystemExit as e:
            assert "agent-b" in str(e)


def test_check_benchmark_results_agent_filter_scopes_to_one(tmp_path):
    _make_registry(tmp_path, ["agent-a", "agent-b"])
    _make_result(tmp_path, "agent-a", 1.0, 1)
    # agent-b has no result — but we only check agent-a
    with patch.object(_setup, "root", tmp_path):
        _setup._check_benchmark_results(agent="agent-a")  # should not raise


def test_check_benchmark_results_uses_latest_per_agent(tmp_path):
    _make_registry(tmp_path, ["agent-a"])
    _make_result(tmp_path, "agent-a", 0.5, 1)  # old run, failed
    _make_result(tmp_path, "agent-a", 1.0, 2)  # new run, passed
    with patch.object(_setup, "root", tmp_path):
        _setup._check_benchmark_results()  # should not raise, latest is 1.0


def test_clean_removes_aim_dirs_preserves_private(tmp_path):
    build = tmp_path / "build"
    for d in ["agents", "skills", "private"]:
        (build / d).mkdir(parents=True)
    (build / "aim-build-test.json").write_text("{}")
    _setup._clean(tmp_path)
    assert not (build / "agents").exists()
    assert not (build / "aim-build-test.json").exists()
    assert (build / "private").exists()


def test_run_aim_validation_skips_when_missing(capsys):
    with patch.object(_setup.shutil, "which", return_value=None), \
         patch.object(_setup.subprocess, "run") as mock_run:
        _setup._run_aim_validation()
        mock_run.assert_not_called()
        assert "Skipping" in capsys.readouterr().out


def test_run_benchmarks_raises_on_failure():
    with patch.object(_setup.shutil, "which", return_value="/usr/bin/aim"), \
         patch.object(_setup, "_run_benchmark_test_setup", return_value=None) as mock_setup, \
         patch.object(_setup.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        try:
            _setup._run_benchmarks()
            assert False, "Should have raised"
        except SystemExit as e:
            assert "failed" in str(e).lower()
        # setup called twice: once to set up, once to clean up
        assert mock_setup.call_count == 2
        assert mock_setup.call_args_list[-1].kwargs == {"clean": True}


def test_run_benchmarks_cleans_up_on_setup_failure():
    calls = []
    def fake_setup(*args, **kwargs):
        calls.append(kwargs)
        if not kwargs.get("clean"):
            raise Exception("setup failed")
    with patch.object(_setup.shutil, "which", return_value="/usr/bin/aim"), \
         patch.object(_setup, "_run_benchmark_test_setup", side_effect=fake_setup), \
         patch.object(_setup.subprocess, "run") as mock_run:
        try:
            _setup._run_benchmarks()
            assert False, "Should have raised"
        except Exception as e:
            assert "setup failed" in str(e)
        mock_run.assert_not_called()
        assert calls[-1] == {"clean": True}  # cleanup still ran


def test_run_benchmarks_cleans_up_on_success():
    with patch.object(_setup.shutil, "which", return_value="/usr/bin/aim"), \
         patch.object(_setup, "_run_benchmark_test_setup", return_value=None) as mock_setup, \
         patch.object(_setup.subprocess, "run", return_value=MagicMock(returncode=0)), \
         patch.object(_setup, "_check_benchmark_results"):
        _setup._run_benchmarks()
        assert mock_setup.call_count == 2
        assert mock_setup.call_args_list[-1].kwargs == {"clean": True}


def _make_agent_spec(tmp_path, agent_name, skill_names):
    """Create a minimal agent-spec.json with skill dependencies."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / f"{agent_name}.agent-spec.json").write_text(json.dumps({
        "dependencies": {"skills": {"skillNames": skill_names}}
    }))


def _make_skill_setup(tmp_path, skill_name):
    """Create a skill's setup_dry_run.sh script."""
    scripts = tmp_path / "skills" / skill_name / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "setup_dry_run.sh").write_text("")


def test_run_benchmark_test_setup_clean_invokes_clean_flag(tmp_path):
    ran = []
    _make_agent_spec(tmp_path, "a1", ["skill-a"])
    _make_skill_setup(tmp_path, "skill-a")
    with patch.object(_setup, "root", tmp_path), \
         patch.object(_setup.subprocess, "run", side_effect=lambda cmd, **kw: ran.append(cmd)):
        _setup._run_benchmark_test_setup(agent="a1", clean=True)
    assert len(ran) == 1 and ran[0][-1] == "--clean"


def test_run_benchmark_test_setup_clean_ignores_nonzero_exit(tmp_path):
    """Cleanup uses check=False, so a non-zero exit does not raise."""
    _make_agent_spec(tmp_path, "a1", ["skill-a"])
    _make_skill_setup(tmp_path, "skill-a")
    with patch.object(_setup, "root", tmp_path), \
         patch.object(_setup.subprocess, "run", return_value=MagicMock(returncode=1)):
        _setup._run_benchmark_test_setup(agent="a1", clean=True)  # must not raise


def test_run_benchmark_test_setup_no_agent_runs_all_mapped(tmp_path):
    ran = []
    _make_agent_spec(tmp_path, "a1", ["skill-a"])
    _make_agent_spec(tmp_path, "a2", ["skill-b"])
    _make_skill_setup(tmp_path, "skill-a")
    _make_skill_setup(tmp_path, "skill-b")
    def _record(cmd, **kw):
        ran.append(cmd[-1])
        return MagicMock(returncode=0)
    with patch.object(_setup, "root", tmp_path), \
         patch.object(_setup.subprocess, "run", side_effect=_record):
        _setup._run_benchmark_test_setup()
    assert sorted(ran) == sorted([
        str(tmp_path / "skills" / "skill-a" / "scripts" / "setup_dry_run.sh"),
        str(tmp_path / "skills" / "skill-b" / "scripts" / "setup_dry_run.sh"),
    ])


def test_run_benchmark_test_setup_with_agent_filters(tmp_path):
    ran = []
    _make_agent_spec(tmp_path, "a1", ["skill-a"])
    _make_agent_spec(tmp_path, "a2", ["skill-b"])
    _make_skill_setup(tmp_path, "skill-a")
    _make_skill_setup(tmp_path, "skill-b")
    def _record(cmd, **kw):
        ran.append(cmd[-1])
        return MagicMock(returncode=0)
    with patch.object(_setup, "root", tmp_path), \
         patch.object(_setup.subprocess, "run", side_effect=_record):
        _setup._run_benchmark_test_setup(agent="a1")
    assert ran == [str(tmp_path / "skills" / "skill-a" / "scripts" / "setup_dry_run.sh")]


def test_run_benchmark_test_setup_unmapped_agent_is_noop(tmp_path):
    _make_agent_spec(tmp_path, "a1", ["skill-a"])
    with patch.object(_setup, "root", tmp_path), \
         patch.object(_setup.subprocess, "run") as mock_run:
        _setup._run_benchmark_test_setup(agent="unmapped-agent")
        mock_run.assert_not_called()


def test_run_benchmark_test_setup_missing_script_is_noop(tmp_path):
    _make_agent_spec(tmp_path, "a1", ["skill-a"])
    with patch.object(_setup, "root", tmp_path), \
         patch.object(_setup.subprocess, "run") as mock_run:
        _setup._run_benchmark_test_setup(agent="a1")  # skill dir exists but no setup_dry_run.sh
        mock_run.assert_not_called()


def test_run_benchmark_test_setup_failure_raises(tmp_path):
    """Setup failure (non-clean) raises SystemExit with descriptive message."""
    _make_agent_spec(tmp_path, "a1", ["skill-a"])
    _make_skill_setup(tmp_path, "skill-a")
    with patch.object(_setup, "root", tmp_path), \
         patch.object(_setup.subprocess, "run", return_value=MagicMock(returncode=1)):
        try:
            _setup._run_benchmark_test_setup(agent="a1")
            assert False, "Should have raised"
        except SystemExit as e:
            assert "a1" in str(e)


def test_check_benchmark_results_fails_on_missing_without_skip(tmp_path):
    """Without skip parameter, missing results for a registered agent raises."""
    _make_registry(tmp_path, ["agent-a", "agent-b"])
    _make_result(tmp_path, "agent-a", 1.0, 1)
    # agent-b has no result
    with patch.object(_setup, "root", tmp_path):
        try:
            _setup._check_benchmark_results()
            assert False, "Should have raised"
        except SystemExit as e:
            assert "agent-b" in str(e)


def test_check_benchmark_results_agent_filter_nonexistent_agent(tmp_path):
    _make_registry(tmp_path, ["agent-a"])
    _make_result(tmp_path, "agent-a", 1.0, 1)
    with patch.object(_setup, "root", tmp_path):
        try:
            _setup._check_benchmark_results(agent="nonexistent-agent")
            assert False, "Should have raised"
        except SystemExit as e:
            assert "nonexistent-agent" in str(e)
