"""Tests for deploy.py functions."""
import json
import tempfile
from pathlib import Path

from neuron_agentic_development.deploy import (
    _copy_artifacts, _deploy, _find_artifact_src, _transform_agent_spec,
)


def _make_aim_spec(path, prompt="{{aim:include:agents/test.md}}", resources=None):
    spec = {
        "schemaVersion": "1", "name": "test-agent",
        "config": {"description": "test", "systemPrompt": prompt, "model": "opus"},
        "clientConfig": {"kiroCli": {"allowedTools": ["fs_read"]}},
    }
    if resources:
        spec["clientConfig"]["kiroCli"]["resources"] = resources
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(spec, f)


def test_transform_resolves_prompt_and_resources():
    with tempfile.TemporaryDirectory() as td:
        dest = Path(td).resolve()
        (dest / "agents").mkdir()
        (dest / "agents" / "test.md").write_text("prompt")
        (dest / "skills").mkdir()
        (dest / "skills" / "ref.md").write_text("ref")
        spec = dest / "agents" / "test.agent-spec.json"
        _make_aim_spec(spec, resources=["file://skills/ref.md"])
        _transform_agent_spec(spec, spec, dest)
        data = json.loads(spec.read_text())
        assert data["prompt"] == f"file://{dest}/agents/test.md"
        assert data["resources"] == [f"file://{dest}/skills/ref.md"]


def test_transform_warns_on_missing_resource(capsys):
    with tempfile.TemporaryDirectory() as td:
        dest = Path(td)
        (dest / "agents").mkdir()
        spec = dest / "agents" / "test.agent-spec.json"
        _make_aim_spec(spec, resources=["file://missing/file.md"])
        _transform_agent_spec(spec, spec, dest)
        assert "WARN: resource not found" in capsys.readouterr().err


def test_copy_artifacts_replaces_existing_dest():
    with tempfile.TemporaryDirectory() as td:
        src, dest = Path(td) / "src", Path(td) / "dest"
        (src / "agents").mkdir(parents=True)
        (src / "agents" / "new.json").write_text("{}")
        (dest / "agents").mkdir(parents=True)
        (dest / "agents" / "old.json").write_text("{}")
        _copy_artifacts(src, dest)
        assert (dest / "agents" / "new.json").exists()
        assert not (dest / "agents" / "old.json").exists()


def test_deploy_resolves_all_resources():
    with tempfile.TemporaryDirectory() as td:
        dest = Path(td) / ".kiro"
        assert _deploy("kiro", dest=dest) == 0
        for spec in (dest / "agents").glob("*.agent-spec.json"):
            data = json.loads(spec.read_text())
            for r in data.get("resources", []):
                assert r.startswith("file:///"), f"Unresolved resource: {r}"


def test_deploy_returns_1_when_artifacts_missing():
    with tempfile.TemporaryDirectory() as td:
        import neuron_agentic_development.deploy as mod
        orig = mod.__file__
        # Point __file__ at a location with no artifacts/ and no repo-root markers
        mod.__file__ = str(Path(td) / "src" / "pkg" / "deploy.py")
        try:
            assert _deploy("kiro", dest=Path(td) / ".kiro") == 1
        finally:
            mod.__file__ = orig


def test_find_artifact_src_falls_back_to_repo_root():
    """When artifacts/ doesn't exist, _find_artifact_src returns repo root."""
    with tempfile.TemporaryDirectory() as td:
        import neuron_agentic_development.deploy as mod
        # Simulate repo layout: <root>/src/neuron_agentic_development/deploy.py
        repo = Path(td) / "repo"
        pkg = repo / "src" / "neuron_agentic_development"
        pkg.mkdir(parents=True)
        (repo / "agents").mkdir()
        (repo / "skills").mkdir()
        orig = mod.__file__
        mod.__file__ = str(pkg / "deploy.py")
        try:
            result = _find_artifact_src()
            assert result == repo
        finally:
            mod.__file__ = orig
