"""Deploy agentic artifacts to ~/.kiro or ~/.claude.

Handles the AIM-to-kiro-cli agent spec format transform:
  - Repo stores AIM format (required by aim-build validation/benchmarks)
  - kiro-cli requires native format (name, prompt, model, allowedTools)
  - This module transforms during deployment so both systems are satisfied
"""
import json
import os
import shutil
import sys
from pathlib import Path

# Artifact directories deployed from the wheel to the target
ARTIFACT_DIRS = ["agent-sops", "agents", "context", "evals", "hooks", "skills", "tools"]


def _transform_agent_spec(src_path, dst_path, dest):
    """Transform an AIM-format agent spec to kiro-cli native format.

    AIM format:  schemaVersion, config.{description,systemPrompt,model},
                 dependencies, clientConfig.kiroCli.allowedTools
    kiro-cli:    name, description, prompt, model, allowedTools

    Resolves {{aim:include:path}} prompts and file:// resource paths to
    absolute file:// URIs relative to dest.
    """
    with open(src_path) as f:
        aim = json.load(f)
    # If not AIM format, copy as-is
    if "config" not in aim:
        shutil.copy2(src_path, dst_path)
        return
    config = aim.get("config", {})
    cc = aim.get("clientConfig", {}).get("kiroCli", {})

    # Resolve prompt: {{aim:include:path}} -> absolute file:// URI
    prompt = config.get("systemPrompt", "")
    if "{{aim:include:" in prompt:
        bare_path = prompt.replace("{{aim:include:", "").replace("}}", "")
        resolved = (dest / bare_path).resolve()
        if resolved.exists():
            prompt = f"file://{resolved}"
        else:
            print(f"  WARN: prompt file not found: {resolved}", file=sys.stderr)
            prompt = bare_path

    # Resolve resources: relative file:// -> absolute file:// URIs
    resources = []
    for r in cc.get("resources", []):
        if r.startswith("file://") and not r.startswith("file:///"):
            rel_path = r[len("file://"):]
            resolved = (dest / rel_path).resolve()
            if resolved.exists():
                resources.append(f"file://{resolved}")
            else:
                print(f"  WARN: resource not found: {resolved}", file=sys.stderr)
                resources.append(r)
        else:
            resources.append(r)

    native = {
        "name": aim.get("name", ""),
        "description": config.get("description", ""),
        "prompt": prompt,
        "model": config.get("model", ""),
        "tools": cc.get("tools", []),
        "allowedTools": cc.get("allowedTools", []),
    }
    if resources:
        native["resources"] = resources
    with open(dst_path, "w") as f:
        json.dump(native, f, indent=2)
        f.write("\n")


def _copy_artifacts(src, dest):
    """Copy artifact directories from src to dest."""
    for d in ARTIFACT_DIRS:
        s = src / d
        if not s.exists():
            continue
        t = dest / d
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(s, t, dirs_exist_ok=True)
        print(f"  {d}/ -> {t}")


def _resolve_agent_specs(dest):
    """Transform all agent specs in dest/agents/ from AIM to native format."""
    agents_dir = dest / "agents"
    if agents_dir.exists():
        for spec in agents_dir.glob("*.agent-spec.json"):
            _transform_agent_spec(spec, spec, dest)


def _find_artifact_src():
    """Locate artifact source: bundled artifacts/ or repo root."""
    # Bundled artifacts (wheel install or build)
    bundled = Path(__file__).parent / "artifacts"
    if bundled.exists():
        return bundled
    # Repo root fallback (editable install / pip install -e .)
    # deploy.py is at src/neuron_agentic_development/deploy.py, repo root is 2 levels up
    repo_root = Path(__file__).parent.parent.parent
    if (repo_root / "agents").is_dir() and (repo_root / "skills").is_dir():
        return repo_root
    return None


def _deploy(target, dest=None):
    """Deploy artifacts to target directory (~/.kiro or ~/.claude).

    Args:
        target: 'kiro' or 'claude'
        dest: Override destination (used by tests with temp dirs)
    Returns:
        0 on success, 1 on error
    """
    dest = Path(dest or os.path.expanduser(f"~/.{target}"))
    src = _find_artifact_src()
    if src is None:
        print("Error: no artifact source found (checked artifacts/ and repo root)", file=sys.stderr)
        return 1
    _copy_artifacts(src, dest)
    _resolve_agent_specs(dest)
    print(f"Deployed to {dest}")
    return 0


def deploy_to_kiro():
    """Console script entry point: deploy-neuron-agentic-development-to-kiro."""
    sys.exit(_deploy("kiro"))


def deploy_to_claude():
    """Console script entry point: deploy-neuron-agentic-development-to-claude."""
    sys.exit(_deploy("claude"))
