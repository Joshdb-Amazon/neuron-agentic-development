"""Custom build backend: copies repo-root artifacts into the package before building."""
import shutil
from pathlib import Path
from setuptools.build_meta import *  # noqa: F401,F403
from setuptools.build_meta import build_wheel as _build_wheel, build_sdist as _build_sdist

_ROOT = Path(__file__).parent
_PKG_ARTIFACTS = _ROOT / "src" / "neuron_agentic_development" / "artifacts"
_ARTIFACT_DIRS = ("agent-sops", "agents", "context", "evals", "hooks", "skills", "tools")


def _bundle():
    _PKG_ARTIFACTS.mkdir(exist_ok=True)
    for d in _ARTIFACT_DIRS:
        src = _ROOT / d
        dst = _PKG_ARTIFACTS / d
        if src.is_dir() and not dst.exists():
            shutil.copytree(src, dst)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _bundle()
    return _build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    _bundle()
    return _build_sdist(sdist_directory, config_settings)
