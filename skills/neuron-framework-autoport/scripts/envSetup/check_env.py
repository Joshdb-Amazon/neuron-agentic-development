#!/usr/bin/env python3
"""
Autoport Skill — Environment Validator

Called by setup_autoport.sh after venv is activated.
Checks versions, imports, and prints resolved paths.

Exit codes:
    0 = all imports succeeded (version mismatches are warnings only)
    2 = import failure (packages missing or broken)

Output format:
    Lines starting with "RESOLVED:" are machine-readable (on success):
        RESOLVED:NXDI_SRC=/path/to/...
        RESOLVED:NXD_SRC=/path/to/...
        RESOLVED:TRANSFORMERS_SRC=/path/to/...

    Lines starting with "FAILED:" name each package whose import
    failed (consumed by setup_autoport.sh):
        FAILED:neuronx_distributed_inference
        FAILED:transformers

    Version mismatches are reported as human-readable warnings only;
    they never emit FAILED: and never change the exit code.
"""

import sys
import os
import traceback
import warnings

warnings.filterwarnings("ignore")

PACKAGES = [
    ("neuronx_distributed_inference", "nxdi", "NXDI_SRC"),
    ("neuronx_distributed", "nxd", "NXD_SRC"),
    ("transformers", "transformers", "TRANSFORMERS_SRC"),
]


def check_versions(req_file):
    """Read requirements file, report installed vs expected.

    Mismatches and missing packages are reported as warnings only —
    the env is deemed usable if imports succeed (see check_imports).
    Returns the list of packages with mismatched/missing versions
    for informational display by the caller; it does not affect
    the final exit code.
    """
    from importlib.metadata import version, PackageNotFoundError

    expected = {}
    with open(req_file) as f:
        for line in f:
            line = line.strip()
            if "==" in line and not line.startswith("#") and not line.startswith("-"):
                pkg, ver = line.split("==", 1)
                expected[pkg.strip()] = ".".join(ver.strip().split(".")[:2])

    mismatched = []
    for pkg, want in expected.items():
        try:
            got = version(pkg)
            major_minor = ".".join(got.split(".")[:2])
            if major_minor == want:
                print(f"  ✓ {pkg}=={got}")
            else:
                print(f"  ⚠ {pkg}=={got} (expected {want}.x) — warning only")
                mismatched.append(pkg)
        except PackageNotFoundError:
            print(f"  ⚠ {pkg} NOT INSTALLED (expected {want}.x) — warning only")
            mismatched.append(pkg)

    return mismatched


def check_imports():
    """Import the 3 required packages.

    Returns (resolved_paths, failed_pkgs, runtime_errors).
    - FAILED:<name> means the package is not installed (reinstall may help).
    - RUNTIME_ERROR:<name> means the package exists but import fails due to
      system-level issues (GLIBC, drivers, etc.) — reinstall will NOT help.
    """
    from importlib.metadata import version, PackageNotFoundError

    resolved = {}
    failed = []
    runtime_errors = []

    for module_name, short_name, var_name in PACKAGES:
        try:
            mod = __import__(module_name)
            path = mod.__path__[0]
            print(f"  ✓ import {short_name}: {path}")
            resolved[var_name] = path
        except ModuleNotFoundError:
            print(f"  ✗ import {short_name} FAILED (not installed)")
            traceback.print_exc()
            print(f"FAILED:{module_name}")
            failed.append(module_name)
        except Exception:
            pkg_installed = False
            try:
                version(module_name.replace("_", "-"))
                pkg_installed = True
            except PackageNotFoundError:
                pass
            if not pkg_installed:
                try:
                    version(module_name)
                    pkg_installed = True
                except PackageNotFoundError:
                    pass

            if pkg_installed:
                print(f"  ✗ import {short_name} FAILED (package present but runtime error)")
                traceback.print_exc()
                print(f"RUNTIME_ERROR:{module_name}")
                runtime_errors.append(module_name)
            else:
                print(f"  ✗ import {short_name} FAILED (not installed)")
                traceback.print_exc()
                print(f"FAILED:{module_name}")
                failed.append(module_name)

    return resolved, failed, runtime_errors


def print_resolved(resolved):
    """Print resolved paths in machine-readable format."""
    for module_name, short_name, var_name in PACKAGES:
        path = resolved.get(var_name, "<not resolved>")
        print(f"RESOLVED:{var_name}={path}")


def main():
    req_file = None

    # Accept requirements file as argument
    if len(sys.argv) > 1:
        req_file = sys.argv[1]

    # Auto-detect if not provided
    if not req_file:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for candidate in ["requirements.txt", "requirements-al2023.txt"]:
            path = os.path.join(script_dir, candidate)
            if os.path.exists(path):
                req_file = path
                break

    if not req_file or not os.path.exists(req_file):
        print(f"  ✗ No requirements file found")
        return 2

    # Version check (informational only — never fails the script).
    print(f"  Version check ({os.path.basename(req_file)}):")
    check_versions(req_file)

    # Import check — this is the source of truth.
    print()
    print(f"  Import check:")
    resolved, import_failed, runtime_errors = check_imports()

    if runtime_errors and not import_failed:
        print()
        print(f"  RUNTIME_ISSUE: Packages are installed but imports fail due to system-level")
        print(f"  incompatibility (e.g. GLIBC version, driver mismatch). Reinstalling will NOT help.")
        print(f"  Check system packages and library versions.")
        return 5

    if import_failed:
        return 2

    print()
    print_resolved(resolved)
    return 0


if __name__ == "__main__":
    sys.exit(main())
