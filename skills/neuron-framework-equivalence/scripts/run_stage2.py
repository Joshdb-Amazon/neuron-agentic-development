#!/usr/bin/env python3
"""
Stage 2: Component Test Runner.

Imports and executes test_*.py files in a tests directory, captures R-ratios.
No external dependencies beyond torch and the test files themselves.

Usage:
    python3 scripts/run_stage2.py --tests-dir /path/to/tests --tau-r 1.2
"""

import argparse
import importlib
import importlib.util
import io
import json
import re
import sys
import os


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Component Test Runner")
    parser.add_argument("--tests-dir", required=True)
    parser.add_argument("--tau-r", type=float, default=1.2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    os.environ["NXD_CPU_MODE"] = "1"

    from pathlib import Path
    tests_path = Path(args.tests_dir)
    test_files = sorted(tests_path.glob("test_*.py"))

    if not test_files:
        print(f"No test_*.py files found in {args.tests_dir}")
        return 1

    if str(tests_path) not in sys.path:
        sys.path.insert(0, str(tests_path))

    r_pattern = re.compile(r"R-ratio\s*=\s*([\d.]+)\s*\(threshold=([\d.]+)\)")
    pass_pattern = re.compile(r"\[PASS\]\s+(.+)")
    fail_pattern = re.compile(r"\[FAIL\]\s+(.+)")

    results = []
    total = passed = failed = 0

    for test_file in test_files:
        try:
            mod_name = f"_eq_{test_file.stem}"
            spec = importlib.util.spec_from_file_location(mod_name, str(test_file))
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
        except Exception as e:
            name = test_file.stem.split("_", 2)[-1]
            print(f"\n  ✗ {name}: IMPORT ERROR — {e}")
            results.append({"component": name, "r_ratio": None, "passed": False, "error": str(e)})
            total += 1; failed += 1
            continue

        test_fns = [(n, getattr(module, n)) for n in sorted(dir(module))
                    if n.startswith("test_") and callable(getattr(module, n))]

        for fn_name, fn in test_fns:
            total += 1
            name = fn_name.replace("test_", "")

            old_stdout = sys.stdout
            captured = io.StringIO()
            sys.stdout = captured
            ok = False
            err = None
            try:
                fn()
                ok = True
            except AssertionError as e:
                err = str(e)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            finally:
                sys.stdout = old_stdout

            output = captured.getvalue()
            print(output, end="")

            r_match = r_pattern.search(output)
            r_ratio = float(r_match.group(1)) if r_match else None
            threshold = float(r_match.group(2)) if r_match else args.tau_r

            name_match = pass_pattern.search(output) or fail_pattern.search(output)
            if name_match:
                name = name_match.group(1).strip()

            test_passed = ok and (r_ratio is not None and r_ratio < threshold)
            if test_passed:
                passed += 1
            else:
                failed += 1
            if err:
                print(f"  Error: {err}")

            results.append({"component": name, "r_ratio": r_ratio, "passed": test_passed})

    print(f"\n{'=' * 60}")
    print(f"  Stage 2 Summary: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    for r in results:
        tag = "✓" if r["passed"] else "✗"
        ratio = f"R={r['r_ratio']:.4f}" if r["r_ratio"] is not None else "R=ERROR"
        print(f"  {tag} {r['component']}: {ratio}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"results": results, "passed": passed, "failed": failed, "total": total},
                      f, indent=2, default=str)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
