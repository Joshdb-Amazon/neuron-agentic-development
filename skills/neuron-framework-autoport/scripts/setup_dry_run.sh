#!/usr/bin/env bash
# Create (or clean) a venv with NxDI/NxD/transformers (no deps) for autoport dry-run benchmarks.
# Usage: setup_dry_run.sh          # idempotent setup (no-op if venv exists)
#        setup_dry_run.sh --clean  # rmtree the venv
set -euo pipefail

venv="$HOME/tmp/autoport/venv"

if [[ "${1:-}" == "--clean" ]]; then
    if [[ -d "$venv" ]]; then
        echo "Removing autoport dry-run venv: $venv"
        rm -rf "$venv"
    else
        echo "No autoport dry-run venv to clean at $venv"
    fi
    exit 0
fi

req="$(dirname "$0")/envSetup/requirements.txt"
[[ -d "$venv" ]] && exit 0

mkdir -p "$(dirname "$venv")"
python3 -m venv "$venv"
"$venv/bin/pip" install --no-deps -r "$req"
