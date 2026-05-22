#!/usr/bin/env bash
# ============================================================================
# Autoport Skill — Environment Setup (SDK 2.28)
#
# Thin shell orchestrator. Single non-interactive path:
#   Step 1: Detect OS
#   Step 2: Check system Neuron packages
#   Step 3: Activate an existing venv or create one; install requirements
#   Step 4: Validate (calls check_env.py); on failure, re-install once
#
# Usage:
#   bash setup_autoport.sh                             # creates ./_venv, installs
#   bash setup_autoport.sh --venv /path/to/venv        # use/create at that path
#   bash setup_autoport.sh --venv /path --validate-only  # check only, no install
#
# Exit codes:
#   0  — environment is ready (emits RESOLVED:<VAR>=<path> lines)
#   2  — hard failure (missing system packages, no Python 3.10+, pip failed,
#        or --validate-only given and env is not usable)
#   3  — validation recovery needed (imports still fail after install)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/envSetup"
VENV_PATH=""
VALIDATE_ONLY=false
DRY_RUN=false
IS_AL2023=false

# ── Parse args ──────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)           VENV_PATH="$2"; shift 2 ;;
        --validate-only)  VALIDATE_ONLY=true; shift ;;
        --dry-run)        DRY_RUN=true; shift ;;
        --help|-h)
            echo "Usage: setup_autoport.sh [--venv PATH] [--validate-only] [--dry-run]"
            echo ""
            echo "  --venv PATH        Path to venv (use if exists, else create here)"
            echo "  --validate-only    Only validate the venv; do not create or install."
            echo "                     Requires --venv. Exits 2 if venv is unusable."
            echo "  --dry-run          Show what would be installed without making changes."
            echo "  -h, --help         Show this help"
            echo ""
            echo "If --venv is omitted, the script creates \$(pwd)/_venv."
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 2 ;;
    esac
done

if $VALIDATE_ONLY && [[ -z "$VENV_PATH" ]]; then
    echo "  ✗ --validate-only requires --venv <path>"
    exit 2
fi

# ── Helpers ─────────────────────────────────────────────────────────────────

find_python() {
    for candidate in python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" &>/dev/null; then
            local ver
            ver=$("$candidate" -c "import sys; print(sys.version_info.minor)")
            if [[ "$ver" -ge 10 ]]; then
                echo "$candidate"
                return 0
            fi
        fi
    done
    return 1
}

pick_requirements() {
    if $IS_AL2023 && [[ -f "$SCRIPT_DIR/requirements-al2023.txt" ]]; then
        echo "$SCRIPT_DIR/requirements-al2023.txt"
    else
        echo "$SCRIPT_DIR/requirements.txt"
    fi
}

# Run check_env.py, capture output, emit to stdout, extract FAILED:<pkg> list.
# Sets: VALIDATE_EXIT, VALIDATE_OUTPUT, FAILED_PKGS.
run_check_env() {
    local req_file="$1"
    VALIDATE_EXIT=0
    VALIDATE_OUTPUT=$(python3 "$SCRIPT_DIR/check_env.py" "$req_file" 2>&1) || VALIDATE_EXIT=$?
    echo "$VALIDATE_OUTPUT"
    FAILED_PKGS=$(echo "$VALIDATE_OUTPUT" | grep '^FAILED:' | cut -d: -f2 | tr '\n' ' ' | sed 's/  */ /g; s/ $//' || true)
}

print_ready() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  ✓ READY"
    echo "  Venv: $VENV_PATH"
    echo "  Reuse: --venv $VENV_PATH"
    echo "════════════════════════════════════════════════════════════"
}

# ── Dry-run: report what would happen, then exit ─────────────────────────────

if $DRY_RUN; then
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  DRY-RUN: Environment Setup Plan"
    echo "════════════════════════════════════════════════════════════"

    # Determine OS for requirements selection
    _OS_ID=$(. /etc/os-release 2>/dev/null && echo "$ID" || echo "unknown")
    _OS_VER=$(. /etc/os-release 2>/dev/null && echo "$VERSION_ID" || echo "unknown")
    if [[ "$_OS_ID" == "amzn" && "$_OS_VER" == "2023" ]]; then
        IS_AL2023=true
    fi

    if [[ -z "$VENV_PATH" ]]; then
        VENV_PATH="$(pwd)/_venv"
    fi

    REQ_FILE=$(pick_requirements)
    _NEEDS_INSTALL=true

    echo ""
    echo "  Venv path: $VENV_PATH"
    if [[ -f "$VENV_PATH/bin/activate" ]]; then
        # Actually validate the venv to give an accurate status
        export PATH="$VENV_PATH/bin:$PATH"
        _VALIDATE_EXIT=0
        _VALIDATE_OUT=$(python3 "$SCRIPT_DIR/check_env.py" "$REQ_FILE" 2>&1) || _VALIDATE_EXIT=$?
        if [[ $_VALIDATE_EXIT -eq 0 ]]; then
            _NEEDS_INSTALL=false
            echo "  Status:    EXISTS — validated, all imports OK"
            echo ""
            echo "$_VALIDATE_OUT" | sed 's/^/  /'
            echo ""
            echo "  No install needed. No changes will be made."
        elif [[ $_VALIDATE_EXIT -eq 5 ]]; then
            _NEEDS_INSTALL=false
            echo "  Status:    EXISTS — runtime error (packages present but imports failing)"
            echo ""
            echo "$_VALIDATE_OUT" | sed 's/^/  /'
            echo ""
            echo "  RUNTIME_ISSUE: Reinstalling will NOT fix this. Check system libraries and drivers."
        else
            echo "  Status:    EXISTS — needs packages installed"
            echo ""
            echo "$_VALIDATE_OUT" | sed 's/^/  /'
            echo ""
            echo "  Packages to install (from $(basename "$REQ_FILE")):"
            grep -E '^[^#-]' "$REQ_FILE" | sed 's/^/    /' || true
            echo ""
            echo "  Index: $(grep -oP '(?<=--extra-index-url ).*' "$REQ_FILE" || echo 'default PyPI')"
            echo ""
            echo "  ACTION_REQUIRED: Ask the user for consent before running the install."
        fi
    else
        echo "  Status:    DOES NOT EXIST (will be created)"
        echo ""
        echo "  Packages to install (from $(basename "$REQ_FILE")):"
        grep -E '^[^#-]' "$REQ_FILE" | sed 's/^/    /' || true
        echo ""
        echo "  Index: $(grep -oP '(?<=--extra-index-url ).*' "$REQ_FILE" || echo 'default PyPI')"
        echo ""
        echo "  ACTION_REQUIRED: Ask the user (1) consent to install and (2) where to create the venv."
    fi

    echo ""
    echo "  No changes made."
    echo "════════════════════════════════════════════════════════════"
    if $_NEEDS_INSTALL; then
        exit 4
    fi
    if [[ -f "$VENV_PATH/bin/activate" && $_VALIDATE_EXIT -eq 5 ]]; then
        exit 5
    fi
    exit 0
fi

# ============================================================================
# STEP 1: DETECT OS
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Step 1: Detect Environment"
echo "════════════════════════════════════════════════════════════"

OS_ID=$(. /etc/os-release 2>/dev/null && echo "$ID" || echo "unknown")
OS_VERSION=$(. /etc/os-release 2>/dev/null && echo "$VERSION_ID" || echo "unknown")
GLIBC_VER=$(ldd --version 2>&1 | head -1 | grep -oP '\d+\.\d+$' || echo "unknown")

echo "  OS:    $OS_ID $OS_VERSION"
echo "  GLIBC: $GLIBC_VER"

if [[ "$OS_ID" == "amzn" && "$OS_VERSION" == "2023" ]]; then
    IS_AL2023=true
    echo "  Note:  AL2023 detected"
fi

# ============================================================================
# STEP 2: CHECK SYSTEM PACKAGES
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Step 2: Check System Neuron Packages"
echo "════════════════════════════════════════════════════════════"

missing=""
for pkg in aws-neuronx-dkms aws-neuronx-runtime-lib aws-neuronx-collectives aws-neuronx-tools; do
    if dpkg-query -W "$pkg" > /dev/null 2>&1; then
        echo "  ✓ $pkg = $(dpkg-query -W -f '${Version}' "$pkg")"
    elif rpm -q "$pkg" > /dev/null 2>&1; then
        echo "  ✓ $pkg = $(rpm -q --queryformat '%{VERSION}' "$pkg")"
    else
        echo "  ✗ $pkg — NOT INSTALLED"
        missing="$missing $pkg"
    fi
done

if [[ -n "$missing" ]]; then
    echo ""
    echo "  STOP — Missing system packages:$missing"
    echo ""
    echo "  Ask your admin to install them:"
    if [[ "$OS_ID" == "amzn" ]]; then
        echo "    sudo dnf install -y$missing"
    else
        echo "    sudo apt-get update -y"
        echo "    sudo apt-get install -y$missing"
    fi
    echo ""
    echo "  Then re-run this script."
    exit 2
fi

echo ""
echo "  All system packages present."

# ============================================================================
# STEP 3: RESOLVE VENV
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Step 3: Resolve Python Environment"
echo "════════════════════════════════════════════════════════════"

if [[ -z "$VENV_PATH" ]]; then
    VENV_PATH="$(pwd)/_venv"
    echo "  No --venv provided; using default: $VENV_PATH"
fi

REQ_FILE=$(pick_requirements)

if [[ -f "$VENV_PATH/bin/activate" ]]; then
    echo "  Found existing venv: $VENV_PATH"
    export PATH="$VENV_PATH/bin:$PATH"
elif $VALIDATE_ONLY; then
    echo "  ✗ No venv at $VENV_PATH (and --validate-only forbids creating one)."
    exit 2
else
    echo "  No venv at $VENV_PATH — creating."
    PY=$(find_python) || {
        echo ""
        echo "  ✗ No Python 3.10+ found."
        echo "    Searched: python3.12, python3.11, python3.10, python3"
        exit 2
    }
    echo "  Using $PY ($($PY --version 2>&1))"
    "$PY" -m venv "$VENV_PATH"
    export PATH="$VENV_PATH/bin:$PATH"
    python3 -m pip install --quiet --upgrade pip
    echo "  ✓ Venv created"

    echo ""
    echo "  Installing from $(basename "$REQ_FILE"):"
    grep -E '^[^#-]' "$REQ_FILE" | sed 's/^/    /' || true
    python3 -m pip install -r "$REQ_FILE" || {
        echo "  ✗ pip install failed."
        exit 2
    }
    echo "  ✓ Dependencies installed."
fi

# ============================================================================
# STEP 4: VALIDATE (delegates to check_env.py)
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Step 4: Validate"
echo "════════════════════════════════════════════════════════════"

run_check_env "$REQ_FILE"

if [[ $VALIDATE_EXIT -eq 0 ]]; then
    print_ready
    exit 0
fi

if [[ $VALIDATE_EXIT -eq 5 ]]; then
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  ✗ RUNTIME ERROR — packages present but imports failing"
    echo ""
    echo "  Reinstalling will NOT fix this."
    echo "  Check system libraries and drivers (e.g., GLIBC, Neuron drivers)."
    echo "════════════════════════════════════════════════════════════"
    exit 5
fi

# In validate-only mode we never install; a failure here is terminal.
if $VALIDATE_ONLY; then
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  ✗ VALIDATION FAILED (validate-only mode)"
    echo ""
    if [[ -n "$FAILED_PKGS" ]]; then
        echo "  Failed packages: $FAILED_PKGS"
    fi
    echo "  Cannot use this venv. Provide a different venv, or re-run"
    echo "  without --validate-only to install packages."
    echo "════════════════════════════════════════════════════════════"
    exit 2
fi

# ── Validation failed — install once and re-check ──────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✗ VALIDATION FAILED"
echo ""
if [[ -n "$FAILED_PKGS" ]]; then
    echo "  Failed packages: $FAILED_PKGS"
else
    echo "  Check the errors above."
fi
echo ""
echo "  Installing requirements and re-validating..."
echo "════════════════════════════════════════════════════════════"

python3 -m pip install -r "$REQ_FILE" || {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  ✗ PIP INSTALL FAILED"
    echo "  Check the pip output above."
    echo "════════════════════════════════════════════════════════════"
    exit 2
}
echo "  ✓ Installed."
echo ""

run_check_env "$REQ_FILE"

if [[ $VALIDATE_EXIT -eq 0 ]]; then
    print_ready
    exit 0
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✗ STILL FAILING AFTER INSTALL"
echo ""
if [[ -n "$FAILED_PKGS" ]]; then
    echo "  Still failing: $FAILED_PKGS"
fi
echo "  Could not diagnose automatically."
echo "  → Agent should present recovery options (a)/(b)."
echo "════════════════════════════════════════════════════════════"
exit 3
