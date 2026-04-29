#!/bin/bash
# NKI Dev Suite - Environment Check Script
# Runs on SessionStart to verify NKI venv is configured and has expected packages

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Resolve NKI_VENV_PATH from environment or .claude/nki-dev-suite.local.md frontmatter
VENV_PATH="${NKI_VENV_PATH:-${nki_venv_path:-}}"

if [ -z "$VENV_PATH" ]; then
    LOCAL_MD=".claude/nki-dev-suite.local.md"
    if [ -f "$LOCAL_MD" ]; then
        VENV_PATH=$(sed -n '/^---$/,/^---$/{ /^nki_venv_path:/{ s/^nki_venv_path:[[:space:]]*["'\'']\{0,1\}\([^"'\'']*\)["'\'']\{0,1\}[[:space:]]*$/\1/; p; } }' "$LOCAL_MD")
    fi
fi

if [ -z "$VENV_PATH" ]; then
    echo -e "${YELLOW}[NKI]${NC} NKI_VENV_PATH not configured. Set the environment variable or add nki_venv_path to .claude/nki-dev-suite.local.md"
    exit 0
fi

# Verify venv directory and activate script exist
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}[NKI]${NC} NKI_VENV_PATH directory not found: $VENV_PATH"
    exit 0
fi

if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo -e "${YELLOW}[NKI]${NC} Not a valid venv (missing bin/activate): $VENV_PATH"
    exit 0
fi

# Check for expected packages in site-packages (without activating the venv)
SITE_PACKAGES=$(find "$VENV_PATH/lib" -maxdepth 2 -type d -name "site-packages" 2>/dev/null | head -1)

if [ -z "$SITE_PACKAGES" ]; then
    echo -e "${YELLOW}[NKI]${NC} Venv found but no site-packages directory: $VENV_PATH"
    exit 0
fi

MISSING=""
for pkg in neuronxcc nki; do
    if [ ! -d "$SITE_PACKAGES/$pkg" ]; then
        MISSING="$MISSING $pkg"
    fi
done

if [ -n "$MISSING" ]; then
    echo -e "${YELLOW}[NKI]${NC} Venv at $VENV_PATH is missing packages:$MISSING"
else
    echo -e "${GREEN}[NKI]${NC} Venv OK: $VENV_PATH"
fi

# Check for neuron-profile CLI on PATH (system executable, not a venv package)
if command -v neuron-profile &>/dev/null; then
    echo -e "${GREEN}[NKI]${NC} neuron-profile found: $(command -v neuron-profile)"
else
    echo -e "${YELLOW}[NKI]${NC} neuron-profile not found on PATH (needed for /neuron-nki-profiling and /neuron-nki-analyzing-profile-visual)"
fi

# Always exit successfully - missing packages are warnings, not errors
exit 0
