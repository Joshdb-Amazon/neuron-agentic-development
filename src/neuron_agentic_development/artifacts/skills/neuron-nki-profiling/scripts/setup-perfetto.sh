#!/bin/bash
# setup-perfetto.sh - Launch Perfetto trace processor for NKI profile visualization
#
# Usage: setup-perfetto.sh <path_to_pftrace>
#
# This script:
# 1. Checks if trace_processor exists in the skill's scripts directory
# 2. If not, downloads it from https://get.perfetto.dev/trace_processor
# 3. Makes it executable
# 4. Launches it with --httpd flag for the given .pftrace file
#
# After launching, open https://ui.perfetto.dev and click "Open trace processor (RPC)"
# to load the trace from localhost.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACE_PROCESSOR="$SCRIPT_DIR/trace_processor"
PFTRACE_FILE="${1:?Usage: $0 <path_to_pftrace>}"

# Validate input file exists
if [ ! -f "$PFTRACE_FILE" ]; then
    echo "Error: Perfetto trace file not found: $PFTRACE_FILE"
    exit 1
fi

# Ensure trace_processor exists
if [ ! -f "$TRACE_PROCESSOR" ]; then
    echo "Downloading Perfetto trace processor..."
    curl -L https://get.perfetto.dev/trace_processor -o "$TRACE_PROCESSOR"
    chmod +x "$TRACE_PROCESSOR"
    echo "Downloaded trace_processor to $TRACE_PROCESSOR"
fi

# Verify trace_processor is executable
if [ ! -x "$TRACE_PROCESSOR" ]; then
    chmod +x "$TRACE_PROCESSOR"
fi

echo "=============================================="
echo "Starting Perfetto trace processor HTTP server"
echo "=============================================="
echo ""
echo "Trace file: $PFTRACE_FILE"
echo ""
echo "To visualize:"
echo "  1. Open https://ui.perfetto.dev in your browser"
echo "  2. Click 'Open trace processor (RPC)'"
echo "  3. The trace will load from localhost"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Launch HTTP server for Perfetto UI
"$TRACE_PROCESSOR" --httpd "$PFTRACE_FILE"
