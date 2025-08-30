#!/usr/bin/env bash
# compare_benchmarks.sh - Compare current benchmark performance against baseline
#
# This script is a simple wrapper around the Python benchmark utilities.
# It preserves the original CLI interface while using the more maintainable
# Python implementation.
#
# Usage: compare_benchmarks.sh [--dev]
#   --dev    Use development mode with faster benchmark settings
#            (sample_size=10, measurement_time=2s, warmup_time=1s)

set -euo pipefail

# Find script directory and project root
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR" && cd .. && pwd)

# Check if Python utilities are available
PYTHON_BENCHMARK_UTILS="${SCRIPT_DIR}/benchmark_utils.py"
if [[ ! -f "$PYTHON_BENCHMARK_UTILS" ]]; then
	echo "ERROR: Python benchmark utilities not found: $PYTHON_BENCHMARK_UTILS" >&2
	echo "       Make sure benchmark_utils.py exists in the scripts/ directory." >&2
	exit 1
fi

# Check for Python 3
if ! command -v python3 >/dev/null 2>&1; then
	echo "ERROR: python3 is not installed or not in PATH" >&2
	echo "       Please install Python 3 to use this script." >&2
	exit 1
fi

# Baseline file path
BASELINE_FILE="${PROJECT_ROOT}/benches/baseline_results.txt"

# Parse arguments and show help
for arg in "$@"; do
	if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
		cat <<EOF
Usage: compare_benchmarks.sh [--dev]

Run benchmark and compare results with baseline, creating compare_results.txt

Options:
  --dev       Use development mode with faster benchmark settings
              (sample_size=10, measurement_time=2s, warmup_time=1s)
  -h, --help  Show this help message and exit

Exit Codes:
  0  Success - no significant regressions
  1  Error occurred or significant regressions found

Input:
  Requires benches/baseline_results.txt (run generate_baseline.sh first)

Output:
  Creates benches/compare_results.txt with comparison results

This script is a wrapper around the Python benchmark utilities for
backward compatibility with existing workflows.
EOF
		exit 0
	fi
done

# Check if baseline exists
if [[ ! -f "$BASELINE_FILE" ]]; then
	echo "ERROR: Baseline results file not found: $BASELINE_FILE" >&2
	echo "       Run generate_baseline.sh first to create a baseline." >&2
	exit 1
fi

# Build Python command with baseline file and arguments
PYTHON_CMD=("python3" "$PYTHON_BENCHMARK_UTILS" "compare" "--baseline" "$BASELINE_FILE")
PYTHON_CMD+=("$@") # Pass through all arguments

# Change to project root for consistent behavior
cd "$PROJECT_ROOT"

# Execute Python benchmark comparison
echo "🐍 Using Python-based benchmark comparison..."
exec "${PYTHON_CMD[@]}"
