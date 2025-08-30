#!/usr/bin/env bash
# generate_baseline.sh - Generate performance baseline using Python utilities
#
# This script is a simple wrapper around the Python benchmark utilities.
# It preserves the original CLI interface while using the more maintainable
# Python implementation.
#
# Usage: generate_baseline.sh [--dev]
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

# Parse arguments and show help
for arg in "$@"; do
	if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
		cat <<EOF
Usage: generate_baseline.sh [--dev]

Generate performance baseline by running benchmarks and parsing Criterion results.

Options:
  --dev       Use development mode with faster benchmark settings
              (sample_size=10, measurement_time=2s, warmup_time=1s)
  -h, --help  Show this help message and exit

Output:
  Creates benches/baseline_results.txt with performance baseline data.

This script is a wrapper around the Python benchmark utilities for
backward compatibility with existing workflows.
EOF
		exit 0
	fi
done

# Build Python command with all arguments
PYTHON_CMD=("python3" "$PYTHON_BENCHMARK_UTILS" "generate-baseline")
PYTHON_CMD+=("$@") # Pass through all arguments

# Change to project root for consistent behavior
cd "$PROJECT_ROOT"

# Execute Python baseline generation
echo "🐍 Using Python-based baseline generation..."
exec "${PYTHON_CMD[@]}"
