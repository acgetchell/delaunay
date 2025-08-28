#!/usr/bin/env bash
set -euo pipefail

# Error handling function
error_exit() {
    local message="$1"
    local code="${2:-1}"
    echo "ERROR: $message" >&2
    exit "$code"
}

# Script to run all examples in the delaunay project

# Dependency checking function
check_dependencies() {
    # Array of required commands
    local required_commands=("cargo")

    # Check each required command
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            error_exit "$cmd is required but not found. Please install it to proceed."
        fi
    done
}

# Run dependency checks
check_dependencies

# Find project root (directory containing Cargo.toml)
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)

# Create results directory for future use
mkdir -p "${PROJECT_ROOT}/benches/results"

# Ensure we're executing from the project root
cd "${PROJECT_ROOT}"

echo "Running all examples for delaunay project..."
echo "=============================================="

# Automatically discover all examples from the examples directory
all_examples=()
while IFS= read -r -d '' file; do
    # Extract filename without path and .rs extension
    example_name=$(basename "$file" .rs)
    all_examples+=("$example_name")
done < <(find "${PROJECT_ROOT}/examples" -name "*.rs" -type f -print0)

# Define special example that needs special handling
special_example="test_circumsphere"

# Filter all_examples to exclude test_circumsphere into simple_examples
simple_examples=()
for example in "${all_examples[@]}"; do
    if [[ "$example" != "$special_example" ]]; then
        simple_examples+=("$example")
    fi
done

# Run simple examples
for example in "${simple_examples[@]}"; do
    echo "=== Running $example ==="
    cargo run --release --example "$example" || error_exit "Example $example failed!"
done

# Run test_circumsphere with comprehensive test categories
test_circumsphere_tests=(
    "all"              # All basic dimensional tests and orientation tests
    "test-all-points"   # Single point tests in all dimensions
    "debug-all"         # All debug tests
)

echo
echo "=== Running test_circumsphere comprehensive tests ==="
echo "---------------------------------------------------"

for test_name in "${test_circumsphere_tests[@]}"; do
    echo
    echo "--- Running test_circumsphere $test_name ---"
    if ! cargo run --release --example test_circumsphere -- "$test_name"; then
        error_exit "test_circumsphere $test_name failed!"
    fi
done

echo
echo "=============================================="
echo "All examples and tests completed successfully!"
