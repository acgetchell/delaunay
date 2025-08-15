#!/usr/bin/env bash
# compare_benchmarks.sh - Compare current benchmark performance against baseline
#
# PARSING LOGIC AND FORMATTING CONVENTIONS:
# ========================================
# 
# This script runs a fresh benchmark and compares results against the established baseline.
# It parses both current results and baseline data to calculate performance changes.
#
# BENCHMARK EXECUTION:
# - Runs `cargo bench --bench small_scale_triangulation` (keeping previous results)
# - Captures Criterion output including change metrics vs. baseline
# - Parses both current results and baseline file for comparison
#
# INPUT FORMATS:
# - Current run: Criterion output with time/throughput/change data
# - Baseline: benches/baseline_results.txt with standardized format
#
# OUTPUT FORMAT CONVENTIONS:
# =========================
# - Header: Date, Git commit, and baseline metadata
# - Section format: "=== {N} Points ({D}D) ===" matching baseline
# - Current metrics: Time, Throughput (from fresh run)
# - Baseline metrics: Time, Throughput (from baseline file)
# - Change analysis: Time Change %, Throughput Change % with ✅/⚠️ indicators
# - Regression threshold: >5% change triggers warning
#
set -euo pipefail

# Error handling function
error_exit() {
    local message="$1"
    local code="${2:-1}"
    echo "ERROR: $message" >&2
    exit "$code"
}


# Print usage information
usage() {
    echo "Usage: compare_benchmarks.sh [-h|--help] [--dev]"
    echo
    echo "Runs benchmark and compares results with baseline, creating compare_results.txt"
    echo
    echo "Options:"
    echo "  -h, --help      Show this help message and exit"
    echo "  --dev           Use development mode with faster benchmark settings"
    echo "                  (sample_size=10, measurement_time=2s, warmup_time=1s)"
    echo
    echo "Exit Codes:"
    echo "  0  Success - no significant regressions"
    echo "  1  Error occurred or significant regressions found"
    exit 0
}

# Find project root and set default values
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)

# Check if benchmark_parser.sh dependency exists
BENCHMARK_PARSER_PATH="${PROJECT_ROOT}/scripts/benchmark_parser.sh"
if [[ ! -f "$BENCHMARK_PARSER_PATH" ]]; then
    error_exit "Required dependency not found: $BENCHMARK_PARSER_PATH. This script depends on benchmark_parser.sh for parsing functions."
fi

# Check if hardware_info.sh dependency exists
HARDWARE_INFO_PATH="${PROJECT_ROOT}/scripts/hardware_info.sh"
if [[ ! -f "$HARDWARE_INFO_PATH" ]]; then
    error_exit "Required dependency not found: $HARDWARE_INFO_PATH. This script depends on hardware_info.sh for hardware detection."
fi

# Source the shared utilities
# Note: We explicitly check file existence above, so disable shellcheck warnings
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/benchmark_parser.sh"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/hardware_info.sh"

# Parse command line arguments
DEV_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
        *)
            error_exit "Unknown argument: $1. Use -h for help."
            ;;
    esac
done

# Error handling with explicit checks (trap removed to avoid parsing issues)

# Check dependencies
command -v bc > /dev/null 2>&1 || error_exit "bc is required but not found. Please install via your system package manager (e.g., apt, brew, winget)"

# File paths
BASELINE_FILE="${PROJECT_ROOT}/benches/baseline_results.txt"
COMPARE_FILE="${PROJECT_ROOT}/benches/compare_results.txt"
TEMP_FILE=$(mktemp)

# Check if baseline exists
if [[ ! -f "$BASELINE_FILE" ]]; then
    error_exit "Baseline results file not found: $BASELINE_FILE. Run scripts/generate_baseline.sh first."
fi

echo "Running benchmark and comparing against baseline..."
echo "Baseline file: $BASELINE_FILE"
echo "Compare file: $COMPARE_FILE"

# Get current date and git commit
CURRENT_DATE=$(date)
GIT_COMMIT=$(git rev-parse HEAD)

# Collect current hardware information using shared utility
echo "Collecting current hardware information..."
CURRENT_HARDWARE=$(get_hardware_info_kv)

# Extract baseline hardware information
BASELINE_HARDWARE=$(extract_baseline_hardware "$BASELINE_FILE")

# Extract baseline metadata
BASELINE_DATE=$(grep "^Date:" "$BASELINE_FILE" | cut -d' ' -f2-)
BASELINE_COMMIT=$(grep "^Git commit:" "$BASELINE_FILE" | cut -d' ' -f3)

# Run fresh benchmark
if [[ "$DEV_MODE" == "true" ]]; then
    echo "Running cargo bench --bench small_scale_triangulation (DEV MODE)..."
    if ! cargo bench --bench small_scale_triangulation -- --sample-size 10 --measurement-time 2 --warm-up-time 1 --noplot > "$TEMP_FILE" 2>&1; then
        rm -f "$TEMP_FILE"
        error_exit "Failed to run benchmark in dev mode"
    fi
else
    echo "Running cargo bench --bench small_scale_triangulation..."
    if ! cargo bench --bench small_scale_triangulation > "$TEMP_FILE" 2>&1; then
        rm -f "$TEMP_FILE"
        error_exit "Failed to run benchmark"
    fi
fi

# Create compare results file with headers and hardware comparison
cat > "$COMPARE_FILE" <<EOF
Comparison Results
==================
Current Date: $CURRENT_DATE
Current Git commit: $GIT_COMMIT

Baseline Date: $BASELINE_DATE
Baseline Git commit: $BASELINE_COMMIT

EOF

# Add hardware comparison section
compare_hardware "$CURRENT_HARDWARE" "$BASELINE_HARDWARE" >> "$COMPARE_FILE"

echo "Parsing benchmark results and comparing..."

# Variables for tracking
regression_found=false

# Parse current benchmark results and compare with baseline
current_benchmark=""
current_points=""
current_dimension=""
current_time_vals=""
current_thrpt_vals=""
current_time_change=""
current_thrpt_change=""

while IFS= read -r line; do
    # Detect benchmark result lines - handle both μ (U+03BC) and µ (U+00B5) micro symbols and nanoseconds
    if [[ "$line" =~ ^(tds_new_([0-9]+)d/tds_new/([0-9]+))[[:space:]]+time:[[:space:]]+\[([0-9.]+)[[:space:]](ns|[μµ]s|ms|s)[[:space:]]([0-9.]+)[[:space:]](ns|[μµ]s|ms|s)[[:space:]]([0-9.]+)[[:space:]](ns|[μµ]s|ms|s)\] ]]; then
        current_benchmark="${BASH_REMATCH[1]}"
        current_dimension="${BASH_REMATCH[2]}D"
        current_points="${BASH_REMATCH[3]}"
        
        # Extract time values and unit
        time_low="${BASH_REMATCH[4]}"
        time_unit="${BASH_REMATCH[5]}"
        time_mean="${BASH_REMATCH[6]}"
        time_high="${BASH_REMATCH[8]}"
        
        # Normalize the micro symbol to the Greek mu (μ) for consistency
        normalized_unit="$time_unit"
        if [[ "$time_unit" == "µs" ]]; then
            normalized_unit="μs"
        fi
        
        current_time_vals="[$time_low, $time_mean, $time_high] $normalized_unit"
        
    # Detect throughput lines
    elif [[ "$line" =~ ^[[:space:]]+thrpt:[[:space:]]+\[([0-9.]+)[[:space:]](Kelem/s|elem/s)[[:space:]]([0-9.]+)[[:space:]](Kelem/s|elem/s)[[:space:]]([0-9.]+)[[:space:]](Kelem/s|elem/s)\] ]] && [[ -n "$current_points" ]]; then
        thrpt_low="${BASH_REMATCH[1]}"
        thrpt_unit="${BASH_REMATCH[2]}"
        thrpt_mean="${BASH_REMATCH[3]}"
        thrpt_high="${BASH_REMATCH[5]}"
        
        current_thrpt_vals="[$thrpt_low, $thrpt_mean, $thrpt_high] $thrpt_unit"
        
    # Detect time change lines (from Criterion comparison)
    elif [[ "$line" =~ ^[[:space:]]+time:[[:space:]]+\[([−+-]?[0-9.]+)%[[:space:]]([−+-]?[0-9.]+)%[[:space:]]([−+-]?[0-9.]+)%\] ]] && [[ -n "$current_points" ]]; then
        time_change_low="${BASH_REMATCH[1]}"
        time_change_mean="${BASH_REMATCH[2]}"
        time_change_high="${BASH_REMATCH[3]}"
        
        current_time_change="[$time_change_low%, $time_change_mean%, $time_change_high%]"
        
    # Detect throughput change lines
    elif [[ "$line" =~ ^[[:space:]]+thrpt:[[:space:]]+\[([+−+-]?[0-9.]+)%[[:space:]]([+−+-]?[0-9.]+)%[[:space:]]([+−+-]?[0-9.]+)%\] ]] && [[ -n "$current_points" ]]; then
        thrpt_change_low="${BASH_REMATCH[1]}"
        thrpt_change_mean="${BASH_REMATCH[2]}"
        thrpt_change_high="${BASH_REMATCH[3]}"
        
        current_thrpt_change="[$thrpt_change_low%, $thrpt_change_mean%, $thrpt_change_high%]"
        
    # When we encounter trigger to output results
    elif { [[ "$line" =~ ^Benchmarking ]] && [[ -n "$current_benchmark" ]]; } || { [[ "$line" =~ ^Found.*outliers ]] && [[ -n "$current_time_change" ]] && [[ -n "$current_benchmark" ]]; }; then
        # Get baseline values for this benchmark
        baseline_time_line=$(grep -A 1 "=== $current_points Points ($current_dimension) ===" "$BASELINE_FILE" | grep "Time:" | head -1 || true)
        baseline_thrpt_line=$(grep -A 2 "=== $current_points Points ($current_dimension) ===" "$BASELINE_FILE" | grep "Throughput:" | head -1 || true)
        
        # Output the comparison section
        printf "=== %s Points (%s) ===\n" "$current_points" "$current_dimension" >> "$COMPARE_FILE"
        printf "Current Time: %s\n" "$current_time_vals" >> "$COMPARE_FILE"
        if [[ -n "$current_thrpt_vals" ]]; then
            printf "Current Throughput: %s\n" "$current_thrpt_vals" >> "$COMPARE_FILE"
        fi
        
        if [[ -n "$baseline_time_line" ]]; then
            baseline_time="${baseline_time_line//Time: /}"
            printf "Baseline Time: %s\n" "$baseline_time" >> "$COMPARE_FILE"
        fi
        
        if [[ -n "$baseline_thrpt_line" ]]; then
            baseline_thrpt="${baseline_thrpt_line//Throughput: /}"
            printf "Baseline Throughput: %s\n" "$baseline_thrpt" >> "$COMPARE_FILE"
        fi
        
        # Add change information if available
        if [[ -n "$current_time_change" ]]; then
            printf "Time Change: %s\n" "$current_time_change" >> "$COMPARE_FILE"
            
            # Check for significant regression (>5% increase in time = slower = bad)
            # Extract the mean change value (middle value) - handle both unicode − and ASCII - minus signs
            change_line=$(echo "$current_time_change" | grep -oE '[−+\-][0-9.]+%' | sed -n '2p')
            # Remove all instances of −, +, -, and % characters
            change_value="${change_line//[−+\-%]/}"
            
            # Only perform regression analysis if we have a valid change value
            if [[ -n "$change_line" && -n "$change_value" ]]; then
                # Determine if this is positive (regression) or negative (improvement)
                if [[ "$change_line" == +* ]]; then
                    # Positive change = slower = regression
                    if (( $(echo "$change_value > 5.0" | bc -l) )); then
                        printf "⚠️  REGRESSION: Time increased by %s%% (slower performance)\n" "$change_value" >> "$COMPARE_FILE"
                        regression_found=true
                    else
                        printf "✅ OK: Time change within acceptable range\n" >> "$COMPARE_FILE"
                    fi
                elif [[ "$change_line" == −* ]] || [[ "$change_line" == -* ]]; then
                    # Negative change = faster = improvement
                    if (( $(echo "$change_value > 5.0" | bc -l) )); then
                        printf "✅ IMPROVEMENT: Time decreased by %s%% (faster performance)\n" "$change_value" >> "$COMPARE_FILE"
                    else
                        printf "✅ OK: Time change within acceptable range\n" >> "$COMPARE_FILE"
                    fi
                else
                    printf "✅ OK: Time change within acceptable range\n" >> "$COMPARE_FILE"
                fi
            else
                printf "✅ OK: Time change within acceptable range (unable to parse change value)\n" >> "$COMPARE_FILE"
            fi
        fi
        
        if [[ -n "$current_thrpt_change" ]]; then
            printf "Throughput Change: %s\n" "$current_thrpt_change" >> "$COMPARE_FILE"
        fi
        
        printf "\n" >> "$COMPARE_FILE"
        
        # Reset state variables
        current_benchmark=""
        current_points=""
        current_dimension=""
        current_time_vals=""
        current_thrpt_vals=""
        current_time_change=""
        current_thrpt_change=""
    fi
done < "$TEMP_FILE"

# Handle the final benchmark if it wasn't processed
if [[ -n "$current_benchmark" ]] && [[ -n "$current_time_change" ]]; then
    # Get baseline values for this benchmark
    baseline_time_line=$(grep -A 1 "=== $current_points Points ($current_dimension) ===" "$BASELINE_FILE" | grep "Time:" | head -1 || true)
    baseline_thrpt_line=$(grep -A 2 "=== $current_points Points ($current_dimension) ===" "$BASELINE_FILE" | grep "Throughput:" | head -1 || true)
    
    # Output the comparison section
    printf "=== %s Points (%s) ===\n" "$current_points" "$current_dimension" >> "$COMPARE_FILE"
    printf "Current Time: %s\n" "$current_time_vals" >> "$COMPARE_FILE"
    if [[ -n "$current_thrpt_vals" ]]; then
        printf "Current Throughput: %s\n" "$current_thrpt_vals" >> "$COMPARE_FILE"
    fi
    
    if [[ -n "$baseline_time_line" ]]; then
        baseline_time="${baseline_time_line//Time: /}"
        printf "Baseline Time: %s\n" "$baseline_time" >> "$COMPARE_FILE"
    fi
    
    if [[ -n "$baseline_thrpt_line" ]]; then
        baseline_thrpt="${baseline_thrpt_line//Throughput: /}"
        printf "Baseline Throughput: %s\n" "$baseline_thrpt" >> "$COMPARE_FILE"
    fi
    
    # Add change information if available
    if [[ -n "$current_time_change" ]]; then
        printf "Time Change: %s\n" "$current_time_change" >> "$COMPARE_FILE"
        
        # Check for significant regression (>5% increase in time = slower = bad)
        # Extract the mean change value (middle value) - handle both unicode − and ASCII - minus signs
        change_line=$(echo "$current_time_change" | grep -oE '[−+\-][0-9.]+%' | sed -n '2p')
        # Remove all instances of −, +, -, and % characters
        change_value="${change_line//[−+\-%]/}"
        
        # Only perform regression analysis if we have a valid change value
        if [[ -n "$change_line" && -n "$change_value" ]]; then
            # Determine if this is positive (regression) or negative (improvement)
            if [[ "$change_line" == +* ]]; then
                # Positive change = slower = regression
                if (( $(echo "$change_value > 5.0" | bc -l) )); then
                    printf "⚠️  REGRESSION: Time increased by %s%% (slower performance)\n" "$change_value" >> "$COMPARE_FILE"
                    regression_found=true
                else
                    printf "✅ OK: Time change within acceptable range\n" >> "$COMPARE_FILE"
                fi
            elif [[ "$change_line" == −* ]] || [[ "$change_line" == -* ]]; then
                # Negative change = faster = improvement
                if (( $(echo "$change_value > 5.0" | bc -l) )); then
                    printf "✅ IMPROVEMENT: Time decreased by %s%% (faster performance)\n" "$change_value" >> "$COMPARE_FILE"
                else
                    printf "✅ OK: Time change within acceptable range\n" >> "$COMPARE_FILE"
                fi
            else
                printf "✅ OK: Time change within acceptable range\n" >> "$COMPARE_FILE"
            fi
        else
            printf "✅ OK: Time change within acceptable range (unable to parse change value)\n" >> "$COMPARE_FILE"
        fi
    fi
    
    if [[ -n "$current_thrpt_change" ]]; then
        printf "Throughput Change: %s\n" "$current_thrpt_change" >> "$COMPARE_FILE"
    fi
    
    printf "\n" >> "$COMPARE_FILE"
fi

# Clean up
rm -f "$TEMP_FILE"

echo "Comparison results written to $COMPARE_FILE"

# Exit with appropriate code
if [[ "$regression_found" == "true" ]]; then
    echo "⚠️  Significant performance regressions detected!"
    exit 1
else
    echo "✅ No significant performance regressions found."
    exit 0
fi

