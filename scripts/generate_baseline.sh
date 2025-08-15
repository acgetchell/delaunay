#!/usr/bin/env bash
# generate_baseline.sh - Generate performance baseline using Criterion JSON data
#
# This script runs a fresh benchmark and creates a baseline results file
# using the working approach from test_baseline_from_criterion.sh
#
# Usage: generate_baseline.sh [--dev]
#   --dev    Use development mode with faster benchmark settings
#            (sample_size=10, measurement_time=2s, warmup_time=1s)

set -euo pipefail

# Error handling function
error_exit() {
    local message="$1"
    local code="${2:-1}"
    echo "ERROR: $message" >&2
    exit "$code"
}

# Parse command line arguments
DEV_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            DEV_MODE=true
            shift
            ;;
        -h|--help)
            echo "Usage: generate_baseline.sh [--dev]"
            echo "  --dev    Use development mode with faster benchmark settings"
            echo "           (sample_size=10, measurement_time=2s, warmup_time=1s)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use -h for help" >&2
            exit 1
            ;;
    esac
done

# Find project root
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)

# Check if hardware_info.sh dependency exists
HARDWARE_INFO_PATH="${PROJECT_ROOT}/scripts/hardware_info.sh"
if [[ ! -f "$HARDWARE_INFO_PATH" ]]; then
    error_exit "Required dependency not found: $HARDWARE_INFO_PATH. This script depends on hardware_info.sh for hardware detection."
fi

# Source the shared hardware detection utility
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/hardware_info.sh"

# Get current date and git commit
CURRENT_DATE=$(date)
GIT_COMMIT=$(git rev-parse HEAD)

# Collect hardware information using shared utility
echo "Collecting hardware information..."
HARDWARE_INFO=$(get_hardware_info)

# Output file path
OUTPUT_FILE="${PROJECT_ROOT}/benches/baseline_results.txt"

if [[ "$DEV_MODE" == "true" ]]; then
    echo "Generating baseline results in DEV MODE (faster benchmarks)..."
else
    echo "Generating baseline results from fresh benchmark run..."
fi
echo "Output file: $OUTPUT_FILE"

# Check mandatory tools
for cmd in jq bc; do
    if ! command -v "$cmd" > /dev/null 2>&1; then
        error_exit "Required command '$cmd' is not installed. Please install it to proceed."
    fi
done

# Change to project root directory for cargo commands
echo "Changing to project root: $PROJECT_ROOT"
pushd "$PROJECT_ROOT" > /dev/null || error_exit "Failed to change to project root directory: $PROJECT_ROOT"

# Set up trap to ensure we return to original directory even on error
trap 'popd > /dev/null 2>&1 || true' EXIT

#==============================================================================
# STEP 1: Clean previous benchmark results
#==============================================================================
echo "Step 1: Cleaning previous benchmark results..."

cargo clean

#==============================================================================
# STEP 2: Run fresh benchmark
#==============================================================================
echo "Step 2: Running fresh benchmark..."

if [[ "$DEV_MODE" == "true" ]]; then
    echo "Using dev mode: sample_size=10, measurement_time=2s, warmup_time=1s, no plots"
    if ! cargo bench --bench small_scale_triangulation -- --sample-size 10 --measurement-time 2 --warm-up-time 1 --noplot >/dev/null 2>&1; then
        error_exit "Failed to run benchmark in dev mode"
    fi
else
    if ! cargo bench --bench small_scale_triangulation >/dev/null 2>&1; then
        error_exit "Failed to run benchmark"
    fi
fi

#==============================================================================
# STEP 3: Create baseline results file using Criterion JSON data
#==============================================================================
echo "Step 3: Parsing Criterion results and creating baseline file..."

# Create header with hardware information
cat > "$OUTPUT_FILE" <<EOF
Date: $CURRENT_DATE
Git commit: $GIT_COMMIT
$HARDWARE_INFO
EOF

# Function to extract timing and throughput from Criterion data
extract_criterion_data() {
    local estimates_file="$1" 
    local points="$2"
    local dimension="$3"
    
    if [[ -f "$estimates_file" ]]; then
        echo "Processing $points points (${dimension}D)..."
        
        # Extract timing data (in nanoseconds from Criterion)
        local mean_ns low_ns high_ns
        mean_ns=$(jq -r '.mean.point_estimate' "$estimates_file" 2>/dev/null || echo "0")
        low_ns=$(jq -r '.mean.confidence_interval.lower_bound' "$estimates_file" 2>/dev/null || echo "$mean_ns")
        high_ns=$(jq -r '.mean.confidence_interval.upper_bound' "$estimates_file" 2>/dev/null || echo "$mean_ns")
        
        if [[ "$mean_ns" != "0" && "$mean_ns" != "null" ]]; then
            # Convert nanoseconds to microseconds
            local mean_us low_us high_us
            mean_us=$(printf "%.2f" "$(echo "scale=2; $mean_ns / 1000" | bc -l)")
            low_us=$(printf "%.2f" "$(echo "scale=2; $low_ns / 1000" | bc -l)")  
            high_us=$(printf "%.2f" "$(echo "scale=2; $high_ns / 1000" | bc -l)")
            
            # Calculate throughput in Kelem/s
            # Throughput = points / time_in_seconds
            # For time in microseconds: throughput = points / (time_us / 1,000,000) = points * 1,000,000 / time_us
            # For Kelem/s: throughput_kelem = (points * 1,000,000 / time_us) / 1000 = points * 1000 / time_us
            local thrpt_mean thrpt_low thrpt_high
            thrpt_mean=$(printf "%.3f" "$(echo "scale=3; $points * 1000 / $mean_us" | bc -l)")
            thrpt_low=$(printf "%.3f" "$(echo "scale=3; $points * 1000 / $high_us" | bc -l)")   # Lower time = higher throughput  
            thrpt_high=$(printf "%.3f" "$(echo "scale=3; $points * 1000 / $low_us" | bc -l)")  # Higher time = lower throughput
            
            # Write to baseline file in expected format
            cat >> "$OUTPUT_FILE" << EOF
=== $points Points (${dimension}D) ===
Time: [$low_us, $mean_us, $high_us] μs
Throughput: [$thrpt_low, $thrpt_mean, $thrpt_high] Kelem/s

EOF
            
            echo "  Added: Time [$low_us, $mean_us, $high_us] μs, Throughput [$thrpt_low, $thrpt_mean, $thrpt_high] Kelem/s"
        else
            echo "  Warning: Could not extract timing data from $estimates_file"
        fi
    else
        echo "  Warning: File not found: $estimates_file"
    fi
}

#==============================================================================
# STEP 4: Process existing Criterion data
#==============================================================================
echo "Step 4: Processing Criterion JSON data..."

# Create temporary file to collect benchmarks for sorting
BENCHMARKS_FILE=$(mktemp)

for dim in 2 3 4; do
    criterion_dir="$PROJECT_ROOT/target/criterion/tds_new_${dim}d/tds_new"
    
    if [[ -d "$criterion_dir" ]]; then
        echo "Processing ${dim}D benchmarks..."
        for point_dir in "$criterion_dir"/*/; do
            if [[ -d "$point_dir" ]]; then
                point_count=$(basename "$point_dir")
                
                # Look for estimates.json (prefer new/ over base/)
                estimates_file=""
                if [[ -f "${point_dir}new/estimates.json" ]]; then
                    estimates_file="${point_dir}new/estimates.json"
                elif [[ -f "${point_dir}base/estimates.json" ]]; then
                    estimates_file="${point_dir}base/estimates.json"
                fi
                
                if [[ -n "$estimates_file" ]]; then
                    # Create sort key and add to temp file: sort_key|estimates_file|point_count|dimension
                    sort_key=$(printf "%d_%03d" "$dim" "$point_count")
                    echo "$sort_key|$estimates_file|$point_count|$dim" >> "$BENCHMARKS_FILE"
                fi
            fi
        done
    fi
done

# Process benchmarks in sorted order
sort "$BENCHMARKS_FILE" | while IFS='|' read -r sort_key estimates_file point_count dimension; do
    extract_criterion_data "$estimates_file" "$point_count" "$dimension"
done

#==============================================================================
# STEP 5: Clean up and finish
#==============================================================================
echo "Step 5: Cleaning up temporary files..."
rm -f "$BENCHMARKS_FILE"

# Return to original directory
popd > /dev/null

echo ""
echo "Baseline results generated successfully!"
echo "Location: $OUTPUT_FILE"
echo ""
echo "Summary:"
benchmark_count=$(grep -c "^===" "$OUTPUT_FILE" 2>/dev/null || echo "0")
echo "Total benchmarks: $benchmark_count"
