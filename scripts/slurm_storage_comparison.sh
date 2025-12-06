#!/bin/bash
#SBATCH --job-name=delaunay-storage-comparison
#SBATCH --account=adamgrp
#SBATCH --partition=med2
#SBATCH --output=slurm-%j-storage-comparison.out
#SBATCH --error=slurm-%j-storage-comparison.err
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Storage Backend Comparison for Delaunay Triangulation
# Compares SlotMap vs DenseSlotMap performance using Phase 4 benchmarks
#
# This script operates in two modes:
# 1. SUBMISSION MODE (no SLURM_JOB_ID): Submits itself to Slurm with specified time limit
# 2. EXECUTION MODE (has SLURM_JOB_ID): Runs benchmarks with per-phase timeouts
#
# Usage:
#   ./scripts/slurm_storage_comparison.sh                    # Default time (3 days), standard benchmarks
#   ./scripts/slurm_storage_comparison.sh --large            # Default time (3 days), large-scale (4D@10K)
#   ./scripts/slurm_storage_comparison.sh --time=14-00:00:00 # Custom time (2 weeks), standard benchmarks
#   ./scripts/slurm_storage_comparison.sh --time=14-00:00:00 --large  # Custom time, large-scale
#
# Benchmark differences:
#   Standard: 4D uses [1K, 3K] points (~2-3h per backend)
#   Large:    4D uses [1K, 5K, 10K] points (~4-6h per backend)
#
# Requirements:
#   - Rust toolchain installed (rustup)
#   - uv installed for Python utilities
#   - delaunay project cloned to working directory

set -euo pipefail

# ============================================================================
# SUBMISSION MODE: Submit to Slurm if not already in a job
# ============================================================================

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
	# Not in Slurm job - parse args and submit

	TIME_LIMIT="3-00:00:00" # Default: 3 days
	LARGE_SCALE=0

	while [[ $# -gt 0 ]]; do
		case "$1" in
		--large)
			LARGE_SCALE=1 # Only affects BENCH_LARGE_SCALE, not time limit
			shift
			;;
		--time=*)
			TIME_LIMIT="${1#*=}" # Explicit time override
			shift
			;;
		--help | -h)
			cat <<'EOF'
Storage Backend Comparison - Submit to Slurm

Submits a job comparing SlotMap vs DenseSlotMap performance.

Usage:
  slurm_storage_comparison.sh [OPTIONS]

Options:
  --large              Large-scale benchmarks: 4D@10K points (vs 4D@3K standard)
                       Does NOT affect time limit - only benchmark point counts
  --time=DURATION      Custom Slurm time limit (Slurm format: D-HH:MM:SS)
                       Default: 3-00:00:00 (3 days)
  --help, -h           Show this help

Defaults:
  Time limit: 3 days (3-00:00:00)
  Benchmark scale: Standard (4D uses [1K, 3K] points, ~6h total)

Examples:
  # Standard benchmarks, 3-day limit (default)
  ./scripts/slurm_storage_comparison.sh
  
  # Large-scale benchmarks (4D@10K), 3-day limit
  ./scripts/slurm_storage_comparison.sh --large
  
  # Standard benchmarks, 1-week limit
  ./scripts/slurm_storage_comparison.sh --time=7-00:00:00
  
  # Large-scale benchmarks, 2-week limit (recommended for completion)
  ./scripts/slurm_storage_comparison.sh --large --time=14-00:00:00

Time allocation:
  Total time is split: ~50% Phase 1, ~50% Phase 2, 2h buffer for cleanup
EOF
			exit 0
			;;
		*)
			echo "❌ Error: Unknown argument: $1" >&2
			echo "Run with --help for usage" >&2
			exit 1
			;;
		esac
	done

	if ! command -v sbatch &>/dev/null; then
		echo "❌ Error: sbatch not found. Are you on a Slurm cluster?" >&2
		exit 1
	fi

	echo "🚀 Submitting storage comparison job"
	echo "   Time limit: $TIME_LIMIT"
	if [[ $LARGE_SCALE -eq 1 ]]; then
		echo "   Mode: large-scale (BENCH_LARGE_SCALE=1, 4D@10K points)"
	else
		echo "   Mode: standard (4D@3K points)"
	fi
	echo

	JOB_ARGS=()
	[[ $LARGE_SCALE -eq 1 ]] && JOB_ARGS+=(--large)

	JOB_ID=$(sbatch --time="$TIME_LIMIT" "$0" "${JOB_ARGS[@]}" | grep -oP '\d+$')

	if [[ -n "$JOB_ID" ]]; then
		echo "✅ Job submitted: $JOB_ID"
		echo "Monitor: squeue -j $JOB_ID"
		echo "Logs: tail -f slurm-${JOB_ID}-storage-comparison.out"
		exit 0
	else
		echo "❌ Failed to submit job" >&2
		exit 1
	fi
fi

# ============================================================================
# EXECUTION MODE: Running inside Slurm job
# ============================================================================

# Parse command-line arguments (passed from submission mode)
LARGE_SCALE=0
if [[ "${1:-}" == "--large" ]]; then
	LARGE_SCALE=1
	echo "🚀 Running LARGE-SCALE comparison (BENCH_LARGE_SCALE=1)"
fi

# Print job information
echo "=================================="
echo "Storage Backend Comparison Job"
echo "=================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "Memory: ${SLURM_MEM_PER_NODE:-32G}"
echo "Start time: $(date)"
echo "=================================="
echo

# Initialize module system if possible
if [[ -f /etc/profile.d/modules.sh ]]; then
	# shellcheck disable=SC1091
	source /etc/profile.d/modules.sh
elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
	# shellcheck disable=SC1091
	source /usr/share/lmod/lmod/init/bash
fi

# Load modules if the module system is available
if command -v module &>/dev/null; then
	module load rust/1.91.0 || echo "⚠️  Warning: failed to load rust/1.91.0" >&2
	module load python/3.11 || echo "⚠️  Warning: failed to load python/3.11" >&2
else
	echo "ℹ️  module command not available; using existing PATH for rust/python" >&2
fi

# Verify Rust is available
if ! command -v cargo &>/dev/null; then
	echo "❌ Error: cargo not found. Please ensure Rust is installed."
	exit 1
fi

# Verify uv is available
if ! command -v uv &>/dev/null; then
	echo "❌ Error: uv not found. Please install uv (https://github.com/astral-sh/uv)"
	exit 1
fi

# Verify timeout is available (GNU coreutils)
if ! command -v timeout &>/dev/null; then
	echo "❌ Error: timeout not found (GNU coreutils). Please install it on this node." >&2
	exit 1
fi

# Navigate to project directory (adjust if needed)
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
echo "📂 Project directory: $PROJECT_DIR"
echo

# Use node-local scratch for build artifacts if available
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
	SCRATCH_BASE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
	export CARGO_TARGET_DIR="${SCRATCH_BASE}/delaunay-target-${SLURM_JOB_ID}"
	mkdir -p "$CARGO_TARGET_DIR"
	echo "📂 Using CARGO_TARGET_DIR=$CARGO_TARGET_DIR"
fi

# Create artifacts directory
mkdir -p artifacts

# Set benchmark configuration
if [[ $LARGE_SCALE -eq 1 ]]; then
	export BENCH_LARGE_SCALE=1
	BENCHMARK_MODE="large-scale"
else
	BENCHMARK_MODE="standard"
fi

# Calculate per-phase timeout from Slurm time limit
# Extract time limit using scontrol (works for current job)
TIME_LIMIT_STR=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oP 'TimeLimit=\K[^ ]+' || echo "UNLIMITED")

if [[ "$TIME_LIMIT_STR" == "UNLIMITED" ]]; then
	PHASE_TIMEOUT_HOURS=336 # 2 weeks per phase if unlimited
	echo "ℹ️  Slurm time limit: UNLIMITED (using ${PHASE_TIMEOUT_HOURS}h per phase)"
else
	echo "ℹ️  Slurm time limit: $TIME_LIMIT_STR"

	# Parse Slurm time format: D-HH:MM:SS, HH:MM:SS, MM:SS, or MM
	if [[ "$TIME_LIMIT_STR" =~ ^([0-9]+)-([0-9]+):([0-9]+):([0-9]+)$ ]]; then
		# Days-Hours:Minutes:Seconds
		TOTAL_HOURS=$((BASH_REMATCH[1] * 24 + BASH_REMATCH[2]))
	elif [[ "$TIME_LIMIT_STR" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
		# Hours:Minutes:Seconds
		TOTAL_HOURS=${BASH_REMATCH[1]}
	elif [[ "$TIME_LIMIT_STR" =~ ^([0-9]+):([0-9]+)$ ]]; then
		# Minutes:Seconds (convert to hours)
		TOTAL_HOURS=$((BASH_REMATCH[1] / 60))
	elif [[ "$TIME_LIMIT_STR" =~ ^([0-9]+)$ ]]; then
		# Minutes only (convert to hours)
		TOTAL_HOURS=$((BASH_REMATCH[1] / 60))
	else
		echo "⚠️  Warning: Could not parse time limit '$TIME_LIMIT_STR', using 36h per phase" >&2
		TOTAL_HOURS=72
	fi

	# Reserve 2 hours for buffer/cleanup, split remaining between 2 phases
	BUFFER_HOURS=2
	AVAILABLE_HOURS=$((TOTAL_HOURS - BUFFER_HOURS))
	PHASE_TIMEOUT_HOURS=$((AVAILABLE_HOURS / 2))

	if [[ $PHASE_TIMEOUT_HOURS -lt 1 ]]; then
		echo "⚠️  Warning: Time limit too short ($TIME_LIMIT_STR), phases may not complete" >&2
		PHASE_TIMEOUT_HOURS=1
	fi

	echo "ℹ️  Calculated: ${TOTAL_HOURS}h total, ${PHASE_TIMEOUT_HOURS}h per phase"
fi

PHASE_TIMEOUT="${PHASE_TIMEOUT_HOURS}h"

echo "🔧 Configuration:"
echo "   Mode: $BENCHMARK_MODE"
echo "   Per-phase timeout: $PHASE_TIMEOUT"
echo "   Threads: ${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo

# Track status of each phase for summary
SLOTMAP_STATUS="not-run"
DENSESLOTMAP_STATUS="not-run"

# Ensure project is clean and dependencies are up to date
echo "🧹 Cleaning previous build artifacts..."
if ! cargo clean; then
	echo "⚠️  Warning: cargo clean failed (likely NFS .nfs* file still in use); continuing anyway." >&2
fi
echo

if [[ "${CARGO_UPDATE_IN_JOB:-0}" == "1" ]]; then
	echo "📦 Updating dependencies (CARGO_UPDATE_IN_JOB=1)..."
	cargo update
	echo
else
	echo "📦 Skipping cargo update (CARGO_UPDATE_IN_JOB=0)."
	echo
fi

# Run benchmarks with SlotMap (default backend)
echo "=================================="
echo "Phase 1: SlotMap Benchmarks"
echo "=================================="
SLOTMAP_STATUS="pending"
SLOTMAP_START=$(date +%s)
echo "⏰ Start time: $(date)"
echo

if timeout "$PHASE_TIMEOUT" cargo bench --bench large_scale_performance -- --save-baseline slotmap; then
	echo "✅ SlotMap benchmarks completed successfully"
	SLOTMAP_STATUS="ok"
else
	EXIT_CODE=$?
	if [[ $EXIT_CODE -eq 124 ]]; then
		echo "⏰ SlotMap benchmarks timed out after $PHASE_TIMEOUT"
		SLOTMAP_STATUS="timeout"
	else
		echo "❌ SlotMap benchmarks failed with exit code: $EXIT_CODE"
		SLOTMAP_STATUS="failed"
		exit $EXIT_CODE
	fi
fi

SLOTMAP_END=$(date +%s)
SLOTMAP_DURATION=$((SLOTMAP_END - SLOTMAP_START))
echo "⏱️  Duration: ${SLOTMAP_DURATION}s ($(date -ud "@$SLOTMAP_DURATION" +%H:%M:%S))"
echo

# Preserve SlotMap baselines before cleaning (cargo clean would wipe them)
echo "💾 Preserving SlotMap baselines..."
SLOTMAP_BACKUP="artifacts/slotmap-baselines-${SLURM_JOB_ID:-local}"
mkdir -p "$SLOTMAP_BACKUP"
CRIT_DIR="${CARGO_TARGET_DIR:-target}/criterion"
if [[ -d "$CRIT_DIR" ]]; then
	cp -r "$CRIT_DIR" "$SLOTMAP_BACKUP/"
	echo "   ✓ Backed up to $SLOTMAP_BACKUP/criterion"
else
	echo "⚠️  Warning: No criterion directory at $CRIT_DIR" >&2
fi
echo

# Clean build artifacts before switching backends
echo "🧹 Cleaning previous build artifacts..."
if ! cargo clean; then
	echo "⚠️  Warning: cargo clean failed (likely NFS .nfs* file still in use); continuing anyway." >&2
fi
echo

# Run benchmarks with DenseSlotMap backend
echo "=================================="
echo "Phase 2: DenseSlotMap Benchmarks"
echo "=================================="
DENSESLOTMAP_STATUS="pending"
DENSESLOTMAP_START=$(date +%s)
echo "⏰ Start time: $(date)"
echo

if timeout "$PHASE_TIMEOUT" cargo bench --bench large_scale_performance --features dense-slotmap -- --save-baseline denseslotmap; then
	echo "✅ DenseSlotMap benchmarks completed successfully"
	DENSESLOTMAP_STATUS="ok"
else
	EXIT_CODE=$?
	if [[ $EXIT_CODE -eq 124 ]]; then
		echo "⏰ DenseSlotMap benchmarks timed out after $PHASE_TIMEOUT"
		DENSESLOTMAP_STATUS="timeout"
	else
		echo "❌ DenseSlotMap benchmarks failed with exit code: $EXIT_CODE"
		DENSESLOTMAP_STATUS="failed"
		exit $EXIT_CODE
	fi
fi

DENSESLOTMAP_END=$(date +%s)
DENSESLOTMAP_DURATION=$((DENSESLOTMAP_END - DENSESLOTMAP_START))
echo "⏱️  Duration: ${DENSESLOTMAP_DURATION}s ($(date -ud "@$DENSESLOTMAP_DURATION" +%H:%M:%S))"
echo

# Generate comparison report using Python utility
echo "=================================="
echo "Phase 3: Generate Comparison Report"
echo "=================================="
echo "📊 Analyzing results..."
echo

REPORT_FILE="artifacts/storage_comparison_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S).md"

# Note: The Python script reads from target/criterion, which contains results from both runs
# We'll generate a custom report since we used --save-baseline
# shellcheck disable=SC2016  # Backticks are markdown syntax, not command substitution
cat >"$REPORT_FILE" <<EOF
# Storage Backend Comparison Report

**Job ID**: ${SLURM_JOB_ID:-local}
**Node**: ${SLURM_NODELIST:-$(hostname)}
**Mode**: $BENCHMARK_MODE
**Generated**: $(date -u '+%Y-%m-%d %H:%M:%S UTC')

## Benchmark Duration

- **SlotMap**: ${SLOTMAP_DURATION}s ($(date -ud "@$SLOTMAP_DURATION" +%H:%M:%S))
- **DenseSlotMap**: ${DENSESLOTMAP_DURATION}s ($(date -ud "@$DENSESLOTMAP_DURATION" +%H:%M:%S))
- **Total**: $((SLOTMAP_DURATION + DENSESLOTMAP_DURATION))s

## Comparison

Use Criterion's built-in comparison tools:

\`\`\`bash
# Compare baselines
critcmp slotmap denseslotmap

# Or use cargo-criterion if available
cargo criterion --baseline slotmap --load-baseline denseslotmap
\`\`\`

## Artifacts

Benchmark results saved in:
- \`criterion/\` under the job's \`CARGO_TARGET_DIR\` - Detailed Criterion reports
- Baselines: \`slotmap\` and \`denseslotmap\` (both preserved and merged)
- Note: SlotMap baselines were backed up before \`cargo clean\` and restored for comparison

---
*Generated by slurm_storage_comparison.sh*
EOF

echo "✅ Report saved: $REPORT_FILE"
echo

# Try to use critcmp if available for detailed comparison
if command -v critcmp &>/dev/null; then
	echo "📊 Generating detailed comparison with critcmp..."
	critcmp slotmap denseslotmap | tee -a "$REPORT_FILE"
	echo
else
	echo "ℹ️  critcmp not found. Install with: cargo install critcmp"
	echo "   To compare results later, run: critcmp slotmap denseslotmap"
	echo
fi

# Archive results
ARCHIVE_DIR="artifacts/storage-comparison-${SLURM_JOB_ID:-local}"
mkdir -p "$ARCHIVE_DIR"

echo "📦 Archiving results..."

# Restore SlotMap baselines (were backed up before cargo clean)
SLOTMAP_BACKUP="artifacts/slotmap-baselines-${SLURM_JOB_ID:-local}"
if [[ -d "$SLOTMAP_BACKUP/criterion" ]]; then
	echo "   ✓ Restoring SlotMap baselines from backup..."
	# Merge SlotMap baselines into current criterion directory
	CRIT_DIR="${CARGO_TARGET_DIR:-target}/criterion"
	mkdir -p "$CRIT_DIR"
	cp -rn "$SLOTMAP_BACKUP/criterion/." "$CRIT_DIR/" # -n: no-clobber (preserve DenseSlotMap)
	echo "   ✓ Merged SlotMap + DenseSlotMap baselines"
else
	echo "⚠️  Warning: SlotMap backup not found at $SLOTMAP_BACKUP" >&2
fi

# Copy merged Criterion results (now contains both baselines)
CRIT_DIR="${CARGO_TARGET_DIR:-target}/criterion"
if [[ -d "$CRIT_DIR" ]]; then
	cp -r "$CRIT_DIR" "$ARCHIVE_DIR/"
	echo "   ✓ Copied Criterion reports (both backends) from $CRIT_DIR"
else
	echo "ℹ️  No Criterion directory found at $CRIT_DIR; skipping copy."
fi

# Copy report
cp "$REPORT_FILE" "$ARCHIVE_DIR/report.md"
echo "   ✓ Copied comparison report"

# Create tarball
TARBALL="artifacts/storage-comparison-${SLURM_JOB_ID:-local}.tar.gz"
if tar -czf "$TARBALL" -C artifacts "storage-comparison-${SLURM_JOB_ID:-local}"; then
	echo "   ✓ Created archive: $TARBALL"
else
	echo "⚠️  Warning: failed to create archive $TARBALL" >&2
fi

# Print summary
TOTAL_DURATION=$((DENSESLOTMAP_END - SLOTMAP_START))
echo "=================================="
echo "Job Summary"
echo "=================================="
echo "Status:"
echo "  SlotMap:      $SLOTMAP_STATUS"
echo "  DenseSlotMap: $DENSESLOTMAP_STATUS"
OVERALL_STATUS="COMPLETED"
if [[ "$SLOTMAP_STATUS" != "ok" || "$DENSESLOTMAP_STATUS" != "ok" ]]; then
	OVERALL_STATUS="COMPLETED_WITH_ISSUES"
fi
echo "Overall: $OVERALL_STATUS"
echo "Total Duration: ${TOTAL_DURATION}s ($(date -ud "@$TOTAL_DURATION" +%H:%M:%S))"
echo "Report: $REPORT_FILE"
echo "Archive: $TARBALL"
echo "End time: $(date)"
echo "=================================="

exit 0
