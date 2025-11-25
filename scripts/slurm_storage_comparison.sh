#!/bin/bash -l
#SBATCH --job-name=delaunay-storage-comparison
#SBATCH --account=adamgrp
#SBATCH --partition=med2
#SBATCH --output=slurm-%j-storage-comparison.out
#SBATCH --error=slurm-%j-storage-comparison.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Storage Backend Comparison for Delaunay Triangulation
# Compares SlotMap vs DenseSlotMap performance using Phase 4 benchmarks
#
# Usage:
#   sbatch scripts/slurm_storage_comparison.sh
#
# Or with large-scale benchmarks:
#   sbatch --time=24:00:00 scripts/slurm_storage_comparison.sh --large
#
# Requirements:
#   - Rust toolchain installed (rustup)
#   - uv installed for Python utilities
#   - delaunay project cloned to working directory

set -euo pipefail

# Parse command-line arguments
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
	TIMEOUT="24h"
else
	BENCHMARK_MODE="standard"
	TIMEOUT="12h"
fi

echo "🔧 Configuration:"
echo "   Mode: $BENCHMARK_MODE"
echo "   Timeout: $TIMEOUT"
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

if timeout "$TIMEOUT" cargo bench --bench large_scale_performance -- --save-baseline slotmap; then
	echo "✅ SlotMap benchmarks completed successfully"
	SLOTMAP_STATUS="ok"
else
	EXIT_CODE=$?
	if [[ $EXIT_CODE -eq 124 ]]; then
		echo "⏰ SlotMap benchmarks timed out after $TIMEOUT"
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

if timeout "$TIMEOUT" cargo bench --bench large_scale_performance --features dense-slotmap -- --save-baseline denseslotmap; then
	echo "✅ DenseSlotMap benchmarks completed successfully"
	DENSESLOTMAP_STATUS="ok"
else
	EXIT_CODE=$?
	if [[ $EXIT_CODE -eq 124 ]]; then
		echo "⏰ DenseSlotMap benchmarks timed out after $TIMEOUT"
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
- Baselines: \`slotmap\` and \`denseslotmap\`

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
# Copy Criterion results
CRIT_DIR="${CARGO_TARGET_DIR:-target}/criterion"
if [[ -d "$CRIT_DIR" ]]; then
	cp -r "$CRIT_DIR" "$ARCHIVE_DIR/"
	echo "   ✓ Copied Criterion reports from $CRIT_DIR"
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
