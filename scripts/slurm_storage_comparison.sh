#!/bin/bash
#SBATCH --job-name=delaunay-storage-comparison
#SBATCH --output=slurm-%j-storage-comparison.out
#SBATCH --error=slurm-%j-storage-comparison.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=compute

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

# Load any required modules (adjust for your cluster)
# Examples:
# module load rust/1.90.0
# module load python/3.11

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

# Ensure project is clean and dependencies are up to date
echo "🧹 Cleaning previous build artifacts..."
cargo clean
echo

echo "📦 Updating dependencies..."
cargo update
echo

# Run benchmarks with SlotMap (default backend)
echo "=================================="
echo "Phase 1: SlotMap Benchmarks"
echo "=================================="
SLOTMAP_START=$(date +%s)
echo "⏰ Start time: $(date)"
echo

if timeout "$TIMEOUT" cargo bench --bench large_scale_performance -- --save-baseline slotmap; then
	echo "✅ SlotMap benchmarks completed successfully"
else
	EXIT_CODE=$?
	if [[ $EXIT_CODE -eq 124 ]]; then
		echo "⏰ SlotMap benchmarks timed out after $TIMEOUT"
	else
		echo "❌ SlotMap benchmarks failed with exit code: $EXIT_CODE"
		exit $EXIT_CODE
	fi
fi

SLOTMAP_END=$(date +%s)
SLOTMAP_DURATION=$((SLOTMAP_END - SLOTMAP_START))
echo "⏱️  Duration: ${SLOTMAP_DURATION}s ($(date -ud "@$SLOTMAP_DURATION" +%H:%M:%S))"
echo

# Clean build artifacts before switching backends
echo "🧹 Cleaning build artifacts..."
cargo clean
echo

# Run benchmarks with DenseSlotMap backend
echo "=================================="
echo "Phase 2: DenseSlotMap Benchmarks"
echo "=================================="
DENSESLOTMAP_START=$(date +%s)
echo "⏰ Start time: $(date)"
echo

if timeout "$TIMEOUT" cargo bench --bench large_scale_performance --features dense-slotmap -- --save-baseline denseslotmap; then
	echo "✅ DenseSlotMap benchmarks completed successfully"
else
	EXIT_CODE=$?
	if [[ $EXIT_CODE -eq 124 ]]; then
		echo "⏰ DenseSlotMap benchmarks timed out after $TIMEOUT"
	else
		echo "❌ DenseSlotMap benchmarks failed with exit code: $EXIT_CODE"
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
- \`target/criterion/\` - Detailed Criterion reports
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
if [[ -d target/criterion ]]; then
	cp -r target/criterion "$ARCHIVE_DIR/"
	echo "   ✓ Copied Criterion reports"
fi

# Copy report
cp "$REPORT_FILE" "$ARCHIVE_DIR/report.md"
echo "   ✓ Copied comparison report"

# Create tarball
TARBALL="artifacts/storage-comparison-${SLURM_JOB_ID:-local}.tar.gz"
tar -czf "$TARBALL" -C artifacts "storage-comparison-${SLURM_JOB_ID:-local}"
echo "   ✓ Created archive: $TARBALL"
echo

# Print summary
TOTAL_DURATION=$((DENSESLOTMAP_END - SLOTMAP_START))
echo "=================================="
echo "Job Summary"
echo "=================================="
echo "Status: ✅ COMPLETED"
echo "Total Duration: ${TOTAL_DURATION}s ($(date -ud "@$TOTAL_DURATION" +%H:%M:%S))"
echo "Report: $REPORT_FILE"
echo "Archive: $TARBALL"
echo "End time: $(date)"
echo "=================================="

exit 0
