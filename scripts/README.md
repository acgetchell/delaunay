# Scripts Directory

This directory contains utility scripts for building, testing, and benchmarking the delaunay library.

**Note**: Tests for the Python utilities are located in `scripts/tests/` and can be run with `uv run pytest`.

## Prerequisites

Before running these scripts, ensure you have the following dependencies installed:

### Python 3.11+ (Required)

```bash
# Install Python 3.11+ and uv package manager
brew install python@3.11 uv  # macOS with Homebrew
# or follow installation instructions for your platform
```

### Additional Dependencies

```bash
# macOS (using Homebrew)
brew install jq findutils coreutils

# Ubuntu/Debian 
sudo apt-get install -y jq
```

## Scripts Overview

### Python Utilities (Primary)

All Python utilities require Python 3.11+ and support `--help` for detailed usage. The project uses modern Python with comprehensive utilities for
benchmarking, changelog management, and hardware detection.

**Available Commands**:

- `uv run benchmark-utils` - Performance baseline generation and comparison
- `uv run changelog-utils` - Enhanced changelog generation with AI commit processing and git tagging
- `uv run hardware-utils` - Cross-platform hardware information detection
- `uv run enhance-commits` - AI-powered commit message enhancement (internal utility)

#### `benchmark_utils.py` 🐍

**Purpose**: Complete benchmark parsing, baseline generation, and performance comparison utilities.

**Features**:

- **Criterion JSON Parsing**: Direct parsing of Criterion's estimates.json for accuracy
- **Baseline Generation**: `generate-baseline` command with git metadata
- **Performance Comparison**: `compare` command with regression detection (>5% threshold)
- **Flexible Baseline Formats**: Handles standard and tag-specific baseline file naming patterns
- **Automatic File Conversion**: Converts tag-specific baselines to standard format for compatibility
- **Hardware Integration**: Automatic hardware info inclusion and comparison
- **Development Mode**: `--dev` flag for faster benchmarks (10x speedup)
- **Timezone-Aware Dating**: Proper timezone handling for timestamps
- **Modern Python**: Python 3.11+ with type hints and union syntax

**Commands**:

```bash
# Generate performance baseline
uv run benchmark-utils generate-baseline [--dev] [--output FILE]

# Compare against baseline 
uv run benchmark-utils compare --baseline FILE [--dev] [--output FILE]
```

**Output Format**:

```bash
# Baseline file format:
=== 10 Points (2D) ===
Time: [354.30, 356.10, 357.91] μs
Throughput: [28.135, 28.257, 28.381] Kelem/s

# Comparison file format:
Current Time: [338.45, 340.12, 341.78] μs
Baseline Time: [336.95, 338.61, 340.26] μs
Time Change: [+0.45%, +0.45%, +0.45%]
✅ OK: Time change within acceptable range
```

**Dependencies**: Python 3.11+, `hardware_utils.py`

**Regression Testing Workflow Commands**:

```bash
# Prepare downloaded baseline artifact (handles tag-specific files)
uv run benchmark-utils prepare-baseline [--baseline-dir DIR]

# Extract commit SHA from baseline artifact
uv run benchmark-utils extract-baseline-commit [--baseline-dir DIR]

# Determine if benchmarks should be skipped based on changes
uv run benchmark-utils determine-skip --baseline-commit SHA --current-commit SHA

# Run performance regression test
uv run benchmark-utils run-regression-test --baseline FILE

# Display regression test results
uv run benchmark-utils display-results [--results FILE]

# Generate regression testing summary
uv run benchmark-utils regression-summary
```

**Baseline File Compatibility**:

- **Standard format**: `baseline_results.txt` (always supported)
- **Tag-specific format**: `baseline-vX.Y.Z.txt` (automatically converted to standard format)
- **Generic format**: `baseline*.txt` (fallback for any baseline file)
- **Metadata support**: Uses `metadata.json` when baseline files lack commit info

---

#### `changelog_utils.py` 🐍

**Purpose**: Comprehensive changelog management tool with AI commit processing and Keep a Changelog categorization.

**Features**:

- **Enhanced Changelog Generation**: Creates changelogs with commit dates instead of tag creation dates
- **AI Commit Processing**: Uses `enhance_commits.py` for intelligent commit categorization
- **Git Tag Management**: Creates git tags with changelog content as tag messages
- **Squashed PR Expansion**: Advanced parsing of squashed PR commits to extract detailed commit message bodies
- **Multi-format Support**: Handles various commit message formats and bullet styles
- **Cross-platform Compatibility**: Works consistently across Windows, macOS, and Linux
- **Comprehensive Error Handling**: Clear error messages and usage instructions

**Commands**:

```bash
# Generate enhanced changelog (default command)
uv run changelog-utils
uv run changelog-utils generate

# Generate changelog with debug output (keeps intermediate files)
uv run changelog-utils generate --debug

# Create git tag with changelog content as message
uv run changelog-utils tag vX.Y.Z

# Force recreate existing tag
uv run changelog-utils tag vX.Y.Z --force
```

**Enhanced Features**:

- **Accurate Dating**: Shows when development work was actually completed
- **Squashed PR Expansion**: Extracts bullet points and descriptions from squashed commits
- **AI Categorization**: Uses Keep a Changelog format (Added/Changed/Fixed/Removed/Deprecated/Security)
- **GitHub Integration**: Tag messages work with `gh release create --notes-from-tag`

**Dependencies**: Python 3.11+, `enhance_commits.py`, `subprocess_utils.py`, `git-cliff`

---

#### `hardware_utils.py` 🐍

**Purpose**: Cross-platform hardware information detection and comparison.

**Features**:

- **Cross-platform**: macOS, Linux, Windows detection
- **Hardware Detection**: CPU (model, cores, threads), memory, Rust toolchain
- **Output Formats**: Formatted display (`info`), key=value pairs (`kv`), or JSON format
- **Baseline Comparison**: Hardware compatibility warnings
- **Modern Architecture Support**: Enhanced ARM/heterogeneous core detection

**Commands**:

```bash
# Display formatted hardware information
uv run hardware-utils info

# Display as key=value pairs
uv run hardware-utils kv

# Display as JSON
uv run hardware-utils info --json

# Compare with baseline file
uv run hardware-utils compare --baseline-file FILE
```

**Output Format**:

```bash
# Formatted output:
Hardware Information:
  OS: macOS
  CPU: Apple M2 Pro
  CPU Cores: 10
  CPU Threads: 10
  Memory: 16.0 GB
  Rust: rustc 1.89.0
  Target: aarch64-apple-darwin
```

**Dependencies**: Python 3.11+, `subprocess_utils.py`, system tools (`sysctl`, `lscpu`, PowerShell)

---

#### `enhance_commits.py` 🐍

**Purpose**: AI-powered commit message enhancement with Keep a Changelog categorization.

**Features**:

- **Keep a Changelog Format**: Categorizes commits as Added/Changed/Fixed/Removed/Deprecated/Security
- **Pattern Matching**: Advanced regex patterns for accurate categorization
- **Markdown Processing**: Handles markdown formatting and line wrapping
- **Internal Utility**: Used by `changelog_utils.py` for AI-enhanced changelog generation

**Usage**: This is an internal utility called by `changelog-utils`. Not typically used directly.

**Dependencies**: Python 3.11+

---

#### `subprocess_utils.py` 🐍

**Purpose**: Secure subprocess utilities for all Python scripts providing security-hardened subprocess execution.

**Features**:

- **Secure Execution**: Uses full executable paths instead of command names
- **Executable Validation**: Validates executables exist before running
- **Consistent Error Handling**: Standardized error handling across all utilities
- **Security Mitigation**: Addresses security vulnerabilities flagged by static analysis
- **Git Integration**: Convenient wrappers for common git operations

**Key Functions**:

- `get_safe_executable(command)` - Get validated full path to executable
- `run_safe_command(command, args, **kwargs)` - Secure subprocess execution
- `run_git_command(args, **kwargs)` - Git-specific secure execution
- `run_cargo_command(args, **kwargs)` - Cargo-specific secure execution
- `check_git_repo()` - Validate git repository
- `check_git_history()` - Validate git history exists

**Usage**: This is a shared library used by all other Python utilities. Not typically used directly.

**Dependencies**: Python 3.11+ standard library

---

#### `compare_storage_backends.py` 🐍

**Purpose**: Compare SlotMap vs DenseSlotMap storage backend performance for Phase 4 evaluation.

**Features**:

- **Automated Comparison**: Runs benchmarks with both backends and generates detailed reports
- **Criterion Integration**: Parses Criterion output (JSON and text) for robust comparison
- **Performance Metrics**: Analyzes construction time, iteration speed, query performance, and validation overhead
- **Memory Tracking**: Reports RSS memory usage internally during benchmarks
- **Summary Statistics**: Calculates average, best-case, and worst-case performance differences
- **Development Mode**: Fast iteration with reduced scale (`--dev` flag)
- **Markdown Reports**: Professional comparison reports with tables and recommendations

**Commands**:

```bash
# Run comparison with default settings
uv run compare-storage-backends

# Quick comparison (development mode)
uv run compare-storage-backends --dev

# Custom output location
uv run compare-storage-backends --output artifacts/storage_comparison.md

# Specify benchmark to run
uv run compare-storage-backends --bench large_scale_performance

# Filter specific benchmarks
uv run compare-storage-backends --filter "construction/3D"
```

**Report Contents**:

- Performance comparison table with percentage differences
- Summary statistics (average, best/worst case)
- Recommendations based on results
- Reproduction instructions

**Dependencies**: Python 3.11+, `subprocess_utils.py`

---

### Shell Scripts (Specialized)

#### `slurm_storage_comparison.sh`

**Purpose**: Slurm HPC script for comprehensive SlotMap vs DenseSlotMap storage backend comparison.

This script benchmarks the library's two storage backend options (SlotMap and DenseSlotMap) on high-performance computing clusters using the Slurm
workload manager. It runs the `large_scale_performance` benchmark suite with each backend and generates detailed comparison reports.

**Features**:

- **Automated 3-phase execution**: SlotMap benchmarks → DenseSlotMap benchmarks → Analysis
- **Dual submission modes**: Self-submitting with `sbatch` or direct execution within Slurm job
- **Baseline saving**: Uses `--save-baseline` for precise Criterion comparisons with `critcmp`
- **Smart timeout management**: Automatically calculates per-phase timeouts from Slurm time limit
- **Build isolation**: Uses node-local scratch (`$SLURM_TMPDIR`) for fast compilation
- **Baseline preservation**: Backs up SlotMap results before `cargo clean` to enable comparison
- **Progress tracking**: Detailed timing and status for each phase
- **Artifact archiving**: Packages all results in timestamped tarball with merged baselines
- **critcmp integration**: Automatic detailed comparison if available
- **Error handling**: Per-phase timeout protection with status tracking

**Usage**:

```bash
# Standard comparison (4D with 1K, 3K points)
./scripts/slurm_storage_comparison.sh
sbatch scripts/slurm_storage_comparison.sh

# Large-scale comparison (4D with 1K, 5K, 10K points)
./scripts/slurm_storage_comparison.sh --large

# Custom time limit (default: 3 days)
./scripts/slurm_storage_comparison.sh --time=7-00:00:00

# Use specific partition (default: med2)
./scripts/slurm_storage_comparison.sh --partition=high2

# Use specific account (default: adamgrp)
./scripts/slurm_storage_comparison.sh --account=myaccount

# Large-scale with extended time (recommended for completion)
./scripts/slurm_storage_comparison.sh --large --time=14-00:00:00

# Combine multiple options
./scripts/slurm_storage_comparison.sh --partition=high2 --account=myaccount --large --time=7-00:00:00

# Help information
./scripts/slurm_storage_comparison.sh --help
```

**Submission Modes**:

1. **Self-submission** (no `SLURM_JOB_ID`): Submits itself to Slurm with specified options
2. **Direct execution** (inside Slurm job): Runs the benchmark workflow

The script automatically detects which mode to use, making it easy to submit jobs without writing separate submission scripts.

**Command-line Options**:

- `--large`: Enable large-scale benchmarks (4D with 1K, 5K, 10K points)
- `--time=DURATION`: Custom Slurm time limit (format: D-HH:MM:SS, default: 3-00:00:00)
- `--partition=NAME`: Slurm partition/queue to use (default: med2)
- `--account=NAME`: Slurm account/allocation to use (default: adamgrp)
- `--help, -h`: Show detailed help information

**Benchmark Scale**:

- **Standard** (default): 4D triangulations use [1K, 3K] points (~2-3h per backend, ~6h total)
- **Large** (`--large`): 4D triangulations use [1K, 5K, 10K] points (~4-6h per backend, ~12h total)

The `--large` flag sets `BENCH_LARGE_SCALE=1`, which is read by the benchmark suite to enable larger point counts for more comprehensive performance testing.

**Time Management**:

```bash
# Automatic timeout calculation from Slurm time limit:
# - Reserves 2 hours for cleanup/buffer
# - Splits remaining time equally between Phase 1 (SlotMap) and Phase 2 (DenseSlotMap)
# - Example: 3-day limit → ~34h per phase

# View calculated timeouts in job output:
squeue -j <job-id>
tail -f slurm-<job-id>-storage-comparison.out
```

**Prerequisites**:

- Slurm workload manager
- Rust toolchain (rustup) - loads `rust/1.93.0` module if available
- uv package manager
- GNU coreutils (`timeout` command)
- critcmp (optional but recommended): `cargo install critcmp`

**Cluster Configuration**:

You can configure Slurm options in two ways:

1. **Command-line arguments** (recommended):

```bash
./scripts/slurm_storage_comparison.sh --account=your_account --partition=your_partition --time=7-00:00:00
```

2. **Edit script header** (for permanent defaults):

```bash
#SBATCH --account=your_account     # Billing account (default: adamgrp)
#SBATCH --partition=your_partition # Compute partition (default: med2)
#SBATCH --time=3-00:00:00         # Job time limit (3 days default)
#SBATCH --cpus-per-task=8         # CPU cores
#SBATCH --mem=32G                 # Memory allocation
```

Command-line arguments override the script header defaults.

**Output Files**:

```text
artifacts/
├── storage_comparison_<job-id>_<timestamp>.md   # Main comparison report
├── storage-comparison-<job-id>/                 # Full archive directory
│   ├── criterion/                               # Merged Criterion reports
│   │   ├── <benchmark>/                         # Per-benchmark results
│   │   │   ├── slotmap/                         # SlotMap baseline
│   │   │   ├── denseslotmap/                    # DenseSlotMap baseline
│   │   │   └── report/                          # HTML reports
│   └── report.md                                # Report copy
└── storage-comparison-<job-id>.tar.gz           # Compressed archive

slurm-<job-id>-storage-comparison.out            # Job stdout (progress log)
slurm-<job-id>-storage-comparison.err            # Job stderr (errors/warnings)
```

**Analysis Workflow**:

```bash
# 1. Monitor job progress
squeue -j <job-id>                               # Check job status
tail -f slurm-<job-id>-storage-comparison.out   # Live progress

# 2. After completion, view report on cluster
cat artifacts/storage_comparison_<job-id>_<timestamp>.md

# 3. Use critcmp for detailed comparison (if installed)
critcmp slotmap denseslotmap

# 4. Download results for local analysis
scp cluster:/path/to/artifacts/storage-comparison-<job-id>.tar.gz .
tar -xzf storage-comparison-<job-id>.tar.gz
cd storage-comparison-<job-id>

# 5. View HTML reports in browser
open criterion/*/report/index.html              # macOS
xdg-open criterion/*/report/index.html          # Linux

# 6. Compare with critcmp locally (requires criterion directory)
critcmp slotmap denseslotmap
```

**Understanding Results**:

The comparison report includes:

- **Job metadata**: Job ID, node, mode, duration for each phase
- **Baseline locations**: Paths to saved Criterion baselines
- **critcmp output**: Detailed performance comparison (if available)
- **Status tracking**: Success/timeout/failure for each phase

Criterion baselines are saved as:

- `denseslotmap`: Default DenseSlotMap backend (enabled by default)
- `slotmap`: SlotMap backend (run with `--no-default-features`)

These can be compared using `critcmp slotmap denseslotmap` or Criterion's CLI tools.

**Common Issues**:

1. **Module loading failures**: Script continues with PATH-based Rust/Python if modules unavailable
2. **NFS .nfs* files**: `cargo clean` warnings are expected on shared filesystems, script continues
3. **Timeout before completion**: Increase time limit with `--time=` or use standard mode instead of `--large`
4. **Missing critcmp**: Install with `cargo install critcmp` for detailed comparison output

**Environment Variables**:

- `BENCH_LARGE_SCALE=1`: Automatically set when using `--large` flag
- `CARGO_TARGET_DIR`: Set to node-local scratch for faster builds
- `CARGO_UPDATE_IN_JOB=1`: Optional, runs `cargo update` before benchmarks (default: skip)
- `PROJECT_DIR`: Project root directory (default: current directory)

**Related**: See [Issue #74](https://github.com/acgetchell/delaunay/issues/74) for Phase 4 storage backend evaluation.

**Dependencies**: Slurm, Rust toolchain, cargo, uv, GNU coreutils, optional critcmp

---

#### `run_all_examples.sh`

**Purpose**: Executes all example programs in the project to verify functionality.

**Features**:

- Automatically discovers all examples in the `examples/` directory
- Runs examples in release mode for representative performance
- Creates results directory structure

**Usage**:

```bash
./scripts/run_all_examples.sh
```

**Dependencies**: Requires `cargo`, `find`, `sort` (GNU sort preferred but not required)

---

#### Git Tagging from Changelog (Python-based)

**Purpose**: Creates git tags with changelog content as tag messages for seamless GitHub release integration.

**Modern Implementation**: Uses Python utilities instead of shell scripts for better cross-platform compatibility and maintainability.

**Usage**:

```bash
# Create new tag with changelog content
uv run changelog-utils tag vX.Y.Z

# Force recreate existing tag
uv run changelog-utils tag vX.Y.Z --force

# Show help information
uv run changelog-utils tag --help
```

**Features**:

- **Automatic changelog extraction**: Parses CHANGELOG.md to find version-specific content
- **Multiple version formats**: Supports `## [X.Y.Z]`, `## vX.Y.Z`, and `## X.Y.Z` headers
- **GitHub release integration**: Tag messages work with `gh release create --notes-from-tag`
- **Safety checks**: Validates git repository, changelog existence, and version format
- **Force recreation**: Option to recreate existing tags with `--force` flag
- **Smart content extraction**: Removes headers and cleans whitespace automatically
- **Preview functionality**: Shows tag message preview before creation
- **Comprehensive error handling**: Clear error messages and usage instructions
- **Cross-platform compatibility**: Works consistently across Windows, macOS, and Linux

**Integration with GitHub Releases**:

```bash
# Workflow for GitHub releases:
1. Create tag with changelog content:
   uv run changelog-utils tag vX.Y.Z

2. Push tag to remote:
   git push origin vX.Y.Z

3. Create GitHub release using tag message:
   gh release create vX.Y.Z --notes-from-tag
```

**Advanced Usage**:

```bash
# The changelog-utils tool also supports changelog generation:
uv run changelog-utils generate           # Generate enhanced changelog
uv run changelog-utils generate --debug   # Keep intermediate files for debugging
```

**Dependencies**: Requires Python 3.11+, `uv`, `git`, and access to CHANGELOG.md

---

## Workflow Examples

### Performance Baseline Setup (One-time)

```bash
# 1. Generate initial performance baseline
uv run benchmark-utils generate-baseline

# 2. Commit baseline for CI regression testing
git add benches/baseline_results.txt
git commit -m "Add performance baseline for CI regression testing"
```

### Performance Regression Testing (Development)

```bash
# 1. Make code changes
# ... your modifications ...

# 2. Test for performance regressions
uv run benchmark-utils compare --baseline benches/baseline_results.txt

# 3. Review results in benches/compare_results.txt
# 4. If regressions are acceptable, update baseline:
uv run benchmark-utils generate-baseline
git add benches/baseline_results.txt
git commit -m "Update performance baseline after optimization"
```

### Fast Development Workflow (Development Mode)

```bash
# Quick iteration during development using --dev flag
# (Reduces benchmark time from ~10 minutes to ~30 seconds)

# 1. Make code changes
# ... your modifications ...

# 2. Quick performance check
uv run benchmark-utils compare --baseline benches/baseline_results.txt --dev

# 3. If major changes needed, generate new dev baseline:
uv run benchmark-utils generate-baseline --dev

# 4. Final validation with full benchmarks before commit:
uv run benchmark-utils generate-baseline          # Full baseline
uv run benchmark-utils compare --baseline benches/baseline_results.txt         # Full comparison
```

**Development Mode Benefits**:

- **10x faster**: Reduces sample size and measurement time
- **Quick feedback**: Ideal for iterative development
- **Same accuracy**: Still detects significant performance changes
- **Settings**: `sample_size=10, measurement_time=2s, warmup_time=1s`

### Changelog Generation Workflow

```bash
# 1. Make commits and create git tags
git tag vX.Y.Z
git push origin vX.Y.Z

# 2. Generate updated changelog with accurate commit dates and AI enhancement
uv run changelog-utils generate

# 3. Review and commit the updated changelog
git add CHANGELOG.md
git commit -m "Update changelog with AI enhancement for vX.Y.Z"
git push origin main
```

### Git Tagging from Changelog

```bash
# Create new tag with changelog content for GitHub releases
uv run changelog-utils tag vX.Y.Z

# Force recreate existing tag
uv run changelog-utils tag vX.Y.Z --force

# Push tag and create GitHub release
git push origin vX.Y.Z
gh release create vX.Y.Z --notes-from-tag
```

**Benefits of Using changelog-utils**:

- **Accurate Dating**: Shows when development work was actually completed
- **AI Enhancement**: Categorizes commits using Keep a Changelog format
- **Squashed PR Expansion**: Extracts detailed information from squashed commits
- **Professional Presentation**: Avoids all releases showing the same tag creation date
- **GitHub Integration**: Seamless integration with GitHub releases

### Manual Benchmark Analysis

```bash
# 1. Run benchmarks directly (CI performance suite)
cargo bench --bench ci_performance_suite

# 2. Generate new baseline
uv run benchmark-utils generate-baseline

# 3. Compare against previous baseline
uv run benchmark-utils compare --baseline benches/baseline_results.txt
```

**CI Performance Suite**: The benchmark utilities now use `benches/ci_performance_suite.rs` for CI/CD-optimized performance testing:

- **Dimensions**: 2D, 3D, 4D, and 5D triangulations.
- **Point counts**: [10, 25, 50].
- **Runtime**: ~5–10 minutes.
- **Coverage**: Core triangulation performance across all supported dimensions.

**Migration Notes**:

- The CI performance suite now includes 2D triangulations for comprehensive coverage
- Existing baselines remain compatible as the CI suite maintains the same benchmark format
- Development workflow unchanged - use `--dev` flag for fast iteration

### Continuous Integration

The repository includes automated performance regression testing via GitHub Actions:

#### Automated baseline generation

- **Workflow file**: `.github/workflows/generate-baseline.yml`
- **Trigger**: Automatic on git tag creation
- **Artifacts**: Creates performance baseline artifacts for download
- **Integration**: Baselines are automatically available for benchmark comparisons

#### Separate Benchmark Workflow

- **Workflow file**: `.github/workflows/benchmarks.yml`
- **Trigger conditions**:
  - Manual trigger (`workflow_dispatch`)
  - Pushes to `main` branch affecting performance-critical files
  - Changes to `src/`, `benches/`, `Cargo.toml`, `Cargo.lock`

#### CI Behavior

```bash
# If baseline exists:
# 1. Downloads baseline from artifacts (performance-baseline-vX.Y.Z)
# 2. Automatically converts tag-specific files (baseline-vX.Y.Z.txt → baseline_results.txt)
# 3. Runs uv run benchmark-utils run-regression-test --baseline baseline-artifact/baseline_results.txt
# 4. Flags regressions (sets BENCHMARK_REGRESSION_DETECTED); CI may fail in a later step if configured
# 5. Uploads comparison results as artifacts

# If no baseline exists:
# 1. Logs instructions for creating baseline
# 2. Skips regression testing (does not fail CI)
# 3. Suggests creating a git tag to generate baseline automatically
```

#### CI Integration Benefits

- **Automated baseline management**: No manual baseline commits needed
- **Flexible baseline formats**: Handles both standard (`baseline_results.txt`) and tag-specific (`baseline-vX.Y.Z.txt`) naming
- **Automatic normalization**: Converts tag-specific artifacts to standard format for compatibility
- **Separate from main CI**: Avoids slowing down regular development workflow
- **Environment consistency**: Uses macOS runners (Apple Silicon) for reproducible benchmark comparisons
- **Smart triggering**: Only runs on changes that could affect performance
- **Graceful degradation**: Skips if baseline missing, with clear setup instructions
- **Artifact collection**: Stores benchmark results for historical analysis

## Error Handling and Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install required packages using your system's package manager
2. **Permission Errors**: Ensure scripts are executable with `chmod +x scripts/*.sh`
3. **Path Issues**: Run scripts from the project root directory
4. **Missing Baseline**: Create a git tag to automatically generate baseline via CI, or run `uv run benchmark-utils generate-baseline` locally
5. **Python Version**: Ensure Python 3.11+ is installed and available
6. **Baseline Format Issues**: The system automatically handles different baseline file formats:
   - `baseline-vX.Y.Z.txt` (from generate-baseline workflow) → converted to `baseline_results.txt`
   - `baseline_results.txt` (standard format) → used directly
   - Multiple files → uses first available in preference order

### Exit Codes

- `0` - Success
- `1` - General error
- `2` - Missing dependency
- `3` - File/directory not found

### Debug Mode

```bash
# For Python scripts, use built-in help and verbose options
uv run benchmark-utils --help
uv run changelog-utils --help
uv run hardware-utils --help

# For changelog generation with debug output
uv run changelog-utils generate --debug

# For shell scripts
bash -x ./scripts/run_all_examples.sh
```

## Development Integration

### CI/CD Integration

The scripts are fully integrated with GitHub Actions workflows:

- **`generate-baseline.yml`**: Automatically generates performance baselines on git tag creation
- **`benchmarks.yml`**: Runs performance regression testing on relevant changes
- **`ci.yml`**: Includes Python code quality checks for all utilities

### Code Quality

All Python scripts are automatically checked in CI:

```bash
# Format Python code
uvx ruff format scripts/

# Lint and auto-fix Python code
uvx ruff check --fix scripts/

# Run tests
uv run pytest
```

### Module Organization

- **`subprocess_utils.py`**: Shared security-hardened subprocess utilities
- **`benchmark_utils.py`**: Standalone benchmarking functionality
- **`changelog_utils.py`**: Shared changelog operations and utilities
- **`hardware_utils.py`**: Standalone hardware detection functionality
- **`enhance_commits.py`**: AI-powered commit categorization (used by changelog_utils)

This modular design ensures code reuse and maintainability across all utilities.

## Script Maintenance

All scripts follow consistent patterns:

### Python Scripts

- **Modern Python**: Python 3.11+ with type hints and union syntax
- **Security**: Uses `subprocess_utils.py` for secure subprocess execution
- **Error Handling**: Custom exception classes with clear error messages
- **Configuration**: Uses `pyproject.toml` for dependencies and tool configuration
- **Code Quality**: Comprehensive linting with ruff and formatting standards

### Shell Scripts

- **Error Handling**: Strict mode with `set -euo pipefail`
- **Dependency Checking**: Validation of required commands
- **Usage Information**: Help text with `--help` flag
- **Project Root Detection**: Automatic detection of project directory
- **Error Messages**: Descriptive error output to stderr

When modifying scripts, maintain these patterns for consistency and reliability.
