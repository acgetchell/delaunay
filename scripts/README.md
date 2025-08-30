# Scripts Directory

This directory contains utility scripts for building, testing, and benchmarking the delaunay library.

## Prerequisites

Before running these scripts, ensure you have the following dependencies installed:

### macOS (using Homebrew)

```bash
brew install jq findutils coreutils
```

### Ubuntu/Debian

```bash
sudo apt-get install jq findutils coreutils
```

### Other Systems

Install equivalent packages for `jq`, `find`, and `sort` using your system's package manager.

## Scripts Overview

### Scripts (Alphabetical)

All scripts support the `--help` flag for detailed usage information.

#### `benchmark_parser.sh`

**Purpose**: Shared utility library for parsing benchmark data across multiple scripts.

**Features**:

- **Reusable Functions**: Common benchmark parsing functions used by other scripts
- **Multiple Parsing Methods**: Both `while read` and `awk` implementations for different use cases
- **Robust Regex Patterns**: Handles various benchmark output formats and edge cases
- **Unit Normalization**: Standardizes time units (µs, us, μs) for consistency
- **Dependency Validation**: Built-in checks for required tools
- **Flexible Output**: Configurable output formatting for different consumers

**Key Functions**:

```bash
# Benchmark detection and parsing
parse_benchmark_start("line")            # Extracts metadata from "Benchmarking..." lines
extract_timing_data("line")              # Parses timing data from result lines
parse_benchmark_identifier("line")       # Extracts point count and dimension

# Output formatting
format_benchmark_result(...)             # Consistent output formatting

# High-level parsing
parse_benchmarks_with_while_read()       # Shell-based parsing implementation
parse_benchmarks_with_awk()             # AWK-based parsing for complex scenarios
```

**Input Format Support**:

```bash
# Supports Criterion benchmark output formats:
# Benchmarking tds_new_2d/tds_new/10
# tds_new_2d/tds_new/10   time:   [354.30 µs 356.10 µs 357.91 µs]
#                         thrpt:  [28.135 Kelem/s 28.257 Kelem/s 28.381 Kelem/s]
```

**Output Format Generated**:

```bash
# Standardized output for baseline and comparison files:
=== 10 Points (2D) ===
Time: [354.30, 356.10, 357.91] μs
Throughput: [28.135, 28.257, 28.381] Kelem/s
```

**Usage Example**:

```bash
# Source the shared functions
source "$(dirname "$0")/benchmark_parser.sh"

# Parse benchmark output
parse_benchmarks_with_while_read "input.txt" "output.txt"
```

**Dependencies**: No external dependencies beyond standard POSIX tools

---

#### `compare_benchmarks.sh`

**Purpose**: Runs fresh benchmark and compares results against baseline performance metrics.

**Features**:

- Runs `cargo bench --bench small_scale_triangulation` to get current performance data
- Reads baseline values from `benches/baseline_results.txt`
- Parses both current Criterion output and baseline file for comparison
- Creates detailed `benches/compare_results.txt` with side-by-side comparison
- Identifies performance regressions (>5% threshold) with clear indicators
- Includes metadata from both current run and baseline (dates, git commits)
- Exits with error code if significant regressions are detected (CI integration)
- **Development Mode**: `--dev` flag for faster benchmarks during development
- Robust parsing with extended regex support for unicode characters
- Improved error handling with validation for all calculations

**Parsing Logic and Formatting Conventions**:

```bash
# INPUT: Fresh Criterion output + existing baseline file
# OUTPUT FORMAT (benches/compare_results.txt):
# === 10 Points (2D) ===
# Current Time: [338.45, 340.12, 341.78] µs
# Current Throughput: [29.467, 29.542, 29.618] Kelem/s
# Baseline Time: [336.95, 338.61, 340.26] µs
# Baseline Throughput: [29.389, 29.533, 29.678] Kelem/s
# Time Change: [-1.24%, +0.45%, +0.45%]
# ✅ OK: Time change within acceptable range
```

**Regression Detection**:

- Extracts Criterion's change percentages when available
- Calculates manual comparison against baseline when needed
- >5% performance degradation triggers ⚠️ REGRESSION warning
- Returns exit code 1 for CI failure on significant regressions

**Usage**:

```bash
# Standard benchmarking (full duration)
./scripts/compare_benchmarks.sh

# Development mode (faster benchmarks)
./scripts/compare_benchmarks.sh --dev

# Help information
./scripts/compare_benchmarks.sh --help
```

**Dependencies**: Requires `cargo`, shared `benchmark_parser.sh`

---

#### `generate_baseline.sh`

**Purpose**: Generates baseline performance results from fresh benchmark run.

**Features**:

- Runs `cargo clean` to clear previous benchmark history and ensure fresh measurements
- Executes `cargo bench --bench small_scale_triangulation` with full Criterion output capture
- Parses Criterion JSON output directly for accurate data extraction
- Creates standardized `benches/baseline_results.txt` with consistent formatting
- Includes git commit hash and timestamp for traceability
- Captures both timing and throughput metrics for comprehensive analysis
- Handles various time units (µs, ms, s) and throughput units (Kelem/s, elem/s) automatically
- **Development Mode**: `--dev` flag for faster benchmarks during development

**Parsing Logic and Formatting Conventions**:

```bash
# INPUT FORMAT (Criterion stdout):
# tds_new_2d/tds_new/10   time:   [336.95 µs 338.61 µs 340.26 µs]
#                         thrpt:  [29.389 Kelem/s 29.533 Kelem/s 29.678 Kelem/s]

# OUTPUT FORMAT (benches/baseline_results.txt):
# === 10 Points (2D) ===
# Time: [336.95, 338.61, 340.26] µs
# Throughput: [29.389, 29.533, 29.678] Kelem/s
```

**Regex Patterns Used**:

- Benchmark results: `^(tds_new_([0-9])d/tds_new/([0-9]+))[[:space:]]+time:`
- Timing extraction: Extract `[low, mean, high]` values and units
- Throughput extraction: Extract throughput values from subsequent lines
- Multi-line state machine parsing to associate timing with throughput data

**Usage**:

```bash
# Standard baseline generation (full benchmarks)
./scripts/generate_baseline.sh

# Development mode (faster benchmarks)
./scripts/generate_baseline.sh --dev

# Help information
./scripts/generate_baseline.sh --help
```

**Dependencies**: Requires `cargo`, `git`, `date`, shared `benchmark_parser.sh`

---

#### `generate_changelog.sh`

**Purpose**: Generates changelog with commit dates instead of tag creation dates and enhanced squashed PR commit expansion for more comprehensive release documentation.

**Features**:

- **Enhanced Error Handling**: Comprehensive validation of prerequisites (npx, git, configuration files)
- **Backup/Recovery**: Automatic backup creation with rollback capability on failure
- **Configuration Validation**: Verifies `.auto-changelog` config and custom template existence
- **Git Repository Validation**: Ensures script runs in valid git repository with history
- **Safe Processing**: Uses temporary files to prevent partial writes to CHANGELOG.md
- **Progress Reporting**: Clear status messages and success confirmation with statistics
- **Robust Date Processing**: Improved regex for converting ISO 8601 to YYYY-MM-DD format
- **Automatic Root Detection**: Uses `BASH_SOURCE[0]` for reliable project root detection
- **🆕 Enhanced Squashed PR Expansion**: Advanced parsing of squashed PR commits to extract detailed commit message bodies
- **🆕 Multi-format Bullet Support**: Handles `*`, `-`, and numbered (`1.`, `2.`, etc.) bullet points in commit messages
- **🆕 Paragraph Preservation**: Maintains multi-line descriptions with proper paragraph breaks and formatting
- **🆕 Word Wrapping**: Intelligent word wrapping at 75 characters with fallback handling

**Comparison with Standard auto-changelog**:

```bash
# Standard auto-changelog (tag creation dates):
# v0.3.4: 2025-08-14  (all releases show same date)
# v0.3.3: 2025-08-14
# v0.3.2: 2025-08-14
# v0.3.1: 2025-08-14

# generate_changelog.sh (actual commit dates):
# v0.3.4: 2025-08-15
# v0.3.3: 2025-08-14  
# v0.3.2: 2025-08-14
# v0.3.1: 2025-07-26
# v0.3.0: 2025-06-17
```

**Technical Implementation**:

- **Configuration**: Uses `docs/templates/changelog.hbs` template configured in `.auto-changelog`
- **Date Extraction**: Template extracts `commits.[0].date` (ISO 8601 timestamp) instead of `isoDate` (tag date)
- **Date Processing**: Converts `2025-08-15T04:44:21.000Z` → `2025-08-15` using `sed 's/T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].*Z//g'`
- **Safety Measures**: Creates `CHANGELOG.md.backup` before modification, restores on failure
- **Temporary Files**: Uses `CHANGELOG.md.tmp` for atomic writes
- **Error Capture**: Captures stderr from auto-changelog for debugging

**Safety Features**:

```bash
# Script creates backups and handles failures gracefully:
# 1. Backs up existing CHANGELOG.md → CHANGELOG.md.backup
# 2. Generates to temporary file CHANGELOG.md.tmp
# 3. Processes dates and writes final output
# 4. Removes backup only on success
# 5. Restores backup if any step fails
```

**Validation Checks**:

- ✅ `npx` command availability (Node.js/npm installation)
- ✅ Git repository detection (`git rev-parse --git-dir`)
- ✅ Git history existence (`git log --oneline -n 1`)
- ✅ `.auto-changelog` configuration file presence
- ✅ `docs/templates/changelog.hbs` template file existence

**Usage**:

```bash
# Generate changelog with accurate commit dates (recommended)
./scripts/generate_changelog.sh

# Alternative: Direct auto-changelog usage (less accurate dating)
npx auto-changelog

# Manual auto-changelog with specific options
npx auto-changelog --unreleased --commit-limit 10
```

**Enhanced Squashed PR Expansion**:

The script now provides sophisticated parsing of squashed PR commits to extract detailed information from commit message bodies:

```bash
# INPUT: Squashed PR commit message
# Feature/convex hull (#50)
# 
# * Implements incremental Delaunay triangulation
# 
# Refactors the triangulation algorithm to use a pure
# incremental Bowyer-Watson approach, improving performance and code organization.
# 
# * Refactors TDS to use IncrementalBoyerWatson
# 
# Completes the TDS refactoring to exclusively use the 
# IncrementalBoyerWatson algorithm, removing legacy methods.

# OUTPUT: Expanded changelog entries
# - **implements incremental delaunay triangulation**
#   Refactors the triangulation algorithm to use a pure incremental 
#   Bowyer-Watson approach, improving performance and code organization.
# 
# - **refactors tds to use incrementalboyerwatson**
#   Completes the TDS refactoring to exclusively use the 
#   IncrementalBoyerWatson algorithm, removing legacy methods.
```

**Supported Bullet Formats**:

- **Asterisk bullets**: `* Title` followed by description
- **Dash bullets**: `- Title` followed by description  
- **Numbered bullets**: `1. Title`, `2. Title` etc. followed by description
- **Fallback handling**: Non-bullet commits treated as single entries

**Parsing Features**:

- **Multi-line descriptions**: Preserves paragraph breaks and formatting
- **Word wrapping**: Intelligent wrapping at 75 characters
- **Whitespace handling**: Cleans leading/trailing whitespace while preserving structure
- **Error recovery**: Falls back to original changelog line if parsing fails

**Testing**:

```bash
# Test the parsing logic with sample commit messages
./scripts/test_squashed_pr_parsing.sh
```

**Dependencies**:

- **Required**: `npx` (Node.js), `git`, `sed`, `awk`
- **Configuration Files**: `.auto-changelog`, `docs/templates/changelog.hbs`
- **npm Package**: `auto-changelog` (installed automatically by npx)
- **Testing**: `scripts/test_squashed_pr_parsing.sh` for validation

---

#### `hardware_info.sh`

**Purpose**: Cross-platform hardware information detection utility for benchmark baseline generation and comparison.

**Features**:

- **Cross-platform detection** for macOS, Linux, and Windows (MSYS2/Cygwin)
- **CPU information**: Brand, core count, thread count detection
- **Memory detection**: Total system memory with GB conversion
- **Rust toolchain info**: Version and target architecture
- **Multiple output formats**: Formatted display or key=value pairs
- **Baseline extraction**: Parse hardware info from existing baseline files
- **Hardware comparison**: Side-by-side environment comparison with warnings
- **Robust fallbacks**: Multiple detection methods per platform

**Key Functions**:

```bash
# Primary hardware detection
get_hardware_info()                     # Returns formatted hardware block
get_hardware_info_kv()                  # Returns key=value pairs

# Baseline file integration
extract_baseline_hardware("file.txt")    # Extracts hardware from baseline
compare_hardware(current, baseline)      # Compares two hardware configs
```

**Hardware Detection Methods**:

- **macOS**: Uses `sysctl` for CPU/memory information
- **Linux**: Parses `/proc/cpuinfo` and `/proc/meminfo`
- **Windows**: Uses PowerShell by default (`Get-CimInstance`), with legacy `wmic` fallback
- **Rust info**: Extracted via `rustc --version` and `rustc -vV`

**Output Formats**:

```bash
# Formatted output (default):
# Hardware Information:
#   OS: macOS
#   CPU: Apple M2 Pro
#   CPU Cores: 10
#   CPU Threads: 10
#   Memory: 16.0 GB
#   Rust: rustc 1.82.0
#   Target: aarch64-apple-darwin

# Key=value output (--kv flag):
# OS=macOS
# CPU=Apple M2 Pro
# CPU_CORES=10
# CPU_THREADS=10
# MEMORY=16.0 GB
# RUST=rustc 1.82.0
# TARGET=aarch64-apple-darwin
```

**Hardware Comparison Warnings**:

```bash
# Example comparison output with warnings:
# Hardware Compatibility:
# ⚠️  CPU differs: Results may not be directly comparable
# ⚠️  CPU core count differs: 8 vs 10 cores
# ⚠️  Rust version differs: Performance may be affected by compiler changes
```

**Usage**:

```bash
# Display formatted hardware information
./scripts/hardware_info.sh

# Display as key=value pairs (useful for parsing)
./scripts/hardware_info.sh --kv

# Source for use in other scripts
source scripts/hardware_info.sh
hardware_info=$(get_hardware_info)
```

**Integration with Other Scripts**:

- Used by `generate_baseline.sh` to include hardware context
- Used by `compare_benchmarks.sh` to detect environment differences
- Provides hardware compatibility warnings for benchmark comparisons

**Dependencies**:

- macOS: `sysctl` (built-in)
- Linux: `/proc/cpuinfo` and `/proc/meminfo` (built-in)
- Windows: PowerShell (`pwsh` or `powershell`) preferred, legacy `wmic` as fallback
- All platforms: `rustc` for Rust toolchain info

---

#### `run_all_examples.sh`

**Purpose**: Executes all example programs in the project to verify functionality.

**Features**:

- Automatically discovers all examples in the `examples/` directory
- Runs simple examples in release mode for representative performance
- Provides comprehensive testing for `test_circumsphere` example
- Creates results directory structure

**Usage**:

```bash
./scripts/run_all_examples.sh
```

**Test Categories for `test_circumsphere`**:

- `all` - Basic dimensional tests and orientation tests
- `test-all-points` - Single point tests in all dimensions
- `debug-all` - All debug tests

**Dependencies**: Requires `cargo`, `find`, `sort` (GNU sort preferred but not required)

---

#### `tag-from-changelog.sh`

**Purpose**: Creates git tags with changelog content as tag messages for seamless GitHub release integration.

**Features**:

- **Automatic changelog extraction**: Parses CHANGELOG.md to find version-specific content
- **Multiple version formats**: Supports `## [X.Y.Z]`, `## vX.Y.Z`, and `## X.Y.Z` headers
- **GitHub release integration**: Tag messages work with `gh release create --notes-from-tag`
- **Safety checks**: Validates git repository, changelog existence, and version format
- **Force recreation**: Option to recreate existing tags with `--force` flag
- **Smart content extraction**: Removes headers and cleans whitespace automatically
- **Preview functionality**: Shows tag message preview before creation
- **Comprehensive error handling**: Clear error messages and usage instructions

**Version Format Support**:

```bash
# Supported CHANGELOG.md section headers:
## [0.3.5] - 2025-08-15     # Standard Keep a Changelog format
## v0.3.5 - 2025-08-15     # Version with 'v' prefix
## 0.3.5 (2025-08-15)     # Alternative format
## v0.3.5                 # Minimal format
```

**Changelog Content Processing**:

- Extracts everything from version header until next `##` header
- Removes the header line itself from tag message
- Cleans leading/trailing whitespace
- Preserves markdown formatting and bullet points
- Handles empty sections gracefully with minimal fallback message

**Integration with GitHub Releases**:

```bash
# Workflow for GitHub releases:
1. Create tag with changelog content:
   ./scripts/tag-from-changelog.sh v0.3.5

2. Push tag to remote:
   git push origin v0.3.5

3. Create GitHub release using tag message:
   gh release create v0.3.5 --notes-from-tag
```

**Usage Examples**:

```bash
# Create new tag with changelog content
./scripts/tag-from-changelog.sh v0.3.5

# Force recreate existing tag
./scripts/tag-from-changelog.sh v0.3.5 --force

# Show help information
./scripts/tag-from-changelog.sh --help
```

**Tag Message Preview**:

```bash
# Example output before tag creation:
Tag message preview:
----------------------------------------
### Added
- New triangulation validation methods
- Enhanced error handling for malformed inputs

### Changed  
- Improved performance for 3D triangulations
- Updated dependency versions

### Fixed
- Resolved edge cases in boundary detection
----------------------------------------
```

**Error Handling**:

- **Repository validation**: Ensures script runs in valid git repository
- **Changelog detection**: Searches current and parent directories for CHANGELOG.md
- **Version format validation**: Requires `vX.Y.Z` format for consistency
- **Existing tag detection**: Prevents accidental overwriting without `--force`
- **Content validation**: Warns if no changelog content found for version

**Safety Features**:

- Validates version format before processing
- Checks for existing tags to prevent accidental overwrites
- Shows preview of tag message before creation
- Provides clear next steps after successful tag creation
- Graceful fallback if no changelog content is found

**Use Cases**:

- **Release automation**: Streamlines GitHub release creation process
- **Changelog integration**: Ensures release notes match changelog content
- **Tag recreation**: Useful for fixing tag messages or updating content
- **Documentation consistency**: Maintains alignment between tags and changelog

**Dependencies**: Requires `git`, `awk`, `sed`, `gh` (GitHub CLI), and access to CHANGELOG.md

---

## Workflow Examples

### Performance Baseline Setup (One-time)

```bash
# 1. Generate initial performance baseline
./scripts/generate_baseline.sh

# 2. Commit baseline for CI regression testing
git add benches/baseline_results.txt
git commit -m "Add performance baseline for CI regression testing"
```

### Performance Regression Testing (Development)

```bash
# 1. Make code changes
# ... your modifications ...

# 2. Test for performance regressions
./scripts/compare_benchmarks.sh

# 3. Review results in benches/compare_results.txt
# 4. If regressions are acceptable, update baseline:
./scripts/generate_baseline.sh
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
./scripts/compare_benchmarks.sh --dev

# 3. If major changes needed, generate new dev baseline:
./scripts/generate_baseline.sh --dev

# 4. Final validation with full benchmarks before commit:
./scripts/generate_baseline.sh          # Full baseline
./scripts/compare_benchmarks.sh         # Full comparison
```

**Development Mode Benefits**:

- **10x faster**: Reduces sample size and measurement time
- **Quick feedback**: Ideal for iterative development
- **Same accuracy**: Still detects significant performance changes
- **Settings**: `sample_size=10, measurement_time=2s, warmup_time=1s`

### Changelog Generation Workflow

```bash
# 1. Make commits and create git tags
git tag v0.4.0
git push origin v0.4.0

# 2. Generate updated changelog with accurate commit dates
./scripts/generate_changelog.sh

# 3. Review and commit the updated changelog
git add CHANGELOG.md
git commit -m "Update changelog with commit dates for v0.4.0"
git push origin main
```

**Benefits of Using generate_changelog.sh**:

- **Accurate Dating**: Shows when development work was actually completed
- **Chronological Accuracy**: Releases show their true development timeline
- **Professional Presentation**: Avoids all releases showing the same tag creation date
- **Historical Clarity**: Makes it easier to understand project development pace

### Manual Benchmark Analysis

```bash
# 1. Run benchmarks directly
cargo bench --bench small_scale_triangulation

# 2. Generate new baseline
./scripts/generate_baseline.sh

# 3. Compare against previous baseline
./scripts/compare_benchmarks.sh
```

### Continuous Integration

The repository includes automated performance regression testing via GitHub Actions:

#### Separate Benchmark Workflow

- **Workflow file**: `.github/workflows/benchmarks.yml`
- **Trigger conditions**:
  - Manual trigger (`workflow_dispatch`)
  - Pushes to `main` branch affecting performance-critical files
  - Changes to `src/`, `benches/`, `Cargo.toml`, `Cargo.lock`

#### CI Behavior

```bash
# If baseline exists:
# 1. Runs ./scripts/compare_benchmarks.sh
# 2. Fails CI if >5% performance regression detected
# 3. Uploads comparison results as artifacts

# If no baseline exists:
# 1. Logs instructions for creating baseline
# 2. Skips regression testing (does not fail CI)
# 3. Suggests running ./scripts/generate_baseline.sh locally
```

#### CI Integration Benefits

- **Separate from main CI**: Avoids slowing down regular development workflow
- **Environment consistency**: Uses Ubuntu runners for reproducible benchmark comparisons
- **Smart triggering**: Only runs on changes that could affect performance
- **Graceful degradation**: Skips if baseline missing, with clear setup instructions
- **Artifact collection**: Stores benchmark results for historical analysis

## Error Handling and Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install required packages using your system's package manager
2. **Permission Errors**: Ensure scripts are executable with `chmod +x scripts/*.sh`
3. **Path Issues**: Run scripts from the project root directory
4. **Missing Baseline**: Run `./scripts/generate_baseline.sh` to generate initial baseline

### Exit Codes

- `0` - Success
- `1` - General error
- `2` - Missing dependency
- `3` - File/directory not found

### Debug Mode

Add `set -x` to any script for verbose execution output:

```bash
bash -x ./scripts/generate_baseline.sh
bash -x ./scripts/compare_benchmarks.sh
```

## Script Maintenance

All scripts follow consistent patterns:

- **Error Handling**: Strict mode with `set -euo pipefail`
- **Dependency Checking**: Validation of required commands
- **Usage Information**: Help text with `--help` flag
- **Project Root Detection**: Automatic detection of project directory
- **Error Messages**: Descriptive error output to stderr

When modifying scripts, maintain these patterns for consistency and reliability.
