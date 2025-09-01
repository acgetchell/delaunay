# Scripts Directory

This directory contains utility scripts for building, testing, and benchmarking the delaunay library.

## Prerequisites

Before running these scripts, ensure you have the following dependencies installed:

### Python 3.13+ (Required)

```bash
# Install Python 3.13+ and uv package manager
brew install python@3.13 uv  # macOS with Homebrew
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

All Python utilities require Python 3.13+ and support `--help` for detailed usage.

#### `benchmark_utils.py` 🐍

**Purpose**: Complete benchmark parsing, baseline generation, and performance comparison utilities.

**Features**:

- **Criterion JSON Parsing**: Direct parsing of Criterion's estimates.json for accuracy
- **Baseline Generation**: `generate-baseline` command with git metadata
- **Performance Comparison**: `compare` command with regression detection (>5% threshold)
- **Hardware Integration**: Automatic hardware info inclusion and comparison
- **Development Mode**: `--dev` flag for faster benchmarks (10x speedup)
- **Timezone-Aware Dating**: Proper timezone handling for timestamps
- **Modern Python**: Python 3.13+ with type hints and union syntax

**Commands**:

```bash
# Generate performance baseline
uv run scripts/benchmark_utils.py generate-baseline [--dev] [--output FILE]

# Compare against baseline 
uv run scripts/benchmark_utils.py compare --baseline FILE [--dev] [--output FILE]
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

**Dependencies**: Python 3.13+, `hardware_utils.py`

---

#### `hardware_utils.py` 🐍

**Purpose**: Cross-platform hardware information detection and comparison.

**Features**:

- **Cross-platform**: macOS, Linux, Windows detection
- **Hardware Detection**: CPU (model, cores, threads), memory, Rust toolchain
- **Output Formats**: Formatted display (`info`) or key=value pairs (`kv`)
- **Baseline Comparison**: Hardware compatibility warnings
- **Modern Architecture Support**: Enhanced ARM/heterogeneous core detection

**Commands**:

```bash
# Display formatted hardware information
uv run scripts/hardware_utils.py info

# Display as key=value pairs
uv run scripts/hardware_utils.py kv

# Compare with baseline file
uv run scripts/hardware_utils.py compare --baseline-file FILE
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
  Rust: rustc 1.82.0
  Target: aarch64-apple-darwin
```

**Dependencies**: Python 3.13+, system tools (`sysctl`, `lscpu`, PowerShell)

---

### Bash Scripts (Specialized)

The following bash scripts handle complex integrations with external tools (Node.js, Git) and provide specialized functionality.

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
- **Linux**: Uses `lscpu` for accurate multi-socket core detection, with `/proc/cpuinfo` and `/proc/meminfo` fallbacks
- **Windows**: Uses PowerShell (`Get-CimInstance`) for hardware detection
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
- Windows: PowerShell (`pwsh` or `powershell`) for hardware detection
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

#### Git Tagging from Changelog (Python-based)

**Purpose**: Creates git tags with changelog content as tag messages for seamless GitHub release integration.

**Modern Implementation**: Uses Python utilities instead of shell scripts for better cross-platform compatibility and maintainability.

**Usage**:

```bash
# Create new tag with changelog content
uv run changelog-utils tag v0.4.2

# Force recreate existing tag
uv run changelog-utils tag v0.4.2 --force

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
   uv run changelog-utils tag v0.4.2

2. Push tag to remote:
   git push origin v0.4.2

3. Create GitHub release using tag message:
   gh release create v0.4.2 --notes-from-tag
```

**Advanced Usage**:

```bash
# The changelog-utils tool also supports changelog generation:
uv run changelog-utils generate           # Generate enhanced changelog
uv run changelog-utils generate --debug   # Keep intermediate files for debugging
```

**Dependencies**: Requires Python 3.13+, `uv`, `git`, and access to CHANGELOG.md

---

## Workflow Examples

### Performance Baseline Setup (One-time)

```bash
# 1. Generate initial performance baseline
uv run scripts/benchmark_utils.py generate-baseline

# 2. Commit baseline for CI regression testing
git add benches/baseline_results.txt
git commit -m "Add performance baseline for CI regression testing"
```

### Performance Regression Testing (Development)

```bash
# 1. Make code changes
# ... your modifications ...

# 2. Test for performance regressions
uv run scripts/benchmark_utils.py compare --baseline benches/baseline_results.txt

# 3. Review results in benches/compare_results.txt
# 4. If regressions are acceptable, update baseline:
uv run scripts/benchmark_utils.py generate-baseline
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
uv run scripts/benchmark_utils.py compare --baseline benches/baseline_results.txt --dev

# 3. If major changes needed, generate new dev baseline:
uv run scripts/benchmark_utils.py generate-baseline --dev

# 4. Final validation with full benchmarks before commit:
uv run scripts/benchmark_utils.py generate-baseline          # Full baseline
uv run scripts/benchmark_utils.py compare --baseline benches/baseline_results.txt         # Full comparison
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
uv run scripts/benchmark_utils.py generate-baseline

# 3. Compare against previous baseline
uv run scripts/benchmark_utils.py compare --baseline benches/baseline_results.txt
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
# 1. Runs uv run scripts/benchmark_utils.py compare --baseline benches/baseline_results.txt
# 2. Fails CI if >5% performance regression detected
# 3. Uploads comparison results as artifacts

# If no baseline exists:
# 1. Logs instructions for creating baseline
# 2. Skips regression testing (does not fail CI)
# 3. Suggests running uv run scripts/benchmark_utils.py generate-baseline locally
```

#### CI Integration Benefits

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
4. **Missing Baseline**: Run `uv run scripts/benchmark_utils.py generate-baseline` to generate initial baseline

### Exit Codes

- `0` - Success
- `1` - General error
- `2` - Missing dependency
- `3` - File/directory not found

### Debug Mode

Add `set -x` to any script for verbose execution output:

```bash
# For Python scripts, use -v flag for verbose output
uv run scripts/benchmark_utils.py generate-baseline --help
uv run scripts/hardware_utils.py info --help

# For remaining bash scripts
bash -x ./scripts/generate_changelog.sh
bash -x ./scripts/run_all_examples.sh
```

## Script Maintenance

All scripts follow consistent patterns:

- **Error Handling**: Strict mode with `set -euo pipefail`
- **Dependency Checking**: Validation of required commands
- **Usage Information**: Help text with `--help` flag
- **Project Root Detection**: Automatic detection of project directory
- **Error Messages**: Descriptive error output to stderr

When modifying scripts, maintain these patterns for consistency and reliability.
