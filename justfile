# shellcheck disable=SC2148
# Justfile for delaunay development workflow
# Install just: https://github.com/casey/just
# Usage: just <command> or just --list

# Use bash with strict error handling for all recipes
set shell := ["bash", "-euo", "pipefail", "-c"]

# Internal helper: ensure uv is installed
_ensure-uv:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v uv >/dev/null || { echo "❌ 'uv' not found. See 'just setup' or https://github.com/astral-sh/uv"; exit 1; }

# GitHub Actions workflow validation
action-lint:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! command -v actionlint >/dev/null; then
        echo "⚠️ 'actionlint' not found. See 'just setup' or https://github.com/rhysd/actionlint"
        exit 0
    fi
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '.github/workflows/*.yml' '.github/workflows/*.yaml')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 actionlint
    else
        echo "No workflow files found to lint."
    fi

# Benchmarks
bench:
    cargo bench --workspace

bench-baseline: _ensure-uv
    uv run benchmark-utils generate-baseline

# CI regression benchmarks (fast, suitable for CI)
bench-ci:
    cargo bench --bench ci_performance_suite

bench-compare: _ensure-uv
    uv run benchmark-utils compare --baseline baseline-artifact/baseline_results.txt

bench-compile:
    cargo bench --workspace --no-run

# Development mode benchmarks: fast iteration with reduced sample sizes
bench-dev: _ensure-uv
    CRIT_SAMPLE_SIZE=10 CRIT_MEASUREMENT_MS=1000 CRIT_WARMUP_MS=500 uv run benchmark-utils compare --baseline baseline-artifact/baseline_results.txt --dev

# Generate performance summary with fresh benchmark runs (for releases)
bench-perf-summary: _ensure-uv
    uv run benchmark-utils generate-summary --run-benchmarks

# Phase 4 SlotMap evaluation benchmarks
bench-phase4:
    @echo "🔬 Running Phase 4 SlotMap evaluation benchmarks (~10-30 min default scale)"
    cargo bench --bench large_scale_performance

bench-phase4-large:
    @echo "🔬 Running Phase 4 large-scale benchmarks with BENCH_LARGE_SCALE=1 (~2-3 hours)"
    BENCH_LARGE_SCALE=1 cargo bench --bench large_scale_performance

bench-phase4-quick:
    @echo "⚡ Quick Phase 4 validation tests (~90 seconds)"
    cargo test --release --test storage_backend_compatibility -- --ignored

# Quick benchmark validation: minimal samples for sanity checking
bench-quick:
    CRIT_SAMPLE_SIZE=5 CRIT_MEASUREMENT_MS=500 CRIT_WARMUP_MS=200 cargo bench --workspace

# Build commands
build:
    cargo build

build-release:
    cargo build --release

# Changelog management
changelog: _ensure-uv
    uv run changelog-utils generate

changelog-tag version: _ensure-uv
    uv run changelog-utils tag {{version}}

changelog-update: changelog
    @echo "📝 Changelog updated successfully!"
    @echo "To create a git tag with changelog content for a specific version, run:"
    @echo "  just changelog-tag <version>  # e.g., just changelog-tag v0.4.2"

# CI with performance baseline
ci-baseline tag="ci":
    just ci
    just perf-baseline {{tag}}

# CI simulation: quality checks + tests + benchmark compilation (matches .github/workflows/ci.yml)
ci: quality test bench-compile
    @echo "🎯 CI simulation complete!"

# Clean build artifacts
clean:
    cargo clean
    rm -rf target/tarpaulin
    rm -rf coverage_report

# Code quality and formatting
clippy:
    cargo clippy --workspace --all-targets --all-features -- -D warnings -W clippy::pedantic -W clippy::nursery -W clippy::cargo

# Pre-commit workflow: CI + examples (most comprehensive validation)
commit-check: ci examples
    @echo "🚀 Ready to commit! All checks passed!"

# Compare SlotMap vs DenseSlotMap storage backends
compare-storage: _ensure-uv
    @echo "📊 Comparing SlotMap vs DenseSlotMap performance (~4-6 hours)"
    uv run compare-storage-backends --bench large_scale_performance

compare-storage-large: _ensure-uv
    @echo "📊 Comparing storage backends at large scale (~8-12 hours, use on compute cluster)"
    BENCH_LARGE_SCALE=1 uv run compare-storage-backends --bench large_scale_performance

# Coverage analysis (matches CI configuration)
coverage:
    cargo tarpaulin --exclude-files 'benches/*' --exclude-files 'examples/*' --all-features --workspace --out Html --output-dir target/tarpaulin
    @echo "📊 Coverage report generated: target/tarpaulin/tarpaulin-report.html"

# Default recipe shows available commands
default:
    @just --list

# Development workflow: quick format, lint, and test cycle
dev: fmt clippy test
    @echo "⚡ Quick development check complete!"

doc-check:
    RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --document-private-items

# Examples and validation
examples:
    ./scripts/run_all_examples.sh

fmt:
    cargo fmt --all

help-workflows:
    @echo "Common Just workflows:"
    @echo "  just dev           # Quick development cycle (format, lint, test)"
    @echo "  just quality       # All quality checks + tests (comprehensive)"
    @echo "  just ci            # CI simulation (quality + release tests + bench compile)"
    @echo "  just commit-check  # Pre-commit validation (CI + examples) - most thorough"
    @echo "  just ci-baseline   # CI + save performance baseline"
    @echo ""
    @echo "Testing:"
    @echo "  just test          # Rust lib and doc tests (debug mode)"
    @echo "  just test-all      # All tests (Rust + Python, debug mode)"
    @echo "  just test-release  # All tests in release mode"
    @echo "  just test-slow     # Include slow/stress tests (100+ vertices)"
    @echo "  just test-slow-release # Slow tests in release mode (faster)"
    @echo "  just test-debug    # Run debug tools with output"
    @echo "  just test-allocation # Memory allocation profiling"
    @echo "  just examples      # Run all examples"
    @echo "  just coverage      # Generate coverage report"
    @echo ""
    @echo "Quality Check Groups:"
    @echo "  just quality       # All quality checks + tests (standard, fast)"
    @echo "  just quality-slow  # All quality checks + tests including slow tests"
    @echo "  just lint          # All linting (code + docs + config)"
    @echo "  just lint-code     # Code linting (Rust, Python, Shell)"
    @echo "  just lint-docs     # Documentation linting (Markdown, Spelling)"
    @echo "  just lint-config   # Configuration validation (JSON, TOML, Actions)"
    @echo ""
    @echo "Benchmark System:"
    @echo "  just bench              # Run all benchmarks"
    @echo "  just bench-baseline     # Generate performance baseline"
    @echo "  just bench-ci           # CI regression benchmarks (fast, ~5-10 min)"
    @echo "  just bench-compare      # Compare against baseline"
    @echo "  just bench-dev          # Development mode (10x faster, ~1-2 min)"
    @echo "  just bench-perf-summary # Generate performance summary for releases (~30-45 min)"
    @echo "  just bench-quick        # Quick validation (minimal samples, ~30 sec)"
    @echo ""
    @echo "Phase 4 SlotMap Evaluation:"
    @echo "  just bench-phase4       # Run Phase 4 benchmarks (~10-30 min default)"
    @echo "  just bench-phase4-large # Large scale with BENCH_LARGE_SCALE=1 (~2-3 hours)"
    @echo "  just bench-phase4-quick # Quick validation tests (~90 seconds)"
    @echo ""
    @echo "Storage Backend Comparison:"
    @echo "  just compare-storage       # Compare SlotMap vs DenseSlotMap (~4-6 hours)"
    @echo "  just compare-storage-large # Large scale comparison (~8-12 hours, compute cluster)"
    @echo ""
    @echo "Performance Analysis:"
    @echo "  just perf-help     # Show performance analysis commands"
    @echo "  just perf-check    # Check for performance regressions"
    @echo "  just perf-baseline # Save current performance as baseline"
    @echo ""
    @echo "Note: Some recipes require external tools. See 'just setup' output."

# All linting: code + documentation + configuration
lint: lint-code lint-docs lint-config

# Code linting: Rust (fmt, clippy, docs) + Python (ruff) + Shell scripts
lint-code: fmt clippy doc-check python-lint shell-lint

# Configuration validation: JSON, TOML, GitHub Actions workflows
lint-config: validate-json validate-toml action-lint

# Documentation linting: Markdown + spell checking
lint-docs: markdown-lint spell-check

# Shell and markdown quality
markdown-lint:
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '*.md')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 -n100 npx markdownlint --config .markdownlint.json --fix
    else
        echo "No markdown files found to lint."
    fi

# Performance analysis framework
perf-baseline tag="": _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    tag_value="{{tag}}"
    if [ -n "$tag_value" ]; then
        uv run benchmark-utils generate-baseline --tag "$tag_value"
    else
        uv run benchmark-utils generate-baseline
    fi

perf-check threshold="5.0": _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -f "baseline-artifact/baseline_results.txt" ]; then
        uv run benchmark-utils compare --baseline baseline-artifact/baseline_results.txt --threshold {{threshold}}
    else
        echo "❌ No baseline found. Run 'just perf-baseline' first."
        exit 1
    fi

perf-compare file: _ensure-uv
    uv run benchmark-utils compare --baseline "{{file}}"

perf-help:
    @echo "Performance Analysis Commands:"
    @echo "  just perf-baseline [tag]    # Save current performance as baseline (optionally tagged)"
    @echo "  just perf-check [threshold] # Check for regressions (default: 5% threshold)"
    @echo "  just perf-compare <file>    # Compare with specific baseline file"
    @echo "  just bench-dev             # Development mode benchmarks (10x faster)"
    @echo "  just bench-quick           # Quick validation benchmarks (minimal samples)"
    @echo ""
    @echo "Profiling Commands:"
    @echo "  just profile               # Profile full triangulation_scaling benchmark"
    @echo "  just profile-dev           # Profile 3D dev mode (faster iteration)"
    @echo "  just profile-mem           # Profile memory allocations (with count-allocations feature)"
    @echo ""
    @echo "Benchmark System (Delaunay-specific):"
    @echo "  just bench-baseline        # Generate baseline via benchmark-utils"
    @echo "  just bench-compare         # Compare against stored baseline"
    @echo "  just bench-dev             # Fast development comparison"
    @echo "  just bench-quick           # Quick validation (minimal samples)"
    @echo ""
    @echo "Environment Variables (Benchmark Configuration):"
    @echo "  CRIT_SAMPLE_SIZE=N         # Number of samples per benchmark"
    @echo "  CRIT_MEASUREMENT_MS=N      # Measurement time in milliseconds"
    @echo "  CRIT_WARMUP_MS=N           # Warm-up time in milliseconds"
    @echo "  DELAUNAY_BENCH_SEED=N      # Random seed (decimal or 0x-hex)"
    @echo ""
    @echo "Examples:"
    @echo "  just perf-baseline v1.0.0  # Save tagged baseline"
    @echo "  just perf-check 10.0       # Check with 10% threshold"
    @echo "  just bench-dev             # Quick benchmark iteration"
    @echo "  CRIT_SAMPLE_SIZE=100 just bench  # Custom sample size"

# Profiling
profile:
    samply record cargo bench --bench profiling_suite -- triangulation_scaling

profile-dev:
    PROFILING_DEV_MODE=1 samply record cargo bench --bench profiling_suite -- "triangulation_scaling_3d/tds_new/random_3d"

profile-mem:
    samply record cargo bench --bench profiling_suite --features count-allocations -- memory_profiling

# Python code quality
python-lint: _ensure-uv
    uv run ruff check scripts/ --fix
    uv run ruff format scripts/

# Comprehensive quality check: all linting + all tests (Rust + Python)
quality: lint-code lint-docs lint-config test-all
    @echo "✅ All quality checks and tests passed!"

# Comprehensive quality check including slow tests (for local development/pre-commit)
quality-slow: lint-code lint-docs lint-config test-all test-slow-release
    @echo "✅ All quality checks and tests (including slow tests) passed!"

# Development setup
setup:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Setting up delaunay development environment..."
    echo "Note: Rust toolchain and components managed by rust-toolchain.toml (if present)"
    echo ""
    echo "Installing Rust components..."
    rustup component add clippy rustfmt rust-docs rust-src
    echo ""
    echo "Installing Rust tools..."
    # Install cargo tools if not already installed
    if ! command -v cargo-tarpaulin &> /dev/null; then
        echo "Installing cargo-tarpaulin..."
        cargo install cargo-tarpaulin
    else
        echo "cargo-tarpaulin already installed"
    fi
    if ! command -v samply &> /dev/null; then
        echo "Installing samply..."
        cargo install samply
    else
        echo "samply already installed"
    fi
    echo ""
    echo "Additional tools (will check if installed):"
    # Check for system tools
    for tool in uv actionlint shfmt shellcheck jq node; do
        if command -v "$tool" &> /dev/null; then
            echo "  ✓ $tool installed"
        else
            echo "  ✗ $tool NOT installed"
            case "$tool" in
                uv)
                    echo "    Install: https://github.com/astral-sh/uv"
                    echo "    macOS: brew install uv"
                    echo "    Linux/WSL: curl -LsSf https://astral.sh/uv/install.sh | sh"
                    ;;
                actionlint) echo "    Install: https://github.com/rhysd/actionlint" ;;
                shfmt|shellcheck) echo "    Install: brew install $tool" ;;
                jq) echo "    Install: brew install jq" ;;
                node) echo "    Install: https://nodejs.org" ;;
            esac
        fi
    done
    echo ""
    # Ensure uv is installed before proceeding
    if ! command -v uv &> /dev/null; then
        echo "❌ 'uv' is required but not installed. Please install it first (see instructions above)."
        exit 1
    fi
    echo ""
    echo "Installing Python tooling..."
    uv sync --group dev
    echo ""
    echo "Building project..."
    cargo build
    echo "✅ Setup complete! Run 'just help-workflows' to see available commands."

shell-lint:
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '*.sh')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 -n1 shfmt -w
        printf '%s\0' "${files[@]}" | xargs -0 -n4 shellcheck -x
    else
        echo "No shell files found to lint."
    fi
    # Note: justfiles are not shell scripts and are excluded from shellcheck

# Spell checking with robust bash implementation
spell-check:
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    # Use -z for NUL-delimited output to handle filenames with spaces
    while IFS= read -r -d '' status_line; do
        # Extract filename from git status --porcelain -z format
        # Format: XY filename or XY oldname -> newname (for renames)
        if [[ "$status_line" =~ ^..[[:space:]](.*)$ ]]; then
            filename="${BASH_REMATCH[1]}"
            # For renames (format: "old -> new"), take the new filename
            if [[ "$filename" == *" -> "* ]]; then
                filename="${filename#* -> }"
            fi
            files+=("$filename")
        fi
    done < <(git status --porcelain -z --ignored=no)
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 npx cspell lint --config cspell.json --no-progress --gitignore --cache --exclude cspell.json
    else
        echo "No modified files to spell-check."
    fi

# Testing
test:
    cargo test --lib --verbose
    cargo test --doc --verbose

test-all: test test-python
    @echo "✅ All tests passed!"

test-allocation:
    cargo test --test allocation_api --features count-allocations -- --nocapture

test-debug:
    cargo test --test circumsphere_debug_tools -- --nocapture

test-python: _ensure-uv
    uv run pytest

test-release:
    cargo test --release

# Run tests including slow/stress tests (100+ vertices, multiple dimensions)
# These are gated behind the 'slow-tests' feature to keep CI fast
test-slow:
    cargo test --features slow-tests

test-slow-release:
    cargo test --release --features slow-tests

# File validation
validate-json:
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '*.json')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 -n1 jq empty
    else
        echo "No JSON files found to validate."
    fi

validate-toml: _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '*.toml')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 -I {} uv run python -c "import tomllib; tomllib.load(open('{}', 'rb')); print('{} is valid TOML')"
    else
        echo "No TOML files found to validate."
    fi
