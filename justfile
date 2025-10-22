# shellcheck disable=SC2148
# Justfile for delaunay development workflow
# Install just: https://github.com/casey/just
# Usage: just <command> or just --list

# GitHub Actions workflow validation
action-lint:
    #!/usr/bin/env bash
    set -euo pipefail
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

bench-baseline:
    uv run benchmark-utils generate-baseline

bench-compare:
    uv run benchmark-utils compare --baseline baseline-artifact/baseline_results.txt

bench-compile:
    cargo bench --workspace --no-run

bench-dev:
    uv run benchmark-utils compare --baseline baseline-artifact/baseline_results.txt --dev

# Phase 4 SlotMap evaluation benchmarks
bench-phase4:
    @echo "🔬 Running Phase 4 SlotMap evaluation benchmarks (~2-3 hours)"
    cargo bench --bench large_scale_performance

bench-phase4-large:
    @echo "🔬 Running Phase 4 large-scale benchmarks (~4-6 hours, use on compute cluster)"
    BENCH_LARGE_SCALE=1 cargo bench --bench large_scale_performance

bench-phase4-quick:
    @echo "⚡ Quick Phase 4 validation tests (~90 seconds)"
    cargo test --release --test storage_backend_compatibility -- --ignored

# Compare SlotMap vs DenseSlotMap storage backends
compare-storage:
    @echo "📊 Comparing SlotMap vs DenseSlotMap performance (~4-6 hours)"
    uv run compare-storage-backends --bench large_scale_performance

compare-storage-large:
    @echo "📊 Comparing storage backends at large scale (~8-12 hours, use on compute cluster)"
    BENCH_LARGE_SCALE=1 uv run compare-storage-backends --bench large_scale_performance

# Build commands
build:
    cargo build

build-release:
    cargo build --release

# Changelog management
changelog:
    uv run changelog-utils generate

changelog-tag version:
    uv run changelog-utils tag {{version}}

changelog-update: changelog
    @echo "📝 Changelog updated successfully!"
    @echo "To create a git tag with changelog content for a specific version, run:"
    @echo "  just changelog-tag <version>  # e.g., just changelog-tag v0.4.2"

# CI simulation: quality checks + release tests + benchmark compilation
ci: quality test-release bench-compile
    @echo "🎯 CI simulation complete!"

# CI with performance baseline
ci-baseline tag="ci":
    just ci
    just perf-baseline {{tag}}

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
    @echo "🚀 Ready to commit! All checks passed."

# Coverage analysis
coverage:
    cargo tarpaulin --exclude-files 'benches/**' --exclude-files 'examples/**' --exclude-files 'tests/**' --out Html --output-dir target/tarpaulin
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
    @echo "  just test-debug    # Run debug tools with output"
    @echo "  just test-allocation # Memory allocation profiling"
    @echo "  just examples      # Run all examples"
    @echo "  just coverage      # Generate coverage report"
    @echo ""
    @echo "Quality Check Groups:"
    @echo "  just lint          # All linting (code + docs + config)"
    @echo "  just lint-code     # Code linting (Rust, Python, Shell)"
    @echo "  just lint-docs     # Documentation linting (Markdown, Spelling)"
    @echo "  just lint-config   # Configuration validation (JSON, TOML, Actions)"
    @echo ""
    @echo "Benchmark System:"
    @echo "  just bench         # Run all benchmarks"
    @echo "  just bench-baseline # Generate performance baseline"
    @echo "  just bench-compare # Compare against baseline"
    @echo "  just bench-dev     # Development mode (10x faster)"
    @echo ""
    @echo "Phase 4 SlotMap Evaluation:"
    @echo "  just bench-phase4       # Run Phase 4 benchmarks (~2-3 hours)"
    @echo "  just bench-phase4-large # Large scale (~4-6 hours, compute cluster)"
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

# Code linting: Rust (fmt, clippy, docs) + Python (ruff) + Shell scripts
lint-code: fmt clippy doc-check python-lint shell-lint

# Documentation linting: Markdown + spell checking
lint-docs: markdown-lint spell-check

# Configuration validation: JSON, TOML, GitHub Actions workflows
lint-config: validate-json validate-toml action-lint

# All linting: code + documentation + configuration
lint: lint-code lint-docs lint-config

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
perf-baseline tag="":
    #!/usr/bin/env bash
    set -euo pipefail
    tag_value="{{tag}}"
    if [ -n "$tag_value" ]; then
        uv run benchmark-utils generate-baseline --tag "$tag_value"
    else
        uv run benchmark-utils generate-baseline
    fi

perf-check threshold="5.0":
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -f "baseline-artifact/baseline_results.txt" ]; then
        uv run benchmark-utils compare --baseline baseline-artifact/baseline_results.txt --threshold {{threshold}}
    else
        echo "❌ No baseline found. Run 'just perf-baseline' first."
        exit 1
    fi

perf-compare file:
    uv run benchmark-utils compare --baseline "{{file}}"

perf-help:
    @echo "Performance Analysis Commands:"
    @echo "  just perf-baseline [tag]    # Save current performance as baseline (optionally tagged)"
    @echo "  just perf-check [threshold] # Check for regressions (default: 5% threshold)"
    @echo "  just perf-compare <file>    # Compare with specific baseline file"
    @echo "  just bench-dev             # Development mode benchmarks (10x faster)"
    @echo ""
    @echo "Benchmark System (Delaunay-specific):"
    @echo "  just bench-baseline        # Generate baseline via benchmark-utils"
    @echo "  just bench-compare         # Compare against stored baseline"
    @echo "  just bench-dev             # Fast development comparison"
    @echo ""
    @echo "Examples:"
    @echo "  just perf-baseline v1.0.0  # Save tagged baseline"
    @echo "  just perf-check 10.0       # Check with 10% threshold"
    @echo "  just bench-dev             # Quick benchmark iteration"

# Profiling
profile:
    samply record cargo bench --bench profiling_suite -- triangulation_scaling

profile-dev:
    PROFILING_DEV_MODE=1 samply record cargo bench --bench profiling_suite -- "triangulation_scaling_3d/tds_new/random_3d"

# Python code quality
python-lint:
    uv run ruff check scripts/ --fix
    uv run ruff format scripts/

# Comprehensive quality check: all linting + all tests (Rust + Python)
quality: lint-code lint-docs lint-config test-all
    @echo "✅ All quality checks and tests passed!"

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
                uv) echo "    Install: https://github.com/astral-sh/uv" ;;
                actionlint) echo "    Install: https://github.com/rhysd/actionlint" ;;
                shfmt|shellcheck) echo "    Install: brew install $tool" ;;
                jq) echo "    Install: brew install jq" ;;
                node) echo "    Install: https://nodejs.org" ;;
            esac
        fi
    done
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

test-python:
    uv run pytest

test-release:
    cargo test --release

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

validate-toml:
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
