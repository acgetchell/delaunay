# shellcheck disable=SC2148
# Justfile for delaunay development workflow
# Install just: https://github.com/casey/just
# Usage: just <command> or just --list

# GitHub Actions workflow validation
action-lint:
    git ls-files -z '.github/workflows/*.yml' '.github/workflows/*.yaml' | xargs -0 -r actionlint

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

# CI simulation (run what CI runs)
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

# Pre-commit workflow (recommended before pushing)
commit-check: quality test-all examples
    @echo "🚀 Ready to commit! All checks passed."

# Coverage analysis
coverage:
    cargo tarpaulin --exclude-files 'benches/**' --exclude-files 'examples/**' --exclude-files 'tests/**' --out Html --output-dir target/tarpaulin
    @echo "📊 Coverage report generated: target/tarpaulin/tarpaulin-report.html"

# Default recipe shows available commands
default:
    @just --list

# Development workflow
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
    @echo "  just action-lint   # Lint all GitHub workflows with actionlint"
    @echo "  just ci            # Simulate CI pipeline"
    @echo "  just ci-baseline   # CI + save performance baseline"
    @echo "  just commit-check  # Full pre-commit checks (includes examples)"
    @echo "  just coverage      # Generate coverage report"
    @echo "  just dev           # Quick development cycle (format, lint, test)"
    @echo "  just perf-help     # Show performance analysis commands"
    @echo "  just quality       # All quality checks"
    @echo "  just test-all      # All tests (Rust + Python)"
    @echo "  just test-debug    # Run debug tools with output"
    @echo "  just test-allocation # Memory allocation profiling"
    @echo ""
    @echo "Benchmark System:"
    @echo "  just bench         # Run all benchmarks"
    @echo "  just bench-baseline # Generate performance baseline"
    @echo "  just bench-compare # Compare against baseline"
    @echo "  just bench-dev     # Development mode (10x faster)"
    @echo ""
    @echo "Performance Analysis:"
    @echo "  just perf-help     # Show performance analysis commands"
    @echo "  just perf-check    # Check for performance regressions"
    @echo "  just perf-baseline # Save current performance as baseline"
    @echo ""
    @echo "Note: Some recipes require external tools. See 'just setup' output."

lint: fmt clippy doc-check

# Shell and markdown quality
markdown-lint:
    git ls-files -z '*.md' | xargs -0 -r -n100 npx markdownlint --config .markdownlint.json --fix

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

# Python code quality
python-lint:
    uv run ruff check scripts/ --fix
    uv run ruff format scripts/

# Comprehensive quality check
quality: fmt clippy doc-check python-lint shell-lint markdown-lint spell-check validate-json validate-toml action-lint
    @echo "✅ All quality checks passed!"

# Development setup
setup:
    @echo "Setting up delaunay development environment..."
    @echo "Note: Rust toolchain and components managed by rust-toolchain.toml (if present)"
    @echo ""
    @echo "Installing Rust components..."
    rustup component add clippy rustfmt rust-docs rust-src
    @echo ""
    @echo "Additional tools required (install separately):"
    @echo "  - uv: https://github.com/astral-sh/uv"
    @echo "  - actionlint: https://github.com/rhysd/actionlint"
    @echo "  - shfmt, shellcheck: via package manager (brew install shfmt shellcheck)"
    @echo "  - jq: via package manager (brew install jq)"
    @echo "  - Node.js (for npx/cspell): https://nodejs.org"
    @echo "  - cargo-tarpaulin: cargo install cargo-tarpaulin"
    @echo ""
    @echo "Installing Python tooling..."
    uv sync --group dev
    @echo ""
    @echo "Building project..."
    cargo build
    @echo "✅ Setup complete! Run 'just help-workflows' to see available commands."

shell-lint:
    git ls-files -z '*.sh' | xargs -0 -r -n1 shfmt -w
    git ls-files -z '*.sh' | xargs -0 -r -n4 shellcheck -x
    @# Note: justfiles are not shell scripts and are excluded from shellcheck

# Spell checking with robust bash implementation
spell-check:
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git status --porcelain | awk '{print $2}' | tr '\n' '\0')
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
    git ls-files -z '*.json' | xargs -0 -r -n1 jq empty

validate-toml:
    git ls-files -z '*.toml' | xargs -0 -r -I {} uv run python -c "import tomllib; tomllib.load(open('{}', 'rb')); print('{} is valid TOML')"
