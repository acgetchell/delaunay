# shellcheck disable=SC2148
# Justfile for delaunay development workflow
# Install just: https://github.com/casey/just
# Usage: just <command> or just --list

# Default recipe shows available commands
default:
    @just --list

# Development setup
setup:
    @echo "Setting up development environment..."
    rustup component add clippy rustfmt rust-docs rust-src
    uv sync --group dev
    cargo build

# Code quality and formatting
fmt:
    cargo fmt --all

clippy:
    cargo clippy --workspace --all-targets --all-features -- -D warnings -W clippy::pedantic -W clippy::nursery -W clippy::cargo

doc-check:
    RUSTDOCFLAGS='-D warnings' cargo doc-check-strict

lint: fmt clippy doc-check

# Python code quality
python-lint:
    uv run ruff check scripts/ --fix
    uv run ruff format scripts/

# Shell and markdown quality
shell-lint:
    git ls-files -z '*.sh' | xargs -0 -r -n1 shfmt -w
    git ls-files -z '*.sh' | xargs -0 -r -n4 shellcheck -x
    @# Note: justfiles are not shell scripts and are excluded from shellcheck

markdown-lint:
    git ls-files -z '*.md' | xargs -0 -r -n100 npx markdownlint --config .markdownlint.json --fix

# Spell checking
spell-check:
    files="$(git status --porcelain | awk '{print $2}')"; \
    if [ -n "$files" ]; then \
        npx cspell lint --config cspell.json --no-progress --gitignore --cache --exclude cspell.json $files; \
    else \
        echo "No modified files to spell-check."; \
    fi

# File validation
validate-json:
    git ls-files -z '*.json' | xargs -0 -r -n1 jq empty

validate-toml:
    git ls-files -z '*.toml' | xargs -0 -r -I {} uv run python -c "import tomllib; tomllib.load(open('{}', 'rb')); print('{} is valid TOML')"

# Comprehensive quality check
quality: fmt clippy doc-check python-lint shell-lint markdown-lint spell-check validate-json validate-toml
    @echo "✅ All quality checks passed!"

# Testing
test:
    cargo test --lib --verbose
    cargo test --doc --verbose

test-release:
    cargo test --release

test-debug:
    cargo test --test circumsphere_debug_tools -- --nocapture

test-allocation:
    cargo test --test allocation_api --features count-allocations -- --nocapture

test-python:
    uv run pytest

test-all: test test-python
    @echo "✅ All tests passed!"

# Examples and validation
examples:
    ./scripts/run_all_examples.sh

# Benchmarks
bench-compile:
    cargo bench --workspace --no-run

bench:
    cargo bench --workspace

bench-baseline:
    uv run benchmark-utils generate-baseline

bench-compare:
    uv run benchmark-utils compare --baseline baseline-artifact/baseline_results.txt

bench-dev:
    uv run benchmark-utils compare --baseline baseline-artifact/baseline_results.txt --dev

# Coverage analysis
coverage:
    cargo tarpaulin --exclude-files 'benches/**' --exclude-files 'examples/**' --exclude-files 'tests/**' --out Html --output-dir target/tarpaulin
    @echo "📊 Coverage report generated: target/tarpaulin/tarpaulin-report.html"

# Changelog management
changelog:
    uv run changelog-utils generate

changelog-update: changelog
    @echo "📝 Changelog updated successfully!"
    @echo "To create a git tag with changelog content for a specific version, run:"
    @echo "  just changelog-tag <version>  # e.g., just changelog-tag v0.4.2"

changelog-tag version:
    uv run changelog-utils tag {{version}}

# Pre-commit workflow (recommended before pushing)
pre-commit: quality test-all examples
    @echo "🚀 Ready to commit! All checks passed."

# CI simulation (run what CI runs)
ci: quality test-release bench-compile
    @echo "🎯 CI simulation complete!"

# Development workflow
dev: fmt clippy test
    @echo "⚡ Quick development check complete!"

# Clean build artifacts
clean:
    cargo clean
    rm -rf target/tarpaulin
    rm -rf coverage_report

# Help with common workflows
help-workflows:
    @echo "Common Just workflows:"
    @echo "  just dev           # Quick development cycle (format, lint, test)"
    @echo "  just pre-commit    # Full pre-commit checks"
    @echo "  just ci            # Simulate CI pipeline"
    @echo "  just quality       # All quality checks"
    @echo "  just test-all      # All tests (Rust + Python)"
    @echo "  just coverage      # Generate coverage report"
