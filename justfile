# shellcheck disable=SC2148
# Justfile for delaunay development workflow
# Install just: https://github.com/casey/just
# Usage: just <command> or just --list

# Use bash with strict error handling for all recipes
set shell := ["bash", "-euo", "pipefail", "-c"]

# Common tarpaulin arguments for all coverage runs
# Note: -t 300 sets per-test timeout to 5 minutes (needed for slow CI environments)
# Excludes: storage_backend_compatibility (all tests ignored - Phase 4 evaluation tests)
_coverage_base_args := '''--exclude-files 'benches/*' --exclude-files 'examples/*' \
  --workspace --lib --tests \
  --exclude storage_backend_compatibility \
  -t 300 --verbose --implicit-test-threads'''

_ensure-actionlint:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v actionlint >/dev/null || { echo "❌ 'actionlint' not found. See 'just setup' or https://github.com/rhysd/actionlint"; exit 1; }

# Internal helpers: ensure external tooling is installed
_ensure-jq:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v jq >/dev/null || { echo "❌ 'jq' not found. See 'just setup' or install: brew install jq"; exit 1; }

_ensure-git-cliff:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v git-cliff >/dev/null || {
        echo "❌ 'git-cliff' not found. Install via Homebrew: brew install git-cliff"
        echo "   Or via Cargo: cargo install git-cliff"
        exit 1
    }

_ensure-npx:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v npx >/dev/null || { echo "❌ 'npx' not found. See 'just setup' or install Node.js (for npx tools): https://nodejs.org"; exit 1; }

_ensure-prettier-or-npx:
    #!/usr/bin/env bash
    set -euo pipefail
    if command -v prettier >/dev/null; then
        exit 0
    fi
    command -v npx >/dev/null || {
        echo "❌ Neither 'prettier' nor 'npx' found. Install via npm (recommended): npm i -g prettier"
        echo "   Or install Node.js (for npx): https://nodejs.org"
        exit 1
    }

_ensure-shellcheck:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v shellcheck >/dev/null || { echo "❌ 'shellcheck' not found. See 'just setup' or https://www.shellcheck.net"; exit 1; }

_ensure-shfmt:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v shfmt >/dev/null || { echo "❌ 'shfmt' not found. See 'just setup' or install: brew install shfmt"; exit 1; }

# Internal helper: ensure taplo is installed
_ensure-taplo:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v taplo >/dev/null || { echo "❌ 'taplo' not found. See 'just setup' or install: brew install taplo (or: cargo install taplo-cli)"; exit 1; }

# Internal helper: ensure typos-cli is installed
_ensure-typos:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v typos >/dev/null || { echo "❌ 'typos' not found. See 'just setup-tools' or install: cargo install typos-cli"; exit 1; }

# Internal helper: ensure uv is installed
_ensure-uv:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v uv >/dev/null || { echo "❌ 'uv' not found. See 'just setup' or https://github.com/astral-sh/uv"; exit 1; }

_ensure-yamllint:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v yamllint >/dev/null || { echo "❌ 'yamllint' not found. See 'just setup' or install: brew install yamllint"; exit 1; }

# GitHub Actions workflow validation
action-lint: _ensure-actionlint
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

bench-baseline: _ensure-uv
    uv run benchmark-utils generate-baseline

# CI regression benchmarks (fast, suitable for CI)
bench-ci:
    cargo bench --bench ci_performance_suite

bench-compare: _ensure-uv
    uv run benchmark-utils compare --baseline baseline-artifact/baseline_results.txt

# Compile benchmarks without running them, treating warnings as errors.
# This catches bench/release-profile-only warnings (e.g. debug_assertions-gated unused vars)
# that won't show up in normal debug-profile `cargo test` / `cargo clippy` runs.
bench-compile:
    RUSTFLAGS='-D warnings' cargo bench --workspace --no-run

# Development mode benchmarks: fast iteration with reduced sample sizes
bench-dev: _ensure-uv
    CRIT_SAMPLE_SIZE=10 CRIT_MEASUREMENT_MS=1000 CRIT_WARMUP_MS=500 uv run benchmark-utils compare --baseline baseline-artifact/baseline_results.txt --dev

# Generate performance summary with fresh benchmark runs (for releases)
bench-perf-summary: _ensure-uv
    uv run benchmark-utils generate-summary --run-benchmarks

# Quick benchmark validation: minimal samples for sanity checking
bench-quick:
    CRIT_SAMPLE_SIZE=5 CRIT_MEASUREMENT_MS=500 CRIT_WARMUP_MS=200 cargo bench --workspace

# Build commands
build:
    cargo build

build-release:
    cargo build --release

# Changelog management
changelog: _ensure-uv _ensure-git-cliff
    uv run changelog-utils generate

changelog-tag version: _ensure-uv
    uv run changelog-utils tag {{version}}

changelog-update: changelog
    @echo "📝 Changelog updated successfully!"
    @echo "To create a git tag with changelog content for a specific version, run:"
    @echo "  just changelog-tag <version>  # e.g., just changelog-tag v0.4.2"

# Check (non-mutating): run all linters/validators
check: lint
    @echo "✅ Checks complete!"

# Fast compile check (no binary produced)
check-fast:
    cargo check

# CI simulation: comprehensive validation (matches .github/workflows/ci.yml)
# Runs: checks + all tests (Rust + Python) + examples + bench compile
ci: check bench-compile test-all examples
    @echo "🎯 CI checks complete!"

# CI with performance baseline
ci-baseline tag="ci":
    just ci
    just perf-baseline {{tag}}

# CI + slow/stress tests (100+ vertices, stress tests)
ci-slow: ci test-slow
    @echo "✅ CI + slow tests passed!"

# Clean build artifacts
clean:
    cargo clean
    rm -rf target/tarpaulin
    rm -rf coverage_report
    rm -rf coverage

# Code quality and formatting
clippy:
    # SlotMap backend (disabled default DenseSlotMap)
    cargo clippy --workspace --all-targets --no-default-features -- -D warnings -W clippy::pedantic -W clippy::nursery -W clippy::cargo

    # DenseSlotMap backend (default)
    cargo clippy --workspace --all-targets -- -D warnings -W clippy::pedantic -W clippy::nursery -W clippy::cargo

    # All features
    cargo clippy --workspace --all-targets --all-features -- -D warnings -W clippy::pedantic -W clippy::nursery -W clippy::cargo

# Compare SlotMap vs DenseSlotMap storage backends
compare-storage: _ensure-uv
    @echo "📊 Comparing SlotMap vs DenseSlotMap performance (~4-6 hours)"
    uv run compare-storage-backends --bench large_scale_performance

compare-storage-large: _ensure-uv
    @echo "📊 Comparing storage backends at large scale (~8-12 hours, use on compute cluster)"
    BENCH_LARGE_SCALE=1 uv run compare-storage-backends --bench large_scale_performance

# Coverage analysis for local development (HTML output)
coverage:
    cargo tarpaulin {{_coverage_base_args}} --out Html --output-dir target/tarpaulin
    @echo "📊 Coverage report generated: target/tarpaulin/tarpaulin-report.html"

# Coverage analysis for CI (XML output for codecov/codacy)
coverage-ci:
    cargo tarpaulin {{_coverage_base_args}} --out Xml --output-dir coverage -- --skip prop_

debug-large-scale-3d-100:
    DELAUNAY_LARGE_DEBUG_N_3D=100 cargo test --test large_scale_debug debug_large_scale_3d -- --ignored --exact --nocapture

debug-large-scale-3d-1000:
    DELAUNAY_LARGE_DEBUG_N_3D=1000 cargo test --test large_scale_debug debug_large_scale_3d -- --ignored --exact --nocapture

debug-large-scale-3d-incremental-bisect total="1000":
    DELAUNAY_LARGE_DEBUG_PREFIX_TOTAL={{total}} cargo test --test large_scale_debug debug_large_scale_3d_incremental_prefix_bisect -- --ignored --nocapture

debug-large-scale-4d-100:
    DELAUNAY_LARGE_DEBUG_N_4D=100 DELAUNAY_LARGE_DEBUG_ALLOW_SKIPS=1 cargo test --test large_scale_debug debug_large_scale_4d -- --ignored --nocapture

debug-large-scale-4d:
    cargo test --test large_scale_debug debug_large_scale_4d -- --ignored --nocapture

# Default recipe shows available commands
default:
    @just --list

doc-check:
    RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --document-private-items

# Examples and validation
examples:
    ./scripts/run_all_examples.sh

# Fix (mutating): apply formatters/auto-fixes
fix: toml-fmt fmt python-fix shell-fmt markdown-fix yaml-fix
    @echo "✅ Fixes applied!"

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all -- --check

help-workflows:
    @echo "Common Just workflows:"
    @echo "  just fix               # Apply formatters/auto-fixes (mutating)"
    @echo "  just check             # Run lint/validators (non-mutating)"
    @echo "  just check-fast        # Fast compile check (cargo check)"
    @echo "  just ci                # Full CI run (checks + all tests + examples + bench compile)"
    @echo "  just ci-slow           # CI + slow tests (100+ vertices)"
    @echo "  just ci-baseline       # CI + save performance baseline"
    @echo ""
    @echo "Testing:"
    @echo "  just test              # Lib and doc tests only (fast, used by CI)"
    @echo "  just test-integration  # All integration tests (includes proptests)"
    @echo "  just test-integration-fast # Integration tests (skips proptests)"
    @echo "  just test-all          # All tests (lib + doc + integration + Python)"
    @echo "  just test-python       # Python tests only (pytest)"
    @echo "  just test-release      # All tests in release mode"
    @echo "  just test-slow         # Run slow/stress tests with --features slow-tests"
    @echo "  just test-slow-release # Slow tests in release mode (faster)"
    @echo "  just test-debug            # Run debug tools with output"
    @echo "  just debug-large-scale-3d-100   # Run large-scale 3D debug harness at 100 points (ball)"
    @echo "  just debug-large-scale-3d-1000  # Run large-scale 3D debug harness at 1000 points (ball)"
    @echo "  just debug-large-scale-3d-incremental-bisect [total] # Bisect minimal failing 3D incremental prefix (default total=1000)"
    @echo "  just debug-large-scale-4d-100   # Run large-scale 4D debug harness at 100 points (ball)"
    @echo "  just debug-large-scale-4d       # Run large-scale 4D debug harness with default point count"
    @echo "  just test-allocation   # Memory allocation profiling"
    @echo "  just examples          # Run all examples"
    @echo "  just coverage          # Generate coverage report (HTML)"
    @echo "  just coverage-ci       # Generate coverage for CI (XML)"
    @echo ""
    @echo "Quality Check Groups:"
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
    @echo "Storage Backend Comparison:"
    @echo "  just compare-storage       # Compare SlotMap vs DenseSlotMap (~4-6 hours)"
    @echo "  just compare-storage-large # Large scale comparison (~8-12 hours, compute cluster)"
    @echo ""
    @echo "Performance Analysis:"
    @echo "  just perf-help     # Show performance analysis commands"
    @echo "  just perf-check    # Check for performance regressions"
    @echo "  just perf-baseline # Save current performance as baseline"
    @echo ""
    @echo "Note: Some recipes require external tools. Run 'just setup-tools' (tooling) or 'just setup' (full env) first."

# All linting: code + documentation + configuration
lint: lint-code lint-docs lint-config

# Code linting: Rust (fmt-check, clippy, docs) + Python (ruff, ty, mypy) + Shell scripts
lint-code: fmt-check clippy doc-check python-lint shell-lint

# Configuration validation: JSON, TOML, YAML, GitHub Actions workflows
lint-config: validate-json toml-lint toml-fmt-check yaml-lint action-lint

# Documentation linting: Markdown + spell checking
lint-docs: markdown-check spell-check

markdown-check: _ensure-npx
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '*.md')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 -n100 npx markdownlint --config .markdownlint.json
    else
        echo "No markdown files found to check."
    fi

# Shell, markdown, and YAML quality
markdown-fix: _ensure-npx
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '*.md')
    if [ "${#files[@]}" -gt 0 ]; then
        echo "📝 markdownlint --fix (${#files[@]} files)"
        printf '%s\0' "${files[@]}" | xargs -0 -n100 npx markdownlint --config .markdownlint.json --fix
    else
        echo "No markdown files found to format."
    fi

markdown-lint: markdown-check

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

python-check: _ensure-uv
    uv run ruff format --check scripts/
    uv run ruff check scripts/
    just python-typecheck

# Python code quality
python-fix: _ensure-uv
    uv run ruff check scripts/ --fix
    uv run ruff format scripts/

python-lint: python-check

python-typecheck: _ensure-uv
    uv run ty check scripts/
    cd scripts && uv run mypy . --exclude tests

# Development setup
setup: setup-tools
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Setting up delaunay development environment..."
    echo "Note: Rust toolchain and components managed by rust-toolchain.toml (if present)"
    echo ""
    echo "Installing Python tooling..."
    uv sync --group dev
    echo ""
    echo "Building project..."
    cargo build
    echo "✅ Setup complete! Run 'just help-workflows' to see available commands."

# Development tooling installation (best-effort)
#
# Note: this recipe is intentionally self-contained. If it grows further, consider splitting
# it into smaller helper recipes (e.g. brew installs, cargo tool installs, verification).
setup-tools:
    #!/usr/bin/env bash
    set -euo pipefail

    echo "🔧 Ensuring tooling required by just recipes is installed..."
    echo ""

    os="$(uname -s || true)"

    have() { command -v "$1" >/dev/null 2>&1; }

    install_with_brew() {
        local formula="$1"
        if brew list --versions "$formula" >/dev/null 2>&1; then
            echo "  ✓ $formula (brew)"
        else
            echo "  ⏳ Installing $formula (brew)..."
            HOMEBREW_NO_AUTO_UPDATE=1 brew install "$formula"
        fi
    }

    brew_available=0
    if have brew; then
        brew_available=1
        echo "Using Homebrew (brew) to install missing tools..."
        install_with_brew uv
        install_with_brew jq
        install_with_brew taplo
        install_with_brew yamllint
        install_with_brew shfmt
        install_with_brew shellcheck
        install_with_brew actionlint
        install_with_brew node
        echo ""
    else
        echo "⚠️  'brew' not found. Skipping Homebrew installs."
        if [[ "$os" == "Darwin" ]]; then
            echo "Install Homebrew from https://brew.sh, or ensure required tools are on PATH."
        else
            echo "Install required tools via your system package manager, or ensure they are on PATH."
        fi
        echo "Required tools: uv, jq, taplo, yamllint, shfmt, shellcheck, actionlint, git-cliff, node+npx, typos"
        echo ""
    fi

    echo "Ensuring Rust toolchain + components..."
    if ! have rustup; then
        echo "❌ 'rustup' not found. Install Rust via https://rustup.rs and re-run: just setup-tools"
        exit 1
    fi
    rustup component add clippy rustfmt rust-docs rust-src
    echo ""

    echo "Ensuring cargo tools..."
    if ! have samply; then
        echo "  ⏳ Installing samply (cargo)..."
        cargo install --locked samply
    else
        echo "  ✓ samply"
    fi

    if ! have typos; then
        echo "  ⏳ Installing typos-cli (cargo)..."
        cargo install --locked typos-cli
    else
        echo "  ✓ typos"
    fi

    if ! have git-cliff; then
        echo "  ⏳ Installing git-cliff (cargo)..."
        cargo install --locked git-cliff
    else
        echo "  ✓ git-cliff"
    fi

    if ! have cargo-tarpaulin; then
        if [[ "$os" == "Linux" ]]; then
            echo "  ⏳ Installing cargo-tarpaulin (cargo)..."
            cargo install --locked cargo-tarpaulin
        else
            echo "  ⚠️  Skipping cargo-tarpaulin install on $os (coverage is typically Linux-only)"
        fi
    else
        echo "  ✓ cargo-tarpaulin"
    fi

    echo ""
    echo "Verifying required commands are available..."
    missing=0

    cmds=(uv jq taplo yamllint shfmt shellcheck actionlint git-cliff node npx typos)
    if [[ "$os" == "Linux" ]]; then
        cmds+=(cargo-tarpaulin)
    fi

    for cmd in "${cmds[@]}"; do
        if have "$cmd"; then
            echo "  ✓ $cmd"
        else
            echo "  ✗ $cmd"
            missing=1
        fi
    done
    if [ "$missing" -ne 0 ]; then
        echo ""
        echo "❌ Some required tools are still missing."
        if [ "$brew_available" -ne 0 ]; then
            echo "Fix the installs above (brew) and re-run: just setup-tools"
        else
            if [[ "$os" == "Darwin" ]]; then
                echo "Install Homebrew (https://brew.sh) or install the missing tools manually, then re-run: just setup-tools"
            else
                echo "Install the missing tools via your system package manager, then re-run: just setup-tools"
            fi
        fi
        exit 1
    fi

    echo ""
    echo "✅ Tooling setup complete."

# Shell scripts: lint/check (non-mutating)
shell-check: _ensure-shellcheck _ensure-shfmt
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '*.sh')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 -n4 shellcheck -x
        printf '%s\0' "${files[@]}" | xargs -0 shfmt -d
    else
        echo "No shell files found to check."
    fi

# Shell scripts: format (mutating)
shell-fmt: _ensure-shfmt
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '*.sh')
    if [ "${#files[@]}" -gt 0 ]; then
        echo "🧹 shfmt -w (${#files[@]} files)"
        printf '%s\0' "${files[@]}" | xargs -0 shfmt -w
    else
        echo "No shell files found to format."
    fi
    # Note: justfiles are not shell scripts and are excluded from shellcheck

shell-lint: shell-check

# Spell check (typos)
spell-check: _ensure-typos
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    # Use -z for NUL-delimited output to handle filenames with spaces.
    #
    # Note: For renames/copies, `git status --porcelain -z` emits *two* NUL-separated paths.
    # The ordering can differ depending on the porcelain output, so we read both and
    # spell-check whichever one exists on disk.
    while IFS= read -r -d '' status_line; do
        status="${status_line:0:2}"
        filename="${status_line:3}"

        # For renames/copies, consume the second path token to keep parsing in sync.
        # Prefer the path that exists on disk to avoid passing stale paths to typos.
        if [[ "$status" == *"R"* || "$status" == *"C"* ]]; then
            if IFS= read -r -d '' other_path; then
                if [ ! -e "$filename" ] && [ -e "$other_path" ]; then
                    filename="$other_path"
                fi
            fi
        fi

        # Skip deletions (file may no longer exist).
        if [[ "$status" == *"D"* ]]; then
            continue
        fi

        files+=("$filename")
    done < <(git status --porcelain -z --ignored=no)
    if [ "${#files[@]}" -gt 0 ]; then
        # Exclude typos.toml itself: it intentionally contains allowlisted fragments.
        printf '%s\0' "${files[@]}" | xargs -0 -n100 typos --config typos.toml --force-exclude --exclude typos.toml --
    else
        echo "No modified files to spell-check."
    fi

# Testing
# test: runs only lib and doc tests (fast, used by CI and dev)
test:
    cargo test --lib --verbose
    cargo test --doc --verbose

# test-all: runs lib, doc, integration, and Python tests (comprehensive)
test-all: test test-integration test-python
    @echo "✅ All tests passed!"

test-allocation:
    cargo test --test allocation_api --features count-allocations -- --nocapture

test-debug:
    cargo test --test circumsphere_debug_tools -- --nocapture

# test-integration: runs all integration tests (includes proptests)
test-integration:
    cargo test --tests --verbose

# test-integration-fast: runs integration tests but skips proptests (tests prefixed with `prop_`)
#
# Useful for quick local validation on changes that don't touch the property-test surface area.
# To run the full (slow) property suite, use: just test-integration
#
# Note: `--skip prop_` is a substring filter applied by the Rust test harness.
test-integration-fast:
    cargo test --tests --verbose -- --skip prop_

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

toml-fmt: _ensure-taplo
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '*.toml')
    if [ "${#files[@]}" -gt 0 ]; then
        taplo fmt "${files[@]}"
    else
        echo "No TOML files found to format."
    fi

toml-fmt-check: _ensure-taplo
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '*.toml')
    if [ "${#files[@]}" -gt 0 ]; then
        taplo fmt --check "${files[@]}"
    else
        echo "No TOML files found to check."
    fi

toml-lint: _ensure-taplo
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '*.toml')
    if [ "${#files[@]}" -gt 0 ]; then
        taplo lint "${files[@]}"
    else
        echo "No TOML files found to lint."
    fi

# File validation
validate-json: _ensure-jq
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

yaml-fix: _ensure-prettier-or-npx
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '*.yml' '*.yaml')
    if [ "${#files[@]}" -gt 0 ]; then
        echo "📝 prettier --write (YAML, ${#files[@]} files)"

        cmd=()
        if command -v prettier >/dev/null; then
            cmd=(prettier --write --print-width 120)
        elif command -v npx >/dev/null; then
            # Prefer non-interactive installs when supported (newer npm/npx).
            # NOTE: With `set -u`, expanding an empty array like "${arr[@]}" can error on older bash.
            cmd=(npx)
            if npx --help 2>&1 | grep -q -- '--yes'; then
                cmd+=(--yes)
            fi
            cmd+=(prettier --write --print-width 120)
        else
            echo "❌ 'prettier' not found. Install via npm (recommended): npm i -g prettier"
            echo "   Or install Node.js (for npx): https://nodejs.org"
            exit 1
        fi

        # Use CLI flags instead of a repo-wide prettier config: keeps the scope to YAML only.
        printf '%s\0' "${files[@]}" | xargs -0 -n100 "${cmd[@]}"
    else
        echo "No YAML files found to format."
    fi

yaml-lint: _ensure-yamllint
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '*.yml' '*.yaml')
    if [ "${#files[@]}" -gt 0 ]; then
        echo "🔍 yamllint (${#files[@]} files)"
        yamllint --strict -c .yamllint "${files[@]}"
    else
        echo "No YAML files found to lint."
    fi
