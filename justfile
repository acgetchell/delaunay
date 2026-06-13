# shellcheck disable=SC2148
# Justfile for delaunay development workflow
# Install just: https://github.com/casey/just
# Usage: just <command> or just --list

# Use bash with strict error handling for all recipes
set shell := ["bash", "-euo", "pipefail", "-c"]

home_dir := env_var_or_default("HOME", env_var_or_default("USERPROFILE", ""))
cargo_home := env_var_or_default("CARGO_HOME", home_dir + "/.cargo")
path_separator := if os_family() == "windows" { ";" } else { ":" }
export PATH := cargo_home + "/bin" + path_separator + env_var("PATH")

cargo_llvm_cov_version := "0.8.7"
dprint_version := "0.54.0"
just_version := "1.52.0"
nextest_version := "0.9.137"
rumdl_version := "0.2.14"
taplo_version := "0.10.0"
typos_version := "1.47.2"
zizmor_version := "1.25.2"

# Common cargo-llvm-cov arguments for all coverage runs.
# Excludes benches/examples from reports while allowing integration tests to
# exercise library code.
[private]
_coverage_base_args := '''--ignore-filename-regex '(^|/)(benches|examples)/' \
  --workspace --lib --tests \
  --verbose'''

_ensure-actionlint:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v uv >/dev/null || { echo "❌ 'uv' not found. See 'just setup' or https://github.com/astral-sh/uv"; exit 1; }
    uv run actionlint -version >/dev/null

_ensure-cargo-llvm-cov:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if command -v cargo-llvm-cov >/dev/null; then
        installed_version="$(cargo llvm-cov --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
    fi
    if [[ "$installed_version" != "{{ cargo_llvm_cov_version }}" ]]; then
        echo "❌ 'cargo-llvm-cov' {{ cargo_llvm_cov_version }} not found. See 'just setup-tools' or install:"
        echo "   cargo install --locked cargo-llvm-cov --version {{ cargo_llvm_cov_version }}"
        exit 1
    fi

_ensure-cargo-machete:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! cargo machete --version >/dev/null 2>&1; then
        echo "❌ 'cargo-machete' not found. Install with:"
        echo "   cargo install --locked cargo-machete"
        exit 1
    fi

_ensure-dprint:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if command -v dprint >/dev/null; then
        installed_version="$(dprint --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
    fi
    if [[ "$installed_version" != "{{ dprint_version }}" ]]; then
        echo "❌ 'dprint' {{ dprint_version }} not found. See 'just setup-tools' or install:"
        echo "   cargo install --locked dprint --version {{ dprint_version }}"
        exit 1
    fi

# Internal helpers: ensure external tooling is installed
_ensure-git-cliff:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v git-cliff >/dev/null || {
        echo "❌ 'git-cliff' not found. Install with:"
        echo "   cargo install --locked git-cliff"
        exit 1
    }

_ensure-jq:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v jq >/dev/null || { echo "❌ 'jq' not found. Install jq and ensure it is on PATH."; exit 1; }

_ensure-nextest:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if cargo nextest --version >/dev/null 2>&1; then
        installed_version="$(cargo nextest --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
    fi
    if [[ "$installed_version" != "{{ nextest_version }}" ]]; then
        echo "❌ 'cargo-nextest' {{ nextest_version }} not found. See 'just setup-tools' or install:"
        echo "   cargo install --locked cargo-nextest --version {{ nextest_version }}"
        exit 1
    fi

_ensure-rumdl:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if command -v rumdl >/dev/null; then
        installed_version="$(rumdl --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
    fi
    if [[ "$installed_version" != "{{ rumdl_version }}" ]]; then
        echo "❌ 'rumdl' {{ rumdl_version }} not found. See 'just setup-tools' or install:"
        echo "   cargo install --locked rumdl --version {{ rumdl_version }}"
        exit 1
    fi

_ensure-shellcheck:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v uv >/dev/null || { echo "❌ 'uv' not found. See 'just setup' or https://github.com/astral-sh/uv"; exit 1; }
    uv run shellcheck --version >/dev/null

_ensure-shfmt:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v uv >/dev/null || { echo "❌ 'uv' not found. See 'just setup' or https://github.com/astral-sh/uv"; exit 1; }
    uv run shfmt --version >/dev/null

# Internal helper: ensure taplo is installed
_ensure-taplo:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if command -v taplo >/dev/null; then
        installed_version="$(taplo --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
    fi
    if [[ "$installed_version" != "{{ taplo_version }}" ]]; then
        echo "❌ 'taplo' {{ taplo_version }} not found. See 'just setup-tools' or install:"
        echo "   cargo install --locked taplo-cli --version {{ taplo_version }}"
        exit 1
    fi

# Internal helper: ensure typos-cli is installed
_ensure-typos:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if command -v typos >/dev/null; then
        installed_version="$(typos --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
    fi
    if [[ "$installed_version" != "{{ typos_version }}" ]]; then
        echo "❌ 'typos' {{ typos_version }} not found. See 'just setup-tools' or install:"
        echo "   cargo install --locked typos-cli --version {{ typos_version }}"
        exit 1
    fi

# Internal helper: ensure uv is installed
_ensure-uv:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v uv >/dev/null || { echo "❌ 'uv' not found. See 'just setup' or https://github.com/astral-sh/uv"; exit 1; }

_ensure-yamllint:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v uv >/dev/null || { echo "❌ 'uv' not found. See 'just setup' or https://github.com/astral-sh/uv"; exit 1; }
    uv run yamllint --version >/dev/null

_ensure-zizmor:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if command -v zizmor >/dev/null; then
        installed_version="$(zizmor --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
    fi
    if [[ "$installed_version" != "{{ zizmor_version }}" ]]; then
        echo "❌ 'zizmor' {{ zizmor_version }} not found. See 'just setup-tools' or install:"
        echo "   cargo install --locked zizmor --version {{ zizmor_version }}"
        exit 1
    fi

# GitHub Actions workflow validation
action-lint: _ensure-actionlint
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '.github/workflows/*.yml' '.github/workflows/*.yaml')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 uv run actionlint
    else
        echo "No workflow files found to lint."
    fi

# Benchmark recipes that produce performance numbers use Cargo's perf profile.
bench:
    cargo bench --workspace --profile perf

# Allocation-contract microbenchmarks for public hot paths.
bench-allocations:
    cargo bench --profile perf --bench allocation_hot_paths --features count-allocations -- --noplot

# CI regression benchmarks with the perf profile.
bench-ci:
    cargo bench --profile perf --bench ci_performance_suite

# Compile benchmarks without running them. Manifest lints enforce the warning
# policy without using RUSTFLAGS that fragment Cargo artifact caches.
bench-compile:
    cargo bench --workspace --no-run

# Generate performance summary with fresh perf-profile benchmark runs (for releases)
bench-perf-summary: _ensure-uv
    uv run benchmark-utils generate-summary --run-benchmarks --profile perf

# Smoke-test benchmark harnesses with minimal samples; not for performance data.
# Criterion requires sample_size >= 10; use the minimum with short measurement/warm-up windows.
bench-smoke:
    CRIT_SAMPLE_SIZE=10 CRIT_MEASUREMENT_MS=500 CRIT_WARMUP_MS=200 cargo bench --workspace --profile perf

# Compile benchmarks and integration tests without running. This catches
# release-profile-only warnings (e.g. cfg-gated unused-mut) that debug-mode
# clippy/test won't see.
bench-test-compile: _ensure-nextest
    cargo bench --workspace --no-run
    cargo nextest run --release --tests --no-run

# Build commands
build:
    cargo build

build-release:
    cargo build --release

# Changelog management (git-cliff + post-processing + archiving + rumdl formatting)
changelog: _ensure-git-cliff _ensure-rumdl python-sync
    #!/usr/bin/env bash
    set -euo pipefail
    GIT_CLIFF_OFFLINE=true git-cliff -o CHANGELOG.md
    uv run postprocess-changelog
    uv run archive-changelog
    rumdl fmt --silent CHANGELOG.md docs/archive/changelog/*.md

changelog-tag version:
    just tag {{ version }}

changelog-unreleased version: _ensure-git-cliff _ensure-rumdl python-sync
    #!/usr/bin/env bash
    set -euo pipefail
    GIT_CLIFF_OFFLINE=true git-cliff --tag {{ version }} -o CHANGELOG.md
    uv run postprocess-changelog
    uv run archive-changelog
    rumdl fmt --silent CHANGELOG.md docs/archive/changelog/*.md

changelog-update: changelog
    @echo "📝 Changelog updated successfully!"
    @echo "To create a git tag with changelog content for a specific version, run:"
    @echo "  just tag <version>  # e.g., just tag v0.4.2"

# Check (non-mutating): run all linters/validators.
check: lint
    @echo "✅ Checks complete!"

# Fast compile check (no binary produced)
check-fast:
    cargo check

# CI simulation: comprehensive validation.
# Runs: checks + test workflow + examples
ci: check test examples
    @echo "🎯 CI checks complete!"

# CI followed by an explicit persistent local baseline refresh.
ci-baseline ref="main":
    just ci
    just perf-baseline {{ ref }}

# CI plus the explicit slow correctness bucket.
ci-slow: ci test-slow
    @echo "✅ CI + slow tests passed!"

# Validate CITATION.cff against the Citation File Format schema.
citation-check: _ensure-uv
    uvx --from cffconvert==2.0.0 cffconvert --validate -i CITATION.cff

# Clean build artifacts
clean:
    cargo clean
    rm -rf target/llvm-cov
    rm -rf coverage_report
    rm -rf coverage

# Code quality and formatting
clippy:
    cargo clippy --workspace --all-targets -- -D warnings -W clippy::pedantic -W clippy::nursery -W clippy::cargo

    # All features, split by target class so feature-gated benchmark/example
    # code stays covered without rebuilding every target in one oversized graph.
    cargo clippy --workspace --lib --tests --all-features -- -D warnings -W clippy::pedantic -W clippy::nursery -W clippy::cargo
    cargo clippy --workspace --benches --examples --all-features -- -D warnings -W clippy::pedantic -W clippy::nursery -W clippy::cargo

# Coverage analysis for local development (HTML output)
coverage: _ensure-cargo-llvm-cov
    mkdir -p target/llvm-cov
    cargo llvm-cov {{ _coverage_base_args }} --html --output-dir target/llvm-cov
    @echo "📊 Coverage report generated: target/llvm-cov/html/index.html"

# Coverage analysis for CI (XML output for codecov/codacy)
coverage-ci: _ensure-cargo-llvm-cov
    mkdir -p coverage
    cargo llvm-cov {{ _coverage_base_args }} --cobertura --output-path coverage/cobertura.xml -- --skip prop_

debug-large-scale-2d n="36000" repair_every="1": _ensure-nextest
    DELAUNAY_BULK_PROGRESS_EVERY=2000 DELAUNAY_LARGE_DEBUG_MAX_RUNTIME_SECS=1800 DELAUNAY_LARGE_DEBUG_N_2D={{ n }} DELAUNAY_LARGE_DEBUG_REPAIR_EVERY={{ repair_every }} cargo nextest run --release --profile slow --features slow-tests --test large_scale_debug debug_large_scale_2d -- --exact --nocapture

debug-large-scale-3d n="7500" repair_every="1": _ensure-nextest
    DELAUNAY_BULK_PROGRESS_EVERY=500 DELAUNAY_LARGE_DEBUG_MAX_RUNTIME_SECS=1800 DELAUNAY_LARGE_DEBUG_N_3D={{ n }} DELAUNAY_LARGE_DEBUG_REPAIR_EVERY={{ repair_every }} cargo nextest run --release --profile slow --features slow-tests --test large_scale_debug debug_large_scale_3d -- --exact --nocapture

debug-large-scale-4d n="800" repair_every="1": _ensure-nextest
    DELAUNAY_BULK_PROGRESS_EVERY=100 DELAUNAY_LARGE_DEBUG_MAX_RUNTIME_SECS=1800 DELAUNAY_LARGE_DEBUG_N_4D={{ n }} DELAUNAY_LARGE_DEBUG_REPAIR_EVERY={{ repair_every }} cargo nextest run --release --profile slow --features slow-tests --test large_scale_debug debug_large_scale_4d -- --exact --nocapture

debug-large-scale-5d n="140" repair_every="1": _ensure-nextest
    DELAUNAY_BULK_PROGRESS_EVERY=20 DELAUNAY_LARGE_DEBUG_MAX_RUNTIME_SECS=1800 DELAUNAY_LARGE_DEBUG_N_5D={{ n }} DELAUNAY_LARGE_DEBUG_REPAIR_EVERY={{ repair_every }} cargo nextest run --release --profile slow --features slow-tests --test large_scale_debug debug_large_scale_5d -- --exact --nocapture

# Default recipe shows available commands
default:
    @just --list

doc-check:
    RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --document-private-items

# Examples and validation
examples:
    ./scripts/run_all_examples.sh

# Fix (mutating): apply formatters/auto-fixes
fix: toml-fix fmt python-fix shell-fix markdown-fix yaml-fix
    @echo "✅ Fixes applied!"

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all -- --check

help-workflows:
    @echo "Recommended Just workflow:"
    @echo "  just check             # Run all non-mutating lints/validators"
    @echo "  just fix               # Apply formatters/auto-fixes (mutating)"
    @echo "  just test              # Run tests + default-profile benchmark/release compile smoke"
    @echo "  just ci                # Comprehensive checks + tests + examples"
    @echo ""
    @echo "Focused testing:"
    @echo "  just test-unit         # Lib and doc tests only"
    @echo "  just test-integration  # All integration tests (includes proptests)"
    @echo "  just test-integration-fast # Integration tests (skips proptests)"
    @echo "  just test-python       # Python tests only (pytest)"
    @echo "  just test-slow         # Run correctness tests over the 10s default-suite budget"
    @echo "  just examples          # Run all examples"
    @echo ""
    @echo "Active large-scale debugging:"
    @echo "  just test-diagnostics      # Run diagnostics tools with output"
    @echo "  just debug-large-scale-2d [n] [repair_every] # 2D acceptance/profiling (defaults n=36000, repair_every=1)"
    @echo "  just debug-large-scale-3d [n] [repair_every] # Issue #341: 3D scalability (defaults n=7500, repair_every=1)"
    @echo "  just debug-large-scale-4d [n] [repair_every] # Issue #340: 4D large-scale runtime (defaults n=800, repair_every=1)"
    @echo "  just debug-large-scale-5d [n] [repair_every] # Issue #342: 5D feasibility (defaults n=140, repair_every=1)"
    @echo ""
    @echo "Benchmark workflows:"
    @echo "  just bench-smoke        # Smoke-test benchmark harnesses (minimal samples)"
    @echo "  just bench              # Run all benchmarks with perf profile (ThinLTO)"
    @echo "  just bench-ci           # CI regression benchmarks with perf profile (~5-10 min)"
    @echo "  just perf-large-scale-smoke [max_secs] # Quick pre-push 2D-5D wall-clock guard (default 60s)"
    @echo "  just perf-no-regressions [threshold] # Fast pre-PR 2D-5D regression guard (default 7.5%)"
    @echo "  just perf-baseline [ref] # Persist/update default local baseline (default: main)"
    @echo "  just perf-baseline-to <out> [ref] # Generate scratch baseline without replacing default"
    @echo "  just bench-perf-summary # Generate perf-profile release summary (~30-45 min)"
    @echo "  just profile [toolchain] [code_ref] # Run ci_performance_suite for a compiler/code pair"
    @echo ""
    @echo "Larger/optional workflows:"
    @echo "  just ci-slow             # CI + slow correctness tests"
    @echo "  just ci-baseline         # CI + persist default performance baseline"
    @echo "  just coverage            # Generate coverage report (HTML)"
    @echo "  just semgrep             # Run repository-owned Semgrep rules"
    @echo ""
    @echo "Use 'just --list' for every granular recipe."

# Check JSON files parse cleanly.
json-check: _ensure-jq
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        [ -f "$file" ] && files+=("$file")
    done < <(git ls-files -z '*.json')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 -n1 jq empty
    else
        echo "No JSON files found to check."
    fi

# All linting: code + documentation + configuration
lint: lint-code lint-docs lint-config

# Code linting: Rust (fmt-check, clippy, docs, Semgrep) + Python (Ruff, Ty) + Shell scripts
lint-code: fmt-check clippy doc-check semgrep semgrep-test python-lint shell-lint

# Configuration checks: JSON, TOML, YAML/CFF, GitHub Actions workflows
lint-config: json-check toml-check toml-lint toml-fmt-check yaml-check citation-check action-lint zizmor

# Documentation linting: Markdown + spell checking
lint-docs: markdown-check spell-check

markdown-check: _ensure-rumdl
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        case "$file" in
            CHANGELOG.md|docs/archive/*) continue ;;
        esac
        files+=("$file")
    done < <(git ls-files -z '*.md')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 -n100 rumdl check
        violations=0
        for file in "${files[@]}"; do
            line_number=0
            while IFS= read -r line || [[ -n "$line" ]]; do
                line_number=$((line_number + 1))
                if [ "${#line}" -gt 160 ]; then
                    printf '%s:%d: line length %d exceeds 160\n' "$file" "$line_number" "${#line}" >&2
                    violations=$((violations + 1))
                fi
            done < "$file"
        done
        if [ "$violations" -gt 0 ]; then
            echo "Markdown raw line-length check failed." >&2
            exit 1
        fi
    else
        echo "No markdown files found to check."
    fi

# Shell, markdown, and YAML quality
markdown-fix: _ensure-rumdl
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        case "$file" in
            CHANGELOG.md|docs/archive/*) continue ;;
        esac
        files+=("$file")
    done < <(git ls-files -z '*.md')
    if [ "${#files[@]}" -gt 0 ]; then
        echo "📝 rumdl check --fix (${#files[@]} files)"
        printf '%s\0' "${files[@]}" | xargs -0 -n100 rumdl check --fix
    else
        echo "No markdown files found to format."
    fi

markdown-lint: markdown-check

# Generate a same-machine dev-mode baseline for a GitHub ref.
perf-baseline ref="main": _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    uv run benchmark-utils generate-ref-baseline --ref "{{ ref }}" --out baseline-artifact --dev

perf-baseline-to out ref="main": _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    uv run benchmark-utils generate-ref-baseline --ref "{{ ref }}" --out "{{ out }}" --dev

perf-compare file threshold="7.5": _ensure-uv
    uv run benchmark-utils compare --baseline "{{ file }}" --threshold {{ threshold }} --dev

perf-help:
    @echo "Performance Analysis Commands:"
    @echo "  just perf-large-scale-smoke # Quick pre-push 2D-5D wall-clock smoke guard"
    @echo "  just perf-no-regressions   # Fast pre-PR guard with a cached same-machine main baseline"
    @echo "  just perf-vs-ref <ref> [threshold] # Compare current tree vs a cached same-machine ref baseline"
    @echo "  just perf-baseline [ref]    # Persist/update baseline-artifact for a GitHub ref (default: main)"
    @echo "  just perf-baseline-to <out> [ref] # Generate a scratch baseline artifact without replacing the default"
    @echo "  just perf-compare <file> [threshold] # Compare current tree with a specific dev-mode baseline"
    @echo "  just bench-smoke           # Smoke-test benchmark harnesses"
    @echo ""
    @echo "Profiling Commands:"
    @echo "  just profile               # Run ci_performance_suite for the current tree/toolchain"
    @echo "  just profile [toolchain] [code_ref]"
    @echo "                              # Run ci_performance_suite for a compiler/code pair"
    @echo "  just profile-dev           # Samply profile 3D construction in profiling_suite"
    @echo "  just profile-mem           # Samply profile memory allocations (with count-allocations feature)"
    @echo ""
    @echo "Benchmark System (Delaunay-specific):"
    @echo "  just perf-large-scale-smoke # Pre-push guard using debug-large-scale 2D-5D with a short cap"
    @echo "  just perf-no-regressions   # Reuse cached main baseline, compare current tree"
    @echo "  just perf-vs-ref <ref>     # Reuse cached ref baseline, compare current tree"
    @echo "  just perf-baseline [ref]   # Persist baseline-artifact/baseline_results.txt from a GitHub ref"
    @echo "  just perf-baseline-to <out> [ref] # Generate an alternate local baseline artifact directory"
    @echo "  just perf-compare <file>   # Compare against a specific dev-mode baseline"
    @echo "  just bench                 # Full benchmark suite with perf profile"
    @echo "  just bench-ci              # CI benchmark suite with perf profile"
    @echo "  just bench-allocations     # Allocation-contract microbenchmarks"
    @echo "  just perf-no-regressions   # Fast pre-PR 2D-5D regression guard"
    @echo "  just bench-smoke           # Smoke-test benchmark harnesses"
    @echo ""
    @echo "Environment Variables (Benchmark Configuration):"
    @echo "  CRIT_SAMPLE_SIZE=N         # Number of samples per benchmark"
    @echo "  CRIT_MEASUREMENT_MS=N      # Measurement time in milliseconds"
    @echo "  CRIT_WARMUP_MS=N           # Warm-up time in milliseconds"
    @echo "  DELAUNAY_BENCH_SEED=N      # Random seed (decimal or 0x-hex)"
    @echo ""
    @echo "Examples:"
    @echo "  just perf-large-scale-smoke # Run before pushing to catch obvious performance drift"
    @echo "  just perf-no-regressions   # Recommended local PR performance guard"
    @echo "  just perf-vs-ref v0.7.8    # Compare current branch against the v0.7.8 release locally"
    @echo "  just perf-baseline         # Persist/update default local baseline for GitHub main"
    @echo "  just perf-baseline v0.7.5  # Persist/update default local baseline for a release tag"
    @echo "  just perf-baseline-to /tmp/delaunay-main-baseline"
    @echo "                              # Generate scratch main baseline without overwriting baseline-artifact"
    @echo "  CRIT_SAMPLE_SIZE=100 just bench  # Custom sample size"
    @echo "  just bench-ci              # Final optimized CI-suite benchmark run"
    @echo "  just profile v0.7.5        # v0.7.5 code on its declared Rust toolchain"
    @echo "  just profile 1.96.0        # Current tree on Rust 1.96.0"
    @echo "  just profile 1.96.0 v0.7.5 # v0.7.5 code on Rust 1.96.0"

# Quick pre-push 2D-5D large-scale wall-clock smoke guard.
perf-large-scale-smoke max_secs="60": _ensure-nextest
    #!/usr/bin/env bash
    set -euo pipefail

    max_secs="{{ max_secs }}"
    if [[ ! "$max_secs" =~ ^[1-9][0-9]*$ ]]; then
        echo "❌ max_secs must be a positive integer, got: $max_secs" >&2
        exit 2
    fi

    status=0
    failures=()

    run_case() {
        local dimension="$1"
        local test_name="$2"
        local n_env="$3"
        local n_points="$4"
        local progress_every="$5"

        echo ""
        echo "▶ ${dimension}: ${test_name} (${n_points} vertices, ${max_secs}s cap)"
        if env \
            DELAUNAY_BULK_PROGRESS_EVERY="$progress_every" \
            DELAUNAY_LARGE_DEBUG_MAX_RUNTIME_SECS="$max_secs" \
            "$n_env=$n_points" \
            DELAUNAY_LARGE_DEBUG_REPAIR_EVERY=1 \
            cargo nextest run --release --profile slow --features slow-tests --test large_scale_debug "$test_name" -- --exact --nocapture; then
            echo "✅ ${dimension} completed within the ${max_secs}s test-runtime cap"
        else
            local code=$?
            echo "❌ ${dimension} failed or exceeded the ${max_secs}s test-runtime cap (exit ${code})"
            failures+=("$dimension")
            status=1
        fi
    }

    run_case "2D" "debug_large_scale_2d" "DELAUNAY_LARGE_DEBUG_N_2D" "36000" "2000"
    run_case "3D" "debug_large_scale_3d" "DELAUNAY_LARGE_DEBUG_N_3D" "7500" "500"
    run_case "4D" "debug_large_scale_4d" "DELAUNAY_LARGE_DEBUG_N_4D" "800" "100"
    run_case "5D" "debug_large_scale_5d" "DELAUNAY_LARGE_DEBUG_N_5D" "140" "20"

    if (( ${#failures[@]} > 0 )); then
        echo ""
        echo "❌ Large-scale smoke guard failed for: ${failures[*]}"
        exit "$status"
    fi

    echo ""
    echo "✅ Large-scale smoke guard passed for 2D-5D"

# Fast pre-PR performance guard against a cached same-machine main baseline.
perf-no-regressions threshold="7.5": _ensure-uv
    uv run benchmark-utils compare-ref --ref main --threshold {{ threshold }} --dev --output benches/worktree_vs_main_compare_results.txt

perf-vs-ref ref threshold="7.5": _ensure-uv
    uv run benchmark-utils compare-ref --ref "{{ ref }}" --threshold {{ threshold }} --dev

# Run the selected CI benchmark suite for one compiler/code pair.
profile toolchain="" code_ref="current":
    #!/usr/bin/env bash
    set -euo pipefail

    command -v rustup >/dev/null || { echo "❌ 'rustup' not found. Install Rust via https://rustup.rs"; exit 1; }

    repo_root="$(pwd)"
    requested_toolchain="{{ toolchain }}"
    requested_ref="{{ code_ref }}"
    workdir="$repo_root"
    cleanup_worktree=0

    cleanup() {
        if [[ "$cleanup_worktree" -eq 1 ]]; then
            git worktree remove --force "$workdir" >/dev/null 2>&1 || true
            rm -rf "$(dirname "$workdir")"
        fi
    }

    if [[ "$requested_ref" == "current" && -n "$requested_toolchain" ]]; then
        if [[ ! "$requested_toolchain" =~ ^([0-9]+(\.[0-9]+){0,2}|stable|beta|nightly)([-+].*)?$ ]]; then
            requested_ref="$requested_toolchain"
            requested_toolchain=""
        fi
    fi

    if [[ "$requested_ref" != "current" && "$requested_ref" != "." ]]; then
        tmp_parent="$(mktemp -d "${TMPDIR:-/tmp}/delaunay-profile.XXXXXX")"
        workdir="$tmp_parent/worktree"
        cleanup_worktree=1
        trap cleanup EXIT
        git worktree add --detach "$workdir" "$requested_ref"
    fi

    if [[ -z "$requested_toolchain" ]]; then
        requested_toolchain="$(
            grep -E '^[[:space:]]*channel[[:space:]]*=' "$workdir/rust-toolchain.toml" \
                | head -n 1 \
                | cut -d '=' -f 2 \
                | tr -d ' "' \
                || true
        )"
    fi

    if [[ -z "$requested_toolchain" ]]; then
        echo "❌ No toolchain argument provided and no rust-toolchain.toml channel found."
        exit 1
    fi

    safe_ref="$(
        if [[ "$requested_ref" == "current" || "$requested_ref" == "." ]]; then
            printf 'current'
        else
            printf '%s' "$requested_ref"
        fi | tr -c 'A-Za-z0-9._-' '_'
    )"
    safe_toolchain="$(printf '%s' "$requested_toolchain" | tr -c 'A-Za-z0-9._-' '_')"
    run_dir="$repo_root/target/profile-runs/${safe_ref}-${safe_toolchain}"
    mkdir -p "$run_dir"

    echo "📌 Code ref: $requested_ref"
    echo "🦀 Rust toolchain: $requested_toolchain"
    echo "📊 Benchmark: ci_performance_suite"
    echo "📁 Results: $run_dir"

    rustup toolchain install "$requested_toolchain" --profile minimal

    {
        echo "# Profile Run"
        echo
        echo "- Code ref: $requested_ref"
        echo "- Workdir: $workdir"
        echo "- Commit: $(git -C "$workdir" rev-parse HEAD)"
        echo "- Dirty tree: $(if [[ "$workdir" == "$repo_root" && -n "$(git status --short)" ]]; then echo yes; else echo no; fi)"
        echo "- Requested toolchain: $requested_toolchain"
        echo "- rustc: $(rustup run "$requested_toolchain" rustc --version)"
        echo "- cargo: $(rustup run "$requested_toolchain" cargo --version)"
        echo "- Cargo profile: cargo bench --profile perf"
        echo "- Benchmark harness: ci_performance_suite"
    } > "$run_dir/profile_metadata.md"

    (
        cd "$workdir"
        CARGO_TARGET_DIR="$run_dir/target" \
            rustup run "$requested_toolchain" cargo bench --profile perf --bench ci_performance_suite \
            2>&1 | tee "$run_dir/ci_performance_suite.log"
    )

profile-dev:
    PROFILING_DEV_MODE=1 samply record cargo bench --profile perf --bench profiling_suite -- "construction/3D/5000v/construct"

profile-mem:
    samply record cargo bench --profile perf --bench profiling_suite --features count-allocations -- memory_profiling

# Pre-publish validation: checks crates.io metadata rules that cargo publish --dry-run does NOT catch
publish-check: _ensure-jq
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🔍 Validating crates.io metadata..."
    errors=0

    # Keywords: max 5, each ≤20 chars, ASCII alphanumeric/hyphen only
    keywords=$(cargo metadata --no-deps --format-version=1 2>/dev/null \
        | jq -r '.packages[0].keywords[]')
    count=0
    while IFS= read -r kw; do
        [[ -z "$kw" ]] && continue
        count=$((count + 1))
        if (( ${#kw} > 20 )); then
            echo "  ❌ keyword '${kw}' exceeds 20-char limit (${#kw} chars)"
            errors=1
        fi
        if ! [[ "$kw" =~ ^[a-zA-Z0-9_-]+$ ]]; then
            echo "  ❌ keyword '${kw}' contains invalid characters"
            errors=1
        fi
    done <<< "$keywords"
    if (( count > 5 )); then
        echo "  ❌ too many keywords ($count > 5)"
        errors=1
    fi
    echo "  ✓ keywords ($count): $keywords"

    # Categories: max 5
    cat_count=$(cargo metadata --no-deps --format-version=1 2>/dev/null \
        | jq '.packages[0].categories | length')
    if (( cat_count > 5 )); then
        echo "  ❌ too many categories ($cat_count > 5)"
        errors=1
    fi
    echo "  ✓ categories ($cat_count)"

    # Description: required, ≤1000 chars
    desc=$(cargo metadata --no-deps --format-version=1 2>/dev/null \
        | jq -r '.packages[0].description // ""')
    if [[ -z "$desc" ]]; then
        echo "  ❌ description is missing"
        errors=1
    elif (( ${#desc} > 1000 )); then
        echo "  ❌ description exceeds 1000-char limit (${#desc} chars)"
        errors=1
    fi
    echo "  ✓ description (${#desc} chars)"

    if (( errors )); then
        echo ""
        echo "❌ Metadata validation failed. Fix Cargo.toml before publishing."
        exit 1
    fi

    echo ""
    echo "📦 Running cargo publish --dry-run..."
    cargo publish --locked --allow-dirty --dry-run
    echo ""
    echo "✅ Publish check passed!"

python-check: _ensure-uv
    uv run ruff format --check scripts/
    uv run ruff check scripts/
    just python-typecheck

# Python code quality
python-fix: _ensure-uv
    uv run ruff check scripts/ --fix
    uv run ruff format scripts/

python-lint: python-check

python-sync: _ensure-uv
    uv sync --group dev

python-typecheck: _ensure-uv
    uv run ty check scripts/ --error all

# Repository-owned Semgrep rules for project-specific Rust diagnostics.
semgrep: _ensure-uv
    uv run semgrep --error --strict --timeout 30 --config semgrep.yaml .

semgrep-test: _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    config_dir="$(mktemp -d "${TMPDIR:-/tmp}/delaunay-semgrep-config.XXXXXX")"
    state_root="$(mktemp -d "${TMPDIR:-/tmp}/delaunay-semgrep-state.XXXXXX")"
    cleanup() {
        rm -rf "$config_dir" "$state_root"
    }
    trap cleanup EXIT

    # Semgrep directory test mode maps fixture paths to config paths, so mirror
    # each fixture to the shared config while keeping semgrep.yaml authoritative.
    # Run one fixture/config pair per Semgrep process so Windows does not race
    # Semgrep's shared settings file across test-mode worker processes.
    while IFS= read -r -d '' fixture; do
        rel="${fixture#tests/semgrep/}"
        config_path="$config_dir/${rel%.*}.yaml"
        state_dir="$state_root/${rel%.*}"
        mkdir -p "$(dirname "$config_path")"
        mkdir -p "$state_dir"
        ln -s "$PWD/semgrep.yaml" "$config_path"

        SEMGREP_SEND_METRICS=off SEMGREP_SETTINGS_FILE="$state_dir/settings.yml" uv run semgrep scan --test --strict --config "$config_path" "$fixture"
    done < <(find tests/semgrep -type f ! -name '*.fixed' -print0)

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
# it into smaller helper recipes (e.g. cargo tool installs, verification).
setup-tools:
    #!/usr/bin/env bash
    set -euo pipefail

    echo "🔧 Ensuring tooling required by just recipes is installed..."
    echo ""

    have() { command -v "$1" >/dev/null 2>&1; }

    echo "This recipe installs pinned Rust CLI tools through cargo."
    echo "External prerequisites that must already be on PATH: uv, jq, rustup, cargo."
    echo ""

    echo "Ensuring uv-managed Python tooling..."
    if ! have uv; then
        echo "❌ 'uv' not found. Install uv from https://github.com/astral-sh/uv and re-run: just setup-tools"
        exit 1
    fi
    uv sync --group dev
    echo ""

    echo "Ensuring Rust toolchain + components..."
    if ! have rustup; then
        echo "❌ 'rustup' not found. Install Rust via https://rustup.rs and re-run: just setup-tools"
        exit 1
    fi
    rustup component add clippy rustfmt rust-docs rust-src
    echo ""

    echo "Ensuring cargo tools..."
    installed_just_version=""
    if command -v just >/dev/null; then
        installed_just_version="$(just --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
    fi
    if [[ "$installed_just_version" != "{{ just_version }}" ]]; then
        echo "  ⏳ Installing just {{ just_version }} (cargo)..."
        cargo install --locked just --version {{ just_version }}
    else
        echo "  ✓ just {{ just_version }}"
    fi

    if ! have samply; then
        echo "  ⏳ Installing samply (cargo)..."
        cargo install --locked samply
    else
        echo "  ✓ samply"
    fi

    installed_taplo_version=""
    if command -v taplo >/dev/null; then
        installed_taplo_version="$(taplo --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
    fi
    if [[ "$installed_taplo_version" != "{{ taplo_version }}" ]]; then
        echo "  ⏳ Installing taplo-cli {{ taplo_version }} (cargo)..."
        cargo install --locked taplo-cli --version {{ taplo_version }}
    else
        echo "  ✓ taplo {{ taplo_version }}"
    fi

    installed_dprint_version=""
    if command -v dprint >/dev/null; then
        installed_dprint_version="$(dprint --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
    fi
    if [[ "$installed_dprint_version" != "{{ dprint_version }}" ]]; then
        echo "  ⏳ Installing dprint {{ dprint_version }} (cargo)..."
        cargo install --locked dprint --version {{ dprint_version }}
    else
        echo "  ✓ dprint {{ dprint_version }}"
    fi

    installed_rumdl_version=""
    if command -v rumdl >/dev/null; then
        installed_rumdl_version="$(rumdl --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
    fi
    if [[ "$installed_rumdl_version" != "{{ rumdl_version }}" ]]; then
        echo "  ⏳ Installing rumdl {{ rumdl_version }} (cargo)..."
        cargo install --locked rumdl --version {{ rumdl_version }}
    else
        echo "  ✓ rumdl {{ rumdl_version }}"
    fi

    installed_typos_version=""
    if command -v typos >/dev/null; then
        installed_typos_version="$(typos --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
    fi
    if [[ "$installed_typos_version" != "{{ typos_version }}" ]]; then
        echo "  ⏳ Installing typos-cli {{ typos_version }} (cargo)..."
        cargo install --locked typos-cli --version {{ typos_version }}
    else
        echo "  ✓ typos {{ typos_version }}"
    fi

    if ! have git-cliff; then
        echo "  ⏳ Installing git-cliff (cargo)..."
        cargo install --locked git-cliff
    else
        echo "  ✓ git-cliff"
    fi

    if ! rustup component list --installed | grep -q '^llvm-tools'; then
        echo "  ⏳ Installing llvm-tools-preview (rustup)..."
        rustup component add llvm-tools-preview
    else
        echo "  ✓ llvm-tools-preview"
    fi

    installed_nextest_version=""
    if cargo nextest --version >/dev/null 2>&1; then
        installed_nextest_version="$(cargo nextest --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
    fi
    if [[ "$installed_nextest_version" != "{{ nextest_version }}" ]]; then
        echo "  ⏳ Installing cargo-nextest {{ nextest_version }} (cargo)..."
        cargo install --locked cargo-nextest --version {{ nextest_version }}
    else
        echo "  ✓ cargo-nextest {{ nextest_version }}"
    fi

    installed_llvm_cov_version=""
    if command -v cargo-llvm-cov >/dev/null; then
        installed_llvm_cov_version="$(cargo llvm-cov --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
    fi
    if [[ "$installed_llvm_cov_version" != "{{ cargo_llvm_cov_version }}" ]]; then
        echo "  ⏳ Installing cargo-llvm-cov {{ cargo_llvm_cov_version }} (cargo)..."
        cargo install --locked cargo-llvm-cov --version {{ cargo_llvm_cov_version }}
    else
        echo "  ✓ cargo-llvm-cov {{ cargo_llvm_cov_version }}"
    fi

    installed_zizmor_version=""
    if command -v zizmor >/dev/null; then
        installed_zizmor_version="$(zizmor --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
    fi
    if [[ "$installed_zizmor_version" != "{{ zizmor_version }}" ]]; then
        echo "  ⏳ Installing zizmor {{ zizmor_version }} (cargo)..."
        cargo install --locked zizmor --version {{ zizmor_version }}
    else
        echo "  ✓ zizmor {{ zizmor_version }}"
    fi

    echo ""
    echo "Verifying required commands are available..."
    missing=0

    cmds=(uv jq taplo dprint rumdl git-cliff typos zizmor)
    cmds+=(cargo-nextest cargo-llvm-cov)

    for cmd in "${cmds[@]}"; do
        if have "$cmd"; then
            echo "  ✓ $cmd"
        else
            echo "  ✗ $cmd"
            missing=1
        fi
    done

    for cmd in actionlint shellcheck shfmt yamllint; do
        if uv run "$cmd" --version >/dev/null 2>&1 || uv run "$cmd" -version >/dev/null 2>&1; then
            echo "  ✓ $cmd (uv)"
        else
            echo "  ✗ $cmd (uv)"
            missing=1
        fi
    done
    if [ "$missing" -ne 0 ]; then
        echo ""
        echo "❌ Some required tools are still missing."
        echo "Install the missing prerequisites or cargo tools, then re-run: just setup-tools"
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
        printf '%s\0' "${files[@]}" | xargs -0 -n4 uv run shellcheck -x
        printf '%s\0' "${files[@]}" | xargs -0 uv run shfmt -d
    else
        echo "No shell files found to check."
    fi

shell-fix: shell-fmt

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
        printf '%s\0' "${files[@]}" | xargs -0 uv run shfmt -w
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

# Create an annotated git tag from the CHANGELOG.md section for the given version
tag version: python-sync
    uv run tag-release {{ version }}

# Replace an existing annotated tag from the CHANGELOG.md section.
tag-force version: python-sync
    uv run tag-release {{ version }} --force

# Testing
# test: runs default-profile benchmark/release compile checks plus all tests.
test: bench-test-compile test-all
    @echo "✅ Test workflow passed!"

# test-all: runs lib, doc, integration, and Python tests (comprehensive)
test-all: test-unit test-integration test-python
    @echo "✅ All tests passed!"

test-allocation: _ensure-nextest
    cargo nextest run --profile ci --test allocation_api --features count-allocations -- --nocapture

test-diagnostics: _ensure-nextest
    cargo nextest run --profile ci --test circumsphere_debug_tools --features diagnostics -- --nocapture

# test-integration: runs all default integration tests under the 10s per-test budget.
test-integration: _ensure-nextest
    cargo nextest run --release --profile ci --tests

# test-integration-fast: runs integration tests but skips proptests (tests prefixed with `prop_`)
#
# Useful for quick local validation on changes that don't touch the property-test surface area.
# To run the full (slow) property suite, use: just test-integration
#
# Note: `--skip prop_` is a substring filter applied by the Rust test harness.
test-integration-fast: _ensure-nextest
    cargo nextest run --release --profile ci --tests -- --skip prop_

test-python: _ensure-uv
    uv run pytest

test-release: _ensure-nextest
    cargo nextest run --release --profile ci
    cargo test --doc --release

# Run correctness tests that exceed the 10s default-suite budget.
# Slow tests run in release mode because debug exact-predicate paths can turn
# a slow correctness check into a local timeout.
test-slow: _ensure-nextest
    cargo nextest run --release --profile slow --features slow-tests
    cargo test --doc --release --features slow-tests

test-slow-release: test-slow

# test-unit: runs lib and doc tests.
test-unit: _ensure-nextest
    cargo nextest run --profile ci --lib
    cargo test --doc --verbose

# Check TOML files parse cleanly.
toml-check: _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '*.toml')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 -I {} uv run python -c "import sys, tomllib; exec(\"with open(sys.argv[1], 'rb') as f:\\n    tomllib.load(f)\"); print(f'{sys.argv[1]} is valid TOML')" {}
    else
        echo "No TOML files found to check."
    fi

toml-fix: toml-fmt

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

# Check for unused direct Cargo dependencies.
unused-deps: _ensure-cargo-machete
    cargo machete

verify-expect-counts:
    #!/usr/bin/env bash
    set -euo pipefail

    check_count() {
        local label="$1"
        local expected="$2"
        local pattern="$3"
        shift 3

        local actual
        actual="$( (rg -o "$pattern" "$@" || true) | wc -l | tr -d ' ')"

        if [[ "$actual" != "$expected" ]]; then
            echo "❌ $label: expected $expected, found $actual"
            return 1
        fi

        echo "✓ $label: $actual"
    }

    check_count 'src/**/*.rs doc-comment .expect(' 0 '^\s*//[/!].*\.expect\(' src

yaml-check: yaml-fmt-check yaml-lint

yaml-fix: _ensure-dprint
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '*.yml' '*.yaml' 'CITATION.cff')
    if [ "${#files[@]}" -gt 0 ]; then
        echo "📝 dprint fmt (YAML/CFF, ${#files[@]} files)"
        dprint fmt --incremental=false "${files[@]}"
    else
        echo "No YAML files found to format."
    fi

yaml-fmt-check: _ensure-dprint
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '*.yml' '*.yaml' 'CITATION.cff')
    if [ "${#files[@]}" -gt 0 ]; then
        echo "🔍 dprint check (YAML/CFF, ${#files[@]} files)"
        dprint check --incremental=false "${files[@]}"
    else
        echo "No YAML files found to check."
    fi

yaml-lint: _ensure-yamllint
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '*.yml' '*.yaml' 'CITATION.cff')
    if [ "${#files[@]}" -gt 0 ]; then
        echo "🔍 yamllint (${#files[@]} YAML/CFF files)"
        uv run yamllint --strict -c .yamllint "${files[@]}"
    else
        echo "No YAML files found to lint."
    fi

zizmor: _ensure-zizmor
    zizmor .github
