# shellcheck disable=SC2148
# Justfile for delaunay development workflow
# Install just: https://github.com/casey/just
# Usage: just <command> or just --list

# Use bash with strict error handling for all recipes
set shell := ["bash", "-euo", "pipefail", "-c"]

cargo_llvm_cov_version := "0.8.5"

# Common cargo-llvm-cov arguments for all coverage runs.
# Excludes benches/examples from reports while allowing integration tests to
# exercise library code.
_coverage_base_args := '''--ignore-filename-regex '(^|/)(benches|examples)/' \
  --workspace --lib --tests \
  --verbose'''

_ensure-actionlint:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v actionlint >/dev/null || { echo "❌ 'actionlint' not found. See 'just setup' or https://github.com/rhysd/actionlint"; exit 1; }

_ensure-cargo-llvm-cov:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! command -v cargo-llvm-cov >/dev/null; then
        echo "❌ 'cargo-llvm-cov' not found. See 'just setup-tools' or install:"
        echo "   cargo install --locked cargo-llvm-cov --version {{cargo_llvm_cov_version}}"
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

# Internal helpers: ensure external tooling is installed
_ensure-git-cliff:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v git-cliff >/dev/null || {
        echo "❌ 'git-cliff' not found. Install via Homebrew: brew install git-cliff"
        echo "   Or via Cargo: cargo install git-cliff"
        exit 1
    }

_ensure-jq:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v jq >/dev/null || { echo "❌ 'jq' not found. See 'just setup' or install: brew install jq"; exit 1; }

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

# Benchmark recipes that produce performance numbers use Cargo's perf profile.
bench:
    cargo bench --workspace --profile perf

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
bench-test-compile:
    cargo bench --workspace --no-run
    cargo test --tests --release --no-run

# Build commands
build:
    cargo build

build-release:
    cargo build --release

# Changelog management (git-cliff + post-processing + archiving)
changelog: _ensure-git-cliff python-sync
    #!/usr/bin/env bash
    set -euo pipefail
    GIT_CLIFF_OFFLINE=true git-cliff -o CHANGELOG.md
    uv run postprocess-changelog
    uv run archive-changelog

changelog-tag version:
    just tag {{version}}

changelog-unreleased version: _ensure-git-cliff python-sync
    #!/usr/bin/env bash
    set -euo pipefail
    GIT_CLIFF_OFFLINE=true git-cliff --tag {{version}} -o CHANGELOG.md
    uv run postprocess-changelog
    uv run archive-changelog

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

# CI simulation: comprehensive validation (matches .github/workflows/ci.yml)
# Runs: checks + test workflow + examples
ci: check test examples
    @echo "🎯 CI checks complete!"

# CI followed by an explicit persistent local baseline refresh.
ci-baseline ref="main":
    just ci
    just perf-baseline {{ref}}

# CI + slow/stress tests (100+ vertices, stress tests)
ci-slow: ci test-slow
    @echo "✅ CI + slow tests passed!"

# Clean build artifacts
clean:
    cargo clean
    rm -rf target/llvm-cov
    rm -rf coverage_report
    rm -rf coverage

# Code quality and formatting
clippy:
    cargo clippy --workspace --all-targets -- -D warnings -W clippy::pedantic -W clippy::nursery -W clippy::cargo

    # All features
    cargo clippy --workspace --all-targets --all-features -- -D warnings -W clippy::pedantic -W clippy::nursery -W clippy::cargo

# Coverage analysis for local development (HTML output)
coverage: _ensure-cargo-llvm-cov
    mkdir -p target/llvm-cov
    cargo llvm-cov {{_coverage_base_args}} --html --output-dir target/llvm-cov
    @echo "📊 Coverage report generated: target/llvm-cov/html/index.html"

# Coverage analysis for CI (XML output for codecov/codacy)
coverage-ci: _ensure-cargo-llvm-cov
    mkdir -p coverage
    cargo llvm-cov {{_coverage_base_args}} --cobertura --output-path coverage/cobertura.xml -- --skip prop_

debug-large-scale-2d n="36000" repair_every="1":
    DELAUNAY_BULK_PROGRESS_EVERY=2000 DELAUNAY_LARGE_DEBUG_MAX_RUNTIME_SECS=1800 DELAUNAY_LARGE_DEBUG_N_2D={{n}} DELAUNAY_LARGE_DEBUG_REPAIR_EVERY={{repair_every}} cargo test --release --test large_scale_debug debug_large_scale_2d -- --ignored --exact --nocapture

debug-large-scale-3d n="8000" repair_every="1":
    DELAUNAY_BULK_PROGRESS_EVERY=500 DELAUNAY_LARGE_DEBUG_MAX_RUNTIME_SECS=1800 DELAUNAY_LARGE_DEBUG_N_3D={{n}} DELAUNAY_LARGE_DEBUG_REPAIR_EVERY={{repair_every}} cargo test --release --test large_scale_debug debug_large_scale_3d -- --ignored --exact --nocapture

debug-large-scale-4d n="900" repair_every="1":
    DELAUNAY_BULK_PROGRESS_EVERY=100 DELAUNAY_LARGE_DEBUG_MAX_RUNTIME_SECS=1800 DELAUNAY_LARGE_DEBUG_N_4D={{n}} DELAUNAY_LARGE_DEBUG_REPAIR_EVERY={{repair_every}} cargo test --release --test large_scale_debug debug_large_scale_4d -- --ignored --exact --nocapture

debug-large-scale-5d n="140" repair_every="1":
    DELAUNAY_BULK_PROGRESS_EVERY=20 DELAUNAY_LARGE_DEBUG_MAX_RUNTIME_SECS=1800 DELAUNAY_LARGE_DEBUG_N_5D={{n}} DELAUNAY_LARGE_DEBUG_REPAIR_EVERY={{repair_every}} cargo test --release --test large_scale_debug debug_large_scale_5d -- --ignored --exact --nocapture

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
    @echo "  just test-slow         # Run slow/stress tests with --features slow-tests"
    @echo "  just examples          # Run all examples"
    @echo ""
    @echo "Active large-scale debugging:"
    @echo "  just test-diagnostics      # Run diagnostics tools with output"
    @echo "  just debug-large-scale-2d [n] [repair_every] # 2D acceptance/profiling (defaults n=36000, repair_every=1)"
    @echo "  just debug-large-scale-3d [n] [repair_every] # Issue #341: 3D scalability (defaults n=8000, repair_every=1)"
    @echo "  just debug-large-scale-4d [n] [repair_every] # Issue #340: 4D large-scale runtime (defaults n=900, repair_every=1)"
    @echo "  just debug-large-scale-5d [n] [repair_every] # Issue #342: 5D feasibility (defaults n=140, repair_every=1)"
    @echo ""
    @echo "Benchmark workflows:"
    @echo "  just bench-smoke        # Smoke-test benchmark harnesses (minimal samples)"
    @echo "  just bench              # Run all benchmarks with perf profile (ThinLTO)"
    @echo "  just bench-ci           # CI regression benchmarks with perf profile (~5-10 min)"
    @echo "  just perf-no-regressions [threshold] # Fast pre-PR 2D-5D regression guard (default 7.5%)"
    @echo "  just perf-baseline [ref] # Persist/update default local baseline (default: main)"
    @echo "  just perf-baseline-to <out> [ref] # Generate scratch baseline without replacing default"
    @echo "  just bench-perf-summary # Generate perf-profile release summary (~30-45 min)"
    @echo "  just profile [toolchain] [code_ref] # Run ci_performance_suite for a compiler/code pair"
    @echo ""
    @echo "Larger/optional workflows:"
    @echo "  just ci-slow             # CI + slow tests (100+ vertices)"
    @echo "  just ci-baseline         # CI + persist default performance baseline"
    @echo "  just coverage            # Generate coverage report (HTML)"
    @echo "  just semgrep             # Run repository-owned Semgrep rules"
    @echo ""
    @echo "Use 'just --list' for every granular recipe."

# All linting: code + documentation + configuration
lint: lint-code lint-docs lint-config

# Code linting: Rust (fmt-check, clippy, docs, Semgrep) + Python (Ruff, Ty) + Shell scripts
lint-code: fmt-check clippy doc-check semgrep semgrep-test python-lint shell-lint

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

# Generate a same-machine dev-mode baseline for a GitHub ref.
perf-baseline ref="main": _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    uv run benchmark-utils generate-ref-baseline --ref "{{ref}}" --out baseline-artifact --dev

perf-baseline-to out ref="main": _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    uv run benchmark-utils generate-ref-baseline --ref "{{ref}}" --out "{{out}}" --dev

perf-compare file threshold="7.5": _ensure-uv
    uv run benchmark-utils compare --baseline "{{file}}" --threshold {{threshold}} --dev

perf-help:
    @echo "Performance Analysis Commands:"
    @echo "  just perf-no-regressions   # Fast pre-PR guard with a temporary same-machine main baseline"
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
    @echo "  just perf-no-regressions   # Generate temporary main baseline, compare current tree, clean up"
    @echo "  just perf-baseline [ref]   # Persist baseline-artifact/baseline_results.txt from a GitHub ref"
    @echo "  just perf-baseline-to <out> [ref] # Generate an alternate local baseline artifact directory"
    @echo "  just perf-compare <file>   # Compare against a specific dev-mode baseline"
    @echo "  just bench                 # Full benchmark suite with perf profile"
    @echo "  just bench-ci              # CI benchmark suite with perf profile"
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
    @echo "  just perf-no-regressions   # Recommended local PR performance guard"
    @echo "  just perf-baseline         # Persist/update default local baseline for GitHub main"
    @echo "  just perf-baseline v0.7.5  # Persist/update default local baseline for a release tag"
    @echo "  just perf-baseline-to /tmp/delaunay-main-baseline"
    @echo "                              # Generate scratch main baseline without overwriting baseline-artifact"
    @echo "  CRIT_SAMPLE_SIZE=100 just bench  # Custom sample size"
    @echo "  just bench-ci              # Final optimized CI-suite benchmark run"
    @echo "  just profile v0.7.5        # v0.7.5 code on its declared Rust toolchain"
    @echo "  just profile 1.95          # Current tree on Rust 1.95"
    @echo "  just profile 1.95 v0.7.5   # v0.7.5 code on Rust 1.95"

# Fast pre-PR performance guard against a temporary same-machine main baseline.
perf-no-regressions threshold="7.5": _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail

    relevant_worktree_dirty() {
        if ! git diff --quiet -- src benches Cargo.toml Cargo.lock scripts/benchmark_utils.py; then
            return 0
        fi
        if ! git diff --cached --quiet -- src benches Cargo.toml Cargo.lock scripts/benchmark_utils.py; then
            return 0
        fi
        if [ -n "$(git ls-files --others --exclude-standard -- src benches Cargo.toml Cargo.lock scripts/benchmark_utils.py)" ]; then
            return 0
        fi
        return 1
    }

    current_commit="$(git rev-parse HEAD)"
    remote_line="$(git ls-remote origin refs/heads/main || true)"
    remote_main_commit=""
    if [ -n "$remote_line" ]; then
        read -r remote_main_commit _ <<< "$remote_line"
    fi
    if [ -n "$remote_main_commit" ] && [ "$remote_main_commit" = "$current_commit" ] && ! relevant_worktree_dirty; then
        echo "🔍 origin/main matches HEAD (${current_commit}); no relevant worktree changes to compare."
        echo "   Skipping perf-no-regressions before generating a same-commit baseline."
        exit 0
    fi

    tmp="$(mktemp -d "${TMPDIR:-/tmp}/delaunay-perf-baseline.XXXXXX")"
    trap 'rm -rf "$tmp"' EXIT
    uv run benchmark-utils generate-ref-baseline --ref main --out "$tmp/baseline" --dev
    baseline="$tmp/baseline/baseline_results.txt"
    if ! grep -q 'Benchmark ID: tds_new_2d/tds_new/2000' "$baseline"; then
        echo "❌ Temporary baseline for main does not match the current ci_performance_suite contract."
        echo "   The benchmark contract probably changed on this branch; inspect ci_performance_suite before comparing."
        exit 1
    fi
    baseline_line="$(grep -m1 '^Git commit:' "$baseline" || true)"
    baseline_commit="${baseline_line#Git commit: }"
    if [ -n "$baseline_commit" ] && [ "$baseline_commit" = "$current_commit" ]; then
        if ! relevant_worktree_dirty; then
            echo "🔍 Current commit matches the main baseline (${baseline_commit}); no relevant worktree changes to compare."
            echo "   Skipping perf-no-regressions because a same-commit baseline would mask regressions."
            exit 0
        fi
        echo "⚠️ Main baseline commit matches HEAD, but relevant uncommitted changes exist; comparing the worktree against HEAD."
    fi
    uv run benchmark-utils compare --baseline "$baseline" --threshold {{threshold}} --dev

# Run the selected CI benchmark suite for one compiler/code pair.
profile toolchain="" code_ref="current":
    #!/usr/bin/env bash
    set -euo pipefail

    command -v rustup >/dev/null || { echo "❌ 'rustup' not found. Install Rust via https://rustup.rs"; exit 1; }

    repo_root="$(pwd)"
    requested_toolchain="{{toolchain}}"
    requested_ref="{{code_ref}}"
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
    cleanup() {
        find "$config_dir" -type l -exec unlink {} \;
        find "$config_dir" -depth -type d -exec rmdir {} +
    }
    trap cleanup EXIT

    # Semgrep directory test mode maps fixture paths to config paths, so mirror
    # each fixture to the shared config while keeping semgrep.yaml authoritative.
    while IFS= read -r -d '' fixture; do
        rel="${fixture#tests/semgrep/}"
        config_path="$config_dir/${rel%.*}.yaml"
        mkdir -p "$(dirname "$config_path")"
        ln -s "$PWD/semgrep.yaml" "$config_path"
    done < <(find tests/semgrep -type f ! -name '*.fixed' -print0)

    uv run semgrep scan --test --strict --config "$config_dir" tests/semgrep

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

    if ! rustup component list --installed | grep -q '^llvm-tools'; then
        echo "  ⏳ Installing llvm-tools-preview (rustup)..."
        rustup component add llvm-tools-preview
    else
        echo "  ✓ llvm-tools-preview"
    fi

    if ! have cargo-llvm-cov; then
        echo "  ⏳ Installing cargo-llvm-cov {{cargo_llvm_cov_version}} (cargo)..."
        cargo install --locked cargo-llvm-cov --version {{cargo_llvm_cov_version}}
    else
        echo "  ✓ cargo-llvm-cov"
    fi

    echo ""
    echo "Verifying required commands are available..."
    missing=0

    cmds=(uv jq taplo yamllint shfmt shellcheck actionlint git-cliff node npx typos)
    cmds+=(cargo-llvm-cov)

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

# Create an annotated git tag from the CHANGELOG.md section for the given version
tag version: python-sync
    uv run tag-release {{version}}

# Replace an existing annotated tag from the CHANGELOG.md section.
tag-force version: python-sync
    uv run tag-release {{version}} --force

# Testing
# test: runs default-profile benchmark/release compile checks plus all tests.
test: bench-test-compile test-all
    @echo "✅ Test workflow passed!"

# test-all: runs lib, doc, integration, and Python tests (comprehensive)
test-all: test-unit test-integration test-python
    @echo "✅ All tests passed!"

test-allocation:
    cargo test --test allocation_api --features count-allocations -- --nocapture

test-diagnostics:
    cargo test --test circumsphere_debug_tools --features diagnostics -- --nocapture

# test-integration: runs all integration tests (includes proptests) in release mode.
# Release mode is required because exact-predicate arithmetic in debug mode makes
# 3D+ proptests exceed CI timeout limits (>60s debug vs <1s release).
test-integration:
    cargo test --tests --release --verbose

# test-integration-fast: runs integration tests but skips proptests (tests prefixed with `prop_`)
#
# Useful for quick local validation on changes that don't touch the property-test surface area.
# To run the full (slow) property suite, use: just test-integration
#
# Note: `--skip prop_` is a substring filter applied by the Rust test harness.
test-integration-fast:
    cargo test --tests --release --verbose -- --skip prop_

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

# test-unit: runs lib and doc tests.
test-unit:
    cargo test --lib --verbose
    cargo test --doc --verbose

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

    check_count 'src/**/*.rs doc-comment .expect(' 17 '^\s*//[/!].*\.expect\(' src

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
