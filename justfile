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
dprint_version := "0.55.1"
just_version := "1.55.1"
nextest_version := "0.9.140"
rumdl_version := "0.2.28"
taplo_version := "0.10.0"
tectonic_version := "0.16.9"
tex_fmt_version := "0.5.7"
typos_version := "1.48.0"
zizmor_version := "1.26.1"

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

_ensure-chktex:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v chktex >/dev/null || {
        echo "❌ 'chktex' not found. Install a TeX distribution or package manager copy of chktex and ensure it is on PATH."
        exit 1
    }

_ensure-tectonic:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if command -v tectonic >/dev/null; then
        installed_version="$(tectonic --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
    fi
    if [[ "$installed_version" != "{{ tectonic_version }}" ]]; then
        echo "❌ 'tectonic' {{ tectonic_version }} not found. See 'just setup-tools' or install:"
        echo "   cargo install --locked tectonic --version {{ tectonic_version }}"
        exit 1
    fi

_ensure-tex-fmt:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if command -v tex-fmt >/dev/null; then
        installed_version="$(tex-fmt --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
    fi
    if [[ "$installed_version" != "{{ tex_fmt_version }}" ]]; then
        echo "❌ 'tex-fmt' {{ tex_fmt_version }} not found. See 'just setup-tools' or install:"
        echo "   cargo install --locked tex-fmt --version {{ tex_fmt_version }}"
        exit 1
    fi

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
    cargo bench --workspace --profile perf --features bench

# Allocation-contract microbenchmarks for public hot paths.
bench-allocations:
    cargo bench --profile perf --bench allocation_hot_paths --features count-allocations -- --noplot

# CI regression benchmarks with the perf profile.
bench-ci:
    cargo bench --profile perf --bench ci_performance_suite

# Compile benchmark harnesses without running them.
bench-compile:
    cargo bench --workspace --no-run --features bench

# Generate performance summary with fresh perf-profile benchmark runs (for releases)
bench-perf-summary: _ensure-uv
    uv run benchmark-utils generate-summary --run-benchmarks --profile perf

# Smoke-test benchmark harnesses with minimal samples; not for performance data.
# Criterion requires sample_size >= 10; use the minimum with short measurement/warm-up windows.
bench-smoke:
    CRIT_SAMPLE_SIZE=10 CRIT_MEASUREMENT_MS=500 CRIT_WARMUP_MS=200 cargo bench --workspace --profile perf --features bench

# Run the opt-in companion binary with the CLI feature and perf profile.
run *args:
    cargo run --profile perf --features cli --bin delaunay -- {{ args }}

# Run one 3D and one 4D Pachner Monte Carlo stress chain with reports enabled.
pachner-stress attempts="100000" validate_every="1000":
    just _pachner-stress-dim 3d 10000 "{{ attempts }}" "{{ validate_every }}" target/pachner_stress/3d
    just _pachner-stress-dim 4d 1000 "{{ attempts }}" "{{ validate_every }}" target/pachner_stress/4d

# Run one 3D Pachner Monte Carlo stress chain with reports enabled.
pachner-stress-3d attempts="100000" vertices="10000" validate_every="1000" output_dir="target/pachner_stress/3d":
    just _pachner-stress-dim 3d "{{ vertices }}" "{{ attempts }}" "{{ validate_every }}" "{{ output_dir }}"

# Run one 4D Pachner Monte Carlo stress chain with reports enabled.
pachner-stress-4d attempts="100000" vertices="1000" validate_every="1000" output_dir="target/pachner_stress/4d":
    just _pachner-stress-dim 4d "{{ vertices }}" "{{ attempts }}" "{{ validate_every }}" "{{ output_dir }}"

# Run Criterion's Pachner Monte Carlo stress benchmark for statistical timing.
bench-pachner-stress attempts="100000" validate_every="1000" samples="10":
    just _bench-pachner-stress-dim 3d 10000 "{{ attempts }}" "{{ validate_every }}" "{{ samples }}"
    just _bench-pachner-stress-dim 4d 1000 "{{ attempts }}" "{{ validate_every }}" "{{ samples }}"

# Run Criterion's 3D Pachner Monte Carlo stress benchmark for statistical timing.
bench-pachner-stress-3d attempts="100000" vertices="10000" validate_every="1000" samples="10":
    just _bench-pachner-stress-dim 3d "{{ vertices }}" "{{ attempts }}" "{{ validate_every }}" "{{ samples }}"

# Run Criterion's 4D Pachner Monte Carlo stress benchmark for statistical timing.
bench-pachner-stress-4d attempts="100000" vertices="1000" validate_every="1000" samples="10":
    just _bench-pachner-stress-dim 4d "{{ vertices }}" "{{ attempts }}" "{{ validate_every }}" "{{ samples }}"

[private]
_pachner-stress-dim label vertices attempts validate_every output_dir:
    #!/usr/bin/env bash
    set -euo pipefail

    label="{{ label }}"
    vertices="{{ vertices }}"
    attempts="{{ attempts }}"
    validate_every="{{ validate_every }}"
    output_dir="{{ output_dir }}"

    require_positive_integer() {
        local name="$1"
        local value="$2"
        if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: $name must be a positive integer, got: $value" >&2
            exit 2
        fi
    }

    require_positive_integer "vertices" "$vertices"
    require_positive_integer "attempts" "$attempts"
    require_positive_integer "validate_every" "$validate_every"

    mkdir -p "$output_dir"

    echo "Pachner stress ${label}: ${vertices} vertices, ${attempts} attempted moves."
    cargo run --profile perf --features cli --bin delaunay -- \
        pachner-stress \
        --dimension "$label" \
        --vertices "$vertices" \
        --attempts "$attempts" \
        --validate-every "$validate_every" \
        --progress-csv "$output_dir/progress.csv" \
        --summary-json "$output_dir/summary.json"

[private]
_bench-pachner-stress-dim label vertices attempts validate_every samples:
    #!/usr/bin/env bash
    set -euo pipefail

    label="{{ label }}"
    vertices="{{ vertices }}"
    attempts="{{ attempts }}"
    validate_every="{{ validate_every }}"
    samples="{{ samples }}"

    require_positive_integer() {
        local name="$1"
        local value="$2"
        if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: $name must be a positive integer, got: $value" >&2
            exit 2
        fi
    }

    require_positive_integer "vertices" "$vertices"
    require_positive_integer "attempts" "$attempts"
    require_positive_integer "validate_every" "$validate_every"
    require_positive_integer "samples" "$samples"

    if (( samples < 10 )); then
        echo "ERROR: samples must be at least 10 because Criterion requires sample_size >= 10." >&2
        exit 2
    fi

    case "$label" in
        3d)
            suffix="3D"
            filter="monte_carlo/3d"
            ;;
        4d)
            suffix="4D"
            filter="monte_carlo/4d"
            ;;
        *)
            echo "ERROR: unsupported Pachner stress dimension: $label" >&2
            exit 2
            ;;
    esac

    echo "Pachner stress ${suffix}: ${vertices} vertices, ${attempts} attempted moves per Criterion sample."
    env \
        DELAUNAY_PACHNER_STRESS_REPORT=1 \
        "DELAUNAY_PACHNER_STRESS_VERTICES_${suffix}=$vertices" \
        "DELAUNAY_PACHNER_STRESS_ATTEMPTS_${suffix}=$attempts" \
        "DELAUNAY_PACHNER_STRESS_VALIDATE_EVERY_${suffix}=$validate_every" \
        "MONTE_CARLO_SAMPLE_SIZE=$samples" \
        cargo bench --profile perf --features bench --bench pachner_stress -- "$filter" --noplot

# Compile benchmarks and release integration tests without running.
bench-test-compile: bench-compile test-integration-compile

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

# Check (non-mutating): run non-test validators.
check: lint
    @echo "✅ Checks complete!"

# Fast compile check (no binary produced)
check-fast:
    cargo check

# CI simulation: comprehensive validation.
ci: github-actions-check markdown-ci json-check toml-ci yaml-ci python-ci notebook-clear-outputs-all notebook-check rust-core-check test-rust-ci test-doc bench-compile examples
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
clippy: clippy-all-targets

clippy-all-targets:
    cargo clippy --workspace --all-targets -- -D warnings -W clippy::pedantic -W clippy::nursery -W clippy::cargo
    cargo clippy --workspace --all-targets --all-features -- -D warnings -W clippy::pedantic -W clippy::nursery -W clippy::cargo

# Coverage analysis for local development (HTML output)
coverage: _ensure-cargo-llvm-cov
    mkdir -p target/llvm-cov
    cargo llvm-cov {{ _coverage_base_args }} --html --output-dir target/llvm-cov
    @echo "📊 Coverage report generated: target/llvm-cov/html/index.html"

# Coverage analysis for CI (XML output for codecov/codacy)
coverage-ci: _ensure-cargo-llvm-cov
    mkdir -p coverage
    cargo llvm-cov nextest {{ _coverage_base_args }} --cobertura --output-path coverage/cobertura.xml -P coverage

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

github-actions-check: action-lint zizmor
    @echo "✅ GitHub Actions checks complete!"

help-workflows:
    @echo "Recommended Just workflow:"
    @echo "  just check             # Run all non-mutating lints/validators"
    @echo "  just fix               # Apply formatters/auto-fixes (mutating)"
    @echo "  just test              # Run default test buckets"
    @echo "  just ci                # GitHub-equivalent union of every validation bucket"
    @echo ""
    @echo "Focused validation:"
    @echo "  just rust-core-check   # Formatting, all-targets Clippy, docs, and Semgrep"
    @echo "  just python-ci         # Python lint/typecheck + pytest"
    @echo "  just notebook-check    # Notebook hygiene + fast headless execution"
    @echo "  just markdown-ci       # Markdown lint + spell check"
    @echo "  just toml-ci           # TOML parse/lint/format checks"
    @echo "  just yaml-ci           # YAML/CFF format/lint/citation checks"
    @echo "  just github-actions-check # actionlint + zizmor"
    @echo ""
    @echo "Focused testing:"
    @echo "  just test-rust         # Rust unit, doctest, and integration tests"
    @echo "  just test-rust-ci      # CI Rust unit, integration, and CLI tests"
    @echo "  just test-unit         # Rust lib unit tests only"
    @echo "  just test-doc          # Rust doctests only, in release profile"
    @echo "  just test-integration  # All integration tests (includes proptests)"
    @echo "  just test-integration-fast # Integration tests (skips proptests)"
    @echo "  just test-integration-compile # Compile integration tests without running"
    @echo "  just test-python       # Python tests only (pytest)"
    @echo "  just test-slow         # Run correctness tests over the 10s default-suite budget"
    @echo "  just examples          # Run all examples"
    @echo ""
    @echo "Notebook workflows:"
    @echo "  just notebook          # Launch the default notebook with uv-managed dependencies"
    @echo "  just notebook-lint     # Validate notebook JSON, output hygiene, and extracted code"
    @echo "  just notebook-check    # Lint notebooks and execute fast notebooks under target/notebooks"
    @echo "  just notebook-check-slow # Include slow notebook execution"
    @echo "  just notebook-clear-outputs-all # Clear source notebook outputs"
    @echo "  just notebook-reset-from-git # Restore tracked source notebooks and clear artifacts"
    @echo "  just run <args>        # Run the opt-in delaunay binary with --features cli"
    @echo ""
    @echo "Active large-scale debugging:"
    @echo "  just test-diagnostics      # Run diagnostics tools with output"
    @echo "  just debug-large-scale-2d [n] [repair_every] # 2D acceptance/profiling (defaults n=36000, repair_every=1)"
    @echo "  just debug-large-scale-3d [n] [repair_every] # Issue #341: 3D scalability (defaults n=7500, repair_every=1)"
    @echo "  just debug-large-scale-4d [n] [repair_every] # Issue #340: 4D large-scale runtime (defaults n=800, repair_every=1)"
    @echo "  just debug-large-scale-5d [n] [repair_every] # Issue #342: 5D feasibility (defaults n=140, repair_every=1)"
    @echo ""
    @echo "Benchmark workflows:"
    @echo "  just bench-compile      # Compile benchmark harnesses without running"
    @echo "  just bench-smoke        # Smoke-test benchmark harnesses (minimal samples)"
    @echo "  just bench              # Run all benchmarks with perf profile (ThinLTO)"
    @echo "  just bench-ci           # CI regression benchmarks with perf profile (~5-10 min)"
    @echo "  just pachner-stress     # 3D+4D Pachner MCMC CLI stress with CSV/JSON artifacts"
    @echo "  just pachner-stress-3d [attempts] [vertices] [validate_every] [output_dir]"
    @echo "                          # 3D Pachner MCMC stress (defaults: 100K, 10K vertices)"
    @echo "  just pachner-stress-4d [attempts] [vertices] [validate_every] [output_dir]"
    @echo "                          # 4D Pachner MCMC stress (defaults: 100K, 1K vertices)"
    @echo "  just bench-pachner-stress # Criterion timing for Pachner MCMC stress"
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
lint: github-actions-check markdown-ci json-check toml-ci yaml-ci python-check notebook-lint rust-core-check shell-lint

# Code linting: Rust, Python, notebooks, and shell scripts.
lint-code: rust-core-check python-check notebook-lint shell-lint

# Configuration checks: JSON, TOML, YAML/CFF, GitHub Actions workflows
lint-config: json-check toml-ci yaml-ci github-actions-check

# Documentation linting: Markdown + spell checking
lint-docs: markdown-ci

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

markdown-ci: markdown-check spell-check
    @echo "✅ Markdown checks complete!"

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

notebook notebook="notebooks/00_quickstart.ipynb": _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    notebook_cache="$(pwd)/target/notebooks"
    mkdir -p "$notebook_cache/.ipython" "$notebook_cache/.matplotlib"
    MPLBACKEND=Agg IPYTHONDIR="$notebook_cache/.ipython" MPLCONFIGDIR="$notebook_cache/.matplotlib" uv run --group notebooks jupyter lab --ServerApp.open_browser=True --LabApp.open_browser=True "{{ notebook }}"

notebook-check: notebook-lint notebook-execute-fast
    @echo "📓 Notebook checks complete!"

notebook-check-slow: notebook-check notebook-execute-slow
    @echo "📓 Slow notebook checks complete!"

notebook-clear-outputs notebook="notebooks/00_quickstart.ipynb": _ensure-uv
    uv run --group notebooks jupyter nbconvert --clear-output --inplace "{{ notebook }}"

notebook-clear-outputs-all: _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -d notebooks ]; then
        echo "No notebooks found to clear."
        exit 0
    fi
    found=0
    while IFS= read -r notebook; do
        found=1
        uv run --group notebooks jupyter nbconvert --clear-output --inplace "$notebook"
    done < <(find notebooks -type f -name '*.ipynb' ! -path '*/.ipynb_checkpoints/*' | sort)
    if [ "$found" -eq 0 ]; then
        echo "No notebooks found to clear."
    fi

notebook-reset-from-git source="index":
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -d notebooks ]; then
        echo "No source notebooks directory found."
        exit 0
    fi

    tracked_notebooks=()
    while IFS= read -r notebook; do
        tracked_notebooks+=("$notebook")
    done < <(git ls-files -- notebooks | grep '\.ipynb$' || true)

    tracked_count="${#tracked_notebooks[@]}"
    if [ "$tracked_count" -eq 0 ]; then
        echo "No tracked source notebooks found."
        exit 0
    fi

    if [ "{{ source }}" = "index" ]; then
        git restore --worktree -- "${tracked_notebooks[@]}"
        restored_from="index"
    else
        git restore --source="{{ source }}" --worktree -- "${tracked_notebooks[@]}"
        restored_from="{{ source }}"
    fi

    rm -rf target/notebooks
    find notebooks -type d -name .ipynb_checkpoints -prune -exec rm -rf {} +
    printf 'Restored %s tracked source notebook(s) from %s and removed target/notebooks.\n' "$tracked_count" "$restored_from"

notebook-execute notebook="notebooks/00_quickstart.ipynb" output_dir="target/notebooks": _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    output_path="$(pwd)/{{ output_dir }}"
    notebook_stem="$(basename "{{ notebook }}" .ipynb)"
    notebook_output_dir="$output_path/$notebook_stem"
    mkdir -p "$output_path/.ipython" "$output_path/.matplotlib" "$notebook_output_dir"
    MPLBACKEND=Agg IPYTHONDIR="$output_path/.ipython" MPLCONFIGDIR="$output_path/.matplotlib" uv run --group notebooks jupyter nbconvert --execute --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.shutdown_kernel=immediate --to notebook --output-dir "$notebook_output_dir" "{{ notebook }}"

notebook-execute-fast output_dir="target/notebooks": _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -d notebooks ]; then
        echo "No fast notebooks found to execute."
        exit 0
    fi
    output_path="$(pwd)/{{ output_dir }}"
    mkdir -p "$output_path/.ipython" "$output_path/.matplotlib"
    found=0
    while IFS= read -r notebook; do
        found=1
        notebook_stem="$(basename "$notebook" .ipynb)"
        notebook_output_dir="$output_path/$notebook_stem"
        mkdir -p "$notebook_output_dir"
        MPLBACKEND=Agg IPYTHONDIR="$output_path/.ipython" MPLCONFIGDIR="$output_path/.matplotlib" uv run --group notebooks jupyter nbconvert --execute --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.shutdown_kernel=immediate --to notebook --output-dir "$notebook_output_dir" "$notebook"
    done < <(find notebooks -type f -name '*.ipynb' ! -path '*/.ipynb_checkpoints/*' ! -path 'notebooks/slow/*' ! -name '*_slow.ipynb' | sort)
    if [ "$found" -eq 0 ]; then
        echo "No fast notebooks found to execute."
    fi

notebook-execute-slow output_dir="target/notebooks": _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -d notebooks ]; then
        echo "No slow notebooks found to execute."
        exit 0
    fi
    output_path="$(pwd)/{{ output_dir }}"
    mkdir -p "$output_path/.ipython" "$output_path/.matplotlib"
    found=0
    while IFS= read -r notebook; do
        found=1
        notebook_stem="$(basename "$notebook" .ipynb)"
        notebook_output_dir="$output_path/$notebook_stem"
        mkdir -p "$notebook_output_dir"
        MPLBACKEND=Agg IPYTHONDIR="$output_path/.ipython" MPLCONFIGDIR="$output_path/.matplotlib" uv run --group notebooks jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.shutdown_kernel=immediate --to notebook --output-dir "$notebook_output_dir" "$notebook"
    done < <(find notebooks -type f \( -path 'notebooks/slow/*' -o -name '*_slow.ipynb' \) ! -path '*/.ipynb_checkpoints/*' | sort)
    if [ "$found" -eq 0 ]; then
        echo "No slow notebooks found to execute."
    fi

notebook-lint: _ensure-uv
    uv run --group dev --group notebooks notebook-check lint --repo-root .

notebook-output-check: _ensure-uv
    uv run --group dev --group notebooks notebook-check lint --repo-root . --no-ruff --no-format --no-ty

notebook-setup: _ensure-uv
    uv sync --group notebooks

# Refresh tracked paper figures from reproducible notebooks.
paper-figures: _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p papers/generated
    DELAUNAY_VALIDATION_PAPER_FIGURE_DIR="papers/generated" just notebook-execute notebooks/01_validation.ipynb target/papers/notebooks

# Format publication-facing TeX sources.
paper-tex-fmt: _ensure-tex-fmt
    tex-fmt papers/*.tex

# Check publication-facing TeX formatting and lint diagnostics.
paper-tex-lint: _ensure-chktex _ensure-tex-fmt
    #!/usr/bin/env bash
    set -euo pipefail
    tex-fmt --check papers/*.tex
    # 24 conflicts with tex-fmt's indented figure labels.
    chktex -q -n 1 -n 8 -n 24 -n 46 papers/*.tex

# Compile one paper with Tectonic and copy the reviewer PDF beside its TeX source.
paper-build paper="validation": _ensure-tectonic _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    paper={{ quote(paper) }}
    case "$paper" in
        ""|*[!A-Za-z0-9_-]*)
            echo "❌ Invalid paper name: $paper"
            echo "   Use only ASCII letters, digits, underscores, and hyphens."
            exit 1
            ;;
    esac
    paper_source="papers/${paper}.tex"
    paper_pdf="papers/${paper}.pdf"
    build_dir="target/papers/${paper}"
    if [ ! -f "$paper_source" ]; then
        echo "❌ Paper source not found: $paper_source"
        exit 1
    fi
    rm -rf "$build_dir"
    mkdir -p "$build_dir"
    source_date_epoch="$(uv run paper-source-date-epoch "$paper_source")"
    export SOURCE_DATE_EPOCH="$source_date_epoch"
    tectonic --keep-intermediates --keep-logs --outdir "$build_dir" "$paper_source"
    uv run paper-pdf-normalize "$build_dir/${paper}.pdf" --tex "$paper_source"
    cp "$build_dir/${paper}.pdf" "$paper_pdf"
    echo "📄 Paper PDF written: $paper_pdf"

# Check the compiled reviewer PDF for basic readability.
paper-pdf-check paper="validation": _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    paper={{ quote(paper) }}
    case "$paper" in
        ""|*[!A-Za-z0-9_-]*)
            echo "❌ Invalid paper name: $paper"
            echo "   Use only ASCII letters, digits, underscores, and hyphens."
            exit 1
            ;;
    esac
    uv run paper-pdf-check "papers/${paper}.pdf" \
        --min-pages 1 \
        --require-text "Validation Architecture in delaunay" \
        --require-text "REFERENCES" \
        --forbid-text "\\today" \
        --forbid-text "Manuscript submitted to ACM"

paper-check paper="validation": paper-tex-lint
    #!/usr/bin/env bash
    set -euo pipefail
    paper={{ quote(paper) }}
    just paper-build "$paper"
    just paper-pdf-check "$paper"
    echo "✅ Paper '${paper}' compiled and checked successfully."

# Refresh notebook-owned paper figures, lint TeX, compile, and sanity-check PDFs.
papers: paper-figures paper-check
    @echo "📚 Paper workflow complete!"

paper-clean:
    rm -rf target/papers

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
    @echo "  just pachner-stress        # 3D+4D Pachner MCMC CLI stress with CSV/JSON artifacts"
    @echo "  just pachner-stress-3d     # 3D Pachner MCMC CLI stress (100K moves, 10K vertices)"
    @echo "  just pachner-stress-4d     # 4D Pachner MCMC CLI stress (100K moves, 1K vertices)"
    @echo "  just bench-pachner-stress  # Criterion timing for Pachner MCMC stress"
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
    @echo "  just pachner-stress-4d 100000 1000 1000 target/pachner_stress/4d"
    @echo "                              # 4D long-run Pachner diagnostics with CSV/JSON artifacts"
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
    summaries=()

    run_case() {
        local dimension="$1"
        local test_name="$2"
        local n_env="$3"
        local n_points="$4"
        local progress_every="$5"
        local log_file
        log_file="$(mktemp "${TMPDIR:-/tmp}/delaunay-large-scale-${dimension}.XXXXXX")"

        echo ""
        echo "▶ ${dimension}: ${test_name} (${n_points} vertices, ${max_secs}s cap)"
        # Construction wall-clock guard: validate Levels 1-3 + Level 5 only.
        # Level 4 embedding overlap validation runs at scale under `just test-slow`
        # (full scope); see issue #482.
        if env \
            DELAUNAY_BULK_PROGRESS_EVERY="$progress_every" \
            DELAUNAY_LARGE_DEBUG_MAX_RUNTIME_SECS="$max_secs" \
            "$n_env=$n_points" \
            DELAUNAY_LARGE_DEBUG_REPAIR_EVERY=1 \
            DELAUNAY_LARGE_DEBUG_VALIDATION=construction \
            cargo nextest run --release --profile slow --features slow-tests --test large_scale_debug "$test_name" -- --exact --nocapture 2>&1 | tee "$log_file"; then
            echo "✅ ${dimension} completed within the ${max_secs}s test-runtime cap"
            case_status="PASS"
        else
            local code=$?
            echo "❌ ${dimension} failed or exceeded the ${max_secs}s test-runtime cap (exit ${code})"
            failures+=("$dimension")
            status=1
            case_status="FAIL"
        fi

        local insertion_time total_time simplices
        insertion_time="$(awk -F': ' '/Insertion wall time:/ { value=$2 } END { print value }' "$log_file")"
        total_time="$(awk -F': ' '/Total wall time:/ { value=$2 } END { print value }' "$log_file")"
        simplices="$(awk '/Triangulation size:/ { for (i = 1; i <= NF; i++) if ($i ~ /^simplices=/) { sub(/^simplices=/, "", $i); value=$i } } END { print value }' "$log_file")"
        [[ -n "$insertion_time" ]] || insertion_time="n/a"
        [[ -n "$total_time" ]] || total_time="n/a"
        [[ -n "$simplices" ]] || simplices="n/a"
        summaries+=("$dimension|$n_points|$simplices|$insertion_time|$total_time|$case_status")
        rm -f "$log_file"
    }

    run_case "2D" "debug_large_scale_2d" "DELAUNAY_LARGE_DEBUG_N_2D" "32000" "2000"
    run_case "3D" "debug_large_scale_3d" "DELAUNAY_LARGE_DEBUG_N_3D" "9000" "500"
    run_case "4D" "debug_large_scale_4d" "DELAUNAY_LARGE_DEBUG_N_4D" "1000" "100"
    run_case "5D" "debug_large_scale_5d" "DELAUNAY_LARGE_DEBUG_N_5D" "160" "20"

    echo ""
    echo "Large-scale smoke summary:"
    printf '%-4s %10s %12s %18s %18s %8s\n' "Dim" "Vertices" "Simplices" "Insertion wall" "Total wall" "Status"
    printf '%-4s %10s %12s %18s %18s %8s\n' "----" "--------" "---------" "--------------" "----------" "------"
    for row in "${summaries[@]}"; do
        IFS='|' read -r dimension n_points simplices insertion_time total_time case_status <<< "$row"
        printf '%-4s %10s %12s %18s %18s %8s\n' "$dimension" "$n_points" "$simplices" "$insertion_time" "$total_time" "$case_status"
    done

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

python-ci: python-check test-python
    @echo "✅ Python checks complete!"

# Python code quality
python-fix: _ensure-uv
    uv run ruff check scripts/ --fix
    uv run ruff format scripts/

python-lint: python-check

python-sync: _ensure-uv
    uv sync --group dev

python-typecheck: _ensure-uv
    uv run ty check scripts/ --error all

rust-core-check: fmt-check clippy-all-targets doc-check semgrep semgrep-test
    @echo "✅ Rust core checks complete!"

# Repository-owned Semgrep rules for project-specific Rust diagnostics.
semgrep: _ensure-uv
    uv run semgrep --error --strict --timeout 120 --config semgrep.yaml .

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
        uv run python scripts/semgrep_fixture_config.py "$fixture" "$PWD/semgrep.yaml" "$config_path"

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
    echo "External prerequisites that must already be on PATH: uv, jq, rustup, cargo, chktex, pkg-config, and ICU development files."
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

    echo "Ensuring paper native build dependencies..."
    if ! have pkg-config; then
        echo "❌ 'pkg-config' not found. Install pkg-config before building Tectonic from Cargo."
        exit 1
    fi
    if ! pkg-config --exists icu-uc; then
        shopt -s nullglob
        for candidate in \
            /opt/homebrew/opt/icu4c*/lib/pkgconfig \
            /usr/local/opt/icu4c*/lib/pkgconfig; do
            if [ -d "$candidate" ]; then
                export PKG_CONFIG_PATH="$candidate${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
                break
            fi
        done
        shopt -u nullglob
    fi
    if ! pkg-config --exists icu-uc; then
        echo "❌ 'icu-uc' was not found by pkg-config."
        echo "   Install ICU development files, or set PKG_CONFIG_PATH to the directory containing icu-uc.pc."
        exit 1
    fi
    echo "  ✓ pkg-config can resolve icu-uc"
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

    installed_tectonic_version=""
    if command -v tectonic >/dev/null; then
        installed_tectonic_version="$(tectonic --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
    fi
    if [[ "$installed_tectonic_version" != "{{ tectonic_version }}" ]]; then
        echo "  ⏳ Installing tectonic {{ tectonic_version }} (cargo)..."
        cargo install --locked tectonic --version {{ tectonic_version }}
    else
        echo "  ✓ tectonic {{ tectonic_version }}"
    fi

    installed_tex_fmt_version=""
    if command -v tex-fmt >/dev/null; then
        installed_tex_fmt_version="$(tex-fmt --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
    fi
    if [[ "$installed_tex_fmt_version" != "{{ tex_fmt_version }}" ]]; then
        echo "  ⏳ Installing tex-fmt {{ tex_fmt_version }} (cargo)..."
        cargo install --locked tex-fmt --version {{ tex_fmt_version }}
    else
        echo "  ✓ tex-fmt {{ tex_fmt_version }}"
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

    cmds=(uv jq pkg-config taplo dprint tectonic tex-fmt rumdl git-cliff typos zizmor chktex)
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
# test: runs each default test bucket once.
test: test-all
    @echo "✅ Test workflow passed!"

# test-all: runs Rust and Python tests.
test-all: test-rust test-python
    @echo "✅ All tests passed!"

test-allocation: _ensure-nextest
    cargo nextest run --profile ci --test allocation_api --features count-allocations -- --nocapture

test-diagnostics: _ensure-nextest
    cargo nextest run --profile ci --test circumsphere_debug_tools --features diagnostics -- --nocapture

# test-doc: runs Rust doctests in release profile.
test-doc:
    cargo test --doc --release --verbose

# test-integration: runs all default integration tests under the 10s per-test budget.
test-integration: _ensure-nextest
    cargo nextest run --release --profile ci --tests

# Compile release integration tests without running them.
test-integration-compile: _ensure-nextest
    cargo nextest run --release --tests --no-run

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

test-release: test-rust-ci test-doc

# test-rust: runs each default Rust correctness target class once.
test-rust: test-rust-ci test-doc
    @echo "✅ Rust tests passed!"

# test-rust-ci: runs Rust lib unit tests, integration tests, and feature-gated CLI tests.
test-rust-ci: _ensure-nextest
    cargo nextest run --release --profile ci --lib --tests
    cargo nextest run --release --profile ci --features cli --test cli

# Run correctness tests that exceed the 10s default-suite budget.
# Slow tests run in release mode because debug exact-predicate paths can turn
# a slow correctness check into a local timeout.
test-slow: _ensure-nextest
    cargo nextest run --release --profile slow --features slow-tests
    cargo test --doc --release --features slow-tests

test-slow-release: test-slow

# test-unit: runs Rust lib unit tests.
test-unit: _ensure-nextest
    cargo nextest run --profile ci --lib

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

toml-ci: toml-check toml-lint toml-fmt-check
    @echo "✅ TOML checks complete!"

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

yaml-ci: yaml-check citation-check
    @echo "✅ YAML/CFF checks complete!"

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
