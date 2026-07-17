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
binary_extension := if os_family() == "windows" { ".exe" } else { "" }
perf_delaunay_binary := "target/perf/delaunay" + binary_extension

cargo_audit_version := "0.22.2"
cargo_llvm_cov_version := "0.8.7"
cargo_machete_version := "0.9.2"
clippy_sarif_version := "0.8.0"
dprint_version := "0.55.2"
git_cliff_version := "2.13.1"
just_version := "1.56.0"
nextest_version := "0.9.140"
rumdl_version := "0.2.34"
samply_version := "0.13.1"
sarif_fmt_version := "0.8.0"
taplo_version := "0.10.0"
tectonic_version := "0.16.9"
tex_fmt_version := "0.5.7"
typos_version := "1.48.0"
uv_version := "0.11.29"
zizmor_version := "1.26.1"

# Common cargo-llvm-cov arguments for all coverage runs.
# Excludes benches/examples from reports while allowing integration tests to
# exercise library code.
[private]
_coverage_base_args := '''--ignore-filename-regex '(^|/)(benches|examples)/' \
  --workspace --lib --tests \
  --verbose'''

import 'just/helpers.just'

# GitHub Actions workflow validation
[group('validation')]
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
[group('benchmarks and performance')]
bench:
    cargo bench --workspace --profile perf --features bench

# Allocation-contract microbenchmarks for public hot paths.
[group('benchmarks and performance')]
bench-allocations:
    cargo bench --profile perf --bench allocation_hot_paths --features count-allocations -- --noplot

# CI regression benchmarks with the perf profile.
[group('benchmarks and performance')]
bench-ci:
    cargo bench --profile perf --bench ci_performance_suite

# Render a Markdown comparison against a saved Criterion baseline.
[group('benchmarks and performance')]
bench-compare baseline="last" suite="release-signal": _ensure-uv
    uv run benchmark-utils bench-compare "{{ baseline }}" --suite "{{ suite }}"

# Compile benchmark harnesses without running them.
[group('benchmarks and performance')]
bench-compile:
    @echo "Compiling benchmark harnesses without running them; this can take several minutes on Windows/MSVC."
    cargo bench --workspace --no-run --features bench

# Run the curated release-signal benchmark set and leave Criterion `new` output.
[group('benchmarks and performance')]
bench-latest:
    cargo bench --profile perf --bench ci_performance_suite
    cargo bench --profile perf --bench circumsphere_containment
    cargo bench --profile perf --bench cold_path_predicates
    cargo bench --profile perf --bench topology_guarantee_construction
    cargo bench --profile perf --bench locate

# Run latest measurements and render the latest-vs-last performance report.
[group('benchmarks and performance')]
bench-latest-vs-last baseline="last": bench-latest && (bench-compare baseline)

# Run Criterion's Pachner move and round-trip stress benchmark.
[group('benchmarks and performance')]
bench-pachner-stress samples="10": (_bench-pachner-stress samples)

# Generate a release performance summary from fresh perf-profile benchmark runs.
[group('benchmarks and performance')]
bench-perf-summary: _ensure-uv
    uv run benchmark-utils generate-summary --run-benchmarks --profile perf

# Save a Criterion baseline for a Delaunay benchmark suite.
[group('benchmarks and performance')]
bench-save-baseline tag suite="release-signal":
    #!/usr/bin/env bash
    set -euo pipefail
    tag="{{ tag }}"
    suite="{{ suite }}"
    case "$suite" in
        release-signal)
            targets=(ci_performance_suite circumsphere_containment cold_path_predicates topology_guarantee_construction locate)
            ;;
        ci)
            targets=(ci_performance_suite)
            ;;
        query)
            targets=(circumsphere_containment locate)
            ;;
        predicates)
            targets=(circumsphere_containment cold_path_predicates)
            ;;
        topology)
            targets=(topology_guarantee_construction)
            ;;
        *)
            echo "unknown benchmark suite: $suite" >&2
            exit 2
            ;;
    esac
    for target in "${targets[@]}"; do
        cargo bench --profile perf --bench "$target" -- --save-baseline "$tag"
    done

# Smoke-test benchmark harnesses with minimal samples; not for performance data.
# Criterion requires sample_size >= 10; use the minimum with short measurement/warm-up windows.
[doc('Smoke-test benchmark harnesses with minimal samples; do not use as performance data.')]
[group('benchmarks and performance')]
bench-smoke:
    CRIT_SAMPLE_SIZE=10 CRIT_MEASUREMENT_MS=500 CRIT_WARMUP_MS=200 cargo bench --workspace --profile perf --features bench

# Build the crate in the development profile.
[group('build and setup')]
build:
    cargo build

# Build the crate in the release profile.
[group('build and setup')]
build-release:
    cargo build --release

# Check that Cargo.toml and Cargo.lock are synchronized.
[group('validation')]
cargo-lock-check:
    cargo metadata --locked --format-version 1 --no-deps > /dev/null

# Changelog management (git-cliff + post-processing + archiving + rumdl formatting)
[group('release')]
changelog: _ensure-git-cliff _ensure-rumdl python-sync
    #!/usr/bin/env bash
    set -euo pipefail
    GIT_CLIFF_OFFLINE=true git-cliff -o CHANGELOG.md
    uv run postprocess-changelog
    uv run archive-changelog
    rumdl fmt --silent CHANGELOG.md docs/archive/changelog/*.md

# Generate the changelog as if releasing the requested version.
[group('release')]
changelog-unreleased version: _ensure-git-cliff _ensure-rumdl python-sync
    #!/usr/bin/env bash
    set -euo pipefail
    GIT_CLIFF_OFFLINE=true git-cliff --tag {{ version }} -o CHANGELOG.md
    uv run postprocess-changelog
    uv run archive-changelog
    rumdl fmt --silent CHANGELOG.md docs/archive/changelog/*.md

# Run every non-mutating validator outside the test suites.
[group('workflows')]
check: check-code check-config check-docs
    @echo "✅ Checks complete!"

# Check Rust and dependency hygiene, Python, notebooks, and shell scripts.
[group('validation')]
check-code: rust-core-check unused-deps python-check notebook-check shell-check

# Check justfile, JSON, TOML, YAML/CFF, and GitHub Actions configuration.
[group('validation')]
check-config: justfile-fmt-check cargo-lock-check json-check toml-check yaml-check citation-check github-actions-check

# Check Markdown, spelling, and release-version references.
[group('validation')]
check-docs: markdown-check spell-check docs-version-check

# Fast compile check (no binary produced)
[group('build and setup')]
check-fast:
    cargo check

# CI simulation: comprehensive validation.
[group('workflows')]
ci: check test bench-compile examples
    @echo "🎯 CI checks complete!"

# CI followed by an explicit persistent local baseline refresh.
[group('workflows')]
ci-baseline ref="main": ci && (perf-baseline ref)

# CI plus the explicit slow correctness bucket.
[group('workflows')]
ci-slow: ci test-slow
    @echo "✅ CI + slow tests passed!"

# Validate CITATION.cff against the Citation File Format schema.
[group('validation')]
citation-check: _ensure-uv
    uvx --from cffconvert==2.0.0 cffconvert --validate -i CITATION.cff

# Clean build artifacts
[group('build and setup')]
clean:
    cargo clean
    rm -rf target/llvm-cov
    rm -rf coverage_report
    rm -rf coverage

# Run strict Clippy checks for every target with default and all features.
[group('validation')]
clippy:
    cargo clippy --workspace --all-targets -- -D warnings -W clippy::pedantic -W clippy::nursery -W clippy::cargo
    cargo clippy --workspace --all-targets --all-features -- -D warnings -W clippy::pedantic -W clippy::nursery -W clippy::cargo

# Coverage analysis for local development (HTML output)
[group('tests and coverage')]
coverage: _ensure-cargo-llvm-cov
    mkdir -p target/llvm-cov
    cargo llvm-cov {{ _coverage_base_args }} --html --output-dir target/llvm-cov
    @echo "📊 Coverage report generated: target/llvm-cov/html/index.html"

# Coverage analysis for CI (XML output for codecov/codacy)
[group('tests and coverage')]
coverage-ci: _ensure-cargo-llvm-cov
    mkdir -p coverage
    cargo llvm-cov nextest {{ _coverage_base_args }} --cobertura --output-path coverage/cobertura.xml -P coverage

# Run the large-scale 2D diagnostic fixture with progress output.
[group('diagnostics')]
debug-large-scale-2d n="36000" repair_every="1": _ensure-nextest
    DELAUNAY_BULK_PROGRESS_EVERY=2000 DELAUNAY_LARGE_DEBUG_MAX_RUNTIME_SECS=1800 DELAUNAY_LARGE_DEBUG_N_2D={{ n }} DELAUNAY_LARGE_DEBUG_REPAIR_EVERY={{ repair_every }} cargo nextest run --release --profile slow --features slow-tests --test large_scale_debug debug_large_scale_2d -- --exact --nocapture

# Run the large-scale 3D diagnostic fixture with progress output.
[group('diagnostics')]
debug-large-scale-3d n="7500" repair_every="1": _ensure-nextest
    DELAUNAY_BULK_PROGRESS_EVERY=500 DELAUNAY_LARGE_DEBUG_MAX_RUNTIME_SECS=1800 DELAUNAY_LARGE_DEBUG_N_3D={{ n }} DELAUNAY_LARGE_DEBUG_REPAIR_EVERY={{ repair_every }} cargo nextest run --release --profile slow --features slow-tests --test large_scale_debug debug_large_scale_3d -- --exact --nocapture

# Run the large-scale 4D diagnostic fixture with progress output.
[group('diagnostics')]
debug-large-scale-4d n="800" repair_every="1": _ensure-nextest
    DELAUNAY_BULK_PROGRESS_EVERY=100 DELAUNAY_LARGE_DEBUG_MAX_RUNTIME_SECS=1800 DELAUNAY_LARGE_DEBUG_N_4D={{ n }} DELAUNAY_LARGE_DEBUG_REPAIR_EVERY={{ repair_every }} cargo nextest run --release --profile slow --features slow-tests --test large_scale_debug debug_large_scale_4d -- --exact --nocapture

# Run the large-scale 5D diagnostic fixture with progress output.
[group('diagnostics')]
debug-large-scale-5d n="140" repair_every="1": _ensure-nextest
    DELAUNAY_BULK_PROGRESS_EVERY=20 DELAUNAY_LARGE_DEBUG_MAX_RUNTIME_SECS=1800 DELAUNAY_LARGE_DEBUG_N_5D={{ n }} DELAUNAY_LARGE_DEBUG_REPAIR_EVERY={{ repair_every }} cargo nextest run --release --profile slow --features slow-tests --test large_scale_debug debug_large_scale_5d -- --exact --nocapture

# Show the curated workflow guide when `just` is invoked without a recipe.
[default]
[private]
default: help-workflows

# Build rustdoc for the workspace and reject warnings.
[group('validation')]
doc-check:
    RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --document-private-items

# Check release-version references against Cargo.toml.
[group('validation')]
docs-version-check: _ensure-uv
    uv run --locked check-docs-version-sync

# Build and run every Rust example.
[group('tests and coverage')]
examples:
    ./scripts/run_all_examples.sh

# Fix (mutating): apply formatters/auto-fixes
[group('workflows')]
fix: justfile-fmt toml-fix fmt python-fix shell-fix markdown-fix yaml-fix
    @echo "✅ Fixes applied!"

# Format Rust source files.
[group('validation')]
fmt:
    cargo fmt --all

# Check Rust source formatting without modifying files.
[group('validation')]
fmt-check:
    cargo fmt --all -- --check

# Run actionlint and zizmor over GitHub Actions workflows.
[group('validation')]
github-actions-check: action-lint zizmor
    @echo "✅ GitHub Actions checks complete!"

# Show the curated entry points for common repository workflows.
[group('workflows')]
help-workflows:
    @echo "Recommended workflows:"
    @echo "  just check              # All non-mutating source, config, and docs checks"
    @echo "  just fix                # Apply repository formatters and safe auto-fixes"
    @echo "  just test               # Default Rust and Python test buckets"
    @echo "  just ci                 # GitHub-equivalent default validation suite"
    @echo ""
    @echo "Focused checks:"
    @echo "  just check-code         # Rust, Python, notebooks, and shell"
    @echo "  just check-config       # Just, Cargo, JSON, TOML, YAML/CFF, and Actions"
    @echo "  just check-docs         # Markdown, spelling, and version references"
    @echo "  just rust-core-check    # Rust fmt, Clippy, rustdoc, and Semgrep"
    @echo "  just python-check       # Ruff formatting/lint plus ty type checking"
    @echo "  just notebook-check     # Notebook hygiene and extracted-code checks"
    @echo "  just shell-check        # ShellCheck plus shfmt verification"
    @echo ""
    @echo "Focused tests:"
    @echo "  just test-rust          # Unit, integration, CLI, and doctests"
    @echo "  just test-unit          # Debug and release Rust lib unit tests"
    @echo "  just test-integration   # Release integration tests, including proptests"
    @echo "  just test-integration-fast # Integration tests without proptests"
    @echo "  just test-cli           # CLI-feature integration tests"
    @echo "  just test-doc           # Release doctests"
    @echo "  just test-python        # Python support-script tests"
    @echo "  just test-slow          # Explicit slow correctness bucket"
    @echo "  just examples           # Build and run every Rust example"
    @echo ""
    @echo "Notebooks and papers:"
    @echo "  just notebook           # Launch the default source notebook"
    @echo "  just notebook-execute   # Execute one notebook under target/notebooks"
    @echo "  just validation-doc-figures # Refresh canonical validation figures"
    @echo "  just paper-check        # Lint, build, and check without tracked changes"
    @echo "  just paper-refresh      # Check, then refresh one tracked reviewer PDF"
    @echo "  just papers             # Refresh figures and the validation reviewer PDF"
    @echo ""
    @echo "Performance checks (perf-*):"
    @echo "  just perf-no-regressions # Fast guard against the cached main baseline"
    @echo "  just perf-vs-ref <ref>  # Compare against another cached Git ref"
    @echo "  just perf-large-scale-smoke # Bounded 2D-5D wall-clock guard"
    @echo "  just perf-local         # Compare the current tree with the latest release"
    @echo "  just perf-help          # Detailed performance and profiling commands"
    @echo ""
    @echo "Larger benchmarks (bench-*):"
    @echo "  just bench-compile      # Compile benchmark harnesses without running"
    @echo "  just bench-smoke        # Smoke-test harnesses; not performance evidence"
    @echo "  just bench              # Run the complete benchmark suite"
    @echo "  just bench-ci           # Run the CI regression benchmark suite"
    @echo "  just bench-latest-vs-last # Measure release signals and compare to last"
    @echo "  just bench-perf-summary # Generate the release performance summary"
    @echo ""
    @echo "Release and optional workflows:"
    @echo "  just publish-check      # Validate metadata and cargo publish --dry-run"
    @echo "  just changelog          # Regenerate and format changelog artifacts"
    @echo "  just ci-slow            # Default CI plus slow correctness tests"
    @echo "  just ci-baseline        # Default CI plus persistent perf baseline refresh"
    @echo "  just coverage           # Generate local HTML coverage"
    @echo ""
    @echo "Use 'just --list' for the complete grouped recipe reference."

# Check JSON files parse cleanly.
[group('validation')]
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

# Format the root and helper justfiles.
[group('validation')]
justfile-fmt:
    just --fmt
    just --fmt --justfile just/helpers.just

# Check root and helper justfile formatting without modifying them.
[group('validation')]
justfile-fmt-check:
    just --fmt --check
    just --fmt --check --justfile just/helpers.just

# Check Markdown formatting and raw line length.
[group('validation')]
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

# Apply automatic Markdown fixes.
[group('validation')]
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

# Launch one source notebook in JupyterLab.
[group('notebooks and papers')]
notebook notebook="notebooks/00_quickstart.ipynb": _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    notebook_cache="$(pwd)/target/notebooks"
    mkdir -p "$notebook_cache/.ipython" "$notebook_cache/.matplotlib"
    MPLBACKEND=Agg IPYTHONDIR="$notebook_cache/.ipython" MPLCONFIGDIR="$notebook_cache/.matplotlib" uv run --group notebooks jupyter lab --ServerApp.open_browser=True --LabApp.open_browser=True "{{ notebook }}"

# Run routine non-executing notebook validation.
[group('notebooks and papers')]
notebook-check: _ensure-uv
    uv run --group dev --group notebooks notebook-check lint --repo-root .
    @echo "📓 Notebook checks complete!"

# Clear outputs from one source notebook in place.
[group('notebooks and papers')]
notebook-clear-outputs notebook="notebooks/00_quickstart.ipynb": _ensure-uv
    uv run --group notebooks jupyter nbconvert --clear-output --inplace "{{ notebook }}"

# Clear outputs from every source notebook in place.
[group('notebooks and papers')]
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

# Execute one notebook into target/notebooks without modifying its source.
[group('notebooks and papers')]
notebook-execute notebook="notebooks/00_quickstart.ipynb" output_dir="target/notebooks" timeout="600": _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    output_path="$(pwd)/{{ output_dir }}"
    notebook_stem="$(basename "{{ notebook }}" .ipynb)"
    notebook_output_dir="$output_path/$notebook_stem"
    mkdir -p "$output_path/.ipython" "$output_path/.matplotlib" "$notebook_output_dir"
    MPLBACKEND=Agg IPYTHONDIR="$output_path/.ipython" MPLCONFIGDIR="$output_path/.matplotlib" uv run --group notebooks jupyter nbconvert --execute --ExecutePreprocessor.timeout={{ timeout }} --ExecutePreprocessor.shutdown_kernel=immediate --to notebook --output-dir "$notebook_output_dir" "{{ notebook }}"

# Check notebook structure and output hygiene without extracted-code linting.
[group('notebooks and papers')]
notebook-output-check: _ensure-uv
    uv run --group dev --group notebooks notebook-check lint --repo-root . --no-ruff --no-format --no-ty

# Restore tracked source notebooks and remove generated notebook artifacts.
[group('notebooks and papers')]
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

# Install the optional notebook dependency group.
[group('notebooks and papers')]
notebook-setup: _ensure-uv
    uv sync --group notebooks

# Run one 3D and one 4D direct Pachner stress workload with topology-scope reports enabled.
[group('benchmarks and performance')]
pachner-stress attempts="100" validate_every="10" mode="round-trip": (_pachner-stress-dim "3d" "9000" attempts validate_every "target/pachner_stress/3d" mode) (_pachner-stress-dim "4d" "1000" attempts validate_every "target/pachner_stress/4d" mode)

# Run one 3D direct Pachner stress workload with topology-scope reports enabled.
[group('benchmarks and performance')]
pachner-stress-3d attempts="100" vertices="9000" validate_every="10" output_dir="target/pachner_stress/3d" mode="round-trip": (_pachner-stress-dim "3d" vertices attempts validate_every output_dir mode)

# Run one 4D direct Pachner stress workload with topology-scope reports enabled.
[group('benchmarks and performance')]
pachner-stress-4d attempts="100" vertices="1000" validate_every="10" output_dir="target/pachner_stress/4d" mode="round-trip": (_pachner-stress-dim "4d" vertices attempts validate_every output_dir mode)

# Compile one paper with Tectonic under target/papers/.
[group('notebooks and papers')]
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
    echo "📄 Paper PDF built: $build_dir/${paper}.pdf"

# Compile and check one paper without refreshing tracked artifacts.
[group('notebooks and papers')]
paper-check paper="validation": paper-tex-fmt-check paper-tex-lint (paper-build paper) && (paper-pdf-check paper)

# Remove generated paper build artifacts under target/papers.
[group('notebooks and papers')]
paper-clean:
    rm -rf target/papers

# Build the CLI used by paper notebooks before nbconvert starts its execution timer.
[group('notebooks and papers')]
paper-cli:
    cargo build --profile perf --features cli --bin delaunay

# Check the target-built PDF for basic readability.
[group('notebooks and papers')]
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
    uv run paper-pdf-check "target/papers/${paper}/${paper}.pdf" \
        --min-pages 1 \
        --require-text "Validation Architecture in delaunay" \
        --require-text "REFERENCES" \
        --forbid-text "\\today" \
        --forbid-text "Manuscript submitted to ACM"

# Refresh one tracked reviewer PDF after its non-mutating checks pass.
[group('notebooks and papers')]
paper-refresh paper="validation": (paper-check paper)
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
    source_pdf="target/papers/${paper}/${paper}.pdf"
    reviewer_pdf="papers/${paper}.pdf"
    cp "$source_pdf" "$reviewer_pdf"
    echo "📄 Reviewer PDF refreshed: $reviewer_pdf"

# Format publication-facing TeX sources.
[group('notebooks and papers')]
paper-tex-fmt: _ensure-tex-fmt
    tex-fmt papers/*.tex

# Check publication-facing TeX formatting without modifying files.
[group('notebooks and papers')]
paper-tex-fmt-check: _ensure-tex-fmt
    tex-fmt --check papers/*.tex

# Lint publication-facing TeX sources with ChkTeX.
[group('notebooks and papers')]
paper-tex-lint: _ensure-chktex
    #!/usr/bin/env bash
    set -euo pipefail
    # 24 conflicts with tex-fmt's indented figure labels.
    chktex -q -n 1 -n 8 -n 24 -n 46 papers/*.tex

# Refresh notebook-owned paper figures, lint TeX, compile, and sanity-check PDFs.
[group('notebooks and papers')]
papers: validation-doc-figures (paper-refresh "validation")
    @echo "📚 Paper workflow complete!"

# Generate a same-machine dev-mode baseline for a GitHub ref.
[group('benchmarks and performance')]
perf-baseline ref="main": _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    uv run benchmark-utils generate-ref-baseline --ref "{{ ref }}" --out baseline-artifact --dev

# Generate a scratch same-machine baseline at an explicit output path.
[group('benchmarks and performance')]
perf-baseline-to out ref="main": _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    uv run benchmark-utils generate-ref-baseline --ref "{{ ref }}" --out "{{ out }}" --dev

# Compare the current tree with one dev-mode baseline file.
[group('benchmarks and performance')]
perf-compare file threshold="7.5": _ensure-uv
    uv run benchmark-utils compare --baseline "{{ file }}" --threshold {{ threshold }} --dev

# Compare stored GitHub Release benchmark assets without local cargo runs.
[group('benchmarks and performance')]
perf-github-assets current_tag="" baseline_tag="": _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    current_tag="{{ current_tag }}"
    baseline_tag="{{ baseline_tag }}"
    tag_pair_state="$(just --quiet _performance-tag-pair-state "$current_tag" "$baseline_tag")"
    if [[ "$tag_pair_state" == "invalid" ]]; then
        exit 2
    fi
    if [[ "$tag_pair_state" == "explicit" ]]; then
        uv run benchmark-utils performance-github-assets "$current_tag" "$baseline_tag"
    else
        uv run benchmark-utils performance-github-assets
    fi

# Show detailed performance-check, benchmark, and profiling workflows.
[group('benchmarks and performance')]
perf-help:
    @echo "Performance Analysis Commands:"
    @echo "  just bench-latest          # Run curated release-signal Criterion benchmarks"
    @echo "  just bench-latest-vs-last  # Run latest and compare against saved 'last'"
    @echo "  just bench-compare [base]  # Render a Markdown report from saved Criterion baselines"
    @echo "  just bench-save-baseline <tag> # Save release-signal Criterion baseline"
    @echo "  just bench-save-baseline last # Save release-signal Criterion baseline as 'last'"
    @echo "  just perf-local           # Compare current tree against latest release locally"
    @echo "  just perf-github-assets   # Compare stored GitHub Release benchmark assets"
    @echo "  just perf-release         # Promote release-to-release performance docs"
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
    @echo "  just pachner-stress        # 3D+4D direct Pachner CLI stress with CSV/JSON artifacts"
    @echo "  just pachner-stress-3d     # 3D Pachner CLI stress (100 moves, 9K vertices)"
    @echo "  just pachner-stress-4d     # 4D Pachner CLI stress (100 moves, 1K vertices)"
    @echo "  just bench-pachner-stress  # Criterion timing for Pachner move/round-trip stress"
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
    @echo "  just pachner-stress-4d 100000 1000 1000 target/pachner_stress/4d random-walk"
    @echo "                              # 4D random-walk Pachner diagnostics with CSV/JSON artifacts"
    @echo "  just bench-ci              # Final optimized CI-suite benchmark run"
    @echo "  just profile v0.7.5        # v0.7.5 code on its declared Rust toolchain"
    @echo "  just profile 1.97.0        # Current tree on Rust 1.97.0"
    @echo "  just profile 1.97.0 v0.7.5 # v0.7.5 code on Rust 1.97.0"

# Quick pre-push 2D-5D large-scale wall-clock smoke guard.
[group('benchmarks and performance')]
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

# Compare the current tree against the latest published release in temp worktrees.
[group('benchmarks and performance')]
perf-local: _ensure-uv
    uv run benchmark-utils performance-local

# Fast pre-PR performance guard against a cached same-machine main baseline.
[group('benchmarks and performance')]
perf-no-regressions threshold="7.5": _ensure-uv
    uv run benchmark-utils compare-ref --ref main --threshold {{ threshold }} --dev --output benches/worktree_vs_main_compare_results.txt

# Generate local release-signal measurements in temp worktrees, then promote/archive docs.
[group('benchmarks and performance')]
perf-release current_tag="" baseline_tag="": _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    current_tag="{{ current_tag }}"
    baseline_tag="{{ baseline_tag }}"
    tag_pair_state="$(just --quiet _performance-tag-pair-state "$current_tag" "$baseline_tag")"
    if [[ "$tag_pair_state" == "invalid" ]]; then
        exit 2
    fi
    if [[ "$tag_pair_state" == "explicit" ]]; then
        uv run benchmark-utils performance-release "$current_tag" "$baseline_tag"
    else
        uv run benchmark-utils performance-release
    fi

# Compare the current tree against a cached same-machine ref baseline.
[group('benchmarks and performance')]
perf-vs-ref ref threshold="7.5": _ensure-uv
    uv run benchmark-utils compare-ref --ref "{{ ref }}" --threshold {{ threshold }} --dev

# Run the selected CI benchmark suite for one compiler/code pair.
[group('benchmarks and performance')]
profile toolchain="" code_ref="current": _ensure-jq
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

# Profile 3D construction with Samply in the development configuration.
[group('benchmarks and performance')]
profile-dev: _ensure-samply
    PROFILING_DEV_MODE=1 samply record cargo bench --profile perf --bench profiling_suite -- "construction/3D/5000v/construct"

# Profile allocation-heavy construction with Samply.
[group('benchmarks and performance')]
profile-mem: _ensure-samply
    samply record cargo bench --profile perf --bench profiling_suite --features count-allocations -- memory_profiling

# Pre-publish validation: checks crates.io metadata rules that cargo publish --dry-run does NOT catch
# Validate crates.io metadata and run cargo publish --dry-run.
[group('release')]
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

# Run every non-mutating Python source check.
[group('validation')]
python-check: python-format-check python-lint python-typecheck
    @echo "✅ Python source checks complete!"

# Apply Ruff lint fixes and formatting to Python source.
[group('validation')]
python-fix: _ensure-uv
    uv run ruff check scripts/ --fix
    uv run ruff format scripts/

# Check Python formatting with Ruff.
[group('validation')]
python-format-check: _ensure-uv
    uv run ruff format --check scripts/

# Lint Python source with Ruff.
[group('validation')]
python-lint: _ensure-uv
    uv run ruff check scripts/

# Synchronize development Python dependencies from the lockfile.
[group('build and setup')]
python-sync: _ensure-uv
    uv sync --group dev

# Type-check Python support code with ty.
[group('validation')]
python-typecheck: _ensure-uv
    uv run ty check scripts/ --error all

# Run the opt-in companion binary with the CLI feature and perf profile.
[group('build and setup')]
run *args:
    cargo run --profile perf --features cli --bin delaunay -- {{ args }}

# Run the complete non-mutating Rust validation surface.
[group('validation')]
rust-core-check: fmt-check clippy doc-check semgrep semgrep-test
    @echo "✅ Rust core checks complete!"

# Repository-owned Semgrep rules for project-specific Rust diagnostics.
[group('validation')]
semgrep: _ensure-uv
    uv run semgrep --error --strict --timeout 120 --config semgrep.yaml .

# Test the repository-owned Semgrep rules against their fixtures.
[group('validation')]
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

# Install required tools and build the development profile.
[group('build and setup')]
setup: setup-tools build
    echo "✅ Setup complete! Run 'just help-workflows' to see available commands."

# Install and verify pinned repository development tools.
[doc('Install and verify pinned repository development tools.')]
[group('build and setup')]
setup-tools: _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail

    source scripts/cargo_tool_versions.sh

    echo "🔧 Ensuring tooling required by just recipes is installed..."
    echo ""

    have() { command -v "$1" >/dev/null 2>&1; }

    ensure_pinned_cargo_tool() {
        local binary="$1"
        local package="$2"
        local expected_version="$3"
        local cargo_subcommand="${4:-}"

        if ! cargo_tool_has_exact_version "$binary" "$expected_version" "$cargo_subcommand"; then
            echo "  ⏳ Installing $package $expected_version (cargo)..."
            cargo install --locked "$package" --version "$expected_version"
        else
            echo "  ✓ $binary $expected_version"
        fi
    }

    ensure_tectonic_build_dependencies() {
        echo "Ensuring native dependencies needed to install Tectonic..."
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
    }

    echo "This recipe installs pinned Rust CLI tools through cargo."
    echo "External prerequisites that must already be on PATH: uv, jq, rustup, cargo, and chktex."
    echo "pkg-config and ICU development files are required only when the pinned Tectonic version must be installed."
    echo ""

    echo "Ensuring uv-managed Python tooling..."
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
    if ! rustup component list --installed | grep -q '^llvm-tools'; then
        echo "  ⏳ Installing llvm-tools-preview (rustup)..."
        rustup component add llvm-tools-preview
    else
        echo "  ✓ llvm-tools-preview"
    fi

    ensure_pinned_cargo_tool cargo-llvm-cov cargo-llvm-cov "{{ cargo_llvm_cov_version }}" llvm-cov
    ensure_pinned_cargo_tool cargo-machete cargo-machete "{{ cargo_machete_version }}"
    ensure_pinned_cargo_tool cargo-nextest cargo-nextest "{{ nextest_version }}"
    ensure_pinned_cargo_tool dprint dprint "{{ dprint_version }}"
    ensure_pinned_cargo_tool git-cliff git-cliff "{{ git_cliff_version }}"
    ensure_pinned_cargo_tool just just "{{ just_version }}"
    ensure_pinned_cargo_tool rumdl rumdl "{{ rumdl_version }}"
    ensure_pinned_cargo_tool samply samply "{{ samply_version }}"
    ensure_pinned_cargo_tool taplo taplo-cli "{{ taplo_version }}"
    if ! cargo_tool_has_exact_version tectonic "{{ tectonic_version }}"; then
        ensure_tectonic_build_dependencies
    fi
    ensure_pinned_cargo_tool tectonic tectonic "{{ tectonic_version }}"
    ensure_pinned_cargo_tool tex-fmt tex-fmt "{{ tex_fmt_version }}"
    ensure_pinned_cargo_tool typos typos-cli "{{ typos_version }}"
    ensure_pinned_cargo_tool zizmor zizmor "{{ zizmor_version }}"

    echo ""
    echo "Verifying required commands are available..."
    missing=0

    cmds=(uv jq taplo dprint tectonic tex-fmt rumdl git-cliff typos zizmor chktex samply)
    cmds+=(cargo-nextest cargo-llvm-cov cargo-machete)

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

# Run ShellCheck and verify shfmt formatting.
[group('validation')]
shell-check: shell-lint shell-fmt-check
    @echo "✅ Shell checks complete!"

# Format tracked shell scripts with shfmt.
[group('validation')]
shell-fix: _ensure-shfmt
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

# Check tracked shell-script formatting with shfmt.
[group('validation')]
shell-fmt-check: _ensure-shfmt
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '*.sh')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 uv run shfmt -d
    else
        echo "No shell files found to check."
    fi

# Lint tracked shell scripts with ShellCheck.
[group('validation')]
shell-lint: _ensure-shellcheck
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '*.sh')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 -n4 uv run shellcheck -x
    else
        echo "No shell files found to lint."
    fi

# Spell check (typos)
[group('validation')]
spell-check: _ensure-typos
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    # Check the complete repository surface in clean CI while including new,
    # unignored files during local iteration.
    while IFS= read -r -d '' filename; do
        [ -e "$filename" ] || continue
        [ "$filename" = "typos.toml" ] && continue
        files+=("$filename")
    done < <(git ls-files -z --cached --others --exclude-standard)
    if [ "${#files[@]}" -gt 0 ]; then
        # Exclude typos.toml itself: it intentionally contains allowlisted fragments.
        printf '%s\0' "${files[@]}" | xargs -0 -n100 typos --config typos.toml --force-exclude --exclude typos.toml --
    else
        echo "No repository files to spell-check."
    fi

# Deliberately refresh the slow notebook-backed spherical README hero.
[group('notebooks and papers')]
spherical-readme-hero: _ensure-uv paper-cli
    #!/usr/bin/env bash
    set -euo pipefail
    DELAUNAY_SPHERICAL_HERO_FIGURE="docs/assets/readme/delaunay_spherical_readme.png" just notebook-execute notebooks/02_spherical_hero.ipynb target/docs/notebooks 1800

# Create an annotated git tag from the CHANGELOG.md section for the given version
[group('release')]
tag version: python-sync
    uv run tag-release {{ version }}

# Replace an existing annotated tag from the CHANGELOG.md section.
[group('release')]
tag-force version: python-sync
    uv run tag-release {{ version }} --force

# Run every default Rust and Python test bucket once.
[group('workflows')]
test: test-rust test-python
    @echo "✅ Test workflow passed!"

# Run public allocation-contract integration tests.
[group('tests and coverage')]
test-allocation: _ensure-nextest
    cargo nextest run --profile ci --test allocation_api --features count-allocations -- --nocapture

# Run CLI-feature integration tests in the release profile.
[group('tests and coverage')]
test-cli: _ensure-nextest
    cargo nextest run --release --profile ci --features cli --test cli

# Run diagnostics-feature integration tests with captured output.
[group('diagnostics')]
test-diagnostics: _ensure-nextest
    cargo nextest run --profile ci --test circumsphere_debug_tools --features diagnostics -- --nocapture

# Run Rust doctests in the release profile.
[group('tests and coverage')]
test-doc:
    cargo test --doc --release --verbose

# test-integration: runs all default integration tests under the 10s per-test budget.
[group('tests and coverage')]
test-integration: _ensure-nextest
    cargo nextest run --release --profile ci --test '*'

# Compile release integration tests without running them.
[group('tests and coverage')]
test-integration-compile: _ensure-nextest
    cargo nextest run --release --test '*' --no-run

# test-integration-fast: runs integration tests but skips proptests (tests prefixed with `prop_`)
#
# Useful for quick local validation on changes that don't touch the property-test surface area.
# To run the full (slow) property suite, use: just test-integration
#
# Note: `--skip prop_` is a substring filter applied by the Rust test harness.
[doc('Run release integration tests while skipping property tests.')]
[group('tests and coverage')]
test-integration-fast: _ensure-nextest
    cargo nextest run --release --profile ci --test '*' -- --skip prop_

# Run Python support-script tests with pytest.
[group('tests and coverage')]
test-python: _ensure-uv
    uv run pytest

# Run every default Rust correctness target class once.
[group('tests and coverage')]
test-rust: test-unit test-integration test-cli test-doc
    @echo "✅ Rust tests passed!"

# Run correctness tests that exceed the 10s default-suite budget.
# Slow tests run in release mode because debug exact-predicate paths can turn
# a slow correctness check into a local timeout.
[doc('Run release correctness tests that exceed the default per-test budget.')]
[group('tests and coverage')]
test-slow: _ensure-nextest
    cargo nextest run --release --profile slow --features slow-tests
    cargo test --doc --release --features slow-tests

# Run Rust lib unit tests in debug and release profiles.
[group('tests and coverage')]
test-unit: _ensure-nextest
    cargo nextest run --profile debug --lib
    cargo nextest run --release --profile ci --lib

# Run TOML parsing, lint, and formatting checks.
[group('validation')]
toml-check: toml-parse-check toml-lint toml-fmt-check
    @echo "✅ TOML checks complete!"

# Format tracked TOML files with Taplo.
[group('validation')]
toml-fix: _ensure-taplo
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

# Check tracked TOML formatting with Taplo.
[group('validation')]
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

# Lint tracked TOML files with Taplo.
[group('validation')]
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

# Check that tracked TOML files parse cleanly.
[group('validation')]
toml-parse-check: _ensure-uv
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

# Check for unused direct Cargo dependencies.
[group('validation')]
unused-deps: _ensure-cargo-machete
    cargo machete

# Refresh reviewer-facing validation diagrams from the reproducible notebook.
[group('notebooks and papers')]
validation-doc-figures: _ensure-uv paper-cli
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p docs/assets/validation
    DELAUNAY_BINARY="{{ perf_delaunay_binary }}" DELAUNAY_VALIDATION_DOC_FIGURE_DIR="docs/assets/validation" just notebook-execute notebooks/01_validation.ipynb target/docs/notebooks

# Verify repository-owned source-pattern count invariants.
[group('validation')]
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

# Run YAML/CFF lint and formatting checks.
[group('validation')]
yaml-check: yaml-fmt-check yaml-lint
    @echo "✅ YAML/CFF checks complete!"

# Format tracked YAML/CFF files with dprint.
[group('validation')]
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

# Check tracked YAML/CFF formatting with dprint.
[group('validation')]
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

# Lint tracked YAML/CFF files with yamllint.
[group('validation')]
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

# Audit GitHub Actions workflows with zizmor.
[group('validation')]
zizmor: _ensure-zizmor
    zizmor .github
