# shellcheck disable=SC2148
# Justfile for delaunay development workflow
# Install just: https://github.com/casey/just
# Usage: just <command> or just --list

# Use bash with strict error handling for all recipes
set shell := ["bash", "-euo", "pipefail", "-c"]

binary_extension := if os_family() == "windows" { ".exe" } else { "" }
perf_delaunay_binary := "target/perf/delaunay" + binary_extension

# Invoke the locked shared CLI; managed execution verifies declared tools first.
rrt := "uv run --locked --group dev research-repo-tools"
managed := rrt + " toolchain run --"

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
    {{ rrt }} files run --include '.github/workflows/*.yml' --include '.github/workflows/*.yaml' -- uv run --locked actionlint

# Audit every maintained Python and Rust lockfile without executing dependency code.
[group('security')]
audit:
    {{ rrt }} security osv uv.lock Cargo.lock tests/fixtures/checkpoint_no_float_roundtrip/Cargo.lock

# Benchmark recipes that produce performance numbers use Cargo's perf profile.
[group('benchmarks and performance')]
bench:
    {{ managed }} cargo bench --workspace --profile perf --features bench

# Allocation-contract microbenchmarks for public hot paths.
[group('benchmarks and performance')]
bench-allocations:
    {{ managed }} cargo bench --profile perf --bench allocation_hot_paths --features count-allocations -- --noplot

# CI regression benchmarks with the perf profile.
[group('benchmarks and performance')]
bench-ci:
    {{ managed }} cargo bench --profile perf --bench ci_performance_suite

# Render a Markdown comparison against a saved Criterion baseline.
[group('benchmarks and performance')]
bench-compare baseline="last" suite="release-signal" scope="release-signal": _ensure-uv
    {{ managed }} uv run --locked benchmark-utils bench-compare "{{ baseline }}" --suite "{{ suite }}" --scope "{{ scope }}"

# Compile benchmark harnesses without running them.
[group('benchmarks and performance')]
bench-compile:
    @echo "Compiling benchmark harnesses without running them; this can take several minutes on Windows/MSVC."
    {{ managed }} cargo bench --workspace --no-run --features bench

# Run the curated release-signal benchmark set and leave Criterion `new` output.
[group('benchmarks and performance')]
bench-latest bench_timeout="1800": _ensure-uv
    {{ managed }} uv run --locked benchmark-utils run-release-signal --bench-timeout {{ bench_timeout }}

# Run latest measurements and render the latest-vs-last performance report.
[group('benchmarks and performance')]
bench-latest-vs-last baseline="last" bench_timeout="1800": (bench-latest bench_timeout) && (bench-compare baseline)

# Run Criterion's Pachner move and round-trip stress benchmark.
[group('benchmarks and performance')]
bench-pachner-stress samples="10": (_bench-pachner-stress samples)

# Generate a release performance summary from fresh perf-profile benchmark runs.
[group('benchmarks and performance')]
bench-perf-summary: _ensure-uv
    {{ managed }} uv run --locked benchmark-utils generate-summary --run-benchmarks --profile perf --strict

# Execute every curated release fixture once without producing timing evidence.
[group('benchmarks and performance')]
bench-preflight bench_timeout="600": _ensure-uv
    {{ managed }} uv run --locked benchmark-utils run-release-signal --preflight-only --bench-timeout {{ bench_timeout }}

# Save a Criterion baseline for a Delaunay benchmark suite.
[group('benchmarks and performance')]
bench-save-baseline tag suite="release-signal": _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    tag="{{ tag }}"
    suite="{{ suite }}"
    case "$suite" in
        release-signal)
            {{ managed }} uv run --locked benchmark-utils run-release-signal --save-baseline "$tag"
            exit 0
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
        {{ managed }} cargo bench --profile perf --bench "$target" -- --save-baseline "$tag"
    done

# Smoke-test benchmark harnesses with minimal samples; not for performance data.
# Criterion requires sample_size >= 10; use the minimum with short measurement/warm-up windows.
[doc('Smoke-test benchmark harnesses with minimal samples; do not use as performance data.')]
[group('benchmarks and performance')]
bench-smoke:
    CRIT_SAMPLE_SIZE=10 CRIT_MEASUREMENT_MS=500 CRIT_WARMUP_MS=200 {{ managed }} cargo bench --workspace --profile perf --features bench

# Build the crate in the development profile.
[group('build and setup')]
build:
    {{ managed }} cargo build

# Build the crate in the release profile.
[group('build and setup')]
build-release:
    {{ managed }} cargo build --release

# Check that Cargo.toml and Cargo.lock are synchronized.
[group('validation')]
cargo-lock-check:
    {{ managed }} cargo metadata --locked --format-version 1 --no-deps > /dev/null

# Changelog management through the pinned shared package.
[group('release')]
changelog:
    {{ managed }} research-repo-tools changelog generate

# Rotate completed minor release series without regenerating history.
[group('release')]
changelog-archive:
    {{ rrt }} changelog archive

# Check generated release history and archive links.
[group('release')]
changelog-check:
    {{ rrt }} changelog check

# Preview normalized and rotated history without publishing files.
[group('release')]
changelog-preview:
    {{ managed }} research-repo-tools changelog generate --dry-run

# Generate a prospective release using an explicit UTC date.
[group('release')]
changelog-release tag date:
    {{ managed }} research-repo-tools changelog generate --tag {{ quote(tag) }} --date {{ quote(date) }}

alias changelog-unreleased := changelog-release

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
    {{ managed }} cargo check

# CI simulation: comprehensive validation.
[group('workflows')]
ci: _validation-doc-figures-check-if-canonical check python-fixture-lint test bench-compile examples
    @echo "🎯 CI checks complete!"

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
    {{ managed }} cargo clean
    rm -rf target/llvm-cov
    rm -rf coverage_report
    rm -rf coverage

# Run strict Clippy checks for every target with default and all features.
[group('validation')]
clippy:
    {{ managed }} cargo clippy --workspace --all-targets -- -D warnings -W clippy::pedantic -W clippy::nursery -W clippy::cargo
    {{ managed }} cargo clippy --workspace --all-targets --all-features -- -D warnings -W clippy::pedantic -W clippy::nursery -W clippy::cargo

# Coverage analysis for local development (HTML output)
[group('tests and coverage')]
coverage: _ensure-cargo-llvm-cov
    mkdir -p target/llvm-cov
    {{ managed }} cargo llvm-cov {{ _coverage_base_args }} --html --output-dir target/llvm-cov
    @echo "📊 Coverage report generated: target/llvm-cov/html/index.html"

# Coverage analysis for CI (XML output for codecov/codacy)
[group('tests and coverage')]
coverage-ci: _ensure-cargo-llvm-cov
    mkdir -p coverage
    {{ managed }} cargo llvm-cov nextest {{ _coverage_base_args }} --cobertura --output-path coverage/cobertura.xml -P coverage

# Run the large-scale 2D diagnostic fixture with progress output.
[group('diagnostics')]
debug-large-scale-2d n="36000" repair_every="1": _ensure-nextest
    DELAUNAY_BULK_PROGRESS_EVERY=2000 DELAUNAY_LARGE_DEBUG_MAX_RUNTIME_SECS=1800 DELAUNAY_LARGE_DEBUG_N_2D={{ n }} DELAUNAY_LARGE_DEBUG_REPAIR_EVERY={{ repair_every }} {{ managed }} cargo nextest run --release --profile slow --features slow-tests --test large_scale_debug debug_large_scale_2d -- --exact --nocapture

# Run the large-scale 3D diagnostic fixture with progress output.
[group('diagnostics')]
debug-large-scale-3d n="7500" repair_every="1": _ensure-nextest
    DELAUNAY_BULK_PROGRESS_EVERY=500 DELAUNAY_LARGE_DEBUG_MAX_RUNTIME_SECS=1800 DELAUNAY_LARGE_DEBUG_N_3D={{ n }} DELAUNAY_LARGE_DEBUG_REPAIR_EVERY={{ repair_every }} {{ managed }} cargo nextest run --release --profile slow --features slow-tests --test large_scale_debug debug_large_scale_3d -- --exact --nocapture

# Run the large-scale 4D diagnostic fixture with progress output.
[group('diagnostics')]
debug-large-scale-4d n="800" repair_every="1": _ensure-nextest
    DELAUNAY_BULK_PROGRESS_EVERY=100 DELAUNAY_LARGE_DEBUG_MAX_RUNTIME_SECS=1800 DELAUNAY_LARGE_DEBUG_N_4D={{ n }} DELAUNAY_LARGE_DEBUG_REPAIR_EVERY={{ repair_every }} {{ managed }} cargo nextest run --release --profile slow --features slow-tests --test large_scale_debug debug_large_scale_4d -- --exact --nocapture

# Run the large-scale 5D diagnostic fixture with progress output.
[group('diagnostics')]
debug-large-scale-5d n="140" repair_every="1": _ensure-nextest
    DELAUNAY_BULK_PROGRESS_EVERY=20 DELAUNAY_LARGE_DEBUG_MAX_RUNTIME_SECS=1800 DELAUNAY_LARGE_DEBUG_N_5D={{ n }} DELAUNAY_LARGE_DEBUG_REPAIR_EVERY={{ repair_every }} {{ managed }} cargo nextest run --release --profile slow --features slow-tests --test large_scale_debug debug_large_scale_5d -- --exact --nocapture

# Show the curated workflow guide when `just` is invoked without a recipe.
[default]
[private]
default: help-workflows

# Build rustdoc for the workspace and reject warnings.
[group('validation')]
doc-check:
    RUSTDOCFLAGS='-D warnings' {{ managed }} cargo doc --workspace --no-deps --document-private-items

# Check release-version references against Cargo.toml.
[group('validation')]
docs-version-check:
    {{ rrt }} release check

# Build and run every Rust example.
[group('tests and coverage')]
examples: _ensure-uv
    {{ managed }} research-repo-tools validation cargo-examples tooling/examples.toml

# Fix (mutating): apply formatters/auto-fixes
[group('workflows')]
fix: justfile-fmt toml-fix fmt python-fix shell-fix markdown-fix yaml-fix
    @echo "✅ Fixes applied!"

# Format Rust source files.
[group('validation')]
fmt:
    {{ managed }} cargo fmt --all

# Check Rust source formatting without modifying files.
[group('validation')]
fmt-check:
    {{ managed }} cargo fmt --all -- --check

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
    @echo "  just security           # Network dependency audit and full-history secret scan"
    @echo ""
    @echo "Local CodeRabbit review:"
    @echo "  just review [base]      # Review branch and local edits against verified live origin/main; explicit local bases skip verification"
    @echo "  just review-uncommitted # Review only local edits, including new files"
    @echo ""
    @echo "Setup and maintenance:"
    @echo "  just setup              # Install pinned tools and build the development profile"
    @echo "  just setup-tools        # Install and verify pinned repository tools"
    @echo "  just update             # Update dependency requirements, locks, and repo-owned Cargo tools"
    @echo ""
    @echo "Focused checks:"
    @echo "  just check-code         # Rust, Python, notebooks, and shell"
    @echo "  just check-config       # Just, Cargo, JSON, TOML, YAML/CFF, and Actions"
    @echo "  just check-docs         # Markdown, spelling, and version references"
    @echo "  just rust-core-check    # Rust fmt, Clippy, rustdoc, and Semgrep"
    @echo "  just python-check       # Ruff formatting/lint plus ty type checking"
    @echo "  just notebook-check     # Notebook hygiene and native notebook code checks"
    @echo "  just shell-check        # ShellCheck plus shfmt verification"
    @echo ""
    @echo "Focused tests:"
    @echo "  just test-rust          # Unit, integration, CLI, and doctests"
    @echo "  just test-unit          # Debug and release Rust lib unit tests"
    @echo "  just test-integration   # Release integration tests, including proptests"
    @echo "  just test-integration-fast # Integration tests without proptests"
    @echo "  just test-cli           # CLI-feature binary unit and integration tests"
    @echo "  just test-doc           # Release doctests"
    @echo "  just test-python        # Python support-script tests"
    @echo "  just test-slow          # Explicit slow correctness bucket"
    @echo "  just examples           # Build and run every Rust example"
    @echo ""
    @echo "Notebooks and papers:"
    @echo "  just notebook           # Launch JupyterLab with repository scratch caches"
    @echo "  just notebook-execute   # Execute one notebook under target/notebooks"
    @echo "  just validation-doc-figures # Refresh canonical validation figures"
    @echo "  just validation-doc-figures-check # Verify canonical figures without publishing"
    @echo "  just paper-check        # Lint, build, and check without tracked changes"
    @echo "  just paper-artifact-check # Compare the build with the reviewer PDF"
    @echo "  just paper-refresh      # Check, then refresh one tracked reviewer PDF"
    @echo "  just papers             # Refresh figures and the validation reviewer PDF"
    @echo ""
    @echo "Canonical performance workflows:"
    @echo "  just performance-local  # Measure current tree vs latest release; retain bundle"
    @echo "  just performance-release # Measure, retain, validate, and promote release docs"
    @echo "  just performance-readme # Publish retained release data to README assets/table"
    @echo "  just performance-doc    # Promote docs from retained comparison JSON/evidence; no benchmarks"
    @echo "  just performance-github-assets # Compare stored release assets; retain bundle"
    @echo ""
    @echo "Delaunay-specific performance checks (perf-*):"
    @echo "  just perf-large-scale-smoke # Bounded 2D-5D wall-clock guard"
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
    @echo "  just release-version-check # Require final changelog/citation synchronization"
    @echo "  just update-version <tag>        # Synchronize release metadata using the current UTC date"
    @echo "  just ci-slow            # Default CI plus slow correctness tests"
    @echo "  just coverage           # Generate local HTML coverage"
    @echo ""
    @echo "Use 'just --list' for the complete grouped recipe reference."

# Check JSON files parse cleanly.
[group('validation')]
json-check: _ensure-jq
    {{ rrt }} files run --include '*.json' --batch-size 1 -- jq empty

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
    {{ rrt }} files check-lines
    {{ managed }} research-repo-tools files run --include '*.md' --exclude CHANGELOG.md --exclude 'docs/archive/**' --exclude 'docs/archives/changelog/**' -- rumdl check

# Apply automatic Markdown fixes.
[group('validation')]
markdown-fix: _ensure-rumdl
    {{ managed }} research-repo-tools files run --include '*.md' --exclude CHANGELOG.md --exclude 'docs/archive/**' --exclude 'docs/archives/changelog/**' -- rumdl check --fix

# Launch configured JupyterLab with private session caches.
[group('notebooks and papers')]
[positional-arguments]
notebook *args: _ensure-uv
    uv run --locked --group dev --group notebooks research-repo-tools notebooks launch "$@"

# Run routine non-executing notebook validation.
[group('notebooks and papers')]
notebook-check: _ensure-uv
    uv run --locked --group dev --group notebooks research-repo-tools files run --include 'notebooks/*.ipynb' --exclude '**/.ipynb_checkpoints/**' -- uv run --locked --group dev --group notebooks research-repo-tools notebooks lint
    uv run --locked --group dev --group notebooks research-repo-tools files run --include 'notebooks/*.ipynb' --exclude '**/.ipynb_checkpoints/**' -- uv run --locked --group dev --group notebooks research-repo-tools notebooks advise

# Clear outputs from one source notebook in place.
[group('notebooks and papers')]
notebook-clear-outputs notebook="notebooks/00_quickstart.ipynb": _ensure-uv
    uv run --locked --group dev --group notebooks research-repo-tools notebooks clear {{ quote(notebook) }}

# Clear outputs from every source notebook in place.
[group('notebooks and papers')]
notebook-clear-outputs-all: _ensure-uv
    uv run --locked --group dev --group notebooks research-repo-tools files run --include 'notebooks/*.ipynb' --exclude '**/.ipynb_checkpoints/**' -- uv run --locked --group dev --group notebooks research-repo-tools notebooks clear

# Execute one notebook into target/notebooks without modifying its source.
[group('notebooks and papers')]
notebook-execute notebook="notebooks/00_quickstart.ipynb" output_dir="target/notebooks" timeout="600": _ensure-uv
    uv run --locked --group dev --group notebooks research-repo-tools notebooks execute {{ quote(notebook) }} --output-dir {{ quote(output_dir) }} --timeout {{ quote(timeout) }}

# Check notebook structure and output hygiene without code linting.
[group('notebooks and papers')]
notebook-output-check: _ensure-uv
    uv run --locked --group dev --group notebooks research-repo-tools files run --include 'notebooks/*.ipynb' --exclude '**/.ipynb_checkpoints/**' -- uv run --locked --group dev --group notebooks research-repo-tools notebooks check

# Restore tracked source notebooks and remove generated notebook artifacts.
[group('notebooks and papers')]
notebook-reset-from-git source="index":
    #!/usr/bin/env bash
    set -euo pipefail
    source={{ quote(source) }}
    if [[ "$source" == index ]]; then
        {{ rrt }} notebooks reset --apply
    else
        {{ rrt }} notebooks reset --revision "$source" --apply
    fi

# Install the optional notebook dependency group.
[group('notebooks and papers')]
notebook-setup: _ensure-uv
    uv run --locked --group dev --group notebooks research-repo-tools notebooks sync

# Run one 3D and one 4D direct Pachner stress workload with topology-scope reports enabled.
[group('benchmarks and performance')]
pachner-stress attempts="100" validate_every="10" mode="round-trip": (_pachner-stress-dim "3d" "9000" attempts validate_every "target/pachner_stress/3d" mode) (_pachner-stress-dim "4d" "1000" attempts validate_every "target/pachner_stress/4d" mode)

# Run one 3D direct Pachner stress workload with topology-scope reports enabled.
[group('benchmarks and performance')]
pachner-stress-3d attempts="100" vertices="9000" validate_every="10" output_dir="target/pachner_stress/3d" mode="round-trip": (_pachner-stress-dim "3d" vertices attempts validate_every output_dir mode)

# Run one 4D direct Pachner stress workload with topology-scope reports enabled.
[group('benchmarks and performance')]
pachner-stress-4d attempts="100" vertices="1000" validate_every="10" output_dir="target/pachner_stress/4d" mode="round-trip": (_pachner-stress-dim "4d" vertices attempts validate_every output_dir mode)

# Check that one target-built paper is structurally equivalent to its tracked reviewer PDF.
[group('notebooks and papers')]
paper-artifact-check paper="validation": (paper-check paper)
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
    {{ rrt }} papers check --paper "$paper" \
        --reference "papers/${paper}.pdf"

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
    source_date_epoch="$({{ rrt }} papers source-date --paper "$paper")"
    export SOURCE_DATE_EPOCH="$source_date_epoch"
    {{ managed }} tectonic --keep-intermediates --keep-logs --outdir "$build_dir" "$paper_source"
    {{ rrt }} papers normalize --paper "$paper"
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
    {{ managed }} cargo build --locked --profile perf --features cli --bin delaunay

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
    {{ rrt }} papers check --paper "$paper"

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
    reviewer_pdf="papers/${paper}.pdf"
    {{ rrt }} papers normalize --paper "$paper" --output "$reviewer_pdf"
    echo "📄 Reviewer PDF refreshed: $reviewer_pdf"

# Format publication-facing TeX sources.
[group('notebooks and papers')]
paper-tex-fmt: _ensure-tex-fmt
    {{ managed }} tex-fmt papers/*.tex

# Check publication-facing TeX formatting without modifying files.
[group('notebooks and papers')]
paper-tex-fmt-check: _ensure-tex-fmt
    {{ managed }} tex-fmt --check papers/*.tex

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

# Show detailed performance-check, benchmark, and profiling workflows.
[group('benchmarks and performance')]
perf-help:
    @echo "Performance Analysis Commands:"
    @echo "  just bench-latest          # Run curated release-signal Criterion benchmarks"
    @echo "  just bench-latest-vs-last  # Run latest and compare against saved 'last'"
    @echo "  just bench-compare [base] [suite] [scope] # Render a report from saved Criterion baselines"
    @echo "  just bench-save-baseline <tag> [suite] # Save a named Criterion baseline"
    @echo "  just bench-save-baseline last # Save release-signal Criterion baseline as 'last'"
    @echo "  just performance-local    # Compare current tree against latest release locally"
    @echo "  just performance-github-assets # Compare stored GitHub Release benchmark assets"
    @echo "  just performance-release  # Measure, retain, and promote release performance docs"
    @echo "  just performance-doc      # Promote docs from retained comparison JSON/evidence without benchmarks"
    @echo "  just performance-readme   # Publish retained release data to README assets/table"
    @echo "  just perf-large-scale-smoke # Quick pre-push 2D-5D wall-clock smoke guard"
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
    @echo "  just bench                 # Full benchmark suite with perf profile"
    @echo "  just bench-ci              # CI benchmark suite with perf profile"
    @echo "  just bench-allocations     # Allocation-contract microbenchmarks"
    @echo "  just pachner-stress        # 3D+4D direct Pachner CLI stress with comparison JSON/evidence artifacts"
    @echo "  just pachner-stress-3d     # 3D Pachner CLI stress (100 steps, 9K vertices)"
    @echo "  just pachner-stress-4d     # 4D Pachner CLI stress (100 steps, 1K vertices)"
    @echo "  just bench-pachner-stress  # Criterion timing for Pachner move/round-trip stress"
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
    @echo "                              # Generate scratch main baseline without overwriting baseline-artifact"
    @echo "  CRIT_SAMPLE_SIZE=100 just bench  # Custom sample size"
    @echo "  just pachner-stress-4d 100000 1000 1000 target/pachner_stress/4d random-walk"
    @echo "                              # 4D random-walk Pachner diagnostics with comparison JSON/evidence artifacts"
    @echo "  just bench-ci              # Final optimized CI-suite benchmark run"
    @echo "  just profile v0.7.5        # v0.7.5 code on its declared Rust toolchain"
    @echo "  just profile 1.99.0        # Current tree on Rust 1.99.0"
    @echo "  just profile 1.99.0 v0.7.5 # v0.7.5 code on Rust 1.99.0"

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
        # Level 4 retains bounded full-scope regression coverage under
        # `just test-slow`; broader Level 4/5 work remains in #482/#483.
        if env \
            DELAUNAY_BULK_PROGRESS_EVERY="$progress_every" \
            DELAUNAY_LARGE_DEBUG_MAX_RUNTIME_SECS="$max_secs" \
            "$n_env=$n_points" \
            DELAUNAY_LARGE_DEBUG_REPAIR_EVERY=1 \
            DELAUNAY_LARGE_DEBUG_VALIDATION=construction \
            {{ managed }} cargo nextest run --release --profile slow --features slow-tests --test large_scale_debug "$test_name" -- --exact --nocapture 2>&1 | tee "$log_file"; then
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

# Promote performance documentation solely from the retained canonical artifacts.
[group('benchmarks and performance')]
performance-doc: _ensure-uv
    {{ managed }} uv run --locked benchmark-utils performance-doc

# Compare stored GitHub Release benchmark assets without local Cargo runs.
[group('benchmarks and performance')]
performance-github-assets current_tag="" baseline_tag="": _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    current_tag={{ quote(current_tag) }}
    baseline_tag={{ quote(baseline_tag) }}
    tag_pair_state="$(just --quiet _performance-tag-pair-state "$current_tag" "$baseline_tag")"
    if [[ "$tag_pair_state" == "invalid" ]]; then
        exit 2
    fi
    if [[ "$tag_pair_state" == "explicit" ]]; then
        {{ managed }} uv run --locked benchmark-utils performance-github-assets "$current_tag" "$baseline_tag"
    else
        {{ managed }} uv run --locked benchmark-utils performance-github-assets
    fi

# Compare the current tree against the latest stable release and retain a bundle.
[group('benchmarks and performance')]
performance-local: _ensure-uv
    {{ managed }} uv run --locked benchmark-utils performance-local

# Validate retained release measurements and atomically publish README assets/table.
[group('benchmarks and performance')]
performance-readme: _ensure-uv
    uv run --locked publish-readme-performance

# Measure, retain, reload-validate, and promote release performance documentation.
[group('benchmarks and performance')]
performance-release current_tag="" baseline_tag="": _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    current_tag={{ quote(current_tag) }}
    baseline_tag={{ quote(baseline_tag) }}
    tag_pair_state="$(just --quiet _performance-tag-pair-state "$current_tag" "$baseline_tag")"
    if [[ "$tag_pair_state" == "invalid" ]]; then
        exit 2
    fi
    if [[ "$tag_pair_state" == "explicit" ]]; then
        {{ managed }} uv run --locked benchmark-utils performance-release "$current_tag" "$baseline_tag"
    else
        {{ managed }} uv run --locked benchmark-utils performance-release
    fi

# Run the selected CI benchmark suite for one compiler/code pair.
[group('benchmarks and performance')]
profile toolchain="" code_ref="current": _ensure-jq _ensure-toolchain
    #!/usr/bin/env bash
    set -euo pipefail

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

    {{ managed }} rustup toolchain install "$requested_toolchain" --profile minimal

    {
        echo "# Profile Run"
        echo
        echo "- Code ref: $requested_ref"
        echo "- Workdir: $workdir"
        echo "- Commit: $(git -C "$workdir" rev-parse HEAD)"
        echo "- Dirty tree: $(if [[ "$workdir" == "$repo_root" && -n "$(git status --short)" ]]; then echo yes; else echo no; fi)"
        echo "- Requested toolchain: $requested_toolchain"
        echo "- rustc: $({{ managed }} rustup run "$requested_toolchain" rustc --version)"
        echo "- cargo: $(rustup run "$requested_toolchain" cargo --version)"
        echo "- Cargo profile: cargo bench --profile perf"
        echo "- Benchmark harness: ci_performance_suite"
    } > "$run_dir/profile_metadata.md"

    (
        cd "$workdir"
        CARGO_TARGET_DIR="$run_dir/target" \
            {{ managed }} rustup run "$requested_toolchain" cargo bench --profile perf --bench ci_performance_suite \
            2>&1 | tee "$run_dir/ci_performance_suite.log"
    )

# Profile 3D construction with Samply in the development configuration.
[group('benchmarks and performance')]
profile-dev: _ensure-samply
    PROFILING_DEV_MODE=1 {{ managed }} samply record {{ managed }} cargo bench --profile perf --bench profiling_suite -- "construction/3D/5000v/construct"

# Profile allocation-heavy construction with Samply.
[group('benchmarks and performance')]
profile-mem: _ensure-samply
    {{ managed }} samply record cargo bench --profile perf --bench profiling_suite --features count-allocations -- memory_profiling

# Pre-publish validation: checks crates.io metadata rules that cargo publish --dry-run does NOT catch
# Validate crates.io metadata and run cargo publish --dry-run.
[group('release')]
publish-check: _ensure-toolchain
    {{ managed }} research-repo-tools validation cargo-metadata --package delaunay
    {{ managed }} cargo publish --locked --allow-dirty --dry-run

# Run every non-mutating Python source check.
[group('validation')]
python-check: _ensure-uv
    uv run --locked --group dev --group notebooks research-repo-tools python check
    @echo "✅ Python source checks complete!"

# Apply Ruff lint fixes and formatting to Python source.
[group('validation')]
python-fix: _ensure-uv
    {{ rrt }} python fix

# Lint deliberate Python fixtures with the full configured Ruff policy.
[group('validation')]
python-fixture-lint: _ensure-uv
    uv run --locked ruff check tests/semgrep/

# Check Python formatting with Ruff.
[group('validation')]
python-format-check: _ensure-uv
    {{ rrt }} files run --include '*.py' --include '*.pyi' -- uv run --locked ruff format --check --

# Lint Python source with Ruff.
[group('validation')]
python-lint: _ensure-uv
    {{ rrt }} files run --include '*.py' --include '*.pyi' -- uv run --locked ruff check --

# Synchronize development Python dependencies from the lockfile.
[group('build and setup')]
python-sync: _ensure-uv
    uv sync --locked --group dev

# Type-check Python support code with ty.
[group('validation')]
python-typecheck: _ensure-uv
    uv run --locked --group dev --group notebooks research-repo-tools python typecheck

# Print retained release notes for a tag.
[group('release')]
release-notes tag:
    {{ rrt }} changelog notes {{ quote(tag) }}

# Require matching final changelog and citation dates.
[group('release')]
release-version-check:
    {{ rrt }} release check --final-release

# Review committed and local changes against the PR base with CodeRabbit.
[group('review')]
review base="origin/main":
    {{ rrt }} review branch --base={{ quote(base) }}

# Review staged, unstaged, and new files without committed branch changes.
[group('review')]
review-uncommitted:
    {{ rrt }} review uncommitted

# Run the opt-in companion binary with the CLI feature and perf profile.
[group('build and setup')]
run *args:
    {{ managed }} cargo run --locked --profile perf --features cli --bin delaunay -- {{ args }}

# Run the complete non-mutating Rust validation surface.
[group('validation')]
rust-core-check: fmt-check clippy doc-check semgrep semgrep-test
    @echo "✅ Rust core checks complete!"

# Run the network-dependent vulnerability and full-history secret gates.
[group('security')]
security: audit security-secrets

# Scan reachable Git history and current files with redacted reports.
[group('security')]
security-secrets:
    {{ rrt }} security secrets

# Repository-owned Semgrep rules for project-specific Rust diagnostics.
[group('validation')]
semgrep: semgrep-scan

# Run the shared repository Semgrep target set, optionally emitting SARIF.
[private]
semgrep-scan output="target/semgrep": _ensure-uv
    {{ rrt }} semgrep scan --include '*.rs' --include 'tooling/python/*.py' --include 'tests/tooling/*.py' \
        --include '.github/workflows/*.yml' --include '.github/workflows/*.yaml' \
        --include '*.md' --include CITATION.cff --include justfile \
        --include 'notebooks/*.ipynb' --include 'papers/*.tex' --include 'doctests/*.txt' \
        --exclude 'tests/semgrep/**' --output {{ quote(output) }}

# Test the repository-owned Semgrep rules against their fixtures.
[group('validation')]
semgrep-test: _ensure-uv
    {{ rrt }} semgrep check-fixtures

# Install required tools and build the development profile.
[group('build and setup')]
setup: setup-tools build
    echo "✅ Setup complete! Run 'just help-workflows' to see available commands."

# Install and verify repository development tools.
[doc('Install and verify repository development tools.')]
[group('build and setup')]
setup-tools:
    #!/usr/bin/env bash
    set -euo pipefail
    tectonic_environment="$(uv run --locked --managed-python --only-group tooling research-repo-tools tectonic discover --format shell)"
    eval "$tectonic_environment"
    uv run --locked --managed-python --only-group tooling research-repo-tools setup

# Run ShellCheck and verify shfmt formatting.
[group('validation')]
shell-check: shell-lint shell-fmt-check
    @echo "✅ Shell checks complete!"

# Format tracked and new shell scripts with shfmt.
[group('validation')]
shell-fix: _ensure-shfmt
    {{ rrt }} files run --include '*.sh' -- uv run --locked shfmt -w

# Check tracked and new shell-script formatting with shfmt.
[group('validation')]
shell-fmt-check: _ensure-shfmt
    {{ rrt }} files run --include '*.sh' -- uv run --locked shfmt -d

# Lint tracked and new shell scripts with ShellCheck.
[group('validation')]
shell-lint: _ensure-shellcheck
    {{ rrt }} files run --include '*.sh' --batch-size 4 -- uv run --locked shellcheck -x

# Spell check (typos)
[group('validation')]
spell-check: _ensure-typos
    {{ managed }} research-repo-tools files run --exclude typos.toml -- typos --config typos.toml --force-exclude --exclude typos.toml --

# Deliberately refresh the slow notebook-backed spherical README hero.
[group('notebooks and papers')]
spherical-readme-hero: _ensure-uv paper-cli
    #!/usr/bin/env bash
    set -euo pipefail
    DELAUNAY_SPHERICAL_HERO_FIGURE="docs/assets/readme/delaunay_spherical_readme.png" just notebook-execute notebooks/02_spherical_hero.ipynb target/docs/notebooks 1800

# Create an annotated git tag from the CHANGELOG.md section for the given version
[group('release')]
tag version: release-version-check
    {{ rrt }} changelog tag {{ quote(version) }}

# Replace an existing annotated tag from the CHANGELOG.md section.
[group('release')]
tag-force version: release-version-check
    {{ rrt }} changelog tag {{ quote(version) }} --force

# Run every default Rust and Python test bucket once.
[group('workflows')]
test: test-rust test-python
    @echo "✅ Test workflow passed!"

# Run public allocation-contract integration tests.
[group('tests and coverage')]
test-allocation: _ensure-nextest
    {{ managed }} cargo nextest run --profile ci --test allocation_api --features count-allocations -- --nocapture

# Run CLI-feature binary unit and integration tests in the release profile.
[group('tests and coverage')]
test-cli: _ensure-nextest
    {{ managed }} cargo nextest run --release --profile ci --features cli --bin delaunay --bin pachner-stress --test cli

# Run diagnostics-feature integration tests with captured output.
[group('diagnostics')]
test-diagnostics: _ensure-nextest
    {{ managed }} cargo nextest run --profile ci --test circumsphere_debug_tools --features diagnostics -- --nocapture

# Run Rust doctests in the release profile.
[group('tests and coverage')]
test-doc:
    {{ managed }} cargo test --doc --release --verbose

# Run default integration tests in release mode under the normal 10-second budget.
# Narrow test-specific overrides grant headroom to boundary-running cases.
[group('tests and coverage')]
test-integration: _ensure-nextest
    {{ managed }} cargo nextest run --release --profile ci --test '*'

# Compile release integration tests without running them.
[group('tests and coverage')]
test-integration-compile: _ensure-nextest
    {{ managed }} cargo nextest run --release --test '*' --no-run

# test-integration-fast: runs integration tests but skips proptests (tests prefixed with `prop_`)
#
# Useful for quick local validation on changes that don't touch the property-test surface area.
# To run the full (slow) property suite, use: just test-integration
#
# Note: `--skip prop_` is a substring filter applied by the Rust test harness.
[doc('Run release integration tests while skipping property tests.')]
[group('tests and coverage')]
test-integration-fast: _ensure-nextest
    {{ managed }} cargo nextest run --release --profile ci --test '*' -- --skip prop_

# Run Python support-script tests with pytest.
[group('tests and coverage')]
test-python: _ensure-uv
    uv run --locked pytest

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
    {{ managed }} cargo nextest run --release --profile slow --features slow-tests
    {{ managed }} cargo test --doc --release --features slow-tests

# Run Rust lib unit tests in debug and release profiles.
[group('tests and coverage')]
test-unit: _ensure-nextest
    {{ managed }} cargo nextest run --profile debug --lib
    {{ managed }} cargo nextest run --release --profile ci --lib

# Run TOML parsing, lint, and formatting checks.
[group('validation')]
toml-check: toml-parse-check toml-lint toml-fmt-check
    @echo "✅ TOML checks complete!"

# Format tracked TOML files with Taplo.
[group('validation')]
toml-fix: _ensure-taplo
    {{ managed }} research-repo-tools files run --include '*.toml' -- taplo fmt

# Check tracked TOML formatting with Taplo.
[group('validation')]
toml-fmt-check: _ensure-taplo
    {{ managed }} research-repo-tools files run --include '*.toml' -- taplo fmt --check

# Lint tracked TOML files with Taplo.
[group('validation')]
toml-lint: _ensure-taplo
    {{ managed }} research-repo-tools files run --include '*.toml' -- taplo lint

# Check that tracked TOML files parse cleanly.
[group('validation')]
toml-parse-check: _ensure-uv
    {{ rrt }} files run --include '*.toml' --batch-size 1 -- uv run --locked python -c 'import pathlib, sys, tomllib; tomllib.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))'

# Inspect declared tools without installing or synchronizing dependencies.
[group('build and setup')]
tools-check:
    uv run --locked --no-sync --no-python-downloads research-repo-tools toolchain check

# Export verified managed paths for GitHub Actions.
[group('build and setup')]
tools-export:
    uv run --locked --no-sync --no-python-downloads research-repo-tools toolchain export

# Check for unused direct Cargo dependencies.
[group('validation')]
unused-deps: _ensure-cargo-machete
    {{ managed }} cargo machete

# Update dependency requirements, locks, managed Cargo tools, and the active uv pin.
[group('build and setup')]
update: update-tools update-dependencies
    @echo "✅ Repository dependencies and tools updated."

# Advance Cargo dependency declarations and lockfile entries for every resolution root.
[doc('Update repository Cargo dependency requirements and lockfiles.')]
[group('build and setup')]
update-cargo-dependencies:
    uv run --locked --only-group tooling --inexact research-repo-tools toolchain run -- cargo upgrade --incompatible allow
    uv run --locked --only-group tooling --inexact research-repo-tools toolchain run -- cargo upgrade --manifest-path tests/fixtures/checkpoint_no_float_roundtrip/Cargo.toml --incompatible allow
    uv run --locked --only-group tooling --inexact research-repo-tools toolchain run -- cargo update
    uv run --locked --only-group tooling --inexact research-repo-tools toolchain run -- cargo update --manifest-path tests/fixtures/checkpoint_no_float_roundtrip/Cargo.toml

# Upgrade declared Cargo tools and publish verified exact TOML pins.
[doc('Upgrade only declared managed Cargo tools.')]
[group('build and setup')]
update-cargo-tools:
    uv run --locked --only-group tooling --inexact research-repo-tools toolchain upgrade

# Advance Cargo and exact Python development requirements plus their lockfiles.
[doc('Update Cargo and Python development requirements plus all Cargo/uv locked dependencies.')]
[group('build and setup')]
update-dependencies: update-cargo-dependencies update-python-dependencies

# Resolve latest exact Python development tools, retain ranged requirements, and sync.
[doc('Update exact dependency-groups.dev pins and uv.lock through uv.')]
[group('build and setup')]
update-python-dependencies:
    uv run --locked --only-group tooling --inexact research-repo-tools deps update-python
    uv lock --upgrade
    uv run --locked --no-sync --no-python-downloads research-repo-tools toolchain run -- uv sync --locked --managed-python --group dev

alias update-python-deps := update-python-dependencies

# Upgrade uv and managed Cargo tools, then synchronize the declared environment.
[group('build and setup')]
update-tools: update-uv update-cargo-tools setup-tools

# Upgrade uv through its installation owner and reconcile the project declaration.
[group('build and setup')]
update-uv:
    uv run --no-config --no-sync --no-python-downloads research-repo-tools deps update-uv

# Update release metadata; optional date and predecessor arguments support offline preparation.
[doc('Update package, citation, lockfile, and non-artifact documentation release versions.')]
[group('release')]
[positional-arguments]
update-version tag *args: _ensure-gh
    {{ rrt }} release update "$@"
    {{ managed }} cargo metadata --locked --format-version 1 --no-deps > /dev/null
    {{ rrt }} release check

# Refresh reviewer-facing validation diagrams from the reproducible notebook.
[group('notebooks and papers')]
validation-doc-figures: _ensure-uv paper-cli
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p docs/assets/validation
    DELAUNAY_BINARY="{{ perf_delaunay_binary }}" DELAUNAY_VALIDATION_DOC_FIGURE_DIR="docs/assets/validation" just notebook-execute notebooks/01_validation.ipynb target/docs/notebooks

# Regenerate validation diagrams under target/ and compare them with tracked artifacts.
[group('notebooks and papers')]
validation-doc-figures-check: _ensure-uv paper-cli
    #!/usr/bin/env bash
    set -euo pipefail
    check_root="target/docs/validation-figure-check"
    generated_dir="target/notebooks/01_validation/validation_figures"
    rm -rf "$check_root"
    DELAUNAY_BINARY="{{ perf_delaunay_binary }}" just notebook-execute notebooks/01_validation.ipynb "$check_root/notebook"
    uv run --locked --group notebooks python -m notebook_validation_rendering "$generated_dir" docs/assets/validation

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
    {{ managed }} research-repo-tools files run --include '*.yml' --include '*.yaml' --include CITATION.cff -- dprint fmt --incremental=false

# Check tracked YAML/CFF formatting with dprint.
[group('validation')]
yaml-fmt-check: _ensure-dprint
    {{ managed }} research-repo-tools files run --include '*.yml' --include '*.yaml' --include CITATION.cff -- dprint check --incremental=false

# Lint tracked YAML/CFF files with yamllint.
[group('validation')]
yaml-lint: _ensure-yamllint
    {{ rrt }} files run --include '*.yml' --include '*.yaml' --include CITATION.cff -- uv run --locked yamllint --strict -c .yamllint

# Audit GitHub Actions workflows with zizmor.
[group('validation')]
zizmor:
    {{ rrt }} zizmor check
