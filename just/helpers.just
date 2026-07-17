[private]
_bench-pachner-stress samples:
    #!/usr/bin/env bash
    set -euo pipefail

    samples="{{ samples }}"

    require_positive_integer() {
        local name="$1"
        local value="$2"
        if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: $name must be a positive integer, got: $value" >&2
            exit 2
        fi
    }

    require_positive_integer "samples" "$samples"

    if (( samples < 10 )); then
        echo "ERROR: samples must be at least 10 because Criterion requires sample_size >= 10." >&2
        exit 2
    fi

    echo "Pachner stress Criterion fixture benchmark: ${samples} samples."
    env CRIT_SAMPLE_SIZE="$samples" \
        cargo bench --locked --profile perf --features bench --bench pachner_stress -- --noplot

_ensure-actionlint: _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    uv run actionlint -version >/dev/null

_ensure-cargo-llvm-cov: (_ensure-pinned-cargo-tool "cargo-llvm-cov" "cargo-llvm-cov" cargo_llvm_cov_version "llvm-cov")

_ensure-cargo-machete: (_ensure-pinned-cargo-tool "cargo-machete" "cargo-machete" cargo_machete_version)

_ensure-chktex:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v chktex >/dev/null || {
        echo "❌ 'chktex' not found. Install a TeX distribution or package manager copy of chktex and ensure it is on PATH."
        exit 1
    }

_ensure-dprint: (_ensure-pinned-cargo-tool "dprint" "dprint" dprint_version)

_ensure-git-cliff: (_ensure-pinned-cargo-tool "git-cliff" "git-cliff" git_cliff_version)

_ensure-jq:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v jq >/dev/null || { echo "❌ 'jq' not found. Install jq and ensure it is on PATH."; exit 1; }

_ensure-nextest: (_ensure-pinned-cargo-tool "cargo-nextest" "cargo-nextest" nextest_version)

_ensure-pinned-cargo-tool binary package version cargo_subcommand="":
    #!/usr/bin/env bash
    set -euo pipefail

    source scripts/cargo_tool_versions.sh

    binary="{{ binary }}"
    package="{{ package }}"
    version="{{ version }}"
    cargo_subcommand="{{ cargo_subcommand }}"

    if ! cargo_tool_has_exact_version "$binary" "$version" "$cargo_subcommand"; then
        echo "❌ '$binary' $version not found. See 'just setup-tools' or install:"
        echo "   cargo install --locked $package --version $version"
        exit 1
    fi

_ensure-rumdl: (_ensure-pinned-cargo-tool "rumdl" "rumdl" rumdl_version)

_ensure-samply: (_ensure-pinned-cargo-tool "samply" "samply" samply_version)

_ensure-shellcheck: _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    uv run shellcheck --version >/dev/null

_ensure-shfmt: _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    uv run shfmt --version >/dev/null

_ensure-taplo: (_ensure-pinned-cargo-tool "taplo" "taplo-cli" taplo_version)

_ensure-tectonic: (_ensure-pinned-cargo-tool "tectonic" "tectonic" tectonic_version)

_ensure-tex-fmt: (_ensure-pinned-cargo-tool "tex-fmt" "tex-fmt" tex_fmt_version)

_ensure-typos: (_ensure-pinned-cargo-tool "typos" "typos-cli" typos_version)

_ensure-uv:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if command -v uv >/dev/null; then
        installed_version="$(uv --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
    fi
    if [[ "$installed_version" != "{{ uv_version }}" ]]; then
        echo "❌ 'uv' {{ uv_version }} not found (found: ${installed_version:-missing})."
        echo "   Install the pinned external prerequisite from https://github.com/astral-sh/uv"
        exit 1
    fi

_ensure-yamllint: _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    uv run yamllint --version >/dev/null

_ensure-zizmor: (_ensure-pinned-cargo-tool "zizmor" "zizmor" zizmor_version)

[private]
_pachner-stress-dim label vertices attempts validate_every output_dir mode:
    #!/usr/bin/env bash
    set -euo pipefail

    label="{{ label }}"
    vertices="{{ vertices }}"
    attempts="{{ attempts }}"
    validate_every="{{ validate_every }}"
    output_dir="{{ output_dir }}"
    mode="{{ mode }}"

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
    case "$mode" in
        round-trip|random-walk) ;;
        *)
            echo "ERROR: unsupported Pachner stress mode: $mode" >&2
            exit 2
            ;;
    esac

    mkdir -p "$output_dir"

    echo "Pachner stress ${label}/${mode}: ${vertices} vertices, ${attempts} attempted moves."
    echo "Validation scope: topology (Levels 1-3; Level 4 realization overlap scan deferred to #482)."
    cargo run --locked --profile perf --features cli --bin delaunay -- \
        pachner-stress \
        --dimension "$label" \
        --mode "$mode" \
        --vertices "$vertices" \
        --attempts "$attempts" \
        --validate-every "$validate_every" \
        --progress-csv "$output_dir/progress.csv" \
        --summary-json "$output_dir/summary.json"

_performance-tag-pair-state current_tag baseline_tag:
    #!/usr/bin/env bash
    set -euo pipefail
    current_tag="{{ current_tag }}"
    baseline_tag="{{ baseline_tag }}"
    if [[ -n "$current_tag" || -n "$baseline_tag" ]]; then
        if [[ -z "$current_tag" || -z "$baseline_tag" ]]; then
            echo "current_tag and baseline_tag must be provided together" >&2
            echo "invalid"
        else
            echo "explicit"
        fi
    else
        echo "inferred"
    fi
