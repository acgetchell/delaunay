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
        cargo bench --profile perf --features bench --bench pachner_stress -- --noplot

_ensure-actionlint: _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
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

_ensure-chktex:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v chktex >/dev/null || {
        echo "❌ 'chktex' not found. Install a TeX distribution or package manager copy of chktex and ensure it is on PATH."
        exit 1
    }

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

_ensure-samply:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v samply >/dev/null || {
        echo "❌ 'samply' not found. See 'just setup-tools' or install:"
        echo "   cargo install --locked samply"
        exit 1
    }

_ensure-shellcheck: _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    uv run shellcheck --version >/dev/null

_ensure-shfmt: _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    uv run shfmt --version >/dev/null

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
    echo "Validation scope: topology (Levels 1-3; Level 4 embedding overlap scan deferred to #483)."
    cargo run --profile perf --features cli --bin delaunay -- \
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
