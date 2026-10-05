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
        {{ managed }} cargo bench --locked --profile perf --features bench --bench pachner_stress -- --noplot

_ensure-actionlint: _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    uv run --locked actionlint -version >/dev/null

_ensure-cargo-edit: _ensure-toolchain

_ensure-cargo-llvm-cov: _ensure-toolchain

_ensure-cargo-machete: _ensure-toolchain

_ensure-chktex:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v chktex >/dev/null || {
        echo "❌ 'chktex' not found. Install a TeX distribution or package manager copy of chktex and ensure it is on PATH."
        exit 1
    }

_ensure-dprint: _ensure-toolchain

_ensure-gh:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v gh >/dev/null || { echo "❌ 'gh' not found. Install GitHub CLI and ensure it is on PATH."; exit 1; }

_ensure-git-cliff: _ensure-toolchain

_ensure-jq:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v jq >/dev/null || { echo "❌ 'jq' not found. Install jq and ensure it is on PATH."; exit 1; }

_ensure-nextest: _ensure-toolchain

_ensure-rumdl: _ensure-toolchain

_ensure-samply: _ensure-toolchain

_ensure-shellcheck: _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    uv run --locked shellcheck --version >/dev/null

_ensure-shfmt: _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    uv run --locked shfmt --version >/dev/null

_ensure-taplo: _ensure-toolchain

_ensure-tectonic: _ensure-toolchain

_ensure-tex-fmt: _ensure-toolchain

_ensure-toolchain:
    uv run --locked --no-sync --no-python-downloads research-repo-tools toolchain check

_ensure-typos: _ensure-toolchain

_ensure-uv:
    uv run --locked --no-sync --no-python-downloads research-repo-tools --version > /dev/null

_ensure-yamllint: _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    uv run --locked yamllint --version >/dev/null

_ensure-zizmor: _ensure-toolchain

[private]
_pachner-stress-dim label vertices attempts validate_every output_dir mode:
    #!/usr/bin/env bash
    set -euo pipefail

    label={{ quote(label) }}
    vertices={{ quote(vertices) }}
    attempts={{ quote(attempts) }}
    validate_every={{ quote(validate_every) }}
    output_dir={{ quote(output_dir) }}
    mode={{ quote(mode) }}

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

    echo "Pachner stress ${label}/${mode}: ${vertices} vertices, ${attempts} workload steps."
    echo "Validation scope: topology (Levels 1-3; Level 4 realization overlap scan deferred to #482)."
    {{ managed }} cargo run --locked --profile perf --features cli --bin pachner-stress -- \
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
    current_tag={{ quote(current_tag) }}
    baseline_tag={{ quote(baseline_tag) }}
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

# Run a Python tool over tracked sources plus new, unignored sources.
[private]
_python-tool command: _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    source_list="$(mktemp "${TMPDIR:-/tmp}/delaunay-python-sources.XXXXXX")"
    trap 'rm -f "$source_list"' EXIT
    git --no-pager ls-files --cached --others --exclude-standard --deduplicate -z -- '*.py' '*.pyi' > "$source_list"
    python_files=()
    while IFS= read -r -d '' file; do
        # A tracked file may have been deleted in the working tree.
        if [[ -f "$file" ]]; then
            python_files+=("$file")
        fi
    done < "$source_list"
    if [[ ${#python_files[@]} -eq 0 ]]; then
        echo "No Python source files found" >&2
        exit 1
    fi
    uv run --locked {{ command }} -- "${python_files[@]}"

[private]
_validation-doc-figures-check-if-canonical:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ "$(uname -s)" == "Darwin" ]]; then
        just validation-doc-figures-check
    else
        echo "Skipping canonical validation-figure byte comparison outside macOS."
    fi
