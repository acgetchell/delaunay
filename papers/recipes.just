# Paper commands share the root environment and lockfile, with explicit opt-in.
paper_rrt := "uv run --locked --group dev --group papers research-repo-tools --root . --config papers/tooling.toml"
paper_managed := paper_rrt + " toolchain run --"

_ensure-paper-toolchain:
    {{ paper_rrt }} toolchain check

# Check that the target-built paper matches its tracked reviewer PDF.
[group('papers')]
paper-artifact-check paper="validation": (paper-check paper)
    #!/usr/bin/env bash
    set -euo pipefail
    paper={{ quote(paper) }}
    case "$paper" in
        ""|*[!A-Za-z0-9_-]*)
            echo "Invalid paper name: $paper" >&2
            exit 1
            ;;
    esac
    {{ paper_rrt }} papers check --paper "$paper" --reference "papers/${paper}.pdf"

# Compile one paper under target/papers without refreshing tracked files.
[group('papers')]
paper-build paper="validation": _ensure-paper-toolchain
    #!/usr/bin/env bash
    set -euo pipefail
    paper={{ quote(paper) }}
    case "$paper" in
        ""|*[!A-Za-z0-9_-]*)
            echo "Invalid paper name: $paper" >&2
            exit 1
            ;;
    esac
    paper_source="papers/${paper}.tex"
    build_dir="target/papers/${paper}"
    if [[ ! -f "$paper_source" ]]; then
        echo "Paper source not found: $paper_source" >&2
        exit 1
    fi
    rm -rf "$build_dir"
    mkdir -p "$build_dir"
    source_date_epoch="$({{ paper_rrt }} papers source-date --paper "$paper")"
    export SOURCE_DATE_EPOCH="$source_date_epoch"
    {{ paper_managed }} tectonic --keep-intermediates --keep-logs --outdir "$build_dir" "$paper_source"
    {{ paper_rrt }} papers normalize --paper "$paper"

# Test tooling, lint TeX, compile, and check one paper without publishing changes.
[group('papers')]
paper-check paper="validation": paper-tex-fmt-check paper-tex-lint paper-test (paper-build paper) && (paper-pdf-check paper)

# Remove only generated paper build artifacts.
[group('papers')]
paper-clean:
    rm -rf target/papers

# Check the target-built PDF for basic readability and configured content.
[group('papers')]
paper-pdf-check paper="validation":
    {{ paper_rrt }} papers check --paper {{ quote(paper) }}

# Refresh one tracked reviewer PDF after its non-mutating checks pass.
[group('papers')]
paper-refresh paper="validation": (paper-check paper)
    #!/usr/bin/env bash
    set -euo pipefail
    paper={{ quote(paper) }}
    case "$paper" in
        ""|*[!A-Za-z0-9_-]*)
            echo "Invalid paper name: $paper" >&2
            exit 1
            ;;
    esac
    {{ paper_rrt }} papers normalize --paper "$paper" --output "papers/${paper}.pdf"

# Install optional paper tools after provisioning the native libraries in papers/README.md.
[group('papers')]
paper-setup: setup-tools
    #!/usr/bin/env bash
    set -euo pipefail
    export PATH="/Library/TeX/texbin:$PATH"
    tectonic_environment="$({{ paper_rrt }} tectonic discover --format shell)"
    eval "$tectonic_environment"
    {{ paper_rrt }} validation require chktex
    {{ paper_rrt }} toolchain sync

# Run paper-specific tests in the root Python environment.
[group('papers')]
paper-test:
    uv run --locked --group dev --group papers pytest tests/papers

# Format publication-facing TeX sources.
[group('papers')]
paper-tex-fmt: _ensure-paper-toolchain
    {{ paper_managed }} tex-fmt papers/*.tex

# Check TeX formatting without modifying sources.
[group('papers')]
paper-tex-fmt-check: _ensure-paper-toolchain
    {{ paper_managed }} tex-fmt --check papers/*.tex

# Lint TeX sources with the system ChkTeX installation.
[group('papers')]
paper-tex-lint:
    #!/usr/bin/env bash
    set -euo pipefail
    export PATH="/Library/TeX/texbin:$PATH"
    {{ paper_rrt }} validation require chktex
    # 24 conflicts with tex-fmt's indented figure labels.
    chktex -q -n 1 -n 8 -n 24 -n 46 papers/*.tex

# Refresh shared notebook figures and the tracked reviewer PDF explicitly.
[group('papers')]
papers: validation-doc-figures (paper-refresh "validation")

# Discover or export native Tectonic dependencies without installation.
[group('papers')]
[positional-arguments]
tectonic +args:
    {{ paper_rrt }} tectonic "$@"
