"""Static policy tests for public Rustdoc import examples."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src"
ROOT_PRELUDE_IMPORT = re.compile(r"\buse\s+delaunay\s*::\s*prelude\s*::\s*\*\s*;")
RUSTDOC_RUST_FENCE_TAGS = frozenset(
    {
        "compile_fail",
        "edition2015",
        "edition2018",
        "edition2021",
        "edition2024",
        "ignore",
        "no_run",
        "rust",
        "should_panic",
    }
)


def rustdoc_root_prelude_violations(path: Path) -> list[str]:
    """Return kitchen-sink imports or unterminated public Rustdoc fences."""
    violations: list[str] = []
    in_fence = False
    rust_fence = False
    fence_start = 0

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.lstrip()
        if not stripped.startswith(("///", "//!")):
            if in_fence:
                violations.append(f"{path}:{fence_start}: unterminated Rustdoc code fence")
                in_fence = False
                rust_fence = False
            continue

        doc_line = stripped[3:].lstrip()
        if doc_line.startswith("```"):
            if in_fence:
                in_fence = False
                rust_fence = False
                continue

            fence_start = line_number
            fence_info = doc_line[3:].strip()
            fence_tags = {tag.strip() for tag in fence_info.split(",") if tag.strip()}
            in_fence = True
            rust_fence = not fence_tags or bool(fence_tags & RUSTDOC_RUST_FENCE_TAGS)
            continue

        if rust_fence and ROOT_PRELUDE_IMPORT.search(doc_line):
            violations.append(f"{path}:{line_number}: public Rustdoc must use focused preludes")

    if in_fence:
        violations.append(f"{path}:{fence_start}: unterminated Rustdoc code fence")

    return violations


def test_public_rustdoc_uses_focused_preludes() -> None:
    """Reject the root kitchen-sink prelude inside public Rustdoc code fences."""
    violations = [violation for path in sorted(SOURCE_ROOT.rglob("*.rs")) for violation in rustdoc_root_prelude_violations(path)]

    assert not violations, "\n".join(violations)


def test_detector_rejects_root_prelude_in_rustdoc(tmp_path: Path) -> None:
    """Prove the static guard fails when a public example imports the root prelude."""
    source = tmp_path / "example.rs"
    source.write_text(
        "/// ```rust\n/// use delaunay::prelude::*;\n/// ```\npub struct Example;\n",
        encoding="utf-8",
    )

    assert rustdoc_root_prelude_violations(source) == [f"{source}:2: public Rustdoc must use focused preludes"]


def test_detector_accepts_focused_import_and_prose_mention(tmp_path: Path) -> None:
    """Keep focused code imports distinct from explanatory prose."""
    source = tmp_path / "example.rs"
    source.write_text(
        "//! `use delaunay::prelude::*` remains available for experiments.\n//! ```rust\n//! use delaunay::prelude::query::*;\n//! ```\n",
        encoding="utf-8",
    )

    assert rustdoc_root_prelude_violations(source) == []


def test_detector_rejects_unterminated_rustdoc_fence(tmp_path: Path) -> None:
    """Fail closed when malformed Rustdoc could hide the import boundary."""
    source = tmp_path / "example.rs"
    source.write_text("/// ```rust\n/// let value = 1;\n", encoding="utf-8")

    assert rustdoc_root_prelude_violations(source) == [f"{source}:1: unterminated Rustdoc code fence"]


def test_detector_rejects_hidden_root_prelude_import(tmp_path: Path) -> None:
    """Reject root-prelude imports hidden from rendered Rustdoc examples."""
    source = tmp_path / "example.rs"
    source.write_text(
        "/// ```edition2024\n/// # use delaunay :: prelude :: *;\n/// ```\n",
        encoding="utf-8",
    )

    assert rustdoc_root_prelude_violations(source) == [f"{source}:2: public Rustdoc must use focused preludes"]
