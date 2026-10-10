"""Static policy tests for public Rustdoc import examples."""

import re
from pathlib import Path

from research_repo_tools.semgrep_docs import rust_blocks

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src"
ROOT_PRELUDE_IMPORT = re.compile(r"\buse\s+delaunay\s*::\s*prelude\s*::\s*\*\s*;")


def rustdoc_root_prelude_violations(path: Path) -> list[str]:
    """Apply the focused-prelude policy to shared, line-preserving Rustdoc extraction."""
    try:
        blocks = rust_blocks(path)
    except ValueError as exc:
        return [str(exc)]
    return [
        f"{path}:{line_number}: public Rustdoc must use focused preludes"
        for block in blocks
        for line_number, line in enumerate(block.splitlines(), start=1)
        if ROOT_PRELUDE_IMPORT.search(line)
    ]


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

    assert rustdoc_root_prelude_violations(source) == [f"unclosed Rust documentation fence: {source}"]


def test_detector_rejects_hidden_root_prelude_import(tmp_path: Path) -> None:
    """Reject root-prelude imports hidden from rendered Rustdoc examples."""
    source = tmp_path / "example.rs"
    source.write_text(
        "/// ```edition2024\n/// # use delaunay :: prelude :: *;\n/// ```\n",
        encoding="utf-8",
    )

    assert rustdoc_root_prelude_violations(source) == [f"{source}:2: public Rustdoc must use focused preludes"]
