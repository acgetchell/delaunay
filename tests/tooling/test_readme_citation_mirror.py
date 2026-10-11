"""Tests for README and CITATION.cff documentation coupling."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
CITATION = ROOT / "CITATION.cff"


def normalize_prose(text: str) -> str:
    """Normalize prose for README/CITATION mirror comparison."""
    text = re.sub(r"\[([^\]]+)\]\[[^\]]+\]", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]", r"\1", text)
    return " ".join(text.split())


def read_readme_introduction_first_paragraph() -> str:
    """Read the first paragraph under the README introduction heading."""
    readme_text = README.read_text(encoding="utf-8")
    match = re.search(
        r"^## .*Introduction\n\n(?P<paragraph>.+?)(?:\n\n)",
        readme_text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, "README.md must contain an Introduction paragraph"
    return normalize_prose(match.group("paragraph"))


def read_citation_abstract() -> str:
    """Read the folded block scalar used for CITATION.cff abstract text."""
    citation_text = CITATION.read_text(encoding="utf-8")
    match = re.search(
        r"^abstract: >-\n(?P<body>(?:  .+\n)+)",
        citation_text,
        flags=re.MULTILINE,
    )
    assert match is not None, "CITATION.cff must contain an abstract block scalar"
    return normalize_prose("\n".join(line.strip() for line in match.group("body").splitlines()))


def test_citation_abstract_mirrors_readme_introduction_first_paragraph() -> None:
    """CITATION.cff abstract should mirror README.md's Introduction paragraph."""
    assert read_citation_abstract() == read_readme_introduction_first_paragraph()
