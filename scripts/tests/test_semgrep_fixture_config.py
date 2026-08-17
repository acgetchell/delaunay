#!/usr/bin/env python3
"""Tests for per-fixture Semgrep config generation."""

from typing import TYPE_CHECKING

from semgrep_fixture_config import (
    annotated_rule_ids,
    main,
    unannotated_rule_ids,
    violation_rule_ids,
    write_fixture_config,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_annotated_rule_ids_preserves_unique_project_rule_order() -> None:
    """Only repository rule annotations should drive fixture configs."""
    fixture_text = """
// ruleid: delaunay.rust.first-rule
// ok: external.rule, delaunay.rust.second-rule
// ruleid: delaunay.rust.first-rule
"""

    assert annotated_rule_ids(fixture_text) == [
        "delaunay.rust.first-rule",
        "delaunay.rust.second-rule",
    ]


def test_annotated_rule_ids_requires_annotation_prefix_boundary() -> None:
    """Embedded annotation names must not select fixture rules."""
    assert annotated_rule_ids("// notruleid: delaunay.rust.covered-rule\n") == []
    assert annotated_rule_ids("// notok: delaunay.rust.covered-rule\n") == []


def test_violation_rule_ids_requires_annotation_prefix_boundary() -> None:
    """Embedded rule identifiers must not count as violation annotations."""
    assert violation_rule_ids("// notruleid: delaunay.rust.covered-rule\n") == []


def test_write_fixture_config_extracts_only_annotated_rules(tmp_path: Path) -> None:
    """Generated configs should stay minimal and preserve annotation order."""
    fixture = tmp_path / "fixture.rs"
    source_config = tmp_path / "semgrep.yaml"
    output_config = tmp_path / "generated.yaml"
    fixture.write_text(
        "// ruleid: delaunay.rust.second-rule, delaunay.rust.first-rule\n",
        encoding="utf-8",
    )
    source_config.write_text(
        """rules:
  - id: delaunay.rust.first-rule
    pattern: first()
    message: first
    severity: ERROR
    languages: [rust]
  - id: delaunay.rust.second-rule
    pattern: second()
    message: second
    severity: ERROR
    languages: [rust]
  - id: delaunay.rust.unused-rule
    pattern: unused()
    message: unused
    severity: ERROR
    languages: [rust]
""",
        encoding="utf-8",
    )

    write_fixture_config(fixture, source_config, output_config)

    assert (
        output_config.read_text(encoding="utf-8")
        == """rules:
  - id: delaunay.rust.second-rule
    pattern: second()
    message: second
    severity: ERROR
    languages: [rust]
  - id: delaunay.rust.first-rule
    pattern: first()
    message: first
    severity: ERROR
    languages: [rust]
"""
    )


def test_main_reports_missing_annotated_rule(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Missing annotations should fail loudly before Semgrep test mode runs."""
    fixture = tmp_path / "fixture.rs"
    source_config = tmp_path / "semgrep.yaml"
    output_config = tmp_path / "generated.yaml"
    fixture.write_text("// ruleid: delaunay.rust.missing-rule\n", encoding="utf-8")
    source_config.write_text("rules:\n", encoding="utf-8")

    exit_code = main([str(fixture), str(source_config), str(output_config)])

    assert exit_code == 1
    assert "missing Semgrep rules" in capsys.readouterr().err
    assert not output_config.exists()


def test_unannotated_rule_ids_reports_rules_without_fixture_coverage(tmp_path: Path) -> None:
    """Coverage checks should compare config rules back to all live fixtures."""
    fixture_root = tmp_path / "fixtures"
    fixture_root.mkdir()
    (fixture_root / "fixture.rs").write_text(
        "// ruleid: delaunay.rust.covered-rule\n",
        encoding="utf-8",
    )
    (fixture_root / "ignored.fixed").write_text(
        "// ruleid: delaunay.rust.uncovered-rule\n",
        encoding="utf-8",
    )
    source_config = tmp_path / "semgrep.yaml"
    source_config.write_text(
        """rules:
  - id: delaunay.rust.covered-rule
    pattern: covered()
  - id: delaunay.rust.uncovered-rule
    pattern: uncovered()
""",
        encoding="utf-8",
    )

    assert unannotated_rule_ids(fixture_root, source_config) == [
        "delaunay.rust.uncovered-rule",
    ]


def test_ok_annotation_does_not_count_as_positive_rule_coverage(tmp_path: Path) -> None:
    """A non-match fixture must not substitute for an expected finding."""
    fixture_root = tmp_path / "fixtures"
    fixture_root.mkdir()
    (fixture_root / "fixture.rs").write_text(
        "// ok: delaunay.rust.ok-only-rule\n",
        encoding="utf-8",
    )
    source_config = tmp_path / "semgrep.yaml"
    source_config.write_text(
        """rules:
  - id: delaunay.rust.ok-only-rule
    pattern: covered()
""",
        encoding="utf-8",
    )

    assert violation_rule_ids("// ok: delaunay.rust.ok-only-rule\n") == []
    assert unannotated_rule_ids(fixture_root, source_config) == [
        "delaunay.rust.ok-only-rule",
    ]


def test_main_reports_rules_without_fixture_coverage(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The harness coverage mode should fail before any Semgrep process starts."""
    fixture_root = tmp_path / "fixtures"
    fixture_root.mkdir()
    source_config = tmp_path / "semgrep.yaml"
    source_config.write_text(
        """rules:
  - id: delaunay.rust.uncovered-rule
    pattern: uncovered()
""",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--check-coverage",
            str(fixture_root),
            str(source_config),
        ],
    )

    assert exit_code == 1
    assert "rules without ruleid fixtures" in capsys.readouterr().err
