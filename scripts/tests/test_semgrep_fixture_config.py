#!/usr/bin/env python3
"""Tests for per-fixture Semgrep config generation."""

from pathlib import Path

import pytest

from semgrep_fixture_config import annotated_rule_ids, main, write_fixture_config


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
