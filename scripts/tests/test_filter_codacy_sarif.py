#!/usr/bin/env python3
"""Tests for Codacy SARIF filtering before GitHub Code Scanning upload."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from ci.filter_codacy_sarif import split_sarif_runs, write_github_env

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


def _opengrep_run() -> dict[str, Any]:
    return {
        "tool": {
            "driver": {
                "name": "Opengrep (reported by Codacy)",
                "rules": [
                    {"id": "Semgrep_python_assert_rule-assert-used"},
                    {"id": "delaunay.python.no-broad-exception"},
                ],
            }
        },
        "results": [
            {
                "ruleId": "Semgrep_python_assert_rule-assert-used",
                "message": {"text": "The application was found using assert in non-test code."},
                "locations": [{"physicalLocation": {"artifactLocation": {"uri": "scripts/tests/test_postprocess_changelog.py"}}}],
            },
            {
                "ruleId": "delaunay.python.no-broad-exception",
                "message": {"text": "Catch a specific exception type."},
                "locations": [{"physicalLocation": {"artifactLocation": {"uri": "scripts/postprocess_changelog.py"}}}],
            },
        ],
    }


def test_codacy_opengrep_split_keeps_only_repository_owned_rules(tmp_path: Path) -> None:
    """Default Codacy/OpenGrep rules are dropped before Code Scanning upload."""
    sarif = {"version": "2.1.0", "runs": [_opengrep_run()]}
    out_dir = tmp_path / "sarif"

    assert split_sarif_runs(sarif, out_dir) == 1

    [out_file] = sorted(out_dir.glob("*.sarif"))
    split = json.loads(out_file.read_text(encoding="utf-8"))
    [run] = split["runs"]
    driver = run["tool"]["driver"]

    assert [rule["id"] for rule in driver["rules"]] == ["delaunay.python.no-broad-exception"]
    assert [result["ruleId"] for result in run["results"]] == ["delaunay.python.no-broad-exception"]
    assert run["automationDetails"]["id"] == "codacy-opengrep-reported-by-codacy"


def test_codacy_opengrep_split_skips_default_only_runs(tmp_path: Path) -> None:
    """Runs with only default Codacy/OpenGrep findings produce no uploadable SARIF."""
    run = _opengrep_run()
    run["tool"]["driver"]["rules"] = [{"id": "Semgrep_python_assert_rule-assert-used"}]
    run["results"] = [{"ruleId": "Semgrep_python_assert_rule-assert-used"}]

    assert split_sarif_runs({"runs": [run]}, tmp_path / "sarif") == 0
    assert list((tmp_path / "sarif").glob("*.sarif")) == []


def test_non_opengrep_runs_are_split_without_repository_rule_filtering(tmp_path: Path) -> None:
    """Only Codacy/OpenGrep runs are restricted to repository-owned rules."""
    sarif = {
        "runs": [
            {
                "tool": {"driver": {"name": "ruff", "rules": [{"id": "F401"}]}},
                "results": [{"ruleId": "F401"}],
            }
        ]
    }

    assert split_sarif_runs(sarif, tmp_path / "sarif") == 1

    [out_file] = sorted((tmp_path / "sarif").glob("*.sarif"))
    split = json.loads(out_file.read_text(encoding="utf-8"))
    [run] = split["runs"]
    assert [rule["id"] for rule in run["tool"]["driver"]["rules"]] == ["F401"]
    assert [result["ruleId"] for result in run["results"]] == ["F401"]


def test_split_requires_runs_array(tmp_path: Path) -> None:
    """Malformed SARIF fails loudly instead of silently uploading nothing."""
    with pytest.raises(SystemExit, match="runs array"):
        split_sarif_runs({}, tmp_path / "sarif")


def test_write_github_env_records_upload_state(tmp_path: Path) -> None:
    """The helper writes the environment consumed by the upload step."""
    env_file = tmp_path / "github.env"
    out_dir = tmp_path / "sarif"

    write_github_env(out_dir, 1, env_file)
    write_github_env(out_dir, 0, env_file)

    assert env_file.read_text(encoding="utf-8").splitlines() == [
        f"CODACY_SPLIT_SARIF_DIR={out_dir}",
        "CODACY_HAS_UPLOADABLE_SARIF=true",
        f"CODACY_SPLIT_SARIF_DIR={out_dir}",
        "CODACY_HAS_UPLOADABLE_SARIF=false",
    ]
