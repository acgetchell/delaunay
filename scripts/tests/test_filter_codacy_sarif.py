#!/usr/bin/env python3
"""Tests for Codacy SARIF filtering before GitHub Code Scanning upload."""

import json
from typing import TYPE_CHECKING

import pytest

from ci.filter_codacy_sarif import JsonObject, is_json_object, load_sarif, parse_sarif_document, split_sarif_runs, write_github_env

if TYPE_CHECKING:
    from pathlib import Path


def _opengrep_run() -> JsonObject:
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
    sarif = parse_sarif_document({"version": "2.1.0", "runs": [_opengrep_run()]})
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
    tool = run["tool"]
    assert is_json_object(tool)
    driver = tool["driver"]
    assert is_json_object(driver)
    driver["rules"] = [{"id": "Semgrep_python_assert_rule-assert-used"}]
    run["results"] = [{"ruleId": "Semgrep_python_assert_rule-assert-used"}]

    assert split_sarif_runs(parse_sarif_document({"runs": [run]}), tmp_path / "sarif") == 0
    assert list((tmp_path / "sarif").glob("*.sarif")) == []


def test_non_opengrep_runs_are_split_without_repository_rule_filtering(tmp_path: Path) -> None:
    """Only Codacy/OpenGrep runs are restricted to repository-owned rules."""
    sarif = parse_sarif_document(
        {
            "runs": [
                {
                    "tool": {"driver": {"name": "ruff", "rules": [{"id": "F401"}]}},
                    "results": [{"ruleId": "F401"}],
                }
            ]
        }
    )

    assert split_sarif_runs(sarif, tmp_path / "sarif") == 1

    [out_file] = sorted((tmp_path / "sarif").glob("*.sarif"))
    split = json.loads(out_file.read_text(encoding="utf-8"))
    [run] = split["runs"]
    assert [rule["id"] for rule in run["tool"]["driver"]["rules"]] == ["F401"]
    assert [result["ruleId"] for result in run["results"]] == ["F401"]


def test_split_requires_runs_array(tmp_path: Path) -> None:
    """Malformed SARIF fails loudly instead of silently uploading nothing."""
    with pytest.raises(SystemExit, match="runs array"):
        split_sarif_runs(parse_sarif_document({}), tmp_path / "sarif")


def test_load_sarif_rejects_non_object_root(tmp_path: Path) -> None:
    """The JSON boundary rejects non-object SARIF before split processing."""
    sarif = tmp_path / "codacy.sarif"
    sarif.write_text("[]", encoding="utf-8")

    with pytest.raises(SystemExit, match="root must be a JSON object"):
        load_sarif(sarif)


def test_parse_sarif_document_rejects_non_json_values() -> None:
    """The parser does not let unserializable Python objects masquerade as JSON."""
    with pytest.raises(SystemExit, match="root must be a JSON object"):
        parse_sarif_document({"runs": [{"bad": object()}]})


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_load_sarif_rejects_non_standard_json_constants(tmp_path: Path, value: str) -> None:
    """Python accepts NaN/Infinity by default; SARIF splitting keeps strict JSON."""
    sarif = tmp_path / "codacy.sarif"
    sarif.write_text(f'{{"runs": [{{"tool": {{"driver": {{"name": {value}}}}}}}]}}', encoding="utf-8")

    with pytest.raises(SystemExit, match=f"non-standard JSON constant: {value}"):
        load_sarif(sarif)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_parse_sarif_document_rejects_non_finite_python_floats(value: float) -> None:
    """Direct parser calls reject floats that strict JSON cannot represent."""
    with pytest.raises(SystemExit, match="root must be a JSON object"):
        parse_sarif_document({"runs": [{"bad": value}]})


def test_load_sarif_failure_does_not_touch_output_directory(tmp_path: Path) -> None:
    """Input parsing failures happen before split output cleanup or writes."""
    sarif = tmp_path / "codacy.sarif"
    out_dir = tmp_path / "sarif"
    out_dir.mkdir()
    stale = out_dir / "stale.sarif"
    stale.write_text("keep me\n", encoding="utf-8")
    sarif.write_text('{"runs": [{"tool": {"driver": {"name": NaN}}}]}', encoding="utf-8")

    with pytest.raises(SystemExit, match="non-standard JSON constant: NaN"):
        load_sarif(sarif)

    assert stale.read_text(encoding="utf-8") == "keep me\n"


def test_split_skips_malformed_run_entries(tmp_path: Path) -> None:
    """Malformed run entries stay rejected after the root document is parsed."""
    sarif = parse_sarif_document({"runs": ["not-a-run", {"tool": {"driver": {"name": "ruff"}}, "results": [{"ruleId": "F401"}]}]})

    assert split_sarif_runs(sarif, tmp_path / "sarif") == 1


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
