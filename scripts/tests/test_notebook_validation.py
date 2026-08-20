"""Regression tests for validation-notebook artifact parsing."""

import json
from typing import TYPE_CHECKING, Any

import pytest

from notebook_validation import load_validation_demo

if TYPE_CHECKING:
    from pathlib import Path


def validation_case(level: int) -> dict[str, Any]:
    """Return one minimal valid case for parser tests."""
    return {
        "level": level,
        "layer": f"Level {level}",
        "title": f"case {level}",
        "status": "invalid",
        "public_check": "validate",
        "public_reference": "docs/validation.md",
        "input_summary": "fixture",
        "explanation": "fixture",
        "diagnostic": "fixture",
        "visual": {
            "points": [
                {"label": "a", "coordinates": [0.0, 0.0]},
                {"label": "b", "coordinates": [1.0, 0.0]},
                {"label": "c", "coordinates": [0.0, 1.0]},
            ],
            "simplices": [[0, 1, 2]],
            "highlighted_simplices": [0],
            "highlighted_edges": [[0, 1]],
            "invalid_points": [],
            "isolated_points": [],
            "duplicate_simplices": [],
            "circumcircle": None,
        },
    }


def write_artifact(path: Path) -> None:
    """Write a complete validation-demo transport fixture."""
    artifact = {
        "schema": "delaunay.validation_demo",
        "schema_version": 1,
        "dimension": 2,
        "valid_baseline": validation_case(1),
        "cases": [validation_case(level) for level in range(1, 6)],
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")


def test_load_validation_demo_returns_proof_bearing_records(tmp_path: Path) -> None:
    path = tmp_path / "validation.json"
    write_artifact(path)

    baseline, cases = load_validation_demo(path)

    assert baseline.visual.simplices == ((0, 1, 2),)
    assert [case.level for case in cases] == [1, 2, 3, 4, 5]


def test_load_validation_demo_rejects_out_of_range_visual_index(tmp_path: Path) -> None:
    path = tmp_path / "validation.json"
    write_artifact(path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["cases"][0]["visual"]["simplices"] = [[0, 1, 99]]
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(IndexError, match="outside"):
        load_validation_demo(path)
