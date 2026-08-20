"""Tests for validated notebook visualization records."""

import json
from typing import TYPE_CHECKING, Any

import pytest

from notebook_visualization import load_visualization_3d

if TYPE_CHECKING:
    from pathlib import Path


def visualization_artifact() -> dict[str, Any]:
    """Return one tetrahedral generic visualization transport fixture."""
    vertex_ids = ["a", "b", "c", "d"]
    coordinates = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    return {
        "metadata": {
            "schema": "delaunay.simplicial_complex",
            "schema_version": 1,
            "dimension": 3,
        },
        "vertices": [{"id": vertex_id, "coordinates": point} for vertex_id, point in zip(vertex_ids, coordinates, strict=True)],
        "simplices": [{"id": "tet", "vertex_ids": vertex_ids}],
        "adjacency": [{"simplex_id": "tet", "facet_index": index, "neighbor_simplex_id": None} for index in range(4)],
    }


def test_load_visualization_3d_derives_edges_faces_and_limits(tmp_path: Path) -> None:
    path = tmp_path / "visualization.json"
    path.write_text(json.dumps(visualization_artifact()), encoding="utf-8")

    artifact = load_visualization_3d(path)

    assert len(artifact.coordinates) == 4
    assert len(artifact.edges) == 6
    assert len(artifact.boundary_faces) == 4
    assert all(lower < upper for lower, upper in artifact.axis_limits)


def test_load_visualization_3d_rejects_unknown_vertex_reference(tmp_path: Path) -> None:
    path = tmp_path / "visualization.json"
    artifact = visualization_artifact()
    artifact["simplices"][0]["vertex_ids"][3] = "missing"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(KeyError, match="unknown vertex"):
        load_visualization_3d(path)
