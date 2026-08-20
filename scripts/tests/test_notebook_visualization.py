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


def test_load_visualization_3d_rejects_repeated_simplex_vertex(tmp_path: Path) -> None:
    path = tmp_path / "visualization.json"
    artifact = visualization_artifact()
    artifact["simplices"][0]["vertex_ids"][1] = "a"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="four distinct vertex IDs"):
        load_visualization_3d(path)


def test_load_visualization_3d_rejects_facet_incompatible_neighbor(tmp_path: Path) -> None:
    path = tmp_path / "visualization.json"
    artifact = visualization_artifact()
    artifact["vertices"].append({"id": "e", "coordinates": [1.0, 1.0, 1.0]})
    artifact["simplices"].append({"id": "other", "vertex_ids": ["e", "a", "b", "c"]})
    artifact["adjacency"][0]["neighbor_simplex_id"] = "other"
    artifact["adjacency"].extend({"simplex_id": "other", "facet_index": index, "neighbor_simplex_id": "tet" if index == 0 else None} for index in range(4))
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match=r"source facet vertex .* is absent from the neighbor"):
        load_visualization_3d(path)


def test_load_visualization_3d_rejects_nonreciprocal_self_neighbor(tmp_path: Path) -> None:
    path = tmp_path / "visualization.json"
    artifact = visualization_artifact()
    artifact["adjacency"][0]["neighbor_simplex_id"] = "tet"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="distinct reciprocal adjacency"):
        load_visualization_3d(path)


def test_load_visualization_3d_rejects_nonreciprocal_ordinary_neighbor(tmp_path: Path) -> None:
    path = tmp_path / "visualization.json"
    artifact = visualization_artifact()
    artifact["vertices"].append({"id": "e", "coordinates": [1.0, 1.0, 1.0]})
    artifact["simplices"].append({"id": "other", "vertex_ids": ["e", "b", "c", "d"]})
    artifact["adjacency"][0]["neighbor_simplex_id"] = "other"
    artifact["adjacency"].extend({"simplex_id": "other", "facet_index": index, "neighbor_simplex_id": None} for index in range(4))
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="distinct reciprocal adjacency"):
        load_visualization_3d(path)


def test_load_visualization_3d_accepts_reciprocal_facet_neighbors(tmp_path: Path) -> None:
    path = tmp_path / "visualization.json"
    artifact = visualization_artifact()
    artifact["vertices"].append({"id": "e", "coordinates": [1.0, 1.0, 1.0]})
    artifact["simplices"].append({"id": "other", "vertex_ids": ["e", "b", "c", "d"]})
    artifact["adjacency"][0]["neighbor_simplex_id"] = "other"
    artifact["adjacency"].extend({"simplex_id": "other", "facet_index": index, "neighbor_simplex_id": "tet" if index == 0 else None} for index in range(4))
    path.write_text(json.dumps(artifact), encoding="utf-8")

    loaded = load_visualization_3d(path)

    assert loaded.neighbors["tet"][0] == "other"
    assert loaded.neighbors["other"][0] == "tet"
    assert len(loaded.boundary_faces) == 6
