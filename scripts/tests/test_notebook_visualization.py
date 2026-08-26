"""Tests for validated notebook visualization records."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import notebook_visualization as visualization_module
from notebook_visualization import (
    ReadmeFigureConfig,
    Visualization3D,
    load_spherical_hero,
    load_visualization_3d,
    render_readme_figure,
    select_readme_figure_geometry,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
REPO_ROOT = Path(__file__).resolve().parents[2]


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


def spherical_hero_artifact() -> dict[str, Any]:
    """Return one octahedral spherical triangulation transport fixture."""
    return {
        "schema": "delaunay.spherical_hero",
        "schema_version": 1,
        "intrinsic_dimension": 2,
        "ambient_dimension": 3,
        "vertices": [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        "simplices": [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ],
    }


def write_spherical_hero(path: Path, artifact: dict[str, Any] | None = None) -> None:
    """Write one spherical hero fixture for parser tests."""
    path.write_text(json.dumps(spherical_hero_artifact() if artifact is None else artifact), encoding="utf-8")


def readme_visualization(*, reverse_edges: bool = False) -> Visualization3D:
    """Return two tetrahedra with one deterministic internal cutaway face."""
    coordinates = {
        "a": (0.10, 0.10, 0.10),
        "b": (0.35, 0.30, 0.30),
        "c": (0.55, 0.30, 0.40),
        "d": (0.45, 0.60, 0.50),
        "e": (0.90, 0.90, 0.90),
    }
    edges = [("a", "b"), ("b", "c"), ("c", "d"), ("b", "d"), ("d", "e")]
    if reverse_edges:
        edges.reverse()
    return Visualization3D(
        metadata={"schema": "fixture"},
        coordinates=coordinates,
        simplices={"left": ("a", "b", "c", "d"), "right": ("e", "b", "c", "d")},
        neighbors={"left": ("right", None, None, None), "right": ("left", None, None, None)},
        edges=edges,
        boundary_faces=[("a", "b", "c"), ("b", "c", "e")],
        axis_limits=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    )


def readme_config() -> ReadmeFigureConfig:
    """Return small valid README rendering limits."""
    return ReadmeFigureConfig(
        internal_face_limit=10,
        edge_limit=10,
        point_limit=10,
        axis_scale=0.3,
        transparent=True,
    )


def hull_facets() -> list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]:
    """Return two deliberately unsorted convex-hull triangles."""
    return [
        ((0.5, 0.5, 1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (0.5, 0.5, 1.0)),
    ]


class FakeReadmeAxis:
    """Minimal 3D-axis seam for render-output behavior tests."""

    def set_facecolor(self, _color: str) -> None: ...

    def set_axis_off(self) -> None: ...

    def set_xlim(self, *_limits: float) -> None: ...

    def set_ylim(self, *_limits: float) -> None: ...

    def set_zlim(self, *_limits: float) -> None: ...

    def set_box_aspect(self, _aspect: tuple[float, float, float]) -> None: ...

    def set_proj_type(self, _projection: str) -> None: ...

    def view_init(self, *, elev: float, azim: float) -> None:
        assert (elev, azim) == (20.0, 37.0)

    def add_collection3d(self, _collection: object) -> None: ...

    def scatter(self, *_coordinates: list[float], **_kwargs: object) -> None: ...


class FakeReadmeFigure:
    """Minimal figure seam that records deterministic PNG output."""

    def __init__(self) -> None:
        """Create an empty two-axis figure."""
        self.axes: list[FakeReadmeAxis] = []

    def add_subplot(self, *_args: int, **_kwargs: str) -> FakeReadmeAxis:
        axis = FakeReadmeAxis()
        self.axes.append(axis)
        return axis

    def subplots_adjust(self, **_kwargs: float) -> None: ...

    @staticmethod
    def get_facecolor() -> str:
        return "none"

    @staticmethod
    def savefig(path: Path, **_kwargs: object) -> None:
        path.write_bytes(PNG_SIGNATURE + b"readme")


class FakeReadmePyplot:
    """Minimal pyplot seam for testing renderer publication behavior."""

    def __init__(self) -> None:
        """Create one figure and empty display-state flags."""
        self.figure_value = FakeReadmeFigure()
        self.shown = False
        self.closed = False

    def figure(self, **_kwargs: object) -> FakeReadmeFigure:
        return self.figure_value

    @staticmethod
    def get_cmap(_name: str) -> Any:
        return lambda index: (float(index), 0.0, 0.0, 1.0)

    def show(self) -> None:
        self.shown = True

    def close(self, figure: FakeReadmeFigure) -> None:
        assert figure is self.figure_value
        self.closed = True


def test_readme_geometry_selection_is_deterministic_across_input_order() -> None:
    """Selection depends on coordinates, not mapping or edge iteration order."""
    facets = hull_facets()

    forward = select_readme_figure_geometry(readme_visualization(), facets, readme_config())
    reversed_inputs = select_readme_figure_geometry(readme_visualization(reverse_edges=True), list(reversed(facets)), readme_config())

    assert reversed_inputs == forward
    assert forward.total_internal_faces == 1
    assert len(forward.cutaway_polygons) == 1
    assert forward.total_edges == 5


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("internal_face_limit", 0, ValueError),
        ("edge_limit", True, ValueError),
        ("point_limit", -1, ValueError),
        ("axis_scale", float("nan"), ValueError),
        ("axis_scale", True, TypeError),
        ("transparent", 1, TypeError),
    ],
)
def test_readme_figure_config_rejects_invalid_contract(field: str, value: object, error: type[Exception]) -> None:
    """Reusable renderer limits fail before geometry selection or plotting."""
    values: dict[str, Any] = {
        "internal_face_limit": 10,
        "edge_limit": 10,
        "point_limit": 10,
        "axis_scale": 0.3,
        "transparent": True,
    }
    values[field] = value

    with pytest.raises(error):
        ReadmeFigureConfig(**values)


def test_render_readme_figure_writes_png_and_reports_complete_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rendering saves one output and reports both selected and total geometry."""
    pyplot = FakeReadmePyplot()
    backend = SimpleNamespace(
        pyplot=pyplot,
        line_collection_3d=lambda *args, **kwargs: (args, kwargs),
        polygon_collection_3d=lambda *args, **kwargs: (args, kwargs),
    )
    monkeypatch.setattr(visualization_module, "_readme_plotting_backend", lambda: backend)
    output_path = tmp_path / "nested" / "readme.png"

    result = render_readme_figure(readme_visualization(), hull_facets(), output_path, readme_config())

    assert output_path.read_bytes().startswith(PNG_SIGNATURE)
    assert result.output_path == output_path
    assert (result.hull_facets, result.boundary_triangles) == (2, 2)
    assert (result.internal_faces_drawn, result.total_internal_faces) == (1, 1)
    assert result.cutaway_edges_drawn <= result.total_edges == 5
    assert result.points_drawn == 5
    assert pyplot.shown
    assert pyplot.closed


def test_quickstart_readme_cell_is_typed_orchestration() -> None:
    """Reusable selection and rendering logic stays out of the notebook cell."""
    notebook = json.loads((REPO_ROOT / "notebooks" / "00_quickstart.ipynb").read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == "readme-figure-code")
    source = "".join(cell["source"])

    assert len(source.splitlines()) <= 75
    assert "render_readme_figure(" in source
    assert "convex_hull_facets(" in source
    assert "def " not in source


def test_load_visualization_3d_derives_edges_faces_and_limits(tmp_path: Path) -> None:
    path = tmp_path / "visualization.json"
    path.write_text(json.dumps(visualization_artifact()), encoding="utf-8")

    artifact = load_visualization_3d(path)

    assert len(artifact.coordinates) == 4
    assert len(artifact.edges) == 6
    assert len(artifact.boundary_faces) == 4
    assert all(lower < upper for lower, upper in artifact.axis_limits)


@pytest.mark.parametrize("schema_version", [True, 1.0, "1"])
def test_load_visualization_3d_rejects_non_integer_schema_version(tmp_path: Path, schema_version: object) -> None:
    """Generic visualization schema versions must be exact integers."""
    path = tmp_path / "visualization.json"
    artifact = visualization_artifact()
    artifact["metadata"]["schema_version"] = schema_version
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(TypeError, match="visualization schema_version must be an integer"):
        load_visualization_3d(path)


def test_load_spherical_hero_returns_validated_records(tmp_path: Path) -> None:
    """The spherical parser returns immutable, shape-safe plotting records."""
    path = tmp_path / "spherical.json"
    write_spherical_hero(path)

    artifact = load_spherical_hero(path, expected_vertex_count=6)

    assert len(artifact.vertices) == 6
    assert artifact.vertices[0] == (1.0, 0.0, 0.0)
    assert len(artifact.simplices) == 8
    assert artifact.simplices[0] == (0, 2, 4)


@pytest.mark.parametrize("schema_version", [True, 1.0, "1"])
def test_load_spherical_hero_rejects_non_integer_schema_version(tmp_path: Path, schema_version: object) -> None:
    """Spherical schema versions must be exact JSON integers."""
    path = tmp_path / "spherical.json"
    artifact = spherical_hero_artifact()
    artifact["schema_version"] = schema_version
    write_spherical_hero(path, artifact)

    with pytest.raises(TypeError, match="spherical hero schema_version must be an integer"):
        load_spherical_hero(path, expected_vertex_count=6)


@pytest.mark.parametrize("coordinate", [True, "1.0"])
def test_load_spherical_hero_rejects_non_numeric_coordinate_before_conversion(tmp_path: Path, coordinate: object) -> None:
    """Coordinate coercion must not admit booleans or numeric strings."""
    path = tmp_path / "spherical.json"
    artifact = spherical_hero_artifact()
    artifact["vertices"][0][0] = coordinate
    write_spherical_hero(path, artifact)

    with pytest.raises(TypeError, match=r"vertices\[0\]\[0\] must be a finite JSON number"):
        load_spherical_hero(path, expected_vertex_count=6)


def test_load_spherical_hero_rejects_non_finite_coordinate(tmp_path: Path) -> None:
    """Non-finite JSON coordinates fail before conversion or plotting."""
    path = tmp_path / "spherical.json"
    artifact = spherical_hero_artifact()
    artifact["vertices"][0][0] = float("inf")
    write_spherical_hero(path, artifact)

    with pytest.raises(ValueError, match="JSON artifact contains non-finite value Infinity"):
        load_spherical_hero(path, expected_vertex_count=6)


@pytest.mark.parametrize("simplex_index", [True, 0.0, "0"])
def test_load_spherical_hero_rejects_non_integer_simplex_index(tmp_path: Path, simplex_index: object) -> None:
    """Simplex indices must be exact JSON integers before array conversion."""
    path = tmp_path / "spherical.json"
    artifact = spherical_hero_artifact()
    artifact["simplices"][0][0] = simplex_index
    write_spherical_hero(path, artifact)

    with pytest.raises(TypeError, match=r"simplices\[0\]\[0\] must be an integer"):
        load_spherical_hero(path, expected_vertex_count=6)


def test_load_spherical_hero_rejects_out_of_range_simplex_index(tmp_path: Path) -> None:
    """Every simplex index must refer to a parsed vertex."""
    path = tmp_path / "spherical.json"
    artifact = spherical_hero_artifact()
    artifact["simplices"][0][0] = 6
    write_spherical_hero(path, artifact)

    with pytest.raises(IndexError, match=r"simplices\[0\]\[0\] index 6 is outside 0\.\.5"):
        load_spherical_hero(path, expected_vertex_count=6)


def test_load_spherical_hero_rejects_repeated_simplex_index(tmp_path: Path) -> None:
    """A triangular simplex must carry three distinct vertex indices."""
    path = tmp_path / "spherical.json"
    artifact = spherical_hero_artifact()
    artifact["simplices"][0] = [0, 0, 4]
    write_spherical_hero(path, artifact)

    with pytest.raises(ValueError, match="must contain three distinct vertex indices"):
        load_spherical_hero(path, expected_vertex_count=6)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("vertices", [1.0, 0.0], r"vertices\[0\] must contain exactly three coordinates"),
        ("simplices", [0, 1], r"simplices\[0\] must contain exactly three vertex indices"),
    ],
)
def test_load_spherical_hero_rejects_malformed_nested_shape(tmp_path: Path, field: str, replacement: list[float], message: str) -> None:
    """Nested coordinate and index arrays retain their required widths."""
    path = tmp_path / "spherical.json"
    artifact = spherical_hero_artifact()
    artifact[field][0] = replacement
    write_spherical_hero(path, artifact)

    with pytest.raises(TypeError, match=message):
        load_spherical_hero(path, expected_vertex_count=6)


def test_load_spherical_hero_rejects_non_unit_vertex(tmp_path: Path) -> None:
    """Parsed hero vertices must retain the spherical scientific invariant."""
    path = tmp_path / "spherical.json"
    artifact = spherical_hero_artifact()
    artifact["vertices"][0] = [2.0, 0.0, 0.0]
    write_spherical_hero(path, artifact)

    with pytest.raises(ValueError, match=r"vertices\[0\] must lie on the unit sphere"):
        load_spherical_hero(path, expected_vertex_count=6)


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

    with pytest.raises(ValueError, match="self-referential adjacency is invalid"):
        load_visualization_3d(path)


def test_load_visualization_3d_rejects_two_self_neighbor_slots(tmp_path: Path) -> None:
    path = tmp_path / "visualization.json"
    artifact = visualization_artifact()
    artifact["adjacency"][0]["neighbor_simplex_id"] = "tet"
    artifact["adjacency"][1]["neighbor_simplex_id"] = "tet"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="self-referential adjacency is invalid"):
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
