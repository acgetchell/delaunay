"""Validated 3D visualization records shared by repository notebooks."""

import json
import math
from dataclasses import dataclass
from functools import cache
from itertools import combinations
from typing import TYPE_CHECKING, Any, Never, cast

if TYPE_CHECKING:
    from pathlib import Path

type JsonObject = dict[str, Any]
type Point3 = tuple[float, float, float]
type Triangle = tuple[int, int, int]
type SimplexVertices = tuple[str, str, str, str]
type NeighborSlots = tuple[str | None, str | None, str | None, str | None]
type Edge = tuple[str, str]
type Face = tuple[str, str, str]
type AxisLimits = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
type HullFacet = tuple[Point3, Point3, Point3]
type Polygon3 = tuple[Point3, ...]
type Segment3 = tuple[Point3, Point3]


@dataclass(frozen=True, slots=True)
class Visualization3D:
    """A validated 3D simplicial-complex visualization artifact."""

    metadata: JsonObject
    coordinates: dict[str, Point3]
    simplices: dict[str, SimplexVertices]
    neighbors: dict[str, NeighborSlots]
    edges: list[Edge]
    boundary_faces: list[Face]
    axis_limits: AxisLimits


@dataclass(frozen=True, slots=True)
class SphericalHero:
    """A validated spherical triangulation artifact for the README hero."""

    vertices: tuple[Point3, ...]
    simplices: tuple[Triangle, ...]


@dataclass(frozen=True, slots=True)
class ReadmeFigureConfig:
    """Validated rendering limits for the quickstart README preview."""

    internal_face_limit: int
    edge_limit: int
    point_limit: int
    axis_scale: float
    transparent: bool

    def __post_init__(self) -> None:
        """Reject invalid limits at the reusable rendering boundary."""
        for name, value in (
            ("internal_face_limit", self.internal_face_limit),
            ("edge_limit", self.edge_limit),
            ("point_limit", self.point_limit),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if isinstance(self.axis_scale, bool) or not isinstance(self.axis_scale, int | float):
            raise TypeError(f"axis_scale must be a finite positive number, got {self.axis_scale!r}")
        if not math.isfinite(self.axis_scale) or self.axis_scale <= 0.0:
            raise ValueError(f"axis_scale must be finite and positive, got {self.axis_scale!r}")
        if type(self.transparent) is not bool:
            raise TypeError(f"transparent must be a boolean, got {self.transparent!r}")


@dataclass(frozen=True, slots=True)
class ReadmeFigureSelection:
    """Deterministically selected geometry for the quickstart README preview."""

    hull_polygons: tuple[HullFacet, ...]
    hull_vertices: tuple[Point3, ...]
    boundary_polygons: tuple[Polygon3, ...]
    cutaway_polygons: tuple[Polygon3, ...]
    cutaway_edges: tuple[Segment3, ...]
    points: tuple[Point3, ...]
    axis_limits: AxisLimits
    total_internal_faces: int
    total_edges: int


@dataclass(frozen=True, slots=True)
class ReadmeFigureResult:
    """Output path and complete geometry counts from one README render."""

    output_path: Path
    hull_facets: int
    boundary_triangles: int
    internal_faces_drawn: int
    total_internal_faces: int
    cutaway_edges_drawn: int
    total_edges: int
    points_drawn: int


@dataclass(frozen=True, slots=True)
class _ReadmePlottingBackend:
    """Lazily imported Matplotlib objects used only by README rendering."""

    pyplot: Any
    line_collection_3d: Any
    polygon_collection_3d: Any


@cache
def _readme_plotting_backend() -> _ReadmePlottingBackend:
    """Import Matplotlib only when a caller renders the README preview."""
    from matplotlib import pyplot as plt  # noqa: PLC0415 - optional notebook dependency
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: PLC0415 - optional notebook dependency

    return _ReadmePlottingBackend(
        pyplot=plt,
        line_collection_3d=Line3DCollection,
        polygon_collection_3d=Poly3DCollection,
    )


def _fail_type(message: str) -> Never:
    raise TypeError(message)


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"JSON artifact contains non-finite value {value}")


def load_json_object(path: Path) -> JsonObject:
    """Load one strict top-level JSON object."""
    with path.open(encoding="utf-8") as file:
        value = json.load(file, parse_constant=_reject_json_constant)
    if not isinstance(value, dict):
        _fail_type(f"expected top-level JSON object in {path}")
    return cast("JsonObject", value)


def finite_coordinate(value: object, context: str) -> float:
    """Parse one finite coordinate from a Rust JSON export."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{context} must be a finite JSON number, got {value!r}")
    coordinate = float(value)
    if not math.isfinite(coordinate):
        raise ValueError(f"{context} must be finite, got {coordinate!r}")
    return coordinate


def exact_integer(value: object, context: str) -> int:
    """Parse one exact JSON integer, excluding booleans and floats."""
    if type(value) is not int:
        raise TypeError(f"{context} must be an integer, got {value!r}")
    return value


def point3(value: object) -> Point3:
    """Parse one finite 3D coordinate array."""
    if not isinstance(value, list) or len(value) != 3:
        raise TypeError(f"expected 3D point coordinates, got {value!r}")
    return (
        finite_coordinate(value[0], "point.x"),
        finite_coordinate(value[1], "point.y"),
        finite_coordinate(value[2], "point.z"),
    )


def _metadata(data: JsonObject) -> JsonObject:
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        _fail_type("visualization field 'metadata' must be an object")
    if metadata.get("schema") != "delaunay.simplicial_complex":
        raise ValueError(f"unsupported visualization schema: {metadata.get('schema')!r}")
    schema_version = exact_integer(metadata.get("schema_version"), "visualization schema_version")
    if schema_version != 1:
        raise ValueError(f"unsupported visualization schema_version: {schema_version!r}")
    dimension = exact_integer(metadata.get("dimension"), "visualization dimension")
    if dimension != 3:
        raise ValueError(f"unsupported visualization dimension: {dimension!r}")
    return cast("JsonObject", metadata)


def _spherical_vertices(data: JsonObject, expected_vertex_count: int) -> tuple[Point3, ...]:
    raw_vertices = data.get("vertices")
    if not isinstance(raw_vertices, list):
        _fail_type("spherical hero field 'vertices' must be a list")
    if len(raw_vertices) != expected_vertex_count:
        raise ValueError(f"expected {expected_vertex_count} spherical hero vertices, got {len(raw_vertices)}")
    vertices: list[Point3] = []
    for vertex_index, raw_vertex in enumerate(raw_vertices):
        if not isinstance(raw_vertex, list) or len(raw_vertex) != 3:
            _fail_type(f"spherical hero vertices[{vertex_index}] must contain exactly three coordinates")
        point = (
            finite_coordinate(raw_vertex[0], f"vertices[{vertex_index}][0]"),
            finite_coordinate(raw_vertex[1], f"vertices[{vertex_index}][1]"),
            finite_coordinate(raw_vertex[2], f"vertices[{vertex_index}][2]"),
        )
        squared_norm = math.fsum(coordinate * coordinate for coordinate in point)
        if not math.isclose(squared_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"spherical hero vertices[{vertex_index}] must lie on the unit sphere")
        vertices.append(point)
    return tuple(vertices)


def _spherical_simplices(data: JsonObject, vertex_count: int) -> tuple[Triangle, ...]:
    raw_simplices = data.get("simplices")
    if not isinstance(raw_simplices, list):
        _fail_type("spherical hero field 'simplices' must be a list")
    expected_simplex_count = 2 * vertex_count - 4
    if len(raw_simplices) != expected_simplex_count:
        raise ValueError(f"expected {expected_simplex_count} closed-sphere triangles, got {len(raw_simplices)}")
    simplices: list[Triangle] = []
    for simplex_index, raw_simplex in enumerate(raw_simplices):
        if not isinstance(raw_simplex, list) or len(raw_simplex) != 3:
            _fail_type(f"spherical hero simplices[{simplex_index}] must contain exactly three vertex indices")
        simplex = (
            exact_integer(raw_simplex[0], f"simplices[{simplex_index}][0]"),
            exact_integer(raw_simplex[1], f"simplices[{simplex_index}][1]"),
            exact_integer(raw_simplex[2], f"simplices[{simplex_index}][2]"),
        )
        for offset, vertex_index in enumerate(simplex):
            if vertex_index < 0 or vertex_index >= vertex_count:
                raise IndexError(
                    f"simplices[{simplex_index}][{offset}] index {vertex_index} is outside 0..{vertex_count - 1}",
                )
        if len(set(simplex)) != 3:
            raise ValueError(f"spherical hero simplices[{simplex_index}] must contain three distinct vertex indices")
        simplices.append(simplex)
    return tuple(simplices)


def load_spherical_hero(path: Path, *, expected_vertex_count: int) -> SphericalHero:
    """Parse and validate one ``delaunay.spherical_hero`` artifact."""
    if type(expected_vertex_count) is not int or expected_vertex_count < 4:
        raise ValueError(f"expected_vertex_count must be an integer at least 4, got {expected_vertex_count!r}")
    data = load_json_object(path)
    if data.get("schema") != "delaunay.spherical_hero":
        raise ValueError(f"unsupported spherical hero schema: {data.get('schema')!r}")
    schema_version = exact_integer(data.get("schema_version"), "spherical hero schema_version")
    if schema_version != 1:
        raise ValueError(f"unsupported spherical hero schema_version: {schema_version!r}")
    intrinsic_dimension = exact_integer(data.get("intrinsic_dimension"), "spherical hero intrinsic_dimension")
    if intrinsic_dimension != 2:
        raise ValueError(f"unsupported spherical hero intrinsic_dimension: {intrinsic_dimension!r}")
    ambient_dimension = exact_integer(data.get("ambient_dimension"), "spherical hero ambient_dimension")
    if ambient_dimension != 3:
        raise ValueError(f"unsupported spherical hero ambient_dimension: {ambient_dimension!r}")
    vertices = _spherical_vertices(data, expected_vertex_count)
    simplices = _spherical_simplices(data, len(vertices))
    return SphericalHero(vertices=vertices, simplices=simplices)


def _vertex_coordinates(data: JsonObject) -> dict[str, Point3]:
    records = data.get("vertices")
    if not isinstance(records, list):
        _fail_type("visualization field 'vertices' must be a list")
    coordinates: dict[str, Point3] = {}
    for record in records:
        if not isinstance(record, dict):
            _fail_type("vertex record must be a JSON object")
        vertex_id = record.get("id")
        if not isinstance(vertex_id, str):
            _fail_type("vertex record is missing string field 'id'")
        if vertex_id in coordinates:
            raise ValueError(f"duplicate vertex ID: {vertex_id}")
        coordinates[vertex_id] = point3(record.get("coordinates"))
    return coordinates


def _simplex_vertex_map(data: JsonObject) -> dict[str, SimplexVertices]:
    records = data.get("simplices")
    if not isinstance(records, list):
        _fail_type("visualization field 'simplices' must be a list")
    simplices: dict[str, SimplexVertices] = {}
    for record in records:
        if not isinstance(record, dict):
            _fail_type("simplex record must be a JSON object")
        simplex_id = record.get("id")
        vertex_ids = record.get("vertex_ids")
        if not isinstance(simplex_id, str):
            _fail_type("simplex record is missing string field 'id'")
        if not isinstance(vertex_ids, list) or len(vertex_ids) != 4 or not all(isinstance(item, str) for item in vertex_ids):
            _fail_type("3D simplex records must contain four string vertex IDs")
        if simplex_id in simplices:
            raise ValueError(f"duplicate simplex ID: {simplex_id}")
        typed_ids = cast("list[str]", vertex_ids)
        if len(set(typed_ids)) != 4:
            raise ValueError(f"simplex {simplex_id!r} must contain four distinct vertex IDs")
        simplices[simplex_id] = (typed_ids[0], typed_ids[1], typed_ids[2], typed_ids[3])
    return simplices


def _simplex_neighbor_map(data: JsonObject, simplices: dict[str, SimplexVertices]) -> dict[str, NeighborSlots]:
    records = data.get("adjacency")
    if not isinstance(records, list):
        _fail_type("visualization field 'adjacency' must be a list")
    slots: dict[str, list[str | None]] = {simplex_id: [None, None, None, None] for simplex_id in simplices}
    seen: set[tuple[str, int]] = set()
    for record in records:
        if not isinstance(record, dict):
            _fail_type("adjacency record must be a JSON object")
        simplex_id = record.get("simplex_id")
        facet_index = record.get("facet_index")
        neighbor_id = record.get("neighbor_simplex_id")
        if not isinstance(simplex_id, str) or simplex_id not in slots:
            raise KeyError(f"adjacency references unknown simplex ID: {simplex_id!r}")
        if isinstance(facet_index, bool) or not isinstance(facet_index, int) or not 0 <= facet_index < 4:
            raise ValueError(f"invalid 3D facet index: {facet_index!r}")
        if neighbor_id is not None and not isinstance(neighbor_id, str):
            _fail_type("neighbor simplex ID must be a string or null")
        key = (simplex_id, facet_index)
        if key in seen:
            raise ValueError(f"duplicate adjacency record: {key}")
        seen.add(key)
        slots[simplex_id][facet_index] = neighbor_id
    expected = len(simplices) * 4
    if len(seen) != expected:
        raise ValueError(f"expected {expected} adjacency records, got {len(seen)}")
    return {simplex_id: (values[0], values[1], values[2], values[3]) for simplex_id, values in slots.items()}


def _validate_adjacency_facets(simplices: dict[str, SimplexVertices], neighbors: dict[str, NeighborSlots]) -> None:
    """Validate neighbor facet compatibility and reciprocal records."""
    neighbor_edge_counts: dict[tuple[str, str], int] = {}
    for simplex_id, slots in neighbors.items():
        source_vertices = simplices[simplex_id]
        for facet_index, neighbor_id in enumerate(slots):
            if neighbor_id is None:
                continue
            neighbor_vertices = simplices[neighbor_id]
            missing_facet_vertex = next(
                (vertex_id for vertex_index, vertex_id in enumerate(source_vertices) if vertex_index != facet_index and vertex_id not in neighbor_vertices),
                None,
            )
            if missing_facet_vertex is not None:
                raise ValueError(
                    f"adjacency for simplex {simplex_id!r} facet {facet_index} references neighbor {neighbor_id!r}, "
                    f"but source facet vertex {missing_facet_vertex!r} is absent from the neighbor"
                )
            edge = (simplex_id, neighbor_id)
            neighbor_edge_counts[edge] = neighbor_edge_counts.get(edge, 0) + 1

    for simplex_id, slots in neighbors.items():
        for facet_index, neighbor_id in enumerate(slots):
            if neighbor_id is None:
                continue
            if neighbor_id == simplex_id:
                raise ValueError(f"adjacency for simplex {simplex_id!r} facet {facet_index} references itself, but self-referential adjacency is invalid")
            reciprocal_count = neighbor_edge_counts.get((neighbor_id, simplex_id), 0)
            if reciprocal_count == 0:
                raise ValueError(
                    f"adjacency for simplex {simplex_id!r} facet {facet_index} references neighbor {neighbor_id!r}, "
                    "but the neighbor does not provide a distinct reciprocal adjacency"
                )


def _validate_references(coordinates: dict[str, Point3], simplices: dict[str, SimplexVertices], neighbors: dict[str, NeighborSlots]) -> None:
    missing_vertices = sorted({vertex_id for vertex_ids in simplices.values() for vertex_id in vertex_ids if vertex_id not in coordinates})
    if missing_vertices:
        raise KeyError(f"simplices reference {len(missing_vertices)} unknown vertex ID(s), first: {missing_vertices[:5]}")
    unknown_neighbors = sorted(
        {neighbor_id for values in neighbors.values() for neighbor_id in values if neighbor_id is not None and neighbor_id not in simplices}
    )
    if unknown_neighbors:
        raise KeyError(f"adjacency references {len(unknown_neighbors)} unknown simplex ID(s), first: {unknown_neighbors[:5]}")
    _validate_adjacency_facets(simplices, neighbors)


def unique_edges(simplices: dict[str, SimplexVertices]) -> list[Edge]:
    """Derive sorted unique graph edges from tetrahedral simplex records."""
    edges: set[Edge] = set()
    for vertex_ids in simplices.values():
        for left, right in combinations(vertex_ids, 2):
            edges.add(cast("Edge", tuple(sorted((left, right)))))
    return sorted(edges)


def faces_by_boundary_state(simplices: dict[str, SimplexVertices], neighbors: dict[str, NeighborSlots], *, boundary: bool) -> list[Face]:
    """Derive unique triangular faces selected by neighbor presence."""
    faces: set[Face] = set()
    for simplex_id, vertex_ids in simplices.items():
        slots = neighbors.get(simplex_id)
        if slots is None:
            raise KeyError(f"missing neighbor slots for simplex {simplex_id!r}")
        for opposite_index, neighbor_id in enumerate(slots):
            if (neighbor_id is None) != boundary:
                continue
            face_ids = tuple(vertex_id for index, vertex_id in enumerate(vertex_ids) if index != opposite_index)
            faces.add(cast("Face", tuple(sorted(face_ids))))
    return sorted(faces)


def coordinate_axis_limits(points: list[Point3]) -> AxisLimits:
    """Return equal-scale 3D plot limits for a point cloud."""
    if not points:
        message = "cannot compute display limits for an empty point cloud"
        raise ValueError(message)
    minima = [min(point[axis] for point in points) for axis in range(3)]
    maxima = [max(point[axis] for point in points) for axis in range(3)]
    span = max(maximum - minimum for minimum, maximum in zip(minima, maxima, strict=True))
    half_span = max(span * 1.18, 0.5)
    centers = [(minimum + maximum) / 2.0 for minimum, maximum in zip(minima, maxima, strict=True)]
    return (
        (centers[0] - half_span, centers[0] + half_span),
        (centers[1] - half_span, centers[1] + half_span),
        (centers[2] - half_span, centers[2] + half_span),
    )


def scaled_axis_limits(limits: AxisLimits, scale: float) -> AxisLimits:
    """Return equal-scale limits scaled around their shared center."""
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"axis limit scale must be finite and positive, got {scale}")
    expanded = []
    for lower, upper in limits:
        center = (lower + upper) / 2.0
        half_span = (upper - lower) * scale / 2.0
        expanded.append((center - half_span, center + half_span))
    return (expanded[0], expanded[1], expanded[2])


def normalized_point(point: Point3, limits: AxisLimits) -> Point3:
    """Map one point into display-normalized coordinates."""
    values = [0.5 if upper == lower else (coordinate - lower) / (upper - lower) for coordinate, (lower, upper) in zip(point, limits, strict=True)]
    return (values[0], values[1], values[2])


def face_centroid(face: Face, point_map: dict[str, Point3]) -> Point3:
    """Return the centroid of one triangular face."""
    points = [point_map[vertex_id] for vertex_id in face]
    return (
        sum(point[0] for point in points) / 3.0,
        sum(point[1] for point in points) / 3.0,
        sum(point[2] for point in points) / 3.0,
    )


def polygon_centroid(points: HullFacet) -> Point3:
    """Return the centroid of one triangular polygon."""
    return (
        sum(point[0] for point in points) / 3.0,
        sum(point[1] for point in points) / 3.0,
        sum(point[2] for point in points) / 3.0,
    )


def convex_hull_facets(data: JsonObject) -> list[HullFacet]:
    """Parse 3D convex-hull facet polygons from the CLI export."""
    if data.get("dimension") != 3:
        raise TypeError(f"expected 3D convex hull export, got dimension {data.get('dimension')!r}")
    records = data.get("facets")
    if not isinstance(records, list):
        _fail_type("convex hull export field 'facets' must be a list")
    facets: list[HullFacet] = []
    for record in records:
        if not isinstance(record, dict):
            _fail_type("convex hull facet record must be a JSON object")
        coordinates = record.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) != 3:
            _fail_type("3D convex hull facets must contain three coordinate arrays")
        points = tuple(point3(value) for value in coordinates)
        facets.append((points[0], points[1], points[2]))
    return facets


def display_priority(point: Point3, limits: AxisLimits) -> tuple[float, float, float, float, float]:
    """Return a deterministic pseudo-random display priority from coordinates."""
    unit_point = normalized_point(point, limits)
    mixed = (unit_point[0] * 13.0 + unit_point[1] * 17.0 + unit_point[2] * 19.0) % 1.0
    centered_distance = math.fsum((coordinate - 0.5) ** 2 for coordinate in unit_point)
    return (abs(mixed - 0.5), centered_distance, unit_point[0], unit_point[1], unit_point[2])


def _edge_midpoint(edge: Edge, point_map: dict[str, Point3]) -> Point3:
    left_point = point_map[edge[0]]
    right_point = point_map[edge[1]]
    return (
        (left_point[0] + right_point[0]) / 2.0,
        (left_point[1] + right_point[1]) / 2.0,
        (left_point[2] + right_point[2]) / 2.0,
    )


def in_readme_cutaway(point: Point3, limits: AxisLimits) -> bool:
    """Return whether a point belongs to the documented README cutaway volume."""
    unit_point = normalized_point(point, limits)
    return 0.28 <= unit_point[0] <= 0.70 and 0.14 <= unit_point[1] <= 0.90 and 0.10 <= unit_point[2] <= 0.94


def select_readme_figure_geometry(
    visualization: Visualization3D,
    hull_facets: list[HullFacet],
    config: ReadmeFigureConfig,
) -> ReadmeFigureSelection:
    """Select deterministic cutaway geometry from validated CLI artifacts."""
    coordinates = visualization.coordinates
    limits = visualization.axis_limits
    ordered_hull = tuple(sorted(hull_facets, key=polygon_centroid))
    hull_vertices = tuple(sorted({point for polygon in ordered_hull for point in polygon}, key=lambda point: display_priority(point, limits)))
    boundary_polygons = tuple(tuple(coordinates[vertex_id] for vertex_id in face) for face in visualization.boundary_faces)

    internal_faces = faces_by_boundary_state(visualization.simplices, visualization.neighbors, boundary=False)
    cutaway_faces = [face for face in internal_faces if in_readme_cutaway(face_centroid(face, coordinates), limits)]
    cutaway_faces.sort(key=lambda face: display_priority(face_centroid(face, coordinates), limits))
    cutaway_polygons = tuple(tuple(coordinates[vertex_id] for vertex_id in face) for face in cutaway_faces[: config.internal_face_limit])

    cutaway_edges = [edge for edge in visualization.edges if in_readme_cutaway(_edge_midpoint(edge, coordinates), limits)]
    cutaway_edges.sort(key=lambda edge: display_priority(_edge_midpoint(edge, coordinates), limits))
    edge_segments = tuple((coordinates[left], coordinates[right]) for left, right in cutaway_edges[: config.edge_limit])
    points = tuple(sorted(coordinates.values(), key=lambda point: display_priority(point, limits))[: config.point_limit])
    return ReadmeFigureSelection(
        hull_polygons=ordered_hull,
        hull_vertices=hull_vertices,
        boundary_polygons=boundary_polygons,
        cutaway_polygons=cutaway_polygons,
        cutaway_edges=edge_segments,
        points=points,
        axis_limits=scaled_axis_limits(limits, config.axis_scale),
        total_internal_faces=len(internal_faces),
        total_edges=len(visualization.edges),
    )


def _palette_colors(pyplot: Any, count: int, *, stride: int, offset: int) -> list[tuple[float, ...]]:
    """Return a deterministic bright Matplotlib color cycle."""
    samples: list[tuple[float, ...]] = []
    for name, sample_count in (("tab20", 20), ("Set3", 12), ("Paired", 12)):
        colormap = pyplot.get_cmap(name)
        samples.extend(cast("tuple[float, ...]", colormap(index)) for index in range(sample_count))
    return [samples[(index * stride + offset) % len(samples)] for index in range(count)]


def _apply_axis_limits(axis: Any, limits: AxisLimits) -> None:
    """Apply equal-scale 3D limits to one Matplotlib axis."""
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_zlim(*limits[2])
    axis.set_box_aspect((1.0, 1.0, 1.0))


def render_readme_figure(
    visualization: Visualization3D,
    hull_facets: list[HullFacet],
    output_path: Path,
    config: ReadmeFigureConfig,
) -> ReadmeFigureResult:
    """Render the deterministic quickstart cutaway and convex-hull preview."""
    selection = select_readme_figure_geometry(visualization, hull_facets, config)
    plotting = _readme_plotting_backend()
    background = "none" if config.transparent else "black"
    figure = plotting.pyplot.figure(figsize=(12.0, 6.0), facecolor=background)
    axes = [
        figure.add_subplot(1, 2, 1, projection="3d"),
        figure.add_subplot(1, 2, 2, projection="3d"),
    ]
    figure.subplots_adjust(left=0.055, right=0.945, bottom=0.11, top=0.89, wspace=0.08)
    triangulation_axis, hull_axis = axes

    for axis in axes:
        axis.set_facecolor(background)
        axis.set_axis_off()
        _apply_axis_limits(axis, selection.axis_limits)
        axis.set_proj_type("ortho")
        axis.view_init(elev=20.0, azim=37.0)

    hull_axis.add_collection3d(
        plotting.polygon_collection_3d(
            selection.hull_polygons,
            facecolors=_palette_colors(plotting.pyplot, len(selection.hull_polygons), stride=5, offset=1),
            edgecolors="#050505",
            linewidths=0.36,
            alpha=0.9,
            zsort="average",
        )
    )
    hull_axis.scatter(
        [point[0] for point in selection.hull_vertices],
        [point[1] for point in selection.hull_vertices],
        [point[2] for point in selection.hull_vertices],
        s=6.0,
        color="#ef4444",
        alpha=0.96,
        depthshade=False,
    )
    if selection.boundary_polygons:
        triangulation_axis.add_collection3d(
            plotting.polygon_collection_3d(
                selection.boundary_polygons,
                facecolors="#14b8a6",
                edgecolors="#22d3ee",
                linewidths=0.14,
                alpha=0.045,
                zsort="average",
            )
        )
    if selection.cutaway_polygons:
        triangulation_axis.add_collection3d(
            plotting.polygon_collection_3d(
                selection.cutaway_polygons,
                facecolors=_palette_colors(plotting.pyplot, len(selection.cutaway_polygons), stride=7, offset=3),
                edgecolors="#060606",
                linewidths=0.10,
                alpha=0.18,
                zsort="average",
            )
        )
    triangulation_axis.add_collection3d(
        plotting.line_collection_3d(
            selection.cutaway_edges,
            colors="#60a5fa",
            linewidths=0.20,
            alpha=0.21,
        )
    )
    triangulation_axis.scatter(
        [point[0] for point in selection.points],
        [point[1] for point in selection.points],
        [point[2] for point in selection.points],
        s=1.8,
        color="#f43f5e",
        alpha=0.32,
        depthshade=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, facecolor=figure.get_facecolor(), transparent=config.transparent)
    plotting.pyplot.show()
    plotting.pyplot.close(figure)
    return ReadmeFigureResult(
        output_path=output_path,
        hull_facets=len(selection.hull_polygons),
        boundary_triangles=len(selection.boundary_polygons),
        internal_faces_drawn=len(selection.cutaway_polygons),
        total_internal_faces=selection.total_internal_faces,
        cutaway_edges_drawn=len(selection.cutaway_edges),
        total_edges=selection.total_edges,
        points_drawn=len(selection.points),
    )


def load_visualization_3d(path: Path) -> Visualization3D:
    """Parse a complete generic 3D simplicial-complex visualization artifact."""
    data = load_json_object(path)
    metadata = _metadata(data)
    coordinates = _vertex_coordinates(data)
    simplices = _simplex_vertex_map(data)
    neighbors = _simplex_neighbor_map(data, simplices)
    _validate_references(coordinates, simplices, neighbors)
    return Visualization3D(
        metadata=metadata,
        coordinates=coordinates,
        simplices=simplices,
        neighbors=neighbors,
        edges=unique_edges(simplices),
        boundary_faces=faces_by_boundary_state(simplices, neighbors, boundary=True),
        axis_limits=coordinate_axis_limits(list(coordinates.values())),
    )
