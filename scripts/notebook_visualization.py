"""Validated 3D visualization records shared by repository notebooks."""

import json
import math
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Any, Never, cast

if TYPE_CHECKING:
    from pathlib import Path

type JsonObject = dict[str, Any]
type Point3 = tuple[float, float, float]
type SimplexVertices = tuple[str, str, str, str]
type NeighborSlots = tuple[str | None, str | None, str | None, str | None]
type Edge = tuple[str, str]
type Face = tuple[str, str, str]
type AxisLimits = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
type HullFacet = list[Point3]


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
    expected = {"schema": "delaunay.simplicial_complex", "schema_version": 1, "dimension": 3}
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"unsupported visualization {field}: {metadata.get(field)!r}")
    return cast("JsonObject", metadata)


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
            reciprocal_count = neighbor_edge_counts.get((neighbor_id, simplex_id), 0)
            if reciprocal_count == 0 or (neighbor_id == simplex_id and reciprocal_count == 1):
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
        facets.append([point3(value) for value in coordinates])
    return facets


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
