"""Validated DTOs for the validation notebook's Rust-generated JSON artifact."""

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pathlib import Path

type JsonObject = dict[str, Any]
type Point2 = tuple[float, float]


@dataclass(frozen=True, slots=True)
class ValidationPoint:
    """A labeled 2D point from validation-demo visual metadata."""

    label: str
    coordinates: Point2


@dataclass(frozen=True, slots=True)
class CircumcircleWitness:
    """A finite positive circumcircle witness for a Level 5 visual."""

    center: Point2
    radius: float


@dataclass(frozen=True, slots=True)
class ValidationVisual:
    """Validated visual metadata consumed by notebook plotting."""

    points: tuple[ValidationPoint, ...]
    simplices: tuple[tuple[int, ...], ...]
    highlighted_simplices: frozenset[int]
    highlighted_edges: tuple[tuple[int, int], ...]
    invalid_points: frozenset[int]
    isolated_points: frozenset[int]
    duplicate_simplices: tuple[tuple[int, ...], ...]
    circumcircle: CircumcircleWitness | None


@dataclass(frozen=True, slots=True)
class ValidationCase:
    """A validated failure case consumed by the validation notebook."""

    level: int
    layer: str
    title: str
    status: str
    public_check: str
    public_reference: str
    input_summary: str
    explanation: str
    diagnostic: str
    visual: ValidationVisual


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"JSON artifact contains non-finite value {value}")


def _object(value: Any, context: str) -> JsonObject:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object")
    return cast("JsonObject", value)


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{context} must be a string")
    return value


def _integer(value: Any, context: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{context} must be an integer")
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{context} must be a finite JSON number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite, got {number!r}")
    return number


def _indices(value: Any, context: str) -> tuple[int, ...]:
    return tuple(_integer(item, f"{context}[{index}]") for index, item in enumerate(_list(value, context)))


def _triangles(value: Any, context: str) -> tuple[tuple[int, ...], ...]:
    triangles: list[tuple[int, ...]] = []
    for index, item in enumerate(_list(value, context)):
        triangle = _indices(item, f"{context}[{index}]")
        if len(triangle) != 3:
            raise ValueError(f"{context}[{index}] must contain exactly three point indices")
        triangles.append(triangle)
    return tuple(triangles)


def _edges(value: Any, context: str) -> tuple[tuple[int, int], ...]:
    edges: list[tuple[int, int]] = []
    for index, item in enumerate(_list(value, context)):
        edge = _indices(item, f"{context}[{index}]")
        if len(edge) != 2:
            raise ValueError(f"{context}[{index}] must contain exactly two point indices")
        edges.append((edge[0], edge[1]))
    return tuple(edges)


def _point(raw_point: Any, context: str) -> ValidationPoint:
    point = _object(raw_point, context)
    coordinates = _list(point.get("coordinates"), f"{context}.coordinates")
    if len(coordinates) != 2:
        raise ValueError(f"{context}.coordinates must have length 2")
    return ValidationPoint(
        label=_string(point.get("label"), f"{context}.label"),
        coordinates=(
            _number(coordinates[0], f"{context}.x"),
            _number(coordinates[1], f"{context}.y"),
        ),
    )


def _circumcircle(value: Any, context: str) -> CircumcircleWitness | None:
    if value is None:
        return None
    circle = _object(value, context)
    center = _list(circle.get("center"), f"{context}.center")
    if len(center) != 2:
        raise ValueError(f"{context}.center must have length 2")
    radius = _number(circle.get("radius"), f"{context}.radius")
    if radius <= 0.0:
        raise ValueError(f"{context}.radius must be positive, got {radius}")
    return CircumcircleWitness(
        center=(
            _number(center[0], f"{context}.center.x"),
            _number(center[1], f"{context}.center.y"),
        ),
        radius=radius,
    )


def _bounded_index(index: int, upper_bound: int, context: str) -> None:
    if index < 0 or index >= upper_bound:
        raise IndexError(f"{context} index {index} is outside 0..{upper_bound - 1}")


def _validate_visual_indices(visual: ValidationVisual, context: str) -> None:
    for simplex_index, simplex in enumerate((*visual.simplices, *visual.duplicate_simplices)):
        for offset, point_index in enumerate(simplex):
            _bounded_index(point_index, len(visual.points), f"{context}.simplices[{simplex_index}][{offset}]")
    for simplex_index in visual.highlighted_simplices:
        _bounded_index(simplex_index, len(visual.simplices), f"{context}.highlighted_simplices")
    for edge_index, edge in enumerate(visual.highlighted_edges):
        for offset, point_index in enumerate(edge):
            _bounded_index(point_index, len(visual.points), f"{context}.highlighted_edges[{edge_index}][{offset}]")
    for field, indices in (
        ("invalid_points", visual.invalid_points),
        ("isolated_points", visual.isolated_points),
    ):
        for offset, point_index in enumerate(indices):
            _bounded_index(point_index, len(visual.points), f"{context}.{field}[{offset}]")


def _visual(raw_visual: Any, context: str) -> ValidationVisual:
    visual = _object(raw_visual, context)
    points = tuple(_point(raw_point, f"{context}.points[{index}]") for index, raw_point in enumerate(_list(visual.get("points"), f"{context}.points")))
    parsed = ValidationVisual(
        points=points,
        simplices=_triangles(visual.get("simplices"), f"{context}.simplices"),
        highlighted_simplices=frozenset(_indices(visual.get("highlighted_simplices"), f"{context}.highlighted_simplices")),
        highlighted_edges=_edges(visual.get("highlighted_edges"), f"{context}.highlighted_edges"),
        invalid_points=frozenset(_indices(visual.get("invalid_points"), f"{context}.invalid_points")),
        isolated_points=frozenset(_indices(visual.get("isolated_points"), f"{context}.isolated_points")),
        duplicate_simplices=_triangles(visual.get("duplicate_simplices"), f"{context}.duplicate_simplices"),
        circumcircle=_circumcircle(visual.get("circumcircle"), f"{context}.circumcircle"),
    )
    _validate_visual_indices(parsed, context)
    return parsed


def _case(raw_case: Any, index: int) -> ValidationCase:
    context = "valid_baseline" if index < 0 else f"cases[{index}]"
    case = _object(raw_case, context)
    level = _integer(case.get("level"), f"{context}.level")
    if level not in {1, 2, 3, 4, 5}:
        raise ValueError(f"{context}.level must be in [1, 5], got {level}")
    return ValidationCase(
        level=level,
        layer=_string(case.get("layer"), f"{context}.layer"),
        title=_string(case.get("title"), f"{context}.title"),
        status=_string(case.get("status"), f"{context}.status"),
        public_check=_string(case.get("public_check"), f"{context}.public_check"),
        public_reference=_string(case.get("public_reference"), f"{context}.public_reference"),
        input_summary=_string(case.get("input_summary"), f"{context}.input_summary"),
        explanation=_string(case.get("explanation"), f"{context}.explanation"),
        diagnostic=_string(case.get("diagnostic"), f"{context}.diagnostic"),
        visual=_visual(case.get("visual"), f"{context}.visual"),
    )


def load_validation_demo(path: Path) -> tuple[ValidationCase, tuple[ValidationCase, ...]]:
    """Parse and validate a complete ``delaunay.validation_demo`` artifact."""
    with path.open(encoding="utf-8") as handle:
        artifact = json.load(handle, parse_constant=_reject_json_constant)
    root = _object(artifact, str(path))
    expected_metadata = {
        "schema": "delaunay.validation_demo",
        "schema_version": 1,
        "dimension": 2,
    }
    for field, expected in expected_metadata.items():
        if root.get(field) != expected:
            raise ValueError(f"unexpected validation-demo {field}: expected {expected!r}, got {root.get(field)!r}")
    baseline = _case(root.get("valid_baseline"), -1)
    cases = tuple(_case(raw_case, index) for index, raw_case in enumerate(_list(root.get("cases"), "cases")))
    levels = [case.level for case in cases]
    if levels != [1, 2, 3, 4, 5]:
        raise ValueError(f"expected validation levels [1, 2, 3, 4, 5], got {levels}")
    return baseline, cases
