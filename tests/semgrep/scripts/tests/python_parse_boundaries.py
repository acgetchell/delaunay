import json
from typing import Any


def reject_json_constant(value: str) -> object:
    msg = f"invalid JSON constant: {value}"
    raise ValueError(msg)


def loads_without_parse_constant(source: str) -> Any:
    # ruleid: delaunay.python.ci-json-loads-reject-nonfinite-constants
    return json.loads(source)


def loads_with_parse_constant(source: str) -> Any:
    # ok: delaunay.python.ci-json-loads-reject-nonfinite-constants
    return json.loads(source, parse_constant=reject_json_constant)


def dumps_without_allow_nan(value: object) -> str:
    # ruleid: delaunay.python.ci-json-dumps-forbid-nonfinite-output
    return json.dumps(value)


def dumps_with_allow_nan_false(value: object) -> str:
    # ok: delaunay.python.ci-json-dumps-forbid-nonfinite-output
    return json.dumps(value, allow_nan=False)


def accepts_zero_vertices(vertices: int) -> bool:
    # ruleid: delaunay.python.ci-performance-metrics-require-positive-counts
    return vertices < 0


def rejects_non_positive_vertices(vertices: int) -> bool:
    # ok: delaunay.python.ci-performance-metrics-require-positive-counts
    return vertices <= 0


def accepts_zero_simplices(simplices: int) -> bool:
    # ruleid: delaunay.python.ci-performance-metrics-require-positive-counts
    return simplices < 0


def rejects_non_positive_simplices(simplices: int) -> bool:
    # ok: delaunay.python.ci-performance-metrics-require-positive-counts
    return simplices <= 0
