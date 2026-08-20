"""Render and independently validate validation-demo notebook figures."""

import math
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from notebook_validation import CircumcircleWitness, Point2, ValidationCase, ValidationVisual


@dataclass(frozen=True, slots=True)
class _PlottingBackend:
    """Lazily imported Matplotlib objects for notebook-only rendering."""

    pyplot: Any
    circle: Any
    polygon: Any
    rectangle: Any


@cache
def _plotting_backend() -> _PlottingBackend:
    """Import Matplotlib only when a caller renders a figure."""
    from matplotlib import pyplot as plt  # noqa: PLC0415 - optional notebook dependency
    from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: PLC0415 - optional notebook dependency

    return _PlottingBackend(pyplot=plt, circle=Circle, polygon=Polygon, rectangle=Rectangle)


def save_figure_png(
    figure: Any,
    png_path: Path,
    *,
    tracked_figure_dir: Path | None = None,
    dpi: int = 180,
) -> None:
    """Save a notebook PNG and an explicitly enabled canonical copy."""
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=dpi, facecolor=figure.get_facecolor())
    if tracked_figure_dir is not None:
        tracked_png_path = tracked_figure_dir / png_path.name
        tracked_png_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(tracked_png_path, dpi=dpi, facecolor=figure.get_facecolor())


def visual_points(visual: ValidationVisual) -> list[tuple[str, Point2]]:
    """Return labeled 2D points from parsed visual metadata."""
    return [(point.label, point.coordinates) for point in visual.points]


def visual_simplices(visual: ValidationVisual) -> list[tuple[int, ...]]:
    """Return parsed 2D visual simplices as a mutable drawing sequence."""
    return list(visual.simplices)


def visual_circle(visual: ValidationVisual) -> tuple[Point2, float] | None:
    """Return the parsed circumcircle witness in Matplotlib-friendly form."""
    if visual.circumcircle is None:
        return None
    return (visual.circumcircle.center, visual.circumcircle.radius)


def point_by_index(points: list[Point2], index: int, context: str) -> Point2:
    """Return a point by visual index with contextual bounds checking."""
    if index < 0 or index >= len(points):
        raise IndexError(f"{context} index {index} is outside 0..{len(points) - 1}")
    return points[index]


def simplex_by_index(simplices: list[tuple[int, ...]], index: int, context: str) -> tuple[int, ...]:
    """Return a simplex by visual index with contextual bounds checking."""
    if index < 0 or index >= len(simplices):
        raise IndexError(f"{context} index {index} is outside 0..{len(simplices) - 1}")
    return simplices[index]


def simplex_points(points: list[Point2], simplex: tuple[int, ...], context: str) -> list[Point2]:
    """Return the three points for a 2D simplex visual."""
    if len(simplex) != 3:
        raise ValueError(f"{context} must contain exactly three point indices")
    return [point_by_index(points, point_index, f"{context}[{offset}]") for offset, point_index in enumerate(simplex)]


VISUAL_AREA_TOLERANCE = 1.0e-12
VISUAL_CIRCLE_TOLERANCE = 1.0e-9


@dataclass(frozen=True, slots=True)
class VisualWitness:
    """Parsed visual metadata used for independent case invariant checks."""

    circumcircle: CircumcircleWitness | None
    points: list[Point2]
    simplices: list[tuple[int, ...]]
    highlighted_simplices: frozenset[int]
    invalid_points: frozenset[int]
    isolated_points: frozenset[int]
    duplicate_simplices: list[tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class ValidationHierarchyLayer:
    """One row in the validation hierarchy overview diagram."""

    level: int
    name: str
    proof_owner: str
    scope: tuple[str, ...]
    questions: tuple[str, ...]
    color: str


def canonical_simplex(simplex: tuple[int, ...]) -> tuple[int, ...]:
    """Return the orientation-independent simplex vertex set."""
    return tuple(sorted(simplex))


def simplex_multiplicities(simplices: list[tuple[int, ...]]) -> dict[tuple[int, ...], int]:
    """Count simplices by their abstract vertex set."""
    counts: dict[tuple[int, ...], int] = {}
    for simplex in simplices:
        key = canonical_simplex(simplex)
        counts[key] = counts.get(key, 0) + 1
    return counts


def triangle_signed_area(points: list[Point2], simplex: tuple[int, ...], context: str) -> float:
    """Return the signed area of a 2D simplex visual."""
    a, b, c = simplex_points(points, simplex, context)
    return 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def point_distance(left: Point2, right: Point2) -> float:
    """Return Euclidean distance between two rendered points."""
    return math.hypot(left[0] - right[0], left[1] - right[1])


def circle_tolerance(radius: float) -> float:
    """Return the scale-aware tolerance for rendered circumcircle witnesses."""
    return VISUAL_CIRCLE_TOLERANCE * max(1.0, radius)


def validate_duplicate_simplex_witness(case_index: int, level: int, witness: VisualWitness) -> None:
    """Verify that duplicate-simplex visual witnesses are real duplicates."""
    if level == 2 and not witness.duplicate_simplices:
        raise ValueError(f"cases[{case_index}] Level 2 must include a duplicate simplex witness")
    counts = simplex_multiplicities(witness.simplices)
    for duplicate_index, duplicate in enumerate(witness.duplicate_simplices):
        multiplicity = counts.get(canonical_simplex(duplicate), 0)
        if multiplicity < 2:
            raise ValueError(f"cases[{case_index}].visual.duplicate_simplices[{duplicate_index}] does not duplicate an emitted simplex")


def validate_isolated_point_witness(case_index: int, level: int, witness: VisualWitness) -> None:
    """Verify that isolated-point witnesses are unused by every simplex."""
    used_points = {point_index for simplex in witness.simplices for point_index in simplex}
    for point_index in sorted(witness.isolated_points):
        if point_index in used_points:
            raise ValueError(f"cases[{case_index}].visual.isolated_points contains used point index {point_index}")
    if level == 3:
        unused_points = set(range(len(witness.points))) - used_points
        if witness.isolated_points != unused_points:
            message = f"cases[{case_index}] Level 3 isolated points {sorted(witness.isolated_points)} do not match unused points {sorted(unused_points)}"
            raise ValueError(message)


def validate_degenerate_simplex_witness(case_index: int, level: int, witness: VisualWitness) -> None:
    """Verify that Level 4 highlighted simplices are geometrically degenerate."""
    if level != 4:
        return
    if not witness.highlighted_simplices:
        raise ValueError(f"cases[{case_index}] Level 4 must highlight a degenerate simplex")
    for simplex_index in sorted(witness.highlighted_simplices):
        simplices_context = f"cases[{case_index}].visual.simplices"
        simplex = simplex_by_index(witness.simplices, simplex_index, simplices_context)
        area = abs(triangle_signed_area(witness.points, simplex, f"{simplices_context}[{simplex_index}]"))
        if area > VISUAL_AREA_TOLERANCE:
            raise ValueError(f"cases[{case_index}].visual.simplices[{simplex_index}] area {area} exceeds degeneracy tolerance {VISUAL_AREA_TOLERANCE}")


def validate_circumcircle_witness(case_index: int, level: int, witness: VisualWitness) -> None:
    """Verify that Level 5 circumcircle metadata witnesses a Delaunay violation."""
    if level != 5:
        return
    circle = witness.circumcircle
    if circle is None:
        raise ValueError(f"cases[{case_index}] Level 5 must include a circumcircle witness")
    if not witness.highlighted_simplices:
        raise ValueError(f"cases[{case_index}] Level 5 must highlight a circumcircle-defining simplex")
    if not witness.invalid_points:
        raise ValueError(f"cases[{case_index}] Level 5 must mark an interior invalid point")
    center = circle.center
    radius = circle.radius
    tolerance = circle_tolerance(radius)
    for simplex_index in sorted(witness.highlighted_simplices):
        simplices_context = f"cases[{case_index}].visual.simplices"
        simplex = simplex_by_index(witness.simplices, simplex_index, simplices_context)
        for vertex_index in simplex:
            vertex = point_by_index(witness.points, vertex_index, f"{simplices_context}[{simplex_index}]")
            residual = abs(point_distance(vertex, center) - radius)
            if residual > tolerance:
                raise ValueError(f"cases[{case_index}] simplex vertex {vertex_index} has circumcircle residual {residual}, tolerance {tolerance}")
    for point_index in sorted(witness.invalid_points):
        point = point_by_index(witness.points, point_index, f"cases[{case_index}].visual.invalid_points")
        distance = point_distance(point, center)
        if distance >= radius - tolerance:
            message = f"cases[{case_index}] invalid point {point_index} is not strictly inside the circumcircle: distance {distance}, radius {radius}"
            raise ValueError(message)


def validate_case_visual_invariants(case: ValidationCase, case_index: int) -> None:
    """Verify that rendered visual metadata independently witnesses the claimed failure."""
    level = case.level
    visual = case.visual
    labeled_points = visual_points(visual)
    points = [point for _, point in labeled_points]
    simplices = visual_simplices(visual)
    witness = VisualWitness(
        circumcircle=visual.circumcircle,
        points=points,
        simplices=simplices,
        highlighted_simplices=visual.highlighted_simplices,
        invalid_points=visual.invalid_points,
        isolated_points=visual.isolated_points,
        duplicate_simplices=list(visual.duplicate_simplices),
    )
    if level == 1 and not witness.invalid_points:
        raise ValueError(f"cases[{case_index}] Level 1 must mark the invalid point")
    validate_duplicate_simplex_witness(case_index, level, witness)
    validate_isolated_point_witness(case_index, level, witness)
    validate_degenerate_simplex_witness(case_index, level, witness)
    validate_circumcircle_witness(case_index, level, witness)


def axes_limits(points: list[Point2], circle: tuple[Point2, float] | None) -> tuple[float, float, float, float]:
    """Return padded axis limits that include points and optional circle."""
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    if circle is not None:
        center, radius = circle
        xs.extend([center[0] - radius, center[0] + radius])
        ys.extend([center[1] - radius, center[1] + radius])
    min_x = min(xs) if xs else -1.0
    max_x = max(xs) if xs else 1.0
    min_y = min(ys) if ys else -1.0
    max_y = max(ys) if ys else 1.0
    span = max(max_x - min_x, max_y - min_y, 1.0)
    padding = 0.18 * span
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    half_span = span / 2.0 + padding
    return (center_x - half_span, center_x + half_span, center_y - half_span, center_y + half_span)


def draw_visual(ax: Any, case: ValidationCase, *, show_point_labels: bool = False) -> None:
    """Draw one validation case from generated visual metadata."""
    visual = case.visual
    labeled_points = visual_points(visual)
    points = [point for _, point in labeled_points]
    simplices = visual_simplices(visual)
    highlighted_simplices = visual.highlighted_simplices
    highlighted_edges = visual.highlighted_edges
    invalid_points = visual.invalid_points
    isolated_points = visual.isolated_points
    duplicate_simplices = list(visual.duplicate_simplices)
    circle = visual_circle(visual)

    palette = ["#7dd3fc", "#fca5a5", "#86efac", "#fde68a", "#c4b5fd"]
    for simplex_index, simplex in enumerate(simplices):
        polygon_points = simplex_points(points, simplex, f"visual.simplices[{simplex_index}]")
        facecolor = "#fb7185" if simplex_index in highlighted_simplices else palette[simplex_index % len(palette)]
        edgecolor = "#be123c" if simplex_index in highlighted_simplices else "#334155"
        polygon = _plotting_backend().polygon(
            polygon_points,
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.6,
            alpha=0.34 if simplex_index in highlighted_simplices else 0.24,
        )
        ax.add_patch(polygon)

    for duplicate_index, duplicate in enumerate(duplicate_simplices):
        duplicate_points = simplex_points(points, duplicate, f"visual.duplicate_simplices[{duplicate_index}]")
        polygon = _plotting_backend().polygon(
            duplicate_points,
            closed=True,
            facecolor="none",
            edgecolor="#7f1d1d",
            linewidth=2.0,
            hatch="///",
            alpha=0.8,
        )
        ax.add_patch(polygon)
    for edge_index, (left, right) in enumerate(highlighted_edges):
        left_point = point_by_index(points, left, f"visual.highlighted_edges[{edge_index}][0]")
        right_point = point_by_index(points, right, f"visual.highlighted_edges[{edge_index}][1]")
        ax.plot([left_point[0], right_point[0]], [left_point[1], right_point[1]], color="#111827", linewidth=2.4)

    if circle is not None:
        center, radius = circle
        ax.add_patch(_plotting_backend().circle(center, radius, fill=False, linestyle="--", linewidth=1.6, edgecolor="#f97316", alpha=0.9))
        ax.scatter([center[0]], [center[1]], s=18, color="#f97316", zorder=5)

    for index, (label, point) in enumerate(labeled_points):
        marker = "x" if index in invalid_points else "o"
        color = "#dc2626" if index in invalid_points else "#0f172a"
        size = 58 if index in invalid_points else 42
        ax.scatter([point[0]], [point[1]], marker=marker, s=size, color=color, zorder=6)
        if index in isolated_points:
            ax.scatter([point[0]], [point[1]], marker="o", s=180, facecolor="none", edgecolor="#dc2626", linewidth=1.8, zorder=5)
        if show_point_labels:
            ax.text(point[0] + 0.03, point[1] + 0.03, label, fontsize=9, weight="bold", color=color)

    limits = axes_limits(points, circle)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#f8fafc")


VALIDATION_HIERARCHY_LAYERS = (
    ValidationHierarchyLayer(
        level=1,
        name="Element Validity",
        proof_owner="Tds",
        scope=("Local elements: Vertex / Simplex",),
        questions=(
            "Are vertex coordinates finite and valid?",
            "Are element UUIDs present and non-nil?",
            "Do simplices have D+1 distinct vertex keys?",
            "Do simplex vertex keys resolve to distinct coordinates?",
            "Are local neighbor slots shaped and assigned?",
        ),
        color="#dbeafe",
    ),
    ValidationHierarchyLayer(
        level=2,
        name="Combinatorial Consistency",
        proof_owner="Tds",
        scope=("Triangulation Data Structure",),
        questions=(
            "Are UUID-key maps unique and bidirectional?",
            "Does every vertex, simplex, and neighbor reference resolve in the TDS?",
            "Are incident-simplex hints internally consistent?",
            "Are vertex-to-simplex incidence indexes exact?",
            "Are duplicate simplices absent?",
            "Are facets shared by at most two simplices?",
            "Do neighbor links and coherent combinatorial orientation agree?",
        ),
        color="#e0f2fe",
    ),
    ValidationHierarchyLayer(
        level=3,
        name="Intrinsic PL Topology",
        proof_owner="Triangulation",
        scope=("Pseudomanifold / PL-Manifold",),
        questions=(
            "Is the simplex neighbor graph connected?",
            "Is every vertex incident to a simplex?",
            "Do facet degrees satisfy the topology guarantee?",
            "Is the true boundary closed?",
            "Do fast ridge-link checks pass when required?",
            "Do complete vertex-link checks pass when required?",
            "Does Euler characteristic match the declared topology?",
            "Is the intrinsic PL manifold orientable?",
        ),
        color="#dcfce7",
    ),
    ValidationHierarchyLayer(
        level=4,
        name="Valid Realization",
        proof_owner="Triangulation",
        scope=("Euclidean / toroidal / spherical realization",),
        questions=(
            "Are affine-chart maximal simplices positively oriented?",
            "Are realized simplices nondegenerate?",
            "Do simplices intersect only along shared faces?",
            "Do toroidal lifts fit valid periodic charts?",
            "Do spherical simplices separate the sphere center?",
            "Does the active realization model support the check?",
        ),
        color="#fef3c7",
    ),
    ValidationHierarchyLayer(
        level=5,
        name="Geometric Predicate Satisfaction",
        proof_owner="DelaunayTriangulation",
        scope=("Delaunay optimality",),
        questions=(
            "Do selected geometric predicates accept every cell?",
            "Do local k=2/k=3 flip predicates and inverses pass?",
            "Do Euclidean/toroidal empty-circumsphere checks pass?",
            "Do spherical empty-cap / ambient hull-facet checks pass?",
            "Is Delaunay optimality certified for the active model?",
        ),
        color="#fee2e2",
    ),
)


def render_validation_hierarchy_figure(output_path: Path, *, tracked_figure_dir: Path | None = None) -> None:
    """Render the five-level validation hierarchy overview PNG."""
    plotting = _plotting_backend()
    figure, axis = plotting.pyplot.subplots(figsize=(13.2, 8.8), facecolor="white", layout="constrained")
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)

    box_width = 0.86
    box_left = 0.07
    box_top = 0.86
    box_bottom = 0.055
    box_gap = 0.018
    box_inner_gap = 0.042
    level_font_size = 11.8
    name_font_size = 12.8
    scope_font_size = 9.1
    question_font_size = 8.7
    scope_gap = 0.031
    scope_step = 0.024
    question_step_cap = 0.024
    layer_weights = [len(layer.questions) + 2.0 for layer in VALIDATION_HIERARCHY_LAYERS]
    usable_height = box_top - box_bottom - box_gap * (len(VALIDATION_HIERARCHY_LAYERS) - 1)
    box_heights = [usable_height * weight / sum(layer_weights) for weight in layer_weights]

    current_top = box_top
    for index, (layer, box_height) in enumerate(zip(VALIDATION_HIERARCHY_LAYERS, box_heights, strict=True)):
        scope_lines = (f"Owner: {layer.proof_owner}", *layer.scope)
        y = current_top - box_height
        row_center = y + box_height / 2.0
        rectangle = plotting.rectangle(
            (box_left, y),
            box_width,
            box_height,
            facecolor=layer.color,
            edgecolor="#334155",
            linewidth=1.2,
        )
        axis.add_patch(rectangle)
        scope_group_height = scope_gap + scope_step * max(len(scope_lines) - 1, 0)
        label_y = row_center + scope_group_height / 2.0
        axis.text(box_left + 0.035, label_y, f"Level {layer.level}", fontsize=level_font_size, weight="bold", color="#0f172a", wrap=False)
        axis.text(box_left + 0.18, label_y, layer.name, fontsize=name_font_size, weight="bold", color="#0f172a", wrap=False)
        for scope_index, scope in enumerate(scope_lines):
            axis.text(
                box_left + 0.18,
                label_y - scope_gap - scope_step * scope_index,
                scope,
                fontsize=scope_font_size,
                color="#64748b",
                wrap=False,
            )
        question_step = min(question_step_cap, (box_height - box_inner_gap) / max(len(layer.questions) - 1, 1))
        question_top = row_center + question_step * (len(layer.questions) - 1) / 2.0
        for question_index, question in enumerate(layer.questions):
            axis.text(
                box_left + 0.48,
                question_top - question_step * question_index,
                f"- {question}",
                fontsize=question_font_size,
                color="#475569",
                wrap=False,
            )
        if index < len(VALIDATION_HIERARCHY_LAYERS) - 1:
            arrow_x = box_left + box_width / 2.0
            axis.annotate(
                "",
                xy=(arrow_x, y - box_gap * 0.86),
                xytext=(arrow_x, y - box_gap * 0.14),
                arrowprops={"arrowstyle": "->", "linewidth": 1.1, "color": "#64748b"},
            )
        current_top = y - box_gap

    axis.text(0.07, 0.965, "delaunay validation architecture", fontsize=16.5, weight="bold", color="#0f172a", wrap=False)
    axis.text(
        0.07,
        0.928,
        "Tds owns Levels 1-2; Triangulation adds Levels 3-4; DelaunayTriangulation adds Level 5.",
        fontsize=11.5,
        color="#475569",
        wrap=False,
    )
    save_figure_png(figure, output_path, tracked_figure_dir=tracked_figure_dir, dpi=180)
    plotting.pyplot.show()
    plotting.pyplot.close(figure)


def wrapped_question(question: str, width: int = 38) -> str:
    """Wrap one validation-family question without adding a text dependency."""
    words = question.split()
    lines: list[str] = []
    line: list[str] = []
    for word in words:
        candidate = " ".join((*line, word))
        if line and len(candidate) > width:
            lines.append(" ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


def glyph_nodes(axis: Any, points: tuple[Point2, ...], *, highlight: tuple[int, ...] = ()) -> None:
    """Draw shared graph nodes with optional invalid-node rings."""
    axis.scatter([point[0] for point in points], [point[1] for point in points], s=28, color="#0f172a", zorder=5)
    for index in highlight:
        point = points[index]
        axis.scatter([point[0]], [point[1]], s=95, facecolor="none", edgecolor="#dc2626", linewidth=1.7, zorder=6)


def glyph_triangle(axis: Any, points: tuple[Point2, Point2, Point2], *, color: str = "#64748b", alpha: float = 0.12) -> None:
    """Draw one filled triangular simplex glyph."""
    axis.add_patch(_plotting_backend().polygon(points, closed=True, facecolor=color, edgecolor=color, linewidth=1.5, alpha=alpha))
    glyph_nodes(axis, points)


def glyph_cross(axis: Any, point: Point2) -> None:
    """Mark one failed relation without relying on a text glyph."""
    radius = 0.045
    axis.plot([point[0] - radius, point[0] + radius], [point[1] - radius, point[1] + radius], color="#dc2626", linewidth=2.0)
    axis.plot([point[0] - radius, point[0] + radius], [point[1] + radius, point[1] - radius], color="#dc2626", linewidth=2.0)


def draw_level_1_glyph(axis: Any, family_index: int) -> None:
    """Draw distinct local element-validity failure witnesses."""
    if family_index == 0:
        axis.scatter([0.50], [0.50], s=38, color="#0f172a")
        axis.text(0.50, 0.28, "coords = [NaN, 0]", ha="center", fontsize=8.5, color="#991b1b")
        glyph_cross(axis, (0.50, 0.50))
    elif family_index == 1:
        axis.add_patch(_plotting_backend().rectangle((0.18, 0.34), 0.64, 0.34, facecolor="white", edgecolor="#64748b", linewidth=1.4))
        axis.text(0.28, 0.55, "UUID", fontsize=9, weight="bold", color="#475569")
        axis.text(0.58, 0.55, "nil", fontsize=10, weight="bold", color="#991b1b")
        glyph_cross(axis, (0.73, 0.55))
    elif family_index == 2:
        points = ((0.22, 0.28), (0.78, 0.28), (0.50, 0.72))
        glyph_triangle(axis, points)
        for label, point in zip(("A", "A", "B"), points, strict=True):
            axis.text(point[0], point[1] + 0.09, label, ha="center", fontsize=9, weight="bold")
        glyph_nodes(axis, points, highlight=(0, 1))
    elif family_index == 3:
        axis.text(0.20, 0.66, "key A", fontsize=8.5, ha="center")
        axis.text(0.20, 0.34, "key B", fontsize=8.5, ha="center")
        axis.annotate("", xy=(0.72, 0.50), xytext=(0.32, 0.66), arrowprops={"arrowstyle": "->", "color": "#64748b"})
        axis.annotate("", xy=(0.72, 0.50), xytext=(0.32, 0.34), arrowprops={"arrowstyle": "->", "color": "#64748b"})
        axis.scatter([0.72], [0.50], s=38, color="#0f172a")
        glyph_cross(axis, (0.72, 0.50))
    else:
        points = ((0.20, 0.26), (0.80, 0.26), (0.50, 0.74))
        glyph_triangle(axis, points)
        for midpoint in ((0.50, 0.26), (0.35, 0.50)):
            axis.add_patch(_plotting_backend().rectangle((midpoint[0] - 0.035, midpoint[1] - 0.035), 0.07, 0.07, facecolor="#86efac", edgecolor="#166534"))
        missing = (0.65, 0.50)
        axis.add_patch(
            _plotting_backend().rectangle((missing[0] - 0.035, missing[1] - 0.035), 0.07, 0.07, facecolor="white", edgecolor="#dc2626", linestyle="--")
        )


def draw_level_2_glyph(axis: Any, family_index: int) -> None:
    """Draw distinct combinatorial-consistency failure witnesses."""
    if family_index == 0:
        axis.text(0.22, 0.52, "UUID", fontsize=9, ha="center")
        axis.text(0.78, 0.52, "key", fontsize=9, ha="center")
        axis.annotate("", xy=(0.68, 0.58), xytext=(0.32, 0.58), arrowprops={"arrowstyle": "->", "color": "#64748b"})
        axis.annotate("", xy=(0.32, 0.42), xytext=(0.68, 0.42), arrowprops={"arrowstyle": "->", "color": "#dc2626", "linestyle": "--"})
        glyph_cross(axis, (0.50, 0.42))
    elif family_index == 1:
        axis.text(0.22, 0.52, "simplex", fontsize=8.5, ha="center")
        axis.annotate("", xy=(0.72, 0.52), xytext=(0.34, 0.52), arrowprops={"arrowstyle": "->", "color": "#64748b"})
        axis.add_patch(_plotting_backend().circle((0.76, 0.52), 0.08, fill=False, edgecolor="#dc2626", linestyle="--"))
        axis.text(0.76, 0.34, "missing ref", fontsize=8, ha="center", color="#991b1b")
    elif family_index == 2:
        points = ((0.14, 0.24), (0.52, 0.24), (0.33, 0.62))
        glyph_triangle(axis, points)
        vertex = (0.80, 0.66)
        axis.scatter([vertex[0]], [vertex[1]], s=32, color="#0f172a")
        axis.annotate("hint", xy=(0.33, 0.45), xytext=vertex, fontsize=8, color="#dc2626", arrowprops={"arrowstyle": "->", "color": "#dc2626"})
        glyph_cross(axis, (0.57, 0.55))
    elif family_index == 3:
        vertex = (0.18, 0.50)
        simplices = ((0.60, 0.68), (0.60, 0.32))
        axis.scatter([vertex[0]], [vertex[1]], s=32, color="#0f172a")
        for simplex in simplices:
            axis.plot([vertex[0], simplex[0]], [vertex[1], simplex[1]], color="#64748b", linewidth=1.3)
            axis.scatter([simplex[0]], [simplex[1]], marker="s", s=38, color="#0f172a")
        axis.text(0.78, 0.50, "index: [top]", fontsize=8, ha="center")
        glyph_cross(axis, simplices[1])
    elif family_index == 4:
        points = ((0.22, 0.28), (0.78, 0.28), (0.50, 0.72))
        glyph_triangle(axis, points, color="#7dd3fc", alpha=0.28)
        shifted = tuple((x + 0.035, y + 0.025) for x, y in points)
        axis.add_patch(_plotting_backend().polygon(shifted, closed=True, facecolor="#fca5a5", edgecolor="#991b1b", linewidth=1.5, alpha=0.30))
        axis.text(0.46, 0.48, "s1", fontsize=8, weight="bold")
        axis.text(0.56, 0.43, "s2", fontsize=8, weight="bold", color="#991b1b")
    elif family_index == 5:
        shared = ((0.42, 0.30), (0.42, 0.70))
        for apex in ((0.12, 0.50), (0.72, 0.78), (0.78, 0.22)):
            glyph_triangle(axis, (shared[0], shared[1], apex), color="#fca5a5", alpha=0.20)
        axis.plot([0.42, 0.42], [0.30, 0.70], color="#dc2626", linewidth=3.0)
        axis.text(0.53, 0.50, "degree 3", fontsize=8.5, color="#991b1b")
    else:
        left = ((0.12, 0.28), (0.50, 0.28), (0.31, 0.66))
        right = ((0.50, 0.28), (0.88, 0.28), (0.69, 0.66))
        glyph_triangle(axis, left, alpha=0.16)
        glyph_triangle(axis, right, alpha=0.16)
        axis.annotate("", xy=(0.58, 0.50), xytext=(0.42, 0.50), arrowprops={"arrowstyle": "->", "color": "#dc2626", "linewidth": 1.7})
        glyph_cross(axis, (0.50, 0.50))


def draw_euler_characteristic_glyph(axis: Any) -> None:
    """Draw a planar simplicial complex whose Euler characteristic is not two."""
    left = ((0.14, 0.39), (0.42, 0.39), (0.42, 0.73))
    right = ((0.42, 0.73), (0.70, 0.73), (0.86, 0.41))
    glyph_triangle(axis, left, color="#7dd3fc", alpha=0.24)
    glyph_triangle(axis, right, color="#86efac", alpha=0.24)
    axis.text(0.50, 0.275, "V - E + F = 1 ≠ 2", fontsize=9, weight="bold", ha="center", color="#991b1b")


def draw_mobius_glyph(axis: Any) -> None:
    """Draw a projected Möbius strip witnessing intrinsic non-orientability."""
    samples = 24
    centerline: list[Point2] = []
    upper: list[Point2] = []
    lower: list[Point2] = []
    for sample in range(samples + 1):
        angle = 2.0 * math.pi * sample / samples
        projected_edges: list[Point2] = []
        for offset in (-0.28, 0.28):
            radius = 1.0 + offset * math.cos(angle / 2.0)
            x = radius * math.cos(angle)
            y = 0.58 * radius * math.sin(angle) + 0.34 * offset * math.sin(angle / 2.0)
            projected_edges.append((0.50 + 0.29 * x, 0.51 + 0.29 * y))
        lower.append(projected_edges[0])
        upper.append(projected_edges[1])
        centerline.append(((projected_edges[0][0] + projected_edges[1][0]) / 2.0, (projected_edges[0][1] + projected_edges[1][1]) / 2.0))
    for sample in range(samples):
        face = (lower[sample], lower[sample + 1], upper[sample + 1], upper[sample])
        shade = "#7dd3fc" if sample < samples // 2 else "#c4b5fd"
        axis.add_patch(_plotting_backend().polygon(face, closed=True, facecolor=shade, edgecolor="#64748b", linewidth=0.45, alpha=0.52))
    axis.plot([point[0] for point in centerline], [point[1] for point in centerline], color="#334155", linewidth=1.0, linestyle="--")
    top = centerline[6]
    bottom = centerline[18]
    axis.annotate("", xy=(top[0], top[1] + 0.13), xytext=top, arrowprops={"arrowstyle": "->", "color": "#dc2626", "linewidth": 1.8})
    axis.annotate("", xy=(bottom[0], bottom[1] - 0.06), xytext=bottom, arrowprops={"arrowstyle": "->", "color": "#dc2626", "linewidth": 1.8})
    axis.text(0.50, 0.195, "transported normal returns reversed", fontsize=8.2, ha="center", color="#991b1b")


def draw_level_3_glyph(axis: Any, family_index: int) -> None:  # noqa: C901
    """Draw distinct intrinsic-topology failure witnesses."""
    if family_index == 0:
        for center in ((0.25, 0.50), (0.75, 0.50)):
            nodes = ((center[0] - 0.10, 0.40), (center[0], 0.64), (center[0] + 0.10, 0.40))
            glyph_triangle(axis, nodes, alpha=0.14)
        glyph_cross(axis, (0.50, 0.50))
    elif family_index == 1:
        points = ((0.14, 0.28), (0.56, 0.28), (0.35, 0.66))
        glyph_triangle(axis, points, alpha=0.16)
        isolated = (0.82, 0.52)
        axis.scatter([isolated[0]], [isolated[1]], s=32, color="#0f172a")
        axis.add_patch(_plotting_backend().circle(isolated, 0.10, fill=False, edgecolor="#dc2626", linestyle="--", linewidth=1.7))
        axis.text(0.82, 0.33, "star = empty", fontsize=8, ha="center", color="#991b1b")
    elif family_index == 2:
        axis.add_patch(_plotting_backend().circle((0.50, 0.50), 0.28, fill=False, edgecolor="#64748b", linewidth=1.6))
        axis.plot([0.33, 0.67], [0.72, 0.72], color="#dc2626", linewidth=3.0)
        axis.text(0.50, 0.32, "closed model", fontsize=8, ha="center")
        axis.text(0.50, 0.82, "degree 1", fontsize=8, ha="center", color="#991b1b")
    elif family_index == 3:
        points = ((0.18, 0.32), (0.38, 0.68), (0.60, 0.32), (0.82, 0.60))
        axis.plot([point[0] for point in points], [point[1] for point in points], color="#64748b", linewidth=2.0)
        glyph_nodes(axis, points, highlight=(0, 3))
        axis.text(0.50, 0.18, "open boundary chain", fontsize=8.5, ha="center", color="#991b1b")
    elif family_index == 4:
        center = (0.50, 0.50)
        ring = ((0.50, 0.78), (0.76, 0.56), (0.65, 0.24), (0.35, 0.24), (0.24, 0.56))
        for point in ring:
            axis.plot([center[0], point[0]], [center[1], point[1]], color="#cbd5e1", linewidth=1.0)
        axis.plot([point[0] for point in ring[:-1]], [point[1] for point in ring[:-1]], color="#64748b", linewidth=1.7)
        glyph_nodes(axis, ring)
        glyph_cross(axis, (0.37, 0.67))
    elif family_index == 5:
        center = (0.50, 0.50)
        left = ((0.16, 0.38), (0.28, 0.70), (0.40, 0.38))
        right = ((0.60, 0.62), (0.72, 0.30), (0.84, 0.62))
        for component in (left, right):
            axis.plot([point[0] for point in (*component, component[0])], [point[1] for point in (*component, component[0])], color="#64748b", linewidth=1.5)
        axis.scatter([center[0]], [center[1]], s=36, color="#dc2626")
        axis.text(0.50, 0.82, "disconnected link", fontsize=8.5, ha="center", color="#991b1b")
    elif family_index == 6:
        draw_euler_characteristic_glyph(axis)
    else:
        draw_mobius_glyph(axis)


def draw_level_4_glyph(axis: Any, family_index: int) -> None:
    """Draw distinct realization-validity failure witnesses."""
    if family_index == 0:
        points = ((0.22, 0.28), (0.78, 0.28), (0.50, 0.72))
        glyph_triangle(axis, points, color="#fde68a", alpha=0.30)
        axis.annotate("", xy=(0.35, 0.31), xytext=(0.62, 0.31), arrowprops={"arrowstyle": "->", "color": "#dc2626", "connectionstyle": "arc3,rad=-0.35"})
    elif family_index == 1:
        points = ((0.18, 0.50), (0.50, 0.50), (0.82, 0.50))
        axis.plot([0.18, 0.82], [0.50, 0.50], color="#dc2626", linewidth=2.5)
        glyph_nodes(axis, points)
        axis.text(0.50, 0.28, "zero area", fontsize=8.5, ha="center", color="#991b1b")
    elif family_index == 2:
        vertices = ((0.20, 0.72), (0.80, 0.72), (0.20, 0.22), (0.80, 0.22))
        lower_simplex = (vertices[0], vertices[2], vertices[3])
        upper_simplex = (vertices[0], vertices[1], vertices[2])
        glyph_triangle(axis, lower_simplex, color="#38bdf8", alpha=0.34)
        glyph_triangle(axis, upper_simplex, color="#fb7185", alpha=0.34)
        glyph_nodes(axis, vertices)
        axis.scatter([0.46], [0.50], marker="x", s=80, color="#dc2626", linewidth=2.2, zorder=6)
        axis.text(0.50, 0.86, "overlapping interiors", fontsize=8.5, ha="center", color="#991b1b")
    elif family_index == 3:
        axis.add_patch(_plotting_backend().rectangle((0.18, 0.22), 0.64, 0.56, facecolor="none", edgecolor="#64748b", linewidth=1.5))
        axis.scatter([0.24, 0.76], [0.42, 0.58], s=30, color="#0f172a")
        axis.annotate("", xy=(0.86, 0.58), xytext=(0.76, 0.58), arrowprops={"arrowstyle": "->", "color": "#dc2626"})
        axis.scatter([0.14], [0.58], s=55, facecolor="none", edgecolor="#dc2626", linestyle="--")
        axis.text(0.50, 0.84, "lift mismatch", fontsize=8.5, ha="center", color="#991b1b")
    elif family_index == 4:
        axis.add_patch(_plotting_backend().circle((0.50, 0.50), 0.30, fill=False, edgecolor="#64748b", linewidth=1.5))
        axis.plot([0.26, 0.76], [0.66, 0.62], color="#dc2626", linewidth=2.0)
        axis.scatter([0.50], [0.50], s=30, color="#0f172a")
        axis.text(0.50, 0.39, "center", fontsize=8, ha="center")
        axis.text(0.50, 0.82, "same side", fontsize=8, ha="center", color="#991b1b")
    else:
        axis.add_patch(_plotting_backend().rectangle((0.18, 0.36), 0.64, 0.30, facecolor="white", edgecolor="#64748b", linewidth=1.4))
        axis.text(0.36, 0.52, "model", fontsize=9, weight="bold", ha="center")
        axis.text(0.66, 0.52, "unsupported", fontsize=8.5, ha="center", color="#991b1b")
        glyph_cross(axis, (0.78, 0.52))


def draw_validation_family_glyph(axis: Any, level: int, family_index: int) -> None:
    """Dispatch to a distinct witness for every Level 1-4 validation family."""
    if level == 1:
        draw_level_1_glyph(axis, family_index)
    elif level == 2:
        draw_level_2_glyph(axis, family_index)
    elif level == 3:
        draw_level_3_glyph(axis, family_index)
    elif level == 4:
        draw_level_4_glyph(axis, family_index)
    else:
        raise ValueError(f"layer maps support Levels 1-4, got {level}")
    axis.text(0.12, 0.86, f"L{level}.{family_index + 1}", fontsize=8.5, weight="bold", color="#475569")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_axis_off()


def render_validation_layer_map(
    case: ValidationCase,
    output_path: Path,
    *,
    tracked_figure_dir: Path | None = None,
) -> None:
    """Render every implemented validation family owned by one layer."""
    layer = VALIDATION_HIERARCHY_LAYERS[case.level - 1]
    columns = 3
    rows = math.ceil(len(layer.questions) / columns)
    plotting = _plotting_backend()
    figure, axes = plotting.pyplot.subplots(rows, columns, figsize=(10.8, 2.8 * rows), facecolor="white", layout="constrained")
    flat_axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for family_index, (axis, question) in enumerate(zip(flat_axes, layer.questions, strict=False)):
        axis.set_facecolor(layer.color)
        draw_validation_family_glyph(axis, layer.level, family_index)
        axis.text(0.50, 0.06, wrapped_question(question), fontsize=9.2, color="#0f172a", ha="center", va="bottom")
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#cbd5e1")
            spine.set_linewidth(1.0)
    for axis in flat_axes[len(layer.questions) :]:
        axis.set_visible(False)
    figure.suptitle(
        f"Level {layer.level} — {layer.name}\nOwner: {layer.proof_owner}",
        fontsize=16,
        weight="bold",
        color="#0f172a",
    )
    save_figure_png(figure, output_path, tracked_figure_dir=tracked_figure_dir, dpi=220)
    plotting.pyplot.show()
    plotting.pyplot.close(figure)


def render_validation_case_figure(
    case: ValidationCase,
    output_path: Path,
    *,
    tracked_figure_dir: Path | None = None,
) -> None:
    """Render a complete layer map, retaining the focused Level 5 witness."""
    if case.level < 5:
        render_validation_layer_map(case, output_path, tracked_figure_dir=tracked_figure_dir)
        return
    layer = VALIDATION_HIERARCHY_LAYERS[4]
    plotting = _plotting_backend()
    figure, axis = plotting.pyplot.subplots(figsize=(4.8, 4.7), facecolor="white", layout="constrained")
    draw_visual(axis, case, show_point_labels=False)
    figure.suptitle(
        f"Level {layer.level} — {layer.name}\nOwner: {layer.proof_owner}",
        fontsize=12.5,
        weight="bold",
        color="#0f172a",
    )
    figure.supxlabel(
        wrapped_question(layer.questions[2]),
        fontsize=11.0,
        color="#0f172a",
    )
    save_figure_png(figure, output_path, tracked_figure_dir=tracked_figure_dir, dpi=240)
    plotting.pyplot.show()
    plotting.pyplot.close(figure)
