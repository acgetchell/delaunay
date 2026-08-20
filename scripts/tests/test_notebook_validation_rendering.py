"""Regression tests for validation-notebook witness checks and artifact rendering."""

import json
from pathlib import Path
from typing import Any

import pytest

from notebook_validation import CircumcircleWitness, ValidationCase, ValidationPoint, ValidationVisual
from notebook_validation_rendering import save_figure_png, validate_case_visual_invariants

REPO_ROOT = Path(__file__).resolve().parents[2]


def visual_case(  # noqa: PLR0913 - compact builder for varied witness fixtures
    level: int,
    points: tuple[tuple[float, float], ...],
    simplices: tuple[tuple[int, ...], ...],
    *,
    highlighted_simplices: frozenset[int] = frozenset(),
    invalid_points: frozenset[int] = frozenset(),
    isolated_points: frozenset[int] = frozenset(),
    duplicate_simplices: tuple[tuple[int, ...], ...] = (),
    circumcircle: CircumcircleWitness | None = None,
) -> ValidationCase:
    """Build one parsed validation case for independent witness tests."""
    visual = ValidationVisual(
        points=tuple(ValidationPoint(label=f"p{index}", coordinates=point) for index, point in enumerate(points)),
        simplices=simplices,
        highlighted_simplices=highlighted_simplices,
        highlighted_edges=(),
        invalid_points=invalid_points,
        isolated_points=isolated_points,
        duplicate_simplices=duplicate_simplices,
        circumcircle=circumcircle,
    )
    return ValidationCase(
        level=level,
        layer=f"Level {level}",
        title=f"case {level}",
        status="invalid",
        public_check="validate",
        public_reference="docs/validation.md",
        input_summary="fixture",
        explanation="fixture",
        diagnostic="fixture",
        visual=visual,
    )


@pytest.mark.parametrize(
    "case",
    [
        visual_case(1, ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), ((0, 1, 2),), invalid_points=frozenset({0})),
        visual_case(
            2,
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
            ((0, 1, 2), (2, 1, 0)),
            duplicate_simplices=((0, 1, 2),),
        ),
        visual_case(
            3,
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (2.0, 2.0)),
            ((0, 1, 2),),
            isolated_points=frozenset({3}),
        ),
        visual_case(
            4,
            ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),
            ((0, 1, 2),),
            highlighted_simplices=frozenset({0}),
        ),
        visual_case(
            5,
            ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, 0.0)),
            ((0, 1, 2),),
            highlighted_simplices=frozenset({0}),
            invalid_points=frozenset({3}),
            circumcircle=CircumcircleWitness(center=(0.0, 0.0), radius=1.0),
        ),
    ],
)
def test_visual_witnesses_cover_all_validation_levels(case: ValidationCase) -> None:
    validate_case_visual_invariants(case, case.level - 1)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        (
            visual_case(1, ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), ((0, 1, 2),)),
            "must mark the invalid point",
        ),
        (
            visual_case(
                2,
                ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
                ((0, 1, 2),),
                duplicate_simplices=((0, 1, 2),),
            ),
            "does not duplicate an emitted simplex",
        ),
        (
            visual_case(
                3,
                ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
                ((0, 1, 2),),
                isolated_points=frozenset({0}),
            ),
            "contains used point index",
        ),
        (
            visual_case(
                4,
                ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
                ((0, 1, 2),),
                highlighted_simplices=frozenset({0}),
            ),
            "exceeds degeneracy tolerance",
        ),
        (
            visual_case(
                5,
                ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -2.0)),
                ((0, 1, 2),),
                highlighted_simplices=frozenset({0}),
                invalid_points=frozenset({3}),
                circumcircle=CircumcircleWitness(center=(0.0, 0.0), radius=1.0),
            ),
            "is not strictly inside the circumcircle",
        ),
    ],
)
def test_visual_witnesses_reject_false_evidence(case: ValidationCase, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_case_visual_invariants(case, case.level - 1)


@pytest.mark.parametrize(
    "case",
    [
        visual_case(
            4,
            ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),
            ((0, 1, 2),),
            highlighted_simplices=frozenset({1}),
        ),
        visual_case(
            5,
            ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, 0.0)),
            ((0, 1, 2),),
            highlighted_simplices=frozenset({1}),
            invalid_points=frozenset({3}),
            circumcircle=CircumcircleWitness(center=(0.0, 0.0), radius=1.0),
        ),
    ],
)
def test_visual_witnesses_reject_out_of_range_highlighted_simplex(case: ValidationCase) -> None:
    with pytest.raises(IndexError, match=r"cases\[7\]\.visual\.simplices index 1 is outside 0\.\.0"):
        validate_case_visual_invariants(case, 7)


class FakeFigure:
    """Minimal figure seam for deterministic output-path tests."""

    def __init__(self) -> None:
        """Create an empty list of observed output paths."""
        self.saved_paths: list[Path] = []

    @staticmethod
    def get_facecolor() -> str:
        return "white"

    def savefig(self, path: Path, **_kwargs: Any) -> None:
        self.saved_paths.append(path)
        path.write_bytes(b"png")


def test_save_figure_png_writes_scratch_and_explicit_tracked_copy(tmp_path: Path) -> None:
    figure = FakeFigure()
    scratch = tmp_path / "scratch" / "validation.png"
    tracked_dir = tmp_path / "tracked"

    save_figure_png(figure, scratch, tracked_figure_dir=tracked_dir)

    assert figure.saved_paths == [scratch, tracked_dir / scratch.name]
    assert all(path.read_bytes() == b"png" for path in figure.saved_paths)


def test_save_figure_png_writes_only_scratch_without_tracked_directory(tmp_path: Path) -> None:
    figure = FakeFigure()
    scratch = tmp_path / "scratch" / "validation.png"

    save_figure_png(figure, scratch)

    assert figure.saved_paths == [scratch]
    assert scratch.read_bytes() == b"png"
    assert set(tmp_path.rglob("validation.png")) == {scratch}


def test_validation_notebook_keeps_render_cell_as_orchestration() -> None:
    notebook = json.loads((REPO_ROOT / "notebooks" / "01_validation.ipynb").read_text(encoding="utf-8"))
    render_cell = next(cell for cell in notebook["cells"] if cell.get("id") == "render-validation-figures")
    source = "".join(render_cell["source"])

    assert len(source.splitlines()) <= 30
    assert "validate_case_visual_invariants" in source
    assert "render_validation_case_figure" in source
    assert "def " not in source
