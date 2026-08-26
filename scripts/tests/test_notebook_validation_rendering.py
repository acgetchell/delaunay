"""Regression tests for validation-notebook witness checks and artifact rendering."""

import json
from pathlib import Path
from typing import Any

import pytest

import notebook_validation_rendering as rendering_module
from notebook_validation import CircumcircleWitness, ValidationCase, ValidationPoint, ValidationVisual
from notebook_validation_rendering import (
    EXPECTED_VALIDATION_FIGURE_NAMES,
    PNG_SIGNATURE,
    publish_validation_figure_set,
    render_validation_figure_set,
    save_figure_png,
    validate_case_visual_invariants,
    validate_validation_figure_set,
)

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
        public_reference="docs/construction_and_validation.md",
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
        path.write_bytes(PNG_SIGNATURE + b"figure")


def test_save_figure_png_writes_only_requested_scratch_path(tmp_path: Path) -> None:
    figure = FakeFigure()
    scratch = tmp_path / "scratch" / "validation.png"

    save_figure_png(figure, scratch)

    assert figure.saved_paths == [scratch]
    assert scratch.read_bytes().startswith(PNG_SIGNATURE)
    assert set(tmp_path.rglob("validation.png")) == {scratch}


def valid_validation_cases() -> tuple[ValidationCase, ...]:
    """Return independently valid witness fixtures for all five levels."""
    return (
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
    )


def write_validation_figure_set(directory: Path, marker: bytes, *, obsolete: bool = False) -> None:
    """Write one exact synthetic PNG set with optional obsolete owned output."""
    directory.mkdir(parents=True)
    for name in EXPECTED_VALIDATION_FIGURE_NAMES:
        (directory / name).write_bytes(PNG_SIGNATURE + marker + name.encode())
    if obsolete:
        (directory / "validation_model_failures.png").write_bytes(PNG_SIGNATURE + b"obsolete")


def directory_snapshot(directory: Path) -> dict[str, bytes]:
    """Return exact file bytes for publication rollback assertions."""
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


def install_fake_validation_renderers(monkeypatch: pytest.MonkeyPatch, *, fail_level: int | None = None) -> None:
    """Install deterministic PNG serializers with one optional injected failure."""

    def render_hierarchy(path: Path) -> None:
        path.write_bytes(PNG_SIGNATURE + b"hierarchy")

    def render_case(case: ValidationCase, path: Path) -> None:
        if case.level == fail_level:
            raise OSError(f"injected serialization failure at Level {case.level}")
        path.write_bytes(PNG_SIGNATURE + f"level-{case.level}".encode())

    monkeypatch.setattr(rendering_module, "render_validation_hierarchy_figure", render_hierarchy)
    monkeypatch.setattr(rendering_module, "render_validation_case_figure", render_case)


def test_validation_figure_set_rejects_missing_or_non_png_output(tmp_path: Path) -> None:
    """Publication requires every canonical name and a real PNG signature."""
    staged = tmp_path / "staged"
    write_validation_figure_set(staged, b"new")
    (staged / EXPECTED_VALIDATION_FIGURE_NAMES[-1]).unlink()

    with pytest.raises(ValueError, match="missing="):
        validate_validation_figure_set(staged)

    (staged / EXPECTED_VALIDATION_FIGURE_NAMES[-1]).write_bytes(b"not-png")
    with pytest.raises(ValueError, match="is not a PNG"):
        validate_validation_figure_set(staged)


def test_render_serialization_failure_preserves_previous_complete_sets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A staged render failure cannot change scratch or tracked publications."""
    scratch = tmp_path / "scratch"
    tracked = tmp_path / "tracked"
    write_validation_figure_set(scratch, b"old-scratch", obsolete=True)
    write_validation_figure_set(tracked, b"old-tracked", obsolete=True)
    scratch_before = directory_snapshot(scratch)
    tracked_before = directory_snapshot(tracked)
    install_fake_validation_renderers(monkeypatch, fail_level=3)

    with pytest.raises(OSError, match="serialization failure"):
        render_validation_figure_set(valid_validation_cases(), scratch, tracked_directory=tracked)

    assert directory_snapshot(scratch) == scratch_before
    assert directory_snapshot(tracked) == tracked_before


def test_staging_write_failure_preserves_previous_complete_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A copy failure leaves the published directory and obsolete files intact."""
    staged = tmp_path / "staged"
    destination = tmp_path / "published"
    write_validation_figure_set(staged, b"new")
    write_validation_figure_set(destination, b"old", obsolete=True)
    before = directory_snapshot(destination)
    original_copy = rendering_module._copy_file
    copy_count = 0

    def fail_third_copy(source: Path, target: Path) -> None:
        nonlocal copy_count
        copy_count += 1
        if copy_count == 3:
            message = "injected staged write failure"
            raise OSError(message)
        original_copy(source, target)

    monkeypatch.setattr(rendering_module, "_copy_file", fail_third_copy)

    with pytest.raises(OSError, match="staged write failure"):
        publish_validation_figure_set(staged, destination)

    assert directory_snapshot(destination) == before


def test_replace_failure_rolls_back_previous_complete_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed candidate swap restores the exact previous publication."""
    staged = tmp_path / "staged"
    destination = tmp_path / "published"
    write_validation_figure_set(staged, b"new")
    write_validation_figure_set(destination, b"old", obsolete=True)
    before = directory_snapshot(destination)
    original_replace = rendering_module._replace_path
    replace_count = 0

    def fail_candidate_replace(source: Path, target: Path) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            message = "injected candidate replace failure"
            raise OSError(message)
        original_replace(source, target)

    monkeypatch.setattr(rendering_module, "_replace_path", fail_candidate_replace)

    with pytest.raises(OSError, match="candidate replace failure"):
        publish_validation_figure_set(staged, destination)

    assert replace_count == 3
    assert directory_snapshot(destination) == before


def test_successful_publication_replaces_complete_set_and_removes_obsolete_names(tmp_path: Path) -> None:
    """Obsolete owned names disappear only after the complete candidate swaps in."""
    staged = tmp_path / "staged"
    destination = tmp_path / "published"
    write_validation_figure_set(staged, b"new")
    write_validation_figure_set(destination, b"old", obsolete=True)

    published = publish_validation_figure_set(staged, destination)

    assert published == tuple(destination / name for name in EXPECTED_VALIDATION_FIGURE_NAMES)
    assert set(directory_snapshot(destination)) == set(EXPECTED_VALIDATION_FIGURE_NAMES)
    assert all(path.read_bytes().startswith(PNG_SIGNATURE + b"new") for path in published)


def test_render_validation_figure_set_publishes_exact_scratch_and_tracked_sets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The high-level workflow renders once and publishes two complete sets."""
    scratch = tmp_path / "scratch"
    tracked = tmp_path / "tracked"
    install_fake_validation_renderers(monkeypatch)

    rendered = render_validation_figure_set(valid_validation_cases(), scratch, tracked_directory=tracked)

    assert rendered == tuple(scratch / name for name in EXPECTED_VALIDATION_FIGURE_NAMES)
    validate_validation_figure_set(scratch)
    validate_validation_figure_set(tracked)


def test_validation_notebook_keeps_render_cell_as_orchestration() -> None:
    notebook = json.loads((REPO_ROOT / "notebooks" / "01_validation.ipynb").read_text(encoding="utf-8"))
    render_cell = next(cell for cell in notebook["cells"] if cell.get("id") == "render-validation-figures")
    source = "".join(render_cell["source"])

    assert len(source.splitlines()) <= 30
    assert "render_validation_figure_set" in source
    assert "tracked_directory=DOC_FIGURE_DIR" in source
    assert "render_validation_case_figure" not in source
    assert "def " not in source
