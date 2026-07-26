"""Tests for the paper PDF sanity checker."""

from pathlib import Path
from typing import cast

import pytest
from pypdf.errors import PyPdfError

from paper_check import PdfCheckOptions, PdfInspectionError, PositivePageCount, check_pdf, main


class FakePage:
    """Minimal pypdf page stand-in."""

    def __init__(self, text: str, *, media_box: tuple[float, float, float, float] = (0.0, 0.0, 612.0, 792.0)) -> None:
        """Create a fake page with extractable text."""
        self._text = text
        self.mediabox = FakeRectangle(media_box)
        self.cropbox = FakeRectangle(media_box)

    def extract_text(self) -> str:
        """Return fake extracted page text."""
        return self._text


class FakeRectangle:
    """Minimal pypdf rectangle stand-in."""

    def __init__(self, coordinates: tuple[float, float, float, float]) -> None:
        """Create one fake page rectangle."""
        self.left, self.bottom, self.right, self.top = coordinates


class FakeReader:
    """Minimal pypdf reader stand-in."""

    def __init__(self, _path: str, *, text: str = "Validation Architecture in delaunay", encrypted: bool = False) -> None:
        """Create a fake PDF reader."""
        self.is_encrypted = encrypted
        self.pages = [FakePage(text)]


class FakeReaderWithBrokenPage:
    """Minimal pypdf reader that fails during text extraction."""

    is_encrypted = False

    def __init__(self, _path: str) -> None:
        """Create a fake PDF reader with one broken page."""
        self.pages = [BrokenPage()]


def raise_pypdf_error(_path: str) -> FakeReader:
    """Raise a fake pypdf read error from reader construction."""
    msg = "cannot read PDF"
    raise PyPdfError(msg)


class BrokenPage:
    """Minimal pypdf page stand-in that raises an extraction error."""

    mediabox = FakeRectangle((0.0, 0.0, 612.0, 792.0))
    cropbox = mediabox

    def extract_text(self) -> str:
        """Raise a fake pypdf extraction error."""
        msg = "cannot extract text"
        raise PyPdfError(msg)


def write_pdf_stub(tmp_path: Path) -> Path:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.7\n")
    return path


def min_pages(value: int) -> PositivePageCount:
    """Return a validated PDF page-count threshold."""
    return PositivePageCount.from_raw(value)


def test_check_pdf_accepts_required_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = write_pdf_stub(tmp_path)

    def fake_reader(path: str) -> FakeReader:
        return FakeReader(path, text="Validation Architecture in delaunay\nReferences")

    monkeypatch.setattr("paper_check.PdfReader", fake_reader)

    inspection = check_pdf(
        PdfCheckOptions(
            pdf=pdf,
            min_pages=min_pages(1),
            required_text=("Validation Architecture in delaunay",),
            forbidden_text=(r"\today",),
        )
    )

    assert inspection.page_count == 1


def test_check_pdf_rejects_missing_required_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = write_pdf_stub(tmp_path)

    def fake_reader(path: str) -> FakeReader:
        return FakeReader(path, text="Different title")

    monkeypatch.setattr("paper_check.PdfReader", fake_reader)

    with pytest.raises(PdfInspectionError, match="missing required text"):
        check_pdf(
            PdfCheckOptions(
                pdf=pdf,
                min_pages=min_pages(1),
                required_text=("Validation Architecture in delaunay",),
                forbidden_text=(),
            )
        )


def test_check_pdf_rejects_short_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = write_pdf_stub(tmp_path)

    monkeypatch.setattr("paper_check.PdfReader", FakeReader)

    with pytest.raises(PdfInspectionError, match="expected at least 2 page"):
        check_pdf(PdfCheckOptions(pdf=pdf, min_pages=min_pages(2), required_text=(), forbidden_text=()))


def test_check_pdf_rejects_forbidden_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = write_pdf_stub(tmp_path)

    def fake_reader(path: str) -> FakeReader:
        return FakeReader(path, text=r"Validation Architecture in delaunay \today")

    monkeypatch.setattr("paper_check.PdfReader", fake_reader)

    with pytest.raises(PdfInspectionError, match="found forbidden text"):
        check_pdf(
            PdfCheckOptions(
                pdf=pdf,
                min_pages=min_pages(1),
                required_text=("Validation Architecture in delaunay",),
                forbidden_text=(r"\today",),
            )
        )


def test_check_pdf_rejects_encrypted_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = write_pdf_stub(tmp_path)

    def fake_reader(path: str) -> FakeReader:
        return FakeReader(path, encrypted=True)

    monkeypatch.setattr("paper_check.PdfReader", fake_reader)

    with pytest.raises(PdfInspectionError, match="encrypted PDFs"):
        check_pdf(PdfCheckOptions(pdf=pdf, min_pages=min_pages(1), required_text=(), forbidden_text=()))


def test_check_pdf_wraps_pypdf_reader_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = write_pdf_stub(tmp_path)

    monkeypatch.setattr("paper_check.PdfReader", raise_pypdf_error)

    with pytest.raises(PdfInspectionError, match="failed to read PDF"):
        check_pdf(PdfCheckOptions(pdf=pdf, min_pages=min_pages(1), required_text=(), forbidden_text=()))


def test_check_pdf_wraps_pypdf_page_inspection_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = write_pdf_stub(tmp_path)

    monkeypatch.setattr("paper_check.PdfReader", FakeReaderWithBrokenPage)

    with pytest.raises(PdfInspectionError, match="failed to inspect page 1"):
        check_pdf(PdfCheckOptions(pdf=pdf, min_pages=min_pages(1), required_text=(), forbidden_text=()))


def test_check_pdf_accepts_structurally_equivalent_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = write_pdf_stub(tmp_path)
    reference = tmp_path / "reference.pdf"
    reference.write_bytes(b"%PDF-1.7\n")
    monkeypatch.setattr("paper_check.PdfReader", FakeReader)

    inspection = check_pdf(
        PdfCheckOptions(
            pdf=pdf,
            min_pages=min_pages(1),
            required_text=(),
            forbidden_text=(),
            reference=reference,
        )
    )

    assert inspection.page_count == 1


def test_check_pdf_rejects_reference_with_different_page_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = write_pdf_stub(tmp_path)
    reference = tmp_path / "reference.pdf"
    reference.write_bytes(b"%PDF-1.7\n")

    def fake_reader(path: str) -> FakeReader:
        text = "stale reviewer text" if Path(path) == reference else "rebuilt paper text"
        return FakeReader(path, text=text)

    monkeypatch.setattr("paper_check.PdfReader", fake_reader)

    with pytest.raises(PdfInspectionError, match="reference page 1 text differs"):
        check_pdf(
            PdfCheckOptions(
                pdf=pdf,
                min_pages=min_pages(1),
                required_text=(),
                forbidden_text=(),
                reference=reference,
            )
        )


def test_check_pdf_rejects_reference_with_different_page_geometry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = write_pdf_stub(tmp_path)
    reference = tmp_path / "reference.pdf"
    reference.write_bytes(b"%PDF-1.7\n")

    def fake_reader(path: str) -> FakeReader:
        reader = FakeReader(path)
        if Path(path) == reference:
            reader.pages = [FakePage("Validation Architecture in delaunay", media_box=(0.0, 0.0, 600.0, 792.0))]
        return reader

    monkeypatch.setattr("paper_check.PdfReader", fake_reader)

    with pytest.raises(PdfInspectionError, match="reference page 1 media box differs"):
        check_pdf(
            PdfCheckOptions(
                pdf=pdf,
                min_pages=min_pages(1),
                required_text=(),
                forbidden_text=(),
                reference=reference,
            )
        )


def test_main_reports_success_to_stdout(capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = write_pdf_stub(tmp_path)

    def fake_reader(path: str) -> FakeReader:
        return FakeReader(path, text="Validation Architecture in delaunay")

    monkeypatch.setattr("paper_check.PdfReader", fake_reader)

    result = main([str(pdf), "--require-text", "Validation Architecture in delaunay", "--forbid-text", r"\today"])

    captured = capsys.readouterr()
    assert result == 0
    assert f"OK {pdf}: 1 page(s)" in captured.out
    assert captured.err == ""


def test_main_rejects_invalid_min_pages(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    pdf = write_pdf_stub(tmp_path)

    with pytest.raises(SystemExit) as error:
        main([str(pdf), "--min-pages", "0"])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert "expected a positive integer" in captured.err
    assert captured.out == ""


def test_positive_page_count_rejects_direct_non_positive_values() -> None:
    with pytest.raises(PdfInspectionError, match="expected a positive integer"):
        PositivePageCount.from_raw(0)


def test_positive_page_count_rejects_bool_value() -> None:
    truthy_value = True

    with pytest.raises(PdfInspectionError, match="expected a positive integer"):
        PositivePageCount.from_raw(cast("int", truthy_value))


def test_check_options_reject_raw_min_pages(tmp_path: Path) -> None:
    pdf = write_pdf_stub(tmp_path)

    with pytest.raises(PdfInspectionError, match="min_pages must be a PositivePageCount"):
        PdfCheckOptions(pdf=pdf, min_pages=cast("PositivePageCount", 0), required_text=(), forbidden_text=())


def test_main_reports_missing_pdf(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    result = main([str(tmp_path / "missing.pdf")])

    assert result == 1
    assert "PDF does not exist" in capsys.readouterr().err
