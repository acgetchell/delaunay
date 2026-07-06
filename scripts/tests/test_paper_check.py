"""Tests for the paper PDF sanity checker."""

from typing import TYPE_CHECKING

import pytest
from pypdf.errors import PyPdfError

from paper_check import PdfCheckOptions, PdfInspectionError, check_pdf, main

if TYPE_CHECKING:
    from pathlib import Path


class FakePage:
    """Minimal pypdf page stand-in."""

    def __init__(self, text: str) -> None:
        """Create a fake page with extractable text."""
        self._text = text

    def extract_text(self) -> str:
        """Return fake extracted page text."""
        return self._text


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

    def extract_text(self) -> str:
        """Raise a fake pypdf extraction error."""
        msg = "cannot extract text"
        raise PyPdfError(msg)


def write_pdf_stub(tmp_path: Path) -> Path:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.7\n")
    return path


def test_check_pdf_accepts_required_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = write_pdf_stub(tmp_path)

    def fake_reader(path: str) -> FakeReader:
        return FakeReader(path, text="Validation Architecture in delaunay\nReferences")

    monkeypatch.setattr("paper_check.PdfReader", fake_reader)

    inspection = check_pdf(
        PdfCheckOptions(
            pdf=pdf,
            min_pages=1,
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
                min_pages=1,
                required_text=("Validation Architecture in delaunay",),
                forbidden_text=(),
            )
        )


def test_check_pdf_rejects_short_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = write_pdf_stub(tmp_path)

    monkeypatch.setattr("paper_check.PdfReader", FakeReader)

    with pytest.raises(PdfInspectionError, match="expected at least 2 page"):
        check_pdf(PdfCheckOptions(pdf=pdf, min_pages=2, required_text=(), forbidden_text=()))


def test_check_pdf_rejects_forbidden_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = write_pdf_stub(tmp_path)

    def fake_reader(path: str) -> FakeReader:
        return FakeReader(path, text=r"Validation Architecture in delaunay \today")

    monkeypatch.setattr("paper_check.PdfReader", fake_reader)

    with pytest.raises(PdfInspectionError, match="found forbidden text"):
        check_pdf(
            PdfCheckOptions(
                pdf=pdf,
                min_pages=1,
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
        check_pdf(PdfCheckOptions(pdf=pdf, min_pages=1, required_text=(), forbidden_text=()))


def test_check_pdf_wraps_pypdf_reader_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = write_pdf_stub(tmp_path)

    monkeypatch.setattr("paper_check.PdfReader", raise_pypdf_error)

    with pytest.raises(PdfInspectionError, match="failed to read PDF"):
        check_pdf(PdfCheckOptions(pdf=pdf, min_pages=1, required_text=(), forbidden_text=()))


def test_check_pdf_wraps_pypdf_text_extraction_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = write_pdf_stub(tmp_path)

    monkeypatch.setattr("paper_check.PdfReader", FakeReaderWithBrokenPage)

    with pytest.raises(PdfInspectionError, match="failed to extract text from page 1"):
        check_pdf(PdfCheckOptions(pdf=pdf, min_pages=1, required_text=(), forbidden_text=()))


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


def test_main_reports_missing_pdf(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    result = main([str(tmp_path / "missing.pdf")])

    assert result == 1
    assert "PDF does not exist" in capsys.readouterr().err
