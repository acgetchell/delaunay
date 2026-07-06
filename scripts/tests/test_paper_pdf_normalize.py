"""Tests for deterministic paper PDF metadata normalization."""

from typing import TYPE_CHECKING

import pytest

from paper_pdf_normalize import PdfNormalizationError, XmpIdentityKind, main, normalize_pdf_bytes, stable_trailer_id, stable_uuid
from paper_source_date import read_paper_source_date

if TYPE_CHECKING:
    from pathlib import Path


def volatile_pdf_payload(tex: Path) -> bytes:
    document_uuid = "20a82bb6-7f7f-4200-a390-ae4182b6cef7"
    instance_uuid = "ef7a3ef2-67f7-4a39-85f7-5954c90aef20"
    trailer_id = b"972ec462635817b36da0723429d55f70"
    return b"\n".join(
        (
            b"<rdf:li>2026-07-06T10:36:09Z</rdf:li>",
            b"<xmp:CreateDate>2026-07-06T10:36:09</xmp:CreateDate>",
            b"<xmp:ModifyDate>2026-07-06T10:36:09Z</xmp:ModifyDate>",
            b"<xmp:MetadataDate>2026-07-06T10:36:09Z</xmp:MetadataDate>",
            f"<xmpMM:DocumentID>uuid:{document_uuid}</xmpMM:DocumentID>".encode(),
            f"<xmpMM:InstanceID>uuid:{instance_uuid}</xmpMM:InstanceID>".encode(),
            b"<</Type/XRef/ID[<" + trailer_id + b"><" + trailer_id + b">]/Root 1 0 R>>",
            f"% source marker {tex.name}".encode(),
        )
    )


def write_tex(tmp_path: Path) -> Path:
    tex = tmp_path / "validation.tex"
    tex.write_text(r"\date{July 6, 2026}", encoding="utf-8")
    return tex


def test_normalize_pdf_bytes_replaces_volatile_metadata(tmp_path: Path) -> None:
    tex = write_tex(tmp_path)
    paper_date = read_paper_source_date(tex)

    normalized, replacements = normalize_pdf_bytes(volatile_pdf_payload(tex), tex=tex, paper_date=paper_date)

    assert replacements == 7
    assert b"2026-07-06T10:36:09" not in normalized
    assert b"2026-07-06T00:00:00" in normalized
    assert stable_uuid(tex, paper_date, XmpIdentityKind.DOCUMENT).encode() in normalized
    assert stable_uuid(tex, paper_date, XmpIdentityKind.INSTANCE).encode() in normalized
    assert stable_trailer_id(tex, paper_date) in normalized


def test_stable_uuid_distinguishes_finite_xmp_identity_kinds(tmp_path: Path) -> None:
    tex = write_tex(tmp_path)
    paper_date = read_paper_source_date(tex)

    document_uuid = stable_uuid(tex, paper_date, XmpIdentityKind.DOCUMENT)
    instance_uuid = stable_uuid(tex, paper_date, XmpIdentityKind.INSTANCE)

    assert document_uuid != instance_uuid


def test_normalize_pdf_bytes_is_deterministic(tmp_path: Path) -> None:
    tex = write_tex(tmp_path)
    paper_date = read_paper_source_date(tex)
    first, first_count = normalize_pdf_bytes(volatile_pdf_payload(tex), tex=tex, paper_date=paper_date)
    second, second_count = normalize_pdf_bytes(volatile_pdf_payload(tex), tex=tex, paper_date=paper_date)

    assert first_count == second_count == 7
    assert first == second


def test_normalize_pdf_bytes_rejects_missing_metadata(tmp_path: Path) -> None:
    tex = write_tex(tmp_path)
    paper_date = read_paper_source_date(tex)

    with pytest.raises(PdfNormalizationError, match="expected metadata field"):
        normalize_pdf_bytes(b"%PDF-1.7\n", tex=tex, paper_date=paper_date)


def test_main_normalizes_pdf_in_place(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tex = write_tex(tmp_path)
    pdf = tmp_path / "validation.pdf"
    pdf.write_bytes(volatile_pdf_payload(tex))

    result = main([str(pdf), "--tex", str(tex)])

    captured = capsys.readouterr()
    assert result == 0
    assert "normalized 7 metadata field" in captured.out
    assert b"2026-07-06T10:36:09" not in pdf.read_bytes()
    assert captured.err == ""


def test_main_reports_normalization_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tex = write_tex(tmp_path)
    pdf = tmp_path / "validation.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")

    result = main([str(pdf), "--tex", str(tex)])

    captured = capsys.readouterr()
    assert result == 1
    assert "paper-pdf-normalize:" in captured.err
    assert captured.out == ""
