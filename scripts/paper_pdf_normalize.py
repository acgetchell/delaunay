#!/usr/bin/env python3
"""Normalize volatile Tectonic PDF metadata for reproducible reviewer copies."""

import argparse
import hashlib
import re
import sys
import tempfile
import uuid
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from paper_source_date import PaperSourceDate, PaperSourceDateError, read_paper_source_date

if TYPE_CHECKING:
    from collections.abc import Sequence

TRAILER_ID_RE = re.compile(rb"/ID\[\<[0-9A-Fa-f]{32}\>\<[0-9A-Fa-f]{32}\>\]")


class PdfNormalizationError(RuntimeError):
    """Raised when a PDF cannot be normalized reproducibly."""


class XmpIdentityKind(StrEnum):
    """Finite XMP identity fields that receive deterministic UUIDs."""

    DOCUMENT = "document"
    INSTANCE = "instance"


@dataclass(frozen=True, slots=True)
class PdfNormalization:
    """Facts produced while normalizing a reviewer PDF."""

    pdf: Path
    replacements: int


def stable_paper_identity(tex: Path) -> str:
    """Return a path-spelling-independent identity for paper metadata."""
    return f"papers/{tex.name}"


def stable_uuid(tex: Path, paper_date: PaperSourceDate, kind: XmpIdentityKind) -> str:
    """Return a stable UUID string for XMP identity fields."""
    name = f"{stable_paper_identity(tex)}:{paper_date.raw}:{kind.value}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


def stable_trailer_id(tex: Path, paper_date: PaperSourceDate) -> bytes:
    """Return the deterministic 16-byte trailer ID as lowercase hex."""
    source = f"{stable_paper_identity(tex)}:{paper_date.raw}:trailer-id".encode()
    return hashlib.sha256(source).hexdigest()[:32].encode()


def replace_required_regex(payload: bytes, regex: re.Pattern[bytes], replacement: bytes, *, label: str) -> tuple[bytes, int]:
    """Replace required same-length regex matches."""
    matches = list(regex.finditer(payload))
    if not matches:
        msg = f"{label}: expected metadata field was not found"
        raise PdfNormalizationError(msg)
    for match in matches:
        if len(match.group(0)) != len(replacement):
            msg = f"{label}: replacement length changed"
            raise PdfNormalizationError(msg)
    return regex.sub(replacement, payload), len(matches)


def normalize_pdf_bytes(payload: bytes, *, tex: Path, paper_date: PaperSourceDate) -> tuple[bytes, int]:
    """Return PDF bytes with volatile metadata normalized."""
    date_prefix = paper_date.instant.strftime("%Y-%m-%d").encode()
    timestamp_z = date_prefix + b"T00:00:00Z"
    timestamp = date_prefix + b"T00:00:00"
    document_uuid = stable_uuid(tex, paper_date, XmpIdentityKind.DOCUMENT)
    instance_uuid = stable_uuid(tex, paper_date, XmpIdentityKind.INSTANCE)
    trailer_id = stable_trailer_id(tex, paper_date)
    replacements = 0

    replacements_to_apply = (
        (
            re.compile(rb"<rdf:li>[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z</rdf:li>"),
            b"<rdf:li>" + timestamp_z + b"</rdf:li>",
            "dc:date",
        ),
        (
            re.compile(rb"<xmp:CreateDate>[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}</xmp:CreateDate>"),
            b"<xmp:CreateDate>" + timestamp + b"</xmp:CreateDate>",
            "xmp:CreateDate",
        ),
        (
            re.compile(rb"<xmp:ModifyDate>[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z</xmp:ModifyDate>"),
            b"<xmp:ModifyDate>" + timestamp_z + b"</xmp:ModifyDate>",
            "xmp:ModifyDate",
        ),
        (
            re.compile(rb"<xmp:MetadataDate>[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z</xmp:MetadataDate>"),
            b"<xmp:MetadataDate>" + timestamp_z + b"</xmp:MetadataDate>",
            "xmp:MetadataDate",
        ),
        (
            re.compile(rb"<xmpMM:DocumentID>uuid:[0-9A-Fa-f-]{36}</xmpMM:DocumentID>"),
            f"<xmpMM:DocumentID>uuid:{document_uuid}</xmpMM:DocumentID>".encode(),
            "xmpMM:DocumentID",
        ),
        (
            re.compile(rb"<xmpMM:InstanceID>uuid:[0-9A-Fa-f-]{36}</xmpMM:InstanceID>"),
            f"<xmpMM:InstanceID>uuid:{instance_uuid}</xmpMM:InstanceID>".encode(),
            "xmpMM:InstanceID",
        ),
        (
            TRAILER_ID_RE,
            b"/ID[<" + trailer_id + b"><" + trailer_id + b">]",
            "trailer ID",
        ),
    )

    normalized = payload
    for regex, replacement, label in replacements_to_apply:
        normalized, count = replace_required_regex(normalized, regex, replacement, label=label)
        replacements += count

    return normalized, replacements


def write_bytes_atomically(destination: Path, payload: bytes) -> None:
    """Write bytes through a same-directory temporary file and atomic replace."""
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
        temp_path.replace(destination)
    except OSError:
        if temp_path is not None:
            with suppress(OSError):
                temp_path.unlink(missing_ok=True)
        raise


def normalize_pdf(pdf: Path, *, tex: Path) -> PdfNormalization:
    """Normalize a PDF in place and return normalization facts."""
    try:
        paper_date = read_paper_source_date(tex)
        payload = pdf.read_bytes()
    except (OSError, PaperSourceDateError) as error:
        msg = f"failed to read normalization inputs: {error}"
        raise PdfNormalizationError(msg) from error

    normalized, replacements = normalize_pdf_bytes(payload, tex=tex, paper_date=paper_date)
    try:
        write_bytes_atomically(pdf, normalized)
    except OSError as error:
        msg = f"failed to write normalized PDF {pdf}: {error}"
        raise PdfNormalizationError(msg) from error

    return PdfNormalization(pdf=pdf, replacements=replacements)


def build_parser() -> argparse.ArgumentParser:
    """Build the PDF normalization command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="PDF reviewer copy to normalize in place")
    parser.add_argument("--tex", type=Path, required=True, help="TeX source declaring the reviewer-copy date")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Normalize a generated reviewer PDF."""
    parser = build_parser()
    namespace = parser.parse_args(argv)

    try:
        result = normalize_pdf(namespace.pdf, tex=namespace.tex)
    except PdfNormalizationError as error:
        print(f"paper-pdf-normalize: {error}", file=sys.stderr)
        return 1

    print(f"OK {result.pdf}: normalized {result.replacements} metadata field(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
