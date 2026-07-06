#!/usr/bin/env python3
"""Sanity-check generated paper PDFs."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pypdf import PdfReader
from pypdf.errors import PyPdfError

if TYPE_CHECKING:
    from collections.abc import Sequence


class PdfInspectionError(RuntimeError):
    """Raised when a generated PDF cannot satisfy paper sanity checks."""


@dataclass(frozen=True, slots=True)
class PdfInspection:
    """Extracted PDF facts used by sanity checks."""

    page_count: int
    text: str


@dataclass(frozen=True, slots=True)
class PdfCheckOptions:
    """Command-line options for PDF sanity checks."""

    pdf: Path
    min_pages: int
    required_text: tuple[str, ...]
    forbidden_text: tuple[str, ...]


def parse_positive_int(value: str) -> int:
    """Parse a positive integer command-line argument."""
    try:
        parsed = int(value)
    except ValueError as error:
        msg = f"expected a positive integer, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from error
    if parsed <= 0:
        msg = f"expected a positive integer, got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def inspect_pdf(pdf: Path) -> PdfInspection:
    """Extract page count and text from a PDF."""
    if not pdf.is_file():
        msg = f"PDF does not exist: {pdf}"
        raise FileNotFoundError(msg)
    try:
        reader = PdfReader(str(pdf))
    except PyPdfError as error:
        msg = f"{pdf}: failed to read PDF: {error}"
        raise PdfInspectionError(msg) from error

    if reader.is_encrypted:
        msg = f"{pdf}: encrypted PDFs are not valid reviewer copies"
        raise PdfInspectionError(msg)

    text_parts: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text_parts.append(page.extract_text() or "")
        except PyPdfError as error:
            msg = f"{pdf}: failed to extract text from page {page_number}: {error}"
            raise PdfInspectionError(msg) from error
    return PdfInspection(page_count=len(reader.pages), text="\n".join(text_parts))


def check_pdf(options: PdfCheckOptions) -> PdfInspection:
    """Validate a generated paper PDF against lightweight reviewer-copy checks."""
    inspection = inspect_pdf(options.pdf)
    failures: list[str] = []

    if inspection.page_count < options.min_pages:
        failures.append(f"expected at least {options.min_pages} page(s), found {inspection.page_count}")

    failures.extend(f"missing required text: {required!r}" for required in options.required_text if required not in inspection.text)
    failures.extend(f"found forbidden text: {forbidden!r}" for forbidden in options.forbidden_text if forbidden in inspection.text)

    if failures:
        msg = f"{options.pdf}: " + "; ".join(failures)
        raise PdfInspectionError(msg)

    return inspection


def build_parser() -> argparse.ArgumentParser:
    """Build the paper PDF sanity-check command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="PDF reviewer copy to inspect")
    parser.add_argument("--min-pages", type=parse_positive_int, default=1, help="minimum acceptable page count")
    parser.add_argument("--require-text", action="append", default=None, help="text that must appear in extracted PDF text")
    parser.add_argument("--forbid-text", action="append", default=None, help="text that must not appear in extracted PDF text")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the paper PDF sanity checker."""
    parser = build_parser()
    namespace = parser.parse_args(argv)
    options = PdfCheckOptions(
        pdf=namespace.pdf,
        min_pages=namespace.min_pages,
        required_text=tuple(namespace.require_text or ()),
        forbidden_text=tuple(namespace.forbid_text or ()),
    )

    try:
        inspection = check_pdf(options)
    except (FileNotFoundError, OSError, PdfInspectionError) as error:
        print(f"paper-pdf-check: {error}", file=sys.stderr)
        return 1

    print(f"OK {options.pdf}: {inspection.page_count} page(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
