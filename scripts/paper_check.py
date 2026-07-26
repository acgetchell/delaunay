#!/usr/bin/env python3
"""Sanity-check generated paper PDFs."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

from pypdf import PdfReader
from pypdf.errors import PyPdfError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pypdf.generic import RectangleObject


class PdfInspectionError(RuntimeError):
    """Raised when a generated PDF cannot satisfy paper sanity checks."""


@dataclass(frozen=True, slots=True)
class PdfPageInspection:
    """Platform-stable facts extracted from one PDF page."""

    text: str
    media_box: tuple[float, float, float, float]
    crop_box: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class PdfInspection:
    """Extracted PDF facts used by sanity and reviewer-copy checks."""

    page_count: int
    text: str
    pages: tuple[PdfPageInspection, ...]


@dataclass(frozen=True, slots=True)
class PositivePageCount:
    """Validated positive page-count threshold for PDF sanity checks."""

    value: int

    def __post_init__(self) -> None:
        """Reject non-positive page-count thresholds."""
        if type(self.value) is not int or self.value <= 0:
            msg = f"expected a positive integer, got {self.value!r}"
            raise PdfInspectionError(msg)

    @classmethod
    def from_raw(cls, value: int) -> Self:
        """Parse a raw integer into a positive page-count threshold."""
        return cls(value)


@dataclass(frozen=True, slots=True)
class PdfCheckOptions:
    """Command-line options for PDF sanity checks."""

    pdf: Path
    min_pages: PositivePageCount
    required_text: tuple[str, ...]
    forbidden_text: tuple[str, ...]
    reference: Path | None = None

    def __post_init__(self) -> None:
        """Reject options that bypass the positive page-count parser."""
        if not isinstance(self.min_pages, PositivePageCount):
            msg = "min_pages must be a PositivePageCount"
            raise PdfInspectionError(msg)


def parse_positive_page_count(value: str) -> PositivePageCount:
    """Parse a positive integer command-line argument."""
    try:
        parsed = int(value)
    except ValueError as error:
        msg = f"expected a positive integer, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from error
    try:
        return PositivePageCount.from_raw(parsed)
    except PdfInspectionError as error:
        msg = f"expected a positive integer, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from error


def rectangle_coordinates(rectangle: RectangleObject) -> tuple[float, float, float, float]:
    """Return one PDF rectangle in a representation stable across pypdf objects."""
    return (
        float(rectangle.left),
        float(rectangle.bottom),
        float(rectangle.right),
        float(rectangle.top),
    )


def inspect_pdf(pdf: Path) -> PdfInspection:
    """Extract page count, text, and page geometry from a PDF."""
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

    pages: list[PdfPageInspection] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            pages.append(
                PdfPageInspection(
                    text=page.extract_text() or "",
                    media_box=rectangle_coordinates(page.mediabox),
                    crop_box=rectangle_coordinates(page.cropbox),
                )
            )
        except (PyPdfError, TypeError, ValueError) as error:
            msg = f"{pdf}: failed to inspect page {page_number}: {error}"
            raise PdfInspectionError(msg) from error
    return PdfInspection(
        page_count=len(pages),
        text="\n".join(page.text for page in pages),
        pages=tuple(pages),
    )


def compare_pdf_structure(generated: PdfInspection, reference: PdfInspection) -> list[str]:
    """Return structural differences that imply a stale reviewer copy."""
    failures: list[str] = []
    if generated.page_count != reference.page_count:
        failures.append(f"reference has {reference.page_count} page(s), rebuilt PDF has {generated.page_count}")
        return failures

    for page_number, (generated_page, reference_page) in enumerate(zip(generated.pages, reference.pages, strict=True), start=1):
        if generated_page.text != reference_page.text:
            failures.append(f"reference page {page_number} text differs from rebuilt PDF")
        if generated_page.media_box != reference_page.media_box:
            failures.append(f"reference page {page_number} media box differs from rebuilt PDF")
        if generated_page.crop_box != reference_page.crop_box:
            failures.append(f"reference page {page_number} crop box differs from rebuilt PDF")
    return failures


def check_pdf(options: PdfCheckOptions) -> PdfInspection:
    """Validate a generated paper PDF against lightweight reviewer-copy checks."""
    inspection = inspect_pdf(options.pdf)
    failures: list[str] = []

    if inspection.page_count < options.min_pages.value:
        failures.append(f"expected at least {options.min_pages.value} page(s), found {inspection.page_count}")

    failures.extend(f"missing required text: {required!r}" for required in options.required_text if required not in inspection.text)
    failures.extend(f"found forbidden text: {forbidden!r}" for forbidden in options.forbidden_text if forbidden in inspection.text)
    if options.reference is not None:
        reference = inspect_pdf(options.reference)
        failures.extend(compare_pdf_structure(inspection, reference))

    if failures:
        msg = f"{options.pdf}: " + "; ".join(failures)
        raise PdfInspectionError(msg)

    return inspection


def build_parser() -> argparse.ArgumentParser:
    """Build the paper PDF sanity-check command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="PDF reviewer copy to inspect")
    parser.add_argument("--min-pages", type=parse_positive_page_count, default=PositivePageCount.from_raw(1), help="minimum acceptable page count")
    parser.add_argument("--require-text", action="append", default=None, help="text that must appear in extracted PDF text")
    parser.add_argument("--forbid-text", action="append", default=None, help="text that must not appear in extracted PDF text")
    parser.add_argument(
        "--reference",
        type=Path,
        help="tracked reviewer PDF whose page text and geometry must match",
    )
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
        reference=namespace.reference,
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
