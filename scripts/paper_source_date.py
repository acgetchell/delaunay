#!/usr/bin/env python3
"""Extract deterministic paper build timestamps from TeX sources."""

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from collections.abc import Sequence

DATE_COMMAND_RE = re.compile(r"\\date\{(?P<date>[^{}]+)\}")
PAPER_DATE_RE = re.compile(r"^(?P<month>[A-Z][a-z]+) (?P<day>[1-9]|[12][0-9]|3[01]), (?P<year>[0-9]{4})$")
PAPER_MONTH_BY_NAME = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}
PAPER_MONTH_NAME_BY_NUMBER = {number: name for name, number in PAPER_MONTH_BY_NAME.items()}


class PaperSourceDateError(RuntimeError):
    """Raised when a paper source does not declare a usable build date."""


def parse_raw_paper_date(raw: str) -> datetime:
    """Parse an English paper date without relying on process locale."""
    if raw != raw.strip():
        msg = f"paper source date must not contain leading or trailing whitespace: {raw!r}"
        raise PaperSourceDateError(msg)

    match = PAPER_DATE_RE.fullmatch(raw)
    if match is None:
        msg = f"paper source date must use 'Month day, year' format, got {raw!r}"
        raise PaperSourceDateError(msg)

    month = PAPER_MONTH_BY_NAME.get(match.group("month"))
    if month is None:
        msg = f"paper source date must use 'Month day, year' format, got {raw!r}"
        raise PaperSourceDateError(msg)

    try:
        return datetime(
            int(match.group("year")),
            month,
            int(match.group("day")),
            tzinfo=UTC,
        )
    except ValueError as error:
        msg = f"paper source date must use 'Month day, year' format, got {raw!r}"
        raise PaperSourceDateError(msg) from error


def format_paper_source_date(instant: datetime) -> str:
    """Format a UTC paper instant in the canonical TeX date form."""
    return f"{PAPER_MONTH_NAME_BY_NUMBER[instant.month]} {instant.day}, {instant.year:04d}"


def strip_tex_line_comment(line: str) -> str:
    """Return a TeX line with any unescaped percent comment removed."""
    backslash_count = 0
    for index, character in enumerate(line):
        if character == "%" and backslash_count % 2 == 0:
            return line[:index]
        if character == "\\":
            backslash_count += 1
        else:
            backslash_count = 0
    return line


def uncomment_tex_source(source: str) -> str:
    """Return source text with ordinary TeX line comments removed."""
    return "\n".join(strip_tex_line_comment(line) for line in source.splitlines())


@dataclass(frozen=True, slots=True)
class PaperSourceDate:
    """Parsed date metadata used for reproducible paper builds."""

    raw: str
    instant: datetime

    @classmethod
    def from_raw(cls, raw: str) -> Self:
        """Parse raw ``\\date`` text into deterministic paper date metadata."""
        return cls(raw=raw, instant=parse_raw_paper_date(raw))

    def __post_init__(self) -> None:
        """Verify that direct construction preserves the date metadata invariant."""
        if self.raw != self.raw.strip():
            msg = f"paper source date must not contain leading or trailing whitespace: {self.raw!r}"
            raise PaperSourceDateError(msg)
        if self.instant.tzinfo is None or self.instant.utcoffset() != timedelta(0):
            msg = "paper source date instant must be timezone-aware UTC"
            raise PaperSourceDateError(msg)
        expected = format_paper_source_date(self.instant)
        if self.raw != expected:
            msg = f"paper source date raw value {self.raw!r} and instant {self.instant.isoformat()} disagree"
            raise PaperSourceDateError(msg)

    @property
    def source_date_epoch(self) -> int:
        """Return the UTC epoch second for ``SOURCE_DATE_EPOCH``."""
        return int(self.instant.timestamp())


def parse_paper_source_date(source: str) -> PaperSourceDate:
    """Parse the only uncommented explicit ``\\date{Month day, year}`` command."""
    matches = list(DATE_COMMAND_RE.finditer(uncomment_tex_source(source)))
    if not matches:
        msg = r"paper source must declare an explicit \date{Month day, year}"
        raise PaperSourceDateError(msg)
    if len(matches) > 1:
        msg = rf"paper source must declare exactly one explicit \date{{Month day, year}}, found {len(matches)}"
        raise PaperSourceDateError(msg)

    raw_date = matches[0].group("date")
    return PaperSourceDate.from_raw(raw_date)


def read_paper_source_date(path: Path) -> PaperSourceDate:
    """Read a TeX source file and parse its explicit paper date."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        msg = f"failed to read paper source {path}: {error}"
        raise PaperSourceDateError(msg) from error
    return parse_paper_source_date(source)


def build_parser() -> argparse.ArgumentParser:
    """Build the paper source-date command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tex", type=Path, help="TeX paper source that declares an explicit \\date")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print the deterministic ``SOURCE_DATE_EPOCH`` for a TeX paper."""
    parser = build_parser()
    namespace = parser.parse_args(argv)

    try:
        paper_date = read_paper_source_date(namespace.tex)
    except PaperSourceDateError as error:
        print(f"paper-source-date-epoch: {error}", file=sys.stderr)
        return 1

    print(paper_date.source_date_epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
