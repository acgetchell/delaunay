#!/usr/bin/env python3
"""Extract deterministic paper build timestamps from TeX sources."""

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

DATE_COMMAND_RE = re.compile(r"\\date\{(?P<date>[^{}]+)\}")
PAPER_DATE_FORMAT = "%B %d, %Y"


class PaperSourceDateError(RuntimeError):
    """Raised when a paper source does not declare a usable build date."""


@dataclass(frozen=True, slots=True)
class PaperSourceDate:
    """Parsed date metadata used for reproducible paper builds."""

    raw: str
    instant: datetime

    def __post_init__(self) -> None:
        """Verify that direct construction preserves the parsed-date invariant."""
        if self.raw != self.raw.strip():
            msg = f"paper source date must not contain leading or trailing whitespace: {self.raw!r}"
            raise PaperSourceDateError(msg)
        try:
            expected = datetime.strptime(self.raw, PAPER_DATE_FORMAT).replace(tzinfo=UTC)
        except ValueError as error:
            msg = f"paper source date must use 'Month day, year' format, got {self.raw!r}"
            raise PaperSourceDateError(msg) from error
        if self.instant.tzinfo is None or self.instant.utcoffset() != timedelta(0):
            msg = "paper source date instant must be timezone-aware UTC"
            raise PaperSourceDateError(msg)
        if self.instant != expected:
            msg = f"paper source date raw value {self.raw!r} and instant {self.instant.isoformat()} disagree"
            raise PaperSourceDateError(msg)

    @property
    def source_date_epoch(self) -> int:
        """Return the UTC epoch second for ``SOURCE_DATE_EPOCH``."""
        return int(self.instant.timestamp())


def parse_paper_source_date(source: str) -> PaperSourceDate:
    """Parse the first explicit ``\\date{Month day, year}`` command."""
    match = DATE_COMMAND_RE.search(source)
    if match is None:
        msg = r"paper source must declare an explicit \date{Month day, year}"
        raise PaperSourceDateError(msg)

    raw_date = match.group("date").strip()
    try:
        instant = datetime.strptime(raw_date, PAPER_DATE_FORMAT).replace(tzinfo=UTC)
    except ValueError as error:
        msg = f"paper source date must use 'Month day, year' format, got {raw_date!r}"
        raise PaperSourceDateError(msg) from error

    return PaperSourceDate(raw=raw_date, instant=instant)


def read_paper_source_date(path: Path) -> PaperSourceDate:
    """Read a TeX source file and parse its explicit paper date."""
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as error:
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
