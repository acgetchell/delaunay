"""Tests for deterministic paper source-date extraction."""

from datetime import UTC, datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pytest

from paper_source_date import PaperSourceDate, PaperSourceDateError, main, parse_paper_source_date, read_paper_source_date

if TYPE_CHECKING:
    from pathlib import Path


def test_parse_paper_source_date_returns_utc_epoch() -> None:
    paper_date = parse_paper_source_date(r"\title{Example}\date{July 6, 2026}")

    assert paper_date.raw == "July 6, 2026"
    assert paper_date.source_date_epoch == 1783296000


def test_parse_paper_source_date_accepts_single_digit_day() -> None:
    paper_date = parse_paper_source_date(r"\date{July 6, 2026}")

    assert paper_date.source_date_epoch == 1783296000


def test_parse_paper_source_date_rejects_non_english_month() -> None:
    with pytest.raises(PaperSourceDateError, match="Month day, year"):
        parse_paper_source_date(r"\date{Notamonth 6, 2026}")


def test_parse_paper_source_date_rejects_leading_zero_day() -> None:
    with pytest.raises(PaperSourceDateError, match="Month day, year"):
        parse_paper_source_date(r"\date{July 06, 2026}")


def test_parse_paper_source_date_ignores_commented_date_before_real_date() -> None:
    source = (
        r"% \date{January 1, 1999}"
        "\n"
        r"\title{Example}"
        "\n"
        r"\date{July 6, 2026}"
    )

    paper_date = parse_paper_source_date(source)

    assert paper_date.raw == "July 6, 2026"
    assert paper_date.source_date_epoch == 1783296000


def test_parse_paper_source_date_rejects_multiple_uncommented_dates() -> None:
    source = r"\date{July 6, 2026}" "\n" r"\date{July 7, 2026}"

    with pytest.raises(PaperSourceDateError, match="exactly one explicit"):
        parse_paper_source_date(source)


def test_parse_paper_source_date_rejects_surrounding_date_whitespace() -> None:
    with pytest.raises(PaperSourceDateError, match="leading or trailing whitespace"):
        parse_paper_source_date(r"\date{ July 6, 2026 }")


def test_parse_paper_source_date_accepts_valid_leap_day() -> None:
    paper_date = parse_paper_source_date(r"\date{February 29, 2024}")

    assert paper_date.instant == datetime(2024, 2, 29, tzinfo=UTC)


@pytest.mark.parametrize(
    "source",
    [
        r"\date{February 29, 2025}",
        r"\date{April 31, 2026}",
    ],
)
def test_parse_paper_source_date_rejects_invalid_calendar_dates(source: str) -> None:
    with pytest.raises(PaperSourceDateError, match="Month day, year"):
        parse_paper_source_date(source)


def test_paper_source_date_rejects_non_utc_instant() -> None:
    with pytest.raises(PaperSourceDateError, match="timezone-aware UTC"):
        PaperSourceDate(raw="July 6, 2026", instant=datetime(2026, 7, 6, tzinfo=timezone(timedelta(hours=1))))


def test_paper_source_date_rejects_mismatched_raw_and_instant() -> None:
    with pytest.raises(PaperSourceDateError, match="disagree"):
        PaperSourceDate(raw="July 6, 2026", instant=datetime(2026, 7, 7, tzinfo=UTC))


def test_paper_source_date_rejects_unstripped_raw_date() -> None:
    with pytest.raises(PaperSourceDateError, match="leading or trailing whitespace"):
        PaperSourceDate(raw=" July 6, 2026", instant=datetime(2026, 7, 6, tzinfo=UTC))


def test_parse_paper_source_date_rejects_missing_date() -> None:
    with pytest.raises(PaperSourceDateError, match=r"explicit \\date"):
        parse_paper_source_date(r"\title{Example}")


def test_parse_paper_source_date_rejects_malformed_date() -> None:
    with pytest.raises(PaperSourceDateError, match="Month day, year"):
        parse_paper_source_date(r"\date{2026-07-06}")


def test_read_paper_source_date_wraps_read_errors(tmp_path: Path) -> None:
    missing = tmp_path / "missing.tex"

    with pytest.raises(PaperSourceDateError, match="failed to read paper source"):
        read_paper_source_date(missing)


def test_read_paper_source_date_wraps_decode_errors(tmp_path: Path) -> None:
    source = tmp_path / "paper.tex"
    source.write_bytes(b"\\date{July 6, 2026}\xff")

    with pytest.raises(PaperSourceDateError, match="failed to read paper source"):
        read_paper_source_date(source)


def test_main_prints_source_date_epoch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "paper.tex"
    source.write_text(r"\date{July 6, 2026}", encoding="utf-8")

    result = main([str(source)])

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == "1783296000\n"
    assert captured.err == ""


def test_main_reports_source_date_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "paper.tex"
    source.write_text(r"\date{July 6 2026}", encoding="utf-8")

    result = main([str(source)])

    captured = capsys.readouterr()
    assert result == 1
    assert "paper-source-date-epoch:" in captured.err
    assert captured.out == ""
