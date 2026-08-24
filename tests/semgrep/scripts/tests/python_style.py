"""Fixtures for repository Python style rules."""

# ruleid: delaunay.python.no-future-annotations-on-python314
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

# ok: delaunay.python.no-future-annotations-on-python314
from typing import TYPE_CHECKING


def direct_process_spawning() -> None:
    """Exercise direct process-spawning APIs that bypass the wrapper."""
    # ruleid: delaunay.python.no-direct-subprocess-run-outside-wrapper
    subprocess.run(["true"], check=True)
    # ruleid: delaunay.python.no-direct-subprocess-run-outside-wrapper
    subprocess.Popen(["true"])
    # ruleid: delaunay.python.no-direct-subprocess-run-outside-wrapper
    subprocess.call(["true"])
    # ruleid: delaunay.python.no-direct-subprocess-run-outside-wrapper
    subprocess.check_call(["true"])
    # ruleid: delaunay.python.no-direct-subprocess-run-outside-wrapper
    subprocess.check_output(["true"])
    # ruleid: delaunay.python.no-direct-subprocess-run-outside-wrapper
    subprocess.getoutput("true")
    # ruleid: delaunay.python.no-direct-subprocess-run-outside-wrapper
    subprocess.getstatusoutput("true")
    # ruleid: delaunay.python.no-direct-subprocess-run-outside-wrapper
    os.system("true")


def wrapped_process_spawning() -> None:
    """Document that the repository wrapper remains allowed."""
    # ok: delaunay.python.no-direct-subprocess-run-outside-wrapper
    run_safe_command("true", [])


def tar_extraction(archive: object, target: Path) -> None:
    """Exercise unsafe and filtered tar extraction calls."""
    # ruleid: delaunay.python.safe-tar-extraction
    archive.extractall(target)
    # ruleid: delaunay.python.safe-tar-extraction
    archive.extract("member", target)
    # ok: delaunay.python.safe-tar-extraction
    archive.extractall(target, filter="data")
    # ok: delaunay.python.safe-tar-extraction
    archive.extract("member", target, filter="data")


def performance_artifact_writes(path: Path) -> None:
    """Exercise direct destination writes and a staged temporary write."""
    # ruleid: delaunay.python.performance-artifact-writes-are-transactional
    path.write_text("payload", encoding="utf-8")
    # ruleid: delaunay.python.performance-artifact-writes-are-transactional
    path.write_bytes(b"payload")
    # ruleid: delaunay.python.performance-artifact-writes-are-transactional
    path.open("wb")
    # ruleid: delaunay.python.performance-artifact-writes-are-transactional
    open(path, "w", encoding="utf-8")
    # ruleid: delaunay.python.performance-artifact-writes-are-transactional
    path.open(mode="a", encoding="utf-8")
    # ruleid: delaunay.python.performance-artifact-writes-are-transactional
    open(path, mode="x+b")
    # ok: delaunay.python.performance-artifact-writes-are-transactional
    path.open(mode="rb")
    # ok: delaunay.python.performance-artifact-writes-are-transactional
    open(path, mode="r", encoding="utf-8")
    # ok: delaunay.python.performance-artifact-writes-are-transactional
    tempfile.NamedTemporaryFile("wb", dir=path.parent)


def parquet_promotion_inputs(path: Path) -> None:
    """Exercise Parquet readers that would add a second promotion contract."""
    # ruleid: delaunay.python.performance-promotion-is-csv-only
    pl.read_parquet(path)
    # ruleid: delaunay.python.performance-promotion-is-csv-only
    pandas.read_parquet(path)
    # ruleid: delaunay.python.performance-promotion-is-csv-only
    pl.scan_parquet(path)
    # ruleid: delaunay.python.performance-promotion-is-csv-only
    pq.read_table(path)
    # ruleid: delaunay.python.performance-promotion-is-csv-only
    pyarrow.parquet.ParquetFile(path)
    # ok: delaunay.python.performance-promotion-is-csv-only
    csv.DictReader(path.open(encoding="utf-8"))
