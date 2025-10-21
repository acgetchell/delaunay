#!/usr/bin/env python3
# ruff: noqa: SLF001
"""
Test suite for compare_storage_backends.py module.

Tests storage backend comparison functionality including:
- Criterion output parsing (JSON and regex fallback)
- Benchmark execution with different backends
- Comparison report generation
- Development mode and extra args handling
"""

import json
import os
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from compare_storage_backends import StorageBackendComparator, find_project_root


@pytest.fixture
def temp_project_root(tmp_path):
    """Create a temporary project root with necessary directories."""
    project_root = tmp_path / "project"
    project_root.mkdir()

    # Create target/criterion directory structure
    criterion_dir = project_root / "target" / "criterion"
    criterion_dir.mkdir(parents=True)

    # Create artifacts directory
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir()

    return project_root


@pytest.fixture
def comparator(temp_project_root):
    """Create a StorageBackendComparator instance with temp project root."""
    return StorageBackendComparator(temp_project_root)


@pytest.fixture
def sample_criterion_json():
    """Sample Criterion estimates.json data."""
    return {
        "mean": {
            "point_estimate": 150000000.0,  # 150ms in nanoseconds
            "confidence_interval": {
                "lower_bound": 145000000.0,
                "upper_bound": 155000000.0,
            },
        },
    }


@pytest.fixture
def sample_criterion_stdout():
    """Sample Criterion stdout output for regex parsing."""
    return """
Running benchmarks...
construction/2D/1000v time:   [12.345 ms 12.456 ms 12.567 ms]
                      change: [+2.5% +3.1% +3.8%] (p = 0.00 < 0.05)
                      Performance has regressed.
iteration/vertices/1000v
                      time:   [5.123 µs 5.234 µs 5.345 µs]
queries/neighbors/1000v
                      time:   [8.901 ms 9.012 ms 9.123 ms]
"""


class TestStorageBackendComparator:
    """Test cases for StorageBackendComparator class."""

    def test_init(self, temp_project_root):
        """Test comparator initialization."""
        comparator = StorageBackendComparator(temp_project_root)

        assert comparator.project_root == temp_project_root
        assert comparator.criterion_dir == temp_project_root / "target" / "criterion"

    def test_parse_criterion_output_json_success(self, comparator, sample_criterion_json):
        """Test parsing Criterion output from JSON files."""
        # Create fake criterion directory structure
        bench_dir = comparator.criterion_dir / "construction" / "2D" / "1000v"
        bench_dir.mkdir(parents=True)

        estimates_file = bench_dir / "new" / "estimates.json"
        estimates_file.parent.mkdir(parents=True)

        with estimates_file.open("w") as f:
            json.dump(sample_criterion_json, f)

        # Parse output (stdout doesn't matter since JSON exists)
        results = comparator._parse_criterion_output("")

        assert "benchmarks" in results
        assert len(results["benchmarks"]) == 1

        bench = results["benchmarks"][0]
        assert bench["name"] == "1000v"
        assert bench["estimate"] == 150000000.0
        assert bench["unit"] == "ns"
        assert bench["lower"] == 145000000.0
        assert bench["upper"] == 155000000.0

    def test_parse_criterion_output_regex_fallback(self, comparator, sample_criterion_stdout):
        """Test parsing Criterion output using regex fallback when JSON unavailable."""
        results = comparator._parse_criterion_output(sample_criterion_stdout)

        assert "benchmarks" in results
        assert len(results["benchmarks"]) == 3

        # Check first benchmark (construction)
        bench1 = results["benchmarks"][0]
        assert bench1["name"] == "construction/2D/1000v"
        assert bench1["estimate"] == 12.456
        assert bench1["unit"] == "ms"
        assert bench1["lower"] == 12.345
        assert bench1["upper"] == 12.567

        # Check second benchmark (iteration)
        bench2 = results["benchmarks"][1]
        assert bench2["name"] == "iteration/vertices/1000v"
        assert bench2["estimate"] == 5.234
        assert bench2["unit"] == "µs"

        # Check third benchmark (queries)
        bench3 = results["benchmarks"][2]
        assert bench3["name"] == "queries/neighbors/1000v"
        assert bench3["estimate"] == 9.012
        assert bench3["unit"] == "ms"

    def test_parse_criterion_output_empty(self, comparator):
        """Test parsing empty Criterion output."""
        results = comparator._parse_criterion_output("")

        assert "benchmarks" in results
        assert len(results["benchmarks"]) == 0
        assert "raw_output" in results

    def test_build_comparison_table_basic(self, comparator):
        """Test building comparison table with matching benchmarks."""
        slotmap_by_name = {
            "test1": {"estimate": 100.0, "unit": "ms"},
            "test2": {"estimate": 50.0, "unit": "µs"},
        }

        denseslotmap_by_name = {
            "test1": {"estimate": 95.0, "unit": "ms"},  # 5% faster
            "test2": {"estimate": 55.0, "unit": "µs"},  # 10% slower
        }

        all_names = ["test1", "test2"]

        lines, diffs = comparator._build_comparison_table(slotmap_by_name, denseslotmap_by_name, all_names)

        assert len(lines) == 2
        assert len(diffs) == 2

        # test1: DenseSlotMap 5% faster
        assert "test1" in lines[0]
        assert "100.00 ms" in lines[0]
        assert "95.00 ms" in lines[0]
        assert "-5.0%" in lines[0]
        assert "✅ DenseSlotMap" in lines[0]

        # test2: DenseSlotMap 10% slower
        assert "test2" in lines[1]
        assert "50.00 µs" in lines[1]
        assert "55.00 µs" in lines[1]
        assert "+10.0%" in lines[1]
        assert "✅ SlotMap" in lines[1]

    def test_build_comparison_table_similar_performance(self, comparator):
        """Test comparison table with similar performance (< 2% difference)."""
        slotmap_by_name = {
            "test": {"estimate": 100.0, "unit": "ms"},
        }

        denseslotmap_by_name = {
            "test": {"estimate": 101.0, "unit": "ms"},  # 1% slower (within threshold)
        }

        all_names = ["test"]

        lines, _ = comparator._build_comparison_table(slotmap_by_name, denseslotmap_by_name, all_names)

        assert len(lines) == 1
        assert "~Same" in lines[0]
        assert "🟡" in lines[0]

    def test_build_comparison_table_missing_data(self, comparator):
        """Test comparison table with missing data for one backend."""
        slotmap_by_name = {
            "test1": {"estimate": 100.0, "unit": "ms"},
            "test2": {"estimate": 50.0, "unit": "µs"},
        }

        denseslotmap_by_name = {
            "test1": {"estimate": 95.0, "unit": "ms"},
        }

        all_names = ["test1", "test2"]

        lines, diffs = comparator._build_comparison_table(slotmap_by_name, denseslotmap_by_name, all_names)

        assert len(lines) == 2
        assert len(diffs) == 1  # Only test1 has data for both backends

        # test2 should show N/A for DenseSlotMap
        assert "test2" in lines[1]
        assert "N/A" in lines[1]

    @patch("compare_storage_backends.run_cargo_command")
    def test_run_benchmark_success(self, mock_run_cargo, comparator):
        """Test successful benchmark execution."""
        # Mock successful cargo bench run
        mock_result = CompletedProcess(
            args=["cargo", "bench"],
            returncode=0,
            stdout="test_bench time: [1.0 ms 1.1 ms 1.2 ms]",
            stderr="",
        )
        mock_run_cargo.return_value = mock_result

        result = comparator._run_benchmark("test_bench", use_dense_slotmap=False, dev_mode=False)

        assert result is not None
        assert "backend" in result
        assert result["backend"] == "SlotMap"
        assert "features" in result
        assert result["features"] == []
        assert "benchmarks" in result

    @patch("compare_storage_backends.run_cargo_command")
    def test_run_benchmark_with_dense_slotmap(self, mock_run_cargo, comparator):
        """Test benchmark execution with DenseSlotMap feature."""
        mock_result = CompletedProcess(
            args=["cargo", "bench"],
            returncode=0,
            stdout="test_bench time: [1.0 ms 1.1 ms 1.2 ms]",
            stderr="",
        )
        mock_run_cargo.return_value = mock_result

        result = comparator._run_benchmark("test_bench", use_dense_slotmap=True, dev_mode=False)

        assert result is not None
        assert result["backend"] == "DenseSlotMap"
        assert result["features"] == ["dense-slotmap"]

        # Verify cargo command included --features flag
        call_args = mock_run_cargo.call_args
        args = call_args[0][0]
        assert "--features" in args
        assert "dense-slotmap" in args

    @patch("compare_storage_backends.run_cargo_command")
    def test_run_benchmark_dev_mode(self, mock_run_cargo, comparator):
        """Test benchmark execution in development mode."""
        mock_result = CompletedProcess(
            args=["cargo", "bench"],
            returncode=0,
            stdout="test_bench time: [1.0 ms 1.1 ms 1.2 ms]",
            stderr="",
        )
        mock_run_cargo.return_value = mock_result

        result = comparator._run_benchmark("test_bench", use_dense_slotmap=False, dev_mode=True)

        assert result is not None

        # Verify dev mode args were added
        call_args = mock_run_cargo.call_args
        args = call_args[0][0]
        assert "--sample-size" in args
        assert "10" in args
        assert "--measurement-time" in args
        assert "2" in args
        assert "--noplot" in args

    @patch("compare_storage_backends.run_cargo_command")
    def test_run_benchmark_with_extra_args(self, mock_run_cargo, comparator):
        """Test benchmark execution with extra arguments."""
        mock_result = CompletedProcess(
            args=["cargo", "bench"],
            returncode=0,
            stdout="test_bench time: [1.0 ms 1.1 ms 1.2 ms]",
            stderr="",
        )
        mock_run_cargo.return_value = mock_result

        extra_args = ["construction"]
        result = comparator._run_benchmark(
            "test_bench",
            use_dense_slotmap=False,
            dev_mode=False,
            extra_args=extra_args,
        )

        assert result is not None

        # Verify extra args were added
        call_args = mock_run_cargo.call_args
        args = call_args[0][0]
        assert "construction" in args

    @patch("compare_storage_backends.run_cargo_command")
    def test_run_benchmark_failure(self, mock_run_cargo, comparator, capsys):
        """Test benchmark execution failure handling."""
        # Mock failed cargo bench run
        mock_result = CompletedProcess(
            args=["cargo", "bench"],
            returncode=1,
            stdout="",
            stderr="error: benchmark failed",
        )
        mock_run_cargo.return_value = mock_result

        result = comparator._run_benchmark("test_bench", use_dense_slotmap=False, dev_mode=False)

        assert result is None

        # Check error message was printed
        captured = capsys.readouterr()
        assert "Benchmark failed" in captured.err

    @patch("compare_storage_backends.run_cargo_command")
    def test_run_comparison_success(self, mock_run_cargo, comparator, tmp_path):
        """Test full comparison workflow success."""
        # Mock successful benchmark runs for both backends
        mock_result = CompletedProcess(
            args=["cargo", "bench"],
            returncode=0,
            stdout="test_bench time: [1.0 ms 1.1 ms 1.2 ms]",
            stderr="",
        )
        mock_run_cargo.return_value = mock_result

        output_path = tmp_path / "comparison.md"

        success = comparator.run_comparison(
            benchmark_name="test_bench",
            dev_mode=False,
            output_path=output_path,
        )

        assert success is True
        assert output_path.exists()

        # Verify report contains expected sections
        report = output_path.read_text()
        assert "Storage Backend Comparison" in report
        assert "SlotMap" in report
        assert "DenseSlotMap" in report

    @patch("compare_storage_backends.run_cargo_command")
    def test_run_comparison_slotmap_failure(self, mock_run_cargo, comparator):
        """Test comparison when SlotMap benchmark fails."""
        # Mock failed SlotMap run
        mock_result = CompletedProcess(
            args=["cargo", "bench"],
            returncode=1,
            stdout="",
            stderr="error",
        )
        mock_run_cargo.return_value = mock_result

        success = comparator.run_comparison(benchmark_name="test_bench")

        assert success is False

    @patch("compare_storage_backends.run_cargo_command")
    def test_run_comparison_denseslotmap_failure(self, mock_run_cargo, comparator):
        """Test comparison when DenseSlotMap benchmark fails."""
        # Mock successful SlotMap, failed DenseSlotMap
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call (SlotMap) succeeds
                return CompletedProcess(
                    args=["cargo", "bench"],
                    returncode=0,
                    stdout="test_bench time: [1.0 ms 1.1 ms 1.2 ms]",
                    stderr="",
                )
            # Second call (DenseSlotMap) fails
            return CompletedProcess(
                args=["cargo", "bench"],
                returncode=1,
                stdout="",
                stderr="error",
            )

        mock_run_cargo.side_effect = side_effect

        success = comparator.run_comparison(benchmark_name="test_bench")

        assert success is False

    def test_generate_comparison_report_structure(self, comparator):
        """Test comparison report generation structure."""
        slotmap_results = {
            "backend": "SlotMap",
            "features": [],
            "benchmarks": [
                {"name": "test1", "estimate": 100.0, "unit": "ms", "lower": 95.0, "upper": 105.0},
            ],
        }

        denseslotmap_results = {
            "backend": "DenseSlotMap",
            "features": ["dense-slotmap"],
            "benchmarks": [
                {"name": "test1", "estimate": 95.0, "unit": "ms", "lower": 90.0, "upper": 100.0},
            ],
        }

        report = comparator._generate_comparison_report(
            slotmap_results,
            denseslotmap_results,
            "test_bench",
            dev_mode=False,
        )

        # Verify report structure
        assert "# Storage Backend Comparison Report" in report
        assert "## Executive Summary" in report
        assert "## Detailed Results" in report
        assert "## Summary Statistics" in report
        assert "## Recommendations" in report
        assert "## Reproduction" in report
        assert "**Benchmark**: `test_bench`" in report
        assert "SlotMap" in report
        assert "DenseSlotMap" in report
        assert "test1" in report


class TestIntegration:
    """Integration tests for compare_storage_backends module."""

    def test_find_project_root_integration(self, tmp_path):
        """Test integration with find_project_root utility."""
        # Create a fake project with Cargo.toml
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "Cargo.toml").write_text('[package]\nname = "test"')

        # Change to project directory and test
        original_cwd = Path.cwd()
        try:
            os.chdir(project_root)
            result = find_project_root()
            assert result == project_root
        finally:
            os.chdir(original_cwd)
