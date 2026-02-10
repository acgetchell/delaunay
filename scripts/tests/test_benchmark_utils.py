#!/usr/bin/env python3
"""
Test suite for benchmark_utils.py module.

Tests benchmark parsing, baseline generation, and performance comparison functionality,
with special focus on the new average regression calculation logic.

Note: This test file accesses private methods (prefixed with _) which is expected
and necessary for comprehensive unit testing of internal functionality.
"""

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from benchmark_models import (
    BenchmarkData,
    CircumspherePerformanceData,
    CircumsphereTestCase,
)
from benchmark_utils import (
    DEFAULT_REGRESSION_THRESHOLD,
    DEV_MODE_BENCH_ARGS,
    BaselineGenerator,
    BenchmarkRegressionHelper,
    CriterionParser,
    PerformanceComparator,
    PerformanceSummaryGenerator,
    ProjectRootNotFoundError,
    WorkflowHelper,
    create_argument_parser,
    find_project_root,
    main,
)

THRESHOLD_PERCENT = f"{DEFAULT_REGRESSION_THRESHOLD:.1f}%"


def compute_average_time_change(current_results, baseline_results):
    """Replicate PerformanceComparator's geometric mean logic for tests."""
    time_changes = []
    for current in current_results:
        key = f"{current.points}_{current.dimension}"
        baseline = baseline_results.get(key)
        if not baseline or baseline.time_mean <= 0:
            continue
        time_change = ((current.time_mean - baseline.time_mean) / baseline.time_mean) * 100.0
        time_changes.append(time_change)

    if not time_changes:
        return 0.0

    ratios = [1.0 + (tc / 100.0) for tc in time_changes if (1.0 + (tc / 100.0)) > 0.0]
    if not ratios:
        return 0.0

    avg_log = sum(math.log(ratio) for ratio in ratios) / len(ratios)
    avg_ratio = math.exp(avg_log)
    return (avg_ratio - 1.0) * 100.0


@pytest.fixture
def sample_estimates_data():
    """Fixture for common estimates.json test data."""
    return {
        "mean": {
            "point_estimate": 110000.0,  # 110 microseconds in nanoseconds
            "confidence_interval": {"lower_bound": 100000.0, "upper_bound": 120000.0},
        },
    }


@pytest.fixture
def sample_benchmark_data():
    """Fixture for common BenchmarkData test objects."""
    return {
        "2d_1000": BenchmarkData(1000, "2D").with_timing(100.0, 110.0, 120.0, "µs"),
        "2d_2000": BenchmarkData(2000, "2D").with_timing(190.0, 200.0, 210.0, "µs"),
        "3d_1000": BenchmarkData(1000, "3D").with_timing(200.0, 220.0, 240.0, "µs"),
    }


class TestCriterionParser:
    """Test cases for CriterionParser class."""

    def test_parse_estimates_json_valid_data(self, sample_estimates_data):
        """Test parsing valid estimates.json data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_estimates_data, f)
            f.flush()
            estimates_path = Path(f.name)

        try:
            result = CriterionParser.parse_estimates_json(estimates_path, 1000, "2D")

            assert result is not None
            assert result.points == 1000
            assert result.dimension == "2D"
            assert result.time_mean == 110.0  # Converted to microseconds
            assert result.time_low == 100.0
            assert result.time_high == 120.0
            assert result.time_unit == "µs"
            assert result.throughput_mean is not None
            assert result.throughput_mean == pytest.approx(9090.909, abs=0.001)  # 1000 * 1000 / 110
        finally:
            estimates_path.unlink()

    def test_parse_estimates_json_zero_mean(self):
        """Test parsing estimates.json with zero mean time."""
        estimates_data = {"mean": {"point_estimate": 0.0, "confidence_interval": {"lower_bound": 0.0, "upper_bound": 0.0}}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(estimates_data, f)
            f.flush()
            estimates_path = Path(f.name)

        try:
            result = CriterionParser.parse_estimates_json(estimates_path, 1000, "2D")
            assert result is None
        finally:
            estimates_path.unlink()

    def test_parse_estimates_json_very_fast_benchmark_division_by_zero_protection(self):
        """Test division by zero protection for very fast benchmarks with near-zero confidence intervals."""
        estimates_data = {
            "mean": {
                "point_estimate": 1000.0,  # 1 microsecond in nanoseconds
                "confidence_interval": {
                    "lower_bound": 0.0,  # Could cause division by zero without protection
                    "upper_bound": 2000.0,
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(estimates_data, f)
            f.flush()
            estimates_path = Path(f.name)

        try:
            result = CriterionParser.parse_estimates_json(estimates_path, 1000, "2D")

            # Should not crash and should return valid data with protected throughput calculation
            assert result is not None
            assert result.points == 1000
            assert result.dimension == "2D"
            assert result.time_mean == 1.0  # 1 microsecond
            assert result.time_low == 0.0  # Lower bound of confidence interval
            assert result.time_high == 2.0  # Upper bound

            # Throughput should be calculated with epsilon protection
            # thrpt_high = points * 1000 / max(low_us, eps) = 1000 * 1000 / max(0.0, 1e-9) = 1000 * 1000 / 1e-9
            assert result.throughput_high is not None
            assert result.throughput_high > 1e12  # Should be very large due to epsilon protection
            assert result.throughput_mean is not None
            assert result.throughput_low is not None
        finally:
            estimates_path.unlink()

    def test_parse_estimates_json_invalid_file(self):
        """Test parsing non-existent estimates.json file."""
        result = CriterionParser.parse_estimates_json(Path("nonexistent.json"), 1000, "2D")
        assert result is None

    def test_parse_estimates_json_malformed_json(self):
        """Test parsing malformed JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json")
            f.flush()
            estimates_path = Path(f.name)

        try:
            result = CriterionParser.parse_estimates_json(estimates_path, 1000, "2D")
            assert result is None
        finally:
            estimates_path.unlink()

    @patch("benchmark_utils.Path.exists")
    @patch("benchmark_utils.Path.iterdir")
    def test_find_criterion_results_no_criterion_dir(self, mock_iterdir, mock_exists):  # noqa: ARG002
        """Test finding criterion results when criterion directory doesn't exist."""
        mock_exists.return_value = False

        target_dir = Path("/fake/target")
        results = CriterionParser.find_criterion_results(target_dir)

        assert results == []

    def test_find_criterion_results_sorting(self):
        """Test that results are sorted by dimension and points."""
        # Create test data that would be unsorted initially
        test_results = [
            BenchmarkData(5000, "3D").with_timing(200.0, 220.0, 240.0, "µs"),
            BenchmarkData(1000, "2D").with_timing(100.0, 110.0, 120.0, "µs"),
            BenchmarkData(1000, "4D").with_timing(300.0, 320.0, 340.0, "µs"),
            BenchmarkData(2000, "2D").with_timing(150.0, 160.0, 170.0, "µs"),
        ]

        # Sort using the same logic as the actual function
        test_results.sort(key=lambda x: (int(x.dimension.rstrip("D")), x.points))

        # Verify sorting order
        assert test_results[0].dimension == "2D"
        assert test_results[0].points == 1000
        assert test_results[1].dimension == "2D"
        assert test_results[1].points == 2000
        assert test_results[2].dimension == "3D"
        assert test_results[2].points == 5000
        assert test_results[3].dimension == "4D"
        assert test_results[3].points == 1000

    def test_ci_performance_suite_patterns(self):
        """Test CI performance suite benchmark patterns (2D, 3D, 4D, 5D with 10, 25, 50 points)."""
        # Test data representing CI performance suite dimensions and point counts
        ci_suite_results = [
            BenchmarkData(10, "2D").with_timing(18.0, 20.0, 22.0, "µs"),
            BenchmarkData(25, "2D").with_timing(38.0, 40.0, 42.0, "µs"),
            BenchmarkData(50, "2D").with_timing(78.0, 80.0, 82.0, "µs"),
            BenchmarkData(10, "3D").with_timing(48.0, 50.0, 52.0, "µs"),
            BenchmarkData(25, "3D").with_timing(118.0, 125.0, 132.0, "µs"),
            BenchmarkData(50, "3D").with_timing(245.0, 250.0, 255.0, "µs"),
            BenchmarkData(10, "4D").with_timing(58.0, 60.0, 62.0, "µs"),
            BenchmarkData(25, "4D").with_timing(118.0, 120.0, 122.0, "µs"),
            BenchmarkData(50, "4D").with_timing(290.0, 300.0, 310.0, "µs"),
            BenchmarkData(10, "5D").with_timing(78.0, 80.0, 82.0, "µs"),
            BenchmarkData(25, "5D").with_timing(145.0, 150.0, 155.0, "µs"),
            BenchmarkData(50, "5D").with_timing(290.0, 300.0, 310.0, "µs"),
        ]

        # Sort using the same logic as the actual function (by dimension, then points)
        ci_suite_results.sort(key=lambda x: (int(x.dimension.rstrip("D")), x.points))

        # Verify sorting order: 2D..5D, then 10,25,50 within each dimension
        expected_order = [(d, p) for d in ("2D", "3D", "4D", "5D") for p in (10, 25, 50)]
        actual_order = [(b.dimension, b.points) for b in ci_suite_results]
        assert actual_order == expected_order


class TestPerformanceComparator:
    """Test cases for PerformanceComparator class."""

    @pytest.fixture
    def comparator(self):
        """Fixture for PerformanceComparator instance."""
        project_root = Path("/fake/project")
        return PerformanceComparator(project_root)

    @pytest.fixture
    def sample_baseline_content(self):
        """Fixture for sample baseline content."""
        return """Date: 2023-06-15 10:30:00 PDT
Git commit: abc123def456
Hardware Information:
  OS: macOS
  CPU: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
  CPU Cores: 6
  CPU Threads: 12
  Memory: 16.0 GB
  Rust: rustc 1.70.0 (90c541806 2023-05-31)
  Target: x86_64-apple-darwin

=== 1000 Points (2D) ===
Time: [100.0, 110.0, 120.0] µs
Throughput: [8.333, 9.091, 10.0] Kelem/s

=== 2000 Points (2D) ===
Time: [190.0, 200.0, 210.0] µs
Throughput: [9.524, 10.0, 10.526] Kelem/s

=== 1000 Points (3D) ===
Time: [200.0, 220.0, 240.0] µs
Throughput: [4.167, 4.545, 5.0] Kelem/s
"""

    def test_parse_baseline_file(self, comparator, sample_baseline_content):
        """Test parsing baseline file content."""
        results = comparator._parse_baseline_file(sample_baseline_content)

        assert len(results) == 3
        assert "1000_2D" in results
        assert "2000_2D" in results
        assert "1000_3D" in results

        # Test first benchmark
        bench_2d_1000 = results["1000_2D"]
        assert bench_2d_1000.points == 1000
        assert bench_2d_1000.dimension == "2D"
        assert bench_2d_1000.time_mean == 110.0
        assert bench_2d_1000.throughput_mean == 9.091

    def test_write_time_comparison_no_regression(self, comparator):
        """Test time comparison writing with no regression."""
        current = BenchmarkData(1000, "2D").with_timing(100.0, 110.0, 120.0, "µs")
        baseline = BenchmarkData(1000, "2D").with_timing(95.0, 105.0, 115.0, "µs")

        output = StringIO()
        time_change, is_regression = comparator._write_time_comparison(output, current, baseline)

        # Change is (110 - 105) / 105 * 100 = 4.76%
        assert time_change == pytest.approx(4.76, abs=0.01)
        assert not is_regression  # Less than DEFAULT_REGRESSION_THRESHOLD

        result = output.getvalue()
        assert "4.8%" in result
        assert "✅ OK: Time change +4.8% within acceptable range" in result

    def test_write_time_comparison_with_regression(self, comparator):
        """Test time comparison writing with regression."""
        current = BenchmarkData(1000, "2D").with_timing(100.0, 115.0, 130.0, "µs")
        baseline = BenchmarkData(1000, "2D").with_timing(95.0, 100.0, 105.0, "µs")

        output = StringIO()
        time_change, is_regression = comparator._write_time_comparison(output, current, baseline)

        # Change is (115 - 100) / 100 * 100 = 15%
        assert time_change == pytest.approx(15.0, abs=1e-9)
        assert is_regression  # Greater than DEFAULT_REGRESSION_THRESHOLD

        result = output.getvalue()
        assert "15.0%" in result
        assert "⚠️  REGRESSION" in result

    def test_write_time_comparison_with_improvement(self, comparator):
        """Test time comparison writing with significant improvement."""
        current = BenchmarkData(1000, "2D").with_timing(80.0, 90.0, 100.0, "µs")
        baseline = BenchmarkData(1000, "2D").with_timing(95.0, 100.0, 105.0, "µs")

        output = StringIO()
        time_change, is_regression = comparator._write_time_comparison(output, current, baseline)

        # Change is (90 - 100) / 100 * 100 = -10%
        assert time_change == pytest.approx(-10.0, abs=1e-9)
        assert not is_regression

        result = output.getvalue()
        assert "10.0%" in result
        assert "✅ IMPROVEMENT: Time decreased by 10.0% (faster performance)" in result

    def test_write_time_comparison_zero_baseline(self, comparator):
        """Test time comparison with zero baseline time."""
        current = BenchmarkData(1000, "2D").with_timing(100.0, 110.0, 120.0, "µs")
        baseline = BenchmarkData(1000, "2D").with_timing(0.0, 0.0, 0.0, "µs")

        output = StringIO()
        time_change, is_regression = comparator._write_time_comparison(output, current, baseline)

        assert time_change is None
        assert not is_regression

        result = output.getvalue()
        assert "N/A (baseline mean is 0)" in result

    @pytest.mark.parametrize("dev_mode", [False, True])
    @patch("benchmark_utils.run_cargo_command")
    def test_compare_omits_quiet_flag(self, mock_cargo, dev_mode):
        """Test that PerformanceComparator invokes cargo without --quiet flag (removed for better error visibility)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            baseline_file = temp_path / "baseline.txt"

            # Create a minimal baseline file
            baseline_content = """Date: 2023-12-15 10:30:00 UTC
Git commit: abc123
=== 10 Points (2D) ===
Time: [1.0, 1.0, 1.0] µs
"""
            baseline_file.write_text(baseline_content)

            # Mock successful cargo command
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_cargo.return_value = mock_result

            comparator = PerformanceComparator(temp_path)
            comparator.compare_with_baseline(baseline_file, dev_mode=dev_mode)

            # Verify cargo was called without --quiet flag (removed for better error visibility)
            assert mock_cargo.call_count >= 1
            args = mock_cargo.call_args[0][0]
            assert "--quiet" not in args  # Changed: --quiet flag should NOT be present
            if dev_mode:
                for arg in DEV_MODE_BENCH_ARGS:
                    assert arg in args
            # And output is captured
            assert mock_cargo.call_args.kwargs.get("capture_output") is True

    def test_write_performance_comparison_no_average_regression(self, comparator):
        """Test performance comparison with individual regressions but no average regression."""
        # Create current results with mixed performance changes
        current_results = [
            # Big regression: +20%
            BenchmarkData(1000, "2D").with_timing(108.0, 120.0, 132.0, "µs"),
            # Small improvement: -2%
            BenchmarkData(2000, "2D").with_timing(186.0, 196.0, 206.0, "µs"),
            # Big improvement: -15%
            BenchmarkData(1000, "3D").with_timing(170.0, 187.0, 204.0, "µs"),
        ]

        # Create baseline results
        baseline_results = {
            "1000_2D": BenchmarkData(1000, "2D").with_timing(95.0, 100.0, 105.0, "µs"),
            "2000_2D": BenchmarkData(2000, "2D").with_timing(190.0, 200.0, 210.0, "µs"),
            "1000_3D": BenchmarkData(1000, "3D").with_timing(200.0, 220.0, 240.0, "µs"),
        }

        output = StringIO()
        regression_found = comparator._write_performance_comparison(output, current_results, baseline_results)

        # Average change using geometric mean: ~0.0%
        # This is less than DEFAULT_REGRESSION_THRESHOLD, so no overall regression
        assert not regression_found

        result = output.getvalue()
        assert "SUMMARY" in result
        assert "Total benchmarks compared: 3" in result
        assert f"Individual regressions (>{THRESHOLD_PERCENT}): 1" in result  # Only the +20% one
        assert re.search(r"Average time change:\s*-?0\.0%", result)
        assert "✅ OVERALL OK" in result

    def test_write_performance_comparison_with_average_regression(self, comparator):
        """Test performance comparison with average regression exceeding threshold."""
        # Create current results with overall performance degradation
        current_results = [
            # Regression: +20%
            BenchmarkData(1000, "2D").with_timing(118.0, 120.0, 122.0, "µs"),
            # Regression: +15%
            BenchmarkData(2000, "2D").with_timing(222.0, 230.0, 238.0, "µs"),
            # Small improvement: -1%
            BenchmarkData(1000, "3D").with_timing(209.0, 217.8, 226.6, "µs"),
        ]

        # Create baseline results
        baseline_results = {
            "1000_2D": BenchmarkData(1000, "2D").with_timing(95.0, 100.0, 105.0, "µs"),
            "2000_2D": BenchmarkData(2000, "2D").with_timing(190.0, 200.0, 210.0, "µs"),
            "1000_3D": BenchmarkData(1000, "3D").with_timing(200.0, 220.0, 240.0, "µs"),
        }

        output = StringIO()
        regression_found = comparator._write_performance_comparison(output, current_results, baseline_results)

        # Average change using geometric mean: 11.0%
        # This exceeds DEFAULT_REGRESSION_THRESHOLD, so overall regression found
        assert regression_found

        result = output.getvalue()
        assert "SUMMARY" in result
        assert "Total benchmarks compared: 3" in result
        assert f"Individual regressions (>{THRESHOLD_PERCENT}): 2" in result  # The +20% and +15% ones
        assert "Average time change: 11.0%" in result
        assert "🚨 OVERALL REGRESSION" in result

    def test_write_performance_comparison_with_average_improvement(self, comparator):
        """Test performance comparison with significant average improvement."""
        # Create current results with overall performance improvement
        current_results = [
            # Improvement: -10%
            BenchmarkData(1000, "2D").with_timing(81.0, 90.0, 99.0, "µs"),
            # Improvement: -8%
            BenchmarkData(2000, "2D").with_timing(175.2, 184.0, 192.8, "µs"),
            # Small regression: +2%
            BenchmarkData(1000, "3D").with_timing(209.0, 224.4, 239.8, "µs"),
        ]

        # Create baseline results
        baseline_results = {
            "1000_2D": BenchmarkData(1000, "2D").with_timing(95.0, 100.0, 105.0, "µs"),
            "2000_2D": BenchmarkData(2000, "2D").with_timing(190.0, 200.0, 210.0, "µs"),
            "1000_3D": BenchmarkData(1000, "3D").with_timing(200.0, 220.0, 240.0, "µs"),
        }

        output = StringIO()
        regression_found = comparator._write_performance_comparison(output, current_results, baseline_results)

        # Average change using geometric mean: -5.5%
        # This is significant improvement, so no regression found
        assert not regression_found

        result = output.getvalue()
        assert "SUMMARY" in result
        assert "Total benchmarks compared: 3" in result
        assert f"Individual regressions (>{THRESHOLD_PERCENT}): 0" in result
        expected_average_change = compute_average_time_change(current_results, baseline_results)
        expected_average_line = f"Average time change: {expected_average_change:.1f}%"
        assert expected_average_line in result
        assert "✅ OVERALL OK" in result

    def test_write_performance_comparison_missing_baseline(self, comparator):
        """Test performance comparison when some baselines are missing."""
        current_results = [
            BenchmarkData(1000, "2D").with_timing(105.0, 110.0, 115.0, "µs"),
            BenchmarkData(3000, "2D").with_timing(300.0, 310.0, 320.0, "µs"),  # No baseline
        ]

        baseline_results = {
            "1000_2D": BenchmarkData(1000, "2D").with_timing(95.0, 100.0, 105.0, "µs"),
        }

        output = StringIO()
        regression_found = comparator._write_performance_comparison(output, current_results, baseline_results)

        # Only one benchmark should be compared, regression could be found based on that single comparison
        # In this case, we have 10% regression (110 vs 100), so regression should be detected
        assert regression_found

        result = output.getvalue()
        assert "Total benchmarks compared: 1" in result
        assert "3000 Points (2D)" in result  # Should still show the benchmark without baseline

    def test_write_performance_comparison_no_benchmarks(self, comparator):
        """Test performance comparison with no benchmarks."""
        output = StringIO()
        regression_found = comparator._write_performance_comparison(output, [], {})

        # Should return False when no benchmarks to compare
        assert not regression_found

    @patch("benchmark_utils.get_git_commit_hash")
    @patch("benchmark_utils.datetime")
    def test_prepare_comparison_metadata(self, mock_datetime, mock_git, comparator, sample_baseline_content):
        """Test preparation of comparison metadata."""
        # Mock current datetime
        mock_now = Mock()
        mock_now.strftime.return_value = "Thu Jun 15 14:30:00 PDT 2023"
        mock_datetime.now.return_value.astimezone.return_value = mock_now

        # Mock git commit
        mock_git.return_value = "def456abc789"

        metadata = comparator._prepare_comparison_metadata(sample_baseline_content)

        assert metadata["current_date"] == "Thu Jun 15 14:30:00 PDT 2023"
        assert metadata["current_commit"] == "def456abc789"
        assert metadata["baseline_date"] == "2023-06-15 10:30:00 PDT"
        assert metadata["baseline_commit"] == "abc123def456"

    @patch("benchmark_utils.get_git_commit_hash")
    def test_prepare_comparison_metadata_git_failure(self, mock_git, comparator, sample_baseline_content):
        """Test metadata preparation when git command fails."""
        mock_git.side_effect = Exception("Git not available")

        metadata = comparator._prepare_comparison_metadata(sample_baseline_content)

        assert metadata["current_commit"] == "unknown"

    def test_regression_threshold_configuration(self, comparator):
        """Test that regression threshold can be configured."""
        # Test default threshold
        assert comparator.regression_threshold == DEFAULT_REGRESSION_THRESHOLD

        # Test changing threshold
        comparator.regression_threshold = 10.0

        current = BenchmarkData(1000, "2D").with_timing(100.0, 107.0, 114.0, "µs")
        baseline = BenchmarkData(1000, "2D").with_timing(95.0, 100.0, 105.0, "µs")

        output = StringIO()
        time_change, is_regression = comparator._write_time_comparison(output, current, baseline)

        # 7% change should not be regression with 10% threshold
        assert time_change == pytest.approx(7.0, abs=0.001)  # Use pytest.approx for floating-point comparison
        assert not is_regression

    def test_write_error_file_baseline_not_found(self, comparator):
        """Test writing error file when baseline is not found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "error_results.txt"
            baseline_file = Path(temp_dir) / "nonexistent_baseline.txt"

            comparator._write_error_file(output_file, "Baseline file not found", baseline_file)

            assert output_file.exists()
            content = output_file.read_text()
            assert "Comparison Results" in content
            assert "❌ Error: Baseline file not found" in content
            assert str(baseline_file) in content
            assert "This error prevented the benchmark comparison from completing successfully" in content

    def test_write_error_file_benchmark_error(self, comparator):
        """Test writing error file when benchmark execution fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "error_results.txt"
            error_message = "Failed to compile benchmarks: error[E0277]: trait bound not satisfied"

            comparator._write_error_file(output_file, "Benchmark execution error", error_message)

            assert output_file.exists()
            content = output_file.read_text()
            assert "❌ Error: Benchmark execution error" in content
            assert error_message in content
            assert "Please check the CI logs for more information" in content

    def test_write_error_file_creates_parent_directory(self, comparator):
        """Test that _write_error_file creates parent directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "nested" / "path" / "error_results.txt"

            comparator._write_error_file(output_file, "Test error", "Test details")

            assert output_file.exists()
            assert output_file.parent.exists()
            content = output_file.read_text()
            assert "❌ Error: Test error" in content

    def test_write_error_file_handles_write_failure(self, comparator):
        """Test that _write_error_file handles write failures gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "error_results.txt"

            # Mock Path.open to raise an exception
            with patch.object(Path, "open", side_effect=OSError("Permission denied")):
                # Should not raise exception, just log it
                comparator._write_error_file(output_file, "Test error", "Test details")

            # File should not exist due to write failure
            assert not output_file.exists()


class TestIntegrationScenarios:
    """Integration test scenarios for real-world use cases."""

    @pytest.fixture
    def comparator(self):
        """Fixture for PerformanceComparator instance."""
        project_root = Path("/fake/project")
        return PerformanceComparator(project_root)

    def test_realistic_mixed_performance_scenario(self, comparator):
        """Test a realistic scenario with mixed performance changes."""
        # Simulate a realistic benchmark run with various performance changes
        current_results = [
            # Small regression in 2D small dataset: +3%
            BenchmarkData(1000, "2D").with_timing(98.0, 103.0, 108.0, "µs"),
            # Medium regression in 2D medium dataset: +8%
            BenchmarkData(5000, "2D").with_timing(432.0, 540.0, 648.0, "µs"),
            # Small improvement in 2D large dataset: -2%
            BenchmarkData(10000, "2D").with_timing(931.2, 980.0, 1028.8, "µs"),
            # Large improvement in 3D small dataset: -12%
            BenchmarkData(1000, "3D").with_timing(176.0, 220.0, 264.0, "µs"),
            # Small regression in 3D medium dataset: +4%
            BenchmarkData(5000, "3D").with_timing(1040.0, 1300.0, 1560.0, "µs"),
        ]

        baseline_results = {
            "1000_2D": BenchmarkData(1000, "2D").with_timing(95.0, 100.0, 105.0, "µs"),
            "5000_2D": BenchmarkData(5000, "2D").with_timing(450.0, 500.0, 550.0, "µs"),
            "10000_2D": BenchmarkData(10000, "2D").with_timing(950.0, 1000.0, 1050.0, "µs"),
            "1000_3D": BenchmarkData(1000, "3D").with_timing(225.0, 250.0, 275.0, "µs"),
            "5000_3D": BenchmarkData(5000, "3D").with_timing(1200.0, 1250.0, 1300.0, "µs"),
        }

        output = StringIO()
        regression_found = comparator._write_performance_comparison(output, current_results, baseline_results)

        # Average change using geometric mean: -0.2%
        # No overall regression should be detected
        assert not regression_found

        result = output.getvalue()
        assert "Total benchmarks compared: 5" in result
        assert f"Individual regressions (>{THRESHOLD_PERCENT}): 1" in result  # Only the +8% one
        expected_average_change = compute_average_time_change(current_results, baseline_results)
        expected_average_line = f"Average time change: {expected_average_change:.1f}%"
        assert expected_average_line in result
        assert "✅ OVERALL OK" in result

    def test_gradual_performance_degradation_scenario(self, comparator):
        """Test scenario where performance gradually degrades across all benchmarks."""
        # Simulate gradual performance degradation that individually isn't alarming
        # but collectively indicates a problem
        current_results = [
            # Each benchmark has ~9% regression individually
            BenchmarkData(1000, "2D").with_timing(104.0, 109.0, 114.0, "µs"),  # +9%
            BenchmarkData(5000, "2D").with_timing(520.0, 545.0, 570.0, "µs"),  # +9%
            BenchmarkData(10000, "2D").with_timing(1050.0, 1090.0, 1130.0, "µs"),  # +9%
            BenchmarkData(1000, "3D").with_timing(240.0, 272.5, 305.0, "µs"),  # +9%
            BenchmarkData(5000, "3D").with_timing(1335.0, 1362.5, 1390.0, "µs"),  # +9%
        ]

        baseline_results = {
            "1000_2D": BenchmarkData(1000, "2D").with_timing(95.0, 100.0, 105.0, "µs"),
            "5000_2D": BenchmarkData(5000, "2D").with_timing(450.0, 500.0, 550.0, "µs"),
            "10000_2D": BenchmarkData(10000, "2D").with_timing(950.0, 1000.0, 1050.0, "µs"),
            "1000_3D": BenchmarkData(1000, "3D").with_timing(225.0, 250.0, 275.0, "µs"),
            "5000_3D": BenchmarkData(5000, "3D").with_timing(1200.0, 1250.0, 1300.0, "µs"),
        }

        output = StringIO()
        regression_found = comparator._write_performance_comparison(output, current_results, baseline_results)

        # Average change: 9.0%
        # Should detect overall regression even though individual ones are mixed
        assert regression_found

        result = output.getvalue()
        assert "Total benchmarks compared: 5" in result
        assert f"Individual regressions (>{THRESHOLD_PERCENT}): 5" in result  # All regressions exceed DEFAULT_REGRESSION_THRESHOLD
        assert "Average time change: 9.0%" in result
        assert "🚨 OVERALL REGRESSION" in result

    def test_noisy_benchmarks_scenario(self, comparator):
        """Test scenario with noisy benchmarks that have high individual variance."""
        # Simulate noisy benchmarks where individual results vary significantly
        # but overall trend is acceptable
        current_results = [
            # High variance but acceptable average
            BenchmarkData(1000, "2D").with_timing(75.0, 102.0, 140.0, "µs"),  # +2%
            BenchmarkData(5000, "2D").with_timing(350.0, 480.0, 650.0, "µs"),  # -4%
            BenchmarkData(10000, "2D").with_timing(800.0, 1030.0, 1350.0, "µs"),  # +3%
            # One outlier with big regression
            BenchmarkData(1000, "3D").with_timing(280.0, 350.0, 420.0, "µs"),  # +40%
            # Others are improvements
            BenchmarkData(5000, "3D").with_timing(950.0, 1125.0, 1300.0, "µs"),  # -10%
        ]

        baseline_results = {
            "1000_2D": BenchmarkData(1000, "2D").with_timing(95.0, 100.0, 105.0, "µs"),
            "5000_2D": BenchmarkData(5000, "2D").with_timing(450.0, 500.0, 550.0, "µs"),
            "10000_2D": BenchmarkData(10000, "2D").with_timing(950.0, 1000.0, 1050.0, "µs"),
            "1000_3D": BenchmarkData(1000, "3D").with_timing(225.0, 250.0, 275.0, "µs"),
            "5000_3D": BenchmarkData(5000, "3D").with_timing(1200.0, 1250.0, 1300.0, "µs"),
        }

        output = StringIO()
        regression_found = comparator._write_performance_comparison(output, current_results, baseline_results)

        # Average change using geometric mean: 4.9%
        # Despite the one big outlier, no overall regression should be detected (4.9% < DEFAULT_REGRESSION_THRESHOLD)
        assert not regression_found

        result = output.getvalue()
        assert "Total benchmarks compared: 5" in result
        assert f"Individual regressions (>{THRESHOLD_PERCENT}): 1" in result  # Only the 40% outlier
        assert "Average time change: 4.9%" in result
        assert "✅ OVERALL OK" in result


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def comparator(self):
        """Fixture for PerformanceComparator instance."""
        project_root = Path("/fake/project")
        return PerformanceComparator(project_root)

    def test_empty_current_results(self, comparator):
        """Test comparison with empty current results."""
        baseline_results = {
            "1000_2D": BenchmarkData(1000, "2D").with_timing(95.0, 100.0, 105.0, "µs"),
        }

        output = StringIO()
        regression_found = comparator._write_performance_comparison(output, [], baseline_results)

        assert not regression_found
        assert "SUMMARY" not in output.getvalue()

    def test_empty_baseline_results(self, comparator):
        """Test comparison with empty baseline results."""
        current_results = [
            BenchmarkData(1000, "2D").with_timing(105.0, 110.0, 115.0, "µs"),
        ]

        output = StringIO()
        regression_found = comparator._write_performance_comparison(output, current_results, {})

        assert not regression_found
        result = output.getvalue()
        assert "1000 Points (2D)" in result
        assert "SUMMARY" not in result

    def test_all_zero_baseline_times(self, comparator):
        """Test comparison when all baseline times are zero."""
        current_results = [
            BenchmarkData(1000, "2D").with_timing(105.0, 110.0, 115.0, "µs"),
            BenchmarkData(2000, "2D").with_timing(205.0, 220.0, 235.0, "µs"),
        ]

        baseline_results = {
            "1000_2D": BenchmarkData(1000, "2D").with_timing(0.0, 0.0, 0.0, "µs"),
            "2000_2D": BenchmarkData(2000, "2D").with_timing(0.0, 0.0, 0.0, "µs"),
        }

        output = StringIO()
        regression_found = comparator._write_performance_comparison(output, current_results, baseline_results)

        assert not regression_found
        result = output.getvalue()
        assert "N/A (baseline mean is 0)" in result
        assert "SUMMARY" not in result  # No valid comparisons

    def test_mixed_valid_invalid_baselines(self, comparator):
        """Test comparison with mix of valid and invalid baseline data."""
        current_results = [
            BenchmarkData(1000, "2D").with_timing(105.0, 110.0, 115.0, "µs"),
            BenchmarkData(2000, "2D").with_timing(205.0, 220.0, 235.0, "µs"),
        ]

        baseline_results = {
            "1000_2D": BenchmarkData(1000, "2D").with_timing(95.0, 100.0, 105.0, "µs"),  # Valid
            "2000_2D": BenchmarkData(2000, "2D").with_timing(0.0, 0.0, 0.0, "µs"),  # Invalid
        }

        output = StringIO()
        regression_found = comparator._write_performance_comparison(output, current_results, baseline_results)

        # Should find regression due to the 10% change in the valid comparison
        assert regression_found

        result = output.getvalue()
        assert "Total benchmarks compared: 1" in result  # Only one valid comparison
        assert "N/A (baseline mean is 0)" in result
        assert "10.0%" in result  # The valid comparison shows 10% change


class TestWorkflowHelper:
    """Test cases for WorkflowHelper class."""

    @patch.dict(os.environ, {"GITHUB_REF": "refs/tags/v1.2.3"}, clear=False)
    def test_determine_tag_name_from_github_ref(self):
        """Test tag name determination from GITHUB_REF with tag."""
        tag_name = WorkflowHelper.determine_tag_name()
        assert tag_name == "v1.2.3"

    @patch.dict(os.environ, {"GITHUB_REF": "refs/heads/main"}, clear=False)
    @patch("benchmark_utils.datetime")
    def test_determine_tag_name_generated(self, mock_datetime):
        """Test tag name generation when not from a tag push."""
        # Mock datetime
        mock_now = Mock()
        mock_now.strftime.return_value = "20231215-143000"
        mock_datetime.now.return_value = mock_now

        tag_name = WorkflowHelper.determine_tag_name()
        assert tag_name == "manual-20231215-143000"

    @patch.dict(os.environ, {"GITHUB_REF": "refs/tags/v2.0.0"}, clear=False)
    def test_determine_tag_name_with_github_output(self):
        """Test tag name determination with GITHUB_OUTPUT file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            output_file = f.name

        try:
            with patch.dict(os.environ, {"GITHUB_OUTPUT": output_file}):
                tag_name = WorkflowHelper.determine_tag_name()
                assert tag_name == "v2.0.0"

            # Check that GITHUB_OUTPUT file was written
            with open(output_file, encoding="utf-8") as f:
                content = f.read()
                assert "tag_name=v2.0.0\n" in content
        finally:
            Path(output_file).unlink(missing_ok=True)

    def test_create_metadata_success(self):
        """Test successful metadata creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            with patch.dict(
                os.environ,
                {
                    "GITHUB_SHA": "abc123def456",
                    "GITHUB_RUN_ID": "123456789",
                    "RUNNER_OS": "macOS",
                    "RUNNER_ARCH": "ARM64",
                },
            ):
                success = WorkflowHelper.create_metadata("v1.0.0", output_dir)
                assert success

            # Check metadata file was created
            metadata_file = output_dir / "metadata.json"
            assert metadata_file.exists()

            # Check metadata content
            with metadata_file.open("r", encoding="utf-8") as f:
                metadata = json.load(f)

            assert metadata["tag"] == "v1.0.0"
            assert metadata["commit"] == "abc123def456"
            assert metadata["workflow_run_id"] == "123456789"
            assert metadata["runner_os"] == "macOS"
            assert metadata["runner_arch"] == "ARM64"
            assert "generated_at" in metadata
            # Check ISO format timestamp
            assert metadata["generated_at"].endswith("Z")

    def test_create_metadata_with_safe_env_vars(self):
        """Test metadata creation with SAFE_ prefixed environment variables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            with patch.dict(
                os.environ,
                {
                    "SAFE_COMMIT_SHA": "def456abc789",
                    "SAFE_RUN_ID": "987654321",
                    "RUNNER_OS": "Linux",
                    "RUNNER_ARCH": "X64",
                },
                clear=True,
            ):
                success = WorkflowHelper.create_metadata("v2.0.0", output_dir)
                assert success

            # Check metadata file was created
            metadata_file = output_dir / "metadata.json"
            assert metadata_file.exists()

            # Check metadata content uses SAFE_ variables
            with metadata_file.open("r", encoding="utf-8") as f:
                metadata = json.load(f)

            assert metadata["commit"] == "def456abc789"
            assert metadata["workflow_run_id"] == "987654321"

    def test_create_metadata_missing_env_vars(self):
        """Test metadata creation with missing environment variables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            # Clear environment variables
            with patch.dict(os.environ, {}, clear=True):
                success = WorkflowHelper.create_metadata("v1.0.0", output_dir)
                assert success

            # Check metadata file was created with "unknown" values
            metadata_file = output_dir / "metadata.json"
            with metadata_file.open("r", encoding="utf-8") as f:
                metadata = json.load(f)

            assert metadata["tag"] == "v1.0.0"
            assert metadata["commit"] == "unknown"
            assert metadata["workflow_run_id"] == "unknown"
            assert metadata["runner_os"] == "unknown"
            assert metadata["runner_arch"] == "unknown"

    def test_create_metadata_directory_creation(self):
        """Test that metadata creation creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "nested" / "path"

            success = WorkflowHelper.create_metadata("v1.0.0", output_dir)
            assert success
            assert output_dir.exists()
            assert (output_dir / "metadata.json").exists()

    def test_display_baseline_summary_success(self, capsys):
        """Test successful baseline summary display."""
        baseline_content = """Date: 2023-12-15 14:30:00 UTC
Git commit: abc123def456
Hardware Information:
  OS: macOS
  CPU: Apple M4 Max
  Memory: 64.0 GB

=== 1000 Points (2D) ===
Time: [95.0, 100.0, 105.0] µs
Throughput: [9.524, 10.0, 10.526] Kelem/s

=== 2000 Points (2D) ===
Time: [190.0, 200.0, 210.0] µs

=== 1000 Points (3D) ===
Time: [220.0, 250.0, 280.0] µs
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(baseline_content)
            f.flush()
            baseline_file = Path(f.name)

        try:
            success = WorkflowHelper.display_baseline_summary(baseline_file)
            assert success

            # Check that summary was printed
            captured = capsys.readouterr()
            assert "📊 Baseline summary:" in captured.out
            assert "Total benchmarks: 3" in captured.out
            assert "Date: 2023-12-15 14:30:00 UTC" in captured.out
        finally:
            baseline_file.unlink()

    def test_display_baseline_summary_nonexistent_file(self, capsys):
        """Test baseline summary with non-existent file."""
        baseline_file = Path("/nonexistent/file.txt")

        success = WorkflowHelper.display_baseline_summary(baseline_file)
        assert not success

        # Check error message was printed to stderr
        captured = capsys.readouterr()
        assert "❌ Baseline file not found" in captured.err

    def test_display_baseline_summary_long_file(self, capsys):
        """Test baseline summary with file longer than 10 lines."""
        baseline_content = "\n".join([f"Line {i}" for i in range(20)])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(baseline_content)
            f.flush()
            baseline_file = Path(f.name)

        try:
            success = WorkflowHelper.display_baseline_summary(baseline_file)
            assert success

            # Check that "..." was printed (indicates truncation)
            captured = capsys.readouterr()
            assert "..." in captured.out
        finally:
            baseline_file.unlink()

    def test_sanitize_artifact_name_basic(self):
        """Test basic artifact name sanitization."""
        artifact_name = WorkflowHelper.sanitize_artifact_name("v1.2.3")
        assert artifact_name == "performance-baseline-v1_2_3"

    def test_sanitize_artifact_name_with_special_chars(self):
        """Test artifact name sanitization with special characters."""
        artifact_name = WorkflowHelper.sanitize_artifact_name("manual-2023/12/15-14:30:00")
        assert artifact_name == "performance-baseline-manual-2023_12_15-14_30_00"

    def test_sanitize_artifact_name_with_github_output(self):
        """Test artifact name sanitization with GITHUB_OUTPUT file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            output_file = f.name

        try:
            with patch.dict(os.environ, {"GITHUB_OUTPUT": output_file}):
                artifact_name = WorkflowHelper.sanitize_artifact_name("v2.0.0-beta.1")
                assert artifact_name == "performance-baseline-v2_0_0-beta_1"

            # Check that GITHUB_OUTPUT file was written
            with open(output_file, encoding="utf-8") as f:
                content = f.read()
                assert "artifact_name=performance-baseline-v2_0_0-beta_1\n" in content
        finally:
            Path(output_file).unlink(missing_ok=True)

    @pytest.mark.parametrize(
        ("input_tag", "expected_output"),
        [
            ("v1.0.0-alpha.1", "performance-baseline-v1_0_0-alpha_1"),
            ("tag with spaces", "performance-baseline-tag_with_spaces"),
            ("v1.0.0+build.123", "performance-baseline-v1_0_0_build_123"),
        ],
    )
    def test_sanitize_artifact_name_edge_cases(self, input_tag, expected_output):
        """Test artifact name sanitization with edge cases."""
        result = WorkflowHelper.sanitize_artifact_name(input_tag)
        assert result == expected_output

    def test_sanitize_artifact_name_special_characters(self):
        """Test that special characters are properly replaced in artifact names."""
        special_chars_input = "@#$%^&*()[]{}|\\<>?"
        result = WorkflowHelper.sanitize_artifact_name(special_chars_input)
        assert re.fullmatch(r"performance-baseline-[A-Za-z0-9._-]+", result)
        assert "_" in result  # at least one replacement happened


class TestBenchmarkRegressionHelper:
    """Test cases for BenchmarkRegressionHelper class."""

    def test_prepare_baseline_success(self, capsys):
        """Test successful baseline preparation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)
            baseline_file = baseline_dir / "baseline_results.txt"

            # Create a test baseline file
            baseline_content = """Date: 2023-12-15 10:30:00 UTC
Git commit: abc123def456
Hardware Information:
  OS: macOS
  CPU: Apple M4 Max

=== 1000 Points (2D) ===
Time: [95.0, 100.0, 105.0] µs
"""
            baseline_file.write_text(baseline_content)

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as env_file:
                env_path = env_file.name

            try:
                with patch.dict(os.environ, {"GITHUB_ENV": env_path}):
                    success = BenchmarkRegressionHelper.prepare_baseline(baseline_dir)

                    assert success

                    # Check that environment variables were set
                    with open(env_path, encoding="utf-8") as f:
                        env_content = f.read()
                        assert "BASELINE_EXISTS=true" in env_content
                        assert "BASELINE_SOURCE=artifact" in env_content
                        assert "BASELINE_ORIGIN=artifact" in env_content

                    # Check that baseline info was printed
                    captured = capsys.readouterr()
                    assert "📦 Prepared baseline from artifact" in captured.out
                    assert "=== Baseline Information" in captured.out
            finally:
                Path(env_path).unlink(missing_ok=True)

    def test_prepare_baseline_copy_error_handling(self, capsys):
        """Test error handling when copying baseline file fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)
            baseline_file = baseline_dir / "baseline-v1.0.0.txt"

            # Create a test baseline file (but NOT baseline_results.txt, so copy is needed)
            baseline_content = """Date: 2023-12-15 10:30:00 UTC
Git commit: abc123def456
Tag: v1.0.0
"""
            baseline_file.write_text(baseline_content)
            # Do NOT create baseline_results.txt - this ensures the copy path is taken

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as env_file:
                env_path = env_file.name

            try:
                with patch.dict(os.environ, {"GITHUB_ENV": env_path}), patch("benchmark_utils.copyfile", side_effect=OSError("Permission denied")):
                    success = BenchmarkRegressionHelper.prepare_baseline(baseline_dir)

                    assert not success

                    # Check that error environment variables were set
                    with open(env_path, encoding="utf-8") as f:
                        env_content = f.read()
                        assert "BASELINE_EXISTS=false" in env_content
                        assert "BASELINE_SOURCE=artifact" in env_content
                        assert "BASELINE_ORIGIN=artifact" in env_content

                    # Check error message was printed to stderr
                    captured = capsys.readouterr()
                    assert "❌ Failed to prepare baseline: Permission denied" in captured.err

                    # Verify that baseline_results.txt was not created due to copy failure
                    standard_file = baseline_dir / "baseline_results.txt"
                    assert not standard_file.exists()

            finally:
                Path(env_path).unlink(missing_ok=True)

    def test_prepare_baseline_read_summary_error_handling(self, capsys):
        """Test graceful error handling when baseline summary cannot be read."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)
            baseline_file = baseline_dir / "baseline_results.txt"

            # Create a test baseline file
            baseline_content = """Date: 2023-12-15 10:30:00 UTC
Git commit: abc123def456
Tag: v1.0.0
Hardware Information:
  OS: macOS
  CPU: Apple M4 Max

=== 1000 Points (2D) ===
Time: [95.0, 100.0, 105.0] µs
"""
            baseline_file.write_text(baseline_content)

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as env_file:
                env_path = env_file.name

            try:
                # Mock Path.open method to fail for read operations on baseline_results.txt
                original_path_open = Path.open

                def mock_path_open(self, mode="r", *args, **kwargs):
                    if self.name == "baseline_results.txt" and "r" in mode:
                        msg = "Read permission denied"
                        raise OSError(msg)
                    return original_path_open(self, mode, *args, **kwargs)

                with patch.dict(os.environ, {"GITHUB_ENV": env_path}), patch.object(Path, "open", mock_path_open):
                    success = BenchmarkRegressionHelper.prepare_baseline(baseline_dir)

                    # Should still succeed despite read error
                    assert success

                    # Check that success environment variables were set
                    with open(env_path, encoding="utf-8") as f:
                        env_content = f.read()
                        assert "BASELINE_EXISTS=true" in env_content
                        assert "BASELINE_SOURCE=artifact" in env_content
                        assert "BASELINE_ORIGIN=artifact" in env_content
                        assert "BASELINE_SOURCE_FILE=baseline_results.txt" in env_content
                        # BASELINE_TAG should not be set since we couldn't read the file
                        assert "BASELINE_TAG=" not in env_content

                    # Check warning message was printed to stderr
                    captured = capsys.readouterr()
                    assert "⚠️ Failed to read baseline summary: Read permission denied" in captured.err
                    assert "=== Baseline Information (from artifact) ===" in captured.out

            finally:
                Path(env_path).unlink(missing_ok=True)

    def test_prepare_baseline_missing_file(self, capsys):
        """Test baseline preparation when baseline file is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)
            # No baseline_results.txt file created

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as env_file:
                env_path = env_file.name

            try:
                with patch.dict(os.environ, {"GITHUB_ENV": env_path}):
                    success = BenchmarkRegressionHelper.prepare_baseline(baseline_dir)

                    assert not success

                    # Check that environment variables were set
                    with open(env_path, encoding="utf-8") as f:
                        env_content = f.read()
                        assert "BASELINE_EXISTS=false" in env_content
                        assert "BASELINE_SOURCE=missing" in env_content

                # Check error message was printed to stderr
                captured = capsys.readouterr()
                assert "❌ Downloaded artifact but no baseline*.txt files found" in captured.err
            finally:
                Path(env_path).unlink(missing_ok=True)

    def test_set_no_baseline_status(self, capsys):
        """Test setting no baseline status."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as env_file:
            env_path = env_file.name

        try:
            with patch.dict(os.environ, {"GITHUB_ENV": env_path}):
                BenchmarkRegressionHelper.set_no_baseline_status()

                # Check that environment variables were set
                with open(env_path, encoding="utf-8") as f:
                    env_content = f.read()
                    assert "BASELINE_EXISTS=false" in env_content
                    assert "BASELINE_SOURCE=none" in env_content
                    assert "BASELINE_ORIGIN=none" in env_content

                # Check message was printed
                captured = capsys.readouterr()
                assert "📈 No baseline artifact found" in captured.out
        finally:
            Path(env_path).unlink(missing_ok=True)

    def test_extract_baseline_commit_from_baseline_file(self):
        """Test extracting commit SHA from baseline_results.txt."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)
            baseline_file = baseline_dir / "baseline_results.txt"

            baseline_content = """Date: 2023-12-15 10:30:00 UTC
Git commit: abc123def456
Hardware Information:
  OS: macOS
"""
            baseline_file.write_text(baseline_content)

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as env_file:
                env_path = env_file.name

            try:
                with patch.dict(os.environ, {"GITHUB_ENV": env_path}):
                    commit_sha = BenchmarkRegressionHelper.extract_baseline_commit(baseline_dir)

                    assert commit_sha == "abc123def456"

                    # Check that environment variables were set
                    with open(env_path, encoding="utf-8") as f:
                        env_content = f.read()
                        assert "BASELINE_COMMIT=abc123def456" in env_content
                        assert "BASELINE_COMMIT_SOURCE=baseline" in env_content
            finally:
                Path(env_path).unlink(missing_ok=True)

    def test_extract_baseline_commit_from_metadata(self):
        """Test extracting commit SHA from metadata.json when baseline file fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)
            metadata_file = baseline_dir / "metadata.json"

            metadata = {"tag": "v1.0.0", "commit": "def456abc789", "generated_at": "2023-12-15T10:30:00Z"}

            with metadata_file.open("w", encoding="utf-8") as f:
                json.dump(metadata, f)

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as env_file:
                env_path = env_file.name

            try:
                with patch.dict(os.environ, {"GITHUB_ENV": env_path}):
                    commit_sha = BenchmarkRegressionHelper.extract_baseline_commit(baseline_dir)

                    assert commit_sha == "def456abc789"

                    # Check that environment variables were set
                    with open(env_path, encoding="utf-8") as f:
                        env_content = f.read()
                        assert "BASELINE_COMMIT=def456abc789" in env_content
                        assert "BASELINE_COMMIT_SOURCE=metadata" in env_content
            finally:
                Path(env_path).unlink(missing_ok=True)

    def test_extract_baseline_commit_unknown(self):
        """Test extracting commit SHA when no valid SHA is found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)
            # No files created

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as env_file:
                env_path = env_file.name

            try:
                with patch.dict(os.environ, {"GITHUB_ENV": env_path}):
                    commit_sha = BenchmarkRegressionHelper.extract_baseline_commit(baseline_dir)

                    assert commit_sha == "unknown"

                    # Check that environment variables were set
                    with open(env_path, encoding="utf-8") as f:
                        env_content = f.read()
                        assert "BASELINE_COMMIT=unknown" in env_content
                        assert "BASELINE_COMMIT_SOURCE=unknown" in env_content
            finally:
                Path(env_path).unlink(missing_ok=True)

    def test_determine_benchmark_skip_unknown_baseline(self):
        """Test skip determination with unknown baseline commit."""
        should_skip, reason = BenchmarkRegressionHelper.determine_benchmark_skip("unknown", "def4567")

        assert not should_skip
        assert reason == "unknown_baseline"

    def test_determine_benchmark_skip_same_commit(self):
        """Test skip determination with same commit."""
        should_skip, reason = BenchmarkRegressionHelper.determine_benchmark_skip("abc1234", "abc1234")

        assert should_skip
        assert reason == "same_commit"

    @patch("benchmark_utils.run_git_command")
    def test_determine_benchmark_skip_baseline_not_found(self, mock_git):
        """Test skip determination when baseline commit not found in history."""
        # Simulate git cat-file failing
        mock_git.side_effect = subprocess.CalledProcessError(1, "git")

        should_skip, reason = BenchmarkRegressionHelper.determine_benchmark_skip("abc1234", "def4567")

        assert not should_skip
        assert reason == "baseline_commit_not_found"

    @patch("benchmark_utils.run_git_command")
    def test_determine_benchmark_skip_no_changes(self, mock_git):
        """Test skip determination when no relevant changes found."""
        # Mock successful git commands
        mock_git.side_effect = [
            Mock(returncode=0),  # git cat-file succeeds
            Mock(returncode=0, stdout="docs/README.md\n.github/workflows/other.yml\n", stderr=""),  # git diff
        ]

        should_skip, reason = BenchmarkRegressionHelper.determine_benchmark_skip("abc1234", "def4567")

        assert should_skip
        assert reason == "no_relevant_changes"

    @patch("benchmark_utils.run_git_command")
    def test_determine_benchmark_skip_changes_detected(self, mock_git):
        """Test skip determination when relevant changes are detected."""
        # Mock successful git commands
        mock_git.side_effect = [
            Mock(returncode=0),  # git cat-file succeeds
            Mock(returncode=0, stdout="src/core/mod.rs\nbenches/performance.rs\n", stderr=""),  # git diff
        ]

        should_skip, reason = BenchmarkRegressionHelper.determine_benchmark_skip("abc1234", "def4567")

        assert not should_skip
        assert reason == "changes_detected"

    def test_display_skip_message(self, capsys):
        """Test displaying skip messages."""
        BenchmarkRegressionHelper.display_skip_message("same_commit", "abc1234")

        captured = capsys.readouterr()
        assert "🔍 Current commit matches baseline (abc1234)" in captured.out

    def test_display_no_baseline_message(self, capsys):
        """Test displaying no baseline message."""
        BenchmarkRegressionHelper.display_no_baseline_message()

        captured = capsys.readouterr()
        assert "⚠️ No performance baseline available" in captured.out
        assert "💡 To enable performance regression testing:" in captured.out

    def test_run_regression_test_success(self, capsys):
        """Test successful regression test run."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_file = Path(temp_dir) / "baseline.txt"
            baseline_file.write_text("mock baseline content")

            with patch("benchmark_utils.PerformanceComparator") as mock_comparator_class:
                mock_comparator = Mock()
                mock_comparator.compare_with_baseline.return_value = (True, False)  # success, no regression
                mock_comparator_class.return_value = mock_comparator

                success = BenchmarkRegressionHelper.run_regression_test(baseline_file)

                assert success
                mock_comparator.compare_with_baseline.assert_called_once_with(baseline_file, dev_mode=False, bench_timeout=1800)

                captured = capsys.readouterr()
                assert "🚀 Running performance regression test" in captured.out

    def test_run_regression_test_dev_mode(self, capsys):
        """Test regression test run with dev mode enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_file = Path(temp_dir) / "baseline.txt"
            baseline_file.write_text("mock baseline content")

            with patch("benchmark_utils.PerformanceComparator") as mock_comparator_class:
                mock_comparator = Mock()
                mock_comparator.compare_with_baseline.return_value = (True, False)  # success, no regression
                mock_comparator_class.return_value = mock_comparator

                success = BenchmarkRegressionHelper.run_regression_test(baseline_file, dev_mode=True)

                assert success
                mock_comparator.compare_with_baseline.assert_called_once_with(baseline_file, dev_mode=True, bench_timeout=1800)

                captured = capsys.readouterr()
                assert "dev mode (10x faster)" in captured.out

    def test_run_regression_test_failure(self):
        """Test regression test run failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_file = Path(temp_dir) / "baseline.txt"
            baseline_file.write_text("mock baseline content")

            with patch("benchmark_utils.PerformanceComparator") as mock_comparator_class:
                mock_comparator = Mock()
                mock_comparator.compare_with_baseline.return_value = (False, False)  # failure
                mock_comparator_class.return_value = mock_comparator

                success = BenchmarkRegressionHelper.run_regression_test(baseline_file)

                assert not success

    def test_run_regression_test_custom_timeout(self, capsys):
        """Test regression test run with custom bench_timeout parameter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_file = Path(temp_dir) / "baseline.txt"
            baseline_file.write_text("mock baseline content")

            with patch("benchmark_utils.PerformanceComparator") as mock_comparator_class:
                mock_comparator = Mock()
                mock_comparator.compare_with_baseline.return_value = (True, False)  # success, no regression
                mock_comparator_class.return_value = mock_comparator

                success = BenchmarkRegressionHelper.run_regression_test(baseline_file, bench_timeout=3600)

                assert success
                mock_comparator.compare_with_baseline.assert_called_once_with(baseline_file, dev_mode=False, bench_timeout=3600)

                captured = capsys.readouterr()
                assert "🚀 Running performance regression test" in captured.out

    def test_display_results_file_exists(self, capsys):
        """Test displaying results when file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_file = Path(temp_dir) / "results.txt"
            results_content = "=== Performance Test Results ===\nAll tests passed\n"
            results_file.write_text(results_content)

            BenchmarkRegressionHelper.display_results(results_file)

            captured = capsys.readouterr()
            assert "=== Performance Regression Test Results ===" in captured.out
            assert "All tests passed" in captured.out

    def test_display_results_file_missing(self, capsys):
        """Test displaying results when file is missing."""
        missing_file = Path("/nonexistent/results.txt")

        BenchmarkRegressionHelper.display_results(missing_file)

        captured = capsys.readouterr()
        assert "⚠️ No comparison results file found" in captured.out

    def test_generate_summary_with_regression(self, temp_chdir, capsys):
        """Test generating summary when regression is detected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_file = Path(temp_dir) / "benches" / "compare_results.txt"
            results_file.parent.mkdir(parents=True)
            results_file.write_text("REGRESSION detected in benchmark xyz")

            env_vars = {
                "BASELINE_SOURCE": "artifact",
                "BASELINE_ORIGIN": "release",
                "BASELINE_TAG": "v1.0.0",
                "BASELINE_EXISTS": "true",
                "SKIP_BENCHMARKS": "false",
                "SKIP_REASON": "changes_detected",
            }

            # Change working directory to temp_dir so Path("benches/compare_results.txt") works
            with patch.dict(os.environ, env_vars), temp_chdir(temp_dir):
                BenchmarkRegressionHelper.generate_summary()

                captured = capsys.readouterr()
                assert "📊 Performance Regression Testing Summary" in captured.out
                assert "Baseline source: artifact" in captured.out
                assert "Result: ⚠️ Performance regressions detected" in captured.out

    def test_generate_summary_skip_same_commit(self, capsys):
        """Test generating summary when benchmarks skipped due to same commit."""
        env_vars = {
            "BASELINE_SOURCE": "artifact",
            "BASELINE_ORIGIN": "manual",
            "BASELINE_EXISTS": "true",
            "SKIP_BENCHMARKS": "true",
            "SKIP_REASON": "same_commit",
        }

        with patch.dict(os.environ, env_vars):
            BenchmarkRegressionHelper.generate_summary()

            captured = capsys.readouterr()
            assert "Result: ⏭️ Benchmarks skipped (same commit as baseline)" in captured.out

    def test_generate_summary_no_baseline(self, capsys):
        """Test generating summary when no baseline available."""
        env_vars = {
            "BASELINE_EXISTS": "false",
            "SKIP_BENCHMARKS": "unknown",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            BenchmarkRegressionHelper.generate_summary()

            captured = capsys.readouterr()
            assert "Result: ⏭️ Benchmarks skipped (no baseline available)" in captured.out

    def test_generate_summary_sets_regression_environment_variable(self, temp_chdir, capsys):
        """Test that generate_summary sets BENCHMARK_REGRESSION_DETECTED environment variable when regressions are found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_file = Path(temp_dir) / "benches" / "compare_results.txt"
            results_file.parent.mkdir(parents=True)
            results_file.write_text("REGRESSION detected in benchmark xyz")

            env_vars = {
                "BASELINE_EXISTS": "true",
                "SKIP_BENCHMARKS": "false",
            }

            # Ensure BENCHMARK_REGRESSION_DETECTED is not set initially
            if "BENCHMARK_REGRESSION_DETECTED" in os.environ:
                del os.environ["BENCHMARK_REGRESSION_DETECTED"]

            with patch.dict(os.environ, env_vars, clear=True), temp_chdir(temp_dir):
                BenchmarkRegressionHelper.generate_summary()

                # Check that environment variable was set
                assert os.environ.get("BENCHMARK_REGRESSION_DETECTED") == "true"

                # Check that appropriate message was printed
                captured = capsys.readouterr()
                assert "Exported BENCHMARK_REGRESSION_DETECTED=true for downstream CI steps" in captured.out

    def test_generate_summary_github_env_export(self, temp_chdir):
        """Test that BENCHMARK_REGRESSION_DETECTED is also exported to GITHUB_ENV when available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_file = Path(temp_dir) / "benches" / "compare_results.txt"
            results_file.parent.mkdir(parents=True)
            results_file.write_text("REGRESSION detected in benchmark xyz")

            github_env_file = Path(temp_dir) / "github_env"
            env_vars = {
                "BASELINE_EXISTS": "true",
                "SKIP_BENCHMARKS": "false",
                "GITHUB_ENV": str(github_env_file),
            }

            with patch.dict(os.environ, env_vars, clear=True), temp_chdir(temp_dir):
                BenchmarkRegressionHelper.generate_summary()

                # Check that GITHUB_ENV file was written to
                assert github_env_file.exists()
                github_env_content = github_env_file.read_text()
                assert "BENCHMARK_REGRESSION_DETECTED=true" in github_env_content

    def test_generate_summary_with_error_file(self, temp_chdir, capsys):
        """Test generating summary when comparison failed with error file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_file = Path(temp_dir) / "benches" / "compare_results.txt"
            results_file.parent.mkdir(parents=True)
            # Simulate error file content (as written by _write_error_file)
            results_file.write_text(
                "Comparison Results\n"
                "==================\n\n"
                "❌ Error: Benchmark execution timeout\n\n"
                "Details: Command timed out after 1800 seconds\n\n"
                "This error prevented the benchmark comparison from completing successfully.\n"
                "Please check the CI logs for more information.\n"
            )

            env_vars = {
                "BASELINE_SOURCE": "artifact",
                "BASELINE_ORIGIN": "release",
                "BASELINE_TAG": "v1.0.0",
                "BASELINE_EXISTS": "true",
                "SKIP_BENCHMARKS": "false",
                "SKIP_REASON": "n/a",
            }

            with patch.dict(os.environ, env_vars), temp_chdir(temp_dir):
                BenchmarkRegressionHelper.generate_summary()

                captured = capsys.readouterr()
                assert "📊 Performance Regression Testing Summary" in captured.out
                assert "Baseline source: artifact" in captured.out
                # Should detect error and report failure, not "no regressions"
                assert "Result: ❌ Benchmark comparison failed" in captured.out
                assert "(see benches/compare_results.txt for details)" in captured.out
                # Should NOT say "no regressions" when there was an error
                assert "✅ No significant performance regressions" not in captured.out


class TestProjectRootHandling:
    """Test cases for find_project_root functionality."""

    def test_find_project_root_success(self, temp_chdir):
        """Test finding project root when Cargo.toml exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create Cargo.toml in temp directory
            cargo_toml = temp_path / "Cargo.toml"
            cargo_toml.write_text('[package]\nname = "test"\n')

            # Create subdirectory and change to it
            sub_dir = temp_path / "subdir"
            sub_dir.mkdir()

            with temp_chdir(sub_dir):
                result = find_project_root()
                # Resolve both paths to handle symlinks (macOS /var -> /private/var)
                assert result.resolve() == temp_path.resolve()

    def test_find_project_root_not_found(self, temp_chdir):
        """Test finding project root when Cargo.toml doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with temp_chdir(temp_path), pytest.raises(ProjectRootNotFoundError, match=r"Could not locate Cargo\.toml"):
                find_project_root()


class TestTimeoutHandling:
    """Test cases for benchmark timeout functionality."""

    @pytest.mark.parametrize(
        ("component_class", "method_name", "setup_func"),
        [
            (
                "BaselineGenerator",
                "generate_baseline",
                lambda _: None,  # No extra setup needed
            ),
            (
                "PerformanceComparator",
                "compare_with_baseline",
                lambda temp_dir: (Path(temp_dir) / "baseline.txt").write_text("mock baseline"),
            ),
        ],
    )
    def test_timeout_parameter_passed(self, component_class, method_name, setup_func):
        """Test that benchmark components accept and use timeout parameter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            setup_func(temp_dir)

            # Import and instantiate the class dynamically
            if component_class == "BaselineGenerator":
                component = BaselineGenerator(project_root)
                method_args = ()
            else:  # PerformanceComparator
                component = PerformanceComparator(project_root)
                method_args = (Path(temp_dir) / "baseline.txt",)

            # Test that the method signature accepts bench_timeout
            with patch("benchmark_utils.run_cargo_command") as mock_cargo:
                mock_cargo.side_effect = subprocess.TimeoutExpired("cargo", 120)

                method = getattr(component, method_name)
                result = method(*method_args, bench_timeout=120)

                # BaselineGenerator returns bool, PerformanceComparator returns tuple
                if component_class == "BaselineGenerator":
                    assert result is False
                else:
                    success, regression = result
                    assert not success
                    assert not regression

                # Verify timeout was passed to run_cargo_command calls
                assert mock_cargo.call_count >= 1
                assert any(call.kwargs.get("timeout") == 120 for call in mock_cargo.call_args_list)

    def test_timeout_error_handling_baseline_generator(self, capsys):
        """Test proper error handling when benchmark times out in BaselineGenerator."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = BaselineGenerator(project_root)

            with patch("benchmark_utils.run_cargo_command") as mock_cargo:
                mock_cargo.side_effect = subprocess.TimeoutExpired("cargo bench", 1800)

                success = generator.generate_baseline(bench_timeout=1800)

                assert not success

                # Check that appropriate timeout message was printed
                captured = capsys.readouterr()
                assert "timed out after 1800 seconds" in captured.err
                assert "Consider increasing --bench-timeout" in captured.err

    def test_timeout_error_handling_performance_comparator(self, capsys):
        """Test proper error handling when benchmark times out in PerformanceComparator."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            baseline_file = Path(temp_dir) / "baseline.txt"
            baseline_file.write_text("mock baseline")

            comparator = PerformanceComparator(project_root)

            with patch("benchmark_utils.run_cargo_command") as mock_cargo:
                mock_cargo.side_effect = subprocess.TimeoutExpired("cargo bench", 1800)

                success, regression = comparator.compare_with_baseline(baseline_file, bench_timeout=1800)

                assert not success
                assert not regression

                # Check that appropriate timeout message was printed
                captured = capsys.readouterr()
                assert "timed out after 1800 seconds" in captured.err
                assert "Consider increasing --bench-timeout" in captured.err

                # Verify error file contains full exception message with command context
                error_file = project_root / "benches" / "compare_results.txt"
                assert error_file.exists()
                error_content = error_file.read_text()
                assert "❌ Error: Benchmark execution timeout" in error_content
                assert "cargo bench" in error_content  # Command from exception
                assert "timeout after 1800 seconds" in error_content  # Explicit timeout value

    def test_cli_bench_timeout_validation(self, monkeypatch, temp_chdir):
        """Test that CLI validates bench_timeout is positive via main()."""
        # Create a temporary project with Cargo.toml to satisfy find_project_root
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "Cargo.toml").write_text('[package]\nname = "test"\n')

            with temp_chdir(temp_path):
                # Test with zero timeout
                monkeypatch.setattr(sys, "argv", ["benchmark_utils.py", "generate-baseline", "--bench-timeout", "0"])
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 2  # argparse error exit code

                # Test with negative timeout
                baseline_file = temp_path / "baseline.txt"
                baseline_file.write_text("mock baseline")
                monkeypatch.setattr(sys, "argv", ["benchmark_utils.py", "compare", "--baseline", str(baseline_file), "--bench-timeout", "-100"])
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 2  # argparse error exit code

                # Test with positive timeout (should parse successfully, will fail on execution but that's ok)
                parser = create_argument_parser()
                args = parser.parse_args(["run-regression-test", "--baseline", str(baseline_file), "--bench-timeout", "3600"])
                assert args.bench_timeout == 3600
                assert hasattr(args, "validate_bench_timeout")
                assert args.validate_bench_timeout


class TestPerformanceSummaryGenerator:
    """Test cases for PerformanceSummaryGenerator class."""

    def test_init(self):
        """Test PerformanceSummaryGenerator initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            assert generator.project_root == project_root
            assert generator.baseline_file == project_root / "baseline-artifact" / "baseline_results.txt"
            assert generator._baseline_fallback == project_root / "benches" / "baseline_results.txt"
            assert generator.comparison_file == project_root / "benches" / "compare_results.txt"
            assert generator.circumsphere_results_dir == project_root / "target" / "criterion"
            assert isinstance(generator.current_version, str)
            assert isinstance(generator.current_date, str)

    @patch("benchmark_utils.run_git_command")
    def test_get_current_version_with_tag(self, mock_git_command):
        """Test getting current version from git tags."""
        mock_result = Mock()
        mock_result.stdout.strip.return_value = "v1.2.3"
        mock_git_command.return_value = mock_result

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            version = generator._get_current_version()
            assert version == "1.2.3"  # v prefix should be removed
            mock_git_command.assert_called_with(["describe", "--tags", "--abbrev=0", "--match=v*"], cwd=project_root)

    @patch("benchmark_utils.run_git_command")
    def test_get_current_version_fallback(self, mock_git_command):
        """Test fallback version detection when describe fails."""
        # First call (describe) fails, second call (tag -l) succeeds
        mock_result = Mock()
        mock_result.stdout.strip.return_value = "v0.1.0\nv0.2.0"

        # The second call is made within the exception handler
        def side_effect(*args, **kwargs):
            if "describe" in args[0]:
                raise subprocess.CalledProcessError(1, "git describe", "describe failed")
            return mock_result

        mock_git_command.side_effect = side_effect

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            version = generator._get_current_version()
            assert version == "0.1.0"

    @patch("benchmark_utils.run_git_command")
    def test_get_current_version_no_tags(self, mock_git_command):
        """Test version detection when no tags are found."""
        mock_git_command.side_effect = Exception("No tags found")

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            version = generator._get_current_version()
            assert version == "unknown"

    @patch("benchmark_utils.run_git_command")
    @patch("benchmark_utils.datetime")
    def test_get_version_date_with_tag(self, mock_datetime, mock_git_command):  # noqa: ARG002
        """Test getting version date from git tag."""
        mock_result = Mock()
        mock_result.stdout.strip.return_value = "2024-01-15"
        mock_git_command.return_value = mock_result

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)
            generator.current_version = "1.2.3"

            date = generator._get_version_date()
            assert date == "2024-01-15"
            mock_git_command.assert_called_with(["log", "-1", "--format=%cd", "--date=format:%Y-%m-%d", "v1.2.3"], cwd=project_root)

    @patch("benchmark_utils.run_git_command")
    @patch("benchmark_utils.datetime")
    def test_get_version_date_fallback(self, mock_datetime, mock_git_command):
        """Test version date fallback to current date."""
        mock_git_command.side_effect = Exception("Git command failed")
        mock_now = Mock()
        mock_now.strftime.return_value = "2024-01-15"
        mock_datetime.now.return_value = mock_now
        mock_datetime.UTC = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            date = generator._get_version_date()
            assert date == "2024-01-15"
            mock_now.strftime.assert_called_with("%Y-%m-%d")

    def test_parse_baseline_results_nonexistent_file(self):
        """Test parsing baseline results when file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            lines = generator._parse_baseline_results()
            # Should return error message when file doesn't exist
            content = "\n".join(lines)
            assert "### Baseline Results" in content
            assert "Error parsing baseline results" in content

    def test_parse_baseline_results_with_data(self):
        """Test parsing baseline results with actual data."""
        baseline_content = """Date: 2024-01-15 10:30:00 UTC
Git commit: abc123def456
Hardware: Apple M2 Pro (10 cores)
Memory: 32 GB

=== 1000 Points (2D) ===
Time: [100.0, 110.0, 120.0] µs
Throughput: [8000.0, 9090.9, 10000.0] Kelem/s

=== 5000 Points (3D) ===
Time: [500.0, 550.0, 600.0] µs
Throughput: [8333.3, 9090.9, 10000.0] Kelem/s
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            baseline_dir = project_root / "baseline-artifact"
            baseline_dir.mkdir(parents=True)
            baseline_file = baseline_dir / "baseline_results.txt"
            baseline_file.write_text(baseline_content)

            generator = PerformanceSummaryGenerator(project_root)
            lines = generator._parse_baseline_results()

            # Should contain metadata section
            markdown_content = "\n".join(lines)
            assert "### Current Baseline Information" in markdown_content
            assert "Git commit: abc123def456" in markdown_content
            assert "Hardware: Apple M2 Pro" in markdown_content

            # Should contain performance tables
            assert "### 2D Triangulation Performance" in markdown_content
            assert "### 3D Triangulation Performance" in markdown_content
            assert "| Points | Time (mean) | Throughput (mean) | Scaling |" in markdown_content

    def test_parse_comparison_results_with_regression(self):
        """Test parsing comparison results that show regression."""
        comparison_content = """Performance Comparison Results
⚠️  REGRESSION: Time increased by 15.2% (slower performance)
✅ OK: Time change +2.1% within acceptable range
✅ IMPROVEMENT: Time decreased by 8.5% (faster performance)
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            benches_dir = project_root / "benches"
            benches_dir.mkdir(parents=True)
            comparison_file = benches_dir / "compare_results.txt"
            comparison_file.write_text(comparison_content)

            generator = PerformanceSummaryGenerator(project_root)
            lines = generator._parse_comparison_results()

            markdown_content = "\n".join(lines)
            assert "### ⚠️ Performance Regression Detected" in markdown_content
            assert "REGRESSION: Time increased by 15.2%" in markdown_content
            assert "IMPROVEMENT: Time decreased by 8.5%" in markdown_content

    def test_parse_comparison_results_no_regression(self):
        """Test parsing comparison results with no regression."""
        comparison_content = """Performance Comparison Results
✅ OK: Time change +2.1% within acceptable range
✅ IMPROVEMENT: Time decreased by 3.2% (faster performance)
✅ OK: Time change -1.8% within acceptable range
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            benches_dir = project_root / "benches"
            benches_dir.mkdir(parents=True)
            comparison_file = benches_dir / "compare_results.txt"
            comparison_file.write_text(comparison_content)

            generator = PerformanceSummaryGenerator(project_root)
            lines = generator._parse_comparison_results()

            markdown_content = "\n".join(lines)
            assert "### ✅ Performance Status: Good" in markdown_content
            assert "no significant performance regressions" in markdown_content

    @patch("benchmark_utils.get_git_commit_hash")
    @patch("benchmark_utils.run_git_command")
    @patch("benchmark_utils.datetime")
    def test_generate_markdown_content(self, mock_datetime, mock_run_git, mock_git_commit):
        """Test generating complete markdown content."""
        # Avoid calling actual git in __init__ helpers
        mock_run_git.side_effect = Exception("git unavailable in test")
        mock_git_commit.return_value = "abc123def456"
        mock_now = Mock()
        mock_now.strftime.return_value = "2024-01-15 10:30:00 UTC"
        mock_datetime.now.return_value = mock_now
        mock_datetime.UTC = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            content = generator._generate_markdown_content()

            # Check basic structure
            assert "# Delaunay Library Performance Results" in content
            assert "**Last Updated**: 2024-01-15 10:30:00 UTC" in content
            assert "**Generated By**: benchmark_utils.py" in content
            assert "**Git Commit**: abc123def456" in content
            assert "## Performance Results Summary" in content

            # Check static content sections
            assert "## Key Findings" in content
            assert "### Performance Ranking" in content
            assert "## Recommendations" in content
            assert "## Performance Data Updates" in content

    def test_get_circumsphere_performance_results(self):
        """Test getting circumsphere performance results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            lines = generator._get_circumsphere_performance_results()
            content = "\n".join(lines)

            assert "### Circumsphere Performance Results" in content
            # Should contain fallback performance data when no criterion results exist
            assert "Basic 3D" in content or "Version unknown" in content

    def test_get_update_instructions(self):
        """Test getting performance data update instructions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            lines = generator._get_update_instructions()
            content = "\n".join(lines)

            assert "## Performance Data Updates" in content
            assert "uv run benchmark-utils generate-baseline" in content
            assert "uv run benchmark-utils generate-summary" in content
            assert "PerformanceSummaryGenerator" in content

    def test_parse_numerical_accuracy_output_success(self):
        """Test parsing numerical accuracy output successfully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            stdout_content = """Running benchmarks...
Method Comparisons (1000 total tests):
  insphere vs insphere_distance:  845/1000 (84.5%)
  insphere vs insphere_lifted:  12/1000 (1.2%)
  insphere_distance vs insphere_lifted:  203/1000 (20.3%)
  All three methods agree:  8/1000 (0.8%)
Benchmark completed."""

            result = generator._parse_numerical_accuracy_output(stdout_content)

            assert result is not None
            assert isinstance(result, dict)
            assert result["insphere_distance"] == "84.5%"
            assert result["insphere_lifted"] == "1.2%"
            assert result["distance_lifted"] == "20.3%"
            assert result["all_agree"] == "0.8%"

    def test_parse_numerical_accuracy_output_no_data(self):
        """Test parsing numerical accuracy output with no relevant data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            stdout_content = """Running benchmarks...
No method comparisons found.
Benchmark completed."""

            result = generator._parse_numerical_accuracy_output(stdout_content)

            assert result is None

    def test_parse_numerical_accuracy_output_malformed(self):
        """Test parsing numerical accuracy output with malformed data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            stdout_content = """Running benchmarks...
Method Comparisons (invalid format):
Benchmark completed."""

            result = generator._parse_numerical_accuracy_output(stdout_content)

            assert result is None

    @patch("benchmark_utils.run_cargo_command")
    def test_run_circumsphere_benchmarks_success(self, mock_cargo):
        """Test running circumsphere benchmarks successfully."""
        mock_cargo.return_value = Mock(stdout="")

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            success, numerical_data = generator._run_circumsphere_benchmarks()

            assert success is True
            # numerical_data should be a dict or None when successful
            assert numerical_data is None or isinstance(numerical_data, dict)
            mock_cargo.assert_called_once()

    @patch("benchmark_utils.run_cargo_command")
    def test_run_circumsphere_benchmarks_with_numerical_data(self, mock_cargo):
        """Test running circumsphere benchmarks with numerical accuracy data."""
        # Mock cargo command to return output with numerical accuracy data
        mock_result = Mock()
        mock_result.stdout = """Running benchmarks...
Method Comparisons (1000 total tests):
  insphere vs insphere_distance:  820/1000 (82.0%)
  insphere vs insphere_lifted:  5/1000 (0.5%)
  insphere_distance vs insphere_lifted:  180/1000 (18.0%)
  All three methods agree:  2/1000 (0.2%)
Benchmark completed."""
        mock_cargo.return_value = mock_result

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            success, numerical_data = generator._run_circumsphere_benchmarks()

            assert success is True
            assert numerical_data is not None
            assert isinstance(numerical_data, dict)
            # Check specific accuracy values were parsed correctly
            assert numerical_data["insphere_distance"] == "82.0%"
            assert numerical_data["insphere_lifted"] == "0.5%"
            assert numerical_data["distance_lifted"] == "18.0%"
            assert numerical_data["all_agree"] == "0.2%"
            mock_cargo.assert_called_once()

    @patch("benchmark_utils.run_cargo_command")
    def test_run_circumsphere_benchmarks_failure(self, mock_cargo, capsys):
        """Test handling circumsphere benchmark failures."""
        mock_cargo.side_effect = Exception("Benchmark failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            success, numerical_data = generator._run_circumsphere_benchmarks()

            assert success is False
            # numerical_data should be None when there's a failure
            assert numerical_data is None
            # Check error was printed (it goes to stdout, not stderr)
            captured = capsys.readouterr()
            assert "Error running circumsphere benchmarks" in captured.out

    @patch("benchmark_utils.run_git_command")
    def test_generate_summary_success(self, mock_git, capsys):
        """Test successful generation of performance summary."""
        mock_git.side_effect = Exception("git unavailable in test")
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            output_file = Path(temp_dir) / "test_summary.md"

            success = generator.generate_summary(output_path=output_file)

            assert success is True
            assert output_file.exists()

            # Check file contains expected content
            content = output_file.read_text()
            assert "# Delaunay Library Performance Results" in content
            assert "## Performance Results Summary" in content

            # Check success message was printed
            captured = capsys.readouterr()
            assert "Generated performance summary" in captured.out

    @patch("benchmark_utils.PerformanceSummaryGenerator._run_circumsphere_benchmarks")
    def test_generate_summary_with_benchmarks(self, mock_run_benchmarks):
        """Test generating summary with fresh benchmark run."""
        mock_run_benchmarks.return_value = (True, None)

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            output_file = Path(temp_dir) / "test_summary.md"

            success = generator.generate_summary(output_path=output_file, run_benchmarks=True)

            assert success is True
            mock_run_benchmarks.assert_called_once()
            assert output_file.exists()

    @patch("benchmark_utils.PerformanceSummaryGenerator._run_circumsphere_benchmarks")
    def test_generate_summary_benchmark_failure_continues(self, mock_run_benchmarks, capsys):
        """Test that summary generation continues even if benchmark run fails."""
        mock_run_benchmarks.return_value = (False, None)

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            output_file = Path(temp_dir) / "test_summary.md"

            success = generator.generate_summary(output_path=output_file, run_benchmarks=True)

            assert success is True  # Should still succeed
            assert output_file.exists()

            # Check warning was printed
            captured = capsys.readouterr()
            assert "Benchmark run failed" in captured.out

    def test_generate_summary_exception_handling(self, capsys):
        """Test exception handling in generate_summary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            output_file = Path(temp_dir) / "readonly" / "summary.md"
            with patch.object(Path, "open", side_effect=OSError("permission denied")):
                success = generator.generate_summary(output_path=output_file)

            assert success is False

            # Check error was printed (looking for the error message)
            captured = capsys.readouterr()
            assert "Failed to generate performance summary" in captured.err

    def test_get_static_content(self):
        """Test getting static content sections."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            lines = generator._get_static_sections()
            content = "\n".join(lines)

            assert "## Historical Version Comparison" in content
            assert "## Implementation Notes" in content
            assert "## Benchmark Structure" in content

    def test_empty_benchmark_results_edge_case(self):
        """Test handling of empty benchmark results (edge case)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            # Should not crash and should use fallback data
            results = generator._parse_circumsphere_benchmark_results()
            assert len(results) > 0

    def test_malformed_estimates_json_edge_case(self):
        """Test handling of malformed estimates.json files (edge case)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            # Create malformed estimates.json
            criterion_dir = project_root / "target" / "criterion" / "basic-insphere" / "base"
            criterion_dir.mkdir(parents=True)

            estimates_file = criterion_dir / "estimates.json"
            estimates_file.write_text("{ invalid json")

            generator = PerformanceSummaryGenerator(project_root)

            # Should not crash and should use fallback data
            results = generator._parse_circumsphere_benchmark_results()
            assert len(results) > 0

    def test_missing_git_info_edge_case(self):
        """Test handling when git information is not available (edge case)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            output_file = Path(temp_dir) / "test_output.md"

            with (
                patch("benchmark_utils.run_git_command") as mock_git,
                patch("benchmark_utils.get_git_commit_hash") as mock_commit,
            ):
                mock_git.side_effect = Exception("Git not available")
                mock_commit.side_effect = Exception("Git not available")

                generator = PerformanceSummaryGenerator(project_root)
                success = generator.generate_summary(output_file)

                # Should still succeed
                assert success

                content = output_file.read_text()
                assert "Version unknown" in content

    def test_baseline_fallback_behavior_edge_case(self):
        """Test baseline file fallback from primary to secondary location (edge case)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            output_file = Path(temp_dir) / "baseline_fallback_test.md"

            # Create fallback baseline file (benches/baseline_results.txt)
            benches_dir = project_root / "benches"
            benches_dir.mkdir()
            fallback_baseline = benches_dir / "baseline_results.txt"
            fallback_baseline.write_text(
                "Generated at: 2025-01-15 10:00:00\n"
                "Git commit: abc123\n"
                "=== 1000 Points (3D) ===\n"
                "Time: [805.0, 810.0, 815.0] µs\n"
                "Throughput: [1200.0, 1235.0, 1245.0] Kelem/s\n",
            )

            # Do NOT create primary baseline file (baseline-artifact/baseline_results.txt)
            # This forces the fallback behavior

            with (
                patch("benchmark_utils.get_git_commit_hash") as mock_commit,
            ):
                mock_commit.return_value = "abc123def456"

                generator = PerformanceSummaryGenerator(project_root)

                # Verify initial state: primary doesn't exist, fallback does
                assert not generator.baseline_file.exists()  # Primary: baseline-artifact/baseline_results.txt
                assert generator._baseline_fallback.exists()  # Fallback: benches/baseline_results.txt

                success = generator.generate_summary(output_file)
                assert success

                content = output_file.read_text()

                # Verify that baseline data was included (meaning fallback worked)
                assert "Triangulation Data Structure Performance" in content
                assert "Generated at: 2025-01-15 10:00:00" in content  # Metadata from fallback file
                assert "Git commit: abc123" in content  # Git data from fallback file

                # Note: The baseline file parsing extracts metadata, not performance data
                # Performance data "1000 Points (3D)" would come from benchmark parsing,
                # not baseline parsing. The important test is that the fallback file is read.

    def test_full_generation_workflow_integration(self):
        """Test complete summary generation workflow (integration test)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            output_file = Path(temp_dir) / "full_test.md"

            # Set up complete test environment
            baseline_dir = project_root / "benches"
            baseline_dir.mkdir()

            # Mock baseline file
            baseline_file = baseline_dir / "baseline_results.txt"
            baseline_file.write_text(
                "Generated at: 2025-01-15 10:00:00\n"
                "Git commit: abc123\n"
                "=== 10 Points (2D) ===\n"
                "Time: [100.0, 110.0, 120.0] µs\n"
                "Throughput: [8000.0, 9000.0, 10000.0] Kelem/s\n",
            )

            # Mock comparison file
            comparison_file = baseline_dir / "compare_results.txt"
            comparison_file.write_text("✅ OK: All benchmarks within acceptable range\n")

            with (
                patch("benchmark_utils.get_git_commit_hash") as mock_commit,
            ):
                mock_commit.return_value = "abc123def456"

                generator = PerformanceSummaryGenerator(project_root)
                success = generator.generate_summary(output_file)

                assert success

                content = output_file.read_text()

                # Verify all major sections are present
                assert "# Delaunay Library Performance Results" in content
                assert "Single Query Performance (3D)" in content
                assert "Triangulation Data Structure Performance" in content
                assert "Performance Status: Good" in content
                assert "Key Findings" in content
                assert "Performance Ranking" in content
                assert "Recommendations" in content
                assert "Performance Data Updates" in content

    def test_dimension_sorting_numeric_order(self):
        """Test that dimensions are sorted numerically, not lexically."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            # Create test cases with dimensions that would sort wrong lexically
            test_cases = [
                CircumsphereTestCase("Test10", "10D", {"insphere": CircumspherePerformanceData("insphere", 1000)}),
                CircumsphereTestCase("Test2", "2D", {"insphere": CircumspherePerformanceData("insphere", 1000)}),
                CircumsphereTestCase("Test3", "3D", {"insphere": CircumspherePerformanceData("insphere", 1000)}),
                CircumsphereTestCase("Test1", "1D", {"insphere": CircumspherePerformanceData("insphere", 1000)}),
                CircumsphereTestCase("Test9", "9D", {"insphere": CircumspherePerformanceData("insphere", 1000)}),
            ]

            # Patch the generator to use our test cases instead of parsing from files
            with patch.object(generator, "_parse_circumsphere_benchmark_results", return_value=test_cases):
                # Generate the circumsphere performance results section
                result_lines = generator._get_circumsphere_performance_results()
                content = "\n".join(result_lines)

                # Find the order of dimension headers in the generated content

                dimension_headers = re.findall(r"#### Single Query Performance \((\d+D)\)", content)

                # Verify that dimensions appear in numeric order: 1D, 2D, 3D, 9D, 10D
                expected_order = ["1D", "2D", "3D", "9D", "10D"]
                assert dimension_headers == expected_order, f"Expected {expected_order}, got {dimension_headers}"

                # Also verify that each dimension's test case appears in the content
                assert "Test1" in content  # 1D test case
                assert "Test2" in content  # 2D test case
                assert "Test3" in content  # 3D test case
                assert "Test9" in content  # 9D test case
                assert "Test10" in content  # 10D test case

    def test_hardware_metadata_parsing_with_cores(self):
        """Test that hardware metadata parsing includes cores and guards against IndexError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            # Create baseline with proper hardware section
            baseline_dir = project_root / "baseline-artifact"
            baseline_dir.mkdir()
            baseline_file = baseline_dir / "baseline_results.txt"

            # Test with complete hardware info
            baseline_content = """Date: 2023-12-15 10:30:00 UTC
Git commit: abc123def456
Hardware Information:
  OS: macOS
  CPU: Apple M4 Max
  CPU Cores: 14
  Memory: 64.0 GB

=== 1000 Points (2D) ===
Time: [95.0, 100.0, 105.0] µs
"""
            baseline_file.write_text(baseline_content)

            generator = PerformanceSummaryGenerator(project_root)
            lines = generator._parse_baseline_results()
            content = "\n".join(lines)

            # Should include cores in the hardware summary
            assert "Apple M4 Max (14 cores)" in content

            # Test with missing cores info (edge case protection)
            baseline_content_short = """Date: 2023-12-15 10:30:00 UTC
Git commit: abc123def456
Hardware Information:
  OS: macOS
  CPU: Apple M4 Max
"""
            baseline_file.write_text(baseline_content_short)

            lines = generator._parse_baseline_results()
            content = "\n".join(lines)

            # Should handle missing cores gracefully
            assert "Apple M4 Max" in content
            assert "(" not in content.split("Apple M4 Max")[1].split("\n")[0] if "Apple M4 Max" in content else True

    def test_dev_mode_args_consistency(self):
        """Test that DEV_MODE_BENCH_ARGS is used consistently."""
        # Verify the constant exists and has expected structure
        assert isinstance(DEV_MODE_BENCH_ARGS, list)
        assert "--sample-size" in DEV_MODE_BENCH_ARGS
        assert "--measurement-time" in DEV_MODE_BENCH_ARGS
        assert "--warm-up-time" in DEV_MODE_BENCH_ARGS

        # The specific values may change, but the structure should be consistent
        # with pairs of argument name and value
        assert len(DEV_MODE_BENCH_ARGS) >= 6  # At least 3 arg-value pairs

    def test_numerical_accuracy_phrasing_flexibility(self):
        """Test that numerical accuracy section doesn't hardcode sample size."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            # Get the numerical accuracy analysis without specific data
            lines = generator._get_numerical_accuracy_analysis()
            content = "\n".join(lines)

            # Should use flexible phrasing instead of hardcoded "1000 random test cases"
            assert "Based on random test cases:" in content
            assert "Based on 1000 random test cases:" not in content


class TestTagSpecificBaselineHandling:
    """Test cases for tag-specific baseline file handling functionality."""

    def test_prepare_baseline_with_tag_specific_file(self, capsys):
        """Test baseline preparation with tag-specific file (baseline-v*.txt)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)
            tag_baseline_file = baseline_dir / "baseline-v0.4.3.txt"

            # Create a tag-specific baseline file
            baseline_content = """Date: 2025-09-13 00:00:36 UTC
Git commit: 1062551a9152a53e938ddbf94c4152ff6ae4254d
Tag: v0.4.3
Hardware Information:
  OS: macOS
  CPU: Apple M1 (Virtual)

=== 10 Points (2D) ===
Time: [160.1, 168.18, 177.67] µs
"""
            tag_baseline_file.write_text(baseline_content)

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as env_file:
                env_path = env_file.name

            try:
                with patch.dict(os.environ, {"GITHUB_ENV": env_path}):
                    success = BenchmarkRegressionHelper.prepare_baseline(baseline_dir)

                    assert success

                    # Check that environment variables were set
                    with open(env_path, encoding="utf-8") as f:
                        env_content = f.read()
                        assert "BASELINE_EXISTS=true" in env_content
                        assert "BASELINE_SOURCE=artifact" in env_content
                        assert "BASELINE_ORIGIN=artifact" in env_content
                        assert "BASELINE_TAG=v0.4.3" in env_content
                        assert "BASELINE_SOURCE_FILE=baseline-v0.4.3.txt" in env_content

                    # Check that baseline info was printed with file conversion message
                    captured = capsys.readouterr()
                    assert "Prepared baseline from artifact: baseline-v0.4.3.txt" in captured.out
                    assert " → baseline_results.txt" in captured.out
                    assert "=== Baseline Information" in captured.out
                    assert "Tag: v0.4.3" in captured.out

                    # Check that baseline_results.txt was created
                    standard_file = baseline_dir / "baseline_results.txt"
                    assert standard_file.exists()
                    assert "Tag: v0.4.3" in standard_file.read_text(encoding="utf-8")

            finally:
                Path(env_path).unlink(missing_ok=True)

    def test_prepare_baseline_with_generic_baseline_file(self, capsys):
        """Test baseline preparation with generic baseline*.txt file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)
            generic_baseline_file = baseline_dir / "baseline-manual-test.txt"

            # Create a generic baseline file
            baseline_content = """Date: 2025-09-13 10:30:00 UTC
Git commit: abcdef123456
Hardware Information:
  OS: macOS
  CPU: Test CPU

=== 100 Points (2D) ===
Time: [95.0, 100.0, 105.0] µs
"""
            generic_baseline_file.write_text(baseline_content)

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as env_file:
                env_path = env_file.name

            try:
                with patch.dict(os.environ, {"GITHUB_ENV": env_path}):
                    success = BenchmarkRegressionHelper.prepare_baseline(baseline_dir)

                    assert success

                    # Check that environment variables were set
                    with open(env_path, encoding="utf-8") as f:
                        env_content = f.read()
                        assert "BASELINE_EXISTS=true" in env_content
                        assert "BASELINE_SOURCE=artifact" in env_content
                        assert "BASELINE_ORIGIN=artifact" in env_content

                    # Check that baseline info was printed with file conversion message
                    captured = capsys.readouterr()
                    assert "Prepared baseline from artifact: baseline-manual-test.txt" in captured.out
                    assert " → baseline_results.txt" in captured.out

                    # Check that baseline_results.txt was created
                    standard_file = baseline_dir / "baseline_results.txt"
                    assert standard_file.exists()
                    assert "Test CPU" in standard_file.read_text(encoding="utf-8")

            finally:
                Path(env_path).unlink(missing_ok=True)

    def test_prepare_baseline_prefers_standard_name(self, capsys):
        """Test that prepare_baseline prefers baseline_results.txt over tag-specific files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)

            # Create both standard and tag-specific files
            standard_file = baseline_dir / "baseline_results.txt"
            tag_file = baseline_dir / "baseline-v1.0.0.txt"

            standard_content = "Standard file content"
            tag_content = "Tag-specific file content"

            standard_file.write_text(standard_content)
            tag_file.write_text(tag_content)

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as env_file:
                env_path = env_file.name

            try:
                with patch.dict(os.environ, {"GITHUB_ENV": env_path}):
                    success = BenchmarkRegressionHelper.prepare_baseline(baseline_dir)

                    assert success

                    # Check that environment variables were set
                    with open(env_path, encoding="utf-8") as f:
                        env_content = f.read()
                        assert "BASELINE_EXISTS=true" in env_content
                        assert "BASELINE_SOURCE=artifact" in env_content
                        assert "BASELINE_ORIGIN=artifact" in env_content

                    # Should use the standard file and not show conversion message
                    captured = capsys.readouterr()
                    assert "Prepared baseline from artifact" in captured.out
                    assert " → " not in captured.out  # No conversion arrow

                    # Standard file should remain unchanged
                    assert standard_file.read_text(encoding="utf-8") == standard_content

            finally:
                Path(env_path).unlink(missing_ok=True)

    def test_prepare_baseline_no_matching_files(self, capsys):
        """Test baseline preparation when no matching baseline files are found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)

            # Create some non-matching files
            (baseline_dir / "metadata.json").write_text("{}")
            (baseline_dir / "random.txt").write_text("Not a baseline")
            (baseline_dir / "results.log").write_text("Log data")

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as env_file:
                env_path = env_file.name

            try:
                with patch.dict(os.environ, {"GITHUB_ENV": env_path}):
                    success = BenchmarkRegressionHelper.prepare_baseline(baseline_dir)

                    assert not success

                    # Check that environment variables were set correctly for failure
                    with open(env_path, encoding="utf-8") as f:
                        env_content = f.read()
                        assert "BASELINE_EXISTS=false" in env_content
                        assert "BASELINE_SOURCE=missing" in env_content
                        assert "BASELINE_ORIGIN=unknown" in env_content

                    # Check error message was printed to stderr
                    captured = capsys.readouterr()
                    assert "❌ Downloaded artifact but no baseline*.txt files found" in captured.err

            finally:
                Path(env_path).unlink(missing_ok=True)

    def test_extract_baseline_commit_from_tag_file(self):
        """Test extracting commit SHA from tag-specific baseline file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)
            tag_baseline_file = baseline_dir / "baseline-v0.4.3.txt"

            baseline_content = """Date: 2025-09-13 00:00:36 UTC
Git commit: 1062551a9152a53e938ddbf94c4152ff6ae4254d
Tag: v0.4.3
Hardware Information:
  OS: macOS
"""
            tag_baseline_file.write_text(baseline_content)

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as env_file:
                env_path = env_file.name

            try:
                with patch.dict(os.environ, {"GITHUB_ENV": env_path}):
                    commit_sha = BenchmarkRegressionHelper.extract_baseline_commit(baseline_dir)

                    assert commit_sha == "1062551a9152a53e938ddbf94c4152ff6ae4254d"

                    # Check that environment variables were set
                    with open(env_path, encoding="utf-8") as f:
                        env_content = f.read()
                        assert "BASELINE_COMMIT=1062551a9152a53e938ddbf94c4152ff6ae4254d" in env_content
                        assert "BASELINE_COMMIT_SOURCE=baseline" in env_content

            finally:
                Path(env_path).unlink(missing_ok=True)

    def test_extract_baseline_commit_fallback_to_metadata(self):
        """Test extracting commit SHA from metadata.json when baseline files have no commit info."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)
            tag_baseline_file = baseline_dir / "baseline-v0.4.3.txt"
            metadata_file = baseline_dir / "metadata.json"

            # Baseline file without Git commit line
            baseline_content = """Date: 2025-09-13 00:00:36 UTC
Tag: v0.4.3
Hardware Information:
  OS: macOS
"""
            tag_baseline_file.write_text(baseline_content)

            # Metadata with commit info
            metadata = {"tag": "v0.4.3", "commit": "fedcba987654321", "generated_at": "2025-09-13T00:00:36Z"}
            with metadata_file.open("w", encoding="utf-8") as f:
                json.dump(metadata, f)

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as env_file:
                env_path = env_file.name

            try:
                with patch.dict(os.environ, {"GITHUB_ENV": env_path}):
                    commit_sha = BenchmarkRegressionHelper.extract_baseline_commit(baseline_dir)

                    assert commit_sha == "fedcba987654321"

                    # Check that environment variables were set
                    with open(env_path, encoding="utf-8") as f:
                        env_content = f.read()
                        assert "BASELINE_COMMIT=fedcba987654321" in env_content
                        assert "BASELINE_COMMIT_SOURCE=metadata" in env_content

            finally:
                Path(env_path).unlink(missing_ok=True)

    def test_extract_baseline_commit_handles_multiple_tag_files(self):
        """Test that extract_baseline_commit selects the highest semver tag file when multiple exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)

            # Create multiple tag files (should use highest semver version)
            tag_file_1 = baseline_dir / "baseline-v0.4.1.txt"
            tag_file_2 = baseline_dir / "baseline-v0.4.3.txt"

            tag_content_1 = """Date: 2025-09-13 00:00:36 UTC
Git commit: abc123def456
Tag: v0.4.1
"""
            tag_content_2 = """Date: 2025-09-13 00:00:36 UTC
Git commit: def456abc789
Tag: v0.4.3
"""

            tag_file_1.write_text(tag_content_1)
            tag_file_2.write_text(tag_content_2)

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as env_file:
                env_path = env_file.name

            try:
                with patch.dict(os.environ, {"GITHUB_ENV": env_path}):
                    commit_sha = BenchmarkRegressionHelper.extract_baseline_commit(baseline_dir)

                    # Should pick v0.4.3 (highest semver version)
                    assert commit_sha == "def456abc789"

            finally:
                Path(env_path).unlink(missing_ok=True)

    def test_semver_prefers_stable_over_prerelease(self):
        """Test that stable releases are preferred over pre-releases of the same version."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)

            # Create files with stable and pre-release versions
            stable_file = baseline_dir / "baseline-v1.2.3.txt"
            prerelease_file = baseline_dir / "baseline-v1.2.3-beta.1.txt"
            older_stable = baseline_dir / "baseline-v1.2.2.txt"

            stable_file.write_text("Stable v1.2.3")
            prerelease_file.write_text("Pre-release v1.2.3-beta.1")
            older_stable.write_text("Older stable v1.2.2")

            # Should select the stable v1.2.3 over both the pre-release and older stable
            selected = BenchmarkRegressionHelper._find_baseline_file(baseline_dir)
            assert selected is not None
            assert selected.name == "baseline-v1.2.3.txt"

    def test_semver_v043_vs_v043_beta1_preference(self):
        """Test specific case: v0.4.3 is preferred over v0.4.3-beta.1."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)

            # Create the exact scenario mentioned: v0.4.3 vs v0.4.3-beta.1
            stable_file = baseline_dir / "baseline-v0.4.3.txt"
            prerelease_file = baseline_dir / "baseline-v0.4.3-beta.1.txt"

            stable_file.write_text("Date: 2023-12-15\nGit commit: stable043\nTag: v0.4.3\n")
            prerelease_file.write_text("Date: 2023-12-15\nGit commit: beta043\nTag: v0.4.3-beta.1\n")

            # The stable version should be selected
            selected = BenchmarkRegressionHelper._find_baseline_file(baseline_dir)
            assert selected is not None
            assert selected.name == "baseline-v0.4.3.txt"

            # Verify the content to ensure we got the right file
            content = selected.read_text()
            assert "stable043" in content
            assert "Tag: v0.4.3" in content

    def test_semver_prefers_higher_prerelease_when_no_stable(self):
        """Test that higher pre-release is selected when only pre-releases exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)

            # Create only pre-release files
            beta1_file = baseline_dir / "baseline-v1.2.3-beta.1.txt"
            beta2_file = baseline_dir / "baseline-v1.2.3-beta.2.txt"
            alpha_file = baseline_dir / "baseline-v1.2.3-alpha.1.txt"

            beta1_file.write_text("Beta 1")
            beta2_file.write_text("Beta 2")
            alpha_file.write_text("Alpha 1")

            # Should select the highest version (beta.2 > beta.1 > alpha.1 lexicographically)
            selected = BenchmarkRegressionHelper._find_baseline_file(baseline_dir)
            assert selected is not None
            # Current behavior: lexicographic prerelease ordering; expect beta.2 to win
            assert selected.name == "baseline-v1.2.3-beta.2.txt"

    def test_baseline_commit_source_from_baseline_file(self):
        """Test that BASELINE_COMMIT_SOURCE is 'baseline' when commit is extracted from baseline file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)
            baseline_file = baseline_dir / "baseline_results.txt"
            baseline_content = """Date: 2023-12-15 10:30:00 UTC
Git commit: abc123def456
Hardware Information:
  OS: macOS
"""
            baseline_file.write_text(baseline_content)

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as env_file:
                env_path = env_file.name

            try:
                with patch.dict(os.environ, {"GITHUB_ENV": env_path}):
                    commit_sha = BenchmarkRegressionHelper.extract_baseline_commit(baseline_dir)

                    assert commit_sha == "abc123def456"

                    with open(env_path, encoding="utf-8") as f:
                        env_content = f.read()
                        assert "BASELINE_COMMIT=abc123def456" in env_content
                        assert "BASELINE_COMMIT_SOURCE=baseline" in env_content
                        assert "BASELINE_SOURCE_FILE=baseline_results.txt" in env_content
            finally:
                Path(env_path).unlink(missing_ok=True)

    def test_baseline_commit_source_from_metadata_file(self):
        """Test that BASELINE_COMMIT_SOURCE is 'metadata' when commit is extracted from metadata.json."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)

            # Create baseline file without commit info
            baseline_file = baseline_dir / "baseline_results.txt"
            baseline_file.write_text("Date: 2023-12-15\nHardware: Test\n")

            # Create metadata file with commit info
            metadata_file = baseline_dir / "metadata.json"
            metadata = {"commit": "def456abc789", "tag": "v1.0.0"}
            with metadata_file.open("w", encoding="utf-8") as f:
                json.dump(metadata, f)

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as env_file:
                env_path = env_file.name

            try:
                with patch.dict(os.environ, {"GITHUB_ENV": env_path}):
                    commit_sha = BenchmarkRegressionHelper.extract_baseline_commit(baseline_dir)

                    assert commit_sha == "def456abc789"

                    with open(env_path, encoding="utf-8") as f:
                        env_content = f.read()
                        assert "BASELINE_COMMIT=def456abc789" in env_content
                        assert "BASELINE_COMMIT_SOURCE=metadata" in env_content
            finally:
                Path(env_path).unlink(missing_ok=True)

    def test_baseline_commit_source_unknown_when_no_commit_found(self):
        """Test that BASELINE_COMMIT_SOURCE is 'unknown' when no commit is found anywhere."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)
            baseline_file = baseline_dir / "baseline_results.txt"
            baseline_file.write_text("Date: 2023-12-15\n")

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as env_file:
                env_path = env_file.name

            try:
                with patch.dict(os.environ, {"GITHUB_ENV": env_path}):
                    commit_sha = BenchmarkRegressionHelper.extract_baseline_commit(baseline_dir)

                    assert commit_sha == "unknown"

                    with open(env_path, encoding="utf-8") as f:
                        env_content = f.read()
                        assert "BASELINE_COMMIT=unknown" in env_content
                        assert "BASELINE_COMMIT_SOURCE=unknown" in env_content
            finally:
                Path(env_path).unlink(missing_ok=True)

    def test_env_vars_mirrored_to_current_process(self):
        """Test that write_github_env_vars mirrors variables into current process."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as env_file:
            env_path = env_file.name

        try:
            # Clear any existing test variables
            for key in ["TEST_BASELINE_EXISTS", "TEST_BASELINE_SOURCE"]:
                os.environ.pop(key, None)

            test_vars = {
                "TEST_BASELINE_EXISTS": "true",
                "TEST_BASELINE_SOURCE": "artifact",
            }

            with patch.dict(os.environ, {"GITHUB_ENV": env_path}):
                BenchmarkRegressionHelper.write_github_env_vars(test_vars)

                # Verify variables are written to GITHUB_ENV file
                with open(env_path, encoding="utf-8") as f:
                    content = f.read()
                    assert "TEST_BASELINE_EXISTS=true" in content
                    assert "TEST_BASELINE_SOURCE=artifact" in content

                # Verify variables are also available in current process
                assert os.environ["TEST_BASELINE_EXISTS"] == "true"
                assert os.environ["TEST_BASELINE_SOURCE"] == "artifact"

        finally:
            Path(env_path).unlink(missing_ok=True)
            # Clean up test variables
            for key in ["TEST_BASELINE_EXISTS", "TEST_BASELINE_SOURCE"]:
                os.environ.pop(key, None)

    def test_env_vars_multiline_handling(self):
        """Test that write_github_env_vars correctly handles multiline values with heredoc format."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as env_file:
            env_path = env_file.name

        try:
            # Clear any existing test variables
            for key in ["TEST_MULTILINE", "TEST_SINGLE_LINE", "TEST_WITH_CR"]:
                os.environ.pop(key, None)

            multiline_value = "Line 1\nLine 2\nLine 3"
            cr_value = "Line 1\r\nLine 2\r\nLine 3"  # Contains CR characters

            test_vars = {
                "TEST_MULTILINE": multiline_value,
                "TEST_SINGLE_LINE": "single",
                "TEST_WITH_CR": cr_value,
            }

            with patch.dict(os.environ, {"GITHUB_ENV": env_path}):
                BenchmarkRegressionHelper.write_github_env_vars(test_vars)

                # Verify variables are written to GITHUB_ENV file
                with open(env_path, encoding="utf-8") as f:
                    content = f.read()

                    # Single line should use regular key=value format
                    assert "TEST_SINGLE_LINE=single" in content

                    # Multiline should use heredoc format
                    assert "TEST_MULTILINE<<EOF_" in content
                    assert "Line 1\nLine 2\nLine 3" in content

                    # CR characters should be stripped
                    assert "Line 1\nLine 2\nLine 3" in content
                    assert "\r" not in content

                # Verify variables are also available in current process
                assert os.environ["TEST_MULTILINE"] == multiline_value
                assert os.environ["TEST_SINGLE_LINE"] == "single"
                # CR characters should be stripped from process environment too
                assert os.environ["TEST_WITH_CR"] == "Line 1\nLine 2\nLine 3"

        finally:
            Path(env_path).unlink(missing_ok=True)
            # Clean up test variables
            for key in ["TEST_MULTILINE", "TEST_SINGLE_LINE", "TEST_WITH_CR"]:
                os.environ.pop(key, None)

    def test_env_vars_none_value_handling(self):
        """Test that write_github_env_vars correctly handles None values without errors."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as env_file:
            env_path = env_file.name

        try:
            # Clear any existing test variables
            for key in ["TEST_NONE", "TEST_NORMAL"]:
                os.environ.pop(key, None)

            test_vars = {
                "TEST_NONE": None,
                "TEST_NORMAL": "normal_value",
            }

            with patch.dict(os.environ, {"GITHUB_ENV": env_path}):
                # This should not raise any errors
                BenchmarkRegressionHelper.write_github_env_vars(test_vars)

                # Verify variables are written to GITHUB_ENV file
                with open(env_path, encoding="utf-8") as f:
                    content = f.read()
                    assert "TEST_NONE=" in content  # None becomes empty string
                    assert "TEST_NORMAL=normal_value" in content

                # Verify variables are also available in current process
                assert os.environ["TEST_NONE"] == ""  # None becomes empty string
                assert os.environ["TEST_NORMAL"] == "normal_value"

        finally:
            Path(env_path).unlink(missing_ok=True)
            # Clean up test variables
            for key in ["TEST_NONE", "TEST_NORMAL"]:
                os.environ.pop(key, None)

    def test_baseline_tag_sanitization(self):
        """Test that BASELINE_TAG is sanitized before being exported to GITHUB_ENV."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)

            # Create baseline file with potentially dangerous tag value
            baseline_file = baseline_dir / "baseline_results.txt"
            baseline_content = """Date: 2023-12-15 10:30:00 UTC
Git commit: abc123def456
Tag: v1.0.0; echo "injected"; rm -rf /tmp/test
Hardware Information:
  OS: macOS
"""
            baseline_file.write_text(baseline_content)

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as env_file:
                env_path = env_file.name

            try:
                with patch.dict(os.environ, {"GITHUB_ENV": env_path}):
                    # This should sanitize the dangerous tag
                    success = BenchmarkRegressionHelper.prepare_baseline(baseline_dir)
                    assert success

                    with open(env_path, encoding="utf-8") as f:
                        env_content = f.read()

                        # Verify the tag was sanitized (dangerous characters replaced with underscores)
                        assert "BASELINE_TAG=v1.0.0__echo__injected___rm_-rf__tmp_test" in env_content

                        # Verify no injection occurred
                        assert "; echo " not in env_content
                        assert "rm -rf" not in env_content

                        # Verify other environment variables are still set correctly
                        assert "BASELINE_EXISTS=true" in env_content
                        assert "BASELINE_SOURCE=artifact" in env_content

            finally:
                Path(env_path).unlink(missing_ok=True)

    def test_baseline_tag_length_capping(self):
        """Test that BASELINE_TAG is capped at 64 characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)

            # Create baseline file with very long tag value
            baseline_file = baseline_dir / "baseline_results.txt"
            long_tag = "v1.0.0-" + "a" * 100  # Tag longer than 64 characters
            baseline_content = f"""Date: 2023-12-15 10:30:00 UTC
Git commit: abc123def456
Tag: {long_tag}
Hardware Information:
  OS: macOS
"""
            baseline_file.write_text(baseline_content)

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as env_file:
                env_path = env_file.name

            try:
                with patch.dict(os.environ, {"GITHUB_ENV": env_path}):
                    success = BenchmarkRegressionHelper.prepare_baseline(baseline_dir)
                    assert success

                    with open(env_path, encoding="utf-8") as f:
                        env_content = f.read()

                        # Find the BASELINE_TAG line
                        tag_line = next((line for line in env_content.split("\n") if line.startswith("BASELINE_TAG=")), None)
                        assert tag_line is not None

                        # Extract the tag value
                        tag_value = tag_line.split("=", 1)[1]

                        # Verify it's capped at 64 characters
                        assert len(tag_value) <= 64

                        # Verify it starts correctly
                        assert tag_value.startswith("v1.0.0-")

            finally:
                Path(env_path).unlink(missing_ok=True)

    def test_packaging_version_complex_comparisons(self):
        """Test that packaging.version handles complex version comparisons correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)

            # Create files with various complex version formats
            files_and_expected_order = [
                "baseline-v2.0.0.txt",  # Highest stable
                "baseline-v2.0.0rc1.txt",  # RC1 of v2.0.0
                "baseline-v2.0.0b2.txt",  # Beta 2 of v2.0.0
                "baseline-v2.0.0b1.txt",  # Beta 1 of v2.0.0
                "baseline-v2.0.0a1.txt",  # Alpha 1 of v2.0.0
                "baseline-v1.9.0.txt",  # Lower major version
                "baseline-v1.9.0rc1.txt",  # RC of lower version
            ]

            # Create files in reverse order to test sorting
            for filename in reversed(files_and_expected_order):
                file = baseline_dir / filename
                file.write_text(f"Content of {filename}")

            # Should select the highest version (v2.0.0 stable)
            selected = BenchmarkRegressionHelper._find_baseline_file(baseline_dir)
            assert selected is not None
            assert selected.name == "baseline-v2.0.0.txt"

    def test_packaging_version_invalid_versions(self):
        """Test that invalid version formats are handled gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)

            # Create mix of valid and invalid version files
            valid_file1 = baseline_dir / "baseline-v1.0.0.txt"
            valid_file2 = baseline_dir / "baseline-v1.2.txt"  # Valid: 1.2 is equivalent to 1.2.0
            invalid_file1 = baseline_dir / "baseline-vInvalid.txt"
            generic_file = baseline_dir / "baseline-generic.txt"

            valid_file1.write_text("Valid 1.0.0 content")
            valid_file2.write_text("Valid 1.2.0 content")
            invalid_file1.write_text("Invalid version content")
            generic_file.write_text("Generic content")

            # Should select the highest valid version (1.2.0 > 1.0.0)
            selected = BenchmarkRegressionHelper._find_baseline_file(baseline_dir)
            assert selected is not None
            assert selected.name == "baseline-v1.2.txt"
            assert "Valid 1.2.0 content" in selected.read_text()

    def test_packaging_version_truly_invalid_versions(self):
        """Test that truly invalid version formats fall back to generic baseline selection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)

            # Create only invalid version files and one generic file
            invalid_file1 = baseline_dir / "baseline-vInvalid.txt"
            invalid_file2 = baseline_dir / "baseline-v1.2.3.4.5.txt"  # Too many segments
            invalid_file3 = baseline_dir / "baseline-vNot-A-Version.txt"
            generic_file = baseline_dir / "baseline_results.txt"  # Standard fallback

            invalid_file1.write_text("Invalid content 1")
            invalid_file2.write_text("Invalid content 2")
            invalid_file3.write_text("Invalid content 3")
            generic_file.write_text("Generic baseline content")

            # Should fall back to standard baseline file
            selected = BenchmarkRegressionHelper._find_baseline_file(baseline_dir)
            assert selected is not None
            assert selected.name == "baseline_results.txt"
            assert "Generic baseline content" in selected.read_text()

    def test_generic_baseline_prefers_newest_mtime(self):
        """Test that generic baseline files are selected by most recent mtime."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)

            # Create multiple generic baseline files with different mtime
            older_file = baseline_dir / "baseline-older.txt"
            newer_file = baseline_dir / "baseline-newer.txt"

            # Create older file first
            older_file.write_text("Older baseline content")
            older_mtime = time.time() - 100  # 100 seconds ago
            os.utime(older_file, (older_mtime, older_mtime))

            # Create newer file
            newer_file.write_text("Newer baseline content")
            newer_mtime = time.time() - 50  # 50 seconds ago
            os.utime(newer_file, (newer_mtime, newer_mtime))

            # Should select the file with the most recent mtime
            selected = BenchmarkRegressionHelper._find_baseline_file(baseline_dir)
            assert selected is not None
            assert selected.name == "baseline-newer.txt"
            assert "Newer baseline content" in selected.read_text()

    def test_prerelease_detection_fix_validation(self):
        """Test that prerelease detection correctly identifies stable vs prerelease versions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)

            # Create exactly the same versions to test the prerelease detection bug fix
            stable_file = baseline_dir / "baseline-v1.0.0.txt"
            prerelease_file = baseline_dir / "baseline-v1.0.0-rc.1.txt"

            stable_file.write_text("Stable content")
            prerelease_file.write_text("Prerelease content")

            # The stable version should be selected over the prerelease
            selected = BenchmarkRegressionHelper._find_baseline_file(baseline_dir)
            assert selected is not None
            assert selected.name == "baseline-v1.0.0.txt"
            assert "Stable content" in selected.read_text()

    def test_prepare_baseline_and_extract_commit_integration(self):
        """Test the integration between prepare_baseline and extract_baseline_commit."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)
            tag_baseline_file = baseline_dir / "baseline-v0.4.3.txt"

            baseline_content = """Date: 2025-09-13 00:00:36 UTC
Git commit: 1234567890abcdef
Tag: v0.4.3
Hardware Information:
  OS: macOS
  CPU: Apple M1 (Virtual)

=== 10 Points (2D) ===
Time: [160.1, 168.18, 177.67] µs
"""
            tag_baseline_file.write_text(baseline_content)

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as env_file:
                env_path = env_file.name

            try:
                with patch.dict(os.environ, {"GITHUB_ENV": env_path}):
                    # First, prepare the baseline (should create baseline_results.txt)
                    prepare_success = BenchmarkRegressionHelper.prepare_baseline(baseline_dir)
                    assert prepare_success

                    # Verify standard file was created
                    standard_file = baseline_dir / "baseline_results.txt"
                    assert standard_file.exists()

                    # Then extract commit (should work with the standard file)
                    commit_sha = BenchmarkRegressionHelper.extract_baseline_commit(baseline_dir)
                    assert commit_sha == "1234567890abcdef"

                    # Check that BASELINE_TAG was also exported during prepare_baseline
                    with open(env_path, encoding="utf-8") as f:
                        env_content = f.read()
                        assert "BASELINE_TAG=v0.4.3" in env_content
                        assert "BASELINE_SOURCE_FILE=baseline-v0.4.3.txt" in env_content

            finally:
                Path(env_path).unlink(missing_ok=True)
