#!/usr/bin/env python3
# ruff: noqa: SLF001
"""
Test suite for benchmark_utils.py module.

Tests benchmark parsing, baseline generation, and performance comparison functionality,
with special focus on the new average regression calculation logic.

Note: This test file accesses private methods (prefixed with _) which is expected
and necessary for comprehensive unit testing of internal functionality.
"""

import json
import os
import subprocess
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from benchmark_utils import (
    BenchmarkData,
    BenchmarkRegressionHelper,
    CriterionParser,
    PerformanceComparator,
    WorkflowHelper,
)


class TestBenchmarkData:
    """Test cases for BenchmarkData class."""

    def test_init(self):
        """Test BenchmarkData initialization."""
        data = BenchmarkData(points=1000, dimension="2D")
        assert data.points == 1000
        assert data.dimension == "2D"
        assert data.time_mean == 0.0
        assert data.throughput_mean is None

    def test_with_timing_fluent_interface(self):
        """Test fluent interface for setting timing data."""
        data = BenchmarkData(1000, "3D").with_timing(100.0, 110.0, 120.0, "µs")

        assert data.time_low == 100.0
        assert data.time_mean == 110.0
        assert data.time_high == 120.0
        assert data.time_unit == "µs"

    def test_with_throughput_fluent_interface(self):
        """Test fluent interface for setting throughput data."""
        data = BenchmarkData(1000, "2D").with_throughput(800.0, 900.0, 1000.0, "Kelem/s")

        assert data.throughput_low == 800.0
        assert data.throughput_mean == 900.0
        assert data.throughput_high == 1000.0
        assert data.throughput_unit == "Kelem/s"

    def test_to_baseline_format_with_timing_only(self):
        """Test baseline format output with timing data only."""
        data = BenchmarkData(1000, "2D").with_timing(100.0, 110.0, 120.0, "µs")

        result = data.to_baseline_format()
        expected = """=== 1000 Points (2D) ===
Time: [100.0, 110.0, 120.0] µs
"""
        assert result == expected

    def test_to_baseline_format_with_timing_and_throughput(self):
        """Test baseline format output with both timing and throughput data."""
        data = BenchmarkData(1000, "3D").with_timing(100.0, 110.0, 120.0, "µs").with_throughput(800.0, 900.0, 1000.0, "Kelem/s")

        result = data.to_baseline_format()
        expected = """=== 1000 Points (3D) ===
Time: [100.0, 110.0, 120.0] µs
Throughput: [800.0, 900.0, 1000.0] Kelem/s
"""
        assert result == expected


class TestCriterionParser:
    """Test cases for CriterionParser class."""

    def test_parse_estimates_json_valid_data(self):
        """Test parsing valid estimates.json data."""
        estimates_data = {
            "mean": {
                "point_estimate": 110000.0,  # 110 microseconds in nanoseconds
                "confidence_interval": {"lower_bound": 100000.0, "upper_bound": 120000.0},
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(estimates_data, f)
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
            assert abs(result.throughput_mean - 9090.909) < 0.001  # 1000 * 1000 / 110
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
        assert abs(time_change - 4.76) < 0.01
        assert not is_regression  # Less than 5% threshold

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
        assert is_regression  # Greater than 5% threshold

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
        # This is less than 5% threshold, so no overall regression
        assert not regression_found

        result = output.getvalue()
        assert "SUMMARY" in result
        assert "Total benchmarks compared: 3" in result
        assert "Individual regressions (>5.0%): 1" in result  # Only the +20% one
        assert "Average time change: -0.0%" in result
        assert "✅ OVERALL OK" in result

    def test_write_performance_comparison_with_average_regression(self, comparator):
        """Test performance comparison with average regression exceeding threshold."""
        # Create current results with overall performance degradation
        current_results = [
            # Regression: +10%
            BenchmarkData(1000, "2D").with_timing(105.0, 110.0, 115.0, "µs"),
            # Regression: +8%
            BenchmarkData(2000, "2D").with_timing(205.0, 216.0, 227.0, "µs"),
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

        # Average change using geometric mean: 5.6%
        # This exceeds 5% threshold, so overall regression found
        assert regression_found

        result = output.getvalue()
        assert "SUMMARY" in result
        assert "Total benchmarks compared: 3" in result
        assert "Individual regressions (>5.0%): 2" in result  # The +10% and +8% ones
        assert "Average time change: 5.6%" in result
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
        assert "Individual regressions (>5.0%): 0" in result
        assert "Average time change: -5.5%" in result
        assert "🎉 OVERALL IMPROVEMENT" in result

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
        assert comparator.regression_threshold == 5.0

        # Test changing threshold
        comparator.regression_threshold = 10.0

        current = BenchmarkData(1000, "2D").with_timing(100.0, 107.0, 114.0, "µs")
        baseline = BenchmarkData(1000, "2D").with_timing(95.0, 100.0, 105.0, "µs")

        output = StringIO()
        time_change, is_regression = comparator._write_time_comparison(output, current, baseline)

        # 7% change should not be regression with 10% threshold
        assert abs(time_change - 7.0) < 0.001  # Use tolerance for floating-point comparison
        assert not is_regression


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
            # Medium regression in 2D medium dataset: +7%
            BenchmarkData(5000, "2D").with_timing(428.0, 535.0, 642.0, "µs"),
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
        assert "Individual regressions (>5.0%): 1" in result  # Only the 7% one
        assert "Average time change: -0.2%" in result
        assert "✅ OVERALL OK" in result

    def test_gradual_performance_degradation_scenario(self, comparator):
        """Test scenario where performance gradually degrades across all benchmarks."""
        # Simulate gradual performance degradation that individually isn't alarming
        # but collectively indicates a problem
        current_results = [
            # Each benchmark has 6-7% regression individually
            BenchmarkData(1000, "2D").with_timing(101.0, 106.0, 111.0, "µs"),  # +6%
            BenchmarkData(5000, "2D").with_timing(515.0, 535.0, 555.0, "µs"),  # +7%
            BenchmarkData(10000, "2D").with_timing(1030.0, 1060.0, 1090.0, "µs"),  # +6%
            BenchmarkData(1000, "3D").with_timing(231.0, 265.0, 299.0, "µs"),  # +6%
            BenchmarkData(5000, "3D").with_timing(1300.0, 1325.0, 1350.0, "µs"),  # +6%
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

        # Average change: (6 + 7 + 6 + 6 + 6) / 5 = 6.2%
        # Should detect overall regression even though individual ones are mixed
        assert regression_found

        result = output.getvalue()
        assert "Total benchmarks compared: 5" in result
        assert "Individual regressions (>5.0%): 5" in result  # All 6% and 7% are > 5% threshold
        assert "Average time change: 6.2%" in result
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
        # Despite the one big outlier, no overall regression should be detected (4.9% < 5.0% threshold)
        assert not regression_found

        result = output.getvalue()
        assert "Total benchmarks compared: 5" in result
        assert "Individual regressions (>5.0%): 1" in result  # Only the 40% outlier
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
    @patch("benchmark_utils.os.getenv")
    def test_determine_tag_name_from_github_ref(self, mock_getenv):
        """Test tag name determination from GITHUB_REF with tag."""
        mock_getenv.side_effect = lambda key, default="": {
            "GITHUB_REF": "refs/tags/v1.2.3",
            "GITHUB_OUTPUT": None,
        }.get(key, default)

        tag_name = WorkflowHelper.determine_tag_name()
        assert tag_name == "v1.2.3"

    @patch.dict(os.environ, {"GITHUB_REF": "refs/heads/main"}, clear=False)
    @patch("benchmark_utils.datetime")
    @patch("benchmark_utils.os.getenv")
    def test_determine_tag_name_generated(self, mock_getenv, mock_datetime):
        """Test tag name generation when not from a tag push."""
        # Mock datetime
        mock_now = Mock()
        mock_now.strftime.return_value = "20231215-143000"
        mock_datetime.now.return_value = mock_now

        mock_getenv.side_effect = lambda key, default="": {
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_OUTPUT": None,
        }.get(key, default)

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

    def test_display_baseline_summary_success(self):
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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(baseline_content)
            f.flush()
            baseline_file = Path(f.name)

        try:
            with patch("builtins.print") as mock_print:
                success = WorkflowHelper.display_baseline_summary(baseline_file)
                assert success

                # Check that summary was printed
                print_calls = [call.args[0] for call in mock_print.call_args_list]
                assert any("📊 Baseline summary:" in call for call in print_calls)
                assert any("Total benchmarks: 3" in call for call in print_calls)
                assert any("Date: 2023-12-15 14:30:00 UTC" in call for call in print_calls)
        finally:
            baseline_file.unlink()

    def test_display_baseline_summary_nonexistent_file(self):
        """Test baseline summary with non-existent file."""
        baseline_file = Path("/nonexistent/file.txt")

        with patch("builtins.print") as mock_print:
            success = WorkflowHelper.display_baseline_summary(baseline_file)
            assert not success

            # Check error message was printed to stderr
            stderr_calls = [call for call in mock_print.call_args_list if call.kwargs.get("file") == sys.stderr]
            assert len(stderr_calls) > 0
            assert "❌ Baseline file not found" in stderr_calls[0].args[0]

    def test_display_baseline_summary_long_file(self):
        """Test baseline summary with file longer than 10 lines."""
        baseline_content = "\n".join([f"Line {i}" for i in range(20)])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(baseline_content)
            f.flush()
            baseline_file = Path(f.name)

        try:
            with patch("builtins.print") as mock_print:
                success = WorkflowHelper.display_baseline_summary(baseline_file)
                assert success

                # Check that "..." was printed (indicates truncation)
                print_calls = [call.args[0] for call in mock_print.call_args_list]
                assert "..." in print_calls
        finally:
            baseline_file.unlink()

    def test_sanitize_artifact_name_basic(self):
        """Test basic artifact name sanitization."""
        artifact_name = WorkflowHelper.sanitize_artifact_name("v1.2.3")
        assert artifact_name == "performance-baseline-v1.2.3"

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
                assert artifact_name == "performance-baseline-v2.0.0-beta.1"

            # Check that GITHUB_OUTPUT file was written
            with open(output_file, encoding="utf-8") as f:
                content = f.read()
                assert "artifact_name=performance-baseline-v2.0.0-beta.1\n" in content
        finally:
            Path(output_file).unlink(missing_ok=True)

    def test_sanitize_artifact_name_edge_cases(self):
        """Test artifact name sanitization with edge cases."""
        # Test with unicode and various special characters
        test_cases = [
            ("v1.0.0-alpha.1", "performance-baseline-v1.0.0-alpha.1"),
            ("tag with spaces", "performance-baseline-tag_with_spaces"),
            ("v1.0.0+build.123", "performance-baseline-v1.0.0_build.123"),
            ("@#$%^&*()[]{}|\\<>?", "performance-baseline-__________________"),
        ]

        for input_tag, expected_output in test_cases:
            result = WorkflowHelper.sanitize_artifact_name(input_tag)
            assert result == expected_output


class TestBenchmarkRegressionHelper:
    """Test cases for BenchmarkRegressionHelper class."""

    def test_prepare_baseline_success(self):
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
                with patch.dict(os.environ, {"GITHUB_ENV": env_path}), patch("builtins.print") as mock_print:
                    success = BenchmarkRegressionHelper.prepare_baseline(baseline_dir)

                    assert success

                    # Check that environment variables were set
                    with open(env_path, encoding="utf-8") as f:
                        env_content = f.read()
                        assert "BASELINE_EXISTS=true" in env_content
                        assert "BASELINE_SOURCE=artifact" in env_content

                    # Check that baseline info was printed
                    print_calls = [call.args[0] for call in mock_print.call_args_list]
                    assert any("📦 Prepared baseline from artifact" in call for call in print_calls)
                    assert any("=== Baseline Information" in call for call in print_calls)
            finally:
                Path(env_path).unlink(missing_ok=True)

    def test_prepare_baseline_missing_file(self):
        """Test baseline preparation when baseline file is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)
            # No baseline_results.txt file created

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as env_file:
                env_path = env_file.name

            try:
                with patch.dict(os.environ, {"GITHUB_ENV": env_path}), patch("builtins.print") as mock_print:
                    success = BenchmarkRegressionHelper.prepare_baseline(baseline_dir)

                    assert not success

                    # Check that environment variables were set
                    with open(env_path, encoding="utf-8") as f:
                        env_content = f.read()
                        assert "BASELINE_EXISTS=false" in env_content
                        assert "BASELINE_SOURCE=missing" in env_content

                    # Check error message was printed
                    print_calls = [call.args[0] for call in mock_print.call_args_list]
                    assert any("❌ Downloaded artifact but no baseline_results.txt found" in call for call in print_calls)
            finally:
                Path(env_path).unlink(missing_ok=True)

    def test_set_no_baseline_status(self):
        """Test setting no baseline status."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as env_file:
            env_path = env_file.name

        try:
            with patch.dict(os.environ, {"GITHUB_ENV": env_path}), patch("builtins.print") as mock_print:
                BenchmarkRegressionHelper.set_no_baseline_status()

                # Check that environment variables were set
                with open(env_path, encoding="utf-8") as f:
                    env_content = f.read()
                    assert "BASELINE_EXISTS=false" in env_content
                    assert "BASELINE_SOURCE=none" in env_content
                    assert "BASELINE_ORIGIN=none" in env_content

                # Check message was printed
                print_calls = [call.args[0] for call in mock_print.call_args_list]
                assert any("📈 No baseline artifact found" in call for call in print_calls)
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

                    # Check that environment variable was set
                    with open(env_path, encoding="utf-8") as f:
                        env_content = f.read()
                        assert "BASELINE_COMMIT=abc123def456" in env_content
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

                    # Check that environment variable was set
                    with open(env_path, encoding="utf-8") as f:
                        env_content = f.read()
                        assert "BASELINE_COMMIT=def456abc789" in env_content
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

                    # Check that environment variable was set
                    with open(env_path, encoding="utf-8") as f:
                        env_content = f.read()
                        assert "BASELINE_COMMIT=unknown" in env_content
            finally:
                Path(env_path).unlink(missing_ok=True)

    def test_determine_benchmark_skip_unknown_baseline(self):
        """Test skip determination with unknown baseline commit."""
        should_skip, reason = BenchmarkRegressionHelper.determine_benchmark_skip("unknown", "def456")

        assert not should_skip
        assert reason == "unknown_baseline"

    def test_determine_benchmark_skip_same_commit(self):
        """Test skip determination with same commit."""
        should_skip, reason = BenchmarkRegressionHelper.determine_benchmark_skip("abc123", "abc123")

        assert should_skip
        assert reason == "same_commit"

    @patch("subprocess.run")
    def test_determine_benchmark_skip_baseline_not_found(self, mock_subprocess):
        """Test skip determination when baseline commit not found in history."""
        # Simulate git cat-file failing
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "git cat-file")

        should_skip, reason = BenchmarkRegressionHelper.determine_benchmark_skip("abc123", "def456")

        assert not should_skip
        assert reason == "baseline_commit_not_found"

    @patch("subprocess.run")
    def test_determine_benchmark_skip_no_changes(self, mock_subprocess):
        """Test skip determination when no relevant changes found."""
        # Mock successful git commands
        mock_subprocess.side_effect = [
            Mock(returncode=0),  # git cat-file succeeds
            Mock(returncode=0, stdout="docs/README.md\n.github/workflows/other.yml\n", stderr=""),  # git diff
        ]

        should_skip, reason = BenchmarkRegressionHelper.determine_benchmark_skip("abc123", "def456")

        assert should_skip
        assert reason == "no_relevant_changes"

    @patch("subprocess.run")
    def test_determine_benchmark_skip_changes_detected(self, mock_subprocess):
        """Test skip determination when relevant changes are detected."""
        # Mock successful git commands
        mock_subprocess.side_effect = [
            Mock(returncode=0),  # git cat-file succeeds
            Mock(returncode=0, stdout="src/core/mod.rs\nbenches/performance.rs\n", stderr=""),  # git diff
        ]

        should_skip, reason = BenchmarkRegressionHelper.determine_benchmark_skip("abc123", "def456")

        assert not should_skip
        assert reason == "changes_detected"

    def test_display_skip_message(self):
        """Test displaying skip messages."""
        with patch("builtins.print") as mock_print:
            BenchmarkRegressionHelper.display_skip_message("same_commit", "abc123")

            print_calls = [call.args[0] for call in mock_print.call_args_list]
            assert any("🔍 Current commit matches baseline (abc123)" in call for call in print_calls)

    def test_display_no_baseline_message(self):
        """Test displaying no baseline message."""
        with patch("builtins.print") as mock_print:
            BenchmarkRegressionHelper.display_no_baseline_message()

            print_calls = [call.args[0] for call in mock_print.call_args_list]
            assert any("⚠️ No performance baseline available" in call for call in print_calls)
            assert any("💡 To enable performance regression testing" in call for call in print_calls)

    def test_run_regression_test_success(self):
        """Test successful regression test run."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_file = Path(temp_dir) / "baseline.txt"
            baseline_file.write_text("mock baseline content")

            with patch("benchmark_utils.PerformanceComparator") as mock_comparator_class:
                mock_comparator = Mock()
                mock_comparator.compare_with_baseline.return_value = (True, False)  # success, no regression
                mock_comparator_class.return_value = mock_comparator

                with patch("builtins.print") as mock_print:
                    success = BenchmarkRegressionHelper.run_regression_test(baseline_file)

                    assert success
                    mock_comparator.compare_with_baseline.assert_called_once_with(baseline_file)

                    print_calls = [call.args[0] for call in mock_print.call_args_list]
                    assert any("🚀 Running performance regression test" in call for call in print_calls)

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

    def test_display_results_file_exists(self):
        """Test displaying results when file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_file = Path(temp_dir) / "results.txt"
            results_content = "=== Performance Test Results ===\nAll tests passed\n"
            results_file.write_text(results_content)

            with patch("builtins.print") as mock_print:
                BenchmarkRegressionHelper.display_results(results_file)

                print_calls = [call.args[0] for call in mock_print.call_args_list]
                assert any("=== Performance Regression Test Results ===" in call for call in print_calls)
                assert any("All tests passed" in call for call in print_calls)

    def test_display_results_file_missing(self):
        """Test displaying results when file is missing."""
        missing_file = Path("/nonexistent/results.txt")

        with patch("builtins.print") as mock_print:
            BenchmarkRegressionHelper.display_results(missing_file)

            print_calls = [call.args[0] for call in mock_print.call_args_list]
            assert any("⚠️ No comparison results file found" in call for call in print_calls)

    def test_generate_summary_with_regression(self):
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

            with patch.dict(os.environ, env_vars):
                # Change working directory to temp_dir so Path("benches/compare_results.txt") works
                original_cwd = Path.cwd()
                os.chdir(temp_dir)
                try:
                    with patch("builtins.print") as mock_print:
                        BenchmarkRegressionHelper.generate_summary()

                        print_calls = [call.args[0] for call in mock_print.call_args_list]
                        assert any("📊 Performance Regression Testing Summary" in call for call in print_calls)
                        assert any("Baseline source: artifact" in call for call in print_calls)
                        assert any("Result: ⚠️ Performance regressions detected" in call for call in print_calls)
                finally:
                    os.chdir(original_cwd)

    def test_generate_summary_skip_same_commit(self):
        """Test generating summary when benchmarks skipped due to same commit."""
        env_vars = {
            "BASELINE_SOURCE": "artifact",
            "BASELINE_ORIGIN": "manual",
            "BASELINE_EXISTS": "true",
            "SKIP_BENCHMARKS": "true",
            "SKIP_REASON": "same_commit",
        }

        with patch.dict(os.environ, env_vars), patch("builtins.print") as mock_print:
            BenchmarkRegressionHelper.generate_summary()

            print_calls = [call.args[0] for call in mock_print.call_args_list]
            assert any("Result: ⏭️ Benchmarks skipped (same commit as baseline)" in call for call in print_calls)

    def test_generate_summary_no_baseline(self):
        """Test generating summary when no baseline available."""
        env_vars = {
            "BASELINE_EXISTS": "false",
            "SKIP_BENCHMARKS": "unknown",
        }

        with patch.dict(os.environ, env_vars, clear=True), patch("builtins.print") as mock_print:
            BenchmarkRegressionHelper.generate_summary()

            print_calls = [call.args[0] for call in mock_print.call_args_list]
            assert any("Result: ⏭️ Benchmarks skipped (no baseline available)" in call for call in print_calls)
