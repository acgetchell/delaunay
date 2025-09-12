#!/usr/bin/env python3
"""
Test suite for benchmark_models.py module.

Tests data models, parsing functions, and formatting utilities for benchmark processing.
"""

import re

import pytest

from benchmark_models import (
    BenchmarkData,
    CircumspherePerformanceData,
    CircumsphereTestCase,
    VersionComparisonData,
    extract_benchmark_data,
    format_benchmark_tables,
    format_throughput_value,
    format_time_value,
    parse_benchmark_header,
    parse_throughput_data,
    parse_time_data,
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


class TestCircumspherePerformanceData:
    """Test cases for CircumspherePerformanceData class."""

    def test_init(self):
        """Test CircumspherePerformanceData initialization."""
        data = CircumspherePerformanceData(method="insphere", time_ns=1000.0)
        assert data.method == "insphere"
        assert data.time_ns == 1000.0
        assert data.relative_performance is None
        assert data.winner is False


class TestCircumsphereTestCase:
    """Test cases for CircumsphereTestCase class."""

    def test_init_and_get_winner(self):
        """Test CircumsphereTestCase initialization and winner detection."""
        methods = {
            "insphere": CircumspherePerformanceData("insphere", 1000.0),
            "insphere_distance": CircumspherePerformanceData("insphere_distance", 1200.0),
            "insphere_lifted": CircumspherePerformanceData("insphere_lifted", 800.0),
        }
        test_case = CircumsphereTestCase("test_basic_3d", "3D", methods)

        assert test_case.test_name == "test_basic_3d"
        assert test_case.dimension == "3D"
        assert test_case.get_winner() == "insphere_lifted"  # Lowest time

    def test_get_relative_performance(self):
        """Test relative performance calculation."""
        methods = {
            "insphere": CircumspherePerformanceData("insphere", 1000.0),
            "insphere_distance": CircumspherePerformanceData("insphere_distance", 1200.0),
            "insphere_lifted": CircumspherePerformanceData("insphere_lifted", 800.0),
        }
        test_case = CircumsphereTestCase("test_basic_3d", "3D", methods)

        # Relative to winner (insphere_lifted)
        assert test_case.get_relative_performance("insphere_lifted") == pytest.approx(1.0)
        assert test_case.get_relative_performance("insphere") == pytest.approx(1.25)  # 1000/800
        assert test_case.get_relative_performance("insphere_distance") == pytest.approx(1.5)  # 1200/800

    def test_get_winner_empty_methods(self):
        """Test get_winner with empty methods dict."""
        test_case = CircumsphereTestCase("test_empty", "3D", {})
        assert test_case.get_winner() is None

    def test_get_relative_performance_nonexistent_method(self):
        """Test get_relative_performance with non-existent method returns 0.0."""
        methods = {
            "insphere": CircumspherePerformanceData("insphere", 1000.0),
        }
        test_case = CircumsphereTestCase("test_basic_3d", "3D", methods)

        # Should return 0.0 for non-existent method
        assert test_case.get_relative_performance("nonexistent_method") == pytest.approx(0.0)

    def test_version_comparison_data_division_by_zero_edge_case(self):
        """Test VersionComparisonData handles edge case gracefully."""
        # This doesn't raise an exception but demonstrates pytest usage for edge case testing
        comparison = VersionComparisonData(
            test_case="Edge Case",
            method="insphere",
            old_version="v0.3.0",
            new_version="v0.3.1",
            old_value=0.0,  # Zero old value
            new_value=100.0,
            unit="ns",
        )

        # Should handle division by zero gracefully (returns 0.0)
        assert comparison.improvement_pct == pytest.approx(0.0)


class TestVersionComparisonData:
    """Test cases for VersionComparisonData class."""

    def test_improvement_calculation(self):
        """Test improvement percentage calculation."""
        comparison = VersionComparisonData(
            test_case="Basic 3D",
            method="insphere",
            old_version="v0.3.0",
            new_version="v0.3.1",
            old_value=808.0,
            new_value=805.0,
            unit="ns",
        )

        expected_improvement = ((808.0 - 805.0) / 808.0) * 100
        assert comparison.improvement_pct == pytest.approx(expected_improvement, abs=0.001)

    def test_zero_old_value(self):
        """Test improvement calculation with zero old value."""
        comparison = VersionComparisonData(
            test_case="Basic 3D",
            method="insphere",
            old_version="v0.3.0",
            new_version="v0.3.1",
            old_value=0.0,
            new_value=805.0,
            unit="ns",
        )

        assert comparison.improvement_pct == 0.0


class TestParsingFunctions:
    """Test cases for parsing functions."""

    def test_extract_benchmark_data(self):
        """Test extracting benchmark data from baseline content."""
        baseline_content = """Date: 2024-01-15 10:30:00 UTC
Git commit: abc123def456

=== 1000 Points (2D) ===
Time: [100.0, 110.0, 120.0] µs
Throughput: [8000.0, 9090.9, 10000.0] Kelem/s

=== 5000 Points (3D) ===
Time: [500.0, 550.0, 600.0] µs
Throughput: [8333.3, 9090.9, 10000.0] Kelem/s
"""

        benchmarks = extract_benchmark_data(baseline_content)

        assert len(benchmarks) == 2

        # Check first benchmark
        first = benchmarks[0]
        assert first.points == 1000
        assert first.dimension == "2D"
        assert first.time_mean == 110.0
        assert first.time_unit == "µs"
        assert first.throughput_mean == 9090.9

        # Check second benchmark
        second = benchmarks[1]
        assert second.points == 5000
        assert second.dimension == "3D"
        assert second.time_mean == 550.0

    def test_parse_benchmark_header(self):
        """Test parsing benchmark header lines."""
        # Valid header
        result = parse_benchmark_header("=== 1000 Points (2D) ===")
        assert result is not None
        assert result.points == 1000
        assert result.dimension == "2D"

        # Invalid header
        result = parse_benchmark_header("Invalid header")
        assert result is None

    def test_parse_time_data(self):
        """Test parsing time data lines."""
        benchmark = BenchmarkData(1000, "2D")

        # Valid time data
        success = parse_time_data(benchmark, "Time: [100.0, 110.0, 120.0] µs")
        assert success is True
        assert benchmark.time_mean == 110.0
        assert benchmark.time_unit == "µs"

        # Invalid time data
        benchmark2 = BenchmarkData(1000, "2D")
        success = parse_time_data(benchmark2, "Invalid time data")
        assert success is False

    def test_parse_throughput_data(self):
        """Test parsing throughput data lines."""
        benchmark = BenchmarkData(1000, "2D")

        # Valid throughput data
        success = parse_throughput_data(benchmark, "Throughput: [8000.0, 9090.9, 10000.0] Kelem/s")
        assert success is True
        assert benchmark.throughput_mean == 9090.9
        assert benchmark.throughput_unit == "Kelem/s"

        # Invalid throughput data
        benchmark2 = BenchmarkData(1000, "2D")
        success = parse_throughput_data(benchmark2, "Invalid throughput data")
        assert success is False


class TestFormattingFunctions:
    """Test cases for formatting functions."""

    def test_format_benchmark_tables(self):
        """Test formatting benchmark data as markdown tables."""
        # Create test benchmarks
        benchmarks = [
            BenchmarkData(1000, "2D").with_timing(100.0, 110.0, 120.0, "µs").with_throughput(8000.0, 9090.9, 10000.0, "Kelem/s"),
            BenchmarkData(5000, "2D").with_timing(450.0, 500.0, 550.0, "µs").with_throughput(9000.0, 10000.0, 11000.0, "Kelem/s"),
            BenchmarkData(2000, "3D").with_timing(200.0, 220.0, 240.0, "µs").with_throughput(8000.0, 9090.9, 10000.0, "Kelem/s"),
        ]

        lines = format_benchmark_tables(benchmarks)
        markdown_content = "\n".join(lines)

        # Should have sections for both dimensions
        assert "### 2D Triangulation Performance" in markdown_content
        assert "### 3D Triangulation Performance" in markdown_content

        # Should have table headers
        assert "| Points | Time (mean) | Throughput (mean) | Scaling |" in markdown_content
        assert "|--------|-------------|-------------------|----------|" in markdown_content

        # Should have data rows with scaling calculations
        assert "| 1000 | 110.00 µs | 9090.900 Kelem/s | 1.0x |" in markdown_content
        assert "| 5000 |" in markdown_content  # Should contain the 5000 point row
        assert "4.5x" in markdown_content  # Scaling: 500/110 ≈ 4.5

    def test_format_time_value(self):
        """Test formatting time values with appropriate precision."""
        # Test zero and negative values (should return N/A)
        assert format_time_value(0.0, "µs") == "N/A"
        assert format_time_value(-1.0, "µs") == "N/A"

        # Test different value ranges
        assert format_time_value(0.5, "µs") == "0.500 µs"
        assert format_time_value(110.0, "µs") == "110.00 µs"
        assert format_time_value(1500.0, "µs") == "1.500 ms"  # Converts to ms
        assert format_time_value(2500.0, "ms") == "2.5000 s"  # Converts to s
        assert format_time_value(50000.0, "ms") == "50.0000 s"  # Large values convert to s

    def test_format_throughput_value(self):
        """Test formatting throughput values with appropriate precision."""
        # Test different value ranges
        assert format_throughput_value(0.5, "Kelem/s") == "0.500 Kelem/s"
        assert format_throughput_value(110.0, "Kelem/s") == "110.00 Kelem/s"
        assert format_throughput_value(9090.909, "Kelem/s") == "9090.909 Kelem/s"

        # Test None values
        assert format_throughput_value(None, "Kelem/s") == "N/A"
        assert format_throughput_value(110.0, None) == "N/A"

    def test_format_time_value_with_unit_aliases(self):
        """Test time value formatting with microsecond unit aliases."""
        # Test microsecond alias normalization
        assert format_time_value(500.0, "us") == "500.00 µs"  # us -> µs
        assert format_time_value(500.0, "μs") == "500.00 µs"  # μs -> µs
        assert format_time_value(500.0, "µs") == "500.00 µs"  # already µs

        # Test unit conversion with aliases
        assert format_time_value(1500.0, "us") == "1.500 ms"  # us -> µs -> ms conversion
        assert format_time_value(2500.0, "μs") == "2.500 ms"  # μs -> µs -> ms conversion

    def test_parse_time_data_with_scientific_notation(self):
        """Test parsing time data with scientific notation and flexible formatting."""
        benchmark = BenchmarkData(1000, "3D")

        # Test scientific notation parsing
        success = parse_time_data(benchmark, "Time: [1.0e2, 1.1e2, 1.2e2] µs")
        assert success is True
        assert benchmark.time_mean == 110.0
        assert benchmark.time_unit == "µs"

        # Test negative values
        benchmark2 = BenchmarkData(1000, "3D")
        success = parse_time_data(benchmark2, "Time: [-1.0, 0.0, 1.0] µs")
        assert success is True
        assert benchmark2.time_mean == 0.0

        # Test flexible whitespace
        benchmark3 = BenchmarkData(1000, "3D")
        success = parse_time_data(benchmark3, "Time:   [ 100.0 ,  110.0,   120.0 ]   µs")
        assert success is True
        assert benchmark3.time_mean == 110.0
        assert benchmark3.time_unit == "µs"

    def test_parse_throughput_data_with_scientific_notation(self):
        """Test parsing throughput data with scientific notation and flexible formatting."""
        benchmark = BenchmarkData(1000, "2D")

        # Test scientific notation parsing
        success = parse_throughput_data(benchmark, "Throughput: [8.0e3, 9.09e3, 1.0e4] Kelem/s")
        assert success is True
        assert benchmark.throughput_mean == 9090.0
        assert benchmark.throughput_unit == "Kelem/s"

        # Test flexible whitespace
        benchmark2 = BenchmarkData(1000, "2D")
        success = parse_throughput_data(benchmark2, "Throughput:   [ 8000.0 ,  9090.9,   10000.0 ]   Kelem/s")
        assert success is True
        assert benchmark2.throughput_mean == 9090.9
        assert benchmark2.throughput_unit == "Kelem/s"

    def test_format_benchmark_tables_dimension_sorting(self):
        """Test that dimensions are sorted numerically rather than lexically."""
        # Create benchmarks with dimensions that would sort incorrectly lexically
        benchmarks = [
            BenchmarkData(1000, "10D").with_timing(100.0, 110.0, 120.0, "µs"),
            BenchmarkData(1000, "2D").with_timing(50.0, 55.0, 60.0, "µs"),
            BenchmarkData(1000, "3D").with_timing(70.0, 75.0, 80.0, "µs"),
            BenchmarkData(1000, "1D").with_timing(30.0, 35.0, 40.0, "µs"),
        ]

        lines = format_benchmark_tables(benchmarks)
        markdown_content = "\n".join(lines)

        # Find positions of dimension headers
        pos_1d = markdown_content.find("### 1D Triangulation Performance")
        pos_2d = markdown_content.find("### 2D Triangulation Performance")
        pos_3d = markdown_content.find("### 3D Triangulation Performance")
        pos_10d = markdown_content.find("### 10D Triangulation Performance")

        # Verify they appear in numeric order: 1D < 2D < 3D < 10D
        assert pos_1d < pos_2d < pos_3d < pos_10d

    def test_format_benchmark_tables_mixed_dimension_formats(self):
        """Test dimension sorting with mixed formats and edge cases."""
        benchmarks = [
            BenchmarkData(1000, "2D").with_timing(50.0, 55.0, 60.0, "µs"),
            BenchmarkData(1000, "custom_format").with_timing(90.0, 95.0, 100.0, "µs"),  # No numeric prefix
            BenchmarkData(1000, "  3D  ").with_timing(70.0, 75.0, 80.0, "µs"),  # Whitespace
            BenchmarkData(1000, "1d").with_timing(30.0, 35.0, 40.0, "µs"),  # Lowercase 'd'
        ]

        lines = format_benchmark_tables(benchmarks)
        markdown_content = "\n".join(lines)

        # Find positions (note: whitespace in dimension names is preserved)
        pos_1d = markdown_content.find("### 1d Triangulation Performance")
        pos_2d = markdown_content.find("### 2D Triangulation Performance")
        pos_3d = markdown_content.find("###   3D   Triangulation Performance")  # Whitespace preserved
        pos_custom = markdown_content.find("### custom_format Triangulation Performance")

        # Numeric dimensions should come first (1d, 2D, 3D), then non-numeric (custom_format)
        assert pos_1d < pos_2d < pos_3d < pos_custom

    def test_format_benchmark_tables_scaling_baseline_with_zero_first_entry(self):
        """Test scaling baseline calculation when first entry has zero/empty time.

        This tests the fix for the issue where using 1.0 as fallback when the
        first entry has zero/empty time inflates scaling calculations.
        """
        benchmarks = [
            BenchmarkData(1000, "2D").with_timing(0.0, 0.0, 0.0, "µs"),  # Zero time (invalid)
            BenchmarkData(2000, "2D").with_timing(100.0, 110.0, 120.0, "µs"),  # Valid baseline
            BenchmarkData(5000, "2D").with_timing(450.0, 500.0, 550.0, "µs"),  # Should scale against 110.0
        ]

        lines = format_benchmark_tables(benchmarks)
        markdown_content = "\n".join(lines)

        # First entry should show N/A for time and scaling (because it has zero time)
        assert "| 1000 | N/A | N/A | N/A |" in markdown_content

        # Second entry should be baseline (1.0x) since it's first valid entry
        assert "| 2000 | 110.00 µs | N/A | 1.0x |" in markdown_content

        # Third entry should scale against 110.0 (not against 1.0 fallback)
        # Scaling: 500/110 ≈ 4.5
        assert "4.5x" in markdown_content
        # Should not contain inflated scaling that would result from 1.0 fallback
        assert "500.0x" not in markdown_content  # This would be 500/1.0 if bug existed

    def test_format_benchmark_tables_scaling_baseline_all_zero_times(self):
        """Test scaling baseline calculation when all entries have zero/empty time."""
        benchmarks = [
            BenchmarkData(1000, "2D").with_timing(0.0, 0.0, 0.0, "µs"),
            BenchmarkData(2000, "2D").with_timing(0.0, 0.0, 0.0, "µs"),
            BenchmarkData(5000, "2D").with_timing(0.0, 0.0, 0.0, "µs"),
        ]

        lines = format_benchmark_tables(benchmarks)
        markdown_content = "\n".join(lines)

        # All entries should show N/A for time and scaling since no valid baseline exists
        assert "| 1000 | N/A | N/A | N/A |" in markdown_content
        assert "| 2000 | N/A | N/A | N/A |" in markdown_content
        assert "| 5000 | N/A | N/A | N/A |" in markdown_content

        # Should not contain any numeric scaling values
        numeric_scaling_pattern = r"\| [^|]+ \| [^|]+ \| [^|]+ \| [0-9.]+x \|"
        assert not re.search(numeric_scaling_pattern, markdown_content)
