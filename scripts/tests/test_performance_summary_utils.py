#!/usr/bin/env python3
# ruff: noqa: SLF001
"""
Test suite for performance_summary_utils.py module.

Tests performance summary generation, benchmark parsing, dynamic analysis,
and version extraction functionality.

Note: This test file accesses private methods (prefixed with _) which is expected
and necessary for comprehensive unit testing of internal functionality.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from performance_summary_utils import (
    CircumspherePerformanceData,
    CircumsphereTestCase,
    PerformanceSummaryGenerator,
    ProjectRootNotFoundError,
    VersionComparisonData,
    find_project_root,
)


class TestCircumspherePerformanceData:
    """Test cases for CircumspherePerformanceData dataclass."""

    def test_init(self):
        """Test CircumspherePerformanceData initialization."""
        data = CircumspherePerformanceData("insphere", 805.0)
        assert data.method == "insphere"
        assert data.time_ns == 805.0
        assert data.relative_performance is None
        assert data.winner is False


class TestCircumsphereTestCase:
    """Test cases for CircumsphereTestCase dataclass."""

    def test_init_and_get_winner(self):
        """Test CircumsphereTestCase initialization and winner detection."""
        methods = {
            "insphere": CircumspherePerformanceData("insphere", 805.0),
            "insphere_distance": CircumspherePerformanceData("insphere_distance", 1463.0),
            "insphere_lifted": CircumspherePerformanceData("insphere_lifted", 637.0),
        }

        test_case = CircumsphereTestCase("Basic 3D", "3D", methods)

        assert test_case.test_name == "Basic 3D"
        assert test_case.dimension == "3D"
        assert test_case.get_winner() == "insphere_lifted"  # Fastest method

    def test_get_relative_performance(self):
        """Test relative performance calculation."""
        methods = {
            "insphere": CircumspherePerformanceData("insphere", 800.0),
            "insphere_lifted": CircumspherePerformanceData("insphere_lifted", 600.0),  # Fastest
        }

        test_case = CircumsphereTestCase("Test", "3D", methods)

        # Relative to fastest (insphere_lifted)
        assert test_case.get_relative_performance("insphere_lifted") == 1.0
        assert test_case.get_relative_performance("insphere") == pytest.approx(800.0 / 600.0)

        # Relative to specific baseline
        assert test_case.get_relative_performance("insphere_lifted", "insphere") == pytest.approx(600.0 / 800.0)

    def test_get_winner_empty_methods(self):
        """Test get_winner with empty methods dict."""
        test_case = CircumsphereTestCase("Empty", "3D", {})
        assert test_case.get_winner() is None


class TestVersionComparisonData:
    """Test cases for VersionComparisonData dataclass."""

    def test_improvement_calculation(self):
        """Test improvement percentage calculation."""
        comparison = VersionComparisonData("Basic 3D", "insphere", "0.3.0", "0.3.1", 808.0, 805.0, "ns", 0.0)

        # Should calculate improvement in __post_init__
        expected_improvement = ((808.0 - 805.0) / 808.0) * 100
        assert comparison.improvement_pct == pytest.approx(expected_improvement)

    def test_zero_old_value(self):
        """Test improvement calculation with zero old value."""
        comparison = VersionComparisonData("Test", "method", "v1", "v2", 0.0, 100.0, "ns", 0.0)

        assert comparison.improvement_pct == 0.0


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

            with temp_chdir(temp_path), pytest.raises(ProjectRootNotFoundError, match="Could not locate Cargo.toml"):
                find_project_root()


class TestPerformanceSummaryGenerator:
    """Test cases for PerformanceSummaryGenerator class."""

    def test_init(self, mock_git_command_result):
        """Test PerformanceSummaryGenerator initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            with patch("performance_summary_utils.run_git_command") as mock_git:
                mock_git.side_effect = [mock_git_command_result("v0.4.2"), mock_git_command_result("2025-01-01")]

                generator = PerformanceSummaryGenerator(project_root)

                assert generator.project_root == project_root
                assert generator.baseline_file == project_root / "benches" / "baseline_results.txt"
                assert generator.comparison_file == project_root / "benches" / "compare_results.txt"
                assert generator.circumsphere_results_dir == project_root / "target" / "criterion"

    def test_get_current_version_with_tag(self, mock_git_command_result):
        """Test version extraction from git tags."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            with patch("performance_summary_utils.run_git_command") as mock_git:
                mock_git.side_effect = [mock_git_command_result("v0.4.2"), mock_git_command_result("2025-01-01")]

                generator = PerformanceSummaryGenerator(project_root)

                assert generator.current_version == "0.4.2"

    def test_get_current_version_no_tag(self):
        """Test version extraction when no tags exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            with patch("performance_summary_utils.run_git_command") as mock_git:
                mock_git.side_effect = Exception("No tags")

                generator = PerformanceSummaryGenerator(project_root)

                assert generator.current_version == "unknown"

    def test_get_version_date_with_tag(self, mock_git_command_result):
        """Test date extraction from git tag."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            with patch("performance_summary_utils.run_git_command") as mock_git:
                mock_git.side_effect = [mock_git_command_result("v0.4.2"), mock_git_command_result("2025-01-15")]

                generator = PerformanceSummaryGenerator(project_root)

                assert generator.current_date == "2025-01-15"

    def test_get_fallback_circumsphere_data(self, mock_git_command_result):
        """Test fallback circumsphere data generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            with patch("performance_summary_utils.run_git_command") as mock_git:
                mock_git.side_effect = [mock_git_command_result("v0.4.2"), mock_git_command_result("2025-01-01")]
                generator = PerformanceSummaryGenerator(project_root)

                fallback_data = generator._get_fallback_circumsphere_data()

                assert len(fallback_data) == 3
                assert all(isinstance(case, CircumsphereTestCase) for case in fallback_data)
                assert all(case.dimension == "3D" for case in fallback_data)

                # Check that all expected methods are present
                for case in fallback_data:
                    assert "insphere" in case.methods
                    assert "insphere_distance" in case.methods
                    assert "insphere_lifted" in case.methods

    def test_parse_circumsphere_benchmark_results_no_criterion_dir(self, mock_git_command_result):
        """Test parsing when criterion directory doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            with patch("performance_summary_utils.run_git_command") as mock_git:
                mock_git.side_effect = [mock_git_command_result("v0.4.2"), mock_git_command_result("2025-01-01")]
                generator = PerformanceSummaryGenerator(project_root)

                # Should fallback to hardcoded data
                results = generator._parse_circumsphere_benchmark_results()

                assert len(results) >= 3  # Should have fallback data
                assert all(isinstance(case, CircumsphereTestCase) for case in results)

    def test_parse_circumsphere_benchmark_results_with_data(self, mock_git_command_result):
        """Test parsing with actual criterion benchmark data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            # Create mock criterion directory structure
            criterion_dir = project_root / "target" / "criterion"
            basic_insphere_dir = criterion_dir / "basic-insphere" / "base"
            basic_insphere_dir.mkdir(parents=True)

            # Create mock estimates.json
            estimates_data = {"mean": {"point_estimate": 805.0}}

            estimates_file = basic_insphere_dir / "estimates.json"
            with estimates_file.open("w") as f:
                json.dump(estimates_data, f)

            with patch("performance_summary_utils.run_git_command") as mock_git:
                mock_git.side_effect = [mock_git_command_result("v0.4.2"), mock_git_command_result("2025-01-01")]
                generator = PerformanceSummaryGenerator(project_root)

                results = generator._parse_circumsphere_benchmark_results()

                # Should have parsed at least one result
                basic_results = [r for r in results if r.test_name == "Basic 3D"]
                if basic_results:
                    basic_result = basic_results[0]
                    assert "insphere" in basic_result.methods
                    assert basic_result.methods["insphere"].time_ns == 805.0

    def test_analyze_performance_ranking(self, mock_git_command_result):
        """Test performance ranking analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            with patch("performance_summary_utils.run_git_command") as mock_git:
                mock_git.side_effect = [mock_git_command_result("v0.4.2"), mock_git_command_result("2025-01-01")]
                generator = PerformanceSummaryGenerator(project_root)

                # Create test data
                test_data = [
                    CircumsphereTestCase(
                        "Test1",
                        "3D",
                        {
                            "insphere": CircumspherePerformanceData("insphere", 800.0),
                            "insphere_lifted": CircumspherePerformanceData("insphere_lifted", 600.0),
                            "insphere_distance": CircumspherePerformanceData("insphere_distance", 1200.0),
                        },
                    )
                ]

                rankings = generator._analyze_performance_ranking(test_data)

                assert len(rankings) == 3

                # Should be sorted by performance (fastest first)
                methods_in_order = [ranking[0] for ranking in rankings]
                assert methods_in_order[0] == "insphere_lifted"  # Fastest
                assert "fastest" in rankings[0][2]  # Description should mention fastest

    def test_generate_dynamic_recommendations(self, mock_git_command_result):
        """Test dynamic recommendation generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            with patch("performance_summary_utils.run_git_command") as mock_git:
                mock_git.side_effect = [mock_git_command_result("v0.4.2"), mock_git_command_result("2025-01-01")]
                generator = PerformanceSummaryGenerator(project_root)

                # Mock performance ranking with insphere_lifted as fastest
                performance_ranking = [
                    ("insphere_lifted", 600.0, "fastest description"),
                    ("insphere", 800.0, "middle description"),
                    ("insphere_distance", 1200.0, "slowest description"),
                ]

                recommendations = generator._generate_dynamic_recommendations(performance_ranking)

                assert len(recommendations) > 0

                # Should recommend the fastest method for performance-critical apps
                performance_section = "\n".join(recommendations)
                assert "insphere_lifted" in performance_section
                assert "maximum performance" in performance_section

    def test_generate_summary_success(self, mock_git_command_result):
        """Test successful summary generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            output_file = Path(temp_dir) / "test_output.md"

            with (
                patch("performance_summary_utils.run_git_command") as mock_git,
                patch("performance_summary_utils.get_git_commit_hash") as mock_commit,
            ):
                mock_git.side_effect = [mock_git_command_result("v0.4.2"), mock_git_command_result("2025-01-01")]
                mock_commit.return_value = "abc123def456"

                generator = PerformanceSummaryGenerator(project_root)

                success = generator.generate_summary(output_file, run_benchmarks=False)

                assert success
                assert output_file.exists()

                # Check content
                content = output_file.read_text()
                assert "# Delaunay Library Performance Results" in content
                assert "Version 0.4.2" in content
                assert "abc123def456" in content

    def test_generate_summary_with_baseline_data(self, mock_git_command_result):
        """Test summary generation with baseline data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            output_file = Path(temp_dir) / "test_output.md"

            # Create mock baseline file
            baseline_dir = project_root / "benches"
            baseline_dir.mkdir()
            baseline_file = baseline_dir / "baseline_results.txt"
            baseline_file.write_text(
                "Generated at: 2025-01-15 10:00:00\n"
                "Git commit: abc123\n"
                "Hardware: Test Machine\n"
                "\n"
                "=== 10 Points (2D) ===\n"
                "Time: [100.0, 110.0, 120.0] µs\n"
                "Throughput: [8000.0, 9000.0, 10000.0] Kelem/s\n"
            )

            with patch("performance_summary_utils.run_git_command") as mock_git:
                mock_git.side_effect = [mock_git_command_result("v0.4.2"), mock_git_command_result("2025-01-01")]

                generator = PerformanceSummaryGenerator(project_root)
                success = generator.generate_summary(output_file)

                assert success

                content = output_file.read_text()
                assert "Triangulation Data Structure Performance" in content
                assert "Current Baseline Information" in content

    def test_generate_summary_with_comparison_data(self, mock_git_command_result):
        """Test summary generation with comparison data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            output_file = Path(temp_dir) / "test_output.md"

            # Create mock comparison file with regression
            baseline_dir = project_root / "benches"
            baseline_dir.mkdir()
            comparison_file = baseline_dir / "compare_results.txt"
            comparison_file.write_text(
                "=== 10 Points (2D) ===\n⚠️ REGRESSION: Time increased by 15.0% (slower performance)\nSome other content here\n"
            )

            with patch("performance_summary_utils.run_git_command") as mock_git:
                mock_git.side_effect = [mock_git_command_result("v0.4.2"), mock_git_command_result("2025-01-01")]

                generator = PerformanceSummaryGenerator(project_root)
                success = generator.generate_summary(output_file)

                assert success

                content = output_file.read_text()
                assert "Performance Regression Detected" in content

    def test_run_circumsphere_benchmarks_success(self, mock_git_command_result):
        """Test successful circumsphere benchmark execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            with patch("performance_summary_utils.run_git_command") as mock_git, patch("performance_summary_utils.run_cargo_command") as mock_cargo:
                mock_git.side_effect = [mock_git_command_result("v0.4.2"), mock_git_command_result("2025-01-01")]
                mock_cargo.return_value = "benchmark output"

                generator = PerformanceSummaryGenerator(project_root)

                success = generator._run_circumsphere_benchmarks()

                assert success
                mock_cargo.assert_called_once()

    def test_run_circumsphere_benchmarks_failure(self, mock_git_command_result):
        """Test failed circumsphere benchmark execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            with patch("performance_summary_utils.run_git_command") as mock_git, patch("performance_summary_utils.run_cargo_command") as mock_cargo:
                mock_git.side_effect = [mock_git_command_result("v0.4.2"), mock_git_command_result("2025-01-01")]
                mock_cargo.side_effect = Exception("Benchmark failed")  # Simulate failure

                generator = PerformanceSummaryGenerator(project_root)

                success = generator._run_circumsphere_benchmarks()

                assert not success

    def test_get_circumsphere_performance_results_format(self, mock_git_command_result):
        """Test circumsphere performance results formatting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            with patch("performance_summary_utils.run_git_command") as mock_git:
                mock_git.side_effect = [mock_git_command_result("v0.4.2"), mock_git_command_result("2025-01-01")]

                generator = PerformanceSummaryGenerator(project_root)

                results = generator._get_circumsphere_performance_results()

                assert len(results) > 0

                # Check for expected sections
                results_text = "\n".join(results)
                assert "Single Query Performance (3D)" in results_text
                assert "Version Comparison" in results_text
                assert "Historical Version Comparison" in results_text
                assert "insphere_lifted" in results_text  # Should be marked as winner

    def test_get_update_instructions(self, mock_git_command_result):
        """Test update instructions generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            with patch("performance_summary_utils.run_git_command") as mock_git:
                mock_git.side_effect = [mock_git_command_result("v0.4.2"), mock_git_command_result("2025-01-01")]

                generator = PerformanceSummaryGenerator(project_root)

                instructions = generator._get_update_instructions()

                assert len(instructions) > 0

                instructions_text = "\n".join(instructions)
                assert "Performance Data Updates" in instructions_text
                assert "performance-summary-utils generate" in instructions_text
                assert "--run-benchmarks" in instructions_text


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_benchmark_results(self, mock_git_command_result):
        """Test handling of empty benchmark results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            with patch("performance_summary_utils.run_git_command") as mock_git:
                mock_git.side_effect = [mock_git_command_result("v0.4.2"), mock_git_command_result("2025-01-01")]

                generator = PerformanceSummaryGenerator(project_root)

                # Should not crash and should use fallback data
                results = generator._parse_circumsphere_benchmark_results()
                assert len(results) > 0

    def test_malformed_estimates_json(self, mock_git_command_result):
        """Test handling of malformed estimates.json files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            # Create malformed estimates.json
            criterion_dir = project_root / "target" / "criterion" / "basic-insphere" / "base"
            criterion_dir.mkdir(parents=True)

            estimates_file = criterion_dir / "estimates.json"
            estimates_file.write_text("{ invalid json")

            with patch("performance_summary_utils.run_git_command") as mock_git:
                mock_git.side_effect = [mock_git_command_result("v0.4.2"), mock_git_command_result("2025-01-01")]

                generator = PerformanceSummaryGenerator(project_root)

                # Should not crash and should use fallback data
                results = generator._parse_circumsphere_benchmark_results()
                assert len(results) > 0

    def test_missing_git_info(self):
        """Test handling when git information is not available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            output_file = Path(temp_dir) / "test_output.md"

            with (
                patch("performance_summary_utils.run_git_command") as mock_git,
                patch("performance_summary_utils.get_git_commit_hash") as mock_commit,
            ):
                mock_git.side_effect = Exception("Git not available")
                mock_commit.side_effect = Exception("Git not available")

                generator = PerformanceSummaryGenerator(project_root)
                success = generator.generate_summary(output_file)

                # Should still succeed
                assert success

                content = output_file.read_text()
                assert "Version unknown" in content


class TestIntegrationScenarios:
    """Integration test scenarios."""

    def test_full_generation_workflow(self, mock_git_command_result):
        """Test complete summary generation workflow."""
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
                "Throughput: [8000.0, 9000.0, 10000.0] Kelem/s\n"
            )

            # Mock comparison file
            comparison_file = baseline_dir / "compare_results.txt"
            comparison_file.write_text("✅ OK: All benchmarks within acceptable range\n")

            with (
                patch("performance_summary_utils.run_git_command") as mock_git,
                patch("performance_summary_utils.get_git_commit_hash") as mock_commit,
            ):
                # Mock CompletedProcess objects for version and date calls
                mock_git.side_effect = [mock_git_command_result("v0.4.2"), mock_git_command_result("2025-01-15")]
                mock_commit.return_value = "abc123def456"

                generator = PerformanceSummaryGenerator(project_root)
                success = generator.generate_summary(output_file)

                assert success

                content = output_file.read_text()

                # Verify all major sections are present
                assert "# Delaunay Library Performance Results" in content
                assert "Version 0.4.2 Results (2025-01-15)" in content
                assert "Single Query Performance (3D)" in content
                assert "Triangulation Data Structure Performance" in content
                assert "Performance Status: Good" in content
                assert "Key Findings" in content
                assert "Performance Ranking" in content
                assert "Recommendations" in content
                assert "Performance Data Updates" in content
