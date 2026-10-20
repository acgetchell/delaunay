"""Tests for Delaunay circumsphere rankings."""

from benchmark_models import CircumspherePerformanceData, CircumsphereTestCase


class TestCircumspherePerformanceData:
    """Test cases for CircumspherePerformanceData class."""

    def test_init(self) -> None:
        """Test CircumspherePerformanceData initialization."""
        data = CircumspherePerformanceData(method="insphere", time_ns=1000.0)
        assert data.method == "insphere"
        assert data.time_ns == 1000.0
        assert data.relative_performance is None
        assert data.winner is False


class TestCircumsphereTestCase:
    """Test cases for CircumsphereTestCase class."""

    def test_init_and_get_winner(self) -> None:
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

    def test_get_winner_empty_methods(self) -> None:
        """Test get_winner with empty methods dict."""
        test_case = CircumsphereTestCase("test_empty", "3D", {})
        assert test_case.get_winner() is None
