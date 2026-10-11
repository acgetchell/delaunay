"""Delaunay circumsphere method rankings for the absolute performance summary."""

from dataclasses import dataclass


@dataclass
class CircumspherePerformanceData:
    """Represents circumsphere containment performance data."""

    method: str  # insphere, insphere_distance, insphere_lifted
    time_ns: float
    relative_performance: float | None = None
    winner: bool = False


@dataclass
class CircumsphereTestCase:
    """Represents a circumsphere test case with multiple method results."""

    test_name: str
    dimension: str
    methods: dict[str, CircumspherePerformanceData]
    is_boundary_case: bool = False  # True for boundary/edge cases with early-exit optimizations

    def get_winner(self) -> str | None:
        """Get the method name with the best performance."""
        if not self.methods:
            return None
        return min(self.methods.keys(), key=lambda m: self.methods[m].time_ns)

    def get_relative_performance(self, method: str, baseline_method: str | None = None) -> float:
        """Calculate relative performance compared to baseline method."""
        if method not in self.methods:
            return 0.0

        if baseline_method is None:
            baseline_method = self.get_winner()

        if baseline_method is None or baseline_method not in self.methods:
            return 1.0

        baseline_time = self.methods[baseline_method].time_ns
        method_time = self.methods[method].time_ns

        if baseline_time <= 0:
            return 1.0

        return method_time / baseline_time
