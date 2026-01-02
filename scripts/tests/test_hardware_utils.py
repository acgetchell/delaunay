#!/usr/bin/env python3
"""
Test suite for hardware_utils.py module.

Tests hardware information detection and comparison functionality
across different platforms with proper mocking.
"""

import platform
import subprocess
from unittest.mock import Mock, mock_open, patch

import pytest

from hardware_utils import HardwareComparator, HardwareInfo


@pytest.fixture
def hardware():
    """Fixture for HardwareInfo instance."""
    return HardwareInfo()


class TestHardwareInfo:
    """Test cases for HardwareInfo class."""

    def test_init(self, hardware):
        """Test HardwareInfo initialization."""
        assert hardware.os_type == platform.system()
        assert hardware.machine == platform.machine()

    @patch("hardware_utils.platform.system")
    def test_init_with_different_os(self, mock_system):
        """Test initialization with different OS types."""
        mock_system.return_value = "Linux"
        hardware = HardwareInfo()
        assert hardware.os_type == "Linux"

    def test_run_command_empty_cmd(self, hardware):
        """Test _run_command with empty command list."""
        with pytest.raises(ValueError, match="Command list cannot be empty"):
            hardware._run_command([])  # noqa: SLF001

    @patch("hardware_utils.run_safe_command")
    def test_run_command_success(self, mock_run_safe, hardware):
        """Test successful command execution."""
        mock_result = Mock()
        mock_result.stdout = "test output\n"
        mock_run_safe.return_value = mock_result

        result = hardware._run_command(["echo", "test"])  # noqa: SLF001

        assert result == "test output"
        mock_run_safe.assert_called_once_with("echo", ["test"], capture_output=True, text=True, check=True)

    @patch("hardware_utils.run_safe_command")
    def test_run_command_failure(self, mock_run_safe, hardware):
        """Test command execution failure."""
        mock_run_safe.side_effect = subprocess.CalledProcessError(1, "cmd")

        with pytest.raises(subprocess.CalledProcessError):
            hardware._run_command(["false"])  # noqa: SLF001

    @patch("hardware_utils.platform.system")
    @patch.object(HardwareInfo, "_run_command")
    def test_get_cpu_info_darwin(self, mock_run_command, mock_system):
        """Test CPU info detection on macOS."""
        mock_system.return_value = "Darwin"
        mock_run_command.side_effect = ["Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz", "6", "12"]

        hardware = HardwareInfo()
        cpu_model, cpu_cores, cpu_threads = hardware.get_cpu_info()

        assert cpu_model == "Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz"
        assert cpu_cores == "6"
        assert cpu_threads == "12"

    @patch("hardware_utils.platform.system")
    @patch("hardware_utils.shutil.which")
    @patch.object(HardwareInfo, "_run_command")
    def test_get_cpu_info_linux_with_lscpu(self, mock_run_command, mock_which, mock_system):
        """Test CPU info detection on Linux with lscpu available."""
        mock_system.return_value = "Linux"
        mock_which.side_effect = lambda cmd: cmd in ["lscpu", "nproc"]

        lscpu_output = """Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Model name:          Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz
Core(s) per socket:  6
Socket(s):           1
Thread(s) per core:  2"""

        mock_run_command.side_effect = [
            lscpu_output,  # First lscpu call for model name
            lscpu_output,  # Second lscpu call for cores
            "12",  # nproc call for threads
        ]

        hardware = HardwareInfo()
        cpu_model, cpu_cores, cpu_threads = hardware.get_cpu_info()

        assert cpu_model == "Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz"
        assert cpu_cores == "6"
        assert cpu_threads == "12"

    @patch("hardware_utils.platform.system")
    @patch("hardware_utils.shutil.which")
    @patch("builtins.open", new_callable=mock_open, read_data="processor\t: 0\nmodel name\t: AMD Ryzen 5 3600\nprocessor\t: 1\n")
    def test_get_cpu_info_linux_fallback_cpuinfo(self, _mock_file, mock_which, mock_system):  # noqa: PT019
        """Test CPU info detection on Linux using /proc/cpuinfo fallback."""
        mock_system.return_value = "Linux"
        mock_which.return_value = None  # No commands available

        hardware = HardwareInfo()
        cpu_model, cpu_cores, cpu_threads = hardware.get_cpu_info()

        assert cpu_model == "AMD Ryzen 5 3600"
        assert cpu_cores == "Unknown"  # Can't determine cores without lscpu
        assert cpu_threads == "2"  # Count of processors

    @patch("hardware_utils.platform.system")
    @patch("hardware_utils.shutil.which")
    @patch.object(HardwareInfo, "_run_command")
    def test_get_cpu_info_windows(self, mock_run_command, mock_which, mock_system):
        """Test CPU info detection on Windows."""
        mock_system.return_value = "Windows"
        mock_which.side_effect = lambda cmd: cmd == "powershell"

        mock_run_command.side_effect = ["Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz", "8", "16"]

        hardware = HardwareInfo()
        cpu_model, cpu_cores, cpu_threads = hardware.get_cpu_info()

        assert cpu_model == "Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz"
        assert cpu_cores == "8"
        assert cpu_threads == "16"

    @patch("hardware_utils.platform.system")
    def test_get_cpu_info_unknown_os(self, mock_system):
        """Test CPU info detection on unknown OS."""
        mock_system.return_value = "UnknownOS"

        hardware = HardwareInfo()
        cpu_model, cpu_cores, cpu_threads = hardware.get_cpu_info()

        assert cpu_model == "Unknown"
        assert cpu_cores == "Unknown"
        assert cpu_threads == "Unknown"

    @patch("hardware_utils.platform.system")
    @patch.object(HardwareInfo, "_run_command")
    def test_get_cpu_info_command_failure(self, mock_run_command, mock_system):
        """Test CPU info detection when commands fail."""
        mock_system.return_value = "Darwin"
        mock_run_command.side_effect = subprocess.CalledProcessError(1, "cmd")

        hardware = HardwareInfo()
        cpu_model, cpu_cores, cpu_threads = hardware.get_cpu_info()

        assert cpu_model == "Unknown"
        assert cpu_cores == "Unknown"
        assert cpu_threads == "Unknown"

    @patch("hardware_utils.platform.system")
    @patch.object(HardwareInfo, "_run_command")
    def test_get_memory_info_darwin(self, mock_run_command, mock_system):
        """Test memory info detection on macOS."""
        mock_system.return_value = "Darwin"
        mock_run_command.return_value = "17179869184"  # 16 GB in bytes

        hardware = HardwareInfo()
        memory = hardware.get_memory_info()

        assert memory == "16.0 GB"

    @patch("hardware_utils.platform.system")
    @patch("builtins.open", new_callable=mock_open, read_data="MemTotal:       16384000 kB\n")
    def test_get_memory_info_linux(self, _mock_file, mock_system):  # noqa: PT019
        """Test memory info detection on Linux."""
        mock_system.return_value = "Linux"

        hardware = HardwareInfo()
        memory = hardware.get_memory_info()

        assert memory == "15.6 GB"  # 16384000 kB converted to GB

    @patch("hardware_utils.platform.system")
    @patch("hardware_utils.shutil.which")
    @patch.object(HardwareInfo, "_run_command")
    def test_get_memory_info_windows(self, mock_run_command, mock_which, mock_system):
        """Test memory info detection on Windows."""
        mock_system.return_value = "Windows"
        mock_which.side_effect = lambda cmd: cmd == "powershell"
        mock_run_command.return_value = "32.0 GB"

        hardware = HardwareInfo()
        memory = hardware.get_memory_info()

        assert memory == "32.0 GB"

    @patch("hardware_utils.platform.system")
    def test_get_memory_info_unknown_os(self, mock_system):
        """Test memory info detection on unknown OS."""
        mock_system.return_value = "UnknownOS"

        hardware = HardwareInfo()
        memory = hardware.get_memory_info()

        assert memory == "Unknown"

    @patch("hardware_utils.shutil.which")
    @patch.object(HardwareInfo, "_run_command")
    def test_get_rust_info_success(self, mock_run_command, mock_which, hardware):
        """Test Rust info detection when rustc is available."""
        mock_which.return_value = "/usr/bin/rustc"
        mock_run_command.side_effect = ["rustc 1.70.0 (90c541806 2023-05-31)", "rustc 1.70.0 (90c541806 2023-05-31)\nhost: x86_64-apple-darwin\n"]

        rust_version, rust_target = hardware.get_rust_info()

        assert rust_version == "rustc 1.70.0 (90c541806 2023-05-31)"
        assert rust_target == "x86_64-apple-darwin"

    @patch("hardware_utils.shutil.which")
    def test_get_rust_info_no_rustc(self, mock_which, hardware):
        """Test Rust info detection when rustc is not available."""
        mock_which.return_value = None

        rust_version, rust_target = hardware.get_rust_info()

        assert rust_version == "Unknown"
        assert rust_target == "Unknown"

    @patch("hardware_utils.shutil.which")
    @patch.object(HardwareInfo, "_run_command")
    def test_get_rust_info_command_failure(self, mock_run_command, mock_which, hardware):
        """Test Rust info detection when rustc commands fail."""
        mock_which.return_value = "/usr/bin/rustc"
        mock_run_command.side_effect = subprocess.CalledProcessError(1, "cmd")

        rust_version, rust_target = hardware.get_rust_info()

        assert rust_version == "Unknown"
        assert rust_target == "Unknown"

    def test_get_hardware_info(self, hardware):
        """Test comprehensive hardware info collection."""
        with (
            patch.object(hardware, "get_cpu_info") as mock_cpu,
            patch.object(hardware, "get_memory_info") as mock_memory,
            patch.object(hardware, "get_rust_info") as mock_rust,
        ):
            mock_cpu.return_value = ("Intel i7", "8", "16")
            mock_memory.return_value = "16.0 GB"
            mock_rust.return_value = ("rustc 1.70.0", "x86_64-unknown-linux-gnu")

            info = hardware.get_hardware_info()

            expected_keys = ["OS", "CPU", "CPU_CORES", "CPU_THREADS", "MEMORY", "RUST", "TARGET"]
            assert set(info.keys()) == set(expected_keys)
            assert info["CPU"] == "Intel i7"
            assert info["CPU_CORES"] == "8"
            assert info["CPU_THREADS"] == "16"
            assert info["MEMORY"] == "16.0 GB"
            assert info["RUST"] == "rustc 1.70.0"
            assert info["TARGET"] == "x86_64-unknown-linux-gnu"

    @pytest.mark.parametrize(
        ("system_name", "expected_os"), [("Darwin", "macOS"), ("Linux", "Linux"), ("Windows", "Windows"), ("FreeBSD", "Unknown (FreeBSD)")]
    )
    @patch("hardware_utils.platform.system")
    def test_get_hardware_info_os_mapping(self, mock_system, system_name, expected_os):
        """Test OS name mapping in hardware info."""
        mock_system.return_value = system_name
        hardware = HardwareInfo()

        with (
            patch.object(hardware, "get_cpu_info") as mock_cpu,
            patch.object(hardware, "get_memory_info") as mock_memory,
            patch.object(hardware, "get_rust_info") as mock_rust,
        ):
            mock_cpu.return_value = ("CPU", "4", "8")
            mock_memory.return_value = "8.0 GB"
            mock_rust.return_value = ("rustc 1.70.0", "target")

            info = hardware.get_hardware_info()
            assert info["OS"] == expected_os

    def test_format_hardware_info(self, hardware):
        """Test hardware info formatting."""
        test_info = {
            "OS": "macOS",
            "CPU": "Intel i7",
            "CPU_CORES": "8",
            "CPU_THREADS": "16",
            "MEMORY": "16.0 GB",
            "RUST": "rustc 1.70.0",
            "TARGET": "x86_64-apple-darwin",
        }

        formatted = hardware.format_hardware_info(test_info)

        assert "Hardware Information:" in formatted
        assert "OS: macOS" in formatted
        assert "CPU: Intel i7" in formatted
        assert "CPU Cores: 8" in formatted
        assert "CPU Threads: 16" in formatted
        assert "Memory: 16.0 GB" in formatted
        assert "Rust: rustc 1.70.0" in formatted
        assert "Target: x86_64-apple-darwin" in formatted

    def test_format_hardware_info_none(self, hardware):
        """Test hardware info formatting with None input."""
        with patch.object(hardware, "get_hardware_info") as mock_get_info:
            mock_get_info.return_value = {
                "OS": "Linux",
                "CPU": "AMD Ryzen",
                "CPU_CORES": "6",
                "CPU_THREADS": "12",
                "MEMORY": "32.0 GB",
                "RUST": "rustc 1.70.0",
                "TARGET": "x86_64-unknown-linux-gnu",
            }

            formatted = hardware.format_hardware_info(None)
            assert "OS: Linux" in formatted


class TestHardwareComparator:
    """Test cases for HardwareComparator class."""

    def test_parse_baseline_hardware_complete(self):
        """Test parsing complete baseline hardware info."""
        baseline_content = """Benchmark Results
Generated on: 2023-06-15 10:30:00

Hardware Information:
  OS: macOS
  CPU: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
  CPU Cores: 6
  CPU Threads: 12
  Memory: 16.0 GB
  Rust: rustc 1.70.0 (90c541806 2023-05-31)
  Target: x86_64-apple-darwin

Benchmark Results:
test_benchmark ... 1.234 ms
"""

        info = HardwareComparator.parse_baseline_hardware(baseline_content)

        expected = {
            "OS": "macOS",
            "CPU": "Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz",
            "CPU_CORES": "6",
            "CPU_THREADS": "12",
            "MEMORY": "16.0 GB",
            "RUST": "rustc 1.70.0 (90c541806 2023-05-31)",
            "TARGET": "x86_64-apple-darwin",
        }

        assert info == expected

    def test_parse_baseline_hardware_partial(self):
        """Test parsing partial baseline hardware info."""
        baseline_content = """Hardware Information:
  OS: Linux
  CPU: AMD Ryzen 5 3600
  Memory: 32.0 GB

Other content here...
"""

        info = HardwareComparator.parse_baseline_hardware(baseline_content)

        assert info["OS"] == "Linux"
        assert info["CPU"] == "AMD Ryzen 5 3600"
        assert info["MEMORY"] == "32.0 GB"
        assert info["CPU_CORES"] == "Unknown"  # Not specified
        assert info["RUST"] == "Unknown"  # Not specified

    def test_parse_baseline_hardware_empty(self):
        """Test parsing baseline with no hardware info."""
        baseline_content = "No hardware information found"

        info = HardwareComparator.parse_baseline_hardware(baseline_content)

        # Should return all "Unknown" values
        for value in info.values():
            assert value == "Unknown"

    def test_compare_hardware_identical(self):
        """Test hardware comparison with identical configurations."""
        current_info = {
            "OS": "macOS",
            "CPU": "Intel i7",
            "CPU_CORES": "8",
            "CPU_THREADS": "16",
            "MEMORY": "16.0 GB",
            "RUST": "rustc 1.70.0",
            "TARGET": "x86_64-apple-darwin",
        }

        baseline_info = current_info.copy()

        report, has_warnings = HardwareComparator.compare_hardware(current_info, baseline_info)

        assert not has_warnings
        assert "Hardware configurations are compatible" in report

    def test_compare_hardware_different_os(self):
        """Test hardware comparison with different OS."""
        current_info = {
            "OS": "Linux",
            "CPU": "Intel i7",
            "CPU_CORES": "8",
            "CPU_THREADS": "16",
            "MEMORY": "16.0 GB",
            "RUST": "rustc 1.70.0",
            "TARGET": "x86_64-unknown-linux-gnu",
        }

        baseline_info = current_info.copy()
        baseline_info["OS"] = "macOS"

        report, has_warnings = HardwareComparator.compare_hardware(current_info, baseline_info)

        assert has_warnings
        assert "OS differs: Linux vs macOS" in report

    def test_compare_hardware_different_cpu(self):
        """Test hardware comparison with different CPU."""
        current_info = {
            "OS": "Linux",
            "CPU": "AMD Ryzen 5 3600",
            "CPU_CORES": "6",
            "CPU_THREADS": "12",
            "MEMORY": "16.0 GB",
            "RUST": "rustc 1.70.0",
            "TARGET": "x86_64-unknown-linux-gnu",
        }

        baseline_info = current_info.copy()
        baseline_info["CPU"] = "Intel i7-8700K"

        report, has_warnings = HardwareComparator.compare_hardware(current_info, baseline_info)

        assert has_warnings
        assert "CPU differs:" in report
        assert "results may not be directly comparable" in report

    def test_compare_hardware_different_cores(self):
        """Test hardware comparison with different core counts."""
        current_info = {
            "OS": "Linux",
            "CPU": "Intel i7",
            "CPU_CORES": "8",
            "CPU_THREADS": "16",
            "MEMORY": "16.0 GB",
            "RUST": "rustc 1.70.0",
            "TARGET": "x86_64-unknown-linux-gnu",
        }

        baseline_info = current_info.copy()
        baseline_info["CPU_CORES"] = "6"

        report, has_warnings = HardwareComparator.compare_hardware(current_info, baseline_info)

        assert has_warnings
        assert "CPU core count differs: 8 vs 6 cores" in report

    def test_compare_hardware_memory_tolerance(self):
        """Test memory comparison with numeric (percentage-based) tolerance."""
        current_info = {
            "OS": "Linux",
            "CPU": "Intel i7",
            "CPU_CORES": "8",
            "CPU_THREADS": "16",
            "MEMORY": "16.0 GB",
            "RUST": "rustc 1.70.0",
            "TARGET": "x86_64-unknown-linux-gnu",
        }

        # Test within tolerance (~0.3% difference)
        baseline_info = current_info.copy()
        baseline_info["MEMORY"] = "15.95 GB"

        report, has_warnings = HardwareComparator.compare_hardware(current_info, baseline_info)
        assert not has_warnings

        # Test outside tolerance (>2% difference)
        baseline_info["MEMORY"] = "15.6 GB"

        report, has_warnings = HardwareComparator.compare_hardware(current_info, baseline_info)
        assert has_warnings
        assert "Memory differs:" in report

    def test_compare_hardware_unknown_baseline(self):
        """Test hardware comparison with unknown baseline values."""
        current_info = {
            "OS": "Linux",
            "CPU": "Intel i7",
            "CPU_CORES": "8",
            "CPU_THREADS": "16",
            "MEMORY": "16.0 GB",
            "RUST": "rustc 1.70.0",
            "TARGET": "x86_64-unknown-linux-gnu",
        }

        baseline_info = {
            "OS": "Unknown",
            "CPU": "Unknown",
            "CPU_CORES": "Unknown",
            "CPU_THREADS": "Unknown",
            "MEMORY": "Unknown",
            "RUST": "Unknown",
            "TARGET": "Unknown",
        }

        report, has_warnings = HardwareComparator.compare_hardware(current_info, baseline_info)

        # Should not report warnings when baseline is unknown
        assert not has_warnings
        assert "Hardware configurations are compatible" in report

    @pytest.mark.parametrize(
        ("memory_str", "expected"),
        [
            ("16.0 GB", 16.0),
            ("32.5 GB", 32.5),
            ("8GB", 8.0),
            ("64 GB", 64.0),
            ("16,5 GB", 16.5),  # Comma decimal separator
            ("1.0 TB", 1.0),  # Larger unit still parses numeric part
            ("Unknown", None),
            ("Invalid format", None),
            ("", None),
        ],
    )
    def test_extract_memory_value(self, memory_str, expected):
        """Test memory value extraction from strings."""
        result = HardwareComparator._extract_memory_value(memory_str)  # noqa: SLF001
        if expected is None:
            assert result is None
        else:
            assert result == pytest.approx(expected, abs=1e-9)


class TestHardwareUtilsIntegration:
    """Integration tests for hardware_utils functionality."""

    def test_real_hardware_info_structure(self):
        """Test that real hardware info returns expected structure."""
        hardware = HardwareInfo()
        info = hardware.get_hardware_info()

        expected_keys = {"OS", "CPU", "CPU_CORES", "CPU_THREADS", "MEMORY", "RUST", "TARGET"}
        assert set(info.keys()) == expected_keys

        # All values should be strings
        for key, value in info.items():
            assert isinstance(value, str), f"Key {key} should have string value"

    def test_cpu_info_returns_tuples(self):
        """Test that CPU info methods return proper tuple structure."""
        hardware = HardwareInfo()

        cpu_model, cpu_cores, cpu_threads = hardware.get_cpu_info()
        rust_version, rust_target = hardware.get_rust_info()

        assert isinstance(cpu_model, str)
        assert isinstance(cpu_cores, str)
        assert isinstance(cpu_threads, str)
        assert isinstance(rust_version, str)
        assert isinstance(rust_target, str)

    def test_memory_info_returns_string(self):
        """Test that memory info returns a string."""
        hardware = HardwareInfo()
        memory = hardware.get_memory_info()

        assert isinstance(memory, str)

    def test_formatted_output_structure(self):
        """Test that formatted output has expected structure."""
        hardware = HardwareInfo()
        formatted = hardware.format_hardware_info()

        assert isinstance(formatted, str)
        assert "Hardware Information:" in formatted

        # Should contain all expected fields
        expected_fields = ["OS:", "CPU:", "CPU Cores:", "CPU Threads:", "Memory:", "Rust:", "Target:"]
        for field in expected_fields:
            assert field in formatted
