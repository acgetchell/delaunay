#!/usr/bin/env python3
"""
hardware_utils.py - Cross-platform hardware information detection and comparison

This module provides functions to detect hardware information across different
operating systems for use in benchmark baseline generation and comparison.

Replaces the bash-based hardware_info.sh with more maintainable Python code.
"""

import argparse
import contextlib
import json
import logging
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from subprocess_utils import run_safe_command
else:
    try:
        # When executed as a script from scripts/
        from subprocess_utils import run_safe_command
    except ModuleNotFoundError:
        # When imported as a module (e.g., scripts.hardware_utils)
        from scripts.subprocess_utils import run_safe_command

# Configure a module-level logger
logger = logging.getLogger(__name__)


class HardwareInfo:
    """Cross-platform hardware information detection."""

    def __init__(self):
        self.os_type = platform.system()
        self.machine = platform.machine()

    def get_cpu_info(self) -> tuple[str, str, str]:
        """
        Get CPU information: model, cores, threads.

        Returns:
            Tuple of (cpu_model, cpu_cores, cpu_threads)
        """
        try:
            if self.os_type == "Darwin":
                return self._get_cpu_info_darwin()
            if self.os_type == "Linux":
                return self._get_cpu_info_linux()
            if self.os_type == "Windows":
                return self._get_cpu_info_windows()
        except (subprocess.CalledProcessError, OSError, ValueError) as e:
            logger.debug(
                "Failed to get CPU info for OS %s: %s (%s)",
                self.os_type,
                e,
                type(e).__name__,
            )

        return "Unknown", "Unknown", "Unknown"

    def _get_cpu_info_darwin(self) -> tuple[str, str, str]:
        """
        Get CPU information on macOS using sysctl.

        Returns:
            Tuple of (cpu_model, cpu_cores, cpu_threads)
        """
        cpu_model = self._run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
        cpu_cores = self._run_command(["sysctl", "-n", "hw.physicalcpu"])
        cpu_threads = self._run_command(["sysctl", "-n", "hw.logicalcpu"])
        return cpu_model, cpu_cores, cpu_threads

    def _get_cpu_info_linux(self) -> tuple[str, str, str]:
        """
        Get CPU information on Linux using various methods.

        Returns:
            Tuple of (cpu_model, cpu_cores, cpu_threads)
        """
        cpu_model = self._get_linux_cpu_model()
        cpu_cores = self._get_linux_cpu_cores()
        cpu_threads = self._get_linux_cpu_threads()
        return cpu_model, cpu_cores, cpu_threads

    def _get_linux_cpu_model(self) -> str:
        """
        Get CPU model name on Linux.

        Returns:
            CPU model name or "Unknown"
        """
        # Try lscpu first
        if shutil.which("lscpu"):
            try:
                lscpu_output = self._run_command(["lscpu"])
                for line in lscpu_output.split("\n"):
                    if "Model name:" in line or "Model:" in line:
                        return line.split(":", 1)[1].strip()
            except (subprocess.CalledProcessError, IndexError):
                pass

        # Fallback to /proc/cpuinfo
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith(("model name", "Processor")):
                        return line.split(":", 1)[1].strip()
        except (FileNotFoundError, PermissionError):
            pass

        return "Unknown"

    def _get_linux_cpu_cores(self) -> str:
        """
        Get CPU core count on Linux.

        Returns:
            CPU core count or "Unknown"
        """
        if not shutil.which("lscpu"):
            # Fallback: parse physical core count from /proc/cpuinfo
            try:
                physical_cores: set[tuple[str, str]] = set()
                with open("/proc/cpuinfo", encoding="utf-8") as f:
                    physical_id = core_id = None
                    for line in f:
                        if line.startswith("physical id"):
                            physical_id = line.split(":", 1)[1].strip()
                        elif line.startswith("core id"):
                            core_id = line.split(":", 1)[1].strip()
                        if physical_id is not None and core_id is not None:
                            physical_cores.add((physical_id, core_id))
                            physical_id = core_id = None
                if physical_cores:
                    return str(len(physical_cores))
            except (FileNotFoundError, PermissionError, ValueError):
                return "Unknown"
            return "Unknown"

        try:
            lscpu_output = self._run_command(["lscpu"])
            cores_per_socket = None
            sockets = None

            for line in lscpu_output.split("\n"):
                if "Core(s) per socket:" in line:
                    cores_per_socket = int(line.split(":")[1].strip())
                elif "Socket(s):" in line:
                    sockets = int(line.split(":")[1].strip())

            if cores_per_socket is not None and sockets is not None:
                return str(cores_per_socket * sockets)
        except (subprocess.CalledProcessError, ValueError, IndexError):
            pass

        return "Unknown"

    def _get_linux_cpu_threads(self) -> str:
        """
        Get CPU thread count on Linux.

        Returns:
            CPU thread count or "Unknown"
        """
        # Try nproc first
        if shutil.which("nproc"):
            with contextlib.suppress(subprocess.CalledProcessError):
                return self._run_command(["nproc"])

        # Try getconf
        if shutil.which("getconf"):
            with contextlib.suppress(subprocess.CalledProcessError):
                return self._run_command(["getconf", "_NPROCESSORS_ONLN"])

        # Fallback to /proc/cpuinfo
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                processor_count = sum(1 for line in f if line.startswith("processor"))
                return str(processor_count)
        except (FileNotFoundError, PermissionError):
            pass

        return "Unknown"

    def _get_cpu_info_windows(self) -> tuple[str, str, str]:
        """
        Get CPU information on Windows using PowerShell/WMI.

        Returns:
            Tuple of (cpu_model, cpu_cores, cpu_threads)
        """
        # Try pwsh first, then powershell
        if shutil.which("pwsh"):
            ps_cmd = "pwsh"
        elif shutil.which("powershell"):
            ps_cmd = "powershell"
        else:
            return "Unknown", "Unknown", "Unknown"

        try:
            cpu_model = self._get_windows_cpu_model(ps_cmd)
            cpu_cores = self._get_windows_cpu_cores(ps_cmd)
            cpu_threads = self._get_windows_cpu_threads(ps_cmd)
            return cpu_model, cpu_cores, cpu_threads
        except subprocess.CalledProcessError:
            return "Unknown", "Unknown", "Unknown"

    def _run_powershell_command(self, ps_cmd: str, command: str) -> str:
        """
        Run a PowerShell command and return output.

        Args:
            ps_cmd: PowerShell executable to use
            command: PowerShell command string to execute

        Returns:
            Command output
        """
        return self._run_command(
            [
                ps_cmd,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
            ],
        ).strip()

    def _get_windows_cpu_model(self, ps_cmd: str) -> str:
        """
        Get CPU model on Windows.

        Args:
            ps_cmd: PowerShell command to use

        Returns:
            CPU model name
        """
        return self._run_powershell_command(
            ps_cmd,
            "(Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1).Name",
        )

    def _get_windows_cpu_cores(self, ps_cmd: str) -> str:
        """
        Get CPU core count on Windows.

        Args:
            ps_cmd: PowerShell command to use

        Returns:
            CPU core count
        """
        return self._run_powershell_command(
            ps_cmd,
            "(Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1).NumberOfCores",
        )

    def _get_windows_cpu_threads(self, ps_cmd: str) -> str:
        """
        Get CPU thread count on Windows.

        Args:
            ps_cmd: PowerShell command to use

        Returns:
            CPU thread count
        """
        return self._run_powershell_command(
            ps_cmd,
            "(Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1).NumberOfLogicalProcessors",
        )

    def get_memory_info(self) -> str:
        """
        Get total system memory.

        Returns:
            Memory size as string (e.g., "16.0 GB")
        """
        memory = "Unknown"

        try:
            if self.os_type == "Darwin":
                # macOS - convert bytes to GB
                mem_bytes = int(self._run_command(["sysctl", "-n", "hw.memsize"]))
                memory_gb = mem_bytes / (1024**3)
                memory = f"{memory_gb:.1f} GB"

            elif self.os_type == "Linux":
                # Linux - extract from /proc/meminfo
                try:
                    with open("/proc/meminfo", encoding="utf-8") as f:
                        for line in f:
                            if line.startswith("MemTotal:"):
                                mem_kb = int(line.split()[1])
                                memory_gb = mem_kb / (1024 * 1024)
                                memory = f"{memory_gb:.1f} GB"
                                break
                except (FileNotFoundError, PermissionError, ValueError):
                    pass

            elif self.os_type == "Windows":
                # Windows - Use PowerShell
                ps_cmd = "pwsh" if shutil.which("pwsh") else "powershell"
                if shutil.which(ps_cmd):
                    try:
                        ps_mem_cmd = (
                            "try { "
                            "$mem_bytes = [math]::Round((Get-CimInstance -ClassName Win32_ComputerSystem).TotalPhysicalMemory); "
                            '$mem_gb = [math]::Round($mem_bytes / 1GB, 1); Write-Output "$mem_gb GB" '
                            '} catch { Write-Output "Unknown" }'
                        )
                        memory = self._run_powershell_command(ps_cmd, ps_mem_cmd)
                    except subprocess.CalledProcessError:
                        pass

        except (subprocess.CalledProcessError, OSError, ValueError, FileNotFoundError, PermissionError) as e:
            logger.debug("Failed to get memory info for OS %s: %s", self.os_type, e)

        return memory

    def get_rust_info(self, cwd: Path | None = None) -> tuple[str, str]:
        """Get Rust toolchain information.

        Args:
            cwd: Working directory for rustc commands. This matters when rustup selects a
                toolchain based on `rust-toolchain.toml` in the directory tree.

        Returns:
            Tuple of (rust_version, rust_target)
        """
        rust_version = "Unknown"
        rust_target = "Unknown"

        try:
            if shutil.which("rustc"):
                rust_version = self._run_command(["rustc", "--version"], cwd=cwd)

                # Get target architecture
                rustc_verbose = self._run_command(["rustc", "-vV"], cwd=cwd)
                for line in rustc_verbose.split("\n"):
                    if line.startswith("host:"):
                        rust_target = line.split(":", 1)[1].strip()
                        break
        except subprocess.CalledProcessError as e:
            logger.debug("rustc command failed: %s", e)
        except Exception as e:
            logger.debug("Failed to get Rust info: %s", e)

        return rust_version, rust_target

    def get_hardware_info(self, cwd: Path | None = None) -> dict[str, str]:
        """Get comprehensive hardware information.

        Args:
            cwd: Working directory used for rustc metadata collection.

        Returns:
            Dictionary with hardware information
        """
        # OS name mapping
        os_name_map = {"Darwin": "macOS", "Linux": "Linux", "Windows": "Windows"}

        os_name = os_name_map.get(self.os_type, f"Unknown ({self.os_type})")

        cpu_model, cpu_cores, cpu_threads = self.get_cpu_info()
        memory = self.get_memory_info()
        rust_version, rust_target = self.get_rust_info(cwd=cwd)

        return {
            "OS": os_name,
            "CPU": cpu_model,
            "CPU_CORES": cpu_cores,
            "CPU_THREADS": cpu_threads,
            "MEMORY": memory,
            "RUST": rust_version,
            "TARGET": rust_target,
        }

    def format_hardware_info(self, info: dict[str, str] | None = None, *, cwd: Path | None = None) -> str:
        """Format hardware information as a readable block.

        Args:
            info: Hardware info dict. If None, gets current info.
            cwd: Working directory used for rustc metadata collection.

        Returns:
            Formatted hardware information string
        """
        if info is None:
            info = self.get_hardware_info(cwd=cwd)

        return f"""Hardware Information:
  OS: {info["OS"]}
  CPU: {info["CPU"]}
  CPU Cores: {info["CPU_CORES"]}
  CPU Threads: {info["CPU_THREADS"]}
  Memory: {info["MEMORY"]}
  Rust: {info["RUST"]}
  Target: {info["TARGET"]}

"""

    def _run_command(self, cmd: list[str], cwd: Path | None = None) -> str:
        """Run a command and return its output using secure subprocess wrapper.

        Args:
            cmd: Command to run as list
            cwd: Working directory for the command

        Returns:
            Command output as string

        Raises:
            subprocess.CalledProcessError: If command fails
        """
        if not cmd:
            error_msg = "Command list cannot be empty"
            raise ValueError(error_msg)

        command_name = cmd[0]
        args = cmd[1:] if len(cmd) > 1 else []

        result = run_safe_command(command_name, args, cwd=cwd, capture_output=True, text=True, check=True)

        return result.stdout.strip()


class HardwareComparator:
    """Compare hardware configurations for baseline compatibility."""

    @staticmethod
    def parse_baseline_hardware(baseline_content: str) -> dict[str, str]:
        """
        Parse hardware information from baseline file content.

        Args:
            baseline_content: Content of baseline_results.txt file

        Returns:
            Dictionary with baseline hardware info
        """
        info = {
            "OS": "Unknown",
            "CPU": "Unknown",
            "CPU_CORES": "Unknown",
            "CPU_THREADS": "Unknown",
            "MEMORY": "Unknown",
            "RUST": "Unknown",
            "TARGET": "Unknown",
        }

        # Find hardware information block
        lines = baseline_content.split("\n")
        in_hardware_block = False

        for line in lines:
            line = line.rstrip()

            if line == "Hardware Information:":
                in_hardware_block = True
                continue

            if in_hardware_block:
                # Stop at empty line or next section
                if not line or not line.startswith("  "):
                    break

                # Parse line like "  OS: macOS"
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().upper().replace(" ", "_")
                    value = value.strip()

                    # Only accept known keys
                    if key in info:
                        info[key] = value

        return info

    @staticmethod
    def compare_hardware(current_info: dict[str, str], baseline_info: dict[str, str]) -> tuple[str, bool]:
        """
        Compare current hardware with baseline hardware.

        Args:
            current_info: Current hardware information
            baseline_info: Baseline hardware information

        Returns:
            Tuple of (comparison_report, has_warnings)
        """
        report_lines = [
            "Hardware Comparison:",
            "==================",
            "",
            "Current Environment:",
            f"  OS: {current_info['OS']}",
            f"  CPU: {current_info['CPU']}",
            f"  CPU Cores: {current_info['CPU_CORES']}",
            f"  CPU Threads: {current_info['CPU_THREADS']}",
            f"  Memory: {current_info['MEMORY']}",
            f"  Rust: {current_info['RUST']}",
            f"  Target: {current_info['TARGET']}",
            "",
            "Baseline Environment:",
            f"  OS: {baseline_info['OS']}",
            f"  CPU: {baseline_info['CPU']}",
            f"  CPU Cores: {baseline_info['CPU_CORES']}",
            f"  CPU Threads: {baseline_info['CPU_THREADS']}",
            f"  Memory: {baseline_info['MEMORY']}",
            f"  Rust: {baseline_info['RUST']}",
            f"  Target: {baseline_info['TARGET']}",
            "",
            "Hardware Compatibility:",
        ]

        warnings: list[str] = []

        HardwareComparator._warn_if_different(
            warnings,
            current=current_info["OS"],
            baseline=baseline_info["OS"],
            message=f"⚠️  OS differs: {current_info['OS']} vs {baseline_info['OS']}",
        )

        HardwareComparator._warn_if_different(
            warnings,
            current=current_info["CPU"],
            baseline=baseline_info["CPU"],
            message=(f"⚠️  CPU differs: '{current_info['CPU']}' vs '{baseline_info['CPU']}' — results may not be directly comparable"),
        )

        HardwareComparator._warn_if_different(
            warnings,
            current=current_info["CPU_CORES"],
            baseline=baseline_info["CPU_CORES"],
            message=(f"⚠️  CPU core count differs: {current_info['CPU_CORES']} vs {baseline_info['CPU_CORES']} cores"),
            skip_if_current_unknown=True,
        )

        HardwareComparator._warn_if_different(
            warnings,
            current=current_info["CPU_THREADS"],
            baseline=baseline_info["CPU_THREADS"],
            message=(f"⚠️  CPU thread count differs: {current_info['CPU_THREADS']} vs {baseline_info['CPU_THREADS']} threads"),
            skip_if_current_unknown=True,
        )

        memory_warning = HardwareComparator._memory_warning_message(
            current=current_info["MEMORY"],
            baseline=baseline_info["MEMORY"],
        )
        if memory_warning is not None:
            warnings.append(memory_warning)

        HardwareComparator._warn_if_different(
            warnings,
            current=current_info["RUST"],
            baseline=baseline_info["RUST"],
            message=(f"⚠️  Rust version differs: '{current_info['RUST']}' vs '{baseline_info['RUST']}' — performance may be affected"),
        )

        HardwareComparator._warn_if_different(
            warnings,
            current=current_info["TARGET"],
            baseline=baseline_info["TARGET"],
            message=f"⚠️  Target architecture differs: {current_info['TARGET']} vs {baseline_info['TARGET']}",
        )

        if warnings:
            report_lines.extend(warnings)
        else:
            report_lines.append("✅ Hardware configurations are compatible for comparison")

        report_lines.append("")

        return "\n".join(report_lines), bool(warnings)

    @staticmethod
    def _warn_if_different(
        warnings: list[str],
        *,
        current: str,
        baseline: str,
        message: str,
        skip_if_current_unknown: bool = False,
    ) -> None:
        if baseline == "Unknown":
            return
        if skip_if_current_unknown and current == "Unknown":
            return
        if current != baseline:
            warnings.append(message)

    @staticmethod
    def _memory_warning_message(*, current: str, baseline: str) -> str | None:
        warning: str | None = None

        if current != "Unknown" and baseline != "Unknown":
            current_mem_num = HardwareComparator._extract_memory_value(current)
            baseline_mem_num = HardwareComparator._extract_memory_value(baseline)

            if current_mem_num is not None and baseline_mem_num is not None:
                # Percentage-based tolerance is more robust across small and large memory systems.
                if baseline_mem_num > 0:
                    mem_diff_pct = abs(current_mem_num - baseline_mem_num) / baseline_mem_num * 100
                    if mem_diff_pct > 2.0:  # More than 2% difference
                        warning = f"⚠️  Memory differs: {current} vs {baseline}"
                elif abs(current_mem_num - baseline_mem_num) > 0.1:
                    # Fallback for edge case where baseline is 0 or negative
                    warning = f"⚠️  Memory differs: {current} vs {baseline}"
            elif current != baseline:
                warning = f"⚠️  Memory differs: {current} vs {baseline}"

        return warning

    @staticmethod
    def _extract_memory_value(memory_str: str) -> float | None:
        """Extract numeric memory value from string like '16.0 GB'."""
        try:
            # Replace comma decimal separator with dot and extract first number
            memory_clean = memory_str.replace(",", ".")
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)", memory_clean)
            if match:
                return float(match.group(1))
        except (ValueError, AttributeError):
            pass
        return None


def main():
    """Command-line interface for hardware utilities."""
    parser = argparse.ArgumentParser(description="Cross-platform hardware information detection and comparison")
    parser.add_argument("command", choices=["info", "kv", "compare"], help="Command to run")
    parser.add_argument("--baseline-file", help="Path to baseline file (required for 'compare' command)")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")

    args = parser.parse_args()

    hardware = HardwareInfo()

    if args.command == "info":
        if args.json:
            info = hardware.get_hardware_info()
            print(json.dumps(info, indent=2))
        else:
            formatted_info = hardware.format_hardware_info()
            print(formatted_info, end="")

    elif args.command == "kv":
        info = hardware.get_hardware_info()
        for key, value in info.items():
            print(f"{key}={value}")

    elif args.command == "compare":
        if not args.baseline_file:
            print("error: --baseline-file is required for 'compare'", file=sys.stderr)
            sys.exit(2)

        baseline_path = Path(args.baseline_file)
        if not baseline_path.exists():
            print(f"error: baseline file not found: {baseline_path}", file=sys.stderr)
            sys.exit(2)

        try:
            baseline_content = baseline_path.read_text(encoding="utf-8", errors="replace")
            current_info = hardware.get_hardware_info()
            baseline_info = HardwareComparator.parse_baseline_hardware(baseline_content)

            report, has_warnings = HardwareComparator.compare_hardware(current_info, baseline_info)
            print(report, end="")

            # Exit with warning code if there are hardware differences
            sys.exit(1 if has_warnings else 0)

        except (OSError, UnicodeDecodeError, ValueError) as exc:
            print(f"error: comparison failed: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
