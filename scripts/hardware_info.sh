#!/usr/bin/env bash
# hardware_info.sh - Cross-platform hardware information detection utility
#
# This script provides functions to detect hardware information across different
# operating systems for use in benchmark baseline generation and comparison.
#
# Usage:
#   source scripts/hardware_info.sh
#   hardware_info=$(get_hardware_info)
#   echo "$hardware_info"

# Function to detect hardware information cross-platform
get_hardware_info() {
	# Detect OS
	local os_name
	if [[ "$OSTYPE" == "darwin"* ]]; then
		os_name="macOS"
	elif [[ "$OSTYPE" == "linux"* ]]; then
		os_name="Linux"
	elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
		os_name="Windows"
	else
		os_name="Unknown ($OSTYPE)"
	fi

	# CPU Information
	local cpu_info="Unknown"
	local cpu_cores="Unknown"
	local cpu_threads="Unknown"

	if [[ "$OSTYPE" == "darwin"* ]]; then
		# macOS
		if command -v sysctl >/dev/null 2>&1; then
			cpu_info=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
			cpu_cores=$(sysctl -n hw.physicalcpu 2>/dev/null || echo "Unknown")
			cpu_threads=$(sysctl -n hw.logicalcpu 2>/dev/null || echo "Unknown")
		fi
	elif [[ "$OSTYPE" == "linux"* ]]; then
		# Linux
		if [[ -f "/proc/cpuinfo" ]]; then
			cpu_info=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | sed 's/^ *//' 2>/dev/null || echo "Unknown")
			cpu_cores=$(grep "cpu cores" /proc/cpuinfo | head -1 | cut -d: -f2 | sed 's/^ *//' 2>/dev/null || echo "Unknown")
			# Use nproc for thread count (more portable), fall back to /proc/cpuinfo parsing
			if command -v nproc >/dev/null 2>&1; then
				cpu_threads=$(nproc 2>/dev/null || echo "Unknown")
			else
				cpu_threads=$(grep -c "^processor[[:space:]]*:" /proc/cpuinfo 2>/dev/null || echo "Unknown")
			fi
		fi
	elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
		# Windows (via MSYS2/Cygwin) - Use PowerShell by default (wmic is deprecated)
		if command -v powershell >/dev/null 2>&1 || command -v pwsh >/dev/null 2>&1; then
			local ps_cmd
			if command -v pwsh >/dev/null 2>&1; then
				ps_cmd="pwsh"
			else
				ps_cmd="powershell"
			fi
			# Use Get-CimInstance instead of Get-WmiObject for better compatibility
			local temp_cpu_info temp_cpu_cores temp_cpu_threads
			temp_cpu_info=$($ps_cmd -NoProfile -NonInteractive -Command "(Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1).Name" 2>/dev/null | tr -d '\r')
			temp_cpu_cores=$($ps_cmd -NoProfile -NonInteractive -Command "(Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1).NumberOfCores" 2>/dev/null | tr -d '\r')
			temp_cpu_threads=$($ps_cmd -NoProfile -NonInteractive -Command "(Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1).NumberOfLogicalProcessors" 2>/dev/null | tr -d '\r')
			# Guard against empty outputs from CIM queries
			cpu_info=${temp_cpu_info:-"Unknown"}
			cpu_cores=${temp_cpu_cores:-"Unknown"}
			cpu_threads=${temp_cpu_threads:-"Unknown"}
		elif command -v wmic >/dev/null 2>&1; then
			# Legacy fallback to wmic (deprecated)
			cpu_info=$(wmic cpu get name /format:list 2>/dev/null | grep "Name=" | cut -d= -f2 | head -1 || echo "Unknown")
			cpu_cores=$(wmic cpu get NumberOfCores /format:list 2>/dev/null | grep "NumberOfCores=" | cut -d= -f2 | head -1 || echo "Unknown")
			cpu_threads=$(wmic cpu get NumberOfLogicalProcessors /format:list 2>/dev/null | grep "NumberOfLogicalProcessors=" | cut -d= -f2 | head -1 || echo "Unknown")
		fi
	fi

	# Memory Information
	local memory_total="Unknown"

	if [[ "$OSTYPE" == "darwin"* ]]; then
		# macOS - convert bytes to GB
		if command -v sysctl >/dev/null 2>&1; then
			local mem_bytes
			mem_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo "0")
			if [[ "$mem_bytes" -gt 0 ]]; then
				memory_total=$(echo "scale=1; $mem_bytes / 1024 / 1024 / 1024" | bc -l 2>/dev/null || echo "Unknown")
				memory_total="${memory_total} GB"
			fi
		fi
	elif [[ "$OSTYPE" == "linux"* ]]; then
		# Linux - extract from /proc/meminfo
		if [[ -f "/proc/meminfo" ]]; then
			local mem_kb
			mem_kb=$(grep "MemTotal:" /proc/meminfo | awk '{print $2}' 2>/dev/null || echo "0")
			if [[ "$mem_kb" -gt 0 ]]; then
				memory_total=$(echo "scale=1; $mem_kb / 1024 / 1024" | bc -l 2>/dev/null || echo "Unknown")
				memory_total="${memory_total} GB"
			fi
		fi
	elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
		# Windows - Use PowerShell by default (wmic is deprecated)
		if command -v powershell >/dev/null 2>&1 || command -v pwsh >/dev/null 2>&1; then
			local ps_cmd
			if command -v pwsh >/dev/null 2>&1; then
				ps_cmd="pwsh"
			else
				ps_cmd="powershell"
			fi
			# Use Get-CimInstance and perform GB conversion in PowerShell to avoid bc dependency
			memory_total=$($ps_cmd -NonInteractive -Command "
                try {
                    \$mem_bytes = (Get-CimInstance -ClassName Win32_ComputerSystem).TotalPhysicalMemory
                    \$mem_gb = [math]::Round(\$mem_bytes / 1GB, 1)
                    Write-Output \"\$mem_gb GB\"
                } catch {
                    Write-Output \"Unknown\"
                }
            " 2>/dev/null | tr -d '\r' || echo "Unknown")
		elif command -v wmic >/dev/null 2>&1; then
			# Legacy fallback to wmic (deprecated) - use PowerShell for conversion
			if command -v powershell >/dev/null 2>&1 || command -v pwsh >/dev/null 2>&1; then
				local ps_cmd_fallback
				if command -v pwsh >/dev/null 2>&1; then
					ps_cmd_fallback="pwsh"
				else
					ps_cmd_fallback="powershell"
				fi
				local mem_bytes
				mem_bytes=$(wmic computersystem get TotalPhysicalMemory /format:list 2>/dev/null | grep "TotalPhysicalMemory=" | cut -d= -f2 | head -1 || echo "0")
				if [[ "$mem_bytes" -gt 0 ]]; then
					memory_total=$($ps_cmd_fallback -NonInteractive -Command "
                        try {
                            \$mem_gb = [math]::Round($mem_bytes / 1GB, 1)
                            Write-Output \"\$mem_gb GB\"
                        } catch {
                            Write-Output \"Unknown\"
                        }
                    " 2>/dev/null | tr -d '\r' || echo "Unknown")
				fi
			else
				# Pure bash fallback if no PowerShell available (very rare)
				local mem_bytes
				mem_bytes=$(wmic computersystem get TotalPhysicalMemory /format:list 2>/dev/null | grep "TotalPhysicalMemory=" | cut -d= -f2 | head -1 || echo "0")
				if [[ "$mem_bytes" -gt 0 ]]; then
					# Basic integer division fallback (less precise but avoids bc dependency)
					memory_total=$((mem_bytes / 1073741824))
					memory_total="${memory_total} GB"
				fi
			fi
		fi
	fi

	# Rust toolchain information
	local rust_version="Unknown"
	local rust_target="Unknown"

	if command -v rustc >/dev/null 2>&1; then
		rust_version=$(rustc --version 2>/dev/null || echo "Unknown")
		rust_target=$(rustc -vV 2>/dev/null | grep "host:" | cut -d: -f2 | sed 's/^ *//' || echo "Unknown")
	fi

	# Output hardware information in a formatted block
	cat <<EOF
Hardware Information:
  OS: $os_name
  CPU: $cpu_info
  CPU Cores: $cpu_cores
  CPU Threads: $cpu_threads
  Memory: $memory_total
  Rust: $rust_version
  Target: $rust_target

EOF
}

# Function to get hardware info as key=value pairs (useful for parsing/comparison)
get_hardware_info_kv() {
	# Detect OS
	local os_name
	if [[ "$OSTYPE" == "darwin"* ]]; then
		os_name="macOS"
	elif [[ "$OSTYPE" == "linux"* ]]; then
		os_name="Linux"
	elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
		os_name="Windows"
	else
		os_name="Unknown ($OSTYPE)"
	fi

	# CPU Information
	local cpu_info="Unknown"
	local cpu_cores="Unknown"
	local cpu_threads="Unknown"

	if [[ "$OSTYPE" == "darwin"* ]]; then
		# macOS
		if command -v sysctl >/dev/null 2>&1; then
			cpu_info=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
			cpu_cores=$(sysctl -n hw.physicalcpu 2>/dev/null || echo "Unknown")
			cpu_threads=$(sysctl -n hw.logicalcpu 2>/dev/null || echo "Unknown")
		fi
	elif [[ "$OSTYPE" == "linux"* ]]; then
		# Linux
		if [[ -f "/proc/cpuinfo" ]]; then
			cpu_info=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | sed 's/^ *//' 2>/dev/null || echo "Unknown")
			cpu_cores=$(grep "cpu cores" /proc/cpuinfo | head -1 | cut -d: -f2 | sed 's/^ *//' 2>/dev/null || echo "Unknown")
			# Use nproc for thread count (more portable), fall back to /proc/cpuinfo parsing
			if command -v nproc >/dev/null 2>&1; then
				cpu_threads=$(nproc 2>/dev/null || echo "Unknown")
			else
				cpu_threads=$(grep -c "^processor[[:space:]]*:" /proc/cpuinfo 2>/dev/null || echo "Unknown")
			fi
		fi
	elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
		# Windows (via MSYS2/Cygwin) - Use PowerShell by default (wmic is deprecated)
		if command -v powershell >/dev/null 2>&1 || command -v pwsh >/dev/null 2>&1; then
			local ps_cmd
			if command -v pwsh >/dev/null 2>&1; then
				ps_cmd="pwsh"
			else
				ps_cmd="powershell"
			fi
			# Use Get-CimInstance instead of Get-WmiObject for better compatibility
			local temp_cpu_info temp_cpu_cores temp_cpu_threads
			temp_cpu_info=$($ps_cmd -NoProfile -NonInteractive -Command "(Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1).Name" 2>/dev/null | tr -d '\r')
			temp_cpu_cores=$($ps_cmd -NoProfile -NonInteractive -Command "(Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1).NumberOfCores" 2>/dev/null | tr -d '\r')
			temp_cpu_threads=$($ps_cmd -NoProfile -NonInteractive -Command "(Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1).NumberOfLogicalProcessors" 2>/dev/null | tr -d '\r')
			# Guard against empty outputs from CIM queries
			cpu_info=${temp_cpu_info:-"Unknown"}
			cpu_cores=${temp_cpu_cores:-"Unknown"}
			cpu_threads=${temp_cpu_threads:-"Unknown"}
		elif command -v wmic >/dev/null 2>&1; then
			# Legacy fallback to wmic (deprecated)
			cpu_info=$(wmic cpu get name /format:list 2>/dev/null | grep "Name=" | cut -d= -f2 | head -1 || echo "Unknown")
			cpu_cores=$(wmic cpu get NumberOfCores /format:list 2>/dev/null | grep "NumberOfCores=" | cut -d= -f2 | head -1 || echo "Unknown")
			cpu_threads=$(wmic cpu get NumberOfLogicalProcessors /format:list 2>/dev/null | grep "NumberOfLogicalProcessors=" | cut -d= -f2 | head -1 || echo "Unknown")
		fi
	fi

	# Memory Information
	local memory_total="Unknown"

	if [[ "$OSTYPE" == "darwin"* ]]; then
		# macOS - convert bytes to GB
		if command -v sysctl >/dev/null 2>&1; then
			local mem_bytes
			mem_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo "0")
			if [[ "$mem_bytes" -gt 0 ]]; then
				memory_total=$(echo "scale=1; $mem_bytes / 1024 / 1024 / 1024" | bc -l 2>/dev/null || echo "Unknown")
				memory_total="${memory_total} GB"
			fi
		fi
	elif [[ "$OSTYPE" == "linux"* ]]; then
		# Linux - extract from /proc/meminfo
		if [[ -f "/proc/meminfo" ]]; then
			local mem_kb
			mem_kb=$(grep "MemTotal:" /proc/meminfo | awk '{print $2}' 2>/dev/null || echo "0")
			if [[ "$mem_kb" -gt 0 ]]; then
				memory_total=$(echo "scale=1; $mem_kb / 1024 / 1024" | bc -l 2>/dev/null || echo "Unknown")
				memory_total="${memory_total} GB"
			fi
		fi
	elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
		# Windows - Use PowerShell by default (wmic is deprecated)
		if command -v powershell >/dev/null 2>&1 || command -v pwsh >/dev/null 2>&1; then
			local ps_cmd
			if command -v pwsh >/dev/null 2>&1; then
				ps_cmd="pwsh"
			else
				ps_cmd="powershell"
			fi
			# Use Get-CimInstance and perform GB conversion in PowerShell to avoid bc dependency
			memory_total=$($ps_cmd -NonInteractive -Command "
                try {
                    \$mem_bytes = (Get-CimInstance -ClassName Win32_ComputerSystem).TotalPhysicalMemory
                    \$mem_gb = [math]::Round(\$mem_bytes / 1GB, 1)
                    Write-Output \"\$mem_gb GB\"
                } catch {
                    Write-Output \"Unknown\"
                }
            " 2>/dev/null | tr -d '\r' || echo "Unknown")
		elif command -v wmic >/dev/null 2>&1; then
			# Legacy fallback to wmic (deprecated) - use PowerShell for conversion
			if command -v powershell >/dev/null 2>&1 || command -v pwsh >/dev/null 2>&1; then
				local ps_cmd_fallback
				if command -v pwsh >/dev/null 2>&1; then
					ps_cmd_fallback="pwsh"
				else
					ps_cmd_fallback="powershell"
				fi
				local mem_bytes
				mem_bytes=$(wmic computersystem get TotalPhysicalMemory /format:list 2>/dev/null | grep "TotalPhysicalMemory=" | cut -d= -f2 | head -1 || echo "0")
				if [[ "$mem_bytes" -gt 0 ]]; then
					memory_total=$($ps_cmd_fallback -NonInteractive -Command "
                        try {
                            \$mem_gb = [math]::Round($mem_bytes / 1GB, 1)
                            Write-Output \"\$mem_gb GB\"
                        } catch {
                            Write-Output \"Unknown\"
                        }
                    " 2>/dev/null | tr -d '\r' || echo "Unknown")
				fi
			else
				# Pure bash fallback if no PowerShell available (very rare)
				local mem_bytes
				mem_bytes=$(wmic computersystem get TotalPhysicalMemory /format:list 2>/dev/null | grep "TotalPhysicalMemory=" | cut -d= -f2 | head -1 || echo "0")
				if [[ "$mem_bytes" -gt 0 ]]; then
					# Basic integer division fallback (less precise but avoids bc dependency)
					memory_total=$((mem_bytes / 1073741824))
					memory_total="${memory_total} GB"
				fi
			fi
		fi
	fi

	# Rust toolchain information
	local rust_version="Unknown"
	local rust_target="Unknown"

	if command -v rustc >/dev/null 2>&1; then
		rust_version=$(rustc --version 2>/dev/null || echo "Unknown")
		rust_target=$(rustc -vV 2>/dev/null | grep "host:" | cut -d: -f2 | sed 's/^ *//' || echo "Unknown")
	fi

	# Return as key=value pairs
	echo "OS=$os_name"
	echo "CPU=$cpu_info"
	echo "CPU_CORES=$cpu_cores"
	echo "CPU_THREADS=$cpu_threads"
	echo "MEMORY=$memory_total"
	echo "RUST=$rust_version"
	echo "TARGET=$rust_target"
}

# Function to extract hardware info from baseline file
extract_baseline_hardware() {
	local baseline_file="$1"

	# Check if baseline has hardware information
	if grep -q "Hardware Information:" "$baseline_file"; then
		# Extract hardware information block only (from "Hardware Information:" until next empty line or EOF)
		local hardware_block
		hardware_block=$(awk '/^Hardware Information:/{flag=1; next} flag && /^$/{exit} flag' "$baseline_file")

		# If hardware_block is empty, try alternative extraction (handle cases where block doesn't end with empty line)
		# Only capture indented lines (starting with 2+ spaces) to avoid overrunning past the hardware section
		if [[ -z "$hardware_block" ]]; then
			hardware_block=$(awk '/^Hardware Information:/{flag=1; next} flag && /^  / {print; next} flag {exit}' "$baseline_file")
		fi

		# Extract each piece of hardware info from the scoped block
		local baseline_os baseline_cpu baseline_cores baseline_threads baseline_memory baseline_rust baseline_target

		baseline_os=$(echo "$hardware_block" | grep "  OS:" | cut -d: -f2- | sed 's/^ *//' || echo "Unknown")
		baseline_cpu=$(echo "$hardware_block" | grep "  CPU:" | cut -d: -f2- | sed 's/^ *//' || echo "Unknown")
		baseline_cores=$(echo "$hardware_block" | grep "  CPU Cores:" | cut -d: -f2- | sed 's/^ *//' || echo "Unknown")
		baseline_threads=$(echo "$hardware_block" | grep "  CPU Threads:" | cut -d: -f2- | sed 's/^ *//' || echo "Unknown")
		baseline_memory=$(echo "$hardware_block" | grep "  Memory:" | cut -d: -f2- | sed 's/^ *//' || echo "Unknown")
		baseline_rust=$(echo "$hardware_block" | grep "  Rust:" | cut -d: -f2- | sed 's/^ *//' || echo "Unknown")
		baseline_target=$(echo "$hardware_block" | grep "  Target:" | cut -d: -f2- | sed 's/^ *//' || echo "Unknown")

		# Return as key=value pairs
		echo "OS=$baseline_os"
		echo "CPU=$baseline_cpu"
		echo "CPU_CORES=$baseline_cores"
		echo "CPU_THREADS=$baseline_threads"
		echo "MEMORY=$baseline_memory"
		echo "RUST=$baseline_rust"
		echo "TARGET=$baseline_target"
	else
		# No hardware info in baseline
		echo "OS=Unknown"
		echo "CPU=Unknown"
		echo "CPU_CORES=Unknown"
		echo "CPU_THREADS=Unknown"
		echo "MEMORY=Unknown"
		echo "RUST=Unknown"
		echo "TARGET=Unknown"
	fi
}

# Function to compare hardware configurations
compare_hardware() {
	local current_hw="$1"
	local baseline_hw="$2"

	# Parse current hardware
	local current_os current_cpu current_cores current_threads current_memory current_rust current_target
	while IFS='=' read -r key value; do
		case "$key" in
		"OS") current_os="$value" ;;
		"CPU") current_cpu="$value" ;;
		"CPU_CORES") current_cores="$value" ;;
		"CPU_THREADS") current_threads="$value" ;;
		"MEMORY") current_memory="$value" ;;
		"RUST") current_rust="$value" ;;
		"TARGET") current_target="$value" ;;
		esac
	done <<<"$current_hw"

	# Parse baseline hardware
	local baseline_os baseline_cpu baseline_cores baseline_threads baseline_memory baseline_rust baseline_target
	while IFS='=' read -r key value; do
		case "$key" in
		"OS") baseline_os="$value" ;;
		"CPU") baseline_cpu="$value" ;;
		"CPU_CORES") baseline_cores="$value" ;;
		"CPU_THREADS") baseline_threads="$value" ;;
		"MEMORY") baseline_memory="$value" ;;
		"RUST") baseline_rust="$value" ;;
		"TARGET") baseline_target="$value" ;;
		esac
	done <<<"$baseline_hw"

	# Generate comparison output
	cat <<EOF
Hardware Comparison:
==================

Current Environment:
  OS: $current_os
  CPU: $current_cpu
  CPU Cores: $current_cores
  CPU Threads: $current_threads
  Memory: $current_memory
  Rust: $current_rust
  Target: $current_target

Baseline Environment:
  OS: $baseline_os
  CPU: $baseline_cpu
  CPU Cores: $baseline_cores
  CPU Threads: $baseline_threads
  Memory: $baseline_memory
  Rust: $baseline_rust
  Target: $baseline_target

Hardware Compatibility:
EOF

	# Check for significant differences and add warnings
	local warnings_found=false

	if [[ "$current_os" != "$baseline_os" ]] && [[ "$baseline_os" != "Unknown" ]]; then
		echo "⚠️  OS differs: $current_os vs $baseline_os"
		warnings_found=true
	fi

	if [[ "$current_cpu" != "$baseline_cpu" ]] && [[ "$baseline_cpu" != "Unknown" ]]; then
		echo "⚠️  CPU differs: Results may not be directly comparable"
		warnings_found=true
	fi

	if [[ "$current_cores" != "$baseline_cores" ]] && [[ "$baseline_cores" != "Unknown" ]] && [[ "$current_cores" != "Unknown" ]]; then
		echo "⚠️  CPU core count differs: $current_cores vs $baseline_cores cores"
		warnings_found=true
	fi

	if [[ "$current_threads" != "$baseline_threads" ]] && [[ "$baseline_threads" != "Unknown" ]] && [[ "$current_threads" != "Unknown" ]]; then
		echo "⚠️  CPU thread count differs: $current_threads vs $baseline_threads threads"
		warnings_found=true
	fi

	if [[ "$current_memory" != "$baseline_memory" ]] && [[ "$baseline_memory" != "Unknown" ]] && [[ "$current_memory" != "Unknown" ]]; then
		echo "⚠️  Memory differs: $current_memory vs $baseline_memory"
		warnings_found=true
	fi

	if [[ "$current_rust" != "$baseline_rust" ]] && [[ "$baseline_rust" != "Unknown" ]]; then
		echo "⚠️  Rust version differs: Performance may be affected by compiler changes"
		warnings_found=true
	fi

	if [[ "$current_target" != "$baseline_target" ]] && [[ "$baseline_target" != "Unknown" ]]; then
		echo "⚠️  Target architecture differs: $current_target vs $baseline_target"
		warnings_found=true
	fi

	if [[ "$warnings_found" == "false" ]]; then
		echo "✅ Hardware configurations are compatible for comparison"
	fi

	echo ""
}

# Command-line interface for interactive testing
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
	# Script is being executed directly, not sourced
	case "${1:-}" in
	"--kv" | "-k")
		echo "Hardware Information (Key=Value format):"
		echo "=========================================="
		get_hardware_info_kv
		;;
	"--help" | "-h")
		cat <<EOF
Usage: $0 [OPTIONS]

Cross-platform hardware information detection utility for delaunay benchmarking.

OPTIONS:
    (no args)    Display hardware information in formatted block
    --kv, -k     Display hardware information as key=value pairs
    --help, -h   Show this help message

EXAMPLES:
    $0                    # Show formatted hardware info
    $0 --kv               # Show key=value format
    source $0             # Source functions for use in other scripts

NOTE:
    This script can also be sourced to make its functions available:
    - get_hardware_info()           # Returns formatted hardware block
    - get_hardware_info_kv()        # Returns key=value pairs
    - extract_baseline_hardware()   # Extracts hardware from baseline files
    - compare_hardware()            # Compares two hardware configurations
EOF
		;;
	*)
		echo "Current Hardware Information:"
		echo "============================"
		get_hardware_info
		;;
	esac
fi
