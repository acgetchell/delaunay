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
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
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
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if [[ -f "/proc/cpuinfo" ]]; then
            cpu_info=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | sed 's/^ *//' 2>/dev/null || echo "Unknown")
            cpu_cores=$(grep "cpu cores" /proc/cpuinfo | head -1 | cut -d: -f2 | sed 's/^ *//' 2>/dev/null || echo "Unknown")
            cpu_threads=$(grep -c "processor" /proc/cpuinfo 2>/dev/null || echo "Unknown")
        fi
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        # Windows (via MSYS2/Cygwin)
        if command -v wmic >/dev/null 2>&1; then
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
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
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
        # Windows
        if command -v wmic >/dev/null 2>&1; then
            local mem_bytes
            mem_bytes=$(wmic computersystem get TotalPhysicalMemory /format:list 2>/dev/null | grep "TotalPhysicalMemory=" | cut -d= -f2 | head -1 || echo "0")
            if [[ "$mem_bytes" -gt 0 ]]; then
                memory_total=$(echo "scale=1; $mem_bytes / 1024 / 1024 / 1024" | bc -l 2>/dev/null || echo "Unknown")
                memory_total="${memory_total} GB"
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
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
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
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if [[ -f "/proc/cpuinfo" ]]; then
            cpu_info=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | sed 's/^ *//' 2>/dev/null || echo "Unknown")
            cpu_cores=$(grep "cpu cores" /proc/cpuinfo | head -1 | cut -d: -f2 | sed 's/^ *//' 2>/dev/null || echo "Unknown")
            cpu_threads=$(grep -c "processor" /proc/cpuinfo 2>/dev/null || echo "Unknown")
        fi
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        # Windows (via MSYS2/Cygwin)
        if command -v wmic >/dev/null 2>&1; then
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
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
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
        # Windows
        if command -v wmic >/dev/null 2>&1; then
            local mem_bytes
            mem_bytes=$(wmic computersystem get TotalPhysicalMemory /format:list 2>/dev/null | grep "TotalPhysicalMemory=" | cut -d= -f2 | head -1 || echo "0")
            if [[ "$mem_bytes" -gt 0 ]]; then
                memory_total=$(echo "scale=1; $mem_bytes / 1024 / 1024 / 1024" | bc -l 2>/dev/null || echo "Unknown")
                memory_total="${memory_total} GB"
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
        # Extract each piece of hardware info
        local baseline_os baseline_cpu baseline_cores baseline_threads baseline_memory baseline_rust baseline_target
        
        baseline_os=$(grep "  OS:" "$baseline_file" | cut -d: -f2- | sed 's/^ *//' || echo "Unknown")
        baseline_cpu=$(grep "  CPU:" "$baseline_file" | cut -d: -f2- | sed 's/^ *//' || echo "Unknown")
        baseline_cores=$(grep "  CPU Cores:" "$baseline_file" | cut -d: -f2- | sed 's/^ *//' || echo "Unknown")
        baseline_threads=$(grep "  CPU Threads:" "$baseline_file" | cut -d: -f2- | sed 's/^ *//' || echo "Unknown")
        baseline_memory=$(grep "  Memory:" "$baseline_file" | cut -d: -f2- | sed 's/^ *//' || echo "Unknown")
        baseline_rust=$(grep "  Rust:" "$baseline_file" | cut -d: -f2- | sed 's/^ *//' || echo "Unknown")
        baseline_target=$(grep "  Target:" "$baseline_file" | cut -d: -f2- | sed 's/^ *//' || echo "Unknown")
        
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
    done <<< "$current_hw"
    
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
    done <<< "$baseline_hw"
    
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
