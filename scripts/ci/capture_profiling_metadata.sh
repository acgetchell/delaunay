#!/usr/bin/env bash
set -euo pipefail

mkdir -p profiling-results

declared_toolchain="$(
	grep -E '^[[:space:]]*channel[[:space:]]*=' rust-toolchain.toml 2>/dev/null |
		head -n 1 |
		cut -d '=' -f 2 |
		tr -d ' "' ||
		true
)"
rust_version="$(
	grep -E '^rust-version[[:space:]]*=' Cargo.toml 2>/dev/null |
		head -n 1 |
		cut -d '=' -f 2 |
		tr -d ' "' ||
		true
)"

profiling_mode="${PROFILE_METADATA_MODE:-}"
if [[ -z "$profiling_mode" ]]; then
	profiling_mode="production"
	if [[ "${PROFILING_DEV_MODE:-}" == "1" ]]; then
		profiling_mode="development"
	fi
fi

benchmark_filter="${BENCH_FILTER_VALUE:-${PROFILE_METADATA_FILTER:-All benchmarks}}"
metadata_title="${PROFILE_METADATA_TITLE:-Profiling Environment}"

{
	echo "# ${metadata_title}"
	echo
	echo "- Code ref: ${GITHUB_REF_NAME:-local}"
	echo "- Commit: $(git rev-parse HEAD)"
	echo "- Declared rust-toolchain.toml channel: ${declared_toolchain}"
	echo "- Cargo.toml rust-version: ${rust_version}"
	echo "- rustc: $(rustc --version)"
	echo "- cargo: $(cargo --version)"
	echo "- Cargo profile: perf"
	echo "- Benchmark filter: ${benchmark_filter}"
	echo "- Profiling mode: ${profiling_mode}"
	echo "- Runner: ${RUNNER_OS:-$(uname -s)}"
} >profiling-results/environment_metadata.md
