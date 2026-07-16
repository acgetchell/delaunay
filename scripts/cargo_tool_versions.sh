#!/usr/bin/env bash

# Shared exact-version detection for Cargo-installed development tools.
installed_cargo_tool_version() {
	local binary="$1"
	local cargo_subcommand="${2:-}"
	local semver_pattern='([0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?(\+[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?)'
	local version_output

	if ! command -v "$binary" >/dev/null 2>&1; then
		return 0
	fi
	if [[ -n "$cargo_subcommand" ]]; then
		version_output="$(cargo "$cargo_subcommand" --version 2>/dev/null || true)"
	else
		version_output="$("$binary" --version 2>/dev/null || true)"
	fi
	if [[ "$version_output" =~ $semver_pattern ]]; then
		printf '%s\n' "${BASH_REMATCH[1]}"
	fi
}

cargo_tool_has_exact_version() {
	local binary="$1"
	local expected_version="$2"
	local cargo_subcommand="${3:-}"
	local installed_version

	installed_version="$(installed_cargo_tool_version "$binary" "$cargo_subcommand")"
	[[ "$installed_version" == "$expected_version" ]]
}
