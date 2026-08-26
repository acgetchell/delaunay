#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/.." && pwd -P)"
resolver="$repo_root/.github/actions/setup-just/resolve-version.sh"
pinned_version="$(bash "$resolver" "$repo_root/justfile")"

installed_version=""
if command -v just >/dev/null 2>&1; then
	installed_version="$(just --version 2>/dev/null || true)"
fi

if [[ "$installed_version" == "just $pinned_version" ]]; then
	printf 'just %s is already installed.\n' "$pinned_version"
	exit 0
fi

if ! command -v cargo >/dev/null 2>&1; then
	echo "Cargo is required to install just $pinned_version. Install Rust via https://rustup.rs first." >&2
	exit 1
fi

printf 'Installing just %s...\n' "$pinned_version"
cargo install --locked --version "$pinned_version" just
