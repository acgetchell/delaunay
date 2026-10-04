#!/usr/bin/env bash
# Native font and compression dependencies remain consumer-owned.
# Source this before toolchain sync so Cargo inherits pkg-config discovery.

have() { command -v "$1" >/dev/null 2>&1; }

ensure_tectonic_build_dependencies() {
	echo "Ensuring native dependencies needed to install Tectonic..."

	local candidate homebrew_repository package platform sdk_version
	local -a missing_pkg_config_packages=()
	local -a required_pkg_config_packages=(freetype2 graphite2 icu-uc libpng zlib)

	platform="$(uname -s)"
	case "$platform" in
	MINGW* | MSYS* | CYGWIN*)
		if [[ "${TECTONIC_DEP_BACKEND:-}" != "vcpkg" || ! -d "${VCPKG_ROOT:-}/installed/${VCPKGRS_TRIPLET:-x64-windows-static-md}" ]]; then
			echo "❌ Configure matching vcpkg libraries and TECTONIC_DEP_BACKEND=vcpkg before setup."
			exit 1
		fi
		return
		;;
	esac
	if [ "$platform" != "Darwin" ]; then
		required_pkg_config_packages=(fontconfig freetype2 graphite2 icu-uc libpng openssl zlib)
	fi

	append_pkg_config_path() {
		local directory="$1"
		if [ -d "$directory" ] && [[ ":${PKG_CONFIG_PATH:-}:" != *":$directory:"* ]]; then
			export PKG_CONFIG_PATH="${PKG_CONFIG_PATH:+$PKG_CONFIG_PATH:}$directory"
		fi
	}

	if ! have pkg-config; then
		echo "❌ 'pkg-config' was not found. Install pkgconf or pkg-config before building Tectonic from Cargo."
		exit 1
	fi

	shopt -s nullglob
	for candidate in \
		/opt/homebrew/lib/pkgconfig \
		/opt/homebrew/share/pkgconfig \
		/opt/homebrew/opt/{fontconfig,freetype,graphite2,icu4c*,libpng}/lib/pkgconfig \
		/usr/local/lib/pkgconfig \
		/usr/local/share/pkgconfig \
		/usr/local/opt/{fontconfig,freetype,graphite2,icu4c*,libpng}/lib/pkgconfig; do
		append_pkg_config_path "$candidate"
	done
	shopt -u nullglob

	if have brew && have xcrun && sdk_version="$(xcrun --sdk macosx --show-sdk-version 2>/dev/null)"; then
		homebrew_repository="$(brew --repository)"
		append_pkg_config_path "$homebrew_repository/Library/Homebrew/os/mac/pkgconfig/${sdk_version%%.*}"
	fi

	for package in "${required_pkg_config_packages[@]}"; do
		if ! pkg-config --exists "$package"; then
			missing_pkg_config_packages+=("$package")
		fi
	done
	if ((${#missing_pkg_config_packages[@]})); then
		echo "❌ pkg-config could not resolve: ${missing_pkg_config_packages[*]}"
		echo "   Install the missing native development files, or add their metadata directories to PKG_CONFIG_PATH."
		exit 1
	fi
	echo "  ✓ pkg-config can resolve Tectonic's native dependencies"
	echo ""
}

ensure_tectonic_build_dependencies
