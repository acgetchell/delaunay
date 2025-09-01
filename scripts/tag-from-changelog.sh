#!/bin/bash

# tag-from-changelog.sh
#
# Backwards compatibility wrapper for the new Python implementation.
# This script now delegates to changelog_utils.py tag command.
#
# Usage:
#   ./scripts/tag-from-changelog.sh v0.3.5
#   ./scripts/tag-from-changelog.sh v0.3.5 --force  # Force recreate existing tag

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

# Print usage information
usage() {
	echo "Usage: $0 <tag-version> [--force]"
	echo ""
	echo "Creates a git tag with changelog content as the tag message."
	echo ""
	echo "Arguments:"
	echo "  <tag-version>    The version tag to create (e.g., v0.3.5)"
	echo "  --force          Force recreate if tag already exists"
	echo ""
	echo "Examples:"
	echo "  $0 v0.3.5"
	echo "  $0 v0.3.5 --force"
	echo ""
	echo "The script extracts the changelog section matching the specified version"
	echo "from CHANGELOG.md and uses it as the tag message for GitHub release integration."
	echo ""
	echo "Note: This is now a wrapper around the Python implementation in changelog_utils.py"
}

# Parse command line arguments
if [[ $# -lt 1 ]] || [[ $# -gt 2 ]]; then
	usage
	exit 1
fi

if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
	usage
	exit 0
fi

# Check Python3 prerequisite
if ! command -v python3 >/dev/null 2>&1; then
	echo "Error: python3 is required for changelog processing" >&2
	exit 1
fi

# Delegate to Python implementation
exec python3 "$SCRIPT_DIR/changelog_utils.py" tag "$@"
