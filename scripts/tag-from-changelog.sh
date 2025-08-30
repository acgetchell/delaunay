#!/bin/bash

# tag-from-changelog.sh
#
# Creates or recreates a git tag with changelog content as the tag message.
# This enables GitHub releases to use the changelog content via --notes-from-tag.
#
# Usage:
#   ./scripts/tag-from-changelog.sh v0.3.5
#   ./scripts/tag-from-changelog.sh v0.3.5 --force  # Force recreate existing tag
#
# The script extracts the changelog section matching the specified version from CHANGELOG.md
# and uses it as the tag message, making it easy to create GitHub releases with proper content.

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory for Python utilities
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
}

# Extract changelog content for the specific tag version using Python utilities
extract_changelog() {
	local tag_version="$1"

	# Use Python utilities to extract changelog content
	python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from changelog_utils import ChangelogUtils, ChangelogError

try:
    # Validate git repo and find changelog
    ChangelogUtils.validate_git_repo()
    changelog_path = ChangelogUtils.find_changelog_path()
    
    # Parse version and extract content
    version = ChangelogUtils.parse_version('$tag_version')
    content = ChangelogUtils.extract_changelog_section(changelog_path, version)
    
    print(content)
except ChangelogError as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
"
}

# Main script logic
main() {
	local tag_version="$1"
	local force_recreate="${2:-}"

	# Validate tag format using Python utilities
	if ! python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from changelog_utils import ChangelogUtils, VersionError

try:
    ChangelogUtils.validate_semver('$tag_version')
except VersionError as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null; then
		echo -e "${RED}Error: Invalid SemVer format${NC}" >&2
		exit 1
	fi

	# Check if tag already exists
	if git rev-parse -q --verify "refs/tags/${tag_version}" >/dev/null; then
		if [[ "$force_recreate" != "--force" ]]; then
			echo -e "${YELLOW}Tag '$tag_version' already exists.${NC}" >&2
			echo "Use --force to recreate it, or delete it first with:" >&2
			echo "  git tag -d $tag_version" >&2
			exit 1
		else
			echo -e "${BLUE}Deleting existing tag '$tag_version'...${NC}"
			git tag -d "$tag_version"
		fi
	fi

	# Extract changelog content
	echo -e "${BLUE}Extracting changelog content...${NC}"
	local tag_message
	tag_message=$(extract_changelog "$tag_version")

	# Show preview of tag message
	echo -e "${BLUE}Tag message preview:${NC}"
	echo "----------------------------------------"
	echo "$tag_message"
	echo "----------------------------------------"

	# Create the tag
	if ! git config --get user.name >/dev/null 2>&1 || ! git config --get user.email >/dev/null 2>&1; then
		echo -e "${YELLOW}Warning: git user.name/email not configured; tag creation may fail.${NC}" >&2
	fi
	echo -e "${BLUE}Creating tag '$tag_version' with changelog content...${NC}"
	if echo "$tag_message" | git tag -a "$tag_version" -F -; then
		echo -e "${GREEN}✓ Successfully created tag '$tag_version'${NC}"
		echo ""
		echo "Next steps:"
		echo "  1. Push the tag: ${BLUE}git push origin $tag_version${NC}"
		echo "  2. Create GitHub release: ${BLUE}gh release create $tag_version --notes-from-tag${NC}"
	else
		echo -e "${RED}Error: Failed to create tag${NC}" >&2
		exit 1
	fi
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
	echo -e "${RED}Error: python3 is required for changelog processing${NC}" >&2
	exit 1
fi

# Run main logic
main "$@"
