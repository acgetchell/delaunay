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

# Check if we're in a git repository
check_git_repo() {
	if ! git rev-parse --git-dir >/dev/null 2>&1; then
		echo -e "${RED}Error: Not in a git repository${NC}" >&2
		exit 1
	fi
}

# Check if CHANGELOG.md exists
check_changelog() {
	# Try to find CHANGELOG.md in current directory or parent directory
	local changelog_path
	if [[ -f "CHANGELOG.md" ]]; then
		changelog_path="CHANGELOG.md"
	elif [[ -f "../CHANGELOG.md" ]]; then
		changelog_path="../CHANGELOG.md"
	else
		echo -e "${RED}Error: CHANGELOG.md not found${NC}" >&2
		echo "Please run this script from the project root or scripts/ directory." >&2
		exit 1
	fi

	# Set the changelog path for use by other functions
	CHANGELOG_PATH="$changelog_path"
}

# Extract changelog content for the specific tag version
extract_changelog() {
	local tag_version="$1"
	local version_number
	local changelog_content

	# Strip 'v' prefix if present to get version number
	version_number="${tag_version#v}"

	# Use awk to find the specific version section and extract until next ## header
	# Support formats: "## [0.3.5] ...", "## v0.3.5 ...", "## 0.3.5 ...", "## [0.3.5] - 2025-08-28" (Keep a Changelog)
	# Also supports hyperlink syntax: "## [0.3.5](https://github.com/...)"
	changelog_content=$(awk -v version="$version_number" '
        BEGIN { 
            found = 0; printing = 0;
			# Escape dots in version number for regex
			gsub(/\./, "\\.", version);
		}
        /^##[[:space:]]/ {
            if (printing) {
                # Stop printing when we hit the next ## header
                exit
            }
            # Check if this header matches our version
            # Match: ## [vX.Y.Z] or ## [X.Y.Z] or ## vX.Y.Z or ## X.Y.Z
            if ($0 ~ "^##[[:space:]]*\\[?v?" version "\\]?($|[[:space:]]|\\()") {
                found = 1
                printing = 1
                next  # Skip the header itself
            }
        }
        printing { print }
        END { 
            if (!found) {
                print "VERSION_NOT_FOUND" > "/dev/stderr"
            }
        }
    ' "$CHANGELOG_PATH")

	# Check if version was found
	if [[ -z "$changelog_content" ]]; then
		echo -e "${YELLOW}Warning: No changelog content found for version $tag_version${NC}" >&2
		echo "Searched for version patterns:" >&2
		echo "  - ## [$version_number] - <date> ..." >&2
		echo "  - ## v$version_number ..." >&2
		echo "  - ## $version_number ..." >&2
		echo "Using minimal tag message instead." >&2
		echo "$tag_version"
	else
		# Preserve per-line indentation; only drop leading/trailing empty lines
		local cleaned
		cleaned=$(printf "%s\n" "$changelog_content" | awk '
			BEGIN{n=0}
			{ lines[n++]=$0 }
			END{
				# find first non-empty
				s=0; while (s<n && lines[s] ~ /^[[:space:]]*$/) s++;
				# find last non-empty
				e=n-1; while (e>=s && lines[e] ~ /^[[:space:]]*$/) e--;
				if (e<s) exit 0;
				for (i=s; i<=e; i++) print lines[i];
			}')
		if [[ -z "${cleaned//[$'\n'[:space:]]/}" ]]; then
			echo -e "${YELLOW}Warning: Changelog section is empty after trimming; using minimal tag message.${NC}" >&2
			echo "$tag_version"
		else
			printf "%s\n" "$cleaned"
		fi
	fi
}

# Main script logic
main() {
	local tag_version="$1"
	local force_recreate="${2:-}"

	# Validate tag format (SemVer: vMAJOR.MINOR.PATCH with optional -PRERELEASE and optional +BUILD)
	if [[ ! "$tag_version" =~ ^v[0-9]+(\.[0-9]+){2}(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$ ]]; then
		echo -e "${RED}Error: Tag version should follow SemVer format 'vX.Y.Z' (e.g., v0.3.5, v1.2.3-rc.1, v1.2.3+build.5)${NC}" >&2
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

# Run checks
check_git_repo
check_changelog

# Run main logic
main "$@"
