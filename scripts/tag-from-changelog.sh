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
# The script extracts the first changelog section from CHANGELOG.md and uses it
# as the tag message, making it easy to create GitHub releases with proper content.

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
    echo "The script extracts the first changelog section from CHANGELOG.md"
    echo "and uses it as the tag message for GitHub release integration."
}

# Check if we're in a git repository
check_git_repo() {
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
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

# Extract changelog content for the tag
extract_changelog() {
    local changelog_content
    changelog_content=$(awk '/^## \[/{f=!f; next} f' "$CHANGELOG_PATH")
    
    if [[ -z "$changelog_content" ]]; then
        echo -e "${YELLOW}Warning: No changelog content found${NC}" >&2
        echo "Make sure $CHANGELOG_PATH has at least one release section starting with '## ['" >&2
        echo "Using minimal tag message instead." >&2
        echo "$TAG_VERSION"
    else
        echo "$changelog_content"
    fi
}

# Main script logic
main() {
    local tag_version="$1"
    local force_recreate="${2:-}"
    
    # Validate tag format (basic check for v prefix and version-like pattern)
    if [[ ! "$tag_version" =~ ^v[0-9]+\.[0-9]+\.[0-9]+ ]]; then
        echo -e "${RED}Error: Tag version should follow format 'vX.Y.Z' (e.g., v0.3.5)${NC}" >&2
        exit 1
    fi
    
    # Check if tag already exists
    if git tag -l | grep -q "^${tag_version}$"; then
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
    tag_message=$(extract_changelog)
    
    # Show preview of tag message
    echo -e "${BLUE}Tag message preview:${NC}"
    echo "----------------------------------------"
    echo "$tag_message"
    echo "----------------------------------------"
    
    # Create the tag
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

TAG_VERSION="$1"

# Run checks
check_git_repo
check_changelog

# Run main logic
main "$@"
