#!/bin/bash

# Generate changelog with commit dates instead of tag dates
# This script uses auto-changelog with a custom template and post-processes
# the output to format ISO dates correctly and enhance AI-generated commit messages.

set -euo pipefail

# Script configuration
CHANGELOG_FILE="CHANGELOG.md"
TEMP_CHANGELOG="${CHANGELOG_FILE}.tmp"
PROCESSED_CHANGELOG="${CHANGELOG_FILE}.processed"

# Auto-restore backup on unexpected termination
restore_backup_and_exit() {
	if [[ -f "${CHANGELOG_FILE}.backup" ]]; then
		mv "${CHANGELOG_FILE}.backup" "${CHANGELOG_FILE}"
		echo "Restored original ${CHANGELOG_FILE} from backup."
	fi
	exit 1
}
trap restore_backup_and_exit INT TERM

# Function to show usage information
show_help() {
	cat <<EOF
Usage: $0 [OPTIONS]

Generate an enhanced changelog with AI commit processing and Keep a Changelog categorization.

Options:
  --debug     Preserve intermediate files for debugging
  --help      Show this help message

Examples:
  $0                    # Normal run, clean up intermediate files
  $0 --debug            # Keep intermediate files for debugging

Intermediate files (when using --debug):
  - CHANGELOG.md.tmp (initial auto-changelog output)
  - CHANGELOG.md.processed (after date processing)
  - CHANGELOG.md.processed.expanded (after PR expansion)
  - CHANGELOG.md.tmp2 (after AI enhancement)

EOF
}

# Parse command line arguments
PRESERVE_TEMP_FILES=false
while [[ $# -gt 0 ]]; do
	case $1 in
	--debug)
		PRESERVE_TEMP_FILES=true
		shift
		;;
	--help)
		show_help
		exit 0
		;;
	*)
		echo "Error: Unknown option '$1'" >&2
		echo "Use --help for usage information." >&2
		exit 1
		;;
	esac
done

# Additional script configuration
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Generating changelog with enhanced AI commit processing..."

# Function to expand squashed PR commits in changelog with enhanced body parsing
expand_squashed_prs() {
	local input_file="$1"
	local output_file="$2"

	echo "Expanding squashed PR commits with enhanced body parsing..."

	# Get repository URL using shared utilities
	local repo_url
	repo_url=$(python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from changelog_utils import ChangelogUtils, GitRepoError

try:
    print(ChangelogUtils.get_repository_url())
except GitRepoError:
    print('https://github.com/acgetchell/delaunay')  # Default fallback
")
	echo "  Using repository URL: $repo_url"

	# Create temporary file for processing
	local temp_file="${output_file}.expand_temp"

	# Process the changelog line by line
	while IFS= read -r line || [[ -n "$line" ]]; do
		# Check if this line contains a commit SHA that might be from a squashed PR
		# Handle both "- **title** [`sha`]" and "- title (#PR) [`sha`]" patterns
		if [[ "$line" =~ -\ \*\*.*\*\*.*\[\`([a-f0-9]{7,40})\`\] ]] || [[ "$line" =~ -\ .*\(#[0-9]+\)\ \[\`([a-f0-9]{7,40})\`\] ]]; then
			local commit_sha="${BASH_REMATCH[1]}"

			# Get the full commit message to check if it's a squashed PR
			if git --no-pager show "$commit_sha" --format="%s" --no-patch 2>/dev/null | grep -E -q "\(#[0-9]+\)$"; then
				echo "  Found squashed PR commit: $commit_sha"

				# Use Python utility to process the commit
				python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from changelog_utils import ChangelogUtils, GitRepoError

try:
    result = ChangelogUtils.process_squashed_commit('$commit_sha', '$repo_url')
    print(result)
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
" >>"$temp_file"
			else
				# Not a squashed PR, keep the original line
				echo "$line" >>"$temp_file"
			fi
		else
			# Not a commit line, keep as is
			echo "$line" >>"$temp_file"
		fi
	done <"$input_file"

	# Move the processed file to the output
	mv "$temp_file" "$output_file"
}

# Function to enhance AI-generated commit messages in changelog with Keep a Changelog categorization
enhance_ai_commits() {
	local input_file="$1"
	local output_file="$2"

	echo "Enhancing AI-generated commit formatting with Keep a Changelog categorization..."

	# Check Python3 prerequisite
	if ! command -v python3 >/dev/null 2>&1; then
		echo "Error: python3 is required to run enhance_commits.py for changelog generation." >&2
		echo "Please install Python 3 to use this feature." >&2
		return 1
	fi

	# Use external Python script to avoid shell/Python quoting issues
	if ! python3 "${SCRIPT_DIR}/enhance_commits.py" "$input_file" "$output_file"; then
		return 1
	fi
}

# Change to project root directory to ensure auto-changelog finds config files
cd "${PROJECT_ROOT}"

# Verify prerequisites
if ! command -v npx >/dev/null 2>&1; then
	echo "Error: npx is required to run auto-changelog, please install Node.js and npm." >&2
	echo "Visit https://nodejs.org/ to install Node.js and npm." >&2
	exit 1
fi

# Ensure auto-changelog is runnable non-interactively via npx
if ! npx --yes -p auto-changelog auto-changelog --version >/dev/null 2>&1; then
	echo "Error: auto-changelog is not available via npx (network? registry?)." >&2
	echo "Tip: add it as a devDependency or retry with network access." >&2
	exit 1
fi

# Verify configuration files exist
if [[ ! -f ".auto-changelog" ]]; then
	echo "Error: .auto-changelog configuration file not found in project root." >&2
	exit 1
fi

if [[ ! -f "docs/templates/changelog.hbs" ]]; then
	echo "Error: Custom changelog template not found at docs/templates/changelog.hbs" >&2
	exit 1
fi

# Backup existing changelog if it exists
if [[ -f "${CHANGELOG_FILE}" ]]; then
	cp "${CHANGELOG_FILE}" "${CHANGELOG_FILE}.backup"
	echo "Backed up existing ${CHANGELOG_FILE} to ${CHANGELOG_FILE}.backup"
fi

# Validate git repository and history using shared utilities
echo "Validating git repository..."
if ! python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from changelog_utils import ChangelogUtils, GitRepoError

try:
    ChangelogUtils.validate_git_repo()
    ChangelogUtils.check_git_history()
except GitRepoError as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
"; then
	exit 1
fi

echo "Running auto-changelog with custom template..."
if ! npx --yes -p auto-changelog auto-changelog --stdout >"${TEMP_CHANGELOG}" 2>"${TEMP_CHANGELOG}.err"; then

	echo "Error: auto-changelog failed to generate changelog." >&2
	if [[ -f "${TEMP_CHANGELOG}.err" ]]; then
		echo "Error details:" >&2
		cat "${TEMP_CHANGELOG}.err" >&2
		rm -f "${TEMP_CHANGELOG}.err"
	fi
	rm -f "${TEMP_CHANGELOG}"
	exit 1
fi

# Clean up error file if it exists
rm -f "${TEMP_CHANGELOG}.err"

# Post-process dates to remove time portion (ISO format -> YYYY-MM-DD)
echo "Processing date formats..."
if ! sed 's/T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].*Z//g' "${TEMP_CHANGELOG}" >"${PROCESSED_CHANGELOG}"; then
	echo "Error: Failed to process date formats." >&2
	rm -f "${TEMP_CHANGELOG}"
	exit 1
fi

# Expand squashed PR commits
echo "Expanding squashed PR commits..."
if ! expand_squashed_prs "${PROCESSED_CHANGELOG}" "${PROCESSED_CHANGELOG}.expanded"; then
	echo "Error: Failed to expand squashed PR commits." >&2
	rm -f "${TEMP_CHANGELOG}" "${PROCESSED_CHANGELOG}"
	exit 1
fi

# Enhance AI-generated commits with categorization
if ! enhance_ai_commits "${PROCESSED_CHANGELOG}.expanded" "${CHANGELOG_FILE}.tmp2"; then
	echo "Error: Failed to enhance AI commits." >&2
	rm -f "${TEMP_CHANGELOG}" "${PROCESSED_CHANGELOG}" "${PROCESSED_CHANGELOG}.expanded"
	exit 1
fi

# Final cleanup: remove excessive blank lines (more than 1 consecutive)
echo "Cleaning up excessive blank lines..."
if ! awk '
/^$/ { 
    empty++; 
    if (empty <= 1) print; 
    next 
}
{ 
    empty=0; 
    print 
}
' "${CHANGELOG_FILE}.tmp2" >"${CHANGELOG_FILE}"; then
	echo "Error: Failed to clean up blank lines." >&2
	rm -f "${TEMP_CHANGELOG}" "${PROCESSED_CHANGELOG}" "${PROCESSED_CHANGELOG}.expanded" "${CHANGELOG_FILE}.tmp2"
	exit 1
fi

# Clean up temporary files (unless preservation is requested)
if [[ "${PRESERVE_TEMP_FILES}" != "true" ]]; then
	rm -f "${TEMP_CHANGELOG}" "${PROCESSED_CHANGELOG}" "${PROCESSED_CHANGELOG}.expanded" "${CHANGELOG_FILE}.tmp2"
else
	echo "ℹ Preserving intermediate files for debugging:"
	echo "  - ${TEMP_CHANGELOG} (initial auto-changelog output)"
	echo "  - ${PROCESSED_CHANGELOG} (after date processing)"
	echo "  - ${PROCESSED_CHANGELOG}.expanded (after PR expansion)"
	echo "  - ${CHANGELOG_FILE}.tmp2 (after AI enhancement)"
fi

# Remove backup if everything succeeded
if [[ -f "${CHANGELOG_FILE}.backup" ]]; then
	rm "${CHANGELOG_FILE}.backup"
fi

echo "✓ ${CHANGELOG_FILE} has been updated with enhanced AI commit processing."
echo "✓ Generated $(grep -c '^## ' "${CHANGELOG_FILE}" || echo 0) releases in changelog."
echo "✓ Commits are now categorized according to Keep a Changelog format (Added/Changed/Fixed/etc.)."
