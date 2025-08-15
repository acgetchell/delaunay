#!/bin/bash

# Generate changelog with commit dates instead of tag dates
# This script uses auto-changelog with a custom template and post-processes
# the output to format ISO dates correctly.

set -euo pipefail

# Script configuration
CHANGELOG_FILE="CHANGELOG.md"
TEMP_CHANGELOG="${CHANGELOG_FILE}.tmp"
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Generating changelog with commit dates..."

# Change to project root directory to ensure auto-changelog finds config files
cd "${PROJECT_ROOT}"

# Verify prerequisites
if ! command -v npx > /dev/null 2>&1; then
  echo "Error: npx is required to run auto-changelog, please install Node.js and npm." >&2
  echo "Visit https://nodejs.org/ to install Node.js and npm." >&2
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

# Generate the changelog using our custom template and post-process the dates
# The sed command removes the time portion from ISO dates, leaving only YYYY-MM-DD
# Check if we have git history (needed for changelog generation)
if ! git rev-parse --git-dir > /dev/null 2>&1; then
  echo "Error: This script must be run in a Git repository." >&2
  exit 1
fi

if ! git log --oneline -n 1 > /dev/null 2>&1; then
  echo "Error: No git history found. Cannot generate changelog." >&2
  exit 1
fi

echo "Running auto-changelog with custom template..."
if ! npx auto-changelog --stdout > "${TEMP_CHANGELOG}" 2> "${TEMP_CHANGELOG}.err"; then
  echo "Error: auto-changelog failed to generate changelog." >&2
  if [[ -f "${TEMP_CHANGELOG}.err" ]]; then
    echo "Error details:" >&2
    cat "${TEMP_CHANGELOG}.err" >&2
    rm -f "${TEMP_CHANGELOG}.err"
  fi
  # Restore backup if it exists
  if [[ -f "${CHANGELOG_FILE}.backup" ]]; then
    mv "${CHANGELOG_FILE}.backup" "${CHANGELOG_FILE}"
    echo "Restored original ${CHANGELOG_FILE} from backup."
  fi
  rm -f "${TEMP_CHANGELOG}"
  exit 1
fi

# Clean up error file if it exists
rm -f "${TEMP_CHANGELOG}.err"

# Post-process dates to remove time portion (ISO format -> YYYY-MM-DD)
echo "Processing date formats..."
if ! sed 's/T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].*Z//g' "${TEMP_CHANGELOG}" > "${CHANGELOG_FILE}"; then
  echo "Error: Failed to process date formats." >&2
  # Restore backup if it exists
  if [[ -f "${CHANGELOG_FILE}.backup" ]]; then
    mv "${CHANGELOG_FILE}.backup" "${CHANGELOG_FILE}"
    echo "Restored original ${CHANGELOG_FILE} from backup."
  fi
  exit 1
fi

# Clean up temporary file
rm -f "${TEMP_CHANGELOG}"

# Remove backup if everything succeeded
if [[ -f "${CHANGELOG_FILE}.backup" ]]; then
  rm "${CHANGELOG_FILE}.backup"
fi

echo "✓ ${CHANGELOG_FILE} has been updated with commit dates instead of tag dates."
echo "✓ Generated $(grep -c '^## ' "${CHANGELOG_FILE}" || echo 0) releases in changelog."
