#!/bin/bash

# Generate changelog with commit dates instead of tag dates
# This script uses auto-changelog with a custom template and post-processes
# the output to format ISO dates correctly and enhance AI-generated commit messages.

set -euo pipefail

# Script configuration
CHANGELOG_FILE="CHANGELOG.md"
TEMP_CHANGELOG="${CHANGELOG_FILE}.tmp"
PROCESSED_CHANGELOG="${CHANGELOG_FILE}.processed"
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Generating changelog with enhanced AI commit processing..."

# Function to enhance AI-generated commit messages in changelog
enhance_ai_commits() {
    local input_file="$1"
    local output_file="$2"
    
    echo "Enhancing AI-generated commit formatting..."
    
    # Use awk to process the changelog and improve formatting
    awk '
    BEGIN {
        in_changes_section = 0
        feat_section = 0
        fix_section = 0
        docs_section = 0
        perf_section = 0
        refactor_section = 0
        test_section = 0
        ci_section = 0
        chore_section = 0
        other_section = 0
    }
    
    # Track when we enter Changes section
    /^### Changes$/ {
        in_changes_section = 1
        # Reset section flags for this release
        feat_section = 0
        fix_section = 0
        docs_section = 0
        perf_section = 0
        refactor_section = 0
        test_section = 0
        ci_section = 0
        chore_section = 0
        other_section = 0
        # Skip printing this line, we will replace it with categorized sections
        next
    }
    
    # Stop processing when we hit the next release section or end
    /^## / && in_changes_section {
        in_changes_section = 0
    }
    
    # Process commit lines in Changes section
    in_changes_section && /^- \*\*/ {
        # Print the Changes header if this is the first commit we are processing
        if (!feat_section && !fix_section && !docs_section && !perf_section && !refactor_section && !test_section && !ci_section && !chore_section && !other_section) {
            print "### Changes"
            print ""
        }
        
        # Extract commit type from subject
        if (match($0, /\*\*feat[\(:].*\*\*/)) {
            if (!feat_section) {
                # Add blank line before new section if we have printed commits already
                if (other_section || fix_section || docs_section || perf_section || refactor_section || test_section || ci_section || chore_section) {
                    print ""
                }
                print "#### ✨ Features"
                print ""
                feat_section = 1
            }
        } else if (match($0, /\*\*fix[\(:].*\*\*/)) {
            if (!fix_section) {
                # Add blank line before new section if we have printed commits already
                if (feat_section || other_section || docs_section || perf_section || refactor_section || test_section || ci_section || chore_section) {
                    print ""
                }
                print "#### 🐛 Bug Fixes"
                print ""
                fix_section = 1
            }
        } else if (match($0, /\*\*docs[\(:].*\*\*/)) {
            if (!docs_section) {
                # Add blank line before new section if we have printed commits already
                if (feat_section || fix_section || other_section || perf_section || refactor_section || test_section || ci_section || chore_section) {
                    print ""
                }
                print "#### 📚 Documentation"
                print ""
                docs_section = 1
            }
        } else if (match($0, /\*\*perf[\(:].*\*\*/)) {
            if (!perf_section) {
                # Add blank line before new section if we have printed commits already
                if (feat_section || fix_section || docs_section || other_section || refactor_section || test_section || ci_section || chore_section) {
                    print ""
                }
                print "#### ⚡ Performance"
                print ""
                perf_section = 1
            }
        } else if (match($0, /\*\*refactor[\(:].*\*\*/)) {
            if (!refactor_section) {
                # Add blank line before new section if we have printed commits already
                if (feat_section || fix_section || docs_section || perf_section || other_section || test_section || ci_section || chore_section) {
                    print ""
                }
                print "#### ♻️ Refactoring"
                print ""
                refactor_section = 1
            }
        } else if (match($0, /\*\*test[\(:].*\*\*/)) {
            if (!test_section) {
                # Add blank line before new section if we have printed commits already
                if (feat_section || fix_section || docs_section || perf_section || refactor_section || other_section || ci_section || chore_section) {
                    print ""
                }
                print "#### 🧪 Testing"
                print ""
                test_section = 1
            }
        } else if (match($0, /\*\*ci[\(:].*\*\*/)) {
            if (!ci_section) {
                # Add blank line before new section if we have printed commits already
                if (feat_section || fix_section || docs_section || perf_section || refactor_section || test_section || other_section || chore_section) {
                    print ""
                }
                print "#### 🔄 CI/CD"
                print ""
                ci_section = 1
            }
        } else if (match($0, /\*\*chore[\(:].*\*\*/)) {
            if (!chore_section) {
                # Add blank line before new section if we have printed commits already
                if (feat_section || fix_section || docs_section || perf_section || refactor_section || test_section || ci_section || other_section) {
                    print ""
                }
                print "#### 🔧 Maintenance"
                print ""
                chore_section = 1
            }
        } else {
            if (!other_section) {
                # Add blank line before new section if we have printed commits already
                if (feat_section || fix_section || docs_section || perf_section || refactor_section || test_section || ci_section || chore_section) {
                    print ""
                }
                print "#### 📝 Other Changes"
                print ""
                other_section = 1
            }
        }
        
        # Print the commit line
        print $0
        next
    }
    
    # Print other lines normally, unless we are skipping Changes section
    !in_changes_section || !/^### Changes$/ {
        print $0
    }
    ' "$input_file" > "$output_file"
}

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
if ! sed 's/T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].*Z//g' "${TEMP_CHANGELOG}" > "${PROCESSED_CHANGELOG}"; then
  echo "Error: Failed to process date formats." >&2
  # Restore backup if it exists
  if [[ -f "${CHANGELOG_FILE}.backup" ]]; then
    mv "${CHANGELOG_FILE}.backup" "${CHANGELOG_FILE}"
    echo "Restored original ${CHANGELOG_FILE} from backup."
  fi
  rm -f "${TEMP_CHANGELOG}"
  exit 1
fi

# Enhance AI-generated commits with categorization
if ! enhance_ai_commits "${PROCESSED_CHANGELOG}" "${CHANGELOG_FILE}.tmp2"; then
  echo "Error: Failed to enhance AI commits." >&2
  # Restore backup if it exists
  if [[ -f "${CHANGELOG_FILE}.backup" ]]; then
    mv "${CHANGELOG_FILE}.backup" "${CHANGELOG_FILE}"
    echo "Restored original ${CHANGELOG_FILE} from backup."
  fi
  rm -f "${TEMP_CHANGELOG}" "${PROCESSED_CHANGELOG}"
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
' "${CHANGELOG_FILE}.tmp2" > "${CHANGELOG_FILE}"; then
  echo "Error: Failed to clean up blank lines." >&2
  # Restore backup if it exists
  if [[ -f "${CHANGELOG_FILE}.backup" ]]; then
    mv "${CHANGELOG_FILE}.backup" "${CHANGELOG_FILE}"
    echo "Restored original ${CHANGELOG_FILE} from backup."
  fi
  rm -f "${TEMP_CHANGELOG}" "${PROCESSED_CHANGELOG}" "${CHANGELOG_FILE}.tmp2"
  exit 1
fi

# Clean up temporary files
rm -f "${TEMP_CHANGELOG}" "${PROCESSED_CHANGELOG}" "${CHANGELOG_FILE}.tmp2"

# Remove backup if everything succeeded
if [[ -f "${CHANGELOG_FILE}.backup" ]]; then
  rm "${CHANGELOG_FILE}.backup"
fi

echo "✓ ${CHANGELOG_FILE} has been updated with enhanced AI commit processing."
echo "✓ Generated $(grep -c '^## ' "${CHANGELOG_FILE}" || echo 0) releases in changelog."
echo "✓ Commits are now categorized by type with emoji indicators."
