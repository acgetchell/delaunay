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

# Function to expand squashed PR commits in changelog with enhanced body parsing
expand_squashed_prs() {
    local input_file="$1"
    local output_file="$2"
    
    echo "Expanding squashed PR commits with enhanced body parsing..."
    
    # Create temporary file for processing
    local temp_file="${output_file}.expand_temp"
    
    # Process the changelog line by line
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Check if this line contains a commit SHA that might be from a squashed PR
        # Handle both "- **title** [`sha`]" and "- title (#PR) [`sha`]" patterns
        if [[ "$line" =~ -\ \*\*.*\*\*.*\[\`([a-f0-9]{7})\`\] ]] || [[ "$line" =~ -\ .*\(#[0-9]+\)\ \[\`([a-f0-9]{7})\`\] ]]; then
            commit_sha="${BASH_REMATCH[1]}"
            
            # Get the full commit message to check if it's a squashed PR
            if git --no-pager show "$commit_sha" --format="%s" --no-patch 2>/dev/null | grep -q "(#[0-9]\+)$"; then
                echo "  Found squashed PR commit: $commit_sha"
                
                # Get the full commit message including body
                local commit_msg_file="$(mktemp)"
                git --no-pager show "$commit_sha" --format="%B" --no-patch > "$commit_msg_file"
                
                # Enhanced commit parsing with better body handling
                awk '
                BEGIN {
                    in_entry = 0
                    entry_title = ""
                    entry_body = ""
                    first_entry = 1
                    found_any_entries = 0
                }
                
                # Skip the first line (PR title) and empty lines at start
                NR == 1 { next }
                /^[ \t]*$/ && !found_any_entries { next }
                
                # Detect bullet points: "* ", "- ", or "\d+. "
                /^[ \t]*[*-][ \t]+/ || /^[ \t]*[0-9]+\.[ \t]+/ {
                    # Output previous entry if we have one
                    if (in_entry && entry_title != "") {
                        output_entry()
                    }
                    
                    # Start new entry
                    found_any_entries = 1
                    in_entry = 1
                    
                    # Extract title (remove bullet marker and leading/trailing whitespace)
                    if (/^[ \t]*[*-][ \t]+/) {
                        # Handle "* " or "- " bullets
                        gsub(/^[ \t]*[*-][ \t]+/, "", $0)
                    } else {
                        # Handle numbered bullets "1. ", "2. ", etc.
                        gsub(/^[ \t]*[0-9]+\.[ \t]+/, "", $0)
                    }
                    
                    entry_title = $0
                    entry_body = ""
                    next
                }
                
                # Collect body lines for current entry
                in_entry {
                    # Check if this line starts a new bullet (backup detection)
                    if (/^[ \t]*[*-][ \t]+/ || /^[ \t]*[0-9]+\.[ \t]+/) {
                        # This is a new bullet, so process it in next iteration
                        # First output current entry
                        if (entry_title != "") {
                            output_entry()
                        }
                        
                        # Reset for new entry
                        found_any_entries = 1
                        in_entry = 1
                        
                        # Extract title
                        if (/^[ \t]*[*-][ \t]+/) {
                            gsub(/^[ \t]*[*-][ \t]+/, "", $0)
                        } else {
                            gsub(/^[ \t]*[0-9]+\.[ \t]+/, "", $0)
                        }
                        
                        entry_title = $0
                        entry_body = ""
                        next
                    }
                    
                    # Add line to body (preserve blank lines as paragraph breaks)
                    if (/^[ \t]*$/) {
                        # Blank line - preserve as paragraph break
                        if (entry_body != "" && !match(entry_body, /\n[ \t]*$/)) {
                            entry_body = entry_body "\n\n"
                        }
                    } else {
                        # Non-blank line - clean and add to body
                        line_content = $0
                        # Remove leading whitespace but preserve relative indentation
                        gsub(/^[ \t]+/, "", line_content)
                        
                        if (entry_body == "") {
                            entry_body = line_content
                        } else if (match(entry_body, /\n[ \t]*$/)) {
                            # Previous line was blank, start new paragraph
                            entry_body = entry_body line_content
                        } else {
                            # Continue current paragraph
                            entry_body = entry_body " " line_content
                        }
                    }
                    next
                }
                
                # If no bullets found, treat entire content as single entry
                !found_any_entries {
                    if (!in_entry) {
                        in_entry = 1
                        entry_title = $0
                        entry_body = ""
                        found_any_entries = 1
                        next
                    } else {
                        # Add to body
                        if (/^[ \t]*$/) {
                            if (entry_body != "" && !match(entry_body, /\n[ \t]*$/)) {
                                entry_body = entry_body "\n\n"
                            }
                        } else {
                            line_content = $0
                            gsub(/^[ \t]+/, "", line_content)
                            
                            if (entry_body == "") {
                                entry_body = line_content
                            } else if (match(entry_body, /\n[ \t]*$/)) {
                                entry_body = entry_body line_content
                            } else {
                                entry_body = entry_body " " line_content
                            }
                        }
                        next
                    }
                }
                
                function output_entry() {
                    if (entry_title != "") {
                        if (!first_entry) print ""
                        first_entry = 0
                        
                        # Output title
                        print "- **" tolower(entry_title) "**"
                        
                        # Output body with proper indentation
                        if (entry_body != "") {
                            gsub(/^[ \t]+|[ \t]+$/, "", entry_body)
                            if (length(entry_body) > 0) {
                                # Split body into paragraphs and format each
                                split(entry_body, paragraphs, /\n[ \t]*\n/)
                                for (i = 1; i <= length(paragraphs); i++) {
                                    para = paragraphs[i]
                                    gsub(/^[ \t]+|[ \t]+$/, "", para)
                                    if (length(para) > 0) {
                                        # Word wrap long paragraphs
                                        while (length(para) > 0) {
                                            if (length(para) <= 75) {
                                                print "  " para
                                                break
                                            } else {
                                                # Find last space within 75 chars
                                                wrap_pos = 75
                                                while (wrap_pos > 30 && substr(para, wrap_pos, 1) != " ") {
                                                    wrap_pos--
                                                }
                                                if (wrap_pos <= 30) wrap_pos = 75  # Force break if no space found
                                                
                                                print "  " substr(para, 1, wrap_pos)
                                                para = substr(para, wrap_pos + 1)
                                                gsub(/^[ \t]+/, "", para)
                                            }
                                        }
                                        
                                        # Add blank line between paragraphs
                                        if (i < length(paragraphs)) print ""
                                    }
                                }
                            }
                        }
                    }
                }
                
                END {
                    # Output final entry if we have one
                    if (in_entry && entry_title != "") {
                        output_entry()
                    }
                }
                ' "$commit_msg_file" >> "$temp_file"
                
                # Clean up temp file
                rm -f "$commit_msg_file"
            else
                # Not a squashed PR, keep the original line
                echo "$line" >> "$temp_file"
            fi
        else
            # Not a commit line, keep as is
            echo "$line" >> "$temp_file"
        fi
    done < "$input_file"
    
    # Move the processed file to the output
    mv "$temp_file" "$output_file"
}

# Function to enhance AI-generated commit messages in changelog with Keep a Changelog categorization
enhance_ai_commits() {
    local input_file="$1"
    local output_file="$2"
    
    echo "Enhancing AI-generated commit formatting with Keep a Changelog categorization..."
    
    # Process the changelog line by line and categorize according to Keep a Changelog format
    python3 -c "
import sys
import re

# Read the input file
with open('$input_file', 'r') as f:
    lines = f.readlines()

output_lines = []
in_changes_section = False
in_fixed_issues = False
added_section = False
changed_section = False
deprecated_section = False
removed_section = False
fixed_section = False
security_section = False

def add_section_break():
    global output_lines
    if added_section or changed_section or deprecated_section or removed_section or fixed_section or security_section:
        output_lines.append('')

for line in lines:
    line = line.rstrip()
    
    # Track when we enter Changes section (from template)
    if re.match(r'^### *Changes$', line):
        in_changes_section = True
        in_fixed_issues = False
        # Reset section flags for this release
        added_section = False
        changed_section = False
        deprecated_section = False
        removed_section = False
        fixed_section = False
        security_section = False
        # Skip the original Changes header - we'll create proper sections
        continue
    
    # Track when we enter Fixed Issues section (from template)
    elif re.match(r'^### *Fixed Issues$', line):
        in_fixed_issues = True
        in_changes_section = False
        # Reset section flags for this release
        added_section = False
        changed_section = False
        deprecated_section = False
        removed_section = False
        fixed_section = False
        security_section = False
        # Skip the original Fixed Issues header - we'll create proper sections
        continue
    
    # Stop processing when we hit the next release section
    elif re.match(r'^## ', line) and (in_changes_section or in_fixed_issues):
        in_changes_section = False
        in_fixed_issues = False
    
    # Process commit lines in Changes or Fixed Issues sections
    elif (in_changes_section or in_fixed_issues) and re.match(r'^- \\*\\*', line):
        # Determine category based on commit content and keywords
        commit_line = line.lower()
        
        # ADDED: New features, new functionality
        if ('**add' in commit_line or '**implement' in commit_line or 
            '**introduce' in commit_line or '**new' in commit_line or
            'adds ' in commit_line or 'implement' in commit_line or
            'introduces ' in commit_line or 'new ' in commit_line or
            'feature' in commit_line):
            
            if not added_section:
                add_section_break()
                output_lines.append('### Added')
                output_lines.append('')
                added_section = True
        
        # FIXED: Bug fixes
        elif ('**fix' in commit_line or '**bug' in commit_line or
              '**patch' in commit_line or '**resolve' in commit_line or
              'fixes ' in commit_line or 'fixed ' in commit_line or
              'bug' in commit_line or 'patch' in commit_line or
              'resolve' in commit_line or 'addresses' in commit_line or
              'handles' in commit_line or 'error' in commit_line):
            
            if not fixed_section:
                add_section_break()
                output_lines.append('### Fixed')
                output_lines.append('')
                fixed_section = True
        
        # CHANGED: Changes to existing functionality (improvements, refactors, updates)
        elif ('**refactor' in commit_line or '**improve' in commit_line or
              '**update' in commit_line or '**change' in commit_line or
              '**enhance' in commit_line or '**upgrade' in commit_line or
              'refactor' in commit_line or 'improve' in commit_line or
              'update' in commit_line or 'change' in commit_line or
              'enhance' in commit_line or 'upgrade' in commit_line or
              'bump' in commit_line or 'increase' in commit_line or
              'better' in commit_line or 'optimiz' in commit_line or
              'performance' in commit_line or 'standardiz' in commit_line):
            
            if not changed_section:
                add_section_break()
                output_lines.append('### Changed')
                output_lines.append('')
                changed_section = True
        
        # REMOVED: Removed functionality
        elif ('**remove' in commit_line or '**delete' in commit_line or
              '**drop' in commit_line or 'remov' in commit_line or
              'delet' in commit_line or 'drop' in commit_line or
              'legacy' in commit_line):
            
            if not removed_section:
                add_section_break()
                output_lines.append('### Removed')
                output_lines.append('')
                removed_section = True
        
        # DEPRECATED: Soon-to-be removed features
        elif ('**deprecat' in commit_line or 'deprecat' in commit_line):
            
            if not deprecated_section:
                add_section_break()
                output_lines.append('### Deprecated')
                output_lines.append('')
                deprecated_section = True
        
        # SECURITY: Security fixes
        elif ('**security' in commit_line or 'security' in commit_line or
              'vulnerabilit' in commit_line or 'cve' in commit_line):
            
            if not security_section:
                add_section_break()
                output_lines.append('### Security')
                output_lines.append('')
                security_section = True
        
        # Default to CHANGED for other modifications
        else:
            if not changed_section:
                add_section_break()
                output_lines.append('### Changed')
                output_lines.append('')
                changed_section = True
        
        # Add the commit line
        output_lines.append(line)
    
    else:
        # Print all other lines normally (headers, merged PRs, etc.)
        output_lines.append(line)

# Write the output file
with open('$output_file', 'w') as f:
    for line in output_lines:
        f.write(line + '\n')
"
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

# Expand squashed PR commits
echo "Expanding squashed PR commits..."
if ! expand_squashed_prs "${PROCESSED_CHANGELOG}" "${PROCESSED_CHANGELOG}.expanded"; then
  echo "Error: Failed to expand squashed PR commits." >&2
  # Restore backup if it exists
  if [[ -f "${CHANGELOG_FILE}.backup" ]]; then
    mv "${CHANGELOG_FILE}.backup" "${CHANGELOG_FILE}"
    echo "Restored original ${CHANGELOG_FILE} from backup."
  fi
  rm -f "${TEMP_CHANGELOG}" "${PROCESSED_CHANGELOG}"
  exit 1
fi

# Enhance AI-generated commits with categorization
if ! enhance_ai_commits "${PROCESSED_CHANGELOG}.expanded" "${CHANGELOG_FILE}.tmp2"; then
  echo "Error: Failed to enhance AI commits." >&2
  # Restore backup if it exists
  if [[ -f "${CHANGELOG_FILE}.backup" ]]; then
    mv "${CHANGELOG_FILE}.backup" "${CHANGELOG_FILE}"
    echo "Restored original ${CHANGELOG_FILE} from backup."
  fi
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
' "${CHANGELOG_FILE}.tmp2" > "${CHANGELOG_FILE}"; then
  echo "Error: Failed to clean up blank lines." >&2
  # Restore backup if it exists
  if [[ -f "${CHANGELOG_FILE}.backup" ]]; then
    mv "${CHANGELOG_FILE}.backup" "${CHANGELOG_FILE}"
    echo "Restored original ${CHANGELOG_FILE} from backup."
  fi
  rm -f "${TEMP_CHANGELOG}" "${PROCESSED_CHANGELOG}" "${PROCESSED_CHANGELOG}.expanded" "${CHANGELOG_FILE}.tmp2"
  exit 1
fi

# Clean up temporary files
rm -f "${TEMP_CHANGELOG}" "${PROCESSED_CHANGELOG}" "${PROCESSED_CHANGELOG}.expanded" "${CHANGELOG_FILE}.tmp2"

# Remove backup if everything succeeded
if [[ -f "${CHANGELOG_FILE}.backup" ]]; then
  rm "${CHANGELOG_FILE}.backup"
fi

echo "✓ ${CHANGELOG_FILE} has been updated with enhanced AI commit processing."
echo "✓ Generated $(grep -c '^## ' "${CHANGELOG_FILE}" || echo 0) releases in changelog."
echo "✓ Commits are now categorized according to Keep a Changelog format (Added/Changed/Fixed/etc.)."
