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
    
    # Detect repository URL from git remote origin
    local repo_url
    repo_url=$(git remote get-url origin 2>/dev/null || echo "")
    if [[ -z "$repo_url" ]]; then
        echo "Warning: Could not detect git remote origin URL. Using default GitHub format."
        repo_url="https://github.com/acgetchell/delaunay"
    else
        # Convert SSH URLs to HTTPS format and clean up
        if [[ "$repo_url" =~ ^git@github\.com:(.+)\.git$ ]]; then
            repo_url="https://github.com/${BASH_REMATCH[1]}"
        elif [[ "$repo_url" =~ ^https://github\.com/(.+)\.git$ ]]; then
            repo_url="https://github.com/${BASH_REMATCH[1]}"
        elif [[ "$repo_url" =~ ^https://github\.com/(.+)$ ]]; then
            repo_url="https://github.com/${BASH_REMATCH[1]}"
        fi
        # Remove trailing slash if present
        repo_url="${repo_url%/}"
    fi
    echo "  Using repository URL: $repo_url"
    
    # Create temporary file for processing
    local temp_file="${output_file}.expand_temp"
    
    # Process the changelog line by line
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Check if this line contains a commit SHA that might be from a squashed PR
        # Handle both "- **title** [`sha`]" and "- title (#PR) [`sha`]" patterns
        if [[ "$line" =~ -\ \*\*.*\*\*.*\[\`([a-f0-9]{7,40})\`\] ]] || [[ "$line" =~ -\ .*\(#[0-9]+\)\ \[\`([a-f0-9]{7,40})\`\] ]]; then
            commit_sha="${BASH_REMATCH[1]}"
            
            # Get the full commit message to check if it's a squashed PR
            if git --no-pager show "$commit_sha" --format="%s" --no-patch 2>/dev/null | grep -E -q "\(#[0-9]+\)$"; then
                echo "  Found squashed PR commit: $commit_sha"
                
                # Get the full commit message including body
                local commit_msg_file
                commit_msg_file="$(mktemp)"
                git --no-pager show "$commit_sha" --format="%B" --no-patch > "$commit_msg_file"
                
                # Enhanced commit parsing with better body handling - output only bullet points
                awk -v commit_sha="$commit_sha" -v repo_url="$repo_url" '
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
                        # Preserve leading whitespace to keep Markdown code blocks/lists intact
                        line_content = $0
                        
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
                            # Preserve leading whitespace to keep Markdown code blocks/lists intact
                            line_content = $0
                            
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
                        
                        # Output title with commit SHA link using dynamic repo URL
                        print "- **" entry_title "** [\`" commit_sha "\`](" repo_url "/commit/" commit_sha ")"
                        
                        # Output body with proper indentation
                        if (entry_body != "") {
                            gsub(/^[ \t]+|[ \t]+$/, "", entry_body)
                            if (length(entry_body) > 0) {
                                # Split body into paragraphs and format each
                                num_paragraphs = split(entry_body, paragraphs, /\n[ \t]*\n/)
                                for (i = 1; i <= num_paragraphs; i++) {
                                    para = paragraphs[i]
                                    # Check for special content BEFORE stripping whitespace
                                    preserve_indent = (para ~ /^[ \t]*( {4}|\t)/ || para ~ /`{3}/ || para ~ /\[[^\]]+\]\([^)]*\)|https?:\/\/\S+/)
                                    
                                    if (!preserve_indent) {
                                        # Only strip whitespace for normal text
                                        gsub(/^[ \t]+|[ \t]+$/, "", para)
                                    } else {
                                        # For code/structured content, only strip trailing whitespace
                                        gsub(/[ \t]+$/, "", para)
                                    }
                                    
                                    if (length(para) > 0) {
                                        # Skip wrapping for code blocks or lines with links/inline code
                                        if (preserve_indent) {
                                            print "  " para
                                            if (i < num_paragraphs) print ""
                                            continue
                                        }
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
                                        if (i < num_paragraphs) print ""
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

# Function to categorize and output entries
def process_and_output_categorized_entries(entries, output_lines):
    if not entries:
        return
        
    # Categorize all entries
    added_entries = []
    changed_entries = []
    removed_entries = []
    fixed_entries = []
    deprecated_entries = []
    security_entries = []
    
    for entry in entries:
        commit_line = entry.lower()
        
        # ADDED: New features, new functionality (check first to prioritize)
        if (entry.startswith('- **Add') or entry.startswith('- **Implement') or 
            entry.startswith('- **Introduce') or entry.startswith('- **New') or
            '**Adds ' in entry or '**Added ' in entry or '**Implementing ' in entry or
            '**feat: Add' in entry or '**feat: Implement' in entry or
            'add new' in commit_line or 'adds new' in commit_line or
            entry.count('**') == 2 and any(word in commit_line for word in 
                ['add ', 'adds ', 'implement', 'introduce', 'new feature'])):
            added_entries.append(entry)
        
        # REMOVED: Removed functionality (be very specific to avoid false positives)
        elif (entry.startswith('- **Remove') or entry.startswith('- **Delete') or
              entry.startswith('- **Drop') or '**Removes ' in entry or
              '**Deleted ' in entry or '**Dropped ' in entry or
              (any(word in commit_line for word in ['remove ', 'removes ', 'removed ', 'delete ', 'deletes ', 'deleted ', 'drop ', 'drops ', 'dropped ']) and 
               not 'adds' in commit_line.split()[0])):
            removed_entries.append(entry)
        
        # CHANGED: Changes to existing functionality (includes MSRV bumps, refactoring, updates)
        # Check this BEFORE Fixed section to prioritize MSRV bumps as changes
        elif (entry.startswith('- **Bump') or entry.startswith('- **Update') or
              entry.startswith('- **Refactor') or entry.startswith('- **Change') or
              'msrv' in commit_line or 'minimum supported rust version' in commit_line or
              ('bump' in commit_line and ('version' in commit_line or 'msrv' in commit_line or 'rust' in commit_line)) or
              'refactor' in commit_line or 'update' in commit_line):
            changed_entries.append(entry)
        
        # FIXED: Bug fixes, robustness improvements, and error handling
        # Exclude entries that contain 'bump' + version/msrv/rust to avoid misclassification
        elif (entry.startswith('- **Fix') or '**Bug' in entry or
              '**Patch' in entry or '**Resolve' in entry or
              'fixes ' in commit_line or 'fixed ' in commit_line or
              'bug' in commit_line or 'patch' in commit_line or
              'addresses' in commit_line and ('error' in commit_line or 'issue' in commit_line or 'problem' in commit_line) or
              ('robustness' in commit_line and not ('bump' in commit_line and ('version' in commit_line or 'msrv' in commit_line or 'rust' in commit_line))) or
              'stability' in commit_line or
              'fallback' in commit_line or 'degenerate' in commit_line or
              'precision' in commit_line or 'numerical' in commit_line or
              'error handling' in commit_line or 'consistency check' in commit_line or
              (entry.count('**') == 2 and any(word in commit_line for word in 
                ['fix ', 'fixes ', 'fixed ', 'bug', 'patch', 'resolve', 'error', 'issue', 
                 'fallback', 'degenerate', 'precision']) and not ('bump' in commit_line and ('version' in commit_line or 'msrv' in commit_line or 'rust' in commit_line)))):
            fixed_entries.append(entry)
        
        # Default fallback for anything else
        else:
            changed_entries.append(entry)
    
    # Track if we've output any sections to know when to add breaks
    any_sections_output = False
    
    # Output entries in Keep a Changelog order with proper spacing
    if added_entries:
        if any_sections_output:
            output_lines.append('')  # Blank line before section if not first
        output_lines.append('### Added')
        output_lines.append('')
        for entry in added_entries:
            output_lines.append(entry)
        any_sections_output = True
        
    if changed_entries:
        if any_sections_output:
            output_lines.append('')  # Blank line before section if not first
        output_lines.append('### Changed')
        output_lines.append('')
        for entry in changed_entries:
            output_lines.append(entry)
        any_sections_output = True
        
    if deprecated_entries:
        if any_sections_output:
            output_lines.append('')  # Blank line before section if not first
        output_lines.append('### Deprecated')
        output_lines.append('')
        for entry in deprecated_entries:
            output_lines.append(entry)
        any_sections_output = True
        
    if removed_entries:
        if any_sections_output:
            output_lines.append('')  # Blank line before section if not first
        output_lines.append('### Removed')
        output_lines.append('')
        for entry in removed_entries:
            output_lines.append(entry)
        any_sections_output = True
        
    if fixed_entries:
        if any_sections_output:
            output_lines.append('')  # Blank line before section if not first
        output_lines.append('### Fixed')
        output_lines.append('')
        for entry in fixed_entries:
            output_lines.append(entry)
        any_sections_output = True
        
    if security_entries:
        if any_sections_output:
            output_lines.append('')  # Blank line before section if not first
        output_lines.append('### Security')
        output_lines.append('')
        for entry in security_entries:
            output_lines.append(entry)
        any_sections_output = True

# Read the input file
with open('$input_file', 'r') as f:
    lines = f.readlines()

output_lines = []
in_changes_section = False
in_fixed_issues = False
in_merged_prs_section = False

# Collect entries to categorize
categorize_entries_list = []

line_index = 0
while line_index < len(lines):
    line = lines[line_index].rstrip()
    
    # Track when we enter Changes section (from template or Keep a Changelog format)
    if re.match(r'^### *(Changes|Changed)$', line):
        # Process any pending entries from previous section
        if categorize_entries_list:
            process_and_output_categorized_entries(categorize_entries_list, output_lines)
            categorize_entries_list = []
            
        in_changes_section = True
        in_fixed_issues = False
        # Skip the original Changes header - we'll create proper sections
        line_index += 1
        continue
    
    # Track when we enter Fixed Issues section (from template or Keep a Changelog format)
    elif re.match(r'^### *(Fixed|Fixed Issues)$', line):
        # Process any pending entries from previous section
        if categorize_entries_list:
            process_and_output_categorized_entries(categorize_entries_list, output_lines)
            categorize_entries_list = []
            
        in_fixed_issues = True
        in_changes_section = False
        in_merged_prs_section = False
        # Skip the original Fixed Issues header - we'll create proper sections
        line_index += 1
        continue
    
    # Track when we enter Added section (Keep a Changelog format)
    elif re.match(r'^### *Added$', line):
        # Process any pending entries from previous section
        if categorize_entries_list:
            process_and_output_categorized_entries(categorize_entries_list, output_lines)
            categorize_entries_list = []
            
        in_changes_section = True  # Treat as changes section for processing
        in_fixed_issues = False
        in_merged_prs_section = False
        # Skip the original Added header - we'll create proper sections
        line_index += 1
        continue
    
    # Track when we enter Removed section (Keep a Changelog format)
    elif re.match(r'^### *Removed$', line):
        # Process any pending entries from previous section
        if categorize_entries_list:
            process_and_output_categorized_entries(categorize_entries_list, output_lines)
            categorize_entries_list = []
            
        in_changes_section = True  # Treat as changes section for processing
        in_fixed_issues = False
        in_merged_prs_section = False
        # Skip the original Removed header - we'll create proper sections
        line_index += 1
        continue
    
    # Track when we enter Deprecated section (Keep a Changelog format)
    elif re.match(r'^### *Deprecated$', line):
        # Process any pending entries from previous section
        if categorize_entries_list:
            process_and_output_categorized_entries(categorize_entries_list, output_lines)
            categorize_entries_list = []
            
        in_changes_section = True  # Treat as changes section for processing
        in_fixed_issues = False
        in_merged_prs_section = False
        # Skip the original Deprecated header - we'll create proper sections
        line_index += 1
        continue
    
    # Track when we enter Security section (Keep a Changelog format)
    elif re.match(r'^### *Security$', line):
        # Process any pending entries from previous section
        if categorize_entries_list:
            process_and_output_categorized_entries(categorize_entries_list, output_lines)
            categorize_entries_list = []
            
        in_changes_section = True  # Treat as changes section for processing
        in_fixed_issues = False
        in_merged_prs_section = False
        # Skip the original Security header - we'll create proper sections
        line_index += 1
        continue
    
    # Track when we enter Merged Pull Requests section
    elif re.match(r'^### *Merged Pull Requests$', line):
        # Process any pending entries from previous section
        if categorize_entries_list:
            process_and_output_categorized_entries(categorize_entries_list, output_lines)
            categorize_entries_list = []
            
        in_merged_prs_section = True
        in_changes_section = False
        in_fixed_issues = False
        # Add the Merged Pull Requests header to output (we keep this section)
        output_lines.append(line)
        line_index += 1
        continue
    
    # Process collected entries before starting a new release or ending
    elif ((re.match(r'^## ', line) and (in_changes_section or in_fixed_issues or in_merged_prs_section)) or 
          (line_index == len(lines) - 1)):
        
        # Process any pending entries
        if categorize_entries_list:
            process_and_output_categorized_entries(categorize_entries_list, output_lines)
            categorize_entries_list = []
            
        # Stop processing current section
        in_changes_section = False
        in_fixed_issues = False
        in_merged_prs_section = False
        
        # Add the release header to output if it's a new release
        if re.match(r'^## ', line):
            output_lines.append('')  # Add blank line before new release
            output_lines.append(line)
        else:
            # Not a release header, add normally
            output_lines.append(line)
        line_index += 1
    
    # Process commit lines in Changes or Fixed Issues sections
    elif (in_changes_section or in_fixed_issues) and re.match(r'^- \*\*', line):
        # Store this commit entry for later processing
        # We'll categorize all entries first, then output them in correct sections
        
        # Collect the commit title line
        current_entry = [line]
        
        # Look ahead to collect any indented body content
        next_line_index = line_index + 1
        while (next_line_index < len(lines) and 
               (lines[next_line_index].strip() == '' or  # Empty line
                lines[next_line_index].startswith('  '))):  # Indented body content
            current_entry.append(lines[next_line_index].rstrip())
            next_line_index += 1
        
        # Join the entry (title + body) and add to categorization list
        categorize_entries_list.append('\n'.join(current_entry))
        
        # Skip to after the lines we've processed
        line_index = next_line_index
        continue
    
    else:
        # Skip PR description content (indented lines ONLY within Merged Pull Requests section)
        if in_merged_prs_section and re.match(r'^  ', line):  # Lines starting with 2+ spaces (PR descriptions)
            line_index += 1
            continue  # Skip PR description lines only in Merged PRs section
        
        # Print all other lines normally (headers, merged PRs, legitimate indented content, etc.)
        output_lines.append(line)
        line_index += 1

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
