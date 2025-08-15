#!/bin/bash

# Generate changelog with commit dates instead of tag dates
# This script uses auto-changelog with a custom template and post-processes
# the output to format ISO dates correctly.

set -euo pipefail

echo "Generating changelog with commit dates..."

# Change to project root directory to ensure auto-changelog finds config files
cd "$(dirname "$0")"/.. 

# Generate the changelog using our custom template and post-process the dates
npx auto-changelog --stdout | sed 's/T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]\.[0-9][0-9][0-9]Z//g' > CHANGELOG.md

echo "CHANGELOG.md has been updated with commit dates instead of tag dates."
