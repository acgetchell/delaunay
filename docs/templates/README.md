# Templates

This directory contains templates used for automated documentation and changelog generation.

## Files

### `changelog.hbs`

Custom Handlebars template for auto-changelog that displays commit dates instead of tag creation dates.

**Usage:** Referenced by `.auto-changelog` configuration and used by `./scripts/generate_changelog.sh`

**Purpose:** Provides more accurate release dating that reflects when development work was completed rather than when git tags were created.

**Format:** Compatible with Keep a Changelog standard, showing dates in YYYY-MM-DD format.
