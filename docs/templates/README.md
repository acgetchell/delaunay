# Templates

This directory contains templates used for automated documentation and changelog generation.

## Files

### `changelog.hbs`

Custom Handlebars template for auto-changelog that displays commit dates instead of tag creation dates.

**Usage:** Referenced by `.auto-changelog` configuration and used by `./scripts/generate_changelog.sh`

**Purpose:** Provides more accurate release dating that reflects when development work was completed rather than when git tags were created.

**Format:** Compatible with [Keep a Changelog](https://keepachangelog.com/) standard, showing dates in YYYY-MM-DD format.

## Usage Examples

### Using auto-changelog with the custom template

```bash
# Generate changelog using the custom template (recommended)
./scripts/generate_changelog.sh

# Or use npx auto-changelog directly with the custom template
npx auto-changelog --template keepachangelog

# Generate changelog for a specific version
npx auto-changelog --latest-version v0.3.4

# Include unreleased changes
npx auto-changelog --unreleased

# Preview changes without writing to file
npx auto-changelog --stdout

# Limit commits per release for cleaner output
npx auto-changelog --commit-limit 10
```

### Template Configuration

The `changelog.hbs` template is referenced in the `.auto-changelog` configuration file and provides:

- **Accurate dating**: Shows commit dates rather than tag creation dates
- **Keep a Changelog format**: Follows the standard changelog format
- **Semantic versioning**: Compatible with conventional commits
- **Release grouping**: Organizes changes by version with proper headings
