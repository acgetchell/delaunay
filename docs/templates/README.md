# Templates

This directory contains templates used for automated documentation and changelog generation.

## Files

### `changelog.hbs` (legacy)

Legacy Handlebars template for [`auto-changelog`](https://github.com/CookPete/auto-changelog).

This repository has migrated changelog generation to [`git-cliff`](https://github.com/orhun/git-cliff)
using `cliff.toml` at the repository root (invoked via `just changelog` /
`uv run changelog-utils generate`).

The file is kept for reference while the migration stabilizes.
