# Templates

This directory contains templates used for automated documentation and changelog generation.

## Status

The pinned `research-repo-tools` package owns the git-cliff policy and changelog
normalization, archiving, and release-note extraction. Use `just changelog`,
`just changelog-unreleased <tag> <date>`, or `just release-notes <tag>`.

`changelog_format.toml` supplies the consumer's generated-history Markdown policy.
Historical emphasis and list indentation remain exempt from active prose rules;
archive destinations are created together by the shared transaction.
