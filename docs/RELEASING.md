# Releasing delaunay

This guide documents the release flow for `vX.Y.Z`: prepare a dedicated
release PR, merge it, create the final annotated tag from the generated
changelog, publish to crates.io, and create the GitHub release.

The release changelog is generated with `git-cliff --tag` through
`just changelog-unreleased`, so no temporary local tag is needed.

---

## Conventions and environment

Set these variables to avoid repeating the version string:

```bash
# tag has the leading v, version does not
TAG=vX.Y.Z
VERSION=${TAG#v}
```

Verify your git remotes:

```bash
git remote -v
```

Ensure your local `main` is up to date before beginning:

```bash
git checkout main
git pull --ff-only
```

---

## Step 1: Create a clean release PR

This PR should primarily include version bumps, changelog updates, benchmark
summary updates, and documentation updates. All major code changes should
already be on `main`.

Small, critical fixes discovered during the release process may be included,
but keep them minimal and release-critical.

1. Create the release branch

```bash
git checkout -b "release/$TAG"
```

2. Bump versions

Preferred, if `cargo-edit` is installed:

```bash
cargo set-version "$VERSION"
```

Alternative: edit `Cargo.toml` manually and update `version = "..."` under
`[package]`.

Review version references in documentation:

```bash
rg -n "\bv?[0-9]+\.[0-9]+\.[0-9]+\b" README.md docs/ || true
```

3. Generate the release changelog

```bash
# Generates CHANGELOG.md as though TAG already exists, then applies
# markdown hygiene and archives completed minor release series.
just changelog-unreleased "$TAG"
```

`just changelog-unreleased` runs
`GIT_CLIFF_OFFLINE=true git-cliff --tag "$TAG" -o CHANGELOG.md`, then
`postprocess-changelog`, then `archive-changelog`. The root changelog keeps
Unreleased plus the active minor series; older completed minor series live
under `docs/archive/changelog/`.

4. Refresh generated performance results, if needed

Run this when the release PR updates public performance claims or
`benches/PERFORMANCE_RESULTS.md`:

```bash
just bench-perf-summary
```

Run it after the version bump so the generated file reports the current Cargo
package version. The recipe already runs `benchmark-utils generate-summary
--run-benchmarks --profile perf`, which refreshes the perf-profile Criterion
`ci_performance_suite` data, captures the generated construction simplex
counts used by the Triangulation Data Structure tables, reruns the circumsphere
benchmark, and regenerates `benches/PERFORMANCE_RESULTS.md`. No separate
Criterion refresh step is required.

Review generated benchmark docs before committing them. Confirm the file shows
`Version $VERSION Results`, `Current Criterion Run Information`, and
`Simplices Generated`, and does not retain `Historical Version Comparison`,
`Circumsphere Predicate Analysis`, `Method Disagreements`, or
`Baseline Artifact Information`.

5. Validate the release branch

```bash
just ci
just publish-check
```

6. Stage and commit release artifacts

```bash
git add Cargo.toml Cargo.lock CHANGELOG.md CITATION.cff docs/ benches/PERFORMANCE_RESULTS.md

git commit -m "chore(release): release $TAG

- Bump version to $TAG
- Update changelog with latest changes
- Update documentation for release
- Refresh release benchmark summary"
```

If no benchmark summary changed, omit `benches/PERFORMANCE_RESULTS.md` from
the staged files and omit the benchmark bullet.

7. Push the branch and open a PR

```bash
git push -u origin "release/$TAG"
```

PR metadata:

- Title: `chore(release): release $TAG`
- Description: Clean release PR with version bump, changelog, benchmark
  summary if applicable, and documentation updates. No feature work.

### Handling fixes discovered during the release process

If you discover issues after generating the changelog:

1. For critical fixes that must be in this release, make and commit the fix,
   then regenerate the release changelog:

   ```bash
   just changelog-unreleased "$TAG"
   git add CHANGELOG.md docs/archive/changelog/
   git commit -m "docs: update changelog with release fixes"
   ```

2. For non-critical fixes, document them as known issues in the release notes
   or include them in the next release.

---

## Step 2: After the PR is merged into main

1. Sync your local `main` to the merge commit

```bash
git checkout main
git pull --ff-only
```

2. Create the final annotated tag using the changelog content

```bash
# Creates the annotated tag from the matching CHANGELOG.md section.
# Archived versions are read from docs/archive/changelog/ automatically.
# For large changelogs (>125KB), the tag message points to the changelog
# section instead of embedding the full content.
just tag "$TAG"
```

3. Optional: verify the tag message content

```bash
git tag -l --format='%(contents)' "$TAG"
```

4. Push the tag

```bash
git push origin "$TAG"
```

5. Confirm CI baseline artifact generation

Pushing a version tag triggers `.github/workflows/generate-baseline.yml`,
which generates a performance baseline artifact named
`performance-baseline-$TAG` with dots replaced by underscores, for example
`performance-baseline-v0_6_2`.

If you need to regenerate a missing or expired baseline artifact for a tag:

```bash
gh workflow run generate-baseline.yml -f tag="$TAG" --ref main
```

6. Publish to crates.io

```bash
cargo publish --locked
```

7. Create the GitHub release with notes from the tag annotation

```bash
gh release create "$TAG" --notes-from-tag
```

8. Clean up the release branch

```bash
git push origin --delete "release/$TAG"
git branch -d "release/$TAG"
```

---

## Notes and tips

- Do not create a temporary local release tag for changelog generation; use
  `just changelog-unreleased "$TAG"`.
- Keep the release PR scoped to version, changelog, archive, benchmark summary,
  and documentation changes.
- `just changelog` regenerates the current changelog from existing tags and may
  update `docs/archive/changelog/`.
- `just changelog-unreleased "$TAG"` is for release PR preparation before the
  final tag exists.
- `just tag "$TAG"` is for the final post-merge annotated tag.
- If multiple files reference the version, confirm all of them are updated
  consistently.
