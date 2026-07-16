# Releasing delaunay

This guide documents the release flow for `vX.Y.Z`: prepare a dedicated
release PR, merge it, create the final annotated tag from the generated
changelog, publish to crates.io, and create the GitHub release.

The release changelog is generated with `git-cliff --tag` through
`just changelog-unreleased`, so no temporary local tag is needed.

Applies to versions `vX.Y.Z`. Prefer updating documentation before publishing
to crates.io.

---

## Conventions and environment

Set these variables to avoid repeating the version string:

```bash
# tag has the leading v, version does not
TAG=vX.Y.Z
VERSION=${TAG#v}
PREVIOUS_TAG=vA.B.C
```

Verify your git remotes:

```bash
git remote -v
```

Ensure your local `main` is up to date before beginning:

```bash
git switch main
git pull --ff-only
```

---

## Step 1: Create a clean release PR

This PR should primarily include version bumps, changelog updates, benchmark
summary updates, and documentation updates. All major code changes should
already be on `main`.

Finalize release-facing metadata and documentation in this dedicated release
PR. Ordinary feature, fix, review, and hygiene work should not preemptively
bump versions or prepare release artifacts.

Small, critical fixes discovered during the release process may be included,
but keep them minimal and release-critical.

Update release-facing documentation on this PR branch before publishing. Do not
defer README, `docs/`, or benchmark documentation fixes until after the release:
crates.io, docs.rs, and the generated benchmark summary are all versioned with
the release artifacts.

1. Create the release branch

```bash
git switch -c "release/$TAG"
```

2. Bump versions

Preferred, if `cargo-edit` is installed:

```bash
cargo set-version "$VERSION"
```

Alternative: edit `Cargo.toml` manually and update `version = "..."` under
`[package]`.

Update release metadata to match the crate version:

- `CITATION.cff`: update `version` and `date-released`.
- `CITATION.cff`: keep `abstract` synchronized with the first paragraph under
  `README.md`'s Introduction; update both together when either changes.
- `pyproject.toml`: update `[project] version` for the Python utility package.

Review the citation identity fields at the same time: author, ORCID,
repository, license, and DOI. Keep the DOI policy deliberate: update it only
when the release has a known release-specific DOI or the archival policy
changes.

Refresh both committed lockfiles after manual metadata edits:

```bash
cargo metadata --format-version 1 --no-deps > /dev/null
uv lock
```

Review version references in active documentation and package metadata:

```bash
just docs-version-check
```

The automated check covers package metadata, lockfiles, citation version,
release-pinned README links, active documentation dependency snippets, and
current-tag arguments in benchmark workflow examples. Historical prose,
archived reports, baseline arguments, and tool versions intentionally remain
independent of the current package version.

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

4. Refresh checked-in performance results, if needed

Per-version benchmark storage is handled by the release-published benchmark
workflow in Step 2; do not run this only to preserve release benchmark data.
Run it only when the release PR intentionally updates public performance claims
or the checked-in human-readable `benches/PERFORMANCE_RESULTS.md` summary:

```bash
just bench-perf-summary
```

Run it after the version bump so the generated file reports the current Cargo
package version. The recipe already runs `benchmark-utils generate-summary
--run-benchmarks --profile perf`, which refreshes the perf-profile Criterion
`ci_performance_suite` data, captures the generated construction simplex counts
emitted by the benchmark harness, records current Criterion run metadata,
reruns the circumsphere benchmark, and regenerates
`benches/PERFORMANCE_RESULTS.md`. No separate Criterion refresh step is
required.

Review generated benchmark docs before committing them. Confirm the file shows
`Version $VERSION Results`, `Current Criterion Run Information`, and
`Simplices Generated`, and does not retain `Historical Version Comparison`,
`Circumsphere Predicate Analysis`, `Method Disagreements`, or
`Baseline Artifact Information`. Treat benchmark output as release evidence only
when the underlying harnesses maintain their scientific invariants; a faster
invariant-violating run is release-blocking, not a publishable improvement.

When the release PR needs a curated release-to-release comparison in active
docs, run the temp-worktree promotion workflow after the version bump:

```bash
just perf-release
```

This compares the current package version against the previous stable published
release, writes the curated report to `docs/PERFORMANCE.md`, and archives the
previous curated report under `docs/archive/performance/`. To repair a specific
pair, pass both tags explicitly:

```bash
just perf-release "$TAG" "$PREVIOUS_TAG"
```

For manual investigation only, `DELAUNAY_BENCH_EXPORT_METRICS=1` can print the
construction vertex/simplex metric lines without Criterion sampling. Prefer
pairing it with a `tds_new` Criterion filter, for example:

```bash
DELAUNAY_BENCH_EXPORT_METRICS=1 \
  cargo bench --profile perf --bench ci_performance_suite -- \
  "tds_new_3d/tds_new/750"
```

This helper is not a replacement for `just bench-perf-summary` in the release
flow.

5. Validate the release branch

```bash
just docs-version-check
just cargo-lock-check
just ci
just publish-check
```

6. Stage and commit release artifacts

```bash
git add Cargo.toml Cargo.lock CITATION.cff pyproject.toml uv.lock CHANGELOG.md docs/ benches/PERFORMANCE_RESULTS.md

git commit -m "chore(release): release $TAG

- Bump version to $TAG
- Update citation and utility package metadata
- Update changelog with latest changes
- Update documentation for release
- Refresh release benchmark summary"
```

If no checked-in benchmark summary changed, omit `benches/PERFORMANCE_RESULTS.md`
from the staged files and omit the benchmark bullet.

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
git switch main
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

5. Publish to crates.io

```bash
# Publish the crate. Ensure docs are already updated on main via the PR.
cargo publish --locked
```

6. Create the GitHub release with notes from the tag annotation

```bash
gh release create "$TAG" --title "$TAG" --notes-from-tag
```

Always set the GitHub release title to the exact tag string, including the
leading `v`.

Publishing the release triggers `.github/workflows/release-benchmarks.yml`, which
runs fresh perf-profile release-signal benchmarks on `ubuntu-latest`, packages
`baseline_results.txt`, `PERFORMANCE_RESULTS.md`, raw Criterion data, and
metadata, then attaches
`delaunay-$TAG-criterion-baseline.tar.gz` to the GitHub Release. That release
asset is for GitHub Actions CI comparisons; keep local same-machine timing
baselines under the ignored `baseline-artifact/` or `baseline-artifacts/` paths.
Use `just perf-github-assets "$TAG" "vX.Y.Z"` to compare two stored
release assets without local benchmark runs.

7. Confirm release benchmark assets

After the release benchmark workflow completes, verify that the GitHub Release
has the durable benchmark baseline asset:

```bash
gh release view "$TAG" --json assets \
  --jq ".assets[] | select(.name == \"delaunay-$TAG-criterion-baseline.tar.gz\") | .name" | cat
```

The command must print `delaunay-$TAG-criterion-baseline.tar.gz`. An Actions
artifact alone is not a durable release baseline.

8. Clean up the release branch

```bash
git branch -d "release/$TAG"
git push origin --delete "release/$TAG"
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
