# Releasing delaunay

Prepare each `vX.Y.Z` release in a dedicated PR. After that PR is merged,
create the annotated tag, publish to crates.io, create the GitHub release, and
verify its durable benchmark asset.

The changelog is generated for the target tag before the tag exists, so the
release process does not require a temporary local tag.

Release recipes are content-idempotent where their inputs permit it. Repeating
`update-version` on the same UTC day or repeating `performance-readme` for the
same retained bundle produces no file changes. `performance-release` is the
explicit exception because new measurements may vary. If publication moves to
another UTC day, rerun the release metadata and changelog steps before merge or
publication so both record the new date.

## Prepare the environment

Set the target tag once:

```bash
TAG=vX.Y.Z
```

No `VERSION`, previous-tag, or release-date variable is required. The release
tooling derives the version history from `TAG` and published GitHub Releases
and records the current UTC date when release metadata is updated.

Verify GitHub authentication and the repository remotes, then synchronize
`main`:

```bash
gh auth status
git remote -v
git switch main
git pull --ff-only
```

Install or verify the development tools before running maintenance recipes.
GitHub CLI is an external prerequisite. Setup installs the pinned Rust CLI
tools and the intentionally unpinned `cargo-update` bootstrap package that
provides `cargo-install-update` for `just update`, then verifies both `gh` and
`cargo-install-update`:

```bash
just setup
```

Refresh Cargo dependency requirements, exact direct Python development-tool
pins, lockfiles, and repository-owned Cargo tool pins before creating the
release branch:

```bash
just update
```

`just update` resolves exact pins under `[dependency-groups].dev` together for
the repository's supported Python version before refreshing `uv.lock`. It does
not change ranged development requirements, runtime or optional dependencies,
or `[build-system].requires` through that exact-pin step. Its
`cargo-install-update` preflight runs before either dependency updater can
change declarations or lockfiles.

Review any tracked dependency or tool changes and land them separately before
continuing with the release PR, then synchronize `main` again. Dependency and
tool upgrades remain independently reviewable; `just update-version`
deliberately does not run `just update`.

`update-version` and the release performance recipes use published, non-draft,
non-prerelease GitHub Releases as the authoritative stable release history.

## Step 1: Prepare the release PR

Keep the release PR focused on version metadata, the generated changelog,
benchmark artifacts, and release documentation. Major code changes should
already be on `main`.

### 1. Create the release branch

```bash
git switch -c "release/$TAG"
```

### 2. Update release metadata

```bash
just update-version "$TAG"
```

The recipe requires one stable `vX.Y.Z` target that is not older than a
published stable GitHub Release and that has at least one earlier stable
release. It infers the previous release and atomically synchronizes:

- `Cargo.toml`, `Cargo.lock`, `pyproject.toml`, and `uv.lock` package versions;
- `CITATION.cff` version and current UTC `date-released`;
- README dependency examples and non-performance tag-pinned links;
- active documentation dependency, `cargo add`, and explicit performance-pair
  examples; and
- the target changelog heading date when that generated heading already
  exists.

The complete edit is planned before any file is replaced and caught failures
roll every changed file back. The recipe preserves the Zenodo all-versions
concept DOI `10.5281/zenodo.16931097`; it does not add or rotate a
version-specific DOI. It does not generate the changelog, run benchmarks, or
upgrade dependencies.

Review the resulting metadata diff before continuing.

If the publication day changes, rerun this step and
`just changelog-unreleased "$TAG"`, then amend the release PR before merging.
Never keep a stale date merely to preserve an earlier run.

### 3. Generate the release changelog

```bash
just changelog-unreleased "$TAG"
```

This generates `CHANGELOG.md` as though the target tag already existed,
archives completed minor series under `docs/archive/changelog/`, and
synchronizes the target heading with the intended date recorded in
`CITATION.cff`. Before writing, it rejects a non-stable tag or a tag that does
not match the Cargo package version. Review the generated changelog and archive
changes. Do not edit generated changelog files manually.

### 4. Generate the release performance comparison

Run this after the package version has been updated:

```bash
just performance-release
```

The no-argument form compares the current package version with the previous
stable published release. It runs the release-signal Criterion measurements,
retains `target/bench-reports/performance.{md,csv,provenance.json}`, validates
the CSV/provenance pair after reloading it, promotes `docs/PERFORMANCE.md`, and
archives the prior report plus the exact promoted evidence under
`docs/archive/performance/`.

The temporary current worktree includes staged and unstaged changes to tracked
files but excludes untracked files. Stage any new benchmark-relevant file
before running the comparison. Do not run `just clean` or `cargo clean` until
the retained inputs have been reviewed and the README publication succeeds.

If measurement succeeded and only report promotion must be retried, use
`just performance-doc`. It reads the retained CSV/provenance pair and runs no
Cargo benchmarks or measurement worktrees. Explicit current/baseline tag pairs
are reserved for repair paths and must always be supplied together.

Treat performance output as release evidence only when the measured workflow
preserves Delaunay's numerical and topological invariants. A faster
invariant-violating run is a failed release check.

### 5. Publish the README performance snapshot

```bash
just performance-readme
```

This command consumes the validated bundle retained and promoted by
`performance-release`; it never invokes Cargo or Criterion. It atomically
updates the compact group-level table in `README.md` and the exact canonical
CSV/provenance pair under `docs/assets/bench/`. README links are pinned to
`TAG`, so they resolve to the reviewed release artifacts after the tag is
published. Caught publication failures restore every prior README-owned file.

### 6. Validate the release branch

```bash
just ci
just release-version-check
just publish-check
```

`just ci` includes lockfile, citation, documentation, Python tooling, tests,
and benchmark-compilation checks. `release-version-check` invokes the strict
`check-docs-version-sync --final-release` gate, requiring exactly one current
changelog heading whose date matches `CITATION.cff`. `publish-check` validates
crates.io metadata and dry-runs the exact package from the intentionally dirty
release worktree.

### 7. Review, stage, and commit the release artifacts

Inspect all changes before staging:

```bash
git status --short
git --no-pager diff
```

Expected release artifacts include package metadata and lockfiles,
`CITATION.cff`, `CHANGELOG.md`, `README.md`, `docs/PERFORMANCE.md`, and generated
files under `docs/archive/` and `docs/assets/bench/`. Stage only reviewed paths;
do not stage unrelated work or the entire `docs/` tree without inspecting it.

Then inspect and commit the staged release delta:

```bash
git --no-pager diff --cached

git commit -m "chore(release): release $TAG

- Bump version to $TAG
- Update citation and utility package metadata
- Generate the release changelog
- Update benchmark and performance artifacts
- Update release documentation"
```

### 8. Push the branch and open the PR

```bash
git push -u origin "release/$TAG"
```

Use `chore(release): release $TAG` as the PR title and describe the PR as a
focused release preparation without feature work.

### Handling fixes found during preparation

For a critical fix that must be included, make and commit the fix. If it changes
measured code, benchmark harnesses, toolchain or dependency inputs, or benchmark
configuration, rerun `just performance-release "$TAG" "vA.B.C"` with the same
baseline used for the release comparison, then rerun `just performance-readme`.
Rerun `just changelog-unreleased "$TAG"`, review and stage only the regenerated
release outputs, commit that update separately, and rerun the final release and
publish gates.

For a non-critical fix, file an issue and defer it. Do not hand-edit the
generated changelog to add a known-issue note.

Retries on the same UTC day keep the recorded release date unchanged. If
publication moves to a different UTC day, rerun `just update-version "$TAG"`
and `just changelog-unreleased "$TAG"`, then rerun the final release and
publish checks before merge or publication.

## Step 2: Publish after the PR is merged

### 1. Synchronize `main`

```bash
git switch main
git pull --ff-only
```

### 2. Create and verify the annotated tag

```bash
just tag "$TAG"
git --no-pager tag -l --format='%(contents)' "$TAG"
```

`just tag` first runs `release-version-check`, then builds the annotation from
the matching active or archived changelog section. For a changelog larger than
125 kB, the annotation points to that section instead of embedding it. It fails
before any Git tag query or mutation unless the `CITATION.cff` intended
publication date equals the current UTC day. If the day changed after the
original PR merged, prepare and merge a corrective release-metadata PR by
rerunning `just update-version "$TAG"` on the new UTC day, then resynchronize
local `main` and rerun the gates before tagging. Do not force the stale date
through.

### 3. Push the tag

```bash
git push origin "$TAG"
```

### 4. Publish to crates.io

```bash
cargo publish --locked
```

### 5. Create the GitHub release

```bash
gh release create "$TAG" --title "$TAG" --notes-from-tag
```

Keep the release title identical to the tag, including its leading `v`.

### 6. Verify the durable Criterion baseline

Publishing the GitHub release triggers the `Release Benchmarks` workflow.
After it completes, verify that the release contains its long-lived baseline
archive:

```bash
gh release view "$TAG" --json assets \
  --jq ".assets[] | select(.name == \"delaunay-$TAG-criterion-baseline.tar.gz\") | .name" | cat
```

The command must print `delaunay-$TAG-criterion-baseline.tar.gz`. A short-lived
Actions artifact is not a substitute for this release asset. The archive binds
the requested clean release tag to its source revision, measurement commands,
toolchain, host, normalized measurement plan, completed benchmark targets,
Criterion sample, and content digest.

Use `just performance-github-assets "$TAG" "vA.B.C"` only when comparing two
stored release archives. Because those runs are separate measurement sessions,
the resulting report retains absolute intervals but suppresses ratios.

### 7. Verify the Zenodo release metadata

After the GitHub release has been archived by Zenodo, open the new
version-specific Zenodo record reached from the repository's concept DOI. Check
that its software version exactly matches `$TAG` (including the leading `v`)
and its publication date exactly matches `date-released` in `CITATION.cff`.
Do not treat the all-versions concept
landing page alone as evidence for the new record. Correct a mismatch before
declaring the release complete.

### 8. Remove the merged release branch

After crates.io, GitHub, benchmark-asset, and Zenodo verification succeed:

```bash
git branch -d "release/$TAG"
git push origin --delete "release/$TAG"
```
