# Releasing delaunay

Prepare each `vX.Y.Z` release in a dedicated PR. After that PR is merged,
create the annotated tag and a draft GitHub release. Start `Release Benchmarks`
to attach the durable baseline, then review and publish the draft with
`just release-publish`. The published release starts `Publish to crates.io`;
approve its `crates-io` deployment to permit the Cargo upload.

The local benchmarks used to prepare the release PR do not require a draft.
The draft is needed later as the upload destination for the benchmark archive
produced on GitHub Actions.

The changelog is generated for the target tag before the tag exists, so the
release process does not require a temporary local tag.

Release recipes are content-idempotent where their inputs permit it. Repeating
`update-version` on the same UTC date or repeating `performance-readme` for the
same retained bundle produces no file changes. Rerunning `update-version` on a
later UTC date intentionally advances the declared release date. Crossing
midnight alone requires no update: the `declared` date policy allows preparation
and tagging on different UTC days when citation and changelog dates agree.
Repeated `performance-release` runs can produce different measurements.
With unchanged history and configuration, repeating `changelog-unreleased`
reuses the declared citation date even after UTC midnight.

## Contents

- [Reusable CI evidence](#reusable-ci-evidence)
- [Prepare the environment](#prepare-the-environment)
- [Step 1: Prepare the release PR](#step-1-prepare-the-release-pr)
- [Step 2: Publish after the PR is merged](#step-2-publish-after-the-pr-is-merged)
- [Configure or verify Trusted Publishing](#configure-trusted-publishing-before-the-next-release)

## Reusable CI evidence

The publishing process uses the installed `research-repo-tools==0.2.3`
validation helpers. `tooling/python/release_artifacts.py` supplies Delaunay's
Cargo policy. After all three native `just ci` jobs succeed on a protected
`main` push, `just release-package` builds and verifies clean default and
all-feature packages without upload credentials. CI retains the verified
`.crate` and `validation.json` as `cargo-package-ATTEMPT` for 30 days.

`just release-evidence` verifies this retained evidence for the clean current
checkout. It requires the exact successful main-push run, all required jobs and
steps from the same attempt, and one nonexpired artifact retrieved by immutable
ID. It validates the ZIP digest and inventory, manifest, crate digest, source
commit and version without running Cargo. It works from macOS, Linux or Windows;
the producer's recorded tool identities need not match the inspecting host.
PR, manual-dispatch and partial-rerun results cannot replace main-push evidence.

The release workflow additionally requires a matching published release event
and byte-identical repackaging with the producer's tools, before and after
deployment approval. Its selected run, attempt, jobs and artifact pass directly
through `GITHUB_OUTPUT` using the shared helper; shell stdout is only a display
channel. Local evidence inspection cannot substitute for these workflow gates.
Release metadata, exact-commit SARIF checks, durable Criterion evidence,
immutable release state, registry absence and human approval remain required.

The release commands share la-stack's operator-facing roles:

| Recipe | Purpose |
| --- | --- |
| `changelog-unreleased TAG [DATE]` | Generate prospective notes using the prepared citation date by default |
| `changelog-preview [--tag TAG] [--date DATE]` | Preview the same generation without publishing files |
| `performance-release [CURRENT BASELINE]` | Measure, retain and promote release performance evidence |
| `performance-readme` | Publish the README snapshot from validated, promoted measurements |
| `release-package` | Build and verify clean Cargo packages without uploading |
| `release-evidence [IDENTITY]` | Verify retained main CI evidence, optionally requiring a previous selection |
| `release-publish TAG` | Check notes and CI evidence, then approve the reviewed GitHub draft |
| `release-verify TAG` | Verify GitHub assets and the exact crates.io version |

Delaunay retains its scientific measurement adapter and default/all-feature
coverage. Its performance evidence format and explicit-pair repair behavior
remain documented below; la-stack has a separate measurement harness.

## Prepare the environment

Set the target tag once:

```bash
TAG=vX.Y.Z
```

Verify GitHub authentication and the repository remotes, then synchronize
`main`:

```bash
gh auth status
git remote -v
git switch main
git pull --ff-only
```

Install or verify the declared development tools before running maintenance
recipes. Shared setup installs managed tools and explicitly synchronizes `dev`;
see [development commands](dev/commands.md#justfile-usage) for prerequisites:

```bash
just setup
```

Refresh Cargo dependency requirements, exact direct Python development-tool
pins, lockfiles, and repository-owned Cargo tool pins before creating the
release branch:

```bash
just update
```

`just update` upgrades uv through its installation owner and managed Cargo tool
pins before updating dependencies. It advances exact direct dev pins, retains the
shared tooling constraint, refreshes the full Python lock, and syncs dev with
managed Rust. Cargo updates retain both the root and checkpoint fixture resolution
roots. Failures stop later steps; completed updates remain for review and retry.

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
- `CITATION.cff` version and the current UTC `date-released`;
- README dependency examples, preserving active navigation on `main`;
- active documentation dependency and `cargo add` examples; and
- the target changelog heading date when that generated heading already
  exists.

The complete edit is planned before any file is replaced and caught failures
roll every changed file back. The recipe preserves the Zenodo all-versions
concept DOI `10.5281/zenodo.16931097`; it does not add or rotate a
version-specific DOI. It does not generate the changelog, run benchmarks, or
upgrade dependencies.

Review the resulting metadata diff before continuing.

### 3. Generate the release changelog

```bash
just changelog-unreleased "$TAG"
```

This generates `CHANGELOG.md` as though the target tag already existed,
archives completed minor series under `docs/archives/changelog/`, and
uses `date-released` already recorded in `CITATION.cff`. No `RELEASE_DATE`
shell variable is needed. With this default-date path, the citation version must
match the target tag and the date must be valid `YYYY-MM-DD`; invalid metadata
stops generation before files change. `changelog-release` is the same recipe
name supported by la-stack.

Use `just changelog-preview --tag "$TAG"` for a read-only preview with the same
date default. An explicit date is still available as the second argument to
`changelog-unreleased`, or as `--date YYYY-MM-DD` for the preview. An explicit
date bypasses the adapter's citation lookup and version check; the shared
generator still validates the supplied date. To advance the declared date
deliberately, run `just update-version "$TAG" --date YYYY-MM-DD`
before regenerating. Run the final version check to verify the heading matches
Cargo metadata and the citation date. Review the generated changelog and archive
changes; do not edit them manually.

### 4. Generate release performance measurements

Fresh comparisons establish workload identity from recorded harness and
measurement-plan evidence. Benchmark fixtures must pass their scientific
checks before timings are used. If the baseline fails those checks or its
harness differs, generate absolute results with `just bench-perf-summary` and
skip comparison/README publication. Historical reports keep their original
labels and measurements.

Run this after the package version has been updated:

```bash
just performance-release
```

The no-argument form compares the current package version with the
previous stable published release. It runs the release-signal Criterion measurements,
retains `target/bench-reports/performance.{md,comparison.json,evidence.json}`, validates
the comparison JSON/evidence pair after reloading it, promotes `docs/performance.md`, and
archives the prior report plus the exact promoted evidence under
`docs/archive/performance/`.

For each measured revision, the local comparison workflow gives each of the five
curated benchmark targets separate two-hour discovery and sampling ceilings,
plus a ten-minute preflight ceiling. The sampling child has a ten-hour ceiling;
discovery and preflight run before it, with no combined local workflow deadline.
These ceilings accommodate slow higher-dimensional
measurements; they are not expected runtimes. A timeout in any target means the
measurement failed: rerun the complete `just performance-release` command
instead of promoting a partial run.

The temporary current worktree includes staged and unstaged tracked changes
and nonignored new files through the shared snapshot API. Keep source inputs
stable during capture and measurement. Do not run `just clean` or `cargo clean` until
the retained inputs have been reviewed and the README publication succeeds.

If measurement succeeded, the retained comparison JSON/evidence pair reload-validates,
and only report promotion must be retried, use `just performance-doc`. It reads
that pair and runs no Cargo benchmarks or measurement worktrees. Explicit
current/baseline tag pairs are reserved for repair paths and must always be
supplied together.

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
comparison JSON/evidence pair under `docs/assets/bench/`. README links are pinned to
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
`research-repo-tools release check --final-release` gate, requiring exactly one current
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
`CITATION.cff`, `CHANGELOG.md`, `README.md`, `docs/performance.md`, and generated
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
configuration:

Rerun `just performance-release "$TAG" "vA.B.C"` with the same baseline,
then rerun `just performance-readme`. When the release has only absolute
measurements, rerun `just bench-perf-summary` instead.

Rerun `just changelog-unreleased "$TAG"`, review and stage only the regenerated
release outputs, commit that update separately, and rerun the final release and
publish gates.

For a non-critical fix, file an issue and defer it. Do not hand-edit the
generated changelog to add a known-issue note.

Retries preserve the declared release date unless the maintainer deliberately
advances it with `update-version` and regenerates the matching changelog.

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
125 kB, the annotation points to that section instead of embedding it.

### 3. Push the tag

```bash
git push origin "$TAG"
```

### 4. Verify the release commit

```bash
just release-evidence
```

Require successful native `CI` builds on Linux, macOS and Windows, plus Clippy
and repository-rule SARIF checks, for this exact commit before proceeding.
These retain the declared Rust 1.99.1 baseline, Delaunay's numerical and
topological regressions, feature coverage, examples and benchmark compilation.
The publishing policy in `pyproject.toml` binds all five check names to GitHub
Actions app ID `15368`. CodeRabbit and human review remain protected-PR
requirements; CodeRabbit does not run on the merged `main` commit. Missing,
pending, cancelled or failed native checks block publication. Also require the
main-push package producer to have completed in that same attempt.
Manual CI dispatch cannot supply promotion evidence. If the original main-push
run failed, rerun all its jobs within GitHub's 30-day rerun window. If no eligible
run exists, land a new reviewed commit that triggers main-push CI before tagging
and publishing it; do not move an already published tag.

The main CI producer verifies clean default and all-feature packages and resolves
the required published `la-stack` dependency from crates.io. Local evidence
inspection downloads and checks its retained artifact without rebuilding. The local
`publish-check` permits dirty release-PR artifacts; `release-package` and the
upload workflow never use `--allow-dirty`.

### 5. Create the draft GitHub release

Create an unpublished release where the workflow can attach its benchmark archive:

```bash
NOTES="$(mktemp)"
just release-notes "$TAG" > "$NOTES"
gh release create "$TAG" --verify-tag --draft --title "$TAG" --notes-file "$NOTES"
rm "$NOTES"
```

Keep the release title identical to the tag, including its leading `v`.
Leave it as a stable draft: publishing now would make the release immutable
before the benchmark archive can be attached. GitHub recommends this
[draft-first sequence for immutable releases](https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases).

### 6. Run benchmarks and review the draft

Start the workflow from `main`, passing the release tag as its input:

```bash
gh workflow run release-benchmarks.yml --ref main -f tag="$TAG"
```

Once GitHub accepts the command, you can close the terminal and leave the
workflow running. Check its status later on the
[Release Benchmarks Actions page](https://github.com/acgetchell/delaunay/actions/workflows/release-benchmarks.yml).

Delaunay's adapter preserves the trusted runtime from `main`, captures the draft
release ID and peeled tag commit, then checks out that exact release commit.
This differs from la-stack's tag-dispatched, split-job benchmark workflow.
The workflow must already be merged to `main`.

The workflow automatically:

1. Preflights and runs the curated release-signal benchmarks for the tagged commit.
2. Uploads `delaunay-$TAG-criterion-baseline.tar.gz` to the draft and verifies its
   SHA-256 digest.
3. Leaves the draft and verified archive ready for maintainer review.

Before benchmarking, it rejects missing, prerelease, published, immutable or
mismatched targets. It rechecks the captured release ID and tag commit before
attaching the archive. Do not edit the draft, move the tag, or publish manually
while the workflow is running. Runs for the same tag are serialized without
cancelling an active run.

A short-lived Actions artifact is not a substitute for the release asset.
The archive binds the clean tag to its source revision, measurement commands,
toolchain, host, normalized plan, completed benchmark targets, Criterion samples
and content digest. `just performance-github-assets "$TAG" "vA.B.C"` compares
stored archives from separate sessions using absolute intervals without ratios.

### 7. Publish the reviewed draft and approve Cargo publication

After the workflow completes, open the repository's
[Releases page](https://github.com/acgetchell/delaunay/releases). Review the
draft, generated notes and `delaunay-$TAG-criterion-baseline.tar.gz` asset.
From the clean tagged checkout, using your human GitHub CLI session, run:

```bash
just release-publish "$TAG"
gh run list --workflow publish-crates.yml --limit 5
```

This command authorizes GitHub Release publication. The shared gate checks the
clean checkout, stable tag/package metadata, remote tag commit, protected default
branch ancestry, successful exact-commit checks and required baseline asset.
The local recipe first requires retained main CI evidence through
`release-evidence`, and the release title and body must match the tag and
generated notes. Human authentication is essential: publishing with a workflow's
`GITHUB_TOKEN` does not trigger the separate `release: published` workflow.

The publishing workflow checks out `github.sha`, rechecks the published event's
repository, release ID, tag/ref/commit and evidence, and requires GitHub's immutable
state. `just release-artifact verify` selects the successful trusted `main`-push
CI run for that full commit. It requires all three native CI jobs and the package
producer to have completed their required steps successfully in the same attempt.
It downloads the retained Cargo package by immutable artifact ID, verifies the ZIP
digest and exact inventory, and checks its source, version, tool identities and
archive digest. The nested Cargo archive is limited to 64 MiB expanded, 4,096
regular files, and 1 MiB per metadata file; ambiguous paths and links are rejected.
Download errors redact signed URL query credentials from diagnostics.
Repackaging with `cargo package --locked --no-verify` must produce
identical bytes; no package build or second native test matrix runs at release time.
After the `verify` job succeeds and the `publish` job is waiting for approval,
approve the `crates-io` environment deployment in GitHub.
The upload job rechecks the gate and requires proven absence of the exact registry
version after approval. It revalidates the frozen run/attempt, job IDs and artifact
identity selected by `verify`, downloads that same artifact, and repeats the byte
comparison before obtaining a short-lived OIDC token. Only that job has
`id-token: write`; it has no repository write permission or dependency caches.
Uploads for `delaunay` are serialized without cancellation.

The OIDC-enabled upload job installs only the pinned uv, the declared minimal
Rust toolchain, and the locked `tooling` Python group. It does not run the full
development setup, install the local Python project, or install managed Cargo tools.

Cargo's upload command still repackages the same clean source using
`--locked --no-verify`; stable Cargo has no prebuilt-archive upload option.
The byte comparisons establish reproducibility against the main-verified archive
before credentials are obtained. The workflow retains Cargo as the uploader and
does not claim that Cargo consumes the downloaded file directly. Build scripts do
not run with the crates.io token. See [Cargo's native publication model](https://doc.rust-lang.org/cargo/commands/cargo-publish.html).
Release benchmark measurements, captured target identity, raw Criterion evidence
and draft review retain their existing tag-specific gates.

### 8. Verify registry availability and downstream resolution

After the publishing workflow completes:

```bash
just release-verify "$TAG"
```

This requires the published stable GitHub release, nonempty durable baseline
asset and exact non-yanked crates.io version with a checksum. It retries proven
registry absence seven times at ten-second intervals, never an upload. A
published GitHub release alone is insufficient.

Check a clean consumer with a fresh Cargo home before dependent
causal-triangulations releases:

```bash
CONSUMER="$(mktemp -d)"
CARGO_HOME="$(mktemp -d)"; export CARGO_HOME
RELEASE_CRATE="delaunay@=${TAG#v}"
cargo init --vcs none --name delaunay_release_check "$CONSUMER"
cargo add --manifest-path "$CONSUMER/Cargo.toml" "$RELEASE_CRATE"
cargo check --locked --manifest-path "$CONSUMER/Cargo.toml"
cargo add --manifest-path "$CONSUMER/Cargo.toml" "$RELEASE_CRATE" --all-features
cargo check --locked --manifest-path "$CONSUMER/Cargo.toml"
```

Use a disposable shell for this check. Preserve the output and record the tag,
commit, benchmark run, publishing run, registry URL/checksum and consumer results
in #618. Keep that issue open until the next approved publication is verified.

### Recovering a failed run

Cargo promotion fails on missing, expired, ambiguous or failed evidence, partial
reruns, mismatched source/tools/bytes, and evidence changed while waiting for
approval. It never rebuilds as a fallback or substitutes PR artifacts. Inspect
the run/attempt and immutable artifact recorded in the verification step. Within
GitHub's 30-day rerun window, rerun **all jobs** of the original exact-commit
main-push CI run; individual-job and failed-jobs-only reruns cannot supply the
complete attempt proof. Then rerun the whole publishing workflow so its approval
selection is captured afresh. Evidence older than that window requires a new
reviewed main commit and a new release candidate/version, preserving published
tags and registry versions. Do not use manual CI dispatch to replace a main push.

A benchmark failure leaves the GitHub release as a draft. Crates.io upload or
post-upload verification failures occur after the GitHub release is published;
follow the published-release recovery steps below for those failures.
Rerun the benchmark workflow with the same tag after fixing the cause when no
baseline asset was uploaded. If it already exists, the adapter fails before
measurement and never overwrites it. Inspect the retained Actions archive and
draft asset; after resolving the failure, remove only that asset from the
still-mutable draft before rerunning. A published release is rejected even on a
rerun, including when publication succeeded but its response was lost. Verify
the public release in that case. Immutable releases with missing assets cannot
be repaired by this workflow.

If the crates.io upload fails or Cargo times out, inspect the original run and
exact version before retrying. The workflow checks visibility even after an
upload failure but preserves the failed job result. A visible version makes a
rerun stop before OIDC/upload; use `just release-verify` and the clean consumer
check to finish verification. A registry transport/authentication error does not
prove absence. After a visibility timeout, investigate the accepted-upload state
before authorizing a rerun for the unchanged tag; never blindly retry an upload.
Corrections to published source require a new version. Never move a published tag.

### 9. Verify the Zenodo release metadata

After the GitHub release has been archived by Zenodo, open the new
version-specific record reached from concept DOI `10.5281/zenodo.16931097`.
Check that its software version exactly matches `$TAG`, including the leading
`v`, and its publication date matches `date-released` in `CITATION.cff`.
The all-versions concept landing page alone is not evidence for the new record.
Correct a mismatch before declaring the release complete.

### 10. Remove the merged release branch

After crates.io, GitHub, benchmark-asset and Zenodo verification succeed:

```bash
git branch -d "release/$TAG"
git push origin --delete "release/$TAG"
```

## Configure Trusted Publishing before the next release

Follow the current [crates.io setup requirements](https://crates.io/docs/trusted-publishing).
Trusted Publishing requires an already-published crate and crate ownership;
initial publication uses an API token. `delaunay` already exists, so no throwaway
version or initial upload is needed. As its owner, register this exact identity
under the crate's [Settings → Trusted Publishing](https://crates.io/crates/delaunay/settings):

| Field | Value |
| --- | --- |
| Publisher | GitHub |
| Repository owner | `acgetchell` |
| Repository name | `delaunay` |
| Workflow filename | `publish-crates.yml` |
| Environment name | `crates-io` |

Choose **Add**, enter the values above, and submit **Add** on the registration
form. Use only the workflow filename, without `.github/workflows/`.
Although the form calls the environment optional, include `crates-io` to bind
this publisher to the approval gate. Verify the saved row shows all four
identity fields and owner ID `434849`. An existing matching row needs no new
registration. Adding the publisher does not publish a crate version.

Create GitHub's `crates-io` environment with required maintainer reviewers,
administrator bypass disabled, and selected deployment tags matching `v*` only
(no branches). The reviewed payloads are
[`crates-io-environment.json`](../.github/settings/crates-io-environment.json) and
[`crates-io-tags.json`](../.github/settings/crates-io-tags.json).
Use `acgetchell` (user ID 434849) as the required reviewer and
permit that maintainer to approve their own deployment; self-review prevention
needs another eligible reviewer. The stable-tag and reviewed-commit gates still
apply after deployment approval. If repository Actions policy uses an allowlist,
add only the reviewed, pinned `rust-lang/crates-io-auth-action` entry from
[`actions-selected.json`](../.github/settings/actions-selected.json) while
preserving existing entries. Publishing-workflow dependency updates require
maintainer review rather than Dependabot automatic approval. Do not add a
long-lived `CARGO_REGISTRY_TOKEN` secret.

Workflow YAML does not apply account configuration. Inspect
`gh api repos/acgetchell/delaunay/immutable-releases`; `enabled` must be `true`.
Record the actual crates.io publisher identity, GitHub environment
reviewers/tag policy, allowlist and permissions evidence in #618 before release.
Validate the source with `just check-config`, `just python-check`,
`just test-python`, `just check-docs` and `just doc-check`; PR tests use simulated
providers and never upload crates. Native CI and successful approved publication
remain separate rollout evidence.

On October 22, 2026, authenticated crates.io settings confirmed the exact
publisher above, including owner ID `434849` and environment `crates-io`.
GitHub API readback confirmed release immutability enabled, required
`acgetchell` review with self-review permitted, administrator bypass disabled,
and `v*` tags only. The Actions allowlist contains the exact reviewed
authentication-action commit. These values match the workflow on GitHub and
the artifact-promotion changes.

The crates.io **Require trusted publishing for all new versions** option was
unchecked: API-token publication remains possible. That optional account
restriction is separate from registering the publisher. This readback validates
configuration; the next approved release must still establish successful OIDC
publication and clean downstream resolution for #618. Re-read live settings
before changing them, preserving newer entries.
