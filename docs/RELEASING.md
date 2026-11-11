# Releasing delaunay

Prepare each `vX.Y.Z` release in a dedicated PR. Acquire the declared local
benchmark comparison once in Step 1. After merge, verify that original
acquisition against clean final main and its signed Cargo evidence before tagging.
The tag push prepares a draft. `just release-benchmarks "$TAG"` attaches the
accepted archive without collecting timings. Review and publish the draft with
`just release-publish "$TAG"`, then explicitly dispatch Cargo publication with
`just release-upload "$TAG"` and approve its `crates-io` deployment.

The original acquisition and approval records live outside the checkout.
Back up the owner directory printed by `performance-release`; Git commits and
Actions artifacts do not replace that independent record. Final publication,
native adoption evidence and the verified la-stack v0.5.3 comparison remain
tracked in [v0.8.5 release tracker #696](https://github.com/acgetchell/delaunay/issues/696).

The changelog is generated for the target tag before the tag exists, so the
release process does not require a temporary local tag.

Release recipes are content-idempotent where their inputs permit it. Repeating
`update-version` on the same UTC date or repeating `performance-readme` for the
same retained bundle produces no file changes. Rerunning `update-version` on a
later UTC date intentionally advances the declared release date. Crossing
midnight alone requires no update: the `declared` date policy allows preparation
and tagging on different UTC days when citation and changelog dates agree.
Acquisition retries require the original checkpoint; report and attachment retries reuse retained bytes.
With unchanged history and configuration, repeating `changelog-release`
reuses the declared citation date even after UTC midnight.

## Contents

- [Reusable CI evidence](#reusable-ci-evidence)
- [Prepare the environment](#prepare-the-environment)
- [Step 1: Prepare the release PR](#step-1-prepare-the-release-pr)
- [Step 2: Publish after the PR is merged](#step-2-publish-after-the-pr-is-merged)
  - [Verify the release commit before tagging](#2-verify-the-release-commit)
  - [Inspect automatic draft preparation](#5-inspect-the-prepared-draft-github-release)
- [Configure or verify Trusted Publishing](#configure-trusted-publishing-before-the-next-release)

## Reusable CI evidence

The published, locked `research-repo-tools==0.2.8` owns measurement, candidate,
archive, signature and provider verification. Delaunay's adapters own its
173-case scientific plan, independent local record selection, maintainer dispatch
and Cargo reproduction.

Main CI verifies clean default and all-feature packages after portable and native
Linux/macOS/Windows checks. The package job retains its exact `.crate` and
`validation.json` in an immutable producer handoff. A separate signer, with no
checkout or project code, authenticates that artifact's ID, digest, run and
attempt and signs the original manifest. The final `cargo-package-RUN_ID-ATTEMPT`
artifact contains the original crate, manifest and `validation.sigstore.json`
and is retained for 30 days. The guarantee remains `cargo-archive`.

`just release-dependency-builds` prepares dependency receipts separately.
`just release-evidence` only inspects existing inputs: no acquisition, scientific
preflight, package reproduction or builds. It verifies the original acquisition
digest and publication receipt, the entire candidate-to-final tree binding and
live signed native evidence. Only the two declared report outputs and exact
receipt-bound generated files may differ from Step 1.

The canonical acceptance and matching `native.json` are frozen under
`OWNER/final/COMMIT/reviewed/`. Repeating verification requires those same bytes.
The acceptance binds the acquisition, final commit, native run/attempt/artifact
and derived benchmark archive. Missing or replaced evidence fails closed.

| Recipe | Purpose |
| --- | --- |
| `changelog-release TAG [DATE]` | Generate prospective notes with the prepared citation date |
| `changelog-preview [--tag TAG] [--date DATE]` | Preview generation without publishing files |
| `performance-prepare` | Capture release/ref assertions without timing |
| `performance-release` | Acquire once, freeze owner records, promote reports |
| `performance-resume` | Resume the original checkpoint, subject to shared recovery checks |
| `performance-capture [OWNER]` | Retry original owner retention without acquisition |
| `performance-promote`, `performance-readme` | Render retained evidence without acquisition |
| `release-package` | Build and verify clean Cargo packages without uploading |
| `release-assets OPERATION TAG [ARGS...]` | Inspect/capture targets through the isolated shared API |
| `release-evidence [OWNER] [--selection NAME]` | Freeze or revalidate candidate and final-main acceptance |
| `release-benchmarks TAG [OWNER]` | Stage inert records and dispatch exact-tag attachment |
| `release-publish TAG [OWNER]` | Revalidate the original draft, acceptance and assets, then publish |
| `release-upload TAG [OWNER] [--selection NAME]` | Explicitly dispatch Cargo upload with reviewed acceptance |
| `release-verify TAG` | Verify published assets and the exact crates.io version |

Named selections are immutable review records, not a latest-success selector.
Original local acquisition identity remains distinct from the final-main
signature. No editable, sibling or Git dependency override supplies release
authority.

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
just changelog-release "$TAG"
```

This generates `CHANGELOG.md` as though the target tag already existed,
archives completed minor series under `docs/archives/changelog/`, and
uses `date-released` already recorded in `CITATION.cff`. No `RELEASE_DATE`
shell variable is needed. With this default-date path, the citation version must
match the target tag and the date must be valid `YYYY-MM-DD`; invalid metadata
stops generation before files change.

Use `just changelog-preview --tag "$TAG"` for a read-only preview with the same
date default. An explicit date is still available as the second argument to
`changelog-release`, or as `--date YYYY-MM-DD` for the preview. An explicit
date bypasses the shared citation lookup and version check; the shared
generator still validates the supplied date. To advance the declared date
deliberately, run `just update-version "$TAG" --date YYYY-MM-DD`
before regenerating. Run the final version check to verify the heading matches
Cargo metadata and the citation date. Review the generated changelog and archive
changes; do not edit them manually.

### 4. Generate release performance measurements

After updating the package version, capture independent release/ref assertions,
then measure and promote through the declared shared workflow:

```bash
just performance-prepare
just performance-release
```

`performance-release` runs benchmarks locally and can take hours. The recipe
defaults to one heartbeat every **300 seconds (five minutes)**; stage changes
and failures are reported immediately. Use `--progress-cadence` to select a
shorter or longer heartbeat interval in seconds:

```bash
just performance-release --progress-cadence 60
just performance-release --progress-cadence 600
```

Preparation selects the current package and previous stable published release
without timing, using reviewed tags already present locally. Schema 6 declares
one sequential baseline/current trial and wraps the unchanged common-harness
173-case plan with its 10/15/20/25/100 counts, native discovery and scientific gates.
This preserves the existing acquisition budget; it does not claim counterbalancing.

Read-only source exports preserve both original revisions. The current harness
must compile against both libraries. Discovery and scientific gates pass before
timing. Source, harness, workload, compiler, dependency or host changes invalidate
the acquisition. Fresh and saved `retained` benchmark, estimate, sample and Tukey
bytes must agree. Each discovery has a 30-minute ceiling; each phase gate and
timing command has a six-hour ceiling. These are limits, not expected runtimes.

Before any report promotion, the local owner freezes the preparation, complete-run
v5 payload/envelope, candidate policy with no final commit, acquisition digest,
publication receipt and its digest. By default these live under
`~/.local/state/delaunay/acquisitions/TAG/CHECKPOINT_ID/`; a separate checkout/tag
selection under `~/.local/state/delaunay/releases/` remembers the original owner.
Use `--owner-directory PATH` to choose an external location for a new acquisition.
Never reconstruct trusted assertions from an incoming archive.

The only ordinary output exceptions are `README.md` and
`docs/assets/bench/release-performance.csv`. They must remain outside current
measured inputs and both harness/workload inventories. The library includes
README only under `doc`/`doctest` configuration. Generated report/history JSON
uses the exact shared publication receipt instead of a broad directory exemption.

Successful promotion retains the complete history under `docs/performance-runs/`
and updates `docs/performance_current.md`. Historical runs keep their original
schemas and bytes; they are never upgraded into scheduled acquisitions.

If only rendering failed, run `just performance-promote`, then
`just performance-readme`. If owner capture failed, preserve all scratch and
owner files; `just performance-capture OWNER` retries retained originals.
`--retained-scratch` explicitly authorizes recovery of the original locally
controlled scratch acquisition when the callback could not copy it. Never use
that option to adopt incoming evidence. `just performance-resume` is acquisition
recovery, not a rendering retry; interrupted unretained timings can require a
new acquisition. Preserve the original diagnostic and checkpoint.

### 5. Publish the README performance snapshot

```bash
just performance-readme --preview
just performance-readme
```

The installed publisher consumes the retained complete run and independently
captured preparation assertions. Consumer TOML selects four construction rows;
the complete report retains every case. Publication updates the marked README
section and its selected CSV without invoking Cargo or Criterion. It checks
source, harness, configuration and retained selection freshness before replacing
outputs, and restores the prior generation on caught failures. Preserve any
reported recovery paths after interruption.

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
`CITATION.cff`, `CHANGELOG.md`, `README.md`, `docs/performance_current.md`, and generated
files under `docs/performance-runs/`, `docs/archives/changelog/`, and
`docs/assets/bench/`. Stage only reviewed paths;
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

Finish metadata, changelog, tooling and implementation changes before acquisition.
Whole-tree acceptance permits only declared outputs and receipt-bound generated
files afterward. A measured-input or other undeclared change requires a new
acquisition; report-only retries reuse the original run.

Preserve the previous owner directory and selection. For a corrected candidate,
use a fresh checkout and explicit external owner/checkpoint so the original
checkout/tag selection cannot be silently replaced. Prepare again, acquire,
publish the README and run the required checks. Never rewrite historical run bytes.

### Resuming an unpublished release after another PR

A corrected PR requires its own clean final-main acceptance before tagging.
Do not repeat version or dependency changes solely to resume work.
After draft preparation has captured a release ID, deleting a draft or replacing
a tag does not reset the receipt protocol. Stop and preserve the original journal;
use an explicitly reviewed recovery or a new release version. Never move a
published tag or substitute a new draft for the captured target.

## Step 2: Publish after the PR is merged

### 1. Synchronize `main`

```bash
git switch main
git pull --ff-only
test -z "$(git status --porcelain --untracked-files=all)"
RELEASE_COMMIT="$(git rev-parse HEAD)"
git --no-pager show --no-patch "$RELEASE_COMMIT"
```

Confirm that `RELEASE_COMMIT` is the exact reviewed, merged release SHA. Keep the
checkout at that commit through verification and tagging.

<a id="4-verify-the-release-commit"></a>

### 2. Verify the release commit

```bash
just release-evidence
```

Wait for successful portable checks and native `CI` builds on Linux, macOS and Windows, plus Clippy
and repository-rule SARIF checks, for this exact commit before proceeding.
These retain the declared Rust 1.99.1 baseline, Delaunay's numerical and
topological regressions, feature coverage, examples and benchmark compilation.
The publishing policy in `pyproject.toml` binds all six check names to GitHub
Actions app ID `15368`. CodeRabbit and human review remain protected-PR
requirements; CodeRabbit does not run on the merged `main` commit. Missing,
pending, cancelled or failed native checks block publication. Also require the
main-push package producer and separate signer to have completed in that same attempt.
Manual CI dispatch cannot supply promotion evidence. If the original main-push
run failed, rerun all its jobs within GitHub's 30-day rerun window. If no eligible
run exists, land a new reviewed commit that triggers main-push CI before tagging
and publishing it; do not move an already published tag.

The main CI producer verifies clean default and all-feature packages and resolves
the required published `la-stack` dependency from crates.io. Local evidence
inspection downloads and checks its retained artifact without rebuilding. The local
`publish-check` permits dirty release-PR artifacts; `release-package` and the
upload workflow never use `--allow-dirty`.

Evidence inspection works before a tag exists. Git push does not check CI.
Missing evidence fails immediately; inspect the original run before retrying.
If only native evidence must be renewed, explicitly review a new named acceptance
with `just release-evidence --selection NAME`; the old record remains intact.
A different final commit still must satisfy the original whole-tree binding. Earlier inspection does not replace publication-time and
approval-time frozen evidence checks or byte-identical Cargo reproduction.

<a id="2-create-and-verify-the-annotated-tag"></a>

### 3. Create and verify the annotated tag

```bash
test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"
just tag "$TAG"
test "$(git rev-parse "$TAG^{commit}")" = "$RELEASE_COMMIT"
git --no-pager tag -l --format='%(contents)' "$TAG"
```

`just tag` first runs `release-version-check`, then builds the annotation from
the matching active or archived changelog section. For a changelog larger than
125 kB, the annotation points to that section instead of embedding it.

<a id="3-push-the-tag"></a>

### 4. Push the tag

```bash
git push origin "$TAG"
```

<a id="5-create-the-draft-github-release"></a>

### 5. Inspect the prepared draft GitHub release

The tag push starts `Prepare GitHub release`. Wait for its successful completion:

```bash
gh run list --workflow prepare-release.yml --limit 5
gh release view "$TAG" --repo acgetchell/delaunay --json tagName,name,body,isDraft,isPrerelease
```

Inspect the [preparation run](https://github.com/acgetchell/delaunay/actions/workflows/prepare-release.yml)
and compare the draft body with `just release-notes "$TAG"`. Preparation checks
the exact tag/ref/SHA, protected main ancestry, final package/citation metadata,
generated notes, complete retained main evidence and separately required SARIF
checks. Its bounded verification job passes only inert notes, policy and native
selection to an isolated writer. The writer has no checkout, project environment,
registry token or OIDC permission.

The writer creates or captures the stable draft through RRT's receipt protocol.
Only the original tag-push attempt is eligible for automatic preparation.
Keep its `draft-receipt.json`, `target.json` and immutable artifact locator.
A failed or uncertain creation requires inspection of that journal; rerunning
a workflow does not authorize a second draft. Attachment retrieves the original
receipt and requires its native selection to match the pre-tag acceptance.

Keep the release title identical to the tag, including its leading `v`.
Leave it as a stable draft: publishing now would make the release immutable
before the benchmark archive can be attached. GitHub recommends this
[draft-first sequence for immutable releases](https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases).

<a id="6-run-benchmarks-and-review-the-draft"></a>

### 6. Attach the accepted acquisition and review the draft

From the clean tagged checkout with the original owner record:

```bash
just release-benchmarks "$TAG"
```

The command verifies the existing acceptance, retrieves the original draft
receipt, derives the deterministic archive and stages four inert assets:
policy, publication receipt, full acceptance and archive. It retains their exact
asset IDs and SHA-256 values in `transfer.json`, outside the checkout, and sends
that independent assertion in the maintainer's exact-tag workflow dispatch.

The hosted fetch job executes no checkout code with draft-read/write credentials.
A read-only verifier checks the original acquisition and full frozen acceptance.
An isolated writer authenticates the same-run/attempt artifact and verifies the
accepted bytes before attachment. No job runs benchmarks, preflight sampling or
package builds. Retries may reuse identical bytes on the same captured draft;
conflicting assets or a replaced draft stop the workflow.

Wait for [Release Benchmarks](https://github.com/acgetchell/delaunay/actions/workflows/release-benchmarks.yml)
to complete. Do not edit or publish the draft during attachment.
The canonical archive is `delaunay-$TAG-criterion-baseline.tar.gz`. Its RRT v2
layout retains `evidence/run.json`, `evidence/evidence.json`, `reuse.json` and
original current samples under `criterion/trial-1/release-signal/`.
Measured PR provenance remains distinct from final-main acceptance.

The version-aware reader uses the published immutable candidate index and
accepted archive. `just performance-github-assets "$TAG" "vA.B.C"` renders that
original declared pair when the requested baseline matches; it does not combine
unrelated scheduled acquisitions. Older metadata/flat-Criterion archives retain
their existing historical reader.

### 7. Publish the reviewed draft and approve Cargo publication

After attachment completes, review the draft, generated notes and canonical
archive on the [Releases page](https://github.com/acgetchell/delaunay/releases).
Use the clean tagged checkout and your human GitHub CLI session:

```bash
just release-publish "$TAG"
just release-upload "$TAG"
gh run list --workflow publish-crates.yml --limit 5
```

Publication revalidates the existing acceptance, original draft receipt and
attached archive, retains its ID/digest in `publication-transfer.json`, and
publishes a canonical candidate index for future historical readers. The shared
gate still requires exact source, notes, metadata, protected-main ancestry,
native/SARIF checks and required assets. The final request addresses the original
release ID, with every selected asset checked before and after publication.
The historical reader's candidate index has its own retained ID/digest assertion.
It neither selects another draft by tag nor recreates missing attachments.
An unconfirmed response requires inspection of that original release before
retrying. GitHub publication alone does not dispatch Cargo upload.

`release-upload` sends the independent publication transfer and full acceptance
digest through the maintainer's exact-tag dispatch. Before approval, a read-only
job verifies the immutable published release and freezes that complete acceptance
in an immutable Actions artifact. Approve the `crates-io` environment only after
reviewing it.

After approval the upload job authenticates that exact artifact/run/attempt,
revalidates the same acceptance and assets, and proves registry absence.
Credential-free `cargo package --locked --no-verify` must reproduce the retained
crate byte for byte with matching producer Cargo/rustc identities. A mismatch
stops before token acquisition. No benchmark or package build runs here.

The OIDC upload job keeps its minimal pinned uv/Rust and locked `tooling` setup;
it does not install the local Python project or resynchronize development groups.
Only after the checks does it request the short-lived crates.io token and run
`cargo publish --locked --package delaunay --registry crates-io --no-verify`.
Cargo repackages the same source; it does not upload the downloaded crate file
directly. Uploads remain serialized without cancellation. Visibility verification
runs even after an uncertain upload response.

The separate main signer has attestation permissions only. Draft and attachment
writers have no registry credentials; the upload job has no repository-write
permission. Delaunay retains its stronger Cargo archive guarantee while #693's
published-reference parity gate remains pending.

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
in [#696](https://github.com/acgetchell/delaunay/issues/696), including comparison of
the registry checksum with the retained main-produced crate digest. Keep that
tracker open until publication and downstream validation are complete.

### Recovering a failed run

Missing, expired, replaced, ambiguous or conflicting evidence stops promotion.
Preserve the original acquisition, acceptance, draft journal and transfer records.
Changed measurement inputs require a new acquisition. Rendering and attachment
retries reuse accepted bytes; do not delete an asset to disguise a conflict.

Owner-record writes serialize through a `.candidate-owner.lock` directory.
An existing lock stops a competing or interrupted update. Let an active writer
finish; after an interrupted process, inspect the retained records and any
reported rollback files before manually removing only that lock directory.
Recovery never replaces conflicting original assertions.

If only native evidence expires, rerun all jobs of the original eligible main-push
CI run within GitHub's rerun window, then separately review a new immutable
selection with `just release-evidence --selection NAME`. Before a draft exists,
use that selection for subsequent steps. Once a draft receipt exists, its native
selection is fixed; a different selection cannot silently replace it.
For an already published immutable release, explicit `release-upload` dispatch
may bind a separately reviewed newer native acceptance to the same original
acquisition and published assets. The full reviewed digest is checked before
approval and the exact resulting bytes are pinned afterward. Original records
and published assets remain unchanged.

An uncertain draft creation uses the original receipt's manual recovery protocol.
A published release with missing required assets cannot be repaired by attachment.

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
