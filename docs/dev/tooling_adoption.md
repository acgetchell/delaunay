# Shared tooling adoption

The repository now pins the published
[research-repo-tools v0.2.4](https://github.com/acgetchell/research-repo-tools/releases/tag/v0.2.4)
PyPI distribution in runtime dependencies, the tooling/dev groups, and the
notebook extra/group. `uv.lock` records exact registry artifacts and hashes.
Python follows the shared package's 3.15 floor. Ruff, ty and pytest inherit exact
versions from the package's `python-tools` extra; no independent floating
requirements remain. `pyproject.toml` owns the exact uv declaration used for
the lock and consumer checks.

The locked Semgrep runs in the same Python 3.15 environment. The uv overrides
select patched MCP 1.30.0 and, on Windows, pywin32 312 with Python 3.15 wheels;
Semgrep's own pins still select MCP 1.29.0 and pywin32 311. Its OpenTelemetry
1.37 pins also require a compatible API/SDK/exporter 1.45.1 and instrumentation
0.66b1 override set to address [GHSA-xcq7-p3pf-p88h][otel-advisory]. Retire these
overrides when Semgrep's constraints permit the patched dependencies directly.
Native Linux and Windows execution must be verified by their CI jobs.

[otel-advisory]: https://github.com/advisories/GHSA-xcq7-p3pf-p88h

The Rust dependency is now the published la-stack 0.5.2, retaining disabled
default features and the `exact` feature. This dependency refresh makes the
upstream APIs available; the new numerical delegation is tracked by
[#654](https://github.com/acgetchell/delaunay/issues/654),
[#655](https://github.com/acgetchell/delaunay/issues/655),
[#656](https://github.com/acgetchell/delaunay/issues/656), and
[#657](https://github.com/acgetchell/delaunay/issues/657).

The dependency's polynomial errors introduce 16-byte alignment. To contain that
growth in nested triangulation errors, `CoordinateConversionError::LinearAlgebraFailure`,
`QualityError::EdgeLength`, `RandomPointGenerationError::VectorComputation`, and
`SphericalPointError::AngleComputationFailed`
now store their `source` as `Box<LaError>`. This is an intentional public field
type change: construct these fields with `Box::new(source)` and inspect the
backend variant through `source.as_ref()` or `*source`. Existing `From<LaError>`
conversions remain available. The derived `Error::source()` now exposes the
typed `Box<LaError>`, so downcast to that type before inspecting its inner error.
Allocation occurs only when constructing those errors; the scientific kernels
are unchanged.

The v0.2.4 pin also supplies the shared Windows notebook event-loop fix for
[#661](https://github.com/acgetchell/delaunay/issues/661); native Windows consumer
verification remains required. The performance and Rustdoc integrations are
documented below. The [release-tooling adoption](#v024-release-tooling-adoption)
implements [#653](https://github.com/acgetchell/delaunay/issues/653) using the
same published package.

`tooling/python/shared_tooling.py` temporarily selects Semgrep from the locked
Python 3.15 environment for setup, production scans, and fixture checks. The
published v0.2.4 otherwise provisions Python 3.14/Semgrep 1.182.0 and can resolve
the wrong `pysemgrep` on PATH. The adapter retains the installed shared validation
logic, is restricted to v0.2.4, and restores its process-local overrides.
[#673](https://github.com/acgetchell/delaunay/issues/673) owns deletion after the
shared-baseline fix in
[research-repo-tools #165](https://github.com/acgetchell/research-repo-tools/issues/165)
is published in v0.2.5 and verified here. Existing Semgrep dependency/security
overrides stay in place. This follow-up does not hold v0.8.4 open.

## Declarative independent performance adoption

[#648](https://github.com/acgetchell/delaunay/issues/648) consumes the published
v0.2.4 schema-4 acquisition and complete-run/v3 contract from upstream #127.
The existing exact runtime/tooling/notebook pins already select the supplying
release; `uv lock --offline` regenerated the resolution without changing it.
The unyanked PyPI wheel and sdist hashes match `uv.lock`:

- Wheel: `00aaab2b7efd680d79324c6debb64db4d8d1ff076a460ff46e24e0ef1c24bd7d`.
- Source: `30b6d4d2f8d0cb98d5311656928dd65800088b651dd6c0c29f9a02526e9e902a`.

The [v0.2.4 release](https://github.com/acgetchell/research-repo-tools/releases/tag/v0.2.4)
contains independent acquisition/source exports (#154), prepared performance
interfaces (#157), Criterion suite transport (#159), and evidence protection
(#160). Its installed parser and execution APIs are used directly, without a
sibling checkout, local wheel override or editable shared-package installation.

`performance-local`, performance CI, prepared release comparisons, report
promotion and README publication now call installed commands with the six
consumer TOML declarations under `tooling/`. The Python wheel/sdist also carries
these declarations under `share/delaunay/tooling/`. Read-only source exports
retain original source provenance. The release-signal plan overlays one current
harness onto both library revisions so harness edits do not invalidate the new
comparison. The topology and historical v0.8.2 plans preserve native harnesses
and baseline-only files. Each phase
discovers canonical IDs, completes its scientific gate, executes its declared
counts, and verifies its saved copy against fresh benchmark/estimate/sample/Tukey
bytes. The retained payload contains both complete phases and original raw JSON.
Source/controller mutations reject publication. Absolute union rows survive
incompatible identities; ratios depend only on recorded measurement evidence.

The scientific release plan remains 173 cases at 10/15/20/25/100 observations.
Native read-only discovery found that v0.8.2 has 20 older locate IDs. The named
`measurement-release-signal-v082.toml` declaration keeps its exact baseline
inventory and the current phase's 20 newer locate IDs independently. No original
fixture or harness is substituted; union rows remain absolute evidence. The
default declaration uses the v0.8.3+ contract for current-vs-latest and subsequent
release preparation. Both declared baseline inventories are checked against
original tagged source exports, without collecting new release timings.
The separate topology declaration retains its 12-observation 4D group. New
profiling targets from [#558](https://github.com/acgetchell/delaunay/issues/558)
remain manual and are not promoted into release measurements. The benchmark
[parity inventory](../../benches/README.md#profiling-suite) accounts for every
former monolithic group and its original scales.

Preparation captures independent release/ref assertions before measurement;
acquisition/promotion and README publication are separately retriable. New
complete-run history uses `docs/performance-runs/` and
`docs/performance_current.md`. Strict release promotion rejects same-label or
incompatible comparisons. Existing reports, archives and measured assets are
unchanged; old comparison evidence cannot acquire complete-run proofs by being
relabelled. README publication selects four construction rows while the full
report retains every case.

Retired production paths include paired worktree orchestration, early pair
comparison, local report promotion and the separate README publisher. Their
generic tests are replaced by installed-contract consumer checks.

Counts compare the initial clean `HEAD` with this working tree and include
physical lines (including blanks) in `.py` files directly under the owning
directories. Static-analysis fixtures, generated files and cached bytecode are
excluded. Consumer TOML and the Rust diagnostic extraction are separate surfaces.

| Surface | Before files / lines | After files / lines | Net lines removed |
|---------|----------------------|---------------------|-------------------|
| `tooling/python/` production | 14 / 9,608 | 13 / 8,437 | 1,171 |
| `tests/tooling/` consumer tests | 28 / 10,725 | 28 / 9,665 | 1,060 |

Remaining adapters have distinct ownership reasons:

| Adapter | Remaining responsibility |
|---------|--------------------------|
| Single-checkout paths in `benchmark_utils` | Shared acquisition requires a pair; preserve one run, construction counters and raw verification |
| Historical assets / `performance_artifacts` | Delaunay's nested count/coverage and separate-session policy exceeds the shared legacy-layout reader |
| `release_benchmarks` | Preserve captured release ID/tag commit, exact tagged checkout and refusal to replace an existing baseline asset |
| Summary/ranking code | Delaunay construction diagnostics, circumsphere interpretation and public API coverage |

Historical asset inspection now uses private scratch directly and creates no
Git worktree. It cannot promote its older schema through the new declarations.
The single-checkout adapter remains a composition of installed public APIs; it
must not be replaced by measuring both phases merely to use the paired CLI.

Consumer regressions execute tiny original phase harnesses with a baseline-only
file, native discovery/gates, all six count values, sanitized storage paths,
byte-identical saved copies and offline rendering after scratch cleanup. They
reject failed gates, wrong counts, unexpected observations, changed source,
corrupt saved Tukey data and incompatible acquisition policy. Native process
execution is real; Git revision resolution is modeled, and synthetic observations
make no performance claim. Wheel/sdist consumption, declaration installation,
actual Just/workflow wiring and the existing scientific single-run/asset tests
remain in the Python validation suite.

A compatible synthetic pair also executes the actual preparation, strict report
and README declarations, including immutable offline history and source freshness
checks. Its hardware identity is modeled before acquisition. Native observations
remain nanoseconds throughout the report and README; no samples are rescaled.
Cargo manifests and locks remain captured source inputs. Normalized build policy
and the complete Criterion lock entry are version probes, avoiding artificial
harness changes solely from a package-version edit. The build probe selects
compiler, profile and target flags; registry credentials are excluded and the
consumer regression verifies that boundary.

`just ci` passed locally, including all 895 Python/tooling tests, Rust correctness
buckets, the canonical tracked-figure comparison, benchmark compilation and
examples. The 112 default diagnostic IDs match the original profiling inventory;
all 40 small validation-component preflights passed. Selected normal diagnostic
measurements and final circumsphere/large-validation preflights also passed.
These checks do not constitute a timing campaign over every large fixture.

Hosted CI exposed two declaration defects. Separate
`--save-baseline retained` arguments made the original harness treat `retained`
as a filter, omitting 16 import/proof cases. Joined `--save-baseline=retained`
arguments preserve the complete inventory without changing either native
harness. The publication version assertion now accepts LF and CRLF records;
both encodings run through the complete consumer publication regression.
Follow-up focused validation passed all 899 Python/tooling tests and the
configuration/documentation gates. Native discovery restored the 100-case API
inventory from the broken command's 84; bounded sampling verified ten samples
and identical fresh/saved raw JSON for every restored import/proof case.

Local execution is native aarch64 macOS with Python 3.15.1 and shared tooling
0.2.4. Linux and Windows execution require the configured hosted CI jobs on the
eventual commit; local checks do not establish those results. No historical
timings, tracked scientific figures, release or remote settings were refreshed.

## Strict Actions allowlist adoption

[#649](https://github.com/acgetchell/delaunay/issues/649) consumes the published
v0.2.4 `actions allowlist` CLI. The existing manifest pins and registry lock
already select that containing release; `uv lock --check` confirms consistency.
The versioned selected-actions JSON is the single allowlist owner, with explicit
GitHub-owned identities, each CodeQL subpath, and the reviewed Dependabot workflow
path. Both broad exemption flags are false, and the authentication action retains
`c6f97d42243bad5fab37ca0427f495c86d5b1a18` as its only permitted ref.

`just action-allowlist` checks `.github/workflows` and `.github/actions`
recursively. `github-actions-check` composes it into local configuration checks
and every existing `just ci` platform job. The duplicate Semgrep allowlist is
removed after its accepted/rejected syntax cases pass the installed parser;
`test_actions_allowlist.py` owns those annotations in the existing workflow
fixture. Independent SHA/comment rules now also cover composite definitions.
The Actions updater exclusions and release/authentication checks remain active.
Live repository settings are outside this versioned-policy migration.

Consumer tests exercise authentication drift (full/short SHA, branch, tag),
step/job/composite references, exact nested identities, rejected owner exemptions,
local/same-repository/container handling, source-line diagnostics, and the actual
Just recipe rejecting a newly added nested composite. Native execution here is
aarch64 macOS with Python 3.15.1, research-repo-tools 0.2.4 and Semgrep 1.182.1.
The configured Linux/macOS/Windows CI jobs must validate the eventual commit;
local results do not establish hosted-platform success.

The first allowlist-only validation passed 923 Python/tooling tests, including
36 focused allowlist cases, and all affected focused validators. The combined
adoption evidence below includes the subsequent Rustdoc integration.

## Aggregate Rustdoc adoption boundary

[#650](https://github.com/acgetchell/delaunay/issues/650) enables `--rust-docs`
in the actual `semgrep-scan` and `semgrep-test` recipes. The installed v0.2.4
extractor preserves original logical paths, native/prose inputs, block identity,
line/column mapping, paired-report completeness and one aggregate SARIF category.
The original `src/lib.rs` reproduction now covers its source and all 34 blocks.

`tooling/python/semgrep_compat.py`, called by the version-guarded
`shared_tooling.py` entry point, bridges three remaining shared capabilities:

- `metadata.delaunay-rustdoc-paths` opts numerical-safety rules into extracted
  Rust with explicit `.rs`/`.md` logical paths. The selected rules reject
  non-finite conversion defaults, algebraic f64 operations and defaulted
  partial comparisons. All native rule definitions and path filters remain
  intact. Production-only diagnostics and import/layering rules do not apply to
  extracted examples; legitimate `println!` examples remain unchanged.
- `tool.delaunay.semgrep.rustdoc-exclude` excludes design sketches in
  `docs/api_design.md`, contributor policy in `docs/dev/**`, and historical
  `docs/archive/**` from the **added extraction**. These contain incomplete
  expressions, isolated enum variants, intentionally invalid examples or mixed
  language sketches. Their complete original files still receive every existing
  native/prose rule. Source, README and usage-documentation fences receive Rust
  analysis; no selected block may be silently skipped. The fixture planner
  assigns raw-comment/prose assertions to native inputs and opted-in Rust
  assertions to blocks, retaining the shared parser and exact expected counts.
- The shared [v0.2.4 contract][semgrep-api] intentionally clears output after an
  invalid scan. The bridge instead rolls back that empty generation using the
  shared publication transaction. It initializes `<output>.attempt.json` to
  `published: false` before preconditions or scanning, and sets it true only
  after a complete generation commits. Valid findings still publish reports
  and fail the gate. The hosted workflow clears the marker before invocation,
  uploads only a current generation and propagates scan failure. Previously
  retained reports are historical evidence, never current scan success.

These process-local hooks restore on exit. They do not replace extraction,
native execution, report validation, source mapping, deduplication or aggregate
publication. [research-repo-tools#171](https://github.com/acgetchell/research-repo-tools/issues/171)
owns the upstream capabilities;
[#676](https://github.com/acgetchell/delaunay/issues/676) depends on their release
and owns normal package adoption plus deletion of this bridge. The maintainer
approved completing #650 with this local workaround; that future release does
not block it. The separate scanner-environment adapter remains tracked by #673.

Run a bounded example-scope check without replacing the normal report directory:

```sh
uv run --locked --group dev python tooling/python/shared_tooling.py semgrep scan \
  --rust-docs --include src/core/collections/key_maps.rs \
  --output target/semgrep-rustdoc-scope-probe
```

`tests/tooling/test_semgrep_rustdoc.py` runs the installed native scanner with
unchanged rooted rule definitions, multiple blocks and one-input batches. It
checks original paths with spaces/Unicode, hidden-line columns, late findings,
suppression parity, prose-only inputs, extraction exclusions and cleanup.
Equivalent optional SARIF indices are added to actual native reports at the
process boundary to exercise aggregate reindexing; they are modeled metadata,
not a claim about fields emitted by this Semgrep version. Temporary-consumer Git
inventory is modeled; the real Just gates exercise repository enumeration.

Failure regressions corrupt real later-block reports or introduce invalid Rust,
then require failure and byte-for-byte retention of the preceding valid pair.
They cover missing block coverage, native errors, malformed/missing reports,
JSON/SARIF disagreement, timeouts and precondition failure. Mixed fixture tests
also reject missing and extra findings; the checked-in gate retains existing
prose and raw Rustdoc import assertions alongside new numerical fence cases.

Local validation passed all 938 Python/tooling tests, `just python-check`,
`just python-fixture-lint`, `just check-config`, `just check-docs`,
`just semgrep-test`, `just semgrep`, and `uv lock --check`. The aggregate scan
accounted for 337 original inputs and 830 Rust blocks in 65 batches, with zero
findings/errors and one `semgrep-repository-rules/` SARIF run. The original
`src/lib.rs` case contributes its source plus all 34 blocks.

Native execution is aarch64 macOS, Python 3.15.1 and Semgrep 1.182.1. Hosted
Linux/Windows results require CI on the eventual commit; local tests and workflow
validation do not establish those results. No scientific source example,
tracked notebook artifact, release or live repository setting was changed.

[semgrep-api]: https://github.com/acgetchell/research-repo-tools/blob/v0.2.4/docs/security-api.md

## v0.2.3 Cargo artifact promotion

The v0.2.3 Cargo promotion adapter is superseded by the v0.2.4 adoption below.

## v0.2.4 release tooling adoption

Delaunay consumes the published v0.2.4 `release evidence` and `release assets`
commands and their public APIs. The exact runtime/tooling pins and registry
artifacts already supply upstream #129/#130/#140. No dependency upgrade,
editable install, sibling checkout or private API is used.

`tooling/release-evidence.toml` owns exact workflow/job/step/runner requirements
and the Cargo archive guarantee. Shared code now owns nested archive/source
inspection, manifests, immutable artifact retrieval, run/attempt selection and
frozen handoff comparison. `release_artifacts.py` is deleted;
`cargo_release.py` retains only Cargo's paired dry-run locations, clean source
observation, published-event boundary and credential-free byte reproduction.
`write_release_outputs` transports frozen evidence directly to `GITHUB_OUTPUT`,
preserving previous records and failing on export errors. Just stdout remains
a display channel. Main CI runs the complete native matrix and default plus
all-feature Cargo package builds before retaining an archive. Release jobs require
the exact archive to reproduce without builds, rechecking the frozen selection
after deployment approval. Cargo remains the native uploader and repackages the
verified clean source; it does not directly upload the downloaded archive.
The local `release-evidence` recipe verifies the same retained archive without
Cargo or a release event on any host. `release-publish` requires this check
before making the reviewed draft immutable; the workflow separately enforces
the published event, producer-tool equality and byte-identical repackaging.

The release recipes follow la-stack's operator contract. `release-evidence`
uses the isolated published runtime and accepts `--output`, `--expected` and
`--github-output` paths; it requires neither a tag nor Cargo. `release-assets`
uses the shared CLI and declarative tag-as-title policy. `release_benchmarks`
remains explicitly local because Delaunay rejects an existing baseline asset
even when its bytes match, while the shared asset command supports identical-byte
retries. Captured ID/title/tag/commit and the preserved main runtime remain
required across the tagged checkout.

`release_changelog.py` is deleted. The shared citation-date declaration now
supplies the recorded date for prospective generation and preview. Performance
preparation, release acquisition and separately retriable README publication
remain the already adopted independent native-harness commands; this patch
collects no new timings and rewrites no historical evidence.

Tag-specific scientific benchmark production and provenance, protected-branch
and generated-notes checks, immutable GitHub Releases, crates.io OIDC approval,
registry absence and visibility checks retain their existing owners. No package
version beyond the pinned 0.2.4 is required. Missing/expired/partial-rerun evidence fails
without rebuilding; [release recovery](../RELEASING.md#recovering-a-failed-run)
defines the 30-day retention and complete-main-attempt recovery policy.

Consumer regressions use real Cargo/ZIP fixtures and the installed shared public
interfaces, modeling GitHub transport and native Cargo execution. Native hosted
producer/promotion and the next approved publication remain rollout evidence;
local tests do not establish that those workflows have executed.

The tag-triggered preparation workflow verifies protected-main source, retained
Cargo evidence, metadata/notes and exact-commit native/SARIF checks before draft
creation. Retries validate the same mutable draft and body without touching
assets. It does not publish. The guide verifies evidence before annotated tag
creation/push, then uses explicit `release-benchmarks TAG` dispatch. The
[CI ownership inventory](ci_validation.md) records measured before-change costs,
the portable/native split and deliberately disabled cross-commit reuse.

The published [consumer regression suite][shared-release-tests] covers generic
archive safety/source metadata, run/job/artifact tampering, frozen approval
identity, output rollback, and normal/quiet Just handoffs. Local tests retain
the consumer policy and real Cargo/ZIP fixtures, native dry-run/reproduction
commands, actual isolated pre-tag recipe, draft orchestration, benchmark asset
refusal and dispatch forwarding. Generic parser/transaction cases are owned
upstream; the retired local implementations are not imported.

[shared-release-tests]: https://github.com/acgetchell/research-repo-tools/blob/v0.2.4/tests/releases/public_evidence_consumer.py

## Ownership

Consumers call the public CLI and documented Python APIs. Shared ownership now
includes managed setup and credential isolation, dependency updates, release
metadata, changelog policy, file discovery/batching, process execution, safe
archive extraction, worktree lifecycle, Criterion parsing, file publication,
notebook infrastructure, paper checks and native Tectonic discovery.

Delaunay retains scientific geometry, topology, workloads, benchmark eligibility,
construction metrics, complete confidence intervals, provenance-based
comparability, report interpretation, and geometric-mean README summaries.
Active manuscript prose and production tooling live in
[Papers/delaunay-validation](https://github.com/acgetchell/Papers/tree/main/delaunay-validation).
No tracked scientific report, CSV/JSON baseline, notebook, PDF, or figure is
refreshed by this adoption patch. The canonical macOS `just ci` leg regenerates
temporary validation figures only to compare them with the tracked artifacts.

## Remaining shared-tool gaps

The v0.1.8 implementation is adopted for issues #625–#630, #632–#634 and #636.
Their native Linux/Windows consumer results remain pending hosted CI.
Issue #631 adopts the v0.1.9 discovery, heterogeneous completeness, saved-sample
verification and captured release-target APIs. Delaunay retains workload and
scientific eligibility policy; shared code owns the underlying mechanisms.

| Adoption | Published integration | Remaining ownership |
|-----|-----|-----|
| [#625][d625] | Paper/PDF policy, dates, normalization, Tectonic discovery/export | Active manuscript tooling now lives in acgetchell/Papers |
| [#626][d626] | Live Cargo discovery, feature groups, locked builds and output | Nine example assertions and diagnostics feature |
| [#627][d627] | SARIF split/GitHub outputs and complete-directory publication | Codacy legacy indices, Opengrep policy and six scientific images |
| [#628][d628] | Declared lowercase kebab-case notebook ID pattern | Stable descriptive cell names |
| [#629][d629] | Paired Semgrep scan, suppressions, aggregate SARIF and budgets | Repository rules/scopes and fixture exclusion |
| [#630][d630] | Nullable host, native TOML and profiling source capture | Scientific eligibility and dynamic profiling labels |
| [#631][d631] | Measurement, inventory, completeness, saved samples and captured-target publication | Scientific policy; see below |
| [#632][d632] | Python commands/pins, Just inspection, notebook test project | Lint policy and installed-wheel smoke test |
| [#633][d633] | Shared raw all-lines 160-character Markdown gate | Explicit changelog/history exclusions |
| [#634][d634] | Shared JupyterLab launch and explicit index/revision reset plan | Declared scratch/checkpoint paths; reset remains opt-in |
| [#636][d636] | Public `toolchain sync-binaries` | Token-free package/Cargo/dev steps and existing native CI matrix |

[d625]: https://github.com/acgetchell/delaunay/issues/625
[d626]: https://github.com/acgetchell/delaunay/issues/626
[d627]: https://github.com/acgetchell/delaunay/issues/627
[d628]: https://github.com/acgetchell/delaunay/issues/628
[d629]: https://github.com/acgetchell/delaunay/issues/629
[d630]: https://github.com/acgetchell/delaunay/issues/630
[d631]: https://github.com/acgetchell/delaunay/issues/631
[d632]: https://github.com/acgetchell/delaunay/issues/632
[d633]: https://github.com/acgetchell/delaunay/issues/633
[d634]: https://github.com/acgetchell/delaunay/issues/634
[d636]: https://github.com/acgetchell/delaunay/issues/636

Retired production helpers are `paper_check.py`, `paper_pdf_normalize.py`,
`paper_source_date.py`, `tectonic_native_dependencies.sh`,
`ci/filter_codacy_sarif.py`, and `ci/capture_profiling_metadata.sh`.
Their generic regression suites are retired. `hardware_utils.py` and its tests
are now deleted entirely; consumers use the public `capture_host` API directly.
The legacy text parser/writer, warm baseline cache, artifact polling, hardware
text/tolerance rules, threshold CLI, and manual `generate-baseline.yml` workflow
are retired. `benchmark_models.py` now contains only circumsphere rankings.
Notebook cells call the shared live process runner without a local wrapper.
The shared snapshot API replaces local Git patch copying, includes nonignored
new files and modes, and rejects application to a different checkout revision.
Consumer source identity binds those new file bytes and modes too.
The inline private binary setup adapter, Python command loop, Markdown line
loop, notebook spelling test, figure swap/rollback, local GitHub release DTOs,
release-list parsing, and nested performance publication rollback are removed.
Saved-baseline pairing and ratio arithmetic now use the shared `Sample`,
`Comparison`, and `compare_samples` APIs; local code selects Delaunay workloads
and retains its report presentation and complete-interval requirement.

The pinned Codacy Analysis CLI 7.9.25 emits category-local rule indices and
`-1` for findings without descriptors. A narrow inline compatibility step in
`.github/workflows/codacy.yml` reconstructs those legacy references from rule
IDs before shared SARIF validation. The configured driver name includes Codacy's
`(reported by Codacy)` suffix. The adapter preserves findings and metadata;
the shared splitter still owns namespace selection, validation, reindexing and
publication. Consumer regressions execute the workflow's actual adapter and
verify that unrelated malformed indices remain errors. Retire the adapter when
Codacy or the shared package supports this producer format directly.

Declarative workflow inputs live under `tooling/`, following the same boundary
as la-stack and markov-chain-monte-carlo. The remaining scientific Python modules
are packaged from `tooling/python/`, with consumer tests in `tests/tooling/` and
mirrored fixtures in `tests/semgrep/tests/tooling/`. The last shell helper,
`scripts/release_benchmarks.sh`, is retired. Its small release-policy adapter
is packaged as `tooling/python/release_benchmarks.py`.
Legacy benchmark CLI commands and recipes are deliberately removed.
The retained `performance-*` commands use new shared evidence paths.

Performance retention uses the shared Criterion comparison JSON and evidence
envelope. The envelope retains Delaunay coverage and comparability policy in
consumer context; full commits bind the outer provenance to measured source
identity. The shared parser validates payload integrity, and the consumer
requires complete intervals and the union of both benchmark inventories.
Promotions pass evidence, Markdown, archives and navigation to one shared
publication transaction. Historical CSV/text records stay byte-for-byte intact;
new `.comparison.json` and `.evidence.json` paths distinguish the new contract.
Retired comparison formats are no longer accepted as promotion inputs.

### Performance deletion gates

The published v0.1.9 contracts resolve
[heterogeneous sampling #101](https://github.com/acgetchell/research-repo-tools/issues/101),
[inventory and saved samples #98](https://github.com/acgetchell/research-repo-tools/issues/98),
and [captured release targets #94](https://github.com/acgetchell/research-repo-tools/issues/94).
Fresh measurements use shared `discover_inventory`, `measure_checkout`,
`collect_complete_sample` and `validate_saved_sample`. Discovery uses isolated
Criterion output and canonical IDs from native listing; collection reads
`benchmark.json` identity independently of sanitized storage paths. The policy
declares every case's count before measurement. The release-signal workload
keeps its 10, 15, 20, 25 and default 100 samples; topology experiments also retain
their existing 12-sample group. Missing, unexpected or incomplete cases fail.
Saved copies must match the raw benchmark, estimate, sample and Tukey bytes.

Delaunay selects targets, sections, groups and counts, executes the independent
scientific preflight, and interprets construction diagnostics. The shared runner
checks source, harness, compiler and host identity around execution. Fresh runs
use a private output tree. Shared directory publication replaces the latest
generation only after validation, preserving named baselines and restoring prior
output on caught failures. Publication requires exclusive ownership of the output
tree and is not crash-atomic. If rollback fails, restore the retained backup
identified by the recovery error before retrying. Release packaging revalidates raw observations against
the retained measurement and preflight policy. No samples are synthesized,
padded, truncated or rescaled.

The current release-signal plan overlays the current harness onto both library
revisions and requires it to compile against each. The topology and historical
v0.8.2 plans retain independent native harnesses. Current schema-4 runs retain
complete-run evidence with the declared harness mode, original source provenance,
canonical inventories, count policies and exact raw samples.

The retired independent-checkout runner kept each revision's own harness and
retained separate sample/evidence pairs. Those historical records remain intact;
they are not relabeled as common-harness evidence or given invented proofs.

The report still suppresses ratios for incompatible workload, host or compiler
evidence and across hosted sessions. Current-only and baseline-only rows remain
visible. Same-version scratch comparisons are supported; publication requires
distinct releases and valid coverage. Measurement hosts come from each run's
captured evidence, so a later host probe cannot make two different hosts appear
comparable.

The release workflow preserves its publication adapter, locked registry runtime
and serialized `ReleaseTarget` outside the checkout before measuring the tag.
The shared publisher revalidates the captured release ID and tag commit through
upload and final publication. Delaunay adds its refusal to overwrite an existing
baseline asset and requires the benchmark checkout to match the captured commit.
Publication remains explicit, with credentials confined to the release steps.

Retired local implementation includes the generic saved/latest execution wrappers,
unused summary execution paths, the saved-baseline Just loop and the shell
release lifecycle. Their generic tests are removed. Consumer regressions cover
workload eligibility, unchanged sampling, canonical IDs, malformed/incomplete
output, preflight failure, captured provenance, ratio suppression, release policy
and trusted workflow handoff using the installed package.

## v0.1.9 consumer evidence

The #631 implementation snapshot below compares the initial index, including
the already-staged dependency refresh, with the working tree before follow-up
repository-review fixes.
Counts include physical lines in `.py` and `.sh` files under `tooling/python/`,
`scripts/` and `tests/tooling/`; static-analysis fixtures and generated files
are excluded. The existing staged changes were preserved byte-for-byte.

| Surface | Before files / lines | After files / lines |
|-----|-----|-----|
| Production helpers | 10 / 8,794 | 10 / 8,929 |
| Consumer tests | 20 / 7,736 | 21 / 7,400 |

The 92-line shell lifecycle is deleted and replaced by a 61-line release-policy
adapter. Generic execution and parser tests are removed; new consumer bindings
retain independent measurement evidence, scientific eligibility and named
baselines across fresh directory publication. Production grows by 135 lines
while tests shrink by 336; the combined reduction is 201 lines.

That snapshot passed all 475 consumer tests on native aarch64 macOS, including the
installed-wheel and isolated notebook smoke checks, plus `just python-check`,
`just python-fixture-lint`, `just check-config`, `just check-docs` and
`just shell-check`. The preserved publication environment was also synchronized
from copied locked manifests with `--no-install-project`, and its adapter's
help command loaded the installed registry package successfully.

Measurement integration exercises the installed discovery, complete collection,
saved-sample verification, provenance, evidence and directory-publication APIs
over real temporary files. Cargo/Git observations and remote release responses
are modeled at their boundaries; these tests are not native scientific benchmark
results or live GitHub publication evidence. No long benchmark, scientific
notebook, historical report refresh or release publication ran. Native Linux and
Windows validation remains for hosted CI.

## v0.1.8 consumer evidence

Production/test counts cover `.py` and `.sh` under the owning directories,
counting physical lines including blanks. These historical counts compare
baseline commit `55e2e260e62598cf41ff5ae75bc57d798ee592a9`'s `scripts/`
with adoption commit `4b9458f9fd00f5773962460bc6077582b13a5582`'s production
under `tooling/python/` and `scripts/` and tests under `tests/tooling/`.
Generated files, static-analysis
fixtures and `__pycache__` are excluded.

| Surface | Before files / lines | After files / lines |
|-----|-----|-----|
| Production helpers | 16 / 14,864 | 10 / 8,720 |
| Consumer tests | 25 / 14,228 | 20 / 7,463 |

The installed PyPI package passed 181 upstream toolchain tests and three subtests
on native aarch64 macOS, including cold, warm and damaged-cache credential
models with real child-environment probes. Synthetic authenticated discovery
stubs are boundary models; this is not evidence of live GitHub downloads on all
three native platforms. The consumer action uses no private package APIs and
keeps release credentials on its binary-discovery step only.

The earlier v0.1.8 adoption pass validated 829 tests, notebook lint, configuration,
shell, Semgrep, examples and paper artifact checks. The subsequent format and
legacy-workflow retirement uses the focused validators recorded below; its
smaller consumer suite retains scientific geometry, coverage, malformed
evidence and publication regressions. Generic retired implementations no longer
need local mirror tests; shared format and process behavior belongs upstream.

The retirement pass passed all 527 consumer tests, including installed-wheel
and isolated notebook smoke checks, plus `just python-check`,
`just python-fixture-lint`, `just notebook-check`, `just check-config`,
`just check-docs`, and `just shell-check`. Native Semgrep reported zero findings
and zero errors across 323 inputs. All execution was on aarch64 macOS.
The system uv changed to 0.13.0 during validation. Initial checks used an
isolated 0.12.24 installation under `/private/tmp/`; final checks use the pinned
0.13.0 manager. That retirement pass did not run live benchmarks or hosted CI.

The retirement pass did not execute scientific notebooks or refresh tracked
figures/PDFs. Publication and hosted validation happened in subsequent PR work;
the results below describe that follow-up separately.

### Performance smoke follow-up

The first post-adoption 2D–5D smoke run exceeded the unchanged 60-second cap in
every dimension. The same workloads also timed out at baseline commit
`55e2e260e62598cf41ff5ae75bc57d798ee592a9`, establishing a pre-existing
validation cost. Those failed runs provide no valid performance measurements.

At follow-up commit `6ef1ae4fa5d7268b32d73c591b65e9b9b1c41703`, the unchanged
smoke workloads passed in 31.587, 33.887, 38.229 and 38.279 seconds for 2D,
3D, 4D and 5D respectively, with no skipped insertions. Exact Delaunay
classification remains the fallback for uncertain circumsphere bounds. The
smoke guard is a coarse wall-clock regression check, not release benchmark
evidence. Local `just ci` also passed, including the canonical macOS validation
notebook figure comparison; it did not publish tracked figures. Native
Linux/Windows results require hosted CI and are not established by that local run.

## Regression ownership and consumer evidence

The following records describe the earlier v0.1.7 migration and remain as
historical validation evidence, separate from the v0.1.8 results above.

The [published package's regression suites](https://github.com/acgetchell/research-repo-tools/tree/v0.1.7/tests)
own the common cases formerly tested by the deleted maintenance helpers:

| Shared suite | Contract |
|-----|-----|
| `changelog/test_archive.py`, `test_normalize.py`, `test_workflow.py` | History retention, normalization, preview, formatting, and rollback |
| `changelog/test_cliff_template.py`, `test_notes.py`, `test_tags.py` | Literal code, links, note extraction, annotation limits, and tag failures |
| `releases/test_update.py`, `test_metadata.py`, `test_policy.py` | Complete release candidates, dates, independent assertions, and atomic publication |
| `dependencies/test_python_pins.py`, `test_tool_pins.py`, `test_update_recipes.py` | Included groups, scoped pin updates, composition, and failure ordering |
| `review/test_review.py` | Default remote verification, local overrides, instruction discovery, streams, and status propagation |
| `semgrep/test_contract.py`, `test_findings.py` | Fixture isolation and exact expected finding counts/spans |

These suites ran against the installed PyPI package on this Mac: 767 passed,
13 subtests passed, and 32 skipped because disposable Git mutations were
explicitly disabled. Consumer regressions in `tests/tooling/test_shared_tooling.py`
exercise actual recipe composition, both Cargo resolution roots, every update
failure boundary, exact package provenance, real declarative release checks,
prospective dates and Rust code, and review commands with local executable stubs.
No live CodeRabbit review was run.

The real `update-dependencies` workflow also completed in a temporary consumer
copy with `default-groups = []`; both Cargo manifests/locks remained valid and
tool pins were preserved. The real `update-cargo-tools` command completed with
all declared pins already current. Recording-process tests cover aggregate
tools-first ordering and prove tools-only commands never dispatch dependency
resolution steps. An offline prospective release preview validated the actual
consumer metadata policy without publishing edits.

The direct setup contract also ran in that temporary consumer with isolated uv
tool directories already on PATH, verifying its persistent Just installation
without changing user shell startup files.

The expanded migration uses the installed v0.1.7 distribution, without a sibling
checkout on the import path. Common process, archive, inventory, notebook kernel,
and transaction implementation cases belong to that package's suites. Local
integration tests retain benchmark target/coverage rules, malformed evidence,
consumer callback rollback, README publication failures and stale-input checks,
cell-ID spelling, recipe composition, and wheel installation with notebook
execution in a disposable locked project.

The expanded adoption passed `just python-check`, `just python-fixture-lint`,
`just test-python` (937 tests), `just notebook-check`, `just check-config`,
`just check-docs`, `just shell-check`, and `just semgrep-test` on native aarch64
macOS. The final duplicate Criterion parser removal also passed all 391 benchmark
tests and the Python checks. Native Semgrep reported zero findings across 329
scanned targets; the shared Cargo metadata validator passed. Notebook execution
was limited to the disposable installed-wheel smoke test; no scientific figures
or benchmark evidence were regenerated. Linux/Windows execution remains for CI.

The declarative follow-up removed the notebook CLI and shell example runner.
All 937 Python tests and all nine examples passed through the revised workflows,
including the diagnostics feature build. Python, notebook, configuration,
documentation and shell checks passed, and Semgrep reported zero findings.
These follow-up results are native aarch64 macOS evidence.

## Dependabot automation

The Dependabot caller pins the reviewed reusable workflow revision
`7282cbd90cf5cd0ca8a1c8a01d3fed520f04eb2f` from
`acgetchell/research-repo-tools`. This workflow pin is independent of the PyPI
package pin. Its reviewed `pull_request_target` job runs from trusted base
definitions without checking out or executing PR code and receives no repository
secrets. The local policy lists exact Cargo, uv, workflow, and composite-action
files, including both checkpoint fixture resolution files.

The shared workflow verifies signed Dependabot metadata, PR identity and current
head, commit ancestry, and the complete changed-file list before approving an
eligible update and enabling native squash auto-merge. GitHub Actions must be
allowed to approve PRs. Active branch rules require an approval, dismissal of
stale approvals on push, resolved review threads, and strict required checks.
Existing CI, Codacy, and CodeRabbit status requirements continue to gate merging.
The old review-token request and CodeRabbit approval polling are removed; local
consumer tests cover policy wiring, while the shared suite owns approval logic.
The trigger rule excludes only this reviewed caller's exact path. Semgrep's SARIF
formatter retains inline-suppressed results, which GitHub still reports; a
consumer regression verifies that the reviewed caller produces no SARIF alert
and the same trigger in another workflow remains a finding.

## Platform evidence

The initial setup/changelog migration's managed inventory was installed and
verified on native aarch64 macOS,
including cargo-edit, samply, Tectonic, tex-fmt, and the SARIF tools. Local
`just check` and the comprehensive `just ci` contract passed on 2026-10-04.

| Local validation | Result |
|-----|-----|
| Rust unit tests | 2,772 debug and 2,771 release tests passed |
| Rust integration and CLI tests | 797 integration and 64 CLI tests passed |
| Rust doctests | 771 library and 15 README tests passed |
| Consumer Python tests | 1,023 tests passed, including shared tooling integration |
| Scientific validation | Canonical notebook figures matched tracked artifacts |
| Build and example checks | Benchmark compilation and all examples passed |

Native Linux and Windows execution remains for the hosted CI matrix after
publication of this change; local workflow validation is not evidence that those
jobs ran. Review tests use local stubs; live CodeRabbit invocation remains
separately authorized work.

The shared composite action provisions platform native libraries, restores
versioned managed installations, synchronizes the locked tooling/dev/notebook
groups, and exports verified paths. Windows uses a short installation root.
The CI host check runs after that export, against the compiler actually used.
