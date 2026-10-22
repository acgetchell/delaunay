# Shared tooling adoption

The repository now pins the published
[research-repo-tools v0.2.3](https://github.com/acgetchell/research-repo-tools/releases/tag/v0.2.3)
PyPI distribution in runtime dependencies, the tooling/dev groups, and the
notebook extra/group. `uv.lock` records exact registry artifacts and hashes.
Python follows the shared package's 3.15 floor. Ruff, ty and pytest inherit exact
versions from the package's `python-tools` extra; no independent floating
requirements remain. `pyproject.toml` owns the exact uv declaration used for
the lock and consumer checks.

The locked Semgrep runs in the same Python 3.15 environment. The uv overrides
select patched MCP 1.30.0 and, on Windows, pywin32 312 with Python 3.15 wheels;
Semgrep's own pins still select MCP 1.29.0 and pywin32 311. Native Linux and
Windows execution must be verified by their CI jobs.

The Rust dependency is now the published la-stack 0.5.0, retaining disabled
default features and the `exact` feature. This dependency refresh makes the
upstream APIs available; numerical delegation remains tracked by
[#577](https://github.com/acgetchell/delaunay/issues/577),
[#579](https://github.com/acgetchell/delaunay/issues/579), and
[#610](https://github.com/acgetchell/delaunay/issues/610).

## v0.2.3 Cargo artifact promotion

Delaunay consumes the published `release_validation.ValidationPolicy`,
`select_main_run`, `validate_jobs` and `download_artifact` interfaces. The shared
package owns trusted exact-commit main-run selection, complete-attempt job
verification and immutable ZIP integrity/inventory checks. Its Python
wheel/sdist verifier and producer-specific promotion CLI are not Cargo adapters.
The published `ci.export_environment` helper writes verified selection JSON
directly to `GITHUB_OUTPUT`, preserving existing records and failing on export
errors. Evidence transport does not depend on Just's stdout.

`tooling/python/release_artifacts.py` owns Cargo archive/source/version inspection,
the Delaunay native/package step policy, retained package/tool manifest and the
release-event handoff. Main CI runs the complete native matrix and default plus
all-feature Cargo package builds before retaining an archive. Release jobs require
the exact archive to reproduce without builds, rechecking the frozen selection
after deployment approval. Cargo remains the native uploader and repackages the
verified clean source; it does not directly upload the downloaded archive.

Tag-specific scientific benchmark production and provenance, protected-branch
and generated-notes checks, immutable GitHub Releases, crates.io OIDC approval,
registry absence and visibility checks retain their existing owners. No package
version beyond 0.2.3 is required. Missing/expired/partial-rerun evidence fails
without rebuilding; [release recovery](../RELEASING.md#recovering-a-failed-run)
defines the 30-day retention and complete-main-attempt recovery policy.

Consumer regressions use real Cargo/ZIP fixtures and the installed shared public
interfaces, modeling GitHub transport and native Cargo execution. Native hosted
producer/promotion and the next approved publication remain rollout evidence;
local tests do not establish that those workflows have executed.

## Ownership

Consumers call the public CLI and documented Python APIs. Shared ownership now
includes managed setup and credential isolation, dependency updates, release
metadata, changelog policy, file discovery/batching, process execution, safe
archive extraction, worktree lifecycle, Criterion parsing, file publication,
notebook infrastructure, paper checks and native Tectonic discovery.

Delaunay retains scientific geometry, topology, workloads, benchmark eligibility,
construction metrics, complete confidence intervals, provenance-based
comparability, report interpretation, and geometric-mean README summaries. Publication prose under `papers/` remains author-owned.
No scientific report, CSV/JSON baseline, tracked notebook, PDF, or figure was
regenerated in this adoption patch.

## Remaining shared-tool gaps

The v0.1.8 implementation is adopted for issues #625–#630, #632–#634 and #636.
Their native Linux/Windows consumer results remain pending hosted CI.
Issue #631 adopts the v0.1.9 discovery, heterogeneous completeness, saved-sample
verification and captured release-target APIs. Delaunay retains workload and
scientific eligibility policy; shared code owns the underlying mechanisms.

| Adoption | Published integration | Remaining ownership |
|-----|-----|-----|
| [#625][d625] | Paper/PDF policy, dates, normalization, Tectonic discovery/export | Native libraries and explicit paper recipes |
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

Each revision keeps its own harness. The common-harness runner intentionally
overlays the current harness onto the baseline and requires compatibility;
that would change the release experiment and discard useful absolute results.
Delaunay therefore composes the public independent-checkout runner with the
shared complete collector. Retained comparisons carry both shared sample/evidence
pairs, including source/harness fingerprints, captured hosts, canonical inventory,
count policy and raw-sample digests. They are not labeled as common-harness
`CompleteRun` evidence. Historical records are preserved without inventing those
proofs. The existing consumer digest framing remains in Delaunay context.

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
