# Shared tooling adoption

The repository now pins the published
[research-repo-tools v0.1.8](https://github.com/acgetchell/research-repo-tools/releases/tag/v0.1.8)
PyPI distribution in runtime dependencies, the tooling/dev groups, and the
notebook extra/group. `uv.lock` records exact registry artifacts and hashes.
Python retains the inherited 3.14 floor. Ruff, ty and pytest inherit exact
versions from the package's `python-tools` extra; no independent floating
requirements remain. The uv declaration is 0.12.24, matching the installed
manager used for the lock and consumer checks.

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
Issue #631 is partially implemented; it must remain open until the remaining
performance contracts can be replaced without changing scientific behavior.

| Adoption | Published integration | Remaining ownership |
|-----|-----|-----|
| [#625][d625] | Paper/PDF policy, dates, normalization, Tectonic discovery/export | Native libraries and explicit paper recipes |
| [#626][d626] | Live Cargo discovery, feature groups, locked builds and output | Nine example assertions and diagnostics feature |
| [#627][d627] | SARIF split/GitHub outputs and complete-directory publication | Opengrep policy and exact six scientific PNG images |
| [#628][d628] | Declared lowercase kebab-case notebook ID pattern | Stable descriptive cell names |
| [#629][d629] | Paired Semgrep scan, suppressions, aggregate SARIF and budgets | Repository rules/scopes and fixture exclusion |
| [#630][d630] | Nullable host, native TOML and profiling source capture | Scientific eligibility and dynamic profiling labels |
| [#631][d631] | Shared comparison JSON/evidence, release selection, asset downloads, publication | Mixed sampling and canonical Criterion IDs; see below |
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

Declarative workflow inputs live under `tooling/`, following the same boundary
as la-stack and markov-chain-monte-carlo. The remaining scientific Python modules
are packaged from `tooling/python/`, with consumer tests in `tests/tooling/` and
mirrored fixtures in `tests/semgrep/tests/tooling/`. `scripts/` contains only
`release_benchmarks.sh`. The package boundary remains `tooling/python/`; `scripts/` contains only shell
scripts. Legacy benchmark CLI commands and recipes are deliberately removed.
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

Published `CompletePolicy` has one sample count for a whole phase, and the
common-harness plan requires the same count for both phases. Delaunay's actual
release-signal plan intentionally combines 10, 15, 20, 25 and default 100-sample
groups. A valid two-case fixture with 10 and 100 raw samples is rejected under
both uniform policies by the installed v0.1.8 package. This verified gap is
[research-repo-tools #101](https://github.com/acgetchell/research-repo-tools/issues/101),
recorded as a native blocked-by dependency of [#631][d631].

Consequently, release-signal execution and provenance orchestration remain in
`benchmark_utils.py`. They preserve target/section/group selection, independent
scientific preflight, exact Criterion IDs, incomplete coverage and hosted-session
ratio suppression. Native reports use recorded harness and measurement-plan
identity rather than the historical v0.8.2 label boundary. Differing workload,
host or compiler evidence prevents ratios. Same-version scratch comparisons
are supported; publication still requires distinct releases and valid coverage.
The CI performance job now measures both revisions freshly in one job and
retains descriptive evidence without threshold classification.
Changing sample sizes, synthesizing samples or weakening completeness is not
an acceptable migration.

`release_benchmarks.sh` also retains the preflight release ID/tag-commit binding
across long benchmark runs. The published upload API cannot carry that binding;
the existing upstream
[#94](https://github.com/acgetchell/research-repo-tools/issues/94)
tracks the required replacement. Generic Criterion inventory discovery and
saved-baseline verification are tracked by
[#98](https://github.com/acgetchell/research-repo-tools/issues/98).
Full common-harness execution remains blocked by the verified mixed-sampling
gap. New shared comparison reports already use separate paths; history is kept
as a record rather than reinterpreted as newly certified evidence.

## v0.1.8 consumer evidence

Production/test counts cover `.py` and `.sh` under the owning directories,
counting physical lines including blanks. The baseline is Git HEAD's `scripts/`
before this patch; current production spans `tooling/python/` and `scripts/`,
and current tests live in `tests/tooling/`. Generated files, static-analysis
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
The system uv changed to 0.13.0 during validation; the final checks used an
isolated 0.12.24 installation under `/private/tmp/`, preserving repository pins
and the user's global tools. No live benchmark or hosted CI run was performed.

No benchmark timings, scientific notebook execution or tracked figure/PDF
refresh was performed during this migration. Git commits, tags, pushes,
CodeRabbit review and issue closure were not requested. Native Linux/Windows
results remain for hosted CI; local tests do not establish those results.

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

The Dependabot caller uses the same reusable workflow revision as la-stack and
MCMC: `cbb2ea6dee8866b3f0547bca935aef48fdd71707` from
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
