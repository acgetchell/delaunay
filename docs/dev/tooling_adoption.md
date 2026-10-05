# Shared tooling adoption

Issues [#615](https://github.com/acgetchell/delaunay/issues/615) and
[#616](https://github.com/acgetchell/delaunay/issues/616) adopt the published
`research-repo-tools==0.1.7` PyPI distribution. The exact requirement lives in
runtime dependencies, `tooling` (included by `dev`), and the notebook extra;
`uv.lock` records registry artifacts and hashes. All declarations select the same
published version.
Python inherits the shared 3.14 baseline, uv is declared in `tool.uv.required-version`, and managed
Cargo tools are declared in `tool.research-repo-tools.toolchain.cargo`.
OSV-Scanner and Gitleaks use exact managed pins in `tool.research-repo-tools.toolchain.binaries`.

## Ownership

The installed package owns setup, managed installations, dependency pin updates,
release metadata planning and checks, git-cliff policy, changelog normalization,
archive rotation, notes and tags, Semgrep fixture execution, zizmor authentication,
and opt-in CodeRabbit orchestration. Consumers call its public CLI and documented
Python APIs. The expanded adoption also delegates file inventory and batching,
process execution, safe archive extraction, temporary worktree lifecycle,
Criterion estimate parsing, atomic file replacement, README publication,
Rustdoc fence extraction, and notebook infrastructure.

The MCMC-style security workflow also uses shared commands: `just security`
composes OSV audits of all three maintained lockfiles and Gitleaks scans of
reachable history and current files. Separate Actions workflows call the same
recipes and retain redacted reports. The existing Cargo audit workflow remains
the RustSec-specific check; security scans are separate from the local default
validation tiers. See [dependency and secret scanning](commands.md#dependency-and-secret-scanning).

The current published setup contract installs Just through the package's
`rust-just` dependency and exposes `research-repo-tools setup` directly. It
supersedes the generated shell/PowerShell launchers described in the original
issue. No package checkout, editable install, local wheel, copied cliff template,
or generated launcher is required.

Consumer ownership remains with scientific validation, geometry/rendering,
benchmark selection and eligibility, retained schema-v3 evidence, and repository
policy. The local process wrapper and Semgrep target script are deleted.
The notebook wrapper and shell example runner are also deleted. Notebook lint
and advice call the shared CLI directly; a consumer test retains the lowercase
kebab-case cell-ID rule. `tooling/examples.toml` declares the complete example
inventory, build/feature order, output markers and portable deadlines, checked
against Cargo metadata. Native tools retain their repository configuration;
shared file selection replaces the repeated shell discovery loops. `docs/templates/changelog_format.toml` is an explicit local
formatting policy for historical commit bodies; active prose keeps its existing
stricter policy.

Completed changelog series moved from `docs/archive/changelog/` to
`docs/archives/changelog/`. Their generated bodies were migrated through the
shared normalizer and scoped formatter so repeat generation can compare them
canonically. All 29 archived release dates and all 732 existing commit, PR, and
comparison links were retained. Historical API names remain historical. The
current release snapshot retains its existing body and repairs archive navigation.
Scientific reports and their evidence archives retain their existing ownership.

## Remaining shared-tool gaps

Every reusable gap has a v0.1.8 upstream issue and a consumer adoption issue
with native blocked-by dependencies. Adoption requires a published package pin,
deleting the superseded implementation and generic tests, and preserving the
consumer's scientific and failure contracts.
Example discovery/live output and declarative cell-ID spelling are remaining
enhancements; they no longer block deleting their former wrapper scripts.

| Reusable capability | Upstream v0.1.8 | Adoption here |
|-----|-----|-----|
| Paper/PDF checks and Tectonic discovery | [#70][rrt70] | [#625][d625] |
| Runtime Cargo example discovery and live output | [#71][rrt71] | [#626][d626] |
| SARIF filtering and complete-directory publication | [#72][rrt72] | [#627][d627] |
| Notebook cell-ID spelling policy | [#73][rrt73] | [#628][d628] |
| Batched Semgrep scans and suppression policy | [#74][rrt74] | [#629][d629] |
| Benchmark host and profiling metadata | [#75][rrt75] | [#630][d630] |
| Complete performance workflows and immutable retention | [#64][rrt64] | [#631][d631] |
| Python commands, pins, notebook and Just test helpers | [#56][rrt56], [#57][rrt57], [#58][rrt58], [#59][rrt59] | [#632][d632] |
| Raw Markdown line-length validation | [#76][rrt76] | [#633][d633] |
| JupyterLab launch and explicit notebook reset | [#77][rrt77] | [#634][d634] |

[rrt70]: https://github.com/acgetchell/research-repo-tools/issues/70
[rrt71]: https://github.com/acgetchell/research-repo-tools/issues/71
[rrt72]: https://github.com/acgetchell/research-repo-tools/issues/72
[rrt73]: https://github.com/acgetchell/research-repo-tools/issues/73
[rrt74]: https://github.com/acgetchell/research-repo-tools/issues/74
[rrt75]: https://github.com/acgetchell/research-repo-tools/issues/75
[rrt76]: https://github.com/acgetchell/research-repo-tools/issues/76
[rrt77]: https://github.com/acgetchell/research-repo-tools/issues/77
[rrt64]: https://github.com/acgetchell/research-repo-tools/issues/64
[rrt56]: https://github.com/acgetchell/research-repo-tools/issues/56
[rrt57]: https://github.com/acgetchell/research-repo-tools/issues/57
[rrt58]: https://github.com/acgetchell/research-repo-tools/issues/58
[rrt59]: https://github.com/acgetchell/research-repo-tools/issues/59
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

Retained implementations are `paper_check.py`, `paper_pdf_normalize.py`,
`paper_source_date.py`, `tectonic_native_dependencies.sh`,
`ci/filter_codacy_sarif.py`, and `hardware_utils.py`.
The native Semgrep recipe
retains reviewed suppressions, aggregate SARIF, jobs and timeout settings.
Benchmark orchestration in `benchmark_utils.py`, `performance_artifacts.py`
and `release_benchmarks.sh` still requires frozen target completeness,
historical schema support, immutable retention and nested promotion rollback.
Markdown file selection and batching are shared; its all-lines 160-character
gate remains local. JupyterLab cache/launch setup and the explicitly invoked
Git reset/scratch cleanup recipe also await shared workflow support.

The MCMC-style declarative layout is adopted where published commands cover the
workflow. Shared measurement currently collects one statistic from each
revision's own harness, and report promotion retains evidence by release pair.
Delaunay requires a frozen target/section/group inventory, strict completeness
and eligibility checks, retained schema-v3 identities, and rollback across
retention plus promotion. Replacing that orchestration requires [#64][rrt64];
moving Python into `tooling/` would not remove it. README publication also retains
the consumer's geometric-mean group summaries and exact promoted-bundle checks.

The Codacy adapter runs in a minimal scanner job; it cannot import the package
until that workflow provisions it. Benchmark models, notebook input parsing,
scientific validation/rendering, the README/citation mirror, and focused-prelude
policy are Delaunay responsibilities. They are not generic migration targets.
Publication uses shared file transactions now; consumer callbacks still own
scientific reload validation and promotion rollback until the full workflow
contract can replace them. Tracked historical evidence is unchanged.

## Regression ownership and consumer evidence

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
explicitly disabled. Consumer regressions in `scripts/tests/test_shared_tooling.py`
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
