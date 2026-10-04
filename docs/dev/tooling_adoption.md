# Shared tooling adoption

Issues [#615](https://github.com/acgetchell/delaunay/issues/615) and
[#616](https://github.com/acgetchell/delaunay/issues/616) adopt the published
`research-repo-tools==0.1.7` PyPI distribution. The exact requirement lives in
`tooling`, included by `dev`; `uv.lock` records registry artifacts and hashes.
Python remains 3.14, uv is declared in `tool.uv.required-version`, and managed
Cargo tools are declared in `tool.research-repo-tools.toolchain.cargo`.

## Ownership

The installed package owns setup, managed installations, dependency pin updates,
release metadata planning and checks, git-cliff policy, changelog normalization,
archive rotation, notes and tags, Semgrep fixture execution, zizmor authentication,
and opt-in CodeRabbit orchestration. Consumers call its public CLI.

The current published setup contract installs Just through the package's
`rust-just` dependency and exposes `research-repo-tools setup` directly. It
supersedes the generated shell/PowerShell launchers described in the original
issue. No package checkout, editable install, local wheel, copied cliff template,
or generated launcher is required.

Consumer ownership remains with scientific notebooks and validation, paper/PDF
maintenance, benchmark fixtures and evidence, performance publication,
`subprocess_utils`, scan target selection, native-library discovery, and the
Codacy SARIF filter. `docs/templates/changelog_format.toml` is an explicit local
formatting policy for historical commit bodies; active prose keeps its existing
stricter policy.

Completed changelog series moved from `docs/archive/changelog/` to
`docs/archives/changelog/`. Their generated bodies were migrated through the
shared normalizer and scoped formatter so repeat generation can compare them
canonically. All 29 archived release dates and all 732 existing commit, PR, and
comparison links were retained. Historical API names remain historical. The
current release snapshot retains its existing body and repairs archive navigation.
Scientific reports and their evidence archives retain their existing ownership.

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

## Platform evidence

The managed inventory was installed and verified on native aarch64 macOS,
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
