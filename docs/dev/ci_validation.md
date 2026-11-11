# CI validation ownership

Local `just ci` retains every validator. Hosted CI runs portable checks once,
prepares managed compilation caches on three platforms, then runs four native
stages per platform. The retained `build (OS)` checks aggregate those results;
the separately required SARIF jobs retain their owners.
`tooling/release-evidence.toml` requires portable, preparation, actual native
stage and package jobs in the same trusted-main run/attempt. Native evidence
names the stage jobs and their runner labels, not the Linux aggregate gates.
The separate `sign-package` job authenticates the package producer's immutable
artifact by ID, digest, run and attempt, signs its original manifest, and retains
the crate, manifest and Sigstore bundle together. It executes no checkout code.

## Timing before parallel native stages

The successful macOS job in [run 42962632632](https://github.com/acgetchell/delaunay/actions/runs/42962632632/job/130333440017)
validated PR #697 head `1feadb9c52a9ff3e5403e7f8fff3e8bfe49ea903` through
merge checkout `46573f3cd242377e807d4b197ccbf8bcbd1814a9` on November 11, 2026
(UTC). It restored both managed-tool and consumer compilation caches. The job
lasted about 70 minutes, including about 68 minutes in `ci-native`.

| Sequential work | Observed interval |
|-----|-----|
| Python tests | 5m 03s |
| Debug and release unit tests | 10m 21s |
| Release integrations | 16m 55s, including 15m 49s compilation |
| CLI, doctests, feature-only and allocation tests | 14m 56s |
| Benchmark compilation | 13m 09s |
| Examples through job cleanup | About 3m |
| Earlier checks, figures and packaged docs | About 4m 17s |

These command-to-command intervals include compilation and transition overhead.
They exclude time queued before the job started and do not establish a matched
cold/warm speedup. Splitting Python alone leaves most of the Rust critical path.
The four hosted stages overlap substantial independent compilation and testing,
but duplicate setup and some dependency compilation on separate runners.

After-change native timings, queue time and total runner usage must be recorded
for the final commit under [#694](https://github.com/acgetchell/delaunay/issues/694).
Measure cold and warm caches separately with equivalent validator/test coverage.
Local graph tests prove ordering and coverage; they do not establish hosted speed.

## Earlier portable/native split baseline

The successful [main run 41002070260](https://github.com/acgetchell/delaunay/actions/runs/41002070260),
attempt 1, validated commit `3444210e4f53d57efbd674be5c66283d82ea1a3b`
on October 30, 2026 (UTC). GitHub job/step timestamps give:

| Job | Setup | Validators/package verification | Total job |
|-----|-------|---------------------------------|-----------|
| Linux build | 48s | 57m 30s | 58m 43s |
| macOS build | 11m 46s | 38m | 50m 26s |
| Windows build | 5m 21s | 1h 51m 30s | 2h 3m 20s |
| Clean package | 23m 46s | 54s | 24m 57s |

The workflow elapsed about 2h 28m. This inventory is observed before-change
evidence, not a prediction of savings. The existing
[Windows log](https://github.com/acgetchell/delaunay/actions/runs/41002070260/job/123888535866)
also provides these command-to-command intervals, including compilation and
transition overhead:

| Validator | Observed Windows interval |
|-----------|---------------------------|
| Production Semgrep | 10m 22s, ending at its successful report marker |
| Semgrep fixtures | About 1m 30s, through the next correctness command |
| Debug library units | About 7m 28s |
| Release library units | About 9m 21s |
| Release integrations | About 32m 05s |
| CLI tests | About 3m 15s |
| Feature-only debug/release units | About 11m 23s combined |
| Python consumer tests | About 5m 41s |
| Benchmark compilation | About 19m 18s |
| Examples | About 5m 43s, through completion of the CI step |

The production scan/fixtures move to their single owners; scientific test inputs
and native compilation remain intact. After-change hosted timings require a run
of the final merged commit; local checks do not establish that result. Clean
package setup is still intentionally uncached; reducing its tool inventory is
separate work.

## Validator ownership

| Owner | Checks |
|-------|--------|
| `ci-portable`, Linux once | Configuration/Actions policy; Markdown, spelling and versions; shell/Rust formatting; shell lint; Semgrep fixtures |
| `ci-python-docs`, Linux/macOS/Windows | Python checks and tests; notebook lint/advice; packaged public Rustdoc with default/all features |
| `ci-python-docs`, macOS | Canonical notebook CLI build, execution and tracked-figure byte comparison, in that order |
| `ci-rust-builds`, Linux/macOS/Windows | Cargo lock/dependencies; Clippy; workspace Rustdoc; benchmark compilation and examples |
| `ci-rust-integrations`, Linux/macOS/Windows | Complete default release integration suite |
| `ci-rust-tests`, Linux/macOS/Windows | Debug/release units, CLI tests, doctests, feature-only units/doctests and public allocation tests |
| Repository Rule SARIF | Complete production native/prose/Rustdoc Semgrep scan, blocking findings and current-generation SARIF upload |
| Clippy SARIF | Linux all-targets/all-feature diagnostics and security reporting |
| Main package producer | Clean default/all-feature native Cargo packages and immutable retained archive |
| Main package signer | Same-attempt producer authentication and signed Cargo manifest; no checkout or package execution |
| Explicit `test-slow` / `ci-slow` | Existing bounded slow correctness bucket, unchanged |

Every native stage runs on Linux, macOS and Windows, verifies its compiler host,
and executes its canonical recipe. No platform-sensitive selection moves to
portable CI. Python consumer tests start independently of the Rust stages.
Release recipe fixtures install
isolated console entry points, including Windows `.exe`
launchers; extensionless shell scripts do not intercept Windows executable
lookup. Local macOS success cannot establish Windows portability.

Consumer tests compare the complete recipe graph: the remote validator union,
including the production SARIF scan, equals local `ci`; the local environment
setup is a prerequisite, not another validator. Portable/native subsets and
the four native stages are disjoint. Clippy's reporting invocation remains deliberate overlap: native
configuration coverage and SARIF conversion have distinct evidence roles.
No scientific input, assertion, property count, feature lane or measurement plan
is reduced. Performance acquisition remains separate from correctness checks.

## Local ordering and cache ownership

`just ci` synchronizes the locked development/notebook environment, completes
`check`, Python fixture lint, the canonical figure check and packaged public
docs, then overlaps two lanes: all remaining Cargo work in one sequential lane,
and the complete Python test suite. Managed commands and test subprocesses
borrow the prepared environment with `--no-sync`; they do not race environment
installation. Figure generation stays before this overlap because it builds the
checkout's perf-profile CLI. Packaged docs use the shared isolated build helper.
`just --jobs 1 ci` runs the same graph serially. `just ci-native` remains a
sequential composition of the four hosted stage recipes after normal setup.

Hosted preparation completes before the native matrix starts. Each stage still
installs and verifies exact tool versions in its own runner; preparation shares
only managed Cargo compilation output through RRT v0.2.8's supported build
directory. That cache is keyed by OS, architecture and consumer tool/dependency
declarations, without a broad restore prefix or installed executable cache.
Windows retains the short `C:/rrt` installation and `C:/rrt-build` compilation
roots. A cache miss rebuilds tools without removing a validator.

Preparation does not save consumer `target/` outputs. Each native stage owns a
distinct consumer compilation cache, keyed by stage, OS, architecture, Rust
toolchain, Cargo locks and source commit. Restore prefixes retain the same
stage/platform/toolchain/lock identity. A new source commit can publish refreshed
compilation output instead of competing for one immutable cache entry. Existing
GitHub branch cache isolation remains in force. Clean release-package jobs keep
both cache classes disabled.

## Exact-commit release boundary

Existing `build (OS)` and SARIF check names remain stable. The three lightweight
build gates run on Linux with `always()` and require successful portable,
preparation and complete native-matrix results. Missing, failed, cancelled or
skipped prerequisites fail the gates. Publishing still requires those checks
and `portable` from GitHub Actions app ID 15368; no live branch-rule mutation is
needed. The package producer waits for all gates, and signed evidence separately
requires every actual native stage, required step and native runner label.

Shared v0.2.4 supplies opt-in signed PR/main reuse, but adoption here remains
disabled. Equal trees do not establish equivalent workflow context. Main-push
CI, complete attempts, immutable artifact identity, Cargo byte reproduction,
frozen post-approval selection and OIDC/environment boundaries remain required.
The new policy/schema cannot consume older adapter manifests: an eligible main
run of the adoption commit or a later reviewed commit must produce fresh evidence.
Candidate acceptance additionally binds the independent local Step 1 acquisition
and exact generated outputs. Hosted draft/attachment jobs do not measure.
The full acceptance crosses the deployment approval boundary as an immutable
artifact; Cargo reproduction still precedes token acquisition. Native adoption
and real handoff evidence remain gates in [release tracker #696](https://github.com/acgetchell/delaunay/issues/696).
