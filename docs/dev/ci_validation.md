# CI validation ownership

Local `just ci` retains every validator. Hosted CI runs portable checks once,
then preserves the three native platform builds and the separately required
SARIF jobs. `tooling/release-evidence.toml` requires portable, native and package
jobs and their successful steps in the same trusted-main run/attempt.

## Before-change timing

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

## Coverage after the split

| Owner | Checks |
|-------|--------|
| `ci-portable`, Linux once | Configuration/Actions policy; Markdown, spelling and versions; shell/Rust formatting; shell lint; Semgrep fixtures |
| `ci-native`, Linux/macOS/Windows | Cargo lock, Python, notebooks, dependencies, Clippy/rustdoc, tests, benchmark compilation and examples (details below) |
| `ci-native`, macOS | Canonical notebook execution and tracked-figure byte comparison |
| Repository Rule SARIF | Complete production native/prose/Rustdoc Semgrep scan, blocking findings and current-generation SARIF upload |
| Clippy SARIF | Linux all-targets/all-feature diagnostics and security reporting |
| Main package producer | Clean default/all-feature native Cargo packages and immutable retained archive |
| Explicit `test-slow` / `ci-slow` | Existing bounded slow correctness bucket, unchanged |

The native owner includes Python formatting/lint/types/fixture checks, notebook
lint/advice, default/all-feature all-targets Clippy, and every default Rust/Python
test bucket, including debug/release units, integrations, doctests, CLI and
feature-only lanes. No platform-sensitive selection is moved to portable CI.
Within the test workflow, Python consumer tests run before the Rust buckets so
native tooling failures surface earlier. Release recipe fixtures install
isolated console entry points, including Windows `.exe`
launchers; extensionless shell scripts do not intercept Windows executable
lookup. Local macOS success cannot establish Windows portability.

Consumer tests compare the complete recipe graph: the remote union, including
the production SARIF scan, equals local `ci`, and portable/native subsets are
disjoint. Clippy's reporting invocation remains deliberate overlap: native
configuration coverage and SARIF conversion have distinct evidence roles.
No scientific input, assertion, property count, feature lane or measurement plan
is reduced. Performance acquisition remains separate from correctness checks.

## Exact-commit release boundary

Existing native and SARIF check names remain stable. Each native build depends
on portable success; publishing additionally requires `portable` from GitHub
Actions app ID 15368. No live branch-rule mutation is needed for those existing
build requirements, and no settings are applied by this patch. The native jobs
explicitly run their initial dependency-result guard after portable failure,
producing a failed required check instead of a skipped job.

Shared v0.2.4 supplies opt-in signed PR/main reuse, but adoption here remains
disabled. Equal trees do not establish equivalent workflow context. Main-push
CI, complete attempts, immutable artifact identity, Cargo byte reproduction,
frozen post-approval selection and OIDC/environment boundaries remain required.
The new policy/schema cannot consume older adapter manifests: an eligible main
run of the adoption commit or a later reviewed commit must produce fresh evidence.
