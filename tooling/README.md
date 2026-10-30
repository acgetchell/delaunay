# Tooling configuration

Declare reusable workflows here using the published `research-repo-tools==0.2.4`
contracts. `python/` holds the retained Delaunay-specific scientific modules and
measurement and publication policy; consumer policy checks live in `tests/tooling/`.
Shared infrastructure is called directly through the package's CLI or public
Python APIs.

Optional shared operations are available through `just files`, `just templates`,
`just validate`, and `just performance`, each taking the public
CLI's arguments and explicit configuration. `just coverage-report` summarizes
retained Cobertura data. Notebook inspection and reset previews, managed-store
cleanup previews, binary-only repair, Python pin checks, and selected Actions
update previews have focused recipes listed by bare `just`.

Active manuscript tooling lives in
[acgetchell/Papers](https://github.com/acgetchell/Papers/tree/main/delaunay-validation).
Delaunay retains its root Python environment and lockfile for software tooling and notebooks.

The retained Python ownership is explicit:

| Modules | Local responsibility |
|-----|-----|
| `benchmark_models`, `benchmark_utils` | Workload eligibility, construction metrics, circumsphere rankings and per-case sampling policy |
| `performance_artifacts` | Historical Delaunay asset schema, complete timing intervals and hosted-session policy |
| `notebook_utils`, `notebook_validation` | Delaunay CLI inputs, environment bounds and mesh/validation witness models |
| `notebook_validation_rendering`, `notebook_visualization` | Scientific figure construction and geometry checks |
| `profiling_metadata` | Dynamic CI mode/filter labels over shared capture and publication |
| `release_benchmarks` | Trusted workflow handoff, tagged checkout and refusal to replace an existing release baseline |
| `cargo_release` | Cargo dry-run locations, credential-free reproduction and published-event boundary over shared evidence verification |
| `prepare_release` | Tag-push draft orchestration after Cargo evidence, metadata and required checks |

Generic setup, subprocesses, file discovery, Criterion parsing/pairing, host
probes, notebook infrastructure, paper/PDF maintenance, SARIF and publication
mechanics belong to the shared package. Shared discovery, measurement provenance,
raw-sample completeness and saved-copy verification enforce the local sampling
declarations. The adoption inventory below records the deletion and validation evidence.

`examples.toml` replaces the example runner script. Run it with `just examples`:
the shared validator discovers examples from live Cargo metadata, groups release
builds by their required features, and runs every discovered example. The
diagnostics example requires its named feature. Cargo uses the lockfile. Each build has a 1,800-second
deadline; each example has a 600-second deadline on every supported platform.
Commands stream their output while running. Expected
stdout markers guard against skipped workflows, including the diagnostics stub.

The timeout values live in TOML; the former `EXAMPLE_TIMEOUT` environment
override and optional coreutils dependency are retired. A consumer test compares
the output assertions with Cargo metadata. New ordinary examples are discovered
automatically; add declarations for new feature requirements or output assertions.

Notebook configuration remains in `pyproject.toml` under
`tool.research-repo-tools.notebooks`. `just notebook-check` invokes shared
lint/advice directly and enforces the declared lowercase kebab-case ID pattern.
There is no local notebook-check CLI or orchestration module.

Raw Markdown line limits, SARIF driver/namespace
selection, Semgrep scan budgets, Python pin inheritance, and notebook launch/reset
policy also live in `pyproject.toml`. `profiling-inputs.toml` declares the source,
harness and version probes captured with profiling metadata; it executes no benchmarks.

The [adoption inventory](../docs/dev/tooling_adoption.md) records retired helpers,
consumer validation and platform evidence.

Paired acquisition and publication declarations are `measurement-release-signal.toml`,
`measurement-release-signal-v082.toml`, `measurement-topology.toml`,
`performance-release.toml`, `performance-report.toml`
and `performance-publication.toml`. The Just recipes call installed commands
using these declarations. See the [performance ownership boundary](../docs/dev/tooling_adoption.md#declarative-independent-performance-adoption)
for the single-checkout and historical-asset adapters that remain.

`release-evidence.toml` declares the exact trusted-main jobs, steps, native
runner labels and Cargo archive guarantee. `release-assets.toml` requires the
stable draft title to equal its tag. The installed verifier owns archive bounds,
identity, immutable retrieval and frozen revalidation. The local benchmark
adapter additionally rejects an existing baseline asset even when its bytes
match; the shared asset CLI's identical-byte retry contract remains available
through `just release-assets`.
