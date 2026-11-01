# Roadmap

This page separates v0.8.4 release preparation, completed release
baselines, and future work. The [limitations guide](limitations.md) owns the
current supported operation and geometry scope. Historical task snapshots
live in [`archive/`](archive/).

## v0.8.4 Release Preparation

The current checkout is being reviewed for v0.8.4. The published v0.8.3 release
is the preceding baseline; its completed changes are summarized below.

Release preparation includes coordinated package/citation metadata and
generated changelog updates, the required native validation, and the benchmark
and publication gates in [RELEASING.md](RELEASING.md). A completed local check
does not establish hosted CI, fresh performance measurements, or publication.
The [changelog](../CHANGELOG.md) owns the detailed generated release notes.

<a id="v083-release-preparation"></a>

## v0.8.3 Released Baseline

v0.8.3 was published on October 26, 2026. It required Rust 1.99.1 and included
the following changes since v0.8.2:

- **Checked ownership boundaries:** `Tds` owns Levels 1–2,
  `Triangulation` owns general geometry and topology through Levels 1–4, and
  `DelaunayTriangulation` adds Level 5. General vertex editing, Pachner moves,
  and persistence use these boundaries rather than implicitly retaining a
  Delaunay proof after topology changes.
- **Geometry and recovery fixes:** exterior insertion, hull-corner deletion,
  periodic stellar insertion, point location, duplicate detection, and failed
  insertion recovery preserve their scientific contracts and typed errors.
  Unsupported periodic location and exact intersection dimensions fail
  explicitly; these safeguards do not expand the supported geometry range.
- **Numerical backend adoption:** `la-stack` 0.5.1 supplies the delegated
  numerical kernels, while Delaunay retains geometric certification and
  fallback policy. The built-in `ExactPredicates` contract still stops at D=6.
- **Persistence contracts:** general triangulation snapshots and Delaunay
  checkpoints restore their owning validation levels, preserve coordinate
  storage, and expose recoverable load failures with the required payload bounds.
- **Tooling and release evidence:** the published `research-repo-tools` package
  supplies shared maintenance, managed tools, and evidence primitives. Cargo
  release workflows validate retained archives and trusted exact-commit CI
  evidence; benchmark comparisons retain their scientific provenance rules.

## Release Sequence

The following entries describe earlier baselines and longer-term candidates.
Their compiler versions, measurements, and artifact identities belong to those
releases; they are not current setup instructions.

### v0.7.8 pre-v0.8.0 cleanup (released)

The v0.7.8 line was the final pre-v0.8.0 cleanup baseline. It kept the
performance-summary work from v0.7.7, closes the release-facing documentation
and doctest hygiene pass, tightens default test-suite budgets, validates compact
`T^3` quotient construction, and restores failed topology-repair
workflows to their pre-call state when fallback rebuild does not succeed.

Key takeaways from v0.7.8:

- Default correctness tests now stay under the routine 10-second budget, with
  long-running correctness work routed through `slow-tests`.
- Public examples and doctests avoid release-hostile unwrap/expect patterns in
  favor of typed `Result` wrappers or non-degenerate examples.
- Compact `T^3` quotients are validated through topology and Delaunay
  checks before being returned; higher-dimensional quotient construction fails
  fast until scalable follow-up work lands.
- `delaunayize` now consumes a Levels 1–4 triangulation, keeps intermediate
  flip-repair state private, and publishes only a Levels 1–5 result; raw-TDS
  PL-manifold repair is an orthogonal pre-restoration workflow.
- The release benchmark summary recorded construction throughput, generated
  simplex counts, and circumsphere predicate behavior for that baseline.

### v0.8.0 paper-facing API and topology push (released)

v0.8.0 delivered the paper-facing API and topology work deferred from v0.7.8.
That release required Rust 1.97.1 and included the completed audit of that
toolchain surface. Its [reviewer artifact guide](archive/ARTIFACT.md) retains
the original release identity, commands, and benchmark exception.

- **Pachner/Edit API shape (#252/#253/#350/#337):** unified the Pachner move API,
  expanded public flip benchmark coverage, added Monte-Carlo stress benchmarks,
  and supported periodic external-simplex parity constraints in bistellar flips.
- **Linear algebra and API boundaries (#424):** kept `la-stack` details behind
  `src/geometry/matrix.rs`, preserved typed backend errors in geometry helpers,
  and kept README/API guidance aligned with the focused prelude reference in
  `docs/code_organization.md`.
- **Topology and incidence surface (#359/#304):** added stable incidence queries
  for simplex-local topology and dedicated targeted topology repair stages for
  ridge and vertex-link failures.
- **Transactional rollback architecture (#364):** replaced or centralized
  full-TDS clone rollback where benchmarks show it matters, while preserving
  the current strong failed-mutation rollback guarantee.
- **Naming cleanup (#323):** made the breaking `Cell` → `Simplex` rename.
- **Iterator cleanup (#353):** preferred iterator-based collection-building paths
  where that improves clarity and allocation behavior.
- **Rust 1.97.1 release gate (#329/#496):** raised the v0.8.0 MSRV to Rust
  1.97.1, finished the baseline `assert_matches!` cleanup, audited the new
  integer/`NonZero` bit helpers against Hilbert bit-depth/index invariants,
  reviewed `RepeatN::default` and Cargo 1.97 tooling changes for useful adoption,
  and re-benchmarked predicate `cold_path` decisions under the 1.97.1 compiler.
- **Notebook/export artifact (#64/#408):** shipped the generic simplicial-complex
  JSON export, reproducible quickstart and validation notebooks, tracked
  validation diagrams, and reviewer-facing artifact instructions.

### v0.8.1 validation evidence and performance follow-up

This released baseline completed the following validation work:

- **Level 4 realization validation (#482/#559):** completed the independent
  randomized and degenerate 2D–5D agreement campaign for the revised
  simplex-intersection narrow phase and added focused narrow-phase and
  whole-realization benchmarks.
- **Level 5 Delaunay reporting (#483/#560):** replaced the certified Euclidean
  all-vertices/all-simplices report bottleneck with an O(simplices) local-flip
  path while preserving exhaustive fallback diagnostics, with 2D–5D agreement
  tests and performance canaries.

### v0.8.2 corrected benchmark baseline

v0.8.2 established the corrected benchmark contract and release metadata
baseline. Cross-release performance claims must respect that contract boundary;
old timings are not made comparable by relabeling them. See the
[generated v0.8.2 notes](../CHANGELOG.md) and [benchmark guide](../benches/README.md).

### v0.9.0 and later horizon

These are future candidates, not commitments for v0.8.3 or a fixed v0.9.0
schedule. Scientific correctness and supporting evidence determine readiness.

- **Broader geometry:** constrained Delaunay triangulations
  ([#299](https://github.com/acgetchell/delaunay/issues/299)), Voronoi diagrams
  ([#63](https://github.com/acgetchell/delaunay/issues/63)), and weakly-visible
  hull facets ([#136](https://github.com/acgetchell/delaunay/issues/136)).
- **Concurrent topology mutation:** establish correctness, conflict, rollback,
  and performance evidence before adopting a scheduler
  ([#495](https://github.com/acgetchell/delaunay/issues/495)).
- **Construction rollback:** replace per-insertion topology snapshots with a
  narrower undo log while preserving failure atomicity
  ([#510](https://github.com/acgetchell/delaunay/issues/510)).
- **High-dimensional hull buffers:** investigate allocation beyond D=7
  ([#106](https://github.com/acgetchell/delaunay/issues/106)).
- **Public builder polish:** use downstream experience before finalizing the
  construction API for v1.0.0
  ([#497](https://github.com/acgetchell/delaunay/issues/497)).
- **Shared tooling adapter cleanup:** remove the temporary adapters after their
  upstream replacements are published and adopted
  ([#673](https://github.com/acgetchell/delaunay/issues/673),
  [#676](https://github.com/acgetchell/delaunay/issues/676)); see the
  [completed adoption record](dev/tooling_adoption.md) for context.
- **Spherical integration:** extend the bounded `S^2`/`S^3` prototype to the
  broader spherical construction, realization, and editing contract
  ([#414](https://github.com/acgetchell/delaunay/issues/414)).
- **Toroidal scaling:** make `T^4`/`T^5` periodic quotient construction scalable
  and independently validated before enabling it
  ([#416](https://github.com/acgetchell/delaunay/issues/416)).

## Ongoing Performance Monitoring

- **2D–5D shared large-scale monitoring:** keep
  `just debug-large-scale-{2,3,4,5}d [n] [repair_every]` aligned so
  performance work is measured across the supported small-dimensional range
  instead of tuned for one dimension at another's expense. The `justfile` owns
  current defaults; the [benchmark guide](../benches/README.md#release-mode-debug-defaults)
  owns workload sizes and dated maintainer-hardware timings.
- **Criterion performance canaries:** keep smaller `ci_performance_suite`
  canaries for the same construction path so PR regression checks remain
  practical under Criterion's repeated sampling model.
- **Heavy manual probes:** treat thousands-point 4D and larger 5D workloads as
  explicit characterization work. The completed
  [5D feasibility investigation #342](https://github.com/acgetchell/delaunay/issues/342)
  is historical context, not an open target or an input-size guarantee.
- **Release evidence:** preserve the legacy [performance report](performance.md)
  and dated investigations with their original identities. Publish replacement
  reports and README summaries only from validated retained evidence through
  their named workflows; a documentation refresh does not regenerate timings.

## Not Implemented Today

- Constrained Delaunay triangulations.
- Concurrent topology mutation or multi-threaded construction.
- Full spherical integration beyond the bounded `S^2`/`S^3` prototype, or
  hyperbolic triangulation semantics.
- Massively parallel, GPU, or out-of-core construction.
- Native Rust plotting or viewer APIs; notebook visualization and JSON export
  are already available.
- Voronoi diagram extraction.

See [`limitations.md`](limitations.md) for current operational limits and
[`archive/todo_2026-04-23.md`](archive/todo_2026-04-23.md) for the retired
post-v0.7.5 task snapshot.
