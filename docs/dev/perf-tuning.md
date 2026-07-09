# Performance Tuning

Use this workflow when making performance-sensitive Rust changes. Performance
work in this crate is scientific evidence: measure representative behavior,
preserve invariants, and report exactly what changed.

---

## Contents

- [Core Rule](#core-rule)
- [Benchmark Before Editing](#benchmark-before-editing)
- [Add a Benchmark When None Exists](#add-a-benchmark-when-none-exists)
- [Dimension Coverage](#dimension-coverage)
- [Performance Scope](#performance-scope)
- [Invariant-Preserving Changes](#invariant-preserving-changes)
- [Benchmark After Editing](#benchmark-after-editing)
- [Validation](#validation)
- [Reporting](#reporting)

---

## Core Rule

Do not claim a performance improvement without before/after evidence from the
same representative benchmark command.

The preferred loop is:

1. Identify the hot path and the existing benchmark that covers it.
2. Run the benchmark before editing and record representative results.
3. Make the smallest invariant-preserving change.
4. Rerun the same benchmark and compare the same named cases.
5. Run the appropriate invariant validator before handoff.

## Benchmark Before Editing

Before changing performance-sensitive code, run the smallest benchmark that
covers the claimed hot path. Use the `perf` profile for measured data:

```bash
cargo bench --profile perf --bench <bench-name>
```

Use `just bench-smoke` only for harness validation. Do not treat smoke output
as performance evidence.

Record:

- exact command
- benchmark case names
- before timings or throughput
- whether the benchmark covers the dimensions and input families affected by
  the change

## Add a Benchmark When None Exists

If no benchmark covers the affected hot path, add one before optimizing.

Benchmarks should:

- use deterministic fixtures
- cover realistic sizes for the operation
- include adversarial or near-degenerate inputs when the algorithm is sensitive
  to geometry or topology
- keep setup outside Criterion-measured closures
- avoid logging, parsing, allocation-heavy configuration, or validation work
  inside measured closures unless that is the behavior being measured
- use `#[cfg(feature = "bench")]` fixture helpers only when the benchmark needs
  deliberately invalid-but-structurally-coherent topology

Do not weaken public constructors, validation, typed errors, or topology
guarantees to make benchmark fixtures easier to build.

## Dimension Coverage

For dimension-generic hot paths, benchmarks and correctness tests should cover
2D through 5D whenever feasible.

Prefer per-dimension benchmark groups or cases with explicit names such as:

```text
delete_vertex/success/2d/...
delete_vertex/success/3d/...
delete_vertex/success/4d/...
delete_vertex/success/5d/...
```

If a dimension is intentionally omitted, document why in the benchmark or the
change summary.

## Performance Scope

Performance is a design goal but is strictly subordinate to scientific
invariants: numerical correctness, topological correctness, API stability,
composability, and clarity. Never trade invariants or diagnostics for speed; if
performance and invariants appear to conflict, re-scope the problem.

In scope:

- d-dimensional Delaunay triangulations for small-to-medium dimensions,
  typically `2 <= D <= 7`
- single-threaded in-memory construction
- `DenseSlotMap`-backed topology
- Hilbert-ordered insertion
- allocation-conscious local repair and validation paths

Out of scope:

- massively parallel or GPU meshing
- out-of-core triangulations
- sparse sampling
- dynamic remeshing at scale

Those domains belong to specialized tools such as CGAL, TetGen, or Gmsh.

Within scope, prefer:

- allocation-free hot paths through stack arrays, `SmallBuffer`, borrowed views,
  and iterators
- Shewchuk-style f64 fast filters with cold exact-arithmetic fallbacks
- `const fn` for pure helpers where the inputs allow
- typed flip, insertion, and repair budgets instead of heuristic timeouts

## Invariant-Preserving Changes

Performance fixes must preserve the crate's scientific invariant model.

Prefer optimizations that make existing evidence explicit:

- carry local repair seed scopes instead of rediscovering global state
- reuse validated handles, keys, or proof-bearing types
- move repeated validation out of inner loops only when validation evidence is
  still represented by the API
- reduce allocation through existing stack buffers, borrowed views, or iterator
  streaming
- keep rollback and typed error paths intact

Do not optimize by:

- skipping numerical, topological, or Delaunay validation that protects public
  invariants
- replacing typed errors with `bool`, `Option`, sentinels, strings, or panics
- introducing stale caches without an explicit invalidation story
- changing public semantics without an intentional API decision
- adding `unsafe`

## Benchmark After Editing

After the change, rerun the same benchmark command used for the baseline.

Compare the same named cases. Treat a clear representative regression as a
failed optimization: revert, narrow, or redesign before handoff. Small mixed
changes can be normal for noisy microbenchmarks, but the summary must say so
honestly.

When Criterion reports a statistically significant improvement, include the
before and after medians or the reported percentage change. For example:

```text
4D near-boundary 100 vertices: 291.26 ms -> 52.61 ms, -81.83%
```

## Validation

Benchmarks are not the only invariant oracle, but they must not publish
performance evidence for invariant-violating results.

While iterating, run focused tests that cover the invariant the optimization
relies on. Add a regression test when the optimization depends on internal
evidence that could be lost later, such as a local repair seed scope. Benchmark
harnesses should assert known-answer checks or validation results around the
measured workflow; keep those checks outside the Criterion-measured closure
unless validation itself is the behavior being measured.

For final handoff after Rust code changes, run:

```bash
just ci
```

For documentation-only, configuration-only, or Python-only changes, follow the
command matrix in [`commands.md`](commands.md).

## Reporting

Performance-change summaries should include:

- hot path changed
- invariant-preserving mechanism
- benchmark command
- before/after results for representative 2D-5D cases when applicable
- invariant validation command and result
- any benchmark gaps or dimensions intentionally not covered
