# Insertion rollback performance

[Issue #510](https://github.com/acgetchell/delaunay/issues/510) tracks
insertion-local rollback. [PR #587](https://github.com/acgetchell/delaunay/pull/587)
introduced the [TDS journal](../src/core/tds/rollback.rs). Public insertion and
batch construction now journal the touched records while preserving owner
identity and topology generation. If a journaled insertion fails, the Delaunay
guard restores insertion state and invalidates the derived spatial index.

The local observations use Apple M4 Max, macOS arm64, Rust 1.100.0, and
cargo-nextest 0.9.152 selected by the pinned research-repo-tools package.
The historical runs use committed source exports and independent build
directories. The two commits bracket a broader PR and have different
dependency locks, so the measurements describe the complete revisions.

## Large-scale smoke guard

The original 60-second cap was identical for all four dimensions. Timed-out
cases provide no completed-work timing. The current run used the production
code at `10d5e181c` with the working tree's dependency pins.

| Dimension | Vertices | Before journals (`b99c3b93`) | Journal merge (`3ef8e4d2`) | `main` at 60 seconds | `main` with revised defaults |
| --- | ---: | --- | --- | --- | --- |
| 2D | 32,000 | Timed out | Timed out | 25.22 s insertion, 31.01 s total | 25.80 s insertion, 31.73 s total |
| 3D | 9,000 | Timed out | Timed out | 40.00 s insertion, 44.59 s total | 40.42 s insertion, 45.11 s total |
| 4D | 1,000 | Timed out | Timed out | Timed out | 102.99 s insertion, 114.17 s total |
| 5D | 160 | Timed out | Timed out | 39.69 s insertion, 48.80 s total | 38.93 s insertion, 48.21 s total |

The same 1,000-point 4D fixture on `main` passed at a 180-second cap. It spent
102.89 s in insertion and 113.91 s total, including conversion and validation
of Levels 1–3 plus Level 5. The full rerun passed with a 180-second default
for 4D and 60-second defaults for 2D, 3D, and 5D.

## Successful construction and insertion

Both historical revisions passed untimed preflight for the same eight
well-conditioned cases. Criterion retained ten samples per construction
case and fifteen per insertion case. RRT verified the raw case inventory,
sample counts, and 95% confidence intervals. Each insertion measurement is
for the entire batch against a prepared triangulation.

```bash
selector='^(tds_new_[2-5]d/tds_new/|incremental_insert/insert_[2-5]d/)'
cargo bench --locked --profile perf --bench ci_performance_suite -- --test "$selector"
cargo bench --locked --profile perf --bench ci_performance_suite -- \
  --noplot --warm-up-time 1 --measurement-time 1 "$selector"
```

Medians and marginal 95% intervals are in milliseconds.

| Workload | Before journals (`b99c3b93`) | Journal merge (`3ef8e4d2`) |
| --- | ---: | ---: |
| 2D construction, 4,000 vertices | 4,891.90 (4,779.46–4,990.35) | 4,250.17 (4,199.96–4,275.38) |
| 3D construction, 750 vertices | 1,681.63 (1,656.13–1,981.78) | 1,289.59 (1,271.30–1,314.03) |
| 4D construction, 75 vertices | 794.75 (793.27–809.56) | 768.94 (763.02–788.12) |
| 5D construction, 25 vertices | 726.52 (593.23–798.12) | 566.94 (550.63–584.67) |
| 2D insertion, 10 vertices | 3.438 (3.399–3.506) | 0.978 (0.970–0.984) |
| 3D insertion, 10 vertices | 5.200 (5.184–5.262) | 3.716 (3.692–3.742) |
| 4D insertion, 6 vertices | 27.910 (27.841–27.950) | 28.371 (28.280–28.566) |
| 5D insertion, 4 vertices | 315.503 (315.241–317.296) | 314.184 (312.873–315.867) |

Construction medians fell in all four dimensions. The 2D and 3D insertion
medians fell, 4D was slightly higher, and the 5D insertion intervals overlapped.

## Rejected insertion

The same [3D benchmark fixture](../benches/insertion_rollback.rs) ran on all
three revisions. Its kernel injects a typed conflict-predicate failure after
public insertion stages the candidate vertex. Each timed attempt checks that
error. Repeating insertions against the same owner includes any spatial-index
rebuild needed after the preceding failure. Untimed checks verify that the
owner's vertex and simplex keys and topology generation are preserved and that
the triangulation passes Levels 1–5.

To reproduce the two historical columns, run this from the checkout containing
the fixture. The only additions to each source export are the unchanged fixture
and its Cargo target. Keep each revision's `Cargo.lock`.

```bash
scratch=$(mktemp -d)
for revision in b99c3b93 3ef8e4d2; do
  export_dir="$scratch/$revision"
  mkdir "$export_dir"
  git --no-pager archive "$revision" | tar -x -C "$export_dir"
  cp benches/insertion_rollback.rs "$export_dir/benches/"
  cat >> "$export_dir/Cargo.toml" <<'TOML'

[[bench]]
name = "insertion_rollback"
path = "benches/insertion_rollback.rs"
harness = false
TOML
done
```

Run the untimed preflight and benchmark from each export directory and from the
current checkout. Use Rust 1.100.0 for all runs, including the exports that pin
1.98.0. Each run writes build artifacts to its own source directory.

```bash
export CARGO_TARGET_DIR="$PWD/target"
cargo +1.100.0 bench --locked --profile perf --bench insertion_rollback -- --test
cargo +1.100.0 bench --locked --profile perf --bench insertion_rollback -- \
  --noplot --warm-up-time 1 --measurement-time 1
```

RRT verified 100 samples per case. Medians and marginal 95% intervals are in
microseconds per rejected insertion.

| Existing vertices | Before journals (`b99c3b93`) | Journal merge (`3ef8e4d2`) | `main` (`10d5e181c` with working-tree pins) |
| ---: | ---: | ---: | ---: |
| 16 | 10.110 (10.083–10.130) | 3.647 (3.638–3.658) | 3.723 (3.705–3.743) |
| 128 | 84.663 (84.290–84.867) | 9.914 (9.879–9.928) | 9.858 (9.842–9.882) |

The 128-vertex case showed the larger reduction in median rejection time.

## Rollback correctness

The TDS regressions cover touched-record capture, nested rollback, and retry
with the original keys, owner identity, and topology generation in 2D–5D.
The Delaunay insertion regressions verify exact restoration after repair
and validation failures. The incremental builder regression verifies that
failed publication restores the spatial index before a successful retry.
See the tests in [TDS rollback](../src/core/tds/rollback.rs),
[Delaunay insertion](../src/delaunay/insertion.rs), and the
[incremental builder](../src/delaunay/incremental_builder.rs).

`just ci` passed on macOS, including the rollback regressions in debug and
release profiles and compilation of the new rollback benchmark.
