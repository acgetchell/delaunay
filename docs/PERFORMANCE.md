# Benchmark Performance

> [!WARNING]
> Legacy performance evidence: this report predates the artifact-backed
> promotion contract, and its exact CSV/provenance source pair is unavailable.
> Treat the historical timings and environment fields below as provenance-limited.
> The next curated promotion will replace this page with a generated report and
> retain its exact evidence under `docs/archive/performance/data/`.

**delaunay** v0.8.0 · `1f15b9352` (HEAD) · 2026-07-27 23:33:25 UTC
**Statistic**: median
**Suite**: release-signal
**Scope**: release-signal

## Environment

- **Cargo profile**: `perf`
- **Raw Criterion data**: `target/criterion/`
- **OS**: macOS
- **CPU**: Apple M4 Max (16 cores, 16 threads)
- **Memory**: 64.0 GB
- **Rust**: rustc 1.97.1 (8bab26f4f 2026-07-14)
- **Target**: aarch64-apple-darwin

## Benchmark Results

Comparison against baseline **v0.7.8**:

Negative change = faster. Speedup > 1.00x = improvement.

### 2d_insphere

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| 2d_insphere | 17.5 ns | 20.1 ns | +14.9% | 0.87x |

### 2d_insphere_distance

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| 2d_insphere_distance | 27.0 ns | 20.7 ns | **-23.3%** | 1.30x |

### 2d_insphere_lifted

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| 2d_insphere_lifted | 8.2 ns | 11.7 ns | +42.5% | 0.70x |

### 3d_insphere

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| 3d_insphere | 2.17 µs | 1.47 µs | **-32.2%** | 1.48x |

### 3d_insphere_distance

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| 3d_insphere_distance | 26.1 ns | 35.2 ns | +34.8% | 0.74x |

### 3d_insphere_lifted

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| 3d_insphere_lifted | 18.7 ns | 19.4 ns | +3.9% | 0.96x |

### 4d_insphere

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| 4d_insphere | 5.23 µs | 4.38 µs | **-16.2%** | 1.19x |

### 4d_insphere_distance

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| 4d_insphere_distance | 54.9 ns | 87.4 ns | +59.1% | 0.63x |

### 4d_insphere_lifted

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| 4d_insphere_lifted | 2.88 µs | 2.33 µs | **-19.0%** | 1.23x |

### 5d_insphere

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| 5d_insphere | 8.44 µs | 7.14 µs | **-15.4%** | 1.18x |

### 5d_insphere_distance

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| 5d_insphere_distance | 82.2 ns | 102.6 ns | +24.8% | 0.80x |

### 5d_insphere_lifted

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| 5d_insphere_lifted | 4.91 µs | 4.30 µs | **-12.4%** | 1.14x |

### bistellar_flips_4d

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| k1_roundtrip | 180.50 µs | 592.80 ms | +328318.3% | 0.00x |
| k2_roundtrip | 180.61 µs | 631.28 ms | +349430.8% | 0.00x |
| k3_roundtrip | 181.34 µs | 588.72 ms | +324551.4% | 0.00x |

### boundary_facets

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| boundary_facets_2d/4000 | 1.70 ms | 1.76 ms | +3.4% | 0.97x |
| boundary_facets_2d_adversarial/4000 | 1.70 ms | 1.76 ms | +3.2% | 0.97x |
| boundary_facets_3d/750 | 1.48 ms | 1.58 ms | +6.5% | 0.94x |
| boundary_facets_3d_adversarial/750 | 1.54 ms | 1.62 ms | +5.1% | 0.95x |
| boundary_facets_4d/75 | 483.70 µs | 568.61 µs | +17.6% | 0.85x |
| boundary_facets_4d_adversarial/75 | 508.26 µs | 580.58 µs | +14.2% | 0.88x |
| boundary_facets_5d/25 | 245.84 µs | 315.14 µs | +28.2% | 0.78x |
| boundary_facets_5d_adversarial/25 | 235.00 µs | 293.67 µs | +25.0% | 0.80x |

### convex_hull

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| from_triangulation_2d/4000 | 1.70 ms | 1.81 ms | +6.5% | 0.94x |
| from_triangulation_2d_adversarial/4000 | 1.72 ms | 1.77 ms | +2.5% | 0.98x |
| from_triangulation_3d/750 | 1.48 ms | 1.59 ms | +7.0% | 0.93x |
| from_triangulation_3d_adversarial/750 | 1.51 ms | 1.61 ms | +6.2% | 0.94x |
| from_triangulation_4d/75 | 479.42 µs | 569.31 µs | +18.8% | 0.84x |
| from_triangulation_4d_adversarial/75 | 500.87 µs | 589.94 µs | +17.8% | 0.85x |
| from_triangulation_5d/25 | 243.74 µs | 318.63 µs | +30.7% | 0.76x |
| from_triangulation_5d_adversarial/25 | 238.16 µs | 293.31 µs | +23.2% | 0.81x |

### convex_hull_queries

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| find_nearest_visible_facet_3d/750 | 25.85 µs | 26.68 µs | +3.2% | 0.97x |
| find_nearest_visible_facet_3d_adversarial/750 | 540.65 µs | 325.11 µs | **-39.9%** | 1.66x |
| find_visible_facets_3d/750 | 24.23 µs | 23.42 µs | **-3.4%** | 1.03x |
| find_visible_facets_3d_adversarial/750 | 535.71 µs | 321.04 µs | **-40.1%** | 1.67x |
| is_point_outside_3d/750 | 24.38 µs | 23.60 µs | **-3.2%** | 1.03x |
| is_point_outside_3d_adversarial/750 | 533.14 µs | 328.41 µs | **-38.4%** | 1.62x |

### edge_cases_2d_boundary_point_distance

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_2d_boundary_point_distance | 22.3 ns | 20.8 ns | **-6.5%** | 1.07x |

### edge_cases_2d_boundary_point_insphere

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_2d_boundary_point_insphere | 1.3 ns | 1.0 ns | **-20.5%** | 1.26x |

### edge_cases_2d_boundary_point_lifted

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_2d_boundary_point_lifted | 194.3 ns | 91.7 ns | **-52.8%** | 2.12x |

### edge_cases_2d_far_point_distance

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_2d_far_point_distance | 22.4 ns | 20.7 ns | **-7.9%** | 1.09x |

### edge_cases_2d_far_point_insphere

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_2d_far_point_insphere | 17.4 ns | 19.6 ns | +12.7% | 0.89x |

### edge_cases_2d_far_point_lifted

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_2d_far_point_lifted | 8.0 ns | 11.7 ns | +46.2% | 0.68x |

### edge_cases_2d_near_boundary_distance

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_2d_near_boundary_distance | 22.5 ns | 20.7 ns | **-8.3%** | 1.09x |

### edge_cases_2d_near_boundary_insphere

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_2d_near_boundary_insphere | 17.5 ns | 21.4 ns | +22.2% | 0.82x |

### edge_cases_2d_near_boundary_lifted

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_2d_near_boundary_lifted | 8.2 ns | 13.7 ns | +66.0% | 0.60x |

### edge_cases_3d_boundary_point_distance

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_3d_boundary_point_distance | 25.9 ns | 35.3 ns | +36.2% | 0.73x |

### edge_cases_3d_boundary_point_insphere

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_3d_boundary_point_insphere | 1.3 ns | 1.3 ns | **-1.8%** | 1.02x |

### edge_cases_3d_boundary_point_lifted

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_3d_boundary_point_lifted | 427.0 ns | 167.5 ns | **-60.8%** | 2.55x |

### edge_cases_3d_far_point_distance

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_3d_far_point_distance | 25.7 ns | 35.6 ns | +38.7% | 0.72x |

### edge_cases_3d_far_point_insphere

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_3d_far_point_insphere | 2.11 µs | 1.13 µs | **-46.6%** | 1.87x |

### edge_cases_3d_far_point_lifted

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_3d_far_point_lifted | 18.0 ns | 19.7 ns | +9.5% | 0.91x |

### edge_cases_3d_near_boundary_distance

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_3d_near_boundary_distance | 25.9 ns | 34.8 ns | +34.5% | 0.74x |

### edge_cases_3d_near_boundary_insphere

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_3d_near_boundary_insphere | 3.55 µs | 3.16 µs | **-11.1%** | 1.12x |

### edge_cases_3d_near_boundary_lifted

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_3d_near_boundary_lifted | 18.0 ns | 24.1 ns | +33.7% | 0.75x |

### edge_cases_4d_boundary_point_distance

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_4d_boundary_point_distance | 54.6 ns | 70.4 ns | +28.9% | 0.78x |

### edge_cases_4d_boundary_point_insphere

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_4d_boundary_point_insphere | 1.5 ns | 1.2 ns | **-25.1%** | 1.34x |

### edge_cases_4d_boundary_point_lifted

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_4d_boundary_point_lifted | 1.49 µs | 881.4 ns | **-40.9%** | 1.69x |

### edge_cases_4d_far_point_distance

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_4d_far_point_distance | 54.8 ns | 70.8 ns | +29.2% | 0.77x |

### edge_cases_4d_far_point_insphere

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_4d_far_point_insphere | 3.38 µs | 1.76 µs | **-47.9%** | 1.92x |

### edge_cases_4d_far_point_lifted

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_4d_far_point_lifted | 1.87 µs | 981.2 ns | **-47.7%** | 1.91x |

### edge_cases_4d_near_boundary_distance

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_4d_near_boundary_distance | 54.9 ns | 70.3 ns | +28.1% | 0.78x |

### edge_cases_4d_near_boundary_insphere

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_4d_near_boundary_insphere | 6.09 µs | 5.46 µs | **-10.3%** | 1.11x |

### edge_cases_4d_near_boundary_lifted

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_4d_near_boundary_lifted | 3.07 µs | 2.72 µs | **-11.6%** | 1.13x |

### edge_cases_5d_boundary_point_distance

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_5d_boundary_point_distance | 82.1 ns | 102.0 ns | +24.2% | 0.81x |

### edge_cases_5d_boundary_point_insphere

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_5d_boundary_point_insphere | 2.5 ns | 1.4 ns | **-43.3%** | 1.76x |

### edge_cases_5d_boundary_point_lifted

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_5d_boundary_point_lifted | 2.33 µs | 1.41 µs | **-39.4%** | 1.65x |

### edge_cases_5d_far_point_distance

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_5d_far_point_distance | 82.2 ns | 101.8 ns | +23.8% | 0.81x |

### edge_cases_5d_far_point_insphere

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_5d_far_point_insphere | 5.19 µs | 2.72 µs | **-47.5%** | 1.91x |

### edge_cases_5d_far_point_lifted

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_5d_far_point_lifted | 2.94 µs | 1.63 µs | **-44.6%** | 1.81x |

### edge_cases_5d_near_boundary_distance

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_5d_near_boundary_distance | 82.5 ns | 101.8 ns | +23.4% | 0.81x |

### edge_cases_5d_near_boundary_insphere

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_5d_near_boundary_insphere | 9.91 µs | 8.71 µs | **-12.1%** | 1.14x |

### edge_cases_5d_near_boundary_lifted

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| edge_cases_5d_near_boundary_lifted | 5.07 µs | 4.69 µs | **-7.4%** | 1.08x |

### incremental_insert

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| insert_2d/10 | 4.54 ms | 5.54 ms | +22.0% | 0.82x |
| insert_2d_adversarial/10 | 6.94 ms | 6.74 ms | **-3.0%** | 1.03x |
| insert_3d/10 | 6.11 ms | 5.44 ms | **-11.0%** | 1.12x |
| insert_3d_adversarial/10 | 45.11 ms | 39.17 ms | **-13.2%** | 1.15x |
| insert_4d/6 | 46.84 ms | 28.78 ms | **-38.6%** | 1.63x |
| insert_4d_adversarial/6 | 148.27 ms | 118.53 ms | **-20.1%** | 1.25x |
| insert_5d/4 | 620.19 ms | 452.37 ms | **-27.1%** | 1.37x |
| insert_5d_adversarial/4 | 610.70 ms | 442.25 ms | **-27.6%** | 1.38x |

### predicates_hot

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| insphere_2d/10000 | 121.85 µs | 205.73 µs | +68.8% | 0.59x |
| insphere_3d/10000 | 32.50 ms | 25.40 ms | **-21.9%** | 1.28x |
| insphere_4d/10000 | 54.22 ms | 47.01 ms | **-13.3%** | 1.15x |
| insphere_5d/10000 | 87.09 ms | 75.06 ms | **-13.8%** | 1.16x |
| insphere_lifted_2d/10000 | 68.61 µs | 113.38 µs | +65.2% | 0.61x |
| insphere_lifted_3d/10000 | 156.42 µs | 193.24 µs | +23.5% | 0.81x |
| insphere_lifted_4d/10000 | 30.81 ms | 26.21 ms | **-14.9%** | 1.18x |
| insphere_lifted_5d/10000 | 51.78 ms | 44.30 ms | **-14.5%** | 1.17x |

### predicates_near_boundary

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| insphere_2d/1000 | 11.77 µs | 19.93 µs | +69.3% | 0.59x |
| insphere_3d/1000 | 3.34 ms | 2.57 ms | **-23.2%** | 1.30x |
| insphere_4d/1000 | 5.58 ms | 4.75 ms | **-14.9%** | 1.17x |
| insphere_lifted_2d/1000 | 6.89 µs | 11.37 µs | +65.1% | 0.61x |
| insphere_lifted_3d/1000 | 15.92 µs | 19.62 µs | +23.2% | 0.81x |
| insphere_lifted_4d/1000 | 3.06 ms | 2.51 ms | **-18.1%** | 1.22x |

### random_insphere_1000_queries

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| random_insphere_1000_queries | 6.65 ms | 5.62 ms | **-15.5%** | 1.18x |

### random_insphere_distance_1000_queries

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| random_insphere_distance_1000_queries | 39.82 µs | 49.46 µs | +24.2% | 0.80x |

### random_insphere_lifted_1000_queries

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| random_insphere_lifted_1000_queries | 22.92 µs | 27.25 µs | +18.9% | 0.84x |

### tds_new_2d

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| tds_new/4000 | 1.19 s | 1.27 s | +7.1% | 0.93x |
| tds_new_adversarial/4000 | 1.33 s | 1.37 s | +3.4% | 0.97x |

### tds_new_3d

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| tds_new/750 | 899.38 ms | 776.60 ms | **-13.7%** | 1.16x |
| tds_new_adversarial/750 | 1.15 s | 921.80 ms | **-19.9%** | 1.25x |

### tds_new_4d

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| tds_new/75 | 886.29 ms | 641.04 ms | **-27.7%** | 1.38x |
| tds_new_adversarial/75 | 959.67 ms | 676.12 ms | **-29.5%** | 1.42x |

### tds_new_5d

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| tds_new/25 | 986.17 ms | 724.38 ms | **-26.5%** | 1.36x |
| tds_new_adversarial/25 | 874.73 ms | 616.45 ms | **-29.5%** | 1.42x |

### validation

| Benchmark | v0.7.8 | Latest | Change | Speedup |
|-----------|-------:|-------:|-------:|--------:|
| validate_3d/750 | 35.73 ms | 8.64 s | +24088.1% | 0.00x |
| validate_3d_adversarial/750 | 46.44 ms | 12.89 s | +27647.3% | 0.00x |
| validate_4d/75 | 72.32 ms | 23.75 s | +32746.4% | 0.00x |
| validate_4d_adversarial/75 | 68.22 ms | 32.97 s | +48232.3% | 0.00x |
| validate_5d/25 | 62.27 ms | 33.31 s | +53396.3% | 0.00x |
| validate_5d_adversarial/25 | 57.29 ms | 41.63 s | +72559.1% | 0.00x |

## How to Update

Local performance reports are generated in isolated temporary worktrees:

```bash
# Local development: compare the current tree with the latest release
just performance-local

# Release PR: measure, retain, validate, and promote documentation
just performance-release

# Rebuild and promote documentation from retained CSV/JSON only
just performance-doc

# GitHub Release benchmark assets
just performance-github-assets

# Explicit repair
just performance-release <current-tag> <previous-tag>
```

`just performance-local` writes `performance.md` plus retained `performance.csv` and
`performance.provenance.json` under `target/bench-reports/` without promoting documentation.
`just performance-github-assets` writes a provenance-validated
`github-assets-performance.*` bundle without local Cargo benchmark runs. New archives must
bind their complete metadata to the requested clean tag; existing legacy archives remain
loadable as provenance-limited absolute timing evidence. GitHub-asset ratios are always
suppressed because separately hosted release runs are distinct measurement sessions. `just
performance-doc` consumes the retained canonical CSV/JSON pair without Cargo or measurement
worktrees and rejects incomplete, invalid, stale, same-version, or scientifically
non-comparable inputs. `just performance-release` retains and reload-validates the same bundle,
copies the exact pair to `docs/archive/performance/data/`, and promotes documentation with
per-file atomic replacement plus caught-failure rollback. Scratch reports identify their
adjacent evidence pair; promoted reports identify the durable archived pair.

CSV is the canonical tabular artifact because these small audit records are diffable and usable
without a dataframe runtime. Notebooks may derive Parquet caches for analysis, but Parquet is not
an accepted promotion input and must be regenerated from the validated CSV.

Release-comparison commands are release evidence, not routine pre-`just ci` checks.
Older curated reports and exact evidence for new promotions are archived in
`docs/archive/performance/`.

See `benches/README.md` for the full Delaunay benchmark workflow.
