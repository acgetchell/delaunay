# Large-Scale Debug Characterization — May 2026

> Archived operational snapshot. These are single release-mode debug runs on
> maintainer Apple M4 Max-class hardware, not Criterion benchmarks or portable
> performance promises. Current workloads and timings live in the
> [benchmark guide](../../../benches/README.md#release-mode-debug-defaults).

This snapshot preserves the large-scale observations that were formerly
repeated in `docs/limitations.md`. The harness generated deterministic points,
performed batch construction and final flip repair, and required the final
validation report to pass.

## Recorded Runs

| Dimension | Command or scale | Result |
|-----------|------------------|--------|
| 2D | `just debug-large-scale-2d 40000 1` | On 2026-05-14, inserted 40,000/40,000 vertices with zero skips, performed zero final repair flips, passed validation, and completed in about 66 seconds. |
| 3D | `just debug-large-scale-3d 7500 1` | Completed in roughly 56 seconds in the calibration noted by the former limitations page. |
| 3D | `just debug-large-scale-3d 10000 1` | On 2026-05-14, inserted 10,000/10,000 vertices with zero skips, performed zero final repair flips, passed validation, and completed in 68.4 seconds. |
| 4D | `just debug-large-scale-4d 800 1` | Inserted 800/800 vertices with zero skips, performed zero final repair flips, passed validation, and completed in about 52 seconds. |
| 4D | `just debug-large-scale-4d 3000 1` | On 2026-05-14, inserted 3,000/3,000 vertices with zero skips, built 80,441 cells, passed validation, and completed in 421.4 seconds. |
| 5D | 50-vertex debug probe | Completed in about 7 seconds. |
| 5D | `just debug-large-scale-5d 150 1` | On 2026-05-14, inserted 150/150 vertices with zero skips, performed zero final repair flips, passed validation, and completed in about 62 seconds. |
| 5D | `just debug-large-scale-5d 1000 1` | On 2026-05-14, exceeded the 1,800-second harness timeout before emitting the construction summary. |

The 4D 3,000-vertex run and the optimization that made it practical are
documented in more detail in
[`../issue_204_investigation.md`](../issue_204_investigation.md#340-large-scale-characterization-2026-05-14).
Historical correctness reproducers and their resolutions are preserved in
[`../known_issues_4d_2026-04-23.md`](../known_issues_4d_2026-04-23.md).

## Interpretation

- These observations establish tested envelopes on one machine, not supported
  maximum input sizes.
- The rapid reduction in practical fixture size from 2D to 5D reflects
  dimension-dependent topology growth and exact-predicate cost.
- The 5D 1,000-vertex timeout is a feasibility observation, not a correctness
  failure.
- Current command defaults are intentionally absent from this snapshot because
  the `justfile` owns them and they may change as performance changes.
