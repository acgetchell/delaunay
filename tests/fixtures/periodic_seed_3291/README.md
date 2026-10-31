# Seed-3291 periodic realization fixture

These are exact TDS snapshots for [Delaunay #668](https://github.com/acgetchell/delaunay/issues/668).
They were captured through the public `geometry().triangulation().clone().into_tds()`
API from causal-triangulations commit
[`1af313bdaa8a5c7c9176a99784e580fd95f22c12`](https://github.com/acgetchell/causal-triangulations/commit/1af313bdaa8a5c7c9176a99784e580fd95f22c12),
using its published Delaunay 0.8.3 dependency. No downstream source was modified.

The original input is `from_toroidal_cdt(4, 3)`, period vector `(4, 3)`, RNG seed
3291, and five attempts. The required outcomes remain:

| Attempt | Move | Outcome |
| --- | --- | --- |
| 1 | Move22 | Success |
| 2 | Move22 | Success |
| 3 | Move31Remove | Success |
| 4 | Move31Remove | GeometricViolation |
| 5 | Move22 | Success |

`before_final_move.cbor` is the state after attempt 4; `after_final_move.cbor` is
the state after attempt 5. Both have 11 vertices, 33 edges, 22 triangles and slice
profile `[3, 4, 4]`. UUIDs identify this capture; they are not RNG output. The
snapshots preserve coordinate bits, oriented vertex order, periodic offsets,
adjacency, and payloads (`u32` vertices and `i32` simplices). Runtime keys are
rebuilt by the public TDS deserializer.

CBOR is used because it encodes binary floating-point values directly. Ordinary
JSON decoding without `serde_json/float_roundtrip` can change the original bits.
The tests use the existing `ciborium` decoder with no custom coordinate parser.
These two small fixtures are included in the Cargo package so package consumers
can run the same regression. They are immutable regression evidence, not outputs
to regenerate during CI.

## Independent geometric witness

Let `R(x)` be the exact rational value of binary64 literal `x`. The reported
triangles are `ABC` and `CDE`, with common-chart coordinates:

```text
A = (R(3.0054768045463973) - 4, R(1.9708567415410918) - 3)
B = (R(1.0022551552419197),     R(0.012991573136809986))
C = (R(3.024162370210636) - 4,  R(0.01790730372627647))
D = (R(2.9835695878950927) - 4, R(1.0256320223226238))
E = (R(2.0132087630627957) - 4, R(2.9841994386833512) - 3)
```

The stored charts put `ABC` at offsets `[[-1,-1], [0,0], [-1,0]]` and `CDE` at
`[[0,1], [0,1], [0,0]]`. Translating the latter by `[-1,-1]` gives the common
chart above. The directed line `C -> A` places `B` strictly on its positive side,
and `D` and `E` strictly on its negative side. Convexity proves the intersection
is exactly `C`. The unit test checks these signs with a direct rational 2×2
determinant, independent of the production orientation and LP routines.

The old path computes `fl(fl(C.y + 3) - 3) = C.y - 2^-52`. This inconsistent copy
crosses the separating line into `ABC`, creating the false intersection. Exact
predicates cannot undo this prior rounding. The fix composes integer offsets and
evaluates each image from its canonical coordinate once.

## Local and global validation

The local edit proof runs before the rollback journal commits. Finalizing deleted
simplex tombstones removes dense-storage entries and can change iteration order.
Pair checks use that order to choose which simplex to translate. Thus the local
check can compare `CDE` against a translated `ABC`, while the later global scan
compares `ABC` against a translated `CDE` and exposes the double rounding.

The public regression replays the fifth Pachner move and validates the committed
state, then independently restores the exact final snapshot. The latter also
preserves the original failing storage order. Unit tests check every local scope,
both pair orders, alternative stored charts, equivalent translation paths in
2D–5D, the exact separating-line proof, and an actual-overlap control.

Publication of the corrected crate and downstream adoption remain separate
release gates; these fixtures do not claim that either has occurred.
