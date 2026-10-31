# la-stack 0.5.2 kernel adoption

This records the local implementation and measurements for
[#654](https://github.com/acgetchell/delaunay/issues/654),
[#655](https://github.com/acgetchell/delaunay/issues/655),
[#656](https://github.com/acgetchell/delaunay/issues/656),
[#657](https://github.com/acgetchell/delaunay/issues/657), and
[#659](https://github.com/acgetchell/delaunay/issues/659).

## Scientific contracts

- Exact LP reduced costs, objectives and dual certificates use the borrowed,
  shape-checked `rational_dot` API. Full objectives include 28 entries at D=8;
  they are not truncated to the square basis size.
- Shared-face pivot selection uses `column_basis`, with absolute tolerance
  `1e-12 * max(1, max |a_ij|)` and no equilibration. Zero rows pad the admitted
  affine shape; no input columns are dropped. The result remains provisional.
  Original binary64 coordinate equations are solved exactly with `solve_exact`;
  derived rational LP equations use `RationalMatrix::solve` without conversion
  through binary64. Both routes use published la-stack arithmetic.
- Circumsphere bounds use `IntervalMatrix::solve`. Uncertain pivots and exhausted
  finite range decline the broad phase and retain exhaustive robust predicates.
  Geometry-specific equation assembly, radius bounds and coordinate boxes stay
  in Delaunay.
- SoS uses `SparsePolynomial` and `polynomial_determinant`. Base-3 variable
  assignment, infinitesimal ordering, geometric lifting and parity remain local.
  The complete D≤6 envelope retains 1,024 subset transitions and 55,631,284
  coefficient products. No coefficient truncation or arbitrary sign is allowed.
  Zero polynomials, budget exhaustion and exponent overflow remain distinct.
- Standalone labeled-simplex intersections support D≤8, with exact and
  provisional systems through order ten. Active sets skip only proven singular
  candidates, propagate other solver failures, and stop at an overlap witness.
  The 50,000-candidate limit covers the complete D=8 envelope of 43,758 sets.
  Both simplex bases are checked before high-dimensional separator shortcuts.
  Construction, TDS arity, `ExactPredicates` and complete SoS limits are unchanged.

Typed solver and budget errors are forwarded through realization validation.
`SimplexIntersectionFailure` retains `PartialEq` but no longer derives `Eq`,
because its typed backend diagnostics can contain floating-point values.

The manifest and both maintained Cargo locks already selected the published
la-stack 0.5.2 release at the start of this work. No dependency override or lock
change was needed. Cargo fetched the registry release with checksum
`7d40eeedfbecba074565984936399673de3c8daa4fc5fcd586c479998d41e89a`.

## Independent correctness evidence

Known-solution 9×9 and 10×10 systems force row pivoting, use non-dyadic
coefficients, cover singularity and malformed shapes, and exercise exponents
outside binary64's range. Independent rational multiplication verifies
`A*x == b`. The 7D/8D intersection fixtures cover disjoint pairs, shared faces,
interior overlap, coincident geometry with disjoint labels, and collapsed
simplices. Dimensions above eight remain explicitly unsupported.

Existing 2D–6D intersection agreement and SoS tests remain in place. A separate
six-permutation expansion checks small polynomial determinants, including row
swaps and exact cancellation. Focused tests distinguish zero polynomials from
structural/product budget exhaustion, exponent overflow and malformed shapes.

All 30 analytic sphere fixtures (D=1 through D=6, scales `2^-500`, `2^-50`,
`1`, `2^50`, `2^500`) receive certified boxes. Exact rational inequalities check
the entire sphere enclosure. Existing independent robust/exact predicate tests
check that broad-phase exclusions never discard an inside or boundary point.
Two nonsingular adversarial fixtures exercise inconclusive and unrepresentable
solves. A cancellation fixture and a one-dimensional smallest-subnormal segment
are successfully enclosed; these are not assumed to fail merely because they
are numerically difficult.

Final `just ci` passed on native aarch64 macOS: 874 Python/tooling tests,
2,902 debug unit tests, 2,901 release unit tests, 822 integration tests,
72 CLI tests, 772 library doctests and 15 README doctests, plus the separate
optional-feature checks. The canonical validation figures matched the tracked
set; source scans, documentation checks, benchmark compilation and all examples
also passed. The complete log is retained with the local evidence below.

## Measurement method

Measurements use the same local macOS 27.2 arm64 host, Rust 1.99.1, Criterion
0.8.2 and locked dependencies. Baseline production source is
`f6ebaed0033ca7d46f273eb460ce41f30b310134`. The five coincident-LP benchmark cases
were added before measuring the baseline; their fixtures and timed return
boundary are identical in the final comparison. Four new 7D/8D cases have only
current measurements and carry no speedup claim.

All runs use `cargo bench --locked --profile perf`, ten samples, one second of
warmup and a requested two-second measurement period. The validation-components
harness deliberately overrides measurement time to 15 seconds; expensive SoS
cases extend sampling to obtain ten observations. Fixture construction and
correctness admission occur outside timing. Measurements run sequentially.

These are focused local diagnostics, not release-wide performance claims.
Raw Criterion estimates, confidence intervals and samples remain under
`target/criterion`, with baseline name `group4-before`. Initial adapter trials
are retained separately under `target/group4/initial-adapter`; they are not the
final comparison. Reducing padding and avoiding an unnecessary intermediate
rational copy recovered the shared-face regression found in that trial. The
remaining mixed whole-triangulation results are reported below.

Allocation counts were not measured. Rational arithmetic still allocates for
arbitrary-precision values; runtime exact dot products validate their inputs.
SoS transfers owned entries into fixed-size arrays without deep-cloning them,
and the upstream determinant owns its sparse subset states. Its work budget
bounds combinatorial products, not rational bit sizes or allocated bytes.

## Results

The tables include all 35 before/after cases. Values are Criterion median point
estimates in microseconds; changes are computed from those medians. The retained
raw estimates include 95% confidence intervals and all ten samples per case.
Small differences should not be interpreted as general performance guarantees.

| Workload | D | Before (µs) | After (µs) | Change |
|---|---:|---:|---:|---:|
| Coincident LP | 2 | 20.801 | 16.360 | -21.35% |
| Coincident LP | 3 | 31.225 | 24.964 | -20.05% |
| Coincident LP | 4 | 48.665 | 39.834 | -18.15% |
| Coincident LP | 5 | 69.085 | 57.327 | -17.02% |
| Coincident LP | 6 | 91.887 | 77.765 | -15.37% |
| Shared face | 2 | 0.092896 | 0.090190 | -2.91% |
| Shared face | 3 | 5.843 | 5.697 | -2.51% |
| Shared face | 4 | 9.349 | 9.085 | -2.82% |
| Shared face | 5 | 12.749 | 12.333 | -3.27% |
| Near-degenerate shared face | 2 | 1.185 | 1.156 | -2.49% |
| Near-degenerate shared face | 3 | 19.390 | 18.820 | -2.94% |
| Near-degenerate shared face | 4 | 33.030 | 32.498 | -1.61% |
| Near-degenerate shared face | 5 | 48.907 | 47.896 | -2.07% |
| Single shared vertex | 2 | 0.090592 | 0.089970 | -0.69% |
| Single shared vertex | 3 | 0.172972 | 0.168836 | -2.39% |
| Single shared vertex | 4 | 1.346 | 1.378 | +2.41% |
| Single shared vertex | 5 | 2.226 | 2.257 | +1.39% |

| Degenerate predicate | D | Before (µs) | After (µs) | Change |
|---|---:|---:|---:|---:|
| Orientation | 2 | 4.789 | 3.376 | -29.50% |
| Orientation | 3 | 24.602 | 15.145 | -38.44% |
| Orientation | 4 | 149.975 | 86.391 | -42.40% |
| Orientation | 5 | 1,084.799 | 588.807 | -45.72% |
| Orientation | 6 | 8,876.950 | 4,930.108 | -44.46% |
| In-sphere | 2 | 66.907 | 43.894 | -34.40% |
| In-sphere | 3 | 547.639 | 341.752 | -37.60% |
| In-sphere | 4 | 5,323.857 | 2,978.228 | -44.06% |
| In-sphere | 5 | 49,238.906 | 31,030.805 | -36.98% |
| In-sphere | 6 | 524,138.958 | 311,892.229 | -40.49% |

| Complete check | D | Before (µs) | After (µs) | Change |
|---|---:|---:|---:|---:|
| Realization, 500 vertices | 2 | 6,782.767 | 6,852.263 | +1.02% |
| Realization, 20 vertices | 3 | 104,279.625 | 104,415.573 | +0.13% |
| Realization, 10 vertices | 4 | 103,878.386 | 103,304.729 | -0.55% |
| Realization, 8 vertices | 5 | 14,553.804 | 14,803.797 | +1.72% |
| Delaunay validity | 2 | 67.805 | 67.301 | -0.74% |
| Delaunay validity | 3 | 420.594 | 419.835 | -0.18% |
| Delaunay validity | 4 | 3,436.350 | 3,470.435 | +0.99% |
| Delaunay validity | 5 | 21,450.509 | 21,607.167 | +0.73% |

The strongest gains are the exact LP and degenerate SoS paths. Whole realization
does not uniformly improve: the 5D case has a small statistically detectable
regression (Criterion's change interval is +1.08% to +2.24%). Shared-vertex
microbenchmarks also increase in 4D/5D. The four Delaunay-validity comparisons
show no statistically detected change at Criterion's 5% significance threshold.
These end-to-end measurements include the interval broad phase and its exhaustive
fallback; they do not isolate interval arithmetic cost. Certified-enclosure
success is separately established by the 30 exact analytic fixtures above.

The new standalone dimensions have no supported 8D baseline. Current-only
medians, with their 95% intervals, are:

| Workload | D | Median (µs) | 95% interval (µs) |
|---|---:|---:|---:|
| Coincident LP | 7 | 108.251 | 106.922–108.585 |
| Coincident LP | 8 | 135.612 | 135.492–136.626 |
| Shared face | 7 | 70.536 | 70.384–70.646 |
| Shared face | 8 | 93.961 | 93.556–98.698 |

## Reproduction and retained evidence

The final source overlay is identified by SHA-256
`208380195424aff4f6c31f5b7bb0fc0af62da9b0138d8a730becf0b3582802a7`,
computed with `git diff HEAD --binary -- src benches | shasum -a 256` against the
baseline commit above. The root Cargo lock SHA-256 is
`4e7e1c7dff267618e82cfa50322e95731258d9fbd77504ddf870d1327f6c8fa2`.

Run each command sequentially on the same host. First prepare a separate checkout
of baseline commit `f6ebaed0033ca7d46f273eb460ce41f30b310134`. Add only
`register_overlap`, `bench_realization_lp`, the `SimplexIntersectionFailure`
import, and `bench_realization_lp`'s `criterion_group!` entry from the current
`benches/realization_validation.rs` to that checkout.
This supplies the same five coincident-LP fixtures (2D through 6D) while retaining
the baseline production source. Do not register `bench_realization_high_dimension`
there: its fixture admission runs even when Criterion filters out its timings,
and the baseline cannot admit the 8D cases.

In that prepared baseline checkout, save all three comparison groups explicitly:

```bash
cargo bench --locked --profile perf --bench realization_validation -- \
  '^(realization_narrow_phase|realization_validation|realization_lp)/' \
  --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2 \
  --save-baseline group4-before
cargo bench --locked --profile perf --bench cold_path_predicates -- \
  predicates/sos_degenerate --noplot --warm-up-time 1 --measurement-time 2 \
  --save-baseline group4-before
cargo bench --locked --profile perf --bench validation_components -- \
  'validation_components_[2-5]d_seed_.*/is_valid_delaunay' \
  --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2 \
  --save-baseline group4-before
```

Copy the resulting `target/criterion/` tree into the completed implementation's
`target/criterion/`, preserving its `group4-before` directories. Then run these
commands from the completed implementation's checkout. The high-dimensional
cases are measured only here and saved separately:

```bash
cargo bench --locked --profile perf --bench realization_validation -- \
  '^(realization_narrow_phase|realization_validation|realization_lp)/' \
  --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2 \
  --baseline group4-before
cargo bench --locked --profile perf --bench cold_path_predicates -- \
  predicates/sos_degenerate --noplot --warm-up-time 1 --measurement-time 2 \
  --baseline group4-before
cargo bench --locked --profile perf --bench validation_components -- \
  'validation_components_[2-5]d_seed_.*/is_valid_delaunay' \
  --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2 \
  --baseline group4-before
cargo bench --locked --profile perf --bench realization_validation -- \
  realization_high_dimension --noplot --sample-size 10 --warm-up-time 1 \
  --measurement-time 2 --save-baseline group4-high-dimension
```

Local evidence is retained in `target/group4/final/`: the Criterion archive
contains before/after estimates, samples and benchmark identities; accompanying
logs preserve commands' completion and statistical diagnostics. These build
artifacts are not committed. The source report intentionally retains unsuccessful
tradeoffs as well as improvements.
