//! Property-based tests for Simulation of Simplicity (`SoS`) invariants.
//!
//! The Edelsbrunner & Mücke `SoS` scheme (1990) does not prescribe specific
//! test vectors.  Correctness is verified by testing the mathematical
//! invariants that a valid `SoS` implementation must satisfy:
//!
//! - **Non-degeneracy**: returns ±1 for degenerate inputs in supported dimensions
//! - **Determinism**: same input always produces the same sign
//! - **Translation invariance**: orientation sign is unchanged by translation
//! - **Robustness**: never panics on arbitrary finite inputs
//!
//! The [`gen_sos_tests`] macro generates the full invariant suite for each
//! dimension (2D–6D), using two families of exactly-degenerate inputs:
//!
//! - **Co-hyperplanar points** (orientation): D+1 points with last coordinate
//!   fixed to zero and integer values in the remaining D−1 coordinates,
//!   guaranteeing an exactly-zero orientation determinant. The complete
//!   symbolic determinant must resolve every such input, including lower-rank
//!   configurations that no first-order truncation could resolve.
//! - **Hyper-rectangle vertices** (insphere): the origin corner, D adjacent
//!   axis-aligned corners, and the diagonally opposite corner of an
//!   integer-coordinate hyper-rectangle all lie on a common circumsphere,
//!   giving an exactly-zero insphere determinant.

#![forbid(unsafe_code)]

#[macro_use]
#[path = "common/proptest_config.rs"]
mod proptest_config;

use delaunay::geometry::sos::{sos_insphere_sign, sos_orientation_sign};
use delaunay::prelude::geometry::Point;
use proptest::prelude::*;

// =============================================================================
// HELPERS
// =============================================================================

/// Check that all points have distinct coordinates.
///
/// Uses exact equality because coordinates are derived from integers
/// via `f64::from()`, so no rounding occurs.
#[expect(
    clippy::float_cmp,
    reason = "coordinates are integer-derived f64 values, so exact equality is intentional"
)]
fn points_all_distinct<const D: usize>(points: &[Point<D>]) -> bool {
    (0..points.len())
        .all(|i| ((i + 1)..points.len()).all(|j| points[i].coords() != points[j].coords()))
}

// =============================================================================
// STRATEGIES
// =============================================================================

/// Finite f64 coordinate in a moderate range.
fn finite_coord() -> impl Strategy<Value = f64> {
    (-100.0..100.0).prop_filter("must be finite", |x: &f64| x.is_finite())
}

/// Small integer for degenerate-point construction.
fn small_int() -> impl Strategy<Value = i32> {
    -50i32..50
}

/// Positive integer for hyper-rectangle side lengths.
fn positive_int() -> impl Strategy<Value = i32> {
    1i32..30
}

// =============================================================================
// MACRO-GENERATED TEST SUITES (2D – 6D)
// =============================================================================

/// Generate the full `SoS` property-test suite for a single dimension.
///
/// For each dimension this produces 7 tests:
///   - 3 degenerate-orientation tests (non-zero, deterministic, translation-invariant)
///   - 2 co-spherical insphere tests (non-zero, deterministic)
///   - 2 random-robustness tests (orientation, insphere)
macro_rules! gen_sos_tests {
    ($dim:literal, $uniform:path, $config:expr) => {
        pastey::paste! {
            // =============================================================
            // Degenerate orientation — co-hyperplanar points ($dim D)
            // =============================================================

            repo_proptest! {
                #![proptest_config($config)]
                /// `SoS` orientation returns ±1 for degenerate points.
                #[test]
                fn [<prop_sos_orientation_nonzero_ $dim d>](
                    raw in prop::collection::vec(
                        $uniform(small_int()),
                        ($dim + 1)..=($dim + 1),
                    ),
                ) {
                    let points: Vec<Point<$dim>> = raw
                        .iter()
                        .map(|arr| {
                            // Last coordinate forced to 0 → co-hyperplanar.
                            let coords: [f64; $dim] = std::array::from_fn(|i| {
                                if i < $dim - 1 { f64::from(arr[i]) } else { 0.0 }
                            });
                            Point::try_new(coords).expect("finite point coordinates")
                        })
                        .collect();
                    // In 2D only one coordinate varies (the last is forced
                    // to 0), so collisions are likely.  SoS requires
                    // distinct points — skip inputs with duplicates.
                    prop_assume!(points_all_distinct(&points));

                    let sign = sos_orientation_sign(&points)
                        .expect("finite supported-dimensional SoS input must be resolved");
                    prop_assert!(sign == 1 || sign == -1,
                        "SoS orientation must return ±1 in {}D, got {}", $dim, sign);
                }

                /// `SoS` orientation is deterministic for degenerate points.
                #[test]
                fn [<prop_sos_orientation_deterministic_ $dim d>](
                    raw in prop::collection::vec(
                        $uniform(small_int()),
                        ($dim + 1)..=($dim + 1),
                    ),
                ) {
                    let points: Vec<Point<$dim>> = raw
                        .iter()
                        .map(|arr| {
                            let coords: [f64; $dim] = std::array::from_fn(|i| {
                                if i < $dim - 1 { f64::from(arr[i]) } else { 0.0 }
                            });
                            Point::try_new(coords).expect("finite point coordinates")
                        })
                        .collect();
                    prop_assume!(points_all_distinct(&points));

                    let s1 = sos_orientation_sign(&points)
                        .expect("finite supported-dimensional SoS input must be resolved");
                    let s2 = sos_orientation_sign(&points)
                        .expect("repeated SoS input must remain resolvable");
                    prop_assert_eq!(s1, s2, "SoS must be deterministic in {}D", $dim);
                }

                /// `SoS` orientation is translation-invariant for degenerate points.
                ///
                /// The offset uses integers (not arbitrary f64) because SoS
                /// translation invariance relies on the "1" column cancelling
                /// the shift *exactly*. Non-integer f64 offsets can change the
                /// represented input before the exact symbolic expansion.
                #[test]
                fn [<prop_sos_orientation_translation_invariant_ $dim d>](
                    raw in prop::collection::vec(
                        $uniform(small_int()),
                        ($dim + 1)..=($dim + 1),
                    ),
                    offset in $uniform(small_int()),
                ) {
                    let points: Vec<Point<$dim>> = raw
                        .iter()
                        .map(|arr| {
                            let coords: [f64; $dim] = std::array::from_fn(|i| {
                                if i < $dim - 1 { f64::from(arr[i]) } else { 0.0 }
                            });
                            Point::try_new(coords).expect("finite point coordinates")
                        })
                        .collect();
                    prop_assume!(points_all_distinct(&points));

                    let s1 = sos_orientation_sign(&points)
                        .expect("finite supported-dimensional SoS input must be resolved");
                    let translated: Vec<Point<$dim>> = points
                        .iter()
                        .map(|p| {
                            let coords: [f64; $dim] = std::array::from_fn(|i| {
                                p.coords()[i] + f64::from(offset[i])
                            });
                            Point::try_new(coords).expect("finite point coordinates")
                        })
                        .collect();
                    let s2 = sos_orientation_sign(&translated)
                        .expect("integer translation must preserve SoS resolvability");
                    prop_assert_eq!(s1, s2,
                        "SoS orientation must be translation-invariant in {}D", $dim);
                }
            }

            // =============================================================
            // Co-spherical insphere — hyper-rectangle vertices ($dim D)
            // =============================================================

            repo_proptest! {
                #![proptest_config($config)]
                /// `SoS` insphere returns ±1 for exactly co-spherical points.
                #[test]
                fn [<prop_sos_insphere_nonzero_ $dim d>](
                    base in $uniform(small_int()),
                    sides in $uniform(positive_int()),
                ) {
                    let base_f64: [f64; $dim] = std::array::from_fn(|i| f64::from(base[i]));
                    let sides_f64: [f64; $dim] = std::array::from_fn(|i| f64::from(sides[i]));

                    // Simplex: origin corner + D axis-aligned neighbours.
                    let mut simplex: Vec<Point<$dim>> = Vec::with_capacity($dim + 1);
                    simplex.push(Point::try_new(base_f64).expect("finite point coordinates"));
                    for axis in 0..$dim {
                        let mut p = base_f64;
                        p[axis] += sides_f64[axis];
                        simplex.push(Point::try_new(p).expect("finite point coordinates"));
                    }

                    // Test point: diagonally opposite corner (on the circumsphere).
                    let test = Point::try_new(
                        std::array::from_fn(|i| base_f64[i] + sides_f64[i]),
                    ).expect("finite point coordinates");
                    let sign = sos_insphere_sign(&simplex, &test).unwrap();
                    prop_assert!(sign == 1 || sign == -1,
                        "SoS insphere must return ±1 in {}D, got {}", $dim, sign);
                }

                /// `SoS` insphere is deterministic for co-spherical points.
                #[test]
                fn [<prop_sos_insphere_deterministic_ $dim d>](
                    base in $uniform(small_int()),
                    sides in $uniform(positive_int()),
                ) {
                    let base_f64: [f64; $dim] = std::array::from_fn(|i| f64::from(base[i]));
                    let sides_f64: [f64; $dim] = std::array::from_fn(|i| f64::from(sides[i]));

                    let mut simplex: Vec<Point<$dim>> = Vec::with_capacity($dim + 1);
                    simplex.push(Point::try_new(base_f64).expect("finite point coordinates"));
                    for axis in 0..$dim {
                        let mut p = base_f64;
                        p[axis] += sides_f64[axis];
                        simplex.push(Point::try_new(p).expect("finite point coordinates"));
                    }

                    let test = Point::try_new(
                        std::array::from_fn(|i| base_f64[i] + sides_f64[i]),
                    ).expect("finite point coordinates");
                    let s1 = sos_insphere_sign(&simplex, &test).unwrap();
                    let s2 = sos_insphere_sign(&simplex, &test).unwrap();
                    prop_assert_eq!(s1, s2,
                        "SoS insphere must be deterministic in {}D", $dim);
                }
            }

            // =============================================================
            // Robustness — arbitrary finite inputs ($dim D)
            // =============================================================

            repo_proptest! {
                #![proptest_config($config)]
                /// `SoS` orientation never panics on random inputs.
                #[test]
                fn [<prop_sos_orientation_robust_ $dim d>](
                    points in prop::collection::vec(
                        $uniform(finite_coord()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        ($dim + 1)..=($dim + 1),
                    ),
                ) {
                    prop_assert!(matches!(sos_orientation_sign(&points), Ok(1 | -1)));
                }

                /// `SoS` insphere never panics on random inputs.
                #[test]
                fn [<prop_sos_insphere_robust_ $dim d>](
                    simplex in prop::collection::vec(
                        $uniform(finite_coord()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        ($dim + 1)..=($dim + 1),
                    ),
                    test in $uniform(finite_coord()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                ) {
                    prop_assert!(matches!(sos_insphere_sign(&simplex, &test), Ok(1 | -1)));
                }
            }
        }
    };
}

gen_sos_tests!(
    2,
    prop::array::uniform2,
    proptest_config::repository_default()
);
gen_sos_tests!(
    3,
    prop::array::uniform3,
    proptest_config::repository_default()
);
gen_sos_tests!(
    4,
    prop::array::uniform4,
    proptest_config::repository_default()
);
// Complete 5D symbolic expansion is deliberately a cold path. Eight generated
// cases per property complement deterministic unit regressions while keeping
// routine validation bounded; PROPTEST_CASES still overrides this locally.
gen_sos_tests!(
    5,
    prop::array::uniform5,
    proptest_config::with_default_cases(8)
);
// Exact-degenerate 6D inputs exercise the largest supported symbolic expansion
// and therefore live in bounded unit regressions. Generated 6D inputs normally
// take the exact-determinant fast path; these properties cover that public
// dimensional boundary without multiplying the symbolic cold-path workload.
repo_proptest! {
    #![proptest_config(proptest_config::with_default_cases(8))]
    #[test]
    fn prop_sos_orientation_robust_6d(
        points in prop::collection::vec(
            prop::array::uniform6(finite_coord())
                .prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
            7..=7,
        ),
    ) {
        prop_assert!(matches!(sos_orientation_sign(&points), Ok(1 | -1)));
    }

    #[test]
    fn prop_sos_insphere_robust_6d(
        simplex in prop::collection::vec(
            prop::array::uniform6(finite_coord())
                .prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
            7..=7,
        ),
        test in prop::array::uniform6(finite_coord())
            .prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
    ) {
        prop_assert!(matches!(sos_insphere_sign(&simplex, &test), Ok(1 | -1)));
    }
}

// =============================================================================
// ERROR HANDLING (dimension-independent)
// =============================================================================

repo_proptest! {
    /// `SoS` orientation returns `Err` for wrong point count.
    #[test]
    fn prop_sos_orientation_rejects_wrong_count(
        points in prop::collection::vec(
            prop::array::uniform2(finite_coord()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
            0..2_usize,
        ),
    ) {
        prop_assert!(sos_orientation_sign(&points).is_err());
    }
}
