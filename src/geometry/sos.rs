//! Simulation of Simplicity (SoS) for deterministic degeneracy resolution.
//!
//! When geometric predicates (orientation, in-sphere) evaluate to exactly zero —
//! meaning points are exactly co-planar, co-spherical, etc. — the SoS technique
//! provides a deterministic non-zero answer without modifying any coordinates.
//!
//! # Algorithm
//!
//! Every point coordinate receives a distinct symbolic infinitesimal ordered
//! first by the point's position in the supplied slice and then by coordinate.
//! Determinants are expanded as exact sparse polynomials. The constant term is
//! the ordinary exact predicate; on a true degeneracy, the least exponent in
//! that canonical hierarchy determines the sign. No floating-point coordinate
//! is modified and no finite-order cofactor truncation is used.
//!
//! # Key Properties
//!
//! - **Deterministic**: same input always produces the same sign
//! - **No coordinate modification**: purely a decision rule
//! - **Always non-zero**: returns ±1, never 0
//! - **Dimension-bounded**: complete exact expansion is implemented through D=6
//! - **Translation-invariant**: orientation minors retain the homogeneous "1"
//!   column, ensuring that shifting all points by a constant vector does not
//!   change the result
//!
//! # References
//!
//! - Edelsbrunner, H. and Mücke, E. P. "Simulation of Simplicity: A Technique
//!   to Cope with Degenerate Cases in Geometric Algorithms." ACM Transactions
//!   on Graphics, 9(1):66–104, 1990.

#![forbid(unsafe_code)]

use crate::geometry::matrix::{BigRational, Signed, rational_from_f64};
use crate::geometry::point::Point;
use crate::geometry::predicates::{
    Orientation, relative_insphere_determinant_sign, simplex_orientation,
};
use crate::geometry::traits::coordinate::{
    CoordinateConversionError, DegenerateSimplexReason, InvalidCoordinateValue,
};
use num_traits::Zero;
use std::collections::BTreeMap;

// =============================================================================
// PUBLIC API
// =============================================================================

/// Compute the `SoS` orientation sign for a degenerate simplex.
///
/// Given `D+1` points whose orientation determinant is exactly zero, this
/// function returns a deterministic non-zero sign (±1) using Simulation of
/// Simplicity.
///
/// The orientation matrix has the form:
///
/// ```text
/// | x₀  y₀  z₀ … 1 |
/// | x₁  y₁  z₁ … 1 |
/// | …               |
/// | xD  yD  zD … 1 |
/// ```
///
/// # Translation Invariance
///
/// Coordinate perturbations are applied before the homogeneous determinant is
/// expanded, so the retained constant column preserves translation invariance.
///
/// # Arguments
///
/// * `points` - Exactly `D+1` points (as f64 coordinate arrays) forming the
///   degenerate simplex.  The index of each point in this slice determines its
///   symbolic perturbation priority.
///
/// # Returns
///
/// `Ok(1)` or `Ok(-1)`.  Never returns `Ok(0)`.
///
/// # Errors
///
/// - [`CoordinateConversionError::InvalidSimplexPointCount`] if
///   `points.len() != D + 1`.
/// - [`CoordinateConversionError::UnsupportedMatrixDimension`] above D=6.
/// - [`CoordinateConversionError::NonFiniteValue`] if any coordinate is
///   NaN or infinite.
pub fn sos_orientation_sign<const D: usize>(
    points: &[Point<D>],
) -> Result<i32, CoordinateConversionError> {
    if points.len() != D + 1 {
        return Err(CoordinateConversionError::InvalidSimplexPointCount {
            actual: points.len(),
            expected: D + 1,
            dimension: D,
        });
    }

    // Reject non-finite coordinates (NaN / ±∞) before entering the
    // symbolic expansion. Non-finite values would silently produce
    // meaningless determinant signs.
    for (point_idx, point) in points.iter().enumerate() {
        for (coord_idx, &val) in point.coords().iter().enumerate() {
            if !val.is_finite() {
                return Err(CoordinateConversionError::NonFiniteValue {
                    coordinate_index: point_idx * D + coord_idx,
                    coordinate_value: InvalidCoordinateValue::from_debug(&val),
                });
            }
        }
    }

    if D > 6 {
        return Err(CoordinateConversionError::UnsupportedMatrixDimension {
            requested: D + 1,
            max: 7,
        });
    }

    // The ordinary exact determinant is the constant term of the symbolic
    // polynomial. Resolve it before constructing the complete expansion so
    // callers that defensively invoke SoS on a nondegenerate input do not pay
    // for the genuinely degenerate cold path.
    match simplex_orientation(points)? {
        Orientation::NEGATIVE => return Ok(-1),
        Orientation::POSITIVE => return Ok(1),
        Orientation::DEGENERATE => {}
    }

    let one = constant_polynomial(1.0)?;
    let matrix: Vec<Vec<_>> = points
        .iter()
        .enumerate()
        .map(|(row, point)| {
            let mut values: Vec<_> = point
                .coords()
                .iter()
                .copied()
                .enumerate()
                .map(|(column, value)| coordinate_polynomial(value, row * D + column))
                .collect::<Result<_, _>>()?;
            values.push(one.clone());
            Ok(values)
        })
        .collect::<Result<_, CoordinateConversionError>>()?;

    polynomial_determinant_leading_sign(&matrix).ok_or(
        CoordinateConversionError::DegenerateSimplex {
            dimension: D,
            reason: DegenerateSimplexReason::VanishingSosPolynomial,
        },
    )
}

/// Compute the `SoS` in-sphere sign for a degenerate configuration.
///
/// Given `D+1` simplex points and a test point whose in-sphere determinant is
/// exactly zero (test point lies exactly on the circumsphere), this function
/// returns a deterministic non-zero sign (±1) using Simulation of Simplicity.
///
/// The lifted in-sphere matrix (relative coordinates centered on `simplex[0]`)
/// has the form:
///
/// ```text
/// | Δx₁  Δy₁  … ‖Δp₁‖² |
/// | Δx₂  Δy₂  … ‖Δp₂‖² |
/// | …                    |
/// | Δxₜ  Δyₜ  … ‖Δpₜ‖² |
/// ```
///
/// where `Δpᵢ = pᵢ - p₀` and `t` is the test point.
///
/// Symbolic coordinates are lifted before the complete determinant polynomial
/// is expanded, so coordinate and squared-norm perturbations remain coherent.
///
/// # Raw Determinant Sign
///
/// This function returns the sign of the **perturbed insphere determinant**,
/// *not* a normalized INSIDE/OUTSIDE classification.  The relationship
/// between determinant sign and geometric containment depends on the simplex
/// orientation.  Callers must multiply the result by an appropriate
/// orientation factor (as [`AdaptiveKernel::in_sphere`](crate::geometry::kernel::AdaptiveKernel) does) to obtain the
/// correct INSIDE/OUTSIDE semantics.
///
/// # Arguments
///
/// * `simplex` - Exactly `D+1` points defining the simplex (f64 coordinates).
/// * `test` - The test point to classify.
///
/// # Returns
///
/// `Ok(1)` or `Ok(-1)` (raw determinant sign).  Never returns `Ok(0)`.
///
/// # Errors
///
/// - [`CoordinateConversionError::InvalidSimplexPointCount`] if
///   `simplex.len() != D + 1`.
/// - [`CoordinateConversionError::UnsupportedMatrixDimension`] above D=6.
/// - [`CoordinateConversionError::NonFiniteValue`] if any coordinate is
///   NaN or infinite.
pub fn sos_insphere_sign<const D: usize>(
    simplex: &[Point<D>],
    test: &Point<D>,
) -> Result<i32, CoordinateConversionError> {
    if simplex.len() != D + 1 {
        return Err(CoordinateConversionError::InvalidSimplexPointCount {
            actual: simplex.len(),
            expected: D + 1,
            dimension: D,
        });
    }

    // Reject non-finite coordinates (NaN / ±∞) in simplex and test point.
    for (point_idx, point) in simplex.iter().enumerate() {
        for (coord_idx, &val) in point.coords().iter().enumerate() {
            if !val.is_finite() {
                return Err(CoordinateConversionError::NonFiniteValue {
                    coordinate_index: point_idx * D + coord_idx,
                    coordinate_value: InvalidCoordinateValue::from_debug(&val),
                });
            }
        }
    }
    for (coord_idx, &val) in test.coords().iter().enumerate() {
        if !val.is_finite() {
            return Err(CoordinateConversionError::NonFiniteValue {
                coordinate_index: (D + 1) * D + coord_idx,
                coordinate_value: InvalidCoordinateValue::from_debug(&val),
            });
        }
    }

    if D > 6 {
        return Err(CoordinateConversionError::UnsupportedMatrixDimension {
            requested: D + 2,
            max: 8,
        });
    }

    // As for orientation, the unperturbed exact determinant is the constant
    // term. The complete symbolic polynomial is necessary only when that term
    // vanishes.
    let ordinary_sign = relative_insphere_determinant_sign(simplex, test)?;
    if ordinary_sign != 0 {
        return Ok(ordinary_sign);
    }

    let points = simplex.iter().chain(core::iter::once(test));
    let one = constant_polynomial(1.0)?;
    let zero = constant_polynomial(0.0)?;
    let matrix: Vec<Vec<_>> = points
        .enumerate()
        .map(|(row, point)| {
            let coordinates: Vec<_> = point
                .coords()
                .iter()
                .copied()
                .enumerate()
                .map(|(column, value)| coordinate_polynomial(value, row * D + column))
                .collect::<Result<_, CoordinateConversionError>>()?;
            let lift = coordinates
                .iter()
                .try_fold(zero.clone(), |sum, coordinate| {
                    Ok::<_, CoordinateConversionError>(add_polynomials(
                        sum,
                        multiply_polynomials(coordinate, coordinate)?,
                        1,
                    ))
                })?;
            let mut values = coordinates;
            values.push(lift);
            values.push(one.clone());
            Ok(values)
        })
        .collect::<Result<_, CoordinateConversionError>>()?;

    let absolute_sign = polynomial_determinant_leading_sign(&matrix).ok_or(
        CoordinateConversionError::DegenerateSimplex {
            dimension: D,
            reason: DegenerateSimplexReason::VanishingSosPolynomial,
        },
    )?;
    // Row subtraction and translation from the absolute lifted determinant to
    // the relative formulation used by the kernels contributes (-1)^(D+1).
    Ok(if D.is_multiple_of(2) {
        -absolute_sign
    } else {
        absolute_sign
    })
}

// =============================================================================
// INTERNAL HELPERS
// =============================================================================

/// Base-3 encoding of a symbolic monomial.
///
/// Variable `i` contributes `3^i`; squaring contributes `2 * 3^i`. A
/// determinant selects at most one entry per point row, so no coordinate
/// variable can have degree above two. Base 3 therefore has no carries and
/// numeric key order is exactly the reverse-lexicographic perturbation order:
/// every combination of earlier variables precedes the next variable. D=6
/// insphere uses 48 variables, well inside `u128`.
type Monomial = u128;
type Polynomial = BTreeMap<Monomial, BigRational>;

fn constant_polynomial(value: f64) -> Result<Polynomial, CoordinateConversionError> {
    let coefficient =
        rational_from_f64(value).ok_or_else(|| CoordinateConversionError::NonFiniteValue {
            coordinate_index: 0,
            coordinate_value: InvalidCoordinateValue::from_debug(&value),
        })?;
    let mut polynomial = Polynomial::new();
    if !coefficient.is_zero() {
        polynomial.insert(0, coefficient);
    }
    Ok(polynomial)
}

fn coordinate_polynomial(
    value: f64,
    variable: usize,
) -> Result<Polynomial, CoordinateConversionError> {
    let mut polynomial = constant_polynomial(value)?;
    let exponent_power = u32::try_from(variable).map_err(|_| {
        CoordinateConversionError::UnsupportedMatrixDimension {
            requested: variable + 1,
            max: 48,
        }
    })?;
    let exponent = 3_u128.checked_pow(exponent_power).ok_or(
        CoordinateConversionError::UnsupportedMatrixDimension {
            requested: variable + 1,
            max: 48,
        },
    )?;
    polynomial.insert(
        exponent,
        rational_from_f64(1.0).expect("one is a finite IEEE-754 value"),
    );
    Ok(polynomial)
}

fn add_polynomials(mut left: Polynomial, right: Polynomial, sign: i32) -> Polynomial {
    for (monomial, coefficient) in right {
        let entry = left.entry(monomial).or_insert_with(BigRational::zero);
        if sign > 0 {
            *entry += coefficient;
        } else {
            *entry -= coefficient;
        }
        if entry.is_zero() {
            left.remove(&monomial);
        }
    }
    left
}

fn multiply_polynomials(
    left: &Polynomial,
    right: &Polynomial,
) -> Result<Polynomial, CoordinateConversionError> {
    let mut product = Polynomial::new();
    for (left_monomial, left_coefficient) in left {
        for (right_monomial, right_coefficient) in right {
            let monomial = left_monomial.checked_add(*right_monomial).ok_or(
                CoordinateConversionError::UnsupportedMatrixDimension {
                    requested: 49,
                    max: 48,
                },
            )?;
            let entry = product.entry(monomial).or_insert_with(BigRational::zero);
            *entry += left_coefficient.clone() * right_coefficient.clone();
            if entry.is_zero() {
                product.remove(&monomial);
            }
        }
    }
    Ok(product)
}

fn polynomial_determinant_leading_sign(matrix: &[Vec<Polynomial>]) -> Option<i32> {
    let dimension = matrix.len();
    if matrix.iter().any(|row| row.len() != dimension) {
        return None;
    }
    if dimension == 0 {
        return Some(1);
    }
    let mut partials = vec![None; 1usize << dimension];
    partials[0] = Some(constant_polynomial(1.0).ok()?);

    for mask in 0usize..(1usize << dimension) {
        let row = mask.count_ones() as usize;
        if row >= dimension {
            continue;
        }
        let Some(partial) = partials[mask].clone() else {
            continue;
        };
        for (column, entry) in matrix[row].iter().enumerate() {
            if mask & (1 << column) != 0 {
                continue;
            }
            let term = multiply_polynomials(&partial, entry).ok()?;
            let sign = if (mask >> (column + 1)).count_ones() % 2 == 0 {
                1
            } else {
                -1
            };
            let next = mask | (1 << column);
            partials[next] = Some(match partials[next].take() {
                Some(existing) => add_polynomials(existing, term, sign),
                None if sign > 0 => term,
                None => add_polynomials(Polynomial::new(), term, -1),
            });
        }
    }

    let determinant = partials[(1 << dimension) - 1].as_ref()?;
    let coefficient = determinant.first_key_value()?.1;
    Some(if coefficient.is_positive() { 1 } else { -1 })
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::Point;

    // =========================================================================
    // GENERIC HELPER FUNCTIONS
    // =========================================================================

    /// Build D+1 co-hyperplanar points (all with last coordinate = 0).
    ///
    /// Construction: origin + (D−1) axis-aligned unit vectors (last coord = 0)
    /// + a barycentric combination with weight 0.5 in each active axis.
    fn degenerate_orient_points<const D: usize>() -> Vec<Point<D>> {
        let mut points = Vec::with_capacity(D + 1);
        points.push(Point::try_new([0.0; D]).expect("finite point coordinates"));
        for i in 0..D.saturating_sub(1) {
            let mut coords = [0.0; D];
            coords[i] = 1.0;
            points.push(Point::try_new(coords).expect("finite point coordinates"));
        }
        let mut bary = [0.0; D];
        for c in bary.iter_mut().take(D.saturating_sub(1)) {
            *c = 0.5;
        }
        points.push(Point::try_new(bary).expect("finite point coordinates"));
        points
    }

    /// Build D+1 simplex points and a co-spherical test point.
    ///
    /// The simplex is the origin plus D axis-aligned unit vectors.
    /// The test point (1,1,…,1) lies on the circumsphere (distance from
    /// center = circumradius for all D ≥ 2).
    fn cospherical_points<const D: usize>() -> (Vec<Point<D>>, Point<D>) {
        let mut simplex = Vec::with_capacity(D + 1);
        simplex.push(Point::try_new([0.0; D]).expect("finite point coordinates"));
        for i in 0..D {
            let mut coords = [0.0; D];
            coords[i] = 1.0;
            simplex.push(Point::try_new(coords).expect("finite point coordinates"));
        }
        (
            simplex,
            Point::try_new([1.0; D]).expect("finite point coordinates"),
        )
    }

    /// Translate a point by a deterministic per-axis offset.
    fn translate_point<const D: usize>(p: &Point<D>) -> Point<D> {
        const OFFSETS: [f64; 5] = [1e6, -5e5, 7.77, -3.33e4, 42.0];
        let mut coords = [0.0; D];
        for (i, c) in coords.iter_mut().enumerate() {
            *c = p.coords()[i] + OFFSETS[i % OFFSETS.len()];
        }
        Point::try_new(coords).expect("finite point coordinates")
    }

    // =========================================================================
    // MACRO-GENERATED PER-DIMENSION TESTS (2D–6D)
    // =========================================================================

    /// Generate the standard `SoS` tests for a given dimension:
    ///
    /// - orientation: degenerate nonzero, deterministic, translation-invariant
    /// - insphere: cospherical nonzero, deterministic (10 calls),
    ///   translation-invariant
    /// - repeated coordinates: deterministic symbolic ordering
    macro_rules! gen_sos_dim_tests {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<test_sos_orientation_ $dim d_degenerate_nonzero>]() {
                    let points = degenerate_orient_points::<$dim>();
                    let sign = sos_orientation_sign(&points).unwrap();
                    assert!(sign == 1 || sign == -1, "SoS must return ±1, got {sign}");
                }

                #[test]
                fn [<test_sos_orientation_ $dim d_degenerate_deterministic>]() {
                    let points = degenerate_orient_points::<$dim>();
                    let s1 = sos_orientation_sign(&points).unwrap();
                    let s2 = sos_orientation_sign(&points).unwrap();
                    assert_eq!(s1, s2, "SoS must be deterministic");
                }

                #[test]
                fn [<test_sos_orientation_ $dim d_translation_invariant>]() {
                    let points = degenerate_orient_points::<$dim>();
                    let s1 = sos_orientation_sign(&points).unwrap();
                    let translated: Vec<_> = points.iter().map(translate_point).collect();
                    let s2 = sos_orientation_sign(&translated).unwrap();
                    assert_eq!(s1, s2, "SoS orientation must be translation-invariant");
                }

                #[test]
                fn [<test_sos_insphere_ $dim d_cospherical_nonzero>]() {
                    let (simplex, test) = cospherical_points::<$dim>();
                    let sign = sos_insphere_sign(&simplex, &test).unwrap();
                    assert!(
                        sign == 1 || sign == -1,
                        "SoS insphere must return ±1, got {sign}"
                    );
                }

                #[test]
                fn [<test_sos_insphere_ $dim d_cospherical_deterministic>]() {
                    let (simplex, test) = cospherical_points::<$dim>();
                    let results: Vec<i32> = (0..10)
                        .map(|_| sos_insphere_sign(&simplex, &test).unwrap())
                        .collect();
                    assert!(
                        results.iter().all(|&r| r == results[0]),
                        "SoS insphere must be deterministic across calls"
                    );
                }

                #[test]
                fn [<test_sos_insphere_ $dim d_translation_invariant>]() {
                    let (simplex, test) = cospherical_points::<$dim>();
                    let s1 = sos_insphere_sign(&simplex, &test).unwrap();
                    let translated_simplex: Vec<_> =
                        simplex.iter().map(translate_point).collect();
                    let translated_test = translate_point(&test);
                    let s2 =
                        sos_insphere_sign(&translated_simplex, &translated_test).unwrap();
                    assert_eq!(
                        s1, s2,
                        "SoS insphere must be translation-invariant"
                    );
                }

                #[test]
                fn [<test_sos_orientation_ $dim d_all_identical_is_symbolically_ordered>]() {
                    let points = vec![Point::try_new([0.0; $dim]).expect("finite point coordinates"); $dim + 1];
                    assert_ne!(sos_orientation_sign(&points).unwrap(), 0);
                }

                #[test]
                fn [<test_sos_insphere_ $dim d_all_identical_is_symbolically_ordered>]() {
                    let simplex = vec![Point::try_new([1.0; $dim]).expect("finite point coordinates"); $dim + 1];
                    let test_pt = Point::try_new([1.0; $dim]).expect("finite point coordinates");
                    assert_ne!(sos_insphere_sign(&simplex, &test_pt).unwrap(), 0);
                }
            }
        };
    }

    gen_sos_dim_tests!(2);
    gen_sos_dim_tests!(3);
    gen_sos_dim_tests!(4);
    gen_sos_dim_tests!(5);

    #[test]
    fn complete_orientation_expansion_supports_6d() {
        let points = degenerate_orient_points::<6>();
        let sign = sos_orientation_sign(&points).unwrap();
        assert!(matches!(sign, -1 | 1));
    }

    #[test]
    fn complete_insphere_expansion_supports_6d() {
        let (simplex, test) = cospherical_points::<6>();
        let sign = sos_insphere_sign(&simplex, &test).unwrap();
        assert!(matches!(sign, -1 | 1));
    }

    #[test]
    fn complete_expansion_resolves_four_collinear_points_in_three_dimensions() {
        let points = [
            Point::try_new([0.0, 0.0, 0.0]).unwrap(),
            Point::try_new([1.0, 0.0, 0.0]).unwrap(),
            Point::try_new([2.0, 0.0, 0.0]).unwrap(),
            Point::try_new([3.0, 0.0, 0.0]).unwrap(),
        ];

        assert_ne!(sos_orientation_sign(&points).unwrap(), 0);
        assert_eq!(
            sos_orientation_sign(&points).unwrap(),
            sos_orientation_sign(&points).unwrap()
        );
    }

    // =========================================================================
    // SOS ORIENTATION — NON-DEGENERATE SPOT CHECK
    // =========================================================================

    #[test]
    fn test_sos_orientation_nondegenerate_returns_correct_sign() {
        // Positive orientation triangle.  For this specific non-degenerate
        // configuration the leading SoS term agrees with the true
        // orientation.  (SoS is only guaranteed correct for degenerate inputs;
        // the caller should never invoke SoS for non-degenerate cases.)
        let positive = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0]).expect("finite point coordinates"),
        ];
        let sign = sos_orientation_sign(&positive).unwrap();
        assert_eq!(sign, 1, "Non-degenerate positive triangle should return +1");
    }

    // =========================================================================
    // ERROR HANDLING TESTS
    // =========================================================================

    #[test]
    fn test_sos_orientation_wrong_point_count_returns_error() {
        let points = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
        ];
        let result = sos_orientation_sign(&points);
        assert_eq!(
            result,
            Err(CoordinateConversionError::InvalidSimplexPointCount {
                actual: 2,
                expected: 3,
                dimension: 2,
            })
        );
    }

    #[test]
    fn test_sos_insphere_wrong_simplex_count_returns_error() {
        let simplex = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
        ];
        let test = Point::try_new([0.5, 0.5]).expect("finite point coordinates");
        let result = sos_insphere_sign(&simplex, &test);
        assert_eq!(
            result,
            Err(CoordinateConversionError::InvalidSimplexPointCount {
                actual: 2,
                expected: 3,
                dimension: 2,
            })
        );
    }

    // =========================================================================
    // SOS ORIENTATION — 1D EDGE CASE
    // =========================================================================

    #[test]
    fn test_sos_orientation_1d_identical_points() {
        // Two identical 1D points: orientation determinant is exactly zero.
        // SoS must still resolve to ±1.
        let points = vec![
            Point::try_new([5.0]).expect("finite point coordinates"),
            Point::try_new([5.0]).expect("finite point coordinates"),
        ];
        let sign = sos_orientation_sign(&points).unwrap();
        assert!(
            sign == 1 || sign == -1,
            "SoS must return ±1 for 1D, got {sign}"
        );
    }

    #[test]
    fn test_sos_orientation_1d_distinct_degenerate() {
        // D=1 with distinct points is non-degenerate, but SoS still works.
        let points = vec![
            Point::try_new([0.0]).expect("finite point coordinates"),
            Point::try_new([1.0]).expect("finite point coordinates"),
        ];
        let sign = sos_orientation_sign(&points).unwrap();
        assert!(
            sign == 1 || sign == -1,
            "SoS must return ±1 for 1D, got {sign}"
        );
    }
}
