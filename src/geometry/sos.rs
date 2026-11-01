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
use la_stack::{PolynomialDeterminantBudget, SparsePolynomial, polynomial_determinant};
use num_traits::Zero;

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
/// * `points` - Exactly `D+1` [`Point`] values forming the
///   degenerate simplex.  The index of each point in this slice determines its
///   symbolic perturbation priority.
///
/// # Returns
///
/// `Ok(1)` or `Ok(-1)`.  Never returns `Ok(0)`.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::{
///     CoordinateConversionError, CoordinateValidationError, Point, sos_orientation_sign,
/// };
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Conversion(#[from] CoordinateConversionError),
/// #     #[error(transparent)]
/// #     Validation(#[from] CoordinateValidationError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let origin = Point::try_new([0.0, 0.0])?;
/// // The symbolic row order resolves coincident points without moving them.
/// assert_eq!(sos_orientation_sign(&[origin; 3])?, -1);
/// # Ok(())
/// # }
/// ```
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
    if D > 6 {
        return Err(CoordinateConversionError::UnsupportedMatrixDimension {
            requested: D.saturating_add(1),
            max: 7,
        });
    }
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

    polynomial_determinant_leading_sign(matrix, SOS_BUDGET)?.ok_or(
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
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::{
///     CoordinateConversionError, CoordinateValidationError, Point, sos_insphere_sign,
/// };
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Conversion(#[from] CoordinateConversionError),
/// #     #[error(transparent)]
/// #     Validation(#[from] CoordinateValidationError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let simplex = [
///     Point::try_new([0.0, 0.0])?,
///     Point::try_new([1.0, 0.0])?,
///     Point::try_new([0.0, 1.0])?,
/// ];
/// let boundary = Point::try_new([1.0, 1.0])?;
/// // This is a raw determinant sign, not an INSIDE/OUTSIDE classification.
/// assert_eq!(sos_insphere_sign(&simplex, &boundary)?, -1);
/// # Ok(())
/// # }
/// ```
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
    if D > 6 {
        return Err(CoordinateConversionError::UnsupportedMatrixDimension {
            requested: D.saturating_add(2),
            max: 8,
        });
    }
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

    // As for orientation, the unperturbed exact determinant is the constant
    // term. The complete symbolic polynomial is necessary only when that term
    // vanishes.
    let ordinary_sign = relative_insphere_determinant_sign(simplex, test)?;
    if ordinary_sign != 0 {
        return Ok(ordinary_sign);
    }

    let points = simplex.iter().chain(core::iter::once(test));
    let one = constant_polynomial(1.0)?;
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
            let lift = squared_norm_polynomial(point, row)?;
            let mut values = coordinates;
            values.push(lift);
            values.push(one.clone());
            Ok(values)
        })
        .collect::<Result<_, CoordinateConversionError>>()?;

    let absolute_sign = polynomial_determinant_leading_sign(matrix, SOS_BUDGET)?.ok_or(
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

/// Largest determinant matrix produced by the public D ≤ 6 `SoS` predicates.
const MAX_SOS_MATRIX_DIMENSION: usize = 8;
/// Complete order-eight envelope; the lifted column has at most 13 terms.
/// Column term maxima [2, 2, 2, 2, 2, 2, 13, 1] bound the complete subset
/// expansion by 55,631,284 coefficient products without any truncation.
/// These bound combinatorial work, not coefficient bit sizes or allocated bytes.
const SOS_BUDGET: PolynomialDeterminantBudget = PolynomialDeterminantBudget {
    max_subset_transitions: 1_024,
    max_term_products: 55_631_284,
};

/// Retains exact IEEE-754 coefficients when constructing symbolic entries.
fn constant_polynomial(value: f64) -> Result<SparsePolynomial, CoordinateConversionError> {
    SparsePolynomial::try_from_terms([(0, exact_coordinate(value)?)]).map_err(Into::into)
}

/// Converts a stored coordinate without any rounded intermediate.
fn exact_coordinate(value: f64) -> Result<BigRational, CoordinateConversionError> {
    rational_from_f64(value).ok_or_else(|| CoordinateConversionError::NonFiniteValue {
        coordinate_index: 0,
        coordinate_value: InvalidCoordinateValue::from_debug(&value),
    })
}

/// Base-3 monomials preserve the reverse-lexicographic perturbation order.
/// Each coordinate occurs in at most one selected row and has degree at most
/// two, so exponent addition cannot carry between variables. D=6 uses 48.
fn coordinate_exponent(variable: usize) -> Result<u128, CoordinateConversionError> {
    u32::try_from(variable)
        .ok()
        .and_then(|power| 3_u128.checked_pow(power))
        .ok_or_else(|| CoordinateConversionError::UnsupportedMatrixDimension {
            requested: variable.saturating_add(1),
            max: 48,
        })
}

/// Assigns the caller's row/coordinate perturbation priority to one entry.
fn coordinate_polynomial(
    value: f64,
    variable: usize,
) -> Result<SparsePolynomial, CoordinateConversionError> {
    SparsePolynomial::try_from_terms([
        (0, exact_coordinate(value)?),
        (
            coordinate_exponent(variable)?,
            BigRational::from_integer(1.into()),
        ),
    ])
    .map_err(Into::into)
}

/// Lifts perturbed coordinates coherently: Σ(a + ε)² = Σa² + Σ2aε + Σε².
/// Only this geometric assembly stays local; polynomial reduction and the
/// complete determinant are owned by la-stack.
fn squared_norm_polynomial<const D: usize>(
    point: &Point<D>,
    row: usize,
) -> Result<SparsePolynomial, CoordinateConversionError> {
    let mut terms = Vec::with_capacity(2 * D + 1);
    let mut constant = BigRational::zero();
    for (column, &value) in point.coords().iter().enumerate() {
        let coefficient = exact_coordinate(value)?;
        let exponent = coordinate_exponent(row * D + column)?;
        let squared_exponent = exponent.checked_mul(2).ok_or(
            CoordinateConversionError::UnsupportedMatrixDimension {
                requested: row * D + column + 1,
                max: 48,
            },
        )?;
        constant += &coefficient * &coefficient;
        terms.push((exponent, coefficient * BigRational::from_integer(2.into())));
        terms.push((squared_exponent, BigRational::from_integer(1.into())));
    }
    terms.push((0, constant));
    SparsePolynomial::try_from_terms(terms).map_err(Into::into)
}

/// Admits runtime shapes without cloning coefficient storage. Zero polynomials
/// remain distinct from budget exhaustion, exponent overflow and backend errors.
fn polynomial_determinant_leading_sign(
    matrix: Vec<Vec<SparsePolynomial>>,
    budget: PolynomialDeterminantBudget,
) -> Result<Option<i32>, CoordinateConversionError> {
    match matrix.len() {
        0 => polynomial_sign::<0>(matrix, budget),
        1 => polynomial_sign::<1>(matrix, budget),
        2 => polynomial_sign::<2>(matrix, budget),
        3 => polynomial_sign::<3>(matrix, budget),
        4 => polynomial_sign::<4>(matrix, budget),
        5 => polynomial_sign::<5>(matrix, budget),
        6 => polynomial_sign::<6>(matrix, budget),
        7 => polynomial_sign::<7>(matrix, budget),
        8 => polynomial_sign::<8>(matrix, budget),
        requested => Err(CoordinateConversionError::UnsupportedMatrixDimension {
            requested,
            max: MAX_SOS_MATRIX_DIMENSION,
        }),
    }
}

/// Transfers admitted rows into the upstream const-generic determinant.
fn polynomial_sign<const N: usize>(
    matrix: Vec<Vec<SparsePolynomial>>,
    budget: PolynomialDeterminantBudget,
) -> Result<Option<i32>, CoordinateConversionError> {
    let rows: Vec<[SparsePolynomial; N]> = matrix
        .into_iter()
        .map(|row| {
            let length = row.len();
            row.try_into()
                .map_err(|_| CoordinateConversionError::MatrixDimensionMismatch {
                    active: length,
                    matrix_dimension: N,
                })
        })
        .collect::<Result<_, _>>()?;
    let length = rows.len();
    let rows: [[SparsePolynomial; N]; N] =
        rows.try_into()
            .map_err(|_| CoordinateConversionError::MatrixDimensionMismatch {
                active: length,
                matrix_dimension: N,
            })?;
    let determinant = polynomial_determinant(&rows, budget)?;
    Ok(determinant.leading_term().map(
        |(_, coefficient)| {
            if coefficient.is_positive() { 1 } else { -1 }
        },
    ))
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::Point;
    use la_stack::{LaError, PolynomialWork};
    use std::{assert_matches, collections::BTreeMap};

    // =========================================================================
    // GENERIC HELPER FUNCTIONS
    // =========================================================================

    /// Independent six-permutation expansion, without the subset-DP kernel.
    fn permutation_sign_3(matrix: &[Vec<SparsePolynomial>]) -> Option<i32> {
        let mut terms = BTreeMap::<u128, BigRational>::new();
        for (columns, sign) in [
            ([0, 1, 2], 1),
            ([1, 2, 0], 1),
            ([2, 0, 1], 1),
            ([2, 1, 0], -1),
            ([1, 0, 2], -1),
            ([0, 2, 1], -1),
        ] {
            for (e0, c0) in matrix[0][columns[0]].terms() {
                for (e1, c1) in matrix[1][columns[1]].terms() {
                    for (e2, c2) in matrix[2][columns[2]].terms() {
                        *terms.entry(e0 + e1 + e2).or_default() +=
                            c0 * c1 * c2 * BigRational::from_integer(sign.into());
                    }
                }
            }
        }
        terms
            .values()
            .find(|coefficient| !coefficient.is_zero())
            .map(|coefficient| if coefficient.is_positive() { 1 } else { -1 })
    }

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
    /// - insphere: cospherical nonzero, deterministic,
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
                    // Perturbing the origin's first coordinate contributes the
                    // first monomial. Its absolute cofactor is (-1)^D (D-1);
                    // converting to the relative convention gives -(D-1).
                    assert_eq!(sign, -1);
                }

                #[test]
                fn [<test_sos_insphere_ $dim d_cospherical_deterministic>]() {
                    let (simplex, test) = cospherical_points::<$dim>();
                    let first = sos_insphere_sign(&simplex, &test).unwrap();
                    let second = sos_insphere_sign(&simplex, &test).unwrap();
                    assert_eq!(first, second, "SoS insphere must be deterministic");
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
                    // The least monomial uses the last row's homogeneous 1
                    // and reversed coordinate columns in the first D rows.
                    // Its degree is D and its coefficient is the reversal sign.
                    let expected = if ($dim * ($dim - 1) / 2) % 2 == 0 { 1 } else { -1 };
                    assert_eq!(sos_orientation_sign(&points).unwrap(), expected);
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
        let identical = vec![Point::try_new([0.0; 6]).unwrap(); 7];
        // Reversing six columns takes 15 transpositions.
        assert_eq!(sos_orientation_sign(&identical).unwrap(), -1);
    }

    #[test]
    fn polynomial_adapter_preserves_cancellation_and_permutation_sign() {
        let mut matrix: Vec<Vec<_>> = (0..3)
            .map(|row| {
                (0..3)
                    .map(|col| coordinate_polynomial(1.0, row * 3 + col).unwrap())
                    .collect()
            })
            .collect();
        let expected = permutation_sign_3(&matrix);
        assert!(expected.is_some());
        assert_eq!(
            polynomial_determinant_leading_sign(matrix.clone(), SOS_BUDGET).unwrap(),
            expected
        );
        matrix.swap(0, 1);
        assert_eq!(
            polynomial_determinant_leading_sign(matrix.clone(), SOS_BUDGET).unwrap(),
            expected.map(|sign| -sign)
        );
        assert_eq!(permutation_sign_3(&matrix), expected.map(|sign| -sign));
        matrix[0] = matrix[1].clone();
        assert_eq!(
            polynomial_determinant_leading_sign(matrix, SOS_BUDGET).unwrap(),
            None
        );
        // (1/3 + ε)(1/3 - ε) - (1/3)² = -ε² exactly.
        let third = BigRational::new(1.into(), 3.into());
        let plus = SparsePolynomial::try_from_terms([
            (0, third.clone()),
            (1, BigRational::from_integer(1.into())),
        ])
        .unwrap();
        let minus = SparsePolynomial::try_from_terms([
            (0, third.clone()),
            (1, BigRational::from_integer((-1).into())),
        ])
        .unwrap();
        let constant = SparsePolynomial::try_from_terms([(0, third)]).unwrap();
        assert_eq!(
            polynomial_determinant_leading_sign(
                vec![vec![plus, constant.clone()], vec![constant, minus]],
                SOS_BUDGET
            )
            .unwrap(),
            Some(-1)
        );
    }

    #[test]
    fn polynomial_adapter_keeps_zero_budget_overflow_and_shape_distinct() {
        let one = constant_polynomial(1.0).unwrap();
        assert_eq!(
            polynomial_determinant_leading_sign(vec![], SOS_BUDGET).unwrap(),
            Some(1)
        );
        for (budget, work) in [
            (
                PolynomialDeterminantBudget {
                    max_subset_transitions: 0,
                    max_term_products: 1,
                },
                PolynomialWork::SubsetTransitions,
            ),
            (
                PolynomialDeterminantBudget {
                    max_subset_transitions: 1,
                    max_term_products: 0,
                },
                PolynomialWork::TermProducts,
            ),
        ] {
            assert_matches!(polynomial_determinant_leading_sign(vec![vec![one.clone()]], budget),
                Err(CoordinateConversionError::LinearAlgebraFailure { source })
                    if matches!(*source, LaError::PolynomialBudgetExceeded { work: actual, .. } if actual == work));
        }
        let huge =
            SparsePolynomial::try_from_terms([(u128::MAX, BigRational::from_integer(1.into()))])
                .unwrap();
        let epsilon =
            SparsePolynomial::try_from_terms([(1, BigRational::from_integer(1.into()))]).unwrap();
        assert_matches!(polynomial_determinant_leading_sign(vec![vec![huge, SparsePolynomial::zero()], vec![SparsePolynomial::zero(), epsilon]], SOS_BUDGET),
            Err(CoordinateConversionError::LinearAlgebraFailure { source })
                if matches!(*source, LaError::PolynomialExponentOverflow { .. }));
        assert_matches!(
            polynomial_determinant_leading_sign(vec![vec![]], SOS_BUDGET),
            Err(CoordinateConversionError::MatrixDimensionMismatch { .. })
        );
        assert_matches!(
            polynomial_determinant_leading_sign(vec![vec![one; 9]; 9], SOS_BUDGET),
            Err(CoordinateConversionError::UnsupportedMatrixDimension {
                requested: 9,
                max: 8
            })
        );
    }

    #[test]
    fn complete_insphere_expansion_supports_6d() {
        let (simplex, test) = cospherical_points::<6>();
        let sign = sos_insphere_sign(&simplex, &test).unwrap();
        // The first relative cofactor is -(6 - 1).
        assert_eq!(sign, -1);
    }

    #[test]
    fn orientation_rejects_unsupported_dimension_before_checking_point_count() {
        assert_eq!(
            sos_orientation_sign::<7>(&[]),
            Err(CoordinateConversionError::UnsupportedMatrixDimension {
                requested: 8,
                max: 7,
            })
        );
    }

    #[test]
    fn insphere_rejects_unsupported_dimension_before_checking_point_count() {
        let query = Point::try_new([0.0; 7]).unwrap();
        assert_eq!(
            sos_insphere_sign::<7>(&[], &query),
            Err(CoordinateConversionError::UnsupportedMatrixDimension {
                requested: 9,
                max: 8,
            })
        );
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
