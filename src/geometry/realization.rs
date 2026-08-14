//! Pure predicates for labeled simplex realizations.
//!
//! This module has no TDS, topology, or triangulation storage dependencies. It
//! answers geometric questions about labeled maximal simplices after another
//! layer has chosen the appropriate affine chart. Algorithmic provenance for
//! the Level 4 overlap broad phase is summarized in `REFERENCES.md`,
//! "Realized-Simplex Overlap Detection (Level 4 Validation)".
//!
//! Use [`Triangulation::realization_report`](crate::Triangulation::realization_report)
//! when validating a stored triangulation. Use this module directly when a
//! caller already has chart-local simplex coordinates and wants the pure
//! geometric Level 4 predicate without TDS storage.
//!
//! # Examples
//!
//! ```rust
//! use delaunay::prelude::geometry::{
//!     LabeledSimplexRealization, LabeledSimplexRealizationError, SimplexIntersectionFailure,
//!     validate_simplex_realizations_intersect_only_in_shared_faces,
//! };
//!
//! # fn main() -> Result<(), LabeledSimplexRealizationError> {
//! let first = LabeledSimplexRealization::try_new(
//!     [0_usize, 1, 2],
//!     [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
//! )?;
//! let second = LabeledSimplexRealization::try_new(
//!     [0_usize, 1, 3],
//!     [[0.0, 0.0], [1.0, 0.0], [0.25, 0.25]],
//! )?;
//!
//! std::assert_matches!(
//!     validate_simplex_realizations_intersect_only_in_shared_faces(&first, &second),
//!     Err(SimplexIntersectionFailure::IntersectionOutsideSharedFace { .. })
//! );
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]

use crate::core::collections::{MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer};
use crate::geometry::point::{Point, ValidatedCoordinates};
use crate::geometry::predicates::Orientation;
use crate::geometry::robust_predicates::robust_orientation;
use crate::geometry::traits::coordinate::InvalidCoordinateValue;
use crate::geometry::util::simplex_lp::{
    IntersectionLinearProgramResult, coordinates_are_identical,
    intersection_via_legacy_active_sets, intersection_via_linear_program,
    shared_face_fast_confinement,
};
use thiserror::Error;

/// Stack-backed buffer for per-simplex realization labels and coordinates.
pub type SimplexRealizationBuffer<T> = SmallBuffer<T, MAX_PRACTICAL_DIMENSION_SIZE>;

/// Validated coordinates for one labeled D-simplex in an affine chart.
///
/// Labels implement [`Eq`] and are unique within the simplex so intersection
/// witnesses can distinguish shared faces from accidental duplicate vertices.
/// Coordinates are finite `f64` values ready to be converted into exact
/// predicate inputs by the triangulation-level realization validator.
#[derive(Clone, Debug, PartialEq)]
pub struct LabeledSimplexRealization<L, const D: usize> {
    labels: SimplexRealizationBuffer<L>,
    coordinates: SimplexRealizationBuffer<[f64; D]>,
}

impl<L, const D: usize> LabeledSimplexRealization<L, D> {
    /// Builds a labeled D-simplex realization after checking arity, uniqueness, and finite coordinates.
    ///
    /// This is the parse boundary for pure simplex-realization predicates: callers
    /// supply labels and coordinates in matching order, and the constructor
    /// stores only realizations with exactly `D + 1` distinct labels. Labels use
    /// [`Eq`] because they represent vertex identity, not an approximate value.
    ///
    /// # Errors
    ///
    /// Returns [`LabeledSimplexRealizationError::LabelCoordinateLengthMismatch`]
    /// when the two iterators produce different lengths,
    /// [`LabeledSimplexRealizationError::InvalidArity`] when the simplex does not
    /// contain exactly `D + 1` vertices,
    /// [`LabeledSimplexRealizationError::DuplicateLabel`] when a label appears
    /// more than once, or
    /// [`LabeledSimplexRealizationError::NonFiniteCoordinate`] when any coordinate
    /// is NaN or infinite.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::geometry::{
    ///     LabeledSimplexRealization, LabeledSimplexRealizationError,
    /// };
    ///
    /// # fn main() -> Result<(), LabeledSimplexRealizationError> {
    /// let simplex = LabeledSimplexRealization::try_new(
    ///     ["a", "b", "c"],
    ///     [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    /// )?;
    ///
    /// assert_eq!(simplex.labels(), ["a", "b", "c"]);
    /// assert_eq!(simplex.coordinates().len(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_new(
        labels: impl IntoIterator<Item = L>,
        coordinates: impl IntoIterator<Item = [f64; D]>,
    ) -> Result<Self, LabeledSimplexRealizationError>
    where
        L: Eq,
    {
        let labels: SimplexRealizationBuffer<L> = labels.into_iter().collect();
        let coordinates: SimplexRealizationBuffer<[f64; D]> = coordinates.into_iter().collect();

        if labels.len() != coordinates.len() {
            return Err(
                LabeledSimplexRealizationError::LabelCoordinateLengthMismatch {
                    label_count: labels.len(),
                    coordinate_count: coordinates.len(),
                },
            );
        }

        let expected = D + 1;
        if labels.len() != expected {
            return Err(LabeledSimplexRealizationError::InvalidArity {
                expected,
                actual: labels.len(),
            });
        }

        for (first_index, first_label) in labels.iter().enumerate() {
            if let Some(duplicate_offset) = labels[first_index + 1..]
                .iter()
                .position(|label| label == first_label)
            {
                return Err(LabeledSimplexRealizationError::DuplicateLabel {
                    first_index,
                    duplicate_index: first_index + duplicate_offset + 1,
                });
            }
        }

        validate_coordinate_rows(&coordinates)?;

        Ok(Self {
            labels,
            coordinates,
        })
    }

    /// Returns labels in the same order as the simplex coordinates.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::geometry::{
    ///     LabeledSimplexRealization, LabeledSimplexRealizationError,
    /// };
    ///
    /// # fn main() -> Result<(), LabeledSimplexRealizationError> {
    /// let simplex = LabeledSimplexRealization::try_new(
    ///     ["left", "right", "apex"],
    ///     [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
    /// )?;
    ///
    /// assert_eq!(simplex.labels(), ["left", "right", "apex"]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn labels(&self) -> &[L] {
        &self.labels
    }

    /// Returns the D-dimensional coordinates paired with [`labels`](Self::labels).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use approx::assert_abs_diff_eq;
    /// use delaunay::prelude::geometry::{
    ///     LabeledSimplexRealization, LabeledSimplexRealizationError,
    /// };
    ///
    /// # fn main() -> Result<(), LabeledSimplexRealizationError> {
    /// let simplex = LabeledSimplexRealization::try_new(
    ///     [0_usize, 1, 2],
    ///     [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
    /// )?;
    ///
    /// assert_abs_diff_eq!(simplex.coordinates()[2][0], 0.5, epsilon = f64::EPSILON);
    /// assert_abs_diff_eq!(simplex.coordinates()[2][1], 1.0, epsilon = f64::EPSILON);
    /// # Ok(())
    /// # }
    /// ```
    pub fn coordinates(&self) -> &[[f64; D]] {
        &self.coordinates
    }

    /// Rehydrates a validated coordinate row as a [`Point`] without rechecking finiteness.
    pub(crate) fn point_at(&self, vertex_index: usize) -> Option<Point<D>> {
        self.coordinates.get(vertex_index).copied().map(|coords| {
            Point::from_validated_coordinates(
                ValidatedCoordinates::from_prevalidated_finite_values(coords),
            )
        })
    }

    /// Returns a realization translated by integer multiples of the periodic domain.
    ///
    /// The translated coordinates are re-validated so overflow to non-finite
    /// values becomes a typed realization error rather than a hidden predicate
    /// input.
    ///
    /// # Errors
    ///
    /// Returns [`LabeledSimplexRealizationError::InvalidPeriodicDomainPeriod`] if a
    /// period is non-finite or non-positive, or
    /// [`LabeledSimplexRealizationError::NonFiniteCoordinate`] if translating by
    /// `shift * period` produces a NaN or infinite coordinate.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::geometry::{
    ///     LabeledSimplexRealization, LabeledSimplexRealizationError,
    /// };
    /// use approx::assert_abs_diff_eq;
    ///
    /// # fn main() -> Result<(), LabeledSimplexRealizationError> {
    /// let simplex = LabeledSimplexRealization::try_new(
    ///     [0_usize, 1, 2],
    ///     [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]],
    /// )?;
    /// let translated = simplex.try_translated(&[1.0, 1.0], &[1, -1])?;
    ///
    /// assert_abs_diff_eq!(translated.coordinates()[0][0], 1.0, epsilon = f64::EPSILON);
    /// assert_abs_diff_eq!(translated.coordinates()[0][1], -1.0, epsilon = f64::EPSILON);
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_translated(
        &self,
        periods: &[f64; D],
        shift: &[i32; D],
    ) -> Result<Self, LabeledSimplexRealizationError>
    where
        L: Clone,
    {
        validate_periods(periods)?;

        let mut translated_coordinates = self.coordinates.clone();
        for coords in &mut translated_coordinates {
            for axis in 0..D {
                coords[axis] = f64::from(shift[axis]).mul_add(periods[axis], coords[axis]);
            }
        }
        validate_coordinate_rows(&translated_coordinates)?;
        Ok(Self {
            labels: self.labels.clone(),
            coordinates: translated_coordinates,
        })
    }
}

/// Validates translated or newly parsed coordinate rows before predicate use.
fn validate_coordinate_rows<const D: usize>(
    coordinates: &SimplexRealizationBuffer<[f64; D]>,
) -> Result<(), LabeledSimplexRealizationError> {
    for (vertex_index, coords) in coordinates.iter().enumerate() {
        for (coordinate_index, coordinate) in coords.iter().enumerate() {
            if !coordinate.is_finite() {
                return Err(LabeledSimplexRealizationError::NonFiniteCoordinate {
                    vertex_index,
                    coordinate_index,
                    coordinate_value: InvalidCoordinateValue::from_debug(coordinate),
                });
            }
        }
    }
    Ok(())
}

/// Errors produced while parsing a labeled simplex realization.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum LabeledSimplexRealizationError {
    /// The label and coordinate iterators produced different lengths.
    #[error("label count {label_count} does not match coordinate count {coordinate_count}")]
    LabelCoordinateLengthMismatch {
        /// Number of labels supplied by the caller.
        label_count: usize,
        /// Number of coordinate rows supplied by the caller.
        coordinate_count: usize,
    },
    /// The realization did not contain exactly D + 1 vertices.
    #[error("invalid simplex realization arity: expected {expected}, got {actual}")]
    InvalidArity {
        /// Required vertex count for one maximal D-simplex.
        expected: usize,
        /// Actual vertex count supplied by the caller.
        actual: usize,
    },
    /// A simplex label appeared more than once.
    #[error("duplicate simplex realization label at indices {first_index} and {duplicate_index}")]
    DuplicateLabel {
        /// First index containing the duplicated label.
        first_index: usize,
        /// Later index containing the same label.
        duplicate_index: usize,
    },
    /// A coordinate was NaN or infinite.
    #[error(
        "non-finite coordinate at vertex {vertex_index}, coordinate {coordinate_index}: {coordinate_value}"
    )]
    NonFiniteCoordinate {
        /// Index of the vertex with the invalid coordinate.
        vertex_index: usize,
        /// Coordinate axis containing the invalid value.
        coordinate_index: usize,
        /// Classified invalid floating-point value.
        coordinate_value: InvalidCoordinateValue,
    },
    /// A periodic domain period was invalid.
    #[error(transparent)]
    InvalidPeriodicDomainPeriod {
        /// Underlying invalid-period error.
        #[from]
        source: PeriodicSimplexSpanError,
    },
}

/// Errors produced while checking a simplex against periodic-domain periods.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum PeriodicSimplexSpanError {
    /// A period was NaN or infinite.
    #[error("non-finite periodic period at axis {axis}: {period}")]
    NonFinitePeriod {
        /// Periodic axis with the invalid period.
        axis: usize,
        /// Classified invalid period value.
        period: InvalidCoordinateValue,
    },
    /// A finite period was zero or negative.
    #[error("non-positive periodic period at axis {axis}: {period}")]
    NonPositivePeriod {
        /// Periodic axis with the invalid period.
        axis: usize,
        /// Raw finite non-positive period.
        period: f64,
    },
}

/// Barycentric witness showing where two simplex realizations overlap illegally.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SimplexIntersectionWitness<L> {
    /// Labels appearing in both simplex realizations.
    pub shared: SimplexRealizationBuffer<L>,
    /// Labels from the first simplex with positive witness weight outside the shared face.
    pub first_only_witness: SimplexRealizationBuffer<L>,
    /// Labels from the second simplex with positive witness weight outside the shared face.
    pub second_only_witness: SimplexRealizationBuffer<L>,
}

/// Failure modes for exact simplex-intersection validation.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum SimplexIntersectionFailure<L> {
    /// The first simplex basis is singular, so barycentric coordinates are undefined.
    #[error("simplex barycentric basis is singular")]
    SingularBarycentricBasis,
    /// The simplices intersect at a point involving non-shared vertices.
    #[error("simplices intersect outside their shared face")]
    #[non_exhaustive]
    IntersectionOutsideSharedFace {
        /// Barycentric witness for the illegal intersection.
        witness: SimplexIntersectionWitness<L>,
    },
}

/// Coordinate-span witness for a simplex that is too wide for one periodic chart.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PeriodicSimplexSpan {
    /// Periodic axis whose coordinate span reaches or exceeds the period.
    axis: usize,
    /// Coordinate span along [`axis`](Self::axis).
    span: f64,
    /// Fundamental-domain period along [`axis`](Self::axis).
    period: f64,
}

impl PeriodicSimplexSpan {
    /// Periodic axis whose coordinate span reaches or exceeds the period.
    #[must_use]
    pub const fn axis(&self) -> usize {
        self.axis
    }

    /// Coordinate span along [`axis`](Self::axis).
    #[must_use]
    pub const fn span(&self) -> f64 {
        self.span
    }

    /// Fundamental-domain period along [`axis`](Self::axis).
    #[must_use]
    pub const fn period(&self) -> f64 {
        self.period
    }
}

/// Returns the closed coordinate range of a simplex along one axis.
///
/// Returns [`None`] when `axis >= D`.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::{
///     LabeledSimplexRealization, LabeledSimplexRealizationError, coordinate_range_for_axis,
/// };
/// use approx::abs_diff_eq;
///
/// # fn main() -> Result<(), LabeledSimplexRealizationError> {
/// let simplex = LabeledSimplexRealization::try_new(
///     [0_usize, 1, 2],
///     [[-1.0, 0.0], [2.0, 0.5], [0.0, 1.0]],
/// )?;
///
/// std::assert_matches!(
///     coordinate_range_for_axis(&simplex, 0),
///     Some((min, max))
///         if abs_diff_eq!(min, -1.0, epsilon = f64::EPSILON)
///             && abs_diff_eq!(max, 2.0, epsilon = f64::EPSILON)
/// );
/// assert_eq!(coordinate_range_for_axis(&simplex, 2), None);
/// # Ok(())
/// # }
/// ```
pub fn coordinate_range_for_axis<L, const D: usize>(
    simplex: &LabeledSimplexRealization<L, D>,
    axis: usize,
) -> Option<(f64, f64)> {
    if axis >= D {
        return None;
    }

    Some(simplex.coordinates.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(min_coord, max_coord), coords| (min_coord.min(coords[axis]), max_coord.max(coords[axis])),
    ))
}

/// Returns whether two simplex axis-aligned bounding boxes overlap.
///
/// This is a conservative broad-phase predicate: a `true` result means the
/// boxes overlap and exact simplex-intersection validation may be needed, not
/// that the simplices themselves necessarily intersect.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::{
///     LabeledSimplexRealization, LabeledSimplexRealizationError,
///     axis_aligned_bounding_boxes_overlap,
/// };
///
/// # fn main() -> Result<(), LabeledSimplexRealizationError> {
/// let first = LabeledSimplexRealization::try_new(
///     [0_usize, 1, 2],
///     [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
/// )?;
/// let second = LabeledSimplexRealization::try_new(
///     [3_usize, 4, 5],
///     [[2.0, 2.0], [3.0, 2.0], [2.0, 3.0]],
/// )?;
///
/// assert!(!axis_aligned_bounding_boxes_overlap(&first, &second));
/// # Ok(())
/// # }
/// ```
pub fn axis_aligned_bounding_boxes_overlap<L1, L2, const D: usize>(
    first: &LabeledSimplexRealization<L1, D>,
    second: &LabeledSimplexRealization<L2, D>,
) -> bool {
    (0..D).all(|axis| {
        let Some((first_min, first_max)) = coordinate_range_for_axis(first, axis) else {
            return false;
        };
        let Some((second_min, second_max)) = coordinate_range_for_axis(second, axis) else {
            return false;
        };
        first_max >= second_min && second_max >= first_min
    })
}

/// Finds the first periodic axis whose simplex span cannot fit in one chart.
///
/// # Errors
///
/// Returns [`PeriodicSimplexSpanError`] when any period is non-finite or not
/// strictly positive.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::{
///     LabeledSimplexRealization, LabeledSimplexRealizationError,
///     PeriodicSimplexSpanError, try_periodic_simplex_span,
/// };
///
/// #[derive(Debug, thiserror::Error)]
/// enum ExampleError {
///     #[error(transparent)]
///     Realization(#[from] LabeledSimplexRealizationError),
///     #[error(transparent)]
///     PeriodicSpan(#[from] PeriodicSimplexSpanError),
/// }
///
/// # fn main() -> Result<(), ExampleError> {
/// let simplex = LabeledSimplexRealization::try_new(
///     [0_usize, 1, 2],
///     [[0.0, 0.0], [1.0, 0.0], [0.0, 0.25]],
/// )?;
///
/// let span = try_periodic_simplex_span(&simplex, &[1.0, 2.0])?;
/// assert_eq!(span.map(|witness| witness.axis()), Some(0));
/// # Ok(())
/// # }
/// ```
pub fn try_periodic_simplex_span<L, const D: usize>(
    simplex: &LabeledSimplexRealization<L, D>,
    periods: &[f64; D],
) -> Result<Option<PeriodicSimplexSpan>, PeriodicSimplexSpanError> {
    validate_periods(periods)?;

    for (axis, &period) in periods.iter().enumerate() {
        let (min_coord, max_coord) = simplex.coordinates().iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(min_coord, max_coord), coords| {
                let coord = coords[axis];
                (min_coord.min(coord), max_coord.max(coord))
            },
        );
        let span = max_coord - min_coord;
        if span >= period {
            return Ok(Some(PeriodicSimplexSpan { axis, span, period }));
        }
    }
    Ok(None)
}

/// Proves that every periodic-domain period is finite and strictly positive.
fn validate_periods<const D: usize>(periods: &[f64; D]) -> Result<(), PeriodicSimplexSpanError> {
    for (axis, &period) in periods.iter().enumerate() {
        if !period.is_finite() {
            return Err(PeriodicSimplexSpanError::NonFinitePeriod {
                axis,
                period: InvalidCoordinateValue::from_debug(&period),
            });
        }
        if period <= 0.0 {
            return Err(PeriodicSimplexSpanError::NonPositivePeriod { axis, period });
        }
    }
    Ok(())
}

/// Validates that two simplex realizations meet only along labels they share.
///
/// This is the pure geometric core of Level 4 overlap validation. It uses
/// exact rational barycentric arithmetic after coordinates have been parsed as
/// finite f64 values.
///
/// # Errors
///
/// Returns [`SimplexIntersectionFailure::SingularBarycentricBasis`] when the
/// first simplex cannot define barycentric coordinates, or
/// [`SimplexIntersectionFailure::IntersectionOutsideSharedFace`] when the
/// two simplex interiors overlap away from their shared labels.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::{
///     LabeledSimplexRealization, LabeledSimplexRealizationError, SimplexIntersectionFailure,
///     validate_simplex_realizations_intersect_only_in_shared_faces,
/// };
///
/// # fn main() -> Result<(), LabeledSimplexRealizationError> {
/// let first = LabeledSimplexRealization::try_new(
///     [0_usize, 1, 2],
///     [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
/// )?;
/// let second = LabeledSimplexRealization::try_new(
///     [0_usize, 1, 3],
///     [[0.0, 0.0], [1.0, 0.0], [0.25, 0.25]],
/// )?;
///
/// std::assert_matches!(
///     validate_simplex_realizations_intersect_only_in_shared_faces(&first, &second),
///     Err(SimplexIntersectionFailure::IntersectionOutsideSharedFace { .. })
/// );
/// # Ok(())
/// # }
/// ```
pub fn validate_simplex_realizations_intersect_only_in_shared_faces<L, const D: usize>(
    first: &LabeledSimplexRealization<L, D>,
    second: &LabeledSimplexRealization<L, D>,
) -> Result<(), SimplexIntersectionFailure<L>>
where
    L: Clone + Eq,
{
    let shared_labels = shared_labels(first, second);
    let basis_orientation = realization_orientation(first);
    if basis_orientation == Some(Orientation::DEGENERATE) {
        return Err(SimplexIntersectionFailure::SingularBarycentricBasis);
    }
    let second_orientation = realization_orientation(second);
    if second_orientation == Some(Orientation::DEGENERATE) {
        return intersection_result(intersection_via_legacy_active_sets(
            first,
            second,
            &shared_labels,
        ));
    }
    if shared_face_fast_confinement(first, second, &shared_labels) {
        return Ok(());
    }
    if let Some(orientation) = basis_orientation
        && (simplex_is_strictly_outside_a_facet(first, second, orientation)
            || second_orientation.is_some_and(|orientation| {
                simplex_is_strictly_outside_a_facet(second, first, orientation)
            })
            || intersection_is_confined_by_orientation(first, second, &shared_labels, orientation))
    {
        return Ok(());
    }

    intersection_result(intersection_via_linear_program(
        first,
        second,
        &shared_labels,
        basis_orientation.is_none(),
    ))
}

/// Maps one private exact-intersection result onto the public typed error surface.
fn intersection_result<L>(
    result: IntersectionLinearProgramResult<L>,
) -> Result<(), SimplexIntersectionFailure<L>> {
    match result {
        IntersectionLinearProgramResult::Valid => Ok(()),
        IntersectionLinearProgramResult::Invalid(witness) => {
            Err(SimplexIntersectionFailure::IntersectionOutsideSharedFace { witness })
        }
        IntersectionLinearProgramResult::SingularBarycentricBasis => {
            Err(SimplexIntersectionFailure::SingularBarycentricBasis)
        }
    }
}

/// Computes one simplex orientation through the filtered-exact predicate path.
fn realization_orientation<L, const D: usize>(
    simplex: &LabeledSimplexRealization<L, D>,
) -> Option<Orientation> {
    let points: SimplexRealizationBuffer<_> = (0..simplex.coordinates().len())
        .filter_map(|index| simplex.point_at(index))
        .collect();
    robust_orientation(&points).ok()
}

/// Proves disjointness when one simplex lies strictly beyond a facet of another.
///
/// Replacing facet-opposite basis vertex `i` with a point `p` scales the basis
/// orientation by the barycentric coordinate of `p` at `i`. If every vertex of
/// the other simplex has the opposite non-zero sign for one `i`, convexity
/// places that entire simplex in the open exterior half-space.
fn simplex_is_strictly_outside_a_facet<L, const D: usize>(
    basis: &LabeledSimplexRealization<L, D>,
    other: &LabeledSimplexRealization<L, D>,
    basis_orientation: Orientation,
) -> bool {
    let basis_points: SimplexRealizationBuffer<_> = (0..basis.coordinates().len())
        .filter_map(|index| basis.point_at(index))
        .collect();

    (0..basis.labels().len()).any(|basis_index| {
        (0..other.coordinates().len()).all(|other_index| {
            let Some(other_point) = other.point_at(other_index) else {
                return false;
            };
            let mut replaced_points = basis_points.clone();
            replaced_points[basis_index] = other_point;
            robust_orientation(&replaced_points).is_ok_and(|orientation| {
                orientation != basis_orientation && orientation != Orientation::DEGENERATE
            })
        })
    })
}

/// Uses filtered-exact orientation signs to prove confinement to the shared face.
///
/// Replacing basis vertex `i` with a point `p` changes the orientation
/// determinant by the barycentric coordinate of `p` at `i`. Therefore, when
/// every vertex of the other simplex has a zero or opposite determinant sign
/// for each non-shared basis vertex, convexity proves that any point in both
/// simplices has zero weight outside the shared face.
fn intersection_is_confined_by_orientation<L, const D: usize>(
    basis: &LabeledSimplexRealization<L, D>,
    other: &LabeledSimplexRealization<L, D>,
    shared_labels: &[L],
    basis_orientation: Orientation,
) -> bool
where
    L: Eq,
{
    for shared_label in shared_labels {
        let Some(basis_index) = basis
            .labels()
            .iter()
            .position(|candidate| candidate == shared_label)
        else {
            return false;
        };
        let Some(other_index) = other
            .labels()
            .iter()
            .position(|candidate| candidate == shared_label)
        else {
            return false;
        };
        if !coordinates_are_identical(
            &basis.coordinates()[basis_index],
            &other.coordinates()[other_index],
        ) {
            return false;
        }
    }

    let basis_points: SimplexRealizationBuffer<_> = (0..basis.coordinates().len())
        .filter_map(|index| basis.point_at(index))
        .collect();

    for (basis_index, basis_label) in basis.labels().iter().enumerate() {
        if shared_labels.contains(basis_label) {
            continue;
        }
        for other_index in 0..other.coordinates().len() {
            let Some(other_point) = other.point_at(other_index) else {
                return false;
            };
            let mut replaced_points = basis_points.clone();
            replaced_points[basis_index] = other_point;
            let Ok(replaced_orientation) = robust_orientation(&replaced_points) else {
                return false;
            };
            if replaced_orientation == basis_orientation {
                return false;
            }
        }
    }
    true
}

/// Collects labels common to two simplex realizations so witnesses can distinguish shared faces.
fn shared_labels<L, const D: usize>(
    first: &LabeledSimplexRealization<L, D>,
    second: &LabeledSimplexRealization<L, D>,
) -> SimplexRealizationBuffer<L>
where
    L: Clone + Eq,
{
    first
        .labels()
        .iter()
        .filter(|label| second.labels().contains(label))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::assert_matches;

    #[derive(Clone)]
    struct CloneOnlyLabel;

    #[test]
    fn labeled_simplex_realization_rejects_label_coordinate_length_mismatch() {
        let err =
            LabeledSimplexRealization::<_, 2>::try_new(vec![0, 1, 2], vec![[0.0, 0.0], [1.0, 0.0]])
                .unwrap_err();

        assert_matches!(
            err,
            LabeledSimplexRealizationError::LabelCoordinateLengthMismatch {
                label_count: 3,
                coordinate_count: 2,
            }
        );
    }

    #[test]
    fn labeled_simplex_realization_rejects_invalid_arity() {
        let err = LabeledSimplexRealization::<_, 2>::try_new(
            vec![0, 1, 2, 3],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        )
        .unwrap_err();

        assert_matches!(
            err,
            LabeledSimplexRealizationError::InvalidArity {
                expected: 3,
                actual: 4,
            }
        );
    }

    #[test]
    fn labeled_simplex_realization_rejects_duplicate_labels() {
        let err = LabeledSimplexRealization::<_, 2>::try_new(
            vec![0, 1, 0],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        )
        .unwrap_err();

        assert_matches!(
            err,
            LabeledSimplexRealizationError::DuplicateLabel {
                first_index: 0,
                duplicate_index: 2,
            }
        );
    }

    #[test]
    fn coordinate_range_rejects_out_of_bounds_axis() {
        let simplex = LabeledSimplexRealization::try_new(
            vec![0, 1, 2],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        )
        .unwrap();

        assert_eq!(coordinate_range_for_axis(&simplex, 2), None);
    }

    #[test]
    fn disjoint_triangles_do_not_intersect_outside_shared_face() {
        let first = LabeledSimplexRealization::try_new(
            vec![0, 1, 2],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        )
        .unwrap();
        let second = LabeledSimplexRealization::try_new(
            vec![3, 4, 5],
            vec![[2.0, 2.0], [3.0, 2.0], [2.0, 3.0]],
        )
        .unwrap();

        assert!(
            validate_simplex_realizations_intersect_only_in_shared_faces(&first, &second).is_ok()
        );
    }

    #[test]
    fn orientation_confinement_rejects_mismatched_shared_coordinates() {
        let first =
            LabeledSimplexRealization::try_new([0, 1, 2], [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
                .unwrap();
        let second = LabeledSimplexRealization::try_new(
            [0, 3, 4],
            [[-1.0, -1.0], [-2.0, -1.0], [-1.0, -2.0]],
        )
        .unwrap();
        let orientation = realization_orientation(&first).expect("standard triangle is oriented");

        assert!(!intersection_is_confined_by_orientation(
            &first,
            &second,
            &[0],
            orientation,
        ));
    }

    #[test]
    fn labeled_simplex_realization_rejects_non_finite_coordinates() {
        let err = LabeledSimplexRealization::try_new(
            vec![0, 1, 2],
            vec![[0.0, 0.0], [1.0, f64::NAN], [0.0, 1.0]],
        )
        .unwrap_err();

        assert_matches!(
            err,
            LabeledSimplexRealizationError::NonFiniteCoordinate {
                vertex_index: 1,
                coordinate_index: 1,
                coordinate_value: InvalidCoordinateValue::Nan,
            }
        );
    }

    #[test]
    fn labeled_simplex_realization_rehydrates_points_from_validated_rows() {
        let simplex = LabeledSimplexRealization::try_new(
            vec![0, 1, 2],
            vec![[-0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        )
        .unwrap();

        let point = simplex.point_at(0).expect("vertex index exists");

        assert_eq!(point.coords()[0].to_bits(), 0.0_f64.to_bits());
        assert_eq!(point.coords()[1].to_bits(), 0.0_f64.to_bits());
        assert!(simplex.point_at(3).is_none());
    }

    #[test]
    fn translated_realization_rejects_non_finite_coordinates() {
        let simplex = LabeledSimplexRealization::try_new(
            vec![0, 1, 2],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        )
        .unwrap();

        let err = simplex
            .try_translated(&[f64::MAX, 1.0], &[2, 0])
            .unwrap_err();

        assert_matches!(
            err,
            LabeledSimplexRealizationError::NonFiniteCoordinate {
                vertex_index: 0,
                coordinate_index: 0,
                coordinate_value: InvalidCoordinateValue::PositiveInfinity,
            }
        );
    }

    #[test]
    fn translated_realization_rejects_invalid_periods() {
        let simplex = LabeledSimplexRealization::try_new(
            vec![0, 1, 2],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        )
        .unwrap();

        let err = simplex.try_translated(&[1.0, -1.0], &[0, 1]).unwrap_err();

        assert_matches!(
            err,
            LabeledSimplexRealizationError::InvalidPeriodicDomainPeriod {
                source: PeriodicSimplexSpanError::NonPositivePeriod {
                    axis: 1,
                    period: -1.0,
                },
            }
        );
    }

    #[test]
    fn translated_realization_requires_only_clone_labels() {
        let simplex = LabeledSimplexRealization {
            labels: vec![CloneOnlyLabel, CloneOnlyLabel, CloneOnlyLabel]
                .into_iter()
                .collect(),
            coordinates: vec![[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]]
                .into_iter()
                .collect(),
        };

        let translated = simplex
            .try_translated(&[1.0, 1.0], &[1, 0])
            .expect("translation preserves already-validated labels");

        assert_eq!(translated.labels().len(), 3);
        assert_abs_diff_eq!(translated.coordinates()[1][0], 1.5, epsilon = f64::EPSILON);
    }

    #[test]
    fn crossing_triangles_report_positive_nonshared_witnesses() {
        let first = LabeledSimplexRealization::try_new(
            vec![0, 1, 2],
            vec![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
        )
        .unwrap();
        let second = LabeledSimplexRealization::try_new(
            vec![3, 4, 5],
            vec![[2.0, 2.0], [1.0, -1.0], [3.0, 2.0]],
        )
        .unwrap();

        let err = validate_simplex_realizations_intersect_only_in_shared_faces(&first, &second)
            .unwrap_err();
        assert_matches!(
            err,
            SimplexIntersectionFailure::IntersectionOutsideSharedFace { witness, .. }
                if witness.first_only_witness.iter().any(|label| [0, 1, 2].contains(label))
                    && witness.second_only_witness.iter().any(|label| [3, 4, 5].contains(label))
        );
    }

    #[test]
    fn spanning_periodic_simplex_is_detected() {
        let simplex = LabeledSimplexRealization::try_new(
            vec![0, 1, 2],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 0.25]],
        )
        .unwrap();

        let span = try_periodic_simplex_span(&simplex, &[1.0, 1.0])
            .unwrap()
            .unwrap();
        assert_eq!(span.axis(), 0);
        assert_abs_diff_eq!(span.span(), 1.0, epsilon = f64::EPSILON);
        assert_abs_diff_eq!(span.period(), 1.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn periodic_simplex_span_rejects_invalid_periods() {
        let simplex = LabeledSimplexRealization::try_new(
            vec![0, 1, 2],
            vec![[0.0, 0.0], [0.5, 0.0], [0.0, 0.25]],
        )
        .unwrap();

        let non_finite = try_periodic_simplex_span(&simplex, &[f64::NAN, 1.0]).unwrap_err();
        assert_matches!(
            non_finite,
            PeriodicSimplexSpanError::NonFinitePeriod {
                axis: 0,
                period: InvalidCoordinateValue::Nan,
            }
        );

        let non_positive = try_periodic_simplex_span(&simplex, &[1.0, 0.0]).unwrap_err();
        assert_matches!(
            non_positive,
            PeriodicSimplexSpanError::NonPositivePeriod {
                axis: 1,
                period: 0.0,
            }
        );
    }
}
