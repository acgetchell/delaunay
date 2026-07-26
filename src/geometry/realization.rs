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
use crate::geometry::matrix::{StackMatrixDispatchError, solve_exact_runtime_system};
use crate::geometry::point::{Point, ValidatedCoordinates};
use crate::geometry::predicates::Orientation;
use crate::geometry::robust_predicates::robust_orientation;
use crate::geometry::traits::coordinate::InvalidCoordinateValue;
use la_stack::{BigInt, BigRational, FromPrimitive, Signed, ToPrimitive};
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
    if shared_face_fast_confinement(first, second, &shared_labels) {
        return Ok(());
    }
    let basis_orientation = realization_orientation(first);
    if basis_orientation == Some(Orientation::DEGENERATE) {
        return Err(SimplexIntersectionFailure::SingularBarycentricBasis);
    }
    if let Some(orientation) = basis_orientation
        && (simplex_is_strictly_outside_a_facet(first, second, orientation)
            || realization_orientation(second).is_some_and(|second_orientation| {
                second_orientation != Orientation::DEGENERATE
                    && simplex_is_strictly_outside_a_facet(second, first, second_orientation)
            })
            || intersection_is_confined_by_orientation(first, second, &shared_labels, orientation))
    {
        return Ok(());
    }

    let fallback_barycentric = if basis_orientation.is_none() {
        Some(barycentric_coordinates_of_vertices(second, first)?)
    } else {
        None
    };

    match intersection_via_linear_program(first, second, &shared_labels) {
        IntersectionLinearProgramResult::Valid => return Ok(()),
        IntersectionLinearProgramResult::Invalid(witness) => {
            return Err(SimplexIntersectionFailure::IntersectionOutsideSharedFace { witness });
        }
        IntersectionLinearProgramResult::Fallback => {}
    }

    let second_vertices_in_first = match fallback_barycentric {
        Some(coordinates) => coordinates,
        None => barycentric_coordinates_of_vertices(second, first)?,
    };
    if intersection_is_confined_to_shared_face(
        &second_vertices_in_first,
        first.labels(),
        &shared_labels,
    ) {
        return Ok(());
    }
    let intersection_vertices = intersection_polytope_vertices(&second_vertices_in_first);

    for beta in intersection_vertices {
        let alpha = alpha_from_beta(&beta, &second_vertices_in_first);
        let first_only_witness_labels =
            positive_nonshared_labels(&alpha, first.labels(), &shared_labels);
        let second_only_witness_labels =
            positive_nonshared_labels(&beta, second.labels(), &shared_labels);

        if !first_only_witness_labels.is_empty() || !second_only_witness_labels.is_empty() {
            return Err(SimplexIntersectionFailure::IntersectionOutsideSharedFace {
                witness: SimplexIntersectionWitness {
                    shared: shared_labels,
                    first_only_witness: first_only_witness_labels,
                    second_only_witness: second_only_witness_labels,
                },
            });
        }
    }

    Ok(())
}

/// Tries cheap candidate separators in the normal space of one shared face.
#[expect(
    clippy::too_many_lines,
    reason = "the separator search is one bounded projection-and-certification workflow"
)]
fn shared_face_fast_confinement<L, const D: usize>(
    first: &LabeledSimplexRealization<L, D>,
    second: &LabeledSimplexRealization<L, D>,
    shared_labels: &[L],
) -> bool
where
    L: Eq,
{
    if shared_labels.is_empty() || shared_labels.len() >= D {
        return false;
    }
    let shared_coordinates: Vec<_> = shared_labels
        .iter()
        .map(|label| {
            let first_index = first
                .labels()
                .iter()
                .position(|candidate| candidate == label)?;
            let second_index = second
                .labels()
                .iter()
                .position(|candidate| candidate == label)?;
            coordinates_are_identical(
                &first.coordinates()[first_index],
                &second.coordinates()[second_index],
            )
            .then_some(first.coordinates()[first_index])
        })
        .collect::<Option<_>>()
        .unwrap_or_default();
    if shared_coordinates.len() != shared_labels.len() {
        return false;
    }
    let base = shared_coordinates[0];
    let shared_deltas: Vec<Vec<f64>> = shared_coordinates[1..]
        .iter()
        .map(|coordinates| {
            coordinates
                .iter()
                .zip(base)
                .map(|(coordinate, base_coordinate)| coordinate - base_coordinate)
                .collect()
        })
        .collect();
    let gram: Vec<Vec<f64>> = shared_deltas
        .iter()
        .map(|left| {
            shared_deltas
                .iter()
                .map(|right| dot_product_f64(left, right))
                .collect()
        })
        .collect();

    let mut axis = vec![0.0; D];
    let mut normalized_rays = Vec::with_capacity(2 * D);
    for (label, coordinates) in first.labels().iter().zip(first.coordinates()) {
        if shared_labels.contains(label) {
            continue;
        }
        let raw_ray: Vec<_> = coordinates
            .iter()
            .zip(base)
            .map(|(coordinate, shared)| coordinate - shared)
            .collect();
        let Some(ray) = project_ray_orthogonal_to_shared_face(raw_ray, &shared_deltas, &gram)
        else {
            return false;
        };
        for coordinate in 0..D {
            axis[coordinate] += ray[coordinate];
        }
        push_normalized_ray(&mut normalized_rays, ray);
    }
    for (label, coordinates) in second.labels().iter().zip(second.coordinates()) {
        if shared_labels.contains(label) {
            continue;
        }
        let raw_ray: Vec<_> = base
            .iter()
            .zip(coordinates)
            .map(|(shared, coordinate)| shared - coordinate)
            .collect();
        let Some(ray) = project_ray_orthogonal_to_shared_face(raw_ray, &shared_deltas, &gram)
        else {
            return false;
        };
        for coordinate in 0..D {
            axis[coordinate] += ray[coordinate];
        }
        push_normalized_ray(&mut normalized_rays, ray);
    }
    if axis.iter().all(|value| value.is_finite())
        && provisional_axis_certifies_confinement(first, second, shared_labels, &axis)
    {
        return true;
    }

    let mut refined_axis = vec![0.0; D];
    for ray in &normalized_rays {
        for (value, component) in refined_axis.iter_mut().zip(ray) {
            *value += component;
        }
    }
    for _ in 0..16 {
        let mut adjusted = false;
        for ray in &normalized_rays {
            let projection = dot_product_f64(&refined_axis, ray);
            if projection < 1.0 {
                let correction = 1.0 - projection;
                for (value, component) in refined_axis.iter_mut().zip(ray) {
                    *value = component.mul_add(correction, *value);
                }
                adjusted = true;
            }
        }
        if !adjusted {
            break;
        }
    }
    refined_axis.iter().all(|value| value.is_finite())
        && provisional_axis_certifies_confinement(first, second, shared_labels, &refined_axis)
}

/// Removes provisional components tangent to the shared affine face.
fn project_ray_orthogonal_to_shared_face(
    mut ray: Vec<f64>,
    shared_deltas: &[Vec<f64>],
    gram: &[Vec<f64>],
) -> Option<Vec<f64>> {
    if shared_deltas.is_empty() {
        return Some(ray);
    }
    let residuals: Vec<_> = shared_deltas
        .iter()
        .map(|delta| dot_product_f64(delta, &ray))
        .collect();
    let correction = solve_f64_system(gram.to_vec(), residuals)?;
    for (coordinate, value) in ray.iter_mut().enumerate() {
        for (coefficient, delta) in correction.iter().zip(shared_deltas) {
            *value = (-coefficient).mul_add(delta[coordinate], *value);
        }
    }
    ray.iter().all(|value| value.is_finite()).then_some(ray)
}

/// Certifies one provisional shared-face separator through a filtered or exact path.
fn provisional_axis_certifies_confinement<L, const D: usize>(
    first: &LabeledSimplexRealization<L, D>,
    second: &LabeledSimplexRealization<L, D>,
    shared_labels: &[L],
    axis: &[f64],
) -> bool
where
    L: Eq,
{
    if shared_labels.len() == 1 {
        filtered_single_shared_vertex_confinement(first, second, shared_labels, axis)
    } else {
        exact_shared_face_confinement_certificate(first, second, shared_labels, axis)
    }
}

/// Appends one finite unit ray for provisional separator refinement.
fn push_normalized_ray(rays: &mut Vec<Vec<f64>>, ray: Vec<f64>) {
    let norm = ray
        .iter()
        .fold(0.0_f64, |value, component| value.hypot(*component));
    if norm.is_finite() && norm > 0.0 {
        rays.push(ray.into_iter().map(|component| component / norm).collect());
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

/// Accepts a pair when simplex-facet signs prove any intersection lies in the shared face.
///
/// A point inside the basis simplex has non-negative barycentric coordinates.
/// If every vertex of the other simplex has a non-positive coordinate for each
/// basis vertex outside the shared face, convexity makes those coordinates
/// non-positive throughout the other simplex. Any point in both simplices must
/// therefore have zero weight on every non-shared basis vertex.
fn intersection_is_confined_to_shared_face<L>(
    other_vertices_in_basis: &[Vec<BigRational>],
    basis_labels: &[L],
    shared_labels: &[L],
) -> bool
where
    L: Eq,
{
    basis_labels
        .iter()
        .enumerate()
        .filter(|(_, label)| !shared_labels.contains(label))
        .all(|(basis_index, _)| {
            other_vertices_in_basis
                .iter()
                .all(|barycentric| !barycentric[basis_index].is_positive())
        })
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

/// Expresses every vertex of one simplex in the barycentric basis of another.
fn barycentric_coordinates_of_vertices<L, const D: usize>(
    vertices: &LabeledSimplexRealization<L, D>,
    basis: &LabeledSimplexRealization<L, D>,
) -> Result<Vec<Vec<BigRational>>, SimplexIntersectionFailure<L>> {
    vertices
        .coordinates()
        .iter()
        .map(|coords| barycentric_coordinates(coords, basis))
        .collect()
}

/// Computes exact barycentric coordinates of one point in one simplex basis.
fn barycentric_coordinates<L, const D: usize>(
    point: &[f64; D],
    simplex: &LabeledSimplexRealization<L, D>,
) -> Result<Vec<BigRational>, SimplexIntersectionFailure<L>> {
    if D == 0 {
        return Ok(vec![rational_one()]);
    }

    let origin = &simplex.coordinates()[0];
    let mut matrix = vec![vec![rational_zero(); D]; D];
    let mut rhs = vec![rational_zero(); D];

    for axis in 0..D {
        let origin_coord = rational_from_f64(origin[axis]);
        rhs[axis] = rational_from_f64(point[axis]) - origin_coord.clone();
        for (column, matrix_value) in matrix[axis].iter_mut().enumerate() {
            *matrix_value =
                rational_from_f64(simplex.coordinates()[column + 1][axis]) - origin_coord.clone();
        }
    }

    let lambdas = solve_rational_system(matrix, rhs)
        .ok_or(SimplexIntersectionFailure::SingularBarycentricBasis)?;
    let lambda_sum = lambdas
        .iter()
        .fold(rational_zero(), |acc, value| acc + value.clone());
    let mut barycentric = Vec::with_capacity(D + 1);
    barycentric.push(rational_one() - lambda_sum);
    barycentric.extend(lambdas);
    Ok(barycentric)
}

/// Outcome of the exact linear-program intersection path.
#[derive(Debug)]
enum IntersectionLinearProgramResult<L> {
    /// The simplices are disjoint or meet only in their shared face.
    Valid,
    /// A feasible point has positive weight outside the shared face.
    Invalid(SimplexIntersectionWitness<L>),
    /// An internal simplex-method precondition failed; use active-set enumeration.
    Fallback,
}

/// Uses exact linear programming to avoid enumerating every intersection-polytope basis.
///
/// The variables are the non-negative barycentric weights `alpha` and `beta`
/// of the two simplices. Equality constraints require each weight vector to sum
/// to one and both weighted coordinate sums to describe the same point.
/// Maximizing the total weight on non-shared labels is zero exactly when every
/// feasible intersection point lies in the shared face.
fn intersection_via_linear_program<L, const D: usize>(
    first: &LabeledSimplexRealization<L, D>,
    second: &LabeledSimplexRealization<L, D>,
    shared_labels: &[L],
) -> IntersectionLinearProgramResult<L>
where
    L: Clone + Eq,
{
    let first_labels = first.labels();
    let second_labels = second.labels();
    let simplex_vertex_count = first_labels.len();
    if second_labels.len() != simplex_vertex_count
        || first.coordinates().len() != simplex_vertex_count
        || second.coordinates().len() != simplex_vertex_count
    {
        return IntersectionLinearProgramResult::Fallback;
    }

    let variable_count = simplex_vertex_count * 2;
    let constraint_count = D + 2;
    let mut float_matrix = vec![vec![0.0; variable_count]; constraint_count];
    let mut float_rhs = vec![0.0; constraint_count];
    float_rhs[0] = 1.0;
    float_rhs[1] = 1.0;

    for (alpha_index, coordinates) in first.coordinates().iter().enumerate() {
        float_matrix[0][alpha_index] = 1.0;
        for (axis, coordinate) in coordinates.iter().enumerate() {
            float_matrix[axis + 2][alpha_index] = *coordinate;
        }
    }
    for (beta_index, coordinates) in second.coordinates().iter().enumerate() {
        float_matrix[1][simplex_vertex_count + beta_index] = 1.0;
        for (axis, coordinate) in coordinates.iter().enumerate() {
            float_matrix[axis + 2][simplex_vertex_count + beta_index] = -*coordinate;
        }
    }

    let mut float_objective = vec![0.0; variable_count];
    for (index, label) in first_labels.iter().enumerate() {
        if !shared_labels.contains(label) {
            float_objective[index] = 1.0;
        }
    }
    for (index, label) in second_labels.iter().enumerate() {
        if !shared_labels.contains(label) {
            float_objective[simplex_vertex_count + index] = 1.0;
        }
    }

    let ProvisionalPhaseDuals {
        infeasibility_axis,
        confinement_axis,
        confinement_basis,
    } = provisional_phase_dual_axes(&float_matrix, &float_rhs, &float_objective, variable_count);
    if infeasibility_axis
        .is_some_and(|axis| simplices_are_strictly_separated_along(first, second, &axis))
    {
        return IntersectionLinearProgramResult::Valid;
    }
    let axis_certifies = confinement_axis.is_some_and(|axis| {
        filtered_single_shared_vertex_confinement(first, second, shared_labels, &axis)
            || exact_shared_face_confinement_certificate(first, second, shared_labels, &axis)
    });
    if axis_certifies {
        return IntersectionLinearProgramResult::Valid;
    }

    let matrix = exact_matrix_from_f64(&float_matrix);
    let rhs: Vec<_> = float_rhs.iter().copied().map(rational_from_f64).collect();
    let objective: Vec<_> = float_objective
        .iter()
        .copied()
        .map(rational_from_f64)
        .collect();
    let dual_certifies = confinement_basis.is_some_and(|basis| {
        phase_two_dual_proves_shared_face_confinement(&matrix, &rhs, &objective, &basis)
    });
    if dual_certifies {
        return IntersectionLinearProgramResult::Valid;
    }

    match maximize_nonnegative_equality_linear_program(&matrix, &rhs, &objective) {
        ExactLinearProgramResult::Infeasible => IntersectionLinearProgramResult::Valid,
        ExactLinearProgramResult::Optimal {
            objective_value, ..
        } if !objective_value.is_positive() => IntersectionLinearProgramResult::Valid,
        ExactLinearProgramResult::Optimal { solution, .. } => {
            let first_only_witness = positive_nonshared_labels(
                &solution[..simplex_vertex_count],
                first_labels,
                shared_labels,
            );
            let second_only_witness = positive_nonshared_labels(
                &solution[simplex_vertex_count..],
                second_labels,
                shared_labels,
            );
            if first_only_witness.is_empty() && second_only_witness.is_empty() {
                return IntersectionLinearProgramResult::Fallback;
            }
            IntersectionLinearProgramResult::Invalid(SimplexIntersectionWitness {
                shared: shared_labels.iter().cloned().collect(),
                first_only_witness,
                second_only_witness,
            })
        }
        ExactLinearProgramResult::Failed => IntersectionLinearProgramResult::Fallback,
    }
}

/// Result of an exact non-negative equality-constrained linear program.
enum ExactLinearProgramResult {
    /// The objective is maximized at an exact basic feasible solution.
    Optimal {
        /// Values for the original decision variables.
        solution: Vec<BigRational>,
        /// Exact objective value at `solution`.
        objective_value: BigRational,
    },
    /// No non-negative solution satisfies the equality constraints.
    Infeasible,
    /// Tableau construction or a boundedness invariant failed.
    Failed,
}

/// Result of one revised-simplex optimization phase.
enum RevisedSimplexResult {
    /// The current basis is optimal.
    Optimal {
        /// Values for every variable in the supplied matrix.
        solution: Vec<BigRational>,
        /// Exact objective value at `solution`.
        objective_value: BigRational,
    },
    /// The objective can increase without bound.
    Unbounded,
    /// The supplied basis or iteration budget failed.
    Failed,
}

/// Maximizes an exact objective subject to `matrix * x = rhs` and `x >= 0`.
///
/// Phase I introduces one artificial variable per equality. Bland's pivot
/// ordering makes degenerate exact-arithmetic bases deterministic and avoids
/// cycling. Artificial variables are pivoted out at zero before Phase II.
fn maximize_nonnegative_equality_linear_program(
    matrix: &[Vec<BigRational>],
    rhs: &[BigRational],
    objective: &[BigRational],
) -> ExactLinearProgramResult {
    let constraint_count = rhs.len();
    let variable_count = objective.len();
    if constraint_count == 0
        || matrix.len() != constraint_count
        || matrix.iter().any(|row| row.len() != variable_count)
        || rhs.iter().any(Signed::is_negative)
    {
        return ExactLinearProgramResult::Failed;
    }

    let total_variable_count = variable_count + constraint_count;
    let mut phase_matrix = Vec::with_capacity(constraint_count);
    for (row_index, row) in matrix.iter().enumerate() {
        let mut phase_row = Vec::with_capacity(total_variable_count);
        phase_row.extend(row.iter().cloned());
        phase_row.extend((0..constraint_count).map(|column| {
            if column == row_index {
                rational_one()
            } else {
                rational_zero()
            }
        }));
        phase_matrix.push(phase_row);
    }

    let mut basis: Vec<usize> = (variable_count..total_variable_count).collect();
    let mut phase_one_objective = vec![rational_zero(); total_variable_count];
    for coefficient in &mut phase_one_objective[variable_count..] {
        *coefficient = -rational_one();
    }

    let phase_one = revised_simplex_maximize(
        &phase_matrix,
        rhs,
        &phase_one_objective,
        total_variable_count,
        &mut basis,
    );
    let RevisedSimplexResult::Optimal {
        objective_value, ..
    } = phase_one
    else {
        return ExactLinearProgramResult::Failed;
    };
    if objective_value.is_negative() {
        return ExactLinearProgramResult::Infeasible;
    }
    if objective_value.is_positive()
        || !pivot_artificial_variables_out(&phase_matrix, rhs, variable_count, &mut basis)
    {
        return ExactLinearProgramResult::Failed;
    }

    let mut phase_two_objective = vec![rational_zero(); total_variable_count];
    phase_two_objective[..variable_count].clone_from_slice(objective);
    match revised_simplex_maximize(
        &phase_matrix,
        rhs,
        &phase_two_objective,
        variable_count,
        &mut basis,
    ) {
        RevisedSimplexResult::Optimal {
            solution,
            objective_value,
        } => ExactLinearProgramResult::Optimal {
            solution: solution[..variable_count].to_vec(),
            objective_value,
        },
        RevisedSimplexResult::Unbounded | RevisedSimplexResult::Failed => {
            ExactLinearProgramResult::Failed
        }
    }
}

/// Removes zero-valued Phase I artificial variables from the active basis.
fn pivot_artificial_variables_out(
    matrix: &[Vec<BigRational>],
    rhs: &[BigRational],
    original_variable_count: usize,
    basis: &mut [usize],
) -> bool {
    for basis_position in 0..basis.len() {
        if basis[basis_position] < original_variable_count {
            continue;
        }

        let Some(current_basis_matrix) = basis_matrix(matrix, basis) else {
            return false;
        };
        let Some(basic_solution) =
            solve_rational_system(current_basis_matrix.clone(), rhs.to_vec())
        else {
            return false;
        };
        if basic_solution[basis_position] != rational_zero() {
            return false;
        }

        let mut replacement = None;
        for candidate in 0..original_variable_count {
            if basis.contains(&candidate) {
                continue;
            }
            let Some(direction) = solve_rational_system(
                current_basis_matrix.clone(),
                matrix_column(matrix, candidate),
            ) else {
                continue;
            };
            if direction[basis_position] != rational_zero() {
                replacement = Some(candidate);
                break;
            }
        }
        let Some(candidate) = replacement else {
            return false;
        };
        basis[basis_position] = candidate;
    }
    true
}

/// Uses a floating-point basis search followed by an exact optimality certificate.
///
/// The floating-point phase is only a guide: it cannot return a validation
/// result directly. A candidate basis is accepted only after exact rational
/// primal and dual checks. If certification fails, the deterministic exact
/// simplex path restarts from the original basis.
fn revised_simplex_maximize(
    matrix: &[Vec<BigRational>],
    rhs: &[BigRational],
    objective: &[BigRational],
    entering_variable_count: usize,
    basis: &mut [usize],
) -> RevisedSimplexResult {
    let starting_basis = basis.to_vec();
    if navigate_simplex_basis_f64(matrix, rhs, objective, entering_variable_count, basis).is_some()
        && let Some(optimal) =
            certify_optimal_basis(matrix, rhs, objective, entering_variable_count, basis)
    {
        return optimal;
    }

    basis.clone_from_slice(&starting_basis);
    revised_simplex_maximize_exact(matrix, rhs, objective, entering_variable_count, basis)
}

/// Runs deterministic exact revised simplex from an existing feasible basis.
fn revised_simplex_maximize_exact(
    matrix: &[Vec<BigRational>],
    rhs: &[BigRational],
    objective: &[BigRational],
    entering_variable_count: usize,
    basis: &mut [usize],
) -> RevisedSimplexResult {
    const MAX_SIMPLEX_ITERATIONS: usize = 10_000;

    if matrix.is_empty()
        || matrix.len() != rhs.len()
        || basis.len() != rhs.len()
        || matrix.iter().any(|row| row.len() != objective.len())
        || entering_variable_count > objective.len()
    {
        return RevisedSimplexResult::Failed;
    }

    for _ in 0..MAX_SIMPLEX_ITERATIONS {
        let Some(current_basis_matrix) = basis_matrix(matrix, basis) else {
            return RevisedSimplexResult::Failed;
        };
        let Some(basic_solution) =
            solve_rational_system(current_basis_matrix.clone(), rhs.to_vec())
        else {
            return RevisedSimplexResult::Failed;
        };
        if basic_solution.iter().any(Signed::is_negative) {
            return RevisedSimplexResult::Failed;
        }

        let basic_objective: Vec<_> = basis
            .iter()
            .map(|&index| objective[index].clone())
            .collect();
        let Some(dual_solution) = solve_rational_system(
            transpose_square_matrix(&current_basis_matrix),
            basic_objective,
        ) else {
            return RevisedSimplexResult::Failed;
        };

        let entering = (0..entering_variable_count)
            .filter(|candidate| !basis.contains(candidate))
            .find(|&candidate| {
                let reduced_cost = objective[candidate].clone()
                    - dot_product(&dual_solution, &matrix_column(matrix, candidate));
                reduced_cost.is_positive()
            });
        let Some(entering) = entering else {
            let mut solution = vec![rational_zero(); objective.len()];
            for (&variable, value) in basis.iter().zip(&basic_solution) {
                solution[variable] = value.clone();
            }
            return RevisedSimplexResult::Optimal {
                objective_value: dot_product(objective, &solution),
                solution,
            };
        };

        let Some(direction) =
            solve_rational_system(current_basis_matrix, matrix_column(matrix, entering))
        else {
            return RevisedSimplexResult::Failed;
        };
        let leaving = direction
            .iter()
            .enumerate()
            .filter(|(_, coefficient)| coefficient.is_positive())
            .map(|(position, coefficient)| {
                (
                    position,
                    basic_solution[position].clone() / coefficient.clone(),
                )
            })
            .min_by(
                |(left_position, left_ratio), (right_position, right_ratio)| {
                    left_ratio
                        .cmp(right_ratio)
                        .then_with(|| basis[*left_position].cmp(&basis[*right_position]))
                },
            );
        let Some((leaving_position, _)) = leaving else {
            return RevisedSimplexResult::Unbounded;
        };
        basis[leaving_position] = entering;
    }

    RevisedSimplexResult::Failed
}

/// Searches for an optimal basis quickly using f64 arithmetic.
///
/// Every decision made here is provisional. The caller must pass the resulting
/// basis through [`certify_optimal_basis`] before using it.
fn navigate_simplex_basis_f64(
    matrix: &[Vec<BigRational>],
    rhs: &[BigRational],
    objective: &[BigRational],
    entering_variable_count: usize,
    basis: &mut [usize],
) -> Option<Vec<f64>> {
    let float_matrix = rational_matrix_to_f64(matrix)?;
    let float_rhs = rationals_to_f64(rhs)?;
    let float_objective = rationals_to_f64(objective)?;
    navigate_simplex_basis_f64_values(
        &float_matrix,
        &float_rhs,
        &float_objective,
        entering_variable_count,
        basis,
    )
}

/// Runs provisional revised simplex directly on finite floating-point values.
fn navigate_simplex_basis_f64_values(
    matrix: &[Vec<f64>],
    rhs: &[f64],
    objective: &[f64],
    entering_variable_count: usize,
    basis: &mut [usize],
) -> Option<Vec<f64>> {
    const MAX_SIMPLEX_ITERATIONS: usize = 10_000;
    const REDUCED_COST_TOLERANCE: f64 = 1.0e-10;
    const DIRECTION_TOLERANCE: f64 = 1.0e-12;
    const FEASIBILITY_TOLERANCE: f64 = 1.0e-9;

    if matrix.is_empty()
        || matrix.len() != rhs.len()
        || basis.len() != rhs.len()
        || matrix.iter().any(|row| row.len() != objective.len())
        || entering_variable_count > objective.len()
    {
        return None;
    }

    for _ in 0..MAX_SIMPLEX_ITERATIONS {
        let current_basis_matrix = basis_matrix_f64(matrix, basis)?;
        let mut basic_solution = solve_f64_system(current_basis_matrix.clone(), rhs.to_vec())?;
        if basic_solution
            .iter()
            .any(|&value| value < -FEASIBILITY_TOLERANCE)
        {
            return None;
        }
        for value in &mut basic_solution {
            if value.is_sign_negative() {
                *value = 0.0;
            }
        }

        let basic_objective: Vec<_> = basis.iter().map(|&index| objective[index]).collect();
        let dual_solution = solve_f64_system(
            transpose_square_matrix_f64(&current_basis_matrix),
            basic_objective,
        )?;

        let entering = (0..entering_variable_count)
            .filter(|candidate| !basis.contains(candidate))
            .find(|&candidate| {
                let column = matrix_column_f64(matrix, candidate);
                let dual_contribution = dot_product_f64(&dual_solution, &column);
                let scale = 1.0 + objective[candidate].abs() + dual_contribution.abs();
                objective[candidate] - dual_contribution > REDUCED_COST_TOLERANCE * scale
            });
        let Some(entering) = entering else {
            return Some(dual_solution);
        };

        let direction =
            solve_f64_system(current_basis_matrix, matrix_column_f64(matrix, entering))?;
        let leaving = direction
            .iter()
            .enumerate()
            .filter(|(_, coefficient)| **coefficient > DIRECTION_TOLERANCE)
            .map(|(position, coefficient)| (position, basic_solution[position] / coefficient))
            .min_by(
                |(left_position, left_ratio), (right_position, right_ratio)| {
                    left_ratio
                        .total_cmp(right_ratio)
                        .then_with(|| basis[*left_position].cmp(&basis[*right_position]))
                },
            );
        let (leaving_position, _) = leaving?;
        basis[leaving_position] = entering;
    }

    None
}

/// Finds candidate separating directions from floating-point Phase I and II duals.
///
/// Returned directions are only guides. Callers must verify their geometric
/// certificates with exact arithmetic before accepting them.
struct ProvisionalPhaseDuals {
    /// Candidate direction from the infeasibility phase.
    infeasibility_axis: Option<Vec<f64>>,
    /// Candidate direction from the shared-face-confinement phase.
    confinement_axis: Option<Vec<f64>>,
    /// Provisional Phase II basis available for exact dual certification.
    confinement_basis: Option<Vec<usize>>,
}

impl ProvisionalPhaseDuals {
    /// Returns a result with no provisional certificates.
    const fn none() -> Self {
        Self {
            infeasibility_axis: None,
            confinement_axis: None,
            confinement_basis: None,
        }
    }
}

fn provisional_phase_dual_axes(
    matrix: &[Vec<f64>],
    rhs: &[f64],
    original_objective: &[f64],
    variable_count: usize,
) -> ProvisionalPhaseDuals {
    let constraint_count = rhs.len();
    let Some(total_variable_count) = variable_count.checked_add(constraint_count) else {
        return ProvisionalPhaseDuals::none();
    };
    let mut phase_matrix = Vec::with_capacity(constraint_count);
    for (row_index, row) in matrix.iter().enumerate() {
        let mut phase_row = Vec::with_capacity(total_variable_count);
        phase_row.extend(row.iter().copied());
        phase_row.extend((0..constraint_count).map(
            |column| {
                if column == row_index { 1.0 } else { 0.0 }
            },
        ));
        phase_matrix.push(phase_row);
    }

    let mut basis: Vec<_> = (variable_count..total_variable_count).collect();
    let mut objective = vec![0.0; total_variable_count];
    for coefficient in &mut objective[variable_count..] {
        *coefficient = -1.0;
    }
    let Some(phase_one_dual) = navigate_simplex_basis_f64_values(
        &phase_matrix,
        rhs,
        &objective,
        total_variable_count,
        &mut basis,
    ) else {
        return ProvisionalPhaseDuals::none();
    };
    let phase_one_axis = (phase_one_dual.len() >= 2).then(|| phase_one_dual[2..].to_vec());

    if !pivot_artificial_variables_out_f64(&phase_matrix, rhs, variable_count, &mut basis) {
        return ProvisionalPhaseDuals {
            infeasibility_axis: phase_one_axis,
            confinement_axis: None,
            confinement_basis: None,
        };
    }
    let mut phase_two_objective = vec![0.0; total_variable_count];
    if original_objective.len() != variable_count {
        return ProvisionalPhaseDuals {
            infeasibility_axis: phase_one_axis,
            confinement_axis: None,
            confinement_basis: None,
        };
    }
    phase_two_objective[..variable_count].clone_from_slice(original_objective);
    let phase_two_dual = navigate_simplex_basis_f64_values(
        &phase_matrix,
        rhs,
        &phase_two_objective,
        variable_count,
        &mut basis,
    );
    let phase_two_axis = phase_two_dual
        .as_ref()
        .and_then(|dual| (dual.len() >= 2).then(|| dual[2..].to_vec()));
    let phase_two_basis = phase_two_dual.map(|_dual| basis);
    ProvisionalPhaseDuals {
        infeasibility_axis: phase_one_axis,
        confinement_axis: phase_two_axis,
        confinement_basis: phase_two_basis,
    }
}

/// Removes zero artificial variables from a provisional floating-point basis.
fn pivot_artificial_variables_out_f64(
    matrix: &[Vec<f64>],
    rhs: &[f64],
    original_variable_count: usize,
    basis: &mut [usize],
) -> bool {
    const ZERO_TOLERANCE: f64 = 1.0e-9;
    const PIVOT_TOLERANCE: f64 = 1.0e-12;

    for basis_position in 0..basis.len() {
        if basis[basis_position] < original_variable_count {
            continue;
        }
        let Some(current_basis_matrix) = basis_matrix_f64(matrix, basis) else {
            return false;
        };
        let Some(basic_solution) = solve_f64_system(current_basis_matrix.clone(), rhs.to_vec())
        else {
            return false;
        };
        if basic_solution[basis_position].abs() > ZERO_TOLERANCE {
            return false;
        }

        let replacement = (0..original_variable_count)
            .filter(|candidate| !basis.contains(candidate))
            .find(|&candidate| {
                solve_f64_system(
                    current_basis_matrix.clone(),
                    matrix_column_f64(matrix, candidate),
                )
                .is_some_and(|direction| direction[basis_position].abs() > PIVOT_TOLERANCE)
            });
        let Some(replacement) = replacement else {
            return false;
        };
        basis[basis_position] = replacement;
    }
    true
}

/// Certifies strict projection separation along a finite candidate direction.
fn simplices_are_strictly_separated_along<L, const D: usize>(
    first: &LabeledSimplexRealization<L, D>,
    second: &LabeledSimplexRealization<L, D>,
    axis: &[f64],
) -> bool {
    if axis.len() != D || axis.iter().any(|value| !value.is_finite()) {
        return false;
    }
    let exact_axis: Vec<_> = axis.iter().copied().map(rational_from_f64).collect();
    let Some((first_min, first_max)) = exact_projection_bounds(first, &exact_axis) else {
        return false;
    };
    let Some((second_min, second_max)) = exact_projection_bounds(second, &exact_axis) else {
        return false;
    };
    first_max < second_min || second_max < first_min
}

/// Returns exact projection bounds for one realized simplex.
fn exact_projection_bounds<L, const D: usize>(
    simplex: &LabeledSimplexRealization<L, D>,
    axis: &[BigRational],
) -> Option<(BigRational, BigRational)> {
    let mut projections = simplex.coordinates().iter().map(|coordinates| {
        coordinates
            .iter()
            .zip(axis)
            .fold(rational_zero(), |sum, (coordinate, coefficient)| {
                sum + rational_from_f64(*coordinate) * coefficient
            })
    });
    let first = projections.next()?;
    Some(
        projections.fold((first.clone(), first), |(minimum, maximum), projection| {
            (minimum.min(projection.clone()), maximum.max(projection))
        }),
    )
}

/// Certifies confinement through one shared vertex with a floating-point error bound.
///
/// Scaling the provisional direction is exact because the factor is a power of
/// two. Each signed projection difference is then accepted only when its
/// conservative lower bound exceeds the unit dual margin. Ambiguous,
/// underflow-sensitive, or overflowing cases continue to exact arithmetic.
fn filtered_single_shared_vertex_confinement<L, const D: usize>(
    first: &LabeledSimplexRealization<L, D>,
    second: &LabeledSimplexRealization<L, D>,
    shared_labels: &[L],
    axis: &[f64],
) -> bool
where
    L: Eq,
{
    let [shared_label] = shared_labels else {
        return false;
    };
    let Some(first_shared_index) = first
        .labels()
        .iter()
        .position(|candidate| candidate == shared_label)
    else {
        return false;
    };
    let Some(second_shared_index) = second
        .labels()
        .iter()
        .position(|candidate| candidate == shared_label)
    else {
        return false;
    };
    let first_shared = &first.coordinates()[first_shared_index];
    let second_shared = &second.coordinates()[second_shared_index];
    if !coordinates_are_identical(first_shared, second_shared) {
        return false;
    }

    let scaled_axis: Vec<_> = axis.iter().map(|value| *value * 1024.0).collect();
    if scaled_axis.len() != D || scaled_axis.iter().any(|value| !value.is_finite()) {
        return false;
    }
    first
        .labels()
        .iter()
        .zip(first.coordinates())
        .filter(|(label, _)| *label != shared_label)
        .all(|(_, coordinates)| {
            certified_dot_difference_lower_bound(&scaled_axis, coordinates, first_shared)
                .is_some_and(|lower_bound| lower_bound > 1.0)
        })
        && second
            .labels()
            .iter()
            .zip(second.coordinates())
            .filter(|(label, _)| *label != shared_label)
            .all(|(_, coordinates)| {
                certified_dot_difference_lower_bound(&scaled_axis, second_shared, coordinates)
                    .is_some_and(|lower_bound| lower_bound > 1.0)
            })
}

/// Returns a conservative lower bound for `axis · (left - right)`.
fn certified_dot_difference_lower_bound<const D: usize>(
    axis: &[f64],
    left: &[f64; D],
    right: &[f64; D],
) -> Option<f64> {
    let mut estimate = 0.0;
    let mut magnitude = 0.0;
    for ((coefficient, left_value), right_value) in axis.iter().zip(left).zip(right) {
        let left_product = coefficient * left_value;
        let right_product = coefficient * right_value;
        if !left_product.is_finite()
            || !right_product.is_finite()
            || product_underflowed(*coefficient, *left_value, left_product)
            || product_underflowed(*coefficient, *right_value, right_product)
        {
            return None;
        }
        estimate = coefficient.mul_add(*left_value, estimate);
        estimate = (-coefficient).mul_add(*right_value, estimate);
        magnitude += left_product.abs() + right_product.abs();
        if !estimate.is_finite() || !magnitude.is_finite() {
            return None;
        }
    }

    let operation_count = f64::from(u32::try_from(2 * D).ok()?);
    let denominator = 1.0 - operation_count * f64::EPSILON;
    if denominator <= 0.0 {
        return None;
    }
    let gamma = operation_count * f64::EPSILON / denominator;
    let magnitude_upper_bound = magnitude / denominator.powi(2);
    let error_bound = 4.0 * gamma * magnitude_upper_bound;
    error_bound.is_finite().then_some(estimate - error_bound)
}

/// Detects a product whose exact non-zero value rounded into the subnormal range.
fn product_underflowed(left: f64, right: f64, product: f64) -> bool {
    left != 0.0 && right != 0.0 && (product == 0.0 || product.abs() < f64::MIN_POSITIVE)
}

/// Compares finite coordinates exactly while treating both signed zeros alike.
fn coordinates_are_identical<const D: usize>(left: &[f64; D], right: &[f64; D]) -> bool {
    left.iter().zip(right).all(|(left, right)| {
        let left_bits = left.to_bits();
        let right_bits = right.to_bits();
        left_bits == right_bits || (left_bits << 1 == 0 && right_bits << 1 == 0)
    })
}

/// Certifies a Phase II dual after rebuilding its two affine offsets exactly.
///
/// For direction `w`, the smallest feasible dual offsets are
/// `max_i(c_i - w·a_i)` and `max_j(c_j + w·b_j)`. Their sum is an exact upper
/// bound on the maximum non-shared barycentric weight. A non-positive bound
/// proves that every intersection point lies in the shared face.
fn exact_shared_face_confinement_certificate<L, const D: usize>(
    first: &LabeledSimplexRealization<L, D>,
    second: &LabeledSimplexRealization<L, D>,
    shared_labels: &[L],
    axis: &[f64],
) -> bool
where
    L: Eq,
{
    if axis.len() != D || axis.iter().any(|value| !value.is_finite()) {
        return false;
    }
    let Some(exact_axis) = exact_axis_through_shared_face(first, shared_labels, axis) else {
        return false;
    };
    let first_offset = first
        .labels()
        .iter()
        .zip(first.coordinates())
        .map(|(label, coordinates)| {
            let objective = if shared_labels.contains(label) {
                rational_zero()
            } else {
                rational_one()
            };
            objective - exact_projection(coordinates, &exact_axis)
        })
        .max();
    let second_offset = second
        .labels()
        .iter()
        .zip(second.coordinates())
        .map(|(label, coordinates)| {
            let objective = if shared_labels.contains(label) {
                rational_zero()
            } else {
                rational_one()
            };
            objective + exact_projection(coordinates, &exact_axis)
        })
        .max();
    first_offset
        .zip(second_offset)
        .is_some_and(|(first_offset, second_offset)| {
            first_offset + second_offset <= rational_zero()
        })
}

/// Reconstructs a provisional axis in the shared face's exact affine nullspace.
///
/// Shared vertices define homogeneous constraints on `(axis, offset)`. A
/// floating-point rank search selects pivot columns, then the fraction-free
/// exact solver constructs one nullspace basis vector per free column directly
/// from the original coordinates. Combining those vectors with provisional
/// free-coordinate weights restores exact shared-face equality without a
/// rational Gram solve.
fn exact_axis_through_shared_face<L, const D: usize>(
    simplex: &LabeledSimplexRealization<L, D>,
    shared_labels: &[L],
    axis: &[f64],
) -> Option<Vec<BigRational>>
where
    L: Eq,
{
    let shared_coordinates: Vec<_> = shared_labels
        .iter()
        .map(|label| {
            simplex
                .labels()
                .iter()
                .position(|candidate| candidate == label)
                .and_then(|index| simplex.coordinates().get(index))
        })
        .collect::<Option<_>>()?;
    if shared_coordinates.is_empty() {
        return Some(
            axis.iter()
                .map(|value| rational_from_f64(*value * 1024.0))
                .collect(),
        );
    }

    let constraint_matrix: Vec<Vec<f64>> = shared_coordinates
        .iter()
        .map(|coordinates| {
            let mut row = coordinates.to_vec();
            row.push(1.0);
            row
        })
        .collect();
    let pivot_columns = independent_columns_f64(&constraint_matrix)?;
    let free_columns: Vec<_> = (0..=D)
        .filter(|column| !pivot_columns.contains(column))
        .collect();
    let pivot_matrix: Vec<Vec<f64>> = constraint_matrix
        .iter()
        .map(|row| pivot_columns.iter().map(|&column| row[column]).collect())
        .collect();

    let mut provisional_affine = axis.to_vec();
    provisional_affine.push(-dot_product_f64(axis, shared_coordinates[0]));
    let mut exact_affine = vec![rational_zero(); D + 1];
    for free_column in free_columns {
        let weight = rational_from_f64(provisional_affine[free_column]);
        exact_affine[free_column] += weight.clone();
        let rhs: Vec<_> = constraint_matrix
            .iter()
            .map(|row| -row[free_column])
            .collect();
        let solution = solve_exact_runtime_system(&pivot_matrix, &rhs)?.ok()?;
        for (&pivot_column, coefficient) in pivot_columns.iter().zip(solution) {
            exact_affine[pivot_column] += weight.clone() * coefficient;
        }
    }

    let scale = rational_from_f64(1024.0);
    for value in &mut exact_affine[..D] {
        *value *= scale.clone();
    }
    exact_affine.truncate(D);
    Some(exact_affine)
}

/// Selects a full-rank set of matrix columns with provisional elimination.
fn independent_columns_f64(matrix: &[Vec<f64>]) -> Option<Vec<usize>> {
    const PIVOT_TOLERANCE: f64 = 1.0e-12;

    let row_count = matrix.len();
    let column_count = matrix.first()?.len();
    if row_count == 0 || matrix.iter().any(|row| row.len() != column_count) {
        return None;
    }
    let mut work = matrix.to_vec();
    let scale = work
        .iter()
        .flatten()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    let mut pivot_columns = Vec::with_capacity(row_count);
    for column in 0..column_count {
        let pivot_position = (pivot_columns.len()..row_count).max_by(|&left, &right| {
            work[left][column]
                .abs()
                .total_cmp(&work[right][column].abs())
        })?;
        if work[pivot_position][column].abs() <= PIVOT_TOLERANCE * scale.max(1.0) {
            continue;
        }
        let pivot_row = pivot_columns.len();
        work.swap(pivot_row, pivot_position);
        let pivot = work[pivot_row][column];
        let pivot_values = work[pivot_row].clone();
        for row_values in work.iter_mut().skip(pivot_row + 1) {
            let factor = row_values[column] / pivot;
            for (value, pivot_value) in row_values[column..].iter_mut().zip(&pivot_values[column..])
            {
                *value -= factor * pivot_value;
            }
        }
        pivot_columns.push(column);
        if pivot_columns.len() == row_count {
            return Some(pivot_columns);
        }
    }
    None
}

/// Certifies the floating-point Phase II basis with one exact dual solve.
///
/// Dual feasibility and a non-positive exact objective bound prove confinement
/// without constructing a primal solution or repeating exact simplex pivots.
fn phase_two_dual_proves_shared_face_confinement(
    matrix: &[Vec<BigRational>],
    rhs: &[BigRational],
    objective: &[BigRational],
    basis: &[usize],
) -> bool {
    let constraint_count = rhs.len();
    let variable_count = objective.len();
    if matrix.len() != constraint_count
        || matrix.iter().any(|row| row.len() != variable_count)
        || basis.len() != constraint_count
    {
        return false;
    }

    let total_variable_count = variable_count + constraint_count;
    let phase_matrix: Vec<_> = matrix
        .iter()
        .enumerate()
        .map(|(row_index, row)| {
            let mut phase_row = Vec::with_capacity(total_variable_count);
            phase_row.extend(row.iter().cloned());
            phase_row.extend((0..constraint_count).map(|column| {
                if column == row_index {
                    rational_one()
                } else {
                    rational_zero()
                }
            }));
            phase_row
        })
        .collect();
    let mut phase_objective = vec![rational_zero(); total_variable_count];
    phase_objective[..variable_count].clone_from_slice(objective);

    let Some(current_basis_matrix) = basis_matrix(&phase_matrix, basis) else {
        return false;
    };
    let basic_objective: Vec<_> = basis
        .iter()
        .filter_map(|&index| phase_objective.get(index).cloned())
        .collect();
    if basic_objective.len() != constraint_count {
        return false;
    }
    let Some(dual_solution) = solve_rational_system(
        transpose_square_matrix(&current_basis_matrix),
        basic_objective,
    ) else {
        return false;
    };
    let dual_is_feasible = (0..variable_count).all(|candidate| {
        objective[candidate].clone()
            - dot_product(&dual_solution, &matrix_column(matrix, candidate))
            <= rational_zero()
    });
    dual_is_feasible && dot_product(rhs, &dual_solution) <= rational_zero()
}

/// Computes one exact projection along a finite candidate axis.
fn exact_projection<const D: usize>(coordinates: &[f64; D], axis: &[BigRational]) -> BigRational {
    coordinates
        .iter()
        .zip(axis)
        .fold(rational_zero(), |sum, (coordinate, coefficient)| {
            sum + rational_from_f64(*coordinate) * coefficient
        })
}

/// Certifies exact primal feasibility and dual optimality for one candidate basis.
fn certify_optimal_basis(
    matrix: &[Vec<BigRational>],
    rhs: &[BigRational],
    objective: &[BigRational],
    entering_variable_count: usize,
    basis: &[usize],
) -> Option<RevisedSimplexResult> {
    let current_basis_matrix = basis_matrix(matrix, basis)?;
    let basic_solution = solve_rational_system(current_basis_matrix.clone(), rhs.to_vec())?;
    if basic_solution.iter().any(Signed::is_negative) {
        return None;
    }

    let basic_objective: Vec<_> = basis
        .iter()
        .map(|&index| objective[index].clone())
        .collect();
    let dual_solution = solve_rational_system(
        transpose_square_matrix(&current_basis_matrix),
        basic_objective,
    )?;
    let dual_is_feasible = (0..entering_variable_count)
        .filter(|candidate| !basis.contains(candidate))
        .all(|candidate| {
            let reduced_cost = objective[candidate].clone()
                - dot_product(&dual_solution, &matrix_column(matrix, candidate));
            !reduced_cost.is_positive()
        });
    if !dual_is_feasible {
        return None;
    }

    let mut solution = vec![rational_zero(); objective.len()];
    for (&variable, value) in basis.iter().zip(&basic_solution) {
        solution[variable] = value.clone();
    }
    Some(RevisedSimplexResult::Optimal {
        objective_value: dot_product(objective, &solution),
        solution,
    })
}

/// Converts an exact rational matrix to finite f64 values for basis navigation.
fn rational_matrix_to_f64(matrix: &[Vec<BigRational>]) -> Option<Vec<Vec<f64>>> {
    matrix.iter().map(|row| rationals_to_f64(row)).collect()
}

/// Converts exact rationals to finite f64 values for provisional calculations.
fn rationals_to_f64(values: &[BigRational]) -> Option<Vec<f64>> {
    values
        .iter()
        .map(|value| value.to_f64().filter(|converted| converted.is_finite()))
        .collect()
}

/// Extracts a floating-point basis matrix.
fn basis_matrix_f64(matrix: &[Vec<f64>], basis: &[usize]) -> Option<Vec<Vec<f64>>> {
    let column_count = matrix.first()?.len();
    if matrix.len() != basis.len()
        || matrix.iter().any(|row| row.len() != column_count)
        || basis.iter().any(|&column| column >= column_count)
    {
        return None;
    }
    Some(
        matrix
            .iter()
            .map(|row| basis.iter().map(|&column| row[column]).collect())
            .collect(),
    )
}

/// Copies one floating-point matrix column.
fn matrix_column_f64(matrix: &[Vec<f64>], column: usize) -> Vec<f64> {
    matrix.iter().map(|row| row[column]).collect()
}

/// Transposes a square floating-point matrix.
fn transpose_square_matrix_f64(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let dimension = matrix.len();
    (0..dimension)
        .map(|row| (0..dimension).map(|column| matrix[column][row]).collect())
        .collect()
}

/// Computes a floating-point dot product used only for provisional pivot selection.
fn dot_product_f64(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

/// Solves a small floating-point system with scaled partial pivoting.
#[expect(
    clippy::needless_range_loop,
    reason = "index-based elimination keeps pivot row/column operations explicit"
)]
fn solve_f64_system(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    const PIVOT_TOLERANCE: f64 = 1.0e-14;

    let dimension = rhs.len();
    if matrix.len() != dimension || matrix.iter().any(|row| row.len() != dimension) {
        return None;
    }

    for pivot_column in 0..dimension {
        let pivot_row = (pivot_column..dimension).max_by(|&left, &right| {
            matrix[left][pivot_column]
                .abs()
                .total_cmp(&matrix[right][pivot_column].abs())
        })?;
        let row_scale = matrix[pivot_row]
            .iter()
            .map(|&value| value.abs())
            .fold(0.0_f64, f64::max);
        if !row_scale.is_finite()
            || matrix[pivot_row][pivot_column].abs() <= PIVOT_TOLERANCE * row_scale.max(1.0)
        {
            return None;
        }
        if pivot_row != pivot_column {
            matrix.swap(pivot_column, pivot_row);
            rhs.swap(pivot_column, pivot_row);
        }

        for row in pivot_column + 1..dimension {
            let factor = matrix[row][pivot_column] / matrix[pivot_column][pivot_column];
            matrix[row][pivot_column] = 0.0;
            for column in pivot_column + 1..dimension {
                matrix[row][column] =
                    factor.mul_add(-matrix[pivot_column][column], matrix[row][column]);
            }
            rhs[row] = factor.mul_add(-rhs[pivot_column], rhs[row]);
        }
    }

    let mut solution = vec![0.0; dimension];
    for row in (0..dimension).rev() {
        let mut sum = rhs[row];
        for column in row + 1..dimension {
            sum = matrix[row][column].mul_add(-solution[column], sum);
        }
        solution[row] = sum / matrix[row][row];
        if !solution[row].is_finite() {
            return None;
        }
    }
    Some(solution)
}

/// Extracts the square matrix formed by the current basis columns.
fn basis_matrix(matrix: &[Vec<BigRational>], basis: &[usize]) -> Option<Vec<Vec<BigRational>>> {
    let column_count = matrix.first()?.len();
    if matrix.len() != basis.len()
        || matrix.iter().any(|row| row.len() != column_count)
        || basis.iter().any(|&column| column >= column_count)
    {
        return None;
    }
    Some(
        matrix
            .iter()
            .map(|row| basis.iter().map(|&column| row[column].clone()).collect())
            .collect(),
    )
}

/// Copies one matrix column for an exact linear solve.
fn matrix_column(matrix: &[Vec<BigRational>], column: usize) -> Vec<BigRational> {
    matrix.iter().map(|row| row[column].clone()).collect()
}

/// Transposes a square exact-rational matrix.
fn transpose_square_matrix(matrix: &[Vec<BigRational>]) -> Vec<Vec<BigRational>> {
    let dimension = matrix.len();
    (0..dimension)
        .map(|row| {
            (0..dimension)
                .map(|column| matrix[column][row].clone())
                .collect()
        })
        .collect()
}

/// Computes an exact dot product without introducing floating-point decisions.
fn dot_product(left: &[BigRational], right: &[BigRational]) -> BigRational {
    left.iter()
        .zip(right)
        .fold(rational_zero(), |sum, (left, right)| {
            sum + left.clone() * right.clone()
        })
}

/// Enumerates candidate vertices of the intersection polytope in second-simplex weights.
fn intersection_polytope_vertices(
    second_vertices_in_first: &[Vec<BigRational>],
) -> Vec<Vec<BigRational>> {
    let variable_count = second_vertices_in_first.len();
    let active_count = variable_count.saturating_sub(1);
    let constraint_count = variable_count * 2;
    let mut active_set = Vec::with_capacity(active_count);
    let mut vertices = Vec::new();

    enumerate_active_sets(
        constraint_count,
        active_count,
        0,
        &mut active_set,
        &mut |active_constraints| {
            if let Some(beta) =
                intersection_vertex_for_active_set(second_vertices_in_first, active_constraints)
                && beta_is_feasible(&beta, second_vertices_in_first)
            {
                vertices.push(beta);
            }
        },
    );

    vertices
}

/// Recursively enumerates active constraint sets for the simplex-intersection LP.
fn enumerate_active_sets<F>(
    constraint_count: usize,
    active_count: usize,
    start: usize,
    active_set: &mut Vec<usize>,
    on_active_set: &mut F,
) where
    F: FnMut(&[usize]),
{
    if active_set.len() == active_count {
        on_active_set(active_set);
        return;
    }

    let remaining = active_count - active_set.len();
    let last_start = constraint_count.saturating_sub(remaining);
    for constraint in start..=last_start {
        active_set.push(constraint);
        enumerate_active_sets(
            constraint_count,
            active_count,
            constraint + 1,
            active_set,
            on_active_set,
        );
        active_set.pop();
    }
}

/// Solves one active-constraint system and returns the candidate beta weights.
fn intersection_vertex_for_active_set(
    second_vertices_in_first: &[Vec<BigRational>],
    active_constraints: &[usize],
) -> Option<Vec<BigRational>> {
    let variable_count = second_vertices_in_first.len();
    let mut matrix = Vec::with_capacity(variable_count);
    let mut rhs = Vec::with_capacity(variable_count);

    matrix.push(vec![rational_one(); variable_count]);
    rhs.push(rational_one());

    for &constraint in active_constraints {
        matrix.push(constraint_coefficients(
            second_vertices_in_first,
            constraint,
        ));
        rhs.push(rational_zero());
    }

    solve_rational_system(matrix, rhs)
}

/// Builds coefficients for either a beta non-negativity or alpha non-negativity constraint.
fn constraint_coefficients(
    second_vertices_in_first: &[Vec<BigRational>],
    constraint: usize,
) -> Vec<BigRational> {
    let variable_count = second_vertices_in_first.len();
    if constraint < variable_count {
        let mut coefficients = vec![rational_zero(); variable_count];
        coefficients[constraint] = rational_one();
        return coefficients;
    }

    let alpha_index = constraint - variable_count;
    second_vertices_in_first
        .iter()
        .map(|barycentric| barycentric[alpha_index].clone())
        .collect()
}

/// Checks whether beta weights and the induced alpha weights are all non-negative.
fn beta_is_feasible(beta: &[BigRational], second_vertices_in_first: &[Vec<BigRational>]) -> bool {
    beta.iter().all(|value| !value.is_negative())
        && alpha_from_beta(beta, second_vertices_in_first)
            .iter()
            .all(|value| !value.is_negative())
}

/// Converts second-simplex beta weights into first-simplex alpha weights.
fn alpha_from_beta(
    beta: &[BigRational],
    second_vertices_in_first: &[Vec<BigRational>],
) -> Vec<BigRational> {
    let variable_count = second_vertices_in_first.len();
    let mut alpha = vec![rational_zero(); variable_count];

    for (beta_index, beta_value) in beta.iter().enumerate() {
        for (alpha_index, alpha_value) in alpha.iter_mut().enumerate() {
            *alpha_value = alpha_value.clone()
                + beta_value.clone() * second_vertices_in_first[beta_index][alpha_index].clone();
        }
    }

    alpha
}

/// Returns labels whose barycentric coordinates witness mass outside the shared face.
fn positive_nonshared_labels<L>(
    barycentric: &[BigRational],
    labels: &[L],
    shared_labels: &[L],
) -> SimplexRealizationBuffer<L>
where
    L: Clone + Eq,
{
    labels
        .iter()
        .zip(barycentric)
        .filter(|(label, coordinate)| !shared_labels.contains(label) && coordinate.is_positive())
        .map(|(label, _coordinate)| label.clone())
        .collect()
}

#[expect(
    clippy::needless_range_loop,
    reason = "index-based elimination keeps pivot row/column operations explicit"
)]
/// Solves a square rational linear system by Gaussian elimination.
fn solve_rational_system(
    mut matrix: Vec<Vec<BigRational>>,
    mut rhs: Vec<BigRational>,
) -> Option<Vec<BigRational>> {
    let dimension = rhs.len();
    if matrix.len() != dimension || matrix.iter().any(|row| row.len() != dimension) {
        return None;
    }
    match try_solve_exact_f64_system(&matrix, &rhs) {
        ExactF64Solve::Solution(solution) => return Some(solution),
        ExactF64Solve::Singular => return None,
        ExactF64Solve::Unsupported => {}
    }

    for pivot_col in 0..dimension {
        let pivot_row =
            (pivot_col..dimension).find(|&row| matrix[row][pivot_col] != rational_zero())?;
        if pivot_row != pivot_col {
            matrix.swap(pivot_col, pivot_row);
            rhs.swap(pivot_col, pivot_row);
        }

        let pivot_value = matrix[pivot_col][pivot_col].clone();
        for row in pivot_col + 1..dimension {
            if matrix[row][pivot_col] == rational_zero() {
                continue;
            }
            let factor = matrix[row][pivot_col].clone() / pivot_value.clone();
            matrix[row][pivot_col] = rational_zero();
            for col in pivot_col + 1..dimension {
                matrix[row][col] =
                    matrix[row][col].clone() - factor.clone() * matrix[pivot_col][col].clone();
            }
            rhs[row] = rhs[row].clone() - factor * rhs[pivot_col].clone();
        }
    }

    let mut solution = vec![rational_zero(); dimension];
    for row in (0..dimension).rev() {
        let mut sum = rhs[row].clone();
        for col in row + 1..dimension {
            sum -= matrix[row][col].clone() * solution[col].clone();
        }
        solution[row] = sum / matrix[row][row].clone();
    }

    Some(solution)
}

/// Outcome of attempting the fraction-free exact `f64` solver.
enum ExactF64Solve {
    /// The runtime stack dispatcher cannot represent this rational system.
    Unsupported,
    /// The system is exactly singular.
    Singular,
    /// The exact system solution.
    Solution(Vec<BigRational>),
}

/// Uses fraction-free Bareiss elimination when every coefficient came directly
/// from a finite `f64`; otherwise leaves the rational solver as the fallback.
fn try_solve_exact_f64_system(matrix: &[Vec<BigRational>], rhs: &[BigRational]) -> ExactF64Solve {
    let Some(float_matrix) = exact_rational_matrix_as_f64(matrix) else {
        return ExactF64Solve::Unsupported;
    };
    let Some(float_rhs) = exact_rationals_as_f64(rhs) else {
        return ExactF64Solve::Unsupported;
    };
    let Some(result) = solve_exact_runtime_system(&float_matrix, &float_rhs) else {
        return ExactF64Solve::Unsupported;
    };
    match result {
        Ok(solution) => ExactF64Solve::Solution(solution),
        Err(StackMatrixDispatchError::La {
            source: la_stack::LaError::Singular { .. },
        }) => ExactF64Solve::Singular,
        Err(_) => ExactF64Solve::Unsupported,
    }
}

/// Converts rationals to `f64` only when doing so preserves each value exactly.
fn exact_rational_matrix_as_f64(matrix: &[Vec<BigRational>]) -> Option<Vec<Vec<f64>>> {
    matrix
        .iter()
        .map(|row| exact_rationals_as_f64(row))
        .collect()
}

/// Converts a finite floating-point matrix to its exact rational representation.
fn exact_matrix_from_f64(matrix: &[Vec<f64>]) -> Vec<Vec<BigRational>> {
    matrix
        .iter()
        .map(|row| row.iter().copied().map(rational_from_f64).collect())
        .collect()
}

/// Converts rationals to finite `f64` values and verifies an exact round trip.
fn exact_rationals_as_f64(values: &[BigRational]) -> Option<Vec<f64>> {
    values
        .iter()
        .map(|value| {
            let converted = value.to_f64().filter(|candidate| candidate.is_finite())?;
            (rational_from_f64(converted) == *value).then_some(converted)
        })
        .collect()
}

/// Converts a finite f64 to an exact rational value for barycentric predicates.
fn rational_from_f64(value: f64) -> BigRational {
    BigRational::from_f64(value)
        .expect("validated finite f64 coordinates must convert to BigRational")
}

/// Returns the additive identity used throughout exact barycentric arithmetic.
fn rational_zero() -> BigRational {
    BigRational::from_integer(BigInt::from(0))
}

/// Returns the multiplicative identity used throughout exact barycentric arithmetic.
fn rational_one() -> BigRational {
    BigRational::from_integer(BigInt::from(1))
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
    fn exact_linear_program_classifies_disjoint_triangles_as_valid() {
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
        let shared = shared_labels(&first, &second);

        assert_matches!(
            intersection_via_linear_program(&first, &second, &shared),
            IntersectionLinearProgramResult::Valid
        );
    }

    #[test]
    fn exact_linear_program_accepts_a_shared_facet_intersection() {
        let first = LabeledSimplexRealization::try_new(
            vec![0, 1, 2],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        )
        .unwrap();
        let second = LabeledSimplexRealization::try_new(
            vec![0, 1, 3],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, -1.0]],
        )
        .unwrap();
        let shared = shared_labels(&first, &second);

        assert_matches!(
            intersection_via_linear_program(&first, &second, &shared),
            IntersectionLinearProgramResult::Valid
        );
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
    fn exact_linear_program_reports_crossing_triangle_witnesses() {
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
        let shared = shared_labels(&first, &second);

        assert_matches!(
            intersection_via_linear_program(&first, &second, &shared),
            IntersectionLinearProgramResult::Invalid(witness)
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
