//! Pure predicates for labeled simplex embeddings.
//!
//! This module has no TDS, topology, or triangulation storage dependencies. It
//! answers geometric questions about labeled maximal simplices after another
//! layer has chosen the appropriate affine chart. Algorithmic provenance for
//! the Level 4 overlap broad phase is summarized in `REFERENCES.md`,
//! "Embedded-Geometry Overlap Detection (Level 4 Validation)".
//!
//! Use [`Triangulation::embedding_report`](crate::Triangulation::embedding_report)
//! when validating a stored triangulation. Use this module directly when a
//! caller already has chart-local simplex coordinates and wants the pure
//! geometric Level 4 predicate without TDS storage.
//!
//! # Examples
//!
//! ```rust
//! use delaunay::prelude::geometry::{
//!     LabeledSimplexEmbedding, LabeledSimplexEmbeddingError, SimplexIntersectionFailure,
//!     validate_simplex_embeddings_intersect_only_in_shared_faces,
//! };
//!
//! # fn main() -> Result<(), LabeledSimplexEmbeddingError> {
//! let first = LabeledSimplexEmbedding::try_new(
//!     [0_usize, 1, 2],
//!     [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
//! )?;
//! let second = LabeledSimplexEmbedding::try_new(
//!     [0_usize, 1, 3],
//!     [[0.0, 0.0], [1.0, 0.0], [0.25, 0.25]],
//! )?;
//!
//! std::assert_matches!(
//!     validate_simplex_embeddings_intersect_only_in_shared_faces(&first, &second),
//!     Err(SimplexIntersectionFailure::IntersectionOutsideSharedFace { .. })
//! );
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]

use crate::core::collections::{MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer};
use crate::geometry::point::{Point, ValidatedCoordinates};
use crate::geometry::traits::coordinate::InvalidCoordinateValue;
use la_stack::{BigInt, BigRational, FromPrimitive, Signed};
use thiserror::Error;

/// Stack-backed buffer for per-simplex embedding labels and coordinates.
pub type SimplexEmbeddingBuffer<T> = SmallBuffer<T, MAX_PRACTICAL_DIMENSION_SIZE>;

/// Validated coordinates for one labeled D-simplex in an affine chart.
///
/// Labels implement [`Eq`] and are unique within the simplex so intersection
/// witnesses can distinguish shared faces from accidental duplicate vertices.
/// Coordinates are finite `f64` values ready to be converted into exact
/// predicate inputs by the triangulation-level embedding validator.
#[derive(Clone, Debug, PartialEq)]
pub struct LabeledSimplexEmbedding<L, const D: usize> {
    labels: SimplexEmbeddingBuffer<L>,
    coordinates: SimplexEmbeddingBuffer<[f64; D]>,
}

impl<L, const D: usize> LabeledSimplexEmbedding<L, D> {
    /// Builds a labeled D-simplex embedding after checking arity, uniqueness, and finite coordinates.
    ///
    /// This is the parse boundary for pure simplex-embedding predicates: callers
    /// supply labels and coordinates in matching order, and the constructor
    /// stores only embeddings with exactly `D + 1` distinct labels. Labels use
    /// [`Eq`] because they represent vertex identity, not an approximate value.
    ///
    /// # Errors
    ///
    /// Returns [`LabeledSimplexEmbeddingError::LabelCoordinateLengthMismatch`]
    /// when the two iterators produce different lengths,
    /// [`LabeledSimplexEmbeddingError::InvalidArity`] when the simplex does not
    /// contain exactly `D + 1` vertices,
    /// [`LabeledSimplexEmbeddingError::DuplicateLabel`] when a label appears
    /// more than once, or
    /// [`LabeledSimplexEmbeddingError::NonFiniteCoordinate`] when any coordinate
    /// is NaN or infinite.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::geometry::{
    ///     LabeledSimplexEmbedding, LabeledSimplexEmbeddingError,
    /// };
    ///
    /// # fn main() -> Result<(), LabeledSimplexEmbeddingError> {
    /// let simplex = LabeledSimplexEmbedding::try_new(
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
    ) -> Result<Self, LabeledSimplexEmbeddingError>
    where
        L: Eq,
    {
        let labels: SimplexEmbeddingBuffer<L> = labels.into_iter().collect();
        let coordinates: SimplexEmbeddingBuffer<[f64; D]> = coordinates.into_iter().collect();

        if labels.len() != coordinates.len() {
            return Err(
                LabeledSimplexEmbeddingError::LabelCoordinateLengthMismatch {
                    label_count: labels.len(),
                    coordinate_count: coordinates.len(),
                },
            );
        }

        let expected = D + 1;
        if labels.len() != expected {
            return Err(LabeledSimplexEmbeddingError::InvalidArity {
                expected,
                actual: labels.len(),
            });
        }

        for (first_index, first_label) in labels.iter().enumerate() {
            if let Some(duplicate_offset) = labels[first_index + 1..]
                .iter()
                .position(|label| label == first_label)
            {
                return Err(LabeledSimplexEmbeddingError::DuplicateLabel {
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
    ///     LabeledSimplexEmbedding, LabeledSimplexEmbeddingError,
    /// };
    ///
    /// # fn main() -> Result<(), LabeledSimplexEmbeddingError> {
    /// let simplex = LabeledSimplexEmbedding::try_new(
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
    ///     LabeledSimplexEmbedding, LabeledSimplexEmbeddingError,
    /// };
    ///
    /// # fn main() -> Result<(), LabeledSimplexEmbeddingError> {
    /// let simplex = LabeledSimplexEmbedding::try_new(
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

    /// Returns an embedding translated by integer multiples of the periodic domain.
    ///
    /// The translated coordinates are re-validated so overflow to non-finite
    /// values becomes a typed embedding error rather than a hidden predicate
    /// input.
    ///
    /// # Errors
    ///
    /// Returns [`LabeledSimplexEmbeddingError::InvalidPeriodicDomainPeriod`] if a
    /// period is non-finite or non-positive, or
    /// [`LabeledSimplexEmbeddingError::NonFiniteCoordinate`] if translating by
    /// `shift * period` produces a NaN or infinite coordinate.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::geometry::{
    ///     LabeledSimplexEmbedding, LabeledSimplexEmbeddingError,
    /// };
    /// use approx::assert_abs_diff_eq;
    ///
    /// # fn main() -> Result<(), LabeledSimplexEmbeddingError> {
    /// let simplex = LabeledSimplexEmbedding::try_new(
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
    ) -> Result<Self, LabeledSimplexEmbeddingError>
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
    coordinates: &SimplexEmbeddingBuffer<[f64; D]>,
) -> Result<(), LabeledSimplexEmbeddingError> {
    for (vertex_index, coords) in coordinates.iter().enumerate() {
        for (coordinate_index, coordinate) in coords.iter().enumerate() {
            if !coordinate.is_finite() {
                return Err(LabeledSimplexEmbeddingError::NonFiniteCoordinate {
                    vertex_index,
                    coordinate_index,
                    coordinate_value: InvalidCoordinateValue::from_debug(coordinate),
                });
            }
        }
    }
    Ok(())
}

/// Errors produced while parsing a labeled simplex embedding.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum LabeledSimplexEmbeddingError {
    /// The label and coordinate iterators produced different lengths.
    #[error("label count {label_count} does not match coordinate count {coordinate_count}")]
    LabelCoordinateLengthMismatch {
        /// Number of labels supplied by the caller.
        label_count: usize,
        /// Number of coordinate rows supplied by the caller.
        coordinate_count: usize,
    },
    /// The embedding did not contain exactly D + 1 vertices.
    #[error("invalid simplex embedding arity: expected {expected}, got {actual}")]
    InvalidArity {
        /// Required vertex count for one maximal D-simplex.
        expected: usize,
        /// Actual vertex count supplied by the caller.
        actual: usize,
    },
    /// A simplex label appeared more than once.
    #[error("duplicate simplex embedding label at indices {first_index} and {duplicate_index}")]
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

/// Barycentric witness showing where two simplex embeddings overlap illegally.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SimplexIntersectionWitness<L> {
    /// Labels appearing in both simplex embeddings.
    pub shared: SimplexEmbeddingBuffer<L>,
    /// Labels from the first simplex with positive witness weight outside the shared face.
    pub first_only_witness: SimplexEmbeddingBuffer<L>,
    /// Labels from the second simplex with positive witness weight outside the shared face.
    pub second_only_witness: SimplexEmbeddingBuffer<L>,
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
    pub axis: usize,
    /// Coordinate span along [`axis`](Self::axis).
    pub span: f64,
    /// Fundamental-domain period along [`axis`](Self::axis).
    pub period: f64,
}

/// Returns the closed coordinate range of a simplex along one axis.
///
/// Returns [`None`] when `axis >= D`.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::{
///     LabeledSimplexEmbedding, LabeledSimplexEmbeddingError, coordinate_range_for_axis,
/// };
/// use approx::abs_diff_eq;
///
/// # fn main() -> Result<(), LabeledSimplexEmbeddingError> {
/// let simplex = LabeledSimplexEmbedding::try_new(
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
    simplex: &LabeledSimplexEmbedding<L, D>,
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
///     LabeledSimplexEmbedding, LabeledSimplexEmbeddingError,
///     axis_aligned_bounding_boxes_overlap,
/// };
///
/// # fn main() -> Result<(), LabeledSimplexEmbeddingError> {
/// let first = LabeledSimplexEmbedding::try_new(
///     [0_usize, 1, 2],
///     [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
/// )?;
/// let second = LabeledSimplexEmbedding::try_new(
///     [3_usize, 4, 5],
///     [[2.0, 2.0], [3.0, 2.0], [2.0, 3.0]],
/// )?;
///
/// assert!(!axis_aligned_bounding_boxes_overlap(&first, &second));
/// # Ok(())
/// # }
/// ```
pub fn axis_aligned_bounding_boxes_overlap<L1, L2, const D: usize>(
    first: &LabeledSimplexEmbedding<L1, D>,
    second: &LabeledSimplexEmbedding<L2, D>,
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
///     LabeledSimplexEmbedding, LabeledSimplexEmbeddingError,
///     PeriodicSimplexSpanError, try_periodic_simplex_span,
/// };
///
/// #[derive(Debug, thiserror::Error)]
/// enum ExampleError {
///     #[error(transparent)]
///     Embedding(#[from] LabeledSimplexEmbeddingError),
///     #[error(transparent)]
///     PeriodicSpan(#[from] PeriodicSimplexSpanError),
/// }
///
/// # fn main() -> Result<(), ExampleError> {
/// let simplex = LabeledSimplexEmbedding::try_new(
///     [0_usize, 1, 2],
///     [[0.0, 0.0], [1.0, 0.0], [0.0, 0.25]],
/// )?;
///
/// let span = try_periodic_simplex_span(&simplex, &[1.0, 2.0])?;
/// assert_eq!(span.map(|witness| witness.axis), Some(0));
/// # Ok(())
/// # }
/// ```
pub fn try_periodic_simplex_span<L, const D: usize>(
    simplex: &LabeledSimplexEmbedding<L, D>,
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

/// Validates that two simplex embeddings meet only along labels they share.
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
///     LabeledSimplexEmbedding, LabeledSimplexEmbeddingError, SimplexIntersectionFailure,
///     validate_simplex_embeddings_intersect_only_in_shared_faces,
/// };
///
/// # fn main() -> Result<(), LabeledSimplexEmbeddingError> {
/// let first = LabeledSimplexEmbedding::try_new(
///     [0_usize, 1, 2],
///     [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
/// )?;
/// let second = LabeledSimplexEmbedding::try_new(
///     [0_usize, 1, 3],
///     [[0.0, 0.0], [1.0, 0.0], [0.25, 0.25]],
/// )?;
///
/// std::assert_matches!(
///     validate_simplex_embeddings_intersect_only_in_shared_faces(&first, &second),
///     Err(SimplexIntersectionFailure::IntersectionOutsideSharedFace { .. })
/// );
/// # Ok(())
/// # }
/// ```
pub fn validate_simplex_embeddings_intersect_only_in_shared_faces<L, const D: usize>(
    first: &LabeledSimplexEmbedding<L, D>,
    second: &LabeledSimplexEmbedding<L, D>,
) -> Result<(), SimplexIntersectionFailure<L>>
where
    L: Clone + Eq,
{
    let shared_labels = shared_labels(first, second);
    let second_vertices_in_first = barycentric_coordinates_of_vertices(second, first)?;
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

/// Collects labels common to two simplex embeddings so witnesses can distinguish shared faces.
fn shared_labels<L, const D: usize>(
    first: &LabeledSimplexEmbedding<L, D>,
    second: &LabeledSimplexEmbedding<L, D>,
) -> SimplexEmbeddingBuffer<L>
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
    vertices: &LabeledSimplexEmbedding<L, D>,
    basis: &LabeledSimplexEmbedding<L, D>,
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
    simplex: &LabeledSimplexEmbedding<L, D>,
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
) -> SimplexEmbeddingBuffer<L>
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
    fn labeled_simplex_embedding_rejects_label_coordinate_length_mismatch() {
        let err =
            LabeledSimplexEmbedding::<_, 2>::try_new(vec![0, 1, 2], vec![[0.0, 0.0], [1.0, 0.0]])
                .unwrap_err();

        assert_matches!(
            err,
            LabeledSimplexEmbeddingError::LabelCoordinateLengthMismatch {
                label_count: 3,
                coordinate_count: 2,
            }
        );
    }

    #[test]
    fn labeled_simplex_embedding_rejects_invalid_arity() {
        let err = LabeledSimplexEmbedding::<_, 2>::try_new(
            vec![0, 1, 2, 3],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        )
        .unwrap_err();

        assert_matches!(
            err,
            LabeledSimplexEmbeddingError::InvalidArity {
                expected: 3,
                actual: 4,
            }
        );
    }

    #[test]
    fn labeled_simplex_embedding_rejects_duplicate_labels() {
        let err = LabeledSimplexEmbedding::<_, 2>::try_new(
            vec![0, 1, 0],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        )
        .unwrap_err();

        assert_matches!(
            err,
            LabeledSimplexEmbeddingError::DuplicateLabel {
                first_index: 0,
                duplicate_index: 2,
            }
        );
    }

    #[test]
    fn coordinate_range_rejects_out_of_bounds_axis() {
        let simplex = LabeledSimplexEmbedding::try_new(
            vec![0, 1, 2],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        )
        .unwrap();

        assert_eq!(coordinate_range_for_axis(&simplex, 2), None);
    }

    #[test]
    fn disjoint_triangles_do_not_intersect_outside_shared_face() {
        let first = LabeledSimplexEmbedding::try_new(
            vec![0, 1, 2],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        )
        .unwrap();
        let second = LabeledSimplexEmbedding::try_new(
            vec![3, 4, 5],
            vec![[2.0, 2.0], [3.0, 2.0], [2.0, 3.0]],
        )
        .unwrap();

        assert!(
            validate_simplex_embeddings_intersect_only_in_shared_faces(&first, &second).is_ok()
        );
    }

    #[test]
    fn labeled_simplex_embedding_rejects_non_finite_coordinates() {
        let err = LabeledSimplexEmbedding::try_new(
            vec![0, 1, 2],
            vec![[0.0, 0.0], [1.0, f64::NAN], [0.0, 1.0]],
        )
        .unwrap_err();

        assert_matches!(
            err,
            LabeledSimplexEmbeddingError::NonFiniteCoordinate {
                vertex_index: 1,
                coordinate_index: 1,
                coordinate_value: InvalidCoordinateValue::Nan,
            }
        );
    }

    #[test]
    fn labeled_simplex_embedding_rehydrates_points_from_validated_rows() {
        let simplex = LabeledSimplexEmbedding::try_new(
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
    fn translated_embedding_rejects_non_finite_coordinates() {
        let simplex = LabeledSimplexEmbedding::try_new(
            vec![0, 1, 2],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        )
        .unwrap();

        let err = simplex
            .try_translated(&[f64::MAX, 1.0], &[2, 0])
            .unwrap_err();

        assert_matches!(
            err,
            LabeledSimplexEmbeddingError::NonFiniteCoordinate {
                vertex_index: 0,
                coordinate_index: 0,
                coordinate_value: InvalidCoordinateValue::PositiveInfinity,
            }
        );
    }

    #[test]
    fn translated_embedding_rejects_invalid_periods() {
        let simplex = LabeledSimplexEmbedding::try_new(
            vec![0, 1, 2],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        )
        .unwrap();

        let err = simplex.try_translated(&[1.0, -1.0], &[0, 1]).unwrap_err();

        assert_matches!(
            err,
            LabeledSimplexEmbeddingError::InvalidPeriodicDomainPeriod {
                source: PeriodicSimplexSpanError::NonPositivePeriod {
                    axis: 1,
                    period: -1.0,
                },
            }
        );
    }

    #[test]
    fn translated_embedding_requires_only_clone_labels() {
        let simplex = LabeledSimplexEmbedding {
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
        let first = LabeledSimplexEmbedding::try_new(
            vec![0, 1, 2],
            vec![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
        )
        .unwrap();
        let second = LabeledSimplexEmbedding::try_new(
            vec![3, 4, 5],
            vec![[2.0, 2.0], [1.0, -1.0], [3.0, 2.0]],
        )
        .unwrap();

        let err = validate_simplex_embeddings_intersect_only_in_shared_faces(&first, &second)
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
        let simplex = LabeledSimplexEmbedding::try_new(
            vec![0, 1, 2],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 0.25]],
        )
        .unwrap();

        let span = try_periodic_simplex_span(&simplex, &[1.0, 1.0])
            .unwrap()
            .unwrap();
        assert_eq!(span.axis, 0);
        assert_abs_diff_eq!(span.span, 1.0, epsilon = f64::EPSILON);
        assert_abs_diff_eq!(span.period, 1.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn periodic_simplex_span_rejects_invalid_periods() {
        let simplex = LabeledSimplexEmbedding::try_new(
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
