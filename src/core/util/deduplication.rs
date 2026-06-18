//! Vertex deduplication utilities.

#![forbid(unsafe_code)]

use crate::core::traits::data_type::DataType;
use crate::core::vertex::Vertex;
use crate::geometry::traits::coordinate::OrderedEq;
use thiserror::Error;

/// Errors returned by fallible vertex deduplication helpers.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeduplicationError {
    /// Epsilon must be non-negative for distance-based deduplication.
    #[error("epsilon must be non-negative")]
    NegativeEpsilon,

    /// Epsilon must be finite for distance-based deduplication.
    #[error("epsilon must be finite")]
    NonFiniteEpsilon,
}

/// Filters vertices to remove exact coordinate duplicates.
///
/// Uses `OrderedFloat`-based comparison to detect exact floating-point matches.
/// This treats NaN as equal to NaN and +0.0 as equal to -0.0, which is appropriate
/// for deduplication. More strict than epsilon-based comparison.
///
/// # Complexity
///
/// O(n²) where n is the number of vertices. This is acceptable for small to moderate
/// vertex counts (hundreds to low thousands). For very large point clouds, consider
/// spatial indexing structures or sorting-based approaches.
///
/// # Arguments
///
/// * `vertices` - Slice of vertices to deduplicate
///
/// # Returns
///
/// A new vector containing only unique vertices (by coordinates). The first
/// occurrence of each unique coordinate is kept.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::dedup_vertices_exact;
/// use delaunay::prelude::{Vertex};
///
/// # fn main() -> Result<(), delaunay::prelude::geometry::CoordinateConversionError> {
/// let v1: Vertex<(), 2> = delaunay::vertex![0.0, 0.0]?;
/// let v2: Vertex<(), 2> = delaunay::vertex![0.0, 0.0]?; // Duplicate
/// let v3: Vertex<(), 2> = delaunay::vertex![1.0, 1.0]?;
///
/// let vertices = vec![v1, v2, v3];
/// let unique = dedup_vertices_exact(&vertices);
/// assert_eq!(unique.len(), 2); // Only v1 and v3
/// # Ok(())
/// # }
/// ```
#[must_use]
pub fn dedup_vertices_exact<U, const D: usize>(vertices: &[Vertex<U, D>]) -> Vec<Vertex<U, D>>
where
    U: DataType,
{
    let mut unique: Vec<Vertex<U, D>> = Vec::with_capacity(vertices.len());

    for &v in vertices {
        if unique
            .iter()
            .any(|u| coords_equal_exact(v.point().coords(), u.point().coords()))
        {
            continue;
        }

        unique.push(v);
    }

    unique
}

/// Filters vertices to remove near-duplicates within epsilon tolerance.
///
/// Uses Euclidean distance to detect vertices within `epsilon` of each other.
/// This is more lenient than exact comparison and helps prevent numerical issues
/// from near-duplicate insertions.
///
/// # Complexity
///
/// O(n²) where n is the number of vertices. This is acceptable for small to moderate
/// vertex counts (hundreds to low thousands). For very large point clouds, consider
/// spatial indexing structures (e.g., k-d tree, octree) for efficient nearest-neighbor queries.
///
/// # Arguments
///
/// * `vertices` - Slice of vertices to deduplicate
/// * `epsilon` - Distance threshold below which vertices are considered duplicates
///
/// # Returns
///
/// A new vector containing vertices that are at least `epsilon` apart from each
/// other (distance >= epsilon). The first occurrence of each cluster is kept.
///
/// If `epsilon` is negative, NaN, or infinite, the input is returned unchanged
/// and a warning is emitted. Use [`try_dedup_vertices_epsilon`] when callers
/// should receive a typed error for invalid epsilon values.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::dedup_vertices_epsilon;
/// use delaunay::prelude::{Vertex};
///
/// # fn main() -> Result<(), delaunay::prelude::geometry::CoordinateConversionError> {
/// let v1: Vertex<(), 2> = delaunay::vertex![0.0, 0.0]?;
/// let v2: Vertex<(), 2> = delaunay::vertex![1e-11, 1e-11]?; // Near duplicate
/// let v3: Vertex<(), 2> = delaunay::vertex![1.0, 1.0]?;
///
/// let vertices = vec![v1, v2, v3];
/// let unique = dedup_vertices_epsilon(&vertices, 1e-10);
/// assert_eq!(unique.len(), 2); // v2 filtered as near-duplicate of v1
/// # Ok(())
/// # }
/// ```
pub fn dedup_vertices_epsilon<U, const D: usize>(
    vertices: &[Vertex<U, D>],
    epsilon: f64,
) -> Vec<Vertex<U, D>>
where
    U: DataType,
{
    if !epsilon.is_finite() || epsilon < 0.0 {
        tracing::warn!(
            epsilon = ?epsilon,
            "dedup_vertices_epsilon received non-finite or negative epsilon; returning input unchanged"
        );
        return vertices.to_vec();
    }

    dedup_vertices_epsilon_nonnegative(vertices, epsilon)
}

/// Fallible variant of [`dedup_vertices_epsilon`].
///
/// This function rejects negative, NaN, and infinite epsilon values with a
/// typed error instead of falling back to returning the input unchanged.
///
/// # Errors
///
/// Returns [`DeduplicationError::NegativeEpsilon`] when `epsilon` is negative.
/// Returns [`DeduplicationError::NonFiniteEpsilon`] when `epsilon` is NaN or
/// infinite.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::{DeduplicationError, try_dedup_vertices_epsilon};
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Source(#[from] DeduplicationError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0]?,
/// ];
///
/// let deduped = try_dedup_vertices_epsilon(&vertices, 1e-9)?;
/// assert_eq!(deduped.len(), 2);
///
/// std::assert_matches!(
///     try_dedup_vertices_epsilon(&vertices, f64::NAN),
///     Err(DeduplicationError::NonFiniteEpsilon)
/// );
/// # Ok(())
/// # }
/// ```
pub fn try_dedup_vertices_epsilon<U, const D: usize>(
    vertices: &[Vertex<U, D>],
    epsilon: f64,
) -> Result<Vec<Vertex<U, D>>, DeduplicationError>
where
    U: DataType,
{
    if !epsilon.is_finite() {
        return Err(DeduplicationError::NonFiniteEpsilon);
    }

    if epsilon < 0.0 {
        return Err(DeduplicationError::NegativeEpsilon);
    }

    Ok(dedup_vertices_epsilon_nonnegative(vertices, epsilon))
}

fn dedup_vertices_epsilon_nonnegative<U, const D: usize>(
    vertices: &[Vertex<U, D>],
    epsilon: f64,
) -> Vec<Vertex<U, D>>
where
    U: DataType,
{
    let mut unique: Vec<Vertex<U, D>> = Vec::with_capacity(vertices.len());

    for &v in vertices {
        if unique
            .iter()
            .any(|u| coords_within_epsilon(v.point().coords(), u.point().coords(), epsilon))
        {
            continue;
        }

        unique.push(v);
    }

    unique
}

/// Filters vertices to exclude those matching reference coordinates.
///
/// Useful for removing vertices that coincide with an initial simplex or other
/// fixed reference points. Uses `OrderedFloat`-based exact comparison (NaN-aware).
///
/// # Complexity
///
/// O(n·m) where n is the number of vertices and m is the number of reference vertices.
/// Typically m is small (D+1 for an initial simplex in dimension D), making this effectively
/// O(n) in practice.
///
/// # Arguments
///
/// * `vertices` - Slice of vertices to filter
/// * `reference` - Reference vertices to exclude matches against
///
/// # Returns
///
/// A new vector containing only vertices whose coordinates don't match any
/// reference vertex coordinates.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::filter_vertices_excluding;
/// use delaunay::prelude::{Vertex};
///
/// # fn main() -> Result<(), delaunay::prelude::geometry::CoordinateConversionError> {
/// let v1: Vertex<(), 2> = delaunay::vertex![0.0, 0.0]?;
/// let v2: Vertex<(), 2> = delaunay::vertex![1.0, 1.0]?;
///
/// let reference = vec![v1]; // Exclude origin
/// let vertices = vec![v1, v2];
///
/// let filtered = filter_vertices_excluding(&vertices, &reference);
/// assert_eq!(filtered.len(), 1); // Only v2 remains
/// # Ok(())
/// # }
/// ```
pub fn filter_vertices_excluding<U, const D: usize>(
    vertices: &[Vertex<U, D>],
    reference: &[Vertex<U, D>],
) -> Vec<Vertex<U, D>>
where
    U: DataType,
{
    let mut filtered = Vec::with_capacity(vertices.len());

    for &v in vertices {
        if reference
            .iter()
            .any(|ref_v| coords_equal_exact(v.point().coords(), ref_v.point().coords()))
        {
            continue;
        }

        filtered.push(v);
    }

    filtered
}

/// Check if two coordinate arrays are exactly equal.
///
/// Uses `OrderedEq` which provides NaN-aware equality comparison.
/// For f64, this ensures consistent comparison including special values.
#[inline]
pub(crate) fn coords_equal_exact<const D: usize>(a: &[f64; D], b: &[f64; D]) -> bool {
    a.iter().zip(b.iter()).all(|(x, y)| x.ordered_eq(y))
}

/// Check if two coordinate arrays are within epsilon distance.
///
/// Returns true if Euclidean distance is strictly less than epsilon (distance < epsilon).
#[inline]
pub(crate) fn coords_within_epsilon<const D: usize>(
    a: &[f64; D],
    b: &[f64; D],
    epsilon: f64,
) -> bool {
    let dist_sq: f64 = a.iter().zip(b.iter()).fold(0.0, |acc, (x, y)| {
        let diff = *x - *y;
        diff.mul_add(diff, acc)
    });
    let epsilon_sq = epsilon * epsilon;

    #[cfg(debug_assertions)]
    if dist_sq.to_bits() == epsilon_sq.to_bits() {
        tracing::debug!(
            "[dedup_vertices_epsilon] distance equals epsilon; keeping point (strict < epsilon)"
        );
    }

    dist_sq < epsilon_sq
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::geometry::point::Point;
    use crate::try_vertices_from_points;
    use approx::assert_relative_eq;

    fn vertex(coords: [f64; 2]) -> Vertex<(), 2> {
        Vertex::try_new(coords).expect("finite vertex coordinates")
    }

    #[test]
    fn test_dedup_vertices_exact_comprehensive() {
        // Sub-test: Basic deduplication
        let v1 = vertex([0.0, 0.0]);
        let v2 = vertex([0.0, 0.0]);
        let v3 = vertex([1.0, 1.0]);
        let vertices = vec![v1, v2, v3];
        let unique = dedup_vertices_exact(&vertices);
        assert_eq!(unique.len(), 2, "Should remove exact duplicate");

        assert!(Point::<2>::try_new([f64::NAN, f64::NAN]).is_err());

        // Sub-test: Zero handling - +0.0 should equal -0.0
        let v1_pos_zero = vertex([0.0, 0.0]);
        let v2_neg_zero = vertex([-0.0, -0.0]);
        let v3_one = vertex([1.0, 1.0]);
        let vertices_zero = vec![v1_pos_zero, v2_neg_zero, v3_one];
        let unique_zero = dedup_vertices_exact(&vertices_zero);
        assert_eq!(
            unique_zero.len(),
            2,
            "+0.0 and -0.0 should be considered equal for deduplication"
        );
    }

    #[test]
    fn test_dedup_vertices_epsilon_basic() {
        // Near-duplicates should be filtered
        let v1 = vertex([0.0, 0.0]);
        let v2 = vertex([1e-11, 1e-11]);
        let v3 = vertex([1.0, 1.0]);

        let vertices = vec![v1, v2, v3];
        let unique = dedup_vertices_epsilon(&vertices, 1e-10);
        assert_eq!(
            unique.len(),
            2,
            "Near-duplicate within epsilon should be removed"
        );
    }

    #[test]
    fn test_dedup_vertices_epsilon_boundary() {
        // Test strict < epsilon semantics (distance = epsilon should NOT be filtered)
        let v1 = vertex([0.0, 0.0]);
        // Distance exactly epsilon (1e-10) in x direction
        let v2 = vertex([1e-10, 0.0]);
        // Distance slightly less than epsilon
        let v3 = vertex([0.99e-10, 0.0]);

        let vertices = vec![v1, v2, v3];
        let unique = dedup_vertices_epsilon(&vertices, 1e-10);
        // v1 kept, v3 filtered (< epsilon), v2 kept (= epsilon, not < epsilon)
        assert_eq!(
            unique.len(),
            2,
            "Distance exactly equal to epsilon should NOT be filtered (strict < semantics)"
        );
    }

    #[test]
    fn test_coords_within_epsilon_exact_boundary_keeps_point() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];

        assert!(!coords_within_epsilon(&a, &b, 1.0));
    }

    #[test]
    fn test_dedup_vertices_epsilon_negative_epsilon_returns_input_unchanged() {
        let vertices: Vec<Vertex<(), 2>> = try_vertices_from_points(&[
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
        ])
        .expect("finite point coordinates");

        let unique = dedup_vertices_epsilon(&vertices, -1.0);

        assert_eq!(unique.len(), vertices.len());
        assert_eq!(
            unique
                .iter()
                .map(<&Vertex<_, _> as Into<[f64; 2]>>::into)
                .collect::<Vec<_>>(),
            vertices
                .iter()
                .map(<&Vertex<_, _> as Into<[f64; 2]>>::into)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_try_dedup_vertices_epsilon_negative_epsilon_returns_error() {
        let vertices: Vec<Vertex<(), 2>> = try_vertices_from_points(&[
            Point::try_new([0.0, 0.0]).expect("finite point coordinates")
        ])
        .expect("finite point coordinates");

        let err = try_dedup_vertices_epsilon(&vertices, -1.0).unwrap_err();

        assert_eq!(err, DeduplicationError::NegativeEpsilon);
    }

    #[test]
    fn test_dedup_vertices_epsilon_non_finite_epsilon_returns_input_unchanged() {
        let vertices: Vec<Vertex<(), 2>> = try_vertices_from_points(&[
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
        ])
        .expect("finite point coordinates");

        for epsilon in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let unique = dedup_vertices_epsilon(&vertices, epsilon);

            assert_eq!(unique.len(), vertices.len());
            assert_eq!(
                unique
                    .iter()
                    .map(<&Vertex<_, _> as Into<[f64; 2]>>::into)
                    .collect::<Vec<_>>(),
                vertices
                    .iter()
                    .map(<&Vertex<_, _> as Into<[f64; 2]>>::into)
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_try_dedup_vertices_epsilon_non_finite_epsilon_returns_error() {
        let vertices: Vec<Vertex<(), 2>> = try_vertices_from_points(&[
            Point::try_new([0.0, 0.0]).expect("finite point coordinates")
        ])
        .expect("finite point coordinates");

        for epsilon in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = try_dedup_vertices_epsilon(&vertices, epsilon).unwrap_err();

            assert_eq!(err, DeduplicationError::NonFiniteEpsilon);
        }
    }

    #[test]
    fn test_dedup_vertices_epsilon_preserves_first_occurrence() {
        // Verify that first occurrence is kept, later duplicates removed
        let points = [
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1e-11, 1e-11]).expect("finite point coordinates"), // Near-duplicate of first
            Point::try_new([1.0, 1.0]).expect("finite point coordinates"),
            Point::try_new([1.0 + 1e-11, 1.0 + 1e-11]).expect("finite point coordinates"), // Near-duplicate of third
        ];
        let vertices: Vec<Vertex<(), 2>> =
            try_vertices_from_points(&points).expect("finite point coordinates");

        let unique = dedup_vertices_epsilon(&vertices, 1e-10);
        assert_eq!(unique.len(), 2, "Should keep first of each cluster");

        // Verify first occurrences are kept
        let unique_coords: Vec<_> = unique
            .iter()
            .map(<&Vertex<_, _> as Into<[f64; 2]>>::into)
            .collect();
        assert_relative_eq!(unique_coords[0][0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(unique_coords[0][1], 0.0, epsilon = 1e-12);
        assert_relative_eq!(unique_coords[1][0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(unique_coords[1][1], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_filter_vertices_excluding_comprehensive() {
        // Sub-test: Basic exclusion
        let v1 = vertex([0.0, 0.0]);
        let v2 = vertex([1.0, 1.0]);
        let v3 = vertex([2.0, 2.0]);
        let reference_basic = vec![v1];
        let vertices_basic = vec![v1, v2, v3];
        let filtered_basic = filter_vertices_excluding(&vertices_basic, &reference_basic);
        assert_eq!(
            filtered_basic.len(),
            2,
            "Should exclude vertex matching reference"
        );

        assert!(Point::<2>::try_new([f64::NAN, f64::NAN]).is_err());

        // Sub-test: Zero exclusion - +0.0 reference should match -0.0 vertices
        let v_pos_zero = vertex([0.0, 0.0]);
        let reference_zero = vec![v_pos_zero];
        let vertices_with_neg_zero: Vec<Vertex<(), 2>> = try_vertices_from_points(&[
            Point::try_new([-0.0, -0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 1.0]).expect("finite point coordinates"),
        ])
        .expect("finite point coordinates");
        let filtered_zero = filter_vertices_excluding(&vertices_with_neg_zero, &reference_zero);
        assert_eq!(
            filtered_zero.len(),
            1,
            "+0.0 reference should exclude -0.0 vertex"
        );

        // Sub-test: Multiple reference vertices
        let points = [
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 1.0]).expect("finite point coordinates"),
            Point::try_new([2.0, 2.0]).expect("finite point coordinates"),
            Point::try_new([3.0, 3.0]).expect("finite point coordinates"),
        ];
        let vertices: Vec<Vertex<(), 2>> =
            try_vertices_from_points(&points).expect("finite point coordinates");

        let reference = vec![vertices[0], vertices[2]]; // Exclude first and third
        let filtered = filter_vertices_excluding(&vertices, &reference);

        assert_eq!(filtered.len(), 2, "Should exclude both reference vertices");

        // Verify remaining vertices are second and fourth
        let filtered_coords: Vec<_> = filtered
            .iter()
            .map(<&Vertex<_, _> as Into<[f64; 2]>>::into)
            .collect();
        assert_relative_eq!(filtered_coords[0][0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(filtered_coords[0][1], 1.0, epsilon = 1e-12);
        assert_relative_eq!(filtered_coords[1][0], 3.0, epsilon = 1e-12);
        assert_relative_eq!(filtered_coords[1][1], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_filter_vertices_excluding_empty_reference() {
        let vertices: Vec<Vertex<(), 1>> = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([2.0]).unwrap(),
        ];
        let reference: Vec<Vertex<(), 1>> = vec![];
        let filtered = filter_vertices_excluding(&vertices, &reference);
        assert_eq!(filtered.len(), vertices.len());
    }
}
