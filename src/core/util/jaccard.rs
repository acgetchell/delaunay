//! Set similarity utilities (Jaccard index/distance) and canonical extraction helpers.

#![forbid(unsafe_code)]

use crate::core::facet::{FacetError, FacetView};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation_data_structure::Tds;
use crate::geometry::algorithms::convex_hull::ConvexHull;
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::CoordinateScalar;
use std::collections::HashSet;
use thiserror::Error;

/// Errors that can occur during Jaccard similarity computation.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::util::JaccardComputationError;
///
/// let err = JaccardComputationError::SetSizeTooLarge {
///     intersection: 1,
///     union: 2,
/// };
/// assert!(matches!(err, JaccardComputationError::SetSizeTooLarge { .. }));
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum JaccardComputationError {
    /// Set sizes too large for safe f64 conversion.
    #[error(
        "Set sizes exceed safe f64 conversion limit (2^53): intersection={intersection}, union={union}"
    )]
    SetSizeTooLarge {
        /// The intersection size
        intersection: usize,
        /// The union size
        union: usize,
    },
}

// =============================================================================
// SET SIMILARITY UTILITIES
// =============================================================================

/// Compute intersection and union sizes for two sets.
///
/// Returns `(intersection, union)` where:
/// - `intersection`: number of elements in both sets
/// - `union`: number of elements in at least one set
///
/// This is an internal helper used by `jaccard_index` and `format_jaccard_report`.
/// The function iterates over the smaller set for optimal performance.
fn compute_set_metrics<T, S>(
    a: &std::collections::HashSet<T, S>,
    b: &std::collections::HashSet<T, S>,
) -> (usize, usize)
where
    T: Eq + std::hash::Hash,
    S: std::hash::BuildHasher,
{
    // Optimize: iterate over smaller set for intersection count
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let intersection = small.iter().filter(|x| large.contains(x)).count();
    let union = a.len() + b.len() - intersection;
    (intersection, union)
}

/// Jaccard index (similarity) between two sets: |A ∩ B| / |A ∪ B|.
///
/// Returns 1.0 when both sets are empty by convention.
///
/// References
/// - Jaccard, P. (1901). Étude comparative de la distribution florale.
///   Bulletin de la Société Vaudoise des Sciences Naturelles.
/// - Tanimoto, T. T. (1958). An elementary mathematical theory of classification and
///   prediction. IBM Report (often cited for the Tanimoto coefficient).
///
/// # Examples
/// ```
/// use std::collections::HashSet;
/// use delaunay::core::util::jaccard_index;
///
/// // Identical sets => similarity 1.0
/// let a: HashSet<_> = [1, 2, 3].into_iter().collect();
/// assert_eq!(jaccard_index(&a, &a).unwrap(), 1.0);
///
/// // Partial overlap: {1,2,3} vs {3,4} => |∩|=1, |∪|=4 => 0.25
/// let b: HashSet<_> = [3, 4].into_iter().collect();
/// assert!((jaccard_index(&a, &b).unwrap() - 0.25).abs() < 1e-12);
///
/// // Empty vs empty => 1.0 by convention
/// let empty: HashSet<i32> = HashSet::new();
/// assert_eq!(jaccard_index(&empty, &empty).unwrap(), 1.0);
/// ```
///
/// # Errors
///
/// Returns `JaccardComputationError::SetSizeTooLarge` if set sizes exceed 2^53,
/// which would cause precision loss in f64 conversion.
pub fn jaccard_index<T, S>(
    a: &std::collections::HashSet<T, S>,
    b: &std::collections::HashSet<T, S>,
) -> Result<f64, JaccardComputationError>
where
    T: Eq + std::hash::Hash,
    S: std::hash::BuildHasher,
{
    // f64 can exactly represent integers up to 2^53
    // Use u128 for portability to 32-bit platforms where usize << 53 would overflow
    const MAX_SAFE_INT_U128: u128 = 1u128 << 53;

    if a.is_empty() && b.is_empty() {
        return Ok(1.0);
    }

    let (intersection, union) = compute_set_metrics(a, b);

    // Check for safe conversion before casting
    if (intersection as u128) > MAX_SAFE_INT_U128 || (union as u128) > MAX_SAFE_INT_U128 {
        return Err(JaccardComputationError::SetSizeTooLarge {
            intersection,
            union,
        });
    }

    // Safe to cast: we've verified values are within safe range
    #[expect(
        clippy::cast_precision_loss,
        reason = "Safe: intersection/union are bounded to 2^53 before casting to f64"
    )]
    let inter_f64 = intersection as f64;
    #[expect(
        clippy::cast_precision_loss,
        reason = "Safe: intersection/union are bounded to 2^53 before casting to f64"
    )]
    let union_f64 = union as f64;

    Ok(inter_f64 / union_f64)
}

/// Jaccard distance between two sets: 1.0 - Jaccard index.
///
/// # Examples
/// ```
/// use std::collections::HashSet;
/// use delaunay::core::util::{jaccard_index, jaccard_distance};
///
/// let a: HashSet<_> = [1, 2, 3].into_iter().collect();
/// let b: HashSet<_> = [3, 4].into_iter().collect();
///
/// // Distance is 0.0 for identical sets
/// assert_eq!(jaccard_distance(&a, &a).unwrap(), 0.0);
///
/// // Index + distance = 1.0
/// let sum = jaccard_index(&a, &b).unwrap() + jaccard_distance(&a, &b).unwrap();
/// assert!((sum - 1.0).abs() < 1e-12);
/// ```
///
/// # Errors
///
/// Returns `JaccardComputationError::SetSizeTooLarge` if set sizes exceed 2^53,
/// which would cause precision loss in f64 conversion.
#[inline]
pub fn jaccard_distance<T, S>(
    a: &std::collections::HashSet<T, S>,
    b: &std::collections::HashSet<T, S>,
) -> Result<f64, JaccardComputationError>
where
    T: Eq + std::hash::Hash,
    S: std::hash::BuildHasher,
{
    Ok(1.0 - jaccard_index(a, b)?)
}

// =============================================================================
// CANONICAL SET EXTRACTION FOR JACCARD SIMILARITY TESTING
// =============================================================================

/// Extract vertex coordinate set from a triangulation for Jaccard similarity comparison.
///
/// This function creates a canonical set of vertex coordinates from a triangulation,
/// useful for comparing triangulations before/after operations like serialization.
///
/// # Arguments
///
/// * `tds` - The triangulation data structure to extract vertex coordinates from
///
/// # Returns
///
/// A `HashSet` containing all unique vertex coordinates as `Point<T, D>`
///
/// # Examples
///
/// ```
/// use delaunay::prelude::*;
/// use delaunay::core::util::extract_vertex_coordinate_set;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt = DelaunayTriangulation::new(&vertices).unwrap();
/// let tds = dt.tds();
///
/// let coord_set = extract_vertex_coordinate_set(tds);
/// assert_eq!(coord_set.len(), 4);
/// ```
#[must_use]
pub fn extract_vertex_coordinate_set<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
) -> HashSet<Point<T, D>>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    tds.vertices().map(|(_, vertex)| *vertex.point()).collect()
}

/// Canonicalize an edge by ordering vertex UUIDs.
///
/// Returns the edge with UUIDs in ascending order for consistent comparison.
const fn canonical_edge(u: u128, v: u128) -> (u128, u128) {
    if u <= v { (u, v) } else { (v, u) }
}

/// Extract canonical edge set from a triangulation.
///
/// This function creates a set of all edges in the triangulation, represented as
/// pairs of vertex UUIDs in canonical (sorted) order. This is useful for comparing
/// triangulation topology across different storage backends or algorithms.
///
/// # Arguments
///
/// * `tds` - The triangulation data structure to extract edges from
///
/// # Returns
///
/// A `Result` containing a `HashSet` of canonicalized `(u128, u128)` UUID pairs,
/// or a `FacetError` if vertex keys cannot be resolved
///
/// # Errors
///
/// Returns `FacetError::VertexKeyNotFoundInTriangulation` if a cell references
/// a vertex key that doesn't exist in the TDS
///
/// # Examples
///
/// ```
/// use delaunay::prelude::*;
/// use delaunay::core::util::extract_edge_set;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt = DelaunayTriangulation::new(&vertices).unwrap();
/// let tds = dt.tds();
///
/// let edge_set = extract_edge_set(tds).unwrap();
/// // A tetrahedron has 6 edges
/// assert_eq!(edge_set.len(), 6);
/// ```
pub fn extract_edge_set<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
) -> Result<HashSet<(u128, u128)>, FacetError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let mut edges = HashSet::new();

    for (_, cell) in tds.cells() {
        let vertex_keys = cell.vertices();
        // Generate all pairs of vertices (edges)
        for i in 0..vertex_keys.len() {
            for j in (i + 1)..vertex_keys.len() {
                let v_i = tds.get_vertex_by_key(vertex_keys[i]).ok_or(
                    FacetError::VertexKeyNotFoundInTriangulation {
                        key: vertex_keys[i],
                    },
                )?;
                let v_j = tds.get_vertex_by_key(vertex_keys[j]).ok_or(
                    FacetError::VertexKeyNotFoundInTriangulation {
                        key: vertex_keys[j],
                    },
                )?;

                let uuid_i = v_i.uuid().as_u128();
                let uuid_j = v_j.uuid().as_u128();
                edges.insert(canonical_edge(uuid_i, uuid_j));
            }
        }
    }

    Ok(edges)
}

/// Extract canonical facet identifier set from a triangulation.
///
/// This function creates a set of deterministic 64-bit identifiers for all boundary facets
/// in the triangulation using the existing `FacetView::key()` method.
///
/// # Arguments
///
/// * `tds` - The triangulation data structure to extract facet identifiers from
///
/// # Returns
///
/// A `Result` containing a `HashSet` of facet identifiers, or a `FacetError`
///
/// # Errors
///
/// Returns an error if facet keys cannot be computed or boundary facets cannot be retrieved
///
/// # Examples
///
/// ```
/// use delaunay::prelude::*;
/// use delaunay::core::util::extract_facet_identifier_set;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt = DelaunayTriangulation::new(&vertices).unwrap();
/// let tds = dt.tds();
///
/// let facet_set = extract_facet_identifier_set(tds).unwrap();
/// // A tetrahedron has 4 facets
/// assert_eq!(facet_set.len(), 4);
/// ```
pub fn extract_facet_identifier_set<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
) -> Result<HashSet<u64>, FacetError>
where
    T: CoordinateScalar + std::ops::AddAssign<T> + std::ops::SubAssign<T> + std::iter::Sum,
    U: DataType,
    V: DataType,
{
    use crate::core::traits::boundary_analysis::BoundaryAnalysis;

    let mut facet_ids = HashSet::new();

    // boundary_facets() returns Result<impl Iterator, TriangulationValidationError>
    // Wrap the underlying error for better diagnostics
    let boundary_facets =
        tds.boundary_facets()
            .map_err(|e| FacetError::BoundaryFacetRetrievalFailed {
                source: std::sync::Arc::new(e),
            })?;

    for facet_view in boundary_facets {
        // Use the existing FacetView::key() method
        let facet_id = facet_view.key()?;
        facet_ids.insert(facet_id);
    }

    Ok(facet_ids)
}

/// Extract hull facet identifier set from a convex hull.
///
/// This function creates a set of deterministic 64-bit identifiers for all facets
/// in a convex hull using the existing `FacetView::key()` method.
///
/// # Arguments
///
/// * `hull` - The convex hull to extract facet identifiers from
/// * `tri` - The triangulation used to create the hull
///
/// # Returns
///
/// A `Result` containing a `HashSet` of facet identifiers, or a `FacetError`
///
/// # Errors
///
/// Returns `FacetError` if facet views cannot be created or facet keys cannot be computed
///
/// # Examples
///
/// ```
/// use delaunay::core::util::extract_hull_facet_set;
/// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
/// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
/// use delaunay::vertex;
///
/// let vertices: Vec<_> = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt: DelaunayTriangulation<_, (), (), 3> =
///     DelaunayTriangulation::new(&vertices).unwrap();
/// let hull = ConvexHull::from_triangulation(dt.as_triangulation()).unwrap();
///
/// let facet_set = extract_hull_facet_set(&hull, dt.as_triangulation()).unwrap();
/// assert_eq!(facet_set.len(), 4);
/// ```
pub fn extract_hull_facet_set<K, U, V, const D: usize>(
    hull: &ConvexHull<K, U, V, D>,
    tri: &crate::core::triangulation::Triangulation<K, U, V, D>,
) -> Result<HashSet<u64>, FacetError>
where
    K: crate::geometry::kernel::Kernel<D>,
    K::Scalar: CoordinateScalar + std::ops::AddAssign + std::ops::SubAssign + std::iter::Sum,
    U: DataType,
    V: DataType,
{
    let tds = &tri.tds;
    let mut facet_ids = HashSet::new();

    for facet_handle in hull.facets() {
        // Create FacetView using cell_key() and facet_index() methods from FacetHandle
        let facet_view = FacetView::new(tds, facet_handle.cell_key(), facet_handle.facet_index())?;
        // Use the existing FacetView::key() method
        let facet_id = facet_view.key()?;
        facet_ids.insert(facet_id);
    }

    Ok(facet_ids)
}

// =============================================================================
// JACCARD TESTING UTILITIES
// =============================================================================

/// Format a detailed Jaccard similarity report for test diagnostics.
///
/// This function produces a human-readable report showing:
/// - Set sizes
/// - Intersection and union sizes
/// - Jaccard index
/// - Sample elements from the symmetric difference (unique to each set)
///
/// # Arguments
///
/// * `a` - First set to compare
/// * `b` - Second set to compare
/// * `label_a` - Descriptive label for set A
/// * `label_b` - Descriptive label for set B
///
/// # Returns
///
/// A formatted string with detailed comparison metrics
///
/// # Examples
///
/// ```
/// use std::collections::HashSet;
/// use delaunay::core::util::format_jaccard_report;
///
/// let a: HashSet<i32> = [1, 2, 3, 4].into_iter().collect();
/// let b: HashSet<i32> = [3, 4, 5, 6].into_iter().collect();
///
/// let report = format_jaccard_report(&a, &b, "Set A", "Set B").unwrap();
/// assert!(report.contains("Jaccard Index:"));
/// ```
///
/// # Errors
///
/// Returns `JaccardComputationError::SetSizeTooLarge` if the set sizes exceed
/// the safe range for f64 conversion (2^53).
pub fn format_jaccard_report<T, S>(
    a: &HashSet<T, S>,
    b: &HashSet<T, S>,
    label_a: &str,
    label_b: &str,
) -> Result<String, JaccardComputationError>
where
    T: Eq + std::hash::Hash + std::fmt::Debug,
    S: std::hash::BuildHasher,
{
    // f64 can exactly represent integers up to 2^53
    // Use u128 for portability to 32-bit platforms where usize << 53 would overflow
    const MAX_SAFE_INT_U128: u128 = 1u128 << 53;

    let size_a = a.len();
    let size_b = b.len();

    // Compute intersection and union using shared helper
    let (intersection, union) = compute_set_metrics(a, b);

    // Compute Jaccard index using safe conversion
    let jaccard = if union == 0 {
        1.0
    } else {
        if (intersection as u128) > MAX_SAFE_INT_U128 || (union as u128) > MAX_SAFE_INT_U128 {
            return Err(JaccardComputationError::SetSizeTooLarge {
                intersection,
                union,
            });
        }

        // Safe to cast: we've verified values are within safe range
        #[expect(
            clippy::cast_precision_loss,
            reason = "Safe: intersection/union are bounded to 2^53 before casting to f64"
        )]
        let inter_f64 = intersection as f64;
        #[expect(
            clippy::cast_precision_loss,
            reason = "Safe: intersection/union are bounded to 2^53 before casting to f64"
        )]
        let union_f64 = union as f64;

        inter_f64 / union_f64
    };

    // Sample symmetric difference (elements unique to each set)
    let only_in_a: Vec<_> = a.iter().filter(|x| !b.contains(x)).take(5).collect();
    let only_in_b: Vec<_> = b.iter().filter(|x| !a.contains(x)).take(5).collect();

    Ok(format!(
        "\n╭─ Jaccard Similarity Report ─────────────────────────────\n\
         │ {label_a}: {size_a} elements\n\
         │ {label_b}: {size_b} elements\n\
         │ Intersection: {intersection}\n\
         │ Union: {union}\n\
         │ Jaccard Index: {jaccard:.6}\n\
         ├─ Symmetric Difference (sample) ────────────────────────\n\
         │ Only in {label_a} (first 5): {only_in_a:?}\n\
         │ Only in {label_b} (first 5): {only_in_b:?}\n\
         ╰─────────────────────────────────────────────────────────\n"
    ))
}

/// Assert that the Jaccard index between two sets meets or exceeds a threshold.
///
/// This macro computes the Jaccard similarity between two `HashSet`s and asserts
/// that the index is greater than or equal to the specified threshold. On failure,
/// it provides detailed diagnostics including set sizes, intersection/union counts,
/// and samples of the symmetric difference.
///
/// # Arguments
///
/// * `$a` - First set expression (must evaluate to `&HashSet<T>`)
/// * `$b` - Second set expression (must evaluate to `&HashSet<T>`)
/// * `$threshold` - Minimum acceptable Jaccard index (f64)
/// * `$($label:tt)*` - Optional format string and arguments for context message
///
/// # Examples
///
/// ```
/// use std::collections::HashSet;
/// use delaunay::assert_jaccard_gte;
///
/// let set_a: HashSet<i32> = [1, 2, 3].into_iter().collect();
/// let set_b: HashSet<i32> = [2, 3, 4].into_iter().collect();
///
/// // This will pass (Jaccard = 0.5, which is ≥ 0.4)
/// assert_jaccard_gte!(&set_a, &set_b, 0.4, "comparing set_a and set_b");
/// ```
#[macro_export]
macro_rules! assert_jaccard_gte {
    // 3-arg form (no label) - delegate to 4-arg form with default label
    ($a:expr, $b:expr, $threshold:expr $(,)?) => {
        $crate::assert_jaccard_gte!($a, $b, $threshold, "Jaccard index assertion")
    };
    // 4-arg form (with label)
    ($a:expr, $b:expr, $threshold:expr, $($label:tt)*) => {{
        let a_ref = $a;
        let b_ref = $b;
        let threshold = $threshold;

        // Use jaccard_index function for safety and consistency
        let jaccard_index = $crate::core::util::jaccard_index(a_ref, b_ref)
            .expect("Jaccard computation should not overflow for reasonable test sets");

        if jaccard_index < threshold {
            let report = $crate::core::util::format_jaccard_report(
                a_ref,
                b_ref,
                "Set A",
                "Set B"
            )
            .expect("Failed to format Jaccard report - set sizes too large");
            panic!(
                "Jaccard assertion failed: {}\n\
                 Expected: Jaccard index ≥ {:.6}\n\
                 Actual: {:.6}\n\
                 {}",
                format!($($label)*),
                threshold,
                jaccard_index,
                report
            );
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::core::triangulation_data_structure::{Tds, VertexKey};
    use crate::geometry::traits::coordinate::Coordinate;
    use crate::vertex;
    use approx::assert_relative_eq;
    use slotmap::KeyData;

    // Unit tests for jaccard_index and jaccard_distance across 2D–5D using a macro
    macro_rules! gen_jaccard_set_tests {
        ($name:ident, $dim:literal) => {
            #[test]
            fn $name() {
                use std::collections::HashSet;
                let empty: HashSet<[i32; $dim]> = HashSet::new();
                assert_relative_eq!(jaccard_index(&empty, &empty).unwrap(), 1.0, epsilon = 1e-12);
                assert_relative_eq!(
                    jaccard_distance(&empty, &empty).unwrap(),
                    0.0,
                    epsilon = 1e-12
                );

                let a: HashSet<[i32; $dim]> = HashSet::from([[1; $dim], [2; $dim], [3; $dim]]);
                let b: HashSet<[i32; $dim]> = HashSet::from([[3; $dim], [4; $dim]]);

                // Identical sets
                assert_relative_eq!(jaccard_index(&a, &a).unwrap(), 1.0, epsilon = 1e-12);
                assert_relative_eq!(jaccard_distance(&a, &a).unwrap(), 0.0, epsilon = 1e-12);

                // Partial overlap: |∩|=1, |∪|=4
                assert_relative_eq!(jaccard_index(&a, &b).unwrap(), 0.25, epsilon = 1e-12);
                assert_relative_eq!(jaccard_distance(&a, &b).unwrap(), 0.75, epsilon = 1e-12);

                // Empty vs non-empty
                assert_relative_eq!(jaccard_index(&a, &empty).unwrap(), 0.0, epsilon = 1e-12);
                assert_relative_eq!(jaccard_distance(&a, &empty).unwrap(), 1.0, epsilon = 1e-12);
            }
        };
    }

    gen_jaccard_set_tests!(jaccard_basic_properties_2d, 2);
    gen_jaccard_set_tests!(jaccard_basic_properties_3d, 3);
    gen_jaccard_set_tests!(jaccard_basic_properties_4d, 4);
    gen_jaccard_set_tests!(jaccard_basic_properties_5d, 5);

    #[test]
    fn test_canonical_edge() {
        // Test that edges are canonicalized (smaller UUID first)
        assert_eq!(canonical_edge(1, 2), (1, 2));
        assert_eq!(canonical_edge(2, 1), (1, 2));
        assert_eq!(canonical_edge(9, 2), (2, 9));
        assert_eq!(canonical_edge(100, 50), (50, 100));

        // Test with identical UUIDs
        assert_eq!(canonical_edge(5, 5), (5, 5));
    }

    #[test]
    fn test_set_extraction_utilities_comprehensive() {
        // Create a simple tetrahedron for all extraction tests
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices).unwrap();
        let tds = &dt.as_triangulation().tds;

        // Sub-test: Vertex coordinate extraction
        let coord_set = extract_vertex_coordinate_set(tds);
        assert_eq!(coord_set.len(), 4, "Should have 4 unique coordinates");
        assert!(coord_set.contains(&Point::new([0.0, 0.0, 0.0])));
        assert!(coord_set.contains(&Point::new([1.0, 0.0, 0.0])));
        assert!(coord_set.contains(&Point::new([0.0, 1.0, 0.0])));
        assert!(coord_set.contains(&Point::new([0.0, 0.0, 1.0])));

        // Sub-test: Edge extraction - tetrahedron has 6 edges (binomial(4,2))
        let edge_set = extract_edge_set(tds).unwrap();
        assert_eq!(edge_set.len(), 6, "Tetrahedron should have 6 edges");

        // Sub-test: Facet identifier extraction - tetrahedron has 4 facets
        let facet_set = extract_facet_identifier_set(tds).unwrap();
        assert_eq!(facet_set.len(), 4, "Tetrahedron should have 4 facets");

        // Sub-test: Hull facet extraction
        let dt_hull =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices).unwrap();
        let tri = dt_hull.as_triangulation();
        let hull = ConvexHull::from_triangulation(tri).unwrap();
        let hull_facet_set = extract_hull_facet_set(&hull, tri).unwrap();
        assert_eq!(hull_facet_set.len(), 4, "Hull should have 4 facets");
    }

    #[test]
    fn test_extract_edge_set_errors_on_missing_vertex_key() {
        use crate::core::facet::FacetError;

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut dt =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices).unwrap();

        let cell_key = dt.as_triangulation().tds.cell_keys().next().unwrap();
        let invalid_vkey = VertexKey::from(KeyData::from_ffi(u64::MAX));
        dt.tri
            .tds
            .get_cell_by_key_mut(cell_key)
            .unwrap()
            .push_vertex_key(invalid_vkey);

        let err = extract_edge_set(&dt.as_triangulation().tds).unwrap_err();
        assert!(matches!(
            err,
            FacetError::VertexKeyNotFoundInTriangulation { key } if key == invalid_vkey
        ));
    }

    #[test]
    fn test_extract_facet_identifier_set_errors_on_boundary_facet_retrieval_failure() {
        use crate::core::facet::FacetError;

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut dt =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices).unwrap();

        let cell_key = dt.as_triangulation().tds.cell_keys().next().unwrap();
        let invalid_vkey = VertexKey::from(KeyData::from_ffi(u64::MAX));
        dt.tri
            .tds
            .get_cell_by_key_mut(cell_key)
            .unwrap()
            .push_vertex_key(invalid_vkey);

        let err = extract_facet_identifier_set(&dt.as_triangulation().tds).unwrap_err();
        assert!(matches!(
            err,
            FacetError::BoundaryFacetRetrievalFailed { .. }
        ));
    }

    #[test]
    fn test_format_jaccard_report_includes_metrics_and_handles_empty_sets() {
        use std::collections::HashSet;

        let empty: HashSet<i32> = HashSet::new();
        let report = format_jaccard_report(&empty, &empty, "A", "B").unwrap();
        assert!(report.contains("A: 0 elements"));
        assert!(report.contains("B: 0 elements"));
        assert!(report.contains("Intersection: 0"));
        assert!(report.contains("Union: 0"));
        assert!(report.contains("Jaccard Index: 1"));

        let a: HashSet<_> = [1, 2, 3].into_iter().collect();
        let b: HashSet<_> = [3, 4].into_iter().collect();
        let report = format_jaccard_report(&a, &b, "A", "B").unwrap();
        assert!(report.contains("Intersection: 1"));
        assert!(report.contains("Union: 4"));
    }

    #[test]
    fn test_extract_hull_facet_set_consistent_with_boundary_facets() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices).unwrap();
        let tri = dt.as_triangulation();
        let hull = ConvexHull::from_triangulation(tri).unwrap();

        let hull_facet_set = extract_hull_facet_set(&hull, tri).unwrap();
        let boundary_set = extract_facet_identifier_set(&tri.tds).unwrap();

        assert_eq!(hull_facet_set, boundary_set);
    }

    #[test]
    fn test_extract_edge_set_empty_tds_is_empty() {
        let tds: Tds<f64, (), (), 3> = Tds::empty();
        let edges = extract_edge_set(&tds).unwrap();
        assert!(edges.is_empty());
    }
}
