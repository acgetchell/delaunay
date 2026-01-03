//! General helper utilities

use std::collections::HashSet;
use std::ops::{AddAssign, SubAssign};
use thiserror::Error;
use uuid::Uuid;

use crate::core::cell::CellValidationError;
use crate::core::collections::ViolationBuffer;
use crate::core::facet::{FacetError, FacetView};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation_data_structure::{CellKey, Tds, TdsValidationError, VertexKey};
use crate::core::vertex::Vertex;
use crate::geometry::algorithms::convex_hull::ConvexHull;
use crate::geometry::point::Point;
use crate::geometry::predicates::InSphere;
use crate::geometry::robust_predicates::robust_insphere;
use crate::geometry::traits::coordinate::{CoordinateConversionError, CoordinateScalar};
use num_traits::cast::NumCast;
use smallvec::SmallVec;

// =============================================================================
// TYPES
// =============================================================================

/// Errors that can occur during UUID validation.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum UuidValidationError {
    /// The UUID is nil (all zeros), which is not allowed.
    #[error("UUID is nil (all zeros) which is not allowed")]
    NilUuid,
    /// The UUID is not version 4.
    #[error("UUID is not version 4: expected version 4, found version {found}")]
    InvalidVersion {
        /// The version number that was found.
        found: usize,
    },
}

/// Errors that can occur during Jaccard similarity computation.
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

/// Errors that can occur during Delaunay property validation.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum DelaunayValidationError {
    /// A cell violates the Delaunay property (has an external vertex inside its circumsphere).
    #[error("Cell violates Delaunay property: cell contains vertex that is inside circumsphere")]
    DelaunayViolation {
        /// The key of the cell that violates the Delaunay property
        cell_key: CellKey,
    },
    /// TDS data structure corruption or other structural issues detected during validation.
    #[error("TDS corruption: {source}")]
    TriangulationState {
        /// The underlying TDS validation error (TDS-level invariants).
        #[source]
        source: TdsValidationError,
    },
    /// Invalid cell structure detected during validation.
    #[error("Invalid cell {cell_key:?}: {source}")]
    InvalidCell {
        /// The key of the invalid cell.
        cell_key: CellKey,
        /// The underlying cell error.
        #[source]
        source: CellValidationError,
    },
    /// Numeric predicate failure during Delaunay validation.
    #[error(
        "Numeric predicate failure while validating Delaunay property for cell {cell_key:?}, vertex {vertex_key:?}: {source}"
    )]
    NumericPredicateError {
        /// The key of the cell whose circumsphere was being tested.
        cell_key: CellKey,
        /// The key of the vertex being classified relative to the circumsphere.
        vertex_key: VertexKey,
        /// Underlying robust predicate error (e.g., conversion failure).
        #[source]
        source: CoordinateConversionError,
    },
}

/// Validates that a UUID is not nil and is version 4.
///
/// This function performs comprehensive UUID validation to ensure that UUIDs
/// used throughout the system meet our requirements:
/// - Must not be nil (all zeros)
/// - Must be version 4 (randomly generated)
///
/// # Arguments
///
/// * `uuid` - The UUID to validate
///
/// # Returns
///
/// Returns `Ok(())` if the UUID is valid, or a `UuidValidationError` if invalid.
///
/// # Errors
///
/// Returns `UuidValidationError::NilUuid` if the UUID is nil,
/// or `UuidValidationError::InvalidVersion` if the UUID is not version 4.
///
/// # Examples
///
/// ```
/// use delaunay::core::util::{make_uuid, validate_uuid};
/// use uuid::Uuid;
///
/// // Valid UUID (version 4)
/// let valid_uuid = make_uuid();
/// assert!(validate_uuid(&valid_uuid).is_ok());
///
/// // Invalid UUID (nil)
/// let nil_uuid = Uuid::nil();
/// assert!(validate_uuid(&nil_uuid).is_err());
/// ```
pub const fn validate_uuid(uuid: &Uuid) -> Result<(), UuidValidationError> {
    // Check if UUID is nil
    if uuid.is_nil() {
        return Err(UuidValidationError::NilUuid);
    }

    // Check if UUID is version 4
    let version = uuid.get_version_num();
    if version != 4 {
        return Err(UuidValidationError::InvalidVersion { found: version });
    }

    Ok(())
}

/// The function `make_uuid` generates a version 4 [Uuid].
///
/// # Returns
///
/// a randomly generated [Uuid] (Universally Unique Identifier) using the
/// `new_v4` method from the [Uuid] struct.
///
/// # Example
///
/// ```
/// use delaunay::core::util::make_uuid;
/// let uuid = make_uuid();
/// assert_eq!(uuid.get_version_num(), 4);
/// ```
#[must_use]
pub fn make_uuid() -> Uuid {
    Uuid::new_v4()
}

// =============================================================================
// VERTEX DEDUPLICATION
// =============================================================================

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
/// * `vertices` - Vector of vertices to deduplicate
///
/// # Returns
///
/// A new vector containing only unique vertices (by coordinates). The first
/// occurrence of each unique coordinate is kept.
///
/// # Examples
///
/// ```
/// use delaunay::core::util::dedup_vertices_exact;
/// use delaunay::core::vertex::Vertex;
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::traits::coordinate::Coordinate;
///
/// let v1: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([0.0, 0.0])])
///     .into_iter().next().unwrap();
/// let v2: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([0.0, 0.0])]) // Duplicate
///     .into_iter().next().unwrap();
/// let v3: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([1.0, 1.0])])
///     .into_iter().next().unwrap();
///
/// let vertices = vec![v1, v2, v3];
/// let unique = dedup_vertices_exact(vertices);
/// assert_eq!(unique.len(), 2); // Only v1 and v3
/// ```
#[must_use]
pub fn dedup_vertices_exact<T, U, const D: usize>(
    vertices: Vec<Vertex<T, U, D>>,
) -> Vec<Vertex<T, U, D>>
where
    T: CoordinateScalar,
    U: DataType,
{
    let mut unique: Vec<Vertex<T, U, D>> = Vec::with_capacity(vertices.len());

    'outer: for v in vertices {
        let vc: [T; D] = (&v).into();

        for u in &unique {
            let uc: [T; D] = u.into();

            // Exact floating-point equality (NaN-aware, treats +0.0 == -0.0)
            if coords_equal_exact(&vc, &uc) {
                continue 'outer; // Skip exact duplicate
            }
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
/// * `vertices` - Vector of vertices to deduplicate
/// * `epsilon` - Distance threshold below which vertices are considered duplicates
///
/// # Returns
///
/// A new vector containing vertices that are more than `epsilon` apart from each
/// other (strictly: distance > epsilon). The first occurrence of each cluster is kept.
///
/// # Examples
///
/// ```
/// use delaunay::core::util::dedup_vertices_epsilon;
/// use delaunay::core::vertex::Vertex;
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::traits::coordinate::Coordinate;
///
/// let v1: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([0.0, 0.0])])
///     .into_iter().next().unwrap();
/// let v2: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([1e-11, 1e-11])]) // Near duplicate
///     .into_iter().next().unwrap();
/// let v3: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([1.0, 1.0])])
///     .into_iter().next().unwrap();
///
/// let vertices = vec![v1, v2, v3];
/// let unique = dedup_vertices_epsilon(vertices, 1e-10);
/// assert_eq!(unique.len(), 2); // v2 filtered as near-duplicate of v1
/// ```
pub fn dedup_vertices_epsilon<T, U, const D: usize>(
    vertices: Vec<Vertex<T, U, D>>,
    epsilon: T,
) -> Vec<Vertex<T, U, D>>
where
    T: CoordinateScalar,
    U: DataType,
{
    debug_assert!(
        epsilon >= T::zero(),
        "dedup_vertices_epsilon expects non-negative epsilon",
    );

    let mut unique: Vec<Vertex<T, U, D>> = Vec::with_capacity(vertices.len());

    'outer: for v in vertices {
        let vc: [T; D] = (&v).into();

        for u in &unique {
            let uc: [T; D] = u.into();

            // Euclidean distance check
            if coords_within_epsilon(&vc, &uc, epsilon) {
                continue 'outer; // Skip near-duplicate
            }
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
/// * `vertices` - Vector of vertices to filter
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
/// use delaunay::core::util::filter_vertices_excluding;
/// use delaunay::core::vertex::Vertex;
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::traits::coordinate::Coordinate;
///
/// let v1: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([0.0, 0.0])])
///     .into_iter().next().unwrap();
/// let v2: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([1.0, 1.0])])
///     .into_iter().next().unwrap();
///
/// let reference = vec![v1]; // Exclude origin
/// let vertices = vec![v1, v2];
///
/// let filtered = filter_vertices_excluding(vertices, &reference);
/// assert_eq!(filtered.len(), 1); // Only v2 remains
/// ```
pub fn filter_vertices_excluding<T, U, const D: usize>(
    vertices: Vec<Vertex<T, U, D>>,
    reference: &[Vertex<T, U, D>],
) -> Vec<Vertex<T, U, D>>
where
    T: CoordinateScalar,
    U: DataType,
{
    let mut filtered = Vec::with_capacity(vertices.len());

    'outer: for v in vertices {
        let vc: [T; D] = (&v).into();

        // Check against all reference vertices
        for ref_v in reference {
            let ref_c: [T; D] = ref_v.into();

            if coords_equal_exact(&vc, &ref_c) {
                continue 'outer; // Skip matching vertex
            }
        }

        filtered.push(v);
    }

    filtered
}

/// Check if two coordinate arrays are exactly equal.
///
/// Uses `OrderedEq` which provides NaN-aware equality comparison.
/// For f32/f64, this ensures consistent comparison including special values.
#[inline]
fn coords_equal_exact<T: CoordinateScalar, const D: usize>(a: &[T; D], b: &[T; D]) -> bool {
    // OrderedEq is already in scope via CoordinateScalar bound
    a.iter().zip(b.iter()).all(|(x, y)| x.ordered_eq(y))
}

/// Check if two coordinate arrays are within epsilon distance.
///
/// Returns true if Euclidean distance is strictly less than epsilon (distance < epsilon).
#[inline]
fn coords_within_epsilon<T: CoordinateScalar, const D: usize>(
    a: &[T; D],
    b: &[T; D],
    epsilon: T,
) -> bool {
    let dist_sq: T = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x - *y) * (*x - *y))
        .fold(T::zero(), |acc, d| acc + d);

    dist_sq < epsilon * epsilon
}

/// NOTE: The deprecated `facets_are_adjacent` function has been removed in Phase 3A.
///
/// Use [`facet_views_are_adjacent`] instead, which works with the lightweight `FacetView` API.
///
/// Determines if two facet views are adjacent by comparing their vertices.
///
/// Two facets are considered adjacent if they contain the same set of vertices.
/// This is the modern replacement for `facets_are_adjacent` using `FacetView`.
///
/// # Arguments
///
/// * `facet1` - The first facet view to compare
/// * `facet2` - The second facet view to compare
///
/// # Returns
///
/// `Ok(true)` if the facets share the same vertices, `Ok(false)` if they have
/// different vertices, or `Err(FacetError)` if there was an error accessing
/// the facet data.
///
/// # Errors
///
/// Returns `FacetError` if either facet's vertices cannot be accessed, typically
/// due to missing cells in the triangulation data structure.
///
/// # Examples
///
/// ```rust,no_run
/// use delaunay::core::facet::{FacetView, FacetError};
/// use delaunay::core::util::facet_views_are_adjacent;
/// use delaunay::core::triangulation_data_structure::Tds;
///
/// // This is a conceptual example - in practice you would get these from a real TDS
/// fn example(tds: &Tds<f64, (), (), 3>) -> Result<bool, FacetError> {
///     let cell_keys: Vec<_> = tds.cell_keys().take(2).collect();
///     if cell_keys.len() >= 2 {
///         let facet1 = FacetView::new(tds, cell_keys[0], 0)?;
///         let facet2 = FacetView::new(tds, cell_keys[1], 0)?;
///
///         let adjacent = facet_views_are_adjacent(&facet1, &facet2)?;
///         match adjacent {
///             true => println!("Facets are adjacent"),
///             false => println!("Facets are not adjacent"),
///         }
///         Ok(adjacent)
///     } else {
///         Ok(false)
///     }
/// }
/// ```
pub fn facet_views_are_adjacent<T, U, V, const D: usize>(
    facet1: &FacetView<'_, T, U, V, D>,
    facet2: &FacetView<'_, T, U, V, D>,
) -> Result<bool, FacetError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    use crate::core::collections::FastHashSet;

    // Compare facets by their vertex UUIDs for semantic correctness
    // This works across different TDS instances with the same coordinates
    let vertices1: FastHashSet<_> = facet1
        .vertices()?
        .map(super::vertex::Vertex::uuid)
        .collect();
    let vertices2: FastHashSet<_> = facet2
        .vertices()?
        .map(super::vertex::Vertex::uuid)
        .collect();

    Ok(vertices1 == vertices2)
}

/// Extracts owned vertices from a `FacetView` as a `Vec<Vertex>`.
///
/// This is a convenience utility that creates owned copies of the facet's vertices.
/// Since `Vertex` implements `Copy`, this operation is efficient and avoids the need
/// for manual `.copied().collect()` boilerplate throughout the codebase.
///
/// # Arguments
///
/// * `facet_view` - The facet view to extract vertices from
///
/// # Returns
///
/// A `Result` containing a `Vec` of owned `Vertex` objects, or a `FacetError` if
/// the vertices cannot be accessed.
///
/// # Errors
///
/// Returns `FacetError` if the facet's vertices cannot be accessed, typically
/// due to missing cells in the triangulation data structure.
///
/// # Examples
///
/// ```rust,no_run
/// use delaunay::core::facet::FacetView;
/// use delaunay::core::util::facet_view_to_vertices;
/// use delaunay::core::triangulation_data_structure::Tds;
///
/// fn extract_vertices_example(
///     tds: &Tds<f64, (), (), 3>,
/// ) -> Result<(), Box<dyn std::error::Error>> {
///     let cell_key = tds.cell_keys().next().unwrap();
///     let facet_view = FacetView::new(tds, cell_key, 0)?;
///     
///     // Extract owned vertices
///     let vertices = facet_view_to_vertices(&facet_view)?;
///     println!("Facet has {} vertices", vertices.len());
///     Ok(())
/// }
/// ```
///
/// # Performance
///
/// - Time Complexity: O(D) where D is the dimension (number of vertices in facet)
/// - Space Complexity: O(D) for the returned vector
/// - Uses `Copy` semantics so this is as efficient as possible for owned vertices
pub fn facet_view_to_vertices<T, U, V, const D: usize>(
    facet_view: &FacetView<'_, T, U, V, D>,
) -> Result<Vec<Vertex<T, U, D>>, FacetError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    Ok(facet_view.vertices()?.copied().collect())
}

/// Generates all unique combinations of `k` items from a given slice.
///
/// This function is used to generate vertex combinations for creating k-simplices
/// (e.g., edges, triangles, tetrahedra) from a set of vertices.
///
/// # Arguments
///
/// * `vertices` - A slice of vertices from which to generate combinations.
/// * `k` - The size of each combination.
///
/// # Returns
///
/// A vector of vectors, where each inner vector is a unique combination of `k` vertices.
///
/// # Examples
///
/// This function is made public for testing purposes.
///
/// ```
/// use delaunay::core::util::generate_combinations;
/// use delaunay::core::vertex::Vertex;
/// use delaunay::vertex;
///
/// let vertices: Vec<Vertex<f64, (), 1>> = vec![
///     vertex!([0.0]),
///     vertex!([1.0]),
///     vertex!([2.0]),
/// ];
///
/// // Generate all 2-vertex combinations (edges)
/// let combinations = generate_combinations(&vertices, 2);
///
/// assert_eq!(combinations.len(), 3);
/// assert!(combinations.contains(&vec![vertices[0], vertices[1]]));
/// assert!(combinations.contains(&vec![vertices[0], vertices[2]]));
/// assert!(combinations.contains(&vec![vertices[1], vertices[2]]));
/// ```
pub fn generate_combinations<T, U, const D: usize>(
    vertices: &[Vertex<T, U, D>],
    k: usize,
) -> Vec<Vec<Vertex<T, U, D>>>
where
    T: CoordinateScalar,
    U: DataType,
{
    let mut combinations = Vec::new();

    if k == 0 {
        combinations.push(Vec::new());
        return combinations;
    }

    if k > vertices.len() {
        return combinations;
    }

    if k == vertices.len() {
        combinations.push(vertices.to_vec());
        return combinations;
    }

    // Generate combinations using iterative approach
    let n = vertices.len();
    let mut indices = (0..k).collect::<Vec<_>>();

    loop {
        // Add current combination
        let combination = indices.iter().map(|i| vertices[*i]).collect();
        combinations.push(combination);

        // Find next combination
        let mut i = k;
        loop {
            if i == 0 {
                return combinations;
            }
            i -= 1;
            if indices[i] != i + n - k {
                break;
            }
        }

        indices[i] += 1;
        for j in (i + 1)..k {
            indices[j] = indices[j - 1] + 1;
        }
    }
}

// =============================================================================
// HASH UTILITIES
// =============================================================================

/// Applies a stable hash function to a slice of sorted u64 values.
///
/// This function uses an FNV-based polynomial rolling hash with an avalanche step
/// to produce deterministic hash values. The input slice should be pre-sorted to ensure
/// consistent results regardless of input order.
///
/// # Arguments
///
/// * `sorted_values` - A slice of u64 values that should be pre-sorted
///
/// # Returns
///
/// A `u64` hash value representing the stable hash of the input values
///
/// # Algorithm
///
/// Uses FNV constants with polynomial rolling hash:
/// 1. Start with FNV offset basis
/// 2. For each value: `hash = hash.wrapping_mul(PRIME).wrapping_add(value)`
/// 3. Apply avalanche step for better bit distribution
///
/// # Examples
///
/// ```
/// use delaunay::core::util::stable_hash_u64_slice;
/// let values = vec![1u64, 2u64, 3u64];
/// let hash1 = stable_hash_u64_slice(&values);
///
/// let mut reversed = values.clone();
/// reversed.reverse();
/// let hash2 = stable_hash_u64_slice(&reversed);
///
/// // Different order produces different hash (input should be pre-sorted)
/// assert_ne!(hash1, hash2);
///
/// // Same sorted input produces same hash
/// let mut sorted1 = values;
/// sorted1.sort_unstable();
/// let mut sorted2 = reversed;
/// sorted2.sort_unstable();
/// assert_eq!(stable_hash_u64_slice(&sorted1), stable_hash_u64_slice(&sorted2));
/// ```
#[must_use]
pub fn stable_hash_u64_slice(sorted_values: &[u64]) -> u64 {
    // Hash constants for facet key generation (FNV-based)
    const HASH_PRIME: u64 = 1_099_511_628_211; // Large prime (FNV prime)
    const HASH_OFFSET: u64 = 14_695_981_039_346_656_037; // FNV offset basis

    // Handle empty case
    if sorted_values.is_empty() {
        return 0;
    }

    // Use a polynomial rolling hash for efficient combination
    let mut hash = HASH_OFFSET;
    for &value in sorted_values {
        hash = hash.wrapping_mul(HASH_PRIME).wrapping_add(value);
    }

    // Apply avalanche step for better bit distribution
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    hash ^= hash >> 33;

    hash
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
    #[allow(clippy::cast_precision_loss)]
    let inter_f64 = intersection as f64;
    #[allow(clippy::cast_precision_loss)]
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
/// * `tds` - The triangulation data structure used to create the hull
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
        #[allow(clippy::cast_precision_loss)]
        let inter_f64 = intersection as f64;
        #[allow(clippy::cast_precision_loss)]
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

// =============================================================================
// FACET KEY UTILITIES
// =============================================================================

/// Derives a facet key directly from vertex keys.
///
/// Computes the canonical facet key for lookup in facet-to-cells mappings. This is useful
/// in hot paths like visibility checking in convex hull algorithms and boundary analysis.
///
/// If you have `Vertex` instances instead of `VertexKey`s, obtain the keys via the TDS
/// using the vertex's UUID (e.g., `tds.vertex_key_from_uuid(vertex.uuid())`).
///
/// # Arguments
///
/// * `facet_vertex_keys` - The vertex keys that make up the facet
///
/// # Returns
///
/// A `Result` containing the facet key or a `FacetError` if validation fails.
///
/// # Errors
///
/// Returns `FacetError::InsufficientVertices` if the vertex count doesn't equal `D`
///
/// # Examples
///
/// ```
/// use delaunay::prelude::*;
/// use delaunay::core::util::derive_facet_key_from_vertex_keys;
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
/// // Get facet vertex keys from a cell - no need to materialize Vertex objects
/// if let Some(cell) = tds.cells().map(|(_, cell)| cell).next() {
///     let facet_vertex_keys: Vec<_> = cell.vertices().iter().skip(1).copied().collect(); // Skip 1 vertex to get D vertices
///     assert_eq!(facet_vertex_keys.len(), 3); // For 3D triangulation, facet has 3 vertices
///     let facet_key = derive_facet_key_from_vertex_keys::<f64, (), (), 3>(&facet_vertex_keys).unwrap();
///     println!("Facet key: {}", facet_key);
/// }
/// ```
///
/// # Performance
///
/// - Time Complexity: O(D log D) where D is the facet dimension (for sorting vertex keys)
/// - Space Complexity: O(D) for the temporary sorted buffer (stack-allocated via `SmallBuffer`)
/// - Improves cache locality by working only with compact `VertexKey` types
///
/// # See Also
///
/// - [`crate::core::facet::facet_key_from_vertices`] - Low-level function that computes the hash from keys
pub fn derive_facet_key_from_vertex_keys<T, U, V, const D: usize>(
    facet_vertex_keys: &[crate::core::triangulation_data_structure::VertexKey],
) -> Result<u64, crate::core::facet::FacetError>
where
    T: crate::geometry::traits::coordinate::CoordinateScalar,
    U: crate::core::traits::data_type::DataType,
    V: crate::core::traits::data_type::DataType,
{
    use crate::core::facet::facet_key_from_vertices;

    // Validate that the number of vertex keys matches the expected dimension
    // In a D-dimensional triangulation, a facet should have exactly D vertices
    if facet_vertex_keys.len() != D {
        return Err(crate::core::facet::FacetError::InsufficientVertices {
            expected: D,
            actual: facet_vertex_keys.len(),
            dimension: D,
        });
    }

    // Directly compute the facet key from vertex keys
    // facet_key_from_vertices handles the sorting internally
    Ok(facet_key_from_vertices(facet_vertex_keys))
}

/// Verifies facet index consistency between two neighboring cells.
///
/// This function checks that a shared facet computed from both cells' perspectives
/// produces the same facet key, which is critical for catching subtle neighbor
/// assignment errors in triangulation algorithms.
///
/// # Arguments
///
/// * `tds` - The triangulation data structure
/// * `cell1_key` - Key of the first cell
/// * `cell2_key` - Key of the second (neighboring) cell
/// * `facet_idx` - Index of the facet in cell1 that should match a facet in cell2
///
/// # Returns
///
/// `Ok(true)` if a matching facet is found in cell2 with the same facet key.
/// `Ok(false)` if no matching facet is found.
/// `Err(FacetError)` if there's an error accessing cell or facet data.
///
/// # Errors
///
/// Returns `FacetError` if:
/// - Either cell cannot be found in the TDS
/// - Facet views cannot be created from the cells
/// - Facet vertices cannot be accessed
/// - The facet index is out of bounds
///
/// # Examples
///
/// ```rust,no_run
/// use delaunay::core::util::verify_facet_index_consistency;
/// use delaunay::core::triangulation_data_structure::Tds;
///
/// fn validate_neighbor_consistency(
///     tds: &Tds<f64, (), (), 3>,
/// ) -> Result<(), String> {
///     // Get two neighboring cell keys
///     let cell_keys: Vec<_> = tds.cell_keys().take(2).collect();
///     if cell_keys.len() >= 2 {
///         // Check if facet 0 of cell1 matches a facet in cell2
///         let consistent = verify_facet_index_consistency(
///             tds,
///             cell_keys[0],
///             cell_keys[1],
///             0,
///         )
///         .map_err(|e| format!("Facet error: {}", e))?;
///
///         if consistent {
///             println!("Facet indices are consistent");
///             Ok(())
///         } else {
///             Err("No matching facet found in neighbor".to_string())
///         }
///     } else {
///         Ok(())
///     }
/// }
/// ```
///
/// # Performance
///
/// - Time Complexity: O(D²) where D is the dimension (iterates over facets and vertices)
/// - Space Complexity: O(D) for temporary vertex buffers
pub fn verify_facet_index_consistency<T, U, V, const D: usize>(
    tds: &crate::core::triangulation_data_structure::Tds<T, U, V, D>,
    cell1_key: crate::core::triangulation_data_structure::CellKey,
    cell2_key: crate::core::triangulation_data_structure::CellKey,
    facet_idx: usize,
) -> Result<bool, crate::core::facet::FacetError>
where
    T: crate::geometry::traits::coordinate::CoordinateScalar,
    U: crate::core::traits::data_type::DataType,
    V: crate::core::traits::data_type::DataType,
{
    use crate::core::facet::FacetError;

    // Get facet views from both cells (validates cells exist)
    let cell1_facets = crate::core::cell::Cell::facet_views_from_tds(tds, cell1_key)?;
    let cell2_facets = crate::core::cell::Cell::facet_views_from_tds(tds, cell2_key)?;

    // Check facet index bounds
    if facet_idx >= cell1_facets.len() {
        // Use consistent error handling with proper overflow detection
        let idx_u8 = usize_to_u8(facet_idx, cell1_facets.len())?;
        return Err(FacetError::InvalidFacetIndex {
            index: idx_u8,
            facet_count: cell1_facets.len(),
        });
    }

    // Get the facet from cell1 and compute its key
    let cell1_facet = &cell1_facets[facet_idx];
    let cell1_key_value = cell1_facet.key()?;

    // Find matching facet in cell2
    for cell2_facet in &cell2_facets {
        if cell1_key_value == cell2_facet.key()? {
            return Ok(true);
        }
    }

    Ok(false) // No matching facet found
}

// =============================================================================
// MEMORY MEASUREMENT UTILITIES
// =============================================================================

/// Memory measurement helper for allocation tracking in examples, tests, and benchmarks.
///
/// This utility function provides a consistent interface for measuring memory allocations
/// across different parts of the codebase. It returns both the result of the closure
/// and allocation information when the `count-allocations` feature is enabled.
///
/// # Arguments
///
/// * `f` - A closure to execute while measuring allocations
///
/// # Returns
///
/// When `count-allocations` feature is enabled: Returns a tuple `(R, AllocationInfo)`
/// where `R` is the closure result and `AllocationInfo` contains allocation metrics.
///
/// When feature is disabled: Returns a tuple `(R, ())` where the allocation info is empty.
///
/// # Panics
///
/// This function should never panic under normal usage. The internal `expect()` call
/// is used because the closure is guaranteed to execute and set the result.
///
/// # Examples
///
/// ```rust,ignore
/// // With count-allocations feature enabled
/// use delaunay::core::util::measure_with_result;
///
/// let (result, alloc_info) = measure_with_result(|| {
///     // Some memory-allocating operation
///     vec![1, 2, 3, 4, 5]
/// });
///
/// #[cfg(feature = "count-allocations")]
/// println!("Allocated {} bytes", alloc_info.bytes_total);
/// ```
#[cfg(feature = "count-allocations")]
pub fn measure_with_result<F, R>(f: F) -> (R, allocation_counter::AllocationInfo)
where
    F: FnOnce() -> R,
{
    let mut result: Option<R> = None;
    let info = allocation_counter::measure(|| {
        result = Some(f());
    });
    (result.expect("Closure should have set result"), info)
}

/// Memory measurement helper (no-op version when count-allocations feature is disabled).
///
/// See [`measure_with_result`] for full documentation.
#[cfg(not(feature = "count-allocations"))]
pub fn measure_with_result<F, R>(f: F) -> (R, ())
where
    F: FnOnce() -> R,
{
    (f(), ())
}

/// Helper function to safely convert usize to u8 for facet indices.
///
/// This function provides a centralized, safe conversion from `usize` to `u8`
/// that is commonly needed throughout the triangulation codebase for facet indexing.
/// It handles the conversion error gracefully by returning appropriate `FacetError`
/// variants with detailed error information.
///
/// # Arguments
///
/// * `idx` - The usize index to convert
/// * `facet_count` - The number of facets (for error reporting)
///
/// # Returns
///
/// A `Result` containing the converted u8 index or a `FacetError`.
///
/// # Errors
///
/// Returns `FacetError::InvalidFacetIndexOverflow` if the index cannot fit in a u8.
///
/// # Examples
///
/// ```
/// use delaunay::core::util::usize_to_u8;
///
/// // Successful conversion
/// assert_eq!(usize_to_u8(0, 4), Ok(0));
/// assert_eq!(usize_to_u8(255, 256), Ok(255));
///
/// // Failed conversion
/// let result = usize_to_u8(256, 10);
/// assert!(result.is_err());
/// ```
pub fn usize_to_u8(idx: usize, facet_count: usize) -> Result<u8, FacetError> {
    u8::try_from(idx).map_err(|_| FacetError::InvalidFacetIndexOverflow {
        original_index: idx,
        facet_count,
    })
}

// =============================================================================
// DELAUNAY PROPERTY VALIDATION
// =============================================================================

/// Internal helper: Check if a single cell violates the Delaunay property.
///
/// Returns `Ok(None)` if the cell satisfies the Delaunay property,
/// `Ok(Some(cell_key))` if it violates (has an external vertex inside its circumsphere),
/// or `Err(...)` if validation fails due to structural or numeric issues.
fn validate_cell_delaunay<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    cell_key: CellKey,
    cell_vertex_points: &mut SmallVec<[Point<T, D>; 8]>,
    config: &crate::geometry::robust_predicates::RobustPredicateConfig<T>,
) -> Result<Option<CellKey>, DelaunayValidationError>
where
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    U: DataType,
    V: DataType,
{
    let Some(cell) = tds.get_cell(cell_key) else {
        // Cell doesn't exist (possibly removed), skip validation
        return Ok(None);
    };

    // Validate cell structure first
    cell.is_valid()
        .map_err(|source| DelaunayValidationError::InvalidCell { cell_key, source })?;

    // Get the cell's vertex set for exclusion
    let cell_vertex_keys: SmallVec<[VertexKey; 8]> = cell.vertices().iter().copied().collect();

    // Build the cell's circumsphere
    cell_vertex_points.clear();
    for &vkey in &cell_vertex_keys {
        let Some(v) = tds.get_vertex_by_key(vkey) else {
            return Err(DelaunayValidationError::TriangulationState {
                source: TdsValidationError::InconsistentDataStructure {
                    message: format!("Cell {cell_key:?} references non-existent vertex {vkey:?}"),
                },
            });
        };
        cell_vertex_points.push(*v.point());
    }

    // Check if any OTHER vertex is inside this cell's circumsphere
    for (test_vkey, test_vertex) in tds.vertices() {
        // Skip if this vertex is part of the cell
        if cell_vertex_keys.contains(&test_vkey) {
            continue;
        }

        // Test if this vertex is inside the cell's circumsphere using ROBUST predicates
        match robust_insphere(cell_vertex_points, test_vertex.point(), config) {
            Ok(InSphere::INSIDE) => {
                // Found a violation - this cell has an external vertex inside its circumsphere
                return Ok(Some(cell_key));
            }
            Ok(InSphere::BOUNDARY | InSphere::OUTSIDE) => {
                // Vertex is outside/on boundary; continue checking other vertices
            }
            Err(source) => {
                // Surface robust predicate failures as explicit validation errors
                return Err(DelaunayValidationError::NumericPredicateError {
                    cell_key,
                    vertex_key: test_vkey,
                    source,
                });
            }
        }
    }

    Ok(None)
}

/// Check if a triangulation satisfies the Delaunay property.
///
/// The Delaunay property states that no vertex should be inside the circumsphere
/// of any cell. This function checks all cells in the triangulation using robust
/// geometric predicates.
///
/// # ⚠️ Performance Warning
///
/// **This function is extremely expensive** - O(N×V) where N is the number of cells
/// and V is the number of vertices. For a triangulation with 10,000 cells and 5,000
/// vertices, this performs 50 million insphere tests. Use this primarily for:
/// - Debugging and testing
/// - Final validation after construction
/// - Verification of algorithm correctness
///
/// **Do NOT use this in production hot paths or for every vertex insertion.**
///
/// # Arguments
///
/// * `tds` - The triangulation data structure to validate
///
/// # Returns
///
/// `Ok(())` if all cells satisfy the Delaunay property, otherwise a [`DelaunayValidationError`]
/// describing the first violation found.
///
/// # Errors
///
/// Returns:
/// - [`DelaunayValidationError::DelaunayViolation`] if a cell has an external vertex inside its circumsphere
/// - [`DelaunayValidationError::TriangulationState`] if TDS corruption is detected
/// - [`DelaunayValidationError::InvalidCell`] if a cell has invalid structure
///
/// # Examples
///
/// ```
/// use delaunay::prelude::*;
/// use delaunay::core::util::is_delaunay;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
///
/// let dt = DelaunayTriangulation::new(&vertices).unwrap();
/// let tds = dt.tds();
///
/// // Check if triangulation is Delaunay
/// assert!(is_delaunay(tds).is_ok());
/// ```
#[deprecated(
    since = "0.6.1",
    note = "Use `DelaunayTriangulation::is_valid()` for Delaunay property validation (Level 4) or `DelaunayTriangulation::validate()` for layered validation (Levels 1-4). This will be removed in v0.7.0."
)]
pub fn is_delaunay<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
) -> Result<(), DelaunayValidationError>
where
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    U: DataType,
    V: DataType,
{
    // PERFORMANCE: O(N×V) - extremely expensive, use only for testing/validation
    // Check structural invariants first to distinguish "bad triangulation" from
    // "good triangulation but non-Delaunay"
    tds.is_valid()
        .map_err(|source| DelaunayValidationError::TriangulationState { source })?;

    is_delaunay_property_only(tds)
}

/// Internal helper: validate the Delaunay empty-circumsphere property only.
///
/// This performs the expensive geometric check but intentionally does **not** run
/// `tds.is_valid()` up front. Callers that want cumulative validation should run
/// lower-layer checks separately.
pub(crate) fn is_delaunay_property_only<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
) -> Result<(), DelaunayValidationError>
where
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    U: DataType,
    V: DataType,
{
    // Use robust predicates configuration for reliability
    let config = crate::geometry::robust_predicates::config_presets::general_triangulation::<T>();

    // Reusable buffer to minimize allocations
    let mut cell_vertex_points: SmallVec<[Point<T, D>; 8]> = SmallVec::with_capacity(D + 1);

    // Check each cell using the shared validation helper
    for cell_key in tds.cell_keys() {
        if let Some(violating_cell) =
            validate_cell_delaunay(tds, cell_key, &mut cell_vertex_points, &config)?
        {
            return Err(DelaunayValidationError::DelaunayViolation {
                cell_key: violating_cell,
            });
        }
    }

    Ok(())
}

/// Find cells that violate the Delaunay property.
///
/// This is a variant of [`is_delaunay`] that returns ALL violating cells instead of
/// stopping at the first violation. This is useful for iterative cavity refinement
/// and debugging.
///
/// # Arguments
///
/// * `tds` - The triangulation data structure
/// * `cells_to_check` - Optional subset of cells to check. If `None`, checks all cells.
///   Missing cells (e.g., already removed during refinement) are silently skipped.
///
/// # Returns
///
/// A vector of `CellKey`s for cells that violate the Delaunay property.
///
/// # Errors
///
/// Returns [`DelaunayValidationError`] if:
/// - A cell references a non-existent vertex (TDS corruption)
/// - A cell has invalid structure (cell-level corruption)
/// - Robust geometric predicates fail (numerical issues)
///
/// Note: Missing cells in `cells_to_check` are silently skipped and do not cause errors,
/// as they may have been legitimately removed during iterative refinement.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::*;
/// use delaunay::core::util::find_delaunay_violations;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
///
/// let dt = DelaunayTriangulation::new(&vertices).unwrap();
/// let tds = dt.tds();
///
/// // Find all violating cells (should be empty for valid Delaunay triangulation)
/// let violations = find_delaunay_violations(tds, None).unwrap();
/// assert!(violations.is_empty());
/// ```
pub fn find_delaunay_violations<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    cells_to_check: Option<&[CellKey]>,
) -> Result<ViolationBuffer, DelaunayValidationError>
where
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    U: DataType,
    V: DataType,
{
    let mut violating_cells = ViolationBuffer::new();
    let mut cell_vertex_points: SmallVec<[Point<T, D>; 8]> = SmallVec::with_capacity(D + 1);

    // Use robust predicates configuration for reliability
    let config = crate::geometry::robust_predicates::config_presets::general_triangulation::<T>();

    // Determine which cells to check
    let cells_iter: Box<dyn Iterator<Item = CellKey>> = match cells_to_check {
        Some(keys) => Box::new(keys.iter().copied()),
        None => Box::new(tds.cell_keys()),
    };

    // For each cell to check using the shared validation helper
    for cell_key in cells_iter {
        if let Some(violating_cell) =
            validate_cell_delaunay(tds, cell_key, &mut cell_vertex_points, &config)?
        {
            violating_cells.push(violating_cell);
        }
    }

    Ok(violating_cells)
}

/// Debug helper: print detailed information about the first detected Delaunay
/// violation (or all vertices if none are found) to aid in debugging.
///
/// This function is intended for use in tests and debug builds only. It uses the
/// same robust predicates as [`is_delaunay`] / [`find_delaunay_violations`] and
/// prints:
/// - A triangulation summary (vertex and cell counts)
/// - All vertices (keys, UUIDs, coordinates)
/// - All violating cells' vertices
/// - For the first violating cell:
///   - At least one offending external vertex (if found)
///   - Neighbor information for each facet
#[cfg(any(test, debug_assertions))]
#[allow(clippy::too_many_lines)]
pub fn debug_print_first_delaunay_violation<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    cells_subset: Option<&[CellKey]>,
) where
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    U: DataType,
    V: DataType,
{
    use crate::geometry::robust_predicates::config_presets;

    // First, find violating cells using the standard helper.
    let violations = match find_delaunay_violations(tds, cells_subset) {
        Ok(v) => v,
        Err(e) => {
            eprintln!(
                "[Delaunay debug] debug_print_first_delaunay_violation: error while finding violations: {e}"
            );
            return;
        }
    };

    eprintln!(
        "[Delaunay debug] Triangulation summary: {} vertices, {} cells",
        tds.number_of_vertices(),
        tds.number_of_cells()
    );

    // Dump all input vertices once for reproducibility.
    for (vkey, vertex) in tds.vertices() {
        eprintln!(
            "[Delaunay debug] Vertex {:?}: uuid={}, point={:?}",
            vkey,
            vertex.uuid(),
            vertex.point()
        );
    }

    if violations.is_empty() {
        eprintln!("[Delaunay debug] No Delaunay violations detected for requested cell subset");
        return;
    }

    eprintln!(
        "[Delaunay debug] Delaunay violations detected in {} cell(s):",
        violations.len()
    );

    // Reusable buffer for cell vertex points.
    let mut cell_vertex_points: SmallVec<[Point<T, D>; 8]> = SmallVec::with_capacity(D + 1);

    // Dump each violating cell with its vertices.
    for cell_key in &violations {
        match tds.get_cell(*cell_key) {
            Some(cell) => {
                eprintln!(
                    "[Delaunay debug]  Cell {:?}: uuid={}, vertices:",
                    cell_key,
                    cell.uuid()
                );
                for &vkey in cell.vertices() {
                    match tds.get_vertex_by_key(vkey) {
                        Some(v) => {
                            eprintln!(
                                "[Delaunay debug]    vkey={:?}, uuid={}, point={:?}",
                                vkey,
                                v.uuid(),
                                v.point()
                            );
                        }
                        None => {
                            eprintln!("[Delaunay debug]    vkey={vkey:?} (missing in TDS)");
                        }
                    }
                }
            }
            None => {
                eprintln!(
                    "[Delaunay debug]  Cell {cell_key:?} not found in TDS during violation dump"
                );
            }
        }
    }

    // Focus on the first violating cell to identify at least one offending
    // external vertex and neighbor information.
    let first_cell_key = violations[0];
    let Some(cell) = tds.get_cell(first_cell_key) else {
        eprintln!("[Delaunay debug] First violating cell {first_cell_key:?} not found in TDS");
        return;
    };

    let cell_vertex_keys: SmallVec<[VertexKey; 8]> = cell.vertices().iter().copied().collect();

    cell_vertex_points.clear();
    for &vkey in &cell_vertex_keys {
        if let Some(v) = tds.get_vertex_by_key(vkey) {
            cell_vertex_points.push(*v.point());
        }
    }

    let config = config_presets::general_triangulation::<T>();
    let mut offending: Option<(VertexKey, Point<T, D>)> = None;

    for (test_vkey, test_vertex) in tds.vertices() {
        if cell_vertex_keys.contains(&test_vkey) {
            continue;
        }

        match robust_insphere(&cell_vertex_points, test_vertex.point(), &config) {
            Ok(InSphere::INSIDE) => {
                offending = Some((test_vkey, *test_vertex.point()));
                break;
            }
            Ok(InSphere::BOUNDARY | InSphere::OUTSIDE) => {}
            Err(e) => {
                eprintln!(
                    "[Delaunay debug] robust_insphere error while searching for offending vertex in cell {first_cell_key:?}: {e}",
                );
            }
        }
    }

    if let Some((off_vkey, off_point)) = offending {
        eprintln!(
            "[Delaunay debug]  Offending external vertex: vkey={off_vkey:?}, point={off_point:?}",
        );
    } else {
        eprintln!(
            "[Delaunay debug]  No offending external vertex found for first violating cell (possible degeneracy or removed vertices)"
        );
    }

    // Neighbor information for the first violating cell.
    if let Some(neighbors) = cell.neighbors() {
        for (facet_idx, neighbor_key_opt) in neighbors.iter().enumerate() {
            match neighbor_key_opt {
                Some(neighbor_key) => {
                    if let Some(neighbor_cell) = tds.get_cell(*neighbor_key) {
                        eprintln!(
                            "[Delaunay debug]  facet {facet_idx}: neighbor cell {neighbor_key:?}, uuid={}",
                            neighbor_cell.uuid()
                        );
                    } else {
                        eprintln!(
                            "[Delaunay debug]  facet {facet_idx}: neighbor cell {neighbor_key:?} missing from TDS",
                        );
                    }
                }
                None => {
                    eprintln!(
                        "[Delaunay debug]  facet {facet_idx}: no neighbor (hull facet or unassigned)"
                    );
                }
            }
        }
    } else {
        eprintln!(
            "[Delaunay debug]  First violating cell has no neighbors assigned (neighbors() == None)"
        );
    }
}

#[cfg(test)]
mod tests {

    use crate::core::facet::FacetView;
    use crate::core::triangulation_data_structure::VertexKey;
    use crate::geometry::algorithms::convex_hull::ConvexHull;
    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::Coordinate;
    use crate::vertex;
    use slotmap::KeyData;
    use std::thread;
    use std::time::Instant;

    use super::*;
    use approx::assert_relative_eq;

    // =============================================================================
    // UUID UTILITIES TESTS
    // =============================================================================

    #[test]
    fn test_validate_uuid_comprehensive() {
        // Sub-test: UUID creation uniqueness
        let uuid1 = make_uuid();
        let uuid2 = make_uuid();
        let uuid3 = make_uuid();
        assert_ne!(uuid1, uuid2, "UUIDs should be unique");
        assert_ne!(uuid1, uuid3, "UUIDs should be unique");
        assert_ne!(uuid2, uuid3, "UUIDs should be unique");
        assert_eq!(uuid1.get_version_num(), 4, "Should be version 4");
        assert_eq!(uuid2.get_version_num(), 4, "Should be version 4");
        assert_eq!(uuid3.get_version_num(), 4, "Should be version 4");

        // Sub-test: UUID format validation
        let uuid = make_uuid();
        let uuid_string = uuid.to_string();
        assert_eq!(
            uuid_string.len(),
            36,
            "UUID should be 36 chars (with hyphens)"
        );
        assert_eq!(
            uuid_string.chars().filter(|&c| c == '-').count(),
            4,
            "UUID should have 4 hyphens"
        );
        let parts: Vec<&str> = uuid_string.split('-').collect();
        assert_eq!(parts.len(), 5, "UUID should have 5 hyphen-separated parts");
        assert_eq!(parts[0].len(), 8, "First part should be 8 chars");
        assert_eq!(parts[1].len(), 4, "Second part should be 4 chars");
        assert_eq!(parts[2].len(), 4, "Third part should be 4 chars");
        assert_eq!(parts[3].len(), 4, "Fourth part should be 4 chars");
        assert_eq!(parts[4].len(), 12, "Fifth part should be 12 chars");

        // Sub-test: Valid UUID (version 4)
        let valid_uuid = make_uuid();
        assert!(
            validate_uuid(&valid_uuid).is_ok(),
            "Valid v4 UUID should pass validation"
        );

        // Test nil UUID
        let nil_uuid = Uuid::nil();
        let nil_result = validate_uuid(&nil_uuid);
        assert!(nil_result.is_err(), "Nil UUID should fail validation");
        match nil_result {
            Err(UuidValidationError::NilUuid) => (), // Expected
            Err(other) => panic!("Expected NilUuid error, got: {other:?}"),
            Ok(()) => panic!("Expected error for nil UUID, but validation passed"),
        }

        // Test wrong version UUID (version 1)
        let v1_uuid = Uuid::parse_str("550e8400-e29b-11d4-a716-446655440000").unwrap();
        assert_eq!(v1_uuid.get_version_num(), 1);
        let version_result = validate_uuid(&v1_uuid);
        assert!(
            version_result.is_err(),
            "Non-v4 UUID should fail validation"
        );
        match version_result {
            Err(UuidValidationError::InvalidVersion { found }) => {
                assert_eq!(found, 1, "Should report correct version number");
            }
            Err(other) => panic!("Expected InvalidVersion error, got: {other:?}"),
            Ok(()) => panic!("Expected error for version 1 UUID, but validation passed"),
        }

        // Test error display formatting
        let nil_error = UuidValidationError::NilUuid;
        let nil_error_string = format!("{nil_error}");
        assert!(
            nil_error_string.contains("nil"),
            "Nil error message should contain 'nil'"
        );
        assert!(
            nil_error_string.contains("not allowed"),
            "Nil error message should mention 'not allowed'"
        );

        let version_error = UuidValidationError::InvalidVersion { found: 3 };
        let version_error_string = format!("{version_error}");
        assert!(
            version_error_string.contains("version 4"),
            "Version error should mention 'version 4'"
        );
        assert!(
            version_error_string.contains("found version 3"),
            "Version error should show found version"
        );

        // Test PartialEq for UuidValidationError
        let error1 = UuidValidationError::NilUuid;
        let error2 = UuidValidationError::NilUuid;
        assert_eq!(error1, error2, "Same nil errors should be equal");

        let error3 = UuidValidationError::InvalidVersion { found: 2 };
        let error4 = UuidValidationError::InvalidVersion { found: 2 };
        assert_eq!(error3, error4, "Same version errors should be equal");

        let error5 = UuidValidationError::InvalidVersion { found: 3 };
        assert_ne!(
            error3, error5,
            "Different version errors should not be equal"
        );
        assert_ne!(error1, error3, "Different error types should not be equal");
    }

    // =============================================================================
    // FACET UTILITIES TESTS
    // =============================================================================

    // Phase 3A: Test removed - was testing deprecated Facet type and facets() method
    // This functionality is now comprehensively tested in:
    // - test_facet_views_are_adjacent_comprehensive() (3D)
    // - test_facet_views_are_adjacent_2d_cases() (2D)
    // - test_facet_views_are_adjacent_1d_cases() (1D)
    // - test_facet_views_are_adjacent_4d_cases() (4D)
    // - test_facet_views_are_adjacent_5d_cases() (5D)
    // The modern facet_views_are_adjacent function works with lightweight FacetView API

    // =============================================================================
    // HASH UTILITIES TESTS
    // =============================================================================

    #[test]
    fn test_stable_hash_u64_slice_comprehensive() {
        // Test basic functionality with order sensitivity
        let values = vec![1u64, 2u64, 3u64];
        let hash1 = stable_hash_u64_slice(&values);

        let mut reversed = values.clone();
        reversed.reverse();
        let hash2 = stable_hash_u64_slice(&reversed);
        assert_ne!(
            hash1, hash2,
            "Different order should produce different hash"
        );

        // Same sorted input produces same hash
        let mut sorted1 = values;
        sorted1.sort_unstable();
        let mut sorted2 = reversed;
        sorted2.sort_unstable();
        assert_eq!(
            stable_hash_u64_slice(&sorted1),
            stable_hash_u64_slice(&sorted2),
            "Same sorted input should produce same hash"
        );

        // Test edge cases: empty, single value, different lengths
        let empty: Vec<u64> = vec![];
        assert_eq!(
            stable_hash_u64_slice(&empty),
            0,
            "Empty slice should produce hash 0"
        );

        let single = vec![42u64];
        let single_copy = vec![42u64];
        assert_eq!(
            stable_hash_u64_slice(&single),
            stable_hash_u64_slice(&single_copy),
            "Same single value should produce same hash"
        );

        let different_single = vec![43u64];
        assert_ne!(
            stable_hash_u64_slice(&single),
            stable_hash_u64_slice(&different_single),
            "Different single values should produce different hashes"
        );

        // Test deterministic behavior
        let test_values = vec![100u64, 200u64, 300u64, 400u64];
        let hash_a = stable_hash_u64_slice(&test_values);
        let hash_b = stable_hash_u64_slice(&test_values);
        let hash_c = stable_hash_u64_slice(&test_values);
        assert_eq!(hash_a, hash_b, "Multiple calls should be deterministic");
        assert_eq!(hash_b, hash_c, "Multiple calls should be deterministic");

        // Test different lengths
        let short = vec![1u64, 2u64];
        let long = vec![1u64, 2u64, 3u64];
        assert_ne!(
            stable_hash_u64_slice(&short),
            stable_hash_u64_slice(&long),
            "Different lengths should produce different hashes"
        );

        // Test large values
        let large_values = vec![u64::MAX, u64::MAX - 1, u64::MAX - 2];
        let hash_large1 = stable_hash_u64_slice(&large_values);
        let hash_large2 = stable_hash_u64_slice(&large_values);
        assert_eq!(
            hash_large1, hash_large2,
            "Large values should be handled consistently"
        );

        let different_large = vec![u64::MAX - 3, u64::MAX - 4, u64::MAX - 5];
        assert_ne!(
            hash_large1,
            stable_hash_u64_slice(&different_large),
            "Different large values should produce different hashes"
        );
    }

    // =============================================================================
    // VERTEX DEDUPLICATION UTILITIES TESTS
    // =============================================================================

    #[test]
    fn test_dedup_vertices_exact_comprehensive() {
        // Sub-test: Basic deduplication
        let v1: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([0.0, 0.0])])
            .into_iter()
            .next()
            .unwrap();
        let v2: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([0.0, 0.0])])
            .into_iter()
            .next()
            .unwrap();
        let v3: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([1.0, 1.0])])
            .into_iter()
            .next()
            .unwrap();
        let vertices = vec![v1, v2, v3];
        let unique = dedup_vertices_exact(vertices);
        assert_eq!(unique.len(), 2, "Should remove exact duplicate");

        // Sub-test: NaN handling - NaN should equal NaN
        let v1_nan: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([f64::NAN, f64::NAN])])
            .into_iter()
            .next()
            .unwrap();
        let v2_nan: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([f64::NAN, f64::NAN])])
            .into_iter()
            .next()
            .unwrap();
        let v3_regular: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([1.0, 1.0])])
            .into_iter()
            .next()
            .unwrap();
        let vertices_nan = vec![v1_nan, v2_nan, v3_regular];
        let unique_nan = dedup_vertices_exact(vertices_nan);
        assert_eq!(
            unique_nan.len(),
            2,
            "NaN should be considered equal to NaN for deduplication"
        );

        // Sub-test: Zero handling - +0.0 should equal -0.0
        let v1_pos_zero: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([0.0, 0.0])])
            .into_iter()
            .next()
            .unwrap();
        let v2_neg_zero: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([-0.0, -0.0])])
            .into_iter()
            .next()
            .unwrap();
        let v3_one: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([1.0, 1.0])])
            .into_iter()
            .next()
            .unwrap();
        let vertices_zero = vec![v1_pos_zero, v2_neg_zero, v3_one];
        let unique_zero = dedup_vertices_exact(vertices_zero);
        assert_eq!(
            unique_zero.len(),
            2,
            "+0.0 and -0.0 should be considered equal for deduplication"
        );
    }

    #[test]
    fn test_dedup_vertices_epsilon_basic() {
        // Near-duplicates should be filtered
        let v1: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([0.0, 0.0])])
            .into_iter()
            .next()
            .unwrap();
        let v2: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([1e-11, 1e-11])])
            .into_iter()
            .next()
            .unwrap();
        let v3: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([1.0, 1.0])])
            .into_iter()
            .next()
            .unwrap();

        let vertices = vec![v1, v2, v3];
        let unique = dedup_vertices_epsilon(vertices, 1e-10);
        assert_eq!(
            unique.len(),
            2,
            "Near-duplicate within epsilon should be removed"
        );
    }

    #[test]
    fn test_dedup_vertices_epsilon_boundary() {
        // Test strict < epsilon semantics (distance = epsilon should NOT be filtered)
        let v1: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([0.0, 0.0])])
            .into_iter()
            .next()
            .unwrap();
        // Distance exactly epsilon (1e-10) in x direction
        let v2: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([1e-10, 0.0])])
            .into_iter()
            .next()
            .unwrap();
        // Distance slightly less than epsilon
        let v3: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([0.99e-10, 0.0])])
            .into_iter()
            .next()
            .unwrap();

        let vertices = vec![v1, v2, v3];
        let unique = dedup_vertices_epsilon(vertices, 1e-10);
        // v1 kept, v3 filtered (< epsilon), v2 kept (= epsilon, not < epsilon)
        assert_eq!(
            unique.len(),
            2,
            "Distance exactly equal to epsilon should NOT be filtered (strict < semantics)"
        );
    }

    #[test]
    fn test_dedup_vertices_epsilon_preserves_first_occurrence() {
        // Verify that first occurrence is kept, later duplicates removed
        let points = [
            Point::new([0.0, 0.0]),
            Point::new([1e-11, 1e-11]), // Near-duplicate of first
            Point::new([1.0, 1.0]),
            Point::new([1.0 + 1e-11, 1.0 + 1e-11]), // Near-duplicate of third
        ];
        let vertices: Vec<Vertex<f64, (), 2>> = Vertex::from_points(&points);

        let unique = dedup_vertices_epsilon(vertices, 1e-10);
        assert_eq!(unique.len(), 2, "Should keep first of each cluster");

        // Verify first occurrences are kept
        let unique_coords: Vec<_> = unique
            .iter()
            .map(<&Vertex<_, _, _> as Into<[f64; 2]>>::into)
            .collect();
        assert_relative_eq!(unique_coords[0][0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(unique_coords[0][1], 0.0, epsilon = 1e-12);
        assert_relative_eq!(unique_coords[1][0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(unique_coords[1][1], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_filter_vertices_excluding_comprehensive() {
        // Sub-test: Basic exclusion
        let v1: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([0.0, 0.0])])
            .into_iter()
            .next()
            .unwrap();
        let v2: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([1.0, 1.0])])
            .into_iter()
            .next()
            .unwrap();
        let v3: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([2.0, 2.0])])
            .into_iter()
            .next()
            .unwrap();
        let reference_basic = vec![v1];
        let vertices_basic = vec![v1, v2, v3];
        let filtered_basic = filter_vertices_excluding(vertices_basic, &reference_basic);
        assert_eq!(
            filtered_basic.len(),
            2,
            "Should exclude vertex matching reference"
        );

        // Sub-test: NaN exclusion - NaN reference should match NaN vertices
        let v_nan: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([f64::NAN, f64::NAN])])
            .into_iter()
            .next()
            .unwrap();
        let reference_nan = vec![v_nan];
        let vertices_with_nan: Vec<Vertex<f64, (), 2>> =
            Vertex::from_points(&[Point::new([f64::NAN, f64::NAN]), Point::new([1.0, 1.0])]);
        let filtered_nan = filter_vertices_excluding(vertices_with_nan, &reference_nan);
        assert_eq!(
            filtered_nan.len(),
            1,
            "NaN reference should exclude NaN vertex"
        );

        // Sub-test: Zero exclusion - +0.0 reference should match -0.0 vertices
        let v_pos_zero: Vertex<f64, (), 2> = Vertex::from_points(&[Point::new([0.0, 0.0])])
            .into_iter()
            .next()
            .unwrap();
        let reference_zero = vec![v_pos_zero];
        let vertices_with_neg_zero: Vec<Vertex<f64, (), 2>> =
            Vertex::from_points(&[Point::new([-0.0, -0.0]), Point::new([1.0, 1.0])]);
        let filtered_zero = filter_vertices_excluding(vertices_with_neg_zero, &reference_zero);
        assert_eq!(
            filtered_zero.len(),
            1,
            "+0.0 reference should exclude -0.0 vertex"
        );

        // Sub-test: Multiple reference vertices
        // Multiple reference vertices
        let points = [
            Point::new([0.0, 0.0]),
            Point::new([1.0, 1.0]),
            Point::new([2.0, 2.0]),
            Point::new([3.0, 3.0]),
        ];
        let vertices: Vec<Vertex<f64, (), 2>> = Vertex::from_points(&points);

        let reference = vec![vertices[0], vertices[2]]; // Exclude first and third
        let filtered = filter_vertices_excluding(vertices, &reference);

        assert_eq!(filtered.len(), 2, "Should exclude both reference vertices");

        // Verify remaining vertices are second and fourth
        let filtered_coords: Vec<_> = filtered
            .iter()
            .map(<&Vertex<_, _, _> as Into<[f64; 2]>>::into)
            .collect();
        assert_relative_eq!(filtered_coords[0][0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(filtered_coords[0][1], 1.0, epsilon = 1e-12);
        assert_relative_eq!(filtered_coords[1][0], 3.0, epsilon = 1e-12);
        assert_relative_eq!(filtered_coords[1][1], 3.0, epsilon = 1e-12);
    }

    // =============================================================================
    // SET SIMILARITY UTILITIES TESTS
    // =============================================================================

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

    // =============================================================================
    // COMBINATION UTILITIES TESTS
    // =============================================================================

    #[test]
    fn test_generate_combinations_comprehensive() {
        // Test basic functionality with 4 vertices
        let vertices: Vec<Vertex<f64, (), 1>> = vec![
            vertex!([0.0]),
            vertex!([1.0]),
            vertex!([2.0]),
            vertex!([3.0]),
        ];

        // Combinations of 2 from 4 - should be C(4,2) = 6
        let combinations_2 = generate_combinations(&vertices, 2);
        assert_eq!(combinations_2.len(), 6, "C(4,2) should equal 6");

        // Combinations of 3 from 4 - should be C(4,3) = 4
        let combinations_3 = generate_combinations(&vertices, 3);
        assert_eq!(combinations_3.len(), 4, "C(4,3) should equal 4");
        assert!(
            combinations_3.contains(&vec![vertices[0], vertices[1], vertices[2]]),
            "Should contain specific combination"
        );

        // Single vertex combinations (k=1) - should be C(4,1) = 4
        let combinations_1 = generate_combinations(&vertices, 1);
        assert_eq!(combinations_1.len(), 4, "C(4,1) should equal 4");
        assert!(
            combinations_1.contains(&vec![vertices[0]]),
            "Should contain first vertex"
        );
        assert!(
            combinations_1.contains(&vec![vertices[1]]),
            "Should contain second vertex"
        );
        assert!(
            combinations_1.contains(&vec![vertices[2]]),
            "Should contain third vertex"
        );
        assert!(
            combinations_1.contains(&vec![vertices[3]]),
            "Should contain fourth vertex"
        );

        // Edge case: k=0 - should return one empty combination
        let combinations_0 = generate_combinations(&vertices, 0);
        assert_eq!(combinations_0.len(), 1, "C(4,0) should equal 1");
        assert!(
            combinations_0[0].is_empty(),
            "k=0 should produce empty combination"
        );

        // Edge case: k > len - should return empty result
        let combinations_5 = generate_combinations(&vertices, 5);
        assert!(
            combinations_5.is_empty(),
            "k > n should return no combinations"
        );

        // Edge case: k == len - should return all vertices as single combination
        let combinations_4 = generate_combinations(&vertices, 4);
        assert_eq!(combinations_4.len(), 1, "C(4,4) should equal 1");
        assert_eq!(
            combinations_4[0], vertices,
            "k=n should return all vertices"
        );

        // Test with different size - 3 vertices, choose 2
        let small_vertices: Vec<Vertex<f64, (), 1>> =
            vec![vertex!([1.0]), vertex!([2.0]), vertex!([3.0])];
        let combinations_small = generate_combinations(&small_vertices, 2);
        assert_eq!(combinations_small.len(), 3, "C(3,2) should equal 3");

        // Test larger case - 5 vertices, choose 3 to exercise inner loops
        let large_vertices: Vec<Vertex<f64, (), 1>> = vec![
            vertex!([1.0]),
            vertex!([2.0]),
            vertex!([3.0]),
            vertex!([4.0]),
            vertex!([5.0]),
        ];
        let combinations_large = generate_combinations(&large_vertices, 3);
        assert_eq!(combinations_large.len(), 10, "C(5,3) should equal 10");

        // Verify some specific combinations exist in large case
        assert!(
            combinations_large.contains(&vec![
                large_vertices[0],
                large_vertices[1],
                large_vertices[2]
            ]),
            "Should contain first combination"
        );
        assert!(
            combinations_large.contains(&vec![
                large_vertices[2],
                large_vertices[3],
                large_vertices[4]
            ]),
            "Should contain last combination"
        );

        // Test empty input edge cases
        let empty_vertices: Vec<Vertex<f64, (), 1>> = vec![];
        let combinations_empty_k1 = generate_combinations(&empty_vertices, 1);
        assert!(
            combinations_empty_k1.is_empty(),
            "Empty input with k>0 should return no combinations"
        );

        let combinations_empty_k0 = generate_combinations(&empty_vertices, 0);
        assert_eq!(
            combinations_empty_k0.len(),
            1,
            "Empty input with k=0 should return one empty combination"
        );
        assert!(
            combinations_empty_k0[0].is_empty(),
            "Empty input k=0 combination should be empty"
        );
    }

    // =============================================================================
    // MEMORY MEASUREMENT UTILITIES TESTS
    // =============================================================================

    #[test]
    fn test_measure_with_result_comprehensive() {
        // Test basic functionality - returns correct result
        let expected_result = 42;
        let (result, _alloc_info) = measure_with_result(|| expected_result);
        assert_eq!(result, expected_result);

        // Test with various allocation patterns
        let (vec_result, _alloc_info) = measure_with_result(|| vec![1, 2, 3, 4, 5]);
        assert_eq!(vec_result, vec![1, 2, 3, 4, 5]);

        let (string_result, _alloc_info) = measure_with_result(|| {
            let mut s = String::new();
            s.push_str("Hello, ");
            s.push_str("World!");
            s
        });
        assert_eq!(string_result, "Hello, World!");

        let (complex_result, _alloc_info) = measure_with_result(|| {
            let mut data: Vec<String> = Vec::new();
            for i in 0..5 {
                data.push(format!("Item {i}"));
            }
            data.len()
        });
        assert_eq!(complex_result, 5);

        // Test various return types
        let (tuple_result, _alloc_info) = measure_with_result(|| ("hello", 42));
        assert_eq!(tuple_result, ("hello", 42));

        let (option_result, _alloc_info) = measure_with_result(|| Some("value"));
        assert_eq!(option_result, Some("value"));

        let (result_result, _alloc_info) = measure_with_result(|| Ok::<i32, &str>(123));
        assert_eq!(result_result, Ok(123));

        // Test no-panic behavior
        let (sum_result, _alloc_info) = measure_with_result(|| {
            let data = [1, 2, 3];
            data.iter().sum::<i32>()
        });
        assert_eq!(sum_result, 6);
    }

    #[cfg(feature = "count-allocations")]
    #[test]
    fn test_measure_with_result_allocation_info_structure() {
        // Test that allocation info has expected structure when feature is enabled
        let (_result, alloc_info) = measure_with_result(|| {
            // Allocate some memory
            vec![0u8; 1024]
        });

        // Verify that we got an AllocationInfo struct by accessing its fields
        // This validates that the function properly returns allocation info
        // We access all fields to ensure the struct is properly constructed
        std::hint::black_box(&alloc_info.bytes_total);
        std::hint::black_box(&alloc_info.count_total);
        std::hint::black_box(&alloc_info.bytes_current);
        std::hint::black_box(&alloc_info.count_current);
        std::hint::black_box(&alloc_info.bytes_max);
        std::hint::black_box(&alloc_info.count_max);

        // Test that we can actually use the allocation info
        // For a vec![0u8; 1024], we expect some allocation to have occurred
        assert!(
            alloc_info.bytes_total > 0,
            "Should have allocated memory for the vector"
        );
    }

    #[cfg(not(feature = "count-allocations"))]
    #[test]
    fn test_measure_with_result_no_allocation_feature() {
        // Test that when feature is disabled, we get unit type
        let (_result, alloc_info) = measure_with_result(|| vec![0u8; 1024]);

        // Verify that alloc_info is unit type ()
        let _: () = alloc_info;
    }

    // =============================================================================
    // FACET KEY UTILITIES TESTS
    // =============================================================================

    #[test]
    fn delaunay_validator_reports_no_violations_for_simple_tetrahedron() {
        println!("Testing Delaunay validator and debug helper on a simple 3D tetrahedron");

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let dt =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices).unwrap();
        let tds = &dt.as_triangulation().tds;

        // Basic Delaunay helpers should report no violations.
        assert!(
            is_delaunay_property_only(tds).is_ok(),
            "Simple tetrahedron should satisfy the Delaunay property"
        );
        let violations = find_delaunay_violations(tds, None).unwrap();
        assert!(
            violations.is_empty(),
            "find_delaunay_violations should report no violating cells for a tetrahedron"
        );

        // Smoke test for the debug helper: it should not panic and should print a
        // summary indicating that no violations were found.
        #[cfg(any(test, debug_assertions))]
        debug_print_first_delaunay_violation(tds, None);
    }

    #[test]
    #[expect(clippy::too_many_lines)]
    fn test_derive_facet_key_from_vertex_keys_comprehensive() {
        println!("Testing derive_facet_key_from_vertex_keys comprehensively");

        // Create a triangulation
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices).unwrap();
        let tds = &dt.as_triangulation().tds;

        // Test 1: Basic functionality - successful key derivation
        println!("  Testing basic functionality...");
        let cell = tds.cells().map(|(_, cell)| cell).next().unwrap();
        let facet_vertex_keys: Vec<_> = cell.vertices().iter().skip(1).copied().collect();

        let result = derive_facet_key_from_vertex_keys::<f64, (), (), 3>(&facet_vertex_keys);
        assert!(
            result.is_ok(),
            "Facet key derivation should succeed for valid vertex keys"
        );

        let facet_key = result.unwrap();
        println!("    Derived facet key: {facet_key}");

        // Test deterministic behavior - same vertex keys produce same key
        let result2 = derive_facet_key_from_vertex_keys::<f64, (), (), 3>(&facet_vertex_keys);
        assert!(result2.is_ok(), "Second derivation should also succeed");
        assert_eq!(
            facet_key,
            result2.unwrap(),
            "Same vertex keys should produce same facet key"
        );

        // Test different vertex keys produce different keys
        let all_vertex_keys = cell.vertices();
        let different_facet_vertex_keys: Vec<_> = all_vertex_keys.iter().take(3).copied().collect();
        if different_facet_vertex_keys.len() == 3
            && different_facet_vertex_keys != facet_vertex_keys
        {
            let result3 =
                derive_facet_key_from_vertex_keys::<f64, (), (), 3>(&different_facet_vertex_keys);
            assert!(
                result3.is_ok(),
                "Different facet key derivation should succeed"
            );
            let different_facet_key = result3.unwrap();
            assert_ne!(
                facet_key, different_facet_key,
                "Different vertex keys should produce different facet keys"
            );
            println!("    Different facet key: {different_facet_key}");
        }

        // Test 2: Error cases
        println!("  Testing error handling...");

        // Wrong vertex key count
        let single_key: Vec<VertexKey> = vec![facet_vertex_keys[0]];
        let result_count = derive_facet_key_from_vertex_keys::<f64, (), (), 3>(&single_key);
        assert!(
            result_count.is_err(),
            "Should return error for wrong vertex key count"
        );
        if let Err(error) = result_count {
            match error {
                crate::core::facet::FacetError::InsufficientVertices {
                    expected,
                    actual,
                    dimension,
                } => {
                    assert_eq!(expected, 3, "Expected 3 vertex keys for 3D");
                    assert_eq!(actual, 1, "Got 1 vertex key");
                    assert_eq!(dimension, 3, "Dimension should be 3");
                }
                _ => panic!("Expected InsufficientVertices error, got: {error:?}"),
            }
        }

        // Empty vertex keys
        let empty_keys: Vec<VertexKey> = vec![];
        let result_empty = derive_facet_key_from_vertex_keys::<f64, (), (), 3>(&empty_keys);
        assert!(
            result_empty.is_err(),
            "Empty vertex keys should fail validation"
        );
        if let Err(error) = result_empty {
            match error {
                crate::core::facet::FacetError::InsufficientVertices {
                    expected,
                    actual,
                    dimension,
                } => {
                    assert_eq!(expected, 3, "Expected 3 vertex keys for 3D");
                    assert_eq!(actual, 0, "Got 0 vertex keys");
                    assert_eq!(dimension, 3, "Dimension should be 3");
                }
                _ => {
                    panic!(
                        "Expected InsufficientVertices error for empty vertex keys, got: {error:?}"
                    )
                }
            }
        }

        // Test 3: Consistency with TDS cache
        println!("  Testing consistency with TDS...");
        let cache = tds
            .build_facet_to_cells_map()
            .expect("Should build facet map in test");
        let mut keys_found = 0;
        let mut keys_tested = 0;

        for cell in tds.cells().map(|(_, cell)| cell) {
            let cell_vertex_keys = cell.vertices();
            for skip_vertex_idx in 0..cell_vertex_keys.len() {
                let facet_vertex_keys: Vec<_> = cell_vertex_keys
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != skip_vertex_idx)
                    .map(|(_, &vk)| vk)
                    .collect();

                if !facet_vertex_keys.is_empty() {
                    let key_result =
                        derive_facet_key_from_vertex_keys::<f64, (), (), 3>(&facet_vertex_keys);
                    if let Ok(derived_key) = key_result {
                        keys_tested += 1;
                        if cache.contains_key(&derived_key) {
                            keys_found += 1;
                        }
                    }
                }
            }
        }

        println!("    Found {keys_found}/{keys_tested} derived keys in TDS cache");
        assert!(keys_tested > 0, "Should have tested some keys");
        println!("  ✓ All facet key derivation tests passed");
    }

    // =============================================================================
    // FACET VIEW ADJACENCY TESTS
    // =============================================================================

    #[test]
    fn test_facet_views_are_adjacent_comprehensive() {
        // Test 1: Adjacent facets in 3D (tetrahedra sharing a triangular face)
        println!("Test 1: Adjacent facets in 3D");

        // Create two tetrahedra that share 3 vertices (forming a shared triangular face)
        let shared_vertices = vec![
            vertex!([0.0, 0.0, 0.0]), // v0
            vertex!([1.0, 0.0, 0.0]), // v1
            vertex!([0.5, 1.0, 0.0]), // v2
        ];

        let vertex_a = vertex!([0.5, 0.5, 1.0]); // Above the shared triangle
        let vertex_b = vertex!([0.5, 0.5, -1.0]); // Below the shared triangle

        // Tetrahedron 1: shared triangle + vertex_a
        let mut vertices1 = shared_vertices.clone();
        vertices1.push(vertex_a);

        // Tetrahedron 2: shared triangle + vertex_b
        let mut vertices2 = shared_vertices;
        vertices2.push(vertex_b);

        let dt1 =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices1).unwrap();
        let dt2 =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices2).unwrap();
        let tds1 = &dt1.as_triangulation().tds;
        let tds2 = &dt2.as_triangulation().tds;

        let cell1_key = tds1.cell_keys().next().unwrap();
        let cell2_key = tds2.cell_keys().next().unwrap();

        // Find the facets that correspond to the shared triangle
        // In tetrahedron 1, this is the facet opposite to vertex_a (index 3)
        // In tetrahedron 2, this is the facet opposite to vertex_b (index 3)
        let facet_view1 = FacetView::new(tds1, cell1_key, 3).unwrap();
        let facet_view2 = FacetView::new(tds2, cell2_key, 3).unwrap();

        assert!(
            facet_views_are_adjacent(&facet_view1, &facet_view2).unwrap(),
            "Facets representing the same shared triangle should be adjacent"
        );
        println!("  ✓ Adjacent facets correctly identified");

        // Test 2: Non-adjacent facets from the same tetrahedra
        println!("Test 2: Non-adjacent facets from same tetrahedra");

        // Different facets from the same tetrahedra (not sharing vertices)
        let facet_view1_diff = FacetView::new(tds1, cell1_key, 0).unwrap(); // Different facet
        let facet_view2_diff = FacetView::new(tds2, cell2_key, 1).unwrap(); // Different facet

        assert!(
            !facet_views_are_adjacent(&facet_view1_diff, &facet_view2_diff).unwrap(),
            "Different facets with different vertices should not be adjacent"
        );
        println!("  ✓ Non-adjacent facets correctly identified");

        // Test 3: Same facet should be adjacent to itself
        println!("Test 3: Facet adjacent to itself");

        assert!(
            facet_views_are_adjacent(&facet_view1, &facet_view1).unwrap(),
            "A facet should be adjacent to itself"
        );
        println!("  ✓ Self-adjacency works correctly");
    }

    #[test]
    fn test_facet_views_are_adjacent_2d_cases() {
        println!("Test 2D facet adjacency");

        // Create two 2D triangles that share an edge (2 vertices)
        let shared_edge = vec![
            vertex!([0.0, 0.0]), // v0
            vertex!([1.0, 0.0]), // v1
        ];

        let vertex_c = vertex!([0.5, 1.0]); // Above the shared edge
        let vertex_d = vertex!([0.5, -1.0]); // Below the shared edge

        // Triangle 1: shared edge + vertex_c
        let mut vertices1 = shared_edge.clone();
        vertices1.push(vertex_c);

        // Triangle 2: shared edge + vertex_d
        let mut vertices2 = shared_edge;
        vertices2.push(vertex_d);

        let dt1 =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices1).unwrap();
        let dt2 =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices2).unwrap();
        let tds1 = &dt1.as_triangulation().tds;
        let tds2 = &dt2.as_triangulation().tds;

        let cell1_key = tds1.cell_keys().next().unwrap();
        let cell2_key = tds2.cell_keys().next().unwrap();

        // In 2D, facets are edges. Find the facets that correspond to the shared edge
        // This is the facet opposite to the non-shared vertex
        let facet_view1 = FacetView::new(tds1, cell1_key, 2).unwrap(); // Opposite to vertex_c
        let facet_view2 = FacetView::new(tds2, cell2_key, 2).unwrap(); // Opposite to vertex_d

        assert!(
            facet_views_are_adjacent(&facet_view1, &facet_view2).unwrap(),
            "2D facets (edges) sharing vertices should be adjacent"
        );

        // Test non-adjacent edges
        let facet_view1_diff = FacetView::new(tds1, cell1_key, 0).unwrap();
        let facet_view2_diff = FacetView::new(tds2, cell2_key, 1).unwrap();

        assert!(
            !facet_views_are_adjacent(&facet_view1_diff, &facet_view2_diff).unwrap(),
            "2D facets with different vertices should not be adjacent"
        );

        println!("  ✓ 2D facet adjacency works correctly");
    }

    #[test]
    fn test_facet_views_are_adjacent_1d_cases() {
        println!("Test 1D facet adjacency");

        // In 1D, cells are edges and facets are vertices (0D)
        // Two edges sharing a vertex have adjacent facets

        let shared_vertex = vertex!([0.0]);
        let vertex_left = vertex!([-1.0]);
        let vertex_right = vertex!([1.0]);

        // Edge 1: shared_vertex to vertex_left
        let vertices1 = vec![shared_vertex, vertex_left];
        // Edge 2: shared_vertex to vertex_right
        let vertices2 = vec![shared_vertex, vertex_right];

        let dt1 =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices1).unwrap();
        let dt2 =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices2).unwrap();
        let tds1 = &dt1.as_triangulation().tds;
        let tds2 = &dt2.as_triangulation().tds;

        let cell1_key = tds1.cell_keys().next().unwrap();
        let cell2_key = tds2.cell_keys().next().unwrap();

        // In 1D, the facets are the individual vertices
        // Facet 0: opposite to vertex at index 0 (so contains vertex at index 1)
        // Facet 1: opposite to vertex at index 1 (so contains vertex at index 0)

        // Both edges contain the shared vertex, so we need to find which facet index
        // corresponds to the shared vertex
        let facet_view1_0 = FacetView::new(tds1, cell1_key, 0).unwrap(); // Contains vertex_left
        let facet_view1_1 = FacetView::new(tds1, cell1_key, 1).unwrap(); // Contains shared_vertex

        let facet_view2_0 = FacetView::new(tds2, cell2_key, 0).unwrap(); // Contains vertex_right
        let facet_view2_1 = FacetView::new(tds2, cell2_key, 1).unwrap(); // Contains shared_vertex

        // The facets containing the shared vertex should be adjacent
        assert!(
            facet_views_are_adjacent(&facet_view1_1, &facet_view2_1).unwrap(),
            "1D facets (vertices) that are the same should be adjacent"
        );

        // The facets containing different vertices should not be adjacent
        assert!(
            !facet_views_are_adjacent(&facet_view1_0, &facet_view2_0).unwrap(),
            "1D facets with different vertices should not be adjacent"
        );

        println!("  ✓ 1D facet adjacency works correctly");
    }

    #[test]
    fn test_facet_views_are_adjacent_edge_cases() {
        println!("Test facet adjacency edge cases");

        // Test with minimal triangulation (single tetrahedron)
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let dt =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices).unwrap();
        let tds = &dt.as_triangulation().tds;
        let cell_key = tds.cell_keys().next().unwrap();

        // All facets of the same tetrahedron should be different from each other
        let facet0 = FacetView::new(tds, cell_key, 0).unwrap();
        let facet1 = FacetView::new(tds, cell_key, 1).unwrap();
        let facet2 = FacetView::new(tds, cell_key, 2).unwrap();
        let facet3 = FacetView::new(tds, cell_key, 3).unwrap();

        // Each facet should be adjacent to itself
        assert!(facet_views_are_adjacent(&facet0, &facet0).unwrap());
        assert!(facet_views_are_adjacent(&facet1, &facet1).unwrap());
        assert!(facet_views_are_adjacent(&facet2, &facet2).unwrap());
        assert!(facet_views_are_adjacent(&facet3, &facet3).unwrap());

        // Different facets of the same tetrahedron should not be adjacent
        // (they have different sets of vertices)
        assert!(!facet_views_are_adjacent(&facet0, &facet1).unwrap());
        assert!(!facet_views_are_adjacent(&facet0, &facet2).unwrap());
        assert!(!facet_views_are_adjacent(&facet0, &facet3).unwrap());
        assert!(!facet_views_are_adjacent(&facet1, &facet2).unwrap());
        assert!(!facet_views_are_adjacent(&facet1, &facet3).unwrap());
        assert!(!facet_views_are_adjacent(&facet2, &facet3).unwrap());

        println!("  ✓ Single tetrahedron facet relationships correct");
    }

    #[test]
    fn test_facet_views_are_adjacent_performance() {
        println!("Test facet adjacency performance");

        // Create a moderately complex case to test performance
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([1.0, 2.0, 0.0]),
            vertex!([1.0, 1.0, 2.0]),
        ];

        let dt =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices).unwrap();
        let tds = &dt.as_triangulation().tds;
        let cell_key = tds.cell_keys().next().unwrap();

        let facet1 = FacetView::new(tds, cell_key, 0).unwrap();
        let facet2 = FacetView::new(tds, cell_key, 1).unwrap();

        // Run the adjacency check many times to measure performance
        let start = Instant::now();
        let iterations = 10000;

        for _ in 0..iterations {
            // This should be very fast since it just compares UUID sets
            let _result = facet_views_are_adjacent(&facet1, &facet2).unwrap();
        }

        let duration = start.elapsed();
        println!("  ✓ {iterations} adjacency checks completed in {duration:?}");

        // Performance info: each check is just UUID set comparison
        // Note: Timing can vary significantly based on build type and CI environment
        if duration.as_millis() > 500 {
            println!("  ⚠️  Performance warning: adjacency checks took {duration:?}");
            println!("     This may indicate debug build or slower CI environment");
        }
    }

    #[test]
    fn test_facet_views_are_adjacent_different_geometries() {
        println!("Test facet adjacency with different geometries");

        // Create vertices with different coordinates to ensure different UUIDs
        let vertices1 = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let vertices2 = vec![
            vertex!([10.0, 10.0, 10.0]),
            vertex!([11.0, 10.0, 10.0]),
            vertex!([10.0, 11.0, 10.0]),
            vertex!([10.0, 10.0, 11.0]),
        ];

        let dt1 =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices1).unwrap();
        let dt2 =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices2).unwrap();
        let tds1 = &dt1.as_triangulation().tds;
        let tds2 = &dt2.as_triangulation().tds;

        let cell1_key = tds1.cell_keys().next().unwrap();
        let cell2_key = tds2.cell_keys().next().unwrap();

        let facet1 = FacetView::new(tds1, cell1_key, 0).unwrap();
        let facet2 = FacetView::new(tds2, cell2_key, 0).unwrap();

        // Facets from completely different geometries should not be adjacent
        assert!(
            !facet_views_are_adjacent(&facet1, &facet2).unwrap(),
            "Facets from different geometries should not be adjacent"
        );

        println!("  ✓ Different geometries correctly distinguished");
    }

    #[test]
    fn test_facet_views_are_adjacent_uuid_based_comparison() {
        println!("Test that adjacency is purely UUID-based");

        // Create identical geometry in separate TDS instances
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let dt1 =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices).unwrap();
        let dt2 =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices).unwrap();
        let tds1 = &dt1.as_triangulation().tds;
        let tds2 = &dt2.as_triangulation().tds;

        let cell1_key = tds1.cell_keys().next().unwrap();
        let cell2_key = tds2.cell_keys().next().unwrap();

        let facet1 = FacetView::new(tds1, cell1_key, 0).unwrap();
        let facet2 = FacetView::new(tds2, cell2_key, 0).unwrap();

        // Check if the UUID generation is deterministic based on coordinates
        let facet1_vertex_uuids: Vec<_> = match facet1.vertices() {
            Ok(iter) => iter.map(Vertex::uuid).collect(),
            Err(_) => return, // Skip test if facet1 is invalid
        };
        let facet2_vertex_uuids: Vec<_> = match facet2.vertices() {
            Ok(iter) => iter.map(Vertex::uuid).collect(),
            Err(_) => return, // Skip test if facet2 is invalid
        };

        let uuids_are_same = facet1_vertex_uuids == facet2_vertex_uuids;
        let facets_are_adjacent = facet_views_are_adjacent(&facet1, &facet2).unwrap();

        // The adjacency should match the UUID equality
        assert_eq!(
            uuids_are_same, facets_are_adjacent,
            "Facet adjacency should exactly match vertex UUID equality"
        );

        if uuids_are_same {
            println!("  ✓ Identical coordinates produce identical UUIDs - facets are adjacent");
        } else {
            println!("  ✓ Different UUIDs for identical coordinates - facets are not adjacent");
        }
    }

    #[test]
    fn test_facet_views_are_adjacent_4d_cases() {
        println!("Test 4D facet adjacency");

        // Create two 4D simplices (5-vertices each) that share a 3D facet (4 vertices)
        let shared_tetrahedron = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]), // v0
            vertex!([1.0, 0.0, 0.0, 0.0]), // v1
            vertex!([0.0, 1.0, 0.0, 0.0]), // v2
            vertex!([0.0, 0.0, 1.0, 0.0]), // v3
        ];

        let vertex_e = vertex!([0.25, 0.25, 0.25, 1.0]); // Above in 4th dimension
        let vertex_f = vertex!([0.25, 0.25, 0.25, -1.0]); // Below in 4th dimension

        // 4D Simplex 1: shared tetrahedron + vertex_e
        let mut vertices1 = shared_tetrahedron.clone();
        vertices1.push(vertex_e);

        // 4D Simplex 2: shared tetrahedron + vertex_f
        let mut vertices2 = shared_tetrahedron;
        vertices2.push(vertex_f);

        let dt1 =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices1).unwrap();
        let dt2 =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices2).unwrap();
        let tds1 = &dt1.as_triangulation().tds;
        let tds2 = &dt2.as_triangulation().tds;

        let cell1_key = tds1.cell_keys().next().unwrap();
        let cell2_key = tds2.cell_keys().next().unwrap();

        // In 4D, facets are tetrahedra. Find the facets that correspond to the shared tetrahedron
        // This is the facet opposite to the non-shared vertex (index 4)
        let facet_view1 = FacetView::new(tds1, cell1_key, 4).unwrap(); // Opposite to vertex_e
        let facet_view2 = FacetView::new(tds2, cell2_key, 4).unwrap(); // Opposite to vertex_f

        assert!(
            facet_views_are_adjacent(&facet_view1, &facet_view2).unwrap(),
            "4D facets (tetrahedra) sharing vertices should be adjacent"
        );

        // Test non-adjacent tetrahedra within the same 4D simplices
        let facet_view1_diff = FacetView::new(tds1, cell1_key, 0).unwrap();
        let facet_view2_diff = FacetView::new(tds2, cell2_key, 1).unwrap();

        assert!(
            !facet_views_are_adjacent(&facet_view1_diff, &facet_view2_diff).unwrap(),
            "4D facets with different vertices should not be adjacent"
        );

        println!("  ✓ 4D facet adjacency works correctly");
    }

    #[test]
    fn test_facet_views_are_adjacent_5d_cases() {
        println!("Test 5D facet adjacency");

        // Create two 5D simplices (6-vertices each) that share a 4D facet (5 vertices)
        let shared_4d_simplex = vec![
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0]), // v0
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0]), // v1
            vertex!([0.0, 1.0, 0.0, 0.0, 0.0]), // v2
            vertex!([0.0, 0.0, 1.0, 0.0, 0.0]), // v3
            vertex!([0.0, 0.0, 0.0, 1.0, 0.0]), // v4
        ];

        let vertex_g = vertex!([0.2, 0.2, 0.2, 0.2, 1.0]); // Above in 5th dimension
        let vertex_h = vertex!([0.2, 0.2, 0.2, 0.2, -1.0]); // Below in 5th dimension

        // 5D Simplex 1: shared 4D simplex + vertex_g
        let mut vertices1 = shared_4d_simplex.clone();
        vertices1.push(vertex_g);

        // 5D Simplex 2: shared 4D simplex + vertex_h
        let mut vertices2 = shared_4d_simplex;
        vertices2.push(vertex_h);

        let dt1 =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices1).unwrap();
        let dt2 =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices2).unwrap();
        let tds1 = &dt1.as_triangulation().tds;
        let tds2 = &dt2.as_triangulation().tds;

        let cell1_key = tds1.cell_keys().next().unwrap();
        let cell2_key = tds2.cell_keys().next().unwrap();

        // In 5D, facets are 4D simplices. Find the facets that correspond to the shared 4D simplex
        // This is the facet opposite to the non-shared vertex (index 5)
        let facet_view1 = FacetView::new(tds1, cell1_key, 5).unwrap(); // Opposite to vertex_g
        let facet_view2 = FacetView::new(tds2, cell2_key, 5).unwrap(); // Opposite to vertex_h

        assert!(
            facet_views_are_adjacent(&facet_view1, &facet_view2).unwrap(),
            "5D facets (4D simplices) sharing vertices should be adjacent"
        );

        // Test non-adjacent 4D simplices within the same 5D simplices
        let facet_view1_diff = FacetView::new(tds1, cell1_key, 0).unwrap();
        let facet_view2_diff = FacetView::new(tds2, cell2_key, 1).unwrap();

        assert!(
            !facet_views_are_adjacent(&facet_view1_diff, &facet_view2_diff).unwrap(),
            "5D facets with different vertices should not be adjacent"
        );

        println!("  ✓ 5D facet adjacency works correctly");
    }

    #[test]
    fn test_facet_views_are_adjacent_multidimensional_summary() {
        println!("Testing facet adjacency across all supported dimensions (1D-5D)");

        // This test summarizes the multidimensional support
        let dimensions_tested = vec![
            ("1D", "edges", "vertices"),
            ("2D", "triangles", "edges"),
            ("3D", "tetrahedra", "triangles"),
            ("4D", "4-simplices", "tetrahedra"),
            ("5D", "5-simplices", "4-simplices"),
        ];

        for (dim, cell_type, facet_type) in dimensions_tested {
            println!("  ✓ {dim}: {cell_type} with {facet_type} facets");
        }

        println!("  ✓ All dimensional cases covered comprehensively");
    }

    // =============================================================================
    // USIZE TO U8 CONVERSION UTILITY TESTS
    // =============================================================================

    #[test]
    fn test_usize_to_u8_conversion_comprehensive() {
        // Sub-test: Successful conversions
        assert_eq!(usize_to_u8(0, 4), Ok(0));
        assert_eq!(usize_to_u8(1, 4), Ok(1));
        assert_eq!(usize_to_u8(255, 256), Ok(255));
        assert_eq!(usize_to_u8(u8::MAX as usize, 256), Ok(u8::MAX));

        // Sub-test: All valid u8 values (0..=255)
        for i in 0u8..=255 {
            let result = usize_to_u8(i as usize, 256);
            assert_eq!(result, Ok(i), "Failed to convert {i}");
        }

        // Sub-test: Edge cases around u8::MAX boundary
        assert_eq!(usize_to_u8(254, 300), Ok(254));
        assert_eq!(usize_to_u8(255, 300), Ok(255));
        assert!(usize_to_u8(256, 300).is_err());
        assert!(usize_to_u8(257, 300).is_err());

        // Sub-test: Different facet_count values
        let facet_counts = [1, 10, 100, 255, 256, 1000, usize::MAX];
        for &count in &facet_counts {
            assert_eq!(
                usize_to_u8(0, count),
                Ok(0),
                "Valid conversion should succeed"
            );
            let result_invalid = usize_to_u8(256, count);
            assert!(result_invalid.is_err(), "Invalid conversion should fail");
            if let Err(FacetError::InvalidFacetIndexOverflow { facet_count, .. }) = result_invalid {
                assert_eq!(facet_count, count);
            }
        }

        // Sub-test: Deterministic behavior
        for i in 0..10 {
            assert_eq!(
                usize_to_u8(i, 20),
                usize_to_u8(i, 20),
                "Should be deterministic"
            );
        }
        for i in [256, 1000, usize::MAX] {
            assert_eq!(
                usize_to_u8(i, 100),
                usize_to_u8(i, 100),
                "Error cases should be deterministic"
            );
        }
    }

    #[test]
    fn test_usize_to_u8_error_handling() {
        // Sub-test: Basic error cases
        let result = usize_to_u8(256, 10);
        assert!(result.is_err());
        if let Err(FacetError::InvalidFacetIndexOverflow {
            original_index,
            facet_count,
        }) = result
        {
            assert_eq!(original_index, 256);
            assert_eq!(facet_count, 10);
        } else {
            panic!("Expected InvalidFacetIndexOverflow error");
        }

        let result = usize_to_u8(usize::MAX, 5);
        assert!(result.is_err());
        if let Err(FacetError::InvalidFacetIndexOverflow {
            original_index,
            facet_count,
        }) = result
        {
            assert_eq!(original_index, usize::MAX);
            assert_eq!(facet_count, 5);
        } else {
            panic!("Expected InvalidFacetIndexOverflow error");
        }

        // Sub-test: Error consistency across various inputs
        let test_cases = [
            (256, 100),
            (300, 500),
            (1000, 1500),
            (usize::MAX, 42),
            (65536, 10),
        ];
        for &(idx, count) in &test_cases {
            let result = usize_to_u8(idx, count);
            assert!(result.is_err(), "Should fail for index {idx}");
            match result {
                Err(FacetError::InvalidFacetIndexOverflow {
                    original_index,
                    facet_count,
                }) => {
                    assert_eq!(original_index, idx, "Should preserve original index");
                    assert_eq!(facet_count, count, "Should preserve facet_count");
                }
                _ => panic!("Expected InvalidFacetIndex error for index {idx}"),
            }
        }

        // Sub-test: Large values
        let large_values = [257, 1000, 10000, 65536, usize::MAX];
        for &val in &large_values {
            let result = usize_to_u8(val, val);
            assert!(result.is_err(), "Should fail for value {val}");
            if let Err(FacetError::InvalidFacetIndexOverflow {
                original_index,
                facet_count,
            }) = result
            {
                assert_eq!(original_index, val);
                assert_eq!(facet_count, val);
            }
        }

        // Sub-test: Error message quality
        let result = usize_to_u8(300, 42);
        assert!(result.is_err());
        if let Err(error) = result {
            let error_string = format!("{error}");
            assert!(
                error_string.contains("InvalidFacetIndex") || error_string.contains("index"),
                "Error message should indicate invalid index: {error_string}"
            );
        }

        let result = usize_to_u8(usize::MAX, 7);
        if let Err(FacetError::InvalidFacetIndexOverflow {
            original_index,
            facet_count,
        }) = result
        {
            assert_eq!(original_index, usize::MAX);
            assert_eq!(facet_count, 7);
        } else {
            panic!("Expected InvalidFacetIndexOverflow error");
        }
    }

    #[test]
    fn test_usize_to_u8_performance_and_threading() {
        // Sub-test: Performance characteristics (informational only)
        let start = Instant::now();
        for i in 0..1000 {
            let _ = usize_to_u8(i % 256, 300);
        }
        let duration = start.elapsed();
        eprintln!("usize_to_u8 valid conversions: 1000 iters in {duration:?}");

        let start = Instant::now();
        for i in 256..1256 {
            let _ = usize_to_u8(i, 100);
        }
        let duration = start.elapsed();
        eprintln!("usize_to_u8 error conversions: 1000 iters in {duration:?}");

        // Sub-test: Memory efficiency (stack allocation only)
        let (result, _alloc_info) = measure_with_result(|| {
            let mut results = Vec::new();
            for i in 0..100 {
                results.push(usize_to_u8(i, 200));
            }
            results
        });
        for (i, result) in result.iter().enumerate() {
            assert_eq!(
                *result,
                Ok(u8::try_from(i).unwrap()),
                "Result should be correct for {i}"
            );
        }

        // Sub-test: Thread safety
        let handles: Vec<_> = (0..4)
            .map(|thread_id| {
                thread::spawn(move || {
                    let mut results = Vec::new();
                    for i in 0..100 {
                        let val = (thread_id * 50 + i) % 256;
                        results.push(usize_to_u8(val, 300));
                    }
                    results
                })
            })
            .collect();
        for handle in handles {
            let results = handle.join().expect("Thread should complete successfully");
            for result in results {
                assert!(result.is_ok(), "All results should be successful");
            }
        }
    }

    // =============================================================================
    // CANONICAL SET EXTRACTION TESTS
    // =============================================================================

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
    fn test_verify_facet_index_consistency_true_false_and_error_cases() {
        use crate::core::facet::FacetError;

        // True case: comparing a cell to itself.
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices).unwrap();
        let tds = &dt.as_triangulation().tds;
        let cell_key = tds.cell_keys().next().unwrap();
        assert!(verify_facet_index_consistency(tds, cell_key, cell_key, 0).unwrap());

        // Error case: facet index out of bounds.
        let err = verify_facet_index_consistency(tds, cell_key, cell_key, 99).unwrap_err();
        assert!(matches!(err, FacetError::InvalidFacetIndex { .. }));

        // False case: two disjoint triangles in the same TDS share no facet keys.
        let mut tds2: Tds<f64, (), (), 2> = Tds::empty();
        let v_a = tds2
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]))
            .unwrap();
        let v_b = tds2
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]))
            .unwrap();
        let v_c = tds2
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]))
            .unwrap();
        let v_d = tds2
            .insert_vertex_with_mapping(vertex!([10.0, 10.0]))
            .unwrap();
        let v_e = tds2
            .insert_vertex_with_mapping(vertex!([11.0, 10.0]))
            .unwrap();
        let v_f = tds2
            .insert_vertex_with_mapping(vertex!([10.0, 11.0]))
            .unwrap();

        let c1 = tds2
            .insert_cell_with_mapping(
                crate::core::cell::Cell::new(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let c2 = tds2
            .insert_cell_with_mapping(
                crate::core::cell::Cell::new(vec![v_d, v_e, v_f], None).unwrap(),
            )
            .unwrap();

        assert!(!verify_facet_index_consistency(&tds2, c1, c2, 0).unwrap());
    }
}
