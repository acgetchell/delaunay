//! Keyed map and set aliases for vertex, simplex, and UUID lookups.
//!
//! These types describe the canonical hash-map shapes used for topology storage,
//! validation, and geometric algorithm bookkeeping.

use super::{FastHashMap, FastHashSet, Uuid};
use crate::core::tds::{SimplexKey, VertexKey};

// =============================================================================
// GEOMETRIC ALGORITHM TYPES
// =============================================================================

/// Optimized set for vertex UUID collections in geometric predicates.
/// Used for fast membership testing during facet analysis.
///
/// # Optimization Rationale
///
/// - **Hash Function**: `FastHasher` for fast UUID hashing
/// - **Use Case**: Membership testing, intersection operations
/// - **Performance**: ~2-3x faster than `std::collections::HashSet`
pub type VertexUuidSet = FastHashSet<Uuid>;

// =============================================================================
// UUID-KEY MAPPING TYPES
// =============================================================================

/// Optimized mapping from vertex [`Uuid`] values to [`VertexKey`] values for fast UUID → key lookups.
/// This is the primary direction for most triangulation operations.
///
/// # Optimization Rationale
///
/// - **Primary Direction**: UUID → Key is the hot path in most algorithms
/// - **Hash Function**: `FastHasher` provides ~2-3x faster lookups than default hasher in typical non-adversarial workloads
/// - **Use Case**: Converting vertex UUIDs to keys for slot-map access
/// - **Performance**: O(1) average case, optimized for triangulation algorithms
///
/// # Reverse Lookups
///
/// For key → UUID lookups (less common), use direct topology access:
/// ```rust
/// use delaunay::prelude::*;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ReverseLookupExampleError {
/// #     #[error(transparent)]
/// #     Construction {
/// #         #[from]
/// #         source: DelaunayTriangulationConstructionError,
/// #     },
/// #     #[error("expected at least one vertex in the triangulation")]
/// #     MissingVertex,
/// #     #[error("vertex key should resolve in the triangulation")]
/// #     UnresolvedVertexKey,
/// # }
/// # fn main() -> Result<(), ReverseLookupExampleError> {
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt: DelaunayTriangulation<_, _, _, 3> = DelaunayTriangulationBuilder::new(&vertices)
///     .topology_guarantee(TopologyGuarantee::PLManifold)
///     .build::<()>()?;
/// println!("Topology guarantee: {:?}", dt.topology_guarantee());
/// let tds = dt.tds();
///
/// // Get first vertex key and its UUID
/// let Some((vertex_key, _)) = tds.vertices().next() else {
///     return Err(ReverseLookupExampleError::MissingVertex);
/// };
/// let Some(vertex) = tds.vertex(vertex_key) else {
///     return Err(ReverseLookupExampleError::UnresolvedVertexKey);
/// };
/// let vertex_uuid = vertex.uuid();
/// # Ok(())
/// # }
/// ```
pub type UuidToVertexKeyMap = FastHashMap<Uuid, VertexKey>;

/// Optimized mapping from simplex [`Uuid`] values to [`SimplexKey`] values for fast UUID → key lookups.
/// This is the primary direction for most triangulation operations.
///
/// # Optimization Rationale
///
/// - **Primary Direction**: UUID → Key is the hot path in neighbor assignment
/// - **Hash Function**: `FastHasher` provides ~2-3x faster lookups than default hasher in typical non-adversarial workloads
/// - **Use Case**: Converting simplex UUIDs to keys for slot-map access
/// - **Performance**: O(1) average case, optimized for triangulation algorithms
///
/// # Reverse Lookups
///
/// For key → UUID lookups (less common), use direct topology access:
/// ```rust
/// use delaunay::prelude::*;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ReverseLookupExampleError {
/// #     #[error(transparent)]
/// #     Construction {
/// #         #[from]
/// #         source: DelaunayTriangulationConstructionError,
/// #     },
/// #     #[error("expected at least one simplex in the triangulation")]
/// #     MissingSimplex,
/// #     #[error("simplex key should resolve in the triangulation")]
/// #     UnresolvedSimplexKey,
/// # }
/// # fn main() -> Result<(), ReverseLookupExampleError> {
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt: DelaunayTriangulation<_, _, _, 3> = DelaunayTriangulationBuilder::new(&vertices)
///     .topology_guarantee(TopologyGuarantee::PLManifold)
///     .build::<()>()?;
/// assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);
/// let tds = dt.tds();
///
/// // Get first simplex key and its UUID
/// let Some((simplex_key, _)) = tds.simplices().next() else {
///     return Err(ReverseLookupExampleError::MissingSimplex);
/// };
/// let Some(simplex) = tds.simplex(simplex_key) else {
///     return Err(ReverseLookupExampleError::UnresolvedSimplexKey);
/// };
/// let simplex_uuid = simplex.uuid();
/// assert_eq!(tds.simplex_key_from_uuid(&simplex_uuid), Some(simplex_key));
/// # Ok(())
/// # }
/// ```
pub type UuidToSimplexKeyMap = FastHashMap<Uuid, SimplexKey>;

// =============================================================================
// KEY-BASED INTERNAL TYPES
// =============================================================================

/// Optimized set for `SimplexKey` collections in internal operations.
///
/// This eliminates UUID dependencies in internal algorithms by working directly with `SlotMap` keys.
/// Provides the same performance benefits as `FastHashSet` but for direct key operations.
///
/// # Performance Benefits
///
/// - **Avoids UUID→Key lookups**: Eliminates extra hash table lookups vs UUID→Key mapping
/// - **Direct `SlotMap` compatibility**: Keys can be used directly for data structure access
/// - **Memory efficiency**: `SimplexKey` is typically smaller than `Uuid` (8 bytes vs 16 bytes)
/// - **Cache friendly**: Better memory locality for key-based algorithms
///
/// # Use Cases
///
/// - Internal simplex tracking during algorithms
/// - Validation operations that work with simplex keys
/// - Temporary simplex collections in geometric operations
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::collections::SimplexKeySet;
///
/// let simplex_set: SimplexKeySet = SimplexKeySet::default();
/// assert!(simplex_set.is_empty());
/// ```
pub type SimplexKeySet = FastHashSet<SimplexKey>;

/// Optimized set for `VertexKey` collections in internal operations.
///
/// This eliminates UUID dependencies in internal algorithms by working directly with `SlotMap` keys.
/// Provides the same performance benefits as `FastHashSet` but for direct key operations.
///
/// # Performance Benefits
///
/// - **Avoids UUID→Key lookups**: Eliminates extra hash table lookups vs UUID→Key mapping
/// - **Direct `SlotMap` compatibility**: Keys can be used directly for data structure access
/// - **Memory efficiency**: `VertexKey` is typically smaller than `Uuid` (8 bytes vs 16 bytes)
/// - **Cache friendly**: Better memory locality for key-based algorithms
///
/// # Use Cases
///
/// - Internal vertex tracking during algorithms
/// - Validation operations that work with vertex keys
/// - Temporary vertex collections in geometric operations
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::collections::VertexKeySet;
///
/// let vertex_set: VertexKeySet = VertexKeySet::default();
/// assert!(vertex_set.is_empty());
/// ```
pub type VertexKeySet = FastHashSet<VertexKey>;

/// Key-based mapping for internal simplex operations.
///
/// This provides direct `SimplexKey` → Value mapping without requiring UUID lookups,
/// optimizing internal algorithms that work with simplex keys.
///
/// # Performance Benefits
///
/// - **Direct key access**: No intermediate UUID→Key mapping required
/// - **`SlotMap` integration**: Keys align perfectly with internal data structure access patterns
/// - **Memory efficiency**: Avoids storing redundant UUID→Key associations
/// - **Algorithm optimization**: Direct key operations eliminate extra lookups in hot paths
///
/// # Use Cases
///
/// - Internal simplex metadata storage
/// - Algorithm state tracking per simplex
/// - Temporary mappings during geometric operations
/// - Validation data associated with specific simplices
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::collections::KeyBasedSimplexMap;
///
/// let simplex_map: KeyBasedSimplexMap<f64> = KeyBasedSimplexMap::default();
/// assert!(simplex_map.is_empty());
/// ```
pub type KeyBasedSimplexMap<V> = FastHashMap<SimplexKey, V>;

/// Key-based mapping for internal vertex operations.
///
/// This provides direct `VertexKey` → Value mapping without requiring UUID lookups,
/// optimizing internal algorithms that work with vertex keys.
///
/// # Performance Benefits
///
/// - **Direct key access**: No intermediate UUID→Key mapping required
/// - **`SlotMap` integration**: Keys align perfectly with internal data structure access patterns
/// - **Memory efficiency**: Avoids storing redundant UUID→Key associations
/// - **Algorithm optimization**: Direct key operations eliminate extra lookups in hot paths
///
/// # Use Cases
///
/// - Internal vertex metadata storage
/// - Algorithm state tracking per vertex
/// - Temporary mappings during geometric operations
/// - Validation data associated with specific vertices
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::collections::KeyBasedVertexMap;
///
/// let vertex_map: KeyBasedVertexMap<i32> = KeyBasedVertexMap::default();
/// assert!(vertex_map.is_empty());
/// ```
pub type KeyBasedVertexMap<V> = FastHashMap<VertexKey, V>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tds::{SimplexKey, VertexKey};
    use slotmap::SlotMap;

    #[test]
    fn test_phase1_key_based_roundtrip_operations() {
        // Create mock SlotMaps to generate real keys for testing
        let mut simplex_slots: SlotMap<SimplexKey, i32> = SlotMap::default();
        let mut vertex_slots: SlotMap<VertexKey, i32> = SlotMap::default();

        // Insert some dummy data to get real keys
        let simplex_key1 = simplex_slots.insert(1);
        let simplex_key2 = simplex_slots.insert(2);
        let vertex_key1 = vertex_slots.insert(1);
        let vertex_key2 = vertex_slots.insert(2);

        // Test SimplexKeySet insert/contains roundtrip
        let mut simplex_set: SimplexKeySet = SimplexKeySet::default();
        assert!(!simplex_set.contains(&simplex_key1));
        simplex_set.insert(simplex_key1);
        assert!(simplex_set.contains(&simplex_key1));
        assert!(!simplex_set.contains(&simplex_key2));

        // Test VertexKeySet insert/contains roundtrip
        let mut vertex_set: VertexKeySet = VertexKeySet::default();
        assert!(!vertex_set.contains(&vertex_key1));
        vertex_set.insert(vertex_key1);
        assert!(vertex_set.contains(&vertex_key1));
        assert!(!vertex_set.contains(&vertex_key2));

        // Test KeyBasedSimplexMap insert/get roundtrip
        let mut simplex_map: KeyBasedSimplexMap<String> = KeyBasedSimplexMap::default();
        assert_eq!(simplex_map.get(&simplex_key1), None);
        simplex_map.insert(simplex_key1, "simplex_data".to_string());
        assert_eq!(
            simplex_map.get(&simplex_key1).map(String::as_str),
            Some("simplex_data")
        );
        assert_eq!(simplex_map.get(&simplex_key2), None);

        // Test KeyBasedVertexMap insert/get roundtrip
        let mut vertex_map: KeyBasedVertexMap<i32> = KeyBasedVertexMap::default();
        assert_eq!(vertex_map.get(&vertex_key1), None);
        vertex_map.insert(vertex_key1, 42);
        assert_eq!(vertex_map.get(&vertex_key1).copied(), Some(42));
        assert_eq!(vertex_map.get(&vertex_key2), None);

        // Test that collections have expected sizes
        assert_eq!(simplex_set.len(), 1);
        assert_eq!(vertex_set.len(), 1);
        assert_eq!(simplex_map.len(), 1);
        assert_eq!(vertex_map.len(), 1);
    }

    #[test]
    fn test_key_based_types_compile_and_instantiate() {
        let _simplex_set: SimplexKeySet = SimplexKeySet::default();
        let _vertex_set: VertexKeySet = VertexKeySet::default();
        let _simplex_map: KeyBasedSimplexMap<i32> = KeyBasedSimplexMap::default();
        let _vertex_map: KeyBasedVertexMap<String> = KeyBasedVertexMap::default();

        // Basic operations should work
        let simplex_set: SimplexKeySet = SimplexKeySet::default();
        assert!(simplex_set.is_empty());
        assert_eq!(simplex_set.len(), 0);

        let simplex_map: KeyBasedSimplexMap<f64> = KeyBasedSimplexMap::default();
        assert!(simplex_map.is_empty());
        assert_eq!(simplex_map.len(), 0);
    }
}
