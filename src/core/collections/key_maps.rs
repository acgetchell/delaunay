use super::{FastHashMap, FastHashSet, Uuid};
use crate::core::triangulation_data_structure::{CellKey, VertexKey};

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

/// Optimized mapping from Vertex UUIDs to `VertexKeys` for fast UUID → Key lookups.
/// This is the primary direction for most triangulation operations.
///
/// # Optimization Rationale
///
/// - **Primary Direction**: UUID → Key is the hot path in most algorithms
/// - **Hash Function**: `FastHasher` provides ~2-3x faster lookups than default hasher in typical non-adversarial workloads
/// - **Use Case**: Converting vertex UUIDs to keys for `SlotMap` access
/// - **Performance**: O(1) average case, optimized for triangulation algorithms
///
/// # Reverse Lookups
///
/// For Key → UUID lookups (less common), use direct `SlotMap` access:
/// ```rust
/// use delaunay::prelude::triangulation::*;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt: DelaunayTriangulation<_, _, _, 3> =
///     DelaunayTriangulation::new_with_topology_guarantee(
///         &vertices,
///         TopologyGuarantee::PLManifold,
///     )
///     .unwrap();
/// println!("Topology guarantee: {:?}", dt.topology_guarantee());
/// let tds = dt.tds();
///
/// // Get first vertex key and its UUID
/// let (vertex_key, _) = tds.vertices().next().unwrap();
/// let vertex_uuid = tds.get_vertex_by_key(vertex_key).unwrap().uuid();
/// ```
pub type UuidToVertexKeyMap = FastHashMap<Uuid, VertexKey>;

/// Optimized mapping from Cell UUIDs to `CellKeys` for fast UUID → Key lookups.
/// This is the primary direction for most triangulation operations.
///
/// # Optimization Rationale
///
/// - **Primary Direction**: UUID → Key is the hot path in neighbor assignment
/// - **Hash Function**: `FastHasher` provides ~2-3x faster lookups than default hasher in typical non-adversarial workloads
/// - **Use Case**: Converting cell UUIDs to keys for `SlotMap` access
/// - **Performance**: O(1) average case, optimized for triangulation algorithms
///
/// # Reverse Lookups
///
/// For Key → UUID lookups (less common), use direct `SlotMap` access:
/// ```rust
/// use delaunay::prelude::triangulation::*;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt: DelaunayTriangulation<_, _, _, 3> =
///     DelaunayTriangulation::new_with_topology_guarantee(
///         &vertices,
///         TopologyGuarantee::PLManifold,
///     )
///     .unwrap();
/// println!("Topology guarantee: {:?}", dt.topology_guarantee());
/// let tds = dt.tds();
///
/// // Get first cell key and its UUID
/// let (cell_key, _) = tds.cells().next().unwrap();
/// let cell_uuid = tds.get_cell(cell_key).unwrap().uuid();
/// ```
pub type UuidToCellKeyMap = FastHashMap<Uuid, CellKey>;

// =============================================================================
// PHASE 1 MIGRATION: KEY-BASED INTERNAL TYPES
// =============================================================================

/// **Phase 1 Migration**: Optimized set for `CellKey` collections in internal operations.
///
/// This eliminates UUID dependencies in internal algorithms by working directly with `SlotMap` keys.
/// Provides the same performance benefits as `FastHashSet` but for direct key operations.
///
/// # Performance Benefits
///
/// - **Avoids UUID→Key lookups**: Eliminates extra hash table lookups vs UUID→Key mapping
/// - **Direct `SlotMap` compatibility**: Keys can be used directly for data structure access
/// - **Memory efficiency**: `CellKey` is typically smaller than `Uuid` (8 bytes vs 16 bytes)
/// - **Cache friendly**: Better memory locality for key-based algorithms
///
/// # Use Cases
///
/// - Internal cell tracking during algorithms
/// - Validation operations that work with cell keys
/// - Temporary cell collections in geometric operations
///
/// # Examples
///
/// ```rust
/// use delaunay::core::collections::CellKeySet;
///
/// let cell_set: CellKeySet = CellKeySet::default();
/// assert!(cell_set.is_empty());
/// ```
pub type CellKeySet = FastHashSet<CellKey>;

/// **Phase 1 Migration**: Optimized set for `VertexKey` collections in internal operations.
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
/// use delaunay::core::collections::VertexKeySet;
///
/// let vertex_set: VertexKeySet = VertexKeySet::default();
/// assert!(vertex_set.is_empty());
/// ```
pub type VertexKeySet = FastHashSet<VertexKey>;

/// **Phase 1 Migration**: Key-based mapping for internal cell operations.
///
/// This provides direct `CellKey` → Value mapping without requiring UUID lookups,
/// optimizing internal algorithms that work with cell keys.
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
/// - Internal cell metadata storage
/// - Algorithm state tracking per cell
/// - Temporary mappings during geometric operations
/// - Validation data associated with specific cells
///
/// # Examples
///
/// ```rust
/// use delaunay::core::collections::KeyBasedCellMap;
///
/// let cell_map: KeyBasedCellMap<f64> = KeyBasedCellMap::default();
/// assert!(cell_map.is_empty());
/// ```
pub type KeyBasedCellMap<V> = FastHashMap<CellKey, V>;

/// **Phase 1 Migration**: Key-based mapping for internal vertex operations.
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
/// use delaunay::core::collections::KeyBasedVertexMap;
///
/// let vertex_map: KeyBasedVertexMap<i32> = KeyBasedVertexMap::default();
/// assert!(vertex_map.is_empty());
/// ```
pub type KeyBasedVertexMap<V> = FastHashMap<VertexKey, V>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::triangulation_data_structure::{CellKey, VertexKey};
    use slotmap::SlotMap;

    #[test]
    fn test_phase1_key_based_roundtrip_operations() {
        // Create mock SlotMaps to generate real keys for testing
        let mut cell_slots: SlotMap<CellKey, i32> = SlotMap::default();
        let mut vertex_slots: SlotMap<VertexKey, i32> = SlotMap::default();

        // Insert some dummy data to get real keys
        let cell_key1 = cell_slots.insert(1);
        let cell_key2 = cell_slots.insert(2);
        let vertex_key1 = vertex_slots.insert(1);
        let vertex_key2 = vertex_slots.insert(2);

        // Test CellKeySet insert/contains roundtrip
        let mut cell_set: CellKeySet = CellKeySet::default();
        assert!(!cell_set.contains(&cell_key1));
        cell_set.insert(cell_key1);
        assert!(cell_set.contains(&cell_key1));
        assert!(!cell_set.contains(&cell_key2));

        // Test VertexKeySet insert/contains roundtrip
        let mut vertex_set: VertexKeySet = VertexKeySet::default();
        assert!(!vertex_set.contains(&vertex_key1));
        vertex_set.insert(vertex_key1);
        assert!(vertex_set.contains(&vertex_key1));
        assert!(!vertex_set.contains(&vertex_key2));

        // Test KeyBasedCellMap insert/get roundtrip
        let mut cell_map: KeyBasedCellMap<String> = KeyBasedCellMap::default();
        assert_eq!(cell_map.get(&cell_key1), None);
        cell_map.insert(cell_key1, "cell_data".to_string());
        assert_eq!(
            cell_map.get(&cell_key1).map(String::as_str),
            Some("cell_data")
        );
        assert_eq!(cell_map.get(&cell_key2), None);

        // Test KeyBasedVertexMap insert/get roundtrip
        let mut vertex_map: KeyBasedVertexMap<i32> = KeyBasedVertexMap::default();
        assert_eq!(vertex_map.get(&vertex_key1), None);
        vertex_map.insert(vertex_key1, 42);
        assert_eq!(vertex_map.get(&vertex_key1).copied(), Some(42));
        assert_eq!(vertex_map.get(&vertex_key2), None);

        // Test that collections have expected sizes
        assert_eq!(cell_set.len(), 1);
        assert_eq!(vertex_set.len(), 1);
        assert_eq!(cell_map.len(), 1);
        assert_eq!(vertex_map.len(), 1);
    }

    #[test]
    fn test_key_based_types_compile_and_instantiate() {
        let _cell_set: CellKeySet = CellKeySet::default();
        let _vertex_set: VertexKeySet = VertexKeySet::default();
        let _cell_map: KeyBasedCellMap<i32> = KeyBasedCellMap::default();
        let _vertex_map: KeyBasedVertexMap<String> = KeyBasedVertexMap::default();

        // Basic operations should work
        let cell_set: CellKeySet = CellKeySet::default();
        assert!(cell_set.is_empty());
        assert_eq!(cell_set.len(), 0);

        let cell_map: KeyBasedCellMap<f64> = KeyBasedCellMap::default();
        assert!(cell_map.is_empty());
        assert_eq!(cell_map.len(), 0);
    }
}
