use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet, FxHasher};
use smallvec::SmallVec;

// Import slotmap types for storage backend
#[cfg(not(feature = "dense-slotmap"))]
use slotmap::SlotMap;

#[cfg(feature = "dense-slotmap")]
use slotmap::DenseSlotMap;

/// Compact index type for facet positions within a cell.
///
/// Since a D-dimensional cell has D+1 facets, and practical triangulations work with D ≤ 255,
/// a `u8` provides sufficient range while minimizing memory usage.
///
/// # Range
///
/// - **Valid range**: 0..=D for a D-dimensional triangulation
/// - **Maximum supported**: D ≤ 255 (which covers all practical applications)
///
/// # Performance Benefits
///
/// - **Smaller tuples**: `(CellKey, FacetIndex)` uses less memory than `(CellKey, usize)`
/// - **Better cache density**: More facet mappings fit in cache lines
/// - **Reduced memory bandwidth**: Faster iteration over facet collections
///
/// # Examples
///
/// ```rust
/// use delaunay::core::collections::FacetIndex;
///
/// // 3D triangulation: facets 0, 1, 2, 3 (fits comfortably in u8)
/// let facet: FacetIndex = 2;
/// assert_eq!(usize::from(facet), 2);
/// ```
pub type FacetIndex = u8;

// Re-export UUID for convenience in type aliases
pub use uuid::Uuid;

// =============================================================================
// STORAGE BACKEND
// =============================================================================

/// Internal storage backend for triangulation data structures.
///
/// This type alias abstracts over the concrete storage implementation,
/// allowing the choice between `DenseSlotMap` (**default**) and `SlotMap`
/// (when built with `--no-default-features`) without exposing the choice
/// in public APIs.
///
/// # Feature Flags
///
/// - **default**: Uses `DenseSlotMap` (enabled via the default `dense-slotmap` feature)
/// - **--no-default-features**: Uses `SlotMap` for comparison and experimentation
/// - **dense-slotmap**: Explicitly selects `DenseSlotMap` (redundant with defaults)
///
/// # Internal Use Only
///
/// This type should not be exposed in public API signatures. Instead,
/// public methods should return iterators or use other abstractions
/// that hide the concrete storage backend.
///
/// # Examples
///
/// ```rust,ignore
/// // Internal use - not exposed in public API
/// let vertices: StorageMap<VertexKey, Vertex<f64, (), 3>> = StorageMap::with_key();
/// ```
#[cfg(not(feature = "dense-slotmap"))]
pub type StorageMap<K, V> = SlotMap<K, V>;

#[cfg(feature = "dense-slotmap")]
pub type StorageMap<K, V> = DenseSlotMap<K, V>;

// =============================================================================
// CORE OPTIMIZED TYPES
// =============================================================================

/// Optimized `HashMap` type for performance-critical operations.
/// Uses `FastHasher` (`rustc_hash::FxHasher`) for faster hashing in non-cryptographic contexts.
///
/// # Performance Characteristics
///
/// - **Hash Function**: `FastHasher` (non-cryptographic, very fast)
/// - **Use Case**: Internal mappings where security is not a concern
/// - **Speedup**: ~2-3x faster than `std::collections::HashMap` in typical non-adversarial workloads
///
/// # Security Warning
///
/// ⚠️ **Not DoS-resistant**: Do not use with attacker-controlled keys.
/// Use only with trusted, internal data to avoid hash collision attacks.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::collections::FastHashMap;
///
/// let mut map: FastHashMap<u64, usize> = FastHashMap::default();
/// map.insert(123, 456);
/// ```
pub type FastHashMap<K, V> = FxHashMap<K, V>;

/// Fast non-cryptographic hasher alias for internal collections.
///
/// Wraps [`rustc_hash::FxHasher`] to ensure consistent hashing behavior
/// across [`FastHashMap`] and [`FastHashSet`].
pub type FastHasher = FxHasher;

/// Build hasher that instantiates [`FastHasher`].
///
/// Used by helpers that configure [`FastHashMap`]
/// and [`FastHashSet`] with the optimized hashing strategy.
pub type FastBuildHasher = FxBuildHasher;

/// Re-export the Entry enum for `FastHashMap`.
/// This provides the Entry API for efficient check-and-insert operations.
/// Since `FxHashMap` uses `std::collections::hash_map::Entry`, we re-export that.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::collections::{Entry, FastHashMap};
///
/// let mut map: FastHashMap<String, String> = FastHashMap::default();
/// match map.entry("key".to_string()) {
///     Entry::Occupied(e) => println!("Already exists: {:?}", e.get()),
///     Entry::Vacant(e) => {
///         e.insert("value".to_string());
///     }
/// }
/// ```
pub use std::collections::hash_map::Entry;

/// Optimized `HashSet` type for performance-critical operations.
/// Uses `FastHasher` (`rustc_hash::FxHasher`) for faster hashing in non-cryptographic contexts.
///
/// # Performance Characteristics
///
/// - **Hash Function**: `FastHasher` (non-cryptographic, very fast)
/// - **Use Case**: Internal sets for membership testing
/// - **Speedup**: ~2-3x faster than `std::collections::HashSet` in typical non-adversarial workloads
///
/// # Security Warning
///
/// ⚠️ **Not DoS-resistant**: Do not use with attacker-controlled keys.
/// Use only with trusted, internal data to avoid hash collision attacks.
///
/// # Examples
///
/// External API usage (UUID-based for compatibility):
/// ```rust
/// use delaunay::core::collections::FastHashSet;
/// use uuid::Uuid;
///
/// let mut set: FastHashSet<Uuid> = FastHashSet::default();
/// set.insert(Uuid::new_v4());
/// ```
///
/// **Phase 1**: Internal operations (key-based for performance):
/// ```rust
/// use delaunay::core::collections::{CellKeySet, FastHashSet};
/// use delaunay::core::triangulation_data_structure::CellKey;
///
/// // For internal algorithms, prefer direct key-based collections
/// let mut internal_set: CellKeySet = CellKeySet::default();
/// // internal_set.insert(cell_key); // Avoids extra UUID→Key lookups
/// ```
pub type FastHashSet<T> = FxHashSet<T>;

/// Small-optimized Vec that uses stack allocation for small collections.
/// Generic size parameter allows customization per use case.
/// Provides heap fallback for larger collections.
///
/// # Performance Characteristics
///
/// - **Stack Allocation**: For collections ≤ N elements
/// - **Heap Fallback**: Automatically grows to heap when needed
/// - **Cache Friendly**: Better memory locality for small collections
/// - **Zero-cost**: No overhead when staying within inline capacity
///
/// # Size Guidelines
///
/// - **N=2**: Facet sharing patterns (1-2 cells per facet)
/// - **N=4**: Small temporary operations
/// - **N=8**: Typical vertex/cell degrees
/// - **N=16**: Batch operation buffers
///
/// # Examples
///
/// ```rust
/// use delaunay::core::collections::SmallBuffer;
///
/// // Stack-allocated for ≤8 elements, heap for more
/// let mut buffer: SmallBuffer<i32, 8> = SmallBuffer::new();
/// for i in 0..5 {
///     buffer.push(i); // All stack allocated
/// }
/// ```
pub type SmallBuffer<T, const N: usize> = SmallVec<[T; N]>;

// =============================================================================
// SEMANTIC SIZE CONSTANTS AND TYPE ALIASES
// =============================================================================

/// Semantic constant for the maximum practical dimension in computational geometry.
///
/// Most applications work with dimensions 2D-5D, so 8 provides comfortable headroom
/// while keeping stack allocation efficient.
pub const MAX_PRACTICAL_DIMENSION_SIZE: usize = 8;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_collections_basic_operations() {
        // Test FastHashMap basic operations
        let mut map: FastHashMap<u64, usize> = FastHashMap::default();
        assert!(map.is_empty());

        map.insert(123, 456);
        assert_eq!(map.get(&123), Some(&456));
        assert_eq!(map.len(), 1);

        map.insert(789, 101_112);
        assert_eq!(map.len(), 2);

        // Test FastHashSet basic operations
        let mut set: FastHashSet<u64> = FastHashSet::default();
        assert!(set.is_empty());

        set.insert(789);
        assert!(set.contains(&789));
        assert_eq!(set.len(), 1);

        set.insert(456);
        assert_eq!(set.len(), 2);
        assert!(set.contains(&456));
        assert!(!set.contains(&999));
    }

    #[test]
    fn test_small_buffer_stack_allocation() {
        let mut buffer: SmallBuffer<i32, 4> = SmallBuffer::new();

        // These should use stack allocation
        for i in 0..4 {
            buffer.push(i);
        }
        assert_eq!(buffer.len(), 4);
        assert!(!buffer.spilled()); // Still on stack

        // This should trigger heap allocation
        buffer.push(4);
        assert_eq!(buffer.len(), 5);
        assert!(buffer.spilled()); // Now on heap
    }
}
