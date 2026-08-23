//! Constructors for optimized hash maps, sets, and small buffers.
//!
//! These helpers keep allocation and hasher choices explicit at call sites without
//! repeating the concrete collection aliases throughout the codebase.

use super::{FastBuildHasher, FastHashMap, FastHashSet};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/// Creates a `FastHashMap` with pre-allocated capacity using the optimal hasher.
/// This is more efficient than using the default constructor when the expected size is known.
///
/// # Performance Benefits
///
/// - **Pre-allocation**: Avoids rehashing during insertion
/// - **Optimal Hasher**: Uses `FastHasher` for maximum performance
/// - **Memory Efficiency**: Reduces memory fragmentation
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::collections::fast_hash_map_with_capacity;
///
/// let map = fast_hash_map_with_capacity::<u64, usize>(1000);
/// // Can insert up to ~750 items without rehashing (load factor ~0.75)
/// ```
#[inline]
#[must_use]
pub fn fast_hash_map_with_capacity<K, V>(capacity: usize) -> FastHashMap<K, V> {
    FastHashMap::with_capacity_and_hasher(capacity, FastBuildHasher::default())
}

/// Creates a `FastHashSet` with pre-allocated capacity using the optimal hasher.
/// This is more efficient than using the default constructor when the expected size is known.
///
/// # Performance Benefits
///
/// - **Pre-allocation**: Avoids rehashing during insertion
/// - **Optimal Hasher**: Uses `FastHasher` for maximum performance
/// - **Memory Efficiency**: Reduces memory fragmentation
///
/// # Examples
///
/// External API usage (UUID-based):
/// ```rust
/// use delaunay::prelude::collections::fast_hash_set_with_capacity;
/// use uuid::Uuid;
///
/// let set = fast_hash_set_with_capacity::<Uuid>(500);
/// // Can insert up to ~375 UUIDs without rehashing
/// ```
///
/// Internal operations (key-based for better performance):
/// ```rust
/// use delaunay::prelude::collections::fast_hash_set_with_capacity;
/// use delaunay::prelude::tds::SimplexKey;
///
/// let set = fast_hash_set_with_capacity::<SimplexKey>(500);
/// // Can insert up to ~375 SimplexKeys without rehashing, avoids UUID→Key lookups
/// ```
#[inline]
#[must_use]
pub fn fast_hash_set_with_capacity<T>(capacity: usize) -> FastHashSet<T> {
    FastHashSet::with_capacity_and_hasher(capacity, FastBuildHasher::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capacity_helpers() {
        // Test hash map and set capacity helpers
        let map = fast_hash_map_with_capacity::<u64, usize>(100);
        assert!(map.capacity() >= 100);

        let set = fast_hash_set_with_capacity::<u64>(50);
        assert!(set.capacity() >= 50);
    }
}
