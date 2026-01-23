use super::{FastBuildHasher, FastHashMap, FastHashSet, SmallBuffer};

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
/// use delaunay::core::collections::fast_hash_map_with_capacity;
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
/// use delaunay::core::collections::fast_hash_set_with_capacity;
/// use uuid::Uuid;
///
/// let set = fast_hash_set_with_capacity::<Uuid>(500);
/// // Can insert up to ~375 UUIDs without rehashing
/// ```
///
/// **Phase 1**: Internal operations (key-based for better performance):
/// ```rust
/// use delaunay::core::collections::fast_hash_set_with_capacity;
/// use delaunay::core::triangulation_data_structure::CellKey;
///
/// let set = fast_hash_set_with_capacity::<CellKey>(500);
/// // Can insert up to ~375 CellKeys without rehashing, avoids UUID→Key lookups
/// ```
#[inline]
#[must_use]
pub fn fast_hash_set_with_capacity<T>(capacity: usize) -> FastHashSet<T> {
    FastHashSet::with_capacity_and_hasher(capacity, FastBuildHasher::default())
}

/// Creates a `SmallBuffer` with the specified capacity.
/// Uses stack allocation if the capacity is within the inline size, otherwise uses heap.
///
/// Note: This function is only available for specific sizes due to `SmallVec`'s Array trait constraints.
/// For most use cases, prefer using `SmallBuffer::with_capacity(capacity)` directly with concrete types.
///
/// # Performance Benefits
///
/// - **Smart Allocation**: Uses stack when possible, heap when necessary
/// - **Capacity Hinting**: Pre-allocates heap space if needed
/// - **Zero Overhead**: No cost when staying within inline capacity
///
/// # Examples
///
/// ```rust
/// use delaunay::core::collections::SmallBuffer;
///
/// // Use concrete types directly (preferred)
/// let mut small_buf: SmallBuffer<i32, 8> = SmallBuffer::with_capacity(5);
/// let mut large_buf: SmallBuffer<i32, 8> = SmallBuffer::with_capacity(20);
/// ```
#[must_use]
pub fn small_buffer_with_capacity_8<T>(capacity: usize) -> SmallBuffer<T, 8> {
    SmallBuffer::with_capacity(capacity)
}

/// Creates a small buffer optimized for 2 elements (common facet sharing pattern)
#[must_use]
pub fn small_buffer_with_capacity_2<T>(capacity: usize) -> SmallBuffer<T, 2> {
    SmallBuffer::with_capacity(capacity)
}

/// Creates a small buffer optimized for 16 elements (larger batch operations)
#[must_use]
pub fn small_buffer_with_capacity_16<T>(capacity: usize) -> SmallBuffer<T, 16> {
    SmallBuffer::with_capacity(capacity)
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

        // Test small buffer capacity helpers with spill validation
        let mut buffer_8 = small_buffer_with_capacity_8::<i32>(5);
        assert!(buffer_8.capacity() >= 5);
        // Force growth beyond inline 8 to validate spill
        for i in 0..9 {
            buffer_8.push(i);
        }
        assert!(buffer_8.spilled());

        // Test small_buffer_with_capacity_2 with spill validation
        let mut buffer_2 = small_buffer_with_capacity_2::<i32>(10);
        assert!(buffer_2.capacity() >= 10);
        buffer_2.extend(0..3); // > inline(2) -> heap
        assert!(buffer_2.spilled());

        // Test small_buffer_with_capacity_16 with spill validation
        let mut buffer_16 = small_buffer_with_capacity_16::<String>(25);
        assert!(buffer_16.capacity() >= 25);
        buffer_16.extend(std::iter::repeat_n(String::new(), 17)); // > inline(16)
        assert!(buffer_16.spilled());

        // Test different types work correctly
        let mut test_buffer2: SmallBuffer<f64, 2> = small_buffer_with_capacity_2(3);
        test_buffer2.push(1.0);
        test_buffer2.push(2.0);
        assert_eq!(test_buffer2.len(), 2);

        let mut test_buffer16: SmallBuffer<char, 16> = small_buffer_with_capacity_16(5);
        test_buffer16.push('a');
        test_buffer16.push('b');
        assert_eq!(test_buffer16.len(), 2);

        // Test zero capacity edge case
        let _buffer2_zero = small_buffer_with_capacity_2::<u8>(0);
        let _buffer16_zero = small_buffer_with_capacity_16::<u32>(0);
    }
}
