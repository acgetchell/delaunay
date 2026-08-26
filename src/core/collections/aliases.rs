//! Core collection aliases used throughout triangulation storage and algorithms.
//!
//! These aliases centralize the crate's hasher, slotmap, and small-buffer choices so
//! public APIs and internal algorithms use consistent collection types.

use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet, FxHasher};
use slotmap::{DenseSlotMap, Key};
use smallvec::SmallVec;

/// Compact index type for facet positions within a simplex.
///
/// Since a D-dimensional simplex has D+1 facets, and practical triangulations work with D ≤ 255,
/// a `u8` provides sufficient range while minimizing memory usage.
///
/// # Range
///
/// - **Valid range**: 0..=D for a D-dimensional triangulation
/// - **Maximum supported**: D ≤ 255 (which covers all practical applications)
///
/// # Performance Benefits
///
/// - **Smaller tuples**: `(SimplexKey, FacetIndex)` uses less memory than `(SimplexKey, usize)`
/// - **Better cache density**: More facet mappings fit in cache lines
/// - **Reduced memory bandwidth**: Faster iteration over facet collections
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::collections::FacetIndex;
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
/// This type alias keeps the concrete storage implementation out of public
/// API signatures while using `DenseSlotMap` unconditionally for construction
/// and iteration locality.
///
/// # Internal Use Only
///
/// This type should not be exposed in public API signatures. Instead,
/// public methods should return iterators or use other abstractions
/// that hide the concrete storage backend.
///
/// # Examples
///
/// ```rust,compile_fail
/// // Internal use only: `StorageMap` is intentionally not exported publicly.
/// use delaunay::prelude::collections::StorageMap;
/// ```
#[derive(Clone, Debug)]
pub struct StorageMap<K: Key, V> {
    slots: DenseSlotMap<K, Option<V>>,
    live_len: usize,
}

/// Iterator over the live keys in a [`StorageMap`].
///
/// Transaction tombstones remain allocated in the underlying slot map so that
/// rollback can restore their exact generational keys. This iterator hides
/// those tombstones from ordinary topology traversal.
#[derive(Clone, Debug)]
pub(crate) struct StorageKeys<'a, K: Key, V> {
    inner: slotmap::dense::Iter<'a, K, Option<V>>,
}

impl<K: Key, V> Iterator for StorageKeys<'_, K, V> {
    type Item = K;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .find_map(|(key, value)| value.is_some().then_some(key))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.inner.size_hint();
        (0, upper)
    }
}

impl<K: Key, V> StorageMap<K, V> {
    /// Creates an empty keyed storage map.
    #[must_use]
    pub fn with_key() -> Self {
        Self {
            slots: DenseSlotMap::with_key(),
            live_len: 0,
        }
    }

    /// Creates an empty keyed storage map with capacity for `capacity` values.
    #[must_use]
    pub fn with_capacity_and_key(capacity: usize) -> Self {
        Self {
            slots: DenseSlotMap::with_capacity_and_key(capacity),
            live_len: 0,
        }
    }

    /// Inserts one live value and returns its stable generational key.
    pub fn insert(&mut self, value: V) -> K {
        let key = self.slots.insert(Some(value));
        self.live_len = self.live_len.saturating_add(1);
        key
    }

    /// Removes one live value and invalidates its key immediately.
    pub fn remove(&mut self, key: K) -> Option<V> {
        let value = self.slots.remove(key).flatten()?;
        self.live_len = self.live_len.saturating_sub(1);
        Some(value)
    }

    /// Temporarily removes a live value without invalidating its key.
    ///
    /// Rollback transactions use this to make the value absent from ordinary
    /// queries and validators while retaining an exact restoration slot. The
    /// caller must later call either [`Self::restore_tombstone`] or
    /// [`Self::finalize_tombstone`].
    pub(crate) fn tombstone(&mut self, key: K) -> Option<V> {
        let value = self.slots.get_mut(key)?.take()?;
        self.live_len = self.live_len.saturating_sub(1);
        Some(value)
    }

    /// Restores a transaction tombstone at its original generational key.
    pub(crate) fn restore_tombstone(&mut self, key: K, value: V) -> Result<(), V> {
        let Some(slot) = self.slots.get_mut(key) else {
            return Err(value);
        };
        if slot.is_some() {
            return Err(value);
        }
        *slot = Some(value);
        self.live_len = self.live_len.saturating_add(1);
        Ok(())
    }

    /// Commits a transaction tombstone and invalidates its key.
    pub(crate) fn finalize_tombstone(&mut self, key: K) -> bool {
        matches!(self.slots.remove(key), Some(None))
    }

    /// Returns a shared reference to one live value.
    #[must_use]
    pub fn get(&self, key: K) -> Option<&V> {
        self.slots.get(key).and_then(Option::as_ref)
    }

    /// Returns a mutable reference to one live value.
    #[must_use]
    pub fn get_mut(&mut self, key: K) -> Option<&mut V> {
        self.slots.get_mut(key).and_then(Option::as_mut)
    }

    /// Returns whether `key` identifies a live value.
    #[must_use]
    pub fn contains_key(&self, key: K) -> bool {
        self.get(key).is_some()
    }

    /// Returns the number of live values.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.live_len
    }

    /// Iterates over live key/value pairs.
    pub fn iter(&self) -> impl Iterator<Item = (K, &V)> {
        self.slots
            .iter()
            .filter_map(|(key, value)| value.as_ref().map(|value| (key, value)))
    }

    /// Iterates mutably over live key/value pairs.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (K, &mut V)> {
        self.slots
            .iter_mut()
            .filter_map(|(key, value)| value.as_mut().map(|value| (key, value)))
    }

    /// Iterates over live keys.
    pub(crate) fn keys(&self) -> StorageKeys<'_, K, V> {
        StorageKeys {
            inner: self.slots.iter(),
        }
    }

    /// Iterates over shared live values.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.iter().map(|(_, value)| value)
    }
}

impl<K: Key, V> Default for StorageMap<K, V> {
    fn default() -> Self {
        Self::with_key()
    }
}

#[cfg(test)]
mod test_support {
    use super::*;

    impl<K: Key, V> StorageMap<K, V> {
        /// Iterates over mutable live values for malformed-state fixtures.
        pub(crate) fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
            self.iter_mut().map(|(_, value)| value)
        }
    }
}

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
/// use delaunay::prelude::collections::FastHashMap;
///
/// let mut map: FastHashMap<u64, usize> = FastHashMap::default();
/// map.insert(123, 456);
/// ```
pub type FastHashMap<K, V> = FxHashMap<K, V>;

/// DoS-resistant [`HashMap`](std::collections::HashMap) for keys derived from caller-provided data.
///
/// Use this for hash maps whose keys are directly derived from public input
/// coordinates or other attacker-controlled values. It intentionally keeps
/// Rust's randomized [`std::collections::hash_map::RandomState`] hasher instead
/// of [`FastHasher`].
///
/// # Security
///
/// Prefer [`FastHashMap`] for trusted internal keys such as slotmap keys,
/// UUID-derived identities, or facet hashes built from slotmap keys. Use
/// [`SecureHashMap`] when a caller can influence map keys directly.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::collections::SecureHashMap;
///
/// let mut buckets: SecureHashMap<[u64; 2], usize> = SecureHashMap::default();
/// buckets.insert([12, 34], 1);
///
/// assert_eq!(buckets.get(&[12, 34]), Some(&1));
/// ```
pub type SecureHashMap<K, V> =
    std::collections::HashMap<K, V, std::collections::hash_map::RandomState>;

/// DoS-resistant [`HashSet`](std::collections::HashSet) for keys derived from caller-provided data.
///
/// Use this for sets whose keys are directly derived from public input
/// coordinates or other attacker-controlled values. It intentionally keeps
/// Rust's randomized [`std::collections::hash_map::RandomState`] hasher instead
/// of [`FastHasher`].
///
/// # Security
///
/// Prefer [`FastHashSet`] for trusted internal keys such as slotmap keys,
/// UUID-derived identities, or facet hashes built from slotmap keys. Use
/// [`SecureHashSet`] when a caller can influence set keys directly.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::collections::SecureHashSet;
///
/// let mut buckets: SecureHashSet<[u64; 2]> = SecureHashSet::default();
/// buckets.insert([12, 34]);
///
/// assert!(buckets.contains(&[12, 34]));
/// ```
pub type SecureHashSet<T> = std::collections::HashSet<T, std::collections::hash_map::RandomState>;

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
/// use delaunay::prelude::collections::{Entry, FastHashMap};
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
/// Use only with trusted, internal data to avoid hash collision attacks. Use
/// [`SecureHashSet`] when set keys are derived from public input.
///
/// # Examples
///
/// External API usage (UUID-based for compatibility):
/// ```rust
/// use delaunay::prelude::collections::FastHashSet;
/// use uuid::Uuid;
///
/// let mut set: FastHashSet<Uuid> = FastHashSet::default();
/// set.insert(Uuid::new_v4());
/// ```
///
/// Internal operations (key-based for performance):
/// ```rust
/// use delaunay::prelude::collections::{SimplexKeySet, FastHashSet};
/// use delaunay::prelude::tds::SimplexKey;
///
/// // For internal algorithms, prefer direct key-based collections
/// let mut internal_set: SimplexKeySet = SimplexKeySet::default();
/// // internal_set.insert(simplex_key); // Avoids extra UUID→Key lookups
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
/// - **N=2**: Facet sharing patterns (1-2 simplices per facet)
/// - **N=4**: Small temporary operations
/// - **N=8**: Typical vertex/simplex degrees
/// - **N=16**: Batch operation buffers
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::collections::SmallBuffer;
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

        // Test SecureHashMap basic operations for input-derived keys.
        let mut secure_map: SecureHashMap<u64, usize> = SecureHashMap::default();
        secure_map.insert(123, 456);
        assert_eq!(secure_map.get(&123), Some(&456));

        // Test SecureHashSet basic operations for input-derived keys.
        let mut secure_set: SecureHashSet<u64> = SecureHashSet::default();
        secure_set.insert(123);
        assert!(secure_set.contains(&123));
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
