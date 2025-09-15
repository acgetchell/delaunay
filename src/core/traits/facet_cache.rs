//! Facet caching trait for performance optimization
//!
//! This module provides the `FacetCacheProvider` trait that defines a common
//! interface for components that need to cache facet-to-cells mappings for
//! performance optimization.

use super::data_type::DataType;
use crate::core::{collections::FacetToCellsMap, triangulation_data_structure::Tds};
use crate::geometry::traits::coordinate::CoordinateScalar;
use arc_swap::ArcSwapOption;
use serde::{Serialize, de::DeserializeOwned};
use std::{
    iter::Sum,
    ops::{AddAssign, Div, SubAssign},
    sync::{Arc, atomic::AtomicU64},
};

/// Trait for components that provide cached facet-to-cells mappings.
///
/// This trait abstracts the common pattern of caching expensive facet-to-cells
/// mapping computations with atomic updates and cache invalidation. It's designed
/// to be implemented by algorithms that frequently need to access facet mappings,
/// such as convex hull construction, boundary analysis, and insertion algorithms.
///
/// # Performance Benefits
///
/// - **Avoids recomputation**: Expensive `build_facet_to_cells_hashmap()` calls
/// - **Thread-safe**: Atomic cache updates prevent race conditions  
/// - **Automatic invalidation**: Cache is invalidated when TDS changes
/// - **Memory efficient**: Single shared instance across algorithm operations
///
/// # Examples
///
/// ```
/// use delaunay::core::traits::facet_cache::FacetCacheProvider;
/// use delaunay::core::triangulation_data_structure::Tds;
/// use std::sync::Arc;
/// use std::sync::atomic::{AtomicU64, Ordering};
/// use arc_swap::ArcSwapOption;
///
/// struct MyAlgorithm {
///     facet_to_cells_cache: ArcSwapOption<delaunay::core::collections::FacetToCellsMap>,
///     cached_generation: AtomicU64,
/// }
///
/// impl MyAlgorithm {
///     fn new() -> Self {
///         Self {
///             facet_to_cells_cache: ArcSwapOption::empty(),
///             cached_generation: AtomicU64::new(0),
///         }
///     }
/// }
///
/// impl<T, U, V, const D: usize> FacetCacheProvider<T, U, V, D> for MyAlgorithm
/// where
///     T: delaunay::geometry::traits::coordinate::CoordinateScalar +
///        std::ops::AddAssign<T> + std::ops::SubAssign<T> + std::iter::Sum + num_traits::NumCast,
///     U: delaunay::core::traits::data_type::DataType + serde::de::DeserializeOwned,
///     V: delaunay::core::traits::data_type::DataType + serde::de::DeserializeOwned,
///     for<'a> &'a T: std::ops::Div<T>,
///     [T; D]: Copy + Default + serde::de::DeserializeOwned + serde::Serialize + Sized,
/// {
///     fn facet_cache(&self) -> &ArcSwapOption<delaunay::core::collections::FacetToCellsMap> {
///         &self.facet_to_cells_cache
///     }
///     
///     fn cached_generation(&self) -> &AtomicU64 {
///         &self.cached_generation
///     }
/// }
/// ```
pub trait FacetCacheProvider<T, U, V, const D: usize>
where
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + Sum + num_traits::NumCast,
    U: DataType + DeserializeOwned,
    V: DataType + DeserializeOwned,
    for<'a> &'a T: Div<T>,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// Returns a reference to the facet cache storage.
    fn facet_cache(&self) -> &ArcSwapOption<FacetToCellsMap>;

    /// Returns a reference to the cached generation counter.
    fn cached_generation(&self) -> &AtomicU64;

    /// Gets or builds the facet-to-cells mapping cache with atomic updates.
    ///
    /// This method handles cache invalidation and thread-safe rebuilding of the
    /// facet-to-cells mapping when the triangulation has been modified.
    ///
    /// # Arguments
    ///
    /// * `tds` - The triangulation data structure to build the cache from
    ///
    /// # Returns
    ///
    /// An `Arc<FacetToCellsMap>` containing the current facet-to-cells mapping
    ///
    /// # Performance
    ///
    /// - **Cache hit**: O(1) - Returns cached mapping if TDS generation matches
    /// - **Cache miss**: O(cells × `facets_per_cell`) - Rebuilds and caches mapping
    /// - **Thread-safe**: Uses atomic operations for concurrent access
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let cache = self.get_or_build_facet_cache(&tds);
    /// let facet_to_cells = cache.as_ref();
    ///
    /// // Use the cached mapping for O(1) facet lookups
    /// if let Some(adjacent_cells) = facet_to_cells.get(&facet_key) {
    ///     // Process adjacent cells...
    /// }
    /// ```
    fn get_or_build_facet_cache(&self, tds: &Tds<T, U, V, D>) -> Arc<FacetToCellsMap> {
        use std::sync::atomic::Ordering;

        // Check if cache is stale and needs to be invalidated
        let current_generation = tds.generation.load(Ordering::Relaxed);
        let cached_generation = self.cached_generation().load(Ordering::Relaxed);

        // Get or build the cached facet-to-cells mapping using ArcSwapOption
        // If the TDS generation matches the cached generation, cache is current
        if current_generation == cached_generation {
            // Cache is current - load existing cache or build if it doesn't exist
            self.facet_cache().load_full().map_or_else(
                || {
                    // No cache exists yet - build and store it
                    let new_cache = tds.build_facet_to_cells_hashmap();
                    let new_cache_arc = Arc::new(new_cache);

                    // Try to swap in the new cache (another thread might have done it already)
                    // If another thread beat us to it, use their cache instead
                    let none_ref: Option<Arc<FacetToCellsMap>> = None;
                    let _old = self
                        .facet_cache()
                        .compare_and_swap(&none_ref, Some(new_cache_arc.clone()));
                    // Load the current cache (could be ours or another thread's)
                    self.facet_cache().load_full().unwrap_or(new_cache_arc)
                },
                |existing_cache| {
                    // Cache exists and is current - use it
                    existing_cache
                },
            )
        } else {
            // Cache is stale - need to invalidate and rebuild
            let new_cache = tds.build_facet_to_cells_hashmap();
            let new_cache_arc = Arc::new(new_cache);

            // Atomically swap in the new cache
            self.facet_cache().store(Some(new_cache_arc.clone()));

            // Update the generation snapshot
            self.cached_generation()
                .store(current_generation, Ordering::Relaxed);

            new_cache_arc
        }
    }

    /// Invalidates the facet cache, forcing a rebuild on the next access.
    ///
    /// This method should be called when you know the triangulation has been
    /// modified and the cache is no longer valid. It's typically called
    /// automatically by algorithms that modify the TDS.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // After modifying the triangulation
    /// tds.insert_vertex(new_vertex);
    ///
    /// // Invalidate caches that depend on facet mappings
    /// algorithm.invalidate_facet_cache();
    /// ```
    fn invalidate_facet_cache(&self) {
        use std::sync::atomic::Ordering;

        // Clear the cache - next access will rebuild
        self.facet_cache().store(None);

        // Reset generation to force rebuild
        self.cached_generation().store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::triangulation_data_structure::Tds;
    use crate::core::vertex;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::thread;
    use std::time::Duration;

    /// Test implementation of `FacetCacheProvider` for unit testing
    struct TestCacheProvider {
        facet_to_cells_cache: ArcSwapOption<FacetToCellsMap>,
        cached_generation: AtomicU64,
    }

    impl TestCacheProvider {
        fn new() -> Self {
            Self {
                facet_to_cells_cache: ArcSwapOption::empty(),
                cached_generation: AtomicU64::new(0),
            }
        }
    }

    impl FacetCacheProvider<f64, Option<()>, Option<()>, 3> for TestCacheProvider {
        fn facet_cache(&self) -> &ArcSwapOption<FacetToCellsMap> {
            &self.facet_to_cells_cache
        }

        fn cached_generation(&self) -> &AtomicU64 {
            &self.cached_generation
        }
    }

    /// Create a simple test triangulation for testing
    fn create_test_triangulation() -> Tds<f64, Option<()>, Option<()>, 3> {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        Tds::new(&vertices).expect("Failed to create test triangulation")
    }

    #[test]
    fn test_initial_cache_state() {
        let provider = TestCacheProvider::new();

        // Initially, cache should be empty
        assert!(
            provider.facet_cache().load().is_none(),
            "Cache should be empty initially"
        );
        assert_eq!(
            provider.cached_generation().load(Ordering::Relaxed),
            0,
            "Generation should be 0 initially"
        );
    }

    #[test]
    fn test_cache_building() {
        let provider = TestCacheProvider::new();
        let tds = create_test_triangulation();

        // First call should build the cache
        let cache = provider.get_or_build_facet_cache(&tds);

        assert!(
            !cache.is_empty(),
            "Cache should not be empty after building"
        );

        // Cache should now be stored
        assert!(
            provider.facet_cache().load().is_some(),
            "Cache should be stored after building"
        );

        // Generation should match TDS generation
        let tds_generation = tds.generation.load(Ordering::Relaxed);
        let cached_generation = provider.cached_generation().load(Ordering::Relaxed);
        assert_eq!(
            cached_generation, tds_generation,
            "Cached generation should match TDS generation"
        );
    }

    #[test]
    fn test_cache_reuse() {
        let provider = TestCacheProvider::new();
        let tds = create_test_triangulation();

        // Build cache twice with same generation
        let cache1 = provider.get_or_build_facet_cache(&tds);
        let cache2 = provider.get_or_build_facet_cache(&tds);

        // Should be the same Arc instance (reused)
        let ptr1 = Arc::as_ptr(&cache1);
        let ptr2 = Arc::as_ptr(&cache2);
        assert_eq!(ptr1, ptr2, "Cache should be reused when generation matches");

        // Content should be identical
        assert_eq!(
            cache1.len(),
            cache2.len(),
            "Cache content should be identical"
        );
    }

    #[test]
    fn test_cache_invalidation_on_generation_change() {
        let provider = TestCacheProvider::new();
        let tds = create_test_triangulation();

        // Build initial cache
        let cache1 = provider.get_or_build_facet_cache(&tds);
        let ptr1 = Arc::as_ptr(&cache1);

        // Simulate TDS modification by changing generation
        let old_generation = tds.generation.load(Ordering::Relaxed);
        tds.generation.store(old_generation + 1, Ordering::Relaxed);

        // Next call should rebuild cache
        let cache2 = provider.get_or_build_facet_cache(&tds);
        let ptr2 = Arc::as_ptr(&cache2);

        assert_ne!(
            ptr1, ptr2,
            "Cache should be rebuilt when generation changes"
        );
        assert_eq!(
            cache1.len(),
            cache2.len(),
            "Cache content should remain consistent"
        );

        // Generation should be updated
        let new_cached_generation = provider.cached_generation().load(Ordering::Relaxed);
        let new_tds_generation = tds.generation.load(Ordering::Relaxed);
        assert_eq!(
            new_cached_generation, new_tds_generation,
            "Cached generation should be updated after rebuild"
        );
    }

    #[test]
    fn test_manual_cache_invalidation() {
        let provider = TestCacheProvider::new();
        let tds = create_test_triangulation();

        // Build cache
        let cache1 = provider.get_or_build_facet_cache(&tds);
        assert!(
            provider.facet_cache().load().is_some(),
            "Cache should exist after building"
        );

        let original_generation = provider.cached_generation().load(Ordering::Relaxed);
        assert_ne!(
            original_generation, 0,
            "Generation should not be 0 after caching"
        );

        // Manually invalidate cache
        provider.invalidate_facet_cache();

        // Cache should be cleared
        assert!(
            provider.facet_cache().load().is_none(),
            "Cache should be cleared after invalidation"
        );

        // Generation should be reset to 0
        let reset_generation = provider.cached_generation().load(Ordering::Relaxed);
        assert_eq!(
            reset_generation, 0,
            "Generation should be reset to 0 after invalidation"
        );

        // Next call should rebuild cache
        let cache2 = provider.get_or_build_facet_cache(&tds);
        let ptr1 = Arc::as_ptr(&cache1);
        let ptr2 = Arc::as_ptr(&cache2);
        assert_ne!(
            ptr1, ptr2,
            "Cache should be rebuilt after manual invalidation"
        );
    }

    #[test]
    fn test_cache_lifecycle() {
        let provider = TestCacheProvider::new();
        let tds = create_test_triangulation();

        // Complete lifecycle: empty -> build -> reuse -> invalidate -> rebuild

        // 1. Empty state
        assert!(
            provider.facet_cache().load().is_none(),
            "Should start empty"
        );

        // 2. Build
        let cache1 = provider.get_or_build_facet_cache(&tds);
        assert!(!cache1.is_empty(), "Cache should be built");

        // 3. Reuse
        let cache2 = provider.get_or_build_facet_cache(&tds);
        assert_eq!(
            Arc::as_ptr(&cache1),
            Arc::as_ptr(&cache2),
            "Cache should be reused"
        );

        // 4. Invalidate
        provider.invalidate_facet_cache();
        assert!(
            provider.facet_cache().load().is_none(),
            "Cache should be invalidated"
        );

        // 5. Rebuild
        let cache3 = provider.get_or_build_facet_cache(&tds);
        assert_ne!(
            Arc::as_ptr(&cache1),
            Arc::as_ptr(&cache3),
            "Cache should be rebuilt"
        );
        assert_eq!(
            cache1.len(),
            cache3.len(),
            "Content should be consistent after rebuild"
        );
    }

    #[test]
    fn test_concurrent_cache_access() {
        use std::sync::Barrier;

        let provider = Arc::new(TestCacheProvider::new());
        let tds = Arc::new(create_test_triangulation());
        let barrier = Arc::new(Barrier::new(4));
        let mut handles = vec![];

        // Spawn multiple threads that try to access cache simultaneously
        for i in 0..4 {
            let provider_clone = provider.clone();
            let tds_clone = tds.clone();
            let barrier_clone = barrier.clone();

            let handle = thread::spawn(move || {
                // Wait for all threads to be ready
                barrier_clone.wait();

                // Each thread tries to get cache multiple times
                let mut caches = vec![];
                for _ in 0..5 {
                    let cache = provider_clone.get_or_build_facet_cache(&tds_clone);
                    caches.push(cache);
                    thread::sleep(Duration::from_millis(1)); // Small delay
                }

                // Return thread ID and cache info for verification
                (i, caches.len(), caches[0].len())
            });

            handles.push(handle);
        }

        // Collect results
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All threads should succeed
        assert_eq!(results.len(), 4, "All threads should complete successfully");

        // All should get the same cache content size
        let expected_size = results[0].2;
        for (thread_id, cache_count, cache_size) in results {
            assert_eq!(cache_count, 5, "Thread {thread_id} should get 5 caches");
            assert_eq!(
                cache_size, expected_size,
                "Thread {thread_id} should get consistent cache size"
            );
        }
    }

    #[test]
    fn test_generation_overflow_handling() {
        let provider = TestCacheProvider::new();
        let tds = create_test_triangulation();

        // Set TDS generation near max value
        let near_max = u64::MAX - 1;
        tds.generation.store(near_max, Ordering::Relaxed);

        // Build cache with high generation
        let cache1 = provider.get_or_build_facet_cache(&tds);
        assert_eq!(
            provider.cached_generation().load(Ordering::Relaxed),
            near_max,
            "Should handle near-max generation values"
        );

        // Simulate overflow
        tds.generation.store(0, Ordering::Relaxed); // Wrapped around

        // Should still work correctly (rebuild cache)
        let cache2 = provider.get_or_build_facet_cache(&tds);
        assert_ne!(
            Arc::as_ptr(&cache1),
            Arc::as_ptr(&cache2),
            "Should rebuild cache after generation wraparound"
        );
        assert_eq!(
            provider.cached_generation().load(Ordering::Relaxed),
            0,
            "Should update to wrapped generation"
        );
    }

    #[test]
    fn test_empty_triangulation() {
        use crate::core::triangulation_data_structure::Tds;

        let provider = TestCacheProvider::new();

        // Create an empty triangulation
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&[]).unwrap();

        // Should handle empty triangulation gracefully
        let cache = provider.get_or_build_facet_cache(&tds);
        assert!(
            cache.is_empty(),
            "Cache should be empty for empty triangulation"
        );

        // Should still update generation tracking
        let tds_generation = tds.generation.load(Ordering::Relaxed);
        let cached_generation = provider.cached_generation().load(Ordering::Relaxed);
        assert_eq!(
            cached_generation, tds_generation,
            "Generation should be tracked even for empty triangulation"
        );
    }

    #[test]
    fn test_cache_content_correctness() {
        let provider = TestCacheProvider::new();
        let tds = create_test_triangulation();

        // Get cache from provider
        let provider_cache = provider.get_or_build_facet_cache(&tds);

        // Get reference cache directly from TDS
        let reference_cache = tds.build_facet_to_cells_hashmap();

        // Should have same size
        assert_eq!(
            provider_cache.len(),
            reference_cache.len(),
            "Provider cache should match reference cache size"
        );

        // Should have same keys
        for key in reference_cache.keys() {
            assert!(
                provider_cache.contains_key(key),
                "Provider cache should contain all reference keys"
            );
        }

        // Should have same values for each key
        for (key, reference_cells) in &reference_cache {
            if let Some(provider_cells) = provider_cache.get(key) {
                assert_eq!(
                    provider_cells.len(),
                    reference_cells.len(),
                    "Cell count should match for facet key {key}"
                );

                // Check that all cells match (order might differ)
                for cell in reference_cells {
                    assert!(
                        provider_cells.contains(cell),
                        "Provider cache should contain cell {cell:?} for key {key}"
                    );
                }
            } else {
                panic!("Provider cache missing key {key} found in reference");
            }
        }
    }
}
