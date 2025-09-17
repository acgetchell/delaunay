//! Facet caching trait for performance optimization
//!
//! This module provides the `FacetCacheProvider` trait that defines a common
//! interface for components that need to cache facet-to-cells mappings for
//! performance optimization.

use super::data_type::DataType;
use crate::core::{
    collections::FacetToCellsMap,
    triangulation_data_structure::{Tds, TriangulationValidationError},
};
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

    /// Helper method to build cache with RCU to minimize duplicate work under contention.
    ///
    /// This method uses Read-Copy-Update (RCU) pattern to ensure that only one thread
    /// builds the cache even under high contention, avoiding duplicate work.
    ///
    /// # Arguments
    ///
    /// * `tds` - The triangulation data structure to build the cache from
    ///
    /// # Returns
    ///
    /// The old cache value before update (if any), or `None` if no cache existed
    #[deprecated(note = "Use try_build_cache_with_rcu instead for proper error handling")]
    fn build_cache_with_rcu(&self, tds: &Tds<T, U, V, D>) -> Option<Arc<FacetToCellsMap>> {
        // Forward to strict version, falling back to lenient behavior on error
        match self.try_build_cache_with_rcu(tds) {
            Ok(result) => result,
            #[allow(unused_variables)] // error used in debug_assertions
            Err(error) => {
                #[cfg(debug_assertions)]
                eprintln!(
                    "Warning: build_cache_with_rcu falling back to empty cache due to error: {error}"
                );
                None
            }
        }
    }

    /// Strict helper method to build cache with RCU and proper error handling.
    ///
    /// This method uses Read-Copy-Update (RCU) pattern to ensure that only one thread
    /// builds the cache even under high contention, avoiding duplicate work. Unlike
    /// the deprecated `build_cache_with_rcu`, this method returns errors if the TDS
    /// has missing vertex keys instead of masking them.
    ///
    /// # Arguments
    ///
    /// * `tds` - The triangulation data structure to build the cache from
    ///
    /// # Returns
    ///
    /// A `Result` containing:
    /// - `Ok(Some(Arc<FacetToCellsMap>))`: The old cache value before update
    /// - `Ok(None)`: No cache existed before this build
    /// - `Err(TriangulationValidationError)`: If facet map building fails
    ///
    /// # Errors
    ///
    /// Returns a `TriangulationValidationError` if the TDS has corrupted data
    /// (e.g., missing vertex keys) that prevents building a complete facet map.
    fn try_build_cache_with_rcu(
        &self,
        tds: &Tds<T, U, V, D>,
    ) -> Result<Option<Arc<FacetToCellsMap>>, TriangulationValidationError> {
        // We memoize the built cache outside the RCU closure to avoid recomputation
        // if RCU needs to retry due to concurrent updates.
        let mut built: Option<Result<Arc<FacetToCellsMap>, TriangulationValidationError>> = None;

        let old_cache = self.facet_cache().rcu(|old| {
            if let Some(existing) = old {
                // Another thread built the cache while we were waiting
                return Some(existing.clone());
            }
            // Build the cache only once, even if RCU retries
            #[allow(clippy::option_if_let_else)]
            // Complex error handling doesn't benefit from map_or_else
            match built.get_or_insert_with(|| tds.try_build_facet_to_cells_hashmap().map(Arc::new))
            {
                Ok(arc) => Some(arc.clone()),
                Err(_) => None, // Let the caller handle the error
            }
        });

        // Check what happened during the RCU operation
        match built {
            Some(Ok(_)) => {
                // We built the cache successfully, return None as the old value
                Ok(None)
            }
            Some(Err(e)) => {
                // We tried to build but failed
                Err(e)
            }
            None => {
                // Another thread built it or we returned an existing cache
                Ok(old_cache)
            }
        }
    }

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
    /// - **Contention**: Minimizes duplicate work by building cache lazily inside RCU
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
    #[deprecated(note = "Use try_get_or_build_facet_cache instead for proper error handling")]
    fn get_or_build_facet_cache(&self, tds: &Tds<T, U, V, D>) -> Arc<FacetToCellsMap> {
        // Forward to strict version, falling back to lenient behavior on error
        match self.try_get_or_build_facet_cache(tds) {
            Ok(cache) => cache,
            #[allow(unused_variables)] // error used in debug_assertions
            Err(error) => {
                #[cfg(debug_assertions)]
                eprintln!(
                    "Warning: get_or_build_facet_cache falling back to lenient TDS method due to error: {error}"
                );

                // Fall back to the lenient TDS method as a last resort
                #[allow(deprecated)] // Internal fallback - acceptable for graceful degradation
                Arc::new(tds.build_facet_to_cells_hashmap())
            }
        }
    }

    /// Gets or builds the facet-to-cells mapping cache with strict error handling.
    ///
    /// This method handles cache invalidation and thread-safe rebuilding of the
    /// facet-to-cells mapping when the triangulation has been modified. Unlike
    /// the deprecated `get_or_build_facet_cache`, this method returns errors if
    /// the TDS has corrupted data instead of masking them.
    ///
    /// # Arguments
    ///
    /// * `tds` - The triangulation data structure to build the cache from
    ///
    /// # Returns
    ///
    /// A `Result` containing:
    /// - `Ok(Arc<FacetToCellsMap>)`: The current facet-to-cells mapping
    /// - `Err(TriangulationValidationError)`: If facet map building fails
    ///
    /// # Performance
    ///
    /// - **Cache hit**: O(1) - Returns cached mapping if TDS generation matches
    /// - **Cache miss**: O(cells × `facets_per_cell`) - Rebuilds and caches mapping
    /// - **Thread-safe**: Uses atomic operations for concurrent access
    /// - **Contention**: Minimizes duplicate work by building cache lazily inside RCU
    ///
    /// # Errors
    ///
    /// Returns a `TriangulationValidationError` if the TDS has corrupted data
    /// (e.g., missing vertex keys) that prevents building a complete facet map.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let cache = self.try_get_or_build_facet_cache(&tds)?;
    /// let facet_to_cells = cache.as_ref();
    ///
    /// // Use the cached mapping for O(1) facet lookups
    /// if let Some(adjacent_cells) = facet_to_cells.get(&facet_key) {
    ///     // Process adjacent cells...
    /// }
    /// ```
    fn try_get_or_build_facet_cache(
        &self,
        tds: &Tds<T, U, V, D>,
    ) -> Result<Arc<FacetToCellsMap>, TriangulationValidationError> {
        use std::sync::atomic::Ordering;

        // Check if cache is stale and needs to be invalidated
        // ORDERING: Acquire loads here synchronize with Release stores to ensure
        // we see both the cache and generation updates from writers consistently.
        // This prevents torn reads where we might see a new cache with old generation.
        let current_generation = tds.generation();
        let cached_generation = self.cached_generation().load(Ordering::Acquire);

        // Get or build the cached facet-to-cells mapping using ArcSwapOption
        // If the TDS generation matches the cached generation, cache is current
        if current_generation == cached_generation {
            // Cache is current - load existing cache or build if it doesn't exist
            if let Some(existing_cache) = self.facet_cache().load_full() {
                // Cache exists and is current - use it
                Ok(existing_cache)
            } else {
                // Build cache lazily inside RCU to minimize duplicate work under contention.
                let built_cache = self.try_build_cache_with_rcu(tds)?;

                // Update generation if we were the ones who built it
                // Note: built_cache is the old value before our update
                // Only store generation if the cache is actually present to avoid stale store
                if built_cache.is_none() && self.facet_cache().load_full().is_some() {
                    self.cached_generation()
                        .store(current_generation, Ordering::Release);
                }

                // Return the cache; if concurrently invalidated, retry via slow path
                // Another thread could invalidate between RCU and load_full()
                self.facet_cache()
                    .load_full()
                    .map_or_else(|| self.try_get_or_build_facet_cache(tds), Ok)
            }
        } else {
            // Cache is stale - need to invalidate and rebuild
            let new_cache = tds.try_build_facet_to_cells_hashmap()?;
            let new_cache_arc = Arc::new(new_cache);

            // Atomically swap in the new cache.
            // ORDERING: ArcSwap's store() uses SeqCst ordering, establishing a happens-before
            // relationship with any subsequent loads. This ensures the new cache is visible
            // to all threads before the generation update below.
            self.facet_cache().store(Some(new_cache_arc.clone()));

            // Update the generation snapshot.
            // ORDERING: The Release ordering here synchronizes with Acquire loads in future
            // cache checks. Combined with the SeqCst store above, this ensures readers will
            // see both the new cache and the updated generation consistently.
            self.cached_generation()
                .store(current_generation, Ordering::Release);

            Ok(new_cache_arc)
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
        // ORDERING: The SeqCst store from ArcSwap ensures this None is visible
        // before the generation reset below. If future refactors relax the ordering,
        // an explicit fence would be needed to maintain the happens-before relationship.
        self.facet_cache().store(None);

        // Reset generation to force rebuild
        // ORDERING: Release ensures the None store above is visible before this update.
        self.cached_generation().store(0, Ordering::Release);
    }
}

#[cfg(test)]
#[allow(deprecated)] // Tests intentionally use deprecated methods for backward compatibility testing
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
        let tds_generation = tds.generation();
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
        let mut tds = create_test_triangulation();

        // Build initial cache
        let cache1 = provider.get_or_build_facet_cache(&tds);
        let ptr1 = Arc::as_ptr(&cache1);
        let initial_generation = tds.generation();

        // Modify TDS by adding a new vertex - this will bump the generation
        let new_vertex = vertex!([0.5, 0.5, 0.5]); // Interior point
        tds.add(new_vertex).expect("Failed to add vertex");

        // Verify generation was incremented
        let new_generation = tds.generation();
        assert!(
            new_generation > initial_generation,
            "Generation should increase after adding vertex"
        );

        // Next call should rebuild cache due to generation change
        let cache2 = provider.get_or_build_facet_cache(&tds);
        let ptr2 = Arc::as_ptr(&cache2);

        assert_ne!(
            ptr1, ptr2,
            "Cache should be rebuilt when generation changes"
        );

        // The cache size might be different since we added a vertex and created new cells
        // but both should be valid caches
        assert!(!cache1.is_empty(), "Original cache should not be empty");
        assert!(!cache2.is_empty(), "New cache should not be empty");

        // Generation should be updated in the provider
        let new_cached_generation = provider.cached_generation().load(Ordering::Relaxed);
        assert_eq!(
            new_cached_generation, new_generation,
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
    fn test_build_cache_with_rcu() {
        let provider = TestCacheProvider::new();
        let tds = create_test_triangulation();

        // Initial state: no cache
        assert!(
            provider.facet_cache().load().is_none(),
            "Cache should be empty initially"
        );

        // First call to build_cache_with_rcu should build cache
        let old_value = provider.build_cache_with_rcu(&tds);
        assert!(
            old_value.is_none(),
            "Old value should be None on first build"
        );

        // Cache should now exist
        let cached = provider.facet_cache().load_full();
        assert!(
            cached.is_some(),
            "Cache should exist after build_cache_with_rcu"
        );
        let unwrapped_cache = cached.unwrap();
        assert!(
            !unwrapped_cache.is_empty(),
            "Built cache should not be empty"
        );

        // Second call should return the existing cache as old value
        let old_value2 = provider.build_cache_with_rcu(&tds);
        assert!(
            old_value2.is_some(),
            "Old value should be Some on second build"
        );

        // The old value should be the same Arc as cache1
        let old_arc = old_value2.unwrap();
        assert_eq!(
            Arc::as_ptr(&old_arc),
            Arc::as_ptr(&unwrapped_cache),
            "Old value should be the previously built cache"
        );

        // Final cache should still be the same (RCU detected existing)
        let cached2 = provider.facet_cache().load_full().unwrap();
        assert_eq!(
            Arc::as_ptr(&cached2),
            Arc::as_ptr(&unwrapped_cache),
            "Cache should not be rebuilt when it already exists"
        );
    }

    #[test]
    fn test_build_cache_with_rcu_concurrent() {
        use std::sync::Barrier;
        use std::sync::atomic::AtomicUsize;

        let provider = Arc::new(TestCacheProvider::new());
        let tds = Arc::new(create_test_triangulation());
        let barrier = Arc::new(Barrier::new(4));
        let success_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        // Spawn multiple threads that try to get/build cache simultaneously
        for i in 0..4 {
            let provider_clone = provider.clone();
            let tds_clone = tds.clone();
            let barrier_clone = barrier.clone();
            let success_count_clone = success_count.clone();

            let handle = thread::spawn(move || {
                // Wait for all threads to be ready
                barrier_clone.wait();

                // Each thread tries to get/build cache using strict method
                let cache_result = provider_clone.try_get_or_build_facet_cache(&tds_clone);

                let success = cache_result.is_ok();
                if success {
                    success_count_clone.fetch_add(1, Ordering::Relaxed);
                }

                // Return thread ID and whether it succeeded
                (i, success)
            });

            handles.push(handle);
        }

        // Collect results
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All threads should complete successfully
        assert_eq!(results.len(), 4, "All threads should complete");

        // All threads should succeed in getting the cache
        let success_count_final = success_count.load(Ordering::Relaxed);
        assert_eq!(
            success_count_final, 4,
            "All threads should successfully get the cache"
        );

        // Verify all operations succeeded
        for (thread_id, success) in results {
            assert!(success, "Thread {thread_id} should succeed");
        }

        // Final cache should exist and be consistent
        let final_cache = provider.facet_cache().load_full();
        assert!(
            final_cache.is_some(),
            "Cache should exist after concurrent builds"
        );
        assert!(
            !final_cache.unwrap().is_empty(),
            "Cache should not be empty"
        );
    }

    #[test]
    fn test_generation_overflow_handling() {
        // This test demonstrates that the cache system handles generation changes correctly
        // Since we can't set generation to near-max values directly, we test that
        // the cache invalidation mechanism works correctly with multiple operations
        let provider = TestCacheProvider::new();
        let mut tds = create_test_triangulation();

        // Build initial cache
        let cache1 = provider.get_or_build_facet_cache(&tds);
        let initial_gen = tds.generation();
        let ptr1 = Arc::as_ptr(&cache1);

        // Perform multiple operations to increment generation
        // Each operation should bump the generation counter
        let operations = [
            vertex!([0.2, 0.2, 0.2]),
            vertex!([0.3, 0.3, 0.3]),
            vertex!([0.4, 0.4, 0.4]),
        ];

        for vertex in operations {
            let prev_gen = tds.generation();
            tds.add(vertex).expect("Failed to add vertex");
            let new_gen = tds.generation();
            assert!(
                new_gen > prev_gen,
                "Generation should increment after each operation"
            );
        }

        // Cache should be invalidated and rebuilt
        let cache2 = provider.get_or_build_facet_cache(&tds);
        let ptr2 = Arc::as_ptr(&cache2);
        let final_gen = tds.generation();

        assert_ne!(
            ptr1, ptr2,
            "Cache should be rebuilt after generation changes"
        );
        assert!(
            final_gen > initial_gen,
            "Generation should have increased after operations"
        );
        assert_eq!(
            provider.cached_generation().load(Ordering::Relaxed),
            final_gen,
            "Provider should track current generation"
        );

        // Test that clearing neighbors also bumps generation
        let gen_before_clear = tds.generation();
        tds.clear_all_neighbors();
        let gen_after_clear = tds.generation();
        assert!(
            gen_after_clear > gen_before_clear,
            "Clearing neighbors should bump generation"
        );

        // Cache should be rebuilt again
        let cache3 = provider.get_or_build_facet_cache(&tds);
        let ptr3 = Arc::as_ptr(&cache3);
        assert_ne!(
            ptr2, ptr3,
            "Cache should be rebuilt after clear_all_neighbors"
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
        let tds_generation = tds.generation();
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
        #[allow(deprecated)] // Used for verification in test
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

    #[test]
    fn test_strict_try_get_or_build_facet_cache() {
        let provider = TestCacheProvider::new();
        let tds = create_test_triangulation();

        // First call should build the cache successfully
        let cache_result = provider.try_get_or_build_facet_cache(&tds);
        assert!(
            cache_result.is_ok(),
            "Initial try_get_or_build_facet_cache should succeed"
        );

        let cache = cache_result.unwrap();
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
        let tds_generation = tds.generation();
        let cached_generation = provider.cached_generation().load(Ordering::Relaxed);
        assert_eq!(
            cached_generation, tds_generation,
            "Cached generation should match TDS generation"
        );

        // Second call should return the same cache
        let second_cache_result = provider.try_get_or_build_facet_cache(&tds);
        assert!(
            second_cache_result.is_ok(),
            "Second call should also succeed"
        );

        let cache2 = second_cache_result.unwrap();
        assert_eq!(
            Arc::as_ptr(&cache),
            Arc::as_ptr(&cache2),
            "Cache should be reused when generation matches"
        );
    }

    #[test]
    fn test_strict_try_build_cache_with_rcu() {
        let provider = TestCacheProvider::new();
        let tds = create_test_triangulation();

        // Initial state: no cache
        assert!(
            provider.facet_cache().load().is_none(),
            "Cache should be empty initially"
        );

        // First call to try_build_cache_with_rcu should build cache successfully
        let old_value_result = provider.try_build_cache_with_rcu(&tds);
        assert!(
            old_value_result.is_ok(),
            "try_build_cache_with_rcu should succeed"
        );

        let old_value = old_value_result.unwrap();
        assert!(
            old_value.is_none(),
            "Old value should be None on first build"
        );

        // Cache should now exist
        let cached = provider.facet_cache().load_full();
        assert!(
            cached.is_some(),
            "Cache should exist after try_build_cache_with_rcu"
        );
        let unwrapped_cache = cached.unwrap();
        assert!(
            !unwrapped_cache.is_empty(),
            "Built cache should not be empty"
        );

        // Second call should return the existing cache as old value
        let second_old_result = provider.try_build_cache_with_rcu(&tds);
        assert!(
            second_old_result.is_ok(),
            "Second try_build_cache_with_rcu should succeed"
        );

        let old_value2 = second_old_result.unwrap();
        assert!(
            old_value2.is_some(),
            "Old value should be Some on second build"
        );

        // The old value should be the same Arc as the first cache
        let old_arc = old_value2.unwrap();
        assert_eq!(
            Arc::as_ptr(&old_arc),
            Arc::as_ptr(&unwrapped_cache),
            "Old value should be the previously built cache"
        );
    }

    #[test]
    fn test_strict_cache_invalidation_behavior() {
        let provider = TestCacheProvider::new();
        let mut tds = create_test_triangulation();

        // Build initial cache
        let cache1_result = provider.try_get_or_build_facet_cache(&tds);
        assert!(cache1_result.is_ok(), "Initial cache build should succeed");

        let cache1 = cache1_result.unwrap();
        let ptr1 = Arc::as_ptr(&cache1);
        let initial_generation = tds.generation();

        // Modify TDS by adding a new vertex - this will bump the generation
        let new_vertex = vertex!([0.5, 0.5, 0.5]); // Interior point
        tds.add(new_vertex).expect("Failed to add vertex");

        // Verify generation was incremented
        let new_generation = tds.generation();
        assert!(
            new_generation > initial_generation,
            "Generation should increase after adding vertex"
        );

        // Next call should rebuild cache due to generation change
        let cache2_result = provider.try_get_or_build_facet_cache(&tds);
        assert!(cache2_result.is_ok(), "Cache rebuild should succeed");

        let cache2 = cache2_result.unwrap();
        let ptr2 = Arc::as_ptr(&cache2);

        assert_ne!(
            ptr1, ptr2,
            "Cache should be rebuilt when generation changes"
        );

        // Generation should be updated in the provider
        let new_cached_generation = provider.cached_generation().load(Ordering::Relaxed);
        assert_eq!(
            new_cached_generation, new_generation,
            "Cached generation should be updated after rebuild"
        );
    }

    #[test]
    fn test_deprecated_methods_forward_correctly() {
        let provider = TestCacheProvider::new();
        let tds = create_test_triangulation();

        // Test deprecated get_or_build_facet_cache forwards to try version
        let _cache_deprecated = provider.get_or_build_facet_cache(&tds);
        let cache_strict_result = provider.try_get_or_build_facet_cache(&tds);

        assert!(
            cache_strict_result.is_ok(),
            "Strict version should work after deprecated version"
        );

        // Test deprecated build_cache_with_rcu forwards to try version
        provider.invalidate_facet_cache(); // Clear cache first
        let _old_deprecated = provider.build_cache_with_rcu(&tds);
        let old_strict_result = provider.try_build_cache_with_rcu(&tds);

        assert!(
            old_strict_result.is_ok(),
            "Strict version should work after deprecated version"
        );
    }
}
