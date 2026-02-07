//! Facet caching trait for performance optimization
//!
//! This module provides the `FacetCacheProvider` trait that defines a common
//! interface for components that need to cache facet-to-cells mappings for
//! performance optimization.

use super::data_type::DataType;
use crate::core::{
    collections::FacetToCellsMap,
    triangulation_data_structure::{Tds, TdsValidationError},
};
use crate::geometry::traits::coordinate::ScalarAccumulative;
use arc_swap::ArcSwapOption;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
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
/// use delaunay::geometry::traits::coordinate::ScalarAccumulative;
/// use delaunay::core::traits::data_type::DataType;
/// use std::sync::Arc;
/// use std::sync::atomic::{AtomicU64, Ordering};
/// use serde::de::DeserializeOwned;
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
///     T: ScalarAccumulative,
///     U: DataType,
///     V: DataType,
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
    T: ScalarAccumulative,
    U: DataType,
    V: DataType,
{
    /// Returns a reference to the facet cache storage.
    ///
    /// The cache stores precomputed facet-to-cells mappings to avoid expensive
    /// rebuilds during repeated facet queries.
    fn facet_cache(&self) -> &ArcSwapOption<FacetToCellsMap>;

    /// Returns a reference to the cached generation counter.
    fn cached_generation(&self) -> &AtomicU64;

    /// Strict helper method to build cache with RCU and proper error handling.
    ///
    /// This method uses the Read-Copy-Update (RCU) pattern to ensure that only one thread
    /// builds the cache even under high contention, avoiding duplicate work.
    ///
    /// Unlike best-effort caching helpers that may mask structural issues, this method
    /// returns errors if facet map building fails (e.g., missing vertex keys / corrupted TDS).
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
    /// - `Err(TdsValidationError)`: If facet map building fails
    ///
    /// # Errors
    ///
    /// Returns a `TdsValidationError` if the TDS has corrupted data
    /// (e.g., missing vertex keys) that prevents building a complete facet map.
    fn try_build_cache_with_rcu(
        &self,
        tds: &Tds<T, U, V, D>,
    ) -> Result<Option<Arc<FacetToCellsMap>>, TdsValidationError> {
        // We memoize the built cache outside the RCU closure to avoid recomputation
        // if RCU needs to retry due to concurrent updates.
        let mut built: Option<Result<Arc<FacetToCellsMap>, TdsValidationError>> = None;

        let old_cache = self.facet_cache().rcu(|old| {
            if let Some(existing) = old {
                // Another thread built the cache while we were waiting
                return Some(existing.clone());
            }
            // Build the cache only once, even if RCU retries
            #[expect(clippy::option_if_let_else)]
            // Complex error handling doesn't benefit from map_or_else
            match built.get_or_insert_with(|| tds.build_facet_to_cells_map().map(Arc::new)) {
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

    /// Gets or builds the facet-to-cells mapping cache with strict error handling.
    ///
    /// This method handles cache invalidation and thread-safe rebuilding of the
    /// facet-to-cells mapping when the triangulation has been modified.
    ///
    /// Unlike best-effort caching helpers that may mask structural issues, this method
    /// returns errors if the TDS has corrupted data instead of masking them.
    ///
    /// # Arguments
    ///
    /// * `tds` - The triangulation data structure to build the cache from
    ///
    /// # Returns
    ///
    /// A `Result` containing:
    /// - `Ok(Arc<FacetToCellsMap>)`: The current facet-to-cells mapping
    /// - `Err(TdsValidationError)`: If facet map building fails
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
    /// Returns a `TdsValidationError` if the TDS has corrupted data
    /// (e.g., missing vertex keys) that prevents building a complete facet map.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices)?;
    ///
    /// // Build facet-to-cells mapping
    /// let facet_map = dt.tds().build_facet_to_cells_map()?;
    ///
    /// // Use the mapping for facet lookups
    /// for (facet_key, adjacent_cells) in facet_map.iter() {
    ///     // Each facet has at most 2 adjacent cells
    ///     assert!(adjacent_cells.len() <= 2);
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn try_get_or_build_facet_cache(
        &self,
        tds: &Tds<T, U, V, D>,
    ) -> Result<Arc<FacetToCellsMap>, TdsValidationError> {
        let mut current_generation = tds.generation();

        loop {
            // Check if cache is stale and needs to be invalidated
            // ORDERING: Acquire loads here synchronize with Release stores to ensure
            // we see both the cache and generation updates from writers consistently.
            // This prevents torn reads where we might see a new cache with old generation.
            let cached_generation = self.cached_generation().load(Ordering::Acquire);

            // Get or build the cached facet-to-cells mapping using ArcSwapOption
            // If the TDS generation matches the cached generation, cache is current
            if current_generation == cached_generation {
                // Cache is current - load existing cache or build if it doesn't exist
                if let Some(existing_cache) = self.facet_cache().load_full() {
                    // Cache exists and is current - use it
                    return Ok(existing_cache);
                }
                // Build cache lazily inside RCU to minimize duplicate work under contention.
                let built_cache = self.try_build_cache_with_rcu(tds)?;

                // Re-check generation to avoid stashing a cache built against a stale TDS.
                if tds.generation() != current_generation {
                    current_generation = tds.generation();
                    continue;
                }
                // Update generation if we were the ones who built it
                // Note: built_cache is the old value before our update
                // Only store generation if the cache is actually present to avoid stale store
                if built_cache.is_none() && self.facet_cache().load_full().is_some() {
                    self.cached_generation()
                        .store(current_generation, Ordering::Release);
                }

                // Return the cache; if concurrently invalidated, retry via the loop
                // Another thread could invalidate between RCU and load_full()
                if let Some(cache) = self.facet_cache().load_full() {
                    return Ok(cache);
                }
                // Cache was invalidated after we built it - continue loop to rebuild
            }

            // Cache is stale - coordinate rebuild through RCU to avoid duplicate work
            current_generation = tds.generation();

            // OPTIMIZATION: Invalidate first to coordinate rebuilds via RCU. This prevents
            // multiple threads from rebuilding in parallel when hitting stale cache simultaneously.
            // By setting to None first, we ensure that subsequent threads will use the RCU
            // mechanism in try_build_cache_with_rcu(), where only one thread builds while
            // others wait and reuse the result, avoiding expensive duplicate work.
            self.facet_cache().store(None);

            // Coordinate the build; return value is the OLD cache (if any).
            let _old = self.try_build_cache_with_rcu(tds)?;

            let rebuilt_generation = tds.generation();
            if rebuilt_generation != current_generation {
                current_generation = rebuilt_generation;
                continue;
            }

            if let Some(cache) = self.facet_cache().load_full() {
                self.cached_generation()
                    .store(current_generation, Ordering::Release);
                return Ok(cache);
            }

            // Fallback to direct build to guarantee progress
            let new_cache = tds.build_facet_to_cells_map()?;
            let new_cache_arc = Arc::new(new_cache);
            self.facet_cache().store(Some(new_cache_arc.clone()));

            let rebuilt_generation = tds.generation();
            if rebuilt_generation != current_generation {
                current_generation = rebuilt_generation;
                continue;
            }

            self.cached_generation()
                .store(current_generation, Ordering::Release);
            return Ok(new_cache_arc);
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
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // Build facet-to-cells mapping
    /// let facet_map = dt.tds().build_facet_to_cells_map().unwrap();
    /// assert!(!facet_map.is_empty(), "Facet map should contain entries");
    /// ```
    fn invalidate_facet_cache(&self) {
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
mod tests {
    use super::*;
    use crate::core::delaunay_triangulation::DelaunayTriangulation;
    use crate::core::triangulation_data_structure::Tds;
    use crate::core::vertex;
    use crate::geometry::kernel::FastKernel;
    use std::sync::Arc;
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
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

    impl FacetCacheProvider<f64, (), (), 3> for TestCacheProvider {
        fn facet_cache(&self) -> &ArcSwapOption<FacetToCellsMap> {
            &self.facet_to_cells_cache
        }

        fn cached_generation(&self) -> &AtomicU64 {
            &self.cached_generation
        }
    }

    /// Create a simple test triangulation for testing
    fn create_test_triangulation() -> DelaunayTriangulation<FastKernel<f64>, (), (), 3> {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        DelaunayTriangulation::new(&vertices).expect("Failed to create test triangulation")
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
        let dt = create_test_triangulation();

        // First call should build the cache
        let cache = provider.try_get_or_build_facet_cache(dt.tds()).unwrap();

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
        let tds_generation = dt.tds().generation();
        let cached_generation = provider.cached_generation().load(Ordering::Relaxed);
        assert_eq!(
            cached_generation, tds_generation,
            "Cached generation should match TDS generation"
        );
    }

    #[test]
    fn test_cache_reuse() {
        let provider = TestCacheProvider::new();
        let dt = create_test_triangulation();

        // Build cache twice with same generation
        let cache1 = provider.try_get_or_build_facet_cache(dt.tds()).unwrap();
        let cache2 = provider.try_get_or_build_facet_cache(dt.tds()).unwrap();

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
        let mut dt = create_test_triangulation();

        // Build initial cache
        let cache1 = provider.try_get_or_build_facet_cache(dt.tds()).unwrap();
        let ptr1 = Arc::as_ptr(&cache1);
        let initial_generation = dt.tds().generation();

        // Modify triangulation by adding a new vertex - this will bump the generation
        // Use an interior vertex away from the circumcenter to avoid degenerate insertion cases
        let new_vertex = vertex!([0.2, 0.2, 0.2]);
        dt.insert(new_vertex).expect("Failed to add vertex");

        // Verify generation was incremented
        let new_generation = dt.tds().generation();
        assert!(
            new_generation > initial_generation,
            "Generation should increase after adding vertex"
        );

        // Next call should rebuild cache due to generation change
        let cache2 = provider.try_get_or_build_facet_cache(dt.tds()).unwrap();
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
        let dt = create_test_triangulation();

        // Build cache
        let cache1 = provider.try_get_or_build_facet_cache(dt.tds()).unwrap();
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
        let cache2 = provider.try_get_or_build_facet_cache(dt.tds()).unwrap();
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
        let dt = create_test_triangulation();

        // Complete lifecycle: empty -> build -> reuse -> invalidate -> rebuild

        // 1. Empty state
        assert!(
            provider.facet_cache().load().is_none(),
            "Should start empty"
        );

        // 2. Build
        let cache1 = provider.try_get_or_build_facet_cache(dt.tds()).unwrap();
        assert!(!cache1.is_empty(), "Cache should be built");

        // 3. Reuse
        let cache2 = provider.try_get_or_build_facet_cache(dt.tds()).unwrap();
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
        let cache3 = provider.try_get_or_build_facet_cache(dt.tds()).unwrap();
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
        let provider = Arc::new(TestCacheProvider::new());
        let dt = Arc::new(create_test_triangulation());
        let barrier = Arc::new(Barrier::new(4));
        let mut handles = vec![];

        // Spawn multiple threads that try to access cache simultaneously
        for i in 0..4 {
            let provider_clone = provider.clone();
            let dt_clone = dt.clone();
            let barrier_clone = barrier.clone();

            let handle = thread::spawn(move || {
                // Wait for all threads to be ready
                barrier_clone.wait();

                // Each thread tries to get cache multiple times
                let mut caches = vec![];
                for _ in 0..5 {
                    let cache = provider_clone
                        .try_get_or_build_facet_cache(dt_clone.tds())
                        .unwrap();
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
        let dt = create_test_triangulation();

        // Initial state: no cache
        assert!(
            provider.facet_cache().load().is_none(),
            "Cache should be empty initially"
        );

        // First call to try_build_cache_with_rcu should build cache
        let old_value = provider.try_build_cache_with_rcu(dt.tds()).unwrap();
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
        let old_value2 = provider.try_build_cache_with_rcu(dt.tds()).unwrap();
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
        let provider = Arc::new(TestCacheProvider::new());
        let dt = Arc::new(create_test_triangulation());
        let barrier = Arc::new(Barrier::new(4));
        let success_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        // Spawn multiple threads that try to get/build cache simultaneously
        for i in 0..4 {
            let provider_clone = provider.clone();
            let dt_clone = dt.clone();
            let barrier_clone = barrier.clone();
            let success_count_clone = success_count.clone();

            let handle = thread::spawn(move || {
                // Wait for all threads to be ready
                barrier_clone.wait();

                // Each thread tries to get/build cache using strict method
                let cache_result = provider_clone.try_get_or_build_facet_cache(dt_clone.tds());

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
        let mut dt = create_test_triangulation();

        // Build initial cache
        let cache1 = provider.try_get_or_build_facet_cache(dt.tds()).unwrap();
        let initial_gen = dt.tds().generation();
        let ptr1 = Arc::as_ptr(&cache1);

        // Perform multiple operations to increment generation
        // Each operation should bump the generation counter
        let operations = [
            vertex!([0.2, 0.2, 0.2]),
            vertex!([0.3, 0.3, 0.3]),
            vertex!([0.4, 0.4, 0.4]),
        ];

        for vertex in operations {
            let prev_gen = dt.tds().generation();
            let _ = dt.insert(vertex);
            let new_gen = dt.tds().generation();
            assert!(
                new_gen >= prev_gen,
                "Generation should not go backwards after operation"
            );
        }

        // Cache should be invalidated and rebuilt
        let cache2 = provider.try_get_or_build_facet_cache(dt.tds()).unwrap();
        let ptr2 = Arc::as_ptr(&cache2);
        let final_gen = dt.tds().generation();

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
        let gen_before_clear = dt.tds().generation();
        dt.tri.tds.clear_all_neighbors();
        let gen_after_clear = dt.tds().generation();
        assert!(
            gen_after_clear > gen_before_clear,
            "Clearing neighbors should bump generation"
        );

        // Cache should be rebuilt again
        let cache3 = provider.try_get_or_build_facet_cache(dt.tds()).unwrap();
        let ptr3 = Arc::as_ptr(&cache3);
        assert_ne!(
            ptr2, ptr3,
            "Cache should be rebuilt after clear_all_neighbors"
        );
    }

    #[test]
    fn test_empty_triangulation() {
        let provider = TestCacheProvider::new();

        // Create an empty triangulation
        let tds: Tds<f64, (), (), 3> = Tds::empty();

        // Should handle empty triangulation gracefully
        let cache = provider.try_get_or_build_facet_cache(&tds).unwrap();
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
        let dt = create_test_triangulation();

        // Get cache from provider
        let provider_cache = provider.try_get_or_build_facet_cache(dt.tds()).unwrap();

        // Get reference cache directly from TDS (use current strict API)
        let reference_cache = dt.tds().build_facet_to_cells_map().unwrap();

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
        let dt = create_test_triangulation();

        // Get cache using strict method
        let cache_result = provider.try_get_or_build_facet_cache(dt.tds());
        assert!(cache_result.is_ok(), "Strict method should succeed");

        let cache = cache_result.unwrap();
        assert!(!cache.is_empty(), "Cache should be built and not empty");

        // Cache should be stored
        assert!(
            provider.facet_cache().load().is_some(),
            "Cache should be stored after building"
        );

        // Generation should match TDS generation
        let tds_generation = dt.tds().generation();
        let cached_generation = provider.cached_generation().load(Ordering::Relaxed);
        assert_eq!(
            cached_generation, tds_generation,
            "Cached generation should match TDS generation"
        );

        // Second call should return the same cache
        let second_cache_result = provider.try_get_or_build_facet_cache(dt.tds());
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
        let dt = create_test_triangulation();

        // Initial state: no cache
        assert!(
            provider.facet_cache().load().is_none(),
            "Cache should be empty initially"
        );

        // First call to try_build_cache_with_rcu should build cache successfully
        let old_value_result = provider.try_build_cache_with_rcu(dt.tds());
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
        let second_old_result = provider.try_build_cache_with_rcu(dt.tds());
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
        let mut dt = create_test_triangulation();

        // Build initial cache
        let cache1_result = provider.try_get_or_build_facet_cache(dt.tds());
        assert!(cache1_result.is_ok(), "Initial cache build should succeed");

        let cache1 = cache1_result.unwrap();
        let ptr1 = Arc::as_ptr(&cache1);
        let initial_generation = dt.tds().generation();

        // Modify triangulation by adding a new vertex - this will bump the generation
        // Use an interior vertex away from the circumcenter to avoid degenerate insertion cases
        let new_vertex = vertex!([0.2, 0.2, 0.2]);
        dt.insert(new_vertex).expect("Failed to add vertex");

        // Verify generation was incremented
        let new_generation = dt.tds().generation();
        assert!(
            new_generation > initial_generation,
            "Generation should increase after adding vertex"
        );

        // Next call should rebuild cache due to generation change
        let cache2_result = provider.try_get_or_build_facet_cache(dt.tds());
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
    fn test_cache_methods_can_be_called_multiple_times() {
        let provider = TestCacheProvider::new();
        let dt = create_test_triangulation();

        // Smoke test: calling the cache getter multiple times should succeed.
        let _cache = provider.try_get_or_build_facet_cache(dt.tds()).unwrap();
        let cache_strict_result = provider.try_get_or_build_facet_cache(dt.tds());

        assert!(
            cache_strict_result.is_ok(),
            "Cache getter should succeed on repeated calls"
        );

        // Smoke test: calling the cache builder multiple times should succeed.
        provider.invalidate_facet_cache(); // Clear cache first
        let _old = provider.try_build_cache_with_rcu(dt.tds()).unwrap();
        let old_strict_result = provider.try_build_cache_with_rcu(dt.tds());

        assert!(
            old_strict_result.is_ok(),
            "Cache builder should succeed on repeated calls"
        );
    }

    #[test]
    fn test_cache_invalidation_idempotence() {
        let provider = TestCacheProvider::new();
        let dt = create_test_triangulation();

        // Build cache
        let _cache = provider.try_get_or_build_facet_cache(dt.tds()).unwrap();
        assert!(provider.facet_cache().load().is_some());

        // Invalidate multiple times
        provider.invalidate_facet_cache();
        assert!(provider.facet_cache().load().is_none());
        assert_eq!(provider.cached_generation().load(Ordering::Relaxed), 0);

        provider.invalidate_facet_cache();
        assert!(provider.facet_cache().load().is_none());
        assert_eq!(provider.cached_generation().load(Ordering::Relaxed), 0);

        provider.invalidate_facet_cache();
        assert!(provider.facet_cache().load().is_none());
        assert_eq!(provider.cached_generation().load(Ordering::Relaxed), 0);

        // Should still be able to rebuild
        let cache = provider.try_get_or_build_facet_cache(dt.tds()).unwrap();
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_cache_race_condition_on_invalidation() {
        // Test that cache properly handles race between get and invalidate
        let provider = Arc::new(TestCacheProvider::new());
        let dt = Arc::new(create_test_triangulation());
        let barrier = Arc::new(Barrier::new(2));

        // Build initial cache
        let _initial_cache = provider.try_get_or_build_facet_cache(dt.tds()).unwrap();

        let provider_clone = Arc::clone(&provider);
        let dt_clone = Arc::clone(&dt);
        let barrier_clone = Arc::clone(&barrier);

        // Thread 1: Repeatedly get cache
        let getter_thread = thread::spawn(move || {
            barrier_clone.wait();
            for _ in 0..100 {
                provider_clone
                    .try_get_or_build_facet_cache(dt_clone.tds())
                    .expect("cache retrieval should succeed during invalidation race");
                thread::sleep(Duration::from_micros(10));
            }
        });

        // Thread 2: Repeatedly invalidate
        let invalidator_thread = thread::spawn(move || {
            barrier.wait();
            for _ in 0..50 {
                provider.invalidate_facet_cache();
                thread::sleep(Duration::from_micros(20));
            }
        });

        // Both threads should complete without panicking
        getter_thread.join().expect("Getter thread should complete");
        invalidator_thread
            .join()
            .expect("Invalidator thread should complete");
    }

    #[test]
    fn test_cache_with_modified_tds_during_build() {
        // Test edge case where TDS is modified while cache is being built
        let provider = TestCacheProvider::new();
        let mut dt = create_test_triangulation();
        let initial_gen = dt.tds().generation();

        // Build cache
        let cache1 = provider.try_get_or_build_facet_cache(dt.tds()).unwrap();

        // Modify triangulation multiple times rapidly with unique coordinates
        let test_vertices = [
            // Interior point away from circumcenter to reduce degeneracy
            vertex!([0.2, 0.2, 0.2]),
            vertex!([0.3, 0.3, 0.1]), // Another interior point
            vertex!([0.2, 0.1, 0.3]), // Third interior point
        ];
        for vertex in test_vertices {
            dt.insert(vertex).expect("Failed to add vertex");
        }

        let final_gen = dt.tds().generation();
        assert!(final_gen > initial_gen, "Generation should have increased");

        // Cache should be rebuilt with new generation
        let cache2 = provider.try_get_or_build_facet_cache(dt.tds()).unwrap();
        assert_ne!(
            Arc::as_ptr(&cache1),
            Arc::as_ptr(&cache2),
            "Cache should be different after TDS modifications"
        );

        // Verify generation is tracked correctly
        assert_eq!(
            provider.cached_generation().load(Ordering::Relaxed),
            final_gen,
            "Cached generation should match final TDS generation"
        );
    }

    #[test]
    fn test_cache_methods_succeed_on_valid_tds() {
        // Smoke test that cache methods succeed on a valid triangulation.
        let provider = TestCacheProvider::new();
        let dt = create_test_triangulation();

        // Methods should succeed even with potential errors
        let cache = provider.try_get_or_build_facet_cache(dt.tds()).unwrap();
        assert!(
            !cache.is_empty(),
            "Cache getter should return a non-empty cache"
        );

        // Build using RCU should also succeed
        provider.invalidate_facet_cache();
        let old_value = provider.try_build_cache_with_rcu(dt.tds()).unwrap();
        assert!(old_value.is_none(), "Should return None for first build");

        // Verify cache exists
        assert!(provider.facet_cache().load().is_some());
    }

    #[test]
    fn test_cache_size_consistency_after_operations() {
        // Verify cache size remains consistent across different operations
        let provider = TestCacheProvider::new();
        let mut dt = create_test_triangulation();

        // Initial cache
        let cache1 = provider.try_get_or_build_facet_cache(dt.tds()).unwrap();
        let size1 = cache1.len();

        // Invalidate and rebuild - should get same size
        provider.invalidate_facet_cache();
        let cache2 = provider.try_get_or_build_facet_cache(dt.tds()).unwrap();
        assert_eq!(
            cache2.len(),
            size1,
            "Size should be consistent after invalidate/rebuild"
        );

        // Add vertex - size should change
        // Use an interior vertex away from circumcenter to avoid degenerate insertion cases
        dt.insert(vertex!([0.2, 0.2, 0.2]))
            .expect("Failed to add vertex");
        let cache3 = provider.try_get_or_build_facet_cache(dt.tds()).unwrap();
        // Size might be different after adding a vertex (more cells = more facets)
        assert!(
            cache3.len() >= size1,
            "Cache size should grow or stay same after adding vertex"
        );
    }

    #[test]
    fn test_cache_generation_ordering_semantics() {
        // Test that Acquire/Release ordering works correctly
        let provider = Arc::new(TestCacheProvider::new());
        let dt = Arc::new(create_test_triangulation());

        // Build cache
        let _cache = provider.try_get_or_build_facet_cache(dt.tds()).unwrap();
        let gen_after_build = provider.cached_generation().load(Ordering::Acquire);
        assert_eq!(gen_after_build, dt.tds().generation());

        // Invalidate
        provider.invalidate_facet_cache();
        let gen_after_invalidate = provider.cached_generation().load(Ordering::Acquire);
        assert_eq!(
            gen_after_invalidate, 0,
            "Generation should be reset after invalidate"
        );

        // Rebuild
        let _cache2 = provider.try_get_or_build_facet_cache(dt.tds()).unwrap();
        let gen_after_rebuild = provider.cached_generation().load(Ordering::Acquire);
        assert_eq!(gen_after_rebuild, dt.tds().generation());
    }
}
