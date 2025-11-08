//! Integration tests for facet cache behavior under real-world usage
//!
//! These tests exercise the facet cache implementation through actual algorithm
//! implementations (`IncrementalBowyerWatson`, `ConvexHull`) to cover complex code
//! paths that are difficult to trigger via unit tests:
//!
//! - Concurrent cache access during TDS modifications
//! - Cache invalidation during incremental building
//! - RCU contention with multiple threads
//! - Generation tracking through insertion operations
//! - Retry loops when TDS generation changes mid-build

use delaunay::{
    core::{
        algorithms::bowyer_watson::IncrementalBowyerWatson,
        traits::{facet_cache::FacetCacheProvider, insertion_algorithm::InsertionAlgorithm},
        triangulation_data_structure::Tds,
    },
    geometry::algorithms::convex_hull::ConvexHull,
    vertex,
};
use std::{
    sync::{Arc, Barrier},
    thread,
};

/// Test sequential cache invalidation during vertex insertion operations.
///
/// This test verifies cache consistency when vertices are inserted sequentially
/// with explicit invalidation between insertions:
/// - Invalidate cache
/// - Insert vertex (modifies TDS, increments generation)
/// - Rebuild and query the facet cache
///
/// Expected behavior:
/// - Cache rebuilds correctly after each insertion
/// - Generation tracking works correctly
/// - Cache remains consistent with TDS state
#[test]
fn test_sequential_cache_invalidation_during_insertion() {
    // Create initial triangulation with a few vertices
    let initial_vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
    let mut algorithm = IncrementalBowyerWatson::new();

    // Initial cache build
    let _cache = algorithm.try_get_or_build_facet_cache(&tds).unwrap();

    // Create vertices to insert
    let new_vertices: Vec<_> = (0..10)
        .map(|i| {
            let t = f64::from(i) * 0.1;
            vertex!([0.5 + t, 0.5 - t, t.mul_add(0.5, 0.5)])
        })
        .collect();

    // Insert vertices and verify cache consistency after each insertion
    for v in new_vertices {
        // Invalidate cache before insertion
        algorithm.invalidate_facet_cache();

        // Insert vertex (modifies TDS, changes generation)
        let _ = algorithm.insert_vertex(&mut tds, v).unwrap();

        // Verify cache rebuilds correctly
        let cache = algorithm.try_get_or_build_facet_cache(&tds).unwrap();
        assert!(!cache.is_empty());

        // Verify generation consistency
        assert_eq!(
            algorithm
                .cached_generation()
                .load(std::sync::atomic::Ordering::Acquire),
            tds.generation()
        );
    }

    // Final verification: cache is consistent
    let final_cache = algorithm.try_get_or_build_facet_cache(&tds).unwrap();
    assert!(!final_cache.is_empty());

    // Verify cache size is reasonable (each cell has D+1 facets)
    let cell_count = tds.cells().count();
    assert!(final_cache.len() <= cell_count * (3 + 1));
}

/// Test cache invalidation during incremental building.
///
/// This exercises the code paths where:
/// - Cache is invalidated via `invalidate_facet_cache`
/// - Rebuild happens via `try_build_cache_with_rcu`
/// - Generation check fails and retry occurs in the main retry loop
#[test]
fn test_cache_invalidation_during_incremental_building() {
    let initial_vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
    let mut algorithm = IncrementalBowyerWatson::new();

    // Build initial cache
    let _cache = algorithm.try_get_or_build_facet_cache(&tds).unwrap();
    let initial_generation = tds.generation();

    // Insert vertices with forced invalidation after each insertion
    for i in 0..5 {
        let v = vertex!([f64::from(i).mul_add(0.1, 0.3), 0.3, 0.3]);

        // Invalidate cache before insertion
        algorithm.invalidate_facet_cache();

        // Insert vertex (modifies TDS, changes generation)
        let _ = algorithm.insert_vertex(&mut tds, v).unwrap();

        // Verify generation increased
        assert!(tds.generation() > initial_generation);

        // Force another invalidation immediately
        algorithm.invalidate_facet_cache();

        // Try to get cache - should rebuild
        let cache = algorithm.try_get_or_build_facet_cache(&tds).unwrap();
        assert!(!cache.is_empty());

        // Verify cached_generation matches TDS generation
        assert_eq!(
            algorithm
                .cached_generation()
                .load(std::sync::atomic::Ordering::Acquire),
            tds.generation()
        );
    }
}

/// Test RCU contention with multiple threads building cache simultaneously.
///
/// This exercises the RCU mechanism in `try_build_cache_with_rcu()`
/// to ensure only one thread actually builds the cache while others wait and reuse.
#[test]
#[expect(clippy::needless_collect)] // Collect handles to consume iterator before joining threads
fn test_rcu_contention_multiple_threads() {
    const NUM_THREADS: usize = 10;

    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
        vertex!([0.5, 0.5, 0.5]),
    ];
    let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    let tds_arc = Arc::new(tds);
    let algorithm = Arc::new(IncrementalBowyerWatson::new());

    // Ensure cache is empty initially
    algorithm.invalidate_facet_cache();

    let barrier = Arc::new(Barrier::new(NUM_THREADS));

    // Spawn threads that all try to build cache simultaneously
    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|_| {
            let algorithm_clone = Arc::clone(&algorithm);
            let tds_clone = Arc::clone(&tds_arc);
            let barrier_clone = Arc::clone(&barrier);

            thread::spawn(move || {
                // Synchronize start to maximize contention
                barrier_clone.wait();

                // All threads try to get cache at the same time
                let cache = algorithm_clone
                    .try_get_or_build_facet_cache(&tds_clone)
                    .unwrap();

                // Verify cache is valid
                assert!(!cache.is_empty());

                cache
            })
        })
        .collect();

    // Wait for all threads to complete and collect their caches
    let caches: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All caches should be valid and non-empty
    assert_eq!(caches.len(), NUM_THREADS, "All threads should complete");
    for cache in &caches {
        assert!(!cache.is_empty(), "Each cache should be non-empty");
    }

    // Most threads should share the same Arc (RCU minimizes duplicate builds)
    // Note: Due to timing, some threads might get different Arc instances,
    // but the content should be identical
    let first_cache = &caches[0];
    for cache in &caches[1..] {
        assert_eq!(
            cache.len(),
            first_cache.len(),
            "All caches should have same size"
        );
    }

    // Verify that most threads reused the same Arc (RCU efficiency)
    let shared = caches
        .iter()
        .filter(|c| Arc::ptr_eq(c, first_cache))
        .count();
    assert!(
        shared >= NUM_THREADS / 2,
        "Most threads should share the same Arc (found {shared}/{NUM_THREADS})"
    );
}

/// Test generation tracking through insertion operations.
///
/// Verifies that:
/// - Cache generation increments correctly after TDS modifications
/// - Multiple cache accesses without TDS changes reuse the same cache
/// - Cache invalidation resets generation appropriately
#[test]
fn test_generation_tracking_through_insertions() {
    let initial_vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
    let mut algorithm = IncrementalBowyerWatson::new();

    // Initial cache build
    let cache1 = algorithm.try_get_or_build_facet_cache(&tds).unwrap();
    let gen1 = algorithm
        .cached_generation()
        .load(std::sync::atomic::Ordering::Acquire);

    // Multiple accesses without TDS changes should return same cache
    let cache2 = algorithm.try_get_or_build_facet_cache(&tds).unwrap();
    let gen2 = algorithm
        .cached_generation()
        .load(std::sync::atomic::Ordering::Acquire);
    assert_eq!(gen1, gen2);
    assert_eq!(Arc::as_ptr(&cache1), Arc::as_ptr(&cache2));

    // Insert vertex (changes TDS generation)
    let v = vertex!([0.5, 0.5, 0.5]);
    let _ = algorithm.insert_vertex(&mut tds, v).unwrap();

    // Cache should be stale now
    let cache3 = algorithm.try_get_or_build_facet_cache(&tds).unwrap();
    let gen3 = algorithm
        .cached_generation()
        .load(std::sync::atomic::Ordering::Acquire);

    // Generation should have increased
    assert!(gen3 > gen1);
    // Cache should be different (rebuilt)
    assert_ne!(Arc::as_ptr(&cache1), Arc::as_ptr(&cache3));
}

/// Test `ConvexHull` cache behavior during incremental construction.
///
/// `ConvexHull` also implements `FacetCacheProvider`, so this ensures the trait
/// works correctly with different algorithm implementations.
#[test]
fn test_convex_hull_cache_during_construction() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
        vertex!([0.5, 0.5, 0.5]),
        vertex!([0.2, 0.3, 0.4]),
    ];

    let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    let hull = ConvexHull::from_triangulation(&tds).unwrap();

    // Access cache multiple times
    for _ in 0..5 {
        let cache = hull.try_get_or_build_facet_cache(&tds).unwrap();
        assert!(!cache.is_empty());
    }

    // Invalidate and rebuild
    hull.invalidate_facet_cache();
    let cache_after_invalidation = hull.try_get_or_build_facet_cache(&tds).unwrap();
    assert!(!cache_after_invalidation.is_empty());

    // Verify cached generation matches TDS
    assert_eq!(
        hull.cached_generation()
            .load(std::sync::atomic::Ordering::Acquire),
        tds.generation()
    );
}

/// Test retry loop when generation changes during cache build.
///
/// This exercises the stale cache rebuild path in `try_get_or_build_facet_cache`
/// by repeatedly invalidating and rebuilding the cache while checking that the
/// generation tracking remains consistent.
#[test]
fn test_retry_loop_on_generation_change() {
    let initial_vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
    let mut algorithm = IncrementalBowyerWatson::new();

    // Clear cache to force rebuild
    algorithm.invalidate_facet_cache();

    // Repeatedly insert vertices and check cache consistency
    for i in 0..10 {
        let v = vertex!([f64::from(i).mul_add(0.1, 0.2), 0.1, 0.1]);

        // Invalidate before insertion to force stale cache path
        algorithm.invalidate_facet_cache();

        // Insert vertex (changes generation)
        let _ = algorithm.insert_vertex(&mut tds, v);

        // Try to get cache - should trigger rebuild with new generation
        let cache = algorithm.try_get_or_build_facet_cache(&tds).unwrap();
        assert!(!cache.is_empty());

        // Verify generation consistency
        let cached_gen = algorithm
            .cached_generation()
            .load(std::sync::atomic::Ordering::Acquire);
        assert_eq!(cached_gen, tds.generation());
    }

    // Final verification
    let final_cache = algorithm.try_get_or_build_facet_cache(&tds).unwrap();
    assert!(!final_cache.is_empty());
}

/// Test cache consistency under rapid invalidation cycles.
///
/// Exercises the invalidation logic (lines 430-440) and verifies that
/// rapid invalidate-rebuild cycles maintain consistency.
#[test]
fn test_rapid_invalidation_cycles() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    let algorithm = IncrementalBowyerWatson::new();

    // Rapid invalidate-rebuild cycles
    for _ in 0..50 {
        algorithm.invalidate_facet_cache();
        let cache = algorithm.try_get_or_build_facet_cache(&tds).unwrap();
        assert!(!cache.is_empty());

        // Verify cache is actually empty after invalidation
        algorithm.invalidate_facet_cache();
        assert!(algorithm.facet_cache().load().is_none());

        // Verify cached_generation is reset
        assert_eq!(
            algorithm
                .cached_generation()
                .load(std::sync::atomic::Ordering::Acquire),
            0
        );
    }
}
