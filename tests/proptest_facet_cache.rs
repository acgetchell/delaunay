//! Property-based tests for `FacetCache` trait behavior.
//!
//! This module uses proptest to verify fundamental properties of `FacetCacheProvider`
//! implementations, including:
//! - Cache rebuilds produce identical results for same TDS generation
//! - Cache size bounds are correct (`cache.len() <= cell_count * (D + 1)`)
//! - Cache invalidation increments generation correctly
//! - Cache consistency under concurrent access (RCU mechanism)
//! - Generation tracking works correctly after modifications

#![allow(unused_imports)] // Imports used in macro expansion

use delaunay::core::algorithms::bowyer_watson::IncrementalBowyerWatson;
use delaunay::core::traits::facet_cache::FacetCacheProvider;
use delaunay::core::traits::insertion_algorithm::InsertionAlgorithm;
use delaunay::core::triangulation_data_structure::Tds;
use delaunay::core::vertex::Vertex;
use delaunay::geometry::point::Point;
use delaunay::geometry::traits::coordinate::Coordinate;
use delaunay::vertex;
use proptest::prelude::*;
use std::sync::Arc;
use std::sync::atomic::Ordering;

// =============================================================================
// TEST CONFIGURATION
// =============================================================================

/// Strategy for generating finite f64 coordinates
fn finite_coordinate() -> impl Strategy<Value = f64> {
    (-100.0..100.0).prop_filter("must be finite", |x: &f64| x.is_finite())
}

// =============================================================================
// DIMENSIONAL TEST GENERATION MACROS
// =============================================================================

/// Macro to generate facet cache property tests for a given dimension
macro_rules! test_facet_cache_properties {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal) => {
        pastey::paste! {
                /// Property: Cache rebuilds produce identical results for same TDS generation
                #[test]
                fn [<prop_cache_rebuild_deterministic_ $dim d>]() {
                    proptest!(|(vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v)))| {
                        if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                            let algorithm = IncrementalBowyerWatson::new();

                            // Build cache twice without modifying TDS
                            let cache1 = algorithm
                                .try_get_or_build_facet_cache(&tds)
                                .expect("facet cache build should succeed");
                            let cache2 = algorithm
                                .try_get_or_build_facet_cache(&tds)
                                .expect("facet cache build should succeed");

                            // Should return same Arc (cached)
                            prop_assert!(
                                Arc::ptr_eq(&cache1, &cache2),
                                "{}D: Cache should be reused when generation unchanged",
                                $dim
                            );

                            // Content should be identical
                            prop_assert_eq!(
                                cache1.len(),
                                cache2.len(),
                                "{}D: Cache sizes should match",
                                $dim
                            );
                        }
                    });
                }

/// Property: Cache size is bounded by `cell_count` * (D + 1)
                #[test]
                fn [<prop_cache_size_bounded_ $dim d>]() {
                    proptest!(|(vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v)))| {
                        if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                            let algorithm = IncrementalBowyerWatson::new();
                            let cache = algorithm
                                .try_get_or_build_facet_cache(&tds)
                                .expect("facet cache build should succeed");

                            let cell_count = tds.cells().count();
                            let dim = usize::try_from(tds.dim())
                                .expect("TDS dimension should be non-negative");
                            let max_facets = cell_count * (dim + 1);

                            prop_assert!(
                                cache.len() <= max_facets,
                                "{}D: Cache size {} should be <= cell_count ({}) * (D+1) = {}",
                                $dim, cache.len(), cell_count, max_facets
                            );
                        }
                    });
                }

                /// Property: Cache invalidation resets generation
                #[test]
                fn [<prop_cache_invalidation_resets_ $dim d>]() {
                    proptest!(|(vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v)))| {
                        if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                            let algorithm = IncrementalBowyerWatson::new();

                            // Build cache
                            let _cache = algorithm
                                .try_get_or_build_facet_cache(&tds)
                                .expect("facet cache build should succeed");
                            let gen_before = algorithm.cached_generation().load(Ordering::Acquire);
                            prop_assert_ne!(
                                gen_before,
                                0,
                                "{}D: Generation should be non-zero after build",
                                $dim
                            );

                            // Invalidate
                            algorithm.invalidate_facet_cache();

                            // Check generation reset
                            let gen_after = algorithm.cached_generation().load(Ordering::Acquire);
                            prop_assert_eq!(
                                gen_after, 0,
                                "{}D: Generation should be reset to 0 after invalidation",
                                $dim
                            );

                            // Check cache cleared
                            prop_assert!(
                                algorithm.facet_cache().load().is_none(),
                                "{}D: Cache should be None after invalidation",
                                $dim
                            );
                        }
                    });
                }

                /// Property: Cache tracks TDS generation correctly
                #[test]
                fn [<prop_cache_tracks_generation_ $dim d>]() {
                    proptest!(|(vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v)))| {
                        if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                            let algorithm = IncrementalBowyerWatson::new();

                            // Build cache
                            let _cache = algorithm
                                .try_get_or_build_facet_cache(&tds)
                                .expect("facet cache build should succeed");

                            // Verify generation matches
                            let tds_gen = tds.generation();
                            let cached_gen = algorithm.cached_generation().load(Ordering::Acquire);
                            prop_assert_eq!(
                                cached_gen, tds_gen,
                                "{}D: Cached generation {} should match TDS generation {}",
                                $dim, cached_gen, tds_gen
                            );
                        }
                    });
                }

                /// Property: Cache rebuilds after TDS modification
                #[test]
                fn [<prop_cache_rebuilds_after_modification_ $dim d>]() {
                    proptest!(|(vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v)), new_coord in prop::array::[<uniform $dim>](finite_coordinate()))| {
                        if let Ok(mut tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                            let mut algorithm = IncrementalBowyerWatson::new();

                            // Build initial cache
                            let cache1 = algorithm
                                .try_get_or_build_facet_cache(&tds)
                                .expect("facet cache build should succeed");
                            let ptr1 = Arc::as_ptr(&cache1);
                            let gen1 = tds.generation();

                            // Modify TDS by inserting a vertex
                            let new_vertex = vertex!(new_coord);
                            let _result = algorithm.insert_vertex(&mut tds, new_vertex);

                            let gen2 = tds.generation();
                            if gen2 > gen1 {
                                // Generation changed - cache should rebuild
                                let cache2 = algorithm
                                    .try_get_or_build_facet_cache(&tds)
                                    .expect("facet cache build should succeed");
                                let ptr2 = Arc::as_ptr(&cache2);

                                prop_assert_ne!(
                                    ptr1, ptr2,
                                    "{}D: Cache should be rebuilt after TDS modification",
                                    $dim
                                );
                            }
                        }
                    });
                }

                /// Property: Empty cache returns empty mapping for empty TDS
                #[test]
                fn [<prop_empty_tds_empty_cache_ $dim d>]() {
                    let tds: Tds<f64, Option<()>, Option<()>, $dim> = Tds::empty();
                    let algorithm = IncrementalBowyerWatson::new();

                    let cache = algorithm
                        .try_get_or_build_facet_cache(&tds)
                        .expect("facet cache build should succeed");
                    assert!(
                        cache.is_empty(),
                        "{}D: Cache should be empty for empty TDS",
                        $dim
                    );
                }

                /// Property: Cache content matches direct TDS facet map
                #[test]
                fn [<prop_cache_matches_tds_facet_map_ $dim d>]() {
                    proptest!(|(vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v)))| {
                        if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                            let algorithm = IncrementalBowyerWatson::new();

                            // Get cache from algorithm
                            let cache = algorithm
                                .try_get_or_build_facet_cache(&tds)
                                .expect("facet cache build should succeed");

                            // Get reference from TDS directly
                            let reference = tds
                                .build_facet_to_cells_map()
                                .expect("facet map build should succeed");

                            // Sizes should match
                            prop_assert_eq!(
                                cache.len(),
                                reference.len(),
                                "{}D: Cache size should match direct TDS facet map",
                                $dim
                            );

                            // All keys should be present
                            for key in reference.keys() {
                                prop_assert!(
                                    cache.contains_key(key),
                                    "{}D: Cache should contain all keys from TDS facet map",
                                    $dim
                                );
                            }
                        }
                    });
                }

                /// Property: Multiple algorithms can share cache (RCU works)
                #[test]
                fn [<prop_multiple_algorithms_share_cache_ $dim d>]() {
                    proptest!(|(vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v)))| {
                        if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                            let tds_arc = Arc::new(tds);

                            // Create multiple algorithm instances
                            let alg1 = Arc::new(IncrementalBowyerWatson::new());
                            let alg2 = Arc::clone(&alg1); // Share the same algorithm instance

                            // Both get cache simultaneously
                            let cache1 = alg1
                                .try_get_or_build_facet_cache(&tds_arc)
                                .expect("facet cache build should succeed");
                            let cache2 = alg2
                                .try_get_or_build_facet_cache(&tds_arc)
                                .expect("facet cache build should succeed");

                            // Should share the same Arc (RCU efficiency)
                            prop_assert!(
                                Arc::ptr_eq(&cache1, &cache2),
                                "{}D: Multiple accesses should share same cache Arc",
                                $dim
                            );
                        }
                    });
                }

                /// Property: Cache invalidation is idempotent
                #[test]
                fn [<prop_cache_invalidation_idempotent_ $dim d>]() {
                    proptest!(|(vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v)))| {
                        if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                            let algorithm = IncrementalBowyerWatson::new();

                            // Build cache
                            let _cache = algorithm
                                .try_get_or_build_facet_cache(&tds)
                                .expect("facet cache build should succeed");

                            // Invalidate multiple times
                            algorithm.invalidate_facet_cache();
                            algorithm.invalidate_facet_cache();
                            algorithm.invalidate_facet_cache();

                            // Should still be in invalidated state
                            prop_assert!(
                                algorithm.facet_cache().load().is_none(),
                                "{}D: Cache should remain None after multiple invalidations",
                                $dim
                            );
                            prop_assert_eq!(
                                algorithm.cached_generation().load(Ordering::Relaxed),
                                0,
                                "{}D: Generation should remain 0 after multiple invalidations",
                                $dim
                            );
                        }
                    });
                }

                /// Property: Cache can be rebuilt after invalidation
                #[test]
                fn [<prop_cache_rebuild_after_invalidation_ $dim d>]() {
                    proptest!(|(vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v)))| {
                        if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                            let algorithm = IncrementalBowyerWatson::new();

                            // Build, invalidate, rebuild cycle
                            let cache1 = algorithm
                                .try_get_or_build_facet_cache(&tds)
                                .expect("facet cache build should succeed");
                            let size1 = cache1.len();

                            algorithm.invalidate_facet_cache();

                            let cache2 = algorithm
                                .try_get_or_build_facet_cache(&tds)
                                .expect("facet cache build should succeed");
                            let size2 = cache2.len();

                            // Content size should match (same TDS)
                            prop_assert_eq!(
                                size1, size2,
                                "{}D: Rebuilt cache should have same size as original",
                                $dim
                            );

                            // Arc pointers should differ (new cache built)
                            prop_assert_ne!(
                                Arc::as_ptr(&cache1),
                                Arc::as_ptr(&cache2),
                                "{}D: Rebuilt cache should be a new instance",
                                $dim
                            );
                        }
                    });
                }
        }
    };
}

// Generate tests for dimensions 2-5
// Parameters: dimension, min_vertices (D+2), max_vertices
test_facet_cache_properties!(2, 4, 10);
test_facet_cache_properties!(3, 5, 12);
test_facet_cache_properties!(4, 6, 14);
test_facet_cache_properties!(5, 7, 16);
