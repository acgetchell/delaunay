#![cfg_attr(feature = "slow-tests", expect(deprecated))]
//! Integration tests for `RobustBowyerWatson` algorithm.
//!
//! These tests verify end-to-end behavior of the robust insertion algorithm
//! across dimensions 2D-5D, including:
//! - Large random point set insertions
//! - Exterior vertex insertion sequences (hull extension)
//! - Clustered point patterns
//! - Algorithm reset and reuse

use delaunay::core::algorithms::robust_bowyer_watson::RobustBowyerWatson;
use delaunay::core::traits::insertion_algorithm::InsertionAlgorithm;
use delaunay::core::triangulation_data_structure::Tds;
use delaunay::vertex;

#[cfg(feature = "slow-tests")]
use rand::rngs::StdRng;
#[cfg(feature = "slow-tests")]
use rand::{Rng, SeedableRng};

// =============================================================================
// DIMENSIONAL TEST GENERATION MACRO
// =============================================================================

/// Macro to generate integration tests for a given dimension
#[cfg(feature = "slow-tests")]
macro_rules! test_robust_integration {
    ($dim:literal, $num_random_points:literal) => {
        test_robust_integration!(@impl $dim, $num_random_points, );
    };
    ($dim:literal, $num_random_points:literal, $(#[$attr:meta])*) => {
        test_robust_integration!(@impl $dim, $num_random_points, $(#[$attr])*);
    };
    (@impl $dim:literal, $num_random_points:literal, $(#[$attr:meta])*) => {
        pastey::paste! {
            /// Test: Large random point set insertion
            $(#[$attr])*
            #[test]
            fn [<test_large_random_point_set_ $dim d>]() {
                // Use seeded RNG for reproducibility
                let mut rng = StdRng::seed_from_u64(42);
                let mut algorithm = RobustBowyerWatson::new();

                // Build vertex set: start with a simple initial simplex, then add random points.
                let mut vertices = create_initial_simplex::<$dim>();

                for _ in 0..$num_random_points {
                    let coords: [f64; $dim] =
                        std::array::from_fn(|_| rng.random_range(-5.0..15.0));
                    vertices.push(vertex!(coords));
                }

                // Triangulate using the robust algorithm's high-level API. This exercises
                // the robust insertion pipeline together with global Delaunay repair.
                let mut tds: Tds<f64, Option<()>, Option<()>, $dim> = Tds::empty();
                algorithm
                    .triangulate(&mut tds, &vertices)
                    .unwrap_or_else(|err| {
                        panic!(
                            "{}D: robust triangulation failed for large random point set: {err}",
                            $dim,
                        )
                    });

                // Verify final state
                assert!(
                    tds.is_valid().is_ok(),
                    "{}D: Final TDS should be structurally valid",
                    $dim
                );

                let delaunay_result = tds.validate_delaunay();
                if let Err(err) = delaunay_result {
                    eprintln!(
                        "{}D: Final TDS failed global Delaunay validation: {err:?}",
                        $dim
                    );
                    #[cfg(debug_assertions)]
                    {
                        use delaunay::core::util::debug_print_first_delaunay_violation;
                        debug_print_first_delaunay_violation(&tds, None);
                    }
                    panic!(
                        "{}D: Final TDS should be globally Delaunay",
                        $dim
                    );
                }

                let (processed, created, removed) = algorithm.get_statistics();
                assert!(processed > 0, "{}D: Should have processed vertices", $dim);
                println!(
                    "{}D large dataset: processed={}, created={}, removed={}",
                    $dim, processed, created, removed
                );
            }

            /// Test: Exterior vertex insertion sequence
            $(#[$attr])*
            #[test]
            fn [<test_exterior_vertex_insertion_sequence_ $dim d>]() {
                let mut algorithm = RobustBowyerWatson::new();

                let initial_vertices = create_initial_simplex::<$dim>();
                let mut tds: Tds<f64, Option<()>, Option<()>, $dim> =
                    Tds::new(&initial_vertices).unwrap();

                let initial_cells = tds.number_of_cells();
                assert!(initial_cells > 0 || $dim < 3, "{}D: Should have cells or be low dimension", $dim);

                // Insert interior point first
                let interior_coords: [f64; $dim] = [0.25; $dim];
                let interior_vertex = vertex!(interior_coords);
                let result = algorithm.insert_vertex(&mut tds, interior_vertex);
                if result.is_ok() {
                    assert!(tds.is_valid().is_ok(), "{}D: TDS valid after interior insertion", $dim);
                }

                // Insert exterior points along each axis
                for axis in 0..$dim {
                    let mut coords = [0.5; $dim];
                    coords[axis] = 2.0;
                    let ext_vertex = vertex!(coords);

                    let result = algorithm.insert_vertex(&mut tds, ext_vertex);

                    assert!(
                        tds.is_valid().is_ok(),
                        "{}D: TDS should remain valid after exterior vertex insertion on axis {}",
                        $dim,
                        axis
                    );

                    if result.is_ok() {
                        println!("{}D: Successfully inserted exterior point on axis {}", $dim, axis);
                    }
                }

                assert!(
                    tds.validate_delaunay().is_ok(),
                    "{}D: Final TDS should be globally Delaunay after exterior insertions",
                    $dim
                );
                println!(
                    "{}D: Final TDS has {} vertices, {} cells",
                    $dim,
                    tds.number_of_vertices(),
                    tds.number_of_cells()
                );
            }

            /// Test: Clustered points followed by scattered points
            $(#[$attr])*
            #[test]
            fn [<test_clustered_points_ $dim d>]() {
                let mut algorithm = RobustBowyerWatson::new();
                // Use seeded RNG for reproducibility
                let mut rng = StdRng::seed_from_u64(12345);

                // Phase 1: initial simplex + clustered points near the origin.
                let mut clustered_vertices = create_initial_simplex::<$dim>();
                for _ in 0..30 {
                    let coords: [f64; $dim] =
                        std::array::from_fn(|_| rng.random_range(-0.5..0.5));
                    clustered_vertices.push(vertex!(coords));
                }

                let mut tds_cluster: Tds<f64, Option<()>, Option<()>, $dim> = Tds::empty();
                algorithm
                    .triangulate(&mut tds_cluster, &clustered_vertices)
                    .unwrap_or_else(|err| {
                        panic!(
                            "{}D: robust triangulation failed for clustered points: {err}",
                            $dim,
                        )
                    });

                assert!(
                    tds_cluster.is_valid().is_ok(),
                    "{}D: Clustered-phase TDS should be structurally valid",
                    $dim
                );
                assert!(
                    tds_cluster.validate_delaunay().is_ok(),
                    "{}D: After clustered insertions, TDS should be globally Delaunay before scattering",
                    $dim
                );

                // Phase 2: add scattered points farther away and triangulate again.
                let mut all_vertices = clustered_vertices;
                for _ in 0..30 {
                    let coords: [f64; $dim] =
                        std::array::from_fn(|_| rng.random_range(-20.0..20.0));
                    all_vertices.push(vertex!(coords));
                }

                let mut tds_all: Tds<f64, Option<()>, Option<()>, $dim> = Tds::empty();
                algorithm
                    .triangulate(&mut tds_all, &all_vertices)
                    .unwrap_or_else(|err| {
                        panic!(
                            "{}D: robust triangulation failed for clustered+scattered points: {err}",
                            $dim,
                        )
                    });

                assert!(
                    tds_all.is_valid().is_ok(),
                    "{}D: Final TDS should be structurally valid",
                    $dim
                );
                assert!(
                    tds_all.validate_delaunay().is_ok(),
                    "{}D: Final TDS should be globally Delaunay after clustered+scattered insertions",
                    $dim
                );
                println!(
                    "{}D clustered+scattered: {} vertices, {} cells",
                    $dim,
                    tds_all.number_of_vertices(),
                    tds_all.number_of_cells()
                );
            }
        }
    };
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Create initial simplex for dimension D
#[cfg(feature = "slow-tests")]
fn create_initial_simplex<const D: usize>()
-> Vec<delaunay::core::vertex::Vertex<f64, Option<()>, D>> {
    let mut vertices = Vec::with_capacity(D + 1);

    // Add origin
    vertices.push(vertex!([0.0; D]));

    // Add D more vertices with one coordinate = 10.0
    for i in 0..D {
        let mut coords = [0.0; D];
        coords[i] = 10.0;
        vertices.push(vertex!(coords));
    }

    vertices
}

// =============================================================================
// GENERATE TESTS FOR DIMENSIONS 2D-5D
// =============================================================================

// Parameters: dimension, num_random_points
// These tests now pass with the increased global repair limits (128 iterations, 32 repairs/iteration)
// They are gated behind the 'slow-tests' feature to avoid CI timeouts.
// Run with: cargo test --features slow-tests
#[cfg(feature = "slow-tests")]
test_robust_integration!(2, 100);
#[cfg(feature = "slow-tests")]
test_robust_integration!(3, 100);
#[cfg(feature = "slow-tests")]
test_robust_integration!(4, 50);

// 5D tests are extremely slow even for slow-tests - run manually with --ignored
#[cfg(feature = "slow-tests")]
test_robust_integration!(5, 30, #[ignore = "5D tests are extremely slow - run manually"]);

// =============================================================================
// ADDITIONAL INTEGRATION TESTS (3D-specific)
// =============================================================================

#[test]
fn test_mixed_interior_exterior_insertions_3d() {
    let mut algorithm = RobustBowyerWatson::new();

    // Start with tetrahedron and then add a mix of interior and exterior points.
    let mut vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([2.0, 0.0, 0.0]),
        vertex!([0.0, 2.0, 0.0]),
        vertex!([0.0, 0.0, 2.0]),
    ];

    let mixed_vertices = vec![
        (vertex!([0.5, 0.5, 0.5]), "interior"),
        (vertex!([3.0, 0.0, 0.0]), "exterior"),
        (vertex!([0.3, 0.3, 0.3]), "interior"),
        (vertex!([0.0, 3.0, 0.0]), "exterior"),
        (vertex!([0.7, 0.2, 0.1]), "interior"),
        (vertex!([0.0, 0.0, 3.0]), "exterior"),
        (vertex!([1.0, 1.0, 1.0]), "interior"),
        (vertex!([-1.0, -1.0, -1.0]), "exterior"),
    ];

    for (vertex, _label) in &mixed_vertices {
        vertices.push(*vertex);
    }

    let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::empty();
    algorithm
        .triangulate(&mut tds, &vertices)
        .unwrap_or_else(|err| {
            panic!("3D: robust triangulation failed for mixed interior/exterior insertions: {err}");
        });

    assert!(
        tds.is_valid().is_ok(),
        "3D: Final TDS should be structurally valid after mixed interior/exterior insertions"
    );
    assert!(
        tds.validate_delaunay().is_ok(),
        "3D: Final TDS should be globally Delaunay after mixed interior/exterior insertions"
    );

    // Verify statistics
    let (processed, created, removed) = algorithm.get_statistics();
    assert!(processed > 0, "Should have processed vertices");
    println!("Mixed insertions: processed={processed}, created={created}, removed={removed}");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "Slow test - enable with --features slow-tests"
)]
fn test_grid_pattern_insertion_2d() {
    let mut algorithm = RobustBowyerWatson::new();

    // Create full vertex set: initial triangle plus interior grid points.
    let mut vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([10.0, 0.0]),
        vertex!([0.0, 10.0]),
    ];

    for i in 1..10 {
        for j in 1..10 {
            let x = f64::from(i);
            let y = f64::from(j);
            vertices.push(vertex!([x, y]));
        }
    }

    let mut tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::empty();
    algorithm
        .triangulate(&mut tds, &vertices)
        .unwrap_or_else(|err| {
            panic!("2D: robust triangulation failed for grid pattern: {err}");
        });

    assert!(
        tds.is_valid().is_ok(),
        "2D: Final TDS should be structurally valid for grid pattern"
    );
    assert!(
        tds.validate_delaunay().is_ok(),
        "2D: Final TDS should be globally Delaunay for grid pattern"
    );

    println!(
        "Grid pattern: {} vertices, {} cells",
        tds.number_of_vertices(),
        tds.number_of_cells()
    );
}

#[test]
fn test_degenerate_robust_configuration_3d() {
    let mut algorithm = RobustBowyerWatson::for_degenerate_cases();

    // Create initial tetrahedron plus points with small perturbations.
    // NOTE: The algorithm correctly rejects near-duplicate vertices (within 1e-10)
    // as unsalvageable, so triangulate() will skip them. This is the expected
    // robust behavior - we can't maintain valid Delaunay triangulations with
    // near-duplicate vertices due to numerical precision limits.
    let mut vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    let near_degenerate_vertices = [
        vertex!([1e-10, 1e-10, 1e-10]),
        vertex!([0.5 + 1e-11, 0.5, 0.5]),
        vertex!([0.5, 0.5 + 1e-11, 0.5]),
        vertex!([0.5, 0.5, 0.5 + 1e-11]),
    ];

    vertices.extend(near_degenerate_vertices.iter().copied());

    let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::empty();
    // The algorithm should complete successfully, but may skip near-duplicate
    // vertices that violate the near-duplicate threshold.
    let result = algorithm.triangulate(&mut tds, &vertices);

    // Allow the algorithm to succeed with some vertices skipped (unsalvageable)
    // or to complete fully if all vertices are far enough apart.
    match result {
        Ok(()) => {
            // Triangulation succeeded with all or most vertices
            assert!(
                tds.is_valid().is_ok(),
                "3D: TDS should remain valid for near-degenerate configuration"
            );
            assert!(
                tds.validate_delaunay().is_ok(),
                "3D: Final TDS should be globally Delaunay for near-degenerate configuration"
            );
            println!(
                "Near-degenerate: {} vertices inserted, {} skipped as unsalvageable",
                tds.number_of_vertices(),
                algorithm.unsalvageable_vertices().len()
            );
        }
        Err(err) => {
            // If triangulation fails with duplicate detection, that's also acceptable
            // for this extreme near-duplicate test case
            let err_str = format!("{err}");
            if err_str.contains("duplicate") || err_str.contains("near-duplicate") {
                println!("Near-degenerate test correctly rejected near-duplicates: {err}");
            } else {
                panic!("3D: Unexpected error for near-degenerate configuration: {err}");
            }
        }
    }
}

#[test]
fn test_algorithm_reset_and_reuse() {
    let mut algorithm = RobustBowyerWatson::new();

    // First run: simple tetrahedron with an interior point.
    let vertices1 = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
        vertex!([0.5, 0.5, 0.5]),
    ];
    let mut tds1: Tds<f64, Option<()>, Option<()>, 3> = Tds::empty();

    algorithm
        .triangulate(&mut tds1, &vertices1)
        .unwrap_or_else(|err| {
            panic!("3D: First run robust triangulation failed: {err}");
        });

    assert!(
        tds1.validate_delaunay().is_ok(),
        "3D: First run TDS should be globally Delaunay"
    );

    let (processed1, _, _) = algorithm.get_statistics();
    assert!(
        processed1 > 0,
        "Should have processed vertices in first run"
    );

    // Check unsalvageable vertices (may exist even for simple configurations if duplicate detection kicks in)
    let unsalvageable_count_1 = algorithm.unsalvageable_vertices().len();
    println!(
        "First run: {} unsalvageable vertices out of {} input vertices",
        unsalvageable_count_1,
        vertices1.len()
    );

    // Reset algorithm
    algorithm.reset();

    let (processed_after_reset, created_after_reset, removed_after_reset) =
        algorithm.get_statistics();
    assert_eq!(
        processed_after_reset, 0,
        "Processed should be 0 after reset"
    );
    assert_eq!(created_after_reset, 0, "Created should be 0 after reset");
    assert_eq!(removed_after_reset, 0, "Removed should be 0 after reset");

    // Verify unsalvageable_vertices bookkeeping is also reset
    assert!(
        algorithm.unsalvageable_vertices().is_empty(),
        "Unsalvageable vertices should be cleared after reset"
    );
    println!("After reset: unsalvageable vertices cleared");

    // Second run with fresh TDS and a different configuration.
    let vertices2 = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([2.0, 0.0, 0.0]),
        vertex!([0.0, 2.0, 0.0]),
        vertex!([0.0, 0.0, 2.0]),
        vertex!([1.0, 1.0, 1.0]),
    ];
    let mut tds2: Tds<f64, Option<()>, Option<()>, 3> = Tds::empty();

    algorithm
        .triangulate(&mut tds2, &vertices2)
        .unwrap_or_else(|err| {
            panic!("3D: Second run robust triangulation failed: {err}");
        });

    assert!(
        tds2.validate_delaunay().is_ok(),
        "3D: Second run TDS should be globally Delaunay"
    );

    let (processed2, _, _) = algorithm.get_statistics();
    assert!(
        processed2 > 0,
        "Should have processed vertices in second run"
    );

    // Check unsalvageable vertices in second run
    let unsalvageable_count_2 = algorithm.unsalvageable_vertices().len();
    println!(
        "Second run: {} unsalvageable vertices out of {} input vertices",
        unsalvageable_count_2,
        vertices2.len()
    );

    // The key assertion: reset() should have cleared first run's unsalvageable vertices,
    // so the second run's count is independent and starts fresh.
    // We don't assert zero unsalvageable vertices since even simple configurations
    // may have duplicates detected during construction.
}
