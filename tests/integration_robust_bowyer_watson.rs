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
use rand::Rng;

// =============================================================================
// DIMENSIONAL TEST GENERATION MACRO
// =============================================================================

/// Macro to generate integration tests for a given dimension
macro_rules! test_robust_integration {
    ($dim:literal, $num_random_points:literal) => {
        pastey::paste! {
            /// Test: Large random point set insertion
            #[test]
            fn [<test_large_random_point_set_ $dim d>]() {
                let mut rng = rand::rng();
                let mut algorithm = RobustBowyerWatson::new();

                // Create initial simplex
                let initial_vertices = create_initial_simplex::<$dim>();
                let mut tds: Tds<f64, Option<()>, Option<()>, $dim> =
                    Tds::new(&initial_vertices).unwrap();

                assert!(tds.is_valid().is_ok(), "{}D: Initial TDS should be valid", $dim);

                // Insert random points
                for i in 0..$num_random_points {
                    let coords: [f64; $dim] = std::array::from_fn(|_| rng.random_range(-5.0..15.0));
                    let test_vertex = vertex!(coords);

                    let result = algorithm.insert_vertex(&mut tds, test_vertex);

                    // TDS should remain valid regardless of insertion outcome
                    assert!(
                        tds.is_valid().is_ok(),
                        "{}D: TDS should remain valid after insertion {}",
                        $dim,
                        i + 1
                    );

                    if result.is_ok() {
                        assert!(
                            tds.number_of_vertices() > initial_vertices.len() + i,
                            "{}D: Vertex count should increase with successful insertions",
                            $dim
                        );
                    }
                }

                // Verify final state
                assert!(tds.is_valid().is_ok(), "{}D: Final TDS should be valid", $dim);
                let (processed, created, removed) = algorithm.get_statistics();
                assert!(processed > 0, "{}D: Should have processed vertices", $dim);
                println!(
                    "{}D large dataset: processed={}, created={}, removed={}",
                    $dim, processed, created, removed
                );
            }

            /// Test: Exterior vertex insertion sequence
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

                println!(
                    "{}D: Final TDS has {} vertices, {} cells",
                    $dim,
                    tds.number_of_vertices(),
                    tds.number_of_cells()
                );
            }

            /// Test: Clustered points followed by scattered points
            #[test]
            fn [<test_clustered_points_ $dim d>]() {
                let mut algorithm = RobustBowyerWatson::new();
                let mut rng = rand::rng();

                let initial_vertices = create_initial_simplex::<$dim>();
                let mut tds: Tds<f64, Option<()>, Option<()>, $dim> =
                    Tds::new(&initial_vertices).unwrap();

                // Insert clustered points (all near origin)
                for i in 0..30 {
                    let coords: [f64; $dim] = std::array::from_fn(|_| rng.random_range(-0.5..0.5));
                    let test_vertex = vertex!(coords);

                    let _ = algorithm.insert_vertex(&mut tds, test_vertex);

                    assert!(
                        tds.is_valid().is_ok(),
                        "{}D: TDS should remain valid with clustered points after insertion {}",
                        $dim,
                        i + 1
                    );
                }

                // Now insert scattered points
                for i in 0..30 {
                    let coords: [f64; $dim] = std::array::from_fn(|_| rng.random_range(-20.0..20.0));
                    let test_vertex = vertex!(coords);

                    let _ = algorithm.insert_vertex(&mut tds, test_vertex);

                    assert!(
                        tds.is_valid().is_ok(),
                        "{}D: TDS should remain valid with scattered points after insertion {}",
                        $dim,
                        i + 1
                    );
                }

                println!(
                    "{}D clustered+scattered: {} vertices, {} cells",
                    $dim,
                    tds.number_of_vertices(),
                    tds.number_of_cells()
                );
            }
        }
    };
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Create initial simplex for dimension D
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
test_robust_integration!(2, 100);
test_robust_integration!(3, 100);
test_robust_integration!(4, 50);
test_robust_integration!(5, 30); // Reduced for coverage runs

// =============================================================================
// ADDITIONAL INTEGRATION TESTS (3D-specific)
// =============================================================================

#[test]
fn test_mixed_interior_exterior_insertions_3d() {
    let mut algorithm = RobustBowyerWatson::new();

    // Start with tetrahedron
    let initial_vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([2.0, 0.0, 0.0]),
        vertex!([0.0, 2.0, 0.0]),
        vertex!([0.0, 0.0, 2.0]),
    ];
    let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();

    // Interleave interior and exterior points
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

    for (i, (vertex, label)) in mixed_vertices.iter().enumerate() {
        let result = algorithm.insert_vertex(&mut tds, *vertex);

        assert!(
            tds.is_valid().is_ok(),
            "TDS should remain valid after {} insertion {}",
            label,
            i + 1
        );

        if result.is_err() {
            println!("Warning: {} insertion {} failed (acceptable)", label, i + 1);
        }
    }

    // Verify statistics
    let (processed, created, removed) = algorithm.get_statistics();
    assert!(processed > 0, "Should have processed vertices");
    println!("Mixed insertions: processed={processed}, created={created}, removed={removed}");
}

#[test]
fn test_grid_pattern_insertion_2d() {
    let mut algorithm = RobustBowyerWatson::new();

    // Create initial triangulation
    let initial_vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([10.0, 0.0]),
        vertex!([0.0, 10.0]),
    ];
    let mut tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&initial_vertices).unwrap();

    // Insert grid points
    for i in 1..10 {
        for j in 1..10 {
            let x = f64::from(i);
            let y = f64::from(j);
            let test_vertex = vertex!([x, y]);

            let _ = algorithm.insert_vertex(&mut tds, test_vertex);

            assert!(
                tds.is_valid().is_ok(),
                "TDS should remain valid with grid pattern at ({x}, {y})"
            );
        }
    }

    println!(
        "Grid pattern: {} vertices, {} cells",
        tds.number_of_vertices(),
        tds.number_of_cells()
    );
}

#[test]
fn test_degenerate_robust_configuration_3d() {
    let mut algorithm = RobustBowyerWatson::for_degenerate_cases();

    // Create initial tetrahedron
    let initial_vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();

    // Insert points with small perturbations
    let near_degenerate_vertices = vec![
        vertex!([1e-10, 1e-10, 1e-10]),
        vertex!([0.5 + 1e-11, 0.5, 0.5]),
        vertex!([0.5, 0.5 + 1e-11, 0.5]),
        vertex!([0.5, 0.5, 0.5 + 1e-11]),
    ];

    for (i, vertex) in near_degenerate_vertices.iter().enumerate() {
        let _ = algorithm.insert_vertex(&mut tds, *vertex);

        // Should not panic and should maintain validity
        assert!(
            tds.is_valid().is_ok(),
            "TDS should remain valid with degenerate config after insertion {}",
            i + 1
        );
    }
}

#[test]
fn test_algorithm_reset_and_reuse() {
    let mut algorithm = RobustBowyerWatson::new();

    // First run
    let vertices1 = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let mut tds1: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices1).unwrap();

    let _ = algorithm.insert_vertex(&mut tds1, vertex!([0.5, 0.5, 0.5]));

    let (processed1, _, _) = algorithm.get_statistics();
    assert!(
        processed1 > 0,
        "Should have processed vertices in first run"
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

    // Second run with fresh TDS
    let vertices2 = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([2.0, 0.0, 0.0]),
        vertex!([0.0, 2.0, 0.0]),
        vertex!([0.0, 0.0, 2.0]),
    ];
    let mut tds2: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices2).unwrap();

    let _ = algorithm.insert_vertex(&mut tds2, vertex!([1.0, 1.0, 1.0]));

    let (processed2, _, _) = algorithm.get_statistics();
    assert!(
        processed2 > 0,
        "Should have processed vertices in second run"
    );
}
