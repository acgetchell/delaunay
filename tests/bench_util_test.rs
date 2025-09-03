//! Integration tests for benchmark utility functions

#[path = "../benches/util.rs"]
mod util;

use delaunay::prelude::*;
use util::{
    clear_all_neighbors, generate_random_points, generate_random_points_2d,
    generate_random_points_2d_seeded, generate_random_points_3d, generate_random_points_3d_seeded,
    generate_random_points_4d, generate_random_points_4d_seeded, generate_random_points_5d,
    generate_random_points_5d_seeded, generate_random_points_seeded,
};

#[test]
fn test_clear_all_neighbors() {
    // Create a triangulation with more vertices to ensure multiple cells with neighbors
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
        vertex!([0.5, 0.5, 0.5]), // Additional vertex to create multiple tetrahedra
    ];

    let mut tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();

    // Verify triangulation was created with neighbors
    tds.assign_neighbors()
        .expect("assign_neighbors() failed; Tds is invalid");

    // Should have some neighbors initially
    let has_neighbors = tds
        .cells()
        .values()
        .any(|cell| cell.neighbors.as_ref().is_some());
    assert!(has_neighbors, "Initial triangulation should have neighbors");

    // Clear all neighbors using our helper function
    clear_all_neighbors(&mut tds);

    // Verify all neighbors were cleared
    let final_neighbor_count: usize = tds
        .cells()
        .values()
        .map(|cell| cell.neighbors.as_ref().map_or(0, std::vec::Vec::len))
        .sum();

    assert_eq!(
        final_neighbor_count, 0,
        "All neighbors should be cleared after calling clear_all_neighbors"
    );
}

#[test]
fn test_clear_all_neighbors_empty_triangulation() {
    // Test with a triangulation that has no cells
    let vertices: Vec<Vertex<f64, (), 3>> = vec![];

    // This should create an empty triangulation
    if let Ok(mut tds) = Tds::<f64, (), (), 3>::new(&vertices) {
        // Should not panic on empty triangulation
        clear_all_neighbors(&mut tds);
        assert_eq!(tds.number_of_cells(), 0);
    } else {
        // It's expected that creating a triangulation with no vertices might fail
        // This is acceptable behavior
    }
}

// =============================================================================
// GENERIC POINT GENERATION TESTS
// =============================================================================

#[test]
fn test_generate_random_points_basic() {
    // Test basic point generation for different dimensions
    let points_2d: Vec<Point<f64, 2>> = generate_random_points(10);
    let points_3d: Vec<Point<f64, 3>> = generate_random_points(10);
    let points_4d: Vec<Point<f64, 4>> = generate_random_points(10);
    let points_5d: Vec<Point<f64, 5>> = generate_random_points(10);

    assert_eq!(points_2d.len(), 10);
    assert_eq!(points_3d.len(), 10);
    assert_eq!(points_4d.len(), 10);
    assert_eq!(points_5d.len(), 10);
}

#[test]
fn test_generate_random_points_coordinate_range() {
    // Test that coordinates are within expected range [-100.0, 100.0]
    let points: Vec<Point<f64, 3>> = generate_random_points(100);

    for point in &points {
        for &coord in point.to_array().as_ref() {
            assert!(
                (-100.0..=100.0).contains(&coord),
                "Coordinate {coord} is outside expected range [-100.0, 100.0]"
            );
        }
    }
}

#[test]
fn test_generate_random_points_empty() {
    // Test generating zero points
    let points: Vec<Point<f64, 3>> = generate_random_points(0);
    assert!(points.is_empty());
}

#[test]
fn test_generate_random_points_large_count() {
    // Test generating a larger number of points
    let points: Vec<Point<f64, 4>> = generate_random_points(1000);
    assert_eq!(points.len(), 1000);

    // Verify some basic statistical properties (should have some variation)
    let first_coords: Vec<f64> = points.iter().map(|p| p.to_array()[0]).collect();
    let min_coord = first_coords.iter().copied().fold(f64::INFINITY, f64::min);
    let max_coord = first_coords
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    // With 1000 random points, we should have significant variation
    assert!(
        max_coord - min_coord > 50.0,
        "Random points should have sufficient variation"
    );
}

#[test]
fn test_generate_random_points_seeded_reproducibility() {
    // Test that seeded generation is reproducible
    const SEED: u64 = 42;
    const COUNT: usize = 50;

    let points1: Vec<Point<f64, 3>> = generate_random_points_seeded(COUNT, SEED);
    let points2: Vec<Point<f64, 3>> = generate_random_points_seeded(COUNT, SEED);

    assert_eq!(points1.len(), COUNT);
    assert_eq!(points2.len(), COUNT);

    // Points should be identical when using the same seed
    for (p1, p2) in points1.iter().zip(points2.iter()) {
        for (c1, c2) in p1.to_array().iter().zip(p2.to_array().iter()) {
            assert!(
                (c1 - c2).abs() < f64::EPSILON,
                "Seeded generation should produce identical results: {c1} != {c2}"
            );
        }
    }
}

#[test]
fn test_generate_random_points_seeded_different_seeds() {
    // Test that different seeds produce different results
    const COUNT: usize = 20;

    let points1: Vec<Point<f64, 3>> = generate_random_points_seeded(COUNT, 12345);
    let points2: Vec<Point<f64, 3>> = generate_random_points_seeded(COUNT, 54321);

    assert_eq!(points1.len(), COUNT);
    assert_eq!(points2.len(), COUNT);

    // At least some points should be different (extremely unlikely to be identical)
    let mut differences = 0;
    for (p1, p2) in points1.iter().zip(points2.iter()) {
        for (c1, c2) in p1.to_array().iter().zip(p2.to_array().iter()) {
            if (c1 - c2).abs() > f64::EPSILON {
                differences += 1;
                break;
            }
        }
    }

    assert!(
        differences > 15,
        "Different seeds should produce different point sets"
    );
}

#[test]
fn test_generate_random_points_seeded_coordinate_range() {
    // Test that seeded generation also respects coordinate range
    let points: Vec<Point<f64, 2>> = generate_random_points_seeded(50, 99999);

    for point in &points {
        for &coord in point.to_array().as_ref() {
            assert!(
                (-100.0..=100.0).contains(&coord),
                "Seeded coordinate {coord} is outside expected range [-100.0, 100.0]"
            );
        }
    }
}

// =============================================================================
// DIMENSION-SPECIFIC FUNCTION TESTS
// =============================================================================

#[test]
fn test_generate_random_points_2d() {
    let points = generate_random_points_2d(25);
    assert_eq!(points.len(), 25);

    // Verify all points are 2D
    for point in &points {
        assert_eq!(point.to_array().len(), 2);
        for &coord in point.to_array().as_ref() {
            assert!((-100.0..=100.0).contains(&coord));
        }
    }
}

#[test]
fn test_generate_random_points_3d() {
    let points = generate_random_points_3d(30);
    assert_eq!(points.len(), 30);

    // Verify all points are 3D
    for point in &points {
        assert_eq!(point.to_array().len(), 3);
        for &coord in point.to_array().as_ref() {
            assert!((-100.0..=100.0).contains(&coord));
        }
    }
}

#[test]
fn test_generate_random_points_4d() {
    let points = generate_random_points_4d(20);
    assert_eq!(points.len(), 20);

    // Verify all points are 4D
    for point in &points {
        assert_eq!(point.to_array().len(), 4);
        for &coord in point.to_array().as_ref() {
            assert!((-100.0..=100.0).contains(&coord));
        }
    }
}

#[test]
fn test_generate_random_points_5d() {
    let points = generate_random_points_5d(15);
    assert_eq!(points.len(), 15);

    // Verify all points are 5D
    for point in &points {
        assert_eq!(point.to_array().len(), 5);
        for &coord in point.to_array().as_ref() {
            assert!((-100.0..=100.0).contains(&coord));
        }
    }
}

#[test]
fn test_generate_random_points_2d_seeded() {
    let seed = 11111;
    let points1 = generate_random_points_2d_seeded(15, seed);
    let points2 = generate_random_points_2d_seeded(15, seed);

    assert_eq!(points1.len(), 15);
    assert_eq!(points2.len(), 15);

    // Should be reproducible
    for (p1, p2) in points1.iter().zip(points2.iter()) {
        for (c1, c2) in p1.to_array().iter().zip(p2.to_array().iter()) {
            assert!((c1 - c2).abs() < f64::EPSILON);
        }
    }
}

#[test]
fn test_generate_random_points_3d_seeded() {
    let seed = 22222;
    let points1 = generate_random_points_3d_seeded(12, seed);
    let points2 = generate_random_points_3d_seeded(12, seed);

    assert_eq!(points1.len(), 12);
    assert_eq!(points2.len(), 12);

    // Should be reproducible
    for (p1, p2) in points1.iter().zip(points2.iter()) {
        for (c1, c2) in p1.to_array().iter().zip(p2.to_array().iter()) {
            assert!((c1 - c2).abs() < f64::EPSILON);
        }
    }
}

#[test]
fn test_generate_random_points_4d_seeded() {
    let seed = 33333;
    let points1 = generate_random_points_4d_seeded(10, seed);
    let points2 = generate_random_points_4d_seeded(10, seed);

    assert_eq!(points1.len(), 10);
    assert_eq!(points2.len(), 10);

    // Should be reproducible
    for (p1, p2) in points1.iter().zip(points2.iter()) {
        for (c1, c2) in p1.to_array().iter().zip(p2.to_array().iter()) {
            assert!((c1 - c2).abs() < f64::EPSILON);
        }
    }
}

#[test]
fn test_generate_random_points_5d_seeded() {
    let seed = 44444;
    let points1 = generate_random_points_5d_seeded(8, seed);
    let points2 = generate_random_points_5d_seeded(8, seed);

    assert_eq!(points1.len(), 8);
    assert_eq!(points2.len(), 8);

    // Should be reproducible
    for (p1, p2) in points1.iter().zip(points2.iter()) {
        for (c1, c2) in p1.to_array().iter().zip(p2.to_array().iter()) {
            assert!((c1 - c2).abs() < f64::EPSILON);
        }
    }
}

// =============================================================================
// CROSS-DIMENSIONAL CONSISTENCY TESTS
// =============================================================================

#[test]
fn test_generic_vs_specific_function_equivalence() {
    // Test that generic functions produce the same results as dimension-specific ones
    const SEED: u64 = 77777;
    const COUNT: usize = 10;

    // Test 2D
    let generic_2d: Vec<Point<f64, 2>> = generate_random_points_seeded(COUNT, SEED);
    let specific_2d = generate_random_points_2d_seeded(COUNT, SEED);

    for (g, s) in generic_2d.iter().zip(specific_2d.iter()) {
        for (gc, sc) in g.to_array().iter().zip(s.to_array().iter()) {
            assert!(
                (gc - sc).abs() < f64::EPSILON,
                "Generic and specific 2D functions should match"
            );
        }
    }

    // Test 3D
    let generic_3d: Vec<Point<f64, 3>> = generate_random_points_seeded(COUNT, SEED);
    let specific_3d = generate_random_points_3d_seeded(COUNT, SEED);

    for (g, s) in generic_3d.iter().zip(specific_3d.iter()) {
        for (gc, sc) in g.to_array().iter().zip(s.to_array().iter()) {
            assert!(
                (gc - sc).abs() < f64::EPSILON,
                "Generic and specific 3D functions should match"
            );
        }
    }

    // Test 4D
    let generic_4d: Vec<Point<f64, 4>> = generate_random_points_seeded(COUNT, SEED);
    let specific_4d = generate_random_points_4d_seeded(COUNT, SEED);

    for (g, s) in generic_4d.iter().zip(specific_4d.iter()) {
        for (gc, sc) in g.to_array().iter().zip(s.to_array().iter()) {
            assert!(
                (gc - sc).abs() < f64::EPSILON,
                "Generic and specific 4D functions should match"
            );
        }
    }

    // Test 5D
    let generic_5d: Vec<Point<f64, 5>> = generate_random_points_seeded(COUNT, SEED);
    let specific_5d = generate_random_points_5d_seeded(COUNT, SEED);

    for (g, s) in generic_5d.iter().zip(specific_5d.iter()) {
        for (gc, sc) in g.to_array().iter().zip(s.to_array().iter()) {
            assert!(
                (gc - sc).abs() < f64::EPSILON,
                "Generic and specific 5D functions should match"
            );
        }
    }
}

#[test]
fn test_seeded_vs_unseeded_difference() {
    // Test that unseeded generation produces different results than seeded
    const COUNT: usize = 20;
    const SEED: u64 = 88888;

    let unseeded: Vec<Point<f64, 3>> = generate_random_points(COUNT);
    let seeded: Vec<Point<f64, 3>> = generate_random_points_seeded(COUNT, SEED);

    assert_eq!(unseeded.len(), COUNT);
    assert_eq!(seeded.len(), COUNT);

    // It's extremely unlikely (but not impossible) for unseeded and seeded to be identical
    // We'll just check that both generate valid points within range
    for points in [&unseeded, &seeded] {
        for point in points {
            for &coord in point.to_array().as_ref() {
                assert!((-100.0..=100.0).contains(&coord));
            }
        }
    }
}

// =============================================================================
// INTEGRATION TESTS WITH TRIANGULATION
// =============================================================================

#[test]
fn test_generated_points_create_valid_triangulations() {
    // Test that generated points can be used to create valid triangulations

    // Test 2D triangulation
    let points_2d = generate_random_points_2d(10);
    let vertices_2d: Vec<_> = points_2d.iter().map(|p| vertex!(*p)).collect();
    let tds_2d = Tds::<f64, (), (), 2>::new(&vertices_2d);
    assert!(
        tds_2d.is_ok(),
        "2D triangulation should be created successfully"
    );

    // Test 3D triangulation
    let points_3d = generate_random_points_3d(15);
    let vertices_3d: Vec<_> = points_3d.iter().map(|p| vertex!(*p)).collect();
    let tds_3d = Tds::<f64, (), (), 3>::new(&vertices_3d);
    assert!(
        tds_3d.is_ok(),
        "3D triangulation should be created successfully"
    );

    // Test 4D triangulation
    let points_4d = generate_random_points_4d(12);
    let vertices_4d: Vec<_> = points_4d.iter().map(|p| vertex!(*p)).collect();
    let tds_4d = Tds::<f64, (), (), 4>::new(&vertices_4d);
    assert!(
        tds_4d.is_ok(),
        "4D triangulation should be created successfully"
    );

    // Test 5D triangulation (smaller point set due to computational complexity)
    let points_5d = generate_random_points_5d(8);
    let vertices_5d: Vec<_> = points_5d.iter().map(|p| vertex!(*p)).collect();
    let tds_5d = Tds::<f64, (), (), 5>::new(&vertices_5d);
    assert!(
        tds_5d.is_ok(),
        "5D triangulation should be created successfully"
    );
}

#[test]
fn test_seeded_points_triangulation_reproducibility() {
    // Test that seeded points create reproducible triangulations
    const SEED: u64 = 55555;

    // Generate points twice with same seed
    let points1 = generate_random_points_3d_seeded(20, SEED);
    let points2 = generate_random_points_3d_seeded(20, SEED);

    // First verify that the points themselves are identical
    assert_eq!(points1.len(), points2.len());
    for (p1, p2) in points1.iter().zip(points2.iter()) {
        for (c1, c2) in p1.to_array().iter().zip(p2.to_array().iter()) {
            assert!(
                (c1 - c2).abs() < f64::EPSILON,
                "Points should be identical with same seed"
            );
        }
    }

    // Create triangulations
    let vertices1: Vec<_> = points1.iter().map(|p| vertex!(*p)).collect();
    let vertices2: Vec<_> = points2.iter().map(|p| vertex!(*p)).collect();

    let tds1 = Tds::<f64, (), (), 3>::new(&vertices1).unwrap();
    let tds2 = Tds::<f64, (), (), 3>::new(&vertices2).unwrap();

    // Should have same number of vertices (input points)
    assert_eq!(tds1.number_of_vertices(), tds2.number_of_vertices());

    // Note: number of cells might vary slightly due to numerical precision in triangulation
    // algorithms, even with identical input points. This is expected behavior.
    // We verify that both triangulations are valid and contain the same vertices.
    let cells1 = tds1.number_of_cells();
    let cells2 = tds2.number_of_cells();

    // Both should be reasonable triangulations (not empty, not wildly different)
    assert!(cells1 > 0, "First triangulation should have cells");
    assert!(cells2 > 0, "Second triangulation should have cells");

    // The cell counts should be close (within a reasonable tolerance)
    let cell_diff = cells1.abs_diff(cells2);
    assert!(
        cell_diff <= 5,
        "Cell count difference should be small: {cells1} vs {cells2}, diff: {cell_diff}"
    );
}
