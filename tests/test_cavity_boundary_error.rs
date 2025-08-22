//! Test cases for reproducing and debugging cavity boundary facet errors
//!
//! This module contains tests that reproduce geometric configurations that commonly
//! cause "No cavity boundary facets found" errors in Delaunay triangulation.

use delaunay::prelude::*;
use rand::Rng;

/// Test various geometric degeneracies that could cause cavity boundary facet errors
#[test]
fn test_geometric_degeneracy_cases() {
    println!("Testing geometric configurations that could cause triangulation errors...");

    // Test 1: Nearly coplanar points
    println!("\n1. Testing nearly coplanar points:");
    let near_coplanar_points = vec![
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0]),
        Point::new([0.0, 1.0, 0.0]),
        Point::new([0.5, 0.5, 1e-10]),  // Very slightly off the plane
        Point::new([0.2, 0.3, -1e-10]), // Very slightly off the other side
    ];

    let vertices = Vertex::from_points(near_coplanar_points);
    match Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices) {
        Ok(_) => println!("  Near-coplanar points: Success"),
        Err(e) => println!("  Near-coplanar points: Failed with {e:?}"),
    }

    // Test 2: Points that create very flat tetrahedra
    println!("\n2. Testing points that create flat tetrahedra:");
    let flat_tetrahedron = vec![
        Point::new([0.0, 0.0, 0.0]),
        Point::new([10.0, 0.0, 0.0]),
        Point::new([5.0, 10.0, 0.0]),
        Point::new([5.0, 5.0, 0.01]), // Very small height
    ];

    let vertices = Vertex::from_points(flat_tetrahedron);
    match Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices) {
        Ok(_) => println!("  Flat tetrahedron: Success"),
        Err(e) => println!("  Flat tetrahedron: Failed with {e:?}"),
    }

    // Test 3: Points clustered very close together
    println!("\n3. Testing clustered points:");
    let mut clustered_points = Vec::new();
    let center = [50.0, 50.0, 50.0];
    let mut rng = rand::rng();

    for _ in 0..10 {
        // Reduced from 20 to 10 for faster execution
        clustered_points.push(Point::new([
            (rng.random::<f64>() - 0.5).mul_add(0.001, center[0]), // Very small cluster
            (rng.random::<f64>() - 0.5).mul_add(0.001, center[1]),
            (rng.random::<f64>() - 0.5).mul_add(0.001, center[2]),
        ]));
    }

    let vertices = Vertex::from_points(clustered_points);
    match Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices) {
        Ok(_) => println!("  Clustered points: Success"),
        Err(e) => println!("  Clustered points: Failed with {e:?}"),
    }

    // Test 4: Specific problematic configuration (from old test_simplified_reproduction)
    println!("\n4. Testing known problematic configuration:");
    let problematic_points = vec![
        // Initial tetrahedron
        Point::new([0.0, 0.0, 0.0]),
        Point::new([10.0, 0.0, 0.0]),
        Point::new([5.0, 8.66, 0.0]),  // Roughly equilateral base
        Point::new([5.0, 2.89, 8.16]), // Height for regular tetrahedron
        // Points that might cause issues
        Point::new([5.0, 2.89, 4.08]), // Near circumsphere center
        Point::new([2.5, 1.44, 2.04]), // Another potentially problematic point
        Point::new([7.5, 4.33, 6.12]), // And another
    ];

    let vertices = Vertex::from_points(problematic_points);
    match Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices) {
        Ok(tds) => {
            println!(
                "  Problematic configuration: Success with {} cells",
                tds.number_of_cells()
            );
            if let Err(e) = tds.is_valid() {
                println!("  Warning: Triangulation is invalid: {e:?}");
            }
        }
        Err(e) => {
            println!("  Problematic configuration: Failed with {e:?}");
            let error_string = format!("{e:?}");
            if error_string.contains("No cavity boundary facets found") {
                println!("  *** Successfully identified cavity boundary facets error! ***");
            }
        }
    }
}

/// Quick test to reproduce cavity boundary error with limited attempts
#[test]
fn test_reproduce_cavity_boundary_error_fast() {
    println!("Quick attempt to reproduce the 'No cavity boundary facets found' error...");

    // Fast test with reduced attempts for regular CI
    let mut rng = rand::rng();
    let max_attempts = 5;
    let mut reproduced = false;

    for attempt in 1..=max_attempts {
        println!("Attempt {attempt}/{max_attempts}");

        // Generate random points similar to the benchmark but fewer points for speed
        let n_points = 25;
        let points: Vec<Point<f64, 3>> = (0..n_points)
            .map(|_| {
                Point::new([
                    rng.random::<f64>() * 100.0,
                    rng.random::<f64>() * 100.0,
                    rng.random::<f64>() * 100.0,
                ])
            })
            .collect();

        let vertices = Vertex::from_points(points);

        match Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices) {
            Ok(_) => {
                println!("  Attempt {attempt}: Triangulation successful");
            }
            Err(e) => {
                let error_string = format!("{e:?}");
                if error_string.contains("No cavity boundary facets found") {
                    println!("  *** REPRODUCED THE TARGET ERROR on attempt {attempt}! ***");
                    reproduced = true;
                    break;
                }
                println!("  Attempt {attempt}: Different error: {e}");
            }
        }
    }

    if !reproduced {
        println!("Could not reproduce the specific error in {max_attempts} attempts");
        println!("This suggests the error requires specific geometric configurations");
    }
}

/// Comprehensive test to reproduce cavity boundary error - use `EXPENSIVE_TESTS=1` to enable
#[test]
fn test_reproduce_cavity_boundary_error_comprehensive() {
    // Skip this expensive test unless explicitly requested
    if std::env::var("EXPENSIVE_TESTS").unwrap_or_default() != "1" {
        println!(
            "Skipping comprehensive cavity boundary error test (set EXPENSIVE_TESTS=1 to run)"
        );
        return;
    }

    println!("Comprehensive attempt to reproduce the 'No cavity boundary facets found' error...");

    // Original comprehensive test with more attempts and points
    let mut rng = rand::rng();
    let mut attempts = 0;
    let max_attempts = 50; // Original comprehensive count
    let mut reproduced = false;

    while attempts < max_attempts {
        attempts += 1;
        println!("Attempt {attempts}/{max_attempts}");

        // Generate random points similar to the failing benchmark
        let n_points = 60; // Original benchmark size where it failed
        let points: Vec<Point<f64, 3>> = (0..n_points)
            .map(|_| {
                Point::new([
                    rng.random::<f64>() * 100.0,
                    rng.random::<f64>() * 100.0,
                    rng.random::<f64>() * 100.0,
                ])
            })
            .collect();

        let vertices = Vertex::from_points(points);

        match Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices) {
            Ok(tds) => {
                println!(
                    "  Success: Created triangulation with {} vertices, {} cells",
                    tds.number_of_vertices(),
                    tds.number_of_cells()
                );

                // If we succeed, let's validate it
                if let Err(e) = tds.is_valid() {
                    println!("  Warning: Triangulation is invalid: {e:?}");
                }
            }
            Err(e) => {
                println!("  Failed with error: {e:?}");

                // Check if this is the specific error we're looking for
                let error_string = format!("{e:?}");
                if error_string.contains("No cavity boundary facets found") {
                    println!("  *** REPRODUCED THE TARGET ERROR! ***");

                    // Extract details about the error
                    if let Some(start) = error_string.find("No cavity boundary facets found for ")
                        && let Some(end) = error_string[start..].find(" bad cells")
                    {
                        let bad_cell_count = &error_string[start + 36..start + end];
                        println!("  Error occurred with {bad_cell_count} bad cells");
                    }

                    println!("  This confirms the issue occurs with randomly generated points");
                    reproduced = true;
                    break;
                }
            }
        }
    }

    if reproduced {
        println!("Successfully reproduced the cavity boundary error after {attempts} attempts!");
    } else {
        println!("Could not reproduce the error in {max_attempts} attempts");
        println!(
            "This suggests the error is rare and depends on specific geometric configurations"
        );
    }
}
