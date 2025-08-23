//! # 3D Triangulation Example with 50 Points
//!
//! This example demonstrates creating a 3D Delaunay triangulation using 50 randomly
//! generated points. It showcases:
//!
//! - Creating random 3D vertices
//! - Building a Delaunay triangulation using the Bowyer-Watson algorithm
//! - Analyzing triangulation properties (vertices, cells, dimension)
//! - Validating the triangulation's geometric properties
//! - Computing and displaying boundary information
//! - Performance timing for triangulation construction
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example triangulation_3d_50_points
//! ```
//!
//! ## Output
//!
//! The example produces detailed output showing:
//! - Generated vertex coordinates
//! - Triangulation statistics
//! - Validation results
//! - Boundary analysis
//! - Performance metrics

use delaunay::prelude::*;
use num_traits::cast::cast;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn main() {
    println!("=================================================================");
    println!("3D Delaunay Triangulation Example - 50 Random Points");
    println!("=================================================================\n");

    // Generate 50 random 3D points with a fixed seed for reproducibility
    let vertices = generate_random_vertices_3d(50, 42);

    println!("Generated {} vertices:", vertices.len());
    display_vertices(&vertices[..10]); // Show first 10 vertices
    if vertices.len() > 10 {
        println!("  ... and {} more vertices", vertices.len() - 10);
    }
    println!();

    // Create Delaunay triangulation with timing
    println!("Creating Delaunay triangulation...");
    let start = Instant::now();

    let tds: Tds<f64, Option<()>, Option<()>, 3> = match Tds::new(&vertices) {
        Ok(triangulation) => {
            let construction_time = start.elapsed();
            println!("✓ Triangulation created successfully in {construction_time:?}");
            triangulation
        }
        Err(e) => {
            println!("✗ Failed to create triangulation: {e}");
            return;
        }
    };

    println!();

    // Display triangulation properties
    analyze_triangulation(&tds);

    // Validate the triangulation
    validate_triangulation(&tds);

    // Analyze boundary properties
    analyze_boundary_properties(&tds);

    // Performance analysis
    performance_analysis(&tds);

    println!("\n=================================================================");
    println!("Example completed successfully!");
    println!("=================================================================");
}

/// Generate random 3D vertices with a specified seed for reproducibility
fn generate_random_vertices_3d(count: usize, seed: u64) -> Vec<Vertex<f64, Option<()>, 3>> {
    let mut rng = StdRng::seed_from_u64(seed);

    (0..count)
        .map(|_| {
            vertex!([
                rng.random_range(-10.0..10.0),
                rng.random_range(-10.0..10.0),
                rng.random_range(-10.0..10.0)
            ])
        })
        .collect()
}

/// Display a subset of vertices with their coordinates
fn display_vertices(vertices: &[Vertex<f64, Option<()>, 3>]) {
    for (i, vertex) in vertices.iter().enumerate() {
        let coords: [f64; 3] = vertex.into();
        println!(
            "  v{:2}: [{:8.3}, {:8.3}, {:8.3}]",
            i, coords[0], coords[1], coords[2]
        );
    }
}

/// Analyze and display triangulation properties
fn analyze_triangulation(tds: &Tds<f64, Option<()>, Option<()>, 3>) {
    println!("Triangulation Analysis:");
    println!("======================");
    println!("  Number of vertices: {}", tds.number_of_vertices());
    println!("  Number of cells:    {}", tds.number_of_cells());
    println!("  Dimension:          {}", tds.dim());

    // Calculate vertex-to-cell ratio
    let vertex_count = tds.number_of_vertices();
    let cell_count = tds.number_of_cells();
    if cell_count > 0 {
        let vertex_f64 = cast(vertex_count).unwrap_or(0.0f64);
        let cell_f64 = cast(cell_count).unwrap_or(1.0f64);
        let ratio = vertex_f64 / cell_f64;
        println!("  Vertex/Cell ratio:  {ratio:.2}");
    }

    // Analyze cell properties
    if cell_count > 0 {
        println!("\n  Cell Analysis:");
        let mut valid_cells = 0;
        let mut total_neighbors = 0;

        for (cell_key, cell) in tds.cells() {
            // Count valid cells
            if cell.is_valid().is_ok() {
                valid_cells += 1;
            }

            // Count neighbors
            if let Some(neighbors) = &cell.neighbors {
                total_neighbors += neighbors.len();
            }

            // Show details for first few cells
            if valid_cells <= 3 {
                println!("    Cell {cell_key:?}:");
                println!("      Vertices: {}", cell.vertices().len());
                if let Some(neighbors) = &cell.neighbors {
                    println!("      Neighbors: {}", neighbors.len());
                }
            }
        }

        println!("    Valid cells:     {valid_cells}/{cell_count}");
        if cell_count > 0 {
            let total_f64 = cast(total_neighbors).unwrap_or(0.0f64);
            let cell_f64 = cast(cell_count).unwrap_or(1.0f64);
            let avg_neighbors = total_f64 / cell_f64;
            println!("    Avg neighbors:   {avg_neighbors:.2}");
        }
    }
    println!();
}

/// Validate the triangulation and report results
fn validate_triangulation(tds: &Tds<f64, Option<()>, Option<()>, 3>) {
    println!("Triangulation Validation:");
    println!("========================");

    let start = Instant::now();
    match tds.is_valid() {
        Ok(()) => {
            let validation_time = start.elapsed();
            println!("✓ Triangulation is VALID");
            println!("  Validation completed in {validation_time:?}");

            // Additional validation details
            println!("\n  Validation Details:");
            println!("    • All cells have valid geometry");
            println!("    • Neighbor relationships are consistent");
            println!("    • No duplicate cells detected");
            println!("    • Vertex mappings are consistent");
            println!("    • Facet sharing is valid");
        }
        Err(e) => {
            let validation_time = start.elapsed();
            println!("✗ Triangulation is INVALID");
            println!("  Validation failed in {validation_time:?}");
            println!("  Error: {e}");

            // Provide debugging information
            match e {
                TriangulationValidationError::InvalidCell { cell_id, source } => {
                    println!("  Problem cell ID: {cell_id}");
                    println!("  Cell error: {source}");
                }
                TriangulationValidationError::InvalidNeighbors { message } => {
                    println!("  Neighbor problem: {message}");
                }
                TriangulationValidationError::DuplicateCells { message } => {
                    println!("  Duplicate cells: {message}");
                }
                _ => {
                    println!("  See error message above for details");
                }
            }
        }
    }
    println!();
}

/// Analyze boundary properties of the triangulation
fn analyze_boundary_properties(tds: &Tds<f64, Option<()>, Option<()>, 3>) {
    println!("Boundary Analysis:");
    println!("=================");

    let start = Instant::now();

    // Count boundary facets - use the trait method
    let boundary_facets = tds.boundary_facets().unwrap_or_else(|e| {
        println!("Warning: Failed to get boundary facets: {e}");
        Vec::new()
    });
    let boundary_count = boundary_facets.len();

    let boundary_time = start.elapsed();

    println!("  Boundary facets:     {boundary_count}");
    println!("  Boundary computation: {boundary_time:?}");

    if boundary_count > 0 {
        println!("\n  Boundary Details:");
        println!("    • Boundary facets form the convex hull");
        println!("    • Each boundary facet belongs to exactly one cell");

        // Analyze a few boundary facets
        let sample_size = std::cmp::min(3, boundary_count);
        for (i, facet) in boundary_facets.iter().take(sample_size).enumerate() {
            println!("    • Facet {}: key = {}", i + 1, facet.key());
        }

        if boundary_count > sample_size {
            println!(
                "    • ... and {} more boundary facets",
                boundary_count - sample_size
            );
        }
    }

    // Euler characteristic check (for 3D: V - E + F - C = 1 for convex hull)
    let vertices = tds.number_of_vertices();
    let cells = tds.number_of_cells();
    println!("\n  Topological Properties:");
    println!("    • Vertices (V): {vertices}");
    println!("    • Cells (C):    {cells}");
    println!("    • Boundary facets: {boundary_count}");

    println!();
}

/// Perform performance analysis and benchmarking
fn performance_analysis(tds: &Tds<f64, Option<()>, Option<()>, 3>) {
    println!("Performance Analysis:");
    println!("====================");

    let vertex_count = tds.number_of_vertices();
    let cell_count = tds.number_of_cells();

    // Benchmark validation performance
    let validation_times: Vec<_> = (0..5)
        .map(|_| {
            let start = Instant::now();
            let _ = tds.is_valid();
            start.elapsed()
        })
        .collect();

    let len_u32 = u32::try_from(validation_times.len()).unwrap_or(1u32);
    let avg_validation_time: std::time::Duration =
        validation_times.iter().sum::<std::time::Duration>() / len_u32;
    let min_validation_time = *validation_times.iter().min().unwrap();
    let max_validation_time = *validation_times.iter().max().unwrap();

    println!("  Validation Performance (5 runs):");
    println!("    • Average time: {avg_validation_time:?}");
    println!("    • Min time:     {min_validation_time:?}");
    println!("    • Max time:     {max_validation_time:?}");

    // Benchmark boundary computation
    let boundary_times: Vec<_> = (0..3)
        .map(|_| {
            let start = Instant::now();
            let _ = tds.boundary_facets().unwrap_or_default();
            start.elapsed()
        })
        .collect();

    let len_u32 = u32::try_from(boundary_times.len()).unwrap_or(1u32);
    let avg_boundary_time: std::time::Duration =
        boundary_times.iter().sum::<std::time::Duration>() / len_u32;

    println!("\n  Boundary Computation Performance (3 runs):");
    println!("    • Average time: {avg_boundary_time:?}");

    // Memory usage estimation
    let vertex_size = std::mem::size_of::<Vertex<f64, Option<()>, 3>>();
    let cell_size = std::mem::size_of::<Cell<f64, Option<()>, Option<()>, 3>>();
    let estimated_memory = (vertex_count * vertex_size) + (cell_count * cell_size);

    println!("\n  Memory Usage Estimation:");
    println!("    • Vertex memory: ~{} bytes", vertex_count * vertex_size);
    println!("    • Cell memory:   ~{} bytes", cell_count * cell_size);
    let estimated_f64 = cast(estimated_memory).unwrap_or(0.0f64);
    println!(
        "    • Total memory:  ~{estimated_memory} bytes ({:.1} KB)",
        estimated_f64 / 1024.0
    );

    // Performance per vertex/cell ratios
    if vertex_count > 0 && cell_count > 0 {
        let nanos_f64 = cast(avg_validation_time.as_nanos()).unwrap_or(0.0f64);
        let vertex_f64 = cast(vertex_count).unwrap_or(1.0f64);
        let cell_f64 = cast(cell_count).unwrap_or(1.0f64);
        let validation_per_vertex = nanos_f64 / vertex_f64;
        let validation_per_cell = nanos_f64 / cell_f64;

        println!("\n  Performance Ratios:");
        println!("    • Validation per vertex: {validation_per_vertex:.2} ns");
        println!("    • Validation per cell:   {validation_per_cell:.2} ns");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_generation() {
        let vertices = generate_random_vertices_3d(10, 42);
        assert_eq!(vertices.len(), 10);

        // Verify vertices are within expected range
        for vertex in &vertices {
            let coords: [f64; 3] = vertex.into();
            for &coord in &coords {
                assert!(coord >= -10.0 && coord <= 10.0);
            }
        }
    }

    #[test]
    fn test_small_triangulation() {
        let vertices = generate_random_vertices_3d(5, 123);
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        assert_eq!(tds.number_of_vertices(), 5);
        assert!(tds.number_of_cells() > 0);
        assert_eq!(tds.dim(), 3);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_triangulation_properties() {
        let vertices = generate_random_vertices_3d(20, 456);
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // Basic property checks
        assert!(tds.number_of_vertices() > 0);
        assert!(tds.number_of_cells() > 0);
        assert_eq!(tds.dim(), 3);

        // Validation should pass
        assert!(tds.is_valid().is_ok());

        // Should have boundary facets
        let boundary_facets = tds.boundary_facets().expect("Should get boundary facets");
        assert!(!boundary_facets.is_empty());
    }

    #[test]
    fn test_reproducibility() {
        // Same seed should produce same vertices
        let vertices1 = generate_random_vertices_3d(10, 789);
        let vertices2 = generate_random_vertices_3d(10, 789);

        assert_eq!(vertices1.len(), vertices2.len());
        for (v1, v2) in vertices1.iter().zip(vertices2.iter()) {
            let coords1: [f64; 3] = v1.into();
            let coords2: [f64; 3] = v2.into();
            for (c1, c2) in coords1.iter().zip(coords2.iter()) {
                assert!((c1 - c2).abs() < f64::EPSILON);
            }
        }
    }
}
