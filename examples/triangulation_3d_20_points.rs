//! # 3D Triangulation Example with 20 Points
//!
//! This example demonstrates creating a 3D Delaunay triangulation using 20 randomly
//! generated points. It showcases:
//!
//! - Using the `generate_random_triangulation` utility function for convenience
//! - Building a Delaunay triangulation using the Bowyer-Watson algorithm
//! - Analyzing triangulation properties (vertices, cells, dimension)
//! - Validating the triangulation's geometric properties
//! - Computing and displaying boundary information
//! - Performance timing for triangulation construction
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example triangulation_3d_100_points
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

use delaunay::geometry::util::generate_random_triangulation;
use delaunay::prelude::*;
use num_traits::cast::cast;
use std::time::Instant;

fn main() {
    println!("=================================================================");
    println!("3D Delaunay Triangulation Example - 20 Random Points");
    println!("=================================================================\\n");

    // Create Delaunay triangulation with timing using the utility function.
    // NOTE: The (n_points, bounds, seed) triple matches a configuration covered by
    // `test_generate_random_triangulation_dimensions` in `geometry::util` to avoid
    // pathological Delaunay-repair failures in CI while still exercising a nontrivial 3D case.
    println!(
        "Creating 3D Delaunay triangulation with 20 random points in [-3, 3]^3 (seed = 666)..."
    );
    let start = Instant::now();

    let dt = match generate_random_triangulation(
        20,          // Number of points
        (-3.0, 3.0), // Coordinate bounds
        None,        // No vertex data
        Some(666),   // Fixed seed for reproducibility (matches tested configuration)
    ) {
        Ok(triangulation) => {
            let construction_time = start.elapsed();
            println!("✓ Triangulation created successfully in {construction_time:?}");
            triangulation
        }
        Err(e) => {
            eprintln!("✗ Failed to create triangulation: {e}");
            return;
        }
    };

    // Display some vertex information by accessing the triangulation's vertices
    let vertex_count = dt.tds().number_of_vertices();
    println!("Generated {vertex_count} vertices");
    println!("First few vertices:");
    for (i, (_key, vertex)) in dt.tds().vertices().take(10).enumerate() {
        let coords: [f64; 3] = *vertex.point().coords();
        println!(
            "  v{:2}: [{:8.3}, {:8.3}, {:8.3}]",
            i, coords[0], coords[1], coords[2]
        );
    }
    if vertex_count > 10 {
        println!("  ... and {} more vertices", vertex_count - 10);
    }

    println!();

    // Display triangulation properties
    analyze_triangulation(dt.tds());

    // Validate the triangulation
    validate_triangulation(dt.tds());

    // Analyze boundary properties
    analyze_boundary_properties(dt.tds());

    // Performance analysis
    performance_analysis(dt.tds());

    println!("\n=================================================================");
    println!("Example completed successfully!");
    println!("=================================================================");
}

/// Analyze and display triangulation properties
fn analyze_triangulation(tds: &Tds<f64, (), (), 3>) {
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
        let mut shown = 0;

        for (cell_key, cell) in tds.cells() {
            // Count valid cells
            if cell.is_valid().is_ok() {
                valid_cells += 1;
            }

            // Count neighbors
            if let Some(neighbors) = cell.neighbors() {
                total_neighbors += neighbors.len();
            }

            // Show details for first few valid cells
            if shown < 3 {
                println!("    Cell {cell_key:?}:");
                println!("      Vertices: {}", cell.vertices().len());
                if let Some(neighbors) = cell.neighbors() {
                    println!("      Neighbors: {}", neighbors.len());
                }
                shown += 1;
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
fn validate_triangulation(tds: &Tds<f64, (), (), 3>) {
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
fn analyze_boundary_properties(tds: &Tds<f64, (), (), 3>) {
    println!("Boundary Analysis:");
    println!("=================");

    let start = Instant::now();

    // Count boundary facets - use the trait method
    let boundary_facets = match tds.boundary_facets() {
        Ok(iter) => iter,
        Err(e) => {
            println!("Warning: Failed to get boundary facets: {e}");
            println!("  Boundary facets:     0");
            println!("  Boundary computation: {:?}", start.elapsed());
            println!();
            return;
        }
    };
    let boundary_count = boundary_facets.clone().count();

    let boundary_time = start.elapsed();

    println!("  Boundary facets:     {boundary_count}");
    println!("  Boundary computation: {boundary_time:?}");

    if boundary_count > 0 {
        println!("\n  Boundary Details:");
        println!("    • Boundary facets form the convex hull");
        println!("    • Each boundary facet belongs to exactly one cell");

        // Analyze a few boundary facets
        let sample_size = std::cmp::min(3, boundary_count);
        for (i, facet) in boundary_facets.take(sample_size).enumerate() {
            println!("    • Facet {}: key = {:?}", i + 1, facet.key());
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
fn performance_analysis(tds: &Tds<f64, (), (), 3>) {
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
            let _ = tds.boundary_facets().map(Iterator::count).unwrap_or(0);
            start.elapsed()
        })
        .collect();

    let len_u32_boundary = u32::try_from(boundary_times.len()).unwrap_or(1u32);
    let avg_boundary_time: std::time::Duration =
        boundary_times.iter().sum::<std::time::Duration>() / len_u32_boundary;

    println!("\n  Boundary Computation Performance (3 runs):");
    println!("    • Average time: {avg_boundary_time:?}");

    // Memory usage estimation
    let vertex_size = std::mem::size_of::<Vertex<f64, (), 3>>();
    let cell_size = std::mem::size_of::<Cell<f64, (), (), 3>>();
    let estimated_memory = (vertex_count * vertex_size) + (cell_count * cell_size);

    println!("\n  Memory Usage Estimation (stack only, excludes heap allocations):");
    println!("    • Vertex memory: ~{} bytes", vertex_count * vertex_size);
    println!("    • Cell memory:   ~{} bytes", cell_count * cell_size);
    let estimated_f64 = cast(estimated_memory).unwrap_or(0.0f64);
    println!(
        "    • Total memory:  ~{estimated_memory} bytes ({:.1} KB)",
        estimated_f64 / 1024.0
    );
    println!("    Note: This excludes heap-owned data like neighbors and internal collections");

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
