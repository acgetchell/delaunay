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
//! cargo run --example triangulation_3d_20_points
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

use delaunay::geometry::traits::coordinate::CoordinateScalar;
use delaunay::geometry::util::generate_random_triangulation;
use delaunay::prelude::query::*;
use delaunay::topology::characteristics::validation as topology_validation;
use num_traits::NumCast;
use num_traits::cast::cast;
use std::iter::Sum;
use std::ops::{AddAssign, SubAssign};
use std::time::Instant;

const SEED_CANDIDATES: &[u64] = &[1, 7, 11, 42, 99, 123, 666];

fn main() {
    println!("=================================================================");
    println!("3D Delaunay Triangulation Example - 20 Random Points");
    println!("=================================================================\\n");

    // Create Delaunay triangulation with timing.
    // Use a fixed seed + bounds so that `just examples` is reproducible and robust.
    let n_points = 20;
    let bounds = (-3.0, 3.0);
    let seed_override: Option<u64> = std::env::var("DELAUNAY_EXAMPLE_SEED")
        .ok()
        .and_then(|value| value.parse().ok())
        .or(None);
    let seed_candidates: Vec<u64> =
        seed_override.map_or_else(|| SEED_CANDIDATES.to_vec(), |seed| vec![seed]);

    println!(
        "Creating 3D Delaunay triangulation with {n_points} random points in [{:.1}, {:.1}]^3 (seed candidates={seed_candidates:?})...",
        bounds.0, bounds.1
    );
    let start = Instant::now();

    let mut last_error: Option<String> = None;
    let mut used_seed: Option<u64> = None;
    let mut dt: Option<DelaunayTriangulation<FastKernel<f64>, (), (), 3>> = None;
    for &candidate in &seed_candidates {
        match generate_random_triangulation(n_points, bounds, None, Some(candidate)) {
            Ok(triangulation) => {
                dt = Some(triangulation);
                used_seed = Some(candidate);
                break;
            }
            Err(e) => {
                last_error = Some(format!("{e}"));
            }
        }
    }

    let dt: DelaunayTriangulation<FastKernel<f64>, (), (), 3> = if let Some(triangulation) = dt {
        let construction_time = start.elapsed();
        if let Some(seed) = used_seed {
            println!("✓ Triangulation created successfully in {construction_time:?} (seed={seed})");
        } else {
            println!("✓ Triangulation created successfully in {construction_time:?}");
        }
        triangulation
    } else {
        eprintln!(
            "✗ Failed to create triangulation after trying seeds {seed_candidates:?}: {}",
            last_error.unwrap_or_else(|| "unknown error".to_string())
        );
        return;
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
    analyze_triangulation(&dt);

    // Validate the triangulation
    validate_triangulation(&dt);

    // Analyze boundary properties
    analyze_boundary_properties(&dt);

    // Performance analysis
    performance_analysis(&dt);

    println!("\n=================================================================");
    println!("Example completed successfully!");
    println!("=================================================================");
}

/// Analyze and display triangulation properties
fn analyze_triangulation<K, U, V, const D: usize>(dt: &DelaunayTriangulation<K, U, V, D>)
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar + AddAssign + SubAssign + Sum + NumCast,
    U: DataType,
    V: DataType,
{
    println!("Triangulation Analysis:");
    println!("======================");
    println!("  Number of vertices: {}", dt.number_of_vertices());
    println!("  Number of cells:    {}", dt.number_of_cells());
    println!("  Dimension:          {}", dt.dim());

    // Demonstrate the public topology traversal API using an opt-in adjacency index.
    // This avoids per-call allocations in methods like edges()/incident_edges().
    let tri = dt.as_triangulation();
    let index = tri
        .build_adjacency_index()
        .expect("adjacency index should build for a valid triangulation");

    let edge_count = tri.number_of_edges_with_index(&index);
    println!("  Number of edges:    {edge_count}");

    if let Some((v0, _)) = tri.vertices().next() {
        let degree = tri.number_of_incident_edges_with_index(&index, v0);
        println!("  Degree of first vertex: {degree}");
    }

    // Calculate vertex-to-cell ratio
    let vertex_count = dt.number_of_vertices();
    let cell_count = dt.number_of_cells();
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

        for (cell_key, cell) in dt.cells() {
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
fn validate_triangulation<K, U, V, const D: usize>(dt: &DelaunayTriangulation<K, U, V, D>)
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar + AddAssign + SubAssign + Sum + NumCast,
    U: DataType,
    V: DataType,
{
    println!("Triangulation Validation:");
    println!("========================");

    // Levels 1–3: elements + structure + topology
    let start = Instant::now();
    let level_1_3_result = dt.as_triangulation().validate();
    let level_1_3_time = start.elapsed();

    let level_1_3_ok = level_1_3_result.is_ok();
    match level_1_3_result {
        Ok(()) => {
            println!("✓ Levels 1–3: VALID (elements + structure + topology)");
            println!("  Completed in {level_1_3_time:?}");
            println!("\n  Details:");
            println!("    • All vertices and cells are individually valid");
            println!("    • UUID↔key mappings are consistent");
            println!("    • No duplicate cells detected");
            println!("    • Neighbor relationships and facet sharing are consistent");
            println!("    • Manifold topology + Euler characteristic checks pass");

            // Surface the topology module's computed f-vector.
            match topology_validation::validate_triangulation_euler(dt.tds()) {
                Ok(topology) => {
                    println!("\n  Topology (Euler characteristic):");
                    println!("    • Classification: {:?}", topology.classification);
                    match topology.expected {
                        Some(expected) => println!(
                            "    • Euler characteristic: χ = {} (expected {expected})",
                            topology.chi
                        ),
                        None => println!(
                            "    • Euler characteristic: χ = {} (expected unknown)",
                            topology.chi
                        ),
                    }
                    println!("    • f-vector (f0..f{D}): {:?}", topology.counts.by_dim);
                }
                Err(e) => {
                    println!("\n  Topology (Euler characteristic):");
                    println!("    • Failed to compute topology summary: {e}");
                }
            }
        }
        Err(e) => {
            println!("✗ Levels 1–3: INVALID");
            println!("  Failed in {level_1_3_time:?}");
            println!("  Error: {e}");
        }
    }

    // Level 4: Delaunay property only
    let start = Instant::now();
    let level_4_result = dt.is_valid();
    let level_4_time = start.elapsed();

    let level_4_ok = level_4_result.is_ok();
    match level_4_result {
        Ok(()) => {
            println!("\n✓ Level 4: VALID (Delaunay empty circumsphere property)");
            println!("  Completed in {level_4_time:?}");
        }
        Err(e) => {
            println!("\n✗ Level 4: INVALID (Delaunay property)");
            println!("  Failed in {level_4_time:?}");
            println!("  Error: {e}");
        }
    }

    // If something failed, show the cumulative diagnostics report.
    if !level_1_3_ok || !level_4_ok {
        println!("\nValidation report (all violated invariants):");
        match dt.validation_report() {
            Ok(()) => println!("  ✓ No violations reported"),
            Err(report) => {
                for violation in report.violations {
                    println!("  • {:?}: {}", violation.kind, violation.error);
                }
            }
        }
    }

    println!();
}

/// Analyze boundary properties of the triangulation
fn analyze_boundary_properties<K, U, V, const D: usize>(dt: &DelaunayTriangulation<K, U, V, D>)
where
    K: Kernel<D>,
    K::Scalar: AddAssign + SubAssign + Sum + NumCast,
    U: DataType,
    V: DataType,
{
    println!("Boundary Analysis:");
    println!("=================");

    let start = Instant::now();

    // Count boundary facets - use the trait method
    let boundary_facets = dt.boundary_facets();
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
    let vertices = dt.number_of_vertices();
    let cells = dt.number_of_cells();
    println!("\n  Topological Properties:");
    println!("    • Vertices (V): {vertices}");
    println!("    • Cells (C):    {cells}");
    println!("    • Boundary facets: {boundary_count}");

    println!();
}

/// Perform performance analysis and benchmarking
fn performance_analysis<K, U, V, const D: usize>(dt: &DelaunayTriangulation<K, U, V, D>)
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar + AddAssign + SubAssign + Sum + NumCast,
    U: DataType,
    V: DataType,
{
    println!("Performance Analysis:");
    println!("====================");

    let vertex_count = dt.number_of_vertices();
    let cell_count = dt.number_of_cells();

    // Benchmark full validation performance (Levels 1–4)
    let validation_times: Vec<_> = (0..5)
        .map(|_| {
            let start = Instant::now();
            let _ = dt.validate();
            start.elapsed()
        })
        .collect();

    let len_u32 = u32::try_from(validation_times.len()).unwrap_or(1u32);
    let avg_validation_time: std::time::Duration =
        validation_times.iter().sum::<std::time::Duration>() / len_u32;
    let min_validation_time = *validation_times.iter().min().unwrap();
    let max_validation_time = *validation_times.iter().max().unwrap();

    println!("  Full Validation Performance (Levels 1–4, 5 runs):");
    println!("    • Average time: {avg_validation_time:?}");
    println!("    • Min time:     {min_validation_time:?}");
    println!("    • Max time:     {max_validation_time:?}");

    // Benchmark boundary computation
    let boundary_times: Vec<_> = (0..3)
        .map(|_| {
            let start = Instant::now();
            let _ = dt.boundary_facets().count();
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
