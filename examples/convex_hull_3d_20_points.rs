//! # 3D Convex Hull Example with 20 Points
//!
//! This example demonstrates extracting and analyzing a 3D convex hull from a Delaunay
//! triangulation using 20 randomly generated points. It showcases:
//!
//! - Using the `generate_random_triangulation` utility function for convenience
//! - Building a Delaunay triangulation using the incremental cavity-based insertion algorithm
//! - Extracting the convex hull from the triangulation
//! - Analyzing convex hull properties (facets, vertices, dimension)
//! - Testing point containment (inside vs outside the hull)
//! - Finding visible facets from external points
//! - Validating the convex hull's geometric properties
//! - Performance timing for hull construction and queries
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example convex_hull_3d_20_points
//! ```
//!
//! ## Output
//!
//! The example produces detailed output showing:
//! - Generated vertex coordinates
//! - Triangulation and convex hull statistics
//! - Point containment tests
//! - Visible facet analysis
//! - Validation results
//! - Performance metrics

use delaunay::prelude::io::*;
use num_traits::cast::cast;
use std::time::Instant;

fn main() {
    println!("=================================================================");
    println!("3D Convex Hull Example - 20 Random Points");
    println!("=================================================================\\n");

    // Create Delaunay triangulation with timing using the utility function.
    // NOTE: The (n_points, bounds, seed) triple matches a configuration covered by
    // `test_generate_random_triangulation_dimensions` in `geometry::util` to avoid
    // pathological Delaunay-repair failures in CI while still exercising a nontrivial 3D hull.
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
        Ok(dt) => {
            let construction_time = start.elapsed();
            println!("✓ Triangulation created successfully in {construction_time:?}");
            dt
        }
        Err(e) => {
            println!("✗ Failed to create triangulation: {e}");
            return;
        }
    };

    // Display some vertex information
    let vertex_count = dt.tds().number_of_vertices();
    println!("Generated {vertex_count} vertices");
    println!("First few vertices:");
    for (displayed, (_key, vertex)) in dt.tds().vertices().enumerate() {
        if displayed >= 10 {
            break;
        }
        let coords = vertex.point().coords();
        println!(
            "  v{:2}: [{:8.3}, {:8.3}, {:8.3}]",
            displayed, coords[0], coords[1], coords[2]
        );
    }
    if vertex_count > 10 {
        println!("  ... and {} more vertices", vertex_count - 10);
    }

    // Display triangulation properties
    analyze_triangulation(&dt);

    // Extract and analyze convex hull
    extract_and_analyze_convex_hull(&dt);

    // Test point containment
    test_point_containment(&dt);

    // Analyze visible facets
    analyze_visible_facets(&dt);

    // Performance analysis
    performance_analysis(&dt);

    println!("\n=================================================================");
    println!("Example completed successfully!");
    println!("=================================================================");
}

/// Analyze and display triangulation properties
fn analyze_triangulation(dt: &DelaunayTriangulation<FastKernel<f64>, (), (), 3>) {
    println!("Triangulation Analysis:");
    println!("======================");
    println!("  Number of vertices: {}", dt.tds().number_of_vertices());
    println!("  Number of cells:    {}", dt.tds().number_of_cells());
    println!("  Dimension:          {}", dt.tds().dim());

    // Validate the triangulation
    let start = Instant::now();
    match dt.tds().is_valid() {
        Ok(()) => {
            let validation_time = start.elapsed();
            println!("  TDS validation:     ✓ VALID ({validation_time:?})");
            println!("  (Level 2: structural invariants)");
        }
        Err(e) => {
            let validation_time = start.elapsed();
            println!("  TDS validation:     ✗ INVALID ({validation_time:?})");
            println!("  Error: {e}");
        }
    }
    println!();
}

/// Extract and analyze the convex hull from the triangulation
fn extract_and_analyze_convex_hull(dt: &DelaunayTriangulation<FastKernel<f64>, (), (), 3>) {
    println!("Convex Hull Extraction:");
    println!("=======================");

    let start = Instant::now();
    let hull = match ConvexHull::from_triangulation(dt.triangulation()) {
        Ok(convex_hull) => {
            let extraction_time = start.elapsed();
            println!("✓ Convex hull extracted successfully in {extraction_time:?}");
            convex_hull
        }
        Err(e) => {
            println!("✗ Failed to extract convex hull: {e}");
            return;
        }
    };

    println!();
    println!("Convex Hull Analysis:");
    println!("====================");
    println!("  Dimension:          {}", hull.dimension());
    println!("  Number of facets:   {}", hull.number_of_facets());
    println!("  Is empty:           {}", hull.is_empty());

    // Validate the convex hull
    let start = Instant::now();
    match hull.validate(dt.triangulation()) {
        Ok(()) => {
            let validation_time = start.elapsed();
            println!("  Validation:         ✓ VALID ({validation_time:?})");
        }
        Err(e) => {
            let validation_time = start.elapsed();
            println!("  Validation:         ✗ INVALID ({validation_time:?})");
            println!("  Error: {e}");
        }
    }

    // Analyze hull facets
    if hull.number_of_facets() > 0 {
        println!("\n  Facet Analysis:");
        let facets: Vec<_> = hull.facets().collect();
        let sample_size = std::cmp::min(5, facets.len());

        for (i, facet_handle) in facets.iter().take(sample_size).enumerate() {
            // Create FacetView to access facet properties
            if let Ok(facet_view) = FacetView::new(
                dt.tds(),
                facet_handle.cell_key(),
                facet_handle.facet_index(),
            ) {
                let vertex_count = facet_view.vertices().map(Iterator::count).unwrap_or(0);
                let facet_key = facet_view.key().unwrap_or(0);
                println!(
                    "    Facet {}: key = {}, vertices = {}",
                    i + 1,
                    facet_key,
                    vertex_count
                );
            }
        }

        if facets.len() > sample_size {
            println!("    ... and {} more facets", facets.len() - sample_size);
        }
    }

    println!();
}

/// Test point containment with various points (updated to work with generated triangulation)
fn test_point_containment(dt: &DelaunayTriangulation<FastKernel<f64>, (), (), 3>) {
    println!("Point Containment Tests:");
    println!("=======================");

    // Extract convex hull for containment tests
    let hull = match ConvexHull::from_triangulation(dt.triangulation()) {
        Ok(h) => h,
        Err(e) => {
            println!("✗ Failed to extract convex hull for containment tests: {e}");
            return;
        }
    };

    // Test 1: Points inside the convex hull (centroid and near-centroid points)
    println!("  Testing interior points:");

    // Calculate centroid of triangulation vertices
    let mut centroid = [0.0f64; 3];
    let vertex_count = dt.tds().number_of_vertices();
    for (_, vertex) in dt.tds().vertices() {
        let coords: [f64; 3] = vertex.into();
        for (i, &coord) in coords.iter().enumerate() {
            centroid[i] += coord;
        }
    }
    let vertex_count_f64 = cast(vertex_count).unwrap_or(1.0f64);
    for coord in &mut centroid {
        *coord /= vertex_count_f64;
    }

    let centroid_point = Point::new(centroid);
    test_point_containment_single(&hull, &centroid_point, "Centroid", dt);

    // Test slightly offset from centroid (should still be inside)
    let near_centroid = Point::new([centroid[0] + 0.1, centroid[1] + 0.1, centroid[2] + 0.1]);
    test_point_containment_single(&hull, &near_centroid, "Near centroid", dt);

    // Test 2: Points clearly outside the convex hull
    println!("\n  Testing exterior points:");

    let far_point = Point::new([50.0, 50.0, 50.0]);
    test_point_containment_single(&hull, &far_point, "Far exterior", dt);

    let axis_point = Point::new([20.0, 0.0, 0.0]);
    test_point_containment_single(&hull, &axis_point, "X-axis exterior", dt);

    let negative_point = Point::new([-20.0, -20.0, -20.0]);
    test_point_containment_single(&hull, &negative_point, "Negative exterior", dt);

    // Test 3: Sample triangulation vertices (should be on boundary or inside)
    println!("\n  Testing triangulation vertices:");
    let sample_vertices = std::cmp::min(3, vertex_count);
    for (i, (_, vertex)) in dt.tds().vertices().enumerate().take(sample_vertices) {
        let point: Point<f64, 3> = vertex.into();
        test_point_containment_single(
            &hull,
            &point,
            &format!("Triangulation vertex {}", i + 1),
            dt,
        );
    }

    println!();
}

/// Test containment for a single point and display results
fn test_point_containment_single(
    hull: &ConvexHull<FastKernel<f64>, (), (), 3>,
    point: &Point<f64, 3>,
    description: &str,
    dt: &DelaunayTriangulation<FastKernel<f64>, (), (), 3>,
) {
    let coords = point.coords();

    let start = Instant::now();
    match hull.is_point_outside(point, dt.triangulation()) {
        Ok(is_outside) => {
            let test_time = start.elapsed();
            let status = if is_outside {
                "OUTSIDE"
            } else {
                "INSIDE/BOUNDARY"
            };
            println!(
                "    {} [{:6.2}, {:6.2}, {:6.2}]: {} ({:?})",
                description, coords[0], coords[1], coords[2], status, test_time
            );
        }
        Err(e) => {
            println!(
                "    {} [{:6.2}, {:6.2}, {:6.2}]: ERROR - {}",
                description, coords[0], coords[1], coords[2], e
            );
        }
    }
}

/// Analyze visible facets from external points
fn analyze_visible_facets(dt: &DelaunayTriangulation<FastKernel<f64>, (), (), 3>) {
    println!("Visible Facet Analysis:");
    println!("======================");

    // Extract convex hull for visible facet analysis
    let hull = match ConvexHull::from_triangulation(dt.triangulation()) {
        Ok(h) => h,
        Err(e) => {
            println!("✗ Failed to extract convex hull for visible facet analysis: {e}");
            return;
        }
    };

    // Test points at different distances and directions
    let test_points = vec![
        (Point::new([15.0, 0.0, 0.0]), "X-axis (+15)"),
        (Point::new([0.0, 15.0, 0.0]), "Y-axis (+15)"),
        (Point::new([0.0, 0.0, 15.0]), "Z-axis (+15)"),
        (Point::new([15.0, 15.0, 15.0]), "Diagonal (+15, +15, +15)"),
        (
            Point::new([-15.0, -15.0, -15.0]),
            "Diagonal (-15, -15, -15)",
        ),
    ];

    for (point, description) in test_points {
        let coords = point.coords();

        let start = Instant::now();
        match hull.find_visible_facets(&point, dt.triangulation()) {
            Ok(visible_facets) => {
                let query_time = start.elapsed();
                let visible_count = visible_facets.len();
                let total_facets = hull.number_of_facets();
                let visibility_ratio = if total_facets > 0 {
                    let visible_f64 = cast(visible_count).unwrap_or(0.0f64);
                    let total_f64 = cast(total_facets).unwrap_or(1.0f64);
                    visible_f64 / total_f64 * 100.0
                } else {
                    0.0
                };

                println!(
                    "  {} [{:6.2}, {:6.2}, {:6.2}]:",
                    description, coords[0], coords[1], coords[2]
                );
                println!(
                    "    Visible facets: {visible_count}/{total_facets} ({visibility_ratio:.1}%)"
                );
                println!("    Query time: {query_time:?}");
            }
            Err(e) => {
                println!(
                    "  {} [{:6.2}, {:6.2}, {:6.2}]: ERROR - {}",
                    description, coords[0], coords[1], coords[2], e
                );
            }
        }
    }

    // Test nearest visible facet finding
    println!("\n  Nearest Visible Facet Analysis:");
    let test_point = Point::new([15.0, 15.0, 15.0]);
    let coords = test_point.coords();

    let start = Instant::now();
    match hull.find_nearest_visible_facet(&test_point, dt.triangulation()) {
        Ok(Some(nearest_facet_index)) => {
            let query_time = start.elapsed();
            println!(
                "    Test point [{:.2}, {:.2}, {:.2}]:",
                coords[0], coords[1], coords[2]
            );
            println!("    Nearest visible facet: index = {nearest_facet_index}");
            println!("    Query time: {query_time:?}");
        }
        Ok(None) => {
            println!(
                "    Test point [{:.2}, {:.2}, {:.2}]: No visible facets found",
                coords[0], coords[1], coords[2]
            );
        }
        Err(e) => {
            println!(
                "    Test point [{:.2}, {:.2}, {:.2}]: ERROR - {}",
                coords[0], coords[1], coords[2], e
            );
        }
    }

    println!();
}

/// Perform performance analysis and benchmarking
fn performance_analysis(dt: &DelaunayTriangulation<FastKernel<f64>, (), (), 3>) {
    println!("Performance Analysis:");
    println!("====================");

    // Benchmark convex hull extraction
    let extraction_times: Vec<_> = (0..5)
        .map(|_| {
            let start = Instant::now();
            let _ = ConvexHull::from_triangulation(dt.triangulation());
            start.elapsed()
        })
        .collect();

    let len_u32 = u32::try_from(extraction_times.len()).unwrap_or(1u32);
    let avg_extraction_time: std::time::Duration =
        extraction_times.iter().sum::<std::time::Duration>() / len_u32;
    let min_extraction_time = *extraction_times.iter().min().unwrap();
    let max_extraction_time = *extraction_times.iter().max().unwrap();

    println!("  Convex Hull Extraction (5 runs):");
    println!("    • Average time: {avg_extraction_time:?}");
    println!("    • Min time:     {min_extraction_time:?}");
    println!("    • Max time:     {max_extraction_time:?}");

    // Benchmark point containment queries
    let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();
    let test_point = Point::new([5.0, 5.0, 5.0]);

    let containment_times: Vec<_> = (0..10)
        .map(|_| {
            let start = Instant::now();
            let _ = hull.is_point_outside(&test_point, dt.triangulation());
            start.elapsed()
        })
        .collect();

    let len_u32 = u32::try_from(containment_times.len()).unwrap_or(1u32);
    let avg_containment_time: std::time::Duration =
        containment_times.iter().sum::<std::time::Duration>() / len_u32;

    println!("\n  Point Containment Queries (10 runs):");
    println!("    • Average time: {avg_containment_time:?}");

    // Benchmark visible facet queries
    let external_point = Point::new([15.0, 15.0, 15.0]);

    let visibility_times: Vec<_> = (0..5)
        .map(|_| {
            let start = Instant::now();
            let _ = hull.find_visible_facets(&external_point, dt.triangulation());
            start.elapsed()
        })
        .collect();

    let len_u32 = u32::try_from(visibility_times.len()).unwrap_or(1u32);
    let avg_visibility_time: std::time::Duration =
        visibility_times.iter().sum::<std::time::Duration>() / len_u32;

    println!("\n  Visible Facet Queries (5 runs):");
    println!("    • Average time: {avg_visibility_time:?}");

    // Performance per vertex ratios
    let vertex_count = dt.tds().number_of_vertices();
    let facet_count = hull.number_of_facets();

    if vertex_count > 0 && facet_count > 0 {
        let extraction_nanos = cast(avg_extraction_time.as_nanos()).unwrap_or(0.0f64);
        let vertex_f64 = cast(vertex_count).unwrap_or(1.0f64);
        let facet_f64 = cast(facet_count).unwrap_or(1.0f64);

        let extraction_per_vertex = extraction_nanos / vertex_f64;
        let extraction_per_facet = extraction_nanos / facet_f64;

        println!("\n  Performance Ratios:");
        println!("    • Hull extraction per vertex: {extraction_per_vertex:.2} ns");
        println!("    • Hull extraction per facet:  {extraction_per_facet:.2} ns");
    }

    // Memory usage estimation
    let hull_size = std::mem::size_of::<ConvexHull<FastKernel<f64>, (), (), 3>>();
    // Phase 3C: Facets are now lightweight (CellKey, u8) tuples
    let facet_handle_size = std::mem::size_of::<(delaunay::core::CellKey, u8)>();
    let estimated_hull_memory = hull_size + (facet_count * facet_handle_size);

    println!("\n  Memory Usage Estimation:");
    let estimated_f64 = cast(estimated_hull_memory).unwrap_or(0.0f64);
    println!(
        "    • Convex hull memory: ~{estimated_hull_memory} bytes ({:.1} KB)",
        estimated_f64 / 1024.0
    );
}
