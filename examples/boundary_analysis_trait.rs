//! # `boundary_analysis_trait` Example
//!
//! This example demonstrates the `BoundaryAnalysis` trait for triangulation boundary operations
//! and showcases practical real-world use cases. It shows how boundary analysis functions have
//! been cleanly separated from the TDS struct using a trait-based approach.
//!
//! ## Features Demonstrated
//!
//! - **Multi-dimensional Support**: 2D edge extraction, 3D facet analysis, 4D+ support
//! - **Real-world Use Cases**: Mesh vertex extraction, surface area calculation, performance comparison
//! - **API Patterns**: Different methods for different performance needs
//! - **Error Handling**: Robust error scenarios and edge cases
//! - **Trait Benefits**: Modularity, testability, extensibility
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --example boundary_analysis_trait
//! ```

use approx::relative_eq;
use slotmap::Key;
/// Calculates the perimeter of a 2D triangle.
fn calculate_triangle_perimeter() {
    // Create a 3-4-5 right triangle
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([3.0, 0.0]),
        vertex!([0.0, 4.0]),
    ];

    let tds: Tds<f64, Option<()>, Option<()>, 2> = match Tds::new(&vertices) {
        Ok(tds) => tds,
        Err(e) => {
            println!("    ❌ Failed to create 2D triangulation: {e}");
            return;
        }
    };

    match tds.boundary_facets() {
        Ok(boundary_facets) => {
            println!("    Triangle has {} boundary edges", boundary_facets.len());

            // Calculate perimeter by summing edge lengths
            let mut total_perimeter = 0.0;
            for (i, facet) in boundary_facets.iter().enumerate() {
                let vertices = facet.vertices();
                if vertices.len() == 2 {
                    // Get coordinates of both vertices
                    let p1_coords: [f64; 2] = (&vertices[0]).into();
                    let p2_coords: [f64; 2] = (&vertices[1]).into();

                    // Calculate distance using hypot
                    let diff = [p2_coords[0] - p1_coords[0], p2_coords[1] - p1_coords[1]];
                    let edge_length = hypot(diff);

                    total_perimeter += edge_length;
                    println!(
                        "    Edge {}: ({:.1}, {:.1}) → ({:.1}, {:.1}) = {:.1}",
                        i + 1,
                        p1_coords[0],
                        p1_coords[1],
                        p2_coords[0],
                        p2_coords[1],
                        edge_length
                    );
                }
            }
            println!("    📏 Total Perimeter: {total_perimeter:.1} units");
            println!("    ✅ Expected: 12.0 units (3 + 4 + 5 triangle)");
        }
        Err(e) => println!("    ❌ Failed to get boundary facets: {e}"),
    }
}

/// Calculates the surface area of a 3D tetrahedron.
fn calculate_tetrahedron_surface_area() {
    // Create a tetrahedron with known surface area
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]), // Origin
        vertex!([2.0, 0.0, 0.0]), // Base vertex on X-axis
        vertex!([1.0, 2.0, 0.0]), // Base vertex in XY-plane
        vertex!([1.0, 1.0, 2.0]), // Apex vertex
    ];

    let tds: Tds<f64, Option<()>, Option<()>, 3> = match Tds::new(&vertices) {
        Ok(tds) => tds,
        Err(e) => {
            println!("    ❌ Failed to create 3D triangulation: {e}");
            return;
        }
    };

    match tds.boundary_facets() {
        Ok(boundary_facets) => {
            println!(
                "    Tetrahedron has {} boundary faces (triangles)",
                boundary_facets.len()
            );

            // Calculate surface area using facet measures
            match surface_measure(&boundary_facets) {
                Ok(total_surface_area) => {
                    println!("    📐 Total Surface Area: {total_surface_area:.3} square units");

                    // Also calculate individual facet areas for demonstration
                    println!("    Individual face areas:");
                    for (i, facet) in boundary_facets.iter().enumerate() {
                        let facet_vertices = facet.vertices();
                        let points: Vec<Point<f64, 3>> = facet_vertices
                            .iter()
                            .map(|v| {
                                let coords: [f64; 3] = v.into();
                                Point::new(coords)
                            })
                            .collect();

                        match facet_measure(&points) {
                            Ok(area) => println!("      Face {}: {:.3} square units", i + 1, area),
                            Err(e) => {
                                println!("      Face {}: Error calculating area: {}", i + 1, e);
                            }
                        }
                    }
                }
                Err(e) => println!("    ❌ Failed to calculate surface area: {e}"),
            }
        }
        Err(e) => println!("    ❌ Failed to get boundary facets: {e}"),
    }
}

/// Analyzes the boundary of a 4D triangulation.
fn analyze_4d_boundary() {
    // Create a 4D simplex (5-cell)
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0]), // Origin
        vertex!([1.0, 0.0, 0.0, 0.0]), // Unit X
        vertex!([0.0, 1.0, 0.0, 0.0]), // Unit Y
        vertex!([0.0, 0.0, 1.0, 0.0]), // Unit Z
        vertex!([0.0, 0.0, 0.0, 1.0]), // Unit W (4th dimension)
    ];

    let tds: Tds<f64, Option<()>, Option<()>, 4> = match Tds::new(&vertices) {
        Ok(tds) => tds,
        Err(e) => {
            println!("    ❌ Failed to create 4D triangulation: {e}");
            return;
        }
    };

    let start = Instant::now();
    let boundary_count = tds.number_of_boundary_facets();
    let count_time = start.elapsed();

    println!("    🌌 4D Simplex Analysis:");
    println!("      - {} vertices in 4D space", tds.number_of_vertices());
    println!(
        "      - {} 4D cells (hypertetrahedra)",
        tds.number_of_cells()
    );
    println!("      - {boundary_count} boundary facets (3D tetrahedra) [{count_time:?}]");

    // Analyze boundary facets in detail
    match tds.boundary_facets() {
        Ok(boundary_facets) => {
            let facets_time = Instant::now();

            println!("    📊 Boundary Facet Analysis:");
            println!(
                "      - Retrieved {} facets in {:?}",
                boundary_facets.len(),
                facets_time.elapsed()
            );

            // Calculate total 3D boundary volume
            match surface_measure(&boundary_facets) {
                Ok(total_volume) => {
                    println!("      - Total boundary volume: {total_volume:.6} cubic units");
                }
                Err(e) => println!("      - Error calculating boundary volume: {e}"),
            }

            // Demonstrate facet-to-cell boundary checking
            if let Some(cell) = tds.cells().values().next() {
                match cell.facets() {
                    Ok(cell_facets) => {
                        println!("      - Sample 4D cell has {} 3D faces", cell_facets.len());
                        let mut boundary_face_count = 0;
                        let mut internal_faces = 0;

                        for facet in &cell_facets {
                            if tds.is_boundary_facet(facet) {
                                boundary_face_count += 1;
                            } else {
                                internal_faces += 1;
                            }
                        }

                        println!(
                            "        • {boundary_face_count} boundary faces, {internal_faces} internal faces"
                        );
                    }
                    Err(e) => println!("        • Error getting cell facets: {e}"),
                }
            }

            println!("    ✅ 4D Boundary analysis completed successfully!");
        }
        Err(e) => println!("    ❌ Failed to retrieve boundary facets: {e}"),
    }
}

// # `boundary_analysis_trait` Example

/// Demonstrate multi-dimensional support for `BoundaryAnalysis` trait
fn demonstrate_multi_dimensional_support() {
    println!("\n🌍 2. Multi-Dimensional Support");
    println!("==============================\n");

    // 2D Example: Triangle (2-simplex in 2D)
    println!("📐 2D Triangulation (Triangle):");
    let vertices_2d = vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.5, 1.0]),
    ];

    let tds_2d: Tds<f64, Option<()>, Option<()>, 2> =
        Tds::new(&vertices_2d).expect("Failed to create 2D triangulation");

    let boundary_count_2d = tds_2d.number_of_boundary_facets();
    println!(
        "  - {} vertices → {} cells → {} boundary edges",
        tds_2d.number_of_vertices(),
        tds_2d.number_of_cells(),
        boundary_count_2d
    );
    println!("  - Expected: 3 boundary edges (triangle perimeter)\n");

    // 3D Example: Tetrahedron (3-simplex in 3D)
    println!("🔺 3D Triangulation (Tetrahedron):");
    let vertices_3d = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    let tds_3d: Tds<f64, Option<()>, Option<()>, 3> =
        Tds::new(&vertices_3d).expect("Failed to create 3D triangulation");

    let boundary_count_3d = tds_3d.number_of_boundary_facets();
    println!(
        "  - {} vertices → {} cells → {} boundary triangles",
        tds_3d.number_of_vertices(),
        tds_3d.number_of_cells(),
        boundary_count_3d
    );
    println!("  - Expected: 4 boundary triangles (tetrahedron surface)\n");

    // 4D Example: 5-cell (4-simplex in 4D)
    println!("🌌 4D Triangulation (5-Cell/Pentachoron):");
    let vertices_4d = vec![
        vertex!([0.0, 0.0, 0.0, 0.0]), // Origin
        vertex!([1.0, 0.0, 0.0, 0.0]), // Unit x
        vertex!([0.0, 1.0, 0.0, 0.0]), // Unit y
        vertex!([0.0, 0.0, 1.0, 0.0]), // Unit z
        vertex!([0.0, 0.0, 0.0, 1.0]), // Unit w (4th dimension)
    ];

    let tds_4d: Tds<f64, Option<()>, Option<()>, 4> =
        Tds::new(&vertices_4d).expect("Failed to create 4D triangulation");

    let start = Instant::now();
    let boundary_count_4d = tds_4d.number_of_boundary_facets();
    let count_time_4d = start.elapsed();

    println!(
        "  - {} vertices → {} cells → {} boundary tetrahedra ({:?})",
        tds_4d.number_of_vertices(),
        tds_4d.number_of_cells(),
        boundary_count_4d,
        count_time_4d
    );
    println!("  - Expected: 5 boundary tetrahedra (5-cell surface)");

    // Demonstrate that the API is consistent across dimensions
    let start = Instant::now();
    let boundary_facets_4d = tds_4d
        .boundary_facets()
        .expect("Failed to get 4D boundary facets");
    let facets_time_4d = start.elapsed();

    println!(
        "  - Retrieved {} boundary facets in {:?}",
        boundary_facets_4d.len(),
        facets_time_4d
    );

    // Show individual boundary facet checking in 4D
    if let Some(cell) = tds_4d.cells().values().next() {
        let facets = cell.facets().expect("Failed to get facets from 4D cell");
        println!("  - 4D Cell has {} facets (tetrahedra)", facets.len());

        let mut boundary_facet_count = 0;
        for facet in &facets {
            if tds_4d.is_boundary_facet(facet) {
                boundary_facet_count += 1;
            }
        }
        println!("  - {boundary_facet_count} of those are boundary facets");
    }

    println!("\n📊 Dimensional Scaling Analysis:");
    println!("  • 2D: {boundary_count_2d} boundary edges");
    println!("  • 3D: {boundary_count_3d} boundary triangles");
    println!("  • 4D: {boundary_count_4d} boundary tetrahedra");
    println!("  • Pattern: Each dimension has (D+1) boundary (D-1)-simplices");
    println!("    - 2D triangle: 3 edges");
    println!("    - 3D tetrahedron: 4 triangles");
    println!("    - 4D 5-cell: 5 tetrahedra");

    println!();
}

/// Demonstrate real-world use cases for `BoundaryAnalysis` trait
fn demonstrate_real_world_use_cases() {
    println!("\n💡 3. Real-World Use Cases");
    println!("=========================\n");

    // Use Case 1: 2D Triangle Perimeter Calculation
    println!("📐 2D Triangle Perimeter Calculation:");
    calculate_triangle_perimeter();

    // Use Case 2: 3D Tetrahedron Surface Area
    println!("\n🔺 3D Tetrahedron Surface Area Calculation:");
    calculate_tetrahedron_surface_area();

    // Use Case 3: 4D Boundary Analysis
    println!("\n🌌 4D Triangulation Boundary Analysis:");
    analyze_4d_boundary();

    println!();
}

/// Demonstrate performance characteristics of `BoundaryAnalysis` methods
fn demonstrate_performance_characteristics() {
    println!("\n⚡ 4. Performance Characteristics");
    println!("===============================\n");

    // Test 1: Method Performance Comparison
    println!("📊 Method Performance Comparison:");
    println!("=================================\n");

    // Create triangulations of increasing complexity
    let test_cases = vec![
        ("Simple 3D (4 vertices)", create_simple_tetrahedron()),
        ("Complex 3D (8 vertices)", create_cube_triangulation()),
        ("4D Simplex (5 vertices)", create_4d_simplex()),
    ];

    for (name, tds_result) in test_cases {
        match tds_result {
            Ok(tds) => demonstrate_performance(name, &*tds),
            Err(e) => println!("  ❌ Failed to create {name}: {e}"),
        }
    }

    // Test 2: Scaling Analysis
    println!("\n📈 Scaling Analysis:");
    println!("===================\n");

    demonstrate_scaling_performance();

    // Test 3: Memory Usage Patterns
    println!("\n💾 Memory Usage Patterns:");
    println!("========================\n");

    demonstrate_memory_patterns();
}

/// Create a simple tetrahedron for testing
fn create_simple_tetrahedron() -> Result<Box<dyn BoundaryAnalysisTestable>, String> {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let tds: Tds<f64, Option<()>, Option<()>, 3> =
        Tds::new(&vertices).map_err(|e| format!("Failed to create tetrahedron: {e}"))?;
    Ok(Box::new(tds))
}

/// Create a more complex cube-based triangulation
fn create_cube_triangulation() -> Result<Box<dyn BoundaryAnalysisTestable>, String> {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]), // Bottom face
        vertex!([1.0, 0.0, 0.0]),
        vertex!([1.0, 1.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]), // Top face
        vertex!([1.0, 0.0, 1.0]),
        vertex!([1.0, 1.0, 1.0]),
        vertex!([0.0, 1.0, 1.0]),
    ];
    let tds: Tds<f64, Option<()>, Option<()>, 3> =
        Tds::new(&vertices).map_err(|e| format!("Failed to create cube triangulation: {e}"))?;
    Ok(Box::new(tds))
}

/// Create a 4D simplex
fn create_4d_simplex() -> Result<Box<dyn BoundaryAnalysisTestable>, String> {
    let vertices: Vec<delaunay::core::vertex::Vertex<f64, Option<()>, 4>> = vec![
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0]),
    ];
    let tds: Tds<f64, Option<()>, Option<()>, 4> =
        Tds::new(&vertices).map_err(|e| format!("Failed to create 4D simplex: {e}"))?;
    Ok(Box::new(tds))
}

/// Trait to abstract over different TDS dimensions for performance testing
trait BoundaryAnalysisTestable {
    fn get_stats(&self) -> (usize, usize); // (vertices, cells)
    fn time_boundary_facets(&self) -> (usize, std::time::Duration);
    fn time_boundary_count(&self) -> (usize, std::time::Duration);
    fn time_individual_checks(&self) -> (usize, usize, std::time::Duration); // (total_facets, boundary_facets, duration)
}

/// Implement the testable trait for 3D TDS
impl BoundaryAnalysisTestable for Tds<f64, Option<()>, Option<()>, 3> {
    fn get_stats(&self) -> (usize, usize) {
        (self.number_of_vertices(), self.number_of_cells())
    }

    fn time_boundary_facets(&self) -> (usize, std::time::Duration) {
        let start = Instant::now();
        let facets = self.boundary_facets().unwrap_or_default();
        let duration = start.elapsed();
        (facets.len(), duration)
    }

    fn time_boundary_count(&self) -> (usize, std::time::Duration) {
        let start = Instant::now();
        let count = self.number_of_boundary_facets();
        let duration = start.elapsed();
        (count, duration)
    }

    fn time_individual_checks(&self) -> (usize, usize, std::time::Duration) {
        let start = Instant::now();
        let mut total_facets = 0;
        let mut boundary_facets = 0;

        for cell in self.cells().values() {
            if let Ok(facets) = cell.facets() {
                for facet in &facets {
                    total_facets += 1;
                    if self.is_boundary_facet(facet) {
                        boundary_facets += 1;
                    }
                }
            }
        }

        let duration = start.elapsed();
        (total_facets, boundary_facets, duration)
    }
}

/// Implement the testable trait for 4D TDS
impl BoundaryAnalysisTestable for Tds<f64, Option<()>, Option<()>, 4> {
    fn get_stats(&self) -> (usize, usize) {
        (self.number_of_vertices(), self.number_of_cells())
    }

    fn time_boundary_facets(&self) -> (usize, std::time::Duration) {
        let start = Instant::now();
        let facets = self.boundary_facets().unwrap_or_default();
        let duration = start.elapsed();
        (facets.len(), duration)
    }

    fn time_boundary_count(&self) -> (usize, std::time::Duration) {
        let start = Instant::now();
        let count = self.number_of_boundary_facets();
        let duration = start.elapsed();
        (count, duration)
    }

    fn time_individual_checks(&self) -> (usize, usize, std::time::Duration) {
        let start = Instant::now();
        let mut total_facets = 0;
        let mut boundary_facets = 0;

        for cell in self.cells().values() {
            if let Ok(facets) = cell.facets() {
                for facet in &facets {
                    total_facets += 1;
                    if self.is_boundary_facet(facet) {
                        boundary_facets += 1;
                    }
                }
            }
        }

        let duration = start.elapsed();
        (total_facets, boundary_facets, duration)
    }
}

/// Demonstrate performance characteristics for a triangulation
fn demonstrate_performance(name: &str, tds: &dyn BoundaryAnalysisTestable) {
    let (vertices, cells) = tds.get_stats();
    println!("🔺 {name}:");
    println!("  └─ {vertices} vertices, {cells} cells");

    // Method 1: Full boundary facets retrieval
    let (facet_count, facets_time) = tds.time_boundary_facets();
    println!("  ├─ boundary_facets(): {facet_count} facets in {facets_time:?}");

    // Method 2: Count-only operation
    let (count, count_time) = tds.time_boundary_count();
    println!("  ├─ number_of_boundary_facets(): {count} count in {count_time:?}");

    // Method 3: Individual facet checking
    let (total, boundary, check_time) = tds.time_individual_checks();
    println!("  └─ is_boundary_facet(): {boundary}/{total} checks in {check_time:?}");

    // Performance analysis
    if facets_time > count_time {
        #[allow(clippy::cast_precision_loss)]
        let ratio = facets_time.as_nanos() as f64 / count_time.as_nanos() as f64;
        println!(
            "    💡 boundary_facets() is {ratio:.1}x slower than count (due to facet creation)"
        );
    }

    println!();
}

/// Demonstrate scaling performance across different triangulation sizes
fn demonstrate_scaling_performance() {
    println!("Testing method efficiency across triangulation complexity:\n");

    // Create triangulations of increasing size
    let sizes = vec![
        ("Minimal", 4), // Single tetrahedron
        ("Small", 10),  // Small cluster
        ("Medium", 40), // Medium complexity
        ("Large", 100), // Large triangulation
    ];

    for (size_name, vertex_count) in sizes {
        println!("📏 {size_name} triangulation ({vertex_count} vertices):");

        // Generate vertices in a more complex pattern
        let vertices = generate_test_vertices(vertex_count);

        match Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices) {
            Ok(tds) => {
                let boundary_count = tds.number_of_boundary_facets();

                // Time the count operation (most efficient)
                let start = Instant::now();
                let _ = tds.number_of_boundary_facets();
                let count_time = start.elapsed();

                // Time the full facet retrieval
                let start = Instant::now();
                let _boundary_facets = tds.boundary_facets().unwrap_or_default();
                let facets_time = start.elapsed();

                println!("  ├─ {boundary_count} boundary facets detected");
                println!("  ├─ Count operation: {count_time:?}");
                println!("  ├─ Full retrieval: {facets_time:?}");

                // Calculate efficiency ratio
                if facets_time.as_nanos() > 0 && count_time.as_nanos() > 0 {
                    #[allow(clippy::cast_precision_loss)]
                    let efficiency_ratio =
                        count_time.as_nanos() as f64 / facets_time.as_nanos() as f64;
                    println!(
                        "  └─ Count is {:.1}x more efficient than retrieval\n",
                        1.0 / efficiency_ratio
                    );
                } else {
                    println!("  └─ Both operations too fast to measure accurately\n");
                }
            }
            Err(e) => println!("  ❌ Failed to create triangulation: {e}\n"),
        }
    }
}

/// Generate test vertices for scaling analysis using a simple grid-based approach
fn generate_test_vertices(count: usize) -> Vec<delaunay::core::vertex::Vertex<f64, Option<()>, 3>> {
    let mut vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    // Add vertices in a simple 3D grid pattern with some variation
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let grid_size = ((count - 4) as f64).cbrt().ceil() as usize;
    let spacing = 1.0;
    let mut added = 0;

    for x in 0..grid_size {
        for y in 0..grid_size {
            for z in 0..grid_size {
                if vertices.len() >= count {
                    break;
                }
                if added == 0 && x == 0 && y == 0 && z == 0 {
                    // Skip origin as it's already added
                    continue;
                }

                // Add small perturbation to avoid perfect grid degeneracies
                let perturbation = 0.1;
                let x_f64: f64 = safe_usize_to_scalar(x).unwrap_or(0.0);
                let y_f64: f64 = safe_usize_to_scalar(y).unwrap_or(0.0);
                let z_f64: f64 = safe_usize_to_scalar(z).unwrap_or(0.0);
                let px = x_f64.mul_add(spacing, (x_f64 * 0.123).sin() * perturbation);
                #[allow(clippy::suboptimal_flops)]
                let py = y_f64.mul_add(spacing, (y_f64 * 0.456).cos() * perturbation);
                #[allow(clippy::suboptimal_flops)]
                let pz = z_f64.mul_add(spacing, (z_f64 * 0.789).sin() * perturbation);
                #[allow(clippy::suboptimal_flops)]
                vertices.push(vertex!([px + 0.5, py + 0.5, pz + 0.5]));
                added += 1;
            }
            if vertices.len() >= count {
                break;
            }
        }
        if vertices.len() >= count {
            break;
        }
    }

    vertices
}

/// Demonstrate memory usage patterns for different methods
fn demonstrate_memory_patterns() {
    println!("Analyzing memory allocation patterns:\n");

    // Create a reasonably complex triangulation for memory analysis
    let vertices = generate_test_vertices(20);

    match Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices) {
        Ok(tds) => {
            println!("🧠 Memory Analysis for 20-vertex triangulation:");
            println!("  ├─ {} cells in triangulation", tds.number_of_cells());

            // Method comparison from memory perspective
            let boundary_count = tds.number_of_boundary_facets();
            println!("  ├─ {boundary_count} boundary facets exist");

            println!("\n  📋 Method Memory Characteristics:");
            println!("  ├─ number_of_boundary_facets(): O(1) memory, O(facets) time");
            println!("  ├─   • Just counts, no allocation of facet objects");
            println!("  ├─   • Most efficient for simple boundary counting");
            println!("  ├─ boundary_facets(): O(boundary_facets) memory, O(facets) time");
            println!("  ├─   • Allocates Vec<Facet> with full facet data");
            println!("  ├─   • Use when you need to process facet geometry");
            println!("  └─ is_boundary_facet(): O(1) memory, O(1) time per facet");
            println!("      • No allocation, just boundary test");
            println!("      • Use for selective facet filtering\n");

            println!("  💡 Performance Recommendations:");
            println!("  ├─ Use number_of_boundary_facets() for simple counting");
            println!("  ├─ Use boundary_facets() when you need facet geometry");
            println!("  ├─ Use is_boundary_facet() for filtering specific facets");
            println!("  └─ Combine methods: count first, then retrieve if needed");
        }
        Err(e) => println!("❌ Failed to create test triangulation: {e}"),
    }
}

/// Demonstrate error handling for `BoundaryAnalysis` trait
fn demonstrate_error_handling() {
    println!("\n⛔ 5. Error Handling");
    println!("===================\n");

    // Test Case 1: Empty triangulation (no vertices)
    println!("🧪 Test Case 1: Empty Triangulation");
    demonstrate_empty_triangulation_error();

    // Test Case 2: Insufficient vertices (fewer than D+1)
    println!("\n🧪 Test Case 2: Insufficient Vertices");
    demonstrate_insufficient_vertices_error();

    // Test Case 3: Degenerate/Collinear vertices
    println!("\n🧪 Test Case 3: Degenerate Vertex Configurations");
    demonstrate_degenerate_vertices_error();

    // Test Case 4: Graceful handling of edge cases
    println!("\n🧪 Test Case 4: Boundary Analysis Edge Cases");
    demonstrate_boundary_analysis_edge_cases();

    // Test Case 5: Recovery and best practices
    println!("\n🧪 Test Case 5: Error Recovery Patterns");
    demonstrate_error_recovery_patterns();
}

/// Test boundary analysis with empty triangulation
fn demonstrate_empty_triangulation_error() {
    println!("  Creating triangulation with no vertices...");

    let empty_vertices: Vec<delaunay::core::vertex::Vertex<f64, Option<()>, 3>> = vec![];

    match Tds::<f64, Option<()>, Option<()>, 3>::new(&empty_vertices) {
        Ok(_) => {
            println!("  ❌ Unexpected: Empty triangulation was created");
        }
        Err(e) => {
            println!("  ✅ Expected error caught: {e}");
            println!("  💡 Recommendation: Always validate input has at least D+1 vertices");
        }
    }
}

/// Test boundary analysis with insufficient vertices
fn demonstrate_insufficient_vertices_error() {
    // For 3D triangulation, need at least 4 vertices
    println!("  Creating 3D triangulation with only 3 vertices (need 4)...");

    let insufficient_vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        // Missing 4th vertex for 3D triangulation
    ];

    match Tds::<f64, Option<()>, Option<()>, 3>::new(&insufficient_vertices) {
        Ok(_) => {
            println!("  ❌ Unexpected: Insufficient vertex triangulation was created");
        }
        Err(e) => {
            println!("  ✅ Expected error caught: {e}");
            println!(
                "  💡 Recommendation: For D-dimensional triangulation, provide ≥ D+1 vertices"
            );
        }
    }
}

/// Test with degenerate vertex configurations
fn demonstrate_degenerate_vertices_error() {
    // Test 1: All vertices collinear (degenerate in 3D)
    println!("  Creating triangulation with collinear vertices...");

    let collinear_vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]), // Same line
        vertex!([2.0, 0.0, 0.0]), // Same line
        vertex!([3.0, 0.0, 0.0]), // Same line
    ];

    match Tds::<f64, Option<()>, Option<()>, 3>::new(&collinear_vertices) {
        Ok(tds) => {
            println!("  ⚠️  Triangulation created despite collinear vertices");
            println!(
                "    - {} vertices, {} cells",
                tds.number_of_vertices(),
                tds.number_of_cells()
            );

            // Test boundary analysis with this degenerate case
            match tds.boundary_facets() {
                Ok(facets) => {
                    println!(
                        "    - Boundary facets: {} (may be unexpected)",
                        facets.len()
                    );
                }
                Err(e) => {
                    println!("  ✅ Boundary analysis error caught: {e}");
                }
            }
        }
        Err(e) => {
            println!("  ✅ Triangulation creation error caught: {e}");
            println!("  💡 Recommendation: Validate vertices are not collinear/coplanar");
        }
    }

    // Test 2: Duplicate vertices
    println!("\n  Creating triangulation with duplicate vertices...");

    let duplicate_vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
        vertex!([0.0, 0.0, 0.0]), // Duplicate of first vertex
    ];

    match Tds::<f64, Option<()>, Option<()>, 3>::new(&duplicate_vertices) {
        Ok(tds) => {
            println!("  ⚠️  Triangulation created with duplicate vertices");
            println!(
                "    - Input: {} vertices, Triangulation: {} vertices",
                duplicate_vertices.len(),
                tds.number_of_vertices()
            );

            // The library should handle duplicates gracefully
            let boundary_count = tds.number_of_boundary_facets();
            println!("    - Boundary facets: {boundary_count}");
        }
        Err(e) => {
            println!("  ✅ Triangulation creation error caught: {e}");
            println!("  💡 Recommendation: Remove duplicate vertices before triangulation");
        }
    }
}

/// Test boundary analysis edge cases
fn demonstrate_boundary_analysis_edge_cases() {
    // Create a minimal valid triangulation to test edge cases
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    match Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices) {
        Ok(tds) => {
            println!("  Testing boundary analysis methods on minimal triangulation:");

            // Test 1: Boundary facets retrieval
            match tds.boundary_facets() {
                Ok(facets) => {
                    println!("  ✅ boundary_facets(): {} facets retrieved", facets.len());

                    // Test boundary checking for each facet
                    for (i, facet) in facets.iter().enumerate() {
                        let is_boundary = tds.is_boundary_facet(facet);
                        println!(
                            "    - Facet {}: {} boundary",
                            i,
                            if is_boundary { "IS" } else { "NOT" }
                        );

                        if !is_boundary {
                            println!("    ❌ Error: Retrieved facet is not marked as boundary!");
                        }
                    }
                }
                Err(e) => {
                    println!("  ❌ boundary_facets() failed: {e}");
                }
            }

            // Test 2: Count vs retrieval consistency
            let count_result = tds.number_of_boundary_facets();
            match tds.boundary_facets() {
                Ok(facets) => {
                    if count_result == facets.len() {
                        println!(
                            "  ✅ Count ({}) matches retrieval ({}) - consistent",
                            count_result,
                            facets.len()
                        );
                    } else {
                        println!(
                            "  ❌ Inconsistency: Count ({}) != Retrieval ({})",
                            count_result,
                            facets.len()
                        );
                    }
                }
                Err(e) => {
                    println!("  ⚠️  Cannot verify consistency due to retrieval error: {e}");
                }
            }

            // Test 3: Individual boundary checking for all cell facets
            println!("  Testing individual facet boundary checks:");
            let mut total_facets = 0;
            let mut boundary_facets = 0;

            for (cell_key, cell) in tds.cells() {
                match cell.facets() {
                    Ok(facets) => {
                        for facet in facets {
                            total_facets += 1;
                            if tds.is_boundary_facet(&facet) {
                                boundary_facets += 1;
                            }
                        }
                    }
                    Err(e) => {
                        println!(
                            "    ❌ Failed to get facets for cell {:?}: {}",
                            cell_key.data(),
                            e
                        );
                    }
                }
            }

            println!("    - Total facets checked: {total_facets}");
            println!("    - Boundary facets found: {boundary_facets}");

            if boundary_facets == count_result {
                println!("  ✅ Individual checks consistent with count method");
            } else {
                println!(
                    "  ❌ Individual checks ({boundary_facets}) != count method ({count_result})"
                );
            }
        }
        Err(e) => {
            println!("  ❌ Failed to create test triangulation: {e}");
        }
    }
}

/// Demonstrate error recovery and best practices
fn demonstrate_error_recovery_patterns() {
    println!("  🛡️  Recommended Error Recovery Patterns:");
    println!();

    // Pattern 1: Graceful degradation
    println!("  📋 Pattern 1: Graceful Degradation");

    let potentially_bad_vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    // Safe boundary analysis with fallbacks
    match Tds::<f64, Option<()>, Option<()>, 3>::new(&potentially_bad_vertices) {
        Ok(tds) => {
            // First try the efficient count method
            let boundary_count = tds.number_of_boundary_facets();
            println!("    ✅ Quick count: {boundary_count} boundary facets");

            // Then try full retrieval with error handling
            match tds.boundary_facets() {
                Ok(facets) => {
                    println!(
                        "    ✅ Full retrieval: {} facets with geometry",
                        facets.len()
                    );
                }
                Err(_) => {
                    println!("    ⚠️  Retrieval failed, using count-only result: {boundary_count}");
                }
            }
        }
        Err(e) => {
            println!("    ❌ Triangulation failed: {e}");
            println!("    💡 Fallback: Use alternative triangulation library or method");
        }
    }

    // Pattern 2: Input validation
    println!("\n  📋 Pattern 2: Input Validation");
    demonstrate_input_validation_pattern();

    // Pattern 3: Error categorization and specific handling
    println!("\n  📋 Pattern 3: Error Categorization");
    demonstrate_error_categorization_pattern();
}

/// Demonstrate input validation best practices
fn demonstrate_input_validation_pattern() {
    fn validate_vertices_for_3d(
        vertices: &[delaunay::core::vertex::Vertex<f64, Option<()>, 3>],
    ) -> Result<(), String> {
        // Check 1: Minimum vertex count
        if vertices.len() < 4 {
            return Err(format!(
                "Need at least 4 vertices for 3D triangulation, got {}",
                vertices.len()
            ));
        }

        // Check 2: Look for exact duplicates
        for i in 0..vertices.len() {
            for j in i + 1..vertices.len() {
                let coords_i: [f64; 3] = (&vertices[i]).into();
                let coords_j: [f64; 3] = (&vertices[j]).into();

                // Use approximate comparison to handle floating-point precision issues
                if relative_eq!(coords_i[0], coords_j[0], epsilon = f64::EPSILON)
                    && relative_eq!(coords_i[1], coords_j[1], epsilon = f64::EPSILON)
                    && relative_eq!(coords_i[2], coords_j[2], epsilon = f64::EPSILON)
                {
                    return Err(format!("Duplicate vertices found at indices {i} and {j}"));
                }
            }
        }

        // Check 3: Basic collinearity check for first 4 vertices
        if vertices.len() >= 4 {
            let coords: Vec<[f64; 3]> = vertices[0..4]
                .iter()
                .map(std::convert::Into::into)
                .collect();

            // Simple volume check - if first 4 vertices are coplanar, volume will be ~0
            let v1 = [
                coords[1][0] - coords[0][0],
                coords[1][1] - coords[0][1],
                coords[1][2] - coords[0][2],
            ];
            let v2 = [
                coords[2][0] - coords[0][0],
                coords[2][1] - coords[0][1],
                coords[2][2] - coords[0][2],
            ];
            let v3 = [
                coords[3][0] - coords[0][0],
                coords[3][1] - coords[0][1],
                coords[3][2] - coords[0][2],
            ];

            // Cross product v1 × v2
            let cross = [
                v1[1].mul_add(v2[2], -(v1[2] * v2[1])),
                v1[2].mul_add(v2[0], -(v1[0] * v2[2])),
                v1[0].mul_add(v2[1], -(v1[1] * v2[0])),
            ];

            // Dot product (v1 × v2) · v3 gives signed volume
            let volume = cross[2].mul_add(v3[2], cross[0].mul_add(v3[0], cross[1] * v3[1]));

            if volume.abs() < 1e-10 {
                return Err("First 4 vertices appear to be coplanar (degenerate)".to_string());
            }
        }

        Ok(())
    }

    let test_vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    match validate_vertices_for_3d(&test_vertices) {
        Ok(()) => {
            println!("    ✅ Input validation passed");
            match Tds::<f64, Option<()>, Option<()>, 3>::new(&test_vertices) {
                Ok(tds) => {
                    println!("    ✅ Triangulation created successfully");
                    println!("    - Boundary facets: {}", tds.number_of_boundary_facets());
                }
                Err(e) => println!("    ❌ Triangulation failed despite validation: {e}"),
            }
        }
        Err(validation_error) => {
            println!("    ❌ Input validation failed: {validation_error}");
            println!("    💡 Fix input data before attempting triangulation");
        }
    }
}

/// Demonstrate error categorization and specific handling
fn demonstrate_error_categorization_pattern() {
    println!("    Different error types require different handling strategies:");
    println!();
    println!("    🔴 Critical Errors (Stop execution):");
    println!("      - Empty vertex list");
    println!("      - Insufficient vertices for dimension");
    println!("      - Memory allocation failures");
    println!();
    println!("    🟡 Warning Conditions (Proceed with caution):");
    println!("      - Degenerate vertex configurations");
    println!("      - Numerical precision issues");
    println!("      - Large-scale triangulations with potential instability");
    println!();
    println!("    🟢 Recoverable Issues (Fallback strategies available):");
    println!("      - Individual facet computation failures");
    println!("      - Boundary retrieval errors (use count-only methods)");
    println!("      - Performance degradation (switch to approximate methods)");
    println!();
    println!("    💡 Best Practices:");
    println!("      1. Always validate input before expensive operations");
    println!("      2. Use count-only methods when full geometry isn't needed");
    println!("      3. Implement fallback strategies for non-critical operations");
    println!("      4. Log warnings for degenerate cases but continue processing");
    println!("      5. Consider using robust predicates for numerical stability");
}

// # Comprehensive Boundary Analysis Example
//
// This example demonstrates the `BoundaryAnalysis` trait for triangulation boundary operations
// and showcases practical real-world use cases. It shows how boundary analysis functions have
// been cleanly separated from the TDS struct using a trait-based approach.
//
// ## Features Demonstrated
//
// - **Multi-dimensional Support**: 2D edge extraction, 3D facet analysis, 4D+ support
// - **Real-world Use Cases**: Mesh vertex extraction, surface area calculation, performance comparison
// - **API Patterns**: Different methods for different performance needs
// - **Error Handling**: Robust error scenarios and edge cases
// - **Trait Benefits**: Modularity, testability, extensibility
//
// ## Usage
//
// ```bash
// cargo run --example boundary_analysis_trait
// ```

use delaunay::core::{
    traits::boundary_analysis::BoundaryAnalysis, triangulation_data_structure::Tds,
};
use delaunay::geometry::Coordinate;
use delaunay::geometry::util::safe_usize_to_scalar;
use delaunay::prelude::{Point, facet_measure, hypot, surface_measure};
use delaunay::vertex;
use std::time::Instant;

fn main() {
    println!("🔺 Comprehensive Boundary Analysis Demonstration\n");
    println!("=================================================\n");

    // Demonstrate trait API with simple examples
    demonstrate_basic_api();

    // Show different dimensions
    demonstrate_multi_dimensional_support();

    // Real-world use cases
    demonstrate_real_world_use_cases();

    // Performance comparison
    demonstrate_performance_characteristics();

    // Error handling
    demonstrate_error_handling();

    println!("\n🎯 Summary: The BoundaryAnalysis trait provides:");
    println!("  • Consistent API across all dimensions (2D, 3D, 4D+)");
    println!("  • Multiple methods optimized for different use cases");
    println!("  • Clean separation from triangulation data structure");
    println!("  • Easy extensibility for custom boundary algorithms");
    println!("  • Robust error handling for edge cases");
    println!("\n=================================================\n");
    println!("✅ Example completed successfully!");
}

/// Demonstrate basic `BoundaryAnalysis` trait API usage
fn demonstrate_basic_api() {
    println!("📚 1. Basic BoundaryAnalysis API Demonstration");
    println!("==============================================\n");

    // Create a simple 3D triangulation (tetrahedron)
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    let tds: Tds<f64, Option<()>, Option<()>, 3> =
        Tds::new(&vertices).expect("Failed to create triangulation");

    println!("Created triangulation:");
    println!("  - {} vertices", tds.number_of_vertices());
    println!("  - {} cells", tds.number_of_cells());
    println!("  - dimension: {}", tds.dim());

    println!("\n📊 Boundary Analysis Methods:");

    // Method 1: Get all boundary facets (when you need the actual facet data)
    let start = Instant::now();
    let boundary_facets = tds
        .boundary_facets()
        .expect("Failed to get boundary facets");
    let facets_time = start.elapsed();
    println!(
        "  ✓ boundary_facets(): {} facets ({:?})",
        boundary_facets.len(),
        facets_time
    );

    // Method 2: Count boundary facets efficiently (when you only need the count)
    let start = Instant::now();
    let boundary_count = tds.number_of_boundary_facets();
    let count_time = start.elapsed();
    println!("  ✓ number_of_boundary_facets(): {boundary_count} facets ({count_time:?})");

    // Method 3: Check individual facets (for selective testing)
    if let Some(cell) = tds.cells().values().next() {
        let facets = cell.facets().expect("Failed to get facets from cell");
        println!("  ✓ is_boundary_facet() tests:");

        for (i, facet) in facets.iter().enumerate() {
            let start = Instant::now();
            let is_boundary = tds.is_boundary_facet(facet);
            let test_time = start.elapsed();
            println!(
                "    • Facet {}: {} boundary ({:?})",
                i,
                if is_boundary { "IS" } else { "NOT" },
                test_time
            );
        }
    }

    // Complex triangulation example
    println!("\n🔸 Complex Triangulation (2 Tetrahedra):");
    let complex_vertices = vec![
        vertex!([0.0, 0.0, 0.0]),  // A
        vertex!([1.0, 0.0, 0.0]),  // B
        vertex!([0.5, 1.0, 0.0]),  // C - forms base triangle ABC
        vertex!([0.5, 0.5, 1.0]),  // D - above base
        vertex!([0.5, 0.5, -1.0]), // E - below base
    ];

    let complex_tds: Tds<f64, Option<()>, Option<()>, 3> =
        Tds::new(&complex_vertices).expect("Failed to create complex triangulation");

    let complex_boundary_count = complex_tds.number_of_boundary_facets();
    println!(
        "  - {} vertices → {} cells → {} boundary facets",
        complex_tds.number_of_vertices(),
        complex_tds.number_of_cells(),
        complex_boundary_count
    );
    println!("  - Expected: 6 boundary facets (2 tetrahedra × 4 faces - 2 shared)");

    println!();
}
