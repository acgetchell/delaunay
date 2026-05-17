//! # Topology Editing Example
//!
//! This example demonstrates the **two-track API design** for working with
//! Delaunay triangulations in both 2D and 3D:
//!
//! 1. **Builder API** - High-level construction and maintenance
//! 2. **Edit API** - Low-level topology editing via bistellar flips
//!
//! The example shows:
//! - Building triangulations with the Builder API
//! - Manual topology editing with the Edit API (k=1, k=2, k=3 flips)
//! - The difference in Delaunay property preservation between the two APIs
//! - Dimension-specific flip operations (k=3 is only available in 3D+)
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example topology_editing
//! ```

#![expect(
    clippy::result_large_err,
    reason = "example preserves the crate's typed insertion and flip errors instead of erasing them"
)]

use delaunay::prelude::geometry::{
    CircumcenterError, Coordinate, Kernel, Point, circumcenter, hypot,
};
use delaunay::prelude::triangulation::construction::{
    DelaunayTriangulation, DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
    vertex,
};
use delaunay::prelude::triangulation::flips::*;
use delaunay::prelude::triangulation::insertion::InsertionError;
use delaunay::prelude::triangulation::validation::DelaunayTriangulationValidationError;
use delaunay::prelude::{TdsError, VertexKey};

type ExampleResult<T = ()> = Result<T, TopologyEditingExampleError>;

#[derive(Debug, thiserror::Error)]
enum TopologyEditingExampleError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Validation(#[from] DelaunayTriangulationValidationError),
    #[error(transparent)]
    Insertion(#[from] InsertionError),
    #[error(transparent)]
    Flip(#[from] FlipError),
    #[error(transparent)]
    Tds(#[from] TdsError),
    #[error(transparent)]
    Circumcenter(#[from] CircumcenterError),
    #[error("{demo} triangulation has no simplices")]
    EmptyTriangulation { demo: &'static str },
    #[error("{demo} simplex key was not found")]
    MissingSimplex { demo: &'static str },
    #[error("{demo} vertex key {vertex_key:?} was not found")]
    MissingVertex {
        demo: &'static str,
        vertex_key: VertexKey,
    },
    #[error("{demo} has no interior facet")]
    NoInteriorFacet { demo: &'static str },
}

fn main() -> ExampleResult {
    println!("============================================================");
    println!("Topology Editing: Builder API vs Edit API (2D and 3D)");
    println!("============================================================\n");

    // Part 1: 2D examples (k=1 and k=2 flips)
    demo_2d()?;

    println!("\n############################################################\n");

    // Part 2: 3D examples (k=1, k=2, and k=3 flips)
    demo_3d()?;

    println!("\n============================================================");
    println!("Example complete!");
    println!("============================================================");
    Ok(())
}

// ============================================================================
// 2D DEMONSTRATIONS
// ============================================================================

fn demo_2d() -> ExampleResult {
    println!("PART 1: 2D TRIANGULATION");
    println!("============================================================\n");

    builder_api_2d()?;
    println!("\n------------------------------------------------------------\n");
    edit_api_2d_k1()?;
    println!("\n------------------------------------------------------------\n");
    edit_api_2d_k2()?;
    Ok(())
}

/// Demonstrates the Builder API in 2D.
fn builder_api_2d() -> ExampleResult {
    println!("2D Builder API: Automatic Delaunay Preservation");
    println!("------------------------------------------------\n");

    // Build initial triangulation
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([4.0, 0.0]),
        vertex!([2.0, 3.0]),
    ];

    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

    println!("Initial triangle:");
    print_stats_2d(&dt);
    dt.validate()?;
    println!("  ✓ Delaunay property verified\n");

    // Insert vertices using Builder API
    println!("Inserting 3 vertices using Builder API:");
    let new_vertices = vec![
        vertex!([2.0, 1.0]),
        vertex!([1.0, 1.5]),
        vertex!([3.0, 1.5]),
    ];

    for (i, v) in new_vertices.into_iter().enumerate() {
        dt.insert(v)?;
        println!(
            "  After insert {}: {} vertices, {} simplices",
            i + 1,
            dt.number_of_vertices(),
            dt.number_of_simplices()
        );
    }

    // Verify Delaunay property is maintained
    dt.validate()?;
    println!("\n✓ Builder API automatically maintained Delaunay property");
    Ok(())
}

/// Demonstrates k=1 flips (simplex split/merge) in 2D.
fn edit_api_2d_k1() -> ExampleResult {
    println!("2D Edit API: k=1 Flips (Simplex Split/Merge)");
    println!("------------------------------------------\n");

    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([3.0, 0.0]),
        vertex!([1.5, 2.5]),
    ];

    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

    println!("Initial triangle:");
    print_stats_2d(&dt);

    // Apply k=1 flip (insert vertex into simplex)
    let simplex_key = dt
        .simplices()
        .next()
        .map(|(simplex_key, _)| simplex_key)
        .ok_or(TopologyEditingExampleError::EmptyTriangulation {
            demo: "2D k=1 demo",
        })?;
    let simplex =
        dt.tds()
            .simplex(simplex_key)
            .ok_or(TopologyEditingExampleError::MissingSimplex {
                demo: "2D k=1 demo",
            })?;
    let vertex_points: Vec<Point<f64, 2>> = simplex
        .vertices()
        .iter()
        .map(|vkey| {
            dt.tds().vertex(*vkey).map(|vertex| *vertex.point()).ok_or(
                TopologyEditingExampleError::MissingVertex {
                    demo: "2D k=1 demo",
                    vertex_key: *vkey,
                },
            )
        })
        .collect::<Result<_, _>>()?;
    let circumcenter = circumcenter(&vertex_points)?;
    let circumcenter_coords = circumcenter.to_array();
    let distances: Vec<f64> = vertex_points
        .iter()
        .map(|point| {
            let coords = point.to_array();
            let diff = [
                coords[0] - circumcenter_coords[0],
                coords[1] - circumcenter_coords[1],
            ];
            hypot(&diff)
        })
        .collect();
    println!("  Circumcenter distances to vertices: {distances:?}");

    println!(
        "\nApplying k=1 flip (insert vertex at circumcenter [{:.2}, {:.2}]):",
        circumcenter_coords[0], circumcenter_coords[1]
    );

    let flip_info = dt.flip_k1_insert(simplex_key, vertex!(circumcenter_coords))?;

    println!("After k=1 forward:");
    print_stats_2d(&dt);
    println!("  Removed: {} simplices", flip_info.removed_simplices.len());
    println!("  Inserted: {} simplices", flip_info.new_simplices.len());
    println!("  New vertex: {:?}", flip_info.inserted_face_vertices);

    // Verify structural validity (always maintained)
    dt.tds().is_valid()?;
    println!("  ✓ Structural invariants preserved");

    // Apply inverse k=1 flip (remove vertex)
    println!("\nApplying k=1 inverse (remove vertex):");
    let vertex_to_remove = flip_info.inserted_face_vertices[0];
    dt.flip_k1_remove(vertex_to_remove)?;

    println!("After k=1 inverse:");
    print_stats_2d(&dt);

    // Verify we're back to original state
    dt.validate()?;
    println!("\n✓ k=1 flip roundtrip successful (Edit API)");
    Ok(())
}

/// Demonstrates k=2 flips (edge flip) in 2D.
fn edit_api_2d_k2() -> ExampleResult {
    println!("2D Edit API: k=2 Flips (Edge Flip)");
    println!("-----------------------------------\n");

    // Create a square with diagonal (2 triangles)
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([2.0, 0.0]),
        vertex!([2.0, 2.0]),
        vertex!([0.0, 2.0]),
    ];

    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

    println!("Initial square (2 triangles):");
    print_stats_2d(&dt);

    let initial_valid = dt.is_valid().is_ok();
    println!(
        "  Initial Delaunay: {}",
        if initial_valid { "✓" } else { "⚠️" }
    );

    // Find an interior edge to flip
    let facet =
        find_interior_facet_2d(&dt).ok_or(TopologyEditingExampleError::NoInteriorFacet {
            demo: "2D k=2 demo",
        })?;

    println!("\nApplying k=2 flip (flipping diagonal edge):");
    let flip_info = dt.flip_k2(facet)?;

    println!("After k=2 forward:");
    print_stats_2d(&dt);
    println!("  Removed: {} simplices", flip_info.removed_simplices.len());
    println!("  Inserted: {} simplices", flip_info.new_simplices.len());

    // Check if Delaunay property changed
    let after_valid = dt.is_valid().is_ok();
    println!(
        "  Delaunay after flip: {}",
        if after_valid { "✓" } else { "⚠️" }
    );

    if initial_valid != after_valid {
        println!("  Note: Delaunay property changed (expected with Edit API)");
    }

    // Note: k=2 inverse is only available in 3D+
    println!("\nNote: k=2 inverse flip (from edge star) is only available in 3D+");
    println!("      In 2D, k=2 flips are always reversible by another k=2 flip");

    println!("\n✓ k=2 flip successful (Edit API)");
    Ok(())
}

// ============================================================================
// 3D DEMONSTRATIONS
// ============================================================================

fn demo_3d() -> ExampleResult {
    println!("PART 2: 3D TRIANGULATION");
    println!("============================================================\n");

    builder_api_3d()?;
    println!("\n------------------------------------------------------------\n");
    edit_api_3d_k1()?;
    println!("\n------------------------------------------------------------\n");
    edit_api_3d_k2()?;
    println!("\n------------------------------------------------------------\n");
    edit_api_3d_k3()?;
    Ok(())
}

/// Demonstrates the Builder API in 3D.
fn builder_api_3d() -> ExampleResult {
    println!("3D Builder API: Automatic Delaunay Preservation");
    println!("------------------------------------------------\n");

    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([2.0, 0.0, 0.0]),
        vertex!([1.0, 2.0, 0.0]),
        vertex!([1.0, 0.5, 1.5]),
    ];

    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

    println!("Initial tetrahedron:");
    print_stats_3d(&dt);
    dt.validate()?;
    println!("  ✓ Delaunay property verified\n");

    // Insert vertices using Builder API
    println!("Inserting 2 vertices using Builder API:");
    let new_vertices = vec![vertex!([1.0, 0.5, 0.5]), vertex!([0.8, 0.8, 0.8])];

    for (i, v) in new_vertices.into_iter().enumerate() {
        dt.insert(v)?;
        println!(
            "  After insert {}: {} vertices, {} simplices",
            i + 1,
            dt.number_of_vertices(),
            dt.number_of_simplices()
        );
    }

    dt.validate()?;
    println!("\n✓ Builder API automatically maintained Delaunay property");
    Ok(())
}

/// Demonstrates k=1 flips in 3D.
fn edit_api_3d_k1() -> ExampleResult {
    println!("3D Edit API: k=1 Flips (Simplex Split/Merge)");
    println!("------------------------------------------\n");

    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([2.0, 0.0, 0.0]),
        vertex!([1.0, 2.0, 0.0]),
        vertex!([1.0, 0.5, 1.5]),
    ];

    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

    println!("Initial tetrahedron:");
    print_stats_3d(&dt);

    // Apply k=1 flip
    let simplex_key = dt
        .simplices()
        .next()
        .map(|(simplex_key, _)| simplex_key)
        .ok_or(TopologyEditingExampleError::EmptyTriangulation {
            demo: "3D k=1 demo",
        })?;
    let simplex =
        dt.tds()
            .simplex(simplex_key)
            .ok_or(TopologyEditingExampleError::MissingSimplex {
                demo: "3D k=1 demo",
            })?;
    let vertex_points: Vec<Point<f64, 3>> = simplex
        .vertices()
        .iter()
        .map(|vkey| {
            dt.tds().vertex(*vkey).map(|vertex| *vertex.point()).ok_or(
                TopologyEditingExampleError::MissingVertex {
                    demo: "3D k=1 demo",
                    vertex_key: *vkey,
                },
            )
        })
        .collect::<Result<_, _>>()?;
    let circumcenter = circumcenter(&vertex_points)?;
    let circumcenter_coords = circumcenter.to_array();
    let distances: Vec<f64> = vertex_points
        .iter()
        .map(|point| {
            let coords = point.to_array();
            let diff = [
                coords[0] - circumcenter_coords[0],
                coords[1] - circumcenter_coords[1],
                coords[2] - circumcenter_coords[2],
            ];
            hypot(&diff)
        })
        .collect();
    println!("  Circumcenter distances to vertices: {distances:?}");

    println!(
        "\nApplying k=1 flip (split tetrahedron at circumcenter [{:.2}, {:.2}, {:.2}]):",
        circumcenter_coords[0], circumcenter_coords[1], circumcenter_coords[2]
    );
    let flip_info = dt.flip_k1_insert(simplex_key, vertex!(circumcenter_coords))?;

    println!("After k=1 forward:");
    print_stats_3d(&dt);
    println!(
        "  1 tetrahedron → {} tetrahedra",
        flip_info.new_simplices.len()
    );

    // Apply inverse
    println!("\nApplying k=1 inverse:");
    let vertex_to_remove = flip_info.inserted_face_vertices[0];
    dt.flip_k1_remove(vertex_to_remove)?;

    println!("After k=1 inverse:");
    print_stats_3d(&dt);

    println!("\n✓ k=1 flip roundtrip successful in 3D");
    Ok(())
}

/// Demonstrates k=2 flips (facet flip) in 3D.
fn edit_api_3d_k2() -> ExampleResult {
    println!("3D Edit API: k=2 Flips (Facet Flip: 2↔3)");
    println!("-----------------------------------------\n");

    println!("Note: k=2 flips in 3D replace 2 tetrahedra with 3 (and vice versa)");
    println!("      This demonstrates the inverse flip operation available in 3D+\n");

    // Build a simple regular tetrahedron and add one interior point
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([2.0, 0.0, 0.0]),
        vertex!([1.0, 1.8, 0.0]),
        vertex!([1.0, 0.6, 1.6]),
        vertex!([1.0, 0.6, 0.4]), // Interior point
    ];

    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

    println!("Triangulation with interior point:");
    print_stats_3d(&dt);

    // Find an interior facet to flip
    if let Some(facet) = find_interior_facet_3d(&dt) {
        match dt.flip_k2(facet) {
            Ok(flip_info) => {
                println!("\nApplied k=2 flip:");
                print_stats_3d(&dt);
                println!("  Removed: {} simplices", flip_info.removed_simplices.len());
                println!("  Inserted: {} simplices", flip_info.new_simplices.len());

                // Try inverse
                println!("\nApplying k=2 inverse:");
                let edge = EdgeKey::new(
                    flip_info.inserted_face_vertices[0],
                    flip_info.inserted_face_vertices[1],
                );

                match dt.flip_k2_inverse_from_edge(edge) {
                    Ok(_) => {
                        println!("After k=2 inverse:");
                        print_stats_3d(&dt);
                        println!("\n✓ k=2 flip roundtrip successful in 3D");
                    }
                    Err(e) => {
                        println!("⚠️  k=2 inverse failed: {e}");
                        println!(
                            "   (This demonstrates that not all configurations support inverse flips)"
                        );
                    }
                }
            }
            Err(e) => {
                println!("⚠️  k=2 flip not possible: {e}");
                println!("   (This demonstrates geometric constraints on flip operations)");
            }
        }
    } else {
        println!("⚠️  No interior facet found for k=2 flip demo");
    }

    Ok(())
}

/// Demonstrates k=3 flips (ridge flip) in 3D.
fn edit_api_3d_k3() -> ExampleResult {
    println!("3D Edit API: k=3 Flips (Ridge Flip: 3↔2)");
    println!("-----------------------------------------\n");
    println!("Note: k=3 flips are only available in 3D and higher dimensions\n");

    // k=3 flips are more complex and have strict geometric requirements
    // For this demo, we'll show that the API is available but the geometric
    // conditions may not always be satisfied

    println!("k=3 flips replace 3 tetrahedra around a ridge (edge) with");
    println!("2 tetrahedra around a triangle (and vice versa).\n");

    println!("This operation has strict geometric requirements:");
    println!("  - The ridge must be surrounded by exactly 3 tetrahedra");
    println!("  - The resulting configuration must not be degenerate");
    println!("  - The flip must preserve manifold topology\n");

    println!("Building a simple 3D triangulation...");
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([2.0, 0.0, 0.0]),
        vertex!([1.0, 1.7, 0.0]),
        vertex!([1.0, 0.6, 1.4]),
    ];

    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    print_stats_3d(&dt);

    // Try to find and flip a ridge
    if let Some(ridge) = find_flippable_ridge_3d(&dt) {
        match dt.flip_k3(ridge) {
            Ok(flip_info) => {
                println!("\n✓ k=3 flip succeeded:");
                print_stats_3d(&dt);
                println!("  Removed: {} simplices", flip_info.removed_simplices.len());
                println!("  Inserted: {} simplices", flip_info.new_simplices.len());
            }
            Err(e) => {
                println!("\n⚠️  k=3 flip not applicable: {e}");
                println!("   (Geometric constraints not satisfied for this configuration)");
            }
        }
    } else {
        println!("\n⚠️  No ridges found in this simple triangulation");
    }

    println!("\n✓ k=3 flip API demonstrated (Edit API - 3D+ only)");
    Ok(())
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

fn print_stats_2d<K: Kernel<2>>(dt: &DelaunayTriangulation<K, (), (), 2>) {
    println!(
        "  Vertices: {}, Triangles: {}",
        dt.number_of_vertices(),
        dt.number_of_simplices()
    );
}

fn print_stats_3d<K: Kernel<3>>(dt: &DelaunayTriangulation<K, (), (), 3>) {
    println!(
        "  Vertices: {}, Tetrahedra: {}",
        dt.number_of_vertices(),
        dt.number_of_simplices()
    );
}

fn find_interior_facet_2d<K: Kernel<2>>(
    dt: &DelaunayTriangulation<K, (), (), 2>,
) -> Option<FacetHandle> {
    for (simplex_key, simplex) in dt.simplices() {
        if let Some(neighbors) = simplex.neighbors() {
            for (facet_idx, neighbor) in neighbors.enumerate() {
                if neighbor.is_some() {
                    let Ok(facet_idx) = u8::try_from(facet_idx) else {
                        continue;
                    };
                    return Some(FacetHandle::new(simplex_key, facet_idx));
                }
            }
        }
    }
    None
}

fn find_interior_facet_3d<K: Kernel<3>>(
    dt: &DelaunayTriangulation<K, (), (), 3>,
) -> Option<FacetHandle> {
    for (simplex_key, simplex) in dt.simplices() {
        if let Some(neighbors) = simplex.neighbors() {
            for (facet_idx, neighbor) in neighbors.enumerate() {
                if neighbor.is_some() {
                    let Ok(facet_idx) = u8::try_from(facet_idx) else {
                        continue;
                    };
                    return Some(FacetHandle::new(simplex_key, facet_idx));
                }
            }
        }
    }
    None
}

fn find_flippable_ridge_3d<K: Kernel<3>>(
    dt: &DelaunayTriangulation<K, (), (), 3>,
) -> Option<RidgeHandle> {
    // Try to find any ridge (edge in 3D shared by multiple tetrahedra)
    for (simplex_key, simplex) in dt.simplices() {
        let vertex_count = simplex.number_of_vertices();
        // Try each pair of vertices (each defines a ridge)
        for i in 0..vertex_count {
            if i + 1 >= vertex_count {
                continue;
            }
            let Ok(omit_a) = u8::try_from(i) else {
                continue;
            };
            let Ok(omit_b) = u8::try_from(i + 1) else {
                continue;
            };
            let ridge = RidgeHandle::new(simplex_key, omit_a, omit_b);

            // Just return the first one we find
            // (In practice, you'd want to check if it's actually flippable)
            return Some(ridge);
        }
    }
    None
}
