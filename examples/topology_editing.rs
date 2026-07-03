//! # Topology Editing Example
//!
//! This example demonstrates the **two-track API design** for working with
//! Delaunay triangulations in both 2D and 3D:
//!
//! 1. **Builder API** - High-level construction and maintenance
//! 2. **Pachner Move API** - Local topology editing via Pachner moves
//!
//! The example shows:
//! - Building triangulations with the Builder API
//! - Manual topology editing with the Pachner Move API (k=1, k=2, k=3 moves)
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

use delaunay::prelude::construction::{
    DelaunayTriangulation, DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
    vertex,
};
use delaunay::prelude::geometry::{
    AdaptiveKernel, CircumcenterError, Coordinate, CoordinateConversionError, Point, circumcenter,
    hypot,
};
use delaunay::prelude::insertion::InsertionError;
use delaunay::prelude::pachner::{
    EdgeKey, EdgeKeyError, FacetError, FacetHandle, FlipError, PachnerMove, PachnerMoves,
    RidgeHandle, TriangleHandle, TriangleHandleError, Vertex, VertexKey,
};
use delaunay::prelude::tds::{InvariantError, TdsError};
use delaunay::prelude::validation::DelaunayTriangulationValidationError;

type ExampleResult<T = ()> = Result<T, TopologyEditingExampleError>;
type Dt3 = DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 3>;

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
    Edge(#[from] EdgeKeyError),
    #[error(transparent)]
    Triangle(#[from] TriangleHandleError),
    #[error(transparent)]
    Facet(#[from] FacetError),
    #[error(transparent)]
    Tds(#[from] TdsError),
    #[error(transparent)]
    Invariant(#[from] InvariantError),
    #[error(transparent)]
    Circumcenter(#[from] CircumcenterError),
    #[error(transparent)]
    CoordinateConversion(#[from] CoordinateConversionError),
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
    #[error("{demo} has no roundtrip-capable k=2 facet")]
    NoRoundtripK2Facet { demo: &'static str },
    #[error("{demo} has no flippable k=3 ridge")]
    NoFlippableRidge { demo: &'static str },
    #[error("{demo} expected {expected} inserted-face vertices, got {actual}")]
    UnexpectedInsertedFaceVertices {
        demo: &'static str,
        expected: usize,
        actual: usize,
    },
}

fn main() -> ExampleResult {
    println!("============================================================");
    println!("Topology Editing: Builder API vs Pachner Move API (2D and 3D)");
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
    pachner_2d_k1()?;
    println!("\n------------------------------------------------------------\n");
    pachner_2d_k2()?;
    Ok(())
}

/// Demonstrates the Builder API in 2D.
fn builder_api_2d() -> ExampleResult {
    println!("2D Builder API: Automatic Delaunay Preservation");
    println!("------------------------------------------------\n");

    // Build initial triangulation
    let vertices = vec![vertex![0.0, 0.0]?, vertex![4.0, 0.0]?, vertex![2.0, 3.0]?];

    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build()?;

    println!("Initial triangle:");
    print_stats_2d(&dt);
    dt.validate()?;
    println!("  ✓ Delaunay property verified\n");

    // Insert vertices using Builder API
    println!("Inserting 3 vertices using Builder API:");
    let new_vertices = vec![vertex![2.0, 1.0]?, vertex![1.0, 1.5]?, vertex![3.0, 1.5]?];

    for (i, v) in new_vertices.into_iter().enumerate() {
        dt.insert_vertex(v)?;
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
fn pachner_2d_k1() -> ExampleResult {
    println!("2D Pachner Move API: k=1 Moves (Simplex Split/Merge)");
    println!("------------------------------------------\n");

    let vertices = vec![vertex![0.0, 0.0]?, vertex![3.0, 0.0]?, vertex![1.5, 2.5]?];

    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build()?;

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
    let vertex_points: Vec<Point<2>> = simplex
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

    let flip_info = dt
        .propose_pachner(PachnerMove::K1Insert {
            simplex_key,
            vertex: vertex!(circumcenter_coords)?,
        })?
        .attempt_on(&mut dt)?;

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
    let inverse_info = dt
        .propose_pachner(PachnerMove::K1Remove {
            vertex_key: vertex_to_remove,
        })?
        .attempt_on(&mut dt)?;
    assert!(!inverse_info.removed_simplices.is_empty());

    println!("After k=1 inverse:");
    print_stats_2d(&dt);

    // Verify we're back to original state
    dt.validate()?;
    println!("\n✓ k=1 move roundtrip successful (Pachner Move API)");
    Ok(())
}

/// Demonstrates k=2 flips (edge flip) in 2D.
fn pachner_2d_k2() -> ExampleResult {
    println!("2D Pachner Move API: k=2 Moves (Edge Flip)");
    println!("-----------------------------------\n");

    // Create a square with diagonal (2 triangles)
    let vertices = vec![
        vertex![0.0, 0.0]?,
        vertex![2.0, 0.0]?,
        vertex![2.0, 2.0]?,
        vertex![0.0, 2.0]?,
    ];

    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build()?;

    println!("Initial square (2 triangles):");
    print_stats_2d(&dt);

    let initial_valid = dt.is_valid_delaunay().is_ok();
    println!(
        "  Initial Delaunay: {}",
        if initial_valid { "✓" } else { "⚠️" }
    );

    // Find an interior edge to flip
    let facet = find_interior_facet(&dt)?.ok_or(TopologyEditingExampleError::NoInteriorFacet {
        demo: "2D k=2 demo",
    })?;

    println!("\nApplying k=2 flip (flipping diagonal edge):");
    let flip_info = dt
        .propose_pachner(PachnerMove::K2 { facet })?
        .attempt_on(&mut dt)?;

    println!("After k=2 forward:");
    print_stats_2d(&dt);
    println!("  Removed: {} simplices", flip_info.removed_simplices.len());
    println!("  Inserted: {} simplices", flip_info.new_simplices.len());

    // Check if Delaunay property changed
    let after_valid = dt.is_valid_delaunay().is_ok();
    println!(
        "  Delaunay after flip: {}",
        if after_valid { "✓" } else { "⚠️" }
    );

    if initial_valid != after_valid {
        println!("  Note: Delaunay property changed (expected with Pachner Move API)");
    }

    // Note: k=2 inverse is only available in 3D+
    println!("\nNote: k=2 inverse flip (from edge star) is only available in 3D+");
    println!("      In 2D, k=2 flips are always reversible by another k=2 flip");

    println!("\n✓ k=2 move successful (Pachner Move API)");
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
    pachner_3d_k1()?;
    println!("\n------------------------------------------------------------\n");
    pachner_3d_k2()?;
    println!("\n------------------------------------------------------------\n");
    pachner_3d_k3()?;
    Ok(())
}

/// Demonstrates the Builder API in 3D.
fn builder_api_3d() -> ExampleResult {
    println!("3D Builder API: Automatic Delaunay Preservation");
    println!("------------------------------------------------\n");

    let vertices = vec![
        vertex![0.0, 0.0, 0.0]?,
        vertex![2.0, 0.0, 0.0]?,
        vertex![1.0, 2.0, 0.0]?,
        vertex![1.0, 0.5, 1.5]?,
    ];

    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build()?;

    println!("Initial tetrahedron:");
    print_stats_3d(&dt);
    dt.validate()?;
    println!("  ✓ Delaunay property verified\n");

    // Insert vertices using Builder API
    println!("Inserting 2 vertices using Builder API:");
    let new_vertices = vec![vertex![1.0, 0.5, 0.5]?, vertex![1.0, 0.9, 0.8]?];

    for (i, v) in new_vertices.into_iter().enumerate() {
        dt.insert_vertex(v)?;
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
fn pachner_3d_k1() -> ExampleResult {
    println!("3D Pachner Move API: k=1 Moves (Simplex Split/Merge)");
    println!("------------------------------------------\n");

    let vertices = vec![
        vertex![0.0, 0.0, 0.0]?,
        vertex![2.0, 0.0, 0.0]?,
        vertex![1.0, 2.0, 0.0]?,
        vertex![1.0, 0.5, 1.5]?,
    ];

    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build()?;

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
    let vertex_points: Vec<Point<3>> = simplex
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
    let flip_info = dt
        .propose_pachner(PachnerMove::K1Insert {
            simplex_key,
            vertex: vertex!(circumcenter_coords)?,
        })?
        .attempt_on(&mut dt)?;

    println!("After k=1 forward:");
    print_stats_3d(&dt);
    println!(
        "  1 tetrahedron → {} tetrahedra",
        flip_info.new_simplices.len()
    );

    // Apply inverse
    println!("\nApplying k=1 inverse:");
    let vertex_to_remove = flip_info.inserted_face_vertices[0];
    let inverse_info = dt
        .propose_pachner(PachnerMove::K1Remove {
            vertex_key: vertex_to_remove,
        })?
        .attempt_on(&mut dt)?;
    assert!(!inverse_info.removed_simplices.is_empty());

    println!("After k=1 inverse:");
    print_stats_3d(&dt);

    println!("\n✓ k=1 flip roundtrip successful in 3D");
    Ok(())
}

/// Demonstrates k=2 flips (facet flip) in 3D.
fn pachner_3d_k2() -> ExampleResult {
    println!("3D Pachner Move API: k=2 Moves (Facet Flip: 2↔3)");
    println!("-----------------------------------------\n");

    println!("Note: k=2 flips in 3D replace 2 tetrahedra with 3 (and vice versa)");
    println!("      This demonstrates the inverse flip operation available in 3D+\n");

    let mut dt = build_stable_pachner_dt_3d()?;

    println!("Stable 3D fixture:");
    print_stats_3d(&dt);

    let facet = find_roundtrip_k2_facet_3d(&dt)?.ok_or(
        TopologyEditingExampleError::NoRoundtripK2Facet {
            demo: "3D k=2 demo",
        },
    )?;
    let flip_info = dt
        .propose_pachner(PachnerMove::K2 { facet })?
        .attempt_on(&mut dt)?;
    println!("\nApplied k=2 flip:");
    print_stats_3d(&dt);
    println!("  Removed: {} simplices", flip_info.removed_simplices.len());
    println!("  Inserted: {} simplices", flip_info.new_simplices.len());

    println!("\nApplying k=2 inverse:");
    let edge = inserted_edge_3d(&dt, &flip_info.inserted_face_vertices, "3D k=2 demo")?;
    let inverse_info = dt
        .propose_pachner(PachnerMove::K2Inverse { edge })?
        .attempt_on(&mut dt)?;
    assert!(!inverse_info.removed_simplices.is_empty());
    dt.validate()?;
    println!("After k=2 inverse:");
    print_stats_3d(&dt);
    println!("\n✓ k=2 flip roundtrip successful in 3D");

    Ok(())
}

/// Demonstrates k=3 flips (ridge flip) in 3D.
fn pachner_3d_k3() -> ExampleResult {
    println!("3D Pachner Move API: k=3 Moves (Ridge Flip: 3↔2)");
    println!("-----------------------------------------\n");
    println!("Note: k=3 flips are only available in 3D and higher dimensions\n");

    println!("k=3 flips replace 3 tetrahedra around a ridge (edge) with");
    println!("2 tetrahedra around a triangle (and vice versa).\n");

    println!("This operation has strict geometric requirements, so the example");
    println!("searches a stable fixture for an accepted ridge before mutating:");
    println!("  - The ridge must be surrounded by exactly 3 tetrahedra");
    println!("  - The resulting configuration must not be degenerate");
    println!("  - The flip must preserve manifold topology\n");

    println!("Building a stable 3D fixture...");
    let mut dt = build_stable_pachner_dt_3d()?;
    print_stats_3d(&dt);

    let ridge =
        find_flippable_ridge_3d(&dt)?.ok_or(TopologyEditingExampleError::NoFlippableRidge {
            demo: "3D k=3 demo",
        })?;
    let flip_info = dt
        .propose_pachner(PachnerMove::K3 { ridge })?
        .attempt_on(&mut dt)?;
    dt.as_triangulation().validate()?;
    let triangle = inserted_triangle(&flip_info.inserted_face_vertices, "3D k=3 demo")?;

    println!("\n✓ k=3 flip succeeded:");
    print_stats_3d(&dt);
    println!("  Removed: {} simplices", flip_info.removed_simplices.len());
    println!("  Inserted: {} simplices", flip_info.new_simplices.len());
    println!("  Inverse candidate triangle: {triangle:?}");
    println!("\n✓ k=3 move API demonstrated (Pachner Move API - 3D+ only)");
    Ok(())
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

fn print_stats_2d<K>(dt: &DelaunayTriangulation<K, (), (), 2>) {
    println!(
        "  Vertices: {}, Triangles: {}",
        dt.number_of_vertices(),
        dt.number_of_simplices()
    );
}

fn print_stats_3d<K>(dt: &DelaunayTriangulation<K, (), (), 3>) {
    println!(
        "  Vertices: {}, Tetrahedra: {}",
        dt.number_of_vertices(),
        dt.number_of_simplices()
    );
}

/// Builds the stable 3D local-edit fixture used by the successful k=2/k=3 demos.
fn build_stable_pachner_dt_3d() -> ExampleResult<Dt3> {
    let vertices = stable_pachner_vertices_3d()?;
    Ok(DelaunayTriangulationBuilder::new(&vertices).build()?)
}

/// Returns the stable 3D point set used to find accepted public Pachner moves.
fn stable_pachner_vertices_3d() -> Result<Vec<Vertex<(), 3>>, CoordinateConversionError> {
    Ok(vec![
        vertex![0.0, 0.0, 0.0]?,
        vertex![1.0, 0.0, 0.0]?,
        vertex![0.0, 1.0, 0.0]?,
        vertex![0.0, 0.0, 1.0]?,
        vertex![0.20, 0.20, 0.20]?,
        vertex![0.75, 0.15, 0.30]?,
        vertex![0.20, 0.70, 0.35]?,
        vertex![0.30, 0.25, 0.80]?,
        vertex![0.65, 0.60, 0.55]?,
    ])
}

fn find_interior_facet<K, const D: usize>(
    dt: &DelaunayTriangulation<K, (), (), D>,
) -> ExampleResult<Option<FacetHandle>> {
    for (simplex_key, simplex) in dt.simplices() {
        let Some(neighbors) = simplex.neighbors() else {
            continue;
        };
        for (facet_idx, neighbor) in neighbors.enumerate() {
            if neighbor.is_some() {
                let Ok(facet_idx) = u8::try_from(facet_idx) else {
                    continue;
                };
                return Ok(Some(FacetHandle::try_new(
                    dt.tds(),
                    simplex_key,
                    facet_idx,
                )?));
            }
        }
    }
    Ok(None)
}

/// Finds a 3D k=2 facet whose forward move and public inverse both succeed.
fn find_roundtrip_k2_facet_3d(dt: &Dt3) -> ExampleResult<Option<FacetHandle>> {
    for (simplex_key, simplex) in dt.simplices() {
        let Some(neighbors) = simplex.neighbors() else {
            continue;
        };
        for (facet_idx, neighbor) in neighbors.enumerate() {
            if neighbor.is_none() {
                continue;
            }
            let Ok(facet_idx) = u8::try_from(facet_idx) else {
                continue;
            };
            let facet = FacetHandle::try_new(dt.tds(), simplex_key, facet_idx)?;
            if dt.propose_pachner(PachnerMove::K2 { facet }).is_err() {
                continue;
            }
            let mut trial = dt.clone();
            let Ok(proposal) = trial.propose_pachner(PachnerMove::K2 { facet }) else {
                continue;
            };
            let Ok(info) = proposal.attempt_on(&mut trial) else {
                continue;
            };
            let Ok(edge) = inserted_edge_3d(&trial, &info.inserted_face_vertices, "3D k=2 demo")
            else {
                continue;
            };
            let Ok(inverse) = trial.propose_pachner(PachnerMove::K2Inverse { edge }) else {
                continue;
            };
            if inverse.attempt_on(&mut trial).is_ok() && trial.validate().is_ok() {
                return Ok(Some(facet));
            }
        }
    }
    Ok(None)
}

/// Finds a 3D k=3 ridge whose forward move succeeds and preserves topology.
fn find_flippable_ridge_3d(dt: &Dt3) -> ExampleResult<Option<RidgeHandle>> {
    for (simplex_key, simplex) in dt.simplices() {
        let vertex_count = simplex.number_of_vertices();
        for i in 0..vertex_count {
            for j in (i + 1)..vertex_count {
                let Ok(omit_a) = u8::try_from(i) else {
                    continue;
                };
                let Ok(omit_b) = u8::try_from(j) else {
                    continue;
                };
                let ridge = RidgeHandle::try_new(dt.tds(), simplex_key, omit_a, omit_b)?;
                let mut trial = dt.clone();
                let Ok(proposal) = trial.propose_pachner(PachnerMove::K3 { ridge }) else {
                    continue;
                };
                if proposal.attempt_on(&mut trial).is_ok()
                    && trial.as_triangulation().validate().is_ok()
                {
                    return Ok(Some(ridge));
                }
            }
        }
    }
    Ok(None)
}

/// Parses the inserted face of a k=2 move into the edge expected by its inverse.
fn inserted_edge_3d(
    dt: &Dt3,
    vertices: &[VertexKey],
    demo: &'static str,
) -> ExampleResult<EdgeKey> {
    let [a, b] = vertices else {
        return Err(
            TopologyEditingExampleError::UnexpectedInsertedFaceVertices {
                demo,
                expected: 2,
                actual: vertices.len(),
            },
        );
    };
    Ok(EdgeKey::try_new(dt.tds(), *a, *b)?)
}

/// Parses the inserted face of a k=3 move into its inverse triangle candidate.
fn inserted_triangle(vertices: &[VertexKey], demo: &'static str) -> ExampleResult<TriangleHandle> {
    let [a, b, c] = vertices else {
        return Err(
            TopologyEditingExampleError::UnexpectedInsertedFaceVertices {
                demo,
                expected: 3,
                actual: vertices.len(),
            },
        );
    };
    Ok(TriangleHandle::try_new(*a, *b, *c)?)
}
