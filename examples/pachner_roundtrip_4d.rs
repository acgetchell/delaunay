//! # 4D Pachner Move Roundtrip Example
//!
//! This example constructs a 4D **PL-manifold** Delaunay triangulation, applies
//! all Pachner moves (k=1,2,3) and their inverses, and verifies that the
//! triangulation is unchanged after each paired move + inverse.
//!
//! The example uses a small, stable 12-point configuration (a 4-simplex plus
//! interior points) to keep runs fast and deterministic.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example pachner_roundtrip_4d
//! ```

use ::uuid::Uuid;
use delaunay::core::delaunay_triangulation::{ConstructionOptions, InsertionOrderStrategy};
use delaunay::geometry::kernel::RobustKernel;
use delaunay::prelude::triangulation::Vertex;
use delaunay::prelude::triangulation::flips::*;
use std::time::Instant;

type Dt4 = DelaunayTriangulation<RobustKernel<f64>, (), (), 4>;
const STABLE_POINTS_4D: &[[f64; 4]] = &[
    // 4-simplex hull (convex hull vertices)
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    // Interior points (non-symmetric to avoid degeneracy)
    [0.10, 0.10, 0.10, 0.10],
    [0.15, 0.10, 0.10, 0.10],
    [0.10, 0.15, 0.10, 0.10],
    [0.10, 0.10, 0.15, 0.10],
    [0.12, 0.12, 0.12, 0.12],
    [0.20, 0.15, 0.10, 0.05],
    [0.08, 0.18, 0.12, 0.14],
];

#[derive(Debug, Clone, PartialEq, Eq)]
struct TopologySnapshot {
    vertex_uuids: Vec<Uuid>,
    cell_vertex_uuids: Vec<Vec<Uuid>>,
}

fn main() {
    println!("============================================================");
    println!("4D Pachner Move Roundtrip Example (k=1,2,3 + inverses)");
    println!("============================================================\n");

    let mut dt = match build_triangulation() {
        Ok(dt) => dt,
        Err(err) => {
            eprintln!("⚠️  Unable to build the stable 4D PL-manifold triangulation: {err}");
            eprintln!("    Skipping Pachner roundtrip example for this run.");
            return;
        }
    };

    println!(
        "Triangulation: {} vertices, {} cells",
        dt.number_of_vertices(),
        dt.number_of_cells()
    );

    let start = Instant::now();
    dt.validate()
        .expect("initial triangulation should satisfy Levels 1-4");
    println!(
        "✓ Initial PL-manifold + Delaunay validation passed in {:?}\n",
        start.elapsed()
    );

    let baseline = snapshot_topology(&dt);

    let start = Instant::now();
    roundtrip_k1(&mut dt).expect("k=1 roundtrip failed");
    assert_roundtrip("k=1", &dt, &baseline).expect("k=1 validation failed");
    println!(
        "✓ k=1 roundtrip preserved triangulation ({:?})",
        start.elapsed()
    );

    let start = Instant::now();
    roundtrip_k2(&mut dt).expect("k=2 roundtrip failed");
    assert_roundtrip("k=2", &dt, &baseline).expect("k=2 validation failed");
    println!(
        "✓ k=2 roundtrip preserved triangulation ({:?})",
        start.elapsed()
    );

    let start = Instant::now();
    roundtrip_k3(&mut dt).expect("k=3 roundtrip failed");
    assert_roundtrip("k=3", &dt, &baseline).expect("k=3 validation failed");
    println!(
        "✓ k=3 roundtrip preserved triangulation ({:?})",
        start.elapsed()
    );

    println!("\n============================================================");
    println!("All Pachner move roundtrips preserved the Delaunay triangulation");
    println!("============================================================");
}

fn build_triangulation() -> Result<Dt4, String> {
    let vertices = stable_vertices();
    let start = Instant::now();
    let options =
        ConstructionOptions::default().with_insertion_order(InsertionOrderStrategy::Input);
    let dt: Dt4 = DelaunayTriangulation::with_topology_guarantee_and_options(
        &RobustKernel::new(),
        &vertices,
        TopologyGuarantee::PLManifold,
        options,
    )
    .map_err(|e| format!("construction failed: {e}"))?;
    println!(
        "✓ Stable 4D triangulation built ({} points) in {:?}",
        vertices.len(),
        start.elapsed()
    );

    Ok(dt)
}

fn stable_vertices() -> Vec<Vertex<f64, (), 4>> {
    STABLE_POINTS_4D
        .iter()
        .map(|coords| vertex!(*coords))
        .collect()
}

fn snapshot_topology(dt: &Dt4) -> TopologySnapshot {
    let tds = dt.tds();
    let mut vertex_uuids: Vec<Uuid> = tds.vertices().map(|(_, vertex)| vertex.uuid()).collect();
    vertex_uuids.sort();

    let mut cell_vertex_uuids: Vec<Vec<Uuid>> = tds
        .cells()
        .map(|(_, cell)| {
            let mut uuids: Vec<Uuid> = cell
                .vertices()
                .iter()
                .map(|&vkey| {
                    tds.get_vertex_by_key(vkey)
                        .expect("vertex key missing in TDS")
                        .uuid()
                })
                .collect();
            uuids.sort();
            uuids
        })
        .collect();
    cell_vertex_uuids.sort();

    TopologySnapshot {
        vertex_uuids,
        cell_vertex_uuids,
    }
}

fn assert_roundtrip(step: &str, dt: &Dt4, baseline: &TopologySnapshot) -> Result<(), String> {
    let after = snapshot_topology(dt);
    if &after != baseline {
        return Err(format!(
            "{step} roundtrip changed the combinatorial topology"
        ));
    }

    dt.validate()
        .map_err(|e| format!("{step} roundtrip failed validation: {e}"))?;
    Ok(())
}

fn cell_centroid(dt: &Dt4, cell_key: CellKey) -> Result<[f64; 4], String> {
    let cell = dt
        .tds()
        .get_cell(cell_key)
        .ok_or_else(|| "cell not found in TDS".to_string())?;

    let mut coords = [0.0_f64; 4];
    for &vkey in cell.vertices() {
        let vertex = dt
            .tds()
            .get_vertex_by_key(vkey)
            .ok_or_else(|| "vertex key missing in TDS".to_string())?;
        let vcoords = vertex.point().coords();
        for i in 0..4 {
            coords[i] += vcoords[i];
        }
    }

    let vertex_count =
        u32::try_from(cell.vertices().len()).expect("cell vertex count should fit in u32");
    let inv = 1.0_f64 / f64::from(vertex_count);
    for coord in &mut coords {
        *coord *= inv;
    }
    Ok(coords)
}

fn roundtrip_k1(dt: &mut Dt4) -> Result<(), String> {
    let cell_key = dt
        .cells()
        .next()
        .map(|(cell_key, _)| cell_key)
        .ok_or_else(|| "triangulation has no cells".to_string())?;
    let centroid = cell_centroid(dt, cell_key)?;

    let new_vertex = vertex!(centroid);
    let new_uuid = new_vertex.uuid();

    dt.flip_k1_insert(cell_key, new_vertex)
        .map_err(|e| format!("k=1 insert failed: {e}"))?;

    let new_key = dt
        .tds()
        .vertex_key_from_uuid(&new_uuid)
        .ok_or_else(|| "inserted vertex not found in TDS".to_string())?;

    dt.flip_k1_remove(new_key)
        .map_err(|e| format!("k=1 remove failed: {e}"))?;
    Ok(())
}

fn collect_interior_facets(dt: &Dt4) -> Vec<FacetHandle> {
    let mut facets = Vec::new();
    for (cell_key, cell) in dt.cells() {
        if let Some(neighbors) = cell.neighbors() {
            for (facet_index, neighbor) in neighbors.iter().enumerate() {
                if neighbor.is_some() {
                    let facet_index = u8::try_from(facet_index).expect("facet index fits in u8");
                    facets.push(FacetHandle::new(cell_key, facet_index));
                }
            }
        }
    }
    facets
}

fn roundtrip_k2(dt: &mut Dt4) -> Result<(), String> {
    let candidates = collect_interior_facets(dt);
    let mut last_error: Option<String> = None;

    for facet in candidates {
        match dt.flip_k2(facet) {
            Ok(info) => {
                if info.inserted_face_vertices.len() != 2 {
                    return Err("k=2 flip produced unexpected inserted face".to_string());
                }

                let edge = EdgeKey::new(
                    info.inserted_face_vertices[0],
                    info.inserted_face_vertices[1],
                );
                dt.flip_k2_inverse_from_edge(edge)
                    .map_err(|e| format!("k=2 inverse failed: {e}"))?;
                return Ok(());
            }
            Err(err) => {
                last_error = Some(format!("{err}"));
            }
        }
    }

    Err(format!(
        "No flippable interior facet found for k=2 (last error: {})",
        last_error.unwrap_or_else(|| "none".to_string())
    ))
}

fn collect_ridges(dt: &Dt4) -> Vec<RidgeHandle> {
    let mut ridges = Vec::new();
    for (cell_key, cell) in dt.cells() {
        let vertex_count = cell.number_of_vertices();
        for i in 0..vertex_count {
            for j in (i + 1)..vertex_count {
                let omit_a = u8::try_from(i).expect("ridge index fits in u8");
                let omit_b = u8::try_from(j).expect("ridge index fits in u8");
                ridges.push(RidgeHandle::new(cell_key, omit_a, omit_b));
            }
        }
    }
    ridges
}

fn roundtrip_k3(dt: &mut Dt4) -> Result<(), String> {
    let candidates = collect_ridges(dt);
    let mut last_error: Option<String> = None;

    for ridge in candidates {
        match dt.flip_k3(ridge) {
            Ok(info) => {
                if info.inserted_face_vertices.len() != 3 {
                    return Err("k=3 flip produced unexpected inserted face".to_string());
                }

                let triangle = TriangleHandle::new(
                    info.inserted_face_vertices[0],
                    info.inserted_face_vertices[1],
                    info.inserted_face_vertices[2],
                );
                dt.flip_k3_inverse_from_triangle(triangle)
                    .map_err(|e| format!("k=3 inverse failed: {e}"))?;
                return Ok(());
            }
            Err(err) => {
                last_error = Some(format!("{err}"));
            }
        }
    }

    Err(format!(
        "No flippable ridge found for k=3 (last error: {})",
        last_error.unwrap_or_else(|| "none".to_string())
    ))
}
