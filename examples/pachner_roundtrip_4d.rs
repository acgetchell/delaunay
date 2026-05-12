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
use delaunay::prelude::geometry::RobustKernel;
use delaunay::prelude::triangulation::construction::{
    ConstructionOptions, DelaunayTriangulationConstructionError, InsertionOrderStrategy, Vertex,
};
use delaunay::prelude::triangulation::flips::*;
use delaunay::prelude::triangulation::validation::DelaunayTriangulationValidationError;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MoveKind {
    K1,
    K2,
    K3,
}

impl std::fmt::Display for MoveKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::K1 => f.write_str("k=1"),
            Self::K2 => f.write_str("k=2"),
            Self::K3 => f.write_str("k=3"),
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum PachnerRoundtripError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Validation(#[from] DelaunayTriangulationValidationError),
    #[error(transparent)]
    Flip(#[from] FlipError),
    #[error(transparent)]
    IntegerConversion(#[from] std::num::TryFromIntError),
    #[error("triangulation has no cells")]
    EmptyTriangulation,
    #[error("cell {cell_key:?} not found in TDS")]
    MissingCell { cell_key: CellKey },
    #[error("vertex {vertex_key:?} not found in TDS")]
    MissingVertex { vertex_key: VertexKey },
    #[error("inserted vertex with UUID {uuid} not found in TDS")]
    MissingInsertedVertexUuid { uuid: Uuid },
    #[error("{move_kind} roundtrip changed the combinatorial topology")]
    TopologyChanged { move_kind: MoveKind },
    #[error("{move_kind} flip produced {actual} inserted-face vertices, expected {expected}")]
    UnexpectedInsertedFace {
        move_kind: MoveKind,
        actual: usize,
        expected: usize,
    },
    #[error("no flippable candidate found for {move_kind}; last error: {last_error:?}")]
    NoFlippableMove {
        move_kind: MoveKind,
        last_error: Option<FlipError>,
    },
}

fn main() -> Result<(), PachnerRoundtripError> {
    println!("============================================================");
    println!("4D Pachner Move Roundtrip Example (k=1,2,3 + inverses)");
    println!("============================================================\n");

    let mut dt = match build_triangulation() {
        Ok(dt) => dt,
        Err(err) => {
            eprintln!("⚠️  Unable to build the stable 4D PL-manifold triangulation: {err}");
            eprintln!("    Skipping Pachner roundtrip example for this run.");
            return Ok(());
        }
    };

    println!(
        "Triangulation: {} vertices, {} cells",
        dt.number_of_vertices(),
        dt.number_of_cells()
    );

    let start = Instant::now();
    dt.validate()?;
    println!(
        "✓ Initial PL-manifold + Delaunay validation passed in {:?}\n",
        start.elapsed()
    );

    let baseline = snapshot_topology(&dt)?;

    let start = Instant::now();
    roundtrip_k1(&mut dt)?;
    assert_roundtrip(MoveKind::K1, &dt, &baseline)?;
    println!(
        "✓ k=1 roundtrip preserved triangulation ({:?})",
        start.elapsed()
    );

    let start = Instant::now();
    roundtrip_k2(&mut dt)?;
    assert_roundtrip(MoveKind::K2, &dt, &baseline)?;
    println!(
        "✓ k=2 roundtrip preserved triangulation ({:?})",
        start.elapsed()
    );

    let start = Instant::now();
    roundtrip_k3(&mut dt)?;
    assert_roundtrip(MoveKind::K3, &dt, &baseline)?;
    println!(
        "✓ k=3 roundtrip preserved triangulation ({:?})",
        start.elapsed()
    );

    println!("\n============================================================");
    println!("All Pachner move roundtrips preserved the Delaunay triangulation");
    println!("============================================================");

    Ok(())
}

fn build_triangulation() -> Result<Dt4, PachnerRoundtripError> {
    let vertices = stable_vertices();
    let start = Instant::now();
    let options =
        ConstructionOptions::default().with_insertion_order(InsertionOrderStrategy::Input);
    let dt: Dt4 = DelaunayTriangulation::with_topology_guarantee_and_options(
        &RobustKernel::new(),
        &vertices,
        TopologyGuarantee::PLManifold,
        options,
    )?;
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

fn snapshot_topology(dt: &Dt4) -> Result<TopologySnapshot, PachnerRoundtripError> {
    let tds = dt.tds();
    let mut vertex_uuids: Vec<Uuid> = tds.vertices().map(|(_, vertex)| vertex.uuid()).collect();
    vertex_uuids.sort();

    let mut cell_vertex_uuids: Vec<Vec<Uuid>> = Vec::new();
    for (_, cell) in tds.cells() {
        let mut uuids: Vec<Uuid> = Vec::new();
        for &vkey in cell.vertices() {
            let vertex = tds
                .vertex(vkey)
                .ok_or(PachnerRoundtripError::MissingVertex { vertex_key: vkey })?;
            uuids.push(vertex.uuid());
        }
        uuids.sort();
        cell_vertex_uuids.push(uuids);
    }
    cell_vertex_uuids.sort();

    Ok(TopologySnapshot {
        vertex_uuids,
        cell_vertex_uuids,
    })
}

fn assert_roundtrip(
    move_kind: MoveKind,
    dt: &Dt4,
    baseline: &TopologySnapshot,
) -> Result<(), PachnerRoundtripError> {
    let after = snapshot_topology(dt)?;
    if &after != baseline {
        return Err(PachnerRoundtripError::TopologyChanged { move_kind });
    }

    dt.validate()?;
    Ok(())
}

fn cell_centroid(dt: &Dt4, cell_key: CellKey) -> Result<[f64; 4], PachnerRoundtripError> {
    let cell = dt
        .tds()
        .cell(cell_key)
        .ok_or(PachnerRoundtripError::MissingCell { cell_key })?;

    let mut coords = [0.0_f64; 4];
    for &vkey in cell.vertices() {
        let vertex = dt
            .tds()
            .vertex(vkey)
            .ok_or(PachnerRoundtripError::MissingVertex { vertex_key: vkey })?;
        let vcoords = vertex.point().coords();
        for i in 0..4 {
            coords[i] += vcoords[i];
        }
    }

    let vertex_count = u32::try_from(cell.vertices().len())?;
    let inv = 1.0_f64 / f64::from(vertex_count);
    for coord in &mut coords {
        *coord *= inv;
    }
    Ok(coords)
}

fn roundtrip_k1(dt: &mut Dt4) -> Result<(), PachnerRoundtripError> {
    let cell_key = dt
        .cells()
        .next()
        .map(|(cell_key, _)| cell_key)
        .ok_or(PachnerRoundtripError::EmptyTriangulation)?;
    let centroid = cell_centroid(dt, cell_key)?;

    let new_vertex = vertex!(centroid);
    let new_uuid = new_vertex.uuid();

    dt.flip_k1_insert(cell_key, new_vertex)?;

    let new_key = dt
        .tds()
        .vertex_key_from_uuid(&new_uuid)
        .ok_or(PachnerRoundtripError::MissingInsertedVertexUuid { uuid: new_uuid })?;

    dt.flip_k1_remove(new_key)?;
    Ok(())
}

fn collect_interior_facets(dt: &Dt4) -> Vec<FacetHandle> {
    let mut facets = Vec::new();
    for (cell_key, cell) in dt.cells() {
        if let Some(neighbors) = cell.neighbors() {
            for (facet_index, neighbor) in neighbors.iter().enumerate() {
                if neighbor.is_some() {
                    let Ok(facet_index) = u8::try_from(facet_index) else {
                        continue;
                    };
                    facets.push(FacetHandle::new(cell_key, facet_index));
                }
            }
        }
    }
    facets
}

fn roundtrip_k2(dt: &mut Dt4) -> Result<(), PachnerRoundtripError> {
    let candidates = collect_interior_facets(dt);
    let mut last_error: Option<FlipError> = None;

    for facet in candidates {
        match dt.flip_k2(facet) {
            Ok(info) => {
                if info.inserted_face_vertices.len() != 2 {
                    return Err(PachnerRoundtripError::UnexpectedInsertedFace {
                        move_kind: MoveKind::K2,
                        actual: info.inserted_face_vertices.len(),
                        expected: 2,
                    });
                }

                let edge = EdgeKey::new(
                    info.inserted_face_vertices[0],
                    info.inserted_face_vertices[1],
                );
                dt.flip_k2_inverse_from_edge(edge)?;
                return Ok(());
            }
            Err(err) => {
                last_error = Some(err);
            }
        }
    }

    Err(PachnerRoundtripError::NoFlippableMove {
        move_kind: MoveKind::K2,
        last_error,
    })
}

fn collect_ridges(dt: &Dt4) -> Vec<RidgeHandle> {
    let mut ridges = Vec::new();
    for (cell_key, cell) in dt.cells() {
        let vertex_count = cell.number_of_vertices();
        for i in 0..vertex_count {
            for j in (i + 1)..vertex_count {
                let Ok(omit_a) = u8::try_from(i) else {
                    continue;
                };
                let Ok(omit_b) = u8::try_from(j) else {
                    continue;
                };
                ridges.push(RidgeHandle::new(cell_key, omit_a, omit_b));
            }
        }
    }
    ridges
}

fn roundtrip_k3(dt: &mut Dt4) -> Result<(), PachnerRoundtripError> {
    let candidates = collect_ridges(dt);
    let mut last_error: Option<FlipError> = None;

    for ridge in candidates {
        match dt.flip_k3(ridge) {
            Ok(info) => {
                if info.inserted_face_vertices.len() != 3 {
                    return Err(PachnerRoundtripError::UnexpectedInsertedFace {
                        move_kind: MoveKind::K3,
                        actual: info.inserted_face_vertices.len(),
                        expected: 3,
                    });
                }

                let triangle = TriangleHandle::new(
                    info.inserted_face_vertices[0],
                    info.inserted_face_vertices[1],
                    info.inserted_face_vertices[2],
                );
                dt.flip_k3_inverse_from_triangle(triangle)?;
                return Ok(());
            }
            Err(err) => {
                last_error = Some(err);
            }
        }
    }

    Err(PachnerRoundtripError::NoFlippableMove {
        move_kind: MoveKind::K3,
        last_error,
    })
}
