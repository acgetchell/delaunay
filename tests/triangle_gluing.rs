//! Public consumer tests for intrinsic surfaces with explicit edge occurrences.

use delaunay::prelude::topology::triangle_gluing::{
    CellId, Cells, Edge, EdgeId, EdgeUse, Edit, Triangle, TriangleGluing, TriangleGluingError,
    TriangleId, VertexId,
};
use std::collections::{BTreeMap, BTreeSet};

/// The spatial circle at one slice has three vertices and three edges.
fn add_slice(cells: &mut Cells, slice: u64) {
    for vertex in 0..3 {
        let id = 3 * slice + vertex;
        cells.vertices.push(VertexId(id));
        cells.edges.push(Edge {
            id: EdgeId(id),
            endpoints: [VertexId(id), VertexId(3 * slice + (vertex + 1) % 3)],
        });
    }
}

/// Each step visits a new timelike edge occurrence, including repeated pairs.
fn add_strip(cells: &mut Cells, word: [u8; 6], lower: u64, upper: u64, edge_offset: u64) {
    let mut a = 0;
    let mut b = 0;
    for (index, step) in (0..6).zip(word) {
        let current = edge_offset + index;
        let next = edge_offset + (index + 1) % 6;
        cells.edges.push(Edge {
            id: EdgeId(current),
            endpoints: [VertexId(3 * lower + a), VertexId(3 * upper + b)],
        });
        let spatial = if step == b'U' {
            let edge = 3 * lower + a;
            a = (a + 1) % 3;
            side(edge, false)
        } else {
            let edge = 3 * upper + b;
            b = (b + 1) % 3;
            side(edge, true)
        };
        let sides = if step == b'U' {
            [side(current, true), spatial, side(next, false)]
        } else {
            [side(current, true), side(next, false), spatial]
        };
        cells.triangles.push(Triangle {
            id: TriangleId(current),
            sides,
        });
    }
}

const fn side(edge: u64, reversed: bool) -> EdgeUse {
    EdgeUse {
        edge: EdgeId(edge),
        reversed,
    }
}

fn cylinder(word: [u8; 6]) -> Cells {
    let mut cells = Cells::default();
    add_slice(&mut cells, 0);
    add_slice(&mut cells, 1);
    add_strip(&mut cells, word, 0, 1, 6);
    cells
}

/// Builds control surfaces that need only one edge per unordered vertex pair.
fn simplicial_cells(faces: &[[u64; 3]]) -> Cells {
    let mut cells = Cells::default();
    let mut edges = BTreeMap::new();
    let mut vertices = BTreeSet::new();
    for (id, &[a, b, c]) in (0..).zip(faces) {
        vertices.extend([a, b, c].map(VertexId));
        let sides = [(a, b), (b, c), (c, a)].map(|(from, to)| {
            let endpoints = if from < to { [from, to] } else { [to, from] };
            let edge = edges.entry(endpoints).or_insert_with(|| {
                let id = EdgeId(u64::try_from(cells.edges.len()).expect("small fixture"));
                cells.edges.push(Edge {
                    id,
                    endpoints: endpoints.map(VertexId),
                });
                id
            });
            EdgeUse {
                edge: *edge,
                reversed: from > to,
            }
        });
        cells.triangles.push(Triangle {
            id: TriangleId(id),
            sides,
        });
    }
    cells.vertices.extend(vertices);
    cells
}

#[test]
fn admits_all_ten_marked_cylinder_strips() -> Result<(), TriangleGluingError> {
    // Ambjørn–Loll, hep-th/9805108 §2. For l=l'=3 the one-marked strip
    // begins with U and has two further U steps among the remaining five.
    let mut parallel_words = Vec::new();
    for i in 1..5 {
        for j in i + 1..6 {
            let mut word = *b"UDDDDD";
            word[i] = b'U';
            word[j] = b'U';
            let cells = cylinder(word);
            let gluing = TriangleGluing::try_new(cells.clone())?;
            let [component] = gluing.components() else {
                panic!("expected one cylinder for {word:?}");
            };
            assert_eq!(
                (
                    component.vertex_count,
                    component.edge_count,
                    component.triangles.len()
                ),
                (6, 12, 6)
            );
            assert_eq!(
                (
                    component.euler_characteristic,
                    component.boundary_count,
                    component.orientable
                ),
                (0, 2, true)
            );
            let mut uses = BTreeMap::new();
            for triangle in &cells.triangles {
                for side in triangle.sides {
                    *uses.entry(side.edge).or_insert(0) += 1;
                }
            }
            assert!(cells.edges[6..].iter().all(|edge| uses[&edge.id] == 2));
            assert!(cells.edges[..6].iter().all(|edge| uses[&edge.id] == 1));
            let timelike_pairs: BTreeSet<_> =
                cells.edges[6..].iter().map(|edge| edge.endpoints).collect();
            if timelike_pairs.len() != 6 {
                assert_eq!(timelike_pairs.len(), 5);
                parallel_words.push(word);
            }
        }
    }
    assert_eq!(parallel_words, [*b"UUUDDD", *b"UUDDDU", *b"UDDDUU"]);
    Ok(())
}

#[test]
fn closed_periodic_gluing_preserves_parallel_edges_and_wire_identity()
-> Result<(), Box<dyn std::error::Error>> {
    let mut cells = Cells::default();
    for slice in 0..3 {
        add_slice(&mut cells, slice);
        add_strip(
            &mut cells,
            *b"UUUDDD",
            slice,
            (slice + 1) % 3,
            9 + 6 * slice,
        );
    }
    let gluing = TriangleGluing::try_new(cells)?;
    let [component] = gluing.components() else {
        panic!("expected one torus");
    };
    assert_eq!(
        (
            component.vertex_count,
            component.edge_count,
            component.triangles.len()
        ),
        (9, 27, 18)
    );
    assert_eq!(
        (
            component.euler_characteristic,
            component.boundary_count,
            component.orientable
        ),
        (0, 0, true)
    );
    assert_eq!(
        gluing.cells().edges[9].endpoints,
        gluing.cells().edges[12].endpoints
    );
    assert_ne!(gluing.cells().edges[9].id, gluing.cells().edges[12].id);

    let mut bytes = Vec::new();
    ciborium::into_writer(&gluing, &mut bytes)?;
    assert_eq!(
        ciborium::from_reader::<TriangleGluing, _>(bytes.as_slice())?,
        gluing
    );
    let mut json = serde_json::to_value(&gluing)?;
    assert_eq!(json["schema_version"], 1);
    assert_eq!(
        serde_json::from_value::<TriangleGluing>(json.clone())?,
        gluing
    );
    json["schema_version"] = 2.into();
    assert!(serde_json::from_value::<TriangleGluing>(json.clone()).is_err());
    json["schema_version"] = 1.into();
    json["cells"]["edges"][12]["id"] = json["cells"]["edges"][9]["id"].clone();
    assert!(serde_json::from_value::<TriangleGluing>(json).is_err());
    Ok(())
}

#[test]
fn local_replacement_reverses_without_remapping_parallel_edge_ids()
-> Result<(), TriangleGluingError> {
    let original = cylinder(*b"UUUDDD");
    let after_swap = cylinder(*b"UUDUDD");
    let mut gluing = TriangleGluing::try_new(original.clone())?;
    let baseline = gluing.clone();
    let edit = Edit {
        remove: Cells {
            edges: vec![original.edges[9]],
            triangles: original.triangles[2..4].to_vec(),
            ..Cells::default()
        },
        insert: Cells {
            edges: vec![after_swap.edges[9]],
            triangles: after_swap.triangles[2..4].to_vec(),
            ..Cells::default()
        },
    };
    let reverse = gluing.try_apply(edit.clone())?;
    let changed = TriangleGluing::try_new(after_swap)?;
    assert_eq!(gluing, changed);
    assert_eq!(
        gluing.try_apply(edit),
        Err(TriangleGluingError::EditMismatch {
            cell: CellId::Edge(EdgeId(9))
        })
    );
    assert_eq!(gluing, changed);
    gluing.try_apply(reverse)?;
    assert_eq!(gluing, baseline);
    Ok(())
}

#[test]
fn rejects_nonmanifold_and_inconsistent_replacements_atomically() -> Result<(), TriangleGluingError>
{
    let original = cylinder(*b"UUUDDD");
    let mut gluing = TriangleGluing::try_new(original.clone())?;
    let baseline = gluing.clone();

    let mut repeated = original.triangles[0];
    repeated.id = TriangleId(99);
    let overshare = Edit {
        insert: Cells {
            triangles: vec![repeated],
            ..Cells::default()
        },
        ..Edit::default()
    };
    assert_eq!(
        gluing.try_apply(overshare),
        Err(TriangleGluingError::EdgeIncidence {
            edge: EdgeId(6),
            incidence: 3
        })
    );
    assert_eq!(gluing, baseline);

    let mut open = original.triangles[0];
    open.sides[0].reversed = !open.sides[0].reversed;
    let wrong_direction = Edit {
        remove: Cells {
            triangles: vec![original.triangles[0]],
            ..Cells::default()
        },
        insert: Cells {
            triangles: vec![open],
            ..Cells::default()
        },
    };
    assert_eq!(
        gluing.try_apply(wrong_direction),
        Err(TriangleGluingError::OpenTriangle { triangle: open.id })
    );
    assert_eq!(gluing, baseline);

    let pinched = simplicial_cells(&[[0, 1, 2], [0, 3, 4]]);
    assert_eq!(
        TriangleGluing::try_new(pinched),
        Err(TriangleGluingError::VertexLink {
            vertex: VertexId(0),
            components: 2
        })
    );
    Ok(())
}

#[test]
fn distinguishes_an_orientable_disk_from_a_mobius_band() -> Result<(), TriangleGluingError> {
    let cells = simplicial_cells(&[
        [0, 2, 1],
        [2, 3, 1],
        [2, 4, 3],
        [4, 5, 3],
        [4, 1, 5],
        [1, 0, 5],
        [6, 7, 8],
    ]);
    let gluing = TriangleGluing::try_new(cells)?;
    let [mobius, disk] = gluing.components() else {
        panic!("expected two disjoint components");
    };
    assert_eq!(
        (
            mobius.vertex_count,
            mobius.edge_count,
            mobius.triangles.len()
        ),
        (6, 12, 6)
    );
    assert_eq!(
        (
            mobius.euler_characteristic,
            mobius.boundary_count,
            mobius.orientable
        ),
        (0, 1, false)
    );
    assert_eq!(
        (
            disk.euler_characteristic,
            disk.boundary_count,
            disk.orientable
        ),
        (1, 1, true)
    );
    Ok(())
}

#[test]
fn invalid_cell_records_report_the_offending_identity() {
    let disk = simplicial_cells(&[[0, 1, 2]]);

    let mut missing_vertex = disk.clone();
    missing_vertex.edges[0].endpoints[1] = VertexId(99);
    assert_eq!(
        TriangleGluing::try_new(missing_vertex),
        Err(TriangleGluingError::MissingVertex {
            edge: EdgeId(0),
            vertex: VertexId(99),
        })
    );

    let mut loop_edge = disk.clone();
    loop_edge.edges[0].endpoints = [VertexId(0); 2];
    assert_eq!(
        TriangleGluing::try_new(loop_edge),
        Err(TriangleGluingError::LoopEdge { edge: EdgeId(0) })
    );

    let mut missing_edge = disk.clone();
    missing_edge.edges.remove(0);
    assert_eq!(
        TriangleGluing::try_new(missing_edge),
        Err(TriangleGluingError::MissingEdge {
            triangle: TriangleId(0),
            edge: EdgeId(0),
        })
    );

    let mut unused_edge = disk;
    unused_edge.edges.push(Edge {
        id: EdgeId(99),
        endpoints: [VertexId(0), VertexId(1)],
    });
    assert_eq!(
        TriangleGluing::try_new(unused_edge),
        Err(TriangleGluingError::EdgeIncidence {
            edge: EdgeId(99),
            incidence: 0,
        })
    );
}
