//! Intrinsic 2D surfaces assembled from explicitly glued triangles.
//!
//! Each edge has a caller-chosen identity, so two edges can join the same
//! vertices. Paired sides use the same edge identity and specify their
//! directions independently. Each triangle specifies a cycle of three side
//! occurrences. Corners may identify after gluing, and the two ends of a
//! loop edge remain distinct in its vertex link. Valid surfaces have one
//! or two side uses per edge and a single path or cycle in every vertex link.
//! Coordinates and geometric predicates are outside this representation.
//!
//! The [topology references] describe the PL vertex-link criterion and
//! the Δ-complex construction behind this representation.
//!
//! [topology references]: https://github.com/acgetchell/delaunay/blob/main/REFERENCES.md#topological-manifolds-realizations-and-pl-topology-levels-3-4-validation

#![forbid(unsafe_code)]

use super::characteristics::euler::euler_characteristic_wide;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use std::collections::{BTreeMap, BTreeSet, btree_map::Entry};
use thiserror::Error;

// =============================================================================
// ERRORS
// =============================================================================

/// A rejected construction or edit of a triangle gluing.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum TriangleGluingError {
    /// Two records claim the same identity within one cell kind.
    #[error("duplicate cell {cell:?}")]
    DuplicateCell {
        /// Repeated identity.
        cell: CellId,
    },
    /// A cell listed for removal is absent or differs from the expected record.
    #[error("edit does not match existing cell {cell:?}")]
    EditMismatch {
        /// Identity of the expected record.
        cell: CellId,
    },
    /// An edge references an absent vertex.
    #[error("edge {edge:?} references missing vertex {vertex:?}")]
    MissingVertex {
        /// Edge with the invalid endpoint.
        edge: EdgeId,
        /// Missing endpoint.
        vertex: VertexId,
    },
    /// A triangle references an absent edge.
    #[error("triangle {triangle:?} references missing edge {edge:?}")]
    MissingEdge {
        /// Triangle with the invalid side.
        triangle: TriangleId,
        /// Missing edge.
        edge: EdgeId,
    },
    /// Two consecutive directed sides fail to meet at their shared corner.
    #[error(
        "triangle {triangle:?}: edge {preceding:?} ends at {end:?}, but edge {following:?} starts at {start:?}"
    )]
    TriangleBoundaryMismatch {
        /// Triangle with inconsistent sides.
        triangle: TriangleId,
        /// Edge occurrence ending at the corner.
        preceding: EdgeId,
        /// Edge occurrence starting at the corner.
        following: EdgeId,
        /// End vertex of the preceding directed edge.
        end: VertexId,
        /// Start vertex of the following directed edge.
        start: VertexId,
    },
    /// An edge has no incident side use or has more than two.
    #[error("edge {edge:?} has {incidence} incident sides, expected one or two")]
    EdgeIncidence {
        /// Edge with invalid incidence.
        edge: EdgeId,
        /// Actual number of uses.
        incidence: usize,
    },
    /// A listed vertex has no incident triangle.
    #[error("vertex {vertex:?} is isolated")]
    IsolatedVertex {
        /// Vertex without an incident triangle.
        vertex: VertexId,
    },
    /// An incident vertex has a disconnected link.
    #[error("vertex {vertex:?} has {components} link components, expected one")]
    DisconnectedVertexLink {
        /// Vertex with an invalid neighborhood.
        vertex: VertexId,
        /// Number of connected components in its link.
        components: usize,
    },
    /// A component admits no consistent orientation of its triangle sides.
    #[error("component containing triangle {triangle:?} is nonorientable")]
    NonOrientable {
        /// Smallest triangle ID in the component.
        triangle: TriangleId,
    },
    /// A component's Euler characteristic exceeds the supported range.
    #[error(
        "component containing {triangle:?} has Euler characteristic {value}, outside the isize range"
    )]
    EulerCharacteristicOutOfRange {
        /// A triangle identifying the component.
        triangle: TriangleId,
        /// Exact alternating sum of the cell counts.
        value: i128,
    },
}

// =============================================================================
// CELL RECORDS
// =============================================================================

/// A stable vertex identity within a gluing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct VertexId(pub u64);

/// A stable edge identity, independent of its endpoints.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EdgeId(pub u64);

/// A stable triangle identity, independent of its boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TriangleId(pub u64);

/// Identity of a cell involved in a construction or edit diagnostic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum CellId {
    /// Vertex identity.
    Vertex(VertexId),
    /// Edge identity.
    Edge(EdgeId),
    /// Triangle identity.
    Triangle(TriangleId),
}

/// An edge with a reference direction from the first endpoint to the second.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edge {
    /// Identity of this edge occurrence.
    pub id: EdgeId,
    /// Endpoints in reference order. Both ends may use the same vertex ID.
    pub endpoints: [VertexId; 2],
}

/// One directed use of an edge in a triangle's boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeUse {
    /// Identity of the edge occurrence.
    pub edge: EdgeId,
    /// Whether the traversal runs against the edge's reference direction.
    pub reversed: bool,
}

/// A triangle with three directed edge uses in cyclic order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Triangle {
    /// Identity of this triangle.
    pub id: TriangleId,
    /// Directed side occurrences in boundary order. IDs may repeat after gluing.
    pub sides: [EdgeUse; 3],
}

/// Cell records for construction or an edit. Records may be given in any order.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cells {
    /// Vertex identities.
    pub vertices: Vec<VertexId>,
    /// Edge occurrences.
    pub edges: Vec<Edge>,
    /// Triangle occurrences.
    pub triangles: Vec<Triangle>,
}

/// A replacement of explicitly selected cells.
///
/// A replacement may reuse identities listed in both halves. Reversing a
/// successful edit restores the original records with their original IDs,
/// provided the reverse edit still matches and forms a valid surface.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Edit {
    /// Exact records to remove from the current gluing.
    pub remove: Cells,
    /// Records to insert after removal.
    pub insert: Cells,
}

/// Topological data for a connected component of the gluing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Component {
    /// Triangle identities in ascending order.
    pub triangles: Vec<TriangleId>,
    /// Number of vertices.
    pub vertex_count: usize,
    /// Number of edge occurrences.
    pub edge_count: usize,
    /// Number of boundary circles.
    pub boundary_count: usize,
    /// Whether triangle orientations can be made opposite along every shared edge.
    pub orientable: bool,
    /// Euler characteristic of the component.
    pub euler_characteristic: isize,
}

// =============================================================================
// VALIDATED OWNER
// =============================================================================

/// A validated disjoint union of compact triangulated 2D manifolds with boundary.
///
/// The empty gluing is valid. Nonempty gluings may contain loop edges, parallel
/// edges, and nonorientable components. Specify paired triangle sides with
/// the same edge ID and give identified corners the same vertex ID. Those
/// caller-supplied identities persist through edits and serialization. Serde
/// uses a versioned envelope and revalidates the cells during deserialization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TriangleGluing {
    cells: Cells,
    components: Vec<Component>,
}

impl TriangleGluing {
    /// Validates cell identities, incidences and vertex links.
    ///
    /// # Errors
    ///
    /// Returns a typed error for invalid records or neighborhoods.
    ///
    /// ```rust
    /// use delaunay::prelude::topology::triangle_gluing::{
    ///     Cells, Edge, EdgeId, EdgeUse, Triangle, TriangleGluing, TriangleGluingError,
    ///     TriangleId, VertexId,
    /// };
    /// # fn main() -> Result<(), TriangleGluingError> {
    /// let vertices = [VertexId(0), VertexId(1), VertexId(2)];
    /// let [a, b, c] = vertices;
    /// let edges = [
    ///     Edge { id: EdgeId(0), endpoints: [a, b] },
    ///     Edge { id: EdgeId(1), endpoints: [b, c] },
    ///     Edge { id: EdgeId(2), endpoints: [c, a] },
    /// ];
    /// let disk = TriangleGluing::try_new(Cells {
    ///     vertices: vertices.to_vec(),
    ///     edges: edges.to_vec(),
    ///     triangles: vec![Triangle {
    ///         id: TriangleId(0),
    ///         sides: edges.map(|edge| EdgeUse { edge: edge.id, reversed: false }),
    ///     }],
    /// })?;
    /// assert_eq!(disk.components()[0].euler_characteristic, 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_new(mut cells: Cells) -> Result<Self, TriangleGluingError> {
        sort_unique(&mut cells.vertices, |id| CellId::Vertex(*id))?;
        sort_unique(&mut cells.edges, |edge| CellId::Edge(edge.id))?;
        sort_unique(&mut cells.triangles, |triangle| {
            CellId::Triangle(triangle.id)
        })?;
        let incidence = validate_cells(&cells)?;
        let components = summarize_components(&cells, &incidence)?;
        Ok(Self { cells, components })
    }

    /// Borrows the validated records in ascending identity order within each kind.
    ///
    /// ```rust
    /// use delaunay::prelude::topology::triangle_gluing::{Cells, TriangleGluing, TriangleGluingError};
    /// # fn main() -> Result<(), TriangleGluingError> {
    /// let gluing = TriangleGluing::try_new(Cells::default())?;
    /// assert!(gluing.cells().edges.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn cells(&self) -> &Cells {
        &self.cells
    }

    /// Borrows the component summaries in order of their smallest triangle ID.
    ///
    /// ```rust
    /// use delaunay::prelude::topology::triangle_gluing::{Cells, TriangleGluing, TriangleGluingError};
    /// # fn main() -> Result<(), TriangleGluingError> {
    /// let gluing = TriangleGluing::try_new(Cells::default())?;
    /// assert!(gluing.components().is_empty());
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn components(&self) -> &[Component] {
        self.components.as_slice()
    }

    /// Requires every component to admit a coherent intrinsic orientation.
    ///
    /// # Errors
    ///
    /// Returns [`TriangleGluingError::NonOrientable`] for the first
    /// nonorientable component, in component order.
    ///
    /// ```rust
    /// use delaunay::prelude::topology::triangle_gluing::{Cells, TriangleGluing, TriangleGluingError};
    /// # fn main() -> Result<(), TriangleGluingError> {
    /// let surface = TriangleGluing::try_new(Cells::default())?;
    /// surface.validate_orientability()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn validate_orientability(&self) -> Result<(), TriangleGluingError> {
        for component in &self.components {
            if !component.orientable {
                return Err(TriangleGluingError::NonOrientable {
                    triangle: component.triangles[0],
                });
            }
        }
        Ok(())
    }

    /// Validates a replacement and commits it atomically, returning its inverse.
    ///
    /// This checks the entire candidate surface. Each listed removal must
    /// match its current record exactly. Surviving records keep their identities.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the edit fails to match the current cells
    /// or would produce an invalid surface. An error preserves all state.
    ///
    /// ```rust
    /// use delaunay::prelude::topology::triangle_gluing::{Edit, TriangleGluing, TriangleGluingError};
    /// # fn apply_and_reverse(gluing: &mut TriangleGluing, edit: Edit)
    /// # -> Result<(), TriangleGluingError> {
    /// let original = gluing.clone();
    /// let reverse = gluing.try_apply(edit)?;
    /// gluing.try_apply(reverse)?;
    /// assert_eq!(*gluing, original);
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_apply(&mut self, edit: Edit) -> Result<Edit, TriangleGluingError> {
        let mut candidate = self.cells.clone();
        replace_cells(
            &mut candidate.vertices,
            &edit.remove.vertices,
            &edit.insert.vertices,
            |id| CellId::Vertex(*id),
        )?;
        replace_cells(
            &mut candidate.edges,
            &edit.remove.edges,
            &edit.insert.edges,
            |edge| CellId::Edge(edge.id),
        )?;
        replace_cells(
            &mut candidate.triangles,
            &edit.remove.triangles,
            &edit.insert.triangles,
            |triangle| CellId::Triangle(triangle.id),
        )?;
        *self = Self::try_new(candidate)?;
        Ok(Edit {
            remove: edit.insert,
            insert: edit.remove,
        })
    }
}

// =============================================================================
// CELL AND TOPOLOGY VALIDATION
// =============================================================================

/// One triangle-side occurrence, including self-pairings within a triangle.
#[derive(Clone, Copy)]
struct IncidentSide {
    triangle: TriangleId,
    index: usize,
    reversed: bool,
}

/// A link vertex distinguishes the two ends of an edge even on a loop.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct EdgeEnd {
    edge: EdgeId,
    at_end: bool,
}

type EdgeIncidence = BTreeMap<EdgeId, Vec<IncidentSide>>;
type Adjacency<K> = BTreeMap<K, Vec<K>>;

/// Canonicalizes records so equality and persistence depend on cell identities.
fn sort_unique<T>(records: &mut [T], id: impl Fn(&T) -> CellId) -> Result<(), TriangleGluingError> {
    records.sort_unstable_by_key(&id);
    for pair in records.windows(2) {
        let cell = id(&pair[0]);
        if cell == id(&pair[1]) {
            return Err(TriangleGluingError::DuplicateCell { cell });
        }
    }
    Ok(())
}

/// Applies exact replacements to detached, sorted cell records.
fn replace_cells<T: Copy + PartialEq>(
    records: &mut Vec<T>,
    remove: &[T],
    insert: &[T],
    id: impl Fn(&T) -> CellId,
) -> Result<(), TriangleGluingError> {
    for record in remove {
        let cell = id(record);
        let index = records
            .binary_search_by_key(&cell, &id)
            .map_err(|_| TriangleGluingError::EditMismatch { cell })?;
        if records[index] != *record {
            return Err(TriangleGluingError::EditMismatch { cell });
        }
        records.remove(index);
    }
    for record in insert {
        let cell = id(record);
        match records.binary_search_by_key(&cell, &id) {
            Ok(_) => return Err(TriangleGluingError::DuplicateCell { cell }),
            Err(index) => records.insert(index, *record),
        }
    }
    Ok(())
}

/// Proves that each edge and vertex has a manifold neighborhood.
fn validate_cells(cells: &Cells) -> Result<EdgeIncidence, TriangleGluingError> {
    let mut incidence = EdgeIncidence::new();
    for edge in &cells.edges {
        for vertex in edge.endpoints {
            if cells.vertices.binary_search(&vertex).is_err() {
                return Err(TriangleGluingError::MissingVertex {
                    edge: edge.id,
                    vertex,
                });
            }
        }
        incidence.insert(edge.id, Vec::new());
    }

    let mut links: BTreeMap<VertexId, Adjacency<EdgeEnd>> = BTreeMap::new();
    for triangle in &cells.triangles {
        let vertices = triangle_vertices(&cells.edges, triangle)?;
        for (index, side) in triangle.sides.iter().enumerate() {
            incidence.entry(side.edge).or_default().push(IncidentSide {
                triangle: triangle.id,
                index,
                reversed: side.reversed,
            });
            let previous = triangle.sides[(index + 2) % 3];
            connect(
                links.entry(vertices[index]).or_default(),
                EdgeEnd {
                    edge: previous.edge,
                    at_end: !previous.reversed,
                },
                EdgeEnd {
                    edge: side.edge,
                    at_end: side.reversed,
                },
            );
        }
    }
    for (&edge, uses) in &incidence {
        if !(1..=2).contains(&uses.len()) {
            return Err(TriangleGluingError::EdgeIncidence {
                edge,
                incidence: uses.len(),
            });
        }
    }
    for &vertex in &cells.vertices {
        // Each edge end has degree equal to the edge's side incidence, so a
        // connected link is a path or cycle, including a one-node cycle.
        let link = links
            .get(&vertex)
            .ok_or(TriangleGluingError::IsolatedVertex { vertex })?;
        let components = component_count(link);
        if components != 1 {
            return Err(TriangleGluingError::DisconnectedVertexLink { vertex, components });
        }
    }
    Ok(incidence)
}

/// Recovers the ordered triangle corners from its directed boundary.
fn triangle_vertices(
    edges: &[Edge],
    triangle: &Triangle,
) -> Result<[VertexId; 3], TriangleGluingError> {
    let [first, second, third] = triangle
        .sides
        .map(|side| -> Result<_, TriangleGluingError> {
            let index = edges
                .binary_search_by_key(&side.edge, |edge| edge.id)
                .map_err(|_| TriangleGluingError::MissingEdge {
                    triangle: triangle.id,
                    edge: side.edge,
                })?;
            let mut endpoints = edges[index].endpoints;
            if side.reversed {
                endpoints.reverse();
            }
            Ok(endpoints)
        });
    let endpoints = [first?, second?, third?];
    for (index, &[_, end]) in endpoints.iter().enumerate() {
        let next = (index + 1) % endpoints.len();
        let [start, _] = endpoints[next];
        if end != start {
            return Err(TriangleGluingError::TriangleBoundaryMismatch {
                triangle: triangle.id,
                preceding: triangle.sides[index].edge,
                following: triangle.sides[next].edge,
                end,
                start,
            });
        }
    }
    Ok(endpoints.map(|[start, _]| start))
}

/// Records both directions of an undirected adjacency, preserving multiplicity.
fn connect<K: Copy + Ord>(graph: &mut Adjacency<K>, first: K, second: K) {
    graph.entry(first).or_default().push(second);
    graph.entry(second).or_default().push(first);
}

/// Counts the nonempty components of a vertex link or boundary graph.
fn component_count<K: Copy + Ord>(graph: &Adjacency<K>) -> usize {
    let mut unseen: BTreeSet<_> = graph.keys().copied().collect();
    let mut count = 0;
    let mut pending = Vec::new();
    while let Some(seed) = unseen.pop_first() {
        count += 1;
        pending.push(seed);
        while let Some(node) = pending.pop() {
            pending.extend(
                graph[&node]
                    .iter()
                    .copied()
                    .filter(|next| unseen.remove(next)),
            );
        }
    }
    count
}

/// Traverses each dual component and propagates consistent face directions.
fn summarize_components(
    cells: &Cells,
    incidence: &EdgeIncidence,
) -> Result<Vec<Component>, TriangleGluingError> {
    let triangles: BTreeMap<_, _> = cells.triangles.iter().map(|face| (face.id, face)).collect();
    let edges: BTreeMap<_, _> = cells.edges.iter().map(|edge| (edge.id, edge)).collect();
    let mut orientation = BTreeMap::new();
    let mut components = Vec::new();
    for &seed in triangles.keys() {
        let Entry::Vacant(entry) = orientation.entry(seed) else {
            continue;
        };
        entry.insert(false);
        let mut pending = vec![(seed, false)];
        let mut members = Vec::new();
        let mut seen_edges = BTreeSet::new();
        let mut vertices = BTreeSet::new();
        let mut boundary = Adjacency::new();
        let mut orientable = true;
        while let Some((id, flip)) = pending.pop() {
            members.push(id);
            for (index, side) in triangles[&id].sides.iter().enumerate() {
                let uses = &incidence[&side.edge];
                if seen_edges.insert(side.edge) {
                    let [first, second] = edges[&side.edge].endpoints;
                    vertices.extend([first, second]);
                    if uses.len() == 1 {
                        connect(&mut boundary, first, second);
                    }
                }
                for &incident in uses {
                    if incident.triangle == id && incident.index == index {
                        continue;
                    }
                    // Equal reference directions require exactly one face flip.
                    let required = flip ^ (side.reversed == incident.reversed);
                    match orientation.entry(incident.triangle) {
                        Entry::Occupied(assigned) => orientable &= *assigned.get() == required,
                        Entry::Vacant(unassigned) => {
                            unassigned.insert(required);
                            pending.push((incident.triangle, required));
                        }
                    }
                }
            }
        }
        members.sort_unstable();
        let vertex_count = vertices.len();
        let edge_count = seen_edges.len();
        let value = euler_characteristic_wide(&[vertex_count, edge_count, members.len()]);
        let euler_characteristic = isize::try_from(value).map_err(|_| {
            TriangleGluingError::EulerCharacteristicOutOfRange {
                triangle: seed,
                value,
            }
        })?;
        components.push(Component {
            triangles: members,
            vertex_count,
            edge_count,
            boundary_count: component_count(&boundary),
            orientable,
            euler_characteristic,
        });
    }
    Ok(components)
}

// =============================================================================
// SERIALIZATION
// =============================================================================

const SCHEMA_VERSION: u32 = 1;

#[derive(Serialize)]
#[serde(rename = "TriangleGluing")]
struct GluingRef<'a> {
    schema_version: u32,
    cells: &'a Cells,
}

#[derive(Deserialize)]
#[serde(rename = "TriangleGluing", deny_unknown_fields)]
struct GluingWire {
    schema_version: u32,
    cells: Cells,
}

impl Serialize for TriangleGluing {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        GluingRef {
            schema_version: SCHEMA_VERSION,
            cells: &self.cells,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for TriangleGluing {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = GluingWire::deserialize(deserializer)?;
        if wire.schema_version != SCHEMA_VERSION {
            return Err(de::Error::custom(format_args!(
                "unsupported triangle gluing schema version {}, expected {SCHEMA_VERSION}",
                wire.schema_version,
            )));
        }
        Self::try_new(wire.cells).map_err(de::Error::custom)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(id: u64, endpoints: [u64; 2]) -> Edge {
        Edge {
            id: EdgeId(id),
            endpoints: endpoints.map(VertexId),
        }
    }

    const fn side(id: u64, reversed: bool) -> EdgeUse {
        EdgeUse {
            edge: EdgeId(id),
            reversed,
        }
    }

    fn disk() -> Cells {
        Cells {
            vertices: (0..4).map(VertexId).collect(),
            edges: vec![
                edge(0, [0, 1]),
                edge(1, [1, 2]),
                edge(2, [2, 3]),
                edge(3, [3, 0]),
                edge(4, [0, 2]),
            ],
            triangles: vec![
                Triangle {
                    id: TriangleId(0),
                    sides: [side(0, false), side(1, false), side(4, true)],
                },
                Triangle {
                    id: TriangleId(1),
                    sides: [side(4, false), side(2, false), side(3, false)],
                },
            ],
        }
    }

    /// A Möbius band from one triangle with two sides paired in the same direction.
    fn self_glued_band() -> Cells {
        Cells {
            vertices: vec![VertexId(0)],
            edges: vec![edge(0, [0, 0]), edge(1, [0, 0])],
            triangles: vec![Triangle {
                id: TriangleId(0),
                sides: [side(0, false), side(0, false), side(1, false)],
            }],
        }
    }

    #[test]
    fn canonicalizes_each_cell_kind_by_identity() {
        let expected = disk();
        let mut cells = expected.clone();
        cells.vertices.reverse();
        cells.edges.reverse();
        cells.triangles.reverse();

        let gluing = TriangleGluing::try_new(cells).expect("square disk");
        assert_eq!(gluing.cells(), &expected);
        assert_eq!(
            gluing.components()[0].triangles,
            [TriangleId(0), TriangleId(1)]
        );
    }

    #[test]
    fn rejects_duplicate_ids_even_with_different_records() {
        let disk = disk();
        let mut vertices = disk.clone();
        vertices.vertices.push(VertexId(0));
        assert_eq!(
            TriangleGluing::try_new(vertices),
            Err(TriangleGluingError::DuplicateCell {
                cell: CellId::Vertex(VertexId(0)),
            })
        );

        let mut edges = disk.clone();
        edges.edges.push(edge(0, [0, 2]));
        assert_eq!(
            TriangleGluing::try_new(edges),
            Err(TriangleGluingError::DuplicateCell {
                cell: CellId::Edge(EdgeId(0)),
            })
        );

        let mut triangles = disk;
        triangles.triangles.push(Triangle {
            id: TriangleId(0),
            sides: triangles.triangles[1].sides,
        });
        assert_eq!(
            TriangleGluing::try_new(triangles),
            Err(TriangleGluingError::DuplicateCell {
                cell: CellId::Triangle(TriangleId(0)),
            })
        );
    }

    #[test]
    fn rejects_an_isolated_vertex() {
        let mut cells = disk();
        cells.vertices.push(VertexId(9));
        assert_eq!(
            TriangleGluing::try_new(cells),
            Err(TriangleGluingError::IsolatedVertex {
                vertex: VertexId(9),
            })
        );
    }

    #[test]
    fn rejects_missing_references_and_unused_edges() {
        let disk = disk();
        let mut missing_vertex = disk.clone();
        missing_vertex.edges[0].endpoints[1] = VertexId(99);
        assert_eq!(
            TriangleGluing::try_new(missing_vertex),
            Err(TriangleGluingError::MissingVertex {
                edge: EdgeId(0),
                vertex: VertexId(99),
            })
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
        unused_edge.edges.push(edge(99, [0, 1]));
        assert_eq!(
            TriangleGluing::try_new(unused_edge),
            Err(TriangleGluingError::EdgeIncidence {
                edge: EdgeId(99),
                incidence: 0,
            })
        );
    }

    #[test]
    fn self_paired_sides_constrain_orientation_and_allow_a_loop_boundary() {
        let gluing = TriangleGluing::try_new(self_glued_band()).expect("Möbius band");
        assert_eq!(
            gluing.components(),
            &[Component {
                triangles: vec![TriangleId(0)],
                vertex_count: 1,
                edge_count: 2,
                boundary_count: 1,
                orientable: false,
                euler_characteristic: 0,
            }]
        );
    }

    #[test]
    fn rejects_disconnected_link_at_identified_loop_endpoints() {
        let mut cells = self_glued_band();
        cells.edges.extend([edge(2, [0, 0]), edge(3, [0, 0])]);
        cells.triangles.push(Triangle {
            id: TriangleId(1),
            sides: [side(2, false), side(2, false), side(3, false)],
        });
        assert_eq!(
            TriangleGluing::try_new(cells),
            Err(TriangleGluingError::DisconnectedVertexLink {
                vertex: VertexId(0),
                components: 2,
            })
        );
    }

    #[test]
    fn failed_edit_preserves_owner_after_staging_removals() {
        let mut gluing = TriangleGluing::try_new(disk()).expect("square disk");
        let original = gluing.clone();
        let edit = Edit {
            remove: Cells {
                vertices: vec![VertexId(3)],
                edges: vec![edge(3, [3, 0])],
                ..Cells::default()
            },
            insert: Cells {
                edges: vec![edge(1, [0, 2])],
                ..Cells::default()
            },
        };
        assert_eq!(
            gluing.try_apply(edit),
            Err(TriangleGluingError::DuplicateCell {
                cell: CellId::Edge(EdgeId(1)),
            })
        );
        assert_eq!(gluing, original);
    }

    #[test]
    fn missing_triangle_removal_preserves_owner() {
        let mut gluing = TriangleGluing::try_new(disk()).expect("square disk");
        let original = gluing.clone();
        let missing = Triangle {
            id: TriangleId(99),
            ..original.cells().triangles[0]
        };
        let edit = Edit {
            remove: Cells {
                edges: vec![edge(3, [3, 0])],
                triangles: vec![missing],
                ..Cells::default()
            },
            ..Edit::default()
        };
        assert_eq!(
            gluing.try_apply(edit),
            Err(TriangleGluingError::EditMismatch {
                cell: CellId::Triangle(missing.id),
            })
        );
        assert_eq!(gluing, original);
    }
}
