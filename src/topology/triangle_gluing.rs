//! Intrinsic 2D surfaces assembled from explicitly glued triangles.
//!
//! Each edge has a caller-chosen identity, so two edges can join the same
//! vertices. A triangle specifies a directed cycle of three edge uses.
//! Valid surfaces have one or two triangles per edge and a single path or
//! cycle in every vertex link. Coordinates and geometric predicates are
//! outside this representation.
//!
//! The [topology references] describe the PL vertex-link criterion and
//! the Δ-complex construction behind this representation.
//!
//! [topology references]: https://github.com/acgetchell/delaunay/blob/main/REFERENCES.md#topological-manifolds-realizations-and-pl-topology-levels-3-4-validation

use super::characteristics::{FVector, euler_characteristic};
use super::traits::TopologyError;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::{BTreeMap, BTreeSet, btree_map::Entry};
use thiserror::Error;

/// A rejected construction or edit of a triangle gluing.
#[derive(Clone, Debug, Error, PartialEq)]
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
    /// The two endpoints of an edge coincide.
    #[error("edge {edge:?} has identical endpoints")]
    LoopEdge {
        /// Edge with coincident endpoints.
        edge: EdgeId,
    },
    /// A triangle references an absent edge.
    #[error("triangle {triangle:?} references missing edge {edge:?}")]
    MissingEdge {
        /// Triangle with the invalid side.
        triangle: TriangleId,
        /// Missing edge.
        edge: EdgeId,
    },
    /// The directed sides of a triangle fail to form a cycle.
    #[error("triangle {triangle:?} has sides that do not form a directed cycle")]
    OpenTriangle {
        /// Triangle with inconsistent sides.
        triangle: TriangleId,
    },
    /// An edge has no incident triangle or has more than two.
    #[error("edge {edge:?} has {incidence} incident triangles, expected one or two")]
    EdgeIncidence {
        /// Edge with invalid incidence.
        edge: EdgeId,
        /// Actual number of uses.
        incidence: usize,
    },
    /// A vertex has no incident triangle or has a disconnected link.
    #[error("vertex {vertex:?} has {components} link components, expected one")]
    VertexLink {
        /// Vertex with an invalid neighborhood.
        vertex: VertexId,
        /// Number of connected components in its link.
        components: usize,
    },
    /// A component's Euler characteristic exceeds the supported range.
    #[error("component containing {triangle:?} has an unrepresentable Euler characteristic")]
    EulerCharacteristic {
        /// A triangle identifying the component.
        triangle: TriangleId,
        /// Numerical error with the exact value.
        #[source]
        source: Box<TopologyError>,
    },
}

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
    /// Distinct endpoints in reference order.
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
    /// Sides forming a directed cycle through three distinct vertices.
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

/// A validated disjoint union of compact triangulated 2D manifolds with boundary.
///
/// The empty gluing is valid. Nonempty gluings may contain parallel edges
/// and nonorientable components. Identities are supplied by the caller and
/// persist through edits and serialization. Serde uses a versioned envelope
/// and validates the cell records again during deserialization.
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

type EdgeIncidence = BTreeMap<EdgeId, Vec<(TriangleId, bool)>>;
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
fn replace_cells<T: Copy + Eq>(
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
        let [first, second] = edge.endpoints;
        if first == second {
            return Err(TriangleGluingError::LoopEdge { edge: edge.id });
        }
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

    let mut links: BTreeMap<VertexId, Adjacency<EdgeId>> = BTreeMap::new();
    for triangle in &cells.triangles {
        let vertices = triangle_vertices(&cells.edges, triangle)?;
        for (index, side) in triangle.sides.iter().enumerate() {
            incidence
                .entry(side.edge)
                .or_default()
                .push((triangle.id, side.reversed));
            let previous = triangle.sides[(index + 2) % 3].edge;
            connect(
                links.entry(vertices[index]).or_default(),
                previous,
                side.edge,
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
        // A link node is an edge occurrence at this vertex. Its degree equals
        // the edge incidence, so a connected link is a path or cycle.
        let components = links.get(&vertex).map_or(0, component_count);
        if components != 1 {
            return Err(TriangleGluingError::VertexLink { vertex, components });
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
    let [a, b] = first?;
    let [next_b, c] = second?;
    let [next_c, next_a] = third?;
    if [a, b, c] != [next_a, next_b, next_c] {
        return Err(TriangleGluingError::OpenTriangle {
            triangle: triangle.id,
        });
    }
    Ok([a, b, c])
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
            for side in triangles[&id].sides {
                let uses = &incidence[&side.edge];
                if seen_edges.insert(side.edge) {
                    let [first, second] = edges[&side.edge].endpoints;
                    vertices.extend([first, second]);
                    if uses.len() == 1 {
                        connect(&mut boundary, first, second);
                    }
                }
                for &(neighbor, reversed) in uses {
                    if neighbor == id {
                        continue;
                    }
                    // Equal reference directions require exactly one face flip.
                    let required = flip ^ (side.reversed == reversed);
                    match orientation.entry(neighbor) {
                        Entry::Occupied(assigned) => orientable &= *assigned.get() == required,
                        Entry::Vacant(unassigned) => {
                            unassigned.insert(required);
                            pending.push((neighbor, required));
                        }
                    }
                }
            }
        }
        members.sort_unstable();
        let vertex_count = vertices.len();
        let edge_count = seen_edges.len();
        let euler_characteristic = euler_characteristic(&FVector {
            by_dim: vec![vertex_count, edge_count, members.len()],
        })
        .map_err(|source| TriangleGluingError::EulerCharacteristic {
            triangle: seed,
            source: Box::new(source),
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
            return Err(serde::de::Error::custom(
                "unsupported triangle gluing schema version",
            ));
        }
        Self::try_new(wire.cells).map_err(serde::de::Error::custom)
    }
}
