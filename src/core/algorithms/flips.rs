//! Bistellar flip operations for triangulations.
//!
//! This module implements **Pachner (bistellar) moves** for Delaunay repair and topology editing.
//!
//! We use the **standard convention** where a k-move (1 ≤ k ≤ D+1) replaces the star of a
//! `(D+1−k)`-simplex with the star of the complementary `(k−1)`-simplex:
//! - removed D-simplices = k
//! - inserted D-simplices = D+2−k
//! - removed-face dimension = D+1−k
//! - inserted-face dimension = k−1
//!
//! Implemented moves:
//! - k=1: cell split/merge (1↔(D+1)), valid for D≥1
//! - k=2: facet flips (2↔D), valid for D≥2
//! - k=3: ridge flips (3↔(D−1)), valid for D≥3
//!
//! We represent higher-order flips via **inverse** directions:
//! - In D=4, inverse k=2 is the k=4 flip (4↔2).
//! - In D=5, inverse k=2 is k=5 (5↔2) and inverse k=3 is k=4 (4↔3).
//!
//! Delaunay repair uses k=2 (facets) and k=3 (ridges) only; k=1 is exposed for
//! topological editing via the public API.
//!
//! # References
//! - Edelsbrunner & Shah (1996) - "Incremental Topological Flipping Works for Regular Triangulations"
//! - Bistellar flips implementation notebook (Warp Drive)

use slotmap::Key;
use std::collections::VecDeque;
use std::sync::Arc;

use thiserror::Error;

use crate::core::algorithms::incremental_insertion::wire_cavity_neighbors;
use crate::core::cell::{Cell, CellValidationError};
use crate::core::collections::{
    CellKeyBuffer, FastHashMap, FastHashSet, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer,
};
use crate::core::edge::EdgeKey;
use crate::core::facet::{AllFacetsIter, FacetHandle, facet_key_from_vertices};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation_data_structure::{CellKey, Tds, VertexKey};
use crate::core::vertex::Vertex;
use crate::geometry::kernel::Kernel;
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::CoordinateScalar;

/// Bistellar flip kind descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BistellarFlipKind {
    /// Number of simplices being replaced on the current side (k).
    pub k: usize,
    /// Dimension of the triangulation (D).
    pub d: usize,
}

fn apply_bistellar_flip_with_k<K, U, V, const D: usize>(
    tds: &mut Tds<K::Scalar, U, V, D>,
    kernel: &K,
    k_move: usize,
    removed_face_vertices: &[VertexKey],
    inserted_face_vertices: &[VertexKey],
    removed_cells: &CellKeyBuffer,
    direction: FlipDirection,
) -> Result<FlipInfo<D>, FlipError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    if k_move == 0 || k_move > D + 1 {
        return Err(FlipError::InvalidFlipContext {
            message: format!("k must be in 1..=D+1 (k={k_move}, D={D})"),
        });
    }

    let expected_removed_face = D + 2 - k_move;
    if removed_face_vertices.len() != expected_removed_face {
        return Err(FlipError::InvalidFlipContext {
            message: format!(
                "removed-face must have {expected_removed_face} vertices, got {}",
                removed_face_vertices.len()
            ),
        });
    }
    if inserted_face_vertices.len() != k_move {
        return Err(FlipError::InvalidFlipContext {
            message: format!(
                "inserted-face must have {k_move} vertices, got {}",
                inserted_face_vertices.len()
            ),
        });
    }
    if removed_cells.len() != k_move {
        return Err(FlipError::InvalidFlipContext {
            message: format!(
                "removed_cells must have {k_move} entries, got {}",
                removed_cells.len()
            ),
        });
    }
    if removed_face_vertices
        .iter()
        .any(|v| inserted_face_vertices.contains(v))
    {
        return Err(FlipError::InvalidFlipContext {
            message: "removed-face and inserted-face must be disjoint".to_string(),
        });
    }

    let mut new_cells = CellKeyBuffer::new();
    let mut new_cell_vertices: SmallBuffer<
        SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
        MAX_PRACTICAL_DIMENSION_SIZE,
    > = SmallBuffer::with_capacity(removed_face_vertices.len());

    for &omit in removed_face_vertices {
        let mut vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(D + 1);
        vertices.extend_from_slice(inserted_face_vertices);
        for &v in removed_face_vertices {
            if v != omit {
                vertices.push(v);
            }
        }

        if flip_would_duplicate_cell_any(tds, &vertices, removed_cells) {
            return Err(FlipError::DuplicateCell);
        }
        if flip_would_create_nonmanifold_facets_any(
            tds,
            &vertices,
            removed_cells,
            inserted_face_vertices,
        ) {
            return Err(FlipError::NonManifoldFacet);
        }

        let points = vertices_to_points(tds, &vertices)?;
        let orientation = kernel
            .orientation(&points)
            .map_err(|e| FlipError::PredicateFailure {
                message: format!("orientation failed for flip cell: {e}"),
            })?;
        if orientation == 0 {
            return Err(FlipError::DegenerateCell);
        }

        new_cell_vertices.push(vertices);
    }

    for vertices in new_cell_vertices {
        let cell = Cell::new(vertices, None)?;
        let cell_key = tds
            .insert_cell_with_mapping(cell)
            .map_err(|e| FlipError::TdsMutation {
                message: e.to_string(),
            })?;
        new_cells.push(cell_key);
    }

    wire_cavity_neighbors(tds, &new_cells, Some(removed_cells)).map_err(|e| {
        FlipError::NeighborWiring {
            message: e.to_string(),
        }
    })?;

    tds.remove_cells_by_keys(removed_cells);

    Ok(FlipInfo {
        kind: BistellarFlipKind { k: k_move, d: D },
        direction,
        removed_cells: removed_cells.iter().copied().collect(),
        new_cells,
        removed_face_vertices: removed_face_vertices.iter().copied().collect(),
        inserted_face_vertices: inserted_face_vertices.iter().copied().collect(),
    })
}
/// Check whether a k=3 ridge violates the local Delaunay condition.
///
/// # Errors
///
/// Returns a [`FlipError`] if any referenced cell/vertex is missing or a predicate
/// evaluation fails.
pub fn is_delaunay_violation_k3<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    context: &FlipContext<D, 3>,
) -> Result<bool, FlipError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    delaunay_violation_k3_for_ridge(
        tds,
        kernel,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
    )
}

/// Apply a generic k-move (no Delaunay check).
///
/// # Errors
///
/// Returns a [`FlipError`] if the flip would be degenerate, duplicate an existing cell,
/// create non-manifold topology, if predicate evaluation fails, or if underlying TDS
/// mutations fail.
pub fn apply_bistellar_flip<K, U, V, const D: usize, const K_MOVE: usize>(
    tds: &mut Tds<K::Scalar, U, V, D>,
    kernel: &K,
    context: &FlipContext<D, K_MOVE>,
) -> Result<FlipInfo<D>, FlipError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    apply_bistellar_flip_with_k(
        tds,
        kernel,
        K_MOVE,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
        &context.removed_cells,
        context.direction,
    )
}

fn apply_bistellar_flip_dynamic<K, U, V, const D: usize>(
    tds: &mut Tds<K::Scalar, U, V, D>,
    kernel: &K,
    k_move: usize,
    context: &FlipContextDyn<D>,
) -> Result<FlipInfo<D>, FlipError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    apply_bistellar_flip_with_k(
        tds,
        kernel,
        k_move,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
        &context.removed_cells,
        context.direction,
    )
}

/// Direction of a bistellar flip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlipDirection {
    /// Forward (k → D+2−k).
    Forward,
    /// Inverse (D+2−k → k).
    Inverse,
}

impl FlipDirection {
    /// Return the opposite direction.
    #[must_use]
    pub const fn inverse(self) -> Self {
        match self {
            Self::Forward => Self::Inverse,
            Self::Inverse => Self::Forward,
        }
    }
}

impl BistellarFlipKind {
    /// Construct a k=1 flip kind for the given dimension.
    #[must_use]
    pub const fn k1(d: usize) -> Self {
        Self { k: 1, d }
    }
    /// Construct a k=2 flip kind for the given dimension.
    #[must_use]
    pub const fn k2(d: usize) -> Self {
        Self { k: 2, d }
    }

    /// Construct a k=3 flip kind for the given dimension.
    #[must_use]
    pub const fn k3(d: usize) -> Self {
        Self { k: 3, d }
    }

    /// Construct the inverse flip kind (k' = D + 2 - k).
    #[must_use]
    pub const fn inverse(self) -> Self {
        Self {
            k: self.d + 2 - self.k,
            d: self.d,
        }
    }
}

/// Const-generic move marker for Pachner k-moves.
#[derive(Debug, Clone, Copy)]
pub struct ConstK<const K: usize>;

/// Const-generic descriptor for a Pachner move in dimension `D`.
pub trait BistellarMove<const D: usize> {
    /// Number of removed D-simplices (k).
    const K: usize;
}

impl<const D: usize, const K: usize> BistellarMove<D> for ConstK<K> {
    const K: usize = K;
}

/// Errors that can occur during bistellar flips or repair.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FlipError {
    /// Flips are not supported for this dimension.
    #[error("Bistellar flip not supported for D={dimension}")]
    UnsupportedDimension {
        /// Dimension of the triangulation.
        dimension: usize,
    },
    /// The facet is on the boundary (no adjacent cell).
    #[error("Facet {facet:?} is on the boundary (no neighbor)")]
    BoundaryFacet {
        /// Facet handle.
        facet: FacetHandle,
    },
    /// The referenced cell was not found.
    #[error("Cell not found: {cell_key:?}")]
    MissingCell {
        /// Missing cell key.
        cell_key: CellKey,
    },
    /// The referenced vertex was not found.
    #[error("Vertex not found: {vertex_key:?}")]
    MissingVertex {
        /// Missing vertex key.
        vertex_key: VertexKey,
    },
    /// The neighbor cell across the facet is missing.
    #[error("Neighbor cell {neighbor_key:?} not found for facet {facet:?}")]
    MissingNeighbor {
        /// Facet handle.
        facet: FacetHandle,
        /// Missing neighbor key.
        neighbor_key: CellKey,
    },
    /// Facet adjacency information is inconsistent.
    #[error("Facet adjacency mismatch between cell {cell_key:?} and neighbor {neighbor_key:?}")]
    InvalidFacetAdjacency {
        /// Cell key.
        cell_key: CellKey,
        /// Neighbor cell key.
        neighbor_key: CellKey,
    },
    /// The facet index is out of bounds for the cell.
    #[error(
        "Facet index {facet_index} out of bounds for cell {cell_key:?} with {vertex_count} vertices"
    )]
    InvalidFacetIndex {
        /// Cell key.
        cell_key: CellKey,
        /// Facet index.
        facet_index: u8,
        /// Vertex count for the cell.
        vertex_count: usize,
    },
    /// Ridge indices are invalid for the cell.
    #[error(
        "Ridge indices ({omit_a}, {omit_b}) out of bounds for cell {cell_key:?} with {vertex_count} vertices"
    )]
    InvalidRidgeIndex {
        /// Cell key.
        cell_key: CellKey,
        /// First omitted index.
        omit_a: u8,
        /// Second omitted index.
        omit_b: u8,
        /// Vertex count for the cell.
        vertex_count: usize,
    },
    /// Ridge adjacency information is inconsistent.
    #[error("Ridge adjacency mismatch for cell {cell_key:?}")]
    InvalidRidgeAdjacency {
        /// Cell key.
        cell_key: CellKey,
    },
    /// Ridge has an invalid multiplicity for k=3 flips.
    #[error("Ridge has invalid multiplicity {found}, expected 3")]
    InvalidRidgeMultiplicity {
        /// Number of incident cells found.
        found: usize,
    },
    /// Edge has an invalid multiplicity for inverse k=2 flips.
    #[error("Edge has invalid multiplicity {found}, expected {expected}")]
    InvalidEdgeMultiplicity {
        /// Number of incident cells found.
        found: usize,
        /// Expected multiplicity for the dimension.
        expected: usize,
    },
    /// Triangle has an invalid multiplicity for inverse k=3 flips.
    #[error("Triangle has invalid multiplicity {found}, expected {expected}")]
    InvalidTriangleMultiplicity {
        /// Number of incident cells found.
        found: usize,
        /// Expected multiplicity for the dimension.
        expected: usize,
    },
    /// Edge adjacency information is inconsistent.
    #[error("Edge adjacency mismatch: {message}")]
    InvalidEdgeAdjacency {
        /// Context message.
        message: String,
    },
    /// Triangle adjacency information is inconsistent.
    #[error("Triangle adjacency mismatch: {message}")]
    InvalidTriangleAdjacency {
        /// Context message.
        message: String,
    },
    /// Vertex star has an invalid multiplicity for inverse k=1 flips.
    #[error("Vertex star has invalid multiplicity {found}, expected {expected}")]
    InvalidVertexMultiplicity {
        /// Number of incident cells found.
        found: usize,
        /// Expected multiplicity for the dimension.
        expected: usize,
    },
    /// Vertex adjacency information is inconsistent.
    #[error("Vertex adjacency mismatch: {message}")]
    InvalidVertexAdjacency {
        /// Context message.
        message: String,
    },
    /// Flip direction is not supported for this operation.
    #[error("Flip direction {direction:?} is not supported for this operation")]
    InvalidFlipDirection {
        /// Requested direction.
        direction: FlipDirection,
    },
    /// Flip context is inconsistent with the requested move.
    #[error("Flip context invalid: {message}")]
    InvalidFlipContext {
        /// Context message.
        message: String,
    },
    /// Geometric predicate failed.
    #[error("Geometric predicate failed: {message}")]
    PredicateFailure {
        /// Error message from predicate evaluation.
        message: String,
    },
    /// Flip would create a degenerate cell (zero orientation).
    #[error("Flip would create a degenerate cell (zero orientation)")]
    DegenerateCell,
    /// Flip would create a duplicate cell.
    #[error("Flip would create a duplicate cell")]
    DuplicateCell,
    /// Flip would create a non-manifold facet.
    #[error("Flip would create a non-manifold facet")]
    NonManifoldFacet,
    /// Cell creation failed.
    #[error(transparent)]
    CellCreation(#[from] CellValidationError),
    /// Neighbor wiring failed during flip application.
    #[error("Neighbor wiring failed: {message}")]
    NeighborWiring {
        /// Error message from neighbor wiring.
        message: String,
    },
    /// TDS mutation failed.
    #[error("TDS mutation failed: {message}")]
    TdsMutation {
        /// Error message from TDS mutation.
        message: String,
    },
}

/// Information about a successful flip.
#[derive(Debug, Clone)]
pub struct FlipInfo<const D: usize> {
    /// Flip kind (k, d).
    pub kind: BistellarFlipKind,
    /// Flip direction.
    pub direction: FlipDirection,
    /// Cells removed by the flip.
    pub removed_cells: CellKeyBuffer,
    /// Newly created cells.
    pub new_cells: CellKeyBuffer,
    /// The removed-face simplex (shared by removed cells).
    pub removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// The inserted-face simplex (complementary simplex).
    pub inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
}

/// Flip decision result (used by k=2 repair).
#[derive(Debug, Clone)]
pub enum FlipDecision<const D: usize> {
    /// The facet was flipped.
    Flipped(Arc<FlipInfo<D>>),
    /// No flip was performed (facet already locally Delaunay).
    NoFlip,
}

/// Const-generic flip context for a k-move (forward or inverse).
#[derive(Debug, Clone)]
pub struct FlipContext<const D: usize, const K: usize> {
    /// Vertices of the removed-face simplex (dimension D+1−K).
    pub removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// Vertices of the inserted-face simplex (dimension K−1).
    pub inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// Cells removed by the flip (count = K).
    pub removed_cells: CellKeyBuffer,
    /// Flip direction (forward/inverse).
    pub direction: FlipDirection,
}

impl<const D: usize, const K: usize> FlipContext<D, K> {
    /// Returns the flip direction for this context.
    #[must_use]
    pub const fn direction(&self) -> FlipDirection {
        self.direction
    }
}

/// Runtime-k flip context for moves where k depends on D.
#[derive(Debug, Clone)]
pub struct FlipContextDyn<const D: usize> {
    /// Vertices of the removed-face simplex (dimension D+1−k).
    pub removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// Vertices of the inserted-face simplex (dimension k−1).
    pub inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// Cells removed by the flip (count = k).
    pub removed_cells: CellKeyBuffer,
    /// Flip direction (forward/inverse).
    pub direction: FlipDirection,
}

impl<const D: usize> FlipContextDyn<D> {
    /// Returns the flip direction for this context.
    #[must_use]
    pub const fn direction(&self) -> FlipDirection {
        self.direction
    }
}

/// Canonical handle to a triangle (three vertices).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TriangleHandle {
    v0: VertexKey,
    v1: VertexKey,
    v2: VertexKey,
}

impl TriangleHandle {
    /// Create a canonical triangle handle with ordered vertex keys.
    #[must_use]
    pub fn new(a: VertexKey, b: VertexKey, c: VertexKey) -> Self {
        let mut verts = [a, b, c];
        verts.sort_unstable_by_key(|v| v.data().as_ffi());
        Self {
            v0: verts[0],
            v1: verts[1],
            v2: verts[2],
        }
    }

    /// Return the triangle vertices.
    #[must_use]
    pub const fn vertices(self) -> [VertexKey; 3] {
        [self.v0, self.v1, self.v2]
    }
}
/// Lightweight handle to a ridge (codimension-2 face) within a cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RidgeHandle {
    cell_key: CellKey,
    omit_a: u8,
    omit_b: u8,
}

impl RidgeHandle {
    /// Creates a new ridge handle by specifying the two omitted vertex indices.
    #[must_use]
    pub const fn new(cell_key: CellKey, omit_a: u8, omit_b: u8) -> Self {
        if omit_a <= omit_b {
            Self {
                cell_key,
                omit_a,
                omit_b,
            }
        } else {
            Self {
                cell_key,
                omit_a: omit_b,
                omit_b: omit_a,
            }
        }
    }

    /// Returns the cell key.
    #[must_use]
    pub const fn cell_key(&self) -> CellKey {
        self.cell_key
    }

    /// Returns the first omitted index.
    #[must_use]
    pub const fn omit_a(&self) -> u8 {
        self.omit_a
    }

    /// Returns the second omitted index.
    #[must_use]
    pub const fn omit_b(&self) -> u8 {
        self.omit_b
    }
}

/// Statistics for flip-based Delaunay repair.
#[derive(Debug, Clone, Default)]
pub struct DelaunayRepairStats {
    /// Number of queued items checked (facets, ridges, edges, triangles).
    pub facets_checked: usize,
    /// Number of flips performed.
    pub flips_performed: usize,
    /// Maximum queue length observed.
    pub max_queue_len: usize,
}

/// Errors that can occur during flip-based Delaunay repair.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DelaunayRepairError {
    /// Repair did not converge within the flip budget.
    #[error(
        "Delaunay repair failed to converge after {max_flips} flips (checked {facets_checked} facets)"
    )]
    NonConvergent {
        /// Maximum flips allowed.
        max_flips: usize,
        /// Number of facets checked.
        facets_checked: usize,
    },
    /// Underlying flip error.
    #[error(transparent)]
    Flip(#[from] FlipError),
}

/// Build flip context for a k=2 (facet) flip.
///
/// # Errors
///
/// Returns a [`FlipError`] if the facet is invalid, lies on the boundary, references
/// missing cells/vertices, or the adjacency data is inconsistent.
pub fn build_k2_flip_context<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    facet: FacetHandle,
) -> Result<FlipContext<D, 2>, FlipError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    if D < 2 {
        return Err(FlipError::UnsupportedDimension { dimension: D });
    }

    let cell_a_key = facet.cell_key();
    let cell_a = tds.get_cell(cell_a_key).ok_or(FlipError::MissingCell {
        cell_key: cell_a_key,
    })?;

    let facet_index_a = usize::from(facet.facet_index());
    let vertex_count = cell_a.number_of_vertices();
    if facet_index_a >= vertex_count {
        return Err(FlipError::InvalidFacetIndex {
            cell_key: cell_a_key,
            facet_index: facet.facet_index(),
            vertex_count,
        });
    }

    let neighbor_key = cell_a
        .neighbors()
        .and_then(|n| n.get(facet_index_a).copied().flatten())
        .ok_or(FlipError::BoundaryFacet { facet })?;

    if !tds.contains_cell(neighbor_key) {
        return Err(FlipError::MissingNeighbor {
            facet,
            neighbor_key,
        });
    }

    let cell_b = tds.get_cell(neighbor_key).ok_or(FlipError::MissingCell {
        cell_key: neighbor_key,
    })?;

    let Some(facet_index_b) = cell_a.mirror_facet_index(facet_index_a, cell_b) else {
        return Err(FlipError::InvalidFacetAdjacency {
            cell_key: cell_a_key,
            neighbor_key,
        });
    };

    let opposite_a = cell_a.vertices()[facet_index_a];
    let opposite_b = cell_b.vertices()[facet_index_b];

    let shared_facet = facet_vertices_from_cell(cell_a, facet_index_a);

    if shared_facet.len() != D {
        return Err(FlipError::InvalidFacetAdjacency {
            cell_key: cell_a_key,
            neighbor_key,
        });
    }

    if shared_facet.contains(&opposite_a)
        || shared_facet.contains(&opposite_b)
        || opposite_a == opposite_b
    {
        return Err(FlipError::InvalidFacetAdjacency {
            cell_key: cell_a_key,
            neighbor_key,
        });
    }

    for &v in &shared_facet {
        if !cell_b.contains_vertex(v) {
            return Err(FlipError::InvalidFacetAdjacency {
                cell_key: cell_a_key,
                neighbor_key,
            });
        }
    }

    let removed_cells: CellKeyBuffer = [cell_a_key, neighbor_key].into_iter().collect();
    let mut inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(2);
    inserted_face_vertices.push(opposite_a);
    inserted_face_vertices.push(opposite_b);

    Ok(FlipContext {
        removed_face_vertices: shared_facet,
        inserted_face_vertices,
        removed_cells,
        direction: FlipDirection::Forward,
    })
}

/// Build inverse k=2 flip context from an edge and its incident cells.
///
/// # Errors
///
/// Returns a [`FlipError`] if the edge is invalid, references missing vertices/cells,
/// or the adjacency data is inconsistent.
pub fn build_k2_flip_context_from_edge<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    edge: EdgeKey,
) -> Result<FlipContextDyn<D>, FlipError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    if D < 3 {
        return Err(FlipError::UnsupportedDimension { dimension: D });
    }

    let (v0, v1) = edge.endpoints();
    if v0 == v1 {
        return Err(FlipError::InvalidEdgeAdjacency {
            message: "edge endpoints must be distinct".to_string(),
        });
    }

    if tds.get_vertex_by_key(v0).is_none() {
        return Err(FlipError::MissingVertex { vertex_key: v0 });
    }
    if tds.get_vertex_by_key(v1).is_none() {
        return Err(FlipError::MissingVertex { vertex_key: v1 });
    }

    let cells_v0 = tds.find_cells_containing_vertex_by_key(v0);
    let cells_v1 = tds.find_cells_containing_vertex_by_key(v1);
    let (small, large) = if cells_v0.len() <= cells_v1.len() {
        (&cells_v0, &cells_v1)
    } else {
        (&cells_v1, &cells_v0)
    };

    let mut removed_cells: CellKeyBuffer = CellKeyBuffer::new();
    for &cell_key in small {
        if large.contains(&cell_key) {
            removed_cells.push(cell_key);
        }
    }

    if removed_cells.len() != D {
        return Err(FlipError::InvalidEdgeMultiplicity {
            found: removed_cells.len(),
            expected: D,
        });
    }

    let mut counts: FastHashMap<VertexKey, usize> = FastHashMap::default();
    for &cell_key in &removed_cells {
        let cell = tds
            .get_cell(cell_key)
            .ok_or(FlipError::MissingCell { cell_key })?;
        if !cell.contains_vertex(v0) || !cell.contains_vertex(v1) {
            return Err(FlipError::InvalidEdgeAdjacency {
                message: format!("cell {cell_key:?} does not contain edge vertices"),
            });
        }
        for &vk in cell.vertices() {
            if vk != v0 && vk != v1 {
                *counts.entry(vk).or_insert(0) += 1;
            }
        }
    }

    if counts.len() != D || !counts.values().all(|&count| count == D - 1) {
        return Err(FlipError::InvalidEdgeAdjacency {
            message: format!(
                "edge star must have {D} distinct opposite vertices each appearing {expected} times",
                expected = D - 1
            ),
        });
    }

    let mut inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        counts.keys().copied().collect();
    inserted_face_vertices.sort_unstable_by_key(|v| v.data().as_ffi());

    let mut removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(2);
    removed_face_vertices.push(v0);
    removed_face_vertices.push(v1);

    Ok(FlipContextDyn {
        removed_face_vertices,
        inserted_face_vertices,
        removed_cells,
        direction: FlipDirection::Inverse,
    })
}

fn build_k1_forward_context_from_cell<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    cell_key: CellKey,
    inserted_vertex: VertexKey,
) -> Result<FlipContext<D, 1>, FlipError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    if D < 1 {
        return Err(FlipError::UnsupportedDimension { dimension: D });
    }

    let cell = tds
        .get_cell(cell_key)
        .ok_or(FlipError::MissingCell { cell_key })?;
    if tds.get_vertex_by_key(inserted_vertex).is_none() {
        return Err(FlipError::MissingVertex {
            vertex_key: inserted_vertex,
        });
    }

    let removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        cell.vertices().iter().copied().collect();
    let mut inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(1);
    inserted_face_vertices.push(inserted_vertex);

    let removed_cells: CellKeyBuffer = std::iter::once(cell_key).collect();

    Ok(FlipContext {
        removed_face_vertices,
        inserted_face_vertices,
        removed_cells,
        direction: FlipDirection::Forward,
    })
}

/// Build inverse k=1 flip context from a vertex and its incident cells.
///
/// # Errors
///
/// Returns a [`FlipError`] if the vertex is missing, its incident cell count is
/// not D+1, or the adjacency data is inconsistent.
pub fn build_k1_inverse_context<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    vertex_key: VertexKey,
) -> Result<FlipContextDyn<D>, FlipError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    if D < 1 {
        return Err(FlipError::UnsupportedDimension { dimension: D });
    }

    if tds.get_vertex_by_key(vertex_key).is_none() {
        return Err(FlipError::MissingVertex { vertex_key });
    }

    let removed_cells = tds.find_cells_containing_vertex_by_key(vertex_key);
    let expected = D + 1;
    if removed_cells.len() != expected {
        return Err(FlipError::InvalidVertexMultiplicity {
            found: removed_cells.len(),
            expected,
        });
    }

    let mut counts: FastHashMap<VertexKey, usize> = FastHashMap::default();
    let mut removed_cells_buf: CellKeyBuffer = CellKeyBuffer::new();
    for &cell_key in &removed_cells {
        let cell = tds
            .get_cell(cell_key)
            .ok_or(FlipError::MissingCell { cell_key })?;
        if !cell.contains_vertex(vertex_key) {
            return Err(FlipError::InvalidVertexAdjacency {
                message: format!("cell {cell_key:?} does not contain vertex"),
            });
        }
        removed_cells_buf.push(cell_key);
        for &vk in cell.vertices() {
            if vk != vertex_key {
                *counts.entry(vk).or_insert(0) += 1;
            }
        }
    }

    if counts.len() != expected || !counts.values().all(|&count| count == D) {
        return Err(FlipError::InvalidVertexAdjacency {
            message: format!(
                "vertex star must have {expected} link vertices each appearing {D} times"
            ),
        });
    }

    let mut inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        counts.keys().copied().collect();
    inserted_face_vertices.sort_unstable_by_key(|v| v.data().as_ffi());

    let mut removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(1);
    removed_face_vertices.push(vertex_key);

    Ok(FlipContextDyn {
        removed_face_vertices,
        inserted_face_vertices,
        removed_cells: removed_cells_buf,
        direction: FlipDirection::Inverse,
    })
}

fn delaunay_violation_k2_for_facet<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    facet_vertices: &[VertexKey],
    opposite_a: VertexKey,
    opposite_b: VertexKey,
) -> Result<bool, FlipError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    if facet_vertices.len() != D {
        return Err(FlipError::InvalidFlipContext {
            message: format!(
                "k=2 facet must have {D} vertices, got {}",
                facet_vertices.len()
            ),
        });
    }
    if facet_vertices.contains(&opposite_a)
        || facet_vertices.contains(&opposite_b)
        || opposite_a == opposite_b
    {
        return Err(FlipError::InvalidFlipContext {
            message: "k=2 opposites must be distinct and not in the facet".to_string(),
        });
    }

    let mut cell_vertices: [SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>; 2] = [
        SmallBuffer::with_capacity(D + 1),
        SmallBuffer::with_capacity(D + 1),
    ];
    for vertices in &mut cell_vertices {
        vertices.extend_from_slice(facet_vertices);
    }
    cell_vertices[0].push(opposite_a);
    cell_vertices[1].push(opposite_b);

    let points_a = vertices_to_points(tds, &cell_vertices[0])?;
    let points_b = vertices_to_points(tds, &cell_vertices[1])?;

    let opposite_point_a = tds
        .get_vertex_by_key(opposite_a)
        .ok_or(FlipError::MissingVertex {
            vertex_key: opposite_a,
        })?
        .point();
    let opposite_point_b = tds
        .get_vertex_by_key(opposite_b)
        .ok_or(FlipError::MissingVertex {
            vertex_key: opposite_b,
        })?
        .point();

    let in_a =
        kernel
            .in_sphere(&points_a, opposite_point_b)
            .map_err(|e| FlipError::PredicateFailure {
                message: format!("in_sphere failed for k=2 cell A: {e}"),
            })?;

    let in_b =
        kernel
            .in_sphere(&points_b, opposite_point_a)
            .map_err(|e| FlipError::PredicateFailure {
                message: format!("in_sphere failed for k=2 cell B: {e}"),
            })?;

    Ok(in_a > 0 || in_b > 0)
}
/// Check whether a k=2 facet violates the local Delaunay condition.
///
/// # Errors
///
/// Returns a [`FlipError`] if any referenced cell/vertex is missing or a predicate
/// evaluation fails.
pub fn is_delaunay_violation_k2<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    context: &FlipContext<D, 2>,
) -> Result<bool, FlipError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    if context.inserted_face_vertices.len() != 2 {
        return Err(FlipError::InvalidFlipContext {
            message: format!(
                "k=2 inserted-face must have 2 vertices, got {}",
                context.inserted_face_vertices.len()
            ),
        });
    }
    let opposite_a = context.inserted_face_vertices[0];
    let opposite_b = context.inserted_face_vertices[1];
    delaunay_violation_k2_for_facet(
        tds,
        kernel,
        &context.removed_face_vertices,
        opposite_a,
        opposite_b,
    )
}

/// Apply a k=2 bistellar flip (no Delaunay check).
///
/// # Errors
///
/// Returns a [`FlipError`] if the flip would be degenerate, duplicate an existing cell,
/// create non-manifold topology, if predicate evaluation fails, or if underlying TDS
/// mutations fail.
pub fn apply_bistellar_flip_k2<K, U, V, const D: usize>(
    tds: &mut Tds<K::Scalar, U, V, D>,
    kernel: &K,
    context: &FlipContext<D, 2>,
) -> Result<FlipInfo<D>, FlipError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    apply_bistellar_flip::<K, U, V, D, 2>(tds, kernel, context)
}

/// Attempt a k=2 bistellar flip for Delaunay repair.
///
/// # Errors
///
/// Returns a [`FlipError`] if context construction, predicate evaluation, or flip
/// application fails.
pub fn try_bistellar_flip_k2<K, U, V, const D: usize>(
    tds: &mut Tds<K::Scalar, U, V, D>,
    kernel: &K,
    facet: FacetHandle,
) -> Result<FlipDecision<D>, FlipError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let context = build_k2_flip_context(tds, facet)?;
    if !is_delaunay_violation_k2(tds, kernel, &context)? {
        return Ok(FlipDecision::NoFlip);
    }
    let info = apply_bistellar_flip_k2(tds, kernel, &context)?;
    Ok(FlipDecision::Flipped(Arc::new(info)))
}
/// Build flip context for a k=3 (ridge) flip.
///
/// # Errors
///
/// Returns a [`FlipError`] if the ridge is invalid, references missing cells/vertices,
/// or the adjacency data is inconsistent.
pub fn build_k3_flip_context<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    ridge: RidgeHandle,
) -> Result<FlipContext<D, 3>, FlipError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    if D < 3 {
        return Err(FlipError::UnsupportedDimension { dimension: D });
    }

    let cell_key = ridge.cell_key();
    let cell = tds
        .get_cell(cell_key)
        .ok_or(FlipError::MissingCell { cell_key })?;

    let vertex_count = cell.number_of_vertices();
    let omit_a = usize::from(ridge.omit_a());
    let omit_b = usize::from(ridge.omit_b());
    if omit_a >= vertex_count || omit_b >= vertex_count || omit_a == omit_b {
        return Err(FlipError::InvalidRidgeIndex {
            cell_key,
            omit_a: ridge.omit_a(),
            omit_b: ridge.omit_b(),
            vertex_count,
        });
    }

    let ridge_vertices = ridge_vertices_from_cell(cell, omit_a, omit_b);
    if ridge_vertices.len() != D - 1 {
        return Err(FlipError::InvalidRidgeAdjacency { cell_key });
    }

    let cells = collect_cells_around_ridge(tds, cell_key, &ridge_vertices)?;
    if cells.len() != 3 {
        return Err(FlipError::InvalidRidgeMultiplicity { found: cells.len() });
    }

    let mut opposite_counts: FastHashMap<VertexKey, usize> = FastHashMap::default();
    let mut extras_per_cell: Vec<[VertexKey; 2]> = Vec::with_capacity(3);

    for &ck in &cells {
        let cell = tds
            .get_cell(ck)
            .ok_or(FlipError::MissingCell { cell_key: ck })?;
        let extras = cell_extras_for_ridge(ck, cell, &ridge_vertices)?;
        if extras.len() != 2 {
            return Err(FlipError::InvalidRidgeAdjacency { cell_key: ck });
        }

        for &v in &extras {
            *opposite_counts.entry(v).or_insert(0) += 1;
        }
        let extras_pair: [VertexKey; 2] = extras
            .as_slice()
            .try_into()
            .map_err(|_| FlipError::InvalidRidgeAdjacency { cell_key: ck })?;
        extras_per_cell.push(extras_pair);
    }

    if opposite_counts.len() != 3 || !opposite_counts.values().all(|&count| count == 2) {
        return Err(FlipError::InvalidRidgeAdjacency { cell_key });
    }

    let mut opposite_vertices: SmallBuffer<VertexKey, 3> =
        opposite_counts.keys().copied().collect();
    opposite_vertices.sort_unstable();
    let opposite_vertices: [VertexKey; 3] = opposite_vertices
        .as_slice()
        .try_into()
        .map_err(|_| FlipError::InvalidRidgeAdjacency { cell_key })?;

    for extras in &extras_per_cell {
        let _missing = missing_opposite_for_cell(extras, &opposite_vertices)
            .ok_or(FlipError::InvalidRidgeAdjacency { cell_key })?;
    }

    let mut inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(3);
    inserted_face_vertices.extend(opposite_vertices);

    Ok(FlipContext {
        removed_face_vertices: ridge_vertices,
        inserted_face_vertices,
        removed_cells: cells,
        direction: FlipDirection::Forward,
    })
}

/// Build inverse k=3 flip context from a triangle and its incident cells.
///
/// # Errors
///
/// Returns a [`FlipError`] if the triangle is invalid, references missing vertices/cells,
/// or the adjacency data is inconsistent.
pub fn build_k3_flip_context_from_triangle<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    triangle: TriangleHandle,
) -> Result<FlipContextDyn<D>, FlipError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    if D < 4 {
        return Err(FlipError::UnsupportedDimension { dimension: D });
    }

    let [a, b, c] = triangle.vertices();
    if tds.get_vertex_by_key(a).is_none() {
        return Err(FlipError::MissingVertex { vertex_key: a });
    }
    if tds.get_vertex_by_key(b).is_none() {
        return Err(FlipError::MissingVertex { vertex_key: b });
    }
    if tds.get_vertex_by_key(c).is_none() {
        return Err(FlipError::MissingVertex { vertex_key: c });
    }

    let cells_a = tds.find_cells_containing_vertex_by_key(a);
    let cells_b = tds.find_cells_containing_vertex_by_key(b);
    let cells_c = tds.find_cells_containing_vertex_by_key(c);

    let mut removed_cells: CellKeyBuffer = CellKeyBuffer::new();
    for &cell_key in &cells_a {
        if cells_b.contains(&cell_key) && cells_c.contains(&cell_key) {
            removed_cells.push(cell_key);
        }
    }

    let expected = D - 1;
    if removed_cells.len() != expected {
        return Err(FlipError::InvalidTriangleMultiplicity {
            found: removed_cells.len(),
            expected,
        });
    }

    let mut counts: FastHashMap<VertexKey, usize> = FastHashMap::default();
    for &cell_key in &removed_cells {
        let cell = tds
            .get_cell(cell_key)
            .ok_or(FlipError::MissingCell { cell_key })?;
        if !cell.contains_vertex(a) || !cell.contains_vertex(b) || !cell.contains_vertex(c) {
            return Err(FlipError::InvalidTriangleAdjacency {
                message: format!("cell {cell_key:?} does not contain triangle vertices"),
            });
        }
        for &vk in cell.vertices() {
            if vk != a && vk != b && vk != c {
                *counts.entry(vk).or_insert(0) += 1;
            }
        }
    }

    if counts.len() != expected || !counts.values().all(|&count| count == expected - 1) {
        return Err(FlipError::InvalidTriangleAdjacency {
            message: format!(
                "triangle star must have {expected} ridge vertices each appearing {count} times",
                count = expected - 1
            ),
        });
    }

    let mut inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        counts.keys().copied().collect();
    inserted_face_vertices.sort_unstable_by_key(|v| v.data().as_ffi());

    let mut removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(3);
    removed_face_vertices.push(a);
    removed_face_vertices.push(b);
    removed_face_vertices.push(c);

    Ok(FlipContextDyn {
        removed_face_vertices,
        inserted_face_vertices,
        removed_cells,
        direction: FlipDirection::Inverse,
    })
}

fn delaunay_violation_k3_for_ridge<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    ridge_vertices: &[VertexKey],
    triangle_vertices: &[VertexKey],
) -> Result<bool, FlipError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    if triangle_vertices.len() != 3 {
        return Err(FlipError::InvalidFlipContext {
            message: format!(
                "k=3 inserted-face must have 3 vertices, got {}",
                triangle_vertices.len()
            ),
        });
    }
    if ridge_vertices.len() != D.saturating_sub(1) {
        return Err(FlipError::InvalidFlipContext {
            message: format!(
                "k=3 ridge must have {} vertices, got {}",
                D.saturating_sub(1),
                ridge_vertices.len()
            ),
        });
    }

    for &missing in triangle_vertices {
        let mut cell_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(D + 1);
        cell_vertices.extend_from_slice(ridge_vertices);
        for &v in triangle_vertices {
            if v != missing {
                cell_vertices.push(v);
            }
        }

        let points = vertices_to_points(tds, &cell_vertices)?;
        let missing_point = tds
            .get_vertex_by_key(missing)
            .ok_or(FlipError::MissingVertex {
                vertex_key: missing,
            })?
            .point();

        let in_sphere =
            kernel
                .in_sphere(&points, missing_point)
                .map_err(|e| FlipError::PredicateFailure {
                    message: format!("in_sphere failed for k=3 cell: {e}"),
                })?;
        if in_sphere > 0 {
            return Ok(true);
        }
    }

    Ok(false)
}

/// Apply a k=3 bistellar flip (no Delaunay check).
///
/// # Errors
///
/// Returns a [`FlipError`] if the flip would be degenerate, duplicate an existing cell,
/// create non-manifold topology, if predicate evaluation fails, or if underlying TDS
/// mutations fail.
pub fn apply_bistellar_flip_k3<K, U, V, const D: usize>(
    tds: &mut Tds<K::Scalar, U, V, D>,
    kernel: &K,
    context: &FlipContext<D, 3>,
) -> Result<FlipInfo<D>, FlipError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    apply_bistellar_flip::<K, U, V, D, 3>(tds, kernel, context)
}

/// Apply a forward k=1 move (cell split) by inserting a new vertex.
///
/// # Errors
///
/// Returns a [`FlipError`] if the cell is missing, the vertex cannot be inserted,
/// or the flip would be degenerate.
pub fn apply_bistellar_flip_k1<K, U, V, const D: usize>(
    tds: &mut Tds<K::Scalar, U, V, D>,
    kernel: &K,
    cell_key: CellKey,
    vertex: Vertex<K::Scalar, U, D>,
) -> Result<FlipInfo<D>, FlipError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    if D < 1 {
        return Err(FlipError::UnsupportedDimension { dimension: D });
    }

    let vertex_key =
        tds.insert_vertex_with_mapping(vertex)
            .map_err(|e| FlipError::TdsMutation {
                message: e.to_string(),
            })?;

    let context = build_k1_forward_context_from_cell(tds, cell_key, vertex_key)?;
    let result = apply_bistellar_flip::<K, U, V, D, 1>(tds, kernel, &context);

    if result.is_err()
        && let Some(inserted) = tds.get_vertex_by_key(vertex_key).copied()
    {
        let _ = tds.remove_vertex(&inserted);
    }

    result
}

/// Apply an inverse k=1 move (vertex collapse) by removing a vertex whose star
/// is a simplex.
///
/// # Errors
///
/// Returns a [`FlipError`] if the vertex star is invalid or the flip would be degenerate.
pub fn apply_bistellar_flip_k1_inverse<K, U, V, const D: usize>(
    tds: &mut Tds<K::Scalar, U, V, D>,
    kernel: &K,
    vertex_key: VertexKey,
) -> Result<FlipInfo<D>, FlipError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    if D < 1 {
        return Err(FlipError::UnsupportedDimension { dimension: D });
    }

    let context = build_k1_inverse_context(tds, vertex_key)?;
    let info = apply_bistellar_flip_dynamic(tds, kernel, D + 1, &context)?;

    if let Some(vertex) = tds.get_vertex_by_key(vertex_key).copied() {
        let _ = tds.remove_vertex(&vertex);
    }

    Ok(info)
}

/// Repair Delaunay violations using a k=2 flip queue.
///
/// # Errors
///
/// Returns a [`DelaunayRepairError`] if the repair fails to converge or an underlying
/// flip operation encounters an unrecoverable error.
pub fn repair_delaunay_with_flips_k2<K, U, V, const D: usize>(
    tds: &mut Tds<K::Scalar, U, V, D>,
    kernel: &K,
    seed_cells: Option<&[CellKey]>,
) -> Result<DelaunayRepairStats, DelaunayRepairError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    if D < 2 {
        return Err(FlipError::UnsupportedDimension { dimension: D }.into());
    }

    let max_flips = default_max_flips::<D>(tds.number_of_cells());

    let mut stats = DelaunayRepairStats::default();
    let mut queue: VecDeque<(FacetHandle, u64)> = VecDeque::new();
    let mut queued: FastHashSet<u64> = FastHashSet::default();

    if let Some(seeds) = seed_cells {
        for &cell_key in seeds {
            enqueue_cell_facets(tds, cell_key, &mut queue, &mut queued, &mut stats)?;
        }
    } else {
        for facet in AllFacetsIter::new(tds) {
            let handle = FacetHandle::new(facet.cell_key(), facet.facet_index());
            enqueue_facet(tds, handle, &mut queue, &mut queued, &mut stats);
        }
    }

    while let Some((facet, key)) = queue.pop_front() {
        queued.remove(&key);
        stats.facets_checked += 1;

        let context = match build_k2_flip_context(tds, facet) {
            Ok(ctx) => ctx,
            Err(
                FlipError::BoundaryFacet { .. }
                | FlipError::MissingCell { .. }
                | FlipError::MissingNeighbor { .. }
                | FlipError::InvalidFacetAdjacency { .. }
                | FlipError::InvalidFacetIndex { .. },
            ) => {
                continue;
            }
            Err(e) => return Err(e.into()),
        };

        let violates = match is_delaunay_violation_k2(tds, kernel, &context) {
            Ok(violates) => violates,
            Err(FlipError::PredicateFailure { .. }) => {
                continue;
            }
            Err(e) => return Err(e.into()),
        };

        if !violates {
            continue;
        }

        let info = match apply_bistellar_flip_k2(tds, kernel, &context) {
            Ok(info) => info,
            Err(
                FlipError::DegenerateCell
                | FlipError::DuplicateCell
                | FlipError::NonManifoldFacet
                | FlipError::CellCreation(_),
            ) => {
                continue;
            }
            Err(e) => return Err(e.into()),
        };
        stats.flips_performed += 1;

        if stats.flips_performed > max_flips {
            return Err(DelaunayRepairError::NonConvergent {
                max_flips,
                facets_checked: stats.facets_checked,
            });
        }

        for &cell_key in &info.new_cells {
            enqueue_cell_facets(tds, cell_key, &mut queue, &mut queued, &mut stats)?;
        }
    }

    Ok(stats)
}

/// Repair Delaunay violations using k=2 queues, k=3 queues in 3D,
/// and inverse edge/triangle queues in higher dimensions.
///
/// # Errors
///
/// Returns a [`DelaunayRepairError`] if the repair fails to converge or an underlying
/// flip operation encounters an unrecoverable error.
pub fn repair_delaunay_with_flips_k2_k3<K, U, V, const D: usize>(
    tds: &mut Tds<K::Scalar, U, V, D>,
    kernel: &K,
    seed_cells: Option<&[CellKey]>,
) -> Result<DelaunayRepairStats, DelaunayRepairError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    if D < 2 {
        return Err(FlipError::UnsupportedDimension { dimension: D }.into());
    }
    if D == 2 {
        return repair_delaunay_with_flips_k2(tds, kernel, seed_cells);
    }

    let max_flips = default_max_flips::<D>(tds.number_of_cells());

    let mut stats = DelaunayRepairStats::default();
    let mut queues = RepairQueues::new();
    seed_repair_queues(tds, seed_cells, &mut queues, &mut stats)?;

    let mut prefer_secondary = false;

    while queues.has_work() {
        if prefer_secondary
            && (process_ridge_queue_step(tds, kernel, &mut queues, &mut stats, max_flips)?
                || process_edge_queue_step(tds, kernel, &mut queues, &mut stats, max_flips)?
                || process_triangle_queue_step(tds, kernel, &mut queues, &mut stats, max_flips)?)
        {
            prefer_secondary = false;
            continue;
        }

        if process_facet_queue_step(tds, kernel, &mut queues, &mut stats, max_flips)? {
            prefer_secondary = true;
            continue;
        }

        if process_ridge_queue_step(tds, kernel, &mut queues, &mut stats, max_flips)?
            || process_edge_queue_step(tds, kernel, &mut queues, &mut stats, max_flips)?
            || process_triangle_queue_step(tds, kernel, &mut queues, &mut stats, max_flips)?
        {
            prefer_secondary = false;
        }
    }

    Ok(stats)
}

// =============================================================================
// Internal helpers
// =============================================================================

fn default_max_flips<const D: usize>(cell_count: usize) -> usize {
    let base = cell_count
        .saturating_mul(D.saturating_add(1))
        .saturating_mul(4);
    base.max(128)
}

struct RepairQueues {
    facet_queue: VecDeque<(FacetHandle, u64)>,
    facet_queued: FastHashSet<u64>,
    ridge_queue: VecDeque<(RidgeHandle, u64)>,
    ridge_queued: FastHashSet<u64>,
    edge_queue: VecDeque<(EdgeKey, u64)>,
    edge_queued: FastHashSet<u64>,
    triangle_queue: VecDeque<(TriangleHandle, u64)>,
    triangle_queued: FastHashSet<u64>,
}

impl RepairQueues {
    fn new() -> Self {
        Self {
            facet_queue: VecDeque::new(),
            facet_queued: FastHashSet::default(),
            ridge_queue: VecDeque::new(),
            ridge_queued: FastHashSet::default(),
            edge_queue: VecDeque::new(),
            edge_queued: FastHashSet::default(),
            triangle_queue: VecDeque::new(),
            triangle_queued: FastHashSet::default(),
        }
    }

    fn total_len(&self) -> usize {
        self.facet_queue.len()
            + self.ridge_queue.len()
            + self.edge_queue.len()
            + self.triangle_queue.len()
    }

    fn has_work(&self) -> bool {
        !self.facet_queue.is_empty()
            || !self.ridge_queue.is_empty()
            || !self.edge_queue.is_empty()
            || !self.triangle_queue.is_empty()
    }
}

fn seed_repair_queues<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    seed_cells: Option<&[CellKey]>,
    queues: &mut RepairQueues,
    stats: &mut DelaunayRepairStats,
) -> Result<(), FlipError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    if let Some(seeds) = seed_cells {
        for &cell_key in seeds {
            enqueue_cell_facets(
                tds,
                cell_key,
                &mut queues.facet_queue,
                &mut queues.facet_queued,
                stats,
            )?;
            enqueue_cell_ridges(
                tds,
                cell_key,
                &mut queues.ridge_queue,
                &mut queues.ridge_queued,
                stats,
            )?;
            enqueue_cell_edges(
                tds,
                cell_key,
                &mut queues.edge_queue,
                &mut queues.edge_queued,
                stats,
            );
            enqueue_cell_triangles(
                tds,
                cell_key,
                &mut queues.triangle_queue,
                &mut queues.triangle_queued,
                stats,
            );
            stats.max_queue_len = stats.max_queue_len.max(queues.total_len());
        }
    } else {
        for facet in AllFacetsIter::new(tds) {
            let handle = FacetHandle::new(facet.cell_key(), facet.facet_index());
            enqueue_facet(
                tds,
                handle,
                &mut queues.facet_queue,
                &mut queues.facet_queued,
                stats,
            );
        }
        for (cell_key, _) in tds.cells() {
            enqueue_cell_ridges(
                tds,
                cell_key,
                &mut queues.ridge_queue,
                &mut queues.ridge_queued,
                stats,
            )?;
            enqueue_cell_edges(
                tds,
                cell_key,
                &mut queues.edge_queue,
                &mut queues.edge_queued,
                stats,
            );
            enqueue_cell_triangles(
                tds,
                cell_key,
                &mut queues.triangle_queue,
                &mut queues.triangle_queued,
                stats,
            );
        }
        stats.max_queue_len = stats.max_queue_len.max(queues.total_len());
    }
    Ok(())
}

fn enqueue_new_cells_for_repair<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    new_cells: &[CellKey],
    queues: &mut RepairQueues,
    stats: &mut DelaunayRepairStats,
) -> Result<(), FlipError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    for &cell_key in new_cells {
        enqueue_cell_facets(
            tds,
            cell_key,
            &mut queues.facet_queue,
            &mut queues.facet_queued,
            stats,
        )?;
        enqueue_cell_ridges(
            tds,
            cell_key,
            &mut queues.ridge_queue,
            &mut queues.ridge_queued,
            stats,
        )?;
        enqueue_cell_edges(
            tds,
            cell_key,
            &mut queues.edge_queue,
            &mut queues.edge_queued,
            stats,
        );
        enqueue_cell_triangles(
            tds,
            cell_key,
            &mut queues.triangle_queue,
            &mut queues.triangle_queued,
            stats,
        );
        stats.max_queue_len = stats.max_queue_len.max(queues.total_len());
    }
    Ok(())
}

fn process_ridge_queue_step<K, U, V, const D: usize>(
    tds: &mut Tds<K::Scalar, U, V, D>,
    kernel: &K,
    queues: &mut RepairQueues,
    stats: &mut DelaunayRepairStats,
    max_flips: usize,
) -> Result<bool, DelaunayRepairError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let Some((ridge, key)) = queues.ridge_queue.pop_front() else {
        return Ok(false);
    };
    queues.ridge_queued.remove(&key);
    stats.facets_checked += 1;

    let context = match build_k3_flip_context(tds, ridge) {
        Ok(ctx) => ctx,
        Err(
            FlipError::InvalidRidgeIndex { .. }
            | FlipError::InvalidRidgeAdjacency { .. }
            | FlipError::InvalidRidgeMultiplicity { .. }
            | FlipError::MissingCell { .. },
        ) => {
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };

    let violates = match is_delaunay_violation_k3(tds, kernel, &context) {
        Ok(violates) => violates,
        Err(FlipError::PredicateFailure { .. }) => {
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };

    if !violates {
        return Ok(true);
    }

    let info = match apply_bistellar_flip_k3(tds, kernel, &context) {
        Ok(info) => info,
        Err(
            FlipError::DegenerateCell
            | FlipError::DuplicateCell
            | FlipError::NonManifoldFacet
            | FlipError::CellCreation(_),
        ) => {
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };
    stats.flips_performed += 1;

    if stats.flips_performed > max_flips {
        return Err(DelaunayRepairError::NonConvergent {
            max_flips,
            facets_checked: stats.facets_checked,
        });
    }

    enqueue_new_cells_for_repair(tds, &info.new_cells, queues, stats)?;

    Ok(true)
}

fn process_edge_queue_step<K, U, V, const D: usize>(
    tds: &mut Tds<K::Scalar, U, V, D>,
    kernel: &K,
    queues: &mut RepairQueues,
    stats: &mut DelaunayRepairStats,
    max_flips: usize,
) -> Result<bool, DelaunayRepairError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let Some((edge, key)) = queues.edge_queue.pop_front() else {
        return Ok(false);
    };
    queues.edge_queued.remove(&key);
    stats.facets_checked += 1;

    let context = match build_k2_flip_context_from_edge(tds, edge) {
        Ok(ctx) => ctx,
        Err(
            FlipError::InvalidEdgeMultiplicity { .. }
            | FlipError::InvalidEdgeAdjacency { .. }
            | FlipError::MissingCell { .. }
            | FlipError::MissingVertex { .. },
        ) => {
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };

    if context.removed_face_vertices.len() != 2 {
        return Ok(true);
    }
    let opposite_a = context.removed_face_vertices[0];
    let opposite_b = context.removed_face_vertices[1];

    let violates = match delaunay_violation_k2_for_facet(
        tds,
        kernel,
        &context.inserted_face_vertices,
        opposite_a,
        opposite_b,
    ) {
        Ok(violates) => violates,
        Err(FlipError::PredicateFailure { .. }) => {
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };

    // Only flip if the target (2-cell) configuration is locally Delaunay.
    if violates {
        return Ok(true);
    }

    let info = match apply_bistellar_flip_dynamic(tds, kernel, D, &context) {
        Ok(info) => info,
        Err(
            FlipError::DegenerateCell
            | FlipError::DuplicateCell
            | FlipError::NonManifoldFacet
            | FlipError::CellCreation(_),
        ) => {
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };
    stats.flips_performed += 1;

    if stats.flips_performed > max_flips {
        return Err(DelaunayRepairError::NonConvergent {
            max_flips,
            facets_checked: stats.facets_checked,
        });
    }

    enqueue_new_cells_for_repair(tds, &info.new_cells, queues, stats)?;

    Ok(true)
}

fn process_triangle_queue_step<K, U, V, const D: usize>(
    tds: &mut Tds<K::Scalar, U, V, D>,
    kernel: &K,
    queues: &mut RepairQueues,
    stats: &mut DelaunayRepairStats,
    max_flips: usize,
) -> Result<bool, DelaunayRepairError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let Some((triangle, key)) = queues.triangle_queue.pop_front() else {
        return Ok(false);
    };
    queues.triangle_queued.remove(&key);
    stats.facets_checked += 1;

    let context = match build_k3_flip_context_from_triangle(tds, triangle) {
        Ok(ctx) => ctx,
        Err(
            FlipError::InvalidTriangleMultiplicity { .. }
            | FlipError::InvalidTriangleAdjacency { .. }
            | FlipError::MissingCell { .. }
            | FlipError::MissingVertex { .. },
        ) => {
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };

    let violates = match delaunay_violation_k3_for_ridge(
        tds,
        kernel,
        &context.inserted_face_vertices,
        &context.removed_face_vertices,
    ) {
        Ok(violates) => violates,
        Err(FlipError::PredicateFailure { .. }) => {
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };

    // Only flip if the target (3-cell) configuration is locally Delaunay.
    if violates {
        return Ok(true);
    }

    let info = match apply_bistellar_flip_dynamic(tds, kernel, D - 1, &context) {
        Ok(info) => info,
        Err(
            FlipError::DegenerateCell
            | FlipError::DuplicateCell
            | FlipError::NonManifoldFacet
            | FlipError::CellCreation(_),
        ) => {
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };
    stats.flips_performed += 1;

    if stats.flips_performed > max_flips {
        return Err(DelaunayRepairError::NonConvergent {
            max_flips,
            facets_checked: stats.facets_checked,
        });
    }

    enqueue_new_cells_for_repair(tds, &info.new_cells, queues, stats)?;

    Ok(true)
}

fn process_facet_queue_step<K, U, V, const D: usize>(
    tds: &mut Tds<K::Scalar, U, V, D>,
    kernel: &K,
    queues: &mut RepairQueues,
    stats: &mut DelaunayRepairStats,
    max_flips: usize,
) -> Result<bool, DelaunayRepairError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let Some((facet, key)) = queues.facet_queue.pop_front() else {
        return Ok(false);
    };
    queues.facet_queued.remove(&key);
    stats.facets_checked += 1;

    let context = match build_k2_flip_context(tds, facet) {
        Ok(ctx) => ctx,
        Err(
            FlipError::BoundaryFacet { .. }
            | FlipError::MissingCell { .. }
            | FlipError::MissingNeighbor { .. }
            | FlipError::InvalidFacetAdjacency { .. }
            | FlipError::InvalidFacetIndex { .. },
        ) => {
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };

    let violates = match is_delaunay_violation_k2(tds, kernel, &context) {
        Ok(violates) => violates,
        Err(FlipError::PredicateFailure { .. }) => {
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };

    if !violates {
        return Ok(true);
    }

    let info = match apply_bistellar_flip_k2(tds, kernel, &context) {
        Ok(info) => info,
        Err(
            FlipError::DegenerateCell
            | FlipError::DuplicateCell
            | FlipError::NonManifoldFacet
            | FlipError::CellCreation(_),
        ) => {
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };
    stats.flips_performed += 1;

    if stats.flips_performed > max_flips {
        return Err(DelaunayRepairError::NonConvergent {
            max_flips,
            facets_checked: stats.facets_checked,
        });
    }

    enqueue_new_cells_for_repair(tds, &info.new_cells, queues, stats)?;

    Ok(true)
}

fn facet_vertices_from_cell<T, U, V, const D: usize>(
    cell: &Cell<T, U, V, D>,
    facet_index: usize,
) -> SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let mut vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(D + 1);
    for (i, &vkey) in cell.vertices().iter().enumerate() {
        if i != facet_index {
            vertices.push(vkey);
        }
    }
    vertices
}

fn ridge_vertices_from_cell<T, U, V, const D: usize>(
    cell: &Cell<T, U, V, D>,
    omit_a: usize,
    omit_b: usize,
) -> SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let mut vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(D + 1);
    for (i, &vkey) in cell.vertices().iter().enumerate() {
        if i != omit_a && i != omit_b {
            vertices.push(vkey);
        }
    }
    vertices
}

fn cell_extras_for_ridge<T, U, V, const D: usize>(
    cell_key: CellKey,
    cell: &Cell<T, U, V, D>,
    ridge: &SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
) -> Result<SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>, FlipError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    if !ridge.iter().all(|v| cell.contains_vertex(*v)) {
        return Err(FlipError::InvalidRidgeAdjacency { cell_key });
    }

    let mut extras: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(2);
    for &vkey in cell.vertices() {
        if !ridge.contains(&vkey) {
            extras.push(vkey);
        }
    }
    Ok(extras)
}

fn missing_opposite_for_cell(
    extras: &[VertexKey; 2],
    opposites: &[VertexKey; 3],
) -> Option<VertexKey> {
    opposites
        .iter()
        .copied()
        .find(|v| *v != extras[0] && *v != extras[1])
}

fn collect_cells_around_ridge<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    start_cell: CellKey,
    ridge: &SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
) -> Result<CellKeyBuffer, FlipError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let mut queue: VecDeque<CellKey> = VecDeque::new();
    let mut visited: FastHashSet<CellKey> = FastHashSet::default();
    let mut cells: CellKeyBuffer = CellKeyBuffer::new();

    queue.push_back(start_cell);

    while let Some(cell_key) = queue.pop_front() {
        if !visited.insert(cell_key) {
            continue;
        }

        let cell = tds
            .get_cell(cell_key)
            .ok_or(FlipError::MissingCell { cell_key })?;
        if !ridge.iter().all(|v| cell.contains_vertex(*v)) {
            return Err(FlipError::InvalidRidgeAdjacency { cell_key });
        }

        let mut omit_indices: SmallBuffer<usize, 2> = SmallBuffer::with_capacity(2);
        for (i, &vkey) in cell.vertices().iter().enumerate() {
            if !ridge.contains(&vkey) {
                omit_indices.push(i);
            }
        }
        if omit_indices.len() != 2 {
            return Err(FlipError::InvalidRidgeAdjacency { cell_key });
        }

        cells.push(cell_key);

        if let Some(neighbors) = cell.neighbors() {
            for &omit_idx in &omit_indices {
                if let Some(neighbor_key) = neighbors.get(omit_idx).copied().flatten() {
                    if !tds.contains_cell(neighbor_key) {
                        continue;
                    }
                    let neighbor_cell =
                        tds.get_cell(neighbor_key).ok_or(FlipError::MissingCell {
                            cell_key: neighbor_key,
                        })?;
                    if !ridge.iter().all(|v| neighbor_cell.contains_vertex(*v)) {
                        return Err(FlipError::InvalidRidgeAdjacency { cell_key });
                    }
                    queue.push_back(neighbor_key);
                }
            }
        }
    }

    Ok(cells)
}

fn vertices_to_points<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    vertices: &[VertexKey],
) -> Result<SmallBuffer<Point<T, D>, MAX_PRACTICAL_DIMENSION_SIZE>, FlipError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let mut points: SmallBuffer<Point<T, D>, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(vertices.len());
    for &vkey in vertices {
        let vertex = tds
            .get_vertex_by_key(vkey)
            .ok_or(FlipError::MissingVertex { vertex_key: vkey })?;
        points.push(*vertex.point());
    }
    Ok(points)
}

fn flip_would_duplicate_cell_any<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    vertices: &[VertexKey],
    removed: &[CellKey],
) -> bool
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let mut target: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        vertices.iter().copied().collect();
    target.sort_unstable();

    for (cell_key, cell) in tds.cells() {
        if removed.contains(&cell_key) {
            continue;
        }
        if cell.number_of_vertices() != vertices.len() {
            continue;
        }
        let mut cell_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
            cell.vertices().iter().copied().collect();
        cell_vertices.sort_unstable();
        if cell_vertices == target {
            return true;
        }
    }
    false
}

fn flip_would_create_nonmanifold_facets_any<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    vertices: &[VertexKey],
    removed: &[CellKey],
    opposite_vertices: &[VertexKey],
) -> bool
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    for omit_idx in 0..vertices.len() {
        let mut facet_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(D);
        for (i, &vkey) in vertices.iter().enumerate() {
            if i != omit_idx {
                facet_vertices.push(vkey);
            }
        }

        let mut shared_count = 0usize;
        for (cell_key, cell) in tds.cells() {
            if removed.contains(&cell_key) {
                continue;
            }
            if facet_vertices.iter().all(|v| cell.contains_vertex(*v)) {
                shared_count += 1;
                if shared_count > 1 {
                    return true;
                }
            }
        }

        let internal_facet = opposite_vertices.iter().all(|v| facet_vertices.contains(v));
        if internal_facet && shared_count > 0 {
            return true;
        }
    }
    false
}

fn enqueue_cell_facets<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    cell_key: CellKey,
    queue: &mut VecDeque<(FacetHandle, u64)>,
    queued: &mut FastHashSet<u64>,
    stats: &mut DelaunayRepairStats,
) -> Result<(), FlipError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let Some(cell) = tds.get_cell(cell_key) else {
        return Ok(());
    };
    for facet_index in 0..cell.number_of_vertices() {
        let handle = FacetHandle::new(
            cell_key,
            u8::try_from(facet_index).map_err(|_| FlipError::InvalidFacetIndex {
                cell_key,
                facet_index: u8::MAX,
                vertex_count: cell.number_of_vertices(),
            })?,
        );
        enqueue_facet(tds, handle, queue, queued, stats);
    }
    Ok(())
}

fn enqueue_facet<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    handle: FacetHandle,
    queue: &mut VecDeque<(FacetHandle, u64)>,
    queued: &mut FastHashSet<u64>,
    stats: &mut DelaunayRepairStats,
) where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let Some(cell) = tds.get_cell(handle.cell_key()) else {
        return;
    };

    let facet_index = usize::from(handle.facet_index());
    if facet_index >= cell.number_of_vertices() {
        return;
    }

    let Some(_neighbor_key) = cell
        .neighbors()
        .and_then(|n| n.get(facet_index).copied().flatten())
        .filter(|&nk| tds.contains_cell(nk))
    else {
        return;
    };

    let facet_vertices = facet_vertices_from_cell(cell, facet_index);
    let key = facet_key_from_vertices(&facet_vertices);

    if queued.insert(key) {
        queue.push_back((handle, key));
        stats.max_queue_len = stats.max_queue_len.max(queue.len());
    }
}

fn enqueue_cell_edges<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    cell_key: CellKey,
    queue: &mut VecDeque<(EdgeKey, u64)>,
    queued: &mut FastHashSet<u64>,
    stats: &mut DelaunayRepairStats,
) where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    if D < 4 {
        return;
    }

    let Some(cell) = tds.get_cell(cell_key) else {
        return;
    };

    let vertices = cell.vertices();
    let vertex_count = vertices.len();
    for i in 0..vertex_count {
        for j in (i + 1)..vertex_count {
            let edge = EdgeKey::new(vertices[i], vertices[j]);
            enqueue_edge(edge, queue, queued, stats);
        }
    }
}

fn enqueue_edge(
    edge: EdgeKey,
    queue: &mut VecDeque<(EdgeKey, u64)>,
    queued: &mut FastHashSet<u64>,
    stats: &mut DelaunayRepairStats,
) {
    let key = facet_key_from_vertices(&[edge.v0(), edge.v1()]);
    if queued.insert(key) {
        queue.push_back((edge, key));
        stats.max_queue_len = stats.max_queue_len.max(queue.len());
    }
}

fn enqueue_cell_triangles<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    cell_key: CellKey,
    queue: &mut VecDeque<(TriangleHandle, u64)>,
    queued: &mut FastHashSet<u64>,
    stats: &mut DelaunayRepairStats,
) where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    if D < 5 {
        return;
    }

    let Some(cell) = tds.get_cell(cell_key) else {
        return;
    };

    let vertices = cell.vertices();
    let vertex_count = vertices.len();
    for i in 0..vertex_count {
        for j in (i + 1)..vertex_count {
            for k in (j + 1)..vertex_count {
                let triangle = TriangleHandle::new(vertices[i], vertices[j], vertices[k]);
                enqueue_triangle(triangle, queue, queued, stats);
            }
        }
    }
}

fn enqueue_triangle(
    triangle: TriangleHandle,
    queue: &mut VecDeque<(TriangleHandle, u64)>,
    queued: &mut FastHashSet<u64>,
    stats: &mut DelaunayRepairStats,
) {
    let vertices = triangle.vertices();
    let key = facet_key_from_vertices(&vertices);
    if queued.insert(key) {
        queue.push_back((triangle, key));
        stats.max_queue_len = stats.max_queue_len.max(queue.len());
    }
}

fn enqueue_cell_ridges<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    cell_key: CellKey,
    queue: &mut VecDeque<(RidgeHandle, u64)>,
    queued: &mut FastHashSet<u64>,
    stats: &mut DelaunayRepairStats,
) -> Result<(), FlipError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    if D != 3 {
        return Ok(());
    }

    let Some(cell) = tds.get_cell(cell_key) else {
        return Ok(());
    };

    let vertex_count = cell.number_of_vertices();
    for i in 0..vertex_count {
        for j in (i + 1)..vertex_count {
            let handle = RidgeHandle::new(
                cell_key,
                u8::try_from(i).map_err(|_| FlipError::InvalidRidgeIndex {
                    cell_key,
                    omit_a: u8::MAX,
                    omit_b: u8::MAX,
                    vertex_count,
                })?,
                u8::try_from(j).map_err(|_| FlipError::InvalidRidgeIndex {
                    cell_key,
                    omit_a: u8::MAX,
                    omit_b: u8::MAX,
                    vertex_count,
                })?,
            );
            enqueue_ridge(tds, handle, queue, queued, stats);
        }
    }

    Ok(())
}

fn enqueue_ridge<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    handle: RidgeHandle,
    queue: &mut VecDeque<(RidgeHandle, u64)>,
    queued: &mut FastHashSet<u64>,
    stats: &mut DelaunayRepairStats,
) where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    if D != 3 {
        return;
    }

    let Some(cell) = tds.get_cell(handle.cell_key()) else {
        return;
    };

    let vertex_count = cell.number_of_vertices();
    let omit_a = usize::from(handle.omit_a());
    let omit_b = usize::from(handle.omit_b());
    if omit_a >= vertex_count || omit_b >= vertex_count || omit_a == omit_b {
        return;
    }

    let ridge_vertices = ridge_vertices_from_cell(cell, omit_a, omit_b);
    if ridge_vertices.len() != D - 1 {
        return;
    }

    let key = facet_key_from_vertices(&ridge_vertices);
    if queued.insert(key) {
        queue.push_back((handle, key));
        stats.max_queue_len = stats.max_queue_len.max(queue.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::incremental_insertion::repair_neighbor_pointers;
    use crate::core::delaunay_triangulation::DelaunayTriangulation;
    use crate::geometry::kernel::FastKernel;
    use crate::vertex;
    fn unit_vector<const D: usize>(index: usize) -> [f64; D] {
        let mut coords = [0.0; D];
        coords[index] = 1.0;
        coords
    }

    macro_rules! test_flip_k2_dimension {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<test_flip_k2_ $dim d_two_to_ $dim>]() {
                    let mut tds: Tds<f64, (), (), $dim> = Tds::empty();
                    let mut shared_vertices = Vec::with_capacity($dim);
                    for i in 0..$dim {
                        let v = tds
                            .insert_vertex_with_mapping(vertex!(unit_vector::<$dim>(i)))
                            .unwrap();
                        shared_vertices.push(v);
                    }

                    let opposite_a = tds
                        .insert_vertex_with_mapping(vertex!([0.0; $dim]))
                        .unwrap();
                    let opposite_b = tds
                        .insert_vertex_with_mapping(vertex!([1.0; $dim]))
                        .unwrap();

                    let mut vertices_with_first_opposite = shared_vertices.clone();
                    vertices_with_first_opposite.push(opposite_a);
                    let cell_a = tds
                        .insert_cell_with_mapping(
                            Cell::new(vertices_with_first_opposite, None).unwrap(),
                        )
                        .unwrap();

                    let mut vertices_with_second_opposite = shared_vertices.clone();
                    vertices_with_second_opposite.push(opposite_b);
                    let _cell_b = tds
                        .insert_cell_with_mapping(
                            Cell::new(vertices_with_second_opposite, None).unwrap(),
                        )
                        .unwrap();

                    repair_neighbor_pointers(&mut tds).unwrap();

                    let facet = FacetHandle::new(cell_a, u8::try_from($dim).unwrap());
                    let context = build_k2_flip_context(&tds, facet).unwrap();
                    let kernel = FastKernel::<f64>::new();
                    let info = apply_bistellar_flip_k2(&mut tds, &kernel, &context).unwrap();

                    assert_eq!(info.kind, BistellarFlipKind::k2($dim));
                    assert_eq!(info.removed_cells.len(), 2);
                    assert_eq!(info.new_cells.len(), $dim);
                    assert_eq!(info.removed_face_vertices.len(), $dim);
                    assert!(tds.is_valid().is_ok());
                }
            }
        };
    }

    test_flip_k2_dimension!(2);
    test_flip_k2_dimension!(3);
    test_flip_k2_dimension!(4);
    test_flip_k2_dimension!(5);

    #[test]
    fn test_flip_k2_2d_edge_flip() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();
        let a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let d = tds.insert_vertex_with_mapping(vertex!([1.0, 0.2])).unwrap();

        let c1 = tds
            .insert_cell_with_mapping(Cell::new(vec![a, b, c], None).unwrap())
            .unwrap();
        let _c2 = tds
            .insert_cell_with_mapping(Cell::new(vec![a, b, d], None).unwrap())
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let facet = FacetHandle::new(c1, 2); // facet opposite vertex index 2 (edge AB)
        let kernel = FastKernel::<f64>::new();
        let decision = try_bistellar_flip_k2(&mut tds, &kernel, facet).unwrap();

        let FlipDecision::Flipped(info) = decision else {
            panic!("Expected flip to occur");
        };

        assert_eq!(info.removed_cells.len(), 2);
        assert_eq!(info.new_cells.len(), 2);

        // After flip, we should have an edge between c and d in some cell.
        let mut has_cd = false;
        for (_, cell) in tds.cells() {
            let verts = cell.vertices();
            if verts.contains(&c) && verts.contains(&d) {
                has_cd = true;
            }
        }
        assert!(has_cd, "Expected flipped diagonal between c and d");

        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k2_3d_two_to_three() {
        let mut tds: Tds<f64, (), (), 3> = Tds::empty();
        let v_a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0]))
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0]))
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0]))
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(vertex!([0.2, 0.2, 1.0]))
            .unwrap();
        let v_e = tds
            .insert_vertex_with_mapping(vertex!([0.3, -0.1, -0.8]))
            .unwrap();

        let c1 = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_c, v_d], None).unwrap())
            .unwrap();
        let _c2 = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_c, v_e], None).unwrap())
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let facet = FacetHandle::new(c1, 3); // facet opposite vertex d (ABC)
        let context = build_k2_flip_context(&tds, facet).unwrap();
        let kernel = FastKernel::<f64>::new();
        let info = apply_bistellar_flip_k2(&mut tds, &kernel, &context).unwrap();

        assert_eq!(info.new_cells.len(), 3);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k3_3d_three_to_two() {
        let mut tds: Tds<f64, (), (), 3> = Tds::empty();
        let r0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0]))
            .unwrap();
        let r1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0]))
            .unwrap();
        let a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0]))
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0]))
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(vertex!([0.2, 0.2, -1.0]))
            .unwrap();

        let c1 = tds
            .insert_cell_with_mapping(Cell::new(vec![r0, r1, a, b], None).unwrap())
            .unwrap();
        let _c2 = tds
            .insert_cell_with_mapping(Cell::new(vec![r0, r1, b, c], None).unwrap())
            .unwrap();
        let _c3 = tds
            .insert_cell_with_mapping(Cell::new(vec![r0, r1, c, a], None).unwrap())
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let ridge = RidgeHandle::new(c1, 2, 3);
        let context = build_k3_flip_context(&tds, ridge).unwrap();
        let kernel = FastKernel::<f64>::new();
        let info = apply_bistellar_flip_k3(&mut tds, &kernel, &context).unwrap();

        assert_eq!(info.kind, BistellarFlipKind::k3(3));
        assert_eq!(info.removed_cells.len(), 3);
        assert_eq!(info.new_cells.len(), 2);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k3_4d_three_to_three() {
        let mut tds: Tds<f64, (), (), 4> = Tds::empty();
        let r0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0, 0.0]))
            .unwrap();
        let r1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        let r2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0, 0.0]))
            .unwrap();
        let a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0, 0.0]))
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0, 1.0]))
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(vertex!([0.2, 0.2, 0.2, 0.2]))
            .unwrap();

        let c1 = tds
            .insert_cell_with_mapping(Cell::new(vec![r0, r1, r2, a, b], None).unwrap())
            .unwrap();
        let _c2 = tds
            .insert_cell_with_mapping(Cell::new(vec![r0, r1, r2, b, c], None).unwrap())
            .unwrap();
        let _c3 = tds
            .insert_cell_with_mapping(Cell::new(vec![r0, r1, r2, c, a], None).unwrap())
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let ridge = RidgeHandle::new(c1, 3, 4);
        let context = build_k3_flip_context(&tds, ridge).unwrap();
        let kernel = FastKernel::<f64>::new();
        let info = apply_bistellar_flip_k3(&mut tds, &kernel, &context).unwrap();

        assert_eq!(info.kind, BistellarFlipKind::k3(4));
        assert_eq!(info.removed_cells.len(), 3);
        assert_eq!(info.new_cells.len(), 3);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k3_5d_three_to_four() {
        let mut tds: Tds<f64, (), (), 5> = Tds::empty();
        let r0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0, 0.0, 0.0]))
            .unwrap();
        let r1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0, 0.0, 0.0]))
            .unwrap();
        let r2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        let r3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0, 0.0, 0.0]))
            .unwrap();
        let a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0, 1.0, 0.0]))
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0, 0.0, 1.0]))
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(vertex!([0.2, 0.2, 0.2, 0.2, 0.5]))
            .unwrap();

        let c1 = tds
            .insert_cell_with_mapping(Cell::new(vec![r0, r1, r2, r3, a, b], None).unwrap())
            .unwrap();
        let _c2 = tds
            .insert_cell_with_mapping(Cell::new(vec![r0, r1, r2, r3, b, c], None).unwrap())
            .unwrap();
        let _c3 = tds
            .insert_cell_with_mapping(Cell::new(vec![r0, r1, r2, r3, c, a], None).unwrap())
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let ridge = RidgeHandle::new(c1, 4, 5);
        let context = build_k3_flip_context(&tds, ridge).unwrap();
        let kernel = FastKernel::<f64>::new();
        let info = apply_bistellar_flip_k3(&mut tds, &kernel, &context).unwrap();

        assert_eq!(info.kind, BistellarFlipKind::k3(5));
        assert_eq!(info.removed_cells.len(), 3);
        assert_eq!(info.new_cells.len(), 4);
        assert!(tds.is_valid().is_ok());
    }
    #[test]
    fn test_flip_k1_2d_roundtrip() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();
        let a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();

        let cell = tds
            .insert_cell_with_mapping(Cell::new(vec![a, b, c], None).unwrap())
            .unwrap();

        let kernel = FastKernel::<f64>::new();
        let new_vertex = vertex!([0.2, 0.2]);
        let new_uuid = new_vertex.uuid();
        let info = apply_bistellar_flip_k1(&mut tds, &kernel, cell, new_vertex).unwrap();

        assert_eq!(info.kind.k, 1);
        assert_eq!(info.kind.d, 2);
        assert_eq!(tds.number_of_cells(), 3);

        let new_key = tds.vertex_key_from_uuid(&new_uuid).unwrap();
        let info_back = apply_bistellar_flip_k1_inverse(&mut tds, &kernel, new_key).unwrap();

        assert_eq!(info_back.kind.k, 3);
        assert_eq!(info_back.kind.d, 2);
        assert_eq!(tds.number_of_cells(), 1);
        assert_eq!(tds.number_of_vertices(), 3);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_repair_queue_inverse_k2_smoke_4d() {
        let mut tds: Tds<f64, (), (), 4> = Tds::empty();
        let mut shared_vertices = Vec::with_capacity(4);
        for i in 0..4 {
            let v = tds
                .insert_vertex_with_mapping(vertex!(unit_vector::<4>(i)))
                .unwrap();
            shared_vertices.push(v);
        }

        let opposite_a = tds.insert_vertex_with_mapping(vertex!([0.0; 4])).unwrap();
        let opposite_b = tds.insert_vertex_with_mapping(vertex!([1.0; 4])).unwrap();

        let mut vertices_with_first_opposite = shared_vertices.clone();
        vertices_with_first_opposite.push(opposite_a);
        let cell_a = tds
            .insert_cell_with_mapping(Cell::new(vertices_with_first_opposite, None).unwrap())
            .unwrap();

        let mut vertices_with_second_opposite = shared_vertices.clone();
        vertices_with_second_opposite.push(opposite_b);
        let _cell_b = tds
            .insert_cell_with_mapping(Cell::new(vertices_with_second_opposite, None).unwrap())
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let facet = FacetHandle::new(cell_a, 4);
        let context = build_k2_flip_context(&tds, facet).unwrap();
        let kernel = FastKernel::<f64>::new();
        let info = apply_bistellar_flip_k2(&mut tds, &kernel, &context).unwrap();

        let seed_cells: Vec<CellKey> = info.new_cells.iter().copied().collect();
        let stats =
            repair_delaunay_with_flips_k2_k3(&mut tds, &kernel, Some(seed_cells.as_slice()))
                .unwrap();
        assert!(stats.facets_checked > 0);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_repair_queue_inverse_k3_smoke_5d() {
        let mut tds: Tds<f64, (), (), 5> = Tds::empty();
        let r0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0, 0.0, 0.0]))
            .unwrap();
        let r1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0, 0.0, 0.0]))
            .unwrap();
        let r2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        let r3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0, 0.0, 0.0]))
            .unwrap();
        let a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0, 1.0, 0.0]))
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0, 0.0, 1.0]))
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(vertex!([0.2, 0.2, 0.2, 0.2, 0.5]))
            .unwrap();

        let c1 = tds
            .insert_cell_with_mapping(Cell::new(vec![r0, r1, r2, r3, a, b], None).unwrap())
            .unwrap();
        let _c2 = tds
            .insert_cell_with_mapping(Cell::new(vec![r0, r1, r2, r3, b, c], None).unwrap())
            .unwrap();
        let _c3 = tds
            .insert_cell_with_mapping(Cell::new(vec![r0, r1, r2, r3, c, a], None).unwrap())
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let ridge = RidgeHandle::new(c1, 4, 5);
        let context = build_k3_flip_context(&tds, ridge).unwrap();
        let kernel = FastKernel::<f64>::new();
        let info = apply_bistellar_flip_k3(&mut tds, &kernel, &context).unwrap();

        let seed_cells: Vec<CellKey> = info.new_cells.iter().copied().collect();
        let stats =
            repair_delaunay_with_flips_k2_k3(&mut tds, &kernel, Some(seed_cells.as_slice()))
                .unwrap();
        assert!(stats.facets_checked > 0);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_repair_queue_k2_local_seed() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([1.0, 0.2]),
        ];
        let dt: DelaunayTriangulation<FastKernel<f64>, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let mut tds = dt.tds().clone();
        let kernel = FastKernel::<f64>::new();

        let seed_cell = tds.cell_keys().next().unwrap();
        let stats = repair_delaunay_with_flips_k2(&mut tds, &kernel, Some(&[seed_cell])).unwrap();
        assert!(stats.facets_checked > 0);
        assert!(tds.is_valid().is_ok());
    }
}
