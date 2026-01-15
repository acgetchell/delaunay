//! Bistellar flip operations for triangulations.
//!
//! This module implements a **generic k=2 bistellar flip** (2↔(D) for D≥2),
//! which is the fundamental flip used to restore the local Delaunay property.
//! It supports dimensions 2D–5D and is wired for automatic repair via a
//! flip-queue driver.
//!
//! # References
//! - Edelsbrunner & Shah (1996) - "Incremental Topological Flipping Works for Regular Triangulations"
//! - Bistellar flips implementation notebook (Warp Drive)

use std::collections::VecDeque;
use std::sync::Arc;

use thiserror::Error;

use crate::core::algorithms::incremental_insertion::wire_cavity_neighbors;
use crate::core::cell::{Cell, CellValidationError};
use crate::core::collections::{
    CellKeyBuffer, FastHashSet, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer,
};
use crate::core::facet::{AllFacetsIter, FacetHandle, facet_key_from_vertices};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation_data_structure::{CellKey, Tds, VertexKey};
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

impl BistellarFlipKind {
    /// Construct a k=2 flip kind for the given dimension.
    #[must_use]
    pub const fn k2(d: usize) -> Self {
        Self { k: 2, d }
    }
}

/// Errors that can occur during bistellar flips or repair.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FlipError {
    /// Flips are not supported for this dimension.
    #[error("Bistellar flips are only supported for D>=2, got D={dimension}")]
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
    /// Cells removed by the flip.
    pub removed_cells: [CellKey; 2],
    /// Newly created cells.
    pub new_cells: CellKeyBuffer,
    /// The shared facet (D vertices).
    pub shared_facet: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// The two opposite vertices across the facet.
    pub opposite_vertices: (VertexKey, VertexKey),
}

/// Flip decision result.
#[derive(Debug, Clone)]
pub enum FlipDecision<const D: usize> {
    /// The facet was flipped.
    Flipped(Arc<FlipInfo<D>>),
    /// No flip was performed (facet already locally Delaunay).
    NoFlip,
}

/// Flip context for a k=2 (facet) flip.
#[derive(Debug, Clone)]
pub struct FlipContext<const D: usize> {
    cell_a: CellKey,
    cell_b: CellKey,
    opposite_a: VertexKey,
    opposite_b: VertexKey,
    shared_facet: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
}

/// Statistics for flip-based Delaunay repair.
#[derive(Debug, Clone, Default)]
pub struct DelaunayRepairStats {
    /// Number of facets checked.
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
) -> Result<FlipContext<D>, FlipError>
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

    Ok(FlipContext {
        cell_a: cell_a_key,
        cell_b: neighbor_key,
        opposite_a,
        opposite_b,
        shared_facet,
    })
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
    context: &FlipContext<D>,
) -> Result<bool, FlipError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let points_a = cell_points(tds, context.cell_a)?;
    let points_b = cell_points(tds, context.cell_b)?;

    let opposite_point_a = tds
        .get_vertex_by_key(context.opposite_a)
        .ok_or(FlipError::MissingVertex {
            vertex_key: context.opposite_a,
        })?
        .point();
    let opposite_point_b = tds
        .get_vertex_by_key(context.opposite_b)
        .ok_or(FlipError::MissingVertex {
            vertex_key: context.opposite_b,
        })?
        .point();

    let in_a =
        kernel
            .in_sphere(&points_a, opposite_point_b)
            .map_err(|e| FlipError::PredicateFailure {
                message: format!("in_sphere failed for cell {:?}: {e}", context.cell_a),
            })?;

    let in_b =
        kernel
            .in_sphere(&points_b, opposite_point_a)
            .map_err(|e| FlipError::PredicateFailure {
                message: format!("in_sphere failed for cell {:?}: {e}", context.cell_b),
            })?;

    Ok(in_a > 0 || in_b > 0)
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
    context: &FlipContext<D>,
) -> Result<FlipInfo<D>, FlipError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let mut new_cells = CellKeyBuffer::new();
    let mut new_cell_vertices: SmallBuffer<
        SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
        MAX_PRACTICAL_DIMENSION_SIZE,
    > = SmallBuffer::with_capacity(D);

    let removed = [context.cell_a, context.cell_b];
    let shared = &context.shared_facet;
    let opposite_a = context.opposite_a;
    let opposite_b = context.opposite_b;

    for &omit in shared {
        let mut vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(D + 1);
        vertices.push(opposite_a);
        vertices.push(opposite_b);
        for &v in shared {
            if v != omit {
                vertices.push(v);
            }
        }

        if flip_would_duplicate_cell(tds, &vertices, &removed) {
            return Err(FlipError::DuplicateCell);
        }
        if flip_would_create_nonmanifold_facets(tds, &vertices, &removed, opposite_a, opposite_b) {
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
    let removed_keys: CellKeyBuffer = removed.iter().copied().collect();

    wire_cavity_neighbors(tds, &new_cells, Some(&removed_keys)).map_err(|e| {
        FlipError::NeighborWiring {
            message: e.to_string(),
        }
    })?;

    tds.remove_cells_by_keys(&removed_keys);

    Ok(FlipInfo {
        kind: BistellarFlipKind::k2(D),
        removed_cells: removed,
        new_cells,
        shared_facet: shared.clone(),
        opposite_vertices: (opposite_a, opposite_b),
    })
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

// =============================================================================
// Internal helpers
// =============================================================================

fn default_max_flips<const D: usize>(cell_count: usize) -> usize {
    let base = cell_count
        .saturating_mul(D.saturating_add(1))
        .saturating_mul(4);
    base.max(128)
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

fn cell_points<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    cell_key: CellKey,
) -> Result<SmallBuffer<Point<T, D>, MAX_PRACTICAL_DIMENSION_SIZE>, FlipError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let cell = tds
        .get_cell(cell_key)
        .ok_or(FlipError::MissingCell { cell_key })?;
    let mut points: SmallBuffer<Point<T, D>, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(D + 1);
    for &vkey in cell.vertices() {
        let vertex = tds
            .get_vertex_by_key(vkey)
            .ok_or(FlipError::MissingVertex { vertex_key: vkey })?;
        points.push(*vertex.point());
    }
    Ok(points)
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

fn flip_would_duplicate_cell<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    vertices: &[VertexKey],
    removed: &[CellKey; 2],
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
        if cell_key == removed[0] || cell_key == removed[1] {
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

fn flip_would_create_nonmanifold_facets<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    vertices: &[VertexKey],
    removed: &[CellKey; 2],
    opposite_a: VertexKey,
    opposite_b: VertexKey,
) -> bool
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let vertex_count = vertices.len();
    for omit_idx in 0..vertex_count {
        let mut facet_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(vertex_count.saturating_sub(1));
        for (i, &vkey) in vertices.iter().enumerate() {
            if i != omit_idx {
                facet_vertices.push(vkey);
            }
        }

        let mut shared_count = 0usize;
        for (cell_key, cell) in tds.cells() {
            if cell_key == removed[0] || cell_key == removed[1] {
                continue;
            }
            if facet_vertices.iter().all(|v| cell.contains_vertex(*v)) {
                shared_count += 1;
                if shared_count > 1 {
                    return true;
                }
            }
        }

        let internal_facet =
            facet_vertices.contains(&opposite_a) && facet_vertices.contains(&opposite_b);
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

                    let mut cell_a_vertices = shared_vertices.clone();
                    cell_a_vertices.push(opposite_a);
                    let cell_a = tds
                        .insert_cell_with_mapping(Cell::new(cell_a_vertices, None).unwrap())
                        .unwrap();

                    let mut cell_b_vertices = shared_vertices.clone();
                    cell_b_vertices.push(opposite_b);
                    let _cell_b = tds
                        .insert_cell_with_mapping(Cell::new(cell_b_vertices, None).unwrap())
                        .unwrap();

                    repair_neighbor_pointers(&mut tds).unwrap();

                    let facet = FacetHandle::new(cell_a, u8::try_from($dim).unwrap());
                    let context = build_k2_flip_context(&tds, facet).unwrap();
                    let kernel = FastKernel::<f64>::new();
                    let info = apply_bistellar_flip_k2(&mut tds, &kernel, &context).unwrap();

                    assert_eq!(info.kind, BistellarFlipKind::k2($dim));
                    assert_eq!(info.removed_cells.len(), 2);
                    assert_eq!(info.new_cells.len(), $dim);
                    assert_eq!(info.shared_facet.len(), $dim);
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
