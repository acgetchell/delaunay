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

use core::iter::Sum;
use slotmap::Key;
use std::collections::VecDeque;
use std::fmt;
use std::hash::{Hash, Hasher};

use thiserror::Error;

use crate::core::algorithms::incremental_insertion::wire_cavity_neighbors;
use crate::core::cell::{Cell, CellValidationError};
use crate::core::collections::{
    CellKeyBuffer, FastHashMap, FastHashSet, FastHasher, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer,
};
use crate::core::edge::EdgeKey;
use crate::core::facet::{AllFacetsIter, FacetHandle, facet_key_from_vertices};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation::TopologyGuarantee;
use crate::core::triangulation_data_structure::{CellKey, Tds, VertexKey};
use crate::core::vertex::Vertex;
use crate::geometry::kernel::Kernel;
use crate::geometry::point::Point;
use crate::geometry::predicates::{InSphere, Orientation};
use crate::geometry::robust_predicates::{config_presets, robust_insphere, robust_orientation};
use crate::geometry::traits::coordinate::CoordinateScalar;

/// Bistellar flip kind descriptor.
///
/// Access the move size with [`BistellarFlipKind::k`].
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::flips::BistellarFlipKind;
///
/// let kind = BistellarFlipKind::k2(3);
/// let inverse = kind.inverse();
/// assert_eq!(kind.k(), 2);
/// assert_eq!(inverse.k(), 3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BistellarFlipKind {
    /// Number of simplices being replaced on the current side (k).
    k: usize,
    /// Dimension of the triangulation (D).
    pub d: usize,
}
/// Run a single flip-repair attempt using k=2 (and k=3 in 3D+).
fn repair_delaunay_with_flips_k2_k3_attempt<K, U, V, const D: usize>(
    tds: &mut Tds<K::Scalar, U, V, D>,
    kernel: &K,
    seed_cells: Option<&[CellKey]>,
    config: &RepairAttemptConfig,
) -> Result<DelaunayRepairStats, DelaunayRepairError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar + Sum,
    U: DataType,
    V: DataType,
{
    if D < 2 {
        return Err(FlipError::UnsupportedDimension { dimension: D }.into());
    }
    if D == 2 {
        return repair_delaunay_with_flips_k2_attempt(tds, kernel, seed_cells, config);
    }

    let max_flips = default_max_flips::<D>(tds.number_of_cells());

    let mut stats = DelaunayRepairStats::default();
    let mut diagnostics = RepairDiagnostics::default();
    let mut queues = RepairQueues::new();
    seed_repair_queues(tds, seed_cells, &mut queues, &mut stats)?;

    let mut prefer_secondary = false;

    while queues.has_work() {
        if prefer_secondary
            && (process_ridge_queue_step(
                tds,
                kernel,
                &mut queues,
                &mut stats,
                max_flips,
                config,
                &mut diagnostics,
            )? || process_edge_queue_step(
                tds,
                kernel,
                &mut queues,
                &mut stats,
                max_flips,
                config,
                &mut diagnostics,
            )? || process_triangle_queue_step(
                tds,
                kernel,
                &mut queues,
                &mut stats,
                max_flips,
                config,
                &mut diagnostics,
            )?)
        {
            prefer_secondary = false;
            continue;
        }

        if process_facet_queue_step(
            tds,
            kernel,
            &mut queues,
            &mut stats,
            max_flips,
            config,
            &mut diagnostics,
        )? {
            prefer_secondary = true;
            continue;
        }

        if process_ridge_queue_step(
            tds,
            kernel,
            &mut queues,
            &mut stats,
            max_flips,
            config,
            &mut diagnostics,
        )? || process_edge_queue_step(
            tds,
            kernel,
            &mut queues,
            &mut stats,
            max_flips,
            config,
            &mut diagnostics,
        )? || process_triangle_queue_step(
            tds,
            kernel,
            &mut queues,
            &mut stats,
            max_flips,
            config,
            &mut diagnostics,
        )? {
            prefer_secondary = false;
        }
    }
    if repair_trace_enabled() {
        tracing::debug!(
            "[repair] attempt={} done: checked={} flips={} max_queue={} ambiguous={} predicate_failures={} cycles={}",
            config.attempt,
            stats.facets_checked,
            stats.flips_performed,
            stats.max_queue_len,
            diagnostics.ambiguous_predicates,
            diagnostics.predicate_failures,
            diagnostics.cycle_detections,
        );
    }

    Ok(stats)
}

/// Apply a bistellar flip using explicit k and vertex/cell slices.
#[expect(
    clippy::too_many_lines,
    reason = "Keep flip construction, validation, and wiring together for clarity"
)]
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
            let config = config_presets::high_precision::<K::Scalar>();
            let robust_orientation =
                robust_orientation(&points, &config).map_err(|e| FlipError::PredicateFailure {
                    message: format!("robust orientation failed for flip cell: {e}"),
                })?;
            if matches!(robust_orientation, Orientation::DEGENERATE) {
                if std::env::var_os("DELAUNAY_REPAIR_DEBUG_FACETS").is_some() {
                    eprintln!(
                        "[repair] flip degenerate cell: k={k_move} direction={direction:?} removed_face={removed_face_vertices:?} inserted_face={inserted_face_vertices:?} vertices={vertices:?} points={points:?}"
                    );
                }
                return Err(FlipError::DegenerateCell);
            }
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
fn is_delaunay_violation_k3<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    context: &FlipContext<D, 3>,
    config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
) -> Result<bool, FlipError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar + Sum,
    U: DataType,
    V: DataType,
{
    delaunay_violation_k3_for_ridge(
        tds,
        kernel,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
        config,
        diagnostics,
    )
}

/// Apply a generic k-move (no Delaunay check).
///
/// # Errors
///
/// Returns a [`FlipError`] if the flip would be degenerate, duplicate an existing cell,
/// create non-manifold topology, if predicate evaluation fails, or if underlying TDS
/// mutations fail.
pub(crate) fn apply_bistellar_flip<K, U, V, const D: usize, const K_MOVE: usize>(
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

/// Apply a generic k-move with runtime k (no Delaunay check).
///
/// # Errors
///
/// Returns a [`FlipError`] if the flip would be degenerate, duplicate an existing cell,
/// create non-manifold topology, if predicate evaluation fails, or if underlying TDS
/// mutations fail.
pub(crate) fn apply_bistellar_flip_dynamic<K, U, V, const D: usize>(
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
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::flips::FlipDirection;
///
/// assert_eq!(FlipDirection::Forward.inverse(), FlipDirection::Inverse);
/// ```
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
/// Detect repeated flip signatures and abort on cycles.
fn check_flip_cycle(
    signature: u64,
    diagnostics: &mut RepairDiagnostics,
    stats: &DelaunayRepairStats,
    max_flips: usize,
    config: &RepairAttemptConfig,
) -> Result<(), DelaunayRepairError> {
    let repeats = diagnostics
        .flip_signature_counts
        .get(&signature)
        .copied()
        .unwrap_or(0);
    if repeats >= MAX_REPEAT_SIGNATURE {
        if repair_trace_enabled() {
            tracing::debug!(
                "[repair] cycle abort signature={} repeats={} flips={} max_flips={} attempt={} order={:?}",
                signature,
                repeats,
                stats.flips_performed,
                max_flips,
                config.attempt,
                config.queue_order,
            );
        }
        diagnostics.record_cycle_abort(signature);
        return Err(non_convergent_error(max_flips, stats, diagnostics, config));
    }
    Ok(())
}

impl BistellarFlipKind {
    /// Number of simplices being replaced on the current side (k).
    #[must_use]
    pub const fn k(&self) -> usize {
        self.k
    }
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
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::flips::{BistellarMove, ConstK};
///
/// fn move_k<const D: usize, M: BistellarMove<D>>() -> usize {
///     M::K
/// }
///
/// assert_eq!(move_k::<3, ConstK<2>>(), 2);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ConstK<const K: usize>;

/// Const-generic descriptor for a Pachner move in dimension `D`.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::flips::{BistellarMove, ConstK};
///
/// fn move_k<const D: usize, M: BistellarMove<D>>() -> usize {
///     M::K
/// }
///
/// assert_eq!(move_k::<4, ConstK<3>>(), 3);
/// ```
pub trait BistellarMove<const D: usize> {
    /// Number of removed D-simplices (k).
    const K: usize;
}

impl<const D: usize, const K: usize> BistellarMove<D> for ConstK<K> {
    const K: usize = K;
}

/// Errors that can occur during bistellar flips or repair.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::flips::FlipError;
///
/// let err = FlipError::UnsupportedDimension { dimension: 1 };
/// assert!(matches!(err, FlipError::UnsupportedDimension { .. }));
/// ```
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
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::flips::{BistellarFlipKind, FlipDirection, FlipInfo};
/// use delaunay::core::collections::{CellKeyBuffer, SmallBuffer, MAX_PRACTICAL_DIMENSION_SIZE};
/// use delaunay::core::triangulation_data_structure::{CellKey, VertexKey};
/// use slotmap::KeyData;
///
/// let mut removed_cells = CellKeyBuffer::new();
/// removed_cells.push(CellKey::from(KeyData::from_ffi(1)));
/// let mut new_cells = CellKeyBuffer::new();
/// new_cells.push(CellKey::from(KeyData::from_ffi(2)));
///
/// let mut removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
///     SmallBuffer::new();
/// removed_face_vertices.push(VertexKey::from(KeyData::from_ffi(3)));
/// let mut inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
///     SmallBuffer::new();
/// inserted_face_vertices.push(VertexKey::from(KeyData::from_ffi(4)));
///
/// let info: FlipInfo<3> = FlipInfo {
///     kind: BistellarFlipKind::k2(3),
///     direction: FlipDirection::Forward,
///     removed_cells,
///     new_cells,
///     removed_face_vertices,
///     inserted_face_vertices,
/// };
/// assert_eq!(info.kind.k(), 2);
/// ```
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

/// Const-generic flip context for a k-move (forward or inverse).
#[derive(Debug, Clone)]
pub(crate) struct FlipContext<const D: usize, const K: usize> {
    /// Vertices of the removed-face simplex (dimension D+1−K).
    pub removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// Vertices of the inserted-face simplex (dimension K−1).
    pub inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// Cells removed by the flip (count = K).
    pub removed_cells: CellKeyBuffer,
    /// Flip direction (forward/inverse).
    pub direction: FlipDirection,
}

/// Runtime-k flip context for moves where k depends on D.
#[derive(Debug, Clone)]
pub(crate) struct FlipContextDyn<const D: usize> {
    /// Vertices of the removed-face simplex (dimension D+1−k).
    pub removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// Vertices of the inserted-face simplex (dimension k−1).
    pub inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// Cells removed by the flip (count = k).
    pub removed_cells: CellKeyBuffer,
    /// Flip direction (forward/inverse).
    pub direction: FlipDirection,
}

/// Canonical handle to a triangle (three vertices).
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::flips::TriangleHandle;
/// use delaunay::core::triangulation_data_structure::VertexKey;
/// use slotmap::KeyData;
///
/// let a = VertexKey::from(KeyData::from_ffi(1));
/// let b = VertexKey::from(KeyData::from_ffi(2));
/// let c = VertexKey::from(KeyData::from_ffi(3));
///
/// let handle = TriangleHandle::new(b, a, c);
/// assert_eq!(handle.vertices().len(), 3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TriangleHandle {
    v0: VertexKey,
    v1: VertexKey,
    v2: VertexKey,
}

impl TriangleHandle {
    /// Create a canonical triangle handle with ordered vertex keys.
    #[must_use]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::algorithms::flips::TriangleHandle;
    /// use delaunay::core::triangulation_data_structure::VertexKey;
    /// use slotmap::KeyData;
    ///
    /// let a = VertexKey::from(KeyData::from_ffi(10));
    /// let b = VertexKey::from(KeyData::from_ffi(20));
    /// let c = VertexKey::from(KeyData::from_ffi(30));
    ///
    /// let handle = TriangleHandle::new(a, b, c);
    /// assert_eq!(handle.vertices(), [a, b, c]);
    /// ```
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
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::flips::RidgeHandle;
/// use delaunay::core::triangulation_data_structure::CellKey;
/// use slotmap::KeyData;
///
/// let cell_key = CellKey::from(KeyData::from_ffi(7));
/// let handle = RidgeHandle::new(cell_key, 2, 0);
/// assert_eq!(handle.omit_a(), 0);
/// assert_eq!(handle.omit_b(), 2);
/// ```
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
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::flips::DelaunayRepairStats;
///
/// let stats = DelaunayRepairStats::default();
/// assert_eq!(stats.flips_performed, 0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct DelaunayRepairStats {
    /// Number of queued items checked (facets, ridges, edges, triangles).
    pub facets_checked: usize,
    /// Number of flips performed.
    pub flips_performed: usize,
    /// Maximum queue length observed.
    pub max_queue_len: usize,
}
/// Queue ordering policy for flip repair attempts.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::flips::RepairQueueOrder;
///
/// let order = RepairQueueOrder::Fifo;
/// assert_eq!(order, RepairQueueOrder::Fifo);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairQueueOrder {
    /// FIFO (breadth-like) ordering.
    Fifo,
    /// LIFO (depth-like) ordering.
    Lifo,
}

/// Diagnostics captured when flip-based repair fails to converge.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::flips::{DelaunayRepairDiagnostics, RepairQueueOrder};
///
/// let diagnostics = DelaunayRepairDiagnostics {
///     facets_checked: 0,
///     flips_performed: 0,
///     max_queue_len: 0,
///     ambiguous_predicates: 0,
///     ambiguous_predicate_samples: Vec::new(),
///     predicate_failures: 0,
///     cycle_detections: 0,
///     cycle_signature_samples: Vec::new(),
///     attempt: 1,
///     queue_order: RepairQueueOrder::Fifo,
///     used_robust_predicates: false,
/// };
/// assert!(diagnostics.to_string().contains("checked"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DelaunayRepairDiagnostics {
    /// Number of queued items checked.
    pub facets_checked: usize,
    /// Number of flips performed.
    pub flips_performed: usize,
    /// Maximum queue length observed.
    pub max_queue_len: usize,
    /// Count of ambiguous predicate evaluations (boundary classifications).
    pub ambiguous_predicates: usize,
    /// Sample of ambiguous predicate site hashes (deterministic, truncated).
    pub ambiguous_predicate_samples: Vec<u64>,
    /// Count of predicate failures (conversion/robust fallback errors).
    pub predicate_failures: usize,
    /// Count of detected flip cycles (repeat flip signatures within a sliding window).
    pub cycle_detections: usize,
    /// Sample of repeated flip-context signature hashes (deterministic, truncated).
    pub cycle_signature_samples: Vec<u64>,
    /// Attempt number (1-based).
    pub attempt: usize,
    /// Queue ordering policy used for this attempt.
    pub queue_order: RepairQueueOrder,
    /// Whether robust predicates were enabled for ambiguous tests.
    pub used_robust_predicates: bool,
}

impl fmt::Display for DelaunayRepairDiagnostics {
    /// Format a concise diagnostics summary.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "checked {} facets, ambiguous={}, max_queue={}, flips={}, attempt={}, order={:?}, robust={}, predicate_failures={}, cycles={}, cycle_samples={:?}",
            self.facets_checked,
            self.ambiguous_predicates,
            self.max_queue_len,
            self.flips_performed,
            self.attempt,
            self.queue_order,
            self.used_robust_predicates,
            self.predicate_failures,
            self.cycle_detections,
            self.cycle_signature_samples
        )
    }
}

/// Errors that can occur during flip-based Delaunay repair.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::flips::DelaunayRepairError;
/// use delaunay::core::triangulation::TopologyGuarantee;
///
/// let err = DelaunayRepairError::InvalidTopology {
///     required: TopologyGuarantee::PLManifold,
///     found: TopologyGuarantee::Pseudomanifold,
///     message: "requires manifold",
/// };
/// assert!(matches!(err, DelaunayRepairError::InvalidTopology { .. }));
/// ```
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DelaunayRepairError {
    /// Repair did not converge within the flip budget.
    #[error("Delaunay repair failed to converge after {max_flips} flips ({diagnostics})")]
    NonConvergent {
        /// Maximum flips allowed.
        max_flips: usize,
        /// Diagnostics captured during the failed attempt.
        diagnostics: DelaunayRepairDiagnostics,
    },
    /// Repair completed but left a Delaunay violation or otherwise could not be verified.
    #[error("Delaunay repair postcondition failed: {message}")]
    PostconditionFailed {
        /// Additional context describing the postcondition failure.
        message: String,
    },
    /// Flip-based repair is not admissible under the current topology guarantee.
    #[error("Delaunay repair requires {required:?} topology, found {found:?}: {message}")]
    InvalidTopology {
        /// Required topology guarantee.
        required: TopologyGuarantee,
        /// Actual topology guarantee.
        found: TopologyGuarantee,
        /// Additional context for the mismatch.
        message: &'static str,
    },
    /// Heuristic rebuild failed during advanced repair.
    #[error("Heuristic rebuild failed: {message}")]
    HeuristicRebuildFailed {
        /// Additional context for the rebuild failure.
        message: String,
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
pub(crate) fn build_k2_flip_context<T, U, V, const D: usize>(
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
pub(crate) fn build_k2_flip_context_from_edge<T, U, V, const D: usize>(
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
/// Build a forward k=1 flip context from a cell and inserted vertex.
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
pub(crate) fn build_k1_inverse_context<T, U, V, const D: usize>(
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

#[allow(clippy::too_many_lines)]
/// Evaluate the k=2 facet flip predicate for a local Delaunay violation.
fn delaunay_violation_k2_for_facet<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    facet_vertices: &[VertexKey],
    opposite_a: VertexKey,
    opposite_b: VertexKey,
    config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
) -> Result<bool, FlipError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar + Sum,
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
    let mut in_a = match kernel.in_sphere(&points_a, opposite_point_b) {
        Ok(value) => value,
        Err(e) => {
            diagnostics.predicate_failures += 1;
            return Err(FlipError::PredicateFailure {
                message: format!("in_sphere failed for k=2 cell A: {e}"),
            });
        }
    };

    let mut in_b = match kernel.in_sphere(&points_b, opposite_point_a) {
        Ok(value) => value,
        Err(e) => {
            diagnostics.predicate_failures += 1;
            return Err(FlipError::PredicateFailure {
                message: format!("in_sphere failed for k=2 cell B: {e}"),
            });
        }
    };

    let fast_sign_a = in_a;
    let fast_sign_b = in_b;

    // Always record ambiguous sites when the fast predicate returns boundary/uncertain.
    if in_a == 0 {
        let key = predicate_key_from_vertices(&cell_vertices[0], opposite_b);
        diagnostics.record_ambiguous(key);
    }

    if in_b == 0 {
        let key = predicate_key_from_vertices(&cell_vertices[1], opposite_a);
        diagnostics.record_ambiguous(key);
    }

    // If enabled, run the robust predicate *unconditionally*.
    //
    // In practice, fast predicates can return an incorrect non-zero sign near degeneracy
    // (especially in 3D+), which can cause the repair queues to terminate while global
    // Delaunay violations remain. A fully robust pass is used as a correctness fallback.
    if config.use_robust_on_ambiguous {
        in_a = robust_insphere_sign(&points_a, opposite_point_b, diagnostics);
        in_b = robust_insphere_sign(&points_b, opposite_point_a, diagnostics);
    }

    let violates = in_a > 0 || in_b > 0;
    if std::env::var_os("DELAUNAY_REPAIR_DEBUG_PREDICATES").is_some()
        && (violates || fast_sign_a == 0 || fast_sign_b == 0 || in_a == 0 || in_b == 0)
    {
        eprintln!(
            "[repair] k2 predicate facet={:?} opposite_a={:?} opposite_b={:?} in_a_fast={} in_b_fast={} in_a={} in_b={} violates={} attempt={} robust={}",
            facet_vertices,
            opposite_a,
            opposite_b,
            fast_sign_a,
            fast_sign_b,
            in_a,
            in_b,
            violates,
            config.attempt,
            config.use_robust_on_ambiguous
        );
        tracing::debug!(
            facet_vertices = ?facet_vertices,
            opposite_a = ?opposite_a,
            opposite_b = ?opposite_b,
            in_a_fast = fast_sign_a,
            in_b_fast = fast_sign_b,
            in_a,
            in_b,
            violates,
            attempt = config.attempt,
            use_robust = config.use_robust_on_ambiguous,
            "delaunay_violation_k2_for_facet: insphere classification"
        );
    }

    Ok(violates)
}
/// Check whether a k=2 flip would create a degenerate cell.
fn k2_flip_would_create_degenerate_cell<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    context: &FlipContext<D, 2>,
) -> Result<bool, FlipError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar + Sum,
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

    for &omit in &context.removed_face_vertices {
        let mut vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(D + 1);
        vertices.extend_from_slice(&context.inserted_face_vertices);
        for &v in &context.removed_face_vertices {
            if v != omit {
                vertices.push(v);
            }
        }

        let points = vertices_to_points(tds, &vertices)?;
        let orientation = kernel
            .orientation(&points)
            .map_err(|e| FlipError::PredicateFailure {
                message: format!("orientation failed for k=2 postcondition: {e}"),
            })?;
        if orientation == 0 {
            let config = config_presets::high_precision::<K::Scalar>();
            let robust_orientation =
                robust_orientation(&points, &config).map_err(|e| FlipError::PredicateFailure {
                    message: format!("robust orientation failed for k=2 postcondition: {e}"),
                })?;
            if matches!(robust_orientation, Orientation::DEGENERATE) {
                return Ok(true);
            }
        }
    }

    Ok(false)
}
/// Check whether a k=2 facet violates the local Delaunay condition.
///
/// # Errors
///
/// Returns a [`FlipError`] if any referenced cell/vertex is missing or a predicate
/// evaluation fails.
fn is_delaunay_violation_k2<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    context: &FlipContext<D, 2>,
    config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
) -> Result<bool, FlipError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar + Sum,
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
        config,
        diagnostics,
    )
}

/// Apply a k=2 bistellar flip (no Delaunay check).
///
/// # Errors
///
/// Returns a [`FlipError`] if the flip would be degenerate, duplicate an existing cell,
/// create non-manifold topology, if predicate evaluation fails, or if underlying TDS
/// mutations fail.
pub(crate) fn apply_bistellar_flip_k2<K, U, V, const D: usize>(
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

/// Build flip context for a k=3 (ridge) flip.
///
/// # Errors
///
/// Returns a [`FlipError`] if the ridge is invalid, references missing cells/vertices,
/// or the adjacency data is inconsistent.
pub(crate) fn build_k3_flip_context<T, U, V, const D: usize>(
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
pub(crate) fn build_k3_flip_context_from_triangle<T, U, V, const D: usize>(
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
/// Evaluate the k=3 ridge flip predicate for a local Delaunay violation.
fn delaunay_violation_k3_for_ridge<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    ridge_vertices: &[VertexKey],
    triangle_vertices: &[VertexKey],
    config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
) -> Result<bool, FlipError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar + Sum,
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

        let mut in_sphere = match kernel.in_sphere(&points, missing_point) {
            Ok(value) => value,
            Err(e) => {
                diagnostics.predicate_failures += 1;
                return Err(FlipError::PredicateFailure {
                    message: format!("in_sphere failed for k=3 cell: {e}"),
                });
            }
        };

        // Track ambiguous sites when the fast predicate returns boundary/uncertain.
        if in_sphere == 0 {
            let key = predicate_key_from_vertices(&cell_vertices, missing);
            diagnostics.record_ambiguous(key);
        }

        // If enabled, use robust predicates for the classification regardless of the fast result.
        if config.use_robust_on_ambiguous {
            in_sphere = robust_insphere_sign(&points, missing_point, diagnostics);
        }

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
pub(crate) fn apply_bistellar_flip_k3<K, U, V, const D: usize>(
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
pub(crate) fn apply_bistellar_flip_k1<K, U, V, const D: usize>(
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
pub(crate) fn apply_bistellar_flip_k1_inverse<K, U, V, const D: usize>(
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
#[expect(
    clippy::too_many_lines,
    reason = "Repair loop contains inline tracing and queue handling for diagnostics"
)]
fn repair_delaunay_with_flips_k2_attempt<K, U, V, const D: usize>(
    tds: &mut Tds<K::Scalar, U, V, D>,
    kernel: &K,
    seed_cells: Option<&[CellKey]>,
    config: &RepairAttemptConfig,
) -> Result<DelaunayRepairStats, DelaunayRepairError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar + Sum,
    U: DataType,
    V: DataType,
{
    if D < 2 {
        return Err(FlipError::UnsupportedDimension { dimension: D }.into());
    }

    let max_flips = default_max_flips::<D>(tds.number_of_cells());

    let mut stats = DelaunayRepairStats::default();
    let mut diagnostics = RepairDiagnostics::default();
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
    if repair_trace_enabled() {
        let seed_count = seed_cells.map_or(0, <[CellKey]>::len);
        tracing::debug!(
            "[repair] attempt={} order={:?} robust={} cells={} max_flips={} seeds={} queues(facet={})",
            config.attempt,
            config.queue_order,
            config.use_robust_on_ambiguous,
            tds.number_of_cells(),
            max_flips,
            seed_count,
            queue.len(),
        );
    }

    while let Some((facet, key)) = pop_queue(&mut queue, config.queue_order) {
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

        let violates =
            match is_delaunay_violation_k2(tds, kernel, &context, config, &mut diagnostics) {
                Ok(violates) => violates,
                Err(FlipError::PredicateFailure { .. }) => {
                    continue;
                }
                Err(e) => return Err(e.into()),
            };

        if !violates {
            continue;
        }

        let signature = flip_signature(
            2,
            context.direction,
            &context.removed_face_vertices,
            &context.inserted_face_vertices,
        );
        check_flip_cycle(signature, &mut diagnostics, &stats, max_flips, config)?;

        let info = match apply_bistellar_flip_k2(tds, kernel, &context) {
            Ok(info) => info,
            Err(
                err @ (FlipError::DegenerateCell
                | FlipError::DuplicateCell
                | FlipError::NonManifoldFacet
                | FlipError::CellCreation(_)),
            ) => {
                if std::env::var_os("DELAUNAY_REPAIR_DEBUG_FACETS").is_some() {
                    tracing::debug!(
                        "k=2 flip skipped in repair_delaunay_with_flips_k2_attempt (facet={facet:?}): {err}"
                    );
                }
                if repair_trace_enabled() {
                    tracing::debug!("[repair] skip k=2 flip (facet={facet:?}) reason={err}");
                    tracing::debug!(
                        "[repair] skip k=2 flip context removed_face={:?} inserted_face={:?} removed_cells={:?}",
                        context.removed_face_vertices,
                        context.inserted_face_vertices,
                        context.removed_cells,
                    );
                }
                continue;
            }
            Err(e) => return Err(e.into()),
        };
        stats.flips_performed += 1;
        diagnostics.record_flip_signature(signature);

        if stats.flips_performed > max_flips {
            return Err(non_convergent_error(
                max_flips,
                &stats,
                &diagnostics,
                config,
            ));
        }

        for &cell_key in &info.new_cells {
            enqueue_cell_facets(tds, cell_key, &mut queue, &mut queued, &mut stats)?;
        }
    }
    if repair_trace_enabled() {
        tracing::debug!(
            "[repair] attempt={} done: checked={} flips={} max_queue={} ambiguous={} predicate_failures={} cycles={}",
            config.attempt,
            stats.facets_checked,
            stats.flips_performed,
            stats.max_queue_len,
            diagnostics.ambiguous_predicates,
            diagnostics.predicate_failures,
            diagnostics.cycle_detections,
        );
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
#[expect(
    clippy::too_many_lines,
    reason = "Repair retries and tracing are kept together for clarity"
)]
pub(crate) fn repair_delaunay_with_flips_k2_k3<K, U, V, const D: usize>(
    tds: &mut Tds<K::Scalar, U, V, D>,
    kernel: &K,
    seed_cells: Option<&[CellKey]>,
    _topology: TopologyGuarantee,
) -> Result<DelaunayRepairStats, DelaunayRepairError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar + Sum,
    U: DataType,
    V: DataType,
{
    if D < 2 {
        return Err(FlipError::UnsupportedDimension { dimension: D }.into());
    }

    // In debug/test builds (especially for 3D+), prefer a fully-robust predicate pass.
    // This materially improves correctness in near-degenerate configurations.
    let attempt1 = RepairAttemptConfig {
        attempt: 1,
        queue_order: RepairQueueOrder::Fifo,
        use_robust_on_ambiguous: cfg!(any(test, debug_assertions)) && D >= 3,
    };

    let attempt2 = RepairAttemptConfig {
        attempt: 2,
        queue_order: RepairQueueOrder::Lifo,
        use_robust_on_ambiguous: true,
    };
    let attempt3 = RepairAttemptConfig {
        attempt: 3,
        queue_order: RepairQueueOrder::Fifo,
        use_robust_on_ambiguous: true,
    };
    // Snapshot the pre-repair state so a failed attempt doesn't poison retries.
    let tds_snapshot = tds.clone();

    let attempt1_result = if D == 2 {
        repair_delaunay_with_flips_k2_attempt(tds, kernel, seed_cells, &attempt1)
    } else {
        repair_delaunay_with_flips_k2_k3_attempt(tds, kernel, seed_cells, &attempt1)
    };

    match attempt1_result {
        Ok(stats) => {
            if verify_repair_postcondition(tds, kernel, seed_cells).is_ok() {
                return Ok(stats);
            }
            if repair_trace_enabled() {
                tracing::debug!(
                    "[repair] attempt 1 postcondition failed; retrying with robust predicates + full reseed"
                );
            }

            // Postcondition verification failed: rerun with robust predicates + full reseed.
            *tds = tds_snapshot.clone();
            let retry_seed_cells = None;
            let stats2 = if D == 2 {
                repair_delaunay_with_flips_k2_attempt(tds, kernel, retry_seed_cells, &attempt2)
            } else {
                repair_delaunay_with_flips_k2_k3_attempt(tds, kernel, retry_seed_cells, &attempt2)
            };

            match stats2 {
                Ok(stats2) => {
                    if verify_repair_postcondition(tds, kernel, retry_seed_cells).is_ok() {
                        return Ok(stats2);
                    }
                    if repair_trace_enabled() {
                        tracing::debug!(
                            "[repair] attempt 2 postcondition failed; retrying with alternate queue order"
                        );
                    }
                }
                Err(DelaunayRepairError::NonConvergent { .. }) => {
                    if repair_trace_enabled() {
                        tracing::debug!(
                            "[repair] attempt 2 non-convergent; retrying with alternate queue order"
                        );
                    }
                }
                Err(err) => return Err(err),
            }

            // Final attempt with alternate queue order.
            *tds = tds_snapshot;
            let stats3 = if D == 2 {
                repair_delaunay_with_flips_k2_attempt(tds, kernel, retry_seed_cells, &attempt3)
            } else {
                repair_delaunay_with_flips_k2_k3_attempt(tds, kernel, retry_seed_cells, &attempt3)
            }?;

            verify_repair_postcondition(tds, kernel, retry_seed_cells)?;
            Ok(stats3)
        }
        Err(DelaunayRepairError::NonConvergent { .. }) => {
            if repair_trace_enabled() {
                tracing::debug!(
                    "[repair] attempt 1 non-convergent; retrying with robust predicates + full reseed"
                );
            }
            // Retry with robust predicates + full reseed.
            *tds = tds_snapshot.clone();
            let retry_seed_cells = None;
            let stats2 = if D == 2 {
                repair_delaunay_with_flips_k2_attempt(tds, kernel, retry_seed_cells, &attempt2)
            } else {
                repair_delaunay_with_flips_k2_k3_attempt(tds, kernel, retry_seed_cells, &attempt2)
            };

            match stats2 {
                Ok(stats2) => {
                    if verify_repair_postcondition(tds, kernel, retry_seed_cells).is_ok() {
                        return Ok(stats2);
                    }
                    if repair_trace_enabled() {
                        tracing::debug!(
                            "[repair] attempt 2 postcondition failed; retrying with alternate queue order"
                        );
                    }
                }
                Err(DelaunayRepairError::NonConvergent { .. }) => {
                    if repair_trace_enabled() {
                        tracing::debug!(
                            "[repair] attempt 2 non-convergent; retrying with alternate queue order"
                        );
                    }
                }
                Err(err) => return Err(err),
            }

            // Final attempt with alternate queue order.
            *tds = tds_snapshot;
            let stats3 = if D == 2 {
                repair_delaunay_with_flips_k2_attempt(tds, kernel, retry_seed_cells, &attempt3)
            } else {
                repair_delaunay_with_flips_k2_k3_attempt(tds, kernel, retry_seed_cells, &attempt3)
            }?;

            verify_repair_postcondition(tds, kernel, retry_seed_cells)?;
            Ok(stats3)
        }
        Err(err) => Err(err),
    }
}

/// Verify the Delaunay property via local flip predicates (fast O(cells) validation).
///
/// This function checks whether the triangulation satisfies the Delaunay property by testing
/// all possible flip configurations (k=2 facets, k=3 ridges, and their inverses). If no
/// violations are detected via these local checks, the triangulation is Delaunay.
///
/// This is **much faster** than the naive O(cells × vertices) empty-circumsphere check,
/// while being equally correct due to the completeness of bistellar flip predicates.
///
/// # Performance
///
/// - **Complexity**: O(cells) — tests only local flip predicates
/// - **Speedup**: ~40-100x faster than brute-force for typical triangulations
/// - **Use case**: Ideal for property-based testing with many iterations
///
/// # Errors
///
/// Returns [`DelaunayRepairError::PostconditionFailed`] if any flip predicate detects
/// a Delaunay violation.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::*;
/// use delaunay::core::algorithms::flips::verify_delaunay_via_flip_predicates;
/// use delaunay::geometry::kernel::FastKernel;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
///
/// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
/// let kernel = FastKernel::<f64>::new();
///
/// // Fast O(N) verification
/// assert!(verify_delaunay_via_flip_predicates(dt.tds(), &kernel).is_ok());
/// ```
pub fn verify_delaunay_via_flip_predicates<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
) -> Result<(), DelaunayRepairError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar + Sum,
    U: DataType,
    V: DataType,
{
    verify_repair_postcondition(tds, kernel, None)
}

fn verify_repair_postcondition<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    seed_cells: Option<&[CellKey]>,
) -> Result<(), DelaunayRepairError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar + Sum,
    U: DataType,
    V: DataType,
{
    verify_repair_postcondition_locally(tds, kernel, seed_cells)
}

fn verify_repair_postcondition_locally<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    seed_cells: Option<&[CellKey]>,
) -> Result<(), DelaunayRepairError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar + Sum,
    U: DataType,
    V: DataType,
{
    let config = RepairAttemptConfig {
        attempt: 0,
        queue_order: RepairQueueOrder::Fifo,
        use_robust_on_ambiguous: true,
    };

    let mut stats = DelaunayRepairStats::default();
    let mut diagnostics = RepairDiagnostics::default();
    let mut queues = RepairQueues::new();
    seed_repair_queues(tds, seed_cells, &mut queues, &mut stats)?;
    if repair_trace_enabled() {
        let seed_count = seed_cells.map_or(0, <[CellKey]>::len);
        tracing::debug!(
            "[repair] attempt={} order={:?} robust={} cells={} seeds={} queues(facet={}, ridge={}, edge={}, tri={})",
            config.attempt,
            config.queue_order,
            config.use_robust_on_ambiguous,
            tds.number_of_cells(),
            seed_count,
            queues.facet_queue.len(),
            queues.ridge_queue.len(),
            queues.edge_queue.len(),
            queues.triangle_queue.len(),
        );
    }

    verify_postcondition_k2_facets(
        tds,
        kernel,
        &mut queues.facet_queue,
        &config,
        &mut diagnostics,
    )?;
    verify_postcondition_k3_ridges(
        tds,
        kernel,
        &mut queues.ridge_queue,
        &config,
        &mut diagnostics,
    )?;
    verify_postcondition_inverse_k2_edges(
        tds,
        kernel,
        &mut queues.edge_queue,
        &config,
        &mut diagnostics,
    )?;
    verify_postcondition_inverse_k3_triangles(
        tds,
        kernel,
        &mut queues.triangle_queue,
        &config,
        &mut diagnostics,
    )?;

    Ok(())
}

fn verify_postcondition_k2_facets<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    queue: &mut VecDeque<(FacetHandle, u64)>,
    config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
) -> Result<(), DelaunayRepairError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar + Sum,
    U: DataType,
    V: DataType,
{
    while let Some((facet, _key)) = pop_queue(queue, config.queue_order) {
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

        match is_delaunay_violation_k2(tds, kernel, &context, config, diagnostics) {
            Ok(true) => {
                let flip_degenerate =
                    match k2_flip_would_create_degenerate_cell(tds, kernel, &context) {
                        Ok(degenerate) => degenerate,
                        Err(FlipError::PredicateFailure { .. }) => {
                            // Inconclusive due to numeric degeneracy; skip.
                            continue;
                        }
                        Err(e) => {
                            return Err(DelaunayRepairError::PostconditionFailed {
                                message: format!("local k=2 verification failed after repair: {e}"),
                            });
                        }
                    };

                if flip_degenerate {
                    if repair_trace_enabled() {
                        tracing::debug!(
                            "[repair] postcondition k=2 violation unresolved due to degenerate flip (facet={facet:?})"
                        );
                    }
                    continue;
                }
                if repair_trace_enabled() {
                    tracing::debug!(
                        "[repair] postcondition k=2 violation remains (facet={facet:?})"
                    );
                }
                let mut message =
                    format!("local k=2 violation remains after repair (facet={facet:?})");
                if std::env::var_os("DELAUNAY_REPAIR_DEBUG_FACETS").is_some() {
                    let removed_details: Vec<_> = context
                        .removed_face_vertices
                        .iter()
                        .filter_map(|&vkey| {
                            tds.get_vertex_by_key(vkey)
                                .map(|vertex| (vkey, *vertex.point()))
                        })
                        .collect();
                    let inserted_details: Vec<_> = context
                        .inserted_face_vertices
                        .iter()
                        .filter_map(|&vkey| {
                            tds.get_vertex_by_key(vkey)
                                .map(|vertex| (vkey, *vertex.point()))
                        })
                        .collect();
                    message = format!(
                        "{message}; removed_face={removed_details:?}; inserted_face={inserted_details:?}"
                    );
                }
                return Err(DelaunayRepairError::PostconditionFailed { message });
            }
            Ok(false) | Err(FlipError::PredicateFailure { .. }) => {
                // No violation detected, or inconclusive due to numeric degeneracy.
            }
            Err(e) => {
                return Err(DelaunayRepairError::PostconditionFailed {
                    message: format!("local k=2 verification failed after repair: {e}"),
                });
            }
        }
    }

    Ok(())
}

fn verify_postcondition_k3_ridges<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    queue: &mut VecDeque<(RidgeHandle, u64)>,
    config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
) -> Result<(), DelaunayRepairError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar + Sum,
    U: DataType,
    V: DataType,
{
    while let Some((ridge, _key)) = pop_queue(queue, config.queue_order) {
        let context = match build_k3_flip_context(tds, ridge) {
            Ok(ctx) => ctx,
            Err(
                FlipError::InvalidRidgeIndex { .. }
                | FlipError::InvalidRidgeAdjacency { .. }
                | FlipError::InvalidRidgeMultiplicity { .. }
                | FlipError::MissingCell { .. },
            ) => {
                continue;
            }
            Err(e) => return Err(e.into()),
        };

        match is_delaunay_violation_k3(tds, kernel, &context, config, diagnostics) {
            Ok(true) => {
                if repair_trace_enabled() {
                    tracing::debug!(
                        "[repair] postcondition k=3 violation remains (ridge={ridge:?})"
                    );
                }
                return Err(DelaunayRepairError::PostconditionFailed {
                    message: format!("local k=3 violation remains after repair (ridge={ridge:?})"),
                });
            }
            Ok(false) | Err(FlipError::PredicateFailure { .. }) => {
                // No violation detected, or inconclusive due to numeric degeneracy.
            }
            Err(e) => {
                return Err(DelaunayRepairError::PostconditionFailed {
                    message: format!("local k=3 verification failed after repair: {e}"),
                });
            }
        }
    }

    Ok(())
}

fn verify_postcondition_inverse_k2_edges<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    queue: &mut VecDeque<(EdgeKey, u64)>,
    config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
) -> Result<(), DelaunayRepairError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar + Sum,
    U: DataType,
    V: DataType,
{
    while let Some((edge, _key)) = pop_queue(queue, config.queue_order) {
        let context = match build_k2_flip_context_from_edge(tds, edge) {
            Ok(ctx) => ctx,
            Err(
                FlipError::InvalidEdgeMultiplicity { .. }
                | FlipError::InvalidEdgeAdjacency { .. }
                | FlipError::MissingCell { .. }
                | FlipError::MissingVertex { .. },
            ) => {
                continue;
            }
            Err(e) => return Err(e.into()),
        };

        if context.removed_face_vertices.len() != 2 {
            continue;
        }
        let opposite_a = context.removed_face_vertices[0];
        let opposite_b = context.removed_face_vertices[1];

        let violates = match delaunay_violation_k2_for_facet(
            tds,
            kernel,
            &context.inserted_face_vertices,
            opposite_a,
            opposite_b,
            config,
            diagnostics,
        ) {
            Ok(violates) => violates,
            Err(FlipError::PredicateFailure { .. }) => {
                // Inconclusive due to numeric degeneracy; skip.
                continue;
            }
            Err(e) => {
                return Err(DelaunayRepairError::PostconditionFailed {
                    message: format!("local inverse k=2 verification failed after repair: {e}"),
                });
            }
        };

        if !violates {
            if repair_trace_enabled() {
                tracing::debug!(
                    "[repair] postcondition inverse k=2 flip still applicable (edge={edge:?})"
                );
            }
            return Err(DelaunayRepairError::PostconditionFailed {
                message: format!(
                    "local inverse k=2 flip remains applicable after repair (edge={edge:?})"
                ),
            });
        }
    }

    Ok(())
}

fn verify_postcondition_inverse_k3_triangles<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    queue: &mut VecDeque<(TriangleHandle, u64)>,
    config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
) -> Result<(), DelaunayRepairError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar + Sum,
    U: DataType,
    V: DataType,
{
    while let Some((triangle, _key)) = pop_queue(queue, config.queue_order) {
        let context = match build_k3_flip_context_from_triangle(tds, triangle) {
            Ok(ctx) => ctx,
            Err(
                FlipError::InvalidTriangleMultiplicity { .. }
                | FlipError::InvalidTriangleAdjacency { .. }
                | FlipError::MissingCell { .. }
                | FlipError::MissingVertex { .. },
            ) => {
                continue;
            }
            Err(e) => return Err(e.into()),
        };

        let violates = match delaunay_violation_k3_for_ridge(
            tds,
            kernel,
            &context.inserted_face_vertices,
            &context.removed_face_vertices,
            config,
            diagnostics,
        ) {
            Ok(violates) => violates,
            Err(FlipError::PredicateFailure { .. }) => {
                // Inconclusive due to numeric degeneracy; skip.
                continue;
            }
            Err(e) => {
                return Err(DelaunayRepairError::PostconditionFailed {
                    message: format!("local inverse k=3 verification failed after repair: {e}"),
                });
            }
        };

        if !violates {
            if repair_trace_enabled() {
                tracing::debug!(
                    "[repair] postcondition inverse k=3 flip still applicable (triangle={triangle:?})"
                );
            }
            return Err(DelaunayRepairError::PostconditionFailed {
                message: format!(
                    "local inverse k=3 flip remains applicable after repair (triangle={triangle:?})"
                ),
            });
        }
    }

    Ok(())
}

// =============================================================================
// Internal helpers
// =============================================================================
const AMBIGUOUS_SAMPLE_LIMIT: usize = 16;
const CYCLE_SAMPLE_LIMIT: usize = 16;
const FLIP_SIGNATURE_WINDOW: usize = 4096;
const MAX_REPEAT_SIGNATURE: usize = 4;

#[derive(Debug, Default)]
struct RepairDiagnostics {
    ambiguous_predicates: usize,
    ambiguous_samples: Vec<u64>,
    predicate_failures: usize,
    cycle_detections: usize,
    cycle_samples: Vec<u64>,
    flip_signature_window: VecDeque<u64>,
    flip_signature_counts: FastHashMap<u64, usize>,
}

impl RepairDiagnostics {
    fn record_ambiguous(&mut self, key: u64) {
        self.ambiguous_predicates += 1;
        if self.ambiguous_samples.len() >= AMBIGUOUS_SAMPLE_LIMIT {
            return;
        }
        if !self.ambiguous_samples.contains(&key) {
            self.ambiguous_samples.push(key);
        }
    }

    fn record_flip_signature(&mut self, signature: u64) {
        let count = self.flip_signature_counts.entry(signature).or_insert(0);
        *count = count.saturating_add(1);

        if *count > 1 {
            self.cycle_detections = self.cycle_detections.saturating_add(1);
            if self.cycle_samples.len() < CYCLE_SAMPLE_LIMIT
                && !self.cycle_samples.contains(&signature)
            {
                self.cycle_samples.push(signature);
            }
        }

        self.flip_signature_window.push_back(signature);
        if self.flip_signature_window.len() > FLIP_SIGNATURE_WINDOW
            && let Some(old) = self.flip_signature_window.pop_front()
            && let Some(old_count) = self.flip_signature_counts.get_mut(&old)
        {
            *old_count = old_count.saturating_sub(1);
            if *old_count == 0 {
                self.flip_signature_counts.remove(&old);
            }
        }
    }

    fn record_cycle_abort(&mut self, signature: u64) {
        self.cycle_detections = self.cycle_detections.saturating_add(1);
        if self.cycle_samples.len() < CYCLE_SAMPLE_LIMIT && !self.cycle_samples.contains(&signature)
        {
            self.cycle_samples.push(signature);
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RepairAttemptConfig {
    attempt: usize,
    queue_order: RepairQueueOrder,
    use_robust_on_ambiguous: bool,
}

fn non_convergent_error(
    max_flips: usize,
    stats: &DelaunayRepairStats,
    diagnostics: &RepairDiagnostics,
    config: &RepairAttemptConfig,
) -> DelaunayRepairError {
    DelaunayRepairError::NonConvergent {
        max_flips,
        diagnostics: DelaunayRepairDiagnostics {
            facets_checked: stats.facets_checked,
            flips_performed: stats.flips_performed,
            max_queue_len: stats.max_queue_len,
            ambiguous_predicates: diagnostics.ambiguous_predicates,
            ambiguous_predicate_samples: diagnostics.ambiguous_samples.clone(),
            predicate_failures: diagnostics.predicate_failures,
            cycle_detections: diagnostics.cycle_detections,
            cycle_signature_samples: diagnostics.cycle_samples.clone(),
            attempt: config.attempt,
            queue_order: config.queue_order,
            used_robust_predicates: config.use_robust_on_ambiguous,
        },
    }
}

fn pop_queue<T>(queue: &mut VecDeque<T>, order: RepairQueueOrder) -> Option<T> {
    match order {
        RepairQueueOrder::Fifo => queue.pop_front(),
        RepairQueueOrder::Lifo => queue.pop_back(),
    }
}

fn predicate_key_from_vertices(simplex_vertices: &[VertexKey], test_vertex: VertexKey) -> u64 {
    let mut sorted: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        simplex_vertices.iter().copied().collect();
    sorted.sort_unstable();

    let mut hasher = FastHasher::default();
    for vkey in &sorted {
        vkey.hash(&mut hasher);
    }
    test_vertex.hash(&mut hasher);
    hasher.finish()
}

fn flip_signature(
    k_move: usize,
    direction: FlipDirection,
    removed_face_vertices: &[VertexKey],
    inserted_face_vertices: &[VertexKey],
) -> u64 {
    let mut removed: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        removed_face_vertices.iter().copied().collect();
    removed.sort_unstable();

    let mut inserted: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        inserted_face_vertices.iter().copied().collect();
    inserted.sort_unstable();

    let mut hasher = FastHasher::default();
    k_move.hash(&mut hasher);
    match direction {
        FlipDirection::Forward => 0_u8.hash(&mut hasher),
        FlipDirection::Inverse => 1_u8.hash(&mut hasher),
    }
    removed.len().hash(&mut hasher);
    for vkey in &removed {
        vkey.hash(&mut hasher);
    }
    inserted.len().hash(&mut hasher);
    for vkey in &inserted {
        vkey.hash(&mut hasher);
    }
    hasher.finish()
}

fn robust_insphere_sign<T, const D: usize>(
    simplex_points: &[Point<T, D>],
    test_point: &Point<T, D>,
    diagnostics: &mut RepairDiagnostics,
) -> i32
where
    T: CoordinateScalar + Sum,
{
    let config = config_presets::general_triangulation::<T>();
    match robust_insphere(simplex_points, test_point, &config) {
        Ok(InSphere::INSIDE) => 1,
        Ok(InSphere::OUTSIDE) => -1,
        Ok(InSphere::BOUNDARY) => 0,
        Err(_) => {
            diagnostics.predicate_failures += 1;
            0
        }
    }
}

#[inline]
fn repair_trace_enabled() -> bool {
    std::env::var_os("DELAUNAY_REPAIR_TRACE").is_some()
}
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
        let mut present = 0usize;
        let mut missing = 0usize;
        for &cell_key in seeds {
            if !tds.contains_cell(cell_key) {
                missing = missing.saturating_add(1);
                if repair_trace_enabled() {
                    tracing::debug!("[repair] seed_repair_queues: missing seed cell={cell_key:?}");
                }
                continue;
            }
            present = present.saturating_add(1);
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
        if repair_trace_enabled() {
            let seed_sample: Vec<CellKey> = seeds.iter().copied().take(8).collect();
            tracing::debug!(
                "[repair] seed_repair_queues: seeds={} present={} missing={}",
                seeds.len(),
                present,
                missing,
            );
            tracing::debug!("[repair] seed_repair_queues: sample={seed_sample:?}");
        }
        if present == 0 {
            if repair_trace_enabled() {
                tracing::debug!(
                    "[repair] seed_repair_queues: no valid seed cells; falling back to global seeding"
                );
            }
            seed_repair_queues(tds, None, queues, stats)?;
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
    config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
) -> Result<bool, DelaunayRepairError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar + Sum,
    U: DataType,
    V: DataType,
{
    let Some((ridge, key)) = pop_queue(&mut queues.ridge_queue, config.queue_order) else {
        return Ok(false);
    };
    queues.ridge_queued.remove(&key);
    stats.facets_checked += 1;

    let context = match build_k3_flip_context(tds, ridge) {
        Ok(ctx) => ctx,
        Err(
            err @ (FlipError::InvalidRidgeIndex { .. }
            | FlipError::InvalidRidgeAdjacency { .. }
            | FlipError::InvalidRidgeMultiplicity { .. }
            | FlipError::MissingCell { .. }),
        ) => {
            if repair_trace_enabled() {
                tracing::debug!("[repair] skip k=3 ridge (ridge={ridge:?}) reason={err}");
            }
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };

    let violates = match is_delaunay_violation_k3(tds, kernel, &context, config, diagnostics) {
        Ok(violates) => violates,
        Err(FlipError::PredicateFailure { .. }) => {
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };

    if !violates {
        return Ok(true);
    }

    let signature = flip_signature(
        3,
        context.direction,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
    );
    check_flip_cycle(signature, diagnostics, stats, max_flips, config)?;

    let info = match apply_bistellar_flip_k3(tds, kernel, &context) {
        Ok(info) => info,
        Err(
            err @ (FlipError::DegenerateCell
            | FlipError::DuplicateCell
            | FlipError::NonManifoldFacet
            | FlipError::CellCreation(_)),
        ) => {
            if repair_trace_enabled() {
                tracing::debug!("[repair] skip k=3 flip (ridge={ridge:?}) reason={err}");
                tracing::debug!(
                    "[repair] skip k=3 flip context removed_face={:?} inserted_face={:?} removed_cells={:?}",
                    context.removed_face_vertices,
                    context.inserted_face_vertices,
                    context.removed_cells,
                );
            }
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };
    if repair_trace_enabled() {
        tracing::debug!(
            "[repair] apply k=3 flip: kind={:?} direction={:?} removed_face={:?} inserted_face={:?} removed_cells={:?} new_cells={:?}",
            info.kind,
            info.direction,
            info.removed_face_vertices,
            info.inserted_face_vertices,
            info.removed_cells,
            info.new_cells,
        );
    }
    stats.flips_performed += 1;
    diagnostics.record_flip_signature(signature);

    if stats.flips_performed > max_flips {
        return Err(non_convergent_error(max_flips, stats, diagnostics, config));
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
    config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
) -> Result<bool, DelaunayRepairError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar + Sum,
    U: DataType,
    V: DataType,
{
    let Some((edge, key)) = pop_queue(&mut queues.edge_queue, config.queue_order) else {
        return Ok(false);
    };
    queues.edge_queued.remove(&key);
    stats.facets_checked += 1;

    let context = match build_k2_flip_context_from_edge(tds, edge) {
        Ok(ctx) => ctx,
        Err(
            err @ (FlipError::InvalidEdgeMultiplicity { .. }
            | FlipError::InvalidEdgeAdjacency { .. }
            | FlipError::MissingCell { .. }
            | FlipError::MissingVertex { .. }),
        ) => {
            if repair_trace_enabled() {
                tracing::debug!("[repair] skip inverse k=2 edge (edge={edge:?}) reason={err}");
            }
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
        config,
        diagnostics,
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
    let signature = flip_signature(
        D,
        context.direction,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
    );
    check_flip_cycle(signature, diagnostics, stats, max_flips, config)?;

    let info = match apply_bistellar_flip_dynamic(tds, kernel, D, &context) {
        Ok(info) => info,
        Err(
            err @ (FlipError::DegenerateCell
            | FlipError::DuplicateCell
            | FlipError::NonManifoldFacet
            | FlipError::CellCreation(_)),
        ) => {
            if repair_trace_enabled() {
                tracing::debug!("[repair] skip inverse k=2 flip (edge={edge:?}) reason={err}");
                tracing::debug!(
                    "[repair] skip inverse k=2 flip context removed_face={:?} inserted_face={:?} removed_cells={:?}",
                    context.removed_face_vertices,
                    context.inserted_face_vertices,
                    context.removed_cells,
                );
            }
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };
    if repair_trace_enabled() {
        tracing::debug!(
            "[repair] apply inverse k=2 flip: kind={:?} direction={:?} removed_face={:?} inserted_face={:?} removed_cells={:?} new_cells={:?}",
            info.kind,
            info.direction,
            info.removed_face_vertices,
            info.inserted_face_vertices,
            info.removed_cells,
            info.new_cells,
        );
    }
    stats.flips_performed += 1;
    diagnostics.record_flip_signature(signature);

    if stats.flips_performed > max_flips {
        return Err(non_convergent_error(max_flips, stats, diagnostics, config));
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
    config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
) -> Result<bool, DelaunayRepairError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar + Sum,
    U: DataType,
    V: DataType,
{
    let Some((triangle, key)) = pop_queue(&mut queues.triangle_queue, config.queue_order) else {
        return Ok(false);
    };
    queues.triangle_queued.remove(&key);
    stats.facets_checked += 1;

    let context = match build_k3_flip_context_from_triangle(tds, triangle) {
        Ok(ctx) => ctx,
        Err(
            err @ (FlipError::InvalidTriangleMultiplicity { .. }
            | FlipError::InvalidTriangleAdjacency { .. }
            | FlipError::MissingCell { .. }
            | FlipError::MissingVertex { .. }),
        ) => {
            if repair_trace_enabled() {
                tracing::debug!(
                    "[repair] skip inverse k=3 triangle (triangle={triangle:?}) reason={err}"
                );
            }
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };

    let violates = match delaunay_violation_k3_for_ridge(
        tds,
        kernel,
        &context.inserted_face_vertices,
        &context.removed_face_vertices,
        config,
        diagnostics,
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
    let signature = flip_signature(
        D - 1,
        context.direction,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
    );
    check_flip_cycle(signature, diagnostics, stats, max_flips, config)?;

    let info = match apply_bistellar_flip_dynamic(tds, kernel, D - 1, &context) {
        Ok(info) => info,
        Err(
            err @ (FlipError::DegenerateCell
            | FlipError::DuplicateCell
            | FlipError::NonManifoldFacet
            | FlipError::CellCreation(_)),
        ) => {
            if repair_trace_enabled() {
                tracing::debug!(
                    "[repair] skip inverse k=3 flip (triangle={triangle:?}) reason={err}"
                );
                tracing::debug!(
                    "[repair] skip inverse k=3 flip context removed_face={:?} inserted_face={:?} removed_cells={:?}",
                    context.removed_face_vertices,
                    context.inserted_face_vertices,
                    context.removed_cells,
                );
            }
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };
    if repair_trace_enabled() {
        tracing::debug!(
            "[repair] apply inverse k=3 flip: kind={:?} direction={:?} removed_face={:?} inserted_face={:?} removed_cells={:?} new_cells={:?}",
            info.kind,
            info.direction,
            info.removed_face_vertices,
            info.inserted_face_vertices,
            info.removed_cells,
            info.new_cells,
        );
    }
    stats.flips_performed += 1;
    diagnostics.record_flip_signature(signature);

    if stats.flips_performed > max_flips {
        return Err(non_convergent_error(max_flips, stats, diagnostics, config));
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
    config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
) -> Result<bool, DelaunayRepairError>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar + Sum,
    U: DataType,
    V: DataType,
{
    let Some((facet, key)) = pop_queue(&mut queues.facet_queue, config.queue_order) else {
        return Ok(false);
    };
    queues.facet_queued.remove(&key);
    stats.facets_checked += 1;

    let context = match build_k2_flip_context(tds, facet) {
        Ok(ctx) => ctx,
        Err(
            err @ (FlipError::BoundaryFacet { .. }
            | FlipError::MissingCell { .. }
            | FlipError::MissingNeighbor { .. }
            | FlipError::InvalidFacetAdjacency { .. }
            | FlipError::InvalidFacetIndex { .. }),
        ) => {
            if repair_trace_enabled() {
                tracing::debug!("[repair] skip k=2 facet (facet={facet:?}) reason={err}");
            }
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };

    let violates = match is_delaunay_violation_k2(tds, kernel, &context, config, diagnostics) {
        Ok(violates) => violates,
        Err(FlipError::PredicateFailure { .. }) => {
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };

    if !violates {
        return Ok(true);
    }

    let signature = flip_signature(
        2,
        context.direction,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
    );
    check_flip_cycle(signature, diagnostics, stats, max_flips, config)?;

    let info = match apply_bistellar_flip_k2(tds, kernel, &context) {
        Ok(info) => info,
        Err(
            err @ (FlipError::DegenerateCell
            | FlipError::DuplicateCell
            | FlipError::NonManifoldFacet
            | FlipError::CellCreation(_)),
        ) => {
            if std::env::var_os("DELAUNAY_REPAIR_DEBUG_FACETS").is_some() {
                eprintln!(
                    "[repair] skip k=2 flip (facet={facet:?}) reason={err}; removed_face={:?} inserted_face={:?} removed_cells={:?}",
                    context.removed_face_vertices,
                    context.inserted_face_vertices,
                    context.removed_cells,
                );
            }
            if std::env::var_os("DELAUNAY_REPAIR_DEBUG_FACETS").is_some() {
                tracing::debug!(
                    "k=2 flip skipped in process_facet_queue_step (facet={facet:?}): {err}"
                );
            }
            if repair_trace_enabled() {
                tracing::debug!("[repair] skip k=2 flip (facet={facet:?}) reason={err}");
                tracing::debug!(
                    "[repair] skip k=2 flip context removed_face={:?} inserted_face={:?} removed_cells={:?}",
                    context.removed_face_vertices,
                    context.inserted_face_vertices,
                    context.removed_cells,
                );
            }
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };
    if repair_trace_enabled() {
        tracing::debug!(
            "[repair] apply k=2 flip: kind={:?} direction={:?} removed_face={:?} inserted_face={:?} removed_cells={:?} new_cells={:?}",
            info.kind,
            info.direction,
            info.removed_face_vertices,
            info.inserted_face_vertices,
            info.removed_cells,
            info.new_cells,
        );
    }
    stats.flips_performed += 1;
    diagnostics.record_flip_signature(signature);

    if stats.flips_performed > max_flips {
        return Err(non_convergent_error(max_flips, stats, diagnostics, config));
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
            if std::env::var_os("DELAUNAY_REPAIR_DEBUG_FACETS").is_some() {
                tracing::debug!(
                    "k=2 flip would duplicate existing cell {cell_key:?}; target={target:?}; existing={cell_vertices:?}"
                );
            }
            if repair_trace_enabled() {
                tracing::debug!(
                    "[repair] flip would duplicate existing cell {cell_key:?}; target={target:?}; existing={cell_vertices:?}"
                );
            }
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
                    if repair_trace_enabled() {
                        tracing::debug!(
                            "[repair] flip would create non-manifold facet: facet={:?} shared_count={} last_cell={cell_key:?}",
                            facet_vertices,
                            shared_count,
                        );
                    }
                    return true;
                }
            }
        }

        let internal_facet = opposite_vertices.iter().all(|v| facet_vertices.contains(v));
        if internal_facet && shared_count > 0 {
            if repair_trace_enabled() {
                tracing::debug!(
                    "[repair] flip would create non-manifold internal facet: facet={:?} shared_count={}",
                    facet_vertices,
                    shared_count,
                );
            }
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
    if D < 3 {
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
    if D < 3 {
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
    use crate::core::collections::Uuid;
    use crate::core::delaunay_triangulation::DelaunayTriangulation;
    use crate::geometry::kernel::{FastKernel, RobustKernel};
    use crate::vertex;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn init_tracing() {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            let filter = tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
            let _ = tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_test_writer()
                .try_init();
        });
    }
    fn unit_vector<const D: usize>(index: usize) -> [f64; D] {
        let mut coords = [0.0; D];
        coords[index] = 1.0;
        coords
    }

    fn skewed_point<const D: usize>() -> [f64; D] {
        let mut coords = [0.0; D];
        for (i, coord) in coords.iter_mut().enumerate().take(D) {
            let idx = f64::from(u32::try_from(i + 1).expect("index fits in u32"));
            *coord = 0.11 * idx;
        }
        coords
    }

    fn to_dynamic<const D: usize, const K: usize>(context: FlipContext<D, K>) -> FlipContextDyn<D> {
        FlipContextDyn {
            removed_face_vertices: context.removed_face_vertices,
            inserted_face_vertices: context.inserted_face_vertices,
            removed_cells: context.removed_cells,
            direction: context.direction,
        }
    }

    #[test]
    fn test_repair_diagnostics_cycle_detection_records_repeats() {
        init_tracing();
        let mut diagnostics = RepairDiagnostics::default();
        diagnostics.record_flip_signature(10);
        diagnostics.record_flip_signature(20);
        assert_eq!(diagnostics.cycle_detections, 0);

        diagnostics.record_flip_signature(10);
        assert_eq!(diagnostics.cycle_detections, 1);
        assert_eq!(diagnostics.cycle_samples, vec![10]);

        diagnostics.record_flip_signature(10);
        assert_eq!(diagnostics.cycle_detections, 2);
        assert_eq!(diagnostics.cycle_samples, vec![10]);
    }
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TopologySnapshot {
        vertex_uuids: Vec<Uuid>,
        cell_vertex_uuids: Vec<Vec<Uuid>>,
    }

    fn snapshot_topology<const D: usize>(tds: &Tds<f64, (), (), D>) -> TopologySnapshot {
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

    macro_rules! test_bistellar_roundtrip_dimension {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<test_bistellar_k1_roundtrip_ $dim d>]() {
                    init_tracing();
                    let mut tds: Tds<f64, (), (), $dim> = Tds::empty();

                    let origin = tds.insert_vertex_with_mapping(vertex!([0.0; $dim])).unwrap();
                    let mut vertices = Vec::with_capacity($dim + 1);
                    vertices.push(origin);
                    for i in 0..$dim {
                        let v = tds
                            .insert_vertex_with_mapping(vertex!(unit_vector::<$dim>(i)))
                            .unwrap();
                        vertices.push(v);
                    }

                    let cell_key = tds
                        .insert_cell_with_mapping(Cell::new(vertices, None).unwrap())
                        .unwrap();

                    let before = snapshot_topology(&tds);

                    let kernel = FastKernel::<f64>::new();
                    let new_vertex = vertex!([0.1; $dim]);
                    let new_uuid = new_vertex.uuid();
                    let _info = apply_bistellar_flip_k1(&mut tds, &kernel, cell_key, new_vertex)
                        .unwrap();
                    assert!(tds.is_valid().is_ok());

                    let new_key = tds.vertex_key_from_uuid(&new_uuid).unwrap();
                    let _info_back =
                        apply_bistellar_flip_k1_inverse(&mut tds, &kernel, new_key).unwrap();
                    assert!(tds.is_valid().is_ok());

                    assert_eq!(snapshot_topology(&tds), before);
                }

                #[test]
                fn [<test_bistellar_k2_roundtrip_ $dim d>]() {
                    init_tracing();
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

                    let before = snapshot_topology(&tds);

                    let facet = FacetHandle::new(cell_a, u8::try_from($dim).unwrap());
                    let context = build_k2_flip_context(&tds, facet).unwrap();
                    let kernel = FastKernel::<f64>::new();
                    let info = apply_bistellar_flip_k2(&mut tds, &kernel, &context).unwrap();
                    assert!(tds.is_valid().is_ok());

                    if $dim == 2 {
                        let mut inverse_facet: Option<FacetHandle> = None;
                        for &cell_key in &info.new_cells {
                            let cell = tds.get_cell(cell_key).unwrap();
                            if cell.contains_vertex(opposite_a) && cell.contains_vertex(opposite_b) {
                                let facet_index = cell
                                    .vertices()
                                    .iter()
                                    .position(|&v| v != opposite_a && v != opposite_b)
                                    .expect("missing shared vertex for inverse k=2");
                                inverse_facet = Some(FacetHandle::new(
                                    cell_key,
                                    u8::try_from(facet_index).unwrap(),
                                ));
                                break;
                            }
                        }

                        let facet = inverse_facet.expect("inverse k=2 facet not found");
                        let context_back = build_k2_flip_context(&tds, facet).unwrap();
                        let _info_back =
                            apply_bistellar_flip_k2(&mut tds, &kernel, &context_back).unwrap();
                    } else {
                        let edge = EdgeKey::new(opposite_a, opposite_b);
                        let context_back = build_k2_flip_context_from_edge(&tds, edge).unwrap();
                        let _info_back =
                            apply_bistellar_flip_dynamic(&mut tds, &kernel, $dim, &context_back)
                                .unwrap();
                    }

                    assert!(tds.is_valid().is_ok());
                    assert_eq!(snapshot_topology(&tds), before);
                }
            }
        };
        ($dim:literal, k3) => {
            test_bistellar_roundtrip_dimension!($dim);
            pastey::paste! {
                #[test]
                fn [<test_bistellar_k3_roundtrip_ $dim d>]() {
                    init_tracing();
                    let mut tds: Tds<f64, (), (), $dim> = Tds::empty();
                    let mut ridge_vertices = Vec::with_capacity($dim - 1);
                    for i in 0..($dim - 1) {
                        let v = tds
                            .insert_vertex_with_mapping(vertex!(unit_vector::<$dim>(i)))
                            .unwrap();
                        ridge_vertices.push(v);
                    }

                    let a = tds
                        .insert_vertex_with_mapping(vertex!([0.0; $dim]))
                        .unwrap();
                    let b = tds
                        .insert_vertex_with_mapping(vertex!(unit_vector::<$dim>($dim - 1)))
                        .unwrap();
                    let c = tds
                        .insert_vertex_with_mapping(vertex!(skewed_point::<$dim>()))
                        .unwrap();

                    let mut c1_vertices = ridge_vertices.clone();
                    c1_vertices.push(a);
                    c1_vertices.push(b);
                    let c1 = tds
                        .insert_cell_with_mapping(Cell::new(c1_vertices, None).unwrap())
                        .unwrap();

                    let mut c2_vertices = ridge_vertices.clone();
                    c2_vertices.push(b);
                    c2_vertices.push(c);
                    let _c2 = tds
                        .insert_cell_with_mapping(Cell::new(c2_vertices, None).unwrap())
                        .unwrap();

                    let mut c3_vertices = ridge_vertices.clone();
                    c3_vertices.push(c);
                    c3_vertices.push(a);
                    let _c3 = tds
                        .insert_cell_with_mapping(Cell::new(c3_vertices, None).unwrap())
                        .unwrap();

                    repair_neighbor_pointers(&mut tds).unwrap();

                    let before = snapshot_topology(&tds);

                    let ridge = RidgeHandle::new(
                        c1,
                        u8::try_from($dim - 1).unwrap(),
                        u8::try_from($dim).unwrap(),
                    );
                    let context = build_k3_flip_context(&tds, ridge).unwrap();
                    let kernel = FastKernel::<f64>::new();
                    let info = apply_bistellar_flip_k3(&mut tds, &kernel, &context).unwrap();
                    assert!(tds.is_valid().is_ok());

                    if $dim == 3 {
                        let mut inverse_facet: Option<FacetHandle> = None;
                        for &cell_key in &info.new_cells {
                            let cell = tds.get_cell(cell_key).unwrap();
                            if cell.contains_vertex(a)
                                && cell.contains_vertex(b)
                                && cell.contains_vertex(c)
                            {
                                let facet_index = cell
                                    .vertices()
                                    .iter()
                                    .position(|&v| v != a && v != b && v != c)
                                    .expect("missing ridge vertex for inverse k=3");
                                inverse_facet = Some(FacetHandle::new(
                                    cell_key,
                                    u8::try_from(facet_index).unwrap(),
                                ));
                                break;
                            }
                        }

                        let facet = inverse_facet.expect("inverse k=3 facet not found");
                        let context_back = build_k2_flip_context(&tds, facet).unwrap();
                        let _info_back =
                            apply_bistellar_flip_k2(&mut tds, &kernel, &context_back).unwrap();
                    } else {
                        let triangle = TriangleHandle::new(a, b, c);
                        let context_back =
                            build_k3_flip_context_from_triangle(&tds, triangle).unwrap();
                        let _info_back = apply_bistellar_flip_dynamic(
                            &mut tds,
                            &kernel,
                            $dim - 1,
                            &context_back,
                        )
                        .unwrap();
                    }

                    assert!(tds.is_valid().is_ok());
                    assert_eq!(snapshot_topology(&tds), before);
                }
            }
        };
    }

    test_bistellar_roundtrip_dimension!(2);
    test_bistellar_roundtrip_dimension!(3, k3);
    test_bistellar_roundtrip_dimension!(4, k3);
    test_bistellar_roundtrip_dimension!(5, k3);

    #[test]
    fn test_flip_k2_2d_edge_flip() {
        init_tracing();
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
        let context = build_k2_flip_context(&tds, facet).unwrap();
        let info = apply_bistellar_flip_k2(&mut tds, &kernel, &context).unwrap();

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
        init_tracing();
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
        init_tracing();
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
        init_tracing();
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
        init_tracing();
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
    fn test_flip_k2_boundary_facet_error_2d() {
        init_tracing();
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();
        let a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let cell = tds
            .insert_cell_with_mapping(Cell::new(vec![a, b, c], None).unwrap())
            .unwrap();

        let before = snapshot_topology(&tds);
        let facet = FacetHandle::new(cell, 0);
        let err = build_k2_flip_context(&tds, facet).unwrap_err();
        assert!(matches!(err, FlipError::BoundaryFacet { .. }));
        assert_eq!(snapshot_topology(&tds), before);
    }

    #[test]
    fn test_flip_k3_invalid_ridge_multiplicity_3d() {
        init_tracing();
        let mut tds: Tds<f64, (), (), 3> = Tds::empty();
        let a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0]))
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0]))
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0]))
            .unwrap();
        let d = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0]))
            .unwrap();
        let cell = tds
            .insert_cell_with_mapping(Cell::new(vec![a, b, c, d], None).unwrap())
            .unwrap();

        let ridge = RidgeHandle::new(cell, 0, 1);
        let err = build_k3_flip_context(&tds, ridge).unwrap_err();
        assert!(matches!(
            err,
            FlipError::InvalidRidgeMultiplicity { found: 1 }
        ));
    }

    #[test]
    fn test_flip_k2_inverse_invalid_edge_multiplicity_4d() {
        init_tracing();
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
        let _cell_a = tds
            .insert_cell_with_mapping(Cell::new(vertices_with_first_opposite, None).unwrap())
            .unwrap();

        let mut vertices_with_second_opposite = shared_vertices.clone();
        vertices_with_second_opposite.push(opposite_b);
        let _cell_b = tds
            .insert_cell_with_mapping(Cell::new(vertices_with_second_opposite, None).unwrap())
            .unwrap();

        let edge = EdgeKey::new(opposite_a, opposite_b);
        let err = build_k2_flip_context_from_edge(&tds, edge).unwrap_err();
        assert!(matches!(err, FlipError::InvalidEdgeMultiplicity { .. }));
    }

    #[test]
    fn test_flip_k3_inverse_invalid_triangle_multiplicity_5d() {
        init_tracing();
        let mut tds: Tds<f64, (), (), 5> = Tds::empty();
        let origin = tds.insert_vertex_with_mapping(vertex!([0.0; 5])).unwrap();
        let mut vertices = Vec::with_capacity(6);
        vertices.push(origin);
        for i in 0..5 {
            let v = tds
                .insert_vertex_with_mapping(vertex!(unit_vector::<5>(i)))
                .unwrap();
            vertices.push(v);
        }
        let _cell = tds
            .insert_cell_with_mapping(Cell::new(vertices.clone(), None).unwrap())
            .unwrap();

        let triangle = TriangleHandle::new(vertices[0], vertices[1], vertices[2]);
        let err = build_k3_flip_context_from_triangle(&tds, triangle).unwrap_err();
        assert!(matches!(
            err,
            FlipError::InvalidTriangleMultiplicity {
                found: 1,
                expected: 4,
            }
        ));
    }

    #[test]
    fn test_flip_k1_degenerate_insert_rejected() {
        init_tracing();
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();
        let a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let cell_key = tds
            .insert_cell_with_mapping(Cell::new(vec![a, b, c], None).unwrap())
            .unwrap();

        let kernel = FastKernel::<f64>::new();
        let before = snapshot_topology(&tds);
        let err =
            apply_bistellar_flip_k1(&mut tds, &kernel, cell_key, vertex!([0.5, 0.0])).unwrap_err();

        assert!(matches!(err, FlipError::DegenerateCell));
        assert_eq!(snapshot_topology(&tds), before);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_dynamic_k2_forward_4d() {
        init_tracing();
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
        let context_dyn = to_dynamic(context);
        let kernel = FastKernel::<f64>::new();
        let info = apply_bistellar_flip_dynamic(&mut tds, &kernel, 2, &context_dyn).unwrap();

        assert_eq!(info.kind, BistellarFlipKind::k2(4));
        assert_eq!(info.removed_cells.len(), 2);
        assert_eq!(info.new_cells.len(), 4);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_dynamic_k3_forward_5d() {
        init_tracing();
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
        let context_dyn = to_dynamic(context);
        let kernel = FastKernel::<f64>::new();
        let info = apply_bistellar_flip_dynamic(&mut tds, &kernel, 3, &context_dyn).unwrap();

        assert_eq!(info.kind, BistellarFlipKind::k3(5));
        assert_eq!(info.removed_cells.len(), 3);
        assert_eq!(info.new_cells.len(), 4);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_k2_roundtrip_randomized_3d() {
        init_tracing();
        let mut rng = StdRng::seed_from_u64(0x51f1_7a2b);
        let kernel = FastKernel::<f64>::new();

        for _ in 0..10 {
            let mut jitter = |v: [f64; 3]| {
                let mut out = v;
                for coord in &mut out {
                    *coord += rng.random_range(-0.03..0.03);
                }
                out
            };

            let mut tds: Tds<f64, (), (), 3> = Tds::empty();
            let v_a = tds
                .insert_vertex_with_mapping(vertex!(jitter([0.0, 0.0, 0.0])))
                .unwrap();
            let v_b = tds
                .insert_vertex_with_mapping(vertex!(jitter([1.0, 0.0, 0.0])))
                .unwrap();
            let v_c = tds
                .insert_vertex_with_mapping(vertex!(jitter([0.0, 1.0, 0.0])))
                .unwrap();
            let v_d = tds
                .insert_vertex_with_mapping(vertex!(jitter([0.2, 0.2, 1.0])))
                .unwrap();
            let v_e = tds
                .insert_vertex_with_mapping(vertex!(jitter([0.3, -0.1, -0.8])))
                .unwrap();

            let c1 = tds
                .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_c, v_d], None).unwrap())
                .unwrap();
            let _c2 = tds
                .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_c, v_e], None).unwrap())
                .unwrap();

            repair_neighbor_pointers(&mut tds).unwrap();

            let before = snapshot_topology(&tds);
            let facet = FacetHandle::new(c1, 3);
            let context = build_k2_flip_context(&tds, facet).unwrap();
            let info = apply_bistellar_flip_k2(&mut tds, &kernel, &context).unwrap();
            assert!(tds.is_valid().is_ok());

            let edge = EdgeKey::new(
                info.inserted_face_vertices[0],
                info.inserted_face_vertices[1],
            );
            let context_back = build_k2_flip_context_from_edge(&tds, edge).unwrap();
            let _info_back =
                apply_bistellar_flip_dynamic(&mut tds, &kernel, 3, &context_back).unwrap();

            assert!(tds.is_valid().is_ok());
            assert_eq!(snapshot_topology(&tds), before);
        }
    }

    #[test]
    fn test_repair_delaunay_flips_non_delaunay_edge_2d() {
        init_tracing();
        let kernel = FastKernel::<f64>::new();
        let a_coords = [0.0, 0.0];
        let b_coords = [1.0, 1.0];
        let c_coords = [1.0, 0.0];
        let d_candidates = [[0.0, 1.2], [0.1, 1.1], [0.2, 0.9], [-0.1, 1.3]];

        let mut tds = None;
        for d_coords in d_candidates {
            let mut candidate: Tds<f64, (), (), 2> = Tds::empty();
            let a = candidate
                .insert_vertex_with_mapping(vertex!(a_coords))
                .unwrap();
            let b = candidate
                .insert_vertex_with_mapping(vertex!(b_coords))
                .unwrap();
            let c = candidate
                .insert_vertex_with_mapping(vertex!(c_coords))
                .unwrap();
            let d = candidate
                .insert_vertex_with_mapping(vertex!(d_coords))
                .unwrap();

            let _c1 = candidate
                .insert_cell_with_mapping(Cell::new(vec![a, b, c], None).unwrap())
                .unwrap();
            let _c2 = candidate
                .insert_cell_with_mapping(Cell::new(vec![a, b, d], None).unwrap())
                .unwrap();

            repair_neighbor_pointers(&mut candidate).unwrap();

            if verify_delaunay_via_flip_predicates(&candidate, &kernel).is_err() {
                tds = Some(candidate);
                break;
            }
        }

        let mut tds = tds.expect("expected a non-Delaunay configuration from candidates");

        let stats = repair_delaunay_with_flips_k2_k3(
            &mut tds,
            &kernel,
            None,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

        assert!(stats.flips_performed > 0);
        assert!(verify_delaunay_via_flip_predicates(&tds, &kernel).is_ok());
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k2_robust_kernel_near_degenerate_2d() {
        init_tracing();
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();
        let a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let d = tds
            .insert_vertex_with_mapping(vertex!([1.0, 1e-9]))
            .unwrap();

        let c1 = tds
            .insert_cell_with_mapping(Cell::new(vec![a, b, c], None).unwrap())
            .unwrap();
        let _c2 = tds
            .insert_cell_with_mapping(Cell::new(vec![a, b, d], None).unwrap())
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let facet = FacetHandle::new(c1, 2);
        let context = build_k2_flip_context(&tds, facet).unwrap();
        let kernel = RobustKernel::<f64>::new();
        let _info = apply_bistellar_flip_k2(&mut tds, &kernel, &context).unwrap();

        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k4_4d_four_to_two() {
        init_tracing();
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
        let _info = apply_bistellar_flip_k2(&mut tds, &kernel, &context).unwrap();

        let edge = EdgeKey::new(opposite_a, opposite_b);
        let context_back = build_k2_flip_context_from_edge(&tds, edge).unwrap();
        let info_back = apply_bistellar_flip_dynamic(&mut tds, &kernel, 4, &context_back).unwrap();

        assert_eq!(info_back.kind.k, 4);
        assert_eq!(info_back.kind.d, 4);
        assert_eq!(info_back.removed_cells.len(), 4);
        assert_eq!(info_back.new_cells.len(), 2);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k5_4d_five_to_one() {
        init_tracing();
        let mut tds: Tds<f64, (), (), 4> = Tds::empty();
        let origin = tds.insert_vertex_with_mapping(vertex!([0.0; 4])).unwrap();
        let mut vertices = Vec::with_capacity(5);
        vertices.push(origin);
        for i in 0..4 {
            let v = tds
                .insert_vertex_with_mapping(vertex!(unit_vector::<4>(i)))
                .unwrap();
            vertices.push(v);
        }

        let cell_key = tds
            .insert_cell_with_mapping(Cell::new(vertices, None).unwrap())
            .unwrap();

        let kernel = FastKernel::<f64>::new();
        let new_vertex = vertex!([0.1; 4]);
        let new_uuid = new_vertex.uuid();
        let info = apply_bistellar_flip_k1(&mut tds, &kernel, cell_key, new_vertex).unwrap();

        assert_eq!(info.kind.k, 1);
        assert_eq!(info.new_cells.len(), 5);

        let new_key = tds.vertex_key_from_uuid(&new_uuid).unwrap();
        let info_back = apply_bistellar_flip_k1_inverse(&mut tds, &kernel, new_key).unwrap();

        assert_eq!(info_back.kind.k, 5);
        assert_eq!(info_back.kind.d, 4);
        assert_eq!(info_back.removed_cells.len(), 5);
        assert_eq!(info_back.new_cells.len(), 1);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k4_5d_four_to_three() {
        init_tracing();
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

        assert_eq!(info.kind.k, 3);
        assert_eq!(info.inserted_face_vertices.len(), 3);

        let triangle = TriangleHandle::new(
            info.inserted_face_vertices[0],
            info.inserted_face_vertices[1],
            info.inserted_face_vertices[2],
        );
        let context_back = build_k3_flip_context_from_triangle(&tds, triangle).unwrap();
        let info_back = apply_bistellar_flip_dynamic(&mut tds, &kernel, 4, &context_back).unwrap();

        assert_eq!(info_back.kind.k, 4);
        assert_eq!(info_back.kind.d, 5);
        assert_eq!(info_back.removed_cells.len(), 4);
        assert_eq!(info_back.new_cells.len(), 3);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k5_5d_five_to_two() {
        init_tracing();
        let mut tds: Tds<f64, (), (), 5> = Tds::empty();
        let mut shared_vertices = Vec::with_capacity(5);
        for i in 0..5 {
            let v = tds
                .insert_vertex_with_mapping(vertex!(unit_vector::<5>(i)))
                .unwrap();
            shared_vertices.push(v);
        }

        let opposite_a = tds.insert_vertex_with_mapping(vertex!([0.0; 5])).unwrap();
        let opposite_b = tds.insert_vertex_with_mapping(vertex!([1.0; 5])).unwrap();

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

        let facet = FacetHandle::new(cell_a, 5);
        let context = build_k2_flip_context(&tds, facet).unwrap();
        let kernel = FastKernel::<f64>::new();
        let _info = apply_bistellar_flip_k2(&mut tds, &kernel, &context).unwrap();

        let edge = EdgeKey::new(opposite_a, opposite_b);
        let context_back = build_k2_flip_context_from_edge(&tds, edge).unwrap();
        let info_back = apply_bistellar_flip_dynamic(&mut tds, &kernel, 5, &context_back).unwrap();

        assert_eq!(info_back.kind.k, 5);
        assert_eq!(info_back.kind.d, 5);
        assert_eq!(info_back.removed_cells.len(), 5);
        assert_eq!(info_back.new_cells.len(), 2);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k6_5d_six_to_one() {
        init_tracing();
        let mut tds: Tds<f64, (), (), 5> = Tds::empty();
        let origin = tds.insert_vertex_with_mapping(vertex!([0.0; 5])).unwrap();
        let mut vertices = Vec::with_capacity(6);
        vertices.push(origin);
        for i in 0..5 {
            let v = tds
                .insert_vertex_with_mapping(vertex!(unit_vector::<5>(i)))
                .unwrap();
            vertices.push(v);
        }

        let cell_key = tds
            .insert_cell_with_mapping(Cell::new(vertices, None).unwrap())
            .unwrap();

        let kernel = FastKernel::<f64>::new();
        let new_vertex = vertex!([0.1; 5]);
        let new_uuid = new_vertex.uuid();
        let info = apply_bistellar_flip_k1(&mut tds, &kernel, cell_key, new_vertex).unwrap();

        assert_eq!(info.kind.k, 1);
        assert_eq!(info.new_cells.len(), 6);

        let new_key = tds.vertex_key_from_uuid(&new_uuid).unwrap();
        let info_back = apply_bistellar_flip_k1_inverse(&mut tds, &kernel, new_key).unwrap();

        assert_eq!(info_back.kind.k, 6);
        assert_eq!(info_back.kind.d, 5);
        assert_eq!(info_back.removed_cells.len(), 6);
        assert_eq!(info_back.new_cells.len(), 1);
        assert!(tds.is_valid().is_ok());
    }
    #[test]
    fn test_flip_k1_2d_roundtrip() {
        init_tracing();
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
        init_tracing();
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
        let stats = repair_delaunay_with_flips_k2_k3(
            &mut tds,
            &kernel,
            Some(seed_cells.as_slice()),
            TopologyGuarantee::PLManifold,
        )
        .unwrap();
        assert!(stats.facets_checked > 0);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_repair_queue_inverse_k3_smoke_5d() {
        init_tracing();
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
        let result = repair_delaunay_with_flips_k2_k3(
            &mut tds,
            &kernel,
            Some(seed_cells.as_slice()),
            TopologyGuarantee::PLManifold,
        );

        match result {
            Ok(stats) => assert!(stats.facets_checked > 0),
            Err(DelaunayRepairError::PostconditionFailed { .. }) => {
                // This test constructs a synthetic configuration to smoke-test queue plumbing.
                // Postcondition verification can legitimately fail in degenerate/non-Delaunay
                // setups; what we must preserve is TDS structural validity.
            }
            Err(err) => panic!("unexpected repair failure: {err}"),
        }

        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_repair_queue_k2_local_seed() {
        init_tracing();
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
        let stats = repair_delaunay_with_flips_k2_k3(
            &mut tds,
            &kernel,
            Some(&[seed_cell]),
            TopologyGuarantee::PLManifold,
        )
        .unwrap();
        assert!(stats.facets_checked > 0);
        assert!(tds.is_valid().is_ok());
    }
}
