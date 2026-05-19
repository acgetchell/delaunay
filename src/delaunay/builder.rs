//! Fluent builder for [`DelaunayTriangulation`] with optional toroidal topology.
//!
//! [`DelaunayTriangulationBuilder`] unifies the existing family of `DelaunayTriangulation`
//! constructors under a single, composable API and adds first-class support for
//! toroidal (periodic) construction.
//!
//! # When to use the builder
//!
//! | Situation | Recommended API |
//! |---|---|
//! | Simple Euclidean, default options | [`DelaunayTriangulation::new`] |
//! | Custom `ConstructionOptions` or `TopologyGuarantee` | [`DelaunayTriangulationBuilder`] |
//! | Toroidal Phase 1 (canonicalize only) | [`DelaunayTriangulationBuilder`] with [`.toroidal()`](DelaunayTriangulationBuilder::toroidal) |
//! | Toroidal Phase 2 (true periodic, χ = 0) | [`DelaunayTriangulationBuilder`] with [`.toroidal_periodic()`](DelaunayTriangulationBuilder::toroidal_periodic) |
//! | Custom kernel (`RobustKernel`, etc.) | [`DelaunayTriangulationBuilder::build_with_kernel`] |
//!
//! # Phase 1 vs Phase 2
//!
//! **Phase 1 (`.toroidal()`):** The builder canonicalizes all input vertices into the
//! fundamental domain `[0, L_i)` before passing them to the standard Euclidean
//! constructor. The resulting triangulation is a valid Euclidean Delaunay triangulation
//! of the canonicalized point set; it does **not** identify opposite boundary facets.
//!
//! **Phase 2 (`.toroidal_periodic()`, issue #210):** Full periodic construction using
//! the 3^D image-point method — generating copies of each point shifted by ±L in each
//! dimension, building the full Euclidean DT on the expanded set, normalizing lifted
//! simplices, searching a closed quotient candidate subset, and rebuilding quotient
//! representatives with periodic neighbor pointers. Produces a true toroidal
//! (χ = 0) triangulation. See `REFERENCES.md`, "Periodic and Toroidal
//! Triangulations", first entry.
//!
//! # Examples
//!
//! ## Standard Euclidean construction
//!
//! ```rust
//! use delaunay::prelude::construction::{DelaunayTriangulationBuilder, vertex};
//!
//! # fn main() -> Result<(), delaunay::prelude::construction::DelaunayTriangulationConstructionError> {
//! let vertices = vec![
//!     vertex!([0.0, 0.0]),
//!     vertex!([1.0, 0.0]),
//!     vertex!([0.0, 1.0]),
//! ];
//!
//! let dt = DelaunayTriangulationBuilder::new(&vertices)
//!     .build::<()>()?;
//!
//! assert_eq!(dt.number_of_vertices(), 3);
//! # Ok(())
//! # }
//! ```
//!
//! ## Toroidal construction (Phase 1: canonicalization only)
//!
//! ```rust
//! use delaunay::prelude::construction::{DelaunayTriangulationBuilder, vertex};
//!
//! # fn main() -> Result<(), delaunay::prelude::construction::DelaunayTriangulationConstructionError> {
//! // Vertices that fall outside [0, 1)² are wrapped before triangulation.
//! let vertices = vec![
//!     vertex!([0.2, 0.3]),
//!     vertex!([1.8, 0.1]),  // x wraps to 0.8
//!     vertex!([0.5, 0.7]),
//!     vertex!([-0.4, 0.9]), // x wraps to 0.6
//! ];
//!
//! let dt = DelaunayTriangulationBuilder::new(&vertices)
//!     .toroidal([1.0, 1.0])
//!     .build::<()>()?;
//!
//! assert_eq!(dt.number_of_vertices(), 4);
//! # Ok(())
//! # }
//! ```
//!
//! ## Toroidal construction (Phase 2: full periodic / image-point method)
//!
//! Uses the 3^D image-point method to produce a true toroidal (χ = 0) triangulation
//! where boundary facets are identified and neighbor pointers are rewired periodically.
//!
//! ```rust,no_run
//! use delaunay::prelude::geometry::RobustKernel;
//! use delaunay::prelude::construction::{DelaunayTriangulationBuilder, vertex};
//!
//! # fn main() -> Result<(), delaunay::prelude::construction::DelaunayTriangulationConstructionError> {
//! let vertices = vec![
//!     vertex!([0.1, 0.2]),
//!     vertex!([0.4, 0.7]),
//!     vertex!([0.7, 0.3]),
//!     vertex!([0.2, 0.9]),
//!     vertex!([0.8, 0.6]),
//!     vertex!([0.5, 0.1]),
//!     vertex!([0.3, 0.5]),
//! ];
//!
//! let kernel = RobustKernel::new();
//! let dt = DelaunayTriangulationBuilder::new(&vertices)
//!     .toroidal_periodic([1.0, 1.0])
//!     .build_with_kernel::<_, ()>(&kernel)?;
//!
//! assert_eq!(dt.number_of_vertices(), 7);
//! // Every vertex has a valid incident simplex (no boundary).
//! assert!(dt.tds().is_valid().is_ok());
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]

use crate::construction::{
    ConstructionOptions, DelaunayTriangulationConstructionError, InitialSimplexStrategy,
    RetryPolicy,
};
use crate::core::algorithms::incremental_insertion::{
    DelaunayRepairErrorKind, InsertionError, InsertionErrorSourceKind,
};
use crate::core::collections::{FastHashMap, PeriodicOffsetBuffer, Uuid, VertexKeySet};
use crate::core::construction::TriangulationConstructionError;
use crate::core::operations::InsertionOutcome;
use crate::core::simplex::{Simplex, SimplexValidationError};
use crate::core::tds::{
    InvariantError, InvariantErrorSummaryDetail, SimplexKey, Tds, TdsConstructionError, TdsError,
    TdsErrorKind, TdsMutationError, TriangulationConstructionState,
    TriangulationValidationErrorKind, VertexKey,
};
use crate::core::traits::data_type::DataType;
use crate::core::util::periodic_facet_key_from_lifted_vertices;
use crate::core::validation::TopologyGuarantee;
use crate::core::vertex::Vertex;
use crate::geometry::kernel::{AdaptiveKernel, Kernel};
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::{Coordinate, CoordinateScalar};
use crate::repair::DelaunayRepairPolicy;
use crate::topology::spaces::toroidal::ToroidalSpace;
use crate::topology::traits::global_topology_model::{
    GlobalTopologyModel, GlobalTopologyModelError,
};
use crate::topology::traits::topological_space::{
    GlobalTopology, TopologyKind, ToroidalConstructionMode,
};
use crate::triangulation::DelaunayTriangulation;
use crate::validation::DelaunayTriangulationValidationError;
use num_traits::ToPrimitive;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::num::NonZeroUsize;
use thiserror::Error;
const TWO_POW_52_I64: i64 = 4_503_599_627_370_496; // 2^52
const TWO_POW_52_F64: f64 = 4_503_599_627_370_496.0; // 2^52
const MAX_OFFSET_UNITS: i64 = 1_048_576;
const IMAGE_JITTER_UNITS: i64 = 64;
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;
type LiftedVertex<const D: usize> = (VertexKey, [i8; D]);
type SymbolicSignature<const D: usize> = Vec<LiftedVertex<D>>;
type PeriodicFacetKey = u64;
type PeriodicCandidate<const D: usize> = (
    SymbolicSignature<D>,
    SymbolicSignature<D>,
    Vec<PeriodicFacetKey>,
    bool,
);
/// Sort candidates by heuristic priority: prefer in-domain candidates first,
/// then by cumulative edge rarity (rarer edges first), then by index.
fn sort_candidates_by_rarity_and_domain(
    order: &mut [usize],
    candidate_edges: &[[usize; 3]],
    candidate_in_domain: &[bool],
    edge_count: usize,
) {
    let mut edge_frequency = vec![0usize; edge_count];
    for edges in candidate_edges {
        for &edge in edges {
            edge_frequency[edge] = edge_frequency[edge].saturating_add(1);
        }
    }

    order.sort_by(|a, b| {
        let a_edges = candidate_edges[*a];
        let b_edges = candidate_edges[*b];
        let a_score =
            edge_frequency[a_edges[0]] + edge_frequency[a_edges[1]] + edge_frequency[a_edges[2]];
        let b_score =
            edge_frequency[b_edges[0]] + edge_frequency[b_edges[1]] + edge_frequency[b_edges[2]];
        candidate_in_domain[*b]
            .cmp(&candidate_in_domain[*a])
            .then_with(|| a_score.cmp(&b_score))
            .then_with(|| a.cmp(b))
    });
}

/// DFS state for bounded face-subset search in [`search_closed_2d_selection`].
///
/// Encapsulates the immutable search parameters and mutable traversal state so
/// the recursive [`search`](Self::search) method takes only `pos` and `chosen`.
struct ClosedSelectionDfs<'a> {
    target_faces: usize,
    order: &'a [usize],
    candidate_edges: &'a [[usize; 3]],
    edge_counts: Vec<u8>,
    selected: Vec<bool>,
    nodes: usize,
    node_limit: usize,
}

impl ClosedSelectionDfs<'_> {
    fn search(&mut self, pos: usize, chosen: usize) -> bool {
        if chosen == self.target_faces {
            return true;
        }
        if pos == self.order.len() {
            return false;
        }
        if chosen + (self.order.len() - pos) < self.target_faces {
            return false;
        }
        if self.nodes >= self.node_limit {
            return false;
        }
        self.nodes = self.nodes.saturating_add(1);

        // Capacity-based prune: each additional face consumes 3 remaining edge incidences.
        let remaining_capacity: usize = self
            .edge_counts
            .iter()
            .map(|&count| usize::from(2_u8.saturating_sub(count)))
            .sum();
        if chosen + (remaining_capacity / 3) < self.target_faces {
            return false;
        }

        let idx = self.order[pos];
        let edges = self.candidate_edges[idx];

        if self.edge_counts[edges[0]] < 2
            && self.edge_counts[edges[1]] < 2
            && self.edge_counts[edges[2]] < 2
        {
            self.selected[idx] = true;
            self.edge_counts[edges[0]] += 1;
            self.edge_counts[edges[1]] += 1;
            self.edge_counts[edges[2]] += 1;

            if self.search(pos + 1, chosen + 1) {
                return true;
            }

            self.edge_counts[edges[0]] -= 1;
            self.edge_counts[edges[1]] -= 1;
            self.edge_counts[edges[2]] -= 1;
            self.selected[idx] = false;
        }

        self.search(pos + 1, chosen)
    }
}

/// Finds a bounded-size 2D face subset whose edge incidences can close a quotient boundary.
///
/// Returns a boolean mask aligned with `candidate_edges` when a selection of exactly
/// `target_faces` candidates is found such that no edge is used more than twice. The search
/// uses a DFS with pruning and a heuristic ordering that prefers in-domain candidates first.
fn search_closed_2d_selection(
    candidate_edges: &[[usize; 3]],
    candidate_in_domain: &[bool],
    target_faces: usize,
    edge_count: usize,
    node_limit: usize,
) -> Option<Vec<bool>> {
    let m = candidate_edges.len();
    if m < target_faces {
        return None;
    }

    let mut order: Vec<usize> = (0..m).collect();
    sort_candidates_by_rarity_and_domain(
        &mut order,
        candidate_edges,
        candidate_in_domain,
        edge_count,
    );

    let mut dfs = ClosedSelectionDfs {
        target_faces,
        order: &order,
        candidate_edges,
        edge_counts: vec![0u8; edge_count],
        selected: vec![false; m],
        nodes: 0,
        node_limit,
    };

    dfs.search(0, 0).then_some(dfs.selected)
}

// =============================================================================
// ERROR TYPES
// =============================================================================

/// TDS failure category preserved by explicit construction without retaining a
/// large by-value source error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExplicitTdsErrorKind {
    /// A vertex failed validation.
    InvalidVertex,
    /// A simplex failed validation.
    InvalidSimplex,
    /// Neighbor relationships were invalid.
    InvalidNeighbors,
    /// Adjacent simplices violated coherent orientation.
    OrientationViolation,
    /// Duplicate maximal simplices were detected.
    DuplicateSimplices,
    /// A facet would be incident to too many simplices.
    FacetSharingViolation,
    /// An entity UUID was duplicated during TDS assembly.
    DuplicateUuid,
    /// Simplex creation failed.
    FailedToCreateSimplex,
    /// Expected neighbor relation was absent.
    NotNeighbors,
    /// UUID-to-key mapping was inconsistent.
    MappingInconsistency,
    /// Simplex vertex-key lookup failed.
    VertexKeyRetrievalFailed,
    /// A referenced simplex key was missing.
    SimplexNotFound,
    /// A referenced vertex key was missing.
    VertexNotFound,
    /// A dimension/count invariant was violated.
    DimensionMismatch,
    /// An index exceeded its valid bound.
    IndexOutOfBounds,
    /// Internal TDS state was inconsistent.
    InconsistentDataStructure,
    /// A geometric validation failure occurred.
    Geometric,
    /// A facet operation failed.
    FacetError,
    /// A simplex contained duplicate coordinates.
    DuplicateCoordinatesInSimplex,
}

/// Compact summary of a TDS construction, mutation, or validation failure used
/// by explicit construction.
///
/// The conversion preserves the [`ExplicitTdsErrorKind`] and rendered
/// diagnostic text while dropping the full typed payload. Use this type when
/// explicit construction needs a small source error; keep the original
/// [`TdsError`], [`TdsConstructionError`], or [`TdsMutationError`] when callers
/// need source-chain or payload inspection.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::tds::TdsError;
/// use delaunay::prelude::construction::{ExplicitTdsError, ExplicitTdsErrorKind};
///
/// let source = TdsError::InconsistentDataStructure {
///     message: "dangling simplex key".to_string(),
/// };
/// let summary = ExplicitTdsError::from(source);
///
/// assert_eq!(summary.kind, ExplicitTdsErrorKind::InconsistentDataStructure);
/// ```
#[must_use]
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("{message}")]
pub struct ExplicitTdsError {
    /// Structured TDS failure category.
    pub kind: ExplicitTdsErrorKind,
    /// Full diagnostic text from the original TDS error.
    pub message: String,
}

impl From<TdsError> for ExplicitTdsError {
    fn from(source: TdsError) -> Self {
        let kind = match &source {
            TdsError::InvalidVertex { .. } => ExplicitTdsErrorKind::InvalidVertex,
            TdsError::InvalidSimplex { .. } => ExplicitTdsErrorKind::InvalidSimplex,
            TdsError::InvalidNeighbors { .. } => ExplicitTdsErrorKind::InvalidNeighbors,
            TdsError::OrientationViolation { .. } => ExplicitTdsErrorKind::OrientationViolation,
            TdsError::DuplicateSimplices { .. } => ExplicitTdsErrorKind::DuplicateSimplices,
            TdsError::FacetSharingViolation { .. } => ExplicitTdsErrorKind::FacetSharingViolation,
            TdsError::FailedToCreateSimplex { .. } => ExplicitTdsErrorKind::FailedToCreateSimplex,
            TdsError::NotNeighbors { .. } => ExplicitTdsErrorKind::NotNeighbors,
            TdsError::MappingInconsistency { .. } => ExplicitTdsErrorKind::MappingInconsistency,
            TdsError::VertexKeyRetrievalFailed { .. } => {
                ExplicitTdsErrorKind::VertexKeyRetrievalFailed
            }
            TdsError::SimplexNotFound { .. } => ExplicitTdsErrorKind::SimplexNotFound,
            TdsError::VertexNotFound { .. } => ExplicitTdsErrorKind::VertexNotFound,
            TdsError::DimensionMismatch { .. } => ExplicitTdsErrorKind::DimensionMismatch,
            TdsError::IndexOutOfBounds { .. } => ExplicitTdsErrorKind::IndexOutOfBounds,
            TdsError::InconsistentDataStructure { .. } => {
                ExplicitTdsErrorKind::InconsistentDataStructure
            }
            TdsError::Geometric(_) => ExplicitTdsErrorKind::Geometric,
            TdsError::FacetError(_) => ExplicitTdsErrorKind::FacetError,
            TdsError::DuplicateCoordinatesInSimplex { .. } => {
                ExplicitTdsErrorKind::DuplicateCoordinatesInSimplex
            }
        };
        Self {
            kind,
            message: source.to_string(),
        }
    }
}

impl From<TdsConstructionError> for ExplicitTdsError {
    fn from(source: TdsConstructionError) -> Self {
        match source {
            TdsConstructionError::ValidationError(source) => source.into(),
            duplicate @ TdsConstructionError::DuplicateUuid { .. } => Self {
                kind: ExplicitTdsErrorKind::DuplicateUuid,
                message: duplicate.to_string(),
            },
        }
    }
}

impl From<TdsMutationError> for ExplicitTdsError {
    fn from(source: TdsMutationError) -> Self {
        TdsError::from(source).into()
    }
}

/// Incremental-insertion failure category preserved by explicit construction
/// without retaining a large by-value source error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExplicitInsertionErrorKind {
    /// Conflict-region search failed.
    ConflictRegion,
    /// Point location failed.
    Location,
    /// Cavity filling failed.
    CavityFilling,
    /// Neighbor wiring failed.
    NeighborWiring,
    /// Non-manifold topology was detected.
    NonManifoldTopology,
    /// Hull extension failed.
    HullExtension,
    /// Delaunay validation failed after insertion.
    DelaunayValidationFailed,
    /// Flip-based Delaunay repair failed.
    DelaunayRepairFailed,
    /// Duplicate coordinates were supplied.
    DuplicateCoordinates,
    /// Duplicate UUID was supplied.
    DuplicateUuid,
    /// TDS topology validation failed.
    TopologyValidation,
    /// Triangulation-layer topology validation failed.
    TopologyValidationFailed,
}

/// Compact summary of an [`InsertionError`] used by explicit construction.
///
/// The conversion preserves the [`ExplicitInsertionErrorKind`], an optional
/// nested [`InsertionErrorSourceKind`] for wrapped validation or repair errors,
/// and rendered diagnostic text. It intentionally drops bulky typed payloads and
/// source chains; keep the original [`InsertionError`] when callers need full
/// structured insertion context.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::construction::{
///     ExplicitInsertionError, ExplicitInsertionErrorKind,
/// };
/// use delaunay::prelude::insertion::InsertionError;
///
/// let source = InsertionError::DuplicateCoordinates {
///     coordinates: "[0.0, 0.0]".to_string(),
/// };
/// let summary = ExplicitInsertionError::from(source);
///
/// assert_eq!(summary.kind, ExplicitInsertionErrorKind::DuplicateCoordinates);
/// assert!(summary.source_kind.is_none());
/// ```
#[must_use]
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("{message}")]
pub struct ExplicitInsertionError {
    /// Structured insertion failure category.
    pub kind: ExplicitInsertionErrorKind,
    /// Nested structured source kind when insertion wraps another validation layer.
    pub source_kind: Option<InsertionErrorSourceKind>,
    /// Full diagnostic text from the original insertion error.
    pub message: String,
}

impl From<InsertionError> for ExplicitInsertionError {
    fn from(source: InsertionError) -> Self {
        let kind = match &source {
            InsertionError::ConflictRegion(_) => ExplicitInsertionErrorKind::ConflictRegion,
            InsertionError::Location(_) => ExplicitInsertionErrorKind::Location,
            InsertionError::CavityFilling { .. } => ExplicitInsertionErrorKind::CavityFilling,
            InsertionError::NeighborWiring { .. } => ExplicitInsertionErrorKind::NeighborWiring,
            InsertionError::NonManifoldTopology { .. } => {
                ExplicitInsertionErrorKind::NonManifoldTopology
            }
            InsertionError::HullExtension { .. } => ExplicitInsertionErrorKind::HullExtension,
            InsertionError::DelaunayValidationFailed { .. } => {
                ExplicitInsertionErrorKind::DelaunayValidationFailed
            }
            InsertionError::DelaunayRepairFailed { .. } => {
                ExplicitInsertionErrorKind::DelaunayRepairFailed
            }
            InsertionError::DuplicateCoordinates { .. } => {
                ExplicitInsertionErrorKind::DuplicateCoordinates
            }
            InsertionError::DuplicateUuid { .. } => ExplicitInsertionErrorKind::DuplicateUuid,
            InsertionError::TopologyValidation(_) => ExplicitInsertionErrorKind::TopologyValidation,
            InsertionError::TopologyValidationFailed { .. } => {
                ExplicitInsertionErrorKind::TopologyValidationFailed
            }
        };
        let source_kind = match &source {
            InsertionError::DelaunayValidationFailed { source } => {
                Some(InsertionErrorSourceKind::Delaunay(source.into()))
            }
            InsertionError::DelaunayRepairFailed { source, .. } => Some(
                InsertionErrorSourceKind::DelaunayRepair(source.as_ref().into()),
            ),
            InsertionError::TopologyValidation(source) => {
                Some(InsertionErrorSourceKind::Tds(source.into()))
            }
            InsertionError::TopologyValidationFailed { source, .. } => {
                Some(InsertionErrorSourceKind::Triangulation(source.into()))
            }
            _ => None,
        };
        Self {
            kind,
            source_kind,
            message: source.to_string(),
        }
    }
}

/// Validation-layer category preserved by explicit construction without
/// retaining a large by-value source error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExplicitInvariantErrorKind {
    /// Level 1-2 TDS validation failed.
    Tds,
    /// Level 3 triangulation topology validation failed.
    Triangulation,
    /// Level 4 Delaunay validation failed.
    Delaunay,
}

/// Compact summary of an [`InvariantError`] used by explicit construction.
///
/// The conversion preserves the validation layer in
/// [`ExplicitInvariantErrorKind`], the nested typed discriminant in
/// [`InvariantErrorSummaryDetail`], and rendered diagnostic text. It
/// intentionally drops bulky typed payloads and source chains; keep the original
/// [`InvariantError`] when callers need full validation context.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::tds::{
///     InvariantError, InvariantErrorSummaryDetail, TdsError, TdsErrorKind,
/// };
/// use delaunay::prelude::construction::{
///     ExplicitInvariantError, ExplicitInvariantErrorKind,
/// };
///
/// let source = InvariantError::Tds(TdsError::InconsistentDataStructure {
///     message: "dangling simplex key".to_string(),
/// });
/// let summary = ExplicitInvariantError::from(source);
///
/// assert_eq!(summary.kind, ExplicitInvariantErrorKind::Tds);
/// assert_eq!(
///     summary.detail,
///     InvariantErrorSummaryDetail::Tds(TdsErrorKind::InconsistentDataStructure),
/// );
/// ```
#[must_use]
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("{message}")]
pub struct ExplicitInvariantError {
    /// Structured validation-layer category.
    pub kind: ExplicitInvariantErrorKind,
    /// Nested structured validation error kind.
    pub detail: InvariantErrorSummaryDetail,
    /// Full diagnostic text from the original invariant error.
    pub message: String,
}

impl From<InvariantError> for ExplicitInvariantError {
    fn from(source: InvariantError) -> Self {
        let kind = match &source {
            InvariantError::Tds(_) => ExplicitInvariantErrorKind::Tds,
            InvariantError::Triangulation(_) => ExplicitInvariantErrorKind::Triangulation,
            InvariantError::Delaunay(_) => ExplicitInvariantErrorKind::Delaunay,
        };
        let detail = match &source {
            InvariantError::Tds(source) => InvariantErrorSummaryDetail::Tds(source.into()),
            InvariantError::Triangulation(source) => {
                InvariantErrorSummaryDetail::Triangulation(source.into())
            }
            InvariantError::Delaunay(source) => {
                InvariantErrorSummaryDetail::Delaunay(source.into())
            }
        };
        Self {
            kind,
            detail,
            message: source.to_string(),
        }
    }
}

/// Delaunay validation category preserved by explicit construction without
/// retaining a large by-value source error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExplicitDelaunayValidationErrorKind {
    /// Lower-layer TDS validation failed.
    Tds,
    /// Lower-layer topology validation failed.
    Triangulation,
    /// Level 4 Delaunay verification failed.
    VerificationFailed,
    /// Legacy string-only repair validation failed.
    RepairFailed,
    /// Typed repair validation failed.
    RepairOperationFailed,
}

/// Nested source category preserved by explicit Delaunay validation summaries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExplicitDelaunayValidationSourceKind {
    /// Lower-layer TDS validation failed.
    Tds(TdsErrorKind),
    /// Lower-layer topology validation failed.
    Triangulation(TriangulationValidationErrorKind),
    /// Typed flip repair failed during a mutating operation.
    Repair(DelaunayRepairErrorKind),
}

/// Compact summary of a [`DelaunayTriangulationValidationError`] used by
/// explicit construction.
///
/// The conversion preserves the [`ExplicitDelaunayValidationErrorKind`], an
/// optional nested [`ExplicitDelaunayValidationSourceKind`] for wrapped
/// validation or repair errors, and rendered diagnostic text. It intentionally
/// drops bulky typed payloads and source chains; keep the original
/// [`DelaunayTriangulationValidationError`] when callers need full validation
/// context.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::construction::{
///     DelaunayTriangulationValidationError, ExplicitDelaunayValidationError,
///     ExplicitDelaunayValidationErrorKind,
/// };
///
/// let source = DelaunayTriangulationValidationError::VerificationFailed {
///     message: "non-Delaunay facet".to_string(),
/// };
/// let summary = ExplicitDelaunayValidationError::from(source);
///
/// assert_eq!(
///     summary.kind,
///     ExplicitDelaunayValidationErrorKind::VerificationFailed,
/// );
/// assert!(summary.source_kind.is_none());
/// ```
#[must_use]
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("{message}")]
pub struct ExplicitDelaunayValidationError {
    /// Structured Delaunay validation category.
    pub kind: ExplicitDelaunayValidationErrorKind,
    /// Nested structured source kind when validation wraps another layer.
    pub source_kind: Option<ExplicitDelaunayValidationSourceKind>,
    /// Full diagnostic text from the original Delaunay validation error.
    pub message: String,
}

impl From<DelaunayTriangulationValidationError> for ExplicitDelaunayValidationError {
    fn from(source: DelaunayTriangulationValidationError) -> Self {
        let kind = match &source {
            DelaunayTriangulationValidationError::Tds(_) => {
                ExplicitDelaunayValidationErrorKind::Tds
            }
            DelaunayTriangulationValidationError::Triangulation(_) => {
                ExplicitDelaunayValidationErrorKind::Triangulation
            }
            DelaunayTriangulationValidationError::VerificationFailed { .. } => {
                ExplicitDelaunayValidationErrorKind::VerificationFailed
            }
            DelaunayTriangulationValidationError::RepairFailed { .. } => {
                ExplicitDelaunayValidationErrorKind::RepairFailed
            }
            DelaunayTriangulationValidationError::RepairOperationFailed { .. } => {
                ExplicitDelaunayValidationErrorKind::RepairOperationFailed
            }
        };
        let source_kind = match &source {
            DelaunayTriangulationValidationError::Tds(source) => Some(
                ExplicitDelaunayValidationSourceKind::Tds(source.as_ref().into()),
            ),
            DelaunayTriangulationValidationError::Triangulation(source) => Some(
                ExplicitDelaunayValidationSourceKind::Triangulation(source.as_ref().into()),
            ),
            DelaunayTriangulationValidationError::RepairOperationFailed { source, .. } => Some(
                ExplicitDelaunayValidationSourceKind::Repair(source.as_ref().into()),
            ),
            DelaunayTriangulationValidationError::VerificationFailed { .. }
            | DelaunayTriangulationValidationError::RepairFailed { .. } => None,
        };
        Self {
            kind,
            source_kind,
            message: source.to_string(),
        }
    }
}

/// Errors from explicit triangulation construction.
///
/// Input validation errors (wrong arity, out-of-bounds indices, duplicate vertices,
/// empty simplices) and post-assembly failures (neighbor wiring, orientation
/// normalization, structural/topology/nondegeneracy validation) are returned as
/// variants of this enum — callers should match on
/// [`ExplicitConstructionError`] (wrapped in
/// [`DelaunayTriangulationConstructionError::ExplicitConstruction`]).
///
/// Low-level explicit assembly failures are normalized into
/// [`ExplicitConstructionError::SimplexCreation`] or
/// [`ExplicitConstructionError::TdsAssembly`] so callers can handle the whole
/// explicit-construction path through
/// [`DelaunayTriangulationConstructionError::ExplicitConstruction`].
///
/// [`DelaunayTriangulationConstructionError::ExplicitConstruction`]: crate::DelaunayTriangulationConstructionError::ExplicitConstruction
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::construction::{
///     DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
///     ExplicitConstructionError, vertex,
/// };
///
/// let vertices = vec![vertex!([0.0, 0.0]), vertex!([1.0, 0.0]), vertex!([0.0, 1.0])];
/// let simplices = vec![vec![0, 1]]; // Wrong arity for 2D (needs 3 vertices)
///
/// assert!(matches!(
///     DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices).build::<()>(),
///     Err(DelaunayTriangulationConstructionError::ExplicitConstruction(
///         ExplicitConstructionError::InvalidSimplexArity { simplex_index: 0, actual: 2, expected: 3 },
///     ))
/// ));
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExplicitConstructionError {
    /// A simplex references a vertex index that is out of bounds.
    #[error(
        "Simplex {simplex_index}: vertex index {vertex_index} is out of bounds (vertex count: {bound})"
    )]
    IndexOutOfBounds {
        /// The index of the simplex in the input slice.
        simplex_index: usize,
        /// The out-of-bounds vertex index.
        vertex_index: usize,
        /// The number of vertices provided.
        bound: usize,
    },
    /// A simplex does not have exactly D+1 vertex indices.
    #[error(
        "Simplex {simplex_index}: has {actual} vertex indices, expected {expected} for a simplex"
    )]
    InvalidSimplexArity {
        /// The index of the simplex in the input slice.
        simplex_index: usize,
        /// The actual number of vertex indices.
        actual: usize,
        /// The expected number (D+1).
        expected: usize,
    },
    /// A simplex contains duplicate vertex indices.
    #[error("Simplex {simplex_index}: contains duplicate vertex indices")]
    DuplicateVertexInSimplex {
        /// The index of the simplex in the input slice.
        simplex_index: usize,
    },
    /// No simplices were provided.
    #[error("No simplices provided for explicit construction")]
    EmptySimplices,
    /// Simplex creation failed while assembling explicit connectivity.
    #[error(
        "Simplex {simplex_index}: simplex creation failed during explicit construction: {source}"
    )]
    SimplexCreation {
        /// The index of the simplex in the input slice.
        simplex_index: usize,
        /// Underlying simplex validation error.
        #[source]
        source: SimplexValidationError,
    },
    /// TDS assembly failed while inserting explicit vertices or simplices.
    #[error("TDS assembly failed during explicit construction: {source}")]
    TdsAssembly {
        /// Underlying TDS construction or mutation error.
        #[source]
        source: ExplicitTdsError,
    },
    /// Toroidal topology is incompatible with explicit simplex construction.
    #[error("Toroidal topology cannot be combined with explicit simplex construction")]
    IncompatibleTopology,
    /// Non-default [`ConstructionOptions`] were set on an explicit-simplex builder.
    ///
    /// [`ConstructionOptions`] (insertion order, deduplication, retry policy) apply
    /// only to the Delaunay point-insertion path and are not meaningful for
    /// explicit simplex construction.
    ///
    /// [`ConstructionOptions`]: crate::construction::ConstructionOptions
    #[error(
        "ConstructionOptions are not applicable to explicit simplex construction \
         and must be left at their default values"
    )]
    UnsupportedConstructionOptions,
    /// Neighbor assignment failed while assembling explicit connectivity.
    #[error("Neighbor assignment failed during explicit construction: {source}")]
    NeighborAssignment {
        /// Underlying TDS validation error.
        #[source]
        source: ExplicitTdsError,
    },
    /// Orientation normalization or positive-orientation promotion failed.
    #[error("Orientation normalization failed during explicit construction: {source}")]
    OrientationNormalization {
        /// Underlying insertion/orientation error.
        #[source]
        source: ExplicitInsertionError,
    },
    /// Level 1–2 TDS structural validation failed after assembly.
    #[error("Structural validation failed during explicit construction: {source}")]
    StructuralValidation {
        /// Underlying TDS validation error.
        #[source]
        source: ExplicitTdsError,
    },
    /// Level 3 topology validation failed after assembly.
    #[error("Topology validation failed during explicit construction: {source}")]
    TopologyValidation {
        /// Underlying cumulative validation error.
        #[source]
        source: ExplicitInvariantError,
    },
    /// Completion-time PL-manifold validation failed after assembly.
    #[error("PL-manifold completion validation failed during explicit construction: {source}")]
    CompletionValidation {
        /// Underlying cumulative validation error.
        #[source]
        source: ExplicitInvariantError,
    },
    /// Geometric nondegeneracy validation failed after assembly.
    #[error("Geometric nondegeneracy validation failed during explicit construction: {source}")]
    GeometricNondegeneracy {
        /// Underlying TDS/geometric validation error.
        #[source]
        source: ExplicitTdsError,
    },
    /// Level 4 Delaunay validation failed before returning the wrapper.
    #[error("Delaunay validation failed during explicit construction: {source}")]
    DelaunayValidation {
        /// Underlying Delaunay validation error.
        #[source]
        source: ExplicitDelaunayValidationError,
    },
    /// Explicit quotient connectivity is not supported for the requested topology.
    #[error(
        "Explicit non-Euclidean connectivity is not supported for {topology:?}; Level 4 quotient validation is required"
    )]
    UnsupportedExplicitTopology {
        /// Requested global topology metadata.
        topology: TopologyKind,
    },
}

// =============================================================================
// BUILDER STRUCT
// =============================================================================

/// Fluent builder for [`DelaunayTriangulation`] with optional toroidal topology.
///
/// # Type Parameters
///
/// - `'v` — Lifetime of the borrowed vertex slice.
/// - `T` — Coordinate scalar type (inferred from the vertex slice).
/// - `U` — Vertex data type (inferred from the vertex slice).
/// - `D` — Spatial dimension (inferred from the vertex slice).
///
/// The simplex data type `V` and kernel `K` are deferred to the
/// [`build`](Self::build) / [`build_with_kernel`](Self::build_with_kernel)
/// call, keeping the builder type signature concise.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::construction::{
///     ConstructionOptions, DelaunayTriangulationBuilder, TopologyGuarantee, vertex,
/// };
///
/// # fn main() -> Result<(), delaunay::prelude::construction::DelaunayTriangulationConstructionError> {
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
///
/// let dt = DelaunayTriangulationBuilder::new(&vertices)
///     .topology_guarantee(TopologyGuarantee::Pseudomanifold)
///     .construction_options(ConstructionOptions::default())
///     .build::<()>()?;
///
/// assert_eq!(dt.number_of_vertices(), 4);
/// # Ok(())
/// # }
/// ```
pub struct DelaunayTriangulationBuilder<'v, T, U, const D: usize> {
    vertices: &'v [Vertex<T, U, D>],
    /// Optional toroidal (periodic) topology for the construction.
    ///
    /// When set, all input vertices are canonicalized into the fundamental domain
    /// `[0, L_i)` before the triangulation is built.
    topology: Option<ToroidalSpace<D>>,
    topology_guarantee: TopologyGuarantee,
    construction_options: ConstructionOptions,
    /// When `true` (set by [`.toroidal_periodic()`](Self::toroidal_periodic)), the
    /// Phase 2 image-point algorithm is used instead of the Phase 1 canonicalization path.
    use_image_point_method: bool,
    /// Optional explicit simplex specifications for direct combinatorial construction.
    ///
    /// When set, the builder constructs a triangulation from the given vertices and
    /// simplices directly, bypassing point-insertion-based Delaunay construction.
    /// Each inner slice must contain exactly D+1 vertex indices.
    explicit_simplices: Option<&'v [Vec<usize>]>,
    /// Runtime global topology metadata.
    ///
    /// When set to a non-Euclidean topology (e.g. `Toroidal`), Euler characteristic
    /// validation uses the appropriate expectation (e.g. χ = 0 for a torus).
    /// This is **metadata only** and does not trigger any construction-time
    /// coordinate transformation.
    global_topology: GlobalTopology<D>,
}

// =============================================================================
// SPECIALIZED IMPL — f64 coordinates, any vertex data U
//
// Pins T=f64 so callers using the default AdaptiveKernel never need explicit
// type annotations.  U is inferred from the vertex slice.
//
//   let vertices = vec![vertex!([0.0, 0.0]), ...];
//   let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>();
//
//   let typed: [Vertex<f64, i32, 2>; 3] = [vertex!([0.0, 0.0], 1), ...];
//   let dt = DelaunayTriangulationBuilder::new(&typed).build::<()>();
//
// =============================================================================

impl<'v, U, const D: usize> DelaunayTriangulationBuilder<'v, f64, U, D> {
    /// Creates a builder for `f64` vertices with any user data type `U`.
    ///
    /// `U` is inferred from the vertex slice — no explicit type annotations needed
    /// for either `U = ()` (the common case) or typed vertex data.
    ///
    /// For non-`f64` scalar types, use
    /// [`from_vertices`](Self::from_vertices) in the generic impl.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, Vertex, vertex,
    /// };
    ///
    /// # fn main() -> Result<(), DelaunayTriangulationConstructionError> {
    /// // No vertex data (U = () inferred)
    /// let vertices = vec![vertex!([0.0, 0.0]), vertex!([1.0, 0.0]), vertex!([0.0, 1.0])];
    /// let _dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// // Typed vertex data (U = i32 inferred)
    /// let typed: [Vertex<f64, i32, 2>; 3] = [
    ///     vertex!([0.0, 0.0], 1i32),
    ///     vertex!([1.0, 0.0], 2),
    ///     vertex!([0.0, 1.0], 3),
    /// ];
    /// let _dt = DelaunayTriangulationBuilder::new(&typed).build::<()>()?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn new(vertices: &'v [Vertex<f64, U, D>]) -> Self {
        Self {
            vertices,
            topology: None,
            topology_guarantee: TopologyGuarantee::DEFAULT,
            construction_options: ConstructionOptions::default(),
            use_image_point_method: false,
            explicit_simplices: None,
            global_topology: GlobalTopology::DEFAULT,
        }
    }

    /// `f64` convenience wrapper for
    /// [`from_vertices_and_simplices_generic`](Self::from_vertices_and_simplices_generic).
    ///
    /// Pins `T = f64` so that callers using the default `AdaptiveKernel` never
    /// need explicit type annotations. For non-`f64` scalars, use the generic
    /// [`from_vertices_and_simplices_generic`](Self::from_vertices_and_simplices_generic)
    /// on the `T: CoordinateScalar` impl block.
    ///
    /// This is not an unchecked topology wrapper. The deferred
    /// [`build`](Self::build) or [`build_with_kernel`](Self::build_with_kernel)
    /// call accepts only Euclidean explicit connectivity and validates the
    /// assembled triangulation at Levels 1–4, so the supplied connectivity must
    /// already satisfy the Delaunay empty-circumsphere property. Non-Euclidean
    /// explicit connectivity is rejected because it requires Level 4 handling
    /// that is not available for quotient meshes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
    ///     ExplicitConstructionError, vertex,
    /// };
    ///
    /// # fn main() -> Result<(), DelaunayTriangulationConstructionError> {
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([1.0, 1.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let simplices = vec![vec![0, 1, 2], vec![0, 2, 3]];
    ///
    /// let dt = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
    ///     .build::<()>()?;
    ///
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// assert_eq!(dt.number_of_simplices(), 2);
    ///
    /// let bad_simplices = vec![vec![0, 1]]; // Wrong arity for a 2D simplex.
    /// assert!(matches!(
    ///     DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &bad_simplices).build::<()>(),
    ///     Err(DelaunayTriangulationConstructionError::ExplicitConstruction(
    ///         ExplicitConstructionError::InvalidSimplexArity { .. },
    ///     ))
    /// ));
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn from_vertices_and_simplices(
        vertices: &'v [Vertex<f64, U, D>],
        simplices: &'v [Vec<usize>],
    ) -> Self {
        Self::from_vertices_and_simplices_generic(vertices, simplices)
    }
}

// =============================================================================
// GENERIC IMPL — any scalar T, any vertex data U
// =============================================================================

impl<'v, T, U, const D: usize> DelaunayTriangulationBuilder<'v, T, U, D> {
    /// Creates a builder from explicit vertex and simplex specifications.
    ///
    /// This constructs a triangulation from the given connectivity without
    /// Delaunay point insertion. Works with any scalar type `T`.
    ///
    /// The explicit connectivity is still validated when
    /// [`build`](Self::build) or [`build_with_kernel`](Self::build_with_kernel)
    /// is called. Euclidean explicit meshes are checked at Levels 1–4, including
    /// the Delaunay empty-circumsphere property. Non-Euclidean explicit
    /// connectivity is rejected because there is no successful Levels 1–3-only
    /// path for the public `DelaunayTriangulation` wrapper; quotient meshes need
    /// Level 4 handling before they can be accepted.
    ///
    /// For `f64` coordinates, prefer the convenience wrapper on the
    /// `f64`-specialised impl which avoids explicit type annotations.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::geometry::{Coordinate, Point};
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, Vertex, VertexBuilder,
    /// };
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Vertex(#[from] delaunay::prelude::construction::VertexBuilderError),
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::prelude::construction::DelaunayTriangulationConstructionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices: Vec<Vertex<f32, (), 2>> = vec![
    ///     VertexBuilder::default().point(Point::new([0.0_f32, 0.0])).build()?,
    ///     VertexBuilder::default().point(Point::new([1.0_f32, 0.0])).build()?,
    ///     VertexBuilder::default().point(Point::new([0.0_f32, 1.0])).build()?,
    /// ];
    /// let simplices = vec![vec![0, 1, 2]];
    ///
    /// let dt = DelaunayTriangulationBuilder::from_vertices_and_simplices_generic(&vertices, &simplices)
    ///     .build::<()>()?;
    ///
    /// assert_eq!(dt.number_of_vertices(), 3);
    /// assert_eq!(dt.number_of_simplices(), 1);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn from_vertices_and_simplices_generic(
        vertices: &'v [Vertex<T, U, D>],
        simplices: &'v [Vec<usize>],
    ) -> Self {
        Self {
            vertices,
            topology: None,
            topology_guarantee: TopologyGuarantee::DEFAULT,
            construction_options: ConstructionOptions::default(),
            use_image_point_method: false,
            explicit_simplices: Some(simplices),
            global_topology: GlobalTopology::DEFAULT,
        }
    }

    /// Creates a builder from a vertex slice of any scalar type `T` and user data type `U`.
    ///
    /// For `f64` coordinates, prefer [`new`](DelaunayTriangulationBuilder::new) which
    /// infers all type parameters without explicit annotations. Use `from_vertices`
    /// when `T ≠ f64` (e.g. `f32`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::geometry::{Coordinate, Point};
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, Vertex, VertexBuilder,
    /// };
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Vertex(#[from] delaunay::prelude::construction::VertexBuilderError),
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::prelude::construction::DelaunayTriangulationConstructionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// // f32 vertices — new() is f64-only, so from_vertices is required here.
    /// let vertices: Vec<Vertex<f32, (), 2>> = vec![
    ///     VertexBuilder::default().point(Point::new([0.0_f32, 0.0])).build()?,
    ///     VertexBuilder::default().point(Point::new([1.0_f32, 0.0])).build()?,
    ///     VertexBuilder::default().point(Point::new([0.0_f32, 1.0])).build()?,
    /// ];
    ///
    /// let dt = DelaunayTriangulationBuilder::from_vertices(&vertices)
    ///     .build::<()>()?;
    ///
    /// assert_eq!(dt.number_of_vertices(), 3);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn from_vertices(vertices: &'v [Vertex<T, U, D>]) -> Self {
        Self {
            vertices,
            topology: None,
            topology_guarantee: TopologyGuarantee::DEFAULT,
            construction_options: ConstructionOptions::default(),
            use_image_point_method: false,
            explicit_simplices: None,
            global_topology: GlobalTopology::DEFAULT,
        }
    }

    /// Enables Phase 1 toroidal topology: input vertices are canonicalized into
    /// `[0, L_i)` per dimension before the triangulation is built.
    ///
    /// The resulting triangulation is a valid Euclidean Delaunay triangulation of the
    /// wrapped point set; boundary facets are **not** rewired. Use
    /// [`.toroidal_periodic()`](Self::toroidal_periodic) for a true periodic (χ = 0)
    /// triangulation.
    ///
    /// # Arguments
    ///
    /// * `domain` — Period length for each dimension.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, vertex,
    /// };
    ///
    /// # fn main() -> Result<(), delaunay::prelude::construction::DelaunayTriangulationConstructionError> {
    /// let vertices = vec![
    ///     vertex!([0.2, 0.3]),
    ///     vertex!([0.8, 0.1]),
    ///     vertex!([0.5, 0.7]),
    ///     vertex!([0.1, 0.9]),
    /// ];
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .toroidal([1.0, 1.0])
    ///     .build::<()>()?;
    ///
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn toroidal(mut self, domain: [f64; D]) -> Self {
        self.topology = Some(ToroidalSpace::new(domain));
        self
    }

    /// Enables Phase 2 (full periodic) toroidal topology via the image-point method.
    ///
    /// Vertices are first canonicalized into `[0, L_i)`, then 3^D copies of the point set
    /// are built by shifting each point by every combination of `{-L_i, 0, +L_i}`. A full
    /// Euclidean Delaunay triangulation is built on the expanded set, the fundamental domain
    /// is extracted, and boundary facets are rewired with periodic neighbor pointers.
    ///
    /// The result is a valid toroidal triangulation with Euler characteristic χ = 0 (2D),
    /// χ = 0 (3D), etc.
    ///
    /// **Requires at least `2*D + 1` input points** after canonicalization.
    ///
    /// **Use [`AdaptiveKernel`] or [`RobustKernel`](crate::geometry::kernel::RobustKernel)** for
    /// reliable results. The default [`build()`](Self::build) uses `AdaptiveKernel`.
    /// Numerical near-degeneracies in the expanded set can cause construction failures
    /// with `FastKernel`.
    ///
    /// # Arguments
    ///
    /// * `domain` — Period length `[L_0, …, L_{D-1}]` for each dimension.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use delaunay::prelude::geometry::RobustKernel;
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, vertex,
    /// };
    ///
    /// # fn main() -> Result<(), delaunay::prelude::construction::DelaunayTriangulationConstructionError> {
    /// let vertices = vec![
    ///     vertex!([0.1, 0.2]),
    ///     vertex!([0.4, 0.7]),
    ///     vertex!([0.7, 0.3]),
    ///     vertex!([0.2, 0.9]),
    ///     vertex!([0.8, 0.6]),
    ///     vertex!([0.5, 0.1]),
    ///     vertex!([0.3, 0.5]),
    /// ];
    ///
    /// let kernel = RobustKernel::new();
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .toroidal_periodic([1.0, 1.0])
    ///     .build_with_kernel::<_, ()>(&kernel)?;
    ///
    /// assert_eq!(dt.number_of_vertices(), 7);
    /// assert!(dt.tds().is_valid().is_ok());
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn toroidal_periodic(mut self, domain: [f64; D]) -> Self {
        self.topology = Some(ToroidalSpace::new(domain));
        self.use_image_point_method = true;
        self
    }

    /// Sets the [`TopologyGuarantee`]
    ///
    /// Defaults to [`TopologyGuarantee::DEFAULT`] (`PLManifold`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, TopologyGuarantee, vertex,
    /// };
    ///
    /// # fn main() -> Result<(), delaunay::prelude::construction::DelaunayTriangulationConstructionError> {
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .topology_guarantee(TopologyGuarantee::Pseudomanifold)
    ///     .build::<()>()?;
    ///
    /// assert_eq!(dt.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn topology_guarantee(mut self, topology_guarantee: TopologyGuarantee) -> Self {
        self.topology_guarantee = topology_guarantee;
        self
    }

    /// Sets the [`GlobalTopology`] metadata for the triangulation.
    ///
    /// This declares the intended global topology so that Euler characteristic
    /// validation uses the correct expectation. For example, setting
    /// [`GlobalTopology::Toroidal`] tells the validator to expect χ = 0 for a closed
    /// mesh instead of χ = 2 (the sphere default).
    ///
    /// This is **metadata only** and does not trigger any coordinate
    /// canonicalization or image-point construction. Explicit non-Euclidean
    /// connectivity is rejected until Level 4 validation supports quotient
    /// topology. For construction-time toroidal processing, use
    /// [`.toroidal()`](Self::toroidal) or
    /// [`.toroidal_periodic()`](Self::toroidal_periodic) instead.
    ///
    /// Defaults to [`GlobalTopology::Euclidean`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, GlobalTopology, ToroidalConstructionMode, vertex,
    /// };
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let simplices = vec![vec![0, 1, 2]];
    ///
    /// let result = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
    ///     .global_topology(GlobalTopology::Toroidal {
    ///         domain: [1.0, 1.0],
    ///         mode: ToroidalConstructionMode::Explicit,
    ///     })
    ///     .build::<()>();
    ///
    /// assert!(result.is_err());
    /// ```
    #[must_use]
    pub const fn global_topology(mut self, global_topology: GlobalTopology<D>) -> Self {
        self.global_topology = global_topology;
        self
    }

    /// Sets the [`ConstructionOptions`] (insertion order, deduplication, retry policy).
    ///
    /// Defaults to [`ConstructionOptions::default`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     ConstructionOptions, DelaunayTriangulationBuilder, InsertionOrderStrategy, vertex,
    /// };
    ///
    /// # fn main() -> Result<(), delaunay::prelude::construction::DelaunayTriangulationConstructionError> {
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    ///
    /// let opts = ConstructionOptions::default()
    ///     .with_insertion_order(InsertionOrderStrategy::Input);
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .construction_options(opts)
    ///     .build::<()>()?;
    ///
    /// assert_eq!(dt.number_of_vertices(), 3);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn construction_options(mut self, construction_options: ConstructionOptions) -> Self {
        self.construction_options = construction_options;
        self
    }
}

impl<T, U, const D: usize> DelaunayTriangulationBuilder<'_, T, U, D>
where
    T: CoordinateScalar,
    U: DataType,
{
    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    /// Validates the topology model configuration before using it in construction.
    ///
    /// This helper is called before any topology-based canonicalization or lifting operations
    /// to ensure that the model's runtime parameters (e.g., toroidal domain periods) are valid.
    ///
    /// # Parameters
    ///
    /// * `model` - The topology behavior model to validate.
    ///
    /// # Returns
    ///
    /// - `Ok(())` if the model configuration is valid.
    /// - `Err(DelaunayTriangulationConstructionError)` if validation fails.
    ///
    /// # Errors
    ///
    /// Maps [`GlobalTopologyModelError`] to [`DelaunayTriangulationConstructionError`]:
    /// - [`GlobalTopologyModelError::InvalidToroidalPeriod`] → detailed message with axis and period.
    /// - Other errors → generic configuration error message.
    ///
    /// # Usage
    ///
    /// Called internally by [`build_with_kernel`](Self::build_with_kernel) before
    /// canonicalization in both Phase 1 (canonicalized) and Phase 2 (image-point) paths.
    fn validate_topology_model<M>(model: &M) -> Result<(), DelaunayTriangulationConstructionError>
    where
        M: GlobalTopologyModel<D>,
    {
        model.validate_configuration().map_err(|error| match error {
            GlobalTopologyModelError::InvalidToroidalPeriod { axis, period } => {
                TriangulationConstructionError::GeometricDegeneracy {
                    message: format!(
                        "Invalid toroidal domain at axis {axis}: period {period:?}; expected finite value > 0",
                    ),
                }
                .into()
            }
            other => TriangulationConstructionError::GeometricDegeneracy {
                message: format!("Invalid topology model configuration: {other}"),
            }
            .into(),
        })
    }

    /// Derives a periodic facet key from a lifted simplex and maps derivation
    /// failures into construction-level geometric-degeneracy errors.
    ///
    /// This helper centralizes error conversion for
    /// [`periodic_facet_key_from_lifted_vertices`] so all call sites produce
    /// consistent diagnostics.
    ///
    /// # Parameters
    ///
    /// * `lifted_ordered` - Lifted simplex vertices as `(VertexKey, lattice_offset)`
    ///   pairs (expected arity: `D + 1`).
    /// * `facet_idx` - Index of the facet opposite a vertex in the simplex.
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayTriangulationConstructionError`] wrapping
    /// [`TriangulationConstructionError::GeometricDegeneracy`] when periodic
    /// facet key derivation fails (e.g., invalid arity/index or offset encoding).
    fn derive_periodic_facet_key(
        lifted_ordered: &[(VertexKey, [i8; D])],
        facet_idx: usize,
    ) -> Result<PeriodicFacetKey, DelaunayTriangulationConstructionError> {
        periodic_facet_key_from_lifted_vertices::<D>(lifted_ordered, facet_idx).map_err(|error| {
            TriangulationConstructionError::GeometricDegeneracy {
                message: format!(
                    "Failed to derive periodic candidate facet signature for index {facet_idx}: {error}",
                ),
            }
            .into()
        })
    }

    /// Canonicalizes vertices using a topology behavior model.
    ///
    /// For each input vertex, calls [`GlobalTopologyModel::canonicalize_point_in_place`] to wrap
    /// coordinates into the model's fundamental domain (e.g., [0, L) for toroidal topologies).
    /// Preserves vertex UUIDs and data while transforming coordinates.
    ///
    /// # Parameters
    ///
    /// * `vertices` - Slice of input vertices with potentially out-of-domain coordinates.
    /// * `model` - The topology behavior model that defines canonicalization logic.
    ///
    /// # Returns
    ///
    /// A new vector of vertices with canonicalized coordinates. Each output vertex has:
    /// - The same UUID as the corresponding input vertex (for tracking through construction).
    /// - The same associated data as the input vertex.
    /// - Coordinates transformed according to the model's canonicalization rules.
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayTriangulationConstructionError`] if canonicalization fails for any vertex:
    /// - Non-finite coordinates (NaN, infinity).
    /// - Invalid toroidal periods.
    /// - Scalar conversion failures.
    ///
    /// Error messages include the failing vertex index and original coordinates for debugging.
    ///
    /// # Usage
    ///
    /// Called internally by [`build_with_kernel`](Self::build_with_kernel) before delegating
    /// to the underlying triangulation construction.
    fn canonicalize_vertices<M>(
        vertices: &[Vertex<T, U, D>],
        model: &M,
    ) -> Result<Vec<Vertex<T, U, D>>, DelaunayTriangulationConstructionError>
    where
        M: GlobalTopologyModel<D>,
    {
        let mut out = Vec::with_capacity(vertices.len());

        for (idx, v) in vertices.iter().enumerate() {
            let mut canonicalized_coords = *v.point().coords();
            model
                .canonicalize_point_in_place(&mut canonicalized_coords)
                .map_err(|error| TriangulationConstructionError::GeometricDegeneracy {
                    message: format!(
                        "Failed to canonicalize vertex {idx}: original coords {:?}; reason: {error}",
                        v.point().coords(),
                    ),
                })?;

            let new_point = Point::new(canonicalized_coords);
            let new_vertex = Vertex::new_with_uuid(new_point, v.uuid(), v.data);

            out.push(new_vertex);
        }

        Ok(out)
    }

    // -------------------------------------------------------------------------
    // Build methods
    // -------------------------------------------------------------------------

    /// Builds the triangulation using [`AdaptiveKernel<T>`](crate::geometry::kernel::AdaptiveKernel).
    ///
    /// This is the most common build path. Simplex data type `V` is inferred or
    /// specified at the call site; it is independent of the vertex data type `U`.
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayTriangulationConstructionError`] if:
    /// - Toroidal canonicalization fails (non-finite coordinate in input).
    /// - The underlying triangulation construction fails (insufficient vertices,
    ///   geometric degeneracy, etc.).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, vertex,
    /// };
    ///
    /// # fn main() -> Result<(), delaunay::prelude::construction::DelaunayTriangulationConstructionError> {
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .build::<()>()?;
    ///
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// assert!(dt.validate().is_ok());
    /// # Ok(())
    /// # }
    /// ```
    pub fn build<V>(
        self,
    ) -> Result<
        DelaunayTriangulation<AdaptiveKernel<T>, U, V, D>,
        DelaunayTriangulationConstructionError,
    >
    where
        V: DataType,
    {
        self.build_with_kernel(&AdaptiveKernel::new())
    }

    /// Builds the triangulation using a caller-supplied kernel.
    ///
    /// [`build()`](Self::build) already defaults to [`AdaptiveKernel`], so this method is
    /// only needed when you want a different kernel (e.g. [`FastKernel`](crate::geometry::kernel::FastKernel)
    /// for workloads that prioritize speed over exact predicate correctness, or a custom
    /// implementation).
    ///
    /// **Note:** `FastKernel` is accepted for construction, but the explicit repair methods
    /// ([`repair_delaunay_with_flips`](DelaunayTriangulation::repair_delaunay_with_flips),
    /// [`repair_delaunay_with_flips_advanced`](DelaunayTriangulation::repair_delaunay_with_flips_advanced))
    /// require [`ExactPredicates`](crate::geometry::kernel::ExactPredicates) and are not available
    /// for `FastKernel`.
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayTriangulationConstructionError`] if canonicalization or
    /// construction fails (see [`build`](Self::build) for details).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::geometry::RobustKernel;
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, vertex,
    /// };
    ///
    /// # fn main() -> Result<(), delaunay::prelude::construction::DelaunayTriangulationConstructionError> {
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let kernel = RobustKernel::new();
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .build_with_kernel::<_, ()>(&kernel)?;
    ///
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// # Ok(())
    /// # }
    /// ```
    pub fn build_with_kernel<K, V>(
        self,
        kernel: &K,
    ) -> Result<DelaunayTriangulation<K, U, V, D>, DelaunayTriangulationConstructionError>
    where
        K: Kernel<D, Scalar = T>,
        V: DataType,
    {
        // Explicit-simplices path: bypass Delaunay insertion entirely.
        if let Some(simplices) = self.explicit_simplices {
            if self.topology.is_some() {
                return Err(ExplicitConstructionError::IncompatibleTopology.into());
            }
            if self.construction_options != ConstructionOptions::default() {
                return Err(ExplicitConstructionError::UnsupportedConstructionOptions.into());
            }
            return Self::build_explicit(
                kernel,
                self.vertices,
                simplices,
                self.topology_guarantee,
                self.global_topology,
            );
        }

        match (self.topology, self.use_image_point_method) {
            (None, _) => {
                // Euclidean path: delegate directly.
                let mut dt = DelaunayTriangulation::with_topology_guarantee_and_options(
                    kernel,
                    self.vertices,
                    self.topology_guarantee,
                    self.construction_options,
                )?;
                dt.set_global_topology(self.global_topology);
                Ok(dt)
            }
            (Some(space), false) => {
                let topology = GlobalTopology::Toroidal {
                    domain: space.domain,
                    mode: ToroidalConstructionMode::Canonicalized,
                };
                let topology_model = topology.model();
                Self::validate_topology_model(&topology_model)?;
                // Toroidal Phase 1: canonicalize then delegate.
                let canonical = Self::canonicalize_vertices(self.vertices, &topology_model)?;
                let mut dt = DelaunayTriangulation::with_topology_guarantee_and_options(
                    kernel,
                    &canonical,
                    self.topology_guarantee,
                    self.construction_options,
                )?;
                dt.set_global_topology(topology);
                Ok(dt)
            }
            (Some(space), true) => {
                let topology = GlobalTopology::Toroidal {
                    domain: space.domain,
                    mode: ToroidalConstructionMode::PeriodicImagePoint,
                };
                let topology_model = topology.model();
                Self::validate_topology_model(&topology_model)?;
                if !topology_model.supports_periodic_facet_signatures() {
                    return Err(TriangulationConstructionError::GeometricDegeneracy {
                        message: format!(
                            "Topology {:?} does not support periodic facet signatures required for periodic image-point construction",
                            topology_model.kind(),
                        ),
                    }
                    .into());
                }
                // Toroidal Phase 2: canonicalize then apply 3^D image-point method.
                let canonical = Self::canonicalize_vertices(self.vertices, &topology_model)?;
                let mut dt = Self::build_periodic(
                    kernel,
                    &canonical,
                    &topology_model,
                    self.topology_guarantee,
                    self.construction_options,
                )?;
                dt.set_global_topology(topology);
                dt.tri
                    .normalize_and_promote_positive_orientation()
                    .map_err(|e| TriangulationConstructionError::GeometricDegeneracy {
                        message: format!(
                            "Failed to canonicalize periodic orientation after build: {e}",
                        ),
                    })?;
                dt.as_triangulation()
                    .validate_geometric_simplex_orientation()
                    .map_err(|e| TriangulationConstructionError::GeometricDegeneracy {
                        message: format!(
                            "Periodic geometric orientation validation failed after build: {e}",
                        ),
                    })?;
                Ok(dt)
            }
        }
    }

    /// Builds a triangulation from explicit vertex and simplex specifications.
    ///
    /// This is a purely combinatorial construction that assembles a valid TDS from
    /// the given connectivity without Delaunay point insertion. Euclidean explicit
    /// meshes are validated at Levels 1–4 (elements, structure, topology, and the
    /// Delaunay property). Non-Euclidean explicit connectivity is rejected because
    /// it requires Level 4 quotient-topology validation before the public
    /// `DelaunayTriangulation` wrapper can accept it.
    ///
    /// # Algorithm
    ///
    /// 1. Validate input: each simplex has D+1 in-bounds, unique vertex indices.
    /// 2. Build a `Tds`: insert all vertices, then insert simplices from the specifications.
    /// 3. Compute adjacency via `assign_neighbors()`.
    /// 4. Assign incident simplices via `assign_incident_simplices()`.
    /// 5. Wrap in `DelaunayTriangulation` via `from_tds_with_topology_guarantee`.
    /// 6. Normalize coherent orientation and promote to positive canonical sign
    ///    via `normalize_and_promote_positive_orientation()`.
    /// 7. Reject non-Euclidean explicit connectivity until Level 4 quotient
    ///    validation exists.
    /// 8. Validate Levels 1–2 (TDS structural: `tds.validate()`).
    /// 9. Validate Level 3 topology (excluding geometric orientation).
    /// 10. Validate PL-manifold completion (vertex links, if required).
    /// 11. Validate geometric nondegeneracy (reject zero-volume simplices).
    /// 12. Validate the Euclidean Level 4 Delaunay property.
    fn build_explicit<K, V>(
        kernel: &K,
        vertices: &[Vertex<T, U, D>],
        simplices: &[Vec<usize>],
        topology_guarantee: TopologyGuarantee,
        global_topology: GlobalTopology<D>,
    ) -> Result<DelaunayTriangulation<K, U, V, D>, DelaunayTriangulationConstructionError>
    where
        K: Kernel<D, Scalar = T>,
        V: DataType,
    {
        Self::validate_explicit_simplex_specs(vertices.len(), simplices)?;
        Self::reject_explicit_non_euclidean_topology(global_topology)?;

        let vertex_count = vertices.len();

        // --- Build TDS ---
        let mut tds: Tds<T, U, V, D> = Tds::empty();

        // Insert all vertices and build index → VertexKey map.
        let mut index_to_key = Vec::with_capacity(vertex_count);
        for v in vertices {
            let vk = tds.insert_vertex_with_mapping(*v).map_err(|source| {
                ExplicitConstructionError::TdsAssembly {
                    source: source.into(),
                }
            })?;
            index_to_key.push(vk);
        }

        // Insert simplices.
        for (simplex_idx, simplex_spec) in simplices.iter().enumerate() {
            let vertex_keys: Vec<VertexKey> =
                simplex_spec.iter().map(|&vi| index_to_key[vi]).collect();
            let simplex = Simplex::new(vertex_keys, None).map_err(|e| {
                ExplicitConstructionError::SimplexCreation {
                    simplex_index: simplex_idx,
                    source: e,
                }
            })?;
            tds.insert_simplex_with_mapping(simplex).map_err(|source| {
                ExplicitConstructionError::TdsAssembly {
                    source: source.into(),
                }
            })?;
        }

        // Mark as constructed so validation doesn't reject incomplete state.
        tds.construction_state = TriangulationConstructionState::Constructed;

        // --- Compute adjacency ---
        tds.assign_neighbors()
            .map_err(|source| ExplicitConstructionError::NeighborAssignment {
                source: source.into(),
            })?;

        // --- Assign incident simplices ---
        tds.assign_incident_simplices().map_err(|source| {
            ExplicitConstructionError::TdsAssembly {
                source: source.into(),
            }
        })?;

        // --- Wrap in DelaunayTriangulation ---
        //
        // Construct the DT first so the Triangulation-layer helpers
        // (orientation promotion, topology checks) operate on the assembled
        // complex.
        let mut dt = DelaunayTriangulation::from_tds_with_topology_guarantee(
            tds,
            kernel.clone(),
            topology_guarantee,
        );

        // Set global topology metadata before validation so that
        // validate_topology_core() uses the correct Euler characteristic
        // expectation (e.g. χ = 0 for toroidal instead of χ = 2 for sphere).
        dt.set_global_topology(global_topology);

        // --- Normalize orientation and promote to positive ---
        //
        // normalize_and_promote_positive_orientation() combines:
        //   1. BFS coherent-orientation normalization (adjacent simplices agree)
        //   2. Global sign canonicalization (flip all if representative is negative)
        //   3. Bounded per-simplex promotion passes for FP-precision edge cases
        // This ensures the returned DT has positive geometric orientation,
        // matching the invariant expected by validate_geometric_simplex_orientation.
        dt.tri
            .normalize_and_promote_positive_orientation()
            .map_err(
                |source| ExplicitConstructionError::OrientationNormalization {
                    source: source.into(),
                },
            )?;

        // Level 1–2: TDS structural validation (mappings, neighbors, facet
        // sharing, coherent orientation, etc.).
        if let Err(e) = dt.tri.tds.validate() {
            return Err(
                ExplicitConstructionError::StructuralValidation { source: e.into() }.into(),
            );
        }

        // Level 3 (topology, excluding geometric orientation): connectedness,
        // manifold facets, isolated vertices, Euler characteristic, and
        // PL-manifold vertex/ridge links when the topology guarantee requires
        // them.  We call `is_valid_topology_only()` which covers all these;
        // the only check we intentionally omit is
        // `validate_geometric_simplex_orientation`.
        if let Err(e) = dt.tri.is_valid_topology_only() {
            return Err(ExplicitConstructionError::TopologyValidation { source: e.into() }.into());
        }

        // Completion-time PL-manifold check (vertex links) if required.
        if let Err(e) = dt.tri.validate_at_completion() {
            return Err(
                ExplicitConstructionError::CompletionValidation { source: e.into() }.into(),
            );
        }

        // --- Geometric nondegeneracy ---
        //
        // Reject degenerate simplices (zero-volume simplices from collinear /
        // coplanar vertices).  Unlike the Delaunay construction paths, which
        // may tolerate near-degenerate simplices from flip-based repair,
        // explicit construction should not silently accept geometrically
        // collapsed simplices supplied by the user.
        if let Err(e) = dt.tri.validate_geometric_nondegeneracy() {
            return Err(
                ExplicitConstructionError::GeometricNondegeneracy { source: e.into() }.into(),
            );
        }

        Self::enforce_explicit_delaunay_property(&dt)?;

        Ok(dt)
    }

    /// Validates explicit simplex specifications before constructing a TDS.
    fn validate_explicit_simplex_specs(
        vertex_count: usize,
        simplices: &[Vec<usize>],
    ) -> Result<(), ExplicitConstructionError> {
        if simplices.is_empty() {
            return Err(ExplicitConstructionError::EmptySimplices);
        }

        for (simplex_idx, simplex_spec) in simplices.iter().enumerate() {
            if simplex_spec.len() != D + 1 {
                return Err(ExplicitConstructionError::InvalidSimplexArity {
                    simplex_index: simplex_idx,
                    actual: simplex_spec.len(),
                    expected: D + 1,
                });
            }
            for (i, &vi) in simplex_spec.iter().enumerate() {
                if vi >= vertex_count {
                    return Err(ExplicitConstructionError::IndexOutOfBounds {
                        simplex_index: simplex_idx,
                        vertex_index: vi,
                        bound: vertex_count,
                    });
                }
                for &vj in &simplex_spec[i + 1..] {
                    if vi == vj {
                        return Err(ExplicitConstructionError::DuplicateVertexInSimplex {
                            simplex_index: simplex_idx,
                        });
                    }
                }
            }
        }

        Ok(())
    }

    /// Enforces Level 4 validation before returning the Delaunay wrapper.
    ///
    /// The public return type is `DelaunayTriangulation`, so Euclidean explicit
    /// connectivity must prove the empty-circumsphere property before it crosses
    /// this API boundary. Explicit non-Euclidean topology is rejected earlier in
    /// `build_explicit` until a Level 4 validator exists for quotient connectivity.
    fn enforce_explicit_delaunay_property<K, V>(
        dt: &DelaunayTriangulation<K, U, V, D>,
    ) -> Result<(), DelaunayTriangulationConstructionError>
    where
        K: Kernel<D, Scalar = T>,
        V: DataType,
    {
        dt.is_valid().map_err(|source| {
            ExplicitConstructionError::DelaunayValidation {
                source: source.into(),
            }
            .into()
        })
    }

    /// Rejects explicit quotient connectivity until Level 4 validation supports it.
    fn reject_explicit_non_euclidean_topology(
        global_topology: GlobalTopology<D>,
    ) -> Result<(), DelaunayTriangulationConstructionError> {
        if global_topology.is_euclidean() {
            return Ok(());
        }

        Err(ExplicitConstructionError::UnsupportedExplicitTopology {
            topology: global_topology.kind(),
        }
        .into())
    }

    /// Builds a true periodic (toroidal) Delaunay triangulation using the 3^D image-point method.
    ///
    /// **Algorithm** (see module-level doc for Phase 2 details):
    /// 1. Validate: at least `2*D + 1` canonical vertices required.
    /// 2. Build 3^D-1 image copies of each vertex, shifted by `{-L_i, 0, +L_i}` per axis.
    ///    Every copy of canonical vertex `v_i` (including the zero-offset canonical copy)
    ///    receives the **same** tiny deterministic per-vertex perturbation `δ_i`.
    /// 3. Build a full Euclidean DT on the expanded set (n * 3^D points).
    /// 4. Normalize lifted simplices to canonical quotient signatures.
    /// 5. Search for a closed candidate subset whose periodic facet incidences are valid.
    /// 6. Rebuild quotient representatives from that selection with periodic offsets.
    /// 7. Rebuild neighbor and incident-simplex associations and return the result.
    ///
    /// The output is a `Tds` whose `is_valid()` passes at Level 2 (structural validity).
    ///
    /// # References
    ///
    /// - `REFERENCES.md`, "Periodic and Toroidal Triangulations", first entry
    ///   (Caroli and Teillaud, "Computing 3D Periodic Triangulations").
    /// - CGAL, *2D Periodic Triangulations*:
    ///   <https://doc.cgal.org/latest/Periodic_2_triangulation_2/index.html>
    /// - CGAL, *3D Periodic Triangulations*:
    ///   <https://doc.cgal.org/latest/Periodic_3_triangulation_3/index.html>
    #[expect(
        clippy::too_many_lines,
        reason = "Image-point periodic DT algorithm is inherently multi-step; splitting would harm readability"
    )]
    fn build_periodic<K, V, M>(
        kernel: &K,
        canonical_vertices: &[Vertex<T, U, D>],
        topology_model: &M,
        topology_guarantee: TopologyGuarantee,
        construction_options: ConstructionOptions,
    ) -> Result<DelaunayTriangulation<K, U, V, D>, DelaunayTriangulationConstructionError>
    where
        K: Kernel<D, Scalar = T>,
        V: DataType,
        M: GlobalTopologyModel<D>,
    {
        // Keep `build_periodic` self-protecting even if future call paths bypass outer validation.
        Self::validate_topology_model(topology_model)?;
        if !topology_model.supports_periodic_facet_signatures() {
            return Err(TriangulationConstructionError::GeometricDegeneracy {
                message: format!(
                    "Topology {:?} does not support periodic facet signatures required for periodic image-point construction",
                    topology_model.kind(),
                ),
            }
            .into());
        }

        let domain = topology_model.periodic_domain().ok_or_else(|| {
            TriangulationConstructionError::GeometricDegeneracy {
                message: format!(
                    "Topology {:?} does not expose a periodic domain required for periodic image-point construction",
                    topology_model.kind(),
                ),
            }
        })?;
        let n = canonical_vertices.len();
        let min_points = 2 * D + 1;
        if n < min_points {
            return Err(TriangulationConstructionError::GeometricDegeneracy {
                message: format!(
                    "Periodic {D}D triangulation requires at least {min_points} points, got {n}"
                ),
            }
            .into());
        }

        // 3^D offset grid; zero-offset index = (3^D - 1) / 2.
        let three_pow_d: usize = 3_usize.pow(u32::try_from(D).expect("dimension D fits in u32"));
        let zero_offset_idx = (three_pow_d - 1) / 2;

        // Collect canonical UUIDs for key lookup after full DT is built.
        let canonical_uuids: Vec<Uuid> = canonical_vertices.iter().map(Vertex::uuid).collect();
        let perturb_units = |canon_idx: usize, axis: usize| -> i64 {
            let mut h = FNV_OFFSET_BASIS;
            h ^= u64::try_from(canon_idx).expect("canonical index fits in u64");
            h = h.wrapping_mul(FNV_PRIME);
            h ^= u64::try_from(axis).expect("axis index fits in u64");
            h = h.wrapping_mul(FNV_PRIME);
            let span = u64::try_from(2 * MAX_OFFSET_UNITS + 1).expect("span fits in u64");
            i64::try_from(h % span).expect("residue fits in i64") - MAX_OFFSET_UNITS
        };
        let image_jitter_units = |canon_idx: usize, axis: usize, image_idx: usize| -> i64 {
            let mut h = FNV_OFFSET_BASIS;
            h ^= u64::try_from(canon_idx).expect("canonical index fits in u64");
            h = h.wrapping_mul(FNV_PRIME);
            h ^= u64::try_from(axis).expect("axis index fits in u64");
            h = h.wrapping_mul(FNV_PRIME);
            h ^= u64::try_from(image_idx).expect("image index fits in u64");
            h = h.wrapping_mul(FNV_PRIME);
            let span = u64::try_from(2 * IMAGE_JITTER_UNITS + 1).expect("span fits in u64");
            i64::try_from(h % span).expect("residue fits in i64") - IMAGE_JITTER_UNITS
        };

        let canonical_f64: Vec<[f64; D]> = canonical_vertices
            .iter()
            .enumerate()
            .map(|(canon_idx, v)| {
                let orig_coords = v.point().coords();
                let mut coords = [0_f64; D];
                for i in 0..D {
                    let domain_i = domain[i];
                    let orig = orig_coords[i]
                        .to_f64()
                        .expect("canonical coordinate is finite and convertible");
                    let normalized = (orig / domain_i).clamp(0.0, 1.0 - f64::EPSILON);
                    let u = (normalized * TWO_POW_52_F64)
                        .floor()
                        .to_i64()
                        .expect("grid index fits in i64");
                    let min_off = -u.min(MAX_OFFSET_UNITS);
                    let max_off = (TWO_POW_52_I64 - 1 - u).min(MAX_OFFSET_UNITS);
                    let off = perturb_units(canon_idx, i).clamp(min_off, max_off);
                    let adjusted_u = <f64 as num_traits::NumCast>::from(u + off)
                        .expect("adjusted grid index fits in f64");
                    coords[i] = (adjusted_u / TWO_POW_52_F64) * domain_i;
                }
                coords
            })
            .collect();

        let mut image_uuid_to_canonical_with_offset: FastHashMap<Uuid, (Uuid, [i8; D])> =
            FastHashMap::default();
        let mut expanded: Vec<Vertex<T, U, D>> = Vec::with_capacity(n.saturating_mul(three_pow_d));
        for k in 0..three_pow_d {
            // Per-axis integer offsets {-1, 0, +1}.
            let mut offset = [0i8; D];
            for (i, offset_val) in offset.iter_mut().enumerate() {
                let digit =
                    (k / 3_usize.pow(u32::try_from(i).expect("dimension index fits in u32"))) % 3;
                // Map {0, 1, 2} → {-1, 0, +1}.
                *offset_val = i8::try_from(digit).expect("digit is 0, 1, or 2; fits in i8") - 1;
            }

            let is_canonical = k == zero_offset_idx;
            for (canon_idx, v) in canonical_vertices.iter().enumerate() {
                let mut new_coords = [T::zero(); D];
                for i in 0..D {
                    let shift_f64 = <f64 as From<i8>>::from(offset[i]) * domain[i];
                    let jitter_f64 = if is_canonical {
                        0.0
                    } else {
                        let jitter_units = image_jitter_units(canon_idx, i, k);
                        (<f64 as num_traits::NumCast>::from(jitter_units)
                            .expect("jitter fits in f64")
                            / TWO_POW_52_F64)
                            * domain[i]
                    };
                    let coord_f64 = canonical_f64[canon_idx][i] + shift_f64 + jitter_f64;
                    new_coords[i] =
                        <T as num_traits::NumCast>::from(coord_f64).ok_or_else(|| {
                            TriangulationConstructionError::GeometricDegeneracy {
                                message: format!("Overflow on axis {i}: image coord {coord_f64}"),
                            }
                        })?;
                }
                let new_point = Point::new(new_coords);
                if is_canonical {
                    image_uuid_to_canonical_with_offset.insert(v.uuid(), (v.uuid(), [0_i8; D]));
                    let canonical_v = Vertex::new_with_uuid(new_point, v.uuid(), v.data);
                    expanded.push(canonical_v);
                } else {
                    let image_v: Vertex<T, U, D> = Vertex::from_point(new_point);
                    image_uuid_to_canonical_with_offset.insert(image_v.uuid(), (v.uuid(), offset));
                    expanded.push(image_v);
                }
            }
        }
        let expanded_base_options = construction_options
            .with_initial_simplex_strategy(InitialSimplexStrategy::Balanced)
            .without_global_repair_fallback();
        let expanded_options = match construction_options.retry_policy() {
            RetryPolicy::Disabled => expanded_base_options,
            RetryPolicy::Shuffled { base_seed, .. }
            | RetryPolicy::DebugOnlyShuffled { base_seed, .. } => expanded_base_options
                .with_retry_policy(RetryPolicy::Shuffled {
                    attempts: NonZeroUsize::new(24).expect("literal is non-zero"),
                    base_seed,
                }),
        };
        let full_dt: DelaunayTriangulation<K, U, V, D> =
            match DelaunayTriangulation::with_topology_guarantee_and_options(
                kernel,
                &expanded,
                TopologyGuarantee::Pseudomanifold,
                expanded_options,
            ) {
                Ok(dt) => dt,
                Err(primary_err) if D > 2 => {
                    let (total_attempts, retry_seed) = match expanded_options.retry_policy() {
                        RetryPolicy::Disabled => (0_usize, None),
                        RetryPolicy::Shuffled {
                            attempts,
                            base_seed,
                        }
                        | RetryPolicy::DebugOnlyShuffled {
                            attempts,
                            base_seed,
                        } => (
                            attempts.get().saturating_mul(4).clamp(24, 256),
                            Some(base_seed.unwrap_or(0xA5A5_5A5A_D1E1_A1E1_u64)),
                        ),
                    };

                    let mut built: Option<DelaunayTriangulation<K, U, V, D>> = None;
                    let mut last_insert_error: Option<String> = None;
                    let mut last_skipped_insertion: Option<String> = None;
                    let mut best_fallback_stats: (usize, usize, usize, usize) = (0, 0, 0, 0);
                    let mut insertion_order: Vec<usize> = Vec::with_capacity(expanded.len());
                    let canonical_start = zero_offset_idx * n;
                    let canonical_end = canonical_start + n;
                    for attempt_idx in 0..total_attempts {
                        insertion_order.clear();
                        insertion_order.extend(canonical_start..canonical_end);
                        insertion_order.extend(0..canonical_start);
                        insertion_order.extend(canonical_end..expanded.len());

                        if attempt_idx > 0 {
                            let retry_seed = retry_seed
                                .expect("retry_seed is only used when retry attempts are enabled");
                            let attempt_u64 =
                                u64::try_from(attempt_idx).expect("attempt index fits in u64");
                            let mut rng = StdRng::seed_from_u64(
                                retry_seed
                                    .wrapping_add(attempt_u64.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                            );
                            let (canonical_prefix, image_suffix) = insertion_order.split_at_mut(n);
                            debug_assert_eq!(canonical_prefix.len(), n);
                            image_suffix.shuffle(&mut rng);
                        }

                        let mut candidate_dt: DelaunayTriangulation<K, U, V, D> =
                            DelaunayTriangulation::with_empty_kernel_and_topology_guarantee(
                                kernel.clone(),
                                TopologyGuarantee::Pseudomanifold,
                            );
                        candidate_dt.set_delaunay_repair_policy(DelaunayRepairPolicy::Never);
                        let mut inserted = 0_usize;
                        let mut skipped = 0_usize;
                        let mut hard_errors = 0_usize;
                        for (insert_idx, &source_idx) in insertion_order.iter().enumerate() {
                            match candidate_dt.insert_with_statistics(expanded[source_idx]) {
                                Ok((InsertionOutcome::Inserted { .. }, _stats)) => {
                                    inserted = inserted.saturating_add(1);
                                }
                                Ok((InsertionOutcome::Skipped { error }, _stats)) => {
                                    skipped = skipped.saturating_add(1);
                                    last_skipped_insertion = Some(format!(
                                        "attempt={attempt_idx} insert_idx={insert_idx} source_idx={source_idx}: {error}",
                                    ));
                                }
                                Err(err) => {
                                    hard_errors = hard_errors.saturating_add(1);
                                    last_insert_error = Some(format!(
                                        "attempt={attempt_idx} insert_idx={insert_idx} source_idx={source_idx}: {err}",
                                    ));
                                }
                            }
                        }

                        let canonical_present = canonical_uuids
                            .iter()
                            .filter(|uuid| candidate_dt.tds().vertex_key_from_uuid(uuid).is_some())
                            .count();
                        if canonical_present > best_fallback_stats.0
                            || (canonical_present == best_fallback_stats.0
                                && inserted > best_fallback_stats.1)
                        {
                            best_fallback_stats =
                                (canonical_present, inserted, skipped, hard_errors);
                        }

                        if canonical_present == n
                            && candidate_dt.number_of_simplices() > 0
                            && candidate_dt.tds().is_valid().is_ok()
                        {
                            built = Some(candidate_dt);
                            break;
                        }
                    }

                    if let Some(dt) = built {
                        dt
                    } else {
                        let canonical_vertex_uuid_sample: Vec<Uuid> = canonical_vertices
                            .iter()
                            .take(3)
                            .map(Vertex::uuid)
                            .collect();
                        return Err(TriangulationConstructionError::GeometricDegeneracy {
                            message: format!(
                                "Periodic expanded DT construction failed (no fallback): canonical_vertices_len={}, canonical_vertex_uuid_sample={canonical_vertex_uuid_sample:?}, primary_err={primary_err}, last_insert_error={:?}, last_skipped_insertion={:?}, best_fallback_stats(canonical_present,inserted,skipped,hard_errors)={:?}, topology_guarantee={topology_guarantee:?}, construction_options={construction_options:?}",
                                canonical_vertices.len(),
                                last_insert_error,
                                last_skipped_insertion,
                                best_fallback_stats,
                            ),
                        }
                        .into());
                    }
                }
                Err(err) => return Err(err),
            };

        let tds_ref = full_dt.tds();

        // Map canonical UUIDs → VertexKeys in the full DT.
        let Some(central_keys) = canonical_uuids
            .iter()
            .map(|uuid| tds_ref.vertex_key_from_uuid(uuid))
            .collect::<Option<Vec<_>>>()
        else {
            return Err(TriangulationConstructionError::GeometricDegeneracy {
                message: format!(
                    "Periodic expanded DT is missing at least one canonical vertex: canonical_vertices_len={}",
                    canonical_uuids.len()
                ),
            }
            .into());
        };
        let central_key_set: VertexKeySet = central_keys.into_iter().collect();

        // Map every full-DT vertex key to its canonical key and lattice offset.
        let mut vertex_key_to_lifted: FastHashMap<VertexKey, (VertexKey, [i8; D])> =
            FastHashMap::default();
        for vk in tds_ref.vertex_keys() {
            let Some(vertex) = tds_ref.vertex(vk) else {
                continue;
            };
            let Some((canonical_uuid, offset)) =
                image_uuid_to_canonical_with_offset.get(&vertex.uuid())
            else {
                continue;
            };
            let Some(canonical_key) = tds_ref.vertex_key_from_uuid(canonical_uuid) else {
                continue;
            };
            vertex_key_to_lifted.insert(vk, (canonical_key, *offset));
        }

        let normalize_simplex_lifted =
            |simplex_key: SimplexKey| -> Option<Vec<(VertexKey, [i8; D])>> {
                let simplex = tds_ref.simplex(simplex_key)?;
                let mut lifted: Vec<(VertexKey, [i8; D])> = simplex
                    .vertices()
                    .iter()
                    .map(|vk| vertex_key_to_lifted.get(vk).copied())
                    .collect::<Option<Vec<_>>>()?;

                let mut canonical_keys: Vec<VertexKey> = lifted.iter().map(|(ck, _)| *ck).collect();
                canonical_keys.sort_unstable();
                canonical_keys.dedup();
                if canonical_keys.len() != D + 1 {
                    // Simplex collapses in the quotient (repeated canonical vertex); skip it.
                    return None;
                }

                let (anchor_idx, _) = lifted.iter().enumerate().min_by_key(|(_, (ck, _))| *ck)?;
                let anchor_offset = lifted[anchor_idx].1;
                for (_, offset) in &mut lifted {
                    for axis in 0..D {
                        offset[axis] -= anchor_offset[axis];
                    }
                }

                Some(lifted)
            };
        let simplex_barycenter_in_fundamental_domain = |simplex_key: SimplexKey| -> Option<bool> {
            let simplex = tds_ref.simplex(simplex_key)?;
            let mut sums = [0.0_f64; D];
            for vk in simplex.vertices() {
                let vertex = tds_ref.vertex(*vk)?;
                let coords = vertex.point().coords();
                for (axis, sum) in sums.iter_mut().enumerate() {
                    *sum += coords[axis].to_f64()?;
                }
            }
            let denom = <f64 as num_traits::NumCast>::from(D + 1)
                .expect("simplex vertex count fits in f64 for D");
            for (axis, sum) in sums.iter().enumerate() {
                let bary = *sum / denom;
                let period = domain[axis];
                if !(bary >= 0.0 && bary < period) {
                    return Some(false);
                }
            }
            Some(true)
        };

        // Build unique symbolic candidates from all full-DT simplices.
        // Candidate tuple layout (see type alias):
        // (symbolic_signature, lifted_ordered, periodic_facet_keys, in_domain_hint)
        // where `lifted_ordered` preserves the observed per-simplex vertex order from
        // `normalize_simplex_lifted` (it is not canonical-key-sorted).
        let mut candidates_by_symbolic: FastHashMap<SymbolicSignature<D>, PeriodicCandidate<D>> =
            FastHashMap::default();
        for ck in tds_ref.simplex_keys() {
            let Some(lifted_vertices) = normalize_simplex_lifted(ck) else {
                continue;
            };
            let in_domain = simplex_barycenter_in_fundamental_domain(ck).unwrap_or(false);
            let mut symbolic_signature = lifted_vertices.clone();
            symbolic_signature.sort_unstable();
            let lifted_ordered = lifted_vertices.clone();
            let mut periodic_facets: Vec<PeriodicFacetKey> = Vec::with_capacity(D + 1);
            for facet_idx in 0..=D {
                periodic_facets.push(Self::derive_periodic_facet_key(&lifted_ordered, facet_idx)?);
            }

            if let Some(existing) = candidates_by_symbolic.get_mut(&symbolic_signature) {
                if in_domain {
                    existing.3 = true;
                }
            } else {
                candidates_by_symbolic.insert(
                    symbolic_signature.clone(),
                    (
                        symbolic_signature,
                        lifted_ordered,
                        periodic_facets,
                        in_domain,
                    ),
                );
            }
        }
        let mut candidates: Vec<PeriodicCandidate<D>> =
            candidates_by_symbolic.into_values().collect();
        if candidates.is_empty() {
            return Err(TriangulationConstructionError::GeometricDegeneracy {
                message: "No quotient periodic simplices found in full image DT".to_owned(),
            }
            .into());
        }
        candidates.sort_by(|a, b| b.3.cmp(&a.3).then_with(|| a.0.cmp(&b.0)));

        let (search_attempts, search_seed) = match construction_options.retry_policy() {
            RetryPolicy::Disabled => (1_usize, 0xD1CE_0B5E_2100_0001_u64),
            RetryPolicy::Shuffled {
                attempts,
                base_seed,
            }
            | RetryPolicy::DebugOnlyShuffled {
                attempts,
                base_seed,
            } => (
                attempts
                    .get()
                    .saturating_add(1)
                    .saturating_mul(512)
                    .clamp(512, 4096),
                base_seed.unwrap_or(0xD1CE_0B5E_2100_0001_u64),
            ),
        };

        let mut best_selected: Vec<bool> = Vec::new();
        let mut best_boundary_count = usize::MAX;
        let mut best_selected_count = 0_usize;
        let mut best_coverage_count = 0_usize;
        let mut best_abs_chi = i64::MAX;
        if D == 2 {
            let target_faces = central_key_set.len().saturating_mul(2);
            let mut edge_to_index: FastHashMap<PeriodicFacetKey, usize> = FastHashMap::default();
            let mut candidate_edges: Vec<[usize; 3]> = Vec::with_capacity(candidates.len());
            let mut candidate_in_domain: Vec<bool> = Vec::with_capacity(candidates.len());

            for candidate in &candidates {
                let mut edge_indices = [0usize; 3];
                for (slot, edge_key) in candidate.2.iter().enumerate() {
                    let next_index = edge_to_index.len();
                    let edge_index = *edge_to_index.entry(*edge_key).or_insert(next_index);
                    edge_indices[slot] = edge_index;
                }
                candidate_edges.push(edge_indices);
                candidate_in_domain.push(candidate.3);
            }
            let exact_search_node_limit = candidate_edges
                .len()
                .saturating_mul(edge_to_index.len().max(1))
                .saturating_mul(512)
                .clamp(100_000, 5_000_000);

            if let Some(exact_selected) = search_closed_2d_selection(
                &candidate_edges,
                &candidate_in_domain,
                target_faces,
                edge_to_index.len(),
                exact_search_node_limit,
            ) {
                best_selected_count = exact_selected
                    .iter()
                    .filter(|&&is_selected| is_selected)
                    .count();
                best_coverage_count = central_key_set.len();
                best_boundary_count = 0;
                best_abs_chi = 0;
                best_selected = exact_selected;
            }
        }

        if best_selected.is_empty() {
            let base_order: Vec<usize> = (0..candidates.len()).collect();
            for attempt_idx in 0..search_attempts {
                let mut order = base_order.clone();
                if attempt_idx > 0 {
                    let attempt_u64 =
                        u64::try_from(attempt_idx).expect("attempt index fits in u64");
                    let mut rng = StdRng::seed_from_u64(
                        search_seed.wrapping_add(attempt_u64.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                    );
                    order.shuffle(&mut rng);
                }
                // Keep in-domain representatives first while preserving randomized tie-breaks.
                order.sort_by(|a, b| candidates[*b].3.cmp(&candidates[*a].3));

                let mut selected = vec![false; candidates.len()];
                let mut facet_counts: FastHashMap<PeriodicFacetKey, u8> = FastHashMap::default();

                // Pass 1: greedy maximal subset with no canonical facet incidence > 2.
                for idx in order.iter().copied() {
                    let candidate_facets = &candidates[idx].2;
                    if candidate_facets
                        .iter()
                        .any(|facet| facet_counts.get(facet).copied().unwrap_or(0) >= 2)
                    {
                        continue;
                    }
                    selected[idx] = true;
                    for facet in candidate_facets {
                        *facet_counts.entry(*facet).or_insert(0) += 1;
                    }
                }

                // Pass 2: only add simplices that strictly reduce boundary facets (count == 1).
                let mut improved = true;
                while improved {
                    improved = false;
                    for idx in order.iter().copied() {
                        if selected[idx] {
                            continue;
                        }
                        let candidate_facets = &candidates[idx].2;
                        if candidate_facets
                            .iter()
                            .any(|facet| facet_counts.get(facet).copied().unwrap_or(0) >= 2)
                        {
                            continue;
                        }

                        let boundary_delta: i32 = candidate_facets
                            .iter()
                            .map(
                                |facet| match facet_counts.get(facet).copied().unwrap_or(0) {
                                    0 => 1,
                                    1 => -1,
                                    _ => 0,
                                },
                            )
                            .sum();

                        if boundary_delta < 0 {
                            selected[idx] = true;
                            for facet in candidate_facets {
                                *facet_counts.entry(*facet).or_insert(0) += 1;
                            }
                            improved = true;
                        }
                    }
                }
                // Pass 3: local refinement with both add and remove moves.
                // This escapes add-only local minima in D>2 where closure requires swaps.
                loop {
                    let mut best_move: Option<(bool, usize, i32)> = None;
                    for idx in order.iter().copied() {
                        let candidate_facets = &candidates[idx].2;
                        if selected[idx] {
                            let boundary_delta: i32 = candidate_facets
                                .iter()
                                .map(
                                    |facet| match facet_counts.get(facet).copied().unwrap_or(0) {
                                        1 => -1,
                                        2 => 1,
                                        _ => 0,
                                    },
                                )
                                .sum();
                            if boundary_delta < 0
                                && best_move
                                    .is_none_or(|(_, _, best_delta)| boundary_delta < best_delta)
                            {
                                best_move = Some((false, idx, boundary_delta));
                            }
                        } else {
                            if candidate_facets
                                .iter()
                                .any(|facet| facet_counts.get(facet).copied().unwrap_or(0) >= 2)
                            {
                                continue;
                            }

                            let boundary_delta: i32 = candidate_facets
                                .iter()
                                .map(
                                    |facet| match facet_counts.get(facet).copied().unwrap_or(0) {
                                        0 => 1,
                                        1 => -1,
                                        _ => 0,
                                    },
                                )
                                .sum();
                            if boundary_delta < 0
                                && best_move
                                    .is_none_or(|(_, _, best_delta)| boundary_delta < best_delta)
                            {
                                best_move = Some((true, idx, boundary_delta));
                            }
                        }
                    }

                    let Some((is_add, idx, _)) = best_move else {
                        break;
                    };
                    let candidate_facets = &candidates[idx].2;
                    if is_add {
                        selected[idx] = true;
                        for facet in candidate_facets {
                            *facet_counts.entry(*facet).or_insert(0) += 1;
                        }
                    } else {
                        selected[idx] = false;
                        for facet in candidate_facets {
                            if let Some(count) = facet_counts.get_mut(facet) {
                                *count -= 1;
                                if *count == 0 {
                                    facet_counts.remove(facet);
                                }
                            }
                        }
                    }
                }

                let boundary_count = facet_counts.values().filter(|&&count| count == 1).count();
                let selected_count = selected.iter().filter(|&&is_selected| is_selected).count();
                let mut covered: VertexKeySet = VertexKeySet::default();
                for (idx, is_selected) in selected.iter().copied().enumerate() {
                    if !is_selected {
                        continue;
                    }
                    for (vertex_key, _) in &candidates[idx].1 {
                        covered.insert(*vertex_key);
                    }
                }
                let coverage_count = covered.len();
                let abs_chi = if D == 2 {
                    let v_count =
                        i64::try_from(central_key_set.len()).expect("vertex count fits in i64");
                    let e_count =
                        i64::try_from(facet_counts.len()).expect("edge/facet count fits in i64");
                    let f_count = i64::try_from(selected_count).expect("simplex count fits in i64");
                    (v_count - e_count + f_count).abs()
                } else {
                    0
                };
                if boundary_count < best_boundary_count
                    || (boundary_count == best_boundary_count
                        && (if D == 2 {
                            abs_chi < best_abs_chi
                                || (abs_chi == best_abs_chi && selected_count > best_selected_count)
                        } else {
                            coverage_count > best_coverage_count
                                || (coverage_count == best_coverage_count
                                    && selected_count > best_selected_count)
                        }))
                {
                    best_boundary_count = boundary_count;
                    best_selected_count = selected_count;
                    best_coverage_count = coverage_count;
                    best_abs_chi = abs_chi;
                    best_selected = selected;
                }
                let best_has_full_canonical_coverage = best_coverage_count == central_key_set.len();
                if best_boundary_count == 0
                    && ((D == 2 && best_abs_chi == 0)
                        || (D > 2 && best_has_full_canonical_coverage))
                {
                    break;
                }
            }
        }

        if best_selected.is_empty() {
            return Err(TriangulationConstructionError::GeometricDegeneracy {
                message: "Periodic quotient selection failed to pick any candidate simplices"
                    .to_owned(),
            }
            .into());
        }
        if D == 2 && best_boundary_count > 0 {
            return Err(TriangulationConstructionError::GeometricDegeneracy {
                message: format!(
                    "Periodic quotient selection left {best_boundary_count} boundary facets after {search_attempts} attempts (full_vertices={}, full_simplices={}, canonical_vertices={}, candidates={}, selected_simplices={})",
                    tds_ref.number_of_vertices(),
                    tds_ref.number_of_simplices(),
                    central_key_set.len(),
                    candidates.len(),
                    best_selected_count,
                ),
            }
            .into());
        }
        if D == 2 && best_abs_chi != 0 {
            return Err(TriangulationConstructionError::GeometricDegeneracy {
                message: format!(
                    "Periodic quotient selection could not reach χ=0 in 2D (best |χ|={best_abs_chi}) after {search_attempts} attempts",
                ),
            }
            .into());
        }
        let has_full_canonical_coverage = best_coverage_count == central_key_set.len();
        if D > 2 && !has_full_canonical_coverage {
            return Err(TriangulationConstructionError::GeometricDegeneracy {
                message: format!(
                    "Periodic quotient selection covered only {} of {} canonical vertices in {D}D",
                    best_coverage_count,
                    central_key_set.len(),
                ),
            }
            .into());
        }
        if D > 2 {
            // In the quotient TDS, simplices that collapse to the same canonical vertex set cannot
            // be distinct facet-neighbors: they would share all D+1 vertices and violate the
            // mirror-facet invariant enforced by `set_neighbors_by_key`.
            //
            // Keep at most one selected representative per canonical simplex. Prefer in-domain
            // representatives, then deterministic symbolic ordering.
            let mut selected_by_canonical: FastHashMap<Vec<VertexKey>, usize> =
                FastHashMap::default();
            let mut dedup_selected = vec![false; candidates.len()];

            for (idx, is_selected) in best_selected.iter().copied().enumerate() {
                if !is_selected {
                    continue;
                }
                let mut canonical_keys: Vec<VertexKey> =
                    candidates[idx].1.iter().map(|(vk, _)| *vk).collect();
                canonical_keys.sort_unstable();

                if let Some(existing_idx) = selected_by_canonical.get(&canonical_keys).copied() {
                    let existing_in_domain = candidates[existing_idx].3;
                    let candidate_in_domain = candidates[idx].3;
                    let should_replace = (!existing_in_domain && candidate_in_domain)
                        || (existing_in_domain == candidate_in_domain
                            && candidates[idx].0 < candidates[existing_idx].0);
                    if should_replace {
                        dedup_selected[existing_idx] = false;
                        dedup_selected[idx] = true;
                        selected_by_canonical.insert(canonical_keys, idx);
                    }
                } else {
                    dedup_selected[idx] = true;
                    selected_by_canonical.insert(canonical_keys, idx);
                }
            }

            best_selected = dedup_selected;
        }

        let mut representative_lifted_by_symbolic: FastHashMap<
            SymbolicSignature<D>,
            SymbolicSignature<D>,
        > = FastHashMap::default();
        for (idx, is_selected) in best_selected.iter().copied().enumerate() {
            if !is_selected {
                continue;
            }
            let (symbolic_signature, lifted_ordered, _, _) = &candidates[idx];
            representative_lifted_by_symbolic
                .insert(symbolic_signature.clone(), lifted_ordered.clone());
        }

        // Clone TDS and rebuild simplex complex from quotient representatives.
        let mut tds_mut = tds_ref.clone();

        // Remove all simplices first.
        let all_simplices: Vec<SimplexKey> = tds_mut.simplex_keys().collect();
        tds_mut.remove_simplices_by_keys(&all_simplices);

        // Remove all image vertices.
        let image_vertex_keys: Vec<VertexKey> = tds_mut
            .vertex_keys()
            .filter(|vk| !central_key_set.contains(vk))
            .collect();
        for &vk in &image_vertex_keys {
            tds_mut.remove_vertex(vk).map_err(|e| {
                TriangulationConstructionError::InternalInconsistency {
                    message: format!("Failed to remove image vertex: {e}"),
                }
            })?;
        }

        // Insert quotient simplices.
        let mut signatures_sorted: Vec<Vec<(VertexKey, [i8; D])>> =
            representative_lifted_by_symbolic.keys().cloned().collect();
        signatures_sorted.sort_unstable();

        let mut inserted_simplex_keys: Vec<SimplexKey> =
            Vec::with_capacity(signatures_sorted.len());
        let mut rep_lifted_by_key: FastHashMap<SimplexKey, Vec<(VertexKey, [i8; D])>> =
            FastHashMap::default();

        for signature in signatures_sorted {
            let Some(lifted_vertices) = representative_lifted_by_symbolic.get(&signature) else {
                continue;
            };
            let canonical_vertex_keys: Vec<VertexKey> =
                lifted_vertices.iter().map(|(ck, _)| *ck).collect();
            let mut simplex = Simplex::new(canonical_vertex_keys, None).map_err(|e| {
                TriangulationConstructionError::GeometricDegeneracy {
                    message: format!("Failed to create quotient periodic simplex: {e}"),
                }
            })?;
            let offsets: PeriodicOffsetBuffer<D> =
                lifted_vertices.iter().map(|(_, offset)| *offset).collect();
            simplex.set_periodic_vertex_offsets(offsets).map_err(|e| {
                TriangulationConstructionError::GeometricDegeneracy {
                    message: format!("Failed to set quotient periodic offsets: {e}"),
                }
            })?;
            let ck = tds_mut
                .insert_simplex_with_mapping_trusted_vertices(simplex)
                .map_err(|e| TriangulationConstructionError::GeometricDegeneracy {
                    message: format!("Failed to insert quotient periodic simplex: {e}"),
                })?;
            inserted_simplex_keys.push(ck);
            rep_lifted_by_key.insert(ck, lifted_vertices.clone());
        }
        if inserted_simplex_keys.is_empty() {
            return Err(TriangulationConstructionError::GeometricDegeneracy {
                message: "No simplices survived periodic quotient reconstruction".to_owned(),
            }
            .into());
        }

        // Sanity-check periodic facet multiplicities before neighbor rewiring.
        // In a valid simplicial manifold each facet is incident to at most two simplices.
        let mut periodic_facet_counts: FastHashMap<PeriodicFacetKey, usize> =
            FastHashMap::default();
        for lifted in rep_lifted_by_key.values() {
            for facet_idx in 0..=D {
                let periodic_facet_key = Self::derive_periodic_facet_key(lifted, facet_idx)?;
                *periodic_facet_counts.entry(periodic_facet_key).or_insert(0) += 1;
            }
        }
        let overloaded_facets: Vec<(PeriodicFacetKey, usize)> = periodic_facet_counts
            .into_iter()
            .filter(|(_, count)| *count > 2)
            .collect();
        if !overloaded_facets.is_empty() {
            return Err(TriangulationConstructionError::GeometricDegeneracy {
                message: format!(
                    "Periodic quotient selection overcounts periodic facets ({} overloaded); selected_simplices={}, sample={:?}",
                    overloaded_facets.len(),
                    rep_lifted_by_key.len(),
                    overloaded_facets.iter().take(4).collect::<Vec<_>>(),
                ),
            }
            .into());
        }

        // Rebuild neighbor pointers by pairing equal symbolic facet signatures in the quotient.
        let mut neighbor_updates: FastHashMap<SimplexKey, Vec<Option<SimplexKey>>> =
            inserted_simplex_keys
                .iter()
                .copied()
                .map(|ck| (ck, vec![None; D + 1]))
                .collect();

        let mut facet_occurrences: FastHashMap<PeriodicFacetKey, Vec<(SimplexKey, usize)>> =
            FastHashMap::default();
        for &rep_ck in &inserted_simplex_keys {
            let Some(lifted) = rep_lifted_by_key.get(&rep_ck) else {
                continue;
            };
            for facet_idx in 0..=D {
                let sig = Self::derive_periodic_facet_key(lifted, facet_idx)?;
                facet_occurrences
                    .entry(sig)
                    .or_default()
                    .push((rep_ck, facet_idx));
            }
        }

        for (_facet_sig, occurrences) in facet_occurrences {
            match occurrences.as_slice() {
                [(a_ck, a_idx), (b_ck, b_idx)] => {
                    let a_lifted = rep_lifted_by_key.get(a_ck).ok_or_else(|| {
                        TriangulationConstructionError::InternalInconsistency {
                            message: format!(
                                "missing lifted representative for quotient simplex {a_ck:?}"
                            ),
                        }
                    })?;
                    let b_lifted = rep_lifted_by_key.get(b_ck).ok_or_else(|| {
                        TriangulationConstructionError::InternalInconsistency {
                            message: format!(
                                "missing lifted representative for quotient simplex {b_ck:?}"
                            ),
                        }
                    })?;
                    let shares_all_canonical_vertices = a_lifted
                        .iter()
                        .zip(b_lifted.iter())
                        .all(|((a_vk, _), (b_vk, _))| a_vk == b_vk);

                    if shares_all_canonical_vertices {
                        return Err(TriangulationConstructionError::GeometricDegeneracy {
                            message: format!(
                                "Periodic quotient produced distinct simplices with identical canonical vertices across a shared facet: {a_ck:?}[{a_idx}] <-> {b_ck:?}[{b_idx}]",
                            ),
                        }
                        .into());
                    }
                    neighbor_updates.get_mut(a_ck).ok_or_else(|| {
                        TriangulationConstructionError::InternalInconsistency {
                            message: format!(
                                "missing neighbor vector for quotient simplex {a_ck:?}"
                            ),
                        }
                    })?[*a_idx] = Some(*b_ck);
                    neighbor_updates.get_mut(b_ck).ok_or_else(|| {
                        TriangulationConstructionError::InternalInconsistency {
                            message: format!(
                                "missing neighbor vector for quotient simplex {b_ck:?}"
                            ),
                        }
                    })?[*b_idx] = Some(*a_ck);
                }
                [(a_ck, a_idx)] => {
                    // Self-identified periodic facet.
                    neighbor_updates.get_mut(a_ck).ok_or_else(|| {
                        TriangulationConstructionError::InternalInconsistency {
                            message: format!(
                                "missing neighbor vector for quotient simplex {a_ck:?}"
                            ),
                        }
                    })?[*a_idx] = Some(*a_ck);
                }
                _ => {
                    return Err(TriangulationConstructionError::GeometricDegeneracy {
                        message: format!(
                            "Periodic quotient facet signature has {} occurrences (expected 1 or 2): {occurrences:?}",
                            occurrences.len()
                        ),
                    }
                    .into());
                }
            }
        }

        let unmatched_count = neighbor_updates
            .values()
            .flat_map(|n| n.iter())
            .filter(|n| n.is_none())
            .count();
        if unmatched_count > 0 {
            return Err(TriangulationConstructionError::GeometricDegeneracy {
                message: format!(
                    "Periodic quotient reconstruction left {unmatched_count} unmatched neighbor slots"
                ),
            }
            .into());
        }

        // Apply neighbor updates.
        for &ck in &inserted_simplex_keys {
            let neighbors = neighbor_updates.remove(&ck).ok_or_else(|| {
                TriangulationConstructionError::InternalInconsistency {
                    message: format!(
                        "missing neighbor vector for inserted quotient simplex {ck:?}"
                    ),
                }
            })?;
            tds_mut.set_neighbors_by_key(ck, &neighbors).map_err(|e| {
                TriangulationConstructionError::InternalInconsistency {
                    message: format!("set_neighbors_by_key failed for {ck:?}: {e}"),
                }
            })?;
        }

        // Canonicalize quotient-simplex orientation after symbolic neighbor reconstruction.
        //
        // For periodic quotients, self-neighbor identifications can produce orientation
        // constraints that are contradictory for global normalization even when the local
        // adjacency invariants are still structurally valid. Keep this best-effort here and
        // defer hard failure to the subsequent `is_valid()` check.
        if let Err(_error) = tds_mut.normalize_coherent_orientation() {
            #[cfg(debug_assertions)]
            tracing::debug!(
                ?_error,
                "periodic quotient: skipping coherent-orientation normalization failure"
            );
        }
        // Rebuild incident-simplex pointers after topology surgery.
        tds_mut.assign_incident_simplices().map_err(|e| {
            TriangulationConstructionError::InternalInconsistency {
                message: format!("assign_incident_simplices failed: {e}"),
            }
        })?;
        if let Err(e) = tds_mut.is_valid() {
            return Err(TriangulationConstructionError::GeometricDegeneracy {
                message: format!("Periodic quotient TDS invalid before return: {e}"),
            }
            .into());
        }

        Ok(DelaunayTriangulation::from_tds_with_topology_guarantee(
            tds_mut,
            kernel.clone(),
            topology_guarantee,
        ))
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::construction::{DelaunayConstructionFailure, InsertionOrderStrategy};
    use crate::core::algorithms::flips::DelaunayRepairError;
    use crate::core::algorithms::incremental_insertion::{
        CavityFillingError, DelaunayRepairFailureContext, HullExtensionReason, NeighborWiringError,
    };
    use crate::core::algorithms::locate::{ConflictError, LocateError};
    use crate::core::facet::FacetError;
    use crate::core::simplex::SimplexValidationError;
    use crate::core::tds::{
        DelaunayValidationErrorKind, EntityKind, GeometricError, NeighborValidationError,
        TdsConstructionError,
    };
    use crate::core::util::uuid::UuidValidationError;
    use crate::core::validation::TriangulationValidationError;
    use crate::core::vertex::VertexBuilder;
    use crate::core::vertex::VertexValidationError;
    use crate::geometry::kernel::RobustKernel;
    use crate::repair::DelaunayRepairOperation;
    use crate::topology::traits::global_topology_model::{
        EuclideanModel, GlobalTopologyModel, GlobalTopologyModelError, ToroidalModel,
    };
    use crate::topology::traits::topological_space::{
        GlobalTopology, TopologyKind, ToroidalConstructionMode,
    };
    use crate::vertex;
    use approx::assert_relative_eq;
    use slotmap::{Key, KeyData};

    #[derive(Clone, Copy, Debug)]
    struct ValidationFailureModel;

    impl GlobalTopologyModel<2> for ValidationFailureModel {
        fn kind(&self) -> TopologyKind {
            TopologyKind::Euclidean
        }

        fn allows_boundary(&self) -> bool {
            true
        }

        fn validate_configuration(&self) -> Result<(), GlobalTopologyModelError> {
            Err(GlobalTopologyModelError::NonFiniteCoordinate {
                axis: 0,
                value: f64::NAN,
            })
        }

        fn canonicalize_point_in_place<T>(
            &self,
            _coords: &mut [T; 2],
        ) -> Result<(), GlobalTopologyModelError>
        where
            T: CoordinateScalar,
        {
            Ok(())
        }

        fn lift_for_orientation<T>(
            &self,
            coords: [T; 2],
            periodic_offset: Option<[i8; 2]>,
        ) -> Result<[T; 2], GlobalTopologyModelError>
        where
            T: CoordinateScalar,
        {
            if periodic_offset.is_some() {
                return Err(GlobalTopologyModelError::PeriodicOffsetsUnsupported {
                    kind: TopologyKind::Euclidean,
                });
            }
            Ok(coords)
        }
    }

    #[test]
    fn explicit_error_summaries_preserve_nested_source_kinds() {
        let tds = ExplicitTdsError::from(TdsError::DuplicateSimplices {
            message: "same vertex set appears twice".to_string(),
        });
        assert_eq!(tds.kind, ExplicitTdsErrorKind::DuplicateSimplices);
        assert!(tds.message.contains("Duplicate simplices"));

        let insertion = ExplicitInsertionError::from(InsertionError::TopologyValidation(
            TdsError::InconsistentDataStructure {
                message: "dangling neighbor".to_string(),
            },
        ));
        assert_eq!(
            insertion.source_kind,
            Some(InsertionErrorSourceKind::Tds(
                TdsErrorKind::InconsistentDataStructure,
            ))
        );

        let invariant = ExplicitInvariantError::from(InvariantError::Triangulation(
            TriangulationValidationError::Disconnected { simplex_count: 2 },
        ));
        assert_eq!(invariant.kind, ExplicitInvariantErrorKind::Triangulation);
        assert_eq!(
            invariant.detail,
            InvariantErrorSummaryDetail::Triangulation(
                TriangulationValidationErrorKind::Disconnected,
            )
        );

        let delaunay = ExplicitDelaunayValidationError::from(
            DelaunayTriangulationValidationError::RepairOperationFailed {
                operation: DelaunayRepairOperation::VertexRemoval,
                source: Box::new(DelaunayRepairError::HeuristicRebuildFailed {
                    message: "rebuild failed".to_string(),
                }),
            },
        );
        assert_eq!(
            delaunay.source_kind,
            Some(ExplicitDelaunayValidationSourceKind::Repair(
                DelaunayRepairErrorKind::HeuristicRebuildFailed,
            ))
        );

        let delaunay_tds = ExplicitDelaunayValidationError::from(
            DelaunayTriangulationValidationError::from(TdsError::InconsistentDataStructure {
                message: "dangling simplex".to_string(),
            }),
        );
        assert_eq!(delaunay_tds.kind, ExplicitDelaunayValidationErrorKind::Tds);
        assert_eq!(
            delaunay_tds.source_kind,
            Some(ExplicitDelaunayValidationSourceKind::Tds(
                TdsErrorKind::InconsistentDataStructure,
            ))
        );

        let delaunay_topology =
            ExplicitDelaunayValidationError::from(DelaunayTriangulationValidationError::from(
                TriangulationValidationError::Disconnected { simplex_count: 2 },
            ));
        assert_eq!(
            delaunay_topology.kind,
            ExplicitDelaunayValidationErrorKind::Triangulation,
        );
        assert_eq!(
            delaunay_topology.source_kind,
            Some(ExplicitDelaunayValidationSourceKind::Triangulation(
                TriangulationValidationErrorKind::Disconnected,
            ))
        );
    }

    fn assert_explicit_tds_error_kind(source: TdsError, expected_kind: ExplicitTdsErrorKind) {
        let summary = ExplicitTdsError::from(source);
        assert_eq!(summary.kind, expected_kind);
        assert!(!summary.message.is_empty());
    }

    #[test]
    fn explicit_tds_error_preserves_validation_error_kinds() {
        let simplex_key = SimplexKey::from(KeyData::from_ffi(1));
        let other_simplex_key = SimplexKey::from(KeyData::from_ffi(2));
        let vertex_key = VertexKey::from(KeyData::from_ffi(3));
        let uuid = Uuid::new_v4();

        assert_explicit_tds_error_kind(
            TdsError::InvalidVertex {
                vertex_id: uuid,
                source: VertexValidationError::InvalidUuid {
                    source: UuidValidationError::NilUuid,
                },
            },
            ExplicitTdsErrorKind::InvalidVertex,
        );
        assert_explicit_tds_error_kind(
            TdsError::InvalidSimplex {
                simplex_id: uuid,
                source: SimplexValidationError::DuplicateVertices,
            },
            ExplicitTdsErrorKind::InvalidSimplex,
        );
        assert_explicit_tds_error_kind(
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::Other {
                    message: "neighbor invariant failed".to_string(),
                },
            },
            ExplicitTdsErrorKind::InvalidNeighbors,
        );
        assert_explicit_tds_error_kind(
            TdsError::OrientationViolation {
                simplex1_key: simplex_key,
                simplex1_uuid: uuid,
                simplex2_key: other_simplex_key,
                simplex2_uuid: Uuid::new_v4(),
                simplex1_facet_index: 0,
                simplex2_facet_index: 1,
                facet_vertices: vec![vertex_key],
                simplex2_facet_vertices: vec![vertex_key],
                observed_odd_permutation: false,
                expected_odd_permutation: true,
            },
            ExplicitTdsErrorKind::OrientationViolation,
        );
        assert_explicit_tds_error_kind(
            TdsError::Geometric(GeometricError::DegenerateOrientation {
                message: "zero determinant".to_string(),
            }),
            ExplicitTdsErrorKind::Geometric,
        );
        assert_explicit_tds_error_kind(
            TdsError::FacetError(FacetError::InvalidFacetIndex {
                index: 4,
                facet_count: 4,
            }),
            ExplicitTdsErrorKind::FacetError,
        );
        assert_explicit_tds_error_kind(
            TdsError::DuplicateCoordinatesInSimplex {
                simplex_id: uuid,
                message: "two vertices share coordinates".to_string(),
            },
            ExplicitTdsErrorKind::DuplicateCoordinatesInSimplex,
        );
    }

    #[test]
    fn explicit_tds_error_preserves_lookup_and_operation_error_kinds() {
        let simplex_key = SimplexKey::from(KeyData::from_ffi(1));
        let vertex_key = VertexKey::from(KeyData::from_ffi(3));
        let uuid = Uuid::new_v4();

        assert_explicit_tds_error_kind(
            TdsError::DuplicateSimplices {
                message: "duplicate simplex vertex set".to_string(),
            },
            ExplicitTdsErrorKind::DuplicateSimplices,
        );
        assert_explicit_tds_error_kind(
            TdsError::FacetSharingViolation {
                facet_key: 42,
                existing_incident_count: 2,
                attempted_incident_count: 3,
                max_incident_count: 2,
                candidate_simplex_uuid: uuid,
                candidate_facet_index: 1,
            },
            ExplicitTdsErrorKind::FacetSharingViolation,
        );
        assert_explicit_tds_error_kind(
            TdsError::FailedToCreateSimplex {
                message: "simplex validation failed".to_string(),
            },
            ExplicitTdsErrorKind::FailedToCreateSimplex,
        );
        assert_explicit_tds_error_kind(
            TdsError::NotNeighbors {
                simplex1: uuid,
                simplex2: Uuid::new_v4(),
            },
            ExplicitTdsErrorKind::NotNeighbors,
        );
        assert_explicit_tds_error_kind(
            TdsError::MappingInconsistency {
                entity: EntityKind::Simplex,
                message: "uuid mapping was stale".to_string(),
            },
            ExplicitTdsErrorKind::MappingInconsistency,
        );
        assert_explicit_tds_error_kind(
            TdsError::VertexKeyRetrievalFailed {
                simplex_id: uuid,
                message: "simplex vertices unavailable".to_string(),
            },
            ExplicitTdsErrorKind::VertexKeyRetrievalFailed,
        );
        assert_explicit_tds_error_kind(
            TdsError::SimplexNotFound {
                simplex_key,
                context: "simplex lookup".to_string(),
            },
            ExplicitTdsErrorKind::SimplexNotFound,
        );
        assert_explicit_tds_error_kind(
            TdsError::VertexNotFound {
                vertex_key,
                context: "vertex lookup".to_string(),
            },
            ExplicitTdsErrorKind::VertexNotFound,
        );
        assert_explicit_tds_error_kind(
            TdsError::DimensionMismatch {
                expected: 4,
                actual: 3,
                context: "simplex arity".to_string(),
            },
            ExplicitTdsErrorKind::DimensionMismatch,
        );
        assert_explicit_tds_error_kind(
            TdsError::IndexOutOfBounds {
                index: 4,
                bound: 4,
                context: "facet index".to_string(),
            },
            ExplicitTdsErrorKind::IndexOutOfBounds,
        );
        assert_explicit_tds_error_kind(
            TdsError::InconsistentDataStructure {
                message: "dangling neighbor".to_string(),
            },
            ExplicitTdsErrorKind::InconsistentDataStructure,
        );
    }

    #[test]
    fn explicit_tds_error_preserves_construction_and_mutation_wrappers() {
        let uuid = Uuid::new_v4();
        let duplicate = ExplicitTdsError::from(TdsConstructionError::DuplicateUuid {
            entity: EntityKind::Simplex,
            uuid,
        });
        assert_eq!(duplicate.kind, ExplicitTdsErrorKind::DuplicateUuid);
        assert!(duplicate.message.contains(&uuid.to_string()));

        let mutation = ExplicitTdsError::from(TdsMutationError::from(
            TdsError::InconsistentDataStructure {
                message: "incident simplex assignment failed".to_string(),
            },
        ));
        assert_eq!(
            mutation.kind,
            ExplicitTdsErrorKind::InconsistentDataStructure
        );
        assert!(mutation.message.contains("incident simplex assignment"));
    }

    #[test]
    fn explicit_simplex_creation_error_preserves_typed_source() {
        let err = ExplicitConstructionError::SimplexCreation {
            simplex_index: 7,
            source: SimplexValidationError::DuplicateVertices,
        };

        let ExplicitConstructionError::SimplexCreation {
            simplex_index,
            source,
        } = &err
        else {
            panic!("expected simplex creation error, got {err:?}");
        };

        assert_eq!(*simplex_index, 7);
        assert_eq!(*source, SimplexValidationError::DuplicateVertices);
        assert!(err.to_string().contains("Simplex 7"));
        assert!(err.to_string().contains("Duplicate vertices"));
    }

    fn assert_explicit_insertion_error(
        source: InsertionError,
        expected_kind: ExplicitInsertionErrorKind,
        expected_source_kind: Option<InsertionErrorSourceKind>,
    ) {
        let summary = ExplicitInsertionError::from(source);
        assert_eq!(summary.kind, expected_kind);
        assert_eq!(summary.source_kind, expected_source_kind);
        assert!(!summary.message.is_empty());
    }

    #[test]
    fn explicit_insertion_error_preserves_stage_kinds_without_nested_sources() {
        let simplex_key = SimplexKey::from(KeyData::from_ffi(1));
        let uuid = Uuid::new_v4();

        assert_explicit_insertion_error(
            InsertionError::ConflictRegion(ConflictError::InvalidStartSimplex { simplex_key }),
            ExplicitInsertionErrorKind::ConflictRegion,
            None,
        );
        assert_explicit_insertion_error(
            InsertionError::Location(LocateError::EmptyTriangulation),
            ExplicitInsertionErrorKind::Location,
            None,
        );
        assert_explicit_insertion_error(
            InsertionError::CavityFilling {
                reason: CavityFillingError::MissingBoundarySimplex { simplex_key },
            },
            ExplicitInsertionErrorKind::CavityFilling,
            None,
        );
        assert_explicit_insertion_error(
            InsertionError::NeighborWiring {
                reason: NeighborWiringError::MissingSimplex { simplex_key },
            },
            ExplicitInsertionErrorKind::NeighborWiring,
            None,
        );
        assert_explicit_insertion_error(
            InsertionError::NonManifoldTopology {
                facet_hash: 0xabc,
                simplex_count: 3,
            },
            ExplicitInsertionErrorKind::NonManifoldTopology,
            None,
        );
        assert_explicit_insertion_error(
            InsertionError::HullExtension {
                reason: HullExtensionReason::NoVisibleFacets,
            },
            ExplicitInsertionErrorKind::HullExtension,
            None,
        );
        assert_explicit_insertion_error(
            InsertionError::DuplicateCoordinates {
                coordinates: "[0.0, 0.0]".to_string(),
            },
            ExplicitInsertionErrorKind::DuplicateCoordinates,
            None,
        );
        assert_explicit_insertion_error(
            InsertionError::DuplicateUuid {
                entity: EntityKind::Vertex,
                uuid,
            },
            ExplicitInsertionErrorKind::DuplicateUuid,
            None,
        );
    }

    #[test]
    fn explicit_insertion_error_preserves_nested_validation_source_kinds() {
        assert_explicit_insertion_error(
            InsertionError::DelaunayValidationFailed {
                source: DelaunayTriangulationValidationError::VerificationFailed {
                    message: "non-Delaunay facet".to_string(),
                },
            },
            ExplicitInsertionErrorKind::DelaunayValidationFailed,
            Some(InsertionErrorSourceKind::Delaunay(
                DelaunayValidationErrorKind::VerificationFailed,
            )),
        );
        assert_explicit_insertion_error(
            InsertionError::DelaunayRepairFailed {
                source: Box::new(DelaunayRepairError::PostconditionFailed {
                    message: "remaining violation".to_string(),
                }),
                context: DelaunayRepairFailureContext::LocalRepair,
            },
            ExplicitInsertionErrorKind::DelaunayRepairFailed,
            Some(InsertionErrorSourceKind::DelaunayRepair(
                DelaunayRepairErrorKind::PostconditionFailed,
            )),
        );
        assert_explicit_insertion_error(
            InsertionError::TopologyValidation(TdsError::Geometric(
                GeometricError::DegenerateOrientation {
                    message: "zero determinant".to_string(),
                },
            )),
            ExplicitInsertionErrorKind::TopologyValidation,
            Some(InsertionErrorSourceKind::Tds(TdsErrorKind::Geometric)),
        );
        assert_explicit_insertion_error(
            InsertionError::TopologyValidationFailed {
                source: TriangulationValidationError::RidgeLinkNotManifold {
                    ridge_key: 0xdef,
                    link_vertex_count: 4,
                    link_edge_count: 2,
                    max_degree: 3,
                    degree_one_vertices: 1,
                    connected: false,
                },
                message: "scoped topology validation".to_string(),
            },
            ExplicitInsertionErrorKind::TopologyValidationFailed,
            Some(InsertionErrorSourceKind::Triangulation(
                TriangulationValidationErrorKind::RidgeLinkNotManifold,
            )),
        );
    }

    #[derive(Clone, Copy, Debug)]
    struct CanonicalizationFailureModel;

    impl GlobalTopologyModel<2> for CanonicalizationFailureModel {
        fn kind(&self) -> TopologyKind {
            TopologyKind::Euclidean
        }

        fn allows_boundary(&self) -> bool {
            true
        }

        fn validate_configuration(&self) -> Result<(), GlobalTopologyModelError> {
            Ok(())
        }

        fn canonicalize_point_in_place<T>(
            &self,
            _coords: &mut [T; 2],
        ) -> Result<(), GlobalTopologyModelError>
        where
            T: CoordinateScalar,
        {
            Err(GlobalTopologyModelError::NonFiniteCoordinate {
                axis: 0,
                value: f64::NAN,
            })
        }

        fn lift_for_orientation<T>(
            &self,
            coords: [T; 2],
            periodic_offset: Option<[i8; 2]>,
        ) -> Result<[T; 2], GlobalTopologyModelError>
        where
            T: CoordinateScalar,
        {
            if periodic_offset.is_some() {
                return Err(GlobalTopologyModelError::PeriodicOffsetsUnsupported {
                    kind: TopologyKind::Euclidean,
                });
            }
            Ok(coords)
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct MissingPeriodicDomainModel;

    impl GlobalTopologyModel<2> for MissingPeriodicDomainModel {
        fn kind(&self) -> TopologyKind {
            TopologyKind::Toroidal
        }

        fn allows_boundary(&self) -> bool {
            false
        }

        fn validate_configuration(&self) -> Result<(), GlobalTopologyModelError> {
            Ok(())
        }

        fn canonicalize_point_in_place<T>(
            &self,
            _coords: &mut [T; 2],
        ) -> Result<(), GlobalTopologyModelError>
        where
            T: CoordinateScalar,
        {
            Ok(())
        }

        fn lift_for_orientation<T>(
            &self,
            coords: [T; 2],
            _periodic_offset: Option<[i8; 2]>,
        ) -> Result<[T; 2], GlobalTopologyModelError>
        where
            T: CoordinateScalar,
        {
            Ok(coords)
        }

        fn supports_periodic_facet_signatures(&self) -> bool {
            true
        }

        fn periodic_domain(&self) -> Option<&[f64; 2]> {
            None
        }
    }

    // -------------------------------------------------------------------------
    // Euclidean path — `new` is specialized for f64/(), no type annotations needed
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_euclidean_2d() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .build::<()>()
            .unwrap();
        assert_eq!(dt.number_of_vertices(), 3);
        assert_eq!(dt.dim(), 2);
        assert!(dt.as_triangulation().validate().is_ok());
    }

    #[test]
    fn test_builder_euclidean_3d() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .build::<()>()
            .unwrap();
        assert_eq!(dt.number_of_vertices(), 4);
        assert!(dt.validate().is_ok());
    }

    #[test]
    fn test_builder_topology_guarantee_propagated() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .topology_guarantee(TopologyGuarantee::Pseudomanifold)
            .build::<()>()
            .unwrap();
        assert_eq!(dt.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
    }

    #[test]
    fn test_builder_custom_options_propagated() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let opts =
            ConstructionOptions::default().with_insertion_order(InsertionOrderStrategy::Input);
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .construction_options(opts)
            .build::<()>()
            .unwrap();
        assert_eq!(dt.number_of_vertices(), 3);
    }

    // -------------------------------------------------------------------------
    // Toroidal path
    // -------------------------------------------------------------------------

    /// Vertices outside [0, 1)² must be canonicalized into the domain.
    /// Verified by inspecting each vertex coordinate in the built triangulation.
    #[test]
    fn test_builder_toroidal_canonicalizes_out_of_domain_vertices() {
        let vertices = vec![
            vertex!([0.2, 0.3]),  // in domain
            vertex!([1.8, 0.1]),  // x → 0.8
            vertex!([0.5, 0.7]),  // in domain
            vertex!([-0.4, 0.9]), // x → 0.6
        ];
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .toroidal([1.0, 1.0])
            .build::<()>()
            .unwrap();

        // Every vertex coordinate must lie within [0, 1) × [0, 1)
        for (_, v) in dt.vertices() {
            let c = v.point().coords();
            assert!(c[0] >= 0.0 && c[0] < 1.0, "x = {} not in [0, 1)", c[0]);
            assert!(c[1] >= 0.0 && c[1] < 1.0, "y = {} not in [0, 1)", c[1]);
        }
        assert_eq!(dt.number_of_vertices(), 4);
    }

    /// In-domain vertices should be unchanged by toroidal wrapping.
    #[test]
    fn test_builder_toroidal_in_domain_vertices_unchanged() {
        let vertices = vec![
            vertex!([0.1, 0.2]),
            vertex!([0.8, 0.3]),
            vertex!([0.4, 0.9]),
        ];
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .toroidal([1.0, 1.0])
            .build::<()>()
            .unwrap();

        for (_, v) in dt.vertices() {
            let c = v.point().coords();
            assert!(c[0] >= 0.0 && c[0] < 1.0);
            assert!(c[1] >= 0.0 && c[1] < 1.0);
        }
    }

    #[test]
    fn test_builder_toroidal_build_succeeds_2d() {
        let vertices = vec![
            vertex!([0.2, 0.3]),
            vertex!([0.8, 0.1]),
            vertex!([0.5, 0.7]),
            vertex!([0.1, 0.9]),
        ];
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .toroidal([1.0, 1.0])
            .build::<()>()
            .unwrap();
        assert_eq!(dt.number_of_vertices(), 4);
        assert_eq!(dt.dim(), 2);
        assert!(dt.as_triangulation().validate().is_ok());
        assert!(matches!(
            dt.global_topology(),
            GlobalTopology::Toroidal {
                mode: ToroidalConstructionMode::Canonicalized,
                ..
            }
        ));
    }

    #[test]
    fn test_builder_toroidal_build_out_of_domain_input_2d() {
        let vertices = vec![
            vertex!([2.2, 3.3]),  // → (0.2, 0.3)
            vertex!([-0.2, 1.1]), // → (0.8, 0.1)
            vertex!([1.5, 0.7]),  // → (0.5, 0.7)
            vertex!([-0.9, 2.9]), // → (0.1, 0.9)
        ];
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .toroidal([1.0, 1.0])
            .build::<()>()
            .unwrap();
        assert_eq!(dt.number_of_vertices(), 4);
        assert_eq!(dt.dim(), 2);
        assert!(dt.as_triangulation().validate().is_ok());
    }

    /// A non-finite (NaN) coordinate must cause the toroidal build to return an error.
    /// We create the vertex via `VertexBuilder` + `Point::new` to bypass `try_from`
    /// validation, which would otherwise panic.
    #[test]
    fn test_builder_toroidal_non_finite_coordinate_is_error() {
        let vertices = vec![
            VertexBuilder::<f64, (), 2>::default()
                .point(Point::new([0.2_f64, 0.3]))
                .build()
                .unwrap(),
            VertexBuilder::<f64, (), 2>::default()
                .point(Point::new([f64::NAN, 0.1]))
                .build()
                .unwrap(),
            VertexBuilder::<f64, (), 2>::default()
                .point(Point::new([0.5_f64, 0.7]))
                .build()
                .unwrap(),
        ];
        let result = DelaunayTriangulationBuilder::new(&vertices)
            .toroidal([1.0, 1.0])
            .build::<()>();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_toroidal_invalid_domain_is_error() {
        let vertices = vec![
            vertex!([0.2, 0.3]),
            vertex!([0.8, 0.1]),
            vertex!([0.5, 0.7]),
        ];
        let result = DelaunayTriangulationBuilder::new(&vertices)
            .toroidal([0.0, 1.0])
            .build::<()>();
        let err = result.expect_err("zero period should be rejected");
        assert!(format!("{err}").contains("Invalid toroidal domain"));
    }

    #[test]
    fn test_builder_toroidal_periodic_invalid_domain_is_error() {
        let vertices = vec![
            vertex!([0.1, 0.2]),
            vertex!([0.4, 0.7]),
            vertex!([0.7, 0.3]),
            vertex!([0.2, 0.9]),
            vertex!([0.8, 0.6]),
            vertex!([0.5, 0.1]),
            vertex!([0.3, 0.5]),
        ];
        let result = DelaunayTriangulationBuilder::new(&vertices)
            .toroidal_periodic([1.0, 0.0])
            .build::<()>();
        let err = result.expect_err("zero period should be rejected");
        assert!(format!("{err}").contains("Invalid toroidal domain"));
    }

    #[test]
    fn test_builder_toroidal_periodic_2d_smoke() {
        let vertices = vec![
            vertex!([0.1_f64, 0.2]),
            vertex!([0.4, 0.7]),
            vertex!([0.7, 0.3]),
            vertex!([0.2, 0.9]),
            vertex!([0.8, 0.6]),
            vertex!([0.5, 0.1]),
            vertex!([0.3, 0.5]),
        ];
        let n = vertices.len();
        let kernel = RobustKernel::new();
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .toroidal_periodic([1.0, 1.0])
            .build_with_kernel::<_, ()>(&kernel)
            .unwrap();
        assert_eq!(dt.number_of_vertices(), n);
        assert!(dt.tds().is_valid().is_ok());
        assert!(matches!(
            dt.global_topology(),
            GlobalTopology::Toroidal {
                mode: ToroidalConstructionMode::PeriodicImagePoint,
                ..
            }
        ));
    }

    #[test]
    fn test_builder_toroidal_idempotent_on_canonical_input() {
        let vertices = vec![
            vertex!([0.1, 0.2]),
            vertex!([0.8, 0.3]),
            vertex!([0.4, 0.9]),
        ];
        let dt_euclidean = DelaunayTriangulationBuilder::new(&vertices)
            .build::<()>()
            .unwrap();
        let dt_toroidal = DelaunayTriangulationBuilder::new(&vertices)
            .toroidal([1.0, 1.0])
            .build::<()>()
            .unwrap();
        assert_eq!(
            dt_euclidean.number_of_vertices(),
            dt_toroidal.number_of_vertices()
        );
        assert_eq!(
            dt_euclidean.number_of_simplices(),
            dt_toroidal.number_of_simplices()
        );
    }

    // -------------------------------------------------------------------------
    // Generic path (from_vertices)
    // -------------------------------------------------------------------------

    /// `from_vertices` is required when vertices carry user data (`U ≠ ()`).
    /// Verify that the data is preserved after toroidal canonicalization.
    #[test]
    fn test_builder_from_vertices_preserves_vertex_data() {
        let vertices: Vec<Vertex<f64, i32, 2>> = vec![
            VertexBuilder::default()
                .point(Point::new([0.2_f64, 0.3]))
                .data(1_i32)
                .build()
                .unwrap(),
            VertexBuilder::default()
                .point(Point::new([1.8_f64, 0.1])) // x → 0.8
                .data(2_i32)
                .build()
                .unwrap(),
            VertexBuilder::default()
                .point(Point::new([0.5_f64, 0.7]))
                .data(3_i32)
                .build()
                .unwrap(),
        ];
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .toroidal([1.0, 1.0])
            .build::<()>()
            .unwrap();

        assert_eq!(dt.number_of_vertices(), 3);

        // All coordinates must be in [0, 1) × [0, 1)
        for (_, v) in dt.vertices() {
            let c = v.point().coords();
            assert!(c[0] >= 0.0 && c[0] < 1.0);
            assert!(c[1] >= 0.0 && c[1] < 1.0);
        }

        // All three user-data values must survive the wrap
        let mut data: Vec<i32> = dt.vertices().filter_map(|(_, v)| v.data).collect();
        data.sort_unstable();
        assert_eq!(data, vec![1, 2, 3]);
    }

    #[test]
    fn test_builder_with_robust_kernel() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let kernel = RobustKernel::<f64>::new();
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .build_with_kernel::<_, ()>(&kernel)
            .unwrap();
        assert_eq!(dt.number_of_vertices(), 4);
        assert!(dt.validate().is_ok());
    }

    // -------------------------------------------------------------------------
    // Private helper function tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_topology_model_accepts_valid_toroidal() {
        let model = ToroidalModel::<2>::new([2.0, 3.0], ToroidalConstructionMode::Canonicalized);
        let result = DelaunayTriangulationBuilder::<f64, (), 2>::validate_topology_model(&model);
        assert!(result.is_ok());
    }

    #[test]
    fn test_derive_periodic_facet_key_happy_path_matches_core_derivation() {
        let lifted_ordered = vec![
            (VertexKey::null(), [0_i8, 0_i8]),
            (VertexKey::null(), [1_i8, 0_i8]),
            (VertexKey::null(), [0_i8, 1_i8]),
        ];
        let expected = periodic_facet_key_from_lifted_vertices::<2>(&lifted_ordered, 1).unwrap();
        let derived = DelaunayTriangulationBuilder::<f64, (), 2>::derive_periodic_facet_key(
            &lifted_ordered,
            1,
        )
        .unwrap();
        assert_eq!(derived, expected);
    }

    #[test]
    fn test_derive_periodic_facet_key_maps_errors_to_geometric_degeneracy() {
        let lifted_ordered = vec![
            (VertexKey::null(), [-128_i8, 0_i8]),
            (VertexKey::null(), [0_i8, 0_i8]),
            (VertexKey::null(), [127_i8, 0_i8]),
        ];
        let err = DelaunayTriangulationBuilder::<f64, (), 2>::derive_periodic_facet_key(
            &lifted_ordered,
            1,
        )
        .unwrap_err();
        match err {
            DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::GeometricDegeneracy { message },
            ) => {
                assert!(
                    message.contains(
                        "Failed to derive periodic candidate facet signature for index 1"
                    ),
                    "unexpected message prefix: {message}"
                );
                assert!(
                    message.contains("out of encodable range"),
                    "expected wrapped derivation detail in message: {message}"
                );
            }
            other => panic!("expected GeometricDegeneracy mapping, got: {other:?}"),
        }
    }

    #[test]
    fn test_validate_topology_model_rejects_zero_period() {
        let model = ToroidalModel::<2>::new([0.0, 3.0], ToroidalConstructionMode::Canonicalized);
        let result = DelaunayTriangulationBuilder::<f64, (), 2>::validate_topology_model(&model);
        assert!(result.is_err());
        let err_str = format!("{}", result.unwrap_err());
        assert!(
            err_str.contains("Invalid toroidal domain"),
            "Error message should mention invalid toroidal domain: {err_str}"
        );
        assert!(
            err_str.contains("axis 0"),
            "Error message should mention axis: {err_str}"
        );
    }

    #[test]
    fn test_validate_topology_model_rejects_negative_period() {
        let model =
            ToroidalModel::<3>::new([2.0, -1.0, 3.0], ToroidalConstructionMode::Canonicalized);
        let result = DelaunayTriangulationBuilder::<f64, (), 3>::validate_topology_model(&model);
        assert!(result.is_err());
        let err_str = format!("{}", result.unwrap_err());
        assert!(err_str.contains("Invalid toroidal domain"));
        assert!(err_str.contains("axis 1"));
    }

    #[test]
    fn test_validate_topology_model_rejects_infinite_period() {
        let model = ToroidalModel::<2>::new(
            [f64::INFINITY, 3.0],
            ToroidalConstructionMode::Canonicalized,
        );
        let result = DelaunayTriangulationBuilder::<f64, (), 2>::validate_topology_model(&model);
        assert!(result.is_err());
        let err_str = format!("{}", result.unwrap_err());
        assert!(err_str.contains("Invalid toroidal domain"));
    }

    #[test]
    fn test_validate_topology_model_rejects_nan_period() {
        let model =
            ToroidalModel::<2>::new([f64::NAN, 3.0], ToroidalConstructionMode::Canonicalized);
        let result = DelaunayTriangulationBuilder::<f64, (), 2>::validate_topology_model(&model);
        assert!(result.is_err());
        let err_str = format!("{}", result.unwrap_err());
        assert!(err_str.contains("Invalid toroidal domain"));
    }

    #[test]
    fn test_validate_topology_model_accepts_euclidean() {
        let model = EuclideanModel;
        let result = DelaunayTriangulationBuilder::<f64, (), 2>::validate_topology_model(&model);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_topology_model_maps_non_period_errors() {
        let result = DelaunayTriangulationBuilder::<f64, (), 2>::validate_topology_model(
            &ValidationFailureModel,
        );
        let err = result.expect_err("non-period validation failure should be mapped");
        let err_str = err.to_string();
        assert!(err_str.contains("Invalid topology model configuration"));
    }

    #[test]
    fn test_canonicalize_vertices_preserves_uuids() {
        let vertices = vec![
            vertex!([2.5, 3.7]),
            vertex!([1.8, -0.5]),
            vertex!([0.5, 0.7]),
        ];
        let original_uuids: Vec<_> = vertices.iter().map(Vertex::uuid).collect();
        let model = ToroidalModel::<2>::new([2.0, 3.0], ToroidalConstructionMode::Canonicalized);
        let canonical =
            DelaunayTriangulationBuilder::<f64, (), 2>::canonicalize_vertices(&vertices, &model)
                .unwrap();

        assert_eq!(canonical.len(), vertices.len());
        let canonical_uuids: Vec<_> = canonical.iter().map(Vertex::uuid).collect();
        assert_eq!(canonical_uuids, original_uuids);
    }

    #[test]
    fn test_canonicalize_vertices_preserves_data() {
        let vertices: Vec<Vertex<f64, i32, 2>> = vec![
            VertexBuilder::default()
                .point(Point::new([2.5_f64, 3.7]))
                .data(10_i32)
                .build()
                .unwrap(),
            VertexBuilder::default()
                .point(Point::new([1.8_f64, -0.5]))
                .data(20_i32)
                .build()
                .unwrap(),
            VertexBuilder::default()
                .point(Point::new([0.5_f64, 0.7]))
                .data(30_i32)
                .build()
                .unwrap(),
        ];
        let model = ToroidalModel::<2>::new([2.0, 3.0], ToroidalConstructionMode::Canonicalized);
        let canonical =
            DelaunayTriangulationBuilder::<f64, i32, 2>::canonicalize_vertices(&vertices, &model)
                .unwrap();

        assert_eq!(canonical.len(), vertices.len());
        for (orig, canon) in vertices.iter().zip(canonical.iter()) {
            assert_eq!(orig.data, canon.data);
        }
    }

    #[test]
    fn test_canonicalize_vertices_transforms_coordinates() {
        let vertices = vec![
            vertex!([2.5, 3.7]),  // → (0.5, 0.7)
            vertex!([1.8, -0.5]), // → (1.8, 2.5)
            vertex!([0.3, 0.2]),  // → (0.3, 0.2)
        ];
        let model = ToroidalModel::<2>::new([2.0, 3.0], ToroidalConstructionMode::Canonicalized);
        let canonical =
            DelaunayTriangulationBuilder::<f64, (), 2>::canonicalize_vertices(&vertices, &model)
                .unwrap();

        assert_eq!(canonical.len(), 3);
        assert_relative_eq!(canonical[0].point().coords()[0], 0.5);
        assert_relative_eq!(canonical[0].point().coords()[1], 0.7);
        assert_relative_eq!(canonical[1].point().coords()[0], 1.8);
        assert_relative_eq!(canonical[1].point().coords()[1], 2.5);
        assert_relative_eq!(canonical[2].point().coords()[0], 0.3);
        assert_relative_eq!(canonical[2].point().coords()[1], 0.2);
    }

    #[test]
    fn test_canonicalize_vertices_includes_vertex_context_on_error() {
        let vertices = vec![vertex!([0.25_f64, 0.75_f64]), vertex!([0.9_f64, 0.1_f64])];
        let result = DelaunayTriangulationBuilder::<f64, (), 2>::canonicalize_vertices(
            &vertices,
            &CanonicalizationFailureModel,
        );
        let err = result.expect_err("canonicalization failure should be reported");
        let err_str = err.to_string();
        assert!(err_str.contains("Failed to canonicalize vertex 0"));
        assert!(err_str.contains("reason"));
    }

    #[test]
    fn test_build_periodic_requires_periodic_domain() {
        let kernel = AdaptiveKernel::new();
        let canonical_vertices = vec![
            vertex!([0.1_f64, 0.1_f64]),
            vertex!([0.9_f64, 0.2_f64]),
            vertex!([0.2_f64, 0.8_f64]),
            vertex!([0.7_f64, 0.9_f64]),
            vertex!([0.5_f64, 0.4_f64]),
        ];
        let result = DelaunayTriangulationBuilder::<f64, (), 2>::build_periodic::<_, (), _>(
            &kernel,
            &canonical_vertices,
            &MissingPeriodicDomainModel,
            TopologyGuarantee::default(),
            ConstructionOptions::default(),
        );
        let err = result.expect_err("missing periodic domain must fail");
        assert!(err.to_string().contains(
            "does not expose a periodic domain required for periodic image-point construction"
        ));
    }

    #[test]
    fn test_canonicalize_vertices_euclidean_identity() {
        let vertices = vec![
            vertex!([1.5, 2.5]),
            vertex!([3.7, 4.2]),
            vertex!([-1.0, -2.0]),
        ];
        let model = EuclideanModel;
        let canonical =
            DelaunayTriangulationBuilder::<f64, (), 2>::canonicalize_vertices(&vertices, &model)
                .unwrap();

        assert_eq!(canonical.len(), vertices.len());
        for (orig, canon) in vertices.iter().zip(canonical.iter()) {
            assert_relative_eq!(orig.point().coords()[0], canon.point().coords()[0]);
            assert_relative_eq!(orig.point().coords()[1], canon.point().coords()[1]);
        }
    }

    #[test]
    fn test_canonicalize_vertices_propagates_nan_error() {
        let vertices = vec![
            VertexBuilder::default()
                .point(Point::new([0.5_f64, 0.5]))
                .build()
                .unwrap(),
            VertexBuilder::default()
                .point(Point::new([f64::NAN, 0.5]))
                .build()
                .unwrap(),
            VertexBuilder::default()
                .point(Point::new([0.3_f64, 0.2]))
                .build()
                .unwrap(),
        ];
        let model = ToroidalModel::<2>::new([2.0, 3.0], ToroidalConstructionMode::Canonicalized);
        let result =
            DelaunayTriangulationBuilder::<f64, (), 2>::canonicalize_vertices(&vertices, &model);

        assert!(result.is_err());
        let err_str = format!("{}", result.unwrap_err());
        assert!(
            err_str.contains("Failed to canonicalize vertex"),
            "Error should mention canonicalization failure: {err_str}"
        );
        assert!(
            err_str.contains("vertex 1"),
            "Error should mention vertex index: {err_str}"
        );
    }

    #[test]
    fn test_canonicalize_vertices_propagates_infinity_error() {
        let vertices = vec![
            VertexBuilder::default()
                .point(Point::new([0.5_f64, 0.5]))
                .build()
                .unwrap(),
            VertexBuilder::default()
                .point(Point::new([0.3_f64, 0.2]))
                .build()
                .unwrap(),
            VertexBuilder::default()
                .point(Point::new([f64::INFINITY, 0.5]))
                .build()
                .unwrap(),
        ];
        let model = ToroidalModel::<2>::new([2.0, 3.0], ToroidalConstructionMode::Canonicalized);
        let result =
            DelaunayTriangulationBuilder::<f64, (), 2>::canonicalize_vertices(&vertices, &model);

        assert!(result.is_err());
        let err_str = format!("{}", result.unwrap_err());
        assert!(err_str.contains("Failed to canonicalize vertex"));
        assert!(err_str.contains("vertex 2"));
    }

    #[test]
    fn test_canonicalize_vertices_includes_original_coords_in_error() {
        let vertices = vec![
            VertexBuilder::default()
                .point(Point::new([f64::NAN, 1.5_f64]))
                .build()
                .unwrap(),
        ];
        let model = ToroidalModel::<2>::new([2.0, 3.0], ToroidalConstructionMode::Canonicalized);
        let result =
            DelaunayTriangulationBuilder::<f64, (), 2>::canonicalize_vertices(&vertices, &model);

        assert!(result.is_err());
        let err_str = format!("{}", result.unwrap_err());
        assert!(
            err_str.contains("original coords"),
            "Error should mention original coords: {err_str}"
        );
    }
}
