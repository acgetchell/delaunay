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
//! dimension, building the full Euclidean DT on the expanded set, and extracting the
//! restriction to the fundamental domain with rewired periodic neighbor pointers.
//! Produces a true toroidal (χ = 0) triangulation.
//!
//! # Examples
//!
//! ## Standard Euclidean construction
//!
//! ```rust
//! use delaunay::core::builder::DelaunayTriangulationBuilder;
//! use delaunay::vertex;
//!
//! let vertices = vec![
//!     vertex!([0.0, 0.0]),
//!     vertex!([1.0, 0.0]),
//!     vertex!([0.0, 1.0]),
//! ];
//!
//! let dt = DelaunayTriangulationBuilder::new(&vertices)
//!     .build::<()>()
//!     .unwrap();
//!
//! assert_eq!(dt.number_of_vertices(), 3);
//! ```
//!
//! ## Toroidal construction (Phase 1: canonicalization only)
//!
//! ```rust
//! use delaunay::core::builder::DelaunayTriangulationBuilder;
//! use delaunay::vertex;
//!
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
//!     .build::<()>()
//!     .unwrap();
//!
//! assert_eq!(dt.number_of_vertices(), 4);
//! ```
//!
//! ## Toroidal construction (Phase 2: full periodic / image-point method)
//!
//! Uses the 3^D image-point method to produce a true toroidal (χ = 0) triangulation
//! where boundary facets are identified and neighbor pointers are rewired periodically.
//!
//! ```rust,no_run
//! use delaunay::core::builder::DelaunayTriangulationBuilder;
//! use delaunay::geometry::kernel::RobustKernel;
//! use delaunay::vertex;
//!
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
//!     .build_with_kernel::<_, ()>(&kernel)
//!     .unwrap();
//!
//! assert_eq!(dt.number_of_vertices(), 7);
//! // Every vertex has a valid incident cell (no boundary).
//! assert!(dt.tds().is_valid().is_ok());
//! ```

#![forbid(unsafe_code)]

use crate::core::cell::Cell;
use crate::core::collections::{FastHashMap, Uuid, VertexKeySet};
use crate::core::delaunay_triangulation::{
    ConstructionOptions, DelaunayRepairPolicy, DelaunayTriangulation,
    DelaunayTriangulationConstructionError, InitialSimplexStrategy, RetryPolicy,
};
use crate::core::operations::InsertionOutcome;
use crate::core::traits::data_type::DataType;
use crate::core::triangulation::{TopologyGuarantee, TriangulationConstructionError};
use crate::core::triangulation_data_structure::{CellKey, VertexKey};
use crate::core::util::periodic_facet_key_from_lifted_vertices;
use crate::core::vertex::{Vertex, VertexBuilder};
use crate::geometry::kernel::{FastKernel, Kernel};
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::{Coordinate, CoordinateScalar, ScalarAccumulative};
use crate::topology::spaces::toroidal::ToroidalSpace;
use crate::topology::traits::global_topology_model::{
    GlobalTopologyModel, GlobalTopologyModelError,
};
use crate::topology::traits::topological_space::{GlobalTopology, ToroidalConstructionMode};
use num_traits::ToPrimitive;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::num::NonZeroUsize;
const TWO_POW_52_I64: i64 = 4_503_599_627_370_496; // 2^52
const TWO_POW_52_F64: f64 = 4_503_599_627_370_496.0; // 2^52
const MAX_OFFSET_UNITS: i64 = 1_048_576;
const IMAGE_JITTER_UNITS: i64 = 64;
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;
type LiftedVertex<const D: usize> = (
    crate::core::triangulation_data_structure::VertexKey,
    [i8; D],
);
type SymbolicSignature<const D: usize> = Vec<LiftedVertex<D>>;
type PeriodicFacetKey = u64;
type PeriodicCandidate<const D: usize> = (
    SymbolicSignature<D>,
    SymbolicSignature<D>,
    Vec<PeriodicFacetKey>,
    bool,
);
/// Finds a bounded-size 2D face subset whose edge incidences can close a quotient boundary.
///
/// Returns a boolean mask aligned with `candidate_edges` when a selection of exactly
/// `target_faces` candidates is found such that no edge is used more than twice. The search
/// uses a DFS with pruning and a heuristic ordering that prefers in-domain candidates first.
#[expect(
    clippy::too_many_lines,
    reason = "Depth-first bounded subset search includes pruning logic and is kept self-contained"
)]
#[expect(
    clippy::items_after_statements,
    reason = "Local DFS helper is intentionally colocated with selection setup"
)]
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

    // Frequency of each edge in the candidate pool (rarer edges first).
    let mut edge_frequency = vec![0usize; edge_count];
    for edges in candidate_edges {
        for &edge in edges {
            edge_frequency[edge] = edge_frequency[edge].saturating_add(1);
        }
    }

    let mut order: Vec<usize> = (0..m).collect();
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

    let mut edge_counts = vec![0u8; edge_count];
    let mut selected = vec![false; m];
    let mut nodes = 0usize;

    #[expect(
        clippy::too_many_arguments,
        reason = "Recursive DFS state requires explicit parameterization for pruning"
    )]
    fn dfs(
        pos: usize,
        chosen: usize,
        target_faces: usize,
        order: &[usize],
        candidate_edges: &[[usize; 3]],
        edge_counts: &mut [u8],
        selected: &mut [bool],
        nodes: &mut usize,
        node_limit: usize,
    ) -> bool {
        if chosen == target_faces {
            return true;
        }
        if pos == order.len() {
            return false;
        }
        if chosen + (order.len() - pos) < target_faces {
            return false;
        }
        if *nodes >= node_limit {
            return false;
        }
        *nodes = nodes.saturating_add(1);

        // Capacity-based prune: each additional face consumes 3 remaining edge incidences.
        let remaining_capacity: usize = edge_counts
            .iter()
            .map(|&count| usize::from(2_u8.saturating_sub(count)))
            .sum();
        if chosen + (remaining_capacity / 3) < target_faces {
            return false;
        }

        let idx = order[pos];
        let edges = candidate_edges[idx];

        if edge_counts[edges[0]] < 2 && edge_counts[edges[1]] < 2 && edge_counts[edges[2]] < 2 {
            selected[idx] = true;
            edge_counts[edges[0]] += 1;
            edge_counts[edges[1]] += 1;
            edge_counts[edges[2]] += 1;

            if dfs(
                pos + 1,
                chosen + 1,
                target_faces,
                order,
                candidate_edges,
                edge_counts,
                selected,
                nodes,
                node_limit,
            ) {
                return true;
            }

            edge_counts[edges[0]] -= 1;
            edge_counts[edges[1]] -= 1;
            edge_counts[edges[2]] -= 1;
            selected[idx] = false;
        }

        dfs(
            pos + 1,
            chosen,
            target_faces,
            order,
            candidate_edges,
            edge_counts,
            selected,
            nodes,
            node_limit,
        )
    }

    dfs(
        0,
        0,
        target_faces,
        &order,
        candidate_edges,
        &mut edge_counts,
        &mut selected,
        &mut nodes,
        node_limit,
    )
    .then_some(selected)
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
/// The cell data type `V` and kernel `K` are deferred to the
/// [`build`](Self::build) / [`build_with_kernel`](Self::build_with_kernel)
/// call, keeping the builder type signature concise.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::builder::DelaunayTriangulationBuilder;
/// use delaunay::core::delaunay_triangulation::ConstructionOptions;
/// use delaunay::core::triangulation::TopologyGuarantee;
/// use delaunay::vertex;
///
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
///     .build::<()>()
///     .unwrap();
///
/// assert_eq!(dt.number_of_vertices(), 4);
/// ```
pub struct DelaunayTriangulationBuilder<'v, T, U, const D: usize>
where
    T: CoordinateScalar,
    U: DataType,
{
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
}

// =============================================================================
// SPECIALIZED IMPL — f64 coordinates, no vertex data (common case)
//
// Having `new` here (rather than in the generic impl below) pins T=f64 and
// U=() so callers never need explicit type annotations:
//
//   let vertices = vec![vertex!([0.0, 0.0]), ...];
//   let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>();
//
// This mirrors the existing `DelaunayTriangulation::new` design.
// =============================================================================

impl<'v, const D: usize> DelaunayTriangulationBuilder<'v, f64, (), D> {
    /// Creates a builder for `f64` vertices with no user data — the most common case.
    ///
    /// Type parameters are fully inferred from the input; no explicit annotations are needed.
    /// For non-`f64` scalars or vertices carrying user data, use
    /// [`from_vertices`](Self::from_vertices) in the generic impl.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::builder::DelaunayTriangulationBuilder;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![vertex!([0.0, 0.0]), vertex!([1.0, 0.0]), vertex!([0.0, 1.0])];
    /// let _builder = DelaunayTriangulationBuilder::new(&vertices);
    /// ```
    #[must_use]
    pub fn new(vertices: &'v [Vertex<f64, (), D>]) -> Self {
        Self {
            vertices,
            topology: None,
            topology_guarantee: TopologyGuarantee::DEFAULT,
            construction_options: ConstructionOptions::default(),
            use_image_point_method: false,
        }
    }
}

// =============================================================================
// GENERIC IMPL — any scalar T, any vertex data U
// =============================================================================

impl<'v, T, U, const D: usize> DelaunayTriangulationBuilder<'v, T, U, D>
where
    T: CoordinateScalar,
    U: DataType,
{
    /// Creates a builder from a vertex slice of any scalar type `T` or user data type `U`.
    ///
    /// For the most common case — `f64` coordinates, `()` vertex data — prefer
    /// [`new`](DelaunayTriangulationBuilder::new), which requires no type annotations.
    /// Use `from_vertices` when `T ≠ f64` (e.g. `f32`) or `U ≠ ()` (vertices carry data).
    ///
    /// This mirrors the relationship between [`DelaunayTriangulation::new`] (specialized)
    /// and [`DelaunayTriangulation::with_kernel`] (generic).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::builder::DelaunayTriangulationBuilder;
    /// use delaunay::core::vertex::{Vertex, VertexBuilder};
    /// use delaunay::geometry::point::Point;
    /// use delaunay::geometry::traits::coordinate::Coordinate;
    ///
    /// // Vertices with attached user data — requires from_vertices.
    /// let vertices: Vec<Vertex<f64, i32, 2>> = vec![
    ///     VertexBuilder::default().point(Point::new([0.0, 0.0])).data(1_i32).build().unwrap(),
    ///     VertexBuilder::default().point(Point::new([1.0, 0.0])).data(2_i32).build().unwrap(),
    ///     VertexBuilder::default().point(Point::new([0.0, 1.0])).data(3_i32).build().unwrap(),
    /// ];
    ///
    /// let dt = DelaunayTriangulationBuilder::from_vertices(&vertices)
    ///     .build::<()>()
    ///     .unwrap();
    ///
    /// assert_eq!(dt.number_of_vertices(), 3);
    /// ```
    #[must_use]
    pub fn from_vertices(vertices: &'v [Vertex<T, U, D>]) -> Self {
        Self {
            vertices,
            topology: None,
            topology_guarantee: TopologyGuarantee::DEFAULT,
            construction_options: ConstructionOptions::default(),
            use_image_point_method: false,
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
    /// use delaunay::core::builder::DelaunayTriangulationBuilder;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.2, 0.3]),
    ///     vertex!([0.8, 0.1]),
    ///     vertex!([0.5, 0.7]),
    ///     vertex!([0.1, 0.9]),
    /// ];
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .toroidal([1.0, 1.0])
    ///     .build::<()>()
    ///     .unwrap();
    ///
    /// assert_eq!(dt.number_of_vertices(), 4);
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
    /// **Use [`RobustKernel`](crate::geometry::kernel::RobustKernel) or
    /// [`build_with_kernel`](Self::build_with_kernel)** for reliable results; numerical
    /// near-degeneracies in the expanded set can cause construction failures with
    /// `FastKernel`.
    ///
    /// # Arguments
    ///
    /// * `domain` — Period length `[L_0, …, L_{D-1}]` for each dimension.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use delaunay::core::builder::DelaunayTriangulationBuilder;
    /// use delaunay::geometry::kernel::RobustKernel;
    /// use delaunay::vertex;
    ///
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
    ///     .build_with_kernel::<_, ()>(&kernel)
    ///     .unwrap();
    ///
    /// assert_eq!(dt.number_of_vertices(), 7);
    /// assert!(dt.tds().is_valid().is_ok());
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
    /// use delaunay::core::builder::DelaunayTriangulationBuilder;
    /// use delaunay::core::triangulation::TopologyGuarantee;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .topology_guarantee(TopologyGuarantee::Pseudomanifold)
    ///     .build::<()>()
    ///     .unwrap();
    ///
    /// assert_eq!(dt.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
    /// ```
    #[must_use]
    pub const fn topology_guarantee(mut self, topology_guarantee: TopologyGuarantee) -> Self {
        self.topology_guarantee = topology_guarantee;
        self
    }

    /// Sets the [`ConstructionOptions`] (insertion order, deduplication, retry policy).
    ///
    /// Defaults to [`ConstructionOptions::default`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::builder::DelaunayTriangulationBuilder;
    /// use delaunay::core::delaunay_triangulation::{ConstructionOptions, InsertionOrderStrategy};
    /// use delaunay::vertex;
    ///
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
    ///     .build::<()>()
    ///     .unwrap();
    ///
    /// assert_eq!(dt.number_of_vertices(), 3);
    /// ```
    #[must_use]
    pub const fn construction_options(mut self, construction_options: ConstructionOptions) -> Self {
        self.construction_options = construction_options;
        self
    }

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

    /// Builds the triangulation using [`FastKernel<T>`].
    ///
    /// This is the most common build path. Cell data type `V` is inferred or
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
    /// use delaunay::core::builder::DelaunayTriangulationBuilder;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .build::<()>()
    ///     .unwrap();
    ///
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// assert!(dt.validate().is_ok());
    /// ```
    pub fn build<V>(
        self,
    ) -> Result<DelaunayTriangulation<FastKernel<T>, U, V, D>, DelaunayTriangulationConstructionError>
    where
        T: ScalarAccumulative,
        V: DataType,
    {
        self.build_with_kernel(&FastKernel::new())
    }

    /// Builds the triangulation using a caller-supplied kernel.
    ///
    /// Use this when you need [`RobustKernel`](crate::geometry::kernel::RobustKernel) or
    /// a custom kernel implementation.
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayTriangulationConstructionError`] if canonicalization or
    /// construction fails (see [`build`](Self::build) for details).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::builder::DelaunayTriangulationBuilder;
    /// use delaunay::geometry::kernel::RobustKernel;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let kernel = RobustKernel::new();
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .build_with_kernel::<_, ()>(&kernel)
    ///     .unwrap();
    ///
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// ```
    pub fn build_with_kernel<K, V>(
        self,
        kernel: &K,
    ) -> Result<DelaunayTriangulation<K, U, V, D>, DelaunayTriangulationConstructionError>
    where
        K: Kernel<D, Scalar = T>,
        K::Scalar: ScalarAccumulative,
        V: DataType,
    {
        match (self.topology, self.use_image_point_method) {
            (None, _) => {
                // Euclidean path: delegate directly.
                DelaunayTriangulation::with_topology_guarantee_and_options(
                    kernel,
                    self.vertices,
                    self.topology_guarantee,
                    self.construction_options,
                )
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
                dt.as_triangulation_mut()
                    .normalize_and_promote_positive_orientation()
                    .map_err(|e| TriangulationConstructionError::GeometricDegeneracy {
                        message: format!(
                            "Failed to canonicalize periodic orientation after build: {e}",
                        ),
                    })?;
                dt.as_triangulation()
                    .validate_geometric_cell_orientation()
                    .map_err(|e| TriangulationConstructionError::GeometricDegeneracy {
                        message: format!(
                            "Periodic geometric orientation validation failed after build: {e}",
                        ),
                    })?;
                Ok(dt)
            }
        }
    }

    /// Builds a true periodic (toroidal) Delaunay triangulation using the 3^D image-point method.
    ///
    /// **Algorithm** (see module-level doc for Phase 2 details):
    /// 1. Validate: at least `2*D + 1` canonical vertices required.
    /// 2. Build 3^D-1 image copies of each vertex, shifted by `{-L_i, 0, +L_i}` per axis.
    ///    Every copy of canonical vertex `v_i` (including the zero-offset canonical copy)
    ///    receives the **same** tiny deterministic per-vertex perturbation `δ_i`.
    /// 3. Build a full Euclidean DT on the expanded set (n * 3^D points).
    /// 4. Extract the "central" sub-complex: cells with all vertices in the canonical set.
    /// 5. Rewire boundary facets: for each facet of a central cell adjacent to a non-central cell,
    ///    find the corresponding central neighbor and update the neighbor pointer.
    /// 6. Rebuild incident-cell associations and return the result.
    ///
    /// The output is a `Tds` whose `is_valid()` passes at Level 2 (structural validity).
    ///
    /// # References
    ///
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
        K::Scalar: ScalarAccumulative,
        V: DataType,
        M: GlobalTopologyModel<D>,
    {
        // Keep `build_periodic` self-protecting even if future call paths bypass outer validation.
        Self::validate_topology_model(topology_model)?;

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
                    let image_v: Vertex<T, U, D> = VertexBuilder::default()
                        .point(new_point)
                        .build()
                        .expect("image vertex with valid coords always builds");
                    image_uuid_to_canonical_with_offset.insert(image_v.uuid(), (v.uuid(), offset));
                    expanded.push(image_v);
                }
            }
        }
        let expanded_base_options =
            construction_options.with_initial_simplex_strategy(InitialSimplexStrategy::Balanced);
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
                            && candidate_dt.number_of_cells() > 0
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
        let central_key_set: VertexKeySet = canonical_uuids
            .iter()
            .filter_map(|uuid| tds_ref.vertex_key_from_uuid(uuid))
            .collect();

        // Map every full-DT vertex key to its canonical key and lattice offset.
        let mut vertex_key_to_lifted: FastHashMap<VertexKey, (VertexKey, [i8; D])> =
            FastHashMap::default();
        for vk in tds_ref.vertex_keys() {
            let Some(vertex) = tds_ref.get_vertex_by_key(vk) else {
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

        let normalize_cell_lifted = |cell_key: CellKey| -> Option<Vec<(VertexKey, [i8; D])>> {
            let cell = tds_ref.get_cell(cell_key)?;
            let mut lifted: Vec<(VertexKey, [i8; D])> = cell
                .vertices()
                .iter()
                .map(|vk| vertex_key_to_lifted.get(vk).copied())
                .collect::<Option<Vec<_>>>()?;

            let mut canonical_keys: Vec<VertexKey> = lifted.iter().map(|(ck, _)| *ck).collect();
            canonical_keys.sort_unstable();
            canonical_keys.dedup();
            if canonical_keys.len() != D + 1 {
                // Cell collapses in the quotient (repeated canonical vertex); skip it.
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
        let cell_barycenter_in_fundamental_domain = |cell_key: CellKey| -> Option<bool> {
            let cell = tds_ref.get_cell(cell_key)?;
            let mut sums = [0.0_f64; D];
            for vk in cell.vertices() {
                let vertex = tds_ref.get_vertex_by_key(*vk)?;
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

        // Build unique symbolic candidates from all full-DT cells.
        // Candidate tuple layout (see type alias):
        // (symbolic_signature, lifted_ordered, periodic_facet_keys, in_domain_hint)
        // where `lifted_ordered` preserves the observed per-cell vertex order from
        // `normalize_cell_lifted` (it is not canonical-key-sorted).
        let mut candidates_by_symbolic: FastHashMap<SymbolicSignature<D>, PeriodicCandidate<D>> =
            FastHashMap::default();
        for ck in tds_ref.cell_keys() {
            let Some(lifted_vertices) = normalize_cell_lifted(ck) else {
                continue;
            };
            let in_domain = cell_barycenter_in_fundamental_domain(ck).unwrap_or(false);
            let mut symbolic_signature = lifted_vertices.clone();
            symbolic_signature.sort_unstable();
            let lifted_ordered = lifted_vertices.clone();
            let mut periodic_facets: Vec<PeriodicFacetKey> = Vec::with_capacity(D + 1);
            for facet_idx in 0..=D {
                periodic_facets.push(
                    periodic_facet_key_from_lifted_vertices::<D>(&lifted_ordered, facet_idx)
                        .map_err(|error| {
                            TriangulationConstructionError::GeometricDegeneracy {
                                message: format!(
                                    "Failed to derive periodic candidate facet signature for index {facet_idx}: {error}",
                                ),
                            }
                        })?,
                );
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
                message: "No quotient periodic cells found in full image DT".to_owned(),
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

                // Pass 2: only add cells that strictly reduce boundary facets (count == 1).
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
                    let f_count = i64::try_from(selected_count).expect("cell count fits in i64");
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
                message: "Periodic quotient selection failed to pick any candidate cells"
                    .to_owned(),
            }
            .into());
        }
        if D == 2 && best_boundary_count > 0 {
            return Err(TriangulationConstructionError::GeometricDegeneracy {
                message: format!(
                    "Periodic quotient selection left {best_boundary_count} boundary facets after {search_attempts} attempts (full_vertices={}, full_cells={}, canonical_vertices={}, candidates={}, selected_cells={})",
                    tds_ref.number_of_vertices(),
                    tds_ref.number_of_cells(),
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
            // In the quotient TDS, cells that collapse to the same canonical vertex set cannot
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

        // Clone TDS and rebuild cell complex from quotient representatives.
        let mut tds_mut = tds_ref.clone();

        // Remove all cells first.
        let all_cells: Vec<CellKey> = tds_mut.cell_keys().collect();
        tds_mut.remove_cells_by_keys(&all_cells);

        // Remove all image vertices (collect keys first, then copies, then remove).
        let image_vertex_keys: Vec<VertexKey> = tds_mut
            .vertex_keys()
            .filter(|vk| !central_key_set.contains(vk))
            .collect();
        let image_vertex_copies: Vec<Vertex<T, U, D>> = image_vertex_keys
            .iter()
            .filter_map(|&vk| tds_mut.get_vertex_by_key(vk).copied())
            .collect();
        for iv in &image_vertex_copies {
            tds_mut.remove_vertex(iv).map_err(|e| {
                TriangulationConstructionError::GeometricDegeneracy {
                    message: format!("Failed to remove image vertex: {e}"),
                }
            })?;
        }

        // Insert quotient cells.
        let mut signatures_sorted: Vec<Vec<(VertexKey, [i8; D])>> =
            representative_lifted_by_symbolic.keys().cloned().collect();
        signatures_sorted.sort_unstable();

        let mut inserted_cell_keys: Vec<CellKey> = Vec::with_capacity(signatures_sorted.len());
        let mut rep_lifted_by_key: FastHashMap<CellKey, Vec<(VertexKey, [i8; D])>> =
            FastHashMap::default();

        for signature in signatures_sorted {
            let Some(lifted_vertices) = representative_lifted_by_symbolic.get(&signature) else {
                continue;
            };
            let canonical_vertex_keys: Vec<VertexKey> =
                lifted_vertices.iter().map(|(ck, _)| *ck).collect();
            let mut cell = Cell::new(canonical_vertex_keys, None).map_err(|e| {
                TriangulationConstructionError::GeometricDegeneracy {
                    message: format!("Failed to create quotient periodic cell: {e}"),
                }
            })?;
            cell.set_periodic_vertex_offsets(
                lifted_vertices
                    .iter()
                    .map(|(_, offset)| *offset)
                    .collect::<Vec<_>>(),
            );
            let ck = tds_mut.insert_cell_with_mapping(cell).map_err(|e| {
                TriangulationConstructionError::GeometricDegeneracy {
                    message: format!("Failed to insert quotient periodic cell: {e}"),
                }
            })?;
            inserted_cell_keys.push(ck);
            rep_lifted_by_key.insert(ck, lifted_vertices.clone());
        }
        if inserted_cell_keys.is_empty() {
            return Err(TriangulationConstructionError::GeometricDegeneracy {
                message: "No cells survived periodic quotient reconstruction".to_owned(),
            }
            .into());
        }

        // Sanity-check periodic facet multiplicities before neighbor rewiring.
        // In a valid simplicial manifold each facet is incident to at most two cells.
        let mut periodic_facet_counts: FastHashMap<PeriodicFacetKey, usize> =
            FastHashMap::default();
        for lifted in rep_lifted_by_key.values() {
            for facet_idx in 0..=D {
                let periodic_facet_key =
                    periodic_facet_key_from_lifted_vertices::<D>(lifted, facet_idx).map_err(
                        |error| TriangulationConstructionError::GeometricDegeneracy {
                            message: format!(
                                "Failed to derive periodic multiplicity facet signature for index {facet_idx}: {error}",
                            ),
                        },
                    )?;
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
                    "Periodic quotient selection overcounts periodic facets ({} overloaded); selected_cells={}, sample={:?}",
                    overloaded_facets.len(),
                    rep_lifted_by_key.len(),
                    overloaded_facets.iter().take(4).collect::<Vec<_>>(),
                ),
            }
            .into());
        }

        // Rebuild neighbor pointers by pairing equal symbolic facet signatures in the quotient.
        let mut neighbor_updates: FastHashMap<CellKey, Vec<Option<CellKey>>> = inserted_cell_keys
            .iter()
            .copied()
            .map(|ck| (ck, vec![None; D + 1]))
            .collect();

        let mut facet_occurrences: FastHashMap<PeriodicFacetKey, Vec<(CellKey, usize)>> =
            FastHashMap::default();
        for &rep_ck in &inserted_cell_keys {
            let Some(lifted) = rep_lifted_by_key.get(&rep_ck) else {
                continue;
            };
            for facet_idx in 0..=D {
                let sig =
                    periodic_facet_key_from_lifted_vertices::<D>(lifted, facet_idx).map_err(
                        |error| TriangulationConstructionError::GeometricDegeneracy {
                            message: format!(
                                "Failed to derive periodic neighbor facet signature for cell {rep_ck:?} facet {facet_idx}: {error}",
                            ),
                        },
                    )?;
                facet_occurrences
                    .entry(sig)
                    .or_default()
                    .push((rep_ck, facet_idx));
            }
        }

        for (_facet_sig, occurrences) in facet_occurrences {
            match occurrences.as_slice() {
                [(a_ck, a_idx), (b_ck, b_idx)] => {
                    let a_lifted = rep_lifted_by_key
                        .get(a_ck)
                        .expect("lifted representative exists for quotient cell");
                    let b_lifted = rep_lifted_by_key
                        .get(b_ck)
                        .expect("lifted representative exists for quotient cell");
                    let shares_all_canonical_vertices = a_lifted
                        .iter()
                        .zip(b_lifted.iter())
                        .all(|((a_vk, _), (b_vk, _))| a_vk == b_vk);

                    if shares_all_canonical_vertices {
                        return Err(TriangulationConstructionError::GeometricDegeneracy {
                            message: format!(
                                "Periodic quotient produced distinct cells with identical canonical vertices across a shared facet: {a_ck:?}[{a_idx}] <-> {b_ck:?}[{b_idx}]",
                            ),
                        }
                        .into());
                    }
                    neighbor_updates
                        .get_mut(a_ck)
                        .expect("neighbor vector exists for quotient cell")[*a_idx] = Some(*b_ck);
                    neighbor_updates
                        .get_mut(b_ck)
                        .expect("neighbor vector exists for quotient cell")[*b_idx] = Some(*a_ck);
                }
                [(a_ck, a_idx)] => {
                    // Self-identified periodic facet.
                    neighbor_updates
                        .get_mut(a_ck)
                        .expect("neighbor vector exists for quotient cell")[*a_idx] = Some(*a_ck);
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
        for &ck in &inserted_cell_keys {
            let neighbors = neighbor_updates
                .remove(&ck)
                .expect("neighbor vector exists for inserted quotient cell");
            tds_mut.set_neighbors_by_key(ck, &neighbors).map_err(|e| {
                TriangulationConstructionError::GeometricDegeneracy {
                    message: format!("set_neighbors_by_key failed for {ck:?}: {e}"),
                }
            })?;
        }

        // Canonicalize quotient-cell orientation after symbolic neighbor reconstruction.
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
        // Rebuild incident-cell pointers after topology surgery.
        tds_mut.assign_incident_cells().map_err(|e| {
            TriangulationConstructionError::GeometricDegeneracy {
                message: format!("assign_incident_cells failed: {e}"),
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
    use crate::topology::traits::global_topology_model::{
        GlobalTopologyModel, GlobalTopologyModelError,
    };
    use crate::topology::traits::topological_space::TopologyKind;
    use crate::vertex;

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
        use crate::core::delaunay_triangulation::InsertionOrderStrategy;
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
        use crate::topology::traits::topological_space::{
            GlobalTopology, ToroidalConstructionMode,
        };
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
            VertexBuilder::default()
                .point(Point::new([0.2_f64, 0.3]))
                .build()
                .unwrap(),
            VertexBuilder::default()
                .point(Point::new([f64::NAN, 0.1]))
                .build()
                .unwrap(),
            VertexBuilder::default()
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
        use crate::geometry::kernel::RobustKernel;
        use crate::topology::traits::topological_space::{
            GlobalTopology, ToroidalConstructionMode,
        };

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
            dt_euclidean.number_of_cells(),
            dt_toroidal.number_of_cells()
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
        let dt = DelaunayTriangulationBuilder::from_vertices(&vertices)
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
        use crate::geometry::kernel::RobustKernel;
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
        use crate::topology::traits::global_topology_model::ToroidalModel;
        let model = ToroidalModel::<2>::new([2.0, 3.0], ToroidalConstructionMode::Canonicalized);
        let result = DelaunayTriangulationBuilder::<f64, (), 2>::validate_topology_model(&model);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_topology_model_rejects_zero_period() {
        use crate::topology::traits::global_topology_model::ToroidalModel;
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
        use crate::topology::traits::global_topology_model::ToroidalModel;
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
        use crate::topology::traits::global_topology_model::ToroidalModel;
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
        use crate::topology::traits::global_topology_model::ToroidalModel;
        let model =
            ToroidalModel::<2>::new([f64::NAN, 3.0], ToroidalConstructionMode::Canonicalized);
        let result = DelaunayTriangulationBuilder::<f64, (), 2>::validate_topology_model(&model);
        assert!(result.is_err());
        let err_str = format!("{}", result.unwrap_err());
        assert!(err_str.contains("Invalid toroidal domain"));
    }

    #[test]
    fn test_validate_topology_model_accepts_euclidean() {
        use crate::topology::traits::global_topology_model::EuclideanModel;
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
        use crate::topology::traits::global_topology_model::ToroidalModel;
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
        use crate::topology::traits::global_topology_model::ToroidalModel;
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
        use crate::topology::traits::global_topology_model::ToroidalModel;
        use approx::assert_relative_eq;
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
        let kernel = FastKernel::new();
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
        use crate::topology::traits::global_topology_model::EuclideanModel;
        use approx::assert_relative_eq;
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
        use crate::topology::traits::global_topology_model::ToroidalModel;
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
        use crate::topology::traits::global_topology_model::ToroidalModel;
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
        use crate::topology::traits::global_topology_model::ToroidalModel;
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
