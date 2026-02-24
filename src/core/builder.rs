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

use crate::core::delaunay_triangulation::{
    ConstructionOptions, DelaunayTriangulation, DelaunayTriangulationConstructionError,
};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation::{TopologyGuarantee, TriangulationConstructionError};
use crate::core::vertex::{Vertex, VertexBuilder};
use crate::geometry::kernel::{FastKernel, Kernel};
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::{Coordinate, CoordinateScalar, ScalarAccumulative};
use crate::topology::spaces::toroidal::ToroidalSpace;
const TWO_POW_52_I64: i64 = 4_503_599_627_370_496; // 2^52
const TWO_POW_52_F64: f64 = 4_503_599_627_370_496.0; // 2^52
const MAX_OFFSET_UNITS: i64 = 1_048_576;
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;
type LiftedVertex<const D: usize> = (
    crate::core::triangulation_data_structure::VertexKey,
    [i8; D],
);
type FacetOccurrences<const D: usize> = crate::core::collections::FastHashMap<
    Vec<LiftedVertex<D>>,
    Vec<(crate::core::triangulation_data_structure::CellKey, usize)>,
>;

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

    /// Canonicalizes the vertex slice into the toroidal fundamental domain.
    ///
    /// Returns a new `Vec<Vertex<T, U, D>>` where every coordinate `c` in axis `i`
    /// has been replaced by `c.rem_euclid(domain[i])`. If any coordinate cannot be
    /// wrapped (non-finite or type-conversion failure), returns an error.
    fn canonicalize_vertices(
        vertices: &[Vertex<T, U, D>],
        space: &ToroidalSpace<D>,
    ) -> Result<Vec<Vertex<T, U, D>>, DelaunayTriangulationConstructionError> {
        let mut out = Vec::with_capacity(vertices.len());

        for (idx, v) in vertices.iter().enumerate() {
            let original_coords = v.point().coords();
            let mut wrapped = [T::zero(); D];

            for axis in 0..D {
                wrapped[axis] = space
                    .wrap_coord::<T>(axis, original_coords[axis])
                    .ok_or_else(|| TriangulationConstructionError::GeometricDegeneracy {
                        message: format!(
                            "Failed to canonicalize vertex {idx}: coordinate at axis \
                                     {axis} ({:?}) is not finite or cannot be wrapped into \
                                     the domain {:?}",
                            original_coords[axis], space.domain,
                        ),
                    })?;
            }

            let new_point = Point::new(wrapped);
            let new_vertex = v.data.map_or_else(
                || {
                    VertexBuilder::default()
                        .point(new_point)
                        .build()
                        .expect("vertex without data is always valid")
                },
                |data| {
                    VertexBuilder::default()
                        .point(new_point)
                        .data(data)
                        .build()
                        .expect("vertex with data is always valid")
                },
            );

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
                // Toroidal Phase 1: canonicalize then delegate.
                let canonical = Self::canonicalize_vertices(self.vertices, &space)?;
                DelaunayTriangulation::with_topology_guarantee_and_options(
                    kernel,
                    &canonical,
                    self.topology_guarantee,
                    self.construction_options,
                )
            }
            (Some(space), true) => {
                // Toroidal Phase 2: canonicalize then apply 3^D image-point method.
                let canonical = Self::canonicalize_vertices(self.vertices, &space)?;
                Self::build_periodic::<K, V>(
                    kernel,
                    &canonical,
                    &space,
                    self.topology_guarantee,
                    self.construction_options,
                )
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
    #[expect(
        clippy::too_many_lines,
        reason = "Image-point periodic DT algorithm is inherently multi-step; splitting would harm readability"
    )]
    fn build_periodic<K, V>(
        kernel: &K,
        canonical_vertices: &[Vertex<T, U, D>],
        space: &ToroidalSpace<D>,
        topology_guarantee: TopologyGuarantee,
        construction_options: ConstructionOptions,
    ) -> Result<DelaunayTriangulation<K, U, V, D>, DelaunayTriangulationConstructionError>
    where
        K: Kernel<D, Scalar = T>,
        K::Scalar: ScalarAccumulative,
        V: DataType,
    {
        use crate::core::cell::Cell;
        use crate::core::collections::{FastHashMap, Uuid, VertexKeySet};
        use crate::core::delaunay_triangulation::{DelaunayRepairPolicy, RetryPolicy};
        use crate::core::triangulation_data_structure::{CellKey, VertexKey};
        use num_traits::{NumCast, ToPrimitive};
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use rand::seq::SliceRandom;
        use std::num::NonZeroUsize;

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

        let canonical_f64: Vec<[f64; D]> = canonical_vertices
            .iter()
            .enumerate()
            .map(|(canon_idx, v)| {
                let orig_coords = v.point().coords();
                let mut coords = [0_f64; D];
                for i in 0..D {
                    let domain_i = space.domain[i];
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
                    let adjusted_u =
                        <f64 as NumCast>::from(u + off).expect("adjusted grid index fits in f64");
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
                    let shift_f64 = <f64 as From<i8>>::from(offset[i]) * space.domain[i];
                    let coord_f64 = canonical_f64[canon_idx][i] + shift_f64;
                    new_coords[i] = <T as NumCast>::from(coord_f64).ok_or_else(|| {
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
        let full_dt: DelaunayTriangulation<K, U, V, D> = if D == 2 {
            DelaunayTriangulation::with_topology_guarantee_and_options(
                kernel,
                &expanded,
                TopologyGuarantee::Pseudomanifold,
                construction_options,
            )?
        } else {
            let (retry_attempts, retry_seed) = match construction_options.retry_policy() {
                RetryPolicy::Disabled => (0, 0xA5A5_5A5A_D1E1_A1E1_u64),
                RetryPolicy::Shuffled {
                    attempts,
                    base_seed,
                }
                | RetryPolicy::DebugOnlyShuffled {
                    attempts,
                    base_seed,
                } => (
                    attempts.get(),
                    base_seed.unwrap_or(0xA5A5_5A5A_D1E1_A1E1_u64),
                ),
            };
            let total_attempts = retry_attempts
                .saturating_add(1)
                .max(NonZeroUsize::new(64).expect("literal is non-zero").get());

            let mut built: Option<DelaunayTriangulation<K, U, V, D>> = None;
            let mut last_insert_error: Option<String> = None;
            let mut insertion_order: Vec<usize> = (0..expanded.len()).collect();
            for attempt_idx in 0..total_attempts {
                if attempt_idx == 0 {
                    insertion_order
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, slot)| {
                            *slot = i;
                        });
                } else {
                    let attempt_idx_u64 =
                        u64::try_from(attempt_idx).expect("attempt index fits in u64");
                    let mut rng = StdRng::seed_from_u64(
                        retry_seed
                            .wrapping_add(attempt_idx_u64.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                    );
                    insertion_order.shuffle(&mut rng);
                }

                let mut candidate_dt: DelaunayTriangulation<K, U, V, D> =
                    DelaunayTriangulation::with_empty_kernel_and_topology_guarantee(
                        kernel.clone(),
                        TopologyGuarantee::Pseudomanifold,
                    );
                candidate_dt.set_delaunay_repair_policy(DelaunayRepairPolicy::Never);

                let mut failed = false;
                for (insert_idx, &source_idx) in insertion_order.iter().enumerate() {
                    if let Err(err) = candidate_dt.insert(expanded[source_idx]) {
                        last_insert_error = Some(format!(
                            "attempt={attempt_idx} insert_idx={insert_idx} source_idx={source_idx}: {err}"
                        ));
                        failed = true;
                        break;
                    }
                }

                if !failed {
                    built = Some(candidate_dt);
                    break;
                }
            }

            built.ok_or_else(|| TriangulationConstructionError::GeometricDegeneracy {
                message: format!(
                    "Periodic expanded DT insertion failed after {total_attempts} attempts: {}",
                    last_insert_error.unwrap_or_else(|| "unknown insertion failure".to_owned())
                ),
            })?
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

        let facet_signature =
            |lifted: &[(VertexKey, [i8; D])], opposite_idx: usize| -> Vec<(VertexKey, [i8; D])> {
                let mut facet: Vec<(VertexKey, [i8; D])> = lifted
                    .iter()
                    .enumerate()
                    .filter_map(|(i, descriptor)| (i != opposite_idx).then_some(*descriptor))
                    .collect();

                if let Some((_, anchor_offset)) = facet.iter().min_by_key(|(ck, _)| *ck).copied() {
                    for (_, offset) in &mut facet {
                        for axis in 0..D {
                            offset[axis] -= anchor_offset[axis];
                        }
                    }
                }

                facet.sort_unstable();
                facet
            };

        // Build quotient-cell representatives keyed by canonical vertex set.
        // If multiple lifted representatives map to the same canonical set, keep the
        // lexicographically smallest lifted representative for deterministic behavior.
        let mut representative_lifted_by_canonical: FastHashMap<
            Vec<VertexKey>,
            Vec<(VertexKey, [i8; D])>,
        > = FastHashMap::default();
        for ck in tds_ref.cell_keys() {
            let Some(lifted_vertices) = normalize_cell_lifted(ck) else {
                continue;
            };
            let mut canonical_signature: Vec<VertexKey> =
                lifted_vertices.iter().map(|(vk, _)| *vk).collect();
            canonical_signature.sort_unstable();

            if let Some(existing) = representative_lifted_by_canonical.get_mut(&canonical_signature)
            {
                let mut existing_sorted = existing.clone();
                existing_sorted.sort_unstable();
                let mut candidate_sorted = lifted_vertices.clone();
                candidate_sorted.sort_unstable();
                if candidate_sorted < existing_sorted {
                    *existing = lifted_vertices;
                }
            } else {
                representative_lifted_by_canonical.insert(canonical_signature, lifted_vertices);
            }
        }
        if representative_lifted_by_canonical.is_empty() {
            return Err(TriangulationConstructionError::GeometricDegeneracy {
                message: "No quotient periodic cells found in full image DT".to_owned(),
            }
            .into());
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
        let mut signatures_sorted: Vec<Vec<VertexKey>> =
            representative_lifted_by_canonical.keys().cloned().collect();
        signatures_sorted.sort_unstable();

        let mut inserted_cell_keys: Vec<CellKey> = Vec::with_capacity(signatures_sorted.len());
        let mut rep_lifted_by_key: FastHashMap<CellKey, Vec<(VertexKey, [i8; D])>> =
            FastHashMap::default();

        for signature in signatures_sorted {
            let Some(lifted_vertices) = representative_lifted_by_canonical.get(&signature) else {
                continue;
            };
            let canonical_vertices: Vec<VertexKey> =
                lifted_vertices.iter().map(|(ck, _)| *ck).collect();
            let cell = Cell::new(canonical_vertices, None).map_err(|e| {
                TriangulationConstructionError::GeometricDegeneracy {
                    message: format!("Failed to create quotient periodic cell: {e}"),
                }
            })?;
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

        // Rebuild neighbor pointers by pairing equal symbolic facet signatures in the quotient.
        let mut neighbor_updates: FastHashMap<CellKey, Vec<Option<CellKey>>> = inserted_cell_keys
            .iter()
            .copied()
            .map(|ck| (ck, vec![None; D + 1]))
            .collect();

        let mut facet_occurrences: FacetOccurrences<D> = FastHashMap::default();
        for &rep_ck in &inserted_cell_keys {
            let Some(lifted) = rep_lifted_by_key.get(&rep_ck) else {
                continue;
            };
            for facet_idx in 0..=D {
                let sig = facet_signature(lifted, facet_idx);
                facet_occurrences
                    .entry(sig)
                    .or_default()
                    .push((rep_ck, facet_idx));
            }
        }

        for (_facet_sig, occurrences) in facet_occurrences {
            match occurrences.as_slice() {
                [(a_ck, a_idx), (b_ck, b_idx)] => {
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
                    let mut sorted = occurrences.clone();
                    sorted.sort_unstable();
                    for chunk in sorted.chunks(2) {
                        match chunk {
                            [(a_ck, a_idx), (b_ck, b_idx)] => {
                                neighbor_updates
                                    .get_mut(a_ck)
                                    .expect("neighbor vector exists for quotient cell")[*a_idx] =
                                    Some(*b_ck);
                                neighbor_updates
                                    .get_mut(b_ck)
                                    .expect("neighbor vector exists for quotient cell")[*b_idx] =
                                    Some(*a_ck);
                            }
                            [(a_ck, a_idx)] => {
                                neighbor_updates
                                    .get_mut(a_ck)
                                    .expect("neighbor vector exists for quotient cell")[*a_idx] =
                                    Some(*a_ck);
                            }
                            _ => {}
                        }
                    }
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
    use crate::vertex;

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
}
