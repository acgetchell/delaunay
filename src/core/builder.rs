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
//! | Toroidal / periodic boundary conditions | [`DelaunayTriangulationBuilder`] with [`.toroidal()`](DelaunayTriangulationBuilder::toroidal) |
//! | Custom kernel (`RobustKernel`, etc.) | [`DelaunayTriangulationBuilder::build_with_kernel`] |
//!
//! # Phase 1 vs Phase 2
//!
//! **Phase 1 (current):** The builder canonicalizes all input vertices into the
//! fundamental domain `[0, L_i)` before passing them to the standard Euclidean
//! constructor. The resulting triangulation is a valid Euclidean Delaunay triangulation
//! of the canonicalized point set; it does **not** yet identify opposite boundary facets.
//!
//! **Phase 2 (future, issue #210):** Full periodic construction using the 3^D image-point
//! method — generating copies of each point shifted by ±L in each dimension, building
//! the full Euclidean DT, and extracting the restriction to the fundamental domain.
//! This will yield a true toroidal (χ = 0) triangulation.
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
        }
    }

    /// Enables toroidal
    ///
    /// Input vertices are canonicalized into `[0, L_i)` per dimension before
    /// the triangulation is built (Phase 1). Full periodic construction via the
    /// image-point method is planned for Phase 2 (issue #210).
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

    /// Sets the [`TopologyGuarantee`] used during construction (Level 3 validation strength).
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
        match self.topology {
            None => {
                // Euclidean path: delegate directly.
                DelaunayTriangulation::with_topology_guarantee_and_options(
                    kernel,
                    self.vertices,
                    self.topology_guarantee,
                    self.construction_options,
                )
            }
            Some(ref space) => {
                // Toroidal Phase 1: canonicalize then delegate.
                let canonical = Self::canonicalize_vertices(self.vertices, space)?;
                DelaunayTriangulation::with_topology_guarantee_and_options(
                    kernel,
                    &canonical,
                    self.topology_guarantee,
                    self.construction_options,
                )
            }
        }
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
