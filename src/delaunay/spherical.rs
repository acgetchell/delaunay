//! Prototype spherical Delaunay construction via ambient convex-hull duality.
//!
//! This module treats `D` as the intrinsic manifold dimension. Spherical points
//! live on `S^D`, realized in `R^(D+1)`, and construction uses an ambient
//! Euclidean convex hull only as the duality engine. The returned simplices are
//! intrinsic `D`-simplices on the sphere, not ambient `(D + 1)`-simplices.
//!
//! # Examples
//!
//! ```rust
//! use delaunay::prelude::construction::SphericalDelaunayBuilder;
//!
//! let points = [
//!     [1.0, 1.0, 1.0],
//!     [1.0, -1.0, -1.0],
//!     [-1.0, 1.0, -1.0],
//!     [-1.0, -1.0, 1.0],
//! ];
//!
//! let triangulation = SphericalDelaunayBuilder::<2>::try_new(points)?.build()?;
//! assert_eq!(triangulation.dimension(), 2);
//! assert_eq!(triangulation.ambient_dimension(), 3);
//! assert_eq!(triangulation.number_of_simplices(), 4);
//! assert!(triangulation.validate().is_ok());
//! # Ok::<(), delaunay::prelude::construction::SphericalDelaunayConstructionError>(())
//! ```

#![forbid(unsafe_code)]

use std::fmt;

use thiserror::Error;

use crate::builder::DelaunayTriangulationBuilder;
use crate::collections::{FastHashMap, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer};
use crate::construction::{
    ConstructionOptions, DelaunayTriangulationConstructionError, InitialSimplexStrategy,
};
use crate::geometry::Point;
use crate::geometry::algorithms::convex_hull::{ConvexHull, ConvexHullConstructionError};
use crate::geometry::predicates::{Orientation, simplex_orientation};
use crate::geometry::traits::coordinate::{CoordinateConversionError, CoordinateValidationError};
use crate::geometry::util::safe_usize_to_scalar;
use crate::tds::{
    FacetError, InvariantError, Simplex, SimplexValidationError, Tds, TdsConstructionError,
    TdsError, TdsMutationError, Vertex,
};
use crate::topology::characteristics::validation::validate_triangulation_euler_from_validated_facet_map;
use crate::topology::manifold::{
    ManifoldError, ValidatedFacetDegreeMap, validate_closed_boundary_from_validated_facet_map,
    validate_ridge_links, validate_vertex_links_from_validated_facet_map,
};
use crate::topology::spaces::spherical::{
    SphericalMetric, SphericalPoint, SphericalPointError, ambient_array_from_slice,
};
use crate::topology::traits::topological_space::{GlobalTopology, TopologyError};
use crate::{DelaunayTriangulation, TopologyGuarantee, TriangulationValidationError, vertex};

/// Default tracking issue for full spherical triangulation support.
const SPHERICAL_ROADMAP_ISSUE: u32 = 414;

/// A spherical `D`-simplex, represented by point indices.
///
/// Each simplex has exactly `D + 1` unique vertex indices. For example,
/// simplices on `S^2` are triangles with three indices, while simplices on
/// `S^3` are tetrahedra with four indices.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::construction::SphericalSimplex;
///
/// let simplex = SphericalSimplex::<2>::try_new(vec![0, 1, 2], 4)?;
/// assert_eq!(simplex.dimension(), 2);
/// assert_eq!(simplex.vertex_indices(), &[0, 1, 2]);
/// # Ok::<(), delaunay::prelude::construction::SphericalSimplexError>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SphericalSimplex<const D: usize> {
    vertices: Vec<usize>,
}

impl<const D: usize> SphericalSimplex<D> {
    /// Creates a spherical simplex after validating arity, bounds, and uniqueness.
    ///
    /// # Errors
    ///
    /// Returns [`SphericalSimplexError`] when `vertices` is not a valid simplex
    /// over a point set of size `vertex_count`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::SphericalSimplex;
    ///
    /// let simplex = SphericalSimplex::<2>::try_new(vec![0, 1, 2], 4)?;
    ///
    /// assert_eq!(simplex.vertex_indices(), &[0, 1, 2]);
    /// # Ok::<(), delaunay::prelude::construction::SphericalSimplexError>(())
    /// ```
    pub fn try_new(
        vertices: Vec<usize>,
        vertex_count: usize,
    ) -> Result<Self, SphericalSimplexError> {
        let expected = D + 1;
        if vertices.len() != expected {
            return Err(SphericalSimplexError::InvalidArity {
                dimension: D,
                expected,
                actual: vertices.len(),
            });
        }

        for (position, &vertex_index) in vertices.iter().enumerate() {
            if vertex_index >= vertex_count {
                return Err(SphericalSimplexError::VertexIndexOutOfBounds {
                    vertex_index,
                    vertex_count,
                });
            }
            for &other in &vertices[position + 1..] {
                if vertex_index == other {
                    return Err(SphericalSimplexError::DuplicateVertex { vertex_index });
                }
            }
        }

        Ok(Self { vertices })
    }

    /// Returns the vertex indices that span this spherical simplex.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::SphericalSimplex;
    ///
    /// let simplex = SphericalSimplex::<2>::try_new(vec![0, 1, 2], 4)?;
    ///
    /// assert_eq!(simplex.vertex_indices(), &[0, 1, 2]);
    /// # Ok::<(), delaunay::prelude::construction::SphericalSimplexError>(())
    /// ```
    #[must_use]
    pub fn vertex_indices(&self) -> &[usize] {
        &self.vertices
    }

    /// Returns the intrinsic dimension of this simplex.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::SphericalSimplex;
    ///
    /// let simplex = SphericalSimplex::<3>::try_new(vec![0, 1, 2, 3], 5)?;
    ///
    /// assert_eq!(simplex.dimension(), 3);
    /// # Ok::<(), delaunay::prelude::construction::SphericalSimplexError>(())
    /// ```
    #[must_use]
    pub const fn dimension(&self) -> usize {
        D
    }

    /// Returns a sorted facet key by omitting one vertex from this simplex.
    fn facet_key_omitting(&self, omitted_index: usize) -> Vec<usize> {
        let mut facet = Vec::with_capacity(D);
        for (index, &vertex) in self.vertices.iter().enumerate() {
            if index != omitted_index {
                facet.push(vertex);
            }
        }
        facet.sort_unstable();
        facet
    }
}

/// Prototype spherical Delaunay triangulation.
///
/// This type stores normalized spherical points and hull-derived intrinsic
/// simplices. Its validation surface follows the crate-wide layer model:
/// Level 3 is intrinsic PL topology, Level 4 is spherical realization in
/// `S^D \subset R^(D+1)`, and Level 5 is the spherical Delaunay predicate via
/// ambient convex-hull support.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::construction::SphericalDelaunayBuilder;
///
/// let points = [
///     [1.0, 1.0, 1.0],
///     [1.0, -1.0, -1.0],
///     [-1.0, 1.0, -1.0],
///     [-1.0, -1.0, 1.0],
/// ];
///
/// let triangulation = SphericalDelaunayBuilder::<2>::try_new(points)?.build()?;
/// assert_eq!(triangulation.number_of_vertices(), 4);
/// assert_eq!(triangulation.number_of_simplices(), 4);
/// assert!(triangulation
///     .simplices()
///     .iter()
///     .all(|simplex| simplex.vertex_indices().len() == 3));
/// assert!(triangulation.validate_realization().is_ok());
/// assert!(triangulation.validate_delaunay().is_ok());
/// # Ok::<(), delaunay::prelude::construction::SphericalDelaunayConstructionError>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct SphericalDelaunayTriangulation<const D: usize> {
    points: Vec<SphericalPoint<D>>,
    simplices: Vec<SphericalSimplex<D>>,
    radius: f64,
}

impl<const D: usize> SphericalDelaunayTriangulation<D> {
    /// Returns the normalized spherical input points.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     SphericalDelaunayBuilder, SphericalDelaunayConstructionError,
    ///     SphericalDelaunayTriangulation,
    /// };
    ///
    /// # fn sample() -> Result<SphericalDelaunayTriangulation<2>, SphericalDelaunayConstructionError> {
    /// #     SphericalDelaunayBuilder::<2>::try_new([
    /// #         [1.0, 1.0, 1.0],
    /// #         [1.0, -1.0, -1.0],
    /// #         [-1.0, 1.0, -1.0],
    /// #         [-1.0, -1.0, 1.0],
    /// #     ])?.build()
    /// # }
    /// let triangulation = sample()?;
    ///
    /// assert_eq!(triangulation.points().len(), 4);
    /// # Ok::<(), SphericalDelaunayConstructionError>(())
    /// ```
    #[must_use]
    pub fn points(&self) -> &[SphericalPoint<D>] {
        &self.points
    }

    /// Returns the intrinsic spherical Delaunay simplices.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     SphericalDelaunayBuilder, SphericalDelaunayConstructionError,
    ///     SphericalDelaunayTriangulation,
    /// };
    ///
    /// # fn sample() -> Result<SphericalDelaunayTriangulation<2>, SphericalDelaunayConstructionError> {
    /// #     SphericalDelaunayBuilder::<2>::try_new([
    /// #         [1.0, 1.0, 1.0],
    /// #         [1.0, -1.0, -1.0],
    /// #         [-1.0, 1.0, -1.0],
    /// #         [-1.0, -1.0, 1.0],
    /// #     ])?.build()
    /// # }
    /// let triangulation = sample()?;
    ///
    /// assert_eq!(triangulation.simplices().len(), 4);
    /// # Ok::<(), SphericalDelaunayConstructionError>(())
    /// ```
    #[must_use]
    pub fn simplices(&self) -> &[SphericalSimplex<D>] {
        &self.simplices
    }

    /// Returns the sphere radius used for normalization and distance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{SphericalDelaunayBuilder, SphericalDelaunayConstructionError};
    ///
    /// let triangulation = SphericalDelaunayBuilder::<2>::try_new_with_radius(
    ///     [
    ///         [1.0, 1.0, 1.0],
    ///         [1.0, -1.0, -1.0],
    ///         [-1.0, 1.0, -1.0],
    ///         [-1.0, -1.0, 1.0],
    ///     ],
    ///     2.0,
    /// )?.build()?;
    ///
    /// assert_eq!(triangulation.radius(), 2.0);
    /// # Ok::<(), SphericalDelaunayConstructionError>(())
    /// ```
    #[must_use]
    pub const fn radius(&self) -> f64 {
        self.radius
    }

    /// Returns the intrinsic sphere dimension.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{SphericalDelaunayBuilder, SphericalDelaunayConstructionError};
    ///
    /// let triangulation = SphericalDelaunayBuilder::<2>::try_new([
    ///     [1.0, 1.0, 1.0],
    ///     [1.0, -1.0, -1.0],
    ///     [-1.0, 1.0, -1.0],
    ///     [-1.0, -1.0, 1.0],
    /// ])?.build()?;
    ///
    /// assert_eq!(triangulation.dimension(), 2);
    /// # Ok::<(), SphericalDelaunayConstructionError>(())
    /// ```
    #[must_use]
    pub const fn dimension(&self) -> usize {
        D
    }

    /// Returns the ambient coordinate dimension `D + 1`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{SphericalDelaunayBuilder, SphericalDelaunayConstructionError};
    ///
    /// let triangulation = SphericalDelaunayBuilder::<2>::try_new([
    ///     [1.0, 1.0, 1.0],
    ///     [1.0, -1.0, -1.0],
    ///     [-1.0, 1.0, -1.0],
    ///     [-1.0, -1.0, 1.0],
    /// ])?.build()?;
    ///
    /// assert_eq!(triangulation.ambient_dimension(), 3);
    /// # Ok::<(), SphericalDelaunayConstructionError>(())
    /// ```
    #[must_use]
    pub const fn ambient_dimension(&self) -> usize {
        D + 1
    }

    /// Returns the number of spherical vertices.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{SphericalDelaunayBuilder, SphericalDelaunayConstructionError};
    ///
    /// let triangulation = SphericalDelaunayBuilder::<2>::try_new([
    ///     [1.0, 1.0, 1.0],
    ///     [1.0, -1.0, -1.0],
    ///     [-1.0, 1.0, -1.0],
    ///     [-1.0, -1.0, 1.0],
    /// ])?.build()?;
    ///
    /// assert_eq!(triangulation.number_of_vertices(), 4);
    /// # Ok::<(), SphericalDelaunayConstructionError>(())
    /// ```
    #[must_use]
    pub const fn number_of_vertices(&self) -> usize {
        self.points.len()
    }

    /// Returns the number of spherical simplices.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{SphericalDelaunayBuilder, SphericalDelaunayConstructionError};
    ///
    /// let triangulation = SphericalDelaunayBuilder::<2>::try_new([
    ///     [1.0, 1.0, 1.0],
    ///     [1.0, -1.0, -1.0],
    ///     [-1.0, 1.0, -1.0],
    ///     [-1.0, -1.0, 1.0],
    /// ])?.build()?;
    ///
    /// assert_eq!(triangulation.number_of_simplices(), 4);
    /// # Ok::<(), SphericalDelaunayConstructionError>(())
    /// ```
    #[must_use]
    pub const fn number_of_simplices(&self) -> usize {
        self.simplices.len()
    }

    /// Returns the geodesic distance between two spherical vertices.
    ///
    /// # Errors
    ///
    /// Returns [`SphericalDelaunayValidationError::VertexIndexOutOfBounds`] if
    /// either index is outside the point set. A
    /// [`SphericalDelaunayValidationError::Metric`] error is possible only if
    /// the stored radius invariant has been violated internally. A
    /// [`SphericalDelaunayValidationError::GeodesicDistance`] error indicates
    /// that one of the stored points no longer matches the triangulation's
    /// sphere radius.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{SphericalDelaunayBuilder, SphericalDelaunayConstructionError};
    ///
    /// let triangulation = SphericalDelaunayBuilder::<2>::try_new([
    ///     [1.0, 1.0, 1.0],
    ///     [1.0, -1.0, -1.0],
    ///     [-1.0, 1.0, -1.0],
    ///     [-1.0, -1.0, 1.0],
    /// ])?.build()?;
    ///
    /// std::assert_matches!(
    ///     triangulation.distance_between_vertices(0, 1),
    ///     Ok(distance) if distance > 0.0
    /// );
    /// # Ok::<(), SphericalDelaunayConstructionError>(())
    /// ```
    pub fn distance_between_vertices(
        &self,
        left: usize,
        right: usize,
    ) -> Result<f64, SphericalDelaunayValidationError> {
        let left_point = self.points.get(left).ok_or(
            SphericalDelaunayValidationError::VertexIndexOutOfBounds {
                vertex_index: left,
                vertex_count: self.points.len(),
            },
        )?;
        let right_point = self.points.get(right).ok_or(
            SphericalDelaunayValidationError::VertexIndexOutOfBounds {
                vertex_index: right,
                vertex_count: self.points.len(),
            },
        )?;
        SphericalMetric::<D>::try_new(self.radius)
            .map_err(|source| SphericalDelaunayValidationError::Metric { source })?
            .try_distance(left_point, right_point)
            .map_err(
                |source| SphericalDelaunayValidationError::GeodesicDistance {
                    left,
                    right,
                    source,
                },
            )
    }

    /// Validates supported spherical prototype invariants.
    ///
    /// This is an alias for [`validate_delaunay`](Self::validate_delaunay), the
    /// strongest validation layer implemented by the prototype.
    ///
    /// # Errors
    ///
    /// Returns [`SphericalDelaunayValidationError`] for point normalization,
    /// simplex-shape, closed-adjacency, or ambient hull-support failures.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{SphericalDelaunayBuilder, SphericalDelaunayConstructionError};
    ///
    /// let triangulation = SphericalDelaunayBuilder::<2>::try_new([
    ///     [1.0, 1.0, 1.0],
    ///     [1.0, -1.0, -1.0],
    ///     [-1.0, 1.0, -1.0],
    ///     [-1.0, -1.0, 1.0],
    /// ])?.build()?;
    ///
    /// std::assert_matches!(triangulation.validate(), Ok(()));
    /// # Ok::<(), SphericalDelaunayConstructionError>(())
    /// ```
    pub fn validate(&self) -> Result<(), SphericalDelaunayValidationError> {
        self.validate_delaunay()
    }

    /// Checks the supported Levels 1-3 spherical prototype invariants.
    ///
    /// This fast-fail wrapper matches the crate-wide validation naming surface
    /// and delegates to [`validate_topology`](Self::validate_topology).
    ///
    /// # Errors
    ///
    /// Returns [`SphericalDelaunayValidationError`] when points are not on the
    /// configured sphere, simplices are malformed, or intrinsic PL topology
    /// checks fail.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{SphericalDelaunayBuilder, SphericalDelaunayConstructionError};
    ///
    /// let triangulation = SphericalDelaunayBuilder::<2>::try_new([
    ///     [1.0, 1.0, 1.0],
    ///     [1.0, -1.0, -1.0],
    ///     [-1.0, 1.0, -1.0],
    ///     [-1.0, -1.0, 1.0],
    /// ])?.build()?;
    ///
    /// std::assert_matches!(triangulation.is_valid_topology(), Ok(()));
    /// # Ok::<(), SphericalDelaunayConstructionError>(())
    /// ```
    pub fn is_valid_topology(&self) -> Result<(), SphericalDelaunayValidationError> {
        self.validate_topology()
    }

    /// Checks the prototype Level 4 spherical realization.
    ///
    /// This fast-fail wrapper matches the crate-wide validation naming surface
    /// and delegates to [`validate_realization`](Self::validate_realization).
    ///
    /// # Errors
    ///
    /// Returns [`SphericalDelaunayValidationError`] when lower validation
    /// layers fail, the intrinsic dimension is outside the prototype support
    /// envelope, or a simplex is geometrically degenerate.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{SphericalDelaunayBuilder, SphericalDelaunayConstructionError};
    ///
    /// let triangulation = SphericalDelaunayBuilder::<2>::try_new([
    ///     [1.0, 1.0, 1.0],
    ///     [1.0, -1.0, -1.0],
    ///     [-1.0, 1.0, -1.0],
    ///     [-1.0, -1.0, 1.0],
    /// ])?.build()?;
    ///
    /// std::assert_matches!(triangulation.is_valid_realization(), Ok(()));
    /// # Ok::<(), SphericalDelaunayConstructionError>(())
    /// ```
    pub fn is_valid_realization(&self) -> Result<(), SphericalDelaunayValidationError> {
        self.validate_realization()
    }

    /// Checks the prototype Level 5 spherical Delaunay predicate.
    ///
    /// This fast-fail wrapper matches the crate-wide validation naming surface
    /// and delegates to [`validate_delaunay`](Self::validate_delaunay).
    ///
    /// # Errors
    ///
    /// Returns [`SphericalDelaunayValidationError`] when lower validation
    /// layers fail, the intrinsic dimension is outside the prototype support
    /// envelope, or a simplex is not an ambient supporting hull facet.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{SphericalDelaunayBuilder, SphericalDelaunayConstructionError};
    ///
    /// let triangulation = SphericalDelaunayBuilder::<2>::try_new([
    ///     [1.0, 1.0, 1.0],
    ///     [1.0, -1.0, -1.0],
    ///     [-1.0, 1.0, -1.0],
    ///     [-1.0, -1.0, 1.0],
    /// ])?.build()?;
    ///
    /// std::assert_matches!(triangulation.is_valid_delaunay(), Ok(()));
    /// # Ok::<(), SphericalDelaunayConstructionError>(())
    /// ```
    pub fn is_valid_delaunay(&self) -> Result<(), SphericalDelaunayValidationError> {
        self.validate_delaunay()
    }

    /// Validates the supported Levels 1-3 spherical prototype invariants.
    ///
    /// # Errors
    ///
    /// Returns [`SphericalDelaunayValidationError`] when points are not on the
    /// configured sphere, simplices are malformed, or a codimension-1 face is
    /// not incident to exactly two simplices.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{SphericalDelaunayBuilder, SphericalDelaunayConstructionError};
    ///
    /// let triangulation = SphericalDelaunayBuilder::<2>::try_new([
    ///     [1.0, 1.0, 1.0],
    ///     [1.0, -1.0, -1.0],
    ///     [-1.0, 1.0, -1.0],
    ///     [-1.0, -1.0, 1.0],
    /// ])?.build()?;
    ///
    /// std::assert_matches!(triangulation.validate_topology(), Ok(()));
    /// # Ok::<(), SphericalDelaunayConstructionError>(())
    /// ```
    pub fn validate_topology(&self) -> Result<(), SphericalDelaunayValidationError> {
        SphericalMetric::<D>::try_new(self.radius)
            .map_err(|source| SphericalDelaunayValidationError::Metric { source })?;
        if self.simplices.is_empty() {
            return Err(SphericalDelaunayValidationError::NoSimplices);
        }

        for (point_index, point) in self.points.iter().enumerate() {
            if point.coords().len() != D + 1 {
                return Err(
                    SphericalDelaunayValidationError::InvalidPointAmbientDimension {
                        point_index,
                        expected: D + 1,
                        actual: point.coords().len(),
                    },
                );
            }
            let norm_over_radius = scaled_norm_over_radius(point.coords(), self.radius);
            if (norm_over_radius - 1.0).abs() > 1.0e-12 {
                return Err(SphericalDelaunayValidationError::PointNotOnSphere {
                    point_index,
                    radius: self.radius,
                    norm: scaled_euclidean_norm(point.coords()),
                });
            }
        }

        let mut simplex_indices: FastHashMap<Vec<usize>, usize> = FastHashMap::default();
        let mut facet_counts: FastHashMap<Vec<usize>, usize> = FastHashMap::default();
        let mut used_vertices = vec![false; self.points.len()];
        for (simplex_index, simplex) in self.simplices.iter().enumerate() {
            SphericalSimplex::<D>::try_new(simplex.vertex_indices().to_vec(), self.points.len())
                .map_err(|source| SphericalDelaunayValidationError::InvalidSimplex {
                    simplex_index,
                    source,
                })?;
            let mut simplex_key = simplex.vertex_indices().to_vec();
            simplex_key.sort_unstable();
            if let Some(previous_simplex_index) =
                simplex_indices.insert(simplex_key.clone(), simplex_index)
            {
                return Err(SphericalDelaunayValidationError::DuplicateSimplex {
                    simplex_index,
                    previous_simplex_index,
                    simplex_vertices: simplex_key,
                });
            }
            for omitted_index in 0..=D {
                let key = simplex.facet_key_omitting(omitted_index);
                let count = facet_counts.entry(key).or_insert(0);
                *count = count.saturating_add(1);
            }
            for &vertex_index in simplex.vertex_indices() {
                used_vertices[vertex_index] = true;
            }
        }

        for (facet_vertices, incident_count) in facet_counts {
            if incident_count != 2 {
                return Err(SphericalDelaunayValidationError::InvalidFacetIncidence {
                    facet_vertices,
                    incident_count,
                });
            }
        }
        for (vertex_index, is_used) in used_vertices.into_iter().enumerate() {
            if !is_used {
                return Err(SphericalDelaunayValidationError::UnusedVertex { vertex_index });
            }
        }

        self.validate_intrinsic_pl_topology()?;

        Ok(())
    }

    /// Validates intrinsic PL topology through the crate's topology validators.
    ///
    /// Spherical coordinates live in `R^(D+1)`, but Level 3 is intentionally
    /// realization-independent. This bridge therefore builds a temporary abstract
    /// TDS with synthetic finite coordinates, then runs the same combinatorial
    /// connectedness, closed-boundary, link, and Euler checks used by ordinary
    /// triangulations without invoking Euclidean orientation validation.
    fn validate_intrinsic_pl_topology(&self) -> Result<(), SphericalDelaunayValidationError> {
        let mut tds: Tds<(), (), D> = Tds::empty();
        let mut vertex_keys = Vec::with_capacity(self.points.len());
        for point_index in 0..self.points.len() {
            let coords = synthetic_topology_coordinates::<D>(point_index).map_err(|source| {
                SphericalDelaunayValidationError::AbstractTopologyVertex {
                    point_index,
                    source,
                }
            })?;
            let vertex = vertex!(coords; data = ()).map_err(|source| {
                SphericalDelaunayValidationError::AbstractTopologyVertex {
                    point_index,
                    source,
                }
            })?;
            let vertex_key = tds.insert_vertex_with_mapping(vertex).map_err(|source| {
                SphericalDelaunayValidationError::AbstractTopologyVertexInsertion {
                    point_index,
                    source: Box::new(source),
                }
            })?;
            vertex_keys.push(vertex_key);
        }

        for (simplex_index, simplex) in self.simplices.iter().enumerate() {
            let mut simplex_vertices = Vec::with_capacity(D + 1);
            for &point_index in simplex.vertex_indices() {
                let vertex_key = *vertex_keys.get(point_index).ok_or(
                    SphericalDelaunayValidationError::VertexIndexOutOfBounds {
                        vertex_index: point_index,
                        vertex_count: vertex_keys.len(),
                    },
                )?;
                simplex_vertices.push(vertex_key);
            }
            let abstract_simplex =
                Simplex::try_new_with_data(simplex_vertices, None).map_err(|source| {
                    SphericalDelaunayValidationError::AbstractTopologySimplex {
                        simplex_index,
                        source,
                    }
                })?;
            tds.insert_simplex_with_mapping(abstract_simplex)
                .map_err(|source| {
                    SphericalDelaunayValidationError::AbstractTopologySimplexInsertion {
                        simplex_index,
                        source: Box::new(source),
                    }
                })?;
        }

        tds.assign_neighbors().map_err(intrinsic_tds_error)?;
        tds.assign_incident_simplices().map_err(|source| {
            SphericalDelaunayValidationError::AbstractTopologyMutation { source }
        })?;

        if !tds.is_connected() {
            return Err(intrinsic_topology_error(
                TriangulationValidationError::Disconnected {
                    simplex_count: tds.number_of_simplices(),
                },
            ));
        }

        let facet_to_simplices = tds
            .build_facet_to_simplices_map()
            .map_err(intrinsic_tds_error)?;
        let facet_to_simplices = ValidatedFacetDegreeMap::try_from_facet_map(&facet_to_simplices)
            .map_err(intrinsic_manifold_error)?;
        validate_closed_boundary_from_validated_facet_map(
            &tds,
            facet_to_simplices,
            GlobalTopology::Spherical,
        )
        .map_err(intrinsic_manifold_error)?;
        validate_ridge_links(&tds).map_err(intrinsic_manifold_error)?;
        validate_vertex_links_from_validated_facet_map(
            &tds,
            facet_to_simplices,
            GlobalTopology::Spherical,
        )
        .map_err(intrinsic_manifold_error)?;

        let topology_result = validate_triangulation_euler_from_validated_facet_map(
            &tds,
            facet_to_simplices,
            GlobalTopology::Spherical,
        )
        .map_err(intrinsic_topology_support_error)?;
        if let Some(expected) = topology_result.expected
            && topology_result.chi != expected
        {
            return Err(intrinsic_topology_error(
                TriangulationValidationError::EulerCharacteristicMismatch {
                    computed: topology_result.chi,
                    expected,
                    classification: topology_result.classification,
                },
            ));
        }

        Ok(())
    }

    /// Validates the prototype Level 4 spherical realization.
    ///
    /// The intrinsic PL topology is still checked by
    /// [`validate_topology`](Self::validate_topology). The spherical realization
    /// layer then checks that each maximal simplex is a nondegenerate geodesic
    /// simplex on `S^D` by requiring the ambient hyperplane through its
    /// vertices not to contain the sphere center.
    ///
    /// # Errors
    ///
    /// Returns [`SphericalDelaunayValidationError`] when lower validation
    /// layers fail, the intrinsic dimension is outside the prototype support
    /// envelope, or a stored simplex is geometrically degenerate.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{SphericalDelaunayBuilder, SphericalDelaunayConstructionError};
    ///
    /// let triangulation = SphericalDelaunayBuilder::<2>::try_new([
    ///     [1.0, 1.0, 1.0],
    ///     [1.0, -1.0, -1.0],
    ///     [-1.0, 1.0, -1.0],
    ///     [-1.0, -1.0, 1.0],
    /// ])?.build()?;
    ///
    /// std::assert_matches!(triangulation.validate_realization(), Ok(()));
    /// # Ok::<(), SphericalDelaunayConstructionError>(())
    /// ```
    pub fn validate_realization(&self) -> Result<(), SphericalDelaunayValidationError> {
        self.validate_topology()?;
        match D {
            2 => self.validate_realization_with_ambient::<3>(),
            3 => self.validate_realization_with_ambient::<4>(),
            _ => Err(SphericalDelaunayValidationError::UnsupportedLayer {
                layer: SphericalValidationLayer::Realization,
                dimension: D,
                tracking_issue: SPHERICAL_ROADMAP_ISSUE,
            }),
        }
    }

    /// Validates the prototype Level 5 spherical Delaunay predicate.
    ///
    /// A spherical simplex is accepted when its vertices define an ambient
    /// supporting hyperplane in `R^(D+1)`: the sphere center must lie on the
    /// interior side of the hyperplane, and every non-simplex vertex must lie on
    /// that same side or on the hyperplane. This is the convex-hull duality
    /// certificate for simplices on `S^D`.
    ///
    /// # Errors
    ///
    /// Returns [`SphericalDelaunayValidationError`] when lower validation
    /// layers fail, the intrinsic dimension is outside the prototype support
    /// envelope, or a simplex is not an ambient supporting hull facet.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{SphericalDelaunayBuilder, SphericalDelaunayConstructionError};
    ///
    /// let triangulation = SphericalDelaunayBuilder::<2>::try_new([
    ///     [1.0, 1.0, 1.0],
    ///     [1.0, -1.0, -1.0],
    ///     [-1.0, 1.0, -1.0],
    ///     [-1.0, -1.0, 1.0],
    /// ])?.build()?;
    ///
    /// std::assert_matches!(triangulation.validate_delaunay(), Ok(()));
    /// # Ok::<(), SphericalDelaunayConstructionError>(())
    /// ```
    pub fn validate_delaunay(&self) -> Result<(), SphericalDelaunayValidationError> {
        self.validate_realization()?;
        match D {
            2 => self.validate_delaunay_with_ambient::<3>(),
            3 => self.validate_delaunay_with_ambient::<4>(),
            _ => Err(SphericalDelaunayValidationError::UnsupportedLayer {
                layer: SphericalValidationLayer::Delaunay,
                dimension: D,
                tracking_issue: SPHERICAL_ROADMAP_ISSUE,
            }),
        }
    }

    /// Validates each simplex as nondegenerate for Level 4.
    ///
    /// This helper backs the public spherical realization contract: every simplex
    /// must determine an ambient hyperplane that separates the simplex from the
    /// sphere center.
    fn validate_realization_with_ambient<const A: usize>(
        &self,
    ) -> Result<(), SphericalDelaunayValidationError> {
        Self::validate_ambient_dimension::<A>()?;

        let origin = Self::ambient_origin::<A>()?;
        for (simplex_index, simplex) in self.simplices.iter().enumerate() {
            self.simplex_origin_side::<A>(simplex_index, simplex, origin)?;
        }
        Ok(())
    }

    /// Validates each simplex as an ambient supporting hull facet for Level 5.
    ///
    /// This helper backs the public spherical Delaunay contract by checking the
    /// empty-cap condition through ambient convex-hull support.
    fn validate_delaunay_with_ambient<const A: usize>(
        &self,
    ) -> Result<(), SphericalDelaunayValidationError> {
        Self::validate_ambient_dimension::<A>()?;

        let origin = Self::ambient_origin::<A>()?;
        for (simplex_index, simplex) in self.simplices.iter().enumerate() {
            let origin_side = self.simplex_origin_side::<A>(simplex_index, simplex, origin)?;

            for point_index in 0..self.points.len() {
                if simplex.vertex_indices().contains(&point_index) {
                    continue;
                }
                let test_point = self.ambient_point::<A>(point_index)?;
                let point_side = self.simplex_orientation_with_point::<A>(
                    simplex_index,
                    simplex,
                    test_point,
                    Some(point_index),
                )?;
                if point_side != Orientation::DEGENERATE && point_side != origin_side {
                    return Err(SphericalDelaunayValidationError::NonSupportingSimplex {
                        simplex_index,
                        simplex_vertices: simplex.vertex_indices().to_vec(),
                        point_index,
                        origin_side,
                        point_side,
                    });
                }
            }
        }
        Ok(())
    }

    /// Validates that a const-generic ambient dimension matches `S^D`.
    const fn validate_ambient_dimension<const A: usize>()
    -> Result<(), SphericalDelaunayValidationError> {
        if A != D + 1 {
            return Err(SphericalDelaunayValidationError::AmbientDimensionMismatch {
                dimension: D,
                expected: D + 1,
                actual: A,
            });
        }
        Ok(())
    }

    /// Returns the ambient origin used as the sphere center.
    fn ambient_origin<const A: usize>() -> Result<Point<A>, SphericalDelaunayValidationError> {
        Point::<A>::try_new([0.0; A]).map_err(|source| {
            SphericalDelaunayValidationError::AmbientPointValidation {
                point_index: None,
                source,
            }
        })
    }

    /// Computes which side of a simplex hyperplane contains the sphere center.
    fn simplex_origin_side<const A: usize>(
        &self,
        simplex_index: usize,
        simplex: &SphericalSimplex<D>,
        origin: Point<A>,
    ) -> Result<Orientation, SphericalDelaunayValidationError> {
        let origin_side =
            self.simplex_orientation_with_point::<A>(simplex_index, simplex, origin, None)?;
        if origin_side == Orientation::DEGENERATE {
            return Err(
                SphericalDelaunayValidationError::SimplexHyperplaneContainsOrigin {
                    simplex_index,
                    simplex_vertices: simplex.vertex_indices().to_vec(),
                },
            );
        }
        Ok(origin_side)
    }

    /// Converts a normalized spherical point into an ambient [`Point`].
    fn ambient_point<const A: usize>(
        &self,
        point_index: usize,
    ) -> Result<Point<A>, SphericalDelaunayValidationError> {
        let point = self.points.get(point_index).ok_or(
            SphericalDelaunayValidationError::VertexIndexOutOfBounds {
                vertex_index: point_index,
                vertex_count: self.points.len(),
            },
        )?;
        let coords = ambient_array_from_slice::<A>(point.coords()).map_err(|source| {
            SphericalDelaunayValidationError::AmbientCoordinateArray {
                point_index,
                source,
            }
        })?;
        Point::<A>::try_new(coords).map_err(|source| {
            SphericalDelaunayValidationError::AmbientPointValidation {
                point_index: Some(point_index),
                source,
            }
        })
    }

    /// Computes the ambient orientation certificate for a spherical simplex.
    ///
    /// The extra point is either the sphere center for Level 4/5 orientation
    /// setup or a non-simplex vertex used to prove or refute supporting-facet
    /// status.
    fn simplex_orientation_with_point<const A: usize>(
        &self,
        simplex_index: usize,
        simplex: &SphericalSimplex<D>,
        extra_point: Point<A>,
        extra_point_index: Option<usize>,
    ) -> Result<Orientation, SphericalDelaunayValidationError> {
        let mut simplex_points: SmallBuffer<Point<A>, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(A + 1);
        for &point_index in simplex.vertex_indices() {
            simplex_points.push(self.ambient_point::<A>(point_index)?);
        }
        simplex_points.push(extra_point);
        simplex_orientation::<A>(&simplex_points).map_err(|source| {
            SphericalDelaunayValidationError::SimplexOrientation {
                simplex_index,
                point_index: extra_point_index,
                source,
            }
        })
    }
}

/// Builder for prototype spherical Delaunay construction.
///
/// `D` is the intrinsic sphere dimension. The input coordinates must have
/// length `D + 1`; they are normalized onto the requested radius before
/// construction. The current prototype is release-validated for `D = 2` and
/// `D = 3`.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::construction::SphericalDelaunayBuilder;
///
/// let points = [
///     [1.0, 0.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0, 0.0],
///     [0.0, 0.0, 1.0, 0.0],
///     [0.0, 0.0, 0.0, 1.0],
///     [-1.0, -1.0, -1.0, -1.0],
/// ];
///
/// let triangulation = SphericalDelaunayBuilder::<3>::try_new(points)?.build()?;
/// assert_eq!(triangulation.dimension(), 3);
/// assert_eq!(triangulation.ambient_dimension(), 4);
/// assert!(triangulation.validate().is_ok());
/// # Ok::<(), delaunay::prelude::construction::SphericalDelaunayConstructionError>(())
/// ```
#[derive(Clone, Debug)]
pub struct SphericalDelaunayBuilder<const D: usize> {
    points: Vec<SphericalPoint<D>>,
    radius: f64,
    construction_options: ConstructionOptions,
}

impl<const D: usize> SphericalDelaunayBuilder<D> {
    /// Creates a unit-radius spherical builder from raw ambient coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`SphericalDelaunayConstructionError`] when any point cannot be
    /// normalized as a point on `S^D`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{SphericalDelaunayBuilder, SphericalDelaunayConstructionError};
    ///
    /// let triangulation = SphericalDelaunayBuilder::<2>::try_new([
    ///     [1.0, 1.0, 1.0],
    ///     [1.0, -1.0, -1.0],
    ///     [-1.0, 1.0, -1.0],
    ///     [-1.0, -1.0, 1.0],
    /// ])?.build()?;
    ///
    /// assert_eq!(triangulation.ambient_dimension(), 3);
    /// # Ok::<(), SphericalDelaunayConstructionError>(())
    /// ```
    pub fn try_new<I, P>(points: I) -> Result<Self, SphericalDelaunayConstructionError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<[f64]>,
    {
        Self::try_new_with_radius(points, 1.0)
    }

    /// Creates a spherical builder from raw ambient coordinates and radius.
    ///
    /// # Errors
    ///
    /// Returns [`SphericalDelaunayConstructionError`] when `radius` is invalid
    /// or any point cannot be normalized as a point on `S^D`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{SphericalDelaunayBuilder, SphericalDelaunayConstructionError};
    ///
    /// let triangulation = SphericalDelaunayBuilder::<2>::try_new_with_radius(
    ///     [
    ///         [1.0, 1.0, 1.0],
    ///         [1.0, -1.0, -1.0],
    ///         [-1.0, 1.0, -1.0],
    ///         [-1.0, -1.0, 1.0],
    ///     ],
    ///     2.0,
    /// )?.build()?;
    ///
    /// assert_eq!(triangulation.radius(), 2.0);
    /// # Ok::<(), SphericalDelaunayConstructionError>(())
    /// ```
    pub fn try_new_with_radius<I, P>(
        points: I,
        radius: f64,
    ) -> Result<Self, SphericalDelaunayConstructionError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<[f64]>,
    {
        let metric = SphericalMetric::<D>::try_new(radius)
            .map_err(|source| SphericalDelaunayConstructionError::Metric { source })?;
        let normalized = points
            .into_iter()
            .enumerate()
            .map(|(point_index, point)| {
                metric.canonicalize_slice(point.as_ref()).map_err(|source| {
                    SphericalDelaunayConstructionError::Point {
                        point_index,
                        source,
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            points: normalized,
            radius,
            construction_options: ConstructionOptions::default(),
        })
    }

    /// Creates a builder from already-normalized spherical points.
    ///
    /// # Errors
    ///
    /// Returns [`SphericalDelaunayConstructionError::Point`] when points carry
    /// inconsistent radii.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     SphericalDelaunayBuilder, SphericalDelaunayConstructionError,
    /// };
    /// use delaunay::prelude::topology::spaces::SphericalPoint;
    ///
    /// let raw = [
    ///     [1.0, 1.0, 1.0],
    ///     [1.0, -1.0, -1.0],
    ///     [-1.0, 1.0, -1.0],
    ///     [-1.0, -1.0, 1.0],
    /// ];
    /// let points = raw
    ///     .into_iter()
    ///     .enumerate()
    ///     .map(|(point_index, coords)| {
    ///         SphericalPoint::<2>::try_new(coords).map_err(|source| {
    ///             SphericalDelaunayConstructionError::Point {
    ///                 point_index,
    ///                 source,
    ///             }
    ///         })
    ///     })
    ///     .collect::<Result<Vec<_>, _>>()?;
    /// let triangulation = SphericalDelaunayBuilder::<2>::try_from_points(points)?.build()?;
    ///
    /// assert_eq!(triangulation.number_of_vertices(), 4);
    /// # Ok::<(), SphericalDelaunayConstructionError>(())
    /// ```
    pub fn try_from_points(
        points: Vec<SphericalPoint<D>>,
    ) -> Result<Self, SphericalDelaunayConstructionError> {
        let radius = points.first().map_or(1.0, SphericalPoint::radius);
        for (point_index, point) in points.iter().enumerate() {
            if point.radius().to_bits() != radius.to_bits() {
                return Err(SphericalDelaunayConstructionError::Point {
                    point_index,
                    source: SphericalPointError::MismatchedRadius {
                        expected: radius,
                        actual: point.radius(),
                    },
                });
            }
        }
        Ok(Self {
            points,
            radius,
            construction_options: ConstructionOptions::default(),
        })
    }

    /// Sets construction options for the ambient hull-duality build.
    ///
    /// Final Euclidean Delaunay enforcement is disabled internally because all
    /// spherical inputs are cospherical in the ambient space; the options still
    /// control insertion order, deduplication, and retry behavior.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     ConstructionOptions, SphericalDelaunayBuilder, SphericalDelaunayConstructionError,
    /// };
    ///
    /// let triangulation = SphericalDelaunayBuilder::<2>::try_new([
    ///     [1.0, 1.0, 1.0],
    ///     [1.0, -1.0, -1.0],
    ///     [-1.0, 1.0, -1.0],
    ///     [-1.0, -1.0, 1.0],
    /// ])?
    /// .construction_options(ConstructionOptions::default())
    /// .build()?;
    ///
    /// assert_eq!(triangulation.number_of_simplices(), 4);
    /// # Ok::<(), SphericalDelaunayConstructionError>(())
    /// ```
    #[must_use]
    pub const fn construction_options(mut self, options: ConstructionOptions) -> Self {
        self.construction_options = options;
        self
    }

    /// Builds the prototype spherical Delaunay triangulation.
    ///
    /// # Errors
    ///
    /// Returns [`SphericalDelaunayConstructionError`] if the intrinsic
    /// dimension is outside the prototype support envelope, ambient hull
    /// construction fails, the normalized points do not enclose the origin, or
    /// the constructed simplices fail spherical Level 3, Level 4, or Level 5
    /// validation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{SphericalDelaunayBuilder, SphericalDelaunayConstructionError};
    ///
    /// let triangulation = SphericalDelaunayBuilder::<2>::try_new([
    ///     [1.0, 1.0, 1.0],
    ///     [1.0, -1.0, -1.0],
    ///     [-1.0, 1.0, -1.0],
    ///     [-1.0, -1.0, 1.0],
    /// ])?.build()?;
    ///
    /// std::assert_matches!(triangulation.validate_delaunay(), Ok(()));
    /// # Ok::<(), SphericalDelaunayConstructionError>(())
    /// ```
    pub fn build(
        self,
    ) -> Result<SphericalDelaunayTriangulation<D>, SphericalDelaunayConstructionError> {
        match D {
            2 => self.build_with_ambient::<3>(),
            3 => self.build_with_ambient::<4>(),
            _ => Err(SphericalDelaunayConstructionError::UnsupportedDimension {
                dimension: D,
                max_validated_dimension: 3,
                tracking_issue: SPHERICAL_ROADMAP_ISSUE,
            }),
        }
    }

    /// Builds the ambient Euclidean hull and converts facets into simplices.
    ///
    /// This is the implementation of the public builder contract: normalized
    /// points on `S^D` are temporarily interpreted in `R^(D+1)`, the ambient
    /// convex hull is extracted, and hull facets become intrinsic spherical
    /// `D`-simplices after validation.
    fn build_with_ambient<const A: usize>(
        self,
    ) -> Result<SphericalDelaunayTriangulation<D>, SphericalDelaunayConstructionError> {
        if A != D + 1 {
            return Err(
                SphericalDelaunayConstructionError::AmbientDimensionMismatch {
                    dimension: D,
                    expected: D + 1,
                    actual: A,
                },
            );
        }
        let minimum = A + 1;
        if self.points.len() < minimum {
            return Err(SphericalDelaunayConstructionError::InsufficientVertices {
                dimension: D,
                minimum,
                actual: self.points.len(),
            });
        }

        let ambient_vertices = self.ambient_vertices::<A>()?;
        let ambient_options = self
            .construction_options
            .with_initial_simplex_strategy(InitialSimplexStrategy::Balanced)
            .without_final_delaunay_enforcement();
        let ambient: DelaunayTriangulation<_, usize, (), A> =
            DelaunayTriangulationBuilder::new(&ambient_vertices)
                .topology_guarantee(TopologyGuarantee::Pseudomanifold)
                .construction_options(ambient_options)
                .build()
                .map_err(
                    |source| SphericalDelaunayConstructionError::AmbientConstruction {
                        source: Box::new(source),
                    },
                )?;
        let hull =
            ConvexHull::try_from_triangulation(ambient.as_triangulation()).map_err(|source| {
                SphericalDelaunayConstructionError::ConvexHull {
                    source: Box::new(source),
                }
            })?;

        let origin = Point::<A>::try_new([0.0; A]).map_err(|source| {
            SphericalDelaunayConstructionError::AmbientOriginValidation { source }
        })?;
        if hull
            .is_point_outside(&origin, ambient.as_triangulation())
            .map_err(|source| SphericalDelaunayConstructionError::ConvexHull {
                source: Box::new(source),
            })?
        {
            return Err(SphericalDelaunayConstructionError::OriginOutsideConvexHull);
        }

        let facets = hull
            .try_facets(ambient.as_triangulation())
            .map_err(|source| SphericalDelaunayConstructionError::ConvexHull {
                source: Box::new(source),
            })?;
        let mut simplices = Vec::with_capacity(hull.number_of_facets());
        for (simplex_index, facet_result) in facets.enumerate() {
            let facet = facet_result
                .map_err(|source| SphericalDelaunayConstructionError::Facet { source })?;
            let mut simplex_vertices = Vec::with_capacity(D + 1);
            for vertex in facet.vertices() {
                let original_index = vertex.data().copied().ok_or(
                    SphericalDelaunayConstructionError::MissingAmbientVertexIndex { simplex_index },
                )?;
                simplex_vertices.push(original_index);
            }
            let simplex = SphericalSimplex::try_new(simplex_vertices, self.points.len()).map_err(
                |source| SphericalDelaunayConstructionError::InvalidHullSimplex {
                    simplex_index,
                    source,
                },
            )?;
            simplices.push(simplex);
        }
        if simplices.is_empty() {
            return Err(SphericalDelaunayConstructionError::EmptyHull);
        }

        let spherical = SphericalDelaunayTriangulation {
            points: self.points,
            simplices,
            radius: self.radius,
        };
        spherical.validate_topology().map_err(|source| {
            SphericalDelaunayConstructionError::TopologyValidation {
                source: Box::new(source),
            }
        })?;
        spherical.validate_realization().map_err(|source| {
            SphericalDelaunayConstructionError::RealizationValidation {
                source: Box::new(source),
            }
        })?;
        spherical.validate_delaunay().map_err(|source| {
            SphericalDelaunayConstructionError::DelaunayValidation {
                source: Box::new(source),
            }
        })?;
        Ok(spherical)
    }

    /// Converts normalized spherical points into ambient Euclidean vertices for
    /// the temporary hull-duality build.
    fn ambient_vertices<const A: usize>(
        &self,
    ) -> Result<Vec<Vertex<usize, A>>, SphericalDelaunayConstructionError> {
        self.points
            .iter()
            .enumerate()
            .map(|(point_index, point)| {
                let coords = ambient_array_from_slice::<A>(point.coords()).map_err(|source| {
                    SphericalDelaunayConstructionError::AmbientCoordinateArray {
                        point_index,
                        source,
                    }
                })?;
                vertex!(coords; data = point_index).map_err(|source| {
                    SphericalDelaunayConstructionError::AmbientVertex {
                        point_index,
                        source,
                    }
                })
            })
            .collect()
    }
}

/// Builds finite, distinct coordinates for the temporary abstract topology TDS.
///
/// These coordinates have no realization meaning. They only let the existing TDS
/// element constructors carry vertex identities through intrinsic PL topology
/// validators that operate on the abstract simplicial complex.
fn synthetic_topology_coordinates<const D: usize>(
    point_index: usize,
) -> Result<[f64; D], CoordinateConversionError> {
    let base = safe_usize_to_scalar(point_index)? + 1.0;
    let mut coords = [0.0; D];
    for (axis, coord) in coords.iter_mut().enumerate() {
        *coord = safe_usize_to_scalar(axis)?.mul_add(1.0e-3, base);
    }
    Ok(coords)
}

/// Computes `||coords|| / radius` without squaring very large radii.
fn scaled_norm_over_radius(coords: &[f64], radius: f64) -> f64 {
    let max_abs = coords
        .iter()
        .fold(0.0_f64, |max_abs, &coord| max_abs.max(coord.abs()));
    if max_abs == 0.0 {
        return 0.0;
    }
    let scaled_norm = coords
        .iter()
        .fold(0.0, |acc, &coord| {
            let scaled = coord / max_abs;
            scaled.mul_add(scaled, acc)
        })
        .sqrt();
    (max_abs / radius) * scaled_norm
}

/// Computes a best-effort Euclidean norm for validation diagnostics.
fn scaled_euclidean_norm(coords: &[f64]) -> f64 {
    let max_abs = coords
        .iter()
        .fold(0.0_f64, |max_abs, &coord| max_abs.max(coord.abs()));
    if max_abs == 0.0 {
        return 0.0;
    }
    let scaled_norm = coords
        .iter()
        .fold(0.0, |acc, &coord| {
            let scaled = coord / max_abs;
            scaled.mul_add(scaled, acc)
        })
        .sqrt();
    max_abs * scaled_norm
}

/// Wraps TDS invariant failures from the synthetic topology bridge.
fn intrinsic_tds_error(source: TdsError) -> SphericalDelaunayValidationError {
    intrinsic_error(InvariantError::Tds(source))
}

/// Wraps manifold invariant failures from the synthetic topology bridge.
fn intrinsic_manifold_error(source: ManifoldError) -> SphericalDelaunayValidationError {
    intrinsic_error(InvariantError::from(source))
}

/// Wraps generic triangulation invariant failures as spherical Level 3 errors.
fn intrinsic_topology_error(
    source: TriangulationValidationError,
) -> SphericalDelaunayValidationError {
    intrinsic_error(InvariantError::Triangulation(source))
}

/// Converts topology-support helper failures into the shared invariant tree.
fn intrinsic_topology_support_error(source: TopologyError) -> SphericalDelaunayValidationError {
    let invariant = match source {
        TopologyError::FacetMapBuild { source }
        | TopologyError::BoundaryFacetEnumeration { source }
        | TopologyError::BoundaryFacetCount { source } => InvariantError::Tds(source),
        TopologyError::BoundaryFacetSimplexAccess { source } => InvariantError::Tds(source.into()),
        TopologyError::BoundaryClassification { source } => InvariantError::from(*source),
    };
    intrinsic_error(invariant)
}

/// Boxes the shared invariant tree so spherical validation errors stay compact.
fn intrinsic_error(source: InvariantError) -> SphericalDelaunayValidationError {
    SphericalDelaunayValidationError::IntrinsicTopology {
        source: Box::new(source),
    }
}

/// Errors emitted while validating spherical simplices.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum SphericalSimplexError {
    /// A simplex did not contain exactly `D + 1` vertices.
    #[error("spherical {dimension}D simplex requires {expected} vertices, got {actual}")]
    InvalidArity {
        /// Intrinsic simplex dimension.
        dimension: usize,
        /// Required vertex count.
        expected: usize,
        /// Actual vertex count.
        actual: usize,
    },

    /// A simplex referenced a missing vertex.
    #[error(
        "spherical simplex vertex index {vertex_index} is out of bounds for {vertex_count} vertices"
    )]
    VertexIndexOutOfBounds {
        /// Invalid vertex index.
        vertex_index: usize,
        /// Number of vertices in the point set.
        vertex_count: usize,
    },

    /// A simplex repeated a vertex index.
    #[error("spherical simplex contains duplicate vertex index {vertex_index}")]
    DuplicateVertex {
        /// Duplicated vertex index.
        vertex_index: usize,
    },
}

/// Prototype spherical construction errors.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum SphericalDelaunayConstructionError {
    /// Metric configuration failed.
    #[error("spherical metric configuration failed: {source}")]
    Metric {
        /// Underlying metric configuration error.
        #[source]
        source: SphericalPointError,
    },

    /// One input point failed canonicalization.
    #[error("spherical point {point_index} failed canonicalization: {source}")]
    Point {
        /// Zero-based input point index.
        point_index: usize,
        /// Point canonicalization error.
        #[source]
        source: SphericalPointError,
    },

    /// The prototype is release-validated only through 3D.
    #[error(
        "spherical Delaunay construction is release-validated only up to {max_validated_dimension}D; {dimension}D support is tracked by issue #{tracking_issue}"
    )]
    UnsupportedDimension {
        /// Requested intrinsic dimension.
        dimension: usize,
        /// Highest release-validated intrinsic dimension.
        max_validated_dimension: usize,
        /// Tracking issue for broader support.
        tracking_issue: u32,
    },

    /// Internal ambient dispatch did not match the intrinsic dimension.
    #[error(
        "ambient dimension dispatch mismatch for S^{dimension}: expected {expected}, got {actual}"
    )]
    AmbientDimensionMismatch {
        /// Intrinsic dimension.
        dimension: usize,
        /// Expected ambient dimension.
        expected: usize,
        /// Actual ambient dimension.
        actual: usize,
    },

    /// Too few vertices were supplied for an ambient convex hull.
    #[error(
        "spherical S^{dimension} construction requires at least {minimum} points, got {actual}"
    )]
    InsufficientVertices {
        /// Intrinsic dimension.
        dimension: usize,
        /// Minimum point count.
        minimum: usize,
        /// Actual point count.
        actual: usize,
    },

    /// A normalized point could not be converted to an ambient fixed-size array.
    #[error("spherical point {point_index} could not be converted to an ambient array: {source}")]
    AmbientCoordinateArray {
        /// Zero-based input point index.
        point_index: usize,
        /// Underlying arity error.
        #[source]
        source: SphericalPointError,
    },

    /// Ambient Euclidean vertex construction failed.
    #[error("ambient vertex {point_index} failed construction: {source}")]
    AmbientVertex {
        /// Zero-based input point index.
        point_index: usize,
        /// Underlying coordinate conversion error.
        #[source]
        source: CoordinateConversionError,
    },

    /// Ambient origin point construction failed.
    #[error("ambient origin point failed validation: {source}")]
    AmbientOriginValidation {
        /// Underlying coordinate validation error.
        #[source]
        source: CoordinateValidationError,
    },

    /// Ambient construction failed before convex-hull extraction.
    #[error("ambient convex-hull dual construction failed: {source}")]
    AmbientConstruction {
        /// Underlying ambient Delaunay construction error.
        #[source]
        source: Box<DelaunayTriangulationConstructionError>,
    },

    /// Convex-hull extraction or query failed.
    #[error("ambient convex-hull operation failed: {source}")]
    ConvexHull {
        /// Underlying convex-hull error.
        #[source]
        source: Box<ConvexHullConstructionError>,
    },

    /// A hull facet view failed to resolve.
    #[error("ambient hull facet failed to resolve: {source}")]
    Facet {
        /// Underlying facet error.
        #[source]
        source: FacetError,
    },

    /// Normalized points did not surround the sphere center.
    #[error("normalized spherical points do not enclose the origin in ambient space")]
    OriginOutsideConvexHull,

    /// Convex-hull extraction returned no facets.
    #[error("ambient convex hull produced no facets")]
    EmptyHull,

    /// Ambient vertex metadata did not contain the original spherical point index.
    #[error("ambient hull simplex {simplex_index} was missing an original spherical vertex index")]
    MissingAmbientVertexIndex {
        /// Hull simplex index.
        simplex_index: usize,
    },

    /// A hull facet could not be converted into a spherical simplex.
    #[error("ambient hull simplex {simplex_index} is not a valid spherical simplex: {source}")]
    InvalidHullSimplex {
        /// Hull simplex index.
        simplex_index: usize,
        /// Underlying simplex validation error.
        #[source]
        source: SphericalSimplexError,
    },

    /// Supported topology validation failed after construction.
    #[error("spherical topology validation failed after construction: {source}")]
    TopologyValidation {
        /// Underlying validation error.
        #[source]
        source: Box<SphericalDelaunayValidationError>,
    },

    /// Level 4 spherical realization validation failed after construction.
    #[error("spherical realization validation failed after construction: {source}")]
    RealizationValidation {
        /// Underlying validation error.
        #[source]
        source: Box<SphericalDelaunayValidationError>,
    },

    /// Level 5 spherical Delaunay validation failed after construction.
    #[error("spherical Delaunay validation failed after construction: {source}")]
    DelaunayValidation {
        /// Underlying validation error.
        #[source]
        source: Box<SphericalDelaunayValidationError>,
    },
}

/// Spherical validation layer named in prototype diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SphericalValidationLayer {
    /// Level 4 curved realization validation.
    Realization,
    /// Level 5 spherical Delaunay predicate validation.
    Delaunay,
}

impl fmt::Display for SphericalValidationLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Realization => f.write_str("Level 4 Spherical Realization"),
            Self::Delaunay => f.write_str("Level 5 Geometric Predicates"),
        }
    }
}

/// Prototype spherical validation errors.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum SphericalDelaunayValidationError {
    /// Metric configuration failed.
    #[error("spherical metric configuration failed: {source}")]
    Metric {
        /// Underlying metric configuration error.
        #[source]
        source: SphericalPointError,
    },

    /// Geodesic distance computation failed between stored vertices.
    #[error("geodesic distance between spherical vertices {left} and {right} failed: {source}")]
    GeodesicDistance {
        /// Left vertex index.
        left: usize,
        /// Right vertex index.
        right: usize,
        /// Underlying distance error.
        #[source]
        source: SphericalPointError,
    },

    /// A requested vertex index was outside the point set.
    #[error("vertex index {vertex_index} is out of bounds for {vertex_count} spherical vertices")]
    VertexIndexOutOfBounds {
        /// Invalid vertex index.
        vertex_index: usize,
        /// Point-set size.
        vertex_count: usize,
    },

    /// No simplices are present.
    #[error("spherical triangulation contains no simplices")]
    NoSimplices,

    /// A stored point has the wrong ambient coordinate count.
    #[error("spherical point {point_index} has {actual} ambient coordinates, expected {expected}")]
    InvalidPointAmbientDimension {
        /// Point index.
        point_index: usize,
        /// Expected ambient coordinate count.
        expected: usize,
        /// Actual ambient coordinate count.
        actual: usize,
    },

    /// A stored point is not on the configured sphere.
    #[error("spherical point {point_index} has norm {norm:?}, expected radius {radius:?}")]
    PointNotOnSphere {
        /// Point index.
        point_index: usize,
        /// Expected radius.
        radius: f64,
        /// Observed norm.
        norm: f64,
    },

    /// A stored simplex is malformed.
    #[error("spherical simplex {simplex_index} is invalid: {source}")]
    InvalidSimplex {
        /// Simplex index.
        simplex_index: usize,
        /// Underlying simplex error.
        #[source]
        source: SphericalSimplexError,
    },

    /// A simplex appears more than once in the complex.
    #[error(
        "spherical simplex {simplex_index} duplicates simplex {previous_simplex_index} with vertices {simplex_vertices:?}"
    )]
    DuplicateSimplex {
        /// Duplicate simplex index.
        simplex_index: usize,
        /// First simplex with the same sorted vertices.
        previous_simplex_index: usize,
        /// Sorted duplicate simplex vertices.
        simplex_vertices: Vec<usize>,
    },

    /// A stored spherical vertex is not incident to any simplex.
    #[error("spherical vertex {vertex_index} is not incident to any simplex")]
    UnusedVertex {
        /// Unused vertex index.
        vertex_index: usize,
    },

    /// A codimension-1 face is not incident to exactly two simplices.
    #[error("spherical facet {facet_vertices:?} has incident count {incident_count}, expected 2")]
    InvalidFacetIncidence {
        /// Sorted facet vertex indices.
        facet_vertices: Vec<usize>,
        /// Observed incident simplex count.
        incident_count: usize,
    },

    /// Synthetic abstract-topology vertex construction failed.
    #[error("abstract spherical topology vertex {point_index} failed construction: {source}")]
    AbstractTopologyVertex {
        /// Spherical point index represented by this abstract vertex.
        point_index: usize,
        /// Underlying coordinate conversion error.
        #[source]
        source: CoordinateConversionError,
    },

    /// Synthetic abstract-topology vertex insertion failed.
    #[error("abstract spherical topology vertex {point_index} failed insertion: {source}")]
    AbstractTopologyVertexInsertion {
        /// Spherical point index represented by this abstract vertex.
        point_index: usize,
        /// Underlying TDS construction error.
        #[source]
        source: Box<TdsConstructionError>,
    },

    /// Synthetic abstract-topology simplex construction failed.
    #[error("abstract spherical topology simplex {simplex_index} failed construction: {source}")]
    AbstractTopologySimplex {
        /// Spherical simplex index represented by this abstract simplex.
        simplex_index: usize,
        /// Underlying simplex validation error.
        #[source]
        source: SimplexValidationError,
    },

    /// Synthetic abstract-topology simplex insertion failed.
    #[error("abstract spherical topology simplex {simplex_index} failed insertion: {source}")]
    AbstractTopologySimplexInsertion {
        /// Spherical simplex index represented by this abstract simplex.
        simplex_index: usize,
        /// Underlying TDS construction error.
        #[source]
        source: Box<TdsConstructionError>,
    },

    /// Synthetic abstract-topology mutation failed while preparing PL checks.
    #[error("abstract spherical topology mutation failed: {source}")]
    AbstractTopologyMutation {
        /// Underlying TDS mutation error.
        #[source]
        source: TdsMutationError,
    },

    /// Intrinsic PL-topology validation failed.
    #[error("spherical intrinsic PL topology validation failed: {source}")]
    IntrinsicTopology {
        /// Underlying invariant violation from the shared topology validators.
        #[source]
        source: Box<InvariantError>,
    },

    /// A validation layer is intentionally unsupported in the prototype.
    #[error(
        "{layer} is not defined for the spherical prototype in {dimension}D; full support is tracked by issue #{tracking_issue}"
    )]
    UnsupportedLayer {
        /// Unsupported layer.
        layer: SphericalValidationLayer,
        /// Intrinsic dimension.
        dimension: usize,
        /// Tracking issue for the full contract.
        tracking_issue: u32,
    },

    /// Internal ambient dispatch did not match the intrinsic dimension.
    #[error(
        "ambient dimension dispatch mismatch for S^{dimension}: expected {expected}, got {actual}"
    )]
    AmbientDimensionMismatch {
        /// Intrinsic dimension.
        dimension: usize,
        /// Expected ambient dimension.
        expected: usize,
        /// Actual ambient dimension.
        actual: usize,
    },

    /// A normalized point could not be converted to an ambient fixed-size array.
    #[error("spherical point {point_index} could not be converted to an ambient array: {source}")]
    AmbientCoordinateArray {
        /// Zero-based point index.
        point_index: usize,
        /// Underlying arity error.
        #[source]
        source: SphericalPointError,
    },

    /// Ambient point validation failed during spherical validation.
    #[error(
        "ambient point {point_index:?} failed validation during spherical validation: {source}"
    )]
    AmbientPointValidation {
        /// Zero-based point index, or `None` for the origin.
        point_index: Option<usize>,
        /// Underlying coordinate validation error.
        #[source]
        source: CoordinateValidationError,
    },

    /// Orientation failed for a candidate ambient supporting hyperplane.
    #[error(
        "orientation failed for spherical simplex {simplex_index} with test point {point_index:?}: {source}"
    )]
    SimplexOrientation {
        /// Simplex index.
        simplex_index: usize,
        /// Test point index, or `None` for the origin.
        point_index: Option<usize>,
        /// Underlying orientation error.
        #[source]
        source: CoordinateConversionError,
    },

    /// The ambient hyperplane through a simplex also contains the sphere center.
    #[error(
        "spherical simplex {simplex_index} has a supporting hyperplane through the origin: {simplex_vertices:?}"
    )]
    SimplexHyperplaneContainsOrigin {
        /// Simplex index.
        simplex_index: usize,
        /// Simplex vertex indices.
        simplex_vertices: Vec<usize>,
    },

    /// A simplex is not an ambient convex-hull supporting facet.
    #[error(
        "spherical simplex {simplex_index} is not a supporting hull facet: point {point_index} has side {point_side:?}, origin side is {origin_side:?}, vertices {simplex_vertices:?}"
    )]
    NonSupportingSimplex {
        /// Simplex index.
        simplex_index: usize,
        /// Simplex vertex indices.
        simplex_vertices: Vec<usize>,
        /// Point proving the simplex is not supporting.
        point_index: usize,
        /// Orientation side of the origin.
        origin_side: Orientation,
        /// Orientation side of the violating point.
        point_side: Orientation,
    },
}

#[cfg(test)]
mod tests {
    use std::assert_matches;

    use super::*;

    fn spherical_triangle(vertices: [usize; 3], vertex_count: usize) -> SphericalSimplex<2> {
        SphericalSimplex::<2>::try_new(vertices.to_vec(), vertex_count)
            .expect("fixture simplex should be well-formed")
    }

    fn tetrahedron_boundary_points() -> Vec<SphericalPoint<2>> {
        tetrahedron_boundary_points_with_radius(1.0)
    }

    fn tetrahedron_boundary_points_with_radius(radius: f64) -> Vec<SphericalPoint<2>> {
        vec![
            SphericalPoint::<2>::try_new_with_radius([1.0, 1.0, 1.0], radius)
                .expect("finite point"),
            SphericalPoint::<2>::try_new_with_radius([1.0, -1.0, -1.0], radius)
                .expect("finite point"),
            SphericalPoint::<2>::try_new_with_radius([-1.0, 1.0, -1.0], radius)
                .expect("finite point"),
            SphericalPoint::<2>::try_new_with_radius([-1.0, -1.0, 1.0], radius)
                .expect("finite point"),
        ]
    }

    fn tetrahedron_boundary_simplices(vertex_count: usize) -> Vec<SphericalSimplex<2>> {
        vec![
            spherical_triangle([0, 1, 2], vertex_count),
            spherical_triangle([0, 1, 3], vertex_count),
            spherical_triangle([0, 2, 3], vertex_count),
            spherical_triangle([1, 2, 3], vertex_count),
        ]
    }

    #[test]
    fn level3_rejects_empty_spherical_complex() {
        let triangulation = SphericalDelaunayTriangulation::<2> {
            points: tetrahedron_boundary_points(),
            simplices: Vec::new(),
            radius: 1.0,
        };

        assert_matches!(
            triangulation.validate_topology(),
            Err(SphericalDelaunayValidationError::NoSimplices)
        );
    }

    #[test]
    fn level3_rejects_duplicate_spherical_simplices() {
        let points = tetrahedron_boundary_points();
        let vertex_count = points.len();
        let triangulation = SphericalDelaunayTriangulation::<2> {
            points,
            simplices: vec![
                spherical_triangle([0, 1, 2], vertex_count),
                spherical_triangle([2, 1, 0], vertex_count),
            ],
            radius: 1.0,
        };

        assert_matches!(
            triangulation.validate_topology(),
            Err(SphericalDelaunayValidationError::DuplicateSimplex {
                simplex_index: 1,
                previous_simplex_index: 0,
                simplex_vertices,
            }) if simplex_vertices == vec![0, 1, 2]
        );
    }

    #[test]
    fn level3_rejects_malformed_stored_spherical_simplex() {
        let points = tetrahedron_boundary_points();
        let triangulation = SphericalDelaunayTriangulation::<2> {
            points,
            simplices: vec![SphericalSimplex {
                vertices: vec![0, 1, 4],
            }],
            radius: 1.0,
        };

        assert_matches!(
            triangulation.validate_topology(),
            Err(SphericalDelaunayValidationError::InvalidSimplex {
                simplex_index: 0,
                source: SphericalSimplexError::VertexIndexOutOfBounds {
                    vertex_index: 4,
                    vertex_count: 4,
                },
            })
        );
    }

    #[test]
    fn level3_rejects_open_spherical_facet_incidence() {
        let points = tetrahedron_boundary_points();
        let vertex_count = points.len();
        let triangulation = SphericalDelaunayTriangulation::<2> {
            points,
            simplices: vec![spherical_triangle([0, 1, 2], vertex_count)],
            radius: 1.0,
        };

        assert_matches!(
            triangulation.validate_topology(),
            Err(SphericalDelaunayValidationError::InvalidFacetIncidence {
                incident_count: 1,
                ..
            })
        );
    }

    #[test]
    fn level3_rejects_unused_spherical_vertices() {
        let mut points = tetrahedron_boundary_points();
        points.push(SphericalPoint::<2>::try_new([1.0, 0.0, 0.0]).expect("finite point"));
        let vertex_count = points.len();
        let triangulation = SphericalDelaunayTriangulation::<2> {
            points,
            simplices: tetrahedron_boundary_simplices(vertex_count),
            radius: 1.0,
        };

        assert_matches!(
            triangulation.validate_topology(),
            Err(SphericalDelaunayValidationError::UnusedVertex { vertex_index: 4 })
        );
    }

    #[test]
    fn level3_rejects_points_not_on_configured_radius() {
        let points = tetrahedron_boundary_points();
        let vertex_count = points.len();
        let triangulation = SphericalDelaunayTriangulation::<2> {
            points,
            simplices: tetrahedron_boundary_simplices(vertex_count),
            radius: 2.0,
        };

        assert_matches!(
            triangulation.validate_topology(),
            Err(SphericalDelaunayValidationError::PointNotOnSphere {
                point_index: 0,
                radius,
                norm,
            }) if radius.to_bits() == 2.0_f64.to_bits() && (norm - 1.0).abs() < 1.0e-12
        );
    }

    #[test]
    fn distance_reports_typed_geodesic_failure() {
        let unit =
            SphericalPoint::<2>::try_new([1.0, 0.0, 0.0]).expect("unit point should normalize");
        let radius_two = SphericalPoint::<2>::try_new_with_radius([0.0, 1.0, 0.0], 2.0)
            .expect("radius-two point should normalize");
        let triangulation = SphericalDelaunayTriangulation::<2> {
            points: vec![unit, radius_two],
            simplices: Vec::new(),
            radius: 1.0,
        };

        assert_matches!(
            triangulation.distance_between_vertices(0, 1),
            Err(SphericalDelaunayValidationError::GeodesicDistance {
                left: 0,
                right: 1,
                source: SphericalPointError::MismatchedRadius { expected, actual },
            }) if expected.to_bits() == 1.0_f64.to_bits()
                && actual.to_bits() == 2.0_f64.to_bits()
        );
    }

    #[test]
    fn level3_accepts_large_radius_without_norm_overflow() {
        let radius = f64::MAX / 4.0;
        let points = tetrahedron_boundary_points_with_radius(radius);
        let vertex_count = points.len();
        let triangulation = SphericalDelaunayTriangulation::<2> {
            points,
            simplices: tetrahedron_boundary_simplices(vertex_count),
            radius,
        };

        triangulation
            .validate_topology()
            .expect("scaled norm check should avoid squaring large finite radii");
    }

    #[test]
    fn level4_rejects_s4_after_intrinsic_topology_passes() {
        let points = vec![
            SphericalPoint::<4>::try_new([1.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point"),
            SphericalPoint::<4>::try_new([0.0, 1.0, 0.0, 0.0, 0.0]).expect("finite point"),
            SphericalPoint::<4>::try_new([0.0, 0.0, 1.0, 0.0, 0.0]).expect("finite point"),
            SphericalPoint::<4>::try_new([0.0, 0.0, 0.0, 1.0, 0.0]).expect("finite point"),
            SphericalPoint::<4>::try_new([0.0, 0.0, 0.0, 0.0, 1.0]).expect("finite point"),
            SphericalPoint::<4>::try_new([-1.0, -1.0, -1.0, -1.0, -1.0]).expect("finite point"),
        ];
        let vertex_count = points.len();
        let simplices = (0..vertex_count)
            .map(|omitted| {
                let vertices = (0..vertex_count)
                    .filter(|&vertex_index| vertex_index != omitted)
                    .collect();
                SphericalSimplex::<4>::try_new(vertices, vertex_count)
                    .expect("boundary simplex should be well formed")
            })
            .collect();
        let triangulation = SphericalDelaunayTriangulation::<4> {
            points,
            simplices,
            radius: 1.0,
        };

        triangulation
            .validate_topology()
            .expect("5-simplex boundary is intrinsically S4");
        assert_matches!(
            triangulation.validate_realization(),
            Err(SphericalDelaunayValidationError::UnsupportedLayer {
                layer: SphericalValidationLayer::Realization,
                dimension: 4,
                tracking_issue: SPHERICAL_ROADMAP_ISSUE,
            })
        );
    }

    #[test]
    fn level5_rejects_topological_flip_that_is_not_a_hull_facet() {
        let points = vec![
            SphericalPoint::<2>::try_new([1.0, 1.0, 1.0]).expect("finite point"),
            SphericalPoint::<2>::try_new([1.0, -1.0, -1.0]).expect("finite point"),
            SphericalPoint::<2>::try_new([-1.0, 1.0, -1.0]).expect("finite point"),
            SphericalPoint::<2>::try_new([-1.0, -1.0, 1.0]).expect("finite point"),
            SphericalPoint::<2>::try_new([1.0, 0.9, -0.8]).expect("finite point"),
        ];
        let vertex_count = points.len();
        let triangulation = SphericalDelaunayTriangulation::<2> {
            points,
            simplices: vec![
                spherical_triangle([0, 3, 4], vertex_count),
                spherical_triangle([1, 3, 4], vertex_count),
                spherical_triangle([1, 2, 4], vertex_count),
                spherical_triangle([2, 0, 4], vertex_count),
                spherical_triangle([0, 2, 3], vertex_count),
                spherical_triangle([1, 2, 3], vertex_count),
            ],
            radius: 1.0,
        };

        triangulation
            .validate_topology()
            .expect("flipped fixture remains a closed PL sphere");
        triangulation
            .validate_realization()
            .expect("flipped fixture remains a spherical realization");
        assert_matches!(
            triangulation.validate_delaunay(),
            Err(SphericalDelaunayValidationError::NonSupportingSimplex { .. })
        );
    }

    #[test]
    fn level4_rejects_closed_topology_with_simplex_hyperplane_through_origin() {
        let points = vec![
            SphericalPoint::<2>::try_new([1.0, 0.0, 0.0]).expect("finite point"),
            SphericalPoint::<2>::try_new([-1.0, 0.0, 0.0]).expect("finite point"),
            SphericalPoint::<2>::try_new([0.0, 1.0, 0.0]).expect("finite point"),
            SphericalPoint::<2>::try_new([0.0, 0.0, 1.0]).expect("finite point"),
        ];
        let vertex_count = points.len();
        let triangulation = SphericalDelaunayTriangulation::<2> {
            points,
            simplices: vec![
                spherical_triangle([0, 1, 2], vertex_count),
                spherical_triangle([0, 1, 3], vertex_count),
                spherical_triangle([0, 2, 3], vertex_count),
                spherical_triangle([1, 2, 3], vertex_count),
            ],
            radius: 1.0,
        };

        triangulation
            .validate_topology()
            .expect("fixture is combinatorially a closed sphere");
        assert_matches!(
            triangulation.validate_realization(),
            Err(
                SphericalDelaunayValidationError::SimplexHyperplaneContainsOrigin {
                    simplex_index: 0,
                    ..
                }
            )
        );
    }

    #[test]
    fn level3_rejects_closed_facet_counts_with_disconnected_vertex_wedge() {
        let points = vec![
            SphericalPoint::<2>::try_new([1.0, 0.0, 0.0]).expect("finite point"),
            SphericalPoint::<2>::try_new([0.0, 1.0, 0.0]).expect("finite point"),
            SphericalPoint::<2>::try_new([0.0, 0.0, 1.0]).expect("finite point"),
            SphericalPoint::<2>::try_new([-1.0, -1.0, -1.0]).expect("finite point"),
            SphericalPoint::<2>::try_new([0.0, -1.0, 0.0]).expect("finite point"),
            SphericalPoint::<2>::try_new([0.0, 0.0, -1.0]).expect("finite point"),
            SphericalPoint::<2>::try_new([-1.0, 1.0, 1.0]).expect("finite point"),
        ];
        let vertex_count = points.len();
        let triangulation = SphericalDelaunayTriangulation::<2> {
            points,
            simplices: vec![
                spherical_triangle([0, 1, 2], vertex_count),
                spherical_triangle([0, 1, 3], vertex_count),
                spherical_triangle([0, 2, 3], vertex_count),
                spherical_triangle([1, 2, 3], vertex_count),
                spherical_triangle([0, 4, 5], vertex_count),
                spherical_triangle([0, 4, 6], vertex_count),
                spherical_triangle([0, 5, 6], vertex_count),
                spherical_triangle([4, 5, 6], vertex_count),
            ],
            radius: 1.0,
        };

        assert_matches!(
            triangulation.validate_topology(),
            Err(SphericalDelaunayValidationError::IntrinsicTopology { source })
                if matches!(
                    source.as_ref(),
                    InvariantError::Triangulation(
                        TriangulationValidationError::Disconnected { simplex_count: 8 }
                    )
                )
        );
    }
}
