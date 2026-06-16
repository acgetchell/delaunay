//! Generic triangulation combining kernel and combinatorial data structure.
//!
//! Following CGAL's architecture, the `Triangulation` struct combines:
//! - A geometric `Kernel` for predicates
//! - A purely combinatorial `Tds` for topology
//!
//! This layer provides geometric operations while delegating topology to Tds.
//!
//! Validation policy, topology guarantees, and validation passes are implemented
//! in [`crate::prelude::validation`].
//!

#![forbid(unsafe_code)]

use crate::core::tds::{SimplexKey, Tds, VertexKey};
use crate::core::validation::{TopologyGuarantee, ValidationPolicy};
use crate::geometry::kernel::Kernel;
use crate::topology::traits::topological_space::GlobalTopology;

/// Generic triangulation combining kernel and data structure.
///
/// # Type Parameters
/// - `K`: Geometric kernel implementing predicates
/// - `U`: User data type for vertices
/// - `V`: User data type for simplices
/// - `D`: Dimension of the triangulation
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::triangulation::{FastKernel, Triangulation};
///
/// let tri: Triangulation<FastKernel<f64>, (), (), 3> =
///     Triangulation::new_empty(FastKernel::new());
/// assert_eq!(tri.number_of_vertices(), 0);
/// ```
#[derive(Clone, Debug)]
pub struct Triangulation<K, U, V, const D: usize> {
    /// The geometric kernel for predicates.
    pub(crate) kernel: K,
    /// The combinatorial triangulation data structure.
    pub(crate) tds: Tds<U, V, D>,
    /// Runtime metadata describing the global topological space represented by this triangulation.
    pub(crate) global_topology: GlobalTopology<D>,
    pub(crate) validation_policy: ValidationPolicy,
    pub(crate) topology_guarantee: TopologyGuarantee,
}

// =============================================================================
// Basic Accessors (Minimal Bounds)
// =============================================================================

impl<K, U, V, const D: usize> Triangulation<K, U, V, D>
where
    K: Kernel<D>,
{
    /// Create an empty triangulation with the given kernel.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::{FastKernel, Triangulation};
    ///
    /// let tri: Triangulation<FastKernel<f64>, (), (), 3> =
    ///     Triangulation::new_empty(FastKernel::new());
    /// assert_eq!(tri.number_of_vertices(), 0);
    /// assert_eq!(tri.number_of_simplices(), 0);
    /// assert_eq!(tri.dim(), -1); // Empty triangulation has dimension -1
    /// ```
    #[must_use]
    pub fn new_empty(kernel: K) -> Self {
        Self {
            kernel,
            tds: Tds::empty(),
            global_topology: GlobalTopology::DEFAULT,
            validation_policy: TopologyGuarantee::DEFAULT.default_validation_policy(),
            topology_guarantee: TopologyGuarantee::DEFAULT,
        }
    }

    #[cfg(test)]
    #[inline]
    #[expect(
        clippy::missing_const_for_fn,
        reason = "test-only constructor is not a pure math helper"
    )]
    pub(crate) fn new_with_tds(kernel: K, tds: Tds<U, V, D>) -> Self {
        Self {
            kernel,
            tds,
            global_topology: GlobalTopology::DEFAULT,
            validation_policy: TopologyGuarantee::DEFAULT.default_validation_policy(),
            topology_guarantee: TopologyGuarantee::DEFAULT,
        }
    }

    /// Sets the auxiliary data on a returning the previous value.
    ///
    /// Delegates to [`Tds::set_vertex_data`]. This is a safe O(1) operation
    /// that does not affect geometry, topology, or Delaunay invariants.
    ///
    /// # Returns
    ///
    /// `None` if the key is not found. `Some(previous)` where `previous` is
    /// the old `Option<U>` value if the key exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, Vertex,
    /// };
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error("triangulation unexpectedly contains no vertices")]
    /// #     MissingVertex,
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices: [Vertex<i32, 2>; 3] = [
    ///     delaunay::prelude::Vertex::<_, _>::try_new_with_data([0.0, 0.0], 10i32)?,
    ///     delaunay::prelude::Vertex::<_, _>::try_new_with_data([1.0, 0.0], 20)?,
    ///     delaunay::prelude::Vertex::<_, _>::try_new_with_data([0.0, 1.0], 30)?,
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let key = dt.vertices().next().ok_or(ExampleError::MissingVertex)?.0;
    /// let prev = dt.set_vertex_data(key, Some(99));
    /// assert!(prev.is_some());
    ///
    /// // Clear data
    /// let prev = dt.set_vertex_data(key, None);
    /// assert_eq!(prev, Some(Some(99)));
    /// let vertex = dt.tds().vertex(key).ok_or(ExampleError::MissingVertex)?;
    /// assert_eq!(vertex.data(), None);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn set_vertex_data(&mut self, key: VertexKey, data: Option<U>) -> Option<Option<U>> {
        self.tds.set_vertex_data(key, data)
    }

    /// Sets the auxiliary data on a simplex, returning the previous value.
    ///
    /// Delegates to [`Tds::set_simplex_data`]. This is a safe O(1) operation
    /// that does not affect geometry, topology, or Delaunay invariants.
    ///
    /// # Returns
    ///
    /// `None` if the key is not found. `Some(previous)` where `previous` is
    /// the old `Option<V>` value if the key exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
    /// };
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error("triangulation unexpectedly contains no simplices")]
    /// #     MissingSimplex,
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0])?,
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<i32>()?;
    /// let key = dt.simplices().next().ok_or(ExampleError::MissingSimplex)?.0;
    /// let prev = dt.set_simplex_data(key, Some(42));
    /// assert_eq!(prev, Some(None));
    ///
    /// // Clear data
    /// let prev = dt.set_simplex_data(key, None);
    /// assert_eq!(prev, Some(Some(42)));
    /// let simplex = dt.tds().simplex(key).ok_or(ExampleError::MissingSimplex)?;
    /// assert_eq!(simplex.data(), None);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn set_simplex_data(&mut self, key: SimplexKey, data: Option<V>) -> Option<Option<V>> {
        self.tds.set_simplex_data(key, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::kernel::FastKernel;
    use slotmap::KeyData;

    #[test]
    fn new_empty_sets_default_topology_and_validation_policy() {
        let tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        assert_eq!(tri.tds.number_of_vertices(), 0);
        assert_eq!(tri.tds.number_of_simplices(), 0);
        assert_eq!(tri.global_topology, GlobalTopology::DEFAULT);
        assert_eq!(tri.topology_guarantee, TopologyGuarantee::DEFAULT);
        assert_eq!(
            tri.validation_policy,
            TopologyGuarantee::DEFAULT.default_validation_policy()
        );
    }

    #[test]
    fn set_vertex_data_returns_none_for_invalid_key() {
        let mut tri: Triangulation<FastKernel<f64>, i32, (), 2> =
            Triangulation::new_empty(FastKernel::new());
        let stale = VertexKey::from(KeyData::from_ffi(0xDEAD_BEEF));

        assert_eq!(tri.set_vertex_data(stale, Some(42)), None);
        assert_eq!(tri.tds.number_of_vertices(), 0);
    }

    #[test]
    fn set_simplex_data_returns_none_for_invalid_key() {
        let mut tri: Triangulation<FastKernel<f64>, (), i32, 2> =
            Triangulation::new_empty(FastKernel::new());
        let stale = SimplexKey::from(KeyData::from_ffi(0xDEAD_BEEF));

        assert_eq!(tri.set_simplex_data(stale, Some(42)), None);
        assert_eq!(tri.tds.number_of_simplices(), 0);
    }
}
