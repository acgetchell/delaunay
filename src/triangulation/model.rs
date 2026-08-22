//! Generic triangulation combining a kernel and combinatorial data structure.
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

use std::ops::{Deref, DerefMut};

use crate::core::tds::{
    SimplexKey, Tds, TdsMutationError, TopologyOwner, TopologyOwnerId, VertexKey,
};
use crate::geometry::kernel::Kernel;
use crate::topology::traits::topological_space::GlobalTopology;
use crate::triangulation::validation::{TopologyGuarantee, ValidationPolicy};

/// Proof-bearing Levels 1–4 triangulation.
///
/// `Triangulation` owns a validated [`Tds`] (Levels 1–2), explicit topology
/// context (Level 3), and a valid realization in that context (Level 4). Its
/// public constructors and mutating operations either preserve those proofs or
/// return an error without publishing the attempted state. An empty Euclidean
/// triangulation satisfies the same contract vacuously.
///
/// Use [`TriangulationBuilder`](crate::TriangulationBuilder) for this proof
/// transition. Its default strict mode performs non-mutating certification of
/// an already canonical TDS; explicit canonicalizing mode may normalize
/// orientation. Use
/// [`Triangulation::into_tds`] to demote the owner explicitly. Consume a
/// triangulation with [`DelaunayRefinementBuilder`](crate::DelaunayRefinementBuilder)
/// to certify Level 5 strictly or repair and certify the Delaunay property.
///
/// # Type Parameters
/// - `K`: Geometric kernel implementing predicates
/// - `U`: User data type for vertices
/// - `V`: User data type for simplices
/// - `D`: Dimension of the triangulation.
///
/// Unpublished Levels 3–4 state remains inside the crate-private construction
/// draft, so the public owner has no caller-selectable publication state.
#[derive(Clone, Debug)]
pub struct Triangulation<K, U, V, const D: usize> {
    /// The geometric kernel for predicates.
    pub(crate) kernel: K,
    /// The proof-bearing Levels 1–2 combinatorial owner.
    ///
    /// Higher layers may query it or invoke its checked transitions, but must
    /// not expose mutable storage or edit its canonical fields directly.
    pub(crate) tds: Tds<U, V, D>,
    /// Runtime metadata describing the global topological space represented by this triangulation.
    pub(crate) global_topology: GlobalTopology<D>,
    pub(crate) validation_policy: ValidationPolicy,
    pub(crate) topology_guarantee: TopologyGuarantee,
}

impl<K, U, V, const D: usize> TopologyOwner for Triangulation<K, U, V, D> {
    #[inline]
    fn topology_owner_id(&self) -> TopologyOwnerId {
        self.tds.topology_owner_id()
    }

    #[inline]
    fn topology_generation(&self) -> u64 {
        self.tds.generation()
    }
}

// =============================================================================
// Basic Accessors (Minimal Bounds)
// =============================================================================

impl<K, U, V, const D: usize> Triangulation<K, U, V, D>
where
    K: Kernel<D>,
{
    /// Returns a borrowed view of the canonical triangulation storage to
    /// crate-internal algorithms.
    #[inline]
    #[must_use]
    pub(crate) const fn tds(&self) -> &Tds<U, V, D> {
        &self.tds
    }

    /// Consumes this Levels 1–4 owner and returns its transport/storage value.
    ///
    /// This explicit demotion is the inverse boundary of strict
    /// [`TriangulationBuilder`](crate::TriangulationBuilder) publication.
    /// `Tds` does not retain the runtime [`TopologyGuarantee`] or
    /// [`GlobalTopology`] context, so callers must persist those values
    /// separately when they intend to restore the same domain contract.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulationBuilder};
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let triangulation = DelaunayTriangulationBuilder::new(&vertices)
    ///     .build_triangulation()?;
    ///
    /// let tds = triangulation.into_tds();
    /// assert_eq!(tds.number_of_vertices(), 3);
    /// assert_eq!(tds.number_of_simplices(), 1);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn into_tds(self) -> Tds<U, V, D> {
        self.tds
    }

    /// Sets the auxiliary data on a vertex, returning the previous value.
    ///
    /// Delegates to [`Tds::set_vertex_data`]. This is a safe O(1) operation
    /// that does not affect geometry, topology, or Delaunay invariants.
    ///
    /// # Returns
    ///
    /// The old `Option<U>` value when the key exists.
    ///
    /// # Errors
    ///
    /// Returns [`TdsMutationError`] if `key` does not identify a vertex in the
    /// underlying TDS.
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
    /// #     #[error(transparent)]
    /// #     TdsMutation(#[from] delaunay::prelude::tds::TdsMutationError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices: [Vertex<i32, 2>; 3] = [
    ///     delaunay::vertex![0.0, 0.0; data = 10i32]?,
    ///     delaunay::vertex![1.0, 0.0; data = 20]?,
    ///     delaunay::vertex![0.0, 1.0; data = 30]?,
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let key = dt.vertices().next().ok_or(ExampleError::MissingVertex)?.0;
    /// let prev = dt.set_vertex_data(key, Some(99))?;
    /// assert!(prev.is_some());
    ///
    /// // Clear data
    /// let prev = dt.set_vertex_data(key, None)?;
    /// assert_eq!(prev, Some(99));
    /// let vertex = dt.vertex(key).ok_or(ExampleError::MissingVertex)?;
    /// assert_eq!(vertex.data(), None);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn set_vertex_data(
        &mut self,
        key: VertexKey,
        data: Option<U>,
    ) -> Result<Option<U>, TdsMutationError> {
        self.tds.set_vertex_data(key, data)
    }

    /// Sets the auxiliary data on a simplex, returning the previous value.
    ///
    /// Delegates to [`Tds::set_simplex_data`]. This is a safe O(1) operation
    /// that does not affect geometry, topology, or Delaunay invariants.
    ///
    /// # Returns
    ///
    /// The old `Option<V>` value when the key exists.
    ///
    /// # Errors
    ///
    /// Returns [`TdsMutationError`] if `key` does not identify a simplex in
    /// the underlying TDS.
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
    /// #     #[error(transparent)]
    /// #     TdsMutation(#[from] delaunay::prelude::tds::TdsMutationError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices).simplex_data_type::<i32>().build()?;
    /// let key = dt.simplices().next().ok_or(ExampleError::MissingSimplex)?.0;
    /// let prev = dt.set_simplex_data(key, Some(42))?;
    /// assert_eq!(prev, None);
    ///
    /// // Clear data
    /// let prev = dt.set_simplex_data(key, None)?;
    /// assert_eq!(prev, Some(42));
    /// let simplex = dt.simplex(key).ok_or(ExampleError::MissingSimplex)?;
    /// assert_eq!(simplex.data(), None);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn set_simplex_data(
        &mut self,
        key: SimplexKey,
        data: Option<V>,
    ) -> Result<Option<V>, TdsMutationError> {
        self.tds.set_simplex_data(key, data)
    }
}

/// Crate-private unpublished Levels 3–4 owner used only inside the construction draft.
#[derive(Clone, Debug)]
#[repr(transparent)]
pub(in crate::triangulation) struct UnverifiedTriangulation<K, U, V, const D: usize> {
    pub(in crate::triangulation) storage: Triangulation<K, U, V, D>,
}

impl<K, U, V, const D: usize> UnverifiedTriangulation<K, U, V, D> {
    /// Creates unpublished storage with the selected topology context.
    pub(in crate::triangulation) const fn with_topology_context(
        tds: Tds<U, V, D>,
        kernel: K,
        topology_guarantee: TopologyGuarantee,
        global_topology: GlobalTopology<D>,
    ) -> Self {
        Self {
            storage: Triangulation {
                kernel,
                tds,
                global_topology,
                validation_policy: topology_guarantee.default_validation_policy(),
                topology_guarantee,
            },
        }
    }

    /// Publishes the certified owner by removing its unpublished wrapper.
    pub(in crate::triangulation) fn into_verified(self) -> Triangulation<K, U, V, D> {
        self.storage
    }

    /// Recovers the input TDS when publication fails.
    pub(in crate::triangulation) fn into_tds(self) -> Tds<U, V, D> {
        self.storage.tds
    }
}

impl<K, U, V, const D: usize> Deref for UnverifiedTriangulation<K, U, V, D> {
    type Target = Triangulation<K, U, V, D>;

    fn deref(&self) -> &Self::Target {
        &self.storage
    }
}

impl<K, U, V, const D: usize> DerefMut for UnverifiedTriangulation<K, U, V, D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.storage
    }
}

/// Test-only constructors for deliberately unpublished owner fixtures.
#[cfg(test)]
pub mod test_support {
    use super::*;

    impl<K, U, V, const D: usize> Triangulation<K, U, V, D>
    where
        K: Kernel<D>,
    {
        /// Creates empty storage for tests that exercise bootstrap transitions.
        #[must_use]
        pub(crate) fn new_empty(kernel: K) -> Self {
            Self {
                kernel,
                tds: Tds::empty(),
                global_topology: GlobalTopology::DEFAULT,
                validation_policy: TopologyGuarantee::DEFAULT.default_validation_policy(),
                topology_guarantee: TopologyGuarantee::DEFAULT,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tds::TdsError;
    use crate::geometry::kernel::FastKernel;
    use crate::vertex;
    use slotmap::KeyData;
    use std::assert_matches;

    impl<K, U, V, const D: usize> Triangulation<K, U, V, D> {
        /// Constructs a prepared TDS fixture without crossing a public
        /// validation boundary.
        #[inline]
        pub(crate) const fn new_with_tds(kernel: K, tds: Tds<U, V, D>) -> Self {
            Self {
                kernel,
                tds,
                global_topology: GlobalTopology::DEFAULT,
                validation_policy: TopologyGuarantee::DEFAULT.default_validation_policy(),
                topology_guarantee: TopologyGuarantee::DEFAULT,
            }
        }
    }

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
    fn explicit_empty_context_sets_requested_topology_and_policy() {
        let tri: Triangulation<FastKernel<f64>, (), (), 3> = Triangulation {
            kernel: FastKernel::new(),
            tds: Tds::empty(),
            global_topology: GlobalTopology::Spherical,
            validation_policy: TopologyGuarantee::Pseudomanifold.default_validation_policy(),
            topology_guarantee: TopologyGuarantee::Pseudomanifold,
        };

        assert_eq!(tri.global_topology, GlobalTopology::Spherical);
        assert_eq!(tri.topology_guarantee, TopologyGuarantee::Pseudomanifold);
        assert_eq!(
            tri.validation_policy,
            TopologyGuarantee::Pseudomanifold.default_validation_policy()
        );
    }

    #[test]
    fn topology_owner_and_demotion_preserve_canonical_tds() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());
        let owner_id = tri.topology_owner_id();
        let initial_generation = tri.topology_generation();

        tri.tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();

        assert_eq!(tri.tds().topology_owner_id(), owner_id);
        assert_eq!(tri.topology_generation(), tri.tds().generation());
        assert!(tri.topology_generation() > initial_generation);

        let tds = tri.into_tds();
        assert_eq!(tds.topology_owner_id(), owner_id);
        assert_eq!(tds.number_of_vertices(), 1);
    }

    #[test]
    fn set_vertex_data_returns_error_for_invalid_key() {
        let mut tri: Triangulation<FastKernel<f64>, i32, (), 2> =
            Triangulation::new_empty(FastKernel::new());
        let stale = VertexKey::from(KeyData::from_ffi(0xDEAD_BEEF));

        let err = tri.set_vertex_data(stale, Some(42)).unwrap_err();
        assert_matches!(err.as_tds_error(), TdsError::VertexNotFound { .. });
        assert_eq!(tri.tds.number_of_vertices(), 0);
    }

    #[test]
    fn set_simplex_data_returns_error_for_invalid_key() {
        let mut tri: Triangulation<FastKernel<f64>, (), i32, 2> =
            Triangulation::new_empty(FastKernel::new());
        let stale = SimplexKey::from(KeyData::from_ffi(0xDEAD_BEEF));

        let err = tri.set_simplex_data(stale, Some(42)).unwrap_err();
        assert_matches!(err.as_tds_error(), TdsError::SimplexNotFound { .. });
        assert_eq!(tri.tds.number_of_simplices(), 0);
    }
}
