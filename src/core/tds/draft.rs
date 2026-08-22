//! Unpublished TDS assembly state.

#![forbid(unsafe_code)]

use super::model::UnverifiedTds;
use super::{SimplexKey, Tds, TdsConstructionError, TdsError, TdsMutationError, VertexKey};
use crate::core::collections::SimplexVertexKeyBuffer;
use crate::core::simplex::Simplex;
use crate::core::simplex::SimplexValidationError;
use crate::core::vertex::Vertex;
use thiserror::Error;

/// Typed failures while staging an explicit simplex in a [`TdsDraft`].
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum TdsDraftInsertionError {
    /// The supplied vertex-key list does not define a simplex of dimension `D`.
    #[error("the explicit simplex specification is invalid: {source}")]
    SimplexCreation {
        /// Typed simplex construction failure.
        #[source]
        source: SimplexValidationError,
    },
    /// The simplex conflicts with connectivity already staged in the draft.
    #[error("the explicit simplex could not be inserted into the TDS draft: {source}")]
    SimplexInsertion {
        /// Typed TDS insertion failure.
        #[source]
        source: Box<TdsConstructionError>,
    },
}

/// Typed failures while deriving and publishing a [`TdsDraft`].
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum TdsDraftError {
    /// Reciprocal simplex neighbors could not be derived.
    #[error("neighbor assignment failed while publishing a TDS draft: {source}")]
    NeighborAssignment {
        /// Typed TDS structural failure.
        #[source]
        source: Box<TdsError>,
    },
    /// Vertex-to-simplex incidence could not be derived.
    #[error("incident-simplex assignment failed while publishing a TDS draft: {source}")]
    IncidentAssignment {
        /// Typed TDS mutation failure.
        #[source]
        source: Box<TdsMutationError>,
    },
    /// Coherent combinatorial orientation could not be established.
    #[error("orientation normalization failed while publishing a TDS draft: {source}")]
    OrientationNormalization {
        /// Typed TDS orientation failure.
        #[source]
        source: Box<TdsError>,
    },
    /// Final cumulative Levels 1–2 validation failed.
    #[error("Levels 1-2 validation failed while publishing a TDS draft: {source}")]
    Validation {
        /// Typed TDS validation failure.
        #[source]
        source: Box<TdsError>,
    },
}

/// Mutable, unpublished assembly state for a [`Tds`].
///
/// A draft accepts explicit vertices and maximal simplices without choosing a
/// geometry-specific connectivity algorithm. It may temporarily lack enough
/// elements for a complete `D`-dimensional complex. Consuming
/// [`finish`](Self::finish) derives adjacency, incidence, and coherent
/// orientation before the storage can cross the public Levels 1–2 boundary.
///
/// Use [`TdsBuilder`](crate::tds::TdsBuilder) when the complete vertex slice and
/// index-based simplex specifications are already available. Use `TdsDraft`
/// when those explicit elements must be staged incrementally.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::CoordinateConversionError;
/// use delaunay::prelude::tds::{
///     TdsConstructionError, TdsDraft, TdsDraftError, TdsDraftInsertionError,
/// };
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Coordinate(#[from] CoordinateConversionError),
/// #     #[error(transparent)]
/// #     VertexInsertion(#[from] TdsConstructionError),
/// #     #[error(transparent)]
/// #     SimplexInsertion(#[from] TdsDraftInsertionError),
/// #     #[error(transparent)]
/// #     Publication(#[from] TdsDraftError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let mut draft: TdsDraft<(), (), 2> = TdsDraft::new();
/// let a = draft.insert_vertex(delaunay::vertex![0.0, 0.0]?)?;
/// let b = draft.insert_vertex(delaunay::vertex![1.0, 0.0]?)?;
/// let c = draft.insert_vertex(delaunay::vertex![0.0, 1.0]?)?;
/// draft.insert_simplex([a, b, c])?;
///
/// let tds = draft.finish()?;
/// assert_eq!(tds.number_of_simplices(), 1);
/// assert!(tds.validate().is_ok());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct TdsDraft<U, V, const D: usize> {
    storage: UnverifiedTds<U, V, D>,
}

impl<U, V, const D: usize> TdsDraft<U, V, D> {
    /// Creates an empty unpublished assembly workspace.
    #[must_use]
    pub fn new() -> Self {
        Self {
            storage: UnverifiedTds::empty_unpublished(),
        }
    }

    /// Inserts a vertex into the unpublished assembly workspace.
    ///
    /// The returned key may be used to construct explicit [`Simplex`] values
    /// for [`insert_simplex`](Self::insert_simplex).
    ///
    /// # Errors
    ///
    /// Returns [`TdsConstructionError`] when the vertex cannot be inserted
    /// without violating a local TDS invariant.
    pub fn insert_vertex(
        &mut self,
        vertex: Vertex<U, D>,
    ) -> Result<VertexKey, TdsConstructionError> {
        self.storage.insert_vertex_with_mapping(vertex)
    }

    /// Inserts an explicit maximal simplex into the unpublished workspace.
    ///
    /// This operation validates simplex arity, vertex provenance, duplicate
    /// connectivity, and facet multiplicity. Neighbor links and the cumulative
    /// Levels 1–2 proof are derived by [`finish`](Self::finish).
    ///
    /// # Errors
    ///
    /// Returns [`TdsDraftInsertionError`] when the keys do not form a simplex
    /// of dimension `D` or the simplex violates a local TDS invariant.
    pub fn insert_simplex(
        &mut self,
        vertices: impl IntoIterator<Item = VertexKey>,
    ) -> Result<SimplexKey, TdsDraftInsertionError> {
        self.insert_simplex_with_data(vertices, None)
    }

    /// Inserts an explicit maximal simplex and optional payload.
    ///
    /// # Errors
    ///
    /// Returns [`TdsDraftInsertionError`] when the keys do not form a simplex
    /// of dimension `D` or the simplex violates a local TDS invariant.
    pub fn insert_simplex_with_data(
        &mut self,
        vertices: impl IntoIterator<Item = VertexKey>,
        data: Option<V>,
    ) -> Result<SimplexKey, TdsDraftInsertionError> {
        let vertex_keys: SimplexVertexKeyBuffer = vertices.into_iter().collect();
        let simplex = Simplex::try_new_with_data(vertex_keys, data)
            .map_err(|source| TdsDraftInsertionError::SimplexCreation { source })?;
        self.storage
            .insert_simplex_with_mapping(simplex)
            .map_err(|source| TdsDraftInsertionError::SimplexInsertion {
                source: Box::new(source),
            })
    }

    /// Returns the number of vertices currently staged by the draft.
    #[must_use]
    pub fn number_of_vertices(&self) -> usize {
        self.storage.number_of_vertices()
    }

    /// Returns the number of maximal simplices currently staged by the draft.
    #[must_use]
    pub fn number_of_simplices(&self) -> usize {
        self.storage.number_of_simplices()
    }

    /// Returns the dimension implied by the currently staged vertex count.
    #[must_use]
    pub fn dim(&self) -> i32 {
        self.storage.dim()
    }

    /// Returns a staged vertex by key.
    #[must_use]
    pub fn vertex(&self, key: VertexKey) -> Option<&Vertex<U, D>> {
        self.storage.vertex(key)
    }

    /// Iterates over the vertices currently staged by the draft.
    pub fn vertices(&self) -> impl Iterator<Item = (VertexKey, &Vertex<U, D>)> {
        self.storage.vertices()
    }

    /// Returns whether the currently staged simplices are coherently oriented.
    #[must_use]
    pub fn is_coherently_oriented(&self) -> bool {
        self.storage.is_coherently_oriented()
    }

    /// Validates the currently staged Levels 1–2 state.
    ///
    /// A partial bootstrap is intentionally reported as incomplete; success
    /// means the staged value already satisfies the complete TDS invariants.
    ///
    /// # Errors
    ///
    /// Returns the first cumulative Levels 1–2 validation failure.
    pub fn validate_structure(&self) -> Result<(), TdsError> {
        self.storage.validate()
    }

    /// Copies this workspace while preserving topology identity for rollback.
    pub(crate) fn clone_for_rollback(&self) -> Self
    where
        U: Clone,
        V: Clone,
    {
        Self {
            storage: self.storage.clone_for_rollback(),
        }
    }

    /// Inserts a simplex after the caller has proved cross-simplex topology.
    pub(crate) fn insert_simplex_prechecked_topology(
        &mut self,
        simplex: Simplex<V, D>,
    ) -> Result<SimplexKey, TdsConstructionError> {
        self.storage
            .insert_simplex_with_mapping_prechecked_topology(simplex)
    }

    /// Consumes the draft and publishes it only after Levels 1–2 validation.
    ///
    /// # Errors
    ///
    /// Returns [`TdsDraftError`] when derived adjacency, incidence,
    /// orientation, or cumulative validation fails. The rejected workspace is
    /// dropped rather than exposed as a `Tds`.
    pub fn finish(mut self) -> Result<Tds<U, V, D>, TdsDraftError> {
        self.storage
            .assign_neighbors()
            .map_err(|source| TdsDraftError::NeighborAssignment {
                source: Box::new(source),
            })?;
        self.storage.assign_incident_simplices().map_err(|source| {
            TdsDraftError::IncidentAssignment {
                source: Box::new(source),
            }
        })?;
        self.storage
            .normalize_coherent_orientation()
            .map_err(|source| TdsDraftError::OrientationNormalization {
                source: Box::new(source),
            })?;
        self.storage
            .publish()
            .map_err(|source| TdsDraftError::Validation {
                source: Box::new(source),
            })
    }
}

impl<U, V, const D: usize> Default for TdsDraft<U, V, D> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test_support {
    use super::super::TopologyOwnerId;
    use super::*;

    impl<U, V, const D: usize> TdsDraft<U, V, D> {
        /// Returns the staged storage identity for rollback regression tests.
        pub(crate) fn topology_owner_id(&self) -> TopologyOwnerId {
            self.storage.topology_owner_id()
        }

        /// Returns the staged structural generation for rollback regression tests.
        pub(crate) fn topology_generation(&self) -> u64 {
            self.storage.generation()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vertex;
    use std::assert_matches;

    #[test]
    fn empty_draft_publishes_as_the_verified_empty_complex() {
        let draft: TdsDraft<(), (), 2> = TdsDraft::new();
        let tds = draft.finish().unwrap();

        assert_eq!(tds.number_of_vertices(), 0);
        assert_eq!(tds.number_of_simplices(), 0);
        assert_matches!(
            tds.construction_state(),
            crate::core::tds::TriangulationConstructionState::Constructed
        );
    }

    #[test]
    fn partial_bootstrap_draft_cannot_publish() {
        let mut draft: TdsDraft<(), (), 2> = TdsDraft::new();
        draft.insert_vertex(vertex![0.0, 0.0].unwrap()).unwrap();

        let error = draft.finish().unwrap_err();
        assert_matches!(
            error,
            TdsDraftError::Validation {
                source
            } if matches!(
                *source,
                TdsError::IncompleteConstruction {
                    dimension: 2,
                    vertex_count: 1,
                    simplex_count: 0,
                }
            )
        );
    }

    #[test]
    fn duplicate_simplex_rejection_preserves_the_staged_payload_and_topology() {
        let mut draft: TdsDraft<(), usize, 2> = TdsDraft::default();
        assert_eq!(draft.number_of_vertices(), 0);
        assert_eq!(draft.number_of_simplices(), 0);
        assert_eq!(draft.dim(), -1);

        let v0 = draft.insert_vertex(vertex![0.0, 0.0].unwrap()).unwrap();
        let v1 = draft.insert_vertex(vertex![1.0, 0.0].unwrap()).unwrap();
        let v2 = draft.insert_vertex(vertex![0.0, 1.0].unwrap()).unwrap();
        let simplex_key = draft
            .insert_simplex_with_data([v0, v1, v2], Some(7))
            .unwrap();

        assert_eq!(draft.number_of_vertices(), 3);
        assert_eq!(draft.number_of_simplices(), 1);
        assert_eq!(draft.dim(), 2);
        assert_matches!(
            draft.insert_simplex([v0, v1, v2]),
            Err(TdsDraftInsertionError::SimplexInsertion { source })
                if matches!(
                    source.as_ref(),
                    TdsConstructionError::ValidationError {
                        source: TdsError::DuplicateSimplices { .. }
                    }
                )
        );

        let tds = draft.finish().unwrap();
        assert_eq!(tds.number_of_simplices(), 1);
        assert_eq!(tds.simplex(simplex_key).unwrap().data(), Some(&7));
        assert!(tds.validate().is_ok());
    }

    #[test]
    fn explicit_connectivity_publishes_without_a_geometry_specific_algorithm() {
        let mut draft: TdsDraft<(), (), 2> = TdsDraft::new();
        let v0 = draft.insert_vertex(vertex![0.0, 0.0].unwrap()).unwrap();
        let v1 = draft.insert_vertex(vertex![1.0, 0.0].unwrap()).unwrap();
        let v2 = draft.insert_vertex(vertex![0.0, 1.0].unwrap()).unwrap();
        draft.insert_simplex([v0, v1, v2]).unwrap();

        let tds = draft.finish().unwrap();
        assert_eq!(tds.number_of_vertices(), 3);
        assert_eq!(tds.number_of_simplices(), 1);
        assert!(tds.validate().is_ok());
    }
}
