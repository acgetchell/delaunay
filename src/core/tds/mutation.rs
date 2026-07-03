//! Mutation and repair operations for the triangulation data structure.

use super::errors::{
    EntityKind, NeighborValidationError, TdsConstructionError, TdsError, TdsMutationError,
};
use super::incidence::SimplexIncidenceRemoval;
use super::storage::{SimplexUuidSortKey, Tds};
use super::{SimplexKey, VertexKey};
use crate::core::collections::{
    CLEANUP_OPERATION_BUFFER_SIZE, Entry, FastHashMap, MAX_PRACTICAL_DIMENSION_SIZE,
    NeighborBuffer, SimplexKeySet, SimplexRemovalBuffer, SmallBuffer, VertexKeySet,
    fast_hash_map_with_capacity, fast_hash_set_with_capacity,
};
use crate::core::simplex::{NeighborSlot, Simplex};
use crate::core::vertex::Vertex;
#[cfg(test)]
use crate::deletion::DeleteVertexError;
use std::collections::VecDeque;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

/// Topology validation mode for checked TDS simplex insertion.
#[derive(Clone, Copy)]
enum SimplexInsertionTopologyCheck {
    /// Validate the candidate against all existing simplices.
    Checked,
    /// Skip global topology scans because the caller validated the local cavity.
    Prechecked,
}

type SimplexRemovalRecord = (
    SimplexKey,
    SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
);
type SimplexRemovalRecords = SmallBuffer<SimplexRemovalRecord, CLEANUP_OPERATION_BUFFER_SIZE>;
type SimplexIncidenceRemovalRecords =
    SmallBuffer<SimplexIncidenceRemoval, CLEANUP_OPERATION_BUFFER_SIZE>;
type IncidentSimplexUpdates =
    SmallBuffer<(VertexKey, Option<SimplexKey>), CLEANUP_OPERATION_BUFFER_SIZE>;

/// Membership test for a pending simplex-removal batch.
///
/// Small conflict regions stay inline to avoid per-insertion hash allocation;
/// large batches fall back to a hash set so bulk cleanup keeps O(1) membership.
enum RemovedSimplexMembership<'a> {
    Inline(&'a [SimplexRemovalRecord]),
    Hashed(SimplexKeySet),
}

impl<'a> RemovedSimplexMembership<'a> {
    /// Builds membership over removals without allocating for ordinary small cavities.
    fn from_removals(simplex_removals: &'a [SimplexRemovalRecord]) -> Self {
        if simplex_removals.len() <= CLEANUP_OPERATION_BUFFER_SIZE {
            return Self::Inline(simplex_removals);
        }

        let mut simplex_keys = fast_hash_set_with_capacity(simplex_removals.len());
        simplex_keys.extend(simplex_removals.iter().map(|(simplex_key, _)| *simplex_key));
        Self::Hashed(simplex_keys)
    }

    /// Returns whether `simplex_key` is in the removal batch.
    fn contains(&self, simplex_key: SimplexKey) -> bool {
        match self {
            Self::Inline(simplex_removals) => simplex_removals
                .iter()
                .any(|(removed_key, _)| *removed_key == simplex_key),
            Self::Hashed(simplex_keys) => simplex_keys.contains(&simplex_key),
        }
    }
}

/// Unique affected vertices from a simplex-removal batch.
///
/// Small cavities use a compact inline list; larger maintenance operations use
/// the existing hash-set representation to preserve scaling behavior.
enum AffectedVertexRecords {
    Inline(SmallBuffer<VertexKey, CLEANUP_OPERATION_BUFFER_SIZE>),
    Hashed(VertexKeySet),
}

impl AffectedVertexRecords {
    /// Creates a unique-vertex accumulator sized for the expected removal frontier.
    fn with_expected_len(expected_len: usize) -> Self {
        if expected_len <= CLEANUP_OPERATION_BUFFER_SIZE {
            return Self::Inline(SmallBuffer::with_capacity(expected_len));
        }

        Self::Hashed(fast_hash_set_with_capacity(expected_len))
    }

    /// Inserts a vertex once, preserving set semantics for repair traversal.
    fn insert(&mut self, vertex_key: VertexKey) {
        match self {
            Self::Inline(vertices) => {
                if !vertices.contains(&vertex_key) {
                    vertices.push(vertex_key);
                }
            }
            Self::Hashed(vertices) => {
                vertices.insert(vertex_key);
            }
        }
    }

    /// Returns the number of unique affected vertices.
    fn len(&self) -> usize {
        match self {
            Self::Inline(vertices) => vertices.len(),
            Self::Hashed(vertices) => vertices.len(),
        }
    }

    /// Visits each affected vertex exactly once, stopping on the first error.
    fn try_for_each<E>(&self, mut visit: impl FnMut(VertexKey) -> Result<(), E>) -> Result<(), E> {
        match self {
            Self::Inline(vertices) => {
                for &vertex_key in vertices {
                    visit(vertex_key)?;
                }
            }
            Self::Hashed(vertices) => {
                for &vertex_key in vertices {
                    visit(vertex_key)?;
                }
            }
        }
        Ok(())
    }
}

/// Candidate surviving incident simplices keyed by affected vertex.
///
/// This is deliberately map-like without forcing a hash allocation for the
/// common one-to-few simplex removal performed during insertion.
#[expect(
    clippy::large_enum_variant,
    reason = "inline small-cavity candidate storage avoids hot-path allocations; large batches use the hash-backed variant"
)]
enum CandidateIncidentRecords {
    Inline(SmallBuffer<(VertexKey, SimplexKey), CLEANUP_OPERATION_BUFFER_SIZE>),
    Hashed(FastHashMap<VertexKey, SimplexKey>),
}

impl CandidateIncidentRecords {
    /// Creates a candidate map with inline storage for ordinary small cavities.
    fn with_expected_len(expected_len: usize) -> Self {
        if expected_len <= CLEANUP_OPERATION_BUFFER_SIZE {
            return Self::Inline(SmallBuffer::with_capacity(expected_len));
        }

        Self::Hashed(fast_hash_map_with_capacity(expected_len))
    }

    /// Records the first surviving incident simplex discovered for a vertex.
    fn insert_if_absent(&mut self, vertex_key: VertexKey, simplex_key: SimplexKey) {
        match self {
            Self::Inline(candidates) => {
                if !candidates
                    .iter()
                    .any(|(candidate_vertex, _)| *candidate_vertex == vertex_key)
                {
                    candidates.push((vertex_key, simplex_key));
                }
            }
            Self::Hashed(candidates) => {
                candidates.entry(vertex_key).or_insert(simplex_key);
            }
        }
    }

    /// Returns a candidate surviving incident simplex for `vertex_key`.
    fn get(&self, vertex_key: VertexKey) -> Option<SimplexKey> {
        match self {
            Self::Inline(candidates) => {
                candidates
                    .iter()
                    .find_map(|(candidate_vertex, simplex_key)| {
                        (*candidate_vertex == vertex_key).then_some(*simplex_key)
                    })
            }
            Self::Hashed(candidates) => candidates.get(&vertex_key).copied(),
        }
    }
}

impl<U, V, const D: usize> Tds<U, V, D> {
    /// Assigns neighbor relationships between simplices based on shared facets with semantic ordering.
    ///
    /// This method efficiently builds neighbor relationships by using the `facet_key_from_vertices`
    /// function to compute unique keys for facets. Two simplices are considered neighbors if they share
    /// exactly one facet (which contains D vertices for a D-dimensional triangulation).
    ///
    /// **Note**: This is a purely combinatorial operation that does not perform any coordinate
    /// operations. It works entirely with vertex keys, simplex keys, and topological relationships.
    ///
    /// **Internal use only**: This method rebuilds ALL neighbor pointers from scratch, which is
    /// inefficient for most use cases. For external use, prefer
    /// [`repair_neighbor_pointers`](crate::prelude::insertion::repair_neighbor_pointers),
    /// which rebuilds neighbor pointers from facet incidence.
    ///
    /// # Errors
    ///
    /// Returns `TdsError` if neighbor assignment fails due to inconsistent
    /// data structures or invalid facet sharing patterns.
    pub(crate) fn assign_neighbors(&mut self) -> Result<(), TdsError> {
        // Build facet mapping with vertex index information using optimized collections
        // facet_key -> [(simplex_key, vertex_index_opposite_to_facet)]
        type FacetInfo = (SimplexKey, usize);
        // Use saturating arithmetic to avoid potential overflow on adversarial inputs
        let cap = self.simplices.len().saturating_mul(D.saturating_add(1));
        let mut facet_map: FastHashMap<u64, SmallBuffer<FacetInfo, 2>> =
            fast_hash_map_with_capacity(cap);

        for (simplex_key, simplex) in &self.simplices {
            let vertices = self.simplex_vertices(simplex_key).map_err(|err| {
                TdsError::VertexKeyRetrievalFailed {
                    simplex_id: simplex.uuid(),
                    message: format!(
                        "Failed to retrieve vertex keys for simplex during neighbor assignment: {err}"
                    ),
                }
            })?;

            for i in 0..vertices.len() {
                let facet_key =
                    Self::periodic_facet_key_from_simplex_vertices(simplex, vertices, i)?;
                let facet_entry = facet_map.entry(facet_key).or_default();
                // Detect degenerate case early: more than 2 simplices sharing a facet
                // Note: Check happens before push, so len() reflects current sharing count
                if facet_entry.len() >= 2 {
                    return Err(TdsError::InconsistentDataStructure {
                        message: format!(
                            "Facet with key {} already shared by {} simplices; cannot add simplex {} (would violate 2-manifold property)",
                            facet_key,
                            facet_entry.len(),
                            simplex.uuid()
                        ),
                    });
                }
                facet_entry.push((simplex_key, i));
            }
        }

        // For each simplex, build an ordered neighbor array where neighbors[i] is opposite vertices[i]
        let mut simplex_neighbors: FastHashMap<
            SimplexKey,
            SmallBuffer<Option<SimplexKey>, MAX_PRACTICAL_DIMENSION_SIZE>,
        > = fast_hash_map_with_capacity(self.simplices.len());

        // Initialize each simplex with a SmallBuffer of None values (one per vertex)
        for (simplex_key, simplex) in &self.simplices {
            let vertex_count = simplex.number_of_vertices();
            if vertex_count > MAX_PRACTICAL_DIMENSION_SIZE {
                return Err(TdsError::InconsistentDataStructure {
                    message: format!(
                        "simplex {} vertex count ({vertex_count}) exceeds storage limit MAX_PRACTICAL_DIMENSION_SIZE ({MAX_PRACTICAL_DIMENSION_SIZE}) (would overflow neighbors buffer)",
                        simplex.uuid(),
                    ),
                });
            }
            let mut neighbors = SmallBuffer::with_capacity(vertex_count);
            neighbors.resize(vertex_count, None);
            simplex_neighbors.insert(simplex_key, neighbors);
        }

        // For each facet that is shared by exactly two simplices, establish neighbor relationships
        // Note: >2 simplices per facet already caught by early check during map build (above)
        for (_facet_key, facet_infos) in facet_map {
            if facet_infos.len() != 2 {
                continue;
            }

            let (simplex_key1, vertex_index1) = facet_infos[0];
            let (simplex_key2, vertex_index2) = facet_infos[1];

            // Set neighbors with semantic constraint: neighbors[i] is opposite vertices[i]
            simplex_neighbors.get_mut(&simplex_key1).ok_or_else(|| {
                TdsError::SimplexNotFound {
                    simplex_key: simplex_key1,
                    context: "assign_neighbors: simplex missing from local neighbors map"
                        .to_string(),
                }
            })?[vertex_index1] = Some(simplex_key2);

            simplex_neighbors.get_mut(&simplex_key2).ok_or_else(|| {
                TdsError::SimplexNotFound {
                    simplex_key: simplex_key2,
                    context: "assign_neighbors: simplex missing from local neighbors map"
                        .to_string(),
                }
            })?[vertex_index2] = Some(simplex_key1);
        }

        // Apply updates. Even simplices with only boundary facets receive an
        // assigned boundary buffer so assigned-boundary and unassigned states
        // remain distinct.
        for (simplex_key, neighbors) in &simplex_neighbors {
            if let Some(simplex) = self.simplices.get_mut(*simplex_key) {
                let simplex_id = simplex.uuid();
                simplex
                    .set_neighbors_from_keys(neighbors.iter().copied())
                    .map_err(|source| TdsError::InvalidSimplex { simplex_id, source })?;
            }
        }

        // Topology changed; invalidate caches.
        self.bump_generation();

        Ok(())
    }

    // =========================================================================
    // QUERY OPERATIONS
    // =========================================================================

    /// Atomically inserts a vertex and creates the UUID-to-key mapping.
    ///
    /// This method ensures that both the vertex insertion and UUID mapping are
    /// performed together, maintaining data structure invariants.
    ///
    /// **⚠️ INTERNAL API WARNING**: This method bypasses atomicity guarantees for topology
    /// assignment operations (`assign_neighbors()` and `assign_incident_simplices()`). It only
    /// ensures atomic vertex insertion and UUID mapping. If you need full atomicity including
    /// topology assignment, use `insert_vertex_with_topology_assignment()` instead.
    ///
    /// **Note:** This method does NOT check for duplicate coordinates. It only checks
    /// for UUID uniqueness. Duplicate coordinate checking is performed at a higher layer
    /// in `Triangulation::try_insert_impl()` before calling this method.
    ///
    /// # Arguments
    ///
    /// * `vertex` - The vertex to insert
    ///
    /// # Returns
    ///
    /// The `VertexKey` that can be used to access the inserted vertex.
    ///
    /// # Errors
    ///
    /// Returns [`TdsConstructionError::DuplicateUuid`] if a vertex with the
    /// same UUID already exists in the triangulation.
    ///
    /// # Examples
    ///
    ///
    /// See the unit tests for usage examples of this pub(crate) method.
    pub(crate) fn insert_vertex_with_mapping(
        &mut self,
        vertex: Vertex<U, D>,
    ) -> Result<VertexKey, TdsConstructionError> {
        let vertex_uuid = vertex.uuid();

        if self.uuid_to_vertex_key.contains_key(&vertex_uuid) {
            return Err(TdsConstructionError::DuplicateUuid {
                entity: EntityKind::Vertex,
                uuid: vertex_uuid,
            });
        }

        let vertex_key = self.vertices.insert(vertex);
        if let Err(source) = self.insert_empty_vertex_incidence(vertex_key) {
            self.vertices.remove(vertex_key);
            return Err(TdsConstructionError::ValidationError(source));
        }
        self.uuid_to_vertex_key.insert(vertex_uuid, vertex_key);
        self.refresh_incomplete_construction_state();
        // Topology changed; invalidate caches.
        self.bump_generation();
        Ok(vertex_key)
    }

    /// Atomically inserts a simplex and creates the UUID-to-key mapping.
    ///
    /// This method ensures that both the simplex insertion and UUID mapping are
    /// performed together, maintaining data structure invariants. This is preferred
    /// over separate raw storage insertion plus `uuid_to_simplex_key.insert()` calls, which can
    /// leave the data structure in an inconsistent state if interrupted.
    ///
    /// # Arguments
    ///
    /// * `simplex` - The simplex to insert
    ///
    /// # Returns
    ///
    /// The `SimplexKey` that can be used to access the inserted simplex.
    ///
    /// # Errors
    ///
    /// Returns [`TdsConstructionError::DuplicateUuid`] if a simplex with the
    /// same UUID already exists in the triangulation, or
    /// [`TdsConstructionError::ValidationError`] if the simplex arity does not
    /// match `D + 1`, or if any vertex key referenced by `simplex` is not present
    /// in this TDS.
    ///
    /// # Examples
    ///
    ///
    /// See the unit tests for usage examples of this pub(crate) method.
    pub(crate) fn insert_simplex_with_mapping(
        &mut self,
        simplex: Simplex<V, D>,
    ) -> Result<SimplexKey, TdsConstructionError> {
        self.insert_simplex_with_mapping_impl(simplex, SimplexInsertionTopologyCheck::Checked)
    }

    /// Atomically inserts a simplex after validating vertex provenance.
    ///
    /// Cavity filling, flips, and explicit construction should only build
    /// `simplex` from vertex keys that came from this TDS and are still live. This
    /// method still verifies that invariant in every build mode so stale keys
    /// fail with a typed error instead of corrupting TDS invariants.
    ///
    /// # Errors
    ///
    /// Returns [`TdsConstructionError::DuplicateUuid`] if a simplex with the same
    /// UUID already exists, [`TdsConstructionError::ValidationError`] if the
    /// arity is wrong, or [`TdsConstructionError::ValidationError`] if any
    /// referenced vertex key is not present in this TDS.
    pub(crate) fn insert_simplex_with_mapping_trusted_vertices(
        &mut self,
        simplex: Simplex<V, D>,
    ) -> Result<SimplexKey, TdsConstructionError> {
        self.insert_simplex_with_mapping_impl(simplex, SimplexInsertionTopologyCheck::Checked)
    }

    /// Inserts a caller-validated simplex without global topology scans.
    ///
    /// Hull-extension callers use this only after proving that candidate simplices
    /// are built from visible boundary facets around a fresh apex and before
    /// immediately wiring local neighbors and validating the affected topology.
    pub(crate) fn insert_simplex_with_mapping_prechecked_topology(
        &mut self,
        simplex: Simplex<V, D>,
    ) -> Result<SimplexKey, TdsConstructionError> {
        self.insert_simplex_with_mapping_impl(simplex, SimplexInsertionTopologyCheck::Prechecked)
    }

    /// Shared checked simplex-insertion path used by public and trusted internal wrappers.
    fn insert_simplex_with_mapping_impl(
        &mut self,
        simplex: Simplex<V, D>,
        topology_check: SimplexInsertionTopologyCheck,
    ) -> Result<SimplexKey, TdsConstructionError> {
        if simplex.number_of_vertices() != D + 1 {
            return Err(TdsConstructionError::ValidationError(
                TdsError::DimensionMismatch {
                    expected: D + 1,
                    actual: simplex.number_of_vertices(),
                    context: format!("{D}-dimensional simplex vertex count"),
                },
            ));
        }

        let simplex_uuid = simplex.uuid();
        if self.uuid_to_simplex_key.contains_key(&simplex_uuid) {
            return Err(TdsConstructionError::DuplicateUuid {
                entity: EntityKind::Simplex,
                uuid: simplex_uuid,
            });
        }

        match topology_check {
            SimplexInsertionTopologyCheck::Checked => {
                self.validate_simplex_vertices_exist(&simplex)?;
                self.validate_simplex_topology_safe_for_insertion(&simplex)?;
            }
            SimplexInsertionTopologyCheck::Prechecked => {}
        }

        let simplex_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
            simplex.vertices().iter().copied().collect();
        let simplex_key = self.simplices.insert(simplex);
        self.uuid_to_simplex_key.insert(simplex_uuid, simplex_key);
        if let Err(source) = self.add_simplex_to_vertex_incidence(simplex_key, &simplex_vertices) {
            self.simplices.remove(simplex_key);
            self.uuid_to_simplex_key.remove(&simplex_uuid);
            return Err(TdsConstructionError::ValidationError(source));
        }
        // Topology changed; invalidate caches.
        self.bump_generation();
        Ok(simplex_key)
    }

    /// Verifies that inserting `simplex` would not violate local topology invariants.
    fn validate_simplex_topology_safe_for_insertion(
        &self,
        simplex: &Simplex<V, D>,
    ) -> Result<(), TdsConstructionError> {
        self.validate_no_duplicate_simplex_on_insert(simplex)?;
        self.validate_facet_sharing_on_insert(simplex)?;
        Ok(())
    }

    /// Builds the duplicate-simplex identity for a not-yet-inserted simplex.
    fn candidate_periodic_vertex_uuid_offsets(
        &self,
        simplex: &Simplex<V, D>,
    ) -> Result<SimplexUuidSortKey<D>, TdsError> {
        let vertices = simplex.vertices();
        let periodic_offsets = simplex.periodic_vertex_offsets();
        if let Some(offsets) = periodic_offsets
            && offsets.len() != vertices.len()
        {
            return Err(TdsError::DimensionMismatch {
                expected: vertices.len(),
                actual: offsets.len(),
                context: format!(
                    "candidate simplex {} periodic offset count vs vertex count",
                    simplex.uuid()
                ),
            });
        }

        let mut vertex_uuid_offsets = SimplexUuidSortKey::<D>::new();
        for (vertex_idx, &vertex_key) in vertices.iter().enumerate() {
            let vertex = self
                .vertices
                .get(vertex_key)
                .ok_or_else(|| TdsError::VertexNotFound {
                    vertex_key,
                    context: format!(
                        "referenced by candidate simplex {} at index {vertex_idx} while building periodic vertex identity (UUID/offset)",
                        simplex.uuid()
                    ),
                })?;
            let offset = periodic_offsets.map_or([0_i8; D], |offsets| offsets[vertex_idx]);
            vertex_uuid_offsets.push((vertex.uuid(), offset));
        }
        vertex_uuid_offsets.sort_unstable();

        Ok(vertex_uuid_offsets)
    }

    /// Rejects a candidate simplex that duplicates an existing maximal simplex.
    fn validate_no_duplicate_simplex_on_insert(
        &self,
        simplex: &Simplex<V, D>,
    ) -> Result<(), TdsError> {
        let candidate_identity = self.candidate_periodic_vertex_uuid_offsets(simplex)?;

        for (existing_simplex_key, _existing_simplex) in &self.simplices {
            let vertices = self.simplex_vertices(existing_simplex_key)?;
            let existing_identity =
                self.build_periodic_vertex_uuid_offsets(existing_simplex_key, vertices)?;

            if existing_identity == candidate_identity {
                return Err(TdsError::DuplicateSimplices {
                    message: format!(
                        "Refusing to insert duplicate simplex {} with same vertex UUIDs as existing simplex {existing_simplex_key:?}: {candidate_identity:?}",
                        simplex.uuid()
                    ),
                });
            }
        }

        Ok(())
    }

    /// Rejects a candidate simplex whose facets would be incident to a third simplex.
    fn validate_facet_sharing_on_insert(&self, simplex: &Simplex<V, D>) -> Result<(), TdsError> {
        let vertices = simplex.vertices();
        for candidate_facet_idx in 0..vertices.len() {
            let candidate_facet_key = Self::periodic_facet_key_from_simplex_vertices(
                simplex,
                vertices,
                candidate_facet_idx,
            )?;
            let mut incident_count = 0_usize;

            for (existing_simplex_key, existing_simplex) in &self.simplices {
                let existing_vertices = self.simplex_vertices(existing_simplex_key)?;
                for existing_facet_idx in 0..existing_vertices.len() {
                    let existing_facet_key = Self::periodic_facet_key_from_simplex_vertices(
                        existing_simplex,
                        existing_vertices,
                        existing_facet_idx,
                    )?;
                    if existing_facet_key == candidate_facet_key {
                        incident_count += 1;
                        if incident_count >= 2 {
                            return Err(TdsError::FacetSharingViolation {
                                facet_key: candidate_facet_key,
                                existing_incident_count: incident_count,
                                attempted_incident_count: incident_count + 1,
                                max_incident_count: 2,
                                candidate_simplex_uuid: simplex.uuid(),
                                candidate_facet_index: candidate_facet_idx,
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Verifies every vertex key referenced by `simplex` is live in this TDS.
    fn validate_simplex_vertices_exist(
        &self,
        simplex: &Simplex<V, D>,
    ) -> Result<(), TdsConstructionError> {
        for &vkey in simplex.vertices() {
            if !self.vertices.contains_key(vkey) {
                return Err(TdsConstructionError::ValidationError(
                    TdsError::VertexNotFound {
                        vertex_key: vkey,
                        context: "referenced by simplex being inserted".to_string(),
                    },
                ));
            }
        }
        Ok(())
    }

    /// Inserts a simplex while intentionally bypassing topology safety checks in tests.
    #[cfg(test)]
    pub(crate) fn insert_simplex_bypassing_topology_checks_for_test(
        &mut self,
        simplex: Simplex<V, D>,
    ) -> Result<SimplexKey, TdsConstructionError> {
        self.insert_simplex_with_mapping_impl(simplex, SimplexInsertionTopologyCheck::Prechecked)
    }

    /// Sets the auxiliary data on a vertex, returning the previous value.
    ///
    /// This is a safe O(1) operation that modifies only the user-data field.
    /// It does not affect geometry, topology, or Delaunay invariants.
    ///
    /// # Arguments
    ///
    /// * `key` - The key of the vertex to modify
    /// * `data` - The new data value to set, or `None` to clear
    ///
    /// # Returns
    ///
    /// The old `Option<U>` value when the key exists.
    ///
    /// # Errors
    ///
    /// Returns [`TdsMutationError`] if `key` does not identify a vertex in this
    /// TDS. Detached keys are validated at the mutation boundary so stale
    /// handles cannot be mistaken for an existing vertex with no payload.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] TdsMutation(#[from] delaunay::prelude::tds::TdsMutationError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices: [Vertex<i32, 2>; 3] = [
    ///     delaunay::vertex![0.0, 0.0; data = 10i32]?,
    ///     delaunay::vertex![1.0, 0.0; data = 20]?,
    ///     delaunay::vertex![0.0, 1.0; data = 30]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let mut tds = dt.tds().clone();
    /// let Some(key) = tds.vertex_keys().next() else {
    ///     return Ok(());
    /// };
    ///
    /// // Replace existing data
    /// let prev = tds.set_vertex_data(key, Some(99))?;
    /// assert!(prev.is_some());
    ///
    /// // Verify new value
    /// let Some(vertex) = tds.vertex(key) else {
    ///     return Ok(());
    /// };
    /// assert_eq!(vertex.data(), Some(&99));
    ///
    /// // Clear data
    /// let prev = tds.set_vertex_data(key, None)?;
    /// assert_eq!(prev, Some(99));
    /// assert_eq!(tds.vertex(key).and_then(|vertex| vertex.data()), None);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn set_vertex_data(
        &mut self,
        key: VertexKey,
        data: Option<U>,
    ) -> Result<Option<U>, TdsMutationError> {
        let vertex = self.vertices.get_mut(key).ok_or_else(|| {
            TdsMutationError::from(TdsError::VertexNotFound {
                vertex_key: key,
                context: "set_vertex_data".to_string(),
            })
        })?;
        let previous = vertex.data.take();
        vertex.data = data;
        Ok(previous)
    }

    /// Sets the auxiliary data on a simplex, returning the previous value.
    ///
    /// This is a safe O(1) operation that modifies only the user-data field.
    /// It does not affect geometry, topology, or Delaunay invariants.
    ///
    /// # Arguments
    ///
    /// * `key` - The key of the simplex to modify
    /// * `data` - The new data value to set, or `None` to clear
    ///
    /// # Returns
    ///
    /// The old `Option<V>` value when the key exists.
    ///
    /// # Errors
    ///
    /// Returns [`TdsMutationError`] if `key` does not identify a simplex in
    /// this TDS. Detached keys are validated at the mutation boundary so stale
    /// handles cannot be mistaken for an existing simplex with no payload.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] TdsMutation(#[from] delaunay::prelude::tds::TdsMutationError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).simplex_data_type::<i32>().build()?;
    /// let mut tds = dt.tds().clone();
    /// let Some(key) = tds.simplex_keys().next() else {
    ///     return Ok(());
    /// };
    ///
    /// // Set data on a simplex that had no data
    /// let prev = tds.set_simplex_data(key, Some(42))?;
    /// assert_eq!(prev, None);
    ///
    /// // Verify new value
    /// let Some(simplex) = tds.simplex(key) else {
    ///     return Ok(());
    /// };
    /// assert_eq!(simplex.data(), Some(&42));
    ///
    /// // Clear data
    /// let prev = tds.set_simplex_data(key, None)?;
    /// assert_eq!(prev, Some(42));
    /// assert_eq!(tds.simplex(key).and_then(|simplex| simplex.data()), None);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn set_simplex_data(
        &mut self,
        key: SimplexKey,
        data: Option<V>,
    ) -> Result<Option<V>, TdsMutationError> {
        let simplex = self.simplices.get_mut(key).ok_or_else(|| {
            TdsMutationError::from(TdsError::SimplexNotFound {
                simplex_key: key,
                context: "set_simplex_data".to_string(),
            })
        })?;
        let previous = simplex.data.take();
        simplex.data = data;
        Ok(previous)
    }

    /// Fills every live simplex with data computed from the current simplex view.
    ///
    /// This does not alter topology, UUID mappings, neighbor links, or generation:
    /// simplex payloads are orthogonal to the TDS structural invariants.
    #[inline]
    pub(crate) fn fill_simplex_data<F>(&mut self, mut data_for: F)
    where
        F: FnMut(SimplexKey, &Simplex<V, D>) -> V,
    {
        for (simplex_key, simplex) in &mut self.simplices {
            let data = data_for(simplex_key, simplex);
            simplex.data = Some(data);
        }
    }

    /// Removes multiple simplices by their keys in a batch operation.
    ///
    /// This method performs a **local** topology update:
    /// - Removes the requested simplices (and their UUID→Key mappings)
    /// - Clears neighbor back-references in adjacent surviving simplices so no neighbor points at a removed key
    /// - Repairs `Vertex::incident_simplex` for vertices that previously pointed at a removed simplex
    ///
    /// It does **not** attempt to retriangulate the cavity created by the removals.
    /// If the maintained incidence index rejects the batch before simplex
    /// storage is changed, all incidence edits made by the failed batch are
    /// rolled back exactly.
    ///
    /// # Performance
    ///
    /// When neighbor pointers are present and mutually consistent, this touches only the
    /// boundary of the removed region:
    /// - Time: typically `O(#removed_simplices × (D+1)^2)`
    /// - Space: `O(#removed_simplices × (D+1))` for temporary removal metadata
    ///
    /// In degraded states (e.g., after unsafe mutation where neighbor pointers are missing),
    /// it may fall back to a conservative scan to find replacement incident simplices.
    ///
    /// # Arguments
    ///
    /// * `simplex_keys` - The keys of simplices to remove
    ///
    /// # Returns
    ///
    /// The number of simplices successfully removed.
    ///
    /// # Errors
    ///
    /// Returns [`TdsMutationError`] if the maintained vertex-to-simplices
    /// incidence index cannot remove one of the requested simplices without
    /// violating its stored relation. In that case, no simplex storage or UUID
    /// mapping is removed, and any earlier incidence edits from the same batch
    /// are restored.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] TdsMutation(#[from] delaunay::prelude::tds::TdsMutationError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let mut tds = dt.tds().clone();
    /// let Some(simplex_key) = tds.simplex_keys().next() else {
    ///     return Ok(());
    /// };
    /// let removed = tds.remove_simplices_by_keys(&[simplex_key])?;
    /// assert_eq!(removed, 1);
    /// assert_eq!(tds.number_of_simplices(), 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn remove_simplices_by_keys(
        &mut self,
        simplex_keys: &[SimplexKey],
    ) -> Result<usize, TdsMutationError> {
        if simplex_keys.is_empty() {
            return Ok(0);
        }

        let simplex_removals = self.collect_existing_simplex_removals(simplex_keys);
        if simplex_removals.is_empty() {
            return Ok(0);
        }

        let simplices_to_remove = RemovedSimplexMembership::from_removals(&simplex_removals);

        // 1) Collect boundary candidates before mutation while neighbor
        // pointers to the removed simplices are still available.
        let (affected_vertices, candidate_incident) =
            self.collect_incident_repair_frontier(&simplex_removals, &simplices_to_remove);

        // 2) Remove canonical incidence as one rollbackable transaction before
        // planning fallback incident repairs. This lets fallback queries see
        // the post-removal incidence view while storage is still available for
        // stale-key validation.
        let incidence_removals =
            self.remove_simplices_from_vertex_incidence_transactionally(&simplex_removals)?;

        let incident_updates = match self.plan_incident_simplex_repairs(
            &affected_vertices,
            &simplices_to_remove,
            &candidate_incident,
        ) {
            Ok(incident_updates) => incident_updates,
            Err(err) => {
                self.rollback_incidence_removals(&incidence_removals);
                return Err(err);
            }
        };

        // 3) Clear neighbor back-references in surviving simplices.
        self.clear_surviving_neighbor_backrefs(&simplex_removals, &simplices_to_remove);

        // 4) Remove the simplices and update UUID mappings.
        let removed_count = self.remove_simplices_and_update_uuid_mappings(&simplex_removals);
        if removed_count == 0 {
            return Ok(0);
        }

        // 5) Apply the precomputed `incident_simplex` repairs.
        self.apply_incident_simplex_updates(incident_updates);

        // Bump generation once for all removals (neighbors + incidence + simplex storage).
        self.bump_generation();

        Ok(removed_count)
    }

    /// Removes requested simplices from the incidence index as one transaction.
    ///
    /// The public removal contract requires failures to leave caller-visible TDS
    /// storage untouched. This helper performs the incidence phase before
    /// simplex storage deletion and returns exact rollback records so later
    /// repair-planning failures can restore every touched vertex buffer.
    fn remove_simplices_from_vertex_incidence_transactionally(
        &mut self,
        simplex_removals: &[SimplexRemovalRecord],
    ) -> Result<SimplexIncidenceRemovalRecords, TdsMutationError> {
        let mut incidence_removals = SimplexIncidenceRemovalRecords::new();
        for (simplex_key, vertices) in simplex_removals {
            match self
                .vertex_to_simplices
                .remove_simplex(*simplex_key, vertices)
            {
                Ok(removal) => incidence_removals.push(removal),
                Err(source) => {
                    for removal in incidence_removals.iter().rev() {
                        self.vertex_to_simplices.rollback_removed_simplex(removal);
                    }
                    return Err(source.into());
                }
            }
        }

        Ok(incidence_removals)
    }

    /// Restores a successful incidence-removal transaction after later
    /// read-only planning discovers stale canonical incidence.
    fn rollback_incidence_removals(&mut self, incidence_removals: &SimplexIncidenceRemovalRecords) {
        for removal in incidence_removals.iter().rev() {
            self.vertex_to_simplices.rollback_removed_simplex(removal);
        }
    }

    /// Collects live simplex removals and their vertex keys for a batch request.
    ///
    /// Missing simplex keys and duplicate request keys are ignored to preserve
    /// the historical `remove_simplices_by_keys` no-op behavior for stale keys
    /// and make removal idempotent over the requested key set. The collected
    /// vertex buffers are the proof used by the incidence transaction and later
    /// storage removal phases.
    fn collect_existing_simplex_removals(
        &self,
        simplex_keys: &[SimplexKey],
    ) -> SimplexRemovalRecords {
        let mut seen_large = (simplex_keys.len() > CLEANUP_OPERATION_BUFFER_SIZE)
            .then(|| fast_hash_set_with_capacity(simplex_keys.len()));
        let mut simplex_removals = SimplexRemovalRecords::new();

        for &simplex_key in simplex_keys {
            let duplicate = seen_large.as_mut().map_or_else(
                || {
                    simplex_removals
                        .iter()
                        .any(|(removed_key, _)| *removed_key == simplex_key)
                },
                |seen| !seen.insert(simplex_key),
            );

            if duplicate {
                continue;
            }

            let Some(simplex) = self.simplices.get(simplex_key) else {
                continue;
            };

            let vertices = simplex.vertices().iter().copied().collect();
            simplex_removals.push((simplex_key, vertices));
        }

        simplex_removals
    }

    /// Captures affected vertices and surviving incident-simplex candidates
    /// around a removal batch while removed neighbor pointers are still readable.
    fn collect_incident_repair_frontier(
        &self,
        simplex_removals: &[SimplexRemovalRecord],
        simplices_to_remove: &RemovedSimplexMembership<'_>,
    ) -> (AffectedVertexRecords, CandidateIncidentRecords) {
        let expected_vertices = simplex_removals.len().saturating_mul(D.saturating_add(1));
        let mut affected_vertices = AffectedVertexRecords::with_expected_len(expected_vertices);
        let mut candidate_incident = CandidateIncidentRecords::with_expected_len(expected_vertices);

        for (simplex_key, vertices) in simplex_removals {
            for &vertex_key in vertices {
                affected_vertices.insert(vertex_key);
            }

            let Some(simplex) = self.simplices.get(*simplex_key) else {
                continue;
            };

            if let Some(simplex_neighbors) = simplex.neighbor_keys() {
                for (facet_idx, neighbor_key_opt) in simplex_neighbors.enumerate() {
                    let Some(neighbor_key) = neighbor_key_opt else {
                        continue;
                    };

                    if simplices_to_remove.contains(neighbor_key) {
                        continue;
                    }

                    // The neighbor across facet_idx shares the facet consisting
                    // of all vertices except vertices[facet_idx].
                    for (vertex_index, &vkey) in vertices.iter().enumerate() {
                        if vertex_index == facet_idx {
                            continue;
                        }
                        candidate_incident.insert_if_absent(vkey, neighbor_key);
                    }
                }
            }
        }

        (affected_vertices, candidate_incident)
    }

    /// Clears neighbor back-references in surviving simplices after incidence
    /// removal has succeeded.
    fn clear_surviving_neighbor_backrefs(
        &mut self,
        simplex_removals: &[SimplexRemovalRecord],
        simplices_to_remove: &RemovedSimplexMembership<'_>,
    ) {
        for (simplex_key, _) in simplex_removals {
            let Some(simplex) = self.simplices.get(*simplex_key) else {
                continue;
            };

            let mut neighbors: SmallBuffer<Option<SimplexKey>, MAX_PRACTICAL_DIMENSION_SIZE> =
                SmallBuffer::new();
            if let Some(simplex_neighbors) = simplex.neighbor_keys() {
                neighbors.extend(simplex_neighbors);
            }

            for neighbor_key in neighbors.into_iter().flatten() {
                if simplices_to_remove.contains(neighbor_key) {
                    continue; // neighbor is also being removed
                }

                let Some(neighbor_simplex) = self.simplices.get_mut(neighbor_key) else {
                    continue;
                };
                let Some(neighbors_buf) = neighbor_simplex.neighbor_slots_mut() else {
                    continue;
                };

                // Clear the back-reference in the neighbor simplex's neighbor buffer.
                for slot in neighbors_buf.iter_mut() {
                    if slot.simplex_key() == Some(*simplex_key) {
                        *slot = NeighborSlot::Boundary;
                    }
                }
            }
        }
    }

    /// Deletes simplex storage after the incidence transaction has succeeded.
    ///
    /// This phase is intentionally infallible with respect to incidence: all
    /// fallible relation updates happened before any simplex or UUID mapping is
    /// removed, so failed batches can leave storage and generation untouched.
    fn remove_simplices_and_update_uuid_mappings(
        &mut self,
        simplex_removals: &[SimplexRemovalRecord],
    ) -> usize {
        let mut removed_count = 0;

        for (simplex_key, _) in simplex_removals {
            if let Some(removed_simplex) = self.simplices.remove(*simplex_key) {
                self.uuid_to_simplex_key.remove(&removed_simplex.uuid());
                removed_count += 1;
            }
        }

        removed_count
    }

    /// Plans vertex `incident_simplex` repairs after canonical incidence removal.
    ///
    /// Keeping this phase read-only lets stale incidence be detected before any
    /// vertex fields are rewritten. If planning fails, the caller can roll back
    /// the incidence transaction and preserve the public removal atomicity
    /// contract.
    fn plan_incident_simplex_repairs(
        &self,
        affected_vertices: &AffectedVertexRecords,
        simplices_to_remove: &RemovedSimplexMembership<'_>,
        candidate_incident: &CandidateIncidentRecords,
    ) -> Result<IncidentSimplexUpdates, TdsMutationError> {
        let mut incident_updates = IncidentSimplexUpdates::with_capacity(affected_vertices.len());

        affected_vertices.try_for_each(|vk| {
            if let Some(update) =
                self.incident_simplex_repair_update(vk, simplices_to_remove, candidate_incident)?
            {
                incident_updates.push(update);
            }
            Ok::<(), TdsMutationError>(())
        })?;

        Ok(incident_updates)
    }

    /// Applies prevalidated `incident_simplex` replacements for affected vertices.
    ///
    /// All fallible checks happen in [`Self::plan_incident_simplex_repairs`],
    /// so this write phase can remain small and infallible.
    fn apply_incident_simplex_updates(&mut self, incident_updates: IncidentSimplexUpdates) {
        for (vk, new_incident) in incident_updates {
            if let Some(vertex) = self.vertices.get_mut(vk) {
                vertex.set_incident_simplex(new_incident);
            }
        }
    }

    /// Computes the replacement `incident_simplex` for one affected vertex.
    ///
    /// Separating this read-only phase from the later vertex writes lets removal
    /// repair use borrowed incidence and simplex storage without aliasing
    /// mutable vertex access.
    fn incident_simplex_repair_update(
        &self,
        vertex_key: VertexKey,
        simplices_to_remove: &RemovedSimplexMembership<'_>,
        candidate_incident: &CandidateIncidentRecords,
    ) -> Result<Option<(VertexKey, Option<SimplexKey>)>, TdsMutationError> {
        let vertex = self
            .vertices
            .get(vertex_key)
            .ok_or_else(|| TdsError::VertexNotFound {
                vertex_key,
                context: "simplex-removal incident repair planning".to_string(),
            })?;

        let needs_repair = vertex.incident_simplex().is_none_or(|simplex_key| {
            simplices_to_remove.contains(simplex_key) || !self.simplices.contains_key(simplex_key)
        });

        if !needs_repair {
            return Ok(None);
        }

        // Prefer a candidate simplex discovered on the boundary of the removed region.
        let mut new_incident = candidate_incident.get(vertex_key).filter(|&simplex_key| {
            self.simplices
                .get(simplex_key)
                .is_some_and(|simplex| simplex.contains_vertex(vertex_key))
        });

        // Conservative fallback: pick the first remaining simplex that contains this vertex.
        // This is only hit if neighbor pointers were missing or the boundary had no surviving simplex.
        if new_incident.is_none() {
            new_incident =
                self.valid_surviving_incident_simplex(vertex_key, simplices_to_remove)?;
        }

        Ok(Some((vertex_key, new_incident)))
    }

    /// Finds one surviving incident simplex and validates canonical incidence.
    ///
    /// Returning a typed error for dangling or misaligned incidence prevents
    /// simplex removal from silently repairing through corrupted topology.
    fn valid_surviving_incident_simplex(
        &self,
        vertex_key: VertexKey,
        simplices_to_remove: &RemovedSimplexMembership<'_>,
    ) -> Result<Option<SimplexKey>, TdsMutationError> {
        if !self.vertices.contains_key(vertex_key) {
            return Err(TdsError::VertexNotFound {
                vertex_key,
                context: "simplex-removal incident repair planning".to_string(),
            }
            .into());
        }

        let Some(simplex_key) = self.first_simplex_containing_vertex(vertex_key) else {
            return Ok(None);
        };

        if simplices_to_remove.contains(simplex_key) {
            return Err(TdsError::RemovedSimplexStillIncident {
                vertex_key,
                simplex_key,
            }
            .into());
        }

        let Some(simplex) = self.simplices.get(simplex_key) else {
            return Err(TdsError::SimplexNotFound {
                simplex_key,
                context: format!("vertex-to-simplices repair planning for vertex {vertex_key:?}"),
            }
            .into());
        };

        if !simplex.contains_vertex(vertex_key) {
            return Err(TdsError::VertexIncidenceMismatch {
                vertex_key,
                simplex_key,
            }
            .into());
        }

        Ok(Some(simplex_key))
    }

    pub(crate) fn remove_vertex(
        &mut self,
        vertex_key: VertexKey,
    ) -> Result<usize, TdsMutationError> {
        // Look up the vertex to get its UUID before removal
        let Some(vertex) = self.vertex(vertex_key) else {
            return Ok(0); // Vertex not found, nothing to remove
        };
        let uuid = vertex.uuid();

        // Find all simplices containing this vertex.
        let simplices_to_remove: SimplexRemovalBuffer =
            self.simplex_keys_containing_vertex(vertex_key).collect();

        // Remove all simplices containing the vertex.
        //
        // `remove_simplices_by_keys()` clears neighbor back-references that would otherwise dangle and
        // incrementally repairs `incident_simplex` pointers for vertices that referenced removed simplices.
        let simplices_removed = self.remove_simplices_by_keys(&simplices_to_remove)?;

        self.remove_vertex_incidence(vertex_key)?;

        // Remove the vertex itself.
        self.vertices.remove(vertex_key);
        self.uuid_to_vertex_key.remove(&uuid);
        self.refresh_incomplete_construction_state();
        // Topology changed; invalidate caches
        self.bump_generation();

        Ok(simplices_removed)
    }

    /// Remove an isolated vertex (one with no incident simplices) from the TDS.
    ///
    /// This removes only the vertex and its UUID mapping. It does **not** touch
    /// any simplices. If the vertex has an incident simplex, this is a no-op.
    pub(crate) fn remove_isolated_vertex(
        &mut self,
        vertex_key: VertexKey,
    ) -> Result<(), TdsMutationError> {
        let Some(vertex) = self.vertex(vertex_key) else {
            return Ok(());
        };
        // Only remove if truly isolated according to the canonical incidence index.
        if self
            .simplex_keys_containing_vertex(vertex_key)
            .next()
            .is_some()
        {
            return Ok(());
        }
        let uuid = vertex.uuid();
        self.remove_vertex_incidence(vertex_key)?;
        self.vertices.remove(vertex_key);
        self.uuid_to_vertex_key.remove(&uuid);
        self.refresh_incomplete_construction_state();
        self.bump_generation();
        Ok(())
    }

    // =========================================================================
    // KEY-BASED NEIGHBOR OPERATIONS
    // =========================================================================

    /// Finds neighbor simplex keys for a given simplex without UUID lookups.
    ///
    /// This is the key-based version of neighbor retrieval that avoids
    /// UUID→Key conversions in the hot path.
    ///
    /// # Arguments
    ///
    /// * `simplex_key` - The key of the simplex whose neighbors to find
    ///
    /// # Returns
    ///
    /// A buffer of `Option<SimplexKey>` where `None` indicates no neighbor
    /// at that position (boundary facet). Uses stack allocation for typical dimensions.
    ///
    /// **Special case**: If the simplex does not exist (invalid `simplex_key`), returns a buffer
    /// filled with `None` values. This is a non-panicking fallback that allows callers to
    /// distinguish "simplex missing" from "no neighbors assigned" by checking simplex existence
    /// separately with `simplex()` if needed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let tds = dt.tds();
    /// let Some((simplex_key, _)) = tds.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// // Get neighbors for existing simplex
    /// let neighbors = tds.find_neighbors_by_key(simplex_key);
    /// assert_eq!(neighbors.len(), 3); // D+1 for 2D
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn find_neighbors_by_key(
        &self,
        simplex_key: SimplexKey,
    ) -> NeighborBuffer<Option<SimplexKey>> {
        let mut neighbors = NeighborBuffer::new();
        neighbors.resize(D + 1, None);

        let Some(simplex) = self.simplex(simplex_key) else {
            return neighbors;
        };

        if let Some(neighbors_from_simplex) = simplex.neighbor_keys() {
            // Use zip to avoid potential OOB if neighbors_from_simplex.len() > D+1 (malformed data)
            for (slot, neighbor_key_opt) in neighbors.iter_mut().zip(neighbors_from_simplex) {
                *slot = neighbor_key_opt;
            }
        }

        neighbors
    }

    fn validate_neighbor_update_matches_facet_incidence(
        &self,
        simplex_key: SimplexKey,
        neighbors: &[Option<SimplexKey>],
    ) -> Result<(), TdsError> {
        let simplex = self
            .simplices
            .get(simplex_key)
            .ok_or_else(|| TdsError::SimplexNotFound {
                simplex_key,
                context: "validate_neighbor_update_matches_facet_incidence".to_string(),
            })?;

        let facet_to_simplices = self.build_facet_to_simplices_map()?;
        for (facet_idx, proposed_neighbor) in neighbors.iter().copied().enumerate() {
            let facet_key = self.facet_key_for_simplex_facet(simplex_key, facet_idx)?;
            let Some(simplex_facet_pairs) = facet_to_simplices.get(&facet_key) else {
                return Err(TdsError::InvalidNeighbors {
                    reason: NeighborValidationError::FacetIncidenceMissing {
                        simplex_key,
                        simplex_uuid: simplex.uuid(),
                        facet_index: facet_idx,
                        facet_key,
                    },
                });
            };

            let expected_neighbor = match simplex_facet_pairs.as_slice() {
                [_] => None,
                [a, b] => {
                    if a.simplex_key() == simplex_key && a.facet_index() as usize == facet_idx {
                        Some(b.simplex_key())
                    } else if b.simplex_key() == simplex_key
                        && b.facet_index() as usize == facet_idx
                    {
                        Some(a.simplex_key())
                    } else {
                        return Err(TdsError::InvalidNeighbors {
                            reason:
                                NeighborValidationError::FacetIncidenceDoesNotReferenceSimplex {
                                    simplex_key,
                                    simplex_uuid: simplex.uuid(),
                                    facet_index: facet_idx,
                                    facet_key,
                                },
                        });
                    }
                }
                _ => {
                    return Err(TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::FacetIncidenceMultiplicity {
                            simplex_key,
                            simplex_uuid: simplex.uuid(),
                            facet_index: facet_idx,
                            facet_key,
                            simplex_count: simplex_facet_pairs.len(),
                        },
                    });
                }
            };

            if proposed_neighbor == Some(simplex_key)
                && expected_neighbor.is_none()
                && Self::allows_periodic_self_neighbor(simplex)
            {
                continue;
            }

            if proposed_neighbor != expected_neighbor {
                return Err(TdsError::InvalidNeighbors {
                    reason: NeighborValidationError::NeighborIncidenceMismatch {
                        simplex_key,
                        simplex_uuid: simplex.uuid(),
                        facet_index: facet_idx,
                        proposed_neighbor,
                        expected_neighbor,
                    },
                });
            }
        }

        Ok(())
    }

    fn set_simplex_neighbors_normalized(
        simplex: &mut Simplex<V, D>,
        neighbors: &[Option<SimplexKey>],
    ) -> Result<(), TdsError> {
        let simplex_id = simplex.uuid();
        simplex
            .set_neighbors_from_keys(neighbors.iter().copied())
            .map_err(|source| TdsError::InvalidSimplex { simplex_id, source })
    }

    fn ensure_neighbor_buffer(
        simplex: &mut Simplex<V, D>,
    ) -> Result<&mut SmallBuffer<NeighborSlot, MAX_PRACTICAL_DIMENSION_SIZE>, TdsError> {
        if simplex.neighbor_slots().is_none() {
            let simplex_id = simplex.uuid();
            simplex
                .set_neighbors_from_keys((0..=D).map(|_| None))
                .map_err(|source| TdsError::InvalidSimplex { simplex_id, source })?;
        }
        let simplex_id = simplex.uuid();
        simplex
            .try_ensure_neighbors_buffer_mut()
            .map_err(|source| TdsError::InvalidSimplex { simplex_id, source })
    }

    fn set_neighbor_slot(
        simplex: &mut Simplex<V, D>,
        facet_idx: usize,
        neighbor: Option<SimplexKey>,
    ) -> Result<(), TdsError> {
        let neighbors = Self::ensure_neighbor_buffer(simplex)?;
        let Some(slot) = neighbors.get_mut(facet_idx) else {
            return Err(TdsError::InvalidNeighbors {
                reason: NeighborValidationError::NeighborSlotOutOfBounds {
                    facet_index: facet_idx,
                    slot_count: neighbors.len(),
                },
            });
        };
        *slot = NeighborSlot::from_neighbor_key(neighbor);
        Ok(())
    }

    fn reciprocal_neighbor_updates_for_neighbor_update(
        &self,
        simplex_key: SimplexKey,
        neighbors: &[Option<SimplexKey>],
    ) -> Result<Vec<(SimplexKey, usize, Option<SimplexKey>)>, TdsError> {
        let simplex = self
            .simplices
            .get(simplex_key)
            .ok_or_else(|| TdsError::SimplexNotFound {
                simplex_key,
                context: "set_neighbors_by_key".to_string(),
            })?;
        let old_neighbors: Vec<Option<SimplexKey>> = simplex
            .neighbor_keys()
            .map_or_else(|| vec![None; D + 1], Iterator::collect);

        let mut reciprocal_updates = Vec::new();
        self.collect_stale_reciprocal_neighbor_updates(
            simplex_key,
            simplex,
            &old_neighbors,
            neighbors,
            &mut reciprocal_updates,
        )?;
        self.collect_new_reciprocal_neighbor_updates(
            simplex_key,
            simplex,
            neighbors,
            &mut reciprocal_updates,
        )?;
        Ok(reciprocal_updates)
    }

    fn collect_stale_reciprocal_neighbor_updates(
        &self,
        simplex_key: SimplexKey,
        simplex: &Simplex<V, D>,
        old_neighbors: &[Option<SimplexKey>],
        new_neighbors: &[Option<SimplexKey>],
        reciprocal_updates: &mut Vec<(SimplexKey, usize, Option<SimplexKey>)>,
    ) -> Result<(), TdsError> {
        for (facet_idx, old_neighbor_key) in old_neighbors.iter().copied().enumerate() {
            let Some(old_neighbor_key) = old_neighbor_key else {
                continue;
            };
            if old_neighbor_key == simplex_key
                || new_neighbors
                    .iter()
                    .copied()
                    .any(|neighbor_key| neighbor_key == Some(old_neighbor_key))
            {
                continue;
            }

            let old_neighbor_simplex =
                self.simplices
                    .get(old_neighbor_key)
                    .ok_or_else(|| TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::MissingNeighborSimplex {
                            simplex_key,
                            simplex_uuid: simplex.uuid(),
                            facet_index: facet_idx,
                            neighbor_key: old_neighbor_key,
                            context: "clearing stale reciprocal neighbor".to_string(),
                        },
                    })?;
            let mirror_idx = simplex
                .mirror_facet_index(facet_idx, old_neighbor_simplex)
                .ok_or_else(|| TdsError::InvalidNeighbors {
                    reason: NeighborValidationError::MirrorFacetMissing {
                        simplex_uuid: simplex.uuid(),
                        facet_index: facet_idx,
                        neighbor_uuid: old_neighbor_simplex.uuid(),
                        context: "clearing old back-reference".to_string(),
                    },
                })?;
            let back_ref = old_neighbor_simplex.neighbor_key(mirror_idx).flatten();
            if back_ref == Some(simplex_key) {
                reciprocal_updates.push((old_neighbor_key, mirror_idx, None));
            }
        }

        Ok(())
    }

    fn collect_new_reciprocal_neighbor_updates(
        &self,
        simplex_key: SimplexKey,
        simplex: &Simplex<V, D>,
        neighbors: &[Option<SimplexKey>],
        reciprocal_updates: &mut Vec<(SimplexKey, usize, Option<SimplexKey>)>,
    ) -> Result<(), TdsError> {
        for (facet_idx, neighbor_key) in neighbors.iter().copied().enumerate() {
            let Some(neighbor_key) = neighbor_key else {
                continue;
            };
            if neighbor_key == simplex_key {
                continue;
            }

            let neighbor_simplex =
                self.simplices
                    .get(neighbor_key)
                    .ok_or_else(|| TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::MissingNeighborSimplex {
                            simplex_key,
                            simplex_uuid: simplex.uuid(),
                            facet_index: facet_idx,
                            neighbor_key,
                            context: "setting reciprocal neighbor".to_string(),
                        },
                    })?;
            let mirror_idx = simplex
                .mirror_facet_index(facet_idx, neighbor_simplex)
                .ok_or_else(|| TdsError::InvalidNeighbors {
                    reason: NeighborValidationError::MirrorFacetMissing {
                        simplex_uuid: simplex.uuid(),
                        facet_index: facet_idx,
                        neighbor_uuid: neighbor_simplex.uuid(),
                        context: "setting back-reference".to_string(),
                    },
                })?;
            let existing_back_ref = neighbor_simplex.neighbor_key(mirror_idx).flatten();
            if let Some(existing_back_ref) = existing_back_ref
                && existing_back_ref != simplex_key
            {
                return Err(TdsError::InvalidNeighbors {
                    reason: NeighborValidationError::ExistingBackReferenceConflict {
                        neighbor_uuid: neighbor_simplex.uuid(),
                        mirror_index: mirror_idx,
                        existing_back_ref,
                        requested_back_ref: simplex_key,
                    },
                });
            }
            reciprocal_updates.push((neighbor_key, mirror_idx, Some(simplex_key)));
        }

        Ok(())
    }

    /// Sets neighbor relationships using simplex keys directly.
    ///
    /// # Positional Semantics (Critical Topological Invariant)
    ///
    /// **`neighbors[i]` must be the neighbor opposite to `vertices[i]`**
    ///
    /// This means the two simplices share facet `i`, which contains all vertices **except** vertex `i`.
    ///
    /// ## Example: 3D Tetrahedron
    ///
    /// For a simplex with vertices `[v0, v1, v2, v3]`:
    /// - `neighbors[0]` shares facet `[v1, v2, v3]` (opposite v0)
    /// - `neighbors[1]` shares facet `[v0, v2, v3]` (opposite v1)
    /// - `neighbors[2]` shares facet `[v0, v1, v3]` (opposite v2)
    /// - `neighbors[3]` shares facet `[v0, v1, v2]` (opposite v3)
    ///
    /// **This invariant is always validated** via `validate_neighbor_topology()`.
    ///
    /// # Arguments
    ///
    /// * `simplex_key` - The key of the simplex to update
    /// * `neighbors` - The new neighbor keys (must have length D+1)
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, or an error if validation fails.
    ///
    /// # Errors
    ///
    /// Returns a `TdsMutationError` if:
    /// - The simplex with the given key doesn't exist
    /// - The neighbor vector length is not D+1
    /// - Any neighbor key references a non-existent simplex
    /// - **The topological invariant is violated** (neighbor\[i\] not opposite vertex\[i\])
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsMutation(#[from] delaunay::prelude::tds::TdsMutationError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let mut tds = dt.tds().clone();
    /// let Some(simplex_key) = tds.simplex_keys().next() else {
    ///     return Ok(());
    /// };
    /// let neighbors = vec![None; 3];
    /// tds.set_neighbors_by_key(simplex_key, &neighbors)?;
    /// let Some(simplex) = tds.simplex(simplex_key) else {
    ///     return Ok(());
    /// };
    /// assert!(simplex
    ///     .neighbors()
    ///     .is_some_and(|mut neighbors| neighbors.all(|neighbor| neighbor.is_none())));
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_neighbors_by_key(
        &mut self,
        simplex_key: SimplexKey,
        neighbors: &[Option<SimplexKey>],
    ) -> Result<(), TdsMutationError> {
        // Validate the topological invariant before applying changes
        // (includes length check: neighbors.len() == D+1)
        self.validate_neighbor_topology(simplex_key, neighbors)?;
        self.validate_neighbor_update_matches_facet_incidence(simplex_key, neighbors)?;
        let reciprocal_updates =
            self.reciprocal_neighbor_updates_for_neighbor_update(simplex_key, neighbors)?;

        let simplex_uuid = {
            let simplex =
                self.simplex_mut(simplex_key)
                    .ok_or_else(|| TdsError::SimplexNotFound {
                        simplex_key,
                        context: "set_neighbors_by_key".to_string(),
                    })?;
            let simplex_uuid = simplex.uuid();
            Self::set_simplex_neighbors_normalized(simplex, neighbors)?;
            simplex_uuid
        };

        for (neighbor_key, mirror_idx, back_reference) in reciprocal_updates {
            let neighbor_simplex =
                self.simplices
                    .get_mut(neighbor_key)
                    .ok_or_else(|| TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::MissingNeighborSimplex {
                            simplex_key,
                            simplex_uuid,
                            facet_index: mirror_idx,
                            neighbor_key,
                            context: "applying reciprocal neighbor update".to_string(),
                        },
                    })?;
            Self::set_neighbor_slot(neighbor_simplex, mirror_idx, back_reference)?;
        }

        // Topology changed; invalidate caches
        self.bump_generation();
        Ok(())
    }

    /// Assigns incident simplices to vertices in the triangulation.
    ///
    /// This method establishes a mapping from each vertex to one of the simplices that contains it,
    /// which is useful for various geometric queries and traversals. For each an arbitrary
    /// incident simplex is selected from the simplices that contain that vertex.
    ///
    /// Note: Many topology-mutating operations (like [`Tds::remove_simplices_by_keys`](Self::remove_simplices_by_keys))
    /// attempt to repair `incident_simplex` incrementally for affected vertices. This method exists as a
    /// conservative **full rebuild** after bulk changes (deserialization, large repairs, etc.).
    ///
    /// # Returns
    ///
    /// `Ok(())` if incident simplices were successfully assigned to all vertices,
    /// otherwise a `TdsMutationError`.
    ///
    /// # Errors
    ///
    /// Returns a `TdsMutationError` if a simplex references a non-existent vertex key
    /// (`VertexNotFound`).
    ///
    /// # Algorithm
    ///
    /// 1. Build a fresh vertex-to-simplices index from simplex storage
    /// 2. Replace the maintained index only after every referenced vertex key resolves
    /// 3. Set each vertex's optional `incident_simplex` hint from the rebuilt index
    ///
    /// # Performance
    ///
    /// This method rebuilds incidence **globally** by scanning all simplices:
    /// - Time: O(#vertices + #simplices × (D+1))
    /// - Space: O(#vertices + #simplex incidences) for the rebuilt vertex→simplices map
    ///
    /// It is intended for repair/validation paths after bulk topology changes, not as a per-step
    /// hot-path update.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsMutation(#[from] delaunay::prelude::tds::TdsMutationError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let mut tds = dt.tds().clone();
    /// tds.assign_incident_simplices()?;
    /// let all_assigned = tds.vertices().all(|(_, v)| v.incident_simplex().is_some());
    /// assert!(all_assigned);
    /// # Ok(())
    /// # }
    /// ```
    pub fn assign_incident_simplices(&mut self) -> Result<(), TdsMutationError> {
        self.rebuild_vertex_to_simplices_index()?;

        // `incident_simplex` is only a per-vertex hint. The complete incidence
        // relation lives in `vertex_to_simplices`, so derive the hint from that
        // canonical map instead of scanning simplex storage a second time.
        let incident_updates: Vec<_> = self
            .vertices
            .keys()
            .map(|vertex_key| {
                let incident_simplex = self.vertex_to_simplices.simplex_keys(vertex_key).next();
                (vertex_key, incident_simplex)
            })
            .collect();

        for (vertex_key, incident_simplex) in incident_updates {
            if let Some(vertex) = self.vertices.get_mut(vertex_key) {
                vertex.set_incident_simplex(incident_simplex);
            }
        }

        self.bump_generation();
        Ok(())
    }

    /// Clears all neighbor relationships between simplices in the triangulation.
    ///
    /// This intentionally leaves the TDS below Level-2 structural validity until
    /// neighbors are rebuilt. It is crate-internal so callers cannot observe or
    /// depend on invalid intermediate topology through the public API.
    #[cfg(test)]
    #[inline]
    pub(crate) fn clear_all_neighbors(&mut self) {
        for simplex in self.simplices.values_mut() {
            simplex.clear_neighbors();
        }
        // Topology changed; invalidate caches.
        self.bump_generation();
    }

    #[expect(
        clippy::too_many_lines,
        reason = "orientation normalization is unchanged; simplex nomenclature makes existing names longer"
    )]
    pub(crate) fn normalize_coherent_orientation(&mut self) -> Result<(), TdsError> {
        let mut flip_assignment: FastHashMap<SimplexKey, bool> =
            fast_hash_map_with_capacity(self.simplices.len());

        for root_simplex_key in self.simplices.keys() {
            if flip_assignment.contains_key(&root_simplex_key) {
                continue;
            }

            flip_assignment.insert(root_simplex_key, false);
            let mut queue = VecDeque::new();
            queue.push_back(root_simplex_key);

            while let Some(simplex_key) = queue.pop_front() {
                let this_flip_state = *flip_assignment.get(&simplex_key).ok_or_else(|| {
                    TdsError::InconsistentDataStructure {
                        message: format!(
                            "Missing flip assignment for simplex {simplex_key:?} during orientation normalization",
                        ),
                    }
                })?;

                let simplex =
                    self.simplices
                        .get(simplex_key)
                        .ok_or_else(|| TdsError::SimplexNotFound {
                            simplex_key,
                            context: "orientation normalization traversal".to_string(),
                        })?;
                let Some(neighbors) = simplex.neighbor_keys() else {
                    continue;
                };

                for (facet_idx, neighbor_key_opt) in neighbors.enumerate() {
                    let Some(neighbor_key) = neighbor_key_opt else {
                        continue;
                    };
                    if neighbor_key == simplex_key && Self::allows_periodic_self_neighbor(simplex) {
                        continue;
                    }

                    let neighbor_simplex =
                        self.simplices
                            .get(neighbor_key)
                            .ok_or_else(|| TdsError::SimplexNotFound {
                                simplex_key: neighbor_key,
                                context: format!(
                                    "neighbor of simplex {simplex_key:?} during orientation normalization"
                                ),
                            })?;
                    // Periodic-lifted adjacencies do not have a unique canonical orientation at this
                    // structural layer because the embedding depends on lattice representative choice.
                    // Skip normalization constraints for these pairs.
                    if simplex.periodic_vertex_offsets().is_some()
                        || neighbor_simplex.periodic_vertex_offsets().is_some()
                    {
                        continue;
                    }
                    let mirror_idx = simplex
                        .mirror_facet_index(facet_idx, neighbor_simplex)
                        .ok_or_else(|| TdsError::InvalidNeighbors {
                            reason: NeighborValidationError::MirrorFacetMissing {
                                simplex_uuid: simplex.uuid(),
                                facet_index: facet_idx,
                                neighbor_uuid: neighbor_simplex.uuid(),
                                context: "orientation normalization".to_string(),
                            },
                        })?;

                    let (currently_coherent, _, _) = Self::facet_permutation_parity(
                        simplex,
                        facet_idx,
                        neighbor_simplex,
                        mirror_idx,
                    )?;

                    // Flipping exactly one endpoint toggles the coherence state for this edge.
                    let requires_relative_flip = !currently_coherent;
                    let required_neighbor_flip_state = this_flip_state ^ requires_relative_flip;

                    if let Some(existing_neighbor_flip_state) = flip_assignment.get(&neighbor_key) {
                        if *existing_neighbor_flip_state != required_neighbor_flip_state {
                            return Err(TdsError::InconsistentDataStructure {
                                message: format!(
                                    "Contradictory orientation constraints while normalizing simplices {:?} and {:?}",
                                    simplex.uuid(),
                                    neighbor_simplex.uuid(),
                                ),
                            });
                        }
                    } else {
                        flip_assignment.insert(neighbor_key, required_neighbor_flip_state);
                        queue.push_back(neighbor_key);
                    }
                }
            }
        }

        let mut flipped_any = false;
        for (simplex_key, should_flip) in flip_assignment {
            if !should_flip {
                continue;
            }
            let simplex =
                self.simplices
                    .get_mut(simplex_key)
                    .ok_or_else(|| TdsError::SimplexNotFound {
                        simplex_key,
                        context: "applying orientation normalization".to_string(),
                    })?;
            if simplex.number_of_vertices() >= 2 {
                simplex.swap_vertex_slots(0, 1);
                flipped_any = true;
            }
        }
        if flipped_any {
            self.bump_generation();
        }

        Ok(())
    }

    /// Validates coherent orientation for simplices touched by a local mutation.
    ///
    /// This checks every adjacency owned by `simplices`, including adjacencies to
    /// simplices outside the supplied slice. It is intended for insertion and local
    /// repair paths that already know the mutation frontier and want Level-2
    /// orientation safety without a full-TDS traversal.
    pub(crate) fn validate_coherent_orientation_for_simplices(
        &self,
        simplices: &[SimplexKey],
    ) -> Result<(), TdsError> {
        for &simplex_key in simplices {
            let simplex =
                self.simplices
                    .get(simplex_key)
                    .ok_or_else(|| TdsError::SimplexNotFound {
                        simplex_key,
                        context: "local orientation validation scope".to_string(),
                    })?;
            let Some(neighbors) = simplex.neighbor_keys() else {
                continue;
            };

            for (facet_idx, neighbor_key_opt) in neighbors.enumerate() {
                let Some(neighbor_key) = neighbor_key_opt else {
                    continue;
                };
                if neighbor_key == simplex_key && Self::allows_periodic_self_neighbor(simplex) {
                    continue;
                }

                let neighbor_simplex =
                    self.simplices
                        .get(neighbor_key)
                        .ok_or_else(|| TdsError::SimplexNotFound {
                            simplex_key: neighbor_key,
                            context: format!(
                                "neighbor of simplex {simplex_key:?} during local orientation validation",
                            ),
                        })?;

                let (mirror_idx, uses_periodic_offsets) = Self::orientation_mirror_facet_index(
                    simplex,
                    facet_idx,
                    neighbor_simplex,
                    "local orientation validation",
                )?;
                let observed_back_reference = neighbor_simplex.neighbor_key(mirror_idx).flatten();
                if observed_back_reference != Some(simplex_key) {
                    return Err(TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::BackReferenceMismatch {
                            simplex_key,
                            simplex_uuid: simplex.uuid(),
                            facet_index: facet_idx,
                            neighbor_key,
                            neighbor_uuid: neighbor_simplex.uuid(),
                            mirror_index: mirror_idx,
                            observed: observed_back_reference,
                            context: "local orientation validation".to_string(),
                        },
                    });
                }
                if uses_periodic_offsets {
                    continue;
                }

                let simplex1_facet_vertices =
                    Self::facet_vertices_in_simplex_order(simplex, facet_idx)?;
                let simplex2_facet_vertices =
                    Self::facet_vertices_in_simplex_order(neighbor_simplex, mirror_idx)?;
                let (coherent, observed_odd_permutation, expected_odd_permutation) =
                    Self::facet_permutation_parity(
                        simplex,
                        facet_idx,
                        neighbor_simplex,
                        mirror_idx,
                    )?;
                if !coherent {
                    return Err(TdsError::OrientationViolation {
                        simplex1_key: simplex_key,
                        simplex1_uuid: simplex.uuid(),
                        simplex2_key: neighbor_key,
                        simplex2_uuid: neighbor_simplex.uuid(),
                        simplex1_facet_index: facet_idx,
                        simplex2_facet_index: mirror_idx,
                        facet_vertices: simplex1_facet_vertices.into_iter().collect(),
                        simplex2_facet_vertices: simplex2_facet_vertices.into_iter().collect(),
                        observed_odd_permutation,
                        expected_odd_permutation,
                    });
                }
            }
        }

        Ok(())
    }

    /// Removes duplicate simplices with identical vertex sets.
    ///
    /// Returns the number of duplicate simplices that were removed.
    ///
    /// Duplicate removal is applied to a cloned trial [`Tds`], then the
    /// topology (neighbor relationships and incident simplices) is rebuilt to
    /// maintain data structure invariants and prevent stale references. If the
    /// rebuild or validation fails, the original structure is left unchanged.
    ///
    /// When duplicates are present, the rollback guarantee is implemented by
    /// cloning the current [`Tds`] before removal. This keeps failed mutations
    /// atomic, but the snapshot cost is linear in the size of the stored
    /// topology. The method therefore requires the stored coordinates and
    /// payloads to be cloneable so the trial structure can preserve them.
    ///
    /// # Errors
    ///
    /// Returns a [`TdsMutationError`] if:
    /// - Vertex keys cannot be retrieved for any simplex (data structure corruption)
    /// - Neighbor assignment fails after simplex removal
    /// - Incident simplex assignment fails after simplex removal
    /// - Validation fails after topology rebuild
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::tds::{Tds, TdsMutationError};
    ///
    /// # fn main() -> Result<(), TdsMutationError> {
    /// let mut tds: Tds<(), (), 2> = Tds::empty();
    /// let removed = tds.remove_duplicate_simplices()?;
    /// assert_eq!(removed, 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn remove_duplicate_simplices(&mut self) -> Result<usize, TdsMutationError>
    where
        U: Clone,
        V: Clone,
    {
        let mut unique_simplices = FastHashMap::default();
        let mut simplices_to_remove = SimplexRemovalBuffer::new();

        for simplex_key in self.simplices.keys() {
            let vertices = self.simplex_vertices(simplex_key)?;
            let vertex_uuid_offsets =
                self.build_periodic_vertex_uuid_offsets(simplex_key, vertices)?;

            match unique_simplices.entry(vertex_uuid_offsets) {
                Entry::Occupied(_) => {
                    simplices_to_remove.push(simplex_key);
                }
                Entry::Vacant(e) => {
                    e.insert(simplex_key);
                }
            }
        }

        let duplicate_count = simplices_to_remove.len();

        if duplicate_count == 0 {
            return Ok(0);
        }

        let original_generation = self.generation();
        let mut trial = self.clone_for_rollback();
        trial.generation = Arc::new(AtomicU64::new(original_generation));
        let removed = trial.remove_simplices_by_keys(&simplices_to_remove)?;
        let rebuild_result = (|| -> Result<(), TdsMutationError> {
            trial.assign_neighbors().map_err(TdsMutationError::from)?;
            trial.assign_incident_simplices()?;
            trial.is_valid().map_err(TdsMutationError::from)?;
            Ok(())
        })();

        if let Err(error) = rebuild_result {
            self.generation
                .store(original_generation, Ordering::Relaxed);
            return Err(error);
        }

        *self = trial;
        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DelaunayTriangulation;
    use crate::builder::DelaunayTriangulationBuilder;
    use crate::core::algorithms::incremental_insertion::InsertionError;
    use crate::core::simplex::Simplex;
    use crate::core::tds::errors::TriangulationConstructionState;
    use crate::geometry::point::Point;
    use crate::vertex;
    use slotmap::KeyData;
    use std::assert_matches;
    use uuid::Uuid;

    // =============================================================================
    // TEST HELPER FUNCTIONS
    // =============================================================================

    fn vertex_with_uuid<U, const D: usize>(
        point: Point<D>,
        uuid: Uuid,
        data: Option<U>,
    ) -> Vertex<U, D> {
        Vertex::try_new_with_uuid(point, uuid, data).expect("Failed to build vertex")
    }

    fn initial_simplex_vertices_3d() -> [Vertex<(), 3>; 4] {
        [
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ]
    }

    // =============================================================================
    // VERTEX ADDITION TESTS
    // =============================================================================

    #[test]
    fn test_add_vertex_duplicate_coordinates_rejected() {
        let initial_vertices = initial_simplex_vertices_3d();
        let mut dt = DelaunayTriangulation::builder(&initial_vertices)
            .build()
            .unwrap();

        let vertex = vertex!([1.0, 2.0, 3.0]).unwrap();
        let duplicate = vertex!([1.0, 2.0, 3.0]).unwrap();
        dt.insert_vertex(vertex).unwrap();

        // Same coordinates again (distinct UUID, constructed via Vertex smart constructors)
        let result = dt.insert_vertex(duplicate);
        assert_matches!(
            &result,
            Err(InsertionError::DuplicateCoordinates { .. }),
            "insert() should reject duplicate coordinates created via Vertex::try_new (before UUID), got: {result:?}"
        );
    }

    #[test]
    fn test_add_vertex_duplicate_uuid_rejected() {
        let initial_vertices = initial_simplex_vertices_3d();
        let mut dt = DelaunayTriangulation::builder(&initial_vertices)
            .build()
            .unwrap();

        let vertex1 = vertex!([1.0, 2.0, 3.0]).unwrap();
        let uuid1 = vertex1.uuid();
        dt.insert_vertex(vertex1).unwrap();

        let vertex2 = vertex_with_uuid(
            Point::try_new([4.0, 5.0, 6.0]).expect("finite point coordinates"),
            uuid1,
            None,
        );
        let result = dt.insert_vertex(vertex2);
        assert_matches!(
            &result,
            Err(InsertionError::DuplicateUuid {
                entity: EntityKind::Vertex,
                ..
            }),
            "Same UUID with different coordinates should fail with DuplicateUuid"
        );
    }

    #[test]
    fn test_add_vertex_increases_counts_and_leaves_tds_valid() {
        let initial_vertices = initial_simplex_vertices_3d();
        let mut dt = DelaunayTriangulation::builder(&initial_vertices)
            .build()
            .unwrap();
        let initial_simplex_count = dt.number_of_simplices();

        let new_vertex = vertex!([0.5, 0.5, 0.5]).unwrap();
        dt.insert_vertex(new_vertex).unwrap();

        assert_eq!(dt.number_of_vertices(), 5);
        assert!(
            dt.number_of_simplices() >= initial_simplex_count,
            "Simplex count should not decrease"
        );
        assert!(
            dt.as_triangulation().tds.is_valid().is_ok(),
            "TDS should remain valid"
        );
    }

    #[test]
    fn test_add_vertex_is_accessible_by_uuid_and_coordinates() {
        let initial_vertices = initial_simplex_vertices_3d();
        let mut dt = DelaunayTriangulation::builder(&initial_vertices)
            .build()
            .unwrap();

        let vertex = vertex!([1.0, 2.0, 3.0]).unwrap();
        let uuid = vertex.uuid();
        dt.insert_vertex(vertex).unwrap();
        assert_eq!(dt.number_of_vertices(), 5);

        // Vertex should be findable by UUID.
        let vertex_key = dt.as_triangulation().tds.vertex_key_from_uuid(&uuid);
        assert!(
            vertex_key.is_some(),
            "Added vertex should be findable by UUID"
        );

        // Vertex should be in the vertices collection.
        let stored_vertex = dt
            .as_triangulation()
            .tds
            .vertex(vertex_key.unwrap())
            .unwrap();
        let coords = *stored_vertex.point().coords();
        let expected = [1.0, 2.0, 3.0];
        assert!(
            coords
                .iter()
                .zip(expected.iter())
                .all(|(a, b)| (a - b).abs() < 1e-10),
            "Stored coordinates should match: got {coords:?}, expected {expected:?}"
        );
    }

    // =============================================================================
    // VERTEX REMOVAL TESTS
    // =============================================================================

    #[test]
    fn test_remove_vertex_maintains_topology_consistency() {
        // Test that remove_vertex properly clears dangling neighbor references
        // Create a triangulation with multiple simplices
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.5, 1.0]).unwrap(),
            vertex!([1.5, 1.0]).unwrap(),
        ];
        let mut dt = DelaunayTriangulation::builder(&vertices).build().unwrap();

        // Verify initial state
        let initial_vertices = dt.number_of_vertices();
        let initial_simplices = dt.number_of_simplices();
        assert_eq!(initial_vertices, 4);
        assert!(initial_simplices > 0);

        // Get a vertex to remove (not a corner to ensure we have remaining simplices)
        let (vertex_key, vertex_ref) = dt.vertices().next().unwrap();
        let vertex_uuid = vertex_ref.uuid();

        // Remove the vertex and all simplices containing it
        let simplices_removed = dt.delete_vertex(vertex_key).unwrap();

        // Verify the vertex was removed
        assert!(
            dt.as_triangulation()
                .tds
                .vertex_key_from_uuid(&vertex_uuid)
                .is_none(),
            "Vertex should be removed from TDS"
        );
        assert!(
            simplices_removed > 0,
            "At least one simplex should have been removed"
        );
        assert_eq!(
            dt.number_of_vertices(),
            initial_vertices - 1,
            "Vertex count should decrease by 1"
        );
        assert!(
            dt.number_of_simplices() < initial_simplices,
            "Simplex count should decrease"
        );

        // CRITICAL: Verify that no dangling neighbor references exist
        // This is the key test for the bug fix
        for (simplex_key, simplex) in dt.simplices() {
            if let Some(neighbors) = simplex.neighbors() {
                for (i, neighbor_opt) in neighbors.enumerate() {
                    if let Some(neighbor_key) = neighbor_opt {
                        assert!(
                            dt.as_triangulation()
                                .tds
                                .simplices
                                .contains_key(neighbor_key),
                            "Simplex {simplex_key:?} has dangling neighbor reference at index {i}: {neighbor_key:?}"
                        );
                    }
                }
            }
        }

        // Verify the TDS is valid (this should pass with the bug fix)
        assert!(
            dt.as_triangulation().tds.is_valid().is_ok(),
            "TDS should be valid after removing vertex"
        );
    }

    #[test]
    fn test_delete_vertex_rejects_nonexistent_vertex_key() {
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let mut dt = DelaunayTriangulation::builder(&vertices).build().unwrap();

        // Use a key that was never added
        let nonexistent_key = VertexKey::from(KeyData::from_ffi(u64::MAX));

        let initial_vertices = dt.number_of_vertices();
        let initial_simplices = dt.number_of_simplices();

        let err = dt.delete_vertex(nonexistent_key).unwrap_err();

        assert_matches!(
            err,
            DeleteVertexError::VertexNotFound { vertex_key }
                if vertex_key == nonexistent_key
        );
        assert_eq!(
            dt.number_of_vertices(),
            initial_vertices,
            "Vertex count should not change"
        );
        assert_eq!(
            dt.number_of_simplices(),
            initial_simplices,
            "Simplex count should not change"
        );
    }

    #[test]
    fn test_delete_vertex_rejects_stale_vertex_key() {
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
            vertex!([1.0, 1.0]).unwrap(),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();

        let vertex_key = dt.vertices().next().unwrap().0;

        // First removal succeeds.
        let simplices_removed = dt.delete_vertex(vertex_key).unwrap();
        assert!(simplices_removed > 0);
        let vertices_after = dt.number_of_vertices();
        let simplices_after = dt.number_of_simplices();

        let err = dt.delete_vertex(vertex_key).unwrap_err();
        assert_matches!(
            err,
            DeleteVertexError::VertexNotFound { vertex_key: stale_key }
                if stale_key == vertex_key
        );
        assert_eq!(dt.number_of_vertices(), vertices_after);
        assert_eq!(dt.number_of_simplices(), simplices_after);
    }

    #[test]
    fn test_remove_vertex_multiple_dimensions() {
        // Test remove_vertex in different dimensions

        // 2D test
        {
            let vertices_2d = [
                vertex!([0.0, 0.0]).unwrap(),
                vertex!([1.0, 0.0]).unwrap(),
                vertex!([0.0, 1.0]).unwrap(),
                vertex!([1.0, 1.0]).unwrap(),
            ];
            let mut dt_2d: DelaunayTriangulation<_, (), (), 2> =
                DelaunayTriangulation::builder(&vertices_2d)
                    .build()
                    .unwrap();
            let vertex_key = dt_2d.vertices().next().unwrap().0;
            let simplices_removed = dt_2d.delete_vertex(vertex_key).unwrap();
            assert!(simplices_removed > 0);
            assert!(dt_2d.as_triangulation().tds.is_valid().is_ok());
        }

        // 3D test
        {
            let vertices_3d = [
                vertex!([0.0, 0.0, 0.0]).unwrap(),
                vertex!([1.0, 0.0, 0.0]).unwrap(),
                vertex!([0.0, 1.0, 0.0]).unwrap(),
                vertex!([0.0, 0.0, 1.0]).unwrap(),
                vertex!([0.2, 0.2, 0.2]).unwrap(),
            ];
            let mut dt_3d: DelaunayTriangulation<_, (), (), 3> =
                DelaunayTriangulation::builder(&vertices_3d)
                    .build()
                    .unwrap();
            let vertex_key = dt_3d
                .vertices()
                .find(|(_, vertex)| {
                    let coords = vertex.point().coords();
                    coords
                        .iter()
                        .zip([0.2, 0.2, 0.2])
                        .all(|(coord, expected)| (*coord - expected).abs() < 1e-12)
                })
                .unwrap()
                .0;
            let simplices_removed = dt_3d.delete_vertex(vertex_key).unwrap();
            assert!(simplices_removed > 0);
            assert!(dt_3d.as_triangulation().tds.is_valid().is_ok());
        }

        // 4D test
        {
            let vertices_4d = [
                vertex!([0.0, 0.0, 0.0, 0.0]).unwrap(),
                vertex!([1.0, 0.0, 0.0, 0.0]).unwrap(),
                vertex!([0.0, 1.0, 0.0, 0.0]).unwrap(),
                vertex!([0.0, 0.0, 1.0, 0.0]).unwrap(),
                vertex!([0.0, 0.0, 0.0, 1.0]).unwrap(),
                vertex!([0.2, 0.2, 0.2, 0.2]).unwrap(),
            ];
            let mut dt_4d: DelaunayTriangulation<_, (), (), 4> =
                DelaunayTriangulation::builder(&vertices_4d)
                    .build()
                    .unwrap();
            let vertex_key = dt_4d
                .vertices()
                .find(|(_, vertex)| {
                    let coords = vertex.point().coords();
                    coords
                        .iter()
                        .zip([0.2, 0.2, 0.2, 0.2])
                        .all(|(coord, expected)| (*coord - expected).abs() < 1e-12)
                })
                .unwrap()
                .0;
            let simplices_removed = dt_4d.delete_vertex(vertex_key).unwrap();
            assert!(simplices_removed > 0);
            assert!(dt_4d.as_triangulation().tds.is_valid().is_ok());
        }
    }

    #[test]
    fn test_remove_vertex_no_dangling_references() {
        // Test that after removing a vertex:
        // 1. No simplices contain the deleted vertex
        // 2. No vertices have incident_simplex pointing to a removed simplex
        // 3. All remaining incident_simplex pointers are valid

        let vertices = [
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
            // Interior vertex to remove; offset from circumcenter to avoid degenerate configuration
            vertex!([0.2, 0.2, 0.2]).unwrap(),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();

        // Find the interior vertex by coordinates (order-independent)
        let interior_coords = [0.2, 0.2, 0.2];
        let (removed_vertex_key, removed_vertex_uuid) = dt
            .vertices()
            .find(|(_, v)| {
                v.point()
                    .coords()
                    .as_slice()
                    .iter()
                    .zip(&interior_coords)
                    .all(|(a, b)| (a - b).abs() < 1e-10)
            })
            .map(|(k, v)| (k, v.uuid()))
            .expect("Interior vertex should exist");

        // Remove the vertex
        let simplices_removed = dt.delete_vertex(removed_vertex_key).unwrap();
        assert!(
            simplices_removed > 0,
            "Should have removed at least one simplex"
        );

        // CRITICAL CHECK 1: No simplices should contain the deleted vertex
        for (simplex_key, simplex) in dt.simplices() {
            for &vk in simplex.vertices() {
                assert_ne!(
                    vk, removed_vertex_key,
                    "Simplex {simplex_key:?} still references deleted vertex {removed_vertex_key:?}"
                );
            }
        }

        // CRITICAL CHECK 2: The vertex should no longer exist in TDS
        assert!(
            dt.as_triangulation()
                .tds
                .vertex_key_from_uuid(&removed_vertex_uuid)
                .is_none(),
            "Deleted vertex UUID should not be in mapping"
        );
        assert!(
            dt.as_triangulation()
                .tds
                .vertex(removed_vertex_key)
                .is_none(),
            "Deleted vertex key should not exist in storage"
        );

        // CRITICAL CHECK 3: All remaining vertices should have valid incident_simplex pointers
        for (vertex_key, vertex) in dt.vertices() {
            if let Some(incident_simplex_key) = vertex.incident_simplex() {
                assert!(
                    dt.as_triangulation()
                        .tds
                        .simplices
                        .contains_key(incident_simplex_key),
                    "Vertex {vertex_key:?} has dangling incident_simplex pointer to {incident_simplex_key:?}"
                );

                // Verify the incident simplex actually contains this vertex
                let incident_simplex = dt
                    .as_triangulation()
                    .tds
                    .simplex(incident_simplex_key)
                    .unwrap();
                assert!(
                    incident_simplex.contains_vertex(vertex_key),
                    "Vertex {vertex_key:?} incident_simplex {incident_simplex_key:?} does not contain the vertex"
                );
            }
        }

        // CRITICAL CHECK 4: TDS should be valid
        assert!(
            dt.as_triangulation().tds.is_valid().is_ok(),
            "TDS should be valid after vertex removal"
        );
    }

    #[test]
    fn test_assign_neighbors_errors_on_missing_vertex_key() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();

        // Corrupt the simplex by inserting a vertex key that doesn't exist in the TDS.
        let invalid_vkey = VertexKey::from(KeyData::from_ffi(u64::MAX));
        tds.simplex_mut(simplex_key)
            .unwrap()
            .push_vertex_key(invalid_vkey);

        let err = tds.assign_neighbors().unwrap_err();
        assert_matches!(err, TdsError::VertexKeyRetrievalFailed { .. });
    }

    #[test]
    fn test_assign_neighbors_errors_on_non_manifold_facet_sharing() {
        // Three triangles sharing the same edge (v_a,v_b) is non-manifold in 2D.
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v_a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(vertex!([0.0, -1.0]).unwrap())
            .unwrap();
        let v_e = tds
            .insert_vertex_with_mapping(vertex!([2.0, 0.0]).unwrap())
            .unwrap();

        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
        )
        .unwrap();
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
        )
        .unwrap();
        tds.insert_simplex_bypassing_topology_checks_for_test(
            Simplex::try_new_with_data(vec![v_a, v_b, v_e], None).unwrap(),
        )
        .unwrap();

        let err = tds.assign_neighbors().unwrap_err();
        assert_matches!(err, TdsError::InconsistentDataStructure { .. });
    }

    #[test]
    fn test_remove_simplices_by_keys_empty_is_noop() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let gen_before = tds.generation();
        assert_eq!(tds.remove_simplices_by_keys(&[]).unwrap(), 0);
        assert_eq!(tds.generation(), gen_before);
    }

    #[test]
    fn test_remove_simplices_by_keys_rolls_back_incidence_order_exactly_on_batch_failure() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let shared_vertex = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let kept_vertex_1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let kept_vertex_2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let first_removed_vertex_1 = tds
            .insert_vertex_with_mapping(vertex!([2.0, 0.0]).unwrap())
            .unwrap();
        let first_removed_vertex_2 = tds
            .insert_vertex_with_mapping(vertex!([2.0, 1.0]).unwrap())
            .unwrap();
        let corrupted_vertex = tds
            .insert_vertex_with_mapping(vertex!([3.0, 0.0]).unwrap())
            .unwrap();
        let second_removed_vertex = tds
            .insert_vertex_with_mapping(vertex!([3.0, 1.0]).unwrap())
            .unwrap();

        let kept = tds
            .insert_simplex_bypassing_topology_checks_for_test(
                Simplex::try_new_with_data(vec![shared_vertex, kept_vertex_1, kept_vertex_2], None)
                    .unwrap(),
            )
            .unwrap();
        let first_removed = tds
            .insert_simplex_bypassing_topology_checks_for_test(
                Simplex::try_new_with_data(
                    vec![
                        shared_vertex,
                        first_removed_vertex_1,
                        first_removed_vertex_2,
                    ],
                    None,
                )
                .unwrap(),
            )
            .unwrap();
        let second_removed = tds
            .insert_simplex_bypassing_topology_checks_for_test(
                Simplex::try_new_with_data(
                    vec![shared_vertex, corrupted_vertex, second_removed_vertex],
                    None,
                )
                .unwrap(),
            )
            .unwrap();
        let before = tds
            .vertex_to_simplices_index()
            .simplex_keys(shared_vertex)
            .collect::<Vec<_>>();
        let generation_before = tds.generation();
        assert_eq!(before, vec![kept, first_removed, second_removed]);

        tds.clear_vertex_incidence_for_test(corrupted_vertex);
        let result = tds.remove_simplices_by_keys(&[first_removed, second_removed]);

        assert!(result.is_err());
        assert!(tds.contains_simplex_key(first_removed));
        assert!(tds.contains_simplex_key(second_removed));
        assert_eq!(tds.generation(), generation_before);
        assert_eq!(
            tds.vertex_to_simplices_index()
                .simplex_keys(shared_vertex)
                .collect::<Vec<_>>(),
            before
        );
    }

    #[test]
    fn test_remove_simplices_by_keys_clears_neighbor_pointers() {
        // Two triangles sharing an edge.
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let d = tds
            .insert_vertex_with_mapping(vertex!([1.0, 1.0]).unwrap())
            .unwrap();

        let simplex1 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();
        let simplex2 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![b, d, c], None).unwrap())
            .unwrap();

        // Build neighbor pointers based on facet sharing.
        tds.assign_neighbors().unwrap();

        // Sanity: at least one neighbor pointer should exist before removal.
        assert!(
            tds.simplex(simplex1)
                .unwrap()
                .neighbors()
                .is_some_and(|mut n| n.any(|neighbor| neighbor.is_some()))
        );

        let gen_before = tds.generation();
        assert_eq!(tds.remove_simplices_by_keys(&[simplex2]).unwrap(), 1);
        assert_eq!(tds.generation(), gen_before + 1);

        // All remaining neighbor pointers must not reference the removed simplex.
        for (_, simplex) in tds.simplices() {
            if let Some(neighbors) = simplex.neighbors() {
                for neighbor_opt in neighbors {
                    assert_ne!(neighbor_opt, Some(simplex2));
                }
            }
        }
    }

    #[test]
    fn test_remove_simplices_by_keys_errors_before_mutation_on_stale_incidence() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let d = tds
            .insert_vertex_with_mapping(vertex!([1.0, 1.0]).unwrap())
            .unwrap();

        let stale = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();
        let requested = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![b, d, c], None).unwrap())
            .unwrap();
        let incidence_before = tds
            .vertex_to_simplices_index()
            .simplex_keys(b)
            .collect::<Vec<_>>();
        let generation_before = tds.generation();

        tds.remove_simplex_storage_only_for_test(stale);
        let err = tds.remove_simplices_by_keys(&[requested]).unwrap_err();

        assert_matches!(err.as_tds_error(), TdsError::SimplexNotFound { .. });
        assert!(tds.contains_simplex_key(requested));
        assert_eq!(tds.generation(), generation_before);
        assert_eq!(
            tds.vertex_to_simplices_index()
                .simplex_keys(b)
                .collect::<Vec<_>>(),
            incidence_before
        );
    }

    #[test]
    fn test_remove_simplices_by_keys_repairs_incident_simplices_for_affected_vertices() {
        // Two triangles sharing an edge (B,C). Remove one and ensure incident_simplex pointers are
        // updated without requiring a full assign_incident_simplices() rebuild.
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let d = tds
            .insert_vertex_with_mapping(vertex!([1.0, 1.0]).unwrap())
            .unwrap();

        let simplex1 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();
        let simplex2 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![b, d, c], None).unwrap())
            .unwrap();

        tds.assign_neighbors().unwrap();

        // Force deterministic incident_simplex pointers that require repair.
        tds.vertex_mut(a)
            .unwrap()
            .set_incident_simplex(Some(simplex1));
        tds.vertex_mut(b)
            .unwrap()
            .set_incident_simplex(Some(simplex1));
        tds.vertex_mut(c)
            .unwrap()
            .set_incident_simplex(Some(simplex1));
        tds.vertex_mut(d)
            .unwrap()
            .set_incident_simplex(Some(simplex2));

        assert_eq!(tds.remove_simplices_by_keys(&[simplex1]).unwrap(), 1);

        // A is now isolated (no remaining simplices contain it) => incident_simplex must be None.
        assert!(tds.vertex(a).unwrap().incident_simplex().is_none());

        // B, C, D are still in simplex2 => their incident_simplex must be valid and contain them.
        for vk in [b, c, d] {
            let incident = tds
                .vertex(vk)
                .unwrap()
                .incident_simplex()
                .expect("vertex should have an incident simplex after repair");
            assert!(tds.simplices.contains_key(incident));
            let simplex = tds.simplex(incident).unwrap();
            assert!(simplex.contains_vertex(vk));
        }

        // With remaining simplices, isolated vertices are allowed at the TDS structural level.
        assert!(tds.is_valid().is_ok());

        // Neighbor pointers in the surviving simplex must not reference the removed simplex.
        let simplex2_ref = tds.simplex(simplex2).unwrap();
        if let Some(mut neighbors) = simplex2_ref.neighbors() {
            assert!(neighbors.all(|n| n != Some(simplex1)));
        }
    }

    #[test]
    fn test_tds_remove_vertex_returns_zero_when_vertex_not_in_mapping() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let missing_key = VertexKey::from(KeyData::from_ffi(u64::MAX));
        assert_eq!(tds.remove_vertex(missing_key).unwrap(), 0);
    }

    #[test]
    fn test_remove_isolated_vertex_noop_on_missing_key() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let missing = VertexKey::from(KeyData::from_ffi(u64::MAX));
        let gen_before = tds.generation();
        tds.remove_isolated_vertex(missing).unwrap();
        assert_eq!(tds.generation(), gen_before, "No mutation expected");
    }

    #[test]
    fn test_remove_isolated_vertex_uses_canonical_incidence_not_stale_hint() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let vk = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();

        // Manually set a stale incident-simplex hint. Canonical isolation is
        // defined by the maintained vertex→simplices index, not this optional hint.
        let fake_simplex_key = SimplexKey::from(KeyData::from_ffi(1));
        tds.vertex_mut(vk)
            .unwrap()
            .set_incident_simplex(Some(fake_simplex_key));

        let gen_before = tds.generation();
        tds.remove_isolated_vertex(vk).unwrap();
        assert_eq!(tds.generation(), gen_before + 1);
        assert!(tds.vertex(vk).is_none());
        assert!(!tds.vertex_to_simplices_index().contains_vertex(vk));
    }

    #[test]
    fn test_remove_isolated_vertex_removes_truly_isolated() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let vk = tds
            .insert_vertex_with_mapping(vertex!([1.0, 2.0]).unwrap())
            .unwrap();
        let uuid = tds.vertex(vk).unwrap().uuid();

        // No incident simplex set → truly isolated.
        assert!(tds.vertex(vk).unwrap().incident_simplex().is_none());

        let gen_before = tds.generation();
        tds.remove_isolated_vertex(vk).unwrap();
        assert!(tds.generation() > gen_before);
        assert!(tds.vertex(vk).is_none(), "Vertex should be removed");
        assert!(
            tds.vertex_key_from_uuid(&uuid).is_none(),
            "UUID mapping should be removed"
        );
    }

    #[test]
    fn test_tds_remove_vertex_repairs_neighbors_and_incident_simplices_incrementally() {
        // Two triangles sharing an edge (east,north). Remove the origin (only in simplex1) and ensure:
        // - simplex1 is removed
        // - neighbor back-references in simplex2 are cleared
        // - incident_simplex pointers remain valid for remaining vertices without a full rebuild
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let origin_key = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let east_key = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let north_key = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let diagonal_key = tds
            .insert_vertex_with_mapping(vertex!([1.0, 1.0]).unwrap())
            .unwrap();

        let simplex1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![origin_key, east_key, north_key], None).unwrap(),
            )
            .unwrap();
        let simplex2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![east_key, diagonal_key, north_key], None).unwrap(),
            )
            .unwrap();

        tds.assign_neighbors().unwrap();

        // Seed incident_simplex pointers:
        // - ORIGIN points at simplex1 so star discovery can use the neighbor-walk fast path.
        // - EAST/NORTH point at simplex1 so removal must repair them to simplex2.
        // - DIAGONAL points at simplex2 and should remain valid.
        tds.vertex_mut(origin_key)
            .unwrap()
            .set_incident_simplex(Some(simplex1));
        tds.vertex_mut(east_key)
            .unwrap()
            .set_incident_simplex(Some(simplex1));
        tds.vertex_mut(north_key)
            .unwrap()
            .set_incident_simplex(Some(simplex1));
        tds.vertex_mut(diagonal_key)
            .unwrap()
            .set_incident_simplex(Some(simplex2));

        let origin_uuid = tds.vertex(origin_key).unwrap().uuid();
        let removed = tds.remove_vertex(origin_key).unwrap();
        assert_eq!(removed, 1);

        // The removed vertex should be gone.
        assert!(tds.vertex_key_from_uuid(&origin_uuid).is_none());
        assert!(tds.vertex(origin_key).is_none());

        // simplex2 should remain and must not reference simplex1 as a neighbor.
        assert!(tds.simplices.contains_key(simplex2));
        let simplex2_ref = tds.simplex(simplex2).unwrap();
        if let Some(mut neighbors) = simplex2_ref.neighbors() {
            assert!(neighbors.all(|n| n != Some(simplex1)));
        }

        // Remaining vertices must have valid incident_simplex pointers (if present).
        for vertex_key in [east_key, north_key, diagonal_key] {
            let v = tds.vertex(vertex_key).unwrap();
            let Some(incident) = v.incident_simplex() else {
                panic!("vertex {vertex_key:?} should have an incident simplex after removal");
            };
            assert!(tds.simplices.contains_key(incident));
            assert!(tds.simplex(incident).unwrap().contains_vertex(vertex_key));
        }
    }

    #[test]
    fn test_find_neighbors_by_key_returns_none_buffer_for_missing_simplex() {
        let tds: Tds<(), (), 2> = Tds::empty();
        let missing = SimplexKey::from(KeyData::from_ffi(u64::MAX));
        let neighbors = tds.find_neighbors_by_key(missing);
        assert_eq!(neighbors.len(), 3);
        assert!(neighbors.iter().all(Option::is_none));
    }

    #[test]
    fn test_set_neighbors_by_key_rejects_non_neighbor_and_wrong_slot() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(vertex!([1.0, 1.0]).unwrap())
            .unwrap();
        let v_e = tds
            .insert_vertex_with_mapping(vertex!([2.0, 2.0]).unwrap())
            .unwrap();

        let simplex1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();

        // A simplex that shares only one vertex with simplex1 => not a neighbor in 2D.
        let simplex_far = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_d, v_e], None).unwrap(),
            )
            .unwrap();
        let err = tds
            .set_neighbors_by_key(simplex1, &[Some(simplex_far), None, None])
            .unwrap_err()
            .into_inner();
        assert_matches!(err, TdsError::InvalidNeighbors { .. });

        // A true facet-neighbor (shares {v_a,v_b}) placed at the wrong facet index.
        let simplex2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_b, v_a, v_d], None).unwrap(),
            )
            .unwrap();
        let err = tds
            .set_neighbors_by_key(simplex1, &[Some(simplex2), None, None])
            .unwrap_err()
            .into_inner();
        assert_matches!(err, TdsError::InvalidNeighbors { .. });
    }

    #[test]
    fn test_set_neighbors_by_key_updates_reciprocal_back_reference() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(vertex!([1.0, 1.0]).unwrap())
            .unwrap();

        let simplex1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_b, v_a, v_d], None).unwrap(),
            )
            .unwrap();

        tds.set_neighbors_by_key(simplex1, &[None, None, Some(simplex2)])
            .unwrap();

        assert_eq!(
            tds.simplex(simplex1).unwrap().neighbor_key(2).flatten(),
            Some(simplex2)
        );
        assert_eq!(
            tds.simplex(simplex2).unwrap().neighbor_key(2).flatten(),
            Some(simplex1)
        );
        tds.is_valid().unwrap();
    }

    #[test]
    fn test_set_neighbors_by_key_rejects_unpaired_interior_facet() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(vertex!([1.0, 1.0]).unwrap())
            .unwrap();

        let simplex1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v_b, v_a, v_d], None).unwrap(),
        )
        .unwrap();
        tds.assign_neighbors().unwrap();

        let err = tds
            .set_neighbors_by_key(simplex1, &[None, None, None])
            .unwrap_err()
            .into_inner();
        assert_matches!(err, TdsError::InvalidNeighbors { .. });
        assert!(tds.is_valid().is_ok());
    }

    // =============================================================================
    // NEIGHBOR VALIDATION HELPER TESTS
    // =============================================================================

    macro_rules! test_normalize_repairs_incoherent_adjacent_pair {
        ($name:ident, $dim:literal) => {
            #[test]
            fn $name() {
                let mut tds: Tds<(), (), $dim> = Tds::empty();

                let mut vertex_keys = Vec::with_capacity($dim + 2);
                let mut seed = 1.0_f64;
                for idx in 0..($dim + 2) {
                    let mut coords = [0.0_f64; $dim];
                    if idx < $dim {
                        coords[idx] = 1.0;
                    } else {
                        for coord in &mut coords {
                            *coord = seed;
                            seed += 1.0;
                        }
                    }
                    vertex_keys.push(
                        tds.insert_vertex_with_mapping(vertex!(coords).unwrap())
                            .unwrap(),
                    );
                }

                // Construct two adjacent simplices that share a facet but induce the same shared-facet
                // ordering, making orientation incoherent before normalization:
                // simplex1 = [v0..vD], simplex2 = [v0..v(D-1), v(D+1)].
                let simplex1_vertices: Vec<_> =
                    vertex_keys.iter().take($dim + 1).copied().collect();
                let mut simplex2_vertices: Vec<_> =
                    vertex_keys.iter().take($dim).copied().collect();
                simplex2_vertices.push(vertex_keys[$dim + 1]);

                let simplex1: Simplex<(), $dim> =
                    Simplex::try_new_with_data(simplex1_vertices, None).unwrap();
                let simplex2: Simplex<(), $dim> =
                    Simplex::try_new_with_data(simplex2_vertices, None).unwrap();

                tds.insert_simplex_with_mapping(simplex1).unwrap();
                tds.insert_simplex_with_mapping(simplex2).unwrap();
                tds.assign_neighbors().unwrap();

                assert!(!tds.is_coherently_oriented());

                tds.normalize_coherent_orientation().unwrap();
                assert!(tds.is_coherently_oriented());
            }
        };
    }

    test_normalize_repairs_incoherent_adjacent_pair!(
        test_normalize_repairs_incoherent_adjacent_pair_2d,
        2
    );
    test_normalize_repairs_incoherent_adjacent_pair!(
        test_normalize_repairs_incoherent_adjacent_pair_3d,
        3
    );
    test_normalize_repairs_incoherent_adjacent_pair!(
        test_normalize_repairs_incoherent_adjacent_pair_4d,
        4
    );
    test_normalize_repairs_incoherent_adjacent_pair!(
        test_normalize_repairs_incoherent_adjacent_pair_5d,
        5
    );

    #[test]
    fn test_assign_incident_simplices_clears_incident_simplex_when_no_simplices() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let vkey = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();

        // Corrupt incident_simplex and ensure it gets cleared.
        tds.vertex_mut(vkey)
            .unwrap()
            .set_incident_simplex(Some(SimplexKey::from(KeyData::from_ffi(u64::MAX))));
        assert!(tds.vertex(vkey).unwrap().incident_simplex().is_some());

        tds.assign_incident_simplices().unwrap();
        assert!(tds.vertex(vkey).unwrap().incident_simplex().is_none());
    }

    // =========================================================================
    // COHERENT ORIENTATION NORMALIZATION & GENERATION COUNTER
    // =========================================================================

    #[test]
    fn test_normalize_coherent_orientation_produces_coherent_result() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 1.0]).unwrap())
            .unwrap();

        // Two triangles sharing edge v1-v2, with deliberately inconsistent vertex order
        let c0 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        let c1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v1, v2, v3], None).unwrap(),
            )
            .unwrap();

        // Assign neighbors: shared facet is edge v1-v2.
        // c0[v0,v1,v2]: facet opposite v0 (index 0) = edge [v1,v2] → neighbor c1
        // c1[v1,v2,v3]: facet opposite v3 (index 2) = edge [v1,v2] → neighbor c0
        tds.simplex_mut(c0)
            .unwrap()
            .set_neighbors_from_keys([Some(c1), None, None])
            .unwrap();
        tds.simplex_mut(c1)
            .unwrap()
            .set_neighbors_from_keys([None, None, Some(c0)])
            .unwrap();

        tds.normalize_coherent_orientation().unwrap();
        assert!(tds.is_coherently_oriented());
    }
    #[test]
    fn test_remove_duplicate_simplices_removes_exact_duplicates() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();

        // Insert the same simplex twice
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();
        tds.insert_simplex_bypassing_topology_checks_for_test(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();
        assert_eq!(tds.number_of_simplices(), 2);
        let generation_before = tds.generation();

        let removed = tds.remove_duplicate_simplices().unwrap();
        assert_eq!(removed, 1);
        assert_eq!(tds.number_of_simplices(), 1);
        assert!(tds.generation() > generation_before);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_remove_duplicate_simplices_noop_when_no_duplicates() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::builder(&verts).build().unwrap();
        let mut tds = dt.tds().clone();
        let generation_before = tds.generation();

        let removed = tds.remove_duplicate_simplices().unwrap();
        assert_eq!(removed, 0);
        assert_eq!(tds.generation(), generation_before);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_remove_duplicate_simplices_rolls_back_when_rebuild_fails() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, -1.0]).unwrap())
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(vertex!([0.5, -0.5]).unwrap())
            .unwrap();

        // Three distinct triangles share edge v0-v1, so global neighbor
        // assignment will reject the complex after duplicate removal starts.
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();
        tds.insert_simplex_bypassing_topology_checks_for_test(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();
        tds.insert_simplex_bypassing_topology_checks_for_test(
            Simplex::try_new_with_data(vec![v0, v1, v3], None).unwrap(),
        )
        .unwrap();
        tds.insert_simplex_bypassing_topology_checks_for_test(
            Simplex::try_new_with_data(vec![v0, v1, v4], None).unwrap(),
        )
        .unwrap();
        let before = tds.clone();
        let generation_before = tds.generation();

        let error = tds.remove_duplicate_simplices().unwrap_err();

        assert_matches!(
            error.into_inner(),
            TdsError::InconsistentDataStructure { .. }
        );
        assert_eq!(tds.number_of_simplices(), 4);
        assert_eq!(tds.generation(), generation_before);
        assert_eq!(tds, before);
    }

    #[test]
    fn test_clear_all_neighbors_and_rebuild() {
        // Use 5 vertices so there are multiple simplices with actual neighbor pointers
        let vertices = [
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
            vertex!([0.5, 0.5, 0.5]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let mut tds = dt.tds().clone();
        assert!(tds.number_of_simplices() > 1);

        // Multi-simplex: simplices that share facets have Some(neighbor) entries
        let has_any_neighbor = tds.simplices().any(|(_, simplex)| {
            simplex
                .neighbors()
                .is_some_and(|mut nb| nb.any(|neighbor| neighbor.is_some()))
        });
        assert!(has_any_neighbor);

        tds.clear_all_neighbors();

        // After clearing, no simplex should have neighbors
        for (_, simplex) in tds.simplices() {
            assert!(simplex.neighbors().is_none());
        }
    }

    // =========================================================================
    // INSERT SIMPLEX WITH MAPPING: ERROR PATHS
    // =========================================================================

    #[test]
    fn test_insert_simplex_with_mapping_registers_uuid_mapping() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();

        let simplex = Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap();
        let simplex_uuid = simplex.uuid();
        let ck = tds.insert_simplex_with_mapping(simplex).unwrap();

        // UUID mapping should resolve back to the same key.
        assert_eq!(tds.simplex_key_from_uuid(&simplex_uuid), Some(ck));
        assert_eq!(tds.number_of_simplices(), 1);
    }

    #[test]
    fn test_insert_simplex_with_mapping_rejects_missing_vertex() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        // Use a stale key that doesn't exist in the TDS.
        let stale = VertexKey::from(KeyData::from_ffi(0xDEAD));

        let simplex = Simplex::try_new_with_data(vec![v0, v1, stale], None).unwrap();
        let err = tds.insert_simplex_with_mapping(simplex).unwrap_err();
        assert_matches!(
            err,
            TdsConstructionError::ValidationError(TdsError::VertexNotFound { .. })
        );
    }

    #[test]
    fn test_insert_simplex_with_mapping_trusted_vertices_rejects_missing_vertex() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let stale = VertexKey::from(KeyData::from_ffi(0xDEAD));

        let simplex = Simplex::try_new_with_data(vec![v0, v1, stale], None).unwrap();
        let err = tds
            .insert_simplex_with_mapping_trusted_vertices(simplex)
            .unwrap_err();
        assert_matches!(
            err,
            TdsConstructionError::ValidationError(TdsError::VertexNotFound { .. })
        );
    }

    #[test]
    fn test_insert_simplex_with_mapping_rejects_duplicate_uuid() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();

        let simplex_a = Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap();
        let uuid_a = simplex_a.uuid();
        tds.insert_simplex_with_mapping(simplex_a).unwrap();

        // Create a second simplex with the same UUID.
        let simplex_b = Simplex::try_new_with_uuid(vec![v0, v1, v2], uuid_a, None).unwrap();
        let err = tds.insert_simplex_with_mapping(simplex_b).unwrap_err();
        assert_matches!(err, TdsConstructionError::DuplicateUuid { .. });
    }

    #[test]
    fn test_insert_simplex_rejects_candidate_periodic_offset_count_mismatch() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();

        let mut candidate = Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap();
        let candidate_uuid = candidate.uuid();
        let generation_before = tds.generation();
        candidate.periodic_vertex_offsets = Some(vec![[0_i8, 0_i8], [0_i8, 0_i8]].into());

        let err = tds.insert_simplex_with_mapping(candidate).unwrap_err();

        assert_matches!(
            err,
            TdsConstructionError::ValidationError(TdsError::DimensionMismatch {
                expected: 3,
                actual: 2,
                ..
            })
        );
        assert_eq!(tds.number_of_simplices(), 0);
        assert_eq!(tds.generation(), generation_before);
        assert_eq!(tds.simplex_key_from_uuid(&candidate_uuid), None);
    }

    #[test]
    fn test_insert_simplex_propagates_existing_periodic_facet_key_error() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 1.0]).unwrap())
            .unwrap();

        let existing = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        tds.simplex_mut(existing)
            .unwrap()
            .set_periodic_vertex_offsets(vec![[-128_i8, 0_i8], [127_i8, 0_i8], [0_i8, 0_i8]])
            .unwrap();

        let candidate = Simplex::try_new_with_data(vec![v0, v1, v3], None).unwrap();
        let candidate_uuid = candidate.uuid();
        let generation_before = tds.generation();

        let err = tds.insert_simplex_with_mapping(candidate).unwrap_err();

        assert_matches!(
            err,
            TdsConstructionError::ValidationError(TdsError::InconsistentDataStructure {
                message
            }) if message.contains("Failed to derive periodic facet key")
        );
        assert_eq!(tds.number_of_simplices(), 1);
        assert_eq!(tds.generation(), generation_before);
        assert_eq!(tds.simplex_key_from_uuid(&candidate_uuid), None);
    }

    #[test]
    fn test_insert_simplex_with_mapping_rejects_duplicate_simplex_without_mutation() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();

        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();
        let candidate = Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap();
        let candidate_uuid = candidate.uuid();
        let generation_before = tds.generation();

        let err = tds.insert_simplex_with_mapping(candidate).unwrap_err();

        assert_matches!(
            err,
            TdsConstructionError::ValidationError(TdsError::DuplicateSimplices { .. })
        );
        assert_eq!(tds.number_of_simplices(), 1);
        assert_eq!(tds.generation(), generation_before);
        assert_eq!(tds.simplex_key_from_uuid(&candidate_uuid), None);
    }

    #[test]
    fn test_insert_simplex_with_mapping_rejects_third_incident_facet_without_mutation() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, -1.0]).unwrap())
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 1.0]).unwrap())
            .unwrap();

        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v3], None).unwrap(),
        )
        .unwrap();
        let candidate = Simplex::try_new_with_data(vec![v0, v1, v4], None).unwrap();
        let candidate_uuid = candidate.uuid();
        let generation_before = tds.generation();

        let err = tds.insert_simplex_with_mapping(candidate).unwrap_err();

        match err {
            TdsConstructionError::ValidationError(TdsError::FacetSharingViolation {
                existing_incident_count,
                attempted_incident_count,
                max_incident_count,
                candidate_simplex_uuid,
                candidate_facet_index,
                ..
            }) => {
                assert_eq!(existing_incident_count, 2);
                assert_eq!(attempted_incident_count, 3);
                assert_eq!(max_incident_count, 2);
                assert_eq!(candidate_simplex_uuid, candidate_uuid);
                assert_eq!(candidate_facet_index, 2);
            }
            other => panic!("expected structured facet-sharing violation, got {other:?}"),
        }
        assert_eq!(tds.number_of_simplices(), 2);
        assert_eq!(tds.generation(), generation_before);
        assert_eq!(tds.simplex_key_from_uuid(&candidate_uuid), None);
    }

    /// Verifies removed simplices are absent from subsequent insertion preflight scans.
    #[test]
    fn test_removed_simplices_do_not_block_future_simplex_insertions() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, -1.0]).unwrap())
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 1.0]).unwrap())
            .unwrap();

        let removed = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        let second = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v3], None).unwrap(),
            )
            .unwrap();

        assert_eq!(tds.remove_simplices_by_keys(&[removed]).unwrap(), 1);

        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .expect("removed duplicate simplex should not block reinsertion");
        assert_eq!(tds.remove_simplices_by_keys(&[second]).unwrap(), 1);
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v4], None).unwrap(),
        )
        .expect("removed incident simplex should not block later facet sharing");
    }

    // =========================================================================
    // REMOVE SIMPLICES BY KEYS: BATCH REPAIR
    // =========================================================================

    #[test]
    fn test_remove_simplices_by_keys_batch_repairs_incidence_and_neighbors() {
        let vertices = [
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
            vertex!([0.5, 0.5, 0.5]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let mut tds = dt.tds().clone();
        assert!(tds.number_of_simplices() > 1);

        // Remove the first simplex.
        let first_ck = tds.simplex_keys().next().unwrap();
        let removed = tds.remove_simplices_by_keys(&[first_ck]).unwrap();
        assert_eq!(removed, 1);

        // All remaining vertex incident_simplex pointers should be valid.
        for (vk, v) in tds.vertices() {
            if let Some(ic) = v.incident_simplex() {
                assert!(
                    tds.contains_simplex(ic),
                    "Vertex {vk:?} has dangling incident_simplex after batch removal"
                );
            }
        }

        // No surviving simplex should have a neighbor pointer to the removed simplex.
        for (_, simplex) in tds.simplices() {
            if let Some(neighbors) = simplex.neighbors() {
                for nk in neighbors.flatten() {
                    assert_ne!(nk, first_ck, "Dangling neighbor pointer to removed simplex");
                }
            }
        }
    }

    // =========================================================================
    // ASSIGN INCIDENT SIMPLICES: ERROR ON DANGLING VERTEX KEY
    // =========================================================================

    #[test]
    fn test_assign_incident_simplices_errors_on_dangling_vertex_key() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();

        let _ck = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();

        // Remove v2 from the vertex storage, leaving the simplex with a dangling reference.
        tds.vertices.remove(v2);
        tds.uuid_to_vertex_key.retain(|_, &mut vk| vk != v2);

        let err = tds.assign_incident_simplices().unwrap_err();
        assert_matches!(err.as_tds_error(), TdsError::VertexNotFound { .. });
    }

    // =========================================================================
    // NORMALIZE COHERENT ORIENTATION: SINGLE SIMPLEX (NO NEIGHBORS)
    // =========================================================================

    #[test]
    fn test_normalize_coherent_orientation_handles_single_simplex() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::builder(&verts).build().unwrap();
        let mut tds = dt.tds().clone();
        assert_eq!(tds.number_of_simplices(), 1);

        // Single simplex with no neighbors: should succeed without flipping.
        assert!(tds.normalize_coherent_orientation().is_ok());
    }

    #[test]
    fn test_normalize_coherent_orientation_multi_simplex() {
        let vertices = [
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
            vertex!([0.5, 0.5, 0.5]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let mut tds = dt.tds().clone();
        assert!(tds.number_of_simplices() > 1);

        // Should succeed for a valid multi-simplex triangulation.
        assert!(tds.normalize_coherent_orientation().is_ok());
    }

    // =========================================================================
    // REMOVE SIMPLICES BY KEYS
    // =========================================================================

    #[test]
    fn test_remove_simplices_by_keys_returns_zero_for_missing() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let stale = SimplexKey::from(KeyData::from_ffi(0xDEAD));
        assert_eq!(tds.remove_simplices_by_keys(&[stale]).unwrap(), 0);
    }

    #[test]
    fn test_remove_simplices_by_keys_deduplicates_request_keys() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let simplex = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();

        assert_eq!(
            tds.remove_simplices_by_keys(&[simplex, simplex]).unwrap(),
            1
        );
        assert!(!tds.contains_simplex_key(simplex));
        assert_eq!(tds.number_of_simplices(), 0);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_remove_simplices_by_keys_repairs_local_topology() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 1.0]).unwrap())
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();

        let removed_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        let surviving_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v2, v3], None).unwrap(),
            )
            .unwrap();
        tds.construction_state = TriangulationConstructionState::Constructed;
        tds.assign_neighbors().unwrap();
        tds.assign_incident_simplices().unwrap();

        assert!(
            tds.simplices
                .get(surviving_key)
                .unwrap()
                .neighbors()
                .unwrap()
                .any(|neighbor| neighbor == Some(removed_key))
        );

        let removed_uuid = tds.simplices.get(removed_key).unwrap().uuid();
        let removed_vertices = tds.simplices.get(removed_key).unwrap().vertices().to_vec();

        assert_eq!(tds.remove_simplices_by_keys(&[removed_key]).unwrap(), 1);

        assert_eq!(removed_vertices, vec![v0, v1, v2]);
        assert!(!tds.simplices.contains_key(removed_key));
        assert!(tds.simplices.contains_key(surviving_key));
        assert!(!tds.uuid_to_simplex_key.contains_key(&removed_uuid));
        if let Some(mut neighbors) = tds.simplices.get(surviving_key).unwrap().neighbors() {
            assert!(neighbors.all(|neighbor| neighbor != Some(removed_key)));
        }
        assert_ne!(
            tds.vertices.get(v0).unwrap().incident_simplex(),
            Some(removed_key)
        );
        assert_ne!(
            tds.vertices.get(v1).unwrap().incident_simplex(),
            Some(removed_key)
        );
        assert_ne!(
            tds.vertices.get(v2).unwrap().incident_simplex(),
            Some(removed_key)
        );
        assert_eq!(
            tds.vertices.get(v0).unwrap().incident_simplex(),
            Some(surviving_key)
        );
        assert_eq!(
            tds.vertices.get(v2).unwrap().incident_simplex(),
            Some(surviving_key)
        );
    }

    // =========================================================================
    // SET VERTEX DATA / SET SIMPLEX DATA
    // =========================================================================

    #[test]
    fn test_set_vertex_data_replaces_existing() {
        let vertices: [Vertex<i32, 2>; 3] = [
            vertex!([0.0, 0.0]; data = 10i32).unwrap(),
            vertex!([1.0, 0.0]; data = 20).unwrap(),
            vertex!([0.0, 1.0]; data = 30).unwrap(),
        ];
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .build()
            .unwrap();
        let mut tds = dt.tds().clone();
        let key = tds.vertex_keys().next().unwrap();

        let prev = tds.set_vertex_data(key, Some(99)).unwrap();
        assert!(prev.is_some()); // had data before
        assert_eq!(tds.vertex(key).unwrap().data, Some(99));
    }

    #[test]
    fn test_set_vertex_data_on_no_data_vertex() {
        // Vertices without data have U = (), so set_vertex_data sets ().
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let mut tds = dt.tds().clone();
        let key = tds.vertex_keys().next().unwrap();

        let prev = tds.set_vertex_data(key, Some(())).unwrap();
        // Vertices constructed without explicit data have data = None
        assert_eq!(prev, None);
        assert_eq!(tds.vertex(key).unwrap().data, Some(()));
    }

    #[test]
    fn test_set_vertex_data_invalid_key_returns_error() {
        let mut tds: Tds<i32, (), 2> = Tds::empty();
        let stale = VertexKey::from(KeyData::from_ffi(0xDEAD));
        let err = tds.set_vertex_data(stale, Some(1)).unwrap_err();
        assert_matches!(err.as_tds_error(), TdsError::VertexNotFound { .. });
    }

    #[test]
    fn test_set_simplex_data_on_empty_simplex() {
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .simplex_data_type::<i32>()
            .build()
            .unwrap();
        let mut tds = dt.tds().clone();
        let key = tds.simplex_keys().next().unwrap();

        let prev = tds.set_simplex_data(key, Some(42)).unwrap();
        assert_eq!(prev, None); // key found, no previous data
        assert_eq!(tds.simplex(key).unwrap().data, Some(42));
    }

    #[test]
    fn test_set_simplex_data_replaces_existing() {
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .simplex_data_type::<i32>()
            .build()
            .unwrap();
        let mut tds = dt.tds().clone();
        let key = tds.simplex_keys().next().unwrap();

        tds.set_simplex_data(key, Some(1)).unwrap();
        let prev = tds.set_simplex_data(key, Some(2)).unwrap();
        assert_eq!(prev, Some(1));
        assert_eq!(tds.simplex(key).unwrap().data, Some(2));
    }

    #[test]
    fn test_set_simplex_data_invalid_key_returns_error() {
        let mut tds: Tds<(), i32, 2> = Tds::empty();
        let stale = SimplexKey::from(KeyData::from_ffi(0xDEAD));
        let err = tds.set_simplex_data(stale, Some(1)).unwrap_err();
        assert_matches!(err.as_tds_error(), TdsError::SimplexNotFound { .. });
    }

    #[test]
    fn test_set_vertex_data_preserves_triangulation_validity() {
        let vertices: [Vertex<i32, 2>; 3] = [
            vertex!([0.0, 0.0]; data = 1i32).unwrap(),
            vertex!([1.0, 0.0]; data = 2).unwrap(),
            vertex!([0.0, 1.0]; data = 3).unwrap(),
        ];
        let mut dt = DelaunayTriangulationBuilder::new(&vertices)
            .build()
            .unwrap();

        // Mutate every vertex's data through the DT wrapper.
        let keys: Vec<_> = dt.vertices().map(|(k, _)| k).collect();
        for (key, i) in keys.iter().zip(0i32..) {
            dt.set_vertex_data(*key, Some(i * 100)).unwrap();
        }

        // Triangulation must remain fully valid.
        assert!(dt.validate().is_ok());

        // Verify all data was updated.
        for (key, i) in keys.iter().zip(0i32..) {
            let v = dt.tds().vertex(*key).unwrap();
            assert_eq!(v.data, Some(i * 100));
        }
    }

    #[test]
    fn test_set_simplex_data_preserves_triangulation_validity() {
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.5, 1.0]).unwrap(),
            vertex!([1.5, 0.5]).unwrap(),
        ];
        let mut dt = DelaunayTriangulationBuilder::new(&vertices)
            .simplex_data_type::<i32>()
            .build()
            .unwrap();
        assert!(dt.number_of_simplices() > 1);

        // Mutate every simplex's data through the DT wrapper.
        let keys: Vec<_> = dt.simplices().map(|(k, _)| k).collect();
        for (key, i) in keys.iter().zip(0i32..) {
            dt.set_simplex_data(*key, Some(i)).unwrap();
        }

        // Triangulation must remain fully valid.
        assert!(dt.validate().is_ok());

        // Verify all data was updated.
        for (key, i) in keys.iter().zip(0i32..) {
            let c = dt.tds().simplex(*key).unwrap();
            assert_eq!(c.data, Some(i));
        }
    }

    #[test]
    fn test_set_vertex_data_via_delaunay_wrapper() {
        let vertices: [Vertex<i32, 2>; 3] = [
            vertex!([0.0, 0.0]; data = 10i32).unwrap(),
            vertex!([1.0, 0.0]; data = 20).unwrap(),
            vertex!([0.0, 1.0]; data = 30).unwrap(),
        ];
        let mut dt = DelaunayTriangulationBuilder::new(&vertices)
            .build()
            .unwrap();
        let key = dt.vertices().next().unwrap().0;

        // Set via Delaunay wrapper
        let prev = dt.set_vertex_data(key, Some(99)).unwrap();
        assert!(prev.is_some());
        assert_eq!(dt.tds().vertex(key).unwrap().data, Some(99));

        // Clear via Delaunay wrapper
        let prev = dt.set_vertex_data(key, None).unwrap();
        assert_eq!(prev, Some(99));
        assert_eq!(dt.tds().vertex(key).unwrap().data, None);
    }

    #[test]
    fn test_set_vertex_data_via_delaunay_wrapper_invalid_key_returns_error() {
        let vertices: [Vertex<i32, 2>; 3] = [
            vertex!([0.0, 0.0]; data = 10i32).unwrap(),
            vertex!([1.0, 0.0]; data = 20).unwrap(),
            vertex!([0.0, 1.0]; data = 30).unwrap(),
        ];
        let mut dt = DelaunayTriangulationBuilder::new(&vertices)
            .build()
            .unwrap();
        let live_key = dt.vertices().next().unwrap().0;
        let stale = VertexKey::from(KeyData::from_ffi(0xFEED));

        let err = dt.set_vertex_data(stale, Some(99)).unwrap_err();

        assert_matches!(
            err.as_tds_error(),
            TdsError::VertexNotFound {
                vertex_key,
                ..
            } if *vertex_key == stale
        );
        assert_eq!(dt.tds().vertex(live_key).unwrap().data, Some(10));
    }

    #[test]
    fn test_set_simplex_data_via_delaunay_wrapper() {
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let mut dt = DelaunayTriangulationBuilder::new(&vertices)
            .simplex_data_type::<i32>()
            .build()
            .unwrap();
        let key = dt.simplices().next().unwrap().0;

        // Set via Delaunay wrapper
        let prev = dt.set_simplex_data(key, Some(42)).unwrap();
        assert_eq!(prev, None);
        assert_eq!(dt.tds().simplex(key).unwrap().data, Some(42));

        // Clear via Delaunay wrapper
        let prev = dt.set_simplex_data(key, None).unwrap();
        assert_eq!(prev, Some(42));
        assert_eq!(dt.tds().simplex(key).unwrap().data, None);
    }

    #[test]
    fn test_set_simplex_data_via_delaunay_wrapper_invalid_key_returns_error() {
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let mut dt = DelaunayTriangulationBuilder::new(&vertices)
            .simplex_data_type::<i32>()
            .build()
            .unwrap();
        let live_key = dt.simplices().next().unwrap().0;
        let stale = SimplexKey::from(KeyData::from_ffi(0xFEED));

        let err = dt.set_simplex_data(stale, Some(42)).unwrap_err();

        assert_matches!(
            err.as_tds_error(),
            TdsError::SimplexNotFound {
                simplex_key,
                ..
            } if *simplex_key == stale
        );
        assert_eq!(dt.tds().simplex(live_key).unwrap().data, None);
    }

    #[test]
    fn test_set_data_via_dt_does_not_invalidate_locate_hint() {
        let vertices: [Vertex<i32, 2>; 3] = [
            vertex!([0.0, 0.0]; data = 0i32).unwrap(),
            vertex!([1.0, 0.0]; data = 0).unwrap(),
            vertex!([0.0, 1.0]; data = 0).unwrap(),
        ];
        let mut dt = DelaunayTriangulationBuilder::new(&vertices)
            .build()
            .unwrap();

        // Insert a new vertex so the locate hint is populated.
        let extra = vertex!([0.25, 0.25]; data = 0i32).unwrap();
        dt.insert_vertex(extra).unwrap();

        // Data mutation should NOT clear the insertion hint.
        let key = dt.vertices().next().unwrap().0;
        let prev = dt.set_vertex_data(key, Some(999)).unwrap();
        assert!(prev.is_some(), "set_vertex_data should find the key");
        assert_eq!(
            dt.tds().vertex(key).unwrap().data,
            Some(999),
            "stored value should reflect the mutation"
        );

        // A subsequent insert should still succeed (hint not invalidated).
        let another = vertex!([0.75, 0.1]; data = 0i32).unwrap();
        assert!(dt.insert_vertex(another).is_ok());
        assert!(dt.validate().is_ok());
    }
}
