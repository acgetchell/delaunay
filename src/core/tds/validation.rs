//! TDS structural validation and diagnostic reporting.

#![forbid(unsafe_code)]

use super::errors::{
    EntityKind, InvariantKind, InvariantViolation, NeighborValidationError,
    SharedFacetMismatchSide, TdsError, TriangulationValidationReport,
};
use super::storage::{SimplexUuidSortKey, Tds};
use super::{SimplexKey, VertexKey};
use crate::core::collections::{
    FacetToSimplicesMap, FastHashMap, MAX_PRACTICAL_DIMENSION_SIZE, NeighborBuffer,
    SimplexVerticesMap, SmallBuffer, VertexKeySet, fast_hash_map_with_capacity,
};
use crate::core::facet::{FacetHandle, FacetToSimplicesIndex};
use crate::core::simplex::{NeighborSlot, Simplex};
use crate::core::util::{deduplication::coords_equal_exact, usize_to_u8};
use slotmap::Key;

impl<U, V, const D: usize> Tds<U, V, D> {
    /// Builds an owner-bound facet-to-simplices index with strict error handling.
    ///
    /// This method returns an error if any simplex has missing vertex keys or if
    /// any facet has multiplicity other than 1 (one-sided) or 2 (two-sided),
    /// ensuring complete and validated facet topology information. This is the
    /// public method for repeated facet-incidence and boundary queries.
    ///
    /// # Returns
    ///
    /// A `Result` containing:
    /// - `Ok(FacetToSimplicesIndex)`: A complete index of facet keys to simplices,
    ///   lifetime-bound to this TDS
    /// - `Err(TdsError)`: If simplex vertices cannot be resolved or a facet has
    ///   invalid multiplicity
    ///
    /// # Errors
    ///
    /// Returns [`TdsError`] if the index cannot be built:
    /// - [`VertexNotFound`](TdsError::VertexNotFound) / [`SimplexNotFound`](TdsError::SimplexNotFound) — a simplex cannot resolve its vertex keys.
    /// - [`IndexOutOfBounds`](TdsError::IndexOutOfBounds) — a facet index exceeds the `u8` range.
    /// - [`DimensionMismatch`](TdsError::DimensionMismatch) — periodic offset count does not match vertex count.
    /// - [`InconsistentDataStructure`](TdsError::InconsistentDataStructure) — periodic facet key derivation fails.
    /// - [`FacetError`](TdsError::FacetError) — a facet is incident to a number of simplices other than 1 or 2.
    ///
    /// # Performance
    ///
    /// O(N×F) time complexity where N is the number of simplices and F is the
    /// number of facets per simplex (typically D+1 for D-dimensional simplices).
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
    /// let facet_index = dt.facet_incidence_index()?;
    /// assert!(!facet_index.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn build_facet_to_simplices_index(
        &self,
    ) -> Result<FacetToSimplicesIndex<'_, U, V, D>, TdsError> {
        self.build_facet_to_simplices_map()
            .and_then(|map| FacetToSimplicesIndex::try_from_map(self, &map).map_err(TdsError::from))
    }

    /// Builds the raw facet-to-simplices map used by validation and cache internals.
    pub(crate) fn build_facet_to_simplices_map(&self) -> Result<FacetToSimplicesMap, TdsError> {
        if D > usize::from(u8::MAX) {
            return Err(TdsError::DimensionMismatch {
                expected: usize::from(u8::MAX),
                actual: D,
                context: "facet indices must fit in u8".to_string(),
            });
        }

        let cap = self.simplices.len().saturating_mul(D.saturating_add(1));
        let mut facet_to_simplices: FacetToSimplicesMap = fast_hash_map_with_capacity(cap);

        // Iterate over all simplices and their facets
        for (simplex_id, simplex) in &self.simplices {
            // Use direct key-based method to avoid UUID→Key lookups
            // The error from simplex_vertices is already TdsError
            let vertices = self.simplex_vertices(simplex_id)?;

            for i in 0..vertices.len() {
                let facet_key =
                    Self::periodic_facet_key_from_simplex_vertices(simplex, vertices, i)?;
                let Ok(facet_index_u8) = usize_to_u8(i, vertices.len()) else {
                    return Err(TdsError::IndexOutOfBounds {
                        index: i,
                        bound: u8::MAX as usize + 1,
                        context: format!("facet index exceeds u8 range for {D}D"),
                    });
                };

                facet_to_simplices
                    .entry(facet_key)
                    .or_default()
                    .push(FacetHandle::from_validated(simplex_id, facet_index_u8));
            }
        }

        Ok(facet_to_simplices)
    }
}

impl<U, V, const D: usize> Tds<U, V, D> {
    // =========================================================================
    // VALIDATION & CONSISTENCY CHECKS
    // =========================================================================
    // Note: Structural validation is topology-only. Level-1 element validation
    // checks validated f64 coordinate storage.

    /// Validates the consistency of vertex UUID-to-key mappings.
    ///
    /// This helper function ensures that:
    /// 1. The number of entries in `vertex_uuid_to_key` matches the number of vertices
    /// 2. The number of entries in `vertex_key_to_uuid` matches the number of vertices
    /// 3. Every vertex UUID in the triangulation has a corresponding key mapping
    /// 4. Every vertex key in the triangulation has a corresponding UUID mapping
    /// 5. The mappings are bidirectional and consistent (UUID ↔ Key)
    ///
    /// # Returns
    ///
    /// `Ok(())` if all vertex mappings are consistent, otherwise a `TdsError`.
    ///
    /// This corresponds to [`InvariantKind::VertexMappings`], which is included in
    /// [`Tds::is_valid`](Self::is_valid) and [`Tds::validate`](Self::validate), and is also surfaced by
    /// [`DelaunayTriangulation::validation_report()`](crate::DelaunayTriangulation::validation_report).
    ///
    /// # Errors
    ///
    /// Returns a `TdsError::MappingInconsistency` with a descriptive message if:
    /// - The number of UUID-to-key mappings doesn't match the number of vertices
    /// - The number of key-to-UUID mappings doesn't match the number of vertices
    /// - A vertex exists without a corresponding UUID-to-key mapping
    /// - A vertex exists without a corresponding key-to-UUID mapping
    /// - The bidirectional mappings are inconsistent (UUID maps to key A, but key A maps to different UUID)
    ///
    pub(crate) fn validate_vertex_mappings(&self) -> Result<(), TdsError> {
        if self.uuid_to_vertex_key.len() != self.vertices.len() {
            return Err(TdsError::MappingInconsistency {
                entity: EntityKind::Vertex,
                message: format!(
                    "Number of mapping entries ({}) doesn't match number of vertices ({})",
                    self.uuid_to_vertex_key.len(),
                    self.vertices.len()
                ),
            });
        }

        // Check the key-to-UUID direction first (direct storage map access),
        // then only do UUID-to-key lookup verification when needed.
        for (vertex_key, vertex) in &self.vertices {
            let vertex_uuid = vertex.uuid();

            // Check key-to-UUID direction first (direct storage map access - no hash lookup)
            if self.vertex_uuid_from_key(vertex_key) != Some(vertex_uuid) {
                return Err(TdsError::MappingInconsistency {
                    entity: EntityKind::Vertex,
                    message: format!(
                        "Inconsistent or missing key-to-UUID mapping for key {vertex_key:?}"
                    ),
                });
            }

            // Now verify UUID-to-key direction (requires hash lookup but we know it should exist)
            if self.uuid_to_vertex_key.get(&vertex_uuid) != Some(&vertex_key) {
                return Err(TdsError::MappingInconsistency {
                    entity: EntityKind::Vertex,
                    message: format!(
                        "Inconsistent or missing UUID-to-key mapping for UUID {vertex_uuid:?}"
                    ),
                });
            }
        }
        Ok(())
    }

    /// Validates the consistency of simplex UUID-to-key mappings.
    ///
    /// This helper function ensures that:
    /// 1. The number of entries in `simplex_uuid_to_key` matches the number of simplices
    /// 2. The number of entries in `simplex_key_to_uuid` matches the number of simplices
    /// 3. Every simplex UUID in the triangulation has a corresponding key mapping
    /// 4. Every simplex key in the triangulation has a corresponding UUID mapping
    /// 5. The mappings are bidirectional and consistent (UUID ↔ Key)
    ///
    /// # Returns
    ///
    /// `Ok(())` if all simplex mappings are consistent, otherwise a `TdsError`.
    ///
    /// This corresponds to [`InvariantKind::SimplexMappings`], which is included in
    /// [`Tds::is_valid`](Self::is_valid) and [`Tds::validate`](Self::validate), and is also surfaced by
    /// [`DelaunayTriangulation::validation_report()`].
    ///
    /// [`DelaunayTriangulation::validation_report()`]: crate::DelaunayTriangulation::validation_report
    ///
    /// # Errors
    ///
    /// Returns a `TdsError::MappingInconsistency` with a descriptive message if:
    /// - The number of UUID-to-key mappings doesn't match the number of simplices
    /// - The number of key-to-UUID mappings doesn't match the number of simplices
    /// - A simplex exists without a corresponding UUID-to-key mapping
    /// - A simplex exists without a corresponding key-to-UUID mapping
    /// - The bidirectional mappings are inconsistent (UUID maps to key A, but key A maps to different UUID)
    ///
    pub(crate) fn validate_simplex_mappings(&self) -> Result<(), TdsError> {
        if self.uuid_to_simplex_key.len() != self.simplices.len() {
            return Err(TdsError::MappingInconsistency {
                entity: EntityKind::Simplex,
                message: format!(
                    "Number of mapping entries ({}) doesn't match number of simplices ({})",
                    self.uuid_to_simplex_key.len(),
                    self.simplices.len()
                ),
            });
        }

        // Check the key-to-UUID direction first (direct storage map access),
        // then only do UUID-to-key lookup verification when needed.
        for (simplex_key, simplex) in &self.simplices {
            let simplex_uuid = simplex.uuid();

            // Check key-to-UUID direction first (direct storage map access - no hash lookup)
            if self.simplex_uuid_from_key(simplex_key) != Some(simplex_uuid) {
                return Err(TdsError::MappingInconsistency {
                    entity: EntityKind::Simplex,
                    message: format!(
                        "Inconsistent or missing key-to-UUID mapping for key {simplex_key:?}"
                    ),
                });
            }

            // Now verify UUID-to-key direction (requires hash lookup but we know it should exist)
            if self.uuid_to_simplex_key.get(&simplex_uuid) != Some(&simplex_key) {
                return Err(TdsError::MappingInconsistency {
                    entity: EntityKind::Simplex,
                    message: format!(
                        "Inconsistent or missing UUID-to-key mapping for UUID {simplex_uuid:?}"
                    ),
                });
            }
        }
        Ok(())
    }

    /// Validates that all vertex keys referenced by simplices actually exist in the vertices `storage map`.
    ///
    /// This is a defensive check for data structure corruption. In normal operation,
    /// this should never fail, but it's useful for catching bugs during development
    /// and for comprehensive validation.
    ///
    /// This ensures that no simplex references a stale or invalid vertex key.
    ///
    /// # Returns
    ///
    /// `Ok(())` if all vertex keys in all simplices are valid, otherwise a `TdsError`.
    ///
    /// # Errors
    ///
    /// Returns `TdsError::VertexNotFound` if any simplex
    /// references a vertex key that doesn't exist in the vertices `storage map`.
    pub(crate) fn validate_simplex_vertex_keys(&self) -> Result<(), TdsError> {
        for (simplex_key, simplex) in &self.simplices {
            let simplex_uuid = simplex.uuid();
            for (vertex_idx, &vertex_key) in simplex.vertices().iter().enumerate() {
                if !self.vertices.contains_key(vertex_key) {
                    return Err(TdsError::VertexNotFound {
                        vertex_key,
                        context: format!(
                            "referenced by simplex {simplex_uuid} (key {simplex_key:?}) at position {vertex_idx}"
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    /// Validates that `Vertex::incident_simplex` pointers are non-dangling and internally consistent.
    ///
    /// Note: at the TDS structural layer (Level 2), isolated vertices (vertices not referenced by
    /// any simplex) are allowed, so `Vertex::incident_simplex` may be `None`.
    ///
    /// Level 3 topology validation (`Triangulation::is_valid_topology`) rejects isolated vertices.
    ///
    /// However, any `incident_simplex` pointer that *is* present must:
    /// - point to an existing simplex key, and
    /// - reference a simplex that actually contains the vertex.
    fn validate_vertex_incidence(&self) -> Result<(), TdsError> {
        for (vertex_key, vertex) in &self.vertices {
            let Some(incident_simplex_key) = vertex.incident_simplex() else {
                continue;
            };

            let Some(incident_simplex) = self.simplices.get(incident_simplex_key) else {
                return Err(TdsError::SimplexNotFound {
                    simplex_key: incident_simplex_key,
                    context: format!(
                        "dangling incident_simplex pointer from vertex {vertex_key:?}"
                    ),
                });
            };

            if !incident_simplex.contains_vertex(vertex_key) {
                return Err(TdsError::InconsistentDataStructure {
                    message: format!(
                        "Vertex {vertex_key:?} incident_simplex {incident_simplex_key:?} does not contain the vertex"
                    ),
                });
            }
        }

        Ok(())
    }

    /// Validates the maintained vertex→simplices incidence index.
    ///
    /// This checks the full exact incidence relation. It is distinct from
    /// [`Self::validate_vertex_incidence`], which only validates each vertex's single
    /// optional `incident_simplex` hint.
    pub(in crate::core) fn validate_vertex_to_simplices_index(&self) -> Result<(), TdsError> {
        let vertex_to_simplices = self.vertex_to_simplices.as_map();

        if vertex_to_simplices.len() != self.vertices.len() {
            return Err(TdsError::MappingInconsistency {
                entity: EntityKind::Vertex,
                message: format!(
                    "Vertex-to-simplices index has {} entries, expected one entry per {} vertices",
                    vertex_to_simplices.len(),
                    self.vertices.len()
                ),
            });
        }

        for &vertex_key in vertex_to_simplices.keys() {
            if !self.vertices.contains_key(vertex_key) {
                return Err(TdsError::VertexNotFound {
                    vertex_key,
                    context: "vertex-to-simplices index entry".to_string(),
                });
            }
        }

        let incidence_count = vertex_to_simplices
            .values()
            .map(SmallBuffer::len)
            .sum::<usize>();
        let expected_incidence_count = self.simplices.len().saturating_mul(D.saturating_add(1));
        if incidence_count != expected_incidence_count {
            return Err(TdsError::InconsistentDataStructure {
                message: format!(
                    "Vertex-to-simplices index has {incidence_count} simplex incidences, expected {expected_incidence_count}"
                ),
            });
        }

        for incident_simplices in vertex_to_simplices.values() {
            for (index, &simplex_key) in incident_simplices.iter().enumerate() {
                if incident_simplices[..index].contains(&simplex_key) {
                    return Err(TdsError::InconsistentDataStructure {
                        message: format!(
                            "Vertex-to-simplices index contains duplicate simplex {simplex_key:?}"
                        ),
                    });
                }
            }
        }

        for (simplex_key, simplex) in &self.simplices {
            for vertex_key in simplex.vertices() {
                let Some(incident_simplices) = vertex_to_simplices.get(vertex_key) else {
                    return Err(TdsError::VertexNotFound {
                        vertex_key: *vertex_key,
                        context: format!("simplex {simplex_key:?} vertex-to-simplices index entry"),
                    });
                };
                if !incident_simplices.contains(&simplex_key) {
                    return Err(TdsError::InconsistentDataStructure {
                        message: format!(
                            "Vertex-to-simplices index missing simplex {simplex_key:?} from vertex {vertex_key:?}"
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    /// Check for duplicate simplices and return an error if any are found
    ///
    /// This is useful for validation where you want to detect duplicates
    /// without automatically removing them.
    ///
    /// **Implementation Note**: This method uses `Simplex::vertex_uuids()` to get canonical
    /// vertex UUIDs for each simplex, which are then sorted and compared for duplicate detection.
    ///
    /// # Errors
    ///
    /// Returns a [`TdsError`] if simplex vertex retrieval fails
    /// or if any duplicate simplices are detected.
    ///
    /// This corresponds to [`InvariantKind::DuplicateSimplices`], which is included in
    /// [`Tds::is_valid`](Self::is_valid) and [`Tds::validate`](Self::validate), and is also surfaced by
    /// [`DelaunayTriangulation::validation_report()`].
    ///
    /// [`DelaunayTriangulation::validation_report()`]: crate::DelaunayTriangulation::validation_report
    fn validate_no_duplicate_simplices(&self) -> Result<(), TdsError> {
        // Include periodic per-vertex offsets in the duplicate key so periodic quotient simplices
        // with identical vertex sets but distinct lattice offsets are not collapsed.
        let mut unique_simplices: FastHashMap<SimplexUuidSortKey<D>, SimplexKey> =
            fast_hash_map_with_capacity(self.simplices.len());
        let mut duplicates = Vec::new();

        for (simplex_key, _simplex) in &self.simplices {
            let vertices = self.simplex_vertices(simplex_key)?;
            let vertex_uuid_offsets =
                self.build_periodic_vertex_uuid_offsets(simplex_key, vertices)?;

            if let Some(existing_simplex_key) = unique_simplices.get(&vertex_uuid_offsets) {
                duplicates.push((
                    simplex_key,
                    *existing_simplex_key,
                    vertex_uuid_offsets.clone(),
                ));
            } else {
                unique_simplices.insert(vertex_uuid_offsets, simplex_key);
            }
        }

        if !duplicates.is_empty() {
            let duplicate_descriptions: Vec<String> = duplicates
                .iter()
                .map(|(simplex1, simplex2, vertex_uuids)| {
                    format!(
                        "simplices {simplex1:?} and {simplex2:?} with vertex UUIDs {vertex_uuids:?}"
                    )
                })
                .collect();

            return Err(TdsError::DuplicateSimplices {
                message: format!(
                    "Found {} duplicate simplex(s): {}",
                    duplicates.len(),
                    duplicate_descriptions.join(", ")
                ),
            });
        }

        Ok(())
    }

    /// Validates that no simplex contains vertices with identical coordinates.
    ///
    /// This is an element-level coordinate check complementing [`Simplex::try_new()`]'s vertex-key uniqueness
    /// check. Two different vertex keys can reference geometrically identical points, producing
    /// a zero-volume simplex that is catastrophic for `SoS` orientation and Pachner moves.
    ///
    /// Uses exact `OrderedFloat`-based coordinate comparison (NaN-aware, +0.0 == -0.0).
    ///
    /// # Errors
    ///
    /// Returns [`TdsError::DuplicateCoordinatesInSimplex`] on the first simplex found
    /// containing two vertices with identical coordinates.
    fn validate_simplex_coordinate_uniqueness(&self) -> Result<(), TdsError> {
        for (_simplex_key, simplex) in &self.simplices {
            let vkeys = simplex.vertices();
            // O(D²) pairwise comparison per simplex — acceptable since D is small (≤ 6).
            for i in 0..vkeys.len() {
                let Some(vi) = self.vertex(vkeys[i]) else {
                    continue; // Missing keys are caught by validate_simplex_vertex_keys
                };
                for j in (i + 1)..vkeys.len() {
                    // Same key → same vertex; skip to avoid a misleading
                    // "duplicate coordinates" error for what is really a
                    // duplicate-key issue (caught by Simplex::new).
                    if vkeys[i] == vkeys[j] {
                        continue;
                    }
                    let Some(vj) = self.vertex(vkeys[j]) else {
                        continue;
                    };
                    if coords_equal_exact(vi.point().coords(), vj.point().coords()) {
                        return Err(TdsError::DuplicateCoordinatesInSimplex {
                            simplex_id: simplex.uuid(),
                            message: format!(
                                "vertices {:?} and {:?} (keys {:?}, {:?}) have identical coordinates {:?}",
                                vi.uuid(),
                                vj.uuid(),
                                vkeys[i],
                                vkeys[j],
                                vi.point().coords(),
                            ),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Validates that no facet is shared by more than 2 simplices
    ///
    /// This is a critical property for valid triangulations. Each facet should be
    /// shared by at most 2 simplices - boundary facets belong to 1 simplex, and internal
    /// facets should be shared by exactly 2 adjacent simplices.
    ///
    /// # Errors
    ///
    /// Returns a [`TdsError`] if building the facet map fails
    /// or if any facet is shared by more than two simplices.
    ///
    /// This corresponds to [`InvariantKind::FacetSharing`], which is included in
    /// [`Tds::is_valid`](Self::is_valid) and [`Tds::validate`](Self::validate), and is also surfaced by
    /// [`DelaunayTriangulation::validation_report()`].
    ///
    /// [`DelaunayTriangulation::validation_report()`]: crate::DelaunayTriangulation::validation_report
    pub(crate) fn validate_facet_sharing(&self) -> Result<(), TdsError> {
        // Build a map from facet keys to the simplices that contain them.
        // Use the strict version to ensure we catch any missing vertex keys.
        let facet_to_simplices = self.build_facet_to_simplices_map()?;
        self.validate_facet_sharing_with_facet_to_simplices_map(&facet_to_simplices)
    }

    /// Checks whether all adjacent simplices induce opposite orientations on shared facets.
    ///
    /// This is a combinatorial check based on simplex vertex ordering and neighbor slots.
    /// It does not evaluate geometric predicates.
    ///
    /// Returns `false` on the first detected inconsistency or data-structure error.
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
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 2> =
    ///     DelaunayTriangulationBuilder::new(&vertices).build()?;
    ///
    /// assert!(dt.is_coherently_oriented());
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn is_coherently_oriented(&self) -> bool {
        self.validate_coherent_orientation().is_ok()
    }

    /// Validates coherent combinatorial orientation for all adjacent simplex pairs.
    ///
    /// For two neighboring simplices that share a facet, this verifies the induced
    /// facet orientations are opposite (boundary-orientation convention).
    ///
    /// # Errors
    ///
    /// Returns [`TdsError`] on the first detected problem:
    /// - [`OrientationViolation`](TdsError::OrientationViolation) — adjacent simplices do not induce opposite facet orientations.
    /// - [`InvalidNeighbors`](TdsError::InvalidNeighbors) — a mirror facet cannot be derived, or a
    ///   neighbor's back-reference does not point to the originating simplex.
    /// - [`SimplexNotFound`](TdsError::SimplexNotFound) — a neighbor simplex key is missing from storage.
    /// - [`InconsistentDataStructure`](TdsError::InconsistentDataStructure) — permutation parity cannot be determined.
    /// - [`IndexOutOfBounds`](TdsError::IndexOutOfBounds) / [`DimensionMismatch`](TdsError::DimensionMismatch)
    ///   — facet-extraction helpers encounter invalid indices or periodic-offset count mismatches.
    fn validate_coherent_orientation(&self) -> Result<(), TdsError> {
        for (simplex_key, simplex) in &self.simplices {
            let Some(neighbors) = simplex.neighbor_keys() else {
                continue;
            };

            for (facet_idx, neighbor_key_opt) in neighbors.enumerate() {
                let Some(neighbor_key) = neighbor_key_opt else {
                    continue; // Boundary facet
                };

                let neighbor_simplex =
                    self.simplices
                        .get(neighbor_key)
                        .ok_or_else(|| TdsError::SimplexNotFound {
                            simplex_key: neighbor_key,
                            context: format!(
                                "neighbor of simplex {simplex_key:?} during orientation validation",
                            ),
                        })?;

                let (mirror_idx, _) = Self::orientation_mirror_facet_index(
                    simplex,
                    facet_idx,
                    neighbor_simplex,
                    "orientation validation",
                )?;
                let back_neighbor = neighbor_simplex
                    .neighbor_key(mirror_idx)
                    .flatten()
                    .ok_or_else(|| TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::BackReferenceMismatch {
                            simplex_key,
                            simplex_uuid: simplex.uuid(),
                            facet_index: facet_idx,
                            neighbor_key,
                            neighbor_uuid: neighbor_simplex.uuid(),
                            mirror_index: mirror_idx,
                            observed: None,
                            context: "orientation validation".to_string(),
                        },
                    })?;
                if back_neighbor != simplex_key {
                    return Err(TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::BackReferenceMismatch {
                            simplex_key,
                            simplex_uuid: simplex.uuid(),
                            facet_index: facet_idx,
                            neighbor_key,
                            neighbor_uuid: neighbor_simplex.uuid(),
                            mirror_index: mirror_idx,
                            observed: Some(back_neighbor),
                            context: "orientation validation".to_string(),
                        },
                    });
                }
                let simplex1_facet_vertices =
                    Self::facet_vertices_in_simplex_order(simplex, facet_idx)?;
                let simplex2_facet_vertices =
                    Self::facet_vertices_in_simplex_order(neighbor_simplex, mirror_idx)?;
                let (currently_coherent, observed_odd_permutation, expected_odd_permutation) =
                    Self::facet_permutation_parity(
                        simplex,
                        facet_idx,
                        neighbor_simplex,
                        mirror_idx,
                    )?;

                if !currently_coherent {
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

    /// Resolves the mirror facet for orientation validation, including quotient self-identifications.
    pub(crate) fn orientation_mirror_facet_index(
        simplex: &Simplex<V, D>,
        facet_idx: usize,
        neighbor_simplex: &Simplex<V, D>,
        context: &str,
    ) -> Result<(usize, bool), TdsError> {
        let uses_periodic_offsets = simplex.periodic_vertex_offsets().is_some()
            || neighbor_simplex.periodic_vertex_offsets().is_some();
        let mirror_idx = if uses_periodic_offsets && std::ptr::eq(simplex, neighbor_simplex) {
            Self::matching_lifted_self_mirror_facet_index(simplex, facet_idx, context)?
        } else if uses_periodic_offsets {
            Self::matching_lifted_mirror_facet_index(simplex, facet_idx, neighbor_simplex, context)?
        } else {
            simplex
                .mirror_facet_index(facet_idx, neighbor_simplex)
                .ok_or_else(|| TdsError::InvalidNeighbors {
                    reason: NeighborValidationError::MirrorFacetMissing {
                        simplex_uuid: simplex.uuid(),
                        facet_index: facet_idx,
                        neighbor_uuid: neighbor_simplex.uuid(),
                        context: context.to_string(),
                    },
                })?
        };
        Ok((mirror_idx, uses_periodic_offsets))
    }

    /// Finds the other occurrence of a self-identified lifted facet.
    ///
    /// A quotient may store either two matching facet slots on one simplex or a
    /// single slot whose translated copy is represented implicitly. In the
    /// latter case the slot is its own mirror. More than two matching slots is
    /// structurally ambiguous.
    fn matching_lifted_self_mirror_facet_index(
        simplex: &Simplex<V, D>,
        facet_idx: usize,
        context: &str,
    ) -> Result<usize, TdsError> {
        let facet_key =
            Self::periodic_facet_key_from_simplex_vertices(simplex, simplex.vertices(), facet_idx)?;
        let mut other_match = None;
        for candidate_idx in 0..simplex.number_of_vertices() {
            if candidate_idx == facet_idx {
                continue;
            }
            let candidate_key = Self::periodic_facet_key_from_simplex_vertices(
                simplex,
                simplex.vertices(),
                candidate_idx,
            )?;
            if candidate_key == facet_key && other_match.replace(candidate_idx).is_some() {
                return Err(TdsError::InvalidNeighbors {
                    reason: NeighborValidationError::MirrorFacetAmbiguous {
                        simplex_uuid: simplex.uuid(),
                        neighbor_uuid: simplex.uuid(),
                    },
                });
            }
        }

        let mirror_idx = other_match.unwrap_or(facet_idx);
        if simplex.neighbor_key(mirror_idx).flatten().is_none() {
            return Err(TdsError::InvalidNeighbors {
                reason: NeighborValidationError::MirrorFacetMissing {
                    simplex_uuid: simplex.uuid(),
                    facet_index: facet_idx,
                    neighbor_uuid: simplex.uuid(),
                    context: context.to_string(),
                },
            });
        }
        Ok(mirror_idx)
    }

    pub(super) fn facet_vertices_in_simplex_order(
        simplex: &Simplex<V, D>,
        omit_idx: usize,
    ) -> Result<SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>, TdsError> {
        if omit_idx >= simplex.number_of_vertices() {
            return Err(TdsError::IndexOutOfBounds {
                index: omit_idx,
                bound: simplex.number_of_vertices(),
                context: format!(
                    "facet index for simplex {:?} during orientation validation",
                    simplex.uuid(),
                ),
            });
        }

        let mut facet_vertices = SmallBuffer::new();
        for (idx, &vkey) in simplex.vertices().iter().enumerate() {
            if idx != omit_idx {
                facet_vertices.push(vkey);
            }
        }
        Ok(facet_vertices)
    }
    /// Build facet vertex identities in simplex-local order, including periodic offsets.
    ///
    /// Offsets are normalized by subtracting a deterministic anchor offset so the
    /// same lifted facet can be compared across neighboring simplices independent of a
    /// global translation. The anchor is selected lexicographically by
    /// `(vertex_key_value, offset)`.
    fn facet_vertex_identities_in_simplex_order(
        simplex: &Simplex<V, D>,
        omit_idx: usize,
    ) -> Result<SmallBuffer<(VertexKey, [i16; D]), MAX_PRACTICAL_DIMENSION_SIZE>, TdsError> {
        if omit_idx >= simplex.number_of_vertices() {
            return Err(TdsError::IndexOutOfBounds {
                index: omit_idx,
                bound: simplex.number_of_vertices(),
                context: format!(
                    "facet index for simplex {:?} during orientation validation",
                    simplex.uuid(),
                ),
            });
        }

        let periodic_offsets = simplex.periodic_vertex_offsets();
        if let Some(offsets) = periodic_offsets
            && offsets.len() != simplex.number_of_vertices()
        {
            return Err(TdsError::DimensionMismatch {
                expected: simplex.number_of_vertices(),
                actual: offsets.len(),
                context: format!(
                    "periodic offset count for simplex {:?} during orientation validation",
                    simplex.uuid(),
                ),
            });
        }

        let mut facet_identities: SmallBuffer<(VertexKey, [i16; D]), MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::new();
        for (idx, &vkey) in simplex.vertices().iter().enumerate() {
            if idx == omit_idx {
                continue;
            }

            let raw_offset = periodic_offsets.map_or([0_i8; D], |offsets| offsets[idx]);
            let mut offset = [0_i16; D];
            for axis in 0..D {
                offset[axis] = i16::from(raw_offset[axis]);
            }
            facet_identities.push((vkey, offset));
        }

        let mut anchor_key = u64::MAX;
        let mut anchor_offset = [0_i16; D];
        for (vkey, offset) in &facet_identities {
            let key_value = vkey.data().as_ffi();
            if key_value < anchor_key || (key_value == anchor_key && *offset < anchor_offset) {
                anchor_key = key_value;
                anchor_offset = *offset;
            }
        }
        for (_, offset) in &mut facet_identities {
            for axis in 0..D {
                offset[axis] -= anchor_offset[axis];
            }
        }

        Ok(facet_identities)
    }

    /// Derive observed and expected facet permutation parity between neighboring simplices.
    ///
    /// Returns `(currently_coherent, observed_odd_permutation, expected_odd_permutation)`.
    /// The expected odd parity follows the coherent boundary-orientation convention:
    /// odd is expected exactly when `(facet_idx + mirror_idx)` is even.
    pub(crate) fn facet_permutation_parity(
        simplex: &Simplex<V, D>,
        facet_idx: usize,
        neighbor_simplex: &Simplex<V, D>,
        mirror_idx: usize,
    ) -> Result<(bool, bool, bool), TdsError> {
        // A single stored quotient facet represents adjacency to a translated
        // copy of the same simplex. The deck translation preserves the lifted
        // simplex orientation, so this implicit self-mirror contributes a
        // satisfied equality constraint rather than comparing the facet order
        // with itself as though it were an ordinary two-sided incidence.
        if facet_idx == mirror_idx
            && std::ptr::eq(simplex, neighbor_simplex)
            && simplex.periodic_vertex_offsets().is_some()
        {
            let expected_odd_permutation = (facet_idx + mirror_idx).is_multiple_of(2);
            return Ok((true, expected_odd_permutation, expected_odd_permutation));
        }

        let simplex_facet_identities =
            Self::facet_vertex_identities_in_simplex_order(simplex, facet_idx)?;
        let neighbor_facet_identities =
            Self::facet_vertex_identities_in_simplex_order(neighbor_simplex, mirror_idx)?;

        let observed_odd_permutation = Self::permutation_is_odd(
            &simplex_facet_identities[..],
            &neighbor_facet_identities[..],
        )
        .ok_or_else(|| TdsError::InconsistentDataStructure {
            message: format!(
                "Could not derive facet-order permutation parity between simplices {:?} and {:?}",
                simplex.uuid(),
                neighbor_simplex.uuid(),
            ),
        })?;

        let expected_odd_permutation = (facet_idx + mirror_idx).is_multiple_of(2);
        Ok((
            observed_odd_permutation == expected_odd_permutation,
            observed_odd_permutation,
            expected_odd_permutation,
        ))
    }
    /// Returns whether the permutation mapping `source_order` to `target_order` is odd.
    ///
    /// Returns `None` if the orders are not permutations of each other.
    fn permutation_is_odd<Id: PartialEq>(source_order: &[Id], target_order: &[Id]) -> Option<bool> {
        if source_order.len() != target_order.len() {
            return None;
        }

        let mut target_positions: SmallBuffer<usize, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(source_order.len());
        let mut used_target_indices: SmallBuffer<bool, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::from_elem(false, target_order.len());

        for source_vertex in source_order {
            let mut matched_target_position = None;
            for (target_idx, target_vertex) in target_order.iter().enumerate() {
                if target_vertex == source_vertex && !used_target_indices[target_idx] {
                    matched_target_position = Some(target_idx);
                    used_target_indices[target_idx] = true;
                    break;
                }
            }
            target_positions.push(matched_target_position?);
        }

        if used_target_indices.iter().any(|used| !*used) {
            return None;
        }

        let mut is_odd = false;
        for i in 0..target_positions.len() {
            for j in (i + 1)..target_positions.len() {
                if target_positions[i] > target_positions[j] {
                    is_odd = !is_odd;
                }
            }
        }

        Some(is_odd)
    }

    /// Validate facet multiplicity from an already-built facet incidence map.
    ///
    /// This helper exists so [`Tds::is_valid`] and validation reports can share
    /// one O(N×F) facet-map construction while still emitting the same structured
    /// [`TdsError::FacetSharingViolation`] as insertion preflight. It returns the
    /// first facet incident to more than two simplices, identifying one offending
    /// incident simplex in the `candidate_*` fields.
    fn validate_facet_sharing_with_facet_to_simplices_map(
        &self,
        facet_to_simplices: &FacetToSimplicesMap,
    ) -> Result<(), TdsError> {
        // Check for facets shared by more than 2 simplices.
        for (facet_key, simplex_facet_pairs) in facet_to_simplices {
            let [_, _, candidate, ..] = simplex_facet_pairs.as_slice() else {
                continue;
            };
            let candidate_simplex =
                self.simplex(candidate.simplex_key())
                    .ok_or_else(|| TdsError::SimplexNotFound {
                        simplex_key: candidate.simplex_key(),
                        context: format!(
                            "facet-sharing validation for over-shared facet {facet_key}"
                        ),
                    })?;
            return Err(TdsError::FacetSharingViolation {
                facet_key: *facet_key,
                existing_incident_count: simplex_facet_pairs.len() - 1,
                attempted_incident_count: simplex_facet_pairs.len(),
                max_incident_count: 2,
                candidate_simplex_uuid: candidate_simplex.uuid(),
                candidate_facet_index: usize::from(candidate.facet_index()),
            });
        }

        Ok(())
    }

    /// Checks whether the triangulation data structure is structurally valid.
    ///
    /// This is a **Level 2 (TDS structural)** check in the validation hierarchy.
    /// It intentionally does **not** validate individual vertices/simplices (Level 1),
    /// nor triangulation topology (Level 3), valid realization (Level 4),
    /// or the Delaunay property (Level 5).
    ///
    /// # Structural invariants checked
    /// - Vertex UUID↔key mapping consistency
    /// - Simplex UUID↔key mapping consistency
    /// - Simplices reference only valid vertex keys (no stale/missing vertex keys)
    /// - `Vertex::incident_simplex`, when present, must point at an existing simplex that contains the vertex.
    /// - No duplicate simplices (same vertex set)
    /// - Facet sharing invariant (each facet is shared by at most 2 simplices,
    ///   reported as [`TdsError::FacetSharingViolation`])
    /// - Neighbor consistency (topology + mutual neighbors)
    /// - Coherent orientation (adjacent simplices induce opposite facet orientations)
    ///
    /// # ⚠️ Performance Warning
    ///
    /// **This method can be expensive** for large triangulations:
    /// - **Time Complexity**: O(N×F + N×D²) where N is the number of simplices and F = D+1 facets per simplex
    /// - **Space Complexity**: O(N×F) for facet-to-simplex mappings
    ///
    /// For a cumulative validator that also checks vertices/simplices (Level 1), use
    /// [`Tds::validate`](Self::validate).
    ///
    /// # Errors
    ///
    /// Returns a [`TdsError`] if any structural invariant fails.
    ///
    /// Checks whether the triangulation data structure is structurally valid.
    ///
    /// This is the canonical Level 2 fast-fail API. It returns the first
    /// structural error that proves the TDS incidence graph is invalid.
    ///
    /// For an actionable first diagnostic use
    /// [`structure_diagnostic`](Self::structure_diagnostic). For all checkable
    /// structural failures use [`structure_report`](Self::structure_report).
    ///
    /// # Errors
    ///
    /// Returns a [`TdsError`] if any structural invariant fails.
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
    /// let vertices_4d = [
    ///     delaunay::vertex![0.0, 0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 0.0, 1.0]?,
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulationBuilder::new(&vertices_4d).build()?;
    ///
    /// // Level 2: structural validation
    /// assert!(dt.is_valid_structure().is_ok());
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_valid(&self) -> Result<(), TdsError> {
        // Fast-fail: return the first violated invariant.
        // For full diagnostics across all structural invariants, use `structure_report()`.
        self.validate_vertex_mappings()?;
        self.validate_simplex_mappings()?;

        // Defensive: ensure no simplex references a stale/missing vertex key before
        // higher-level structural checks that assume key validity.
        self.validate_simplex_vertex_keys()?;

        // Structural: ensure `incident_simplex` pointers, when present, are non-dangling + consistent.
        self.validate_vertex_incidence()?;
        self.validate_vertex_to_simplices_index()?;

        self.validate_no_duplicate_simplices()?;

        // Build the facet-to-simplices map once and share it between facet-sharing and neighbor validators.
        let facet_to_simplices = self.build_facet_to_simplices_map()?;
        self.validate_facet_sharing_with_facet_to_simplices_map(&facet_to_simplices)?;
        self.validate_neighbors_with_facet_to_simplices_map(&facet_to_simplices)?;
        self.validate_coherent_orientation()?;

        Ok(())
    }

    /// Returns the first actionable Level 2 structural diagnostic, if any.
    ///
    /// This is the repair/retry-oriented counterpart to
    /// [`is_valid`](Self::is_valid): it preserves the
    /// [`InvariantKind`] grouping used by aggregate reports while still
    /// returning at most one local failure.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::tds::Tds;
    ///
    /// let tds = Tds::<(), (), 2>::empty();
    /// assert!(tds.structure_diagnostic().is_none());
    /// ```
    #[must_use]
    pub fn structure_diagnostic(&self) -> Option<InvariantViolation> {
        self.structure_report()
            .err()
            .and_then(|report| report.violations.into_iter().next())
    }

    /// Performs cumulative validation for Levels 1–2.
    ///
    /// This validates:
    /// - **Level 1**: all vertices (`Vertex::is_valid`), all simplices (`Simplex::is_valid`),
    ///   and simplex-local coordinate uniqueness
    /// - **Level 2**: structural invariants (`Tds::is_valid`)
    ///
    /// # Errors
    ///
    /// Returns a [`TdsError`] if any vertex/simplex is invalid or if any
    /// structural invariant fails.
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
    /// let vertices_4d = [
    ///     delaunay::vertex![0.0, 0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 0.0, 1.0]?,
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulationBuilder::new(&vertices_4d).build()?;
    ///
    /// // Levels 1–2: elements + structure
    /// assert!(dt.validate_structure().is_ok());
    /// # Ok(())
    /// # }
    /// ```
    pub fn validate(&self) -> Result<(), TdsError> {
        for (_vertex_key, vertex) in &self.vertices {
            if let Err(source) = (*vertex).is_valid() {
                return Err(TdsError::InvalidVertex {
                    vertex_id: vertex.uuid(),
                    source,
                });
            }
        }

        for (simplex_key, simplex) in &self.simplices {
            if let Err(source) = simplex.is_valid() {
                let Some(simplex_id) = self.simplex_uuid_from_key(simplex_key) else {
                    return Err(TdsError::InconsistentDataStructure {
                        message: format!(
                            "Simplex key {simplex_key:?} has no UUID mapping during validation",
                        ),
                    });
                };

                return Err(TdsError::InvalidSimplex { simplex_id, source });
            }
        }

        // Coordinate-level duplicate detection: different vertex keys with identical
        // coordinates produce zero-volume simplices that break SoS and Pachner moves.
        // Guard behind simplex-vertex-key validity so that stale keys are reported as
        // key-reference failures (by is_valid below) rather than confusing coordinate
        // errors.  Matches the pattern in validation_report().
        if self.validate_simplex_vertex_keys().is_ok() {
            self.validate_simplex_coordinate_uniqueness()?;
        }

        self.is_valid()
    }

    /// Runs Level 2 structure checks and returns all checkable structural failures.
    ///
    /// Unlike [`is_valid`](Self::is_valid), this method does
    /// **not** stop at the first error. Instead it records a [`TdsError`] for each
    /// invariant group that fails and returns them as a
    /// [`TriangulationValidationReport`].
    ///
    /// **Note**: If UUID↔key mappings are inconsistent, this returns only mapping-related
    /// failures. Additional checks may produce misleading secondary errors or panic.
    ///
    /// **Note**: If any simplex references a stale/missing vertex key, this reports the
    /// key-reference failure (and any vertex-incidence failures) and skips derived
    /// invariants that assume key validity.
    ///
    /// This is primarily intended for debugging, diagnostics, tests, and repair
    /// planning that need local structured failures instead of only the first
    /// error.
    ///
    /// **Note**: This does NOT check the Delaunay property. Use
    /// `Triangulation::validate_realization()` for Levels 1–4, `DelaunayTriangulation::is_valid_delaunay()`
    /// for Level 5 only, or `DelaunayTriangulation::validate()` for cumulative Levels 1–5.
    ///
    /// # Errors
    ///
    /// Returns a [`TriangulationValidationReport`] containing all invariant
    /// violations if any validation step fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let tds: Tds<(), (), 2> = Tds::empty();
    /// assert!(tds.structure_report().is_ok());
    /// ```
    pub fn structure_report(&self) -> Result<(), TriangulationValidationReport> {
        let mut violations = Vec::new();

        // 1. Mapping consistency (vertex + simplex UUID↔key mappings)
        if let Err(e) = self.validate_vertex_mappings() {
            violations.push(InvariantViolation {
                kind: InvariantKind::VertexMappings,
                error: e.into(),
            });
        }
        if let Err(e) = self.validate_simplex_mappings() {
            violations.push(InvariantViolation {
                kind: InvariantKind::SimplexMappings,
                error: e.into(),
            });
        }

        // If mappings are inconsistent, additional checks may produce confusing
        // secondary errors or panic. In that case, stop here and return the
        // mapping-related failures only.
        if !violations.is_empty() {
            return Err(TriangulationValidationReport { violations });
        }

        // 2. Simplex→vertex key references (no stale/missing vertex keys)
        let mut simplex_vertex_keys_ok = true;
        if let Err(e) = self.validate_simplex_vertex_keys() {
            simplex_vertex_keys_ok = false;
            violations.push(InvariantViolation {
                kind: InvariantKind::SimplexVertexKeys,
                error: e.into(),
            });
        }

        // 3. Vertex incidence (non-dangling `incident_simplex` pointers, when present)
        if let Err(e) = self.validate_vertex_incidence() {
            violations.push(InvariantViolation {
                kind: InvariantKind::VertexIncidence,
                error: e.into(),
            });
        }

        if simplex_vertex_keys_ok && let Err(e) = self.validate_vertex_to_simplices_index() {
            violations.push(InvariantViolation {
                kind: InvariantKind::VertexToSimplicesIndex,
                error: e.into(),
            });
        }

        // If simplex vertex keys are invalid, derived invariants may produce confusing secondary
        // errors or panic (many routines assume key validity). Stop here.
        if !simplex_vertex_keys_ok {
            return Err(TriangulationValidationReport { violations });
        }

        // 4. Simplex uniqueness (no duplicate simplices with identical vertex sets)
        if let Err(e) = self.validate_no_duplicate_simplices() {
            violations.push(InvariantViolation {
                kind: InvariantKind::DuplicateSimplices,
                error: e.into(),
            });
        }

        // 5–7. Facet sharing + neighbor consistency + coherent orientation.
        let mut neighbor_consistency_ok = false;
        match self.build_facet_to_simplices_map() {
            Ok(facet_to_simplices) => {
                if let Err(e) =
                    self.validate_facet_sharing_with_facet_to_simplices_map(&facet_to_simplices)
                {
                    violations.push(InvariantViolation {
                        kind: InvariantKind::FacetSharing,
                        error: e.into(),
                    });
                }

                if let Err(e) =
                    self.validate_neighbors_with_facet_to_simplices_map(&facet_to_simplices)
                {
                    violations.push(InvariantViolation {
                        kind: InvariantKind::NeighborConsistency,
                        error: e.into(),
                    });
                } else {
                    neighbor_consistency_ok = true;
                }
            }
            Err(e) => {
                // If we can't build the facet map, both facet-sharing and neighbor checks are blocked.
                //
                // We intentionally record *both* invariant kinds for diagnostic granularity.
                // This requires cloning the error once (so each violation owns an error), which
                // may allocate/copy string payloads, but this is on an error path and facet-map
                // build errors are expected to be rare and small.
                violations.push(InvariantViolation {
                    kind: InvariantKind::FacetSharing,
                    error: e.clone().into(),
                });
                violations.push(InvariantViolation {
                    kind: InvariantKind::NeighborConsistency,
                    error: e.into(),
                });
            }
        }

        if neighbor_consistency_ok && let Err(e) = self.validate_coherent_orientation() {
            violations.push(InvariantViolation {
                kind: InvariantKind::CoherentOrientation,
                error: e.into(),
            });
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(TriangulationValidationReport { violations })
        }
    }

    /// Generate a cumulative validation report for Levels 1–2.
    ///
    /// This report combines Level 1 element validity with the Level 2
    /// [`structure_report`](Self::structure_report).
    ///
    /// # Errors
    ///
    /// Returns `Err(TriangulationValidationReport)` containing all checkable
    /// Level 1-2 invariant violations.
    pub fn validation_report(&self) -> Result<(), TriangulationValidationReport> {
        let mut violations = Vec::new();

        for (_vertex_key, vertex) in &self.vertices {
            if let Err(report) = (*vertex).vertex_report() {
                violations.extend(report.violations.into_iter().map(|source| {
                    InvariantViolation {
                        kind: InvariantKind::VertexValidity,
                        error: TdsError::InvalidVertex {
                            vertex_id: vertex.uuid(),
                            source,
                        }
                        .into(),
                    }
                }));
            }
        }

        for (simplex_key, simplex) in &self.simplices {
            if let Err(report) = simplex.simplex_report() {
                violations.extend(report.violations.into_iter().map(|source| {
                    let error = self.simplex_uuid_from_key(simplex_key).map_or_else(
                        || TdsError::InconsistentDataStructure {
                            message: format!(
                                "Simplex key {simplex_key:?} has no UUID mapping during validation",
                            ),
                        },
                        |simplex_id| TdsError::InvalidSimplex { simplex_id, source },
                    );
                    InvariantViolation {
                        kind: InvariantKind::SimplexValidity,
                        error: error.into(),
                    }
                }));
            }
        }

        if self.validate_simplex_vertex_keys().is_ok()
            && let Err(error) = self.validate_simplex_coordinate_uniqueness()
        {
            violations.push(InvariantViolation {
                kind: InvariantKind::SimplexCoordinateUniqueness,
                error: error.into(),
            });
        }

        if let Err(report) = self.structure_report() {
            violations.extend(report.violations);
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(TriangulationValidationReport { violations })
        }
    }

    /// Validates global neighbor relationships for the triangulation.
    ///
    /// This method validates:
    /// - Topological invariant (neighbor\[i\] is opposite vertex\[i\]) via `validate_neighbor_topology()`
    /// - Mutual neighbor relationships (if A neighbors B, then B neighbors A)
    /// - Shared facet correctness (neighbors share exactly D vertices)
    ///
    /// # Performance / intended use
    ///
    /// This routine is intentionally thorough and defensive. It precomputes per-simplex vertex sets
    /// and performs per-neighbor set-intersection + mirror-facet cross-checks, which can be
    /// relatively expensive for large triangulations.
    ///
    /// It is only invoked from explicit validation APIs (`is_valid()`, `validate()`,
    /// `validation_report()`) and is not intended for per-step hot paths.
    ///
    /// Some small optimizations keep the cost reasonable:
    /// - Early termination on validation failures
    /// - Precomputing per-simplex vertex sets once
    /// - Counting intersections without allocating intermediate collections
    ///
    /// # Errors
    ///
    /// Returns a [`TdsError`] if any neighbor relationship
    /// violates topological or consistency invariants.
    ///
    /// This corresponds to [`InvariantKind::NeighborConsistency`], which is included in
    /// [`Tds::is_valid`](Self::is_valid) and [`Tds::validate`](Self::validate), and is also surfaced by
    /// [`DelaunayTriangulation::validation_report()`].
    ///
    /// [`DelaunayTriangulation::validation_report()`]: crate::DelaunayTriangulation::validation_report
    ///
    /// Note: callers provide `facet_to_simplices` so `is_valid()` and `validation_report()` can share
    /// the precomputed facet map between validators.
    fn validate_neighbors_with_facet_to_simplices_map(
        &self,
        facet_to_simplices: &FacetToSimplicesMap,
    ) -> Result<(), TdsError> {
        self.validate_neighbor_pointers_match_facet_to_simplices_map(facet_to_simplices)?;

        let simplex_vertices = self.build_simplex_vertex_sets()?;
        self.validate_neighbors_with_precomputed_vertex_sets(&simplex_vertices)
    }

    /// Validates the topological invariant for neighbor relationships.
    ///
    /// **Critical Invariant**: For a simplex, `neighbors[i]` must be opposite `vertices[i]`,
    /// meaning the two simplices share a facet containing all vertices **except** vertex `i`.
    ///
    /// # Errors
    ///
    /// Returns [`TdsError`] if topology validation fails.
    pub(super) fn validate_neighbor_topology(
        &self,
        simplex_key: SimplexKey,
        neighbors: &[Option<SimplexKey>],
    ) -> Result<(), TdsError> {
        if neighbors.len() != D + 1 {
            return Err(TdsError::InvalidNeighbors {
                reason: NeighborValidationError::LengthMismatch {
                    actual: neighbors.len(),
                    expected: D + 1,
                    context: "neighbor topology validation".to_string(),
                },
            });
        }

        let simplex = self
            .simplices
            .get(simplex_key)
            .ok_or_else(|| TdsError::SimplexNotFound {
                simplex_key,
                context: "validate_neighbor_topology".to_string(),
            })?;

        let simplex_lifted_vertices = Self::lifted_vertex_identities(simplex_key, simplex)?;

        for (i, neighbor_key_opt) in neighbors.iter().enumerate() {
            if let Some(neighbor_key) = neighbor_key_opt {
                // Self-adjacency: a simplex can be its own neighbor on a closed manifold (e.g.
                // a torus). In that case the invariant "neighbor[i] shares the facet opposite
                // vertex[i]" is trivially satisfied by the periodic identification.
                if *neighbor_key == simplex_key {
                    if Self::allows_periodic_self_neighbor(simplex) {
                        continue;
                    }
                    return Err(TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::NonPeriodicSelfNeighbor {
                            simplex_key,
                            simplex_uuid: simplex.uuid(),
                            facet_index: i,
                        },
                    });
                }

                let neighbor = self.simplices.get(*neighbor_key).ok_or_else(|| {
                    TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::MissingNeighborSimplex {
                            simplex_key,
                            simplex_uuid: simplex.uuid(),
                            facet_index: i,
                            neighbor_key: *neighbor_key,
                            context: "neighbor topology validation".to_string(),
                        },
                    }
                })?;

                let uses_periodic_offsets = simplex.periodic_vertex_offsets().is_some()
                    || neighbor.periodic_vertex_offsets().is_some();
                let (shared_count, missing_vertex_idx) = if uses_periodic_offsets {
                    // Periodic quotient simplices may be represented in different translated
                    // frames. Compare normalized lifted facet identities so offset-distinct
                    // vertices remain distinct while globally translated representatives match.
                    let matching_facet_index =
                        Self::matching_lifted_facet_index(simplex, neighbor)?;
                    (matching_facet_index.map_or(0, |_| D), matching_facet_index)
                } else {
                    let neighbor_lifted_vertices =
                        Self::lifted_vertex_identities(*neighbor_key, neighbor)?;

                    let mut shared_count = 0;
                    let mut missing_vertex_idx = None;

                    for (idx, simplex_vertex_identity) in simplex_lifted_vertices.iter().enumerate()
                    {
                        if neighbor_lifted_vertices
                            .iter()
                            .any(|neighbor_vertex_identity| {
                                neighbor_vertex_identity == simplex_vertex_identity
                            })
                        {
                            shared_count += 1;
                        } else if missing_vertex_idx.is_none() {
                            missing_vertex_idx = Some(idx);
                        }
                    }
                    (shared_count, missing_vertex_idx)
                };

                if shared_count != D {
                    return Err(TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::SharedVertexCountMismatch {
                            simplex_key,
                            simplex_uuid: simplex.uuid(),
                            facet_index: i,
                            shared_count,
                            expected: D,
                        },
                    });
                }

                if missing_vertex_idx != Some(i) {
                    return Err(TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::OppositeVertexMismatch {
                            simplex_key,
                            simplex_uuid: simplex.uuid(),
                            facet_index: i,
                            observed_opposite: missing_vertex_idx,
                            expected_opposite: i,
                        },
                    });
                }
            }
        }

        Ok(())
    }

    fn validate_neighbor_pointers_match_facet_to_simplices_map(
        &self,
        facet_to_simplices: &FacetToSimplicesMap,
    ) -> Result<(), TdsError> {
        for (facet_key, simplex_facet_pairs) in facet_to_simplices {
            match simplex_facet_pairs.as_slice() {
                [handle] => self.validate_boundary_facet_neighbor_pointer(*facet_key, *handle)?,
                [a, b] => self.validate_interior_facet_neighbor_pointer(*facet_key, *a, *b)?,
                _ => {
                    // Non-manifold facet multiplicity should have been caught by facet-sharing validation.
                    return Err(TdsError::InconsistentDataStructure {
                        message: format!(
                            "Facet with key {facet_key} is shared by {} simplices, but should be shared by at most 2 simplices in a valid triangulation",
                            simplex_facet_pairs.len()
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    fn validate_boundary_facet_neighbor_pointer(
        &self,
        facet_key: u64,
        handle: FacetHandle,
    ) -> Result<(), TdsError> {
        let simplex_key = handle.simplex_key();
        let facet_index = handle.facet_index() as usize;
        let simplex = self
            .simplices
            .get(simplex_key)
            .ok_or_else(|| TdsError::SimplexNotFound {
                simplex_key,
                context: "neighbor validation (boundary facet)".to_string(),
            })?;

        let Some(neighbor_slots) = simplex.neighbor_slots() else {
            return Ok(());
        };
        let Some(neighbor_slot) = neighbor_slots.get(facet_index).copied() else {
            return Err(TdsError::InvalidNeighbors {
                reason: NeighborValidationError::LengthMismatch {
                    actual: neighbor_slots.len(),
                    expected: D + 1,
                    context: "neighbor validation (boundary facet)".to_string(),
                },
            });
        };

        match neighbor_slot {
            NeighborSlot::Unassigned => Err(TdsError::InvalidNeighbors {
                reason: NeighborValidationError::UnassignedNeighborSlot {
                    simplex_key,
                    simplex_uuid: simplex.uuid(),
                    facet_index,
                    context: "neighbor validation (boundary facet)".to_string(),
                },
            }),
            NeighborSlot::Boundary => Ok(()),
            NeighborSlot::Neighbor(neighbor) if neighbor == simplex_key => {
                if Self::allows_periodic_self_neighbor(simplex) {
                    return Ok(());
                }
                Err(TdsError::InvalidNeighbors {
                    reason: NeighborValidationError::BoundaryFacetHasNonPeriodicSelfNeighbor {
                        facet_key,
                        simplex_key,
                        simplex_uuid: simplex.uuid(),
                        facet_index,
                    },
                })
            }
            NeighborSlot::Neighbor(neighbor) => Err(TdsError::InvalidNeighbors {
                reason: NeighborValidationError::BoundaryFacetHasNeighbor {
                    facet_key,
                    simplex_key,
                    simplex_uuid: simplex.uuid(),
                    facet_index,
                    neighbor_key: neighbor,
                },
            }),
        }
    }

    fn validate_interior_facet_neighbor_pointer(
        &self,
        facet_key: u64,
        first: FacetHandle,
        second: FacetHandle,
    ) -> Result<(), TdsError> {
        let first_simplex_key = first.simplex_key();
        let first_facet_index = first.facet_index() as usize;
        let second_simplex_key = second.simplex_key();
        let second_facet_index = second.facet_index() as usize;

        let first_simplex =
            self.simplices
                .get(first_simplex_key)
                .ok_or_else(|| TdsError::SimplexNotFound {
                    simplex_key: first_simplex_key,
                    context: "neighbor validation (interior facet, first simplex)".to_string(),
                })?;
        let second_simplex =
            self.simplices
                .get(second_simplex_key)
                .ok_or_else(|| TdsError::SimplexNotFound {
                    simplex_key: second_simplex_key,
                    context: "neighbor validation (interior facet, second simplex)".to_string(),
                })?;

        let first_neighbor = first_simplex.neighbor_key(first_facet_index).flatten();
        let second_neighbor = second_simplex.neighbor_key(second_facet_index).flatten();

        if first_neighbor == Some(second_simplex_key) && second_neighbor == Some(first_simplex_key)
        {
            return Ok(());
        }

        Err(TdsError::InvalidNeighbors {
            reason: NeighborValidationError::InteriorFacetNeighborMismatch {
                facet_key,
                first_simplex_key,
                first_simplex_uuid: first_simplex.uuid(),
                first_facet_index,
                first_neighbor,
                second_simplex_key,
                second_simplex_uuid: second_simplex.uuid(),
                second_facet_index,
                second_neighbor,
            },
        })
    }

    fn validate_neighbors_with_precomputed_vertex_sets(
        &self,
        simplex_vertices: &SimplexVerticesMap,
    ) -> Result<(), TdsError> {
        for (simplex_key, simplex) in &self.simplices {
            let Some(neighbor_keys) = simplex.neighbor_keys() else {
                continue; // Skip simplices without neighbors
            };
            let neighbors_buf: NeighborBuffer<Option<SimplexKey>> = neighbor_keys.collect();

            // Validate topological invariant (neighbor[i] opposite vertex[i])
            self.validate_neighbor_topology(simplex_key, &neighbors_buf)?;

            let this_vertices = simplex_vertices.get(&simplex_key).ok_or_else(|| {
                TdsError::InconsistentDataStructure {
                    message: format!(
                        "Simplex {} (key {simplex_key:?}) missing from precomputed vertex set map during neighbor validation",
                        simplex.uuid()
                    ),
                }
            })?;

            for (facet_idx, neighbor_key_opt) in neighbors_buf.iter().copied().enumerate() {
                // Skip None neighbors (missing neighbors)
                let Some(neighbor_key) = neighbor_key_opt else {
                    continue;
                };

                // Self-adjacency is valid for periodic quotient triangulations.
                if neighbor_key == simplex_key {
                    if Self::allows_periodic_self_neighbor(simplex) {
                        continue;
                    }
                    return Err(TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::NonPeriodicSelfNeighbor {
                            simplex_key,
                            simplex_uuid: simplex.uuid(),
                            facet_index: facet_idx,
                        },
                    });
                }

                // Early termination: check if neighbor exists
                let Some(neighbor_simplex) = self.simplices.get(neighbor_key) else {
                    return Err(TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::MissingNeighborSimplex {
                            simplex_key,
                            simplex_uuid: simplex.uuid(),
                            facet_index: facet_idx,
                            neighbor_key,
                            context: "precomputed neighbor validation".to_string(),
                        },
                    });
                };

                if simplex.periodic_vertex_offsets().is_some()
                    || neighbor_simplex.periodic_vertex_offsets().is_some()
                {
                    continue;
                }

                let neighbor_vertices = simplex_vertices.get(&neighbor_key).ok_or_else(|| {
                    TdsError::InconsistentDataStructure {
                        message: format!(
                            "Neighbor simplex {} (key {neighbor_key:?}) missing from precomputed vertex set map during neighbor validation",
                            neighbor_simplex.uuid()
                        ),
                    }
                })?;

                Self::validate_shared_facet_count(
                    simplex,
                    neighbor_simplex,
                    this_vertices,
                    neighbor_vertices,
                )?;

                let mirror_idx = Self::verified_mirror_facet_index(
                    simplex,
                    facet_idx,
                    neighbor_simplex,
                    this_vertices,
                )?;

                Self::validate_shared_facet_vertices(
                    simplex,
                    facet_idx,
                    neighbor_simplex,
                    mirror_idx,
                    this_vertices,
                    neighbor_vertices,
                )?;

                Self::validate_mutual_neighbor_back_reference(
                    simplex_key,
                    simplex,
                    facet_idx,
                    neighbor_key,
                    neighbor_simplex,
                    mirror_idx,
                )?;
            }
        }

        Ok(())
    }

    fn build_simplex_vertex_sets(&self) -> Result<SimplexVerticesMap, TdsError> {
        // Pre-compute vertex keys for all simplices to avoid repeated computation
        let mut simplex_vertices: SimplexVerticesMap =
            fast_hash_map_with_capacity(self.simplices.len());

        for simplex_key in self.simplices.keys() {
            // Use simplex_vertices to ensure all vertex keys are present
            // The error is already TdsError, so just propagate it
            let vertices = self.simplex_vertices(simplex_key)?;

            // Store the HashSet for containment checks
            let vertex_set: VertexKeySet = vertices.iter().copied().collect();
            simplex_vertices.insert(simplex_key, vertex_set);
        }

        Ok(simplex_vertices)
    }

    fn validate_shared_facet_count(
        simplex: &Simplex<V, D>,
        neighbor_simplex: &Simplex<V, D>,
        this_vertices: &VertexKeySet,
        neighbor_vertices: &VertexKeySet,
    ) -> Result<(), TdsError> {
        let shared_count = this_vertices.intersection(neighbor_vertices).count();

        if shared_count != D {
            return Err(TdsError::NotNeighbors {
                simplex1: simplex.uuid(),
                simplex2: neighbor_simplex.uuid(),
            });
        }

        Ok(())
    }

    fn verified_mirror_facet_index(
        simplex: &Simplex<V, D>,
        facet_idx: usize,
        neighbor_simplex: &Simplex<V, D>,
        this_vertices: &VertexKeySet,
    ) -> Result<usize, TdsError> {
        let mirror_idx = simplex
            .mirror_facet_index(facet_idx, neighbor_simplex)
            .ok_or_else(|| TdsError::InvalidNeighbors {
                reason: NeighborValidationError::MirrorFacetMissing {
                    simplex_uuid: simplex.uuid(),
                    facet_index: facet_idx,
                    neighbor_uuid: neighbor_simplex.uuid(),
                    context: "neighbor validation".to_string(),
                },
            })?;

        // Defensive cross-check: verify the mirror index against shared-vertex analysis.
        // This adds overhead but guards against subtle logic bugs in `mirror_facet_index()`.
        //
        // If validation ever becomes performance-sensitive, this is a good candidate to
        // gate behind a "strict validation" option/flag.
        let expected_mirror_idx =
            Self::expected_mirror_facet_index(simplex, neighbor_simplex, this_vertices)?;

        if mirror_idx != expected_mirror_idx {
            return Err(TdsError::InvalidNeighbors {
                reason: NeighborValidationError::MirrorFacetIndexMismatch {
                    simplex_uuid: simplex.uuid(),
                    facet_index: facet_idx,
                    neighbor_uuid: neighbor_simplex.uuid(),
                    observed_mirror_index: mirror_idx,
                    expected_mirror_index: expected_mirror_idx,
                },
            });
        }

        Ok(mirror_idx)
    }

    fn expected_mirror_facet_index(
        simplex: &Simplex<V, D>,
        neighbor_simplex: &Simplex<V, D>,
        this_vertices: &VertexKeySet,
    ) -> Result<usize, TdsError> {
        let mut expected_mirror_idx: Option<usize> = None;

        for (idx, &neighbor_vkey) in neighbor_simplex.vertices().iter().enumerate() {
            if !this_vertices.contains(&neighbor_vkey) {
                if expected_mirror_idx.is_some() {
                    return Err(TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::MirrorFacetAmbiguous {
                            simplex_uuid: simplex.uuid(),
                            neighbor_uuid: neighbor_simplex.uuid(),
                        },
                    });
                }
                expected_mirror_idx = Some(idx);
            }
        }

        expected_mirror_idx.ok_or_else(|| TdsError::InvalidNeighbors {
            reason: NeighborValidationError::MirrorFacetDuplicateSimplices {
                simplex_uuid: simplex.uuid(),
                neighbor_uuid: neighbor_simplex.uuid(),
            },
        })
    }

    fn validate_shared_facet_vertices(
        simplex: &Simplex<V, D>,
        facet_idx: usize,
        neighbor_simplex: &Simplex<V, D>,
        mirror_idx: usize,
        this_vertices: &VertexKeySet,
        neighbor_vertices: &VertexKeySet,
    ) -> Result<(), TdsError> {
        for (idx, &vkey) in simplex.vertices().iter().enumerate() {
            if idx == facet_idx {
                continue;
            }
            if !neighbor_vertices.contains(&vkey) {
                return Err(TdsError::InvalidNeighbors {
                    reason: NeighborValidationError::SharedFacetMissingVertex {
                        side: SharedFacetMismatchSide::SourceFacet,
                        simplex_uuid: simplex.uuid(),
                        facet_index: facet_idx,
                        neighbor_uuid: neighbor_simplex.uuid(),
                        mirror_index: mirror_idx,
                        missing_vertex: vkey,
                    },
                });
            }
        }

        for (idx, &vkey) in neighbor_simplex.vertices().iter().enumerate() {
            if idx == mirror_idx {
                continue;
            }
            if !this_vertices.contains(&vkey) {
                return Err(TdsError::InvalidNeighbors {
                    reason: NeighborValidationError::SharedFacetMissingVertex {
                        side: SharedFacetMismatchSide::NeighborFacet,
                        simplex_uuid: simplex.uuid(),
                        facet_index: facet_idx,
                        neighbor_uuid: neighbor_simplex.uuid(),
                        mirror_index: mirror_idx,
                        missing_vertex: vkey,
                    },
                });
            }
        }

        Ok(())
    }

    fn validate_mutual_neighbor_back_reference(
        simplex_key: SimplexKey,
        simplex: &Simplex<V, D>,
        facet_idx: usize,
        neighbor_key: SimplexKey,
        neighbor_simplex: &Simplex<V, D>,
        mirror_idx: usize,
    ) -> Result<(), TdsError> {
        let Some(back_ref) = neighbor_simplex.neighbor_key(mirror_idx) else {
            return Err(TdsError::InvalidNeighbors {
                reason: NeighborValidationError::BackReferenceMismatch {
                    simplex_key,
                    simplex_uuid: simplex.uuid(),
                    facet_index: facet_idx,
                    neighbor_key,
                    neighbor_uuid: neighbor_simplex.uuid(),
                    mirror_index: mirror_idx,
                    observed: None,
                    context: "neighbor validation; neighbor has no neighbor buffer".to_string(),
                },
            });
        };

        if back_ref != Some(simplex_key) {
            return Err(TdsError::InvalidNeighbors {
                reason: NeighborValidationError::BackReferenceMismatch {
                    simplex_key,
                    simplex_uuid: simplex.uuid(),
                    facet_index: facet_idx,
                    neighbor_key,
                    neighbor_uuid: neighbor_simplex.uuid(),
                    mirror_index: mirror_idx,
                    observed: back_ref,
                    context: "neighbor validation".to_string(),
                },
            });
        }

        Ok(())
    }
}
// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DelaunayTriangulation;
    use crate::core::simplex::Simplex;
    use crate::core::tds::errors::{InvariantError, NeighborValidationError, TdsError};
    use crate::core::vertex::Vertex;
    use crate::vertex;
    use slotmap::KeyData;
    use std::assert_matches;
    use uuid::Uuid;

    fn initial_simplex_vertices_3d() -> [Vertex<(), 3>; 4] {
        [
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ]
    }

    #[test]
    fn test_facet_vertex_identities_anchor_uses_lexicographic_key_offset() {
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

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();

        // Deliberately corrupt for regression coverage: duplicate v_a with a smaller
        // periodic lift so anchor selection must use the (key, offset) tie-breaker.
        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            simplex.push_vertex_key(v_a);
            simplex
                .set_periodic_vertex_offsets(vec![[5, 0], [9, 0], [8, 0], [1, 0]])
                .unwrap();
        }

        let simplex = tds.simplex(simplex_key).unwrap();
        let identities =
            Tds::<(), (), 2>::facet_vertex_identities_in_simplex_order(simplex, 2).unwrap();
        assert_eq!(identities.len(), 3);

        let mut offsets_for_a: Vec<[i16; 2]> = identities
            .iter()
            .filter_map(|(vkey, offset)| (*vkey == v_a).then_some(*offset))
            .collect();
        offsets_for_a.sort_unstable();
        assert_eq!(offsets_for_a, vec![[0, 0], [4, 0]]);

        let b_offset = identities
            .iter()
            .find_map(|(vkey, offset)| (*vkey == v_b).then_some(*offset))
            .expect("identity list should include v_b");
        assert_eq!(b_offset, [8, 0]);
    }

    #[test]
    fn test_facet_permutation_parity_derives_coherent_and_incoherent_cases() {
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

        // Shared edge for facet_idx=2 is (v_a, v_b).
        let simplex: Simplex<(), 2> =
            Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap();

        // Coherent case: neighbor lists the shared edge in opposite order.
        let coherent_neighbor: Simplex<(), 2> =
            Simplex::try_new_with_data(vec![v_b, v_a, v_d], None).unwrap();
        let coherent_mirror_idx = simplex.mirror_facet_index(2, &coherent_neighbor).unwrap();
        let (currently_coherent, observed_odd_permutation, expected_odd_permutation) =
            Tds::<(), (), 2>::facet_permutation_parity(
                &simplex,
                2,
                &coherent_neighbor,
                coherent_mirror_idx,
            )
            .unwrap();
        assert!(currently_coherent);
        assert!(observed_odd_permutation);
        assert!(expected_odd_permutation);

        // Incoherent case: neighbor lists the shared edge in the same order.
        let incoherent_neighbor: Simplex<(), 2> =
            Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap();
        let incoherent_mirror_idx = simplex.mirror_facet_index(2, &incoherent_neighbor).unwrap();
        let (currently_coherent, observed_odd_permutation, expected_odd_permutation) =
            Tds::<(), (), 2>::facet_permutation_parity(
                &simplex,
                2,
                &incoherent_neighbor,
                incoherent_mirror_idx,
            )
            .unwrap();
        assert!(!currently_coherent);
        assert!(!observed_odd_permutation);
        assert!(expected_odd_permutation);
    }

    #[test]
    fn test_periodic_facet_parity_is_invariant_under_common_translation() {
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

        let mut simplex: Simplex<(), 2> =
            Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap();
        simplex
            .set_periodic_vertex_offsets(vec![[0, 0], [0, 0], [0, 0]])
            .unwrap();
        let mut translated_neighbor: Simplex<(), 2> =
            Simplex::try_new_with_data(vec![v_b, v_a, v_d], None).unwrap();
        translated_neighbor
            .set_periodic_vertex_offsets(vec![[3, -2], [3, -2], [4, -2]])
            .unwrap();

        let (coherent, observed_odd, expected_odd) =
            Tds::<(), (), 2>::facet_permutation_parity(&simplex, 2, &translated_neighbor, 2)
                .unwrap();
        assert!(coherent);
        assert!(observed_odd);
        assert!(expected_odd);
    }

    #[test]
    fn test_orientation_validation_rejects_incoherent_periodic_pair() {
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

        let mut simplex = Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap();
        simplex
            .set_periodic_vertex_offsets(vec![[0, 0], [0, 0], [0, 0]])
            .unwrap();
        let simplex_key = tds.insert_simplex_with_mapping(simplex).unwrap();

        let mut neighbor = Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap();
        neighbor
            .set_periodic_vertex_offsets(vec![[3, -2], [3, -2], [4, -2]])
            .unwrap();
        let neighbor_key = tds.insert_simplex_with_mapping(neighbor).unwrap();

        tds.assign_neighbors().unwrap();
        let error = tds.validate_coherent_orientation().unwrap_err();
        assert_matches!(
            error,
            TdsError::OrientationViolation {
                simplex1_key,
                simplex2_key,
                observed_odd_permutation: false,
                expected_odd_permutation: true,
                ..
            } if [simplex1_key, simplex2_key].contains(&simplex_key)
                && [simplex1_key, simplex2_key].contains(&neighbor_key)
        );
    }

    #[test]
    fn test_facet_permutation_parity_smoke_4d() {
        let mut tds: Tds<(), (), 4> = Tds::empty();
        let v_a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0, 0.0]).unwrap())
            .unwrap();
        let v_e = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0, 1.0]).unwrap())
            .unwrap();
        let v_f = tds
            .insert_vertex_with_mapping(vertex!([1.0, 1.0, 1.0, 1.0]).unwrap())
            .unwrap();

        // Shared 3-face for facet_idx=4 is [v_a, v_b, v_c, v_d].
        let simplex: Simplex<(), 4> =
            Simplex::try_new_with_data(vec![v_a, v_b, v_c, v_d, v_e], None).unwrap();

        // Coherent case: odd permutation of the shared face.
        let coherent_neighbor: Simplex<(), 4> =
            Simplex::try_new_with_data(vec![v_b, v_a, v_c, v_d, v_f], None).unwrap();
        let coherent_mirror_idx = simplex.mirror_facet_index(4, &coherent_neighbor).unwrap();
        let (currently_coherent, observed_odd_permutation, expected_odd_permutation) =
            Tds::<(), (), 4>::facet_permutation_parity(
                &simplex,
                4,
                &coherent_neighbor,
                coherent_mirror_idx,
            )
            .unwrap();
        assert!(currently_coherent);
        assert!(observed_odd_permutation);
        assert!(expected_odd_permutation);

        // Incoherent case: identity ordering of the shared face.
        let incoherent_neighbor: Simplex<(), 4> =
            Simplex::try_new_with_data(vec![v_a, v_b, v_c, v_d, v_f], None).unwrap();
        let incoherent_mirror_idx = simplex.mirror_facet_index(4, &incoherent_neighbor).unwrap();
        let (currently_coherent, observed_odd_permutation, expected_odd_permutation) =
            Tds::<(), (), 4>::facet_permutation_parity(
                &simplex,
                4,
                &incoherent_neighbor,
                incoherent_mirror_idx,
            )
            .unwrap();
        assert!(!currently_coherent);
        assert!(!observed_odd_permutation);
        assert!(expected_odd_permutation);
    }

    #[test]
    fn test_build_simplex_vertex_sets_success() {
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
                Simplex::try_new_with_data(vec![v_a, v_c, v_d], None).unwrap(),
            )
            .unwrap();

        let map = tds.build_simplex_vertex_sets().unwrap();
        assert_eq!(map.len(), 2);

        let expected1: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();
        let expected2: VertexKeySet = [v_a, v_c, v_d].into_iter().collect();

        assert_eq!(map.get(&simplex1), Some(&expected1));
        assert_eq!(map.get(&simplex2), Some(&expected2));
    }

    #[test]
    fn test_build_simplex_vertex_sets_errors_on_missing_vertex_key() {
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

        let simplex = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();

        // Corrupt the simplex by inserting a vertex key that doesn't exist in the TDS.
        let invalid_vkey = VertexKey::from(KeyData::from_ffi(u64::MAX));
        tds.simplex_mut(simplex)
            .unwrap()
            .push_vertex_key(invalid_vkey);

        let err = tds.build_simplex_vertex_sets().unwrap_err();
        assert_matches!(err, TdsError::VertexNotFound { .. });
    }

    #[test]
    fn test_validate_neighbor_topology_rejects_wrong_length() {
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

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();

        let err = tds
            .validate_neighbor_topology(simplex_key, &[None, None])
            .unwrap_err();
        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::LengthMismatch {
                    actual: 2,
                    expected: 3,
                    ..
                },
            }
        );
    }

    #[test]
    fn test_validate_shared_facet_count_ok_for_true_neighbors() {
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

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
            )
            .unwrap();

        let simplex1 = tds.simplex(simplex1_key).unwrap();
        let simplex2 = tds.simplex(simplex2_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();
        let neighbor_vertices: VertexKeySet = [v_a, v_b, v_d].into_iter().collect();

        assert!(
            Tds::<(), (), 2>::validate_shared_facet_count(
                simplex1,
                simplex2,
                &this_vertices,
                &neighbor_vertices
            )
            .is_ok()
        );
    }

    #[test]
    fn test_validate_shared_facet_count_errors_for_non_neighbors() {
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

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex_far_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_d, v_e], None).unwrap(),
            )
            .unwrap();

        let simplex1 = tds.simplex(simplex1_key).unwrap();
        let simplex_far = tds.simplex(simplex_far_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();
        let far_vertices: VertexKeySet = [v_a, v_d, v_e].into_iter().collect();

        let simplex1_uuid = simplex1.uuid();
        let simplex_far_uuid = simplex_far.uuid();

        let err = Tds::<(), (), 2>::validate_shared_facet_count(
            simplex1,
            simplex_far,
            &this_vertices,
            &far_vertices,
        )
        .unwrap_err();

        assert_matches!(
            err,
            TdsError::NotNeighbors { simplex1: c1, simplex2: c2 }
                if c1 == simplex1_uuid && c2 == simplex_far_uuid
        );
    }

    #[test]
    fn test_expected_mirror_facet_index_returns_unique_vertex_index() {
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

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        // Put the unique vertex at index 0 to ensure we test the returned index.
        let simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_d, v_a, v_b], None).unwrap(),
            )
            .unwrap();

        let simplex1 = tds.simplex(simplex1_key).unwrap();
        let simplex2 = tds.simplex(simplex2_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();

        let idx = Tds::<(), (), 2>::expected_mirror_facet_index(simplex1, simplex2, &this_vertices)
            .unwrap();
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_expected_mirror_facet_index_errors_when_ambiguous() {
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

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        // Shares only v_a -> differs by 2 vertices -> ambiguous mirror facet.
        let simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_d, v_e], None).unwrap(),
            )
            .unwrap();

        let simplex1 = tds.simplex(simplex1_key).unwrap();
        let simplex2 = tds.simplex(simplex2_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();

        let err = Tds::<(), (), 2>::expected_mirror_facet_index(simplex1, simplex2, &this_vertices)
            .unwrap_err();

        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::MirrorFacetAmbiguous { .. },
            }
        );
    }

    #[test]
    fn test_expected_mirror_facet_index_errors_when_duplicate_simplices() {
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

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        // Duplicate by vertices (different UUID) -> no unique vertex to identify mirror facet.
        let simplex2_key = tds
            .insert_simplex_bypassing_topology_checks_for_test(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();

        let simplex1 = tds.simplex(simplex1_key).unwrap();
        let simplex2 = tds.simplex(simplex2_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();

        let err = Tds::<(), (), 2>::expected_mirror_facet_index(simplex1, simplex2, &this_vertices)
            .unwrap_err();

        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::MirrorFacetDuplicateSimplices { .. },
            }
        );
    }

    #[test]
    fn test_verified_mirror_facet_index_ok() {
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

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
            )
            .unwrap();

        let simplex1 = tds.simplex(simplex1_key).unwrap();
        let simplex2 = tds.simplex(simplex2_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();

        // Shared edge is (v_a, v_b). In simplex1, that's opposite vertex index 2 (v_c).
        let mirror_idx =
            Tds::<(), (), 2>::verified_mirror_facet_index(simplex1, 2, simplex2, &this_vertices)
                .unwrap();
        assert_eq!(mirror_idx, 2);
    }

    #[test]
    fn test_verified_mirror_facet_index_errors_when_no_shared_facet() {
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

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
            )
            .unwrap();

        let simplex1 = tds.simplex(simplex1_key).unwrap();
        let simplex2 = tds.simplex(simplex2_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();

        // facet_idx=0 corresponds to edge (v_b, v_c) in simplex1, which is not shared with simplex2.
        let err =
            Tds::<(), (), 2>::verified_mirror_facet_index(simplex1, 0, simplex2, &this_vertices)
                .unwrap_err();

        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::MirrorFacetMissing { .. },
            }
        );
    }

    #[test]
    fn test_verified_mirror_facet_index_errors_on_mismatch_with_cross_check() {
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

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
            )
            .unwrap();

        let simplex1 = tds.simplex(simplex1_key).unwrap();
        let simplex2 = tds.simplex(simplex2_key).unwrap();

        // Intentionally WRONG vertex set (includes v_d, excludes v_b) to force the mismatch branch.
        // This is a unit-level test of the helper's defensive cross-check behavior.
        let this_vertices_wrong: VertexKeySet = [v_a, v_c, v_d].into_iter().collect();

        let err = Tds::<(), (), 2>::verified_mirror_facet_index(
            simplex1,
            2,
            simplex2,
            &this_vertices_wrong,
        )
        .unwrap_err();

        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::MirrorFacetIndexMismatch { .. },
            }
        );
    }

    #[test]
    fn test_validate_neighbors_errors_on_mirror_facet_index_mismatch() {
        // This test exercises the same "mirror facet index mismatch" defensive branch, but via the
        // neighbor-validation loop used by `validate_neighbors_with_precomputed_vertex_sets()`.
        //
        // The mismatch is only reachable if the precomputed per-simplex vertex-set map is inconsistent
        // with the simplex's actual vertex buffer (e.g., a bug/corruption in the precompute step). To
        // simulate that scenario deterministically, we build the map normally and then corrupt the
        // entry for one simplex before running the validation loop.
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

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![origin_key, east_key, north_key], None).unwrap(),
            )
            .unwrap();
        let _simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![origin_key, east_key, diagonal_key], None).unwrap(),
            )
            .unwrap();

        tds.assign_neighbors().unwrap();

        let mut simplex_vertices = tds.build_simplex_vertex_sets().unwrap();

        // Corrupt the vertex-set entry for simplex1 so it no longer matches the actual simplex's vertices.
        // (Drop `east_key`, add `diagonal_key`.)
        let corrupted_simplex1_vertices: VertexKeySet =
            [origin_key, north_key, diagonal_key].into_iter().collect();
        simplex_vertices.insert(simplex1_key, corrupted_simplex1_vertices);

        let err = tds
            .validate_neighbors_with_precomputed_vertex_sets(&simplex_vertices)
            .unwrap_err();

        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::MirrorFacetIndexMismatch { .. },
            }
        );
    }

    #[test]
    fn test_validate_shared_facet_vertices_ok() {
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

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
            )
            .unwrap();

        let simplex1 = tds.simplex(simplex1_key).unwrap();
        let simplex2 = tds.simplex(simplex2_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();
        let neighbor_vertices: VertexKeySet = [v_a, v_b, v_d].into_iter().collect();

        assert!(
            Tds::<(), (), 2>::validate_shared_facet_vertices(
                simplex1,
                2, // opposite v_c => shared edge {v_a,v_b}
                simplex2,
                2, // opposite v_d => shared edge {v_a,v_b}
                &this_vertices,
                &neighbor_vertices,
            )
            .is_ok()
        );
    }

    #[test]
    fn test_validate_shared_facet_vertices_errors_when_mirror_index_wrong() {
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

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
            )
            .unwrap();

        let simplex1 = tds.simplex(simplex1_key).unwrap();
        let simplex2 = tds.simplex(simplex2_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();
        let neighbor_vertices: VertexKeySet = [v_a, v_b, v_d].into_iter().collect();

        // mirror_idx=0 is intentionally wrong here; it treats vertex v_a as the "opposite"
        // which makes v_d incorrectly part of the "shared facet".
        let err = Tds::<(), (), 2>::validate_shared_facet_vertices(
            simplex1,
            2, // correct for simplex1
            simplex2,
            0, // intentionally wrong for simplex2
            &this_vertices,
            &neighbor_vertices,
        )
        .unwrap_err();

        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::SharedFacetMissingVertex { .. },
            }
        );
    }

    #[test]
    fn test_validate_mutual_neighbor_back_reference_ok() {
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

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
            )
            .unwrap();

        // Build neighbor pointers so mutual back-references exist.
        tds.assign_neighbors().unwrap();

        let simplex1 = tds.simplex(simplex1_key).unwrap();
        let simplex2 = tds.simplex(simplex2_key).unwrap();

        assert!(
            Tds::<(), (), 2>::validate_mutual_neighbor_back_reference(
                simplex1_key,
                simplex1,
                2, // opposite v_c => shared edge {v_a,v_b}
                simplex2_key,
                simplex2,
                2, // opposite v_d => shared edge {v_a,v_b}
            )
            .is_ok()
        );
    }

    #[test]
    fn test_validate_mutual_neighbor_back_reference_errors_when_neighbor_has_no_neighbors() {
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

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
            )
            .unwrap();

        // NOTE: We intentionally do NOT call assign_neighbors(), so neighbor_simplex.neighbors is None.
        let simplex1 = tds.simplex(simplex1_key).unwrap();
        let simplex2 = tds.simplex(simplex2_key).unwrap();

        let err = Tds::<(), (), 2>::validate_mutual_neighbor_back_reference(
            simplex1_key,
            simplex1,
            2,
            simplex2_key,
            simplex2,
            2,
        )
        .unwrap_err();

        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::BackReferenceMismatch { observed: None, .. },
            }
        );
    }

    #[test]
    fn test_validate_mutual_neighbor_back_reference_errors_when_back_reference_missing() {
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

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
            )
            .unwrap();

        // Build neighbor pointers first, then deliberately corrupt the back-reference.
        tds.assign_neighbors().unwrap();

        {
            let simplex2_mut = tds.simplex_mut(simplex2_key).unwrap();
            let neighbors = simplex2_mut
                .neighbor_slots_mut()
                .expect("simplex2 should have neighbors after assign_neighbors()");
            // For (v_a, v_b, v_d), the shared edge with simplex1 is opposite v_d => index 2.
            neighbors[2] = NeighborSlot::Boundary;
        }

        let simplex1 = tds.simplex(simplex1_key).unwrap();
        let simplex2 = tds.simplex(simplex2_key).unwrap();

        let err = Tds::<(), (), 2>::validate_mutual_neighbor_back_reference(
            simplex1_key,
            simplex1,
            2,
            simplex2_key,
            simplex2,
            2,
        )
        .unwrap_err();

        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::BackReferenceMismatch { observed: None, .. },
            }
        );
    }

    #[test]
    fn test_orientation_validation_rejects_non_periodic_self_neighbor() {
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

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();

        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            simplex
                .set_neighbors_from_keys(vec![Some(simplex_key), None, None])
                .unwrap();
        }

        let simplex = tds.simplex(simplex_key).unwrap();
        assert!(!Tds::<(), (), 2>::allows_periodic_self_neighbor(simplex));

        let err = tds.validate_coherent_orientation().unwrap_err();
        assert_matches!(
            err,
            TdsError::OrientationViolation {
                simplex1_key,
                simplex2_key,
                ..
            } if simplex1_key == simplex_key && simplex2_key == simplex_key
        );
        assert!(!tds.is_coherently_oriented());

        let err = tds.normalize_coherent_orientation().unwrap_err();
        assert_matches!(
            err,
            TdsError::InconsistentDataStructure { message }
                if message.contains("Contradictory orientation constraints")
        );
    }

    #[test]
    fn test_orientation_validation_allows_periodic_self_neighbor() {
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

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();

        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            simplex
                .set_neighbors_from_keys(vec![Some(simplex_key), None, None])
                .unwrap();
            simplex
                .set_periodic_vertex_offsets(vec![[0, 0], [0, 0], [0, 0]])
                .unwrap();
        }

        let simplex = tds.simplex(simplex_key).unwrap();
        assert!(Tds::<(), (), 2>::allows_periodic_self_neighbor(simplex));
        assert!(tds.validate_coherent_orientation().is_ok());
        assert!(tds.is_coherently_oriented());
        assert!(tds.normalize_coherent_orientation().is_ok());
    }

    #[test]
    fn test_orientation_validation_rejects_one_sided_periodic_neighbor() {
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

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
            )
            .unwrap();

        {
            let simplex1 = tds.simplex_mut(simplex1_key).unwrap();
            simplex1
                .set_neighbors_from_keys(vec![None, None, Some(simplex2_key)])
                .unwrap();
            simplex1
                .set_periodic_vertex_offsets(vec![[0, 0], [0, 0], [0, 0]])
                .unwrap();
        }
        {
            let simplex2 = tds.simplex_mut(simplex2_key).unwrap();
            simplex2
                .set_neighbors_from_keys(vec![None, None, None])
                .unwrap();
            simplex2
                .set_periodic_vertex_offsets(vec![[0, 0], [0, 0], [0, 0]])
                .unwrap();
        }

        let err = tds.validate_coherent_orientation().unwrap_err();
        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::BackReferenceMismatch { observed: None, .. },
            }
        );
        assert!(!tds.is_coherently_oriented());

        let err = tds
            .validate_coherent_orientation_for_simplices(&[simplex1_key])
            .unwrap_err();
        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::BackReferenceMismatch { observed: None, .. },
            }
        );
    }

    #[test]
    fn test_boundary_facet_validation_rejects_unassigned_neighbor_slot() {
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
        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        tds.assign_neighbors().unwrap();
        let facet_to_simplices = tds.build_facet_to_simplices_map().unwrap();

        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            simplex.ensure_neighbors_buffer_mut()[1] = NeighborSlot::Unassigned;
        }

        let err = tds
            .validate_neighbor_pointers_match_facet_to_simplices_map(&facet_to_simplices)
            .unwrap_err();

        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::UnassignedNeighborSlot {
                    simplex_key: key,
                    facet_index: 1,
                    ..
                },
            } if key == simplex_key
        );
    }

    #[test]
    fn test_boundary_facet_validation_rejects_non_periodic_self_neighbor() {
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
        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        tds.assign_neighbors().unwrap();
        let facet_to_simplices = tds.build_facet_to_simplices_map().unwrap();

        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            simplex.ensure_neighbors_buffer_mut()[2] = NeighborSlot::Neighbor(simplex_key);
        }

        let err = tds
            .validate_neighbor_pointers_match_facet_to_simplices_map(&facet_to_simplices)
            .unwrap_err();

        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::BoundaryFacetHasNonPeriodicSelfNeighbor {
                    simplex_key: key,
                    facet_index: 2,
                    ..
                },
            } if key == simplex_key
        );
    }

    #[test]
    fn test_orientation_validation_rejects_one_way_neighbor_pointer() {
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

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_b, v_a, v_d], None).unwrap(),
            )
            .unwrap();
        tds.assign_neighbors().unwrap();
        assert!(tds.validate_coherent_orientation().is_ok());

        let mirror_idx = {
            let simplex1 = tds.simplex(simplex1_key).unwrap();
            let mut neighbors = simplex1
                .neighbors()
                .expect("simplex1 should have neighbors after assign_neighbors()");
            let facet_idx = neighbors
                .position(|n| n == Some(simplex2_key))
                .expect("simplex1 should reference simplex2");
            let simplex2 = tds.simplex(simplex2_key).unwrap();
            simplex1
                .mirror_facet_index(facet_idx, simplex2)
                .expect("adjacent simplices should have a mirror facet")
        };

        {
            let simplex2 = tds.simplex_mut(simplex2_key).unwrap();
            let neighbors = simplex2
                .neighbor_slots_mut()
                .expect("simplex2 should have neighbors after assign_neighbors()");
            neighbors[mirror_idx] = NeighborSlot::Boundary;
        }

        let err = tds.validate_coherent_orientation().unwrap_err();
        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::BackReferenceMismatch { observed: None, .. },
            }
        );
        assert!(!tds.is_coherently_oriented());
    }

    #[test]
    fn test_local_orientation_validation_checks_neighbors_outside_scope() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([2.0, 2.0]).unwrap())
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(vertex!([3.0, 3.0]).unwrap())
            .unwrap();

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v3], None).unwrap(),
            )
            .unwrap();
        tds.assign_neighbors().unwrap();

        let err = tds
            .validate_coherent_orientation_for_simplices(&[simplex1_key])
            .unwrap_err();
        assert_matches!(err, TdsError::OrientationViolation { .. });
    }

    #[test]
    fn test_local_orientation_validation_errors_on_missing_scope_simplex() {
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

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        assert_eq!(tds.remove_simplices_by_keys(&[simplex_key]).unwrap(), 1);

        let err = tds
            .validate_coherent_orientation_for_simplices(&[simplex_key])
            .unwrap_err();
        assert_matches!(
            err,
            TdsError::SimplexNotFound {
                simplex_key: missing_key,
                ..
            } if missing_key == simplex_key
        );
    }

    #[test]
    fn test_build_facet_to_simplices_map_errors_on_u8_facet_index_overflow() {
        let tds: Tds<(), (), 256> = Tds::empty();

        let err = tds.build_facet_to_simplices_map().unwrap_err();
        assert_matches!(
            err,
            TdsError::DimensionMismatch {
                expected: 255,
                actual: 256,
                ref context,
            } if context == "facet indices must fit in u8"
        );
    }

    #[test]
    fn test_validate_vertex_and_simplex_mappings_detect_inconsistencies() {
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

        // Start from a consistent state.
        assert!(tds.validate_vertex_mappings().is_ok());
        assert!(tds.validate_simplex_mappings().is_ok());

        // Break vertex mapping: remove one uuid entry (len mismatch).
        let uuid_a = tds.vertex(a).unwrap().uuid();
        tds.uuid_to_vertex_key.remove(&uuid_a);
        assert_matches!(
            tds.validate_vertex_mappings(),
            Err(TdsError::MappingInconsistency {
                entity: EntityKind::Vertex,
                ..
            })
        );

        // Restore length but make the UUID map point at the wrong key.
        tds.uuid_to_vertex_key.insert(uuid_a, b);
        assert_matches!(
            tds.validate_vertex_mappings(),
            Err(TdsError::MappingInconsistency {
                entity: EntityKind::Vertex,
                ..
            })
        );

        // Break simplex mapping similarly.
        let uuid_simplex = tds.simplex(simplex_key).unwrap().uuid();
        tds.uuid_to_simplex_key.remove(&uuid_simplex);
        assert_matches!(
            tds.validate_simplex_mappings(),
            Err(TdsError::MappingInconsistency {
                entity: EntityKind::Simplex,
                ..
            })
        );
    }

    #[test]
    fn test_validate_simplex_vertex_keys_detects_missing_vertices() {
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

        let invalid_vkey = VertexKey::from(KeyData::from_ffi(u64::MAX));
        tds.simplex_mut(simplex_key)
            .unwrap()
            .push_vertex_key(invalid_vkey);

        let err = tds.validate_simplex_vertex_keys().unwrap_err();
        assert_matches!(err, TdsError::VertexNotFound { .. });

        // Now wired into structural validation: is_valid() should fail early with the
        // more precise "missing vertex key" diagnostic.
        let err = tds.is_valid().unwrap_err();
        assert_matches!(err, TdsError::VertexNotFound { .. });
    }

    #[test]
    fn test_is_connected_returns_false_for_isolated_simplices() {
        // Build a TDS with two triangles that have no neighbor wiring between them.
        // Since neither simplex's `neighbors` field is populated, BFS from either simplex
        // cannot reach the other → is_connected() must return false.
        let mut tds: Tds<(), (), 2> = Tds::empty();

        // Component A
        let a0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let a1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let a2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();

        // Component B (far away, no shared vertices)
        let b0 = tds
            .insert_vertex_with_mapping(vertex!([10.0, 0.0]).unwrap())
            .unwrap();
        let b1 = tds
            .insert_vertex_with_mapping(vertex!([11.0, 0.0]).unwrap())
            .unwrap();
        let b2 = tds
            .insert_vertex_with_mapping(vertex!([10.0, 1.0]).unwrap())
            .unwrap();

        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![a0, a1, a2], None).unwrap(),
        )
        .unwrap();
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![b0, b1, b2], None).unwrap(),
        )
        .unwrap();

        // Two simplices with neighbors = None: BFS from the first simplex finds no edges
        // and can never visit the second.
        assert!(
            !tds.is_connected(),
            "TDS with two isolated simplices (no neighbor wiring) must not be connected"
        );
    }

    #[test]
    fn test_validate_vertex_incidence_detects_dangling_incident_simplex() {
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

        let ck = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        tds.assign_incident_simplices().unwrap();

        // Remove the simplex but leave the vertex's incident_simplex pointer dangling
        tds.simplices.remove(ck);

        let err = tds.validate_vertex_incidence().unwrap_err();
        assert_matches!(err, TdsError::SimplexNotFound { .. });
    }

    #[test]
    fn test_validate_vertex_to_simplices_index_detects_missing_reverse_entry() {
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

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        tds.clear_vertex_incidence_for_test(v0);

        let err = tds.validate_vertex_to_simplices_index().unwrap_err();
        assert_matches!(
            &err,
            TdsError::InconsistentDataStructure { message }
                if message.contains("2 simplex incidences")
                    && message.contains("expected 3")
        );
        let _ = simplex_key;

        let report = tds.validation_report().unwrap_err();
        assert!(
            report
                .violations
                .iter()
                .any(|violation| violation.kind == InvariantKind::VertexToSimplicesIndex)
        );
    }

    #[test]
    fn simplex_coordinate_uniqueness_is_level_one_not_level_two() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        // Two distinct vertex keys with identical coordinates
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 1.0]).unwrap())
            .unwrap();

        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();

        tds.is_valid()
            .expect("Level 2 structural validation must be coordinate-independent");
        tds.structure_report()
            .expect("Level 2 diagnostics must be coordinate-independent");

        let err = tds
            .validate()
            .expect_err("cumulative Levels 1-2 validation must reject duplicate coordinates");
        assert_matches!(
            &err,
            TdsError::DuplicateCoordinatesInSimplex { .. },
            "Expected DuplicateCoordinatesInSimplex, got {err:?}"
        );

        let report = tds
            .validation_report()
            .expect_err("the cumulative report must include the Level 1 coordinate violation");
        assert!(
            report
                .violations
                .iter()
                .any(|violation| violation.kind == InvariantKind::SimplexCoordinateUniqueness)
        );
    }

    #[test]
    fn test_validate_facet_sharing_rejects_triple_shared_facet() {
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
            .insert_vertex_with_mapping(vertex!([0.5, 0.5]).unwrap())
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(vertex!([0.3, 0.3]).unwrap())
            .unwrap();

        // Three simplices sharing the v0-v1 edge (facet):
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v3], None).unwrap(),
        )
        .unwrap();
        tds.insert_simplex_bypassing_topology_checks_for_test(
            Simplex::try_new_with_data(vec![v0, v1, v4], None).unwrap(),
        )
        .unwrap();

        let err = tds.validate_facet_sharing().unwrap_err();
        let message = err.to_string();
        assert_matches!(
            &err,
            TdsError::FacetSharingViolation {
                existing_incident_count: 2,
                attempted_incident_count: 3,
                max_incident_count: 2,
                candidate_facet_index: 2,
                ..
            },
            "Expected over-shared facet error, got {err:?}"
        );
        assert!(message.contains("exceeds incident-simplex limit"));
        assert!(!message.contains("inserting candidate simplex"));

        let err = tds.is_valid().unwrap_err();
        assert_matches!(
            &err,
            TdsError::FacetSharingViolation { .. },
            "Expected is_valid to surface facet-sharing violation, got {err:?}"
        );

        let report = tds.validation_report().unwrap_err();
        let facet_violation = report
            .violations
            .iter()
            .find(|violation| violation.kind == InvariantKind::FacetSharing)
            .expect("validation_report should include the facet-sharing violation");
        assert_matches!(
            &facet_violation.error,
            InvariantError::Tds(TdsError::FacetSharingViolation { .. }),
            "Expected validation_report to preserve structured facet-sharing error, got {:?}",
            facet_violation.error
        );
    }

    #[test]
    fn test_validate_no_duplicate_simplices_detects_dupes() {
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
        tds.insert_simplex_bypassing_topology_checks_for_test(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();

        let err = tds.validate_no_duplicate_simplices().unwrap_err();
        assert_matches!(err, TdsError::DuplicateSimplices { .. });
    }

    #[test]
    fn test_tds_is_valid_passes_for_valid_simplex() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::builder(&verts).build().unwrap();
        assert!(dt.tds().is_valid().is_ok());
        assert!(dt.tds().validate().is_ok());
    }

    #[test]
    fn test_build_facet_to_simplices_map_basic() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::builder(&verts).build().unwrap();
        let tds = dt.tds();

        let facet_map = tds.build_facet_to_simplices_map().unwrap();
        // A single tetrahedron in 3D has 4 facets, all boundary (degree 1)
        assert_eq!(facet_map.len(), 4);
        for handles in facet_map.values() {
            assert_eq!(handles.len(), 1);
        }
    }

    #[test]
    fn test_validation_report_accumulates_violations() {
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

        // Create a simplex, then corrupt the UUID mapping
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();
        // Corrupt: add a stray UUID mapping pointing to a non-existent key
        tds.uuid_to_simplex_key
            .insert(Uuid::new_v4(), SimplexKey::from(KeyData::from_ffi(0xBAD)));

        let report = tds.validation_report().unwrap_err();
        assert!(!report.is_empty());
        assert!(
            report
                .violations
                .iter()
                .any(|v| v.kind == InvariantKind::SimplexMappings),
            "Expected SimplexMappings violation"
        );
    }

    #[test]
    fn test_validate_vertex_incidence_detects_inconsistent_incident_simplex() {
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
        tds.assign_incident_simplices().unwrap();

        // Create a second simplex that does NOT contain v0, then point v0 at it.
        let v3 = tds
            .insert_vertex_with_mapping(vertex!([2.0, 2.0]).unwrap())
            .unwrap();
        let ck2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v1, v2, v3], None).unwrap(),
            )
            .unwrap();
        tds.vertex_mut(v0).unwrap().set_incident_simplex(Some(ck2));

        let err = tds.validate_vertex_incidence().unwrap_err();
        assert_matches!(err, TdsError::InconsistentDataStructure { .. });
    }

    #[test]
    fn test_validate_vertex_incidence_detects_dangling_simplex_key() {
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
        tds.assign_incident_simplices().unwrap();

        // Point v0 at a non-existent simplex key.
        let dangling = SimplexKey::from(KeyData::from_ffi(0xDEAD));
        tds.vertex_mut(v0)
            .unwrap()
            .set_incident_simplex(Some(dangling));

        let err = tds.validate_vertex_incidence().unwrap_err();
        assert_matches!(err, TdsError::SimplexNotFound { .. });
    }

    #[test]
    fn test_validate_simplex_coordinate_uniqueness_passes_for_distinct_coords() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::builder(&verts).build().unwrap();
        let tds = dt.tds();
        assert!(tds.validate_simplex_coordinate_uniqueness().is_ok());
    }
}
