//! Canonical vertex-to-simplices incidence index for TDS storage.

use crate::core::collections::{
    MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer, VertexToSimplicesMap, fast_hash_map_with_capacity,
};
use crate::core::tds::errors::TdsError;
use crate::core::tds::{SimplexKey, VertexKey};

/// Invariant-bearing owner of the exact vertex → incident simplices relation.
///
/// Isolated vertices are represented by present-but-empty entries. All mutation
/// methods are relation-specific so callers cannot accidentally create entries
/// for missing vertices or remove vertices that still own simplex incidence.
#[derive(Clone, Debug, Default)]
pub(crate) struct VertexIncidenceIndex {
    map: VertexToSimplicesMap,
}

/// Rollback record for an exact simplex-incidence removal.
///
/// A successful removal may use unordered buffer operations for speed, but a
/// failed multi-step mutation must be able to restore every touched incidence
/// buffer exactly. This record stores the removed simplex and the per-vertex
/// positions needed to undo [`VertexIncidenceIndex::remove_simplex`].
#[derive(Clone, Debug)]
pub(in crate::core::tds) struct SimplexIncidenceRemoval {
    simplex_key: SimplexKey,
    removed_vertices: SmallBuffer<RemovedVertexIncidence, MAX_PRACTICAL_DIMENSION_SIZE>,
}

/// Position of one removed simplex key inside one vertex incidence buffer.
///
/// `remove_simplex` uses `swap_remove` on the success path, so rollback needs
/// both the vertex and original position to restore the displaced tail element.
#[derive(Clone, Copy, Debug)]
struct RemovedVertexIncidence {
    vertex_key: VertexKey,
    position: usize,
}

impl VertexIncidenceIndex {
    /// Creates an empty incidence index with capacity for `vertex_capacity` vertices.
    #[must_use]
    pub(in crate::core::tds) fn with_vertex_capacity(vertex_capacity: usize) -> Self {
        Self {
            map: fast_hash_map_with_capacity(vertex_capacity),
        }
    }

    /// Returns the compact backing map for validation and diagnostics.
    #[must_use]
    pub(in crate::core) const fn as_map(&self) -> &VertexToSimplicesMap {
        &self.map
    }

    /// Returns `true` when the index has no vertex entries.
    #[must_use]
    #[cfg(test)]
    pub(in crate::core) fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns whether `vertex_key` has an incidence entry.
    #[must_use]
    #[cfg(test)]
    pub(in crate::core) fn contains_vertex(&self, vertex_key: VertexKey) -> bool {
        self.map.contains_key(&vertex_key)
    }

    /// Returns every simplex key incident to `vertex_key`.
    ///
    /// The returned order is an implementation detail of the incidence buffers
    /// and is not part of the public adjacency-query contract.
    pub(in crate::core) fn simplex_keys(
        &self,
        vertex_key: VertexKey,
    ) -> impl Iterator<Item = SimplexKey> + '_ {
        self.map
            .get(&vertex_key)
            .into_iter()
            .flat_map(|simplices| simplices.iter().copied())
    }

    /// Returns the number of incident simplices for `vertex_key`.
    #[must_use]
    pub(in crate::core) fn number_of_simplices(&self, vertex_key: VertexKey) -> usize {
        let Some(incident_simplices) = self.map.get(&vertex_key) else {
            return 0;
        };
        incident_simplices.len()
    }

    /// Registers a newly inserted isolated vertex.
    ///
    /// # Errors
    ///
    /// Returns [`TdsError::InconsistentDataStructure`] if the vertex already has
    /// an incidence entry.
    pub(in crate::core::tds) fn insert_vertex(
        &mut self,
        vertex_key: VertexKey,
    ) -> Result<(), TdsError> {
        if self.map.contains_key(&vertex_key) {
            return Err(TdsError::InconsistentDataStructure {
                message: format!(
                    "Vertex-to-simplices index already has an entry for vertex {vertex_key:?}"
                ),
            });
        }

        self.map.insert(
            vertex_key,
            SmallBuffer::<SimplexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new(),
        );
        Ok(())
    }

    /// Removes an isolated vertex entry.
    ///
    /// # Errors
    ///
    /// Returns [`TdsError::VertexNotFound`] if the vertex has no index entry, or
    /// [`TdsError::InconsistentDataStructure`] if the vertex still has incident
    /// simplices.
    pub(in crate::core::tds) fn remove_isolated_vertex(
        &mut self,
        vertex_key: VertexKey,
    ) -> Result<(), TdsError> {
        let Some(incident_simplices) = self.map.get(&vertex_key) else {
            return Err(TdsError::VertexNotFound {
                vertex_key,
                context: "vertex-to-simplices index removal".to_string(),
            });
        };

        if !incident_simplices.is_empty() {
            return Err(TdsError::InconsistentDataStructure {
                message: format!(
                    "Cannot remove vertex {vertex_key:?} from incidence index while it still has {} incident simplices",
                    incident_simplices.len()
                ),
            });
        }

        self.map.remove(&vertex_key);
        Ok(())
    }

    /// Registers `simplex_key` under each of its vertices.
    ///
    /// # Errors
    ///
    /// Returns a typed error if a vertex is missing from the index, if
    /// `vertices` contains duplicate vertex keys, or if this simplex is already
    /// recorded for any listed vertex.
    pub(in crate::core::tds) fn insert_simplex(
        &mut self,
        simplex_key: SimplexKey,
        vertices: &[VertexKey],
    ) -> Result<(), TdsError> {
        let mut inserted_vertices =
            SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::with_capacity(vertices.len());

        for &vertex_key in vertices {
            let Some(incident_simplices) = self.map.get_mut(&vertex_key) else {
                self.rollback_inserted_simplex(simplex_key, &inserted_vertices);
                return Err(TdsError::VertexNotFound {
                    vertex_key,
                    context: format!(
                        "registering simplex {simplex_key:?} in vertex incidence index"
                    ),
                });
            };

            if incident_simplices.contains(&simplex_key) {
                self.rollback_inserted_simplex(simplex_key, &inserted_vertices);
                return Err(TdsError::InconsistentDataStructure {
                    message: format!(
                        "Vertex-to-simplices index already lists simplex {simplex_key:?} for vertex {vertex_key:?}"
                    ),
                });
            }

            incident_simplices.push(simplex_key);
            inserted_vertices.push(vertex_key);
        }

        Ok(())
    }

    /// Removes `simplex_key` from each of its vertices.
    ///
    /// On success, returns a rollback record that can restore the exact previous
    /// incidence buffers. Callers that are completing the mutation should drop
    /// the record; callers aborting a larger transaction should pass it to
    /// [`Self::rollback_removed_simplex`].
    ///
    /// Successful removal does not preserve the order of each vertex's incident
    /// simplex buffer. Query APIs therefore expose this relation as unordered.
    ///
    /// # Errors
    ///
    /// Returns a typed error if a vertex is missing from the index, if
    /// `vertices` contains duplicate vertex keys, or if this simplex is not
    /// currently recorded for any listed vertex. If an error occurs after some
    /// vertex incidence entries have been removed, those entries are rolled back
    /// before the error is returned.
    pub(in crate::core::tds) fn remove_simplex(
        &mut self,
        simplex_key: SimplexKey,
        vertices: &[VertexKey],
    ) -> Result<SimplexIncidenceRemoval, TdsError> {
        Self::validate_simplex_vertices_are_unique(simplex_key, vertices)?;
        let mut removal = SimplexIncidenceRemoval {
            simplex_key,
            removed_vertices:
                SmallBuffer::<RemovedVertexIncidence, MAX_PRACTICAL_DIMENSION_SIZE>::with_capacity(
                    vertices.len(),
                ),
        };

        for &vertex_key in vertices {
            let Some(incident_simplices) = self.map.get_mut(&vertex_key) else {
                self.rollback_removed_simplex(&removal);
                return Err(TdsError::VertexNotFound {
                    vertex_key,
                    context: format!(
                        "removing simplex {simplex_key:?} from vertex incidence index"
                    ),
                });
            };

            let Some(position) = incident_simplices
                .iter()
                .position(|candidate| *candidate == simplex_key)
            else {
                self.rollback_removed_simplex(&removal);
                return Err(TdsError::InconsistentDataStructure {
                    message: format!(
                        "Vertex-to-simplices index does not list simplex {simplex_key:?} for vertex {vertex_key:?}"
                    ),
                });
            };

            incident_simplices.swap_remove(position);
            removal.removed_vertices.push(RemovedVertexIncidence {
                vertex_key,
                position,
            });
        }

        Ok(removal)
    }

    /// Verifies that a simplex vertex list does not contain duplicate keys.
    ///
    /// Incidence mutation is defined over the simplex's vertex set. Rejecting
    /// duplicates here prevents a malformed caller from removing or inserting
    /// the same `(vertex, simplex)` relation twice during one operation.
    fn validate_simplex_vertices_are_unique(
        simplex_key: SimplexKey,
        vertices: &[VertexKey],
    ) -> Result<(), TdsError> {
        for (index, &vertex_key) in vertices.iter().enumerate() {
            if vertices[..index].contains(&vertex_key) {
                return Err(TdsError::InconsistentDataStructure {
                    message: format!(
                        "Simplex {simplex_key:?} has duplicate vertex {vertex_key:?} while updating vertex incidence index"
                    ),
                });
            }
        }

        Ok(())
    }

    /// Removes an already-inserted simplex from partially updated vertices.
    ///
    /// This is the insertion-side rollback path for [`Self::insert_simplex`].
    /// It restores the relation as a set; insertion rollback is only used before
    /// the simplex becomes externally visible, so buffer order is not observable.
    fn rollback_inserted_simplex(&mut self, simplex_key: SimplexKey, vertices: &[VertexKey]) {
        for &vertex_key in vertices {
            if let Some(incident_simplices) = self.map.get_mut(&vertex_key)
                && let Some(position) = incident_simplices
                    .iter()
                    .position(|candidate| *candidate == simplex_key)
            {
                incident_simplices.swap_remove(position);
            }
        }
    }

    /// Restores a previous [`Self::remove_simplex`] operation exactly.
    ///
    /// This rollback is order-preserving: each touched incidence buffer is
    /// restored to its state before the corresponding removal. `Tds` batch
    /// mutation uses this to keep failed removals from perturbing later
    /// adjacency queries or diagnostics.
    pub(in crate::core::tds) fn rollback_removed_simplex(
        &mut self,
        removal: &SimplexIncidenceRemoval,
    ) {
        for removed_vertex in removal.removed_vertices.iter().rev() {
            if let Some(incident_simplices) = self.map.get_mut(&removed_vertex.vertex_key) {
                if removed_vertex.position < incident_simplices.len() {
                    let displaced_simplex = incident_simplices[removed_vertex.position];
                    incident_simplices.push(displaced_simplex);
                    incident_simplices[removed_vertex.position] = removal.simplex_key;
                } else {
                    incident_simplices.push(removal.simplex_key);
                }
            }
        }
    }

    #[cfg(test)]
    pub(in crate::core::tds) fn clear_vertex_for_test(&mut self, vertex_key: VertexKey) {
        if let Some(incident_simplices) = self.map.get_mut(&vertex_key) {
            incident_simplices.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use slotmap::KeyData;

    fn vertex_key(raw: u64) -> VertexKey {
        VertexKey::from(KeyData::from_ffi(raw))
    }

    fn simplex_key(raw: u64) -> SimplexKey {
        SimplexKey::from(KeyData::from_ffi(raw))
    }

    #[test]
    fn insert_simplex_rejects_missing_vertex_entry() {
        let mut index = VertexIncidenceIndex::default();
        let err = index
            .insert_simplex(simplex_key(1), &[vertex_key(1)])
            .unwrap_err();
        assert!(matches!(err, TdsError::VertexNotFound { .. }));
    }

    #[test]
    fn insert_simplex_rolls_back_when_later_vertex_entry_is_missing() {
        let mut index = VertexIncidenceIndex::default();
        let existing = vertex_key(1);
        let missing = vertex_key(2);
        let simplex = simplex_key(1);
        index.insert_vertex(existing).unwrap();

        let err = index
            .insert_simplex(simplex, &[existing, missing])
            .unwrap_err();

        assert!(matches!(err, TdsError::VertexNotFound { .. }));
        assert_eq!(index.number_of_simplices(existing), 0);
    }

    #[test]
    fn insert_simplex_rolls_back_duplicate_vertex_entry() {
        let mut index = VertexIncidenceIndex::default();
        let vertex = vertex_key(1);
        let simplex = simplex_key(1);
        index.insert_vertex(vertex).unwrap();

        let err = index
            .insert_simplex(simplex, &[vertex, vertex])
            .unwrap_err();

        assert!(matches!(err, TdsError::InconsistentDataStructure { .. }));
        assert_eq!(index.number_of_simplices(vertex), 0);
    }

    #[test]
    fn remove_isolated_vertex_rejects_non_isolated_entry() {
        let mut index = VertexIncidenceIndex::default();
        let vertex = vertex_key(1);
        let simplex = simplex_key(1);
        index.insert_vertex(vertex).unwrap();
        index.insert_simplex(simplex, &[vertex]).unwrap();

        let err = index.remove_isolated_vertex(vertex).unwrap_err();
        assert!(matches!(err, TdsError::InconsistentDataStructure { .. }));
    }

    #[test]
    fn remove_simplex_rolls_back_order_exactly_when_later_vertex_is_missing() {
        let mut index = VertexIncidenceIndex::default();
        let vertex = vertex_key(1);
        let missing = vertex_key(2);
        let before = [simplex_key(1), simplex_key(2), simplex_key(3)];
        index.insert_vertex(vertex).unwrap();
        for simplex in before {
            index.insert_simplex(simplex, &[vertex]).unwrap();
        }

        let err = index
            .remove_simplex(before[1], &[vertex, missing])
            .unwrap_err();

        assert!(matches!(err, TdsError::VertexNotFound { .. }));
        assert_eq!(
            index.simplex_keys(vertex).collect::<Vec<_>>(),
            before.to_vec()
        );
    }

    #[test]
    fn rollback_removed_simplex_restores_multiple_removals_exactly() {
        let mut index = VertexIncidenceIndex::default();
        let vertex = vertex_key(1);
        let before = [
            simplex_key(1),
            simplex_key(2),
            simplex_key(3),
            simplex_key(4),
        ];
        index.insert_vertex(vertex).unwrap();
        for simplex in before {
            index.insert_simplex(simplex, &[vertex]).unwrap();
        }

        let first_removal = index.remove_simplex(before[1], &[vertex]).unwrap();
        let second_removal = index.remove_simplex(before[2], &[vertex]).unwrap();
        index.rollback_removed_simplex(&second_removal);
        index.rollback_removed_simplex(&first_removal);

        assert_eq!(
            index.simplex_keys(vertex).collect::<Vec<_>>(),
            before.to_vec()
        );
    }

    #[test]
    fn simplex_incidence_round_trip_updates_only_listed_vertices() {
        let mut index = VertexIncidenceIndex::default();
        let a = vertex_key(1);
        let b = vertex_key(2);
        let isolated = vertex_key(3);
        let simplex = simplex_key(1);
        index.insert_vertex(a).unwrap();
        index.insert_vertex(b).unwrap();
        index.insert_vertex(isolated).unwrap();

        index.insert_simplex(simplex, &[a, b]).unwrap();
        assert_eq!(index.simplex_keys(a).collect::<Vec<_>>(), vec![simplex]);
        assert_eq!(index.simplex_keys(b).collect::<Vec<_>>(), vec![simplex]);
        assert_eq!(index.number_of_simplices(isolated), 0);

        index.remove_simplex(simplex, &[a, b]).unwrap();
        assert_eq!(index.number_of_simplices(a), 0);
        assert_eq!(index.number_of_simplices(b), 0);
        assert_eq!(index.number_of_simplices(isolated), 0);
    }
}
