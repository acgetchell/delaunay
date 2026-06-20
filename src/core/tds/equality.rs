//! Equality implementation for [`Tds`].

use super::storage::{SimplexUuidSortKey, Tds};
use crate::core::{simplex::Simplex, vertex::Vertex};
use std::cmp::Ordering as CmpOrdering;

type SimplexUuidSortEntry<'a, V, const D: usize> = (SimplexUuidSortKey<D>, &'a Simplex<V, D>);

/// Builds stable simplex sort keys once so equality does not hide dangling
/// vertex references or allocate sort keys repeatedly during comparison.
fn simplex_uuid_sort_entries<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
) -> Option<Vec<SimplexUuidSortEntry<'_, V, D>>> {
    tds.simplices
        .values()
        .map(|simplex| {
            let offsets = simplex.periodic_vertex_offsets();
            if let Some(offsets) = offsets
                && offsets.len() != simplex.number_of_vertices()
            {
                return None;
            }

            let mut ids = SimplexUuidSortKey::<D>::new();
            for (idx, &vkey) in simplex.vertices().iter().enumerate() {
                let uuid = tds.vertex(vkey).map(Vertex::uuid)?;
                let offset = offsets.map_or([0_i8; D], |offsets| offsets[idx]);
                ids.push((uuid, offset));
            }
            ids.sort_unstable();
            Some((ids, simplex))
        })
        .collect()
}

/// Orders coordinate arrays using the same total f64 ordering as geometric
/// primitives.
fn compare_coords<const D: usize>(left: &[f64; D], right: &[f64; D]) -> CmpOrdering {
    left.iter()
        .zip(right)
        .find_map(|(left, right)| {
            let ordering = left.total_cmp(right);
            (ordering != CmpOrdering::Equal).then_some(ordering)
        })
        .unwrap_or(CmpOrdering::Equal)
}

/// Manual implementation of `PartialEq` for Tds
///
/// Two triangulation data structures are considered equal if they have:
/// - The same set of vertices (compared by coordinates)
/// - The same set of simplices (compared by vertex sets)
/// - Consistent vertex and simplex mappings
///
/// **Note:** Vertices with NaN coordinates are rejected during construction; equality assumes no NaNs.
/// The triangulation validates coordinates at construction time to ensure no NaN values are present.
///
/// Note: Buffer fields are ignored since they are transient data structures.
impl<U, V, const D: usize> PartialEq for Tds<U, V, D> {
    fn eq(&self, other: &Self) -> bool {
        // Early exit if the basic counts don't match
        if self.vertices.len() != other.vertices.len()
            || self.simplices.len() != other.simplices.len()
            || self.uuid_to_vertex_key.len() != other.uuid_to_vertex_key.len()
            || self.uuid_to_simplex_key.len() != other.uuid_to_simplex_key.len()
        {
            return false;
        }

        // Compare vertices by collecting them into sorted vectors
        // We sort by coordinates to make comparison order-independent
        let mut self_vertices: Vec<_> = self.vertices.values().collect();
        let mut other_vertices: Vec<_> = other.vertices.values().collect();

        // Sort vertices by their coordinates for consistent comparison
        // NaN validation occurs at construction time; total ordering keeps corrupted
        // internal states deterministic in tests and diagnostics.
        self_vertices.sort_by(|a, b| {
            let a_coords = *a.point().coords();
            let b_coords = *b.point().coords();
            compare_coords(&a_coords, &b_coords)
        });

        other_vertices.sort_by(|a, b| {
            let a_coords = *a.point().coords();
            let b_coords = *b.point().coords();
            compare_coords(&a_coords, &b_coords)
        });

        // Compare sorted vertex lists
        if self_vertices != other_vertices {
            return false;
        }

        // Compare simplices using Simplex::eq_by_vertices() which uses
        // Vertex::PartialEq. Missing vertex references make the structures
        // unequal rather than being silently dropped from sort keys.
        let (Some(mut self_simplices), Some(mut other_simplices)) = (
            simplex_uuid_sort_entries(self),
            simplex_uuid_sort_entries(other),
        ) else {
            return false;
        };

        self_simplices.sort_by(|(a_ids, _), (b_ids, _)| a_ids.cmp(b_ids));
        other_simplices.sort_by(|(a_ids, _), (b_ids, _)| a_ids.cmp(b_ids));

        // Compare sorted simplex lists using Simplex::eq_by_vertices
        if self_simplices.len() != other_simplices.len() {
            return false;
        }

        for ((_, self_simplex), (_, other_simplex)) in
            self_simplices.iter().zip(other_simplices.iter())
        {
            if !self_simplex.eq_by_vertices(self, other_simplex, other) {
                return false;
            }
        }

        // If we get here, the triangulations have the same structure
        // UUID→Key maps are derived from the vertices/simplices, so if those match, the maps should be consistent
        // (We don't need to compare the maps directly since they're just indexing structures)

        true
    }
}

/// Eq implementation for Tds
///
/// This is a marker trait implementation that relies on the `PartialEq` implementation.
/// Since Tds represents a well-defined mathematical structure (triangulation),
/// the `PartialEq` relation is indeed an equivalence relation.
impl<U, V, const D: usize> Eq for Tds<U, V, D> {}
#[cfg(test)]
mod tests {
    use crate::DelaunayTriangulation;
    use crate::core::collections::PeriodicOffsetBuffer;
    use crate::core::tds::VertexKey;
    use crate::core::vertex::Vertex;
    use slotmap::KeyData;
    use std::cmp::Ordering as CmpOrdering;

    // =========================================================================
    // VALIDATION ERROR PATHS
    // =========================================================================

    #[test]
    fn compare_coords_uses_total_coordinate_ordering() {
        assert_eq!(
            super::compare_coords(&[f64::NAN, 1.0], &[0.0, 1.0]),
            CmpOrdering::Greater
        );
    }

    #[test]
    fn test_tds_partial_eq_different_counts_not_equal() {
        let verts_a = [
            Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let verts_b = [
            Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
        ];
        let dt_a: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&verts_a).unwrap();
        let dt_b: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&verts_b).unwrap();

        assert_ne!(dt_a.tds(), dt_b.tds());
    }

    #[test]
    fn test_tds_partial_eq_rejects_dangling_simplex_vertex_key() {
        let verts = [
            Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&verts).unwrap();
        let valid = dt.tds().clone();
        let mut corrupted = dt.tds().clone();
        let dangling_vertex = VertexKey::from(KeyData::from_ffi(9999));
        corrupted.push_first_simplex_vertex_key_storage_only_for_test(dangling_vertex);

        assert_ne!(corrupted, valid);
    }

    #[test]
    fn test_tds_partial_eq_rejects_misaligned_periodic_offsets() {
        let verts = [
            Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&verts).unwrap();
        let valid = dt.tds().clone();
        let mut corrupted = dt.tds().clone();
        corrupted.set_first_simplex_periodic_offsets_storage_only_for_test(Some(
            PeriodicOffsetBuffer::new(),
        ));

        assert_ne!(corrupted, valid);
    }

    #[test]
    fn test_tds_partial_eq_different_structures_not_equal() {
        let verts_a = [
            Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let verts_b = [
            Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([2.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 2.0]).unwrap(),
        ];
        let dt_a: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&verts_a).unwrap();
        let dt_b: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&verts_b).unwrap();
        assert_ne!(dt_a.tds(), dt_b.tds());
    }
}
