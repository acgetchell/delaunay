//! Equality implementation for [`Tds`].

use super::storage::{SimplexUuidSortKey, Tds};
use crate::core::{simplex::Simplex, tds::VertexKey, vertex::Vertex};
use std::cmp::Ordering as CmpOrdering;

type SimplexUuidSortEntry<'a, V, const D: usize> = (SimplexUuidSortKey<D>, &'a Simplex<V, D>);
type SimplexCoordinateSignature<const D: usize> = Vec<[f64; D]>;

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

/// Orders normalized simplex coordinate signatures lexicographically.
fn compare_simplex_coordinate_signatures<const D: usize>(
    left: &[[f64; D]],
    right: &[[f64; D]],
) -> CmpOrdering {
    left.iter()
        .zip(right)
        .find_map(|(left, right)| {
            let ordering = compare_coords(left, right);
            (ordering != CmpOrdering::Equal).then_some(ordering)
        })
        .unwrap_or_else(|| left.len().cmp(&right.len()))
}

/// Normalizes a simplex to its coordinate identity for cross-TDS equality.
fn simplex_coordinate_signature<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex: &Simplex<V, D>,
) -> Option<SimplexCoordinateSignature<D>> {
    let mut coords = simplex
        .vertices()
        .iter()
        .map(|&vkey| tds.vertex(vkey).map(|vertex| *vertex.point().coords()))
        .collect::<Option<Vec<_>>>()?;
    coords.sort_by(compare_coords);
    Some(coords)
}

/// Normalizes one vertex star to the coordinate identities of incident simplices.
fn incident_simplex_coordinate_signatures<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    vertex_key: VertexKey,
) -> Option<Vec<SimplexCoordinateSignature<D>>> {
    let incident_simplices = tds.vertex_to_simplices_index().as_map().get(&vertex_key)?;
    let mut signatures = Vec::with_capacity(incident_simplices.len());
    for &simplex_key in incident_simplices {
        signatures.push(simplex_coordinate_signature(
            tds,
            tds.simplex(simplex_key)?,
        )?);
    }
    signatures.sort_by(|left, right| compare_simplex_coordinate_signatures(left, right));
    Some(signatures)
}

/// Compares the canonical vertex-to-simplices incidence relation.
fn vertex_incidence_indices_match<U, V, const D: usize>(
    left: &Tds<U, V, D>,
    right: &Tds<U, V, D>,
) -> bool {
    let mut left_vertices: Vec<_> = left.vertices.iter().collect();
    let mut right_vertices: Vec<_> = right.vertices.iter().collect();
    left_vertices.sort_by(|(_, left), (_, right)| {
        compare_coords(left.point().coords(), right.point().coords())
    });
    right_vertices.sort_by(|(_, left), (_, right)| {
        compare_coords(left.point().coords(), right.point().coords())
    });

    left_vertices
        .iter()
        .zip(right_vertices.iter())
        .all(|(left_vertex, right_vertex)| {
            let left_key = left_vertex.0;
            let right_key = right_vertex.0;
            incident_simplex_coordinate_signatures(left, left_key)
                == incident_simplex_coordinate_signatures(right, right_key)
        })
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

        if !vertex_incidence_indices_match(self, other) {
            return false;
        }

        // If we get here, the triangulations have the same structure
        // UUID→Key maps are derived from the vertices/simplices, so if those match, the maps should be consistent
        // (We don't need to compare the maps directly since they're just indexing structures)

        true
    }
}

#[cfg(test)]
mod tests {
    use super::compare_coords;
    use crate::DelaunayTriangulation;
    use crate::core::collections::PeriodicOffsetBuffer;
    use crate::core::tds::VertexKey;
    use crate::core::vertex::Vertex;
    use crate::vertex;
    use slotmap::KeyData;
    use std::cmp::Ordering as CmpOrdering;

    // =========================================================================
    // VALIDATION ERROR PATHS
    // =========================================================================

    #[test]
    fn compare_coords_uses_total_coordinate_ordering() {
        assert_eq!(
            compare_coords(&[f64::NAN, 1.0], &[0.0, 1.0]),
            CmpOrdering::Greater
        );
    }

    fn standard_vertices<const D: usize>() -> Vec<Vertex<(), D>> {
        let mut vertices = Vec::with_capacity(D + 1);
        vertices.push(vertex!([0.0; D]).unwrap());
        for axis in 0..D {
            let mut coords = [0.0; D];
            coords[axis] = 1.0;
            vertices.push(vertex!(coords).unwrap());
        }
        vertices
    }

    fn standard_vertices_with_extra<const D: usize>() -> Vec<Vertex<(), D>> {
        let mut vertices = standard_vertices::<D>();
        vertices.push(vertex!([0.1; D]).unwrap());
        vertices
    }

    macro_rules! gen_tds_partial_eq_error_path_tests {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<test_tds_partial_eq_different_counts_not_equal_ $dim d>]() {
                    let verts_a = standard_vertices::<$dim>();
                    let verts_b = standard_vertices_with_extra::<$dim>();
                    let dt_a: DelaunayTriangulation<_, (), (), $dim> =
                        DelaunayTriangulation::builder(&verts_a).build().unwrap();
                    let dt_b: DelaunayTriangulation<_, (), (), $dim> =
                        DelaunayTriangulation::builder(&verts_b).build().unwrap();

                    assert_ne!(dt_a.tds(), dt_b.tds());
                }

                #[test]
                fn [<test_tds_partial_eq_rejects_dangling_simplex_vertex_key_ $dim d>]() {
                    let verts = standard_vertices::<$dim>();
                    let dt: DelaunayTriangulation<_, (), (), $dim> =
                        DelaunayTriangulation::builder(&verts).build().unwrap();
                    let valid = dt.tds().clone();
                    let mut corrupted = dt.tds().clone();
                    let dangling_vertex = VertexKey::from(KeyData::from_ffi(9999));
                    corrupted.push_first_simplex_vertex_key_storage_only_for_test(dangling_vertex);

                    assert_ne!(corrupted, valid);
                }

                #[test]
                fn [<test_tds_partial_eq_rejects_misaligned_periodic_offsets_ $dim d>]() {
                    let verts = standard_vertices::<$dim>();
                    let dt: DelaunayTriangulation<_, (), (), $dim> =
                        DelaunayTriangulation::builder(&verts).build().unwrap();
                    let valid = dt.tds().clone();
                    let mut corrupted = dt.tds().clone();
                    corrupted.set_first_simplex_periodic_offsets_storage_only_for_test(Some(
                        PeriodicOffsetBuffer::new(),
                    ));

                    assert_ne!(corrupted, valid);
                }

                #[test]
                fn [<test_tds_partial_eq_rejects_missing_vertex_incidence_ $dim d>]() {
                    let verts = standard_vertices::<$dim>();
                    let dt: DelaunayTriangulation<_, (), (), $dim> =
                        DelaunayTriangulation::builder(&verts).build().unwrap();
                    let valid = dt.tds().clone();
                    let mut corrupted = dt.tds().clone();
                    let vertex_key = corrupted.vertices().next().map(|(key, _)| key).unwrap();
                    corrupted.clear_vertex_incidence_for_test(vertex_key);

                    assert_ne!(corrupted, valid);
                }
            }
        };
    }

    gen_tds_partial_eq_error_path_tests!(2);
    gen_tds_partial_eq_error_path_tests!(3);
    gen_tds_partial_eq_error_path_tests!(4);
    gen_tds_partial_eq_error_path_tests!(5);

    #[test]
    fn test_tds_partial_eq_different_structures_not_equal() {
        let verts_a = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let verts_b = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([2.0, 0.0]).unwrap(),
            vertex!([0.0, 2.0]).unwrap(),
        ];
        let dt_a: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&verts_a).build().unwrap();
        let dt_b: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&verts_b).build().unwrap();
        assert_ne!(dt_a.tds(), dt_b.tds());
    }
}
