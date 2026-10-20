//! Equality implementation for [`Tds`].

use super::model::Tds;
use crate::core::collections::FastHashMap;
use crate::core::simplex::SimplexCoordinateSignature;
use crate::core::tds::SimplexKey;

type SimplexSignatures<const D: usize> = FastHashMap<SimplexKey, SimplexCoordinateSignature<D>>;
type VertexIncidenceSignature<'a, const D: usize> =
    ([u64; D], Vec<&'a SimplexCoordinateSignature<D>>);

/// Resolves each simplex once so matching and incidence use the same identity.
fn simplex_signatures<U, V, const D: usize>(tds: &Tds<U, V, D>) -> Option<SimplexSignatures<D>> {
    tds.simplices
        .iter()
        .map(|(key, simplex)| Some((key, simplex.coordinate_signature(tds)?)))
        .collect()
}

/// Matches coincident vertices by their incident simplex multisets as well as
/// coordinates, rather than pairing different stars in storage order.
fn vertex_incidence_signatures<'a, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplices: &'a SimplexSignatures<D>,
) -> Option<Vec<VertexIncidenceSignature<'a, D>>> {
    let mut vertices = Vec::with_capacity(tds.vertices.len());
    for (key, vertex) in tds.vertices.iter() {
        let incident = tds.vertex_to_simplices_index().as_map().get(&key)?;
        let mut star = incident
            .iter()
            .map(|key| simplices.get(key))
            .collect::<Option<Vec<_>>>()?;
        star.sort_unstable();
        vertices.push((vertex.point().coords().map(f64::to_bits), star));
    }
    vertices.sort_unstable();
    Some(vertices)
}

/// Coordinate-based equality of triangulation data structures.
///
/// Compares vertex and simplex multisets and vertex-to-simplex incidence,
/// independent of UUIDs, payloads, storage order, and simplex vertex order.
/// Periodic simplex identities retain relative lattice offsets but identify
/// common lattice translations; absent offsets mean zero.
///
/// Coordinates compare exactly, following [`Point`](crate::geometry::point::Point)
/// equality, including its canonical treatment of signed zero. Distinct vertices
/// at the same coordinates are matched by their incident simplex multisets.
/// Runtime owner identity, generation, and transient buffers do not participate.
///
/// This comparison assumes the published Levels 1–2 invariants; it is not a
/// substitute for validation. Unresolvable references or malformed offsets
/// encountered during comparison return `false`.
impl<U, V, const D: usize> PartialEq for Tds<U, V, D> {
    fn eq(&self, other: &Self) -> bool {
        if self.vertices.len() != other.vertices.len()
            || self.simplices.len() != other.simplices.len()
            || self.uuid_to_vertex_key.len() != other.uuid_to_vertex_key.len()
            || self.uuid_to_simplex_key.len() != other.uuid_to_simplex_key.len()
        {
            return false;
        }

        let (Some(left), Some(right)) = (simplex_signatures(self), simplex_signatures(other))
        else {
            return false;
        };
        let mut left_simplices: Vec<_> = left.values().collect();
        let mut right_simplices: Vec<_> = right.values().collect();
        left_simplices.sort_unstable();
        right_simplices.sort_unstable();
        if left_simplices != right_simplices {
            return false;
        }

        matches!(
            (vertex_incidence_signatures(self, &left), vertex_incidence_signatures(other, &right)),
            (Some(left), Some(right)) if left == right
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::core::collections::PeriodicOffsetBuffer;
    use crate::core::tds::{Tds, TdsBuilder, VertexKey};
    use crate::core::vertex::Vertex;
    use crate::vertex;
    use slotmap::KeyData;

    // =========================================================================
    // VALIDATION ERROR PATHS
    // =========================================================================

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

    fn standard_tds<const D: usize>(with_interior_vertex: bool) -> Tds<(), (), D> {
        let vertices = if with_interior_vertex {
            standard_vertices_with_extra::<D>()
        } else {
            standard_vertices::<D>()
        };
        let simplices = if with_interior_vertex {
            (0..=D)
                .map(|omitted| {
                    (0..=D)
                        .filter(|&index| index != omitted)
                        .chain(std::iter::once(D + 1))
                        .collect()
                })
                .collect::<Vec<Vec<usize>>>()
        } else {
            vec![(0..=D).collect()]
        };
        TdsBuilder::new(&vertices, &simplices).build().unwrap()
    }

    /// Builds the same lifted simplex with fresh UUIDs and reversed local keys.
    fn periodic_tds<const D: usize>(reverse: bool, shift: i8) -> Tds<(), (), D> {
        let mut vertices = standard_vertices::<D>();
        if reverse {
            vertices.reverse();
        }
        let mut tds = TdsBuilder::new(&vertices, &[(0..=D).collect()])
            .build()
            .unwrap();
        let (_, simplex) = tds.simplices().next().unwrap();
        let offsets: Vec<_> = simplex
            .vertices()
            .iter()
            .map(|&key| {
                tds.vertex(key)
                    .unwrap()
                    .point()
                    .coords()
                    .map(|coord| if coord > 0.0 { 1 + shift } else { shift })
            })
            .collect();
        tds.simplices
            .values_mut()
            .next()
            .unwrap()
            .set_periodic_vertex_offsets(offsets)
            .unwrap();
        tds.validate().unwrap();
        tds
    }

    macro_rules! gen_tds_partial_eq_error_path_tests {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<test_tds_periodic_equality_ $dim d>]() {
                    let original = periodic_tds::<$dim>(false, 0);
                    let translated = periodic_tds::<$dim>(true, -12);
                    assert_eq!(original, translated);
                    let simplex = original.simplices().next().unwrap().1;
                    let translated_simplex = translated.simplices().next().unwrap().1;
                    assert!(simplex.eq_by_vertices(&original, translated_simplex, &translated));

                    let mut different = translated.clone();
                    let simplex = different.simplices.values_mut().next().unwrap();
                    let mut offsets = simplex.periodic_vertex_offsets().unwrap().to_vec();
                    offsets[0][0] += 1;
                    simplex.set_periodic_vertex_offsets(offsets).unwrap();
                    different.validate().unwrap();
                    assert_ne!(original, different);
                    assert!(!original.simplices().next().unwrap().1.eq_by_vertices(
                        &original, different.simplices().next().unwrap().1, &different,
                    ));

                    let plain = standard_tds::<$dim>(true);
                    let mut shifted = plain.clone();
                    // Different representatives may be chosen independently for each simplex.
                    for (index, simplex) in shifted.simplices.values_mut().enumerate() {
                        let shift = if index % 2 == 0 { i8::MIN } else { i8::MAX };
                        simplex.set_periodic_vertex_offsets(vec![[shift; $dim]; $dim + 1]).unwrap();
                    }
                    shifted.validate().unwrap();
                    assert_eq!(plain, shifted);
                }

                #[test]
                fn [<test_tds_partial_eq_different_counts_not_equal_ $dim d>]() {
                    let tds_a = standard_tds::<$dim>(false);
                    let tds_b = standard_tds::<$dim>(true);

                    assert_ne!(&tds_a, &tds_b);
                }

                #[test]
                fn [<test_tds_partial_eq_rejects_dangling_simplex_vertex_key_ $dim d>]() {
                    let tds = standard_tds::<$dim>(false);
                    let valid = tds.clone();
                    let mut corrupted = tds;
                    let dangling_vertex = VertexKey::from(KeyData::from_ffi(9999));
                    corrupted.push_first_simplex_vertex_key_storage_only_for_test(dangling_vertex);

                    assert_ne!(corrupted, valid);
                }

                #[test]
                fn [<test_tds_partial_eq_rejects_misaligned_periodic_offsets_ $dim d>]() {
                    let tds = standard_tds::<$dim>(false);
                    let valid = tds.clone();
                    let mut corrupted = tds;
                    corrupted.set_first_simplex_periodic_offsets_storage_only_for_test(Some(
                        PeriodicOffsetBuffer::new(),
                    ));

                    assert!(!corrupted.simplices().next().unwrap().1.eq_by_vertices(
                        &corrupted, valid.simplices().next().unwrap().1, &valid,
                    ));
                    assert_ne!(corrupted, valid);
                }

                #[test]
                fn [<test_tds_partial_eq_rejects_missing_vertex_incidence_ $dim d>]() {
                    let tds = standard_tds::<$dim>(false);
                    let valid = tds.clone();
                    let mut corrupted = tds;
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
        let simplices = [vec![0, 1, 2]];
        let tds_a = TdsBuilder::new(&verts_a, &simplices).build().unwrap();
        let tds_b = TdsBuilder::new(&verts_b, &simplices).build().unwrap();
        assert_ne!(&tds_a, &tds_b);
    }
}
