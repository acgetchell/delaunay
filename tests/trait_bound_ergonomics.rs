//! Compile coverage for read-only APIs with non-`DataType` payloads.

use std::hash::Hasher;

use delaunay::DelaunayTriangulation;
use delaunay::prelude::Triangulation;
use delaunay::prelude::geometry::{Coordinate, CoordinateValidationError, FastKernel};
use delaunay::prelude::query::BoundaryAnalysis;
use delaunay::prelude::tds::{Simplex, SimplexKey, Tds, verify_facet_index_consistency};
use delaunay::prelude::topology::validation::validate_triangulation_euler;

struct Payload;
struct NotAKernel;

struct MinimalCoordinate<const D: usize> {
    coords: [f64; D],
}

impl<const D: usize> Coordinate<D> for MinimalCoordinate<D> {
    fn try_new(coords: [f64; D]) -> Result<Self, CoordinateValidationError> {
        Ok(Self { coords })
    }

    fn to_array(&self) -> [f64; D] {
        self.coords
    }

    fn get(&self, index: usize) -> Option<f64> {
        self.coords.get(index).copied()
    }

    fn validate(&self) -> Result<(), CoordinateValidationError> {
        Ok(())
    }

    fn hash_coordinate<H: Hasher>(&self, state: &mut H) {
        for coord in self.coords {
            state.write_u64(coord.to_bits());
        }
    }

    fn ordered_equals(&self, other: &Self) -> bool {
        self.coords
            .iter()
            .zip(other.coords.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits())
    }
}

#[test]
fn coordinate_trait_has_minimal_bounds() {
    let coordinate = MinimalCoordinate::<2>::try_new([1.0, 2.0]).unwrap();

    assert_eq!(
        coordinate.to_array().map(f64::to_bits),
        [1.0_f64.to_bits(), 2.0_f64.to_bits()]
    );
    assert_eq!(coordinate.get(1), Some(2.0));
}

#[test]
fn triangulation_types_do_not_require_kernel_bounds() {
    let generic: Option<Triangulation<NotAKernel, Payload, Payload, 2>> = None;
    let delaunay: Option<DelaunayTriangulation<NotAKernel, Payload, Payload, 2>> = None;

    assert!(generic.is_none());
    assert!(delaunay.is_none());
}

#[test]
fn read_only_topology_apis_accept_non_datatype_payloads() {
    let tri: Triangulation<FastKernel<f64>, Payload, Payload, 2> =
        Triangulation::new_empty(FastKernel::new());

    assert_eq!(tri.number_of_vertices(), 0);
    assert_eq!(tri.number_of_simplices(), 0);
    assert_eq!(
        tri.boundary_facets()
            .unwrap()
            .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))
            .unwrap(),
        0
    );

    let index = tri.build_adjacency_index().unwrap();
    assert!(index.vertex_to_simplices.is_empty());
    assert!(index.simplex_to_neighbors.is_empty());
    assert!(index.vertex_to_edges.is_empty());

    let tds: Tds<Payload, Payload, 2> = Tds::empty();
    assert!(tds.build_facet_to_simplices_map().unwrap().is_empty());
    assert_eq!(tds.number_of_boundary_facets().unwrap(), 0);

    let topology = validate_triangulation_euler(&tds).unwrap();
    assert!(topology.is_valid());
}

#[test]
fn facet_index_consistency_accepts_non_datatype_payloads() {
    let tds: Tds<Payload, Payload, 2> = Tds::empty();

    assert!(
        verify_facet_index_consistency(&tds, SimplexKey::default(), SimplexKey::default(), 0)
            .is_err()
    );
}

#[test]
fn facet_views_accept_non_datatype_payloads() {
    let tds: Tds<Payload, Payload, 2> = Tds::empty();

    assert!(Simplex::facet_views_from_tds(&tds, SimplexKey::default()).is_err());
    assert!(Simplex::facet_view_iter(&tds, SimplexKey::default()).is_err());
}
