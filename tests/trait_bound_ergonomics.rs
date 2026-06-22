//! Compile coverage for read-only APIs with non-`DataType` payloads.

use std::hash::Hasher;

use delaunay::DelaunayTriangulation;
use delaunay::prelude::Triangulation;
use delaunay::prelude::construction::{GlobalTopology, TopologyGuarantee, TopologyKind};
use delaunay::prelude::geometry::{Coordinate, CoordinateValidationError, FastKernel, Point};
use delaunay::prelude::query::FacetIncidenceAnalysis;
use delaunay::prelude::tds::{
    SimplexKey, Tds, TdsError, Vertex, VertexKey, verify_facet_index_consistency,
};
use delaunay::prelude::topology::validation::validate_triangulation_euler;
use delaunay::query::{QueryError, TopologyIndexBuildError};
use uuid::Uuid;

struct Payload;
struct NotAKernel;

#[derive(Debug, thiserror::Error)]
enum TraitBoundErgonomicsError {
    #[error(transparent)]
    Adjacency {
        #[from]
        source: TopologyIndexBuildError,
    },
    #[error(transparent)]
    Query {
        #[from]
        source: QueryError,
    },
}

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
fn vertex_uuid_constructor_accepts_non_datatype_payloads() {
    let point = Point::<2>::try_new([1.0, 2.0]).unwrap();
    let uuid = Uuid::from_u128(0x67e5_5044_10b1_426f_9247_bb68_0e5f_e0c8);

    let vertex = Vertex::<Payload, 2>::try_new_with_uuid(point, uuid, Some(Payload)).unwrap();

    assert_eq!(vertex.uuid(), uuid);
    assert!(vertex.data().is_some());
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

    let incidence = tri.incidence().unwrap();
    let edge_index = tri.build_edge_index().unwrap();
    let neighbor_index = tri.build_simplex_neighbor_index().unwrap();
    assert_eq!(edge_index.number_of_edges(), 0);
    assert_eq!(
        incidence.number_of_adjacent_simplices(VertexKey::default()),
        0
    );
    assert_eq!(edge_index.number_of_incident_edges(VertexKey::default()), 0);
    assert_eq!(
        neighbor_index.number_of_simplex_neighbors(SimplexKey::default()),
        0
    );

    let tds: Tds<Payload, Payload, 2> = Tds::empty();
    assert!(tds.build_facet_to_simplices_index().unwrap().is_empty());
    assert_eq!(tds.number_of_one_sided_facets().unwrap(), 0);

    let topology = validate_triangulation_euler(&tds, GlobalTopology::Euclidean).unwrap();
    assert!(topology.is_valid());
}

#[test]
fn tds_equality_accepts_non_datatype_payloads() {
    let left: Tds<Payload, Payload, 2> = Tds::empty();
    let right: Tds<Payload, Payload, 2> = Tds::empty();

    assert!(left == right);
}

#[test]
fn delaunay_empty_query_wrappers_accept_non_datatype_payloads()
-> Result<(), TraitBoundErgonomicsError> {
    let mut dt: DelaunayTriangulation<FastKernel<f64>, Payload, Payload, 2> =
        DelaunayTriangulation::with_empty_kernel(FastKernel::new());

    assert_eq!(dt.number_of_vertices(), 0);
    assert_eq!(dt.number_of_simplices(), 0);
    assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);
    assert_eq!(dt.global_topology(), GlobalTopology::Euclidean);
    assert_eq!(dt.topology_kind(), TopologyKind::Euclidean);

    dt.set_global_topology(GlobalTopology::Euclidean);
    dt.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);
    assert_eq!(dt.topology_guarantee(), TopologyGuarantee::Pseudomanifold);

    assert!(dt.facets().next().is_none());
    assert_eq!(dt.edges().count(), 0);
    assert_eq!(dt.incident_edges(VertexKey::default()).count(), 0);
    assert_eq!(dt.simplex_neighbors(SimplexKey::default()).count(), 0);
    assert!(matches!(
        dt.simplex_vertices(SimplexKey::default()),
        Err(TdsError::SimplexNotFound { .. })
    ));
    assert_eq!(dt.vertex_coords(VertexKey::default()), None);

    let incidence = dt.incidence()?;
    let edge_index = dt.build_edge_index()?;
    let neighbor_index = dt.build_simplex_neighbor_index()?;
    assert_eq!(edge_index.edges().count(), 0);
    assert_eq!(
        incidence.adjacent_simplices(VertexKey::default()).count(),
        0
    );
    assert_eq!(edge_index.incident_edges(VertexKey::default()).count(), 0);
    assert_eq!(
        neighbor_index
            .simplex_neighbors(SimplexKey::default())
            .count(),
        0
    );

    let dt_with_topology: DelaunayTriangulation<FastKernel<f64>, Payload, Payload, 2> =
        DelaunayTriangulation::with_empty_kernel_and_topology_guarantee(
            FastKernel::new(),
            TopologyGuarantee::Pseudomanifold,
        );
    assert_eq!(
        dt_with_topology.topology_guarantee(),
        TopologyGuarantee::Pseudomanifold
    );

    Ok(())
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

    assert!(tds.try_simplex_facets(SimplexKey::default()).is_err());
}
