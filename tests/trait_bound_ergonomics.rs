//! Compile coverage for read-only APIs with non-`DataType` payloads.

use delaunay::prelude::Triangulation;
use delaunay::prelude::geometry::FastKernel;
use delaunay::prelude::query::BoundaryAnalysis;
use delaunay::prelude::tds::{Simplex, SimplexKey, Tds, verify_facet_index_consistency};
use delaunay::prelude::topology::validation::validate_triangulation_euler;

struct Payload;

#[test]
fn read_only_topology_apis_accept_non_datatype_payloads() {
    let tri: Triangulation<FastKernel<f64>, Payload, Payload, 2> =
        Triangulation::new_empty(FastKernel::new());

    assert_eq!(tri.number_of_vertices(), 0);
    assert_eq!(tri.number_of_simplices(), 0);
    assert_eq!(tri.boundary_facets().unwrap().count(), 0);

    let index = tri.build_adjacency_index().unwrap();
    assert!(index.vertex_to_simplices.is_empty());
    assert!(index.simplex_to_neighbors.is_empty());
    assert!(index.vertex_to_edges.is_empty());

    let tds: Tds<f64, Payload, Payload, 2> = Tds::empty();
    assert!(tds.build_facet_to_simplices_map().unwrap().is_empty());
    assert_eq!(tds.number_of_boundary_facets().unwrap(), 0);

    let topology = validate_triangulation_euler(&tds).unwrap();
    assert!(topology.is_valid());
}

#[test]
fn facet_index_consistency_accepts_non_datatype_payloads() {
    let tds: Tds<f64, Payload, Payload, 2> = Tds::empty();

    assert!(
        verify_facet_index_consistency(&tds, SimplexKey::default(), SimplexKey::default(), 0)
            .is_err()
    );
}

#[test]
fn facet_views_accept_non_datatype_payloads() {
    let tds: Tds<f64, Payload, Payload, 2> = Tds::empty();

    assert!(Simplex::facet_views_from_tds(&tds, SimplexKey::default()).is_err());
    assert!(Simplex::facet_view_iter(&tds, SimplexKey::default()).is_err());
}
