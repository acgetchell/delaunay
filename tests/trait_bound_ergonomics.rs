//! Compile coverage for read-only APIs with non-`DataType` payloads.

use delaunay::prelude::geometry::FastKernel;
use delaunay::prelude::query::BoundaryAnalysis;
use delaunay::prelude::tds::{CellKey, Tds, verify_facet_index_consistency};
use delaunay::prelude::topology::validation::validate_triangulation_euler;
use delaunay::prelude::triangulation::Triangulation;

struct Payload;

#[test]
fn read_only_topology_apis_accept_non_datatype_payloads() {
    let tri: Triangulation<FastKernel<f64>, Payload, Payload, 2> =
        Triangulation::new_empty(FastKernel::new());

    assert_eq!(tri.number_of_vertices(), 0);
    assert_eq!(tri.number_of_cells(), 0);
    assert_eq!(tri.boundary_facets().count(), 0);

    let index = tri.build_adjacency_index().unwrap();
    assert!(index.vertex_to_cells.is_empty());
    assert!(index.cell_to_neighbors.is_empty());
    assert!(index.vertex_to_edges.is_empty());

    let tds: Tds<f64, Payload, Payload, 2> = Tds::empty();
    assert!(tds.build_facet_to_cells_map().unwrap().is_empty());
    assert_eq!(tds.number_of_boundary_facets().unwrap(), 0);

    let topology = validate_triangulation_euler(&tds).unwrap();
    assert!(topology.is_valid());
}

#[test]
fn facet_index_consistency_accepts_non_datatype_payloads() {
    let tds: Tds<f64, Payload, Payload, 2> = Tds::empty();

    assert!(
        verify_facet_index_consistency(&tds, CellKey::default(), CellKey::default(), 0).is_err()
    );
}
