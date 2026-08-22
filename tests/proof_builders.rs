//! Public contract tests for the layered proof builders.

use delaunay::prelude::construction::DelaunayIncrementalBuilder;
use delaunay::prelude::delaunayize::DelaunayRefinementBuilder;
use delaunay::prelude::geometry::AdaptiveKernel;
use delaunay::prelude::tds::{Tds, TdsBuilder, TdsBuilderError, VertexKey};
use delaunay::prelude::triangulation::TriangulationBuilder;
use delaunay::vertex;

fn inspect_incremental_builder_without_construction_bounds<K, U, V, const D: usize>(
    builder: &DelaunayIncrementalBuilder<K, U, V, D>,
    vertex_key: VertexKey,
) -> (usize, usize, i32, bool) {
    (
        builder.number_of_vertices(),
        builder.number_of_simplices(),
        builder.dim(),
        builder.vertex(vertex_key).is_some(),
    )
}

#[test]
fn strict_proof_chain_preserves_explicit_tds_representation() {
    let vertices = [
        vertex!([0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];
    let simplices = [vec![0, 1, 2]];
    let mut tds: Tds<(), usize, 2> = TdsBuilder::new(&vertices, &simplices)
        .simplex_data_type::<usize>()
        .build()
        .unwrap();
    let simplex_key = tds.simplex_keys().next().unwrap();
    tds.set_simplex_data(simplex_key, Some(7)).unwrap();

    let expected = serde_json::to_value(&tds).unwrap();
    let owner = tds.topology_owner_id();
    let generation = tds.generation();
    let triangulation = TriangulationBuilder::new(tds, AdaptiveKernel::new())
        .build()
        .unwrap();
    assert_eq!(triangulation.topology_generation(), generation);
    triangulation.validate_realization().unwrap();

    let delaunay = DelaunayRefinementBuilder::new(triangulation)
        .build()
        .unwrap();
    delaunay.validate().unwrap();

    let restored = delaunay.into_triangulation().into_tds();
    assert_eq!(restored.topology_owner_id(), owner);
    assert_eq!(restored.generation(), generation);
    assert_eq!(serde_json::to_value(&restored).unwrap(), expected);
    assert_eq!(restored.simplex(simplex_key).unwrap().data(), Some(&7));
}

#[test]
fn tds_builder_reports_malformed_connectivity_with_typed_context() {
    let vertices = [
        vertex!([0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];
    let malformed = [vec![0, 1, 3]];

    std::assert_matches!(
        TdsBuilder::new(&vertices, &malformed).build(),
        Err(TdsBuilderError::IndexOutOfBounds {
            simplex_index: 0,
            vertex_index: 3,
            bound: 3,
        })
    );
}

#[test]
fn incremental_builder_inspection_does_not_repeat_construction_bounds() {
    let mut builder: DelaunayIncrementalBuilder<_, (), (), 2> = DelaunayIncrementalBuilder::new();
    let vertex_key = builder.insert_vertex(vertex!([0.0, 0.0]).unwrap()).unwrap();

    assert_eq!(
        inspect_incremental_builder_without_construction_bounds(&builder, vertex_key),
        (1, 0, 0, true)
    );
}
