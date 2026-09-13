//! Owner-bound incidence iterators must not borrow temporary query wrappers.

#![forbid(unsafe_code)]

use delaunay::prelude::construction::{DelaunayTriangulationBuilder, Vertex, vertex};
use delaunay::prelude::geometry::{AdaptiveKernel, ExactPredicates};
use delaunay::prelude::tds::VertexKey;
use slotmap::Key;

fn assert_incidence_lifetimes<const D: usize>(vertices: &[Vertex<(), D>])
where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    let owner = DelaunayTriangulationBuilder::new(vertices)
        .build()
        .expect("a standard simplex must construct successfully");
    assert_eq!(owner.number_of_simplices(), 1);
    let (simplex_key, simplex) = owner.simplices().next().expect("the simplex must exist");

    for (vertex_key, _) in owner.vertices() {
        assert!(simplex.contains_vertex(vertex_key));

        let incident = owner.incidence().adjacent_simplices(vertex_key);
        assert_eq!(incident.collect::<Vec<_>>(), vec![simplex_key]);

        let incident = owner
            .as_triangulation()
            .incidence()
            .adjacent_simplices(vertex_key);
        assert_eq!(incident.collect::<Vec<_>>(), vec![simplex_key]);

        let incident = {
            let adjacency = owner.adjacency().expect("valid adjacency must build");
            adjacency.adjacent_simplices(vertex_key)
        };
        assert_eq!(incident.collect::<Vec<_>>(), vec![simplex_key]);

        let incident = {
            let adjacency = owner
                .as_triangulation()
                .adjacency()
                .expect("valid generic adjacency must build");
            adjacency.adjacent_simplices(vertex_key)
        };
        assert_eq!(incident.collect::<Vec<_>>(), vec![simplex_key]);
    }

    let absent = owner.incidence().adjacent_simplices(VertexKey::null());
    assert_eq!(absent.count(), 0);
}

#[test]
fn incidence_iterators_outlive_wrappers_2d() {
    assert_incidence_lifetimes(&[
        vertex![0.0, 0.0].expect("finite coordinates"),
        vertex![1.0, 0.0].expect("finite coordinates"),
        vertex![0.0, 1.0].expect("finite coordinates"),
    ]);
}

#[test]
fn incidence_iterators_outlive_wrappers_3d() {
    assert_incidence_lifetimes(&[
        vertex![0.0, 0.0, 0.0].expect("finite coordinates"),
        vertex![1.0, 0.0, 0.0].expect("finite coordinates"),
        vertex![0.0, 1.0, 0.0].expect("finite coordinates"),
        vertex![0.0, 0.0, 1.0].expect("finite coordinates"),
    ]);
}

#[test]
fn incidence_iterators_outlive_wrappers_4d() {
    assert_incidence_lifetimes(&[
        vertex![0.0, 0.0, 0.0, 0.0].expect("finite coordinates"),
        vertex![1.0, 0.0, 0.0, 0.0].expect("finite coordinates"),
        vertex![0.0, 1.0, 0.0, 0.0].expect("finite coordinates"),
        vertex![0.0, 0.0, 1.0, 0.0].expect("finite coordinates"),
        vertex![0.0, 0.0, 0.0, 1.0].expect("finite coordinates"),
    ]);
}

#[test]
fn incidence_iterators_outlive_wrappers_5d() {
    assert_incidence_lifetimes(&[
        vertex![0.0, 0.0, 0.0, 0.0, 0.0].expect("finite coordinates"),
        vertex![1.0, 0.0, 0.0, 0.0, 0.0].expect("finite coordinates"),
        vertex![0.0, 1.0, 0.0, 0.0, 0.0].expect("finite coordinates"),
        vertex![0.0, 0.0, 1.0, 0.0, 0.0].expect("finite coordinates"),
        vertex![0.0, 0.0, 0.0, 1.0, 0.0].expect("finite coordinates"),
        vertex![0.0, 0.0, 0.0, 0.0, 1.0].expect("finite coordinates"),
    ]);
}
