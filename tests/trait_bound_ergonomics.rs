//! Compile coverage for read-only APIs with non-`DataType` payloads.

use std::{assert_matches, hash::Hasher};

use delaunay::DelaunayTriangulation;
use delaunay::prelude::Triangulation;
use delaunay::prelude::algorithms::{LocateError, locate, locate_with_stats};
use delaunay::prelude::construction::{
    DelaunayIncrementalBuilder, GlobalTopology, TopologyGuarantee, Vertex,
};
use delaunay::prelude::geometry::{
    Coordinate, CoordinateValidationError, FastKernel, Point, surface_measure,
};
use delaunay::prelude::query::{
    ConvexHull, QueryError, TopologyIndexBuildError, extract_edge_set,
    extract_facet_identifier_set, extract_vertex_coordinate_set,
};
use delaunay::prelude::tds::{
    FacetView, InvariantError, SimplexKey, Tds, TdsBuilder, VertexKey,
    verify_facet_index_consistency,
};
use delaunay::prelude::topology::validation::validate_triangulation_euler;
use delaunay::prelude::{dedup_vertices_exact, filter_vertices_excluding};
use uuid::Uuid;

struct Payload;
struct NotAKernel;

#[derive(Clone, PartialEq)]
struct ClonePayload(String);

struct OpaquePayload;

type NotAKernelTriangulation = Triangulation<NotAKernel, Payload, Payload, 2>;
type NotAKernelDelaunay = DelaunayTriangulation<NotAKernel, Payload, Payload, 2>;
type GenericTrySetTopologyFn =
    fn(&mut NotAKernelTriangulation, GlobalTopology<2>) -> Result<(), InvariantError>;

fn accepts_generic_try_set(_: GenericTrySetTopologyFn) {}

fn storage_methods_compile(tri: &mut NotAKernelTriangulation) {
    let _ = tri.set_vertex_data(VertexKey::default(), Some(Payload));
    let _ = tri.set_simplex_data(SimplexKey::default(), Some(Payload));
}

fn into_tds_compiles(tri: NotAKernelTriangulation) {
    let _: Tds<Payload, Payload, 2> = tri.into_tds();
}

fn triangulation_validation_compiles(tri: &NotAKernelTriangulation) {
    let _ = tri.is_valid_topology();
    let _ = tri.topology_report();
    let _ = tri.validate();
    let _ = tri.validation_report();
    let _ = tri.is_valid_realization();
    let _ = tri.realization_diagnostic();
    let _ = tri.realization_report();
    let _ = tri.validate_realization();
}

fn delaunay_validation_compiles(
    triangulation: &DelaunayTriangulation<FastKernel<f64>, Payload, Payload, 2>,
) {
    let _ = triangulation.is_valid_delaunay();
    let _ = triangulation.delaunay_diagnostic();
    let _ = triangulation.delaunay_report();
    let _ = triangulation.validate();
    let _ = triangulation.validation_report();
}

fn jaccard_extractors_compile(tri: &NotAKernelTriangulation) {
    let _ = extract_vertex_coordinate_set(tri);
    let _ = extract_edge_set(tri);
    let _ = extract_facet_identifier_set(tri);
}

fn convex_hull_constructor_compiles(
    tri: &Triangulation<NotAKernel, ClonePayload, OpaquePayload, 2>,
) {
    let _ = ConvexHull::try_from_triangulation(tri);
}

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
    let generic: Option<NotAKernelTriangulation> = None;
    let delaunay: Option<NotAKernelDelaunay> = None;

    assert!(generic.is_none());
    assert!(delaunay.is_none());
}

#[test]
fn storage_and_validation_methods_have_payload_agnostic_contracts() {
    let storage: fn(&mut NotAKernelTriangulation) = storage_methods_compile;
    let demotion: fn(NotAKernelTriangulation) = into_tds_compiles;
    let triangulation_validation: fn(&NotAKernelTriangulation) = triangulation_validation_compiles;
    let delaunay_validation: fn(&DelaunayTriangulation<FastKernel<f64>, Payload, Payload, 2>) =
        delaunay_validation_compiles;

    std::hint::black_box((
        storage,
        demotion,
        triangulation_validation,
        delaunay_validation,
    ));
}

#[test]
fn jaccard_and_convex_hull_extractors_have_minimal_payload_bounds() {
    let jaccard: fn(&NotAKernelTriangulation) = jaccard_extractors_compile;
    let hull: fn(&Triangulation<NotAKernel, ClonePayload, OpaquePayload, 2>) =
        convex_hull_constructor_compiles;

    std::hint::black_box((jaccard, hull));
}

#[test]
fn tds_builder_clones_non_datatype_vertex_payloads() {
    let vertices = [
        delaunay::vertex![0.0, 0.0; data = ClonePayload("first".to_owned())].unwrap(),
        delaunay::vertex![1.0, 0.0; data = ClonePayload("second".to_owned())].unwrap(),
        delaunay::vertex![0.0, 1.0; data = ClonePayload("third".to_owned())].unwrap(),
    ];
    let simplices = [vec![0, 1, 2]];

    let tds: Tds<ClonePayload, OpaquePayload, 2> = TdsBuilder::new(&vertices, &simplices)
        .simplex_data_type::<OpaquePayload>()
        .build()
        .unwrap();
    let mut payloads: Vec<_> = tds
        .vertices()
        .map(|(_, vertex)| vertex.data().map(|payload| payload.0.as_str()))
        .collect();
    payloads.sort_unstable();

    assert_eq!(payloads, [Some("first"), Some("second"), Some("third")]);
}

#[test]
fn deduplication_clones_and_preserves_first_noncopy_payload() {
    let vertices = [
        delaunay::vertex![0.0, 0.0; data = ClonePayload("first".to_owned())].unwrap(),
        delaunay::vertex![0.0, 0.0; data = ClonePayload("duplicate".to_owned())].unwrap(),
        delaunay::vertex![1.0, 0.0; data = ClonePayload("other".to_owned())].unwrap(),
    ];

    let deduplicated = dedup_vertices_exact(&vertices);
    let filtered = filter_vertices_excluding(&vertices, &vertices[2..]);

    assert_eq!(deduplicated.len(), 2);
    assert_eq!(
        deduplicated[0].data().map(|payload| payload.0.as_str()),
        Some("first")
    );
    assert_eq!(filtered.len(), 2);
    assert_eq!(
        filtered[0].data().map(|payload| payload.0.as_str()),
        Some("first")
    );
}

#[test]
fn triangulation_topology_metadata_setter_does_not_require_kernel_bounds() {
    accepts_generic_try_set(
        Triangulation::<NotAKernel, Payload, Payload, 2>::try_set_global_topology,
    );
}

#[test]
fn read_only_topology_apis_accept_non_datatype_payloads() {
    fn triangulation_queries_compile(tri: &Triangulation<FastKernel<f64>, Payload, Payload, 2>) {
        let _ = tri.number_of_vertices();
        let _ = tri.number_of_simplices();
        let _ = tri.boundary_facets();
        let _ = tri.incidence();
        let _ = tri.build_edge_index();
        let _ = tri.build_simplex_neighbor_index();
    }
    let compile_contract: fn(&Triangulation<FastKernel<f64>, Payload, Payload, 2>) =
        triangulation_queries_compile;
    std::hint::black_box(compile_contract);

    let tds: Tds<Payload, Payload, 2> = Tds::empty();
    assert!(tds.build_facet_to_simplices_index().unwrap().is_empty());
    assert_eq!(tds.number_of_one_sided_facets().unwrap(), 0);

    let topology = validate_triangulation_euler(&tds, GlobalTopology::Euclidean).unwrap();
    assert!(topology.is_valid());
}

#[test]
fn locate_and_conflict_apis_accept_non_datatype_payloads() {
    fn owner_queries_compile(
        tri: &Triangulation<FastKernel<f64>, Payload, Payload, 2>,
        dt: &DelaunayTriangulation<FastKernel<f64>, Payload, Payload, 2>,
        point: &Point<2>,
    ) {
        let _ = tri.locate(point, None);
        let _ = tri.locate_with_stats(point, None);
        let _ = tri.find_conflict_region(point, SimplexKey::default());
        let _ = dt.locate(point, None);
        let _ = dt.locate_with_stats(point, None);
        let _ = dt.find_conflict_region(point, SimplexKey::default());
    }
    type OwnerQueriesCompileFn = fn(
        &Triangulation<FastKernel<f64>, Payload, Payload, 2>,
        &DelaunayTriangulation<FastKernel<f64>, Payload, Payload, 2>,
        &Point<2>,
    );
    let compile_contract: OwnerQueriesCompileFn = owner_queries_compile;
    std::hint::black_box(compile_contract);

    let point = Point::try_new([0.25, 0.25]).unwrap();
    let kernel = FastKernel::new();
    let tds: Tds<Payload, Payload, 2> = Tds::empty();

    assert_matches!(
        locate(&tds, &kernel, &point, None),
        Err(LocateError::EmptyTriangulation)
    );
    assert_matches!(
        locate_with_stats(&tds, &kernel, &point, None),
        Err(LocateError::EmptyTriangulation)
    );
}

#[test]
fn tds_equality_accepts_non_datatype_payloads() {
    let left: Tds<Payload, Payload, 2> = Tds::empty();
    let right: Tds<Payload, Payload, 2> = Tds::empty();

    assert!(left == right);
}

#[test]
fn delaunay_query_wrappers_accept_non_datatype_payloads() {
    // Compile-only contract: `Payload` deliberately does not implement `DataType`.
    fn queries_compile(
        dt: &mut DelaunayTriangulation<FastKernel<f64>, Payload, Payload, 2>,
    ) -> Result<(), TraitBoundErgonomicsError> {
        let _ = dt.number_of_vertices();
        let _ = dt.number_of_simplices();
        let _ = dt.topology_guarantee();
        let _ = dt.global_topology();
        let _ = dt.topology_kind();
        dt.try_set_topology_guarantee(TopologyGuarantee::Pseudomanifold)
            .unwrap();
        let _ = dt.facets();
        let _ = dt.edges();
        let _ = dt.incident_edges(VertexKey::default());
        let _ = dt.simplex_neighbors(SimplexKey::default());
        let _ = dt.simplex_vertices(SimplexKey::default());
        let _ = dt.vertex_coords(VertexKey::default());
        let incidence = dt.incidence();
        let edge_index = dt.build_edge_index()?;
        let neighbor_index = dt.build_simplex_neighbor_index()?;
        std::hint::black_box(incidence.number_of_adjacent_simplices(VertexKey::default()));
        std::hint::black_box(edge_index.number_of_edges());
        std::hint::black_box(neighbor_index.number_of_simplex_neighbors(SimplexKey::default()));
        Ok(())
    }
    type QueriesCompileFn = fn(
        &mut DelaunayTriangulation<FastKernel<f64>, Payload, Payload, 2>,
    ) -> Result<(), TraitBoundErgonomicsError>;
    let compile_contract: QueriesCompileFn = queries_compile;
    std::hint::black_box(compile_contract);

    let empty: DelaunayTriangulation<_, (), (), 2> =
        DelaunayIncrementalBuilder::new().finish().unwrap();
    let point = Point::try_new([0.25, 0.25]).unwrap();
    assert_eq!(empty.number_of_vertices(), 0);
    assert_matches!(
        empty.locate(&point, None),
        Err(LocateError::EmptyTriangulation)
    );
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

#[test]
fn surface_measure_accepts_non_datatype_facet_views() {
    let facets: [FacetView<'_, Payload, Payload, 2>; 0] = [];
    let measure = surface_measure(&facets).unwrap();

    assert!(measure.abs() <= f64::EPSILON);
}
