//! Public mesh and visualization export tests.

#![forbid(unsafe_code)]

use std::{
    assert_matches,
    collections::{HashMap, HashSet},
    error::Error as StdError,
};

use delaunay::geometry::CoordinateConversionError;
use delaunay::prelude::construction::{
    DelaunayError, DelaunayResult, DelaunayTriangulationBuilder,
    DelaunayTriangulationConstructionError, TopologyGuarantee, Vertex, vertex,
};
use delaunay::prelude::export::{
    AdjacencyRecord, InvalidCoordinateValue, MESH_EXPORT_SCHEMA, MESH_EXPORT_SCHEMA_VERSION,
    MeshExport, MeshExportError, SimplexRecord, ValidatedMeshExport, ValidatedVisualizationData,
    VertexRecord, VisualizationData, VisualizationDataValidationError, VisualizationExportError,
    VisualizationMetadata, VisualizationTopologyGuarantee, VisualizationTopologyKind,
};
use delaunay::prelude::topology::spaces::TopologyKind;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
enum MeshExportTestError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Coordinate(#[from] CoordinateConversionError),
    #[error(transparent)]
    Export(#[from] MeshExportError),
    #[error(transparent)]
    Validation(#[from] VisualizationDataValidationError),
    #[error(transparent)]
    Serde(#[from] serde_json::Error),
}

struct OpaqueAttribute;

#[derive(Debug, PartialEq, serde::Deserialize, serde::Serialize)]
struct NonDefaultAttribute;

fn sample_export() -> Result<MeshExport<2>, MeshExportTestError> {
    let vertices = vec![
        vertex![0.0, 0.0]?,
        vertex![2.0, 0.0]?,
        vertex![0.0, 2.0]?,
        vertex![0.25, 0.25]?,
    ];
    let triangulation = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

    Ok(triangulation.to_mesh_export()?)
}

fn sample_delaunay_result_export() -> DelaunayResult<MeshExport<2>> {
    let vertices = vec![
        vertex![0.0, 0.0]?,
        vertex![2.0, 0.0]?,
        vertex![0.0, 2.0]?,
        vertex![0.25, 0.25]?,
    ];
    let triangulation = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

    Ok(triangulation.to_mesh_export()?)
}

/// Builds the minimal affinely spanning fixture used by the dimension sweep.
fn sample_vertices_for_dimension<const D: usize>() -> Result<Vec<Vertex<(), D>>, MeshExportTestError>
{
    let mut vertices = Vec::with_capacity(D + 2);
    vertices.push(vertex!([0.0; D])?);
    for axis in 0..D {
        let mut coords = [0.0; D];
        coords[axis] = 1.0;
        vertices.push(vertex!(coords)?);
    }
    vertices.push(vertex!([0.125; D])?);
    Ok(vertices)
}

/// Exports one dimension-sweep fixture through the public builder path.
fn sample_export_for_dimension<const D: usize>() -> Result<MeshExport<D>, MeshExportTestError> {
    let vertices = sample_vertices_for_dimension::<D>()?;
    let triangulation = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    Ok(triangulation.to_mesh_export()?)
}

/// Builds a raw DTO where distinct facets reciprocate a self-neighbor link.
fn reciprocal_self_neighbor_export() -> MeshExport<2> {
    let vertex_ids = [
        Uuid::from_u128(0x1000_0000_0000_0000_0000_0000_0000_0001),
        Uuid::from_u128(0x1000_0000_0000_0000_0000_0000_0000_0002),
        Uuid::from_u128(0x1000_0000_0000_0000_0000_0000_0000_0003),
    ];
    let simplex_id = Uuid::from_u128(0x2000_0000_0000_0000_0000_0000_0000_0001);

    MeshExport {
        metadata: VisualizationMetadata {
            schema: MESH_EXPORT_SCHEMA.to_owned(),
            schema_version: MESH_EXPORT_SCHEMA_VERSION,
            producer: "delaunay-test".to_owned(),
            dimension: 2,
            vertex_count: vertex_ids.len(),
            simplex_count: 1,
            topology_kind: VisualizationTopologyKind::Euclidean,
            topology_guarantee: VisualizationTopologyGuarantee::Pseudomanifold,
            attributes: None,
        },
        vertices: vec![
            VertexRecord {
                id: vertex_ids[0],
                coordinates: vec![0.0, 0.0],
                attributes: None,
            },
            VertexRecord {
                id: vertex_ids[1],
                coordinates: vec![1.0, 0.0],
                attributes: None,
            },
            VertexRecord {
                id: vertex_ids[2],
                coordinates: vec![0.0, 1.0],
                attributes: None,
            },
        ],
        simplices: vec![SimplexRecord {
            id: simplex_id,
            vertex_ids: vertex_ids.to_vec(),
            attributes: None,
        }],
        adjacency: vec![
            AdjacencyRecord {
                simplex_id,
                facet_index: 0,
                neighbor_simplex_id: Some(simplex_id),
                attributes: None,
            },
            AdjacencyRecord {
                simplex_id,
                facet_index: 1,
                neighbor_simplex_id: Some(simplex_id),
                attributes: None,
            },
            AdjacencyRecord {
                simplex_id,
                facet_index: 2,
                neighbor_simplex_id: None,
                attributes: None,
            },
        ],
    }
}

/// Verifies const-generic export invariants and serde round-trip behavior.
fn assert_mesh_export_round_trip_for_dimension<const D: usize>() -> Result<(), MeshExportTestError>
{
    let export = sample_export_for_dimension::<D>()?;
    assert_eq!(export.metadata.dimension, D);
    assert_eq!(export.metadata.vertex_count, export.vertices.len());
    assert_eq!(export.metadata.simplex_count, export.simplices.len());
    assert_eq!(export.adjacency.len(), export.simplices.len() * (D + 1));
    assert_connectivity_ids_exist(&export);
    export.validate()?;

    let json = serde_json::to_string(&export)?;
    let decoded: MeshExport<D> = serde_json::from_str(&json)?;
    assert_connectivity_ids_exist(&decoded);
    decoded.validate()?;
    Ok(())
}

fn assert_connectivity_ids_exist<const D: usize>(export: &MeshExport<D>) {
    let vertex_ids: HashSet<_> = export.vertices.iter().map(|vertex| vertex.id).collect();
    let simplex_ids: HashSet<_> = export.simplices.iter().map(|simplex| simplex.id).collect();

    assert_eq!(vertex_ids.len(), export.vertices.len());
    assert_eq!(simplex_ids.len(), export.simplices.len());

    for simplex in &export.simplices {
        assert_eq!(simplex.vertex_ids.len(), D + 1);
        for vertex_id in &simplex.vertex_ids {
            assert!(
                vertex_ids.contains(vertex_id),
                "simplex {} references missing vertex {}",
                simplex.id,
                vertex_id
            );
        }
    }

    for adjacency in &export.adjacency {
        assert!(
            simplex_ids.contains(&adjacency.simplex_id),
            "adjacency references missing simplex {}",
            adjacency.simplex_id
        );
        assert!(adjacency.facet_index < D + 1);
        if let Some(neighbor_id) = adjacency.neighbor_simplex_id {
            assert!(
                simplex_ids.contains(&neighbor_id),
                "adjacency references missing neighbor {neighbor_id}"
            );
        }
    }
}

fn assert_neighbor_links_are_symmetric(export: &MeshExport<2>) {
    let neighbor_links: HashSet<_> = export
        .adjacency
        .iter()
        .filter_map(|adjacency| {
            adjacency
                .neighbor_simplex_id
                .map(|neighbor_id| (adjacency.simplex_id, neighbor_id))
        })
        .collect();

    for &(simplex_id, neighbor_id) in &neighbor_links {
        assert!(
            neighbor_links.contains(&(neighbor_id, simplex_id)),
            "neighbor link {simplex_id} -> {neighbor_id} is not reciprocal"
        );
    }
}

const fn assert_send_sync_unpin<T: Send + Sync + Unpin>() {}

#[test]
fn mesh_export_json_contains_schema_ids_and_connectivity() -> Result<(), MeshExportTestError> {
    let export = sample_export()?;

    assert_eq!(export.metadata.schema, MESH_EXPORT_SCHEMA);
    assert_eq!(export.metadata.schema_version, MESH_EXPORT_SCHEMA_VERSION);
    assert_eq!(export.metadata.producer, "delaunay");
    assert_eq!(export.metadata.dimension, 2);
    assert_eq!(export.metadata.vertex_count, export.vertices.len());
    assert_eq!(export.metadata.simplex_count, export.simplices.len());
    export.validate()?;
    assert_eq!(export.vertices.len(), 4);
    assert!(!export.simplices.is_empty());
    assert_eq!(export.adjacency.len(), export.simplices.len() * 3);

    assert_connectivity_ids_exist(&export);
    assert_neighbor_links_are_symmetric(&export);

    let json = serde_json::to_value(&export)?;
    assert_eq!(json["metadata"]["schema"], MESH_EXPORT_SCHEMA);
    assert_eq!(
        json["metadata"]["schema_version"],
        MESH_EXPORT_SCHEMA_VERSION
    );
    assert_eq!(json["metadata"]["dimension"], 2);
    assert!(
        json["vertices"]
            .as_array()
            .is_some_and(|vertices| !vertices.is_empty())
    );
    assert!(
        json["simplices"]
            .as_array()
            .is_some_and(|simplices| !simplices.is_empty())
    );
    assert!(
        json["adjacency"]
            .as_array()
            .is_some_and(|adjacency| !adjacency.is_empty())
    );

    Ok(())
}

#[test]
fn mesh_export_round_trips_for_dimensions_2_through_5() -> Result<(), MeshExportTestError> {
    assert_mesh_export_round_trip_for_dimension::<2>()?;
    assert_mesh_export_round_trip_for_dimension::<3>()?;
    assert_mesh_export_round_trip_for_dimension::<4>()?;
    assert_mesh_export_round_trip_for_dimension::<5>()?;
    Ok(())
}

#[test]
fn visualization_topology_schema_categories_are_stable() -> Result<(), MeshExportTestError> {
    for (source, expected, schema) in [
        (
            TopologyKind::Euclidean,
            VisualizationTopologyKind::Euclidean,
            "Euclidean",
        ),
        (
            TopologyKind::Toroidal,
            VisualizationTopologyKind::Toroidal,
            "Toroidal",
        ),
        (
            TopologyKind::Spherical,
            VisualizationTopologyKind::Spherical,
            "Spherical",
        ),
        (
            TopologyKind::Hyperbolic,
            VisualizationTopologyKind::Hyperbolic,
            "Hyperbolic",
        ),
    ] {
        let converted = VisualizationTopologyKind::from(source);

        assert_eq!(converted, expected);
        assert_eq!(converted.to_string(), schema);
        assert_eq!(serde_json::to_value(&converted)?, serde_json::json!(schema));
        assert_eq!(
            serde_json::from_value::<VisualizationTopologyKind>(serde_json::json!(schema))?,
            expected
        );
    }

    for (source, expected, schema) in [
        (
            TopologyGuarantee::Pseudomanifold,
            VisualizationTopologyGuarantee::Pseudomanifold,
            "Pseudomanifold",
        ),
        (
            TopologyGuarantee::PLManifold,
            VisualizationTopologyGuarantee::PLManifold,
            "PLManifold",
        ),
        (
            TopologyGuarantee::PLManifoldStrict,
            VisualizationTopologyGuarantee::PLManifoldStrict,
            "PLManifoldStrict",
        ),
    ] {
        let converted = VisualizationTopologyGuarantee::from(source);

        assert_eq!(converted, expected);
        assert_eq!(converted.to_string(), schema);
        assert_eq!(serde_json::to_value(&converted)?, serde_json::json!(schema));
        assert_eq!(
            serde_json::from_value::<VisualizationTopologyGuarantee>(serde_json::json!(schema))?,
            expected
        );
    }

    Ok(())
}

fn assert_validation_error(
    export: &MeshExport<2>,
    expected: VisualizationDataValidationError,
) -> Result<(), MeshExportTestError> {
    let json = serde_json::to_string(&export)?;
    let decoded: MeshExport<2> = serde_json::from_str(&json)?;

    assert_eq!(decoded.validate(), Err(expected));
    Ok(())
}

#[test]
fn mesh_export_validation_rejects_bad_metadata() -> Result<(), MeshExportTestError> {
    let mut export = sample_export()?;
    export.metadata.schema = "not.delaunay".to_owned();
    assert_validation_error(
        &export,
        VisualizationDataValidationError::InvalidSchema {
            expected: MESH_EXPORT_SCHEMA,
            actual: "not.delaunay".to_owned(),
        },
    )?;

    let mut export = sample_export()?;
    export.metadata.schema_version = 0;
    assert_validation_error(
        &export,
        VisualizationDataValidationError::InvalidSchemaVersion {
            expected: MESH_EXPORT_SCHEMA_VERSION,
            actual: 0,
        },
    )?;

    let mut export = sample_export()?;
    export.metadata.dimension = 3;
    assert_validation_error(
        &export,
        VisualizationDataValidationError::DimensionMismatch {
            expected: 2,
            actual: 3,
        },
    )?;

    let mut export = sample_export()?;
    let actual = export.vertices.len();
    export.metadata.vertex_count = actual + 1;
    assert_validation_error(
        &export,
        VisualizationDataValidationError::VertexCountMismatch {
            expected: actual + 1,
            actual,
        },
    )?;

    let mut export = sample_export()?;
    let actual = export.simplices.len();
    export.metadata.simplex_count = actual + 1;
    assert_validation_error(
        &export,
        VisualizationDataValidationError::SimplexCountMismatch {
            expected: actual + 1,
            actual,
        },
    )?;

    let mut export = sample_export()?;
    export.metadata.topology_kind = VisualizationTopologyKind::Unknown {
        actual: "Cartesian".to_owned(),
    };
    assert_validation_error(
        &export,
        VisualizationDataValidationError::InvalidTopologyKind {
            actual: VisualizationTopologyKind::Unknown {
                actual: "Cartesian".to_owned(),
            },
        },
    )?;

    let mut export = sample_export()?;
    export.metadata.topology_guarantee = VisualizationTopologyGuarantee::Unknown {
        actual: "Triangulation".to_owned(),
    };
    assert_validation_error(
        &export,
        VisualizationDataValidationError::InvalidTopologyGuarantee {
            actual: VisualizationTopologyGuarantee::Unknown {
                actual: "Triangulation".to_owned(),
            },
        },
    )
}

#[test]
fn mesh_export_validation_rejects_bad_vertex_arity() -> Result<(), MeshExportTestError> {
    let mut export = sample_export()?;
    let vertex_id = export.vertices[0].id;
    export.vertices[0].coordinates.pop();
    assert_validation_error(
        &export,
        VisualizationDataValidationError::InvalidVertexCoordinateCount {
            vertex_id,
            expected: 2,
            actual: 1,
        },
    )
}

#[test]
fn mesh_export_validation_rejects_non_finite_coordinates() -> Result<(), MeshExportTestError> {
    for (coordinate, expected_value) in [
        (f64::NAN, InvalidCoordinateValue::Nan),
        (f64::INFINITY, InvalidCoordinateValue::PositiveInfinity),
        (f64::NEG_INFINITY, InvalidCoordinateValue::NegativeInfinity),
    ] {
        let mut export = sample_export()?;
        let vertex_id = export.vertices[0].id;
        export.vertices[0].coordinates[1] = coordinate;

        assert_eq!(
            export.into_validated(),
            Err(
                VisualizationDataValidationError::InvalidVertexCoordinateValue {
                    vertex_id,
                    coordinate_index: 1,
                    value: expected_value,
                }
            )
        );
    }

    Ok(())
}

#[test]
fn mesh_export_validation_rejects_nil_vertex_ids() -> Result<(), MeshExportTestError> {
    let mut export = sample_export()?;
    export.vertices[0].id = Uuid::nil();

    assert_validation_error(
        &export,
        VisualizationDataValidationError::NilVertexId { vertex_index: 0 },
    )
}

#[test]
fn mesh_export_validation_rejects_duplicate_vertex_ids() -> Result<(), MeshExportTestError> {
    let mut export = sample_export()?;
    let vertex_id = export.vertices[0].id;
    export.vertices[1].id = vertex_id;
    assert_validation_error(
        &export,
        VisualizationDataValidationError::DuplicateVertexId { vertex_id },
    )
}

#[test]
fn mesh_export_validation_rejects_bad_simplex_arity() -> Result<(), MeshExportTestError> {
    let mut export = sample_export()?;
    let simplex_id = export.simplices[0].id;
    export.simplices[0].vertex_ids.pop();
    assert_validation_error(
        &export,
        VisualizationDataValidationError::InvalidSimplexVertexCount {
            simplex_id,
            expected: 3,
            actual: 2,
        },
    )
}

#[test]
fn mesh_export_validation_rejects_nil_simplex_ids() -> Result<(), MeshExportTestError> {
    let mut export = sample_export()?;
    export.simplices[0].id = Uuid::nil();

    assert_validation_error(
        &export,
        VisualizationDataValidationError::NilSimplexId { simplex_index: 0 },
    )
}

#[test]
fn mesh_export_validation_rejects_duplicate_simplex_ids() -> Result<(), MeshExportTestError> {
    let mut export = sample_export()?;
    let simplex_id = export.simplices[0].id;
    export.simplices[1].id = simplex_id;
    assert_validation_error(
        &export,
        VisualizationDataValidationError::DuplicateSimplexId { simplex_id },
    )
}

#[test]
fn mesh_export_validation_rejects_duplicate_simplex_vertices() -> Result<(), MeshExportTestError> {
    let mut export = sample_export()?;
    let simplex_id = export.simplices[0].id;
    let vertex_id = export.simplices[0].vertex_ids[0];
    export.simplices[0].vertex_ids[1] = vertex_id;
    assert_validation_error(
        &export,
        VisualizationDataValidationError::DuplicateSimplexVertex {
            simplex_id,
            vertex_id,
        },
    )
}

#[test]
fn mesh_export_validation_rejects_missing_referenced_ids() -> Result<(), MeshExportTestError> {
    let mut export = sample_export()?;
    let missing_vertex_id = Uuid::nil();
    let simplex_id = export.simplices[0].id;
    export.simplices[0].vertex_ids[0] = missing_vertex_id;
    assert_validation_error(
        &export,
        VisualizationDataValidationError::MissingSimplexVertex {
            simplex_id,
            vertex_id: missing_vertex_id,
        },
    )?;

    let mut export = sample_export()?;
    let missing_neighbor_id = Uuid::nil();
    let simplex_id = export.adjacency[0].simplex_id;
    let facet_index = export.adjacency[0].facet_index;
    export.adjacency[0].neighbor_simplex_id = Some(missing_neighbor_id);
    assert_validation_error(
        &export,
        VisualizationDataValidationError::MissingAdjacencyNeighbor {
            simplex_id,
            facet_index,
            neighbor_simplex_id: missing_neighbor_id,
        },
    )
}

#[test]
fn mesh_export_validation_rejects_missing_adjacency_sources() -> Result<(), MeshExportTestError> {
    let mut export = sample_export()?;
    let simplex_id = Uuid::nil();
    export.adjacency[0].simplex_id = simplex_id;
    assert_validation_error(
        &export,
        VisualizationDataValidationError::MissingAdjacencySimplex { simplex_id },
    )
}

#[test]
fn mesh_export_validation_rejects_invalid_facet_indices() -> Result<(), MeshExportTestError> {
    let mut export = sample_export()?;
    let simplex_id = export.adjacency[0].simplex_id;
    export.adjacency[0].facet_index = 3;
    assert_validation_error(
        &export,
        VisualizationDataValidationError::InvalidAdjacencyFacetIndex {
            simplex_id,
            facet_index: 3,
            max_exclusive: 3,
        },
    )
}

#[test]
fn mesh_export_validation_rejects_duplicate_adjacency() -> Result<(), MeshExportTestError> {
    let mut export = sample_export()?;
    let duplicate = export.adjacency[0].clone();
    let simplex_id = duplicate.simplex_id;
    let facet_index = duplicate.facet_index;
    export.adjacency.push(duplicate);
    assert_validation_error(
        &export,
        VisualizationDataValidationError::DuplicateAdjacency {
            simplex_id,
            facet_index,
        },
    )
}

#[test]
fn mesh_export_validation_rejects_missing_adjacency() -> Result<(), MeshExportTestError> {
    let mut export = sample_export()?;
    let removed = export.adjacency.remove(0);
    assert_validation_error(
        &export,
        VisualizationDataValidationError::MissingAdjacency {
            simplex_id: removed.simplex_id,
            facet_index: removed.facet_index,
        },
    )
}

#[test]
fn mesh_export_validation_rejects_wrong_facet_adjacency() -> Result<(), MeshExportTestError> {
    let mut export = sample_export()?;
    let source_index = export
        .adjacency
        .iter()
        .position(|record| record.neighbor_simplex_id.is_some())
        .expect("sample export should contain an interior neighbor");
    let source_simplex_id = export.adjacency[source_index].simplex_id;
    let source_facet_index = export.adjacency[source_index].facet_index;
    let neighbor_simplex_id = export.adjacency[source_index]
        .neighbor_simplex_id
        .expect("source record should reference a neighbor");
    let (wrong_facet_index, missing_vertex_id) = {
        let source_simplex = export
            .simplices
            .iter()
            .find(|simplex| simplex.id == source_simplex_id)
            .expect("source simplex should exist");
        let neighbor_simplex = export
            .simplices
            .iter()
            .find(|simplex| simplex.id == neighbor_simplex_id)
            .expect("neighbor simplex should exist");
        source_simplex
            .vertex_ids
            .iter()
            .enumerate()
            .filter(|(facet_index, _)| *facet_index != source_facet_index)
            .find_map(|(facet_index, _)| {
                let missing_vertex_id = source_simplex
                    .vertex_ids
                    .iter()
                    .enumerate()
                    .filter(|(vertex_index, _)| *vertex_index != facet_index)
                    .map(|(_, vertex_id)| *vertex_id)
                    .find(|vertex_id| !neighbor_simplex.vertex_ids.contains(vertex_id))?;
                Some((facet_index, missing_vertex_id))
            })
            .expect("sample source should have a facet not shared by the same neighbor")
    };
    let swapped_index = export
        .adjacency
        .iter()
        .position(|record| {
            record.simplex_id == source_simplex_id && record.facet_index == wrong_facet_index
        })
        .expect("sample export should contain every source facet");
    export.adjacency[swapped_index].facet_index = source_facet_index;
    export.adjacency[source_index].facet_index = wrong_facet_index;

    assert_validation_error(
        &export,
        VisualizationDataValidationError::InvalidAdjacencyFacetSharing {
            simplex_id: source_simplex_id,
            facet_index: wrong_facet_index,
            neighbor_simplex_id,
            missing_vertex_id,
        },
    )
}

#[test]
fn mesh_export_validation_rejects_asymmetric_adjacency() -> Result<(), MeshExportTestError> {
    let mut export = sample_export()?;
    let source = export
        .adjacency
        .iter()
        .find(|record| record.neighbor_simplex_id.is_some())
        .expect("sample export should contain an interior neighbor");
    let source_simplex_id = source.simplex_id;
    let source_facet_index = source.facet_index;
    let neighbor_simplex_id = source
        .neighbor_simplex_id
        .expect("source record should reference a neighbor");
    let reciprocal = export
        .adjacency
        .iter_mut()
        .find(|record| {
            record.simplex_id == neighbor_simplex_id
                && record.neighbor_simplex_id == Some(source_simplex_id)
        })
        .expect("sample export should contain the reciprocal neighbor link");
    reciprocal.neighbor_simplex_id = None;

    assert_validation_error(
        &export,
        VisualizationDataValidationError::AsymmetricAdjacency {
            simplex_id: source_simplex_id,
            facet_index: source_facet_index,
            neighbor_simplex_id,
        },
    )
}

#[test]
fn mesh_export_validation_rejects_one_sided_self_neighbor() -> Result<(), MeshExportTestError> {
    let mut export = sample_export()?;
    let boundary = export
        .adjacency
        .iter_mut()
        .find(|record| record.neighbor_simplex_id.is_none())
        .expect("sample export should contain a boundary facet");
    let simplex_id = boundary.simplex_id;
    let facet_index = boundary.facet_index;
    boundary.neighbor_simplex_id = Some(simplex_id);

    assert_validation_error(
        &export,
        VisualizationDataValidationError::AsymmetricAdjacency {
            simplex_id,
            facet_index,
            neighbor_simplex_id: simplex_id,
        },
    )
}

#[test]
fn mesh_export_validation_accepts_reciprocal_self_neighbor() -> Result<(), MeshExportTestError> {
    let export = reciprocal_self_neighbor_export();

    let validated = export.into_validated()?;

    assert_eq!(
        validated
            .adjacency()
            .iter()
            .filter(|record| record.neighbor_simplex_id == Some(record.simplex_id))
            .count(),
        2
    );
    Ok(())
}

#[test]
fn delaunay_result_accepts_mesh_export_error_families() -> DelaunayResult<()> {
    let mut export = sample_delaunay_result_export()?;
    export.validate()?;

    export.metadata.schema_version = 0;
    let expected_validation_error = VisualizationDataValidationError::InvalidSchemaVersion {
        expected: MESH_EXPORT_SCHEMA_VERSION,
        actual: 0,
    };
    let validation_error = export.validate().expect_err("schema version should fail");
    assert_eq!(validation_error, expected_validation_error);
    let err = DelaunayError::from(validation_error);
    let source = StdError::source(&err)
        .and_then(|source| source.downcast_ref::<Box<VisualizationDataValidationError>>())
        .expect("DelaunayError should expose the typed boxed visualization validation source");
    assert_eq!(source.as_ref(), &expected_validation_error);
    assert_matches!(&err, DelaunayError::VisualizationDataValidation { .. });

    let export_error = VisualizationExportError::UnassignedNeighborBuffer {
        simplex_id: Uuid::nil(),
    };
    let err = DelaunayError::from(export_error.clone());
    let source = StdError::source(&err)
        .and_then(|source| source.downcast_ref::<Box<VisualizationExportError>>())
        .expect("DelaunayError should expose the typed boxed visualization export source");
    assert_eq!(source.as_ref(), &export_error);
    assert_matches!(&err, DelaunayError::VisualizationExport { .. });

    Ok(())
}

#[test]
fn mesh_export_ids_match_triangulation_uuids() -> Result<(), MeshExportTestError> {
    let vertices = vec![
        vertex![0.0, 0.0, 0.0]?,
        vertex![1.0, 0.0, 0.0]?,
        vertex![0.0, 1.0, 0.0]?,
        vertex![0.0, 0.0, 1.0]?,
    ];
    let triangulation = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    let export = triangulation.to_mesh_export()?;

    let exported_vertices: HashMap<_, _> = export
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.coordinates.as_slice()))
        .collect();
    for (_, vertex) in triangulation.vertices() {
        assert_eq!(
            exported_vertices.get(&vertex.uuid()).copied(),
            Some(vertex.point().coords().as_slice())
        );
    }

    let exported_simplices: HashSet<_> =
        export.simplices.iter().map(|simplex| simplex.id).collect();
    for (_, simplex) in triangulation.simplices() {
        assert!(exported_simplices.contains(&simplex.uuid()));
    }

    Ok(())
}

#[test]
fn visualization_records_support_downstream_attributes() {
    assert_send_sync_unpin::<VisualizationData<3, u8, u16, u32, u64>>();
    assert_send_sync_unpin::<ValidatedVisualizationData<3, u8, u16, u32, u64>>();
    assert_send_sync_unpin::<VertexRecord<3, &'static str>>();
    assert_send_sync_unpin::<SimplexRecord<&'static str>>();
    assert_send_sync_unpin::<AdjacencyRecord<&'static str>>();
    assert_send_sync_unpin::<VisualizationMetadata<&'static str>>();
}

#[test]
fn visualization_validation_does_not_require_attribute_traits() -> Result<(), MeshExportTestError> {
    let VisualizationData {
        metadata,
        vertices,
        simplices,
        adjacency,
    } = sample_export()?;

    let export: VisualizationData<
        2,
        OpaqueAttribute,
        OpaqueAttribute,
        OpaqueAttribute,
        OpaqueAttribute,
    > = VisualizationData {
        metadata: VisualizationMetadata {
            schema: metadata.schema,
            schema_version: metadata.schema_version,
            producer: metadata.producer,
            dimension: metadata.dimension,
            vertex_count: metadata.vertex_count,
            simplex_count: metadata.simplex_count,
            topology_kind: metadata.topology_kind,
            topology_guarantee: metadata.topology_guarantee,
            attributes: None,
        },
        vertices: vertices
            .into_iter()
            .map(|vertex| VertexRecord {
                id: vertex.id,
                coordinates: vertex.coordinates,
                attributes: None,
            })
            .collect(),
        simplices: simplices
            .into_iter()
            .map(|simplex| SimplexRecord {
                id: simplex.id,
                vertex_ids: simplex.vertex_ids,
                attributes: None,
            })
            .collect(),
        adjacency: adjacency
            .into_iter()
            .map(|record| AdjacencyRecord {
                simplex_id: record.simplex_id,
                facet_index: record.facet_index,
                neighbor_simplex_id: record.neighbor_simplex_id,
                attributes: None,
            })
            .collect(),
    };

    export.validate()?;
    Ok(())
}

#[test]
fn visualization_data_into_validated_carries_validation_proof() -> Result<(), MeshExportTestError> {
    let export = sample_export()?;
    let validated: ValidatedMeshExport<2> = export.clone().into_validated()?;

    assert_eq!(validated.as_raw(), &export);
    assert_eq!(validated.metadata().schema, MESH_EXPORT_SCHEMA);
    assert_eq!(validated.vertices().len(), export.vertices.len());
    assert_eq!(validated.simplices().len(), export.simplices.len());
    assert_eq!(validated.adjacency().len(), export.adjacency.len());
    assert_eq!(
        serde_json::to_value(&validated)?,
        serde_json::to_value(&export)?
    );
    assert_eq!(validated.into_raw(), export);

    Ok(())
}

#[test]
fn visualization_data_try_from_carries_validation_proof() -> Result<(), MeshExportTestError> {
    let export = sample_export()?;
    let validated = ValidatedMeshExport::<2>::try_from(export.clone())?;

    assert_eq!(validated.as_raw(), &export);

    let mut invalid = export;
    let actual = invalid.simplices.len();
    invalid.metadata.simplex_count = actual + 1;
    assert_eq!(
        ValidatedMeshExport::<2>::try_from(invalid),
        Err(VisualizationDataValidationError::SimplexCountMismatch {
            expected: actual + 1,
            actual,
        })
    );

    Ok(())
}

#[test]
fn visualization_data_into_validated_rejects_invalid_raw_dto() -> Result<(), MeshExportTestError> {
    let mut export = sample_export()?;
    export.metadata.schema_version = 0;

    assert_eq!(
        export.into_validated(),
        Err(VisualizationDataValidationError::InvalidSchemaVersion {
            expected: MESH_EXPORT_SCHEMA_VERSION,
            actual: 0,
        })
    );

    Ok(())
}

#[test]
fn visualization_data_into_validated_canonicalizes_record_order() -> Result<(), MeshExportTestError>
{
    let canonical = sample_export()?;
    let mut raw = canonical.clone();
    raw.vertices.reverse();
    raw.simplices.reverse();
    raw.adjacency.reverse();

    let validated = raw.into_validated()?;

    assert_eq!(validated.as_raw(), &canonical);
    Ok(())
}

#[test]
fn visualization_attributes_deserialize_without_default_bound() -> Result<(), MeshExportTestError> {
    let json = serde_json::to_string(&sample_export()?)?;
    let decoded: VisualizationData<
        2,
        NonDefaultAttribute,
        NonDefaultAttribute,
        NonDefaultAttribute,
        NonDefaultAttribute,
    > = serde_json::from_str(&json)?;

    let validated = decoded.into_validated()?;
    assert!(validated.metadata().attributes.is_none());
    assert!(
        validated
            .vertices()
            .iter()
            .all(|vertex| vertex.attributes.is_none())
    );
    assert!(
        validated
            .simplices()
            .iter()
            .all(|simplex| simplex.attributes.is_none())
    );
    assert!(
        validated
            .adjacency()
            .iter()
            .all(|adjacency| adjacency.attributes.is_none())
    );
    Ok(())
}
