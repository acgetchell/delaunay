#![forbid(unsafe_code)]

//! # Payloads, Secondary Maps, and Serialization
//!
//! This example stores payloads in vertices and simplices, keeps algorithm-local
//! state in secondary maps, and reconstructs a validated triangulation from a
//! JSON-serialized TDS snapshot.
//!
//! Run it with:
//!
//! ```bash
//! cargo run --release --example data_and_serialization
//! ```

use delaunay::prelude::collections::{SimplexSecondaryMap, VertexSecondaryMap};
use delaunay::prelude::construction::{
    DelaunayError, DelaunayTriangulation, DelaunayTriangulationBuilder,
    DelaunayTriangulationConstructionError, vertex,
};
use delaunay::prelude::geometry::{AdaptiveKernel, CoordinateConversionError};
use delaunay::prelude::query::{
    JaccardComputationError, extract_vertex_coordinate_set, jaccard_index,
};
use delaunay::prelude::tds::{Tds, TdsMutationError};
use delaunay::prelude::validation::DelaunayTriangulationValidationError;

type LabeledTriangulation = DelaunayTriangulation<AdaptiveKernel<f64>, i32, i32, 2>;

#[derive(Debug, thiserror::Error)]
enum DataExampleError {
    #[error(transparent)]
    Delaunay(#[from] DelaunayError),
    #[error(transparent)]
    Serde(#[from] serde_json::Error),
    #[error(transparent)]
    Jaccard(#[from] JaccardComputationError),
    #[error(transparent)]
    Coordinate(#[from] CoordinateConversionError),
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Mutation(#[from] TdsMutationError),
    #[error(transparent)]
    Validation(#[from] DelaunayTriangulationValidationError),
    #[error("constructed triangulation unexpectedly contains no vertices")]
    MissingVertex,
    #[error("constructed triangulation unexpectedly contains no simplices")]
    MissingSimplex,
}

/// Demonstrates owned payloads, detached maps, and checked reconstruction.
fn main() -> Result<(), DataExampleError> {
    let triangulation = build_labeled_triangulation()?;
    let coordinates_before = extract_vertex_coordinate_set(triangulation.as_triangulation());

    let mut vertex_order = VertexSecondaryMap::new();
    for (order, (vertex_key, _)) in triangulation.vertices().enumerate() {
        vertex_order.insert(vertex_key, order);
    }
    let mut visited_simplices = SimplexSecondaryMap::new();
    for (simplex_key, _) in triangulation.simplices() {
        visited_simplices.insert(simplex_key, false);
    }

    println!("Detached algorithm state:");
    println!("  vertex-order entries: {}", vertex_order.len());
    println!("  simplex-visit entries: {}", visited_simplices.len());

    let json = serde_json::to_string_pretty(&triangulation)?;
    let tds: Tds<i32, i32, 2> = serde_json::from_str(&json)?;
    let roundtrip = DelaunayTriangulation::try_from_tds(tds, AdaptiveKernel::new())
        .map_err(DelaunayError::from)?;
    roundtrip.validate().map_err(DelaunayError::from)?;
    let coordinates_after = extract_vertex_coordinate_set(roundtrip.as_triangulation());
    let coordinate_similarity = jaccard_index(&coordinates_before, &coordinates_after)?;

    let mut labels = roundtrip
        .vertices()
        .filter_map(|(_, vertex)| vertex.data().copied())
        .collect::<Vec<_>>();
    labels.sort_unstable();
    let labeled_simplices = roundtrip
        .simplices()
        .filter(|(_, simplex)| simplex.data().is_some())
        .count();

    println!("\nValidated JSON round-trip:");
    println!("  bytes: {}", json.len());
    println!("  vertex labels: {labels:?}");
    println!("  labeled simplices: {labeled_simplices}");
    println!("  coordinate Jaccard similarity: {coordinate_similarity:.3}");
    Ok(())
}

/// Builds a small triangulation and mutates payloads through checked keys.
fn build_labeled_triangulation() -> Result<LabeledTriangulation, DataExampleError> {
    let vertices = [
        vertex![0.0, 0.0; data = 10]?,
        vertex![1.0, 0.0; data = 20]?,
        vertex![0.0, 1.0; data = 30]?,
        vertex![0.25, 0.25; data = 40]?,
    ];
    let mut triangulation: LabeledTriangulation = DelaunayTriangulationBuilder::new(&vertices)
        .simplex_data_type::<i32>()
        .build()?;

    let Some(vertex_key) = triangulation.vertices().next().map(|(key, _)| key) else {
        return Err(DataExampleError::MissingVertex);
    };
    triangulation.set_vertex_data(vertex_key, Some(99))?;

    let Some(simplex_key) = triangulation.simplices().next().map(|(key, _)| key) else {
        return Err(DataExampleError::MissingSimplex);
    };
    triangulation.set_simplex_data(simplex_key, Some(42))?;
    triangulation.validate()?;
    Ok(triangulation)
}
