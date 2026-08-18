#![forbid(unsafe_code)]

//! # Payloads, Secondary Maps, and Serialization
//!
//! This example stores payloads in vertices and simplices, keeps algorithm-local
//! state in secondary maps, and reconstructs a Levels 1–4 [`Triangulation`] from
//! a JSON-serialized TDS snapshot. It then demonstrates the two distinct Level
//! 5 boundaries: strict no-repair certification and consuming `delaunayize`
//! conversion.
//!
//! Run it with:
//!
//! ```bash
//! cargo run --release --example data_and_serialization
//! ```

use approx::assert_abs_diff_eq;
use delaunay::prelude::collections::{SimplexSecondaryMap, VertexSecondaryMap};
use delaunay::prelude::construction::{
    ConstructionOptions, DelaunayTriangulation, DelaunayTriangulationBuilder,
    DelaunayTriangulationConstructionError, vertex,
};
use delaunay::prelude::delaunayize::{DelaunayizeConfig, DelaunayizeError, delaunayize};
use delaunay::prelude::geometry::{AdaptiveKernel, CoordinateConversionError};
use delaunay::prelude::query::{
    JaccardComputationError, extract_vertex_coordinate_set, jaccard_index,
};
use delaunay::prelude::tds::{Tds, TdsMutationError};
use delaunay::prelude::triangulation::{Triangulation, TriangulationRealizationValidationError};
use delaunay::prelude::validation::{
    DelaunayTriangulationValidationError, DelaunayValidationError,
};

type LabeledTriangulation = Triangulation<AdaptiveKernel<f64>, i32, i32, 2>;

#[derive(Debug, thiserror::Error)]
enum DataExampleError {
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
    #[error(transparent)]
    RealizationValidation(#[from] TriangulationRealizationValidationError),
    #[error(transparent)]
    DelaunayProperty(#[from] DelaunayValidationError),
    #[error(transparent)]
    Delaunayize(Box<DelaunayizeError>),
    #[error("constructed triangulation unexpectedly contains no vertices")]
    MissingVertex,
    #[error("constructed triangulation unexpectedly contains no simplices")]
    MissingSimplex,
    #[error("the explicit non-Delaunay fixture unexpectedly passed strict Level 5 certification")]
    UnexpectedStrictCertification,
}

impl From<DelaunayizeError> for DataExampleError {
    fn from(error: DelaunayizeError) -> Self {
        Self::Delaunayize(Box::new(error))
    }
}

/// Demonstrates payloads, detached maps, checked restoration, and Level 5 conversion.
fn main() -> Result<(), DataExampleError> {
    let triangulation = build_labeled_triangulation()?;
    let coordinates_before = extract_vertex_coordinate_set(&triangulation);
    let topology_guarantee = triangulation.topology_guarantee();
    let global_topology = triangulation.global_topology();
    let labels_before = sorted_vertex_labels(&triangulation);
    let labeled_simplices_before = labeled_simplex_count(&triangulation);

    // Secondary maps are intentionally detached algorithm state. Persist them
    // separately if a workflow needs them beyond this in-memory owner.
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

    // Tds is the transport/storage DTO. The runtime topology context is kept
    // alongside it so the same Levels 1–4 contract can be restored.
    let tds = triangulation.into_tds();
    let json = serde_json::to_string_pretty(&tds)?;
    let tds: Tds<i32, i32, 2> = serde_json::from_str(&json)?;
    let restored = Triangulation::try_from_tds_with_topology_context(
        tds,
        AdaptiveKernel::new(),
        topology_guarantee,
        global_topology,
    )?;
    restored.validate_realization()?;

    let coordinates_after = extract_vertex_coordinate_set(&restored);
    let coordinate_similarity = jaccard_index(&coordinates_before, &coordinates_after)?;
    let labels_after = sorted_vertex_labels(&restored);
    let labeled_simplices_after = labeled_simplex_count(&restored);
    assert_abs_diff_eq!(coordinate_similarity, 1.0);
    assert_eq!(labels_after, labels_before);
    assert_eq!(labeled_simplices_after, labeled_simplices_before);

    let Err(strict_error) = DelaunayTriangulation::try_from_triangulation(restored.clone()) else {
        return Err(DataExampleError::UnexpectedStrictCertification);
    };
    let converted = delaunayize(restored, DelaunayizeConfig::default())?;
    converted.triangulation.validate()?;
    let repaired_coordinates =
        extract_vertex_coordinate_set(converted.triangulation.as_triangulation());
    let repaired_similarity = jaccard_index(&coordinates_before, &repaired_coordinates)?;
    let repaired_labels = sorted_vertex_labels(converted.triangulation.as_triangulation());
    assert_abs_diff_eq!(repaired_similarity, 1.0);
    assert_eq!(repaired_labels, labels_before);

    println!("\nValidated Levels 1-4 JSON round-trip:");
    println!("  bytes: {}", json.len());
    println!("  vertex labels: {labels_after:?}");
    println!("  labeled simplices: {labeled_simplices_after}");
    println!("  coordinate Jaccard similarity: {coordinate_similarity:.3}");
    println!("\nLevel 5 choices:");
    println!("  strict certification rejected the stored diagonal: {strict_error}");
    println!(
        "  delaunayize used fallback rebuild: {}",
        converted.outcome.used_fallback_rebuild
    );
    println!("  repaired coordinate similarity: {repaired_similarity:.3}");
    Ok(())
}

/// Builds a small triangulation and mutates payloads through checked keys.
fn build_labeled_triangulation() -> Result<LabeledTriangulation, DataExampleError> {
    let vertices = vec![
        vertex![0.0, 0.0; data = 10]?,
        vertex![4.0, 0.0; data = 20]?,
        vertex![4.0, 2.0; data = 30]?,
        vertex![1.0, 2.0; data = 40]?,
    ];
    let simplices = vec![vec![0, 1, 2], vec![0, 2, 3]];
    let mut triangulation: LabeledTriangulation =
        DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
            .map_err(DelaunayTriangulationConstructionError::from)?
            .simplex_data_type::<i32>()
            .construction_options(
                ConstructionOptions::default().without_final_delaunay_enforcement(),
            )
            .build_triangulation()?;

    let Some(vertex_key) = triangulation.vertices().next().map(|(key, _)| key) else {
        return Err(DataExampleError::MissingVertex);
    };
    triangulation.set_vertex_data(vertex_key, Some(99))?;

    let Some(simplex_key) = triangulation.simplices().next().map(|(key, _)| key) else {
        return Err(DataExampleError::MissingSimplex);
    };
    triangulation.set_simplex_data(simplex_key, Some(42))?;
    triangulation.validate_realization()?;
    assert!(!triangulation.delaunay_violation_report(None)?.is_valid());
    Ok(triangulation)
}

/// Returns sorted owned vertex payloads for round-trip comparisons.
fn sorted_vertex_labels(triangulation: &LabeledTriangulation) -> Vec<i32> {
    let mut labels = triangulation
        .vertices()
        .filter_map(|(_, vertex)| vertex.data().copied())
        .collect::<Vec<_>>();
    labels.sort_unstable();
    labels
}

/// Counts simplex payloads that remain applicable to the current connectivity.
fn labeled_simplex_count(triangulation: &LabeledTriangulation) -> usize {
    triangulation
        .simplices()
        .filter(|(_, simplex)| simplex.data().is_some())
        .count()
}
