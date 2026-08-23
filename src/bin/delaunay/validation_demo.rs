#![forbid(unsafe_code)]

//! Deterministic validation-layer demonstration artifacts for the CLI.

use std::{fmt, path::PathBuf};

use clap::Args;
use delaunay::{
    DelaunayRefinementBuilder,
    prelude::{
        construction::{
            DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
            ExplicitConstructionError, Vertex, vertex,
        },
        geometry::Point,
        validation::DelaunayTriangulationValidationError,
    },
};
use serde::Serialize;

use crate::{
    cli_output::{ArtifactPath, write_json_output},
    config::CliError,
};

const VALIDATION_DEMO_EXPORT_SCHEMA: &str = "delaunay.validation_demo";
const VALIDATION_DEMO_EXPORT_SCHEMA_VERSION: u32 = 1;

/// Stable identity of a generated validation-demo case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ValidationDemoCaseId {
    /// Passing explicit Delaunay triangle used as the baseline.
    ValidBaseline,
    /// Level 1 non-finite coordinate example.
    Level1NonFiniteCoordinate,
    /// Level 2 duplicate maximal-simplex example.
    Level2DuplicateSimplex,
    /// Level 3 isolated-vertex example.
    Level3IsolatedVertex,
    /// Level 4 degenerate Euclidean realization example.
    Level4InvalidEuclideanRealization,
    /// Level 5 non-Delaunay diagonal example.
    Level5NonDelaunayDiagonal,
}

impl fmt::Display for ValidationDemoCaseId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::ValidBaseline => "valid baseline",
            Self::Level1NonFiniteCoordinate => "Level 1 non-finite coordinate",
            Self::Level2DuplicateSimplex => "Level 2 duplicate simplex",
            Self::Level3IsolatedVertex => "Level 3 isolated vertex",
            Self::Level4InvalidEuclideanRealization => "Level 4 invalid Euclidean realization",
            Self::Level5NonDelaunayDiagonal => "Level 5 non-Delaunay diagonal",
        })
    }
}

/// Boundary at which a validation-demo case is expected to fail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ValidationDemoExpectedFailure {
    /// Level 1 coordinate validation.
    ElementValidation,
    /// Explicit Delaunay construction and cumulative validation.
    ExplicitDelaunayConstruction,
    /// Strict Level 5 Delaunay certification.
    StrictDelaunayCertification,
}

impl fmt::Display for ValidationDemoExpectedFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::ElementValidation => "Level 1 element validation",
            Self::ExplicitDelaunayConstruction => "explicit Delaunay construction",
            Self::StrictDelaunayCertification => "strict Level 5 Delaunay certification",
        })
    }
}

/// Typed failures while generating deterministic validation demonstrations.
#[derive(Clone, Debug, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum ValidationDemoError {
    /// Raw explicit connectivity failed before the intended validation boundary.
    #[error("validation demo case {case} failed while parsing explicit connectivity: {source}")]
    ExplicitInput {
        /// Demo case being generated.
        case: ValidationDemoCaseId,
        /// Typed explicit-connectivity parse failure.
        #[source]
        source: ExplicitConstructionError,
    },
    /// A demo case failed during Delaunay construction.
    #[error("validation demo case {case} failed during Delaunay construction: {source}")]
    Construction {
        /// Demo case being generated.
        case: ValidationDemoCaseId,
        /// Typed Delaunay-construction failure.
        #[source]
        source: DelaunayTriangulationConstructionError,
    },
    /// The valid baseline failed cumulative Delaunay validation.
    #[error("validation demo case {case} failed cumulative Delaunay validation: {source}")]
    Validation {
        /// Demo case being generated.
        case: ValidationDemoCaseId,
        /// Typed cumulative validation failure.
        #[source]
        source: DelaunayTriangulationValidationError,
    },
    /// A case unexpectedly passed the boundary intended to reject it.
    #[error("validation demo case {case} unexpectedly passed {expected}")]
    UnexpectedSuccess {
        /// Demo case being generated.
        case: ValidationDemoCaseId,
        /// Boundary that should have rejected the case.
        expected: ValidationDemoExpectedFailure,
    },
}

/// Raw command-line arguments for `delaunay validation-demo`.
#[derive(Debug, Args)]
pub struct ValidationDemoArgs {
    /// Write JSON to a file instead of stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,
}

impl ValidationDemoArgs {
    /// Convert raw validation-demo arguments into a passive artifact config.
    pub fn into_validated(self) -> Result<ValidationDemoConfig, CliError> {
        Ok(ValidationDemoConfig {
            output: self.output.map(ArtifactPath::try_new).transpose()?,
        })
    }
}

/// Validated configuration for the generated validation-model demo artifact.
#[derive(Debug)]
pub struct ValidationDemoConfig {
    pub output: Option<ArtifactPath>,
}

/// Notebook-facing validation-model artifact generated by public failure paths.
#[derive(Debug, Serialize)]
pub struct ValidationDemoExport {
    schema: &'static str,
    schema_version: u32,
    dimension: usize,
    valid_baseline: ValidationDemoCase,
    cases: Vec<ValidationDemoCase>,
}

/// One validation-level example with diagnostic text and renderable geometry.
#[derive(Debug, Serialize)]
struct ValidationDemoCase {
    level: u8,
    layer: &'static str,
    title: &'static str,
    status: &'static str,
    public_check: &'static str,
    public_reference: &'static str,
    input_summary: &'static str,
    explanation: &'static str,
    diagnostic: String,
    visual: ValidationDemoVisual,
}

/// Geometry and emphasis metadata for notebook-generated validation figures.
#[derive(Debug, Serialize)]
struct ValidationDemoVisual {
    points: Vec<ValidationDemoPoint>,
    simplices: Vec<Vec<usize>>,
    highlighted_simplices: Vec<usize>,
    highlighted_edges: Vec<[usize; 2]>,
    invalid_points: Vec<usize>,
    isolated_points: Vec<usize>,
    duplicate_simplices: Vec<Vec<usize>>,
    circumcircle: Option<ValidationDemoCircle>,
}

/// One labeled 2D point in a validation demo visual.
#[derive(Debug, Serialize)]
struct ValidationDemoPoint {
    label: &'static str,
    coordinates: [f64; 2],
}

/// Circumcircle witness for the Level 5 empty-circumsphere example.
#[derive(Debug, Serialize)]
struct ValidationDemoCircle {
    center: [f64; 2],
    radius: f64,
}

/// Generate the validation-model artifact used by the notebook quickstart.
pub fn run(config: &ValidationDemoConfig) -> Result<(), CliError> {
    let export = build_export()?;
    write_json_output(&export, config.output.as_ref())?;
    Ok(())
}

/// Build the deterministic validation examples rendered by the notebook.
pub fn build_export() -> Result<ValidationDemoExport, CliError> {
    Ok(ValidationDemoExport {
        schema: VALIDATION_DEMO_EXPORT_SCHEMA,
        schema_version: VALIDATION_DEMO_EXPORT_SCHEMA_VERSION,
        dimension: 2,
        valid_baseline: baseline_case()?,
        cases: vec![
            level_1_case()?,
            level_2_case()?,
            level_3_case()?,
            level_4_case()?,
            level_5_case()?,
        ],
    })
}

/// Generate a passing explicit triangle used as the visual baseline.
fn baseline_case() -> Result<ValidationDemoCase, CliError> {
    const CASE: ValidationDemoCaseId = ValidationDemoCaseId::ValidBaseline;

    let coordinates = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.866_025_403_784_438_6]];
    let simplices = vec![vec![0, 1, 2]];
    let vertices = demo_vertices(&coordinates)?;
    let dt = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .map_err(|source| ValidationDemoError::ExplicitInput { case: CASE, source })?
        .build()
        .map_err(|source| ValidationDemoError::Construction { case: CASE, source })?;
    dt.validate()
        .map_err(|source| ValidationDemoError::Validation { case: CASE, source })?;

    Ok(ValidationDemoCase {
        level: 0,
        layer: "Valid baseline",
        title: "Passing explicit Delaunay triangle",
        status: "passed",
        public_check: "DelaunayTriangulation::validate",
        public_reference: "tests/triangulation_builder.rs::test_explicit_validate_delaunay_mesh",
        input_summary: "Three non-collinear vertices and one triangle",
        explanation: "This baseline passes the cumulative validation path before the failure rows isolate each layer.",
        diagnostic: format!(
            "validate() passed with {} vertices and {} simplex",
            dt.number_of_vertices(),
            dt.number_of_simplices()
        ),
        visual: demo_visual(coordinates, simplices),
    })
}

/// Generate the Level 1 finite-coordinate failure example.
fn level_1_case() -> Result<ValidationDemoCase, CliError> {
    const CASE: ValidationDemoCaseId = ValidationDemoCaseId::Level1NonFiniteCoordinate;

    let diagnostic = match Point::<2>::try_new([f64::NAN, 0.0]) {
        Ok(_) => {
            return Err(ValidationDemoError::UnexpectedSuccess {
                case: CASE,
                expected: ValidationDemoExpectedFailure::ElementValidation,
            }
            .into());
        }
        Err(error) => error.to_string(),
    };
    let mut visual = demo_visual([[0.0, 0.0]], Vec::new());
    visual.invalid_points.push(0);

    Ok(ValidationDemoCase {
        level: 1,
        layer: "Elements",
        title: "Non-finite point coordinate",
        status: "failed_as_expected",
        public_check: "Point::<2>::try_new",
        public_reference: "src/geometry/point.rs::point_is_valid_f64",
        input_summary: "Point::<2>::try_new([NaN, 0.0])",
        explanation: "Element validation rejects non-finite coordinates before they can enter a vertex, simplex, or TDS.",
        diagnostic,
        visual,
    })
}

/// Generate the Level 2 duplicate-simplex structural failure example.
fn level_2_case() -> Result<ValidationDemoCase, CliError> {
    let coordinates = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let simplices = vec![vec![0, 1, 2], vec![0, 1, 2]];
    let diagnostic = explicit_builder_failure(
        ValidationDemoCaseId::Level2DuplicateSimplex,
        coordinates,
        &simplices,
    )?;
    let mut visual = demo_visual(coordinates, simplices);
    visual.duplicate_simplices.push(vec![0, 1, 2]);
    visual.highlighted_simplices = vec![0, 1];

    Ok(ValidationDemoCase {
        level: 2,
        layer: "Structure",
        title: "Duplicate maximal simplex",
        status: "failed_as_expected",
        public_check: "DelaunayTriangulationBuilder::try_from_vertices_and_simplices(...).build",
        public_reference: "tests/triangulation_builder.rs::test_explicit_error_variant_duplicate_simplices_structural_validation",
        input_summary: "Two copies of simplex [0, 1, 2]",
        explanation: "The TDS layer rejects duplicate maximal simplices because the incidence structure would no longer be a well-defined complex.",
        diagnostic,
        visual,
    })
}

/// Generate the Level 3 isolated-vertex topology failure example.
fn level_3_case() -> Result<ValidationDemoCase, CliError> {
    let coordinates = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.45, 0.85]];
    let simplices = vec![vec![0, 1, 2]];
    let diagnostic = explicit_builder_failure(
        ValidationDemoCaseId::Level3IsolatedVertex,
        coordinates,
        &simplices,
    )?;
    let mut visual = demo_visual(coordinates, simplices);
    visual.isolated_points.push(3);

    Ok(ValidationDemoCase {
        level: 3,
        layer: "Topology",
        title: "Unreferenced vertex",
        status: "failed_as_expected",
        public_check: "Triangulation::is_valid_topology",
        public_reference: "tests/triangulation_builder.rs::test_explicit_unreferenced_vertices_rejected",
        input_summary: "One valid triangle plus vertex D unused by any simplex",
        explanation: "The topology layer rejects isolated vertices because every vertex must belong to the triangulated space.",
        diagnostic,
        visual,
    })
}

/// Generate the Level 4 invalid Euclidean realization failure example.
fn level_4_case() -> Result<ValidationDemoCase, CliError> {
    let coordinates = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
    let simplices = vec![vec![0, 1, 2]];
    let diagnostic = explicit_builder_failure(
        ValidationDemoCaseId::Level4InvalidEuclideanRealization,
        coordinates,
        &simplices,
    )?;
    let mut visual = demo_visual(coordinates, simplices);
    visual.highlighted_simplices = vec![0];
    visual.highlighted_edges.push([0, 2]);

    Ok(ValidationDemoCase {
        level: 4,
        layer: "Valid realization",
        title: "Degenerate realized simplex",
        status: "failed_as_expected",
        public_check: "Triangulation::validate_realization",
        public_reference: "tests/triangulation_builder.rs::test_explicit_error_variant_geometric_nondegeneracy",
        input_summary: "One triangle whose three vertices are collinear",
        explanation: "The Euclidean coordinate realization is invalid because the abstract 2-simplex collapses to zero area.",
        diagnostic,
        visual,
    })
}

/// Generate the Level 5 non-Delaunay diagonal failure example.
fn level_5_case() -> Result<ValidationDemoCase, CliError> {
    let coordinates = [[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [1.0, 2.0]];
    let simplices = vec![vec![0, 1, 2], vec![0, 2, 3]];
    let diagnostic = strict_delaunay_certification_failure(coordinates, &simplices)?;
    let mut visual = demo_visual(coordinates, simplices);
    visual.highlighted_simplices.push(0);
    visual.highlighted_edges.push([0, 2]);
    visual.invalid_points.push(3);
    visual.circumcircle = Some(ValidationDemoCircle {
        center: [2.0, 1.0],
        radius: 5.0_f64.sqrt(),
    });

    Ok(ValidationDemoCase {
        level: 5,
        layer: "Delaunay",
        title: "Interior point in a circumcircle",
        status: "failed_as_expected",
        public_check: "DelaunayRefinementBuilder::new(...).build()",
        public_reference: "tests/triangulation_builder.rs::test_relaxed_explicit_non_delaunay_mesh_succeeds_2d",
        input_summary: "Quadrilateral triangulated with diagonal AC instead of BD",
        explanation: "Point D lies inside the circumcircle of triangle ABC, so the chosen diagonal violates the local Delaunay property.",
        diagnostic,
        visual,
    })
}

/// Return the diagnostic from strict Level 5 certification of valid Levels 1–4 input.
fn strict_delaunay_certification_failure<const N: usize>(
    coordinates: [[f64; 2]; N],
    simplices: &[Vec<usize>],
) -> Result<String, CliError> {
    const CASE: ValidationDemoCaseId = ValidationDemoCaseId::Level5NonDelaunayDiagonal;

    let vertices = demo_vertices(&coordinates)?;
    let triangulation =
        DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, simplices)
            .map_err(|source| ValidationDemoError::ExplicitInput { case: CASE, source })?
            .build_triangulation()
            .map_err(|source| ValidationDemoError::Construction { case: CASE, source })?;

    match DelaunayRefinementBuilder::new(triangulation).build() {
        Ok(_) => Err(ValidationDemoError::UnexpectedSuccess {
            case: CASE,
            expected: ValidationDemoExpectedFailure::StrictDelaunayCertification,
        }
        .into()),
        Err(error) => Ok(stable_diagnostic(&error.to_string())),
    }
}

/// Return the diagnostic from a public explicit-builder case that must fail.
fn explicit_builder_failure<const N: usize>(
    case: ValidationDemoCaseId,
    coordinates: [[f64; 2]; N],
    simplices: &[Vec<usize>],
) -> Result<String, CliError> {
    let vertices = demo_vertices(&coordinates)?;
    let builder =
        DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, simplices)
            .map_err(|source| ValidationDemoError::ExplicitInput { case, source })?;

    match builder.build() {
        Ok(_) => Err(ValidationDemoError::UnexpectedSuccess {
            case,
            expected: ValidationDemoExpectedFailure::ExplicitDelaunayConstruction,
        }
        .into()),
        Err(error) => Ok(stable_diagnostic(&error.to_string())),
    }
}

/// Remove run-specific identifiers from validation-demo diagnostics so paper
/// artifacts are reproducible while preserving the diagnostic shape.
fn stable_diagnostic(diagnostic: &str) -> String {
    let bytes = diagnostic.as_bytes();
    let mut normalized = String::with_capacity(diagnostic.len());
    let mut index = 0;

    while index < bytes.len() {
        if is_uuid_at(bytes, index) {
            normalized.push_str("<uuid>");
            index += 36;
            continue;
        }
        if let Some(character) = diagnostic[index..].chars().next() {
            normalized.push(character);
            index += character.len_utf8();
        } else {
            break;
        }
    }

    normalized
}

/// Detect an ASCII UUID literal at a byte offset in a diagnostic string.
fn is_uuid_at(bytes: &[u8], start: usize) -> bool {
    if start + 36 > bytes.len() {
        return false;
    }
    for offset in 0..36 {
        let byte = bytes[start + offset];
        if matches!(offset, 8 | 13 | 18 | 23) {
            if byte != b'-' {
                return false;
            }
        } else if !byte.is_ascii_hexdigit() {
            return false;
        }
    }
    true
}

/// Convert finite 2D coordinates into vertices for explicit-builder demos.
fn demo_vertices(coordinates: &[[f64; 2]]) -> Result<Vec<Vertex<(), 2>>, CliError> {
    coordinates
        .iter()
        .map(|coords| vertex!(*coords).map_err(|source| CliError::CoordinateConversion { source }))
        .collect()
}

/// Build notebook-renderable visual metadata from case coordinates.
fn demo_visual<const N: usize>(
    coordinates: [[f64; 2]; N],
    simplices: Vec<Vec<usize>>,
) -> ValidationDemoVisual {
    ValidationDemoVisual {
        points: coordinates
            .into_iter()
            .enumerate()
            .map(|(index, coordinates)| ValidationDemoPoint {
                label: match index {
                    0 => "A",
                    1 => "B",
                    2 => "C",
                    3 => "D",
                    4 => "E",
                    _ => "?",
                },
                coordinates,
            })
            .collect(),
        simplices,
        highlighted_simplices: Vec::new(),
        highlighted_edges: Vec::new(),
        invalid_points: Vec::new(),
        isolated_points: Vec::new(),
        duplicate_simplices: Vec::new(),
        circumcircle: None,
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;

    #[test]
    fn cli_error_preserves_validation_demo_source_chain() {
        let error = CliError::from(ValidationDemoError::ExplicitInput {
            case: ValidationDemoCaseId::ValidBaseline,
            source: ExplicitConstructionError::EmptySimplices,
        });

        let cli_source = error
            .source()
            .expect("CLI wrapper should expose the validation-demo source");
        assert!(cli_source.to_string().contains("valid baseline"));

        let CliError::ValidationDemo { source: demo_error } = &error else {
            panic!("expected typed validation-demo CLI failure, got {error:?}");
        };
        let ValidationDemoError::ExplicitInput { case, source } = demo_error.as_ref() else {
            panic!("expected typed explicit-input failure, got {demo_error:?}");
        };
        assert_eq!(*case, ValidationDemoCaseId::ValidBaseline);
        assert_eq!(source, &ExplicitConstructionError::EmptySimplices);

        let explicit_source = demo_error
            .as_ref()
            .source()
            .expect("validation-demo error should expose the explicit-input source");
        assert_eq!(
            explicit_source.downcast_ref::<ExplicitConstructionError>(),
            Some(&ExplicitConstructionError::EmptySimplices)
        );
    }

    #[test]
    fn unexpected_success_preserves_case_and_expected_boundary() {
        let error = ValidationDemoError::UnexpectedSuccess {
            case: ValidationDemoCaseId::Level1NonFiniteCoordinate,
            expected: ValidationDemoExpectedFailure::ElementValidation,
        };

        assert_eq!(
            error.to_string(),
            "validation demo case Level 1 non-finite coordinate unexpectedly passed Level 1 element validation"
        );
    }

    #[test]
    fn export_covers_each_validation_level() {
        let export = build_export().expect("validation demo should build");

        let export = serde_json::to_value(export).expect("validation demo should serialize");
        assert_eq!(export["schema"], "delaunay.validation_demo");
        assert_eq!(export["schema_version"], 1);
        assert_eq!(export["valid_baseline"]["status"], "passed");
        assert_eq!(
            export["cases"]
                .as_array()
                .expect("validation cases should be an array")
                .iter()
                .map(|case| case["level"].as_u64().expect("level should be numeric"))
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5]
        );
        assert!(
            export["cases"]
                .as_array()
                .expect("validation cases should be an array")
                .iter()
                .all(|case| case["status"] == "failed_as_expected")
        );
        assert_eq!(export["cases"][3]["layer"], "Valid realization");
        assert_eq!(
            export["cases"][4]["public_check"],
            "DelaunayRefinementBuilder::new(...).build()"
        );
    }

    #[test]
    fn export_is_reproducible_across_runs() {
        let first = build_export().expect("first validation demo should build");
        let second = build_export().expect("second validation demo should build");

        let first_json =
            serde_json::to_value(&first).expect("first validation demo should serialize");
        let second_json =
            serde_json::to_value(&second).expect("second validation demo should serialize");

        assert_eq!(first_json, second_json);
    }
}
