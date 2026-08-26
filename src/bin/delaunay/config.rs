#![forbid(unsafe_code)]

//! Parsing, validation, typed errors, and dispatch for the artifact CLI.

use crate::{
    cli_output::ArtifactOutputError,
    generate::{self, GenerateArgs, GenerateCommand},
    spherical_hero::{self, SphericalHeroArgs, SphericalHeroConfig},
    validation_demo::{self, ValidationDemoArgs, ValidationDemoConfig, ValidationDemoError},
};

use clap::{Parser, Subcommand};
use delaunay::{
    VisualizationExportError,
    prelude::{
        construction::{
            DelaunayTriangulationConstructionError, SphericalDelaunayConstructionError,
        },
        generators::RandomPointGenerationError,
        geometry::{CoordinateConversionError, CoordinateRangeError},
        query::ConvexHullConstructionError,
        tds::FacetError,
    },
};
use thiserror::Error;

use std::{
    fmt::{self, Display},
    io::{self, Write},
    process::ExitCode,
};

/// Top-level command-line parser for the opt-in binary.
#[derive(Debug, Parser)]
#[command(
    name = "delaunay",
    version,
    about = "Generate d-dimensional Delaunay artifacts for notebooks and scripts"
)]
pub struct DelaunayCliArgs {
    #[command(subcommand)]
    command: DelaunayCommandArgs,
}

impl DelaunayCliArgs {
    /// Parse raw process arguments with clap.
    ///
    /// Clap prints diagnostics and exits the process for malformed command
    /// lines. Use [`Self::into_validated`] afterward to turn parsed raw values
    /// into a command whose semantic invariants have been checked.
    pub fn from_args() -> Self {
        Self::parse()
    }

    /// Convert raw parsed arguments into a validated command.
    ///
    /// # Errors
    ///
    /// Returns [`CliError`] when parsed arguments are syntactically valid but
    /// violate command semantics, such as unsupported generation dimensions,
    /// too few vertices for the requested dimension, conflicting artifact
    /// destinations.
    pub fn into_validated(self) -> Result<ValidatedDelaunayCommand, CliError> {
        Ok(ValidatedDelaunayCommand(self.command.into_validated()?))
    }
}

/// Print a process-level error and return a failing exit code.
pub fn exit_with_error(error: impl Display) -> ExitCode {
    let stderr = io::stderr();
    let mut handle = stderr.lock();
    let _ = writeln!(handle, "error: {error}");
    ExitCode::FAILURE
}

/// Validated CLI command accepted by the binary runner.
#[derive(Debug)]
pub struct ValidatedDelaunayCommand(DelaunayCommand);

impl ValidatedDelaunayCommand {
    /// Run this validated command.
    ///
    /// # Errors
    ///
    /// Returns [`CliError`] when command execution fails. Failure modes include
    /// artifact I/O or JSON serialization errors, random point generation,
    /// Euclidean or spherical triangulation construction, visualization or
    /// convex-hull export, or validation-demo invariant drift.
    pub fn run(&self) -> Result<(), CliError> {
        match &self.0 {
            DelaunayCommand::Generate(command) => generate::run(command),
            DelaunayCommand::SphericalHero(command) => spherical_hero::run(command),
            DelaunayCommand::ValidationDemo(command) => validation_demo::run(command),
        }
    }
}

/// Validated binary subcommands.
#[derive(Debug)]
enum DelaunayCommand {
    Generate(GenerateCommand),
    SphericalHero(SphericalHeroConfig),
    ValidationDemo(ValidationDemoConfig),
}

/// Raw binary subcommands parsed by clap.
#[derive(Debug, Subcommand)]
enum DelaunayCommandArgs {
    /// Generate a random Delaunay triangulation or visualization export.
    Generate(GenerateArgs),
    /// Emit a deterministic `S^2` triangulation for the notebook-backed README hero.
    SphericalHero(SphericalHeroArgs),
    /// Emit deterministic validation-level failure examples for notebooks.
    ValidationDemo(ValidationDemoArgs),
}

impl DelaunayCommandArgs {
    /// Parse raw subcommand arguments into a semantically validated command.
    fn into_validated(self) -> Result<DelaunayCommand, CliError> {
        match self {
            Self::Generate(args) => Ok(DelaunayCommand::Generate(args.into_validated()?)),
            Self::SphericalHero(args) => Ok(DelaunayCommand::SphericalHero(args.into_validated()?)),
            Self::ValidationDemo(args) => {
                Ok(DelaunayCommand::ValidationDemo(args.into_validated()?))
            }
        }
    }
}

/// CLI workflow whose vertex-count invariant is being checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum VertexCountTarget {
    /// Random Euclidean Delaunay generation in the given dimension.
    EuclideanGeneration {
        /// Runtime Euclidean dimension.
        dimension: usize,
    },
    /// Deterministic spherical-hero generation on `S^2`.
    SphericalHero,
}

impl fmt::Display for VertexCountTarget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EuclideanGeneration { dimension } => {
                write!(formatter, "{dimension}D Euclidean generation")
            }
            Self::SphericalHero => formatter.write_str("S^2 spherical-hero generation"),
        }
    }
}

/// Command-line execution errors.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CliError {
    /// Unsupported dimension slipped past CLI parsing.
    #[error("generate supports dimensions 2 through 5, got {dimension}")]
    UnsupportedGenerateDimension {
        /// Requested dimension.
        dimension: usize,
    },
    /// The requested vertex count is below a workflow's supported minimum.
    #[error("{target} requires at least {minimum} vertices, got {vertices}")]
    TooFewVertices {
        /// Generation workflow whose lower bound was violated.
        target: VertexCountTarget,
        /// Requested vertex count.
        vertices: usize,
        /// Minimum supported vertex count.
        minimum: usize,
    },
    /// The requested vertex count exceeds a workflow's supported maximum.
    #[error("{target} supports at most {maximum} vertices, got {vertices}")]
    TooManyVertices {
        /// Generation workflow whose upper bound was violated.
        target: VertexCountTarget,
        /// Requested vertex count.
        vertices: usize,
        /// Maximum supported vertex count.
        maximum: u32,
    },
    /// Artifact-path validation or output failed.
    #[error(transparent)]
    Artifact {
        /// Typed artifact output error.
        #[from]
        source: ArtifactOutputError,
    },
    /// Coordinate range construction failed.
    #[error(transparent)]
    CoordinateRange {
        /// Typed coordinate-range source error.
        #[from]
        source: CoordinateRangeError<f64>,
    },
    /// Random point generation failed.
    #[error(transparent)]
    PointGeneration {
        /// Typed point-generation source error.
        #[from]
        source: RandomPointGenerationError,
    },
    /// Point-to-vertex conversion failed.
    #[error(transparent)]
    CoordinateConversion {
        /// Typed coordinate-conversion source error.
        #[from]
        source: CoordinateConversionError,
    },
    /// Delaunay construction failed.
    #[error(transparent)]
    Construction {
        /// Typed Delaunay-construction source error.
        #[from]
        source: DelaunayTriangulationConstructionError,
    },
    /// Spherical Delaunay construction failed.
    #[error(transparent)]
    SphericalConstruction {
        /// Typed spherical-construction source error.
        #[from]
        source: SphericalDelaunayConstructionError,
    },
    /// Generic visualization export failed.
    #[error(transparent)]
    Visualization {
        /// Typed visualization-export source error.
        #[from]
        source: VisualizationExportError,
    },
    /// Convex hull extraction failed.
    #[error(transparent)]
    ConvexHull {
        /// Boxed convex-hull source error.
        source: Box<ConvexHullConstructionError>,
    },
    /// Facet-view extraction failed.
    #[error(transparent)]
    Facet {
        /// Typed facet source error.
        #[from]
        source: FacetError,
    },
    /// Validation-demo generation failed or an expected invariant no longer held.
    #[error("{source}")]
    ValidationDemo {
        /// Typed validation-demo failure.
        #[source]
        source: Box<ValidationDemoError>,
    },
}

impl From<ValidationDemoError> for CliError {
    fn from(source: ValidationDemoError) -> Self {
        Self::ValidationDemo {
            source: Box::new(source),
        }
    }
}

impl From<ConvexHullConstructionError> for CliError {
    fn from(source: ConvexHullConstructionError) -> Self {
        Self::ConvexHull {
            source: Box::new(source),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use clap::Parser;

    use super::{CliError, DelaunayCliArgs, DelaunayCommand, VertexCountTarget};
    use crate::{
        cli_output::ArtifactPath,
        generate::{GenerateCommand, GenerateConfig, GenerateDistribution},
    };

    fn assert_empty_path_rejected_by_clap(args: &[&str], argument: &str) {
        let error = DelaunayCliArgs::try_parse_from(args.iter().copied())
            .expect_err("empty path should fail during clap parsing");
        let message = error.to_string();
        assert!(message.contains("a value is required"));
        assert!(message.contains(argument));
    }

    fn validated_generate_3d(args: &[&str]) -> GenerateConfig<3> {
        let command = DelaunayCliArgs::try_parse_from(args)
            .expect("CLI arguments should parse")
            .into_validated()
            .expect("CLI arguments should validate");

        match command.0 {
            DelaunayCommand::Generate(GenerateCommand::D3(config)) => config,
            other => panic!("expected 3D generate command, got {other:?}"),
        }
    }

    #[test]
    fn generate_defaults_to_cube_distribution() {
        let config = validated_generate_3d(&[
            "delaunay",
            "generate",
            "triangulation",
            "--dimension",
            "3",
            "--vertices",
            "4",
        ]);

        assert_eq!(config.distribution(), GenerateDistribution::Cube);
    }

    #[test]
    fn generate_accepts_ball_distribution() {
        let config = validated_generate_3d(&[
            "delaunay",
            "generate",
            "triangulation",
            "--dimension",
            "3",
            "--vertices",
            "4",
            "--distribution",
            "ball",
        ]);

        assert_eq!(config.distribution(), GenerateDistribution::Ball);
    }

    #[test]
    fn generate_config_carries_dimension_sufficient_vertex_count() {
        let config = validated_generate_3d(&[
            "delaunay",
            "generate",
            "triangulation",
            "--dimension",
            "3",
            "--vertices",
            "4",
        ]);

        assert_eq!(config.vertex_count().get(), 4);
    }

    #[test]
    fn generate_zero_vertices_preserves_typed_too_few_vertices_error() {
        let error = DelaunayCliArgs::try_parse_from([
            "delaunay",
            "generate",
            "triangulation",
            "--dimension",
            "3",
            "--vertices",
            "0",
        ])
        .expect("CLI arguments should parse")
        .into_validated()
        .expect_err("zero vertices should fail generate validation");

        let CliError::TooFewVertices {
            target,
            vertices,
            minimum,
        } = error
        else {
            panic!("expected TooFewVertices error, got {error:?}");
        };
        assert_eq!(
            target,
            VertexCountTarget::EuclideanGeneration { dimension: 3 }
        );
        assert_eq!(vertices, 0);
        assert_eq!(minimum, 4);
    }

    #[test]
    fn spherical_hero_too_few_vertices_names_the_spherical_target() {
        let error =
            DelaunayCliArgs::try_parse_from(["delaunay", "spherical-hero", "--vertices", "3"])
                .expect("CLI arguments should parse")
                .into_validated()
                .expect_err("three vertices should fail spherical-hero validation");
        let message = error.to_string();

        let CliError::TooFewVertices {
            target,
            vertices,
            minimum,
        } = error
        else {
            panic!("expected TooFewVertices error, got {error:?}");
        };
        assert_eq!(target, VertexCountTarget::SphericalHero);
        assert_eq!(vertices, 3);
        assert_eq!(minimum, 4);
        assert_eq!(
            message,
            "S^2 spherical-hero generation requires at least 4 vertices, got 3"
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn spherical_hero_rejects_counts_that_exceed_exact_generator_range() {
        let requested = usize::try_from(u64::from(u32::MAX) + 1)
            .expect("64-bit usize should represent u32::MAX + 1");
        let error = DelaunayCliArgs::try_parse_from([
            "delaunay",
            "spherical-hero",
            "--vertices",
            &requested.to_string(),
        ])
        .expect("CLI arguments should parse")
        .into_validated()
        .expect_err("oversized spherical count should fail validation");
        let message = error.to_string();

        let CliError::TooManyVertices {
            target,
            vertices,
            maximum,
        } = error
        else {
            panic!("expected TooManyVertices error, got {error:?}");
        };
        assert_eq!(target, VertexCountTarget::SphericalHero);
        assert_eq!(vertices, requested);
        assert_eq!(maximum, u32::MAX);
        assert_eq!(
            message,
            format!(
                "S^2 spherical-hero generation supports at most {} vertices, got {requested}",
                u32::MAX
            )
        );
    }

    #[test]
    fn generate_rejects_unknown_distribution() {
        let error = DelaunayCliArgs::try_parse_from([
            "delaunay",
            "generate",
            "triangulation",
            "--dimension",
            "3",
            "--vertices",
            "4",
            "--distribution",
            "sphere",
        ])
        .expect_err("unknown distribution should fail during parsing");

        assert!(error.to_string().contains("invalid value"));
    }

    #[test]
    fn generate_rejects_empty_output_path_during_parsing() {
        assert_empty_path_rejected_by_clap(
            &[
                "delaunay",
                "generate",
                "triangulation",
                "--dimension",
                "3",
                "--vertices",
                "4",
                "--output",
                "",
            ],
            "--output",
        );
    }

    #[test]
    fn validation_demo_accepts_output_path() {
        let command = DelaunayCliArgs::try_parse_from([
            "delaunay",
            "validation-demo",
            "--output",
            "target/notebooks/validation/demo.json",
        ])
        .expect("CLI arguments should parse")
        .into_validated()
        .expect("CLI arguments should validate");

        match command.0 {
            DelaunayCommand::ValidationDemo(config) => {
                assert_eq!(
                    config.output.as_ref().map(ArtifactPath::as_path),
                    Some(Path::new("target/notebooks/validation/demo.json"))
                );
            }
            other => panic!("expected validation-demo command, got {other:?}"),
        }
    }

    #[test]
    fn validation_demo_rejects_empty_output_path_during_parsing() {
        assert_empty_path_rejected_by_clap(
            &["delaunay", "validation-demo", "--output", ""],
            "--output",
        );
    }
}
