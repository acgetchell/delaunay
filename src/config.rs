#![forbid(unsafe_code)]

//! Configuration and execution for the opt-in `delaunay` command-line binary.
//!
//! This module belongs to the companion binary enabled by the Cargo `cli`
//! feature. It keeps command-line parsing, raw-argument validation, notebook
//! artifact generation, and diagnostic Pachner-stress execution outside the
//! library API while still using the same typed validation boundaries.

use std::{
    fmt::{self, Display, Write as _},
    fs::{self, File},
    io::{self, BufWriter, Write},
    num::{NonZeroUsize, TryFromIntError},
    path::{Path, PathBuf},
    process::ExitCode,
    time::Instant,
};

use clap::{Args, Parser, Subcommand, ValueEnum};
use delaunay::{
    DelaunayTriangulation, InvariantError, TriangulationEmbeddingValidationError,
    prelude::{
        construction::{
            ConstructionOptions, DelaunayTriangulationBuilder,
            DelaunayTriangulationConstructionError, RetryPolicy, TopologyGuarantee, Vertex, vertex,
        },
        generators::{
            RandomPointGenerationError, generate_random_points_in_ball_seeded,
            generate_random_points_in_range_seeded,
        },
        geometry::{
            CoordinateConversionError, CoordinateRange, CoordinateRangeError, Point, RobustKernel,
        },
        pachner::{
            EdgeKey, FacetHandle, FlipError, PachnerMove, PachnerMoveResult, PachnerMoves,
            PachnerProposal, RidgeHandle, SimplexKey, TriangleHandle, VertexKey,
        },
        query::{ConvexHull, ConvexHullConstructionError, QueryError},
        tds::{FacetError, TdsError},
        triangulation::Triangulation,
    },
    try_vertices_from_points,
};
use markov_chain_monte_carlo::{
    McmcError, TraceError,
    prelude::delayed::{
        Chain, ChainId, DelayedProposal, DelayedStep, DelayedStepError, Target, Trace,
        TraceRecorder, TraceStepOutcome,
    },
};
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
use serde::Serialize;
use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System, get_current_pid};
use uuid::Uuid;

type PachnerStressTriangulation<const D: usize> = Triangulation<RobustKernel<f64>, (), (), D>;

const DEFAULT_3D_VERTICES: usize = 10_000;
const DEFAULT_4D_VERTICES: usize = 1_000;
const DEFAULT_ATTEMPTS: usize = 100_000;
const DEFAULT_KEY_REFRESH_EVERY: usize = 256;
const DEFAULT_RETRY_ATTEMPTS: usize = 24;
const DEFAULT_VALIDATE_EVERY: usize = 1_000;
const DEFAULT_VERTEX_GROWTH_DIVISOR: usize = 10;
const DEFAULT_VERTEX_SHRINK_DIVISOR: usize = 20;
const DEFAULT_GENERATE_DIMENSION: usize = 3;
const DEFAULT_GENERATE_VERTICES: usize = 100;
const DEFAULT_GENERATE_SEED: u64 = 0xD3_1A_05_25_03;
const CONVEX_HULL_EXPORT_SCHEMA: &str = "delaunay.convex_hull";
const CONVEX_HULL_EXPORT_SCHEMA_VERSION: u32 = 1;
const VALIDATION_DEMO_EXPORT_SCHEMA: &str = "delaunay.validation_demo";
const VALIDATION_DEMO_EXPORT_SCHEMA_VERSION: u32 = 1;
const TRACE_TAIL: usize = 32;

/// Top-level command-line parser for the opt-in binary.
#[derive(Debug, Parser)]
#[command(
    name = "delaunay",
    version,
    about = "Generate and diagnose d-dimensional Delaunay triangulations"
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
    /// too few vertices for the requested dimension, or invalid Pachner stress
    /// counts.
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
    /// artifact I/O or JSON serialization errors, random point generation or
    /// triangulation construction errors, convex-hull extraction errors,
    /// validation-demo invariant drift, and Pachner stress diagnostic failures.
    pub fn run(self) -> Result<(), CliError> {
        match self.0 {
            DelaunayCommand::Generate(command) => run_generate(&command),
            DelaunayCommand::ValidationDemo(command) => run_validation_demo(&command),
            DelaunayCommand::PachnerStress { config, artifacts } => {
                run_pachner_stress(config, &artifacts)?;
                Ok(())
            }
        }
    }
}

/// Validated binary subcommands.
#[derive(Debug)]
enum DelaunayCommand {
    Generate(GenerateCommand),
    ValidationDemo(ValidationDemoConfig),
    PachnerStress {
        config: PachnerStressConfig,
        artifacts: PachnerStressArtifacts,
    },
}

/// Raw binary subcommands parsed by clap.
#[derive(Debug, Subcommand)]
enum DelaunayCommandArgs {
    /// Generate a random Delaunay triangulation or its convex hull.
    Generate(GenerateArgs),
    /// Emit deterministic validation-level failure examples for notebooks.
    ValidationDemo(ValidationDemoArgs),
    /// Run one MCMC-backed Pachner move stress chain.
    PachnerStress(PachnerStressArgs),
}

impl DelaunayCommandArgs {
    /// Parse raw subcommand arguments into a semantically validated command.
    fn into_validated(self) -> Result<DelaunayCommand, CliError> {
        match self {
            Self::Generate(args) => args.into_validated(),
            Self::ValidationDemo(args) => {
                Ok(DelaunayCommand::ValidationDemo(args.into_validated()))
            }
            Self::PachnerStress(args) => args.into_validated(),
        }
    }
}

/// Generated object requested by the companion binary.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum GenerateKind {
    /// Emit the generated Delaunay triangulation as the crate's serde JSON.
    Triangulation,
    /// Emit the generated triangulation's convex-hull facets as JSON.
    ConvexHull,
}

/// Random point distribution requested by the companion binary.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum GenerateDistribution {
    /// Uniform points in the unit cube `[0, 1]^D`.
    Cube,
    /// Uniform points in the radius-1 ball centered at the origin.
    Ball,
}

/// Raw command-line arguments for `delaunay generate`.
#[derive(Debug, Args)]
struct GenerateArgs {
    /// Output object to generate.
    #[arg(value_enum, default_value = "triangulation")]
    kind: GenerateKind,
    /// Dimension to generate.
    #[arg(short = 'd', long, default_value_t = DEFAULT_GENERATE_DIMENSION)]
    dimension: usize,
    /// Number of random input vertices.
    #[arg(short = 'n', long, default_value_t = DEFAULT_GENERATE_VERTICES)]
    vertices: usize,
    /// Random point distribution.
    #[arg(long, value_enum, default_value = "cube")]
    distribution: GenerateDistribution,
    /// Random seed.
    #[arg(long, default_value_t = DEFAULT_GENERATE_SEED)]
    seed: u64,
    /// Write JSON to a file instead of stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,
}

impl GenerateArgs {
    /// Validate generation arguments and choose the const-generic runner.
    fn into_validated(self) -> Result<DelaunayCommand, CliError> {
        match self.dimension {
            2 => Ok(DelaunayCommand::Generate(GenerateCommand::D2(
                GenerateConfig::try_new(self)?,
            ))),
            3 => Ok(DelaunayCommand::Generate(GenerateCommand::D3(
                GenerateConfig::try_new(self)?,
            ))),
            4 => Ok(DelaunayCommand::Generate(GenerateCommand::D4(
                GenerateConfig::try_new(self)?,
            ))),
            5 => Ok(DelaunayCommand::Generate(GenerateCommand::D5(
                GenerateConfig::try_new(self)?,
            ))),
            dimension => Err(CliError::UnsupportedGenerateDimension { dimension }),
        }
    }
}

/// Validated generation command by dimension.
#[derive(Debug)]
enum GenerateCommand {
    D2(GenerateConfig<2>),
    D3(GenerateConfig<3>),
    D4(GenerateConfig<4>),
    D5(GenerateConfig<5>),
}

/// Validated generation configuration for one const-generic dimension.
#[derive(Debug)]
struct GenerateConfig<const D: usize> {
    kind: GenerateKind,
    vertices: usize,
    distribution: GenerateDistribution,
    seed: u64,
    output: Option<PathBuf>,
}

impl<const D: usize> GenerateConfig<D> {
    /// Validate dimension-dependent generation limits.
    fn try_new(args: GenerateArgs) -> Result<Self, CliError> {
        if args.vertices < D + 1 {
            return Err(CliError::TooFewVertices {
                dimension: D,
                vertices: args.vertices,
                minimum: D + 1,
            });
        }

        Ok(Self {
            kind: args.kind,
            vertices: args.vertices,
            distribution: args.distribution,
            seed: args.seed,
            output: args.output,
        })
    }
}

/// Raw command-line arguments for `delaunay validation-demo`.
#[derive(Debug, Args)]
struct ValidationDemoArgs {
    /// Write JSON to a file instead of stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,
}

impl ValidationDemoArgs {
    /// Convert raw validation-demo arguments into a passive artifact config.
    fn into_validated(self) -> ValidationDemoConfig {
        ValidationDemoConfig {
            output: self.output,
        }
    }
}

/// Validated configuration for the generated validation-model demo artifact.
#[derive(Debug)]
struct ValidationDemoConfig {
    output: Option<PathBuf>,
}

/// Raw command-line arguments for `delaunay pachner-stress`.
#[derive(Debug, Args)]
struct PachnerStressArgs {
    /// Stress dimension.
    #[arg(long, value_enum, default_value = "3d")]
    dimension: PachnerStressDimension,
    /// Initial vertex count. Defaults to 10000 in 3D and 1000 in 4D.
    #[arg(long)]
    vertices: Option<usize>,
    /// Attempted Pachner moves.
    #[arg(long, default_value_t = DEFAULT_ATTEMPTS)]
    attempts: usize,
    /// Validation and progress-reporting cadence.
    #[arg(long, default_value_t = DEFAULT_VALIDATE_EVERY)]
    validate_every: usize,
    /// Cached-key refresh cadence.
    #[arg(long, default_value_t = DEFAULT_KEY_REFRESH_EVERY)]
    key_refresh_every: usize,
    /// Randomized construction retry attempts.
    #[arg(long, default_value_t = DEFAULT_RETRY_ATTEMPTS)]
    retry_attempts: usize,
    /// Random seed. Defaults to a dimension-specific seed.
    #[arg(long)]
    seed: Option<u64>,
    /// Write periodic progress rows to CSV.
    #[arg(long)]
    progress_csv: Option<PathBuf>,
    /// Write final run summary JSON.
    #[arg(long)]
    summary_json: Option<PathBuf>,
    /// Suppress stdout telemetry.
    #[arg(long)]
    quiet: bool,
}

impl PachnerStressArgs {
    /// Convert raw stress-test options into invariant-bearing run settings.
    fn into_validated(self) -> Result<DelaunayCommand, CliError> {
        let config = PachnerStressConfig::try_new(
            self.dimension,
            self.vertices
                .unwrap_or_else(|| self.dimension.default_vertices()),
            positive_nonzero(PachnerStressCountArgument::Attempts, self.attempts)?,
            positive_nonzero(
                PachnerStressCountArgument::ValidateEvery,
                self.validate_every,
            )?,
            positive_nonzero(
                PachnerStressCountArgument::KeyRefreshEvery,
                self.key_refresh_every,
            )?,
            positive_nonzero(
                PachnerStressCountArgument::RetryAttempts,
                self.retry_attempts,
            )?,
            self.seed.unwrap_or_else(|| self.dimension.default_seed()),
        )?;
        let artifacts =
            PachnerStressArtifacts::try_new(self.progress_csv, self.summary_json, !self.quiet)?;
        Ok(DelaunayCommand::PachnerStress { config, artifacts })
    }
}

/// Detached convex-hull export used by notebooks and support scripts.
#[derive(Debug, Serialize)]
struct ConvexHullExport<const D: usize> {
    schema: &'static str,
    schema_version: u32,
    dimension: usize,
    vertex_count: usize,
    simplex_count: usize,
    facet_count: usize,
    facets: Vec<ConvexHullFacetRecord<D>>,
}

/// One convex-hull facet in deterministic iterator order.
#[derive(Debug, Serialize)]
struct ConvexHullFacetRecord<const D: usize> {
    index: usize,
    vertex_ids: Vec<Uuid>,
    coordinates: Vec<Vec<f64>>,
}

/// Notebook-facing validation-model artifact generated by public failure paths.
#[derive(Debug, Serialize)]
struct ValidationDemoExport {
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

/// Dispatch a validated generation command to its const-generic implementation.
fn run_generate(command: &GenerateCommand) -> Result<(), CliError> {
    match command {
        GenerateCommand::D2(config) => run_generate_dimension(config),
        GenerateCommand::D3(config) => run_generate_dimension(config),
        GenerateCommand::D4(config) => run_generate_dimension(config),
        GenerateCommand::D5(config) => run_generate_dimension(config),
    }
}

/// Generate and emit one artifact for a concrete dimension.
fn run_generate_dimension<const D: usize>(config: &GenerateConfig<D>) -> Result<(), CliError> {
    let triangulation =
        build_generated_delaunay::<D>(config.vertices, config.seed, config.distribution)?;
    match config.kind {
        GenerateKind::Triangulation => {
            write_json_output(&triangulation, config.output.as_deref())?;
        }
        GenerateKind::ConvexHull => {
            let hull = build_convex_hull_export(&triangulation)?;
            write_json_output(&hull, config.output.as_deref())?;
        }
    }
    Ok(())
}

/// Generate the validation-model artifact used by the notebook quickstart.
fn run_validation_demo(config: &ValidationDemoConfig) -> Result<(), CliError> {
    let export = build_validation_demo_export()?;
    write_json_output(&export, config.output.as_deref())?;
    Ok(())
}

/// Build a random PL-manifold Delaunay triangulation for CLI export.
fn build_generated_delaunay<const D: usize>(
    vertex_count: usize,
    seed: u64,
    distribution: GenerateDistribution,
) -> Result<DelaunayTriangulation<RobustKernel<f64>, (), (), D>, CliError> {
    let points = match distribution {
        GenerateDistribution::Cube => generate_random_points_in_range_seeded::<D>(
            vertex_count,
            CoordinateRange::try_new(0.0_f64, 1.0)?,
            seed,
        )?,
        GenerateDistribution::Ball => {
            generate_random_points_in_ball_seeded::<D>(vertex_count, 1.0, seed)?
        }
    };
    let vertices = try_vertices_from_points(&points)?;
    Ok(DelaunayTriangulationBuilder::new(&vertices)
        .topology_guarantee(TopologyGuarantee::PLManifold)
        .build_with_kernel(&RobustKernel::new())?)
}

/// Convert a triangulation into the stable convex-hull JSON schema.
fn build_convex_hull_export<const D: usize>(
    triangulation: &DelaunayTriangulation<RobustKernel<f64>, (), (), D>,
) -> Result<ConvexHullExport<D>, CliError> {
    let hull = ConvexHull::try_from_triangulation(triangulation.as_triangulation())?;
    let facets = hull
        .try_facets(triangulation.as_triangulation())?
        .enumerate()
        .map(|(index, facet)| {
            let facet = facet?;
            let (vertex_ids, coordinates) = facet
                .vertices()
                .map(|vertex| (vertex.uuid(), vertex.point().coords().to_vec()))
                .unzip();
            Ok::<_, CliError>(ConvexHullFacetRecord {
                index,
                vertex_ids,
                coordinates,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(ConvexHullExport {
        schema: CONVEX_HULL_EXPORT_SCHEMA,
        schema_version: CONVEX_HULL_EXPORT_SCHEMA_VERSION,
        dimension: D,
        vertex_count: triangulation.number_of_vertices(),
        simplex_count: triangulation.number_of_simplices(),
        facet_count: facets.len(),
        facets,
    })
}

/// Build the deterministic validation examples rendered by the notebook.
fn build_validation_demo_export() -> Result<ValidationDemoExport, CliError> {
    Ok(ValidationDemoExport {
        schema: VALIDATION_DEMO_EXPORT_SCHEMA,
        schema_version: VALIDATION_DEMO_EXPORT_SCHEMA_VERSION,
        dimension: 2,
        valid_baseline: validation_demo_valid_baseline()?,
        cases: vec![
            validation_demo_level_1(),
            validation_demo_level_2()?,
            validation_demo_level_3()?,
            validation_demo_level_4()?,
            validation_demo_level_5()?,
        ],
    })
}

/// Generate a passing explicit triangle used as the visual baseline.
fn validation_demo_valid_baseline() -> Result<ValidationDemoCase, CliError> {
    let coordinates = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.866_025_403_784_438_6]];
    let simplices = vec![vec![0, 1, 2]];
    let vertices = demo_vertices(&coordinates)?;
    let dt = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .map_err(|source| CliError::ValidationDemoInvariant {
            case: "valid baseline",
            message: format!("explicit builder parse failed unexpectedly: {source}"),
        })?
        .build()?;
    dt.validate()
        .map_err(|source| CliError::ValidationDemoInvariant {
            case: "valid baseline",
            message: format!("explicit baseline failed validation unexpectedly: {source}"),
        })?;

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
fn validation_demo_level_1() -> ValidationDemoCase {
    let diagnostic = Point::<2>::try_new([f64::NAN, 0.0])
        .expect_err("non-finite point must fail Level 1 coordinate validation")
        .to_string();
    let mut visual = demo_visual([[0.0, 0.0]], Vec::new());
    visual.invalid_points.push(0);

    ValidationDemoCase {
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
    }
}

/// Generate the Level 2 duplicate-simplex structural failure example.
fn validation_demo_level_2() -> Result<ValidationDemoCase, CliError> {
    let coordinates = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let simplices = vec![vec![0, 1, 2], vec![0, 1, 2]];
    let diagnostic =
        explicit_builder_failure("Level 2 duplicate simplex", coordinates, &simplices)?;
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
fn validation_demo_level_3() -> Result<ValidationDemoCase, CliError> {
    let coordinates = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.45, 0.85]];
    let simplices = vec![vec![0, 1, 2]];
    let diagnostic = explicit_builder_failure("Level 3 isolated vertex", coordinates, &simplices)?;
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

/// Generate the Level 4 invalid affine realization failure example.
fn validation_demo_level_4() -> Result<ValidationDemoCase, CliError> {
    let coordinates = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
    let simplices = vec![vec![0, 1, 2]];
    let diagnostic = explicit_builder_failure(
        "Level 4 invalid affine realization",
        coordinates,
        &simplices,
    )?;
    let mut visual = demo_visual(coordinates, simplices);
    visual.highlighted_simplices = vec![0];
    visual.highlighted_edges.push([0, 2]);

    Ok(ValidationDemoCase {
        level: 4,
        layer: "Valid affine realization",
        title: "Degenerate realized simplex",
        status: "failed_as_expected",
        public_check: "Triangulation::validate_embedding",
        public_reference: "tests/triangulation_builder.rs::test_explicit_error_variant_geometric_nondegeneracy",
        input_summary: "One triangle whose three vertices are collinear",
        explanation: "The coordinate realization is not a valid affine realization because the abstract 2-simplex collapses to zero area.",
        diagnostic,
        visual,
    })
}

/// Generate the Level 5 non-Delaunay diagonal failure example.
fn validation_demo_level_5() -> Result<ValidationDemoCase, CliError> {
    let coordinates = [[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [1.0, 2.0]];
    let simplices = vec![vec![0, 1, 2], vec![0, 2, 3]];
    let diagnostic =
        explicit_builder_failure("Level 5 non-Delaunay diagonal", coordinates, &simplices)?;
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
        public_check: "DelaunayTriangulation::is_valid_delaunay",
        public_reference: "tests/triangulation_builder.rs::test_explicit_non_delaunay_mesh",
        input_summary: "Quadrilateral triangulated with diagonal AC instead of BD",
        explanation: "Point D lies inside the circumcircle of triangle ABC, so the chosen diagonal violates the local Delaunay property.",
        diagnostic,
        visual,
    })
}

/// Return the diagnostic from a public explicit-builder case that must fail.
fn explicit_builder_failure<const N: usize>(
    case: &'static str,
    coordinates: [[f64; 2]; N],
    simplices: &[Vec<usize>],
) -> Result<String, CliError> {
    let vertices = demo_vertices(&coordinates)?;
    let builder =
        DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, simplices)
            .map_err(|source| CliError::ValidationDemoInvariant {
                case,
                message: format!(
                    "explicit builder parse failed before the intended layer: {source}"
                ),
            })?;

    match builder.build() {
        Ok(_) => Err(CliError::ValidationDemoInvariant {
            case,
            message: "expected the explicit validation case to fail, but it passed".to_owned(),
        }),
        Err(error) => Ok(stable_validation_demo_diagnostic(&error.to_string())),
    }
}

/// Remove run-specific identifiers from validation-demo diagnostics so paper
/// artifacts are reproducible while preserving the diagnostic shape.
fn stable_validation_demo_diagnostic(diagnostic: &str) -> String {
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
        .map(|coords| vertex!(*coords).map_err(CliError::CoordinateConversion))
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

/// Write pretty JSON either to a requested path or stdout.
fn write_json_output(value: &impl Serialize, path: Option<&Path>) -> Result<(), CliError> {
    if let Some(path) = path {
        ensure_parent_dir(path)?;
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, value)?;
        writer.flush()?;
    } else {
        let stdout = io::stdout();
        let mut handle = stdout.lock();
        serde_json::to_writer_pretty(&mut handle, value)?;
        writeln!(handle)?;
    }
    Ok(())
}

/// Command-line execution errors.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CliError {
    /// Unsupported dimension slipped past CLI parsing.
    #[error("generate supports dimensions 2 through 5, got {dimension}")]
    UnsupportedGenerateDimension {
        /// Requested dimension.
        dimension: usize,
    },
    /// The requested vertex count cannot support the dimension.
    #[error("{dimension}D generation requires at least {minimum} vertices, got {vertices}")]
    TooFewVertices {
        /// Runtime dimension.
        dimension: usize,
        /// Requested vertex count.
        vertices: usize,
        /// Minimum supported vertex count.
        minimum: usize,
    },
    /// I/O failed while writing an artifact.
    #[error(transparent)]
    Io(#[from] io::Error),
    /// JSON serialization failed while writing an artifact.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Coordinate range construction failed.
    #[error(transparent)]
    CoordinateRange(#[from] CoordinateRangeError<f64>),
    /// Random point generation failed.
    #[error(transparent)]
    PointGeneration(#[from] RandomPointGenerationError),
    /// Point-to-vertex conversion failed.
    #[error(transparent)]
    CoordinateConversion(#[from] CoordinateConversionError),
    /// Delaunay construction failed.
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    /// Convex hull extraction failed.
    #[error(transparent)]
    ConvexHull(Box<ConvexHullConstructionError>),
    /// Facet-view extraction failed.
    #[error(transparent)]
    Facet(#[from] FacetError),
    /// A generated validation-demo case no longer fails at the intended boundary.
    #[error("validation demo case {case} is inconsistent: {message}")]
    ValidationDemoInvariant {
        /// Validation-demo case label.
        case: &'static str,
        /// Explanation of the unexpected result.
        message: String,
    },
    /// Pachner stress failed.
    #[error(transparent)]
    PachnerStress(Box<PachnerStressError>),
}

impl From<ConvexHullConstructionError> for CliError {
    fn from(source: ConvexHullConstructionError) -> Self {
        Self::ConvexHull(Box::new(source))
    }
}

impl From<PachnerStressError> for CliError {
    fn from(source: PachnerStressError) -> Self {
        Self::PachnerStress(Box::new(source))
    }
}

/// Supported dimensions for the manual Pachner stress diagnostic.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum PachnerStressDimension {
    /// Three-dimensional Pachner stress case.
    #[value(name = "3d", alias = "3")]
    Three,
    /// Four-dimensional Pachner stress case.
    #[value(name = "4d", alias = "4")]
    Four,
}

impl PachnerStressDimension {
    /// Return the dimension as a const-generic runtime value for diagnostics.
    const fn value(self) -> usize {
        match self {
            Self::Three => 3,
            Self::Four => 4,
        }
    }

    /// Return the label used in telemetry and artifact names.
    const fn label(self) -> &'static str {
        match self {
            Self::Three => "3d",
            Self::Four => "4d",
        }
    }

    /// Return the default vertex count for this dimension.
    const fn default_vertices(self) -> usize {
        match self {
            Self::Three => DEFAULT_3D_VERTICES,
            Self::Four => DEFAULT_4D_VERTICES,
        }
    }

    /// Return the default RNG seed for this dimension.
    const fn default_seed(self) -> u64 {
        match self {
            Self::Three => 0x0253_0000_0000_0003,
            Self::Four => 0x0253_0000_0000_0004,
        }
    }
}

/// Positive count arguments validated by the Pachner stress diagnostic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PachnerStressCountArgument {
    /// Attempted Pachner moves.
    Attempts,
    /// Validation and progress-reporting cadence.
    ValidateEvery,
    /// Cached-key refresh cadence.
    KeyRefreshEvery,
    /// Randomized construction retry attempts.
    RetryAttempts,
}

impl PachnerStressCountArgument {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Attempts => "attempts",
            Self::ValidateEvery => "validate_every",
            Self::KeyRefreshEvery => "key_refresh_every",
            Self::RetryAttempts => "retry_attempts",
        }
    }
}

impl Display for PachnerStressCountArgument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Configuration for one exact Pachner stress chain.
#[derive(Clone, Copy, Debug)]
struct PachnerStressConfig {
    dimension: PachnerStressDimension,
    label: &'static str,
    vertex_count: usize,
    move_attempts: NonZeroUsize,
    validate_every: NonZeroUsize,
    key_refresh_every: NonZeroUsize,
    retry_attempts: NonZeroUsize,
    min_vertex_count: usize,
    max_vertex_count: usize,
    seed: u64,
}

impl PachnerStressConfig {
    /// Build a validated stress configuration from command-line values.
    fn try_new(
        dimension: PachnerStressDimension,
        vertex_count: usize,
        move_attempts: NonZeroUsize,
        validate_every: NonZeroUsize,
        key_refresh_every: NonZeroUsize,
        retry_attempts: NonZeroUsize,
        seed: u64,
    ) -> Result<Self, PachnerStressError> {
        let minimum_vertices = dimension.value() + 1;
        if vertex_count < minimum_vertices {
            return Err(PachnerStressError::TooFewVertices {
                dimension: dimension.value(),
                vertices: vertex_count,
                minimum: minimum_vertices,
            });
        }
        let validate_every = validate_every.min(move_attempts);
        let growth_slack =
            (vertex_count / DEFAULT_VERTEX_GROWTH_DIVISOR).max(dimension.value() + 1);
        let shrink_slack = vertex_count / DEFAULT_VERTEX_SHRINK_DIVISOR;

        Ok(Self {
            dimension,
            label: dimension.label(),
            vertex_count,
            move_attempts,
            validate_every,
            key_refresh_every,
            retry_attempts,
            min_vertex_count: vertex_count
                .saturating_sub(shrink_slack)
                .max(dimension.value() + 1),
            max_vertex_count: vertex_count.saturating_add(growth_slack),
            seed,
        })
    }

    /// Positive number of attempted moves in this exact chain.
    const fn move_attempts(self) -> NonZeroUsize {
        self.move_attempts
    }

    /// Positive periodic validation cadence.
    const fn validate_every(self) -> NonZeroUsize {
        self.validate_every
    }

    /// Positive cached-key refresh cadence.
    const fn key_refresh_every(self) -> NonZeroUsize {
        self.key_refresh_every
    }

    /// Retry attempts for randomized Delaunay construction.
    const fn retry_attempts(self) -> NonZeroUsize {
        self.retry_attempts
    }
}

/// Artifact paths and stdout behavior for one diagnostic run.
#[derive(Debug)]
struct PachnerStressArtifacts {
    progress_csv: Option<PathBuf>,
    summary_json: Option<PathBuf>,
    stdout: bool,
}

impl PachnerStressArtifacts {
    /// Build a validated artifact configuration for one diagnostic run.
    fn try_new(
        progress_csv: Option<PathBuf>,
        summary_json: Option<PathBuf>,
        stdout: bool,
    ) -> Result<Self, PachnerStressError> {
        if let (Some(progress_csv), Some(summary_json)) = (&progress_csv, &summary_json)
            && progress_csv == summary_json
        {
            return Err(PachnerStressError::DuplicateArtifactPath {
                path: progress_csv.clone(),
            });
        }

        Ok(Self {
            progress_csv,
            summary_json,
            stdout,
        })
    }
}

/// Initial triangulation metadata emitted before the chain starts.
#[derive(Clone, Copy, Debug, Serialize)]
struct PachnerStressSource {
    dimension: usize,
    label: &'static str,
    vertices: usize,
    simplices: usize,
    seed: u64,
}

/// Final aggregate metrics for one exact Pachner stress chain.
#[derive(Clone, Copy, Debug, Serialize)]
struct PachnerStressReport {
    sequence: usize,
    attempts: usize,
    accepted: usize,
    rejected: usize,
    candidate_misses: usize,
    proposal_rejections: usize,
    validations: usize,
    validation_nanos: u128,
    elapsed_nanos: u128,
    attempts_per_second: u128,
    final_vertices: usize,
    final_simplices: usize,
    start_rss_kib: u64,
    max_rss_kib: u64,
    final_rss_kib: u64,
}

/// JSON summary written by the diagnostic CLI.
#[derive(Clone, Debug, Serialize)]
struct PachnerStressSummary {
    dimension: usize,
    label: &'static str,
    configured_vertices: usize,
    attempts: usize,
    validate_every: usize,
    key_refresh_every: usize,
    retry_attempts: usize,
    min_vertex_count: usize,
    max_vertex_count: usize,
    seed: u64,
    source: PachnerStressSource,
    report: PachnerStressReport,
}

/// Per-validation progress row written to stdout and CSV.
#[derive(Clone, Copy)]
struct PachnerStressProgress {
    sequence: usize,
    step: usize,
    attempts: usize,
    accepted: usize,
    rejected: usize,
    candidate_misses: usize,
    proposal_rejections: usize,
    validations: usize,
    validation_nanos: u128,
    acceptance_rate: f64,
    vertices: usize,
    simplices: usize,
    rss_kib: u64,
}

/// Report sink that mirrors causal-triangulations' binary-to-notebook boundary.
struct PachnerStressReporter {
    stdout: bool,
    progress_writer: Option<BufWriter<File>>,
}

impl PachnerStressReporter {
    /// Create the requested file sinks and emit a CSV header when needed.
    fn try_new(artifacts: &PachnerStressArtifacts) -> Result<Self, PachnerStressError> {
        let progress_writer = artifacts
            .progress_csv
            .as_deref()
            .map(create_progress_writer)
            .transpose()?;
        Ok(Self {
            stdout: artifacts.stdout,
            progress_writer,
        })
    }

    /// Emit initial triangulation metadata.
    fn emit_source(&self, source: PachnerStressSource) -> Result<(), PachnerStressError> {
        if self.stdout {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            writeln!(
                handle,
                "pachner_stress_source dimension={} label={} vertices={} simplices={} seed={}",
                source.dimension, source.label, source.vertices, source.simplices, source.seed
            )?;
        }
        Ok(())
    }

    /// Emit a periodic progress record.
    fn emit_progress(
        &mut self,
        config: PachnerStressConfig,
        progress: PachnerStressProgress,
    ) -> Result<(), PachnerStressError> {
        if self.stdout {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            writeln!(
                handle,
                "pachner_stress_progress dimension={} label={} sequence={} step={} attempts={} accepted={} \
                 rejected={} candidate_misses={} proposal_rejections={} validations={} \
                 validation_nanos={} acceptance_rate={:.6} vertices={} simplices={} rss_kib={}",
                config.dimension.value(),
                config.label,
                progress.sequence,
                progress.step,
                progress.attempts,
                progress.accepted,
                progress.rejected,
                progress.candidate_misses,
                progress.proposal_rejections,
                progress.validations,
                progress.validation_nanos,
                progress.acceptance_rate,
                progress.vertices,
                progress.simplices,
                progress.rss_kib
            )?;
        }
        if let Some(writer) = &mut self.progress_writer {
            writeln!(
                writer,
                "{},{},{},{},{},{},{},{},{},{},{},{:.6},{},{},{}",
                config.dimension.value(),
                config.label,
                progress.sequence,
                progress.step,
                progress.attempts,
                progress.accepted,
                progress.rejected,
                progress.candidate_misses,
                progress.proposal_rejections,
                progress.validations,
                progress.validation_nanos,
                progress.acceptance_rate,
                progress.vertices,
                progress.simplices,
                progress.rss_kib
            )?;
        }
        Ok(())
    }

    /// Emit final aggregate metrics.
    fn emit_report(
        &self,
        config: PachnerStressConfig,
        report: PachnerStressReport,
    ) -> Result<(), PachnerStressError> {
        if self.stdout {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            writeln!(
                handle,
                "pachner_stress_metric dimension={} label={} sequence={} attempts={} accepted={} rejected={} \
                 candidate_misses={} proposal_rejections={} validations={} validation_nanos={} \
                 elapsed_nanos={} attempts_per_second={} final_vertices={} final_simplices={} \
                 start_rss_kib={} max_rss_kib={} final_rss_kib={}",
                config.dimension.value(),
                config.label,
                report.sequence,
                report.attempts,
                report.accepted,
                report.rejected,
                report.candidate_misses,
                report.proposal_rejections,
                report.validations,
                report.validation_nanos,
                report.elapsed_nanos,
                report.attempts_per_second,
                report.final_vertices,
                report.final_simplices,
                report.start_rss_kib,
                report.max_rss_kib,
                report.final_rss_kib
            )?;
        }
        Ok(())
    }

    /// Flush any open file sinks.
    fn finish(&mut self) -> Result<(), PachnerStressError> {
        if let Some(writer) = &mut self.progress_writer {
            writer.flush()?;
        }
        Ok(())
    }
}

/// Randomized live topology frontier used for move proposals.
#[derive(Default)]
struct MoveSampler {
    simplex_keys: Vec<SimplexKey>,
    vertex_keys: Vec<VertexKey>,
    facet_handles: Vec<FacetHandle>,
    edge_keys: Vec<EdgeKey>,
    ridge_handles: Vec<RidgeHandle>,
}

impl MoveSampler {
    /// Capture the current live key frontier used for randomized move proposals.
    fn try_from_triangulation<const D: usize>(
        dt: &PachnerStressTriangulation<D>,
    ) -> Result<Self, PachnerStressError> {
        let mut sampler = Self::default();
        sampler.refresh(dt)?;
        Ok(sampler)
    }

    /// Refresh cached keys after enough accepted moves may have stale candidates.
    fn refresh<const D: usize>(
        &mut self,
        dt: &PachnerStressTriangulation<D>,
    ) -> Result<(), PachnerStressError> {
        self.simplex_keys.clear();
        self.simplex_keys
            .extend(dt.simplices().map(|(simplex_key, _)| simplex_key));

        self.vertex_keys.clear();
        self.vertex_keys
            .extend(dt.vertices().map(|(vertex_key, _)| vertex_key));

        self.facet_handles.clear();
        for facet in dt.facets() {
            self.facet_handles.push(facet?.handle());
        }

        self.edge_keys.clear();
        self.edge_keys.extend(dt.edges());

        self.ridge_handles.clear();
        for ridge in dt.ridge_handles() {
            self.ridge_handles.push(ridge?);
        }
        Ok(())
    }

    /// Select a cached simplex key uniformly from the last refresh.
    fn random_simplex_key(&self, rng: &mut (impl Rng + ?Sized)) -> Option<SimplexKey> {
        random_cached(&self.simplex_keys, rng)
    }

    /// Select a cached vertex key uniformly from the last refresh.
    fn random_vertex_key(&self, rng: &mut (impl Rng + ?Sized)) -> Option<VertexKey> {
        random_cached(&self.vertex_keys, rng)
    }

    /// Select a cached facet handle uniformly from the last refresh.
    fn random_facet(&self, rng: &mut (impl Rng + ?Sized)) -> Option<FacetHandle> {
        random_cached(&self.facet_handles, rng)
    }

    /// Select a cached edge key uniformly from the last refresh.
    fn random_edge(&self, rng: &mut (impl Rng + ?Sized)) -> Option<EdgeKey> {
        random_cached(&self.edge_keys, rng)
    }

    /// Select a cached ridge handle uniformly from the last refresh.
    fn random_ridge(&self, rng: &mut (impl Rng + ?Sized)) -> Option<RidgeHandle> {
        random_cached(&self.ridge_handles, rng)
    }
}

/// Select a cached proposal item uniformly while preserving empty-cache misses.
fn random_cached<T: Copy>(values: &[T], rng: &mut (impl Rng + ?Sized)) -> Option<T> {
    if values.is_empty() {
        return None;
    }
    let index = rng.random_range(0..values.len());
    Some(values[index])
}

/// Flat diagnostic target: successful planned Pachner moves accept with probability one.
struct FlatPachnerTarget;

impl<const D: usize> Target<PachnerStressTriangulation<D>> for FlatPachnerTarget {
    fn log_prob(&self, _state: &PachnerStressTriangulation<D>) -> f64 {
        0.0
    }
}

/// Delayed-proposal plan carrying the parsed Pachner proposal.
#[derive(Clone, Debug)]
struct PachnerChainPlan<const D: usize> {
    proposal: PachnerProposal<(), D>,
}

/// Step metadata retained for invariant-failure diagnostics.
#[derive(Clone, Debug)]
enum PachnerStepInfo<const D: usize> {
    Proposed {
        request: PachnerMove<(), D>,
    },
    CandidateMiss,
    ProposalRejected {
        request: PachnerMove<(), D>,
        rejection: FlipError,
    },
}

/// Delayed-proposal kernel that plans local Pachner edits against the live state.
struct PachnerProposalKernel<const D: usize> {
    config: PachnerStressConfig,
    sampler: MoveSampler,
    proposed_steps: usize,
    candidate_misses: usize,
    proposal_rejections: usize,
    last_request: Option<PachnerMove<(), D>>,
    last_result: Option<PachnerMoveResult<D>>,
    last_no_plan_info: Option<PachnerStepInfo<D>>,
}

impl<const D: usize> PachnerProposalKernel<D> {
    /// Create a proposal kernel with an initial live-key cache.
    fn try_new(
        dt: &PachnerStressTriangulation<D>,
        config: PachnerStressConfig,
    ) -> Result<Self, PachnerStressError> {
        Ok(Self {
            config,
            sampler: MoveSampler::try_from_triangulation(dt)?,
            proposed_steps: 0,
            candidate_misses: 0,
            proposal_rejections: 0,
            last_request: None,
            last_result: None,
            last_no_plan_info: None,
        })
    }

    /// Refresh cached keys on the configured cadence.
    fn maybe_refresh(
        &mut self,
        dt: &PachnerStressTriangulation<D>,
    ) -> Result<(), PachnerStressError> {
        if self
            .proposed_steps
            .is_multiple_of(self.config.key_refresh_every().get())
        {
            self.sampler.refresh(dt)?;
        }
        Ok(())
    }
}

impl<const D: usize> DelayedProposal<PachnerStressTriangulation<D>> for PachnerProposalKernel<D> {
    type Plan = PachnerChainPlan<D>;
    type Info = PachnerStepInfo<D>;
    type Error = PachnerStressError;

    fn propose_plan<R: Rng + ?Sized>(
        &mut self,
        state: &PachnerStressTriangulation<D>,
        rng: &mut R,
    ) -> Result<Option<Self::Plan>, Self::Error> {
        self.proposed_steps = self.proposed_steps.saturating_add(1);
        self.maybe_refresh(state)?;
        self.last_result = None;

        let Some(request) = random_pachner_move(state, &self.sampler, rng, self.config) else {
            self.candidate_misses = self.candidate_misses.saturating_add(1);
            self.last_request = None;
            self.last_no_plan_info = Some(PachnerStepInfo::CandidateMiss);
            return Ok(None);
        };

        self.last_request = Some(request);
        match state.propose_pachner(request) {
            Ok(proposal) => {
                self.last_no_plan_info = None;
                Ok(Some(PachnerChainPlan { proposal }))
            }
            Err(error) => {
                self.proposal_rejections = self.proposal_rejections.saturating_add(1);
                self.last_no_plan_info = Some(PachnerStepInfo::ProposalRejected {
                    request,
                    rejection: error,
                });
                Ok(None)
            }
        }
    }

    fn no_plan_info(&mut self) -> Option<Self::Info> {
        self.last_no_plan_info.take()
    }

    fn proposed_log_prob<T: Target<PachnerStressTriangulation<D>>>(
        &self,
        state: &PachnerStressTriangulation<D>,
        _plan: &Self::Plan,
        target: &T,
    ) -> Result<f64, Self::Error> {
        Ok(target.log_prob(state))
    }

    fn info(&self, plan: &Self::Plan) -> Self::Info {
        PachnerStepInfo::Proposed {
            request: *plan.proposal.request(),
        }
    }

    fn commit<R: Rng + ?Sized>(
        &mut self,
        state: &mut PachnerStressTriangulation<D>,
        plan: Self::Plan,
        _rng: &mut R,
    ) -> Result<(), Self::Error> {
        let result = plan.proposal.attempt_on(state)?;
        self.last_result = Some(result);
        Ok(())
    }
}

/// Dispatch one exact diagnostic chain by runtime dimension.
fn run_pachner_stress(
    config: PachnerStressConfig,
    artifacts: &PachnerStressArtifacts,
) -> Result<PachnerStressSummary, PachnerStressError> {
    match config.dimension {
        PachnerStressDimension::Three => run_pachner_stress_dimension::<3>(config, artifacts),
        PachnerStressDimension::Four => run_pachner_stress_dimension::<4>(config, artifacts),
    }
}

/// Run one exact chain and write requested artifacts.
fn run_pachner_stress_dimension<const D: usize>(
    config: PachnerStressConfig,
    artifacts: &PachnerStressArtifacts,
) -> Result<PachnerStressSummary, PachnerStressError> {
    let mut reporter = PachnerStressReporter::try_new(artifacts)?;
    let tri = build_pachner_stress_dt::<D>(config)?;
    let source = PachnerStressSource {
        dimension: D,
        label: config.label,
        vertices: tri.number_of_vertices(),
        simplices: tri.number_of_simplices(),
        seed: config.seed,
    };
    reporter.emit_source(source)?;

    let start = Instant::now();
    let mut report = run_pachner_stress_sequence(tri, config, 1, Some(&mut reporter))?;
    let elapsed = start.elapsed();
    report.elapsed_nanos = elapsed.as_nanos();
    let attempts = u128::try_from(report.attempts)?;
    report.attempts_per_second =
        attempts.saturating_mul(1_000_000_000) / report.elapsed_nanos.max(1);
    reporter.emit_report(config, report)?;
    reporter.finish()?;

    let summary = PachnerStressSummary {
        dimension: D,
        label: config.label,
        configured_vertices: config.vertex_count,
        attempts: config.move_attempts().get(),
        validate_every: config.validate_every().get(),
        key_refresh_every: config.key_refresh_every().get(),
        retry_attempts: config.retry_attempts().get(),
        min_vertex_count: config.min_vertex_count,
        max_vertex_count: config.max_vertex_count,
        seed: config.seed,
        source,
        report,
    };
    if let Some(path) = &artifacts.summary_json {
        write_summary_json(path, &summary)?;
    }
    Ok(summary)
}

/// Builds the initial randomized triangulation for one Monte Carlo stress case.
fn build_pachner_stress_dt<const D: usize>(
    config: PachnerStressConfig,
) -> Result<PachnerStressTriangulation<D>, PachnerStressError> {
    let points = generate_random_points_in_range_seeded::<D>(
        config.vertex_count,
        stress_bounds()?,
        config.seed,
    )?;
    let vertices = try_vertices_from_points(&points)?;
    let options = ConstructionOptions::default().with_retry_policy(RetryPolicy::Shuffled {
        attempts: config.retry_attempts(),
        base_seed: Some(config.seed ^ 0xC0DE_0253_C0DE_0253),
    });

    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .topology_guarantee(TopologyGuarantee::PLManifold)
        .construction_options(options)
        .build_with_kernel(&RobustKernel::new())?;
    let tri = dt.into_triangulation();
    validate_stress_state(&tri, || {
        format!(
            "initial Pachner stress state dimension={D} label={} seed={}",
            config.label, config.seed
        )
    })?;
    Ok(tri)
}

/// Executes one long randomized Pachner sequence and validates periodically.
fn run_pachner_stress_sequence<const D: usize>(
    dt: PachnerStressTriangulation<D>,
    config: PachnerStressConfig,
    sequence: usize,
    mut reporter: Option<&mut PachnerStressReporter>,
) -> Result<PachnerStressReport, PachnerStressError> {
    let mut rng = StdRng::seed_from_u64(config.seed ^ 0x0253_0253_0253_0253);
    let target = FlatPachnerTarget;
    let mut chain = Chain::new(dt, &target)?;
    let mut proposal = PachnerProposalKernel::try_new(chain.state(), config)?;
    let mut recorder = TraceRecorder::new(
        ChainId::new(0),
        [
            "vertices",
            "simplices",
            "candidate_misses",
            "proposal_rejections",
        ],
    )?;
    let start_rss_kib = memory_usage_kib()?;
    let mut max_rss_kib = start_rss_kib;
    let mut validations = 0;
    let mut validation_nanos = 0;
    let mut last_step = None;

    for step in 1..=config.move_attempts().get() {
        let mcmc_step = chain
            .step_delayed(&target, &mut proposal, &mut rng)
            .map_err(|source| PachnerStressError::DelayedStep {
                step,
                source: Box::new(source),
            })?;
        record_stress_step(&mut recorder, &chain, &proposal, &mcmc_step)?;
        last_step = Some(mcmc_step);

        if step.is_multiple_of(config.validate_every().get()) {
            validate_step(
                config,
                step,
                &chain,
                &proposal,
                last_step.as_ref(),
                recorder.trace(),
                &mut validations,
                &mut validation_nanos,
                &mut max_rss_kib,
                &mut reporter,
                sequence,
            )?;
            proposal.sampler.refresh(chain.state())?;
        }
    }

    if !config
        .move_attempts()
        .get()
        .is_multiple_of(config.validate_every().get())
    {
        validate_step(
            config,
            config.move_attempts().get(),
            &chain,
            &proposal,
            last_step.as_ref(),
            recorder.trace(),
            &mut validations,
            &mut validation_nanos,
            &mut max_rss_kib,
            &mut reporter,
            sequence,
        )?;
    }

    let final_rss_kib = memory_usage_kib()?;
    max_rss_kib = max_rss_kib.max(final_rss_kib);
    Ok(PachnerStressReport {
        sequence,
        attempts: config.move_attempts().get(),
        accepted: chain.accepted(),
        rejected: chain.rejected(),
        candidate_misses: proposal.candidate_misses,
        proposal_rejections: proposal.proposal_rejections,
        validations,
        validation_nanos,
        elapsed_nanos: 0,
        attempts_per_second: 0,
        final_vertices: chain.state().number_of_vertices(),
        final_simplices: chain.state().number_of_simplices(),
        start_rss_kib,
        max_rss_kib,
        final_rss_kib,
    })
}

/// Validate one cadence boundary and emit progress.
#[expect(
    clippy::too_many_arguments,
    reason = "stress diagnostics keep all live chain state visible at the validation boundary"
)]
fn validate_step<const D: usize>(
    config: PachnerStressConfig,
    step: usize,
    chain: &Chain<PachnerStressTriangulation<D>>,
    proposal: &PachnerProposalKernel<D>,
    last_step: Option<&DelayedStep<PachnerStepInfo<D>>>,
    trace: &Trace,
    validations: &mut usize,
    validation_nanos: &mut u128,
    max_rss_kib: &mut u64,
    reporter: &mut Option<&mut PachnerStressReporter>,
    sequence: usize,
) -> Result<(), PachnerStressError> {
    let validation_start = Instant::now();
    validate_stress_state(chain.state(), || {
        stress_validation_context(config, step, chain, proposal, last_step, trace)
    })?;
    *validation_nanos += validation_start.elapsed().as_nanos();
    *validations += 1;
    let rss_kib = memory_usage_kib()?;
    *max_rss_kib = (*max_rss_kib).max(rss_kib);
    if let Some(reporter) = reporter.as_mut() {
        reporter.emit_progress(
            config,
            PachnerStressProgress {
                sequence,
                step,
                attempts: config.move_attempts().get(),
                accepted: chain.accepted(),
                rejected: chain.rejected(),
                candidate_misses: proposal.candidate_misses,
                proposal_rejections: proposal.proposal_rejections,
                validations: *validations,
                validation_nanos: *validation_nanos,
                acceptance_rate: chain_acceptance_rate(chain)?,
                vertices: chain.state().number_of_vertices(),
                simplices: chain.state().number_of_simplices(),
                rss_kib,
            },
        )?;
    }
    Ok(())
}

/// Return the coordinate range used for Monte Carlo point clouds.
fn stress_bounds() -> Result<CoordinateRange<f64>, CoordinateRangeError<f64>> {
    CoordinateRange::try_new(0.0_f64, 1.0)
}

/// Validate the invariants Pachner moves are expected to preserve.
fn validate_stress_state<const D: usize>(
    dt: &PachnerStressTriangulation<D>,
    context: impl FnOnce() -> String,
) -> Result<(), PachnerStressError> {
    if let Err(source) = dt.validate() {
        return Err(PachnerStressError::TopologyValidation {
            context: context(),
            source: Box::new(source),
        });
    }
    if let Err(source) = dt.is_valid_embedding() {
        return Err(PachnerStressError::EmbeddingValidation {
            context: context(),
            source: Box::new(source),
        });
    }
    Ok(())
}

/// Return current process memory usage in KiB.
fn memory_usage_kib() -> Result<u64, PachnerStressError> {
    let pid = get_current_pid()
        .map_err(|source| PachnerStressError::CurrentProcess { message: source })?;
    let mut system = System::new_with_specifics(
        RefreshKind::nothing().with_processes(ProcessRefreshKind::nothing().with_memory()),
    );
    system.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing().with_memory(),
    );
    Ok(system
        .process(pid)
        .map_or(0, |process| process.memory() / 1024))
}

/// Convert bounded diagnostic counters into trace-observable values.
fn trace_value(value: usize) -> Result<f64, TryFromIntError> {
    u32::try_from(value).map(f64::from)
}

/// Compute the accepted-step ratio from chain counters without scanning trace rows.
fn chain_acceptance_rate<S>(chain: &Chain<S>) -> Result<f64, PachnerStressError> {
    let total = trace_value(chain.total_steps())?;
    if total == 0.0 {
        Ok(0.0)
    } else {
        Ok(trace_value(chain.accepted())? / total)
    }
}

/// Numeric observables recorded for each completed MCMC step.
fn stress_observables<const D: usize>(
    dt: &PachnerStressTriangulation<D>,
    proposal: &PachnerProposalKernel<D>,
) -> Result<[f64; 4], PachnerStressError> {
    Ok([
        trace_value(dt.number_of_vertices())?,
        trace_value(dt.number_of_simplices())?,
        trace_value(proposal.candidate_misses)?,
        trace_value(proposal.proposal_rejections)?,
    ])
}

/// Record a completed MCMC step in the shared trace format.
fn record_stress_step<const D: usize>(
    recorder: &mut TraceRecorder,
    chain: &Chain<PachnerStressTriangulation<D>>,
    proposal: &PachnerProposalKernel<D>,
    step: &DelayedStep<PachnerStepInfo<D>>,
) -> Result<(), PachnerStressError> {
    recorder.record(
        chain,
        TraceStepOutcome::from(step),
        stress_observables(chain.state(), proposal)?,
    )?;
    Ok(())
}

/// Format the short proposal metadata attached to the last delayed step.
fn describe_step_info<const D: usize>(info: &PachnerStepInfo<D>) -> String {
    match info {
        PachnerStepInfo::Proposed { request } => format!("proposed request={request:?}"),
        PachnerStepInfo::CandidateMiss => String::from("candidate_miss"),
        PachnerStepInfo::ProposalRejected { request, rejection } => {
            format!("proposal_rejected request={request:?} rejection={rejection}")
        }
    }
}

/// Format the tail of the MCMC trace for invariant-failure diagnostics.
fn trace_tail(trace: &Trace) -> String {
    let records = trace.records();
    let start = records.len().saturating_sub(TRACE_TAIL);
    let mut output = String::new();
    for record in &records[start..] {
        let values = record.observable_values();
        let vertices = values.first().copied().unwrap_or_default();
        let simplices = values.get(1).copied().unwrap_or_default();
        let candidate_misses = values.get(2).copied().unwrap_or_default();
        let proposal_rejections = values.get(3).copied().unwrap_or_default();
        let outcome = record.outcome();
        let _ = write!(
            &mut output,
            "step={} accepted={} proposed={} vertices={} simplices={} \
             candidate_misses={} proposal_rejections={}; ",
            record.step(),
            outcome.is_accepted(),
            outcome.had_proposal(),
            vertices,
            simplices,
            candidate_misses,
            proposal_rejections
        );
    }
    output
}

/// Build a diagnostic validation context from chain and trace state.
fn stress_validation_context<const D: usize>(
    config: PachnerStressConfig,
    step: usize,
    chain: &Chain<PachnerStressTriangulation<D>>,
    proposal: &PachnerProposalKernel<D>,
    last_step: Option<&DelayedStep<PachnerStepInfo<D>>>,
    trace: &Trace,
) -> String {
    let chain_id = ChainId::new(0);
    let mut context = format!(
        "Pachner stress validation dimension={D} label={} step={} attempts={} accepted={} \
         rejected={} candidate_misses={} proposal_rejections={} acceptance_rate={:.6} \
         last_request={:?} last_result={:?}",
        config.label,
        step,
        config.move_attempts().get(),
        chain.accepted(),
        chain.rejected(),
        proposal.candidate_misses,
        proposal.proposal_rejections,
        trace.acceptance_rate(chain_id),
        proposal.last_request,
        proposal.last_result
    );
    if let Some(step) = last_step {
        let info = step
            .info
            .as_ref()
            .map_or_else(|| String::from("none"), describe_step_info);
        let _ = write!(
            &mut context,
            " last_step_outcome={:?} last_step_info={} last_log_alpha={:?}",
            step.outcome, info, step.log_alpha
        );
    }
    let _ = write!(&mut context, " trace_tail=[{}]", trace_tail(trace));
    context
}

/// Choose one raw Pachner request from the current cached topology frontier.
fn random_pachner_move<const D: usize>(
    dt: &PachnerStressTriangulation<D>,
    sampler: &MoveSampler,
    rng: &mut (impl Rng + ?Sized),
    config: PachnerStressConfig,
) -> Option<PachnerMove<(), D>> {
    let move_kind_count = if D >= 4 { 6 } else { 5 };
    let mut move_kind = rng.random_range(0..move_kind_count);
    let vertex_count = dt.number_of_vertices();
    if vertex_count >= config.max_vertex_count && move_kind == 0 {
        move_kind = 1;
    } else if vertex_count <= config.min_vertex_count && move_kind == 1 {
        move_kind = 0;
    }

    match move_kind {
        0 => random_k1_insert(dt, sampler, rng),
        1 => sampler
            .random_vertex_key(rng)
            .map(|vertex_key| PachnerMove::K1Remove { vertex_key }),
        2 => sampler
            .random_facet(rng)
            .map(|facet| PachnerMove::K2 { facet }),
        3 => sampler
            .random_edge(rng)
            .map(|edge| PachnerMove::K2Inverse { edge }),
        4 => sampler
            .random_ridge(rng)
            .map(|ridge| PachnerMove::K3 { ridge }),
        5 => random_k3_inverse(dt, sampler, rng),
        _ => None,
    }
}

/// Choose a random simplex and inserts a vertex at its centroid.
fn random_k1_insert<const D: usize>(
    dt: &PachnerStressTriangulation<D>,
    sampler: &MoveSampler,
    rng: &mut (impl Rng + ?Sized),
) -> Option<PachnerMove<(), D>> {
    let simplex_key = sampler.random_simplex_key(rng)?;
    let coords = random_simplex_centroid(dt, simplex_key)?;
    let vertex: Vertex<(), D> = vertex!(coords).ok()?;
    Some(PachnerMove::K1Insert {
        simplex_key,
        vertex,
    })
}

/// Choose three vertices from a random simplex as an inverse k=3 triangle candidate.
fn random_k3_inverse<const D: usize>(
    dt: &PachnerStressTriangulation<D>,
    sampler: &MoveSampler,
    rng: &mut (impl Rng + ?Sized),
) -> Option<PachnerMove<(), D>> {
    let simplex_key = sampler.random_simplex_key(rng)?;
    let vertices = dt.simplex_vertices(simplex_key).ok()?;
    let [a, b, c] = three_distinct_indices(rng, vertices.len())?;
    let triangle = TriangleHandle::try_new(vertices[a], vertices[b], vertices[c]).ok()?;
    Some(PachnerMove::K3Inverse { triangle })
}

/// Compute a live simplex centroid when the cached key still exists.
fn random_simplex_centroid<const D: usize>(
    dt: &PachnerStressTriangulation<D>,
    simplex_key: SimplexKey,
) -> Option<[f64; D]> {
    let vertices = dt.simplex_vertices(simplex_key).ok()?;
    let mut coords = [0.0; D];
    for &vertex_key in vertices {
        let vertex_coords = dt.vertex_coords(vertex_key)?;
        for (coord, value) in coords.iter_mut().zip(vertex_coords) {
            *coord += *value;
        }
    }

    let vertex_count = f64::from(u32::try_from(vertices.len()).ok()?);
    for coord in &mut coords {
        *coord /= vertex_count;
    }
    Some(coords)
}

/// Choose three distinct indices from a collection length.
fn three_distinct_indices(rng: &mut (impl Rng + ?Sized), len: usize) -> Option<[usize; 3]> {
    if len < 3 {
        return None;
    }
    let first = rng.random_range(0..len);
    let mut second = rng.random_range(0..len);
    while second == first {
        second = rng.random_range(0..len);
    }
    let mut third = rng.random_range(0..len);
    while third == first || third == second {
        third = rng.random_range(0..len);
    }
    Some([first, second, third])
}

/// Create a parent directory when the caller requested a file artifact.
fn ensure_parent_dir(path: &Path) -> Result<(), io::Error> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

/// Open a progress CSV and write its stable header.
fn create_progress_writer(path: &Path) -> Result<BufWriter<File>, PachnerStressError> {
    ensure_parent_dir(path)?;
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "dimension,label,sequence,step,attempts,accepted,rejected,candidate_misses,\
         proposal_rejections,validations,validation_nanos,acceptance_rate,vertices,simplices,rss_kib"
    )?;
    Ok(writer)
}

/// Write the stable run-level summary JSON artifact.
fn write_summary_json(
    path: &Path,
    summary: &PachnerStressSummary,
) -> Result<(), PachnerStressError> {
    ensure_parent_dir(path)?;
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, summary)?;
    writer.flush()?;
    Ok(())
}

/// Errors surfaced by the Pachner stress diagnostic runner.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PachnerStressError {
    /// A positive count was required.
    #[error("{argument} must be positive, got {value}")]
    NonPositive {
        /// Argument that received the invalid value.
        argument: PachnerStressCountArgument,
        /// Provided value.
        value: usize,
    },

    /// The requested vertex count cannot support the dimension.
    #[error("{dimension}D stress requires at least {minimum} vertices, got {vertices}")]
    TooFewVertices {
        /// Runtime dimension.
        dimension: usize,
        /// Requested vertex count.
        vertices: usize,
        /// Minimum supported vertex count.
        minimum: usize,
    },

    /// I/O failed while writing diagnostic artifacts.
    #[error("failed to write Pachner stress artifact: {source}")]
    Io {
        /// Underlying I/O error.
        #[from]
        source: io::Error,
    },

    /// JSON serialization failed while writing the summary artifact.
    #[error("failed to write Pachner stress summary JSON: {source}")]
    Json {
        /// Underlying JSON serialization error.
        #[from]
        source: serde_json::Error,
    },

    /// Coordinate range construction failed.
    #[error("failed to build random point coordinate range: {source}")]
    CoordinateRange {
        /// Underlying coordinate range error.
        #[from]
        source: CoordinateRangeError<f64>,
    },

    /// Random point generation failed.
    #[error("failed to generate random stress points: {source}")]
    PointGeneration {
        /// Underlying random point generation error.
        #[from]
        source: RandomPointGenerationError,
    },

    /// Point-to-vertex conversion failed.
    #[error("failed to convert random points into vertices: {source}")]
    CoordinateConversion {
        /// Underlying coordinate conversion error.
        #[from]
        source: CoordinateConversionError,
    },

    /// Delaunay construction failed.
    #[error("failed to construct initial Delaunay triangulation: {source}")]
    Construction {
        /// Underlying construction error.
        #[from]
        source: DelaunayTriangulationConstructionError,
    },

    /// Public topology query failed.
    #[error("Pachner stress topology query failed: {source}")]
    Query {
        /// Underlying query error.
        #[from]
        source: QueryError,
    },

    /// TDS lookup failed.
    #[error("Pachner stress TDS lookup failed: {source}")]
    Tds {
        /// Underlying TDS error.
        #[from]
        source: TdsError,
    },

    /// Topology validation failed.
    #[error("{context}: topology validation failed: {source}")]
    TopologyValidation {
        /// Diagnostic chain context.
        context: String,
        /// Underlying invariant error.
        #[source]
        source: Box<InvariantError>,
    },

    /// Embedding validation failed.
    #[error("{context}: embedding validation failed: {source}")]
    EmbeddingValidation {
        /// Diagnostic chain context.
        context: String,
        /// Underlying invariant error.
        #[source]
        source: Box<TriangulationEmbeddingValidationError>,
    },

    /// Public facet query failed.
    #[error("Pachner stress facet query failed: {source}")]
    Facet {
        /// Underlying facet query error.
        #[from]
        source: FacetError,
    },

    /// Initial MCMC chain setup failed.
    #[error("failed to create Pachner stress MCMC chain: {source}")]
    Mcmc {
        /// Underlying MCMC error.
        #[from]
        source: McmcError,
    },

    /// MCMC trace setup or recording failed.
    #[error("failed to record Pachner stress MCMC trace: {source}")]
    Trace {
        /// Underlying trace error.
        #[from]
        source: TraceError,
    },

    /// One delayed MCMC step failed exceptionally.
    #[error("Pachner stress MCMC step {step} failed: {source}")]
    DelayedStep {
        /// One-based attempted move step.
        step: usize,
        /// Underlying delayed-step error.
        #[source]
        source: Box<DelayedStepError<Self>>,
    },

    /// A committed Pachner proposal failed.
    #[error("Pachner proposal commit failed: {source}")]
    Flip {
        /// Underlying flip error.
        #[from]
        source: FlipError,
    },

    /// Diagnostic counter conversion exceeded f64-safe trace storage.
    #[error("Pachner stress diagnostic counter conversion failed: {source}")]
    CounterConversion {
        /// Underlying integer conversion error.
        #[from]
        source: TryFromIntError,
    },

    /// The current process could not be identified for RSS telemetry.
    #[error("failed to query current process for RSS telemetry: {message}")]
    CurrentProcess {
        /// Platform-specific process lookup message.
        message: &'static str,
    },

    /// Two requested artifacts target the same path.
    #[error("progress CSV and summary JSON must use different paths: {path:?}")]
    DuplicateArtifactPath {
        /// Path supplied for multiple artifacts.
        path: PathBuf,
    },
}

/// Convert a raw positive count into `NonZeroUsize`.
fn positive_nonzero(
    argument: PachnerStressCountArgument,
    value: usize,
) -> Result<NonZeroUsize, PachnerStressError> {
    NonZeroUsize::new(value).ok_or(PachnerStressError::NonPositive { argument, value })
}

#[cfg(test)]
mod tests {
    use std::{
        num::NonZeroUsize,
        path::{Path, PathBuf},
    };

    use clap::Parser;

    use super::{
        DelaunayCliArgs, DelaunayCommand, GenerateCommand, GenerateConfig, GenerateDistribution,
        PachnerStressArtifacts, PachnerStressConfig, PachnerStressCountArgument,
        PachnerStressDimension, PachnerStressError, build_validation_demo_export, positive_nonzero,
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

        assert_eq!(config.distribution, GenerateDistribution::Cube);
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

        assert_eq!(config.distribution, GenerateDistribution::Ball);
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
    fn pachner_positive_count_errors_preserve_typed_argument() {
        let error = positive_nonzero(PachnerStressCountArgument::ValidateEvery, 0)
            .expect_err("zero should fail positive-count validation");

        match error {
            PachnerStressError::NonPositive { argument, value } => {
                assert_eq!(argument, PachnerStressCountArgument::ValidateEvery);
                assert_eq!(argument.to_string(), "validate_every");
                assert_eq!(value, 0);
            }
            other => panic!("expected NonPositive error, got {other:?}"),
        }
    }

    #[test]
    fn pachner_current_process_error_preserves_static_message() {
        let error = PachnerStressError::CurrentProcess {
            message: "pid unavailable",
        };

        match error {
            PachnerStressError::CurrentProcess { message } => {
                assert_eq!(message, "pid unavailable");
            }
            other => panic!("expected CurrentProcess error, got {other:?}"),
        }
    }

    #[test]
    fn pachner_artifacts_reject_duplicate_paths_before_storage() {
        let path = PathBuf::from("target/notebooks/pachner/shared.csv");
        let error = PachnerStressArtifacts::try_new(Some(path.clone()), Some(path), true)
            .expect_err("duplicate artifact paths should fail validation");

        match error {
            PachnerStressError::DuplicateArtifactPath { path } => {
                assert_eq!(path, Path::new("target/notebooks/pachner/shared.csv"));
            }
            other => panic!("expected DuplicateArtifactPath error, got {other:?}"),
        }
    }

    #[test]
    fn pachner_stress_clamps_validate_every_to_move_attempts() {
        let config = PachnerStressConfig::try_new(
            PachnerStressDimension::Three,
            5,
            NonZeroUsize::new(2).expect("literal is nonzero"),
            NonZeroUsize::new(100).expect("literal is nonzero"),
            NonZeroUsize::new(7).expect("literal is nonzero"),
            NonZeroUsize::new(4).expect("literal is nonzero"),
            42,
        )
        .expect("valid 3D Pachner stress config should build");

        assert_eq!(config.move_attempts().get(), 2);
        assert_eq!(config.validate_every().get(), 2);
        assert_eq!(config.key_refresh_every().get(), 7);
        assert_eq!(config.retry_attempts().get(), 4);
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
                    config.output.as_deref(),
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

    #[test]
    fn pachner_stress_rejects_empty_artifact_paths_during_parsing() {
        assert_empty_path_rejected_by_clap(
            &["delaunay", "pachner-stress", "--progress-csv", ""],
            "--progress-csv",
        );
        assert_empty_path_rejected_by_clap(
            &["delaunay", "pachner-stress", "--summary-json", ""],
            "--summary-json",
        );
    }

    #[test]
    fn validation_demo_export_covers_each_validation_level() {
        let export = build_validation_demo_export().expect("validation demo should build");

        assert_eq!(export.schema, "delaunay.validation_demo");
        assert_eq!(export.schema_version, 1);
        assert_eq!(export.valid_baseline.status, "passed");
        assert_eq!(
            export
                .cases
                .iter()
                .map(|case| case.level)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5]
        );
        assert!(
            export
                .cases
                .iter()
                .all(|case| case.status == "failed_as_expected")
        );
        assert_eq!(export.cases[3].layer, "Valid affine realization");
    }

    #[test]
    fn validation_demo_export_is_reproducible_across_runs() {
        let first = build_validation_demo_export().expect("first validation demo should build");
        let second = build_validation_demo_export().expect("second validation demo should build");

        let first_json =
            serde_json::to_value(&first).expect("first validation demo should serialize");
        let second_json =
            serde_json::to_value(&second).expect("second validation demo should serialize");

        assert_eq!(first_json, second_json);
    }
}
