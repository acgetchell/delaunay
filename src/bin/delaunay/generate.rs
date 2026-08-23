#![forbid(unsafe_code)]

//! Random Euclidean construction and detached export commands for the CLI.

use std::{num::NonZeroUsize, path::PathBuf};

use clap::{Args, ValueEnum};
use delaunay::{
    DelaunayTriangulation,
    prelude::{
        construction::{DelaunayTriangulationBuilder, TopologyGuarantee},
        generators::{
            generate_random_points_in_ball_seeded, generate_random_points_in_range_seeded,
        },
        geometry::{CoordinateRange, ExactPredicates, RobustKernel},
        query::ConvexHull,
    },
    try_vertices_from_points,
};
use serde::Serialize;
use uuid::Uuid;

use crate::{
    cli_output::{ArtifactPath, write_json_output},
    config::{CliError, VertexCountTarget},
};

const DEFAULT_GENERATE_DIMENSION: usize = 3;
const DEFAULT_GENERATE_VERTICES: usize = 100;
const DEFAULT_GENERATE_SEED: u64 = 0xD3_1A_05_25_03;
const CONVEX_HULL_EXPORT_SCHEMA: &str = "delaunay.convex_hull";
const CONVEX_HULL_EXPORT_SCHEMA_VERSION: u32 = 1;

/// Generated object requested by the companion binary.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum GenerateKind {
    /// Emit the generated Delaunay triangulation as the crate's serde JSON.
    Triangulation,
    /// Emit generic simplicial-complex visualization primitives as JSON.
    Visualization,
    /// Emit the generated triangulation's convex-hull facets as JSON.
    ConvexHull,
}

/// Random point distribution requested by the companion binary.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum GenerateDistribution {
    /// Uniform points in the unit cube `[0, 1]^D`.
    Cube,
    /// Uniform points in the radius-1 ball centered at the origin.
    Ball,
}

/// Raw command-line arguments for `delaunay generate`.
#[derive(Debug, Args)]
pub struct GenerateArgs {
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
    pub fn into_validated(self) -> Result<GenerateCommand, CliError> {
        match self.dimension {
            2 => Ok(GenerateCommand::D2(GenerateConfig::try_new(self)?)),
            3 => Ok(GenerateCommand::D3(GenerateConfig::try_new(self)?)),
            4 => Ok(GenerateCommand::D4(GenerateConfig::try_new(self)?)),
            5 => Ok(GenerateCommand::D5(GenerateConfig::try_new(self)?)),
            dimension => Err(CliError::UnsupportedGenerateDimension { dimension }),
        }
    }
}

/// Validated generation command by dimension.
#[derive(Debug)]
pub enum GenerateCommand {
    D2(GenerateConfig<2>),
    D3(GenerateConfig<3>),
    D4(GenerateConfig<4>),
    D5(GenerateConfig<5>),
}

/// Validated generation configuration for one const-generic dimension.
#[derive(Debug)]
pub struct GenerateConfig<const D: usize> {
    kind: GenerateKind,
    pub vertices: NonZeroUsize,
    pub distribution: GenerateDistribution,
    seed: u64,
    output: Option<ArtifactPath>,
}

impl<const D: usize> GenerateConfig<D> {
    /// Validate dimension-dependent generation limits.
    fn try_new(args: GenerateArgs) -> Result<Self, CliError> {
        let minimum = D + 1;
        let vertices = validated_nonzero_count(
            args.vertices,
            |vertices| vertices.get() >= minimum,
            || CliError::TooFewVertices {
                target: VertexCountTarget::EuclideanGeneration { dimension: D },
                vertices: args.vertices,
                minimum,
            },
        )?;

        Ok(Self {
            kind: args.kind,
            vertices,
            distribution: args.distribution,
            seed: args.seed,
            output: args.output.map(ArtifactPath::try_new).transpose()?,
        })
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

/// Dispatch a validated generation command to its const-generic implementation.
pub fn run(command: &GenerateCommand) -> Result<(), CliError> {
    match command {
        GenerateCommand::D2(config) => run_dimension(config),
        GenerateCommand::D3(config) => run_dimension(config),
        GenerateCommand::D4(config) => run_dimension(config),
        GenerateCommand::D5(config) => run_dimension(config),
    }
}

/// Generate and emit one artifact for a concrete dimension.
fn run_dimension<const D: usize>(config: &GenerateConfig<D>) -> Result<(), CliError>
where
    RobustKernel<f64>: ExactPredicates<D>,
{
    let triangulation = build_delaunay::<D>(config.vertices, config.seed, config.distribution)?;
    match config.kind {
        GenerateKind::Triangulation => {
            write_json_output(&triangulation, config.output.as_ref())?;
        }
        GenerateKind::Visualization => {
            let visualization = triangulation.to_visualization_data()?;
            write_json_output(&visualization, config.output.as_ref())?;
        }
        GenerateKind::ConvexHull => {
            let hull = build_convex_hull_export(&triangulation)?;
            write_json_output(&hull, config.output.as_ref())?;
        }
    }
    Ok(())
}

/// Build a random PL-manifold Delaunay triangulation for CLI export.
fn build_delaunay<const D: usize>(
    vertex_count: NonZeroUsize,
    seed: u64,
    distribution: GenerateDistribution,
) -> Result<DelaunayTriangulation<RobustKernel<f64>, (), (), D>, CliError>
where
    RobustKernel<f64>: ExactPredicates<D>,
{
    let points = match distribution {
        GenerateDistribution::Cube => generate_random_points_in_range_seeded::<D>(
            vertex_count.get(),
            CoordinateRange::try_new(0.0_f64, 1.0)?,
            seed,
        )?,
        GenerateDistribution::Ball => {
            generate_random_points_in_ball_seeded::<D>(vertex_count.get(), 1.0, seed)?
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
        .facets()
        .enumerate()
        .map(|(index, facet)| {
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

/// Parse a count once while preserving the command's threshold diagnostic.
fn validated_nonzero_count<E>(
    value: usize,
    is_valid: impl FnOnce(NonZeroUsize) -> bool,
    error: impl FnOnce() -> E,
) -> Result<NonZeroUsize, E> {
    NonZeroUsize::new(value)
        .filter(|count| is_valid(*count))
        .ok_or_else(error)
}
