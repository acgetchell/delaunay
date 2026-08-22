#![forbid(unsafe_code)]

//! Deterministic spherical-hero artifact command for the CLI.

use std::{num::NonZeroU32, path::PathBuf};

use clap::Args;
use delaunay::prelude::construction::SphericalDelaunayBuilder;
use serde::Serialize;

use crate::{
    cli_output::{ArtifactPath, write_json_output},
    config::{CliError, VertexCountTarget},
};

const SPHERICAL_HERO_EXPORT_SCHEMA: &str = "delaunay.spherical_hero";
const SPHERICAL_HERO_EXPORT_SCHEMA_VERSION: u32 = 1;

/// Raw command-line arguments for `delaunay spherical-hero`.
#[derive(Debug, Args)]
pub struct SphericalHeroArgs {
    /// Number of deterministic Fibonacci-sphere vertices.
    #[arg(short = 'n', long, default_value_t = 160)]
    vertices: usize,
    /// Write JSON to a file instead of stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,
}

impl SphericalHeroArgs {
    /// Validate the `S^2` vertex count.
    pub fn into_validated(self) -> Result<SphericalHeroConfig, CliError> {
        let vertices = u32::try_from(self.vertices).map_err(|_| CliError::TooManyVertices {
            target: VertexCountTarget::SphericalHero,
            vertices: self.vertices,
            maximum: u32::MAX,
        })?;
        let vertices = NonZeroU32::new(vertices)
            .filter(|vertices| vertices.get() >= 4)
            .ok_or(CliError::TooFewVertices {
                target: VertexCountTarget::SphericalHero,
                vertices: self.vertices,
                minimum: 4,
            })?;
        Ok(SphericalHeroConfig {
            vertices,
            output: self.output.map(ArtifactPath::try_new).transpose()?,
        })
    }
}

/// Validated deterministic `S^2` hero configuration.
#[derive(Debug)]
pub struct SphericalHeroConfig {
    vertices: NonZeroU32,
    output: Option<ArtifactPath>,
}

/// Detached `S^2` triangulation rendered by the spherical hero notebook.
#[derive(Debug, Serialize)]
struct SphericalHeroExport {
    schema: &'static str,
    schema_version: u32,
    intrinsic_dimension: usize,
    ambient_dimension: usize,
    vertices: Vec<Vec<f64>>,
    simplices: Vec<Vec<usize>>,
}

/// Build and emit a deterministic `S^2` Delaunay triangulation in `R^3`.
pub fn run(config: &SphericalHeroConfig) -> Result<(), CliError> {
    let vertex_count = config.vertices.get();
    let count = f64::from(vertex_count);
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let points = (0..vertex_count)
        .map(|index| {
            let position = f64::from(index) + 0.5;
            let z = 1.0 - 2.0 * position / count;
            let radial = (1.0 - z * z).sqrt();
            let azimuth = golden_angle * f64::from(index);
            [radial * azimuth.cos(), radial * azimuth.sin(), z]
        })
        .collect::<Vec<_>>();
    let triangulation = SphericalDelaunayBuilder::<2>::try_new(points)?.build()?;
    let export = SphericalHeroExport {
        schema: SPHERICAL_HERO_EXPORT_SCHEMA,
        schema_version: SPHERICAL_HERO_EXPORT_SCHEMA_VERSION,
        intrinsic_dimension: triangulation.dimension(),
        ambient_dimension: triangulation.ambient_dimension(),
        vertices: triangulation
            .points()
            .iter()
            .map(|point| point.coords().to_vec())
            .collect(),
        simplices: triangulation
            .simplices()
            .iter()
            .map(|simplex| simplex.vertex_indices().to_vec())
            .collect(),
    };
    write_json_output(&export, config.output.as_ref())?;
    Ok(())
}
