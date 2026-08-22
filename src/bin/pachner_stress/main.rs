#![forbid(unsafe_code)]
#![deny(dead_code_pub_in_binary)]

//! Standalone validated Pachner-move stress diagnostic.

#[path = "../shared/cli_output.rs"]
mod cli_output;

use std::{
    ffi::OsString,
    fmt::{self, Display},
    fs::{self, File},
    io::{self, BufWriter, Write},
    num::{NonZeroUsize, TryFromIntError},
    path::{Component, Path, PathBuf},
    process::ExitCode,
    time::Instant,
};

use clap::{Parser, ValueEnum};
use delaunay::{
    InvariantError,
    prelude::{
        construction::{
            ConstructionOptions, DelaunayTriangulationBuilder,
            DelaunayTriangulationConstructionError, RetryPolicy, TopologyGuarantee, Vertex, vertex,
        },
        generators::{RandomPointGenerationError, generate_random_points_in_range_seeded},
        geometry::{
            CoordinateConversionError, CoordinateRange, CoordinateRangeError, RobustKernel,
        },
        pachner::{
            EdgeKey, FacetHandle, FlipError, PachnerMove, PachnerMoveResult, PachnerMoves,
            RidgeHandle, SimplexKey, TriangleHandle, TriangleHandleError, VertexKey,
        },
        query::QueryError,
        tds::{FacetError, TdsError},
        triangulation::Triangulation,
    },
    try_vertices_from_points,
};
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
use serde::Serialize;

use crate::cli_output::{ArtifactOutputError, ArtifactPath, write_json_output};

type PachnerStressTriangulation<const D: usize> = Triangulation<RobustKernel<f64>, (), (), D>;

const DEFAULT_3D_VERTICES: usize = 10_000;
const DEFAULT_4D_VERTICES: usize = 1_000;
const DEFAULT_ATTEMPTS: usize = 100_000;
const DEFAULT_KEY_REFRESH_EVERY: usize = 256;
const DEFAULT_RETRY_ATTEMPTS: usize = 24;
const DEFAULT_VALIDATE_EVERY: usize = 1_000;
const PACHNER_STRESS_VALIDATION_SCOPE_LABEL: &str = "topology";
const PACHNER_STRESS_EXPORT_SCHEMA: &str = "delaunay.pachner_stress";
const PACHNER_STRESS_EXPORT_SCHEMA_VERSION: u32 = 1;
const DEFAULT_VERTEX_GROWTH_DIVISOR: usize = 10;
const DEFAULT_VERTEX_SHRINK_DIVISOR: usize = 20;

/// Raw command-line arguments for the `pachner-stress` diagnostic binary.
#[derive(Debug, Parser)]
#[command(
    name = "pachner-stress",
    version,
    about = "Run validated Pachner-move stress diagnostics"
)]
struct PachnerStressArgs {
    /// Stress dimension.
    #[arg(long, value_enum, default_value = "3d")]
    dimension: PachnerStressDimension,
    /// Stress workload to execute.
    #[arg(long, value_enum, default_value = "round-trip")]
    mode: PachnerStressMode,
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
    fn into_validated(self) -> Result<PachnerStressCommand, PachnerStressError> {
        let config = PachnerStressConfig::try_new(PachnerStressConfigInput {
            mode: self.mode,
            dimension: self.dimension,
            vertex_count: self
                .vertices
                .unwrap_or_else(|| self.dimension.default_vertices()),
            move_attempts: positive_nonzero(PachnerStressCountArgument::Attempts, self.attempts)?,
            validate_every: positive_nonzero(
                PachnerStressCountArgument::ValidateEvery,
                self.validate_every,
            )?,
            key_refresh_every: positive_nonzero(
                PachnerStressCountArgument::KeyRefreshEvery,
                self.key_refresh_every,
            )?,
            retry_attempts: positive_nonzero(
                PachnerStressCountArgument::RetryAttempts,
                self.retry_attempts,
            )?,
            seed: self.seed.unwrap_or_else(|| self.dimension.default_seed()),
        })?;
        let artifacts =
            PachnerStressArtifacts::try_new(self.progress_csv, self.summary_json, !self.quiet)?;
        Ok(PachnerStressCommand { config, artifacts })
    }
}

fn main() -> ExitCode {
    let command = match PachnerStressArgs::parse().into_validated() {
        Ok(command) => command,
        Err(error) => return exit_with_error(error),
    };

    run(&command).map_or_else(exit_with_error, |()| ExitCode::SUCCESS)
}

fn exit_with_error(error: impl Display) -> ExitCode {
    let stderr = io::stderr();
    let mut handle = stderr.lock();
    let _ = writeln!(handle, "error: {error}");
    ExitCode::FAILURE
}

/// Opaque validated Pachner stress command accepted by the CLI dispatcher.
#[derive(Debug)]
struct PachnerStressCommand {
    config: PachnerStressConfig,
    artifacts: PachnerStressArtifacts,
}

/// Execute one validated Pachner stress command.
fn run(command: &PachnerStressCommand) -> Result<(), PachnerStressError> {
    run_pachner_stress(command.config, &command.artifacts).map(|_| ())
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

/// Pachner stress workload selected by the CLI.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum PachnerStressMode {
    /// Apply a forward move and immediately apply its inverse witness.
    #[value(name = "round-trip")]
    RoundTrip,
    /// Apply accepted valid moves over an evolving triangulation.
    #[value(name = "random-walk")]
    RandomWalk,
}

impl PachnerStressMode {
    /// Return the label used in telemetry and artifacts.
    const fn label(self) -> &'static str {
        match self {
            Self::RoundTrip => "round-trip",
            Self::RandomWalk => "random-walk",
        }
    }

    /// Return the expected mutation attempts for one configured step.
    const fn expected_moves_per_step(self) -> usize {
        match self {
            Self::RoundTrip => 2,
            Self::RandomWalk => 1,
        }
    }
}

/// Positive count arguments validated by the Pachner stress diagnostic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
enum PachnerStressCountArgument {
    /// Attempted Pachner moves.
    Attempts,
    /// Validation and progress-reporting cadence.
    ValidateEvery,
    /// Cached-key refresh cadence.
    KeyRefreshEvery,
    /// Randomized construction retry attempts.
    RetryAttempts,
}

/// Stress move context for an inserted-face arity diagnostic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
enum PachnerStressInsertedFaceContext {
    /// Any forward move whose inserted face should determine an inverse move.
    ForwardMove,
    /// The edge witness expected after a k=2 forward move.
    K2Move,
}

impl Display for PachnerStressInsertedFaceContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::ForwardMove => "forward Pachner move",
            Self::K2Move => "k=2",
        };
        f.write_str(label)
    }
}

/// Expected inserted-face arity for a stress move witness.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
enum PachnerStressInsertedFaceArity {
    /// A forward move must insert a vertex, edge, or triangle for supported inverses.
    InvertibleForwardMove,
    /// A k=2 forward move must insert an edge.
    Edge,
}

impl Display for PachnerStressInsertedFaceArity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::InvertibleForwardMove => "1, 2, or 3",
            Self::Edge => "2",
        };
        f.write_str(label)
    }
}

impl PachnerStressCountArgument {
    /// Return the CLI flag spelling used in user-facing diagnostics.
    const fn as_str(self) -> &'static str {
        match self {
            Self::Attempts => "--attempts",
            Self::ValidateEvery => "--validate-every",
            Self::KeyRefreshEvery => "--key-refresh-every",
            Self::RetryAttempts => "--retry-attempts",
        }
    }
}

impl Display for PachnerStressCountArgument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Configuration for one exact Pachner stress workload.
#[derive(Clone, Copy, Debug)]
struct PachnerStressConfig {
    mode: PachnerStressMode,
    dimension: PachnerStressDimension,
    pub vertex_count: NonZeroUsize,
    move_attempts: NonZeroUsize,
    validate_every: NonZeroUsize,
    key_refresh_every: NonZeroUsize,
    retry_attempts: NonZeroUsize,
    min_vertex_count: usize,
    max_vertex_count: usize,
    seed: u64,
}

/// Validated constructor input for one Pachner stress workload.
#[derive(Clone, Copy, Debug)]
struct PachnerStressConfigInput {
    pub mode: PachnerStressMode,
    pub dimension: PachnerStressDimension,
    pub vertex_count: usize,
    pub move_attempts: NonZeroUsize,
    pub validate_every: NonZeroUsize,
    pub key_refresh_every: NonZeroUsize,
    pub retry_attempts: NonZeroUsize,
    pub seed: u64,
}

impl PachnerStressConfig {
    /// Build a validated stress configuration from command-line values.
    fn try_new(input: PachnerStressConfigInput) -> Result<Self, PachnerStressError> {
        let minimum_vertices = input.dimension.value() + 1;
        let vertex_count = validated_nonzero_count(
            input.vertex_count,
            |vertex_count| vertex_count.get() >= minimum_vertices,
            || PachnerStressError::TooFewVertices {
                dimension: input.dimension.value(),
                vertices: input.vertex_count,
                minimum: minimum_vertices,
            },
        )?;
        let validate_every = input.validate_every.min(input.move_attempts);
        let growth_slack =
            (vertex_count.get() / DEFAULT_VERTEX_GROWTH_DIVISOR).max(input.dimension.value() + 1);
        let shrink_slack = vertex_count.get() / DEFAULT_VERTEX_SHRINK_DIVISOR;

        Ok(Self {
            mode: input.mode,
            dimension: input.dimension,
            vertex_count,
            move_attempts: input.move_attempts,
            validate_every,
            key_refresh_every: input.key_refresh_every,
            retry_attempts: input.retry_attempts,
            min_vertex_count: vertex_count
                .get()
                .saturating_sub(shrink_slack)
                .max(input.dimension.value() + 1),
            max_vertex_count: vertex_count.get().saturating_add(growth_slack),
            seed: input.seed,
        })
    }

    /// Positive number of attempted moves in this exact workload.
    pub const fn move_attempts(self) -> NonZeroUsize {
        self.move_attempts
    }

    /// Positive periodic validation cadence.
    pub const fn validate_every(self) -> NonZeroUsize {
        self.validate_every
    }

    /// Positive cached-key refresh cadence.
    pub const fn key_refresh_every(self) -> NonZeroUsize {
        self.key_refresh_every
    }

    /// Retry attempts for randomized Delaunay construction.
    pub const fn retry_attempts(self) -> NonZeroUsize {
        self.retry_attempts
    }

    /// Stable dimension label written to telemetry and artifacts.
    const fn label(self) -> &'static str {
        self.dimension.label()
    }
}

/// Artifact paths and stdout behavior for one diagnostic run.
#[derive(Debug)]
struct PachnerStressArtifacts {
    progress_csv: Option<ArtifactPath>,
    summary_json: Option<ArtifactPath>,
    stdout: bool,
}

impl PachnerStressArtifacts {
    /// Build a validated artifact configuration for one diagnostic run.
    fn try_new(
        progress_csv: Option<PathBuf>,
        summary_json: Option<PathBuf>,
        stdout: bool,
    ) -> Result<Self, PachnerStressError> {
        let progress_csv = progress_csv.map(ArtifactPath::try_new).transpose()?;
        let summary_json = summary_json.map(ArtifactPath::try_new).transpose()?;
        if let (Some(progress_csv), Some(summary_json)) = (&progress_csv, &summary_json)
            && artifact_paths_conflict(progress_csv, summary_json)?
        {
            return Err(PachnerStressError::DuplicateArtifactPath {
                progress_path: progress_csv.as_path().to_owned(),
                summary_path: summary_json.as_path().to_owned(),
            });
        }

        Ok(Self {
            progress_csv,
            summary_json,
            stdout,
        })
    }
}

fn artifact_paths_conflict(
    first: &ArtifactPath,
    second: &ArtifactPath,
) -> Result<bool, PachnerStressError> {
    let first_identity = artifact_path_identity(first.as_path())?;
    let second_identity = artifact_path_identity(second.as_path())?;
    if first_identity == second_identity {
        return Ok(true);
    }

    let first_exists = try_artifact_exists(first.as_path())?;
    let second_exists = try_artifact_exists(second.as_path())?;
    if !first_exists || !second_exists {
        return Ok(false);
    }

    same_file::is_same_file(first.as_path(), second.as_path()).map_err(|source| {
        PachnerStressError::ArtifactCompareIdentity {
            first: first.as_path().to_owned(),
            second: second.as_path().to_owned(),
            source,
        }
    })
}

fn artifact_path_identity(path: &Path) -> Result<PathBuf, PachnerStressError> {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        let working_directory = std::env::current_dir()
            .map_err(|source| PachnerStressError::ArtifactWorkingDirectory { source })?;
        working_directory.join(path)
    };
    let normalized = normalize_absolute_path(&absolute);
    canonicalize_existing_prefix(&normalized).map(platform_identity)
}

fn normalize_absolute_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => {
                normalized.push(component.as_os_str());
            }
        }
    }
    normalized
}

fn canonicalize_existing_prefix(path: &Path) -> Result<PathBuf, PachnerStressError> {
    let mut existing_prefix = path;
    let mut missing_suffix = Vec::<OsString>::new();

    loop {
        match existing_prefix.try_exists() {
            Ok(true) => break,
            Ok(false) => {
                let Some(name) = existing_prefix.file_name() else {
                    return Err(PachnerStressError::ArtifactUnresolvablePath {
                        path: path.to_owned(),
                    });
                };
                missing_suffix.push(name.to_owned());
                let Some(parent) = existing_prefix.parent() else {
                    return Err(PachnerStressError::ArtifactUnresolvablePath {
                        path: path.to_owned(),
                    });
                };
                existing_prefix = parent;
            }
            Err(source) => {
                return Err(PachnerStressError::ArtifactInspect {
                    path: existing_prefix.to_owned(),
                    source,
                });
            }
        }
    }

    let mut identity = fs::canonicalize(existing_prefix).map_err(|source| {
        PachnerStressError::ArtifactResolveIdentity {
            path: existing_prefix.to_owned(),
            source,
        }
    })?;
    for component in missing_suffix.into_iter().rev() {
        identity.push(component);
    }
    Ok(identity)
}

fn platform_identity(path: PathBuf) -> PathBuf {
    if cfg!(any(target_os = "macos", target_os = "windows")) {
        PathBuf::from(path.to_string_lossy().to_lowercase())
    } else {
        path
    }
}

fn try_artifact_exists(path: &Path) -> Result<bool, PachnerStressError> {
    path.try_exists()
        .map_err(|source| PachnerStressError::ArtifactInspect {
            path: path.to_owned(),
            source,
        })
}

/// Initial triangulation metadata emitted before the stress workload starts.
#[derive(Clone, Copy, Debug, Serialize)]
struct PachnerStressSource {
    dimension: usize,
    label: &'static str,
    mode: &'static str,
    validation_scope: &'static str,
    vertices: usize,
    simplices: usize,
    seed: u64,
}

/// Final aggregate metrics for one exact Pachner stress workload.
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
}

/// JSON summary written by the diagnostic CLI.
#[derive(Clone, Debug, Serialize)]
struct PachnerStressSummary {
    schema: &'static str,
    schema_version: u32,
    dimension: usize,
    label: &'static str,
    mode: &'static str,
    validation_scope: &'static str,
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
}

/// Mutable counters accumulated by one stress workload.
#[derive(Clone, Copy, Debug, Default)]
struct PachnerStressCounters {
    accepted: usize,
    candidate_misses: usize,
    proposal_rejections: usize,
    validations: usize,
    validation_nanos: u128,
}

impl PachnerStressCounters {
    /// Count steps that did not produce an accepted move.
    const fn rejected(self) -> usize {
        self.candidate_misses
            .saturating_add(self.proposal_rejections)
    }
}

/// Report sink for scriptable Pachner stress artifacts.
struct PachnerStressReporter {
    stdout: bool,
    progress: Option<ProgressArtifact>,
}

/// Open progress destination retained so streaming errors keep path context.
struct ProgressArtifact {
    path: ArtifactPath,
    writer: BufWriter<File>,
}

impl PachnerStressReporter {
    /// Create the requested file sinks and emit a CSV header when needed.
    fn try_new(artifacts: &PachnerStressArtifacts) -> Result<Self, PachnerStressError> {
        let progress = artifacts
            .progress_csv
            .as_ref()
            .map(|path| {
                Ok::<_, PachnerStressError>(ProgressArtifact {
                    path: path.clone(),
                    writer: create_progress_writer(path)?,
                })
            })
            .transpose()?;
        Ok(Self {
            stdout: artifacts.stdout,
            progress,
        })
    }

    /// Emit initial triangulation metadata.
    fn emit_source(&self, source: PachnerStressSource) -> Result<(), PachnerStressError> {
        if self.stdout {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            writeln!(
                handle,
                "pachner_stress_source dimension={} label={} mode={} validation_scope={} vertices={} simplices={} seed={}",
                source.dimension,
                source.label,
                source.mode,
                source.validation_scope,
                source.vertices,
                source.simplices,
                source.seed
            )
            .map_err(PachnerStressError::stdout)?;
            handle.flush().map_err(PachnerStressError::stdout)?;
        }
        Ok(())
    }

    /// Emit a coarse-grained stage checkpoint for long-running setup phases.
    fn emit_stage(
        &self,
        config: PachnerStressConfig,
        stage: &'static str,
        vertices: Option<usize>,
        simplices: Option<usize>,
    ) -> Result<(), PachnerStressError> {
        if self.stdout {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            write!(
                handle,
                "pachner_stress_stage dimension={} label={} mode={} validation_scope={} stage={stage}",
                config.dimension.value(),
                config.label(),
                config.mode.label(),
                PACHNER_STRESS_VALIDATION_SCOPE_LABEL
            )
            .map_err(PachnerStressError::stdout)?;
            if let Some(vertices) = vertices {
                write!(handle, " vertices={vertices}").map_err(PachnerStressError::stdout)?;
            }
            if let Some(simplices) = simplices {
                write!(handle, " simplices={simplices}").map_err(PachnerStressError::stdout)?;
            }
            writeln!(handle).map_err(PachnerStressError::stdout)?;
            handle.flush().map_err(PachnerStressError::stdout)?;
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
                "pachner_stress_progress dimension={} label={} mode={} validation_scope={} sequence={} step={} attempts={} accepted={} \
                 rejected={} candidate_misses={} proposal_rejections={} validations={} \
                 validation_nanos={} acceptance_rate={:.6} vertices={} simplices={}",
                config.dimension.value(),
                config.label(),
                config.mode.label(),
                PACHNER_STRESS_VALIDATION_SCOPE_LABEL,
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
                progress.simplices
            )
            .map_err(PachnerStressError::stdout)?;
            handle.flush().map_err(PachnerStressError::stdout)?;
        }
        if let Some(progress_artifact) = &mut self.progress {
            writeln!(
                progress_artifact.writer,
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{:.6},{},{}",
                config.dimension.value(),
                config.label(),
                config.mode.label(),
                PACHNER_STRESS_VALIDATION_SCOPE_LABEL,
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
                progress.simplices
            )
            .map_err(|source| PachnerStressError::ArtifactWrite {
                path: progress_artifact.path.as_path().to_owned(),
                source,
            })?;
            progress_artifact
                .writer
                .flush()
                .map_err(|source| progress_artifact.path.flush_error(source))?;
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
                "pachner_stress_metric dimension={} label={} mode={} validation_scope={} sequence={} attempts={} accepted={} rejected={} \
                 candidate_misses={} proposal_rejections={} validations={} validation_nanos={} \
                 elapsed_nanos={} attempts_per_second={} final_vertices={} final_simplices={}",
                config.dimension.value(),
                config.label(),
                config.mode.label(),
                PACHNER_STRESS_VALIDATION_SCOPE_LABEL,
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
                report.final_simplices
            )
            .map_err(PachnerStressError::stdout)?;
            handle.flush().map_err(PachnerStressError::stdout)?;
        }
        Ok(())
    }

    /// Flush any open file sinks.
    fn finish(&mut self) -> Result<(), PachnerStressError> {
        if let Some(progress_artifact) = &mut self.progress {
            progress_artifact
                .writer
                .flush()
                .map_err(|source| progress_artifact.path.flush_error(source))?;
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

/// Dispatch one exact diagnostic workload by runtime dimension.
fn run_pachner_stress(
    config: PachnerStressConfig,
    artifacts: &PachnerStressArtifacts,
) -> Result<PachnerStressSummary, PachnerStressError> {
    match config.dimension {
        PachnerStressDimension::Three => run_pachner_stress_dimension::<3>(config, artifacts),
        PachnerStressDimension::Four => run_pachner_stress_dimension::<4>(config, artifacts),
    }
}

/// Run one exact workload and write requested artifacts.
fn run_pachner_stress_dimension<const D: usize>(
    config: PachnerStressConfig,
    artifacts: &PachnerStressArtifacts,
) -> Result<PachnerStressSummary, PachnerStressError> {
    let mut reporter = PachnerStressReporter::try_new(artifacts)?;
    let mut tri = build_pachner_stress_dt::<D>(config, &reporter)?;
    let source = PachnerStressSource {
        dimension: D,
        label: config.label(),
        mode: config.mode.label(),
        validation_scope: PACHNER_STRESS_VALIDATION_SCOPE_LABEL,
        vertices: tri.number_of_vertices(),
        simplices: tri.number_of_simplices(),
        seed: config.seed,
    };
    reporter.emit_source(source)?;

    let start = Instant::now();
    let counters = match config.mode {
        PachnerStressMode::RoundTrip => {
            run_pachner_round_trip_sequence(&mut tri, config, 1, Some(&mut reporter))?
        }
        PachnerStressMode::RandomWalk => {
            run_pachner_random_walk_sequence(&mut tri, config, 1, Some(&mut reporter))?
        }
    };
    let elapsed = start.elapsed();
    let attempts = u128::try_from(config.move_attempts().get())?;
    let report = PachnerStressReport {
        sequence: 1,
        attempts: config.move_attempts().get(),
        accepted: counters.accepted,
        rejected: counters.rejected(),
        candidate_misses: counters.candidate_misses,
        proposal_rejections: counters.proposal_rejections,
        validations: counters.validations,
        validation_nanos: counters.validation_nanos,
        elapsed_nanos: elapsed.as_nanos(),
        attempts_per_second: attempts.saturating_mul(1_000_000_000) / elapsed.as_nanos().max(1),
        final_vertices: tri.number_of_vertices(),
        final_simplices: tri.number_of_simplices(),
    };
    reporter.emit_report(config, report)?;
    reporter.finish()?;

    let summary = PachnerStressSummary {
        schema: PACHNER_STRESS_EXPORT_SCHEMA,
        schema_version: PACHNER_STRESS_EXPORT_SCHEMA_VERSION,
        dimension: D,
        label: config.label(),
        mode: config.mode.label(),
        validation_scope: PACHNER_STRESS_VALIDATION_SCOPE_LABEL,
        configured_vertices: config.vertex_count.get(),
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

/// Build the initial randomized triangulation for one stress workload.
fn build_pachner_stress_dt<const D: usize>(
    config: PachnerStressConfig,
    reporter: &PachnerStressReporter,
) -> Result<PachnerStressTriangulation<D>, PachnerStressError> {
    reporter.emit_stage(
        config,
        "generate_points_start",
        Some(config.vertex_count.get()),
        None,
    )?;
    let points = generate_random_points_in_range_seeded::<D>(
        config.vertex_count.get(),
        stress_bounds()?,
        config.seed,
    )?;
    reporter.emit_stage(config, "convert_vertices_start", Some(points.len()), None)?;
    let vertices = try_vertices_from_points(&points)?;
    let options = ConstructionOptions::default().with_retry_policy(RetryPolicy::Shuffled {
        attempts: config.retry_attempts(),
        base_seed: Some(config.seed ^ 0xC0DE_0253_C0DE_0253),
    });

    reporter.emit_stage(config, "construction_start", Some(vertices.len()), None)?;
    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .topology_guarantee(TopologyGuarantee::PLManifold)
        .construction_options(options)
        .build_with_kernel(&RobustKernel::new())?;
    let tri = dt.into_triangulation();
    reporter.emit_stage(
        config,
        "initial_topology_validation_start",
        Some(tri.number_of_vertices()),
        Some(tri.number_of_simplices()),
    )?;
    validate_stress_topology_state(&tri, || {
        format!(
            "initial Pachner stress state dimension={D} label={} mode={} seed={}",
            config.label(),
            config.mode.label(),
            config.seed
        )
    })?;
    reporter.emit_stage(
        config,
        "initial_topology_validation_done",
        Some(tri.number_of_vertices()),
        Some(tri.number_of_simplices()),
    )?;
    Ok(tri)
}

/// Execute forward/inverse Pachner pairs and validate periodically.
fn run_pachner_round_trip_sequence<const D: usize>(
    dt: &mut PachnerStressTriangulation<D>,
    config: PachnerStressConfig,
    sequence: usize,
    mut reporter: Option<&mut PachnerStressReporter>,
) -> Result<PachnerStressCounters, PachnerStressError> {
    let mut rng = StdRng::seed_from_u64(config.seed ^ 0x0253_0253_5252_2525);
    let mut sampler = MoveSampler::try_from_triangulation(dt)?;
    let mut counters = PachnerStressCounters::default();

    for step in 1..=config.move_attempts().get() {
        if step > 1 && step.is_multiple_of(config.key_refresh_every().get()) {
            sampler.refresh(dt)?;
        }
        let Some(request) = random_round_trip_move(dt, &sampler, &mut rng) else {
            counters.candidate_misses = counters.candidate_misses.saturating_add(1);
            maybe_validate_stress_step(dt, config, step, &mut counters, &mut reporter, sequence)?;
            continue;
        };

        match dt.propose_pachner(request) {
            Ok(proposal) => {
                let Ok(forward) = proposal.attempt_on(dt) else {
                    counters.proposal_rejections = counters.proposal_rejections.saturating_add(1);
                    maybe_validate_stress_step(
                        dt,
                        config,
                        step,
                        &mut counters,
                        &mut reporter,
                        sequence,
                    )?;
                    continue;
                };
                counters.accepted = counters.accepted.saturating_add(1);
                let inverse = inverse_move_from_forward_result(dt, &forward)?;
                let _inverse_result = dt.propose_pachner(inverse)?.attempt_on(dt)?;
                counters.accepted = counters.accepted.saturating_add(1);
            }
            Err(_) => {
                counters.proposal_rejections = counters.proposal_rejections.saturating_add(1);
            }
        }

        maybe_validate_stress_step(dt, config, step, &mut counters, &mut reporter, sequence)?;
    }
    validate_final_stress_step(dt, config, &mut counters, &mut reporter, sequence)?;
    Ok(counters)
}

/// Execute valid random Pachner moves over an evolving triangulation.
fn run_pachner_random_walk_sequence<const D: usize>(
    dt: &mut PachnerStressTriangulation<D>,
    config: PachnerStressConfig,
    sequence: usize,
    mut reporter: Option<&mut PachnerStressReporter>,
) -> Result<PachnerStressCounters, PachnerStressError> {
    let mut rng = StdRng::seed_from_u64(config.seed ^ 0x0253_0253_0253_0253);
    let mut sampler = MoveSampler::try_from_triangulation(dt)?;
    let mut counters = PachnerStressCounters::default();

    for step in 1..=config.move_attempts().get() {
        if step > 1 && step.is_multiple_of(config.key_refresh_every().get()) {
            sampler.refresh(dt)?;
        }
        let Some(request) = random_pachner_move(dt, &sampler, &mut rng, config) else {
            counters.candidate_misses = counters.candidate_misses.saturating_add(1);
            maybe_validate_stress_step(dt, config, step, &mut counters, &mut reporter, sequence)?;
            continue;
        };

        match dt.propose_pachner(request) {
            Ok(proposal) => match proposal.attempt_on(dt) {
                Ok(_) => {
                    counters.accepted = counters.accepted.saturating_add(1);
                }
                Err(_) => {
                    counters.proposal_rejections = counters.proposal_rejections.saturating_add(1);
                }
            },
            Err(_) => {
                counters.proposal_rejections = counters.proposal_rejections.saturating_add(1);
            }
        }

        maybe_validate_stress_step(dt, config, step, &mut counters, &mut reporter, sequence)?;
    }
    validate_final_stress_step(dt, config, &mut counters, &mut reporter, sequence)?;
    Ok(counters)
}

/// Validate when a configured cadence boundary is reached.
fn maybe_validate_stress_step<const D: usize>(
    dt: &PachnerStressTriangulation<D>,
    config: PachnerStressConfig,
    step: usize,
    counters: &mut PachnerStressCounters,
    reporter: &mut Option<&mut PachnerStressReporter>,
    sequence: usize,
) -> Result<(), PachnerStressError> {
    if step.is_multiple_of(config.validate_every().get()) {
        validate_stress_step(dt, config, step, counters, reporter, sequence)?;
    }
    Ok(())
}

/// Ensure the final state is checked even when the cadence did not divide attempts.
fn validate_final_stress_step<const D: usize>(
    dt: &PachnerStressTriangulation<D>,
    config: PachnerStressConfig,
    counters: &mut PachnerStressCounters,
    reporter: &mut Option<&mut PachnerStressReporter>,
    sequence: usize,
) -> Result<(), PachnerStressError> {
    let final_step = config.move_attempts().get();
    if !final_step.is_multiple_of(config.validate_every().get()) {
        validate_stress_step(dt, config, final_step, counters, reporter, sequence)?;
    }
    Ok(())
}

/// Validate one cadence boundary and emit progress.
fn validate_stress_step<const D: usize>(
    dt: &PachnerStressTriangulation<D>,
    config: PachnerStressConfig,
    step: usize,
    counters: &mut PachnerStressCounters,
    reporter: &mut Option<&mut PachnerStressReporter>,
    sequence: usize,
) -> Result<(), PachnerStressError> {
    let validation_start = Instant::now();
    validate_stress_topology_state(dt, || stress_validation_context(config, step, *counters))?;
    counters.validation_nanos = counters
        .validation_nanos
        .saturating_add(validation_start.elapsed().as_nanos());
    counters.validations = counters.validations.saturating_add(1);
    if let Some(reporter) = reporter.as_mut() {
        reporter.emit_progress(
            config,
            PachnerStressProgress {
                sequence,
                step,
                attempts: config.move_attempts().get(),
                accepted: counters.accepted,
                rejected: counters.rejected(),
                candidate_misses: counters.candidate_misses,
                proposal_rejections: counters.proposal_rejections,
                validations: counters.validations,
                validation_nanos: counters.validation_nanos,
                acceptance_rate: stress_acceptance_rate(config, *counters)?,
                vertices: dt.number_of_vertices(),
                simplices: dt.number_of_simplices(),
            },
        )?;
    }
    Ok(())
}

/// Return the coordinate range used for stress point clouds.
fn stress_bounds() -> Result<CoordinateRange<f64>, CoordinateRangeError<f64>> {
    CoordinateRange::try_new(0.0_f64, 1.0)
}

/// Validate the intrinsic topology invariants Pachner moves are expected to preserve.
fn validate_stress_topology_state<const D: usize>(
    dt: &PachnerStressTriangulation<D>,
    context: impl FnOnce() -> String,
) -> Result<(), PachnerStressError> {
    if let Err(source) = dt.validate() {
        return Err(PachnerStressError::TopologyValidation {
            context: context(),
            source: Box::new(source),
        });
    }
    Ok(())
}

/// Convert bounded diagnostic counters into f64 telemetry values.
fn trace_value(value: usize) -> Result<f64, TryFromIntError> {
    u32::try_from(value).map(f64::from)
}

/// Compute accepted mutation ratio from flat stress counters.
fn stress_acceptance_rate(
    config: PachnerStressConfig,
    counters: PachnerStressCounters,
) -> Result<f64, PachnerStressError> {
    let expected = config
        .move_attempts()
        .get()
        .saturating_mul(config.mode.expected_moves_per_step());
    let total = trace_value(expected)?;
    if total == 0.0 {
        Ok(0.0)
    } else {
        Ok(trace_value(counters.accepted)? / total)
    }
}

/// Build a compact diagnostic validation context from flat stress counters.
fn stress_validation_context(
    config: PachnerStressConfig,
    step: usize,
    counters: PachnerStressCounters,
) -> String {
    format!(
        "Pachner stress validation label={} mode={} step={} attempts={} accepted={} \
         rejected={} candidate_misses={} proposal_rejections={}",
        config.label(),
        config.mode.label(),
        step,
        config.move_attempts().get(),
        counters.accepted,
        counters.rejected(),
        counters.candidate_misses,
        counters.proposal_rejections
    )
}

/// Choose a forward move whose inverse can be derived from its result.
fn random_round_trip_move<const D: usize>(
    dt: &PachnerStressTriangulation<D>,
    sampler: &MoveSampler,
    rng: &mut (impl Rng + ?Sized),
) -> Option<PachnerMove<(), D>> {
    let move_kind_count = if D >= 4 { 3 } else { 2 };
    match rng.random_range(0..move_kind_count) {
        0 => random_k1_insert(dt, sampler, rng),
        1 => sampler
            .random_facet(rng)
            .map(|facet| PachnerMove::K2 { facet }),
        2 => sampler
            .random_ridge(rng)
            .map(|ridge| PachnerMove::K3 { ridge }),
        _ => None,
    }
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

/// Choose a random simplex and insert a vertex at its centroid.
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

/// Build the inverse request from the inserted face reported by a forward move.
fn inverse_move_from_forward_result<const D: usize>(
    dt: &PachnerStressTriangulation<D>,
    result: &PachnerMoveResult<D>,
) -> Result<PachnerMove<(), D>, PachnerStressError> {
    match result.inserted_face_vertices.as_slice() {
        [vertex_key] => Ok(PachnerMove::K1Remove {
            vertex_key: *vertex_key,
        }),
        vertices @ [_, _] => Ok(PachnerMove::K2Inverse {
            edge: inserted_edge(dt, vertices)?,
        }),
        [a, b, c] => Ok(PachnerMove::K3Inverse {
            triangle: TriangleHandle::try_new(*a, *b, *c)?,
        }),
        vertices => Err(PachnerStressError::InsertedFaceArity {
            context: PachnerStressInsertedFaceContext::ForwardMove,
            expected: PachnerStressInsertedFaceArity::InvertibleForwardMove,
            actual: vertices.len(),
        }),
    }
}

/// Convert a reported inserted edge face into the live edge handle expected by k=2 inverse.
fn inserted_edge<const D: usize>(
    dt: &PachnerStressTriangulation<D>,
    vertices: &[VertexKey],
) -> Result<EdgeKey, PachnerStressError> {
    let [a, b] = vertices else {
        return Err(PachnerStressError::InsertedFaceArity {
            context: PachnerStressInsertedFaceContext::K2Move,
            expected: PachnerStressInsertedFaceArity::Edge,
            actual: vertices.len(),
        });
    };
    dt.edges()
        .find(|edge| {
            let (first, second) = edge.endpoints();
            (first == *a && second == *b) || (first == *b && second == *a)
        })
        .ok_or(PachnerStressError::InsertedEdgeMissing {
            left: *a,
            right: *b,
        })
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

/// Open a progress CSV and write its stable header.
fn create_progress_writer(path: &ArtifactPath) -> Result<BufWriter<File>, PachnerStressError> {
    let mut writer = path.create_writer()?;
    writeln!(
        writer,
        "dimension,label,mode,validation_scope,sequence,step,attempts,accepted,rejected,candidate_misses,\
         proposal_rejections,validations,validation_nanos,acceptance_rate,vertices,simplices"
    )
    .map_err(|source| PachnerStressError::ArtifactWrite {
        path: path.as_path().to_owned(),
        source,
    })?;
    writer.flush().map_err(|source| path.flush_error(source))?;
    Ok(writer)
}

/// Write the stable run-level summary JSON artifact.
fn write_summary_json(
    path: &ArtifactPath,
    summary: &PachnerStressSummary,
) -> Result<(), PachnerStressError> {
    write_json_output(summary, Some(path))?;
    Ok(())
}

/// Errors surfaced by the Pachner stress diagnostic runner.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
enum PachnerStressError {
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

    /// Artifact validation or file output failed.
    #[error(transparent)]
    Artifact {
        /// Typed artifact destination or output error.
        #[from]
        source: ArtifactOutputError,
    },

    /// The process working directory could not be read for path comparison.
    #[error("failed to resolve the working directory for an artifact path: {source}")]
    ArtifactWorkingDirectory {
        /// Underlying operating-system error.
        #[source]
        source: io::Error,
    },

    /// A diagnostic artifact path component could not be inspected.
    #[error("failed to inspect artifact path {path:?}: {source}")]
    ArtifactInspect {
        /// Path being inspected.
        path: PathBuf,
        /// Underlying operating-system error.
        #[source]
        source: io::Error,
    },

    /// No existing ancestor could anchor an artifact identity.
    #[error("artifact path has no resolvable existing ancestor: {path:?}")]
    ArtifactUnresolvablePath {
        /// Rejected path.
        path: PathBuf,
    },

    /// An existing path prefix could not be canonicalized.
    #[error("failed to resolve artifact path identity for {path:?}: {source}")]
    ArtifactResolveIdentity {
        /// Existing prefix being canonicalized.
        path: PathBuf,
        /// Underlying operating-system error.
        #[source]
        source: io::Error,
    },

    /// Two existing diagnostic destinations could not be compared.
    #[error("failed to compare artifact paths {first:?} and {second:?}: {source}")]
    ArtifactCompareIdentity {
        /// First output path.
        first: PathBuf,
        /// Second output path.
        second: PathBuf,
        /// Underlying operating-system error.
        #[source]
        source: io::Error,
    },

    /// Streaming a diagnostic artifact failed.
    #[error("failed to write artifact {path:?}: {source}")]
    ArtifactWrite {
        /// Requested artifact path.
        path: PathBuf,
        /// Underlying operating-system error.
        #[source]
        source: io::Error,
    },

    /// Writing machine-readable telemetry to stdout failed.
    #[error("failed to write Pachner stress telemetry to stdout: {source}")]
    Stdout {
        /// Underlying standard-output error.
        #[source]
        source: io::Error,
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
        /// Diagnostic workload context.
        context: String,
        /// Underlying invariant error.
        #[source]
        source: Box<InvariantError>,
    },

    /// Public facet query failed.
    #[error("Pachner stress facet query failed: {source}")]
    Facet {
        /// Underlying facet query error.
        #[from]
        source: FacetError,
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

    /// A forward move reported an inserted face arity this stress mode cannot invert.
    #[error("{context} inserted face should have {expected} vertices, got {actual}")]
    InsertedFaceArity {
        /// Move whose inserted-face result could not be interpreted.
        context: PachnerStressInsertedFaceContext,
        /// Expected inserted-face arity.
        expected: PachnerStressInsertedFaceArity,
        /// Actual inserted-face arity.
        actual: usize,
    },

    /// A k=2 forward move reported an inserted edge that is not live afterward.
    #[error("inserted k=2 edge {left:?}-{right:?} is missing")]
    InsertedEdgeMissing {
        /// First reported endpoint.
        left: VertexKey,
        /// Second reported endpoint.
        right: VertexKey,
    },

    /// Triangle handle construction failed for an inverse k=3 move.
    #[error("failed to build inverse k=3 triangle handle: {source}")]
    TriangleHandle {
        /// Underlying triangle-handle error.
        #[from]
        source: TriangleHandleError,
    },

    /// Two requested artifacts target the same file identity.
    #[error(
        "progress CSV and summary JSON must use different paths; got {progress_path:?} and {summary_path:?}"
    )]
    DuplicateArtifactPath {
        /// Requested progress CSV path.
        progress_path: PathBuf,
        /// Requested summary JSON path.
        summary_path: PathBuf,
    },
}

impl PachnerStressError {
    /// Attach the stdout destination to telemetry I/O failures.
    const fn stdout(source: io::Error) -> Self {
        Self::Stdout { source }
    }
}

/// Convert a raw positive count into `NonZeroUsize`.
fn positive_nonzero(
    argument: PachnerStressCountArgument,
    value: usize,
) -> Result<NonZeroUsize, PachnerStressError> {
    NonZeroUsize::new(value).ok_or(PachnerStressError::NonPositive { argument, value })
}

/// Parses a raw count once while preserving the caller's typed threshold error.
///
/// CLI commands use this so zero and below-minimum counts report the same
/// domain-specific "too few vertices" diagnostic instead of leaking
/// `NonZeroUsize` parsing as a separate user-facing error class.
fn validated_nonzero_count<E>(
    value: usize,
    is_valid: impl FnOnce(NonZeroUsize) -> bool,
    error: impl FnOnce() -> E,
) -> Result<NonZeroUsize, E> {
    NonZeroUsize::new(value)
        .filter(|count| is_valid(*count))
        .ok_or_else(error)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::os::unix::fs::symlink;
    use std::{
        fs,
        path::Path,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn target_artifact_path(label: &str, extension: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after UNIX epoch")
            .as_nanos();
        PathBuf::from("target")
            .join("pachner-stress-tests")
            .join(format!("{label}-{stamp}.{extension}"))
    }

    #[test]
    fn positive_count_errors_preserve_typed_argument() {
        let error = positive_nonzero(PachnerStressCountArgument::ValidateEvery, 0)
            .expect_err("zero should fail positive-count validation");

        let PachnerStressError::NonPositive { argument, value } = error else {
            panic!("expected NonPositive error, got {error:?}");
        };
        assert_eq!(argument, PachnerStressCountArgument::ValidateEvery);
        assert_eq!(argument.to_string(), "--validate-every");
        assert_eq!(value, 0);
    }

    #[test]
    fn zero_vertices_preserve_typed_too_few_vertices_error() {
        let error = PachnerStressConfig::try_new(PachnerStressConfigInput {
            mode: PachnerStressMode::RoundTrip,
            dimension: PachnerStressDimension::Three,
            vertex_count: 0,
            move_attempts: NonZeroUsize::new(2).expect("literal is nonzero"),
            validate_every: NonZeroUsize::new(1).expect("literal is nonzero"),
            key_refresh_every: NonZeroUsize::new(7).expect("literal is nonzero"),
            retry_attempts: NonZeroUsize::new(4).expect("literal is nonzero"),
            seed: 42,
        })
        .expect_err("zero vertices should fail Pachner stress validation");

        let PachnerStressError::TooFewVertices {
            dimension,
            vertices,
            minimum,
        } = error
        else {
            panic!("expected TooFewVertices error, got {error:?}");
        };
        assert_eq!(dimension, 3);
        assert_eq!(vertices, 0);
        assert_eq!(minimum, 4);
    }

    #[test]
    fn inserted_face_arity_errors_preserve_typed_context() {
        let error = PachnerStressError::InsertedFaceArity {
            context: PachnerStressInsertedFaceContext::ForwardMove,
            expected: PachnerStressInsertedFaceArity::InvertibleForwardMove,
            actual: 4,
        };

        let PachnerStressError::InsertedFaceArity {
            context,
            expected,
            actual,
        } = error
        else {
            panic!("expected InsertedFaceArity error, got {error:?}");
        };
        assert_eq!(context, PachnerStressInsertedFaceContext::ForwardMove);
        assert_eq!(
            expected,
            PachnerStressInsertedFaceArity::InvertibleForwardMove
        );
        assert_eq!(context.to_string(), "forward Pachner move");
        assert_eq!(expected.to_string(), "1, 2, or 3");
        assert_eq!(actual, 4);
    }

    #[test]
    fn artifacts_reject_duplicate_paths_before_storage() {
        let path = PathBuf::from("target/notebooks/pachner/shared.csv");
        let error = PachnerStressArtifacts::try_new(Some(path.clone()), Some(path), true)
            .expect_err("duplicate artifact paths should fail validation");

        let PachnerStressError::DuplicateArtifactPath {
            progress_path,
            summary_path,
        } = error
        else {
            panic!("expected DuplicateArtifactPath error, got {error:?}");
        };
        assert_eq!(
            progress_path,
            Path::new("target/notebooks/pachner/shared.csv")
        );
        assert_eq!(
            summary_path,
            Path::new("target/notebooks/pachner/shared.csv")
        );
    }

    #[test]
    fn lexical_aliases_share_one_diagnostic_artifact_identity() {
        let directory = target_artifact_path("lexical-alias", "dir");
        let direct = ArtifactPath::try_new(directory.join("summary.json"))
            .expect("direct output path should validate");
        let alias = ArtifactPath::try_new(directory.join("nested/../summary.json"))
            .expect("aliased output path should validate");

        assert!(
            artifact_paths_conflict(&direct, &alias)
                .expect("diagnostic artifact identities should compare")
        );
    }

    #[test]
    fn existing_hard_link_aliases_share_one_diagnostic_artifact_identity() {
        let directory = target_artifact_path("hard-link-alias", "dir");
        fs::create_dir_all(&directory).expect("scratch directory should be created");
        let original = directory.join("original.json");
        let alias = directory.join("alias.json");
        fs::write(&original, b"fixture").expect("scratch artifact should be written");
        fs::hard_link(&original, &alias).expect("scratch hard link should be created");

        let original = ArtifactPath::try_new(original).expect("original path should validate");
        let alias = ArtifactPath::try_new(alias).expect("hard-link path should validate");
        assert!(
            artifact_paths_conflict(&original, &alias)
                .expect("existing diagnostic artifact identities should compare")
        );

        fs::remove_dir_all(directory).expect("scratch hard-link fixture should be removed");
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_prefixes_share_one_missing_diagnostic_artifact_identity() {
        let directory = target_artifact_path("symlink-prefix-alias", "dir");
        let real_directory = directory.join("real");
        let alias_directory = directory.join("alias");
        fs::create_dir_all(&real_directory).expect("real scratch directory should be created");
        symlink(Path::new("real"), &alias_directory)
            .expect("scratch directory symlink should be created");

        let direct = ArtifactPath::try_new(real_directory.join("summary.json"))
            .expect("direct missing-file path should validate");
        let alias = ArtifactPath::try_new(alias_directory.join("summary.json"))
            .expect("symlinked missing-file path should validate");
        assert!(
            artifact_paths_conflict(&direct, &alias)
                .expect("symlinked diagnostic artifact identities should compare")
        );

        fs::remove_dir_all(directory).expect("scratch symlink fixture should be removed");
    }

    #[test]
    fn progress_writer_flushes_header_on_create() {
        let path = target_artifact_path("progress-header", "csv");
        let artifact_path =
            ArtifactPath::try_new(path.clone()).expect("progress path should validate");
        let writer = create_progress_writer(&artifact_path)
            .expect("progress writer should create parent directories and header");
        let visible_len = fs::metadata(&path)
            .expect("progress CSV should exist while writer is alive")
            .len();
        assert!(
            visible_len > 0,
            "progress CSV header should be visible before the run finishes"
        );
        drop(writer);

        let header = fs::read_to_string(&path).expect("progress CSV header should be readable");
        assert_eq!(
            header,
            "dimension,label,mode,validation_scope,sequence,step,attempts,accepted,rejected,candidate_misses,\
             proposal_rejections,validations,validation_nanos,acceptance_rate,vertices,simplices\n"
        );
        fs::remove_file(path).expect("progress CSV fixture should be removed");
    }

    #[test]
    fn config_clamps_validate_every_to_move_attempts() {
        let config = PachnerStressConfig::try_new(PachnerStressConfigInput {
            mode: PachnerStressMode::RoundTrip,
            dimension: PachnerStressDimension::Three,
            vertex_count: 5,
            move_attempts: NonZeroUsize::new(2).expect("literal is nonzero"),
            validate_every: NonZeroUsize::new(100).expect("literal is nonzero"),
            key_refresh_every: NonZeroUsize::new(7).expect("literal is nonzero"),
            retry_attempts: NonZeroUsize::new(4).expect("literal is nonzero"),
            seed: 42,
        })
        .expect("valid 3D Pachner stress config should build");

        assert_eq!(config.vertex_count.get(), 5);
        assert_eq!(config.move_attempts().get(), 2);
        assert_eq!(config.validate_every().get(), 2);
        assert_eq!(config.key_refresh_every().get(), 7);
        assert_eq!(config.retry_attempts().get(), 4);
    }
}
