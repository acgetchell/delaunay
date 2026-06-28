//! Bistellar flip operations for triangulations.
//!
//! This module implements **Pachner (bistellar) moves** for Delaunay repair and topology editing.
//!
//! We use the **standard convention** where a k-move (1 ≤ k ≤ D+1) replaces the star of a
//! `(D+1−k)`-simplex with the star of the complementary `(k−1)`-simplex:
//! - removed D-simplices = k
//! - inserted D-simplices = D+2−k
//! - removed-face dimension = D+1−k
//! - inserted-face dimension = k−1
//!
//! Implemented moves:
//! - k=1: simplex split/merge (1↔(D+1)), valid for D≥1
//! - k=2: facet flips (2↔D), valid for D≥2
//! - k=3: ridge flips (3↔(D−1)), valid for D≥3
//!
//! We represent higher-order flips via **inverse** directions:
//! - In D=4, inverse k=2 is the k=4 flip (4↔2).
//! - In D=5, inverse k=2 is k=5 (5↔2) and inverse k=3 is k=4 (4↔3).
//!
//! Delaunay repair uses k=2 (facets) and k=3 (ridges) only; k=1 is exposed for
//! topological editing via the public API.
//!
//! # References
//! - Edelsbrunner & Shah (1996) - "Incremental Topological Flipping Works for Regular Triangulations"
//! - Bistellar flips implementation notebook (Warp Drive)

#![forbid(unsafe_code)]

use crate::core::algorithms::incremental_insertion::{
    CavityFillingError, HullExtensionReason, InsertionError, InsertionErrorKind,
    InsertionTopologyValidationContext, NeighborWiringError, SpatialIndexConstructionFailure,
    TdsConstructionFailure, TdsValidationFailure, external_facets_for_boundary,
    wire_cavity_neighbors,
};
use crate::core::algorithms::locate::{ConflictError, LocateError, extract_cavity_boundary};
use crate::core::collections::{
    FastHashMap, FastHashSet, FastHasher, MAX_PRACTICAL_DIMENSION_SIZE, PeriodicOffsetBuffer,
    SimplexKeyBuffer, SmallBuffer,
};
use crate::core::edge::{EdgeKey, EdgeKeyError};
use crate::core::embedding::TriangulationEmbeddingValidationErrorKind;
use crate::core::facet::{AllFacetsIter, FacetError, FacetHandle, facet_key_from_vertices};
use crate::core::operations::TopologicalOperation;
use crate::core::simplex::{NeighborSlot, Simplex, SimplexValidationError};
use crate::core::tds::{
    EntityKind, NeighborValidationError, SimplexKey, Tds, TdsMutationError, TdsRollbackTransaction,
    VertexKey,
};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation::Triangulation;
use crate::core::util::stable_hash_u64_slice;
use crate::core::validation::{TopologyGuarantee, TriangulationValidationError};
use crate::core::vertex::Vertex;
use crate::geometry::kernel::Kernel;
use crate::geometry::point::Point;
use crate::geometry::predicates::{Orientation, simplex_orientation_fast_filter_sign};
use crate::geometry::robust_predicates::robust_orientation;
use crate::geometry::traits::coordinate::{
    CoordinateConversionError, CoordinateValidationError, CoordinateValues,
};
use crate::topology::traits::global_topology_model::{
    GlobalTopologyModel, GlobalTopologyModelAdapter,
};
use crate::topology::traits::topological_space::GlobalTopology;
use crate::validation::DelaunayTriangulationValidationError;
use slotmap::Key;
use std::borrow::Cow;
use std::collections::VecDeque;
use std::env;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};
use thiserror::Error;

type VertexKeyList = SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>;
type RemovedSimplexVertexSnapshot = SmallBuffer<VertexKeyList, MAX_PRACTICAL_DIMENSION_SIZE>;
type ReplacementPeriodicOffsets<const D: usize> =
    SmallBuffer<Option<PeriodicOffsetBuffer<D>>, MAX_PRACTICAL_DIMENSION_SIZE>;

/// Bistellar flip kind descriptor.
///
/// Access the move size with [`BistellarFlipKind::k`].
///
/// # Examples
///
/// ```rust
/// use delaunay::flips::BistellarFlipKind;
///
/// let kind = BistellarFlipKind::k2(3);
/// let inverse = kind.inverse();
/// assert_eq!(kind.k(), 2);
/// assert_eq!(inverse.k(), 3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BistellarFlipKind {
    /// Number of simplices being replaced on the current side (k).
    k: usize,
    /// Dimension of the triangulation (D).
    pub d: usize,
}
/// Run a single flip-repair attempt using k=2 (and k=3 in 3D+).
fn repair_delaunay_with_flips_k2_k3_attempt<K, U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    kernel: &K,
    seed_simplices: Option<&[SimplexKey]>,
    config: &RepairAttemptConfig,
) -> Result<RepairAttemptOutcome, DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    repair_delaunay_with_flips_k2_k3_attempt_timed(tds, kernel, seed_simplices, config, None)
}

/// Run a single flip-repair attempt while reporting queue-family timings.
#[expect(
    clippy::too_many_lines,
    reason = "Repair loop contains inline tracing and queue handling for diagnostics"
)]
fn repair_delaunay_with_flips_k2_k3_attempt_timed<K, U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    kernel: &K,
    seed_simplices: Option<&[SimplexKey]>,
    config: &RepairAttemptConfig,
    mut timing: Option<&mut LocalRepairPhaseTiming>,
) -> Result<RepairAttemptOutcome, DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    if D < 2 {
        return Err(FlipError::UnsupportedDimension { dimension: D }.into());
    }
    if D == 2 {
        return repair_delaunay_with_flips_k2_attempt(tds, kernel, seed_simplices, config);
    }

    let max_flips = config
        .max_flips_override
        .unwrap_or_else(|| default_max_flips::<D>(tds.number_of_simplices()));

    let mut stats = DelaunayRepairStats::default();
    let mut diagnostics = RepairDiagnostics::default();
    let mut queues = RepairQueues::new();
    let mut last_applied_flip: Option<LastAppliedFlip> = None;
    let seed_started = timing.is_some().then(Instant::now);
    let used_full_reseed = seed_repair_queues(tds, seed_simplices, &mut queues, &mut stats)?;
    if let (Some(timing), Some(seed_started)) = (timing.as_deref_mut(), seed_started) {
        timing.record_attempt_seed(seed_started.elapsed());
    }
    let mut touched_simplices = SimplexKeyBuffer::new();
    let mut touched_simplex_set = FastHashSet::<SimplexKey>::default();
    let mut flip_workspace = Tds::empty();

    let mut prefer_secondary = false;

    macro_rules! timed_step {
        ($recorder:ident, $step:expr) => {{
            if timing.is_some() {
                let started = Instant::now();
                let processed = $step?;
                if let Some(timing) = timing.as_deref_mut() {
                    timing.$recorder(started.elapsed());
                }
                processed
            } else {
                $step?
            }
        }};
    }

    while queues.has_work() {
        if prefer_secondary {
            let processed_ridge = timed_step!(
                record_attempt_ridge,
                run_next_ridge_repair_step(
                    tds,
                    &mut flip_workspace,
                    kernel,
                    &mut queues,
                    &mut stats,
                    max_flips,
                    config,
                    &mut diagnostics,
                    &mut last_applied_flip,
                    &mut touched_simplices,
                    &mut touched_simplex_set,
                )
            );
            let processed_edge = !processed_ridge
                && timed_step!(
                    record_attempt_edge,
                    run_next_edge_repair_step(
                        tds,
                        &mut flip_workspace,
                        kernel,
                        &mut queues,
                        &mut stats,
                        max_flips,
                        config,
                        &mut diagnostics,
                        &mut last_applied_flip,
                        &mut touched_simplices,
                        &mut touched_simplex_set,
                    )
                );
            let processed_triangle = !processed_ridge
                && !processed_edge
                && timed_step!(
                    record_attempt_triangle,
                    run_next_triangle_repair_step(
                        tds,
                        &mut flip_workspace,
                        kernel,
                        &mut queues,
                        &mut stats,
                        max_flips,
                        config,
                        &mut diagnostics,
                        &mut last_applied_flip,
                        &mut touched_simplices,
                        &mut touched_simplex_set,
                    )
                );
            if processed_ridge || processed_edge || processed_triangle {
                prefer_secondary = false;
                continue;
            }
        }

        if timed_step!(
            record_attempt_facet,
            run_next_facet_repair_step(
                tds,
                &mut flip_workspace,
                kernel,
                &mut queues,
                &mut stats,
                max_flips,
                config,
                &mut diagnostics,
                &mut last_applied_flip,
                &mut touched_simplices,
                &mut touched_simplex_set,
            )
        ) {
            prefer_secondary = true;
            continue;
        }

        let processed_ridge = timed_step!(
            record_attempt_ridge,
            run_next_ridge_repair_step(
                tds,
                &mut flip_workspace,
                kernel,
                &mut queues,
                &mut stats,
                max_flips,
                config,
                &mut diagnostics,
                &mut last_applied_flip,
                &mut touched_simplices,
                &mut touched_simplex_set,
            )
        );
        let processed_edge = !processed_ridge
            && timed_step!(
                record_attempt_edge,
                run_next_edge_repair_step(
                    tds,
                    &mut flip_workspace,
                    kernel,
                    &mut queues,
                    &mut stats,
                    max_flips,
                    config,
                    &mut diagnostics,
                    &mut last_applied_flip,
                    &mut touched_simplices,
                    &mut touched_simplex_set,
                )
            );
        let processed_triangle = !processed_ridge
            && !processed_edge
            && timed_step!(
                record_attempt_triangle,
                run_next_triangle_repair_step(
                    tds,
                    &mut flip_workspace,
                    kernel,
                    &mut queues,
                    &mut stats,
                    max_flips,
                    config,
                    &mut diagnostics,
                    &mut last_applied_flip,
                    &mut touched_simplices,
                    &mut touched_simplex_set,
                )
            );
        if processed_ridge || processed_edge || processed_triangle {
            prefer_secondary = false;
        }
    }
    if repair_trace_enabled() {
        tracing::debug!(
            "[repair] attempt={} done: checked={} flips={} max_queue={} ambiguous={} predicate_failures={} cycles={}",
            config.attempt,
            stats.facets_checked,
            stats.flips_performed,
            stats.max_queue_len,
            diagnostics.ambiguous_predicates,
            diagnostics.predicate_failures,
            diagnostics.cycle_detections,
        );
    }
    emit_repair_debug_summary("attempt_done", &stats, &diagnostics, config, max_flips);

    Ok(RepairAttemptOutcome {
        postcondition_required: repair_postcondition_required(&stats, &diagnostics),
        stats,
        last_applied_flip,
        touched_simplices,
        used_full_reseed,
    })
}

/// Captures each removed simplex's vertex list before a flip deletes the simplices.
///
/// The snapshot lets later diagnostics describe removed simplices even after
/// their `SimplexKey`s no longer resolve in the TDS.
fn snapshot_removed_simplex_vertices<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    removed_simplices: &SimplexKeyBuffer,
) -> Result<RemovedSimplexVertexSnapshot, FlipError>
where
    U: DataType,
    V: DataType,
{
    removed_simplices
        .iter()
        .copied()
        .map(|simplex_key| {
            let simplex = tds
                .simplex(simplex_key)
                .ok_or(FlipError::MissingSimplex { simplex_key })?;
            Ok(simplex.vertices().iter().copied().collect())
        })
        .collect()
}

/// Applies a bistellar flip using explicit k and vertex/simplex slices.
///
/// # Errors
///
/// Returns [`FlipError::DanglingVertexIncidence`] if the maintained incidence
/// index references a simplex that is no longer present, or another
/// [`FlipError`] when the move is invalid, geometrically degenerate,
/// non-manifold, or cannot be applied atomically.
#[expect(
    clippy::too_many_arguments,
    reason = "Flip mutation needs explicit move, cavity, policy, and validation inputs"
)]
fn apply_bistellar_flip_with_k<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    k_move: usize,
    removed_face_vertices: &[VertexKey],
    inserted_face_vertices: &[VertexKey],
    removed_simplices: &SimplexKeyBuffer,
    direction: FlipDirection,
    orientation_policy: ReplacementOrientationPolicy,
    validation_scope: FlipValidationScope,
) -> Result<AppliedFlip<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    apply_bistellar_flip_with_k_inner(
        tds,
        k_move,
        removed_face_vertices,
        inserted_face_vertices,
        removed_simplices,
        direction,
        orientation_policy,
        validation_scope,
        None,
    )
}

/// Applies a bistellar flip with caller-owned rollback scratch storage.
///
/// This preserves the same failure-atomic public flip contract as
/// [`apply_bistellar_flip_with_k`] while letting local repair loops reuse one
/// trial TDS allocation across many candidate flips.
#[expect(
    clippy::too_many_arguments,
    reason = "Flip mutation needs explicit move, cavity, policy, validation inputs, and scratch storage"
)]
fn apply_bistellar_flip_with_k_in_workspace<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    k_move: usize,
    removed_face_vertices: &[VertexKey],
    inserted_face_vertices: &[VertexKey],
    removed_simplices: &SimplexKeyBuffer,
    direction: FlipDirection,
    orientation_policy: ReplacementOrientationPolicy,
    validation_scope: FlipValidationScope,
    trial_workspace: &mut Tds<U, V, D>,
) -> Result<AppliedFlip<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    apply_bistellar_flip_with_k_inner(
        tds,
        k_move,
        removed_face_vertices,
        inserted_face_vertices,
        removed_simplices,
        direction,
        orientation_policy,
        validation_scope,
        Some(trial_workspace),
    )
}

/// Shared implementation for failure-atomic bistellar mutation.
///
/// The original TDS is mutated only after the trial TDS has been fully rewired
/// and locally validated. Passing `trial_workspace` lets hot repair paths reuse
/// rollback storage; leaving it `None` keeps the standalone API's independent
/// trial allocation behavior.
#[expect(
    clippy::too_many_arguments,
    reason = "Flip mutation needs explicit move, cavity, policy, validation inputs, and optional scratch storage"
)]
#[expect(
    clippy::too_many_lines,
    reason = "Keep flip construction, validation, and wiring together for clarity"
)]
fn apply_bistellar_flip_with_k_inner<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    k_move: usize,
    removed_face_vertices: &[VertexKey],
    inserted_face_vertices: &[VertexKey],
    removed_simplices: &SimplexKeyBuffer,
    direction: FlipDirection,
    orientation_policy: ReplacementOrientationPolicy,
    validation_scope: FlipValidationScope,
    trial_workspace: Option<&mut Tds<U, V, D>>,
) -> Result<AppliedFlip<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    if k_move == 0 || k_move > D + 1 {
        return Err(FlipContextError::InvalidMoveSize {
            k_move,
            dimension: D,
        }
        .into());
    }

    let expected_removed_face = D + 2 - k_move;
    if removed_face_vertices.len() != expected_removed_face {
        return Err(FlipContextError::WrongRemovedFaceArity {
            expected: expected_removed_face,
            found: removed_face_vertices.len(),
        }
        .into());
    }
    if inserted_face_vertices.len() != k_move {
        return Err(FlipContextError::WrongInsertedFaceArity {
            k_move,
            expected: k_move,
            found: inserted_face_vertices.len(),
        }
        .into());
    }
    if removed_simplices.len() != k_move {
        return Err(FlipContextError::WrongRemovedSimplexCount {
            expected: k_move,
            found: removed_simplices.len(),
        }
        .into());
    }
    if removed_face_vertices
        .iter()
        .any(|v| inserted_face_vertices.contains(v))
    {
        return Err(FlipContextError::OverlappingFaces.into());
    }
    #[cfg(debug_assertions)]
    {
        // Coherent orientation is a validation-scale invariant. Keep the typed
        // diagnostic in debug/test builds, but do not scan the whole TDS inside
        // every release-mode flip on construction/repair hot paths.
        if !tds.is_coherently_oriented() {
            return Err(FlipContextError::CoherentOrientationViolation {
                stage: FlipOrientationCheckStage::BeforeMutation,
                k_move,
                direction,
            }
            .into());
        }
    }

    // Bistellar move legality: the inserted simplex must not already exist in the complex.
    //
    // If it does, applying the move can create non-manifold codimension>1 singularities
    // (e.g., disconnected ridge links in 3D when a k=2 flip inserts an already-existing edge).
    //
    // For facets (k==D) and full simplices (k==D+1), this is already covered by the existing
    // non-manifold facet / duplicate-simplex checks.
    if k_move >= 2
        && k_move < D
        && let Some(existing_simplex) =
            find_simplex_containing_simplex(tds, inserted_face_vertices, removed_simplices)?
    {
        if repair_trace_enabled() || env::var_os("DELAUNAY_REPAIR_DEBUG_FACETS").is_some() {
            tracing::debug!(
                "[repair] skip flip: inserted simplex already exists (k={k_move}, inserted_face={inserted_face_vertices:?}, existing_simplex={existing_simplex:?})"
            );
        }
        return Err(FlipError::InsertedSimplexAlreadyExists {
            k_move,
            simplex_vertices: Box::new(inserted_face_vertices.iter().copied().collect()),
            existing_simplex,
        });
    }

    let mut new_simplex_vertices: SmallBuffer<
        SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
        MAX_PRACTICAL_DIMENSION_SIZE,
    > = SmallBuffer::with_capacity(removed_face_vertices.len());

    for &omit in removed_face_vertices {
        let mut vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(D + 1);
        vertices.extend_from_slice(inserted_face_vertices);
        for &v in removed_face_vertices {
            if v != omit {
                vertices.push(v);
            }
        }

        new_simplex_vertices.push(vertices);
    }

    let boundary_facets = extract_cavity_boundary(tds, removed_simplices).map_err(|source| {
        FlipError::from(FlipNeighborWiringError::BoundaryExtraction { source })
    })?;

    let external_facets = external_facets_for_boundary(tds, removed_simplices, &boundary_facets)
        .map_err(FlipNeighborWiringError::from)?;

    let topology_index = build_flip_topology_index(
        tds,
        &new_simplex_vertices,
        removed_simplices,
        inserted_face_vertices,
    );

    for vertices in &mut new_simplex_vertices {
        if flip_would_duplicate_simplex_any(tds, vertices, &topology_index) {
            return Err(FlipError::DuplicateSimplex);
        }
        if flip_would_create_nonmanifold_facets_any(vertices, &topology_index) {
            return Err(FlipError::NonManifoldFacet);
        }

        let points = vertices_to_points(tds, vertices)?;

        // Exact orientation: reject degenerate simplices and canonicalize to
        // positive orientation in one pass.  This function uses
        // robust_orientation (exact arithmetic, no SoS) rather than any
        // kernel predicate, so it is kernel-independent.
        match robust_orientation(&points) {
            Err(e) => {
                return Err(FlipPredicateError::coordinate_conversion(
                    FlipPredicateOperation::ReplacementSimplexOrientation,
                    e,
                )
                .into());
            }
            Ok(Orientation::DEGENERATE) => {
                if env::var_os("DELAUNAY_REPAIR_DEBUG_FACETS").is_some() {
                    tracing::debug!(
                        k_move,
                        direction = ?direction,
                        removed_face = ?removed_face_vertices,
                        inserted_face = ?inserted_face_vertices,
                        vertices = ?vertices,
                        "[repair] flip degenerate simplex (exact)"
                    );
                }
                return Err(FlipError::DegenerateSimplex);
            }
            Ok(Orientation::NEGATIVE) => {
                // Canonicalize to positive orientation by swapping two vertices.
                vertices.swap(0, 1);
            }
            Ok(Orientation::POSITIVE) => {}
        }
    }

    let newly_inserted_vertex = if k_move == 1 {
        inserted_face_vertices.first().copied()
    } else {
        None
    };
    let mut new_simplex_offsets = replacement_simplex_periodic_offsets(
        tds,
        &new_simplex_vertices,
        removed_simplices,
        &external_facets,
        newly_inserted_vertex,
    )?;

    orient_replacement_simplices(
        tds,
        &mut new_simplex_vertices,
        &mut new_simplex_offsets,
        &external_facets,
    )?;
    if matches!(
        orientation_policy,
        ReplacementOrientationPolicy::RequirePositive
    ) {
        validate_replacement_orientation(tds, &new_simplex_vertices)?;
    }

    // Snapshot the removed simplices' vertex lists before any TDS mutation so an
    // unexpected missing simplex aborts without leaving replacement simplices behind.
    // After `tds.remove_simplices_by_keys` runs, `tds.simplex(removed_key)` returns
    // `None`, which would strip the most useful context from predecessor-flip
    // traces (see #204 investigation).
    let removed_simplex_vertices = snapshot_removed_simplex_vertices(tds, removed_simplices)?;

    let apply_to_trial = |trial: &mut Tds<U, V, D>| -> Result<SimplexKeyBuffer, FlipError> {
        let mut new_simplices = SimplexKeyBuffer::new();

        for (vertices, periodic_offsets) in
            new_simplex_vertices.into_iter().zip(new_simplex_offsets)
        {
            let mut simplex = Simplex::try_new(vertices)?;
            if let Some(offsets) = periodic_offsets {
                simplex.set_periodic_vertex_offsets(offsets)?;
            }
            let simplex_key = trial
                .insert_simplex_with_mapping_prechecked_topology(simplex)
                .map_err(|source| FlipMutationError::SimplexInsertion {
                    source: source.into(),
                })?;
            new_simplices.push(simplex_key);
        }

        wire_cavity_neighbors(
            trial,
            &new_simplices,
            external_facets.iter().copied(),
            Some(removed_simplices),
        )
        .map_err(FlipNeighborWiringError::from)?;

        trial
            .remove_simplices_by_keys(removed_simplices)
            .map_err(|source| FlipError::from(FlipMutationError::SimplexRemoval { source }))?;

        let validation_result = match validation_scope {
            FlipValidationScope::FullTds => trial.is_valid().map_err(TdsValidationFailure::from),
            FlipValidationScope::LocalCavity => validate_flip_trial_cavity(
                trial,
                &new_simplices,
                &external_facets,
                removed_simplices,
            ),
        };
        validation_result.map_err(|source| {
            FlipError::from(FlipMutationError::TrialValidation {
                k_move,
                direction,
                source,
            })
        })?;

        #[cfg(debug_assertions)]
        {
            // This is intentionally debug/test-only for the same reason as the
            // pre-flip scan above: production validation already checks coherent
            // orientation at explicit validation boundaries.
            if !trial.is_coherently_oriented() {
                return Err(FlipError::from(
                    FlipMutationError::CoherentOrientationViolation {
                        stage: FlipOrientationCheckStage::AfterTrialMutation,
                        k_move,
                        direction,
                    },
                ));
            }
        }

        Ok(new_simplices)
    };

    let new_simplices = if let Some(trial) = trial_workspace {
        trial.clone_from_for_rollback(tds);
        let new_simplices = apply_to_trial(trial)?;
        std::mem::swap(tds, trial);
        new_simplices
    } else {
        let mut trial = tds.clone_for_rollback();
        let new_simplices = apply_to_trial(&mut trial)?;
        *tds = trial;
        new_simplices
    };

    Ok(AppliedFlip {
        info: FlipInfo {
            kind: BistellarFlipKind { k: k_move, d: D },
            direction,
            removed_simplices: removed_simplices.iter().copied().collect(),
            new_simplices,
            removed_face_vertices: removed_face_vertices.iter().copied().collect(),
            inserted_face_vertices: inserted_face_vertices.iter().copied().collect(),
        },
        removed_simplex_vertices,
    })
}

/// Selects whether a flip is only topological or must preserve Delaunay geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReplacementOrientationPolicy {
    /// Allow coherent replacement simplices regardless of geometric sign.
    AllowSigned,
    /// Require replacement simplices to stay in positive canonical orientation.
    RequirePositive,
}

/// Selects the amount of TDS structure checked before committing a flip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FlipValidationScope {
    /// Validate the whole triangulation data structure.
    FullTds,
    /// Validate only the simplices whose adjacency can change during a cavity flip.
    LocalCavity,
}

/// Checks the flip cavity after mutation without rescanning the full TDS.
fn validate_flip_trial_cavity<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    new_simplices: &[SimplexKey],
    external_facets: &[FacetHandle],
    removed_simplices: &[SimplexKey],
) -> Result<(), TdsValidationFailure>
where
    U: DataType,
    V: DataType,
{
    for &simplex_key in removed_simplices {
        if tds.contains_simplex(simplex_key) {
            return Err(TdsValidationFailure::InconsistentDataStructure {
                message: format!("flip trial still contains removed simplex {simplex_key:?}"),
            });
        }
        if tds.simplex_uuid_from_key(simplex_key).is_some() {
            return Err(TdsValidationFailure::MappingInconsistency {
                entity: EntityKind::Simplex,
                message: format!("flip trial still maps removed simplex key {simplex_key:?}"),
            });
        }
    }

    let mut affected_simplices = SimplexKeyBuffer::new();
    let mut affected_set = FastHashSet::default();
    for &simplex_key in new_simplices {
        push_unique_simplex_key(simplex_key, &mut affected_simplices, &mut affected_set);
    }
    for facet in external_facets {
        push_unique_simplex_key(
            facet.simplex_key(),
            &mut affected_simplices,
            &mut affected_set,
        );
    }

    validate_flip_trial_local_facet_sharing(tds, &affected_simplices)?;

    for &simplex_key in &affected_simplices {
        validate_flip_trial_simplex(tds, simplex_key, removed_simplices)?;
    }

    Ok(())
}

/// Adds a simplex to a small worklist while preserving first-seen order.
fn push_unique_simplex_key(
    simplex_key: SimplexKey,
    simplices: &mut SimplexKeyBuffer,
    seen: &mut FastHashSet<SimplexKey>,
) {
    if seen.insert(simplex_key) {
        simplices.push(simplex_key);
    }
}

/// Ensures affected replacement simplices agree on shared facets and multiplicity.
fn validate_flip_trial_local_facet_sharing<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    affected_simplices: &[SimplexKey],
) -> Result<(), TdsValidationFailure>
where
    U: DataType,
    V: DataType,
{
    type FacetIncidents = SmallBuffer<(SimplexKey, u8), 2>;
    let mut facet_to_simplices: FastHashMap<u64, FacetIncidents> = FastHashMap::default();

    for &simplex_key in affected_simplices {
        let simplex =
            tds.simplex(simplex_key)
                .ok_or_else(|| TdsValidationFailure::SimplexNotFound {
                    simplex_key,
                    context: "flip trial local facet sharing".to_string(),
                })?;
        if simplex.number_of_vertices() != D + 1 {
            return Err(TdsValidationFailure::DimensionMismatch {
                expected: D + 1,
                actual: simplex.number_of_vertices(),
                context: format!("flip trial simplex {simplex_key:?} arity"),
            });
        }

        for facet_idx in 0..simplex.number_of_vertices() {
            let facet_vertices = facet_vertices_from_simplex(simplex, facet_idx);
            let facet_idx_u8 =
                u8::try_from(facet_idx).map_err(|_| TdsValidationFailure::IndexOutOfBounds {
                    index: facet_idx,
                    bound: usize::from(u8::MAX),
                    context: "flip trial facet index".to_string(),
                })?;
            facet_to_simplices
                .entry(facet_key_from_vertices(&facet_vertices))
                .or_default()
                .push((simplex_key, facet_idx_u8));
        }
    }

    for (facet_key, incidents) in facet_to_simplices {
        match incidents.as_slice() {
            [_] => {}
            [(simplex_a, facet_a), (simplex_b, facet_b)] => {
                validate_flip_trial_mutual_facet_neighbors(
                    tds,
                    facet_key,
                    *simplex_a,
                    usize::from(*facet_a),
                    *simplex_b,
                    usize::from(*facet_b),
                )?;
            }
            _ => {
                return Err(TdsValidationFailure::Facet {
                    source: FacetError::InvalidFacetMultiplicity {
                        facet_key,
                        found: incidents.len(),
                    },
                });
            }
        }
    }

    Ok(())
}

/// Checks one affected simplex's local references after a flip mutation.
fn validate_flip_trial_simplex<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
    removed_simplices: &[SimplexKey],
) -> Result<(), TdsValidationFailure>
where
    U: DataType,
    V: DataType,
{
    let simplex =
        tds.simplex(simplex_key)
            .ok_or_else(|| TdsValidationFailure::SimplexNotFound {
                simplex_key,
                context: "flip trial local simplex validation".to_string(),
            })?;
    if tds.simplex_uuid_from_key(simplex_key) != Some(simplex.uuid()) {
        return Err(TdsValidationFailure::MappingInconsistency {
            entity: EntityKind::Simplex,
            message: format!(
                "missing or inconsistent UUID mapping for flip trial simplex {simplex_key:?}"
            ),
        });
    }

    if simplex.number_of_vertices() != D + 1 {
        return Err(TdsValidationFailure::DimensionMismatch {
            expected: D + 1,
            actual: simplex.number_of_vertices(),
            context: format!("flip trial simplex {simplex_key:?} arity"),
        });
    }

    validate_flip_trial_simplex_vertices(tds, simplex_key, simplex)?;
    validate_flip_trial_simplex_neighbors(tds, simplex_key, simplex, removed_simplices)
}

/// Verifies that affected simplices reference existing vertices with valid incidence.
fn validate_flip_trial_simplex_vertices<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
    simplex: &Simplex<V, D>,
) -> Result<(), TdsValidationFailure>
where
    U: DataType,
    V: DataType,
{
    let mut seen_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(simplex.number_of_vertices());
    for &vertex_key in simplex.vertices() {
        if seen_vertices.contains(&vertex_key) {
            return Err(TdsValidationFailure::InconsistentDataStructure {
                message: format!(
                    "flip trial simplex {simplex_key:?} repeats vertex {vertex_key:?}"
                ),
            });
        }
        seen_vertices.push(vertex_key);

        let vertex =
            tds.vertex(vertex_key)
                .ok_or_else(|| TdsValidationFailure::VertexNotFound {
                    vertex_key,
                    context: format!("flip trial simplex {simplex_key:?} vertex reference"),
                })?;
        if tds.vertex_uuid_from_key(vertex_key) != Some(vertex.uuid()) {
            return Err(TdsValidationFailure::MappingInconsistency {
                entity: EntityKind::Vertex,
                message: format!(
                    "missing or inconsistent UUID mapping for flip trial vertex {vertex_key:?}"
                ),
            });
        }
        let Some(incident_simplex_key) = vertex.incident_simplex() else {
            continue;
        };
        let incident_simplex = tds.simplex(incident_simplex_key).ok_or_else(|| {
            TdsValidationFailure::SimplexNotFound {
                simplex_key: incident_simplex_key,
                context: format!("dangling incident_simplex pointer from vertex {vertex_key:?}"),
            }
        })?;
        if !incident_simplex.contains_vertex(vertex_key) {
            return Err(TdsValidationFailure::InconsistentDataStructure {
                message: format!(
                    "Vertex {vertex_key:?} incident_simplex {incident_simplex_key:?} does not contain the vertex"
                ),
            });
        }
    }

    Ok(())
}

/// Verifies affected-simplex neighbor links, mirror facets, and orientation parity.
fn validate_flip_trial_simplex_neighbors<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
    simplex: &Simplex<V, D>,
    removed_simplices: &[SimplexKey],
) -> Result<(), TdsValidationFailure>
where
    U: DataType,
    V: DataType,
{
    let Some(neighbors) = simplex.neighbor_slots() else {
        return Ok(());
    };
    if neighbors.len() != D + 1 {
        return Err(TdsValidationFailure::InvalidNeighbors {
            reason: NeighborValidationError::LengthMismatch {
                actual: neighbors.len(),
                expected: D + 1,
                context: "flip trial neighbor validation".to_string(),
            },
        });
    }

    for (facet_idx, neighbor_slot) in neighbors.iter().copied().enumerate() {
        let neighbor_key = match neighbor_slot {
            NeighborSlot::Unassigned => {
                return Err(TdsValidationFailure::InvalidNeighbors {
                    reason: NeighborValidationError::UnassignedNeighborSlot {
                        simplex_key,
                        simplex_uuid: simplex.uuid(),
                        facet_index: facet_idx,
                        context: "flip trial neighbor validation".to_string(),
                    },
                });
            }
            NeighborSlot::Boundary => continue,
            NeighborSlot::Neighbor(neighbor_key) => neighbor_key,
        };
        if removed_simplices.contains(&neighbor_key) {
            return Err(TdsValidationFailure::InvalidNeighbors {
                reason: NeighborValidationError::ReferencedRemovedNeighbor {
                    simplex_key,
                    simplex_uuid: simplex.uuid(),
                    facet_index: facet_idx,
                    neighbor_key,
                },
            });
        }
        if neighbor_key == simplex_key {
            if simplex_allows_periodic_self_neighbor(simplex) {
                continue;
            }
            return Err(TdsValidationFailure::InvalidNeighbors {
                reason: NeighborValidationError::NonPeriodicSelfNeighbor {
                    simplex_key,
                    simplex_uuid: simplex.uuid(),
                    facet_index: facet_idx,
                },
            });
        }

        let neighbor_simplex =
            tds.simplex(neighbor_key)
                .ok_or_else(|| TdsValidationFailure::InvalidNeighbors {
                    reason: NeighborValidationError::MissingNeighborSimplex {
                        simplex_key,
                        simplex_uuid: simplex.uuid(),
                        facet_index: facet_idx,
                        neighbor_key,
                        context: "flip trial neighbor validation".to_string(),
                    },
                })?;
        let mirror_idx = simplex
            .mirror_facet_index(facet_idx, neighbor_simplex)
            .ok_or_else(|| TdsValidationFailure::InvalidNeighbors {
                reason: NeighborValidationError::MirrorFacetMissing {
                    simplex_uuid: simplex.uuid(),
                    facet_index: facet_idx,
                    neighbor_uuid: neighbor_simplex.uuid(),
                    context: "flip trial neighbor validation".to_string(),
                },
            })?;
        validate_flip_trial_mutual_facet_neighbors(
            tds,
            facet_key_from_vertices(&facet_vertices_from_simplex(simplex, facet_idx)),
            simplex_key,
            facet_idx,
            neighbor_key,
            mirror_idx,
        )?;
        validate_flip_trial_neighbor_orientation(
            simplex_key,
            simplex,
            facet_idx,
            neighbor_key,
            neighbor_simplex,
            mirror_idx,
        )?;
    }

    Ok(())
}

/// Mirrors TDS validation's periodic self-neighbor allowance locally.
fn simplex_allows_periodic_self_neighbor<V, const D: usize>(simplex: &Simplex<V, D>) -> bool {
    let Some(offsets) = simplex.periodic_vertex_offsets() else {
        return false;
    };
    !offsets.is_empty() && offsets.len() == simplex.number_of_vertices()
}

/// Requires two simplices sharing an affected facet to point back to each other.
fn validate_flip_trial_mutual_facet_neighbors<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    facet_key: u64,
    source_simplex_key: SimplexKey,
    source_facet: usize,
    target_simplex_key: SimplexKey,
    target_facet: usize,
) -> Result<(), TdsValidationFailure>
where
    U: DataType,
    V: DataType,
{
    let source_simplex =
        tds.simplex(source_simplex_key)
            .ok_or_else(|| TdsValidationFailure::SimplexNotFound {
                simplex_key: source_simplex_key,
                context: "flip trial mutual neighbor validation".to_string(),
            })?;
    let target_simplex =
        tds.simplex(target_simplex_key)
            .ok_or_else(|| TdsValidationFailure::SimplexNotFound {
                simplex_key: target_simplex_key,
                context: "flip trial mutual neighbor validation".to_string(),
            })?;

    let source_neighbor = source_simplex.neighbor_key(source_facet).flatten();
    let target_neighbor = target_simplex.neighbor_key(target_facet).flatten();

    if source_neighbor != Some(target_simplex_key) || target_neighbor != Some(source_simplex_key) {
        return Err(TdsValidationFailure::InvalidNeighbors {
            reason: NeighborValidationError::InteriorFacetNeighborMismatch {
                facet_key,
                first_simplex_key: source_simplex_key,
                first_simplex_uuid: source_simplex.uuid(),
                first_facet_index: source_facet,
                first_neighbor: source_neighbor,
                second_simplex_key: target_simplex_key,
                second_simplex_uuid: target_simplex.uuid(),
                second_facet_index: target_facet,
                second_neighbor: target_neighbor,
            },
        });
    }

    Ok(())
}

/// Checks coherent orientation across one locally affected neighbor pair.
fn validate_flip_trial_neighbor_orientation<V, const D: usize>(
    simplex_key: SimplexKey,
    simplex: &Simplex<V, D>,
    facet_idx: usize,
    neighbor_key: SimplexKey,
    neighbor_simplex: &Simplex<V, D>,
    mirror_idx: usize,
) -> Result<(), TdsValidationFailure> {
    let (observed_odd_permutation, expected_odd_permutation, facet_vertex_count, target_count) =
        match flip_trial_neighbor_orientation_parity(
            simplex_key,
            simplex,
            facet_idx,
            neighbor_key,
            neighbor_simplex,
            mirror_idx,
        ) {
            Ok(parity) => parity,
            Err(FlipError::InvalidFlipContext { reason })
                if matches!(
                    reason.as_ref(),
                    FlipContextError::FacetOrderParityUnavailable
                ) =>
            {
                return Err(TdsValidationFailure::InconsistentDataStructure {
                    message: format!(
                        "Could not derive facet-order permutation parity between simplices {:?} and {:?}",
                        simplex.uuid(),
                        neighbor_simplex.uuid()
                    ),
                });
            }
            Err(err) => {
                return Err(TdsValidationFailure::InvalidNeighbors {
                    reason: NeighborValidationError::FacetOrderUnavailable {
                        simplex_key,
                        simplex_uuid: simplex.uuid(),
                        facet_index: facet_idx,
                        context: "facet parity in local flip validation".to_string(),
                        source: Box::new(err),
                    },
                });
            }
        };
    if observed_odd_permutation != expected_odd_permutation {
        return Err(TdsValidationFailure::OrientationViolation {
            simplex1_key: simplex_key,
            simplex1_uuid: simplex.uuid(),
            simplex2_key: neighbor_key,
            simplex2_uuid: neighbor_simplex.uuid(),
            simplex1_facet_index: facet_idx,
            simplex2_facet_index: mirror_idx,
            facet_vertex_count,
            simplex2_facet_vertex_count: target_count,
            observed_odd_permutation,
            expected_odd_permutation,
        });
    }

    Ok(())
}

/// Computes local neighbor-orientation parity, including periodic facet offsets.
fn flip_trial_neighbor_orientation_parity<V, const D: usize>(
    simplex_key: SimplexKey,
    simplex: &Simplex<V, D>,
    facet_idx: usize,
    neighbor_key: SimplexKey,
    neighbor_simplex: &Simplex<V, D>,
    mirror_idx: usize,
) -> Result<(bool, bool, usize, usize), FlipError> {
    let expected_odd_permutation = (facet_idx + mirror_idx).is_multiple_of(2);
    if simplex.periodic_vertex_offsets().is_some()
        || neighbor_simplex.periodic_vertex_offsets().is_some()
    {
        let source_offsets = periodic_offsets_or_zero_frame(simplex_key, simplex)?;
        let target_offsets = periodic_offsets_or_zero_frame(neighbor_key, neighbor_simplex)?;
        let source_order = normalized_facet_order_with_offsets(
            simplex_key,
            simplex.vertices(),
            source_offsets.as_ref(),
            facet_idx,
        )?;
        let target_order = normalized_facet_order_with_offsets(
            neighbor_key,
            neighbor_simplex.vertices(),
            target_offsets.as_ref(),
            mirror_idx,
        )?;
        let observed_odd_permutation = permutation_odd(&source_order, &target_order)
            .ok_or(FlipContextError::FacetOrderParityUnavailable)?;
        return Ok((
            observed_odd_permutation,
            expected_odd_permutation,
            source_order.len(),
            target_order.len(),
        ));
    }

    let source_order = facet_order(simplex.vertices(), facet_idx)?;
    let target_order = facet_order(neighbor_simplex.vertices(), mirror_idx)?;
    let observed_odd_permutation = permutation_odd(&source_order, &target_order)
        .ok_or(FlipContextError::FacetOrderParityUnavailable)?;
    Ok((
        observed_odd_permutation,
        expected_odd_permutation,
        source_order.len(),
        target_order.len(),
    ))
}

/// Detects replacement simplices that already exist outside the flip cavity.
///
/// This protects the bistellar link condition while also treating stale
/// incidence entries as structural corruption instead of silently ignoring
/// them.
///
/// # Errors
///
/// Returns [`FlipError::DanglingVertexIncidence`] if
/// [`Tds::simplex_keys_containing_vertex`] yields a simplex key that is no
/// longer present in storage.
fn find_simplex_containing_simplex<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_vertices: &[VertexKey],
    removed_simplices: &[SimplexKey],
) -> Result<Option<SimplexKey>, FlipError> {
    let Some(&first) = simplex_vertices.first() else {
        return Ok(None);
    };

    for simplex_key in tds.simplex_keys_containing_vertex(first) {
        let Some(simplex) = tds.simplex(simplex_key) else {
            return Err(FlipError::DanglingVertexIncidence {
                vertex_key: first,
                simplex_key,
            });
        };

        if removed_simplices.contains(&simplex_key) {
            continue;
        }

        if simplex_vertices
            .iter()
            .copied()
            .all(|vk| simplex.contains_vertex(vk))
        {
            return Ok(Some(simplex_key));
        }
    }

    Ok(None)
}

/// Chooses replacement-simplex parity from the oriented cavity boundary.
fn orient_replacement_simplices<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplices: &mut [SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>],
    periodic_offsets: &mut [Option<PeriodicOffsetBuffer<D>>],
    external_facets: &[FacetHandle],
) -> Result<(), FlipError> {
    let mut flips = SmallBuffer::from_elem(None, simplices.len());
    if periodic_offsets.len() != simplices.len() {
        return Err(FlipContextError::ReplacementPeriodicOffsetCountMismatch {
            simplex_count: simplices.len(),
            offset_count: periodic_offsets.len(),
        }
        .into());
    }

    assign_external_replacement_orientation(
        tds,
        simplices,
        periodic_offsets,
        external_facets,
        &mut flips,
    )?;

    loop {
        let mut changed = false;

        for source_idx in 0..simplices.len() {
            for target_idx in (source_idx + 1)..simplices.len() {
                let Some((source_facet_idx, target_facet_idx)) =
                    shared_facet_indices(&simplices[source_idx], &simplices[target_idx])
                else {
                    continue;
                };
                let coherent = facet_orders_coherent(
                    &simplices[source_idx],
                    source_facet_idx,
                    &simplices[target_idx],
                    target_facet_idx,
                )?;
                match (flips[source_idx], flips[target_idx]) {
                    (Some(source_flip), Some(target_flip)) => {
                        if target_flip != (source_flip ^ !coherent) {
                            return Err(
                                FlipContextError::ConflictingReplacementOrientationBetweenSimplices {
                                    source_simplex_index: source_idx,
                                    target_simplex_index: target_idx,
                                }
                                .into(),
                            );
                        }
                    }
                    (Some(source_flip), None) => {
                        changed |=
                            set_flip_assignment(&mut flips, target_idx, source_flip ^ !coherent)?;
                    }
                    (None, Some(target_flip)) => {
                        changed |=
                            set_flip_assignment(&mut flips, source_idx, target_flip ^ !coherent)?;
                    }
                    (None, None) => {}
                }
            }
        }

        if flips.iter().all(Option::is_some) {
            break;
        }

        if !changed {
            let Some(root_idx) = flips.iter().position(Option::is_none) else {
                break;
            };
            flips[root_idx] = Some(false);
        }
    }

    for ((vertices, offsets), should_flip) in simplices.iter_mut().zip(periodic_offsets).zip(flips)
    {
        if should_flip.unwrap_or(false) {
            if vertices.len() < 2 {
                return Err(FlipContextError::ReplacementSimplexTooSmallForOrientationFlip.into());
            }
            vertices.swap(0, 1);
            if let Some(offsets) = offsets {
                offsets.swap(0, 1);
            }
        }
    }

    Ok(())
}

/// Applies external boundary-facet parity constraints to replacement simplices.
fn assign_external_replacement_orientation<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplices: &[SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>],
    periodic_offsets: &[Option<PeriodicOffsetBuffer<D>>],
    external_facets: &[FacetHandle],
    flips: &mut SmallBuffer<Option<bool>, MAX_PRACTICAL_DIMENSION_SIZE>,
) -> Result<(), FlipError> {
    for &external in external_facets {
        let external_simplex =
            tds.simplex(external.simplex_key())
                .ok_or_else(|| FlipError::MissingSimplex {
                    simplex_key: external.simplex_key(),
                })?;
        let external_offsets =
            periodic_offsets_or_zero_frame(external.simplex_key(), external_simplex)?;

        let external_facet_idx = usize::from(external.facet_index());
        for (simplex_idx, vertices) in simplices.iter().enumerate() {
            let Some(replacement_facet_idx) =
                matching_facet_index(external_simplex.vertices(), external_facet_idx, vertices)?
            else {
                continue;
            };
            let coherent = if external_simplex.periodic_vertex_offsets().is_some()
                || periodic_offsets[simplex_idx].is_some()
            {
                let Some(replacement_offsets) = periodic_offsets[simplex_idx].as_deref() else {
                    return Err(FlipContextError::MissingReplacementPeriodicOffsets {
                        simplex_index: simplex_idx,
                    }
                    .into());
                };
                facet_orders_coherent_with_periodic_offsets(&PeriodicFacetParityContext {
                    source_vertices: external_simplex.vertices(),
                    source_offsets: external_offsets.as_ref(),
                    source_facet_idx: external_facet_idx,
                    target_vertices: vertices,
                    target_offsets: replacement_offsets,
                    target_facet_idx: replacement_facet_idx,
                    source_simplex_key: external.simplex_key(),
                    target_simplex_index: simplex_idx,
                })?
            } else {
                facet_orders_coherent(
                    external_simplex.vertices(),
                    external_facet_idx,
                    vertices,
                    replacement_facet_idx,
                )?
            };
            set_flip_assignment(flips, simplex_idx, !coherent)?;
        }
    }

    Ok(())
}

/// Records a required local parity flip and rejects contradictory constraints.
fn set_flip_assignment(
    assignments: &mut SmallBuffer<Option<bool>, MAX_PRACTICAL_DIMENSION_SIZE>,
    simplex_idx: usize,
    required: bool,
) -> Result<bool, FlipError> {
    if simplex_idx >= assignments.len() {
        return Err(FlipContextError::ReplacementOrientationIndexOutOfRange {
            simplex_index: simplex_idx,
        }
        .into());
    }

    match assignments[simplex_idx] {
        Some(existing) if existing != required => Err(
            FlipContextError::ConflictingReplacementOrientationForSimplex {
                simplex_index: simplex_idx,
            }
            .into(),
        ),
        Some(_) => Ok(false),
        None => {
            assignments[simplex_idx] = Some(required);
            Ok(true)
        }
    }
}

/// Builds periodic offsets for replacement simplices in one shared cavity frame.
fn replacement_simplex_periodic_offsets<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplices: &[SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>],
    removed_simplices: &[SimplexKey],
    external_facets: &[FacetHandle],
    newly_inserted_vertex: Option<VertexKey>,
) -> Result<ReplacementPeriodicOffsets<D>, FlipError> {
    let source_simplices =
        replacement_periodic_source_simplices(removed_simplices, external_facets);
    if !replacement_sources_use_periodic_offsets(tds, &source_simplices)? {
        return Ok(SmallBuffer::from_elem(None, simplices.len()));
    }

    let target_simplex_key = *removed_simplices
        .first()
        .ok_or(FlipContextError::MissingRemovedSimplexFrame)?;
    let mut offsets_by_simplex = ReplacementPeriodicOffsets::<D>::with_capacity(simplices.len());

    for vertices in simplices {
        let mut offsets = PeriodicOffsetBuffer::<D>::with_capacity(vertices.len());
        for &vertex_key in vertices {
            let offset = if Some(vertex_key) == newly_inserted_vertex
                && !source_simplices_contain_vertex(tds, &source_simplices, vertex_key)?
            {
                new_vertex_periodic_offset_in_frame(tds, target_simplex_key)?
            } else {
                periodic_offset_lifted_into_simplex(
                    tds,
                    vertex_key,
                    target_simplex_key,
                    &source_simplices,
                )?
            };
            offsets.push(offset);
        }
        offsets_by_simplex.push(Some(offsets));
    }

    Ok(offsets_by_simplex)
}

/// Collects removed and external simplices that can witness periodic frame alignment.
fn replacement_periodic_source_simplices(
    removed_simplices: &[SimplexKey],
    external_facets: &[FacetHandle],
) -> SimplexKeyBuffer {
    let mut source_simplices = SimplexKeyBuffer::new();
    let mut seen = FastHashSet::default();
    for &simplex_key in removed_simplices {
        push_unique_simplex_key(simplex_key, &mut source_simplices, &mut seen);
    }
    for external in external_facets {
        push_unique_simplex_key(external.simplex_key(), &mut source_simplices, &mut seen);
    }
    source_simplices
}

/// Returns whether any source simplex carries explicit periodic offsets.
fn replacement_sources_use_periodic_offsets<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    source_simplices: &[SimplexKey],
) -> Result<bool, FlipError> {
    let mut uses_periodic_offsets = false;
    for &simplex_key in source_simplices {
        let simplex = tds
            .simplex(simplex_key)
            .ok_or(FlipError::MissingSimplex { simplex_key })?;
        if let Some(offsets) = simplex.periodic_vertex_offsets() {
            validate_periodic_offset_len(simplex_key, simplex, offsets)?;
            uses_periodic_offsets = true;
        }
    }
    Ok(uses_periodic_offsets)
}

/// Checks whether a vertex already has a periodic representative in any source simplex.
fn source_simplices_contain_vertex<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    source_simplices: &[SimplexKey],
    vertex_key: VertexKey,
) -> Result<bool, FlipError> {
    for &simplex_key in source_simplices {
        let simplex = tds
            .simplex(simplex_key)
            .ok_or(FlipError::MissingSimplex { simplex_key })?;
        if simplex.contains_vertex(vertex_key) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Places a newly inserted k=1 vertex in the target simplex's local lattice sheet.
fn new_vertex_periodic_offset_in_frame<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    target_simplex_key: SimplexKey,
) -> Result<[i8; D], FlipError> {
    let target_simplex = tds
        .simplex(target_simplex_key)
        .ok_or(FlipError::MissingSimplex {
            simplex_key: target_simplex_key,
        })?;
    let target_offsets = periodic_offsets_or_zero_frame(target_simplex_key, target_simplex)?;
    Ok(target_offsets.first().copied().unwrap_or([0_i8; D]))
}

/// Finds the target facet opposite the source facet, if the simplices share it.
fn matching_facet_index(
    source_vertices: &[VertexKey],
    source_facet_idx: usize,
    target_vertices: &[VertexKey],
) -> Result<Option<usize>, FlipError> {
    if source_vertices.len() != target_vertices.len() {
        return Ok(None);
    }

    let source_facet = facet_order(source_vertices, source_facet_idx)?;
    if !source_facet
        .iter()
        .copied()
        .all(|vertex| target_vertices.contains(&vertex))
    {
        return Ok(None);
    }

    let mut target_facet_idx = None;
    for (idx, &vertex) in target_vertices.iter().enumerate() {
        if source_facet.contains(&vertex) {
            continue;
        }
        if target_facet_idx.is_some() {
            return Ok(None);
        }
        target_facet_idx = Some(idx);
    }

    Ok(target_facet_idx)
}

/// Finds the opposite slots for two replacement simplices that share a facet.
fn shared_facet_indices(
    source_vertices: &[VertexKey],
    target_vertices: &[VertexKey],
) -> Option<(usize, usize)> {
    if source_vertices.len() != target_vertices.len() {
        return None;
    }

    let source_facet_idx = unique_vertex_index(source_vertices, target_vertices)?;
    let target_facet_idx = unique_vertex_index(target_vertices, source_vertices)?;
    Some((source_facet_idx, target_facet_idx))
}

/// Returns the single vertex slot in `vertices` that is absent from `other`.
fn unique_vertex_index(vertices: &[VertexKey], other: &[VertexKey]) -> Option<usize> {
    let mut unique_idx = None;
    for (idx, &vertex) in vertices.iter().enumerate() {
        if other.contains(&vertex) {
            continue;
        }
        if unique_idx.is_some() {
            return None;
        }
        unique_idx = Some(idx);
    }
    unique_idx
}

/// Checks the TDS coherent-orientation parity convention for one shared facet.
fn facet_orders_coherent(
    source_vertices: &[VertexKey],
    source_facet_idx: usize,
    target_vertices: &[VertexKey],
    target_facet_idx: usize,
) -> Result<bool, FlipError> {
    let source_order = facet_order(source_vertices, source_facet_idx)?;
    let target_order = facet_order(target_vertices, target_facet_idx)?;
    let observed_odd = permutation_odd(&source_order, &target_order)
        .ok_or(FlipContextError::FacetOrderParityUnavailable)?;
    let expected_odd = (source_facet_idx + target_facet_idx).is_multiple_of(2);
    Ok(observed_odd == expected_odd)
}

/// Inputs needed to compare one periodic source facet with a replacement facet.
struct PeriodicFacetParityContext<'a, const D: usize> {
    source_vertices: &'a [VertexKey],
    source_offsets: &'a [[i8; D]],
    source_facet_idx: usize,
    target_vertices: &'a [VertexKey],
    target_offsets: &'a [[i8; D]],
    target_facet_idx: usize,
    source_simplex_key: SimplexKey,
    target_simplex_index: usize,
}

/// Checks facet parity after aligning a periodic source facet into a replacement frame.
fn facet_orders_coherent_with_periodic_offsets<const D: usize>(
    context: &PeriodicFacetParityContext<'_, D>,
) -> Result<bool, FlipError> {
    if context.source_offsets.len() != context.source_vertices.len() {
        return Err(FlipContextError::PeriodicOffsetCountMismatch {
            simplex_key: context.source_simplex_key,
            offset_count: context.source_offsets.len(),
            vertex_count: context.source_vertices.len(),
        }
        .into());
    }
    if context.target_offsets.len() != context.target_vertices.len() {
        return Err(FlipContextError::ReplacementPeriodicOffsetLengthMismatch {
            simplex_index: context.target_simplex_index,
            offset_count: context.target_offsets.len(),
            vertex_count: context.target_vertices.len(),
        }
        .into());
    }

    let source_order = facet_order_with_offsets(
        context.source_vertices,
        context.source_offsets,
        context.source_facet_idx,
    )?;
    let target_order = facet_order_with_offsets(
        context.target_vertices,
        context.target_offsets,
        context.target_facet_idx,
    )?;
    let aligned_source_order = align_periodic_facet_order(
        &source_order,
        &target_order,
        context.source_simplex_key,
        context.target_simplex_index,
    )?;
    let observed_odd = permutation_odd(&aligned_source_order, &target_order)
        .ok_or(FlipContextError::FacetOrderParityUnavailable)?;
    let expected_odd = (context.source_facet_idx + context.target_facet_idx).is_multiple_of(2);
    Ok(observed_odd == expected_odd)
}

/// Returns facet `(offset)` identities in simplex-local order.
fn facet_order_with_offsets<const D: usize>(
    vertices: &[VertexKey],
    offsets: &[[i8; D]],
    omit_idx: usize,
) -> Result<SmallBuffer<(VertexKey, [i8; D]), MAX_PRACTICAL_DIMENSION_SIZE>, FlipError> {
    if omit_idx >= vertices.len() {
        return Err(FlipContextError::ReplacementFacetIndexOutOfRange {
            facet_index: omit_idx,
            vertex_count: vertices.len(),
        }
        .into());
    }

    let mut order: SmallBuffer<(VertexKey, [i8; D]), MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(vertices.len().saturating_sub(1));
    for (idx, &vertex) in vertices.iter().enumerate() {
        if idx != omit_idx {
            order.push((vertex, offsets[idx]));
        }
    }
    Ok(order)
}

/// Returns simplex-local facet identities with offsets normalized by a stable anchor.
fn normalized_facet_order_with_offsets<const D: usize>(
    simplex_key: SimplexKey,
    vertices: &[VertexKey],
    offsets: &[[i8; D]],
    omit_idx: usize,
) -> Result<SmallBuffer<(VertexKey, [i16; D]), MAX_PRACTICAL_DIMENSION_SIZE>, FlipError> {
    if offsets.len() != vertices.len() {
        return Err(FlipContextError::PeriodicOffsetCountMismatch {
            simplex_key,
            offset_count: offsets.len(),
            vertex_count: vertices.len(),
        }
        .into());
    }
    if omit_idx >= vertices.len() {
        return Err(FlipContextError::ReplacementFacetIndexOutOfRange {
            facet_index: omit_idx,
            vertex_count: vertices.len(),
        }
        .into());
    }

    let mut order: SmallBuffer<(VertexKey, [i16; D]), MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(vertices.len().saturating_sub(1));
    for (idx, &vertex) in vertices.iter().enumerate() {
        if idx == omit_idx {
            continue;
        }
        let mut offset = [0_i16; D];
        for axis in 0..D {
            offset[axis] = i16::from(offsets[idx][axis]);
        }
        order.push((vertex, offset));
    }

    let mut anchor_key = u64::MAX;
    let mut anchor_offset = [0_i16; D];
    for (vertex, offset) in &order {
        let key_value = (*vertex).data().as_ffi();
        if key_value < anchor_key || (key_value == anchor_key && *offset < anchor_offset) {
            anchor_key = key_value;
            anchor_offset = *offset;
        }
    }
    for (_, offset) in &mut order {
        for axis in 0..D {
            offset[axis] -= anchor_offset[axis];
        }
    }

    Ok(order)
}

/// Translates source facet offsets into the target replacement frame.
fn align_periodic_facet_order<const D: usize>(
    source_order: &[(VertexKey, [i8; D])],
    target_order: &[(VertexKey, [i8; D])],
    source_simplex_key: SimplexKey,
    target_simplex_index: usize,
) -> Result<SmallBuffer<(VertexKey, [i8; D]), MAX_PRACTICAL_DIMENSION_SIZE>, FlipError> {
    let mut aligned_order = SmallBuffer::with_capacity(source_order.len());
    for &(vertex_key, source_vertex_offset) in source_order {
        let mut aligned_offset: Option<[i8; D]> = None;
        for &(reference_vertex, source_reference_offset) in source_order {
            let Some((_, target_reference_offset)) = target_order
                .iter()
                .find(|(target_vertex, _)| *target_vertex == reference_vertex)
            else {
                return Err(FlipContextError::FacetOrderParityUnavailable.into());
            };
            let candidate_offset = align_periodic_offset(
                source_vertex_offset,
                source_reference_offset,
                *target_reference_offset,
            )?;
            if let Some(expected_offset) = aligned_offset {
                if candidate_offset != expected_offset {
                    return Err(
                        FlipContextError::ConflictingReplacementPeriodicFrameTranslation {
                            vertex_key,
                            source_simplex_key,
                            target_simplex_index,
                            expected_offset: expected_offset.into(),
                            found_offset: candidate_offset.into(),
                        }
                        .into(),
                    );
                }
            } else {
                aligned_offset = Some(candidate_offset);
            }
        }
        let Some(offset) = aligned_offset else {
            return Err(FlipContextError::FacetOrderParityUnavailable.into());
        };
        aligned_order.push((vertex_key, offset));
    }
    Ok(aligned_order)
}

/// Returns facet vertices in simplex-local order.
fn facet_order(
    vertices: &[VertexKey],
    omit_idx: usize,
) -> Result<SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>, FlipError> {
    if omit_idx >= vertices.len() {
        return Err(FlipContextError::ReplacementFacetIndexOutOfRange {
            facet_index: omit_idx,
            vertex_count: vertices.len(),
        }
        .into());
    }

    let mut order = SmallBuffer::with_capacity(vertices.len().saturating_sub(1));
    for (idx, &vertex) in vertices.iter().enumerate() {
        if idx != omit_idx {
            order.push(vertex);
        }
    }
    Ok(order)
}

/// Returns whether the permutation from `source_order` to `target_order` is odd.
fn permutation_odd<Id: PartialEq>(source_order: &[Id], target_order: &[Id]) -> Option<bool> {
    if source_order.len() != target_order.len() {
        return None;
    }

    let mut target_positions: SmallBuffer<usize, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(source_order.len());
    let mut used_target_indices: SmallBuffer<bool, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::from_elem(false, target_order.len());

    for source_vertex in source_order {
        let mut matched_target_position = None;
        for (target_idx, target_vertex) in target_order.iter().enumerate() {
            if target_vertex == source_vertex && !used_target_indices[target_idx] {
                matched_target_position = Some(target_idx);
                used_target_indices[target_idx] = true;
                break;
            }
        }
        target_positions.push(matched_target_position?);
    }

    let mut inversion_count = 0usize;
    for i in 0..target_positions.len() {
        for j in (i + 1)..target_positions.len() {
            if target_positions[i] > target_positions[j] {
                inversion_count += 1;
            }
        }
    }

    Some(inversion_count % 2 == 1)
}

/// Ensures Delaunay-repair replacement simplices have positive geometric orientation.
fn validate_replacement_orientation<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplices: &[SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>],
) -> Result<(), FlipError>
where
    U: DataType,
    V: DataType,
{
    for vertices in simplices {
        let points = vertices_to_points(tds, vertices)?;
        match robust_orientation(&points) {
            Ok(Orientation::POSITIVE) => {}
            Ok(Orientation::DEGENERATE) => return Err(FlipError::DegenerateSimplex),
            Ok(Orientation::NEGATIVE) => {
                return Err(FlipError::NegativeOrientation {
                    simplex_vertices: vertices.iter().copied().collect(),
                });
            }
            Err(error) => {
                return Err(FlipPredicateError::coordinate_conversion(
                    FlipPredicateOperation::DelaunayRepairReplacementOrientation,
                    error,
                )
                .into());
            }
        }
    }
    Ok(())
}

/// Scans the whole TDS for ridge diagnostics when local neighbor links are the
/// thing being investigated.
fn simplices_containing_vertices<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    vertices: &[VertexKey],
) -> SimplexKeyBuffer
where
    U: DataType,
    V: DataType,
{
    let mut simplices = SimplexKeyBuffer::new();
    'simplices: for (simplex_key, simplex) in tds.simplices() {
        for &vkey in vertices {
            if !simplex.contains_vertex(vkey) {
                continue 'simplices;
            }
        }
        simplices.push(simplex_key);
    }
    simplices
}

/// Emits a bounded ridge snapshot so repair failures can distinguish bad local
/// handles from genuinely inconsistent global incidence.
///
/// The local neighbor walk and the global simplex scan are logged side by side
/// because #204 currently fails in cases where those two views disagree.
fn debug_ridge_context<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    ridge: RidgeHandle,
    reported_multiplicity: Option<usize>,
    diagnostics: &mut RepairDiagnostics,
    last_applied_flip: Option<&LastAppliedFlip>,
) where
    U: DataType,
    V: DataType,
{
    if !should_emit_ridge_debug(diagnostics, reported_multiplicity) {
        return;
    }
    let Some(simplex) = tds.simplex(ridge.simplex_key()) else {
        tracing::debug!(
            ridge = ?ridge,
            reported_multiplicity,
            "repair: ridge debug skipped (simplex missing)"
        );
        return;
    };
    let omit_a = usize::from(ridge.omit_a());
    let omit_b = usize::from(ridge.omit_b());
    if omit_a >= simplex.number_of_vertices()
        || omit_b >= simplex.number_of_vertices()
        || omit_a == omit_b
    {
        tracing::debug!(
            ridge = ?ridge,
            omit_a,
            omit_b,
            vertex_count = simplex.number_of_vertices(),
            reported_multiplicity,
            "repair: ridge debug skipped (invalid indices)"
        );
        return;
    }

    let ridge_vertices = ridge_vertices_from_simplex(simplex, omit_a, omit_b);
    let neighbor_walk =
        collect_simplices_around_ridge(tds, ridge.simplex_key(), &ridge_vertices, None)
            .map(|simplices| simplices.into_iter().collect::<Vec<_>>());
    let global_simplices = simplices_containing_vertices(tds, &ridge_vertices);
    let neighbor_snapshot: Option<SmallBuffer<Option<SimplexKey>, MAX_PRACTICAL_DIMENSION_SIZE>> =
        simplex.neighbor_keys().map(Iterator::collect);
    let global_simplex_details: Vec<String> = global_simplices
        .iter()
        .copied()
        .map(|simplex_key| ridge_incident_simplex_summary(tds, simplex_key, &ridge_vertices))
        .collect();
    // Attach the immediately preceding flip so the snapshot can say whether repair
    // just created this ridge instead of forcing us to correlate separate log lines.
    let predecessor_summary =
        last_applied_flip.map(|last| predecessor_flip_summary(tds, ridge, &global_simplices, last));

    tracing::debug!(
        ridge = ?ridge,
        ridge_vertices = ?ridge_vertices,
        reported_multiplicity,
        neighbor_walk = ?neighbor_walk,
        global_count = global_simplices.len(),
        global_simplices = ?global_simplices,
        global_simplex_details = ?global_simplex_details,
        predecessor = ?predecessor_summary,
        simplex_neighbors = ?neighbor_snapshot,
        "repair: ridge adjacency debug snapshot"
    );
}

/// Formats one incident simplex around a ridge so debug output can distinguish
/// oversharing from bad local neighbor traversal.
fn ridge_incident_simplex_summary<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
    ridge_vertices: &SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
) -> String
where
    U: DataType,
    V: DataType,
{
    let Some(simplex) = tds.simplex(simplex_key) else {
        return format!("{simplex_key:?}: missing");
    };

    let extras = match simplex_extras_for_ridge(simplex_key, simplex, ridge_vertices) {
        Ok(extras) => extras,
        Err(err) => return format!("{simplex_key:?}: extras_error={err}"),
    };
    let ridge_neighbors = ridge_neighbor_simplices_for_simplex(simplex, ridge_vertices);
    format!("{simplex_key:?}: extras={extras:?} ridge_neighbors={ridge_neighbors:?}")
}

/// Extracts the neighbors reached by omitting the two vertices opposite the
/// ridge, which is exactly the adjacency walk used by k=3 context recovery.
fn ridge_neighbor_simplices_for_simplex<V, const D: usize>(
    simplex: &Simplex<V, D>,
    ridge_vertices: &SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
) -> SmallBuffer<SimplexKey, 2>
where
    V: DataType,
{
    let mut ridge_neighbors: SmallBuffer<SimplexKey, 2> = SmallBuffer::new();
    for (idx, &vertex_key) in simplex.vertices().iter().enumerate() {
        if ridge_vertices.contains(&vertex_key) {
            continue;
        }
        if let Some(neighbor_key) = simplex.neighbor_key(idx).flatten() {
            ridge_neighbors.push(neighbor_key);
        }
    }

    ridge_neighbors
}

/// Relates the current bad ridge to the immediately preceding flip so #204
/// traces can confirm whether repair just created the inconsistent local star.
fn predecessor_flip_summary<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    ridge: RidgeHandle,
    global_simplices: &[SimplexKey],
    last_applied_flip: &LastAppliedFlip,
) -> String
where
    U: DataType,
    V: DataType,
{
    let global_simplices_in_new: Vec<SimplexKey> = global_simplices
        .iter()
        .copied()
        .filter(|simplex_key| last_applied_flip.new_simplices.contains(simplex_key))
        .collect();
    // Show the predecessor's concrete simplices because simplex ids alone become hard to
    // interpret once slot reuse and additional flips start churning the local region.
    let predecessor_new_simplex_vertices: Vec<String> = last_applied_flip
        .new_simplices
        .iter()
        .copied()
        .map(|simplex_key| simplex_vertex_summary(tds, simplex_key))
        .collect();

    format!(
        "k={} removed_face={:?} inserted_face={:?} removed_simplices={:?} new_simplices={:?} ridge_simplex_is_new={} global_simplices_in_new={global_simplices_in_new:?} predecessor_new_simplex_vertices={predecessor_new_simplex_vertices:?}",
        last_applied_flip.kind.k(),
        last_applied_flip.removed_face_vertices,
        last_applied_flip.inserted_face_vertices,
        last_applied_flip.removed_simplices,
        last_applied_flip.new_simplices,
        last_applied_flip
            .new_simplices
            .contains(&ridge.simplex_key()),
    )
}

/// Formats one simplex's current vertex set so predecessor-flip traces can show
/// the exact simplices that were introduced before a bad ridge appeared.
fn simplex_vertex_summary<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
) -> String
where
    U: DataType,
    V: DataType,
{
    let Some(simplex) = tds.simplex(simplex_key) else {
        return format!("{simplex_key:?}: missing");
    };
    format!("{simplex_key:?}: vertices={:?}", simplex.vertices())
}

/// Captures the first unresolved k=2 postcondition site so #204 debugging can
/// compare the violating facet directly against the last applied repair flip.
fn debug_postcondition_facet_context<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    facet: FacetHandle,
    context: &FlipContext<D, 2>,
    diagnostics: &mut RepairDiagnostics,
    last_applied_flip: Option<&LastAppliedFlip>,
) where
    U: DataType,
    V: DataType,
{
    if !should_emit_postcondition_facet_debug(diagnostics) {
        return;
    }

    let removed_face_details: Vec<_> = context
        .removed_face_vertices
        .iter()
        .filter_map(|&vkey| tds.vertex(vkey).map(|vertex| (vkey, *vertex.point())))
        .collect();
    let inserted_face_details: Vec<_> = context
        .inserted_face_vertices
        .iter()
        .filter_map(|&vkey| tds.vertex(vkey).map(|vertex| (vkey, *vertex.point())))
        .collect();
    let incident_simplex_details: Vec<String> = context
        .removed_simplices
        .iter()
        .copied()
        .map(|simplex_key| {
            facet_incident_simplex_summary(tds, simplex_key, &context.removed_face_vertices)
        })
        .collect();
    let predecessor_summary = last_applied_flip
        .map(|last| postcondition_facet_predecessor_summary(tds, &context.removed_simplices, last));

    tracing::debug!(
        facet = ?facet,
        removed_face = ?removed_face_details,
        inserted_face = ?inserted_face_details,
        incident_simplices = ?context.removed_simplices,
        incident_simplex_details = ?incident_simplex_details,
        predecessor = ?predecessor_summary,
        "repair: postcondition facet debug snapshot"
    );
}

/// Formats the two simplices incident to a violating facet so postcondition traces
/// can see both their full simplex vertices and their opposite vertices.
fn facet_incident_simplex_summary<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
    facet_vertices: &[VertexKey],
) -> String
where
    U: DataType,
    V: DataType,
{
    let Some(simplex) = tds.simplex(simplex_key) else {
        return format!("{simplex_key:?}: missing");
    };

    let opposite_vertices: Vec<VertexKey> = simplex
        .vertices()
        .iter()
        .copied()
        .filter(|vkey| !facet_vertices.contains(vkey))
        .collect();
    let neighbor_snapshot: Option<SmallBuffer<Option<SimplexKey>, MAX_PRACTICAL_DIMENSION_SIZE>> =
        simplex.neighbor_keys().map(Iterator::collect);

    format!(
        "{simplex_key:?}: vertices={:?} opposite_vertices={opposite_vertices:?} neighbors={neighbor_snapshot:?}",
        simplex.vertices()
    )
}

/// Relates the first unresolved postcondition facet to the immediately
/// preceding repair flip so we can tell whether that last move touched the bad
/// local neighborhood or whether the violation was already present.
fn postcondition_facet_predecessor_summary<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    incident_simplices: &[SimplexKey],
    last_applied_flip: &LastAppliedFlip,
) -> String
where
    U: DataType,
    V: DataType,
{
    let incident_simplices_in_new: Vec<SimplexKey> = incident_simplices
        .iter()
        .copied()
        .filter(|simplex_key| last_applied_flip.new_simplices.contains(simplex_key))
        .collect();
    let incident_simplices_in_removed: Vec<SimplexKey> = incident_simplices
        .iter()
        .copied()
        .filter(|simplex_key| last_applied_flip.removed_simplices.contains(simplex_key))
        .collect();
    let predecessor_new_simplex_vertices: Vec<String> = last_applied_flip
        .new_simplices
        .iter()
        .copied()
        .map(|simplex_key| simplex_vertex_summary(tds, simplex_key))
        .collect();
    // Removed simplices are already deleted from the TDS by the time this summary
    // runs, so reach for the pre-flip snapshot in `LastAppliedFlip` to avoid
    // emitting "SimplexKey(N): missing" for every entry.
    let predecessor_removed_simplex_vertices: Vec<String> =
        last_applied_flip.removed_simplex_vertex_lines();

    format!(
        "k={} removed_face={:?} inserted_face={:?} removed_simplices={:?} new_simplices={:?} incident_simplices_in_new={incident_simplices_in_new:?} incident_simplices_in_removed={incident_simplices_in_removed:?} predecessor_new_simplex_vertices={predecessor_new_simplex_vertices:?} predecessor_removed_simplex_vertices={predecessor_removed_simplex_vertices:?}",
        last_applied_flip.kind.k(),
        last_applied_flip.removed_face_vertices,
        last_applied_flip.inserted_face_vertices,
        last_applied_flip.removed_simplices,
        last_applied_flip.new_simplices,
    )
}

/// Check whether a k=3 ridge violates the local Delaunay condition.
///
/// # Errors
///
/// Returns a [`FlipError`] if any referenced simplex/vertex is missing or a predicate
/// evaluation fails.
fn is_delaunay_violation_k3<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    topology_model: &GlobalTopologyModelAdapter<D>,
    context: &FlipContext<D, 3>,
    config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
) -> Result<bool, FlipError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    delaunay_violation_k3_for_ridge(
        tds,
        kernel,
        topology_model,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
        &context.removed_simplices,
        None,
        config,
        diagnostics,
    )
}

/// Apply a generic k-move (no Delaunay check).
///
/// # Errors
///
/// Returns a [`FlipError`] if the flip would be degenerate, duplicate an existing simplex,
/// create non-manifold topology, if the incidence index references a missing simplex,
/// if predicate evaluation fails, or if underlying TDS mutations fail.
pub(crate) fn apply_bistellar_flip<U, V, const D: usize, const K_MOVE: usize>(
    tds: &mut Tds<U, V, D>,
    context: &FlipContext<D, K_MOVE>,
) -> Result<FlipInfo<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    Ok(apply_bistellar_flip_with_k(
        tds,
        K_MOVE,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
        &context.removed_simplices,
        context.direction,
        ReplacementOrientationPolicy::AllowSigned,
        FlipValidationScope::FullTds,
    )?
    .info)
}

/// Apply a generic k-move with runtime k (no Delaunay check).
///
/// # Errors
///
/// Returns a [`FlipError`] if the flip would be degenerate, duplicate an existing simplex,
/// create non-manifold topology, if the incidence index references a missing simplex,
/// if predicate evaluation fails, or if underlying TDS mutations fail.
pub(crate) fn apply_bistellar_flip_dynamic<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    k_move: usize,
    context: &FlipContextDyn<D>,
) -> Result<FlipInfo<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    Ok(apply_bistellar_flip_with_k(
        tds,
        k_move,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
        &context.removed_simplices,
        context.direction,
        ReplacementOrientationPolicy::AllowSigned,
        FlipValidationScope::FullTds,
    )?
    .info)
}

/// Apply a k=2 Delaunay-repair move with positive replacement geometry.
fn apply_delaunay_flip_k2<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    context: &FlipContext<D, 2>,
) -> Result<AppliedFlip<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    apply_bistellar_flip_with_k(
        tds,
        2,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
        &context.removed_simplices,
        context.direction,
        ReplacementOrientationPolicy::RequirePositive,
        FlipValidationScope::LocalCavity,
    )
}

/// Apply a k=2 Delaunay-repair move with reusable rollback storage.
///
/// This is the local-repair hot-path variant of [`apply_delaunay_flip_k2`]; it
/// preserves positive replacement geometry and failure atomicity while avoiding
/// a fresh whole-TDS allocation for each attempted flip.
fn apply_delaunay_flip_k2_in_workspace<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    context: &FlipContext<D, 2>,
    trial_workspace: &mut Tds<U, V, D>,
) -> Result<AppliedFlip<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    apply_bistellar_flip_with_k_in_workspace(
        tds,
        2,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
        &context.removed_simplices,
        context.direction,
        ReplacementOrientationPolicy::RequirePositive,
        FlipValidationScope::LocalCavity,
        trial_workspace,
    )
}

/// Apply a k=3 Delaunay-repair move with reusable rollback storage.
///
/// This preserves positive replacement geometry and failure atomicity while
/// avoiding a fresh whole-TDS allocation for each attempted local-repair flip.
fn apply_delaunay_flip_k3_in_workspace<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    context: &FlipContext<D, 3>,
    trial_workspace: &mut Tds<U, V, D>,
) -> Result<AppliedFlip<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    apply_bistellar_flip_with_k_in_workspace(
        tds,
        3,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
        &context.removed_simplices,
        context.direction,
        ReplacementOrientationPolicy::RequirePositive,
        FlipValidationScope::LocalCavity,
        trial_workspace,
    )
}

/// Apply a dynamic-size Delaunay-repair move with reusable rollback storage.
///
/// This variant is used when the repair search cannot statically name `k`; it
/// still routes through the same validated, failure-atomic bistellar mutation
/// path as the dimension-specific helpers.
fn apply_delaunay_flip_dynamic_in_workspace<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    k_move: usize,
    context: &FlipContextDyn<D>,
    trial_workspace: &mut Tds<U, V, D>,
) -> Result<AppliedFlip<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    apply_bistellar_flip_with_k_in_workspace(
        tds,
        k_move,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
        &context.removed_simplices,
        context.direction,
        ReplacementOrientationPolicy::RequirePositive,
        FlipValidationScope::LocalCavity,
        trial_workspace,
    )
}

/// Direction of a bistellar flip.
///
/// # Examples
///
/// ```rust
/// use delaunay::flips::FlipDirection;
///
/// assert_eq!(FlipDirection::Forward.inverse(), FlipDirection::Inverse);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlipDirection {
    /// Forward (k → D+2−k).
    Forward,
    /// Inverse (D+2−k → k).
    Inverse,
}

impl FlipDirection {
    /// Return the opposite direction.
    #[must_use]
    pub const fn inverse(self) -> Self {
        match self {
            Self::Forward => Self::Inverse,
            Self::Inverse => Self::Forward,
        }
    }
}

/// Stage where debug/test flip validation checked coherent orientation.
///
/// Coherent orientation is a validation-scale TDS invariant. Release-mode flip
/// hot paths rely on explicit validation boundaries rather than scanning the
/// whole TDS before and after every attempted flip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlipOrientationCheckStage {
    /// Before applying the flip to the trial TDS.
    BeforeMutation,
    /// After applying the flip to the trial TDS and before committing it.
    AfterTrialMutation,
}
/// Detect repeated flip signatures and abort on cycles.
#[derive(Debug, Clone, Copy)]
struct FlipCycleContext<'a> {
    signature: FlipSignature,
    kind: BistellarFlipKind,
    direction: FlipDirection,
    removed_face_vertices: &'a [VertexKey],
    inserted_face_vertices: &'a [VertexKey],
}

impl<'a> FlipCycleContext<'a> {
    /// Bundles the flip data needed for diagnostics without cloning vertex
    /// buffers on every repair step.
    const fn from_validated_flip(
        signature: FlipSignature,
        kind: BistellarFlipKind,
        direction: FlipDirection,
        removed_face_vertices: &'a [VertexKey],
        inserted_face_vertices: &'a [VertexKey],
    ) -> Self {
        Self {
            signature,
            kind,
            direction,
            removed_face_vertices,
            inserted_face_vertices,
        }
    }
}

/// Converts repeated flip signatures into typed non-convergence before the
/// repair loop burns the full budget on a short oscillation.
fn check_flip_cycle<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    context: FlipCycleContext<'_>,
    diagnostics: &mut RepairDiagnostics,
    stats: &DelaunayRepairStats,
    max_flips: usize,
    config: &RepairAttemptConfig,
) -> Result<(), DelaunayRepairError>
where
    U: DataType,
    V: DataType,
{
    let repeats = diagnostics
        .flip_signature_counts
        .get(&context.signature)
        .copied()
        .unwrap_or(0);
    if repeats >= MAX_REPEAT_SIGNATURE {
        if repair_trace_enabled() {
            let removed_details: Vec<_> = context
                .removed_face_vertices
                .iter()
                .filter_map(|&vkey| tds.vertex(vkey).map(|v| (vkey, *v.point())))
                .collect();
            let inserted_details: Vec<_> = context
                .inserted_face_vertices
                .iter()
                .filter_map(|&vkey| tds.vertex(vkey).map(|v| (vkey, *v.point())))
                .collect();

            tracing::debug!(
                "[repair] cycle abort signature={} repeats={} flips={} max_flips={} attempt={} order={:?} k={} direction={:?} removed_face={:?} inserted_face={:?}",
                context.signature,
                repeats,
                stats.flips_performed,
                max_flips,
                config.attempt,
                config.queue_order,
                context.kind.k(),
                context.direction,
                removed_details,
                inserted_details,
            );
        }
        diagnostics.record_cycle_abort(context.signature);
        return Err(non_convergent_error(max_flips, stats, diagnostics, config));
    }
    Ok(())
}

/// Resolve a possibly stale facet handle by matching its stable facet key.
///
/// Slot swaps can invalidate the original facet index while preserving the facet
/// vertex set (and therefore its hash key). This helper checks the original
/// index first, then scans the owning simplex to recover the correct index for `key`.
fn resolve_facet_handle_for_key<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    handle: FacetHandle,
    key: u64,
) -> Option<FacetHandle>
where
    U: DataType,
    V: DataType,
{
    let simplex_key = handle.simplex_key();
    let simplex = tds.simplex(simplex_key)?;

    let facet_index = usize::from(handle.facet_index());
    if facet_index < simplex.number_of_vertices() {
        let facet_vertices = facet_vertices_from_simplex(simplex, facet_index);
        if facet_key_from_vertices(&facet_vertices) == key {
            return Some(handle);
        }
    }

    for candidate_idx in 0..simplex.number_of_vertices() {
        let facet_vertices = facet_vertices_from_simplex(simplex, candidate_idx);
        if facet_key_from_vertices(&facet_vertices) == key {
            let facet_index = u8::try_from(candidate_idx).ok()?;
            return Some(FacetHandle::from_validated(simplex_key, facet_index));
        }
    }

    None
}

/// Resolve a possibly stale ridge handle by matching its stable ridge key.
///
/// Slot swaps can invalidate the original omit-index pair while preserving the
/// ridge vertex set (and therefore its hash key). This helper checks the original
/// pair first, then scans the owning simplex for the pair matching `key`.
fn resolve_ridge_handle_for_key<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    handle: RidgeHandle,
    key: u64,
) -> Option<RidgeHandle>
where
    U: DataType,
    V: DataType,
{
    if D < 3 {
        return None;
    }

    let simplex_key = handle.simplex_key();
    let simplex = tds.simplex(simplex_key)?;
    let vertex_count = simplex.number_of_vertices();

    let omit_a = usize::from(handle.omit_a());
    let omit_b = usize::from(handle.omit_b());
    if omit_a < vertex_count && omit_b < vertex_count && omit_a != omit_b {
        let ridge_vertices = ridge_vertices_from_simplex(simplex, omit_a, omit_b);
        if ridge_vertices.len() == D - 1 && facet_key_from_vertices(&ridge_vertices) == key {
            return Some(handle);
        }
    }

    for i in 0..vertex_count {
        for j in (i + 1)..vertex_count {
            let ridge_vertices = ridge_vertices_from_simplex(simplex, i, j);
            if ridge_vertices.len() != D - 1 {
                continue;
            }
            if facet_key_from_vertices(&ridge_vertices) == key {
                let omit_a = u8::try_from(i).ok()?;
                let omit_b = u8::try_from(j).ok()?;
                return Some(RidgeHandle::from_validated(simplex_key, omit_a, omit_b));
            }
        }
    }

    None
}

impl BistellarFlipKind {
    /// Number of simplices being replaced on the current side (k).
    #[must_use]
    pub const fn k(&self) -> usize {
        self.k
    }
    /// Construct a k=1 flip kind for the given dimension.
    #[must_use]
    pub const fn k1(d: usize) -> Self {
        Self { k: 1, d }
    }
    /// Construct a k=2 flip kind for the given dimension.
    #[must_use]
    pub const fn k2(d: usize) -> Self {
        Self { k: 2, d }
    }

    /// Construct a k=3 flip kind for the given dimension.
    #[must_use]
    pub const fn k3(d: usize) -> Self {
        Self { k: 3, d }
    }

    /// Construct the inverse flip kind (k' = D + 2 - k).
    #[must_use]
    pub const fn inverse(self) -> Self {
        Self {
            k: self.d + 2 - self.k,
            d: self.d,
        }
    }
}

/// Const-generic move marker for Pachner k-moves.
///
/// # Examples
///
/// ```rust
/// use delaunay::flips::{BistellarMove, ConstK};
///
/// fn move_k<const D: usize, M: BistellarMove<D>>() -> usize {
///     M::K
/// }
///
/// assert_eq!(move_k::<3, ConstK<2>>(), 2);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ConstK<const K: usize>;

/// Const-generic descriptor for a Pachner move in dimension `D`.
///
/// # Examples
///
/// ```rust
/// use delaunay::flips::{BistellarMove, ConstK};
///
/// fn move_k<const D: usize, M: BistellarMove<D>>() -> usize {
///     M::K
/// }
///
/// assert_eq!(move_k::<4, ConstK<3>>(), 3);
/// ```
pub trait BistellarMove<const D: usize> {
    /// Number of removed D-simplices (k).
    const K: usize;
}

impl<const D: usize, const K: usize> BistellarMove<D> for ConstK<K> {
    const K: usize = K;
}

/// Predicate operation being evaluated by flip logic.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlipPredicateOperation {
    /// Replacement-simplex orientation check while applying a flip.
    #[error("replacement-simplex orientation")]
    ReplacementSimplexOrientation,
    /// Replacement-simplex orientation postcondition during Delaunay repair.
    #[error("Delaunay-repair replacement-simplex orientation")]
    DelaunayRepairReplacementOrientation,
    /// Degenerate-simplex precheck before applying a flip.
    #[error("degenerate-simplex precheck")]
    DegenerateSimplexPrecheck,
    /// First k=2 insphere predicate.
    #[error("k=2 simplex-A insphere")]
    K2SimplexAInSphere,
    /// Second k=2 insphere predicate.
    #[error("k=2 simplex-B insphere")]
    K2SimplexBInSphere,
    /// k=3 insphere predicate.
    #[error("k=3 simplex insphere")]
    K3SimplexInSphere,
}

/// Structured reason a geometric predicate failed during a flip.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum FlipPredicateError {
    /// A coordinate-conversion or exact-predicate helper failed.
    #[error("{operation} predicate failed: {source}")]
    CoordinateConversion {
        /// Predicate operation being evaluated.
        operation: FlipPredicateOperation,
        /// Underlying coordinate conversion failure.
        #[source]
        source: CoordinateConversionError,
    },
    /// A topology model failed to lift a periodic vertex for predicate evaluation.
    #[error("failed to lift vertex {vertex_key:?} for periodic predicate: {details}")]
    PeriodicVertexLift {
        /// Vertex being lifted.
        vertex_key: VertexKey,
        /// Underlying topology-model error, captured in display form because it
        /// may contain floating-point values and therefore is not `Eq`.
        details: String,
    },
}

impl FlipPredicateError {
    const fn coordinate_conversion(
        operation: FlipPredicateOperation,
        source: CoordinateConversionError,
    ) -> Self {
        Self::CoordinateConversion { operation, source }
    }
}

/// Structured reason a flip context is invalid before mutation.
///
/// These reasons are wrapped by [`FlipError::InvalidFlipContext`] so callers can
/// distinguish shape errors, replacement-orientation conflicts, and periodic
/// frame-alignment failures before any TDS mutation is committed.
///
/// # Examples
///
/// ```rust
/// use delaunay::flips::{FlipContextError, FlipError};
///
/// let reason = FlipContextError::ReplacementPeriodicOffsetCountMismatch {
///     simplex_count: 2,
///     offset_count: 1,
/// };
/// let err: FlipError = reason.into();
/// std::assert_matches!(err, FlipError::InvalidFlipContext { .. });
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlipContextError {
    /// The requested move size is outside `1..=D+1`.
    #[error("k must be in 1..=D+1 (k={k_move}, D={dimension})")]
    InvalidMoveSize {
        /// Requested k-move.
        k_move: usize,
        /// Triangulation dimension.
        dimension: usize,
    },
    /// Removed face has the wrong arity for the k-move.
    #[error("removed-face must have {expected} vertices, got {found}")]
    WrongRemovedFaceArity {
        /// Expected removed-face vertex count.
        expected: usize,
        /// Observed removed-face vertex count.
        found: usize,
    },
    /// Inserted face has the wrong arity for the k-move.
    #[error("k={k_move} inserted-face must have {expected} vertices, got {found}")]
    WrongInsertedFaceArity {
        /// Requested k-move.
        k_move: usize,
        /// Expected inserted-face vertex count.
        expected: usize,
        /// Observed inserted-face vertex count.
        found: usize,
    },
    /// The number of simplices selected for removal does not match the k-move.
    #[error("removed_simplices must have {expected} entries, got {found}")]
    WrongRemovedSimplexCount {
        /// Expected number of removed simplices.
        expected: usize,
        /// Observed number of removed simplices.
        found: usize,
    },
    /// Removed and inserted faces are not disjoint.
    #[error("removed-face and inserted-face must be disjoint")]
    OverlappingFaces,
    /// The TDS coherent-orientation invariant failed during debug/test flip validation.
    ///
    /// This diagnostic is reserved for validation builds because the check scans
    /// global TDS orientation state. Production callers should use explicit TDS
    /// or triangulation validation when they need this invariant checked.
    #[error(
        "TDS coherent orientation invariant violated during {stage:?} for k={k_move}, direction={direction:?}"
    )]
    CoherentOrientationViolation {
        /// Stage where the invariant was checked.
        stage: FlipOrientationCheckStage,
        /// k for the attempted move.
        k_move: usize,
        /// Direction of the attempted move.
        direction: FlipDirection,
    },
    /// Replacement-simplex offset sidecar length does not match the replacement simplices.
    #[error(
        "replacement periodic offset count {offset_count} does not match replacement simplex count {simplex_count}"
    )]
    ReplacementPeriodicOffsetCountMismatch {
        /// Number of replacement simplices.
        simplex_count: usize,
        /// Number of periodic-offset entries.
        offset_count: usize,
    },
    /// A periodic parity constraint referenced a replacement simplex without offsets.
    #[error(
        "replacement simplex {simplex_index} is missing periodic offsets for periodic facet parity"
    )]
    MissingReplacementPeriodicOffsets {
        /// Local replacement-simplex index.
        simplex_index: usize,
    },
    /// Replacement-simplex periodic offsets are not aligned with its vertex slots.
    #[error(
        "replacement simplex {simplex_index} periodic offset count {offset_count} does not match vertex count {vertex_count}"
    )]
    ReplacementPeriodicOffsetLengthMismatch {
        /// Local replacement-simplex index.
        simplex_index: usize,
        /// Number of periodic offsets.
        offset_count: usize,
        /// Number of replacement-simplex vertices.
        vertex_count: usize,
    },
    /// Replacement-simplex orientation constraints disagree.
    #[error(
        "conflicting replacement-simplex orientation constraints between local simplices {source_simplex_index} and {target_simplex_index}"
    )]
    ConflictingReplacementOrientationBetweenSimplices {
        /// First local replacement-simplex index.
        source_simplex_index: usize,
        /// Second local replacement-simplex index.
        target_simplex_index: usize,
    },
    /// Replacement-simplex orientation cannot be flipped because the simplex is too small.
    #[error("replacement simplex needs at least two vertices to flip orientation")]
    ReplacementSimplexTooSmallForOrientationFlip,
    /// Replacement orientation assignment referenced a missing local simplex.
    #[error("replacement orientation index {simplex_index} out of range")]
    ReplacementOrientationIndexOutOfRange {
        /// Local replacement-simplex index.
        simplex_index: usize,
    },
    /// Two parity constraints disagree for the same replacement simplex.
    #[error(
        "conflicting replacement-simplex orientation constraints for local simplex {simplex_index}"
    )]
    ConflictingReplacementOrientationForSimplex {
        /// Local replacement-simplex index.
        simplex_index: usize,
    },
    /// The facet-order permutation parity could not be derived.
    #[error("could not derive replacement facet-order permutation parity")]
    FacetOrderParityUnavailable,
    /// A facet index is outside the replacement simplex's vertex range.
    #[error(
        "facet index {facet_index} out of range for replacement simplex with {vertex_count} vertices"
    )]
    ReplacementFacetIndexOutOfRange {
        /// Invalid facet index.
        facet_index: usize,
        /// Replacement-simplex vertex count.
        vertex_count: usize,
    },
    /// A k=2 facet predicate received the wrong number of facet vertices.
    #[error("k=2 facet must have {expected} vertices, got {found}")]
    K2FacetArity {
        /// Expected facet vertex count.
        expected: usize,
        /// Observed facet vertex count.
        found: usize,
    },
    /// k=2 opposite vertices are not a valid complementary face.
    #[error("k=2 opposites must be distinct and not in the facet")]
    InvalidK2Opposites,
    /// A k=3 predicate received the wrong number of ridge vertices.
    #[error("k=3 ridge must have {expected} vertices, got {found}")]
    K3RidgeArity {
        /// Expected ridge vertex count.
        expected: usize,
        /// Observed ridge vertex count.
        found: usize,
    },
    /// Periodic frame alignment found contradictory translations.
    #[error(
        "conflicting periodic frame translations while aligning vertex {vertex_key:?} from simplex {source_simplex_key:?} into frame {target_simplex_key:?}: expected {expected_offset:?}, got {found_offset:?}"
    )]
    ConflictingPeriodicFrameTranslation {
        /// Vertex being aligned.
        vertex_key: VertexKey,
        /// Source simplex used for alignment.
        source_simplex_key: SimplexKey,
        /// Target simplex frame.
        target_simplex_key: SimplexKey,
        /// Previously derived offset.
        expected_offset: Vec<i8>,
        /// Conflicting candidate offset.
        found_offset: Vec<i8>,
    },
    /// Periodic frame alignment disagreed for an external-to-replacement facet.
    #[error(
        "conflicting periodic frame translations while aligning vertex {vertex_key:?} from external simplex {source_simplex_key:?} into replacement simplex {target_simplex_index}: expected {expected_offset:?}, got {found_offset:?}"
    )]
    ConflictingReplacementPeriodicFrameTranslation {
        /// Vertex being aligned.
        vertex_key: VertexKey,
        /// External source simplex used for alignment.
        source_simplex_key: SimplexKey,
        /// Target replacement-simplex index.
        target_simplex_index: usize,
        /// Previously derived offset.
        expected_offset: Vec<i8>,
        /// Conflicting candidate offset.
        found_offset: Vec<i8>,
    },
    /// No source simplex could align a periodic vertex into the target frame.
    #[error("cannot align periodic vertex {vertex_key:?} into frame {target_simplex_key:?}")]
    PeriodicVertexAlignmentFailed {
        /// Vertex being aligned.
        vertex_key: VertexKey,
        /// Target simplex frame.
        target_simplex_key: SimplexKey,
    },
    /// Periodic offset count does not match the simplex's vertex count.
    #[error(
        "simplex {simplex_key:?} periodic offset count {offset_count} does not match vertex count {vertex_count}"
    )]
    PeriodicOffsetCountMismatch {
        /// Simplex with malformed offsets.
        simplex_key: SimplexKey,
        /// Number of stored offsets.
        offset_count: usize,
        /// Number of simplex vertices.
        vertex_count: usize,
    },
    /// Periodic offset subtraction overflowed on an axis.
    #[error("periodic offset subtraction overflow on axis {axis}")]
    PeriodicOffsetSubtractionOverflow {
        /// Coordinate axis.
        axis: usize,
    },
    /// Periodic offset addition overflowed on an axis.
    #[error("periodic offset addition overflow on axis {axis}")]
    PeriodicOffsetAdditionOverflow {
        /// Coordinate axis.
        axis: usize,
    },
    /// Inverse predicate evaluation had no removed-simplex frame.
    #[error("inverse flip predicate requires at least one removed simplex frame")]
    MissingRemovedSimplexFrame,
}

/// Non-recursive summary of a flip error that reached another flip error path.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlipFailureKind {
    /// Flips are not supported for this dimension.
    #[error("unsupported dimension")]
    UnsupportedDimension,
    /// Boundary facet.
    #[error("boundary facet")]
    BoundaryFacet,
    /// Missing simplex.
    #[error("missing simplex")]
    MissingSimplex,
    /// Dangling vertex-to-simplex incidence reference.
    #[error("dangling vertex incidence")]
    DanglingVertexIncidence,
    /// Missing vertex.
    #[error("missing vertex")]
    MissingVertex,
    /// Missing neighbor.
    #[error("missing neighbor")]
    MissingNeighbor,
    /// Dangling ridge-neighbor reference.
    #[error("dangling ridge neighbor")]
    DanglingRidgeNeighbor,
    /// Invalid facet adjacency.
    #[error("invalid facet adjacency")]
    InvalidFacetAdjacency,
    /// Invalid facet index.
    #[error("invalid facet index")]
    InvalidFacetIndex,
    /// Invalid ridge index.
    #[error("invalid ridge index")]
    InvalidRidgeIndex,
    /// Invalid ridge adjacency.
    #[error("invalid ridge adjacency")]
    InvalidRidgeAdjacency,
    /// Invalid ridge multiplicity.
    #[error("invalid ridge multiplicity")]
    InvalidRidgeMultiplicity,
    /// Invalid edge multiplicity.
    #[error("invalid edge multiplicity")]
    InvalidEdgeMultiplicity,
    /// Invalid triangle multiplicity.
    #[error("invalid triangle multiplicity")]
    InvalidTriangleMultiplicity,
    /// Invalid edge adjacency.
    #[error("invalid edge adjacency")]
    InvalidEdgeAdjacency,
    /// Invalid triangle adjacency.
    #[error("invalid triangle adjacency")]
    InvalidTriangleAdjacency,
    /// Invalid vertex multiplicity.
    #[error("invalid vertex multiplicity")]
    InvalidVertexMultiplicity,
    /// Invalid vertex adjacency.
    #[error("invalid vertex adjacency")]
    InvalidVertexAdjacency,
    /// Invalid flip context.
    #[error("invalid flip context")]
    InvalidFlipContext,
    /// Predicate failure.
    #[error("predicate failure")]
    PredicateFailure,
    /// Degenerate simplex.
    #[error("degenerate simplex")]
    DegenerateSimplex,
    /// Negative orientation.
    #[error("negative orientation")]
    NegativeOrientation,
    /// Duplicate simplex.
    #[error("duplicate simplex")]
    DuplicateSimplex,
    /// Non-manifold facet.
    #[error("non-manifold facet")]
    NonManifoldFacet,
    /// Inserted simplex already exists.
    #[error("inserted simplex already exists")]
    InsertedSimplexAlreadyExists,
    /// Facet iteration failed.
    #[error("facet iteration")]
    FacetIteration,
    /// Simplex creation failed.
    #[error("simplex creation")]
    SimplexCreation,
    /// Neighbor wiring failed.
    #[error("neighbor wiring")]
    NeighborWiring,
    /// Trial TDS validation failed before committing a flip.
    #[error("trial validation")]
    TrialValidation,
    /// Neighbor wiring reached a validation failure.
    #[error("wiring validation")]
    WiringValidation,
    /// Neighbor wiring reached a Delaunay repair failure.
    #[error("Delaunay repair failed")]
    DelaunayRepairFailed,
    /// TDS mutation failed.
    #[error("TDS mutation")]
    TdsMutation,
}

/// Non-recursive summary of a cavity-filling error at the flip wiring boundary.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlipNeighborCavityFailureKind {
    /// Boundary simplex was missing.
    #[error("missing boundary simplex")]
    MissingBoundarySimplex,
    /// Inserted vertex was missing.
    #[error("missing inserted vertex")]
    MissingInsertedVertex,
    /// Boundary simplex had the wrong arity.
    #[error("wrong simplex arity")]
    WrongSimplexArity,
    /// Facet index was invalid.
    #[error("invalid facet index")]
    InvalidFacetIndex,
    /// Replacement simplex creation failed.
    #[error("simplex creation")]
    SimplexCreation,
    /// Replacement simplex insertion failed.
    #[error("simplex insertion")]
    SimplexInsertion,
    /// Initial simplex construction failed.
    #[error("initial simplex construction")]
    InitialSimplexConstruction,
    /// Rebuilt TDS lost the inserted vertex.
    #[error("rebuilt vertex missing")]
    RebuiltVertexMissing,
    /// Conflict region was empty.
    #[error("empty conflict region")]
    EmptyConflictRegion,
    /// Cavity boundary was empty.
    #[error("empty boundary")]
    EmptyBoundary,
    /// Facet sharing remained invalid after repair.
    #[error("invalid facet sharing after repair")]
    InvalidFacetSharingAfterRepair,
    /// Cavity filling created the wrong number of replacement simplices.
    #[error("boundary simplex count mismatch")]
    BoundarySimplexCountMismatch,
    /// Neighbor rebuild failed.
    #[error("neighbor rebuild")]
    NeighborRebuild,
    /// Perturbation scale conversion failed.
    #[error("perturbation scale conversion")]
    PerturbationScaleConversion,
    /// Degenerate insertion location is unsupported.
    #[error("unsupported degenerate location")]
    UnsupportedDegenerateLocation,
    /// Fan filling produced no simplices.
    #[error("empty fan triangulation")]
    EmptyFanTriangulation,
}

impl From<&CavityFillingError> for FlipNeighborCavityFailureKind {
    fn from(source: &CavityFillingError) -> Self {
        match source {
            CavityFillingError::MissingBoundarySimplex { .. } => Self::MissingBoundarySimplex,
            CavityFillingError::MissingInsertedVertex { .. } => Self::MissingInsertedVertex,
            CavityFillingError::WrongSimplexArity { .. } => Self::WrongSimplexArity,
            CavityFillingError::InvalidFacetIndex { .. } => Self::InvalidFacetIndex,
            CavityFillingError::SimplexCreation { .. } => Self::SimplexCreation,
            CavityFillingError::SimplexInsertion { .. } => Self::SimplexInsertion,
            CavityFillingError::InitialSimplexConstruction { .. } => {
                Self::InitialSimplexConstruction
            }
            CavityFillingError::RebuiltVertexMissing { .. } => Self::RebuiltVertexMissing,
            CavityFillingError::EmptyConflictRegion { .. } => Self::EmptyConflictRegion,
            CavityFillingError::EmptyBoundary { .. } => Self::EmptyBoundary,
            CavityFillingError::InvalidFacetSharingAfterRepair { .. } => {
                Self::InvalidFacetSharingAfterRepair
            }
            CavityFillingError::BoundarySimplexCountMismatch { .. } => {
                Self::BoundarySimplexCountMismatch
            }
            CavityFillingError::NeighborRebuild { .. } => Self::NeighborRebuild,
            CavityFillingError::PerturbationScaleConversion { .. } => {
                Self::PerturbationScaleConversion
            }
            CavityFillingError::UnsupportedDegenerateLocation { .. } => {
                Self::UnsupportedDegenerateLocation
            }
            CavityFillingError::EmptyFanTriangulation => Self::EmptyFanTriangulation,
        }
    }
}

impl From<CavityFillingError> for FlipNeighborCavityFailureKind {
    fn from(source: CavityFillingError) -> Self {
        Self::from(&source)
    }
}

/// Non-recursive summary of a hull-extension error at the flip wiring boundary.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlipNeighborHullExtensionFailureKind {
    /// No visible facets were found.
    #[error("no visible facets")]
    NoVisibleFacets,
    /// Boundary-edge split matched the wrong number of facets.
    #[error("boundary edge split facet count")]
    BoundaryEdgeSplitFacetCount,
    /// Boundary-edge split matched more than one candidate facet.
    #[error("multiple boundary edge split facets")]
    MultipleBoundaryEdgeSplitFacets,
    /// Visible facets formed a disconnected or non-manifold patch.
    #[error("disconnected visible patch")]
    DisconnectedVisiblePatch,
    /// Geometric predicate failed.
    #[error("predicate failed")]
    PredicateFailed,
    /// Lower-layer TDS error.
    #[error("TDS")]
    Tds,
}

impl From<&HullExtensionReason> for FlipNeighborHullExtensionFailureKind {
    fn from(source: &HullExtensionReason) -> Self {
        match source {
            HullExtensionReason::NoVisibleFacets => Self::NoVisibleFacets,
            HullExtensionReason::BoundaryEdgeSplitFacetCount { .. } => {
                Self::BoundaryEdgeSplitFacetCount
            }
            HullExtensionReason::MultipleBoundaryEdgeSplitFacets => {
                Self::MultipleBoundaryEdgeSplitFacets
            }
            HullExtensionReason::DisconnectedVisiblePatch { .. } => Self::DisconnectedVisiblePatch,
            HullExtensionReason::PredicateFailed(_) => Self::PredicateFailed,
            HullExtensionReason::Tds(_) => Self::Tds,
        }
    }
}

impl From<HullExtensionReason> for FlipNeighborHullExtensionFailureKind {
    fn from(source: HullExtensionReason) -> Self {
        Self::from(&source)
    }
}

/// Non-recursive summary of a Delaunay validation error at the flip wiring boundary.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlipNeighborDelaunayValidationFailureKind {
    /// Lower-layer TDS validation failed.
    #[error("TDS")]
    Tds,
    /// Lower-layer topology validation failed.
    #[error("triangulation")]
    Triangulation,
    /// Embedded-geometry validation failed.
    #[error("embedding")]
    Embedding,
    /// Delaunay verification failed.
    #[error("verification failed")]
    VerificationFailed,
    /// Repair operation validation failed.
    #[error("repair operation failed")]
    RepairOperationFailed,
}

impl From<&DelaunayTriangulationValidationError> for FlipNeighborDelaunayValidationFailureKind {
    fn from(source: &DelaunayTriangulationValidationError) -> Self {
        match source {
            DelaunayTriangulationValidationError::Tds(_) => Self::Tds,
            DelaunayTriangulationValidationError::Triangulation(_) => Self::Triangulation,
            DelaunayTriangulationValidationError::Embedding(_) => Self::Embedding,
            DelaunayTriangulationValidationError::VerificationFailed { .. } => {
                Self::VerificationFailed
            }
            DelaunayTriangulationValidationError::RepairOperationFailed { .. } => {
                Self::RepairOperationFailed
            }
        }
    }
}

impl From<DelaunayTriangulationValidationError> for FlipNeighborDelaunayValidationFailureKind {
    fn from(source: DelaunayTriangulationValidationError) -> Self {
        Self::from(&source)
    }
}

/// Compact repair diagnostics preserved when embedding repair failures in flip wiring errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlipNeighborRepairDiagnostics {
    /// Number of queued items checked.
    pub facets_checked: usize,
    /// Number of flips performed.
    pub flips_performed: usize,
    /// Maximum queue length observed.
    pub max_queue_len: usize,
    /// Count of ambiguous predicate evaluations.
    pub ambiguous_predicates: usize,
    /// Count of predicate failures.
    pub predicate_failures: usize,
    /// Count of detected flip cycles.
    pub cycle_detections: usize,
    /// Attempt number.
    pub attempt: usize,
    /// Queue ordering policy used for this attempt.
    pub queue_order: RepairQueueOrder,
}

impl fmt::Display for FlipNeighborRepairDiagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "checked={}, flips={}, max_queue={}, ambiguous={}, predicate_failures={}, cycles={}, attempt={}, order={:?}",
            self.facets_checked,
            self.flips_performed,
            self.max_queue_len,
            self.ambiguous_predicates,
            self.predicate_failures,
            self.cycle_detections,
            self.attempt,
            self.queue_order
        )
    }
}

impl From<DelaunayRepairDiagnostics> for FlipNeighborRepairDiagnostics {
    fn from(source: DelaunayRepairDiagnostics) -> Self {
        Self {
            facets_checked: source.facets_checked,
            flips_performed: source.flips_performed,
            max_queue_len: source.max_queue_len,
            ambiguous_predicates: source.ambiguous_predicates,
            predicate_failures: source.predicate_failures,
            cycle_detections: source.cycle_detections,
            attempt: source.attempt,
            queue_order: source.queue_order,
        }
    }
}

/// Non-recursive reason Delaunay repair reached flip neighbor wiring.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlipNeighborRepairFailure {
    /// Repair did not converge within the flip budget.
    #[error("repair did not converge after {max_flips} flips ({diagnostics})")]
    NonConvergent {
        /// Maximum flips allowed.
        max_flips: usize,
        /// Diagnostics captured during the failed attempt.
        diagnostics: FlipNeighborRepairDiagnostics,
    },
    /// Repair completed but left a Delaunay violation.
    #[error("repair postcondition failed: {reason}")]
    PostconditionFailed {
        /// Structured postcondition failure reason.
        #[source]
        reason: DelaunayRepairPostconditionFailure,
    },
    /// Post-repair verification could not evaluate a local flip predicate.
    #[error("repair verification failed during {context}: {source_kind}")]
    VerificationFailed {
        /// Verification phase that failed.
        context: DelaunayRepairVerificationContext,
        /// Non-recursive class of the underlying flip error.
        source_kind: FlipFailureKind,
    },
    /// Repair completed but orientation canonicalization failed.
    #[error("repair orientation canonicalization failed: {reason}")]
    OrientationCanonicalizationFailed {
        /// Structured canonicalization failure reason.
        reason: DelaunayRepairOrientationCanonicalizationFailureKind,
    },
    /// Flip-based repair is not admissible under the current topology guarantee.
    #[error("repair requires {required:?} topology, found {found:?}: {message}")]
    InvalidTopology {
        /// Required topology guarantee.
        required: TopologyGuarantee,
        /// Actual topology guarantee.
        found: TopologyGuarantee,
        /// Additional context for the mismatch.
        message: &'static str,
    },
    /// Heuristic rebuild failed during advanced repair.
    #[error("heuristic rebuild failed: {reason}")]
    HeuristicRebuildFailed {
        /// Structured rebuild failure category.
        reason: DelaunayRepairHeuristicRebuildFailureKind,
    },
    /// Underlying flip error.
    #[error("flip error: {source_kind}")]
    Flip {
        /// Non-recursive class of the underlying flip error.
        source_kind: FlipFailureKind,
    },
}

/// Structured reason neighbor wiring failed during flip application.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum FlipNeighborWiringError {
    /// Boundary extraction failed before replacement simplices were created.
    #[error("flip boundary extraction failed: {source}")]
    BoundaryExtraction {
        /// Underlying conflict-region error.
        #[source]
        source: ConflictError,
    },
    /// Neighbor wiring failed with a structured insertion-layer reason.
    #[error("neighbor wiring failed: {source}")]
    NeighborWiring {
        /// Underlying neighbor-wiring error.
        #[source]
        source: NeighborWiringError,
    },
    /// The replacement simplices would create a non-manifold facet.
    #[error("non-manifold topology: facet {facet_hash:#x} shared by {simplex_count} simplices")]
    NonManifoldTopology {
        /// Over-shared facet hash.
        facet_hash: u64,
        /// Number of incident simplices.
        simplex_count: usize,
    },
    /// TDS topology validation failed while wiring neighbors.
    #[error("topology validation failed during neighbor wiring: {source}")]
    TopologyValidation {
        /// Underlying TDS validation error.
        #[source]
        source: TdsValidationFailure,
    },
    /// Conflict-region extraction reached flip neighbor wiring.
    #[error("conflict-region error reached flip neighbor wiring: {source}")]
    ConflictRegion {
        /// Underlying conflict-region error.
        #[source]
        source: ConflictError,
    },
    /// Point-location failure reached flip neighbor wiring.
    #[error("point-location error reached flip neighbor wiring: {source}")]
    Location {
        /// Underlying point-location error.
        #[source]
        source: LocateError,
    },
    /// Cavity filling failed while preparing flip neighbor wiring.
    #[error("cavity filling error reached flip neighbor wiring: {reason}")]
    CavityFilling {
        /// Structured cavity-filling reason.
        reason: FlipNeighborCavityFailureKind,
    },
    /// Hull extension failed while preparing flip neighbor wiring.
    #[error("hull extension error reached flip neighbor wiring: {reason}")]
    HullExtension {
        /// Structured hull-extension reason.
        reason: FlipNeighborHullExtensionFailureKind,
    },
    /// Delaunay validation failed while preparing flip neighbor wiring.
    #[error("Delaunay validation error reached flip neighbor wiring: {reason}")]
    DelaunayValidation {
        /// Structured validation reason.
        reason: FlipNeighborDelaunayValidationFailureKind,
    },
    /// Embedding validation failed while preparing flip neighbor wiring.
    #[error("embedding validation error reached flip neighbor wiring: {reason:?}")]
    EmbeddingValidation {
        /// Structured embedding-validation reason.
        reason: TriangulationEmbeddingValidationErrorKind,
    },
    /// Delaunay repair failed while preparing flip neighbor wiring.
    #[error("Delaunay repair error reached flip neighbor wiring: {reason}")]
    DelaunayRepair {
        /// Structured non-recursive repair reason.
        #[source]
        reason: FlipNeighborRepairFailure,
    },
    /// Duplicate coordinates reached flip neighbor wiring.
    #[error("duplicate coordinates reached flip neighbor wiring: {coordinates}")]
    DuplicateCoordinates {
        /// Duplicate coordinate tuple stored as typed coordinate payloads.
        coordinates: CoordinateValues,
    },
    /// Duplicate UUID reached flip neighbor wiring.
    #[error("duplicate UUID reached flip neighbor wiring: {entity:?} {uuid}")]
    DuplicateUuid {
        /// Entity kind.
        entity: EntityKind,
        /// Duplicated UUID.
        uuid: uuid::Uuid,
    },
    /// Level 3 topology validation failed while preparing flip neighbor wiring.
    #[error("topology validation error reached flip neighbor wiring: {context}: {source}")]
    TopologyValidationFailed {
        /// High-level insertion context.
        context: InsertionTopologyValidationContext,
        /// Underlying topology validation error.
        #[source]
        source: TriangulationValidationError,
    },
    /// Local repair would exceed its simplex-removal budget.
    #[error(
        "local repair removal budget reached flip neighbor wiring: attempted {attempted}, max {max_simplices_removed}"
    )]
    MaxSimplicesRemovedExceeded {
        /// Maximum simplices allowed for removal.
        max_simplices_removed: usize,
        /// Number of simplices selected for removal.
        attempted: usize,
    },
    /// Spatial index construction failed before insertion.
    #[error("spatial index construction reached flip neighbor wiring: {reason}")]
    SpatialIndexConstruction {
        /// Structured spatial-index construction failure.
        #[source]
        reason: SpatialIndexConstructionFailure,
    },
    /// Perturbation retry produced invalid coordinates.
    #[error(
        "perturbation retry produced invalid coordinates before flip neighbor wiring: {source}"
    )]
    PerturbedCoordinateInvalid {
        /// Structured coordinate validation failure for the perturbed point.
        #[source]
        source: CoordinateValidationError,
    },
}

impl From<InsertionError> for FlipNeighborWiringError {
    fn from(source: InsertionError) -> Self {
        match source {
            InsertionError::NeighborWiring { reason } => Self::NeighborWiring { source: reason },
            InsertionError::NonManifoldTopology {
                facet_hash,
                simplex_count,
            } => Self::NonManifoldTopology {
                facet_hash,
                simplex_count,
            },
            InsertionError::TopologyValidation(source) => Self::TopologyValidation {
                source: source.into(),
            },
            InsertionError::ConflictRegion(source) => Self::ConflictRegion { source },
            InsertionError::Location(source) => Self::Location { source },
            InsertionError::CavityFilling { reason } => Self::CavityFilling {
                reason: reason.into(),
            },
            InsertionError::HullExtension { reason } => Self::HullExtension {
                reason: reason.into(),
            },
            InsertionError::DelaunayValidationFailed { source } => Self::DelaunayValidation {
                reason: source.into(),
            },
            InsertionError::EmbeddingValidationFailed { source } => Self::EmbeddingValidation {
                reason: TriangulationEmbeddingValidationErrorKind::from(&source),
            },
            InsertionError::DelaunayRepairFailed { source, context: _ } => Self::DelaunayRepair {
                reason: FlipNeighborRepairFailure::from(*source),
            },
            InsertionError::DuplicateCoordinates { coordinates } => {
                Self::DuplicateCoordinates { coordinates }
            }
            InsertionError::DuplicateUuid { entity, uuid } => Self::DuplicateUuid { entity, uuid },
            InsertionError::TopologyValidationFailed { context, source } => {
                Self::TopologyValidationFailed { context, source }
            }
            InsertionError::MaxSimplicesRemovedExceeded {
                max_simplices_removed,
                attempted,
            } => Self::MaxSimplicesRemovedExceeded {
                max_simplices_removed,
                attempted,
            },
            InsertionError::SpatialIndexConstruction { reason } => {
                Self::SpatialIndexConstruction { reason }
            }
            InsertionError::PerturbedCoordinateInvalid { source } => {
                Self::PerturbedCoordinateInvalid { source }
            }
        }
    }
}

/// Structured reason a TDS mutation failed while applying a flip.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum FlipMutationError {
    /// Vertex insertion failed before a k=1 flip.
    #[error("vertex insertion failed: {source}")]
    VertexInsertion {
        /// Underlying TDS construction error.
        #[source]
        source: TdsConstructionFailure,
    },
    /// Replacement-simplex insertion failed.
    #[error("simplex insertion failed: {source}")]
    SimplexInsertion {
        /// Underlying TDS construction error.
        #[source]
        source: TdsConstructionFailure,
    },
    /// Removed-simplex deletion failed.
    #[error("simplex removal failed: {source}")]
    SimplexRemoval {
        /// Underlying TDS mutation error.
        #[source]
        source: TdsMutationError,
    },
    /// Trial TDS validation failed before committing a flip.
    #[error(
        "trial TDS validation failed after bistellar flip (k={k_move}, direction={direction:?}): {source}"
    )]
    TrialValidation {
        /// k for the attempted move.
        k_move: usize,
        /// Direction of the attempted move.
        direction: FlipDirection,
        /// Underlying TDS validation error.
        #[source]
        source: TdsValidationFailure,
    },
    /// Trial TDS coherent-orientation validation failed before committing a flip.
    ///
    /// This diagnostic is debug/test-only in the flip hot path because it scans
    /// global TDS orientation state. Release-mode callers should use explicit
    /// validation boundaries when they need this invariant checked.
    #[error(
        "trial TDS coherent orientation invariant violated during {stage:?} (k={k_move}, direction={direction:?})"
    )]
    CoherentOrientationViolation {
        /// Stage where the invariant was checked.
        stage: FlipOrientationCheckStage,
        /// k for the attempted move.
        k_move: usize,
        /// Direction of the attempted move.
        direction: FlipDirection,
    },
}

/// Structured reason inverse k=2 edge adjacency is inconsistent.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlipEdgeAdjacencyError {
    /// Edge endpoints are identical.
    #[error("edge endpoints must be distinct ({vertex_key:?})")]
    DuplicateEndpoints {
        /// Repeated endpoint key.
        vertex_key: VertexKey,
    },
    /// Incident simplex does not contain both edge endpoints.
    #[error("simplex {simplex_key:?} does not contain edge vertices {v0:?} and {v1:?}")]
    SimplexMissingEdgeVertices {
        /// Simplex expected to contain the edge.
        simplex_key: SimplexKey,
        /// First edge endpoint.
        v0: VertexKey,
        /// Second edge endpoint.
        v1: VertexKey,
    },
    /// Stored simplex data contains the edge, but the edge is missing from the maintained incidence index.
    #[error("vertex incidence index does not list any simplex containing edge {v0:?}-{v1:?}")]
    MissingEdgeIncidence {
        /// First edge endpoint.
        v0: VertexKey,
        /// Second edge endpoint.
        v1: VertexKey,
    },
    /// A simplex contains an edge endpoint, but that endpoint's incidence index does not list it.
    #[error("vertex incidence index for {vertex_key:?} is missing simplex {simplex_key:?}")]
    MissingVertexIncidence {
        /// Vertex whose incidence list is missing the simplex key.
        vertex_key: VertexKey,
        /// Simplex expected in the vertex's incidence list.
        simplex_key: SimplexKey,
    },
    /// A vertex incidence entry points to a simplex that does not contain that vertex.
    #[error(
        "vertex incidence index for {vertex_key:?} incorrectly references simplex {simplex_key:?}"
    )]
    VertexIncidenceMismatch {
        /// Vertex whose incidence list contains the inconsistent simplex key.
        vertex_key: VertexKey,
        /// Simplex expected to contain the vertex.
        simplex_key: SimplexKey,
    },
    /// Edge star has the wrong opposite-vertex incidence pattern.
    #[error(
        "edge star must have {expected_vertices} distinct opposite vertices each appearing {expected_occurrences} times, found {found_vertices} distinct vertices"
    )]
    InvalidOppositeVertexIncidence {
        /// Expected number of distinct opposite vertices.
        expected_vertices: usize,
        /// Observed number of distinct opposite vertices.
        found_vertices: usize,
        /// Expected occurrence count for each opposite vertex.
        expected_occurrences: usize,
    },
}

/// Structured reason inverse k=3 triangle adjacency is inconsistent.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlipTriangleAdjacencyError {
    /// Incident simplex does not contain all triangle vertices.
    #[error("simplex {simplex_key:?} does not contain triangle vertices {a:?}, {b:?}, and {c:?}")]
    SimplexMissingTriangleVertices {
        /// Simplex expected to contain the triangle.
        simplex_key: SimplexKey,
        /// First triangle vertex.
        a: VertexKey,
        /// Second triangle vertex.
        b: VertexKey,
        /// Third triangle vertex.
        c: VertexKey,
    },
    /// Triangle star has the wrong ridge-vertex incidence pattern.
    #[error(
        "triangle star must have {expected_vertices} ridge vertices each appearing {expected_occurrences} times, found {found_vertices} distinct vertices"
    )]
    InvalidRidgeVertexIncidence {
        /// Expected number of distinct ridge vertices.
        expected_vertices: usize,
        /// Observed number of distinct ridge vertices.
        found_vertices: usize,
        /// Expected occurrence count for each ridge vertex.
        expected_occurrences: usize,
    },
}

/// Error returned when constructing a [`TriangleHandle`] from invalid vertices.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum TriangleHandleError {
    /// At least two triangle vertices refer to the same vertex.
    #[error("triangle vertices must be distinct, got {vertices:?}")]
    DuplicateVertices {
        /// The supplied triangle vertices.
        vertices: [VertexKey; 3],
    },
}

/// Structured reason inverse k=1 vertex-star adjacency is inconsistent.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlipVertexAdjacencyError {
    /// Incident simplex does not contain the removed vertex.
    #[error("simplex {simplex_key:?} does not contain vertex {vertex_key:?}")]
    SimplexMissingVertex {
        /// Simplex expected to contain the vertex.
        simplex_key: SimplexKey,
        /// Removed vertex.
        vertex_key: VertexKey,
    },
    /// Vertex star has the wrong link-vertex incidence pattern.
    #[error(
        "vertex star must have {expected_vertices} link vertices each appearing {expected_occurrences} times, found {found_vertices} distinct vertices"
    )]
    InvalidLinkVertexIncidence {
        /// Expected number of distinct link vertices.
        expected_vertices: usize,
        /// Observed number of distinct link vertices.
        found_vertices: usize,
        /// Expected occurrence count for each link vertex.
        expected_occurrences: usize,
    },
}

/// Errors that can occur during bistellar flips or repair.
///
/// The enum keeps small scalar, key, and short [`Vec`] diagnostics inline, but
/// boxes nested typed error payloads and exposes them as `#[source]` values.
/// Constructors and pattern matches for those variants use [`Box`], while typed
/// inspection remains available through `reason.as_ref()`, `source.as_ref()`,
/// [`Error::source`](std::error::Error::source), or the boxed [`SmallBuffer`]
/// witness directly without string parsing.
///
/// # Examples
///
/// ```rust
/// use delaunay::flips::{FlipContextError, FlipError};
///
/// let err = FlipError::UnsupportedDimension { dimension: 1 };
/// std::assert_matches!(err, FlipError::UnsupportedDimension { .. });
///
/// let err = FlipError::InvalidFlipContext {
///     reason: Box::new(FlipContextError::OverlappingFaces),
/// };
/// std::assert_matches!(
///     err,
///     FlipError::InvalidFlipContext { reason }
///         if matches!(reason.as_ref(), FlipContextError::OverlappingFaces)
/// );
/// ```
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum FlipError {
    /// Flips are not supported for this dimension.
    #[error("Bistellar flip not supported for D={dimension}")]
    UnsupportedDimension {
        /// Dimension of the triangulation.
        dimension: usize,
    },
    /// The facet is on the boundary (no adjacent simplex).
    #[error("Facet {facet:?} is on the boundary (no neighbor)")]
    BoundaryFacet {
        /// Facet handle.
        facet: FacetHandle,
    },
    /// The referenced simplex was not found.
    #[error("Simplex not found: {simplex_key:?}")]
    MissingSimplex {
        /// Missing simplex key.
        simplex_key: SimplexKey,
    },
    /// The vertex-to-simplices incidence index references a missing simplex.
    #[error("Vertex incidence index for {vertex_key:?} references missing simplex {simplex_key:?}")]
    DanglingVertexIncidence {
        /// Vertex whose incidence list contains the dangling simplex key.
        vertex_key: VertexKey,
        /// Missing simplex key referenced by the incidence index.
        simplex_key: SimplexKey,
    },
    /// The referenced vertex was not found.
    #[error("Vertex not found: {vertex_key:?}")]
    MissingVertex {
        /// Missing vertex key.
        vertex_key: VertexKey,
    },
    /// The neighbor simplex across the facet is missing.
    #[error("Neighbor simplex {neighbor_key:?} not found for facet {facet:?}")]
    MissingNeighbor {
        /// Facet handle.
        facet: FacetHandle,
        /// Missing neighbor key.
        neighbor_key: SimplexKey,
    },
    /// Ridge adjacency references a neighbor simplex key that is no longer live.
    #[error(
        "Ridge adjacency from simplex {simplex_key:?} references missing neighbor {neighbor_key:?}"
    )]
    DanglingRidgeNeighbor {
        /// Simplex whose neighbor table contains the dangling key.
        simplex_key: SimplexKey,
        /// Missing neighbor simplex key.
        neighbor_key: SimplexKey,
    },
    /// Facet adjacency information is inconsistent.
    #[error(
        "Facet adjacency mismatch between simplex {simplex_key:?} and neighbor {neighbor_key:?}"
    )]
    InvalidFacetAdjacency {
        /// Simplex key.
        simplex_key: SimplexKey,
        /// Neighbor simplex key.
        neighbor_key: SimplexKey,
    },
    /// The facet index is out of bounds for the simplex.
    #[error(
        "Facet index {facet_index} out of bounds for simplex {simplex_key:?} with {vertex_count} vertices"
    )]
    InvalidFacetIndex {
        /// Simplex key.
        simplex_key: SimplexKey,
        /// Facet index.
        facet_index: u8,
        /// Vertex count for the simplex.
        vertex_count: usize,
    },
    /// Ridge indices are invalid for the simplex.
    #[error(
        "Ridge indices ({omit_a}, {omit_b}) out of bounds for simplex {simplex_key:?} with {vertex_count} vertices"
    )]
    InvalidRidgeIndex {
        /// Simplex key.
        simplex_key: SimplexKey,
        /// First omitted index.
        omit_a: u8,
        /// Second omitted index.
        omit_b: u8,
        /// Vertex count for the simplex.
        vertex_count: usize,
    },
    /// Ridge adjacency information is inconsistent.
    #[error("Ridge adjacency mismatch for simplex {simplex_key:?}")]
    InvalidRidgeAdjacency {
        /// Simplex key.
        simplex_key: SimplexKey,
    },
    /// Ridge has an invalid multiplicity for k=3 flips.
    #[error("Ridge has invalid multiplicity {found}, expected 3")]
    InvalidRidgeMultiplicity {
        /// Number of incident simplices found.
        found: usize,
    },
    /// Edge has an invalid multiplicity for inverse k=2 flips.
    #[error("Edge has invalid multiplicity {found}, expected {expected}")]
    InvalidEdgeMultiplicity {
        /// Number of incident simplices found.
        found: usize,
        /// Expected multiplicity for the dimension.
        expected: usize,
    },
    /// Triangle has an invalid multiplicity for inverse k=3 flips.
    #[error("Triangle has invalid multiplicity {found}, expected {expected}")]
    InvalidTriangleMultiplicity {
        /// Number of incident simplices found.
        found: usize,
        /// Expected multiplicity for the dimension.
        expected: usize,
    },
    /// Edge adjacency information is inconsistent.
    #[error("Edge adjacency mismatch: {reason}")]
    InvalidEdgeAdjacency {
        /// Structured edge-adjacency reason.
        #[source]
        reason: Box<FlipEdgeAdjacencyError>,
    },
    /// Triangle adjacency information is inconsistent.
    #[error("Triangle adjacency mismatch: {reason}")]
    InvalidTriangleAdjacency {
        /// Structured triangle-adjacency reason.
        #[source]
        reason: Box<FlipTriangleAdjacencyError>,
    },
    /// Vertex star has an invalid multiplicity for inverse k=1 flips.
    #[error("Vertex star has invalid multiplicity {found}, expected {expected}")]
    InvalidVertexMultiplicity {
        /// Number of incident simplices found.
        found: usize,
        /// Expected multiplicity for the dimension.
        expected: usize,
    },
    /// Vertex adjacency information is inconsistent.
    #[error("Vertex adjacency mismatch: {reason}")]
    InvalidVertexAdjacency {
        /// Structured vertex-adjacency reason.
        #[source]
        reason: Box<FlipVertexAdjacencyError>,
    },
    /// Flip context is inconsistent with the requested move.
    #[error("Flip context invalid: {reason}")]
    InvalidFlipContext {
        /// Structured invalid-context reason.
        #[source]
        reason: Box<FlipContextError>,
    },
    /// Geometric predicate failed.
    #[error("Geometric predicate failed: {reason}")]
    PredicateFailure {
        /// Structured predicate failure.
        #[source]
        reason: Box<FlipPredicateError>,
    },
    /// Flip would create a degenerate simplex (zero orientation).
    #[error("Flip would create a degenerate simplex (zero orientation)")]
    DegenerateSimplex,
    /// Delaunay repair would create a negative-orientation replacement simplex.
    #[error(
        "Delaunay repair would create a negative-orientation replacement simplex {simplex_vertices:?}"
    )]
    NegativeOrientation {
        /// Replacement simplex vertices in the rejected order.
        simplex_vertices: Vec<VertexKey>,
    },
    /// Flip would create a duplicate simplex.
    #[error("Flip would create a duplicate simplex")]
    DuplicateSimplex,
    /// Flip would create a non-manifold facet.
    #[error("Flip would create a non-manifold facet")]
    NonManifoldFacet,
    /// Flip would insert a simplex that already exists in the triangulation.
    ///
    /// This violates the bistellar move link condition and can create non-manifold
    /// codimension>1 singularities (e.g., disconnected ridge links).
    #[error(
        "Flip would insert simplex that already exists (k={k_move}, simplex={simplex_vertices:?}, existing_simplex={existing_simplex:?})"
    )]
    InsertedSimplexAlreadyExists {
        /// k for the attempted move.
        k_move: usize,
        /// Vertex keys of the inserted simplex.
        simplex_vertices: Box<SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>>,
        /// A witness simplex key that already contains the inserted simplex.
        existing_simplex: SimplexKey,
    },
    /// Facet iteration failed while seeding flip or repair work.
    #[error("Facet iteration failed: {source}")]
    FacetIteration {
        /// Structured facet iteration failure.
        #[source]
        source: Box<FacetError>,
    },
    /// Simplex creation failed.
    #[error(transparent)]
    SimplexCreation(#[from] Box<SimplexValidationError>),
    /// Neighbor wiring failed during flip application.
    #[error("Neighbor wiring failed: {reason}")]
    NeighborWiring {
        /// Structured neighbor-wiring failure.
        #[source]
        reason: Box<FlipNeighborWiringError>,
    },
    /// TDS mutation failed.
    #[error("TDS mutation failed: {reason}")]
    TdsMutation {
        /// Structured TDS mutation failure.
        #[source]
        reason: Box<FlipMutationError>,
    },
}

impl From<FlipContextError> for FlipError {
    fn from(reason: FlipContextError) -> Self {
        Self::InvalidFlipContext {
            reason: Box::new(reason),
        }
    }
}

impl From<FlipPredicateError> for FlipError {
    fn from(reason: FlipPredicateError) -> Self {
        Self::PredicateFailure {
            reason: Box::new(reason),
        }
    }
}

impl From<FlipEdgeAdjacencyError> for FlipError {
    fn from(reason: FlipEdgeAdjacencyError) -> Self {
        Self::InvalidEdgeAdjacency {
            reason: Box::new(reason),
        }
    }
}

impl From<FlipTriangleAdjacencyError> for FlipError {
    fn from(reason: FlipTriangleAdjacencyError) -> Self {
        Self::InvalidTriangleAdjacency {
            reason: Box::new(reason),
        }
    }
}

impl From<FlipVertexAdjacencyError> for FlipError {
    fn from(reason: FlipVertexAdjacencyError) -> Self {
        Self::InvalidVertexAdjacency {
            reason: Box::new(reason),
        }
    }
}

impl From<SimplexValidationError> for FlipError {
    fn from(source: SimplexValidationError) -> Self {
        Self::SimplexCreation(Box::new(source))
    }
}

impl From<FacetError> for FlipError {
    fn from(source: FacetError) -> Self {
        Self::FacetIteration {
            source: Box::new(source),
        }
    }
}

impl From<FlipNeighborWiringError> for FlipError {
    fn from(reason: FlipNeighborWiringError) -> Self {
        Self::NeighborWiring {
            reason: Box::new(reason),
        }
    }
}

impl From<FlipMutationError> for FlipError {
    fn from(reason: FlipMutationError) -> Self {
        Self::TdsMutation {
            reason: Box::new(reason),
        }
    }
}

impl From<&FlipError> for FlipFailureKind {
    fn from(source: &FlipError) -> Self {
        match source {
            FlipError::UnsupportedDimension { .. } => Self::UnsupportedDimension,
            FlipError::BoundaryFacet { .. } => Self::BoundaryFacet,
            FlipError::MissingSimplex { .. } => Self::MissingSimplex,
            FlipError::DanglingVertexIncidence { .. } => Self::DanglingVertexIncidence,
            FlipError::MissingVertex { .. } => Self::MissingVertex,
            FlipError::MissingNeighbor { .. } => Self::MissingNeighbor,
            FlipError::DanglingRidgeNeighbor { .. } => Self::DanglingRidgeNeighbor,
            FlipError::InvalidFacetAdjacency { .. } => Self::InvalidFacetAdjacency,
            FlipError::InvalidFacetIndex { .. } => Self::InvalidFacetIndex,
            FlipError::InvalidRidgeIndex { .. } => Self::InvalidRidgeIndex,
            FlipError::InvalidRidgeAdjacency { .. } => Self::InvalidRidgeAdjacency,
            FlipError::InvalidRidgeMultiplicity { .. } => Self::InvalidRidgeMultiplicity,
            FlipError::InvalidEdgeMultiplicity { .. } => Self::InvalidEdgeMultiplicity,
            FlipError::InvalidTriangleMultiplicity { .. } => Self::InvalidTriangleMultiplicity,
            FlipError::InvalidEdgeAdjacency { .. } => Self::InvalidEdgeAdjacency,
            FlipError::InvalidTriangleAdjacency { .. } => Self::InvalidTriangleAdjacency,
            FlipError::InvalidVertexMultiplicity { .. } => Self::InvalidVertexMultiplicity,
            FlipError::InvalidVertexAdjacency { .. } => Self::InvalidVertexAdjacency,
            FlipError::InvalidFlipContext { .. } => Self::InvalidFlipContext,
            FlipError::PredicateFailure { .. } => Self::PredicateFailure,
            FlipError::DegenerateSimplex => Self::DegenerateSimplex,
            FlipError::NegativeOrientation { .. } => Self::NegativeOrientation,
            FlipError::DuplicateSimplex => Self::DuplicateSimplex,
            FlipError::NonManifoldFacet => Self::NonManifoldFacet,
            FlipError::InsertedSimplexAlreadyExists { .. } => Self::InsertedSimplexAlreadyExists,
            FlipError::FacetIteration { .. } => Self::FacetIteration,
            FlipError::SimplexCreation(_) => Self::SimplexCreation,
            FlipError::NeighborWiring { reason } => match reason.as_ref() {
                FlipNeighborWiringError::TopologyValidation { .. }
                | FlipNeighborWiringError::DelaunayValidation { .. }
                | FlipNeighborWiringError::TopologyValidationFailed { .. } => {
                    Self::WiringValidation
                }
                FlipNeighborWiringError::DelaunayRepair { .. } => Self::DelaunayRepairFailed,
                _ => Self::NeighborWiring,
            },
            FlipError::TdsMutation { reason }
                if matches!(
                    reason.as_ref(),
                    FlipMutationError::TrialValidation { .. }
                        | FlipMutationError::CoherentOrientationViolation { .. }
                ) =>
            {
                Self::TrialValidation
            }
            FlipError::TdsMutation { .. } => Self::TdsMutation,
        }
    }
}

impl From<FlipError> for FlipFailureKind {
    fn from(source: FlipError) -> Self {
        Self::from(&source)
    }
}

/// Information about a successful flip.
///
/// # Examples
///
/// ```rust
/// use delaunay::flips::{BistellarFlipKind, FlipDirection, FlipInfo};
/// use delaunay::prelude::collections::{SimplexKeyBuffer, SmallBuffer, MAX_PRACTICAL_DIMENSION_SIZE};
/// use delaunay::prelude::tds::{SimplexKey, VertexKey};
/// use slotmap::KeyData;
///
/// let mut removed_simplices = SimplexKeyBuffer::new();
/// removed_simplices.push(SimplexKey::from(KeyData::from_ffi(1)));
/// let mut new_simplices = SimplexKeyBuffer::new();
/// new_simplices.push(SimplexKey::from(KeyData::from_ffi(2)));
///
/// let mut removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
///     SmallBuffer::new();
/// removed_face_vertices.push(VertexKey::from(KeyData::from_ffi(3)));
/// let mut inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
///     SmallBuffer::new();
/// inserted_face_vertices.push(VertexKey::from(KeyData::from_ffi(4)));
///
/// let info: FlipInfo<3> = FlipInfo {
///     kind: BistellarFlipKind::k2(3),
///     direction: FlipDirection::Forward,
///     removed_simplices,
///     new_simplices,
///     removed_face_vertices,
///     inserted_face_vertices,
/// };
/// assert_eq!(info.kind.k(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct FlipInfo<const D: usize> {
    /// Flip kind (k, d).
    pub kind: BistellarFlipKind,
    /// Flip direction.
    pub direction: FlipDirection,
    /// Simplices removed by the flip.
    pub removed_simplices: SimplexKeyBuffer,
    /// Newly created simplices.
    pub new_simplices: SimplexKeyBuffer,
    /// The removed-face simplex (shared by removed simplices).
    pub removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// The inserted-face simplex (complementary simplex).
    pub inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
}

#[derive(Debug, Clone)]
struct AppliedFlip<const D: usize> {
    info: FlipInfo<D>,
    removed_simplex_vertices: RemovedSimplexVertexSnapshot,
}

/// Const-generic flip context for a k-move (forward or inverse).
#[derive(Debug, Clone)]
pub(crate) struct FlipContext<const D: usize, const K: usize> {
    /// Vertices of the removed-face simplex (dimension D+1−K).
    pub removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// Vertices of the inserted-face simplex (dimension K−1).
    pub inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// Simplices removed by the flip (count = K).
    pub removed_simplices: SimplexKeyBuffer,
    /// Flip direction (forward/inverse).
    pub direction: FlipDirection,
}

/// Runtime-k flip context for moves where k depends on D.
#[derive(Debug, Clone)]
pub(crate) struct FlipContextDyn<const D: usize> {
    /// Vertices of the removed-face simplex (dimension D+1−k).
    pub removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// Vertices of the inserted-face simplex (dimension k−1).
    pub inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// Simplices removed by the flip (count = k).
    pub removed_simplices: SimplexKeyBuffer,
    /// Flip direction (forward/inverse).
    pub direction: FlipDirection,
}

/// Canonical handle to a triangle (three vertices).
///
/// # Examples
///
/// ```rust
/// use delaunay::flips::TriangleHandle;
/// use delaunay::prelude::tds::VertexKey;
/// use slotmap::KeyData;
///
/// let a = VertexKey::from(KeyData::from_ffi(1));
/// let b = VertexKey::from(KeyData::from_ffi(2));
/// let c = VertexKey::from(KeyData::from_ffi(3));
///
/// let handle = TriangleHandle::try_new(b, a, c)?;
/// assert_eq!(handle.vertices().len(), 3);
/// # Ok::<(), delaunay::flips::TriangleHandleError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TriangleHandle {
    v0: VertexKey,
    v1: VertexKey,
    v2: VertexKey,
}

impl TriangleHandle {
    /// Create a canonical triangle handle with ordered vertex keys.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::flips::TriangleHandle;
    /// use delaunay::prelude::tds::VertexKey;
    /// use slotmap::KeyData;
    ///
    /// let a = VertexKey::from(KeyData::from_ffi(10));
    /// let b = VertexKey::from(KeyData::from_ffi(20));
    /// let c = VertexKey::from(KeyData::from_ffi(30));
    ///
    /// let handle = TriangleHandle::try_new(a, b, c)?;
    /// assert_eq!(handle.vertices(), [a, b, c]);
    /// # Ok::<(), delaunay::flips::TriangleHandleError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`TriangleHandleError::DuplicateVertices`] if any two supplied
    /// vertices are equal.
    pub fn try_new(a: VertexKey, b: VertexKey, c: VertexKey) -> Result<Self, TriangleHandleError> {
        if a == b || a == c || b == c {
            return Err(TriangleHandleError::DuplicateVertices {
                vertices: [a, b, c],
            });
        }

        Ok(Self::from_validated_vertices(a, b, c))
    }

    /// Creates a canonical triangle handle from vertices already known to be distinct.
    #[must_use]
    pub(crate) fn from_validated_vertices(a: VertexKey, b: VertexKey, c: VertexKey) -> Self {
        let mut verts = [a, b, c];
        verts.sort_unstable_by_key(|v| v.data().as_ffi());
        Self {
            v0: verts[0],
            v1: verts[1],
            v2: verts[2],
        }
    }

    /// Return the triangle vertices.
    #[must_use]
    pub const fn vertices(self) -> [VertexKey; 3] {
        [self.v0, self.v1, self.v2]
    }
}

impl TryFrom<[VertexKey; 3]> for TriangleHandle {
    type Error = TriangleHandleError;

    fn try_from([a, b, c]: [VertexKey; 3]) -> Result<Self, Self::Error> {
        Self::try_new(a, b, c)
    }
}

/// Lightweight handle to a ridge (codimension-2 face) within a simplex.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::*;
/// use delaunay::flips::RidgeHandle;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Flip(#[from] delaunay::flips::FlipError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = [
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
/// let Some((simplex_key, _)) = dt.simplices().next() else {
///     return Ok(());
/// };
/// let handle = RidgeHandle::try_new(dt.tds(), simplex_key, 2, 0)?;
/// assert_eq!(handle.omit_a(), 0);
/// assert_eq!(handle.omit_b(), 2);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RidgeHandle {
    simplex_key: SimplexKey,
    omit_a: u8,
    omit_b: u8,
}

impl RidgeHandle {
    /// Creates a new ridge handle by validating the omitted vertex indices
    /// against a live TDS simplex.
    ///
    /// # Errors
    ///
    /// Returns [`FlipError::UnsupportedDimension`] for dimensions below 3,
    /// [`FlipError::MissingSimplex`] if `simplex_key` is not present in `tds`,
    /// or [`FlipError::InvalidRidgeIndex`] if either omitted index is out of
    /// bounds or both indices are the same.
    pub fn try_new<U, V, const D: usize>(
        tds: &Tds<U, V, D>,
        simplex_key: SimplexKey,
        omit_a: u8,
        omit_b: u8,
    ) -> Result<Self, FlipError> {
        if D < 3 {
            return Err(FlipError::UnsupportedDimension { dimension: D });
        }

        let simplex = tds
            .simplex(simplex_key)
            .ok_or(FlipError::MissingSimplex { simplex_key })?;
        let vertex_count = simplex.number_of_vertices();
        let first_omit_index = usize::from(omit_a);
        let second_omit_index = usize::from(omit_b);
        if first_omit_index >= vertex_count || second_omit_index >= vertex_count || omit_a == omit_b
        {
            return Err(FlipError::InvalidRidgeIndex {
                simplex_key,
                omit_a,
                omit_b,
                vertex_count,
            });
        }

        Ok(Self::from_validated(simplex_key, omit_a, omit_b))
    }

    /// Creates a ridge handle from omitted vertex indices already proven valid
    /// by the caller.
    #[inline]
    pub(crate) const fn from_validated(simplex_key: SimplexKey, omit_a: u8, omit_b: u8) -> Self {
        if omit_a <= omit_b {
            Self {
                simplex_key,
                omit_a,
                omit_b,
            }
        } else {
            Self {
                simplex_key,
                omit_a: omit_b,
                omit_b: omit_a,
            }
        }
    }

    /// Returns the simplex key.
    #[must_use]
    pub const fn simplex_key(&self) -> SimplexKey {
        self.simplex_key
    }

    /// Returns the first omitted index.
    #[must_use]
    pub const fn omit_a(&self) -> u8 {
        self.omit_a
    }

    /// Returns the second omitted index.
    #[must_use]
    pub const fn omit_b(&self) -> u8 {
        self.omit_b
    }
}

/// Statistics for flip-based Delaunay repair.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::repair::DelaunayRepairStats;
///
/// let stats = DelaunayRepairStats::default();
/// assert_eq!(stats.flips_performed, 0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct DelaunayRepairStats {
    /// Number of queued items checked (facets, ridges, edges, triangles).
    pub facets_checked: usize,
    /// Number of flips performed.
    pub flips_performed: usize,
    /// Maximum queue length observed.
    pub max_queue_len: usize,
}

/// Wall-clock phase timing for one batch-local repair pass.
#[expect(
    clippy::struct_field_names,
    reason = "phase timing telemetry keeps units explicit on every exported field"
)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LocalRepairPhaseTiming {
    /// Nanoseconds spent cloning the TDS snapshot used for rollback.
    pub(crate) snapshot_nanos: u64,
    /// Nanoseconds spent applying flip-repair attempts.
    pub(crate) attempt_nanos: u64,
    /// Nanoseconds spent seeding repair attempt queues.
    pub(crate) attempt_seed_nanos: u64,
    /// Nanoseconds spent processing k=2 facet queue items.
    pub(crate) attempt_facet_nanos: u64,
    /// Nanoseconds spent processing k=3 ridge queue items.
    pub(crate) attempt_ridge_nanos: u64,
    /// Nanoseconds spent processing inverse k=2 edge queue items.
    pub(crate) attempt_edge_nanos: u64,
    /// Nanoseconds spent processing inverse k=3 triangle queue items.
    pub(crate) attempt_triangle_nanos: u64,
    /// Nanoseconds spent replaying postcondition predicates.
    pub(crate) postcondition_nanos: u64,
    /// Nanoseconds spent restoring the TDS from a saved snapshot.
    pub(crate) restore_nanos: u64,
}

impl LocalRepairPhaseTiming {
    /// Adds rollback snapshot-clone time so setup cost stays separate from repair work.
    fn record_snapshot(&mut self, elapsed: Duration) {
        self.snapshot_nanos = self
            .snapshot_nanos
            .saturating_add(duration_nanos_saturating(elapsed));
    }

    /// Adds total flip-attempt time across queue seeding and queue processing.
    fn record_attempt(&mut self, elapsed: Duration) {
        self.attempt_nanos = self
            .attempt_nanos
            .saturating_add(duration_nanos_saturating(elapsed));
    }

    /// Adds time spent building the queue for one repair attempt.
    fn record_attempt_seed(&mut self, elapsed: Duration) {
        self.attempt_seed_nanos = self
            .attempt_seed_nanos
            .saturating_add(duration_nanos_saturating(elapsed));
    }

    /// Adds time spent processing k=2 facet queue items.
    fn record_attempt_facet(&mut self, elapsed: Duration) {
        self.attempt_facet_nanos = self
            .attempt_facet_nanos
            .saturating_add(duration_nanos_saturating(elapsed));
    }

    /// Adds time spent processing k=3 ridge queue items.
    fn record_attempt_ridge(&mut self, elapsed: Duration) {
        self.attempt_ridge_nanos = self
            .attempt_ridge_nanos
            .saturating_add(duration_nanos_saturating(elapsed));
    }

    /// Adds time spent processing inverse k=2 edge queue items.
    fn record_attempt_edge(&mut self, elapsed: Duration) {
        self.attempt_edge_nanos = self
            .attempt_edge_nanos
            .saturating_add(duration_nanos_saturating(elapsed));
    }

    /// Adds time spent processing inverse k=3 triangle queue items.
    fn record_attempt_triangle(&mut self, elapsed: Duration) {
        self.attempt_triangle_nanos = self
            .attempt_triangle_nanos
            .saturating_add(duration_nanos_saturating(elapsed));
    }

    /// Adds time spent replaying local Delaunay postconditions after repair attempts.
    fn record_postcondition(&mut self, elapsed: Duration) {
        self.postcondition_nanos = self
            .postcondition_nanos
            .saturating_add(duration_nanos_saturating(elapsed));
    }

    /// Adds time spent restoring the saved TDS after a failed repair attempt.
    fn record_restore(&mut self, elapsed: Duration) {
        self.restore_nanos = self
            .restore_nanos
            .saturating_add(duration_nanos_saturating(elapsed));
    }
}

/// Publishes one local repair pass's phase timing when the caller requested telemetry.
fn publish_local_repair_phase_timing(
    timing: &mut Option<&mut LocalRepairPhaseTiming>,
    phase_timing: LocalRepairPhaseTiming,
) {
    if let Some(timing) = timing.as_deref_mut() {
        *timing = phase_timing;
    }
}

/// Crate-private repair result with the validation frontier for callers that
/// need post-repair topology checks without scanning the whole TDS.
#[derive(Debug, Clone)]
pub(crate) struct DelaunayRepairRun {
    /// Public aggregate repair statistics.
    pub stats: DelaunayRepairStats,
    /// Simplices to validate after the final repair attempt.
    ///
    /// This records the simplices created by successful flips, regardless of
    /// whether the repair queues were seeded locally or from the full TDS.  The
    /// queue frontier controls Delaunay postcondition replay; ridge-link
    /// topology validation only needs the simplices whose incidence changed.
    pub touched_simplices: SimplexKeyBuffer,
    /// Whether the final attempt used full-TDS queue seeding.
    pub used_full_reseed: bool,
}

/// Carries both aggregate attempt stats and the final flip context so
/// postcondition diagnostics can relate the first unresolved local violation to
/// the last repair move that modified the TDS.
#[derive(Debug)]
struct RepairAttemptOutcome {
    postcondition_required: bool,
    stats: DelaunayRepairStats,
    last_applied_flip: Option<LastAppliedFlip>,
    touched_simplices: SimplexKeyBuffer,
    used_full_reseed: bool,
}

/// Determines whether repair changed or observed enough local state to require postcondition replay.
const fn repair_postcondition_required(
    stats: &DelaunayRepairStats,
    diagnostics: &RepairDiagnostics,
) -> bool {
    stats.flips_performed > 0 || diagnostics.saw_applicable_repair_site
}

/// Adds newly-created simplices to the repair mutation frontier without duplicates.
fn record_touched_simplices(
    touched_simplices: &mut SimplexKeyBuffer,
    touched_simplex_set: &mut FastHashSet<SimplexKey>,
    new_simplices: &[SimplexKey],
) {
    for &simplex_key in new_simplices {
        if touched_simplex_set.insert(simplex_key) {
            touched_simplices.push(simplex_key);
        }
    }
}

/// Builds the local postcondition frontier from the caller's seed simplices plus
/// simplices created by successful flips.
fn local_postcondition_frontier(
    seed_simplices: &[SimplexKey],
    touched_simplices: &[SimplexKey],
) -> SimplexKeyBuffer {
    let mut frontier = SimplexKeyBuffer::new();
    let mut seen = FastHashSet::<SimplexKey>::default();
    for &simplex_key in seed_simplices.iter().chain(touched_simplices) {
        if seen.insert(simplex_key) {
            frontier.push(simplex_key);
        }
    }
    frontier
}

/// Converts an attempt outcome into the crate-private repair run result.
fn repair_run_from_attempt(outcome: RepairAttemptOutcome) -> DelaunayRepairRun {
    let RepairAttemptOutcome {
        stats,
        touched_simplices,
        used_full_reseed,
        ..
    } = outcome;

    DelaunayRepairRun {
        stats,
        touched_simplices,
        used_full_reseed,
    }
}

/// Queue ordering policy for flip repair attempts.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::repair::RepairQueueOrder;
///
/// let order = RepairQueueOrder::Fifo;
/// assert_eq!(order, RepairQueueOrder::Fifo);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairQueueOrder {
    /// FIFO (breadth-like) ordering.
    Fifo,
    /// LIFO (depth-like) ordering.
    Lifo,
}

/// Diagnostics captured when flip-based repair fails to converge.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::repair::{
///     DelaunayRepairDiagnostics, RepairQueueOrder,
/// };
///
/// let diagnostics = DelaunayRepairDiagnostics {
///     facets_checked: 0,
///     flips_performed: 0,
///     max_queue_len: 0,
///     ambiguous_predicates: 0,
///     ambiguous_predicate_samples: Vec::new(),
///     predicate_failures: 0,
///     cycle_detections: 0,
///     cycle_signature_samples: Vec::new(),
///     attempt: 1,
///     queue_order: RepairQueueOrder::Fifo,
/// };
/// assert!(diagnostics.to_string().contains("checked"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DelaunayRepairDiagnostics {
    /// Number of queued items checked.
    pub facets_checked: usize,
    /// Number of flips performed.
    pub flips_performed: usize,
    /// Maximum queue length observed.
    pub max_queue_len: usize,
    /// Count of ambiguous predicate evaluations (boundary classifications).
    pub ambiguous_predicates: usize,
    /// Sample of ambiguous predicate site hashes (deterministic, truncated).
    pub ambiguous_predicate_samples: Vec<u64>,
    /// Count of predicate failures (conversion/robust fallback errors).
    pub predicate_failures: usize,
    /// Count of detected flip cycles (repeat flip signatures within a sliding window).
    pub cycle_detections: usize,
    /// Sample of repeated flip-context signature hashes (deterministic, truncated).
    pub cycle_signature_samples: Vec<u64>,
    /// Attempt number (1-based).
    pub attempt: usize,
    /// Queue ordering policy used for this attempt.
    pub queue_order: RepairQueueOrder,
}

impl fmt::Display for DelaunayRepairDiagnostics {
    /// Format a concise diagnostics summary.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "checked {} facets, ambiguous={}, max_queue={}, flips={}, attempt={}, order={:?}, predicate_failures={}, cycles={}, cycle_samples={:?}",
            self.facets_checked,
            self.ambiguous_predicates,
            self.max_queue_len,
            self.flips_performed,
            self.attempt,
            self.queue_order,
            self.predicate_failures,
            self.cycle_detections,
            self.cycle_signature_samples
        )
    }
}

/// Verification phase that failed during flip-based Delaunay repair.
///
/// This context is carried by [`DelaunayRepairError::VerificationFailed`] so
/// callers can distinguish generic post-repair validation from local k=2/k=3
/// postcondition checks without parsing the display message.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::repair::{
///     DelaunayRepairError, DelaunayRepairVerificationContext, FlipError,
/// };
///
/// let err = DelaunayRepairError::VerificationFailed {
///     context: DelaunayRepairVerificationContext::StrictValidation,
///     source: Box::new(FlipError::DegenerateSimplex),
/// };
///
/// std::assert_matches!(
///     err,
///     DelaunayRepairError::VerificationFailed {
///         context: DelaunayRepairVerificationContext::StrictValidation,
///         ..
///     },
/// );
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DelaunayRepairVerificationContext {
    /// Generic post-repair verification.
    PostRepairVerification,
    /// Strict validation pass.
    StrictValidation,
    /// Local k=2 degeneracy verification.
    LocalK2DegeneracyVerification,
    /// Local k=2 postcondition verification.
    LocalK2PostconditionVerification,
    /// Local k=3 degeneracy verification.
    LocalK3DegeneracyVerification,
    /// Local k=3 postcondition verification.
    LocalK3PostconditionVerification,
    /// Local inverse k=2 postcondition verification.
    LocalInverseK2PostconditionVerification,
    /// Local inverse k=3 postcondition verification.
    LocalInverseK3PostconditionVerification,
}

impl fmt::Display for DelaunayRepairVerificationContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PostRepairVerification => f.write_str("post-repair verification"),
            Self::StrictValidation => f.write_str("strict validation"),
            Self::LocalK2DegeneracyVerification => f.write_str("local k=2 degeneracy verification"),
            Self::LocalK2PostconditionVerification => {
                f.write_str("local k=2 postcondition verification")
            }
            Self::LocalK3DegeneracyVerification => f.write_str("local k=3 degeneracy verification"),
            Self::LocalK3PostconditionVerification => {
                f.write_str("local k=3 postcondition verification")
            }
            Self::LocalInverseK2PostconditionVerification => {
                f.write_str("local inverse k=2 postcondition verification")
            }
            Self::LocalInverseK3PostconditionVerification => {
                f.write_str("local inverse k=3 postcondition verification")
            }
        }
    }
}

/// Structured reason a repair pass failed its postcondition.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DelaunayRepairPostconditionFailure {
    /// Repair disconnected the triangulation neighbor graph.
    Disconnected {
        /// Number of simplices remaining when the disconnected graph was detected.
        simplex_count: usize,
    },
    /// A local k=2 facet flip opportunity remained after repair.
    LocalK2Violation {
        /// Facet whose flip predicate still reports a violation.
        facet: FacetHandle,
        /// Optional opt-in diagnostic details captured under repair debug flags.
        debug_details: Option<String>,
    },
    /// A local k=3 ridge flip opportunity remained after repair.
    LocalK3Violation {
        /// Ridge whose flip predicate still reports a violation.
        ridge: RidgeHandle,
    },
    /// A local inverse k=2 edge-collapse opportunity remained after repair.
    LocalInverseK2Violation {
        /// Edge whose inverse flip predicate still reports a violation.
        edge: EdgeKey,
    },
    /// A local inverse k=3 triangle-collapse opportunity remained after repair.
    LocalInverseK3Violation {
        /// Triangle whose inverse flip predicate still reports a violation.
        triangle: TriangleHandle,
    },
}

impl fmt::Display for DelaunayRepairPostconditionFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disconnected { simplex_count } => write!(
                f,
                "repair pass disconnected the triangulation ({simplex_count} simplices remain); neighbor wiring is incomplete"
            ),
            Self::LocalK2Violation {
                facet,
                debug_details,
            } => {
                write!(
                    f,
                    "local k=2 violation remains after repair (facet={facet:?})"
                )?;
                if let Some(details) = debug_details {
                    write!(f, "; {details}")?;
                }
                Ok(())
            }
            Self::LocalK3Violation { ridge } => {
                write!(
                    f,
                    "local k=3 violation remains after repair (ridge={ridge:?})"
                )
            }
            Self::LocalInverseK2Violation { edge } => {
                write!(
                    f,
                    "local inverse k=2 flip remains applicable after repair (edge={edge:?})"
                )
            }
            Self::LocalInverseK3Violation { triangle } => write!(
                f,
                "local inverse k=3 flip remains applicable after repair (triangle={triangle:?})"
            ),
        }
    }
}

impl std::error::Error for DelaunayRepairPostconditionFailure {}

/// Structured reason orientation canonicalization failed after repair.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum DelaunayRepairOrientationCanonicalizationFailure {
    /// Positive-orientation promotion failed after a flip-repair pass.
    #[error("after flip repair: {source}")]
    AfterFlipRepair {
        /// Insertion-layer failure produced by orientation promotion.
        #[source]
        source: Box<InsertionError>,
    },
}

/// Compact orientation-canonicalization failure category for non-recursive summaries.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum DelaunayRepairOrientationCanonicalizationFailureKind {
    /// Positive-orientation promotion failed after a flip-repair pass.
    #[error("after flip repair: {source_kind:?}")]
    AfterFlipRepair {
        /// Category of the insertion-layer failure.
        source_kind: InsertionErrorKind,
    },
}

impl From<&DelaunayRepairOrientationCanonicalizationFailure>
    for DelaunayRepairOrientationCanonicalizationFailureKind
{
    fn from(source: &DelaunayRepairOrientationCanonicalizationFailure) -> Self {
        match source {
            DelaunayRepairOrientationCanonicalizationFailure::AfterFlipRepair { source } => {
                Self::AfterFlipRepair {
                    source_kind: insertion_error_kind(source),
                }
            }
        }
    }
}

/// Passive vertex context reported with heuristic rebuild failures.
///
/// The fields identify the vertex position, UUID, and coordinates that were
/// being replayed when rebuild failed. This is diagnostic context only; repair
/// algorithms do not accept it back as proof of a valid vertex.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct DelaunayRepairHeuristicVertexContext {
    /// Position of the vertex in the shuffled rebuild order.
    pub index: usize,
    /// Stable vertex UUID.
    pub vertex_uuid: uuid::Uuid,
    /// Vertex coordinates at the rebuild boundary.
    pub coordinates: CoordinateValues,
}

impl fmt::Display for DelaunayRepairHeuristicVertexContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "idx={} uuid={} coords={}",
            self.index, self.vertex_uuid, self.coordinates
        )
    }
}

/// Structured reason heuristic rebuild failed during advanced repair.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum DelaunayRepairHeuristicRebuildFailure {
    /// Heuristic rebuild recursion exceeded its guard depth.
    #[error("heuristic rebuild recursion depth exceeded {max_depth}")]
    RecursionDepthExceeded {
        /// Maximum permitted nested heuristic rebuild depth.
        max_depth: usize,
    },
    /// Primary repair, robust fallback, and heuristic rebuild all failed.
    #[error("primary repair failed ({primary}); robust fallback failed ({robust}); {heuristic}")]
    FallbackChainFailed {
        /// Primary flip-repair failure.
        #[source]
        primary: Box<DelaunayRepairError>,
        /// Robust-kernel fallback failure.
        robust: Box<DelaunayRepairError>,
        /// Heuristic rebuild failure.
        heuristic: Box<Self>,
    },
    /// A non-heuristic repair error escaped the heuristic rebuild path.
    #[error("heuristic rebuild failed with unexpected repair error: {source}")]
    UnexpectedRepairFailure {
        /// Repair error returned by the heuristic path.
        #[source]
        source: Box<DelaunayRepairError>,
    },
    /// The attempt loop exited without recording a rebuild attempt.
    #[error("heuristic rebuild made no attempts")]
    NoAttempts,
    /// Vertex insertion failed during heuristic rebuild.
    #[error("heuristic rebuild insertion failed at {vertex}: {source}")]
    InsertionFailed {
        /// Vertex being inserted.
        vertex: DelaunayRepairHeuristicVertexContext,
        /// Insertion failure.
        #[source]
        source: Box<InsertionError>,
    },
    /// Local repair failed after a heuristic rebuild insertion.
    #[error("heuristic rebuild repair failed at {vertex}: {source}")]
    RepairFailed {
        /// Vertex whose insertion triggered repair.
        vertex: DelaunayRepairHeuristicVertexContext,
        /// Insertion-layer repair failure.
        #[source]
        source: Box<InsertionError>,
    },
    /// Delaunay check failed after a heuristic rebuild insertion.
    #[error("heuristic rebuild Delaunay check failed at {vertex}: {source}")]
    DelaunayCheckFailed {
        /// Vertex whose insertion triggered the check.
        vertex: DelaunayRepairHeuristicVertexContext,
        /// Insertion-layer check failure.
        #[source]
        source: Box<InsertionError>,
    },
    /// A vertex was skipped during heuristic rebuild.
    #[error("heuristic rebuild skipped vertex at {vertex}: {source}")]
    SkippedVertex {
        /// Skipped vertex.
        vertex: DelaunayRepairHeuristicVertexContext,
        /// Insertion-layer skip reason.
        #[source]
        source: Box<InsertionError>,
    },
    /// One deterministic rebuild attempt failed.
    #[error(
        "attempt {attempt}/{max_attempts} (shuffle_seed={shuffle_seed} perturbation_seed={perturbation_seed}): {source}"
    )]
    AttemptFailed {
        /// 1-based attempt number.
        attempt: usize,
        /// Maximum number of attempts.
        max_attempts: usize,
        /// Shuffle seed used for this attempt.
        shuffle_seed: u64,
        /// Perturbation seed used for this attempt.
        perturbation_seed: u64,
        /// Attempt failure.
        #[source]
        source: Box<DelaunayRepairError>,
    },
    /// Every deterministic heuristic rebuild attempt failed.
    #[error("heuristic rebuild failed after {attempts} attempts: {last_failure}")]
    ExhaustedAttempts {
        /// Number of attempts tried.
        attempts: usize,
        /// Last observed attempt failure.
        #[source]
        last_failure: Box<Self>,
    },
}

/// Compact heuristic-rebuild failure category for non-recursive summaries.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum DelaunayRepairHeuristicRebuildFailureKind {
    /// Heuristic rebuild recursion exceeded its guard depth.
    #[error("recursion depth exceeded")]
    RecursionDepthExceeded,
    /// Primary repair, robust fallback, and heuristic rebuild all failed.
    #[error("fallback chain failed")]
    FallbackChainFailed,
    /// A non-heuristic repair error escaped the heuristic rebuild path.
    #[error("unexpected repair failure")]
    UnexpectedRepairFailure,
    /// The attempt loop exited without recording a rebuild attempt.
    #[error("no attempts")]
    NoAttempts,
    /// Vertex insertion failed during heuristic rebuild.
    #[error("insertion failed")]
    InsertionFailed,
    /// Local repair failed after a heuristic rebuild insertion.
    #[error("repair failed")]
    RepairFailed,
    /// Delaunay check failed after a heuristic rebuild insertion.
    #[error("Delaunay check failed")]
    DelaunayCheckFailed,
    /// A vertex was skipped during heuristic rebuild.
    #[error("skipped vertex")]
    SkippedVertex,
    /// One deterministic rebuild attempt failed.
    #[error("attempt failed")]
    AttemptFailed,
    /// Every deterministic heuristic rebuild attempt failed.
    #[error("attempts exhausted")]
    ExhaustedAttempts,
}

impl From<&DelaunayRepairHeuristicRebuildFailure> for DelaunayRepairHeuristicRebuildFailureKind {
    fn from(source: &DelaunayRepairHeuristicRebuildFailure) -> Self {
        match source {
            DelaunayRepairHeuristicRebuildFailure::RecursionDepthExceeded { .. } => {
                Self::RecursionDepthExceeded
            }
            DelaunayRepairHeuristicRebuildFailure::FallbackChainFailed { .. } => {
                Self::FallbackChainFailed
            }
            DelaunayRepairHeuristicRebuildFailure::UnexpectedRepairFailure { .. } => {
                Self::UnexpectedRepairFailure
            }
            DelaunayRepairHeuristicRebuildFailure::NoAttempts => Self::NoAttempts,
            DelaunayRepairHeuristicRebuildFailure::InsertionFailed { .. } => Self::InsertionFailed,
            DelaunayRepairHeuristicRebuildFailure::RepairFailed { .. } => Self::RepairFailed,
            DelaunayRepairHeuristicRebuildFailure::DelaunayCheckFailed { .. } => {
                Self::DelaunayCheckFailed
            }
            DelaunayRepairHeuristicRebuildFailure::SkippedVertex { .. } => Self::SkippedVertex,
            DelaunayRepairHeuristicRebuildFailure::AttemptFailed { .. } => Self::AttemptFailed,
            DelaunayRepairHeuristicRebuildFailure::ExhaustedAttempts { .. } => {
                Self::ExhaustedAttempts
            }
        }
    }
}

const fn insertion_error_kind(source: &InsertionError) -> InsertionErrorKind {
    match source {
        InsertionError::ConflictRegion(_) => InsertionErrorKind::ConflictRegion,
        InsertionError::Location(_) => InsertionErrorKind::Location,
        InsertionError::CavityFilling { .. } => InsertionErrorKind::CavityFilling,
        InsertionError::NeighborWiring { .. } => InsertionErrorKind::NeighborWiring,
        InsertionError::NonManifoldTopology { .. } => InsertionErrorKind::NonManifoldTopology,
        InsertionError::HullExtension { .. } => InsertionErrorKind::HullExtension,
        InsertionError::DelaunayValidationFailed { .. } => {
            InsertionErrorKind::DelaunayValidationFailed
        }
        InsertionError::EmbeddingValidationFailed { .. } => {
            InsertionErrorKind::EmbeddingValidationFailed
        }
        InsertionError::DelaunayRepairFailed { .. } => InsertionErrorKind::DelaunayRepairFailed,
        InsertionError::DuplicateCoordinates { .. } => InsertionErrorKind::DuplicateCoordinates,
        InsertionError::DuplicateUuid { .. } => InsertionErrorKind::DuplicateUuid,
        InsertionError::TopologyValidation(_) => InsertionErrorKind::TopologyValidation,
        InsertionError::TopologyValidationFailed { .. } => {
            InsertionErrorKind::TopologyValidationFailed
        }
        InsertionError::MaxSimplicesRemovedExceeded { .. } => {
            InsertionErrorKind::MaxSimplicesRemovedExceeded
        }
        InsertionError::SpatialIndexConstruction { .. } => {
            InsertionErrorKind::SpatialIndexConstruction
        }
        InsertionError::PerturbedCoordinateInvalid { .. } => {
            InsertionErrorKind::PerturbedCoordinateInvalid
        }
    }
}

/// Errors that can occur during flip-based Delaunay repair.
///
/// Large typed payloads are boxed to keep the public enum small and cheap to
/// move, while scalar fields and short diagnostic strings stay inline. Boxed
/// variants still preserve their concrete source type so callers can inspect
/// or pattern-match the full error chain when they need repair diagnostics.
/// For example, [`DelaunayRepairError::NonConvergent`] boxes
/// [`DelaunayRepairDiagnostics`], and [`DelaunayRepairError::Flip`] boxes the
/// underlying [`FlipError`].
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::repair::{DelaunayRepairError, FlipError, TopologyGuarantee};
///
/// let err = DelaunayRepairError::InvalidTopology {
///     required: TopologyGuarantee::PLManifold,
///     found: TopologyGuarantee::Pseudomanifold,
///     message: "requires manifold",
/// };
/// std::assert_matches!(err, DelaunayRepairError::InvalidTopology { .. });
///
/// let flip_err = DelaunayRepairError::from(FlipError::DegenerateSimplex);
/// std::assert_matches!(
///     flip_err,
///     DelaunayRepairError::Flip { source }
///         if matches!(source.as_ref(), FlipError::DegenerateSimplex)
/// );
/// ```
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum DelaunayRepairError {
    /// Repair did not converge within the flip budget.
    #[error("Delaunay repair failed to converge after {max_flips} flips ({diagnostics})")]
    NonConvergent {
        /// Maximum flips allowed.
        max_flips: usize,
        /// Diagnostics captured during the failed attempt (boxed to keep the
        /// error enum small on the stack).
        diagnostics: Box<DelaunayRepairDiagnostics>,
    },
    /// Repair completed but left a Delaunay violation.
    #[error("Delaunay repair postcondition failed: {reason}")]
    PostconditionFailed {
        /// Structured postcondition failure reason.
        #[source]
        reason: Box<DelaunayRepairPostconditionFailure>,
    },
    /// Post-repair verification could not evaluate a local flip predicate.
    #[error("Delaunay repair verification failed during {context}: {source}")]
    VerificationFailed {
        /// Verification phase that failed.
        context: DelaunayRepairVerificationContext,
        /// Underlying flip or predicate error.
        #[source]
        source: Box<FlipError>,
    },
    /// Repair completed but orientation canonicalization failed.
    #[error("Delaunay repair orientation canonicalization failed: {reason}")]
    OrientationCanonicalizationFailed {
        /// Structured canonicalization failure reason.
        #[source]
        reason: Box<DelaunayRepairOrientationCanonicalizationFailure>,
    },
    /// Flip-based repair is not admissible under the current topology guarantee.
    #[error("Delaunay repair requires {required:?} topology, found {found:?}: {message}")]
    InvalidTopology {
        /// Required topology guarantee.
        required: TopologyGuarantee,
        /// Actual topology guarantee.
        found: TopologyGuarantee,
        /// Additional context for the mismatch.
        message: &'static str,
    },
    /// Heuristic rebuild failed during advanced repair.
    #[error("Heuristic rebuild failed: {reason}")]
    HeuristicRebuildFailed {
        /// Structured rebuild failure reason.
        #[source]
        reason: Box<DelaunayRepairHeuristicRebuildFailure>,
    },
    /// A lower-level [`FlipError`] stopped repair.
    ///
    /// The source is boxed to keep [`DelaunayRepairError`] compact while
    /// preserving the concrete flip failure for callers that need to inspect it.
    #[error("flip error: {source}")]
    Flip {
        /// Typed flip failure that stopped repair.
        #[source]
        source: Box<FlipError>,
    },
}

impl From<FlipError> for DelaunayRepairError {
    fn from(source: FlipError) -> Self {
        Self::Flip {
            source: Box::new(source),
        }
    }
}

impl From<FacetError> for DelaunayRepairError {
    fn from(source: FacetError) -> Self {
        Self::from(FlipError::from(source))
    }
}

impl From<DelaunayRepairError> for FlipNeighborRepairFailure {
    fn from(source: DelaunayRepairError) -> Self {
        match source {
            DelaunayRepairError::NonConvergent {
                max_flips,
                diagnostics,
            } => Self::NonConvergent {
                max_flips,
                diagnostics: (*diagnostics).into(),
            },
            DelaunayRepairError::PostconditionFailed { reason } => {
                Self::PostconditionFailed { reason: *reason }
            }
            DelaunayRepairError::VerificationFailed { context, source } => {
                Self::VerificationFailed {
                    context,
                    source_kind: FlipFailureKind::from(source.as_ref()),
                }
            }
            DelaunayRepairError::OrientationCanonicalizationFailed { reason } => {
                Self::OrientationCanonicalizationFailed {
                    reason: reason.as_ref().into(),
                }
            }
            DelaunayRepairError::InvalidTopology {
                required,
                found,
                message,
            } => Self::InvalidTopology {
                required,
                found,
                message,
            },
            DelaunayRepairError::HeuristicRebuildFailed { reason } => {
                Self::HeuristicRebuildFailed {
                    reason: reason.as_ref().into(),
                }
            }
            DelaunayRepairError::Flip { source } => Self::Flip {
                source_kind: FlipFailureKind::from(source.as_ref()),
            },
        }
    }
}

/// Build flip context for a k=2 (facet) flip.
///
/// # Errors
///
/// Returns a [`FlipError`] if the facet is invalid, lies on the boundary, references
/// missing simplices/vertices, or the adjacency data is inconsistent.
pub(crate) fn build_k2_flip_context<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    facet: FacetHandle,
) -> Result<FlipContext<D, 2>, FlipError>
where
    U: DataType,
    V: DataType,
{
    if D < 2 {
        return Err(FlipError::UnsupportedDimension { dimension: D });
    }

    let simplex_a_key = facet.simplex_key();
    let simplex_a = tds
        .simplex(simplex_a_key)
        .ok_or(FlipError::MissingSimplex {
            simplex_key: simplex_a_key,
        })?;

    let facet_index_a = usize::from(facet.facet_index());
    let vertex_count = simplex_a.number_of_vertices();
    if facet_index_a >= vertex_count {
        return Err(FlipError::InvalidFacetIndex {
            simplex_key: simplex_a_key,
            facet_index: facet.facet_index(),
            vertex_count,
        });
    }

    let neighbor_key = simplex_a
        .neighbor_key(facet_index_a)
        .flatten()
        .ok_or(FlipError::BoundaryFacet { facet })?;

    let simplex_b = tds
        .simplex(neighbor_key)
        .ok_or(FlipError::MissingNeighbor {
            facet,
            neighbor_key,
        })?;

    let Some(facet_index_b) = simplex_a
        .mirror_facet_index(facet_index_a, simplex_b)
        .or_else(|| back_reference_facet_index(simplex_a_key, simplex_b))
    else {
        return Err(FlipError::InvalidFacetAdjacency {
            simplex_key: simplex_a_key,
            neighbor_key,
        });
    };

    let opposite_a = simplex_a.vertices()[facet_index_a];
    let opposite_b = simplex_b.vertices()[facet_index_b];

    let shared_facet = facet_vertices_from_simplex(simplex_a, facet_index_a);

    if shared_facet.len() != D {
        return Err(FlipError::InvalidFacetAdjacency {
            simplex_key: simplex_a_key,
            neighbor_key,
        });
    }

    if shared_facet.contains(&opposite_a)
        || shared_facet.contains(&opposite_b)
        || opposite_a == opposite_b
    {
        return Err(FlipError::InvalidFacetAdjacency {
            simplex_key: simplex_a_key,
            neighbor_key,
        });
    }

    for &v in &shared_facet {
        if !simplex_b.contains_vertex(v) {
            return Err(FlipError::InvalidFacetAdjacency {
                simplex_key: simplex_a_key,
                neighbor_key,
            });
        }
    }

    let removed_simplices: SimplexKeyBuffer = [simplex_a_key, neighbor_key].into_iter().collect();
    let mut inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(2);
    inserted_face_vertices.push(opposite_a);
    inserted_face_vertices.push(opposite_b);

    Ok(FlipContext {
        removed_face_vertices: shared_facet,
        inserted_face_vertices,
        removed_simplices,
        direction: FlipDirection::Forward,
    })
}

/// Finds the neighbor slot that points back to a source simplex when reciprocal
/// neighbor pointers are already available.
fn back_reference_facet_index<V, const D: usize>(
    source_simplex: SimplexKey,
    neighbor_simplex: &Simplex<V, D>,
) -> Option<usize> {
    neighbor_simplex
        .neighbor_keys()?
        .position(|neighbor| neighbor == Some(source_simplex))
}

/// Increments a small vertex-incidence count buffer without allocating a hash map
/// for the tiny opposite-face sets used by inverse flip context builders.
fn increment_vertex_count(
    counts: &mut SmallBuffer<(VertexKey, usize), MAX_PRACTICAL_DIMENSION_SIZE>,
    vertex_key: VertexKey,
) {
    if let Some((_vertex, count)) = counts
        .iter_mut()
        .find(|(existing_vertex, _count)| *existing_vertex == vertex_key)
    {
        *count += 1;
    } else {
        counts.push((vertex_key, 1));
    }
}

/// Resolves a simplex key that came from a vertex incidence list.
///
/// A miss here means the maintained vertex-to-simplices index is stale, not
/// that the caller supplied an arbitrary missing simplex key.
fn simplex_from_vertex_incidence<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    vertex_key: VertexKey,
    simplex_key: SimplexKey,
) -> Result<&Simplex<V, D>, FlipError> {
    tds.simplex(simplex_key)
        .ok_or(FlipError::DanglingVertexIncidence {
            vertex_key,
            simplex_key,
        })
}
/// Converts borrowed edge-view validation errors into k=2 flip context errors.
///
/// [`build_k2_flip_context_from_edge`] exposes flip-specific error variants even
/// though it validates runtime edge handles through [`crate::core::edge::EdgeView`].
/// This mapping preserves caller-visible distinctions such as stale endpoints,
/// dangling vertex incidence, and invalid edge multiplicity.
fn flip_error_from_edge_key_error<const D: usize>(error: EdgeKeyError) -> FlipError {
    match error {
        EdgeKeyError::DuplicateEndpoint { endpoint } => {
            FlipEdgeAdjacencyError::DuplicateEndpoints {
                vertex_key: endpoint,
            }
            .into()
        }
        EdgeKeyError::MissingEndpoint { endpoint } => FlipError::MissingVertex {
            vertex_key: endpoint,
        },
        EdgeKeyError::EdgeNotFound { .. } => FlipError::InvalidEdgeMultiplicity {
            found: 0,
            expected: D,
        },
        EdgeKeyError::MissingEdgeIncidence { v0, v1 } => {
            FlipEdgeAdjacencyError::MissingEdgeIncidence { v0, v1 }.into()
        }
        EdgeKeyError::MissingVertexIncidence {
            vertex_key,
            simplex_key,
        } => FlipEdgeAdjacencyError::MissingVertexIncidence {
            vertex_key,
            simplex_key,
        }
        .into(),
        EdgeKeyError::DanglingVertexIncidence {
            vertex_key,
            simplex_key,
        } => FlipError::DanglingVertexIncidence {
            vertex_key,
            simplex_key,
        },
        EdgeKeyError::VertexIncidenceMismatch {
            simplex_key,
            vertex_key,
        } => FlipEdgeAdjacencyError::VertexIncidenceMismatch {
            vertex_key,
            simplex_key,
        }
        .into(),
    }
}

/// Build inverse k=2 flip context from an edge and its incident simplices.
///
/// # Errors
///
/// Returns a [`FlipError`] if the edge is invalid, references missing vertices/simplices,
/// or the adjacency data is inconsistent.
pub(crate) fn build_k2_flip_context_from_edge<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    edge: EdgeKey,
) -> Result<FlipContextDyn<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    if D < 3 {
        return Err(FlipError::UnsupportedDimension { dimension: D });
    }

    let edge_view = edge
        .view(tds)
        .map_err(flip_error_from_edge_key_error::<D>)?;
    let (v0, v1) = edge_view.endpoint_keys();
    let removed_simplices: SimplexKeyBuffer =
        edge_view.incident_simplices().iter().copied().collect();

    if removed_simplices.len() != D {
        return Err(FlipError::InvalidEdgeMultiplicity {
            found: removed_simplices.len(),
            expected: D,
        });
    }

    let mut counts: SmallBuffer<(VertexKey, usize), MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::new();
    for &simplex_key in &removed_simplices {
        let simplex = tds
            .simplex(simplex_key)
            .ok_or(FlipError::MissingSimplex { simplex_key })?;
        if !simplex.contains_vertex(v0) || !simplex.contains_vertex(v1) {
            return Err(FlipEdgeAdjacencyError::SimplexMissingEdgeVertices {
                simplex_key,
                v0,
                v1,
            }
            .into());
        }
        for &vk in simplex.vertices() {
            if vk != v0 && vk != v1 {
                increment_vertex_count(&mut counts, vk);
            }
        }
    }

    if counts.len() != D || !counts.iter().all(|(_vertex, count)| *count == D - 1) {
        return Err(FlipEdgeAdjacencyError::InvalidOppositeVertexIncidence {
            expected_vertices: D,
            found_vertices: counts.len(),
            expected_occurrences: D - 1,
        }
        .into());
    }

    let mut inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        counts.iter().map(|(vertex, _count)| *vertex).collect();
    inserted_face_vertices.sort_unstable_by_key(|v| v.data().as_ffi());

    let mut removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(2);
    removed_face_vertices.push(v0);
    removed_face_vertices.push(v1);

    Ok(FlipContextDyn {
        removed_face_vertices,
        inserted_face_vertices,
        removed_simplices,
        direction: FlipDirection::Inverse,
    })
}
/// Build a forward k=1 flip context from a simplex and inserted vertex.
fn build_k1_forward_context_from_simplex<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
    inserted_vertex: VertexKey,
) -> Result<FlipContext<D, 1>, FlipError>
where
    U: DataType,
    V: DataType,
{
    if D < 1 {
        return Err(FlipError::UnsupportedDimension { dimension: D });
    }

    let simplex = tds
        .simplex(simplex_key)
        .ok_or(FlipError::MissingSimplex { simplex_key })?;
    if tds.vertex(inserted_vertex).is_none() {
        return Err(FlipError::MissingVertex {
            vertex_key: inserted_vertex,
        });
    }

    let removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        simplex.vertices().iter().copied().collect();
    let mut inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(1);
    inserted_face_vertices.push(inserted_vertex);

    let removed_simplices: SimplexKeyBuffer = std::iter::once(simplex_key).collect();

    Ok(FlipContext {
        removed_face_vertices,
        inserted_face_vertices,
        removed_simplices,
        direction: FlipDirection::Forward,
    })
}

/// Build inverse k=1 flip context from a vertex and its incident simplices.
///
/// # Errors
///
/// Returns a [`FlipError`] if the vertex is missing, its incident simplex count is
/// not D+1, or the adjacency data is inconsistent.
pub(crate) fn build_k1_inverse_context<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    vertex_key: VertexKey,
) -> Result<FlipContextDyn<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    if D < 1 {
        return Err(FlipError::UnsupportedDimension { dimension: D });
    }

    if tds.vertex(vertex_key).is_none() {
        return Err(FlipError::MissingVertex { vertex_key });
    }

    let removed_simplices: SimplexKeyBuffer =
        tds.simplex_keys_containing_vertex(vertex_key).collect();
    for &simplex_key in &removed_simplices {
        simplex_from_vertex_incidence(tds, vertex_key, simplex_key)?;
    }

    let expected = D + 1;
    if removed_simplices.len() != expected {
        return Err(FlipError::InvalidVertexMultiplicity {
            found: removed_simplices.len(),
            expected,
        });
    }

    let mut counts: FastHashMap<VertexKey, usize> = FastHashMap::default();
    let mut removed_simplices_buf: SimplexKeyBuffer = SimplexKeyBuffer::new();
    for &simplex_key in &removed_simplices {
        let simplex = simplex_from_vertex_incidence(tds, vertex_key, simplex_key)?;
        if !simplex.contains_vertex(vertex_key) {
            return Err(FlipVertexAdjacencyError::SimplexMissingVertex {
                simplex_key,
                vertex_key,
            }
            .into());
        }
        removed_simplices_buf.push(simplex_key);
        for &vk in simplex.vertices() {
            if vk != vertex_key {
                *counts.entry(vk).or_insert(0) += 1;
            }
        }
    }

    if counts.len() != expected || !counts.values().all(|&count| count == D) {
        return Err(FlipVertexAdjacencyError::InvalidLinkVertexIncidence {
            expected_vertices: expected,
            found_vertices: counts.len(),
            expected_occurrences: D,
        }
        .into());
    }

    let mut inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        counts.keys().copied().collect();
    inserted_face_vertices.sort_unstable_by_key(|v| v.data().as_ffi());

    let mut removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(1);
    removed_face_vertices.push(vertex_key);

    Ok(FlipContextDyn {
        removed_face_vertices,
        inserted_face_vertices,
        removed_simplices: removed_simplices_buf,
        direction: FlipDirection::Inverse,
    })
}

/// Return whether source-simplex points are safe for the positive-oriented insphere path.
///
/// This helper exists so flip predicates only use
/// [`Kernel::in_sphere_positive_oriented`] when the actual point ordering is
/// provably positive-oriented. It returns `false` for synthetic/non-source simplices
/// and for geometries whose orientation cannot be certified by the f64 fast
/// filter, causing callers to use the full orientation-aware predicate instead.
#[inline]
fn source_simplex_is_certified_positive<const D: usize>(
    source_simplex: Option<SimplexKey>,
    points: &[Point<D>],
) -> bool {
    if source_simplex.is_none() {
        return false;
    }

    let known_positive =
        matches!(simplex_orientation_fast_filter_sign(points), Ok(Some(sign)) if sign > 0);
    known_positive
}

#[expect(
    clippy::too_many_arguments,
    reason = "local predicate evaluation threads topology, source simplices, and diagnostics explicitly"
)]
#[expect(
    clippy::too_many_lines,
    reason = "local predicate evaluation keeps frame alignment, diagnostics, and exact predicate calls together"
)]
/// Evaluate the k=2 facet flip predicate for a local Delaunay violation.
fn delaunay_violation_k2_for_facet<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    topology_model: &GlobalTopologyModelAdapter<D>,
    facet_vertices: &[VertexKey],
    opposite_a: VertexKey,
    opposite_b: VertexKey,
    source_simplices: &[SimplexKey],
    frame_simplex: Option<SimplexKey>,
    config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
) -> Result<bool, FlipError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    if facet_vertices.len() != D {
        return Err(FlipContextError::K2FacetArity {
            expected: D,
            found: facet_vertices.len(),
        }
        .into());
    }
    if facet_vertices.contains(&opposite_a)
        || facet_vertices.contains(&opposite_b)
        || opposite_a == opposite_b
    {
        return Err(FlipContextError::InvalidK2Opposites.into());
    }

    let mut simplex_vertices: [SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>; 2] = [
        SmallBuffer::with_capacity(D + 1),
        SmallBuffer::with_capacity(D + 1),
    ];
    for vertices in &mut simplex_vertices {
        vertices.extend_from_slice(facet_vertices);
    }
    simplex_vertices[0].push(opposite_a);
    simplex_vertices[1].push(opposite_b);

    // Sort by VertexKey for canonical SoS perturbation ordering
    simplex_vertices[0].sort_unstable_by_key(|v| v.data().as_ffi());
    simplex_vertices[1].sort_unstable_by_key(|v| v.data().as_ffi());

    let (
        points_a,
        points_b,
        opposite_point_a,
        opposite_point_b,
        positive_oriented_a,
        positive_oriented_b,
    ) = if matches!(topology_model, GlobalTopologyModelAdapter::Euclidean(_)) {
        let mut point_cache = EuclideanPointCache::new();
        let source_a = matching_source_simplex(tds, &simplex_vertices[0], source_simplices);
        let source_b = matching_source_simplex(tds, &simplex_vertices[1], source_simplices);
        let points_a = if let Some(source_simplex) = source_a {
            let simplex = tds
                .simplex(source_simplex)
                .ok_or(FlipError::MissingSimplex {
                    simplex_key: source_simplex,
                })?;
            point_cache.points_for_vertices(tds, simplex.vertices())?
        } else {
            point_cache.points_for_vertices(tds, &simplex_vertices[0])?
        };
        let points_b = if let Some(source_simplex) = source_b {
            let simplex = tds
                .simplex(source_simplex)
                .ok_or(FlipError::MissingSimplex {
                    simplex_key: source_simplex,
                })?;
            point_cache.points_for_vertices(tds, simplex.vertices())?
        } else {
            point_cache.points_for_vertices(tds, &simplex_vertices[1])?
        };
        let positive_oriented_a = source_simplex_is_certified_positive(source_a, &points_a);
        let positive_oriented_b = source_simplex_is_certified_positive(source_b, &points_b);
        (
            points_a,
            points_b,
            point_cache.point(tds, opposite_a)?,
            point_cache.point(tds, opposite_b)?,
            positive_oriented_a,
            positive_oriented_b,
        )
    } else {
        let source_a =
            matching_source_simplex(tds, &simplex_vertices[0], source_simplices).or(frame_simplex);
        let source_b =
            matching_source_simplex(tds, &simplex_vertices[1], source_simplices).or(frame_simplex);
        (
            vertices_to_points_with_optional_lift(
                tds,
                topology_model,
                &simplex_vertices[0],
                source_a,
                source_simplices,
            )?,
            vertices_to_points_with_optional_lift(
                tds,
                topology_model,
                &simplex_vertices[1],
                source_b,
                source_simplices,
            )?,
            vertex_point_lifted_into_simplex(
                tds,
                topology_model,
                opposite_a,
                source_b,
                source_simplices,
            )?,
            vertex_point_lifted_into_simplex(
                tds,
                topology_model,
                opposite_b,
                source_a,
                source_simplices,
            )?,
            false,
            false,
        )
    };
    let sphere_a = if positive_oriented_a {
        kernel.in_sphere_positive_oriented(&points_a, &opposite_point_b)
    } else {
        kernel.in_sphere(&points_a, &opposite_point_b)
    };
    let in_a = match sphere_a {
        Ok(value) => value,
        Err(e) => {
            diagnostics.record_predicate_failure();
            return Err(FlipPredicateError::coordinate_conversion(
                FlipPredicateOperation::K2SimplexAInSphere,
                e,
            )
            .into());
        }
    };

    let sphere_b = if positive_oriented_b {
        kernel.in_sphere_positive_oriented(&points_b, &opposite_point_a)
    } else {
        kernel.in_sphere(&points_b, &opposite_point_a)
    };
    let in_b = match sphere_b {
        Ok(value) => value,
        Err(e) => {
            diagnostics.record_predicate_failure();
            return Err(FlipPredicateError::coordinate_conversion(
                FlipPredicateOperation::K2SimplexBInSphere,
                e,
            )
            .into());
        }
    };

    // Record ambiguous sites when the predicate returns boundary/uncertain.
    if in_a == 0 {
        let key = predicate_key_from_vertices(&simplex_vertices[0], opposite_b);
        diagnostics.record_ambiguous(key);
    }

    if in_b == 0 {
        let key = predicate_key_from_vertices(&simplex_vertices[1], opposite_a);
        diagnostics.record_ambiguous(key);
    }

    let violates = in_a > 0 || in_b > 0;
    if env::var_os("DELAUNAY_REPAIR_DEBUG_PREDICATES").is_some()
        && (violates || in_a == 0 || in_b == 0)
    {
        tracing::debug!(
            facet_vertices = ?facet_vertices,
            opposite_a = ?opposite_a,
            opposite_b = ?opposite_b,
            in_a,
            in_b,
            violates,
            attempt = config.attempt,
            "delaunay_violation_k2_for_facet: insphere classification"
        );
    }

    Ok(violates)
}
/// Check whether a flip would create a degenerate (zero-volume) simplex.
///
/// Builds the replacement simplices from the given removed/inserted face vertices
/// and checks each with [`robust_orientation`].  Returns `Ok(true)` if any
/// replacement simplex is degenerate.
fn flip_would_create_degenerate_simplex<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    removed_face_vertices: &[VertexKey],
    inserted_face_vertices: &[VertexKey],
) -> Result<bool, FlipError>
where
    U: DataType,
    V: DataType,
{
    for &omit in removed_face_vertices {
        let mut vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(D + 1);
        vertices.extend_from_slice(inserted_face_vertices);
        for &v in removed_face_vertices {
            if v != omit {
                vertices.push(v);
            }
        }

        let points = vertices_to_points(tds, &vertices)?;
        // Use exact orientation (no SoS) so that truly degenerate simplices are
        // detected even when the kernel uses SoS.  Matches the pattern in
        // apply_bistellar_flip_with_k.
        match robust_orientation(&points) {
            Err(e) => {
                return Err(FlipPredicateError::coordinate_conversion(
                    FlipPredicateOperation::DegenerateSimplexPrecheck,
                    e,
                )
                .into());
            }
            Ok(Orientation::DEGENERATE) => return Ok(true),
            Ok(_) => {}
        }
    }

    Ok(false)
}

/// Check whether a k=2 flip would create a degenerate simplex.
fn k2_flip_would_create_degenerate_simplex<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    context: &FlipContext<D, 2>,
) -> Result<bool, FlipError>
where
    U: DataType,
    V: DataType,
{
    if context.inserted_face_vertices.len() != 2 {
        return Err(FlipContextError::WrongInsertedFaceArity {
            k_move: 2,
            expected: 2,
            found: context.inserted_face_vertices.len(),
        }
        .into());
    }

    flip_would_create_degenerate_simplex(
        tds,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
    )
}
/// Check whether a k=2 facet violates the local Delaunay condition.
///
/// # Errors
///
/// Returns a [`FlipError`] if any referenced simplex/vertex is missing or a predicate
/// evaluation fails.
fn is_delaunay_violation_k2<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    topology_model: &GlobalTopologyModelAdapter<D>,
    context: &FlipContext<D, 2>,
    config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
) -> Result<bool, FlipError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    if context.inserted_face_vertices.len() != 2 {
        return Err(FlipContextError::WrongInsertedFaceArity {
            k_move: 2,
            expected: 2,
            found: context.inserted_face_vertices.len(),
        }
        .into());
    }
    let opposite_a = context.inserted_face_vertices[0];
    let opposite_b = context.inserted_face_vertices[1];
    delaunay_violation_k2_for_facet(
        tds,
        kernel,
        topology_model,
        &context.removed_face_vertices,
        opposite_a,
        opposite_b,
        &context.removed_simplices,
        None,
        config,
        diagnostics,
    )
}

/// Apply a k=2 bistellar flip (no Delaunay check).
///
/// # Errors
///
/// Returns a [`FlipError`] if the flip would be degenerate, duplicate an existing simplex,
/// create non-manifold topology, if predicate evaluation fails, or if underlying TDS
/// mutations fail.
pub(crate) fn apply_bistellar_flip_k2<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    context: &FlipContext<D, 2>,
) -> Result<FlipInfo<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    apply_bistellar_flip::<U, V, D, 2>(tds, context)
}

/// Build flip context for a k=3 (ridge) flip.
///
/// # Errors
///
/// Returns a [`FlipError`] if the ridge is invalid, references missing simplices/vertices,
/// or the adjacency data is inconsistent.
pub(crate) fn build_k3_flip_context<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    ridge: RidgeHandle,
) -> Result<FlipContext<D, 3>, FlipError>
where
    U: DataType,
    V: DataType,
{
    build_k3_flip_context_with_star_limit(tds, ridge, None)
}

/// Builds k=3 repair context only for true three-simplex ridge stars.
fn build_k3_flip_context_for_repair<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    ridge: RidgeHandle,
) -> Result<FlipContext<D, 3>, FlipError>
where
    U: DataType,
    V: DataType,
{
    build_k3_flip_context_with_star_limit(tds, ridge, Some(3))
}

/// Builds k=3 flip context while optionally rejecting ridge stars above a caller limit.
fn build_k3_flip_context_with_star_limit<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    ridge: RidgeHandle,
    max_simplices: Option<usize>,
) -> Result<FlipContext<D, 3>, FlipError>
where
    U: DataType,
    V: DataType,
{
    if D < 3 {
        return Err(FlipError::UnsupportedDimension { dimension: D });
    }

    let simplex_key = ridge.simplex_key();
    let simplex = tds
        .simplex(simplex_key)
        .ok_or(FlipError::MissingSimplex { simplex_key })?;

    let vertex_count = simplex.number_of_vertices();
    let omit_a = usize::from(ridge.omit_a());
    let omit_b = usize::from(ridge.omit_b());
    if omit_a >= vertex_count || omit_b >= vertex_count || omit_a == omit_b {
        return Err(FlipError::InvalidRidgeIndex {
            simplex_key,
            omit_a: ridge.omit_a(),
            omit_b: ridge.omit_b(),
            vertex_count,
        });
    }

    let ridge_vertices = ridge_vertices_from_simplex(simplex, omit_a, omit_b);
    if ridge_vertices.len() != D - 1 {
        return Err(FlipError::InvalidRidgeAdjacency { simplex_key });
    }

    let simplices =
        collect_simplices_around_ridge(tds, simplex_key, &ridge_vertices, max_simplices)?;
    if simplices.len() != 3 {
        return Err(FlipError::InvalidRidgeMultiplicity {
            found: simplices.len(),
        });
    }

    // k=3 flip contexts are tiny (exactly 3 simplices, with 2 "extra" vertices per simplex).
    // Use flat buffers + linear counting to avoid HashMap/Vec overhead in this hot path.
    let mut opposite_counts: SmallBuffer<(VertexKey, u8), 3> = SmallBuffer::new();
    let mut extras_per_simplex: SmallBuffer<[VertexKey; 2], 3> = SmallBuffer::new();

    for &ck in &simplices {
        let simplex = tds
            .simplex(ck)
            .ok_or(FlipError::MissingSimplex { simplex_key: ck })?;
        let extras = simplex_extras_for_ridge(ck, simplex, &ridge_vertices)?;
        if extras.len() != 2 {
            return Err(FlipError::InvalidRidgeAdjacency { simplex_key: ck });
        }

        let extras_pair: [VertexKey; 2] = extras
            .as_slice()
            .try_into()
            .map_err(|_| FlipError::InvalidRidgeAdjacency { simplex_key: ck })?;

        for &v in &extras_pair {
            if let Some((_key, count)) = opposite_counts.iter_mut().find(|(key, _)| *key == v) {
                *count += 1;
            } else {
                opposite_counts.push((v, 1));
            }
        }

        extras_per_simplex.push(extras_pair);
    }

    if opposite_counts.len() != 3 || !opposite_counts.iter().all(|(_v, count)| *count == 2) {
        return Err(FlipError::InvalidRidgeAdjacency { simplex_key });
    }

    let mut opposite_vertices: SmallBuffer<VertexKey, 3> =
        opposite_counts.iter().map(|(v, _count)| *v).collect();
    opposite_vertices.sort_unstable();
    let opposite_vertices: [VertexKey; 3] = opposite_vertices
        .as_slice()
        .try_into()
        .map_err(|_| FlipError::InvalidRidgeAdjacency { simplex_key })?;

    for extras in &extras_per_simplex {
        let _missing = missing_opposite_for_simplex(extras, &opposite_vertices)
            .ok_or(FlipError::InvalidRidgeAdjacency { simplex_key })?;
    }

    let mut inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(3);
    inserted_face_vertices.extend(opposite_vertices);

    Ok(FlipContext {
        removed_face_vertices: ridge_vertices,
        inserted_face_vertices,
        removed_simplices: simplices,
        direction: FlipDirection::Forward,
    })
}

/// Build inverse k=3 flip context from a triangle and its incident simplices.
///
/// # Errors
///
/// Returns a [`FlipError`] if the triangle is invalid, references missing vertices/simplices,
/// or the adjacency data is inconsistent.
pub(crate) fn build_k3_flip_context_from_triangle<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    triangle: TriangleHandle,
) -> Result<FlipContextDyn<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    if D < 4 {
        return Err(FlipError::UnsupportedDimension { dimension: D });
    }

    let [a, b, c] = triangle.vertices();
    if tds.vertex(a).is_none() {
        return Err(FlipError::MissingVertex { vertex_key: a });
    }
    if tds.vertex(b).is_none() {
        return Err(FlipError::MissingVertex { vertex_key: b });
    }
    if tds.vertex(c).is_none() {
        return Err(FlipError::MissingVertex { vertex_key: c });
    }

    let mut removed_simplices: SimplexKeyBuffer = SimplexKeyBuffer::new();
    for simplex_key in tds.simplex_keys_containing_vertex(a) {
        let simplex = simplex_from_vertex_incidence(tds, a, simplex_key)?;
        if simplex.contains_vertex(b) && simplex.contains_vertex(c) {
            removed_simplices.push(simplex_key);
        }
    }

    let expected = D - 1;
    if removed_simplices.len() != expected {
        return Err(FlipError::InvalidTriangleMultiplicity {
            found: removed_simplices.len(),
            expected,
        });
    }

    let mut counts: SmallBuffer<(VertexKey, usize), MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::new();
    for &simplex_key in &removed_simplices {
        let simplex = tds
            .simplex(simplex_key)
            .ok_or(FlipError::MissingSimplex { simplex_key })?;
        if !simplex.contains_vertex(a) || !simplex.contains_vertex(b) || !simplex.contains_vertex(c)
        {
            return Err(FlipTriangleAdjacencyError::SimplexMissingTriangleVertices {
                simplex_key,
                a,
                b,
                c,
            }
            .into());
        }
        for &vk in simplex.vertices() {
            if vk != a && vk != b && vk != c {
                increment_vertex_count(&mut counts, vk);
            }
        }
    }

    if counts.len() != expected || !counts.iter().all(|(_vertex, count)| *count == expected - 1) {
        return Err(FlipTriangleAdjacencyError::InvalidRidgeVertexIncidence {
            expected_vertices: expected,
            found_vertices: counts.len(),
            expected_occurrences: expected - 1,
        }
        .into());
    }

    let mut inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        counts.iter().map(|(vertex, _count)| *vertex).collect();
    inserted_face_vertices.sort_unstable_by_key(|v| v.data().as_ffi());

    let mut removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(3);
    removed_face_vertices.push(a);
    removed_face_vertices.push(b);
    removed_face_vertices.push(c);

    Ok(FlipContextDyn {
        removed_face_vertices,
        inserted_face_vertices,
        removed_simplices,
        direction: FlipDirection::Inverse,
    })
}
#[expect(
    clippy::too_many_arguments,
    reason = "Local predicate evaluation threads topology, source simplices, and diagnostics explicitly"
)]
/// Evaluate the k=3 ridge flip predicate for a local Delaunay violation.
fn delaunay_violation_k3_for_ridge<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    topology_model: &GlobalTopologyModelAdapter<D>,
    ridge_vertices: &[VertexKey],
    triangle_vertices: &[VertexKey],
    source_simplices: &[SimplexKey],
    frame_simplex: Option<SimplexKey>,
    _config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
) -> Result<bool, FlipError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    if triangle_vertices.len() != 3 {
        return Err(FlipContextError::WrongInsertedFaceArity {
            k_move: 3,
            expected: 3,
            found: triangle_vertices.len(),
        }
        .into());
    }
    if ridge_vertices.len() != D.saturating_sub(1) {
        return Err(FlipContextError::K3RidgeArity {
            expected: D.saturating_sub(1),
            found: ridge_vertices.len(),
        }
        .into());
    }

    let is_euclidean_topology = matches!(topology_model, GlobalTopologyModelAdapter::Euclidean(_));
    let mut euclidean_point_cache = EuclideanPointCache::new();

    for &missing in triangle_vertices {
        let mut simplex_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(D + 1);
        simplex_vertices.extend_from_slice(ridge_vertices);
        for &v in triangle_vertices {
            if v != missing {
                simplex_vertices.push(v);
            }
        }

        // Sort by VertexKey for canonical SoS perturbation ordering
        simplex_vertices.sort_unstable_by_key(|v| v.data().as_ffi());

        let (points, missing_point, positive_oriented) = if is_euclidean_topology {
            let source_simplex = matching_source_simplex(tds, &simplex_vertices, source_simplices);
            let points = if let Some(source_simplex) = source_simplex {
                let simplex = tds
                    .simplex(source_simplex)
                    .ok_or(FlipError::MissingSimplex {
                        simplex_key: source_simplex,
                    })?;
                euclidean_point_cache.points_for_vertices(tds, simplex.vertices())?
            } else {
                euclidean_point_cache.points_for_vertices(tds, &simplex_vertices)?
            };
            let positive_oriented = source_simplex_is_certified_positive(source_simplex, &points);
            (
                points,
                euclidean_point_cache.point(tds, missing)?,
                positive_oriented,
            )
        } else {
            let source_simplex =
                matching_source_simplex(tds, &simplex_vertices, source_simplices).or(frame_simplex);
            (
                vertices_to_points_with_optional_lift(
                    tds,
                    topology_model,
                    &simplex_vertices,
                    source_simplex,
                    source_simplices,
                )?,
                vertex_point_lifted_into_simplex(
                    tds,
                    topology_model,
                    missing,
                    source_simplex,
                    source_simplices,
                )?,
                false,
            )
        };

        let in_sphere_result = if positive_oriented {
            kernel.in_sphere_positive_oriented(&points, &missing_point)
        } else {
            kernel.in_sphere(&points, &missing_point)
        };
        let in_sphere = match in_sphere_result {
            Ok(value) => value,
            Err(e) => {
                diagnostics.record_predicate_failure();
                return Err(FlipPredicateError::coordinate_conversion(
                    FlipPredicateOperation::K3SimplexInSphere,
                    e,
                )
                .into());
            }
        };

        // Track ambiguous sites when the fast predicate returns boundary/uncertain.
        if in_sphere == 0 {
            let key = predicate_key_from_vertices(&simplex_vertices, missing);
            diagnostics.record_ambiguous(key);
        }

        if in_sphere > 0 {
            return Ok(true);
        }
    }

    Ok(false)
}

/// Apply a k=3 bistellar flip (no Delaunay check).
///
/// # Errors
///
/// Returns a [`FlipError`] if the flip would be degenerate, duplicate an existing simplex,
/// create non-manifold topology, if predicate evaluation fails, or if underlying TDS
/// mutations fail.
pub(crate) fn apply_bistellar_flip_k3<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    context: &FlipContext<D, 3>,
) -> Result<FlipInfo<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    apply_bistellar_flip::<U, V, D, 3>(tds, context)
}

/// Apply a forward k=1 move (simplex split) by inserting a new vertex.
///
/// # Errors
///
/// Returns a [`FlipError`] if the simplex is missing, the vertex cannot be inserted,
/// or the flip would be degenerate.
pub(crate) fn apply_bistellar_flip_k1<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    simplex_key: SimplexKey,
    vertex: Vertex<U, D>,
) -> Result<FlipInfo<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    if D < 1 {
        return Err(FlipError::UnsupportedDimension { dimension: D });
    }

    let vertex_key = tds.insert_vertex_with_mapping(vertex).map_err(|source| {
        FlipMutationError::VertexInsertion {
            source: source.into(),
        }
    })?;

    let context = match build_k1_forward_context_from_simplex(tds, simplex_key, vertex_key) {
        Ok(ctx) => ctx,
        Err(e) => {
            // Remove the just-inserted vertex to avoid leaving an orphan.
            let _ = tds.remove_vertex(vertex_key);
            return Err(e);
        }
    };

    let result = apply_bistellar_flip::<U, V, D, 1>(tds, &context);

    if result.is_err() {
        let _ = tds.remove_vertex(vertex_key);
    }

    result
}

/// Apply an inverse k=1 move (vertex collapse) by removing a vertex whose star
/// is a simplex.
///
/// # Errors
///
/// Returns a [`FlipError`] if the vertex star is invalid or the flip would be degenerate.
pub(crate) fn apply_bistellar_flip_k1_inverse<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    vertex_key: VertexKey,
) -> Result<FlipInfo<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    if D < 1 {
        return Err(FlipError::UnsupportedDimension { dimension: D });
    }

    let context = build_k1_inverse_context(tds, vertex_key)?;
    let info = apply_bistellar_flip_dynamic(tds, D + 1, &context)?;

    let _ = tds.remove_vertex(vertex_key);

    Ok(info)
}

/// Repair Delaunay violations using a k=2 flip queue.
///
/// # Errors
///
/// Returns a [`DelaunayRepairError`] if the repair fails to converge or an underlying
/// flip operation encounters an unrecoverable error.
#[expect(
    clippy::too_many_lines,
    reason = "Repair loop contains inline tracing and queue handling for diagnostics"
)]
fn repair_delaunay_with_flips_k2_attempt<K, U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    kernel: &K,
    seed_simplices: Option<&[SimplexKey]>,
    config: &RepairAttemptConfig,
) -> Result<RepairAttemptOutcome, DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    if D < 2 {
        return Err(FlipError::UnsupportedDimension { dimension: D }.into());
    }

    let max_flips = config
        .max_flips_override
        .unwrap_or_else(|| default_max_flips::<D>(tds.number_of_simplices()));

    let mut stats = DelaunayRepairStats::default();
    let mut diagnostics = RepairDiagnostics::default();
    let mut queue: VecDeque<(FacetHandle, u64)> = VecDeque::new();
    let mut queued: FastHashSet<u64> = FastHashSet::default();
    let mut facet_handles: FastHashMap<u64, FacetHandle> = FastHashMap::default();
    let mut last_applied_flip: Option<LastAppliedFlip> = None;
    let mut touched_simplices = SimplexKeyBuffer::new();
    let mut touched_simplex_set = FastHashSet::<SimplexKey>::default();
    let used_full_reseed = seed_simplices.is_none();
    let topology_model = GlobalTopology::DEFAULT.model();

    if let Some(seeds) = seed_simplices {
        for &simplex_key in seeds {
            enqueue_simplex_facets(
                tds,
                simplex_key,
                &mut queue,
                &mut queued,
                &mut facet_handles,
                &mut stats,
            )?;
        }
    } else {
        for facet in AllFacetsIter::try_new(tds)? {
            let facet = facet?;
            let handle = FacetHandle::from_validated(facet.simplex_key(), facet.facet_index());
            enqueue_facet(
                tds,
                handle,
                &mut queue,
                &mut queued,
                &mut facet_handles,
                &mut stats,
            );
        }
    }
    if repair_trace_enabled() {
        let seed_count = seed_simplices.map_or(0, <[SimplexKey]>::len);
        tracing::debug!(
            "[repair] attempt={} order={:?} simplices={} max_flips={} seeds={} queues(facet={})",
            config.attempt,
            config.queue_order,
            tds.number_of_simplices(),
            max_flips,
            seed_count,
            queue.len(),
        );
    }

    while let Some((facet, key)) = pop_queue(&mut queue, config.queue_order) {
        queued.remove(&key);
        let facet = facet_handles.remove(&key).unwrap_or(facet);
        let Some(facet) = resolve_facet_handle_for_key(tds, facet, key) else {
            continue;
        };
        stats.facets_checked += 1;

        let context = match build_k2_flip_context(tds, facet) {
            Ok(ctx) => ctx,
            Err(
                FlipError::BoundaryFacet { .. }
                | FlipError::MissingSimplex { .. }
                | FlipError::MissingNeighbor { .. }
                | FlipError::InvalidFacetAdjacency { .. }
                | FlipError::InvalidFacetIndex { .. },
            ) => {
                continue;
            }
            Err(e) => return Err(e.into()),
        };

        let violates = match is_delaunay_violation_k2(
            tds,
            kernel,
            &topology_model,
            &context,
            config,
            &mut diagnostics,
        ) {
            Ok(violates) => violates,
            Err(FlipError::PredicateFailure { .. }) => {
                continue;
            }
            Err(e) => return Err(e.into()),
        };

        if !violates {
            continue;
        }
        diagnostics.record_applicable_repair_site();

        let kind = BistellarFlipKind::k2(D);
        let signature = flip_signature(
            kind,
            context.direction,
            &context.removed_face_vertices,
            &context.inserted_face_vertices,
        );
        check_flip_cycle(
            tds,
            FlipCycleContext::from_validated_flip(
                signature,
                kind,
                context.direction,
                &context.removed_face_vertices,
                &context.inserted_face_vertices,
            ),
            &mut diagnostics,
            &stats,
            max_flips,
            config,
        )?;

        // Enforce flip budget before applying the flip so that Some(0) means zero flips.
        if stats.flips_performed >= max_flips {
            return Err(non_convergent_error(
                max_flips,
                &stats,
                &diagnostics,
                config,
            ));
        }

        let applied = match apply_delaunay_flip_k2(tds, &context) {
            Ok(applied) => applied,
            Err(
                err @ (FlipError::DegenerateSimplex
                | FlipError::NegativeOrientation { .. }
                | FlipError::DuplicateSimplex
                | FlipError::NonManifoldFacet
                | FlipError::InsertedSimplexAlreadyExists { .. }
                | FlipError::SimplexCreation(_)),
            ) => {
                if env::var_os("DELAUNAY_REPAIR_DEBUG_FACETS").is_some() {
                    tracing::debug!(
                        "k=2 flip skipped in repair_delaunay_with_flips_k2_attempt (facet={facet:?}): {err}"
                    );
                }
                if repair_trace_enabled() {
                    tracing::debug!("[repair] skip k=2 flip (facet={facet:?}) reason={err}");
                    tracing::debug!(
                        "[repair] skip k=2 flip context removed_face={:?} inserted_face={:?} removed_simplices={:?}",
                        context.removed_face_vertices,
                        context.inserted_face_vertices,
                        context.removed_simplices,
                    );
                }
                continue;
            }
            Err(e) => return Err(e.into()),
        };
        stats.flips_performed += 1;
        diagnostics.record_flip_signature(signature);
        last_applied_flip = Some(LastAppliedFlip::from_applied_flip(&applied));
        let info = applied.info;
        record_touched_simplices(
            &mut touched_simplices,
            &mut touched_simplex_set,
            &info.new_simplices,
        );

        for &simplex_key in &info.new_simplices {
            enqueue_simplex_facets(
                tds,
                simplex_key,
                &mut queue,
                &mut queued,
                &mut facet_handles,
                &mut stats,
            )?;
        }
    }
    if repair_trace_enabled() {
        tracing::debug!(
            "[repair] attempt={} done: checked={} flips={} max_queue={} ambiguous={} predicate_failures={} cycles={}",
            config.attempt,
            stats.facets_checked,
            stats.flips_performed,
            stats.max_queue_len,
            diagnostics.ambiguous_predicates,
            diagnostics.predicate_failures,
            diagnostics.cycle_detections,
        );
    }
    emit_repair_debug_summary("attempt_done", &stats, &diagnostics, config, max_flips);

    Ok(RepairAttemptOutcome {
        postcondition_required: repair_postcondition_required(&stats, &diagnostics),
        stats,
        last_applied_flip,
        touched_simplices,
        used_full_reseed,
    })
}

/// Repair Delaunay violations using k=2 queues, k=3 queues in 3D,
/// and inverse edge/triangle queues in higher dimensions.
///
/// # Errors
///
/// Returns a [`DelaunayRepairError`] if the repair fails to converge or an underlying
/// flip operation encounters an unrecoverable error.
pub(crate) fn repair_delaunay_with_flips_k2_k3<K, U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    kernel: &K,
    seed_simplices: Option<&[SimplexKey]>,
    topology: TopologyGuarantee,
    max_flips_override: Option<usize>,
) -> Result<DelaunayRepairStats, DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    repair_delaunay_with_flips_k2_k3_run(tds, kernel, seed_simplices, topology, max_flips_override)
        .map(|run| run.stats)
}

fn run_full_reseed_retry<K, U, V, const D: usize>(
    transaction: &mut TdsRollbackTransaction<'_, U, V, D>,
    kernel: &K,
    config: &RepairAttemptConfig,
) -> Result<DelaunayRepairRun, DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    transaction.restore();
    let retry_seed_simplices = None;
    let attempt_result = if D == 2 {
        repair_delaunay_with_flips_k2_attempt(
            transaction.tds_mut(),
            kernel,
            retry_seed_simplices,
            config,
        )
    } else {
        repair_delaunay_with_flips_k2_k3_attempt(
            transaction.tds_mut(),
            kernel,
            retry_seed_simplices,
            config,
        )
    };

    let outcome = attempt_result?;
    verify_repair_postcondition(
        transaction.tds_mut(),
        kernel,
        retry_seed_simplices,
        outcome.last_applied_flip.as_ref(),
    )?;
    Ok(repair_run_from_attempt(outcome))
}

/// Repair Delaunay violations and return the final validation frontier.
///
/// # Errors
///
/// Returns a [`DelaunayRepairError`] if the repair fails to converge or an underlying
/// flip operation encounters an unrecoverable error.
pub(crate) fn repair_delaunay_with_flips_k2_k3_run<K, U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    kernel: &K,
    seed_simplices: Option<&[SimplexKey]>,
    topology: TopologyGuarantee,
    max_flips_override: Option<usize>,
) -> Result<DelaunayRepairRun, DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    if D < 2 {
        return Err(FlipError::UnsupportedDimension { dimension: D }.into());
    }

    let operation = TopologicalOperation::FacetFlip;
    if !operation.is_admissible_under(topology) {
        return Err(DelaunayRepairError::InvalidTopology {
            required: operation.required_topology(),
            found: topology,
            message: "flip-based Delaunay repair requires admissible topology",
        });
    }

    // Two-attempt strategy: FIFO then LIFO queue ordering.
    // Predicate correctness depends on the caller supplying a kernel with
    // exact predicates (e.g. `AdaptiveKernel` or `RobustKernel`);
    // the retry exists only to escape queue-order-dependent flip cycles.
    let attempt1 = RepairAttemptConfig {
        attempt: 1,
        queue_order: RepairQueueOrder::Fifo,
        max_flips_override,
    };

    let attempt2 = RepairAttemptConfig {
        attempt: 2,
        queue_order: RepairQueueOrder::Lifo,
        max_flips_override,
    };

    // Snapshot the pre-repair state so a failed attempt doesn't poison retries.
    let mut transaction = TdsRollbackTransaction::begin(tds);

    let attempt1_result = if D == 2 {
        repair_delaunay_with_flips_k2_attempt(
            transaction.tds_mut(),
            kernel,
            seed_simplices,
            &attempt1,
        )
    } else {
        repair_delaunay_with_flips_k2_k3_attempt(
            transaction.tds_mut(),
            kernel,
            seed_simplices,
            &attempt1,
        )
    };

    match attempt1_result {
        Ok(outcome) => {
            if verify_repair_postcondition(
                transaction.tds_mut(),
                kernel,
                seed_simplices,
                outcome.last_applied_flip.as_ref(),
            )
            .is_ok()
            {
                let run = repair_run_from_attempt(outcome);
                transaction.commit();
                return Ok(run);
            }
            if repair_trace_enabled() {
                tracing::debug!(
                    "[repair] attempt 1 postcondition failed; retrying with LIFO + full reseed"
                );
            }
        }
        Err(DelaunayRepairError::NonConvergent { .. }) => {
            if repair_trace_enabled() {
                tracing::debug!(
                    "[repair] attempt 1 non-convergent; retrying with LIFO + full reseed"
                );
            }
        }
        Err(err) => {
            transaction.rollback();
            return Err(err);
        }
    }

    // Retry with LIFO + full reseed.
    match run_full_reseed_retry(&mut transaction, kernel, &attempt2) {
        Ok(run) => {
            transaction.commit();
            Ok(run)
        }
        Err(err) => {
            transaction.rollback();
            Err(err)
        }
    }
}

/// Run a seeded, bounded Delaunay repair capped to a specific set of simplices.
///
/// Unlike [`repair_delaunay_with_flips_k2_k3`], this function normally reseeds from the
/// provided `seed_simplices` rather than `None` / all simplices. This keeps the queue size
/// bounded to `O(seed_simplices × queues_per_simplex)` regardless of the total triangulation size,
/// which is critical for D≥4 where a full-triangulation seed would generate O(simplices×30)
/// items (prohibitively expensive with robust predicates). An explicit empty seed slice
/// is a bounded no-op seed set; callers that want a whole-TDS repair pass `None`.
///
/// Two attempts are made with alternating queue orders (FIFO → LIFO) to escape
/// flip cycles — the same strategy as [`repair_delaunay_with_flips_k2_k3`], but without the
/// `None`-reseed fallback.  A TDS snapshot is taken so that a failed attempt does not
/// leave the triangulation partially modified.
///
/// It is designed for per-insertion bulk construction and for the final bounded pass in
/// `finalize_bulk_construction`.  On non-convergence after both attempts the caller
/// should soft-fail and record the seed simplices for a subsequent repair pass, or let
/// `build_with_shuffled_retries` try a different vertex ordering.
///
/// `max_flips` is the per-attempt flip budget; use a seed-proportional value, e.g.
/// `(seed_simplices.len() * (D + 1) * 8).max(64)` for D ≥ 4.
///
/// # Errors
///
/// Returns [`DelaunayRepairError::NonConvergent`] if both attempts fail to converge.
/// Other errors (topology violations, predicate failures) are forwarded as-is.
pub(crate) fn repair_delaunay_local_single_pass<K, U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    kernel: &K,
    seed_simplices: &[SimplexKey],
    max_flips: usize,
) -> Result<DelaunayRepairStats, DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    repair_delaunay_local_single_pass_timed(tds, kernel, seed_simplices, max_flips, None)
}

/// Run a seeded, bounded repair pass while reporting phase timing to the caller.
#[expect(
    clippy::too_many_lines,
    reason = "bounded two-attempt repair keeps rollback, retry, and postcondition timing together"
)]
pub(crate) fn repair_delaunay_local_single_pass_timed<K, U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    kernel: &K,
    seed_simplices: &[SimplexKey],
    max_flips: usize,
    mut timing: Option<&mut LocalRepairPhaseTiming>,
) -> Result<DelaunayRepairStats, DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    let mut phase_timing = LocalRepairPhaseTiming::default();
    // Two-attempt strategy: FIFO then LIFO queue ordering.
    // Predicate correctness depends on the caller supplying a kernel with
    // exact predicates (e.g. `AdaptiveKernel` or `RobustKernel`);
    // the retry exists only to escape queue-order-dependent flip cycles.
    let attempt1 = RepairAttemptConfig {
        attempt: 1,
        queue_order: RepairQueueOrder::Fifo,
        max_flips_override: Some(max_flips),
    };
    let attempt2 = RepairAttemptConfig {
        attempt: 2,
        queue_order: RepairQueueOrder::Lifo,
        max_flips_override: Some(max_flips),
    };
    // Snapshot so a failed attempt does not leave the TDS in a partially-modified state.
    let snapshot_started = Instant::now();
    let mut transaction = TdsRollbackTransaction::begin(tds);
    phase_timing.record_snapshot(snapshot_started.elapsed());

    let attempt_started = Instant::now();
    let attempt1_result = if D == 2 {
        repair_delaunay_with_flips_k2_attempt(
            transaction.tds_mut(),
            kernel,
            Some(seed_simplices),
            &attempt1,
        )
    } else {
        repair_delaunay_with_flips_k2_k3_attempt_timed(
            transaction.tds_mut(),
            kernel,
            Some(seed_simplices),
            &attempt1,
            Some(&mut phase_timing),
        )
    };
    phase_timing.record_attempt(attempt_started.elapsed());

    match attempt1_result {
        Ok(outcome) => {
            // D>=4 bulk construction uses local repair as a bounded stabilizer
            // and performs strict final validation after construction. Replaying
            // the same local queues after every successful repair adds quadratic
            // predicate work without strengthening the final correctness gate.
            if !outcome.postcondition_required || D >= 4 {
                let stats = outcome.stats;
                transaction.commit();
                publish_local_repair_phase_timing(&mut timing, phase_timing);
                return Ok(stats);
            }
            let postcondition_frontier =
                local_postcondition_frontier(seed_simplices, &outcome.touched_simplices);
            let postcondition_started = Instant::now();
            let postcondition_result = verify_local_repair_postcondition(
                transaction.tds_mut(),
                kernel,
                &postcondition_frontier,
                outcome.last_applied_flip.as_ref(),
            );
            phase_timing.record_postcondition(postcondition_started.elapsed());
            if postcondition_result.is_ok() {
                let stats = outcome.stats;
                transaction.commit();
                publish_local_repair_phase_timing(&mut timing, phase_timing);
                return Ok(stats);
            }
            if repair_trace_enabled() {
                tracing::debug!("[repair] local attempt 1 postcondition failed; retrying LIFO");
            }
        }
        Err(DelaunayRepairError::NonConvergent { .. }) => {
            if repair_trace_enabled() {
                tracing::debug!("[repair] local attempt 1 non-convergent; retrying LIFO");
            }
        }
        Err(err) => {
            let restore_started = Instant::now();
            transaction.rollback();
            phase_timing.record_restore(restore_started.elapsed());
            publish_local_repair_phase_timing(&mut timing, phase_timing);
            return Err(err);
        }
    }
    let restore_started = Instant::now();
    transaction.restore();
    phase_timing.record_restore(restore_started.elapsed());

    let attempt_started = Instant::now();
    let attempt2_result = if D == 2 {
        repair_delaunay_with_flips_k2_attempt(
            transaction.tds_mut(),
            kernel,
            Some(seed_simplices),
            &attempt2,
        )
    } else {
        repair_delaunay_with_flips_k2_k3_attempt_timed(
            transaction.tds_mut(),
            kernel,
            Some(seed_simplices),
            &attempt2,
            Some(&mut phase_timing),
        )
    };
    phase_timing.record_attempt(attempt_started.elapsed());

    match attempt2_result {
        Ok(outcome) => {
            // See attempt 1: D>=4 local postconditions are deferred to the
            // construction finalization/validation path.
            if !outcome.postcondition_required || D >= 4 {
                let stats = outcome.stats;
                transaction.commit();
                publish_local_repair_phase_timing(&mut timing, phase_timing);
                return Ok(stats);
            }
            let postcondition_frontier =
                local_postcondition_frontier(seed_simplices, &outcome.touched_simplices);
            let postcondition_started = Instant::now();
            let postcondition_result = verify_local_repair_postcondition(
                transaction.tds_mut(),
                kernel,
                &postcondition_frontier,
                outcome.last_applied_flip.as_ref(),
            );
            phase_timing.record_postcondition(postcondition_started.elapsed());
            match postcondition_result {
                Ok(()) => {
                    let stats = outcome.stats;
                    transaction.commit();
                    publish_local_repair_phase_timing(&mut timing, phase_timing);
                    Ok(stats)
                }
                Err(verifier_err) => {
                    // Postcondition failed: restore the TDS so callers that
                    // soft-fail receive a structurally valid triangulation.
                    let restore_started = Instant::now();
                    transaction.rollback();
                    phase_timing.record_restore(restore_started.elapsed());
                    publish_local_repair_phase_timing(&mut timing, phase_timing);
                    Err(verifier_err)
                }
            }
        }
        Err(err) => {
            // On failure, restore the TDS to the pre-repair snapshot so callers that
            // soft-fail (e.g. D≥4 bulk construction) receive a structurally valid
            // triangulation rather than a partially-modified one.
            let restore_started = Instant::now();
            transaction.rollback();
            phase_timing.record_restore(restore_started.elapsed());
            publish_local_repair_phase_timing(&mut timing, phase_timing);
            Err(err)
        }
    }
}

/// Verify the Delaunay property via local flip predicates (fast O(simplices) validation).
///
/// This function checks whether the triangulation satisfies the Delaunay property by testing
/// all possible flip configurations (k=2 facets, k=3 ridges, and their inverses). If no
/// violations are detected via these local checks, the triangulation is Delaunay.
///
/// This is **much faster** than the naive O(simplices × vertices) empty-circumsphere check,
/// while being equally correct due to the completeness of bistellar flip predicates.
///
/// # Performance
///
/// - **Complexity**: O(simplices) — tests only local flip predicates
/// - **Speedup**: ~40-100x faster than brute-force for typical triangulations
/// - **Use case**: Ideal for property-based testing with many iterations
///
/// # Errors
///
/// Returns [`DelaunayRepairError::PostconditionFailed`] if any flip predicate detects
/// a Delaunay violation, or [`DelaunayRepairError::VerificationFailed`] if a
/// local predicate cannot be evaluated.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::*;
/// use delaunay::prelude::repair::verify_delaunay_via_flip_predicates;
/// use delaunay::prelude::geometry::AdaptiveKernel;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Source(#[from] DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
///
/// let dt: DelaunayTriangulation<_, (), (), 3> =
///     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
/// let kernel = AdaptiveKernel::<f64>::new();
///
/// // Fast O(N) verification
/// assert!(verify_delaunay_via_flip_predicates(dt.tds(), &kernel).is_ok());
/// # Ok(())
/// # }
/// ```
pub fn verify_delaunay_via_flip_predicates<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
) -> Result<(), DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    verify_delaunay_with_topology(tds, kernel, GlobalTopology::DEFAULT)
}

/// Verify the Delaunay property via local flip predicates for a full triangulation.
///
/// This is the preferred Level 5 validation entry point because it carries the
/// triangulation's global topology alongside the TDS.  For periodic topologies
/// (e.g. toroidal), insphere predicates are evaluated in lifted coordinates so
/// that facets spanning periodic boundaries are not reported as false violations.
///
/// # Errors
///
/// Returns [`DelaunayRepairError::PostconditionFailed`] if any flip predicate detects
/// a Delaunay violation, or [`DelaunayRepairError::VerificationFailed`] if
/// verification cannot evaluate the local predicates.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::*;
/// use delaunay::prelude::repair::verify_delaunay_for_triangulation;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Source(#[from] DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
///
/// let dt: DelaunayTriangulation<_, (), (), 3> =
///     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
///
/// // Topology-aware O(N) verification
/// assert!(verify_delaunay_for_triangulation(dt.as_triangulation()).is_ok());
/// # Ok(())
/// # }
/// ```
pub fn verify_delaunay_for_triangulation<K, U, V, const D: usize>(
    triangulation: &Triangulation<K, U, V, D>,
) -> Result<(), DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    verify_delaunay_with_topology(
        &triangulation.tds,
        &triangulation.kernel,
        triangulation.global_topology,
    )
}

/// Verify the Delaunay property via local flip predicates under a global topology model.
///
/// For periodic topologies this evaluates predicates in lifted coordinates using the
/// per-simplex periodic vertex offsets stored on quotient simplices.
fn verify_delaunay_with_topology<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    global_topology: GlobalTopology<D>,
) -> Result<(), DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    verify_repair_postcondition_with_topology(
        tds,
        kernel,
        None,
        global_topology,
        PostconditionMode::Strict,
        None,
        ConnectivityPostcondition::Check,
    )
}

/// Keeps legacy Euclidean repair checks on the same validation path as the
/// topology-aware verifier.
fn verify_repair_postcondition<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    seed_simplices: Option<&[SimplexKey]>,
    last_applied_flip: Option<&LastAppliedFlip>,
) -> Result<(), DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    verify_repair_postcondition_with_topology(
        tds,
        kernel,
        seed_simplices,
        GlobalTopology::DEFAULT,
        PostconditionMode::Repair,
        last_applied_flip,
        ConnectivityPostcondition::Check,
    )
}

/// Replays local repair postconditions without forcing the full connectivity check.
fn verify_local_repair_postcondition<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    seed_simplices: &[SimplexKey],
    last_applied_flip: Option<&LastAppliedFlip>,
) -> Result<(), DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    verify_repair_postcondition_with_topology(
        tds,
        kernel,
        Some(seed_simplices),
        GlobalTopology::DEFAULT,
        PostconditionMode::Repair,
        last_applied_flip,
        ConnectivityPostcondition::Defer,
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PostconditionMode {
    Repair,
    Strict,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ConnectivityPostcondition {
    Check,
    Defer,
}

/// Builds a verification failure that preserves the structured flip error.
fn verification_failed(
    context: DelaunayRepairVerificationContext,
    source: FlipError,
) -> DelaunayRepairError {
    DelaunayRepairError::VerificationFailed {
        context,
        source: Box::new(source),
    }
}

/// Adapts the public topology enum into the model used for lifted predicate
/// evaluation.
fn verify_repair_postcondition_with_topology<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    seed_simplices: Option<&[SimplexKey]>,
    global_topology: GlobalTopology<D>,
    mode: PostconditionMode,
    last_applied_flip: Option<&LastAppliedFlip>,
    connectivity: ConnectivityPostcondition,
) -> Result<(), DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    let topology_model = global_topology.model();
    verify_repair_postcondition_locally(
        tds,
        kernel,
        seed_simplices,
        &topology_model,
        mode,
        last_applied_flip,
        connectivity,
    )
}

/// Replays the repair queues without mutating the TDS so postconditions cover
/// the same local predicates that drive repair.
fn verify_repair_postcondition_locally<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    seed_simplices: Option<&[SimplexKey]>,
    topology_model: &GlobalTopologyModelAdapter<D>,
    mode: PostconditionMode,
    last_applied_flip: Option<&LastAppliedFlip>,
    connectivity: ConnectivityPostcondition,
) -> Result<(), DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    let config = RepairAttemptConfig {
        attempt: 0,
        queue_order: RepairQueueOrder::Fifo,
        max_flips_override: None,
    };

    let mut stats = DelaunayRepairStats::default();
    let mut diagnostics = RepairDiagnostics::default();
    let mut queues = RepairQueues::new();
    let _ = seed_repair_queues(tds, seed_simplices, &mut queues, &mut stats)?;
    if repair_trace_enabled() {
        let seed_count = seed_simplices.map_or(0, <[SimplexKey]>::len);
        tracing::debug!(
            "[repair] attempt={} order={:?} simplices={} seeds={} queues(facet={}, ridge={}, edge={}, tri={})",
            config.attempt,
            config.queue_order,
            tds.number_of_simplices(),
            seed_count,
            queues.facet_queue.len(),
            queues.ridge_queue.len(),
            queues.edge_queue.len(),
            queues.triangle_queue.len(),
        );
    }

    verify_postcondition_k2_facets(
        tds,
        kernel,
        topology_model,
        &mut queues.facet_queue,
        &config,
        &mut diagnostics,
        mode,
        last_applied_flip,
    )?;
    verify_postcondition_k3_ridges(
        tds,
        kernel,
        topology_model,
        &mut queues.ridge_queue,
        &config,
        &mut diagnostics,
        mode,
        last_applied_flip,
    )?;
    verify_postcondition_inverse_k2_edges(
        tds,
        kernel,
        topology_model,
        &mut queues.edge_queue,
        &config,
        &mut diagnostics,
        mode,
    )?;
    verify_postcondition_inverse_k3_triangles(
        tds,
        kernel,
        topology_model,
        &mut queues.triangle_queue,
        &config,
        &mut diagnostics,
        mode,
    )?;

    // After all flip predicates pass, full repair checks that the repair did not
    // disconnect the neighbor graph. Batch-local construction repair defers this
    // whole-TDS check to the construction finalization topology validation; doing
    // it after every small local repair dominates large 3D runs without adding a
    // stronger boundary guarantee than final validation already enforces.
    if connectivity == ConnectivityPostcondition::Check && !tds.is_connected() {
        return Err(DelaunayRepairError::PostconditionFailed {
            reason: Box::new(DelaunayRepairPostconditionFailure::Disconnected {
                simplex_count: tds.number_of_simplices(),
            }),
        });
    }

    Ok(())
}

/// Centralizes Strict/Repair handling so inconclusive predicates fail validation
/// while remaining skippable during best-effort repair passes.
fn resolve_postcondition_predicate_failure(
    mode: PostconditionMode,
    context: DelaunayRepairVerificationContext,
    error: &FlipError,
) -> Result<(), DelaunayRepairError> {
    match mode {
        PostconditionMode::Repair => Ok(()),
        PostconditionMode::Strict => Err(verification_failed(context, error.clone())),
    }
}

/// Rechecks queued facets after repair so unresolved k=2 violations surface as
/// postcondition failures instead of latent invalid triangulations.
#[expect(
    clippy::too_many_arguments,
    reason = "Postcondition replay threads topology, diagnostics, and predecessor context explicitly"
)]
fn verify_postcondition_k2_facets<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    topology_model: &GlobalTopologyModelAdapter<D>,
    queue: &mut VecDeque<(FacetHandle, u64)>,
    config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
    mode: PostconditionMode,
    last_applied_flip: Option<&LastAppliedFlip>,
) -> Result<(), DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    while let Some((facet, _key)) = pop_queue(queue, config.queue_order) {
        let context = match build_k2_flip_context(tds, facet) {
            Ok(ctx) => ctx,
            Err(
                FlipError::BoundaryFacet { .. }
                | FlipError::MissingSimplex { .. }
                | FlipError::MissingNeighbor { .. }
                | FlipError::InvalidFacetAdjacency { .. }
                | FlipError::InvalidFacetIndex { .. },
            ) => {
                continue;
            }
            Err(e) => return Err(e.into()),
        };

        match is_delaunay_violation_k2(tds, kernel, topology_model, &context, config, diagnostics) {
            Ok(true) => {
                let flip_degenerate = match k2_flip_would_create_degenerate_simplex(tds, &context) {
                    Ok(degenerate) => degenerate,
                    Err(error @ FlipError::PredicateFailure { .. }) => {
                        resolve_postcondition_predicate_failure(
                            mode,
                            DelaunayRepairVerificationContext::LocalK2DegeneracyVerification,
                            &error,
                        )?;
                        continue;
                    }
                    Err(e) => {
                        return Err(verification_failed(
                            DelaunayRepairVerificationContext::LocalK2DegeneracyVerification,
                            e,
                        ));
                    }
                };

                if flip_degenerate {
                    if repair_trace_enabled() {
                        tracing::debug!(
                            "[repair] postcondition k=2 violation unresolved due to degenerate flip (facet={facet:?})"
                        );
                    }
                    continue;
                }
                if repair_trace_enabled() {
                    tracing::debug!(
                        "[repair] postcondition k=2 violation remains (facet={facet:?})"
                    );
                }
                debug_postcondition_facet_context(
                    tds,
                    facet,
                    &context,
                    diagnostics,
                    last_applied_flip,
                );
                let debug_details = if env::var_os("DELAUNAY_REPAIR_DEBUG_FACETS").is_some() {
                    let removed_details: Vec<_> = context
                        .removed_face_vertices
                        .iter()
                        .filter_map(|&vkey| tds.vertex(vkey).map(|vertex| (vkey, *vertex.point())))
                        .collect();
                    let inserted_details: Vec<_> = context
                        .inserted_face_vertices
                        .iter()
                        .filter_map(|&vkey| tds.vertex(vkey).map(|vertex| (vkey, *vertex.point())))
                        .collect();
                    Some(format!(
                        "removed_face={removed_details:?}; inserted_face={inserted_details:?}"
                    ))
                } else {
                    None
                };
                return Err(DelaunayRepairError::PostconditionFailed {
                    reason: Box::new(DelaunayRepairPostconditionFailure::LocalK2Violation {
                        facet,
                        debug_details,
                    }),
                });
            }
            Ok(false) => {
                // No violation detected.
            }
            Err(error @ FlipError::PredicateFailure { .. }) => {
                resolve_postcondition_predicate_failure(
                    mode,
                    DelaunayRepairVerificationContext::LocalK2PostconditionVerification,
                    &error,
                )?;
            }
            Err(e) => {
                return Err(verification_failed(
                    DelaunayRepairVerificationContext::LocalK2PostconditionVerification,
                    e,
                ));
            }
        }
    }

    Ok(())
}

/// Rechecks queued ridges after repair so higher-dimensional k=3 violations get
/// the same explicit postcondition treatment as facets.
#[expect(
    clippy::too_many_arguments,
    reason = "Postcondition replay threads topology, diagnostics, and predecessor context explicitly (matches k=2 signature)"
)]
fn verify_postcondition_k3_ridges<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    topology_model: &GlobalTopologyModelAdapter<D>,
    queue: &mut VecDeque<(RidgeHandle, u64)>,
    config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
    mode: PostconditionMode,
    last_applied_flip: Option<&LastAppliedFlip>,
) -> Result<(), DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    while let Some((ridge, _key)) = pop_queue(queue, config.queue_order) {
        let context = match build_k3_flip_context(tds, ridge) {
            Ok(ctx) => ctx,
            Err(
                FlipError::InvalidRidgeIndex { .. }
                | FlipError::InvalidRidgeAdjacency { .. }
                | FlipError::InvalidRidgeMultiplicity { .. }
                | FlipError::MissingSimplex { .. },
            ) => {
                continue;
            }
            Err(e) => return Err(e.into()),
        };

        match is_delaunay_violation_k3(tds, kernel, topology_model, &context, config, diagnostics) {
            Ok(true) => {
                let flip_degenerate = match flip_would_create_degenerate_simplex(
                    tds,
                    &context.removed_face_vertices,
                    &context.inserted_face_vertices,
                ) {
                    Ok(degenerate) => degenerate,
                    Err(error @ FlipError::PredicateFailure { .. }) => {
                        resolve_postcondition_predicate_failure(
                            mode,
                            DelaunayRepairVerificationContext::LocalK3DegeneracyVerification,
                            &error,
                        )?;
                        continue;
                    }
                    Err(e) => {
                        return Err(verification_failed(
                            DelaunayRepairVerificationContext::LocalK3DegeneracyVerification,
                            e,
                        ));
                    }
                };

                if flip_degenerate {
                    if repair_trace_enabled() {
                        tracing::debug!(
                            "[repair] postcondition k=3 violation unresolved due to degenerate flip (ridge={ridge:?})"
                        );
                    }
                    continue;
                }
                if repair_trace_enabled() {
                    tracing::debug!(
                        "[repair] postcondition k=3 violation remains (ridge={ridge:?})"
                    );
                }
                // Emit the ridge adjacency snapshot only under the opt-in ridge
                // debug flag; the helper performs global incidence scans.
                if repair_ridge_debug_enabled() {
                    debug_ridge_context(tds, ridge, None, diagnostics, last_applied_flip);
                }
                return Err(DelaunayRepairError::PostconditionFailed {
                    reason: Box::new(DelaunayRepairPostconditionFailure::LocalK3Violation {
                        ridge,
                    }),
                });
            }
            Ok(false) => {
                // No violation detected.
            }
            Err(error @ FlipError::PredicateFailure { .. }) => {
                resolve_postcondition_predicate_failure(
                    mode,
                    DelaunayRepairVerificationContext::LocalK3PostconditionVerification,
                    &error,
                )?;
            }
            Err(e) => {
                return Err(verification_failed(
                    DelaunayRepairVerificationContext::LocalK3PostconditionVerification,
                    e,
                ));
            }
        }
    }

    Ok(())
}

/// Exercises inverse k=2 predicates after repair because an apparently valid
/// facet pass can still leave an edge-collapse move applicable.
fn verify_postcondition_inverse_k2_edges<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    topology_model: &GlobalTopologyModelAdapter<D>,
    queue: &mut VecDeque<(EdgeKey, u64)>,
    config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
    mode: PostconditionMode,
) -> Result<(), DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    while let Some((edge, _key)) = pop_queue(queue, config.queue_order) {
        let context = match build_k2_flip_context_from_edge(tds, edge) {
            Ok(ctx) => ctx,
            Err(
                FlipError::InvalidEdgeMultiplicity { .. }
                | FlipError::InvalidEdgeAdjacency { .. }
                | FlipError::MissingSimplex { .. }
                | FlipError::MissingVertex { .. },
            ) => {
                continue;
            }
            Err(e) => return Err(e.into()),
        };

        if context.removed_face_vertices.len() != 2 {
            continue;
        }
        let opposite_a = context.removed_face_vertices[0];
        let opposite_b = context.removed_face_vertices[1];
        let frame_simplex = removed_simplex_frame(&context.removed_simplices)?;

        let violates = match delaunay_violation_k2_for_facet(
            tds,
            kernel,
            topology_model,
            &context.inserted_face_vertices,
            opposite_a,
            opposite_b,
            &context.removed_simplices,
            Some(frame_simplex),
            config,
            diagnostics,
        ) {
            Ok(violates) => violates,
            Err(error @ FlipError::PredicateFailure { .. }) => {
                resolve_postcondition_predicate_failure(
                    mode,
                    DelaunayRepairVerificationContext::LocalInverseK2PostconditionVerification,
                    &error,
                )?;
                continue;
            }
            Err(e) => {
                return Err(verification_failed(
                    DelaunayRepairVerificationContext::LocalInverseK2PostconditionVerification,
                    e,
                ));
            }
        };

        if !violates {
            if repair_trace_enabled() {
                tracing::debug!(
                    "[repair] postcondition inverse k=2 flip still applicable (edge={edge:?})"
                );
            }
            return Err(DelaunayRepairError::PostconditionFailed {
                reason: Box::new(
                    DelaunayRepairPostconditionFailure::LocalInverseK2Violation { edge },
                ),
            });
        }
    }

    Ok(())
}

/// Exercises inverse k=3 predicates after repair so triangle-collapse moves do
/// not hide behind forward-only verification.
fn verify_postcondition_inverse_k3_triangles<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    topology_model: &GlobalTopologyModelAdapter<D>,
    queue: &mut VecDeque<(TriangleHandle, u64)>,
    config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
    mode: PostconditionMode,
) -> Result<(), DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    while let Some((triangle, _key)) = pop_queue(queue, config.queue_order) {
        let context = match build_k3_flip_context_from_triangle(tds, triangle) {
            Ok(ctx) => ctx,
            Err(
                FlipError::InvalidTriangleMultiplicity { .. }
                | FlipError::InvalidTriangleAdjacency { .. }
                | FlipError::MissingSimplex { .. }
                | FlipError::MissingVertex { .. },
            ) => {
                continue;
            }
            Err(e) => return Err(e.into()),
        };

        let frame_simplex = removed_simplex_frame(&context.removed_simplices)?;
        let violates = match delaunay_violation_k3_for_ridge(
            tds,
            kernel,
            topology_model,
            &context.inserted_face_vertices,
            &context.removed_face_vertices,
            &context.removed_simplices,
            Some(frame_simplex),
            config,
            diagnostics,
        ) {
            Ok(violates) => violates,
            Err(error @ FlipError::PredicateFailure { .. }) => {
                resolve_postcondition_predicate_failure(
                    mode,
                    DelaunayRepairVerificationContext::LocalInverseK3PostconditionVerification,
                    &error,
                )?;
                continue;
            }
            Err(e) => {
                return Err(verification_failed(
                    DelaunayRepairVerificationContext::LocalInverseK3PostconditionVerification,
                    e,
                ));
            }
        };

        if !violates {
            if repair_trace_enabled() {
                tracing::debug!(
                    "[repair] postcondition inverse k=3 flip still applicable (triangle={triangle:?})"
                );
            }
            return Err(DelaunayRepairError::PostconditionFailed {
                reason: Box::new(
                    DelaunayRepairPostconditionFailure::LocalInverseK3Violation { triangle },
                ),
            });
        }
    }

    Ok(())
}

// =============================================================================
// Internal helpers
// =============================================================================
const AMBIGUOUS_SAMPLE_LIMIT: usize = 16;
const CYCLE_SAMPLE_LIMIT: usize = 16;
const FLIP_SIGNATURE_WINDOW: usize = 4096;
// Allow extended repeats to capture diagnostics in long-running repairs.  A threshold of
// 32 caused false non-convergence on valid 3D inputs (see #306); 128 still bounds
// pathological cases while giving legitimate repair sequences room to converge.
const MAX_REPEAT_SIGNATURE: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct FlipSignature(u64);

impl fmt::Display for FlipSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Default)]
struct RepairDiagnostics {
    ambiguous_predicates: usize,
    ambiguous_samples: Vec<u64>,
    predicate_failures: usize,
    cycle_detections: usize,
    cycle_samples: Vec<FlipSignature>,
    inserted_simplex_skips: usize,
    inserted_simplex_sample: Option<InsertedSimplexSkipSample>,
    invalid_ridge_multiplicity_skips: usize,
    invalid_ridge_multiplicity_sample: Option<RidgeMultiplicitySkipSample>,
    missing_simplex_skips: usize,
    missing_simplex_sample: Option<MissingSimplexSkipSample>,
    saw_applicable_repair_site: bool,
    flip_signature_window: VecDeque<FlipSignature>,
    flip_signature_counts: FastHashMap<FlipSignature, usize>,
    ridge_debug_emitted: usize,
    postcondition_facet_debug_emitted: usize,
}

#[derive(Clone, PartialEq, Eq)]
struct InsertedSimplexSkipSample {
    location: RepairSkipLocation,
    removed_face: VertexKeyList,
    inserted_face: VertexKeyList,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct RidgeMultiplicitySkipSample {
    ridge: RidgeHandle,
    multiplicity: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct MissingSimplexSkipSample {
    location: RepairSkipLocation,
    simplex_key: SimplexKey,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RepairSkipLocation {
    Edge(EdgeKey),
    Facet(FacetHandle),
    Ridge(RidgeHandle),
    Triangle(TriangleHandle),
}

impl RepairSkipLocation {
    fn fmt_label(self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Edge(edge) => write!(f, "edge={edge:?}"),
            Self::Facet(facet) => write!(f, "facet={facet:?}"),
            Self::Ridge(ridge) => write!(f, "ridge={ridge:?}"),
            Self::Triangle(triangle) => write!(f, "triangle={triangle:?}"),
        }
    }
}

impl fmt::Display for RepairSkipLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_label(f)
    }
}

impl fmt::Debug for RepairSkipLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for InsertedSimplexSkipSample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.location.fmt_label(f)?;
        write!(
            f,
            " removed_face={:?} inserted_face={:?}",
            self.removed_face, self.inserted_face
        )
    }
}

impl fmt::Debug for InsertedSimplexSkipSample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.to_string(), f)
    }
}

impl fmt::Display for RidgeMultiplicitySkipSample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ridge={:?} multiplicity={}",
            self.ridge, self.multiplicity
        )
    }
}

impl fmt::Debug for RidgeMultiplicitySkipSample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.to_string(), f)
    }
}

impl fmt::Display for MissingSimplexSkipSample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.location.fmt_label(f)?;
        write!(f, " missing_simplex={:?}", self.simplex_key)
    }
}

impl fmt::Debug for MissingSimplexSkipSample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.to_string(), f)
    }
}

fn vertex_key_list(vertices: &[VertexKey]) -> VertexKeyList {
    vertices.iter().copied().collect()
}

impl RepairDiagnostics {
    /// Records uncertain predicate sites with bounded samples so diagnostics stay
    /// actionable on large repairs.
    fn record_ambiguous(&mut self, key: u64) {
        self.ambiguous_predicates += 1;
        if self.ambiguous_samples.len() >= AMBIGUOUS_SAMPLE_LIMIT {
            return;
        }
        if !self.ambiguous_samples.contains(&key) {
            self.ambiguous_samples.push(key);
        }
    }

    /// Counts predicate failures separately from ambiguity because failures abort
    /// the current local check.
    const fn record_predicate_failure(&mut self) {
        self.predicate_failures = self.predicate_failures.saturating_add(1);
    }

    /// Maintains a sliding signature window so cycle detection is bounded in
    /// memory but still catches local oscillations.
    fn record_flip_signature(&mut self, signature: FlipSignature) {
        let count = self.flip_signature_counts.entry(signature).or_insert(0);
        *count = count.saturating_add(1);

        if *count > 1 {
            self.cycle_detections = self.cycle_detections.saturating_add(1);
            if self.cycle_samples.len() < CYCLE_SAMPLE_LIMIT
                && !self.cycle_samples.contains(&signature)
            {
                self.cycle_samples.push(signature);
            }
        }

        self.flip_signature_window.push_back(signature);
        if self.flip_signature_window.len() > FLIP_SIGNATURE_WINDOW
            && let Some(old) = self.flip_signature_window.pop_front()
            && let Some(old_count) = self.flip_signature_counts.get_mut(&old)
        {
            *old_count = old_count.saturating_sub(1);
            if *old_count == 0 {
                self.flip_signature_counts.remove(&old);
            }
        }
    }

    /// Preserves the signature that triggered a non-convergence abort even if it
    /// was already sampled earlier.
    fn record_cycle_abort(&mut self, signature: FlipSignature) {
        self.cycle_detections = self.cycle_detections.saturating_add(1);
        if self.cycle_samples.len() < CYCLE_SAMPLE_LIMIT && !self.cycle_samples.contains(&signature)
        {
            self.cycle_samples.push(signature);
        }
    }

    /// Captures one duplicate-simplex skip sample with typed context.
    fn record_inserted_simplex_skip(&mut self, sample: InsertedSimplexSkipSample) {
        self.inserted_simplex_skips = self.inserted_simplex_skips.saturating_add(1);
        if self.inserted_simplex_sample.is_none() {
            self.inserted_simplex_sample = Some(sample);
        }
    }

    /// Captures one invalid-ridge sample with typed context.
    const fn record_invalid_ridge_multiplicity_skip(
        &mut self,
        sample: RidgeMultiplicitySkipSample,
    ) {
        self.invalid_ridge_multiplicity_skips =
            self.invalid_ridge_multiplicity_skips.saturating_add(1);
        if self.invalid_ridge_multiplicity_sample.is_none() {
            self.invalid_ridge_multiplicity_sample = Some(sample);
        }
    }

    /// Captures one stale-simplex sample so slot-swap churn is visible when
    /// repair diagnostics are inspected.
    const fn record_missing_simplex_skip(&mut self, sample: MissingSimplexSkipSample) {
        self.missing_simplex_skips = self.missing_simplex_skips.saturating_add(1);
        if self.missing_simplex_sample.is_none() {
            self.missing_simplex_sample = Some(sample);
        }
    }

    /// Marks that a repair predicate found an applicable flip even if no mutation followed.
    const fn record_applicable_repair_site(&mut self) {
        self.saw_applicable_repair_site = true;
    }
}

#[derive(Debug, Clone, Copy)]
struct RepairAttemptConfig {
    attempt: usize,
    queue_order: RepairQueueOrder,
    /// Override the flip budget. `None` uses `default_max_flips` (proportional to total simplex
    /// count). Set to `Some(n)` for per-insertion local repairs to avoid a runaway budget when
    /// the triangulation is large but the seed set is small.
    max_flips_override: Option<usize>,
}

/// Builds the public non-convergence error in one place so diagnostics, queue
/// order, and attempt metadata stay consistent.
fn non_convergent_error(
    max_flips: usize,
    stats: &DelaunayRepairStats,
    diagnostics: &RepairDiagnostics,
    config: &RepairAttemptConfig,
) -> DelaunayRepairError {
    emit_repair_debug_summary("non_convergent", stats, diagnostics, config, max_flips);
    DelaunayRepairError::NonConvergent {
        max_flips,
        diagnostics: Box::new(DelaunayRepairDiagnostics {
            facets_checked: stats.facets_checked,
            flips_performed: stats.flips_performed,
            max_queue_len: stats.max_queue_len,
            ambiguous_predicates: diagnostics.ambiguous_predicates,
            ambiguous_predicate_samples: diagnostics.ambiguous_samples.clone(),
            predicate_failures: diagnostics.predicate_failures,
            cycle_detections: diagnostics.cycle_detections,
            cycle_signature_samples: diagnostics
                .cycle_samples
                .iter()
                .map(|signature| signature.0)
                .collect(),
            attempt: config.attempt,
            queue_order: config.queue_order,
        }),
    }
}

/// Converts a measured duration to nanoseconds while saturating pathological
/// values that exceed telemetry counter width.
fn duration_nanos_saturating(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

/// Gates the expensive repair summary behind an environment variable while
/// keeping all attempts logged in a uniform shape.
fn emit_repair_debug_summary(
    label: &str,
    stats: &DelaunayRepairStats,
    diagnostics: &RepairDiagnostics,
    config: &RepairAttemptConfig,
    max_flips: usize,
) {
    if env::var_os("DELAUNAY_REPAIR_DEBUG_SUMMARY").is_none() {
        return;
    }

    tracing::debug!(
        label = %label,
        attempt = config.attempt,
        order = ?config.queue_order,
        flips = stats.flips_performed,
        max_flips,
        checked = stats.facets_checked,
        max_queue = stats.max_queue_len,
        ambiguous = diagnostics.ambiguous_predicates,
        predicate_failures = diagnostics.predicate_failures,
        cycles = diagnostics.cycle_detections,
        inserted_simplex_skips = diagnostics.inserted_simplex_skips,
        invalid_ridge_multiplicity_skips = diagnostics.invalid_ridge_multiplicity_skips,
        missing_simplex_skips = diagnostics.missing_simplex_skips,
        inserted_simplex_sample = ?diagnostics.inserted_simplex_sample,
        invalid_ridge_multiplicity_sample = ?diagnostics.invalid_ridge_multiplicity_sample,
        missing_simplex_sample = ?diagnostics.missing_simplex_sample,
        "repair summary"
    );
}

/// Shares FIFO/LIFO behavior across repair queues so alternate attempts only
/// differ by scheduling policy.
fn pop_queue<T>(queue: &mut VecDeque<T>, order: RepairQueueOrder) -> Option<T> {
    match order {
        RepairQueueOrder::Fifo => queue.pop_front(),
        RepairQueueOrder::Lifo => queue.pop_back(),
    }
}

/// Hashes a predicate site canonically so ambiguous-predicate samples are stable
/// across vertex ordering in a simplex.
fn predicate_key_from_vertices(simplex_vertices: &[VertexKey], test_vertex: VertexKey) -> u64 {
    let mut sorted: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        simplex_vertices.iter().copied().collect();
    sorted.sort_unstable();

    let mut hasher = FastHasher::default();
    for vkey in &sorted {
        vkey.hash(&mut hasher);
    }
    test_vertex.hash(&mut hasher);
    hasher.finish()
}

/// Canonicalizes a flip attempt into a compact key for cycle detection.
fn flip_signature(
    kind: BistellarFlipKind,
    direction: FlipDirection,
    removed_face_vertices: &[VertexKey],
    inserted_face_vertices: &[VertexKey],
) -> FlipSignature {
    let mut removed: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        removed_face_vertices.iter().copied().collect();
    removed.sort_unstable();

    let mut inserted: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        inserted_face_vertices.iter().copied().collect();
    inserted.sort_unstable();

    let mut hasher = FastHasher::default();
    kind.k().hash(&mut hasher);
    match direction {
        FlipDirection::Forward => 0_u8.hash(&mut hasher),
        FlipDirection::Inverse => 1_u8.hash(&mut hasher),
    }
    removed.len().hash(&mut hasher);
    for vkey in &removed {
        vkey.hash(&mut hasher);
    }
    inserted.len().hash(&mut hasher);
    for vkey in &inserted {
        vkey.hash(&mut hasher);
    }
    FlipSignature(hasher.finish())
}

#[derive(Debug, Clone)]
struct LastAppliedFlip {
    kind: BistellarFlipKind,
    removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    removed_simplices: SimplexKeyBuffer,
    new_simplices: SimplexKeyBuffer,
    /// Snapshot of each removed simplex's vertex list captured before the flip's
    /// `remove_simplices_by_keys` call; pairs 1:1 with `removed_simplices`. Empty
    /// inner buffers only appear in placeholder instances built from validated
    /// flip faces.
    removed_simplex_vertices: RemovedSimplexVertexSnapshot,
}

impl LastAppliedFlip {
    /// Sorts faces so immediate-reversal detection is independent of local simplex
    /// vertex order. Simplex lists stay empty here because this constructor is also
    /// used for temporary reversal checks.
    fn from_validated_flip_faces(
        kind: BistellarFlipKind,
        removed: &[VertexKey],
        inserted: &[VertexKey],
    ) -> Self {
        let mut removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
            removed.iter().copied().collect();
        removed_face_vertices.sort_unstable();

        let mut inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
            inserted.iter().copied().collect();
        inserted_face_vertices.sort_unstable();

        Self {
            kind,
            removed_face_vertices,
            inserted_face_vertices,
            removed_simplices: SimplexKeyBuffer::new(),
            new_simplices: SimplexKeyBuffer::new(),
            removed_simplex_vertices: SmallBuffer::new(),
        }
    }

    /// Preserves the concrete flip footprint so a later ridge snapshot can tell
    /// whether the immediately preceding move created the bad local star.
    fn from_applied_flip<const D: usize>(applied: &AppliedFlip<D>) -> Self {
        let info = &applied.info;
        let mut last = Self::from_validated_flip_faces(
            info.kind,
            &info.removed_face_vertices,
            &info.inserted_face_vertices,
        );
        last.removed_simplices.clone_from(&info.removed_simplices);
        last.new_simplices.clone_from(&info.new_simplices);
        last.removed_simplex_vertices
            .clone_from(&applied.removed_simplex_vertices);
        last
    }

    /// Formats each removed simplex as `SimplexKey(N): vertices=[...]` using the
    /// snapshot captured before the flip's simplex removal. Falls back to
    /// `missing-snapshot` only for placeholder rows built from validated flip
    /// faces.
    fn removed_simplex_vertex_lines(&self) -> Vec<String> {
        self.removed_simplices
            .iter()
            .copied()
            .enumerate()
            .map(
                |(idx, simplex_key)| match self.removed_simplex_vertices.get(idx) {
                    Some(verts) if !verts.is_empty() => {
                        format!("{simplex_key:?}: vertices={verts:?}")
                    }
                    _ => format!("{simplex_key:?}: missing-snapshot"),
                },
            )
            .collect()
    }
}

/// Catches two-step flip oscillations before they inflate repair diagnostics or
/// consume the global flip budget.
fn would_immediately_reverse_last_flip<const D: usize>(
    last: Option<&LastAppliedFlip>,
    kind: BistellarFlipKind,
    removed_face_vertices: &[VertexKey],
    inserted_face_vertices: &[VertexKey],
) -> bool {
    let Some(last_flip) = last else {
        return false;
    };

    if kind.k() + last_flip.kind.k() != D + 2 {
        return false;
    }

    let current = LastAppliedFlip::from_validated_flip_faces(
        kind,
        removed_face_vertices,
        inserted_face_vertices,
    );
    current.removed_face_vertices == last_flip.inserted_face_vertices
        && current.inserted_face_vertices == last_flip.removed_face_vertices
}

/// Keeps verbose repair tracing opt-in because the hot repair loop calls this
/// frequently.
#[inline]
fn repair_trace_enabled() -> bool {
    env::var_os("DELAUNAY_REPAIR_TRACE").is_some()
}

/// Treats full repair tracing as enabling ridge snapshots so one debug switch
/// gives enough topology context.
#[inline]
fn repair_ridge_debug_enabled() -> bool {
    env::var_os("DELAUNAY_REPAIR_DEBUG_RIDGE").is_some() || repair_trace_enabled()
}

const RIDGE_DEBUG_LIMIT_DEFAULT: usize = 64;
const RIDGE_DEBUG_MIN_MULTIPLICITY_DEFAULT: usize = 0;

/// Rate-limits ridge snapshots to keep pathological repair runs from flooding
/// logs.
fn ridge_debug_limit() -> usize {
    env::var("DELAUNAY_REPAIR_DEBUG_RIDGE_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(RIDGE_DEBUG_LIMIT_DEFAULT)
}

/// Lets callers skip the common multiplicity-1/2 boundary cases and capture
/// the first genuinely overshared ridge instead.
fn ridge_debug_min_multiplicity() -> usize {
    env::var("DELAUNAY_REPAIR_DEBUG_RIDGE_MIN_MULTIPLICITY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(RIDGE_DEBUG_MIN_MULTIPLICITY_DEFAULT)
}

/// Applies the ridge debug limit per repair attempt so independent repairs do
/// not consume each other's diagnostic budget.
fn should_emit_ridge_debug(
    diagnostics: &mut RepairDiagnostics,
    reported_multiplicity: Option<usize>,
) -> bool {
    let min_multiplicity = ridge_debug_min_multiplicity();
    match reported_multiplicity {
        // Multiplicity-based skips dominate large 4D traces, so let callers suppress
        // the expected 1/2 boundary cases and wait for the first real fan.
        Some(found) if found < min_multiplicity => return false,
        // If the caller asked for a multiplicity threshold, suppress adjacency-only
        // snapshots too so they do not consume the one-shot debug budget first.
        None if min_multiplicity > 0 => return false,
        _ => {}
    }

    let limit = ridge_debug_limit();
    if limit == 0 {
        return false;
    }
    let current = diagnostics.ridge_debug_emitted;
    diagnostics.ridge_debug_emitted = diagnostics.ridge_debug_emitted.saturating_add(1);
    if current == limit {
        tracing::debug!(
            "repair: ridge debug output limit reached; suppressing further ridge snapshots"
        );
    }
    current < limit
}

/// Keeps the first unresolved postcondition-facet snapshot opt-in because the
/// local verifier can traverse many queued facets in one pass.
#[inline]
fn postcondition_facet_debug_enabled() -> bool {
    env::var_os("DELAUNAY_REPAIR_DEBUG_POSTCONDITION_FACET").is_some()
}

/// Emits at most one postcondition facet snapshot per repair attempt so the
/// focused #204 debug path stays readable.
fn should_emit_postcondition_facet_debug(diagnostics: &mut RepairDiagnostics) -> bool {
    if !postcondition_facet_debug_enabled() {
        return false;
    }
    let current = diagnostics.postcondition_facet_debug_emitted;
    diagnostics.postcondition_facet_debug_emitted = diagnostics
        .postcondition_facet_debug_emitted
        .saturating_add(1);
    current == 0
}

/// Computes a dimension-sensitive flip budget so non-convergent repair fails
/// predictably instead of running unbounded.
fn default_max_flips<const D: usize>(simplex_count: usize) -> usize {
    // Flip budget strategy by dimension and build mode:
    //
    // - D<=2: use 4× budget in debug/test (2D flips are fast).
    // - D=3: use 8× budget in debug/test.  Previously 16× but that caused the global repair
    //   to spend hours cycling through flip loops when many star-splits produced a heavily
    //   non-Delaunay triangulation.  8× still provides headroom for legitimate convergence
    //   while failing faster (triggering the heuristic rebuild sooner) when cycling.
    // - D>=4: use simplices×(D+1)×4 (min 4096) in debug/test.  Flip convergence is not
    //   guaranteed in D>=4 (Edelsbrunner-Shah 1996), so this budget is intentionally
    //   conservative: it bounds the cost of user-facing repair APIs (repair_delaunay_with_flips
    //   and run_flip_repair_fallbacks during incremental insertion) while failing fast
    //   when cycling occurs.  Bulk construction for D>=4 does NOT rely on post-construction
    //   flip repair; correctness is ensured by the robust conflict-region detection in
    //   find_conflict_region and the is_delaunay_property_only() check in
    //   build_with_shuffled_retries.
    if D >= 4 {
        return simplex_count
            .saturating_mul(D.saturating_add(1))
            .saturating_mul(4)
            .max(4096);
    }
    let multiplier = match D {
        3 => 8,
        _ => 4, // D<=2
    };
    let base = simplex_count
        .saturating_mul(D.saturating_add(1))
        .saturating_mul(multiplier);
    base.max(512)
}

struct RepairQueues {
    facet_queue: VecDeque<(FacetHandle, u64)>,
    facet_queued: FastHashSet<u64>,
    facet_handles: FastHashMap<u64, FacetHandle>,
    ridge_queue: VecDeque<(RidgeHandle, u64)>,
    ridge_queued: FastHashSet<u64>,
    ridge_handles: FastHashMap<u64, RidgeHandle>,
    edge_queue: VecDeque<(EdgeKey, u64)>,
    edge_queued: FastHashSet<u64>,
    triangle_queue: VecDeque<(TriangleHandle, u64)>,
    triangle_queued: FastHashSet<u64>,
}

impl RepairQueues {
    /// Initializes all repair worklists together so queue state cannot be
    /// partially seeded.
    fn new() -> Self {
        Self {
            facet_queue: VecDeque::new(),
            facet_queued: FastHashSet::default(),
            facet_handles: FastHashMap::default(),
            ridge_queue: VecDeque::new(),
            ridge_queued: FastHashSet::default(),
            ridge_handles: FastHashMap::default(),
            edge_queue: VecDeque::new(),
            edge_queued: FastHashSet::default(),
            triangle_queue: VecDeque::new(),
            triangle_queued: FastHashSet::default(),
        }
    }

    /// Reports aggregate queued work for diagnostics that compare scheduling
    /// strategies.
    fn total_len(&self) -> usize {
        self.facet_queue.len()
            + self.ridge_queue.len()
            + self.edge_queue.len()
            + self.triangle_queue.len()
    }

    /// Gives the repair loop one invariant-preserving exit check across all
    /// dimension-specific queues.
    fn has_work(&self) -> bool {
        !self.facet_queue.is_empty()
            || !self.ridge_queue.is_empty()
            || !self.edge_queue.is_empty()
            || !self.triangle_queue.is_empty()
    }
}

/// Seeds exactly the queues supported by the current dimension so repair and
/// verification inspect the same local neighborhoods.
#[expect(
    clippy::too_many_lines,
    reason = "seeding logic mirrors runtime queues and stays as one diagnostic flow"
)]
fn seed_repair_queues<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    seed_simplices: Option<&[SimplexKey]>,
    queues: &mut RepairQueues,
    stats: &mut DelaunayRepairStats,
) -> Result<bool, FlipError>
where
    U: DataType,
    V: DataType,
{
    if let Some(seeds) = seed_simplices {
        let mut present = 0usize;
        let mut missing = 0usize;
        for &simplex_key in seeds {
            if !tds.contains_simplex(simplex_key) {
                missing = missing.saturating_add(1);
                if repair_trace_enabled() {
                    tracing::debug!(
                        "[repair] seed_repair_queues: missing seed simplex={simplex_key:?}"
                    );
                }
                continue;
            }
            present = present.saturating_add(1);
            enqueue_simplex_facets(
                tds,
                simplex_key,
                &mut queues.facet_queue,
                &mut queues.facet_queued,
                &mut queues.facet_handles,
                stats,
            )?;
            enqueue_simplex_ridges(
                tds,
                simplex_key,
                &mut queues.ridge_queue,
                &mut queues.ridge_queued,
                &mut queues.ridge_handles,
                stats,
            )?;
            enqueue_simplex_edges(
                tds,
                simplex_key,
                &mut queues.edge_queue,
                &mut queues.edge_queued,
                stats,
            );
            enqueue_simplex_triangles(
                tds,
                simplex_key,
                &mut queues.triangle_queue,
                &mut queues.triangle_queued,
                stats,
            );
            stats.max_queue_len = stats.max_queue_len.max(queues.total_len());
        }
        if repair_trace_enabled() {
            let seed_sample: SimplexKeyBuffer = seeds.iter().copied().take(8).collect();
            tracing::debug!(
                "[repair] seed_repair_queues: seeds={} present={} missing={}",
                seeds.len(),
                present,
                missing,
            );
            tracing::debug!("[repair] seed_repair_queues: sample={seed_sample:?}");
        }
        // Only fall back to global seeding if specific seeds were requested but all were
        // stale (deleted by prior flips).  If the caller explicitly provides an empty
        // slice they want no seeding — returning with an empty queue is correct here.
        if present == 0 && !seeds.is_empty() {
            if repair_trace_enabled() {
                tracing::debug!(
                    "[repair] seed_repair_queues: all seed simplices stale; falling back to global seeding"
                );
            }
            seed_repair_queues(tds, None, queues, stats)?;
            return Ok(true);
        }
    } else {
        for facet in AllFacetsIter::try_new(tds)? {
            let facet = facet?;
            let handle = FacetHandle::from_validated(facet.simplex_key(), facet.facet_index());
            enqueue_facet(
                tds,
                handle,
                &mut queues.facet_queue,
                &mut queues.facet_queued,
                &mut queues.facet_handles,
                stats,
            );
        }
        for (simplex_key, _) in tds.simplices() {
            enqueue_simplex_ridges(
                tds,
                simplex_key,
                &mut queues.ridge_queue,
                &mut queues.ridge_queued,
                &mut queues.ridge_handles,
                stats,
            )?;
            enqueue_simplex_edges(
                tds,
                simplex_key,
                &mut queues.edge_queue,
                &mut queues.edge_queued,
                stats,
            );
            enqueue_simplex_triangles(
                tds,
                simplex_key,
                &mut queues.triangle_queue,
                &mut queues.triangle_queued,
                stats,
            );
        }
        stats.max_queue_len = stats.max_queue_len.max(queues.total_len());
        return Ok(true);
    }
    Ok(false)
}

/// Requeues the local neighborhood created by a flip so the repair loop follows
/// newly exposed violations instead of rescanning the whole triangulation.
fn enqueue_new_simplices_for_repair<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    new_simplices: &[SimplexKey],
    queues: &mut RepairQueues,
    stats: &mut DelaunayRepairStats,
) -> Result<(), FlipError>
where
    U: DataType,
    V: DataType,
{
    for &simplex_key in new_simplices {
        enqueue_simplex_facets(
            tds,
            simplex_key,
            &mut queues.facet_queue,
            &mut queues.facet_queued,
            &mut queues.facet_handles,
            stats,
        )?;
        enqueue_simplex_ridges(
            tds,
            simplex_key,
            &mut queues.ridge_queue,
            &mut queues.ridge_queued,
            &mut queues.ridge_handles,
            stats,
        )?;
        enqueue_simplex_edges(
            tds,
            simplex_key,
            &mut queues.edge_queue,
            &mut queues.edge_queued,
            stats,
        );
        enqueue_simplex_triangles(
            tds,
            simplex_key,
            &mut queues.triangle_queue,
            &mut queues.triangle_queued,
            stats,
        );
        stats.max_queue_len = stats.max_queue_len.max(queues.total_len());
    }
    Ok(())
}

/// Runs one queued ridge repair because k=3 moves are only meaningful in D>=3 and
/// need their own adjacency validation.
#[expect(
    clippy::too_many_arguments,
    reason = "Repair step threads queues, diagnostics, and config explicitly"
)]
#[expect(
    clippy::too_many_lines,
    reason = "Repair step contains inline tracing and queue handling for diagnostics"
)]
fn run_next_ridge_repair_step<K, U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    trial_workspace: &mut Tds<U, V, D>,
    kernel: &K,
    queues: &mut RepairQueues,
    stats: &mut DelaunayRepairStats,
    max_flips: usize,
    config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
    last_applied_flip: &mut Option<LastAppliedFlip>,
    touched_simplices: &mut SimplexKeyBuffer,
    touched_simplex_set: &mut FastHashSet<SimplexKey>,
) -> Result<bool, DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    let Some((ridge, key)) = pop_queue(&mut queues.ridge_queue, config.queue_order) else {
        return Ok(false);
    };
    queues.ridge_queued.remove(&key);
    let ridge = queues.ridge_handles.remove(&key).unwrap_or(ridge);
    let Some(ridge) = resolve_ridge_handle_for_key(tds, ridge, key) else {
        return Ok(true);
    };
    stats.facets_checked += 1;

    let context = match build_k3_flip_context_for_repair(tds, ridge) {
        Ok(ctx) => ctx,
        Err(
            err @ (FlipError::InvalidRidgeIndex { .. }
            | FlipError::InvalidRidgeAdjacency { .. }
            | FlipError::InvalidRidgeMultiplicity { .. }
            | FlipError::MissingSimplex { .. }),
        ) => {
            match &err {
                FlipError::InvalidRidgeMultiplicity { found } => {
                    diagnostics.record_invalid_ridge_multiplicity_skip(
                        RidgeMultiplicitySkipSample {
                            ridge,
                            multiplicity: *found,
                        },
                    );
                    // This is the main #204 failure mode: capture both the local ridge walk
                    // and the full global incidence so we can see whether repair is skipping
                    // a stale handle or a genuinely overshared ridge.
                    if repair_ridge_debug_enabled() {
                        debug_ridge_context(
                            tds,
                            ridge,
                            Some(*found),
                            diagnostics,
                            last_applied_flip.as_ref(),
                        );
                    }
                }
                FlipError::InvalidRidgeAdjacency { .. } if repair_ridge_debug_enabled() => {
                    debug_ridge_context(tds, ridge, None, diagnostics, last_applied_flip.as_ref());
                }
                FlipError::MissingSimplex { simplex_key } => {
                    diagnostics.record_missing_simplex_skip(MissingSimplexSkipSample {
                        location: RepairSkipLocation::Ridge(ridge),
                        simplex_key: *simplex_key,
                    });
                }
                _ => {}
            }
            if repair_trace_enabled() {
                tracing::debug!("[repair] skip k=3 ridge (ridge={ridge:?}) reason={err}");
            }
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };

    let topology_model = GlobalTopology::DEFAULT.model();
    let violates =
        match is_delaunay_violation_k3(tds, kernel, &topology_model, &context, config, diagnostics)
        {
            Ok(violates) => violates,
            Err(FlipError::PredicateFailure { .. }) => {
                return Ok(true);
            }
            Err(e) => return Err(e.into()),
        };

    if !violates {
        return Ok(true);
    }
    diagnostics.record_applicable_repair_site();

    if would_immediately_reverse_last_flip::<D>(
        last_applied_flip.as_ref(),
        BistellarFlipKind::k3(D),
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
    ) {
        if repair_trace_enabled() {
            tracing::debug!(
                "[repair] skip k=3 flip (ridge={ridge:?}) reason=immediate reverse of prior flip"
            );
        }
        return Ok(true);
    }

    let kind = BistellarFlipKind::k3(D);
    let signature = flip_signature(
        kind,
        context.direction,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
    );
    check_flip_cycle(
        tds,
        FlipCycleContext::from_validated_flip(
            signature,
            kind,
            context.direction,
            &context.removed_face_vertices,
            &context.inserted_face_vertices,
        ),
        diagnostics,
        stats,
        max_flips,
        config,
    )?;

    // Enforce flip budget before applying the flip so that Some(0) means zero flips.
    if stats.flips_performed >= max_flips {
        return Err(non_convergent_error(max_flips, stats, diagnostics, config));
    }

    // Shared trace tail for apply-k=3 skip arms below.
    let log_apply_skip = |err: &FlipError| {
        if repair_trace_enabled() {
            tracing::debug!("[repair] skip k=3 flip (ridge={ridge:?}) reason={err}");
            tracing::debug!(
                "[repair] skip k=3 flip context removed_face={:?} inserted_face={:?} removed_simplices={:?}",
                context.removed_face_vertices,
                context.inserted_face_vertices,
                context.removed_simplices,
            );
        }
    };
    let applied = match apply_delaunay_flip_k3_in_workspace(tds, &context, trial_workspace) {
        Ok(applied) => applied,
        Err(err) if let FlipError::InsertedSimplexAlreadyExists { .. } = &err => {
            diagnostics.record_inserted_simplex_skip(InsertedSimplexSkipSample {
                location: RepairSkipLocation::Ridge(ridge),
                removed_face: vertex_key_list(&context.removed_face_vertices),
                inserted_face: vertex_key_list(&context.inserted_face_vertices),
            });
            log_apply_skip(&err);
            return Ok(true);
        }
        Err(
            err @ (FlipError::DegenerateSimplex
            | FlipError::NegativeOrientation { .. }
            | FlipError::DuplicateSimplex
            | FlipError::NonManifoldFacet
            | FlipError::SimplexCreation(_)),
        ) => {
            log_apply_skip(&err);
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };
    *last_applied_flip = Some(LastAppliedFlip::from_applied_flip(&applied));
    let info = applied.info;
    if repair_trace_enabled() {
        tracing::debug!(
            "[repair] apply k=3 flip: kind={:?} direction={:?} removed_face={:?} inserted_face={:?} removed_simplices={:?} new_simplices={:?}",
            info.kind,
            info.direction,
            info.removed_face_vertices,
            info.inserted_face_vertices,
            info.removed_simplices,
            info.new_simplices,
        );
    }
    stats.flips_performed += 1;
    diagnostics.record_flip_signature(signature);
    record_touched_simplices(touched_simplices, touched_simplex_set, &info.new_simplices);

    enqueue_new_simplices_for_repair(tds, &info.new_simplices, queues, stats)?;

    Ok(true)
}

/// Runs one queued edge repair for inverse k=2 moves so higher-dimensional repair
/// can collapse locally Delaunay edge stars.
#[expect(
    clippy::too_many_arguments,
    reason = "Repair step threads queues, diagnostics, and config explicitly"
)]
#[expect(
    clippy::too_many_lines,
    reason = "Repair step contains inline tracing and queue handling for diagnostics"
)]
fn run_next_edge_repair_step<K, U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    trial_workspace: &mut Tds<U, V, D>,
    kernel: &K,
    queues: &mut RepairQueues,
    stats: &mut DelaunayRepairStats,
    max_flips: usize,
    config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
    last_applied_flip: &mut Option<LastAppliedFlip>,
    touched_simplices: &mut SimplexKeyBuffer,
    touched_simplex_set: &mut FastHashSet<SimplexKey>,
) -> Result<bool, DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    let Some((edge, key)) = pop_queue(&mut queues.edge_queue, config.queue_order) else {
        return Ok(false);
    };
    queues.edge_queued.remove(&key);
    stats.facets_checked += 1;

    // Shared trace tail for build-k=2-edge skip arms below.
    let log_build_skip = |err: &FlipError| {
        if repair_trace_enabled() {
            tracing::debug!("[repair] skip inverse k=2 edge (edge={edge:?}) reason={err}");
        }
    };
    let context = match build_k2_flip_context_from_edge(tds, edge) {
        Ok(ctx) => ctx,
        Err(ref err) if let FlipError::MissingSimplex { simplex_key } = err => {
            diagnostics.record_missing_simplex_skip(MissingSimplexSkipSample {
                location: RepairSkipLocation::Edge(edge),
                simplex_key: *simplex_key,
            });
            log_build_skip(err);
            return Ok(true);
        }
        Err(
            ref err @ (FlipError::InvalidEdgeMultiplicity { .. }
            | FlipError::InvalidEdgeAdjacency { .. }
            | FlipError::MissingVertex { .. }),
        ) => {
            log_build_skip(err);
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };

    if context.removed_face_vertices.len() != 2 {
        return Ok(true);
    }
    let opposite_a = context.removed_face_vertices[0];
    let opposite_b = context.removed_face_vertices[1];
    let frame_simplex = removed_simplex_frame(&context.removed_simplices)?;

    let violates = match delaunay_violation_k2_for_facet(
        tds,
        kernel,
        &GlobalTopology::DEFAULT.model(),
        &context.inserted_face_vertices,
        opposite_a,
        opposite_b,
        &context.removed_simplices,
        Some(frame_simplex),
        config,
        diagnostics,
    ) {
        Ok(violates) => violates,
        Err(FlipError::PredicateFailure { .. }) => {
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };

    // Normally we only apply inverse k=2 if the target (2-simplex) configuration is locally
    // Delaunay. On the second attempt (LIFO queue order), allow exploratory inverse moves
    // to escape trapped non-regular configurations; postcondition verification still
    // enforces correctness.
    let allow_exploratory_inverse = config.attempt >= 2;
    if violates && !allow_exploratory_inverse {
        return Ok(true);
    }
    diagnostics.record_applicable_repair_site();

    if would_immediately_reverse_last_flip::<D>(
        last_applied_flip.as_ref(),
        BistellarFlipKind::k2(D).inverse(),
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
    ) {
        if repair_trace_enabled() {
            tracing::debug!(
                "[repair] skip inverse k=2 flip (edge={edge:?}) reason=immediate reverse of prior flip"
            );
        }
        return Ok(true);
    }
    let kind = BistellarFlipKind::k2(D).inverse();
    let signature = flip_signature(
        kind,
        context.direction,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
    );
    check_flip_cycle(
        tds,
        FlipCycleContext::from_validated_flip(
            signature,
            kind,
            context.direction,
            &context.removed_face_vertices,
            &context.inserted_face_vertices,
        ),
        diagnostics,
        stats,
        max_flips,
        config,
    )?;

    // Enforce flip budget before applying the flip so that Some(0) means zero flips.
    if stats.flips_performed >= max_flips {
        return Err(non_convergent_error(max_flips, stats, diagnostics, config));
    }

    // Shared trace tail for apply-inverse-k=2 skip arms below.
    let log_apply_skip = |err: &FlipError| {
        if repair_trace_enabled() {
            tracing::debug!("[repair] skip inverse k=2 flip (edge={edge:?}) reason={err}");
            tracing::debug!(
                "[repair] skip inverse k=2 flip context removed_face={:?} inserted_face={:?} removed_simplices={:?}",
                context.removed_face_vertices,
                context.inserted_face_vertices,
                context.removed_simplices,
            );
        }
    };
    let applied =
        match apply_delaunay_flip_dynamic_in_workspace(tds, kind.k(), &context, trial_workspace) {
            Ok(applied) => applied,
            Err(err) if let FlipError::InsertedSimplexAlreadyExists { .. } = &err => {
                diagnostics.record_inserted_simplex_skip(InsertedSimplexSkipSample {
                    location: RepairSkipLocation::Edge(edge),
                    removed_face: vertex_key_list(&context.removed_face_vertices),
                    inserted_face: vertex_key_list(&context.inserted_face_vertices),
                });
                log_apply_skip(&err);
                return Ok(true);
            }
            Err(
                err @ (FlipError::DegenerateSimplex
                | FlipError::NegativeOrientation { .. }
                | FlipError::DuplicateSimplex
                | FlipError::NonManifoldFacet
                | FlipError::SimplexCreation(_)),
            ) => {
                log_apply_skip(&err);
                return Ok(true);
            }
            Err(e) => return Err(e.into()),
        };
    *last_applied_flip = Some(LastAppliedFlip::from_applied_flip(&applied));
    let info = applied.info;
    if repair_trace_enabled() {
        tracing::debug!(
            "[repair] apply inverse k=2 flip: kind={:?} direction={:?} removed_face={:?} inserted_face={:?} removed_simplices={:?} new_simplices={:?}",
            info.kind,
            info.direction,
            info.removed_face_vertices,
            info.inserted_face_vertices,
            info.removed_simplices,
            info.new_simplices,
        );
    }
    stats.flips_performed += 1;
    diagnostics.record_flip_signature(signature);
    record_touched_simplices(touched_simplices, touched_simplex_set, &info.new_simplices);

    enqueue_new_simplices_for_repair(tds, &info.new_simplices, queues, stats)?;

    Ok(true)
}

/// Runs one queued triangle repair for inverse k=3 moves, which only appear once
/// D is high enough for a triangle star to be replaced.
#[expect(
    clippy::too_many_arguments,
    reason = "Repair step threads queues, diagnostics, and config explicitly"
)]
#[expect(
    clippy::too_many_lines,
    reason = "Repair step contains inline tracing and queue handling for diagnostics"
)]
fn run_next_triangle_repair_step<K, U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    trial_workspace: &mut Tds<U, V, D>,
    kernel: &K,
    queues: &mut RepairQueues,
    stats: &mut DelaunayRepairStats,
    max_flips: usize,
    config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
    last_applied_flip: &mut Option<LastAppliedFlip>,
    touched_simplices: &mut SimplexKeyBuffer,
    touched_simplex_set: &mut FastHashSet<SimplexKey>,
) -> Result<bool, DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    let Some((triangle, key)) = pop_queue(&mut queues.triangle_queue, config.queue_order) else {
        return Ok(false);
    };
    queues.triangle_queued.remove(&key);
    stats.facets_checked += 1;

    // Shared trace tail for build-k=3-triangle skip arms below.
    let log_build_skip = |err: &FlipError| {
        if repair_trace_enabled() {
            tracing::debug!(
                "[repair] skip inverse k=3 triangle (triangle={triangle:?}) reason={err}"
            );
        }
    };
    let context = match build_k3_flip_context_from_triangle(tds, triangle) {
        Ok(ctx) => ctx,
        Err(ref err) if let FlipError::MissingSimplex { simplex_key } = err => {
            diagnostics.record_missing_simplex_skip(MissingSimplexSkipSample {
                location: RepairSkipLocation::Triangle(triangle),
                simplex_key: *simplex_key,
            });
            log_build_skip(err);
            return Ok(true);
        }
        Err(
            ref err @ (FlipError::InvalidTriangleMultiplicity { .. }
            | FlipError::InvalidTriangleAdjacency { .. }
            | FlipError::MissingVertex { .. }),
        ) => {
            log_build_skip(err);
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };

    let frame_simplex = removed_simplex_frame(&context.removed_simplices)?;
    let violates = match delaunay_violation_k3_for_ridge(
        tds,
        kernel,
        &GlobalTopology::DEFAULT.model(),
        &context.inserted_face_vertices,
        &context.removed_face_vertices,
        &context.removed_simplices,
        Some(frame_simplex),
        config,
        diagnostics,
    ) {
        Ok(violates) => violates,
        Err(FlipError::PredicateFailure { .. }) => {
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };

    // Only flip if the target (3-simplex) configuration is locally Delaunay.
    if violates {
        return Ok(true);
    }
    diagnostics.record_applicable_repair_site();

    if would_immediately_reverse_last_flip::<D>(
        last_applied_flip.as_ref(),
        BistellarFlipKind::k3(D).inverse(),
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
    ) {
        if repair_trace_enabled() {
            tracing::debug!(
                "[repair] skip inverse k=3 flip (triangle={triangle:?}) reason=immediate reverse of prior flip"
            );
        }
        return Ok(true);
    }
    let kind = BistellarFlipKind::k3(D).inverse();
    let signature = flip_signature(
        kind,
        context.direction,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
    );
    check_flip_cycle(
        tds,
        FlipCycleContext::from_validated_flip(
            signature,
            kind,
            context.direction,
            &context.removed_face_vertices,
            &context.inserted_face_vertices,
        ),
        diagnostics,
        stats,
        max_flips,
        config,
    )?;

    // Enforce flip budget before applying the flip so that Some(0) means zero flips.
    if stats.flips_performed >= max_flips {
        return Err(non_convergent_error(max_flips, stats, diagnostics, config));
    }

    // Shared trace tail for apply-inverse-k=3 skip arms below.
    let log_apply_skip = |err: &FlipError| {
        if repair_trace_enabled() {
            tracing::debug!("[repair] skip inverse k=3 flip (triangle={triangle:?}) reason={err}");
            tracing::debug!(
                "[repair] skip inverse k=3 flip context removed_face={:?} inserted_face={:?} removed_simplices={:?}",
                context.removed_face_vertices,
                context.inserted_face_vertices,
                context.removed_simplices,
            );
        }
    };
    let applied =
        match apply_delaunay_flip_dynamic_in_workspace(tds, kind.k(), &context, trial_workspace) {
            Ok(applied) => applied,
            Err(err) if let FlipError::InsertedSimplexAlreadyExists { .. } = &err => {
                diagnostics.record_inserted_simplex_skip(InsertedSimplexSkipSample {
                    location: RepairSkipLocation::Triangle(triangle),
                    removed_face: vertex_key_list(&context.removed_face_vertices),
                    inserted_face: vertex_key_list(&context.inserted_face_vertices),
                });
                log_apply_skip(&err);
                return Ok(true);
            }
            Err(
                err @ (FlipError::DegenerateSimplex
                | FlipError::NegativeOrientation { .. }
                | FlipError::DuplicateSimplex
                | FlipError::NonManifoldFacet
                | FlipError::SimplexCreation(_)),
            ) => {
                log_apply_skip(&err);
                return Ok(true);
            }
            Err(e) => return Err(e.into()),
        };
    *last_applied_flip = Some(LastAppliedFlip::from_applied_flip(&applied));
    let info = applied.info;
    if repair_trace_enabled() {
        tracing::debug!(
            "[repair] apply inverse k=3 flip: kind={:?} direction={:?} removed_face={:?} inserted_face={:?} removed_simplices={:?} new_simplices={:?}",
            info.kind,
            info.direction,
            info.removed_face_vertices,
            info.inserted_face_vertices,
            info.removed_simplices,
            info.new_simplices,
        );
    }
    stats.flips_performed += 1;
    diagnostics.record_flip_signature(signature);
    record_touched_simplices(touched_simplices, touched_simplex_set, &info.new_simplices);

    enqueue_new_simplices_for_repair(tds, &info.new_simplices, queues, stats)?;

    Ok(true)
}

/// Runs one queued facet repair because k=2 facet flips are the primary local
/// repair move across supported dimensions.
#[expect(
    clippy::too_many_arguments,
    reason = "Repair step threads queues, diagnostics, and config explicitly"
)]
#[expect(
    clippy::too_many_lines,
    reason = "Repair step contains inline tracing and queue handling for diagnostics"
)]
fn run_next_facet_repair_step<K, U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    trial_workspace: &mut Tds<U, V, D>,
    kernel: &K,
    queues: &mut RepairQueues,
    stats: &mut DelaunayRepairStats,
    max_flips: usize,
    config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
    last_applied_flip: &mut Option<LastAppliedFlip>,
    touched_simplices: &mut SimplexKeyBuffer,
    touched_simplex_set: &mut FastHashSet<SimplexKey>,
) -> Result<bool, DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    let Some((facet, key)) = pop_queue(&mut queues.facet_queue, config.queue_order) else {
        return Ok(false);
    };
    queues.facet_queued.remove(&key);
    let facet = queues.facet_handles.remove(&key).unwrap_or(facet);
    let Some(facet) = resolve_facet_handle_for_key(tds, facet, key) else {
        return Ok(true);
    };
    stats.facets_checked += 1;

    // Shared trace tail for build-k=2-facet skip arms below.
    let log_build_skip = |err: &FlipError| {
        if repair_trace_enabled() {
            tracing::debug!("[repair] skip k=2 facet (facet={facet:?}) reason={err}");
        }
    };
    let context = match build_k2_flip_context(tds, facet) {
        Ok(ctx) => ctx,
        Err(ref err) if let FlipError::MissingSimplex { simplex_key } = err => {
            diagnostics.record_missing_simplex_skip(MissingSimplexSkipSample {
                location: RepairSkipLocation::Facet(facet),
                simplex_key: *simplex_key,
            });
            log_build_skip(err);
            return Ok(true);
        }
        Err(
            ref err @ (FlipError::BoundaryFacet { .. }
            | FlipError::MissingNeighbor { .. }
            | FlipError::InvalidFacetAdjacency { .. }
            | FlipError::InvalidFacetIndex { .. }),
        ) => {
            log_build_skip(err);
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };

    let topology_model = GlobalTopology::DEFAULT.model();
    let violates =
        match is_delaunay_violation_k2(tds, kernel, &topology_model, &context, config, diagnostics)
        {
            Ok(violates) => violates,
            Err(FlipError::PredicateFailure { .. }) => {
                return Ok(true);
            }
            Err(e) => return Err(e.into()),
        };

    if !violates {
        return Ok(true);
    }
    diagnostics.record_applicable_repair_site();

    if would_immediately_reverse_last_flip::<D>(
        last_applied_flip.as_ref(),
        BistellarFlipKind::k2(D),
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
    ) {
        if repair_trace_enabled() {
            tracing::debug!(
                "[repair] skip k=2 flip (facet={facet:?}) reason=immediate reverse of prior flip"
            );
        }
        return Ok(true);
    }

    let kind = BistellarFlipKind::k2(D);
    let signature = flip_signature(
        kind,
        context.direction,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
    );
    check_flip_cycle(
        tds,
        FlipCycleContext::from_validated_flip(
            signature,
            kind,
            context.direction,
            &context.removed_face_vertices,
            &context.inserted_face_vertices,
        ),
        diagnostics,
        stats,
        max_flips,
        config,
    )?;

    // Enforce flip budget before applying the flip so that Some(0) means zero flips.
    if stats.flips_performed >= max_flips {
        return Err(non_convergent_error(max_flips, stats, diagnostics, config));
    }

    // Shared trace tail for apply-k=2-facet skip arms below.
    let log_apply_skip = |err: &FlipError| {
        if env::var_os("DELAUNAY_REPAIR_DEBUG_FACETS").is_some() {
            tracing::debug!(
                facet = ?facet,
                reason = %err,
                removed_face = ?context.removed_face_vertices,
                inserted_face = ?context.inserted_face_vertices,
                removed_simplices = ?context.removed_simplices,
                "[repair] skip k=2 flip"
            );
        }
        if repair_trace_enabled() {
            tracing::debug!("[repair] skip k=2 flip (facet={facet:?}) reason={err}");
            tracing::debug!(
                "[repair] skip k=2 flip context removed_face={:?} inserted_face={:?} removed_simplices={:?}",
                context.removed_face_vertices,
                context.inserted_face_vertices,
                context.removed_simplices,
            );
        }
    };
    let applied = match apply_delaunay_flip_k2_in_workspace(tds, &context, trial_workspace) {
        Ok(applied) => applied,
        Err(err) if let FlipError::InsertedSimplexAlreadyExists { .. } = &err => {
            diagnostics.record_inserted_simplex_skip(InsertedSimplexSkipSample {
                location: RepairSkipLocation::Facet(facet),
                removed_face: vertex_key_list(&context.removed_face_vertices),
                inserted_face: vertex_key_list(&context.inserted_face_vertices),
            });
            log_apply_skip(&err);
            return Ok(true);
        }
        Err(
            err @ (FlipError::DegenerateSimplex
            | FlipError::NegativeOrientation { .. }
            | FlipError::DuplicateSimplex
            | FlipError::NonManifoldFacet
            | FlipError::SimplexCreation(_)),
        ) => {
            log_apply_skip(&err);
            return Ok(true);
        }
        Err(e) => return Err(e.into()),
    };
    *last_applied_flip = Some(LastAppliedFlip::from_applied_flip(&applied));
    let info = applied.info;
    if repair_trace_enabled() {
        tracing::debug!(
            "[repair] apply k=2 flip: kind={:?} direction={:?} removed_face={:?} inserted_face={:?} removed_simplices={:?} new_simplices={:?}",
            info.kind,
            info.direction,
            info.removed_face_vertices,
            info.inserted_face_vertices,
            info.removed_simplices,
            info.new_simplices,
        );
    }
    stats.flips_performed += 1;
    diagnostics.record_flip_signature(signature);
    record_touched_simplices(touched_simplices, touched_simplex_set, &info.new_simplices);

    enqueue_new_simplices_for_repair(tds, &info.new_simplices, queues, stats)?;

    Ok(true)
}

/// Extracts facet vertices by omitted slot so facet hashing matches the simplex's
/// current vertex ordering.
fn facet_vertices_from_simplex<V, const D: usize>(
    simplex: &Simplex<V, D>,
    facet_index: usize,
) -> SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> {
    let mut vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(D + 1);
    for (i, &vkey) in simplex.vertices().iter().enumerate() {
        if i != facet_index {
            vertices.push(vkey);
        }
    }
    vertices
}

/// Extracts ridge vertices by omitted slots so ridge handles remain compact but
/// can still be converted into stable vertex sets.
fn ridge_vertices_from_simplex<V, const D: usize>(
    simplex: &Simplex<V, D>,
    omit_a: usize,
    omit_b: usize,
) -> SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> {
    let mut vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(D + 1);
    for (i, &vkey) in simplex.vertices().iter().enumerate() {
        if i != omit_a && i != omit_b {
            vertices.push(vkey);
        }
    }
    vertices
}

/// Finds the two vertices opposite a ridge in one simplex while validating that the
/// requested ridge is actually incident to that simplex.
fn simplex_extras_for_ridge<V, const D: usize>(
    simplex_key: SimplexKey,
    simplex: &Simplex<V, D>,
    ridge: &SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
) -> Result<SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>, FlipError> {
    if !ridge.iter().all(|v| simplex.contains_vertex(*v)) {
        return Err(FlipError::InvalidRidgeAdjacency { simplex_key });
    }

    let mut extras: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(2);
    for &vkey in simplex.vertices() {
        if !ridge.contains(&vkey) {
            extras.push(vkey);
        }
    }
    Ok(extras)
}

/// Identifies the one opposite vertex needed to complete a k=3 cavity cycle.
fn missing_opposite_for_simplex(
    extras: &[VertexKey; 2],
    opposites: &[VertexKey; 3],
) -> Option<VertexKey> {
    opposites
        .iter()
        .copied()
        .find(|v| *v != extras[0] && *v != extras[1])
}

/// Walks the neighbor graph around a ridge so k=3 context construction uses the
/// local star rather than a global incidence scan.
///
/// When `max_simplices` is set, the walk stops after discovering more than that
/// many incident simplices. Repair uses this to reject non-k=3 edge stars as soon
/// as they are known to be too large, while public flip construction leaves the
/// value unset to preserve exact multiplicity diagnostics.
fn collect_simplices_around_ridge<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    start_simplex: SimplexKey,
    ridge: &SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    max_simplices: Option<usize>,
) -> Result<SimplexKeyBuffer, FlipError>
where
    U: DataType,
    V: DataType,
{
    let mut queue: SimplexKeyBuffer = SimplexKeyBuffer::new();
    let mut visited: SimplexKeyBuffer = SimplexKeyBuffer::new();
    let mut simplices: SimplexKeyBuffer = SimplexKeyBuffer::new();
    let mut queue_cursor = 0usize;

    queue.push(start_simplex);

    while queue_cursor < queue.len() {
        let simplex_key = queue[queue_cursor];
        queue_cursor += 1;

        if visited.contains(&simplex_key) {
            continue;
        }
        visited.push(simplex_key);

        let simplex = tds
            .simplex(simplex_key)
            .ok_or(FlipError::MissingSimplex { simplex_key })?;
        if !ridge.iter().all(|v| simplex.contains_vertex(*v)) {
            return Err(FlipError::InvalidRidgeAdjacency { simplex_key });
        }

        let mut omit_indices: SmallBuffer<usize, 2> = SmallBuffer::with_capacity(2);
        for (i, &vkey) in simplex.vertices().iter().enumerate() {
            if !ridge.contains(&vkey) {
                omit_indices.push(i);
            }
        }
        if omit_indices.len() != 2 {
            return Err(FlipError::InvalidRidgeAdjacency { simplex_key });
        }

        simplices.push(simplex_key);
        if max_simplices.is_some_and(|limit| simplices.len() > limit) {
            return Ok(simplices);
        }

        for &omit_idx in &omit_indices {
            if let Some(neighbor_key) = simplex.neighbor_key(omit_idx).flatten() {
                let Some(neighbor_simplex) = tds.simplex(neighbor_key) else {
                    return Err(FlipError::DanglingRidgeNeighbor {
                        simplex_key,
                        neighbor_key,
                    });
                };
                if !ridge.iter().all(|v| neighbor_simplex.contains_vertex(*v)) {
                    return Err(FlipError::InvalidRidgeAdjacency { simplex_key });
                }
                queue.push(neighbor_key);
            }
        }
    }

    Ok(simplices)
}

/// Returns a vertex's Euclidean point without applying topology-frame lifting.
fn vertex_point<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    vertex_key: VertexKey,
) -> Result<Point<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    let vertex = tds
        .vertex(vertex_key)
        .ok_or(FlipError::MissingVertex { vertex_key })?;
    Ok(*vertex.point())
}

/// Small per-predicate cache for Euclidean vertex coordinates.
struct EuclideanPointCache<const D: usize> {
    points: SmallBuffer<(VertexKey, Point<D>), MAX_PRACTICAL_DIMENSION_SIZE>,
}

impl<const D: usize> EuclideanPointCache<D> {
    /// Starts an empty cache for one local predicate evaluation.
    fn new() -> Self {
        Self {
            points: SmallBuffer::new(),
        }
    }
}

impl<const D: usize> EuclideanPointCache<D> {
    /// Returns a cached Euclidean point, loading it from the TDS on first use.
    fn point<U, V>(
        &mut self,
        tds: &Tds<U, V, D>,
        vertex_key: VertexKey,
    ) -> Result<Point<D>, FlipError>
    where
        U: DataType,
        V: DataType,
    {
        if let Some((_key, point)) = self.points.iter().find(|(key, _point)| *key == vertex_key) {
            return Ok(*point);
        }

        let point = vertex_point(tds, vertex_key)?;
        self.points.push((vertex_key, point));
        Ok(point)
    }

    /// Converts a small vertex-key slice into Euclidean points while sharing cache hits.
    fn points_for_vertices<U, V>(
        &mut self,
        tds: &Tds<U, V, D>,
        vertices: &[VertexKey],
    ) -> Result<SmallBuffer<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE>, FlipError>
    where
        U: DataType,
        V: DataType,
    {
        let mut points: SmallBuffer<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(vertices.len());
        for &vertex_key in vertices {
            points.push(self.point(tds, vertex_key)?);
        }
        Ok(points)
    }
}

/// Converts vertex keys to Euclidean points for predicates that do not need a
/// periodic frame.
fn vertices_to_points<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    vertices: &[VertexKey],
) -> Result<SmallBuffer<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE>, FlipError>
where
    U: DataType,
    V: DataType,
{
    let mut points: SmallBuffer<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(vertices.len());
    for &vkey in vertices {
        points.push(vertex_point(tds, vkey)?);
    }
    Ok(points)
}

/// Builds predicate points in one periodic frame so quotient-simplex coordinates
/// compare as lifted representatives.
fn vertices_to_points_with_optional_lift<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    topology_model: &GlobalTopologyModelAdapter<D>,
    vertices: &[VertexKey],
    source_simplex: Option<SimplexKey>,
    source_simplices: &[SimplexKey],
) -> Result<SmallBuffer<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE>, FlipError>
where
    U: DataType,
    V: DataType,
{
    let mut points: SmallBuffer<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(vertices.len());
    for &vkey in vertices {
        points.push(vertex_point_lifted_into_simplex(
            tds,
            topology_model,
            vkey,
            source_simplex,
            source_simplices,
        )?);
    }
    Ok(points)
}

/// Applies a simplex-local periodic offset when the vertex is already present in
/// the selected source simplex.
fn vertex_point_with_optional_lift<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    topology_model: &GlobalTopologyModelAdapter<D>,
    vertex_key: VertexKey,
    source_simplex: Option<SimplexKey>,
) -> Result<Point<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    let periodic_offset = if topology_model.supports_periodic_orientation_offsets() {
        match source_simplex {
            Some(simplex_key) => periodic_offset_for_simplex_vertex(tds, simplex_key, vertex_key)?,
            None => None,
        }
    } else {
        None
    };
    lift_vertex_point(tds, topology_model, vertex_key, periodic_offset)
}

/// Lifts a vertex into a target simplex's frame, aligning from neighboring source
/// simplices instead of falling back to bare periodic coordinates.
fn vertex_point_lifted_into_simplex<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    topology_model: &GlobalTopologyModelAdapter<D>,
    vertex_key: VertexKey,
    target_simplex: Option<SimplexKey>,
    source_simplices: &[SimplexKey],
) -> Result<Point<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    let Some(target_simplex_key) = target_simplex else {
        return vertex_point_with_optional_lift(tds, topology_model, vertex_key, None);
    };

    if !topology_model.supports_periodic_orientation_offsets() {
        return lift_vertex_point(tds, topology_model, vertex_key, None);
    }

    let offset =
        periodic_offset_lifted_into_simplex(tds, vertex_key, target_simplex_key, source_simplices)?;
    lift_vertex_point(tds, topology_model, vertex_key, Some(offset))
}

/// Aligns a vertex's periodic offset into a target simplex frame.
fn periodic_offset_lifted_into_simplex<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    vertex_key: VertexKey,
    target_simplex_key: SimplexKey,
    source_simplices: &[SimplexKey],
) -> Result<[i8; D], FlipError> {
    let target_offset = periodic_offset_for_simplex_vertex(tds, target_simplex_key, vertex_key)?;
    if let Some(offset) = target_offset {
        return Ok(offset);
    }

    let target_simplex = tds
        .simplex(target_simplex_key)
        .ok_or(FlipError::MissingSimplex {
            simplex_key: target_simplex_key,
        })?;
    let target_offsets = periodic_offsets_or_zero_frame(target_simplex_key, target_simplex)?;

    for &source_simplex_key in source_simplices {
        let Some(source_simplex) = tds.simplex(source_simplex_key) else {
            continue;
        };
        if !source_simplex.contains_vertex(vertex_key) {
            continue;
        }
        let source_offsets = periodic_offsets_or_zero_frame(source_simplex_key, source_simplex)?;
        let Some(source_vertex_index) = source_simplex
            .vertices()
            .iter()
            .position(|&vkey| vkey == vertex_key)
        else {
            continue;
        };
        let shared_indices = shared_vertex_indices(target_simplex, source_simplex);
        if shared_indices.is_empty() {
            continue;
        }
        let source_vertex_offset = source_offsets[source_vertex_index];
        let mut aligned_offset: Option<[i8; D]> = None;
        for (target_shared_index, source_shared_index) in shared_indices {
            let target_offset = target_offsets[target_shared_index];
            let source_offset = source_offsets[source_shared_index];
            let candidate_offset =
                align_periodic_offset(source_vertex_offset, source_offset, target_offset)?;
            if let Some(expected_offset) = aligned_offset {
                if candidate_offset != expected_offset {
                    return Err(FlipContextError::ConflictingPeriodicFrameTranslation {
                        vertex_key,
                        source_simplex_key,
                        target_simplex_key,
                        expected_offset: expected_offset.into(),
                        found_offset: candidate_offset.into(),
                    }
                    .into());
                }
            } else {
                aligned_offset = Some(candidate_offset);
            }
        }
        if let Some(offset) = aligned_offset {
            return Ok(offset);
        }
    }

    Err(FlipContextError::PeriodicVertexAlignmentFailed {
        vertex_key,
        target_simplex_key,
    }
    .into())
}

/// Centralizes topology-model lifting so missing vertices and non-liftable
/// offsets become typed flip errors.
fn lift_vertex_point<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    topology_model: &GlobalTopologyModelAdapter<D>,
    vertex_key: VertexKey,
    periodic_offset: Option<[i8; D]>,
) -> Result<Point<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    let vertex = tds
        .vertex(vertex_key)
        .ok_or(FlipError::MissingVertex { vertex_key })?;
    let lifted_coords = topology_model
        .lift_for_orientation(*vertex.point().coords(), periodic_offset)
        .map_err(|source| FlipPredicateError::PeriodicVertexLift {
            vertex_key,
            details: source.to_string(),
        })?;
    Point::try_new(lifted_coords).map_err(|source| {
        FlipPredicateError::PeriodicVertexLift {
            vertex_key,
            details: source.to_string(),
        }
        .into()
    })
}

/// Looks up the offset paired with a vertex slot, preserving the invariant that
/// periodic offsets are indexed exactly like simplex vertices.
fn periodic_offset_for_simplex_vertex<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
    vertex_key: VertexKey,
) -> Result<Option<[i8; D]>, FlipError> {
    let simplex = tds
        .simplex(simplex_key)
        .ok_or(FlipError::MissingSimplex { simplex_key })?;
    let offsets = periodic_offsets_or_zero_frame(simplex_key, simplex)?;
    Ok(simplex
        .vertices()
        .iter()
        .position(|&vkey| vkey == vertex_key)
        .map(|index| offsets[index]))
}

/// Borrows stored periodic offsets, or treats a periodic simplex without explicit
/// offsets as a zero-offset frame.
fn periodic_offsets_or_zero_frame<V, const D: usize>(
    simplex_key: SimplexKey,
    simplex: &Simplex<V, D>,
) -> Result<Cow<'_, [[i8; D]]>, FlipError> {
    let offsets = simplex.periodic_vertex_offsets().map_or_else(
        // The fallback frame is synthesized locally, so `Cow::Owned` keeps the
        // temporary vector alive while the stored-offset path can stay borrowed.
        || Cow::Owned(vec![[0_i8; D]; simplex.number_of_vertices()]),
        Cow::Borrowed,
    );
    validate_periodic_offset_len(simplex_key, simplex, offsets.as_ref())?;
    Ok(offsets)
}

/// Rejects malformed quotient simplices before offset indexing can desynchronize
/// vertices from their lifted representatives.
fn validate_periodic_offset_len<V, const D: usize>(
    simplex_key: SimplexKey,
    simplex: &Simplex<V, D>,
    offsets: &[[i8; D]],
) -> Result<(), FlipError> {
    if offsets.len() == simplex.number_of_vertices() {
        return Ok(());
    }
    Err(FlipContextError::PeriodicOffsetCountMismatch {
        simplex_key,
        offset_count: offsets.len(),
        vertex_count: simplex.number_of_vertices(),
    }
    .into())
}

/// Finds every common vertex to act as a consistency check when aligning two
/// periodic simplex frames.
fn shared_vertex_indices<V, const D: usize>(
    target_simplex: &Simplex<V, D>,
    source_simplex: &Simplex<V, D>,
) -> SmallBuffer<(usize, usize), MAX_PRACTICAL_DIMENSION_SIZE> {
    let mut shared = SmallBuffer::new();
    for (target_index, &target_vertex) in target_simplex.vertices().iter().enumerate() {
        if let Some(source_index) = source_simplex
            .vertices()
            .iter()
            .position(|&source_vertex| source_vertex == target_vertex)
        {
            shared.push((target_index, source_index));
        }
    }
    shared
}

/// Aligns a periodic vertex offset from a source simplex's frame into a target
/// simplex's frame so cross-simplex insphere predicates see consistent lifted
/// coordinates.
fn align_periodic_offset<const D: usize>(
    source_vertex_offset: [i8; D],
    source_reference_offset: [i8; D],
    target_reference_offset: [i8; D],
) -> Result<[i8; D], FlipError> {
    let mut aligned = [0_i8; D];
    for axis in 0..D {
        let delta = target_reference_offset[axis]
            .checked_sub(source_reference_offset[axis])
            .ok_or(FlipContextError::PeriodicOffsetSubtractionOverflow { axis })?;
        aligned[axis] = source_vertex_offset[axis]
            .checked_add(delta)
            .ok_or(FlipContextError::PeriodicOffsetAdditionOverflow { axis })?;
    }
    Ok(aligned)
}

/// Reuses an existing removed simplex as the predicate frame when the candidate
/// simplex exactly matches that simplex.
fn matching_source_simplex<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    vertices: &[VertexKey],
    source_simplices: &[SimplexKey],
) -> Option<SimplexKey>
where
    U: DataType,
    V: DataType,
{
    source_simplices.iter().copied().find(|&simplex_key| {
        tds.simplex(simplex_key).is_some_and(|simplex| {
            simplex.number_of_vertices() == vertices.len()
                && vertices
                    .iter()
                    .all(|&vertex_key| simplex.contains_vertex(vertex_key))
        })
    })
}

/// Selects a concrete removed-simplex frame for inverse predicates, where no
/// forward replacement simplex may match exactly.
fn removed_simplex_frame(source_simplices: &[SimplexKey]) -> Result<SimplexKey, FlipError> {
    source_simplices
        .first()
        .copied()
        .ok_or_else(|| FlipContextError::MissingRemovedSimplexFrame.into())
}

#[derive(Debug, Default)]
struct FlipTopologyIndex {
    /// Candidate simplex signature → the first existing simplex that matches it.
    ///
    /// The number of candidate simplices per flip is small (≤ D+1), so a flat buffer is
    /// faster than a `HashMap` in this hot path.
    duplicate_signature_to_simplex: SmallBuffer<(u64, SimplexKey), MAX_PRACTICAL_DIMENSION_SIZE>,

    /// Candidate *internal* facet hash → topology metadata, sorted by hash for binary search.
    ///
    /// We only track internal facets (facets that contain the inserted face). Boundary facets
    /// lie on the cavity boundary and cannot become non-manifold when the surrounding topology is
    /// valid.
    candidate_facet_info: SmallBuffer<
        (u64, CandidateFacetInfo),
        { MAX_PRACTICAL_DIMENSION_SIZE * MAX_PRACTICAL_DIMENSION_SIZE },
    >,
}

#[derive(Debug, Clone, Copy)]
struct CandidateFacetInfo {
    existing_count: u8,
    last_simplex: Option<SimplexKey>,
}

/// Sorts stable slotmap key values before hashing so signatures are independent
/// of local simplex vertex order.
fn sorted_vertex_key_values(
    vertices: &[VertexKey],
) -> SmallBuffer<u64, MAX_PRACTICAL_DIMENSION_SIZE> {
    let mut key_values: SmallBuffer<u64, MAX_PRACTICAL_DIMENSION_SIZE> =
        vertices.iter().map(|key| key.data().as_ffi()).collect();
    key_values.sort_unstable();
    key_values
}

/// Hashes a complete simplex vertex set for duplicate-simplex detection during flips.
fn simplex_signature(vertices: &[VertexKey]) -> u64 {
    let key_values = sorted_vertex_key_values(vertices);
    stable_hash_u64_slice(&key_values)
}

/// Builds the small topology index needed to reject duplicate simplices and
/// non-manifold internal facets without repeated global scans.
fn build_flip_topology_index<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    new_simplex_vertices: &[SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>],
    removed_simplices: &[SimplexKey],
    inserted_face_vertices: &[VertexKey],
) -> FlipTopologyIndex
where
    U: DataType,
    V: DataType,
{
    let inserted_values = sorted_vertex_key_values(inserted_face_vertices);

    let mut candidate_simplex_signatures: SmallBuffer<u64, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(new_simplex_vertices.len());

    let mut candidate_facet_info: SmallBuffer<
        (u64, CandidateFacetInfo),
        { MAX_PRACTICAL_DIMENSION_SIZE * MAX_PRACTICAL_DIMENSION_SIZE },
    > = SmallBuffer::new();

    // Seed the facet map with the facets that will exist after the flip.
    for vertices in new_simplex_vertices {
        let simplex_values = sorted_vertex_key_values(vertices);
        candidate_simplex_signatures.push(stable_hash_u64_slice(&simplex_values));

        let mut facet_values: SmallBuffer<u64, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(simplex_values.len().saturating_sub(1));
        for omit_idx in 0..simplex_values.len() {
            facet_values.clear();
            for (i, &val) in simplex_values.iter().enumerate() {
                if i != omit_idx {
                    facet_values.push(val);
                }
            }

            let facet_hash = stable_hash_u64_slice(&facet_values);
            let internal = inserted_values
                .iter()
                .all(|v| facet_values.binary_search(v).is_ok());

            // Only internal facets can become non-manifold: boundary facets are part of the cavity
            // boundary and already exist in the surrounding triangulation.
            if !internal {
                continue;
            }

            // Intentional hash-only dedup (no vertex-level tie-break): a 64-bit collision is
            // astronomically unlikely, and avoiding extra comparisons keeps this hot path fast.
            if candidate_facet_info
                .iter()
                .any(|(hash, _info)| *hash == facet_hash)
            {
                continue;
            }

            candidate_facet_info.push((
                facet_hash,
                CandidateFacetInfo {
                    existing_count: 0,
                    last_simplex: None,
                },
            ));
        }
    }

    candidate_facet_info.sort_unstable_by_key(|(hash, _info)| *hash);

    let mut duplicate_signature_to_simplex: SmallBuffer<
        (u64, SimplexKey),
        MAX_PRACTICAL_DIMENSION_SIZE,
    > = SmallBuffer::new();

    let mut facet_values: SmallBuffer<u64, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(D);
    let mut simplex_values: SmallBuffer<u64, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(D + 1);

    // Scan existing simplices once.
    //
    // Both duplicate simplices and existing internal facets must contain all inserted-face vertices.
    for (simplex_key, simplex) in tds.simplices() {
        if removed_simplices.contains(&simplex_key) {
            continue;
        }
        if !inserted_face_vertices
            .iter()
            .all(|v| simplex.contains_vertex(*v))
        {
            continue;
        }

        simplex_values.clear();
        for key in simplex.vertices() {
            simplex_values.push(key.data().as_ffi());
        }
        simplex_values.sort_unstable();

        let signature = stable_hash_u64_slice(&simplex_values);
        if candidate_simplex_signatures.contains(&signature)
            && !duplicate_signature_to_simplex
                .iter()
                .any(|(s, _simplex_key)| *s == signature)
        {
            duplicate_signature_to_simplex.push((signature, simplex_key));
        }

        // If there are no internal facets to check, skip facet hashing.
        if candidate_facet_info.is_empty() {
            continue;
        }

        for omit_idx in 0..simplex_values.len() {
            facet_values.clear();
            for (i, &val) in simplex_values.iter().enumerate() {
                if i != omit_idx {
                    facet_values.push(val);
                }
            }
            let facet_hash = stable_hash_u64_slice(&facet_values);

            // Hash-only lookup (see comment above); collision risk is astronomically low.
            let Ok(idx) =
                candidate_facet_info.binary_search_by_key(&facet_hash, |(hash, _info)| *hash)
            else {
                continue;
            };
            let info = &mut candidate_facet_info[idx].1;

            if info.existing_count < 2 {
                info.existing_count += 1;
            }
            info.last_simplex = Some(simplex_key);
        }
    }

    FlipTopologyIndex {
        duplicate_signature_to_simplex,
        candidate_facet_info,
    }
}

/// Checks candidate simplices against the topology index before mutation so a flip
/// cannot introduce two simplices with the same vertex set.
fn flip_would_duplicate_simplex_any<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    vertices: &[VertexKey],
    topology: &FlipTopologyIndex,
) -> bool
where
    U: DataType,
    V: DataType,
{
    let signature = simplex_signature(vertices);
    let Some(simplex_key) = topology
        .duplicate_signature_to_simplex
        .iter()
        .find_map(|(s, ck)| (*s == signature).then_some(*ck))
    else {
        return false;
    };

    if env::var_os("DELAUNAY_REPAIR_DEBUG_FACETS").is_some() || repair_trace_enabled() {
        let mut target: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
            vertices.iter().copied().collect();
        target.sort_unstable();

        let existing_sorted = tds.simplex(simplex_key).map(|simplex| {
            let mut v: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                simplex.vertices().iter().copied().collect();
            v.sort_unstable();
            v
        });

        if env::var_os("DELAUNAY_REPAIR_DEBUG_FACETS").is_some() {
            tracing::debug!(
                "k=2 flip would duplicate existing simplex {simplex_key:?}; target={target:?}; existing={existing_sorted:?}"
            );
        }
        if repair_trace_enabled() {
            tracing::debug!(
                "[repair] flip would duplicate existing simplex {simplex_key:?}; target={target:?}; existing={existing_sorted:?}"
            );
        }
    }

    true
}

/// Checks candidate internal facets against existing incidence so a flip cannot
/// create facet multiplicity greater than two.
fn flip_would_create_nonmanifold_facets_any(
    vertices: &[VertexKey],
    topology: &FlipTopologyIndex,
) -> bool {
    let sorted_values = sorted_vertex_key_values(vertices);

    let mut sorted_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        vertices.iter().copied().collect();
    sorted_vertices.sort_unstable_by_key(|v| v.data().as_ffi());

    let mut facet_values: SmallBuffer<u64, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(sorted_values.len().saturating_sub(1));
    let mut facet_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(sorted_vertices.len().saturating_sub(1));

    for omit_idx in 0..sorted_values.len() {
        facet_values.clear();
        facet_vertices.clear();

        for (i, &value) in sorted_values.iter().enumerate() {
            if i != omit_idx {
                facet_values.push(value);
                facet_vertices.push(sorted_vertices[i]);
            }
        }

        let facet_hash = stable_hash_u64_slice(&facet_values);
        let Ok(idx) = topology
            .candidate_facet_info
            .binary_search_by_key(&facet_hash, |(hash, _info)| *hash)
        else {
            // Boundary facet: not tracked in the index.
            continue;
        };
        let info = &topology.candidate_facet_info[idx].1;

        if info.existing_count > 0 {
            if repair_trace_enabled() {
                tracing::debug!(
                    "[repair] flip would create non-manifold internal facet: facet={facet_vertices:?} shared_count={} last_simplex={:?}",
                    info.existing_count,
                    info.last_simplex,
                );
            }
            return true;
        }
    }

    false
}

/// Queues all interior facets of a simplex because k=2 repair is driven by shared
/// facet predicates.
fn enqueue_simplex_facets<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
    queue: &mut VecDeque<(FacetHandle, u64)>,
    queued: &mut FastHashSet<u64>,
    handles: &mut FastHashMap<u64, FacetHandle>,
    stats: &mut DelaunayRepairStats,
) -> Result<(), FlipError>
where
    U: DataType,
    V: DataType,
{
    let Some(simplex) = tds.simplex(simplex_key) else {
        return Ok(());
    };
    for facet_index in 0..simplex.number_of_vertices() {
        let handle = FacetHandle::from_validated(
            simplex_key,
            u8::try_from(facet_index).map_err(|_| FlipError::InvalidFacetIndex {
                simplex_key,
                facet_index: u8::MAX,
                vertex_count: simplex.number_of_vertices(),
            })?,
        );
        enqueue_facet(tds, handle, queue, queued, handles, stats);
    }
    Ok(())
}

/// Enqueues a facet by stable vertex hash so stale handles can be resolved after
/// slot swaps.
fn enqueue_facet<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    handle: FacetHandle,
    queue: &mut VecDeque<(FacetHandle, u64)>,
    queued: &mut FastHashSet<u64>,
    handles: &mut FastHashMap<u64, FacetHandle>,
    stats: &mut DelaunayRepairStats,
) where
    U: DataType,
    V: DataType,
{
    let Some(simplex) = tds.simplex(handle.simplex_key()) else {
        return;
    };

    let facet_index = usize::from(handle.facet_index());
    if facet_index >= simplex.number_of_vertices() {
        return;
    }

    let Some(_neighbor_key) = simplex
        .neighbor_key(facet_index)
        .flatten()
        .filter(|&nk| tds.contains_simplex(nk))
    else {
        return;
    };

    let facet_vertices = facet_vertices_from_simplex(simplex, facet_index);
    let key = facet_key_from_vertices(&facet_vertices);

    handles.insert(key, handle);
    if queued.insert(key) {
        queue.push_back((handle, key));
        stats.max_queue_len = stats.max_queue_len.max(queue.len());
    }
}

/// Queues simplex edges only in dimensions where inverse k=2 repair is admissible.
fn enqueue_simplex_edges<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
    queue: &mut VecDeque<(EdgeKey, u64)>,
    queued: &mut FastHashSet<u64>,
    stats: &mut DelaunayRepairStats,
) where
    U: DataType,
    V: DataType,
{
    if D < 4 {
        return;
    }

    let Some(simplex) = tds.simplex(simplex_key) else {
        return;
    };

    let vertices = simplex.vertices();
    let vertex_count = vertices.len();
    for i in 0..vertex_count {
        for j in (i + 1)..vertex_count {
            let edge = EdgeKey::from_validated_endpoints(vertices[i], vertices[j]);
            enqueue_edge(edge, queue, queued, stats);
        }
    }
}

/// Deduplicates inverse k=2 edge work by vertex-set hash across incident simplices.
fn enqueue_edge(
    edge: EdgeKey,
    queue: &mut VecDeque<(EdgeKey, u64)>,
    queued: &mut FastHashSet<u64>,
    stats: &mut DelaunayRepairStats,
) {
    let key = facet_key_from_vertices(&[edge.v0(), edge.v1()]);
    if queued.insert(key) {
        queue.push_back((edge, key));
        stats.max_queue_len = stats.max_queue_len.max(queue.len());
    }
}

/// Queues simplex triangles only in dimensions where inverse k=3 repair is
/// admissible.
fn enqueue_simplex_triangles<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
    queue: &mut VecDeque<(TriangleHandle, u64)>,
    queued: &mut FastHashSet<u64>,
    stats: &mut DelaunayRepairStats,
) where
    U: DataType,
    V: DataType,
{
    if D < 5 {
        return;
    }

    let Some(simplex) = tds.simplex(simplex_key) else {
        return;
    };

    let vertices = simplex.vertices();
    let vertex_count = vertices.len();
    for i in 0..vertex_count {
        for j in (i + 1)..vertex_count {
            for k in (j + 1)..vertex_count {
                let triangle =
                    TriangleHandle::from_validated_vertices(vertices[i], vertices[j], vertices[k]);
                enqueue_triangle(triangle, queue, queued, stats);
            }
        }
    }
}

/// Deduplicates inverse k=3 triangle work by vertex-set hash across incident
/// simplices.
fn enqueue_triangle(
    triangle: TriangleHandle,
    queue: &mut VecDeque<(TriangleHandle, u64)>,
    queued: &mut FastHashSet<u64>,
    stats: &mut DelaunayRepairStats,
) {
    let vertices = triangle.vertices();
    let key = facet_key_from_vertices(&vertices);
    if queued.insert(key) {
        queue.push_back((triangle, key));
        stats.max_queue_len = stats.max_queue_len.max(queue.len());
    }
}

/// Queues all ridges of a simplex because k=3 repair needs codimension-two local
/// stars.
fn enqueue_simplex_ridges<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
    queue: &mut VecDeque<(RidgeHandle, u64)>,
    queued: &mut FastHashSet<u64>,
    handles: &mut FastHashMap<u64, RidgeHandle>,
    stats: &mut DelaunayRepairStats,
) -> Result<(), FlipError>
where
    U: DataType,
    V: DataType,
{
    if D < 3 {
        return Ok(());
    }

    let Some(simplex) = tds.simplex(simplex_key) else {
        return Ok(());
    };

    let vertex_count = simplex.number_of_vertices();
    for i in 0..vertex_count {
        for j in (i + 1)..vertex_count {
            let handle = RidgeHandle::from_validated(
                simplex_key,
                u8::try_from(i).map_err(|_| FlipError::InvalidRidgeIndex {
                    simplex_key,
                    omit_a: u8::MAX,
                    omit_b: u8::MAX,
                    vertex_count,
                })?,
                u8::try_from(j).map_err(|_| FlipError::InvalidRidgeIndex {
                    simplex_key,
                    omit_a: u8::MAX,
                    omit_b: u8::MAX,
                    vertex_count,
                })?,
            );
            enqueue_ridge(tds, handle, queue, queued, handles, stats);
        }
    }

    Ok(())
}

/// Enqueues a ridge by stable vertex hash so post-flip slot swaps do not strand
/// stale ridge handles.
fn enqueue_ridge<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    handle: RidgeHandle,
    queue: &mut VecDeque<(RidgeHandle, u64)>,
    queued: &mut FastHashSet<u64>,
    handles: &mut FastHashMap<u64, RidgeHandle>,
    stats: &mut DelaunayRepairStats,
) where
    U: DataType,
    V: DataType,
{
    if D < 3 {
        return;
    }

    let Some(simplex) = tds.simplex(handle.simplex_key()) else {
        return;
    };

    let vertex_count = simplex.number_of_vertices();
    let omit_a = usize::from(handle.omit_a());
    let omit_b = usize::from(handle.omit_b());
    if omit_a >= vertex_count || omit_b >= vertex_count || omit_a == omit_b {
        return;
    }

    let ridge_vertices = ridge_vertices_from_simplex(simplex, omit_a, omit_b);
    if ridge_vertices.len() != D - 1 {
        return;
    }

    let key = facet_key_from_vertices(&ridge_vertices);
    handles.insert(key, handle);
    if queued.insert(key) {
        queue.push_back((handle, key));
        stats.max_queue_len = stats.max_queue_len.max(queue.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DelaunayTriangulation;
    use crate::core::algorithms::incremental_insertion::{
        DelaunayRepairFailureContext, repair_neighbor_pointers,
    };
    use crate::core::algorithms::locate::LocateResult;
    use crate::core::collections::Uuid;
    use crate::core::validation::TopologyGuarantee;
    use crate::geometry::kernel::{AdaptiveKernel, FastKernel};
    use crate::geometry::traits::coordinate::CoordinateConversionValue;
    use crate::repair::DelaunayRepairOperation;
    use crate::topology::traits::topological_space::ToroidalConstructionMode;
    use approx::assert_relative_eq;
    use proptest::prelude::*;
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use slotmap::KeyData;
    use std::assert_matches;
    use std::{
        error::Error as _,
        iter::once,
        mem::{align_of, size_of},
        sync::Once,
    };

    fn init_tracing() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let filter = tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
            let _ = tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_test_writer()
                .try_init();
        });
    }

    fn sample_heuristic_vertex_context() -> DelaunayRepairHeuristicVertexContext {
        DelaunayRepairHeuristicVertexContext {
            index: 3,
            vertex_uuid: Uuid::nil(),
            coordinates: CoordinateValues::from([1.0, 2.0]),
        }
    }

    /// Builds a simplex-basis vertex coordinate for dimension-generic flip tests.
    fn unit_vector<const D: usize>(index: usize) -> [f64; D] {
        let mut coords = [0.0; D];
        coords[index] = 1.0;
        coords
    }

    /// Places a test vertex on a chosen coordinate axis to create degenerate simplices.
    fn scaled_unit_vector<const D: usize>(index: usize, scale: f64) -> [f64; D] {
        let mut coords = [0.0; D];
        coords[index] = scale;
        coords
    }

    #[test]
    fn triangle_handle_rejects_duplicate_vertices() {
        let a = VertexKey::from(KeyData::from_ffi(1));
        let b = VertexKey::from(KeyData::from_ffi(2));

        assert_matches!(
            TriangleHandle::try_new(a, b, a),
            Err(TriangleHandleError::DuplicateVertices { vertices })
                if vertices == [a, b, a]
        );
    }

    /// Inserts the canonical D-simplex fixture shared by replacement-orientation tests.
    fn insert_standard_simplex_vertices<const D: usize>(
        tds: &mut Tds<(), (), D>,
    ) -> Vec<VertexKey> {
        let mut vertices = Vec::with_capacity(D + 1);
        vertices.push(
            tds.insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0; D]).unwrap(),
            )
            .unwrap(),
        );
        for axis in 0..D {
            vertices.push(
                tds.insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(unit_vector::<D>(axis)).unwrap(),
                )
                .unwrap(),
            );
        }
        vertices
    }

    /// Creates distinct periodic offsets so tests can verify slot-preserving swaps.
    fn periodic_test_offsets<const D: usize>(len: usize) -> Vec<[i8; D]> {
        let mut offsets = Vec::with_capacity(len);
        for index in 0..len {
            let mut offset = [0_i8; D];
            offset[index % D] = i8::try_from(index).expect("test offset index fits in i8");
            offsets.push(offset);
        }
        offsets
    }

    /// Asserts exact vertex-to-periodic-offset slot pairing independent of simplex orientation.
    fn assert_simplex_offsets_by_vertex<const D: usize>(
        tds: &Tds<(), (), D>,
        simplex_key: SimplexKey,
        expected_offsets: &[(VertexKey, [i8; D])],
    ) {
        let simplex = tds.simplex(simplex_key).unwrap();
        let offsets = simplex
            .periodic_vertex_offsets()
            .expect("simplex should carry periodic offsets");
        assert_eq!(offsets.len(), simplex.number_of_vertices());
        assert_eq!(expected_offsets.len(), simplex.number_of_vertices());
        for &(vertex_key, expected_offset) in expected_offsets {
            let index = simplex
                .vertices()
                .iter()
                .position(|&candidate| candidate == vertex_key)
                .expect("expected vertex should be present in simplex");
            assert_eq!(
                offsets[index], expected_offset,
                "unexpected periodic offset for vertex {vertex_key:?} in simplex {simplex_key:?}"
            );
        }
    }

    /// Verifies source-simplex orientation gating only accepts certified positive orderings.
    #[test]
    fn test_source_simplex_is_certified_positive_requires_source_and_positive_order() {
        let source_simplex = SimplexKey::from(KeyData::from_ffi(42));
        let positive = [
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0]).expect("finite point coordinates"),
        ];
        let negative = [positive[1], positive[0], positive[2]];

        assert!(source_simplex_is_certified_positive(
            Some(source_simplex),
            &positive
        ));
        assert!(!source_simplex_is_certified_positive(None, &positive));
        assert!(!source_simplex_is_certified_positive(
            Some(source_simplex),
            &negative,
        ));
    }

    /// Converts vertex-key slices into the fixed-capacity buffer used by flip helpers.
    fn vertex_key_buffer(
        vertices: &[VertexKey],
    ) -> SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> {
        vertices.iter().copied().collect()
    }

    /// Creates a non-axis-aligned point for high-dimensional roundtrip fixtures.
    fn skewed_point<const D: usize>() -> [f64; D] {
        let mut coords = [0.0; D];
        for (i, coord) in coords.iter_mut().enumerate().take(D) {
            let idx = f64::from(u32::try_from(i + 1).expect("index fits in u32"));
            *coord = 0.11 * idx;
        }
        coords
    }

    /// Builds a translated and scaled simplex-basis coordinate for proptests.
    fn translated_scaled_unit_vector<const D: usize>(
        index: usize,
        offset: f64,
        scale: f64,
    ) -> [f64; D] {
        let mut coords = [offset; D];
        coords[index] += scale;
        coords
    }

    /// Creates a translated non-axis-aligned point for k=3 flip proptests.
    fn translated_scaled_skewed_point<const D: usize>(offset: f64, scale: f64) -> [f64; D] {
        let mut coords = [offset; D];
        for (i, coord) in coords.iter_mut().enumerate().take(D) {
            let idx = f64::from(u32::try_from(i + 1).expect("index fits in u32"));
            *coord = (scale * 0.11).mul_add(idx, *coord);
        }
        coords
    }

    /// Returns inserted-face vertices after verifying the expected flip arity.
    fn inserted_face_vertices<const D: usize>(
        info: &FlipInfo<D>,
        expected: usize,
    ) -> Result<Vec<VertexKey>, TestCaseError> {
        let vertices: Vec<_> = info.inserted_face_vertices.iter().copied().collect();
        if vertices.len() != expected {
            return Err(TestCaseError::fail(format!(
                "flip reported {} inserted-face vertices, expected {expected}",
                vertices.len()
            )));
        }
        Ok(vertices)
    }

    fn to_dynamic<const D: usize, const K: usize>(context: FlipContext<D, K>) -> FlipContextDyn<D> {
        FlipContextDyn {
            removed_face_vertices: context.removed_face_vertices,
            inserted_face_vertices: context.inserted_face_vertices,
            removed_simplices: context.removed_simplices,
            direction: context.direction,
        }
    }

    macro_rules! gen_removed_simplex_snapshot_tests {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<test_snapshot_removed_simplex_vertices_captures_vertices_and_reports_missing_simplex_ $dim d>]() {
                    let mut tds: Tds<(), (), $dim> = Tds::empty();
                    let vertices = insert_standard_simplex_vertices::<$dim>(&mut tds);
                    let simplex_key = tds
                        .insert_simplex_with_mapping(Simplex::try_new_with_data(vertices.clone(), None).unwrap())
                        .unwrap();

                    let removed_simplices: SimplexKeyBuffer = std::iter::once(simplex_key).collect();
                    let snapshot = snapshot_removed_simplex_vertices(&tds, &removed_simplices).unwrap();
                    assert_eq!(snapshot.len(), 1);
                    assert_eq!(snapshot[0].iter().copied().collect::<Vec<_>>(), vertices);

                    let missing_simplex = SimplexKey::from(KeyData::from_ffi(999_999 + $dim));
                    let missing_simplices: SimplexKeyBuffer = std::iter::once(missing_simplex).collect();
                    let err = snapshot_removed_simplex_vertices(&tds, &missing_simplices).unwrap_err();
                    assert_matches!(
                        err,
                        FlipError::MissingSimplex { simplex_key } if simplex_key == missing_simplex
                    );
                }

                #[test]
                fn [<test_last_applied_flip_preserves_removed_simplex_vertex_snapshots_ $dim d>]() {
                    let removed_simplex = SimplexKey::from(KeyData::from_ffi(101 + $dim));
                    let new_simplex = SimplexKey::from(KeyData::from_ffi(102 + $dim));
                    let v1 = VertexKey::from(KeyData::from_ffi(201 + $dim));
                    let v2 = VertexKey::from(KeyData::from_ffi(202 + $dim));
                    let v3 = VertexKey::from(KeyData::from_ffi(203 + $dim));
                    let v4 = VertexKey::from(KeyData::from_ffi(204 + $dim));

                    let mut removed_simplex_vertices = RemovedSimplexVertexSnapshot::new();
                    removed_simplex_vertices.push([v1, v2, v3].into_iter().collect::<VertexKeyList>());

                    let applied = AppliedFlip::<$dim> {
                        info: FlipInfo {
                            kind: BistellarFlipKind::k2($dim),
                            direction: FlipDirection::Forward,
                            removed_simplices: std::iter::once(removed_simplex).collect(),
                            new_simplices: std::iter::once(new_simplex).collect(),
                            removed_face_vertices: [v3, v1].into_iter().collect(),
                            inserted_face_vertices: [v4, v2].into_iter().collect(),
                        },
                        removed_simplex_vertices,
                    };

                    let last = LastAppliedFlip::from_applied_flip(&applied);
                    assert_eq!(last.kind, BistellarFlipKind::k2($dim));
                    assert_eq!(
                        last.removed_face_vertices
                            .iter()
                            .copied()
                            .collect::<Vec<_>>(),
                        vec![v1, v3]
                    );
                    assert_eq!(
                        last.inserted_face_vertices
                            .iter()
                            .copied()
                            .collect::<Vec<_>>(),
                        vec![v2, v4]
                    );
                    assert_eq!(
                        last.removed_simplices.iter().copied().collect::<Vec<_>>(),
                        vec![removed_simplex]
                    );
                    assert_eq!(
                        last.new_simplices.iter().copied().collect::<Vec<_>>(),
                        vec![new_simplex]
                    );

                    let lines = last.removed_simplex_vertex_lines();
                    assert_eq!(lines.len(), 1);
                    assert!(lines[0].contains(&format!("{removed_simplex:?}: vertices=")));
                    assert!(!lines[0].contains("missing-snapshot"));

                    let mut placeholder =
                        LastAppliedFlip::from_validated_flip_faces(BistellarFlipKind::k2($dim), &[v1], &[v2]);
                    placeholder.removed_simplices.push(removed_simplex);
                    assert_eq!(
                        placeholder.removed_simplex_vertex_lines(),
                        vec![format!("{removed_simplex:?}: missing-snapshot")]
                    );
                }
            }
        };
    }

    gen_removed_simplex_snapshot_tests!(2);
    gen_removed_simplex_snapshot_tests!(3);
    gen_removed_simplex_snapshot_tests!(4);
    gen_removed_simplex_snapshot_tests!(5);

    struct RidgeDiagnosticFixture3d {
        tds: Tds<(), (), 3>,
        origin_vertex: VertexKey,
        x_axis_vertex: VertexKey,
        y_axis_vertex: VertexKey,
        upper_apex_vertex: VertexKey,
        lower_apex_vertex: VertexKey,
        upper_tetrahedron: SimplexKey,
        lower_neighbor: SimplexKey,
    }

    impl RidgeDiagnosticFixture3d {
        fn new() -> Self {
            let mut tds: Tds<(), (), 3> = Tds::empty();
            let origin_vertex = tds
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
                )
                .unwrap();
            let x_axis_vertex = tds
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
                )
                .unwrap();
            let y_axis_vertex = tds
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
                )
                .unwrap();
            let upper_apex_vertex = tds
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
                )
                .unwrap();
            let lower_apex_vertex = tds
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, -1.0]).unwrap(),
                )
                .unwrap();

            let upper_tetrahedron = tds
                .insert_simplex_with_mapping(
                    Simplex::try_new_with_data(
                        vec![
                            origin_vertex,
                            x_axis_vertex,
                            y_axis_vertex,
                            upper_apex_vertex,
                        ],
                        None,
                    )
                    .unwrap(),
                )
                .unwrap();
            let lower_neighbor = tds
                .insert_simplex_with_mapping(
                    Simplex::try_new_with_data(
                        vec![
                            origin_vertex,
                            x_axis_vertex,
                            y_axis_vertex,
                            lower_apex_vertex,
                        ],
                        None,
                    )
                    .unwrap(),
                )
                .unwrap();
            repair_neighbor_pointers(&mut tds).unwrap();

            Self {
                tds,
                origin_vertex,
                x_axis_vertex,
                y_axis_vertex,
                upper_apex_vertex,
                lower_apex_vertex,
                upper_tetrahedron,
                lower_neighbor,
            }
        }

        fn ridge_ab(&self) -> SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> {
            [self.origin_vertex, self.x_axis_vertex]
                .into_iter()
                .collect()
        }

        fn ridge_handle_abcd(&self) -> RidgeHandle {
            RidgeHandle::from_validated(self.upper_tetrahedron, 2, 3)
        }

        fn last_applied_flip(&self) -> LastAppliedFlip {
            let mut removed_simplex_vertices = RemovedSimplexVertexSnapshot::new();
            removed_simplex_vertices.push(
                [
                    self.origin_vertex,
                    self.x_axis_vertex,
                    self.y_axis_vertex,
                    self.upper_apex_vertex,
                ]
                .into_iter()
                .collect::<VertexKeyList>(),
            );

            let applied = AppliedFlip::<3> {
                info: FlipInfo {
                    kind: BistellarFlipKind::k2(3),
                    direction: FlipDirection::Forward,
                    removed_simplices: std::iter::once(self.upper_tetrahedron).collect(),
                    new_simplices: std::iter::once(self.lower_neighbor).collect(),
                    removed_face_vertices: [
                        self.origin_vertex,
                        self.x_axis_vertex,
                        self.y_axis_vertex,
                    ]
                    .into_iter()
                    .collect(),
                    inserted_face_vertices: [self.upper_apex_vertex, self.lower_apex_vertex]
                        .into_iter()
                        .collect(),
                },
                removed_simplex_vertices,
            };

            LastAppliedFlip::from_applied_flip(&applied)
        }
    }

    #[test]
    fn test_ridge_diagnostic_helpers_format_valid_missing_and_invalid_simplices() {
        init_tracing();
        let fixture = RidgeDiagnosticFixture3d::new();
        let ridge = fixture.ridge_ab();
        let simplex = fixture.tds.simplex(fixture.upper_tetrahedron).unwrap();

        let ridge_neighbors = ridge_neighbor_simplices_for_simplex(simplex, &ridge);
        assert!(
            ridge_neighbors.contains(&fixture.lower_neighbor),
            "shared-face neighbor should be visible from the ridge diagnostics"
        );

        let incident =
            ridge_incident_simplex_summary(&fixture.tds, fixture.upper_tetrahedron, &ridge);
        assert!(incident.contains(&format!("{:?}: extras=", fixture.upper_tetrahedron)));
        assert!(incident.contains("ridge_neighbors="));
        assert!(incident.contains(&format!("{:?}", fixture.lower_neighbor)));

        let simplex_summary = simplex_vertex_summary(&fixture.tds, fixture.upper_tetrahedron);
        assert!(simplex_summary.contains("vertices="));

        let facet_summary = facet_incident_simplex_summary(
            &fixture.tds,
            fixture.upper_tetrahedron,
            &[
                fixture.origin_vertex,
                fixture.x_axis_vertex,
                fixture.y_axis_vertex,
            ],
        );
        assert!(facet_summary.contains("opposite_vertices="));
        assert!(facet_summary.contains("neighbors="));

        let missing_simplex = SimplexKey::from(KeyData::from_ffi(999_901));
        assert_eq!(
            ridge_incident_simplex_summary(&fixture.tds, missing_simplex, &ridge),
            format!("{missing_simplex:?}: missing")
        );
        assert_eq!(
            simplex_vertex_summary(&fixture.tds, missing_simplex),
            format!("{missing_simplex:?}: missing")
        );
        assert_eq!(
            facet_incident_simplex_summary(
                &fixture.tds,
                missing_simplex,
                &[fixture.origin_vertex, fixture.x_axis_vertex],
            ),
            format!("{missing_simplex:?}: missing")
        );

        let missing_vertex = VertexKey::from(KeyData::from_ffi(999_902));
        let invalid_ridge: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
            [fixture.origin_vertex, missing_vertex]
                .into_iter()
                .collect();
        let invalid_summary =
            ridge_incident_simplex_summary(&fixture.tds, fixture.upper_tetrahedron, &invalid_ridge);
        assert!(invalid_summary.contains("extras_error="));
    }

    #[test]
    fn test_predecessor_diagnostic_summaries_include_flip_overlap() {
        init_tracing();
        let fixture = RidgeDiagnosticFixture3d::new();
        let last = fixture.last_applied_flip();

        let ridge_summary = predecessor_flip_summary(
            &fixture.tds,
            RidgeHandle::from_validated(fixture.lower_neighbor, 2, 3),
            &[fixture.lower_neighbor],
            &last,
        );
        assert!(ridge_summary.contains("ridge_simplex_is_new=true"));
        assert!(ridge_summary.contains("global_simplices_in_new"));
        assert!(ridge_summary.contains("predecessor_new_simplex_vertices"));

        let postcondition_summary = postcondition_facet_predecessor_summary(
            &fixture.tds,
            &[fixture.upper_tetrahedron, fixture.lower_neighbor],
            &last,
        );
        assert!(postcondition_summary.contains("incident_simplices_in_new"));
        assert!(postcondition_summary.contains("incident_simplices_in_removed"));
        assert!(postcondition_summary.contains("predecessor_removed_simplex_vertices"));
        assert!(!postcondition_summary.contains("missing-snapshot"));
    }

    #[test]
    fn test_debug_ridge_context_exercises_valid_missing_and_invalid_paths() {
        init_tracing();
        let fixture = RidgeDiagnosticFixture3d::new();
        let last = fixture.last_applied_flip();
        let mut diagnostics = RepairDiagnostics::default();

        debug_ridge_context(
            &fixture.tds,
            fixture.ridge_handle_abcd(),
            Some(2),
            &mut diagnostics,
            Some(&last),
        );
        assert_eq!(diagnostics.ridge_debug_emitted, 1);

        let missing_simplex = SimplexKey::from(KeyData::from_ffi(999_903));
        debug_ridge_context(
            &fixture.tds,
            RidgeHandle::from_validated(missing_simplex, 0, 1),
            None,
            &mut diagnostics,
            None,
        );
        assert_eq!(diagnostics.ridge_debug_emitted, 2);

        debug_ridge_context(
            &fixture.tds,
            RidgeHandle::from_validated(fixture.upper_tetrahedron, 0, 0),
            None,
            &mut diagnostics,
            None,
        );
        assert_eq!(diagnostics.ridge_debug_emitted, 3);
    }

    #[test]
    fn test_ridge_debug_limit_suppresses_after_attempt_budget() {
        let mut diagnostics = RepairDiagnostics {
            ridge_debug_emitted: RIDGE_DEBUG_LIMIT_DEFAULT,
            ..RepairDiagnostics::default()
        };

        assert!(!should_emit_ridge_debug(&mut diagnostics, Some(99)));
        assert_eq!(
            diagnostics.ridge_debug_emitted,
            RIDGE_DEBUG_LIMIT_DEFAULT + 1
        );
    }

    #[test]
    fn test_postcondition_facet_debug_context_is_noop_without_env_flag() {
        init_tracing();
        let fixture = RidgeDiagnosticFixture3d::new();
        let last = fixture.last_applied_flip();
        let context = FlipContext::<3, 2> {
            removed_face_vertices: [
                fixture.origin_vertex,
                fixture.x_axis_vertex,
                fixture.y_axis_vertex,
            ]
            .into_iter()
            .collect(),
            inserted_face_vertices: [fixture.upper_apex_vertex, fixture.lower_apex_vertex]
                .into_iter()
                .collect(),
            removed_simplices: [fixture.upper_tetrahedron, fixture.lower_neighbor]
                .into_iter()
                .collect(),
            direction: FlipDirection::Forward,
        };
        let mut diagnostics = RepairDiagnostics::default();

        debug_postcondition_facet_context(
            &fixture.tds,
            FacetHandle::from_validated(fixture.upper_tetrahedron, 3),
            &context,
            &mut diagnostics,
            Some(&last),
        );

        assert_eq!(diagnostics.postcondition_facet_debug_emitted, 0);
    }

    fn facet_index_for_edge_2d(
        tds: &Tds<(), (), 2>,
        simplex_key: SimplexKey,
        edge_start: VertexKey,
        edge_end: VertexKey,
    ) -> u8 {
        let simplex = tds
            .simplex(simplex_key)
            .expect("simplex key missing in TDS");
        for facet_idx in 0..simplex.number_of_vertices() {
            let facet = facet_vertices_from_simplex(simplex, facet_idx);
            if facet.len() == 2 && facet.contains(&edge_start) && facet.contains(&edge_end) {
                return u8::try_from(facet_idx).expect("facet index fits in u8");
            }
        }

        panic!("edge ({edge_start:?}, {edge_end:?}) not found in simplex {simplex_key:?}");
    }

    fn facet_index_for_face_3d(
        tds: &Tds<(), (), 3>,
        simplex_key: SimplexKey,
        face_v0: VertexKey,
        face_v1: VertexKey,
        face_v2: VertexKey,
    ) -> u8 {
        let simplex = tds
            .simplex(simplex_key)
            .expect("simplex key missing in TDS");
        for facet_idx in 0..simplex.number_of_vertices() {
            let facet = facet_vertices_from_simplex(simplex, facet_idx);
            if facet.len() == 3
                && facet.contains(&face_v0)
                && facet.contains(&face_v1)
                && facet.contains(&face_v2)
            {
                return u8::try_from(facet_idx).expect("facet index fits in u8");
            }
        }

        panic!("face ({face_v0:?}, {face_v1:?}, {face_v2:?}) not found in simplex {simplex_key:?}");
    }

    /// Assert that `robust_orientation` returns a non-degenerate sign for
    /// every new-simplex point set that a k=2 flip context would produce.
    fn assert_context_has_nonzero_robust_orientation(
        tds: &Tds<(), (), 2>,
        context: &FlipContext<2, 2>,
    ) {
        for &omit in &context.removed_face_vertices {
            let mut verts: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                SmallBuffer::with_capacity(3);
            verts.extend_from_slice(&context.inserted_face_vertices);
            for &v in &context.removed_face_vertices {
                if v != omit {
                    verts.push(v);
                }
            }
            let points = vertices_to_points(tds, &verts).unwrap();
            match robust_orientation(&points) {
                Ok(Orientation::POSITIVE | Orientation::NEGATIVE) => {}
                other => panic!("robust_orientation must resolve to ±1, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_resolve_facet_handle_for_key_remaps_after_slot_swap() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        let stale_handle = FacetHandle::from_validated(simplex_key, 0);
        let stable_key = {
            let simplex = tds.simplex(simplex_key).unwrap();
            let facet_vertices =
                facet_vertices_from_simplex(simplex, usize::from(stale_handle.facet_index()));
            facet_key_from_vertices(&facet_vertices)
        };

        // Reorder slots so the original index no longer identifies the same facet.
        tds.simplex_mut(simplex_key)
            .unwrap()
            .swap_vertex_slots(0, 1);

        let resolved = resolve_facet_handle_for_key(&tds, stale_handle, stable_key)
            .expect("facet handle should be recoverable by stable key");
        assert_eq!(resolved.simplex_key(), simplex_key);
        assert_eq!(usize::from(resolved.facet_index()), 1);

        let resolved_key = {
            let simplex = tds.simplex(simplex_key).unwrap();
            let facet_vertices =
                facet_vertices_from_simplex(simplex, usize::from(resolved.facet_index()));
            facet_key_from_vertices(&facet_vertices)
        };
        assert_eq!(resolved_key, stable_key);
    }

    #[test]
    fn test_resolve_ridge_handle_for_key_remaps_after_slot_swap() {
        let mut tds: Tds<(), (), 3> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v3], None).unwrap(),
            )
            .unwrap();
        let stale_handle = RidgeHandle::from_validated(simplex_key, 0, 1);
        let stable_key = {
            let simplex = tds.simplex(simplex_key).unwrap();
            let ridge_vertices = ridge_vertices_from_simplex(
                simplex,
                usize::from(stale_handle.omit_a()),
                usize::from(stale_handle.omit_b()),
            );
            facet_key_from_vertices(&ridge_vertices)
        };

        // Reorder slots so the original omit pair no longer identifies the same ridge.
        tds.simplex_mut(simplex_key)
            .unwrap()
            .swap_vertex_slots(0, 2);

        let resolved = resolve_ridge_handle_for_key(&tds, stale_handle, stable_key)
            .expect("ridge handle should be recoverable by stable key");
        assert_eq!(resolved.simplex_key(), simplex_key);
        assert_eq!((resolved.omit_a(), resolved.omit_b()), (1, 2));

        let resolved_key = {
            let simplex = tds.simplex(simplex_key).unwrap();
            let ridge_vertices = ridge_vertices_from_simplex(
                simplex,
                usize::from(resolved.omit_a()),
                usize::from(resolved.omit_b()),
            );
            facet_key_from_vertices(&ridge_vertices)
        };
        assert_eq!(resolved_key, stable_key);
    }

    #[test]
    fn test_k2_flip_rewires_external_neighbors_across_cavity_boundary() {
        init_tracing();
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_left_bottom = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_right_bottom = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([2.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_left_top = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 2.0]).unwrap(),
            )
            .unwrap();
        let v_right_top = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([2.0, 2.0]).unwrap(),
            )
            .unwrap();
        let v_external = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([-1.0, 1.0]).unwrap(),
            )
            .unwrap();

        // Flip cavity: two triangles sharing the bottom edge.
        let simplex_cavity_left = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_left_bottom, v_right_bottom, v_left_top], None)
                    .unwrap(),
            )
            .unwrap();
        let simplex_cavity_right = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_right_bottom, v_left_bottom, v_right_top], None)
                    .unwrap(),
            )
            .unwrap();

        // External simplex glued along the left edge of the cavity.
        let simplex_external_left = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_left_bottom, v_left_top, v_external], None)
                    .unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();
        assert!(tds.is_valid().is_ok());

        let facet_idx_flip_edge =
            facet_index_for_edge_2d(&tds, simplex_cavity_left, v_left_bottom, v_right_bottom);
        let ctx = build_k2_flip_context(
            &tds,
            FacetHandle::from_validated(simplex_cavity_left, facet_idx_flip_edge),
        )
        .unwrap();

        let info = apply_bistellar_flip(&mut tds, &ctx).unwrap();

        assert!(!tds.contains_simplex(simplex_cavity_left));
        assert!(!tds.contains_simplex(simplex_cavity_right));
        assert!(tds.contains_simplex(simplex_external_left));

        // External simplex must be rewired from the removed simplex to a newly inserted simplex.
        let facet_idx_glue_edge =
            facet_index_for_edge_2d(&tds, simplex_external_left, v_left_bottom, v_left_top);
        let external_simplex = tds.simplex(simplex_external_left).unwrap();
        let neighbor_key_glue = external_simplex
            .neighbor_key(usize::from(facet_idx_glue_edge))
            .expect("external neighbors should exist")
            .expect("external simplex should have a neighbor across the glue edge after the flip");

        assert!(tds.contains_simplex(neighbor_key_glue));
        assert!(
            info.new_simplices
                .iter()
                .copied()
                .any(|k| k == neighbor_key_glue),
            "expected external neighbor across glue edge to be one of the flip-inserted simplices"
        );

        // Neighbor relation must be symmetric.
        let neighbor_simplex = tds.simplex(neighbor_key_glue).unwrap();
        let mirror_idx = external_simplex
            .mirror_facet_index(usize::from(facet_idx_glue_edge), neighbor_simplex)
            .expect("mirror facet index should exist");
        let neighbor_back = neighbor_simplex.neighbor_key(mirror_idx).flatten();
        assert_eq!(neighbor_back, Some(simplex_external_left));

        // Ensure flip did not leave any dangling neighbor pointers in the newly inserted simplices.
        for &simplex_key in &info.new_simplices {
            let simplex = tds.simplex(simplex_key).unwrap();
            if let Some(ns) = simplex.neighbors() {
                for neighbor_key in ns.flatten() {
                    assert!(
                        tds.contains_simplex(neighbor_key),
                        "dangling neighbor pointer from {simplex_key:?} to {neighbor_key:?}"
                    );
                }
            }
        }

        assert!(tds.is_valid().is_ok());
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "regression test keeps the periodic flip fixture explicit"
    )]
    fn test_k2_flip_preserves_periodic_external_offsets() {
        init_tracing();
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_left_bottom = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_right_bottom = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([2.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_left_top = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 2.0]).unwrap(),
            )
            .unwrap();
        let v_right_top = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([2.0, 2.0]).unwrap(),
            )
            .unwrap();
        let v_external = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([-1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let offset_left_bottom = [0_i8, 0_i8];
        let offset_right_bottom = [1_i8, 0_i8];
        let offset_left_top = [0_i8, 1_i8];
        let offset_right_top = [1_i8, 1_i8];
        let offset_external = [0_i8, -1_i8];
        let simplex_cavity_left = insert_periodic_simplex_with_offsets(
            &mut tds,
            vec![v_left_bottom, v_right_bottom, v_left_top],
            vec![offset_left_bottom, offset_right_bottom, offset_left_top],
        );
        let simplex_cavity_right = insert_periodic_simplex_with_offsets(
            &mut tds,
            vec![v_right_bottom, v_left_bottom, v_right_top],
            vec![offset_right_bottom, offset_left_bottom, offset_right_top],
        );
        let simplex_external_left = insert_periodic_simplex_with_offsets(
            &mut tds,
            vec![v_left_bottom, v_left_top, v_external],
            vec![offset_left_bottom, offset_left_top, offset_external],
        );

        repair_neighbor_pointers(&mut tds).unwrap();
        assert!(tds.is_valid().is_ok());

        let facet_idx_flip_edge =
            facet_index_for_edge_2d(&tds, simplex_cavity_left, v_left_bottom, v_right_bottom);
        let ctx = build_k2_flip_context(
            &tds,
            FacetHandle::from_validated(simplex_cavity_left, facet_idx_flip_edge),
        )
        .unwrap();

        let info = apply_bistellar_flip_with_k(
            &mut tds,
            2,
            &ctx.removed_face_vertices,
            &ctx.inserted_face_vertices,
            &ctx.removed_simplices,
            ctx.direction,
            ReplacementOrientationPolicy::AllowSigned,
            FlipValidationScope::LocalCavity,
        )
        .unwrap()
        .info;

        assert!(!tds.contains_simplex(simplex_cavity_left));
        assert!(!tds.contains_simplex(simplex_cavity_right));
        assert!(tds.contains_simplex(simplex_external_left));
        let expected_left_replacement = [
            (v_left_bottom, offset_left_bottom),
            (v_left_top, offset_left_top),
            (v_right_top, offset_right_top),
        ];
        let expected_right_replacement = [
            (v_right_bottom, offset_right_bottom),
            (v_left_top, offset_left_top),
            (v_right_top, offset_right_top),
        ];
        for &simplex_key in &info.new_simplices {
            let simplex = tds.simplex(simplex_key).unwrap();
            let expected = if simplex.contains_vertex(v_left_bottom) {
                &expected_left_replacement
            } else {
                &expected_right_replacement
            };
            assert_simplex_offsets_by_vertex(&tds, simplex_key, expected);
        }

        let facet_idx_glue_edge =
            facet_index_for_edge_2d(&tds, simplex_external_left, v_left_bottom, v_left_top);
        let external_simplex = tds.simplex(simplex_external_left).unwrap();
        let neighbor_key_glue = external_simplex
            .neighbor_key(usize::from(facet_idx_glue_edge))
            .expect("external neighbors should exist")
            .expect("external simplex should have a replacement neighbor across the glue edge");
        assert!(
            info.new_simplices
                .iter()
                .copied()
                .any(|simplex_key| simplex_key == neighbor_key_glue),
            "expected periodic external facet to be wired to a flip replacement simplex"
        );
        assert_simplex_offsets_by_vertex(
            &tds,
            simplex_external_left,
            &[
                (v_left_bottom, offset_left_bottom),
                (v_left_top, offset_left_top),
                (v_external, offset_external),
            ],
        );
        assert_simplex_offsets_by_vertex(&tds, neighbor_key_glue, &expected_left_replacement);
        assert!(tds.is_valid().is_ok());
    }

    macro_rules! gen_replacement_orientation_helper_tests {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<test_orient_replacement_simplices_uses_periodic_external_simplex_ $dim d>]() {
                    let mut tds: Tds<(), (), $dim> = Tds::empty();
                    let simplex_vertices = insert_standard_simplex_vertices(&mut tds);

                    let offsets = periodic_test_offsets::<$dim>($dim + 1);
                    let mut external_simplex = Simplex::try_new_with_data(simplex_vertices.clone(), None).unwrap();
                    external_simplex.set_periodic_vertex_offsets(offsets.clone()).unwrap();
                    let external_simplex_key = tds.insert_simplex_with_mapping(external_simplex).unwrap();

                    let mut replacement_simplices = vec![vertex_key_buffer(&simplex_vertices)];
                    let mut replacement_offsets: Vec<Option<PeriodicOffsetBuffer<$dim>>> =
                        vec![Some(offsets.clone().into())];
                    orient_replacement_simplices(
                        &tds,
                        &mut replacement_simplices,
                        &mut replacement_offsets,
                        &[FacetHandle::from_validated(external_simplex_key, 0)],
                    )
                    .unwrap();

                    let mut expected_vertices = simplex_vertices.clone();
                    expected_vertices.swap(0, 1);
                    assert_eq!(
                        replacement_simplices[0].iter().copied().collect::<Vec<_>>(),
                        expected_vertices,
                        "periodic external facet parity should flip a same-order replacement simplex"
                    );
                    let mut expected_offsets = offsets;
                    expected_offsets.swap(0, 1);
                    assert_eq!(
                        replacement_offsets[0].as_deref(),
                        Some(expected_offsets.as_slice()),
                        "periodic offsets should stay aligned with swapped replacement vertices"
                    );
                }

                #[test]
                fn [<test_orient_replacement_simplices_rejects_conflicting_periodic_external_offsets_ $dim d>]() {
                    let mut tds: Tds<(), (), $dim> = Tds::empty();
                    let simplex_vertices = insert_standard_simplex_vertices(&mut tds);

                    let external_offsets = vec![[0_i8; $dim]; $dim + 1];
                    let mut external_simplex = Simplex::try_new_with_data(simplex_vertices.clone(), None).unwrap();
                    external_simplex
                        .set_periodic_vertex_offsets(external_offsets)
                        .unwrap();
                    let external_simplex_key = tds.insert_simplex_with_mapping(external_simplex).unwrap();

                    let mut replacement_simplices = vec![vertex_key_buffer(&simplex_vertices)];
                    let mut replacement_offsets = vec![[0_i8; $dim]; $dim + 1];
                    replacement_offsets[1][0] = 1;
                    let mut replacement_offsets: Vec<Option<PeriodicOffsetBuffer<$dim>>> =
                        vec![Some(replacement_offsets.into())];
                    let result = orient_replacement_simplices(
                        &tds,
                        &mut replacement_simplices,
                        &mut replacement_offsets,
                        &[FacetHandle::from_validated(external_simplex_key, 0)],
                    );

                    assert_matches!(
                        &result,
                        Err(FlipError::InvalidFlipContext { reason })
                            if matches!(
                                reason.as_ref(),
                                FlipContextError::ConflictingReplacementPeriodicFrameTranslation {
                                    source_simplex_key,
                                    target_simplex_index: 0,
                                    ..
                                } if *source_simplex_key == external_simplex_key
                            ),
                        "conflicting periodic external facet translations should fail before mutation: {result:?}"
                    );
                }

                #[test]
                fn [<test_orient_replacement_simplices_rejects_periodic_offset_count_mismatch_ $dim d>]() {
                    let mut tds: Tds<(), (), $dim> = Tds::empty();
                    let simplex_vertices = insert_standard_simplex_vertices(&mut tds);
                    let mut replacement_simplices = vec![vertex_key_buffer(&simplex_vertices)];
                    let mut replacement_offsets: Vec<Option<PeriodicOffsetBuffer<$dim>>> = Vec::new();

                    let result = orient_replacement_simplices(
                        &tds,
                        &mut replacement_simplices,
                        &mut replacement_offsets,
                        &[],
                    );

                    assert_matches!(
                        &result,
                        Err(FlipError::InvalidFlipContext { reason })
                            if matches!(
                                reason.as_ref(),
                                FlipContextError::ReplacementPeriodicOffsetCountMismatch {
                                    simplex_count: 1,
                                    offset_count: 0,
                                }
                            ),
                        "replacement offset sidecar length mismatch should fail explicitly: {result:?}"
                    );
                }

                #[test]
                fn [<test_orient_replacement_simplices_rejects_missing_replacement_periodic_offsets_ $dim d>]() {
                    let mut tds: Tds<(), (), $dim> = Tds::empty();
                    let simplex_vertices = insert_standard_simplex_vertices(&mut tds);

                    let mut external_simplex = Simplex::try_new_with_data(simplex_vertices.clone(), None).unwrap();
                    external_simplex
                        .set_periodic_vertex_offsets(vec![[0_i8; $dim]; $dim + 1])
                        .unwrap();
                    let external_simplex_key = tds.insert_simplex_with_mapping(external_simplex).unwrap();

                    let mut replacement_simplices = vec![vertex_key_buffer(&simplex_vertices)];
                    let mut replacement_offsets: Vec<Option<PeriodicOffsetBuffer<$dim>>> = vec![None];
                    let result = orient_replacement_simplices(
                        &tds,
                        &mut replacement_simplices,
                        &mut replacement_offsets,
                        &[FacetHandle::from_validated(external_simplex_key, 0)],
                    );

                    assert_matches!(
                        &result,
                        Err(FlipError::InvalidFlipContext { reason })
                            if matches!(
                                reason.as_ref(),
                                FlipContextError::MissingReplacementPeriodicOffsets {
                                    simplex_index: 0,
                                }
                            ),
                        "periodic external parity should require replacement offsets: {result:?}"
                    );
                }

                #[test]
                fn [<test_orient_replacement_simplices_rejects_replacement_periodic_offset_length_mismatch_ $dim d>]() {
                    let mut tds: Tds<(), (), $dim> = Tds::empty();
                    let simplex_vertices = insert_standard_simplex_vertices(&mut tds);

                    let mut external_simplex = Simplex::try_new_with_data(simplex_vertices.clone(), None).unwrap();
                    external_simplex
                        .set_periodic_vertex_offsets(vec![[0_i8; $dim]; $dim + 1])
                        .unwrap();
                    let external_simplex_key = tds.insert_simplex_with_mapping(external_simplex).unwrap();

                    let mut replacement_simplices = vec![vertex_key_buffer(&simplex_vertices)];
                    let replacement_offsets = vec![[0_i8; $dim]; $dim];
                    let mut replacement_offsets: Vec<Option<PeriodicOffsetBuffer<$dim>>> =
                        vec![Some(replacement_offsets.into())];
                    let result = orient_replacement_simplices(
                        &tds,
                        &mut replacement_simplices,
                        &mut replacement_offsets,
                        &[FacetHandle::from_validated(external_simplex_key, 0)],
                    );

                    assert_matches!(
                        &result,
                        Err(FlipError::InvalidFlipContext { reason })
                            if matches!(
                                reason.as_ref(),
                                FlipContextError::ReplacementPeriodicOffsetLengthMismatch {
                                    simplex_index: 0,
                                    offset_count: $dim,
                                    vertex_count,
                                } if *vertex_count == $dim + 1
                            ),
                        "replacement periodic offsets should stay slot-aligned with vertices: {result:?}"
                    );
                }

                #[test]
                fn [<test_orient_replacement_simplices_rejects_missing_external_simplex_ $dim d>]() {
                    let mut tds: Tds<(), (), $dim> = Tds::empty();
                    let simplex_vertices = insert_standard_simplex_vertices(&mut tds);
                    let external_simplex_key = tds
                        .insert_simplex_with_mapping(Simplex::try_new_with_data(simplex_vertices.clone(), None).unwrap())
                        .unwrap();
                    assert_eq!(
                        tds.remove_simplices_by_keys(&[external_simplex_key])
                            .unwrap(),
                        1
                    );

                    let mut replacement_simplices = vec![vertex_key_buffer(&simplex_vertices)];
                    let mut replacement_offsets: Vec<Option<PeriodicOffsetBuffer<$dim>>> =
                        vec![None; replacement_simplices.len()];
                    let result = orient_replacement_simplices(
                        &tds,
                        &mut replacement_simplices,
                        &mut replacement_offsets,
                        &[FacetHandle::from_validated(external_simplex_key, 0)],
                    );

                    assert_matches!(
                        &result,
                        Err(FlipError::MissingSimplex { simplex_key }) if *simplex_key == external_simplex_key,
                        "missing external simplex should fail explicitly: {result:?}"
                    );
                }

                #[test]
                fn [<test_replacement_orientation_helpers_cover_error_paths_ $dim d>]() {
                    let mut tds: Tds<(), (), $dim> = Tds::empty();
                    let simplex_vertices = insert_standard_simplex_vertices(&mut tds);
                    let v_square = tds.insert_vertex_with_mapping(crate::core::vertex::Vertex::<(), _>::try_new([1.0; $dim]).unwrap()).unwrap();
                    let v_collinear = tds
                        .insert_vertex_with_mapping(crate::core::vertex::Vertex::<(), _>::try_new(scaled_unit_vector::<$dim>(0, 2.0)).unwrap())
                        .unwrap();

                    let source = vertex_key_buffer(&simplex_vertices);
                    let mut target_vertices = simplex_vertices[1..].to_vec();
                    target_vertices.push(v_square);
                    let target = vertex_key_buffer(&target_vertices);

                    let mut neighbor_vertices = simplex_vertices[..$dim].to_vec();
                    neighbor_vertices.push(v_square);
                    let neighbor = vertex_key_buffer(&neighbor_vertices);
                    let short = vertex_key_buffer(&simplex_vertices[..$dim]);
                    let mut two_unique_vertices = simplex_vertices.clone();
                    two_unique_vertices[1] = v_square;
                    two_unique_vertices[2] = v_collinear;
                    let two_unique = vertex_key_buffer(&two_unique_vertices);

                    let order = facet_order(&source, 1).unwrap();
                    let expected_order = simplex_vertices
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, &vertex)| (idx != 1).then_some(vertex))
                        .collect::<Vec<_>>();
                    assert_eq!(order.iter().copied().collect::<Vec<_>>(), expected_order);
                    assert_matches!(
                        facet_order(&source, source.len()),
                        Err(FlipError::InvalidFlipContext { reason })
                            if matches!(
                                reason.as_ref(),
                                FlipContextError::ReplacementFacetIndexOutOfRange {
                                    facet_index,
                                    vertex_count,
                                } if *facet_index == source.len() && *vertex_count == source.len()
                            ),
                        "out-of-range facet indices should be rejected"
                    );

                    assert_eq!(matching_facet_index(&source, 0, &target).unwrap(), Some($dim));
                    assert_eq!(matching_facet_index(&source, 0, &short).unwrap(), None);
                    assert_eq!(matching_facet_index(&source, 0, &two_unique).unwrap(), None);

                    assert_eq!(shared_facet_indices(&source, &neighbor), Some(($dim, $dim)));
                    assert_eq!(shared_facet_indices(&source, &short), None);
                    assert_eq!(shared_facet_indices(&source, &two_unique), None);

                    assert!(!facet_orders_coherent(&source, $dim, &neighbor, $dim).unwrap());
                    assert_matches!(
                        facet_orders_coherent(&source, source.len(), &neighbor, $dim),
                        Err(FlipError::InvalidFlipContext { .. }),
                        "invalid facet-order constraints should surface as invalid context"
                    );

                    let mut odd_target_vertices = simplex_vertices.clone();
                    odd_target_vertices.swap(1, 2);
                    let odd_target = vertex_key_buffer(&odd_target_vertices);
                    assert_eq!(permutation_odd(&source, &odd_target), Some(true));
                    assert_eq!(permutation_odd(&source, &short), None);
                    assert_eq!(permutation_odd(&source, &neighbor), None);
                }

                #[test]
                fn [<test_set_flip_assignment_rejects_conflicts_and_invalid_indices_ $dim d>]() {
                    let mut assignments: SmallBuffer<Option<bool>, MAX_PRACTICAL_DIMENSION_SIZE> =
                        SmallBuffer::from_elem(None, 1);

                    assert!(set_flip_assignment(&mut assignments, 0, true).unwrap());
                    assert_eq!(assignments[0], Some(true));
                    assert!(!set_flip_assignment(&mut assignments, 0, true).unwrap());
                    assert_matches!(
                        set_flip_assignment(&mut assignments, 0, false),
                        Err(FlipError::InvalidFlipContext { reason })
                            if matches!(
                                reason.as_ref(),
                                FlipContextError::ConflictingReplacementOrientationForSimplex {
                                    simplex_index: 0,
                                }
                            ),
                        "conflicting parity assignments should fail"
                    );
                    assert_matches!(
                        set_flip_assignment(&mut assignments, 1, false),
                        Err(FlipError::InvalidFlipContext { reason })
                            if matches!(
                                reason.as_ref(),
                                FlipContextError::ReplacementOrientationIndexOutOfRange {
                                    simplex_index: 1,
                                }
                            ),
                        "out-of-range parity assignments should fail"
                    );
                }

                #[test]
                fn [<test_orient_replacement_simplices_aligns_external_and_internal_facets_ $dim d>]() {
                    let mut tds: Tds<(), (), $dim> = Tds::empty();
                    let simplex_vertices = insert_standard_simplex_vertices(&mut tds);
                    let v_square = tds.insert_vertex_with_mapping(crate::core::vertex::Vertex::<(), _>::try_new([1.0; $dim]).unwrap()).unwrap();
                    let external_simplex_key = tds
                        .insert_simplex_with_mapping(Simplex::try_new_with_data(simplex_vertices.clone(), None).unwrap())
                        .unwrap();

                    let mut external_aligned = vec![vertex_key_buffer(&simplex_vertices)];
                    let mut external_offsets: Vec<Option<PeriodicOffsetBuffer<$dim>>> =
                        vec![None; external_aligned.len()];
                    orient_replacement_simplices(
                        &tds,
                        &mut external_aligned,
                        &mut external_offsets,
                        &[FacetHandle::from_validated(external_simplex_key, 0)],
                    )
                    .unwrap();
                    let mut expected_external = simplex_vertices.clone();
                    expected_external.swap(0, 1);
                    assert_eq!(
                        external_aligned[0].iter().copied().collect::<Vec<_>>(),
                        expected_external,
                        "external facet parity should flip a same-order replacement simplex"
                    );

                    let mut adjacent_vertices = simplex_vertices[..$dim].to_vec();
                    adjacent_vertices.push(v_square);
                    let mut internally_aligned = vec![
                        vertex_key_buffer(&simplex_vertices),
                        vertex_key_buffer(&adjacent_vertices),
                    ];
                    let mut internal_offsets: Vec<Option<PeriodicOffsetBuffer<$dim>>> =
                        vec![None; internally_aligned.len()];
                    orient_replacement_simplices(
                        &tds,
                        &mut internally_aligned,
                        &mut internal_offsets,
                        &[],
                    )
                    .unwrap();
                    let (source_facet_idx, target_facet_idx) =
                        shared_facet_indices(&internally_aligned[0], &internally_aligned[1]).unwrap();
                    assert!(
                        facet_orders_coherent(
                            &internally_aligned[0],
                            source_facet_idx,
                            &internally_aligned[1],
                            target_facet_idx,
                        )
                        .unwrap(),
                        "internal shared facets should be coherent after parity propagation"
                    );
                }

                #[test]
                fn [<test_validate_replacement_orientation_rejects_bad_geometry_ $dim d>]() {
                    let mut tds: Tds<(), (), $dim> = Tds::empty();
                    let simplex_vertices = insert_standard_simplex_vertices(&mut tds);
                    let v_collinear = tds
                        .insert_vertex_with_mapping(crate::core::vertex::Vertex::<(), _>::try_new(scaled_unit_vector::<$dim>(0, 2.0)).unwrap())
                        .unwrap();

                    let mut positive_vertices = simplex_vertices.clone();
                    if $dim % 2 == 1 {
                        positive_vertices.swap(1, 2);
                    }
                    let positive = vertex_key_buffer(&positive_vertices);
                    let positive_simplices = vec![positive];
                    assert!(validate_replacement_orientation(&tds, &positive_simplices).is_ok());

                    let mut negative_vertices = positive_vertices.clone();
                    negative_vertices.swap(1, 2);
                    let negative = vertex_key_buffer(&negative_vertices);
                    let negative_result = validate_replacement_orientation(&tds, &[negative]);
                    assert_matches!(
                        &negative_result,
                        Err(FlipError::NegativeOrientation { simplex_vertices })
                            if simplex_vertices == &negative_vertices,
                        "negative replacement simplices should fail before mutation: {negative_result:?}"
                    );

                    let mut degenerate_vertices = positive_vertices;
                    degenerate_vertices[$dim] = v_collinear;
                    let degenerate = vertex_key_buffer(&degenerate_vertices);
                    let degenerate_result = validate_replacement_orientation(&tds, &[degenerate]);
                    assert_matches!(
                        &degenerate_result,
                        Err(FlipError::DegenerateSimplex),
                        "degenerate replacement simplices should fail before mutation: {degenerate_result:?}"
                    );
                }
            }
        };
    }

    gen_replacement_orientation_helper_tests!(2);
    gen_replacement_orientation_helper_tests!(3);
    gen_replacement_orientation_helper_tests!(4);
    gen_replacement_orientation_helper_tests!(5);

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "Test constructs an explicit k=3 ridge-flip fixture and checks neighbor rewiring"
    )]
    fn test_k3_flip_rewires_external_neighbors_across_cavity_boundary() {
        init_tracing();
        let mut tds: Tds<(), (), 3> = Tds::empty();

        // NOTE: keep `v_edge_start` off the plane of (v_cycle_0, v_cycle_1, v_cycle_2)
        // so the post-flip inserted tetrahedra are non-degenerate.
        let v_edge_start = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_edge_end = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([2.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();

        let v_cycle_0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 2.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_cycle_1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 2.0]).unwrap(),
            )
            .unwrap();
        let v_cycle_2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 2.0, 2.0]).unwrap(),
            )
            .unwrap();

        let v_external = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([-1.0, 1.0, 1.0]).unwrap(),
            )
            .unwrap();

        // Three tetrahedra around the ridge (edge) (v_edge_start, v_edge_end).
        // This is the configuration removed by a k=3 flip (3→2).
        let simplex_around_edge_0 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(
                    vec![v_edge_start, v_edge_end, v_cycle_0, v_cycle_1],
                    None,
                )
                .unwrap(),
            )
            .unwrap();
        let simplex_around_edge_1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(
                    vec![v_edge_start, v_edge_end, v_cycle_1, v_cycle_2],
                    None,
                )
                .unwrap(),
            )
            .unwrap();
        let simplex_around_edge_2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(
                    vec![v_edge_start, v_edge_end, v_cycle_2, v_cycle_0],
                    None,
                )
                .unwrap(),
            )
            .unwrap();

        // External tetrahedron glued to a boundary face of `simplex_around_edge_0`.
        // This face must be rewired to a newly inserted tetrahedron after the flip.
        let simplex_external = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(
                    vec![v_edge_start, v_cycle_0, v_cycle_1, v_external],
                    None,
                )
                .unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();
        assert!(tds.is_valid().is_ok());

        // In `simplex_around_edge_0`, the ridge is the edge (v_edge_start, v_edge_end).
        // We omitted the two non-ridge vertices by construction (indices 2 and 3).
        let ridge = RidgeHandle::from_validated(simplex_around_edge_0, 2, 3);
        let ctx = build_k3_flip_context(&tds, ridge).unwrap();
        assert_eq!(ctx.removed_simplices.len(), 3);
        assert!(
            ctx.removed_simplices
                .iter()
                .copied()
                .any(|simplex_key| simplex_key == simplex_around_edge_0)
        );
        assert!(
            ctx.removed_simplices
                .iter()
                .copied()
                .any(|simplex_key| simplex_key == simplex_around_edge_1)
        );
        assert!(
            ctx.removed_simplices
                .iter()
                .copied()
                .any(|simplex_key| simplex_key == simplex_around_edge_2)
        );

        let info = apply_bistellar_flip(&mut tds, &ctx).unwrap();

        // Removed simplices should be gone.
        assert!(!tds.contains_simplex(simplex_around_edge_0));
        assert!(!tds.contains_simplex(simplex_around_edge_1));
        assert!(!tds.contains_simplex(simplex_around_edge_2));
        for &removed_simplex in &info.removed_simplices {
            assert!(!tds.contains_simplex(removed_simplex));
        }
        assert!(tds.contains_simplex(simplex_external));

        // The external simplex must now neighbor one of the new simplices across face
        // (v_edge_start, v_cycle_0, v_cycle_1).
        let glue_face_facet_index =
            facet_index_for_face_3d(&tds, simplex_external, v_edge_start, v_cycle_0, v_cycle_1);
        let external_simplex = tds.simplex(simplex_external).unwrap();
        let glued_neighbor = external_simplex
            .neighbor_key(usize::from(glue_face_facet_index))
            .expect("external simplex should have neighbors after repair")
            .expect("external simplex should have a neighbor across the glue face");

        assert!(tds.contains_simplex(glued_neighbor));
        assert!(
            info.new_simplices
                .iter()
                .copied()
                .any(|simplex_key| simplex_key == glued_neighbor),
            "expected glued neighbor to be one of the flip-inserted simplices"
        );

        // Neighbor relation must be symmetric.
        let neighbor_simplex = tds.simplex(glued_neighbor).unwrap();
        let mirror_idx = external_simplex
            .mirror_facet_index(usize::from(glue_face_facet_index), neighbor_simplex)
            .expect("mirror facet index should exist");
        let neighbor_back = neighbor_simplex.neighbor_key(mirror_idx).flatten();
        assert_eq!(neighbor_back, Some(simplex_external));

        // Ensure the newly inserted simplices do not reference removed simplices.
        for &simplex_key in &info.new_simplices {
            let simplex = tds.simplex(simplex_key).unwrap();
            if let Some(ns) = simplex.neighbors() {
                for neighbor_key in ns.flatten() {
                    assert!(
                        tds.contains_simplex(neighbor_key),
                        "dangling neighbor pointer from {simplex_key:?} to {neighbor_key:?}"
                    );
                }
            }
        }

        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_repair_diagnostics_cycle_detection_records_repeats() {
        init_tracing();
        let mut diagnostics = RepairDiagnostics::default();
        diagnostics.record_flip_signature(FlipSignature(10));
        diagnostics.record_flip_signature(FlipSignature(20));
        assert_eq!(diagnostics.cycle_detections, 0);

        diagnostics.record_flip_signature(FlipSignature(10));
        assert_eq!(diagnostics.cycle_detections, 1);
        assert_eq!(diagnostics.cycle_samples, vec![FlipSignature(10)]);

        diagnostics.record_flip_signature(FlipSignature(10));
        assert_eq!(diagnostics.cycle_detections, 2);
        assert_eq!(diagnostics.cycle_samples, vec![FlipSignature(10)]);
    }

    #[test]
    fn test_skip_recording_keeps_first_typed_sample() {
        let mut diagnostics = RepairDiagnostics::default();
        let simplex = SimplexKey::from(KeyData::from_ffi(91));
        let missing_simplex = SimplexKey::from(KeyData::from_ffi(92));
        let v0 = VertexKey::from(KeyData::from_ffi(101));
        let v1 = VertexKey::from(KeyData::from_ffi(102));
        let v2 = VertexKey::from(KeyData::from_ffi(103));
        let edge = EdgeKey::from_validated_endpoints(v0, v1);
        let facet = FacetHandle::from_validated(simplex, 0);
        let ridge = RidgeHandle::from_validated(simplex, 0, 1);
        let triangle = TriangleHandle::try_new(v0, v1, v2).unwrap();

        let first_inserted_sample = InsertedSimplexSkipSample {
            location: RepairSkipLocation::Facet(facet),
            removed_face: [v0, v1].into_iter().collect(),
            inserted_face: std::iter::once(v2).collect(),
        };
        diagnostics.record_inserted_simplex_skip(first_inserted_sample.clone());
        assert_eq!(diagnostics.inserted_simplex_skips, 1);
        assert_eq!(
            diagnostics.inserted_simplex_sample,
            Some(first_inserted_sample.clone())
        );

        diagnostics.record_inserted_simplex_skip(InsertedSimplexSkipSample {
            location: RepairSkipLocation::Edge(edge),
            removed_face: std::iter::once(v1).collect(),
            inserted_face: [v0, v2].into_iter().collect(),
        });
        assert_eq!(diagnostics.inserted_simplex_skips, 2);
        assert_eq!(
            diagnostics.inserted_simplex_sample,
            Some(first_inserted_sample)
        );

        // Same contract for ridge-multiplicity and missing-simplex helpers.
        let first_ridge_sample = RidgeMultiplicitySkipSample {
            ridge,
            multiplicity: 3,
        };
        diagnostics.record_invalid_ridge_multiplicity_skip(first_ridge_sample);
        diagnostics.record_invalid_ridge_multiplicity_skip(RidgeMultiplicitySkipSample {
            ridge: RidgeHandle::from_validated(simplex, 1, 2),
            multiplicity: 4,
        });
        assert_eq!(diagnostics.invalid_ridge_multiplicity_skips, 2);
        assert_eq!(
            diagnostics.invalid_ridge_multiplicity_sample,
            Some(first_ridge_sample)
        );

        let first_missing_sample = MissingSimplexSkipSample {
            location: RepairSkipLocation::Triangle(triangle),
            simplex_key: missing_simplex,
        };
        diagnostics.record_missing_simplex_skip(first_missing_sample);
        diagnostics.record_missing_simplex_skip(MissingSimplexSkipSample {
            location: RepairSkipLocation::Ridge(ridge),
            simplex_key: SimplexKey::from(KeyData::from_ffi(93)),
        });
        assert_eq!(diagnostics.missing_simplex_skips, 2);
        assert_eq!(
            diagnostics.missing_simplex_sample,
            Some(first_missing_sample)
        );
    }

    #[test]
    fn test_repair_skip_samples_keep_legacy_debug_shape() {
        let simplex = SimplexKey::from(KeyData::from_ffi(91));
        let missing_simplex = SimplexKey::from(KeyData::from_ffi(92));
        let v0 = VertexKey::from(KeyData::from_ffi(101));
        let v1 = VertexKey::from(KeyData::from_ffi(102));
        let v2 = VertexKey::from(KeyData::from_ffi(103));
        let facet = FacetHandle::from_validated(simplex, 0);
        let ridge = RidgeHandle::from_validated(simplex, 0, 1);
        let triangle = TriangleHandle::try_new(v0, v1, v2).unwrap();

        let removed_face: VertexKeyList = [v0, v1].into_iter().collect();
        let inserted_face: VertexKeyList = std::iter::once(v2).collect();
        let inserted_sample = InsertedSimplexSkipSample {
            location: RepairSkipLocation::Facet(facet),
            removed_face: removed_face.clone(),
            inserted_face: inserted_face.clone(),
        };
        assert_eq!(
            format!("{:?}", Some(inserted_sample)),
            format!(
                "{:?}",
                Some(format!(
                    "facet={facet:?} removed_face={removed_face:?} inserted_face={inserted_face:?}"
                ))
            )
        );

        let ridge_sample = RidgeMultiplicitySkipSample {
            ridge,
            multiplicity: 3,
        };
        assert_eq!(
            format!("{:?}", Some(ridge_sample)),
            format!("{:?}", Some(format!("ridge={ridge:?} multiplicity=3")))
        );

        let missing_sample = MissingSimplexSkipSample {
            location: RepairSkipLocation::Triangle(triangle),
            simplex_key: missing_simplex,
        };
        assert_eq!(
            format!("{:?}", Some(missing_sample)),
            format!(
                "{:?}",
                Some(format!(
                    "triangle={triangle:?} missing_simplex={missing_simplex:?}"
                ))
            )
        );
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TopologySnapshot {
        vertices: Vec<Uuid>,
        simplex_vertices: Vec<Vec<Uuid>>,
        simplex_neighbors: Vec<Vec<Option<Uuid>>>,
    }

    fn snapshot_topology<const D: usize>(tds: &Tds<(), (), D>) -> TopologySnapshot {
        let mut vertices: Vec<Uuid> = tds.vertices().map(|(_, vertex)| vertex.uuid()).collect();
        vertices.sort();

        let mut simplex_vertices: Vec<Vec<Uuid>> = tds
            .simplices()
            .map(|(_, simplex)| {
                let mut uuids: Vec<Uuid> = simplex
                    .vertices()
                    .iter()
                    .map(|&vkey| tds.vertex(vkey).expect("vertex key missing in TDS").uuid())
                    .collect();
                uuids.sort();
                uuids
            })
            .collect();
        simplex_vertices.sort();

        let simplex_neighbors = snapshot_neighbors(tds);

        TopologySnapshot {
            vertices,
            simplex_vertices,
            simplex_neighbors,
        }
    }

    fn snapshot_neighbors<const D: usize>(tds: &Tds<(), (), D>) -> Vec<Vec<Option<Uuid>>> {
        let mut simplex_neighbors: Vec<Vec<Option<Uuid>>> = tds
            .simplices()
            .map(|(_, simplex)| {
                let mut neighbors: Vec<Option<Uuid>> = simplex
                    .neighbors()
                    .map(|neighbor_keys| {
                        neighbor_keys
                            .map(|neighbor| {
                                neighbor.and_then(|neighbor_key| {
                                    tds.simplex(neighbor_key).map(Simplex::uuid)
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                neighbors.sort();
                neighbors
            })
            .collect();
        simplex_neighbors.sort();
        simplex_neighbors
    }

    fn snapshot_incidence<const D: usize>(tds: &Tds<(), (), D>) -> Vec<(Uuid, Option<Uuid>)> {
        let mut incident_simplices: Vec<(Uuid, Option<Uuid>)> = tds
            .vertices()
            .map(|(_, vertex)| {
                (
                    vertex.uuid(),
                    vertex
                        .incident_simplex()
                        .and_then(|simplex_key| tds.simplex(simplex_key).map(Simplex::uuid)),
                )
            })
            .collect();
        incident_simplices.sort();
        incident_simplices
    }

    fn assert_same_vertex_simplex_topology(actual: &TopologySnapshot, expected: &TopologySnapshot) {
        assert_eq!(actual.vertices, expected.vertices);
        assert_eq!(actual.simplex_vertices, expected.simplex_vertices);
    }

    fn insert_translated_simplex<const D: usize>(
        tds: &mut Tds<(), (), D>,
        offset: f64,
    ) -> (Vec<VertexKey>, SimplexKey) {
        let mut vertices = Vec::with_capacity(D + 1);
        vertices.push(
            tds.insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([offset; D]).unwrap(),
            )
            .unwrap(),
        );

        for axis in 0..D {
            let mut coords = [offset; D];
            coords[axis] += 1.0;
            vertices.push(
                tds.insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(coords).unwrap(),
                )
                .unwrap(),
            );
        }

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vertices.clone(), None).unwrap(),
            )
            .unwrap();
        (vertices, simplex_key)
    }

    fn test_flip_trial_validation_rollback_for_dim<const D: usize>() {
        let mut tds: Tds<(), (), D> = Tds::empty();
        let (_first_vertices, first_simplex) = insert_translated_simplex(&mut tds, 0.0);
        let (_second_vertices, second_simplex) = insert_translated_simplex(&mut tds, 10.0);
        repair_neighbor_pointers(&mut tds).unwrap();
        tds.assign_incident_simplices().unwrap();

        let isolated_vertex = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([20.0; D]).unwrap(),
            )
            .unwrap();
        tds.vertex_mut(isolated_vertex)
            .unwrap()
            .set_incident_simplex(Some(second_simplex));

        let before = snapshot_topology(&tds);
        let before_incidence = snapshot_incidence(&tds);
        let denominator = f64::from(u32::try_from(D + 2).unwrap());
        let new_vertex =
            crate::core::vertex::Vertex::<(), _>::try_new([1.0 / denominator; D]).unwrap();

        let result = apply_bistellar_flip_k1(&mut tds, first_simplex, new_vertex);
        match result {
            Err(FlipError::TdsMutation { reason })
                if matches!(reason.as_ref(), FlipMutationError::TrialValidation { .. }) => {}
            other => panic!("expected FlipMutationError::TrialValidation, got {other:?}"),
        }

        assert_eq!(
            snapshot_topology(&tds),
            before,
            "trial.is_valid() failure must leave the original TDS unchanged"
        );
        assert_eq!(
            snapshot_incidence(&tds),
            before_incidence,
            "trial.is_valid() failure must leave incident_simplex pointers unchanged"
        );
    }

    macro_rules! gen_trial_validation_rollback_tests {
        ($($dim:literal),+ $(,)?) => {
            pastey::paste! {
                $(
                    #[test]
                    fn [<test_flip_trial_validation_rollback_ $dim d>]() {
                        test_flip_trial_validation_rollback_for_dim::<$dim>();
                    }
                )+
            }
        };
    }

    gen_trial_validation_rollback_tests!(2, 3, 4, 5);

    #[test]
    fn test_flip_trial_validation_rejects_unassigned_neighbor_slot() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        tds.assign_neighbors().unwrap();

        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            simplex.ensure_neighbors_buffer_mut()[0] = NeighborSlot::Unassigned;
        }

        let simplex = tds.simplex(simplex_key).unwrap();
        let err =
            validate_flip_trial_simplex_neighbors(&tds, simplex_key, simplex, &[]).unwrap_err();

        assert_matches!(
            err,
            TdsValidationFailure::InvalidNeighbors {
                reason: NeighborValidationError::UnassignedNeighborSlot { facet_index: 0, .. },
            }
        );
    }

    /// Checks that a k=2 flip and its inverse preserve topology in dimension `D`.
    #[expect(
        clippy::too_many_lines,
        reason = "The property fixture keeps k=2 setup, forward flip, inverse flip, and invariant checks together so failing cases shrink with full context."
    )]
    fn prop_bistellar_k2_roundtrip_for_dim<const D: usize>(
        offset: f64,
        scale: f64,
    ) -> Result<(), TestCaseError> {
        init_tracing();
        let mut tds: Tds<(), (), D> = Tds::empty();
        let mut shared_vertices = Vec::with_capacity(D);
        for i in 0..D {
            let vertex = tds
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(
                        translated_scaled_unit_vector::<D>(i, offset, scale),
                    )
                    .unwrap(),
                )
                .map_err(|err| {
                    TestCaseError::fail(format!("shared vertex insertion failed: {err:?}"))
                })?;
            shared_vertices.push(vertex);
        }

        let opposite_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([offset; D]).unwrap(),
            )
            .map_err(|err| TestCaseError::fail(format!("opposite A insertion failed: {err:?}")))?;
        let opposite_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([offset + scale; D]).unwrap(),
            )
            .map_err(|err| TestCaseError::fail(format!("opposite B insertion failed: {err:?}")))?;

        let mut vertices_with_first_opposite = shared_vertices.clone();
        vertices_with_first_opposite.push(opposite_a);
        let simplex_a = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vertices_with_first_opposite, None).map_err(|err| {
                    TestCaseError::fail(format!("simplex A creation failed: {err:?}"))
                })?,
            )
            .map_err(|err| TestCaseError::fail(format!("simplex A insertion failed: {err:?}")))?;

        let mut vertices_with_second_opposite = shared_vertices.clone();
        vertices_with_second_opposite.push(opposite_b);
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vertices_with_second_opposite, None).map_err(|err| {
                TestCaseError::fail(format!("simplex B creation failed: {err:?}"))
            })?,
        )
        .map_err(|err| TestCaseError::fail(format!("simplex B insertion failed: {err:?}")))?;

        repair_neighbor_pointers(&mut tds)
            .map_err(|err| TestCaseError::fail(format!("neighbor repair failed: {err:?}")))?;

        let before = snapshot_topology(&tds);
        let facet = FacetHandle::from_validated(
            simplex_a,
            u8::try_from(D).map_err(|err| {
                TestCaseError::fail(format!("facet index conversion failed: {err:?}"))
            })?,
        );
        let context = build_k2_flip_context(&tds, facet)
            .map_err(|err| TestCaseError::fail(format!("k=2 context build failed: {err:?}")))?;
        let info = apply_bistellar_flip_k2(&mut tds, &context)
            .map_err(|err| TestCaseError::fail(format!("k=2 flip failed: {err:?}")))?;
        tds.is_valid()
            .map_err(|err| TestCaseError::fail(format!("post k=2 TDS invalid: {err:?}")))?;

        if D == 2 {
            let mut inverse_facet: Option<FacetHandle> = None;
            for &simplex_key in &info.new_simplices {
                let simplex = tds
                    .simplex(simplex_key)
                    .ok_or_else(|| TestCaseError::fail("new k=2 simplex missing"))?;
                if simplex.contains_vertex(opposite_a) && simplex.contains_vertex(opposite_b) {
                    let facet_index = simplex
                        .vertices()
                        .iter()
                        .position(|&vertex| vertex != opposite_a && vertex != opposite_b)
                        .ok_or_else(|| TestCaseError::fail("missing inverse k=2 facet vertex"))?;
                    inverse_facet = Some(FacetHandle::from_validated(
                        simplex_key,
                        u8::try_from(facet_index).map_err(|err| {
                            TestCaseError::fail(format!(
                                "inverse facet index conversion failed: {err:?}"
                            ))
                        })?,
                    ));
                    break;
                }
            }

            let facet =
                inverse_facet.ok_or_else(|| TestCaseError::fail("inverse k=2 facet not found"))?;
            let context_back = build_k2_flip_context(&tds, facet).map_err(|err| {
                TestCaseError::fail(format!("inverse k=2 context build failed: {err:?}"))
            })?;
            apply_bistellar_flip_k2(&mut tds, &context_back)
                .map_err(|err| TestCaseError::fail(format!("inverse k=2 flip failed: {err:?}")))?;
        } else {
            let inserted = inserted_face_vertices(&info, 2)?;
            let edge = match inserted.as_slice() {
                [a, b] => EdgeKey::from_validated_endpoints(*a, *b),
                _ => {
                    return Err(TestCaseError::fail(
                        "validated k=2 inserted-face arity changed",
                    ));
                }
            };
            let context_back = build_k2_flip_context_from_edge(&tds, edge).map_err(|err| {
                TestCaseError::fail(format!("inverse k=2 context build failed: {err:?}"))
            })?;
            apply_bistellar_flip_dynamic(&mut tds, D, &context_back)
                .map_err(|err| TestCaseError::fail(format!("inverse k=2 flip failed: {err:?}")))?;
        }

        tds.is_valid()
            .map_err(|err| TestCaseError::fail(format!("post inverse k=2 TDS invalid: {err:?}")))?;
        let after = snapshot_topology(&tds);
        prop_assert_eq!(after.vertices, before.vertices);
        prop_assert_eq!(after.simplex_vertices, before.simplex_vertices);
        Ok(())
    }

    /// Checks that a k=3 flip and its inverse preserve topology in dimension `D`.
    #[expect(
        clippy::too_many_lines,
        reason = "The property fixture keeps k=3 setup, forward flip, inverse flip, and invariant checks together so failing cases shrink with full context."
    )]
    fn prop_bistellar_k3_roundtrip_for_dim<const D: usize>(
        offset: f64,
        scale: f64,
    ) -> Result<(), TestCaseError> {
        init_tracing();
        let ridge_vertex_count = D
            .checked_sub(1)
            .ok_or_else(|| TestCaseError::fail("k=3 fixture requires D >= 1"))?;
        let mut tds: Tds<(), (), D> = Tds::empty();
        let mut ridge_vertices = Vec::with_capacity(ridge_vertex_count);
        for i in 0..ridge_vertex_count {
            let vertex = tds
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(
                        translated_scaled_unit_vector::<D>(i, offset, scale),
                    )
                    .unwrap(),
                )
                .map_err(|err| {
                    TestCaseError::fail(format!("ridge vertex insertion failed: {err:?}"))
                })?;
            ridge_vertices.push(vertex);
        }

        let a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([offset; D]).unwrap(),
            )
            .map_err(|err| TestCaseError::fail(format!("opposite A insertion failed: {err:?}")))?;
        let b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new(translated_scaled_unit_vector::<D>(
                    ridge_vertex_count,
                    offset,
                    scale,
                ))
                .unwrap(),
            )
            .map_err(|err| TestCaseError::fail(format!("opposite B insertion failed: {err:?}")))?;
        let c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new(translated_scaled_skewed_point::<D>(
                    offset, scale,
                ))
                .unwrap(),
            )
            .map_err(|err| TestCaseError::fail(format!("opposite C insertion failed: {err:?}")))?;

        let mut first_vertices = ridge_vertices.clone();
        first_vertices.push(a);
        first_vertices.push(b);
        let first_simplex = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(first_vertices, None).map_err(
                |err| TestCaseError::fail(format!("simplex A creation failed: {err:?}")),
            )?)
            .map_err(|err| TestCaseError::fail(format!("simplex A insertion failed: {err:?}")))?;

        let mut second_vertices = ridge_vertices.clone();
        second_vertices.push(b);
        second_vertices.push(c);
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(second_vertices, None).map_err(|err| {
                TestCaseError::fail(format!("simplex B creation failed: {err:?}"))
            })?,
        )
        .map_err(|err| TestCaseError::fail(format!("simplex B insertion failed: {err:?}")))?;

        let mut third_vertices = ridge_vertices.clone();
        third_vertices.push(c);
        third_vertices.push(a);
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(third_vertices, None).map_err(|err| {
                TestCaseError::fail(format!("simplex C creation failed: {err:?}"))
            })?,
        )
        .map_err(|err| TestCaseError::fail(format!("simplex C insertion failed: {err:?}")))?;

        repair_neighbor_pointers(&mut tds)
            .map_err(|err| TestCaseError::fail(format!("neighbor repair failed: {err:?}")))?;

        let before = snapshot_topology(&tds);
        let ridge = RidgeHandle::from_validated(
            first_simplex,
            u8::try_from(ridge_vertex_count).map_err(|err| {
                TestCaseError::fail(format!("ridge index conversion failed: {err:?}"))
            })?,
            u8::try_from(D).map_err(|err| {
                TestCaseError::fail(format!("ridge index conversion failed: {err:?}"))
            })?,
        );
        let context = build_k3_flip_context(&tds, ridge)
            .map_err(|err| TestCaseError::fail(format!("k=3 context build failed: {err:?}")))?;
        let info = apply_bistellar_flip_k3(&mut tds, &context)
            .map_err(|err| TestCaseError::fail(format!("k=3 flip failed: {err:?}")))?;
        tds.is_valid()
            .map_err(|err| TestCaseError::fail(format!("post k=3 TDS invalid: {err:?}")))?;

        if D == 3 {
            let mut inverse_facet: Option<FacetHandle> = None;
            for &simplex_key in &info.new_simplices {
                let simplex = tds
                    .simplex(simplex_key)
                    .ok_or_else(|| TestCaseError::fail("new k=3 simplex missing"))?;
                if simplex.contains_vertex(a)
                    && simplex.contains_vertex(b)
                    && simplex.contains_vertex(c)
                {
                    let facet_index = simplex
                        .vertices()
                        .iter()
                        .position(|&vertex| vertex != a && vertex != b && vertex != c)
                        .ok_or_else(|| TestCaseError::fail("missing inverse k=3 facet vertex"))?;
                    inverse_facet = Some(FacetHandle::from_validated(
                        simplex_key,
                        u8::try_from(facet_index).map_err(|err| {
                            TestCaseError::fail(format!(
                                "inverse facet index conversion failed: {err:?}"
                            ))
                        })?,
                    ));
                    break;
                }
            }

            let facet =
                inverse_facet.ok_or_else(|| TestCaseError::fail("inverse k=3 facet not found"))?;
            let context_back = build_k2_flip_context(&tds, facet).map_err(|err| {
                TestCaseError::fail(format!("inverse k=3 context build failed: {err:?}"))
            })?;
            apply_bistellar_flip_k2(&mut tds, &context_back)
                .map_err(|err| TestCaseError::fail(format!("inverse k=3 flip failed: {err:?}")))?;
        } else {
            let inserted = inserted_face_vertices(&info, 3)?;
            let triangle = match inserted.as_slice() {
                [a, b, c] => TriangleHandle::try_new(*a, *b, *c).map_err(|err| {
                    TestCaseError::fail(format!("invalid inserted triangle: {err}"))
                })?,
                _ => {
                    return Err(TestCaseError::fail(
                        "validated k=3 inserted-face arity changed",
                    ));
                }
            };
            let context_back =
                build_k3_flip_context_from_triangle(&tds, triangle).map_err(|err| {
                    TestCaseError::fail(format!("inverse k=3 context build failed: {err:?}"))
                })?;
            apply_bistellar_flip_dynamic(&mut tds, ridge_vertex_count, &context_back)
                .map_err(|err| TestCaseError::fail(format!("inverse k=3 flip failed: {err:?}")))?;
        }

        tds.is_valid()
            .map_err(|err| TestCaseError::fail(format!("post inverse k=3 TDS invalid: {err:?}")))?;
        let after = snapshot_topology(&tds);
        prop_assert_eq!(after.vertices, before.vertices);
        prop_assert_eq!(after.simplex_vertices, before.simplex_vertices);
        Ok(())
    }

    macro_rules! gen_bistellar_k2_roundtrip_properties {
        ($($dim:literal),+ $(,)?) => {
            pastey::paste! {
                $(
                    proptest! {
                        #![proptest_config(ProptestConfig::with_cases(16))]

                        #[test]
                        fn [<prop_bistellar_k2_roundtrip_ $dim d>](
                            offset in -2.0_f64..2.0,
                            scale in 0.5_f64..2.0,
                        ) {
                            prop_bistellar_k2_roundtrip_for_dim::<$dim>(offset, scale)?;
                        }
                    }
                )+
            }
        };
    }

    /// Exercises the 2D k=2 roundtrip fixture under non-proptest coverage runs.
    #[test]
    fn test_bistellar_k2_roundtrip_smoke_2d() {
        prop_bistellar_k2_roundtrip_for_dim::<2>(0.25, 1.0).unwrap();
    }

    /// Exercises the higher-dimensional k=2 inverse path under non-proptest coverage runs.
    #[test]
    fn test_bistellar_k2_roundtrip_smoke_4d() {
        prop_bistellar_k2_roundtrip_for_dim::<4>(-0.25, 1.25).unwrap();
    }

    macro_rules! gen_bistellar_k3_roundtrip_properties {
        ($($dim:literal),+ $(,)?) => {
            pastey::paste! {
                $(
                    proptest! {
                        #![proptest_config(ProptestConfig::with_cases(16))]

                        #[test]
                        fn [<prop_bistellar_k3_roundtrip_ $dim d>](
                            offset in -2.0_f64..2.0,
                            scale in 0.5_f64..2.0,
                        ) {
                            prop_bistellar_k3_roundtrip_for_dim::<$dim>(offset, scale)?;
                        }
                    }
                )+
            }
        };
    }

    /// Exercises the 3D k=3 roundtrip fixture under non-proptest coverage runs.
    #[test]
    fn test_bistellar_k3_roundtrip_smoke_3d() {
        prop_bistellar_k3_roundtrip_for_dim::<3>(0.25, 1.0).unwrap();
    }

    /// Exercises the higher-dimensional k=3 inverse path under non-proptest coverage runs.
    #[test]
    fn test_bistellar_k3_roundtrip_smoke_4d() {
        prop_bistellar_k3_roundtrip_for_dim::<4>(-0.25, 1.25).unwrap();
    }

    gen_bistellar_k2_roundtrip_properties!(2, 3, 4, 5);
    gen_bistellar_k3_roundtrip_properties!(3, 4, 5);

    macro_rules! test_bistellar_roundtrip_dimension {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<test_bistellar_k1_roundtrip_ $dim d>]() {
                    init_tracing();
                    let mut tds: Tds<(), (), $dim> = Tds::empty();

                    let origin = tds.insert_vertex_with_mapping(crate::core::vertex::Vertex::<(), _>::try_new([0.0; $dim]).unwrap()).unwrap();
                    let mut vertices = Vec::with_capacity($dim + 1);
                    vertices.push(origin);
                    for i in 0..$dim {
                        let v = tds
                            .insert_vertex_with_mapping(crate::core::vertex::Vertex::<(), _>::try_new(unit_vector::<$dim>(i)).unwrap())
                            .unwrap();
                        vertices.push(v);
                    }

                    let simplex_key = tds
                        .insert_simplex_with_mapping(Simplex::try_new_with_data(vertices, None).unwrap())
                        .unwrap();

                    let before = snapshot_topology(&tds);

                    let new_vertex = crate::core::vertex::Vertex::<(), _>::try_new([0.1; $dim]).unwrap();
                    let new_uuid = new_vertex.uuid();
                    let _info = apply_bistellar_flip_k1(&mut tds, simplex_key, new_vertex)
                        .unwrap();
                    assert!(tds.is_valid().is_ok());

                    let new_key = tds.vertex_key_from_uuid(&new_uuid).unwrap();
                    let _info_back =
                        apply_bistellar_flip_k1_inverse(&mut tds, new_key).unwrap();
                    assert!(tds.is_valid().is_ok());

                    assert_eq!(snapshot_topology(&tds), before);
                }

                #[test]
                fn [<test_bistellar_k2_roundtrip_ $dim d>]() {
                    init_tracing();
                    let mut tds: Tds<(), (), $dim> = Tds::empty();
                    let mut shared_vertices = Vec::with_capacity($dim);
                    for i in 0..$dim {
                        let v = tds
                            .insert_vertex_with_mapping(crate::core::vertex::Vertex::<(), _>::try_new(unit_vector::<$dim>(i)).unwrap())
                            .unwrap();
                        shared_vertices.push(v);
                    }

                    let opposite_a = tds
                        .insert_vertex_with_mapping(crate::core::vertex::Vertex::<(), _>::try_new([0.0; $dim]).unwrap())
                        .unwrap();
                    let opposite_b = tds
                        .insert_vertex_with_mapping(crate::core::vertex::Vertex::<(), _>::try_new([1.0; $dim]).unwrap())
                        .unwrap();

                    let mut vertices_with_first_opposite = shared_vertices.clone();
                    vertices_with_first_opposite.push(opposite_a);
                    let simplex_a = tds
                        .insert_simplex_with_mapping(
                            Simplex::try_new_with_data(vertices_with_first_opposite, None).unwrap(),
                        )
                        .unwrap();

                    let mut vertices_with_second_opposite = shared_vertices.clone();
                    vertices_with_second_opposite.push(opposite_b);
                    let _simplex_b = tds
                        .insert_simplex_with_mapping(
                            Simplex::try_new_with_data(vertices_with_second_opposite, None).unwrap(),
                        )
                        .unwrap();

                    repair_neighbor_pointers(&mut tds).unwrap();

                    let before = snapshot_topology(&tds);

                    let facet = FacetHandle::from_validated(simplex_a, u8::try_from($dim).unwrap());
                    let context = build_k2_flip_context(&tds, facet).unwrap();
                    let info = apply_bistellar_flip_k2(&mut tds, &context).unwrap();
                    assert!(tds.is_valid().is_ok());

                    if $dim == 2 {
                        let mut inverse_facet: Option<FacetHandle> = None;
                        for &simplex_key in &info.new_simplices {
                            let simplex = tds.simplex(simplex_key).unwrap();
                            if simplex.contains_vertex(opposite_a) && simplex.contains_vertex(opposite_b) {
                                let facet_index = simplex
                                    .vertices()
                                    .iter()
                                    .position(|&v| v != opposite_a && v != opposite_b)
                                    .expect("missing shared vertex for inverse k=2");
                                inverse_facet = Some(FacetHandle::from_validated(
                                    simplex_key,
                                    u8::try_from(facet_index).unwrap(),
                                ));
                                break;
                            }
                        }

                        let facet = inverse_facet.expect("inverse k=2 facet not found");
                        let context_back = build_k2_flip_context(&tds, facet).unwrap();
                        let _info_back =
                            apply_bistellar_flip_k2(&mut tds, &context_back).unwrap();
                    } else {
                        let edge = EdgeKey::from_validated_endpoints(opposite_a, opposite_b);
                        let context_back = build_k2_flip_context_from_edge(&tds, edge).unwrap();
                        let _info_back =
                            apply_bistellar_flip_dynamic(&mut tds, $dim, &context_back)
                                .unwrap();
                    }

                    assert!(tds.is_valid().is_ok());
                    let after = snapshot_topology(&tds);
                    assert_same_vertex_simplex_topology(&after, &before);
                }
            }
        };
        ($dim:literal, k3) => {
            test_bistellar_roundtrip_dimension!($dim);
            pastey::paste! {
                #[test]
                fn [<test_bistellar_k3_roundtrip_ $dim d>]() {
                    init_tracing();
                    let mut tds: Tds<(), (), $dim> = Tds::empty();
                    let mut ridge_vertices = Vec::with_capacity($dim - 1);
                    for i in 0..($dim - 1) {
                        let v = tds
                            .insert_vertex_with_mapping(crate::core::vertex::Vertex::<(), _>::try_new(unit_vector::<$dim>(i)).unwrap())
                            .unwrap();
                        ridge_vertices.push(v);
                    }

                    let a = tds
                        .insert_vertex_with_mapping(crate::core::vertex::Vertex::<(), _>::try_new([0.0; $dim]).unwrap())
                        .unwrap();
                    let b = tds
                        .insert_vertex_with_mapping(crate::core::vertex::Vertex::<(), _>::try_new(unit_vector::<$dim>($dim - 1)).unwrap())
                        .unwrap();
                    let c = tds
                        .insert_vertex_with_mapping(crate::core::vertex::Vertex::<(), _>::try_new(skewed_point::<$dim>()).unwrap())
                        .unwrap();

                    let mut c1_vertices = ridge_vertices.clone();
                    c1_vertices.push(a);
                    c1_vertices.push(b);
                    let c1 = tds
                        .insert_simplex_with_mapping(Simplex::try_new_with_data(c1_vertices, None).unwrap())
                        .unwrap();

                    let mut c2_vertices = ridge_vertices.clone();
                    c2_vertices.push(b);
                    c2_vertices.push(c);
                    let _c2 = tds
                        .insert_simplex_with_mapping(Simplex::try_new_with_data(c2_vertices, None).unwrap())
                        .unwrap();

                    let mut c3_vertices = ridge_vertices.clone();
                    c3_vertices.push(c);
                    c3_vertices.push(a);
                    let _c3 = tds
                        .insert_simplex_with_mapping(Simplex::try_new_with_data(c3_vertices, None).unwrap())
                        .unwrap();

                    repair_neighbor_pointers(&mut tds).unwrap();

                    let before = snapshot_topology(&tds);

                    let ridge = RidgeHandle::from_validated(
                        c1,
                        u8::try_from($dim - 1).unwrap(),
                        u8::try_from($dim).unwrap(),
                    );
                    let context = build_k3_flip_context(&tds, ridge).unwrap();
                    let info = apply_bistellar_flip_k3(&mut tds, &context).unwrap();
                    assert!(tds.is_valid().is_ok());

                    if $dim == 3 {
                        let mut inverse_facet: Option<FacetHandle> = None;
                        for &simplex_key in &info.new_simplices {
                            let simplex = tds.simplex(simplex_key).unwrap();
                            if simplex.contains_vertex(a)
                                && simplex.contains_vertex(b)
                                && simplex.contains_vertex(c)
                            {
                                let facet_index = simplex
                                    .vertices()
                                    .iter()
                                    .position(|&v| v != a && v != b && v != c)
                                    .expect("missing ridge vertex for inverse k=3");
                                inverse_facet = Some(FacetHandle::from_validated(
                                    simplex_key,
                                    u8::try_from(facet_index).unwrap(),
                                ));
                                break;
                            }
                        }

                        let facet = inverse_facet.expect("inverse k=3 facet not found");
                        let context_back = build_k2_flip_context(&tds, facet).unwrap();
                        let _info_back =
                            apply_bistellar_flip_k2(&mut tds, &context_back).unwrap();
                    } else {
                        let triangle = TriangleHandle::try_new(a, b, c).unwrap();
                        let context_back =
                            build_k3_flip_context_from_triangle(&tds, triangle).unwrap();
                        let _info_back = apply_bistellar_flip_dynamic(
                            &mut tds,
                            $dim - 1,
                            &context_back,
                        )
                        .unwrap();
                    }

                    assert!(tds.is_valid().is_ok());
                    let after = snapshot_topology(&tds);
                    assert_same_vertex_simplex_topology(&after, &before);
                }
            }
        };
    }

    test_bistellar_roundtrip_dimension!(2);
    test_bistellar_roundtrip_dimension!(3, k3);
    test_bistellar_roundtrip_dimension!(4, k3);
    test_bistellar_roundtrip_dimension!(5, k3);

    fn synthetic_vertex_key(index: u64) -> VertexKey {
        VertexKey::from(KeyData::from_ffi(index))
    }

    fn synthetic_simplex_key(index: u64) -> SimplexKey {
        SimplexKey::from(KeyData::from_ffi(index))
    }

    #[test]
    fn test_local_postcondition_frontier_deduplicates_seed_and_touched_simplices() {
        let seed_a = synthetic_simplex_key(1);
        let seed_b = synthetic_simplex_key(2);
        let touched_a = synthetic_simplex_key(3);
        let frontier = local_postcondition_frontier(
            &[seed_a, seed_b, seed_a],
            &[seed_b, touched_a, touched_a],
        );

        assert_eq!(frontier.len(), 3);
        assert_eq!(frontier[0], seed_a);
        assert_eq!(frontier[1], seed_b);
        assert_eq!(frontier[2], touched_a);
    }

    #[test]
    fn test_repair_postcondition_required_tracks_mutation_or_applicable_site() {
        let mut stats = DelaunayRepairStats::default();
        let mut diagnostics = RepairDiagnostics::default();

        assert!(!repair_postcondition_required(&stats, &diagnostics));

        diagnostics.record_applicable_repair_site();
        assert!(repair_postcondition_required(&stats, &diagnostics));

        diagnostics = RepairDiagnostics::default();
        stats.flips_performed = 1;
        assert!(repair_postcondition_required(&stats, &diagnostics));
    }

    fn dynamic_flip_rejects_bad_context_for_dimension<const D: usize>() {
        init_tracing();
        let mut tds: Tds<(), (), D> = Tds::empty();
        let vertices = (1..=D + 2)
            .map(|index| {
                synthetic_vertex_key(
                    u64::try_from(index).expect("test vertex key index should fit in u64"),
                )
            })
            .collect::<Vec<_>>();
        let c0 = synthetic_simplex_key(11);
        let c1 = synthetic_simplex_key(12);

        let valid_shape = FlipContextDyn {
            removed_face_vertices: vertices[..D].iter().copied().collect(),
            inserted_face_vertices: vertices[D..D + 2].iter().copied().collect(),
            removed_simplices: [c0, c1].into_iter().collect(),
            direction: FlipDirection::Forward,
        };

        assert_matches!(
            apply_bistellar_flip_dynamic(&mut tds, 0, &valid_shape),
            Err(FlipError::InvalidFlipContext { reason })
                if matches!(
                    reason.as_ref(),
                    FlipContextError::InvalidMoveSize {
                        k_move: 0,
                        dimension,
                    } if *dimension == D
                )
        );
        assert_matches!(
            apply_bistellar_flip_dynamic(&mut tds, D + 2, &valid_shape),
            Err(FlipError::InvalidFlipContext { reason })
                if matches!(
                    reason.as_ref(),
                    FlipContextError::InvalidMoveSize {
                        k_move,
                        dimension,
                    } if *k_move == D + 2 && *dimension == D
                )
        );

        let wrong_removed_face = FlipContextDyn {
            removed_face_vertices: vertices[..D - 1].iter().copied().collect(),
            ..valid_shape.clone()
        };
        assert_matches!(
            apply_bistellar_flip_dynamic(&mut tds, 2, &wrong_removed_face),
            Err(FlipError::InvalidFlipContext { reason })
                if matches!(
                    reason.as_ref(),
                    FlipContextError::WrongRemovedFaceArity {
                        expected,
                        found,
                    } if *expected == D && *found == D - 1
                )
        );

        let wrong_inserted_face = FlipContextDyn {
            inserted_face_vertices: once(vertices[D]).collect(),
            ..valid_shape.clone()
        };
        assert_matches!(
            apply_bistellar_flip_dynamic(&mut tds, 2, &wrong_inserted_face),
            Err(FlipError::InvalidFlipContext { reason })
                if matches!(
                    reason.as_ref(),
                    FlipContextError::WrongInsertedFaceArity {
                        k_move: 2,
                        expected: 2,
                        found: 1,
                    }
                )
        );

        let wrong_removed_simplices = FlipContextDyn {
            removed_simplices: once(c0).collect(),
            ..valid_shape.clone()
        };
        assert_matches!(
            apply_bistellar_flip_dynamic(&mut tds, 2, &wrong_removed_simplices),
            Err(FlipError::InvalidFlipContext { reason })
                if matches!(
                    reason.as_ref(),
                    FlipContextError::WrongRemovedSimplexCount {
                        expected: 2,
                        found: 1,
                    }
                )
        );

        let overlapping_faces = FlipContextDyn {
            inserted_face_vertices: [vertices[D - 1], vertices[D]].into_iter().collect(),
            ..valid_shape
        };
        assert_matches!(
            apply_bistellar_flip_dynamic(&mut tds, 2, &overlapping_faces),
            Err(FlipError::InvalidFlipContext { reason })
                if matches!(reason.as_ref(), FlipContextError::OverlappingFaces)
        );
        assert_eq!(tds.number_of_vertices(), 0);
        assert_eq!(tds.number_of_simplices(), 0);
    }

    macro_rules! gen_dynamic_flip_bad_context_tests {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<dynamic_flip_rejects_bad_context_ $dim d>]() {
                    dynamic_flip_rejects_bad_context_for_dimension::<$dim>();
                }
            }
        };
    }

    gen_dynamic_flip_bad_context_tests!(2);
    gen_dynamic_flip_bad_context_tests!(3);
    gen_dynamic_flip_bad_context_tests!(4);
    gen_dynamic_flip_bad_context_tests!(5);

    #[test]
    fn test_flip_k2_2d_edge_flip() {
        init_tracing();
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.2]).unwrap(),
            )
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();
        let _c2 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, d], None).unwrap())
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let facet = FacetHandle::from_validated(c1, 2); // facet opposite vertex index 2 (edge AB)
        let context = build_k2_flip_context(&tds, facet).unwrap();
        let info = apply_bistellar_flip_k2(&mut tds, &context).unwrap();

        assert_eq!(info.removed_simplices.len(), 2);
        assert_eq!(info.new_simplices.len(), 2);

        // After flip, we should have an edge between c and d in some simplex.
        let mut has_cd = false;
        for (_, simplex) in tds.simplices() {
            let verts = simplex.vertices();
            if verts.contains(&c) && verts.contains(&d) {
                has_cd = true;
            }
        }
        assert!(has_cd, "Expected flipped diagonal between c and d");

        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k2_rejects_duplicate_simplex() {
        init_tracing();
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.2]).unwrap(),
            )
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();
        let _c2 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, d], None).unwrap())
            .unwrap();

        // Pre-existing simplex that the flip would recreate (B,C,D)
        let _existing = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![b, c, d], None).unwrap())
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let facet = FacetHandle::from_validated(c1, 2); // facet opposite vertex index 2 (edge AB)
        let context = build_k2_flip_context(&tds, facet).unwrap();
        let result = apply_bistellar_flip_k2(&mut tds, &context);

        assert_matches!(result, Err(FlipError::DuplicateSimplex));
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k2_rejects_inserting_existing_edge_in_3d() {
        init_tracing();
        let mut tds: Tds<(), (), 3> = Tds::empty();

        // Opposite vertices across the shared face.
        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();

        // Shared face vertices.
        let v_x = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_y = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_z = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 1.0]).unwrap(),
            )
            .unwrap();

        // Extra vertices for an existing tetrahedron containing the edge (v_a, v_b).
        let v_p = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([2.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_q = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([2.0, 1.0, 0.0]).unwrap(),
            )
            .unwrap();

        // Two tetrahedra sharing face (v_x, v_y, v_z): a k=2 flip across that face would insert edge (v_a, v_b).
        let simplex_a = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_x, v_y, v_z], None).unwrap(),
            )
            .unwrap();
        let _simplex_b = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_b, v_x, v_y, v_z], None).unwrap(),
            )
            .unwrap();

        // Existing tetrahedron that already contains edge (v_a, v_b) but does not contain any of
        // the shared-face vertices (v_x, v_y, v_z).
        let _edge_witness = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_p, v_q], None).unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();
        assert!(tds.is_valid().is_ok());

        // Face (v_x, v_y, v_z) is opposite v_a in `simplex_a` (index 0 by construction).
        let facet = FacetHandle::from_validated(simplex_a, 0);
        let ctx = build_k2_flip_context(&tds, facet).unwrap();

        let result = apply_bistellar_flip_k2(&mut tds, &ctx);

        assert_matches!(result, Err(FlipError::InsertedSimplexAlreadyExists { .. }));
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k2_rejects_nonmanifold_internal_facet() {
        init_tracing();
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.2]).unwrap(),
            )
            .unwrap();
        let v_e = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([2.0, 2.0]).unwrap(),
            )
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let _c2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
            )
            .unwrap();

        // Existing simplex containing the would-be inserted diagonal (C,D).
        let _cd_external = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_c, v_d, v_e], None).unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let facet = FacetHandle::from_validated(c1, 2); // facet opposite vertex index 2 (edge AB)
        let context = build_k2_flip_context(&tds, facet).unwrap();
        let result = apply_bistellar_flip_k2(&mut tds, &context);

        assert_matches!(result, Err(FlipError::NonManifoldFacet));
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k2_3d_two_to_three() {
        init_tracing();
        let mut tds: Tds<(), (), 3> = Tds::empty();
        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.2, 0.2, 1.0]).unwrap(),
            )
            .unwrap();
        let v_e = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.3, -0.1, -0.8]).unwrap(),
            )
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c, v_d], None).unwrap(),
            )
            .unwrap();
        let _c2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c, v_e], None).unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let facet = FacetHandle::from_validated(c1, 3); // facet opposite vertex d (ABC)
        let context = build_k2_flip_context(&tds, facet).unwrap();
        let info = apply_bistellar_flip_k2(&mut tds, &context).unwrap();

        assert_eq!(info.new_simplices.len(), 3);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k3_3d_three_to_two() {
        init_tracing();
        let mut tds: Tds<(), (), 3> = Tds::empty();
        let r0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let r1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.2, 0.2, -1.0]).unwrap(),
            )
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, a, b], None).unwrap(),
            )
            .unwrap();
        let _c2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, b, c], None).unwrap(),
            )
            .unwrap();
        let _c3 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, c, a], None).unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let ridge = RidgeHandle::from_validated(c1, 2, 3);
        let context = build_k3_flip_context(&tds, ridge).unwrap();
        let info = apply_bistellar_flip_k3(&mut tds, &context).unwrap();

        assert_eq!(info.kind, BistellarFlipKind::k3(3));
        assert_eq!(info.removed_simplices.len(), 3);
        assert_eq!(info.new_simplices.len(), 2);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k3_4d_three_to_three() {
        init_tracing();
        let mut tds: Tds<(), (), 4> = Tds::empty();
        let r0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let r1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let r2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.2, 0.2, 0.2, 0.2]).unwrap(),
            )
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, a, b], None).unwrap(),
            )
            .unwrap();
        let _c2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, b, c], None).unwrap(),
            )
            .unwrap();
        let _c3 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, c, a], None).unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let ridge = RidgeHandle::from_validated(c1, 3, 4);
        let context = build_k3_flip_context(&tds, ridge).unwrap();
        let info = apply_bistellar_flip_k3(&mut tds, &context).unwrap();

        assert_eq!(info.kind, BistellarFlipKind::k3(4));
        assert_eq!(info.removed_simplices.len(), 3);
        assert_eq!(info.new_simplices.len(), 3);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k3_5d_three_to_four() {
        init_tracing();
        let mut tds: Tds<(), (), 5> = Tds::empty();
        let r0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let r1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let r2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let r3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.2, 0.2, 0.2, 0.2, 0.5]).unwrap(),
            )
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, r3, a, b], None).unwrap(),
            )
            .unwrap();
        let _c2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, r3, b, c], None).unwrap(),
            )
            .unwrap();
        let _c3 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, r3, c, a], None).unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let ridge = RidgeHandle::from_validated(c1, 4, 5);
        let context = build_k3_flip_context(&tds, ridge).unwrap();
        let info = apply_bistellar_flip_k3(&mut tds, &context).unwrap();

        assert_eq!(info.kind, BistellarFlipKind::k3(5));
        assert_eq!(info.removed_simplices.len(), 3);
        assert_eq!(info.new_simplices.len(), 4);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k2_boundary_facet_error_2d() {
        init_tracing();
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let simplex = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();

        let before = snapshot_topology(&tds);
        let facet = FacetHandle::from_validated(simplex, 0);
        let err = build_k2_flip_context(&tds, facet).unwrap_err();
        assert_matches!(err, FlipError::BoundaryFacet { .. });
        assert_eq!(snapshot_topology(&tds), before);
    }

    #[test]
    fn test_flip_k3_invalid_ridge_multiplicity_3d() {
        init_tracing();
        let mut tds: Tds<(), (), 3> = Tds::empty();
        let a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let simplex = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![a, b, c, d], None).unwrap(),
            )
            .unwrap();

        let ridge = RidgeHandle::from_validated(simplex, 0, 1);
        let err = build_k3_flip_context(&tds, ridge).unwrap_err();
        assert_matches!(err, FlipError::InvalidRidgeMultiplicity { found: 1 });
    }

    #[test]
    fn test_flip_k3_reports_dangling_ridge_neighbor_3d() {
        init_tracing();
        let mut tds: Tds<(), (), 3> = Tds::empty();
        let ridge_start = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let ridge_end = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let first_opposite = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let second_opposite = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let dangling_opposite = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0, 1.0]).unwrap(),
            )
            .unwrap();
        let simplex = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(
                    vec![ridge_start, ridge_end, first_opposite, second_opposite],
                    None,
                )
                .unwrap(),
            )
            .unwrap();
        let dangling_neighbor = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(
                    vec![ridge_start, ridge_end, first_opposite, dangling_opposite],
                    None,
                )
                .unwrap(),
            )
            .unwrap();

        assert_eq!(
            tds.remove_simplices_by_keys(&[dangling_neighbor]).unwrap(),
            1
        );
        tds.simplex_mut(simplex)
            .expect("test simplex should exist")
            .set_neighbors_from_keys([Some(dangling_neighbor), None, None, None])
            .unwrap();

        let ridge = RidgeHandle::from_validated(simplex, 0, 1);
        let err = build_k3_flip_context(&tds, ridge).unwrap_err();
        assert_eq!(
            err,
            FlipError::DanglingRidgeNeighbor {
                simplex_key: simplex,
                neighbor_key: dangling_neighbor,
            }
        );
    }

    #[test]
    fn test_flip_k2_inverse_invalid_edge_multiplicity_4d() {
        init_tracing();
        let mut tds: Tds<(), (), 4> = Tds::empty();
        let mut shared_vertices = Vec::with_capacity(4);
        for i in 0..4 {
            let v = tds
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(unit_vector::<4>(i)).unwrap(),
                )
                .unwrap();
            shared_vertices.push(v);
        }

        let opposite_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0; 4]).unwrap(),
            )
            .unwrap();
        let opposite_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0; 4]).unwrap(),
            )
            .unwrap();

        let mut vertices_with_first_opposite = shared_vertices.clone();
        vertices_with_first_opposite.push(opposite_a);
        let _simplex_a = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vertices_with_first_opposite, None).unwrap(),
            )
            .unwrap();

        let mut vertices_with_second_opposite = shared_vertices.clone();
        vertices_with_second_opposite.push(opposite_b);
        let _simplex_b = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vertices_with_second_opposite, None).unwrap(),
            )
            .unwrap();

        let edge = EdgeKey::from_validated_endpoints(opposite_a, opposite_b);
        let err = build_k2_flip_context_from_edge(&tds, edge).unwrap_err();
        assert_matches!(err, FlipError::InvalidEdgeMultiplicity { .. });
    }

    #[test]
    fn test_flip_k3_inverse_invalid_triangle_multiplicity_5d() {
        init_tracing();
        let mut tds: Tds<(), (), 5> = Tds::empty();
        let origin = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0; 5]).unwrap(),
            )
            .unwrap();
        let mut vertices = Vec::with_capacity(6);
        vertices.push(origin);
        for i in 0..5 {
            let v = tds
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(unit_vector::<5>(i)).unwrap(),
                )
                .unwrap();
            vertices.push(v);
        }
        let _simplex = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vertices.clone(), None).unwrap(),
            )
            .unwrap();

        let triangle = TriangleHandle::try_new(vertices[0], vertices[1], vertices[2]).unwrap();
        let err = build_k3_flip_context_from_triangle(&tds, triangle).unwrap_err();
        assert_matches!(
            err,
            FlipError::InvalidTriangleMultiplicity {
                found: 1,
                expected: 4,
            }
        );
    }

    #[test]
    fn test_flip_k1_degenerate_insert_rejected() {
        init_tracing();
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let simplex_key = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();

        let before = snapshot_topology(&tds);
        let err = apply_bistellar_flip_k1(
            &mut tds,
            simplex_key,
            crate::core::vertex::Vertex::<(), _>::try_new([0.5, 0.0]).unwrap(),
        )
        .unwrap_err();

        assert_matches!(err, FlipError::DegenerateSimplex);
        assert_eq!(snapshot_topology(&tds), before);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_dynamic_k2_forward_4d() {
        init_tracing();
        let mut tds: Tds<(), (), 4> = Tds::empty();
        let mut shared_vertices = Vec::with_capacity(4);
        for i in 0..4 {
            let v = tds
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(unit_vector::<4>(i)).unwrap(),
                )
                .unwrap();
            shared_vertices.push(v);
        }

        let opposite_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0; 4]).unwrap(),
            )
            .unwrap();
        let opposite_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0; 4]).unwrap(),
            )
            .unwrap();

        let mut vertices_with_first_opposite = shared_vertices.clone();
        vertices_with_first_opposite.push(opposite_a);
        let simplex_a = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vertices_with_first_opposite, None).unwrap(),
            )
            .unwrap();

        let mut vertices_with_second_opposite = shared_vertices.clone();
        vertices_with_second_opposite.push(opposite_b);
        let _simplex_b = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vertices_with_second_opposite, None).unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let facet = FacetHandle::from_validated(simplex_a, 4);
        let context = build_k2_flip_context(&tds, facet).unwrap();
        let context_dyn = to_dynamic(context);
        let info = apply_bistellar_flip_dynamic(&mut tds, 2, &context_dyn).unwrap();

        assert_eq!(info.kind, BistellarFlipKind::k2(4));
        assert_eq!(info.removed_simplices.len(), 2);
        assert_eq!(info.new_simplices.len(), 4);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_dynamic_k3_forward_5d() {
        init_tracing();
        let mut tds: Tds<(), (), 5> = Tds::empty();
        let r0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let r1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let r2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let r3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.2, 0.2, 0.2, 0.2, 0.5]).unwrap(),
            )
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, r3, a, b], None).unwrap(),
            )
            .unwrap();
        let _c2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, r3, b, c], None).unwrap(),
            )
            .unwrap();
        let _c3 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, r3, c, a], None).unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let ridge = RidgeHandle::from_validated(c1, 4, 5);
        let context = build_k3_flip_context(&tds, ridge).unwrap();
        let context_dyn = to_dynamic(context);
        let info = apply_bistellar_flip_dynamic(&mut tds, 3, &context_dyn).unwrap();

        assert_eq!(info.kind, BistellarFlipKind::k3(5));
        assert_eq!(info.removed_simplices.len(), 3);
        assert_eq!(info.new_simplices.len(), 4);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_k2_roundtrip_randomized_3d() {
        init_tracing();
        let mut rng = StdRng::seed_from_u64(0x51f1_7a2b);

        for _ in 0..10 {
            let mut jitter = |v: [f64; 3]| {
                let mut out = v;
                for coord in &mut out {
                    *coord += rng.random_range(-0.03..0.03);
                }
                out
            };

            let mut tds: Tds<(), (), 3> = Tds::empty();
            let v_a = tds
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(jitter([0.0, 0.0, 0.0])).unwrap(),
                )
                .unwrap();
            let v_b = tds
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(jitter([1.0, 0.0, 0.0])).unwrap(),
                )
                .unwrap();
            let v_c = tds
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(jitter([0.0, 1.0, 0.0])).unwrap(),
                )
                .unwrap();
            let v_d = tds
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(jitter([0.2, 0.2, 1.0])).unwrap(),
                )
                .unwrap();
            let v_e = tds
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(jitter([0.3, -0.1, -0.8]))
                        .unwrap(),
                )
                .unwrap();

            let c1 = tds
                .insert_simplex_with_mapping(
                    Simplex::try_new_with_data(vec![v_a, v_b, v_c, v_d], None).unwrap(),
                )
                .unwrap();
            let _c2 = tds
                .insert_simplex_with_mapping(
                    Simplex::try_new_with_data(vec![v_a, v_b, v_c, v_e], None).unwrap(),
                )
                .unwrap();

            repair_neighbor_pointers(&mut tds).unwrap();

            let before = snapshot_topology(&tds);
            let facet = FacetHandle::from_validated(c1, 3);
            let context = build_k2_flip_context(&tds, facet).unwrap();
            let info = apply_bistellar_flip_k2(&mut tds, &context).unwrap();
            assert!(tds.is_valid().is_ok());

            let edge = EdgeKey::from_validated_endpoints(
                info.inserted_face_vertices[0],
                info.inserted_face_vertices[1],
            );
            let context_back = build_k2_flip_context_from_edge(&tds, edge).unwrap();
            let _info_back = apply_bistellar_flip_dynamic(&mut tds, 3, &context_back).unwrap();

            assert!(tds.is_valid().is_ok());
            let after = snapshot_topology(&tds);
            assert_same_vertex_simplex_topology(&after, &before);
        }
    }

    #[test]
    fn test_repair_delaunay_flips_non_delaunay_edge_2d() {
        init_tracing();
        let kernel = AdaptiveKernel::<f64>::new();
        let a_coords = [0.0, 0.0];
        let b_coords = [1.0, 1.0];
        let c_coords = [1.0, 0.0];
        let d_candidates = [[0.0, 1.2], [0.1, 1.1], [0.2, 0.9], [-0.1, 1.3]];

        let mut tds = None;
        for d_coords in d_candidates {
            let mut candidate: Tds<(), (), 2> = Tds::empty();
            let a = candidate
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(a_coords).unwrap(),
                )
                .unwrap();
            let b = candidate
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(b_coords).unwrap(),
                )
                .unwrap();
            let c = candidate
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(c_coords).unwrap(),
                )
                .unwrap();
            let d = candidate
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(d_coords).unwrap(),
                )
                .unwrap();

            let _c1 = candidate
                .insert_simplex_with_mapping(
                    Simplex::try_new_with_data(vec![a, b, c], None).unwrap(),
                )
                .unwrap();
            let _c2 = candidate
                .insert_simplex_with_mapping(
                    Simplex::try_new_with_data(vec![a, b, d], None).unwrap(),
                )
                .unwrap();

            repair_neighbor_pointers(&mut candidate).unwrap();

            if verify_delaunay_via_flip_predicates(&candidate, &kernel).is_err() {
                tds = Some(candidate);
                break;
            }
        }

        let mut tds = tds.expect("expected a non-Delaunay configuration from candidates");

        let stats = repair_delaunay_with_flips_k2_k3(
            &mut tds,
            &kernel,
            None,
            TopologyGuarantee::PLManifold,
            None,
        )
        .unwrap();

        assert!(stats.flips_performed > 0);
        assert!(verify_delaunay_via_flip_predicates(&tds, &kernel).is_ok());
        assert!(tds.is_valid().is_ok());
    }

    /// Verifies that `max_flips_override: Some(0)` causes immediate `NonConvergent` when
    /// there is at least one Delaunay violation requiring a flip.
    #[test]
    fn test_repair_max_flips_override_caps_repair() {
        init_tracing();
        let kernel = AdaptiveKernel::<f64>::new();
        let d_candidates = [[0.0, 1.2], [0.1, 1.1], [0.2, 0.9], [-0.1, 1.3]];

        let mut tds = None;
        for d_coords in d_candidates {
            let mut candidate: Tds<(), (), 2> = Tds::empty();
            let a = candidate
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
                )
                .unwrap();
            let b = candidate
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
                )
                .unwrap();
            let c = candidate
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
                )
                .unwrap();
            let d = candidate
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(d_coords).unwrap(),
                )
                .unwrap();

            let _c1 = candidate
                .insert_simplex_with_mapping(
                    Simplex::try_new_with_data(vec![a, b, c], None).unwrap(),
                )
                .unwrap();
            let _c2 = candidate
                .insert_simplex_with_mapping(
                    Simplex::try_new_with_data(vec![a, b, d], None).unwrap(),
                )
                .unwrap();

            repair_neighbor_pointers(&mut candidate).unwrap();

            if verify_delaunay_via_flip_predicates(&candidate, &kernel).is_err() {
                tds = Some(candidate);
                break;
            }
        }

        let mut tds = tds.expect("expected a non-Delaunay configuration from candidates");
        let before = snapshot_topology(&tds);

        // With max_flips=0 the repair must fail immediately with zero flips performed
        // and leave the TDS unchanged.
        let result = repair_delaunay_with_flips_k2_k3(
            &mut tds,
            &kernel,
            None,
            TopologyGuarantee::PLManifold,
            Some(0),
        );
        match result {
            Err(DelaunayRepairError::NonConvergent { diagnostics, .. }) => {
                assert_eq!(
                    diagnostics.flips_performed, 0,
                    "max_flips_override=Some(0) should prevent any flips, got: {}",
                    diagnostics.flips_performed
                );
            }
            other => panic!("expected NonConvergent, got: {other:?}"),
        }
        assert_eq!(
            snapshot_topology(&tds),
            before,
            "TDS must remain unchanged when max_flips=0 prevents all flips"
        );
    }

    /// 3D variant of the `max_flips` cap test.
    ///
    /// Exercises `run_next_facet_repair_step` and `run_next_ridge_repair_step` (only
    /// reached for D≥3) to verify the pre-flip budget guard works in the
    /// multi-queue repair loop.
    #[test]
    #[expect(
        clippy::many_single_char_names,
        reason = "vertex names a-e mirror standard simplex labelling in geometry tests"
    )]
    fn test_repair_max_flips_override_caps_repair_3d() {
        init_tracing();
        let kernel = AdaptiveKernel::<f64>::new();

        let mut tds: Tds<(), (), 3> = Tds::empty();
        let a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let e = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.3, 0.3, 0.3]).unwrap(),
            )
            .unwrap();

        let _c1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![a, b, c, d], None).unwrap(),
            )
            .unwrap();
        let _c2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![a, b, c, e], None).unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        // The fixture must be non-Delaunay for this test to be meaningful.
        assert!(
            verify_delaunay_via_flip_predicates(&tds, &kernel).is_err(),
            "3D fixture must be non-Delaunay (e inside circumsphere of {{a,b,c,d}})"
        );

        let before = snapshot_topology(&tds);
        let result = repair_delaunay_with_flips_k2_k3(
            &mut tds,
            &kernel,
            None,
            TopologyGuarantee::PLManifold,
            Some(0),
        );
        match result {
            Err(DelaunayRepairError::NonConvergent { diagnostics, .. }) => {
                assert_eq!(diagnostics.flips_performed, 0);
            }
            other => panic!("expected NonConvergent for 3D, got: {other:?}"),
        }
        assert_eq!(
            snapshot_topology(&tds),
            before,
            "3D TDS must remain unchanged when max_flips=0 prevents all flips"
        );
    }

    #[test]
    fn test_verify_delaunay_via_flip_predicates_reports_non_delaunay_2d() {
        init_tracing();
        let kernel = FastKernel::<f64>::new();
        let a_coords = [0.0, 0.0];
        let b_coords = [1.0, 1.0];
        let c_coords = [1.0, 0.0];
        let d_candidates = [[0.0, 1.2], [0.1, 1.1], [0.2, 0.9], [-0.1, 1.3]];

        let mut tds = None;
        for d_coords in d_candidates {
            let mut candidate: Tds<(), (), 2> = Tds::empty();
            let a = candidate
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(a_coords).unwrap(),
                )
                .unwrap();
            let b = candidate
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(b_coords).unwrap(),
                )
                .unwrap();
            let c = candidate
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(c_coords).unwrap(),
                )
                .unwrap();
            let d = candidate
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(d_coords).unwrap(),
                )
                .unwrap();

            let _c1 = candidate
                .insert_simplex_with_mapping(
                    Simplex::try_new_with_data(vec![a, b, c], None).unwrap(),
                )
                .unwrap();
            let _c2 = candidate
                .insert_simplex_with_mapping(
                    Simplex::try_new_with_data(vec![a, b, d], None).unwrap(),
                )
                .unwrap();

            repair_neighbor_pointers(&mut candidate).unwrap();

            if verify_delaunay_via_flip_predicates(&candidate, &kernel).is_err() {
                tds = Some(candidate);
                break;
            }
        }

        let tds = tds.expect("expected a non-Delaunay configuration from candidates");
        let result = verify_delaunay_via_flip_predicates(&tds, &kernel);

        assert_matches!(result, Err(DelaunayRepairError::PostconditionFailed { .. }));
    }

    #[test]
    fn test_repair_delaunay_with_flips_rejects_unsupported_dimension_1d() {
        init_tracing();
        let mut tds: Tds<(), (), 1> = Tds::empty();
        let kernel = AdaptiveKernel::<f64>::new();

        let result = repair_delaunay_with_flips_k2_k3(
            &mut tds,
            &kernel,
            None,
            TopologyGuarantee::PLManifold,
            None,
        );

        assert_matches!(
            result,
            Err(DelaunayRepairError::Flip { source })
                if matches!(
                    source.as_ref(),
                    FlipError::UnsupportedDimension { dimension: 1 }
                )
        );
    }

    #[test]
    fn test_flip_k2_robust_kernel_near_degenerate_2d() {
        init_tracing();
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1e-9]).unwrap(),
            )
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();
        let _c2 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, d], None).unwrap())
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let facet = FacetHandle::from_validated(c1, 2);
        let context = build_k2_flip_context(&tds, facet).unwrap();
        let _info = apply_bistellar_flip_k2(&mut tds, &context).unwrap();

        assert!(tds.is_valid().is_ok());
    }

    /// Verifies that `k2_flip_would_create_degenerate_simplex` detects a degenerate
    /// replacement simplex (collinear vertices in 2D).
    #[test]
    fn test_k2_flip_would_create_degenerate_simplex_degenerate() {
        init_tracing();
        let mut tds: Tds<(), (), 2> = Tds::empty();
        // a, c, d are collinear on the x-axis → replacement simplex {a,c,d} is degenerate
        let a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.5, 0.0]).unwrap(),
            )
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();
        let _c2 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, d], None).unwrap())
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let facet = FacetHandle::from_validated(c1, 2);
        let context = build_k2_flip_context(&tds, facet).unwrap();

        let degenerate = k2_flip_would_create_degenerate_simplex(&tds, &context).unwrap();
        assert!(
            degenerate,
            "replacement simplices with collinear vertices should be degenerate"
        );
    }

    /// Verifies that `k2_flip_would_create_degenerate_simplex` returns false for
    /// non-degenerate simplices using `robust_orientation` (kernel-independent).
    #[test]
    fn test_k2_flip_would_create_degenerate_simplex_nondegenerate() {
        init_tracing();
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();
        let _c2 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, d], None).unwrap())
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let facet = FacetHandle::from_validated(c1, 2);
        let context = build_k2_flip_context(&tds, facet).unwrap();

        assert_context_has_nonzero_robust_orientation(&tds, &context);

        let degenerate = k2_flip_would_create_degenerate_simplex(&tds, &context).unwrap();
        assert!(!degenerate);
    }

    #[test]
    fn test_flip_k4_4d_four_to_two() {
        init_tracing();
        let mut tds: Tds<(), (), 4> = Tds::empty();
        let mut shared_vertices = Vec::with_capacity(4);
        for i in 0..4 {
            let v = tds
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(unit_vector::<4>(i)).unwrap(),
                )
                .unwrap();
            shared_vertices.push(v);
        }

        let opposite_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0; 4]).unwrap(),
            )
            .unwrap();
        let opposite_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0; 4]).unwrap(),
            )
            .unwrap();

        let mut vertices_with_first_opposite = shared_vertices.clone();
        vertices_with_first_opposite.push(opposite_a);
        let simplex_a = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vertices_with_first_opposite, None).unwrap(),
            )
            .unwrap();

        let mut vertices_with_second_opposite = shared_vertices.clone();
        vertices_with_second_opposite.push(opposite_b);
        let _simplex_b = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vertices_with_second_opposite, None).unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let facet = FacetHandle::from_validated(simplex_a, 4);
        let context = build_k2_flip_context(&tds, facet).unwrap();
        let _info = apply_bistellar_flip_k2(&mut tds, &context).unwrap();

        let edge = EdgeKey::from_validated_endpoints(opposite_a, opposite_b);
        let context_back = build_k2_flip_context_from_edge(&tds, edge).unwrap();
        let info_back = apply_bistellar_flip_dynamic(&mut tds, 4, &context_back).unwrap();

        assert_eq!(info_back.kind.k, 4);
        assert_eq!(info_back.kind.d, 4);
        assert_eq!(info_back.removed_simplices.len(), 4);
        assert_eq!(info_back.new_simplices.len(), 2);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k5_4d_five_to_one() {
        init_tracing();
        let mut tds: Tds<(), (), 4> = Tds::empty();
        let origin = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0; 4]).unwrap(),
            )
            .unwrap();
        let mut vertices = Vec::with_capacity(5);
        vertices.push(origin);
        for i in 0..4 {
            let v = tds
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(unit_vector::<4>(i)).unwrap(),
                )
                .unwrap();
            vertices.push(v);
        }

        let simplex_key = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vertices, None).unwrap())
            .unwrap();

        let new_vertex = crate::core::vertex::Vertex::<(), _>::try_new([0.1; 4]).unwrap();
        let new_uuid = new_vertex.uuid();
        let info = apply_bistellar_flip_k1(&mut tds, simplex_key, new_vertex).unwrap();

        assert_eq!(info.kind.k, 1);
        assert_eq!(info.new_simplices.len(), 5);

        let new_key = tds.vertex_key_from_uuid(&new_uuid).unwrap();
        let info_back = apply_bistellar_flip_k1_inverse(&mut tds, new_key).unwrap();

        assert_eq!(info_back.kind.k, 5);
        assert_eq!(info_back.kind.d, 4);
        assert_eq!(info_back.removed_simplices.len(), 5);
        assert_eq!(info_back.new_simplices.len(), 1);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k4_5d_four_to_three() {
        init_tracing();
        let mut tds: Tds<(), (), 5> = Tds::empty();
        let r0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let r1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let r2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let r3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.2, 0.2, 0.2, 0.2, 0.5]).unwrap(),
            )
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, r3, a, b], None).unwrap(),
            )
            .unwrap();
        let _c2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, r3, b, c], None).unwrap(),
            )
            .unwrap();
        let _c3 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, r3, c, a], None).unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let ridge = RidgeHandle::from_validated(c1, 4, 5);
        let context = build_k3_flip_context(&tds, ridge).unwrap();
        let info = apply_bistellar_flip_k3(&mut tds, &context).unwrap();

        assert_eq!(info.kind.k, 3);
        assert_eq!(info.inserted_face_vertices.len(), 3);

        let triangle = TriangleHandle::try_new(
            info.inserted_face_vertices[0],
            info.inserted_face_vertices[1],
            info.inserted_face_vertices[2],
        )
        .unwrap();
        let context_back = build_k3_flip_context_from_triangle(&tds, triangle).unwrap();
        let info_back = apply_bistellar_flip_dynamic(&mut tds, 4, &context_back).unwrap();

        assert_eq!(info_back.kind.k, 4);
        assert_eq!(info_back.kind.d, 5);
        assert_eq!(info_back.removed_simplices.len(), 4);
        assert_eq!(info_back.new_simplices.len(), 3);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k5_5d_five_to_two() {
        init_tracing();
        let mut tds: Tds<(), (), 5> = Tds::empty();
        let mut shared_vertices = Vec::with_capacity(5);
        for i in 0..5 {
            let v = tds
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(unit_vector::<5>(i)).unwrap(),
                )
                .unwrap();
            shared_vertices.push(v);
        }

        let opposite_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0; 5]).unwrap(),
            )
            .unwrap();
        let opposite_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0; 5]).unwrap(),
            )
            .unwrap();

        let mut vertices_with_first_opposite = shared_vertices.clone();
        vertices_with_first_opposite.push(opposite_a);
        let simplex_a = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vertices_with_first_opposite, None).unwrap(),
            )
            .unwrap();

        let mut vertices_with_second_opposite = shared_vertices.clone();
        vertices_with_second_opposite.push(opposite_b);
        let _simplex_b = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vertices_with_second_opposite, None).unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let facet = FacetHandle::from_validated(simplex_a, 5);
        let context = build_k2_flip_context(&tds, facet).unwrap();
        let _info = apply_bistellar_flip_k2(&mut tds, &context).unwrap();

        let edge = EdgeKey::from_validated_endpoints(opposite_a, opposite_b);
        let context_back = build_k2_flip_context_from_edge(&tds, edge).unwrap();
        let info_back = apply_bistellar_flip_dynamic(&mut tds, 5, &context_back).unwrap();

        assert_eq!(info_back.kind.k, 5);
        assert_eq!(info_back.kind.d, 5);
        assert_eq!(info_back.removed_simplices.len(), 5);
        assert_eq!(info_back.new_simplices.len(), 2);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k6_5d_six_to_one() {
        init_tracing();
        let mut tds: Tds<(), (), 5> = Tds::empty();
        let origin = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0; 5]).unwrap(),
            )
            .unwrap();
        let mut vertices = Vec::with_capacity(6);
        vertices.push(origin);
        for i in 0..5 {
            let v = tds
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(unit_vector::<5>(i)).unwrap(),
                )
                .unwrap();
            vertices.push(v);
        }

        let simplex_key = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vertices, None).unwrap())
            .unwrap();

        let new_vertex = crate::core::vertex::Vertex::<(), _>::try_new([0.1; 5]).unwrap();
        let new_uuid = new_vertex.uuid();
        let info = apply_bistellar_flip_k1(&mut tds, simplex_key, new_vertex).unwrap();

        assert_eq!(info.kind.k, 1);
        assert_eq!(info.new_simplices.len(), 6);

        let new_key = tds.vertex_key_from_uuid(&new_uuid).unwrap();
        let info_back = apply_bistellar_flip_k1_inverse(&mut tds, new_key).unwrap();

        assert_eq!(info_back.kind.k, 6);
        assert_eq!(info_back.kind.d, 5);
        assert_eq!(info_back.removed_simplices.len(), 6);
        assert_eq!(info_back.new_simplices.len(), 1);
        assert!(tds.is_valid().is_ok());
    }
    #[test]
    fn test_flip_k1_2d_roundtrip() {
        init_tracing();
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();

        let new_vertex = crate::core::vertex::Vertex::<(), _>::try_new([0.2, 0.2]).unwrap();
        let new_uuid = new_vertex.uuid();
        let info = apply_bistellar_flip_k1(&mut tds, simplex, new_vertex).unwrap();

        assert_eq!(info.kind.k, 1);
        assert_eq!(info.kind.d, 2);
        assert_eq!(tds.number_of_simplices(), 3);

        let new_key = tds.vertex_key_from_uuid(&new_uuid).unwrap();
        let info_back = apply_bistellar_flip_k1_inverse(&mut tds, new_key).unwrap();

        assert_eq!(info_back.kind.k, 3);
        assert_eq!(info_back.kind.d, 2);
        assert_eq!(tds.number_of_simplices(), 1);
        assert_eq!(tds.number_of_vertices(), 3);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_repair_queue_inverse_k2_smoke_4d() {
        init_tracing();
        let mut tds: Tds<(), (), 4> = Tds::empty();
        let mut shared_vertices = Vec::with_capacity(4);
        for i in 0..4 {
            let v = tds
                .insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(unit_vector::<4>(i)).unwrap(),
                )
                .unwrap();
            shared_vertices.push(v);
        }

        let opposite_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0; 4]).unwrap(),
            )
            .unwrap();
        let opposite_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0; 4]).unwrap(),
            )
            .unwrap();

        let mut vertices_with_first_opposite = shared_vertices.clone();
        vertices_with_first_opposite.push(opposite_a);
        let simplex_a = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vertices_with_first_opposite, None).unwrap(),
            )
            .unwrap();

        let mut vertices_with_second_opposite = shared_vertices.clone();
        vertices_with_second_opposite.push(opposite_b);
        let _simplex_b = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vertices_with_second_opposite, None).unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let facet = FacetHandle::from_validated(simplex_a, 4);
        let context = build_k2_flip_context(&tds, facet).unwrap();
        let info = apply_bistellar_flip_k2(&mut tds, &context).unwrap();

        let kernel = AdaptiveKernel::<f64>::new();
        let seed_simplices: SimplexKeyBuffer = info.new_simplices.iter().copied().collect();
        let stats = repair_delaunay_with_flips_k2_k3(
            &mut tds,
            &kernel,
            Some(seed_simplices.as_slice()),
            TopologyGuarantee::PLManifold,
            None,
        )
        .unwrap();
        assert!(stats.facets_checked > 0);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_repair_queue_inverse_k3_smoke_5d() {
        init_tracing();
        let mut tds: Tds<(), (), 5> = Tds::empty();
        let r0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let r1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let r2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let r3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.2, 0.2, 0.2, 0.2, 0.5]).unwrap(),
            )
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, r3, a, b], None).unwrap(),
            )
            .unwrap();
        let _c2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, r3, b, c], None).unwrap(),
            )
            .unwrap();
        let _c3 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, r3, c, a], None).unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let ridge = RidgeHandle::from_validated(c1, 4, 5);
        let context = build_k3_flip_context(&tds, ridge).unwrap();
        let info = apply_bistellar_flip_k3(&mut tds, &context).unwrap();

        let kernel = AdaptiveKernel::<f64>::new();
        let seed_simplices: SimplexKeyBuffer = info.new_simplices.iter().copied().collect();
        let result = repair_delaunay_with_flips_k2_k3(
            &mut tds,
            &kernel,
            Some(seed_simplices.as_slice()),
            TopologyGuarantee::PLManifold,
            None,
        );

        match result {
            Ok(stats) => assert!(stats.facets_checked > 0),
            Err(DelaunayRepairError::PostconditionFailed { .. }) => {
                // This test constructs a synthetic configuration to smoke-test queue plumbing.
                // Postcondition verification can legitimately fail in degenerate/non-Delaunay
                // setups; what we must preserve is TDS structural validity.
            }
            Err(err) => panic!("unexpected repair failure: {err}"),
        }

        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_coherent_orientation_violation_maps_to_invalid_flip_context() {
        let err: FlipError = FlipContextError::CoherentOrientationViolation {
            stage: FlipOrientationCheckStage::BeforeMutation,
            k_move: 2,
            direction: FlipDirection::Forward,
        }
        .into();

        assert_matches!(
            err,
            FlipError::InvalidFlipContext { reason }
                if matches!(
                    reason.as_ref(),
                    FlipContextError::CoherentOrientationViolation {
                        stage: FlipOrientationCheckStage::BeforeMutation,
                        k_move: 2,
                        direction: FlipDirection::Forward
                    }
                )
        );
    }

    #[test]
    fn test_coherent_orientation_violation_maps_to_tds_mutation() {
        let err: FlipError = FlipMutationError::CoherentOrientationViolation {
            stage: FlipOrientationCheckStage::AfterTrialMutation,
            k_move: 2,
            direction: FlipDirection::Forward,
        }
        .into();

        assert_eq!(
            FlipFailureKind::from(&err),
            FlipFailureKind::TrialValidation
        );
        assert_matches!(
            err,
            FlipError::TdsMutation { reason }
                if matches!(
                    reason.as_ref(),
                    FlipMutationError::CoherentOrientationViolation {
                        stage: FlipOrientationCheckStage::AfterTrialMutation,
                        k_move: 2,
                        direction: FlipDirection::Forward
                    }
                )
        );
    }

    #[test]
    fn test_flip_error_partial_eq() {
        let unsupported_1 = FlipError::UnsupportedDimension { dimension: 1 };
        let unsupported_1_copy = FlipError::UnsupportedDimension { dimension: 1 };
        let unsupported_2 = FlipError::UnsupportedDimension { dimension: 2 };
        assert_eq!(unsupported_1, unsupported_1_copy);
        assert_ne!(unsupported_1, unsupported_2);

        assert_ne!(FlipError::DegenerateSimplex, FlipError::DuplicateSimplex);
        assert_eq!(FlipError::NonManifoldFacet, FlipError::NonManifoldFacet);

        let ridge_4 = FlipError::InvalidRidgeMultiplicity { found: 4 };
        let ridge_4_copy = FlipError::InvalidRidgeMultiplicity { found: 4 };
        let ridge_5 = FlipError::InvalidRidgeMultiplicity { found: 5 };
        assert_eq!(ridge_4, ridge_4_copy);
        assert_ne!(ridge_4, ridge_5);
    }

    fn sample_tds_validation_failure() -> TdsValidationFailure {
        TdsValidationFailure::InconsistentDataStructure {
            message: "synthetic neighbor mismatch".to_string(),
        }
    }

    fn sample_repair_diagnostics() -> DelaunayRepairDiagnostics {
        DelaunayRepairDiagnostics {
            facets_checked: 7,
            flips_performed: 3,
            max_queue_len: 5,
            ambiguous_predicates: 2,
            ambiguous_predicate_samples: vec![11, 13],
            predicate_failures: 1,
            cycle_detections: 4,
            cycle_signature_samples: vec![17, 19],
            attempt: 2,
            queue_order: RepairQueueOrder::Lifo,
        }
    }

    #[test]
    fn test_flip_failure_kind_preserves_nested_validation_and_repair_reasons() {
        let trial_error = FlipError::from(FlipMutationError::TrialValidation {
            k_move: 2,
            direction: FlipDirection::Forward,
            source: sample_tds_validation_failure(),
        });
        assert_eq!(
            FlipFailureKind::from(&trial_error),
            FlipFailureKind::TrialValidation
        );

        let wiring_validation = FlipError::from(FlipNeighborWiringError::TopologyValidation {
            source: sample_tds_validation_failure(),
        });
        assert_eq!(
            FlipFailureKind::from(&wiring_validation),
            FlipFailureKind::WiringValidation
        );

        let repair_reason =
            FlipNeighborRepairFailure::from(DelaunayRepairError::VerificationFailed {
                context: DelaunayRepairVerificationContext::PostRepairVerification,
                source: Box::new(trial_error),
            });
        match &repair_reason {
            FlipNeighborRepairFailure::VerificationFailed {
                context,
                source_kind,
            } => {
                assert_eq!(
                    *context,
                    DelaunayRepairVerificationContext::PostRepairVerification
                );
                assert_eq!(*source_kind, FlipFailureKind::TrialValidation);
            }
            other => panic!("expected verification failure, got {other:?}"),
        }

        let flip_reason =
            FlipNeighborRepairFailure::from(DelaunayRepairError::from(FlipError::DuplicateSimplex));
        assert_eq!(
            flip_reason,
            FlipNeighborRepairFailure::Flip {
                source_kind: FlipFailureKind::DuplicateSimplex,
            }
        );

        let wiring_repair = FlipError::from(FlipNeighborWiringError::DelaunayRepair {
            reason: repair_reason,
        });
        assert_eq!(
            FlipFailureKind::from(&wiring_repair),
            FlipFailureKind::DelaunayRepairFailed
        );

        let dangling_ridge_neighbor = FlipError::DanglingRidgeNeighbor {
            simplex_key: SimplexKey::from(KeyData::from_ffi(1)),
            neighbor_key: SimplexKey::from(KeyData::from_ffi(2)),
        };
        assert_eq!(
            FlipFailureKind::from(&dangling_ridge_neighbor),
            FlipFailureKind::DanglingRidgeNeighbor
        );

        let dangling_vertex_incidence = FlipError::DanglingVertexIncidence {
            vertex_key: VertexKey::from(KeyData::from_ffi(1)),
            simplex_key: SimplexKey::from(KeyData::from_ffi(2)),
        };
        assert_eq!(
            FlipFailureKind::from(&dangling_vertex_incidence),
            FlipFailureKind::DanglingVertexIncidence
        );
    }

    fn assert_hull_extension_failure_kind(
        source: &HullExtensionReason,
        expected: FlipNeighborHullExtensionFailureKind,
        expected_display: &str,
    ) {
        let hull_kind = FlipNeighborHullExtensionFailureKind::from(source);
        assert_eq!(hull_kind, expected);
        assert_eq!(hull_kind.to_string(), expected_display);
    }

    #[test]
    fn test_flip_neighbor_hull_extension_failure_kind_conversions() {
        assert_hull_extension_failure_kind(
            &HullExtensionReason::BoundaryEdgeSplitFacetCount {
                expected: 2,
                actual: 1,
            },
            FlipNeighborHullExtensionFailureKind::BoundaryEdgeSplitFacetCount,
            "boundary edge split facet count",
        );

        assert_hull_extension_failure_kind(
            &HullExtensionReason::MultipleBoundaryEdgeSplitFacets,
            FlipNeighborHullExtensionFailureKind::MultipleBoundaryEdgeSplitFacets,
            "multiple boundary edge split facets",
        );

        assert_hull_extension_failure_kind(
            &HullExtensionReason::DisconnectedVisiblePatch {
                boundary_ridges: 1,
                ridge_fans: 0,
                components: 2,
                boundary_components: 2,
                boundary_subface_nonmanifold: 0,
            },
            FlipNeighborHullExtensionFailureKind::DisconnectedVisiblePatch,
            "disconnected visible patch",
        );
    }

    #[test]
    fn test_flip_neighbor_conversion_kinds_cover_insertion_suberrors() {
        let cavity_kind = FlipNeighborCavityFailureKind::from(
            &CavityFillingError::BoundarySimplexCountMismatch {
                boundary_facet_count: 3,
                new_simplex_count: 2,
            },
        );
        assert_eq!(
            cavity_kind,
            FlipNeighborCavityFailureKind::BoundarySimplexCountMismatch
        );
        assert_eq!(cavity_kind.to_string(), "boundary simplex count mismatch");

        let cavity_kind = FlipNeighborCavityFailureKind::from(
            &CavityFillingError::UnsupportedDegenerateLocation {
                location: LocateResult::Outside,
            },
        );
        assert_eq!(
            cavity_kind,
            FlipNeighborCavityFailureKind::UnsupportedDegenerateLocation
        );
        assert_eq!(cavity_kind.to_string(), "unsupported degenerate location");

        let validation_kind = FlipNeighborDelaunayValidationFailureKind::from(
            &DelaunayTriangulationValidationError::RepairOperationFailed {
                operation: DelaunayRepairOperation::VertexRemoval,
                source: Box::new(DelaunayRepairError::InvalidTopology {
                    required: TopologyGuarantee::PLManifold,
                    found: TopologyGuarantee::Pseudomanifold,
                    message: "repair requires PL topology",
                }),
            },
        );
        assert_eq!(
            validation_kind,
            FlipNeighborDelaunayValidationFailureKind::RepairOperationFailed
        );
        assert_eq!(validation_kind.to_string(), "repair operation failed");

        let repair_wiring = FlipNeighborWiringError::from(InsertionError::DelaunayRepairFailed {
            source: Box::new(DelaunayRepairError::InvalidTopology {
                required: TopologyGuarantee::PLManifold,
                found: TopologyGuarantee::Pseudomanifold,
                message: "repair requires PL topology",
            }),
            context: DelaunayRepairFailureContext::PostInsertionRepair,
        });
        match repair_wiring {
            FlipNeighborWiringError::DelaunayRepair {
                reason:
                    FlipNeighborRepairFailure::InvalidTopology {
                        required,
                        found,
                        message,
                    },
            } => {
                assert_eq!(required, TopologyGuarantee::PLManifold);
                assert_eq!(found, TopologyGuarantee::Pseudomanifold);
                assert_eq!(message, "repair requires PL topology");
            }
            other => panic!("expected preserved Delaunay repair reason, got {other:?}"),
        }

        let budget_wiring =
            FlipNeighborWiringError::from(InsertionError::MaxSimplicesRemovedExceeded {
                max_simplices_removed: 2,
                attempted: 3,
            });
        assert_eq!(
            budget_wiring,
            FlipNeighborWiringError::MaxSimplicesRemovedExceeded {
                max_simplices_removed: 2,
                attempted: 3,
            }
        );

        let spatial_index_wiring =
            FlipNeighborWiringError::from(InsertionError::SpatialIndexConstruction {
                reason: SpatialIndexConstructionFailure::NonPositiveCellSize {
                    value: CoordinateConversionValue::from_f64(0.0),
                },
            });
        assert_eq!(
            spatial_index_wiring,
            FlipNeighborWiringError::SpatialIndexConstruction {
                reason: SpatialIndexConstructionFailure::NonPositiveCellSize {
                    value: CoordinateConversionValue::from_f64(0.0),
                },
            }
        );
    }

    #[test]
    fn test_flip_neighbor_repair_diagnostics_preserve_summary_fields() {
        let diagnostics = sample_repair_diagnostics();
        let summary = FlipNeighborRepairDiagnostics::from(diagnostics.clone());

        assert_eq!(summary.facets_checked, diagnostics.facets_checked);
        assert_eq!(summary.flips_performed, diagnostics.flips_performed);
        assert_eq!(summary.max_queue_len, diagnostics.max_queue_len);
        assert_eq!(
            summary.ambiguous_predicates,
            diagnostics.ambiguous_predicates
        );
        assert_eq!(summary.predicate_failures, diagnostics.predicate_failures);
        assert_eq!(summary.cycle_detections, diagnostics.cycle_detections);
        assert_eq!(summary.attempt, diagnostics.attempt);
        assert_eq!(summary.queue_order, diagnostics.queue_order);
        assert_eq!(
            summary.to_string(),
            "checked=7, flips=3, max_queue=5, ambiguous=2, predicate_failures=1, cycles=4, attempt=2, order=Lifo"
        );

        let non_convergent = FlipNeighborRepairFailure::from(DelaunayRepairError::NonConvergent {
            max_flips: 42,
            diagnostics: Box::new(diagnostics),
        });
        match non_convergent {
            FlipNeighborRepairFailure::NonConvergent {
                max_flips,
                diagnostics,
            } => {
                assert_eq!(max_flips, 42);
                assert_eq!(diagnostics.flips_performed, 3);
            }
            other => panic!("expected non-convergent repair summary, got {other:?}"),
        }
    }

    #[test]
    fn test_delaunay_repair_error_partial_eq() {
        let post_test = DelaunayRepairError::PostconditionFailed {
            reason: Box::new(DelaunayRepairPostconditionFailure::Disconnected { simplex_count: 1 }),
        };
        let post_test_copy = DelaunayRepairError::PostconditionFailed {
            reason: Box::new(DelaunayRepairPostconditionFailure::Disconnected { simplex_count: 1 }),
        };
        let post_other = DelaunayRepairError::PostconditionFailed {
            reason: Box::new(DelaunayRepairPostconditionFailure::Disconnected { simplex_count: 2 }),
        };
        assert_eq!(post_test, post_test_copy);
        assert_ne!(post_test, post_other);

        let verification_err = DelaunayRepairError::VerificationFailed {
            context: DelaunayRepairVerificationContext::StrictValidation,
            source: Box::new(FlipError::DegenerateSimplex),
        };
        let verification_err_copy = DelaunayRepairError::VerificationFailed {
            context: DelaunayRepairVerificationContext::StrictValidation,
            source: Box::new(FlipError::DegenerateSimplex),
        };
        let verification_other = DelaunayRepairError::VerificationFailed {
            context: DelaunayRepairVerificationContext::StrictValidation,
            source: Box::new(FlipError::DuplicateSimplex),
        };
        assert_eq!(verification_err, verification_err_copy);
        assert_ne!(verification_err, verification_other);

        let flip_err = DelaunayRepairError::from(FlipError::DegenerateSimplex);
        let flip_err_copy = DelaunayRepairError::from(FlipError::DegenerateSimplex);
        let flip_other = DelaunayRepairError::from(FlipError::DuplicateSimplex);
        assert_eq!(flip_err, flip_err_copy);
        assert_ne!(flip_err, flip_other);

        let canonicalization_err = DelaunayRepairError::OrientationCanonicalizationFailed {
            reason: Box::new(
                DelaunayRepairOrientationCanonicalizationFailure::AfterFlipRepair {
                    source: Box::new(InsertionError::DuplicateCoordinates {
                        coordinates: CoordinateValues::from([0.0, 0.0]),
                    }),
                },
            ),
        };
        let canonicalization_err_copy = DelaunayRepairError::OrientationCanonicalizationFailed {
            reason: Box::new(
                DelaunayRepairOrientationCanonicalizationFailure::AfterFlipRepair {
                    source: Box::new(InsertionError::DuplicateCoordinates {
                        coordinates: CoordinateValues::from([0.0, 0.0]),
                    }),
                },
            ),
        };
        let canonicalization_other = DelaunayRepairError::OrientationCanonicalizationFailed {
            reason: Box::new(
                DelaunayRepairOrientationCanonicalizationFailure::AfterFlipRepair {
                    source: Box::new(InsertionError::DuplicateCoordinates {
                        coordinates: CoordinateValues::from([1.0, 1.0]),
                    }),
                },
            ),
        };
        assert_eq!(canonicalization_err, canonicalization_err_copy);
        assert_ne!(canonicalization_err, canonicalization_other);

        let topo_err = DelaunayRepairError::InvalidTopology {
            required: TopologyGuarantee::PLManifold,
            found: TopologyGuarantee::Pseudomanifold,
            message: "test",
        };
        let topo_err_copy = DelaunayRepairError::InvalidTopology {
            required: TopologyGuarantee::PLManifold,
            found: TopologyGuarantee::Pseudomanifold,
            message: "test",
        };
        assert_eq!(topo_err, topo_err_copy);

        // Different variants are never equal.
        assert_ne!(post_test, topo_err);
        assert_ne!(post_test, verification_err);
        assert_ne!(post_test, canonicalization_err);
    }

    #[test]
    fn test_postcondition_failure_display_covers_variants() {
        let simplex = SimplexKey::from(KeyData::from_ffi(91));
        let v0 = VertexKey::from(KeyData::from_ffi(101));
        let v1 = VertexKey::from(KeyData::from_ffi(102));
        let v2 = VertexKey::from(KeyData::from_ffi(103));
        let facet = FacetHandle::from_validated(simplex, 0);
        let ridge = RidgeHandle::from_validated(simplex, 0, 1);
        let edge = EdgeKey::from_validated_endpoints(v0, v1);
        let triangle = TriangleHandle::try_new(v0, v1, v2).unwrap();

        assert_eq!(
            DelaunayRepairPostconditionFailure::Disconnected { simplex_count: 2 }.to_string(),
            "repair pass disconnected the triangulation (2 simplices remain); neighbor wiring is incomplete"
        );

        let k2 = DelaunayRepairPostconditionFailure::LocalK2Violation {
            facet,
            debug_details: Some("debug facet snapshot".to_string()),
        }
        .to_string();
        assert!(k2.contains("local k=2 violation remains after repair"));
        assert!(k2.contains("debug facet snapshot"));

        let k3 = DelaunayRepairPostconditionFailure::LocalK3Violation { ridge }.to_string();
        assert!(k3.contains("local k=3 violation remains after repair"));

        let inverse_k2 =
            DelaunayRepairPostconditionFailure::LocalInverseK2Violation { edge }.to_string();
        assert!(inverse_k2.contains("local inverse k=2 flip remains applicable after repair"));

        let inverse_k3 =
            DelaunayRepairPostconditionFailure::LocalInverseK3Violation { triangle }.to_string();
        assert!(inverse_k3.contains("local inverse k=3 flip remains applicable after repair"));
    }

    #[test]
    fn test_postcondition_failure_exposes_source() {
        let reason = DelaunayRepairPostconditionFailure::Disconnected { simplex_count: 1 };
        let repair = DelaunayRepairError::PostconditionFailed {
            reason: Box::new(reason.clone()),
        };
        let source = repair
            .source()
            .and_then(|source| source.downcast_ref::<Box<DelaunayRepairPostconditionFailure>>());
        assert_eq!(source.map(Box::as_ref), Some(&reason));

        let neighbor_repair = FlipNeighborRepairFailure::PostconditionFailed { reason };
        assert_matches!(
            std::error::Error::source(&neighbor_repair)
                .and_then(|source| source.downcast_ref::<DelaunayRepairPostconditionFailure>()),
            Some(DelaunayRepairPostconditionFailure::Disconnected { simplex_count: 1 })
        );
    }

    #[test]
    fn test_heuristic_vertex_context_display() {
        let context = sample_heuristic_vertex_context().to_string();

        assert!(context.contains("idx=3"));
        assert!(context.contains("uuid=00000000-0000-0000-0000-000000000000"));
        assert!(context.contains("coords=[1.0, 2.0]"));
    }

    #[test]
    fn test_orientation_failure_kind_conversion() {
        let orientation_failure =
            DelaunayRepairOrientationCanonicalizationFailure::AfterFlipRepair {
                source: Box::new(InsertionError::DuplicateCoordinates {
                    coordinates: CoordinateValues::from([0.0, 0.0]),
                }),
            };

        assert_eq!(
            DelaunayRepairOrientationCanonicalizationFailureKind::from(&orientation_failure),
            DelaunayRepairOrientationCanonicalizationFailureKind::AfterFlipRepair {
                source_kind: InsertionErrorKind::DuplicateCoordinates,
            },
        );

        let orientation_repair = DelaunayRepairError::OrientationCanonicalizationFailed {
            reason: Box::new(orientation_failure),
        };
        assert_matches!(
            FlipNeighborRepairFailure::from(orientation_repair),
            FlipNeighborRepairFailure::OrientationCanonicalizationFailed {
                reason: DelaunayRepairOrientationCanonicalizationFailureKind::AfterFlipRepair {
                    source_kind: InsertionErrorKind::DuplicateCoordinates
                }
            }
        );
    }

    #[test]
    fn test_heuristic_rebuild_failure_kind_conversion() {
        let insertion_failure = InsertionError::DuplicateCoordinates {
            coordinates: CoordinateValues::from([0.0, 0.0]),
        };
        let repair_source = || DelaunayRepairError::from(FlipError::DegenerateSimplex);
        let vertex = sample_heuristic_vertex_context();
        let heuristic_cases = [
            (
                DelaunayRepairHeuristicRebuildFailure::RecursionDepthExceeded { max_depth: 1 },
                DelaunayRepairHeuristicRebuildFailureKind::RecursionDepthExceeded,
            ),
            (
                DelaunayRepairHeuristicRebuildFailure::FallbackChainFailed {
                    primary: Box::new(repair_source()),
                    robust: Box::new(repair_source()),
                    heuristic: Box::new(DelaunayRepairHeuristicRebuildFailure::NoAttempts),
                },
                DelaunayRepairHeuristicRebuildFailureKind::FallbackChainFailed,
            ),
            (
                DelaunayRepairHeuristicRebuildFailure::UnexpectedRepairFailure {
                    source: Box::new(repair_source()),
                },
                DelaunayRepairHeuristicRebuildFailureKind::UnexpectedRepairFailure,
            ),
            (
                DelaunayRepairHeuristicRebuildFailure::NoAttempts,
                DelaunayRepairHeuristicRebuildFailureKind::NoAttempts,
            ),
            (
                DelaunayRepairHeuristicRebuildFailure::InsertionFailed {
                    vertex: vertex.clone(),
                    source: Box::new(insertion_failure.clone()),
                },
                DelaunayRepairHeuristicRebuildFailureKind::InsertionFailed,
            ),
            (
                DelaunayRepairHeuristicRebuildFailure::RepairFailed {
                    vertex: vertex.clone(),
                    source: Box::new(insertion_failure.clone()),
                },
                DelaunayRepairHeuristicRebuildFailureKind::RepairFailed,
            ),
            (
                DelaunayRepairHeuristicRebuildFailure::DelaunayCheckFailed {
                    vertex: vertex.clone(),
                    source: Box::new(insertion_failure.clone()),
                },
                DelaunayRepairHeuristicRebuildFailureKind::DelaunayCheckFailed,
            ),
            (
                DelaunayRepairHeuristicRebuildFailure::SkippedVertex {
                    vertex,
                    source: Box::new(insertion_failure),
                },
                DelaunayRepairHeuristicRebuildFailureKind::SkippedVertex,
            ),
            (
                DelaunayRepairHeuristicRebuildFailure::AttemptFailed {
                    attempt: 1,
                    max_attempts: 2,
                    shuffle_seed: 3,
                    perturbation_seed: 4,
                    source: Box::new(repair_source()),
                },
                DelaunayRepairHeuristicRebuildFailureKind::AttemptFailed,
            ),
            (
                DelaunayRepairHeuristicRebuildFailure::ExhaustedAttempts {
                    attempts: 2,
                    last_failure: Box::new(DelaunayRepairHeuristicRebuildFailure::NoAttempts),
                },
                DelaunayRepairHeuristicRebuildFailureKind::ExhaustedAttempts,
            ),
        ];

        for (failure, expected_kind) in heuristic_cases {
            assert_eq!(
                DelaunayRepairHeuristicRebuildFailureKind::from(&failure),
                expected_kind,
            );
        }
    }

    #[test]
    fn test_flip_neighbor_repair_failure_conversion() {
        let heuristic_repair = DelaunayRepairError::HeuristicRebuildFailed {
            reason: Box::new(DelaunayRepairHeuristicRebuildFailure::NoAttempts),
        };
        assert_matches!(
            FlipNeighborRepairFailure::from(heuristic_repair),
            FlipNeighborRepairFailure::HeuristicRebuildFailed {
                reason: DelaunayRepairHeuristicRebuildFailureKind::NoAttempts
            }
        );
    }

    #[test]
    fn test_delaunay_repair_error_boxes_large_flip_sources() {
        assert!(
            std::mem::size_of::<DelaunayRepairError>() <= std::mem::size_of::<FlipError>(),
            "DelaunayRepairError should box FlipError payloads without exceeding FlipError size"
        );

        let err = DelaunayRepairError::from(FlipError::DegenerateSimplex);
        let source = err.source().expect("boxed flip source should be exposed");
        let source = source
            .downcast_ref::<Box<FlipError>>()
            .expect("source should remain a typed boxed FlipError");
        assert_matches!(source.as_ref(), FlipError::DegenerateSimplex);

        let DelaunayRepairError::Flip { source } = err else {
            panic!("expected boxed flip source");
        };
        assert_matches!(source.as_ref(), FlipError::DegenerateSimplex);
    }

    #[test]
    fn test_heuristic_exhausted_attempts_exposes_last_failure_source() {
        let exhausted = DelaunayRepairHeuristicRebuildFailure::ExhaustedAttempts {
            attempts: 6,
            last_failure: Box::new(DelaunayRepairHeuristicRebuildFailure::NoAttempts),
        };

        let source = exhausted
            .source()
            .expect("exhausted attempts should expose the last failure source")
            .downcast_ref::<Box<DelaunayRepairHeuristicRebuildFailure>>()
            .expect("source should remain a typed boxed heuristic failure");
        assert_matches!(
            source.as_ref(),
            DelaunayRepairHeuristicRebuildFailure::NoAttempts
        );
    }

    #[test]
    fn test_flip_error_boxes_nested_typed_payloads() {
        let max_nested_payload_size = [
            size_of::<FlipContextError>(),
            size_of::<FlipPredicateError>(),
            size_of::<FlipEdgeAdjacencyError>(),
            size_of::<FlipTriangleAdjacencyError>(),
            size_of::<FlipVertexAdjacencyError>(),
            size_of::<SimplexValidationError>(),
            size_of::<FlipNeighborWiringError>(),
            size_of::<FlipMutationError>(),
            size_of::<SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>>(),
        ]
        .into_iter()
        .max()
        .unwrap_or(0);

        assert!(
            size_of::<FlipError>() < max_nested_payload_size,
            "boxed FlipError should stay smaller than its largest nested payload"
        );
        assert_eq!(align_of::<Result<(), FlipError>>(), align_of::<FlipError>());
        assert!(
            size_of::<Result<(), FlipError>>() <= size_of::<FlipError>() + size_of::<usize>(),
            "Result<(), FlipError> should remain within one machine word of FlipError"
        );

        let mutation = FlipError::from(FlipMutationError::TrialValidation {
            k_move: 2,
            direction: FlipDirection::Forward,
            source: sample_tds_validation_failure(),
        });
        let source = mutation
            .source()
            .expect("boxed mutation source should be exposed")
            .downcast_ref::<Box<FlipMutationError>>()
            .expect("source should remain a typed boxed FlipMutationError");
        assert_matches!(
            source.as_ref(),
            FlipMutationError::TrialValidation {
                k_move: 2,
                direction: FlipDirection::Forward,
                ..
            }
        );

        let FlipError::TdsMutation { reason } = mutation else {
            panic!("expected boxed TDS mutation reason");
        };
        assert_matches!(
            reason.as_ref(),
            FlipMutationError::TrialValidation {
                k_move: 2,
                direction: FlipDirection::Forward,
                ..
            }
        );

        let mut simplex_vertices = SmallBuffer::new();
        simplex_vertices.push(VertexKey::from(KeyData::from_ffi(1)));
        simplex_vertices.push(VertexKey::from(KeyData::from_ffi(2)));
        let duplicate = FlipError::InsertedSimplexAlreadyExists {
            k_move: 2,
            simplex_vertices: Box::new(simplex_vertices),
            existing_simplex: SimplexKey::from(KeyData::from_ffi(3)),
        };

        let FlipError::InsertedSimplexAlreadyExists {
            simplex_vertices, ..
        } = duplicate
        else {
            panic!("expected boxed simplex witness");
        };
        assert_eq!(simplex_vertices.as_ref().len(), 2);
    }

    #[test]
    fn test_flip_error_boxes_adjacency_payload_sources() {
        let edge_vertex = VertexKey::from(KeyData::from_ffi(4));
        let edge = FlipError::from(FlipEdgeAdjacencyError::DuplicateEndpoints {
            vertex_key: edge_vertex,
        });
        let source = edge
            .source()
            .expect("boxed edge-adjacency source should be exposed")
            .downcast_ref::<Box<FlipEdgeAdjacencyError>>()
            .expect("source should remain a typed boxed FlipEdgeAdjacencyError");
        assert_matches!(
            source.as_ref(),
            FlipEdgeAdjacencyError::DuplicateEndpoints { vertex_key }
                if *vertex_key == edge_vertex
        );

        let FlipError::InvalidEdgeAdjacency { reason } = edge else {
            panic!("expected boxed edge-adjacency reason");
        };
        assert_matches!(
            reason.as_ref(),
            FlipEdgeAdjacencyError::DuplicateEndpoints { vertex_key }
                if *vertex_key == edge_vertex
        );

        let simplex_key = SimplexKey::from(KeyData::from_ffi(5));
        let triangle_a = VertexKey::from(KeyData::from_ffi(6));
        let triangle_b = VertexKey::from(KeyData::from_ffi(7));
        let triangle_c = VertexKey::from(KeyData::from_ffi(8));
        let triangle =
            FlipError::from(FlipTriangleAdjacencyError::SimplexMissingTriangleVertices {
                simplex_key,
                a: triangle_a,
                b: triangle_b,
                c: triangle_c,
            });
        let source = triangle
            .source()
            .expect("boxed triangle-adjacency source should be exposed")
            .downcast_ref::<Box<FlipTriangleAdjacencyError>>()
            .expect("source should remain a typed boxed FlipTriangleAdjacencyError");
        assert_matches!(
            source.as_ref(),
            FlipTriangleAdjacencyError::SimplexMissingTriangleVertices { a, b, c, .. }
                if *a == triangle_a && *b == triangle_b && *c == triangle_c
        );

        let FlipError::InvalidTriangleAdjacency { reason } = triangle else {
            panic!("expected boxed triangle-adjacency reason");
        };
        assert_matches!(
            reason.as_ref(),
            FlipTriangleAdjacencyError::SimplexMissingTriangleVertices { a, b, c, .. }
                if *a == triangle_a && *b == triangle_b && *c == triangle_c
        );

        let vertex = FlipError::from(FlipVertexAdjacencyError::SimplexMissingVertex {
            simplex_key,
            vertex_key: edge_vertex,
        });
        let source = vertex
            .source()
            .expect("boxed vertex-adjacency source should be exposed")
            .downcast_ref::<Box<FlipVertexAdjacencyError>>()
            .expect("source should remain a typed boxed FlipVertexAdjacencyError");
        assert_matches!(
            source.as_ref(),
            FlipVertexAdjacencyError::SimplexMissingVertex { vertex_key, .. }
                if *vertex_key == edge_vertex
        );

        let FlipError::InvalidVertexAdjacency { reason } = vertex else {
            panic!("expected boxed vertex-adjacency reason");
        };
        assert_matches!(
            reason.as_ref(),
            FlipVertexAdjacencyError::SimplexMissingVertex { vertex_key, .. }
                if *vertex_key == edge_vertex
        );
    }

    #[test]
    fn test_delaunay_repair_verification_context_display_covers_all_variants() {
        let cases = [
            (
                DelaunayRepairVerificationContext::PostRepairVerification,
                "post-repair verification",
            ),
            (
                DelaunayRepairVerificationContext::StrictValidation,
                "strict validation",
            ),
            (
                DelaunayRepairVerificationContext::LocalK2DegeneracyVerification,
                "local k=2 degeneracy verification",
            ),
            (
                DelaunayRepairVerificationContext::LocalK2PostconditionVerification,
                "local k=2 postcondition verification",
            ),
            (
                DelaunayRepairVerificationContext::LocalK3DegeneracyVerification,
                "local k=3 degeneracy verification",
            ),
            (
                DelaunayRepairVerificationContext::LocalK3PostconditionVerification,
                "local k=3 postcondition verification",
            ),
            (
                DelaunayRepairVerificationContext::LocalInverseK2PostconditionVerification,
                "local inverse k=2 postcondition verification",
            ),
            (
                DelaunayRepairVerificationContext::LocalInverseK3PostconditionVerification,
                "local inverse k=3 postcondition verification",
            ),
        ];

        for (context, expected_display) in cases {
            assert_eq!(context.to_string(), expected_display);
            let err = DelaunayRepairError::VerificationFailed {
                context,
                source: Box::new(FlipError::DegenerateSimplex),
            };

            match err {
                DelaunayRepairError::VerificationFailed {
                    context: observed, ..
                } => assert_eq!(observed, context),
                other => panic!("expected verification failure, got {other:?}"),
            }
        }
    }

    macro_rules! gen_align_periodic_offset_tests {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<test_align_periodic_offset_identity_ $dim d>]() {
                    // Same reference offset in source and target -> no change.
                    let mut source_vertex_offset = [0_i8; $dim];
                    source_vertex_offset[$dim - 1] = 1;
                    let result = align_periodic_offset(
                        source_vertex_offset,
                        [0_i8; $dim],
                        [0_i8; $dim],
                    )
                    .unwrap();
                    assert_eq!(result, source_vertex_offset);
                }

                #[test]
                fn [<test_align_periodic_offset_shifts_by_delta_ $dim d>]() {
                    // delta = target reference - source reference.
                    let mut source_vertex_offset = [0_i8; $dim];
                    source_vertex_offset[0] = 1;
                    let mut target_reference_offset = [0_i8; $dim];
                    target_reference_offset[$dim - 1] = 1;
                    let mut expected = source_vertex_offset;
                    expected[$dim - 1] = expected[$dim - 1].saturating_add(1);

                    let result = align_periodic_offset(
                        source_vertex_offset,
                        [0_i8; $dim],
                        target_reference_offset,
                    )
                    .unwrap();
                    assert_eq!(result, expected);
                }

                #[test]
                fn [<test_align_periodic_offset_negative_delta_ $dim d>]() {
                    let source_vertex_offset = [1_i8; $dim];
                    let mut source_reference_offset = [0_i8; $dim];
                    source_reference_offset[0] = 1;
                    let mut expected = source_vertex_offset;
                    expected[0] = 0;

                    let result = align_periodic_offset(
                        source_vertex_offset,
                        source_reference_offset,
                        [0_i8; $dim],
                    )
                    .unwrap();
                    assert_eq!(result, expected);
                }

                #[test]
                fn [<test_align_periodic_offset_subtraction_overflow_ $dim d>]() {
                    // i8::MIN - 1 overflows.
                    let mut source_reference_offset = [0_i8; $dim];
                    source_reference_offset[0] = 1;
                    let mut target_reference_offset = [0_i8; $dim];
                    target_reference_offset[0] = i8::MIN;

                    let result = align_periodic_offset(
                        [0_i8; $dim],
                        source_reference_offset,
                        target_reference_offset,
                    );
                    assert!(result.is_err());
                }

                #[test]
                fn [<test_align_periodic_offset_addition_overflow_ $dim d>]() {
                    // i8::MAX + 1 overflows.
                    let mut source_vertex_offset = [0_i8; $dim];
                    source_vertex_offset[0] = i8::MAX;
                    let mut target_reference_offset = [0_i8; $dim];
                    target_reference_offset[0] = 1;

                    let result = align_periodic_offset(
                        source_vertex_offset,
                        [0_i8; $dim],
                        target_reference_offset,
                    );
                    assert!(result.is_err());
                }
            }
        };
    }

    gen_align_periodic_offset_tests!(2);
    gen_align_periodic_offset_tests!(3);
    gen_align_periodic_offset_tests!(4);
    gen_align_periodic_offset_tests!(5);

    fn toroidal_model<const D: usize>() -> GlobalTopologyModelAdapter<D> {
        GlobalTopology::try_toroidal([1.0; D], ToroidalConstructionMode::PeriodicImagePoint)
            .unwrap()
            .model()
    }

    fn insert_periodic_simplex_with_lifted_vertex<const D: usize>(
        tds: &mut Tds<(), (), D>,
        vertices: Vec<VertexKey>,
        lifted_vertex: VertexKey,
    ) -> SimplexKey {
        let mut offsets = vec![[0_i8; D]; vertices.len()];
        if let Some(index) = vertices.iter().position(|&vkey| vkey == lifted_vertex) {
            offsets[index][0] = 1;
        }
        let mut simplex = Simplex::try_new_with_data(vertices, None).unwrap();
        simplex.set_periodic_vertex_offsets(offsets).unwrap();
        tds.insert_simplex_with_mapping(simplex).unwrap()
    }

    fn insert_periodic_simplex_with_offsets<const D: usize>(
        tds: &mut Tds<(), (), D>,
        vertices: Vec<VertexKey>,
        offsets: Vec<[i8; D]>,
    ) -> SimplexKey {
        let mut simplex = Simplex::try_new_with_data(vertices, None).unwrap();
        simplex.set_periodic_vertex_offsets(offsets).unwrap();
        tds.insert_simplex_with_mapping(simplex).unwrap()
    }

    fn insert_plain_simplex<const D: usize>(
        tds: &mut Tds<(), (), D>,
        vertices: Vec<VertexKey>,
    ) -> SimplexKey {
        tds.insert_simplex_with_mapping(Simplex::try_new_with_data(vertices, None).unwrap())
            .unwrap()
    }

    fn assert_dangling_vertex_incidence(
        err: &FlipError,
        expected_vertex: VertexKey,
        expected_simplex: SimplexKey,
    ) {
        assert_matches!(
            err,
            FlipError::DanglingVertexIncidence {
                vertex_key,
                simplex_key,
            } if *vertex_key == expected_vertex && *simplex_key == expected_simplex
        );
    }

    macro_rules! gen_k2_edge_adjacency_validation_tests {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<build_k2_inverse_context_rejects_missing_endpoint_incidence_as_adjacency_ $dim d>]() {
                    let mut tds: Tds<(), (), $dim> = Tds::empty();
                    let vertices = insert_standard_simplex_vertices(&mut tds);
                    let simplex_key = insert_plain_simplex(&mut tds, vertices.clone());
                    tds.clear_vertex_incidence_for_test(vertices[1]);

                    let edge = EdgeKey::from_validated_endpoints(vertices[0], vertices[1]);
                    let err = build_k2_flip_context_from_edge(&tds, edge).unwrap_err();

                    assert_matches!(
                        err,
                        FlipError::InvalidEdgeAdjacency { reason }
                            if matches!(
                                reason.as_ref(),
                                FlipEdgeAdjacencyError::MissingVertexIncidence {
                                    vertex_key,
                                    simplex_key: reported_simplex,
                                } if *vertex_key == vertices[1] && *reported_simplex == simplex_key
                            )
                    );
                }

                #[test]
                fn [<build_k2_inverse_context_rejects_missing_edge_incidence_as_adjacency_ $dim d>]() {
                    let mut tds: Tds<(), (), $dim> = Tds::empty();
                    let vertices = insert_standard_simplex_vertices(&mut tds);
                    insert_plain_simplex(&mut tds, vertices.clone());
                    tds.clear_vertex_incidence_for_test(vertices[0]);
                    tds.clear_vertex_incidence_for_test(vertices[1]);

                    let edge = EdgeKey::from_validated_endpoints(vertices[0], vertices[1]);
                    let err = build_k2_flip_context_from_edge(&tds, edge).unwrap_err();

                    assert_matches!(
                        err,
                        FlipError::InvalidEdgeAdjacency { reason }
                            if matches!(
                                reason.as_ref(),
                                FlipEdgeAdjacencyError::MissingEdgeIncidence { v0, v1 }
                                    if (*v0, *v1) == edge.endpoints()
                            )
                    );
                }

                #[test]
                fn [<build_k2_inverse_context_rejects_vertex_incidence_mismatch_as_adjacency_ $dim d>]() {
                    let mut tds: Tds<(), (), $dim> = Tds::empty();
                    let vertices = insert_standard_simplex_vertices(&mut tds);
                    insert_plain_simplex(&mut tds, vertices.clone());
                    let mut extra_coords = [0.0_f64; $dim];
                    extra_coords[0] = 2.0;
                    let extra_vertex = tds
                        .insert_vertex_with_mapping(Vertex::<(), _>::try_new(extra_coords).unwrap())
                        .unwrap();
                    let mut mismatched_vertices = vertices[1..].to_vec();
                    mismatched_vertices.push(extra_vertex);
                    let mismatched_simplex = insert_plain_simplex(&mut tds, mismatched_vertices);
                    tds.add_simplex_to_vertex_incidence_for_test(vertices[0], mismatched_simplex);

                    let edge = EdgeKey::from_validated_endpoints(vertices[0], vertices[1]);
                    let err = build_k2_flip_context_from_edge(&tds, edge).unwrap_err();

                    assert_matches!(
                        err,
                        FlipError::InvalidEdgeAdjacency { reason }
                            if matches!(
                                reason.as_ref(),
                                FlipEdgeAdjacencyError::VertexIncidenceMismatch {
                                    vertex_key,
                                    simplex_key,
                                } if *vertex_key == vertices[0] && *simplex_key == mismatched_simplex
                            )
                    );
                }
            }
        };
    }

    gen_k2_edge_adjacency_validation_tests!(3);
    gen_k2_edge_adjacency_validation_tests!(4);
    gen_k2_edge_adjacency_validation_tests!(5);

    macro_rules! gen_stale_incidence_context_tests {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<find_simplex_containing_simplex_rejects_stale_incidence_key_ $dim d>]() {
                    let mut tds: Tds<(), (), $dim> = Tds::empty();
                    let vertices = insert_standard_simplex_vertices(&mut tds);
                    let stale_simplex = insert_plain_simplex(&mut tds, vertices.clone());
                    tds.remove_simplex_storage_only_for_test(stale_simplex);

                    let err = find_simplex_containing_simplex(
                        &tds,
                        &vertices[..2],
                        &SimplexKeyBuffer::new(),
                    )
                    .unwrap_err();

                    assert_dangling_vertex_incidence(&err, vertices[0], stale_simplex);
                }

                #[test]
                fn [<build_k1_inverse_context_rejects_stale_incidence_before_multiplicity_ $dim d>]() {
                    let mut tds: Tds<(), (), $dim> = Tds::empty();
                    let vertices = insert_standard_simplex_vertices(&mut tds);
                    let stale_simplex = insert_plain_simplex(&mut tds, vertices.clone());
                    tds.remove_simplex_storage_only_for_test(stale_simplex);

                    let err = build_k1_inverse_context(&tds, vertices[0]).unwrap_err();

                    assert_dangling_vertex_incidence(&err, vertices[0], stale_simplex);
                }
            }
        };
    }

    macro_rules! gen_k2_stale_incidence_context_tests {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<build_k2_inverse_context_rejects_stale_incidence_key_ $dim d>]() {
                    let mut tds: Tds<(), (), $dim> = Tds::empty();
                    let vertices = insert_standard_simplex_vertices(&mut tds);
                    let stale_simplex = insert_plain_simplex(&mut tds, vertices.clone());
                    tds.remove_simplex_storage_only_for_test(stale_simplex);

                    let edge = EdgeKey::from_validated_endpoints(vertices[0], vertices[1]);
                    let err = build_k2_flip_context_from_edge(&tds, edge).unwrap_err();

                    assert_dangling_vertex_incidence(&err, vertices[0], stale_simplex);
                }
            }
        };
    }

    macro_rules! gen_k3_stale_incidence_context_tests {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<build_k3_inverse_context_rejects_stale_incidence_key_ $dim d>]() {
                    let mut tds: Tds<(), (), $dim> = Tds::empty();
                    let vertices = insert_standard_simplex_vertices(&mut tds);
                    let stale_simplex = insert_plain_simplex(&mut tds, vertices.clone());
                    tds.remove_simplex_storage_only_for_test(stale_simplex);

                    let triangle =
                        TriangleHandle::try_new(vertices[0], vertices[1], vertices[2]).unwrap();
                    let err = build_k3_flip_context_from_triangle(&tds, triangle).unwrap_err();

                    assert_dangling_vertex_incidence(&err, vertices[0], stale_simplex);
                }
            }
        };
    }

    gen_stale_incidence_context_tests!(2);
    gen_stale_incidence_context_tests!(3);
    gen_stale_incidence_context_tests!(4);
    gen_stale_incidence_context_tests!(5);
    gen_k2_stale_incidence_context_tests!(3);
    gen_k2_stale_incidence_context_tests!(4);
    gen_k2_stale_incidence_context_tests!(5);
    gen_k3_stale_incidence_context_tests!(4);
    gen_k3_stale_incidence_context_tests!(5);

    fn periodic_helper_vertices<const D: usize>(
        tds: &mut Tds<(), (), D>,
        count: usize,
    ) -> Vec<VertexKey> {
        (0..count)
            .map(|index| {
                let mut coords = [0.0; D];
                coords[index % D] =
                    0.05 * f64::from(u32::try_from(index + 1).expect("test index fits in u32"));
                let next_index = (index + 1) % D;
                coords[next_index] = 0.01_f64.mul_add(
                    f64::from(u32::try_from(index + 2).expect("test index fits in u32")),
                    coords[next_index],
                );
                tds.insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(coords).unwrap(),
                )
                .unwrap()
            })
            .collect()
    }

    macro_rules! gen_periodic_lift_helper_tests {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<test_periodic_lift_helpers_use_simplex_offsets_ $dim d>]() {
                    let mut tds: Tds<(), (), $dim> = Tds::empty();
                    let lifted_vertex = tds
                        .insert_vertex_with_mapping(crate::core::vertex::Vertex::<(), _>::try_new(unit_vector::<$dim>(0)).unwrap())
                        .unwrap();
                    let mut simplex_vertices = Vec::with_capacity($dim + 1);
                    simplex_vertices.push(lifted_vertex);
                    simplex_vertices.extend(periodic_helper_vertices::<$dim>(&mut tds, $dim));
                    let mut offsets = vec![[0_i8; $dim]; simplex_vertices.len()];
                    offsets[0][0] = 1;
                    let simplex_key =
                        insert_periodic_simplex_with_offsets(&mut tds, simplex_vertices.clone(), offsets);
                    let topology_model = toroidal_model::<$dim>();

                    let direct = vertex_point_with_optional_lift(
                        &tds,
                        &topology_model,
                        lifted_vertex,
                        Some(simplex_key),
                    )
                    .unwrap();
                    let mut expected = unit_vector::<$dim>(0);
                    expected[0] += 1.0;
                    assert_relative_eq!(direct.coords().as_slice(), expected.as_slice());

                    let framed = vertex_point_lifted_into_simplex(
                        &tds,
                        &topology_model,
                        lifted_vertex,
                        Some(simplex_key),
                        &[],
                    )
                    .unwrap();
                    assert_relative_eq!(framed.coords().as_slice(), expected.as_slice());

                    let points = vertices_to_points_with_optional_lift(
                        &tds,
                        &topology_model,
                        &[lifted_vertex],
                        Some(simplex_key),
                        &[simplex_key],
                    )
                    .unwrap();
                    assert_relative_eq!(points[0].coords().as_slice(), expected.as_slice());
                    assert_eq!(matching_source_simplex(&tds, &simplex_vertices, &[simplex_key]), Some(simplex_key));
                }

                #[test]
                fn [<test_periodic_lift_treats_missing_source_offsets_as_zero_frame_ $dim d>]() {
                    let mut tds: Tds<(), (), $dim> = Tds::empty();
                    let shared_vertex = tds
                        .insert_vertex_with_mapping(crate::core::vertex::Vertex::<(), _>::try_new([0.0; $dim]).unwrap())
                        .unwrap();
                    let lifted_vertex = tds
                        .insert_vertex_with_mapping(crate::core::vertex::Vertex::<(), _>::try_new(unit_vector::<$dim>(0)).unwrap())
                        .unwrap();

                    let mut target_vertices = Vec::with_capacity($dim + 1);
                    target_vertices.push(shared_vertex);
                    target_vertices.extend(periodic_helper_vertices::<$dim>(&mut tds, $dim));
                    let target_offsets = vec![[0_i8; $dim]; target_vertices.len()];
                    let target_simplex =
                        insert_periodic_simplex_with_offsets(&mut tds, target_vertices, target_offsets);

                    let mut source_vertices = Vec::with_capacity($dim + 1);
                    source_vertices.push(shared_vertex);
                    source_vertices.push(lifted_vertex);
                    source_vertices.extend(periodic_helper_vertices::<$dim>(
                        &mut tds,
                        $dim - 1,
                    ));
                    let source_simplex = insert_plain_simplex(&mut tds, source_vertices);
                    let topology_model = toroidal_model::<$dim>();

                    let result = vertex_point_lifted_into_simplex(
                        &tds,
                        &topology_model,
                        lifted_vertex,
                        Some(target_simplex),
                        &[source_simplex],
                    );
                    let lifted = result.unwrap();
                    assert_relative_eq!(
                        lifted.coords().as_slice(),
                        unit_vector::<$dim>(0).as_slice()
                    );
                }

                #[test]
                fn [<test_periodic_lift_rejects_conflicting_shared_translations_ $dim d>]() {
                    let mut tds: Tds<(), (), $dim> = Tds::empty();
                    let shared_a = tds
                        .insert_vertex_with_mapping(crate::core::vertex::Vertex::<(), _>::try_new([0.0; $dim]).unwrap())
                        .unwrap();
                    let mut shared_b_coords = [0.0; $dim];
                    shared_b_coords[0] = 0.2;
                    let shared_b = tds
                        .insert_vertex_with_mapping(crate::core::vertex::Vertex::<(), _>::try_new(shared_b_coords).unwrap())
                        .unwrap();
                    let lifted_vertex = tds
                        .insert_vertex_with_mapping(crate::core::vertex::Vertex::<(), _>::try_new(unit_vector::<$dim>(0)).unwrap())
                        .unwrap();

                    let mut target_vertices = Vec::with_capacity($dim + 1);
                    target_vertices.push(shared_a);
                    target_vertices.push(shared_b);
                    target_vertices.extend(periodic_helper_vertices::<$dim>(&mut tds, $dim - 1));
                    let mut target_offsets = vec![[0_i8; $dim]; target_vertices.len()];
                    target_offsets[1][0] = 1;
                    let target_simplex =
                        insert_periodic_simplex_with_offsets(&mut tds, target_vertices, target_offsets);

                    let mut source_vertices = Vec::with_capacity($dim + 1);
                    source_vertices.push(shared_a);
                    source_vertices.push(shared_b);
                    source_vertices.push(lifted_vertex);
                    source_vertices.extend(periodic_helper_vertices::<$dim>(
                        &mut tds,
                        $dim - 2,
                    ));
                    let source_offsets = vec![[0_i8; $dim]; source_vertices.len()];
                    let source_simplex =
                        insert_periodic_simplex_with_offsets(&mut tds, source_vertices, source_offsets);
                    let topology_model = toroidal_model::<$dim>();

                    let result = vertex_point_lifted_into_simplex(
                        &tds,
                        &topology_model,
                        lifted_vertex,
                        Some(target_simplex),
                        &[source_simplex],
                    );
                    assert!(
                        matches!(
                            result,
                            Err(FlipError::InvalidFlipContext { ref reason })
                                if matches!(
                                    reason.as_ref(),
                                    FlipContextError::ConflictingPeriodicFrameTranslation { .. }
                                )
                        ),
                        "conflicting shared translations should be rejected: {result:?}"
                    );
                }

                #[test]
                fn [<test_removed_simplex_frame_requires_source_simplex_ $dim d>]() {
                    let result = removed_simplex_frame(&[]);
                    assert_matches!(result, Err(FlipError::InvalidFlipContext { .. }));
                }
            }
        };
    }

    gen_periodic_lift_helper_tests!(2);
    gen_periodic_lift_helper_tests!(3);
    gen_periodic_lift_helper_tests!(4);
    gen_periodic_lift_helper_tests!(5);

    fn periodic_inverse_k2_fixture<const D: usize>() -> (
        Tds<(), (), D>,
        Vec<VertexKey>,
        VertexKey,
        VertexKey,
        SimplexKeyBuffer,
    ) {
        let mut tds: Tds<(), (), D> = Tds::empty();
        let mut face_vertices = Vec::with_capacity(D);
        for axis in 0..D {
            face_vertices.push(
                tds.insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(unit_vector::<D>(axis)).unwrap(),
                )
                .unwrap(),
            );
        }
        let opposite_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0; D]).unwrap(),
            )
            .unwrap();
        let opposite_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.25; D]).unwrap(),
            )
            .unwrap();

        let lifted_vertex = face_vertices[0];
        let mut removed_simplices = SimplexKeyBuffer::new();
        for skip in 0..D {
            let mut vertices = Vec::with_capacity(D + 1);
            vertices.push(opposite_a);
            vertices.push(opposite_b);
            for (index, &vertex) in face_vertices.iter().enumerate() {
                if index != skip {
                    vertices.push(vertex);
                }
            }
            removed_simplices.push(insert_periodic_simplex_with_lifted_vertex(
                &mut tds,
                vertices,
                lifted_vertex,
            ));
        }

        (
            tds,
            face_vertices,
            opposite_a,
            opposite_b,
            removed_simplices,
        )
    }

    fn periodic_inverse_k3_fixture<const D: usize>() -> (
        Tds<(), (), D>,
        Vec<VertexKey>,
        Vec<VertexKey>,
        SimplexKeyBuffer,
    ) {
        let mut tds: Tds<(), (), D> = Tds::empty();
        let mut ridge_vertices = Vec::with_capacity(D - 1);
        for axis in 0..(D - 1) {
            ridge_vertices.push(
                tds.insert_vertex_with_mapping(
                    crate::core::vertex::Vertex::<(), _>::try_new(unit_vector::<D>(axis)).unwrap(),
                )
                .unwrap(),
            );
        }
        let a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0; D]).unwrap(),
            )
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new(unit_vector::<D>(D - 1)).unwrap(),
            )
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new(skewed_point::<D>()).unwrap(),
            )
            .unwrap();
        let triangle_vertices = vec![a, b, c];

        let lifted_vertex = ridge_vertices[0];
        let mut removed_simplices = SimplexKeyBuffer::new();
        for skip in 0..(D - 1) {
            let mut vertices = Vec::with_capacity(D + 1);
            vertices.extend_from_slice(&triangle_vertices);
            for (index, &vertex) in ridge_vertices.iter().enumerate() {
                if index != skip {
                    vertices.push(vertex);
                }
            }
            removed_simplices.push(insert_periodic_simplex_with_lifted_vertex(
                &mut tds,
                vertices,
                lifted_vertex,
            ));
        }

        (tds, ridge_vertices, triangle_vertices, removed_simplices)
    }

    macro_rules! gen_periodic_inverse_predicate_tests {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<test_periodic_inverse_k2_uses_removed_simplex_frame_ $dim d>]() {
                    let (tds, face_vertices, opposite_a, opposite_b, removed_simplices) =
                        periodic_inverse_k2_fixture::<$dim>();
                    let mut target_simplex_vertices = face_vertices.clone();
                    target_simplex_vertices.push(opposite_a);
                    target_simplex_vertices.sort_unstable_by_key(|v| v.data().as_ffi());
                    assert!(
                        matching_source_simplex(&tds, &target_simplex_vertices, &removed_simplices)
                            .is_none(),
                        "inverse k=2 target simplex should require explicit frame alignment",
                    );

                    let topology_model = toroidal_model::<$dim>();
                    let frame_simplex = removed_simplex_frame(&removed_simplices).unwrap();
                    let lifted = vertex_point_lifted_into_simplex(
                        &tds,
                        &topology_model,
                        face_vertices[0],
                        Some(frame_simplex),
                        &removed_simplices,
                    )
                    .unwrap();
                    let mut expected = unit_vector::<$dim>(0);
                    expected[0] += 1.0;
                    assert_relative_eq!(lifted.coords().as_slice(), expected.as_slice());

                    let kernel = AdaptiveKernel::<f64>::new();
                    let config = RepairAttemptConfig {
                        attempt: 0,
                        queue_order: RepairQueueOrder::Fifo,
                        max_flips_override: None,
                    };
                    let mut diagnostics = RepairDiagnostics::default();
                    let result = delaunay_violation_k2_for_facet(
                        &tds,
                        &kernel,
                        &topology_model,
                        &face_vertices,
                        opposite_a,
                        opposite_b,
                        &removed_simplices,
                        Some(frame_simplex),
                        &config,
                        &mut diagnostics,
                    );
                    assert!(result.is_ok(), "inverse k=2 predicate should align periodic frame: {result:?}");
                }

                #[test]
                fn [<test_periodic_inverse_k3_uses_removed_simplex_frame_ $dim d>]() {
                    let (tds, ridge_vertices, triangle_vertices, removed_simplices) =
                        periodic_inverse_k3_fixture::<$dim>();
                    let mut target_simplex_vertices = ridge_vertices.clone();
                    target_simplex_vertices.extend_from_slice(&triangle_vertices[1..]);
                    target_simplex_vertices.sort_unstable_by_key(|v| v.data().as_ffi());
                    assert!(
                        matching_source_simplex(&tds, &target_simplex_vertices, &removed_simplices)
                            .is_none(),
                        "inverse k=3 target simplex should require explicit frame alignment",
                    );

                    let topology_model = toroidal_model::<$dim>();
                    let frame_simplex = removed_simplex_frame(&removed_simplices).unwrap();
                    let lifted = vertex_point_lifted_into_simplex(
                        &tds,
                        &topology_model,
                        ridge_vertices[0],
                        Some(frame_simplex),
                        &removed_simplices,
                    )
                    .unwrap();
                    let mut expected = unit_vector::<$dim>(0);
                    expected[0] += 1.0;
                    assert_relative_eq!(lifted.coords().as_slice(), expected.as_slice());

                    let kernel = AdaptiveKernel::<f64>::new();
                    let config = RepairAttemptConfig {
                        attempt: 0,
                        queue_order: RepairQueueOrder::Fifo,
                        max_flips_override: None,
                    };
                    let mut diagnostics = RepairDiagnostics::default();
                    let result = delaunay_violation_k3_for_ridge(
                        &tds,
                        &kernel,
                        &topology_model,
                        &ridge_vertices,
                        &triangle_vertices,
                        &removed_simplices,
                        Some(frame_simplex),
                        &config,
                        &mut diagnostics,
                    );
                    assert!(result.is_ok(), "inverse k=3 predicate should align periodic frame: {result:?}");
                }
            }
        };
    }

    gen_periodic_inverse_predicate_tests!(4);
    gen_periodic_inverse_predicate_tests!(5);

    #[test]
    fn test_non_periodic_lift_ignores_stored_periodic_offsets() {
        let (tds, face_vertices, _opposite_a, _opposite_b, removed_simplices) =
            periodic_inverse_k2_fixture::<4>();
        let lifted_vertex = face_vertices[0];
        let source_simplex = removed_simplices
            .iter()
            .copied()
            .find(|&simplex_key| {
                tds.simplex(simplex_key)
                    .is_some_and(|simplex| simplex.contains_vertex(lifted_vertex))
            })
            .expect("fixture should contain a removed simplex with the lifted vertex");
        let topology_model = GlobalTopology::Euclidean.model();

        let direct = vertex_point_with_optional_lift(
            &tds,
            &topology_model,
            lifted_vertex,
            Some(source_simplex),
        )
        .unwrap();
        assert_relative_eq!(direct.coords().as_slice(), unit_vector::<4>(0).as_slice());

        let framed = vertex_point_lifted_into_simplex(
            &tds,
            &topology_model,
            lifted_vertex,
            Some(source_simplex),
            &removed_simplices,
        )
        .unwrap();
        assert_relative_eq!(framed.coords().as_slice(), unit_vector::<4>(0).as_slice());
    }

    #[test]
    fn test_periodic_inverse_k2_alignment_failure_is_error() {
        let (tds, face_vertices, opposite_a, opposite_b, removed_simplices) =
            periodic_inverse_k2_fixture::<4>();
        let topology_model = toroidal_model::<4>();
        let frame_simplex = removed_simplex_frame(&removed_simplices).unwrap();
        let truncated_removed_simplices: SimplexKeyBuffer =
            std::iter::once(frame_simplex).collect();
        let lift_result = vertex_point_lifted_into_simplex(
            &tds,
            &topology_model,
            face_vertices[0],
            Some(frame_simplex),
            &truncated_removed_simplices,
        );
        assert_matches!(lift_result, Err(FlipError::InvalidFlipContext { .. }));

        let kernel = AdaptiveKernel::<f64>::new();
        let config = RepairAttemptConfig {
            attempt: 0,
            queue_order: RepairQueueOrder::Fifo,
            max_flips_override: None,
        };
        let mut diagnostics = RepairDiagnostics::default();

        let result = delaunay_violation_k2_for_facet(
            &tds,
            &kernel,
            &topology_model,
            &face_vertices,
            opposite_a,
            opposite_b,
            &truncated_removed_simplices,
            Some(frame_simplex),
            &config,
            &mut diagnostics,
        );

        assert!(
            result.is_err(),
            "periodic inverse predicate should not fall back to bare coordinates"
        );
    }

    #[test]
    fn test_repair_run_full_reseed_preserves_mutation_frontier() {
        init_tracing();
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.2]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let tds = dt.tds();
        let local_simplex = tds.simplex_keys().next().unwrap();
        let outcome = RepairAttemptOutcome {
            postcondition_required: false,
            stats: DelaunayRepairStats::default(),
            last_applied_flip: None,
            touched_simplices: once(local_simplex).collect(),
            used_full_reseed: true,
        };

        let run = repair_run_from_attempt(outcome);

        assert!(run.used_full_reseed);
        assert!(
            tds.simplex_keys().count() > 1,
            "fixture should distinguish local and full frontiers"
        );
        assert_eq!(run.touched_simplices.len(), 1);
        assert_eq!(run.touched_simplices[0], local_simplex);
    }

    #[test]
    fn test_repair_k2_empty_seed_does_not_full_reseed() {
        init_tracing();
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.2]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let mut tds = dt.tds().clone();
        let before = snapshot_topology(&tds);
        let kernel = AdaptiveKernel::<f64>::new();
        let config = RepairAttemptConfig {
            attempt: 1,
            queue_order: RepairQueueOrder::Fifo,
            max_flips_override: None,
        };
        let empty_seeds: &[SimplexKey] = &[];

        let outcome =
            repair_delaunay_with_flips_k2_attempt(&mut tds, &kernel, Some(empty_seeds), &config)
                .unwrap();

        assert!(!outcome.used_full_reseed);
        assert_eq!(outcome.stats.facets_checked, 0);
        assert!(outcome.touched_simplices.is_empty());
        assert_eq!(snapshot_topology(&tds), before);
    }

    #[test]
    fn test_repair_queue_k2_local_seed() {
        init_tracing();
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.2]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let mut tds = dt.tds().clone();
        let kernel = AdaptiveKernel::<f64>::new();

        let seed_simplex = tds.simplex_keys().next().unwrap();
        let stats = repair_delaunay_with_flips_k2_k3(
            &mut tds,
            &kernel,
            Some(&[seed_simplex]),
            TopologyGuarantee::PLManifold,
            None,
        )
        .unwrap();
        assert!(stats.facets_checked > 0);
        assert!(tds.is_valid().is_ok());
    }
}
