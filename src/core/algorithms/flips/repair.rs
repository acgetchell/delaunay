//! Queued flip-based Delaunay verification, repair, and diagnostics.
//!
//! Incremental topological flipping for regular triangulations provides the
//! mathematical basis for this repair strategy \[4]. The implementation adds
//! explicit PL-manifold preconditions, finite work budgets, rollback, and final
//! certification; the theorem does not turn every arbitrary higher-dimensional
//! local flip schedule into an unconditional convergence guarantee. See
//! `REFERENCES.md` for the numbered bibliography.

#![forbid(unsafe_code)]

use super::{
    AllFacetsIter, BistellarFlipKind, DataType, DelaunayRepairError,
    DelaunayRepairPostconditionFailure, DelaunayRepairVerificationContext, Duration, EdgeKey,
    FacetHandle, FastHashMap, FastHashSet, FlipDirection, FlipError, FlipSignature, GlobalTopology,
    GlobalTopologyModelAdapter, Instant, Kernel, LastAppliedFlip, MAX_REPEAT_SIGNATURE,
    RepairAttemptConfig, RepairDiagnostics, RepairQueueOrder, RepairQueues, RidgeHandle,
    RobustKernel, SimplexKey, SimplexKeyBuffer, Tds, TdsRollbackTransaction, TdsRollbackWindow,
    TopologicalOperation, TopologyGuarantee, TriangleHandle, VecDeque, VertexKey,
    apply_delaunay_flip_k2, build_k2_flip_context, build_k2_flip_context_from_edge,
    build_k3_flip_context, build_k3_flip_context_from_triangle, debug_postcondition_facet_context,
    debug_ridge_context, default_max_flips, delaunay_violation_k2_for_facet,
    delaunay_violation_k3_for_ridge, duration_nanos_saturating, emit_repair_debug_summary,
    enqueue_facet, enqueue_simplex_facets, env, facet_key_from_vertices,
    facet_vertices_from_simplex, flip_signature, flip_would_create_degenerate_simplex,
    is_delaunay_violation_k2, is_delaunay_violation_k3, k2_flip_would_create_degenerate_simplex,
    non_convergent_error, pop_queue, removed_simplex_frame, repair_ridge_debug_enabled,
    repair_trace_enabled, ridge_vertices_from_simplex, run_next_edge_repair_step,
    run_next_facet_repair_step, run_next_ridge_repair_step, run_next_triangle_repair_step,
    seed_repair_queues,
};

/// Run a single flip-repair attempt using k=2 (and k=3 in 3D+).
pub(super) fn repair_delaunay_with_flips_k2_k3_attempt<K, U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    kernel: &K,
    seed_simplices: Option<&[SimplexKey]>,
    global_topology: GlobalTopology<D>,
    config: &RepairAttemptConfig,
) -> Result<RepairAttemptOutcome, DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    repair_delaunay_with_flips_k2_k3_attempt_timed(
        tds,
        kernel,
        seed_simplices,
        global_topology,
        config,
        None,
    )
}

/// Run a single flip-repair attempt while reporting queue-family timings.
#[expect(
    clippy::too_many_lines,
    reason = "Repair loop contains inline tracing and queue handling for diagnostics"
)]
pub(super) fn repair_delaunay_with_flips_k2_k3_attempt_timed<K, U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    kernel: &K,
    seed_simplices: Option<&[SimplexKey]>,
    global_topology: GlobalTopology<D>,
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
    let topology_model = global_topology.model();
    if D == 2 {
        return repair_delaunay_with_flips_k2_attempt(
            tds,
            kernel,
            seed_simplices,
            &topology_model,
            config,
        );
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
                    kernel,
                    &topology_model,
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
                        kernel,
                        &topology_model,
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
                        kernel,
                        &topology_model,
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
                kernel,
                &topology_model,
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
                kernel,
                &topology_model,
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
                    kernel,
                    &topology_model,
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
                    kernel,
                    &topology_model,
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

/// Detect repeated flip signatures and abort on cycles.
#[derive(Debug, Clone, Copy)]
pub(super) struct FlipCycleContext<'a> {
    signature: FlipSignature,
    kind: BistellarFlipKind,
    direction: FlipDirection,
    removed_face_vertices: &'a [VertexKey],
    inserted_face_vertices: &'a [VertexKey],
}

impl<'a> FlipCycleContext<'a> {
    /// Bundles the flip data needed for diagnostics without cloning vertex
    /// buffers on every repair step.
    pub(super) const fn from_validated_flip(
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
pub(super) fn check_flip_cycle<U, V, const D: usize>(
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
pub(super) fn resolve_facet_handle_for_key<U, V, const D: usize>(
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
pub(super) fn resolve_ridge_handle_for_key<U, V, const D: usize>(
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
pub(super) fn publish_local_repair_phase_timing(
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
pub(super) struct RepairAttemptOutcome {
    pub(super) postcondition_required: bool,
    pub(super) stats: DelaunayRepairStats,
    pub(super) last_applied_flip: Option<LastAppliedFlip>,
    pub(super) touched_simplices: SimplexKeyBuffer,
    pub(super) used_full_reseed: bool,
}

/// Determines whether repair changed or observed enough local state to require postcondition replay.
pub(super) const fn repair_postcondition_required(
    stats: &DelaunayRepairStats,
    diagnostics: &RepairDiagnostics,
) -> bool {
    stats.flips_performed > 0 || diagnostics.saw_applicable_repair_site
}

/// Adds newly-created simplices to the repair mutation frontier without duplicates.
pub(super) fn record_touched_simplices(
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
pub(super) fn local_postcondition_frontier(
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
pub(super) fn repair_run_from_attempt(outcome: RepairAttemptOutcome) -> DelaunayRepairRun {
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
pub(super) fn repair_delaunay_with_flips_k2_attempt<K, U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    kernel: &K,
    seed_simplices: Option<&[SimplexKey]>,
    topology_model: &GlobalTopologyModelAdapter<D>,
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
            topology_model,
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

        let kind = BistellarFlipKind::from_validated(2, D);
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
                | FlipError::SimplexCreation { source: _ }),
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
    global_topology: GlobalTopology<D>,
    max_flips_override: Option<usize>,
) -> Result<DelaunayRepairStats, DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    repair_delaunay_with_flips_k2_k3_run(
        tds,
        kernel,
        seed_simplices,
        topology,
        global_topology,
        max_flips_override,
    )
    .map(|run| run.stats)
}

pub(super) fn run_full_reseed_retry<K, U, V, W, const D: usize>(
    transaction: &mut W,
    kernel: &K,
    global_topology: GlobalTopology<D>,
    config: &RepairAttemptConfig,
) -> Result<DelaunayRepairRun, DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
    W: TdsRollbackWindow<U, V, D>,
{
    transaction.restore_rollback_tds();
    let retry_seed_simplices = None;
    let topology_model = global_topology.model();
    let attempt_result = if D == 2 {
        repair_delaunay_with_flips_k2_attempt(
            transaction.rollback_tds_mut(),
            kernel,
            retry_seed_simplices,
            &topology_model,
            config,
        )
    } else {
        repair_delaunay_with_flips_k2_k3_attempt(
            transaction.rollback_tds_mut(),
            kernel,
            retry_seed_simplices,
            global_topology,
            config,
        )
    };

    let outcome = attempt_result?;
    verify_repair_postcondition_with_topology(
        transaction.rollback_tds_mut(),
        kernel,
        retry_seed_simplices,
        global_topology,
        PostconditionMode::Repair,
        outcome.last_applied_flip.as_ref(),
        ConnectivityPostcondition::Check,
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
    global_topology: GlobalTopology<D>,
    max_flips_override: Option<usize>,
) -> Result<DelaunayRepairRun, DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    let mut transaction = TdsRollbackTransaction::begin(tds);
    match repair_delaunay_with_flips_k2_k3_run_in_transaction(
        &mut transaction,
        kernel,
        seed_simplices,
        topology,
        global_topology,
        max_flips_override,
    ) {
        Ok(run) => {
            transaction.commit();
            Ok(run)
        }
        Err(error) => {
            transaction.rollback();
            Err(error)
        }
    }
}

/// Runs Delaunay repair inside a rollback window owned by the caller.
///
/// This lets a higher proof transition retain the same snapshot through
/// orientation normalization and final Level 5 certification. The caller must
/// commit on complete success or roll back before publishing any failure.
pub(crate) fn repair_delaunay_with_flips_k2_k3_run_in_transaction<K, U, V, W, const D: usize>(
    transaction: &mut W,
    kernel: &K,
    seed_simplices: Option<&[SimplexKey]>,
    topology: TopologyGuarantee,
    global_topology: GlobalTopology<D>,
    max_flips_override: Option<usize>,
) -> Result<DelaunayRepairRun, DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
    W: TdsRollbackWindow<U, V, D>,
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

    let topology_model = global_topology.model();

    let attempt1_result = if D == 2 {
        repair_delaunay_with_flips_k2_attempt(
            transaction.rollback_tds_mut(),
            kernel,
            seed_simplices,
            &topology_model,
            &attempt1,
        )
    } else {
        repair_delaunay_with_flips_k2_k3_attempt(
            transaction.rollback_tds_mut(),
            kernel,
            seed_simplices,
            global_topology,
            &attempt1,
        )
    };

    match attempt1_result {
        Ok(outcome) => {
            if verify_repair_postcondition_with_topology(
                transaction.rollback_tds_mut(),
                kernel,
                seed_simplices,
                global_topology,
                PostconditionMode::Repair,
                outcome.last_applied_flip.as_ref(),
                ConnectivityPostcondition::Check,
            )
            .is_ok()
            {
                let run = repair_run_from_attempt(outcome);
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
            return Err(err);
        }
    }

    // Retry with LIFO + full reseed.
    run_full_reseed_retry(transaction, kernel, global_topology, &attempt2)
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
    let global_topology = GlobalTopology::DEFAULT;
    let topology_model = global_topology.model();
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
            &topology_model,
            &attempt1,
        )
    } else {
        repair_delaunay_with_flips_k2_k3_attempt_timed(
            transaction.tds_mut(),
            kernel,
            Some(seed_simplices),
            global_topology,
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
            &topology_model,
            &attempt2,
        )
    } else {
        repair_delaunay_with_flips_k2_k3_attempt_timed(
            transaction.tds_mut(),
            kernel,
            Some(seed_simplices),
            global_topology,
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

/// Crate-internal TDS verifier for the Delaunay property via local flip predicates.
///
/// Public Delaunay owners expose this as
/// [`DelaunayTriangulation::verify_via_flip_predicates`](crate::DelaunayTriangulation::verify_via_flip_predicates).
/// The caller must hold a Levels 3–4 [`Triangulation`](crate::Triangulation)
/// proof for `tds`, `kernel`, and `global_topology`; the explicit
/// `assuming_connected` name keeps that precondition visible while allowing
/// this TDS-level helper to remain independent of the stronger owner.
///
/// # Errors
///
/// Returns [`DelaunayRepairError::PostconditionFailed`] if any flip predicate detects
/// a Delaunay violation, or [`DelaunayRepairError::VerificationFailed`] if
/// verification cannot evaluate the local predicates.
pub(crate) fn verify_tds_via_flip_predicates_assuming_connected<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    global_topology: GlobalTopology<D>,
) -> Result<(), DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    // The caller's `Triangulation` already proves connectivity as part of Levels 1–4.
    // Promotion to `DelaunayTriangulation` therefore replays only the local
    // Level 5 predicates instead of treating the proof-bearing owner as raw
    // TDS input and paying for a redundant whole-complex traversal.
    verify_repair_postcondition_with_topology(
        tds,
        kernel,
        None,
        global_topology,
        PostconditionMode::Strict,
        None,
        ConnectivityPostcondition::Defer,
    )
}

/// Verifies a complete Euclidean point-set triangulation with robust local predicates.
///
/// This is the certificate predicate used by the structured Level 5 report.
/// It deliberately ignores the owner's generic kernel because the report's
/// fallback oracle is defined by the unperturbed robust empty-sphere predicate.
/// Callers must separately prove that the complex triangulates the complete
/// Euclidean point set; local predicates are not a global certificate for
/// arbitrary explicit or constrained connectivity.
///
/// # Errors
///
/// Returns any [`DelaunayRepairError`] surfaced by the topology-aware local
/// verifier, including predicate, connectivity, and postcondition failures.
pub(crate) fn verify_complete_euclidean_tds_via_robust_flip_predicates<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
) -> Result<(), DelaunayRepairError>
where
    U: DataType,
    V: DataType,
{
    verify_delaunay_with_topology(tds, &RobustKernel::new(), GlobalTopology::Euclidean)
}

/// Verify the Delaunay property via local flip predicates under a global topology model.
///
/// For periodic topologies this evaluates predicates in lifted coordinates using the
/// per-simplex periodic vertex offsets stored on quotient simplices.
pub(super) fn verify_delaunay_with_topology<K, U, V, const D: usize>(
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

/// Replays local repair postconditions without forcing the full connectivity check.
pub(super) fn verify_local_repair_postcondition<K, U, V, const D: usize>(
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
pub(super) enum PostconditionMode {
    Repair,
    Strict,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ConnectivityPostcondition {
    Check,
    Defer,
}

/// Builds a verification failure that preserves the structured flip error.
pub(super) fn verification_failed(
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
pub(super) fn verify_repair_postcondition_with_topology<K, U, V, const D: usize>(
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
pub(super) fn verify_repair_postcondition_locally<K, U, V, const D: usize>(
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
pub(super) fn resolve_postcondition_predicate_failure(
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
pub(super) fn verify_postcondition_k2_facets<K, U, V, const D: usize>(
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
pub(super) fn verify_postcondition_k3_ridges<K, U, V, const D: usize>(
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
pub(super) fn verify_postcondition_inverse_k2_edges<K, U, V, const D: usize>(
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
pub(super) fn verify_postcondition_inverse_k3_triangles<K, U, V, const D: usize>(
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
pub(super) const AMBIGUOUS_SAMPLE_LIMIT: usize = 16;
pub(super) const CYCLE_SAMPLE_LIMIT: usize = 16;
pub(super) const FLIP_SIGNATURE_WINDOW: usize = 4096;

#[cfg(test)]
mod tests {
    use super::super::test_support::init_tracing;
    use super::super::*;
    use super::*;
    use crate::DelaunayTriangulation;
    use crate::core::algorithms::incremental_insertion::repair_neighbor_pointers;
    use crate::core::collections::Uuid;
    use crate::geometry::kernel::{AdaptiveKernel, FastKernel};
    use crate::triangulation::validation::TopologyGuarantee;
    use crate::vertex;
    use slotmap::KeyData;
    use std::assert_matches;
    use std::iter::once;

    /// Verifies a deliberately raw TDS fixture through local Delaunay predicates.
    fn verify_tds_via_flip_predicates<K, U, V, const D: usize>(
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
                .insert_vertex_with_mapping(vertex!(a_coords).unwrap())
                .unwrap();
            let b = candidate
                .insert_vertex_with_mapping(vertex!(b_coords).unwrap())
                .unwrap();
            let c = candidate
                .insert_vertex_with_mapping(vertex!(c_coords).unwrap())
                .unwrap();
            let d = candidate
                .insert_vertex_with_mapping(vertex!(d_coords).unwrap())
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

            if verify_tds_via_flip_predicates(&candidate, &kernel).is_err() {
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
            GlobalTopology::DEFAULT,
            None,
        )
        .unwrap();

        assert!(stats.flips_performed > 0);
        assert!(verify_tds_via_flip_predicates(&tds, &kernel).is_ok());
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
                .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
                .unwrap();
            let b = candidate
                .insert_vertex_with_mapping(vertex!([1.0, 1.0]).unwrap())
                .unwrap();
            let c = candidate
                .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
                .unwrap();
            let d = candidate
                .insert_vertex_with_mapping(vertex!(d_coords).unwrap())
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

            if verify_tds_via_flip_predicates(&candidate, &kernel).is_err() {
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
            GlobalTopology::DEFAULT,
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
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0]).unwrap())
            .unwrap();
        let d = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0]).unwrap())
            .unwrap();
        let e = tds
            .insert_vertex_with_mapping(vertex!([0.3, 0.3, 0.3]).unwrap())
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
            verify_tds_via_flip_predicates(&tds, &kernel).is_err(),
            "3D fixture must be non-Delaunay (e inside circumsphere of {{a,b,c,d}})"
        );

        let before = snapshot_topology(&tds);
        let result = repair_delaunay_with_flips_k2_k3(
            &mut tds,
            &kernel,
            None,
            TopologyGuarantee::PLManifold,
            GlobalTopology::DEFAULT,
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
    fn test_verify_tds_via_flip_predicates_reports_non_delaunay_2d() {
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
                .insert_vertex_with_mapping(vertex!(a_coords).unwrap())
                .unwrap();
            let b = candidate
                .insert_vertex_with_mapping(vertex!(b_coords).unwrap())
                .unwrap();
            let c = candidate
                .insert_vertex_with_mapping(vertex!(c_coords).unwrap())
                .unwrap();
            let d = candidate
                .insert_vertex_with_mapping(vertex!(d_coords).unwrap())
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

            if verify_tds_via_flip_predicates(&candidate, &kernel).is_err() {
                tds = Some(candidate);
                break;
            }
        }

        let tds = tds.expect("expected a non-Delaunay configuration from candidates");
        let result = verify_tds_via_flip_predicates(&tds, &kernel);

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
            GlobalTopology::DEFAULT,
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
    fn test_repair_run_full_reseed_preserves_mutation_frontier() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
            vertex!([1.0, 0.2]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
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
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
            vertex!([1.0, 0.2]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let mut tds = dt.tds().clone();
        let before = snapshot_topology(&tds);
        let kernel = AdaptiveKernel::<f64>::new();
        let config = RepairAttemptConfig {
            attempt: 1,
            queue_order: RepairQueueOrder::Fifo,
            max_flips_override: None,
        };
        let empty_seeds: &[SimplexKey] = &[];

        let topology_model = GlobalTopology::DEFAULT.model();
        let outcome = repair_delaunay_with_flips_k2_attempt(
            &mut tds,
            &kernel,
            Some(empty_seeds),
            &topology_model,
            &config,
        )
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
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
            vertex!([1.0, 0.2]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let mut tds = dt.tds().clone();
        let kernel = AdaptiveKernel::<f64>::new();

        let seed_simplex = tds.simplex_keys().next().unwrap();
        let stats = repair_delaunay_with_flips_k2_k3(
            &mut tds,
            &kernel,
            Some(&[seed_simplex]),
            TopologyGuarantee::PLManifold,
            GlobalTopology::DEFAULT,
            None,
        )
        .unwrap();
        assert!(stats.facets_checked > 0);
        assert!(tds.is_valid().is_ok());
    }
}
