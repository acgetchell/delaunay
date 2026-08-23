//! Delaunay-repair diagnostics, work queues, and individual repair steps.

#![forbid(unsafe_code)]

use super::{
    AMBIGUOUS_SAMPLE_LIMIT, AllFacetsIter, AppliedFlip, BistellarFlipKind, CYCLE_SAMPLE_LIMIT,
    DataType, DelaunayRepairDiagnostics, DelaunayRepairError, DelaunayRepairStats, Duration,
    EdgeKey, FLIP_SIGNATURE_WINDOW, FacetHandle, FastHashMap, FastHashSet, FastHasher,
    FlipCycleContext, FlipDirection, FlipError, GlobalTopologyModelAdapter, Hash, Hasher, Kernel,
    MAX_PRACTICAL_DIMENSION_SIZE, RemovedSimplexVertexSnapshot, RepairQueueOrder, RidgeHandle,
    SimplexKey, SimplexKeyBuffer, SmallBuffer, Tds, TriangleHandle, VecDeque, VertexKey,
    VertexKeyList, apply_delaunay_flip_dynamic, apply_delaunay_flip_k2, apply_delaunay_flip_k3,
    build_k2_flip_context, build_k2_flip_context_from_edge, build_k3_flip_context_for_repair,
    build_k3_flip_context_from_triangle, check_flip_cycle, debug_ridge_context,
    delaunay_violation_k2_for_facet, delaunay_violation_k3_for_ridge, env, facet_key_from_vertices,
    facet_vertices_from_simplex, fmt, is_delaunay_violation_k2, is_delaunay_violation_k3,
    record_touched_simplices, removed_simplex_frame, resolve_facet_handle_for_key,
    resolve_ridge_handle_for_key, ridge_vertices_from_simplex,
};

// Allow extended repeats to capture diagnostics in long-running repairs.  A threshold of
// 32 caused false non-convergence on valid 3D inputs (see #306); 128 still bounds
// pathological cases while giving legitimate repair sequences room to converge.
pub(super) const MAX_REPEAT_SIGNATURE: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct FlipSignature(pub(super) u64);

impl fmt::Display for FlipSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Default)]
pub(super) struct RepairDiagnostics {
    pub(super) ambiguous_predicates: usize,
    pub(super) ambiguous_samples: Vec<u64>,
    pub(super) predicate_failures: usize,
    pub(super) cycle_detections: usize,
    pub(super) cycle_samples: Vec<FlipSignature>,
    pub(super) inserted_simplex_skips: usize,
    pub(super) inserted_simplex_sample: Option<InsertedSimplexSkipSample>,
    pub(super) invalid_ridge_multiplicity_skips: usize,
    pub(super) invalid_ridge_multiplicity_sample: Option<RidgeMultiplicitySkipSample>,
    pub(super) missing_simplex_skips: usize,
    pub(super) missing_simplex_sample: Option<MissingSimplexSkipSample>,
    pub(super) saw_applicable_repair_site: bool,
    pub(super) flip_signature_window: VecDeque<FlipSignature>,
    pub(super) flip_signature_counts: FastHashMap<FlipSignature, usize>,
    pub(super) ridge_debug_emitted: usize,
    pub(super) postcondition_facet_debug_emitted: usize,
}

#[derive(Clone, PartialEq, Eq)]
pub(super) struct InsertedSimplexSkipSample {
    pub(super) location: RepairSkipLocation,
    pub(super) removed_face: VertexKeyList,
    pub(super) inserted_face: VertexKeyList,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct RidgeMultiplicitySkipSample {
    pub(super) ridge: RidgeHandle,
    pub(super) multiplicity: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct MissingSimplexSkipSample {
    pub(super) location: RepairSkipLocation,
    pub(super) simplex_key: SimplexKey,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum RepairSkipLocation {
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

pub(super) fn vertex_key_list(vertices: &[VertexKey]) -> VertexKeyList {
    vertices.iter().copied().collect()
}

impl RepairDiagnostics {
    /// Records uncertain predicate sites with bounded samples so diagnostics stay
    /// actionable on large repairs.
    pub(super) fn record_ambiguous(&mut self, key: u64) {
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
    pub(super) const fn record_predicate_failure(&mut self) {
        self.predicate_failures = self.predicate_failures.saturating_add(1);
    }

    /// Maintains a sliding signature window so cycle detection is bounded in
    /// memory but still catches local oscillations.
    pub(super) fn record_flip_signature(&mut self, signature: FlipSignature) {
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
    pub(super) fn record_cycle_abort(&mut self, signature: FlipSignature) {
        self.cycle_detections = self.cycle_detections.saturating_add(1);
        if self.cycle_samples.len() < CYCLE_SAMPLE_LIMIT && !self.cycle_samples.contains(&signature)
        {
            self.cycle_samples.push(signature);
        }
    }

    /// Captures one duplicate-simplex skip sample with typed context.
    pub(super) fn record_inserted_simplex_skip(&mut self, sample: InsertedSimplexSkipSample) {
        self.inserted_simplex_skips = self.inserted_simplex_skips.saturating_add(1);
        if self.inserted_simplex_sample.is_none() {
            self.inserted_simplex_sample = Some(sample);
        }
    }

    /// Captures one invalid-ridge sample with typed context.
    pub(super) const fn record_invalid_ridge_multiplicity_skip(
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
    pub(super) const fn record_missing_simplex_skip(&mut self, sample: MissingSimplexSkipSample) {
        self.missing_simplex_skips = self.missing_simplex_skips.saturating_add(1);
        if self.missing_simplex_sample.is_none() {
            self.missing_simplex_sample = Some(sample);
        }
    }

    /// Marks that a repair predicate found an applicable flip even if no mutation followed.
    pub(super) const fn record_applicable_repair_site(&mut self) {
        self.saw_applicable_repair_site = true;
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct RepairAttemptConfig {
    pub(super) attempt: usize,
    pub(super) queue_order: RepairQueueOrder,
    /// Override the flip budget. `None` uses `default_max_flips` (proportional to total simplex
    /// count). Set to `Some(n)` for per-insertion local repairs to avoid a runaway budget when
    /// the triangulation is large but the seed set is small.
    pub(super) max_flips_override: Option<usize>,
}

/// Builds the public non-convergence error in one place so diagnostics, queue
/// order, and attempt metadata stay consistent.
pub(super) fn non_convergent_error(
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
pub(super) fn duration_nanos_saturating(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

/// Gates the expensive repair summary behind an environment variable while
/// keeping all attempts logged in a uniform shape.
pub(super) fn emit_repair_debug_summary(
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
pub(super) fn pop_queue<T>(queue: &mut VecDeque<T>, order: RepairQueueOrder) -> Option<T> {
    match order {
        RepairQueueOrder::Fifo => queue.pop_front(),
        RepairQueueOrder::Lifo => queue.pop_back(),
    }
}

/// Hashes a predicate site canonically so ambiguous-predicate samples are stable
/// across vertex ordering in a simplex.
pub(super) fn predicate_key_from_vertices(
    simplex_vertices: &[VertexKey],
    test_vertex: VertexKey,
) -> u64 {
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
pub(super) fn flip_signature(
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
pub(super) struct LastAppliedFlip {
    pub(super) kind: BistellarFlipKind,
    pub(super) removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    pub(super) inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    pub(super) removed_simplices: SimplexKeyBuffer,
    pub(super) new_simplices: SimplexKeyBuffer,
    /// Snapshot of each removed simplex's vertex list captured before the flip's
    /// `remove_simplices_by_keys` call; pairs 1:1 with `removed_simplices`. Empty
    /// inner buffers only appear in placeholder instances built from validated
    /// flip faces.
    pub(super) removed_simplex_vertices: RemovedSimplexVertexSnapshot,
}

impl LastAppliedFlip {
    /// Sorts faces so immediate-reversal detection is independent of local simplex
    /// vertex order. Simplex lists stay empty here because this constructor is also
    /// used for temporary reversal checks.
    pub(super) fn from_validated_flip_faces(
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
    pub(super) fn from_applied_flip<const D: usize>(applied: &AppliedFlip<D>) -> Self {
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
    pub(super) fn removed_simplex_vertex_lines(&self) -> Vec<String> {
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
pub(super) fn would_immediately_reverse_last_flip<const D: usize>(
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
pub(super) fn repair_trace_enabled() -> bool {
    env::var_os("DELAUNAY_REPAIR_TRACE").is_some()
}

/// Treats full repair tracing as enabling ridge snapshots so one debug switch
/// gives enough topology context.
#[inline]
pub(super) fn repair_ridge_debug_enabled() -> bool {
    env::var_os("DELAUNAY_REPAIR_DEBUG_RIDGE").is_some() || repair_trace_enabled()
}

pub(super) const RIDGE_DEBUG_LIMIT_DEFAULT: usize = 64;
pub(super) const RIDGE_DEBUG_MIN_MULTIPLICITY_DEFAULT: usize = 0;

/// Rate-limits ridge snapshots to keep pathological repair runs from flooding
/// logs.
pub(super) fn ridge_debug_limit() -> usize {
    env::var("DELAUNAY_REPAIR_DEBUG_RIDGE_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(RIDGE_DEBUG_LIMIT_DEFAULT)
}

/// Lets callers skip the common multiplicity-1/2 boundary cases and capture
/// the first genuinely overshared ridge instead.
pub(super) fn ridge_debug_min_multiplicity() -> usize {
    env::var("DELAUNAY_REPAIR_DEBUG_RIDGE_MIN_MULTIPLICITY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(RIDGE_DEBUG_MIN_MULTIPLICITY_DEFAULT)
}

/// Applies the ridge debug limit per repair attempt so independent repairs do
/// not consume each other's diagnostic budget.
pub(super) fn should_emit_ridge_debug(
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
pub(super) fn postcondition_facet_debug_enabled() -> bool {
    env::var_os("DELAUNAY_REPAIR_DEBUG_POSTCONDITION_FACET").is_some()
}

/// Emits at most one postcondition facet snapshot per repair attempt so the
/// focused #204 debug path stays readable.
pub(super) fn should_emit_postcondition_facet_debug(diagnostics: &mut RepairDiagnostics) -> bool {
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
pub(super) fn default_max_flips<const D: usize>(simplex_count: usize) -> usize {
    // Flip budget strategy by dimension and build mode:
    //
    // - D<=2: use 4× budget in debug/test (2D flips are fast).
    // - D=3: use 8× budget in debug/test.  Previously 16× but that caused the global repair
    //   to spend hours cycling through flip loops when many star-splits produced a heavily
    //   non-Delaunay triangulation.  8× still provides headroom for legitimate convergence
    //   while failing faster (triggering the heuristic rebuild sooner) when cycling.
    // - D>=4: use simplices×(D+1)×4 (min 4096) in debug/test.  Flip convergence is not
    //   guaranteed in D>=4 (Edelsbrunner-Shah 1996), so this budget is intentionally
    //   conservative: it bounds the cost of consuming Delaunay conversion and
    //   run_flip_repair_fallbacks during incremental insertion while failing fast
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

pub(super) struct RepairQueues {
    pub(super) facet_queue: VecDeque<(FacetHandle, u64)>,
    facet_queued: FastHashSet<u64>,
    facet_handles: FastHashMap<u64, FacetHandle>,
    pub(super) ridge_queue: VecDeque<(RidgeHandle, u64)>,
    ridge_queued: FastHashSet<u64>,
    ridge_handles: FastHashMap<u64, RidgeHandle>,
    pub(super) edge_queue: VecDeque<(EdgeKey, u64)>,
    edge_queued: FastHashSet<u64>,
    pub(super) triangle_queue: VecDeque<(TriangleHandle, u64)>,
    triangle_queued: FastHashSet<u64>,
}

impl RepairQueues {
    /// Initializes all repair worklists together so queue state cannot be
    /// partially seeded.
    pub(super) fn new() -> Self {
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
    pub(super) fn has_work(&self) -> bool {
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
pub(super) fn seed_repair_queues<U, V, const D: usize>(
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
pub(super) fn enqueue_new_simplices_for_repair<U, V, const D: usize>(
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
pub(super) fn run_next_ridge_repair_step<K, U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    kernel: &K,
    topology_model: &GlobalTopologyModelAdapter<D>,
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

    let violates = match is_delaunay_violation_k3(
        tds,
        kernel,
        topology_model,
        &context,
        config,
        diagnostics,
    ) {
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
        BistellarFlipKind::from_validated(3, D),
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

    let kind = BistellarFlipKind::from_validated(3, D);
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
    let applied = match apply_delaunay_flip_k3(tds, &context) {
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
            | FlipError::SimplexCreation { source: _ }),
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
pub(super) fn run_next_edge_repair_step<K, U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    kernel: &K,
    topology_model: &GlobalTopologyModelAdapter<D>,
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
        BistellarFlipKind::from_validated(2, D).inverse(),
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
    let kind = BistellarFlipKind::from_validated(2, D).inverse();
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
    let applied = match apply_delaunay_flip_dynamic(tds, kind.k(), &context) {
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
            | FlipError::SimplexCreation { source: _ }),
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
pub(super) fn run_next_triangle_repair_step<K, U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    kernel: &K,
    topology_model: &GlobalTopologyModelAdapter<D>,
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
        topology_model,
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
        BistellarFlipKind::from_validated(3, D).inverse(),
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
    let kind = BistellarFlipKind::from_validated(3, D).inverse();
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
    let applied = match apply_delaunay_flip_dynamic(tds, kind.k(), &context) {
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
            | FlipError::SimplexCreation { source: _ }),
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
pub(super) fn run_next_facet_repair_step<K, U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    kernel: &K,
    topology_model: &GlobalTopologyModelAdapter<D>,
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

    let violates = match is_delaunay_violation_k2(
        tds,
        kernel,
        topology_model,
        &context,
        config,
        diagnostics,
    ) {
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
        BistellarFlipKind::from_validated(2, D),
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
    let applied = match apply_delaunay_flip_k2(tds, &context) {
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
            | FlipError::SimplexCreation { source: _ }),
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
/// Queues all interior facets of a simplex because k=2 repair is driven by shared
/// facet predicates.
pub(super) fn enqueue_simplex_facets<U, V, const D: usize>(
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
pub(super) fn enqueue_facet<U, V, const D: usize>(
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
pub(super) fn enqueue_simplex_edges<U, V, const D: usize>(
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
pub(super) fn enqueue_edge(
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
pub(super) fn enqueue_simplex_triangles<U, V, const D: usize>(
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
pub(super) fn enqueue_triangle(
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
pub(super) fn enqueue_simplex_ridges<U, V, const D: usize>(
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
pub(super) fn enqueue_ridge<U, V, const D: usize>(
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
    use super::super::test_support::init_tracing;
    use super::super::*;
    use super::*;
    use crate::core::algorithms::insertion::repair_neighbor_pointers;
    use crate::geometry::kernel::AdaptiveKernel;
    use crate::triangulation::validation::TopologyGuarantee;
    use crate::vertex;
    use slotmap::KeyData;
    use std::iter::once;

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
            inserted_face: once(v2).collect(),
        };
        diagnostics.record_inserted_simplex_skip(first_inserted_sample.clone());
        assert_eq!(diagnostics.inserted_simplex_skips, 1);
        assert_eq!(
            diagnostics.inserted_simplex_sample,
            Some(first_inserted_sample.clone())
        );

        diagnostics.record_inserted_simplex_skip(InsertedSimplexSkipSample {
            location: RepairSkipLocation::Edge(edge),
            removed_face: once(v1).collect(),
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
        let inserted_face: VertexKeyList = once(v2).collect();
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
    fn unit_vector<const D: usize>(index: usize) -> [f64; D] {
        let mut coords = [0.0; D];
        coords[index] = 1.0;
        coords
    }

    #[test]
    fn test_repair_queue_inverse_k2_smoke_4d() {
        init_tracing();
        let mut tds: Tds<(), (), 4> = Tds::empty();
        let mut shared_vertices = Vec::with_capacity(4);
        for i in 0..4 {
            let v = tds
                .insert_vertex_with_mapping(vertex!(unit_vector::<4>(i)).unwrap())
                .unwrap();
            shared_vertices.push(v);
        }

        let opposite_a = tds
            .insert_vertex_with_mapping(vertex!([0.0; 4]).unwrap())
            .unwrap();
        let opposite_b = tds
            .insert_vertex_with_mapping(vertex!([1.1; 4]).unwrap())
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
        let info = apply_bistellar_flip_raw(&mut tds, &context).unwrap();

        let kernel = AdaptiveKernel::<f64>::new();
        let seed_simplices: SimplexKeyBuffer = info.new_simplices.iter().copied().collect();
        let stats = repair_delaunay_with_flips_k2_k3(
            &mut tds,
            &kernel,
            Some(seed_simplices.as_slice()),
            TopologyGuarantee::PLManifold,
            GlobalTopology::DEFAULT,
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
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let r1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let r2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let r3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0, 1.0, 0.0]).unwrap())
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0, 0.0, 1.0]).unwrap())
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(vertex!([0.2, 0.2, 0.2, 0.2, 0.5]).unwrap())
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
        let info = apply_bistellar_flip_raw(&mut tds, &context).unwrap();

        let kernel = AdaptiveKernel::<f64>::new();
        let seed_simplices: SimplexKeyBuffer = info.new_simplices.iter().copied().collect();
        let result = repair_delaunay_with_flips_k2_k3(
            &mut tds,
            &kernel,
            Some(seed_simplices.as_slice()),
            TopologyGuarantee::PLManifold,
            GlobalTopology::DEFAULT,
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
}
