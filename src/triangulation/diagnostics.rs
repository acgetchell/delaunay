//! Construction and performance diagnostics for triangulation workflows.
//!
//! # Examples
//!
//! ```rust
//! use delaunay::prelude::triangulation::diagnostics::ConstructionTelemetry;
//!
//! let telemetry = ConstructionTelemetry::default();
//! assert!(!telemetry.has_data());
//! ```

#![forbid(unsafe_code)]

use crate::core::algorithms::flips::LocalRepairPhaseTiming;
use crate::core::operations::InsertionTelemetry;

/// Reason a batch local repair pass was scheduled.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BatchLocalRepairTrigger {
    /// The configured repair cadence fired.
    Cadence,
    /// The pending repair seed frontier exceeded the backlog threshold.
    SeedBacklog,
}

/// Aggregate release-visible telemetry collected during batch construction.
///
/// These counters summarize batch construction at a coarse level so large-scale
/// debug runs can separate construction phases, per-insertion primitive costs,
/// batch-local repair work, and global exterior conflict scans without enabling
/// per-insertion tracing.
#[derive(Debug, Default, Clone)]
#[non_exhaustive]
pub struct ConstructionTelemetry {
    /// Number of transactional insertion calls with wall-clock timing.
    pub insertion_wall_time_calls: usize,
    /// Wall-clock nanoseconds spent in transactional insertion calls.
    pub insertion_wall_time_nanos: u64,
    /// Maximum wall-clock nanoseconds spent in one transactional insertion call.
    pub insertion_wall_time_nanos_max: u64,

    /// Wall-clock nanoseconds spent preprocessing vertices before topology construction.
    pub construction_preprocessing_nanos: u64,
    /// Wall-clock nanoseconds spent in the bulk insertion loop, including batch local repair.
    pub construction_insert_loop_nanos: u64,
    /// Wall-clock nanoseconds spent finalizing bulk construction after the insertion loop.
    pub construction_finalize_nanos: u64,
    /// Wall-clock nanoseconds spent in the seeded completion repair during finalization.
    pub construction_completion_repair_nanos: u64,
    /// Wall-clock nanoseconds spent canonicalizing orientation during finalization.
    pub construction_orientation_nanos: u64,
    /// Wall-clock nanoseconds spent in final topology validation during finalization.
    pub construction_topology_validation_nanos: u64,
    /// Wall-clock nanoseconds spent in the final global Delaunay validation pass.
    pub construction_final_delaunay_validation_nanos: u64,

    /// Number of point-location calls performed during construction.
    pub locate_calls: usize,
    /// Total facet-walk steps across all point-location calls.
    pub locate_walk_steps_total: usize,
    /// Maximum facet-walk steps taken by a single point-location call.
    pub locate_walk_steps_max: usize,
    /// Number of point-location calls that used a caller-provided hint.
    pub locate_hint_uses: usize,
    /// Number of point-location calls that fell back to a brute-force scan.
    pub locate_scan_fallbacks: usize,
    /// Number of point-location calls that ended inside a cell.
    pub located_inside: usize,
    /// Number of point-location calls that ended outside the convex hull.
    pub located_outside: usize,
    /// Number of point-location calls that ended on a lower-dimensional feature.
    pub located_on_boundary: usize,

    /// Number of local conflict-region computations observed during construction.
    pub conflict_region_calls: usize,
    /// Total number of cells in local conflict regions.
    pub conflict_region_cells_total: usize,
    /// Maximum number of cells in a single local conflict region.
    pub conflict_region_cells_max: usize,
    /// Wall-clock nanoseconds spent computing local conflict regions.
    pub conflict_region_nanos: u64,
    /// Maximum wall-clock nanoseconds spent computing one local conflict region.
    pub conflict_region_nanos_max: u64,

    /// Number of cavity insertion attempts observed during construction.
    pub cavity_insertion_calls: usize,
    /// Wall-clock nanoseconds spent filling cavities and wiring neighbors.
    pub cavity_insertion_nanos: u64,
    /// Maximum wall-clock nanoseconds spent in one cavity insertion attempt.
    pub cavity_insertion_nanos_max: u64,

    /// Number of hull extension attempts observed during construction.
    pub hull_extension_calls: usize,
    /// Wall-clock nanoseconds spent extending the convex hull.
    pub hull_extension_nanos: u64,
    /// Maximum wall-clock nanoseconds spent in one hull extension attempt.
    pub hull_extension_nanos_max: u64,

    /// Number of post-insertion topology validations observed during construction.
    pub topology_validation_calls: usize,
    /// Wall-clock nanoseconds spent in post-insertion topology validation.
    pub topology_validation_nanos: u64,
    /// Maximum wall-clock nanoseconds spent in one post-insertion validation.
    pub topology_validation_nanos_max: u64,

    /// Number of batch local Delaunay repair calls during construction.
    pub local_repair_calls: usize,
    /// Wall-clock nanoseconds spent in batch local Delaunay repair.
    pub local_repair_nanos: u64,
    /// Maximum wall-clock nanoseconds spent in one batch local repair call.
    pub local_repair_nanos_max: u64,
    /// Wall-clock nanoseconds spent cloning local-repair rollback snapshots.
    pub local_repair_snapshot_nanos: u64,
    /// Maximum wall-clock nanoseconds spent cloning one local-repair rollback snapshot.
    pub local_repair_snapshot_nanos_max: u64,
    /// Wall-clock nanoseconds spent applying local-repair flip attempts.
    pub local_repair_attempt_nanos: u64,
    /// Maximum wall-clock nanoseconds spent applying flip attempts in one local repair.
    pub local_repair_attempt_nanos_max: u64,
    /// Wall-clock nanoseconds spent seeding local-repair attempt queues.
    pub local_repair_attempt_seed_nanos: u64,
    /// Maximum wall-clock nanoseconds spent seeding local-repair queues in one repair.
    pub local_repair_attempt_seed_nanos_max: u64,
    /// Wall-clock nanoseconds spent processing k=2 facet queue items.
    pub local_repair_attempt_facet_nanos: u64,
    /// Maximum wall-clock nanoseconds spent processing k=2 facets in one repair.
    pub local_repair_attempt_facet_nanos_max: u64,
    /// Wall-clock nanoseconds spent processing k=3 ridge queue items.
    pub local_repair_attempt_ridge_nanos: u64,
    /// Maximum wall-clock nanoseconds spent processing k=3 ridges in one repair.
    pub local_repair_attempt_ridge_nanos_max: u64,
    /// Wall-clock nanoseconds spent processing inverse k=2 edge queue items.
    pub local_repair_attempt_edge_nanos: u64,
    /// Maximum wall-clock nanoseconds spent processing inverse k=2 edges in one repair.
    pub local_repair_attempt_edge_nanos_max: u64,
    /// Wall-clock nanoseconds spent processing inverse k=3 triangle queue items.
    pub local_repair_attempt_triangle_nanos: u64,
    /// Maximum wall-clock nanoseconds spent processing inverse k=3 triangles in one repair.
    pub local_repair_attempt_triangle_nanos_max: u64,
    /// Wall-clock nanoseconds spent checking local-repair postconditions.
    pub local_repair_postcondition_nanos: u64,
    /// Maximum wall-clock nanoseconds spent checking postconditions in one local repair.
    pub local_repair_postcondition_nanos_max: u64,
    /// Wall-clock nanoseconds spent restoring local-repair rollback snapshots.
    pub local_repair_restore_nanos: u64,
    /// Maximum wall-clock nanoseconds spent restoring one local-repair rollback snapshot.
    pub local_repair_restore_nanos_max: u64,
    /// Total pending seed cells repaired by batch local repair calls.
    pub local_repair_seed_cells_total: usize,
    /// Maximum pending seed-cell frontier repaired by one batch local repair call.
    pub local_repair_seed_cells_max: usize,
    /// Number of batch local repair calls fired by the configured cadence.
    pub local_repair_cadence_triggers: usize,
    /// Number of batch local repair calls fired by the seed-backlog threshold.
    pub local_repair_backlog_triggers: usize,
    /// Total queued repair items checked by successful batch local repair calls.
    pub local_repair_items_checked_total: usize,
    /// Total flips performed by successful batch local repair calls.
    pub local_repair_flips_total: usize,
    /// Maximum flips performed by one successful batch local repair call.
    pub local_repair_flips_max: usize,
    /// Maximum queue length reported by one successful batch local repair call.
    pub local_repair_queue_len_max: usize,
    /// Number of successful batch local repair calls that performed no flips.
    pub local_repair_no_flip_calls: usize,

    /// Number of bulk local-repair seed accumulation calls.
    pub repair_seed_accumulation_calls: usize,
    /// Wall-clock nanoseconds spent accumulating bulk local-repair seeds.
    pub repair_seed_accumulation_nanos: u64,
    /// Maximum wall-clock nanoseconds spent in one bulk seed accumulation call.
    pub repair_seed_accumulation_nanos_max: u64,
    /// Total live seed cells added to pending bulk local-repair frontiers.
    pub repair_seed_cells_added_total: usize,
    /// Maximum live seed cells added by one bulk seed accumulation call.
    pub repair_seed_cells_added_max: usize,

    /// Number of global exterior-point conflict scans.
    pub global_conflict_scans: usize,
    /// Total cells scanned by global exterior-point conflict scans.
    pub global_conflict_cells_scanned: usize,
    /// Total cells found by global exterior-point conflict scans.
    pub global_conflict_cells_found_total: usize,
    /// Maximum cells found by a single global exterior-point conflict scan.
    pub global_conflict_cells_found_max: usize,
    /// Wall-clock nanoseconds spent in global exterior-point conflict scans.
    pub global_conflict_scan_nanos: u64,
}

impl ConstructionTelemetry {
    /// Returns true when any construction telemetry was recorded.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::diagnostics::ConstructionTelemetry;
    ///
    /// let mut telemetry = ConstructionTelemetry::default();
    /// assert!(!telemetry.has_data());
    ///
    /// telemetry.construction_insert_loop_nanos = 1;
    /// assert!(telemetry.has_data());
    /// ```
    #[must_use]
    pub const fn has_data(&self) -> bool {
        self.insertion_wall_time_calls > 0
            || self.insertion_wall_time_nanos > 0
            || self.construction_preprocessing_nanos > 0
            || self.construction_insert_loop_nanos > 0
            || self.construction_finalize_nanos > 0
            || self.construction_completion_repair_nanos > 0
            || self.construction_orientation_nanos > 0
            || self.construction_topology_validation_nanos > 0
            || self.construction_final_delaunay_validation_nanos > 0
            || self.locate_calls > 0
            || self.conflict_region_calls > 0
            || self.cavity_insertion_calls > 0
            || self.hull_extension_calls > 0
            || self.topology_validation_calls > 0
            || self.local_repair_calls > 0
            || self.local_repair_snapshot_nanos > 0
            || self.local_repair_attempt_nanos > 0
            || self.local_repair_attempt_seed_nanos > 0
            || self.local_repair_attempt_facet_nanos > 0
            || self.local_repair_attempt_ridge_nanos > 0
            || self.local_repair_attempt_edge_nanos > 0
            || self.local_repair_attempt_triangle_nanos > 0
            || self.local_repair_postcondition_nanos > 0
            || self.local_repair_restore_nanos > 0
            || self.local_repair_seed_cells_total > 0
            || self.local_repair_items_checked_total > 0
            || self.local_repair_flips_total > 0
            || self.local_repair_no_flip_calls > 0
            || self.repair_seed_accumulation_calls > 0
            || self.global_conflict_scans > 0
    }

    /// Records the wall-clock duration of one transactional insertion call.
    pub(crate) fn record_insertion_timing(&mut self, elapsed_nanos: u64) {
        self.insertion_wall_time_calls = self.insertion_wall_time_calls.saturating_add(1);
        self.insertion_wall_time_nanos =
            self.insertion_wall_time_nanos.saturating_add(elapsed_nanos);
        self.insertion_wall_time_nanos_max = self.insertion_wall_time_nanos_max.max(elapsed_nanos);
    }

    /// Records the wall-clock duration of construction preprocessing.
    pub(crate) const fn record_construction_preprocessing_timing(&mut self, elapsed_nanos: u64) {
        self.construction_preprocessing_nanos = self
            .construction_preprocessing_nanos
            .saturating_add(elapsed_nanos);
    }

    /// Records the wall-clock duration of the bulk insertion loop.
    pub(crate) const fn record_construction_insert_loop_timing(&mut self, elapsed_nanos: u64) {
        self.construction_insert_loop_nanos = self
            .construction_insert_loop_nanos
            .saturating_add(elapsed_nanos);
    }

    /// Records the wall-clock duration of bulk-construction finalization.
    pub(crate) const fn record_construction_finalize_timing(&mut self, elapsed_nanos: u64) {
        self.construction_finalize_nanos = self
            .construction_finalize_nanos
            .saturating_add(elapsed_nanos);
    }

    /// Records the wall-clock duration of seeded completion repair.
    pub(crate) const fn record_construction_completion_repair_timing(
        &mut self,
        elapsed_nanos: u64,
    ) {
        self.construction_completion_repair_nanos = self
            .construction_completion_repair_nanos
            .saturating_add(elapsed_nanos);
    }

    /// Records the wall-clock duration of orientation canonicalization.
    pub(crate) const fn record_construction_orientation_timing(&mut self, elapsed_nanos: u64) {
        self.construction_orientation_nanos = self
            .construction_orientation_nanos
            .saturating_add(elapsed_nanos);
    }

    /// Records the wall-clock duration of final topology validation.
    pub(crate) const fn record_construction_topology_validation_timing(
        &mut self,
        elapsed_nanos: u64,
    ) {
        self.construction_topology_validation_nanos = self
            .construction_topology_validation_nanos
            .saturating_add(elapsed_nanos);
    }

    /// Records the wall-clock duration of final global Delaunay validation.
    pub(crate) const fn record_construction_final_delaunay_validation_timing(
        &mut self,
        elapsed_nanos: u64,
    ) {
        self.construction_final_delaunay_validation_nanos = self
            .construction_final_delaunay_validation_nanos
            .saturating_add(elapsed_nanos);
    }

    /// Records the wall-clock duration of one batch local repair call.
    pub(crate) fn record_local_repair_timing(&mut self, elapsed_nanos: u64) {
        self.local_repair_calls = self.local_repair_calls.saturating_add(1);
        self.local_repair_nanos = self.local_repair_nanos.saturating_add(elapsed_nanos);
        self.local_repair_nanos_max = self.local_repair_nanos_max.max(elapsed_nanos);
    }

    /// Records phase timing for one batch local repair call.
    pub(crate) fn record_local_repair_phase_timing(&mut self, timing: &LocalRepairPhaseTiming) {
        self.local_repair_snapshot_nanos = self
            .local_repair_snapshot_nanos
            .saturating_add(timing.snapshot_nanos);
        self.local_repair_snapshot_nanos_max = self
            .local_repair_snapshot_nanos_max
            .max(timing.snapshot_nanos);
        self.local_repair_attempt_nanos = self
            .local_repair_attempt_nanos
            .saturating_add(timing.attempt_nanos);
        self.local_repair_attempt_nanos_max = self
            .local_repair_attempt_nanos_max
            .max(timing.attempt_nanos);
        self.local_repair_attempt_seed_nanos = self
            .local_repair_attempt_seed_nanos
            .saturating_add(timing.attempt_seed_nanos);
        self.local_repair_attempt_seed_nanos_max = self
            .local_repair_attempt_seed_nanos_max
            .max(timing.attempt_seed_nanos);
        self.local_repair_attempt_facet_nanos = self
            .local_repair_attempt_facet_nanos
            .saturating_add(timing.attempt_facet_nanos);
        self.local_repair_attempt_facet_nanos_max = self
            .local_repair_attempt_facet_nanos_max
            .max(timing.attempt_facet_nanos);
        self.local_repair_attempt_ridge_nanos = self
            .local_repair_attempt_ridge_nanos
            .saturating_add(timing.attempt_ridge_nanos);
        self.local_repair_attempt_ridge_nanos_max = self
            .local_repair_attempt_ridge_nanos_max
            .max(timing.attempt_ridge_nanos);
        self.local_repair_attempt_edge_nanos = self
            .local_repair_attempt_edge_nanos
            .saturating_add(timing.attempt_edge_nanos);
        self.local_repair_attempt_edge_nanos_max = self
            .local_repair_attempt_edge_nanos_max
            .max(timing.attempt_edge_nanos);
        self.local_repair_attempt_triangle_nanos = self
            .local_repair_attempt_triangle_nanos
            .saturating_add(timing.attempt_triangle_nanos);
        self.local_repair_attempt_triangle_nanos_max = self
            .local_repair_attempt_triangle_nanos_max
            .max(timing.attempt_triangle_nanos);
        self.local_repair_postcondition_nanos = self
            .local_repair_postcondition_nanos
            .saturating_add(timing.postcondition_nanos);
        self.local_repair_postcondition_nanos_max = self
            .local_repair_postcondition_nanos_max
            .max(timing.postcondition_nanos);
        self.local_repair_restore_nanos = self
            .local_repair_restore_nanos
            .saturating_add(timing.restore_nanos);
        self.local_repair_restore_nanos_max = self
            .local_repair_restore_nanos_max
            .max(timing.restore_nanos);
    }

    /// Records the repaired local frontier size and why the repair fired.
    pub(crate) fn record_local_repair_frontier(
        &mut self,
        seed_cells: usize,
        trigger: BatchLocalRepairTrigger,
    ) {
        self.local_repair_seed_cells_total = self
            .local_repair_seed_cells_total
            .saturating_add(seed_cells);
        self.local_repair_seed_cells_max = self.local_repair_seed_cells_max.max(seed_cells);
        match trigger {
            BatchLocalRepairTrigger::Cadence => {
                self.local_repair_cadence_triggers =
                    self.local_repair_cadence_triggers.saturating_add(1);
            }
            BatchLocalRepairTrigger::SeedBacklog => {
                self.local_repair_backlog_triggers =
                    self.local_repair_backlog_triggers.saturating_add(1);
            }
        }
    }

    /// Records aggregate work reported by one successful local repair pass.
    pub(crate) fn record_local_repair_work(
        &mut self,
        items_checked: usize,
        flips_performed: usize,
        max_queue_len: usize,
    ) {
        self.local_repair_items_checked_total = self
            .local_repair_items_checked_total
            .saturating_add(items_checked);
        self.local_repair_flips_total = self
            .local_repair_flips_total
            .saturating_add(flips_performed);
        self.local_repair_flips_max = self.local_repair_flips_max.max(flips_performed);
        self.local_repair_queue_len_max = self.local_repair_queue_len_max.max(max_queue_len);
        if flips_performed == 0 {
            self.local_repair_no_flip_calls = self.local_repair_no_flip_calls.saturating_add(1);
        }
    }

    /// Records one bulk local-repair seed accumulation step.
    pub(crate) fn record_repair_seed_accumulation(
        &mut self,
        elapsed_nanos: u64,
        cells_added: usize,
    ) {
        self.repair_seed_accumulation_calls = self.repair_seed_accumulation_calls.saturating_add(1);
        self.repair_seed_accumulation_nanos = self
            .repair_seed_accumulation_nanos
            .saturating_add(elapsed_nanos);
        self.repair_seed_accumulation_nanos_max =
            self.repair_seed_accumulation_nanos_max.max(elapsed_nanos);
        self.repair_seed_cells_added_total = self
            .repair_seed_cells_added_total
            .saturating_add(cells_added);
        self.repair_seed_cells_added_max = self.repair_seed_cells_added_max.max(cells_added);
    }

    /// Adds one insertion's telemetry into this construction summary.
    pub(crate) fn record_insertion(&mut self, telemetry: &InsertionTelemetry) {
        self.locate_calls = self.locate_calls.saturating_add(telemetry.locate_calls);
        self.locate_walk_steps_total = self
            .locate_walk_steps_total
            .saturating_add(telemetry.locate_walk_steps_total);
        self.locate_walk_steps_max = self
            .locate_walk_steps_max
            .max(telemetry.locate_walk_steps_max);
        self.locate_hint_uses = self
            .locate_hint_uses
            .saturating_add(telemetry.locate_hint_uses);
        self.locate_scan_fallbacks = self
            .locate_scan_fallbacks
            .saturating_add(telemetry.locate_scan_fallbacks);
        self.located_inside = self.located_inside.saturating_add(telemetry.located_inside);
        self.located_outside = self
            .located_outside
            .saturating_add(telemetry.located_outside);
        self.located_on_boundary = self
            .located_on_boundary
            .saturating_add(telemetry.located_on_boundary);

        self.conflict_region_calls = self
            .conflict_region_calls
            .saturating_add(telemetry.conflict_region_calls);
        self.conflict_region_cells_total = self
            .conflict_region_cells_total
            .saturating_add(telemetry.conflict_region_cells_total);
        self.conflict_region_cells_max = self
            .conflict_region_cells_max
            .max(telemetry.conflict_region_cells_max);
        self.conflict_region_nanos = self
            .conflict_region_nanos
            .saturating_add(telemetry.conflict_region_nanos);
        self.conflict_region_nanos_max = self
            .conflict_region_nanos_max
            .max(telemetry.conflict_region_nanos_max);

        self.cavity_insertion_calls = self
            .cavity_insertion_calls
            .saturating_add(telemetry.cavity_insertion_calls);
        self.cavity_insertion_nanos = self
            .cavity_insertion_nanos
            .saturating_add(telemetry.cavity_insertion_nanos);
        self.cavity_insertion_nanos_max = self
            .cavity_insertion_nanos_max
            .max(telemetry.cavity_insertion_nanos_max);

        self.hull_extension_calls = self
            .hull_extension_calls
            .saturating_add(telemetry.hull_extension_calls);
        self.hull_extension_nanos = self
            .hull_extension_nanos
            .saturating_add(telemetry.hull_extension_nanos);
        self.hull_extension_nanos_max = self
            .hull_extension_nanos_max
            .max(telemetry.hull_extension_nanos_max);

        self.topology_validation_calls = self
            .topology_validation_calls
            .saturating_add(telemetry.topology_validation_calls);
        self.topology_validation_nanos = self
            .topology_validation_nanos
            .saturating_add(telemetry.topology_validation_nanos);
        self.topology_validation_nanos_max = self
            .topology_validation_nanos_max
            .max(telemetry.topology_validation_nanos_max);

        self.global_conflict_scans = self
            .global_conflict_scans
            .saturating_add(telemetry.global_conflict_scans);
        self.global_conflict_cells_scanned = self
            .global_conflict_cells_scanned
            .saturating_add(telemetry.global_conflict_cells_scanned);
        self.global_conflict_cells_found_total = self
            .global_conflict_cells_found_total
            .saturating_add(telemetry.global_conflict_cells_found_total);
        self.global_conflict_cells_found_max = self
            .global_conflict_cells_found_max
            .max(telemetry.global_conflict_cells_found_max);
        self.global_conflict_scan_nanos = self
            .global_conflict_scan_nanos
            .saturating_add(telemetry.global_conflict_scan_nanos);
    }

    /// Merges another construction telemetry summary into this one.
    pub(crate) fn merge_from(&mut self, other: &Self) {
        self.insertion_wall_time_nanos = self
            .insertion_wall_time_nanos
            .saturating_add(other.insertion_wall_time_nanos);
        self.insertion_wall_time_calls = self
            .insertion_wall_time_calls
            .saturating_add(other.insertion_wall_time_calls);
        self.insertion_wall_time_nanos_max = self
            .insertion_wall_time_nanos_max
            .max(other.insertion_wall_time_nanos_max);

        self.merge_construction_phase_timings_from(other);

        self.locate_calls = self.locate_calls.saturating_add(other.locate_calls);
        self.locate_walk_steps_total = self
            .locate_walk_steps_total
            .saturating_add(other.locate_walk_steps_total);
        self.locate_walk_steps_max = self.locate_walk_steps_max.max(other.locate_walk_steps_max);
        self.locate_hint_uses = self.locate_hint_uses.saturating_add(other.locate_hint_uses);
        self.locate_scan_fallbacks = self
            .locate_scan_fallbacks
            .saturating_add(other.locate_scan_fallbacks);
        self.located_inside = self.located_inside.saturating_add(other.located_inside);
        self.located_outside = self.located_outside.saturating_add(other.located_outside);
        self.located_on_boundary = self
            .located_on_boundary
            .saturating_add(other.located_on_boundary);

        self.conflict_region_calls = self
            .conflict_region_calls
            .saturating_add(other.conflict_region_calls);
        self.conflict_region_cells_total = self
            .conflict_region_cells_total
            .saturating_add(other.conflict_region_cells_total);
        self.conflict_region_cells_max = self
            .conflict_region_cells_max
            .max(other.conflict_region_cells_max);
        self.conflict_region_nanos = self
            .conflict_region_nanos
            .saturating_add(other.conflict_region_nanos);
        self.conflict_region_nanos_max = self
            .conflict_region_nanos_max
            .max(other.conflict_region_nanos_max);

        self.cavity_insertion_calls = self
            .cavity_insertion_calls
            .saturating_add(other.cavity_insertion_calls);
        self.cavity_insertion_nanos = self
            .cavity_insertion_nanos
            .saturating_add(other.cavity_insertion_nanos);
        self.cavity_insertion_nanos_max = self
            .cavity_insertion_nanos_max
            .max(other.cavity_insertion_nanos_max);

        self.hull_extension_calls = self
            .hull_extension_calls
            .saturating_add(other.hull_extension_calls);
        self.hull_extension_nanos = self
            .hull_extension_nanos
            .saturating_add(other.hull_extension_nanos);
        self.hull_extension_nanos_max = self
            .hull_extension_nanos_max
            .max(other.hull_extension_nanos_max);

        self.topology_validation_calls = self
            .topology_validation_calls
            .saturating_add(other.topology_validation_calls);
        self.topology_validation_nanos = self
            .topology_validation_nanos
            .saturating_add(other.topology_validation_nanos);
        self.topology_validation_nanos_max = self
            .topology_validation_nanos_max
            .max(other.topology_validation_nanos_max);

        self.merge_local_repair_from(other);

        self.merge_repair_seed_accumulation_from(other);

        self.global_conflict_scans = self
            .global_conflict_scans
            .saturating_add(other.global_conflict_scans);
        self.global_conflict_cells_scanned = self
            .global_conflict_cells_scanned
            .saturating_add(other.global_conflict_cells_scanned);
        self.global_conflict_cells_found_total = self
            .global_conflict_cells_found_total
            .saturating_add(other.global_conflict_cells_found_total);
        self.global_conflict_cells_found_max = self
            .global_conflict_cells_found_max
            .max(other.global_conflict_cells_found_max);
        self.global_conflict_scan_nanos = self
            .global_conflict_scan_nanos
            .saturating_add(other.global_conflict_scan_nanos);
    }

    /// Keeps construction-phase merge accounting isolated so aggregate merges stay readable.
    const fn merge_construction_phase_timings_from(&mut self, other: &Self) {
        self.construction_preprocessing_nanos = self
            .construction_preprocessing_nanos
            .saturating_add(other.construction_preprocessing_nanos);
        self.construction_insert_loop_nanos = self
            .construction_insert_loop_nanos
            .saturating_add(other.construction_insert_loop_nanos);
        self.construction_finalize_nanos = self
            .construction_finalize_nanos
            .saturating_add(other.construction_finalize_nanos);
        self.construction_completion_repair_nanos = self
            .construction_completion_repair_nanos
            .saturating_add(other.construction_completion_repair_nanos);
        self.construction_orientation_nanos = self
            .construction_orientation_nanos
            .saturating_add(other.construction_orientation_nanos);
        self.construction_topology_validation_nanos = self
            .construction_topology_validation_nanos
            .saturating_add(other.construction_topology_validation_nanos);
        self.construction_final_delaunay_validation_nanos = self
            .construction_final_delaunay_validation_nanos
            .saturating_add(other.construction_final_delaunay_validation_nanos);
    }

    /// Keeps local-repair merge accounting isolated so the aggregate merge stays readable.
    fn merge_local_repair_from(&mut self, other: &Self) {
        self.local_repair_calls = self
            .local_repair_calls
            .saturating_add(other.local_repair_calls);
        self.local_repair_nanos = self
            .local_repair_nanos
            .saturating_add(other.local_repair_nanos);
        self.local_repair_nanos_max = self
            .local_repair_nanos_max
            .max(other.local_repair_nanos_max);
        self.local_repair_snapshot_nanos = self
            .local_repair_snapshot_nanos
            .saturating_add(other.local_repair_snapshot_nanos);
        self.local_repair_snapshot_nanos_max = self
            .local_repair_snapshot_nanos_max
            .max(other.local_repair_snapshot_nanos_max);
        self.local_repair_attempt_nanos = self
            .local_repair_attempt_nanos
            .saturating_add(other.local_repair_attempt_nanos);
        self.local_repair_attempt_nanos_max = self
            .local_repair_attempt_nanos_max
            .max(other.local_repair_attempt_nanos_max);
        self.local_repair_attempt_seed_nanos = self
            .local_repair_attempt_seed_nanos
            .saturating_add(other.local_repair_attempt_seed_nanos);
        self.local_repair_attempt_seed_nanos_max = self
            .local_repair_attempt_seed_nanos_max
            .max(other.local_repair_attempt_seed_nanos_max);
        self.local_repair_attempt_facet_nanos = self
            .local_repair_attempt_facet_nanos
            .saturating_add(other.local_repair_attempt_facet_nanos);
        self.local_repair_attempt_facet_nanos_max = self
            .local_repair_attempt_facet_nanos_max
            .max(other.local_repair_attempt_facet_nanos_max);
        self.local_repair_attempt_ridge_nanos = self
            .local_repair_attempt_ridge_nanos
            .saturating_add(other.local_repair_attempt_ridge_nanos);
        self.local_repair_attempt_ridge_nanos_max = self
            .local_repair_attempt_ridge_nanos_max
            .max(other.local_repair_attempt_ridge_nanos_max);
        self.local_repair_attempt_edge_nanos = self
            .local_repair_attempt_edge_nanos
            .saturating_add(other.local_repair_attempt_edge_nanos);
        self.local_repair_attempt_edge_nanos_max = self
            .local_repair_attempt_edge_nanos_max
            .max(other.local_repair_attempt_edge_nanos_max);
        self.local_repair_attempt_triangle_nanos = self
            .local_repair_attempt_triangle_nanos
            .saturating_add(other.local_repair_attempt_triangle_nanos);
        self.local_repair_attempt_triangle_nanos_max = self
            .local_repair_attempt_triangle_nanos_max
            .max(other.local_repair_attempt_triangle_nanos_max);
        self.local_repair_postcondition_nanos = self
            .local_repair_postcondition_nanos
            .saturating_add(other.local_repair_postcondition_nanos);
        self.local_repair_postcondition_nanos_max = self
            .local_repair_postcondition_nanos_max
            .max(other.local_repair_postcondition_nanos_max);
        self.local_repair_restore_nanos = self
            .local_repair_restore_nanos
            .saturating_add(other.local_repair_restore_nanos);
        self.local_repair_restore_nanos_max = self
            .local_repair_restore_nanos_max
            .max(other.local_repair_restore_nanos_max);
        self.local_repair_seed_cells_total = self
            .local_repair_seed_cells_total
            .saturating_add(other.local_repair_seed_cells_total);
        self.local_repair_seed_cells_max = self
            .local_repair_seed_cells_max
            .max(other.local_repair_seed_cells_max);
        self.local_repair_cadence_triggers = self
            .local_repair_cadence_triggers
            .saturating_add(other.local_repair_cadence_triggers);
        self.local_repair_backlog_triggers = self
            .local_repair_backlog_triggers
            .saturating_add(other.local_repair_backlog_triggers);
        self.local_repair_items_checked_total = self
            .local_repair_items_checked_total
            .saturating_add(other.local_repair_items_checked_total);
        self.local_repair_flips_total = self
            .local_repair_flips_total
            .saturating_add(other.local_repair_flips_total);
        self.local_repair_flips_max = self
            .local_repair_flips_max
            .max(other.local_repair_flips_max);
        self.local_repair_queue_len_max = self
            .local_repair_queue_len_max
            .max(other.local_repair_queue_len_max);
        self.local_repair_no_flip_calls = self
            .local_repair_no_flip_calls
            .saturating_add(other.local_repair_no_flip_calls);
    }

    /// Keeps seed-accumulation merge accounting isolated from the aggregate merge body.
    fn merge_repair_seed_accumulation_from(&mut self, other: &Self) {
        self.repair_seed_accumulation_calls = self
            .repair_seed_accumulation_calls
            .saturating_add(other.repair_seed_accumulation_calls);
        self.repair_seed_accumulation_nanos = self
            .repair_seed_accumulation_nanos
            .saturating_add(other.repair_seed_accumulation_nanos);
        self.repair_seed_accumulation_nanos_max = self
            .repair_seed_accumulation_nanos_max
            .max(other.repair_seed_accumulation_nanos_max);
        self.repair_seed_cells_added_total = self
            .repair_seed_cells_added_total
            .saturating_add(other.repair_seed_cells_added_total);
        self.repair_seed_cells_added_max = self
            .repair_seed_cells_added_max
            .max(other.repair_seed_cells_added_max);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "single-field telemetry regression covers every aggregate counter"
    )]
    fn test_construction_telemetry_records_all_counters() {
        let mut summary = ConstructionTelemetry::default();
        let telemetry = InsertionTelemetry {
            locate_calls: 2,
            locate_walk_steps_total: 9,
            locate_walk_steps_max: 7,
            locate_hint_uses: 1,
            locate_scan_fallbacks: 1,
            located_inside: 1,
            located_outside: 1,
            conflict_region_calls: 1,
            conflict_region_cells_total: 4,
            conflict_region_cells_max: 4,
            conflict_region_nanos: 125_000,
            conflict_region_nanos_max: 125_000,
            cavity_insertion_calls: 1,
            cavity_insertion_nanos: 375_000,
            cavity_insertion_nanos_max: 375_000,
            hull_extension_calls: 1,
            hull_extension_nanos: 500_000,
            hull_extension_nanos_max: 500_000,
            topology_validation_calls: 1,
            topology_validation_nanos: 625_000,
            topology_validation_nanos_max: 625_000,
            global_conflict_scans: 1,
            global_conflict_cells_scanned: 12,
            global_conflict_cells_found_total: 3,
            global_conflict_cells_found_max: 3,
            global_conflict_scan_nanos: 250_000,
            ..InsertionTelemetry::default()
        };

        summary.record_insertion(&telemetry);
        summary.record_insertion_timing(1_000_000);
        summary.record_local_repair_timing(2_000_000);
        summary.record_local_repair_phase_timing(&LocalRepairPhaseTiming {
            snapshot_nanos: 100_000,
            attempt_nanos: 1_250_000,
            attempt_seed_nanos: 10_000,
            attempt_facet_nanos: 750_000,
            attempt_ridge_nanos: 450_000,
            attempt_edge_nanos: 25_000,
            attempt_triangle_nanos: 15_000,
            postcondition_nanos: 500_000,
            restore_nanos: 25_000,
        });
        summary.record_local_repair_frontier(11, BatchLocalRepairTrigger::SeedBacklog);
        summary.record_local_repair_work(123, 5, 17);
        summary.record_repair_seed_accumulation(500_000, 7);
        summary.record_construction_preprocessing_timing(10_000);
        summary.record_construction_insert_loop_timing(20_000);
        summary.record_construction_finalize_timing(30_000);
        summary.record_construction_completion_repair_timing(40_000);
        summary.record_construction_orientation_timing(50_000);
        summary.record_construction_topology_validation_timing(60_000);
        summary.record_construction_final_delaunay_validation_timing(70_000);

        assert!(summary.has_data());
        assert_eq!(summary.insertion_wall_time_calls, 1);
        assert_eq!(summary.insertion_wall_time_nanos, 1_000_000);
        assert_eq!(summary.insertion_wall_time_nanos_max, 1_000_000);
        assert_eq!(summary.construction_preprocessing_nanos, 10_000);
        assert_eq!(summary.construction_insert_loop_nanos, 20_000);
        assert_eq!(summary.construction_finalize_nanos, 30_000);
        assert_eq!(summary.construction_completion_repair_nanos, 40_000);
        assert_eq!(summary.construction_orientation_nanos, 50_000);
        assert_eq!(summary.construction_topology_validation_nanos, 60_000);
        assert_eq!(summary.construction_final_delaunay_validation_nanos, 70_000);
        assert_eq!(summary.locate_calls, 2);
        assert_eq!(summary.locate_walk_steps_total, 9);
        assert_eq!(summary.locate_walk_steps_max, 7);
        assert_eq!(summary.locate_hint_uses, 1);
        assert_eq!(summary.locate_scan_fallbacks, 1);
        assert_eq!(summary.located_inside, 1);
        assert_eq!(summary.located_outside, 1);
        assert_eq!(summary.conflict_region_calls, 1);
        assert_eq!(summary.conflict_region_cells_total, 4);
        assert_eq!(summary.conflict_region_nanos, 125_000);
        assert_eq!(summary.conflict_region_nanos_max, 125_000);
        assert_eq!(summary.cavity_insertion_calls, 1);
        assert_eq!(summary.cavity_insertion_nanos, 375_000);
        assert_eq!(summary.hull_extension_calls, 1);
        assert_eq!(summary.hull_extension_nanos, 500_000);
        assert_eq!(summary.topology_validation_calls, 1);
        assert_eq!(summary.topology_validation_nanos, 625_000);
        assert_eq!(summary.local_repair_calls, 1);
        assert_eq!(summary.local_repair_nanos, 2_000_000);
        assert_eq!(summary.local_repair_snapshot_nanos, 100_000);
        assert_eq!(summary.local_repair_snapshot_nanos_max, 100_000);
        assert_eq!(summary.local_repair_attempt_nanos, 1_250_000);
        assert_eq!(summary.local_repair_attempt_nanos_max, 1_250_000);
        assert_eq!(summary.local_repair_attempt_seed_nanos, 10_000);
        assert_eq!(summary.local_repair_attempt_seed_nanos_max, 10_000);
        assert_eq!(summary.local_repair_attempt_facet_nanos, 750_000);
        assert_eq!(summary.local_repair_attempt_facet_nanos_max, 750_000);
        assert_eq!(summary.local_repair_attempt_ridge_nanos, 450_000);
        assert_eq!(summary.local_repair_attempt_ridge_nanos_max, 450_000);
        assert_eq!(summary.local_repair_attempt_edge_nanos, 25_000);
        assert_eq!(summary.local_repair_attempt_edge_nanos_max, 25_000);
        assert_eq!(summary.local_repair_attempt_triangle_nanos, 15_000);
        assert_eq!(summary.local_repair_attempt_triangle_nanos_max, 15_000);
        assert_eq!(summary.local_repair_postcondition_nanos, 500_000);
        assert_eq!(summary.local_repair_postcondition_nanos_max, 500_000);
        assert_eq!(summary.local_repair_restore_nanos, 25_000);
        assert_eq!(summary.local_repair_restore_nanos_max, 25_000);
        assert_eq!(summary.local_repair_seed_cells_total, 11);
        assert_eq!(summary.local_repair_seed_cells_max, 11);
        assert_eq!(summary.local_repair_cadence_triggers, 0);
        assert_eq!(summary.local_repair_backlog_triggers, 1);
        assert_eq!(summary.local_repair_items_checked_total, 123);
        assert_eq!(summary.local_repair_flips_total, 5);
        assert_eq!(summary.local_repair_flips_max, 5);
        assert_eq!(summary.local_repair_queue_len_max, 17);
        assert_eq!(summary.local_repair_no_flip_calls, 0);
        assert_eq!(summary.repair_seed_accumulation_calls, 1);
        assert_eq!(summary.repair_seed_accumulation_nanos, 500_000);
        assert_eq!(summary.repair_seed_cells_added_total, 7);
        assert_eq!(summary.repair_seed_cells_added_max, 7);
        assert_eq!(summary.global_conflict_scans, 1);
        assert_eq!(summary.global_conflict_cells_scanned, 12);
        assert_eq!(summary.global_conflict_cells_found_total, 3);
        assert_eq!(summary.global_conflict_scan_nanos, 250_000);
    }

    #[test]
    fn test_construction_telemetry_merge_preserves_local_repair_frontiers() {
        let mut left = ConstructionTelemetry::default();
        left.record_local_repair_timing(10);
        left.record_local_repair_phase_timing(&LocalRepairPhaseTiming {
            snapshot_nanos: 1,
            attempt_nanos: 2,
            attempt_seed_nanos: 3,
            attempt_facet_nanos: 4,
            attempt_ridge_nanos: 5,
            attempt_edge_nanos: 6,
            attempt_triangle_nanos: 7,
            postcondition_nanos: 3,
            restore_nanos: 4,
        });
        left.record_local_repair_frontier(5, BatchLocalRepairTrigger::Cadence);
        left.record_local_repair_work(10, 0, 5);
        left.record_construction_insert_loop_timing(100);
        left.record_construction_final_delaunay_validation_timing(200);

        let mut right = ConstructionTelemetry::default();
        right.record_local_repair_timing(30);
        right.record_local_repair_phase_timing(&LocalRepairPhaseTiming {
            snapshot_nanos: 10,
            attempt_nanos: 20,
            attempt_seed_nanos: 30,
            attempt_facet_nanos: 40,
            attempt_ridge_nanos: 50,
            attempt_edge_nanos: 60,
            attempt_triangle_nanos: 70,
            postcondition_nanos: 30,
            restore_nanos: 40,
        });
        right.record_local_repair_frontier(11, BatchLocalRepairTrigger::SeedBacklog);
        right.record_local_repair_work(30, 4, 12);
        right.record_construction_insert_loop_timing(300);
        right.record_construction_final_delaunay_validation_timing(400);

        left.merge_from(&right);

        assert!(left.has_data());
        assert_eq!(left.construction_insert_loop_nanos, 400);
        assert_eq!(left.construction_final_delaunay_validation_nanos, 600);
        assert_eq!(left.local_repair_calls, 2);
        assert_eq!(left.local_repair_nanos, 40);
        assert_eq!(left.local_repair_nanos_max, 30);
        assert_eq!(left.local_repair_snapshot_nanos, 11);
        assert_eq!(left.local_repair_snapshot_nanos_max, 10);
        assert_eq!(left.local_repair_attempt_nanos, 22);
        assert_eq!(left.local_repair_attempt_nanos_max, 20);
        assert_eq!(left.local_repair_attempt_seed_nanos, 33);
        assert_eq!(left.local_repair_attempt_seed_nanos_max, 30);
        assert_eq!(left.local_repair_attempt_facet_nanos, 44);
        assert_eq!(left.local_repair_attempt_facet_nanos_max, 40);
        assert_eq!(left.local_repair_attempt_ridge_nanos, 55);
        assert_eq!(left.local_repair_attempt_ridge_nanos_max, 50);
        assert_eq!(left.local_repair_attempt_edge_nanos, 66);
        assert_eq!(left.local_repair_attempt_edge_nanos_max, 60);
        assert_eq!(left.local_repair_attempt_triangle_nanos, 77);
        assert_eq!(left.local_repair_attempt_triangle_nanos_max, 70);
        assert_eq!(left.local_repair_postcondition_nanos, 33);
        assert_eq!(left.local_repair_postcondition_nanos_max, 30);
        assert_eq!(left.local_repair_restore_nanos, 44);
        assert_eq!(left.local_repair_restore_nanos_max, 40);
        assert_eq!(left.local_repair_seed_cells_total, 16);
        assert_eq!(left.local_repair_seed_cells_max, 11);
        assert_eq!(left.local_repair_cadence_triggers, 1);
        assert_eq!(left.local_repair_backlog_triggers, 1);
        assert_eq!(left.local_repair_items_checked_total, 40);
        assert_eq!(left.local_repair_flips_total, 4);
        assert_eq!(left.local_repair_flips_max, 4);
        assert_eq!(left.local_repair_queue_len_max, 12);
        assert_eq!(left.local_repair_no_flip_calls, 1);
    }
}
