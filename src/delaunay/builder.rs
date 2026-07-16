//! Fluent builder for [`DelaunayTriangulation`] with optional toroidal topology
//! and typed simplex storage.
//!
//! [`DelaunayTriangulationBuilder`] unifies the existing family of `DelaunayTriangulation`
//! constructors under a single, composable API and adds first-class support for
//! periodic toroidal construction.
//!
//! # When to use the builder
//!
//! | Situation | Recommended API |
//! |---|---|
//! | Simple Euclidean, default options | [`DelaunayTriangulationBuilder::new`] |
//! | Custom `ConstructionOptions` or `TopologyGuarantee` | [`DelaunayTriangulationBuilder`] |
//! | Periodic toroidal quotient (χ = 0) | [`DelaunayTriangulationBuilder`] with [`.try_toroidal()`](DelaunayTriangulationBuilder::try_toroidal) |
//! | Euler/topology metadata expectation only | [`DelaunayTriangulationBuilder`] with [`.global_topology(...)`](DelaunayTriangulationBuilder::global_topology) |
//! | Persisted simplex payloads | [`DelaunayTriangulationBuilder`] with [`.simplex_data_type::<V>()`](DelaunayTriangulationBuilder::simplex_data_type), then [`DelaunayTriangulation::fill_simplex_data`] |
//! | Custom kernel (`RobustKernel`, etc.) | [`DelaunayTriangulationBuilder::build_with_kernel`] |
//!
//! # Periodic toroidal construction
//!
//! **Image-point construction (`.try_toroidal()`, issue #210):** Periodic construction using
//! the 3^D image-point method — generating copies of each point shifted by ±L in each
//! dimension, building the full Euclidean DT on the expanded set, normalizing lifted
//! simplices, searching a closed quotient candidate subset, and rebuilding quotient
//! representatives with periodic neighbor pointers. The `T^2` and compact `T^3` paths
//! are release-validated as true toroidal (χ = 0) triangulations. `T^4`/`T^5` periodic
//! quotients fail fast until issue #416 makes quotient selection scalable enough
//! for routine validation. See `REFERENCES.md`, "Periodic and Toroidal
//! Triangulations", first entry.
//!
//! # Examples
//!
//! ## Standard Euclidean construction
//!
//! ```rust
//! use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulationBuilder};
//!
//! # fn main() -> DelaunayResult<()> {
//! let vertices = vec![
//!     delaunay::vertex![0.0, 0.0]?,
//!     delaunay::vertex![1.0, 0.0]?,
//!     delaunay::vertex![0.0, 1.0]?,
//! ];
//!
//! let dt = DelaunayTriangulationBuilder::new(&vertices)
//!     .build()?;
//!
//! assert_eq!(dt.number_of_vertices(), 3);
//! # Ok(())
//! # }
//! ```
//!
//! ## Toroidal construction
//!
//! Uses the 3^D image-point method to produce a true toroidal (χ = 0) triangulation
//! where boundary facets are identified and neighbor pointers are rewired periodically.
//!
//! ```rust,no_run
//! use delaunay::prelude::geometry::RobustKernel;
//! use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulationBuilder};
//!
//! # fn main() -> DelaunayResult<()> {
//! let vertices = vec![
//!     delaunay::vertex![0.1, 0.2]?,
//!     delaunay::vertex![0.4, 0.7]?,
//!     delaunay::vertex![0.7, 0.3]?,
//!     delaunay::vertex![0.2, 0.9]?,
//!     delaunay::vertex![0.8, 0.6]?,
//!     delaunay::vertex![0.5, 0.1]?,
//!     delaunay::vertex![0.3, 0.5]?,
//! ];
//!
//! let kernel = RobustKernel::new();
//! let dt = DelaunayTriangulationBuilder::new(&vertices)
//!     .try_toroidal([1.0, 1.0])
//!     ?
//!     .build_with_kernel(&kernel)?;
//!
//! assert_eq!(dt.number_of_vertices(), 7);
//! dt.validate()?;
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]

use crate::construction::{
    ConstructionOptions, ConstructionStatistics, DelaunayConstructionFailure,
    DelaunayTriangulationConstructionError, DelaunayTriangulationConstructionErrorWithStatistics,
    InitialSimplexStrategy, RetryPolicy, duration_nanos_saturating,
};
use crate::core::algorithms::incremental_insertion::InsertionError;
use crate::core::collections::{
    Entry, FastHashMap, MAX_PRACTICAL_DIMENSION_SIZE, PeriodicOffsetBuffer, SimplexVertexKeyBuffer,
    SmallBuffer, Uuid, VertexKeySet,
};
use crate::core::construction::{
    FinalDelaunayValidationContext, FinalTopologyValidationContext, TriangulationConstructionError,
};
use crate::core::facet::facet_key_from_vertices;
use crate::core::simplex::{Simplex, SimplexValidationError};
use crate::core::tds::{
    InvariantError, SimplexKey, Tds, TdsConstructionError, TdsError, TdsMutationError,
    TriangulationConstructionState, VertexKey,
};
use crate::core::traits::data_type::DataType;
use crate::core::util::periodic_facet_key_from_lifted_vertices;
use crate::core::validation::{TopologyGuarantee, ValidationPolicy};
use crate::core::vertex::Vertex;
use crate::geometry::kernel::{AdaptiveKernel, Kernel};
use crate::geometry::point::Point;
use crate::geometry::util::circumcenter;
use crate::topology::traits::global_topology_model::GlobalTopologyModel;
use crate::topology::traits::topological_space::{
    GlobalTopology, TopologyKind, ToroidalConstructionMode, ToroidalDomain, ToroidalDomainError,
};
use crate::triangulation::DelaunayTriangulation;
use crate::validation::{
    DelaunayTriangulationCandidate, DelaunayTriangulationValidationError,
    DelaunayTriangulationValidationProof,
};
use num_traits::ToPrimitive;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::time::Instant;
use thiserror::Error;
const TWO_POW_52_I64: i64 = 4_503_599_627_370_496; // 2^52
const TWO_POW_52_F64: f64 = 4_503_599_627_370_496.0; // 2^52
const MAX_OFFSET_UNITS: i64 = 1_048_576;
const IMAGE_JITTER_UNITS: i64 = 64;
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;
type LiftedVertex<const D: usize> = (VertexKey, [i8; D]);
type SymbolicSignature<const D: usize> = Vec<LiftedVertex<D>>;
type PeriodicFacetKey = u64;
type OrientedPeriodicFacet = (PeriodicFacetKey, bool);
type PeriodicCandidate<const D: usize> = (
    SymbolicSignature<D>,
    SymbolicSignature<D>,
    Vec<OrientedPeriodicFacet>,
    bool,
);

/// Returns the induced orientation side of a periodic facet.
///
/// The facet order inherited from the positively oriented lifted simplex is
/// compared with canonical vertex-key order. Including the omitted-slot parity
/// makes opposite sides of the same lifted facet produce opposite bits.
fn periodic_facet_orientation_side<const D: usize>(
    lifted_vertices: &[LiftedVertex<D>],
    facet_index: usize,
) -> bool {
    let facet_vertices: Vec<LiftedVertex<D>> = lifted_vertices
        .iter()
        .enumerate()
        .filter_map(|(index, lifted_vertex)| (index != facet_index).then_some(*lifted_vertex))
        .collect();
    let mut inversion_count = 0_usize;
    for left in 0..facet_vertices.len() {
        for right in (left + 1)..facet_vertices.len() {
            if facet_vertices[left] > facet_vertices[right] {
                inversion_count += 1;
            }
        }
    }

    !inversion_count.is_multiple_of(2) ^ !facet_index.is_multiple_of(2)
}

/// Counts both oriented incidences of one periodic facet without overflowing.
#[inline]
const fn periodic_facet_incidence_total(incidence: [u8; 2]) -> u8 {
    incidence[0].saturating_add(incidence[1])
}

/// Returns whether a candidate can be added without duplicating either
/// oriented side of a periodic facet.
fn periodic_candidate_can_be_added(
    candidate_facets: &[OrientedPeriodicFacet],
    facet_incidences: &FastHashMap<PeriodicFacetKey, [u8; 2]>,
) -> bool {
    candidate_facets
        .iter()
        .enumerate()
        .all(|(position, &(facet_key, side))| {
            let side_index = usize::from(side);
            let already_selected = facet_incidences
                .get(&facet_key)
                .is_some_and(|incidence| incidence[side_index] > 0);
            let duplicated_within_candidate =
                candidate_facets[..position]
                    .iter()
                    .any(|&(earlier_key, earlier_side)| {
                        earlier_key == facet_key && earlier_side == side
                    });
            !already_selected && !duplicated_within_candidate
        })
}

/// Applies or rolls back one candidate's oriented facet incidences.
///
/// Keeping this update symmetric lets greedy and exact selection paths share
/// the same one-incidence-per-side invariant during backtracking.
fn update_periodic_facet_incidences(
    candidate_facets: &[OrientedPeriodicFacet],
    facet_incidences: &mut FastHashMap<PeriodicFacetKey, [u8; 2]>,
    is_add: bool,
) {
    for &(facet_key, side) in candidate_facets {
        let side_index = usize::from(side);
        let incidence = facet_incidences.entry(facet_key).or_insert([0; 2]);
        incidence[side_index] = if is_add {
            incidence[side_index].saturating_add(1)
        } else {
            incidence[side_index].saturating_sub(1)
        };
        if *incidence == [0; 2] {
            facet_incidences.remove(&facet_key);
        }
    }
}

/// Computes the change in unmatched-facet count for adding or removing one candidate.
fn periodic_boundary_delta(
    candidate_facets: &[OrientedPeriodicFacet],
    facet_incidences: &FastHashMap<PeriodicFacetKey, [u8; 2]>,
    is_add: bool,
) -> i32 {
    let mut delta = 0_i32;
    for (position, &(facet_key, _)) in candidate_facets.iter().enumerate() {
        if candidate_facets[..position]
            .iter()
            .any(|&(earlier_key, _)| earlier_key == facet_key)
        {
            continue;
        }
        let candidate_count = candidate_facets
            .iter()
            .filter(|&&(candidate_key, _)| candidate_key == facet_key)
            .count();
        let current_count = facet_incidences
            .get(&facet_key)
            .copied()
            .map_or(0_usize, |incidence| {
                usize::from(periodic_facet_incidence_total(incidence))
            });
        let updated_count = if is_add {
            current_count.saturating_add(candidate_count)
        } else {
            current_count.saturating_sub(candidate_count)
        };
        delta += i32::from(updated_count == 1) - i32::from(current_count == 1);
    }
    delta
}

/// Keeps the strongest connected component from a periodic candidate selection.
///
/// Finite image triangulations can contribute closed boundary-artifact components
/// in addition to the component representing the fundamental quotient. A torus
/// must be connected, so rank components by canonical-vertex coverage, then by
/// closure, in-domain support, and finally compactness.
fn best_connected_periodic_component<const D: usize>(
    selected: &[bool],
    candidates: &[PeriodicCandidate<D>],
) -> Vec<bool> {
    let mut adjacency = vec![Vec::new(); candidates.len()];
    let mut first_owner: FastHashMap<PeriodicFacetKey, usize> = FastHashMap::default();
    for (candidate_index, is_selected) in selected.iter().copied().enumerate() {
        if !is_selected {
            continue;
        }
        for &(facet_key, _) in &candidates[candidate_index].2 {
            if let Some(other_index) = first_owner.insert(facet_key, candidate_index)
                && other_index != candidate_index
            {
                adjacency[candidate_index].push(other_index);
                adjacency[other_index].push(candidate_index);
            }
        }
    }

    let mut visited = vec![false; candidates.len()];
    let mut best_component = Vec::new();
    let mut best_coverage = 0_usize;
    let mut best_boundary_count = usize::MAX;
    let mut best_in_domain_count = 0_usize;

    for root in 0..candidates.len() {
        if !selected[root] || visited[root] {
            continue;
        }
        let mut stack = vec![root];
        let mut component = Vec::new();
        visited[root] = true;
        while let Some(candidate_index) = stack.pop() {
            component.push(candidate_index);
            for &neighbor_index in &adjacency[candidate_index] {
                if !visited[neighbor_index] {
                    visited[neighbor_index] = true;
                    stack.push(neighbor_index);
                }
            }
        }

        let mut covered: VertexKeySet = VertexKeySet::default();
        let mut facet_counts: FastHashMap<PeriodicFacetKey, u8> = FastHashMap::default();
        let mut in_domain_count = 0_usize;
        for &candidate_index in &component {
            in_domain_count += usize::from(candidates[candidate_index].3);
            for &(vertex_key, _) in &candidates[candidate_index].1 {
                covered.insert(vertex_key);
            }
            for &(facet_key, _) in &candidates[candidate_index].2 {
                *facet_counts.entry(facet_key).or_insert(0) += 1;
            }
        }
        let boundary_count = facet_counts.values().filter(|&&count| count == 1).count();
        let coverage = covered.len();
        let is_better = coverage > best_coverage
            || (coverage == best_coverage
                && (boundary_count < best_boundary_count
                    || (boundary_count == best_boundary_count
                        && (in_domain_count > best_in_domain_count
                            || (in_domain_count == best_in_domain_count
                                && (best_component.is_empty()
                                    || component.len() < best_component.len()))))));
        if is_better {
            best_coverage = coverage;
            best_boundary_count = boundary_count;
            best_in_domain_count = in_domain_count;
            best_component = component;
        }
    }

    let mut connected_selection = vec![false; candidates.len()];
    for candidate_index in best_component {
        connected_selection[candidate_index] = true;
    }
    connected_selection
}

/// Bounded DFS for a connected, closed periodic quotient component.
struct ClosedPeriodicSelectionDfs<'a, const D: usize> {
    candidates: &'a [PeriodicCandidate<D>],
    candidates_by_facet_side: FastHashMap<PeriodicFacetKey, [Vec<usize>; 2]>,
    target_vertex_count: usize,
    nodes: usize,
    node_limit: usize,
}

impl<const D: usize> ClosedPeriodicSelectionDfs<'_, D> {
    /// Completes the selected component while preserving facet orientation and
    /// covering every canonical vertex required by the quotient.
    fn search(
        &mut self,
        selected: &mut [bool],
        facet_incidences: &mut FastHashMap<PeriodicFacetKey, [u8; 2]>,
        covered_vertex_counts: &mut FastHashMap<VertexKey, usize>,
    ) -> bool {
        if self.nodes >= self.node_limit {
            return false;
        }
        self.nodes = self.nodes.saturating_add(1);

        let mut unmatched_facets: Vec<PeriodicFacetKey> = facet_incidences
            .iter()
            .filter_map(|(&facet_key, &incidence)| {
                (periodic_facet_incidence_total(incidence) == 1).then_some(facet_key)
            })
            .collect();
        unmatched_facets.sort_unstable();
        if unmatched_facets.is_empty() {
            return covered_vertex_counts.len() == self.target_vertex_count;
        }

        let mut branch_candidates: Option<Vec<usize>> = None;
        for facet_key in unmatched_facets {
            let incidence = facet_incidences.get(&facet_key).copied().unwrap_or([0; 2]);
            let missing_side = usize::from(incidence[0] > 0);
            let Some(candidates_by_side) = self.candidates_by_facet_side.get(&facet_key) else {
                return false;
            };
            let viable: Vec<usize> = candidates_by_side[missing_side]
                .iter()
                .copied()
                .filter(|&candidate_index| {
                    !selected[candidate_index]
                        && periodic_candidate_can_be_added(
                            &self.candidates[candidate_index].2,
                            facet_incidences,
                        )
                })
                .collect();
            if viable.is_empty() {
                return false;
            }
            if branch_candidates
                .as_ref()
                .is_none_or(|current| viable.len() < current.len())
            {
                branch_candidates = Some(viable);
            }
        }

        let Some(branch_candidates) = branch_candidates else {
            return false;
        };
        for candidate_index in branch_candidates {
            selected[candidate_index] = true;
            update_periodic_facet_incidences(
                &self.candidates[candidate_index].2,
                facet_incidences,
                true,
            );
            for &(vertex_key, _) in &self.candidates[candidate_index].1 {
                *covered_vertex_counts.entry(vertex_key).or_insert(0) += 1;
            }

            if self.search(selected, facet_incidences, covered_vertex_counts) {
                return true;
            }

            for &(vertex_key, _) in &self.candidates[candidate_index].1 {
                if let Some(count) = covered_vertex_counts.get_mut(&vertex_key) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        covered_vertex_counts.remove(&vertex_key);
                    }
                }
            }
            update_periodic_facet_incidences(
                &self.candidates[candidate_index].2,
                facet_incidences,
                false,
            );
            selected[candidate_index] = false;
        }
        false
    }
}

/// Searches candidate seeds for one closed quotient component within a bounded
/// amount of work so periodic construction fails explicitly instead of hanging.
fn search_closed_connected_periodic_selection<const D: usize>(
    candidates: &[PeriodicCandidate<D>],
    target_vertex_count: usize,
    node_limit: usize,
) -> Option<Vec<bool>> {
    let mut candidates_by_facet_side: FastHashMap<PeriodicFacetKey, [Vec<usize>; 2]> =
        FastHashMap::default();
    for (candidate_index, candidate) in candidates.iter().enumerate() {
        for &(facet_key, side) in &candidate.2 {
            candidates_by_facet_side
                .entry(facet_key)
                .or_insert_with(|| [Vec::new(), Vec::new()])[usize::from(side)]
            .push(candidate_index);
        }
    }
    let mut search = ClosedPeriodicSelectionDfs {
        candidates,
        candidates_by_facet_side,
        target_vertex_count,
        nodes: 0,
        node_limit,
    };

    for seed_index in 0..candidates.len() {
        let mut selected = vec![false; candidates.len()];
        let mut facet_incidences: FastHashMap<PeriodicFacetKey, [u8; 2]> = FastHashMap::default();
        if !periodic_candidate_can_be_added(&candidates[seed_index].2, &facet_incidences) {
            continue;
        }
        selected[seed_index] = true;
        update_periodic_facet_incidences(&candidates[seed_index].2, &mut facet_incidences, true);
        let mut covered_vertex_counts: FastHashMap<VertexKey, usize> = FastHashMap::default();
        for &(vertex_key, _) in &candidates[seed_index].1 {
            *covered_vertex_counts.entry(vertex_key).or_insert(0) += 1;
        }
        if search.search(
            &mut selected,
            &mut facet_incidences,
            &mut covered_vertex_counts,
        ) {
            return Some(selected);
        }
        if search.nodes >= search.node_limit {
            break;
        }
    }
    None
}
/// Sort candidates by heuristic priority: prefer in-domain candidates first,
/// then by cumulative edge rarity (rarer edges first), then by index.
fn sort_candidates_by_rarity_and_domain(
    order: &mut [usize],
    candidate_edges: &[[(usize, bool); 3]],
    candidate_in_domain: &[bool],
    edge_count: usize,
) {
    let mut edge_frequency = vec![0usize; edge_count];
    for edges in candidate_edges {
        for &(edge, _) in edges {
            edge_frequency[edge] = edge_frequency[edge].saturating_add(1);
        }
    }

    order.sort_by(|a, b| {
        let a_edges = candidate_edges[*a];
        let b_edges = candidate_edges[*b];
        let a_score = edge_frequency[a_edges[0].0]
            + edge_frequency[a_edges[1].0]
            + edge_frequency[a_edges[2].0];
        let b_score = edge_frequency[b_edges[0].0]
            + edge_frequency[b_edges[1].0]
            + edge_frequency[b_edges[2].0];
        candidate_in_domain[*b]
            .cmp(&candidate_in_domain[*a])
            .then_with(|| a_score.cmp(&b_score))
            .then_with(|| a.cmp(b))
    });
}

/// DFS state for bounded face-subset search in [`search_closed_2d_selection`].
///
/// Encapsulates the immutable search parameters and mutable traversal state so
/// the recursive [`search`](Self::search) method takes only `pos` and `chosen`.
struct ClosedSelectionDfs<'a> {
    target_faces: usize,
    target_vertex_count: usize,
    order: &'a [usize],
    candidate_edges: &'a [[(usize, bool); 3]],
    candidate_vertices: &'a [Vec<VertexKey>],
    edge_incidences: Vec<[u8; 2]>,
    covered_vertex_counts: FastHashMap<VertexKey, usize>,
    selected: Vec<bool>,
    nodes: usize,
    node_limit: usize,
}

impl ClosedSelectionDfs<'_> {
    fn search(&mut self, pos: usize, chosen: usize) -> bool {
        if chosen == self.target_faces {
            return self
                .edge_incidences
                .iter()
                .all(|&incidence| incidence == [0, 0] || incidence == [1, 1])
                && self.covered_vertex_counts.len() == self.target_vertex_count
                && self.selected_faces_are_connected();
        }
        if pos == self.order.len() {
            return false;
        }
        if chosen + (self.order.len() - pos) < self.target_faces {
            return false;
        }
        if self.nodes >= self.node_limit {
            return false;
        }
        self.nodes = self.nodes.saturating_add(1);

        // Capacity-based prune: each additional face consumes 3 remaining edge incidences.
        let remaining_capacity: usize = self
            .edge_incidences
            .iter()
            .map(|incidence| {
                usize::from(1_u8.saturating_sub(incidence[0]))
                    + usize::from(1_u8.saturating_sub(incidence[1]))
            })
            .sum();
        if chosen + (remaining_capacity / 3) < self.target_faces {
            return false;
        }

        let idx = self.order[pos];
        let edges = self.candidate_edges[idx];

        let can_add = edges.iter().enumerate().all(|(position, &(edge, side))| {
            self.edge_incidences[edge][usize::from(side)] == 0
                && !edges[..position].contains(&(edge, side))
        });
        if can_add {
            self.selected[idx] = true;
            for (edge, side) in edges {
                self.edge_incidences[edge][usize::from(side)] += 1;
            }
            for &vertex_key in &self.candidate_vertices[idx] {
                *self.covered_vertex_counts.entry(vertex_key).or_insert(0) += 1;
            }

            if self.search(pos + 1, chosen + 1) {
                return true;
            }

            for &vertex_key in &self.candidate_vertices[idx] {
                if let Some(count) = self.covered_vertex_counts.get_mut(&vertex_key) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        self.covered_vertex_counts.remove(&vertex_key);
                    }
                }
            }
            for (edge, side) in edges {
                self.edge_incidences[edge][usize::from(side)] -= 1;
            }
            self.selected[idx] = false;
        }

        self.search(pos + 1, chosen)
    }

    /// Returns whether every selected face belongs to one facet-connected component.
    fn selected_faces_are_connected(&self) -> bool {
        let Some(root) = self.selected.iter().position(|is_selected| *is_selected) else {
            return false;
        };
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); self.selected.len()];
        let mut first_owner: Vec<Option<usize>> = vec![None; self.edge_incidences.len()];
        for (candidate_index, is_selected) in self.selected.iter().copied().enumerate() {
            if !is_selected {
                continue;
            }
            for &(edge, _) in &self.candidate_edges[candidate_index] {
                if let Some(other_index) = first_owner[edge] {
                    if other_index != candidate_index {
                        adjacency[candidate_index].push(other_index);
                        adjacency[other_index].push(candidate_index);
                    }
                } else {
                    first_owner[edge] = Some(candidate_index);
                }
            }
        }

        let mut visited = vec![false; self.selected.len()];
        let mut stack = vec![root];
        visited[root] = true;
        while let Some(candidate_index) = stack.pop() {
            for &neighbor_index in &adjacency[candidate_index] {
                if !visited[neighbor_index] {
                    visited[neighbor_index] = true;
                    stack.push(neighbor_index);
                }
            }
        }

        self.selected
            .iter()
            .zip(visited)
            .all(|(is_selected, was_visited)| !*is_selected || was_visited)
    }
}

/// Finds a bounded-size 2D face subset whose oriented edge incidences close a quotient boundary.
///
/// Returns a boolean mask aligned with `candidate_edges` when a selection of exactly
/// `target_faces` candidates is found such that every used edge has one incidence on each
/// orientation side, every canonical vertex is covered, and the selected faces are connected.
/// The search uses a DFS with pruning and a heuristic ordering that prefers in-domain candidates
/// first.
fn search_closed_2d_selection(
    candidate_edges: &[[(usize, bool); 3]],
    candidate_vertices: &[Vec<VertexKey>],
    candidate_in_domain: &[bool],
    target_faces: usize,
    target_vertex_count: usize,
    edge_count: usize,
    node_limit: usize,
) -> Option<Vec<bool>> {
    let m = candidate_edges.len();
    if m < target_faces {
        return None;
    }

    let mut order: Vec<usize> = (0..m).collect();
    sort_candidates_by_rarity_and_domain(
        &mut order,
        candidate_edges,
        candidate_in_domain,
        edge_count,
    );

    let mut dfs = ClosedSelectionDfs {
        target_faces,
        target_vertex_count,
        order: &order,
        candidate_edges,
        candidate_vertices,
        edge_incidences: vec![[0u8; 2]; edge_count],
        covered_vertex_counts: FastHashMap::default(),
        selected: vec![false; m],
        nodes: 0,
        node_limit,
    };

    dfs.search(0, 0).then_some(dfs.selected)
}

/// Preserves periodic-quotient TDS mutation failures as typed construction errors.
///
/// Quotient reconstruction mutates an already-built image TDS. If those
/// relation edits fail, the source is still a TDS validation/mutation failure,
/// not a stringly internal inconsistency.
fn periodic_quotient_tds_mutation_error(
    source: TdsMutationError,
) -> TriangulationConstructionError {
    TriangulationConstructionError::Tds(TdsConstructionError::ValidationError(source.into()))
}

// =============================================================================
// ERROR TYPES
// =============================================================================

/// Errors from explicit triangulation construction.
///
/// Input validation errors (wrong arity, out-of-bounds indices, duplicate vertices,
/// empty simplices) and post-assembly failures (neighbor wiring, orientation
/// normalization, structural/topology/nondegeneracy validation) are returned as
/// variants of this enum — callers should match on
/// [`ExplicitConstructionError`] (wrapped in
/// [`DelaunayTriangulationConstructionError::ExplicitConstruction`]).
///
/// Low-level explicit assembly failures are normalized into
/// [`ExplicitConstructionError::SimplexCreation`] or
/// [`ExplicitConstructionError::TdsAssembly`] so callers can handle the whole
/// explicit-construction path through
/// [`DelaunayTriangulationConstructionError::ExplicitConstruction`].
///
/// [`DelaunayTriangulationConstructionError::ExplicitConstruction`]: crate::DelaunayTriangulationConstructionError::ExplicitConstruction
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::construction::{
///     DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
///     ExplicitConstructionError,
/// };
///
/// # fn main() -> Result<(), delaunay::prelude::geometry::CoordinateConversionError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0]?,
/// ];
/// let simplices = vec![vec![0, 1]]; // Wrong arity for 2D (needs 3 vertices)
///
/// let result =
///     DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices);
/// std::assert_matches!(
///     result.err(),
///     Some(ExplicitConstructionError::InvalidSimplexArity {
///         simplex_index: 0,
///         actual: 2,
///         expected: 3,
///     })
/// );
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum ExplicitConstructionError {
    /// A simplex references a vertex index that is out of bounds.
    #[error(
        "Simplex {simplex_index}: vertex index {vertex_index} is out of bounds (vertex count: {bound})"
    )]
    IndexOutOfBounds {
        /// The index of the simplex in the input slice.
        simplex_index: usize,
        /// The out-of-bounds vertex index.
        vertex_index: usize,
        /// The number of vertices provided.
        bound: usize,
    },
    /// A simplex does not have exactly D+1 vertex indices.
    #[error(
        "Simplex {simplex_index}: has {actual} vertex indices, expected {expected} for a simplex"
    )]
    InvalidSimplexArity {
        /// The index of the simplex in the input slice.
        simplex_index: usize,
        /// The actual number of vertex indices.
        actual: usize,
        /// The expected number (D+1).
        expected: usize,
    },
    /// A simplex contains duplicate vertex indices.
    #[error("Simplex {simplex_index}: contains duplicate vertex index {vertex_index}")]
    DuplicateVertexInSimplex {
        /// The index of the simplex in the input slice.
        simplex_index: usize,
        /// The duplicated vertex index.
        vertex_index: usize,
    },
    /// No simplices were provided.
    #[error("No simplices provided for explicit construction")]
    EmptySimplices,
    /// Simplex creation failed while assembling explicit connectivity.
    #[error(
        "Simplex {simplex_index}: simplex creation failed during explicit construction: {source}"
    )]
    SimplexCreation {
        /// The index of the simplex in the input slice.
        simplex_index: usize,
        /// Underlying simplex validation error.
        #[source]
        source: SimplexValidationError,
    },
    /// TDS assembly failed while inserting explicit vertices or simplices.
    #[error("TDS assembly failed during explicit construction: {source}")]
    TdsAssembly {
        /// Underlying TDS construction or mutation error.
        #[source]
        source: Box<TdsConstructionError>,
    },
    /// Toroidal topology is incompatible with explicit simplex construction.
    #[error("Toroidal topology cannot be combined with explicit simplex construction")]
    IncompatibleTopology,
    /// Unsupported [`ConstructionOptions`] were set on an explicit-simplex builder.
    ///
    /// Most [`ConstructionOptions`] (insertion order, deduplication, retry
    /// policy) apply only to the Delaunay point-insertion path and are not
    /// meaningful for explicit simplex construction. The supported exception is
    /// [`ConstructionOptions::without_final_delaunay_enforcement`], which lets
    /// callers import valid Levels 1-4 explicit connectivity without proving the
    /// Level 5 Delaunay property.
    ///
    /// [`ConstructionOptions`]: crate::construction::ConstructionOptions
    #[error(
        "Only default ConstructionOptions or without_final_delaunay_enforcement() \
         are supported for explicit simplex construction"
    )]
    UnsupportedConstructionOptions,
    /// Neighbor assignment failed while assembling explicit connectivity.
    #[error("Neighbor assignment failed during explicit construction: {source}")]
    NeighborAssignment {
        /// Underlying TDS validation error.
        #[source]
        source: Box<TdsError>,
    },
    /// Orientation normalization or positive-orientation promotion failed.
    #[error("Orientation normalization failed during explicit construction: {source}")]
    OrientationNormalization {
        /// Underlying insertion/orientation error.
        #[source]
        source: Box<InsertionError>,
    },
    /// Level 1–2 TDS structural validation failed after assembly.
    #[error("Structural validation failed during explicit construction: {source}")]
    StructuralValidation {
        /// Underlying TDS validation error.
        #[source]
        source: Box<TdsError>,
    },
    /// Level 3 topology validation failed after assembly.
    #[error("Topology validation failed during explicit construction: {source}")]
    TopologyValidation {
        /// Underlying cumulative validation error.
        #[source]
        source: Box<InvariantError>,
    },
    /// Completion-time PL-manifold validation failed after assembly.
    #[error("PL-manifold completion validation failed during explicit construction: {source}")]
    CompletionValidation {
        /// Underlying cumulative validation error.
        #[source]
        source: Box<InvariantError>,
    },
    /// Geometric nondegeneracy validation failed after assembly.
    #[error("Geometric nondegeneracy validation failed during explicit construction: {source}")]
    GeometricNondegeneracy {
        /// Underlying TDS/geometric validation error.
        #[source]
        source: Box<TdsError>,
    },
    /// Level 4 realization validation failed before returning the wrapper.
    #[error("Realization validation failed during explicit construction: {source}")]
    RealizationValidation {
        /// Underlying cumulative realization validation error.
        #[source]
        source: Box<DelaunayTriangulationValidationError>,
    },
    /// Level 5 Delaunay validation failed before returning the wrapper.
    #[error("Delaunay validation failed during explicit construction: {source}")]
    DelaunayValidation {
        /// Underlying Delaunay validation error.
        #[source]
        source: Box<DelaunayTriangulationValidationError>,
    },
    /// Explicit quotient connectivity is not supported for the requested topology.
    #[error(
        "Explicit non-Euclidean connectivity is not supported for {topology:?}; quotient realization validation is required"
    )]
    UnsupportedExplicitTopology {
        /// Requested global topology metadata.
        topology: TopologyKind,
    },
}

#[derive(Clone, Copy)]
struct ValidatedExplicitSimplices<'v> {
    specs: &'v [Vec<usize>],
}

impl<'v> ValidatedExplicitSimplices<'v> {
    fn try_new<const D: usize>(
        vertex_count: usize,
        specs: &'v [Vec<usize>],
    ) -> Result<Self, ExplicitConstructionError> {
        validate_explicit_simplex_specs::<D>(vertex_count, specs)?;
        Ok(Self { specs })
    }

    const fn as_slice(self) -> &'v [Vec<usize>] {
        self.specs
    }
}

/// Validates explicit simplex specifications before storing them in a builder.
fn validate_explicit_simplex_specs<const D: usize>(
    vertex_count: usize,
    simplices: &[Vec<usize>],
) -> Result<(), ExplicitConstructionError> {
    if simplices.is_empty() {
        return Err(ExplicitConstructionError::EmptySimplices);
    }

    for (simplex_idx, simplex_spec) in simplices.iter().enumerate() {
        if simplex_spec.len() != D + 1 {
            return Err(ExplicitConstructionError::InvalidSimplexArity {
                simplex_index: simplex_idx,
                actual: simplex_spec.len(),
                expected: D + 1,
            });
        }
        for (i, &vi) in simplex_spec.iter().enumerate() {
            if vi >= vertex_count {
                return Err(ExplicitConstructionError::IndexOutOfBounds {
                    simplex_index: simplex_idx,
                    vertex_index: vi,
                    bound: vertex_count,
                });
            }
            for &vj in &simplex_spec[i + 1..] {
                if vi == vj {
                    return Err(ExplicitConstructionError::DuplicateVertexInSimplex {
                        simplex_index: simplex_idx,
                        vertex_index: vi,
                    });
                }
            }
        }
    }

    Ok(())
}

// =============================================================================
// BUILDER STRUCT
// =============================================================================

/// Fluent builder for [`DelaunayTriangulation`] with optional toroidal topology.
///
/// # Type Parameters
///
/// - `'v` — Lifetime of the borrowed vertex slice.
/// - `U` — Vertex data type (inferred from the vertex slice).
/// - `D` — Spatial dimension (inferred from the vertex slice).
/// - `V` — Simplex data type selected before construction; defaults to `()`.
///
/// The simplex data type `V` defaults to `()` and can be inferred from the
/// caller's expected return type. The builder never computes or validates
/// payload values; attach persisted simplex data afterward with follow-on
/// methods on [`DelaunayTriangulation`], keeping construction focused on the
/// Levels 1–5 Delaunay validity boundary.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::construction::{
///     ConstructionOptions, DelaunayResult, DelaunayTriangulationBuilder, TopologyGuarantee,
/// };
///
/// # fn main() -> DelaunayResult<()> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
///
/// let dt = DelaunayTriangulationBuilder::new(&vertices)
///     .topology_guarantee(TopologyGuarantee::Pseudomanifold)
///     .construction_options(ConstructionOptions::default())
///     .build()?;
///
/// assert_eq!(dt.number_of_vertices(), 4);
/// # Ok(())
/// # }
/// ```
pub struct DelaunayTriangulationBuilder<'v, U, const D: usize, V = ()> {
    vertices: &'v [Vertex<U, D>],
    /// Topology mode for construction.
    ///
    /// The toroidal mode carries a validated domain for periodic image-point
    /// construction.
    topology: BuilderTopology<D>,
    topology_guarantee: TopologyGuarantee,
    construction_options: ConstructionOptions,
    /// Optional explicit simplex specifications for direct combinatorial construction.
    ///
    /// When set, the builder constructs a triangulation from the given vertices and
    /// simplices directly, bypassing point-insertion-based Delaunay construction.
    explicit_simplices: Option<ValidatedExplicitSimplices<'v>>,
    /// Explicit runtime global topology metadata requested by the caller.
    ///
    /// `None` means the construction path supplies its own default metadata:
    /// Euclidean paths default to [`GlobalTopology::Euclidean`], while periodic
    /// image-point construction derives closed toroidal metadata from its domain.
    requested_global_topology: Option<GlobalTopology<D>>,
    _simplex_data: PhantomData<V>,
}

/// Topology mode requested by the public builder chain.
///
/// This stores only proof-carrying topology values, so fallible parsing stays at
/// raw-input setters and the build terminals never need to revalidate periods.
#[derive(Clone, Copy)]
enum BuilderTopology<const D: usize> {
    /// Ordinary Euclidean batch construction.
    Euclidean,
    /// Construct a toroidal quotient through periodic image points.
    PeriodicImagePoint(ToroidalDomain<D>),
}

// =============================================================================
// BUILDER IMPL — f64 coordinate storage, any vertex data U
//
// U is inferred from the vertex slice. V defaults to () and can be inferred
// from the expected DelaunayTriangulation return type when callers need typed
// persisted simplex payloads.
//
//   let vertices = vec![Vertex::<(), _>::try_new([0.0, 0.0])?, ...];
//   let dt = DelaunayTriangulationBuilder::new(&vertices).build();
//
//   let typed: [Vertex<i32, 2>; 3] = [Vertex::<_, _>::try_new_with_data([0.0, 0.0], 1)?, ...];
//   let dt = DelaunayTriangulationBuilder::new(&typed).build();
//
// =============================================================================

impl<'v, U, const D: usize> DelaunayTriangulationBuilder<'v, U, D> {
    /// Creates a builder for `f64` vertices with any user data type `U`.
    ///
    /// This is intentionally infallible: it creates an inert construction
    /// request and does not parse the point set into a triangulation. The
    /// fallible invariant boundary is [`build`](Self::build) or
    /// [`build_with_kernel`](Self::build_with_kernel), which validates the
    /// constructed triangulation before returning it.
    ///
    /// `U` is inferred from the vertex slice — no explicit type annotations needed
    /// for either `U = ()` (the common case) or typed vertex data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder, Vertex,
    /// };
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// // No vertex data (U = () inferred)
    /// let vertices = vec![delaunay::vertex![0.0, 0.0]?, delaunay::vertex![1.0, 0.0]?, delaunay::vertex![0.0, 1.0]?];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// dt.validate()?;
    ///
    /// // Typed vertex data (U = i32 inferred)
    /// let typed: [Vertex<i32, 2>; 3] = [
    ///     delaunay::vertex![0.0, 0.0; data = 1i32]?,
    ///     delaunay::vertex![1.0, 0.0; data = 2]?,
    ///     delaunay::vertex![0.0, 1.0; data = 3]?,
    /// ];
    /// let typed_dt = DelaunayTriangulationBuilder::new(&typed).build()?;
    /// typed_dt.validate()?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn new(vertices: &'v [Vertex<U, D>]) -> Self {
        Self {
            vertices,
            topology: BuilderTopology::Euclidean,
            topology_guarantee: TopologyGuarantee::DEFAULT,
            construction_options: ConstructionOptions::default(),
            explicit_simplices: None,
            requested_global_topology: None,
            _simplex_data: PhantomData,
        }
    }

    /// Creates a default-kernel builder from explicit vertex and simplex indices.
    ///
    /// This is not an unchecked topology wrapper. The deferred
    /// [`build`](Self::build) or [`build_with_kernel`](Self::build_with_kernel)
    /// call accepts only Euclidean explicit connectivity and validates the
    /// assembled triangulation at Levels 1–4, so the supplied connectivity must
    /// already satisfy the Delaunay empty-circumsphere property. Non-Euclidean
    /// explicit connectivity is rejected because it requires Level 4 handling
    /// that is not available for quotient meshes.
    ///
    /// # Errors
    ///
    /// Returns [`ExplicitConstructionError::EmptySimplices`] when no simplices
    /// are provided, [`ExplicitConstructionError::InvalidSimplexArity`] when a
    /// simplex does not contain `D + 1` vertex indices,
    /// [`ExplicitConstructionError::IndexOutOfBounds`] when a simplex references
    /// a missing vertex, or
    /// [`ExplicitConstructionError::DuplicateVertexInSimplex`] when a simplex
    /// repeats a vertex index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder,
    ///     DelaunayTriangulationConstructionError, ExplicitConstructionError,
    /// };
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![1.0, 1.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let simplices = vec![vec![0, 1, 2], vec![0, 2, 3]];
    ///
    /// let dt = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
    ///     .map_err(DelaunayTriangulationConstructionError::from)?
    ///     .build()?;
    ///
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// assert_eq!(dt.number_of_simplices(), 2);
    ///
    /// let bad_simplices = vec![vec![0, 1]]; // Wrong arity for a 2D simplex.
    /// let result =
    ///     DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &bad_simplices);
    /// std::assert_matches!(
    ///     result.err(),
    ///     Some(ExplicitConstructionError::InvalidSimplexArity { .. })
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_from_vertices_and_simplices(
        vertices: &'v [Vertex<U, D>],
        simplices: &'v [Vec<usize>],
    ) -> Result<Self, ExplicitConstructionError> {
        Self::try_from_vertices_and_simplices_generic(vertices, simplices)
    }
    /// Creates a builder from explicit vertex and simplex specifications.
    ///
    /// This constructs a triangulation from the given connectivity without
    /// Delaunay point insertion.
    ///
    /// Simplex arity, bounds, and duplicate indices are validated before the
    /// builder stores the explicit connectivity. Euclidean explicit meshes are
    /// checked at Levels 1–4 during [`build`](Self::build) or
    /// [`build_with_kernel`](Self::build_with_kernel). By default they must also
    /// prove the Level 5 Delaunay empty-circumsphere property. Use
    /// [`ConstructionOptions::without_final_delaunay_enforcement`] when importing
    /// exact degenerate or externally constrained connectivity that should be
    /// valid as a triangulation but is not required to be strict Delaunay.
    /// Non-Euclidean explicit connectivity is rejected at build time because
    /// quotient meshes need Level 4 handling before they can be accepted.
    ///
    /// # Errors
    ///
    /// Returns [`ExplicitConstructionError::EmptySimplices`] when no simplices
    /// are provided, [`ExplicitConstructionError::InvalidSimplexArity`] when a
    /// simplex does not contain `D + 1` vertex indices,
    /// [`ExplicitConstructionError::IndexOutOfBounds`] when a simplex references
    /// a missing vertex, or
    /// [`ExplicitConstructionError::DuplicateVertexInSimplex`] when a simplex
    /// repeats a vertex index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder,
    ///     DelaunayTriangulationConstructionError, Vertex,
    /// };
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices: Vec<Vertex<(), 2>> = vec![
    ///     Vertex::try_new([0.0, 0.0])?,
    ///     Vertex::try_new([1.0, 0.0])?,
    ///     Vertex::try_new([0.0, 1.0])?,
    /// ];
    /// let simplices = vec![vec![0, 1, 2]];
    ///
    /// let dt = DelaunayTriangulationBuilder::try_from_vertices_and_simplices_generic(&vertices, &simplices)
    ///     .map_err(DelaunayTriangulationConstructionError::from)?
    ///     .build()?;
    ///
    /// assert_eq!(dt.number_of_vertices(), 3);
    /// assert_eq!(dt.number_of_simplices(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_from_vertices_and_simplices_generic(
        vertices: &'v [Vertex<U, D>],
        simplices: &'v [Vec<usize>],
    ) -> Result<Self, ExplicitConstructionError> {
        let explicit_simplices =
            ValidatedExplicitSimplices::try_new::<D>(vertices.len(), simplices)?;
        let mut builder = Self::new(vertices);
        builder.explicit_simplices = Some(explicit_simplices);
        Ok(builder)
    }
}

impl<'v, U, V, const D: usize> DelaunayTriangulationBuilder<'v, U, D, V> {
    /// Selects the simplex payload type before topology storage is allocated.
    ///
    /// This method does not compute or validate payload values. It only chooses
    /// the persisted simplex storage type so later calls such as
    /// [`DelaunayTriangulation::fill_simplex_data`] can assign values without
    /// rebuilding topology or remapping simplex keys.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder,
    /// };
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    ///
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .simplex_data_type::<usize>()
    ///     .build()?;
    /// dt.fill_simplex_data(|_, simplex| simplex.number_of_vertices());
    ///
    /// for (_, simplex) in dt.simplices() {
    ///     assert_eq!(simplex.data(), Some(&3));
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn simplex_data_type<W>(self) -> DelaunayTriangulationBuilder<'v, U, D, W> {
        let Self {
            vertices,
            topology,
            topology_guarantee,
            construction_options,
            explicit_simplices,
            requested_global_topology,
            _simplex_data: _,
        } = self;
        DelaunayTriangulationBuilder {
            vertices,
            topology,
            topology_guarantee,
            construction_options,
            explicit_simplices,
            requested_global_topology,
            _simplex_data: PhantomData,
        }
    }

    /// Enables periodic toroidal topology via the image-point method.
    ///
    /// This is the correctness-first toroidal constructor: it builds a periodic
    /// quotient with rewired neighbor pointers and Euler characteristic χ = 0 for
    /// the validated `T^2` and compact `T^3` cases.
    /// Periodic construction applies a deterministic perturbation of at most
    /// approximately `2^-32` of each domain period to canonical coordinates to
    /// resolve covering-space degeneracies; stable vertex UUIDs and payloads are
    /// preserved.
    ///
    /// # Arguments
    ///
    /// * `domain` — Period length `[L_0, …, L_{D-1}]` for each dimension.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use delaunay::prelude::geometry::RobustKernel;
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder,
    /// };
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.1, 0.2]?,
    ///     delaunay::vertex![0.4, 0.7]?,
    ///     delaunay::vertex![0.7, 0.3]?,
    ///     delaunay::vertex![0.2, 0.9]?,
    ///     delaunay::vertex![0.8, 0.6]?,
    ///     delaunay::vertex![0.5, 0.1]?,
    ///     delaunay::vertex![0.3, 0.5]?,
    /// ];
    ///
    /// let kernel = RobustKernel::new();
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .try_toroidal([1.0, 1.0])
    ///     ?
    ///     .build_with_kernel(&kernel)?;
    ///
    /// assert_eq!(dt.number_of_vertices(), 7);
    /// dt.validate()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ToroidalDomainError`] when any period is non-finite, zero, or
    /// negative.
    pub fn try_toroidal(mut self, domain: [f64; D]) -> Result<Self, ToroidalDomainError> {
        self.topology = BuilderTopology::PeriodicImagePoint(ToroidalDomain::try_new(domain)?);
        Ok(self)
    }

    /// Enables periodic toroidal topology from an already-validated domain.
    ///
    /// This infallible setter is for callers that already hold a
    /// [`ToroidalDomain`]. Use [`Self::try_toroidal`] at raw numeric boundaries
    /// when the periods still need validation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulationBuilder};
    /// use delaunay::prelude::geometry::RobustKernel;
    /// use delaunay::prelude::topology::spaces::ToroidalDomain;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.1, 0.2]?,
    ///     delaunay::vertex![0.4, 0.7]?,
    ///     delaunay::vertex![0.7, 0.3]?,
    ///     delaunay::vertex![0.2, 0.9]?,
    ///     delaunay::vertex![0.8, 0.6]?,
    ///     delaunay::vertex![0.5, 0.1]?,
    ///     delaunay::vertex![0.3, 0.5]?,
    /// ];
    ///
    /// let domain = ToroidalDomain::<2>::unit();
    /// let kernel = RobustKernel::new();
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .toroidal(domain)
    ///     .build_with_kernel(&kernel)?;
    ///
    /// assert_eq!(dt.number_of_vertices(), 7);
    /// dt.validate()?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn toroidal(mut self, domain: ToroidalDomain<D>) -> Self {
        self.topology = BuilderTopology::PeriodicImagePoint(domain);
        self
    }

    /// Sets the [`TopologyGuarantee`]
    ///
    /// Defaults to [`TopologyGuarantee::DEFAULT`] (`PLManifold`).
    /// The builder derives the initial [`ValidationPolicy`]
    /// from this guarantee; it does not expose a separate construction-time validation
    /// policy knob.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder, TopologyGuarantee,
    /// };
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .topology_guarantee(TopologyGuarantee::Pseudomanifold)
    ///     .build()?;
    ///
    /// assert_eq!(dt.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn topology_guarantee(mut self, topology_guarantee: TopologyGuarantee) -> Self {
        self.topology_guarantee = topology_guarantee;
        self
    }

    /// Sets the [`GlobalTopology`] metadata for the triangulation.
    ///
    /// This declares the intended global topology so that Euler characteristic
    /// validation uses the correct expectation. For example, setting
    /// [`GlobalTopology::Toroidal`] tells the validator to expect χ = 0 for a closed
    /// mesh instead of χ = 2 (the sphere default).
    ///
    /// This is **metadata only** and does not trigger coordinate canonicalization
    /// or image-point construction. Plain Euclidean and explicit-simplex
    /// construction reject non-Euclidean metadata because those paths do not build
    /// closed quotient connectivity. For construction-time toroidal processing,
    /// use [`.try_toroidal()`](Self::try_toroidal). If explicit metadata is supplied
    /// on the periodic image-point path, it must exactly match the toroidal topology
    /// derived from that method.
    ///
    /// When this setter is not called, Euclidean and explicit construction paths
    /// use [`GlobalTopology::Euclidean`]. The periodic image-point path derives
    /// [`GlobalTopology::Toroidal`] metadata from
    /// [`.try_toroidal()`](Self::try_toroidal) instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder,
    ///     DelaunayTriangulationConstructionError, GlobalTopology, ToroidalConstructionMode,
    /// };
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let simplices = vec![vec![0, 1, 2]];
    ///
    /// let topology = GlobalTopology::try_toroidal(
    ///     [1.0, 1.0],
    ///     ToroidalConstructionMode::Explicit,
    /// )
    /// ?;
    /// let result = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
    ///     .map_err(DelaunayTriangulationConstructionError::from)?
    ///     .global_topology(topology)
    ///     .build();
    ///
    /// assert!(result.is_err());
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn global_topology(mut self, global_topology: GlobalTopology<D>) -> Self {
        self.requested_global_topology = Some(global_topology);
        self
    }

    /// Sets the [`ConstructionOptions`] for construction and final validation.
    ///
    /// Defaults to [`ConstructionOptions::default`], which enforces final Level
    /// 5 Delaunay validation. Explicit-simplex builders accept either the
    /// default options or
    /// [`ConstructionOptions::without_final_delaunay_enforcement`] when callers
    /// intentionally import valid Levels 1-4 connectivity without requiring a
    /// strict Delaunay proof.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     ConstructionOptions, DelaunayResult, DelaunayTriangulationBuilder,
    ///     InsertionOrderStrategy,
    /// };
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    ///
    /// let opts = ConstructionOptions::default()
    ///     .with_insertion_order(InsertionOrderStrategy::Input);
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .construction_options(opts)
    ///     .build()?;
    ///
    /// assert_eq!(dt.number_of_vertices(), 3);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn construction_options(mut self, construction_options: ConstructionOptions) -> Self {
        self.construction_options = construction_options;
        self
    }
}

impl<U, V, const D: usize> DelaunayTriangulationBuilder<'_, U, D, V>
where
    U: DataType,
    V: DataType,
{
    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    /// Validates the topology model configuration before using it in construction.
    ///
    /// This helper is called before any topology-based canonicalization or lifting operations
    /// to ensure that the model's runtime parameters (e.g., toroidal domain periods) are valid.
    ///
    /// # Parameters
    ///
    /// * `model` - The topology behavior model to validate.
    ///
    /// # Returns
    ///
    /// - `Ok(())` if the model configuration is valid.
    /// - `Err(DelaunayTriangulationConstructionError)` if validation fails.
    ///
    /// # Errors
    ///
    /// Preserves
    /// [`GlobalTopologyModelError`](crate::topology::traits::GlobalTopologyModelError)
    /// as
    /// [`DelaunayConstructionFailure::TopologyModelConfiguration`].
    ///
    /// # Usage
    ///
    /// Called internally by [`build_with_kernel`](Self::build_with_kernel) before
    /// periodic toroidal coordinate canonicalization.
    fn validate_topology_model<M>(model: &M) -> Result<(), DelaunayTriangulationConstructionError>
    where
        M: GlobalTopologyModel<D>,
    {
        model.validate_configuration().map_err(|source| {
            DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::TopologyModelConfiguration { source },
            )
        })
    }

    /// Derives a periodic facet key from a lifted simplex and maps derivation failures.
    ///
    /// This helper centralizes error conversion for
    /// [`periodic_facet_key_from_lifted_vertices`] so all call sites produce
    /// consistent diagnostics.
    ///
    /// # Parameters
    ///
    /// * `lifted_ordered` - Lifted simplex vertices as `(VertexKey, lattice_offset)`
    ///   pairs (expected arity: `D + 1`).
    /// * `facet_idx` - Index of the facet opposite a vertex in the simplex.
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayTriangulationConstructionError`] wrapping
    /// [`TriangulationConstructionError::PeriodicQuotientFacetKeyDerivation`]
    /// when periodic facet key derivation fails (e.g., invalid arity/index or
    /// offset encoding).
    fn derive_periodic_facet_key(
        lifted_ordered: &[(VertexKey, [i8; D])],
        facet_idx: usize,
    ) -> Result<PeriodicFacetKey, DelaunayTriangulationConstructionError> {
        periodic_facet_key_from_lifted_vertices::<D>(lifted_ordered, facet_idx).map_err(|error| {
            TriangulationConstructionError::PeriodicQuotientFacetKeyDerivation {
                facet_index: facet_idx,
                reason: error.into(),
            }
            .into()
        })
    }

    /// Canonicalizes vertices using a topology behavior model.
    ///
    /// For each input calls [`GlobalTopologyModel::canonicalize_point_in_place`] to wrap
    /// coordinates into the model's fundamental domain (e.g., [0, L) for toroidal topologies).
    /// Preserves vertex UUIDs and data while transforming coordinates.
    ///
    /// # Parameters
    ///
    /// * `vertices` - Slice of input vertices with potentially out-of-domain coordinates.
    /// * `model` - The topology behavior model that defines canonicalization logic.
    ///
    /// # Returns
    ///
    /// A new vector of vertices with canonicalized coordinates. Each output vertex has:
    /// - The same UUID as the corresponding input vertex (for tracking through construction).
    /// - The same associated data as the input vertex.
    /// - Coordinates transformed according to the model's canonicalization rules.
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayTriangulationConstructionError`] if canonicalization
    /// fails for any vertex. Topology-model failures are reported as
    /// [`DelaunayConstructionFailure::VertexCanonicalization`]; invalid
    /// canonicalized coordinates are reported as
    /// [`DelaunayConstructionFailure::CanonicalizedPointValidation`].
    ///
    /// # Usage
    ///
    /// Called internally by [`build_with_kernel`](Self::build_with_kernel) before delegating
    /// to the underlying triangulation construction.
    fn canonicalize_vertices<M>(
        vertices: &[Vertex<U, D>],
        model: &M,
    ) -> Result<Vec<Vertex<U, D>>, DelaunayTriangulationConstructionError>
    where
        M: GlobalTopologyModel<D>,
    {
        let mut out = Vec::with_capacity(vertices.len());

        for (idx, v) in vertices.iter().enumerate() {
            let mut canonicalized_coords = *v.point().coords();
            model
                .canonicalize_point_in_place(&mut canonicalized_coords)
                .map_err(|source| {
                    DelaunayTriangulationConstructionError::Triangulation(
                        DelaunayConstructionFailure::VertexCanonicalization {
                            vertex_index: idx,
                            source,
                        },
                    )
                })?;

            let new_point = Point::try_new(canonicalized_coords).map_err(|error| {
                DelaunayTriangulationConstructionError::Triangulation(
                    DelaunayConstructionFailure::CanonicalizedPointValidation {
                        vertex_index: idx,
                        source: error,
                    },
                )
            })?;
            let new_vertex = Vertex::from_validated_point_with_uuid(new_point, v.uuid(), v.data);

            out.push(new_vertex);
        }

        Ok(out)
    }

    // -------------------------------------------------------------------------
    // Build methods
    // -------------------------------------------------------------------------

    /// Builds the triangulation using [`AdaptiveKernel<f64>`](crate::geometry::kernel::AdaptiveKernel).
    ///
    /// This is the most common build path. Simplex data type `V` is inferred or
    /// specified at the call site; it is independent of the vertex data type `U`.
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayTriangulationConstructionError`] if:
    /// - The underlying triangulation construction fails (insufficient vertices,
    ///   geometric degeneracy, etc.).
    /// - Explicit-simplex construction is requested with unsupported
    ///   [`ConstructionOptions`], non-Euclidean topology, invalid topology or
    ///   realization, or a failed Level 5 Delaunay check when final enforcement is
    ///   enabled.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder,
    /// };
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .build()?;
    ///
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// assert!(dt.validate().is_ok());
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(
        self,
    ) -> Result<
        DelaunayTriangulation<AdaptiveKernel<f64>, U, V, D>,
        DelaunayTriangulationConstructionError,
    > {
        self.build_with_kernel(&AdaptiveKernel::new())
    }

    /// Builds the triangulation and returns aggregate construction statistics.
    ///
    /// This is the fluent builder terminal for callers that want the configured
    /// workflow and skipped-input observability in one chain. Euclidean
    /// construction returns statistics from the same batch backend used by
    /// [`Self::build_with_kernel`]. Explicit connectivity and periodic quotient
    /// construction bypass the batch
    /// insertion statistics collector; on those successful paths the returned
    /// statistics record the final vertex count in
    /// [`ConstructionStatistics::inserted`] and total construction timing in
    /// [`crate::diagnostics::ConstructionTelemetry::construction_total_nanos`],
    /// while per-insertion telemetry stays at its default values.
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayTriangulationConstructionErrorWithStatistics`] if
    /// construction fails. The error carries partial statistics when the selected
    /// construction path collects them; otherwise it carries default counters
    /// plus total construction timing when that timing was observed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, DelaunayTriangulationConstructionErrorWithStatistics,
    /// };
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] DelaunayTriangulationConstructionErrorWithStatistics),
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
    /// let (dt, stats) = DelaunayTriangulationBuilder::new(&vertices)
    ///     .build_with_statistics()?;
    ///
    /// assert_eq!(dt.number_of_vertices(), stats.inserted);
    /// # Ok(())
    /// # }
    /// ```
    #[expect(
        clippy::result_large_err,
        reason = "Public API intentionally returns by-value construction statistics"
    )]
    #[expect(
        clippy::type_complexity,
        reason = "Public API returns the constructed triangulation together with construction statistics"
    )]
    pub fn build_with_statistics(
        self,
    ) -> Result<
        (
            DelaunayTriangulation<AdaptiveKernel<f64>, U, V, D>,
            ConstructionStatistics,
        ),
        DelaunayTriangulationConstructionErrorWithStatistics,
    > {
        self.build_with_kernel_and_statistics(&AdaptiveKernel::new())
    }

    /// Builds the triangulation using a caller-supplied kernel.
    ///
    /// [`build()`](Self::build) already defaults to [`AdaptiveKernel`], so this method is
    /// only needed when you want a different kernel (e.g. [`FastKernel`](crate::geometry::kernel::FastKernel)
    /// for workloads that prioritize speed over exact predicate correctness, or a custom
    /// implementation).
    ///
    /// **Note:** `FastKernel` is accepted for construction, but the explicit repair methods
    /// ([`repair_delaunay_with_flips`](DelaunayTriangulation::repair_delaunay_with_flips),
    /// [`repair_delaunay_with_flips_advanced`](DelaunayTriangulation::repair_delaunay_with_flips_advanced))
    /// require [`ExactPredicates`](crate::geometry::kernel::ExactPredicates) and are not available
    /// for `FastKernel`.
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayTriangulationConstructionError`] if canonicalization,
    /// point-insertion construction, or explicit-simplex construction fails (see
    /// [`build`](Self::build) for the policy-level conditions).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::geometry::RobustKernel;
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder,
    /// };
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    ///
    /// let kernel = RobustKernel::new();
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .build_with_kernel(&kernel)?;
    ///
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// # Ok(())
    /// # }
    /// ```
    pub fn build_with_kernel<K>(
        self,
        kernel: &K,
    ) -> Result<DelaunayTriangulation<K, U, V, D>, DelaunayTriangulationConstructionError>
    where
        K: Kernel<D, Scalar = f64>,
    {
        // Explicit-simplices path: bypass Delaunay insertion entirely.
        if let Some(simplices) = self.explicit_simplices {
            if !matches!(self.topology, BuilderTopology::Euclidean) {
                return Err(ExplicitConstructionError::IncompatibleTopology.into());
            }
            if !Self::supports_explicit_construction_options(self.construction_options) {
                return Err(ExplicitConstructionError::UnsupportedConstructionOptions.into());
            }
            return Self::build_explicit(
                kernel,
                self.vertices,
                simplices,
                self.topology_guarantee,
                self.global_topology_or_default(),
                self.construction_options.enforces_final_delaunay(),
            );
        }

        match self.topology {
            BuilderTopology::Euclidean => {
                Self::reject_euclidean_non_euclidean_topology(self.global_topology_or_default())?;
            }
            BuilderTopology::PeriodicImagePoint(domain) => {
                let topology = Self::periodic_image_global_topology(domain);
                Self::reject_periodic_conflicting_global_topology(
                    self.requested_global_topology,
                    topology,
                )?;
                let topology_model = topology.model();
                Self::validate_topology_model(&topology_model)?;
                if !topology_model.supports_periodic_facet_signatures() {
                    return Err(
                        TriangulationConstructionError::PeriodicImageUnsupportedTopology {
                            topology: topology_model.kind(),
                        }
                        .into(),
                    );
                }
                // Periodic toroidal construction: canonicalize then apply 3^D image-point method.
                let canonical = Self::canonicalize_vertices(self.vertices, &topology_model)?;
                let mut dt = Self::build_periodic(
                    kernel,
                    &canonical,
                    &topology_model,
                    self.topology_guarantee,
                    self.construction_options,
                )?;
                dt.tri
                    .normalize_and_promote_positive_orientation()
                    .map_err(|source| {
                        TriangulationConstructionError::PeriodicImageOrientationCanonicalization {
                            source: Box::new(source),
                        }
                    })?;
                dt.as_triangulation()
                    .validate_geometric_simplex_orientation()
                    .map_err(|source| {
                        TriangulationConstructionError::PeriodicImageGeometricOrientationValidation {
                            source: Box::new(source),
                        }
                    })?;
                dt.as_triangulation().validate().map_err(|e| {
                    TriangulationConstructionError::FinalTopologyValidation {
                        context: FinalTopologyValidationContext::PeriodicQuotientTopology,
                        source: Box::new(e),
                    }
                })?;
                dt.is_valid_delaunay().map_err(|e| {
                    TriangulationConstructionError::FinalDelaunayValidation {
                        context: FinalDelaunayValidationContext::PeriodicQuotientDelaunay,
                        source: e,
                    }
                })?;
                return Ok(dt);
            }
        }

        let dt = DelaunayTriangulation::build_with_kernel_options(
            kernel,
            self.vertices,
            self.topology_guarantee,
            self.construction_options,
        )?;
        Ok(dt)
    }

    /// Builds the triangulation with a caller-supplied kernel and returns statistics.
    ///
    /// This is the statistics-returning counterpart of
    /// [`build_with_kernel`](Self::build_with_kernel). Euclidean construction
    /// returns batch insertion statistics. Explicit connectivity and periodic
    /// quotient construction bypass the
    /// batch insertion statistics collector; on those successful paths the
    /// returned statistics record the final vertex count in
    /// [`ConstructionStatistics::inserted`] and total construction timing in
    /// [`crate::diagnostics::ConstructionTelemetry::construction_total_nanos`],
    /// while per-insertion telemetry stays at its default values.
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayTriangulationConstructionErrorWithStatistics`] if
    /// construction fails. The error carries partial statistics when the selected
    /// construction path collects them; otherwise it carries default counters
    /// plus total construction timing when that timing was observed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, DelaunayTriangulationConstructionErrorWithStatistics,
    /// };
    /// use delaunay::prelude::geometry::RobustKernel;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] DelaunayTriangulationConstructionErrorWithStatistics),
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
    /// let kernel = RobustKernel::new();
    ///
    /// let (dt, stats) = DelaunayTriangulationBuilder::new(&vertices)
    ///     .build_with_kernel_and_statistics(&kernel)?;
    ///
    /// assert_eq!(dt.number_of_vertices(), stats.inserted);
    /// # Ok(())
    /// # }
    /// ```
    #[expect(
        clippy::result_large_err,
        reason = "Public API intentionally returns by-value construction statistics"
    )]
    pub fn build_with_kernel_and_statistics<K>(
        self,
        kernel: &K,
    ) -> Result<
        (DelaunayTriangulation<K, U, V, D>, ConstructionStatistics),
        DelaunayTriangulationConstructionErrorWithStatistics,
    >
    where
        K: Kernel<D, Scalar = f64>,
    {
        let construction_started = Instant::now();

        let result = if self.explicit_simplices.is_some()
            || matches!(&self.topology, BuilderTopology::PeriodicImagePoint(_))
        {
            self.build_with_kernel(kernel)
                .map(Self::with_minimal_success_statistics)
                .map_err(Self::with_default_error_statistics)
        } else {
            let global_topology = self.global_topology_or_default();
            let vertices = self.vertices;
            let topology_guarantee = self.topology_guarantee;
            let construction_options = self.construction_options;

            Self::reject_euclidean_non_euclidean_topology(global_topology)
                .map_err(Self::with_default_error_statistics)
                .and_then(|()| {
                    DelaunayTriangulation::build_with_kernel_options_and_statistics(
                        kernel,
                        vertices,
                        topology_guarantee,
                        construction_options,
                    )
                })
        };

        Self::record_construction_total_timing(result, construction_started)
    }

    #[expect(
        clippy::result_large_err,
        reason = "Internal helper preserves the public by-value construction-statistics error"
    )]
    fn record_construction_total_timing<K>(
        result: Result<
            (DelaunayTriangulation<K, U, V, D>, ConstructionStatistics),
            DelaunayTriangulationConstructionErrorWithStatistics,
        >,
        construction_started: Instant,
    ) -> Result<
        (DelaunayTriangulation<K, U, V, D>, ConstructionStatistics),
        DelaunayTriangulationConstructionErrorWithStatistics,
    > {
        let elapsed_nanos = duration_nanos_saturating(construction_started.elapsed());
        match result {
            Ok((dt, mut statistics)) => {
                statistics
                    .telemetry
                    .record_construction_total_timing(elapsed_nanos);
                Ok((dt, statistics))
            }
            Err(mut error) => {
                error
                    .statistics
                    .telemetry
                    .record_construction_total_timing(elapsed_nanos);
                Err(error)
            }
        }
    }

    fn with_minimal_success_statistics<K>(
        dt: DelaunayTriangulation<K, U, V, D>,
    ) -> (DelaunayTriangulation<K, U, V, D>, ConstructionStatistics) {
        let statistics = ConstructionStatistics {
            inserted: dt.number_of_vertices(),
            ..ConstructionStatistics::default()
        };
        (dt, statistics)
    }

    fn with_default_error_statistics(
        error: DelaunayTriangulationConstructionError,
    ) -> DelaunayTriangulationConstructionErrorWithStatistics {
        DelaunayTriangulationConstructionErrorWithStatistics {
            error,
            statistics: ConstructionStatistics::default(),
        }
    }

    /// Checks whether explicit-simplex construction can honor the requested options.
    ///
    /// Explicit connectivity bypasses insertion ordering, deduplication, and
    /// retry policies, so accepting arbitrary [`ConstructionOptions`] would make
    /// those knobs look meaningful when they are ignored. The one supported
    /// non-default policy is
    /// [`ConstructionOptions::without_final_delaunay_enforcement`], which is the
    /// public opt-in for importing valid Levels 1-4 constrained connectivity.
    fn supports_explicit_construction_options(options: ConstructionOptions) -> bool {
        options == ConstructionOptions::default()
            || options == ConstructionOptions::default().without_final_delaunay_enforcement()
    }

    /// Proves explicit simplex topology once before taking the prechecked TDS insertion path.
    ///
    /// The public constructor has already rejected empty input, wrong arity,
    /// out-of-bounds vertex indices, and repeated vertex indices within a
    /// simplex. This helper checks the cross-simplex topology that only becomes
    /// visible after raw indices are mapped to canonical [`VertexKey`] values.
    fn validate_explicit_topology(
        simplices: ValidatedExplicitSimplices<'_>,
        index_to_key: &[VertexKey],
    ) -> Result<(), TdsConstructionError> {
        let simplices = simplices.as_slice();
        Self::reject_duplicate_explicit_simplices(simplices, index_to_key)?;
        Self::reject_overshared_explicit_facets(simplices, index_to_key)?;
        Ok(())
    }

    /// Rejects duplicate explicit maximal simplices before the insertion loop.
    ///
    /// The duplicate check uses sorted [`VertexKey`] identities so callers cannot
    /// bypass the explicit-construction invariant by permuting the same input
    /// simplex indices.
    fn reject_duplicate_explicit_simplices(
        simplices: &[Vec<usize>],
        index_to_key: &[VertexKey],
    ) -> Result<(), TdsConstructionError> {
        let mut seen: FastHashMap<SimplexVertexKeyBuffer, usize> = FastHashMap::default();

        for (simplex_index, simplex_spec) in simplices.iter().enumerate() {
            let identity = Self::explicit_simplex_identity(simplex_spec, index_to_key);
            match seen.entry(identity) {
                Entry::Occupied(entry) => {
                    let existing_index = *entry.get();
                    return Err(TdsConstructionError::ValidationError(
                        TdsError::DuplicateExplicitSimplices {
                            existing_simplex_index: existing_index,
                            duplicate_simplex_index: simplex_index,
                            vertex_indices: Self::explicit_simplex_input_indices(simplex_spec),
                        },
                    ));
                }
                Entry::Vacant(entry) => {
                    entry.insert(simplex_index);
                }
            }
        }

        Ok(())
    }

    /// Rejects explicit facets that would exceed PL-manifold multiplicity.
    ///
    /// This protects the public explicit-import path from committing topology
    /// that later Level 3 validation must reject after neighbor assignment.
    fn reject_overshared_explicit_facets(
        simplices: &[Vec<usize>],
        index_to_key: &[VertexKey],
    ) -> Result<(), TdsConstructionError> {
        let mut facet_incident_counts: FastHashMap<SimplexVertexKeyBuffer, usize> =
            FastHashMap::default();

        for (simplex_index, simplex_spec) in simplices.iter().enumerate() {
            for facet_index in 0..=D {
                let facet_identity =
                    Self::explicit_facet_identity(simplex_spec, index_to_key, facet_index);
                match facet_incident_counts.entry(facet_identity) {
                    Entry::Occupied(mut entry) => {
                        let incident_count = *entry.get();
                        if incident_count >= 2 {
                            return Err(TdsConstructionError::ValidationError(
                                TdsError::ExplicitFacetSharingViolation {
                                    facet_key: facet_key_from_vertices(entry.key().as_slice()),
                                    facet_vertex_indices: Self::explicit_facet_input_indices(
                                        simplex_spec,
                                        facet_index,
                                    ),
                                    existing_incident_count: incident_count,
                                    attempted_incident_count: incident_count + 1,
                                    max_incident_count: 2,
                                    candidate_simplex_index: simplex_index,
                                    candidate_facet_index: facet_index,
                                },
                            ));
                        }
                        *entry.get_mut() += 1;
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(1);
                    }
                }
            }
        }

        Ok(())
    }

    /// Builds a canonical simplex identity from prevalidated explicit vertex indices.
    fn explicit_simplex_identity(
        simplex_spec: &[usize],
        index_to_key: &[VertexKey],
    ) -> SimplexVertexKeyBuffer {
        let mut identity = Self::explicit_simplex_vertex_keys(simplex_spec, index_to_key);
        identity.as_mut_slice().sort_unstable();
        identity
    }

    /// Builds a canonical input-index identity from prevalidated explicit vertex indices.
    fn explicit_simplex_input_indices(simplex_spec: &[usize]) -> Vec<usize> {
        let mut identity = simplex_spec.to_vec();
        identity.sort_unstable();
        identity
    }

    /// Builds a canonical facet identity for one prevalidated explicit simplex facet.
    fn explicit_facet_identity(
        simplex_spec: &[usize],
        index_to_key: &[VertexKey],
        facet_index: usize,
    ) -> SimplexVertexKeyBuffer {
        let mut identity = SimplexVertexKeyBuffer::new();
        identity.extend(simplex_spec.iter().enumerate().filter_map(
            |(local_vertex_index, &input_vertex_index)| {
                (local_vertex_index != facet_index).then_some(index_to_key[input_vertex_index])
            },
        ));
        identity.as_mut_slice().sort_unstable();
        identity
    }

    /// Builds a canonical input-index identity for one prevalidated explicit simplex facet.
    fn explicit_facet_input_indices(simplex_spec: &[usize], facet_index: usize) -> Vec<usize> {
        let mut identity: Vec<usize> = simplex_spec
            .iter()
            .enumerate()
            .filter_map(|(local_vertex_index, &input_vertex_index)| {
                (local_vertex_index != facet_index).then_some(input_vertex_index)
            })
            .collect();
        identity.sort_unstable();
        identity
    }

    /// Maps prevalidated explicit vertex indices into TDS keys without an intermediate `Vec`.
    fn explicit_simplex_vertex_keys(
        simplex_spec: &[usize],
        index_to_key: &[VertexKey],
    ) -> SimplexVertexKeyBuffer {
        simplex_spec.iter().map(|&vi| index_to_key[vi]).collect()
    }

    /// Builds a triangulation from explicit vertex and simplex specifications.
    ///
    /// This explicit-connectivity construction assembles a valid TDS from the given
    /// vertices and connectivity without Delaunay point insertion. Euclidean explicit
    /// meshes are validated at Levels 1–4 and, by default, Level 5. When
    /// [`ConstructionOptions::without_final_delaunay_enforcement`] is used, the
    /// explicit connectivity is returned after Levels 1–4 validation without
    /// Delaunay repair or proof. Non-Euclidean explicit connectivity is rejected
    /// because it requires quotient realization validation before the public
    /// `DelaunayTriangulation` wrapper can accept it.
    ///
    /// # Algorithm
    ///
    /// 1. Receive prevalidated simplex specs: each simplex has D+1 in-bounds,
    ///    unique vertex indices.
    /// 2. Reject non-Euclidean explicit connectivity until quotient realization
    ///    validation exists.
    /// 3. Build a `Tds`: insert all vertices, then insert simplices from the specifications.
    /// 4. Compute adjacency via `assign_neighbors()`.
    /// 5. Assign incident simplices via `assign_incident_simplices()`.
    /// 6. Wrap in a validation candidate.
    /// 7. Normalize coherent orientation and promote to positive canonical sign
    ///    via `normalize_and_promote_positive_orientation()`.
    /// 8. Validate Levels 1–2 (TDS structural: `tds.validate()`).
    /// 9. Validate Level 3 topology (excluding geometric orientation).
    /// 10. Validate PL-manifold completion (vertex links, if required).
    /// 11. Validate geometric nondegeneracy (reject zero-volume simplices).
    /// 12. Validate the Euclidean Level 5 Delaunay property unless final
    ///     enforcement was disabled in [`ConstructionOptions`].
    fn build_explicit<K>(
        kernel: &K,
        vertices: &[Vertex<U, D>],
        simplices: ValidatedExplicitSimplices<'_>,
        topology_guarantee: TopologyGuarantee,
        global_topology: GlobalTopology<D>,
        enforce_final_delaunay: bool,
    ) -> Result<DelaunayTriangulation<K, U, V, D>, DelaunayTriangulationConstructionError>
    where
        K: Kernel<D, Scalar = f64>,
    {
        Self::reject_explicit_non_euclidean_topology(global_topology)?;

        let vertex_count = vertices.len();
        let simplex_specs = simplices.as_slice();

        // --- Build TDS ---
        let mut tds: Tds<U, V, D> = Tds::empty();

        // Insert all vertices and build index → VertexKey map.
        let mut index_to_key = Vec::with_capacity(vertex_count);
        for v in vertices {
            let vk = tds.insert_vertex_with_mapping(*v).map_err(|source| {
                ExplicitConstructionError::TdsAssembly {
                    source: Box::new(source),
                }
            })?;
            index_to_key.push(vk);
        }

        Self::validate_explicit_topology(simplices, &index_to_key).map_err(|source| {
            ExplicitConstructionError::TdsAssembly {
                source: Box::new(source),
            }
        })?;

        // Construct each simplex only at insertion time after the topology
        // precheck has proved duplicate and facet-sharing constraints.
        for (simplex_idx, simplex_spec) in simplex_specs.iter().enumerate() {
            let vertex_keys = Self::explicit_simplex_vertex_keys(simplex_spec, &index_to_key);
            let simplex = Simplex::try_new(vertex_keys).map_err(|e| {
                ExplicitConstructionError::SimplexCreation {
                    simplex_index: simplex_idx,
                    source: e,
                }
            })?;
            tds.insert_simplex_with_mapping_prechecked_topology(simplex)
                .map_err(|source| ExplicitConstructionError::TdsAssembly {
                    source: Box::new(source),
                })?;
        }

        // Mark as constructed so validation doesn't reject incomplete state.
        tds.construction_state = TriangulationConstructionState::Constructed;

        // --- Compute adjacency ---
        tds.assign_neighbors()
            .map_err(|source| ExplicitConstructionError::NeighborAssignment {
                source: Box::new(source),
            })?;

        // --- Assign incident simplices ---
        tds.assign_incident_simplices().map_err(|source| {
            ExplicitConstructionError::TdsAssembly {
                source: Box::new(TdsConstructionError::ValidationError(source.into())),
            }
        })?;

        // --- Wrap in DelaunayTriangulation ---
        //
        // Construct the DT first so the Triangulation-layer helpers
        // (orientation promotion, topology checks) operate on the assembled
        // complex.
        // Include global topology metadata before validation so that
        // validate_topology_core() uses the correct Euler characteristic
        // expectation (e.g. χ = 0 for toroidal instead of χ = 2 for sphere).
        let mut candidate = DelaunayTriangulationCandidate::assemble(
            tds,
            kernel.clone(),
            topology_guarantee,
            global_topology,
        );

        // --- Normalize orientation and promote to positive ---
        //
        // normalize_and_promote_positive_orientation() combines:
        //   1. BFS coherent-orientation normalization (adjacent simplices agree)
        //   2. Global sign canonicalization (flip all if representative is negative)
        //   3. Bounded per-simplex promotion passes for FP-precision edge cases
        // This ensures the returned DT has positive geometric orientation,
        // matching the invariant expected by validate_geometric_simplex_orientation.
        candidate
            .normalize_and_promote_positive_orientation()
            .map_err(
                |source| ExplicitConstructionError::OrientationNormalization {
                    source: Box::new(source),
                },
            )?;

        // Level 1–2: TDS structural validation (mappings, neighbors, facet
        // sharing, coherent orientation, etc.).
        if let Err(e) = candidate.validate_tds_structure() {
            return Err(ExplicitConstructionError::StructuralValidation {
                source: Box::new(e),
            }
            .into());
        }

        // Level 3 intrinsic topology: connectedness, manifold facets, isolated
        // vertices, Euler characteristic, and PL-manifold vertex/ridge links
        // when the topology guarantee requires them. Coordinate-dependent
        // orientation belongs to the Level 4 realization checks below.
        if let Err(e) = candidate.validate_topology() {
            return Err(ExplicitConstructionError::TopologyValidation {
                source: Box::new(e),
            }
            .into());
        }

        // Completion-time PL-manifold check (vertex links) if required.
        if let Err(e) = candidate.validate_at_completion() {
            return Err(ExplicitConstructionError::CompletionValidation {
                source: Box::new(e),
            }
            .into());
        }

        // --- Geometric nondegeneracy ---
        //
        // Reject degenerate simplices (zero-volume simplices from collinear /
        // coplanar vertices).  Unlike the Delaunay construction paths, which
        // may tolerate near-degenerate simplices from flip-based repair,
        // explicit construction should not silently accept geometrically
        // collapsed simplices supplied by the user.
        if let Err(e) = candidate.validate_geometric_nondegeneracy() {
            return Err(ExplicitConstructionError::GeometricNondegeneracy {
                source: Box::new(e),
            }
            .into());
        }

        if !enforce_final_delaunay {
            let proof = candidate.validate_realization_only().map_err(|source| {
                ExplicitConstructionError::RealizationValidation {
                    source: Box::new(source),
                }
            })?;
            return Ok(candidate.into_realization_validated_delaunay(proof));
        }

        let proof = Self::enforce_explicit_delaunay_property(&candidate)?;
        Ok(candidate.into_validated_delaunay(proof))
    }

    /// Enforces Level 5 validation before returning the Delaunay wrapper.
    ///
    /// The public return type is `DelaunayTriangulation`, so Euclidean explicit
    /// connectivity must prove the empty-circumsphere property before it crosses
    /// this API boundary. Explicit non-Euclidean topology is rejected earlier in
    /// `build_explicit` until quotient realization validation exists for explicit
    /// connectivity.
    fn enforce_explicit_delaunay_property<K>(
        candidate: &DelaunayTriangulationCandidate<K, U, V, D>,
    ) -> Result<DelaunayTriangulationValidationProof, DelaunayTriangulationConstructionError>
    where
        K: Kernel<D, Scalar = f64>,
    {
        candidate.validate_delaunay_property().map_err(|source| {
            ExplicitConstructionError::DelaunayValidation {
                source: Box::new(source),
            }
            .into()
        })
    }

    /// Rejects explicit quotient connectivity until realization validation supports it.
    fn reject_explicit_non_euclidean_topology(
        global_topology: GlobalTopology<D>,
    ) -> Result<(), DelaunayTriangulationConstructionError> {
        if global_topology.is_euclidean() {
            return Ok(());
        }

        Err(ExplicitConstructionError::UnsupportedExplicitTopology {
            topology: global_topology.kind(),
        }
        .into())
    }

    /// Returns the requested topology metadata or the Euclidean builder default.
    const fn global_topology_or_default(&self) -> GlobalTopology<D> {
        match self.requested_global_topology {
            Some(global_topology) => global_topology,
            None => GlobalTopology::DEFAULT,
        }
    }

    /// Builds the global topology metadata derived by periodic image-point construction.
    const fn periodic_image_global_topology(domain: ToroidalDomain<D>) -> GlobalTopology<D> {
        GlobalTopology::Toroidal {
            domain,
            mode: ToroidalConstructionMode::PeriodicImagePoint,
        }
    }

    /// Rejects topology metadata that would misclassify Euclidean construction boundaries.
    ///
    /// The plain Euclidean builder path does not create quotient-space neighbor
    /// links, so accepting closed metadata here would make boundary queries and
    /// Euler validation describe topology that was never assembled.
    const fn reject_euclidean_non_euclidean_topology(
        global_topology: GlobalTopology<D>,
    ) -> Result<(), DelaunayTriangulationConstructionError> {
        if global_topology.is_euclidean() {
            return Ok(());
        }

        Err(DelaunayTriangulationConstructionError::Triangulation(
            DelaunayConstructionFailure::EuclideanUnsupportedGlobalTopology {
                topology: global_topology.kind(),
            },
        ))
    }

    /// Rejects explicit metadata that conflicts with derived periodic topology.
    ///
    /// Periodic image-point construction derives the only valid closed toroidal
    /// metadata from its validated domain. An explicitly supplied matching value
    /// is harmless, but a different value would make the builder silently discard
    /// caller intent.
    fn reject_periodic_conflicting_global_topology(
        requested_global_topology: Option<GlobalTopology<D>>,
        derived_global_topology: GlobalTopology<D>,
    ) -> Result<(), DelaunayTriangulationConstructionError> {
        let Some(requested_global_topology) = requested_global_topology else {
            return Ok(());
        };

        if requested_global_topology == derived_global_topology {
            return Ok(());
        }

        Err(DelaunayTriangulationConstructionError::Triangulation(
            DelaunayConstructionFailure::PeriodicImageConflictingGlobalTopology {
                requested_topology: requested_global_topology.kind(),
                requested_mode: Self::toroidal_mode(requested_global_topology),
                requested_periods: Self::toroidal_periods(requested_global_topology),
                expected_mode: ToroidalConstructionMode::PeriodicImagePoint,
                expected_periods: Self::toroidal_periods(derived_global_topology)
                    .unwrap_or_default(),
            },
        ))
    }

    /// Extracts toroidal construction mode from topology metadata for diagnostics.
    const fn toroidal_mode(global_topology: GlobalTopology<D>) -> Option<ToroidalConstructionMode> {
        match global_topology {
            GlobalTopology::Toroidal { mode, .. } => Some(mode),
            GlobalTopology::Euclidean | GlobalTopology::Spherical | GlobalTopology::Hyperbolic => {
                None
            }
        }
    }

    /// Extracts toroidal periods from topology metadata for diagnostics.
    fn toroidal_periods(global_topology: GlobalTopology<D>) -> Option<Vec<f64>> {
        match global_topology {
            GlobalTopology::Toroidal { domain, .. } => Some(domain.periods().to_vec()),
            GlobalTopology::Euclidean | GlobalTopology::Spherical | GlobalTopology::Hyperbolic => {
                None
            }
        }
    }

    /// Builds a true periodic (toroidal) Delaunay triangulation using the 3^D image-point method.
    ///
    /// **Algorithm** (see module-level doc for periodic image-point details):
    /// 1. Validate: at least `2*D + 1` canonical vertices required.
    /// 2. Apply one tiny deterministic perturbation `δ_i` to each canonical
    ///    vertex, then build 3^D-1 image copies shifted by
    ///    `{-L_i, 0, +L_i}` per axis. Noncentral scaffolding images receive a
    ///    smaller construction-only joggle to escape finite-cover insertion degeneracy.
    /// 3. Build a full Euclidean DT on the expanded set (n * 3^D points).
    /// 4. Normalize lifted simplices to canonical quotient signatures.
    /// 5. Search for a closed candidate subset whose periodic facet incidences are valid.
    /// 6. Rebuild quotient representatives from that selection with periodic offsets.
    /// 7. Rebuild neighbor and incident-simplex associations and return the result.
    ///
    /// The output is a `Tds` whose `is_valid()` passes at Level 2 (structural validity).
    ///
    /// # References
    ///
    /// - `REFERENCES.md`, "Periodic and Toroidal Triangulations", first entry
    ///   (Caroli and Teillaud, "Computing 3D Periodic Triangulations").
    /// - CGAL, *2D Periodic Triangulations*:
    ///   <https://doc.cgal.org/latest/Periodic_2_triangulation_2/index.html>
    /// - CGAL, *3D Periodic Triangulations*:
    ///   <https://doc.cgal.org/latest/Periodic_3_triangulation_3/index.html>
    #[expect(
        clippy::too_many_lines,
        reason = "Image-point periodic DT algorithm is inherently multi-step; splitting would harm readability"
    )]
    fn build_periodic<K, M>(
        kernel: &K,
        canonical_vertices: &[Vertex<U, D>],
        topology_model: &M,
        topology_guarantee: TopologyGuarantee,
        construction_options: ConstructionOptions,
    ) -> Result<DelaunayTriangulation<K, U, V, D>, DelaunayTriangulationConstructionError>
    where
        K: Kernel<D, Scalar = f64>,
        M: GlobalTopologyModel<D>,
    {
        // Keep `build_periodic` self-protecting even if future call paths bypass outer validation.
        Self::validate_topology_model(topology_model)?;
        if D > 3 {
            return Err(
                TriangulationConstructionError::UnsupportedPeriodicDimension {
                    dimension: D,
                    max_validated_dimension: 3,
                    tracking_issue: 416,
                }
                .into(),
            );
        }
        if !topology_model.supports_periodic_facet_signatures() {
            return Err(
                TriangulationConstructionError::PeriodicImageUnsupportedTopology {
                    topology: topology_model.kind(),
                }
                .into(),
            );
        }

        let domain = topology_model.periodic_domain().ok_or_else(|| {
            TriangulationConstructionError::PeriodicImageMissingDomain {
                topology: topology_model.kind(),
            }
        })?;
        let domain_periods = domain.into_periods();
        let global_topology = GlobalTopology::Toroidal {
            domain,
            mode: ToroidalConstructionMode::PeriodicImagePoint,
        };
        let n = canonical_vertices.len();
        let min_points = 2 * D + 1;
        if n < min_points {
            return Err(
                TriangulationConstructionError::PeriodicImageInsufficientVertices {
                    dimension: D,
                    minimum_vertex_count: min_points,
                    actual_vertex_count: n,
                }
                .into(),
            );
        }

        // 3^D offset grid; zero-offset index = (3^D - 1) / 2.
        let image_width = 3_usize;
        let image_radius = 1_usize;
        let image_count = image_width.pow(u32::try_from(D).expect("dimension D fits in u32"));
        let zero_offset_idx = (image_count - 1) / 2;

        // Collect canonical UUIDs for key lookup after full DT is built.
        let canonical_uuids: Vec<Uuid> = canonical_vertices.iter().map(Vertex::uuid).collect();
        let perturb_units = |canon_idx: usize, axis: usize| -> i64 {
            let mut h = FNV_OFFSET_BASIS;
            h ^= u64::try_from(canon_idx).expect("canonical index fits in u64");
            h = h.wrapping_mul(FNV_PRIME);
            h ^= u64::try_from(axis).expect("axis index fits in u64");
            h = h.wrapping_mul(FNV_PRIME);
            let span = u64::try_from(2 * MAX_OFFSET_UNITS + 1).expect("span fits in u64");
            i64::try_from(h % span).expect("residue fits in i64") - MAX_OFFSET_UNITS
        };
        let image_jitter_units = |canon_idx: usize, axis: usize, image_idx: usize| -> i64 {
            let mut h = FNV_OFFSET_BASIS;
            h ^= u64::try_from(canon_idx).expect("canonical index fits in u64");
            h = h.wrapping_mul(FNV_PRIME);
            h ^= u64::try_from(axis).expect("axis index fits in u64");
            h = h.wrapping_mul(FNV_PRIME);
            h ^= u64::try_from(image_idx).expect("image index fits in u64");
            h = h.wrapping_mul(FNV_PRIME);
            let span = u64::try_from(2 * IMAGE_JITTER_UNITS + 1).expect("span fits in u64");
            i64::try_from(h % span).expect("residue fits in i64") - IMAGE_JITTER_UNITS
        };
        let canonical_f64: Vec<[f64; D]> = canonical_vertices
            .iter()
            .enumerate()
            .map(|(canon_idx, v)| {
                let orig_coords = v.point().coords();
                let mut coords = [0_f64; D];
                for i in 0..D {
                    let domain_i = domain_periods[i];
                    let orig = orig_coords[i]
                        .to_f64()
                        .expect("canonical coordinate is finite and convertible");
                    let normalized = (orig / domain_i).clamp(0.0, 1.0 - f64::EPSILON);
                    let u = (normalized * TWO_POW_52_F64)
                        .floor()
                        .to_i64()
                        .expect("grid index fits in i64");
                    let min_off = -u.min(MAX_OFFSET_UNITS);
                    let max_off = (TWO_POW_52_I64 - 1 - u).min(MAX_OFFSET_UNITS);
                    let off = perturb_units(canon_idx, i).clamp(min_off, max_off);
                    let adjusted_u = <f64 as num_traits::NumCast>::from(u + off)
                        .expect("adjusted grid index fits in f64");
                    coords[i] = (adjusted_u / TWO_POW_52_F64) * domain_i;
                }
                coords
            })
            .collect();

        let mut image_uuid_to_canonical_with_offset: FastHashMap<Uuid, (Uuid, [i8; D])> =
            FastHashMap::default();
        let mut expanded: Vec<Vertex<U, D>> = Vec::with_capacity(n.saturating_mul(image_count));
        for k in 0..image_count {
            let mut offset = [0i8; D];
            for (i, offset_val) in offset.iter_mut().enumerate() {
                let digit = (k / image_width
                    .pow(u32::try_from(i).expect("dimension index fits in u32")))
                    % image_width;
                *offset_val = i8::try_from(digit).expect("image digit fits in i8")
                    - i8::try_from(image_radius).expect("image radius fits in i8");
            }

            let is_canonical = k == zero_offset_idx;
            for (canon_idx, v) in canonical_vertices.iter().enumerate() {
                let mut new_coords = [0.0; D];
                for i in 0..D {
                    let shift_f64 = <f64 as From<i8>>::from(offset[i]) * domain_periods[i];
                    let jitter_f64 = if is_canonical {
                        0.0
                    } else {
                        let jitter_units = image_jitter_units(canon_idx, i, k);
                        (<f64 as num_traits::NumCast>::from(jitter_units)
                            .expect("jitter fits in f64")
                            / TWO_POW_52_F64)
                            * domain_periods[i]
                    };
                    new_coords[i] = canonical_f64[canon_idx][i] + shift_f64 + jitter_f64;
                }
                let new_point = Point::try_new(new_coords).map_err(|source| {
                    TriangulationConstructionError::PeriodicImageCoordinateValidation {
                        canonical_vertex_index: canon_idx,
                        image_index: k,
                        source,
                    }
                })?;
                if is_canonical {
                    image_uuid_to_canonical_with_offset.insert(v.uuid(), (v.uuid(), [0_i8; D]));
                    let canonical_v =
                        Vertex::from_validated_point_with_uuid(new_point, v.uuid(), v.data);
                    expanded.push(canonical_v);
                } else {
                    let image_v: Vertex<U, D> = Vertex::from_validated_point(new_point, None);
                    image_uuid_to_canonical_with_offset.insert(image_v.uuid(), (v.uuid(), offset));
                    expanded.push(image_v);
                }
            }
        }
        let expanded_base_options = construction_options
            .with_initial_simplex_strategy(InitialSimplexStrategy::Balanced)
            .with_insertion_validation_policy(ValidationPolicy::ExplicitOnly)
            .with_degenerate_realization_retries()
            .without_global_repair_fallback();
        let expanded_options = match construction_options.retry_policy() {
            RetryPolicy::Disabled => expanded_base_options,
            RetryPolicy::Shuffled { base_seed, .. }
            | RetryPolicy::DebugOnlyShuffled { base_seed, .. } => expanded_base_options
                .with_retry_policy(RetryPolicy::Shuffled {
                    attempts: NonZeroUsize::new(24).expect("literal is non-zero"),
                    base_seed,
                }),
        };
        let image_kernel = AdaptiveKernel::<f64>::new();
        let full_dt: DelaunayTriangulation<AdaptiveKernel<f64>, U, V, D> =
            DelaunayTriangulation::build_with_kernel_options(
                &image_kernel,
                &expanded,
                TopologyGuarantee::Pseudomanifold,
                expanded_options,
            )?;

        let uuid_to_key: FastHashMap<Uuid, VertexKey> = full_dt
            .vertices()
            .map(|(key, vertex)| (vertex.uuid(), key))
            .collect();

        // Map canonical UUIDs → VertexKeys in the full DT.
        let Some(central_keys) = canonical_uuids
            .iter()
            .map(|uuid| uuid_to_key.get(uuid).copied())
            .collect::<Option<Vec<_>>>()
        else {
            return Err(
                TriangulationConstructionError::PeriodicImageMissingCanonicalVertices {
                    canonical_vertex_count: canonical_uuids.len(),
                }
                .into(),
            );
        };
        let central_key_set: VertexKeySet = central_keys.into_iter().collect();

        // Map every full-DT vertex key to its canonical key and lattice offset.
        let mut vertex_key_to_lifted: FastHashMap<VertexKey, (VertexKey, [i8; D])> =
            FastHashMap::default();
        for (vk, vertex) in full_dt.vertices() {
            let Some((canonical_uuid, offset)) =
                image_uuid_to_canonical_with_offset.get(&vertex.uuid())
            else {
                continue;
            };
            let Some(canonical_key) = uuid_to_key.get(canonical_uuid).copied() else {
                continue;
            };
            vertex_key_to_lifted.insert(vk, (canonical_key, *offset));
        }

        let normalize_simplex_lifted =
            |simplex_key: SimplexKey| -> Option<Vec<(VertexKey, [i8; D])>> {
                let simplex = full_dt.simplex(simplex_key)?;
                let mut lifted: Vec<(VertexKey, [i8; D])> = simplex
                    .vertices()
                    .iter()
                    .map(|vk| vertex_key_to_lifted.get(vk).copied())
                    .collect::<Option<Vec<_>>>()?;

                let (anchor_idx, _) = lifted
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, lifted_vertex)| **lifted_vertex)?;
                let anchor_offset = lifted[anchor_idx].1;
                for (_, offset) in &mut lifted {
                    for axis in 0..D {
                        offset[axis] -= anchor_offset[axis];
                    }
                }

                Some(lifted)
            };
        let simplex_circumcenter_in_fundamental_domain = |simplex_key: SimplexKey| -> Option<bool> {
            let simplex = full_dt.simplex(simplex_key)?;
            let mut points: SmallBuffer<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE> =
                SmallBuffer::with_capacity(D + 1);
            for vk in simplex.vertices() {
                let vertex = full_dt.vertex(*vk)?;
                points.push(*vertex.point());
            }
            let center = circumcenter(&points).ok()?;
            for (axis, coord) in center.coords().iter().enumerate() {
                let center_coord = coord.to_f64()?;
                let period = domain_periods[axis];
                if !(center_coord >= 0.0 && center_coord < period) {
                    return Some(false);
                }
            }
            Some(true)
        };

        // Build unique symbolic candidates from all full-DT simplices.
        // Candidate tuple layout (see type alias):
        // (symbolic_signature, lifted_ordered, periodic_facet_keys, in_domain_hint)
        // where `lifted_ordered` preserves the observed per-simplex vertex order from
        // `normalize_simplex_lifted` (it is not canonical-key-sorted).
        let mut candidates_by_symbolic: FastHashMap<SymbolicSignature<D>, PeriodicCandidate<D>> =
            FastHashMap::default();
        for (ck, _) in full_dt.simplices() {
            let Some(lifted_vertices) = normalize_simplex_lifted(ck) else {
                continue;
            };
            let in_domain = simplex_circumcenter_in_fundamental_domain(ck).unwrap_or(false);
            let mut symbolic_signature = lifted_vertices.clone();
            symbolic_signature.sort_unstable();
            let lifted_ordered = lifted_vertices.clone();
            let mut periodic_facets: Vec<OrientedPeriodicFacet> = Vec::with_capacity(D + 1);
            for facet_idx in 0..=D {
                let facet_key = Self::derive_periodic_facet_key(&lifted_ordered, facet_idx)?;
                let orientation_side = periodic_facet_orientation_side(&lifted_ordered, facet_idx);
                periodic_facets.push((facet_key, orientation_side));
            }

            if let Some(existing) = candidates_by_symbolic.get_mut(&symbolic_signature) {
                if in_domain {
                    existing.3 = true;
                }
            } else {
                candidates_by_symbolic.insert(
                    symbolic_signature.clone(),
                    (
                        symbolic_signature,
                        lifted_ordered,
                        periodic_facets,
                        in_domain,
                    ),
                );
            }
        }
        let mut candidates: Vec<PeriodicCandidate<D>> =
            candidates_by_symbolic.into_values().collect();
        if candidates.is_empty() {
            return Err(
                TriangulationConstructionError::PeriodicQuotientNoCandidates {
                    full_simplex_count: full_dt.number_of_simplices(),
                    canonical_vertex_count: central_key_set.len(),
                }
                .into(),
            );
        }
        candidates.sort_by(|a, b| b.3.cmp(&a.3).then_with(|| a.0.cmp(&b.0)));

        let (search_attempts, search_seed) = match construction_options.retry_policy() {
            RetryPolicy::Disabled => (1, 0xD1CE_0B5E_2100_0001_u64),
            RetryPolicy::Shuffled {
                attempts,
                base_seed,
            }
            | RetryPolicy::DebugOnlyShuffled {
                attempts,
                base_seed,
            } => (
                attempts
                    .get()
                    .saturating_add(1)
                    .saturating_mul(512)
                    .clamp(512, 4096),
                base_seed.unwrap_or(0xD1CE_0B5E_2100_0001_u64),
            ),
        };

        let mut best_selected: Vec<bool> = Vec::new();
        let mut best_boundary_count = usize::MAX;
        let mut best_selected_count = 0_usize;
        let mut best_coverage_count = 0_usize;
        let mut best_abs_chi = i64::MAX;
        if D > 2 {
            let selected: Vec<bool> = candidates.iter().map(|candidate| candidate.3).collect();
            let mut facet_incidences: FastHashMap<PeriodicFacetKey, [u8; 2]> =
                FastHashMap::default();
            let mut covered: VertexKeySet = VertexKeySet::default();
            let mut has_compatible_facet_orientations = true;
            for (idx, is_selected) in selected.iter().copied().enumerate() {
                if !is_selected {
                    continue;
                }
                let candidate_facets = &candidates[idx].2;
                if !periodic_candidate_can_be_added(candidate_facets, &facet_incidences) {
                    has_compatible_facet_orientations = false;
                    break;
                }
                update_periodic_facet_incidences(candidate_facets, &mut facet_incidences, true);
                for (vertex_key, _) in &candidates[idx].1 {
                    covered.insert(*vertex_key);
                }
            }
            if has_compatible_facet_orientations
                && facet_incidences
                    .values()
                    .all(|&incidence| periodic_facet_incidence_total(incidence) == 2)
                && covered.len() == central_key_set.len()
            {
                best_boundary_count = 0;
                best_selected_count = selected.iter().filter(|&&is_selected| is_selected).count();
                best_coverage_count = covered.len();
                best_abs_chi = 0;
                best_selected = selected;
            }
        } else if D == 2 {
            let target_faces = central_key_set.len().saturating_mul(2);
            let mut edge_to_index: FastHashMap<PeriodicFacetKey, usize> = FastHashMap::default();
            let mut candidate_edges: Vec<[(usize, bool); 3]> = Vec::with_capacity(candidates.len());
            let mut candidate_vertices: Vec<Vec<VertexKey>> = Vec::with_capacity(candidates.len());
            let mut candidate_in_domain: Vec<bool> = Vec::with_capacity(candidates.len());

            for candidate in &candidates {
                let mut edge_indices = [(0usize, false); 3];
                for (slot, &(edge_key, side)) in candidate.2.iter().enumerate() {
                    let next_index = edge_to_index.len();
                    let edge_index = *edge_to_index.entry(edge_key).or_insert(next_index);
                    edge_indices[slot] = (edge_index, side);
                }
                candidate_edges.push(edge_indices);
                candidate_vertices.push(
                    candidate
                        .1
                        .iter()
                        .map(|(vertex_key, _)| *vertex_key)
                        .collect(),
                );
                candidate_in_domain.push(candidate.3);
            }
            let exact_search_node_limit = candidate_edges
                .len()
                .saturating_mul(edge_to_index.len().max(1))
                .saturating_mul(512)
                .clamp(100_000, 5_000_000);

            if let Some(exact_selected) = search_closed_2d_selection(
                &candidate_edges,
                &candidate_vertices,
                &candidate_in_domain,
                target_faces,
                central_key_set.len(),
                edge_to_index.len(),
                exact_search_node_limit,
            ) {
                best_selected_count = exact_selected
                    .iter()
                    .filter(|&&is_selected| is_selected)
                    .count();
                best_coverage_count = central_key_set.len();
                best_boundary_count = 0;
                best_abs_chi = 0;
                best_selected = exact_selected;
            }
        }

        if D == 3 && best_selected.is_empty() {
            let exact_search_node_limit = candidates
                .len()
                .saturating_mul(D + 1)
                .saturating_mul(4096)
                .clamp(250_000, 5_000_000);
            if let Some(exact_selected) = search_closed_connected_periodic_selection(
                &candidates,
                central_key_set.len(),
                exact_search_node_limit,
            ) {
                best_selected_count = exact_selected
                    .iter()
                    .filter(|&&is_selected| is_selected)
                    .count();
                best_coverage_count = central_key_set.len();
                best_boundary_count = 0;
                best_abs_chi = 0;
                best_selected = exact_selected;
            }
        }

        if best_selected.is_empty() {
            let base_order: Vec<usize> = (0..candidates.len()).collect();
            for attempt_idx in 0..search_attempts {
                let mut order = base_order.clone();
                if attempt_idx > 0 {
                    let attempt_u64 =
                        u64::try_from(attempt_idx).expect("attempt index fits in u64");
                    let mut rng = StdRng::seed_from_u64(
                        search_seed.wrapping_add(attempt_u64.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                    );
                    order.shuffle(&mut rng);
                }
                // Keep in-domain representatives first while preserving randomized tie-breaks.
                order.sort_by(|a, b| candidates[*b].3.cmp(&candidates[*a].3));

                let mut selected = vec![false; candidates.len()];
                let mut facet_incidences: FastHashMap<PeriodicFacetKey, [u8; 2]> =
                    FastHashMap::default();

                // Pass 1: greedy maximal subset with at most one incidence on
                // each oriented side of a canonical periodic facet.
                for idx in order.iter().copied() {
                    let candidate_facets = &candidates[idx].2;
                    if !periodic_candidate_can_be_added(candidate_facets, &facet_incidences) {
                        continue;
                    }
                    selected[idx] = true;
                    update_periodic_facet_incidences(candidate_facets, &mut facet_incidences, true);
                }

                // Pass 2: only add simplices that strictly reduce boundary facets (count == 1).
                let mut improved = true;
                while improved {
                    improved = false;
                    for idx in order.iter().copied() {
                        if selected[idx] {
                            continue;
                        }
                        let candidate_facets = &candidates[idx].2;
                        if !periodic_candidate_can_be_added(candidate_facets, &facet_incidences) {
                            continue;
                        }

                        let boundary_delta =
                            periodic_boundary_delta(candidate_facets, &facet_incidences, true);

                        if boundary_delta < 0 {
                            selected[idx] = true;
                            update_periodic_facet_incidences(
                                candidate_facets,
                                &mut facet_incidences,
                                true,
                            );
                            improved = true;
                        }
                    }
                }
                // Pass 3: local refinement with both add and remove moves.
                // This escapes add-only local minima in D>2 where closure requires swaps.
                loop {
                    let mut best_move: Option<(bool, usize, i32)> = None;
                    for idx in order.iter().copied() {
                        let candidate_facets = &candidates[idx].2;
                        if selected[idx] {
                            let boundary_delta =
                                periodic_boundary_delta(candidate_facets, &facet_incidences, false);
                            if boundary_delta < 0
                                && best_move
                                    .is_none_or(|(_, _, best_delta)| boundary_delta < best_delta)
                            {
                                best_move = Some((false, idx, boundary_delta));
                            }
                        } else {
                            if !periodic_candidate_can_be_added(candidate_facets, &facet_incidences)
                            {
                                continue;
                            }

                            let boundary_delta =
                                periodic_boundary_delta(candidate_facets, &facet_incidences, true);
                            if boundary_delta < 0
                                && best_move
                                    .is_none_or(|(_, _, best_delta)| boundary_delta < best_delta)
                            {
                                best_move = Some((true, idx, boundary_delta));
                            }
                        }
                    }

                    let Some((is_add, idx, _)) = best_move else {
                        break;
                    };
                    let candidate_facets = &candidates[idx].2;
                    if is_add {
                        selected[idx] = true;
                        update_periodic_facet_incidences(
                            candidate_facets,
                            &mut facet_incidences,
                            true,
                        );
                    } else {
                        selected[idx] = false;
                        update_periodic_facet_incidences(
                            candidate_facets,
                            &mut facet_incidences,
                            false,
                        );
                    }
                }

                if D > 2 {
                    selected = best_connected_periodic_component(&selected, &candidates);
                    facet_incidences.clear();
                    for (idx, is_selected) in selected.iter().copied().enumerate() {
                        if is_selected {
                            update_periodic_facet_incidences(
                                &candidates[idx].2,
                                &mut facet_incidences,
                                true,
                            );
                        }
                    }
                }

                let boundary_count = facet_incidences
                    .values()
                    .filter(|&&incidence| periodic_facet_incidence_total(incidence) == 1)
                    .count();
                let selected_count = selected.iter().filter(|&&is_selected| is_selected).count();
                let mut covered: VertexKeySet = VertexKeySet::default();
                for (idx, is_selected) in selected.iter().copied().enumerate() {
                    if !is_selected {
                        continue;
                    }
                    for (vertex_key, _) in &candidates[idx].1 {
                        covered.insert(*vertex_key);
                    }
                }
                let coverage_count = covered.len();
                let abs_chi = if D == 2 {
                    let v_count =
                        i64::try_from(central_key_set.len()).expect("vertex count fits in i64");
                    let e_count = i64::try_from(facet_incidences.len())
                        .expect("edge/facet count fits in i64");
                    let f_count = i64::try_from(selected_count).expect("simplex count fits in i64");
                    (v_count - e_count + f_count).abs()
                } else {
                    0
                };
                if boundary_count < best_boundary_count
                    || (boundary_count == best_boundary_count
                        && (if D == 2 {
                            abs_chi < best_abs_chi
                                || (abs_chi == best_abs_chi && selected_count > best_selected_count)
                        } else {
                            coverage_count > best_coverage_count
                                || (coverage_count == best_coverage_count
                                    && selected_count > best_selected_count)
                        }))
                {
                    best_boundary_count = boundary_count;
                    best_selected_count = selected_count;
                    best_coverage_count = coverage_count;
                    best_abs_chi = abs_chi;
                    best_selected = selected;
                }
                if D == 2 && best_boundary_count == 0 && best_abs_chi == 0 {
                    break;
                }
            }
        }

        if best_selected.is_empty() {
            return Err(
                TriangulationConstructionError::PeriodicQuotientSelectionEmpty {
                    candidate_count: candidates.len(),
                    search_attempts,
                }
                .into(),
            );
        }
        if D == 2 && best_boundary_count > 0 {
            return Err(
                TriangulationConstructionError::PeriodicQuotientSelectionBoundaryFacets {
                    boundary_facet_count: best_boundary_count,
                    search_attempts,
                    full_vertex_count: full_dt.number_of_vertices(),
                    full_simplex_count: full_dt.number_of_simplices(),
                    canonical_vertex_count: central_key_set.len(),
                    candidate_count: candidates.len(),
                    selected_simplex_count: best_selected_count,
                }
                .into(),
            );
        }
        if D == 2 && best_abs_chi != 0 {
            return Err(
                TriangulationConstructionError::PeriodicQuotientSelectionEulerCharacteristic {
                    best_abs_chi,
                    search_attempts,
                }
                .into(),
            );
        }
        let has_full_canonical_coverage = best_coverage_count == central_key_set.len();
        if D > 2 && !has_full_canonical_coverage {
            return Err(
                TriangulationConstructionError::PeriodicQuotientSelectionIncompleteCoverage {
                    dimension: D,
                    covered_vertex_count: best_coverage_count,
                    canonical_vertex_count: central_key_set.len(),
                }
                .into(),
            );
        }
        let mut representative_lifted_by_symbolic: FastHashMap<
            SymbolicSignature<D>,
            SymbolicSignature<D>,
        > = FastHashMap::default();
        for (idx, is_selected) in best_selected.iter().copied().enumerate() {
            if !is_selected {
                continue;
            }
            let (symbolic_signature, lifted_ordered, _, _) = &candidates[idx];
            representative_lifted_by_symbolic
                .insert(symbolic_signature.clone(), lifted_ordered.clone());
        }

        // Clone TDS and rebuild simplex complex from quotient representatives.
        let tds_ref = full_dt.tds();
        let mut tds_mut = tds_ref.clone();

        // Remove all simplices first.
        let all_simplices: Vec<SimplexKey> = tds_mut.simplex_keys().collect();
        tds_mut
            .remove_simplices_by_keys(&all_simplices)
            .map_err(periodic_quotient_tds_mutation_error)?;

        // Remove all image vertices.
        let image_vertex_keys: Vec<VertexKey> = tds_mut
            .vertex_keys()
            .filter(|vk| !central_key_set.contains(vk))
            .collect();
        for &vk in &image_vertex_keys {
            tds_mut
                .remove_vertex(vk)
                .map_err(periodic_quotient_tds_mutation_error)?;
        }

        // Insert quotient simplices.
        let mut signatures_sorted: Vec<Vec<(VertexKey, [i8; D])>> =
            representative_lifted_by_symbolic.keys().cloned().collect();
        signatures_sorted.sort_unstable();

        let mut inserted_simplex_keys: Vec<SimplexKey> =
            Vec::with_capacity(signatures_sorted.len());
        let mut rep_lifted_by_key: FastHashMap<SimplexKey, Vec<(VertexKey, [i8; D])>> =
            FastHashMap::default();

        for signature in signatures_sorted {
            let Some(lifted_vertices) = representative_lifted_by_symbolic.get(&signature) else {
                continue;
            };
            let canonical_vertex_keys: Vec<VertexKey> =
                lifted_vertices.iter().map(|(ck, _)| *ck).collect();
            let offsets: PeriodicOffsetBuffer<D> =
                lifted_vertices.iter().map(|(_, offset)| *offset).collect();
            let simplex =
                Simplex::try_new_periodic(canonical_vertex_keys, offsets).map_err(|e| {
                    TriangulationConstructionError::PeriodicQuotientSimplexCreation { source: e }
                })?;
            let ck = tds_mut
                .insert_simplex_with_mapping_trusted_vertices(simplex)
                .map_err(TriangulationConstructionError::Tds)?;
            inserted_simplex_keys.push(ck);
            rep_lifted_by_key.insert(ck, lifted_vertices.clone());
        }
        if inserted_simplex_keys.is_empty() {
            return Err(TriangulationConstructionError::PeriodicQuotientEmptyReconstruction.into());
        }

        // Sanity-check periodic facet multiplicities before neighbor rewiring.
        // In a valid simplicial manifold each facet is incident to at most two simplices.
        let mut periodic_facet_counts: FastHashMap<PeriodicFacetKey, usize> =
            FastHashMap::default();
        for lifted in rep_lifted_by_key.values() {
            for facet_idx in 0..=D {
                let periodic_facet_key = Self::derive_periodic_facet_key(lifted, facet_idx)?;
                *periodic_facet_counts.entry(periodic_facet_key).or_insert(0) += 1;
            }
        }
        let overloaded_facets: Vec<(PeriodicFacetKey, usize)> = periodic_facet_counts
            .into_iter()
            .filter(|(_, count)| *count > 2)
            .collect();
        if !overloaded_facets.is_empty() {
            return Err(
                TriangulationConstructionError::PeriodicQuotientOverloadedFacets {
                    overloaded_facet_count: overloaded_facets.len(),
                    selected_simplex_count: rep_lifted_by_key.len(),
                }
                .into(),
            );
        }

        // Rebuild neighbor pointers by pairing equal symbolic facet signatures in the quotient.
        let mut neighbor_updates: FastHashMap<SimplexKey, Vec<Option<SimplexKey>>> =
            inserted_simplex_keys
                .iter()
                .copied()
                .map(|ck| (ck, vec![None; D + 1]))
                .collect();

        let mut facet_occurrences: FastHashMap<PeriodicFacetKey, Vec<(SimplexKey, usize)>> =
            FastHashMap::default();
        for &rep_ck in &inserted_simplex_keys {
            let Some(lifted) = rep_lifted_by_key.get(&rep_ck) else {
                continue;
            };
            for facet_idx in 0..=D {
                let sig = Self::derive_periodic_facet_key(lifted, facet_idx)?;
                facet_occurrences
                    .entry(sig)
                    .or_default()
                    .push((rep_ck, facet_idx));
            }
        }

        for (_facet_sig, occurrences) in facet_occurrences {
            match occurrences.as_slice() {
                [(a_ck, a_idx), (b_ck, b_idx)] => {
                    neighbor_updates.get_mut(a_ck).ok_or_else(|| {
                        TriangulationConstructionError::PeriodicQuotientMissingNeighborVector {
                            simplex_key: *a_ck,
                        }
                    })?[*a_idx] = Some(*b_ck);
                    neighbor_updates.get_mut(b_ck).ok_or_else(|| {
                        TriangulationConstructionError::PeriodicQuotientMissingNeighborVector {
                            simplex_key: *b_ck,
                        }
                    })?[*b_idx] = Some(*a_ck);
                }
                [(a_ck, a_idx)] => {
                    // Self-identified periodic facet.
                    neighbor_updates.get_mut(a_ck).ok_or_else(|| {
                        TriangulationConstructionError::PeriodicQuotientMissingNeighborVector {
                            simplex_key: *a_ck,
                        }
                    })?[*a_idx] = Some(*a_ck);
                }
                _ => {
                    return Err(
                        TriangulationConstructionError::PeriodicQuotientFacetMultiplicity {
                            occurrence_count: occurrences.len(),
                        }
                        .into(),
                    );
                }
            }
        }

        let unmatched_count = neighbor_updates
            .values()
            .flat_map(|n| n.iter())
            .filter(|n| n.is_none())
            .count();
        if unmatched_count > 0 {
            return Err(
                TriangulationConstructionError::PeriodicQuotientUnmatchedNeighbors {
                    unmatched_neighbor_slots: unmatched_count,
                }
                .into(),
            );
        }

        // Apply neighbor updates.
        for &ck in &inserted_simplex_keys {
            let neighbors = neighbor_updates.remove(&ck).ok_or_else(|| {
                TriangulationConstructionError::PeriodicQuotientMissingNeighborVector {
                    simplex_key: ck,
                }
            })?;
            tds_mut
                .set_neighbors_by_key(ck, &neighbors)
                .map_err(periodic_quotient_tds_mutation_error)?;
        }

        // Canonicalize quotient-simplex orientation after symbolic neighbor reconstruction.
        // Contradictory quotient parity is a typed construction failure.
        tds_mut
            .normalize_coherent_orientation()
            .map_err(TdsMutationError::from)
            .map_err(periodic_quotient_tds_mutation_error)?;
        // Rebuild incident-simplex pointers after topology surgery.
        tds_mut
            .assign_incident_simplices()
            .map_err(periodic_quotient_tds_mutation_error)?;
        if let Err(e) = tds_mut.is_valid() {
            return Err(TriangulationConstructionError::FinalTopologyValidation {
                context: FinalTopologyValidationContext::PeriodicQuotientTopology,
                source: Box::new(InvariantError::Tds(e)),
            }
            .into());
        }

        let candidate = DelaunayTriangulationCandidate::assemble(
            tds_mut,
            kernel.clone(),
            topology_guarantee,
            global_topology,
        );
        let proof = candidate.validate_tds_structure().map_err(|source| {
            TriangulationConstructionError::FinalTopologyValidation {
                context: FinalTopologyValidationContext::PeriodicQuotientTopology,
                source: Box::new(InvariantError::Tds(source)),
            }
        })?;
        Ok(candidate.into_structurally_valid_delaunay(proof))
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::construction::{DelaunayConstructionFailure, InsertionOrderStrategy};
    use crate::core::algorithms::incremental_insertion::{
        TdsConstructionFailure, TdsValidationFailure,
    };
    use crate::core::construction::PeriodicQuotientFacetKeyDerivationFailure;
    use crate::core::simplex::SimplexValidationError;
    use crate::core::tds::TdsConstructionError;
    use crate::geometry::kernel::RobustKernel;
    use crate::topology::traits::GlobalTopologyModelError;
    use crate::topology::traits::global_topology_model::{
        EuclideanModel, GlobalTopologyModel, ToroidalModel,
    };
    use crate::topology::traits::topological_space::{
        GlobalTopology, TopologyKind, ToroidalConstructionMode, ToroidalDomain, ToroidalDomainError,
    };
    use crate::vertex;
    use approx::assert_relative_eq;
    use slotmap::Key;
    use std::assert_matches;

    fn toroidal_model<const D: usize>(
        domain: [f64; D],
        mode: ToroidalConstructionMode,
    ) -> ToroidalModel<D> {
        ToroidalModel::new(ToroidalDomain::try_new(domain).unwrap(), mode)
    }

    fn assert_invalid_toroidal_domain_error(
        err: ToroidalDomainError,
        expected_axis: usize,
        expected_period: f64,
    ) {
        assert_matches!(
            err,
            ToroidalDomainError::InvalidPeriod { axis, period }
                if axis == expected_axis && period.to_bits() == expected_period.to_bits()
        );
    }

    fn periodic_fixture_vertices_t2() -> Vec<Vertex<(), 2>> {
        vec![
            vertex!([0.1_f64, 0.2]).unwrap(),
            vertex!([0.4, 0.7]).unwrap(),
            vertex!([0.7, 0.3]).unwrap(),
            vertex!([0.2, 0.9]).unwrap(),
            vertex!([0.8, 0.6]).unwrap(),
            vertex!([0.5, 0.1]).unwrap(),
            vertex!([0.3, 0.5]).unwrap(),
        ]
    }

    /// Builds one synthetic periodic candidate for quotient-selection tests.
    fn periodic_selection_candidate<const D: usize>(
        lifted_vertices: &[(VertexKey, [i8; D])],
        facets: &[OrientedPeriodicFacet],
        in_domain: bool,
    ) -> PeriodicCandidate<D> {
        let lifted_vertices = lifted_vertices.to_vec();
        (
            lifted_vertices.clone(),
            lifted_vertices,
            facets.to_vec(),
            in_domain,
        )
    }

    #[derive(Clone, Copy, Debug)]
    struct ValidationFailureModel;

    impl GlobalTopologyModel<2> for ValidationFailureModel {
        fn kind(&self) -> TopologyKind {
            TopologyKind::Euclidean
        }

        fn allows_boundary(&self) -> bool {
            true
        }

        fn validate_configuration(&self) -> Result<(), GlobalTopologyModelError> {
            Err(GlobalTopologyModelError::NonFiniteCoordinate {
                axis: 0,
                value: f64::NAN,
            })
        }

        fn canonicalize_point_in_place(
            &self,
            _coords: &mut [f64; 2],
        ) -> Result<(), GlobalTopologyModelError> {
            Ok(())
        }

        fn lift_for_orientation(
            &self,
            coords: [f64; 2],
            periodic_offset: Option<[i8; 2]>,
        ) -> Result<[f64; 2], GlobalTopologyModelError> {
            if periodic_offset.is_some() {
                return Err(GlobalTopologyModelError::PeriodicOffsetsUnsupported {
                    kind: TopologyKind::Euclidean,
                });
            }
            Ok(coords)
        }
    }

    #[test]
    fn explicit_simplex_creation_error_preserves_typed_source() {
        let err = ExplicitConstructionError::SimplexCreation {
            simplex_index: 7,
            source: SimplexValidationError::DuplicateVertices,
        };

        let ExplicitConstructionError::SimplexCreation {
            simplex_index,
            source,
        } = &err
        else {
            panic!("expected simplex creation error, got {err:?}");
        };

        assert_eq!(*simplex_index, 7);
        assert_eq!(*source, SimplexValidationError::DuplicateVertices);
        assert!(err.to_string().contains("Simplex 7"));
        assert!(err.to_string().contains("Duplicate vertices"));
    }

    #[derive(Clone, Copy, Debug)]
    struct CanonicalizationFailureModel;

    impl GlobalTopologyModel<2> for CanonicalizationFailureModel {
        fn kind(&self) -> TopologyKind {
            TopologyKind::Euclidean
        }

        fn allows_boundary(&self) -> bool {
            true
        }

        fn validate_configuration(&self) -> Result<(), GlobalTopologyModelError> {
            Ok(())
        }

        fn canonicalize_point_in_place(
            &self,
            _coords: &mut [f64; 2],
        ) -> Result<(), GlobalTopologyModelError> {
            Err(GlobalTopologyModelError::NonFiniteCoordinate {
                axis: 0,
                value: f64::NAN,
            })
        }

        fn lift_for_orientation(
            &self,
            coords: [f64; 2],
            periodic_offset: Option<[i8; 2]>,
        ) -> Result<[f64; 2], GlobalTopologyModelError> {
            if periodic_offset.is_some() {
                return Err(GlobalTopologyModelError::PeriodicOffsetsUnsupported {
                    kind: TopologyKind::Euclidean,
                });
            }
            Ok(coords)
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct MissingPeriodicDomainModel;

    impl GlobalTopologyModel<2> for MissingPeriodicDomainModel {
        fn kind(&self) -> TopologyKind {
            TopologyKind::Toroidal
        }

        fn allows_boundary(&self) -> bool {
            false
        }

        fn validate_configuration(&self) -> Result<(), GlobalTopologyModelError> {
            Ok(())
        }

        fn canonicalize_point_in_place(
            &self,
            _coords: &mut [f64; 2],
        ) -> Result<(), GlobalTopologyModelError> {
            Ok(())
        }

        fn lift_for_orientation(
            &self,
            coords: [f64; 2],
            _periodic_offset: Option<[i8; 2]>,
        ) -> Result<[f64; 2], GlobalTopologyModelError> {
            Ok(coords)
        }

        fn supports_periodic_facet_signatures(&self) -> bool {
            true
        }

        fn periodic_domain(&self) -> Option<ToroidalDomain<2>> {
            None
        }
    }

    // -------------------------------------------------------------------------
    // Euclidean path — `new` is specialized for f64/(), no type annotations needed
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_euclidean_2d() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .build()
            .unwrap();
        assert_eq!(dt.number_of_vertices(), 3);
        assert_eq!(dt.dim(), 2);
        assert!(dt.as_triangulation().validate().is_ok());
    }

    #[test]
    fn test_builder_euclidean_3d() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .build()
            .unwrap();
        assert_eq!(dt.number_of_vertices(), 4);
        assert!(dt.validate().is_ok());
    }

    #[test]
    fn test_builder_euclidean_rejects_non_euclidean_global_topology() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let result = DelaunayTriangulationBuilder::new(&vertices)
            .global_topology(GlobalTopology::Spherical)
            .build();

        assert_matches!(
            result,
            Err(DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::EuclideanUnsupportedGlobalTopology {
                    topology: TopologyKind::Spherical,
                }
            ))
        );
    }

    #[test]
    fn test_builder_statistics_rejects_non_euclidean_global_topology_with_default_stats() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let err = DelaunayTriangulationBuilder::new(&vertices)
            .global_topology(GlobalTopology::Spherical)
            .build_with_statistics()
            .unwrap_err();

        assert_matches!(
            err.error,
            DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::EuclideanUnsupportedGlobalTopology {
                    topology: TopologyKind::Spherical,
                }
            )
        );
        assert_eq!(err.statistics.inserted, 0);
        assert_eq!(err.statistics.skipped_duplicate, 0);
        assert_eq!(err.statistics.skipped_degeneracy, 0);
        assert_eq!(err.statistics.total_attempts, 0);
        assert!(err.statistics.attempts_histogram.is_empty());
        assert!(err.statistics.slow_insertions.is_empty());
        assert!(err.statistics.skip_samples.is_empty());
        assert_eq!(err.statistics.telemetry.insertion_wall_time_calls, 0);
        assert_eq!(err.statistics.telemetry.construction_preprocessing_nanos, 0);
        assert_eq!(err.statistics.telemetry.construction_insert_loop_nanos, 0);
        assert_eq!(err.statistics.telemetry.construction_finalize_nanos, 0);
    }

    #[test]
    fn test_builder_topology_guarantee_propagated() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .topology_guarantee(TopologyGuarantee::Pseudomanifold)
            .build()
            .unwrap();
        assert_eq!(dt.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
    }

    #[test]
    fn test_builder_custom_options_propagated() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let opts =
            ConstructionOptions::default().with_insertion_order(InsertionOrderStrategy::Input);
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .construction_options(opts)
            .build()
            .unwrap();
        assert_eq!(dt.number_of_vertices(), 3);
    }

    #[test]
    fn test_builder_toroidal_invalid_domain_is_error() {
        let vertices = vec![
            vertex!([0.1, 0.2]).unwrap(),
            vertex!([0.4, 0.7]).unwrap(),
            vertex!([0.7, 0.3]).unwrap(),
            vertex!([0.2, 0.9]).unwrap(),
            vertex!([0.8, 0.6]).unwrap(),
            vertex!([0.5, 0.1]).unwrap(),
            vertex!([0.3, 0.5]).unwrap(),
        ];
        let Err(err) = DelaunayTriangulationBuilder::new(&vertices).try_toroidal([1.0, 0.0]) else {
            panic!("zero period should be rejected");
        };
        assert_invalid_toroidal_domain_error(err, 1, 0.0);
    }

    #[test]
    fn test_builder_toroidal_t2_smoke() {
        let vertices = periodic_fixture_vertices_t2();
        let n = vertices.len();
        let kernel = RobustKernel::new();
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .try_toroidal([1.0, 1.0])
            .unwrap()
            .build_with_kernel(&kernel)
            .unwrap();
        assert_eq!(dt.number_of_vertices(), n);
        assert!(dt.is_valid_structure().is_ok());
        assert_matches!(
            dt.global_topology(),
            GlobalTopology::Toroidal {
                mode: ToroidalConstructionMode::PeriodicImagePoint,
                ..
            }
        );
    }

    #[test]
    fn test_builder_toroidal_rejects_dimension_above_validated_range() {
        let vertices = vec![vertex!([0.1_f64, 0.2, 0.3, 0.4]).unwrap()];
        let result = DelaunayTriangulationBuilder::new(&vertices)
            .try_toroidal([1.0, 1.0, 1.0, 1.0])
            .unwrap()
            .build();

        assert_matches!(
            result,
            Err(DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::UnsupportedPeriodicDimension {
                    dimension: 4,
                    max_validated_dimension: 3,
                    tracking_issue: 416,
                }
            ))
        );
    }

    #[test]
    fn test_builder_toroidal_rejects_global_topology_before_toroidal_setter() {
        let vertices = periodic_fixture_vertices_t2();
        let result = DelaunayTriangulationBuilder::new(&vertices)
            .global_topology(GlobalTopology::Spherical)
            .try_toroidal([1.0, 1.0])
            .unwrap()
            .build();

        assert_matches!(
            result,
            Err(DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::PeriodicImageConflictingGlobalTopology {
                    requested_topology: TopologyKind::Spherical,
                    requested_mode: None,
                    requested_periods: None,
                    expected_mode: ToroidalConstructionMode::PeriodicImagePoint,
                    expected_periods,
                }
            )) if expected_periods.as_slice() == [1.0, 1.0]
        );
    }

    #[test]
    fn test_builder_toroidal_rejects_global_topology_after_toroidal_setter() {
        let vertices = periodic_fixture_vertices_t2();
        let result = DelaunayTriangulationBuilder::new(&vertices)
            .try_toroidal([1.0, 1.0])
            .unwrap()
            .global_topology(GlobalTopology::Euclidean)
            .build();

        assert_matches!(
            result,
            Err(DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::PeriodicImageConflictingGlobalTopology {
                    requested_topology: TopologyKind::Euclidean,
                    requested_mode: None,
                    requested_periods: None,
                    expected_mode: ToroidalConstructionMode::PeriodicImagePoint,
                    expected_periods,
                }
            )) if expected_periods.as_slice() == [1.0, 1.0]
        );
    }

    #[test]
    fn test_builder_toroidal_rejects_conflicting_explicit_toroidal_mode() {
        let vertices = periodic_fixture_vertices_t2();
        let requested_topology =
            GlobalTopology::try_toroidal([1.0, 1.0], ToroidalConstructionMode::Explicit).unwrap();
        let result = DelaunayTriangulationBuilder::new(&vertices)
            .try_toroidal([1.0, 1.0])
            .unwrap()
            .global_topology(requested_topology)
            .build();

        assert_matches!(
            result,
            Err(DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::PeriodicImageConflictingGlobalTopology {
                    requested_topology: TopologyKind::Toroidal,
                    requested_mode: Some(ToroidalConstructionMode::Explicit),
                    requested_periods: Some(requested_periods),
                    expected_mode: ToroidalConstructionMode::PeriodicImagePoint,
                    expected_periods,
                }
            )) if requested_periods.as_slice() == [1.0, 1.0]
                && expected_periods.as_slice() == [1.0, 1.0]
        );
    }

    #[test]
    fn test_builder_toroidal_rejects_conflicting_explicit_toroidal_domain() {
        let vertices = periodic_fixture_vertices_t2();
        let requested_topology =
            GlobalTopology::try_toroidal([2.0, 1.0], ToroidalConstructionMode::PeriodicImagePoint)
                .unwrap();
        let result = DelaunayTriangulationBuilder::new(&vertices)
            .try_toroidal([1.0, 1.0])
            .unwrap()
            .global_topology(requested_topology)
            .build();

        assert_matches!(
            result,
            Err(DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::PeriodicImageConflictingGlobalTopology {
                    requested_topology: TopologyKind::Toroidal,
                    requested_mode: Some(ToroidalConstructionMode::PeriodicImagePoint),
                    requested_periods: Some(requested_periods),
                    expected_mode: ToroidalConstructionMode::PeriodicImagePoint,
                    expected_periods,
                }
            )) if requested_periods.as_slice() == [2.0, 1.0]
                && expected_periods.as_slice() == [1.0, 1.0]
        );
    }

    #[test]
    fn test_builder_toroidal_accepts_matching_explicit_global_topology() {
        let vertices = periodic_fixture_vertices_t2();
        let topology =
            GlobalTopology::try_toroidal([1.0, 1.0], ToroidalConstructionMode::PeriodicImagePoint)
                .unwrap();
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .try_toroidal([1.0, 1.0])
            .unwrap()
            .global_topology(topology)
            .build()
            .unwrap();

        assert_eq!(dt.global_topology(), topology);
        assert!(dt.validate().is_ok());
    }

    // -------------------------------------------------------------------------
    // Generic builder path
    // -------------------------------------------------------------------------

    /// `new` accepts vertices carrying user data (`U ≠ ()`).
    #[test]
    fn test_builder_new_preserves_vertex_data() {
        let vertices: Vec<Vertex<i32, 2>> = vec![
            vertex!([0.2_f64, 0.3]; data = 1_i32).unwrap(),
            vertex!([0.8_f64, 0.1]; data = 2_i32).unwrap(),
            vertex!([0.5_f64, 0.7]; data = 3_i32).unwrap(),
        ];
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .build()
            .unwrap();

        assert_eq!(dt.number_of_vertices(), 3);
        let mut data: Vec<i32> = dt.vertices().filter_map(|(_, v)| v.data).collect();
        data.sort_unstable();
        assert_eq!(data, vec![1, 2, 3]);
    }

    #[test]
    fn test_builder_with_robust_kernel() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let kernel = RobustKernel::<f64>::new();
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .build_with_kernel(&kernel)
            .unwrap();
        assert_eq!(dt.number_of_vertices(), 4);
        assert!(dt.validate().is_ok());
    }

    // -------------------------------------------------------------------------
    // Private helper function tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_topology_model_accepts_valid_toroidal() {
        let model = toroidal_model::<2>([2.0, 3.0], ToroidalConstructionMode::PeriodicImagePoint);
        let result = DelaunayTriangulationBuilder::<(), 2>::validate_topology_model(&model);
        assert!(result.is_ok());
    }

    #[test]
    fn test_derive_periodic_facet_key_happy_path_matches_core_derivation() {
        let lifted_ordered = vec![
            (VertexKey::null(), [0_i8, 0_i8]),
            (VertexKey::null(), [1_i8, 0_i8]),
            (VertexKey::null(), [0_i8, 1_i8]),
        ];
        let expected = periodic_facet_key_from_lifted_vertices::<2>(&lifted_ordered, 1).unwrap();
        let derived =
            DelaunayTriangulationBuilder::<(), 2>::derive_periodic_facet_key(&lifted_ordered, 1)
                .unwrap();
        assert_eq!(derived, expected);
    }

    #[test]
    fn test_derive_periodic_facet_key_maps_errors_to_typed_quotient_variant() {
        let lifted_ordered = vec![
            (VertexKey::null(), [-128_i8, 0_i8]),
            (VertexKey::null(), [0_i8, 0_i8]),
            (VertexKey::null(), [127_i8, 0_i8]),
        ];
        let err =
            DelaunayTriangulationBuilder::<(), 2>::derive_periodic_facet_key(&lifted_ordered, 1)
                .unwrap_err();
        let DelaunayTriangulationConstructionError::Triangulation(
            DelaunayConstructionFailure::PeriodicQuotientFacetKeyDerivation {
                facet_index,
                reason,
            },
        ) = err
        else {
            panic!("expected PeriodicQuotientFacetKeyDerivation mapping, got: {err:?}");
        };

        assert_eq!(facet_index, 1);
        assert_matches!(
            reason,
            PeriodicQuotientFacetKeyDerivationFailure::RelativeOffsetOutOfRange {
                axis: 0,
                component: 383,
            }
        );
    }

    #[test]
    fn test_periodic_facet_incidence_updates_are_reversible() {
        let candidate_facets = [(10, false), (10, true), (11, false)];
        let mut facet_incidences = FastHashMap::default();

        assert!(periodic_candidate_can_be_added(
            &candidate_facets,
            &facet_incidences
        ));
        assert_eq!(
            periodic_boundary_delta(&candidate_facets, &facet_incidences, true),
            1
        );

        update_periodic_facet_incidences(&candidate_facets, &mut facet_incidences, true);
        assert_eq!(facet_incidences.get(&10), Some(&[1, 1]));
        assert_eq!(facet_incidences.get(&11), Some(&[1, 0]));
        assert!(!periodic_candidate_can_be_added(
            &candidate_facets,
            &facet_incidences
        ));
        assert!(!periodic_candidate_can_be_added(
            &[(12, false), (12, false)],
            &FastHashMap::default()
        ));
        assert_eq!(
            periodic_boundary_delta(&candidate_facets, &facet_incidences, false),
            -1
        );

        update_periodic_facet_incidences(&candidate_facets, &mut facet_incidences, false);
        assert!(facet_incidences.is_empty());
    }

    #[test]
    fn test_best_connected_periodic_component_ranks_quotient_invariants() {
        let vertex_key = |value| VertexKey::from(slotmap::KeyData::from_ffi(value));
        let [a, b, c] = [1_u64, 2, 3].map(vertex_key);
        let two_vertex_coverage = [(a, [0, 0]), (a, [1, 0]), (b, [0, 0])];
        let full_coverage = [(a, [0, 0]), (b, [0, 0]), (c, [0, 0])];
        let candidates = vec![
            periodic_selection_candidate(
                &two_vertex_coverage,
                &[(10, false), (11, false), (12, false)],
                true,
            ),
            periodic_selection_candidate(
                &two_vertex_coverage,
                &[(10, true), (11, true), (12, true)],
                true,
            ),
            periodic_selection_candidate(
                &full_coverage,
                &[(20, false), (21, false), (22, false)],
                false,
            ),
            periodic_selection_candidate(
                &full_coverage,
                &[(20, true), (21, true), (23, true)],
                false,
            ),
            periodic_selection_candidate(
                &full_coverage,
                &[(30, false), (31, false), (32, false)],
                false,
            ),
            periodic_selection_candidate(
                &full_coverage,
                &[(30, true), (31, true), (32, true)],
                false,
            ),
            periodic_selection_candidate(
                &full_coverage,
                &[(40, false), (41, false), (42, false)],
                true,
            ),
            periodic_selection_candidate(
                &full_coverage,
                &[(40, true), (41, true), (42, true)],
                false,
            ),
        ];

        assert_eq!(
            best_connected_periodic_component(&vec![true; candidates.len()], &candidates),
            vec![false, false, false, false, false, false, true, true]
        );
    }

    #[test]
    fn test_closed_periodic_selection_backtracks_and_obeys_budget() {
        let vertex_key = |value| VertexKey::from(slotmap::KeyData::from_ffi(value));
        let [a, b] = [1_u64, 2].map(vertex_key);
        let seed_vertices = [(a, [0, 0]), (a, [1, 0]), (a, [2, 0])];
        let covering_vertices = [(b, [0, 0]), (b, [1, 0]), (b, [2, 0])];
        let forward_facets = [(0, false), (1, false), (2, false)];
        let reverse_facets = [(0, true), (1, true), (2, true)];
        let candidates = vec![
            periodic_selection_candidate(&seed_vertices, &forward_facets, true),
            periodic_selection_candidate(&seed_vertices, &reverse_facets, false),
            periodic_selection_candidate(&covering_vertices, &reverse_facets, false),
        ];

        let selected = search_closed_connected_periodic_selection(&candidates, 2, 100)
            .expect("the search should backtrack to the candidate completing vertex coverage");
        assert_eq!(selected, vec![true, false, true]);

        assert!(
            search_closed_connected_periodic_selection(&candidates, 2, 1).is_none(),
            "the node budget must bound exact quotient selection"
        );
        assert!(
            search_closed_connected_periodic_selection(&candidates[..1], 1, 100).is_none(),
            "an open periodic component must not be accepted"
        );
    }

    #[test]
    fn test_closed_2d_selection_requires_opposite_oriented_edge_incidences() {
        let vertex_key = |value| VertexKey::from(slotmap::KeyData::from_ffi(value));
        let [a, b, c, d] = [1_u64, 2, 3, 4].map(vertex_key);
        let first_face = [(0, false), (1, false), (2, false)];
        let closed_faces = [first_face, [(0, true), (1, true), (2, true)]];
        let connected_vertices = [vec![a, b, a], vec![a, b, b]];
        let selected = search_closed_2d_selection(
            &closed_faces,
            &connected_vertices,
            &[true, false],
            2,
            2,
            3,
            100,
        )
        .expect("opposite edge incidences should close the quotient");
        assert_eq!(selected, vec![true, true]);

        let same_sided_faces = [first_face, first_face];
        assert!(
            search_closed_2d_selection(
                &same_sided_faces,
                &connected_vertices,
                &[true, false],
                2,
                2,
                3,
                100,
            )
            .is_none(),
            "same-sided incidences must not be mistaken for a closed quotient",
        );

        let open_faces = [first_face, [(3, true), (4, true), (5, true)]];
        assert!(
            search_closed_2d_selection(
                &open_faces,
                &connected_vertices,
                &[true, false],
                2,
                2,
                6,
                100,
            )
            .is_none(),
            "a face-count target alone must not pass with unmatched edges",
        );

        let incomplete_vertices = [vec![a, a, a], vec![a, a, a]];
        assert!(
            search_closed_2d_selection(
                &closed_faces,
                &incomplete_vertices,
                &[true, false],
                2,
                2,
                3,
                100,
            )
            .is_none(),
            "a closed subset must cover every canonical vertex",
        );

        let disconnected_faces = [
            first_face,
            [(0, true), (1, true), (2, true)],
            [(3, false), (4, false), (5, false)],
            [(3, true), (4, true), (5, true)],
        ];
        let disconnected_vertices = [vec![a, b, a], vec![a, b, b], vec![c, d, c], vec![c, d, d]];
        assert!(
            search_closed_2d_selection(
                &disconnected_faces,
                &disconnected_vertices,
                &[true, false, true, false],
                4,
                4,
                6,
                1_000,
            )
            .is_none(),
            "closed components must not be mistaken for one connected quotient",
        );
    }

    #[test]
    fn test_periodic_quotient_tds_mutation_error_preserves_typed_tds_source() {
        let source = TdsError::InconsistentDataStructure {
            message: "incidence rollback failed".to_owned(),
        };
        let err = periodic_quotient_tds_mutation_error(TdsMutationError::from(source.clone()));

        assert_matches!(
            &err,
            TriangulationConstructionError::Tds(TdsConstructionError::ValidationError(tds_source))
                if tds_source == &source
        );

        let public_failure = DelaunayConstructionFailure::from(err);
        assert_matches!(
            public_failure,
            DelaunayConstructionFailure::Tds {
                reason: TdsConstructionFailure::Validation {
                    reason: TdsValidationFailure::InconsistentDataStructure { message },
                },
            } if message == "incidence rollback failed"
        );
    }

    #[test]
    fn test_validate_topology_model_accepts_euclidean() {
        let model = EuclideanModel;
        let result = DelaunayTriangulationBuilder::<(), 2>::validate_topology_model(&model);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_topology_model_maps_non_period_errors() {
        let result =
            DelaunayTriangulationBuilder::<(), 2>::validate_topology_model(&ValidationFailureModel);
        let err = result.expect_err("non-period validation failure should be mapped");
        assert_matches!(
            err,
            DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::TopologyModelConfiguration {
                    source: GlobalTopologyModelError::NonFiniteCoordinate {
                        axis: 0,
                        value,
                    },
                },
            ) if value.is_nan()
        );
    }

    #[test]
    fn test_canonicalize_vertices_preserves_uuids() {
        let vertices = vec![
            vertex!([2.5, 3.7]).unwrap(),
            vertex!([1.8, -0.5]).unwrap(),
            vertex!([0.5, 0.7]).unwrap(),
        ];
        let original_uuids: Vec<_> = vertices.iter().map(Vertex::uuid).collect();
        let model = toroidal_model::<2>([2.0, 3.0], ToroidalConstructionMode::PeriodicImagePoint);
        let canonical =
            DelaunayTriangulationBuilder::<(), 2>::canonicalize_vertices(&vertices, &model)
                .unwrap();

        assert_eq!(canonical.len(), vertices.len());
        let canonical_uuids: Vec<_> = canonical.iter().map(Vertex::uuid).collect();
        assert_eq!(canonical_uuids, original_uuids);
    }

    #[test]
    fn test_canonicalize_vertices_preserves_data() {
        let vertices: Vec<Vertex<i32, 2>> = vec![
            vertex!([2.5_f64, 3.7]; data = 10_i32).unwrap(),
            vertex!([1.8_f64, -0.5]; data = 20_i32).unwrap(),
            vertex!([0.5_f64, 0.7]; data = 30_i32).unwrap(),
        ];
        let model = toroidal_model::<2>([2.0, 3.0], ToroidalConstructionMode::PeriodicImagePoint);
        let canonical =
            DelaunayTriangulationBuilder::<i32, 2>::canonicalize_vertices(&vertices, &model)
                .unwrap();

        assert_eq!(canonical.len(), vertices.len());
        for (orig, canon) in vertices.iter().zip(canonical.iter()) {
            assert_eq!(orig.data, canon.data);
        }
    }

    #[test]
    fn test_canonicalize_vertices_transforms_coordinates() {
        let vertices = vec![
            vertex!([2.5, 3.7]).unwrap(),  // → (0.5, 0.7)
            vertex!([1.8, -0.5]).unwrap(), // → (1.8, 2.5)
            vertex!([0.3, 0.2]).unwrap(),  // → (0.3, 0.2)
        ];
        let model = toroidal_model::<2>([2.0, 3.0], ToroidalConstructionMode::PeriodicImagePoint);
        let canonical =
            DelaunayTriangulationBuilder::<(), 2>::canonicalize_vertices(&vertices, &model)
                .unwrap();

        assert_eq!(canonical.len(), 3);
        assert_relative_eq!(canonical[0].point().coords()[0], 0.5);
        assert_relative_eq!(canonical[0].point().coords()[1], 0.7);
        assert_relative_eq!(canonical[1].point().coords()[0], 1.8);
        assert_relative_eq!(canonical[1].point().coords()[1], 2.5);
        assert_relative_eq!(canonical[2].point().coords()[0], 0.3);
        assert_relative_eq!(canonical[2].point().coords()[1], 0.2);
    }

    #[test]
    fn test_canonicalize_vertices_includes_vertex_context_on_error() {
        let vertices = vec![
            vertex!([0.25_f64, 0.75_f64]).unwrap(),
            vertex!([0.9_f64, 0.1_f64]).unwrap(),
        ];
        let result = DelaunayTriangulationBuilder::<(), 2>::canonicalize_vertices(
            &vertices,
            &CanonicalizationFailureModel,
        );
        let err = result.expect_err("canonicalization failure should be reported");
        assert_matches!(
            err,
            DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::VertexCanonicalization {
                    vertex_index: 0,
                    source: GlobalTopologyModelError::NonFiniteCoordinate {
                        axis: 0,
                        value,
                    },
                },
            ) if value.is_nan()
        );
    }

    #[test]
    fn test_build_periodic_requires_periodic_domain() {
        let kernel = AdaptiveKernel::new();
        let canonical_vertices = vec![
            vertex!([0.1_f64, 0.1_f64]).unwrap(),
            vertex!([0.9_f64, 0.2_f64]).unwrap(),
            vertex!([0.2_f64, 0.8_f64]).unwrap(),
            vertex!([0.7_f64, 0.9_f64]).unwrap(),
            vertex!([0.5_f64, 0.4_f64]).unwrap(),
        ];
        let result = DelaunayTriangulationBuilder::<(), 2>::build_periodic::<_, _>(
            &kernel,
            &canonical_vertices,
            &MissingPeriodicDomainModel,
            TopologyGuarantee::default(),
            ConstructionOptions::default(),
        );
        let err = result.expect_err("missing periodic domain must fail");
        assert_matches!(
            err,
            DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::PeriodicImageMissingDomain {
                    topology: TopologyKind::Toroidal,
                }
            )
        );
    }

    #[test]
    fn test_canonicalize_vertices_euclidean_identity() {
        let vertices = vec![
            vertex!([1.5, 2.5]).unwrap(),
            vertex!([3.7, 4.2]).unwrap(),
            vertex!([-1.0, -2.0]).unwrap(),
        ];
        let model = EuclideanModel;
        let canonical =
            DelaunayTriangulationBuilder::<(), 2>::canonicalize_vertices(&vertices, &model)
                .unwrap();

        assert_eq!(canonical.len(), vertices.len());
        for (orig, canon) in vertices.iter().zip(canonical.iter()) {
            assert_relative_eq!(orig.point().coords()[0], canon.point().coords()[0]);
            assert_relative_eq!(orig.point().coords()[1], canon.point().coords()[1]);
        }
    }

    #[test]
    fn test_canonicalize_vertices_propagates_nan_error() {
        assert!(Point::<2>::try_new([f64::NAN, 0.5]).is_err());
    }

    #[test]
    fn test_canonicalize_vertices_propagates_infinity_error() {
        assert!(Point::<2>::try_new([f64::INFINITY, 0.5]).is_err());
    }

    #[test]
    fn test_canonicalize_vertices_includes_original_coords_in_error() {
        assert!(Point::<2>::try_new([f64::NAN, 1.5]).is_err());
    }
}
