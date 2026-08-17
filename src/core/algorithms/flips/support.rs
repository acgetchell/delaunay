//! Shared local topology, periodic-frame, and predicate helpers.

#![forbid(unsafe_code)]

use super::{
    Cow, DataType, FlipContextError, FlipError, FlipPredicateError, GlobalTopologyModel,
    GlobalTopologyModelAdapter, Key, MAX_PRACTICAL_DIMENSION_SIZE, Point, Simplex, SimplexKey,
    SimplexKeyBuffer, SmallBuffer, Tds, VertexKey, env, repair_trace_enabled,
    stable_hash_u64_slice,
};

/// Extracts facet vertices by omitted slot so facet hashing matches the simplex's
/// current vertex ordering.
pub(super) fn facet_vertices_from_simplex<V, const D: usize>(
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
pub(super) fn ridge_vertices_from_simplex<V, const D: usize>(
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
pub(super) fn simplex_extras_for_ridge<V, const D: usize>(
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
pub(super) fn missing_opposite_for_simplex(
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
pub(super) fn collect_simplices_around_ridge<U, V, const D: usize>(
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
pub(super) fn vertex_point<U, V, const D: usize>(
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
pub(super) struct EuclideanPointCache<const D: usize> {
    points: SmallBuffer<(VertexKey, Point<D>), MAX_PRACTICAL_DIMENSION_SIZE>,
}

impl<const D: usize> EuclideanPointCache<D> {
    /// Starts an empty cache for one local predicate evaluation.
    pub(super) fn new() -> Self {
        Self {
            points: SmallBuffer::new(),
        }
    }
}

impl<const D: usize> EuclideanPointCache<D> {
    /// Returns a cached Euclidean point, loading it from the TDS on first use.
    pub(super) fn point<U, V>(
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
    pub(super) fn points_for_vertices<U, V>(
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
pub(super) fn vertices_to_points<U, V, const D: usize>(
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
pub(super) fn vertices_to_points_with_optional_lift<U, V, const D: usize>(
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
pub(super) fn vertex_point_with_optional_lift<U, V, const D: usize>(
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
pub(super) fn vertex_point_lifted_into_simplex<U, V, const D: usize>(
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
pub(super) fn periodic_offset_lifted_into_simplex<U, V, const D: usize>(
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
pub(super) fn lift_vertex_point<U, V, const D: usize>(
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
        .map_err(|source| FlipPredicateError::PeriodicVertexLift { vertex_key, source })?;
    Point::try_new(lifted_coords).map_err(|source| {
        FlipPredicateError::PeriodicLiftedPointValidation { vertex_key, source }.into()
    })
}

/// Looks up the offset paired with a vertex slot, preserving the invariant that
/// periodic offsets are indexed exactly like simplex vertices.
pub(super) fn periodic_offset_for_simplex_vertex<U, V, const D: usize>(
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
pub(super) fn periodic_offsets_or_zero_frame<V, const D: usize>(
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
pub(super) fn validate_periodic_offset_len<V, const D: usize>(
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
pub(super) fn shared_vertex_indices<V, const D: usize>(
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
pub(super) fn align_periodic_offset<const D: usize>(
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
pub(super) fn matching_source_simplex<U, V, const D: usize>(
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
pub(super) fn removed_simplex_frame(
    source_simplices: &[SimplexKey],
) -> Result<SimplexKey, FlipError> {
    source_simplices
        .first()
        .copied()
        .ok_or_else(|| FlipContextError::MissingRemovedSimplexFrame.into())
}

#[derive(Debug, Default)]
pub(super) struct FlipTopologyIndex {
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
pub(super) struct CandidateFacetInfo {
    existing_count: u8,
    last_simplex: Option<SimplexKey>,
}

/// Sorts stable slotmap key values before hashing so signatures are independent
/// of local simplex vertex order.
pub(super) fn sorted_vertex_key_values(
    vertices: &[VertexKey],
) -> SmallBuffer<u64, MAX_PRACTICAL_DIMENSION_SIZE> {
    let mut key_values: SmallBuffer<u64, MAX_PRACTICAL_DIMENSION_SIZE> =
        vertices.iter().map(|key| key.data().as_ffi()).collect();
    key_values.sort_unstable();
    key_values
}

/// Hashes a complete simplex vertex set for duplicate-simplex detection during flips.
pub(super) fn simplex_signature(vertices: &[VertexKey]) -> u64 {
    let key_values = sorted_vertex_key_values(vertices);
    stable_hash_u64_slice(&key_values)
}

/// Builds the small topology index needed to reject duplicate simplices and
/// non-manifold internal facets without repeated global scans.
pub(super) fn build_flip_topology_index<U, V, const D: usize>(
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
pub(super) fn flip_would_duplicate_simplex_any<U, V, const D: usize>(
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
pub(super) fn flip_would_create_nonmanifold_facets_any(
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

#[cfg(test)]
mod tests {
    use super::super::support::vertex_point_with_optional_lift;
    use super::super::*;
    use super::*;
    use crate::geometry::kernel::AdaptiveKernel;
    use crate::topology::traits::topological_space::ToroidalConstructionMode;
    use crate::vertex;
    use approx::assert_relative_eq;
    use std::assert_matches;
    use std::iter::once;

    fn unit_vector<const D: usize>(index: usize) -> [f64; D] {
        let mut coords = [0.0; D];
        coords[index] = 1.0;
        coords
    }
    fn insert_standard_simplex_vertices<const D: usize>(
        tds: &mut Tds<(), (), D>,
    ) -> Vec<VertexKey> {
        let mut vertices = Vec::with_capacity(D + 1);
        vertices.push(
            tds.insert_vertex_with_mapping(vertex!([0.0; D]).unwrap())
                .unwrap(),
        );
        for axis in 0..D {
            vertices.push(
                tds.insert_vertex_with_mapping(vertex!(unit_vector::<D>(axis)).unwrap())
                    .unwrap(),
            );
        }
        vertices
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

    #[test]
    fn k1_periodic_preflight_uses_lifted_repeated_vertex_slots() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let repeated_vertex = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let other_vertex = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let simplex = Simplex::try_new_periodic(
            vec![repeated_vertex, other_vertex, repeated_vertex],
            vec![[0, 0], [0, 0], [1, 0]],
        )
        .expect("distinct lifted identities should form a periodic simplex");
        let simplex_key = tds.insert_simplex_with_mapping(simplex).unwrap();
        let topology_model = toroidal_model::<2>();
        let interior_candidate = vertex!([0.25, 0.25]).unwrap();

        let feasibility = validate_bistellar_flip_k1_insert(
            &tds,
            &topology_model,
            simplex_key,
            &interior_candidate,
        )
        .expect("lifted replacement simplices should be non-degenerate");
        assert_eq!(feasibility.kind, BistellarFlipKind::k1(2));

        let boundary_candidate = vertex!([0.5, 0.0]).unwrap();
        assert_eq!(
            validate_bistellar_flip_k1_insert(
                &tds,
                &topology_model,
                simplex_key,
                &boundary_candidate,
            )
            .expect_err("candidate on a lifted facet should be degenerate"),
            FlipError::DegenerateSimplex
        );

        let exterior_candidate = vertex!([0.9, 0.25]).unwrap();

        let error = validate_bistellar_flip_k1_insert(
            &tds,
            &topology_model,
            simplex_key,
            &exterior_candidate,
        )
        .expect_err("candidate should cross the facet opposite slot zero");

        assert_eq!(
            FlipFailureKind::from(&error),
            FlipFailureKind::K1InsertionOutsideSimplex
        );
        assert_matches!(
            error,
            FlipError::K1InsertionOutsideSimplex {
                simplex_key: rejected,
                opposite_vertex,
                opposite_vertex_index: 0,
            } if rejected == simplex_key && opposite_vertex == repeated_vertex
        );
    }

    #[test]
    fn k1_periodic_preflight_reports_candidate_lift_overflow() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let vertices = vec![
            tds.insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
                .unwrap(),
            tds.insert_vertex_with_mapping(vertex!([0.0, 0.5]).unwrap())
                .unwrap(),
            tds.insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
                .unwrap(),
        ];
        let simplex = Simplex::try_new_periodic(vertices, vec![[1, 0], [0, 0], [0, 0]]).unwrap();
        let simplex_key = tds.insert_simplex_with_mapping(simplex).unwrap();
        let topology_model = GlobalTopology::try_toroidal(
            [f64::MAX, 1.0],
            ToroidalConstructionMode::PeriodicImagePoint,
        )
        .unwrap()
        .model();
        let candidate = vertex!([f64::MAX, 0.25]).unwrap();

        let error =
            validate_bistellar_flip_k1_insert(&tds, &topology_model, simplex_key, &candidate)
                .expect_err("candidate lift should overflow its selected lattice sheet");
        assert_matches!(
            error,
            FlipError::PredicateFailure { reason }
                if matches!(
                    reason.as_ref(),
                    FlipPredicateError::K1InsertedVertexLift {
                        simplex_key: rejected,
                        source: GlobalTopologyModelError::NonFiniteCoordinate { axis: 0, value },
                    } if *rejected == simplex_key && value.is_infinite()
                )
        );
    }

    #[test]
    fn k1_periodic_preflight_rejects_degenerate_lifted_source() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let vertices = vec![
            tds.insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
                .unwrap(),
            tds.insert_vertex_with_mapping(vertex!([0.25, 0.0]).unwrap())
                .unwrap(),
            tds.insert_vertex_with_mapping(vertex!([0.75, 0.0]).unwrap())
                .unwrap(),
        ];
        let simplex = Simplex::try_new_periodic(vertices, vec![[0, 0]; 3]).unwrap();
        let simplex_key = tds.insert_simplex_with_mapping(simplex).unwrap();
        let topology_model = toroidal_model::<2>();
        let candidate = vertex!([0.5, 0.25]).unwrap();

        assert_eq!(
            validate_bistellar_flip_k1_insert(&tds, &topology_model, simplex_key, &candidate,)
                .expect_err("degenerate lifted source should fail preflight"),
            FlipError::DegenerateSimplex
        );
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
                        .insert_vertex_with_mapping(vertex!(extra_coords).unwrap())
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
                tds.insert_vertex_with_mapping(vertex!(coords).unwrap())
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
                        .insert_vertex_with_mapping(vertex!(unit_vector::<$dim>(0)).unwrap())
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
                        .insert_vertex_with_mapping(vertex!([0.0; $dim]).unwrap())
                        .unwrap();
                    let lifted_vertex = tds
                        .insert_vertex_with_mapping(vertex!(unit_vector::<$dim>(0)).unwrap())
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
                        .insert_vertex_with_mapping(vertex!([0.0; $dim]).unwrap())
                        .unwrap();
                    let mut shared_b_coords = [0.0; $dim];
                    shared_b_coords[0] = 0.2;
                    let shared_b = tds
                        .insert_vertex_with_mapping(vertex!(shared_b_coords).unwrap())
                        .unwrap();
                    let lifted_vertex = tds
                        .insert_vertex_with_mapping(vertex!(unit_vector::<$dim>(0)).unwrap())
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

    fn skewed_point<const D: usize>() -> [f64; D] {
        let mut coords = [0.0; D];
        for (i, coord) in coords.iter_mut().enumerate().take(D) {
            let idx = f64::from(u32::try_from(i + 1).expect("index fits in u32"));
            *coord = 0.11 * idx;
        }
        coords
    }

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
                tds.insert_vertex_with_mapping(vertex!(unit_vector::<D>(axis)).unwrap())
                    .unwrap(),
            );
        }
        let opposite_a = tds
            .insert_vertex_with_mapping(vertex!([0.0; D]).unwrap())
            .unwrap();
        let opposite_b = tds
            .insert_vertex_with_mapping(vertex!([0.25; D]).unwrap())
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
                tds.insert_vertex_with_mapping(vertex!(unit_vector::<D>(axis)).unwrap())
                    .unwrap(),
            );
        }
        let a = tds
            .insert_vertex_with_mapping(vertex!([0.0; D]).unwrap())
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(vertex!(unit_vector::<D>(D - 1)).unwrap())
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(vertex!(skewed_point::<D>()).unwrap())
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
        let truncated_removed_simplices: SimplexKeyBuffer = once(frame_simplex).collect();
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
}
