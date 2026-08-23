//! Replacement orientation, local diagnostics, and k=3 predicate support.

#![forbid(unsafe_code)]

use super::{
    DataType, FacetHandle, FastHashSet, FlipContext, FlipContextError, FlipError,
    FlipPredicateError, FlipPredicateOperation, Key, LastAppliedFlip, MAX_PRACTICAL_DIMENSION_SIZE,
    Orientation, PeriodicOffsetBuffer, RepairDiagnostics, ReplacementPeriodicOffsets, RidgeHandle,
    Simplex, SimplexKey, SimplexKeyBuffer, SmallBuffer, Tds, VertexKey, align_periodic_offset,
    collect_simplices_around_ridge, periodic_offset_lifted_into_simplex,
    periodic_offsets_or_zero_frame, push_unique_simplex_key, ridge_vertices_from_simplex,
    robust_orientation, should_emit_postcondition_facet_debug, should_emit_ridge_debug,
    simplex_extras_for_ridge, validate_periodic_offset_len, vertices_to_points,
};

/// Chooses replacement-simplex parity from the oriented cavity boundary.
pub(super) fn orient_replacement_simplices<U, V, const D: usize>(
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
pub(super) fn assign_external_replacement_orientation<U, V, const D: usize>(
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
pub(super) fn set_flip_assignment(
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
pub(super) fn replacement_simplex_periodic_offsets<U, V, const D: usize>(
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
pub(super) fn replacement_periodic_source_simplices(
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
pub(super) fn replacement_sources_use_periodic_offsets<U, V, const D: usize>(
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
pub(super) fn source_simplices_contain_vertex<U, V, const D: usize>(
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
pub(super) fn new_vertex_periodic_offset_in_frame<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    target_simplex_key: SimplexKey,
) -> Result<[i8; D], FlipError> {
    let target_simplex = tds
        .simplex(target_simplex_key)
        .ok_or(FlipError::MissingSimplex {
            simplex_key: target_simplex_key,
        })?;
    k1_inserted_vertex_periodic_offset(target_simplex_key, target_simplex)
}

/// Selects the local lattice sheet used by both k=1 preflight and mutation planning.
pub(super) fn k1_inserted_vertex_periodic_offset<V, const D: usize>(
    simplex_key: SimplexKey,
    simplex: &Simplex<V, D>,
) -> Result<[i8; D], FlipError> {
    let offsets = periodic_offsets_or_zero_frame(simplex_key, simplex)?;
    Ok(offsets.first().copied().unwrap_or([0_i8; D]))
}

/// Finds the target facet opposite the source facet, if the simplices share it.
pub(super) fn matching_facet_index(
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
pub(super) fn shared_facet_indices(
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
pub(super) fn unique_vertex_index(vertices: &[VertexKey], other: &[VertexKey]) -> Option<usize> {
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
pub(super) fn facet_orders_coherent(
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
pub(super) struct PeriodicFacetParityContext<'a, const D: usize> {
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
pub(super) fn facet_orders_coherent_with_periodic_offsets<const D: usize>(
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
pub(super) fn facet_order_with_offsets<const D: usize>(
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
pub(super) fn normalized_facet_order_with_offsets<const D: usize>(
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
pub(super) fn align_periodic_facet_order<const D: usize>(
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
pub(super) fn facet_order(
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
pub(super) fn permutation_odd<Id: PartialEq>(
    source_order: &[Id],
    target_order: &[Id],
) -> Option<bool> {
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
pub(super) fn validate_replacement_orientation<U, V, const D: usize>(
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
pub(super) fn simplices_containing_vertices<U, V, const D: usize>(
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
pub(super) fn debug_ridge_context<U, V, const D: usize>(
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
pub(super) fn ridge_incident_simplex_summary<U, V, const D: usize>(
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
pub(super) fn ridge_neighbor_simplices_for_simplex<V, const D: usize>(
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
pub(super) fn predecessor_flip_summary<U, V, const D: usize>(
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
pub(super) fn simplex_vertex_summary<U, V, const D: usize>(
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
pub(super) fn debug_postcondition_facet_context<U, V, const D: usize>(
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
pub(super) fn facet_incident_simplex_summary<U, V, const D: usize>(
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
pub(super) fn postcondition_facet_predecessor_summary<U, V, const D: usize>(
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

#[cfg(test)]
mod tests {
    use super::super::repair_queue::RIDGE_DEBUG_LIMIT_DEFAULT;
    use super::super::test_support::init_tracing;
    use super::super::*;
    use super::*;
    use crate::core::algorithms::insertion::repair_neighbor_pointers;
    use crate::vertex;
    use slotmap::KeyData;
    use std::assert_matches;
    use std::iter::once;
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
    /// Converts vertex-key slices into the fixed-capacity buffer used by flip helpers.
    fn vertex_key_buffer(
        vertices: &[VertexKey],
    ) -> SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> {
        vertices.iter().copied().collect()
    }
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
                .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0]).unwrap())
                .unwrap();
            let x_axis_vertex = tds
                .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0]).unwrap())
                .unwrap();
            let y_axis_vertex = tds
                .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0]).unwrap())
                .unwrap();
            let upper_apex_vertex = tds
                .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0]).unwrap())
                .unwrap();
            let lower_apex_vertex = tds
                .insert_vertex_with_mapping(vertex!([0.0, 0.0, -1.0]).unwrap())
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
                    kind: BistellarFlipKind::from_validated(2, 3),
                    direction: FlipDirection::Forward,
                    removed_simplices: once(self.upper_tetrahedron).collect(),
                    new_simplices: once(self.lower_neighbor).collect(),
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
                    let v_square = tds.insert_vertex_with_mapping(vertex!([1.0; $dim]).unwrap()).unwrap();
                    let v_collinear = tds
                        .insert_vertex_with_mapping(vertex!(scaled_unit_vector::<$dim>(0, 2.0)).unwrap())
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
                    let v_square = tds.insert_vertex_with_mapping(vertex!([1.0; $dim]).unwrap()).unwrap();
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
                        .insert_vertex_with_mapping(vertex!(scaled_unit_vector::<$dim>(0, 2.0)).unwrap())
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
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let v_edge_end = tds
            .insert_vertex_with_mapping(vertex!([2.0, 0.0, 0.0]).unwrap())
            .unwrap();

        let v_cycle_0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 2.0, 0.0]).unwrap())
            .unwrap();
        let v_cycle_1 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 2.0]).unwrap())
            .unwrap();
        let v_cycle_2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 2.0, 2.0]).unwrap())
            .unwrap();

        let v_external = tds
            .insert_vertex_with_mapping(vertex!([-1.0, 1.0, 1.0]).unwrap())
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

        let info = apply_bistellar_flip_raw(&mut tds, &ctx).unwrap();

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
}
