//! Shared permutation parity for TDS validation and flip orientation.

use crate::core::collections::{MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer};

/// Returns whether the permutation from `source_order` to `target_order` is odd.
///
/// Each source occurrence matches the first unused equal target occurrence.
/// This defines a stable matching when IDs repeat. Returns `None` if the
/// lengths or ID multiplicities differ; empty orders have even parity.
pub(crate) fn permutation_odd<Id: PartialEq>(
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

    let mut is_odd = false;
    for i in 0..target_positions.len() {
        for j in (i + 1)..target_positions.len() {
            if target_positions[i] > target_positions[j] {
                is_odd = !is_odd;
            }
        }
    }

    Some(is_odd)
}

#[cfg(test)]
mod tests {
    use super::permutation_odd;

    #[test]
    fn test_permutation_odd_matches_occurrences() {
        // Simplex orders for dimensions 2 through 5.
        for vertex_count in 3_u8..=6 {
            let source: Vec<_> = (0..vertex_count).collect();
            assert_eq!(permutation_odd(&source, &source), Some(false));

            let mut target = source.clone();
            target.swap(1, 2);
            assert_eq!(permutation_odd(&source, &target), Some(true));
            assert_eq!(permutation_odd(&source, &source[1..]), None);

            target[0] = vertex_count;
            assert_eq!(permutation_odd(&source, &target), None);
        }

        assert_eq!(permutation_odd::<u8>(&[], &[]), Some(false));
        assert_eq!(permutation_odd(&[1, 1, 2], &[1, 1, 2]), Some(false));
        assert_eq!(permutation_odd(&[1, 1, 2], &[1, 2, 1]), Some(true));
        assert_eq!(permutation_odd(&[1, 1, 2], &[2, 1, 1]), Some(false));
        assert_eq!(permutation_odd(&[1, 1, 2], &[1, 2, 2]), None);
    }
}
