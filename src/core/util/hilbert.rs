//! Hilbert space-filling curve ordering utilities.
//!
//! This module provides stateless, pure functions for mapping D-dimensional coordinates
//! to 1D Hilbert curve indices and for sorting arbitrary items by that ordering.
//!
//! ## Scope
//! - No triangulation types (no `Vertex`, no keys, no TDS access)
//! - Pure ordering primitives suitable for reuse across the crate

use crate::geometry::traits::coordinate::CoordinateScalar;

/// Quantize D-dimensional coordinates into integer grid coordinates in `[0, 2^bits)`.
///
/// The coordinates are normalized using a scalar `(min, max)` bound applied to every
/// dimension and then clamped to `[0, 1]` before quantization.
///
/// # Panics
/// Panics if `bits == 0` or `bits > 31`.
#[must_use]
pub fn hilbert_quantize<T: CoordinateScalar, const D: usize>(
    coords: &[T; D],
    bounds: (T, T),
    bits: u32,
) -> [u32; D] {
    assert!(bits > 0 && bits <= 31, "bits must be in range [1, 31]");

    if D == 0 {
        return [0_u32; D];
    }

    let extent = bounds.1 - bounds.0;

    // `2^bits - 1` as u32.
    let max_val_u32 = (1_u32 << bits) - 1;
    let max_val_t: T = num_traits::NumCast::from(max_val_u32).unwrap_or_else(T::zero);

    let mut quantized = [0_u32; D];
    for (i, &coord) in coords.iter().enumerate() {
        // Normalize to [0, 1]. If bounds are degenerate or non-finite, fall back to 0.
        let normalized = if extent > T::zero() {
            let t = (coord - bounds.0) / extent;
            if t.is_finite_generic() {
                t.max(T::zero()).min(T::one())
            } else {
                T::zero()
            }
        } else {
            T::zero()
        };

        let scaled = normalized * max_val_t;
        let q: u32 = num_traits::NumCast::from(scaled)
            .unwrap_or(0)
            .min(max_val_u32);
        quantized[i] = q;
    }

    quantized
}

/// Compute the Hilbert curve index for a point in D-dimensional space.
///
/// Internally, coordinates are quantized to an integer grid and then mapped to a
/// single index using an iterative Gray-code based algorithm.
///
/// # Panics
/// - Panics if `bits == 0` or `bits > 31`.
/// - Panics if `D * bits > 128` (index would not fit in `u128`).
#[must_use]
pub fn hilbert_index<T: CoordinateScalar, const D: usize>(
    coords: &[T; D],
    bounds: (T, T),
    bits: u32,
) -> u128 {
    assert!(bits > 0 && bits <= 31, "bits must be in range [1, 31]");
    let d_u32 = u32::try_from(D).expect("D should fit in u32 for overflow check");
    let total_bits = u128::from(d_u32) * u128::from(bits);
    assert!(
        total_bits <= 128,
        "Hilbert index would overflow u128 for D={D} and bits={bits}"
    );

    if D == 0 {
        return 0;
    }

    let q = hilbert_quantize(coords, bounds, bits);
    hilbert_index_from_quantized(&q, bits)
}

/// Compute Hilbert index from pre-quantized integer coordinates.
///
/// This uses the Skilling (2004) algorithm (“Programming the Hilbert curve”) to map
/// `D` integer coordinates (each `bits` bits wide) to a single Hilbert index.
///
/// The resulting ordering is continuous on the integer grid (successive indices move to
/// adjacent cells).
#[must_use]
fn hilbert_index_from_quantized<const D: usize>(coords: &[u32; D], bits: u32) -> u128 {
    debug_assert!(D > 0, "caller should handle D==0");

    // Work on a local copy in "transposed" form.
    let mut transposed = *coords;

    // See: J. Skilling, "Programming the Hilbert curve", AIP Conference Proceedings 707 (2004).
    // Step 1: transform axes to 'transpose' form.
    let highest_bit_mask: u32 = 1_u32 << (bits - 1);
    let mut bit_mask: u32 = highest_bit_mask;
    while bit_mask > 1 {
        let mask_minus_one = bit_mask - 1;

        // i = 0 case (special-cased to avoid borrow conflicts in the iterator loop below).
        if (transposed[0] & bit_mask) != 0 {
            transposed[0] ^= mask_minus_one;
        }

        let (first, rest) = transposed.split_at_mut(1);
        let first_coord = &mut first[0];

        for coord in rest {
            if (*coord & bit_mask) != 0 {
                *first_coord ^= mask_minus_one;
            } else {
                let toggle = (*first_coord ^ *coord) & mask_minus_one;
                *first_coord ^= toggle;
                *coord ^= toggle;
            }
        }

        bit_mask >>= 1;
    }

    // Step 2: Gray encode.
    let mut prev = transposed[0];
    for coord in transposed.iter_mut().skip(1) {
        *coord ^= prev;
        prev = *coord;
    }

    let mut gray_mask: u32 = 0;
    bit_mask = highest_bit_mask;
    while bit_mask > 1 {
        if (transposed[D - 1] & bit_mask) != 0 {
            gray_mask ^= bit_mask - 1;
        }
        bit_mask >>= 1;
    }

    for coord in &mut transposed {
        *coord ^= gray_mask;
    }

    // Step 3: interleave the transposed bits into the final index.
    let mut index: u128 = 0;
    for bit_pos in (0..bits).rev() {
        for &coord in &transposed {
            let bit_value = (coord >> bit_pos) & 1;
            index = (index << 1) | u128::from(bit_value);
        }
    }

    index
}

#[inline]
fn hilbert_sort_key<T: CoordinateScalar, const D: usize>(
    coords: &[T; D],
    bounds: (T, T),
    bits: u32,
) -> (u128, [u32; D]) {
    // Keep assertions consistent with `hilbert_index`.
    assert!(bits > 0 && bits <= 31, "bits must be in range [1, 31]");
    let d_u32 = u32::try_from(D).expect("D should fit in u32 for overflow check");
    let total_bits = u128::from(d_u32) * u128::from(bits);
    assert!(
        total_bits <= 128,
        "Hilbert index would overflow u128 for D={D} and bits={bits}"
    );

    let q = hilbert_quantize(coords, bounds, bits);
    let idx = hilbert_index_from_quantized(&q, bits);
    (idx, q)
}

/// Stable sort helper: sort items by Hilbert index + quantized-coordinate tie-break.
///
/// This is a generic helper that does not depend on triangulation types.
pub fn hilbert_sort_by_stable<Item, T, F, const D: usize>(
    items: &mut [Item],
    bounds: (T, T),
    bits: u32,
    coords_of: F,
) where
    T: CoordinateScalar,
    F: Fn(&Item) -> [T; D],
{
    items.sort_by_cached_key(|item| {
        let c = coords_of(item);
        hilbert_sort_key(&c, bounds, bits)
    });
}

/// Unstable sort helper: sort items by Hilbert index + quantized-coordinate tie-break.
///
/// This avoids allocations beyond what the sort implementation may use internally,
/// but recomputes indices during comparisons.
pub fn hilbert_sort_by_unstable<Item, T, F, const D: usize>(
    items: &mut [Item],
    bounds: (T, T),
    bits: u32,
    coords_of: F,
) where
    T: CoordinateScalar,
    F: Fn(&Item) -> [T; D],
{
    items.sort_unstable_by(|a, b| {
        let ca = coords_of(a);
        let cb = coords_of(b);
        hilbert_sort_key(&ca, bounds, bits).cmp(&hilbert_sort_key(&cb, bounds, bits))
    });
}

/// Return the indices that would sort `coords` by Hilbert order.
#[must_use]
pub fn hilbert_sorted_indices<T: CoordinateScalar, const D: usize>(
    coords: &[[T; D]],
    bounds: (T, T),
    bits: u32,
) -> Vec<usize> {
    let mut keyed: Vec<((u128, [u32; D]), usize)> = coords
        .iter()
        .enumerate()
        .map(|(i, c)| (hilbert_sort_key(c, bounds, bits), i))
        .collect();

    keyed.sort_by(|(ka, ia), (kb, ib)| ka.cmp(kb).then_with(|| ia.cmp(ib)));
    keyed.into_iter().map(|(_, i)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::Coordinate;

    #[test]
    fn test_hilbert_index_2d() {
        let origin = hilbert_index(&[0.0_f64, 0.0], (0.0, 1.0), 4);
        let corner = hilbert_index(&[1.0_f64, 1.0], (0.0, 1.0), 4);
        let center = hilbert_index(&[0.5_f64, 0.5], (0.0, 1.0), 4);

        assert_eq!(origin, 0);
        assert_ne!(origin, center);
        assert_ne!(center, corner);
    }

    #[test]
    fn test_hilbert_index_3d() {
        let origin = hilbert_index(&[0.0_f64, 0.0, 0.0], (-1.0, 1.0), 8);
        let corner = hilbert_index(&[1.0_f64, 1.0, 1.0], (-1.0, 1.0), 8);
        assert_ne!(origin, corner);
    }

    #[test]
    fn test_hilbert_sorted_indices_and_sort_helpers() {
        let coords: Vec<[f64; 2]> =
            vec![[0.9, 0.9], [0.1, 0.1], [0.5, 0.5], [0.1, 0.9], [0.9, 0.1]];
        let order = hilbert_sorted_indices(&coords, (0.0, 1.0), 16);
        assert_eq!(order.len(), coords.len());

        // Apply the ordering to a parallel payload.
        let mut payload: Vec<usize> = (0..coords.len()).collect();
        hilbert_sort_by_stable(&mut payload, (0.0_f64, 1.0), 16, |&i| coords[i]);

        // Sorting by stable helper should be deterministic.
        let mut payload2: Vec<usize> = (0..coords.len()).collect();
        hilbert_sort_by_stable(&mut payload2, (0.0_f64, 1.0), 16, |&i| coords[i]);
        assert_eq!(payload, payload2);
    }

    #[test]
    fn test_hilbert_curve_is_continuous_on_2d_grid() {
        // A defining property of the (discrete) Hilbert curve is continuity:
        // successive indices correspond to adjacent grid cells.
        let bits: u32 = 4;
        let n: u32 = 1_u32 << bits;

        let mut points: Vec<([u32; 2], u128)> = Vec::with_capacity((n * n) as usize);
        for x in 0..n {
            for y in 0..n {
                let q = [x, y];
                let idx = hilbert_index_from_quantized(&q, bits);
                points.push((q, idx));
            }
        }

        points.sort_by_key(|(_, idx)| *idx);

        // Indices should form a permutation of 0..n^2.
        for (i, (_, idx)) in points.iter().enumerate() {
            let i_u128 = u128::from(u32::try_from(i).expect("grid size should fit in u32"));
            assert_eq!(*idx, i_u128);
        }

        // Continuity: successive points differ by Manhattan distance exactly 1.
        for window in points.windows(2) {
            let a = window[0].0;
            let b = window[1].0;
            let dx = a[0].abs_diff(b[0]);
            let dy = a[1].abs_diff(b[1]);
            assert_eq!(dx + dy, 1, "Non-adjacent step: a={a:?}, b={b:?}");
        }
    }

    #[test]
    fn test_point_coords_work_with_hilbert() {
        let p: Point<f64, 2> = Point::new([0.25, 0.75]);
        let idx = hilbert_index(p.coords(), (0.0, 1.0), 16);
        assert!(idx > 0);
    }
}
