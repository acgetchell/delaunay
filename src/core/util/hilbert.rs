//! Hilbert space-filling curve ordering utilities.
//!
//! This module provides stateless, pure functions for mapping D-dimensional coordinates
//! to 1D Hilbert curve indices and for sorting arbitrary items by that ordering.
//!
//! ## Scope
//! - No triangulation types (no `Vertex`, no keys, no TDS access)
//! - Pure ordering primitives suitable for reuse across the crate

#![forbid(unsafe_code)]

use crate::geometry::traits::coordinate::CoordinateScalar;

/// Errors that can occur during Hilbert curve operations.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum HilbertError {
    /// The `bits` parameter is out of valid range [1, 31].
    #[error("bits parameter {bits} is out of valid range [1, 31]")]
    InvalidBitsParameter {
        /// The invalid bits value provided.
        bits: u32,
    },

    /// The combination of dimension and bits would cause index overflow.
    #[error(
        "Hilbert index would overflow u128: dimension {dimension} * bits {bits} = {total_bits} > 128"
    )]
    IndexOverflow {
        /// The dimension of the coordinate space.
        dimension: usize,
        /// The bits parameter.
        bits: u32,
        /// The total number of bits required (dimension * bits).
        total_bits: u128,
    },

    /// The dimension is too large to represent.
    #[error("dimension {dimension} is too large (exceeds u32::MAX)")]
    DimensionTooLarge {
        /// The dimension that exceeded representable limits.
        dimension: usize,
    },
}

/// Quantize D-dimensional coordinates into integer grid coordinates in `[0, 2^bits)`.
///
/// The coordinates are normalized using a scalar `(min, max)` bound applied to every
/// dimension and then clamped to `[0, 1]` before quantization.
///
/// # Errors
///
/// Returns [`HilbertError::InvalidBitsParameter`] if `bits` is 0 or greater than 31.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::util::hilbert::hilbert_quantize;
///
/// let coords = [0.5_f64, 0.25];
/// let q = hilbert_quantize(&coords, (0.0, 1.0), 2)?;
/// assert!(q[0] <= 3 && q[1] <= 3);
/// # Ok::<(), delaunay::core::util::hilbert::HilbertError>(())
/// ```
pub fn hilbert_quantize<T: CoordinateScalar, const D: usize>(
    coords: &[T; D],
    bounds: (T, T),
    bits: u32,
) -> Result<[u32; D], HilbertError> {
    if bits == 0 || bits > 31 {
        return Err(HilbertError::InvalidBitsParameter { bits });
    }

    if D == 0 {
        return Ok([0_u32; D]);
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
        // Round to nearest grid cell (instead of truncating) for fairer distribution.
        let q: u32 = scaled.round().to_u32().unwrap_or(0).min(max_val_u32);
        quantized[i] = q;
    }

    Ok(quantized)
}

/// Internal unchecked quantization - assumes bits parameter is already validated.
/// This is used in hot paths after pre-validation to avoid Result overhead.
#[inline]
fn hilbert_quantize_unchecked<T: CoordinateScalar, const D: usize>(
    coords: &[T; D],
    bounds: (T, T),
    bits: u32,
) -> [u32; D] {
    debug_assert!(bits > 0 && bits <= 31, "bits must be pre-validated");

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
        // Round to nearest grid cell (instead of truncating) for fairer distribution.
        let q: u32 = scaled.round().to_u32().unwrap_or(0).min(max_val_u32);
        quantized[i] = q;
    }

    quantized
}

/// Compute the Hilbert curve index for a point in D-dimensional space.
///
/// Internally, coordinates are quantized to an integer grid and then mapped to a
/// single index using an iterative Gray-code based algorithm.
///
/// # Errors
///
/// Returns [`HilbertError::InvalidBitsParameter`] if `bits` is 0 or greater than 31.
///
/// Returns [`HilbertError::IndexOverflow`] if `D * bits > 128` (index would not fit in `u128`).
///
/// Returns [`HilbertError::DimensionTooLarge`] if the dimension `D` exceeds `u32::MAX`
/// (extremely unlikely in practice).
///
/// # Examples
///
/// ```rust
/// use delaunay::core::util::hilbert::hilbert_index;
///
/// let idx = hilbert_index(&[0.0_f64, 0.0], (0.0, 1.0), 4)?;
/// assert_eq!(idx, 0);
/// # Ok::<(), delaunay::core::util::hilbert::HilbertError>(())
/// ```
pub fn hilbert_index<T: CoordinateScalar, const D: usize>(
    coords: &[T; D],
    bounds: (T, T),
    bits: u32,
) -> Result<u128, HilbertError> {
    // Validate bits parameter (same as hilbert_quantize)
    if bits == 0 || bits > 31 {
        return Err(HilbertError::InvalidBitsParameter { bits });
    }

    // Validate dimension fits in u32
    let d_u32 = u32::try_from(D).map_err(|_| HilbertError::DimensionTooLarge { dimension: D })?;

    // Validate overflow
    let total_bits = u128::from(d_u32) * u128::from(bits);
    if total_bits > 128 {
        return Err(HilbertError::IndexOverflow {
            dimension: D,
            bits,
            total_bits,
        });
    }

    if D == 0 {
        return Ok(0);
    }

    let q = hilbert_quantize(coords, bounds, bits)?;
    Ok(hilbert_index_from_quantized(&q, bits))
}

/// Compute Hilbert index from pre-quantized integer coordinates.
///
/// This uses the Skilling (2004) algorithm ("Programming the Hilbert curve") to map
/// `D` integer coordinates (each `bits` bits wide) to a single Hilbert index.
///
/// The resulting ordering is continuous on the integer grid (successive indices move to
/// adjacent cells).
#[must_use]
fn hilbert_index_from_quantized<const D: usize>(coords: &[u32; D], bits: u32) -> u128 {
    debug_assert!(D > 0, "caller should handle D==0");
    debug_assert!(
        bits > 0 && bits <= 31,
        "bits must be in range [1, 31], got {bits}"
    );
    debug_assert!(
        (D as u128) * u128::from(bits) <= 128,
        "Hilbert index would overflow u128 for D={D} and bits={bits}"
    );

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

/// Stable sort helper: sort items by Hilbert index + quantized-coordinate tie-break.
///
/// This is a generic helper that does not depend on triangulation types.
///
/// When `D == 0`, all items are considered equivalent (index 0) and the sort is stable
/// based on original order.
///
/// # Errors
///
/// Returns [`HilbertError::InvalidBitsParameter`] if `bits` is 0 or greater than 31.
///
/// Returns [`HilbertError::IndexOverflow`] if `D * bits > 128` (index would not fit in `u128`).
///
/// Returns [`HilbertError::DimensionTooLarge`] if the dimension `D` exceeds `u32::MAX`
/// (extremely unlikely in practice).
///
/// # Examples
///
/// ```rust
/// use delaunay::core::util::hilbert::hilbert_sort_by_stable;
///
/// let mut points = vec![[0.9_f64, 0.9], [0.1, 0.1], [0.5, 0.5]];
/// hilbert_sort_by_stable(&mut points, (0.0, 1.0), 8, |p| *p)?;
/// assert_eq!(points[0], [0.1, 0.1]);
/// # Ok::<(), delaunay::core::util::hilbert::HilbertError>(())
/// ```
pub fn hilbert_sort_by_stable<Item, T, F, const D: usize>(
    items: &mut [Item],
    bounds: (T, T),
    bits: u32,
    coords_of: F,
) -> Result<(), HilbertError>
where
    T: CoordinateScalar,
    F: Fn(&Item) -> [T; D],
{
    // Pre-validate parameters once before sorting
    if bits == 0 || bits > 31 {
        return Err(HilbertError::InvalidBitsParameter { bits });
    }

    let d_u32 = u32::try_from(D).map_err(|_| HilbertError::DimensionTooLarge { dimension: D })?;
    let total_bits = u128::from(d_u32) * u128::from(bits);
    if total_bits > 128 {
        return Err(HilbertError::IndexOverflow {
            dimension: D,
            bits,
            total_bits,
        });
    }

    // Sort using cached keys - parameters are pre-validated
    items.sort_by_cached_key(|item| {
        let c = coords_of(item);
        let q = hilbert_quantize_unchecked(&c, bounds, bits);
        let idx = if D == 0 {
            0
        } else {
            hilbert_index_from_quantized(&q, bits)
        };
        (idx, q)
    });

    Ok(())
}

/// Unstable sort helper: sort items by Hilbert index + quantized-coordinate tie-break.
///
/// This avoids allocations beyond what the sort implementation may use internally,
/// but recomputes indices during comparisons. Prefer [`hilbert_sort_by_stable`] unless
/// memory pressure is critical, as the unstable variant recomputes keys O(n log n) times.
///
/// When `D == 0`, all items are considered equivalent (index 0) and the sort order is
/// implementation-defined.
///
/// # Errors
///
/// Returns [`HilbertError::InvalidBitsParameter`] if `bits` is 0 or greater than 31.
///
/// Returns [`HilbertError::IndexOverflow`] if `D * bits > 128` (index would not fit in `u128`).
///
/// Returns [`HilbertError::DimensionTooLarge`] if the dimension `D` exceeds `u32::MAX`
/// (extremely unlikely in practice).
///
/// # Examples
///
/// ```rust
/// use delaunay::core::util::hilbert::hilbert_sort_by_unstable;
///
/// let mut points = vec![[0.9_f64, 0.9], [0.1, 0.1], [0.5, 0.5]];
/// hilbert_sort_by_unstable(&mut points, (0.0, 1.0), 8, |p| *p)?;
/// assert_eq!(points[0], [0.1, 0.1]);
/// # Ok::<(), delaunay::core::util::hilbert::HilbertError>(())
/// ```
pub fn hilbert_sort_by_unstable<Item, T, F, const D: usize>(
    items: &mut [Item],
    bounds: (T, T),
    bits: u32,
    coords_of: F,
) -> Result<(), HilbertError>
where
    T: CoordinateScalar,
    F: Fn(&Item) -> [T; D],
{
    // Pre-validate parameters once before sorting
    if bits == 0 || bits > 31 {
        return Err(HilbertError::InvalidBitsParameter { bits });
    }

    let d_u32 = u32::try_from(D).map_err(|_| HilbertError::DimensionTooLarge { dimension: D })?;
    let total_bits = u128::from(d_u32) * u128::from(bits);
    if total_bits > 128 {
        return Err(HilbertError::IndexOverflow {
            dimension: D,
            bits,
            total_bits,
        });
    }

    items.sort_unstable_by(|a, b| {
        let ca = coords_of(a);
        let cb = coords_of(b);
        let qa = hilbert_quantize_unchecked(&ca, bounds, bits);
        let qb = hilbert_quantize_unchecked(&cb, bounds, bits);
        let ida = if D == 0 {
            0
        } else {
            hilbert_index_from_quantized(&qa, bits)
        };
        let idb = if D == 0 {
            0
        } else {
            hilbert_index_from_quantized(&qb, bits)
        };
        (ida, qa).cmp(&(idb, qb))
    });

    Ok(())
}

/// Compute Hilbert indices for a batch of pre-quantized coordinates.
///
/// This is a bulk API that avoids recomputing quantization parameters for large
/// insertion batches. When inserting many points, quantize them once using
/// [`hilbert_quantize`] and then call this function to compute all indices in bulk.
///
/// # Performance
///
/// This function validates parameters once and then maps each quantized coordinate
/// through the internal Hilbert index computation. For large batches, this is significantly
/// faster than calling [`hilbert_index`] individually for each point.
///
/// # Errors
///
/// Returns [`HilbertError::InvalidBitsParameter`] if `bits` is 0 or greater than 31.
///
/// Returns [`HilbertError::IndexOverflow`] if `D * bits > 128` (index would not fit in `u128`).
///
/// Returns [`HilbertError::DimensionTooLarge`] if the dimension `D` exceeds `u32::MAX`
/// (extremely unlikely in practice).
///
/// # Examples
///
/// ```rust
/// use delaunay::core::util::hilbert::{hilbert_quantize, hilbert_indices_prequantized};
///
/// let coords = vec![[0.1_f64, 0.2], [0.5, 0.5], [0.9, 0.8]];
/// let bounds = (0.0, 1.0);
/// let bits = 8;
///
/// // Quantize once
/// let quantized: Vec<[u32; 2]> = coords
///     .iter()
///     .map(|c| hilbert_quantize(c, bounds, bits).unwrap())
///     .collect();
///
/// // Compute all indices in bulk
/// let indices = hilbert_indices_prequantized(&quantized, bits)?;
/// assert_eq!(indices.len(), coords.len());
/// # Ok::<(), delaunay::core::util::hilbert::HilbertError>(())
/// ```
///
/// Error handling:
///
/// ```rust
/// use delaunay::core::util::hilbert::{hilbert_indices_prequantized, HilbertError};
///
/// let quantized = vec![[1_u32, 2]];
///
/// // Invalid bits parameter
/// let result = hilbert_indices_prequantized(&quantized, 0);
/// assert!(matches!(result, Err(HilbertError::InvalidBitsParameter { bits: 0 })));
///
/// // Overflow (D=5, bits=26 => 130 > 128)
/// let quantized_5d = vec![[1_u32, 2, 3, 4, 5]];
/// let result = hilbert_indices_prequantized(&quantized_5d, 26);
/// assert!(matches!(result, Err(HilbertError::IndexOverflow { .. })));
/// ```
pub fn hilbert_indices_prequantized<const D: usize>(
    quantized: &[[u32; D]],
    bits: u32,
) -> Result<Vec<u128>, HilbertError> {
    // Validate bits parameter
    if bits == 0 || bits > 31 {
        return Err(HilbertError::InvalidBitsParameter { bits });
    }

    // Validate dimension fits in u32
    let d_u32 = u32::try_from(D).map_err(|_| HilbertError::DimensionTooLarge { dimension: D })?;

    // Validate overflow
    let total_bits = u128::from(d_u32) * u128::from(bits);
    if total_bits > 128 {
        return Err(HilbertError::IndexOverflow {
            dimension: D,
            bits,
            total_bits,
        });
    }

    // Handle D == 0 case: zero-dimensional space has only one point, all map to index 0
    if D == 0 {
        return Ok(vec![0_u128; quantized.len()]);
    }

    Ok(quantized
        .iter()
        .map(|q| hilbert_index_from_quantized(q, bits))
        .collect())
}

/// Return the indices that would sort `coords` by Hilbert order.
///
/// When `D == 0`, all coordinates are considered equivalent (index 0) and the returned
/// indices preserve the original order.
///
/// # Errors
///
/// Returns [`HilbertError::InvalidBitsParameter`] if `bits` is 0 or greater than 31.
///
/// Returns [`HilbertError::IndexOverflow`] if `D * bits > 128` (index would not fit in `u128`).
///
/// Returns [`HilbertError::DimensionTooLarge`] if the dimension `D` exceeds `u32::MAX`
/// (extremely unlikely in practice).
///
/// # Examples
///
/// ```rust
/// use delaunay::core::util::hilbert::hilbert_sorted_indices;
///
/// let coords = vec![[0.9_f64, 0.9], [0.1, 0.1], [0.5, 0.5]];
/// let order = hilbert_sorted_indices(&coords, (0.0, 1.0), 8)?;
/// assert_eq!(order.len(), coords.len());
/// # Ok::<(), delaunay::core::util::hilbert::HilbertError>(())
/// ```
pub fn hilbert_sorted_indices<T: CoordinateScalar, const D: usize>(
    coords: &[[T; D]],
    bounds: (T, T),
    bits: u32,
) -> Result<Vec<usize>, HilbertError> {
    // Pre-validate parameters once
    if bits == 0 || bits > 31 {
        return Err(HilbertError::InvalidBitsParameter { bits });
    }

    let d_u32 = u32::try_from(D).map_err(|_| HilbertError::DimensionTooLarge { dimension: D })?;
    let total_bits = u128::from(d_u32) * u128::from(bits);
    if total_bits > 128 {
        return Err(HilbertError::IndexOverflow {
            dimension: D,
            bits,
            total_bits,
        });
    }

    let mut keyed: Vec<((u128, [u32; D]), usize)> = coords
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let q = hilbert_quantize_unchecked(c, bounds, bits);
            let idx = if D == 0 {
                0
            } else {
                hilbert_index_from_quantized(&q, bits)
            };
            ((idx, q), i)
        })
        .collect();

    keyed.sort_by(|(ka, ia), (kb, ib)| ka.cmp(kb).then_with(|| ia.cmp(ib)));
    Ok(keyed.into_iter().map(|(_, i)| i).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::Coordinate;

    #[test]
    fn test_hilbert_index_2d() {
        let origin = hilbert_index(&[0.0_f64, 0.0], (0.0, 1.0), 4).unwrap();
        let corner = hilbert_index(&[1.0_f64, 1.0], (0.0, 1.0), 4).unwrap();
        let center = hilbert_index(&[0.5_f64, 0.5], (0.0, 1.0), 4).unwrap();

        assert_eq!(origin, 0);
        assert_ne!(origin, center);
        assert_ne!(center, corner);
    }

    #[test]
    fn test_hilbert_index_3d() {
        let origin = hilbert_index(&[0.0_f64, 0.0, 0.0], (-1.0, 1.0), 8).unwrap();
        let corner = hilbert_index(&[1.0_f64, 1.0, 1.0], (-1.0, 1.0), 8).unwrap();
        assert_ne!(origin, corner);
    }

    #[test]
    fn test_hilbert_sorted_indices_and_sort_helpers() {
        let coords: Vec<[f64; 2]> =
            vec![[0.9, 0.9], [0.1, 0.1], [0.5, 0.5], [0.1, 0.9], [0.9, 0.1]];
        let order = hilbert_sorted_indices(&coords, (0.0, 1.0), 16).unwrap();
        assert_eq!(order.len(), coords.len());

        // Apply the ordering to a parallel payload.
        let mut payload: Vec<usize> = (0..coords.len()).collect();
        hilbert_sort_by_stable(&mut payload, (0.0_f64, 1.0), 16, |&i| coords[i]).unwrap();

        // Sorting by stable helper should be deterministic.
        let mut payload2: Vec<usize> = (0..coords.len()).collect();
        hilbert_sort_by_stable(&mut payload2, (0.0_f64, 1.0), 16, |&i| coords[i]).unwrap();
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
    fn test_hilbert_curve_is_continuous_on_4d_grid() {
        // A defining property of the (discrete) Hilbert curve is continuity:
        // successive indices correspond to adjacent grid cells.
        let bits: u32 = 2;
        let n: u32 = 1_u32 << bits;

        let mut points: Vec<([u32; 4], u128)> = Vec::with_capacity((n * n * n * n) as usize);
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    for w in 0..n {
                        let q = [x, y, z, w];
                        let idx = hilbert_index_from_quantized(&q, bits);
                        points.push((q, idx));
                    }
                }
            }
        }

        points.sort_by_key(|(_, idx)| *idx);

        // Indices should form a permutation of 0..n^4.
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
            let dz = a[2].abs_diff(b[2]);
            let dw = a[3].abs_diff(b[3]);
            assert_eq!(dx + dy + dz + dw, 1, "Non-adjacent step: a={a:?}, b={b:?}");
        }
    }

    #[test]
    fn test_point_coords_work_with_hilbert() {
        let p: Point<f64, 2> = Point::new([0.25, 0.75]);
        let idx = hilbert_index(p.coords(), (0.0, 1.0), 16).unwrap();
        assert!(idx > 0);
    }

    #[test]
    fn test_hilbert_bits_boundaries() {
        let origin = hilbert_index(&[0.0_f64, 0.0], (0.0, 1.0), 1).unwrap();
        let corner = hilbert_index(&[1.0_f64, 1.0], (0.0, 1.0), 1).unwrap();
        tracing::debug!(origin, corner, "bits=1 boundaries");
        assert_eq!(origin, 0, "bits=1 origin should map to 0");
        assert_ne!(origin, corner, "bits=1 should distinguish corners");

        let origin_31 = hilbert_index(&[0.0_f64, 0.0], (0.0, 1.0), 31).unwrap();
        let corner_31 = hilbert_index(&[1.0_f64, 1.0], (0.0, 1.0), 31).unwrap();
        tracing::debug!(origin_31, corner_31, "bits=31 boundaries");
        assert_eq!(origin_31, 0, "bits=31 origin should map to 0");
        assert_ne!(origin_31, corner_31, "bits=31 should distinguish corners");
    }

    #[test]
    fn test_hilbert_index_1d_monotonic() {
        let bounds = (0.0_f64, 1.0_f64);
        let bits = 8;
        let a = hilbert_index(&[0.0_f64], bounds, bits).unwrap();
        let b = hilbert_index(&[0.25_f64], bounds, bits).unwrap();
        let c = hilbert_index(&[0.5_f64], bounds, bits).unwrap();
        let d = hilbert_index(&[1.0_f64], bounds, bits).unwrap();
        tracing::debug!(a, b, c, d, "1d indices");
        assert!(
            a < b && b < c && c < d,
            "1D Hilbert indices should be monotonic"
        );
    }

    #[test]
    fn test_hilbert_degenerate_bounds_quantize_to_zero() {
        let bounds = (1.0_f64, 1.0_f64);
        let coords = [2.0_f64, -2.0_f64];
        let q = hilbert_quantize(&coords, bounds, 8).unwrap();
        let idx = hilbert_index(&coords, bounds, 8).unwrap();
        tracing::debug!(?q, idx, "degenerate bounds");
        assert_eq!(q, [0, 0], "degenerate bounds should quantize to zeros");
        assert_eq!(idx, 0, "degenerate bounds should map to index 0");
    }

    #[test]
    fn test_hilbert_quantize_clamps_out_of_range() {
        let bounds = (0.0_f64, 1.0_f64);
        let bits = 4;
        let coords = [-1.0_f64, 2.0_f64];
        let q = hilbert_quantize(&coords, bounds, bits).unwrap();
        let max_val = (1_u32 << bits) - 1;
        tracing::debug!(?q, max_val, "clamp quantize");
        assert_eq!(
            q,
            [0, max_val],
            "out-of-range coords should clamp to bounds"
        );

        let idx = hilbert_index(&coords, bounds, bits).unwrap();
        let idx_clamped = hilbert_index(&[0.0_f64, 1.0_f64], bounds, bits).unwrap();
        tracing::debug!(idx, idx_clamped, "clamp index");
        assert_eq!(
            idx, idx_clamped,
            "clamped coords should match clamped index"
        );
    }

    #[test]
    fn test_hilbert_indices_prequantized_matches_individual_calls() {
        let coords = [
            [0.1_f64, 0.2, 0.3],
            [0.5, 0.5, 0.5],
            [0.9, 0.8, 0.7],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ];
        let bounds = (0.0_f64, 1.0_f64);
        let bits = 8;

        // Quantize all coordinates
        let quantized: Vec<[u32; 3]> = coords
            .iter()
            .map(|c| hilbert_quantize(c, bounds, bits).unwrap())
            .collect();

        // Compute indices via bulk API
        let indices_bulk = hilbert_indices_prequantized(&quantized, bits)
            .expect("valid parameters should succeed");

        // Compute indices individually
        let indices_individual: Vec<u128> = coords
            .iter()
            .map(|c| hilbert_index(c, bounds, bits).unwrap())
            .collect();

        assert_eq!(indices_bulk.len(), coords.len());
        assert_eq!(indices_bulk, indices_individual);
    }

    #[test]
    fn test_hilbert_indices_prequantized_empty_input() {
        let empty: Vec<[u32; 2]> = vec![];
        let bits = 4;

        let indices =
            hilbert_indices_prequantized(&empty, bits).expect("valid parameters should succeed");
        assert_eq!(indices.len(), 0);
    }

    #[test]
    fn test_hilbert_indices_prequantized_validates_bits_zero() {
        let quantized = vec![[1_u32, 2]];
        let result = hilbert_indices_prequantized(&quantized, 0);
        assert!(matches!(
            result,
            Err(HilbertError::InvalidBitsParameter { bits: 0 })
        ));
    }

    #[test]
    fn test_hilbert_indices_prequantized_validates_bits_too_large() {
        let quantized = vec![[1_u32, 2]];
        let result = hilbert_indices_prequantized(&quantized, 32);
        assert!(matches!(
            result,
            Err(HilbertError::InvalidBitsParameter { bits: 32 })
        ));
    }

    #[test]
    fn test_hilbert_indices_prequantized_validates_overflow() {
        // With D=5 and bits=26, total_bits = 130 > 128
        let quantized = vec![[1_u32, 2, 3, 4, 5]];
        let result = hilbert_indices_prequantized(&quantized, 26);
        assert!(matches!(
            result,
            Err(HilbertError::IndexOverflow {
                dimension: 5,
                bits: 26,
                total_bits: 130
            })
        ));
    }

    #[test]
    fn test_hilbert_indices_prequantized_handles_zero_dimension() {
        // Zero-dimensional space has only one point, all map to index 0
        let quantized: Vec<[u32; 0]> = vec![[], [], []];
        let bits = 8;

        let indices = hilbert_indices_prequantized(&quantized, bits).expect("D=0 should succeed");

        assert_eq!(indices.len(), 3);
        assert_eq!(indices, vec![0_u128, 0_u128, 0_u128]);
    }

    #[test]
    fn test_hilbert_quantize_uses_rounding_not_truncation() {
        let bounds = (0.0_f64, 1.0_f64);
        let bits = 2; // Grid has 4 cells: 0, 1, 2, 3

        // With bits=2, max_val = 3, so we scale by 3.0.
        // coord * 3.0 is then rounded to nearest integer.
        // Cell boundaries (where rounding changes) are at:
        // 0.5/3 ≈ 0.167 (rounds from 0 to 1)
        // 1.5/3 = 0.5 (rounds from 1 to 2)
        // 2.5/3 ≈ 0.833 (rounds from 2 to 3)

        // Test points that should round to different cells
        let test_cases = [
            (0.0, 0),  // 0.0 * 3 = 0.0, rounds to 0
            (0.1, 0),  // 0.1 * 3 = 0.3, rounds to 0
            (0.17, 1), // 0.17 * 3 = 0.51, rounds to 1
            (0.3, 1),  // 0.3 * 3 = 0.9, rounds to 1
            (0.5, 2),  // 0.5 * 3 = 1.5, rounds to 2
            (0.7, 2),  // 0.7 * 3 = 2.1, rounds to 2
            (0.85, 3), // 0.85 * 3 = 2.55, rounds to 3
            (1.0, 3),  // exactly 1.0 -> cell 3 (clamped)
        ];

        for (coord, expected_cell) in test_cases {
            let q = hilbert_quantize(&[coord], bounds, bits).unwrap();
            assert_eq!(
                q[0], expected_cell,
                "coordinate {coord} should quantize to cell {expected_cell}, got {}",
                q[0]
            );
        }

        // Verify rounding distribution:
        // With rounding and bits=2 (max_val=3), cell boundaries are at:
        // - Cell 0: coord * 3 < 0.5 → coord < 0.167 (width 0.167)
        // - Cell 1: 0.5 <= coord * 3 < 1.5 → 0.167 <= coord < 0.5 (width 0.333)
        // - Cell 2: 1.5 <= coord * 3 < 2.5 → 0.5 <= coord < 0.833 (width 0.333)
        // - Cell 3: 2.5 <= coord * 3 <= 3.0 → 0.833 <= coord <= 1.0 (width 0.167)
        // So cells 1 and 2 should get roughly twice as many samples as cells 0 and 3.
        let samples = 1000;
        let mut cell_counts = [0_usize; 4];
        for i in 0..samples {
            let coord = f64::from(i) / f64::from(samples);
            let q = hilbert_quantize(&[coord], bounds, bits).unwrap();
            cell_counts[q[0] as usize] += 1;
        }

        tracing::debug!(?cell_counts, "cell distribution for {samples} samples");

        // Expected distribution: ~167 samples in cells 0 and 3, ~333 in cells 1 and 2.
        // Allow ±50 tolerance for discrete sampling effects.
        assert!(
            cell_counts[0] >= 100 && cell_counts[0] <= 217,
            "cell 0 should have ~167 samples with rounding, got {}",
            cell_counts[0]
        );
        assert!(
            cell_counts[1] >= 283 && cell_counts[1] <= 383,
            "cell 1 should have ~333 samples with rounding, got {}",
            cell_counts[1]
        );
        assert!(
            cell_counts[2] >= 283 && cell_counts[2] <= 383,
            "cell 2 should have ~333 samples with rounding, got {}",
            cell_counts[2]
        );
        assert!(
            cell_counts[3] >= 100 && cell_counts[3] <= 217,
            "cell 3 should have ~167 samples with rounding, got {}",
            cell_counts[3]
        );
    }
}
