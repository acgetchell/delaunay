//! Hilbert space-filling curve ordering utilities.
//!
//! This module provides stateless, pure functions for mapping D-dimensional coordinates
//! to 1D Hilbert curve indices and for sorting arbitrary items by that ordering.
//!
//! ## Scope
//! - No triangulation types (no `Vertex`, no keys, no TDS access)
//! - Pure ordering primitives suitable for reuse across the crate

#![forbid(unsafe_code)]

use crate::geometry::coordinate_range::{
    CoordinateRange, CoordinateRangeError, CoordinateRangeOrdering,
};
use core::fmt;
use num_traits::ToPrimitive;
use std::num::NonZeroU32;

/// Maximum supported Hilbert bit depth per coordinate accepted by [`HilbertBitDepth`].
pub const MAX_HILBERT_BITS: u32 = 31;

/// Errors that can occur during Hilbert curve operations.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
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

    /// A Hilbert quantization bound was non-finite.
    #[error(
        "Hilbert quantization bounds must be finite: lower bound finite = {lower_bound_finite}, upper bound finite = {upper_bound_finite}"
    )]
    NonFiniteBounds {
        /// Whether the lower quantization bound is finite.
        lower_bound_finite: bool,
        /// Whether the upper quantization bound is finite.
        upper_bound_finite: bool,
    },

    /// Finite Hilbert quantization bounds were equal or decreasing.
    #[error("Hilbert quantization bounds must satisfy min < max ({ordering})")]
    NonIncreasingBounds {
        /// Whether the bounds were equal or decreasing.
        ordering: CoordinateRangeOrdering,
    },

    /// Finite bounds produced a non-finite quantization extent.
    #[error("Hilbert quantization bounds produced a non-finite extent")]
    NonFiniteBoundsExtent {},

    /// A coordinate to quantize was non-finite.
    #[error("Hilbert coordinate at index {coordinate_index} must be finite")]
    NonFiniteCoordinate {
        /// The coordinate index whose value was non-finite.
        coordinate_index: usize,
    },

    /// A finite coordinate and finite bounds produced a non-finite normalized value.
    #[error(
        "Hilbert coordinate at index {coordinate_index} produced a non-finite normalized value"
    )]
    NonFiniteNormalizedCoordinate {
        /// The coordinate index whose normalized value was non-finite.
        coordinate_index: usize,
    },

    /// A rounded, scaled coordinate could not be represented as a `u32`.
    #[error(
        "Hilbert quantized coordinate at index {coordinate_index} for {bits} bits and grid maximum {max_grid_value} cannot be represented as u32"
    )]
    QuantizedCoordinateConversionFailed {
        /// The requested number of Hilbert bits per coordinate.
        bits: u32,
        /// The computed grid maximum, `2^bits - 1`.
        max_grid_value: u32,
        /// The coordinate index whose rounded scaled value could not be converted.
        coordinate_index: usize,
    },

    /// A pre-quantized coordinate exceeded the grid range implied by the bit depth.
    #[error(
        "pre-quantized Hilbert coordinate at point {point_index}, coordinate {coordinate_index} has value {coordinate}, which exceeds the maximum {max_grid_value} for {bits} bits"
    )]
    PrequantizedCoordinateOutOfRange {
        /// The requested number of Hilbert bits per coordinate.
        bits: u32,
        /// The computed grid maximum, `2^bits - 1`.
        max_grid_value: u32,
        /// The point index whose pre-quantized coordinate was out of range.
        point_index: usize,
        /// The coordinate index whose value was out of range.
        coordinate_index: usize,
        /// The out-of-range pre-quantized coordinate value.
        coordinate: u32,
    },

    /// An internally constructed Hilbert sort permutation had the wrong length.
    #[error(
        "Hilbert sort permutation length mismatch: item count {item_count}, permutation count {permutation_count}"
    )]
    InvalidSortPermutationLength {
        /// Number of items being sorted.
        item_count: usize,
        /// Number of indices in the permutation.
        permutation_count: usize,
    },

    /// An internally constructed Hilbert sort permutation referenced an invalid item index.
    #[error(
        "Hilbert sort permutation index {permutation_index} has value {item_index}, which is outside item count {item_count}"
    )]
    InvalidSortPermutationIndex {
        /// Position in the permutation whose value was invalid.
        permutation_index: usize,
        /// Invalid source item index.
        item_index: usize,
        /// Number of items being sorted.
        item_count: usize,
    },

    /// An internally constructed Hilbert sort permutation referenced the same item twice.
    #[error("Hilbert sort permutation index {permutation_index} repeats item index {item_index}")]
    InvalidSortPermutationDuplicate {
        /// Position in the permutation whose value repeated an earlier item.
        permutation_index: usize,
        /// Repeated source item index.
        item_index: usize,
    },
}

/// Validated Hilbert bit depth per coordinate.
///
/// Values are constrained to the inclusive range from `1` through
/// [`MAX_HILBERT_BITS`], matching the `u32` quantization grid used by the
/// Hilbert ordering implementation. Public Hilbert APIs accept this type so raw
/// bit-depth validation happens once at the boundary instead of being repeated
/// during sorting or index computation.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::ordering::{HilbertBitDepth, HilbertError, MAX_HILBERT_BITS};
///
/// let bits = HilbertBitDepth::try_new(8)?;
/// assert_eq!(bits.get(), 8);
/// assert!(HilbertBitDepth::try_new(MAX_HILBERT_BITS).is_ok());
///
/// std::assert_matches!(
///     HilbertBitDepth::try_new(0),
///     Err(HilbertError::InvalidBitsParameter { bits: 0 })
/// );
/// # Ok::<(), HilbertError>(())
/// ```
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[must_use]
pub struct HilbertBitDepth(NonZeroU32);

impl HilbertBitDepth {
    /// Parses a raw bit depth into a validated Hilbert bit depth.
    ///
    /// # Errors
    ///
    /// Returns [`HilbertError::InvalidBitsParameter`] if `bits` is outside
    /// the supported `1..=31` range.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::ordering::{HilbertBitDepth, HilbertError};
    ///
    /// let bits = HilbertBitDepth::try_new(12)?;
    /// assert_eq!(bits.get(), 12);
    ///
    /// std::assert_matches!(
    ///     HilbertBitDepth::try_new(32),
    ///     Err(HilbertError::InvalidBitsParameter { bits: 32 })
    /// );
    /// # Ok::<(), HilbertError>(())
    /// ```
    pub const fn try_new(bits: u32) -> Result<Self, HilbertError> {
        let Some(bits) = NonZeroU32::new(bits) else {
            return Err(HilbertError::InvalidBitsParameter { bits: 0 });
        };
        if bits.get() > MAX_HILBERT_BITS {
            return Err(HilbertError::InvalidBitsParameter { bits: bits.get() });
        }
        Ok(Self(bits))
    }

    /// Returns the validated bit depth as a raw integer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::ordering::{HilbertBitDepth, HilbertError};
    ///
    /// let bits = HilbertBitDepth::try_new(16)?;
    /// assert_eq!(bits.get(), 16);
    /// # Ok::<(), HilbertError>(())
    /// ```
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

impl TryFrom<u32> for HilbertBitDepth {
    type Error = HilbertError;

    fn try_from(bits: u32) -> Result<Self, Self::Error> {
        Self::try_new(bits)
    }
}

impl fmt::Display for HilbertBitDepth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.get().fmt(f)
    }
}

/// Pre-quantized Hilbert coordinates proven to fit a selected bit-depth grid.
///
/// This borrowed wrapper carries the validation evidence for a batch of
/// caller-supplied quantized coordinates. Use [`Self::try_new`] at the boundary,
/// then call [`Self::indices`] or [`hilbert_indices_for_quantized_batch`] for an
/// infallible mapping to Hilbert indices.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::ordering::{
///     HilbertBitDepth, HilbertError, HilbertQuantizedBatch,
/// };
///
/// let quantized = [[0_u32, 0], [3, 3]];
/// let bits = HilbertBitDepth::try_new(2)?;
/// let batch = HilbertQuantizedBatch::try_new(&quantized, bits)?;
///
/// let indices = batch.indices();
/// assert_eq!(indices.len(), quantized.len());
/// # Ok::<(), HilbertError>(())
/// ```
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct HilbertQuantizedBatch<'a, const D: usize> {
    quantized: &'a [[u32; D]],
    index_mode: HilbertIndexMode<D>,
}

impl<'a, const D: usize> HilbertQuantizedBatch<'a, D> {
    /// Parses pre-quantized coordinates into a validated Hilbert batch.
    ///
    /// # Errors
    ///
    /// Returns [`HilbertError::IndexOverflow`] if `D * bits > 128` (index would not fit in `u128`).
    /// Returns [`HilbertError::DimensionTooLarge`] if the dimension `D` exceeds `u32::MAX`.
    /// Returns [`HilbertError::PrequantizedCoordinateOutOfRange`] if any pre-quantized
    /// coordinate exceeds `2^bits - 1`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::ordering::{
    ///     HilbertBitDepth, HilbertError, HilbertQuantizedBatch,
    /// };
    ///
    /// let quantized = [[0_u32, 0], [1, 2], [3, 3]];
    /// let bits = HilbertBitDepth::try_new(2)?;
    ///
    /// let batch = HilbertQuantizedBatch::try_new(&quantized, bits)?;
    /// assert_eq!(batch.coordinates(), quantized.as_slice());
    /// # Ok::<(), HilbertError>(())
    /// ```
    pub fn try_new(quantized: &'a [[u32; D]], bits: HilbertBitDepth) -> Result<Self, HilbertError> {
        let index_mode = HilbertIndexMode::try_new(bits)?;

        if D != 0 {
            validate_prequantized_coordinates(quantized, bits)?;
        }

        Ok(Self {
            quantized,
            index_mode,
        })
    }

    /// Returns the validated pre-quantized coordinates.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::ordering::{
    ///     HilbertBitDepth, HilbertError, HilbertQuantizedBatch,
    /// };
    ///
    /// let quantized = [[0_u32, 0], [3, 3]];
    /// let bits = HilbertBitDepth::try_new(2)?;
    /// let batch = HilbertQuantizedBatch::try_new(&quantized, bits)?;
    ///
    /// assert_eq!(batch.coordinates(), quantized.as_slice());
    /// # Ok::<(), HilbertError>(())
    /// ```
    #[must_use]
    pub const fn coordinates(self) -> &'a [[u32; D]] {
        self.quantized
    }

    /// Returns the bit depth whose grid bounds were checked.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::ordering::{
    ///     HilbertBitDepth, HilbertError, HilbertQuantizedBatch,
    /// };
    ///
    /// let quantized = [[0_u32, 0], [3, 3]];
    /// let bits = HilbertBitDepth::try_new(2)?;
    /// let batch = HilbertQuantizedBatch::try_new(&quantized, bits)?;
    ///
    /// assert_eq!(batch.bits(), bits);
    /// # Ok::<(), HilbertError>(())
    /// ```
    pub const fn bits(self) -> HilbertBitDepth {
        self.index_mode.bits()
    }

    /// Computes Hilbert indices without revalidating the batch.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::ordering::{
    ///     HilbertBitDepth, HilbertError, HilbertQuantizedBatch,
    /// };
    ///
    /// let quantized = [[0_u32, 0], [3, 3]];
    /// let bits = HilbertBitDepth::try_new(2)?;
    /// let batch = HilbertQuantizedBatch::try_new(&quantized, bits)?;
    ///
    /// let indices = batch.indices();
    /// assert_eq!(indices.len(), quantized.len());
    /// # Ok::<(), HilbertError>(())
    /// ```
    #[must_use]
    pub fn indices(self) -> Vec<u128> {
        hilbert_indices_for_quantized_batch(self)
    }
}

/// Owned, pre-quantized Hilbert coordinates proven in-grid by the quantizer
/// that produced them.
///
/// Unlike [`HilbertQuantizedBatch`], which borrows caller-supplied coordinates
/// and revalidates them at the boundary, this type is only constructed by
/// [`hilbert_quantize_batch_in_range`]. That constructor clamps every
/// coordinate into the selected bit-depth grid, so the in-grid invariant is
/// carried structurally by the stored, validated index mode and
/// [`Self::indices`] is infallible — no second per-coordinate scan is needed.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::CoordinateRange;
/// use delaunay::prelude::ordering::{
///     HilbertBitDepth, HilbertError, hilbert_quantize_batch_in_range,
/// };
///
/// # fn main() -> Result<(), HilbertError> {
/// let points = [[0.1_f64, 0.2], [0.9, 0.8], [0.5, 0.5]];
/// # let Ok(bounds) = CoordinateRange::try_new(0.0_f64, 1.0) else {
/// #     return Ok(());
/// # };
/// let bits = HilbertBitDepth::try_new(8)?;
///
/// let batch = hilbert_quantize_batch_in_range(&points, bounds, bits, |p| *p)?;
/// assert_eq!(batch.indices().len(), points.len());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use]
pub struct HilbertQuantizedVec<const D: usize> {
    quantized: Vec<[u32; D]>,
    index_mode: HilbertIndexMode<D>,
}

impl<const D: usize> HilbertQuantizedVec<D> {
    /// Returns the validated pre-quantized coordinates.
    #[must_use]
    pub fn coordinates(&self) -> &[[u32; D]] {
        &self.quantized
    }

    /// Returns the bit depth whose grid the coordinates were quantized to.
    pub const fn bits(&self) -> HilbertBitDepth {
        self.index_mode.bits()
    }

    /// Returns the number of quantized points in the batch.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.quantized.len()
    }

    /// Returns `true` when the batch contains no points.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.quantized.is_empty()
    }

    /// Computes Hilbert indices without revalidating the batch.
    ///
    /// This is infallible: the constructor already validated the index width
    /// and clamped every coordinate into the selected bit-depth grid.
    #[must_use]
    pub fn indices(&self) -> Vec<u128> {
        indices_for_mode(&self.quantized, self.index_mode)
    }

    /// Consumes the batch, returning its Hilbert indices alongside the owned
    /// quantized coordinates.
    ///
    /// This fuses the two products batch callers usually need — the per-point
    /// Hilbert index and the quantized cell used as a sort tie-break — without
    /// an extra allocation or a per-coordinate revalidation pass.
    #[must_use]
    pub fn into_indices_and_coordinates(self) -> (Vec<u128>, Vec<[u32; D]>) {
        let indices = self.indices();
        (indices, self.quantized)
    }

    /// Consumes the batch, returning only the owned quantized coordinates.
    #[must_use]
    pub fn into_coordinates(self) -> Vec<[u32; D]> {
        self.quantized
    }
}

/// Quantizes a batch of items into an owned, proof-bearing Hilbert batch.
///
/// Coordinates are extracted with `coords_of`, normalized against `bounds`
/// (already parsed at an upstream boundary), and clamped into the
/// `0..=2^bits - 1` grid. The index width and quantization scale are validated
/// once for the whole batch, after which [`HilbertQuantizedVec::indices`] is
/// infallible.
///
/// This is the single-pass bulk entry point preferred by construction
/// preprocessing: it avoids both the per-item bound parsing of
/// [`try_hilbert_quantize`] and the per-coordinate revalidation that
/// [`hilbert_indices_prequantized`] performs on caller-supplied grids.
///
/// # Errors
///
/// Returns [`HilbertError::IndexOverflow`] if `D * bits > 128` (index would not fit in `u128`).
/// Returns [`HilbertError::DimensionTooLarge`] if the dimension `D` exceeds `u32::MAX`.
/// Returns [`HilbertError::NonFiniteBoundsExtent`], [`HilbertError::NonFiniteCoordinate`],
/// [`HilbertError::NonFiniteNormalizedCoordinate`], or
/// [`HilbertError::QuantizedCoordinateConversionFailed`] if coordinate
/// quantization fails.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::CoordinateRange;
/// use delaunay::prelude::ordering::{
///     HilbertBitDepth, HilbertError, hilbert_quantize_batch_in_range,
/// };
///
/// # fn main() -> Result<(), HilbertError> {
/// let points = [[0.0_f64, 0.0], [1.0, 1.0]];
/// # let Ok(bounds) = CoordinateRange::try_new(0.0_f64, 1.0) else {
/// #     return Ok(());
/// # };
/// let batch =
///     hilbert_quantize_batch_in_range(&points, bounds, HilbertBitDepth::try_new(4)?, |p| *p)?;
///
/// let (indices, quantized) = batch.into_indices_and_coordinates();
/// assert_eq!(indices.len(), 2);
/// assert_eq!(quantized.len(), 2);
/// # Ok(())
/// # }
/// ```
pub fn hilbert_quantize_batch_in_range<Item, const D: usize>(
    items: &[Item],
    bounds: CoordinateRange<f64>,
    bits: HilbertBitDepth,
    mut coords_of: impl FnMut(&Item) -> [f64; D],
) -> Result<HilbertQuantizedVec<D>, HilbertError> {
    let index_mode = HilbertIndexMode::try_new(bits)?;

    if D == 0 {
        return Ok(HilbertQuantizedVec {
            quantized: vec![[0_u32; D]; items.len()],
            index_mode,
        });
    }

    let (max_val_u32, max_val_t) = quantization_scale(bits);

    let quantized = items
        .iter()
        .map(|item| {
            let coords = coords_of(item);
            quantize_with_scale(&coords, bounds, bits, max_val_u32, max_val_t)
        })
        .collect::<Result<Vec<[u32; D]>, HilbertError>>()?;

    Ok(HilbertQuantizedVec {
        quantized,
        index_mode,
    })
}

/// Converts a validated bit depth into the exact f64 grid maximum.
fn quantization_scale(bits: HilbertBitDepth) -> (u32, f64) {
    let max_grid_value = max_quantized_coordinate(bits);
    (max_grid_value, f64::from(max_grid_value))
}

/// Returns the largest coordinate accepted by Hilbert APIs for the selected grid.
///
/// Keeping this calculation shared prevents the quantization and pre-quantized
/// validation paths from drifting on the inclusive `0..=2^bits - 1` contract.
const fn max_quantized_coordinate(bits: HilbertBitDepth) -> u32 {
    (1_u32 << bits.get()) - 1
}

/// Computes encoded index width once so overflow errors report consistent context.
fn total_bits<const D: usize>(bits: HilbertBitDepth) -> Result<u128, HilbertError> {
    let d_u32 = u32::try_from(D).map_err(|_| HilbertError::DimensionTooLarge { dimension: D })?;
    Ok(u128::from(d_u32) * u128::from(bits.get()))
}

/// Positive-dimensional Hilbert index parameters after width validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HilbertIndexParams<const D: usize> {
    bits: HilbertBitDepth,
}

impl<const D: usize> HilbertIndexParams<D> {
    const fn bits(self) -> u32 {
        self.bits.get()
    }
}

/// Validated Hilbert indexing mode, including the zero-dimensional special case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HilbertIndexMode<const D: usize> {
    ZeroDimensional { bits: HilbertBitDepth },
    Positive(HilbertIndexParams<D>),
}

impl<const D: usize> HilbertIndexMode<D> {
    fn try_new(bits: HilbertBitDepth) -> Result<Self, HilbertError> {
        validate_index_width::<D>(bits)?;
        if D == 0 {
            Ok(Self::ZeroDimensional { bits })
        } else {
            Ok(Self::Positive(HilbertIndexParams { bits }))
        }
    }

    const fn bits(self) -> HilbertBitDepth {
        match self {
            Self::ZeroDimensional { bits } => bits,
            Self::Positive(params) => params.bits,
        }
    }
}

/// Centralizes index-width validation shared by indexing and ordering APIs.
fn validate_index_width<const D: usize>(bits: HilbertBitDepth) -> Result<(), HilbertError> {
    let total_bits = total_bits::<D>(bits)?;
    if total_bits > 128 {
        return Err(HilbertError::IndexOverflow {
            dimension: D,
            bits: bits.get(),
            total_bits,
        });
    }
    Ok(())
}

fn parse_hilbert_bounds(bounds: (f64, f64)) -> Result<CoordinateRange<f64>, HilbertError> {
    let lower_bound_finite = bounds.0.is_finite();
    let upper_bound_finite = bounds.1.is_finite();

    CoordinateRange::try_from(bounds).map_err(|error| match error {
        CoordinateRangeError::NonFiniteBound { .. } => HilbertError::NonFiniteBounds {
            lower_bound_finite,
            upper_bound_finite,
        },
        CoordinateRangeError::NonIncreasing { ordering, .. } => {
            HilbertError::NonIncreasingBounds { ordering }
        }
    })
}

/// Quantize D-dimensional coordinates into integer grid coordinates in `[0, 2^bits)`.
///
/// The coordinates are normalized using a scalar `(min, max)` bound applied to every
/// dimension and then clamped to `[0, 1]` before quantization.
///
/// # Errors
///
/// Returns [`HilbertError::NonFiniteBounds`],
/// [`HilbertError::NonIncreasingBounds`],
/// [`HilbertError::NonFiniteBoundsExtent`],
/// [`HilbertError::NonFiniteCoordinate`], or
/// [`HilbertError::NonFiniteNormalizedCoordinate`] if quantization input or
/// normalization arithmetic is non-finite.
///
/// Returns [`HilbertError::QuantizedCoordinateConversionFailed`] if a rounded
/// scaled coordinate cannot be represented as `u32`.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::ordering::{HilbertBitDepth, HilbertError, try_hilbert_quantize};
///
/// let coords = [0.5_f64, 0.25];
/// let q = try_hilbert_quantize(&coords, (0.0, 1.0), HilbertBitDepth::try_new(2)?)?;
/// assert!(q[0] <= 3 && q[1] <= 3);
/// # Ok::<(), HilbertError>(())
/// ```
pub fn try_hilbert_quantize<const D: usize>(
    coords: &[f64; D],
    bounds: (f64, f64),
    bits: HilbertBitDepth,
) -> Result<[u32; D], HilbertError> {
    let bounds = parse_hilbert_bounds(bounds)?;

    if D == 0 {
        return Ok([0_u32; D]);
    }

    let (max_val_u32, max_val_t) = quantization_scale(bits);

    quantize_with_scale(coords, bounds, bits, max_val_u32, max_val_t)
}

/// Quantizes coordinates against bounds already parsed by an upstream boundary.
///
/// # Errors
///
/// Returns [`HilbertError::NonFiniteBoundsExtent`] if the validated bounds
/// produce a non-finite extent. Returns [`HilbertError::NonFiniteCoordinate`] or
/// [`HilbertError::NonFiniteNormalizedCoordinate`] if a coordinate or its
/// normalized value is non-finite. Returns
/// [`HilbertError::QuantizedCoordinateConversionFailed`] if a rounded scaled
/// coordinate cannot be represented as `u32`.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::CoordinateRange;
/// use delaunay::prelude::ordering::{
///     HilbertBitDepth, HilbertError, hilbert_quantize_in_range,
/// };
///
/// # fn main() -> Result<(), HilbertError> {
/// # let Ok(bounds) = CoordinateRange::try_new(0.0_f64, 1.0) else {
/// #     return Ok(());
/// # };
/// let q = hilbert_quantize_in_range(&[0.5_f64, 0.25], bounds, HilbertBitDepth::try_new(2)?)?;
/// assert!(q[0] <= 3 && q[1] <= 3);
/// # Ok(())
/// # }
/// ```
pub fn hilbert_quantize_in_range<const D: usize>(
    coords: &[f64; D],
    bounds: CoordinateRange<f64>,
    bits: HilbertBitDepth,
) -> Result<[u32; D], HilbertError> {
    if D == 0 {
        return Ok([0_u32; D]);
    }

    let (max_val_u32, max_val_t) = quantization_scale(bits);
    quantize_with_scale(coords, bounds, bits, max_val_u32, max_val_t)
}

/// Quantizes coordinates with a precomputed scalar grid maximum so hot callers
/// can validate conversion once before sorting or batch index generation.
#[inline]
fn quantize_with_scale<const D: usize>(
    coords: &[f64; D],
    bounds: CoordinateRange<f64>,
    bits: HilbertBitDepth,
    max_val_u32: u32,
    max_val_t: f64,
) -> Result<[u32; D], HilbertError> {
    let min = bounds.min();
    let max = bounds.max();
    let extent = max - min;
    if !extent.is_finite() {
        return Err(HilbertError::NonFiniteBoundsExtent {});
    }

    let mut quantized = [0_u32; D];
    for (i, &coord) in coords.iter().enumerate() {
        if !coord.is_finite() {
            return Err(HilbertError::NonFiniteCoordinate {
                coordinate_index: i,
            });
        }

        let t = (coord - min) / extent;
        let normalized = if t.is_finite() {
            t.clamp(0.0, 1.0)
        } else {
            return Err(HilbertError::NonFiniteNormalizedCoordinate {
                coordinate_index: i,
            });
        };

        let scaled = normalized * max_val_t;
        // Round to nearest grid cell (instead of truncating) for fairer distribution.
        let Some(value) = scaled.round().to_u32() else {
            return Err(HilbertError::QuantizedCoordinateConversionFailed {
                bits: bits.get(),
                max_grid_value: max_val_u32,
                coordinate_index: i,
            });
        };
        let q = value.min(max_val_u32);
        quantized[i] = q;
    }

    Ok(quantized)
}

/// Applies a prevalidated permutation after key construction succeeds so sort
/// helpers never partially reorder items before returning a Hilbert error.
fn apply_order<Item>(
    items: &mut [Item],
    order: impl ExactSizeIterator<Item = usize>,
) -> Result<(), HilbertError> {
    let item_len = items.len();
    let permutation_count = order.len();
    if item_len != permutation_count {
        return Err(HilbertError::InvalidSortPermutationLength {
            item_count: item_len,
            permutation_count,
        });
    }

    let mut ranks = vec![usize::MAX; item_len];
    let mut observed_count = 0_usize;
    for (new_index, old_index) in order.into_iter().enumerate() {
        observed_count = new_index + 1;
        if new_index >= item_len {
            return Err(HilbertError::InvalidSortPermutationLength {
                item_count: item_len,
                permutation_count: observed_count,
            });
        }
        if old_index >= item_len {
            return Err(HilbertError::InvalidSortPermutationIndex {
                permutation_index: new_index,
                item_index: old_index,
                item_count: item_len,
            });
        }
        if ranks[old_index] != usize::MAX {
            return Err(HilbertError::InvalidSortPermutationDuplicate {
                permutation_index: new_index,
                item_index: old_index,
            });
        }
        ranks[old_index] = new_index;
    }
    if observed_count != item_len {
        return Err(HilbertError::InvalidSortPermutationLength {
            item_count: item_len,
            permutation_count: observed_count,
        });
    }

    for index in 0..item_len {
        while ranks[index] != index {
            let target = ranks[index];
            items.swap(index, target);
            ranks.swap(index, target);
        }
    }

    Ok(())
}

/// Compute the Hilbert curve index for a point in D-dimensional space.
///
/// Internally, coordinates are quantized to an integer grid and then mapped to a
/// single index using an iterative Gray-code based algorithm.
///
/// # Errors
///
/// Returns [`HilbertError::IndexOverflow`] if `D * bits > 128` (index would not fit in `u128`).
///
/// Returns [`HilbertError::DimensionTooLarge`] if the dimension `D` exceeds `u32::MAX`
/// (extremely unlikely in practice).
///
/// Returns [`HilbertError::NonFiniteBounds`],
/// [`HilbertError::NonIncreasingBounds`],
/// [`HilbertError::NonFiniteBoundsExtent`],
/// [`HilbertError::NonFiniteCoordinate`], or
/// [`HilbertError::NonFiniteNormalizedCoordinate`] if quantization input or
/// normalization arithmetic is non-finite.
///
/// Returns [`HilbertError::QuantizedCoordinateConversionFailed`] if a rounded
/// scaled coordinate cannot be represented as `u32`.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::ordering::{HilbertBitDepth, HilbertError, try_hilbert_index};
///
/// let idx = try_hilbert_index(&[0.0_f64, 0.0], (0.0, 1.0), HilbertBitDepth::try_new(4)?)?;
/// assert_eq!(idx, 0);
/// # Ok::<(), HilbertError>(())
/// ```
pub fn try_hilbert_index<const D: usize>(
    coords: &[f64; D],
    bounds: (f64, f64),
    bits: HilbertBitDepth,
) -> Result<u128, HilbertError> {
    let bounds = parse_hilbert_bounds(bounds)?;
    hilbert_index_in_range(coords, bounds, bits)
}

/// Computes a Hilbert index against bounds already parsed by an upstream boundary.
///
/// # Errors
///
/// Returns [`HilbertError::IndexOverflow`] if `D * bits > 128` (index would not fit in `u128`).
/// Returns [`HilbertError::DimensionTooLarge`] if the dimension `D` exceeds `u32::MAX`.
/// Returns [`HilbertError::NonFiniteBoundsExtent`], [`HilbertError::NonFiniteCoordinate`],
/// [`HilbertError::NonFiniteNormalizedCoordinate`], or
/// [`HilbertError::QuantizedCoordinateConversionFailed`] if coordinate
/// quantization fails.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::CoordinateRange;
/// use delaunay::prelude::ordering::{HilbertBitDepth, HilbertError, hilbert_index_in_range};
///
/// # fn main() -> Result<(), HilbertError> {
/// # let Ok(bounds) = CoordinateRange::try_new(0.0_f64, 1.0) else {
/// #     return Ok(());
/// # };
/// let idx = hilbert_index_in_range(&[0.0_f64, 0.0], bounds, HilbertBitDepth::try_new(4)?)?;
/// assert_eq!(idx, 0);
/// # Ok(())
/// # }
/// ```
pub fn hilbert_index_in_range<const D: usize>(
    coords: &[f64; D],
    bounds: CoordinateRange<f64>,
    bits: HilbertBitDepth,
) -> Result<u128, HilbertError> {
    let index_mode = HilbertIndexMode::try_new(bits)?;
    let HilbertIndexMode::Positive(index_params) = index_mode else {
        return Ok(0);
    };

    let q = hilbert_quantize_in_range(coords, bounds, bits)?;
    Ok(index_from_quantized(&q, index_params))
}

/// Compute Hilbert index from pre-quantized integer coordinates.
///
/// This uses the Skilling (2004) algorithm ("Programming the Hilbert curve") to map
/// `D` integer coordinates (each `bits` bits wide) to a single Hilbert index.
///
/// The resulting ordering is continuous on the integer grid (successive indices move to
/// adjacent cells).
#[must_use]
fn index_from_quantized<const D: usize>(coords: &[u32; D], params: HilbertIndexParams<D>) -> u128 {
    let bits = params.bits();

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
/// Returns [`HilbertError::IndexOverflow`] if `D * bits > 128` (index would not fit in `u128`).
///
/// Returns [`HilbertError::DimensionTooLarge`] if the dimension `D` exceeds `u32::MAX`
/// (extremely unlikely in practice).
///
/// Returns [`HilbertError::NonFiniteBounds`],
/// [`HilbertError::NonIncreasingBounds`],
/// [`HilbertError::NonFiniteBoundsExtent`],
/// [`HilbertError::NonFiniteCoordinate`], or
/// [`HilbertError::NonFiniteNormalizedCoordinate`] if quantization input or
/// normalization arithmetic is non-finite.
///
/// Returns [`HilbertError::QuantizedCoordinateConversionFailed`] if a rounded
/// scaled coordinate cannot be represented as `u32`.
///
/// Returns [`HilbertError::InvalidSortPermutationLength`],
/// [`HilbertError::InvalidSortPermutationIndex`], or
/// [`HilbertError::InvalidSortPermutationDuplicate`] if an internally
/// constructed permutation is inconsistent.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::ordering::{HilbertBitDepth, HilbertError, try_hilbert_sort_by_stable};
///
/// let mut points = vec![[0.9_f64, 0.9], [0.1, 0.1], [0.5, 0.5]];
/// try_hilbert_sort_by_stable(&mut points, (0.0, 1.0), HilbertBitDepth::try_new(8)?, |p| *p)?;
/// assert_eq!(points[0], [0.1, 0.1]);
/// # Ok::<(), HilbertError>(())
/// ```
pub fn try_hilbert_sort_by_stable<Item, const D: usize>(
    items: &mut [Item],
    bounds: (f64, f64),
    bits: HilbertBitDepth,
    coords_of: impl FnMut(&Item) -> [f64; D],
) -> Result<(), HilbertError> {
    let bounds = parse_hilbert_bounds(bounds)?;
    hilbert_sort_by_stable_in_range(items, bounds, bits, coords_of)
}

/// Stable sort helper using bounds already parsed by an upstream boundary.
///
/// This is equivalent to [`try_hilbert_sort_by_stable`], but accepts a
/// [`CoordinateRange`] so callers can carry range validation evidence inward.
///
/// # Errors
///
/// Returns [`HilbertError::IndexOverflow`] if `D * bits > 128` (index would not fit in `u128`).
/// Returns [`HilbertError::DimensionTooLarge`] if the dimension `D` exceeds `u32::MAX`.
/// Returns [`HilbertError::NonFiniteBoundsExtent`], [`HilbertError::NonFiniteCoordinate`],
/// [`HilbertError::NonFiniteNormalizedCoordinate`], or
/// [`HilbertError::QuantizedCoordinateConversionFailed`] if coordinate
/// quantization fails.
///
/// Returns [`HilbertError::InvalidSortPermutationLength`],
/// [`HilbertError::InvalidSortPermutationIndex`], or
/// [`HilbertError::InvalidSortPermutationDuplicate`] if an internally
/// constructed permutation is inconsistent.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::CoordinateRange;
/// use delaunay::prelude::ordering::{
///     HilbertBitDepth, HilbertError, hilbert_sort_by_stable_in_range,
/// };
///
/// # fn main() -> Result<(), HilbertError> {
/// let mut points = vec![[0.9_f64, 0.9], [0.1, 0.1], [0.5, 0.5]];
/// # let Ok(bounds) = CoordinateRange::try_new(0.0_f64, 1.0) else {
/// #     return Ok(());
/// # };
/// hilbert_sort_by_stable_in_range(&mut points, bounds, HilbertBitDepth::try_new(8)?, |p| *p)?;
/// assert_eq!(points[0], [0.1, 0.1]);
/// # Ok(())
/// # }
/// ```
pub fn hilbert_sort_by_stable_in_range<Item, const D: usize>(
    items: &mut [Item],
    bounds: CoordinateRange<f64>,
    bits: HilbertBitDepth,
    mut coords_of: impl FnMut(&Item) -> [f64; D],
) -> Result<(), HilbertError> {
    let index_mode = HilbertIndexMode::try_new(bits)?;
    let HilbertIndexMode::Positive(index_params) = index_mode else {
        return Ok(());
    };

    let (max_val_u32, max_val_t) = quantization_scale(bits);

    let mut keyed: Vec<((u128, [u32; D]), usize)> = items
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let c = coords_of(item);
            let q = quantize_with_scale(&c, bounds, bits, max_val_u32, max_val_t)?;
            let idx = index_from_quantized(&q, index_params);
            Ok(((idx, q), i))
        })
        .collect::<Result<_, HilbertError>>()?;

    keyed.sort_by_key(|(key, _)| *key);
    apply_order(items, keyed.into_iter().map(|(_, i)| i))?;

    Ok(())
}

/// Unstable sort helper: sort items by Hilbert index + quantized-coordinate tie-break.
///
/// This precomputes fallible Hilbert keys once, then applies an unstable ordering.
/// Prefer [`try_hilbert_sort_by_stable`] when equal-key items must preserve their
/// original relative order.
///
/// When `D == 0`, all items are considered equivalent (index 0) and the sort order is
/// implementation-defined.
///
/// # Errors
///
/// Returns [`HilbertError::IndexOverflow`] if `D * bits > 128` (index would not fit in `u128`).
///
/// Returns [`HilbertError::DimensionTooLarge`] if the dimension `D` exceeds `u32::MAX`
/// (extremely unlikely in practice).
///
/// Returns [`HilbertError::NonFiniteBounds`],
/// [`HilbertError::NonIncreasingBounds`],
/// [`HilbertError::NonFiniteBoundsExtent`],
/// [`HilbertError::NonFiniteCoordinate`], or
/// [`HilbertError::NonFiniteNormalizedCoordinate`] if quantization input or
/// normalization arithmetic is non-finite.
///
/// Returns [`HilbertError::QuantizedCoordinateConversionFailed`] if a rounded
/// scaled coordinate cannot be represented as `u32`.
///
/// Returns [`HilbertError::InvalidSortPermutationLength`],
/// [`HilbertError::InvalidSortPermutationIndex`], or
/// [`HilbertError::InvalidSortPermutationDuplicate`] if an internally
/// constructed permutation is inconsistent.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::ordering::{HilbertBitDepth, HilbertError, try_hilbert_sort_by_unstable};
///
/// let mut points = vec![[0.9_f64, 0.9], [0.1, 0.1], [0.5, 0.5]];
/// try_hilbert_sort_by_unstable(&mut points, (0.0, 1.0), HilbertBitDepth::try_new(8)?, |p| *p)?;
/// assert_eq!(points[0], [0.1, 0.1]);
/// # Ok::<(), HilbertError>(())
/// ```
pub fn try_hilbert_sort_by_unstable<Item, const D: usize>(
    items: &mut [Item],
    bounds: (f64, f64),
    bits: HilbertBitDepth,
    coords_of: impl FnMut(&Item) -> [f64; D],
) -> Result<(), HilbertError> {
    let bounds = parse_hilbert_bounds(bounds)?;
    hilbert_sort_by_unstable_in_range(items, bounds, bits, coords_of)
}

/// Unstable sort helper using bounds already parsed by an upstream boundary.
///
/// This is equivalent to [`try_hilbert_sort_by_unstable`], but accepts a
/// [`CoordinateRange`] so callers can carry range validation evidence inward.
///
/// # Errors
///
/// Returns [`HilbertError::IndexOverflow`] if `D * bits > 128` (index would not fit in `u128`).
/// Returns [`HilbertError::DimensionTooLarge`] if the dimension `D` exceeds `u32::MAX`.
/// Returns [`HilbertError::NonFiniteBoundsExtent`], [`HilbertError::NonFiniteCoordinate`],
/// [`HilbertError::NonFiniteNormalizedCoordinate`], or
/// [`HilbertError::QuantizedCoordinateConversionFailed`] if coordinate
/// quantization fails.
///
/// Returns [`HilbertError::InvalidSortPermutationLength`],
/// [`HilbertError::InvalidSortPermutationIndex`], or
/// [`HilbertError::InvalidSortPermutationDuplicate`] if an internally
/// constructed permutation is inconsistent.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::CoordinateRange;
/// use delaunay::prelude::ordering::{
///     HilbertBitDepth, HilbertError, hilbert_sort_by_unstable_in_range,
/// };
///
/// # fn main() -> Result<(), HilbertError> {
/// let mut points = vec![[0.9_f64, 0.9], [0.1, 0.1], [0.5, 0.5]];
/// # let Ok(bounds) = CoordinateRange::try_new(0.0_f64, 1.0) else {
/// #     return Ok(());
/// # };
/// hilbert_sort_by_unstable_in_range(&mut points, bounds, HilbertBitDepth::try_new(8)?, |p| *p)?;
/// assert_eq!(points[0], [0.1, 0.1]);
/// # Ok(())
/// # }
/// ```
pub fn hilbert_sort_by_unstable_in_range<Item, const D: usize>(
    items: &mut [Item],
    bounds: CoordinateRange<f64>,
    bits: HilbertBitDepth,
    mut coords_of: impl FnMut(&Item) -> [f64; D],
) -> Result<(), HilbertError> {
    let index_mode = HilbertIndexMode::try_new(bits)?;
    let HilbertIndexMode::Positive(index_params) = index_mode else {
        return Ok(());
    };

    let (max_val_u32, max_val_t) = quantization_scale(bits);

    let mut keyed: Vec<((u128, [u32; D]), usize)> = items
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let c = coords_of(item);
            let q = quantize_with_scale(&c, bounds, bits, max_val_u32, max_val_t)?;
            let idx = index_from_quantized(&q, index_params);
            Ok(((idx, q), i))
        })
        .collect::<Result<_, HilbertError>>()?;

    keyed.sort_unstable_by_key(|(key, _)| *key);
    apply_order(items, keyed.into_iter().map(|(_, i)| i))?;

    Ok(())
}

/// Validates that caller-supplied pre-quantized coordinates fit the selected grid.
///
/// This protects [`hilbert_indices_prequantized`] from passing high-bit values
/// into [`index_from_quantized`], where bits above the selected depth are
/// intentionally ignored by the Hilbert interleaving loop.
fn validate_prequantized_coordinates<const D: usize>(
    quantized: &[[u32; D]],
    bits: HilbertBitDepth,
) -> Result<(), HilbertError> {
    let max_grid_value = max_quantized_coordinate(bits);
    for (point_index, point) in quantized.iter().enumerate() {
        for (coordinate_index, &coordinate) in point.iter().enumerate() {
            if coordinate > max_grid_value {
                return Err(HilbertError::PrequantizedCoordinateOutOfRange {
                    bits: bits.get(),
                    max_grid_value,
                    point_index,
                    coordinate_index,
                    coordinate,
                });
            }
        }
    }

    Ok(())
}

/// Compute Hilbert indices for a batch of pre-quantized coordinates.
///
/// This is a bulk API that avoids recomputing quantization parameters for large
/// insertion batches. When inserting many points, quantize them once using
/// [`try_hilbert_quantize`] and then call this function to compute all indices in bulk.
/// Pre-quantized coordinates must be in the inclusive range `0..=2^bits - 1`;
/// values outside that grid are rejected instead of being truncated.
///
/// # Performance
///
/// This function validates index width and pre-quantized coordinate ranges, then
/// maps each quantized coordinate through the internal Hilbert index computation.
/// For large batches, this is significantly faster than calling [`try_hilbert_index`]
/// individually for each point.
/// If the same pre-quantized batch is reused, construct a [`HilbertQuantizedBatch`]
/// once and call [`HilbertQuantizedBatch::indices`] to avoid repeated validation.
///
/// # Errors
///
/// Returns [`HilbertError::IndexOverflow`] if `D * bits > 128` (index would not fit in `u128`).
///
/// Returns [`HilbertError::DimensionTooLarge`] if the dimension `D` exceeds `u32::MAX`
/// (extremely unlikely in practice).
///
/// Returns [`HilbertError::PrequantizedCoordinateOutOfRange`] if any pre-quantized
/// coordinate exceeds `2^bits - 1`.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::ordering::{
///     HilbertBitDepth, HilbertError, hilbert_indices_prequantized, try_hilbert_quantize,
/// };
///
/// let coords = vec![[0.1_f64, 0.2], [0.5, 0.5], [0.9, 0.8]];
/// let bounds = (0.0, 1.0);
/// let bits = HilbertBitDepth::try_new(8)?;
///
/// // Quantize once
/// let quantized: Vec<[u32; 2]> = coords
///     .iter()
///     .map(|c| try_hilbert_quantize(c, bounds, bits))
///     .collect::<Result<_, _>>()?;
///
/// // Compute all indices in bulk
/// let indices = hilbert_indices_prequantized(&quantized, bits)?;
/// assert_eq!(indices.len(), coords.len());
/// # Ok::<(), HilbertError>(())
/// ```
///
/// Error handling:
///
/// ```rust
/// use delaunay::prelude::ordering::{
///     HilbertBitDepth, HilbertError, hilbert_indices_prequantized,
/// };
///
/// # fn main() -> Result<(), HilbertError> {
/// let quantized = vec![[1_u32, 2]];
///
/// // Zero cannot cross the typed API boundary.
/// std::assert_matches!(
///     HilbertBitDepth::try_new(0),
///     Err(HilbertError::InvalidBitsParameter { bits: 0 })
/// );
///
/// // Overflow (D=5, bits=26 => 130 > 128)
/// let quantized_5d = vec![[1_u32, 2, 3, 4, 5]];
/// let result = hilbert_indices_prequantized(&quantized_5d, HilbertBitDepth::try_new(26)?);
/// std::assert_matches!(result, Err(HilbertError::IndexOverflow { .. }));
///
/// // Pre-quantized coordinates must fit the selected grid.
/// let out_of_range = vec![[4_u32, 1]];
/// let result = hilbert_indices_prequantized(&out_of_range, HilbertBitDepth::try_new(2)?);
/// std::assert_matches!(
///     result,
///     Err(HilbertError::PrequantizedCoordinateOutOfRange { .. })
/// );
/// # Ok(())
/// # }
/// ```
pub fn hilbert_indices_prequantized<const D: usize>(
    quantized: &[[u32; D]],
    bits: HilbertBitDepth,
) -> Result<Vec<u128>, HilbertError> {
    Ok(HilbertQuantizedBatch::try_new(quantized, bits)?.indices())
}

/// Computes Hilbert indices from a validated pre-quantized batch.
///
/// This is the infallible companion to [`hilbert_indices_prequantized`]. The
/// [`HilbertQuantizedBatch`] constructor has already checked both the Hilbert
/// index width and every coordinate against the selected bit-depth grid.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::ordering::{
///     HilbertBitDepth, HilbertError, HilbertQuantizedBatch,
///     hilbert_indices_for_quantized_batch,
/// };
///
/// let quantized = [[0_u32, 0], [3, 3]];
/// let bits = HilbertBitDepth::try_new(2)?;
/// let batch = HilbertQuantizedBatch::try_new(&quantized, bits)?;
///
/// let indices = hilbert_indices_for_quantized_batch(batch);
/// assert_eq!(indices.len(), quantized.len());
/// # Ok::<(), HilbertError>(())
/// ```
#[must_use]
pub fn hilbert_indices_for_quantized_batch<const D: usize>(
    batch: HilbertQuantizedBatch<'_, D>,
) -> Vec<u128> {
    indices_for_mode(batch.quantized, batch.index_mode)
}

/// Maps validated quantized coordinates to Hilbert indices for a known index
/// mode.
///
/// Both [`HilbertQuantizedBatch`] and [`HilbertQuantizedVec`] carry a validated
/// [`HilbertIndexMode`], so neither revalidates coordinates before computing
/// indices.
fn indices_for_mode<const D: usize>(
    quantized: &[[u32; D]],
    index_mode: HilbertIndexMode<D>,
) -> Vec<u128> {
    match index_mode {
        HilbertIndexMode::ZeroDimensional { .. } => vec![0_u128; quantized.len()],
        HilbertIndexMode::Positive(index_params) => quantized
            .iter()
            .map(|q| index_from_quantized(q, index_params))
            .collect(),
    }
}

/// Return the indices that would sort `coords` by Hilbert order.
///
/// When `D == 0`, all coordinates are considered equivalent (index 0) and the returned
/// indices preserve the original order.
///
/// # Errors
///
/// Returns [`HilbertError::IndexOverflow`] if `D * bits > 128` (index would not fit in `u128`).
///
/// Returns [`HilbertError::DimensionTooLarge`] if the dimension `D` exceeds `u32::MAX`
/// (extremely unlikely in practice).
///
/// Returns [`HilbertError::NonFiniteBounds`],
/// [`HilbertError::NonIncreasingBounds`],
/// [`HilbertError::NonFiniteBoundsExtent`],
/// [`HilbertError::NonFiniteCoordinate`], or
/// [`HilbertError::NonFiniteNormalizedCoordinate`] if quantization input or
/// normalization arithmetic is non-finite.
///
/// Returns [`HilbertError::QuantizedCoordinateConversionFailed`] if a rounded
/// scaled coordinate cannot be represented as `u32`.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::ordering::{HilbertBitDepth, HilbertError, try_hilbert_sorted_indices};
///
/// let coords = vec![[0.9_f64, 0.9], [0.1, 0.1], [0.5, 0.5]];
/// let order = try_hilbert_sorted_indices(&coords, (0.0, 1.0), HilbertBitDepth::try_new(8)?)?;
/// assert_eq!(order.len(), coords.len());
/// # Ok::<(), HilbertError>(())
/// ```
pub fn try_hilbert_sorted_indices<const D: usize>(
    coords: &[[f64; D]],
    bounds: (f64, f64),
    bits: HilbertBitDepth,
) -> Result<Vec<usize>, HilbertError> {
    let bounds = parse_hilbert_bounds(bounds)?;
    hilbert_sorted_indices_in_range(coords, bounds, bits)
}

/// Return the indices that would sort `coords` by Hilbert order using validated bounds.
///
/// This is equivalent to [`try_hilbert_sorted_indices`], but accepts a
/// [`CoordinateRange`] so callers can carry range validation evidence inward.
///
/// # Errors
///
/// Returns [`HilbertError::IndexOverflow`] if `D * bits > 128` (index would not fit in `u128`).
/// Returns [`HilbertError::DimensionTooLarge`] if the dimension `D` exceeds `u32::MAX`.
/// Returns [`HilbertError::NonFiniteBoundsExtent`], [`HilbertError::NonFiniteCoordinate`],
/// [`HilbertError::NonFiniteNormalizedCoordinate`], or
/// [`HilbertError::QuantizedCoordinateConversionFailed`] if coordinate
/// quantization fails.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::CoordinateRange;
/// use delaunay::prelude::ordering::{
///     HilbertBitDepth, HilbertError, hilbert_sorted_indices_in_range,
/// };
///
/// # fn main() -> Result<(), HilbertError> {
/// let coords = vec![[0.9_f64, 0.9], [0.1, 0.1], [0.5, 0.5]];
/// # let Ok(bounds) = CoordinateRange::try_new(0.0_f64, 1.0) else {
/// #     return Ok(());
/// # };
/// let order = hilbert_sorted_indices_in_range(&coords, bounds, HilbertBitDepth::try_new(8)?)?;
/// assert_eq!(order.len(), coords.len());
/// # Ok(())
/// # }
/// ```
pub fn hilbert_sorted_indices_in_range<const D: usize>(
    coords: &[[f64; D]],
    bounds: CoordinateRange<f64>,
    bits: HilbertBitDepth,
) -> Result<Vec<usize>, HilbertError> {
    let index_mode = HilbertIndexMode::try_new(bits)?;
    let HilbertIndexMode::Positive(index_params) = index_mode else {
        return Ok((0..coords.len()).collect());
    };

    let (max_val_u32, max_val_t) = quantization_scale(bits);

    let mut keyed: Vec<((u128, [u32; D]), usize)> = coords
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let q = quantize_with_scale(c, bounds, bits, max_val_u32, max_val_t)?;
            let idx = index_from_quantized(&q, index_params);
            Ok(((idx, q), i))
        })
        .collect::<Result<_, HilbertError>>()?;

    keyed.sort_by(|(ka, ia), (kb, ib)| ka.cmp(kb).then_with(|| ia.cmp(ib)));
    Ok(keyed.into_iter().map(|(_, i)| i).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::assert_matches;

    use crate::geometry::point::Point;

    fn bit_depth(value: u32) -> HilbertBitDepth {
        HilbertBitDepth::try_new(value).expect("test bit depth must be valid")
    }

    fn positive_index_params<const D: usize>(bits: HilbertBitDepth) -> HilbertIndexParams<D> {
        let HilbertIndexMode::Positive(params) =
            HilbertIndexMode::<D>::try_new(bits).expect("test index parameters must be valid")
        else {
            panic!("test index parameters must be positive-dimensional");
        };
        params
    }

    struct LyingOrder {
        values: std::vec::IntoIter<usize>,
        reported_len: usize,
    }

    impl LyingOrder {
        fn new(values: Vec<usize>, reported_len: usize) -> Self {
            Self {
                values: values.into_iter(),
                reported_len,
            }
        }
    }

    impl Iterator for LyingOrder {
        type Item = usize;

        fn next(&mut self) -> Option<Self::Item> {
            self.values.next()
        }
    }

    impl ExactSizeIterator for LyingOrder {
        fn len(&self) -> usize {
            self.reported_len
        }
    }

    /// Asserts that the bulk pre-quantized API matches per-point Hilbert indexing.
    fn assert_prequantized_matches_hilbert_index<const D: usize>(
        coords: &[[f64; D]],
        bounds: (f64, f64),
        bits: HilbertBitDepth,
    ) {
        let quantized: Vec<[u32; D]> = coords
            .iter()
            .map(|c| try_hilbert_quantize(c, bounds, bits).unwrap())
            .collect();
        let indices_bulk =
            hilbert_indices_prequantized(&quantized, bits).expect("valid quantized points");
        let indices_individual: Vec<u128> = coords
            .iter()
            .map(|c| try_hilbert_index(c, bounds, bits).unwrap())
            .collect();

        assert_eq!(indices_bulk, indices_individual);
    }

    #[test]
    fn test_hilbert_bit_depth_boundaries_and_traits() {
        let min_bits = HilbertBitDepth::try_new(1).expect("minimum bit depth should be valid");
        let max_bits =
            HilbertBitDepth::try_new(MAX_HILBERT_BITS).expect("maximum bit depth should be valid");

        assert_eq!(min_bits.get(), 1);
        assert_eq!(max_bits.get(), MAX_HILBERT_BITS);
        assert_eq!(max_bits.to_string(), MAX_HILBERT_BITS.to_string());
        assert_eq!(HilbertBitDepth::try_from(8), HilbertBitDepth::try_new(8));
        assert_matches!(
            HilbertBitDepth::try_new(0),
            Err(HilbertError::InvalidBitsParameter { bits: 0 })
        );
        assert_matches!(
            HilbertBitDepth::try_new(32),
            Err(HilbertError::InvalidBitsParameter { bits: 32 })
        );
        assert_matches!(
            HilbertBitDepth::try_from(0),
            Err(HilbertError::InvalidBitsParameter { bits: 0 })
        );
        assert_matches!(
            HilbertBitDepth::try_from(MAX_HILBERT_BITS + 1),
            Err(HilbertError::InvalidBitsParameter { bits }) if bits == MAX_HILBERT_BITS + 1
        );
    }

    #[test]
    fn test_hilbert_index_2d() {
        let bits = bit_depth(4);
        let origin = try_hilbert_index(&[0.0_f64, 0.0], (0.0, 1.0), bits).unwrap();
        let corner = try_hilbert_index(&[1.0_f64, 1.0], (0.0, 1.0), bits).unwrap();
        let center = try_hilbert_index(&[0.5_f64, 0.5], (0.0, 1.0), bits).unwrap();

        assert_eq!(origin, 0);
        assert_ne!(origin, center);
        assert_ne!(center, corner);
    }

    #[test]
    fn test_hilbert_index_3d() {
        let bits = bit_depth(8);
        let origin = try_hilbert_index(&[0.0_f64, 0.0, 0.0], (-1.0, 1.0), bits).unwrap();
        let corner = try_hilbert_index(&[1.0_f64, 1.0, 1.0], (-1.0, 1.0), bits).unwrap();
        assert_ne!(origin, corner);
    }

    macro_rules! gen_prequantized_matches_hilbert_index_tests {
        ($dim:literal, $coords:expr, $bounds:expr, $bits:expr) => {
            pastey::paste! {
                #[test]
                fn [<test_hilbert_indices_prequantized_matches_hilbert_index_ $dim d>]() {
                    let coords: [[f64; $dim]; 4] = $coords;
                    assert_prequantized_matches_hilbert_index(
                        &coords,
                        $bounds,
                        bit_depth($bits),
                    );
                }
            }
        };
    }

    gen_prequantized_matches_hilbert_index_tests!(
        2,
        [[-2.0, -1.0], [-1.5, 0.25], [0.1, -0.7], [3.0, 3.0]],
        (-2.0_f64, 3.0_f64),
        8
    );
    gen_prequantized_matches_hilbert_index_tests!(
        3,
        [
            [-2.0, -1.0, 0.0],
            [-1.5, 0.25, 1.75],
            [0.1, -0.7, 2.2],
            [3.0, 3.0, -2.0],
        ],
        (-2.0_f64, 3.0_f64),
        8
    );
    gen_prequantized_matches_hilbert_index_tests!(
        4,
        [
            [-2.0, -1.0, 0.0, 1.0],
            [-1.5, 0.25, 1.75, 2.5],
            [0.1, -0.7, 2.2, -1.8],
            [3.0, 3.0, -2.0, -2.0],
        ],
        (-2.0_f64, 3.0_f64),
        8
    );
    gen_prequantized_matches_hilbert_index_tests!(
        5,
        [
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            [-1.5, 0.25, 1.75, 2.5, -0.5],
            [0.1, -0.7, 2.2, -1.8, 1.4],
            [3.0, 3.0, -2.0, -2.0, 0.5],
        ],
        (-2.0_f64, 3.0_f64),
        8
    );

    #[test]
    fn test_hilbert_sorted_indices_and_sort_helpers() {
        let coords: Vec<[f64; 2]> =
            vec![[0.9, 0.9], [0.1, 0.1], [0.5, 0.5], [0.1, 0.9], [0.9, 0.1]];
        let bits = bit_depth(16);
        let order = try_hilbert_sorted_indices(&coords, (0.0, 1.0), bits).unwrap();
        assert_eq!(order.len(), coords.len());

        // Apply the ordering to a parallel payload.
        let mut payload: Vec<usize> = (0..coords.len()).collect();
        try_hilbert_sort_by_stable(&mut payload, (0.0_f64, 1.0), bits, |&i| coords[i]).unwrap();

        // Sorting by stable helper should be deterministic.
        let mut payload2: Vec<usize> = (0..coords.len()).collect();
        try_hilbert_sort_by_stable(&mut payload2, (0.0_f64, 1.0), bits, |&i| coords[i]).unwrap();
        assert_eq!(payload, order);
        assert_eq!(payload, payload2);
    }

    #[test]
    fn test_sort_helpers_accept_stateful_coordinate_closures() {
        let coords: Vec<[f64; 2]> = vec![[0.9, 0.9], [0.1, 0.1], [0.5, 0.5]];
        let bits = bit_depth(8);
        let expected_order = try_hilbert_sorted_indices(&coords, (0.0, 1.0), bits).unwrap();

        let mut stable_calls = 0_usize;
        let mut stable_payload: Vec<usize> = (0..coords.len()).collect();
        try_hilbert_sort_by_stable(&mut stable_payload, (0.0_f64, 1.0), bits, |&i| {
            stable_calls += 1;
            coords[i]
        })
        .unwrap();
        assert_eq!(stable_payload, expected_order);
        assert_eq!(stable_calls, coords.len());

        let mut unstable_calls = 0_usize;
        let mut unstable_payload: Vec<usize> = (0..coords.len()).collect();
        try_hilbert_sort_by_unstable(&mut unstable_payload, (0.0_f64, 1.0), bits, |&i| {
            unstable_calls += 1;
            coords[i]
        })
        .unwrap();
        assert_eq!(unstable_payload, expected_order);
        assert_eq!(unstable_calls, coords.len());
    }

    #[test]
    fn test_unstable_sort_orders_by_key() {
        let coords: Vec<[f64; 2]> =
            vec![[0.9, 0.9], [0.1, 0.1], [0.5, 0.5], [0.1, 0.9], [0.9, 0.1]];
        let bits = bit_depth(16);
        let expected_order = try_hilbert_sorted_indices(&coords, (0.0, 1.0), bits).unwrap();

        let mut payload: Vec<usize> = (0..coords.len()).collect();
        try_hilbert_sort_by_unstable(&mut payload, (0.0_f64, 1.0), bits, |&i| coords[i]).unwrap();

        assert_eq!(payload, expected_order);
    }

    #[test]
    fn test_zero_dim_sort_helpers_noop() {
        let coords: Vec<[f64; 0]> = vec![[], [], []];
        let bits = bit_depth(8);
        let order = try_hilbert_sorted_indices(&coords, (0.0, 1.0), bits).unwrap();
        assert_eq!(order, vec![0, 1, 2]);

        let mut stable_payload = vec![3, 2, 1];
        try_hilbert_sort_by_stable(&mut stable_payload, (0.0_f64, 1.0), bits, |_| []).unwrap();
        assert_eq!(stable_payload, vec![3, 2, 1]);

        let mut unstable_payload = vec![3, 2, 1];
        try_hilbert_sort_by_unstable(&mut unstable_payload, (0.0_f64, 1.0), bits, |_| []).unwrap();
        assert_eq!(unstable_payload, vec![3, 2, 1]);
    }

    #[test]
    fn test_zero_dim_raw_bound_apis_reject_invalid_bounds() {
        let bits = bit_depth(8);
        let coords = [0.0_f64; 0];
        let coordinate_batch = vec![coords; 3];
        let mut payload = vec![3, 2, 1];

        assert_matches!(
            try_hilbert_quantize(&coords, (f64::NAN, 1.0), bits),
            Err(HilbertError::NonFiniteBounds {
                lower_bound_finite: false,
                upper_bound_finite: true
            })
        );
        assert_matches!(
            try_hilbert_index(&coords, (1.0, 1.0), bits),
            Err(HilbertError::NonIncreasingBounds {
                ordering: CoordinateRangeOrdering::Equal
            })
        );
        assert_matches!(
            try_hilbert_sorted_indices(&coordinate_batch, (1.0, 0.0), bits),
            Err(HilbertError::NonIncreasingBounds {
                ordering: CoordinateRangeOrdering::Decreasing
            })
        );
        assert_matches!(
            try_hilbert_sort_by_stable(&mut payload, (f64::NEG_INFINITY, 1.0), bits, |_| coords),
            Err(HilbertError::NonFiniteBounds {
                lower_bound_finite: false,
                upper_bound_finite: true
            })
        );
    }

    #[test]
    fn test_scaled_quantize_reports_conversion_error() {
        let result = quantize_with_scale(
            &[1.0_f64],
            CoordinateRange::try_new(0.0, 1.0).unwrap(),
            bit_depth(31),
            u32::MAX,
            f64::INFINITY,
        );

        assert_matches!(
            result,
            Err(HilbertError::QuantizedCoordinateConversionFailed {
                bits: 31,
                max_grid_value: u32::MAX,
                coordinate_index: 0
            })
        );
    }

    macro_rules! gen_in_range_quantization_tests {
        ($dim:literal, $points:expr, $sample:expr) => {
            pastey::paste! {
                #[test]
                fn [<test_quantize_in_range_matches_tuple_boundary_ $dim d>]() {
                    let bits = bit_depth(8);
                    let range = CoordinateRange::try_new(-2.0_f64, 3.0).unwrap();
                    let coords: [f64; $dim] = $sample;

                    let parsed = try_hilbert_quantize(&coords, range.bounds(), bits).unwrap();
                    let prevalidated = hilbert_quantize_in_range(&coords, range, bits).unwrap();

                    assert_eq!(prevalidated, parsed);
                }

                #[test]
                fn [<test_quantized_batch_carries_prequantized_validation_ $dim d>]() {
                    let bits = bit_depth(8);
                    let range = CoordinateRange::try_new(-2.0_f64, 3.0).unwrap();
                    let points: [[f64; $dim]; 4] = $points;
                    let quantized: Vec<[u32; $dim]> = points
                        .iter()
                        .map(|point| hilbert_quantize_in_range(point, range, bits).unwrap())
                        .collect();
                    let batch = HilbertQuantizedBatch::try_new(&quantized, bits).unwrap();

                    assert_eq!(batch.coordinates(), quantized.as_slice());
                    assert_eq!(batch.bits(), bits);

                    let checked = hilbert_indices_prequantized(&quantized, bits).unwrap();
                    assert_eq!(batch.indices(), checked);
                    assert_eq!(hilbert_indices_for_quantized_batch(batch), checked);
                }

                #[test]
                fn [<test_quantize_batch_in_range_matches_two_step_path_ $dim d>]() {
                    let bits = bit_depth(8);
                    let bounds = CoordinateRange::try_new(-2.0_f64, 3.0).unwrap();
                    let points: [[f64; $dim]; 4] = $points;

                    let two_step: Vec<[u32; $dim]> = points
                        .iter()
                        .map(|point| hilbert_quantize_in_range(point, bounds, bits).unwrap())
                        .collect();
                    let two_step_indices = hilbert_indices_prequantized(&two_step, bits).unwrap();

                    let batch = hilbert_quantize_batch_in_range(&points, bounds, bits, |point| *point)
                        .unwrap();
                    assert_eq!(batch.coordinates(), two_step.as_slice());
                    assert_eq!(batch.bits(), bits);
                    assert_eq!(batch.len(), points.len());
                    assert!(!batch.is_empty());

                    let (indices, quantized) = batch.into_indices_and_coordinates();
                    assert_eq!(quantized, two_step);
                    assert_eq!(indices, two_step_indices);
                }
            }
        };
    }

    gen_in_range_quantization_tests!(
        2,
        [[-2.0_f64, -1.0], [-1.5, 0.25], [0.1, -0.7], [3.0, 3.0]],
        [0.25_f64, 0.75]
    );
    gen_in_range_quantization_tests!(
        3,
        [
            [-2.0_f64, -1.0, 0.0],
            [-1.5, 0.25, 1.75],
            [0.1, -0.7, 2.2],
            [3.0, 3.0, -2.0]
        ],
        [0.25_f64, 0.75, -1.0]
    );
    gen_in_range_quantization_tests!(
        4,
        [
            [-2.0_f64, -1.0, 0.0, 1.0],
            [-1.5, 0.25, 1.75, 2.5],
            [0.1, -0.7, 2.2, -1.8],
            [3.0, 3.0, -2.0, -2.0],
        ],
        [0.25_f64, 0.75, -1.0, 2.5]
    );
    gen_in_range_quantization_tests!(
        5,
        [
            [-2.0_f64, -1.0, 0.0, 1.0, 2.0],
            [-1.5, 0.25, 1.75, 2.5, -0.5],
            [0.1, -0.7, 2.2, -1.8, 1.4],
            [3.0, 3.0, -2.0, -2.0, 0.5],
        ],
        [0.25_f64, 0.75, -1.0, 2.5, 0.0]
    );

    #[test]
    fn test_quantized_batch_rejects_out_of_range_coordinate() {
        let bits = bit_depth(2);
        let quantized = [[0_u32, 0], [4, 1]];
        let result = HilbertQuantizedBatch::try_new(&quantized, bits);

        assert_matches!(
            result,
            Err(HilbertError::PrequantizedCoordinateOutOfRange {
                bits: 2,
                max_grid_value: 3,
                point_index: 1,
                coordinate_index: 0,
                coordinate: 4
            })
        );
    }

    #[test]
    fn test_quantized_batch_rejects_index_overflow() {
        let quantized = [[1_u32, 2, 3, 4, 5]];
        let result = HilbertQuantizedBatch::try_new(&quantized, bit_depth(26));

        assert_matches!(
            result,
            Err(HilbertError::IndexOverflow {
                dimension: 5,
                bits: 26,
                total_bits: 130
            })
        );
    }

    #[test]
    fn test_quantized_batch_handles_zero_dimension() {
        let bits = bit_depth(8);
        let quantized = [[], [], []];
        let batch = HilbertQuantizedBatch::try_new(&quantized, bits).unwrap();

        assert_eq!(batch.coordinates(), quantized.as_slice());
        assert_eq!(batch.bits(), bits);
        assert_eq!(batch.indices(), vec![0_u128, 0_u128, 0_u128]);
        assert_eq!(
            hilbert_indices_for_quantized_batch(batch),
            vec![0_u128, 0_u128, 0_u128]
        );
    }

    #[test]
    fn test_quantize_batch_in_range_handles_zero_dimension() {
        let bits = bit_depth(8);
        let bounds = CoordinateRange::try_new(0.0_f64, 1.0).unwrap();
        let items = [(), (), ()];

        let batch =
            hilbert_quantize_batch_in_range(&items, bounds, bits, |()| [0.0_f64; 0]).unwrap();
        assert_eq!(batch.len(), 3);
        assert!(!batch.is_empty());
        assert_eq!(batch.indices(), vec![0_u128, 0_u128, 0_u128]);
        assert_eq!(batch.into_coordinates(), vec![[0_u32; 0]; 3]);
    }

    #[test]
    fn test_quantize_in_range_handles_zero_dimension() {
        let bits = bit_depth(8);
        let bounds = CoordinateRange::try_new(0.0_f64, 1.0).unwrap();
        let coords = [0.0_f64; 0];

        assert_eq!(
            hilbert_quantize_in_range(&coords, bounds, bits).unwrap(),
            [0_u32; 0]
        );
    }

    #[test]
    fn test_quantize_batch_in_range_rejects_index_overflow() {
        let bits = bit_depth(26);
        let bounds = CoordinateRange::try_new(0.0_f64, 1.0).unwrap();
        let items = [[0.0_f64; 5]];

        assert_matches!(
            hilbert_quantize_batch_in_range(&items, bounds, bits, |p| *p),
            Err(HilbertError::IndexOverflow {
                dimension: 5,
                bits: 26,
                total_bits: 130
            })
        );
    }

    #[test]
    fn test_quantize_rejects_nonfinite_bounds() {
        let result = try_hilbert_quantize(&[0.5_f64], (f64::NAN, 1.0), bit_depth(8));

        assert_eq!(
            result,
            Err(HilbertError::NonFiniteBounds {
                lower_bound_finite: false,
                upper_bound_finite: true
            })
        );

        let both_non_finite =
            try_hilbert_quantize(&[0.5_f64], (f64::NAN, f64::INFINITY), bit_depth(8));
        assert_eq!(
            both_non_finite,
            Err(HilbertError::NonFiniteBounds {
                lower_bound_finite: false,
                upper_bound_finite: false
            })
        );
    }

    #[test]
    fn test_quantize_rejects_nonfinite_extent() {
        let result = try_hilbert_quantize(&[0.0_f64], (-f64::MAX, f64::MAX), bit_depth(8));

        assert_eq!(result, Err(HilbertError::NonFiniteBoundsExtent {}));
    }

    #[test]
    fn test_quantize_rejects_nonfinite_coordinate() {
        let result = try_hilbert_quantize(&[0.25_f64, f64::INFINITY], (0.0, 1.0), bit_depth(8));

        assert_eq!(
            result,
            Err(HilbertError::NonFiniteCoordinate {
                coordinate_index: 1
            })
        );
    }

    #[test]
    fn test_quantize_rejects_nonfinite_normalized() {
        let result =
            try_hilbert_quantize(&[f64::MAX], (-f64::MAX / 2.0, f64::MAX / 2.0), bit_depth(8));

        assert_eq!(
            result,
            Err(HilbertError::NonFiniteNormalizedCoordinate {
                coordinate_index: 0
            })
        );
    }

    #[test]
    fn test_sort_error_keeps_order() {
        let coords = [[0.5_f64], [f64::NAN], [0.25]];
        let mut payload = vec![0_usize, 1, 2];

        let result =
            try_hilbert_sort_by_stable(&mut payload, (0.0, 1.0), bit_depth(8), |&i| coords[i]);

        assert_eq!(
            result,
            Err(HilbertError::NonFiniteCoordinate {
                coordinate_index: 0
            })
        );
        assert_eq!(payload, vec![0, 1, 2]);
    }

    #[test]
    fn test_apply_order_rejects_length_mismatch_without_reordering() {
        let mut payload = vec![10, 20, 30];
        let result = apply_order(&mut payload, [0_usize, 1].into_iter());

        assert_eq!(payload, vec![10, 20, 30]);
        assert_matches!(
            result,
            Err(HilbertError::InvalidSortPermutationLength {
                item_count: 3,
                permutation_count: 2
            })
        );
    }

    #[test]
    fn test_apply_order_rejects_out_of_range_index_without_reordering() {
        let mut payload = vec![10, 20, 30];
        let result = apply_order(&mut payload, [0_usize, 3, 1].into_iter());

        assert_eq!(payload, vec![10, 20, 30]);
        assert_matches!(
            result,
            Err(HilbertError::InvalidSortPermutationIndex {
                permutation_index: 1,
                item_index: 3,
                item_count: 3
            })
        );
    }

    #[test]
    fn test_apply_order_rejects_duplicate_index_without_reordering() {
        let mut payload = vec![10, 20, 30];
        let result = apply_order(&mut payload, [0_usize, 1, 1].into_iter());

        assert_eq!(payload, vec![10, 20, 30]);
        assert_matches!(
            result,
            Err(HilbertError::InvalidSortPermutationDuplicate {
                permutation_index: 2,
                item_index: 1
            })
        );
    }

    #[test]
    fn test_apply_order_rejects_short_iterator_length_lie_without_reordering() {
        let mut payload = vec![10, 20, 30];
        let result = apply_order(&mut payload, LyingOrder::new(vec![0, 1], 3));

        assert_eq!(payload, vec![10, 20, 30]);
        assert_matches!(
            result,
            Err(HilbertError::InvalidSortPermutationLength {
                item_count: 3,
                permutation_count: 2
            })
        );
    }

    #[test]
    fn test_apply_order_rejects_long_iterator_length_lie_without_reordering() {
        let mut payload = vec![10, 20, 30];
        let result = apply_order(&mut payload, LyingOrder::new(vec![0, 1, 2, 0], 3));

        assert_eq!(payload, vec![10, 20, 30]);
        assert_matches!(
            result,
            Err(HilbertError::InvalidSortPermutationLength {
                item_count: 3,
                permutation_count: 4
            })
        );
    }

    #[test]
    fn test_quantize_clamps_f64_endpoint() {
        let q = try_hilbert_quantize(&[1.0], (0.0, 1.0), bit_depth(31)).unwrap();

        assert_eq!(q, [(1_u32 << 31) - 1]);
    }

    #[test]
    fn test_hilbert_curve_is_continuous_on_2d_grid() {
        // A defining property of the (discrete) Hilbert curve is continuity:
        // successive indices correspond to adjacent grid cells.
        let bits: u32 = 4;
        let n: u32 = 1_u32 << bits;

        let mut points: Vec<([u32; 2], u128)> = Vec::with_capacity((n * n) as usize);
        let params = positive_index_params::<2>(bit_depth(bits));
        for x in 0..n {
            for y in 0..n {
                let q = [x, y];
                let idx = index_from_quantized(&q, params);
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
        let params = positive_index_params::<4>(bit_depth(bits));
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    for w in 0..n {
                        let q = [x, y, z, w];
                        let idx = index_from_quantized(&q, params);
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
        let p: Point<2> = Point::from_validated_coords([0.25, 0.75]);
        let idx = try_hilbert_index(p.coords(), (0.0, 1.0), bit_depth(16)).unwrap();
        assert!(idx > 0);
    }

    #[test]
    fn test_hilbert_bits_boundaries() {
        let coarsest_bits = bit_depth(1);
        let origin = try_hilbert_index(&[0.0_f64, 0.0], (0.0, 1.0), coarsest_bits).unwrap();
        let corner = try_hilbert_index(&[1.0_f64, 1.0], (0.0, 1.0), coarsest_bits).unwrap();
        tracing::debug!(origin, corner, "bits=1 boundaries");
        assert_eq!(origin, 0, "bits=1 origin should map to 0");
        assert_ne!(origin, corner, "bits=1 should distinguish corners");

        let finest_bits = bit_depth(31);
        let origin_31 = try_hilbert_index(&[0.0_f64, 0.0], (0.0, 1.0), finest_bits).unwrap();
        let corner_31 = try_hilbert_index(&[1.0_f64, 1.0], (0.0, 1.0), finest_bits).unwrap();
        tracing::debug!(origin_31, corner_31, "bits=31 boundaries");
        assert_eq!(origin_31, 0, "bits=31 origin should map to 0");
        assert_ne!(origin_31, corner_31, "bits=31 should distinguish corners");
    }

    #[test]
    fn test_hilbert_index_1d_monotonic() {
        let bounds = (0.0_f64, 1.0_f64);
        let bits = bit_depth(8);
        let a = try_hilbert_index(&[0.0_f64], bounds, bits).unwrap();
        let b = try_hilbert_index(&[0.25_f64], bounds, bits).unwrap();
        let c = try_hilbert_index(&[0.5_f64], bounds, bits).unwrap();
        let d = try_hilbert_index(&[1.0_f64], bounds, bits).unwrap();
        tracing::debug!(a, b, c, d, "1d indices");
        assert!(
            a < b && b < c && c < d,
            "1D Hilbert indices should be monotonic"
        );
    }

    #[test]
    fn test_hilbert_rejects_degenerate_bounds() {
        let bounds = (1.0_f64, 1.0_f64);
        let coords = [2.0_f64, -2.0_f64];
        let bits = bit_depth(8);

        assert_matches!(
            try_hilbert_quantize(&coords, bounds, bits),
            Err(HilbertError::NonIncreasingBounds {
                ordering: CoordinateRangeOrdering::Equal
            })
        );
        assert_matches!(
            try_hilbert_index(&coords, bounds, bits),
            Err(HilbertError::NonIncreasingBounds {
                ordering: CoordinateRangeOrdering::Equal
            })
        );
    }

    #[test]
    fn test_hilbert_rejects_decreasing_bounds() {
        let bounds = (1.0_f64, 0.0_f64);
        let coords = [0.5_f64, 0.25_f64];
        let bits = bit_depth(8);

        assert_matches!(
            try_hilbert_quantize(&coords, bounds, bits),
            Err(HilbertError::NonIncreasingBounds {
                ordering: CoordinateRangeOrdering::Decreasing
            })
        );
    }

    #[test]
    fn test_hilbert_quantize_clamps_out_of_range() {
        let bounds = (0.0_f64, 1.0_f64);
        let bits = bit_depth(4);
        let coords = [-1.0_f64, 2.0_f64];
        let q = try_hilbert_quantize(&coords, bounds, bits).unwrap();
        let max_val = (1_u32 << bits.get()) - 1;
        tracing::debug!(?q, max_val, "clamp quantize");
        assert_eq!(
            q,
            [0, max_val],
            "out-of-range coords should clamp to bounds"
        );

        let idx = try_hilbert_index(&coords, bounds, bits).unwrap();
        let idx_clamped = try_hilbert_index(&[0.0_f64, 1.0_f64], bounds, bits).unwrap();
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
        let bits = bit_depth(8);

        // Quantize all coordinates
        let quantized: Vec<[u32; 3]> = coords
            .iter()
            .map(|c| try_hilbert_quantize(c, bounds, bits).unwrap())
            .collect();

        // Compute indices via bulk API
        let indices_bulk = hilbert_indices_prequantized(&quantized, bits)
            .expect("valid parameters should succeed");

        // Compute indices individually
        let indices_individual: Vec<u128> = coords
            .iter()
            .map(|c| try_hilbert_index(c, bounds, bits).unwrap())
            .collect();

        assert_eq!(indices_bulk.len(), coords.len());
        assert_eq!(indices_bulk, indices_individual);
    }

    #[test]
    fn test_hilbert_indices_prequantized_empty_input() {
        let empty: Vec<[u32; 2]> = vec![];
        let bits = bit_depth(4);

        let indices =
            hilbert_indices_prequantized(&empty, bits).expect("valid parameters should succeed");
        assert_eq!(indices.len(), 0);
    }

    #[test]
    fn test_hilbert_indices_prequantized_validates_overflow() {
        // With D=5 and bits=26, total_bits = 130 > 128
        let quantized = vec![[1_u32, 2, 3, 4, 5]];
        let result = hilbert_indices_prequantized(&quantized, bit_depth(26));
        assert_matches!(
            result,
            Err(HilbertError::IndexOverflow {
                dimension: 5,
                bits: 26,
                total_bits: 130
            })
        );
    }

    #[test]
    fn test_hilbert_indices_prequantized_validates_coordinate_range() {
        let bits = bit_depth(2);
        let quantized = vec![[0_u32, 3], [4, 1]];
        let result = hilbert_indices_prequantized(&quantized, bits);

        assert_matches!(
            result,
            Err(HilbertError::PrequantizedCoordinateOutOfRange {
                bits: 2,
                max_grid_value: 3,
                point_index: 1,
                coordinate_index: 0,
                coordinate: 4
            })
        );
    }

    #[test]
    fn test_hilbert_indices_prequantized_handles_zero_dimension() {
        // Zero-dimensional space has only one point, all map to index 0
        let quantized: Vec<[u32; 0]> = vec![[], [], []];
        let bits = bit_depth(8);

        let indices = hilbert_indices_prequantized(&quantized, bits).expect("D=0 should succeed");

        assert_eq!(indices.len(), 3);
        assert_eq!(indices, vec![0_u128, 0_u128, 0_u128]);
    }

    #[test]
    fn test_hilbert_quantize_uses_rounding_not_truncation() {
        let bounds = (0.0_f64, 1.0_f64);
        let bits = bit_depth(2); // Grid has 4 cells: 0, 1, 2, 3

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
            let q = try_hilbert_quantize(&[coord], bounds, bits).unwrap();
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
            let q = try_hilbert_quantize(&[coord], bounds, bits).unwrap();
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
