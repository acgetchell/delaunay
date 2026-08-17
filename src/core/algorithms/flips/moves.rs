//! Typed vocabulary for bistellar/Pachner moves.

#![forbid(unsafe_code)]

use thiserror::Error;

/// Error returned when constructing an invalid bistellar flip kind.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum BistellarFlipKindError {
    /// The requested move size is outside the mathematical Pachner range.
    #[error("k must be in 1..=D+1 (k={k_move}, D={dimension})")]
    MoveSizeOutOfRange {
        /// Requested number of simplices replaced on the current side.
        k_move: usize,
        /// Triangulation dimension supplied by the caller.
        dimension: usize,
    },
    /// The inverse move size cannot be represented by `usize`.
    #[error("inverse move size D+2-k is not representable (k={k_move}, D={dimension})")]
    InverseMoveSizeOverflow {
        /// Requested number of simplices replaced on the current side.
        k_move: usize,
        /// Triangulation dimension supplied by the caller.
        dimension: usize,
    },
}

/// Bistellar flip kind descriptor.
///
/// Access the move size with [`BistellarFlipKind::k`].
/// Access the triangulation dimension with [`BistellarFlipKind::d`].
///
/// # Examples
///
/// ```rust
/// use delaunay::flips::BistellarFlipKind;
///
/// # fn main() -> Result<(), delaunay::flips::BistellarFlipKindError> {
/// let kind = BistellarFlipKind::try_k2(3)?;
/// let inverse = kind.inverse();
/// assert_eq!(kind.k(), 2);
/// assert_eq!(kind.d(), 3);
/// assert_eq!(inverse.k(), 3);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BistellarFlipKind {
    /// Number of simplices being replaced on the current side (k).
    k: usize,
    /// Dimension of the triangulation (D).
    d: usize,
}
/// Direction of a bistellar flip.
///
/// # Examples
///
/// ```rust
/// use delaunay::flips::FlipDirection;
///
/// assert_eq!(FlipDirection::Forward.inverse(), FlipDirection::Inverse);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlipDirection {
    /// Forward (k → D+2−k).
    Forward,
    /// Inverse (D+2−k → k).
    Inverse,
}

impl FlipDirection {
    /// Return the opposite direction.
    #[must_use]
    pub const fn inverse(self) -> Self {
        match self {
            Self::Forward => Self::Inverse,
            Self::Inverse => Self::Forward,
        }
    }
}

/// Stage where debug/test flip validation checked coherent orientation.
///
/// Coherent orientation is a validation-scale TDS invariant. Release-mode flip
/// hot paths rely on explicit validation boundaries rather than scanning the
/// whole TDS before and after every attempted flip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlipOrientationCheckStage {
    /// Before applying the flip inside the rollback transaction.
    BeforeMutation,
    /// After applying the flip inside the rollback transaction and before committing it.
    AfterTrialMutation,
}
impl BistellarFlipKind {
    /// Number of simplices being replaced on the current side (k).
    #[must_use]
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Dimension of the triangulation (D).
    #[must_use]
    pub const fn d(&self) -> usize {
        self.d
    }

    /// Construct a k=1 flip kind for the given dimension.
    ///
    /// # Errors
    ///
    /// Returns [`BistellarFlipKindError::InverseMoveSizeOverflow`] when the
    /// inverse move size cannot be represented by `usize`.
    pub const fn try_k1(d: usize) -> Result<Self, BistellarFlipKindError> {
        Self::try_from_raw(1, d)
    }

    /// Construct a k=2 flip kind for the given dimension.
    ///
    /// # Errors
    ///
    /// Returns [`BistellarFlipKindError::MoveSizeOutOfRange`] when a k=2 move
    /// is not defined for the supplied dimension.
    pub const fn try_k2(d: usize) -> Result<Self, BistellarFlipKindError> {
        Self::try_from_raw(2, d)
    }

    /// Construct a k=3 flip kind for the given dimension.
    ///
    /// # Errors
    ///
    /// Returns [`BistellarFlipKindError::MoveSizeOutOfRange`] when a k=3 move
    /// is not defined for the supplied dimension.
    pub const fn try_k3(d: usize) -> Result<Self, BistellarFlipKindError> {
        Self::try_from_raw(3, d)
    }

    /// Parses raw move metadata into a kind whose inverse is representable.
    const fn try_from_raw(k_move: usize, d: usize) -> Result<Self, BistellarFlipKindError> {
        if k_move == 0 || k_move > d.saturating_add(1) {
            return Err(BistellarFlipKindError::MoveSizeOutOfRange {
                k_move,
                dimension: d,
            });
        }
        if k_move == 1 && d == usize::MAX {
            return Err(BistellarFlipKindError::InverseMoveSizeOverflow {
                k_move,
                dimension: d,
            });
        }
        Ok(Self { k: k_move, d })
    }

    /// Constructs a kind from move metadata already proven valid and invertible.
    #[must_use]
    pub(super) const fn from_validated(k_move: usize, d: usize) -> Self {
        Self { k: k_move, d }
    }

    /// Construct the inverse flip kind (k' = D + 2 - k).
    #[must_use]
    pub const fn inverse(self) -> Self {
        let k = if self.k <= 2 {
            self.d + (2 - self.k)
        } else {
            self.d - (self.k - 2)
        };
        Self { k, d: self.d }
    }
}

/// Const-generic move marker for Pachner k-moves.
///
/// # Examples
///
/// ```rust
/// use delaunay::flips::{BistellarMove, ConstK};
///
/// fn move_k<const D: usize, M: BistellarMove<D>>() -> usize {
///     M::K
/// }
///
/// assert_eq!(move_k::<3, ConstK<2>>(), 2);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ConstK<const K: usize>;

/// Const-generic descriptor for a Pachner move in dimension `D`.
///
/// # Examples
///
/// ```rust
/// use delaunay::flips::{BistellarMove, ConstK};
///
/// fn move_k<const D: usize, M: BistellarMove<D>>() -> usize {
///     M::K
/// }
///
/// assert_eq!(move_k::<4, ConstK<3>>(), 3);
/// ```
pub trait BistellarMove<const D: usize> {
    /// Number of removed D-simplices (k).
    const K: usize;
}

impl<const D: usize, const K: usize> BistellarMove<D> for ConstK<K> {
    const K: usize = K;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_reject_invalid_move_metadata() {
        assert_eq!(
            BistellarFlipKind::try_k2(0),
            Err(BistellarFlipKindError::MoveSizeOutOfRange {
                k_move: 2,
                dimension: 0,
            })
        );
        assert_eq!(
            BistellarFlipKind::try_k3(1),
            Err(BistellarFlipKindError::MoveSizeOutOfRange {
                k_move: 3,
                dimension: 1,
            })
        );
        assert_eq!(
            BistellarFlipKind::try_k1(usize::MAX),
            Err(BistellarFlipKindError::InverseMoveSizeOverflow {
                k_move: 1,
                dimension: usize::MAX,
            })
        );
    }

    #[test]
    fn inverse_preserves_valid_formula_and_roundtrips() {
        for kind in [
            BistellarFlipKind::try_k1(3).unwrap(),
            BistellarFlipKind::try_k2(3).unwrap(),
            BistellarFlipKind::try_k3(3).unwrap(),
            BistellarFlipKind::try_k2(usize::MAX).unwrap(),
        ] {
            assert_eq!(kind.inverse().inverse(), kind);
        }
        assert_eq!(BistellarFlipKind::try_k2(3).unwrap().inverse().k(), 3);
    }
}
