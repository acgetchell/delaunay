//! Typed vocabulary for bistellar/Pachner moves.

#![forbid(unsafe_code)]

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
/// let kind = BistellarFlipKind::k2(3);
/// let inverse = kind.inverse();
/// assert_eq!(kind.k(), 2);
/// assert_eq!(kind.d(), 3);
/// assert_eq!(inverse.k(), 3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BistellarFlipKind {
    /// Number of simplices being replaced on the current side (k).
    pub(super) k: usize,
    /// Dimension of the triangulation (D).
    pub(super) d: usize,
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
    #[must_use]
    pub const fn k1(d: usize) -> Self {
        Self { k: 1, d }
    }
    /// Construct a k=2 flip kind for the given dimension.
    #[must_use]
    pub const fn k2(d: usize) -> Self {
        Self { k: 2, d }
    }

    /// Construct a k=3 flip kind for the given dimension.
    #[must_use]
    pub const fn k3(d: usize) -> Self {
        Self { k: 3, d }
    }

    /// Construct the inverse flip kind (k' = D + 2 - k).
    #[must_use]
    pub const fn inverse(self) -> Self {
        Self {
            k: self.d + 2 - self.k,
            d: self.d,
        }
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
