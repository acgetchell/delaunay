//! Recoverable transitions between proof-bearing domain owners.
//!
//! A refinement consumes a value that already proves one invariant layer and
//! attempts to establish the next layer. Failure returns both the unchanged
//! lower-layer owner and the typed rejection reason so callers can inspect the
//! diagnostic, change repair policy, and retry without cloning canonical state.

#![forbid(unsafe_code)]

/// A failed proof refinement together with the still-valid lower-layer owner.
///
/// `T` is the domain value accepted by the attempted transition and `E` is the
/// typed reason that the stronger proof could not be established. The owner is
/// intentionally private so callers cannot accidentally separate it from the
/// failure without choosing one of the consuming accessors. It is boxed inside
/// the error so the common successful `Result` path remains compact; refinement
/// failure pays the allocation while recovery still moves the original owner.
///
/// # Examples
///
/// ```rust
/// use delaunay::refinement::RefinementError;
///
/// let failure = RefinementError::new(3_u8, "not strong enough");
/// assert_eq!(*failure.owner(), 3);
/// assert_eq!(failure.reason(), &"not strong enough");
/// assert_eq!(failure.into_parts(), (3, "not strong enough"));
/// ```
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
#[error("proof refinement failed: {reason}")]
pub struct RefinementError<T, E> {
    owner: Box<T>,
    #[source]
    reason: E,
}

impl<T, E> RefinementError<T, E> {
    /// Creates a recoverable refinement failure at a checked transition boundary.
    pub fn new(owner: T, reason: E) -> Self {
        Self {
            owner: Box::new(owner),
            reason,
        }
    }

    /// Returns the still-valid lower-layer owner.
    #[must_use]
    pub fn owner(&self) -> &T {
        &self.owner
    }

    /// Returns the typed reason the stronger proof could not be established.
    #[must_use]
    pub const fn reason(&self) -> &E {
        &self.reason
    }

    /// Recovers the lower-layer owner and intentionally discards the diagnostic.
    #[must_use]
    pub fn into_owner(self) -> T {
        *self.owner
    }

    /// Recovers the diagnostic and intentionally drops the lower-layer owner.
    #[must_use]
    pub fn into_reason(self) -> E {
        self.reason
    }

    /// Separates the lower-layer owner from its typed rejection reason.
    #[must_use]
    pub fn into_parts(self) -> (T, E) {
        (*self.owner, self.reason)
    }
}
