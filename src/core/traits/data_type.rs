//! Data type traits for Delaunay triangulation structures.
//!
//! This module contains trait definitions for data types that can be
//! stored in vertices and cells of the triangulation data structure.

use serde::{Serialize, de::DeserializeOwned};
use std::{fmt::Debug, hash::Hash};

/// Trait alias for data types that can be stored in vertices and cells.
///
/// This trait alias captures all the requirements for data types that can be associated
/// with vertices and cells in the triangulation data structure. Data types must implement
/// `Copy` to enable efficient passing by value and to avoid ownership complications.
///
/// # Required Traits
///
/// - `Copy`: For efficient copying by value (includes `Clone`)
/// - `Eq`: For equality comparison
/// - `Hash`: For use in hash-based collections
/// - `Ord`: For ordering and sorting
/// - `PartialEq`: For partial equality comparison
/// - `PartialOrd`: For partial ordering
/// - `Debug`: For debug formatting
/// - `Serialize`: For serialization support
/// - `DeserializeOwned`: For deserialization support
///
/// # Usage
///
/// ```rust
/// use delaunay::core::DataType;
///
/// fn process_data<T: DataType>(data: T) {
///     // T has all the necessary bounds for use as vertex/cell data
/// }
///
/// // Examples of types that implement DataType:
/// // - i32, u32, f64, &str (Copy types)
/// // - Option<T> where T: DataType (optional Copy data)
/// // - () (unit type for no data)
/// ```
pub trait DataType:
    Copy + Eq + Hash + Ord + PartialEq + PartialOrd + Debug + Serialize + DeserializeOwned
{
}

// Blanket implementation for all types that satisfy the bounds
impl<T> DataType for T where
    T: Copy + Eq + Hash + Ord + PartialEq + PartialOrd + Debug + Serialize + DeserializeOwned
{
}
