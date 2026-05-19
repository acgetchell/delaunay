//! Data type traits for Delaunay triangulation structures.
//!
//! This module contains trait definitions for data types that can be
//! stored in vertices and simplices of the triangulation data structure.

use serde::{Serialize, de::DeserializeOwned};
use std::{fmt::Debug, hash::Hash};

/// Marker for data payloads that can be copied without ownership transfer.
///
/// This is the storage-level requirement used by much of the triangulation
/// code when vertex or simplex data has to be duplicated while preserving rollback
/// semantics.
pub trait DataCopy: Copy {}

impl<T> DataCopy for T where T: Copy {}

/// Marker for data payloads that have stable identity/order semantics.
///
/// Use this bound when a payload participates in sorting or hashing; `Ord`
/// supplies the equality contract, while pure storage and serialization paths
/// should not require either ordering or hashing.
pub trait DataIdentity: Hash + Ord {}

impl<T> DataIdentity for T where T: Hash + Ord {}

/// Marker for data payloads that can be formatted for diagnostics.
pub trait DataDebug: Debug {}

impl<T> DataDebug for T where T: Debug {}

/// Marker for data payloads that can be serialized.
pub trait DataSerialize: Serialize {}

impl<T> DataSerialize for T where T: Serialize {}

/// Marker for data payloads that can be deserialized without borrowed input.
pub trait DataDeserialize: DeserializeOwned {}

impl<T> DataDeserialize for T where T: DeserializeOwned {}

/// Marker for data payloads that can cross the serde persistence boundary.
pub trait DataSerde: DataSerialize + DataDeserialize {}

impl<T> DataSerde for T where T: DataSerialize + DataDeserialize {}

/// Trait alias for data types that can be stored in vertices and simplices.
///
/// This trait alias captures all the requirements for data types that can be associated
/// with vertices and simplices in the triangulation data structure. Data types must implement
/// `Copy` to enable efficient passing by value and to avoid ownership complications.
///
/// # Required Traits
///
/// - [`DataCopy`]: For efficient copying by value (includes `Clone`)
/// - [`DataIdentity`]: For hashing and total ordering (including equality)
/// - [`DataDebug`]: For diagnostic formatting
/// - [`DataSerde`]: For serialization and owned deserialization
///
/// Prefer the narrower marker traits on APIs that only need one part of this
/// contract. For example, serde implementations should use [`DataSerialize`] or
/// [`DataDeserialize`] when they do not inspect, hash, order, or copy payloads.
///
/// # Usage
///
/// ```rust
/// use delaunay::prelude::DataType;
///
/// fn process_data<T: DataType>(data: T) {
///     // T has all the necessary bounds for use as vertex/simplex data
/// }
///
/// // Examples of types that implement DataType:
/// // - i32, u32, char (primitive Copy types with total ordering)
/// // - Option<T> where T: DataType (optional Copy data)
/// // - () (unit type for no data)
/// // - Custom Copy enums with serde support
/// ```
///
/// # String Data Limitations
///
/// **String types have significant limitations:**
/// - `String` doesn't work (doesn't implement `Copy`)
/// - `&str` has complex lifetime issues that make it impractical
/// - `&'static str` works but only for compile-time constants
///
/// **Recommended alternatives for string-like data:**
/// - Numeric IDs with external `HashMap` lookup (most flexible)
/// - Character codes (`char` type for single characters)
/// - Custom Copy enums for predefined categories
/// - Fixed-size byte arrays for very short strings
///
/// See the `vertex_string_data_usage_examples` test for detailed examples.
pub trait DataType: DataCopy + DataIdentity + DataDebug + DataSerde {}

// Blanket implementation for all types that satisfy the bounds
impl<T> DataType for T where T: DataCopy + DataIdentity + DataDebug + DataSerde {}
