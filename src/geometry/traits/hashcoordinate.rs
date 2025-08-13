//! Hash coordinate trait for floating-point types.
//!
//! This module provides the `HashCoordinate` trait which enables consistent hashing
//! of floating-point coordinate values, including proper handling of special values
//! like NaN and infinity.

use ordered_float::OrderedFloat;
use std::hash::{Hash, Hasher};

/// Helper trait for hashing individual coordinates for non-hashable types
/// like f32 and f64.
pub trait HashCoordinate {
    /// Hashes a single coordinate value using the provided hasher.
    ///
    /// This method provides a consistent way to hash coordinate values,
    /// including floating-point types that don't normally implement Hash.
    /// For floating-point types, this uses `OrderedFloat` to ensure consistent
    /// hashing behavior, including proper handling of NaN values.
    ///
    /// # Arguments
    ///
    /// * `state` - The hasher state to write the hash value to
    ///
    /// # Example
    ///
    /// ```
    /// use delaunay::geometry::HashCoordinate;
    /// use std::collections::hash_map::DefaultHasher;
    /// use std::hash::Hasher;
    ///
    /// let mut hasher = DefaultHasher::new();
    /// let value = 3.14f64;
    /// value.hash_coord(&mut hasher);
    /// let hash_value = hasher.finish();
    /// ```
    fn hash_coord<H: Hasher>(&self, state: &mut H);
}

// Unified macro for implementing HashCoordinate
macro_rules! impl_hash_coordinate {
    (float: $($t:ty),*) => {
        $(
            impl HashCoordinate for $t {
                #[inline(always)]
                fn hash_coord<H: Hasher>(&self, state: &mut H) {
                    OrderedFloat(*self).hash(state);
                }
            }
        )*
    };
}

impl_hash_coordinate!(float: f32, f64);

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    #[test]
    fn hash_coordinate_trait_coverage() {
        // Helper function to get hash for a coordinate
        fn hash_coord<T: HashCoordinate>(value: &T) -> u64 {
            let mut hasher = DefaultHasher::new();
            value.hash_coord(&mut hasher);
            hasher.finish()
        }

        // Test floating point types
        let hash_f32 = hash_coord(&std::f32::consts::PI);
        let hash_f64 = hash_coord(&std::f64::consts::PI);
        assert!(hash_f32 > 0);
        assert!(hash_f64 > 0);

        // Test that same values hash to same result
        assert_eq!(hash_coord(&1.0f32), hash_coord(&1.0f32));
        assert_eq!(hash_coord(&1.0f64), hash_coord(&1.0f64));

        // Test NaN hashing consistency
        assert_eq!(hash_coord(&f32::NAN), hash_coord(&f32::NAN));
        assert_eq!(hash_coord(&f64::NAN), hash_coord(&f64::NAN));

        // Test infinity hashing
        assert_eq!(hash_coord(&f32::INFINITY), hash_coord(&f32::INFINITY));
        assert_eq!(hash_coord(&f64::INFINITY), hash_coord(&f64::INFINITY));
        assert_eq!(
            hash_coord(&f32::NEG_INFINITY),
            hash_coord(&f32::NEG_INFINITY)
        );
        assert_eq!(
            hash_coord(&f64::NEG_INFINITY),
            hash_coord(&f64::NEG_INFINITY)
        );

        // Test that different special values hash differently
        assert_ne!(hash_coord(&f64::INFINITY), hash_coord(&f64::NEG_INFINITY));
    }
}
