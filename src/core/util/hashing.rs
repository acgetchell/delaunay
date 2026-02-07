//! Hashing utilities.

#![forbid(unsafe_code)]

/// Applies a stable hash function to a slice of sorted u64 values.
///
/// This function uses an FNV-based polynomial rolling hash with an avalanche step
/// to produce deterministic hash values. The input slice should be pre-sorted to ensure
/// consistent results regardless of input order.
///
/// # Arguments
///
/// * `sorted_values` - A slice of u64 values that should be pre-sorted
///
/// # Returns
///
/// A `u64` hash value representing the stable hash of the input values
///
/// # Algorithm
///
/// Uses FNV constants with polynomial rolling hash:
/// 1. Start with FNV offset basis
/// 2. For each value: `hash = hash.wrapping_mul(PRIME).wrapping_add(value)`
/// 3. Apply avalanche step for better bit distribution
///
/// # Examples
///
/// ```
/// use delaunay::core::util::stable_hash_u64_slice;
/// let values = vec![1u64, 2u64, 3u64];
/// let hash1 = stable_hash_u64_slice(&values);
///
/// let mut reversed = values.clone();
/// reversed.reverse();
/// let hash2 = stable_hash_u64_slice(&reversed);
///
/// // Different order produces different hash (input should be pre-sorted)
/// assert_ne!(hash1, hash2);
///
/// // Same sorted input produces same hash
/// let mut sorted1 = values;
/// sorted1.sort_unstable();
/// let mut sorted2 = reversed;
/// sorted2.sort_unstable();
/// assert_eq!(stable_hash_u64_slice(&sorted1), stable_hash_u64_slice(&sorted2));
/// ```
#[must_use]
pub fn stable_hash_u64_slice(sorted_values: &[u64]) -> u64 {
    // Hash constants for facet key generation (FNV-based)
    const HASH_PRIME: u64 = 1_099_511_628_211; // Large prime (FNV prime)
    const HASH_OFFSET: u64 = 14_695_981_039_346_656_037; // FNV offset basis

    // Handle empty case
    if sorted_values.is_empty() {
        return 0;
    }

    // Use a polynomial rolling hash for efficient combination
    let mut hash = HASH_OFFSET;
    for &value in sorted_values {
        hash = hash.wrapping_mul(HASH_PRIME).wrapping_add(value);
    }

    // Apply avalanche step for better bit distribution
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    hash ^= hash >> 33;

    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stable_hash_u64_slice_comprehensive() {
        // Test basic functionality with order sensitivity
        let values = vec![1u64, 2u64, 3u64];
        let hash1 = stable_hash_u64_slice(&values);

        let mut reversed = values.clone();
        reversed.reverse();
        let hash2 = stable_hash_u64_slice(&reversed);
        assert_ne!(
            hash1, hash2,
            "Different order should produce different hash"
        );

        // Same sorted input produces same hash
        let mut sorted1 = values;
        sorted1.sort_unstable();
        let mut sorted2 = reversed;
        sorted2.sort_unstable();
        assert_eq!(
            stable_hash_u64_slice(&sorted1),
            stable_hash_u64_slice(&sorted2),
            "Same sorted input should produce same hash"
        );

        // Test edge cases: empty, single value, different lengths
        let empty: Vec<u64> = vec![];
        assert_eq!(
            stable_hash_u64_slice(&empty),
            0,
            "Empty slice should produce hash 0"
        );

        let single = vec![42u64];
        let single_copy = vec![42u64];
        assert_eq!(
            stable_hash_u64_slice(&single),
            stable_hash_u64_slice(&single_copy),
            "Same single value should produce same hash"
        );

        let different_single = vec![43u64];
        assert_ne!(
            stable_hash_u64_slice(&single),
            stable_hash_u64_slice(&different_single),
            "Different single values should produce different hashes"
        );

        // Test deterministic behavior
        let test_values = vec![100u64, 200u64, 300u64, 400u64];
        let hash_a = stable_hash_u64_slice(&test_values);
        let hash_b = stable_hash_u64_slice(&test_values);
        let hash_c = stable_hash_u64_slice(&test_values);
        assert_eq!(hash_a, hash_b, "Multiple calls should be deterministic");
        assert_eq!(hash_b, hash_c, "Multiple calls should be deterministic");

        // Test different lengths
        let short = vec![1u64, 2u64];
        let long = vec![1u64, 2u64, 3u64];
        assert_ne!(
            stable_hash_u64_slice(&short),
            stable_hash_u64_slice(&long),
            "Different lengths should produce different hashes"
        );

        // Test large values
        let large_values = vec![u64::MAX, u64::MAX - 1, u64::MAX - 2];
        let hash_large1 = stable_hash_u64_slice(&large_values);
        let hash_large2 = stable_hash_u64_slice(&large_values);
        assert_eq!(
            hash_large1, hash_large2,
            "Large values should be handled consistently"
        );

        let different_large = vec![u64::MAX - 3, u64::MAX - 4, u64::MAX - 5];
        assert_ne!(
            hash_large1,
            stable_hash_u64_slice(&different_large),
            "Different large values should produce different hashes"
        );
    }
}
