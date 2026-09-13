//! Memory/allocation measurement utilities.

#![forbid(unsafe_code)]

/// Memory measurement helper for allocation tracking in examples, tests, and benchmarks.
///
/// Available only with the `count-allocations` feature, which installs the allocation-counting
/// global allocator. Measurements cover the calling thread, not work spawned on other threads.
/// Enabling this feature adds the API without changing an existing function's return type.
///
/// # Arguments
///
/// * `f` - A closure to execute while measuring allocations
///
/// # Returns
///
/// Returns `(R, allocation_counter::AllocationInfo)`, containing the closure result and
/// allocation metrics. There is no no-op fallback that could be mistaken for a measurement.
///
/// # Panics
///
/// Panics if `f` panics. The internal `expect()` call is used because a
/// normally returning closure is guaranteed to set the result. The profiler also panics
/// when its supported nesting limit of 64 measurements is exceeded.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::tds::measure_with_result;
///
/// let (result, alloc_info) = measure_with_result(|| {
///     vec![1, 2, 3, 4, 5]
/// });
/// assert_eq!(result, vec![1, 2, 3, 4, 5]);
///
/// assert!(alloc_info.bytes_total > 0);
/// ```
#[cfg_attr(docsrs, doc(cfg(feature = "count-allocations")))]
pub fn measure_with_result<F, R>(f: F) -> (R, allocation_counter::AllocationInfo)
where
    F: FnOnce() -> R,
{
    let mut result: Option<R> = None;
    let info = allocation_counter::measure(|| {
        result = Some(f());
    });
    (result.expect("Closure should have set result"), info)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn non_negative(value: i64) -> u64 {
        u64::try_from(value.max(0)).unwrap_or(0)
    }

    #[test]
    fn test_measure_with_result_comprehensive() {
        // Test basic functionality - returns correct result
        let expected_result = 42;
        let (result, _alloc_info) = measure_with_result(|| expected_result);
        assert_eq!(result, expected_result);

        // Test with various allocation patterns
        let (vec_result, _alloc_info) = measure_with_result(|| vec![1, 2, 3, 4, 5]);
        assert_eq!(vec_result, vec![1, 2, 3, 4, 5]);

        let (string_result, _alloc_info) = measure_with_result(|| {
            let mut s = String::new();
            s.push_str("Hello, ");
            s.push_str("World!");
            s
        });
        assert_eq!(string_result, "Hello, World!");

        let (complex_result, _alloc_info) = measure_with_result(|| {
            let mut data: Vec<String> = Vec::new();
            for i in 0..5 {
                data.push(format!("Item {i}"));
            }
            data.len()
        });
        assert_eq!(complex_result, 5);

        // Test various return types
        let (tuple_result, _alloc_info) = measure_with_result(|| ("hello", 42));
        assert_eq!(tuple_result, ("hello", 42));

        let (option_result, _alloc_info) = measure_with_result(|| Some("value"));
        assert_eq!(option_result, Some("value"));

        let (result_result, _alloc_info) = measure_with_result(|| Ok::<i32, &str>(123));
        assert_eq!(result_result, Ok(123));

        // Test no-panic behavior
        let (sum_result, _alloc_info) = measure_with_result(|| {
            let data = [1, 2, 3];
            data.iter().sum::<i32>()
        });
        assert_eq!(sum_result, 6);
    }

    #[test]
    fn test_measure_with_result_executes_once() {
        let calls = Cell::new(0);
        let (result, _alloc_info) = measure_with_result(|| {
            calls.set(calls.get() + 1);
            123
        });

        assert_eq!(result, 123);
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn test_measure_with_result_panics_propagate() {
        let should_panic = std::hint::black_box(true);
        let result = std::panic::catch_unwind(|| {
            let _ = measure_with_result(|| -> i32 {
                assert!(!should_panic, "boom");
                0
            });
        });

        assert!(result.is_err());
    }

    #[test]
    fn test_measure_with_result_allocation_info_structure() {
        // Test that allocation info has expected structure when feature is enabled
        let (_result, alloc_info) = measure_with_result(|| {
            // Allocate some memory
            vec![0u8; 1024]
        });

        // Verify that we got an AllocationInfo struct by accessing its fields
        // This validates that the function properly returns allocation info
        // We access all fields to ensure the struct is properly constructed
        std::hint::black_box(&alloc_info.bytes_total);
        std::hint::black_box(&alloc_info.count_total);
        std::hint::black_box(&alloc_info.bytes_current);
        std::hint::black_box(&alloc_info.count_current);
        std::hint::black_box(&alloc_info.bytes_max);
        std::hint::black_box(&alloc_info.count_max);

        // Test that we can actually use the allocation info
        // For a vec![0u8; 1024], we expect some allocation to have occurred
        assert!(
            alloc_info.bytes_total > 0,
            "Should have allocated memory for the vector"
        );

        let bytes_current = non_negative(alloc_info.bytes_current);
        let count_current = non_negative(alloc_info.count_current);

        assert!(
            alloc_info.bytes_total >= bytes_current,
            "Total bytes should be >= current bytes"
        );
        assert!(
            alloc_info.bytes_max >= bytes_current,
            "Max bytes should be >= current bytes"
        );
        assert!(
            alloc_info.count_total >= count_current,
            "Total allocations should be >= current allocations"
        );
        assert!(
            alloc_info.count_max >= count_current,
            "Max allocation count should be >= current allocation count"
        );
    }

    #[test]
    fn test_measure_with_result_no_allocation_invariants() {
        let (_result, alloc_info) = measure_with_result(|| 7usize);

        let bytes_current = non_negative(alloc_info.bytes_current);
        let count_current = non_negative(alloc_info.count_current);

        assert!(
            alloc_info.bytes_total >= bytes_current,
            "Total bytes should be >= current bytes"
        );
        assert!(
            alloc_info.bytes_max >= bytes_current,
            "Max bytes should be >= current bytes"
        );
        assert!(
            alloc_info.count_total >= count_current,
            "Total allocations should be >= current allocations"
        );
        assert!(
            alloc_info.count_max >= count_current,
            "Max allocation count should be >= current allocation count"
        );
    }
}
