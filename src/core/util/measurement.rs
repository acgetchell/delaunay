//! Memory/allocation measurement utilities.

#![forbid(unsafe_code)]

use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

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
/// If `f` unwinds, the profiler finishes its measurement before the original
/// panic payload is resumed. Changes made by `f` are not rolled back.
/// The internal `expect()` is unreachable because the measured closure always
/// records its outcome. The upstream profiler may also panic when its nesting
/// limit is exceeded; recovery from a profiler-internal failure is not guaranteed.
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
    let mut result = None;
    let info = allocation_counter::measure(|| {
        // Let the profiler pop its frame even when the callback unwinds. No
        // caller state is inspected after a panic; its payload is resumed below.
        result = Some(catch_unwind(AssertUnwindSafe(f)));
    });
    match result.expect("Measured closure should have recorded its outcome") {
        Ok(value) => (value, info),
        Err(payload) => resume_unwind(payload),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, hint::black_box, ptr};

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
        let calls = Cell::new(0);
        // More sequential failures than the profiler's nesting limit must not
        // consume measurement frames, including when both nested calls unwind.
        for index in 0..128 {
            let payload = Box::new(index);
            let address = ptr::from_ref(payload.as_ref());
            let panic = catch_unwind(AssertUnwindSafe(|| {
                measure_with_result(|| {
                    measure_with_result(|| -> () {
                        calls.set(calls.get() + 1);
                        resume_unwind(payload);
                    })
                })
            }))
            .expect_err("the original panic should propagate");
            let payload = panic
                .downcast::<i32>()
                .expect("panic payload type is preserved");
            assert!(ptr::eq(payload.as_ref(), address));
            assert_eq!(*payload, index);
            assert_eq!(calls.get(), index + 1);

            let (value, info) = measure_with_result(|| 7);
            assert_eq!(value, 7);
            assert_eq!(info, allocation_counter::AllocationInfo::default());
        }
    }

    #[test]
    fn test_measure_with_result_nested_panic_restores_outer_measurement() {
        // Allocate the panic payload outside the measurement and retain it
        // until afterward. resume_unwind also avoids panic-hook allocations.
        let payload = Box::new(73_u32);
        let ((panic, inner), outer) = measure_with_result(|| {
            let before = black_box(vec![0_u8; 1024]);
            let panic = catch_unwind(|| measure_with_result(|| -> () { resume_unwind(payload) }))
                .expect_err("the nested panic should propagate");
            let (after, inner) = measure_with_result(|| black_box(vec![0_u8; 32]));
            drop(after);
            drop(before);
            (panic, inner)
        });

        assert_eq!(*panic.downcast::<u32>().unwrap(), 73);
        assert_eq!(inner.count_total, 1);
        assert_eq!(inner.bytes_total, 32);
        assert_eq!(inner.count_current, 1);
        assert_eq!(inner.bytes_current, 32);
        assert!(
            outer.bytes_total >= 1024 + 32,
            "outer allocation totals: {outer:?}"
        );
        assert_eq!(outer.count_current, 0);
        assert_eq!(outer.bytes_current, 0);
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
        black_box(&alloc_info.bytes_total);
        black_box(&alloc_info.count_total);
        black_box(&alloc_info.bytes_current);
        black_box(&alloc_info.count_current);
        black_box(&alloc_info.bytes_max);
        black_box(&alloc_info.count_max);

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
