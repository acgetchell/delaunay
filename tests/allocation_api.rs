//! Test file to figure out allocation-counter API with Delaunay triangulation testing
//!
//! This module provides comprehensive testing utilities for memory allocation tracking
//! in Delaunay triangulation operations.

// Import Delaunay triangulation crate components
#[cfg(feature = "count-allocations")]
use allocation_counter::measure;
use delaunay::prelude::*;

// Testing utilities
use rand::Rng;

/// Common test helpers for initializing and working with the allocator
pub mod test_helpers {
    use super::*;

    /// Initialize a simple allocator test environment
    pub fn init_test_env() {
        println!("Initializing test environment...");
        #[cfg(feature = "count-allocations")]
        println!("✓ Allocation counting enabled");
        #[cfg(not(feature = "count-allocations"))]
        println!("⚠ Allocation counting disabled - enable with --features count-allocations");
    }

    /// Helper to measure allocations with error handling
    ///
    /// # Panics
    ///
    /// Panics if the closure `f` does not complete successfully.
    #[cfg(feature = "count-allocations")]
    pub fn measure_with_result<F, R>(f: F) -> (R, allocation_counter::AllocationInfo)
    where
        F: FnOnce() -> R,
    {
        let mut result: Option<R> = None;
        let info = measure(|| {
            result = Some(f());
        });
        println!("Memory info: {info:?}");
        (result.expect("Closure should have set result"), info)
    }

    /// Fallback for when allocation counting is disabled
    #[cfg(not(feature = "count-allocations"))]
    pub fn measure_with_result<F, R>(f: F) -> (R, ())
    where
        F: FnOnce() -> R,
    {
        println!("Allocation counting not available");
        (f(), ())
    }

    /// Create a set of test points in various dimensions
    #[must_use]
    pub fn create_test_points_2d(count: usize) -> Vec<Point<f64, 2>> {
        let mut rng = rand::rng();
        (0..count)
            .map(|_| Point::new([rng.random_range(-10.0..10.0), rng.random_range(-10.0..10.0)]))
            .collect()
    }

    /// Create a set of test points in 3D
    #[must_use]
    pub fn create_test_points_3d(count: usize) -> Vec<Point<f64, 3>> {
        let mut rng = rand::rng();
        (0..count)
            .map(|_| {
                Point::new([
                    rng.random_range(-10.0..10.0),
                    rng.random_range(-10.0..10.0),
                    rng.random_range(-10.0..10.0),
                ])
            })
            .collect()
    }

    /// Create a simple triangulation data structure for testing
    ///
    /// # Panics
    ///
    /// Panics if triangulation creation fails.
    #[must_use]
    pub fn create_test_tds()
    -> DelaunayTriangulation<delaunay::geometry::kernel::FastKernel<f64>, (), (), 4> {
        // Create an empty triangulation with no vertices
        DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold)
    }

    /// Create a triangulation with some test vertices
    ///
    /// # Panics
    ///
    /// Panics if triangulation creation with vertices fails.
    #[must_use]
    pub fn create_test_tds_with_vertices()
    -> DelaunayTriangulation<delaunay::geometry::kernel::FastKernel<f64>, (), (), 3> {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        DelaunayTriangulation::new_with_topology_guarantee(&vertices, TopologyGuarantee::PLManifold)
            .expect("Failed to create triangulation with vertices")
    }

    /// Print memory allocation summary
    #[cfg(feature = "count-allocations")]
    pub fn print_alloc_summary(info: &allocation_counter::AllocationInfo, operation: &str) {
        println!("\n=== Memory Allocation Summary for {operation} ===");
        println!("Total allocations: {}", info.count_total);
        println!("Current allocations: {}", info.count_current);
        println!("Max allocations: {}", info.count_max);
        println!("Total bytes allocated: {}", info.bytes_total);
        println!("Current bytes allocated: {}", info.bytes_current);
        println!("Max bytes allocated: {}", info.bytes_max);
        println!("=====================================\n");
    }

    /// Print memory allocation summary (fallback for when allocation counting is disabled)
    #[cfg(not(feature = "count-allocations"))]
    pub fn print_alloc_summary(_info: &(), operation: &str) {
        println!("\n=== Memory Allocation Summary for {operation} ===");
        println!("Allocation counting not enabled");
        println!("=====================================\n");
    }
}

#[cfg(test)]
mod tests {
    use super::test_helpers::*;
    // Only import what we actually use from the parent module
    // (the test_helpers module provides all the functions we need)

    #[test]
    fn test_basic_allocation_counting() {
        init_test_env();

        let (result, info) = measure_with_result(|| {
            let _v: Vec<i32> = (0..100).collect();
            42
        });

        assert_eq!(result, 42);
        print_alloc_summary(&info, "basic vector creation");
    }

    #[test]
    fn test_point_creation_allocations() {
        init_test_env();

        let (points, info) = measure_with_result(|| create_test_points_2d(10));

        assert_eq!(points.len(), 10);
        print_alloc_summary(&info, "2D point creation");
    }

    #[test]
    fn test_3d_point_creation_allocations() {
        init_test_env();

        let (points, info) = measure_with_result(|| create_test_points_3d(10));

        assert_eq!(points.len(), 10);
        print_alloc_summary(&info, "3D point creation");
    }

    #[test]
    fn test_tds_creation_allocations() {
        init_test_env();

        let (dt, info) = measure_with_result(create_test_tds);

        // Verify triangulation was created successfully
        assert_eq!(dt.number_of_vertices(), 0);
        print_alloc_summary(&info, "triangulation creation");
    }

    #[test]
    fn test_complex_triangulation_workflow() {
        init_test_env();

        let (result, info) = measure_with_result(|| {
            // Create points
            let points = create_test_points_3d(5);

            // Create triangulation
            let dt = create_test_tds();

            // Return some result to verify the workflow
            (points.len(), dt.number_of_vertices())
        });

        assert_eq!(result.0, 5); // 5 points created
        assert_eq!(result.1, 0); // Empty triangulation
        print_alloc_summary(&info, "complex triangulation workflow");
    }
}
