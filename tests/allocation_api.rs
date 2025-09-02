//! Test file to figure out allocation-counter API with Delaunay triangulation testing
//!
//! This module provides comprehensive testing utilities for memory allocation tracking
//! in Delaunay triangulation operations.

// Import Delaunay triangulation crate components
use delaunay::core::triangulation_data_structure::Tds;
use delaunay::geometry::Point;

// Testing utilities
use rand::Rng;

/// Common test helpers for initializing and working with the allocator
pub mod test_helpers {
    use super::{Point, Rng, Tds};
    #[cfg(feature = "count-allocations")]
    use allocation_counter::measure;
    use delaunay::geometry::Coordinate;
    use delaunay::vertex;

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
    /// Panics if TDS creation fails.
    #[must_use]
    pub fn create_test_tds() -> Tds<f64, Option<()>, Option<()>, 4> {
        // Create an empty TDS with no vertices
        Tds::default()
    }

    /// Create a triangulation with some test vertices
    ///
    /// # Panics
    ///
    /// Panics if TDS creation with vertices fails.
    #[must_use]
    pub fn create_test_tds_with_vertices() -> Tds<f64, Option<()>, Option<()>, 3> {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        Tds::new(&vertices).expect("Failed to create TDS with vertices")
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

/// Test the allocation counter API
fn main() {
    test_helpers::init_test_env();
    println!("Testing allocation counter APIs with Delaunay triangulation...");

    // Test 1: Basic allocation counting
    println!("\n1. Testing basic allocation counting:");
    #[cfg(feature = "count-allocations")]
    {
        let (result, info) = test_helpers::measure_with_result(|| {
            let _v: Vec<i32> = (0..100).collect();
            println!("Created vector with {} elements", 100);
            "success"
        });
        println!("✓ allocation_counter::measure() works: {result}");
        test_helpers::print_alloc_summary(&info, "basic vector creation");
    }

    #[cfg(not(feature = "count-allocations"))]
    {
        println!("count-allocations feature not enabled");
        let (result, info) = test_helpers::measure_with_result(|| {
            let v: Vec<i32> = (0..100).collect();
            println!("Created vector with {} elements", v.len());
            "success"
        });
        println!("✓ Fallback executed: {result}");
        test_helpers::print_alloc_summary(&info, "basic vector creation");
    }

    // Test 2: Point creation
    println!("\n2. Testing Point creation:");
    let (points_2d, info_2d) =
        test_helpers::measure_with_result(|| test_helpers::create_test_points_2d(50));
    println!("✓ Created {} 2D points", points_2d.len());
    test_helpers::print_alloc_summary(&info_2d, "2D points creation");

    // Test 3: 3D Point creation
    println!("\n3. Testing 3D Point creation:");
    let (points_3d, info_3d) =
        test_helpers::measure_with_result(|| test_helpers::create_test_points_3d(25));
    println!("✓ Created {} 3D points", points_3d.len());
    test_helpers::print_alloc_summary(&info_3d, "3D points creation");

    // Test 4: TDS creation
    println!("\n4. Testing TDS creation:");
    let (tds, info_tds) = test_helpers::measure_with_result(test_helpers::create_test_tds);
    println!("✓ Created TDS with {} vertices", tds.number_of_vertices());
    test_helpers::print_alloc_summary(&info_tds, "TDS creation");

    // Test 5: Complex workflow
    println!("\n5. Testing complex triangulation workflow:");
    let (workflow_result, info_workflow) = test_helpers::measure_with_result(|| {
        // Create various geometric objects
        let points_2d = test_helpers::create_test_points_2d(10);
        let points_3d = test_helpers::create_test_points_3d(10);
        let tds = test_helpers::create_test_tds();

        // Perform some basic operations
        let total_points = points_2d.len() + points_3d.len();
        let vertex_count = tds.number_of_vertices();

        (total_points, vertex_count)
    });
    println!(
        "✓ Complex workflow completed: {} total points, {} vertices",
        workflow_result.0, workflow_result.1
    );
    test_helpers::print_alloc_summary(&info_workflow, "complex workflow");

    println!("\n🎉 All allocation counter tests completed successfully!");
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

        let (tds, info) = measure_with_result(create_test_tds);

        // Verify TDS was created successfully
        assert_eq!(tds.number_of_vertices(), 0);
        print_alloc_summary(&info, "TDS creation");
    }

    #[test]
    fn test_complex_triangulation_workflow() {
        init_test_env();

        let (result, info) = measure_with_result(|| {
            // Create points
            let points = create_test_points_3d(5);

            // Create TDS
            let tds = create_test_tds();

            // Return some result to verify the workflow
            (points.len(), tds.number_of_vertices())
        });

        assert_eq!(result.0, 5); // 5 points created
        assert_eq!(result.1, 0); // Empty TDS
        print_alloc_summary(&info, "complex triangulation workflow");
    }
}
