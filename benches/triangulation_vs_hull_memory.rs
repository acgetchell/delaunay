//! Triangulation vs Hull memory usage benchmark
//!
//! This benchmark measures and compares memory consumption patterns between:
//! - Delaunay triangulation (TDS) construction
//! - Convex hull extraction from triangulations
//!
//! Provides comparative analysis of memory efficiency and allocation patterns.

#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::geometry::algorithms::ConvexHull;
use delaunay::prelude::*;
use delaunay::vertex;
use std::fs::File;
use std::hint::black_box;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;

mod util;
use util::{
    generate_random_points_2d_seeded, generate_random_points_3d_seeded,
    generate_random_points_4d_seeded, generate_random_points_5d_seeded,
};

/// Convex hull memory measurement record
#[derive(Debug, Clone)]
struct HullMemoryRecord {
    dimension: usize,
    points: usize,
    vertices: usize,
    cells: usize,
    hull_facets: usize,
    tri_allocations: usize,
    tri_bytes: usize,
    hull_allocations: usize,
    hull_bytes: usize,
    tri_bytes_per_vertex: f64,
    hull_bytes_per_facet: f64,
    hull_to_tri_ratio: f64,
}

impl HullMemoryRecord {
    /// Create a new hull memory record
    #[cfg(feature = "count-allocations")]
    #[allow(clippy::cast_precision_loss)]
    fn new(
        dimension: usize,
        points: usize,
        vertices: usize,
        cells: usize,
        hull_facets: usize,
        tri_info: &allocation_counter::AllocationInfo,
        hull_info: &allocation_counter::AllocationInfo,
    ) -> Self {
        let tri_bytes_per_vertex = if vertices > 0 {
            tri_info.bytes_total as f64 / vertices as f64
        } else {
            0.0
        };

        let hull_bytes_per_facet = if hull_facets > 0 {
            hull_info.bytes_total as f64 / hull_facets as f64
        } else {
            0.0
        };

        let hull_to_tri_ratio = if tri_info.bytes_total > 0 {
            hull_info.bytes_total as f64 / tri_info.bytes_total as f64
        } else {
            0.0
        };

        Self {
            dimension,
            points,
            vertices,
            cells,
            hull_facets,
            tri_allocations: tri_info.count_total.try_into().unwrap(),
            tri_bytes: tri_info.bytes_total.try_into().unwrap(),
            hull_allocations: hull_info.count_total.try_into().unwrap(),
            hull_bytes: hull_info.bytes_total.try_into().unwrap(),
            tri_bytes_per_vertex,
            hull_bytes_per_facet,
            hull_to_tri_ratio,
        }
    }

    /// Create a placeholder record when allocation counting is disabled
    #[cfg(not(feature = "count-allocations"))]
    fn new_placeholder(
        dimension: usize,
        points: usize,
        vertices: usize,
        cells: usize,
        hull_facets: usize,
    ) -> Self {
        Self {
            dimension,
            points,
            vertices,
            cells,
            hull_facets,
            tri_allocations: 0,
            tri_bytes: 0,
            hull_allocations: 0,
            hull_bytes: 0,
            tri_bytes_per_vertex: 0.0,
            hull_bytes_per_facet: 0.0,
            hull_to_tri_ratio: 0.0,
        }
    }

    /// Write CSV header
    fn write_csv_header(writer: &mut impl Write) -> std::io::Result<()> {
        writeln!(
            writer,
            "dimension,points,vertices,cells,hull_facets,tri_allocations,tri_bytes,hull_allocations,hull_bytes,tri_bytes_per_vertex,hull_bytes_per_facet,hull_to_tri_ratio"
        )
    }

    /// Write this record as a CSV row
    fn write_csv_row(&self, writer: &mut impl Write) -> std::io::Result<()> {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{:.2},{:.2},{:.4}",
            self.dimension,
            self.points,
            self.vertices,
            self.cells,
            self.hull_facets,
            self.tri_allocations,
            self.tri_bytes,
            self.hull_allocations,
            self.hull_bytes,
            self.tri_bytes_per_vertex,
            self.hull_bytes_per_facet,
            self.hull_to_tri_ratio
        )
    }
}

// Point generation functions are now imported from util module

/// Store hull memory records for later CSV output
static HULL_MEMORY_RECORDS: Mutex<Vec<HullMemoryRecord>> = Mutex::new(Vec::new());

/// Helper function to measure memory usage with result
#[cfg(feature = "count-allocations")]
fn measure_with_result<F, R>(f: F) -> (R, allocation_counter::AllocationInfo)
where
    F: FnOnce() -> R,
{
    let mut result: Option<R> = None;
    let info = allocation_counter::measure(|| {
        result = Some(f());
    });
    (result.expect("Closure should have set result"), info)
}

#[cfg(not(feature = "count-allocations"))]
fn measure_with_result<F, R>(f: F) -> (R, ())
where
    F: FnOnce() -> R,
{
    (f(), ())
}

/// Measure memory usage for 2D triangulation and hull construction
#[allow(unused_variables)] // tri_info and hull_info are used conditionally based on count-allocations feature
fn measure_hull_memory_2d(points: &[Point<f64, 2>]) -> HullMemoryRecord {
    let (tds, tri_info) = measure_with_result(|| {
        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
        Tds::<f64, (), (), 2>::new(&vertices).unwrap()
    });

    let num_vertices = tds.number_of_vertices();
    let num_cells = tds.number_of_cells();

    // Measure hull extraction
    let (hull, hull_info) = measure_with_result(|| ConvexHull::from_triangulation(&tds).unwrap());

    let hull_facets = hull.facet_count();

    // Drop structures to release memory
    drop(hull);
    drop(tds);

    #[cfg(feature = "count-allocations")]
    return HullMemoryRecord::new(
        2,
        points.len(),
        num_vertices,
        num_cells,
        hull_facets,
        &tri_info,
        &hull_info,
    );

    #[cfg(not(feature = "count-allocations"))]
    return HullMemoryRecord::new_placeholder(
        2,
        points.len(),
        num_vertices,
        num_cells,
        hull_facets,
    );
}

/// Measure memory usage for 3D triangulation and hull construction
#[allow(unused_variables)] // tri_info and hull_info are used conditionally based on count-allocations feature
fn measure_hull_memory_3d(points: &[Point<f64, 3>]) -> HullMemoryRecord {
    let (tds, tri_info) = measure_with_result(|| {
        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
        Tds::<f64, (), (), 3>::new(&vertices).unwrap()
    });

    let num_vertices = tds.number_of_vertices();
    let num_cells = tds.number_of_cells();

    // Measure hull extraction
    let (hull, hull_info) = measure_with_result(|| ConvexHull::from_triangulation(&tds).unwrap());

    let hull_facets = hull.facet_count();

    // Drop structures to release memory
    drop(hull);
    drop(tds);

    #[cfg(feature = "count-allocations")]
    return HullMemoryRecord::new(
        3,
        points.len(),
        num_vertices,
        num_cells,
        hull_facets,
        &tri_info,
        &hull_info,
    );

    #[cfg(not(feature = "count-allocations"))]
    return HullMemoryRecord::new_placeholder(
        3,
        points.len(),
        num_vertices,
        num_cells,
        hull_facets,
    );
}

/// Measure memory usage for 4D triangulation and hull construction
#[allow(unused_variables)] // tri_info and hull_info are used conditionally based on count-allocations feature
fn measure_hull_memory_4d(points: &[Point<f64, 4>]) -> HullMemoryRecord {
    let (tds, tri_info) = measure_with_result(|| {
        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
        Tds::<f64, (), (), 4>::new(&vertices).unwrap()
    });

    let num_vertices = tds.number_of_vertices();
    let num_cells = tds.number_of_cells();

    // Measure hull extraction
    let (hull, hull_info) = measure_with_result(|| ConvexHull::from_triangulation(&tds).unwrap());

    let hull_facets = hull.facet_count();

    // Drop structures to release memory
    drop(hull);
    drop(tds);

    #[cfg(feature = "count-allocations")]
    return HullMemoryRecord::new(
        4,
        points.len(),
        num_vertices,
        num_cells,
        hull_facets,
        &tri_info,
        &hull_info,
    );

    #[cfg(not(feature = "count-allocations"))]
    return HullMemoryRecord::new_placeholder(
        4,
        points.len(),
        num_vertices,
        num_cells,
        hull_facets,
    );
}

/// Measure memory usage for 5D triangulation and hull construction
#[allow(unused_variables)] // tri_info and hull_info are used conditionally based on count-allocations feature
fn measure_hull_memory_5d(points: &[Point<f64, 5>]) -> HullMemoryRecord {
    let (tds, tri_info) = measure_with_result(|| {
        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
        Tds::<f64, (), (), 5>::new(&vertices).unwrap()
    });

    let num_vertices = tds.number_of_vertices();
    let num_cells = tds.number_of_cells();

    // Measure hull extraction
    let (hull, hull_info) = measure_with_result(|| ConvexHull::from_triangulation(&tds).unwrap());

    let hull_facets = hull.facet_count();

    // Drop structures to release memory
    drop(hull);
    drop(tds);

    #[cfg(feature = "count-allocations")]
    return HullMemoryRecord::new(
        5,
        points.len(),
        num_vertices,
        num_cells,
        hull_facets,
        &tri_info,
        &hull_info,
    );

    #[cfg(not(feature = "count-allocations"))]
    return HullMemoryRecord::new_placeholder(
        5,
        points.len(),
        num_vertices,
        num_cells,
        hull_facets,
    );
}

/// Benchmark hull memory usage for 2D
fn benchmark_hull_memory_2d(c: &mut Criterion) {
    let point_counts = [10, 20, 50, 100];
    let mut group = c.benchmark_group("hull_memory_2d");

    for &n_points in &point_counts {
        group.throughput(Throughput::Elements(n_points as u64));

        group.bench_with_input(
            BenchmarkId::new("hull_2d", n_points),
            &n_points,
            |b, &n_points| {
                let points = generate_random_points_2d_seeded(n_points, 54321);

                b.iter(|| {
                    let record = measure_hull_memory_2d(&points);

                    // Store record for CSV output
                    HULL_MEMORY_RECORDS.lock().unwrap().push(record.clone());

                    black_box(record);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark hull memory usage for 3D
fn benchmark_hull_memory_3d(c: &mut Criterion) {
    let point_counts = [10, 20, 30, 50];
    let mut group = c.benchmark_group("hull_memory_3d");

    // Reduce sample size for 3D
    group.sample_size(20);

    for &n_points in &point_counts {
        group.throughput(Throughput::Elements(n_points as u64));

        group.bench_with_input(
            BenchmarkId::new("hull_3d", n_points),
            &n_points,
            |b, &n_points| {
                let points = generate_random_points_3d_seeded(n_points, 65432);

                b.iter(|| {
                    let record = measure_hull_memory_3d(&points);

                    // Store record for CSV output
                    HULL_MEMORY_RECORDS.lock().unwrap().push(record.clone());

                    black_box(record);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark hull memory usage for 4D
fn benchmark_hull_memory_4d(c: &mut Criterion) {
    let point_counts = [10, 15, 20];
    let mut group = c.benchmark_group("hull_memory_4d");

    // Further reduce sample size for 4D
    group.sample_size(10);

    for &n_points in &point_counts {
        group.throughput(Throughput::Elements(n_points as u64));

        group.bench_with_input(
            BenchmarkId::new("hull_4d", n_points),
            &n_points,
            |b, &n_points| {
                let points = generate_random_points_4d_seeded(n_points, 76543);

                b.iter(|| {
                    let record = measure_hull_memory_4d(&points);

                    // Store record for CSV output
                    HULL_MEMORY_RECORDS.lock().unwrap().push(record.clone());

                    black_box(record);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark hull memory usage for 5D
fn benchmark_hull_memory_5d(c: &mut Criterion) {
    let point_counts = [8, 10, 12];
    let mut group = c.benchmark_group("hull_memory_5d");

    // Minimal sample size for 5D due to computational complexity
    group.sample_size(5);

    for &n_points in &point_counts {
        group.throughput(Throughput::Elements(n_points as u64));

        group.bench_with_input(
            BenchmarkId::new("hull_5d", n_points),
            &n_points,
            |b, &n_points| {
                let points = generate_random_points_5d_seeded(n_points, 87654);

                b.iter(|| {
                    let record = measure_hull_memory_5d(&points);

                    // Store record for CSV output
                    HULL_MEMORY_RECORDS.lock().unwrap().push(record.clone());

                    black_box(record);
                });
            },
        );
    }

    group.finish();
}

/// Write all collected hull memory records to CSV
fn write_hull_memory_records_to_csv() {
    let target_dir = Path::new("target");
    if !target_dir.exists() {
        let _ = std::fs::create_dir_all(target_dir);
    }

    let csv_path = target_dir.join("convex_hull_memory.csv");

    if let Ok(mut file) = File::create(&csv_path) {
        let _ = HullMemoryRecord::write_csv_header(&mut file);

        {
            let records = HULL_MEMORY_RECORDS.lock().unwrap();
            for record in records.iter() {
                let _ = record.write_csv_row(&mut file);
            }
        }

        println!(
            "Convex hull memory results written to: {}",
            csv_path.display()
        );
    }
}

/// Custom benchmark cleanup for hull memory
struct HullBenchmarkGroup;

impl Drop for HullBenchmarkGroup {
    fn drop(&mut self) {
        write_hull_memory_records_to_csv();
    }
}

/// Main hull benchmark function
fn run_all_hull_memory_benchmarks(c: &mut Criterion) {
    let _cleanup = HullBenchmarkGroup;

    benchmark_hull_memory_2d(c);
    benchmark_hull_memory_3d(c);
    benchmark_hull_memory_4d(c);
    benchmark_hull_memory_5d(c);

    // Print summary
    {
        let all_records = {
            let records = HULL_MEMORY_RECORDS.lock().unwrap();
            records.clone()
        };

        if !all_records.is_empty() {
            println!("\n=== Convex Hull Memory Summary ===");
            println!("Total measurements: {}", all_records.len());

            for dimension in [2, 3, 4, 5] {
                let dim_records: Vec<_> = all_records
                    .iter()
                    .filter(|r| r.dimension == dimension)
                    .collect();

                if !dim_records.is_empty() {
                    println!("\n{dimension}D Convex Hulls:");
                    for record in &dim_records {
                        #[cfg(feature = "count-allocations")]
                        {
                            #[allow(clippy::cast_precision_loss)]
                            let tri_kb = record.tri_bytes as f64 / 1024.0;
                            #[allow(clippy::cast_precision_loss)]
                            let hull_kb = record.hull_bytes as f64 / 1024.0;
                            println!(
                                "  {} points: {} hull facets, tri={:.1}KB hull={:.1}KB ratio={:.3}",
                                record.points,
                                record.hull_facets,
                                tri_kb,
                                hull_kb,
                                record.hull_to_tri_ratio
                            );
                        }

                        #[cfg(not(feature = "count-allocations"))]
                        println!(
                            "  {} points: {} hull facets (allocation counting disabled)",
                            record.points, record.hull_facets
                        );
                    }
                }
            }
        }
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = run_all_hull_memory_benchmarks
);
criterion_main!(benches);
