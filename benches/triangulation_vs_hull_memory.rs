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
use delaunay::geometry::util::generate_random_points_seeded;
use delaunay::prelude::*;
use delaunay::vertex;
use std::fs::File;
use std::hint::black_box;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;

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
            tri_allocations: tri_info.count_total.try_into().unwrap_or_default(),
            tri_bytes: tri_info.bytes_total.try_into().unwrap_or_default(),
            hull_allocations: hull_info.count_total.try_into().unwrap_or_default(),
            hull_bytes: hull_info.bytes_total.try_into().unwrap_or_default(),
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

/// Macro to generate dimension-specific hull memory measurement functions
macro_rules! generate_hull_memory_measurement {
    ($name:ident, $dim:literal, $point_type:ty) => {
        /// Measure memory usage for triangulation and hull construction
        #[allow(unused_variables)]
        fn $name(points: &[$point_type]) -> HullMemoryRecord {
            let (tds, tri_info) = measure_with_result(|| {
                let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                Tds::<f64, (), (), $dim>::new(&vertices).unwrap()
            });

            let num_vertices = tds.number_of_vertices();
            let num_cells = tds.number_of_cells();

            let (hull, hull_info) =
                measure_with_result(|| ConvexHull::from_triangulation(&tds).unwrap());
            let hull_facets = hull.facet_count();

            drop(hull);
            drop(tds);

            #[cfg(feature = "count-allocations")]
            return HullMemoryRecord::new(
                $dim,
                points.len(),
                num_vertices,
                num_cells,
                hull_facets,
                &tri_info,
                &hull_info,
            );

            #[cfg(not(feature = "count-allocations"))]
            return HullMemoryRecord::new_placeholder(
                $dim,
                points.len(),
                num_vertices,
                num_cells,
                hull_facets,
            );
        }
    };
}

// Generate dimension-specific hull memory measurement functions
generate_hull_memory_measurement!(measure_hull_memory_2d, 2, Point<f64, 2>);
generate_hull_memory_measurement!(measure_hull_memory_3d, 3, Point<f64, 3>);
generate_hull_memory_measurement!(measure_hull_memory_4d, 4, Point<f64, 4>);
generate_hull_memory_measurement!(measure_hull_memory_5d, 5, Point<f64, 5>);

/// Macro to generate hull memory benchmark functions
macro_rules! generate_hull_memory_benchmark {
    ($name:ident, $dim:literal, $measure_fn:ident, $point_counts:expr, $sample_size:expr, $seed:expr) => {
        /// Benchmark hull memory usage
        fn $name(c: &mut Criterion) {
            let point_counts = $point_counts;
            let mut group = c.benchmark_group(&format!("hull_memory_{}d", $dim));

            if $sample_size > 0 {
                group.sample_size($sample_size);
            }

            for &n_points in &point_counts {
                group.throughput(Throughput::Elements(n_points as u64));

                group.bench_with_input(
                    BenchmarkId::new(&format!("hull_{}d", $dim), n_points),
                    &n_points,
                    |b, &n_points| {
                        let points: Vec<Point<f64, $dim>> =
                            generate_random_points_seeded(n_points, (-100.0, 100.0), $seed)
                                .unwrap();

                        // Store a single measurement for CSV output
                        let record = $measure_fn(&points);
                        HULL_MEMORY_RECORDS.lock().unwrap().push(record);

                        b.iter(|| {
                            let record = $measure_fn(&points);
                            black_box(record);
                        });
                    },
                );
            }

            group.finish();
        }
    };
}

// Generate hull memory benchmark functions
generate_hull_memory_benchmark!(
    benchmark_hull_memory_2d,
    2,
    measure_hull_memory_2d,
    [10, 20, 50, 100],
    0,
    54321
);
generate_hull_memory_benchmark!(
    benchmark_hull_memory_3d,
    3,
    measure_hull_memory_3d,
    [10, 20, 30, 50],
    20,
    65432
);
generate_hull_memory_benchmark!(
    benchmark_hull_memory_4d,
    4,
    measure_hull_memory_4d,
    [10, 15, 20],
    10,
    76543
);
generate_hull_memory_benchmark!(
    benchmark_hull_memory_5d,
    5,
    measure_hull_memory_5d,
    [8, 10, 12],
    5,
    87654
);

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

/// Write all collected hull memory records to CSV
fn write_hull_memory_records_to_csv() {
    let target_dir = Path::new("target");
    if !target_dir.exists()
        && let Err(e) = std::fs::create_dir_all(target_dir)
    {
        eprintln!("Warning: Failed to create target directory: {e}");
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
