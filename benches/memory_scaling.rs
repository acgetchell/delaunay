//! Memory scaling benchmark for Delaunay triangulation
//!
//! This benchmark measures memory allocation patterns as triangulation size increases,
//! providing insights into memory efficiency and scaling behavior across dimensions.

#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
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

/// Memory measurement record for CSV output
#[derive(Debug, Clone)]
struct MemoryRecord {
    dimension: usize,
    points: usize,
    vertices: usize,
    cells: usize,
    allocations_total: usize,
    bytes_total: usize,
    allocations_peak: usize,
    bytes_peak: usize,
    bytes_per_vertex: f64,
    bytes_per_cell: f64,
}

impl MemoryRecord {
    /// Create a new memory record
    #[cfg(feature = "count-allocations")]
    #[allow(clippy::cast_precision_loss)]
    fn new(
        dimension: usize,
        points: usize,
        vertices: usize,
        cells: usize,
        info: &allocation_counter::AllocationInfo,
    ) -> Self {
        let bytes_per_vertex = if vertices > 0 {
            info.bytes_total as f64 / vertices as f64
        } else {
            0.0
        };

        let bytes_per_cell = if cells > 0 {
            info.bytes_total as f64 / cells as f64
        } else {
            0.0
        };

        Self {
            dimension,
            points,
            vertices,
            cells,
            allocations_total: info.count_total.try_into().unwrap_or(0),
            bytes_total: info.bytes_total.try_into().unwrap_or(0),
            allocations_peak: info.count_max.try_into().unwrap_or(0),
            bytes_peak: info.bytes_max.try_into().unwrap_or(0),
            bytes_per_vertex,
            bytes_per_cell,
        }
    }

    /// Create a placeholder record when allocation counting is disabled
    #[cfg(not(feature = "count-allocations"))]
    fn new_placeholder(dimension: usize, points: usize, vertices: usize, cells: usize) -> Self {
        Self {
            dimension,
            points,
            vertices,
            cells,
            allocations_total: 0,
            bytes_total: 0,
            allocations_peak: 0,
            bytes_peak: 0,
            bytes_per_vertex: 0.0,
            bytes_per_cell: 0.0,
        }
    }

    /// Write CSV header
    fn write_csv_header(writer: &mut impl Write) -> std::io::Result<()> {
        writeln!(
            writer,
            "dimension,points,vertices,cells,allocations_total,bytes_total,allocations_peak,bytes_peak,bytes_per_vertex,bytes_per_cell"
        )
    }

    /// Write this record as a CSV row
    fn write_csv_row(&self, writer: &mut impl Write) -> std::io::Result<()> {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{:.2},{:.2}",
            self.dimension,
            self.points,
            self.vertices,
            self.cells,
            self.allocations_total,
            self.bytes_total,
            self.allocations_peak,
            self.bytes_peak,
            self.bytes_per_vertex,
            self.bytes_per_cell
        )
    }
}

/// Global storage for memory records (safe access using Mutex)
static MEMORY_RECORDS: Mutex<Vec<MemoryRecord>> = Mutex::new(Vec::new());

/// Helper function to measure memory usage
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

// Point generation functions are now imported from util module

/// Store a memory record safely
fn store_memory_record(record: MemoryRecord) {
    if let Ok(mut records) = MEMORY_RECORDS.lock() {
        records.push(record);
    }
}

/// Measure memory for 2D triangulations
#[allow(unused_variables)] // info is used conditionally based on count-allocations feature
fn measure_2d_memory(points: &[Point<f64, 2>], n_points: usize) -> MemoryRecord {
    let (tds, info) = measure_with_result(|| {
        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
        Tds::<f64, (), (), 2>::new(&vertices).unwrap()
    });

    let vertices = tds.number_of_vertices();
    let cells = tds.number_of_cells();

    #[cfg(feature = "count-allocations")]
    return MemoryRecord::new(2, n_points, vertices, cells, &info);

    #[cfg(not(feature = "count-allocations"))]
    return MemoryRecord::new_placeholder(2, n_points, vertices, cells);
}

/// Measure memory for 3D triangulations
#[allow(unused_variables)] // info is used conditionally based on count-allocations feature
fn measure_3d_memory(points: &[Point<f64, 3>], n_points: usize) -> MemoryRecord {
    let (tds, info) = measure_with_result(|| {
        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
        Tds::<f64, (), (), 3>::new(&vertices).unwrap()
    });

    let vertices = tds.number_of_vertices();
    let cells = tds.number_of_cells();

    #[cfg(feature = "count-allocations")]
    return MemoryRecord::new(3, n_points, vertices, cells, &info);

    #[cfg(not(feature = "count-allocations"))]
    return MemoryRecord::new_placeholder(3, n_points, vertices, cells);
}

/// Measure memory for 4D triangulations
#[allow(unused_variables)] // info is used conditionally based on count-allocations feature
fn measure_4d_memory(points: &[Point<f64, 4>], n_points: usize) -> MemoryRecord {
    let (tds, info) = measure_with_result(|| {
        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
        Tds::<f64, (), (), 4>::new(&vertices).unwrap()
    });

    let vertices = tds.number_of_vertices();
    let cells = tds.number_of_cells();

    #[cfg(feature = "count-allocations")]
    return MemoryRecord::new(4, n_points, vertices, cells, &info);

    #[cfg(not(feature = "count-allocations"))]
    return MemoryRecord::new_placeholder(4, n_points, vertices, cells);
}

/// Measure memory for 5D triangulations
#[allow(unused_variables)] // info is used conditionally based on count-allocations feature
fn measure_5d_memory(points: &[Point<f64, 5>], n_points: usize) -> MemoryRecord {
    let (tds, info) = measure_with_result(|| {
        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
        Tds::<f64, (), (), 5>::new(&vertices).unwrap()
    });

    let vertices = tds.number_of_vertices();
    let cells = tds.number_of_cells();

    #[cfg(feature = "count-allocations")]
    return MemoryRecord::new(5, n_points, vertices, cells, &info);

    #[cfg(not(feature = "count-allocations"))]
    return MemoryRecord::new_placeholder(5, n_points, vertices, cells);
}

/// Benchmark memory scaling for 2D triangulations
fn benchmark_memory_scaling_2d(c: &mut Criterion) {
    let point_counts = [10, 20, 50, 100];
    let mut group = c.benchmark_group("memory_scaling_2d");

    for &n_points in &point_counts {
        group.throughput(Throughput::Elements(n_points as u64));

        group.bench_with_input(
            BenchmarkId::new("triangulation_2d", n_points),
            &n_points,
            |b, &n_points| {
                let points = generate_random_points_2d_seeded(n_points, 12345);

                b.iter(|| {
                    let record = measure_2d_memory(&points, n_points);
                    store_memory_record(record.clone());
                    black_box(record)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory scaling for 3D triangulations
fn benchmark_memory_scaling_3d(c: &mut Criterion) {
    let point_counts = [10, 20, 30];
    let mut group = c.benchmark_group("memory_scaling_3d");
    group.sample_size(20);

    for &n_points in &point_counts {
        group.throughput(Throughput::Elements(n_points as u64));

        group.bench_with_input(
            BenchmarkId::new("triangulation_3d", n_points),
            &n_points,
            |b, &n_points| {
                let points = generate_random_points_3d_seeded(n_points, 23456);

                b.iter(|| {
                    let record = measure_3d_memory(&points, n_points);
                    store_memory_record(record.clone());
                    black_box(record)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory scaling for 4D triangulations
fn benchmark_memory_scaling_4d(c: &mut Criterion) {
    let point_counts = [10, 15];
    let mut group = c.benchmark_group("memory_scaling_4d");
    group.sample_size(10);

    for &n_points in &point_counts {
        group.throughput(Throughput::Elements(n_points as u64));

        group.bench_with_input(
            BenchmarkId::new("triangulation_4d", n_points),
            &n_points,
            |b, &n_points| {
                let points = generate_random_points_4d_seeded(n_points, 34567);

                b.iter(|| {
                    let record = measure_4d_memory(&points, n_points);
                    store_memory_record(record.clone());
                    black_box(record)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory scaling for 5D triangulations
fn benchmark_memory_scaling_5d(c: &mut Criterion) {
    let point_counts = [8, 10];
    let mut group = c.benchmark_group("memory_scaling_5d");
    group.sample_size(5); // Further reduce sample size for 5D

    for &n_points in &point_counts {
        group.throughput(Throughput::Elements(n_points as u64));

        group.bench_with_input(
            BenchmarkId::new("triangulation_5d", n_points),
            &n_points,
            |b, &n_points| {
                let points = generate_random_points_5d_seeded(n_points, 45678);

                b.iter(|| {
                    let record = measure_5d_memory(&points, n_points);
                    store_memory_record(record.clone());
                    black_box(record)
                });
            },
        );
    }

    group.finish();
}

/// Write all collected memory records to CSV
fn write_memory_records_to_csv() {
    let target_dir = Path::new("target");
    if !target_dir.exists() {
        let _ = std::fs::create_dir_all(target_dir);
    }

    let csv_path = target_dir.join("memory_scaling.csv");

    if let Ok(mut file) = File::create(&csv_path) {
        let _ = MemoryRecord::write_csv_header(&mut file);

        if let Ok(records) = MEMORY_RECORDS.lock() {
            for record in records.iter() {
                let _ = record.write_csv_row(&mut file);
            }
        }

        println!("Memory scaling results written to: {}", csv_path.display());
    }
}

/// Print summary of collected records
fn print_memory_summary() {
    if let Ok(records) = MEMORY_RECORDS.lock()
        && !records.is_empty()
    {
        println!("\n=== Memory Scaling Summary ===");
        println!("Total measurements: {}", records.len());

        for dimension in [2, 3, 4, 5] {
            let dim_records: Vec<_> = records
                .iter()
                .filter(|r| r.dimension == dimension)
                .collect();

            if !dim_records.is_empty() {
                println!("\n{dimension}D Triangulations:");
                for record in &dim_records {
                    #[cfg(feature = "count-allocations")]
                    {
                        #[allow(clippy::cast_precision_loss)]
                        let kb_total = record.bytes_total as f64 / 1024.0;
                        println!(
                            "  {} points: {} vertices, {} cells, {:.1} KB total, {:.2} bytes/vertex",
                            record.points,
                            record.vertices,
                            record.cells,
                            kb_total,
                            record.bytes_per_vertex
                        );
                    }

                    #[cfg(not(feature = "count-allocations"))]
                    println!(
                        "  {} points: {} vertices, {} cells (allocation counting disabled)",
                        record.points, record.vertices, record.cells
                    );
                }
            }
        }
    }
}

/// Custom benchmark configuration that writes CSV after completion
struct MemoryBenchmarkGroup;

impl Drop for MemoryBenchmarkGroup {
    fn drop(&mut self) {
        write_memory_records_to_csv();
        print_memory_summary();
    }
}

/// Main benchmark function that includes cleanup
fn run_all_memory_benchmarks(c: &mut Criterion) {
    let _cleanup = MemoryBenchmarkGroup;

    benchmark_memory_scaling_2d(c);
    benchmark_memory_scaling_3d(c);
    benchmark_memory_scaling_4d(c);
    benchmark_memory_scaling_5d(c);
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = run_all_memory_benchmarks
);
criterion_main!(benches);
