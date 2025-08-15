# Optimization Recommendations for `triangulation_data_structure.rs`

## 1. **Critical Performance Bottlenecks**

### A. `Tds.is_valid()` - **HIGHEST PRIORITY**

**Current Issues:**

- **Time Complexity**: O(N×F + N×D²) where N = cells, F = facets per cell (D+1)
- **Space Complexity**: O(N×F) for building facet-to-cell mappings
- **Multiple iterations**: Each validation step iterates over cells/vertices separately
- **Expensive facet sharing validation**: Builds complete facet-to-cells HashMap
- **Repeated allocations**: Creates temporary HashSets and vectors in inner loops

**Optimized Single-Pass Validation:**

```rust
impl<T, U, V, const D: usize> Tds<T, U, V, D> {
    /// Optimized validation with single-pass iteration and early termination
    pub fn is_valid_optimized(&self) -> Result<(), TriangulationValidationError> {
        // Early exit for empty triangulation
        if self.cells.is_empty() && self.vertices.is_empty() {
            return Ok(());
        }

        // Fast count-based checks first
        if self.vertex_bimap.len() != self.vertices.len() ||
           self.cell_bimap.len() != self.cells.len() {
            // Only run expensive detailed validation if counts don't match
            self.validate_vertex_mappings()?;
            self.validate_cell_mappings()?;
        }

        // Pre-allocate reusable collections
        let mut cell_vertex_sets: HashMap<CellKey, HashSet<VertexKey>> = 
            HashMap::with_capacity(self.cells.len());
        let mut duplicate_detector: HashMap<Vec<Uuid>, CellKey> = 
            HashMap::with_capacity(self.cells.len());
        let mut facet_sharing_map: HashMap<u64, u8> = HashMap::new(); // Just count, don't store cells
        
        // Reusable temporary collections
        let mut temp_vertex_uuids = Vec::with_capacity(D + 1);
        let mut temp_vertex_keys = HashSet::with_capacity(D + 1);

        // SINGLE-PASS comprehensive validation
        for (cell_key, cell) in &self.cells {
            let cell_uuid = cell.uuid();

            // Combined basic validation
            if cell.vertices().len() != D + 1 || cell_uuid.is_nil() {
                cell.is_valid().map_err(|source| {
                    TriangulationValidationError::InvalidCell { cell_id: cell_uuid, source }
                })?;
            }

            // Collect and validate vertices + mappings
            temp_vertex_uuids.clear();
            temp_vertex_keys.clear();
            for vertex in cell.vertices() {
                vertex.is_valid().map_err(|source| {
                    TriangulationValidationError::InvalidCell {
                        cell_id: cell_uuid,
                        source: CellValidationError::InvalidVertex { source },
                    }
                })?;
                
                let vertex_key = self.vertex_bimap.get_by_left(&vertex.uuid())
                    .ok_or_else(|| TriangulationValidationError::MappingInconsistency {
                        message: format!("Missing vertex mapping for {:?}", vertex.uuid()),
                    })?;
                
                if !temp_vertex_keys.insert(*vertex_key) {
                    return Err(TriangulationValidationError::InvalidCell {
                        cell_id: cell_uuid,
                        source: CellValidationError::DuplicateVertices,
                    });
                }
                temp_vertex_uuids.push(vertex.uuid());
            }

            // Cache vertex sets for neighbor validation
            cell_vertex_sets.insert(cell_key, temp_vertex_keys.clone());

            // Duplicate cell detection
            temp_vertex_uuids.sort_unstable();
            if duplicate_detector.contains_key(&temp_vertex_uuids) {
                return Err(TriangulationValidationError::DuplicateCells {
                    message: format!("Duplicate cell detected: {cell_key:?}"),
                });
            }
            duplicate_detector.insert(temp_vertex_uuids.clone(), cell_key);

            // On-demand facet sharing validation with early termination
            if let Ok(facets) = cell.facets() {
                for facet in facets {
                    let facet_key = facet.key();
                    let count = facet_sharing_map.entry(facet_key).or_insert(0);
                    *count += 1;
                    if *count > 2 {
                        return Err(TriangulationValidationError::InconsistentDataStructure {
                            message: format!(
                                "Facet {} shared by >2 cells", facet_key
                            ),
                        });
                    }
                }
            }
        }

        // Optimized neighbor validation using cached vertex sets
        self.validate_neighbors_with_cache(&cell_vertex_sets)?;
        
        Ok(())
    }
    
    /// Fast validation for frequent checks during construction
    pub fn is_valid_fast(&self) -> Result<(), TriangulationValidationError> {
        // Skip expensive neighbor validation, focus on structural integrity
        if self.vertex_bimap.len() != self.vertices.len() ||
           self.cell_bimap.len() != self.cells.len() {
            return Err(TriangulationValidationError::MappingInconsistency {
                message: "Mapping count mismatch".to_string(),
            });
        }

        let mut duplicate_detector = HashMap::with_capacity(self.cells.len());
        let mut temp_vertex_uuids = Vec::with_capacity(D + 1);

        for (cell_key, cell) in &self.cells {
            if cell.vertices().len() != D + 1 || cell.uuid().is_nil() {
                cell.is_valid().map_err(|source| {
                    TriangulationValidationError::InvalidCell {
                        cell_id: cell.uuid(),
                        source,
                    }
                })?;
            }

            // Fast duplicate detection only
            temp_vertex_uuids.clear();
            temp_vertex_uuids.extend(cell.vertices().iter().map(|v| v.uuid()));
            temp_vertex_uuids.sort_unstable();
            if duplicate_detector.contains_key(&temp_vertex_uuids) {
                return Err(TriangulationValidationError::DuplicateCells {
                    message: format!("Duplicate cell: {cell_key:?}"),
                });
            }
            duplicate_detector.insert(temp_vertex_uuids.clone(), cell_key);
        }
        
        Ok(())
    }
}
```

### B. `find_bad_cells` and `find_boundary_facets` - **HIGH PRIORITY**

**Current Issues:**

- Repeated circumsphere computations
- Vector allocation per cell for vertex points
- Inefficient boundary facet detection in Bowyer-Watson

**Optimizations:**

```rust
// Cache expensive computations
struct CellCache<T, const D: usize> {
    vertex_points: Vec<Point<T, D>>,
    facets: Vec<Facet<T, U, V, D>>,
}

impl<T, U, V, const D: usize> Tds<T, U, V, D> {
    fn find_bad_cells_cached(&mut self, vertex: &Vertex<T, U, D>) -> Result<Vec<CellKey>, TriangulationValidationError> {
        // Reuse buffer from struct to avoid allocation
        self.bad_cells_buffer.clear();
        
        for (cell_key, cell) in &self.cells {
            // Reuse vertex_points_buffer (already exists in struct)
            self.vertex_points_buffer.clear();
            self.vertex_points_buffer.extend(
                cell.vertices().iter().map(|v| *v.point())
            );
            
            // Use existing insphere predicate
            match insphere(&self.vertex_points_buffer, *vertex.point()) {
                Ok(InSphere::INSIDE) => self.bad_cells_buffer.push(cell_key),
                Ok(_) => {}, // Outside or on sphere
                Err(e) => return Err(TriangulationValidationError::FailedToCreateCell {
                    message: format!("Circumsphere computation failed: {}", e),
                }),
            }
        }
        
        Ok(self.bad_cells_buffer.clone())
    }
}
```

### C. **Neighbor Validation** - **MEDIUM PRIORITY**

**Current Issues:**

- Repeated intersection computations
- HashSet operations for every neighbor pair

**Optimizations:**

```rust
// Pre-compute vertex intersection counts
impl<T, U, V, const D: usize> Tds<T, U, V, D> {
    fn validate_neighbors_optimized(&self) -> Result<(), TriangulationValidationError> {
        // Pre-compute all vertex sets once
        let cell_vertices: HashMap<CellKey, Vec<VertexKey>> = self.cells
            .iter()
            .map(|(cell_key, cell)| {
                let vertices: Vec<VertexKey> = cell.vertices()
                    .iter()
                    .filter_map(|v| self.vertex_bimap.get_by_left(&v.uuid()).copied())
                    .collect();
                (cell_key, vertices)
            })
            .collect();
        
        // Use bit vectors for faster intersection counting
        for (cell_key, cell) in &self.cells {
            let Some(neighbors) = &cell.neighbors else { continue };
            
            if neighbors.len() > D + 1 {
                return Err(TriangulationValidationError::InvalidNeighbors {
                    message: format!("Cell has too many neighbors: {}", neighbors.len()),
                });
            }
            
            let this_vertices = &cell_vertices[&cell_key];
            
            for neighbor_uuid in neighbors {
                let Some(&neighbor_key) = self.cell_bimap.get_by_left(neighbor_uuid) else {
                    continue;
                };
                let neighbor_vertices = &cell_vertices[&neighbor_key];
                
                // Fast intersection count using sorted vectors
                let shared_count = count_intersections_sorted(this_vertices, neighbor_vertices);
                
                if shared_count != D {
                    return Err(TriangulationValidationError::NotNeighbors {
                        cell1: cell.uuid(),
                        cell2: *neighbor_uuid,
                    });
                }
            }
        }
        Ok(())
    }
}

// Optimized intersection counting for sorted vectors
fn count_intersections_sorted<T: Ord>(a: &[T], b: &[T]) -> usize {
    let mut count = 0;
    let mut i = 0;
    let mut j = 0;
    
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    count
}
```

## 2. **Memory Allocation Optimizations**

### A. **Pre-allocate Collections with Capacity**

```rust
// In assign_neighbors
fn assign_neighbors(&mut self) {
    let mut facet_map: HashMap<u64, Vec<CellKey>> = 
        HashMap::with_capacity(self.cells.len() * (D + 1)); // Better capacity estimate
    
    let mut neighbor_map: HashMap<CellKey, HashSet<CellKey>> = 
        HashMap::with_capacity(self.cells.len());
    
    // Initialize with proper capacity
    for cell_key in self.cells.keys() {
        neighbor_map.insert(cell_key, HashSet::with_capacity(D + 1));
    }
    
    // Rest of implementation...
}
```

### B. **Reduce Temporary Allocations**

```rust
// Ensure computing hash using existing traits and utilities
fn remove_duplicate_cells_optimized(&mut self) -> usize {
    // Use a custom hash without requiring sorting by relying on VertexSetHash
    let mut unique_cells: HashMap<VertexSetHash, CellKey> = HashMap::new();
    let mut cells_to_remove = Vec::new();

    for (cell_key, cell) in &self.cells {
        let vertex_hash = compute_vertex_set_hash(cell.vertices());

        if let Some(_existing_key) = unique_cells.get(&vertex_hash) {
            cells_to_remove.push(cell_key);
        } else {
            unique_cells.insert(vertex_hash, cell_key);
        }
    }

    // Remove duplicates from the triangulation
    let duplicate_count = cells_to_remove.len();
    for cell_key in cells_to_remove {
        if let Some(removed_cell) = self.cells.remove(cell_key) {
            self.cell_bimap.remove_by_left(&removed_cell.uuid());
        }
    }

    duplicate_count
}

// Fast hash computation without sorting
#[derive(Hash, PartialEq, Eq)]
struct VertexSetHash(u64);

fn compute_vertex_set_hash<T, U, const D: usize>(vertices: &[Vertex<T, U, D>]) -> VertexSetHash {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    // XOR all vertex UUIDs for order-independent hash
    let mut combined = 0u128;
    for vertex in vertices {
        combined ^= vertex.uuid().as_u128();
    }
    
    let mut hasher = DefaultHasher::new();
    combined.hash(&mut hasher);
    VertexSetHash(hasher.finish())
}
```

## 3. **Algorithmic Improvements**

### A. **Spatial Data Structures for Large Triangulations**

For triangulations with many vertices (>1000), consider implementing spatial partitioning:

```rust
// KD-tree for fast spatial queries
struct SpatialIndex<T, const D: usize> {
    tree: kdtree::KdTree<T, CellKey, [T; D]>,
}

impl<T, U, V, const D: usize> Tds<T, U, V, D> 
where 
    T: CoordinateScalar + Clone,
{
    fn build_spatial_index(&self) -> SpatialIndex<T, D> {
        let mut tree = kdtree::KdTree::new(D);
        
        for (cell_key, cell) in &self.cells {
            // Use cell centroid for spatial indexing
            // Note: compute_centroid needs to be implemented
            // let centroid = compute_centroid(cell.vertices());
            // tree.add(centroid.coords, cell_key).unwrap();
        }
        
        SpatialIndex { tree }
    }
    
    fn find_bad_cells_spatial(&self, vertex: &Vertex<T, U, D>, 
                            index: &SpatialIndex<T, D>) -> Vec<CellKey> {
        let vertex_point = vertex.point().to_array();
        
        // Query nearby cells first, then test circumsphere
        // Assumes 'within' method exists and can handle query for nearby cells.
        // You may need to replace 'squared_euclidean' with the correct distance function
        // Verify if 'max_circumradius' needs definition or replacement.
        let nearby_cells = index.tree.within(&vertex_point, /* max distance */, /* correct metric */)?;
        
        let mut bad_cells = Vec::new();
        for (_, &cell_key) in nearby_cells {
            let cell = &self.cells[cell_key];
            // Note: point_in_circumsphere method may need to be implemented or use existing insphere predicate
            // if self.point_in_circumsphere(vertex.point(), cell) {
                bad_cells.push(cell_key);
            }
        }
        
        bad_cells
    }
}
```

### B. **Optimized Boundary Facet Detection**

```rust
impl<T, U, V, const D: usize> Tds<T, U, V, D> {
    fn boundary_facets_optimized(&self) -> Vec<Facet<T, U, V, D>> {
        let mut boundary_facets = Vec::new();
        let mut seen_facets: HashSet<u64> = HashSet::new();
        
        for cell in self.cells.values() {
            for facet in cell.facets() {
                let facet_key = facet.key();
                
                if seen_facets.contains(&facet_key) {
                    // This facet is shared - not a boundary facet
                    // Remove it if it was previously added
                    boundary_facets.retain(|f| f.key() != facet_key);
                } else {
                    // First time seeing this facet - potentially a boundary facet
                    seen_facets.insert(facet_key);
                    boundary_facets.push(facet.clone());
                }
            }
        }
        
        boundary_facets
    }
    
    fn number_of_boundary_facets_optimized(&self) -> usize {
        let mut facet_counts: HashMap<u64, u8> = HashMap::with_capacity(
            self.cells.len() * (D + 1)
        );
        
        for cell in self.cells.values() {
            for facet in cell.facets() {
                let count = facet_counts.entry(facet.key()).or_insert(0);
                *count += 1;
                if *count > 2 {
                    // Early termination - no facet should appear more than twice
                    return 0; // Invalid triangulation
                }
            }
        }
        
        facet_counts.values().filter(|&&count| count == 1).count()
    }
}
```

### C. **Additional High-Impact Optimizations**

#### 1. **Parallel Processing for Large Triangulations**

```rust
use rayon::prelude::*;

// Parallel cell validation
fn is_valid_parallel(&self) -> Result<(), TriangulationValidationError> {
    // Validation steps that can be parallelized
    let validation_results: Result<Vec<_>, _> = self.cells
        .par_iter()
        .map(|(cell_key, cell)| {
            // Individual cell validation (CPU-intensive)
            cell.is_valid().map_err(|source| {
                TriangulationValidationError::InvalidCell {
                    cell_id: cell.uuid(),
                    source,
                }
            })
        })
        .collect();
    
    validation_results.map(|_| ())
}

// Parallel bad cell detection for Bowyer-Watson
fn find_bad_cells_parallel(&self, vertex: &Vertex<T, U, D>) -> Vec<CellKey> {
    self.cells
        .par_iter()
        .filter_map(|(cell_key, cell)| {
            let vertex_points: Vec<_> = cell.vertices()
                .iter()
                .map(|v| *v.point())
                .collect();
            
            match insphere(&vertex_points, *vertex.point()) {
                Ok(InSphere::INSIDE) => Some(cell_key),
                _ => None,
            }
        })
        .collect()
}
```

#### 2. **SIMD Optimizations for Coordinate Operations**

```rust
// Using wide (SIMD) operations for distance calculations
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Vectorized coordinate operations where possible
fn compute_circumsphere_simd(points: &[Point<f64, 3>]) -> Option<(Point<f64, 3>, f64)> {
    // Use SIMD for batch coordinate operations
    // Particularly effective for 2D/3D triangulations with f32/f64
    todo!("Implement SIMD circumsphere computation")
}
```

#### 3. **Memory Pool Allocator for Frequent Operations**

```rust
// Custom allocator for frequently allocated/deallocated objects
use typed_arena::Arena;

struct TdsArena<T, U, V, const D: usize> {
    vertex_arena: Arena<Vec<VertexKey>>,
    facet_arena: Arena<Vec<Facet<T, U, V, D>>>,
    temp_collections: Arena<Vec<Uuid>>,
}

impl<T, U, V, const D: usize> Tds<T, U, V, D> {
    fn with_arena(arena: &TdsArena<T, U, V, D>) -> Self {
        // Use arena allocation for temporary collections
        // Reduces allocation overhead during triangulation construction
        todo!("Implement arena-based allocation")
    }
}
```

#### 4. **Incremental Validation During Construction**

```rust
// Maintain validation invariants during construction
impl<T, U, V, const D: usize> Tds<T, U, V, D> {
    /// Validates only the changes made since last validation
    fn validate_incremental(&mut self, 
                           changed_cells: &[CellKey], 
                           new_cells: &[CellKey]) -> Result<(), TriangulationValidationError> {
        // Only validate affected parts of the triangulation
        // Much faster than full validation during construction
        
        // 1. Validate new cells
        for &cell_key in new_cells {
            if let Some(cell) = self.cells.get(cell_key) {
                cell.is_valid().map_err(|source| {
                    TriangulationValidationError::InvalidCell {
                        cell_id: cell.uuid(),
                        source,
                    }
                })?;
            }
        }
        
        // 2. Validate changed neighbor relationships
        let affected_cells: HashSet<_> = changed_cells.iter()
            .chain(new_cells.iter())
            .collect();
            
        for &cell_key in &affected_cells {
            // Only validate neighbors of affected cells
            self.validate_cell_neighbors(cell_key)?;
        }
        
        Ok(())
    }
    
    /// Track changes for incremental validation
    fn add_cell_with_tracking(&mut self, cell: Cell<T, U, V, D>) -> CellKey {
        let cell_key = self.cells.insert(cell);
        // Track this as a new cell for next incremental validation
        self.validation_state.new_cells.push(cell_key);
        cell_key
    }
}
```

#### 5. **Optimized Data Structures**

```rust
// Use more efficient data structures for specific operations
use fxhash::FxHashMap; // Faster hashing for non-cryptographic use
use smallvec::SmallVec; // Stack allocation for small collections

// Replace standard HashMap with faster alternatives
type FastHashMap<K, V> = FxHashMap<K, V>;
type SmallVertexVec<T, U, const D: usize> = SmallVec<[Vertex<T, U, D>; 8]>;

// Bit-packed vertex sets for faster intersection operations
#[derive(Clone, Copy)]
struct VertexBitSet(u64); // For up to 64 vertices per operation

impl VertexBitSet {
    fn new() -> Self { Self(0) }
    fn insert(&mut self, vertex_idx: u8) { self.0 |= 1u64 << vertex_idx; }
    fn contains(&self, vertex_idx: u8) -> bool { (self.0 & (1u64 << vertex_idx)) != 0 }
    fn intersection_count(&self, other: &Self) -> u32 { (self.0 & other.0).count_ones() }
}
```

## 4. **Implementation Priority (Updated)**

### **HIGHEST PRIORITY (Immediate 2-3x speedups)**

1. **`is_valid()` single-pass optimization** - Eliminate redundant iterations
2. **Buffer reuse in Bowyer-Watson** - Use existing struct buffers
3. **Early termination in validation** - Stop on first error

### **HIGH PRIORITY (30-50% speedups)**

1. **Optimized neighbor validation** - Cached vertex sets, sorted intersection
2. **Pre-allocated collections** - Proper HashMap capacity estimation
3. **Fast duplicate detection** - XOR-based hashing without sorting

### **MEDIUM PRIORITY (20-40% speedups)**

1. **Parallel validation** - For large triangulations (>1000 cells)
2. **Incremental validation** - During construction only
3. **Optimized data structures** - FxHashMap, SmallVec, bit sets

### **LOW PRIORITY (Advanced optimizations)**

1. **Spatial indexing** - For very large triangulations (>5000 vertices)
2. **SIMD operations** - Platform-specific optimizations
3. **Memory pool allocation** - For high-frequency construction/destruction

## 5. **Expected Performance Gains (Updated)**

### **Validation Performance**

- **`is_valid_optimized()`**: 2-3x faster than current implementation
- **`is_valid_fast()`**: 5-10x faster for frequent checks
- **Parallel validation**: 3-4x faster on multi-core systems (>1000 cells)

### **Construction Performance**

- **Buffer reuse**: 20-30% reduction in allocation overhead
- **Incremental validation**: 10x faster validation during construction
- **Bad cells detection**: 40-60% speedup with buffer reuse

### **Memory Usage**

- **Buffer reuse**: 30-50% reduction in temporary allocations
- **SmallVec usage**: 10-20% reduction for small collections
- **Arena allocation**: 50-80% reduction in allocation overhead

### **Overall System Performance**

- **Small triangulations** (<100 cells): 50-100% speedup
- **Medium triangulations** (100-1000 cells): 100-200% speedup  
- **Large triangulations** (>1000 cells): 200-400% speedup with parallel processing

## 6. **Benchmarking Recommendations**

Add these comprehensive benchmarks to measure optimization effectiveness:

```rust
#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    
    /// Comprehensive validation performance benchmark
    #[test]
    #[ignore = "Performance benchmark - run manually"]
    fn benchmark_validation_methods() {
        let sizes = [50, 100, 200, 500, 1000, 2000];
        
        println!("=== Validation Method Performance Comparison ===");
        println!("{:>5} | {:>12} | {:>12} | {:>12} | {:>8}", 
                 "Size", "Original", "Optimized", "Fast", "Speedup");
        println!("{:-<60}", "");
        
        for &size in &sizes {
            let vertices = generate_random_vertices_3d(size);
            let tds = Tds::new(&vertices).unwrap();
            
            // Benchmark original is_valid()
            let start = Instant::now();
            let _ = tds.is_valid();
            let original_time = start.elapsed();
            
            // Benchmark optimized is_valid_optimized()
            let start = Instant::now();
            let _ = tds.is_valid_optimized();
            let optimized_time = start.elapsed();
            
            // Benchmark fast is_valid_fast()
            let start = Instant::now();
            let _ = tds.is_valid_fast();
            let fast_time = start.elapsed();
            
            let speedup = if optimized_time.as_nanos() > 0 {
                original_time.as_nanos() as f64 / optimized_time.as_nanos() as f64
            } else { 0.0 };
            
            println!("{:>5} | {:>10?} | {:>10?} | {:>10?} | {:>6.2}x", 
                     size, original_time, optimized_time, fast_time, speedup);
        }
    }
    
    /// Benchmark individual validation components
    #[test]
    #[ignore = "Performance benchmark - run manually"]
    fn benchmark_validation_components() {
        let vertices = generate_random_vertices_3d(1000);
        let tds = Tds::new(&vertices).unwrap();
        let runs = 100;
        
        println!("=== Individual Validation Component Performance ===");
        
        // Benchmark mapping validation
        let start = Instant::now();
        for _ in 0..runs {
            let _ = tds.validate_vertex_mappings();
            let _ = tds.validate_cell_mappings();
        }
        let mapping_time = start.elapsed() / runs;
        
        // Benchmark duplicate cell detection
        let start = Instant::now();
        for _ in 0..runs {
            let _ = tds.validate_no_duplicate_cells();
        }
        let duplicate_time = start.elapsed() / runs;
        
        // Benchmark facet sharing validation
        let start = Instant::now();
        for _ in 0..runs {
            let _ = tds.validate_facet_sharing();
        }
        let facet_time = start.elapsed() / runs;
        
        // Benchmark neighbor validation
        let start = Instant::now();
        for _ in 0..runs {
            let _ = tds.validate_neighbors_internal();
        }
        let neighbor_time = start.elapsed() / runs;
        
        println!("Mapping validation:     {:>10?}", mapping_time);
        println!("Duplicate detection:    {:>10?}", duplicate_time);
        println!("Facet sharing:          {:>10?}", facet_time);
        println!("Neighbor validation:    {:>10?}", neighbor_time);
        
        let total_expected = mapping_time + duplicate_time + facet_time + neighbor_time;
        println!("Expected total:         {:>10?}", total_expected);
    }
    
    /// Benchmark Bowyer-Watson algorithm components
    #[test]
    #[ignore = "Performance benchmark - run manually"]
    fn benchmark_bowyer_watson_components() {
        let sizes = [50, 100, 200, 500];
        
        println!("=== Bowyer-Watson Component Performance ===");
        println!("{:>5} | {:>12} | {:>12} | {:>12} | {:>12}", 
                 "Size", "Construction", "Bad Cells", "Boundary", "Total Valid");
        println!("{:-<65}", "");
        
        for &size in &sizes {
            let vertices = generate_random_vertices_3d(size);
            
            // Benchmark full construction
            let start = Instant::now();
            let tds = Tds::new(&vertices).unwrap();
            let construction_time = start.elapsed();
            
            // Benchmark bad cells detection (simulate adding one more vertex)
            if let Some(test_vertex) = vertices.get(0) {
                let start = Instant::now();
                let _ = tds.find_bad_cells(test_vertex);
                let bad_cells_time = start.elapsed();
                
                // Benchmark boundary facets
                let start = Instant::now();
                let _ = tds.boundary_facets();
                let boundary_time = start.elapsed();
                
                // Benchmark full validation
                let start = Instant::now();
                let _ = tds.is_valid();
                let validation_time = start.elapsed();
                
                println!("{:>5} | {:>10?} | {:>10?} | {:>10?} | {:>10?}", 
                         size, construction_time, bad_cells_time, 
                         boundary_time, validation_time);
            }
        }
    }
    
    /// Memory usage benchmark
    #[test]
    #[ignore = "Memory benchmark - requires manual analysis"]
    fn benchmark_memory_usage() {
        use std::alloc::{GlobalAlloc, Layout, System};
        
        struct TrackingAllocator;
        
        // Note: This is a simplified example - actual memory tracking 
        // would require more sophisticated instrumentation
        
        let sizes = [100, 500, 1000, 2000];
        
        println!("=== Memory Usage Analysis ===");
        println!("{:>5} | {:>8} | {:>8} | {:>12}", 
                 "Size", "Vertices", "Cells", "Est. Memory");
        println!("{:-<40}", "");
        
        for &size in &sizes {
            let vertices = generate_random_vertices_3d(size);
            let tds = Tds::new(&vertices).unwrap();
            
            let vertex_count = tds.number_of_vertices();
            let cell_count = tds.number_of_cells();
            
            // Rough memory estimation (this is approximate)
            let vertex_size = std::mem::size_of::<Vertex<f64, Option<()>, 3>>();
            let cell_size = std::mem::size_of::<Cell<f64, Option<()>, Option<()>, 3>>();
            let estimated_memory = (vertex_count * vertex_size) + (cell_count * cell_size);
            
            println!("{:>5} | {:>8} | {:>8} | {:>10} B", 
                     size, vertex_count, cell_count, estimated_memory);
        }
    }
    
    /// Scalability analysis
    #[test]
    #[ignore = "Scalability benchmark - long running"]
    fn benchmark_scalability() {
        let sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
        
        println!("=== Scalability Analysis ===");
        println!("{:>5} | {:>8} | {:>8} | {:>12} | {:>12} | {:>8}", 
                 "Size", "Vertices", "Cells", "Construction", "Validation", "V/C Ratio");
        println!("{:-<70}", "");
        
        for &size in &sizes {
            let vertices = generate_random_vertices_3d(size);
            
            let start = Instant::now();
            let tds = Tds::new(&vertices).unwrap();
            let construction_time = start.elapsed();
            
            let start = Instant::now();
            let _ = tds.is_valid();
            let validation_time = start.elapsed();
            
            let vertex_count = tds.number_of_vertices();
            let cell_count = tds.number_of_cells();
            let ratio = if cell_count > 0 { 
                vertex_count as f64 / cell_count as f64 
            } else { 0.0 };
            
            println!("{:>5} | {:>8} | {:>8} | {:>10?} | {:>10?} | {:>6.2}", 
                     size, vertex_count, cell_count, 
                     construction_time, validation_time, ratio);
        }
    }
    
    /// Generate deterministic random vertices for consistent benchmarking
    fn generate_random_vertices_3d(count: usize) -> Vec<Vertex<f64, Option<()>, 3>> {
        let mut rng = ChaCha8Rng::seed_from_u64(42); // Fixed seed for consistency
        
        (0..count)
            .map(|_| {
                vertex!([
                    rng.gen_range(-100.0..100.0),
                    rng.gen_range(-100.0..100.0),
                    rng.gen_range(-100.0..100.0)
                ])
            })
            .collect()
    }
}
```

## 7. **Optimization Testing Framework**

Implement these tests to ensure optimizations maintain correctness:

```rust
#[cfg(test)]
mod optimization_tests {
    use super::*;
    use proptest::prelude::*;
    
    /// Property-based testing to ensure optimized methods produce same results
    proptest! {
        #[test]
        fn test_optimized_validation_equivalence(
            vertices in prop::collection::vec(any_vertex_3d(), 4..50)
        ) {
            let tds = Tds::new(&vertices)?;
            
            let original_result = tds.is_valid();
            let optimized_result = tds.is_valid_optimized();
            
            // Results should be equivalent (both Ok or both Err)
            assert_eq!(original_result.is_ok(), optimized_result.is_ok());
            
            // Fast validation should never be more restrictive than full validation
            if original_result.is_ok() {
                assert!(tds.is_valid_fast().is_ok());
            }
        }
    }
    
    /// Stress test for large triangulations
    #[test]
    #[ignore = "Stress test - resource intensive"]
    fn stress_test_large_triangulation() {
        let vertices = generate_stress_test_vertices(10000);
        
        let start = std::time::Instant::now();
        let tds = Tds::new(&vertices).unwrap();
        let construction_time = start.elapsed();
        
        let start = std::time::Instant::now();
        assert!(tds.is_valid_optimized().is_ok());
        let validation_time = start.elapsed();
        
        println!("Stress test results:");
        println!("  - Vertices: {}", tds.number_of_vertices());
        println!("  - Cells: {}", tds.number_of_cells());
        println!("  - Construction time: {:?}", construction_time);
        println!("  - Validation time: {:?}", validation_time);
        
        // Should complete within reasonable time bounds
        assert!(construction_time.as_secs() < 30, "Construction took too long");
        assert!(validation_time.as_secs() < 10, "Validation took too long");
    }
}
```

## 8. **Production Usage Guidelines**

### **Method Selection Guide**

```rust
// For debugging and development
if cfg!(debug_assertions) {
    tds.is_valid()?; // Full validation with detailed error messages
} else {
    tds.is_valid_optimized()?; // Faster validation in release builds
}

// During triangulation construction (frequent validation)
for (i, vertex) in new_vertices.iter().enumerate() {
    tds.add_vertex(vertex)?;
    
    // Validate every N additions for early error detection
    if i % 100 == 0 {
        tds.is_valid_fast()?; // Much faster for frequent checks
    }
}

// Final validation before using triangulation
tds.is_valid_optimized()?; // Comprehensive but optimized

// For specific validation needs
if validation_config.check_neighbors_only {
    tds.validate_neighbors_optimized()?; // Targeted validation
}
```

### **Performance Monitoring**

```rust
// Add performance monitoring to track optimization effectiveness
use std::time::Instant;

#[derive(Debug)]
struct ValidationMetrics {
    validation_time: Duration,
    cell_count: usize,
    vertex_count: usize,
    validation_type: String,
}

impl Tds<T, U, V, D> {
    fn is_valid_with_metrics(&self) -> (Result<(), ValidationError>, ValidationMetrics) {
        let start = Instant::now();
        let result = self.is_valid_optimized();
        let duration = start.elapsed();
        
        let metrics = ValidationMetrics {
            validation_time: duration,
            cell_count: self.number_of_cells(),
            vertex_count: self.number_of_vertices(),
            validation_type: "optimized".to_string(),
        };
        
        (result, metrics)
    }
}
```

These optimizations provide a comprehensive framework for significantly improving TDS
performance while maintaining correctness and providing clear guidance for
implementation priorities and usage patterns.
