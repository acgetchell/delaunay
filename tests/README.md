# Integration Tests

This directory contains integration tests for the delaunay library, focusing on comprehensive testing scenarios, debugging utilities,
regression testing, and performance analysis.

## Test Categories

### 🎲 Property-Based Testing

#### [`proptest_predicates.rs`](./proptest_predicates.rs)

Property-based tests for geometric predicates using proptest to verify mathematical properties that must hold universally.

**Test Coverage:**

- **Orientation Properties**:
  - Sign flip under vertex swap (orientation reversal)
  - Cyclic permutation invariance (2D/3D)
  - Degenerate case consistency
- **Insphere Properties**:
  - Simplex vertices on boundary (defining vertices should be on/near circumsphere)
  - Scaling property (far points are OUTSIDE)
  - Cross-dimensional consistency (2D-4D)
- **Cross-Predicate Consistency**:
  - Degenerate orientation implies potential insphere failures
  - Robustness under near-degenerate configurations

**Run with:** `cargo test --test proptest_predicates` or included in `just test`

#### [`proptest_point.rs`](./proptest_point.rs)

Property-based tests for Point data structures verifying fundamental properties.

**Test Coverage:**

- **Equality and Hashing**:
  - Hash consistency (equal points have equal hashes)
  - Equality reflexivity, symmetry, transitivity
  - HashMap key usage correctness
- **Coordinate Operations**:
  - Coordinate extraction roundtrips
  - Into<[T; D]> conversion consistency
  - get() method correctness
- **Serialization**:
  - Serde roundtrip preservation
  - Cross-dimensional serialization
- **NaN Handling**:
  - Custom equality semantics (NaN == NaN)
  - Hash consistency with NaN coordinates
  - HashMap key usage with NaN
- **Validation**:
  - Finite coordinates validate successfully
  - Infinite/NaN coordinates fail validation
- **Ordering**:
  - Consistency with equality
  - Antisymmetry and transitivity

**Run with:** `cargo test --test proptest_point` or included in `just test`

#### [`proptest_tds.rs`](./proptest_tds.rs)

Property-based tests for Tds (Triangulation Data Structure) combinatorial/topological invariants.

**Architectural Layer:** Pure combinatorial structure (no geometric predicates)

**Test Coverage:**

- **Vertex Mappings**: UUID↔key consistency for all vertices
- **Cell Mappings**: UUID↔key consistency for all cells
- **No Duplicate Cells**: No two cells share the same vertex set
- **Cell Validity**: Each cell has correct vertex count and passes internal consistency checks
- **Cell Vertex Count**: Maximal cells have exactly D+1 vertices (fundamental Tds constraint)
- **Facet Sharing**: Each facet is shared by at most 2 cells
- **Neighbor Consistency**: Neighbor relationships are mutual and reference shared facets
  - Neighbor symmetry (if A neighbors B, then B neighbors A)
  - Neighbor index semantics (correct facet-based indexing)
- **Vertex-Cell Incidence**: All cell vertices exist in the TDS
- **Vertex Count Consistency**: Vertex key count matches reported vertex count
- **Dimension Consistency**: Reported dimension matches actual structure

**Dimensions Tested:** 2D-5D

**Run with:** `cargo test --test proptest_tds` or included in `just test`

#### [`proptest_orientation.rs`](./proptest_orientation.rs)

Property-based tests focused on coherent orientation invariants in the TDS layer.

**Test Coverage:**

- **Construction coherence**: successfully-built triangulations are coherently oriented
- **Tamper detection**: cell-order tampering is detected as `OrientationViolation`
- **Incremental coherence**: orientation remains coherent after each successful insertion

**Dimensions Tested:** 2D-5D (4D/5D marked slow/ignored in test-integration profile)

**Run with:** `cargo test --test proptest_orientation` or included in `just test`

#### [`proptest_triangulation.rs`](./proptest_triangulation.rs)

Property-based tests for Triangulation layer invariants (generic geometric layer with kernel).

**Architectural Layer:** Generic geometric operations with kernel (delegates topology to Tds)

**Test Coverage:**

- **Geometric Quality Metrics**:
  - Radius ratio bounds (R/r ≥ D for D-dimensional simplex)
  - Radius ratio scaling and translation invariance
  - Normalized volume invariance properties
  - Quality metric consistency (degeneracy detection)
  - Quality degradation under deformation
- **Future Tests**:
  - Facet iteration consistency
  - Boundary facet detection
  - Topology repair (fix_invalid_facet_sharing)
  - Kernel consistency validation

**Note:** Tests use `DelaunayTriangulation` for construction (most convenient way to obtain valid triangulations).
The properties tested are generic Triangulation-layer concerns applicable to any triangulation with a kernel.

**Dimensions Tested:** 2D-5D

**Run with:** `cargo test --test proptest_triangulation` or included in `just test`

#### [`proptest_delaunay_triangulation.rs`](./proptest_delaunay_triangulation.rs)

Property-based tests for `DelaunayTriangulation` invariants (all Delaunay-specific properties).

**Architectural Layer:** Delaunay-specific operations and the empty circumsphere property

**Test Coverage:**

- **Structural Invariants (Fast)**:
  - Incremental insertion maintains validity after each insertion
  - Duplicate coordinate rejection (geometric duplicate detection at insertion time)
- **Delaunay Property (Fast O(N) via Flip Predicates)**:
  - Empty circumsphere condition - No vertex lies strictly inside any cell's circumsphere (2D-5D) ✅ **PASSING**
  - Insertion-order robustness - Levels 1–3 validity across insertion orders (2D-5D)
  - Duplicate cloud integration - Full pipeline with messy real-world inputs (2D-5D: duplicates + near-duplicates) ✅ **PASSING**

**Status:** ✅ All Delaunay property tests are enabled and passing (as of v0.7.0+)

**Implementation:** Bistellar flips (k=2 facets, k=3 ridges) with automatic Delaunay repair:

- Fast O(N) flip-based validation provides 40-100x speedup over brute-force
- Automatic repair runs after insertion/removal via `DelaunayRepairPolicy`
- Inverse edge/triangle queues for 4D/5D repair
- See `src/core/algorithms/flips.rs` for implementation

**Remaining ignored tests** (separate from Issue #120):

- `prop_incremental_insertion_maintains_validity_4d/5d` - RidgeLinkNotManifold topology issues (see new plan)
- `prop_duplicate_coordinates_rejected_3d/4d/5d` - Slow tests (>60s), ignored for performance

**Dimensions Tested:** 2D-5D

**Run with:** `cargo test --test proptest_delaunay_triangulation` or included in `just test`

#### [`proptest_cell.rs`](./proptest_cell.rs)

Property-based tests for Cell data structure verifying cell-level invariants and topological consistency.

**Test Coverage:**

- **Orientation Consistency**: Cell vertex ordering and orientation preservation
- **Neighbor Linkage**: Neighbor references validity and symmetry
- **Facet Completeness**: All facets properly defined and accessible
- **Vertex References**: All vertex keys are valid and consistent

**Run with:** `cargo test --release --test proptest_cell`

#### [`proptest_convex_hull.rs`](./proptest_convex_hull.rs)

Property-based tests for convex hull computation verifying hull properties and integration with triangulation.

**Test Coverage:**

- **Hull Vertex Extremeness**: Hull vertices are extreme points of the point set
- **Hull Facet Consistency**: All hull facets are valid and properly oriented
- **Boundary Subset Property**: Hull is a subset of triangulation boundary
- **Dimension Consistency**: Hull dimension matches point set dimension

**Run with:** `cargo test --release --test proptest_convex_hull`

#### [`proptest_facet.rs`](./proptest_facet.rs)

Property-based tests for Facet operations verifying facet adjacency and orientation across neighboring cells.

**Test Coverage:**

- **Mutual Neighbor References**: If cell A has neighbor B via facet F, then B has A as neighbor
- **Co-facet Consistency**: Shared facets reference same vertices (possibly different order)
- **Orientation Alternation**: Adjacent cells have opposite facet orientations
- **Facet Key Validity**: All facet identifiers are valid and retrievable

**Run with:** `cargo test --release --test proptest_facet`

#### [`proptest_geometry.rs`](./proptest_geometry.rs)

Property-based tests for geometric utilities and predicates.

**Test Coverage:**

- **Orientation Antisymmetry**: Swapping vertices reverses orientation
- **Insphere/Outsphere Consistency**: Points are consistently classified relative to circumsphere
- **Circumsphere Invariants**: Simplex vertices lie on their circumsphere
- **Geometric Utility Correctness**: Helper functions produce valid results

**Run with:** `cargo test --release --test proptest_geometry`

#### [`proptest_serialization.rs`](./proptest_serialization.rs)

Property-based tests for serialization and deserialization verifying data preservation via randomized structures.

**Test Coverage:**

- **Round-trip Equality**: Serialize → deserialize preserves structure and data
- **Neighbor Graph Preservation**: Cell neighbor relationships survive round-trip
- **Vertex Data Integrity**: Vertex coordinates and associated data are preserved
- **Cell Data Integrity**: Cell-associated data is preserved
- **Cross-dimensional Serialization**: Works correctly for all supported dimensions

**Run with:** `cargo test --release --test proptest_serialization`

**Property Testing Notes:**

- Property tests use randomized inputs to discover edge cases
- Tests may take longer than unit tests due to multiple iterations
- Failures include shrunk minimal failing cases for debugging
- Configure test cases via `PROPTEST_CASES=N` environment variable (default: 256)
- Reproduce failures using `PROPTEST_SEED=<seed>` from test output
- For deterministic ordering when debugging, use `--test-threads=1`
- Always prefer `--release` mode for representative performance

**About `.proptest-regressions` Files:**

Proptest automatically captures minimal failing test cases in `.proptest-regressions` files located in the
`tests/` directory. These files serve as regression test suites:

- **Purpose**: Minimal failing cases that are re-run first to guard against regressions
- **Version Control**: Always commit these files so CI and all developers validate past failures
- **Automatic Updates**: Tests automatically update these files when new failures are discovered
- **Do Not Hand-Edit**: Let proptest manage these files; manual edits may break the format
- **Reproduction**: To debug a failure, copy the seed from test output:

  ```bash
  PROPTEST_SEED=12345 cargo test --release --test proptest_triangulation -- --nocapture
  ```

- **Performance Note**: Regression cases run before random cases; many entries can slow tests
- **Filtering**: Use test filters to narrow scope when iterating on specific properties
- **Maintenance**: It's acceptable to prune obsolete entries in follow-up PRs (keep diffs focused)

**Current Regression Files:**

- `proptest_convex_hull.proptest-regressions`
- `proptest_serialization.proptest-regressions`

### 🔧 Debugging and Analysis Tools

#### [`circumsphere_debug_tools.rs`](./circumsphere_debug_tools.rs)

Interactive debugging and testing tools for circumsphere calculations. Demonstrates and compares three methods for testing whether
a point lies inside the circumsphere of a simplex in 2D, 3D, and 4D.

**Key Features:**

- Comprehensive circumsphere method comparison
- Step-by-step matrix analysis
- Interactive testing across dimensions
- Geometric property analysis
- Orientation impact demonstration

**Usage:**

```bash
# Run specific debug test functions with verbose output
cargo test --test circumsphere_debug_tools test_2d_circumsphere_debug -- --nocapture
cargo test --test circumsphere_debug_tools test_3d_circumsphere_debug -- --nocapture
cargo test --test circumsphere_debug_tools test_all_debug -- --nocapture

# Run all debug tests at once (recommended)
just test-debug
```

**Available Test Functions:**

- `test_2d_circumsphere_debug` - 2D triangle circumsphere testing
- `test_3d_circumsphere_debug` - 3D tetrahedron circumsphere testing  
- `test_4d_circumsphere_debug` - 4D simplex circumsphere testing
- `test_3d_matrix_analysis_debug` - Step-by-step matrix method analysis
- `test_compare_methods_debug` - Cross-dimensional method comparison
- `test_all_debug` - Complete comprehensive test suite

[View source](./circumsphere_debug_tools.rs)

### 🧪 Algorithm Integration Testing

#### [`tds_basic_integration.rs`](./tds_basic_integration.rs)

Fundamental integration tests for triangulation data structure (TDS) operations, verifying core functionality with simple, well-understood geometries.

**Test Coverage:**

- **TDS Creation**: Validates basic TDS construction with various vertex configurations
  - Single cell (minimal 3D simplex/tetrahedron)
  - Multiple adjacent cells sharing facets
- **Neighbor Assignment**: Verifies correct neighbor connectivity between cells
  - Initial simplex neighbor initialization
  - Shared facet neighbor relationships
- **Boundary Analysis**: Tests boundary facet computation
  - Single cell boundary identification (4 facets for tetrahedron)
  - Multi-cell boundary detection (shared facets excluded)
- **Basic Validation**: Establishes baseline correctness for simple geometries

**Test Functions:**

- `test_tds_creates_one_cell` - Single tetrahedron creation
- `test_tds_creates_two_cells` - Two adjacent tetrahedra
- `test_initial_simplex_has_neighbors` - Neighbor assignment validation
- `test_two_cells_share_facet` - Shared facet connectivity

**Run with:** `cargo test --test tds_basic_integration` or `just test-release`

**Purpose:** These tests establish foundational TDS behavior. For complex scenarios involving algorithms like Bowyer-Watson or convex hull
operations, see other integration tests.

#### [`test_insertion_algorithm_utils.rs`](./test_insertion_algorithm_utils.rs)

Integration tests for insertion algorithm utility types (`InsertionBuffers` and `InsertionStatistics`) that support vertex insertion algorithms.

**Test Coverage:**

- **InsertionBuffers** (9 tests):
  - Buffer creation and initialization (new, default, with_capacity)
  - Buffer management (clear_all, prepare methods)
  - Vec compatibility methods for legacy API support
  - FacetView conversion for boundary analysis
- **InsertionStatistics** (4 tests):
  - Statistics initialization and recording
  - Vertex insertion tracking
  - Fallback strategy usage tracking
  - Rate calculation methods (fallback_usage_rate)

**Run with:** `cargo test --test test_insertion_algorithm_utils --release`

**Purpose:** These tests ensure the utility types used by insertion algorithms (Bowyer-Watson, robust variants) work correctly
for performance optimization (buffer reuse) and algorithm instrumentation (statistics tracking).

#### [`convex_hull_bowyer_watson_integration.rs`](./convex_hull_bowyer_watson_integration.rs)

Integration tests for convex hull algorithms with Bowyer-Watson triangulation, focusing on the interaction between hull computation and triangulation construction.

**Test Coverage:**

- Hull extension execution and validation
- Cache behavior and reset operations
- Multiple hull extension scenarios
- Triangulation validity after hull operations
- Mixed insertion strategy testing

**Run with:** `cargo test --test convex_hull_bowyer_watson_integration` or `just test-release`

#### [`robust_predicates_comparison.rs`](./robust_predicates_comparison.rs)

Comparative testing between robust and standard geometric predicates, focusing on numerical accuracy and edge case handling.

**Test Scenarios:**

- Cocircular and nearly coplanar points
- High precision coordinate handling
- Extreme aspect ratio configurations
- Vertex insertion robustness analysis
- Performance cost-benefit analysis

**Run with:** `cargo test --test robust_predicates_comparison` or `just test-release`

#### [`robust_predicates_showcase.rs`](./robust_predicates_showcase.rs)

Demonstration and stress testing of robust geometric predicates with focus on numerical edge cases and degenerate configurations.

**Features:**

- Degenerate failure recovery demonstrations
- Tolerance limit stress testing
- Real-world triangulation scenarios
- Performance impact analysis

**Run with:** `cargo test --test robust_predicates_showcase` or `just test-release`

#### [`serialization_vertex_preservation.rs`](./serialization_vertex_preservation.rs)

Integration tests for serialization ensuring vertex identifiers and associated data are preserved across serialize/deserialize cycles.

**Test Coverage:**

- **Vertex UUID Preservation**: Vertex identifiers remain stable across serialization
- **Coordinate Preservation**: Exact coordinate values are preserved
- **Vertex Data Preservation**: Associated vertex data survives round-trip
- **Cell References**: Cell-to-vertex references remain valid after deserialization

**Run with:** `cargo test --release --test serialization_vertex_preservation`

#### [`storage_backend_compatibility.rs`](./storage_backend_compatibility.rs)

Integration tests verifying equivalence of triangulation behavior across different storage backends (e.g., SlotMap vs DenseSlotMap).

**Test Coverage:**

- **Behavioral Equivalence**: Triangulation operations produce identical results across backends
- **Data Structure Integrity**: Cell and vertex relationships consistent regardless of backend
- **API Compatibility**: All public APIs work consistently across backends
- **Performance Characteristics**: Backend-specific performance trade-offs are documented

**Run with:** `cargo test --release --test storage_backend_compatibility`

**Note**: If storage backends are feature-gated, specify the feature:

```bash
cargo test --release --features <backend_feature> --test storage_backend_compatibility
```

#### [`integration_robust_bowyer_watson.rs`](./integration_robust_bowyer_watson.rs)

End-to-end integration tests for `RobustBowyerWatson` algorithm verifying robust insertion behavior across dimensions 2D-5D.

**Test Coverage:**

- **Large Random Point Sets**: Insertion of 50-100 random points with TDS validity maintained throughout
- **Exterior Vertex Insertion**: Hull extension scenarios with vertices inserted along each axis
- **Clustered Point Patterns**: Mixed clustered (near-origin) and scattered point distributions
- **Algorithm Reset and Reuse**: Statistics tracking and algorithm state management
- **Cross-dimensional Testing**: Generated tests for 2D, 3D, 4D, and 5D using macro-based approach

**Run with:** `cargo test --test integration_robust_bowyer_watson` or `just test-release`

**Purpose:** Verifies that the robust insertion algorithm handles real-world point distributions while maintaining TDS validity and
geometric correctness across varying dimensions and insertion patterns.

#### [`test_insertion_algorithm_trait.rs`](./test_insertion_algorithm_trait.rs)

Integration tests for public API methods of `InsertionAlgorithm` trait and supporting types.

**Test Coverage:**

- **InsertionBuffers Public API** (14 tests):
  - Buffer creation methods (new, default, with_capacity)
  - Buffer management (clear_all, prepare methods)
  - Vec compatibility helpers (bad_cells_as_vec, set_bad_cells_from_vec)
  - FacetView conversion (boundary_facets_as_views, visible_facets_as_views)
  - Accessor methods for all buffer types
- **Error Type Conversions**:
  - InsertionError display formatting
  - BadCellsError conversion and context
  - TriangulationValidationError integration

**Run with:** `cargo test --test test_insertion_algorithm_trait --release`

**Purpose:** Verifies public accessor methods and API contracts. Complements unit tests in source (which use private field access) by
testing only through public interfaces.

#### [`test_facet_cache_integration.rs`](./test_facet_cache_integration.rs)

Integration tests for facet cache behavior under concurrent access, TDS modifications, and algorithmic usage patterns.

**Test Coverage:**

- **Concurrent Cache Access**: Multi-threaded cache building and querying during vertex insertions
- **Cache Invalidation**: Cache rebuilding after TDS generation changes
- **RCU Contention**: Multiple threads simultaneously building cache with Read-Copy-Update mechanism
- **Generation Tracking**: Verification of cache staleness detection and retry loops
- **Algorithm Integration**: Cache behavior through `IncrementalBowyerWatson` operations

**Run with:** `cargo test --test test_facet_cache_integration` or `just test-release`

**Purpose:** Exercises complex cache synchronization code paths (retry loops, RCU, generation tracking) that are difficult to trigger
via unit tests. Validates thread-safe cache operations during real algorithm usage.

#### [`test_geometry_util.rs`](./test_geometry_util.rs)

Integration tests for geometry utility functions focusing on error paths, edge cases, and multi-dimensional behavior.

**Test Coverage:**

- **Random Point Generation Errors**:
  - Invalid ranges (min >= max)
  - Zero point counts for grid generation
  - Impossible Poisson spacing constraints
  - Reproducibility with seeded generation
- **Circumcenter/Circumradius Errors**:
  - Empty point sets
  - Invalid simplex dimensions (wrong vertex count)
  - Degenerate simplices (collinear, coincident points)
- **Simplex Volume Edge Cases**:
  - Multi-dimensional degenerate cases (1D-2D)
  - Coordinate conversion failures
  - Invalid simplex configurations

**Run with:** `cargo test --test test_geometry_util` or `just test-release`

**Purpose:** Tests error handling and edge cases for geometric utilities that complement doctest coverage, ensuring robust behavior with
invalid inputs and degenerate configurations.

#### [`test_robust_fallbacks.rs`](./test_robust_fallbacks.rs)

Configuration API tests for `RobustBowyerWatson` algorithm, documenting constructor variants and predicate configuration presets.

**Test Coverage:**

- **Constructor Variants**:
  - `new()` - Default constructor with general_triangulation config
  - `with_config()` - Custom configuration with various presets
  - `for_degenerate_cases()` - Degenerate-optimized constructor
- **Configuration Presets**:
  - `general_triangulation` - Balanced performance/robustness
  - `high_precision` - Maximum numerical accuracy
  - `degenerate_robust` - Optimized for near-degenerate cases
- **Multi-dimensional Configuration**: Testing presets across 2D-4D
- **Custom Tolerance Settings**: User-defined tolerance configurations

**Run with:** `cargo test --test test_robust_fallbacks` or `just test-release`

**Purpose:** Serves as API documentation and contract validation for configuration system. Minimal coverage impact (implementation already
covered) but ensures public API contracts work correctly.

#### [`test_tds_edge_cases.rs`](./test_tds_edge_cases.rs)

Edge case integration tests for Triangulation Data Structure (TDS) operations, focusing on removal operations and topology consistency.

**Test Coverage:**

- **Cell Removal Operations**:
  - Single cell removal (remove_cell_by_key)
  - Multiple cell removal (remove_cells_by_keys)
  - Nonexistent cell handling
  - Partial removal with invalid keys
- **Duplicate Cell Removal**:
  - Detection of duplicate cells
  - Clean TDS validation (no duplicates)
- **Neighbor Clearing**:
  - Clear all neighbor relationships
  - Neighbor reassignment after clearing
  - Topology validation after operations

**Run with:** `cargo test --test test_tds_edge_cases` or `just test-release`

**Purpose:** Tests TDS removal operations and topology maintenance that are not covered by basic integration tests. Complements
`tds_basic_integration.rs` which focuses on construction and validation.

#### [`test_convex_hull_error_paths.rs`](./test_convex_hull_error_paths.rs)

Error path tests for ConvexHull module targeting uncovered error conditions and edge cases.

**Test Coverage:**

- **Construction Errors**:
  - InsufficientData - Empty triangulations (0 vertices, 0 cells)
  - No boundary facets scenarios
- **Stale Hull Detection**:
  - Using hull with modified TDS (generation mismatch)
  - StaleHull error in visibility checks
  - StaleHull error in find_visible_facets
- **Cache Invalidation**:
  - Cache rebuilding after invalidation
  - Generation consistency verification

**Run with:** `cargo test --test test_convex_hull_error_paths` or `just test-release`

**Purpose:** Improves coverage of `src/geometry/algorithms/convex_hull.rs` by testing error paths that are difficult to trigger through
normal usage. Focuses on defensive validation and staleness detection.

### 🐛 Regression and Error Reproduction

#### [`test_cavity_boundary_error.rs`](./test_cavity_boundary_error.rs)

Reproduces and tests specific cavity boundary errors encountered during triangulation, ensuring fixes remain effective.

**Note**: This test can be slow (~3 minutes locally, longer in CI). The comprehensive test is gated behind `EXPENSIVE_TESTS=1` to avoid excessive CI time.

**Purpose:**

- Systematic reproduction of reported boundary errors
- Geometric degeneracy case testing
- Error condition validation and recovery

**Run with:** `cargo test --test test_cavity_boundary_error` or `just test-release`

#### [`coordinate_conversion_errors.rs`](./coordinate_conversion_errors.rs)

Tests error handling for coordinate conversion operations, particularly focusing on special floating-point values.

**Error Scenarios:**

- NaN coordinate handling
- Infinity value processing
- Subnormal value behavior
- Mixed problematic coordinate combinations
- Error message validation and context

**Run with:** `cargo test --test coordinate_conversion_errors` or `just test-release`

### 📊 Performance and Memory Testing

#### [`allocation_api.rs`](./allocation_api.rs)

Memory allocation profiling and testing utilities for tracking memory usage patterns during triangulation operations.

**Monitoring Areas:**

- Point and vertex creation allocations
- Triangulation data structure memory usage
- Complex workflow allocation patterns
- Memory efficiency validation

**Run with:** `just test-allocation`

**Note:** This uses the `count-allocations` feature flag automatically.

## Running Tests

### All Integration Tests

```bash
# Run all integration tests (recommended)
just test-release

# Run with verbose output for debugging
just test-debug
```

### Individual Test Files

```bash
# Run specific test file
cargo test --test <test_file_name>

# Examples
just test-debug                                  # circumsphere_debug_tools
cargo test --test robust_predicates_comparison   # specific integration test
just test-allocation                             # allocation profiling
```

### Performance Considerations

⚠️ **Important**: Integration tests may run significantly slower in debug mode. For optimal performance and accurate performance
measurements, run tests in release mode:

```bash
# Recommended: Run in release mode
just test-release

# Debug mode with verbose output
just test-debug
```

### Test Output

Many integration tests produce detailed analysis output:

```bash
# See detailed test output
just test-debug
```

## Test Development Guidelines

### Adding New Integration Tests

1. **File Naming**: Use descriptive names ending with the test purpose:
   - `*_debug_tools.rs` - Interactive debugging utilities
   - `*_integration.rs` - Algorithm integration testing
   - `*_comparison.rs` - Comparative analysis testing
   - `*_error.rs` - Error reproduction and regression testing

2. **Test Categories**: Organize tests by function:
   - **Debugging Tools**: Interactive analysis and debugging utilities
   - **Integration Testing**: Multi-component interaction testing
   - **Regression Testing**: Ensuring fixes remain effective
   - **Performance Testing**: Memory and execution time analysis

3. **Documentation**: Each test file should include:
   - Clear module documentation explaining the test purpose
   - Usage instructions with example commands
   - Description of test coverage and scenarios

### Test Output Standards

- Use `--nocapture` flag for verbose output in debugging tests
- Include performance timing information where relevant
- Provide clear success/failure indicators
- Include contextual information for debugging

### Performance Testing

- Always run performance-sensitive tests in release mode
- Include baseline comparisons where applicable
- Document expected performance characteristics
- Monitor memory allocation patterns

## Integration with Development Workflow

### Continuous Integration

All integration tests are automatically run in the CI pipeline:

- **GitHub Actions**: `.github/workflows/ci.yml`
- **Coverage Tracking**: Results are uploaded to Codecov (5-minute per-test timeout for slow CI environments)
- **Performance Regression**: Baseline comparisons are performed

### Development Testing

Integration tests should be run during development to:

1. **Validate Algorithm Changes**: Ensure modifications don't break existing functionality
2. **Debug Complex Issues**: Use debugging tools to analyze geometric edge cases
3. **Performance Impact**: Monitor performance implications of changes
4. **Regression Prevention**: Verify that known issues remain fixed

### Release Testing

Before releases, run the full integration test suite:

```bash
# Complete test validation
just test-release

# Include allocation testing
just test-allocation

# Comprehensive pre-release checks
just ci
```

## Contributing

When contributing integration tests:

1. **Follow Existing Patterns**: Use established test organization and naming conventions
2. **Include Documentation**: Provide clear descriptions and usage instructions
3. **Test Coverage**: Ensure comprehensive coverage of the functionality being tested
4. **Performance Awareness**: Consider performance implications and use release mode for timing-sensitive tests
5. **Error Handling**: Include appropriate error handling and validation

## Jaccard Similarity Testing Utilities

The test suite uses Jaccard similarity for robust set-based comparisons, enabling fuzzy-tolerant validation that handles
floating-point precision variations and near-degenerate cases.

### Available Utilities

#### Extraction Helpers (`delaunay::core::util`)

Canonical set extraction functions for comparing triangulation topology:

```rust
use delaunay::core::util::{
    extract_vertex_coordinate_set,    // HashSet<Point<T, D>>
    extract_edge_set,                  // HashSet<(u128, u128)>
    extract_facet_identifier_set,      // Result<HashSet<u64>, FacetError>
    extract_hull_facet_set,            // HashSet<u64>
};
```

**Features:**

- Deterministic canonicalization (sorted edges/facets)
- Uses existing `FacetView::key()` API for facet identification
- Safe f64 conversions with overflow detection (2^53 limit)
- No external hashing dependencies

#### Assertion Macro

```rust
use delaunay::assert_jaccard_gte;

let before = extract_vertex_coordinate_set(&tds_before);
let after = extract_vertex_coordinate_set(&tds_after);

// With custom label (4-arg form)
assert_jaccard_gte!(
    &before,
    &after,
    0.99,  // threshold: minimum acceptable similarity
    "Vertex preservation through operation"
);

// Without label (3-arg form) - uses default message
assert_jaccard_gte!(&before, &after, 0.99);
```

**On failure, provides detailed diagnostics:**

- Set sizes and Jaccard index value
- Intersection and union counts
- Sample symmetric differences (first 5 unique elements per set)

#### Diagnostic Reporting

```rust
use delaunay::core::util::format_jaccard_report;

let report = format_jaccard_report(
    &set_a,
    &set_b,
    "Expected",
    "Actual"
)?;
println!("{}", report);
```

### Threshold Conventions

| Test Scenario | Threshold | Rationale |
|--------------|-----------|------------|
| **Serialization** (vertex coords) | ≥ 0.99 | Strict preservation expected; allows minor floating-point drift |
| **Storage backend** (edge topology) | ≥ 0.999 | Near-exact equivalence; backends should be equivalent |
| **Hull reconstruction** (facet sets) | = 1.0 | Exact match when reconstructing from same TDS |
| **Property tests** (diagnostics) | N/A | Report similarity on failure; maintain strict invariants |

### Usage Examples

#### Vertex Coordinate Preservation

```rust
use delaunay::assert_jaccard_gte;
use delaunay::core::util::extract_vertex_coordinate_set;

let original_coords = extract_vertex_coordinate_set(&tds);
// ... perform operation (serialization, transformation, etc.) ...
let result_coords = extract_vertex_coordinate_set(&tds_after);

assert_jaccard_gte!(
    &original_coords,
    &result_coords,
    0.99,
    "Serialization vertex preservation"
);
```

#### Edge Set Comparison

```rust
use delaunay::core::util::extract_edge_set;

let edges_a = extract_edge_set(&tds_a);
let edges_b = extract_edge_set(&tds_b);

assert_jaccard_gte!(
    &edges_a,
    &edges_b,
    0.999,
    "Storage backend edge-set equivalence"
);
```

#### Hull Facet Topology

```rust
use delaunay::core::util::extract_hull_facet_set;
use delaunay::geometry::algorithms::convex_hull::ConvexHull;

let hull1 = ConvexHull::from_triangulation(&tds)?;
let hull2 = ConvexHull::from_triangulation(&tds)?;

let facets1 = extract_hull_facet_set(&hull1, &tds);
let facets2 = extract_hull_facet_set(&hull2, &tds);

assert_jaccard_gte!(
    &facets1,
    &facets2,
    1.0,
    "Hull reconstruction consistency"
);
```

### Design Decisions

**Why Jaccard similarity?**

- Handles floating-point precision variations gracefully
- Provides meaningful similarity metric (0.0 to 1.0)
- Better than exact equality for numeric/geometric computations
- Rich diagnostics on failure (shows what differs)

**Safety guarantees:**

- All `usize→f64` casts checked against 2^53 limit
- Proper error handling via `JaccardComputationError`
- No precision loss in computation

**Determinism:**

- Facet keys use FNV-based hashing (no random seeds)
- Edges canonicalized by sorting UUIDs
- Stable across runs and platforms

### Test Coverage

**Currently using Jaccard similarity:**

- ✅ `serialization_vertex_preservation.rs` - 4 tests with vertex coordinate comparison
- ✅ `proptest_convex_hull.rs` - 24 property tests (2D-5D) with hull facet topology comparison
- ✅ `proptest_triangulation.rs` - 4 neighbor symmetry tests (2D-5D) with enhanced failure diagnostics
  - Strict invariants maintained (no relaxation)
  - On failure: reports Jaccard similarity, set sizes, and common neighbors
  - Helps debug "near-miss" failures by quantifying similarity

**Deferred (not currently active):**

- `storage_backend_compatibility.rs` - Edge set comparison (all tests ignored - Phase 4 evaluation)

### Related Documentation

- **[Jaccard Similarity Theory](../docs/archive/jaccard.md)**: Mathematical background, adoption plan (completed in v0.5.4)
- **API Documentation**: `cargo doc --open` → `delaunay::core::util` module

## Related Documentation

- **[Examples](../examples/README.md)**: Usage demonstrations and library examples
- **[Benchmarks](../benches/README.md)**: Performance benchmarks and analysis
- **[Code Organization](../docs/code_organization.md)**: Complete project structure overview
- **[Numerical Robustness Guide](../docs/numerical_robustness_guide.md)**: Numerical stability documentation
- **[Jaccard Similarity Guide](../docs/archive/jaccard.md)**: Set similarity testing framework (archived - completed)
