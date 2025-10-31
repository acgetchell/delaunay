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

#### [`proptest_triangulation.rs`](./proptest_triangulation.rs)

Property-based tests for triangulation structural invariants.

**Test Coverage:**

- **Triangulation Validity**:
  - Constructed triangulations pass is_valid()
  - Cross-dimensional validity (2D-3D)
- **Neighbor Symmetry**:
  - If A neighbors B, then B neighbors A
  - Reciprocal neighbor relationships
- **Vertex-Cell Incidence**:
  - All cell vertices exist in TDS
  - Vertex key consistency
- **No Duplicate Cells**:
  - No two cells have identical vertex sets
  - Unique cell configurations
- **Incremental Construction**:
  - Validity maintained after vertex insertion
  - Dimension consistency after growth
- **Dimension Consistency**:
  - Dimension matches vertex count expectations
  - Proper dimension evolution
- **Vertex Count Consistency**:
  - Vertex keys count matches number_of_vertices()
  - Iterator consistency

**Run with:** `cargo test --test proptest_triangulation` or included in `just test`

#### [`proptest_bowyer_watson.rs`](./proptest_bowyer_watson.rs)

Property-based tests for Bowyer-Watson insertion algorithm verifying invariants during randomized vertex insertion sequences across dimensions (2D-5D).

**Test Coverage:**

- **Cavity Boundary Correctness**:
  - Conflict zone boundary facets are correctly identified
  - No orphaned facets after insertion
- **Delaunay Property Preservation**:
  - Delaunay property holds after each insertion
  - Circumsphere test consistency
- **Neighbor Symmetry**:
  - Reciprocal neighbor relationships maintained
  - No broken neighbor links
- **Structural Integrity**:
  - No orphan vertices or cells after insertion
  - Vertex count consistency

**Run with:**

```bash
# Standard test run
cargo test --release --test proptest_bowyer_watson

# With increased test cases and verbose output
PROPTEST_CASES=512 cargo test --release --test proptest_bowyer_watson -- --nocapture

# Reproduce a specific failure
PROPTEST_SEED=<seed> cargo test --release --test proptest_bowyer_watson -- --nocapture
```

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

#### [`proptest_quality.rs`](./proptest_quality.rs)

Property-based tests for geometry quality metrics (radius ratio and normalized volume).

**Test Coverage:**

- **Valid Ranges**: Metrics produce finite, positive values for valid simplices
- **Scale Invariance**: Metrics remain unchanged under uniform scaling
- **Translation Invariance**: Metrics remain unchanged under translation
- **Degeneracy Sensitivity**: Metrics detect and handle degenerate configurations
- **Cross-dimensional Consistency**: Metrics behave correctly across dimensions

**Run with:**

```bash
# Standard test run
cargo test --release --test proptest_quality

# With increased test cases for thorough validation
PROPTEST_CASES=1024 cargo test --release --test proptest_quality -- --nocapture
```

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
  PROPTEST_SEED=12345 cargo test --release --test proptest_quality -- --nocapture
  ```

- **Performance Note**: Regression cases run before random cases; many entries can slow tests
- **Filtering**: Use test filters to narrow scope when iterating on specific properties
- **Maintenance**: It's acceptable to prune obsolete entries in follow-up PRs (keep diffs focused)

**Current Regression Files:**

- `proptest_bowyer_watson.proptest-regressions`
- `proptest_convex_hull.proptest-regressions`
- `proptest_quality.proptest-regressions`
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

### 🐛 Regression and Error Reproduction

#### [`test_cavity_boundary_error.rs`](./test_cavity_boundary_error.rs)

Reproduces and tests specific cavity boundary errors encountered during triangulation, ensuring fixes remain effective.

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
- **Coverage Tracking**: Results are uploaded to Codecov
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
just commit-check
```

## Contributing

When contributing integration tests:

1. **Follow Existing Patterns**: Use established test organization and naming conventions
2. **Include Documentation**: Provide clear descriptions and usage instructions
3. **Test Coverage**: Ensure comprehensive coverage of the functionality being tested
4. **Performance Awareness**: Consider performance implications and use release mode for timing-sensitive tests
5. **Error Handling**: Include appropriate error handling and validation

## Related Documentation

- **[Examples](../examples/README.md)**: Usage demonstrations and library examples
- **[Benchmarks](../benches/README.md)**: Performance benchmarks and analysis
- **[Code Organization](../docs/code_organization.md)**: Complete project structure overview
- **[Numerical Robustness Guide](../docs/numerical_robustness_guide.md)**: Numerical stability documentation
