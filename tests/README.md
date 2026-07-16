# Integration Tests

This directory contains integration tests for the delaunay library, focusing on comprehensive testing scenarios, debugging utilities,
regression testing, and performance analysis.

## Routine Test Buckets

Correctness tests live in two routine buckets:

- Default tests run through `just test`; each test should normally stay under
  about 10 seconds.
- Slow tests are deterministic correctness or regression tests that exceed that
  budget; gate them with `#[cfg(feature = "slow-tests")]` and run them with
  `just test-slow`.

Do not use `#[ignore]` as a slow-test marker. Slow correctness tests belong
behind `#[cfg(feature = "slow-tests")]`; benchmark-style measurements belong in
`benches/`; known limitations should be asserted explicitly instead of hidden.
High-dimensional benchmark-fixture certification that runs full Levels 1-5
validation also belongs behind `slow-tests`; keep the 2D/3D fixture smoke checks
in the default suite.

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
- **Simplex Mappings**: UUID↔key consistency for all simplices
- **No Duplicate Simplices**: No two simplices share the same vertex set
- **Simplex Validity**: Each simplex has correct vertex count and passes internal consistency checks
- **Simplex Vertex Count**: Maximal simplices have exactly D+1 vertices (fundamental Tds constraint)
- **Facet Sharing**: Each facet is shared by at most 2 simplices
- **Neighbor Consistency**: Neighbor relationships are mutual and reference shared facets
  - Neighbor symmetry (if A neighbors B, then B neighbors A)
  - Neighbor index semantics (correct facet-based indexing)
- **Vertex-Simplex Incidence**: All simplex vertices exist in the TDS
- **Vertex Count Consistency**: Vertex key count matches reported vertex count
- **Dimension Consistency**: Reported dimension matches actual structure

**Dimensions Tested:** 2D-5D; 4D/5D full TDS property variants over the 10-second budget run through `just test-slow`.

**Run with:** `cargo test --test proptest_tds` or included in `just test`

#### [`proptest_orientation.rs`](./proptest_orientation.rs)

Property-based tests focused on coherent orientation invariants in the TDS layer.

**Test Coverage:**

- **Construction coherence**: successfully-built triangulations are coherently oriented
- **Tamper detection**: simplex-order tampering is detected as `OrientationViolation`
- **Incremental coherence**: orientation remains coherent after each successful insertion

**Dimensions Tested:** 2D-5D; 4D/5D full orientation property variants over the 10-second budget run through `just test-slow`.

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

**Dimensions Tested:** 2D-5D; 4D/5D quality and facet-topology variants over the 10-second budget run through `just test-slow`.

**Run with:** `cargo test --test proptest_triangulation` or included in `just test`

#### [`proptest_delaunay_triangulation.rs`](./proptest_delaunay_triangulation.rs)

Property-based tests for `DelaunayTriangulation` invariants (all Delaunay-specific properties).

**Architectural Layer:** Delaunay-specific operations and the empty circumsphere property

**Test Coverage:**

- **Structural Invariants (Fast)**:
  - Incremental insertion maintains validity after each insertion
  - Duplicate coordinate rejection (geometric duplicate detection at insertion time)
- **Delaunay Property (Fast O(N) via Flip Predicates)**:
  - Empty circumsphere condition - No vertex lies strictly inside any simplex's circumsphere
  - Insertion-order robustness - Levels 1–3 validity across insertion orders
  - Duplicate cloud integration - Full pipeline with messy real-world inputs

**Status:** ✅ Fast Delaunay property tests run in the default suite; high-dimensional variants over the 10-second budget run through `just test-slow`.

**Implementation:** Bistellar flips (k=2 facets, k=3 ridges) with automatic Delaunay repair:

- Fast O(N) flip-based validation provides 40-100x speedup over brute-force
- Automatic repair runs after insertion/deletion via `DelaunayRepairPolicy`
- Inverse edge/triangle queues for 4D/5D repair
- See `src/core/algorithms/flips.rs` for implementation

**Slow variants:** 4D/5D empty-circumsphere, duplicate-coordinate, duplicate-cloud, and insertion-order robustness properties are gated by `slow-tests`.

**Dimensions Tested:** 2D-5D; variants over the 10-second budget run through `just test-slow`.

**Run with:** `cargo test --test proptest_delaunay_triangulation` or included in `just test`

#### [`proptest_simplex.rs`](./proptest_simplex.rs)

Property-based tests for Simplex data structure verifying simplex-level invariants and topological consistency.

**Test Coverage:**

- **Orientation Consistency**: Simplex vertex ordering and orientation preservation
- **Neighbor Linkage**: Neighbor references validity and symmetry
- **Facet Completeness**: All facets properly defined and accessible
- **Vertex References**: All vertex keys are valid and consistent

**Dimensions Tested:** 2D-5D; 4D/5D full simplex property variants over the 10-second budget run through `just test-slow`.

**Run with:** `cargo test --release --test proptest_simplex`

#### [`proptest_convex_hull.rs`](./proptest_convex_hull.rs)

Property-based tests for convex hull computation verifying hull properties and integration with triangulation.

**Test Coverage:**

- **Hull Vertex Extremeness**: Hull vertices are extreme points of the point set
- **Hull Facet Consistency**: All hull facets are valid and properly oriented
- **Boundary Subset Property**: Hull is a subset of triangulation boundary
- **Dimension Consistency**: Hull dimension matches point set dimension

**Dimensions Tested:** 2D-5D; convex-hull property variants over the 10-second budget run through `just test-slow`.

**Run with:** `cargo test --release --test proptest_convex_hull`

#### [`proptest_facet.rs`](./proptest_facet.rs)

Property-based tests for Facet operations verifying facet adjacency and orientation across neighboring simplices.

**Test Coverage:**

- **Mutual Neighbor References**: If simplex A has neighbor B via facet F, then B has A as neighbor
- **Co-facet Consistency**: Shared facets reference same vertices (possibly different order)
- **Orientation Alternation**: Adjacent simplices have opposite facet orientations
- **Facet Key Validity**: All facet identifiers are valid and retrievable

**Dimensions Tested:** 2D-5D; 4D/5D full facet property variants over the 10-second budget run through `just test-slow`.

**Run with:** `cargo test --release --test proptest_facet`

#### [`proptest_geometry.rs`](./proptest_geometry.rs)

Property-based tests for geometric utilities and predicates.

**Test Coverage:**

- **Orientation Antisymmetry**: Swapping vertices reverses orientation
- **Insphere/Outsphere Consistency**: Points are consistently classified relative to circumsphere
- **Circumsphere Invariants**: Simplex vertices lie on their circumsphere
- **Geometric Utility Correctness**: Helper functions produce valid results

**Run with:** `cargo test --release --test proptest_geometry`

#### [`proptest_sos.rs`](./proptest_sos.rs)

Property-based tests for the Simulation of Simplicity (SoS) module verifying the mathematical invariants
that a valid SoS implementation must satisfy.

**Test Coverage:**

- **Orientation Non-Degeneracy**: SoS orientation always returns ±1 for exactly degenerate (co-hyperplanar) inputs
- **Orientation Determinism**: same degenerate input always produces the same sign
- **Orientation Translation Invariance**: shifting all points by a constant integer offset preserves the sign
- **Insphere Non-Degeneracy**: SoS insphere always returns ±1 for exactly co-spherical (hyper-rectangle vertex) inputs
- **Insphere Determinism**: same co-spherical input always produces the same sign
- **Random Robustness**: never panics on arbitrary finite inputs (orientation and insphere)

**Dimensions Tested:** 2D–5D (via `gen_sos_tests!` macro)

**Run with:** `cargo test --test proptest_sos` or included in `just test-integration`

#### [`proptest_serialization.rs`](./proptest_serialization.rs)

Property-based tests for serialization and deserialization verifying data preservation via randomized structures.

**Test Coverage:**

- **Round-trip Equality**: Serialize → deserialize preserves structure and data
- **Neighbor Graph Preservation**: Simplex neighbor relationships survive round-trip
- **Vertex Data Integrity**: Vertex coordinates and associated data are preserved
- **Simplex Data Integrity**: Simplex-associated data is preserved
- **Cross-dimensional Serialization**: Works correctly for all supported dimensions

**Dimensions Tested:** 2D-5D; serialization property variants over the 10-second budget run through `just test-slow`.

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

**Current Proptest Regression Files:**

- `proptest_delaunay_triangulation.proptest-regressions`
- `proptest_sos.proptest-regressions`

These generated property-test corpora are separate from fixed-bug integration
regressions, which belong in [`regressions.rs`](./regressions.rs).

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
cargo test --test circumsphere_debug_tools --features diagnostics test_2d_circumsphere_debug -- --nocapture
cargo test --test circumsphere_debug_tools --features diagnostics test_3d_circumsphere_debug -- --nocapture
cargo test --test circumsphere_debug_tools --features diagnostics test_all_debug -- --exact --nocapture

# Run all debug tests at once (recommended)
just test-diagnostics
```

**Available Test Functions:**

- `test_2d_circumsphere_debug` - 2D triangle circumsphere testing
- `test_3d_circumsphere_debug` - 3D tetrahedron circumsphere testing  
- `test_4d_circumsphere_debug` - 4D simplex circumsphere testing
- `test_3d_matrix_analysis_debug` - Step-by-step matrix method analysis
- `test_compare_methods_debug` - Cross-dimensional method comparison
- `test_all_debug` - Complete comprehensive test suite

[View source](./circumsphere_debug_tools.rs)

### 🧪 Core Integration Testing

Core integration coverage currently includes:

- [`dedup_batch_construction.rs`](./dedup_batch_construction.rs)
- [`delaunay_incremental_insertion.rs`](./delaunay_incremental_insertion.rs)
- [`delaunay_repair_fallback.rs`](./delaunay_repair_fallback.rs)
- [`delaunay_edge_cases.rs`](./delaunay_edge_cases.rs)
- [`delaunayize_workflow.rs`](./delaunayize_workflow.rs)
- [`triangulation_builder.rs`](./triangulation_builder.rs)
- [`public_topology_api.rs`](./public_topology_api.rs)
- [`euler_characteristic.rs`](./euler_characteristic.rs)
- [`insert_with_statistics.rs`](./insert_with_statistics.rs)
- [`regressions.rs`](./regressions.rs)

#### [`serialization_vertex_preservation.rs`](./serialization_vertex_preservation.rs)

Integration tests for serialization ensuring vertex identifiers and associated data are preserved across serialize/deserialize cycles.

**Test Coverage:**

- **Vertex UUID Preservation**: Vertex identifiers remain stable across serialization
- **Coordinate Preservation**: Exact coordinate values are preserved
- **Vertex Data Preservation**: Associated vertex data survives round-trip
- **Simplex References**: Simplex-to-vertex references remain valid after deserialization

**Run with:** `cargo test --release --test serialization_vertex_preservation`

#### [`delaunayize_workflow.rs`](./delaunayize_workflow.rs)

Integration tests for the `delaunayize_by_flips` workflow validating the public API in `delaunay::delaunayize`.

**Test Coverage:**

- **Config Defaults**: `DelaunayizeConfig` default values
- **Non-Delaunay PL-Manifold Repair**: 2D and 3D success cases
- **Fallback Behavior**: fallback off/on does not trigger on valid triangulations
- **Outcome Stats**: stats populated correctly after repair
- **Determinism**: repeat runs produce identical outcome stats

**Run with:** `cargo test --test delaunayize_workflow`

### 🐛 Regression and Error Reproduction

#### [`large_scale_debug.rs`](./large_scale_debug.rs)

Reproduction-oriented debug harnesses for larger 2D-5D datasets tracked in
issues #340, #341, and #342.

**Run with:** `cargo test --release --features slow-tests --test large_scale_debug -- --nocapture`
or one of the active large-scale helpers:

- `just debug-large-scale-2d [n] [repair_every]` — default `n=36000`
- `just debug-large-scale-3d [n] [repair_every]` — issue #341, default `n=7500`
- `just debug-large-scale-4d [n] [repair_every]` — issue #340, default `n=800`
- `just debug-large-scale-5d [n] [repair_every]` — issue #342, default `n=140`

The `just` defaults are calibrated as roughly one-minute release-mode
acceptance runs on maintainer Apple M4 Max hardware. Each should insert all
vertices with zero skips, run final repair, and pass `validation_report` for
Levels 1–4. Expect normal hardware/load variation. Pass `n` explicitly when
reproducing heavier characterization probes such as 2D=40,000 or 5D=150.

**Note:** Use `--release` for runs above roughly 30 vertices; debug-mode
overhead makes large 3D/4D cases look hung even when the algorithm is making
progress. For the `new`/batch path, set
`DELAUNAY_BULK_PROGRESS_EVERY=<N>` to emit periodic batch-construction
summaries.

#### [`regressions.rs`](./regressions.rs)

Fixed-bug regression tests for known Delaunay construction and repair failures,
including the issue #120 2D minimal input, issue #306 3D flip-repair cycling
seed, and issue #307 4D bulk repair orientation case.

**Run with:** `cargo test --release --test regressions`

**Note:** Add new fixed-bug regression cases to this file by default. Keep them
as integration tests so the library is compiled without `cfg(test)`; only create
a separate regression test crate when the case needs separate crate-level
configuration, feature flags, or profile isolation.

#### [`coordinate_conversion_errors.rs`](./coordinate_conversion_errors.rs)

Tests error handling for coordinate conversion operations, particularly focusing on special floating-point values.

**Error Scenarios:**

- NaN coordinate handling
- Infinity value processing
- Subnormal value behavior
- Mixed problematic coordinate combinations
- Error message validation and context

**Run with:** `cargo test --test coordinate_conversion_errors` or `just test-integration`

### 📊 Performance and Memory Testing

#### [`allocation_api.rs`](./allocation_api.rs)

Smoke test for the optional allocation measurement API.

**Coverage:**

- `count-allocations` feature wiring
- `measure_with_result` returns the measured value
- allocation counters record an intentional allocation

**Run with:** `just test-allocation`

**Note:** Hot-path allocation budgets live in
[`benches/allocation_hot_paths.rs`](../benches/README.md),
not in the default test suite.

## Running Tests

### All Integration Tests

```bash
# Run all integration tests (recommended)
just test-integration

# Run with verbose output for debugging
just test-diagnostics
```

### Individual Test Files

```bash
# Run specific test file
cargo test --test <test_file_name>

# Examples
just test-diagnostics                            # circumsphere_debug_tools
cargo test --test delaunay_incremental_insertion # specific integration test
just test-allocation                             # allocation measurement wiring
```

### Performance Considerations

⚠️ **Important**: Integration tests may run significantly slower in debug mode. For optimal performance and accurate performance
measurements, run tests in release mode:

```bash
# Recommended: Run in release mode
just test-integration

# Debug mode with verbose output
just test-diagnostics
```

### Test Output

Many integration tests produce detailed analysis output:

```bash
# See detailed test output
just test-diagnostics
```

## Test Development Guidelines

### Adding New Integration Tests

1. **File Naming**: Use descriptive names ending with the test purpose:
   - `*_debug_tools.rs` - Interactive debugging utilities
   - `*_integration.rs` - Algorithm integration testing
   - `*_comparison.rs` - Comparative analysis testing
   - `*_errors.rs` - Error-path testing
   - `regressions.rs` - Fixed-bug regression tests; add new bug regressions here

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
- **Coverage Tracking**: Coverage is generated with `cargo-llvm-cov` through `just coverage-ci` and uploaded to Codecov
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
just test-integration

# Verify allocation measurement wiring
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

#### Extraction Helpers (`delaunay::prelude::query`)

Canonical set extraction functions for comparing triangulation topology:

```rust
use delaunay::prelude::query::{
    extract_vertex_coordinate_set,    // HashSet<Point<D>>
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

let before = extract_vertex_coordinate_set(tri_before);
let after = extract_vertex_coordinate_set(tri_after);

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
use delaunay::prelude::query::format_jaccard_report;

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
use delaunay::prelude::query::extract_vertex_coordinate_set;

let original_coords = extract_vertex_coordinate_set(tri);
// ... perform operation (serialization, transformation, etc.) ...
let result_coords = extract_vertex_coordinate_set(tri_after);

assert_jaccard_gte!(
    &original_coords,
    &result_coords,
    0.99,
    "Serialization vertex preservation"
);
```

#### Edge Set Comparison

```rust
use delaunay::prelude::query::extract_edge_set;

let edges_a = extract_edge_set(tri_a);
let edges_b = extract_edge_set(tri_b);

assert_jaccard_gte!(
    &edges_a,
    &edges_b,
    0.999,
    "Storage backend edge-set equivalence"
);
```

#### Hull Facet Topology

```rust
use delaunay::prelude::query::extract_hull_facet_set;
use delaunay::geometry::algorithms::convex_hull::ConvexHull;

let hull1 = ConvexHull::try_from_triangulation(&tds)?;
let hull2 = ConvexHull::try_from_triangulation(&tds)?;

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

- ✅ `serialization_vertex_preservation.rs` - 3 tests with vertex coordinate comparison
- ✅ `proptest_convex_hull.rs` - 24 property tests (2D-5D) with hull facet topology comparison
- ✅ `proptest_triangulation.rs` - 4 neighbor symmetry tests (2D-5D) with enhanced failure diagnostics
  - Strict invariants maintained (no relaxation)
  - On failure: reports Jaccard similarity, set sizes, and common neighbors
  - Helps debug "near-miss" failures by quantifying similarity

### Related Documentation

- **[Jaccard Similarity Theory](../docs/archive/jaccard.md)**: Mathematical background, adoption plan (completed in v0.5.4)
- **API Documentation**: `cargo doc --open` → `delaunay::query` or
  `delaunay::prelude::query` for curated test helpers

## Related Documentation

- **[Examples](../examples/README.md)**: Usage demonstrations and library examples
- **[Benchmarks](../benches/README.md)**: Performance benchmarks and analysis
- **[Code Organization](../docs/code_organization.md)**: Complete project structure overview
- **[Numerical Robustness Guide](../docs/numerical_robustness_guide.md)**: Numerical stability documentation
- **[Jaccard Similarity Guide](../docs/archive/jaccard.md)**: Set similarity testing framework (archived - completed)
