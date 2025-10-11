# Integration Tests

This directory contains integration tests for the delaunay library, focusing on comprehensive testing scenarios, debugging utilities,
regression testing, and performance analysis.

## Test Categories

### 🔧 Debugging and Analysis Tools

#### `circumsphere_debug_tools.rs`

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

#### `convex_hull_bowyer_watson_integration.rs`

Integration tests for convex hull algorithms with Bowyer-Watson triangulation, focusing on the interaction between hull computation and triangulation construction.

**Test Coverage:**

- Hull extension execution and validation
- Cache behavior and reset operations
- Multiple hull extension scenarios
- Triangulation validity after hull operations
- Mixed insertion strategy testing

**Run with:** `cargo test --test convex_hull_bowyer_watson_integration` or `just test-release`

#### `robust_predicates_comparison.rs`

Comparative testing between robust and standard geometric predicates, focusing on numerical accuracy and edge case handling.

**Test Scenarios:**

- Cocircular and nearly coplanar points
- High precision coordinate handling
- Extreme aspect ratio configurations
- Vertex insertion robustness analysis
- Performance cost-benefit analysis

**Run with:** `cargo test --test robust_predicates_comparison` or `just test-release`

#### `robust_predicates_showcase.rs`

Demonstration and stress testing of robust geometric predicates with focus on numerical edge cases and degenerate configurations.

**Features:**

- Degenerate failure recovery demonstrations
- Tolerance limit stress testing
- Real-world triangulation scenarios
- Performance impact analysis

**Run with:** `cargo test --test robust_predicates_showcase` or `just test-release`

### 🐛 Regression and Error Reproduction

#### `test_cavity_boundary_error.rs`

Reproduces and tests specific cavity boundary errors encountered during triangulation, ensuring fixes remain effective.

**Purpose:**

- Systematic reproduction of reported boundary errors
- Geometric degeneracy case testing
- Error condition validation and recovery

**Run with:** `cargo test --test test_cavity_boundary_error` or `just test-release`

#### `coordinate_conversion_errors.rs`

Tests error handling for coordinate conversion operations, particularly focusing on special floating-point values.

**Error Scenarios:**

- NaN coordinate handling
- Infinity value processing
- Subnormal value behavior
- Mixed problematic coordinate combinations
- Error message validation and context

**Run with:** `cargo test --test coordinate_conversion_errors` or `just test-release`

### 📊 Performance and Memory Testing

#### `allocation_api.rs`

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
