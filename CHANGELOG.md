# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.3] - 2025-08-04

### Added

- SlotMap integration for vertices and cells in the triangulation data structure
  - Introduces `VertexKey` and `CellKey` types for stable key access
  - Maintains HashMap for UUID-to-key mappings to preserve external UUID-based access
  - Enhances memory management and overall efficiency of triangulation process
- New `vertex` module with `VertexBuilder` for simplified vertex creation
- Comprehensive benchmarking infrastructure
  - Small-scale triangulation benchmarks for 2D, 3D, and 4D implementations
  - Allocation tracking using `allocation-counter` crate
  - Performance regression analysis scripts
  - Baseline performance metrics recording
- BoundaryAnalysis trait for modular boundary facet identification
- Prelude module for simplified imports
  - Consolidates commonly used types and macros
  - Reduces boilerplate in examples and user code
  - Re-exports vertex! and cell! macros for convenience
- Enhanced triangulation data structure documentation
  - Comprehensive module-level documentation with examples
  - Detailed complexity analysis for algorithms
  - Usage examples for different dimensional triangulations

### Changed

- Major triangulation data structure refactoring with SlotMap
  - Replaced HashMap with SlotMap for vertices and cells storage
  - Implemented BiMaps for efficient UUID-to-key bidirectional lookup
  - Custom serialization/deserialization logic for BiMap integration
- Improved Bowyer-Watson triangulation algorithm
  - Renamed `bowyer_watson_logic` to `bowyer_watson` for consistency
  - Added reusable buffers for intermediate data to reduce memory allocations
  - Enhanced performance with pre-allocation and early-exit strategies
- Enhanced vertex and cell implementations
  - Vertex equality and hashing now only consider coordinates for consistency
  - Cell hashing based on sorted vertices for proper Eq/Hash contract compliance
  - Improved facet adjacency detection using HashSet for vertex comparison
- Streamlined imports with prelude module
- Error handling improvements in supercell creation and neighbor assignments
- Streamlined supercell removal process
  - Simplified `remove_cells_containing_supercell_vertices` method signature
  - Removed redundant supercell parameter for cleaner API
  - Eliminated duplicate calls to `remove_duplicate_cells` for better performance
- Improved error handling in supercell creation
  - Replaced unwrap() calls with proper error propagation
  - Added informative error messages for triangulation failures
  - Enhanced robustness during triangulation initialization
- Code organization and structure improvements
  - Standardized organizational patterns across modules
  - Enhanced readability and maintainability
  - Improved documentation and naming conventions

### Performance

- Significant performance enhancements in key triangulation methods
  - Optimized Bowyer-Watson algorithm with buffer reuse
  - Enhanced neighbor assignment with facet-based identification
  - Improved memory allocation patterns with pre-allocation strategies
- Allocation tracking and optimization
  - Integration of `allocation-counter` crate for memory usage measurement
  - Performance regression analysis and baseline establishment
- Boundary facet detection optimization
  - Utilizes precomputed facet-to-cells mapping for faster boundary identification
  - Avoids iterating through all cells for boundary determination
- Optimized supercell removal logic
  - Reduced redundant operations in cleanup phase
  - Streamlined cell filtering and removal process
- Improved boundary facet detection algorithms
  - Reduced complexity from O(N²·F²) to O(N·F) using HashMap-based approach
  - Single-pass facet-to-cells mapping for efficient boundary identification
  - Significant performance gains for large triangulations (2000+ cells)

### Fixed

- UUID collision prevention in vertex creation
- Regression in neighbor assignment logic preventing proper neighbor relationships
- Vertex equality and hashing consistency issues
- Scientific notation handling in benchmark comparison scripts
- Test code refactoring and maintainability improvements
- Eliminated unnecessary cloning in supercell insertion
- Corrected method signatures for consistency
- Improved memory efficiency in cell removal operations

### Technical Details

- SlotMap provides stable keys and efficient access for large triangulations
- BiMap integration enables bidirectional UUID-to-key lookup with improved performance
- Custom serialization ensures data integrity during serialization/deserialization
- Benchmark infrastructure includes 2D, 3D, and 4D triangulation performance testing
- BoundaryAnalysis trait promotes modularity and testability of boundary algorithms

## [0.3.2] - 2025-07-29

### Added

- Complete test coverage for coordinate traits (100% coverage)
  - Comprehensive tests for hash collision edge cases and coordinate distribution
  - Full coverage of `CoordinateScalar` trait implementations and tolerance methods
  - Edge case testing for coordinate patterns, extreme dimensions, and boundary values
  - Tests for mixed special floating-point values (NaN, infinity, negative zero)
  - Validation testing for different dimensions and error scenarios
- Enhanced testing framework
  - Robust testing for `Cell`, `Facet`, `Vertex`, and utility functions
  - Coverage for serialization/deserialization edge cases
  - Error handling validation in geometric predicates
  - Validation tests for data structure integrity
- New macros `vertex!` and `cell!`
  - Simplified creation of vertices and cells
  - `vertex!` macro for vertex instantiation
  - `cell!` macro for cell creation in tests and examples

### Changed

- Coordinate trait simplification
  - Reduced lines in `coordinate.rs` while maintaining functionality
  - Consolidated redundant test cases and improved test organization
  - Enhanced code clarity and maintainability
- Facet structure enhancements
  - Improved facet creation, validation, and serialization
  - Simplified example usage with macro support
  - Strengthened error handling and edge case coverage
- Cell structure improvements
  - Optimized cell creation and validation processes
  - Enhanced circumsphere and geometric predicates
  - Improved memory efficiency and code organization
- Testing framework refactoring
  - Removed redundant test utilities and consolidated testing patterns
  - Improved test readability and maintainability
  - Enhanced error message validation

### Fixed

- Clippy compliance
  - Resolved all clippy pedantic warnings
  - Fixed unreadable literal warnings with separators
  - Replaced float comparisons with relative comparisons
  - Annotated intentional type conversions
  - Fixed floating-point array comparisons
  - Improved code readability and maintainability
- Variable naming
  - Renamed variables to avoid similar name confusion
- Code organization
  - Eliminated redundant code
  - Improved error handling patterns
  - Enhanced documentation and code comments

### Performance

- Reduced code complexity
  - Significant reduction in lines while maintaining functionality
  - Overall improvement in compilation times and maintainability

### Technical Details

- All 26 coordinate trait tests pass
  - Comprehensive coverage of functionality
  - Handling of special floating-point values
  - Hash collision resistance and distribution testing
  - Ordered equality semantics for floating-point coordinates
  - Validation error handling
  - Tolerance method implementations for precision levels
- Enhanced data structure validation ensures geometric consistency
- Improved floating-point comparison practices
- Strengthened serialization/deserialization robustness
- All geometric predicates maintain numerical stability

## [0.3.1] - 2025-07-26

### Changed

- Distance calculation consistency
  - Standardized all distance calculations by using utility functions
- Code quality improvements

### Performance

- Measurable performance gains
  - Updated benchmarks to reflect improvements
- Cumulative improvements

### Technical Details

- The `hypot` function provides stable Euclidean distance calculations
- Improved test code clarity

## [0.3.0] - 2025-07-25

### Changed

- Generalized coordinate types
  - Coordinate structures now accept a generic floating-point type instead of being hardcoded
- Point implementation improvements

### Added

- `CoordinateScalar` trait alias
- Support for multiple floating-point precision levels

### Fixed

- Resolved clippy warnings
- Improved code clarity in Point conversion implementations

### Technical Details

- The `Coordinate` trait provides a unified interface

## [0.2.0] - 2024-12-19

### Initial Release

- Initial release with Delaunay triangulation support
- Serialization/Deserialization capabilities
- CGAL-inspired API design

[Unreleased]: https://github.com/acgetchell/delaunay/compare/v0.3.3...HEAD
[0.3.3]: https://github.com/acgetchell/delaunay/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/acgetchell/delaunay/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/acgetchell/delaunay/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/acgetchell/delaunay/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/acgetchell/delaunay/releases/tag/v0.2.0
