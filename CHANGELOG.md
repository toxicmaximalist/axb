# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-01-29

### Added
- PLU decomposition (LU with partial pivoting) for improved numerical stability
- Cholesky decomposition for symmetric positive-definite matrices
- Comprehensive examples in `/examples` directory

### Changed
- Converted from binary to library-only structure
- Updated documentation with complete API reference

## [0.2.0] - 2026-01-29

### Added
- Complete rewrite with production-ready code
- Result-based error handling with `MatrixError` enum
- Matrix constructors: `new`, `from_vec`, `zeros`, `identity`, `diagonal`, `fill`
- Matrix accessors: `rows`, `cols`, `shape`, `get`, `set`, `row`, `col`
- Matrix operations: `transpose`, `determinant`, `inverse`, `trace`, `rank`
- Matrix factorizations: QR, LU, LDU decompositions
- Linear system solver (`solve`)
- Vector operations: `norm`, `normalize`, `dot`, `cross` (3D)
- Matrix properties: `is_symmetric`, `is_orthogonal`, `is_projection`
- Binary exponentiation for `pow()` method
- Operator overloading: `+`, `-`, `*`, unary `-`, indexing
- Standard traits: `PartialEq`, `Default`, `Display`, `From`
- Comprehensive documentation with doc examples
- 34+ unit tests

### Changed
- Renamed internal `matrice` field to `data`
- All fallible operations now return `Result<T, MatrixError>`

### Fixed
- `power()` method bug (was squaring instead of multiplying by original)
- `is_projection_matrix()` was checking P²=I instead of P²=P
- `new_from_vec()` dimension calculation bug

## [0.1.0] - Initial Release

### Added
- Basic matrix struct with const generics
- Simple matrix operations
