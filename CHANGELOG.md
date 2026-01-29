# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-01-29

### Changed
- Updated documentation and changelog formatting

## [1.0.0] - 2026-01-29

### Added
- Comprehensive matrix operations (transpose, inverse, determinant, trace, rank)
- Matrix decompositions: LU, PLU (with partial pivoting), LDU, QR, Cholesky
- Linear system solver (`solve`)
- Vector operations: norm, normalize, dot, cross (3D)
- Matrix properties: is_symmetric, is_orthogonal, is_projection
- Operator overloading (+, -, *, indexing)
- Compile-time dimension checking via const generics
- Proper error handling with `Result<T, MatrixError>`
- Constructors: new, zeros, identity, diagonal, fill, from_vec
- Binary exponentiation for matrix power
- Comprehensive test suite (53 unit tests, 18 doc tests)
- CI/CD pipeline with GitHub Actions
- MIT and Apache-2.0 dual licensing
- Examples: basic_usage, decompositions

### Changed
- Library structure (removed main.rs, proper lib.rs exports)
- All fallible operations return `Result<T, MatrixError>`

## [0.1.0] - Initial release

### Added
- Basic matrix struct with const generics
- Simple matrix operations
- Basic matrix struct with const generics
- Simple matrix operations
