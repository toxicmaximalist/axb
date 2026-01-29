//! # axb - Linear Algebra Library for Rust
//!
//! `axb` is a lightweight, zero-dependency linear algebra library that provides
//! matrix operations with compile-time dimension checking using const generics.
//!
//! ## Features
//!
//! - **Compile-time dimensions**: Matrix dimensions are checked at compile time
//! - **Zero dependencies**: No external dependencies for the core library
//! - **Comprehensive operations**: Determinant, inverse, transpose, and more
//! - **Matrix factorizations**: QR, LU, and LDU decompositions
//! - **Operator overloading**: Natural syntax with `+`, `-`, `*` operators
//! - **Result-based error handling**: No panics in library code
//!
//! ## Quick Start
//!
//! ```rust
//! use axb::Matrix;
//!
//! // Create matrices
//! let a = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
//! let b = Matrix::<2, 2>::new([[5.0, 6.0], [7.0, 8.0]]);
//!
//! // Arithmetic operations
//! let sum = (&a + &b).unwrap();
//! let product = (&a * &b).unwrap();
//!
//! // Matrix operations
//! let det = a.determinant().unwrap();
//! let inv = a.inverse().unwrap();
//! let transposed = a.transpose();
//!
//! // Factorizations
//! let (q, r) = a.qr().unwrap();
//! let (l, u) = a.lu().unwrap();
//! ```
//!
//! ## Error Handling
//!
//! All fallible operations return `Result<T, MatrixError>`:
//!
//! ```rust
//! use axb::{Matrix, MatrixError};
//!
//! let singular = Matrix::<2, 2>::new([[1.0, 2.0], [2.0, 4.0]]);
//! match singular.inverse() {
//!     Ok(inv) => println!("Inverse: {}", inv),
//!     Err(MatrixError::NotInvertible) => println!("Matrix is singular"),
//!     Err(e) => println!("Error: {}", e),
//! }
//! ```

pub mod matrix;

// Re-export commonly used types at crate root
pub use matrix::{Matrix, MatrixError, MatrixResult, EPSILON};