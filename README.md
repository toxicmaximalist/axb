# axb

[![Crates.io](https://img.shields.io/crates/v/axb.svg)](https://crates.io/crates/axb)
[![Documentation](https://docs.rs/axb/badge.svg)](https://docs.rs/axb)
[![License](https://img.shields.io/crates/l/axb.svg)](https://github.com/Fugazzii/Axb)

A lightweight, zero-dependency linear algebra library for Rust with compile-time dimension checking using const generics.

## Features

- 🔒 **Compile-time dimension checking** - Catch dimension mismatches at compile time
- 📦 **Zero dependencies** - No external dependencies for the core library
- 🛡️ **Result-based error handling** - No panics, all errors are returned as `Result`
- ➕ **Operator overloading** - Natural syntax with `+`, `-`, `*` operators
- 📐 **Matrix factorizations** - QR, LU, and LDU decompositions
- 🧮 **Comprehensive operations** - Determinant, inverse, transpose, trace, rank, and more

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
axb = "0.2"
```

## Quick Start

```rust
use axb::matrix::Matrix;

fn main() {
    // Create matrices
    let a = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
    let b = Matrix::<2, 2>::new([[5.0, 6.0], [7.0, 8.0]]);

    // Arithmetic operations
    let sum = (&a + &b).unwrap();
    let diff = (&a - &b).unwrap();
    let product = (&a * &b).unwrap();
    let scaled = &a * 2.0;

    // Matrix operations
    let det = a.determinant().unwrap();      // -2.0
    let inv = a.inverse().unwrap();
    let transposed = a.transpose();
    let trace = a.trace();                    // 5.0

    // Indexing
    let element = a[(0, 1)];                  // 2.0
    
    println!("Determinant: {}", det);
    println!("Inverse:\n{}", inv);
}
```

## Constructors

```rust
use axb::matrix::Matrix;

// From 2D array
let m = Matrix::<2, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

// From nested Vec
let v = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
let m = Matrix::<2, 2>::from_vec(v);

// Special matrices
let zeros = Matrix::<3, 3>::zeros();
let identity = Matrix::<3, 3>::identity();
let diagonal = Matrix::<3, 3>::diagonal(&[1.0, 2.0, 3.0]);
let filled = Matrix::<2, 2>::fill(5.0);
```

## Matrix Factorizations

```rust
use axb::matrix::Matrix;

let m = Matrix::<3, 3>::new([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 10.0],
]);

// QR Factorization (A = QR)
let (q, r) = m.qr().unwrap();

// LU Factorization (A = LU)
let (l, u) = m.lu().unwrap();

// LDU Factorization (A = LDU)
let (l, d, u) = m.ldu().unwrap();
```

## Solving Linear Systems

```rust
use axb::matrix::Matrix;

// Solve Ax = b
let a = Matrix::<2, 2>::new([[2.0, 1.0], [1.0, 3.0]]);
let b = Matrix::<2, 1>::new([[5.0], [10.0]]);

let x = a.solve(&b).unwrap();
// x ≈ [1.0, 3.0]
```

## Vector Operations

```rust
use axb::matrix::Matrix;

let v1 = Matrix::<3, 1>::new([[1.0], [2.0], [3.0]]);
let v2 = Matrix::<3, 1>::new([[4.0], [5.0], [6.0]]);

let dot = v1.dot(&v2);           // 32.0
let norm = v1.norm();            // √14
let unit = v1.normalize().unwrap();
let cross = v1.cross(&v2);       // 3D cross product
```

## Matrix Properties

```rust
use axb::matrix::Matrix;

let m = Matrix::<3, 3>::new([
    [1.0, 2.0, 3.0],
    [2.0, 4.0, 5.0],
    [3.0, 5.0, 6.0],
]);

m.is_symmetric();     // true
m.is_orthogonal();    // false
m.is_projection();    // false
m.rank();             // matrix rank
m.frobenius_norm();   // Frobenius norm
```

## Error Handling

All fallible operations return `Result<T, MatrixError>`:

```rust
use axb::matrix::{Matrix, MatrixError};

let singular = Matrix::<2, 2>::new([[1.0, 2.0], [2.0, 4.0]]);

match singular.inverse() {
    Ok(inv) => println!("Inverse: {}", inv),
    Err(MatrixError::NotInvertible) => println!("Matrix is singular"),
    Err(e) => println!("Error: {}", e),
}

// Non-square matrix operations
let rect = Matrix::<2, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
assert!(rect.determinant().is_err()); // NonSquare error
```

## API Reference

### Basic Operations
| Method | Description |
|--------|-------------|
| `new(arr)` | Create from 2D array |
| `from_vec(v)` | Create from nested Vec |
| `zeros()` | Create zero matrix |
| `identity()` | Create identity matrix |
| `diagonal(values)` | Create diagonal matrix |
| `transpose()` | Transpose matrix |
| `determinant()` | Compute determinant |
| `inverse()` | Compute inverse |
| `trace()` | Sum of diagonal elements |
| `rank()` | Compute matrix rank |
| `pow(n)` | Matrix exponentiation |

### Factorizations
| Method | Description |
|--------|-------------|
| `qr()` | QR factorization |
| `lu()` | LU factorization |
| `ldu()` | LDU factorization |

### Vector Operations (for `Matrix<R, 1>`)
| Method | Description |
|--------|-------------|
| `norm()` | Euclidean norm |
| `normalize()` | Unit vector |
| `dot(other)` | Dot product |
| `cross(other)` | Cross product (3D only) |

### Properties
| Method | Description |
|--------|-------------|
| `is_symmetric()` | Check if A = A^T |
| `is_orthogonal()` | Check if A * A^T = I |
| `is_projection()` | Check if P² = P |
| `frobenius_norm()` | Frobenius norm |

## Minimum Supported Rust Version

This crate requires Rust 1.51 or later (for const generics).

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
