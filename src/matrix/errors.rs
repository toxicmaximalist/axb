use std::fmt;

/// Tolerance for floating-point comparisons
pub const EPSILON: f32 = 1e-10;

/// Error types for matrix operations
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixError {
    /// Matrix operation requires a square matrix but received non-square
    NonSquare { rows: usize, cols: usize },
    /// Matrix is singular and cannot be inverted
    NotInvertible,
    /// Matrix dimensions do not match for the operation
    DimensionMismatch {
        expected: (usize, usize),
        got: (usize, usize),
    },
    /// Operation requires a vector (single column) matrix
    NotAVector { cols: usize },
    /// Division by zero or near-zero value encountered
    NumericalInstability,
    /// Empty matrix provided where non-empty required
    EmptyMatrix,
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonSquare { rows, cols } => {
                write!(f, "Matrix must be square, got {}x{}", rows, cols)
            }
            Self::NotInvertible => write!(f, "Matrix is singular and cannot be inverted"),
            Self::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "Dimension mismatch: expected {}x{}, got {}x{}",
                    expected.0, expected.1, got.0, got.1
                )
            }
            Self::NotAVector { cols } => {
                write!(f, "Expected a vector (1 column), got {} columns", cols)
            }
            Self::NumericalInstability => {
                write!(
                    f,
                    "Numerical instability: division by zero or near-zero value"
                )
            }
            Self::EmptyMatrix => write!(f, "Empty matrix provided"),
        }
    }
}

impl std::error::Error for MatrixError {}

/// Result type alias for matrix operations
pub type MatrixResult<T> = Result<T, MatrixError>;
