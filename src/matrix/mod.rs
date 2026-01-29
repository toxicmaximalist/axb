//! Matrix module providing linear algebra operations.
//!
//! This module contains the [`Matrix`] struct and associated operations for
//! linear algebra computations including:
//! - Basic operations (transpose, determinant, inverse)
//! - Matrix factorizations (QR, LU, LDU)
//! - Arithmetic operations via operator overloading
//!
//! # Example
//! ```
//! use axb::matrix::Matrix;
//!
//! let a = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
//! let b = Matrix::<2, 2>::new([[5.0, 6.0], [7.0, 8.0]]);
//!
//! // Matrix addition
//! let sum = (&a + &b).unwrap();
//!
//! // Matrix multiplication  
//! let product = (&a * &b).unwrap();
//!
//! // Determinant
//! let det = a.determinant().unwrap();
//! ```

mod errors;
pub mod transform;

pub use errors::{MatrixError, MatrixResult, EPSILON};

use std::fmt;
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

use self::transform::{is_near_zero, mult};

/// A matrix with compile-time known dimensions.
///
/// # Type Parameters
/// * `R` - Number of rows
/// * `C` - Number of columns
///
/// # Example
/// ```
/// use axb::matrix::Matrix;
///
/// // Create a 2x3 matrix
/// let m = Matrix::<2, 3>::new([
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0],
/// ]);
/// ```
#[derive(Debug, Clone)]
pub struct Matrix<const R: usize, const C: usize> {
    data: Vec<Vec<f32>>,
}

// ============================================================================
// Constructors
// ============================================================================

impl<const R: usize, const C: usize> Matrix<R, C> {
    /// Creates a new matrix from a 2D array.
    ///
    /// # Arguments
    /// * `arr` - A 2D array of dimensions R×C
    ///
    /// # Example
    /// ```
    /// use axb::matrix::Matrix;
    ///
    /// let m = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
    /// ```
    pub fn new(arr: [[f32; C]; R]) -> Self {
        let data: Vec<Vec<f32>> = arr.iter().map(|row| row.to_vec()).collect();
        Self { data }
    }

    /// Creates a new matrix from a nested Vec.
    ///
    /// # Arguments
    /// * `v` - A nested vector. Dimensions should match R×C.
    ///
    /// # Note
    /// If the input dimensions don't match R×C, the matrix will be
    /// padded with zeros or truncated as needed.
    pub fn from_vec(v: Vec<Vec<f32>>) -> Self {
        let mut data = Vec::with_capacity(R);
        for row_idx in 0..v.len().min(R) {
            let mut row = Vec::with_capacity(C);
            for col_idx in 0..v[row_idx].len().min(C) {
                row.push(v[row_idx][col_idx]);
            }
            // Pad with zeros if row is shorter than C
            while row.len() < C {
                row.push(0.0);
            }
            data.push(row);
        }
        // Pad with zero rows if fewer than R rows provided
        while data.len() < R {
            data.push(vec![0.0; C]);
        }
        Self { data }
    }

    /// Creates a zero matrix.
    ///
    /// # Example
    /// ```
    /// use axb::matrix::Matrix;
    ///
    /// let zeros = Matrix::<3, 3>::zeros();
    /// ```
    pub fn zeros() -> Self {
        Self {
            data: vec![vec![0.0; C]; R],
        }
    }

    /// Creates an identity matrix.
    ///
    /// # Example
    /// ```
    /// use axb::matrix::Matrix;
    ///
    /// let identity = Matrix::<3, 3>::identity();
    /// assert_eq!(identity[(0, 0)], 1.0);
    /// assert_eq!(identity[(0, 1)], 0.0);
    /// ```
    pub fn identity() -> Self {
        let mut data = vec![vec![0.0; C]; R];
        for i in 0..R.min(C) {
            data[i][i] = 1.0;
        }
        Self { data }
    }

    /// Creates a diagonal matrix from a slice of values.
    ///
    /// # Arguments
    /// * `values` - Diagonal values. Length should be min(R, C).
    ///
    /// # Example
    /// ```
    /// use axb::matrix::Matrix;
    ///
    /// let diag = Matrix::<3, 3>::diagonal(&[1.0, 2.0, 3.0]);
    /// ```
    pub fn diagonal(values: &[f32]) -> Self {
        let mut data = vec![vec![0.0; C]; R];
        for (i, &val) in values.iter().enumerate().take(R.min(C)) {
            data[i][i] = val;
        }
        Self { data }
    }

    /// Creates a matrix filled with a single value.
    ///
    /// # Arguments
    /// * `value` - The value to fill the matrix with
    pub fn fill(value: f32) -> Self {
        Self {
            data: vec![vec![value; C]; R],
        }
    }
}

// ============================================================================
// Accessors
// ============================================================================

impl<const R: usize, const C: usize> Matrix<R, C> {
    /// Returns the number of rows.
    #[inline]
    pub const fn rows(&self) -> usize {
        R
    }

    /// Returns the number of columns.
    #[inline]
    pub const fn cols(&self) -> usize {
        C
    }

    /// Returns the dimensions as a tuple (rows, cols).
    #[inline]
    pub const fn shape(&self) -> (usize, usize) {
        (R, C)
    }

    /// Returns a reference to the element at (row, col).
    ///
    /// # Panics
    /// Panics if indices are out of bounds.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row][col]
    }

    /// Sets the element at (row, col) to the given value.
    ///
    /// # Panics
    /// Panics if indices are out of bounds.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[row][col] = value;
    }

    /// Returns a reference to the underlying data.
    pub fn as_slice(&self) -> &Vec<Vec<f32>> {
        &self.data
    }

    /// Returns a mutable reference to the underlying data.
    pub fn as_mut_slice(&mut self) -> &mut Vec<Vec<f32>> {
        &mut self.data
    }

    /// Returns a specific row as a slice.
    pub fn row(&self, idx: usize) -> &[f32] {
        &self.data[idx]
    }

    /// Returns a specific column as a Vec.
    pub fn col(&self, idx: usize) -> Vec<f32> {
        self.data.iter().map(|row| row[idx]).collect()
    }
}

// ============================================================================
// Basic Operations
// ============================================================================

impl<const R: usize, const C: usize> Matrix<R, C> {
    /// Transposes the matrix.
    ///
    /// # Returns
    /// A new matrix with dimensions C×R where element (i,j) becomes (j,i).
    ///
    /// # Example
    /// ```
    /// use axb::matrix::Matrix;
    ///
    /// let m = Matrix::<2, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// let t = m.transpose();
    /// assert_eq!(t.shape(), (3, 2));
    /// ```
    pub fn transpose(&self) -> Matrix<C, R> {
        let mut data = vec![vec![0.0; R]; C];
        for i in 0..R {
            for j in 0..C {
                data[j][i] = self.data[i][j];
            }
        }
        Matrix::<C, R>::from_vec(data)
    }

    /// Prints the matrix to stdout with aligned columns.
    pub fn print(&self) {
        for row in &self.data {
            for el in row {
                if *el < 0.0 {
                    print!("{:8.4}", el);
                } else {
                    print!("{:8.4}", el);
                }
            }
            println!();
        }
    }

    /// Computes the trace of the matrix (sum of diagonal elements).
    ///
    /// # Returns
    /// The sum of diagonal elements.
    ///
    /// # Example
    /// ```
    /// use axb::matrix::Matrix;
    ///
    /// let m = Matrix::<3, 3>::new([
    ///     [1.0, 2.0, 3.0],
    ///     [4.0, 5.0, 6.0],
    ///     [7.0, 8.0, 9.0],
    /// ]);
    /// assert_eq!(m.trace(), 15.0); // 1 + 5 + 9
    /// ```
    pub fn trace(&self) -> f32 {
        let mut sum = 0.0;
        for i in 0..R.min(C) {
            sum += self.data[i][i];
        }
        sum
    }

    /// Computes the Frobenius norm of the matrix.
    ///
    /// The Frobenius norm is the square root of the sum of squares of all elements.
    ///
    /// # Example
    /// ```
    /// use axb::matrix::Matrix;
    ///
    /// let m = Matrix::<2, 2>::new([[3.0, 0.0], [4.0, 0.0]]);
    /// assert!((m.frobenius_norm() - 5.0).abs() < 1e-6);
    /// ```
    pub fn frobenius_norm(&self) -> f32 {
        let mut sum = 0.0;
        for row in &self.data {
            for &val in row {
                sum += val * val;
            }
        }
        sum.sqrt()
    }

    /// Checks if the matrix is symmetric.
    ///
    /// A matrix is symmetric if A = A^T.
    pub fn is_symmetric(&self) -> bool {
        if R != C {
            return false;
        }
        for i in 0..R {
            for j in (i + 1)..C {
                if (self.data[i][j] - self.data[j][i]).abs() > EPSILON {
                    return false;
                }
            }
        }
        true
    }

    /// Checks if the matrix is orthogonal.
    ///
    /// A matrix is orthogonal if A * A^T = I.
    pub fn is_orthogonal(&self) -> bool {
        if R != C {
            return false;
        }
        let transpose = self.transpose();
        if let Ok(product) = mult(&self.data, &transpose.data) {
            for i in 0..R {
                for j in 0..C {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    if (product[i][j] - expected).abs() > 1e-5 {
                        return false;
                    }
                }
            }
            true
        } else {
            false
        }
    }
}

// ============================================================================
// Determinant & Inverse (Square Matrix Operations)
// ============================================================================

impl<const R: usize, const C: usize> Matrix<R, C> {
    /// Computes the determinant of the matrix.
    ///
    /// Uses Gaussian elimination to convert to upper triangular form,
    /// then multiplies diagonal elements.
    ///
    /// # Returns
    /// * `Ok(f32)` - The determinant value
    /// * `Err(MatrixError::NonSquare)` - If the matrix is not square
    ///
    /// # Example
    /// ```
    /// use axb::matrix::Matrix;
    ///
    /// let m = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert!((m.determinant().unwrap() - (-2.0)).abs() < 1e-6);
    /// ```
    pub fn determinant(&self) -> MatrixResult<f32> {
        if C != R {
            return Err(MatrixError::NonSquare { rows: R, cols: C });
        }

        let mut upper = self.data.clone();
        let n = R;
        let mut sign = 1.0;

        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if upper[k][i].abs() > upper[max_row][i].abs() {
                    max_row = k;
                }
            }

            // Swap rows if needed
            if max_row != i {
                upper.swap(i, max_row);
                sign *= -1.0;
            }

            let pivot = upper[i][i];
            if is_near_zero(pivot) {
                return Ok(0.0);
            }

            for j in (i + 1)..n {
                let factor = upper[j][i] / pivot;
                for k in i..n {
                    upper[j][k] -= factor * upper[i][k];
                }
            }
        }

        let mut det = sign;
        for i in 0..n {
            det *= upper[i][i];
        }
        Ok(det)
    }

    /// Computes the inverse of the matrix using Gauss-Jordan elimination.
    ///
    /// # Returns
    /// * `Ok(Matrix)` - The inverse matrix
    /// * `Err(MatrixError::NonSquare)` - If the matrix is not square
    /// * `Err(MatrixError::NotInvertible)` - If the matrix is singular
    ///
    /// # Example
    /// ```
    /// use axb::matrix::Matrix;
    ///
    /// let m = Matrix::<2, 2>::new([[4.0, 7.0], [2.0, 6.0]]);
    /// let inv = m.inverse().unwrap();
    /// ```
    pub fn inverse(&self) -> MatrixResult<Matrix<R, C>> {
        if C != R {
            return Err(MatrixError::NonSquare { rows: R, cols: C });
        }

        let n = R;
        let mut augmented = self.data.clone();

        // Append identity matrix
        for i in 0..n {
            for j in 0..n {
                augmented[i].push(if i == j { 1.0 } else { 0.0 });
            }
        }

        // Forward elimination with partial pivoting
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if augmented[k][i].abs() > augmented[max_row][i].abs() {
                    max_row = k;
                }
            }

            if max_row != i {
                augmented.swap(i, max_row);
            }

            let pivot = augmented[i][i];
            if is_near_zero(pivot) {
                return Err(MatrixError::NotInvertible);
            }

            // Scale pivot row
            for j in 0..(2 * n) {
                augmented[i][j] /= pivot;
            }

            // Eliminate column
            for j in 0..n {
                if j != i {
                    let factor = augmented[j][i];
                    for k in 0..(2 * n) {
                        augmented[j][k] -= factor * augmented[i][k];
                    }
                }
            }
        }

        // Extract inverse
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            result.push(augmented[i][n..].to_vec());
        }

        Ok(Matrix::<R, C>::from_vec(result))
    }

    /// Solves the linear system Ax = b.
    ///
    /// # Arguments
    /// * `b` - The right-hand side vector (as a column matrix)
    ///
    /// # Returns
    /// * `Ok(Matrix)` - The solution vector x
    /// * `Err(MatrixError)` - If the system cannot be solved
    ///
    /// # Example
    /// ```
    /// use axb::matrix::Matrix;
    ///
    /// let a = Matrix::<2, 2>::new([[2.0, 1.0], [1.0, 3.0]]);
    /// let b = Matrix::<2, 1>::new([[5.0], [10.0]]);
    /// let x = a.solve(&b).unwrap();
    /// ```
    pub fn solve(&self, b: &Matrix<R, 1>) -> MatrixResult<Matrix<R, 1>> {
        if R != C {
            return Err(MatrixError::NonSquare { rows: R, cols: C });
        }

        let inv = self.inverse()?;
        let result = mult(&inv.data, &b.data)?;
        Ok(Matrix::<R, 1>::from_vec(result))
    }
}

// ============================================================================
// Matrix Factorizations
// ============================================================================

impl<const R: usize, const C: usize> Matrix<R, C> {
    /// Computes the QR factorization using Modified Gram-Schmidt.
    ///
    /// # Returns
    /// * `Ok((Q, R))` - Where Q is orthogonal and R is upper triangular
    /// * `Err(MatrixError)` - If factorization fails
    ///
    /// # Example
    /// ```
    /// use axb::matrix::Matrix;
    ///
    /// let m = Matrix::<3, 3>::new([
    ///     [1.0, 2.0, 3.0],
    ///     [4.0, 5.0, 6.0],
    ///     [7.0, 8.0, 10.0],
    /// ]);
    /// let (q, r) = m.qr().unwrap();
    /// ```
    #[allow(non_snake_case)]
    pub fn qr(&self) -> MatrixResult<(Matrix<R, C>, Matrix<C, C>)> {
        let m = R;
        let n = C;

        let mut q: Vec<Vec<f32>> = vec![vec![0.0; n]; m];
        let mut r: Vec<Vec<f32>> = vec![vec![0.0; n]; n];

        for j in 0..n {
            // Copy column j of A into q
            for i in 0..m {
                q[i][j] = self.data[i][j];
            }

            // Orthogonalize against previous columns
            for k in 0..j {
                let dot: f32 = (0..m).map(|i| q[i][j] * q[i][k]).sum();
                r[k][j] = dot;
                for i in 0..m {
                    q[i][j] -= dot * q[i][k];
                }
            }

            // Normalize
            let norm: f32 = (0..m).map(|i| q[i][j] * q[i][j]).sum::<f32>().sqrt();
            if is_near_zero(norm) {
                return Err(MatrixError::NumericalInstability);
            }
            r[j][j] = norm;
            for i in 0..m {
                q[i][j] /= norm;
            }
        }

        Ok((Matrix::<R, C>::from_vec(q), Matrix::<C, C>::from_vec(r)))
    }

    /// Computes the LU factorization (without pivoting).
    ///
    /// # Returns
    /// * `Ok((L, U))` - Where L is lower triangular with 1s on diagonal, U is upper triangular
    /// * `Err(MatrixError)` - If factorization fails
    #[allow(non_snake_case)]
    pub fn lu(&self) -> MatrixResult<(Matrix<R, C>, Matrix<R, C>)> {
        if R != C {
            return Err(MatrixError::NonSquare { rows: R, cols: C });
        }

        let n = R;
        let mut l = vec![vec![0.0; n]; n];
        let mut u = vec![vec![0.0; n]; n];

        for i in 0..n {
            // Upper triangular
            for k in i..n {
                let sum: f32 = (0..i).map(|j| l[i][j] * u[j][k]).sum();
                u[i][k] = self.data[i][k] - sum;
            }

            // Lower triangular
            for k in i..n {
                if i == k {
                    l[i][i] = 1.0;
                } else {
                    let sum: f32 = (0..i).map(|j| l[k][j] * u[j][i]).sum();
                    if is_near_zero(u[i][i]) {
                        return Err(MatrixError::NumericalInstability);
                    }
                    l[k][i] = (self.data[k][i] - sum) / u[i][i];
                }
            }
        }

        Ok((Matrix::<R, C>::from_vec(l), Matrix::<R, C>::from_vec(u)))
    }

    /// Computes the LDU factorization.
    ///
    /// # Returns
    /// * `Ok((L, D, U))` - Where L is unit lower triangular, D is diagonal, U is unit upper triangular
    /// * `Err(MatrixError)` - If factorization fails
    #[allow(non_snake_case)]
    pub fn ldu(&self) -> MatrixResult<(Matrix<R, C>, Matrix<R, C>, Matrix<R, C>)> {
        if R != C {
            return Err(MatrixError::NonSquare { rows: R, cols: C });
        }

        let n = R;
        let mut work = self.data.clone();
        let mut l = vec![vec![0.0; n]; n];
        let mut d = vec![vec![0.0; n]; n];
        let mut u = vec![vec![0.0; n]; n];

        // Initialize diagonal of L and U to 1
        for i in 0..n {
            l[i][i] = 1.0;
            u[i][i] = 1.0;
        }

        for k in 0..n {
            d[k][k] = work[k][k];
            if is_near_zero(d[k][k]) {
                return Err(MatrixError::NumericalInstability);
            }

            for i in (k + 1)..n {
                l[i][k] = work[i][k] / d[k][k];
                u[k][i] = work[k][i] / d[k][k];
            }

            for i in (k + 1)..n {
                for j in (k + 1)..n {
                    work[i][j] -= l[i][k] * d[k][k] * u[k][j];
                }
            }
        }

        Ok((
            Matrix::<R, C>::from_vec(l),
            Matrix::<R, C>::from_vec(d),
            Matrix::<R, C>::from_vec(u),
        ))
    }

    /// Computes the PLU factorization (LU with partial pivoting).
    ///
    /// This is more numerically stable than plain LU decomposition and works
    /// for matrices where regular LU would fail due to zero pivots.
    ///
    /// # Returns
    /// * `Ok((P, L, U))` - Where P is permutation matrix, L is lower triangular, U is upper triangular
    /// * `Err(MatrixError)` - If matrix is not square or singular
    ///
    /// # Property
    /// PA = LU, so A = P⁻¹LU = PᵀLU (since P is orthogonal)
    ///
    /// # Example
    /// ```
    /// use axb::Matrix;
    ///
    /// // This matrix would fail with regular LU (zero pivot in position [0,0])
    /// let m = Matrix::<3, 3>::new([
    ///     [0.0, 1.0, 2.0],
    ///     [1.0, 2.0, 3.0],
    ///     [2.0, 3.0, 5.0],
    /// ]);
    /// let (p, l, u) = m.plu().unwrap();
    /// // Verify: P * A = L * U
    /// ```
    #[allow(non_snake_case)]
    pub fn plu(&self) -> MatrixResult<(Matrix<R, C>, Matrix<R, C>, Matrix<R, C>)> {
        if R != C {
            return Err(MatrixError::NonSquare { rows: R, cols: C });
        }

        let n = R;
        let mut a = self.data.clone();
        let mut p = vec![vec![0.0; n]; n];
        let mut l = vec![vec![0.0; n]; n];

        // Initialize P as identity
        for i in 0..n {
            p[i][i] = 1.0;
        }

        for k in 0..n {
            // Find pivot (largest absolute value in column k, from row k onwards)
            let mut max_val = a[k][k].abs();
            let mut max_row = k;
            for i in (k + 1)..n {
                if a[i][k].abs() > max_val {
                    max_val = a[i][k].abs();
                    max_row = i;
                }
            }

            // Check for singularity
            if is_near_zero(max_val) {
                return Err(MatrixError::NotInvertible);
            }

            // Swap rows in A, P, and L
            if max_row != k {
                a.swap(k, max_row);
                p.swap(k, max_row);
                // Swap already computed L entries
                for j in 0..k {
                    let tmp = l[k][j];
                    l[k][j] = l[max_row][j];
                    l[max_row][j] = tmp;
                }
            }

            // Compute L and eliminate
            l[k][k] = 1.0;
            for i in (k + 1)..n {
                l[i][k] = a[i][k] / a[k][k];
                for j in k..n {
                    a[i][j] -= l[i][k] * a[k][j];
                }
            }
        }

        // A now contains U
        Ok((
            Matrix::<R, C>::from_vec(p),
            Matrix::<R, C>::from_vec(l),
            Matrix::<R, C>::from_vec(a),
        ))
    }

    /// Computes the Cholesky decomposition for symmetric positive-definite matrices.
    ///
    /// # Returns
    /// * `Ok(L)` - Lower triangular matrix where A = LLᵀ
    /// * `Err(MatrixError)` - If matrix is not square, not symmetric, or not positive-definite
    ///
    /// # Example
    /// ```
    /// use axb::Matrix;
    ///
    /// // Symmetric positive-definite matrix
    /// let m = Matrix::<3, 3>::new([
    ///     [4.0, 2.0, 2.0],
    ///     [2.0, 5.0, 1.0],
    ///     [2.0, 1.0, 6.0],
    /// ]);
    /// let l = m.cholesky().unwrap();
    /// // Verify: L * Lᵀ = A
    /// let lt = l.transpose();
    /// let reconstructed = (&l * &lt).unwrap();
    /// ```
    pub fn cholesky(&self) -> MatrixResult<Matrix<R, C>> {
        if R != C {
            return Err(MatrixError::NonSquare { rows: R, cols: C });
        }

        // Check symmetry
        if !self.is_symmetric() {
            return Err(MatrixError::NotInvertible); // Could add a NotSymmetric variant
        }

        let n = R;
        let mut l = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;

                if j == i {
                    // Diagonal element
                    for k in 0..j {
                        sum += l[j][k] * l[j][k];
                    }
                    let val = self.data[j][j] - sum;
                    if val <= 0.0 {
                        // Not positive definite
                        return Err(MatrixError::NotInvertible);
                    }
                    l[j][j] = val.sqrt();
                } else {
                    // Off-diagonal element
                    for k in 0..j {
                        sum += l[i][k] * l[j][k];
                    }
                    if is_near_zero(l[j][j]) {
                        return Err(MatrixError::NumericalInstability);
                    }
                    l[i][j] = (self.data[i][j] - sum) / l[j][j];
                }
            }
        }

        Ok(Matrix::<R, C>::from_vec(l))
    }
}

// ============================================================================
// Vector Operations (for column vectors)
// ============================================================================

impl<const R: usize> Matrix<R, 1> {
    /// Computes the Euclidean (L2) norm of a vector.
    pub fn norm(&self) -> f32 {
        self.data
            .iter()
            .map(|row| row[0] * row[0])
            .sum::<f32>()
            .sqrt()
    }

    /// Returns a normalized (unit) vector.
    pub fn normalize(&self) -> MatrixResult<Matrix<R, 1>> {
        let n = self.norm();
        if is_near_zero(n) {
            return Err(MatrixError::NumericalInstability);
        }
        let data: Vec<Vec<f32>> = self.data.iter().map(|row| vec![row[0] / n]).collect();
        Ok(Matrix::<R, 1>::from_vec(data))
    }

    /// Computes the dot product with another vector.
    pub fn dot(&self, other: &Matrix<R, 1>) -> f32 {
        (0..R).map(|i| self.data[i][0] * other.data[i][0]).sum()
    }
}

// Specialization for 3D cross product
impl Matrix<3, 1> {
    /// Computes the cross product of two 3D vectors.
    pub fn cross(&self, other: &Matrix<3, 1>) -> Matrix<3, 1> {
        let a = &self.data;
        let b = &other.data;
        Matrix::<3, 1>::new([
            [a[1][0] * b[2][0] - a[2][0] * b[1][0]],
            [a[2][0] * b[0][0] - a[0][0] * b[2][0]],
            [a[0][0] * b[1][0] - a[1][0] * b[0][0]],
        ])
    }
}

// ============================================================================
// Matrix Properties
// ============================================================================

impl<const R: usize, const C: usize> Matrix<R, C> {
    /// Checks if the matrix is a projection matrix (P² = P).
    pub fn is_projection(&self) -> bool {
        if R != C {
            return false;
        }

        // Compute P²
        if let Ok(p_squared) = mult(&self.data, &self.data) {
            // Check if P² = P
            for i in 0..R {
                for j in 0..C {
                    if (p_squared[i][j] - self.data[i][j]).abs() > EPSILON * 100.0 {
                        return false;
                    }
                }
            }
            true
        } else {
            false
        }
    }

    /// Raises the matrix to a power using binary exponentiation.
    ///
    /// # Arguments
    /// * `exp` - The exponent (non-negative integer)
    ///
    /// # Returns
    /// * `Ok(Matrix)` - The resulting matrix A^exp
    /// * `Err(MatrixError::NonSquare)` - If the matrix is not square
    ///
    /// # Example
    /// ```
    /// use axb::matrix::Matrix;
    ///
    /// let m = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let m_squared = m.pow(2).unwrap();
    /// ```
    pub fn pow(&self, exp: u32) -> MatrixResult<Matrix<R, C>> {
        if R != C {
            return Err(MatrixError::NonSquare { rows: R, cols: C });
        }

        if exp == 0 {
            return Ok(Matrix::<R, C>::identity());
        }

        // Binary exponentiation for efficiency
        let mut result = Matrix::<R, C>::identity();
        let mut base = self.clone();
        let mut n = exp;

        while n > 0 {
            if n % 2 == 1 {
                result = Matrix::<R, C>::from_vec(mult(&result.data, &base.data)?);
            }
            base = Matrix::<R, C>::from_vec(mult(&base.data, &base.data)?);
            n /= 2;
        }

        Ok(result)
    }

    /// Computes the rank of the matrix using row echelon form.
    pub fn rank(&self) -> usize {
        let mut work = self.data.clone();
        let mut rank = 0;

        for col in 0..C {
            // Find pivot
            let mut pivot_row = None;
            for row in rank..R {
                if !is_near_zero(work[row][col]) {
                    pivot_row = Some(row);
                    break;
                }
            }

            if let Some(pr) = pivot_row {
                work.swap(rank, pr);

                let pivot = work[rank][col];
                for j in col..C {
                    work[rank][j] /= pivot;
                }

                for row in 0..R {
                    if row != rank && !is_near_zero(work[row][col]) {
                        let factor = work[row][col];
                        for j in col..C {
                            work[row][j] -= factor * work[rank][j];
                        }
                    }
                }

                rank += 1;
            }
        }

        rank
    }
}

// ============================================================================
// Scalar Operations
// ============================================================================

impl<const R: usize, const C: usize> Matrix<R, C> {
    /// Multiplies the matrix by a scalar.
    pub fn scale(&self, scalar: f32) -> Matrix<R, C> {
        let data: Vec<Vec<f32>> = self
            .data
            .iter()
            .map(|row| row.iter().map(|&x| x * scalar).collect())
            .collect();
        Matrix::<R, C>::from_vec(data)
    }

    /// Adds a scalar to all elements.
    pub fn add_scalar(&self, scalar: f32) -> Matrix<R, C> {
        let data: Vec<Vec<f32>> = self
            .data
            .iter()
            .map(|row| row.iter().map(|&x| x + scalar).collect())
            .collect();
        Matrix::<R, C>::from_vec(data)
    }
}

// ============================================================================
// Operator Implementations
// ============================================================================

impl<const R: usize, const C: usize> Index<(usize, usize)> for Matrix<R, C> {
    type Output = f32;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.data[row][col]
    }
}

impl<const R: usize, const C: usize> IndexMut<(usize, usize)> for Matrix<R, C> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.data[row][col]
    }
}

impl<const R: usize, const C: usize> Add for &Matrix<R, C> {
    type Output = MatrixResult<Matrix<R, C>>;

    fn add(self, rhs: Self) -> Self::Output {
        let data: Vec<Vec<f32>> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(row_a, row_b)| {
                row_a
                    .iter()
                    .zip(row_b.iter())
                    .map(|(&a, &b)| a + b)
                    .collect()
            })
            .collect();
        Ok(Matrix::<R, C>::from_vec(data))
    }
}

impl<const R: usize, const C: usize> Sub for &Matrix<R, C> {
    type Output = MatrixResult<Matrix<R, C>>;

    fn sub(self, rhs: Self) -> Self::Output {
        let data: Vec<Vec<f32>> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(row_a, row_b)| {
                row_a
                    .iter()
                    .zip(row_b.iter())
                    .map(|(&a, &b)| a - b)
                    .collect()
            })
            .collect();
        Ok(Matrix::<R, C>::from_vec(data))
    }
}

impl<const R: usize, const C: usize, const K: usize> Mul<&Matrix<C, K>> for &Matrix<R, C> {
    type Output = MatrixResult<Matrix<R, K>>;

    fn mul(self, rhs: &Matrix<C, K>) -> Self::Output {
        let result = mult(&self.data, &rhs.data)?;
        Ok(Matrix::<R, K>::from_vec(result))
    }
}

impl<const R: usize, const C: usize> Neg for &Matrix<R, C> {
    type Output = Matrix<R, C>;

    fn neg(self) -> Self::Output {
        self.scale(-1.0)
    }
}

// Scalar multiplication: Matrix * f32
impl<const R: usize, const C: usize> Mul<f32> for &Matrix<R, C> {
    type Output = Matrix<R, C>;

    fn mul(self, rhs: f32) -> Self::Output {
        self.scale(rhs)
    }
}

// ============================================================================
// Standard Trait Implementations
// ============================================================================

impl<const R: usize, const C: usize> PartialEq for Matrix<R, C> {
    fn eq(&self, other: &Self) -> bool {
        for i in 0..R {
            for j in 0..C {
                if (self.data[i][j] - other.data[i][j]).abs() > EPSILON {
                    return false;
                }
            }
        }
        true
    }
}

impl<const R: usize, const C: usize> Default for Matrix<R, C> {
    fn default() -> Self {
        Self::zeros()
    }
}

impl<const R: usize, const C: usize> fmt::Display for Matrix<R, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Matrix<{}, {}> [", R, C)?;
        for row in &self.data {
            write!(f, "  [")?;
            for (i, &val) in row.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:8.4}", val)?;
            }
            writeln!(f, "]")?;
        }
        write!(f, "]")
    }
}

impl<const R: usize, const C: usize> From<[[f32; C]; R]> for Matrix<R, C> {
    fn from(arr: [[f32; C]; R]) -> Self {
        Self::new(arr)
    }
}

impl<const R: usize, const C: usize> From<Vec<Vec<f32>>> for Matrix<R, C> {
    fn from(v: Vec<Vec<f32>>) -> Self {
        Self::from_vec(v)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Constructor tests
    #[test]
    fn test_new() {
        let m = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 1)], 4.0);
    }

    #[test]
    fn test_zeros() {
        let m = Matrix::<3, 3>::zeros();
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(m[(i, j)], 0.0);
            }
        }
    }

    #[test]
    fn test_identity() {
        let m = Matrix::<3, 3>::identity();
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(m[(i, j)], if i == j { 1.0 } else { 0.0 });
            }
        }
    }

    #[test]
    fn test_diagonal() {
        let m = Matrix::<3, 3>::diagonal(&[1.0, 2.0, 3.0]);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 1)], 2.0);
        assert_eq!(m[(2, 2)], 3.0);
        assert_eq!(m[(0, 1)], 0.0);
    }

    #[test]
    fn test_from_vec_non_square() {
        let v = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let m = Matrix::<2, 3>::from_vec(v);
        assert_eq!(m[(0, 2)], 3.0);
        assert_eq!(m[(1, 0)], 4.0);
    }

    // Basic operations
    #[test]
    fn test_transpose() {
        let m = Matrix::<2, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let t = m.transpose();
        assert_eq!(t.shape(), (3, 2));
        assert_eq!(t[(0, 1)], 4.0);
        assert_eq!(t[(2, 0)], 3.0);
    }

    #[test]
    fn test_trace() {
        let m = Matrix::<3, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        assert_eq!(m.trace(), 15.0);
    }

    #[test]
    fn test_frobenius_norm() {
        let m = Matrix::<2, 2>::new([[3.0, 0.0], [4.0, 0.0]]);
        assert!((m.frobenius_norm() - 5.0).abs() < 1e-6);
    }

    // Determinant tests
    #[test]
    fn test_determinant_2x2() {
        let m = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
        assert!((m.determinant().unwrap() - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_determinant_3x3() {
        let m = Matrix::<3, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]);
        assert!((m.determinant().unwrap() - (-3.0)).abs() < 1e-4);
    }

    #[test]
    fn test_determinant_singular() {
        let m = Matrix::<2, 2>::new([[1.0, 2.0], [2.0, 4.0]]);
        assert!((m.determinant().unwrap()).abs() < 1e-6);
    }

    #[test]
    fn test_determinant_non_square() {
        let m = Matrix::<2, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert!(m.determinant().is_err());
    }

    // Inverse tests
    #[test]
    fn test_inverse_identity() {
        let m = Matrix::<2, 2>::identity();
        let inv = m.inverse().unwrap();
        assert!((inv[(0, 0)] - 1.0).abs() < 1e-6);
        assert!((inv[(1, 1)] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_inverse_2x2() {
        let m = Matrix::<2, 2>::new([[4.0, 7.0], [2.0, 6.0]]);
        let inv = m.inverse().unwrap();
        assert!((inv[(0, 0)] - 0.6).abs() < 1e-6);
        assert!((inv[(0, 1)] - (-0.7)).abs() < 1e-6);
    }

    #[test]
    fn test_inverse_singular() {
        let m = Matrix::<2, 2>::new([[1.0, 2.0], [2.0, 4.0]]);
        assert!(m.inverse().is_err());
    }

    // Power tests
    #[test]
    fn test_pow_zero() {
        let m = Matrix::<2, 2>::new([[2.0, 3.0], [1.0, 4.0]]);
        let p = m.pow(0).unwrap();
        assert_eq!(p, Matrix::<2, 2>::identity());
    }

    #[test]
    fn test_pow_one() {
        let m = Matrix::<2, 2>::new([[2.0, 3.0], [1.0, 4.0]]);
        let p = m.pow(1).unwrap();
        assert_eq!(p, m);
    }

    #[test]
    fn test_pow_squared() {
        let m = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
        let p = m.pow(2).unwrap();
        // [[1,2],[3,4]]^2 = [[7,10],[15,22]]
        assert!((p[(0, 0)] - 7.0).abs() < 1e-6);
        assert!((p[(0, 1)] - 10.0).abs() < 1e-6);
        assert!((p[(1, 0)] - 15.0).abs() < 1e-6);
        assert!((p[(1, 1)] - 22.0).abs() < 1e-6);
    }

    // Operator tests
    #[test]
    fn test_add() {
        let a = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::<2, 2>::new([[5.0, 6.0], [7.0, 8.0]]);
        let c = (&a + &b).unwrap();
        assert_eq!(c[(0, 0)], 6.0);
        assert_eq!(c[(1, 1)], 12.0);
    }

    #[test]
    fn test_sub() {
        let a = Matrix::<2, 2>::new([[5.0, 6.0], [7.0, 8.0]]);
        let b = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
        let c = (&a - &b).unwrap();
        assert_eq!(c[(0, 0)], 4.0);
        assert_eq!(c[(1, 1)], 4.0);
    }

    #[test]
    fn test_mul() {
        let a = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::<2, 2>::new([[5.0, 6.0], [7.0, 8.0]]);
        let c = (&a * &b).unwrap();
        assert_eq!(c[(0, 0)], 19.0);
        assert_eq!(c[(0, 1)], 22.0);
        assert_eq!(c[(1, 0)], 43.0);
        assert_eq!(c[(1, 1)], 50.0);
    }

    #[test]
    fn test_neg() {
        let m = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
        let n = -&m;
        assert_eq!(n[(0, 0)], -1.0);
        assert_eq!(n[(1, 1)], -4.0);
    }

    #[test]
    fn test_scalar_mul() {
        let m = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
        let n = &m * 2.0;
        assert_eq!(n[(0, 0)], 2.0);
        assert_eq!(n[(1, 1)], 8.0);
    }

    // Vector tests
    #[test]
    fn test_vector_norm() {
        let v = Matrix::<3, 1>::new([[3.0], [4.0], [0.0]]);
        assert!((v.norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_vector_dot() {
        let a = Matrix::<3, 1>::new([[1.0], [2.0], [3.0]]);
        let b = Matrix::<3, 1>::new([[4.0], [5.0], [6.0]]);
        assert_eq!(a.dot(&b), 32.0);
    }

    #[test]
    fn test_vector_cross() {
        let a = Matrix::<3, 1>::new([[1.0], [0.0], [0.0]]);
        let b = Matrix::<3, 1>::new([[0.0], [1.0], [0.0]]);
        let c = a.cross(&b);
        assert!((c[(0, 0)] - 0.0).abs() < 1e-6);
        assert!((c[(1, 0)] - 0.0).abs() < 1e-6);
        assert!((c[(2, 0)] - 1.0).abs() < 1e-6);
    }

    // Factorization tests
    #[test]
    fn test_qr() {
        let m = Matrix::<3, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]);
        let (q, r) = m.qr().unwrap();
        // Q should be orthogonal
        assert!(q.is_orthogonal());
        // Q * R should equal original
        let reconstructed = (&q * &r).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!((reconstructed[(i, j)] - m[(i, j)]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn test_lu() {
        let m = Matrix::<3, 3>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
        let (l, u) = m.lu().unwrap();
        // L * U should equal original
        let reconstructed = (&l * &u).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!((reconstructed[(i, j)] - m[(i, j)]).abs() < 1e-4);
            }
        }
    }

    // Property tests
    #[test]
    fn test_is_symmetric() {
        let m = Matrix::<3, 3>::new([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]);
        assert!(m.is_symmetric());

        let n = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
        assert!(!n.is_symmetric());
    }

    #[test]
    fn test_is_projection() {
        // A projection matrix: P = A(A^T A)^-1 A^T for full column rank A
        // Simple example: the identity is a projection
        let identity = Matrix::<2, 2>::identity();
        assert!(identity.is_projection());

        // Zero matrix is also a projection
        let zero = Matrix::<2, 2>::zeros();
        assert!(zero.is_projection());
    }

    #[test]
    fn test_rank() {
        let m = Matrix::<3, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        assert_eq!(m.rank(), 2); // Linearly dependent rows

        let n = Matrix::<2, 2>::identity();
        assert_eq!(n.rank(), 2);
    }

    // Edge cases
    #[test]
    fn test_1x1_matrix() {
        let m = Matrix::<1, 1>::new([[5.0]]);
        assert_eq!(m.determinant().unwrap(), 5.0);
        assert!((m.inverse().unwrap()[(0, 0)] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_display() {
        let m = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
        let s = format!("{}", m);
        assert!(s.contains("Matrix<2, 2>"));
    }

    #[test]
    fn test_equality() {
        let a = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
        let c = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 5.0]]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_plu() {
        // Matrix with zero pivot that would fail regular LU
        let m = Matrix::<3, 3>::new([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 5.0]]);
        let (p, l, u) = m.plu().unwrap();

        // Verify P * A = L * U
        let pa = (&p * &m).unwrap();
        let lu = (&l * &u).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                assert!((pa[(i, j)] - lu[(i, j)]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_plu_identity() {
        let m = Matrix::<3, 3>::identity();
        let (_p, l, u) = m.plu().unwrap();

        // For identity, P=I, L=I, U=I
        for i in 0..3 {
            assert!((l[(i, i)] - 1.0).abs() < 1e-6);
            assert!((u[(i, i)] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_cholesky() {
        // Symmetric positive-definite matrix
        let m = Matrix::<3, 3>::new([[4.0, 2.0, 2.0], [2.0, 5.0, 1.0], [2.0, 1.0, 6.0]]);
        let l = m.cholesky().unwrap();

        // Verify L * L^T = A
        let lt = l.transpose();
        let result = (&l * &lt).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                assert!((result[(i, j)] - m[(i, j)]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_cholesky_not_positive_definite() {
        // Not positive definite (negative eigenvalue)
        let m = Matrix::<2, 2>::new([[1.0, 2.0], [2.0, 1.0]]);
        assert!(m.cholesky().is_err());
    }

    #[test]
    fn test_cholesky_not_symmetric() {
        let m = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
        assert!(m.cholesky().is_err());
    }

    // ========================================================================
    // Edge Case Tests for v1.0.0
    // ========================================================================

    #[test]
    fn test_large_matrix_identity() {
        let m = Matrix::<10, 10>::identity();
        assert_eq!(m.determinant().unwrap(), 1.0);
        assert_eq!(m.rank(), 10);
        assert!(m.is_symmetric());
        assert!(m.is_orthogonal());
    }

    #[test]
    fn test_solve_singular_system() {
        // Singular matrix - no unique solution
        let a = Matrix::<2, 2>::new([[1.0, 2.0], [2.0, 4.0]]);
        let b = Matrix::<2, 1>::new([[1.0], [2.0]]);
        assert!(a.solve(&b).is_err());
    }

    #[test]
    fn test_inverse_times_original_is_identity() {
        let m = Matrix::<3, 3>::new([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]);
        let inv = m.inverse().unwrap();
        let product = (&m * &inv).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((product[(i, j)] - expected).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn test_transpose_twice_is_original() {
        let m = Matrix::<2, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let tt = m.transpose().transpose();
        assert_eq!(m, tt);
    }

    #[test]
    fn test_zero_matrix_properties() {
        let z = Matrix::<3, 3>::zeros();
        assert_eq!(z.determinant().unwrap(), 0.0);
        assert_eq!(z.trace(), 0.0);
        assert_eq!(z.rank(), 0);
        assert!(z.is_symmetric());
        assert!(z.is_projection()); // 0² = 0
    }

    #[test]
    fn test_scalar_multiplication_properties() {
        let m = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
        let scaled = m.scale(2.0);

        // (kA)^T = k(A^T)
        assert_eq!(scaled.transpose(), m.transpose().scale(2.0));

        // det(kA) = k^n * det(A) for n×n matrix
        let det_scaled = scaled.determinant().unwrap();
        let det_original = m.determinant().unwrap();
        assert!((det_scaled - 4.0 * det_original).abs() < 1e-6);
    }

    #[test]
    fn test_matrix_addition_commutativity() {
        let a = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::<2, 2>::new([[5.0, 6.0], [7.0, 8.0]]);
        assert_eq!((&a + &b).unwrap(), (&b + &a).unwrap());
    }

    #[test]
    fn test_matrix_multiplication_associativity() {
        let a = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::<2, 2>::new([[5.0, 6.0], [7.0, 8.0]]);
        let c = Matrix::<2, 2>::new([[9.0, 10.0], [11.0, 12.0]]);

        let ab_c = (&(&a * &b).unwrap() * &c).unwrap();
        let a_bc = (&a * &(&b * &c).unwrap()).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((ab_c[(i, j)] - a_bc[(i, j)]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn test_frobenius_norm_properties() {
        let m = Matrix::<2, 2>::new([[3.0, 4.0], [0.0, 0.0]]);
        assert!((m.frobenius_norm() - 5.0).abs() < 1e-6);

        // ||kA|| = |k| * ||A||
        let scaled = m.scale(2.0);
        assert!((scaled.frobenius_norm() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_pow_larger_exponent() {
        let m = Matrix::<2, 2>::new([[1.0, 1.0], [0.0, 1.0]]);
        let p5 = m.pow(5).unwrap();
        // For upper triangular [[1,1],[0,1]]^n = [[1,n],[0,1]]
        assert!((p5[(0, 1)] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_diagonal_matrix_properties() {
        let d = Matrix::<3, 3>::diagonal(&[2.0, 3.0, 4.0]);
        assert_eq!(d.determinant().unwrap(), 24.0); // 2*3*4
        assert_eq!(d.trace(), 9.0); // 2+3+4
        assert!(d.is_symmetric());
    }

    #[test]
    fn test_indexing() {
        let mut m = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(m[(0, 1)], 2.0);
        m[(0, 1)] = 10.0;
        assert_eq!(m[(0, 1)], 10.0);
    }

    #[test]
    fn test_default_is_zeros() {
        let m: Matrix<3, 3> = Matrix::default();
        assert_eq!(m, Matrix::<3, 3>::zeros());
    }

    #[test]
    fn test_from_vec_of_vecs() {
        let v = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let m: Matrix<2, 2> = v.into();
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 1)], 4.0);
    }
}
