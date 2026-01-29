use super::errors::{MatrixError, MatrixResult, EPSILON};

/// Multiplies two matrices represented as `Vec<Vec<f32>>`
/// 
/// # Arguments
/// * `a` - Left matrix (rows_a × cols_a)
/// * `b` - Right matrix (rows_b × cols_b)
/// 
/// # Returns
/// * `Ok(Vec<Vec<f32>>)` - Result matrix (rows_a × cols_b)
/// * `Err(MatrixError::DimensionMismatch)` - If cols_a != rows_b
/// * `Err(MatrixError::EmptyMatrix)` - If either matrix is empty
pub fn mult(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>) -> MatrixResult<Vec<Vec<f32>>> {
    if a.is_empty() || b.is_empty() {
        return Err(MatrixError::EmptyMatrix);
    }
    
    let rows_a = a.len();
    let cols_a = a[0].len();
    let rows_b = b.len();
    let cols_b = b[0].len();

    if cols_a != rows_b {
        return Err(MatrixError::DimensionMismatch {
            expected: (cols_a, cols_a),
            got: (rows_b, cols_b),
        });
    }

    let mut result = vec![vec![0.0; cols_b]; rows_a];

    for i in 0..rows_a {
        for j in 0..cols_b {
            for k in 0..cols_a {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    Ok(result)
}

/// Checks if a value is effectively zero within tolerance
#[inline]
pub fn is_near_zero(value: f32) -> bool {
    value.abs() < EPSILON
}