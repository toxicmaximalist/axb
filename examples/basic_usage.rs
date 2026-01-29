//! Basic usage examples for the axb library

use axb::Matrix;

fn main() {
    println!("=== axb Linear Algebra Library Demo ===\n");

    // -------------------------------------------------------------------------
    // 1. Creating Matrices
    // -------------------------------------------------------------------------
    println!("1. Creating Matrices");
    println!("{}", "-".repeat(50));

    // From a 2D array
    let a = Matrix::<3, 3>::new([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 10.0],
    ]);
    println!("Matrix A (from array):");
    println!("{}", a);

    // Special constructors
    let identity = Matrix::<3, 3>::identity();
    println!("Identity matrix:");
    println!("{}", identity);

    let zeros = Matrix::<2, 3>::zeros();
    println!("Zero matrix (2x3):");
    println!("{}", zeros);

    let diagonal = Matrix::<3, 3>::diagonal(&[1.0, 2.0, 3.0]);
    println!("Diagonal matrix:");
    println!("{}", diagonal);

    // -------------------------------------------------------------------------
    // 2. Basic Properties
    // -------------------------------------------------------------------------
    println!("\n2. Basic Properties");
    println!("{}", "-".repeat(50));

    println!("Matrix A:");
    println!("{}", a);
    println!("Dimensions: {:?}", a.shape());
    println!("Determinant: {:.4}", a.determinant().unwrap());
    println!("Trace: {:.4}", a.trace());
    println!("Rank: {}", a.rank());
    println!("Frobenius norm: {:.4}", a.frobenius_norm());

    // -------------------------------------------------------------------------
    // 3. Transpose
    // -------------------------------------------------------------------------
    println!("\n3. Transpose");
    println!("{}", "-".repeat(50));

    let rect = Matrix::<2, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    println!("Original (2x3):");
    println!("{}", rect);
    println!("Transposed (3x2):");
    println!("{}", rect.transpose());

    // -------------------------------------------------------------------------
    // 4. Arithmetic Operations
    // -------------------------------------------------------------------------
    println!("\n4. Arithmetic Operations");
    println!("{}", "-".repeat(50));

    let m1 = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
    let m2 = Matrix::<2, 2>::new([[5.0, 6.0], [7.0, 8.0]]);

    println!("M1:");
    println!("{}", m1);
    println!("M2:");
    println!("{}", m2);

    println!("M1 + M2:");
    println!("{}", (&m1 + &m2).unwrap());

    println!("M1 - M2:");
    println!("{}", (&m1 - &m2).unwrap());

    println!("M1 * M2 (matrix multiplication):");
    println!("{}", (&m1 * &m2).unwrap());

    println!("M1 * 2.0 (scalar multiplication):");
    println!("{}", &m1 * 2.0);

    println!("-M1 (negation):");
    println!("{}", -&m1);

    // -------------------------------------------------------------------------
    // 5. Inverse
    // -------------------------------------------------------------------------
    println!("\n5. Matrix Inverse");
    println!("{}", "-".repeat(50));

    println!("M1:");
    println!("{}", m1);

    match m1.inverse() {
        Ok(inv) => {
            println!("M1 inverse:");
            println!("{}", inv);
            println!("M1 * M1^(-1) (should be identity):");
            println!("{}", (&m1 * &inv).unwrap());
        }
        Err(e) => println!("Error computing inverse: {}", e),
    }

    // Singular matrix example
    let singular = Matrix::<2, 2>::new([[1.0, 2.0], [2.0, 4.0]]);
    println!("\nSingular matrix:");
    println!("{}", singular);
    match singular.inverse() {
        Ok(_) => println!("Has inverse (unexpected)"),
        Err(e) => println!("Error (expected): {}", e),
    }

    // -------------------------------------------------------------------------
    // 6. Matrix Power
    // -------------------------------------------------------------------------
    println!("\n6. Matrix Power (Binary Exponentiation)");
    println!("{}", "-".repeat(50));

    let fib = Matrix::<2, 2>::new([[1.0, 1.0], [1.0, 0.0]]);
    println!("Fibonacci matrix F:");
    println!("{}", fib);

    println!("F^10 (contains Fibonacci numbers):");
    println!("{}", fib.pow(10).unwrap());

    // -------------------------------------------------------------------------
    // 7. Solving Linear Systems
    // -------------------------------------------------------------------------
    println!("\n7. Solving Linear Systems (Ax = b)");
    println!("{}", "-".repeat(50));

    let coef = Matrix::<2, 2>::new([[2.0, 1.0], [1.0, 3.0]]);
    let b = Matrix::<2, 1>::new([[5.0], [10.0]]);

    println!("Coefficient matrix A:");
    println!("{}", coef);
    println!("Right-hand side b:");
    println!("{}", b);

    match coef.solve(&b) {
        Ok(x) => {
            println!("Solution x:");
            println!("{}", x);
            println!("Verification A*x:");
            println!("{}", (&coef * &x).unwrap());
        }
        Err(e) => println!("Error: {}", e),
    }

    // -------------------------------------------------------------------------
    // 8. Vector Operations
    // -------------------------------------------------------------------------
    println!("\n8. Vector Operations");
    println!("{}", "-".repeat(50));

    let v1 = Matrix::<3, 1>::new([[1.0], [2.0], [3.0]]);
    let v2 = Matrix::<3, 1>::new([[4.0], [5.0], [6.0]]);

    println!("v1: [{}, {}, {}]", v1[(0, 0)], v1[(1, 0)], v1[(2, 0)]);
    println!("v2: [{}, {}, {}]", v2[(0, 0)], v2[(1, 0)], v2[(2, 0)]);
    println!("||v1|| (norm): {:.4}", v1.norm());
    println!("v1 · v2 (dot product): {:.4}", v1.dot(&v2));

    let cross = v1.cross(&v2);
    println!(
        "v1 × v2 (cross product): [{}, {}, {}]",
        cross[(0, 0)],
        cross[(1, 0)],
        cross[(2, 0)]
    );

    // -------------------------------------------------------------------------
    // 9. Matrix Properties
    // -------------------------------------------------------------------------
    println!("\n9. Matrix Properties");
    println!("{}", "-".repeat(50));

    let sym = Matrix::<3, 3>::new([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]);
    println!("Symmetric matrix:");
    println!("{}", sym);
    println!("Is symmetric: {}", sym.is_symmetric());

    println!("\nIdentity matrix:");
    println!("Is orthogonal: {}", identity.is_orthogonal());

    let proj = Matrix::<2, 2>::zeros();
    println!("\nZero matrix:");
    println!("Is projection (P² = P): {}", proj.is_projection());

    // -------------------------------------------------------------------------
    // 10. Indexing
    // -------------------------------------------------------------------------
    println!("\n10. Indexing");
    println!("{}", "-".repeat(50));

    let mut m = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
    println!("Original matrix:");
    println!("{}", m);
    println!("Element at (0, 1): {}", m[(0, 1)]);

    m[(0, 1)] = 99.0;
    println!("After setting (0, 1) = 99:");
    println!("{}", m);

    println!("\n=== Demo Complete ===");
}
