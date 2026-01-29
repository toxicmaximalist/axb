//! Matrix decomposition examples for the axb library

use axb::Matrix;

fn main() {
    println!("=== Matrix Decompositions ===\n");

    let m = Matrix::<3, 3>::new([[4.0, 2.0, 1.0], [2.0, 5.0, 3.0], [1.0, 3.0, 6.0]]);

    println!("Original matrix A:");
    println!("{}", m);
    println!("Determinant: {:.4}", m.determinant().unwrap());

    // =========================================================================
    // LU Decomposition
    // =========================================================================
    println!("\n{}", "=".repeat(50));
    println!("1. LU Decomposition (A = LU)");
    println!("{}", "=".repeat(50));

    match m.lu() {
        Ok((l, u)) => {
            println!("\nL (lower triangular with 1s on diagonal):");
            println!("{}", l);

            println!("U (upper triangular):");
            println!("{}", u);

            println!("Verification: L * U =");
            println!("{}", (&l * &u).unwrap());
        }
        Err(e) => println!("LU decomposition failed: {}", e),
    }

    // =========================================================================
    // PLU Decomposition (LU with Partial Pivoting)
    // =========================================================================
    println!("\n{}", "=".repeat(50));
    println!("2. PLU Decomposition (PA = LU) - More Stable");
    println!("{}", "=".repeat(50));

    // Use a matrix where regular LU might struggle
    let m2 = Matrix::<3, 3>::new([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 5.0]]);

    println!("\nMatrix with zero pivot:");
    println!("{}", m2);

    match m2.plu() {
        Ok((p, l, u)) => {
            println!("P (permutation matrix):");
            println!("{}", p);

            println!("L (lower triangular):");
            println!("{}", l);

            println!("U (upper triangular):");
            println!("{}", u);

            // Verify: P * A = L * U
            let pa = (&p * &m2).unwrap();
            let lu = (&l * &u).unwrap();
            println!("Verification: P * A =");
            println!("{}", pa);
            println!("L * U =");
            println!("{}", lu);
        }
        Err(e) => println!("PLU decomposition failed: {}", e),
    }

    // =========================================================================
    // LDU Decomposition
    // =========================================================================
    println!("\n{}", "=".repeat(50));
    println!("3. LDU Decomposition (A = LDU)");
    println!("{}", "=".repeat(50));

    match m.ldu() {
        Ok((l, d, u)) => {
            println!("\nL (unit lower triangular):");
            println!("{}", l);

            println!("D (diagonal):");
            println!("{}", d);

            println!("U (unit upper triangular):");
            println!("{}", u);

            // Verify: L * D * U = A
            let ld = (&l * &d).unwrap();
            let ldu = (&ld * &u).unwrap();
            println!("Verification: L * D * U =");
            println!("{}", ldu);
        }
        Err(e) => println!("LDU decomposition failed: {}", e),
    }

    // =========================================================================
    // QR Decomposition
    // =========================================================================
    println!("\n{}", "=".repeat(50));
    println!("4. QR Decomposition (A = QR)");
    println!("{}", "=".repeat(50));

    match m.qr() {
        Ok((q, r)) => {
            println!("\nQ (orthogonal matrix):");
            println!("{}", q);

            println!("R (upper triangular):");
            println!("{}", r);

            println!("Verification: Q * R =");
            println!("{}", (&q * &r).unwrap());

            println!("Q^T * Q (should be identity):");
            println!("{}", (&q.transpose() * &q).unwrap());

            println!("Is Q orthogonal? {}", q.is_orthogonal());
        }
        Err(e) => println!("QR decomposition failed: {}", e),
    }

    // =========================================================================
    // Cholesky Decomposition
    // =========================================================================
    println!("\n{}", "=".repeat(50));
    println!("5. Cholesky Decomposition (A = LL^T)");
    println!("{}", "=".repeat(50));

    // Symmetric positive-definite matrix
    let spd = Matrix::<3, 3>::new([[4.0, 2.0, 2.0], [2.0, 5.0, 1.0], [2.0, 1.0, 6.0]]);

    println!("\nSymmetric positive-definite matrix:");
    println!("{}", spd);
    println!("Is symmetric? {}", spd.is_symmetric());

    match spd.cholesky() {
        Ok(l) => {
            println!("\nL (lower triangular):");
            println!("{}", l);

            let lt = l.transpose();
            println!("L^T:");
            println!("{}", lt);

            println!("Verification: L * L^T =");
            println!("{}", (&l * &lt).unwrap());
        }
        Err(e) => println!("Cholesky decomposition failed: {}", e),
    }

    // =========================================================================
    // Applications
    // =========================================================================
    println!("\n{}", "=".repeat(50));
    println!("6. Application: Solving Ax = b using LU");
    println!("{}", "=".repeat(50));

    let b = Matrix::<3, 1>::new([[1.0], [2.0], [3.0]]);
    println!("\nSolving Ax = b where b =");
    println!("{}", b);

    match m.solve(&b) {
        Ok(x) => {
            println!("Solution x:");
            println!("{}", x);

            println!("Verification: A * x =");
            println!("{}", (&m * &x).unwrap());
        }
        Err(e) => println!("Failed to solve: {}", e),
    }

    // =========================================================================
    // Singular Matrix Example
    // =========================================================================
    println!("\n{}", "=".repeat(50));
    println!("7. Handling Singular Matrices");
    println!("{}", "=".repeat(50));

    let singular = Matrix::<3, 3>::new([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0], // Row 3 = Row 1 + Row 2 (linearly dependent)
    ]);

    println!("\nSingular matrix (rank deficient):");
    println!("{}", singular);
    println!("Determinant: {:.6}", singular.determinant().unwrap());
    println!("Rank: {} (should be 2, not 3)", singular.rank());

    match singular.inverse() {
        Ok(_) => println!("Inverse exists (unexpected)"),
        Err(e) => println!("Cannot invert: {} (expected)", e),
    }

    println!("\n=== Decompositions Complete ===");
}
