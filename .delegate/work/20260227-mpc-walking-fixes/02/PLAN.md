# Loop 02: Reduce MPC solver Clarabel overhead

## Context

Clarabel API does NOT support: in-place CscMatrix updates, warm-starting,
or solver instance reuse. Each solve must create a fresh solver.

The actual optimizable overhead is:
1. Settings rebuilt each call (trivial but wasteful)
2. Constraint matrix: dense DMatrix::zeros(~300, 120) allocated then scanned
   to build CSC. The matrix is ~98% zeros — scanning 36K elements to find ~500
   non-zeros is wasteful. Build CSC directly instead.
3. P matrix CSC conversion: scans 14K dense entries. Pre-allocate vecs.

## Changes

### 1. Cache DefaultSettings in MpcSolver (solver.rs)
- Build once in `new()`, store as field
- Clone per solve (cheap: it's a small struct of scalars)

### 2. Build constraint CSC directly (solver.rs)
- Replace `build_constraints` returning `(DMatrix, DVector, usize, usize)`
  with `build_constraints_csc` returning `(CscMatrix, Vec<f64>, usize, usize)`
- Build column-by-column directly into CSC vectors
- Avoids 288KB dense allocation + full scan

### 3. Pre-allocate P CSC vectors with capacity (solver.rs)
- In `dmatrix_to_csc_upper_tri`, pre-allocate rowval/nzval with known capacity
  (n*(n+1)/2 for upper triangle of n×n)

## Files modified
- `crates/clankers-mpc/src/solver.rs`
