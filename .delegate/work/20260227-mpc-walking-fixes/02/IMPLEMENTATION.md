# Loop 02 Implementation

## Changes (solver.rs)

1. **Cached Clarabel settings**: `DefaultSettings<f64>` built once in `new()`, cloned per solve
2. **Direct CSC constraint building**: Replaced `build_constraints` (dense DMatrix + scan) with
   `build_constraints_csc` that builds CSC column-by-column. Avoids ~288KB dense allocation +
   scanning ~36K elements to find ~500 non-zeros.
3. **Pre-allocated CSC capacity**: `dmatrix_to_csc_upper_tri` now allocates rowval/nzval with
   `n*(n+1)/2` capacity upfront (7260 for 120×120 P matrix).
4. **Removed dead code**: `dmatrix_to_csc` (full dense→CSC) no longer needed.

## Test results
- `cargo test -p clankers-mpc`: 41/41 passed
- `cargo build -p clankers-examples`: compiles clean
