# Loop 01: Performance — Dev Profile + Solver Workspace

## Changes

1. **Root Cargo.toml**: Add `[profile.dev.package."*"] opt-level = 2`
   - Optimizes nalgebra, clarabel, rapier in debug builds (~10x speedup)
   - Only affects dependencies, not workspace crates (preserves debug info)

2. **solver.rs**: Pre-allocate workspace matrices in `MpcSolver`
   - New fields: a_qp, b_qp, b_qp_t, sb, p_mat, q_vec, s_diag, a_powers, ab_products
   - `new(config, n_feet)` allocates ~600KB workspace once
   - `solve(&mut self, ...)` reuses workspace — zero allocation for big matrices
   - `fill_prediction_matrices` uses `gemm` + `split_at_mut` for in-place computation
   - `fill_condensed_cost` uses `transpose_to` + `gemm` + `gemv` for in-place cost build
   - `build_constraints` extracted as free function (still allocates — small, variable-size)

3. **Callers updated**: plugin.rs, both examples, test file — pass n_feet=4
