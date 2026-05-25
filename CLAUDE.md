## Build & Test

- Always limit parallelism when running cargo: `cargo test -j 24` / `cargo build -j 24`
- Never use full CPU cores (machine has 32) — leave headroom for the OS
- **Exception:** `cargo build --workspace` (link-heavy step that mmaps large rlibs + runs LLVM codegen) can exhaust the Windows paging file at `-j 24`. Use `-j 8` for full-workspace builds: `cargo build -j 8 --workspace`. Per-crate builds and all `check`/`test`/`clippy` invocations stay at `-j 24`.

## Delegate

This project uses Delegate for spec-driven development.

**Commands:**
| Command | Purpose |
|---------|---------|
| `/dg:study [model] [theme]` | SITR cycles → TASKs in `.delegate/study/` |
| `/dg:work {stump}` | Execute TASK → loops in `.delegate/work/` |

**Workflow:**
1. `/dg:study auth` — explores codebase, produces TASK in `.delegate/study/{stump}/`
2. `/dg:work {stump}` — implements TASK as loops in `.delegate/work/`

**Output:**
```
.delegate/
├── study/{stump}/    # S.md, I.md, T.md, TASK.md
├── work/{stump}/     # TASK.md, LOOPS.yaml, 01/, 02/...
├── templates/        # Cloned repos, patterns
└── doc/              # Auto-generated docs
```
