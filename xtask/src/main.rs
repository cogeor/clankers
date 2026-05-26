//! Workspace developer tooling dispatch shell.
//!
//! Subcommands:
//! - `check-bin-size` — enforce LOC tier ceilings on every example bin.

use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args();
    args.next(); // bin name

    match args.next().as_deref() {
        Some("check-bin-size") => match xtask::line_count::run_check_bin_size() {
            Ok(()) => ExitCode::SUCCESS,
            Err(report) => {
                eprintln!("{report}");
                ExitCode::FAILURE
            }
        },
        Some(other) => {
            eprintln!("xtask: unknown subcommand '{other}'");
            eprintln!("Available: check-bin-size");
            ExitCode::from(2)
        }
        None => {
            eprintln!("xtask: missing subcommand");
            eprintln!("Available: check-bin-size");
            ExitCode::from(2)
        }
    }
}
