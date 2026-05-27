// Single integration-test binary for clankers-app. One binary instead of
// N keeps test-link disk well below the ubuntu-latest /tmp ceiling — each
// binary statically links the full clankers-app dep graph (~40 MB
// unstripped). `autotests = false` + the `[[test]]` entry in Cargo.toml
// route every CLI test through this one entry point; submodules below
// resolve normally because this file is the crate root.

mod cli_bench;
mod cli_info_json;
mod cli_inspect;
mod cli_record;
mod cli_replay;
mod cli_run_scenario;
mod cli_serve_protocol;
mod cli_validate;
