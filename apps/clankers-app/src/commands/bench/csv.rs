//! CSV schemas, row types, header-locked writers, and human pretty-printers.
//!
//! Two CSV headers are exposed:
//!
//! - [`CSV_HEADER`] (V1) — 11-column W5 PR4 lock used by the legacy
//!   single-scenario path.
//! - [`CSV_HEADER_V2`] — additive 15-column superset used by every
//!   subcommand body (`vec`, `protocol`, `record`, `mpc`).
//!
//! Both header strings are byte-identical on their shared 11-column
//! prefix so existing tooling keyed on column names keeps working.

use std::fs::OpenOptions;
use std::path::Path;

use serde::Serialize;

// ---------------------------------------------------------------------------
// CSV schema constants
// ---------------------------------------------------------------------------

/// W5 PR4 11-column header — LOCKED. The legacy single-scenario path
/// (`bench [--scenario ...]`) writes this exact byte sequence.
pub const CSV_HEADER: &str = "scenario,steps,runs,max_steps,wall_ms_total,wall_ms_per_run_mean,wall_ms_per_run_std,steps_per_sec_mean,step_us_p50,step_us_p99,notes";

/// W7 PR4 additive 15-column header. First 11 columns are byte-equal to
/// [`CSV_HEADER`] so existing tooling keyed on `(scenario, num_envs)`
/// via header-name lookup keeps working.
pub const CSV_HEADER_V2: &str = "scenario,steps,runs,max_steps,wall_ms_total,wall_ms_per_run_mean,wall_ms_per_run_std,steps_per_sec_mean,step_us_p50,step_us_p99,notes,num_envs,p95_us,dropped_frames,throughput_x";

// ---------------------------------------------------------------------------
// Row types
// ---------------------------------------------------------------------------

/// One row of the W5 PR4 v1 CSV schema. Field order matches
/// [`CSV_HEADER`] exactly.
#[derive(Serialize, Debug, Clone)]
pub(super) struct BenchRow {
    pub scenario: String,
    pub steps: u32,
    pub runs: u32,
    pub max_steps: u32,
    pub wall_ms_total: f64,
    pub wall_ms_per_run_mean: f64,
    pub wall_ms_per_run_std: f64,
    pub steps_per_sec_mean: f64,
    pub step_us_p50: f64,
    pub step_us_p99: f64,
    pub notes: String,
}

/// One row of the W7 PR4 v2 additive CSV schema. The first 11 fields
/// match [`BenchRow`] for byte equality on the shared header prefix.
#[derive(Serialize, Debug, Clone)]
pub(super) struct BenchRowV2 {
    pub scenario: String,
    pub steps: u32,
    pub runs: u32,
    pub max_steps: u32,
    pub wall_ms_total: f64,
    pub wall_ms_per_run_mean: f64,
    pub wall_ms_per_run_std: f64,
    pub steps_per_sec_mean: f64,
    pub step_us_p50: f64,
    pub step_us_p99: f64,
    pub notes: String,
    /// Number of parallel envs (vec/protocol); 0 elsewhere.
    pub num_envs: u32,
    /// 95th percentile per-step latency in microseconds.
    pub p95_us: f64,
    /// Dropped frames during the run (record); 0 elsewhere.
    pub dropped_frames: u64,
    /// Binary-vs-JSON encoding throughput ratio (protocol); 0 elsewhere.
    pub throughput_x: f64,
}

// ---------------------------------------------------------------------------
// CSV IO
// ---------------------------------------------------------------------------

pub(super) fn write_csv_row_v1(path: &Path, row: &BenchRow) -> std::io::Result<()> {
    write_csv_row_inner(path, row, CSV_HEADER)
}

pub(super) fn write_csv_row_v2(path: &Path, row: &BenchRowV2) -> std::io::Result<()> {
    write_csv_row_inner(path, row, CSV_HEADER_V2)
}

fn write_csv_row_inner<T: Serialize>(path: &Path, row: &T, header: &str) -> std::io::Result<()> {
    let file_existed_with_content = path.exists()
        && std::fs::metadata(path)
            .map(|m| m.len() > 0)
            .unwrap_or(false);

    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent)?;
    }

    let file = OpenOptions::new().create(true).append(true).open(path)?;
    let mut wtr = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(file);

    if !file_existed_with_content {
        wtr.write_record(header.split(','))?;
    }
    wtr.serialize(row)?;
    wtr.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Human pretty-print
// ---------------------------------------------------------------------------

pub(super) fn print_human_v1(row: &BenchRow) {
    println!("bench: {}", row.scenario);
    println!("  steps:                {}", row.steps);
    println!("  runs:                 {}", row.runs);
    println!("  max_steps:            {}", row.max_steps);
    println!("  wall_ms_total:        {:.3}", row.wall_ms_total);
    println!("  wall_ms_per_run_mean: {:.3}", row.wall_ms_per_run_mean);
    println!("  wall_ms_per_run_std:  {:.3}", row.wall_ms_per_run_std);
    println!("  steps_per_sec_mean:   {:.1}", row.steps_per_sec_mean);
    println!("  step_us_p50:          {:.2}", row.step_us_p50);
    println!("  step_us_p99:          {:.2}", row.step_us_p99);
    println!("  notes:                {}", row.notes);
}

pub(super) fn print_human_v2(row: &BenchRowV2) {
    println!("bench: {}", row.scenario);
    println!("  steps:                {}", row.steps);
    println!("  runs:                 {}", row.runs);
    println!("  max_steps:            {}", row.max_steps);
    println!("  wall_ms_total:        {:.3}", row.wall_ms_total);
    println!("  wall_ms_per_run_mean: {:.3}", row.wall_ms_per_run_mean);
    println!("  wall_ms_per_run_std:  {:.3}", row.wall_ms_per_run_std);
    println!("  steps_per_sec_mean:   {:.1}", row.steps_per_sec_mean);
    println!("  step_us_p50:          {:.2}", row.step_us_p50);
    println!("  step_us_p99:          {:.2}", row.step_us_p99);
    println!("  num_envs:             {}", row.num_envs);
    println!("  p95_us:               {:.2}", row.p95_us);
    println!("  dropped_frames:       {}", row.dropped_frames);
    println!("  throughput_x:         {:.3}", row.throughput_x);
    println!("  notes:                {}", row.notes);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_header_v1_field_count_unchanged() {
        // 11 columns LOCKED. If this fails we broke the W5 PR4 contract.
        let cols = CSV_HEADER.split(',').count();
        assert_eq!(cols, 11, "schema lock: 11 columns");
    }

    #[test]
    fn csv_header_v1_column_names_locked() {
        let expected = [
            "scenario",
            "steps",
            "runs",
            "max_steps",
            "wall_ms_total",
            "wall_ms_per_run_mean",
            "wall_ms_per_run_std",
            "steps_per_sec_mean",
            "step_us_p50",
            "step_us_p99",
            "notes",
        ];
        let actual: Vec<&str> = CSV_HEADER.split(',').collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn csv_header_v2_extends_v1_by_four_columns() {
        let v1: Vec<&str> = CSV_HEADER.split(',').collect();
        let v2: Vec<&str> = CSV_HEADER_V2.split(',').collect();
        assert_eq!(v2.len(), v1.len() + 4);
        assert_eq!(&v2[..v1.len()], v1.as_slice());
        assert_eq!(
            &v2[v1.len()..],
            ["num_envs", "p95_us", "dropped_frames", "throughput_x"].as_slice(),
        );
    }
}
