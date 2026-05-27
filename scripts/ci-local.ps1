#!/usr/bin/env pwsh
# Mirror of .github/workflows/ci.yml. Run before pushing to catch CI failures
# locally. Defaults to the fast subset (skips the release-build bench gate
# and per-scenario validate matrix, which together take ~15 min on a clean
# target/).
#
# Python is run in a throwaway venv built from python/requirements-ci.txt so
# missing-deps regressions (the kind that broke main on 2026-05-26) are
# caught locally — your everyday python/.venv has torch etc. installed that
# CI does not.
#
# Usage:
#   scripts/ci-local.ps1                  # fast subset (fmt+clippy+check+test+doc+xtask+python)
#   scripts/ci-local.ps1 -SkipRust        # just python
#   scripts/ci-local.ps1 -SkipPython      # just rust
#   scripts/ci-local.ps1 -IncludeBench    # also build --release + bench-vs-baseline
#   scripts/ci-local.ps1 -RefreshVenv     # rebuild the CI mirror venv

[CmdletBinding()]
param(
    [switch]$SkipRust,
    [switch]$SkipPython,
    [switch]$IncludeBench,
    [switch]$RefreshVenv
)

$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
$failed = @()

function Step {
    param([string]$Name, [scriptblock]$Block)
    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    $sw = [Diagnostics.Stopwatch]::StartNew()
    try {
        & $Block
        if ($LASTEXITCODE -ne 0) { throw "exit $LASTEXITCODE" }
        $sw.Stop()
        Write-Host "    ok ($([int]$sw.Elapsed.TotalSeconds)s)" -ForegroundColor Green
    } catch {
        $sw.Stop()
        Write-Host "    FAIL ($([int]$sw.Elapsed.TotalSeconds)s): $_" -ForegroundColor Red
        $script:failed += $Name
    }
}

Push-Location $root
try {
    if (-not $SkipRust) {
        Step 'cargo fmt --check' { cargo fmt --all --check }
        Step 'cargo clippy (workspace, all-targets, tests, benches)' {
            cargo clippy -j 8 --workspace --all-targets --tests --benches -- -D warnings
        }
        Step 'cargo check (workspace, all-targets)' {
            cargo check -j 8 --workspace --all-targets
        }
        Step 'cargo test (workspace)' { cargo test -j 8 --workspace }
        Step 'cargo doc (RUSTDOCFLAGS=-D warnings)' {
            $prev = $env:RUSTDOCFLAGS
            $env:RUSTDOCFLAGS = '-D warnings'
            try { cargo doc -j 8 --workspace --no-deps }
            finally { $env:RUSTDOCFLAGS = $prev }
        }
        Step 'xtask check-bin-size' { cargo xtask check-bin-size }
    }

    if (-not $SkipPython) {
        $venv = Join-Path $root 'python\.venv-ci'
        $req  = Join-Path $root 'python\requirements-ci.txt'
        $stamp = Join-Path $venv '.req-stamp'

        $needBuild = $RefreshVenv -or
            -not (Test-Path $venv) -or
            -not (Test-Path $stamp) -or
            ((Get-Item $req).LastWriteTime -gt (Get-Item $stamp).LastWriteTime)

        if ($needBuild) {
            Step 'build CI mirror venv (python/.venv-ci)' {
                if (Test-Path $venv) { Remove-Item -Recurse -Force $venv }
                python -m venv $venv
                & (Join-Path $venv 'Scripts\python.exe') -m pip install --quiet --upgrade pip
                & (Join-Path $venv 'Scripts\python.exe') -m pip install --quiet -r $req
                (Get-Item $req).LastWriteTime | Out-File $stamp
            }
        }

        $py = Join-Path $venv 'Scripts\python.exe'
        Push-Location (Join-Path $root 'python')
        try {
            Step 'ruff check' { & $py -m ruff check . }
            Step 'ruff format --check' { & $py -m ruff format --check . }
            Step 'mypy' { & $py -m mypy --config-file pyproject.toml clankers/ }
            Step 'pytest (fast, --ignore=tests/slow)' {
                & $py -m pytest tests/ --ignore=tests/slow -x -q --timeout=60 -m "not slow"
            }
        } finally { Pop-Location }
    }

    if ($IncludeBench) {
        Step 'cargo build --release -p clankers-app' {
            cargo build -j 8 -p clankers-app --release
        }
        Step 'bench vec sweep' {
            $env:RAYON_NUM_THREADS = '8'
            & .\target\release\clankers-app.exe bench vec `
                --envs 1,2,4,8 --runs 3 --warmup-runs 1 --max-steps 500 `
                --csv benches/current_vec.csv
        }
        Step 'compare baseline (15% tolerance)' {
            python scripts/compare_baseline.py `
                benches/current_vec.csv notes/baselines/vec_baseline.csv `
                --tolerance 0.15
        }
    }
} finally { Pop-Location }

Write-Host ""
if ($failed.Count -gt 0) {
    Write-Host "FAILED: $($failed -join ', ')" -ForegroundColor Red
    exit 1
} else {
    Write-Host "All CI checks passed." -ForegroundColor Green
}
