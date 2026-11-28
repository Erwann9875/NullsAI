param(
    [string]$Output = (Join-Path $PSScriptRoot "..\out\nullsroyale-memory-reader"),
    [string]$Python = (Join-Path $PSScriptRoot "..\..\..\.venv\Scripts\python.exe")
)

$ErrorActionPreference = "Stop"
$manifest = Join-Path $PSScriptRoot "Cargo.toml"
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))
$targetRoot = Join-Path $workspace ".runtime\nullsroyale-memory-reader-target"
$outputPath = [IO.Path]::GetFullPath($Output)
$env:CARGO_TARGET_DIR = $targetRoot
$env:RUSTFLAGS = "-C linker=rust-lld -C target-feature=+crt-static -C link-arg=-z -C link-arg=relro -C link-arg=-z -C link-arg=now"

foreach ($required in @($manifest, $Python)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing external-reader build input: $required"
    }
}

& cargo +1.97-x86_64-pc-windows-msvc build `
    --manifest-path $manifest `
    --target x86_64-unknown-linux-musl `
    --release `
    --locked
if ($LASTEXITCODE -ne 0) { throw "Could not compile the external memory reader." }

$built = Join-Path $targetRoot "x86_64-unknown-linux-musl\release\nullsroyale-memory-reader"
if (-not (Test-Path -LiteralPath $built -PathType Leaf)) {
    throw "Cargo did not produce the expected static Linux reader: $built"
}
$verification = & $Python (Join-Path $PSScriptRoot "verify_reader.py") $built
if ($LASTEXITCODE -ne 0) { throw "External reader ELF verification failed." }

New-Item -ItemType Directory -Path (Split-Path -Parent $outputPath) -Force | Out-Null
Copy-Item -LiteralPath $built -Destination $outputPath -Force
$report = $verification | ConvertFrom-Json
$report.output = $outputPath
$report | ConvertTo-Json -Compress
