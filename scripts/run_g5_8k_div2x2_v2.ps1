# run_g5_8k_div2x2_v2.ps1 -- serial fixed-config re-measurement for the 8K divergence 2x2.
# ASCII only (PowerShell 5.1). One independent process per step, serial, no retry loop.
# Stops immediately on the first non-zero exit so a failure can never be reported as success.
$root = 'D:\RealDSH\specdec'
$py   = Join-Path $root '.venv\Scripts\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
if (-not (Test-Path $py)) { Write-Host "!! venv python missing: $py"; exit 9 }

$tag = if ($env:DIV2X2_TAG) { $env:DIV2X2_TAG } else { 'div2x2v2' }
$kvs = if ($env:DIV2X2_KVS) { $env:DIV2X2_KVS.Split(',') } else { @('base_bf16', 'v_fp8') }

Write-Host "RUNNER tag=$tag kvs=$($kvs -join ',')"

foreach ($kv in $kvs) {
  foreach ($mode in @('ar', 'spec', 'compare')) {
    $env:DIAG_KV   = $kv
    $env:DIAG_TAG  = $tag
    $env:DIAG_MODE = $mode
    $log = Join-Path $root ('output\g5\attn\{0}_{1}_{2}.log' -f $tag, $kv, $mode)
    Write-Host "=== RUN kv=$kv mode=$mode"
    & $py -m specdec.engine.g5_8k_divergence_2x2 2>&1 | Tee-Object -FilePath $log
    $code = $LASTEXITCODE
    Write-Host "=== END kv=$kv mode=$mode exit=$code"
    if ($code -ne 0) {
      Write-Host "!! STOP: kv=$kv mode=$mode exit=$code (no retry)"
      exit $code
    }
  }
}
Write-Host 'RUNNER ALL DONE'
exit 0
