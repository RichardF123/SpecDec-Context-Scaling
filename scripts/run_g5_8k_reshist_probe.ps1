# run_g5_8k_reshist_probe.ps1 -- base_bf16 residual-divergence cache-history probe.
# ASCII only (PowerShell 5.1). One independent process per step, serial, no retry loop.
# Stops immediately on the first non-zero exit.
# Scope: ONLY base_bf16, index 52 / input position 8243. No v_fp8, no backend change,
# no max_new shortening, no generalized 2x2.
$root = 'D:\RealDSH\specdec'
$py   = Join-Path $root '.venv\Scripts\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
if (-not (Test-Path $py)) { Write-Host "!! venv python missing: $py"; exit 9 }

$tag = if ($env:RP_TAG) { $env:RP_TAG } else { 'reshist' }
$env:RP_KV = 'base_bf16'
Write-Host "RUNNER tag=$tag kv=base_bf16"

foreach ($m in @('ar', 'spec', 'compare')) {
  $env:RP_MODE = $m
  $env:RP_TAG  = $tag
  $log = Join-Path $root ('output\g5\attn\{0}_base_bf16_{1}.log' -f $tag, $m)
  Write-Host "=== RUN mode=$m"
  & $py -m specdec.engine.g5_8k_residual_history_probe 2>&1 | Tee-Object -FilePath $log
  $code = $LASTEXITCODE
  Write-Host "=== END mode=$m exit=$code"
  if ($code -ne 0) {
    Write-Host "!! STOP: mode=$m exit=$code (no retry)"
    exit $code
  }
}
Write-Host 'RUNNER ALL DONE'
exit 0
