# run_g5_8k_qk32.ps1 -- QK FP32 minimal-intervention test: 2 groups x 2 KV x 2 modes = 8 full
# 64-token generations. ASCII only (PowerShell 5.1). One independent process per run, serial,
# no retry loop. Stops immediately on the first non-zero exit.
# Order: control first, then intervention (qk32); within each group: base_bf16 then v_fp8,
# each running AR then SpecDec.
$root = 'D:\RealDSH\specdec'
$py   = Join-Path $root '.venv\Scripts\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
if (-not (Test-Path $py)) { Write-Host "!! venv python missing: $py"; exit 9 }

$tag = if ($env:QK_TAG) { $env:QK_TAG } else { 'qk32' }
Write-Host "RUNNER tag=$tag"

foreach ($g in @('control', 'qk32')) {
  foreach ($kv in @('base_bf16', 'v_fp8')) {
    foreach ($m in @('ar', 'spec')) {
      $env:QK_TAG   = $tag
      $env:QK_GROUP = $g
      $env:QK_KV    = $kv
      $env:QK_MODE  = $m
      $log = Join-Path $root ('output\g5\attn\{0}_{1}_{2}_{3}.log' -f $tag, $g, $kv, $m)
      Write-Host "=== RUN group=$g kv=$kv mode=$m"
      & $py -m specdec.engine.g5_8k_qk32_intervention 2>&1 | Tee-Object -FilePath $log
      $code = $LASTEXITCODE
      Write-Host "=== END group=$g kv=$kv mode=$m exit=$code"
      if ($code -ne 0) {
        Write-Host "!! STOP: group=$g kv=$kv mode=$m exit=$code (no retry)"
        exit $code
      }
    }
  }
}
Write-Host 'RUNNER ALL DONE'
exit 0
