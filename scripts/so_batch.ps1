# so_batch.ps1 — 串行 oracle 的**顺序**批量运行器。
#
# 规则（与既有轮次一致）：
#   * **一个进程 = 一个 (case, mode, rep)**（每次调用都新起 python 进程，进程内不共享 cache）；
#   * **顺序**执行（同一块 8GB GPU 不并发驻留两套 target+draft）；
#   * **不自动重试**（失败就保留失败产物与日志，由人决定下一步）；
#   * 控制台输出**逐行**同时写入 per-run 日志（UTF-8）与 batch 日志。
#
# 用法：
#   .\scripts\so_batch.ps1 -Tag div8209_so -Cases F13 -Modes ar,prod,serial -Reps 1
param(
  [Parameter(Mandatory = $true)][string] $Tag,
  [Parameter(Mandatory = $true)][string[]] $Cases,
  [Parameter(Mandatory = $true)][string[]] $Modes,
  [Parameter(Mandatory = $true)][int[]] $Reps
)
$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root '.venv\Scripts\python.exe'
$logdir = Join-Path $root 'output\g5\attn'
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$batch = Join-Path $logdir ("{0}_batch_{1}.log" -f $Tag, $stamp)
$summary = @()
"=== BATCH tag=$Tag cases=$($Cases -join ',') modes=$($Modes -join ',') reps=$($Reps -join ',') start=$stamp ===" |
  Out-File -FilePath $batch -Encoding utf8

foreach ($c in $Cases) {
  foreach ($m in $Modes) {
    foreach ($r in $Reps) {
      $env:SO_TAG = $Tag; $env:SO_CASE = $c; $env:SO_MODE = $m; $env:SO_REP = "$r"
      $logp = Join-Path $logdir ("{0}_{1}_{2}_rep{3}.log" -f $Tag, $c, $m, $r)
      $hdr = "=== SO_CASE=$c SO_MODE=$m SO_REP=$r $(Get-Date -Format o) ==="
      $hdr | Out-File -FilePath $logp -Encoding utf8
      $hdr | Out-File -FilePath $batch -Append -Encoding utf8
      Write-Host $hdr
      & $py -X utf8 -m specdec.engine.serial_oracle_gpu 2>&1 | ForEach-Object {
        $_ | Out-File -FilePath $logp -Append -Encoding utf8
        $_
      }
      $rc = $LASTEXITCODE
      $line = "RC=$rc case=$c mode=$m rep=$r"
      $line | Out-File -FilePath $logp -Append -Encoding utf8
      $line | Out-File -FilePath $batch -Append -Encoding utf8
      Write-Host $line
      $summary += $line
    }
  }
}
$tot = "=== BATCH END tag=$Tag runs={0} nonzero={1} ===" -f $summary.Count,
  (($summary | Where-Object { $_ -notmatch 'RC=0 ' }).Count)
$tot | Out-File -FilePath $batch -Append -Encoding utf8
Write-Host $tot
$summary | ForEach-Object { Write-Host $_ }
exit 0
