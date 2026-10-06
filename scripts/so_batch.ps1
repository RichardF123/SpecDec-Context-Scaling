# so_batch.ps1 — 串行 oracle 的**顺序**批量运行器。
#
# 规则（与既有轮次一致）：
#   * **一个进程 = 一个 (case, mode, rep)**（每次调用都新起 python 进程，进程内不共享 cache）；
#   * **顺序**执行（同一块 8GB GPU 不并发驻留两套 target+draft）；
#   * **不自动重试**（失败就保留失败产物与日志，由人决定下一步）；
#   * 控制台输出**逐行**同时写入 per-run 日志（UTF-8）与 batch 日志。
#
# 2026-10-05 增补（**用户本轮明确授权的最小改动**：失败即停 + 分组验收；不改任何数值路径与门限）：
#   ① **失败即停**：任何子进程（运行器或验收器）非 0 退出 ⇒ 保留日志与产物、**立即停止**、返回非 0。
#   ② **退出码 0 不等于通过**：每个 (case, mode) 组的两次重复跑完后，调用
#      `b6_orchestrate_cpu` 做**分组验收**：产物齐备 / 运行完成（stage=complete）/
#      非环境门拒绝 / 硬检查与自洽性逐项取值 / 诊断精度 / 冻结配置 / **臂未被漂移** /
#      跨模式与跨重复一致 / **serial 与 AR 完整序列相等**；任一硬失败 ⇒ STOP。
#   ③ **只记录项**（不算硬失败、不会让批次停，但**用例不得判 PASS**）：
#      production 与 AR 的分歧、声明分支/EOS 未触发（覆盖缺口）、分派规则证据边界。
#   ④ **不覆盖、不静默重跑**：计划内的槽位若已有产物 ⇒ 默认**停**（并打印状态）；
#      要接着跑必须显式 `-Resume`（此时跳过已有槽位，不重跑）。
#   ⑤ **配置来自单一来源**：用例的 `max_new` 与字面 prompt 文本取自 `b6_cases.py`
#      （避免"计划里的用例"与"实际跑的用例"漂移）；非字面用例显式清除 `SO_PROMPT` 残留，
#      并显式设置 `SO_MAXNEW`（避免上一条命令的环境变量被继承）。
#   ⑥ 结束时输出**已完成 / 未完成**清单；默认再跑一次 `b6_branch_coverage_cpu` 做最终验收，
#      批次退出码取该验收器的退出码（**不得**把部分完成写成整体通过）。
#
# 用法：
#   .\scripts\so_batch.ps1 -Tag div8209_b6 -Cases natural-1,math -Modes ar,prod,serial -Reps 1,2
#   .\scripts\so_batch.ps1 -Tag div8209_b6 -Cases F09 -Modes ar,prod,serial -Reps 1,2 `
#       -Declared docs\b6_f09_recheck_declared_2026-10-05.json -RequireFusedPrefill
param(
  [Parameter(Mandatory = $true)][string] $Tag,
  [Parameter(Mandatory = $true)][string[]] $Cases,
  [Parameter(Mandatory = $true)][string[]] $Modes,
  [Parameter(Mandatory = $true)][int[]] $Reps,
  [string] $Declared = '',
  [string] $ExpectArm = 'cudnn_prefill',
  [string] $ExpectBackend = 'CUDNN',
  [int] $DefaultMaxNew = 64,
  [switch] $RequireFusedPrefill,
  [switch] $Resume,
  [switch] $SkipFinalCheck,
  # ⚠️ 下面两项**仅供无 GPU 的编排自测**（`scripts\selftest_so_batch_orchestration.ps1`）使用。
  #    真实运行必须保持默认值：`-RunnerModule specdec.engine.serial_oracle_gpu`、`-LogDir output\g5\attn`。
  #    结构上禁止用假运行器 + 真实 tag 造出"像真实取证的产物"：tag 不以 `SELFTEST` 开头 ⇒ 拒绝（退出码 3）。
  [string] $RunnerModule = 'specdec.engine.serial_oracle_gpu',
  [string] $LogDir = ''
)
$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root '.venv\Scripts\python.exe'
$realRunner = 'specdec.engine.serial_oracle_gpu'
if ($RunnerModule -ne $realRunner -and -not $Tag.StartsWith('SELFTEST')) {
  Write-Host "[refuse] 非默认运行器（$RunnerModule）只允许 SELFTEST* tag（防止伪造真实取证）；退出码 3"
  exit 3
}
if ($LogDir -ne '' -and -not $Tag.StartsWith('SELFTEST')) {
  Write-Host "[refuse] 自定义 -LogDir 只允许 SELFTEST* tag；退出码 3"
  exit 3
}
$logdir = $(if ($LogDir -ne '') { $LogDir } else { Join-Path $root 'output\g5\attn' })
# ⚠️ 参数健全性（实测踩到的**静默**陷阱，2026-10-05）：用 `powershell -File` 调用本脚本时
#    `-Modes ar,prod,serial` 会**整串**绑成一个元素、`-Reps 1,2` 会被当成数字 **12**（逗号被当千分位），
#    两者都**不报错**；只有人类式调用（`.\scripts\so_batch.ps1 -Modes ar,prod,serial -Reps 1,2`）才正确。
#    ⇒ 这里显式拒绝"含逗号的元素"，把静默错误变成显式失败。
$commaArgs = @(@($Modes) + @($Cases) + @($Reps) | Where-Object { "$_" -match ',' })
if ($commaArgs.Count -gt 0) {
  Write-Host "[refuse] 参数被整串绑定（含逗号）：$($commaArgs -join '; ') ⇒ 请用人类式调用；退出码 3"
  exit 3
}
if (@($Reps | Where-Object { $_ -lt 1 }).Count -gt 0) {
  Write-Host "[refuse] 非法 -Reps（必须 ≥1）：$($Reps -join ',')；退出码 3"
  exit 3
}
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$batch = Join-Path $logdir ("{0}_batch_{1}.log" -f $Tag, $stamp)
$summary = @()

function Write-Both([string] $line, [string] $logp) {
  $line | Out-File -FilePath $logp -Append -Encoding utf8
  Write-Host $line
}

"=== BATCH tag=$Tag cases=$($Cases -join ',') modes=$($Modes -join ',') reps=$($Reps -join ',') start=$stamp ===" |
  Out-File -FilePath $batch -Encoding utf8
Write-Host "=== BATCH tag=$Tag cases=$($Cases -join ',') modes=$($Modes -join ',') reps=$($Reps -join ',') ==="

# ---- 逐用例配置（单一来源：b6_cases.py；非字面用例走 manifest）----
$caseTable = @{}
$dump = & $py -X utf8 -c "import json;from specdec.engine import b6_cases as BC;print(json.dumps({c['case_id']:{'text':c['text'],'max_new':c['max_new']} for c in BC.B6_CASES}))"
if ($LASTEXITCODE -eq 0 -and $dump) {
  $obj = $dump | ConvertFrom-Json
  foreach ($p in $obj.PSObject.Properties) { $caseTable[$p.Name] = $p.Value }
}
if ($caseTable.Count -eq 0) {
  Write-Host "[warn] 未能读取 b6_cases.py 用例表 ⇒ 字面用例无法自动注入 prompt（非字面用例不受影响）"
}

# ---- 声明文件里的逐用例 `text`/`max_new`（2026-10-05 新增；用于**非内建**的字面用例，例如 EOS 补测 `eos1`）----
# 口径：**声明文件优先于内建表**；只有声明里明确给了 `text` 才走 literal 模式，否则仍走 manifest（既有行为不变）。
$declTable = @{}
if ($Declared -ne '' -and (Test-Path -LiteralPath $Declared)) {
  try {
    $dj = Get-Content -Raw -Encoding UTF8 -LiteralPath $Declared | ConvertFrom-Json
    foreach ($p in $dj.cases.PSObject.Properties) { $declTable[$p.Name] = $p.Value }
    Write-Host "[decl] 声明文件用例：$(($declTable.Keys | Sort-Object) -join ',')"
  } catch {
    Write-Host "[warn] 声明文件无法解析（$($_.Exception.Message)）⇒ 不注入声明用例的 prompt"
  }
}

# ---- 计划内已有产物？默认停；-Resume 则跳过已有槽位 ----
$existing = @()
foreach ($c in $Cases) { foreach ($m in $Modes) { foreach ($r in $Reps) {
  $hits = @(Get-ChildItem -Path $logdir -Filter ("{0}_{1}_{2}_rep{3}_*.json" -f $Tag, $c, $m, $r) -ErrorAction SilentlyContinue)
  if ($hits.Count -gt 0) { $existing += ("{0}/{1}/rep{2}（{3} 份：{4}）" -f $c, $m, $r, $hits.Count, (($hits | ForEach-Object { $_.Name }) -join ', ')) }
} } }
if ($existing.Count -gt 0) {
  Write-Host "== 计划内**已存在产物** $($existing.Count) 个槽位：不覆盖、不静默重跑 =="
  $existing | ForEach-Object { Write-Host "  [exists] $_" }
  $existing | ForEach-Object { $_ | Out-File -FilePath $batch -Append -Encoding utf8 }
  if (-not $Resume) {
    Write-Host "⇒ 默认**停止**（退出码 2）。确认后可加 -Resume 只补跑未完成槽位，或改用新 tag。"
    "=== BATCH ABORT（计划内已有产物；未加 -Resume）===" | Out-File -FilePath $batch -Append -Encoding utf8
    exit 2
  }
  Write-Host "⇒ -Resume：跳过已有槽位，只跑未完成的。"
}

$stopRc = 0
$stopped = $false
$stopWhy = ''
$orchestrationRc = 0

function Invoke-Orchestrator([string] $case, [string[]] $modes, [string] $logp) {
  $oargs = @('--tag', $Tag, '--case', $case, '--modes', ($modes -join ','),
             '--reps', ($Reps -join ','), '--expect-arm', $ExpectArm,
             '--expect-backend', $ExpectBackend, '--outdir', $logdir)
  if ($Declared -ne '') { $oargs += @('--declared', $Declared) }
  if ($RequireFusedPrefill) { $oargs += @('--require-fused-prefill') }
  # ⚠️ 用 `Write-Host`，**不要**把这些行输出到管道：否则它们会和退出码一起成为函数返回值，
  #    调用方 `if ($orc -ne 0)` 会把"非空数组"误判成失败（实测踩到：分组验收恒报 STOP）。
  & $py -X utf8 -m specdec.engine.b6_orchestrate_cpu @oargs 2>&1 | ForEach-Object {
    $_ | Out-File -FilePath $logp -Append -Encoding utf8
    Write-Host $_
  }
  return $LASTEXITCODE
}

foreach ($c in $Cases) {
  if ($stopped) { break }
  foreach ($m in $Modes) {
    if ($stopped) { break }
    # ---- 逐 run ----
    foreach ($r in $Reps) {
      $env:SO_TAG = $Tag; $env:SO_CASE = $c; $env:SO_MODE = $m; $env:SO_REP = "$r"
      # 输入来源与长度：显式设值，**不继承上一条命令的残留环境变量**
      Remove-Item Env:SMOKE_PROMPT -ErrorAction SilentlyContinue
      Remove-Item Env:SMOKE_PROMPT_ID -ErrorAction SilentlyContinue
      Remove-Item Env:SMOKE_FAMILY -ErrorAction SilentlyContinue
      if ($declTable.ContainsKey($c) -and $declTable[$c].text) {
        # 声明文件里**明确给了 text** ⇒ literal 模式（用于非内建字面用例，如 EOS 补测 eos1）
        $env:SO_PROMPT = [string]$declTable[$c].text
        $env:SO_PROMPT_ID = $(if ($declTable[$c].prompt_id) { [string]$declTable[$c].prompt_id } else { $c })
        $env:SO_MAXNEW = "$($declTable[$c].max_new)"
        $srcDesc = "literal_prompt(declared) id=$($env:SO_PROMPT_ID) max_new=$($declTable[$c].max_new)"
      } elseif ($caseTable.ContainsKey($c)) {
        $env:SO_PROMPT = [string]$caseTable[$c].text
        $env:SO_PROMPT_ID = $c
        $env:SO_MAXNEW = "$($caseTable[$c].max_new)"
        $srcDesc = "literal_prompt id=$c max_new=$($caseTable[$c].max_new)"
      } else {
        Remove-Item Env:SO_PROMPT -ErrorAction SilentlyContinue
        Remove-Item Env:SO_PROMPT_ID -ErrorAction SilentlyContinue
        $env:SO_MAXNEW = "$DefaultMaxNew"
        $srcDesc = "manifest family=$c max_new=$DefaultMaxNew"
      }
      $logp = Join-Path $logdir ("{0}_{1}_{2}_rep{3}.log" -f $Tag, $c, $m, $r)
      $hdr = "=== SO_CASE=$c SO_MODE=$m SO_REP=$r $srcDesc arm=$ExpectArm $(Get-Date -Format o) ==="
      $hdr | Out-File -FilePath $logp -Encoding utf8
      $hdr | Out-File -FilePath $batch -Append -Encoding utf8
      Write-Host $hdr
      $already = @(Get-ChildItem -Path $logdir -Filter ("{0}_{1}_{2}_rep{3}_*.json" -f $Tag, $c, $m, $r) -ErrorAction SilentlyContinue)
      if ($Resume -and $already.Count -gt 0) {
        $line = "SKIP(已存在产物) case=$c mode=$m rep=$r files=$($already.Count)"
        $line | Out-File -FilePath $logp -Append -Encoding utf8
        $line | Out-File -FilePath $batch -Append -Encoding utf8
        Write-Host $line
        $summary += $line
        continue
      }
      & $py -X utf8 -m $RunnerModule 2>&1 | ForEach-Object {
        $_ | Out-File -FilePath $logp -Append -Encoding utf8
        $_
      }
      $rc = $LASTEXITCODE
      $line = "RC=$rc case=$c mode=$m rep=$r"
      $line | Out-File -FilePath $logp -Append -Encoding utf8
      $line | Out-File -FilePath $batch -Append -Encoding utf8
      Write-Host $line
      $summary += $line
      if ($rc -ne 0) {
        # ① 失败即停：**不重试、不替换参数、不覆盖产物**
        $stopRc = $rc; $stopped = $true; $stopWhy = "运行器非 0 退出（case=$c mode=$m rep=$r rc=$rc）"
        $m0 = "=== FAIL-FAST：$stopWhy ⇒ 立即停止（保留日志与产物）==="
        $m0 | Out-File -FilePath $batch -Append -Encoding utf8
        Write-Host $m0
        break
      }
    }
    if ($stopped) { break }
    # ---- ② 分组验收（该 case+mode 的两次重复齐备后）----
    $orc = Invoke-Orchestrator $c @($m) $logp
    "ORCH_GROUP=$orc case=$c mode=$m" | Out-File -FilePath $batch -Append -Encoding utf8
    if ($orc -ne 0) {
      $stopped = $true; $stopRc = 1; $orchestrationRc = $orc
      $stopWhy = "分组验收 STOP（case=$c mode=$m）：见上文 [STOP]/产物与日志"
      $m1 = "=== FAIL-FAST：$stopWhy ==="
      $m1 | Out-File -FilePath $batch -Append -Encoding utf8
      Write-Host $m1
      break
    }
  }
  if ($stopped) { break }
  # ---- ②b case 级验收（全模式齐备后：跨模式一致 + serial 与 AR 完整相等）----
  $logp = Join-Path $logdir ("{0}_{1}_allmodes.log" -f $Tag, $c)
  $hdr2 = "=== ORCH CASE-LEVEL case=$c modes=$($Modes -join ',') $(Get-Date -Format o) ==="
  $hdr2 | Out-File -FilePath $logp -Append -Encoding utf8
  $hdr2 | Out-File -FilePath $batch -Append -Encoding utf8
  Write-Host $hdr2
  $orc2 = Invoke-Orchestrator $c $Modes $logp
  "ORCH_CASE=$orc2 case=$c" | Out-File -FilePath $batch -Append -Encoding utf8
  if ($orc2 -ne 0) {
    $stopped = $true; $stopRc = 1; $orchestrationRc = $orc2
    $stopWhy = "case 级验收 STOP（case=$c）"
    Write-Host "=== FAIL-FAST：$stopWhy ==="
    $stopWhy | Out-File -FilePath $batch -Append -Encoding utf8
  }
}

# ---- ⑥ 已完成 / 未完成清单（照实；部分完成不得写成整体完成）----
$pargs = @('--tag', $Tag, '--progress', '--plan-cases', ($Cases -join ','),
           '--modes', ($Modes -join ','), '--reps', ($Reps -join ','),
           '--outdir', $logdir)
& $py -X utf8 -m specdec.engine.b6_orchestrate_cpu @pargs 2>&1 | ForEach-Object {
  $_ | Out-File -FilePath $batch -Append -Encoding utf8
  $_
}
$progRc = $LASTEXITCODE
"PROGRESS_RC=$progRc" | Out-File -FilePath $batch -Append -Encoding utf8
$finalRc = 0
if (-not $SkipFinalCheck) {
  # ---- 最终验收（B6 分支覆盖 / 严格验收）；**退出码取它** ----
  $cargs = @('--tag', $Tag, '--cases', ($Cases -join ','), '--expect-backend', $ExpectBackend,
             '--outdir', $logdir)
  if ($Declared -ne '') { $cargs += @('--declared', $Declared) }
  & $py -X utf8 -m specdec.engine.b6_branch_coverage_cpu @cargs 2>&1 | ForEach-Object {
    $_ | Out-File -FilePath $batch -Append -Encoding utf8
    $_
  }
  $finalRc = $LASTEXITCODE
  "FINAL_CHECK_RC=$finalRc（0 = 全部 CASE_PASS 且集合级要求满足）" | Out-File -FilePath $batch -Append -Encoding utf8
}
$tot = "=== BATCH END tag=$Tag runs={0} nonzero={1} stopped={2} stop_rc={3} final_check_rc={4} ===" -f `
  $summary.Count, (($summary | Where-Object { $_ -match '^RC=' -and $_ -notmatch 'RC=0 ' }).Count), `
  $stopped, $stopRc, $finalRc
$tot | Out-File -FilePath $batch -Append -Encoding utf8
Write-Host $tot
$summary | ForEach-Object { Write-Host $_ }
if ($stopped) { exit $stopRc }
if ($finalRc -ne 0) { exit $finalRc }
exit 0
