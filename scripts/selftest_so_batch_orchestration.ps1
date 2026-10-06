# selftest_so_batch_orchestration.ps1 — `so_batch.ps1` 的**无 GPU 编排自测**（纯 CPU；不加载模型）。
#
# 目的：验证"失败即停 + 分组验收 + 不覆盖/不重跑 + 最终验收退出码"这套编排逻辑**真的成立**，
#       而不是只在模块级自测里成立。
#
# 做法（**不碰真实产物**）：
#   * 用一个**假运行器** `specdec.engine.b6_orchestrate_stub_runner_cpu`（自带拒绝保护：tag 必须以
#     `SELFTEST` 开头、必须显式给 `STUB_ORCH_OUTDIR`）替代真运行器；
#   * `so_batch.ps1` 自身也有结构保护：非默认运行器 / 自定义 -LogDir 只允许 `SELFTEST*` tag ⇒
#     不可能用假运行器往真实 tag 里造出"像真实取证的产物"；
#   * 产物写在临时目录，场景由 `STUB_ORCH_SCENARIO` 选择。
#
# 运行：powershell -NoProfile -ExecutionPolicy Bypass -File scripts\selftest_so_batch_orchestration.ps1
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$scriptPath = Join-Path $PSScriptRoot 'so_batch.ps1'
$py = Join-Path $root '.venv\Scripts\python.exe'
$tmpRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("so_batch_orch_selftest_" + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null

$checks = @()
function Ck([string] $name, [bool] $ok, $extra) {
  $script:checks += [pscustomobject]@{ name = $name; ok = $ok; extra = $extra }
  if (-not $ok) { Write-Host "  [FAIL] $name  $extra" }
}

# ---- 声明文件（合成用例；max_new=16；分支要求两种版本）----
# ⚠️ 合成用例 id **不得**含 `selftest` / `coverage` / `verdict` / `margins` / `aggregate` 等字样：
#    `b6_branch_coverage_cpu.find_runs()` 按**文件名**过滤这些词，含之会被**静默跳过**
#    （实测踩到：产物明明写了，进度却显示"已完成 0"）。
$caseId = 'orccase'
$declBasic = Join-Path $tmpRoot 'decl_basic.json'
$declBranch = Join-Path $tmpRoot 'decl_branch.json'
@{
  cases = @{ $caseId = @{ kind = 'orccase'; max_new = 16; branches = @(); expect_prefill_fused = $null } }
  set_requirements = @{ branches = @(); types = @() }
} | ConvertTo-Json -Depth 6 | Out-File -FilePath $declBasic -Encoding utf8
@{
  cases = @{ $caseId = @{ kind = 'orccase'; max_new = 16; branches = @('all_accept', 'trim_last_round'); expect_prefill_fused = $null } }
  set_requirements = @{ branches = @(); types = @() }
} | ConvertTo-Json -Depth 6 | Out-File -FilePath $declBranch -Encoding utf8

function Run-Scenario {
  param([string] $Name, [string] $Scenario, [string] $Decl, [string] $Arm = 'cudnn_prefill',
        [switch] $RequireFused, [switch] $Resume, [switch] $SkipFinal,
        [string] $TagSuffix = '')
  $dir = Join-Path $tmpRoot ("s_" + $Name)
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  $tag = "SELFTEST_$Name$TagSuffix"
  $env:STUB_ORCH_SCENARIO = $Scenario
  $env:STUB_ORCH_OUTDIR = $dir
  $env:SO_ARM = $Arm
  $env:SO_KV = 'base_bf16'
  # ⚠️ 必须用 `-Command`（人类式调用）而不是 `-File`：`-File` 会把 `ar,prod,serial` 整串绑成一个元素、
  #    把 `1,2` 绑成数字 12，且**不报错**（脚本内已加拒绝保护，见 `so_batch.ps1` 顶部说明）。
  $cmd = "& '$scriptPath' -Tag '$tag' -Cases '$caseId' -Modes ar,prod,serial -Reps 1,2 " +
         "-Declared '$Decl' -LogDir '$dir' -DefaultMaxNew 16 " +
         "-RunnerModule 'specdec.engine.b6_orchestrate_stub_runner_cpu'"
  if ($RequireFused) { $cmd += ' -RequireFusedPrefill' }
  if ($Resume) { $cmd += ' -Resume' }
  if ($SkipFinal) { $cmd += ' -SkipFinalCheck' }
  # ⚠️ 必须在子进程里显式 `exit $LASTEXITCODE`：`-Command "& script"` 会**丢掉**脚本的退出码
  #    （实测：脚本 `exit 3` 而外层只看到 1）；而 `-File` 虽能传退出码却会**静默**错绑数组参数。
  $cmd += '; exit $LASTEXITCODE'
  $log = Join-Path $dir '_console.log'
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -Command $cmd *>&1 |
    Out-File -FilePath $log -Encoding utf8
  $rc = $LASTEXITCODE
  $artifacts = @(Get-ChildItem -Path $dir -Filter ("{0}_{1}_*_rep*_*.json" -f $tag, $caseId) -ErrorAction SilentlyContinue)
  $text = (Get-Content -Raw -Path $log -ErrorAction SilentlyContinue)
  return [pscustomobject]@{ rc = $rc; dir = $dir; tag = $tag; artifacts = $artifacts.Count; text = $text; log = $log; cmd = $cmd }
}

Write-Host "== so_batch.ps1 无 GPU 编排自测（临时目录 $tmpRoot）=="

# 1) 合法完成 ⇒ 退出码 0，6 个产物
$r = Run-Scenario -Name 'ok' -Scenario 'ok' -Decl $declBranch
Ck 'S1 合法完成⇒退出码0' ($r.rc -eq 0) $r.rc
Ck 'S1 合法完成⇒6 个产物' ($r.artifacts -eq 6) $r.artifacts
Ck 'S1 合法完成⇒最终验收 0' ($r.text -match 'FINAL_CHECK_RC=0') $r.rc

# 2) 运行器非 0 退出 ⇒ 立即停、返回该非 0 码（且不继续跑后续组）
$r = Run-Scenario -Name 'rc3' -Scenario 'rc_nonzero' -Decl $declBranch
Ck 'S2 运行器非0⇒脚本返回非0（3）' ($r.rc -eq 3) $r.rc
Ck 'S2 运行器非0⇒fail-fast 标记' ($r.text -match 'FAIL-FAST') $null
Ck 'S2 运行器非0⇒没有产物' ($r.artifacts -eq 0) $r.artifacts

# 3) 退出码 0 但产物硬检查失败 ⇒ 分组验收 STOP（退出码 0 不等于通过）
$r = Run-Scenario -Name 'hardfail' -Scenario 'hardfail' -Decl $declBranch
Ck 'S3 退出码0但硬检查失败⇒STOP(1)' ($r.rc -eq 1) $r.rc
Ck 'S3 硬检查失败类别出现' ($r.text -match 'HARD_CHECK_FAIL') $null

# 4) 退出码 0 但缺 rep2 产物 ⇒ STOP
$r = Run-Scenario -Name 'missing' -Scenario 'missing_rep2' -Decl $declBranch
Ck 'S4 缺产物⇒STOP(1)' ($r.rc -eq 1) $r.rc
Ck 'S4 缺产物类别出现' ($r.text -match 'ARTIFACT_MISSING') $null

# 5) 重复产物 ⇒ STOP（不静默择一）
$r = Run-Scenario -Name 'dup' -Scenario 'duplicate' -Decl $declBranch
Ck 'S5 重复产物⇒STOP(1)' ($r.rc -eq 1) $r.rc
Ck 'S5 重复产物类别出现' ($r.text -match 'ARTIFACT_DUPLICATE') $null

# 6) 同模式跨重复不一致 ⇒ STOP
$r = Run-Scenario -Name 'repmis' -Scenario 'repmismatch' -Decl $declBranch
Ck 'S6 重复不一致⇒STOP(1)' ($r.rc -eq 1) $r.rc
Ck 'S6 重复不一致类别出现' ($r.text -match 'REPEAT_MISMATCH') $null

# 7) serial 与 AR 不一致 ⇒ STOP
$r = Run-Scenario -Name 'sermis' -Scenario 'serialmismatch' -Decl $declBranch
Ck 'S7 serial≠AR⇒STOP(1)' ($r.rc -eq 1) $r.rc
Ck 'S7 serial≠AR 类别出现' ($r.text -match 'SERIAL_VS_AR_MISMATCH') $null

# 8) **只有** production 分歧 ⇒ 不停、只记录，最终验收仍通过
$r = Run-Scenario -Name 'prodonly' -Scenario 'prodonly' -Decl $declBranch
Ck 'S8 仅production分歧⇒继续(0)' ($r.rc -eq 0) $r.rc
Ck 'S8 production 分歧被记录' ($r.text -match 'PRODUCTION_DIVERGENCE') $null
Ck 'S8 6 个产物齐备' ($r.artifacts -eq 6) $r.artifacts

# 9) 臂漂移（SO_ARM=eager 而本批声明 cudnn_prefill）⇒ STOP
$r = Run-Scenario -Name 'armeager' -Scenario 'ok' -Decl $declBranch -Arm 'eager'
Ck 'S9 臂漂移⇒STOP(1)' ($r.rc -eq 1) $r.rc
Ck 'S9 臂漂移类别出现' ($r.text -match 'ARM_MISMATCH') $null

# 10) 候选臂但无融合：不加 -RequireFusedPrefill ⇒ 继续；加 ⇒ STOP（P-B 的"必须证明融合"）
$r = Run-Scenario -Name 'nofuse_n' -Scenario 'nofuse' -Decl $declBranch
Ck 'S10 无融合(不要求)⇒继续(0)' ($r.rc -eq 0) $r.rc
$r = Run-Scenario -Name 'nofuse_y' -Scenario 'nofuse' -Decl $declBranch -RequireFused
Ck 'S10 无融合(要求证明)⇒STOP(1)' ($r.rc -eq 1) $r.rc
Ck 'S10 融合缺失类别出现' ($r.text -match 'FUSED_PREFILL_REQUIRED_BUT_ABSENT') $null

# 11) 声明分支未触发 ⇒ 不停（覆盖缺口）但最终验收 **不得** 判通过 ⇒ 退出码 1
$r = Run-Scenario -Name 'covgap' -Scenario 'covgap' -Decl $declBranch
Ck 'S11 覆盖缺口⇒不停（跑满 6 个产物）' ($r.artifacts -eq 6) $r.artifacts
Ck 'S11 覆盖缺口⇒最终验收非0（不得写成通过）' ($r.rc -eq 1) $r.rc
Ck 'S11 覆盖缺口被记为只记录项' ($r.text -match 'BRANCH_UNCOVERED') $null

# 12) 计划内已有产物 ⇒ 默认停（退出码 2，不覆盖不重跑）；-Resume ⇒ 只补跑未完成
$r = Run-Scenario -Name 'exists' -Scenario 'ok' -Decl $declBranch -SkipFinal
Ck 'S12 首次跑完⇒退出码0' ($r.rc -eq 0) $r.rc
$core2 = "& '$scriptPath' -Tag '$($r.tag)' -Cases '$caseId' -Modes ar,prod,serial -Reps 1,2 " +
         "-Declared '$declBranch' -LogDir '$($r.dir)' -DefaultMaxNew 16 " +
         "-RunnerModule 'specdec.engine.b6_orchestrate_stub_runner_cpu' -SkipFinalCheck"
$log2 = Join-Path $r.dir '_console2.log'
$env:STUB_ORCH_SCENARIO = 'ok'; $env:STUB_ORCH_OUTDIR = $r.dir; $env:SO_ARM = 'cudnn_prefill'
& powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ($core2 + '; exit $LASTEXITCODE') *>&1 |
  Out-File -FilePath $log2 -Encoding utf8
$rc2 = $LASTEXITCODE
$t2 = (Get-Content -Raw -Path $log2)
Ck 'S12 已有产物且未加-Resume⇒停(2)' ($rc2 -eq 2) $rc2
# 只断言**控制台**上可见的 ASCII 标记（`BATCH ABORT` 那行按设计只写 batch 日志）
Ck 'S12 已如实说明已有产物状态' ($t2 -match '\[exists\]') $null
$batchAbort = $false
foreach ($bf in @(Get-ChildItem -Path $r.dir -Filter ("{0}_batch_*.log" -f $r.tag))) {
  if ((Get-Content -Raw -Encoding UTF8 -Path $bf.FullName) -match 'BATCH ABORT') { $batchAbort = $true }
}
Ck 'S12 batch 日志记录了 BATCH ABORT' $batchAbort $null
# -Resume：已有槽位跳过、不重跑（⚠️ -Resume 必须加在 `; exit $LASTEXITCODE` **之前**）
$log3 = Join-Path $r.dir '_console3.log'
& powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ($core2 + ' -Resume; exit $LASTEXITCODE') *>&1 |
  Out-File -FilePath $log3 -Encoding utf8
$rc3 = $LASTEXITCODE
$t3 = (Get-Content -Raw -Path $log3)
$n3 = @(Get-ChildItem -Path $r.dir -Filter ("{0}_{1}_*_rep*_*.json" -f $r.tag, $caseId)).Count
Ck 'S12 -Resume⇒退出码0' ($rc3 -eq 0) $rc3
Ck 'S12 -Resume⇒跳过全部已有槽位、产物数不变' (($t3 -match 'SKIP\(已存在产物\)') -and ($n3 -eq 6)) $n3

# 13) 结构保护：非默认运行器 + 真实风格 tag ⇒ 拒绝（退出码 3）
New-Item -ItemType Directory -Force -Path (Join-Path $tmpRoot 's_refuse') | Out-Null
$cmd4 = "& '$scriptPath' -Tag 'div8209_b6_notrun' -Cases $caseId -Modes ar -Reps 1 " +
        "-Declared '$declBranch' -LogDir '$(Join-Path $tmpRoot 's_refuse')' " +
        "-RunnerModule 'specdec.engine.b6_orchestrate_stub_runner_cpu'" +
        '; exit $LASTEXITCODE'
$log4 = Join-Path $tmpRoot 'refuse.log'
& powershell.exe -NoProfile -ExecutionPolicy Bypass -Command $cmd4 *>&1 | Out-File -FilePath $log4 -Encoding utf8
$rc4 = $LASTEXITCODE
Ck 'S13 假运行器+非SELFTEST tag⇒拒绝(3)' ($rc4 -eq 3) $rc4

# 14) 进度清单含"已完成/未完成"
$r = Run-Scenario -Name 'progress' -Scenario 'missing_rep2' -Decl $declBranch
Ck 'S14 中断后打印已完成/未完成清单' (($r.text -match '\[done\]') -and ($r.text -match '\[pending\]')) $null

$npass = @($checks | Where-Object { $_.ok }).Count
$nfail = @($checks | Where-Object { -not $_.ok }).Count
Write-Host "selftest_so_batch_orchestration: $npass/$($checks.Count) 通过（失败 $nfail）"
Write-Host "临时目录：$tmpRoot（可保留用于人工查看控制台日志）"
if ($nfail -gt 0) { exit 1 }
exit 0
