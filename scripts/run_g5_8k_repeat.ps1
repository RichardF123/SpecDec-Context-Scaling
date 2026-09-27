<#
  run_g5_8k_repeat.ps1 -- 8K 双模型 smoke 的**重复可运行性**编排（3 x base + 3 x V-only，交错顺序）。

  为什么要编排脚本：任务要求
    * 两档各 3 个**独立子进程**，每个进程只运行一次；
    * **串行**执行，且预先固定**交错**顺序：base1 -> V-only1 -> base2 -> V-only2 -> base3 -> V-only3；
    * **每次重查环境**（活跃分页文件 / commit / GPU / P-state），不沿用旧快照；
    * 保持**默认 16 GiB commit 护栏**（不给 runner 传第 2 个参数）；
    * 任一步 OOM / 崩溃 / 检查失败 / 环境不合格 ⇒ **保留记录并停止**，不重试挑选成功样本。
  每一步都是一个全新的 `cmd /c`(=> 全新 python 进程)，产出独立 JSON 与日志。

  用法（在 D:\RealDSH\specdec 下）:
      pwsh -File scripts\run_g5_8k_repeat.ps1
  可选环境变量：
      REPEAT_TAG=...    结果文件名前缀（默认 run_g5_8k_repeat_<UTC时间戳>）
      REPEAT_MAXNEW=16  生成数（默认 16）
      REPEAT_DRYRUN=1   只打印计划与环境，不跑
#>
$ErrorActionPreference = 'Stop'
$root = 'D:\RealDSH\specdec'
$py = Join-Path $root '.venv\Scripts\python.exe'
$outdir = Join-Path $root 'output\g5\attn'
$tag = if ($env:REPEAT_TAG) { $env:REPEAT_TAG } else { 'run_g5_8k_repeat_' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') }
$maxnew = if ($env:REPEAT_MAXNEW) { $env:REPEAT_MAXNEW } else { '16' }
$dry = ($env:REPEAT_DRYRUN -eq '1')
$master = Join-Path $outdir "$tag.master.tsv"
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'; $env:HF_HUB_OFFLINE = '1'

# 预先固定顺序（不因中途结果改变）。REPEAT_PLAN: repeat6（默认）| ar_spec4
$planSel = if ($env:REPEAT_PLAN) { $env:REPEAT_PLAN } else { 'repeat6' }
if ($planSel -eq 'ar_spec4') {
  # 8K/64-token 同后端 AR/SpecDec 对照：每条件独立进程、串行、各一次
  $plan = @(
    @{ n = 1; label = 'base_ar';   kv = 'base_bf16'; mode = 'ar'   },
    @{ n = 2; label = 'base_spec'; kv = 'base_bf16'; mode = 'spec' },
    @{ n = 3; label = 'vfps_ar';   kv = 'v_fp8';     mode = 'ar'   },
    @{ n = 4; label = 'vfps_spec'; kv = 'v_fp8';     mode = 'spec' }
  )
} else {
  $plan = @(
    @{ n = 1; label = 'base1';  kv = 'base_bf16'; mode = 'spec' },
    @{ n = 2; label = 'vfps1';  kv = 'v_fp8';     mode = 'spec' },
    @{ n = 3; label = 'base2';  kv = 'base_bf16'; mode = 'spec' },
    @{ n = 4; label = 'vfps2';  kv = 'v_fp8';     mode = 'spec' },
    @{ n = 5; label = 'base3';  kv = 'base_bf16'; mode = 'spec' },
    @{ n = 6; label = 'vfps3';  kv = 'v_fp8';     mode = 'spec' }
  )
}

function Sample-Env {
  param([string]$StepTag)
  $pf = Get-CimInstance Win32_PageFileUsage -ErrorAction SilentlyContinue |
        Select-Object Name, AllocatedBaseSize, CurrentUsage
  $pfLine = ($pf | ForEach-Object { "$($_.Name)=$($_.AllocatedBaseSize)MiB(used $($_.CurrentUsage))" }) -join '; '
  $pyOut = & $py -c "import sys;sys.path.insert(0,'.');from specdec.engine import g5_8k_spec_variance_probe as P;sm=P.system_memory();print('commit_limit=%.2f commit_used=%.2f commit_free=%.2f phys_free=%.2f MIN=%.1f'%(sm['commit_limit_mib']/1024,sm['commit_used_mib']/1024,sm['commit_free_mib']/1024,sm['phys_free_mib']/1024,P.MIN_COMMIT_GIB));from specdec.engine import gpu_pstate_preflight as PF;pf=PF.qualify();print('pstate_qualified=%s sm=%s mem=%s'%(pf.get('qualified'),pf.get('sm_median'),pf.get('mem_median')))"
  $gpu = & nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader
  $line = [pscustomobject]@{
    step = $StepTag; pagefiles = $pfLine; gpu = ($gpu -join ' ')
    py = ($pyOut -join ' | ')
  }
  Write-Host "[env $StepTag] $pfLine"
  Write-Host "[env $StepTag] gpu: $($line.gpu)"
  Write-Host "[env $StepTag] $($line.py)"
  return $line
}

"step`tlabel`tmode`tkv`texit_code`trun_id`tjson`tlog`tcommit_free`tgpu_free`tstatus`tpair_correctness`tcommon_prefix`tfirst_divergence" | Set-Content -Path $master -Encoding UTF8

foreach ($s in $plan) {
  Write-Host ""
  Write-Host ("=" * 70)
  Write-Host "STEP $($s.n)/$($plan.Count)  label=$($s.label)  mode=$($s.mode)  target_kv=$($s.kv)"
  Write-Host ("=" * 70)
  $e = Sample-Env -StepTag $s.label
  $commitFree = ([regex]::Match($e.py, 'commit_free=([0-9.]+)')).Groups[1].Value
  $gpuFree = ($e.gpu -split ',')[2].Trim()
  if ($dry) {
    "$($s.n)`t$($s.label)`t$($s.mode)`t$($s.kv)`tDRYRUN`t-`t-`t$commitFree`t$gpuFree`tdryrun" |
      Add-Content -Path $master -Encoding UTF8
    continue
  }
  $log = Join-Path $outdir "$($tag)_$($s.label).log"
  # 独立进程；**不传第 2 个参数** ⇒ 保持默认 16 GiB 护栏
  cmd /c "cd /d $root && set `"SMOKE_MODE=$($s.mode)`" && set `"SMOKE_KV=$($s.kv)`" && set `"SMOKE_MAXNEW=$maxnew`" && scripts\run_g5_attn_forensics.cmd smoke > `"$log`" 2>&1"
  $rc = $LASTEXITCODE
  # **按明确 run_id 关联产物**（不再猜"目录里最新的 JSON"）
  $rid = ([regex]::Match((Get-Content $log -Raw -Encoding UTF8), 'RUN_ID=([0-9TZ]+)')).Groups[1].Value
  $jsonPath = if ($rid) { Join-Path $outdir "smoke8k_cudnn_prefill_$rid.json" } else { '' }
  $hasJson = ($jsonPath -and (Test-Path $jsonPath))
  $status = if (-not $hasJson) { 'NO_RESULT' }
            elseif ($rc -eq 0) { 'PASS' }
            elseif ($rc -eq 3) { 'ENV_REFUSED' }
            else { "FAIL_rc$rc" }
  $s.runId = $rid; $s.jsonPath = $jsonPath
  "$($s.n)`t$($s.label)`t$($s.mode)`t$($s.kv)`t$rc`t$rid`t$jsonPath`t$log`t$commitFree`t$gpuFree`t$status`t`t`t" |
    Add-Content -Path $master -Encoding UTF8
  Write-Host "[step $($s.n)] exit=$rc status=$status run_id=$rid json=$(if($hasJson){Split-Path $jsonPath -Leaf}else{'(missing)'})"

  # ---- 同一 KV 档的 AR/SpecDec 配对：本步一并结束后**立即**比较生成序列 ----
  if ($s.mode -eq 'spec' -and $hasJson) {
    $prev = $plan | Where-Object { $_.mode -eq 'ar' -and $_.kv -eq $s.kv -and $_.n -lt $s.n } |
            Select-Object -Last 1
    if ($prev -and $prev.jsonPath -and (Test-Path $prev.jsonPath)) {
      $cmp = & $py -c "import json,sys;a=json.load(open(sys.argv[1],encoding='utf-8'))['run']['gen'];b=json.load(open(sys.argv[2],encoding='utf-8'))['run']['gen'];n=min(len(a),len(b));p=0
for i in range(n):
    if a[i]==b[i]: p+=1
    else: break
same=(a==b)
print('%s	%d	%s	%d	%d'%('SAME' if same else 'DIFF',p,'-' if same else '%d|%d'%((a[p] if p<len(a) else -1),(b[p] if p<len(b) else -1)),len(a),len(b)))" $prev.jsonPath $jsonPath
      $parts = ($cmp -split "`t")
      $verdict = if ($parts[0] -eq 'SAME') { 'PASS' } else { 'CHECK' }
      Write-Host "[pair $($prev.label) vs $($s.label)] pair_correctness=$verdict common_prefix=$($parts[1]) first_divergence=$($parts[2]) lens=$($parts[3])/$($parts[4])"
      "$($s.n)`tPAIR:$($prev.label)-vs-$($s.label)`t-$($s.kv)`t-$rc`t-$($jsonPath)`t-$log`t$commitFree`t$gpuFree`tPAIR_$verdict`t$verdict`t$($parts[1])`t$($parts[2])" |
        Add-Content -Path $master -Encoding UTF8
      if ($verdict -ne 'PASS') {
        Write-Host "!! PAIR $($prev.label)/$($s.label) token divergence (prefix=$($parts[1])): records kept, STOPPING."
        Write-Host "   NOT treated as exit-code failure, NOT excused as BF16 drift -- needs targeted diagnosis."
        "STOPPED_AT_PAIR=$($prev.label)/$($s.label) pair_correctness=$verdict" | Add-Content -Path $master -Encoding UTF8
        Write-Host "MASTER=$master"
        exit 1
      }
    }
  }
  if ($rc -ne 0) {
    Write-Host "!! STEP $($s.n) FAILED (rc=$rc): keeping records and STOPPING. No retry, no cherry-picking."
    "STOPPED_AT=$($s.label) rc=$rc" | Add-Content -Path $master -Encoding UTF8
    Write-Host "MASTER=$master"
    exit 1
  }
}
if ($dry) {
  Write-Host ""
  Write-Host "DRY RUN complete: env sampled for all $($plan.Count) steps, NOTHING was run."
  Write-Host "MASTER=$master"
  exit 0
}
Write-Host ""
Write-Host "ALL $($plan.Count) STEPS EXIT 0 (each once, serial, interleaved, default guard)"
Write-Host "MASTER=$master"
exit 0
