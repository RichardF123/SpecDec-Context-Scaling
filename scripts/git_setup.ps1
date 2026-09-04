# git_setup.ps1 — one-shot Git init + first push for the specdec repo
# Usage:
#   powershell -ExecutionPolicy Bypass -File D:\RealDSH\specdec\scripts\git_setup.ps1 -GitHubUser "你的GitHub用户名"
# Prerequisite: an EMPTY private repo named "specdec" already created on github.com
# (or run with -CreateWithGh to create it via the gh CLI).
param(
    [string]$GitHubUser = "",
    [string]$RepoName = "specdec",
    [switch]$CreateWithGh
)

$ErrorActionPreference = "Continue"
$repoRoot = Split-Path -Parent $PSScriptRoot   # D:\RealDSH\specdec
Set-Location $repoRoot

Write-Host "== 0) git installed? ==" -ForegroundColor Cyan
$gitVer = git --version 2>&1
if ($LASTEXITCODE -ne 0 -or $gitVer -match "not recognized|不是") {
    Write-Host "[FAIL] git not found. Install it, then close & reopen the terminal:" -ForegroundColor Red
    Write-Host "       winget install --id Git.Git -e" -ForegroundColor Yellow
    exit 1
}
Write-Host $gitVer

Write-Host "`n== 1) repo root = $repoRoot ==" -ForegroundColor Cyan
if (-not (Test-Path (Join-Path $repoRoot ".git"))) {
    Write-Host "> git init" -ForegroundColor DarkGray
    git init 2>&1 | Out-String | Write-Host
}

Write-Host "`n== 2) branch to main ==" -ForegroundColor Cyan
git branch -M main 2>&1 | Out-String | Write-Host

Write-Host "`n== 3) commit identity (GitHub noreply email) ==" -ForegroundColor Cyan
$name = git config user.name
if (-not $name) { git config user.name $GitHubUser }
$email = git config user.email
if (-not $email) {
    if ($GitHubUser) { git config user.email "$GitHubUser@users.noreply.github.com" }
    else { git config user.email "student@users.noreply.github.com" }
}
Write-Host ("user.name  = " + (git config user.name))
Write-Host ("user.email = " + (git config user.email))

Write-Host "`n== 4) stage & commit ==" -ForegroundColor Cyan
git add .
git commit -m "Phase 0 scaffold: RunConfig hash-ledger CLI + docs (SpecDec)" 2>&1 | Out-String | Write-Host
# note: "nothing to commit" is fine on re-runs

Write-Host "`n== 5) remote origin ==" -ForegroundColor Cyan
$remote = git remote get-url origin 2>&1
if ($LASTEXITCODE -ne 0 -or -not $remote) {
    if ($CreateWithGh -and $GitHubUser) {
        Write-Host "> gh repo create $RepoName --private --source . --push" -ForegroundColor DarkGray
        gh repo create $RepoName --private --source . --push 2>&1 | Out-String | Write-Host
        Write-Host "[OK] created & pushed via gh. Done." -ForegroundColor Green
        exit 0
    }
    if ($GitHubUser) {
        git remote add origin "https://github.com/$GitHubUser/$RepoName.git"
        Write-Host "origin -> https://github.com/$GitHubUser/$RepoName.git" -ForegroundColor DarkGray
    } else {
        Write-Host "[WARN] no -GitHubUser given; add remote manually, e.g.:" -ForegroundColor Yellow
        Write-Host "       git remote add origin https://github.com/<YOU>/$RepoName.git"
    }
} else {
    Write-Host "origin = $remote"
}

Write-Host "`n== 6) push ==" -ForegroundColor Cyan
if ($GitHubUser -or (git remote get-url origin 2>$null)) {
    git push -u origin main 2>&1 | Out-String | Write-Host
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] pushed. Check https://github.com/ on the web." -ForegroundColor Green
    } else {
        Write-Host "[FAIL] push failed (usually auth). Fixes:" -ForegroundColor Red
        Write-Host "   A) run:  gh auth login   then re-run this script with -CreateWithGh"
        Write-Host "   B) or create a Personal Access Token (repo scope) and use:"
        Write-Host "      git remote set-url origin https://<USER>:<TOKEN>@github.com/<USER>/$RepoName.git"
    }
} else {
    Write-Host "[WARN] skipped push (no origin URL)." -ForegroundColor Yellow
}

Write-Host "`n== done ==" -ForegroundColor Cyan
