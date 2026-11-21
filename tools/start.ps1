# Запуск «упряжи» подбора одной командой (docs/HARNESS.md): проверить .env, Python, доступ к доске и к GLM, поднять
# прогонщик доски и роутер (если ещё не работают). Повторный запуск безопасен: живые процессы не дублируются.
#
#   pwsh tools/start.ps1            — проверить и запустить
#   pwsh tools/start.ps1 -Stop      — остановить прогонщик и роутер (идущая пара стенда доработает)
#   pwsh tools/start.ps1 -Status    — только показать состояние
param([switch]$Stop, [switch]$Status)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $repo
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Read-Env {
    $values = @{}
    if (Test-Path "$repo\.env") {
        Get-Content "$repo\.env" -Encoding utf8 | Where-Object { $_ -match '^\s*[A-Z_]+\s*=' } | ForEach-Object {
            $key, $value = $_ -split '=', 2; $values[$key.Trim()] = $value.Trim()
        }
    }
    return $values
}
$conf = Read-Env
$stand = if ($env:DAMP_STAND) { $env:DAMP_STAND } elseif ($conf['DAMP_STAND']) { $conf['DAMP_STAND'] } else { "$repo\tools\tuning\stand" }
$python = if ($env:DAMP_PYTHON) { $env:DAMP_PYTHON } elseif ($conf['DAMP_PYTHON']) { $conf['DAMP_PYTHON'] } else { (Get-Command python).Source }
$procs = @(
    @{ Name = 'board_runner'; Script = 'tools/tuning/board_runner.py'; Title = 'прогонщик доски' },
    @{ Name = 'router'; Script = 'tools/glm/router.py'; Title = 'роутер (Claude / GLM)' }
)

function Test-Alive([string]$name) {
    $pidFile = Join-Path $stand "$name.pid"
    if (-not (Test-Path $pidFile)) { return $null }
    $id = [int](Get-Content $pidFile)
    try { Get-Process -Id $id -ErrorAction Stop | Out-Null; return $id } catch { return $null }
}

if ($Stop) {
    foreach ($p in $procs) { $id = Test-Alive $p.Name; if ($id) { Stop-Process -Id $id -Force; Write-Host "остановлен: $($p.Title) ($id)" } }
    return
}

Write-Host "== DAMP: упряжь подбора" -ForegroundColor Cyan
Write-Host "репозиторий: $repo"
Write-Host "стенд:       $stand"
Write-Host "Python:      $python"
$fail = $false
foreach ($key in 'GITHUB_URL', 'GITHUB_TOKEN', 'GML_SSH_HOST', 'PASS_SSH_PASS') {
    if (-not $conf[$key]) { Write-Host "нет $key в .env (образец — .env.example)" -ForegroundColor Red; $fail = $true }
}
if (-not $conf['GITHUB_ISSUES_TOKEN']) { Write-Host "нет GITHUB_ISSUES_TOKEN — задачи пишутся токеном доски" -ForegroundColor Yellow }
if (-not (Test-Path $python)) { Write-Host "нет Python: $python" -ForegroundColor Red; $fail = $true }
if (-not (Test-Path "$stand\tune2_log.jsonl")) { Write-Host "нет журнала прогонов $stand\tune2_log.jsonl" -ForegroundColor Yellow }
if ($fail) { throw "исправьте .env и запустите снова" }

# Python и пакеты проекта, доска (токен и его области), GLM (туннель и модель).
& $python -X utf8 -c "import sys; sys.path[:0]=['system','client']; import numpy, damp; print('Python и пакеты: ok, DAMP', damp.__version__)"
& $python -X utf8 tools/tuning/board.py scopes
& $python -X utf8 tools/glm/agent.py status
if ($Status) {
    foreach ($p in $procs) { $id = Test-Alive $p.Name; Write-Host "$($p.Title): $(if ($id) { "работает ($id)" } else { 'не запущен' })" }
    return
}

New-Item -ItemType Directory -Force $stand | Out-Null
foreach ($p in $procs) {
    $id = Test-Alive $p.Name
    if ($id) { Write-Host "$($p.Title): уже работает ($id)"; continue }
    $proc = Start-Process -FilePath $python -ArgumentList @('-X', 'utf8', $p.Script) -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $stand "$($p.Name).out") -RedirectStandardError (Join-Path $stand "$($p.Name).err")
    $proc.Id | Out-File (Join-Path $stand "$($p.Name).pid") -Encoding ascii
    Write-Host "$($p.Title): запущен ($($proc.Id))" -ForegroundColor Green
}
Write-Host ""
Write-Host "Журналы: $stand\board_runner.log, $stand\router.log. Claude в начале сессии — по CLAUDE.md (отметка присутствия, слежение)."
