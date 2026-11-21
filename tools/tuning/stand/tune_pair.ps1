# Стенд подбора: прогоны одновременно (пара от прогонщика доски), на Count обучающих и Tests проверочных цифрах.
# Каждый номер — файл tune2-<номер>.json (label, changes) рядом. Вывод — tune2-<номер>.out; итог — строка «ИТОГ».
# Если «ИТОГ» уже есть, а процесс висит на выходе, через минуту он останавливается. Готовые номера — в tune_pair.done.
# Пути не зашиты: репозиторий — три уровня выше этой папки (tools/tuning/stand); Python — DAMP_PYTHON из окружения
# или .env, иначе python из PATH (docs/HARNESS.md).
param([string]$List, [int]$Count = 100, [int]$Tests = 100)
$Numbers = @($List -split '[ ,;]+' | Where-Object { $_ } | ForEach-Object { [int]$_ })
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..')).Path
$python = $env:DAMP_PYTHON
if (-not $python -and (Test-Path (Join-Path $repo '.env'))) {
    $line = Get-Content (Join-Path $repo '.env') -Encoding utf8 | Where-Object { $_ -match '^\s*DAMP_PYTHON\s*=' } | Select-Object -First 1
    if ($line) { $python = ($line -split '=', 2)[1].Trim() }
}
if (-not $python) { $python = (Get-Command python -ErrorAction Stop).Source }
Set-Location $repo
$Host.UI.RawUI.WindowTitle = "DAMP: стенд, прогоны $($Numbers -join ', ') на $Count цифрах"
$env:RERUN = 'off'  # автор 2026-11-21: «ты все ресурсы сожрал» — окно Rerun держало 1.5 ГБ; стенд — без Rerun
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$start = Get-Date
$running = @{}
foreach ($number in $Numbers) {
    $out = "$here\tune2-$number.out"; $err = "$here\tune2-$number.err"
    Remove-Item $out, $err -ErrorAction SilentlyContinue
    $spec = Get-Content "$here\tune2-$number.json" -Raw -Encoding utf8 | ConvertFrom-Json
    Write-Host "== прогон ${number}: $($spec.label)" -ForegroundColor Cyan
    $running[$number] = Start-Process -FilePath $python -ArgumentList @('-X', 'utf8', "`"$here\tune_cycle.py`"", $number, "`"$here\tune2-$number.json`"", $Count, $Tests) -NoNewWindow -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
}
$seen = @{}
while ($running.Count -gt 0) {
    Start-Sleep -Seconds 5
    foreach ($number in @($running.Keys)) {
        $process = $running[$number]
        $text = (Get-Content "$here\tune2-$number.out" -Raw -ErrorAction SilentlyContinue) + (Get-Content "$here\tune2-$number.err" -Raw -ErrorAction SilentlyContinue)
        $done = $process.HasExited
        if (-not $done -and ($text -cmatch '(?m)^ИТОГ \d')) {
            if (-not $seen.ContainsKey($number)) { $seen[$number] = Get-Date }
            if (((Get-Date) - $seen[$number]).TotalSeconds -gt 60) {
                Get-CimInstance Win32_Process -Filter "ParentProcessId = $($process.Id)" | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
                Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
                $done = $true
            }
        }
        if ($done) {
            $running.Remove($number)
            $minutes = [Math]::Round(((Get-Date) - $start).TotalMinutes, 1)
            Write-Host "== прогон $number закончился, $minutes мин" -ForegroundColor Cyan
            Get-Content "$here\tune2-$number.out" -ErrorAction SilentlyContinue | Where-Object { $_ -match '^\|' -or $_ -match 'ИТОГ|уровней|точек' }
            "$number minutes=$minutes" | Add-Content "$here\tune_pair.done"
        }
    }
}
