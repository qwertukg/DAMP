# Окно для автора: прогоны подбора окон одновременно (обычно пара), на 100 обучающих и 100 проверочных цифрах.
# Каждый номер — файл tune2-<номер>.json (label, changes) рядом. Вывод — tune2-<номер>.out; итог — строка «ИТОГ».
# Если «ИТОГ» уже есть, а процесс висит на выходе (Rerun держит его), через минуту он останавливается.
param([string]$List, [int]$Count = 100, [int]$Tests = 100)
$Numbers = @($List -split '[ ,;]+' | Where-Object { $_ } | ForEach-Object { [int]$_ })
$repo = 'c:\Users\danii\AppData\Local\Temp\claude\90c048c3c2430e10\50fe34b5-999a-4743-bfe4-e4327ebbf623\scratchpad\damp-second-level'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = 'D:\Projects\DAMP\.venv\Scripts\python.exe'
Set-Location $repo
$Host.UI.RawUI.WindowTitle = "DAMP: подбор окон уровней 2–3, прогоны $($Numbers -join ', ') на $Count цифрах"
$env:RERUN = 'on'  # автор 2026-11-20: «если rerun не сильно влияет на память, запускай его» (окно ~350 МБ)
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
