# Один прогон стенда на 250 цифрах: настройки лучшего варианта (base.json) с правками поверх.
#   run_one.ps1 -Number 501 -Label "что изменено" -Changes '{"LADDER_KEYHOLE_RADIUS_SHARE": [1.15, 0.35, 1.75]}'
param([int]$Number, [string]$Label, [string]$Changes = '{}')
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$base = Get-Content "$dir\base.json" -Raw -Encoding utf8 | ConvertFrom-Json -AsHashtable
$over = $Changes | ConvertFrom-Json -AsHashtable
foreach ($key in $over.Keys) { $base[$key] = $over[$key] }
$spec = @{ label = "250 цифр: $Label"; changes = $base }
[IO.File]::WriteAllText("$dir\tune2-$Number.json", ($spec | ConvertTo-Json -Depth 5), (New-Object Text.UTF8Encoding($false)))
Remove-Item "$dir\tune_pair.done" -ErrorAction SilentlyContinue
Start-Process pwsh -WindowStyle Hidden -ArgumentList '-NoProfile', '-File', "$dir\tune_pair.ps1", '-List', "$Number", '-Count', '250', '-Tests', '0'  # без окна (автор 2026-11-20)
"запущено: $Number — $Label"
