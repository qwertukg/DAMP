# Точка оси плана подбора v2: пара прогонов на 250 цифрах — зерно 0 (номер First) и зерно 2 (First + 1), скрыто.
#   run_pair.ps1 -First 576 -Label "перекрытие кода угла 0.85" -Changes '{"MNIST_ANGLE_OVERLAP": 0.85}'
# Опора — client/config.toml (с прогона 574 там вариант 561) + принятое в фазе (base.json); правки — подбираемая ось.
# Зёрна (анализ 2026-11-21): опора 0.85 выбрана по зёрнам 0 и 2 и на них завышена — отбор идёт парно на свежих зёрнах 1 и 3
# (опора на них есть: 606, 607); подтверждение — на других зёрнах, вместе с опорой.
param([int]$First, [string]$Label, [string]$Changes = '{}', [string]$Seeds = '1,3')
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$utf8 = New-Object Text.UTF8Encoding($false)
$seedList = @($Seeds -split '[ ,;]+' | Where-Object { $_ } | ForEach-Object { [int]$_ })
$numbers = @()
for ($k = 0; $k -lt $seedList.Count; $k++) {
    $seed = $seedList[$k]
    $number = $First + $k
    $numbers += $number
    # Принятые в этой фазе точки (base.json) + подбираемая ось; в конфиг принятое уходит в конце фазы.
    $over = Get-Content "$dir\base.json" -Raw -Encoding utf8 | ConvertFrom-Json -AsHashtable
    if ($null -eq $over) { $over = @{} }
    $axis = $Changes | ConvertFrom-Json -AsHashtable
    foreach ($key in $axis.Keys) { $over[$key] = $axis[$key] }
    if ($seed -ne 0) { $over['ENCODER_RANDOM_SEED'] = $seed }
    $spec = @{ label = "250 цифр: $Label, зерно $seed"; changes = $over }
    [IO.File]::WriteAllText("$dir\tune2-$number.json", ($spec | ConvertTo-Json -Depth 5), $utf8)
}
Remove-Item "$dir\tune_pair.done" -ErrorAction SilentlyContinue
# Номера — через запятую: строка с пробелом дошла бы до tune_pair.ps1 двумя аргументами (так 2026-11-20 пропал 575).
Start-Process pwsh -WindowStyle Hidden -ArgumentList '-NoProfile', '-File', "$dir\tune_pair.ps1", '-List', ($numbers -join ','), '-Count', '250', '-Tests', '100'  # автор 2026-11-21: «добавь проверку точности» — 100 проверочных
"запущено: $($numbers -join ', ') (зёрна $($seedList -join ', ')) — $Label"
