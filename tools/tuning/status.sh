#!/usr/bin/env bash
# Строка состояния подбора: время, идущий прогон (последний изменённый tune2-*.out), его этап и свежесть журнала, лучший.
d="C:/Users/danii/AppData/Local/Temp/claude/156391f67facb0d2/4902e47e-f207-47e9-8247-6a73f171df92/scratchpad"
f=$(ls -t "$d"/tune2-*.out 2>/dev/null | head -1)
n=$(basename "$f" .out | sed 's/tune2-//')
age=$(( ($(date +%s) - $(stat -c %Y "$f")) / 60 ))
stage=$(grep -a -o -E "^ladder\.cycle\.begin level=[0-9]+" "$f" | tail -1 | sed 's/.*level=/уровень /')
if grep -aq "^ИТОГ $n" "$f"; then state="готов"; else state="идёт: ${stage:-начало}, журнал обновлён $age мин назад"; fi
py=$(tasklist 2>/dev/null | grep -ci python.exe)
best=$(PYTHONIOENCODING=utf-8 python "$d/summary.py" 250 496 2>/dev/null | grep "^ЛУЧШИЙ" | cut -c1-90)
top=$(PYTHONIOENCODING=utf-8 python "$d/summary.py" 250 496 2>/dev/null | grep -E "^\| \*\*Итого" | sed -E 's/.*лучший — \*\*№([0-9]+)\*\*: ([0-9.]+) \| ([0-9]+) \| ([0-9.]+).*/№\1: сходимость \2, пятен \3, в главном \4/')
echo "$(date +%H:%M) | прогон $n — $state | python: $py | лучший $top"
