#!/usr/bin/env bash
# Ждёт конца прогонов стенда (в tune_pair.done по строке на прогон; ждать столько, сколько номеров передано),
# ловит застой (журнал прогона не растёт 15 минут — строка STALL и выход), печатает ошибки и сводку серии.
d="C:/Users/danii/AppData/Local/Temp/claude/156391f67facb0d2/4902e47e-f207-47e9-8247-6a73f171df92/scratchpad"
want=$#
sleep 60
while true; do
  done_lines=$( [ -f "$d/tune_pair.done" ] && wc -l < "$d/tune_pair.done" || echo 0 )
  [ "$done_lines" -ge "$want" ] && break
  now=$(date +%s)
  for n in "$@"; do
    f="$d/tune2-$n.out"
    if [ -f "$f" ] && ! grep -aq "^ИТОГ $n" "$f"; then
      age=$(( now - $(stat -c %Y "$f") ))
      if [ "$age" -gt 900 ]; then echo "STALL: прогон $n — журнал не растёт $((age / 60)) мин"; exit 1; fi
    fi
  done
  sleep 20
done
cat "$d/tune_pair.done"
for n in "$@"; do
  grep -a -A4 "^Traceback" "$d/tune2-$n.err" 2>/dev/null | grep -av "Logging\|rollover\|rename" | tail -4
done
PYTHONIOENCODING=utf-8 python "$d/summary.py" 250 496
