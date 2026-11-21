#!/usr/bin/env bash
# Ждёт конца прогонов стенда, но не дольше 5 минут (отчёт автору каждые 5 минут), и печатает состояние каждого:
#   wait5.sh 578 579
# Прогон готов, когда в его выводе есть «ИТОГ N»; прогон без процесса и без итога — «упал или остановлен».
d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
end=$((SECONDS + 300))
alive() { wmic process where "name='python.exe'" get CommandLine 2>/dev/null | grep -q "tune_cycle.py\"* $1 "; }
while [ $SECONDS -lt $end ]; do
  left=0
  for n in "$@"; do grep -aq "^ИТОГ $n" "$d/tune2-$n.out" 2>/dev/null || { alive "$n" && left=$((left + 1)); }; done
  [ "$left" -eq 0 ] && break
  sleep 15
done
now=$(date +%s)
for n in "$@"; do
  f="$d/tune2-$n.out"
  if grep -aq "^ИТОГ $n" "$f" 2>/dev/null; then echo "$n: готов"; continue; fi
  alive "$n" || { echo "$n: упал или остановлен"; continue; }
  stage=$(grep -a -o -E "^ladder\.cycle\.begin level=[0-9]+|^layout\.[a-z_.]+|^activation\.[a-z_.]+" "$f" 2>/dev/null | tail -1)
  age=$(( (now - $(stat -c %Y "$f" 2>/dev/null || echo $now)) / 60 ))
  echo "$n: идёт — ${stage:-начало}; журнал обновлён $age мин назад; ошибок $(grep -a -c '^Traceback' "$d/tune2-$n.err" 2>/dev/null)"
done
# Ресурсы (автор 2026-11-21: «мониторь ресурсы, оставляй ресурсы для работы системы»): запас ≥ 6 ГБ ОЗУ, ≥ 1.5 ГБ видео.
free=$(wmic OS get FreePhysicalMemory 2>/dev/null | grep -o '[0-9]*' | head -1)
gpu=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
used=${gpu%,*}; total=${gpu#*,}
ram_mb=$((free / 1024))
echo "ресурсы: ОЗУ свободно $((ram_mb / 1024)).$(( (ram_mb % 1024) * 10 / 1024 )) ГБ; видеопамять $used из $total МиБ"
[ "$ram_mb" -lt 6144 ] && echo "ВНИМАНИЕ: ОЗУ меньше запаса 6 ГБ"
[ -n "$used" ] && [ $((total - used)) -lt 1536 ] && echo "ВНИМАНИЕ: видеопамяти меньше запаса 1.5 ГБ"
exit 0
