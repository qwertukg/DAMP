#!/usr/bin/env bash
# Очередь прогонов стенда: строки файла «номер|подпись|правки JSON»; каждый ждёт конца предыдущего.
d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # папка стенда — там, где лежит скрипт
jobs="$1"
while IFS='|' read -r number label changes; do
  [ -z "$number" ] && continue
  # ждать, пока не идёт ни один прогон стенда
  while tasklist //FI "IMAGENAME eq python.exe" //V 2>/dev/null | grep -q . && wmic process where "name='python.exe'" get CommandLine 2>/dev/null | grep -q "tune_cycle.py"; do sleep 15; done
  pwsh -NoProfile -File "$d/run_one.ps1" -Number "$number" -Label "$label" -Changes "$changes" >/dev/null
  sleep 30
  bash "$d/wait_summary.sh" "$number" 2>&1 | grep -E "^$number|^\| $number|^\| \*\*$number|STALL|Traceback"
done < "$jobs"
