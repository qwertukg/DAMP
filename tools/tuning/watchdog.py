"""Сторож стенда (автор 2026-11-22: «следи за тем, чтобы прогон всегда шёл»).

Раз в минуту: жив ли прогонщик (нет — поднять tools/start.ps1), идёт ли пара (процесс tune_cycle), сколько стенд стоит
без пары, не пуста ли Очередь. Печатает строку только при событии — её читает Claude (Monitor):
  ПРОГОНЩИК УПАЛ / ПОДНЯТ, СТЕНД СТОИТ N мин (+ стек прогонщика py-spy после 8 мин), ОЧЕРЕДЬ ПУСТА, ПАРА ИДЁТ.
Запуск: python -X utf8 tools/tuning/watchdog.py (из корня репозитория).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
STAND = os.path.join(HERE, "stand")
LOG = os.path.join(STAND, "board_runner.log")
PID = os.path.join(STAND, "board_runner.pid")
IDLE_WARN, IDLE_DUMP, REPEAT = 4 * 60, 8 * 60, 5 * 60


def say(text: str) -> None:
    print(time.strftime("%H:%M:%S"), text, flush=True)


def processes() -> list[tuple[int, str]]:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | % { \"$($_.ProcessId)`t$($_.CommandLine)\" }"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60).stdout
    found = []
    for line in out.splitlines():
        pid, _, command = line.partition("\t")
        if pid.strip().isdigit():
            found.append((int(pid), command))
    return found


def last_line() -> str:
    try:
        lines = open(LOG, encoding="utf-8", errors="replace").read().splitlines()
        return lines[-1][:200] if lines else ""
    except OSError:
        return ""


def main() -> None:
    idle_since, last_warn, was_running, empty_said, missing = None, 0.0, None, False, 0
    say("сторож стенда запущен")
    while True:
        try:
            procs = processes()
            runner = [pid for pid, command in procs if "board_runner.py" in command]
            pair = any("tune_cycle" in command for _, command in procs)
            now = time.time()
            missing = missing + 1 if not runner else 0
            if missing == 1:
                say("прогонщика нет — жду минуту (может, его перезапускают)")
            if missing >= 2:
                missing = 0
                say("ПРОГОНЩИК УПАЛ — поднимаю tools/start.ps1")
                subprocess.run(["pwsh", "-NoProfile", "-File", os.path.join(ROOT, "tools", "start.ps1")], cwd=ROOT,
                               capture_output=True, timeout=300)
                say("ПРОГОНЩИК ПОДНЯТ" if any("board_runner.py" in c for _, c in processes()) else "ПРОГОНЩИК НЕ ПОДНЯЛСЯ")
            tail = last_line()
            if "очередь пуста" in tail and not empty_said:
                say(f"ОЧЕРЕДЬ ПУСТА — нужен углублённый анализ и новый план | {tail}")
                empty_said = True
            elif "очередь пуста" not in tail:
                empty_said = False
            if pair:
                if was_running is False:
                    say("ПАРА ИДЁТ")
                idle_since, last_warn = None, 0.0
            else:
                idle_since = idle_since or now
                idle = now - idle_since
                if idle >= IDLE_WARN and now - last_warn >= REPEAT:
                    say(f"СТЕНД СТОИТ {idle / 60:.0f} мин | последняя строка: {tail}")
                    last_warn = now
                    if idle >= IDLE_DUMP and runner:
                        dump = subprocess.run(["py-spy", "dump", "--pid", str(max(runner))], capture_output=True, text=True,
                                              encoding="utf-8", errors="replace", timeout=60).stdout
                        frames = [l.strip() for l in dump.splitlines() if "board_runner.py" in l or "board.py" in l][:4]
                        say("стек прогонщика: " + " ← ".join(frames))
            was_running = pair
        except Exception as error:  # noqa: BLE001  сторож не падает
            say(f"сторож: ошибка {error!r}")
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
