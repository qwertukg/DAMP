# -*- coding: utf-8 -*-
"""Живой дашборд на опоре стенда подбора (вне статьи): настройки tools/tuning/stand/base.json поверх client/config.toml.

    python tools/tuning/dashboard_base.py [зерно] [обучающих=500] [порт=8765] [рабочих процессов=0]
    python tools/tuning/dashboard_base.py run <номер прогона> [порт=8765] [рабочих процессов=0]
    … --save ПАПКА — после обучения записать снимок обученного цикла; потом `python client/run.py dashboard --load ПАПКА`

Второй вид — дашборд на любом прогоне стенда: его правки (опора, зерно, предел карты) и число картинок берутся из
журнала стенда (tools/tuning/stand/tune2_log.jsonl). Например, `run 1553` — 20 000 картинок, самые чёткие кластеры верха.

Обычный `python client/run.py dashboard` берёт настройки только из client/config.toml, а опора подбора живёт в настройках
стенда. Здесь цикл учится теми же настройками, что прогон стенда (опора, зерно кодера, «лёгкие» настройки стенда), поэтому
шаги берутся из кеша прогона опоры на этом зерне. Зерно по умолчанию — лучший прогон опоры (tools/tuning/base_runs.json).
Рабочих процессов по умолчанию нет: каждая копия цикла на 500 картинках держит свою видеопамять.
Перед любым счётом на стенде дашборд закрывать (он держит видеокарту).
"""
import json
import os
import sys
import webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "client"), os.path.join(ROOT, "system"), os.path.join(ROOT, "tools", "tuning")]
os.environ["RERUN"] = "off"
os.environ["DAMP_RERUN_VIEWER"] = "off"

# Те же «лёгкие» настройки, что у прогона стенда (tune_cycle.py): с ними совпадают ключи кеша шагов. С 2026-11-25 стенд не
# ограничивает потоки кодера и детекторов («отдать стенду все ядра»), поэтому ENCODING_WORKERS и ACTIVATION_PARALLEL_WORKERS
# здесь больше не задаются.
LIGHT = {"LAYOUT_LOG_VISUALS": False, "LADDER_CLASS_MAP": False}


def best_seed() -> int:
    """Зерно лучшего по М1u прогона опоры."""
    import board_runner as R

    rows = {int(k): v for k, v in R.log_rows().items()}
    base = json.load(open(os.path.join(ROOT, "tools", "tuning", "base_runs.json"), encoding="utf-8"))
    return int(max(base, key=lambda seed: R.m1u(rows[int(base[seed])]) if int(base[seed]) in rows else -1.0))


def run_settings(number: int) -> tuple[dict, int]:
    """Правки и число обучающих картинок прогона стенда — из его строки журнала."""
    import board_runner as R

    row = {int(k): v for k, v in R.log_rows().items()}[number]
    return dict(row["changes"]), int(row["count"])


def main() -> None:
    from damp_api import Damp
    from dashboard.server import DashboardServer

    save = None
    if "--save" in sys.argv:  # --save ПАПКА — после обучения записать снимок обученного цикла (дашборд потом --load ПАПКА)
        at = sys.argv.index("--save")
        save = sys.argv[at + 1]
        del sys.argv[at : at + 2]
    if len(sys.argv) > 2 and sys.argv[1] == "run":
        number = int(sys.argv[2])
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 8765
        workers = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        changes, count = run_settings(number)
        seed = changes.get("ENCODER_RANDOM_SEED")
        title = f"прогон стенда {number}, зерно {seed}"
    else:
        seed = int(sys.argv[1]) if len(sys.argv) > 1 else best_seed()
        count = int(sys.argv[2]) if len(sys.argv) > 2 else 500
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 8765
        workers = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        base = json.load(open(os.path.join(ROOT, "tools", "tuning", "stand", "base.json"), encoding="utf-8-sig"))
        changes = {**base, "ENCODER_RANDOM_SEED": seed}
        title = f"опора стенда, зерно {seed}"
    damp = Damp(changes={**changes, **LIGHT, "DASHBOARD_WORKERS": workers})
    server = DashboardServer(damp, count=count, map_points=None, port=port, save=save, label=title)
    print(f"dashboard: {title}; учу цикл на {count} обучающих цифрах (готовые шаги — из кеша стенда)…", flush=True)
    ladder = server.prepare()
    if server.saved is not None:
        print(f"dashboard: снимок записан — {server.saved.path}, {server.saved.megabytes} МБ "
              f"(запуск без обучения: python client/run.py dashboard --load {server.saved.path})", flush=True)
    print(f"dashboard: уровней {ladder.levels}, обучающих картинок {ladder.train_pictures}; страница — {server.address}", flush=True)
    webbrowser.open(server.address)
    server.serve()


if __name__ == "__main__":
    main()
