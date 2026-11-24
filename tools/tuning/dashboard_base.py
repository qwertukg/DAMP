# -*- coding: utf-8 -*-
"""Живой дашборд на опоре стенда подбора (вне статьи): настройки tools/tuning/stand/base.json поверх client/config.toml.

    python tools/tuning/dashboard_base.py [зерно] [обучающих=500] [порт=8765] [рабочих процессов=0]

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

# Те же «лёгкие» настройки, что у прогона стенда (tune_cycle.py): с ними совпадают ключи кеша шагов.
LIGHT = {"LAYOUT_LOG_VISUALS": False, "ENCODING_WORKERS": 4, "ACTIVATION_PARALLEL_WORKERS": 2, "LADDER_CLASS_MAP": False}


def best_seed() -> int:
    """Зерно лучшего по М1u прогона опоры."""
    import board_runner as R

    rows = {int(k): v for k, v in R.log_rows().items()}
    base = json.load(open(os.path.join(ROOT, "tools", "tuning", "base_runs.json"), encoding="utf-8"))
    return int(max(base, key=lambda seed: R.m1u(rows[int(base[seed])]) if int(base[seed]) in rows else -1.0))


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else best_seed()
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 8765
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    from damp_api import Damp
    from dashboard.server import DashboardServer

    base = json.load(open(os.path.join(ROOT, "tools", "tuning", "stand", "base.json"), encoding="utf-8-sig"))
    damp = Damp(changes={**base, **LIGHT, "ENCODER_RANDOM_SEED": seed, "DASHBOARD_WORKERS": workers})
    server = DashboardServer(damp, count=count, map_points=None, port=port)
    print(f"dashboard: опора стенда, зерно {seed}; учу цикл на {count} обучающих цифрах (шаги — из кеша прогона опоры)…", flush=True)
    ladder = server.prepare()
    print(f"dashboard: уровней {ladder.levels}, обучающих картинок {ladder.train_pictures}; страница — {server.address}", flush=True)
    webbrowser.open(server.address)
    server.serve()


if __name__ == "__main__":
    main()
