# -*- coding: utf-8 -*-
"""Скорость раскладки карты уровня на путях слитого ядра: таблица сходств, таблица общих бит (целиком и половиной),
общие биты на лету.

    python tools/tuning/layout_tiers.py [зерно=6] [уровень=1] [шагов дальней фазы=300] [повторов кодов=1] [all]

«all» — мерить и путь «общие биты на лету» (он в 16 раз медленнее таблиц; без слова — пропускается).

Зачем: таблица сходств (4·N² байт на видеокарте) помещается до 33–35 тыс. точек карты — это потолок ≈ 600 картинок на
стенде. Дальше раскладка сама переходит на таблицу общих бит (1–2 байта на пару) и на счёт общих бит на лету по кодам
int8. Здесь на настоящих кодах карты уровня (опора стенда, из кеша) меряется время шага каждого пути и сверяется, что
расстановка та же до клетки. «Повторов кодов» k > 1 — карта в k раз больше: коды повторяются k раз, у каждой копии —
свой добавочный бит (копии точки почти равны, но различны). Так меряется время шага на карте 2N, 3N точек без нового
прогона; сама такая карта смысла не несёт.
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "client"), os.path.join(ROOT, "system")]
os.environ["RERUN"] = "off"
os.environ["DAMP_RERUN_VIEWER"] = "off"
os.environ.setdefault("DAMP_LOG_CONSOLE", "heartbeat")
import numpy as np  # noqa: E402

LIGHT = {"LAYOUT_LOG_VISUALS": False, "ENCODING_WORKERS": 4, "ACTIVATION_PARALLEL_WORKERS": 2, "LADDER_CLASS_MAP": False}


def tiers(points: int, item_bytes: int, slow: bool) -> list:
    """Пути слитого ядра: имя и настройки. Половина таблицы берётся, когда квадратная в предел не помещается: предел
    ставится между их размерами."""
    square = (points + 1) ** 2 * item_bytes
    found = [
        ("таблица сходств", {"LAYOUT_FUSED_TABLE_MEMORY_FRACTION": 0.9}),
        ("таблица общих бит", {"LAYOUT_FUSED_SIMILARITY_TABLE": False, "LAYOUT_FUSED_TABLE_MEMORY_FRACTION": 0.9}),
        ("половина таблицы общих бит", {"LAYOUT_FUSED_SIMILARITY_TABLE": False, "LAYOUT_FUSED_TABLE_MAX_BYTES": int(square * 0.75)}),
    ]
    if slow:
        found.append(("общие биты на лету", {"LAYOUT_FUSED_SIMILARITY_TABLE": False, "LAYOUT_FUSED_TABLE_MAX_BYTES": 0}))
    return found


def last_source() -> str:
    """Источник сходства последнего слитого расчёта — из журнала (событие layout.fused.energy.init)."""
    found = "?"
    try:
        with open(os.path.join(ROOT, "logs", "damp.jsonl"), encoding="utf-8", errors="replace") as log:
            for line in log:
                if "layout.fused.energy.init" in line:
                    found = str((json.loads(line).get("data") or {}).get("source"))
    except Exception:  # noqa: BLE001
        pass
    return found


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    number = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    copies = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    slow = len(sys.argv) > 5 and sys.argv[5] == "all"
    from damp_api import Damp
    from damp import settings
    from damp.layout.level_memory import MemoryMap

    base = json.load(open(os.path.join(ROOT, "tools", "tuning", "stand", "base.json"), encoding="utf-8-sig"))
    damp = Damp(changes={**base, **LIGHT, "ENCODER_RANDOM_SEED": seed, "DASHBOARD_FILE_PREFIX": "data/tune/ladder"})
    ladder = damp.cycle.trained(500)
    run = ladder._ladder
    memory = run.primary.memory if number == 0 else next(level.memory for level in run.levels if int(level.number) == number)
    bits = np.asarray(memory.layout.code_table.bits(), dtype=np.uint8)
    labels = np.asarray([int(v) for v in memory.layout.code_table.labels()], dtype=np.int64)
    if copies > 1:
        # Копия k получает свой добавочный бит: коды копий одной точки почти равны, но различны (слияния точек нет).
        marks = np.zeros((bits.shape[0] * copies, copies), dtype=np.uint8)
        for k in range(copies):
            marks[k * bits.shape[0] : (k + 1) * bits.shape[0], k] = 1
        bits = np.concatenate([np.tile(bits, (copies, 1)), marks], axis=1)
        labels = np.tile(labels, copies)
    del ladder, run
    import torch

    torch.cuda.empty_cache()
    ones = bits.sum(axis=1, dtype=np.int64)
    print(f"карта уровня {number}: точек {bits.shape[0]}, длина кода {bits.shape[1]}, единиц в коде до {int(ones.max())} "
          f"(в среднем {float(ones.mean()):.0f}); дальняя фаза — {steps} шагов", flush=True)
    short = {"LAYOUT_TUNE_LONG_STEPS_MIN": steps, "LAYOUT_TUNE_LONG_STEPS_MAX": steps}
    found = []
    print("путь | источник по журналу | секунд всего | видеопамять, пик ГБ | расстановка как у первого пути", flush=True)
    for name, changes in tiers(int(bits.shape[0]), 1 if int(ones.max()) <= 255 else 2, slow):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            with settings.overridden({**base, **LIGHT, **short, **changes}):
                layout = MemoryMap().lay_out(bits, labels, f"level{number}")
            cells = np.asarray(layout.cells_array(), dtype=np.int64)
            layout.release_device()
        except Exception as error:  # noqa: BLE001
            print(f"{name} | ошибка: {repr(error)[:200]}", flush=True)
            torch.cuda.empty_cache()
            continue
        seconds = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated() / 2**30
        same = "—" if not found else ("да" if np.array_equal(found[0], cells) else "НЕТ")
        found.append(cells)
        print(f"{name} | {last_source()} | {seconds:.1f} | {peak:.2f} | {same}", flush=True)
    print("Итого: расстановка всех путей должна совпадать; время — на весь проход (дальняя фаза указанной длины и ближняя)", flush=True)


if __name__ == "__main__":
    main()
