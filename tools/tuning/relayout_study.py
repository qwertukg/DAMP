# -*- coding: utf-8 -*-
"""Код или раскладка: верхняя карта обученной лестницы раскладывается заново на ТЕХ ЖЕ кодах с другими зёрнами раскладки.

    python tools/tuning/relayout_study.py [зерно прогона=0] [зёрна раскладки=1,2,3,4,5] [уровень=верхний]

Если М1 цифры сильно скачет от зерна раскладки — её рвёт раскладка (застревание), а не код; если держится — дело в кодах.
Опора — tools/tuning/base.json + облегчённые настройки стенда, обучение из кеша. Итог — data/maps/class-study/relayout-*.json.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sigma_levels as SL  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

EIGHT = np.ones((3, 3), bool)


def m1_by_digit(cells, labels, height, width) -> dict:
    """М1 по цифрам — как class_gathering в tune_cycle.py (8 соседей, пустая клетка пятно не рвёт)."""
    grid = np.full((int(height), int(width)), -1, dtype=np.int64)
    grid[cells[:, 0], cells[:, 1]] = labels
    empty = grid < 0
    out = {}
    for d in range(10):
        mask = grid == d
        if not mask.any():
            continue
        joined = mask | (empty & ndimage.binary_dilation(mask, structure=EIGHT))
        parts, _ = ndimage.label(joined, structure=EIGHT)
        sizes = np.bincount(parts[mask])[1:]
        out[d] = {"points": int(mask.sum()), "main": float(sizes.max() / mask.sum()), "patches": int((sizes >= 5).sum())}
    return out

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
LAYOUT_SEEDS = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "1,2,3,4,5").split(",")]
LEVEL = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] not in ("", "top") else None
# Правки раскладки только для повторной раскладки (JSON), например {"LAYOUT_SMOOTH_PINWHEELS": false} — режим статьи.
OVERRIDE = (json.load(open(sys.argv[4], encoding="utf-8")) if sys.argv[4].endswith(".json") else json.loads(sys.argv[4])) if len(sys.argv) > 4 else {}
TAG = sys.argv[5] if len(sys.argv) > 5 else ""


def main() -> None:
    from damp import settings
    from damp.layout.level_memory import MemoryMap

    base = json.load(open(os.path.join(HERE, "base.json"), encoding="utf-8"))
    light = {"LAYOUT_LOG_VISUALS": False, "ENCODING_WORKERS": 4, "ACTIVATION_PARALLEL_WORKERS": 2, "LADDER_CLASS_MAP": False}
    changes = {**base, **light, "DASHBOARD_FILE_PREFIX": "data/tune/ladder"}
    if SEED:
        changes["ENCODER_RANDOM_SEED"] = SEED
    damp = SL.Damp(changes=changes)
    ladder = damp.cycle.trained(250)
    run = ladder._ladder
    levels = {int(item.number): item for item in run.levels}
    L = LEVEL or max(levels)
    layout = levels[L].memory.layout
    bits = np.asarray(layout.code_table.bits(), dtype=np.uint8)
    labels = np.asarray([int(v) for v in layout.code_table.labels()], dtype=np.int64)
    cached = m1_by_digit(np.asarray(layout.cells_array(), dtype=np.int64), labels, layout.height, layout.width)
    rows = [{"seed": "прогон", "m1": cached, "seconds": 0.0}]
    original = settings.LAYOUT_RANDOM_SEED
    print(f"уровень {L}: точек {labels.size}, зерно раскладки в прогоне {original}")
    print("прогон", " ".join(f"{d}:{v['main']:.2f}" for d, v in cached.items()))
    print("правки раскладки:", OVERRIDE or "нет")
    for s in ([original] if not OVERRIDE else []) + LAYOUT_SEEDS:
        settings.LAYOUT_RANDOM_SEED = s
        start = time.time()
        with settings.overridden(OVERRIDE):
            fresh = MemoryMap().lay_out(bits, labels, f"relayout-L{L}-s{s}")
        cells = np.asarray(fresh.cells_array(), dtype=np.int64)
        got_labels = np.asarray([int(v) for v in fresh.code_table.labels()], dtype=np.int64)
        m1 = m1_by_digit(cells, got_labels, fresh.height, fresh.width)
        total = sum(v["main"] * v["points"] for v in m1.values()) / got_labels.size
        radius = int(settings.LAYOUT_ENERGY_RADIUS)
        energies = np.asarray(fresh.point_energies(radius), dtype=np.float64)
        quality = float(fresh.average_local_energy(radius))  # Ē (§5.8): Ê = E / E_max, среднее
        rows.append({"seed": s, "m1": m1, "total": total, "energy_mean": float(energies.mean()), "quality": quality, "seconds": time.time() - start, "cells": cells.tolist(), "labels": got_labels.tolist(),
                     "height": int(fresh.height), "width": int(fresh.width)})
        print(f"зерно {s}: М1 {total:.3f} Ē {quality:.4f} E {energies.mean():.4f} ({time.time() - start:.0f} с)", " ".join(f"{d}:{v['main']:.2f}" for d, v in m1.items()), flush=True)
    settings.LAYOUT_RANDOM_SEED = original
    os.makedirs("data/maps/class-study", exist_ok=True)
    json.dump(rows, open(f"data/maps/class-study/relayout-seed{SEED}-L{L}{TAG}.json", "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()
