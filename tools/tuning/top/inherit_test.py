# -*- coding: utf-8 -*-
"""Проверка v19.10 (наследование топографии) на стенде верха: точки карты ур. N стартуют не со случайных клеток, а с мест
своих деталей на карте ур. N−1, затем движок доводит раскладку с малого радиуса.

    python tools/tuning/top/inherit_test.py [зёрна=1,3,0,2,4,5] [раскладки=1,2] [радиусы=0.1,0.2,0.3] [уровень=5]

Приближение по выгрузке: окно ур. 5 у 87.5 % картинок содержит все детали картинки, поэтому место старта точки — центр
тяжести клеток точек ур. N−1 той же картинки (у всех окон картинки одно, с малым разбросом). Итоги —
data/top-stand/inherit.jsonl. Один поток, видеопамять около 0.5 ГБ.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import top_stand as T  # noqa: E402  (ставит пути к client и system, RERUN=off)
import numpy as np  # noqa: E402

OUT = os.path.join(T.FOLDER, "inherit.jsonl")


def snap(xy, height, width):
    """Точки плоскости → клетки сетки без повторов: рекурсивное деление пополам (соседство сохраняется)."""
    sys.setrecursionlimit(100000)
    cells = np.zeros((xy.shape[0], 2), dtype=np.int64)

    def assign(idx, y0, y1, x0, x1):
        n = idx.shape[0]
        if n == 0:
            return
        if (y1 - y0) * (x1 - x0) == 1:
            cells[idx[0]] = (y0, x0)
            return
        capacity = (y1 - y0) * (x1 - x0)
        if (y1 - y0) >= (x1 - x0):
            mid = (y0 + y1) // 2
            cap1 = (mid - y0) * (x1 - x0)
            order = idx[np.argsort(xy[idx, 0], kind="stable")]
            n1 = int(min(max(round(n * cap1 / capacity), n - (capacity - cap1)), min(n, cap1)))
            assign(order[:n1], y0, mid, x0, x1)
            assign(order[n1:], mid, y1, x0, x1)
        else:
            mid = (x0 + x1) // 2
            cap1 = (y1 - y0) * (mid - x0)
            order = idx[np.argsort(xy[idx, 1], kind="stable")]
            n1 = int(min(max(round(n * cap1 / capacity), n - (capacity - cap1)), min(n, cap1)))
            assign(order[:n1], y0, y1, x0, mid)
            assign(order[n1:], y0, y1, mid, x1)

    assign(np.arange(xy.shape[0]), 0, int(height), 0, int(width))
    return cells


def inherited(z, level, rng):
    """Место старта каждой точки ур. level: центр тяжести клеток точек ур. level − 1 её картинки, в долях карты."""
    picture = z[f"L{level}_picture"]
    below_picture, below_cells = z[f"L{level - 1}_picture"], z[f"L{level - 1}_cells"].astype(np.float64)
    below_size = z[f"L{level - 1}_size"].astype(np.float64)
    centre = {}
    for p in np.unique(below_picture):
        centre[int(p)] = below_cells[below_picture == p].mean(axis=0) / below_size
    middle = below_cells.mean(axis=0) / below_size
    xy = np.asarray([centre.get(int(p), middle) for p in picture])
    return xy + rng.normal(0.0, 0.004, size=xy.shape)


def main():
    seeds = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "1,3,0,2,4,5").split(",")]
    layout_seeds = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "1,2").split(",")]
    radii = [float(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "0.1,0.2,0.3").split(",")]
    level = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    from damp import settings
    from damp.layout.level_memory import MemoryMap
    from damp.layout.mnist_layout import _run_long_phase, _run_short_phase

    base = json.load(open(os.path.join(T.ROOT, "tools", "tuning", "stand", "base.before-v18.1.json"), encoding="utf-8-sig"))
    light = {"LAYOUT_LOG_VISUALS": False, "ENCODING_WORKERS": 4, "ACTIVATION_PARALLEL_WORKERS": 2, "LADDER_CLASS_MAP": False}
    for seed in seeds:
        z = np.load(os.path.join(T.FOLDER, f"dump-seed{seed}.npz"))
        bits = np.unpackbits(z[f"L{level}_bits"], axis=1)[:, : int(z[f"L{level}_length"])]
        labels, picture = z[f"L{level}_labels"], z[f"L{level}_picture"]
        height, width = [int(v) for v in z[f"L{level}_size"]]
        groups = list(dict.fromkeys(labels.tolist()))  # выгрузка уже в порядке раскладки (по цвету класса)
        order = np.concatenate([np.flatnonzero(labels == g) for g in groups])
        bits_o, labels_o, picture_o = np.ascontiguousarray(bits[order]), labels[order], picture[order]
        xy = inherited(z, level, np.random.default_rng(seed))[order]
        start_cells = snap(xy * np.array([height, width]), height, width)

        def write(name, s, cells, began):
            m1, patches = T.m1_by_digit(cells, labels_o, height, width)
            row = {"seed": seed, "level": level, "variant": name, "layout_seed": s, "m1u": float(np.mean([m1.get(d, 0.0) for d in range(10)])),
                   "m1": [round(m1.get(d, 0.0), 3) for d in range(10)], "patches": patches, "honest": T.map_purity(cells, labels_o, picture_o),
                   "seconds": round(time.time() - began, 1)}
            with open(OUT, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{name} зерно {seed} раскладка {s}: М1u {row['m1u']:.3f} честная {row['honest']:.3f} пятен {patches} ({row['seconds']} с)", flush=True)

        write("старт с карты ниже, без доводки", 0, start_cells, time.time())
        with settings.overridden({**base, **light}):
            schedule = MemoryMap.lambda_schedule(bits_o, f"inherit-{seed}")
            params = MemoryMap._tuner().tune(bits_o.shape[0])
            for share in radii:
                for s in layout_seeds:
                    began = time.time()
                    with settings.overridden({"LAYOUT_RANDOM_SEED": int(settings.LAYOUT_RANDOM_SEED) + s}):
                        layout = MemoryMap().restore(bits_o, labels_o, start_cells, width, schedule.start, None)
                        radius = max(int(settings.LAYOUT_ADAPTIVE_RADIUS_MIN), int(round(width * share)))
                        _run_long_phase(layout, params, step_offset=1, layout_path=None, start_radius=radius, lambda_max=schedule.maximum)
                        steps = int(layout.last_steps)
                        _run_short_phase(layout, params, step_offset=1 + steps, similarity_cutoff=schedule.short_cutoff, lambda_max=schedule.maximum)
                        cells = np.asarray(layout.cells_array(), dtype=np.int64)
                        assert np.array_equal(np.asarray([int(v) for v in layout.code_table.labels()]), labels_o), "порядок точек не тот"
                        layout.release_device()
                    write(f"старт с карты ниже → доводка, радиус {share:g}·W", s, cells, began)


if __name__ == "__main__":
    main()
