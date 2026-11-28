# -*- coding: utf-8 -*-
"""Стенд верха, две фазы со сменой кода (§5.13.2.4: сначала грубые коды с большим перекрытием, потом тонкие; §7.1.8.5: замена
кода на месте): первая фаза раскладывает «грубые» коды со случайного старта, вторая — настоящие коды с готовых клеток и малого
радиуса.

    python tools/tuning/top/two_phase.py <серии.json> [зёрна] [раскладки=1,2] [уровень=5]

Серии: [{"name": …, "first": "snn40" | "avg20" | "base", "radius": доля ширины для второй фазы}]. Грубые коды (без меток):
snnK — набор K ближайших по коду (сходство = доля общих соседей), avgK — биты, общие не меньше чем у половины K ближайших.
Итоги — data/top-stand/two-phase.jsonl.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import top_stand as T  # noqa: E402
import numpy as np  # noqa: E402

OUT = os.path.join(T.FOLDER, "two-phase.jsonl")


def coarse(name, bits):
    if name == "base":
        return bits
    k = int("".join(ch for ch in name if ch.isdigit()))
    x = bits.astype(np.float32)
    x /= np.maximum(np.sqrt(x.sum(axis=1, keepdims=True)), 1e-6)
    sim = x @ x.T
    np.fill_diagonal(sim, -1.0)
    near = np.argpartition(-sim, k, axis=1)[:, :k]
    if name.startswith("snn"):
        code = np.zeros((bits.shape[0], bits.shape[0]), dtype=np.uint8)
        code[np.arange(bits.shape[0])[:, None], near] = 1
        code[np.arange(bits.shape[0]), np.arange(bits.shape[0])] = 1
        return code
    if name.startswith("avg"):
        return (bits[near].mean(axis=1) >= 0.5).astype(np.uint8)
    raise ValueError(name)


def main():
    series = json.load(open(sys.argv[1], encoding="utf-8"))
    seeds = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "1,3,0,2,4,5").split(",")]
    layout_seeds = [int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "1,2").split(",")]
    level = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    from damp import settings
    from damp.layout.level_memory import MemoryMap
    from damp.layout.mnist_layout import _run_long_phase, _run_short_phase

    base = json.load(open(os.path.join(T.ROOT, "tools", "tuning", "stand", "base.before-v18.1.json"), encoding="utf-8-sig"))
    light = {"LAYOUT_LOG_VISUALS": False, "ENCODING_WORKERS": 4, "ACTIVATION_PARALLEL_WORKERS": 2, "LADDER_CLASS_MAP": False}
    for seed in seeds:
        z = np.load(os.path.join(T.FOLDER, f"dump-seed{seed}.npz"))
        bits = np.ascontiguousarray(np.unpackbits(z[f"L{level}_bits"], axis=1)[:, : int(z[f"L{level}_length"])])
        labels, picture = z[f"L{level}_labels"], z[f"L{level}_picture"]  # выгрузка — в порядке раскладки (по цвету класса)
        cache = {}
        with settings.overridden({**base, **light}):
            schedule = MemoryMap.lambda_schedule(bits, f"two-phase-{seed}")
            params = MemoryMap._tuner().tune(bits.shape[0])
            for s in layout_seeds:
                first_cells = {}
                for item in series:
                    began = time.time()
                    with settings.overridden({"LAYOUT_RANDOM_SEED": int(settings.LAYOUT_RANDOM_SEED) + s}):
                        if item["first"] not in first_cells:
                            if item["first"] not in cache:
                                cache[item["first"]] = np.ascontiguousarray(coarse(item["first"], bits))
                            first = MemoryMap().lay_out(cache[item["first"]], labels, f"two-phase-{seed}-{s}")
                            assert np.array_equal(np.asarray([int(v) for v in first.code_table.labels()]), labels), "порядок точек не тот"
                            first_cells[item["first"]] = (np.asarray(first.cells_array(), dtype=np.int64), int(first.height), int(first.width))
                            first.release_device()
                        cells, height, width = first_cells[item["first"]]
                        layout = MemoryMap().restore(bits, labels, cells, width, schedule.start, None)
                        radius = max(int(settings.LAYOUT_ADAPTIVE_RADIUS_MIN), int(round(width * float(item["radius"]))))
                        _run_long_phase(layout, params, step_offset=1, layout_path=None, start_radius=radius, lambda_max=schedule.maximum)
                        steps = int(layout.last_steps)
                        _run_short_phase(layout, params, step_offset=1 + steps, similarity_cutoff=schedule.short_cutoff, lambda_max=schedule.maximum)
                        moved = np.asarray(layout.cells_array(), dtype=np.int64)
                        layout.release_device()
                    m1, patches = T.m1_by_digit(moved, labels, height, width)
                    row = {"seed": seed, "level": level, "variant": item["name"], "layout_seed": s,
                           "m1u": float(np.mean([m1.get(d, 0.0) for d in range(10)])), "m1": [round(m1.get(d, 0.0), 3) for d in range(10)],
                           "patches": patches, "honest": T.map_purity(moved, labels, picture), "seconds": round(time.time() - began, 1)}
                    with open(OUT, "a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(f"{item['name']} зерно {seed} раскладка {s}: М1u {row['m1u']:.3f} честная {row['honest']:.3f} пятен {patches} ({row['seconds']} с)", flush=True)


if __name__ == "__main__":
    main()
