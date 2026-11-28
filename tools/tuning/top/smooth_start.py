# -*- coding: utf-8 -*-
"""Стенд верха: гладкий старт (клетки из tsne_cells.py) → доводка движком с малого радиуса, и честная опора — много раскладок
со случайного старта (ошибка 2026-11-23: опора по 2–3 раскладкам на набор кодов была занижена на 0.022, и все Δ завышены).

    python tools/tuning/top/smooth_start.py [зёрна=1,3,0,2,4,5] [радиусы=0.1,0.2] [раскладок опоры=6] [уровень=5]
Итоги — data/top-stand/smooth-start.jsonl. Один-два потока, стенд на это время остановить.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import top_stand as T  # noqa: E402
import numpy as np  # noqa: E402

OUT = os.path.join(T.FOLDER, "smooth-start.jsonl")


def main():
    seeds = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "1,3,0,2,4,5").split(",")]
    radii = [float(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "0.1,0.2").split(",")]
    base_draws = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    level = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    method = sys.argv[5] if len(sys.argv) > 5 else "tsne"  # чьи стартовые клетки: tsne | umap15 | umap30 (tsne_cells.py)
    title = "гладкий старт" if method == "tsne" else f"гладкий старт ({method})"
    from damp import settings
    from damp.layout.level_memory import MemoryMap
    from damp.layout.mnist_layout import _run_long_phase, _run_short_phase

    base = json.load(open(os.path.join(T.ROOT, "tools", "tuning", "stand", "base.before-v18.1.json"), encoding="utf-8-sig"))
    light = {"LAYOUT_LOG_VISUALS": False, "ENCODING_WORKERS": 4, "ACTIVATION_PARALLEL_WORKERS": 2, "LADDER_CLASS_MAP": False}
    for seed in seeds:
        z = np.load(os.path.join(T.FOLDER, f"dump-seed{seed}.npz"))
        bits = np.ascontiguousarray(np.unpackbits(z[f"L{level}_bits"], axis=1)[:, : int(z[f"L{level}_length"])])
        labels, picture = z[f"L{level}_labels"], z[f"L{level}_picture"]
        height, width = [int(v) for v in z[f"L{level}_size"]]
        starts = np.load(os.path.join(T.FOLDER, f"{method}-L{level}-seed{seed}.npz"))

        def write(name, s, cells, began):
            m1, patches = T.m1_by_digit(cells, labels, height, width)
            row = {"seed": seed, "level": level, "variant": name, "layout_seed": s, "m1u": float(np.mean([m1.get(d, 0.0) for d in range(10)])),
                   "m1": [round(m1.get(d, 0.0), 3) for d in range(10)], "patches": patches, "honest": T.map_purity(cells, labels, picture),
                   "seconds": round(time.time() - began, 1)}
            with open(OUT, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{name} зерно {seed} раскладка {s}: М1u {row['m1u']:.3f} честная {row['honest']:.3f} пятен {patches} ({row['seconds']} с)", flush=True)

        with settings.overridden({**base, **light}):
            schedule = MemoryMap.lambda_schedule(bits, f"smooth-{seed}")
            params = MemoryMap._tuner().tune(bits.shape[0])
            for key in starts.files:
                state = int(key[5:])
                write(f"{title}, без доводки", state, starts[key], time.time())
                for share in radii:
                    began = time.time()
                    with settings.overridden({"LAYOUT_RANDOM_SEED": int(settings.LAYOUT_RANDOM_SEED) + 1 + state}):
                        layout = MemoryMap().restore(bits, labels, starts[key], width, schedule.start, None)
                        radius = max(int(settings.LAYOUT_ADAPTIVE_RADIUS_MIN), int(round(width * share)))
                        _run_long_phase(layout, params, step_offset=1, layout_path=None, start_radius=radius, lambda_max=schedule.maximum)
                        steps = int(layout.last_steps)
                        _run_short_phase(layout, params, step_offset=1 + steps, similarity_cutoff=schedule.short_cutoff, lambda_max=schedule.maximum)
                        cells = np.asarray(layout.cells_array(), dtype=np.int64)
                        assert np.array_equal(np.asarray([int(v) for v in layout.code_table.labels()]), labels), "порядок точек не тот"
                        layout.release_device()
                    write(f"{title} → доводка, радиус {share:g}·W", state, cells, began)
            for s in range(11, 11 + base_draws):  # опора: случайный старт, свои зёрна раскладки
                began = time.time()
                with settings.overridden({"LAYOUT_RANDOM_SEED": int(settings.LAYOUT_RANDOM_SEED) + s}):
                    fresh = MemoryMap().lay_out(bits, labels, f"level{level}")
                cells = np.asarray(fresh.cells_array(), dtype=np.int64)
                fresh.release_device()
                write("случайный старт (опора)", s, cells, began)


if __name__ == "__main__":
    main()
