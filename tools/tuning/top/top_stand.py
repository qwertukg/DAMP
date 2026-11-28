# -*- coding: utf-8 -*-
"""Стенд верха: карта одного уровня раскладывается заново на ТЕХ ЖЕ кодах с разными настройками раскладки.

Зачем: полный прогон меряет правку с шумом 0.036 на зерно; здесь коды закреплены, меняется только раскладка — на 6 наборах
кодов × 2–4 раскладки ошибка среднего ≈ 0.01, раскладка ур. 5 идёт 7–10 с (docs/CONVERGENCE.md).

    python tools/tuning/top/top_stand.py run <серии.json> [зёрна=1,3,0,2,4,5] [раскладки=1,2] [уровень=5]
    python tools/tuning/top/top_stand.py sum [файл итогов]

Серии — список {"name": имя, "set": {настройка: значение}}; имя "base" с пустым set — опора. Коды — выгрузка
data/top-stand/dump-seedN.npz (tools/tuning/top/dump_maps.py). Итоги — data/top-stand/layout-series.jsonl (дописываются; уже
посчитанные «зерно, уровень, серия, раскладка» пропускаются). Питон — стенда (DAMP_PYTHON); стенд на это время остановить
либо считать одним потоком: видеопамять одной раскладки ур. 5 — около 0.5 ГБ.
"""
import collections
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "client"), os.path.join(ROOT, "system")]
os.environ.setdefault("DAMP_LOG_CONSOLE", "heartbeat")
os.environ.setdefault("RERUN", "off")
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

FOLDER = os.path.join(ROOT, "data", "top-stand")
OUT = os.path.join(FOLDER, "layout-series.jsonl")
EIGHT = np.ones((3, 3), bool)


def m1_by_digit(cells, labels, height, width):
    """М1 по цифрам (доля точек цифры в главном пятне) и число пятен от 5 точек — как class_gathering стенда."""
    grid = np.full((int(height), int(width)), -1, dtype=np.int64)
    grid[cells[:, 0], cells[:, 1]] = labels
    empty = grid < 0
    out, count = {}, 0
    for d in range(10):
        mask = grid == d
        if not mask.any():
            continue
        parts, _ = ndimage.label(mask | (empty & ndimage.binary_dilation(mask, structure=EIGHT)), structure=EIGHT)
        sizes = np.bincount(parts[mask])[1:]
        out[d] = float(sizes.max() / mask.sum())
        count += int((sizes >= 5).sum())
    return out, count


def map_purity(cells, labels, picture, k=10):
    """Честная чистота: доля своего класса среди k ближайших на карте точек с других картинок."""
    tree = cKDTree(cells.astype(np.float64))
    _, near = tree.query(cells.astype(np.float64), k=min(k * 6, cells.shape[0]))
    pure = np.zeros(cells.shape[0])
    for q in range(cells.shape[0]):
        cand = near[q]
        cand = cand[picture[cand] != picture[q]][:k]
        pure[q] = (labels[cand] == labels[q]).mean() if cand.size else 0.0
    return float(pure.mean())


def run(series_path, seeds, layout_seeds, level):
    from damp import settings
    from damp.layout.level_memory import MemoryMap

    series = json.load(open(series_path, encoding="utf-8"))
    base = json.load(open(os.path.join(ROOT, "tools", "tuning", "stand", "base.json"), encoding="utf-8"))
    light = {"LAYOUT_LOG_VISUALS": False, "ENCODING_WORKERS": 4, "ACTIVATION_PARALLEL_WORKERS": 2, "LADDER_CLASS_MAP": False}
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["seed"], r["level"], r["variant"], r["layout_seed"]))
    for seed in seeds:
        z = np.load(os.path.join(FOLDER, f"dump-seed{seed}.npz"))
        bits = np.ascontiguousarray(np.unpackbits(z[f"L{level}_bits"], axis=1)[:, : int(z[f"L{level}_length"])])
        labels, picture = z[f"L{level}_labels"], z[f"L{level}_picture"]
        for item in series:
            for s in layout_seeds:
                if (seed, level, item["name"], s) in done:
                    continue
                start = time.time()
                try:
                    with settings.overridden({**base, **light, **item["set"]}):
                        with settings.overridden({"LAYOUT_RANDOM_SEED": int(settings.LAYOUT_RANDOM_SEED) + s}):
                            fresh = MemoryMap().lay_out(bits, labels, f"level{level}")  # имя уровня — чтобы действовали настройки «с уровня»
                except Exception as error:  # noqa: BLE001
                    print(item["name"], "ошибка:", repr(error)[:300], flush=True)
                    break
                cells = np.asarray(fresh.cells_array(), dtype=np.int64)
                got = np.asarray([int(v) for v in fresh.code_table.labels()], dtype=np.int64)
                groups = list(dict.fromkeys(got.tolist()))  # точки раскладки сгруппированы по цвету класса
                order = np.concatenate([np.flatnonzero(labels == g) for g in groups])
                m1, patches = m1_by_digit(cells, got, fresh.height, fresh.width)
                row = {"seed": seed, "level": level, "variant": item["name"], "layout_seed": s,
                       "m1u": float(np.mean([m1.get(d, 0.0) for d in range(10)])), "m1": [round(m1.get(d, 0.0), 3) for d in range(10)],
                       "patches": patches, "honest": map_purity(cells, got, picture[order]), "lambda": float(fresh.lambda_threshold),
                       "seconds": round(time.time() - start, 1)}
                fresh.release_device()
                with open(OUT, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"{item['name']} зерно {seed} раскладка {s}: М1u {row['m1u']:.3f} честная {row['honest']:.3f} пятен {patches} ({row['seconds']} с)", flush=True)


def summary(path=OUT):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    found = collections.defaultdict(list)
    for r in rows:
        found[r["variant"]].append(r)
    base = {(r["seed"], r["layout_seed"]): r for r in found.get("base", [])}
    print(f"{'серия':34s}   n   М1u   SD   Δ к опоре (t)  честная пятен | М1 по цифрам 0…9")
    for name, rs in found.items():
        d = np.array([r["m1u"] - base[(r["seed"], r["layout_seed"])]["m1u"] for r in rs if (r["seed"], r["layout_seed"]) in base])
        t = d.mean() / (d.std(ddof=1) / np.sqrt(d.size) + 1e-9) if d.size > 2 else 0.0
        m1 = np.mean([r["m1"] for r in rs], axis=0)
        print(f"{name[:34]:34s} {len(rs):3d} {np.mean([r['m1u'] for r in rs]):.3f} {np.std([r['m1u'] for r in rs]):.3f}  {d.mean() if d.size else 0:+.3f} ({t:+5.1f})  "
              f"{np.mean([r['honest'] for r in rs]):.3f} {np.mean([r['patches'] for r in rs]):5.1f} | " + " ".join(f"{x:.2f}" for x in m1))
    print("Итого: Δ — парная разность к опоре на тех же кодах и зерне раскладки, t — её значимость;")
    print("       серии, не меняющие счёт, дают ровно 0; общий сдвиг всех Δ от случая раскладок опоры — до ±0.014 при 12 раскладках.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run(sys.argv[2], [int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "1,3,0,2,4,5").split(",")],
            [int(v) for v in (sys.argv[4] if len(sys.argv) > 4 else "1,2").split(",")], int(sys.argv[5]) if len(sys.argv) > 5 else 5)
    else:
        summary(sys.argv[2] if len(sys.argv) > 2 else OUT)
