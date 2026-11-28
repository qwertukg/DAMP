# -*- coding: utf-8 -*-
"""Стенд верха на кодах из кеша шагов: коды верхнего уровня прогонов стенда берутся из кеша (без обучения цикла),
верхняя карта раскладывается заново с пробными настройками.

    python tools/tuning/top/cache_top.py export <номер прогона> [<номер> …]
    python tools/tuning/top/cache_top.py lay <имя серии> <зёрна через запятую> <раскладок на зерно> '<JSON: имя → настройки>'
    python tools/tuning/top/cache_top.py sum <имя серии>

export (процессор): по журналу прогона находит записи кеша «окна верхнего уровня» (cycle.keyholes — картинка каждого окна) и
«коды окон» (coding.keyholes), пишет data/top-stand/top-seed<зерно>-<картинок>.npz: коды, метки MNIST, картинки.
lay (видеокарта, карта в сотни точек — секунды): раскладывает коды зерна настройками опоры стенда + настройки варианта
зёрнами раскладки 0 … n−1; М1 по цифрам, М1u, число пятен — в data/top-stand/<имя серии>.jsonl (готовые строки не
пересчитываются). sum — таблица серии: вариант, М1u ± ошибка среднего, Δ к варианту «опора» по парам (зерно, раскладка).
"""
import glob
import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "client"), os.path.join(ROOT, "system"), os.path.dirname(os.path.abspath(__file__))]
os.environ["RERUN"] = "off"
os.environ["DAMP_RERUN_VIEWER"] = "off"
os.environ.setdefault("DAMP_LOG_CONSOLE", "heartbeat")
FOLDER = os.path.join(ROOT, "data", "top-stand")
STAND = os.path.join(ROOT, "tools", "tuning", "stand")


def record(cache: str, step: str, key: str) -> dict:
    found = glob.glob(os.path.join(cache, "steps", step, f"{key}*.json"))
    if len(found) != 1:
        raise SystemExit(f"запись кеша {step} {key}: найдено {len(found)}")
    return json.load(open(found[0], encoding="utf-8"))


def blob(cache: str, item: dict) -> "np.ndarray":
    return np.load(os.path.join(cache, "blobs", item["blob"][:2], f"{item['blob']}.npy"), allow_pickle=False)


def export(numbers: list) -> None:
    from damp import settings

    cache = str(settings.CACHE_DIR)
    mnist = np.frombuffer(open(os.path.join(ROOT, "data", "MNIST", "raw", "train-labels-idx1-ubyte"), "rb").read(), dtype=np.uint8, offset=8)
    for number in numbers:
        spec = json.load(open(os.path.join(STAND, f"tune2-{number}.json"), encoding="utf-8"))
        seed = int(spec["changes"].get("ENCODER_RANDOM_SEED", 0))
        run = re.search(r"run=(\S+)", open(os.path.join(STAND, f"tune2-{number}.out"), encoding="utf-8", errors="replace").read(4000)).group(1)
        keys: dict = {}
        for name in sorted(glob.glob(os.path.join(ROOT, "logs", "damp.jsonl*")), key=os.path.getmtime):
            for line in open(name, encoding="utf-8", errors="replace"):
                if run not in line or "keyholes" not in line or '"cache.' not in line:
                    continue
                event = json.loads(line)
                if event.get("run") == run and event.get("event") in ("cache.miss", "cache.hit"):
                    keys[(event["data"]["step"], int(event.get("cycle", -1)))] = event["data"]["key"]
        top = max(cycle for (step, cycle) in keys if step == "coding.keyholes")
        codes = record(cache, "coding.keyholes", keys[("coding.keyholes", top)])["outputs"]
        holes = record(cache, "cycle.keyholes", keys[("cycle.keyholes", top)])["outputs"]
        packed, picture = blob(cache, codes["packed"]), blob(cache, holes["picture"]).astype(np.int64)
        # Обучающих картинок в прогоне — из подписи задания стенда («750 цифр: …»); окна проверочных картинок (номера
        # от этого числа) в раскладку не идут. У картинки без окна кода нет — кодов бывает меньше, чем картинок.
        taught = int(re.match(r"(\d+) цифр", spec["label"]).group(1))
        if packed.shape[0] != picture.shape[0]:
            raise SystemExit(f"прогон {number}: кодов {packed.shape[0]}, окон {picture.shape[0]}")
        rows = np.flatnonzero(picture < taught)
        path = os.path.join(FOLDER, f"top-seed{seed}-{taught}.npz")
        np.savez_compressed(path, bits=packed[rows], length=int(codes["code_length"]["value"]), labels=mnist[picture[rows]].astype(np.int64),
                            picture=picture[rows], level=top, run=number)
        print(f"прогон {number}, зерно {seed}: уровень {top}, кодов {rows.size} из {packed.shape[0]} окон, длина "
              f"{codes['code_length']['value']} → {os.path.relpath(path, ROOT)}", flush=True)


def criteria(bits: "np.ndarray", cells: "np.ndarray", threshold: float) -> tuple[float, float]:
    """Мерки раскладки без меток (коды — в порядке точек карты): дальняя энергия статьи φ = Σ s·d по парам с s ≥ λ
    (§5.5.1; MemoryMap.far_energy) и среднее расстояние на карте до 10 ближайших по коду соседей. Меньше — лучше."""
    from damp.layout.level_memory import MemoryMap

    codes = np.asarray(bits, dtype=np.float32)
    norms = np.sqrt(np.maximum(codes.sum(axis=1), 1e-9))
    sim = (codes @ codes.T) / (norms[:, None] * norms[None, :])
    np.fill_diagonal(sim, -1.0)
    nearest = np.argsort(-sim, axis=1)[:, :10]
    place = np.asarray(cells, dtype=np.float64)
    near = float(np.sqrt(((place[:, None, :] - place[nearest]) ** 2).sum(axis=2)).mean())
    return float(MemoryMap.far_energy(bits, cells, threshold)), near


def lay_series(name: str, seeds: list, layouts: int, variants: dict) -> None:
    from extra_pictures import LIGHT, base_settings, gathering, lay

    out = os.path.join(FOLDER, f"{name}.jsonl")
    done = set()
    if os.path.exists(out):
        for line in open(out, encoding="utf-8"):
            row = json.loads(line)
            done.add((row["seed"], row["variant"], row["layout_seed"]))
    base = base_settings()
    for seed in seeds:
        path = sorted(glob.glob(os.path.join(FOLDER, f"top-seed{seed}-*.npz")))[-1]
        data = np.load(path)
        bits = np.ascontiguousarray(np.unpackbits(data["bits"], axis=1)[:, : int(data["length"])])
        labels, level = data["labels"].astype(np.int64), int(data["level"])
        for variant, changes in variants.items():
            for s in range(layouts):
                if (seed, variant, s) in done:
                    continue
                start = time.time()
                cells, got, order, size, lam = lay(bits, labels, level, {**base, **LIGHT, **changes}, s)
                m1, patches, _ = gathering(cells, got, size[0], size[1], {})
                far, near = criteria(bits[order], cells, lam)
                row = {"seed": seed, "variant": variant, "layout_seed": s, "points": int(cells.shape[0]), "size": list(size),
                       "m1u": float(np.mean([m1.get(d, 0.0) for d in range(10)])), "m1": [round(m1.get(d, 0.0), 3) for d in range(10)],
                       "patches": patches, "lambda": lam, "far": far, "near": near, "set": changes,
                       "seconds": round(time.time() - start, 1)}
                with open(out, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"{variant} | зерно {seed} раскладка {s}: точек {row['points']}, М1u {row['m1u']:.3f}, двойка {row['m1'][2]:.2f}, "
                      f"пятёрка {row['m1'][5]:.2f}, пятен {patches} ({row['seconds']} с)", flush=True)


def summary(name: str) -> None:
    rows = [json.loads(line) for line in open(os.path.join(FOLDER, f"{name}.jsonl"), encoding="utf-8")]
    variants = list(dict.fromkeys(row["variant"] for row in rows))
    by = {variant: {(row["seed"], row["layout_seed"]): row for row in rows if row["variant"] == variant} for variant in variants}
    first = by[variants[0]]
    print(f"вариант | раскладок | М1u | двойка | пятёрка | пятен | Δ М1u к «{variants[0]}» (те же зерно и раскладка) | t")
    for variant in variants:
        mine = by[variant]
        m1u = np.array([row["m1u"] for row in mine.values()])
        common = sorted(set(mine) & set(first))
        delta = np.array([mine[key]["m1u"] - first[key]["m1u"] for key in common])
        error = delta.std(ddof=1) / np.sqrt(delta.size) if delta.size > 1 else float("nan")
        spread = m1u.std(ddof=1) / np.sqrt(m1u.size) if m1u.size > 1 else float("nan")
        print(f"{variant} | {m1u.size} | {m1u.mean():.3f} ± {spread:.3f} | {np.mean([row['m1'][2] for row in mine.values()]):.2f} | "
              f"{np.mean([row['m1'][5] for row in mine.values()]):.2f} | {np.mean([row['patches'] for row in mine.values()]):.1f} | "
              + ("—" if variant == variants[0] else f"{delta.mean():+.3f} ± {error:.3f}") + " | "
              + ("—" if variant == variants[0] or not error > 0 else f"{delta.mean() / error:.1f}"))
    print(f"Итого | {len(rows)} | вариантов {len(variants)}, зёрен {len({row['seed'] for row in rows})} | | | | Δ — по парам, шум раскладки вычтен | ")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "export":
        export([int(value) for value in sys.argv[2:]])
    elif mode == "lay":
        lay_series(sys.argv[2], [int(value) for value in sys.argv[3].split(",")], int(sys.argv[4]), json.loads(sys.argv[5]))
    elif mode == "sum":
        summary(sys.argv[2])
