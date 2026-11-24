# -*- coding: utf-8 -*-
"""Новые картинки на верхней карте: уровни обучены на N картинках стенда, на верхнюю карту кладутся ещё M новых.

Цель автора 2026-11-24: «сходимость на верхнем уровне важнее, если в ней больше точек». При одном окне на картинку
точек верха — по числу картинок, а быстрый стенд держит ≈ 600 картинок (docs/SPEEDUP.md). Здесь нижние уровни остаются
обученными на картинках стенда, новая картинка идёт через них обученным путём (TrainedLadder, §6.5–6.6: «повторная
подача»), её верхнее окно получает код — и верхняя карта раскладывается на кодах старых и новых картинок вместе.

    python tools/tuning/top/extra_pictures.py codes <зерно> [новых=3500] [картинок стенда=500]
    python tools/tuning/top/extra_pictures.py maps <зерно> [раскладки=0,1,2] [серии.json]
    python tools/tuning/top/extra_pictures.py sheet <зерно> [раскладка=0] [серии.json]
    python tools/tuning/top/extra_pictures.py sum

codes — коды верхних окон обучающих и новых картинок (обученный цикл — из кеша шагов прогона опоры на этом зерне) в
data/top-stand/extra-seedN.npz; сверка: код обучающей картинки, полученный обученным путём, равен коду её точки на
карте прогона. maps — раскладки верха на наборах кодов (серии по умолчанию — SERIES) настройками опоры, итоги —
data/top-stand/extra-pictures.jsonl. sheet — лист карт серий одной раскладки (цвет — цифра, белая точка —
картинка обучения): data/top-stand/extra-seedN.png. Питон — стенда; стенд на это время не должен считать прогон (ОЗУ, видеопамять).
"""
import collections
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "client"), os.path.join(ROOT, "system"), os.path.dirname(os.path.abspath(__file__))]
os.environ.setdefault("DAMP_LOG_CONSOLE", "heartbeat")
os.environ.setdefault("RERUN", "off")
os.environ.setdefault("DAMP_RERUN_VIEWER", "off")
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

FOLDER = os.path.join(ROOT, "data", "top-stand")
OUT = os.path.join(FOLDER, "extra-pictures.jsonl")
EIGHT = np.ones((3, 3), bool)
LIGHT = {"LAYOUT_LOG_VISUALS": False, "ENCODING_WORKERS": 4, "ACTIVATION_PARALLEL_WORKERS": 2, "LADDER_CLASS_MAP": False}
# Набор кодов серии: old — сколько обучающих картинок, new — сколько новых (первые по порядку MNIST); set — правки настроек.
SERIES = [
    {"name": "свои 500", "old": 500, "new": 0, "set": {}},
    {"name": "новые 500", "old": 0, "new": 500, "set": {}},
    {"name": "свои 500 + новые 500", "old": 500, "new": 500, "set": {}},
    {"name": "свои 500 + новые 1500", "old": 500, "new": 1500, "set": {}},
    {"name": "свои 500 + новые 3500", "old": 500, "new": 3500, "set": {}},
]


def base_settings() -> dict:
    return json.load(open(os.path.join(ROOT, "tools", "tuning", "stand", "base.json"), encoding="utf-8"))


def codes(seed: int, extra: int, count: int) -> None:
    from damp_api import Damp
    from damp.cycle.trained import TrainedLadder as SystemLadder

    damp = Damp(changes={**base_settings(), **LIGHT, "ENCODER_RANDOM_SEED": seed, "DASHBOARD_FILE_PREFIX": "data/tune/ladder"})
    started = time.time()
    ladder = damp.cycle.trained(count)
    run = ladder._ladder
    top = run.levels[-1]
    number = int(top.number)
    layout = top.memory.layout
    map_bits = np.asarray(layout.code_table.bits(), dtype=np.uint8)
    length = int(map_bits.shape[1])
    map_picture = np.asarray(top.point_picture, dtype=np.int64)
    print(f"обученный цикл: {time.time() - started:.0f} с; верх — уровень {number}, точек {map_bits.shape[0]}, длина кода {length}", flush=True)

    # Код верхнего окна — то, что TrainedLadder готовит уровню перед срабатыванием детекторов (_level_prepare).
    found: dict = {}
    current = {"picture": -1}
    prepare = SystemLadder._level_prepare

    def spy(self, trained, elements):  # noqa: ANN001, ANN202
        result = prepare(self, trained, elements)
        if result is not None and int(trained.number) == number:
            found[current["picture"]] = np.asarray(result[2].packed, dtype=np.uint8).copy()
        return result

    SystemLadder._level_prepare = spy
    try:
        digits = damp.stimuli.digits(train=count + extra, test=1, train_views=())
        first, last = digits.groups["train"]
        atoms = digits._groups.atoms
        index = np.asarray(atoms.index, dtype=np.int64)
        own = set(np.asarray(run.groups.atoms.index, dtype=np.int64).tolist())
        started = time.time()
        with ladder._session.active():
            for picture in range(first, last):
                current["picture"] = picture
                image = np.asarray(digits.picture(picture).image, dtype=np.uint8)
                run._see_batch([image], [-1], [SystemLadder.atom_sticks(atoms, picture)])
                if (picture - first + 1) % 250 == 0:
                    print(f"  картинок {picture - first + 1} из {last - first}: {time.time() - started:.0f} с", flush=True)
    finally:
        SystemLadder._level_prepare = prepare
    seconds = time.time() - started
    rows, picture_of, windows = [], [], collections.Counter()
    for picture in range(first, last):
        packed = found.get(picture)
        windows[0 if packed is None else int(packed.shape[0])] += 1
        if packed is None:
            continue
        bits = np.unpackbits(packed, axis=1)[:, :length]
        rows.append(bits)
        picture_of.extend([picture] * bits.shape[0])
    bits = np.concatenate(rows).astype(np.uint8)
    picture_of = np.asarray(picture_of, dtype=np.int64)
    labels = np.asarray(atoms.labels, dtype=np.int64)[picture_of]
    is_new = np.asarray([int(index[p]) not in own for p in picture_of], dtype=bool)
    print(f"картинок {last - first} ({seconds:.0f} с, {1000 * seconds / max(1, last - first):.0f} мс на картинку); "
          f"окон верха на картинку: {dict(sorted(windows.items()))}; своих {int((~is_new).sum())}, новых {int(is_new.sum())}", flush=True)

    # Сверка: код обучающей картинки обученным путём = код её точки на карте прогона (слитые точки — код первой картинки).
    stand_index = np.asarray(run.groups.atoms.index, dtype=np.int64)
    by_mnist = {int(index[p]): row for row, p in enumerate(picture_of) if not is_new[row]}
    same = checked = 0
    for point in range(map_bits.shape[0]):
        row = by_mnist.get(int(stand_index[map_picture[point]]))
        if row is None:
            continue
        checked += 1
        same += int(np.array_equal(bits[row], map_bits[point]))
    print(f"сверка с картой прогона: код совпал до бита у {same} из {checked} точек карты", flush=True)
    np.savez_compressed(
        os.path.join(FOLDER, f"extra-seed{seed}.npz"), bits=np.packbits(bits, axis=1), length=length, labels=labels,
        picture=picture_of, mnist=index[picture_of], new=is_new, level=number, same=same, checked=checked,
        map_bits=np.packbits(map_bits, axis=1), map_labels=np.asarray([int(v) for v in layout.code_table.labels()], dtype=np.int64),
        map_cells=np.asarray(layout.cells_array(), dtype=np.int64), map_size=np.asarray([layout.height, layout.width]),
    )
    print(f"зерно {seed}: записано {bits.shape[0]} кодов", flush=True)


def gathering(cells, labels, height, width, subsets) -> tuple[dict, int, dict]:  # noqa: ANN001
    """М1 по цифрам (доля точек цифры в главном пятне, как class_gathering стенда), число пятен от 5 точек и доля точек
    каждого подмножества (имя → маска точек) в главном пятне своей цифры."""
    grid = np.full((int(height), int(width)), -1, dtype=np.int64)
    grid[cells[:, 0], cells[:, 1]] = labels
    empty = grid < 0
    m1, patches = {}, 0
    inside = np.zeros(cells.shape[0], dtype=bool)
    for digit in range(10):
        mask = grid == digit
        if not mask.any():
            continue
        parts, _ = ndimage.label(mask | (empty & ndimage.binary_dilation(mask, structure=EIGHT)), structure=EIGHT)
        sizes = np.bincount(parts[mask])
        sizes[0] = 0
        m1[digit] = float(sizes.max() / mask.sum())
        patches += int((sizes >= 5).sum())
        of_digit = labels == digit
        inside[of_digit] = parts[cells[of_digit, 0], cells[of_digit, 1]] == int(sizes.argmax())
    shares = {}
    for name, subset in subsets.items():
        per = [float(inside[subset & (labels == d)].mean()) for d in range(10) if (subset & (labels == d)).any()]
        shares[name] = float(np.mean(per)) if per else None
    return m1, patches, shares


def load_codes(seed: int):  # noqa: ANN201
    z = np.load(os.path.join(FOLDER, f"extra-seed{seed}.npz"))
    length, level = int(z["length"]), int(z["level"])
    bits = np.ascontiguousarray(np.unpackbits(z["bits"], axis=1)[:, :length])
    return bits, z["labels"], z["picture"], z["new"], level


def lay(bits, labels, level: int, changes: dict, s: int, embed: "dict | None" = None):  # noqa: ANN001, ANN201
    # Раскладка кодов верха настройками changes и зерном раскладки s: клетки, цифры точек (в порядке раскладки), порядок
    # исходных строк, размер карты, порог λ. embed — параметры t-SNE гладкого старта для пробы (iterations, explore,
    # exaggeration): в прогоне они не настройки, здесь подменяются на время раскладки.
    import functools

    from damp import settings
    from damp.layout import smooth_start
    from damp.layout.level_memory import MemoryMap

    original = smooth_start.embed
    if embed:
        smooth_start.embed = functools.partial(original, **embed)
    try:
        with settings.overridden(changes):
            with settings.overridden({"LAYOUT_RANDOM_SEED": int(settings.LAYOUT_RANDOM_SEED) + s}):
                fresh = MemoryMap().lay_out(bits, labels, f"level{level}")  # имя уровня — чтобы действовали настройки «с уровня»
    finally:
        smooth_start.embed = original
    cells = np.asarray(fresh.cells_array(), dtype=np.int64)
    got = np.asarray([int(v) for v in fresh.code_table.labels()], dtype=np.int64)
    groups = list(dict.fromkeys(got.tolist()))  # точки раскладки сгруппированы по цвету класса
    order = np.concatenate([np.flatnonzero(labels == g) for g in groups])
    found = cells, got, order, (int(fresh.height), int(fresh.width)), float(fresh.lambda_threshold)
    fresh.release_device()
    return found


def maps(seed: int, layout_seeds: list, series: list) -> None:
    from top_stand import map_purity

    bits, labels, picture, new, level = load_codes(seed)
    old_rows, new_rows = np.flatnonzero(~new), np.flatnonzero(new)
    base = base_settings()
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["seed"], r["variant"], r["layout_seed"]))
    for item in series:
        chosen = np.concatenate([old_rows[: int(item["old"])], new_rows[: int(item["new"])]])
        if chosen.size == 0 or new_rows.size < int(item["new"]):
            print(item["name"], "— кодов не хватает, пропуск", flush=True)
            continue
        for s in layout_seeds:
            if (seed, item["name"], s) in done:
                continue
            start = time.time()
            cells, got, order, size, lam = lay(bits[chosen], labels[chosen], level, {**base, **LIGHT, **item.get("set", {})}, s,
                                               item.get("embed"))
            fresh_new = new[chosen][order]
            m1, patches, shares = gathering(cells, got, size[0], size[1], {"old": ~fresh_new, "new": fresh_new})
            row = {"seed": seed, "variant": item["name"], "layout_seed": s, "points": int(chosen.size),
                   "old": int((~fresh_new).sum()), "new": int(fresh_new.sum()), "size": list(size),
                   "m1u": float(np.mean([m1.get(d, 0.0) for d in range(10)])), "m1": [round(m1.get(d, 0.0), 3) for d in range(10)],
                   "in_patch_old": shares["old"], "in_patch_new": shares["new"], "patches": patches,
                   "honest": map_purity(cells, got, picture[chosen][order]), "lambda": lam,
                   "set": item.get("set", {}), "seconds": round(time.time() - start, 1)}
            with open(OUT, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{item['name']} зерно {seed} раскладка {s}: точек {row['points']}, М1u {row['m1u']:.3f}, честная {row['honest']:.3f}, "
                  f"пятен {patches} ({row['seconds']} с)", flush=True)


def sheet(seed: int, s: int, series: list) -> None:
    # Лист карт серий одной раскладки: цвет — цифра (палитра карт стенда), белая точка — картинка обучения.
    from PIL import Image, ImageDraw, ImageFont
    from top_stand import map_purity

    bits, labels, picture, new, level = load_codes(seed)
    old_rows, new_rows = np.flatnonzero(~new), np.flatnonzero(new)
    base = base_settings()
    colours = [tuple(int(c) for c in Image.new("HSV", (1, 1), (int(d * 25.5), 220, 255)).convert("RGB").getpixel((0, 0))) for d in range(10)]
    side, pad, top = 380, 20, 84
    image = Image.new("RGB", (pad + len(series) * (side + pad), top + side + 44), "black")
    draw = ImageDraw.Draw(image)
    try:
        big, small = ImageFont.truetype("arialbd.ttf", 19), ImageFont.truetype("arial.ttf", 14)
    except OSError:
        big = small = ImageFont.load_default()
    draw.text((pad, 10), f"Новые картинки на верхней карте: уровни обучены на картинках стенда (зерно {seed}, раскладка {s}); "
              "белая точка — картинка обучения", fill="white", font=big)
    for n, item in enumerate(series):
        chosen = np.concatenate([old_rows[: int(item["old"])], new_rows[: int(item["new"])]])
        cells, got, order, size, _ = lay(bits[chosen], labels[chosen], level, {**base, **LIGHT, **item.get("set", {})}, s)
        fresh_new = new[chosen][order]
        m1, _, _ = gathering(cells, got, size[0], size[1], {})
        m1u = float(np.mean([m1.get(d, 0.0) for d in range(10)]))
        honest = map_purity(cells, got, picture[chosen][order])
        left, unit = pad + n * (side + pad), side / max(size)
        draw.text((left, 40), f"{item['name']}: {chosen.size} точек", fill="white", font=small)
        draw.text((left, 59), f"М1u {m1u:.3f} · честная {honest:.3f}", fill=(200, 200, 200), font=small)
        for (y, x), digit, fresh in zip(cells, got, fresh_new):
            draw.rectangle((left + x * unit, top + y * unit, left + (x + 1) * unit - 1, top + (y + 1) * unit - 1), fill=colours[int(digit)])
            if not fresh and item["new"] and unit >= 4:
                c = unit / 2
                draw.rectangle((left + x * unit + c - 1, top + y * unit + c - 1, left + x * unit + c, top + y * unit + c), fill="white")
        print(f"{item['name']}: точек {chosen.size}, М1u {m1u:.3f}, честная {honest:.3f}", flush=True)
    for d in range(10):
        draw.rectangle((pad + d * 56, top + side + 16, pad + d * 56 + 14, top + side + 30), fill=colours[d])
        draw.text((pad + d * 56 + 20, top + side + 15), str(d), fill="white", font=small)
    path = os.path.join(FOLDER, f"extra-seed{seed}.png")
    image.save(path)
    print("лист карт:", path, flush=True)


def summary() -> None:
    rows = [json.loads(line) for line in open(OUT, encoding="utf-8")]
    found = collections.defaultdict(list)
    for r in rows:
        found[r["variant"]].append(r)
    print(f"{'серия':30s}  n  точек   М1u    SD  свои в пятне  новые в пятне  честная  пятен | М1 по цифрам 0…9")
    for name, rs in found.items():
        def mean(key):  # noqa: ANN001, ANN202
            values = [r[key] for r in rs if r[key] is not None]
            return float(np.mean(values)) if values else float("nan")
        m1 = np.mean([r["m1"] for r in rs], axis=0)
        print(f"{name[:30]:30s} {len(rs):2d} {rs[0]['points']:6d}  {mean('m1u'):.3f} {np.std([r['m1u'] for r in rs]):.3f}     {mean('in_patch_old'):.3f}"
              f"          {mean('in_patch_new'):.3f}    {mean('honest'):.3f}  {mean('patches'):5.1f} | " + " ".join(f"{x:.2f}" for x in m1))
    print("Итого: М1u — средняя по цифрам доля точек цифры в её главном пятне; «в пятне» — то же отдельно по своим и новым картинкам")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "sum"
    if command == "codes":
        codes(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 3500, int(sys.argv[4]) if len(sys.argv) > 4 else 500)
    elif command == "maps":
        chosen = json.load(open(sys.argv[4], encoding="utf-8")) if len(sys.argv) > 4 else SERIES
        maps(int(sys.argv[2]), [int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "0,1,2").split(",")], chosen)
    elif command == "sheet":
        chosen = json.load(open(sys.argv[4], encoding="utf-8")) if len(sys.argv) > 4 else SERIES
        sheet(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0, chosen)
    else:
        summary()
