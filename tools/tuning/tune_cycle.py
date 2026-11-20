# -*- coding: utf-8 -*-
"""Один прогон подбора: обученный цикл с изменёнными настройками → сходимость классов на картах уровней.

    tune_cycle.py <номер> <файл с изменениями JSON> [обучающих цифр]

Сходимость — доля соседних точек карты одной цифры:
    честная — только пары точек с РАЗНЫХ обучающих картинок (собрались ли вместе разные картинки одной цифры);
    прежняя — все пары (завышена: окна одной картинки ложатся рядом).
Случайная карта даёт ≈ 0.10. Побочно: ответ читателя на 200 проверочных цифрах MNIST t10k.
Рабочие файлы цикла — отдельно от дашборда (data/tune/ladder). Итог дописывается в tune2_log.jsonl.
"""
import json
import os
import sys
import time

sys.path[:0] = [os.path.join(os.getcwd(), "client"), os.path.join(os.getcwd(), "system")]

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from damp_api import Damp  # noqa: E402


def memoize_heavy_steps() -> None:
    """Второй проход цикла (мерка для таблицы, funnel.py) повторял срабатывания детекторов на всех окнах каждого уровня и
    чистоту слияния — на 1000 цифрах до 20 минут. Срабатывания в системе не кешируются нарочно (выход огромный), поэтому
    здесь они запоминаются на время процесса: ключ — содержимое входа (коды окон, маска обучающих, детекторы, нормировка),
    ответ — тот же объект, что вернул первый расчёт. Счёт не меняется."""
    import hashlib

    from damp import settings
    from damp.cycle.step import CycleStep
    from damp.decode.merge_purity import MergePurity

    def digest(*arrays) -> str:  # noqa: ANN002
        hasher = hashlib.sha1()
        for array in arrays:
            array = np.ascontiguousarray(np.asarray(array))
            hasher.update(str(array.dtype).encode())
            hasher.update(str(array.shape).encode())
            hasher.update(array.tobytes())
        return hasher.hexdigest()

    fired, measured = {}, {}
    fire, measure = CycleStep.fire, MergePurity.measure

    def remembered_fire(detectors, codes, trained):  # noqa: ANN001, ANN202
        found = np.asarray([(d.center[0], d.center[1], d.radius, d.lambda_threshold, d.energy) for d in detectors.detectors],
                           dtype=np.float64)
        key = digest(codes.packed, trained, found) + f"|{codes.code_length}|{settings.LADDER_LEVEL_NORMALISATION}"
        if key not in fired:
            fired[key] = fire(detectors, codes, trained)
        return fired[key]

    def remembered_measure(self, packed, code_length, picture, labels, dictionary, *, name):  # noqa: ANN001, ANN202
        key = digest(packed, picture, labels, dictionary.packed) + f"|{code_length}"
        if key not in measured:
            measured[key] = measure(self, packed, code_length, picture, labels, dictionary, name=name)
        return measured[key]

    CycleStep.fire = staticmethod(remembered_fire)
    MergePurity.measure = remembered_measure


memoize_heavy_steps()


LEVELS: dict = {}  # числа окон по уровням — собирает наблюдатель прогона по ходу обучения


def collect_levels_during_training() -> None:
    """Мерка для таблицы по ходу обучения: наблюдатель прогона (тот же, что рисует вкладки Rerun) получает окна каждого
    уровня и его карту, как только они готовы. Второй проход цикла не нужен; счёт не меняется — наблюдатель только
    смотрит."""
    from damp.cycle import trained as trained_module
    from damp.cycle.ladder_trace import LadderObserver

    class Collector(LadderObserver):
        def part(self, number: int, name: str, values: dict) -> None:
            if name == "keyholes":
                LEVELS.setdefault(int(number), {}).update(values)

        def level(self, trace) -> None:  # noqa: ANN001
            LEVELS.setdefault(int(trace.number), {})["points"] = int(np.asarray(trace.map_cells).reshape(-1, 2).shape[0])

    train = trained_module.TrainedLadder.train.__func__

    def train_with_collector(cls, count, *, map_points, decay, observer=None):  # noqa: ANN001, ANN202
        return train(cls, count, map_points=map_points, decay=decay, observer=observer or Collector())

    trained_module.TrainedLadder.train = classmethod(train_with_collector)


def level_rows(pictures: int) -> list:
    """Строки таблицы по уровням из собранного по ходу обучения (те же ключи, что у прежней мерки funnel.py)."""
    rows = []
    for number in sorted(LEVELS):
        found = LEVELS[number]
        if "points" not in found:
            continue  # уровень не сложился: окна были, карты нет
        windows = int(np.asarray(found["keyhole_picture"]).shape[0])
        rows.append({
            "level": number,
            "elements_per_picture": round(int(np.asarray(found["element_picture"]).shape[0]) / pictures, 1),
            "windows_per_picture": round(windows / pictures, 1),
            "elements_per_window": round(int(np.asarray(found["entry_keyhole"]).shape[0]) / max(1, windows), 1),
            "window_sees": round(float(found["seen"]), 3),
            "window_radius": round(float(found.get("keyhole_radius", 0.0)), 2),
            "window_field": round(float(np.asarray(found["keyhole_extent"]).mean()), 2) if windows else 0.0,
            "points": int(found["points"]),
        })
    return rows


collect_levels_during_training()


def skip_primary_map() -> None:
    """TUNE_SKIP_PRIMARY=1: первичная карта («уровень 0» дашборда, 60 точек на цифру) не строится — в мерках уровней она
    не участвует, а на 2000 цифрах это самая большая карта (~120 тыс. точек): часы и ~9 ГБ на кластеры детекторов.
    Обученный цикл без неё работает так же (в дашборде нет только карточки уровня 0)."""
    if os.environ.get("TUNE_SKIP_PRIMARY") != "1":
        return
    from damp.cycle import trained as trained_module
    from damp.cycle.step import LadderLevelDegenerate

    def skipped(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise LadderLevelDegenerate("первичная карта пропущена стендом (TUNE_SKIP_PRIMARY=1)")

    trained_module.PrimaryLevel.build = staticmethod(skipped)


skip_primary_map()


def level0_shape(cells, angle, period: float, height: int, width: int) -> dict:  # noqa: ANN001
    """Цельность и гладкость карты уровня 0 (автор 2026-11-20: «гладкость первичной карты очень важна!»).

    smooth_deg — средняя разница углов соседних точек (по кругу периода; меньше — пинвилы глаже);
    inner_empty — пустые клетки внутри карты: и по строке, и по столбцу с обеих сторон есть точки (трещины, дыры);
    longest_gap — самая длинная полоса пустых клеток внутри карты по строке или столбцу (трещина, островок)."""
    cells = np.asarray(cells, dtype=np.int64)
    grid = np.full((height, width), np.nan)
    grid[cells[:, 0], cells[:, 1]] = np.asarray(angle, dtype=np.float64)
    diffs = []
    for a, b in ((grid[:, :-1], grid[:, 1:]), (grid[:-1, :], grid[1:, :])):
        held = ~np.isnan(a) & ~np.isnan(b)
        d = np.abs(a[held] - b[held]) % period
        diffs.append(np.minimum(d, period - d))
    filled = ~np.isnan(grid)
    inside_rows = np.zeros_like(filled)
    inside_cols = np.zeros_like(filled)
    longest = 0
    for y in range(height):
        xs = np.flatnonzero(filled[y])
        if xs.size >= 2:
            inside_rows[y, xs.min():xs.max() + 1] = True
            run = 0
            for v in ~filled[y, xs.min():xs.max() + 1]:
                run = run + 1 if v else 0
                longest = max(longest, run)
    for x in range(width):
        ys = np.flatnonzero(filled[:, x])
        if ys.size >= 2:
            inside_cols[ys.min():ys.max() + 1, x] = True
            run = 0
            for v in ~filled[ys.min():ys.max() + 1, x]:
                run = run + 1 if v else 0
                longest = max(longest, run)
    inner = int(np.count_nonzero(~filled & inside_rows & inside_cols))
    return {"smooth_deg": round(float(np.mean(np.concatenate(diffs))), 2), "inner_empty": inner, "longest_gap": int(longest)}


def level0_retinotopy(cells, y, x, picture) -> dict:  # noqa: ANN001
    """Ретинотопия карты уровня 0 (автор 2026-11-20: идеал — гладко связанные пинвилы, как на ретинотопических картах).

    Место штриха — относительно центра своей цифры (среднее мест её штрихов на карте). retino_step — средняя разница
    мест соседних точек карты, клеток (меньше — глаже); retino_spearman — корреляция Спирмена расстояния по карте и
    расстояния мест у 20 000 случайных пар (ближе к 1 — место разложено по карте непрерывно, без разрывов)."""
    cells = np.asarray(cells, dtype=np.int64)
    y, x, picture = (np.asarray(v, dtype=np.float64) for v in (y, x, picture))
    picture = picture.astype(np.int64)
    count = int(picture.max()) + 1
    sy, sx, n = np.zeros(count), np.zeros(count), np.zeros(count)
    ok = picture >= 0
    np.add.at(sy, picture[ok], y[ok]); np.add.at(sx, picture[ok], x[ok]); np.add.at(n, picture[ok], 1)
    p = np.maximum(picture, 0)
    dy, dx = y - sy[p] / np.maximum(n[p], 1), x - sx[p] / np.maximum(n[p], 1)
    index = np.full((int(cells[:, 0].max()) + 1, int(cells[:, 1].max()) + 1), -1)
    index[cells[:, 0], cells[:, 1]] = np.arange(cells.shape[0])
    steps = []
    for a, b in ((index[:, :-1], index[:, 1:]), (index[:-1, :], index[1:, :])):
        held = (a >= 0) & (b >= 0)
        steps.append(np.hypot(dy[a[held]] - dy[b[held]], dx[a[held]] - dx[b[held]]))
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, cells.shape[0], 20000), rng.integers(0, cells.shape[0], 20000)
    rank = lambda v: np.argsort(np.argsort(v))  # noqa: E731
    spearman = float(np.corrcoef(rank(np.hypot(*(cells[i] - cells[j]).T)), rank(np.hypot(dy[i] - dy[j], dx[i] - dx[j])))[0, 1])
    return {"retino_step": round(float(np.mean(np.concatenate(steps))), 3), "retino_spearman": round(spearman, 3)}


def main() -> None:
    number, changes_path = int(sys.argv[1]), sys.argv[2]
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    tests = int(sys.argv[4]) if len(sys.argv) > 4 else 200  # проверочных картинок MNIST t10k для читателя и памяти
    spec = json.load(open(changes_path, encoding="utf-8"))
    label, changes = spec["label"], dict(spec["changes"])
    here = os.path.dirname(os.path.abspath(__file__))

    damp = Damp(changes={**changes, "DASHBOARD_FILE_PREFIX": "data/tune/ladder"})
    started = time.perf_counter()
    ladder = damp.cycle.trained(count)  # предел карты — DASHBOARD_MAP_POINTS (его можно задать и в changes)
    maps = ladder.maps()
    trained_seconds = time.perf_counter() - started


    def convergence(level) -> dict:  # noqa: ANN001
        cells = np.asarray(level.cells, dtype=np.int64).reshape(-1, 2)
        labels = np.asarray(level.labels, dtype=np.int64)
        picture = np.asarray(level.point_picture, dtype=np.int64)
        grid = np.full((int(level.height), int(level.width)), -1, dtype=np.int64)
        grid[cells[:, 0], cells[:, 1]] = np.arange(cells.shape[0])
        pairs = same = other = other_same = 0
        for first, second in ((grid[:, :-1], grid[:, 1:]), (grid[:-1, :], grid[1:, :])):
            held = (first >= 0) & (second >= 0)
            a, b = first[held], second[held]
            alike, apart = labels[a] == labels[b], picture[a] != picture[b]
            pairs += int(a.shape[0])
            same += int(alike.sum())
            other += int(apart.sum())
            other_same += int((alike & apart).sum())
        # Собранность классов (цель автора 2026-11-20 — 10 кластеров): пятна цифры — связные по соседству (4 соседа)
        # куски её клеток на карте; «в главном пятне» — доля точек карты, лежащих в самом большом пятне своей цифры
        # (идеал 1.0), «пятен» — сколько кусков от 5 точек на всех цифрах (идеал 10).
        from scipy import ndimage

        digit_grid = np.full(grid.shape, -1, dtype=np.int64)
        digit_grid[cells[:, 0], cells[:, 1]] = labels
        in_main = patches = 0
        for digit in np.unique(labels):
            parts, count = ndimage.label(digit_grid == digit)
            sizes = np.bincount(parts.reshape(-1))[1:] if count else np.zeros(1, dtype=np.int64)
            in_main += int(sizes.max())
            patches += int(np.count_nonzero(sizes >= 5))
        # Одиночки (автор 2026-11-20: «ближний порядок должен растолкать одинокие точки по своим кластерам»): точки,
        # у которых ни один из четырёх соседей на карте не той же цифры (соседи без точки не в счёт; без соседей — не в счёт).
        lonely = counted = 0
        padded = np.pad(digit_grid, 1, constant_values=-1)
        for (y, x), digit in zip(cells, labels):
            around = (padded[y, x + 1], padded[y + 2, x + 1], padded[y + 1, x], padded[y + 1, x + 2])
            present = [value for value in around if value >= 0]
            if present:
                counted += 1
                lonely += int(all(value != digit for value in present))
        return {"points": int(labels.shape[0]), "plain": same / max(1, pairs), "honest": other_same / max(1, other),
                "other_pairs_share": other / max(1, pairs), "pictures": int(np.unique(picture).shape[0]),
                "in_main": in_main / max(1, labels.shape[0]), "patches": patches, "lonely": lonely / max(1, counted)}


    levels = [convergence(level) for level in maps]

    def detector_shape(level) -> dict:  # noqa: ANN001
        """Детекторы карты по слоям (§6.4, §7.1.10): сколько их и медиана радиуса r_d, клеток."""
        table = np.asarray(level.detectors, dtype=np.float64).reshape(-1, 5)
        layer = np.asarray(level.layer, dtype=np.int64)
        layers = sorted(set(layer.tolist()))
        return {"count": [int((layer == k).sum()) for k in layers],
                "radius": [round(float(np.median(table[layer == k, 2])), 2) if (layer == k).any() else 0.0 for k in layers]}

    shapes = [detector_shape(level) for level in maps]

    # Уровень 0 (автор 2026-11-20: «на картах раскладки показывай 0 уровень»): первичная карта штрихов, цвет — цифра
    # обучающей картинки штриха; мерки — те же, что у уровней цикла, в сводку серии не входят.
    zero = zero_found = None
    primary = ladder.primary_map()
    if primary is not None:
        from types import SimpleNamespace

        zero = SimpleNamespace(cells=np.asarray(primary.cells, dtype=np.int64), labels=np.asarray(primary.labels, dtype=np.int64),
                               point_picture=np.asarray(ladder._ladder.primary.point_picture, dtype=np.int64),
                               height=int(primary.height), width=int(primary.width), angle=np.asarray(primary.angle, dtype=np.float64))
        zero_found = convergence(zero)
        # Доля единиц места в коде каждой точки уровня 0 (автор 2026-11-20: биты места и угла почти поровну, но место
        # всегда чуть весомее). Код точки — [место | содержание]; длина места = длина круга × число копий (бит на круг).
        import tomllib

        with open(os.path.join("client", "config.toml"), "rb") as handle:
            config = tomllib.load(handle)

        def first(name: str):  # noqa: ANN202
            found = {**config.get("ladder", {}), **{k: v for section in config.values() if isinstance(section, dict)
                                                      for k, v in section.items()}}.get(name)
            value = changes.get(name, found)
            return int(value[0] if isinstance(value, (list, tuple)) else value)

        place_length = first("LADDER_BINDING_PLACE_CODE_LENGTH") * first("LADDER_BINDING_PLACE_BITS")
        table = ladder._ladder.primary.memory.layout.code_table
        bits = np.unpackbits(np.asarray(table.packed), axis=1, count=int(table.code_length)).astype(np.int64)
        place_ones = bits[:, :place_length].sum(axis=1)
        content_ones = bits[:, place_length:].sum(axis=1)
        share = place_ones / np.maximum(1, place_ones + content_ones)
        zero_found["place_share"] = {"place_length": int(place_length), "code_length": int(table.code_length),
                                     "place_ones": round(float(place_ones.mean()), 1), "content_ones": round(float(content_ones.mean()), 1),
                                     "mean": round(float(share.mean()), 4), "min": round(float(share.min()), 4),
                                     "max": round(float(share.max()), 4), "angle_heavier": int((place_ones <= content_ones).sum()),
                                     "points": int(share.shape[0])}
        print(f"  уровень 0: доля единиц места {zero_found['place_share']}")
        zero_found.update(level0_shape(zero.cells, zero.angle, float(primary.angle_period), int(zero.height), int(zero.width)))
        zero_found.update(level0_retinotopy(zero.cells, primary.y, primary.x, zero.point_picture))
        print(f"  уровень 0: гладкость {zero_found['smooth_deg']}°, пустых клеток внутри {zero_found['inner_empty']}, "
              f"самая длинная пустая полоса {zero_found['longest_gap']}")
        # Точки карты уровня 0 для разбора (островки, пинвилы): клетка, угол и место штриха, цифра и картинка.
        os.makedirs("data/maps", exist_ok=True)
        np.savez_compressed(f"data/maps/tune2-{number}-level0.npz", cells=zero.cells, angle=zero.angle,
                            y=np.asarray(primary.y, dtype=np.float64), x=np.asarray(primary.x, dtype=np.float64),
                            labels=zero.labels, picture=zero.point_picture, height=zero.height, width=zero.width,
                            period=float(primary.angle_period))

    def glow(trained, level) -> dict:  # noqa: ANN001
        """Мерки уровня без меток (статья): «свечение» карты (§7.1.8.5: «λ < 0.6 caused the space to glow, as all points
        were treated as similar… at λ = 0.45 a clear structure is visible») — какая доля точек карты в среднем похожа на
        точку не меньше λ (косинус кодов, сама точка не в счёт); Ē — средняя энергия точек (§5.8); детекторов (§6.4)."""
        table = trained.memory.layout.code_table
        bits = np.unpackbits(np.asarray(table.packed), axis=1, count=int(table.code_length)).astype(np.float32)
        ones = np.maximum(bits.sum(axis=1), 1.0)
        similarity = (bits @ bits.T) / np.sqrt(ones[:, None] * ones[None, :])
        np.fill_diagonal(similarity, -1.0)
        count = max(1, bits.shape[0] - 1)
        return {"glow45": float((similarity >= 0.45).sum(axis=1).mean() / count),
                "glow60": float((similarity >= 0.60).sum(axis=1).mean() / count),
                "energy": float(np.asarray(level.energies, dtype=np.float64).mean()),
                "detectors": int(np.asarray(level.detectors).shape[0])}

    unlabelled = [glow(trained, level) for trained, level in zip(ladder._ladder.levels, maps)]

    # Побочно: ответы на 200 проверочных цифрах (в обучении их нет).
    # Отбор без проверочных картинок (tests = 0): мерки — только по картам обучения.
    digits = damp.stimuli.digits(train=1, test=tests, train_views=()) if tests > 0 else None
    first, last = digits.groups["test"] if digits is not None else (0, 0)
    right = 0
    memory = {"agree": 0, "agree_right": 0, "mostly": 0, "mostly_right": 0, "split": 0, "split_right": 0, "unknown": 0}
    started = time.perf_counter()
    for i in range(first, last):
        picture = digits.picture(i)
        sight = ladder.see(np.asarray(picture.image, dtype=np.uint8), log=False)
        right += int(sight.digit == int(picture.digit))
        memory[sight.verdict] += 1
        if sight.verdict != "unknown":
            memory[sight.verdict + "_right"] += int(sight.memory_digit == int(picture.digit))
    accuracy = right / max(1, last - first)
    memory["right_share"] = round((memory["agree_right"] + memory["mostly_right"] + memory["split_right"]) / max(1, last - first), 3)
    memory["thresholds"] = [round(value, 3) for value in ladder.knowledge_thresholds]
    memory["see_ms"] = round(1000.0 * (time.perf_counter() - started) / max(1, last - first), 1)

    # Карты уровней: цвет — цифра.
    colours = [tuple(int(c) for c in Image.new("HSV", (1, 1), (int(d * 25.5), 220, 255)).convert("RGB").getpixel((0, 0))) for d in range(10)]
    side, pad, top = 300, 20, 74
    panels = ([(0, zero, zero_found)] if zero is not None else []) + [(n + 1, level, found) for n, (level, found) in enumerate(zip(maps, levels))]
    sheet = Image.new("RGB", (pad + len(panels) * (side + pad), top + side + 44), "black")
    draw = ImageDraw.Draw(sheet)
    try:
        big, small = ImageFont.truetype("arialbd.ttf", 19), ImageFont.truetype("arial.ttf", 14)
    except OSError:
        big = small = ImageFont.load_default()
    draw.text((pad, 10), f"подбор {number}: {label}", fill="white", font=big)

    def pinwheel(angle: float) -> tuple:
        """Цвет угла штриха — как пинвил в дашборде: оттенок = угол / период · 360°, hsl(·, 95 %, 60 %)."""
        import colorsys

        red, green, blue = colorsys.hls_to_rgb(((float(angle) / zero_period) % 1.0), 0.60, 0.95)
        return int(red * 255), int(green * 255), int(blue * 255)

    zero_period = float(primary.angle_period) if zero is not None else 180.0
    for n, (number_of_level, level, found) in enumerate(panels):
        left = pad + n * (side + pad)
        cells = np.asarray(level.cells, dtype=np.int64).reshape(-1, 2)
        labels = np.asarray(level.labels, dtype=np.int64)
        unit = side / max(int(level.height), int(level.width))
        draw.text((left, 38), f"Уровень {number_of_level}: {found['points']} точек", fill="white", font=small)
        draw.text((left, 55), f"честная {found['honest']:.3f} · прежняя {found['plain']:.3f}", fill=(200, 200, 200), font=small)
        # Уровень 0 — в цветах пинвила, по углу штриха (автор 2026-11-20); уровни цикла — по цифре.
        fills = [pinwheel(angle) for angle in level.angle] if number_of_level == 0 else [colours[int(digit)] for digit in labels]
        for (y, x), fill in zip(cells, fills):
            draw.rectangle((left + x * unit, top + y * unit, left + (x + 1) * unit - 1, top + (y + 1) * unit - 1), fill=fill)
    legend_left = pad
    if zero is not None:
        # Шкала угла под уровнем 0, цифры — под уровнями цикла.
        bar = side - 90
        for i in range(bar):
            draw.line((pad + 30 + i, top + side + 16, pad + 30 + i, top + side + 30), fill=pinwheel(zero_period * i / bar))
        draw.text((pad, top + side + 15), "0°", fill="white", font=small)
        draw.text((pad + 36 + bar, top + side + 15), f"{zero_period:.0f}°  угол", fill="white", font=small)
        legend_left = pad + side + pad
    for d in range(10):
        draw.rectangle((legend_left + d * 56, top + side + 16, legend_left + d * 56 + 14, top + side + 30), fill=colours[d])
        draw.text((legend_left + d * 56 + 20, top + side + 15), str(d), fill="white", font=small)
    os.makedirs("data/maps", exist_ok=True)
    image_path = f"data/maps/tune2-{number}.png"
    sheet.save(image_path)

    entry = {"run": number, "label": label, "changes": changes, "count": count, "levels": len(maps),
             "detector_shapes": shapes,
             "level0": None if zero_found is None else {key: (round(value, 4) if isinstance(value, float) else value)
                                                        for key, value in zero_found.items()},
             "honest": [round(found["honest"], 4) for found in levels], "plain": [round(found["plain"], 4) for found in levels],
             "points": [found["points"] for found in levels], "other_pairs": [round(found["other_pairs_share"], 3) for found in levels],
             "top_honest": round(levels[-1]["honest"], 4), "top_plain": round(levels[-1]["plain"], 4),
             "top_in_main": round(levels[-1]["in_main"], 3), "top_patches": int(levels[-1]["patches"]),
             "top_lonely": round(levels[-1]["lonely"], 3), "lonely": [round(found["lonely"], 3) for found in levels],
             "in_main": [round(found["in_main"], 3) for found in levels], "patches": [int(found["patches"]) for found in levels],
             "glow45": [round(found["glow45"], 4) for found in unlabelled], "glow60": [round(found["glow60"], 4) for found in unlabelled],
             "energy": [round(found["energy"], 4) for found in unlabelled], "detectors": [found["detectors"] for found in unlabelled],
             "accuracy200": round(accuracy, 3), "memory200": memory, "train_seconds": round(trained_seconds, 1), "image": image_path, "test_count": tests,
             "date": time.strftime("%Y-%m-%d %H:%M"),
             "merge": [{key: round(float(value), 4) for key, value in (level.merge or {}).items()} for level in maps]}
    # Мерка воронки по уровням (шаг 0 плана «естественная воронка»): счёт тот же, шаги — из кеша прогона.
    # Числа окон по уровням собраны по ходу обучения (collect_levels_during_training): второго прохода нет.
    entry["funnel"] = level_rows(count)
    # Таблица автора по уровням (формат от 2026-11-18): радиус окна, окно видит цифры, деталей в окне, точек, сходимость.
    names = ["штрихов", "деталей уровня 1", "деталей уровня 2", "деталей уровня 3", "деталей уровня 4", "деталей уровня 5"]
    # Наш «X»: тот же рисунок 28×28, что рисует снимок дашборда (dash_x_shot.py сохраняет его в x_image.b64).
    x_votes, x_path = {}, os.path.join(here, "x_image.b64")
    if os.path.exists(x_path):
        import base64

        pixels = np.frombuffer(base64.b64decode(open(x_path, encoding="ascii").read()), dtype=np.uint8).reshape(28, 28)
        sight = ladder.see(pixels, log=False)
        x_votes = {int(vote.number): (int(vote.digit), float(vote.share), int(vote.keyholes), int(vote.voted)) for vote in sight.votes}
        probabilities = [float(value) for value in sight.probabilities]
        entry["x"] = {
            "memory_digit": int(sight.memory_digit), "memory_share": round(float(sight.memory_share), 3), "verdict": sight.verdict,
            "memory_agreed": int(sight.memory_agreed), "reader_digit": int(sight.digit),
            "reader_probability": round(max(probabilities), 3) if probabilities else 0.0,
            "levels": {str(level): {"digit": digit, "share": round(share, 3), "keyholes": keyholes, "voted": voted}
                       for level, (digit, share, keyholes, voted) in x_votes.items()},
        }
    lines = ["| Уровень | Радиус окна, клеток | Окно видит цифры | Что в окне: детали предыдущего уровня | Точек карты | Сходимость | Голос уровня за наш «X» |",
             "|---|---|---|---|---|---|---|"]
    rows = [row for row in entry.get("funnel", []) if "failed" not in row]
    for index, row in enumerate(rows):
        honest = entry["honest"][index] if index < len(entry["honest"]) else float("nan")
        united = index > 0 and rows[index - 1].get("united")
        inside = (f"все {rows[index - 1]['windows_per_picture']} деталей уровня {index} (объединение)" if united
                  else f"{row['elements_per_window']} {names[min(index, len(names) - 1)]}")
        radius = "вся картинка" if united else row.get("window_radius", "—")
        vote = x_votes.get(int(row["level"]))
        said = "не дошёл" if vote is None else ("молчит" if vote[0] < 0 else f"«{vote[0]}» {100 * vote[1]:.0f} %")
        lines.append(f"| {row['level']} | {radius} | {100 * row['window_sees']:.0f} % | {inside} | {row['points']} | {honest:.3f} | {said} |")
    if "x" in entry:
        x = entry["x"]
        lines.append(f"| Итого | | | | | | память «{x['memory_digit']}» {100 * x['memory_share']:.0f} % ({x['verdict']}); читатель «{x['reader_digit']}» {100 * x['reader_probability']:.0f} % |")
    entry["table"] = lines
    with open(os.path.join(here, f"tune2-{number}.table.md"), "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    log_path = os.path.join(here, "tune2_log.jsonl")
    # Пары прогонов идут одновременно: журнал стенда переписывается под блокировкой (файл-замок) и подменяется
    # атомарно — иначе один прогон читает недописанную строку другого либо затирает его запись.
    lock = log_path + ".lock"
    handle = None
    for _ in range(1200):
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock) > 120:
                    os.remove(lock)  # замок брошен упавшим прогоном
            except OSError:
                pass
            time.sleep(0.1)
    try:
        known = []
        if os.path.exists(log_path):
            for line in open(log_path, encoding="utf-8"):
                try:
                    known.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        known = [item for item in known if item["run"] != number] + [entry]
        temporary = f"{log_path}.{os.getpid()}.tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            for item in sorted(known, key=lambda item: item["run"]):
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        os.replace(temporary, log_path)
    finally:
        if handle is not None:
            os.close(handle)
            os.remove(lock)
    print(f"ИТОГ {number}: {label}")
    print(f"  уровней {len(maps)}; честная сходимость по уровням {entry['honest']}; прежняя {entry['plain']}")
    print(f"  верх: пятен от 5 точек {entry['top_patches']} (идеал 10); в главном пятне своей цифры {entry['top_in_main']:.3f} (идеал 1.0)")
    print(f"  по уровням: пятен {entry['patches']}; в главном пятне {entry['in_main']}")
    print(f"  без меток: свечение λ0.45 {entry['glow45']}; λ0.60 {entry['glow60']}; Ē {entry['energy']}; детекторов {entry['detectors']}")
    print(f"  окно видит по уровням {[row['window_sees'] for row in entry['funnel']]}; деталей в окне {[row['elements_per_window'] for row in entry['funnel']]}")
    print(f"  точек {entry['points']}; доля пар с разных картинок {entry['other_pairs']}; ответ на 200 проверочных: читатель {accuracy:.3f}, память {memory}; обучение {trained_seconds:.0f} с")


# Раскладка малой карты запускает параллельные процессы; на Windows они заново читают этот файл —
# без этой защиты каждый начал бы весь прогон сначала.
if __name__ == "__main__":
    main()
