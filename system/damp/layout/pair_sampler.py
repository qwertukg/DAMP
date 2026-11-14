"""Выборка пар шага раскладки скомпилированным кодом (§5.3, §5.9.1, §5.9.3, §5.10): те же пары, что у numpy.

Правила выбора — из статьи: первая точка пары любая (в ближней фазе — по весам −ln Ê, §5.9.3), вторая —
ячейка в радиусе r (§5.3); пары ближней фазы со сходством ниже отсечки отбрасываются сразу (§5.9.1).
Выборка numpy в Layout делает это массивами: каждая попытка — десяток вызовов numpy, на шаг ближней фазы
приходится около сотни попыток по убывающему остатку пар, и время шага определяют накладные расходы
вызовов, а не арифметика.

Здесь те же циклы скомпилированы numba и берут случайные числа у того же генератора numpy теми же
запросами в том же порядке: integers(−r, r + 1, 2·остаток) на попытку, random(число пар) на выбор первых
точек по весам, integers(lo, hi + 1) на запасной выбор ячейки. numba реализует эти методы генератора
побитово так же, как numpy, поэтому пары и состояние генератора после выборки совпадают; Layout сверяет
это на первых шагах и затем периодически (при расхождении выборка возвращается к numpy).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from damp.article_refs import OPTIM_PAIR_SELECTION, PAIR_SELECTION
from damp.logging import LOGGER

try:  # numba — необязательная зависимость: без неё пары выбирает numpy.
    import numba
except Exception:  # noqa: BLE001
    numba = None

# Число попыток выбора второй ячейки в радиусе r, после которых берётся любая ячейка в границах (§5.3).
NEARBY_ATTEMPTS = 50


def _popcount_py(value):
    """Число единичных бит 64-битного слова (параллельный подсчёт целочисленными операциями)."""
    value = value - ((value >> np.uint64(1)) & np.uint64(0x5555555555555555))
    value = (value & np.uint64(0x3333333333333333)) + ((value >> np.uint64(2)) & np.uint64(0x3333333333333333))
    value = (value + (value >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (value * np.uint64(0x0101010101010101)) >> np.uint64(56)


def _nearby_py(rng, y1, x1, radius, lo_y, hi_y, lo_x, hi_x, height, width):
    """Вторые ячейки пар: до NEARBY_ATTEMPTS попыток в радиусе r (§5.3, §5.9.1), затем любая ячейка в границах.

    Границы (lo_y…hi_x) — плитка разбиения сетки (§5.10) или вся сетка; плитка из одной ячейки, совпадающей
    с первой точкой пары, расширяется до всей сетки. radius < 0 — радиус не задан.
    """
    count = y1.size
    y2 = np.empty(count, dtype=np.int64)
    x2 = np.empty(count, dtype=np.int64)
    one_cell = lo_y == hi_y and lo_x == hi_x
    left = np.arange(count)
    remaining = count
    if radius >= 0 and count > 0:
        radius_sq = radius * radius
        attempts = 0
        while attempts < NEARBY_ATTEMPTS and remaining > 0:
            # Смещения dy и dx оставшихся пар — один запрос к генератору, как у выборки numpy.
            offsets = rng.integers(-radius, radius + 1, 2 * remaining)
            attempts += 1
            kept = 0
            for position in range(remaining):
                pair = left[position]
                dy = offsets[position]
                dx = offsets[remaining + position]
                accepted = False
                if dy * dy + dx * dx <= radius_sq and (dy != 0 or dx != 0):
                    cell_y = y1[pair] + dy
                    cell_x = x1[pair] + dx
                    if one_cell and y1[pair] == lo_y and x1[pair] == lo_x:
                        accepted = 0 <= cell_y and cell_y < height and 0 <= cell_x and cell_x < width
                    else:
                        accepted = lo_y <= cell_y and cell_y <= hi_y and lo_x <= cell_x and cell_x <= hi_x
                    if accepted:
                        y2[pair] = cell_y
                        x2[pair] = cell_x
                if not accepted:
                    left[kept] = pair
                    kept += 1
            remaining = kept
    while remaining > 0:
        cells_y = np.empty(remaining, dtype=np.int64)
        cells_x = np.empty(remaining, dtype=np.int64)
        for position in range(remaining):
            pair = left[position]
            if one_cell and y1[pair] == lo_y and x1[pair] == lo_x:
                cells_y[position] = rng.integers(0, height)
            else:
                cells_y[position] = rng.integers(lo_y, hi_y + 1)
        for position in range(remaining):
            pair = left[position]
            if one_cell and y1[pair] == lo_y and x1[pair] == lo_x:
                cells_x[position] = rng.integers(0, width)
            else:
                cells_x[position] = rng.integers(lo_x, hi_x + 1)
        kept = 0
        for position in range(remaining):
            pair = left[position]
            if cells_y[position] == y1[pair] and cells_x[position] == x1[pair]:
                left[kept] = pair
                kept += 1
            else:
                y2[pair] = cells_y[position]
                x2[pair] = cells_x[position]
        remaining = kept
    return y2, x2


def _long_pairs_py(rng, pos, count, radius, height, width):
    """Пары дальней фазы (§5.3): первая точка — любая, вторая — ячейка в радиусе r; массив count × 4."""
    first = rng.integers(0, pos.shape[0], count)
    y1 = np.empty(count, dtype=np.int64)
    x1 = np.empty(count, dtype=np.int64)
    for index in range(count):
        y1[index] = pos[first[index], 0]
        x1[index] = pos[first[index], 1]
    y2, x2 = _nearby(rng, y1, x1, radius, 0, height - 1, 0, width - 1, height, width)
    pairs = np.empty((count, 4), dtype=np.int64)
    for index in range(count):
        pairs[index, 0] = y1[index]
        pairs[index, 1] = x1[index]
        pairs[index, 2] = y2[index]
        pairs[index, 3] = x2[index]
    return pairs


def _pair_similarity_py(packed, ones, cosine, point, second):
    """sim(a, b) пары точек без порога (§2.2.4) — те же операции float64, что у выборки numpy."""
    if point == second:
        return 1.0
    # Число общих бит — popcount(a & b) по 64-битным словам кодов (§2.2.4).
    common = 0
    for word in range(packed.shape[1]):
        common += _popcount(packed[point, word] & packed[second, word])
    ones_a = ones[point]
    ones_b = ones[second]
    if cosine:
        denominator = math.sqrt(ones_a * ones_b)
    else:
        denominator = ones_a + ones_b - common
    if denominator > 0.0 and ones_a > 0.0 and ones_b > 0.0:
        return common / denominator
    return 0.0


def _weighted_pick_py(cdf, guide, value):
    """Число элементов cdf, не превосходящих value, — то же, что cdf.searchsorted(value, side="right").

    guide[b] — число элементов cdf, не превосходящих b / (len(guide) − 1): поиск начинается с этой границы
    и почти всегда кончается за шаг-два; если округление номера отрезка увело границу выше ответа,
    выполняется обычный двоичный поиск.
    """
    size = cdf.size
    buckets = guide.size - 1
    bucket = int(value * buckets)
    if bucket >= buckets:
        bucket = buckets - 1
    position = guide[bucket]
    if position > 0 and cdf[position - 1] > value:
        low = 0
        high = position
        while low < high:
            middle = (low + high) // 2
            if cdf[middle] <= value:
                low = middle + 1
            else:
                high = middle
        return low
    while position < size and cdf[position] <= value:
        position += 1
    return position


def _tile_groups_py(pos, y_ends, x_ends, col_parts, tile_count):
    """Точки по плиткам сетки (§5.10): номера точек, сгруппированные по плиткам, и начала групп.

    Плитка точки — (первая полоса строк с концом не меньше y) · col_parts + (то же по столбцам), как
    у searchsorted(…, side="left"); внутри плитки точки идут по возрастанию номера.
    """
    count = pos.shape[0]
    tile = np.empty(count, dtype=np.int64)
    starts = np.zeros(tile_count + 1, dtype=np.int64)
    for index in range(count):
        row = 0
        while row < y_ends.size and y_ends[row] < pos[index, 0]:
            row += 1
        col = 0
        while col < x_ends.size and x_ends[col] < pos[index, 1]:
            col += 1
        tile[index] = row * col_parts + col
        starts[tile[index] + 1] += 1
    for index in range(tile_count):
        starts[index + 1] += starts[index]
    order = np.empty(count, dtype=np.int64)
    fill = starts[:-1].copy()
    for index in range(count):
        order[fill[tile[index]]] = index
        fill[tile[index]] += 1
    return order, starts


def _short_tile_pairs_py(
    rng, candidates, weights, total, weighted, pos, grid, packed, ones, cosine, cutoff, use_cutoff, count, radius,
    lo_y, hi_y, lo_x, hi_x, height, width,
):
    """Пары ближней фазы одной плитки (§5.3, §5.9.1, §5.9.3): первая точка по весам, вторая рядом, отсечка по сходству.

    weights — веса первых точек (−ln Ê, §5.9.3), total — их сумма: накопленные вероятности считаются так же,
    как у Generator.choice с весами (вероятности w / total, их накопленная сумма, деление на последнюю);
    weighted = False — первая точка равновероятна. Возвращает пары, число отброшенных по сходству и попыток.
    """
    pairs = np.empty((count, 4), dtype=np.int64)
    collected = 0
    attempts = 0
    rejected = 0
    max_attempts = max(count * 5, count)
    size = candidates.size
    cdf = np.empty(size if weighted else 0, dtype=np.float64)
    guide = np.empty(size + 1 if weighted else 0, dtype=np.int64)
    if weighted:
        running = 0.0
        for index in range(size):
            running += weights[index] / total
            cdf[index] = running
        last = cdf[size - 1]
        for index in range(size):
            cdf[index] /= last
        position = 0
        for bucket in range(size + 1):
            bound = bucket / size
            while position < size and cdf[position] <= bound:
                position += 1
            guide[bucket] = position
    while collected < count and attempts < max_attempts:
        batch = min(count - collected, max_attempts - attempts)
        if weighted:
            draws = rng.random(batch)
            picks = np.empty(batch, dtype=np.int64)
            for index in range(batch):
                picks[index] = _weighted_pick(cdf, guide, draws[index])
        else:
            picks = rng.integers(0, candidates.size, batch)
        first = np.empty(batch, dtype=np.int64)
        y1 = np.empty(batch, dtype=np.int64)
        x1 = np.empty(batch, dtype=np.int64)
        for index in range(batch):
            point = candidates[picks[index]]
            first[index] = point
            y1[index] = pos[point, 0]
            x1[index] = pos[point, 1]
        y2, x2 = _nearby(rng, y1, x1, radius, lo_y, hi_y, lo_x, hi_x, height, width)
        for index in range(batch):
            keep = True
            if use_cutoff:
                # Отсечка по сходству (§5.9.1): пара с занятой второй ячейкой и sim < отсечки отбрасывается.
                second = grid[y2[index], x2[index]]
                if second >= 0 and _pair_similarity(packed, ones, cosine, first[index], second) < cutoff:
                    keep = False
                    rejected += 1
            if keep:
                pairs[collected, 0] = y1[index]
                pairs[collected, 1] = x1[index]
                pairs[collected, 2] = y2[index]
                pairs[collected, 3] = x2[index]
                collected += 1
        attempts += batch
    return pairs[:collected], rejected, attempts


def _occupied_pairs_py(pairs, grid):
    """Пары шага для расчёта энергий: без пар из двух пустых ячеек, непустая точка всегда первая (§5.9.1).

    Строка результата — (точка a, точка b или −1, ячейка a (y, x), ячейка b (y, x)).
    """
    count = pairs.shape[0]
    result = np.empty((count, 6), dtype=np.int64)
    kept = 0
    for index in range(count):
        y1 = pairs[index, 0]
        x1 = pairs[index, 1]
        y2 = pairs[index, 2]
        x2 = pairs[index, 3]
        point_a = grid[y1, x1]
        point_b = grid[y2, x2]
        if point_a < 0 and point_b < 0:
            continue
        if point_a < 0:
            point_a, point_b = point_b, point_a
            y1, y2 = y2, y1
            x1, x2 = x2, x1
        result[kept, 0] = point_a
        result[kept, 1] = point_b
        result[kept, 2] = y1
        result[kept, 3] = x1
        result[kept, 4] = y2
        result[kept, 5] = x2
        kept += 1
    return result[:kept]


def _row_table_py(occupied, degree, code_class, table):
    """Таблица строк пар для PairSignKernel (int32) и наибольшая степень строки — см. PairSignKernel.pair_table.

    У пары точек с одинаковыми кодами (один класс code_class) суммы φ_c и φ_s равны точно: её строки
    помечаются как отсутствующие, ядро вернёт статус «равно» без расчёта.
    """
    count = occupied.shape[0]
    widest = 0
    for index in range(count):
        point_a = occupied[index, 0]
        point_b = occupied[index, 1]
        row = index * 6
        table[row] = point_a
        table[row + 1] = point_b
        table[row + 2] = occupied[index, 2]
        table[row + 3] = occupied[index, 3]
        table[row + 4] = occupied[index, 4]
        table[row + 5] = occupied[index, 5]
        row = (count + index) * 6
        table[row] = point_b
        table[row + 1] = point_a
        table[row + 2] = occupied[index, 4]
        table[row + 3] = occupied[index, 5]
        table[row + 4] = occupied[index, 2]
        table[row + 5] = occupied[index, 3]
        if point_b >= 0 and code_class[point_a] == code_class[point_b]:
            table[index * 6] = -1
            table[(count + index) * 6] = -1
            table[count * 12 + index] = 0
            continue
        degree_a = degree[point_a]
        degree_b = degree[point_b] if point_b >= 0 else 0
        table[count * 12 + index] = degree_a + degree_b
        if degree_a > widest:
            widest = degree_a
        if degree_b > widest:
            widest = degree_b
    return widest


if numba is not None:
    _popcount = numba.njit(cache=True, inline="always")(_popcount_py)
    _pair_similarity = numba.njit(cache=True, inline="always")(_pair_similarity_py)
    _nearby = numba.njit(cache=True)(_nearby_py)
    _long_pairs = numba.njit(cache=True)(_long_pairs_py)
    _weighted_pick = numba.njit(cache=True, inline="always")(_weighted_pick_py)
    _tile_groups = numba.njit(cache=True)(_tile_groups_py)
    _short_tile_pairs = numba.njit(cache=True)(_short_tile_pairs_py)
    _occupied_pairs = numba.njit(cache=True)(_occupied_pairs_py)
    _row_table = numba.njit(cache=True)(_row_table_py)
else:  # pragma: no cover - без numba класс CompiledPairSampler не создаётся
    _popcount = _popcount_py
    _pair_similarity = _pair_similarity_py
    _nearby = _nearby_py
    _long_pairs = _long_pairs_py
    _weighted_pick = _weighted_pick_py
    _tile_groups = _tile_groups_py
    _short_tile_pairs = _short_tile_pairs_py
    _occupied_pairs = _occupied_pairs_py
    _row_table = _row_table_py


class CompiledPairSampler:
    """Выборка пар шага (§5.3, §5.9.1, §5.9.3) скомпилированными циклами на общем с numpy генераторе.

    Методы возвращают то же, что одноимённые шаги выборки numpy в Layout, и оставляют генератор в том же
    состоянии. Массивы расстановки (позиции точек, сетка ячеек) передаются как есть, без копий.
    """

    def __init__(self, *, height: int, width: int) -> None:
        self._height = int(height)
        self._width = int(width)
        LOGGER.event(
            "layout.pair_sampling.compiled",
            section=PAIR_SELECTION,
            data={
                "numba": getattr(numba, "__version__", None),
                "grid": (self._height, self._width),
                "attempts": NEARBY_ATTEMPTS,
                "note": "выборка пар шага (§5.3, §5.9.1, §5.9.3) скомпилирована: те же запросы к генератору numpy "
                "в том же порядке, пары и состояние генератора совпадают с выборкой numpy",
            },
        )

    @classmethod
    def create(cls, *, height: int, width: int) -> "CompiledPairSampler | None":
        """Выборка на numba; None — numba недоступна (пары выбирает numpy)."""
        if numba is None:
            LOGGER.event(
                "layout.pair_sampling.compiled.unavailable",
                section=PAIR_SELECTION,
                data={"note": "numba не установлена: пары шага выбирает numpy (§5.3)"},
            )
            return None
        return cls(height=height, width=width)

    @staticmethod
    def clone_generator(rng: "np.random.Generator") -> "np.random.Generator":
        """Независимый генератор в том же состоянии — для сверки с выборкой numpy."""
        clone = np.random.Generator(type(rng.bit_generator)())
        clone.bit_generator.state = rng.bit_generator.state
        return clone

    def nearby(
        self,
        rng: "np.random.Generator",
        y1: "np.ndarray",
        x1: "np.ndarray",
        radius: int | None,
        bounds: tuple[int, int, int, int] | None,
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """Вторые ячейки пар в радиусе r от первых (§5.3) в границах плитки (§5.10) или всей сетки."""
        lo_y, hi_y, lo_x, hi_x = (0, self._height - 1, 0, self._width - 1) if bounds is None else bounds
        return _nearby(
            rng, np.ascontiguousarray(y1, dtype=np.int64), np.ascontiguousarray(x1, dtype=np.int64),
            -1 if radius is None else int(radius), int(lo_y), int(hi_y), int(lo_x), int(hi_x),
            self._height, self._width,
        )

    def long_pairs(
        self, rng: "np.random.Generator", pos: "np.ndarray", count: int, radius: int | None
    ) -> "np.ndarray":
        """Пары дальней фазы (§5.3): массив count × 4 (y1, x1, y2, x2)."""
        return _long_pairs(rng, pos, int(count), -1 if radius is None else int(radius), self._height, self._width)

    @staticmethod
    def tile_groups(
        pos: "np.ndarray", y_ends: "np.ndarray", x_ends: "np.ndarray", col_parts: int, tile_count: int
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """Точки по плиткам сетки (§5.10): (номера точек по плиткам, начала групп) — точки плитки t лежат
        в order[starts[t]:starts[t + 1]] по возрастанию номера."""
        return _tile_groups(pos, y_ends, x_ends, int(col_parts), int(tile_count))

    def short_tile_pairs(
        self,
        rng: "np.random.Generator",
        candidates: "np.ndarray",
        weights: "np.ndarray | None",
        total: float,
        *,
        pos: "np.ndarray",
        grid: "np.ndarray",
        packed_codes: "np.ndarray",
        ones: "np.ndarray",
        cosine: bool,
        similarity_cutoff: float | None,
        count: int,
        radius: int | None,
        bounds: tuple[int, int, int, int] | None,
    ) -> tuple["np.ndarray", int, int]:
        """Пары ближней фазы одной плитки (§5.9.1, §5.9.3): (пары, отброшено по сходству, попыток)."""
        lo_y, hi_y, lo_x, hi_x = (0, self._height - 1, 0, self._width - 1) if bounds is None else bounds
        weighted = weights is not None
        pairs, rejected, attempts = _short_tile_pairs(
            rng, np.ascontiguousarray(candidates, dtype=np.int64),
            np.ascontiguousarray(weights, dtype=np.float64) if weighted else np.zeros(0, dtype=np.float64),
            float(total), weighted, pos, grid, packed_codes,
            ones, bool(cosine), 0.0 if similarity_cutoff is None else float(similarity_cutoff),
            similarity_cutoff is not None, int(count), -1 if radius is None else int(radius),
            int(lo_y), int(hi_y), int(lo_x), int(hi_x), self._height, self._width,
        )
        return pairs, int(rejected), int(attempts)

    @staticmethod
    def occupied_pairs(pairs: "np.ndarray", grid: "np.ndarray") -> "np.ndarray":
        """Пары для расчёта энергий (§5.9.1): строки (a, b или −1, ячейка a, ячейка b), пустые пары отброшены."""
        return _occupied_pairs(pairs, grid)

    @staticmethod
    def row_table(
        occupied: "np.ndarray", degree: "np.ndarray", code_class: "np.ndarray", table: "np.ndarray | None" = None
    ) -> tuple["np.ndarray", int]:
        """Таблица строк пар для ядра PairSignKernel и наибольшая степень строки сходства.

        table — готовый массив int32 длины 13·пары (например, закреплённая память для загрузки на устройство);
        без него таблица создаётся здесь.
        """
        if table is None:
            table = np.empty(occupied.shape[0] * 13, dtype=np.int32)
        widest = _row_table(occupied, degree, code_class, table)
        return table, int(widest)

    def log_mismatch(self, *, kind: str, step: int | None) -> None:
        LOGGER.event(
            "layout.pair_sampling.compiled.mismatch",
            section=OPTIM_PAIR_SELECTION,
            data={
                "kind": kind,
                "step": -1 if step is None else step,
                "note": "скомпилированная выборка пар разошлась с выборкой numpy: дальше пары выбирает numpy (§5.3)",
            },
            force=True,
        )
