"""Суммы float32 пар «у нуля разности» на хосте: тот же порядок сложения, что у редукции исходного прохода (§5.6, §5.12).

Зачем. Обмен решает знак φ_c − φ_s (§5.6), а обе суммы исходный проход TorchPairEnergyEngine считает в float32
редукцией sum(dim=1) устройства. Почти у всех пар знак следует из оценки ошибки суммирования (PairSignKernel);
у оставшихся (единицы на шаг, разность порядка 10⁻⁴ от суммы) нужно само число редукции. Считать его
на устройстве — отдельная загрузка, запуск ядер и выгрузка на каждом таком шаге.

Порядок сложения редукции. Редукция CUDA суммирует строку ширины W блоком потоков bw × bh:
  1. элементы читаются четвёрками; четвёрку с номером i обрабатывает поток (i mod step), у потока четыре
     накопителя — по одному на место элемента в четвёрке, каждый складывает свои элементы по порядку;
     если начало строки не выровнено на 16 байт, первые элементы до границы и хвост после последней полной
     четвёрки добавляются к нулевому накопителю потоков первого варпа;
  2. накопители потока складываются по порядку: ((a0 + a1) + a2) + a3;
  3. если строка делится между варпами блока, потоки одной полосы складываются по варпам деревом со сдвигами
     bh/2 … 1;
  4. полосы складываются деревом со сдвигами 1, 2, 4, … (для bw > 32 перед этим — сдвигами bw/2 … 32).
Размеры блока и шаг определяются шириной строки и числом строк тензора (shape). Сложение с нулём точное,
поэтому для строки, где ненулевых слагаемых мало, достаточно пройти только их — в порядке их мест в строке.

Проверка. Порядок сложения — свойство установленной версии torch и устройства, поэтому для каждой формы
порции реплика сверяется с настоящей редукцией на случайных строках (calibrated), а первые расчёты пар
и затем каждый CHECK_EVERY-й — с расчётом PairSignKernel по участкам редукции. При расхождении реплика
отключается, суммы снова считает устройство.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from damp.article_refs import GPU_IMPLEMENTATION, OPTIM_SIM_MATRIX, POINT_EXCHANGE
from damp.layout.pair_energy_gpu import TorchPairEnergyEngine
from damp.layout.sparse_long_energy import SimilarityRows
from damp.logging import LOGGER

try:  # numba — необязательная зависимость: без неё суммы считает устройство.
    import numba
except Exception:  # noqa: BLE001
    numba = None

# Элементов в одном чтении редукции (четвёрка float32 — 16 байт).
VECTOR = 4


def _reduce_py(accumulators, block_width, warps):
    """Сумма строки по накопителям потоков [потоки, 4] (float32): шаги 2–4 порядка сложения редукции."""
    threads = block_width * warps
    value = np.empty(threads, dtype=np.float32)
    for thread in range(threads):
        total = accumulators[thread, 0] + accumulators[thread, 1]
        total = total + accumulators[thread, 2]
        value[thread] = total + accumulators[thread, 3]
    lanes = np.empty(block_width, dtype=np.float32)
    for lane in range(block_width):
        offset = warps // 2
        while offset > 0:
            for warp in range(offset):
                if warp + offset < warps:
                    value[warp * block_width + lane] = (
                        value[warp * block_width + lane] + value[(warp + offset) * block_width + lane]
                    )
            offset //= 2
        lanes[lane] = value[lane]
    offset = block_width // 2
    while offset >= 32:
        for lane in range(offset):
            lanes[lane] = lanes[lane] + lanes[lane + offset]
        offset //= 2
    width = min(block_width, 32)
    scratch = np.empty(width, dtype=np.float32)
    offset = 1
    while offset < width:
        for lane in range(width):
            if lane + offset < width:
                scratch[lane] = lanes[lane] + lanes[lane + offset]
            else:
                scratch[lane] = lanes[lane]
        for lane in range(width):
            lanes[lane] = scratch[lane]
        offset *= 2
    return lanes[0]


def _accumulate_py(accumulators, place, value, width, shift, step):
    """Добавить элемент строки (место place, значение value) в накопитель его потока — шаг 1 порядка сложения."""
    head = VECTOR - shift if shift > 0 else 0
    if place < head:
        accumulators[place + shift, 0] = accumulators[place + shift, 0] + value
        return
    position = place - head
    remaining = width - head
    tail_start = remaining - remaining % VECTOR
    if position >= tail_start:
        accumulators[position - tail_start, 0] = accumulators[position - tail_start, 0] + value
        return
    thread = (position // VECTOR) % step
    slot = position % VECTOR
    accumulators[thread, slot] = accumulators[thread, slot] + value


def _row_sum_py(places, values, width, shift, block_width, warps, step):
    """Сумма строки с ненулевыми элементами values на местах places (по возрастанию мест)."""
    accumulators = np.zeros((block_width * warps, VECTOR), dtype=np.float32)
    for index in range(places.size):
        _accumulate(accumulators, places[index], values[index], width, shift, step)
    return _reduce(accumulators, block_width, warps)


def _pair_sums_py(
    col, val, row_ptr, pos, point_a, point_b, ya, xa, yb, xb,
    neighbourhood, euclidean, local_radius, eps, reach, width, shift, block_width, warps, step,
):
    """(φ_c, φ_s) одной пары по строкам сходства её точек — числа редукции float32 исходного прохода.

    Элемент суммы на месте точки v: fl(s1·w1 + s2·w2) для φ_c и fl(s2·w1 + s1·w2) для φ_s, w — расстояние
    (§5.5.1) или вес 1/(d + ε) точек круга (§5.5.2). Место элемента — номер точки v либо номер её ячейки
    в окне пары (reach ≥ 0). Строки сходства упорядочены по номеру точки: обе строки сливаются по порядку.
    """
    start_a = row_ptr[point_a]
    stop_a = row_ptr[point_a + 1]
    if point_b >= 0:
        start_b = row_ptr[point_b]
        stop_b = row_ptr[point_b + 1]
    else:
        start_b = 0
        stop_b = 0
    capacity = (stop_a - start_a) + (stop_b - start_b)
    places = np.empty(capacity, dtype=np.int64)
    terms_c = np.empty(capacity, dtype=np.float32)
    terms_s = np.empty(capacity, dtype=np.float32)
    fya = np.float32(ya)
    fxa = np.float32(xa)
    fyb = np.float32(yb)
    fxb = np.float32(xb)
    one = np.float32(1.0)
    feps = np.float32(eps)
    radius_sq = np.float32(0.0)
    centre_y = np.float32(0.0)
    centre_x = np.float32(0.0)
    corner_y = 0
    corner_x = 0
    side = 2 * reach + 1
    if neighbourhood:
        half = np.float32(0.5)
        centre_y = (fya + fyb) * half
        centre_x = (fxa + fxb) * half
        dy = fya - fyb
        dx = fxa - fxb
        radius = np.float32(math.ceil(np.sqrt(dy * dy + dx * dx))) + one
        if radius < np.float32(local_radius):
            radius = np.float32(local_radius)
        radius_sq = radius * radius
        corner_y = (ya + yb) // 2 - reach
        corner_x = (xa + xb) // 2 - reach
    count = 0
    ia = start_a
    ib = start_b
    while ia < stop_a or ib < stop_b:
        va = col[ia] if ia < stop_a else 2147483647
        vb = col[ib] if ib < stop_b else 2147483647
        v = va if va < vb else vb
        sa = np.float32(0.0)
        sb = np.float32(0.0)
        if va == v:
            sa = val[ia]
            ia += 1
        if vb == v:
            sb = val[ib]
            ib += 1
        if v == point_a or v == point_b:
            continue
        vy = np.float32(pos[v, 0])
        vx = np.float32(pos[v, 1])
        if neighbourhood:
            oy = vy - centre_y
            ox = vx - centre_x
            if oy * oy + ox * ox > radius_sq:
                continue
        day = fya - vy
        dax = fxa - vx
        dby = fyb - vy
        dbx = fxb - vx
        qa = day * day + dax * dax
        qb = dby * dby + dbx * dbx
        if euclidean:
            qa = np.sqrt(qa)
            qb = np.sqrt(qb)
        if neighbourhood:
            qa = one / (qa + feps)
            qb = one / (qb + feps)
        # Строка a: s·w(своя ячейка a) в φ_c, s·w(ячейка b) в φ_s; строка b — наоборот.
        element_c = np.float32(0.0)
        element_s = np.float32(0.0)
        if sa != np.float32(0.0):
            element_c = sa * qa
            element_s = sa * qb
        if sb != np.float32(0.0):
            element_c = element_c + sb * qb
            element_s = element_s + sb * qa
        if reach >= 0:
            places[count] = (np.int64(pos[v, 0]) - corner_y) * side + (np.int64(pos[v, 1]) - corner_x)
        else:
            places[count] = v
        terms_c[count] = element_c
        terms_s[count] = element_s
        count += 1
    places = places[:count]
    terms_c = terms_c[:count]
    terms_s = terms_s[:count]
    if reach >= 0 and count > 1:
        # Места ячеек окна идут не по порядку номеров точек: накопители складывают элементы по порядку мест.
        order = np.argsort(places)
        places = places[order]
        terms_c = terms_c[order]
        terms_s = terms_s[order]
    sum_c = _row_sum(places, terms_c, width, shift, block_width, warps, step)
    sum_s = _row_sum(places, terms_s, width, shift, block_width, warps, step)
    return sum_c, sum_s


if numba is not None:
    _reduce = numba.njit(cache=True)(_reduce_py)
    _accumulate = numba.njit(cache=True, inline="always")(_accumulate_py)
    _row_sum = numba.njit(cache=True)(_row_sum_py)
    _pair_sums = numba.njit(cache=True)(_pair_sums_py)
else:  # pragma: no cover - без numba класс ReductionReplica не создаётся
    _reduce = _reduce_py
    _accumulate = _accumulate_py
    _row_sum = _row_sum_py
    _pair_sums = _pair_sums_py


class ReductionReplica:
    """Суммы φ_c, φ_s пар на хосте в порядке сложения редукции float32 исходного прохода.

    Строки сходства копируются на хост (пока помещаются в заданный объём памяти); сумма пары — проход
    по двум её строкам. Для каждой формы порции исходного прохода реплика сверяется с редукцией устройства.
    """

    # Наибольшее число потоков блока редукции и размер варпа устройства.
    MAX_THREADS = 512
    WARP = 32
    # Поток делит строку с другими варпами блока, если ему достаётся не меньше стольких значений.
    WARP_SPLIT_VALUES = 16
    MAX_VALUES_PER_THREAD = 256
    # Строки шире этого читаются четвёрками (иначе порядок сложения другой — реплика неприменима).
    VECTOR_MIN_WIDTH = 128
    # Пробных строк и ненулевых элементов в строке при сверке формы с редукцией устройства.
    CALIBRATION_ROWS = 12
    CALIBRATION_NONZERO = 384
    # Столько первых расчётов и затем каждый CHECK_EVERY-й сверяются с расчётом устройства.
    CHECK_FIRST = 16
    CHECK_EVERY = 128

    def __init__(self, engine: TorchPairEnergyEngine, rows: SimilarityRows, *, max_bytes: int) -> None:
        self._engine = engine
        self._rows = rows
        self._torch = engine.torch_mod
        self._device = engine.torch_device
        self._max_bytes = int(max_bytes)
        self._col: "np.ndarray | None" = None
        self._val: "np.ndarray | None" = None
        self._row_ptr: "np.ndarray | None" = None
        self._host_key: tuple[float, int] | None = None
        self._calibration: dict[tuple[int, Any], bool] = {}
        self._enabled = True
        self._calls = 0
        LOGGER.event(
            "layout.replica.init",
            section=GPU_IMPLEMENTATION,
            data={
                "points": engine.point_count,
                "max_mb": self._max_bytes / 1e6,
                "note": "суммы φ_c, φ_s пар у нуля разности (§5.6) считаются на хосте в порядке сложения редукции "
                "float32 исходного прохода (§5.12); порядок сверяется с редукцией устройства для каждой формы порции",
            },
        )

    @classmethod
    def create(cls, engine: TorchPairEnergyEngine, rows: SimilarityRows, *, max_bytes: int) -> "ReductionReplica | None":
        """Реплика редукции; None — numba недоступна или устройство не CUDA (порядок сложения у него свой)."""
        if numba is None or engine.torch_device.type != "cuda" or max_bytes <= 0:
            return None
        return cls(engine, rows, max_bytes=max_bytes)

    @property
    def rows(self) -> SimilarityRows:
        return self._rows

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------ форма блока редукции
    @classmethod
    def shape(cls, width: int, rows: int) -> tuple[int, int, int] | None:
        """(ширина блока, варпов на строку, шаг четвёрок) редукции тензора [rows, width]; None — не воспроизводится."""
        if width <= cls.VECTOR_MIN_WIDTH or rows <= 0:
            return None
        vectors = width // VECTOR

        def last_power(value: int) -> int:
            power = 1
            while power * 2 <= value:
                power *= 2
            return power

        vectors_power = last_power(vectors) if vectors < cls.MAX_THREADS else cls.MAX_THREADS
        rows_power = last_power(rows) if rows < cls.MAX_THREADS else cls.MAX_THREADS
        block_width = min(vectors_power, cls.WARP)
        block_height = min(rows_power, cls.MAX_THREADS // block_width)
        block_width = min(vectors_power, cls.MAX_THREADS // block_height)
        if block_width < cls.WARP:
            return None
        step = block_width
        per_thread = -(-width // step)
        if per_thread >= min(block_height * cls.WARP_SPLIT_VALUES, cls.MAX_VALUES_PER_THREAD):
            step *= block_height
            if -(-width // step) >= cls.MAX_VALUES_PER_THREAD:
                # Строка может делиться ещё и между блоками (редукция через общую память устройства).
                return None
            return block_width, block_height, step
        return block_width, 1, step

    def _calibrated(self, rows_full: int, width: int) -> bool:
        """Совпадает ли реплика с редукцией устройства на случайных строках тензора [rows_full, width]."""
        shape = self.shape(width, rows_full)
        # Порядок сложения зависит от числа строк только через размеры блока: одна сверка на форму блока.
        key = (int(width), shape)
        known = self._calibration.get(key)
        if known is not None:
            return known
        torch = self._torch
        matched = shape is not None
        if matched:
            block_width, warps, step = shape
            generator = np.random.default_rng(1_000_003 * rows_full + width)
            tested = sorted(
                {
                    row % rows_full
                    for row in (0, 1, 2, 3, 4, 5, 6, 7, rows_full // 2, rows_full // 3,
                                rows_full - 1, rows_full - 2, rows_full - 3, rows_full - 4)
                }
            )[: self.CALIBRATION_ROWS]
            nonzero = min(width, self.CALIBRATION_NONZERO)
            full = np.zeros((rows_full, width), dtype=np.float32)
            samples = []
            for row in tested:
                # Места покрывают начало и конец строки (выравнивание и хвост) и случайную середину.
                places = np.unique(
                    np.concatenate([np.arange(min(8, width)), np.arange(max(0, width - 8), width),
                                    generator.choice(width, size=nonzero, replace=False)])
                )
                values = np.exp(generator.random(places.size) * 4.0).astype(np.float32)
                full[row, places] = values
                samples.append((row, places, values))
            with torch.no_grad():
                expected = torch.as_tensor(full, device=self._device).sum(dim=1).cpu().numpy()
            for row, places, values in samples:
                got = _row_sum(places, values, width, (row * width) % VECTOR, block_width, warps, step)
                if got != expected[row]:
                    matched = False
                    break
        self._calibration[key] = matched
        LOGGER.event(
            "layout.replica.calibration",
            section=GPU_IMPLEMENTATION,
            data={
                "rows": int(rows_full),
                "width": int(width),
                "block": None if shape is None else shape,
                "identical": matched,
                "note": "порядок сложения редукции float32 устройства для порции [rows, width] (§5.12): identical — "
                "реплика на хосте даёт те же суммы побитово",
            },
        )
        return matched

    # ------------------------------------------------------------------ строки сходства на хосте
    def _host_rows(self) -> bool:
        """Строки сходства на хосте (копия CSR); False — не помещаются в отведённую память."""
        rows = self._rows
        key = (rows.lambda_threshold, rows.edges)
        if self._host_key == key:
            return self._col is not None
        self._host_key = key
        self._col = None
        self._val = None
        if rows.edges * SimilarityRows.BYTES_PER_EDGE > self._max_bytes:
            LOGGER.event(
                "layout.replica.rows.skipped",
                section=OPTIM_SIM_MATRIX,
                data={
                    "edges": rows.edges,
                    "memory_mb": rows.edges * SimilarityRows.BYTES_PER_EDGE / 1e6,
                    "max_mb": self._max_bytes / 1e6,
                    "note": "строки сходства не копируются на хост: суммы пар у нуля разности считает устройство",
                },
            )
            return False
        edges = rows.edges
        self._col = rows.col[:edges].cpu().numpy()
        self._val = rows.val[:edges].cpu().numpy()
        self._row_ptr = rows.row_ptr_np
        LOGGER.event(
            "layout.replica.rows",
            section=OPTIM_SIM_MATRIX,
            data={
                "edges": edges,
                "memory_mb": edges * SimilarityRows.BYTES_PER_EDGE / 1e6,
                "lambda_threshold": rows.lambda_threshold,
                "note": "строки сходства скопированы на хост (§5.9.5): суммы пар у нуля разности считаются без "
                "обращения к устройству",
            },
        )
        return True

    # ------------------------------------------------------------------ суммы пар
    def pair_sums(
        self,
        idx_a: "np.ndarray",
        idx_b: "np.ndarray",
        pos_a: "np.ndarray",
        pos_b: "np.ndarray",
        pending: "np.ndarray",
        positions: "np.ndarray",
        *,
        kind: str,
        euclidean: bool,
        geometry: tuple[int, int, int],
        local_radius: float | None = None,
        distance_eps: float = 0.0,
    ) -> tuple["np.ndarray", "np.ndarray"] | None:
        """(φ_c, φ_s) пар pending — числа редукции float32 исходного прохода; None — реплика неприменима.

        positions — ячейки всех точек на хосте (N × 2, int64); geometry — форма редукции исходного прохода
        для пар шага (PairSignKernel.reference_geometry): ширина строки, пар в порции, полуширина окна или −1.
        """
        if not self._enabled or pending.size == 0 or not self._host_rows():
            return None
        width, chunk, reach = geometry
        pair_count = int(idx_a.size)
        chunks = -(-pair_count // chunk)
        last_rows = pair_count - (chunks - 1) * chunk
        energy_c = np.empty(pending.size, dtype=np.float32)
        energy_s = np.empty(pending.size, dtype=np.float32)
        neighbourhood = kind == "neighbourhood"
        for position, pair in enumerate(pending.tolist()):
            chunk_index = pair // chunk
            row = pair - chunk_index * chunk
            rows_full = last_rows if chunk_index == chunks - 1 else chunk
            if not self._calibrated(rows_full, width):
                return None
            block_width, warps, step = self.shape(width, rows_full)
            energy_c[position], energy_s[position] = _pair_sums(
                self._col, self._val, self._row_ptr, positions,
                int(idx_a[pair]), int(idx_b[pair]),
                int(pos_a[pair, 0]), int(pos_a[pair, 1]), int(pos_b[pair, 0]), int(pos_b[pair, 1]),
                neighbourhood, bool(euclidean), 0.0 if local_radius is None else float(local_radius),
                float(distance_eps), int(reach), int(width), (row * width) % VECTOR, block_width, warps, step,
            )
        return energy_c, energy_s

    def check_due(self) -> bool:
        """Пора ли сверить суммы реплики с расчётом устройства: первые расчёты и затем периодически."""
        calls = self._calls
        self._calls = calls + 1
        return calls < self.CHECK_FIRST or calls % self.CHECK_EVERY == 0

    def checked(self, matched: bool, *, kind: str, pairs: int) -> None:
        """Итог сверки с расчётом устройства; при расхождении реплика отключается до конца работы."""
        LOGGER.event(
            "layout.replica.verify",
            section=POINT_EXCHANGE,
            data={
                "kind": kind,
                "pairs": pairs,
                "call": self._calls - 1,
                "bitwise_equal": matched,
                "note": "суммы φ_c, φ_s пар у нуля разности, посчитанные на хосте в порядке сложения редукции, сверены "
                "с расчётом устройства (§5.5.1, §5.5.2): обмены §5.6 совпадают при побитовом равенстве сумм",
            },
            force=True,
        )
        if not matched:
            self._enabled = False
