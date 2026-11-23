"""Эмбеддинги наборов стимулов из сработавших детекторов (§6.5, §6.6, §7.1.12).

HierarchyDetection.active_sets отвечает, какие детекторы и с каким уровнем сработали на каждый набор. Здесь
из них собирается выходной код:

    - у детектора один случайный бит выходного кода (§6.3);
    - в код идут самые сильные детекторы набора, не больше σ (динамический порог μ_c, §7.1.12);
    - код набора — объединение их бит (§6.5; при одном бите на детектор colour merge с бюджетом σ и есть
      выбор σ сильнейших);
    - длина кода — наименьшая, при которой средняя плотность кодов не выше заданной (§6.6, шаг 3; §2).

Скорость (вне статьи; ответ тот же, tests/test_detail_codes_fast.py). На 5 000 цифр сборка шла 120 с на уровень:
90 % — подбор длины кода, где на каждом шаге двоичного поиска сортировались все пары «набор — бит». Теперь единицы
считаются по битовой карте, без сортировки, а σ сильнейших детекторов набора выбирает скомпилированный цикл numba
(без numba и на парах не по порядку — прежняя плотная матрица).
"""
from __future__ import annotations

import numpy as np

from damp.article_refs import MORPHOLOGICAL_EMBEDDINGS, NEURAL_ANALOGY
from damp.logging import LOGGER

try:  # numba — необязательная зависимость: без неё σ сильнейших выбирает numpy по плотной матрице.
    import numba
except ImportError:  # pragma: no cover
    numba = None


def _strongest_py(starts, columns, levels, saturation, kept):  # noqa: ANN001, ANN201
    """Отбор σ самых сильных детекторов в каждом наборе — то же, что по плотной матрице «наборы × детекторы».

    starts — границы наборов в парах, идущих по наборам; kept — булева маска пар, остающихся после отбора (пишется
    на месте). В наборе, где детекторов не больше σ, остаются все. В остальных — уровни выше σ-го по силе и столько
    равных ему, сколько не хватает до σ, по возрастанию номера детектора. Возвращает False, если в каком-то наборе
    номера детекторов идут не строго по возрастанию (пары не по порядку или повторы): тогда отбор делает прямой путь.
    """
    for row in range(starts.shape[0] - 1):
        first, last = starts[row], starts[row + 1]
        count = last - first
        if count == 0:
            continue
        for index in range(first + 1, last):
            if columns[index] <= columns[index - 1]:
                return False
        if count <= saturation:
            for index in range(first, last):
                kept[index] = True
            continue
        border = np.partition(levels[first:last], count - saturation)[count - saturation]
        missing = saturation
        for index in range(first, last):
            if levels[index] > border:
                missing -= 1
        for index in range(first, last):
            if levels[index] > border:
                kept[index] = True
            elif levels[index] == border and missing > 0:
                kept[index] = True
                missing -= 1
    return True


_strongest_compiled = None if numba is None else numba.njit(cache=True, nogil=True)(_strongest_py)


class DetectorEmbeddings:
    """Коды наборов по сработавшим детекторам: выбор σ сильнейших, длина под плотность, один бит на детектор."""

    # Длины перебираются кратными этому числу: код хранится упакованным по 8 бит в байт.
    LENGTH_STEP = 8
    # Строк за одну сборку бит: матрица «наборы × биты» целиком не создаётся.
    CHUNK_ROWS = 32768
    # Ячеек плотной матрицы «наборы × детекторы» за один проход отбора самых сильных детекторов.
    SELECT_CELLS = 8_000_000
    # Ячеек битовой карты «наборы × биты» за один проход счёта плотности кодов.
    DENSITY_CELLS = 64_000_000

    def __init__(
        self,
        *,
        detectors: int,
        saturation_limit: int,
        target_density: float,
        min_length: int,
        max_length: int,
        seed: int,
        name: str = "sets",
        saturation_share: float = 0.0,
        colour: "np.ndarray | None" = None,
        topography: "np.ndarray | None" = None,
        ancestors: "np.ndarray | None" = None,
        layers: "np.ndarray | None" = None,
    ) -> None:
        if detectors <= 0:
            raise ValueError("detectors must be positive")
        if not 0.0 <= saturation_share <= 1.0:
            raise ValueError("saturation_share must be in [0, 1] (0 — σ задан числом)")
        if saturation_limit < 0:
            raise ValueError("saturation_limit must be non-negative (0 — без ограничения)")
        if not 0.0 < target_density < 1.0:
            raise ValueError("target_density must be in (0, 1)")
        if min_length <= 0 or max_length < min_length:
            raise ValueError("code length bounds must satisfy 0 < min_length <= max_length")
        self._detectors = int(detectors)
        self._saturation = int(saturation_limit)
        # σ долей от сработавших в наборе (вне статьи; разметка автора 2026-11-20 «середина плато»): берётся
        # round(share · сработавших), не меньше 1 и не больше saturation_limit (если он задан); 0 — σ числом.
        self._share = float(saturation_share)
        self._target = float(target_density)
        self._min_steps = -(-int(min_length) // self.LENGTH_STEP)
        self._max_steps = max(self._min_steps, int(max_length) // self.LENGTH_STEP)
        self._seed = int(seed)
        self._name = str(name)
        # Цвет детектора (§6.5: λ_d — «цвет» кода детектора): номер слоя иерархии, 0 — самый низкий порог λ_d, самые
        # широкие области («длинные волны», §3.3.1). Задан — σ в наборе отбираются сначала по цвету, внутри цвета по
        # силе: colour merge, отбрасывающий коротковолновые биты (§3.3.2). None — только по силе.
        self._colour = None if colour is None else np.asarray(colour, dtype=np.int64)
        # Топографические биты (план v4.11, вне статьи по форме: §6.3 — бит детектора случайный): ключи ячеек карты
        # (детекторы × сетки), бит = ключ по модулю длины кода — соседние детекторы одной ячейки делят бит. None — как было.
        self._topography = None if topography is None else np.asarray(topography, dtype=np.uint64)
        if self._topography is not None and self._topography.shape[0] != self._detectors:
            raise ValueError("topography: по строке ключей на детектор")
        # Предки детекторов (план v8.2): к каждому из σ выбранных добавляется его широкий предок (детекторы × слоёв вверх,
        # −1 — нет) со своим случайным битом. None — как было.
        self._ancestors = None if ancestors is None else np.asarray(ancestors, dtype=np.int64).reshape(self._detectors, -1)
        # Квота σ по слоям (план v8.6): слой каждого детектора; задан — в наборе сначала не больше ⌈σ / слоёв⌉ сильнейших
        # от каждого слоя, остаток σ — по силе из остальных. Широкие слои всегда в коде, узкие не вытесняются. None — выкл.
        self._layers = None if layers is None else np.asarray(layers, dtype=np.int64)

    def with_ancestors(self, rows: "np.ndarray", columns: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
        """Пары (набор, детектор) выбранных σ и их предков без повторов; без предков — как есть."""
        if self._ancestors is None or rows.size == 0:
            return rows, columns
        extra = self._ancestors[columns]
        rows_all = np.concatenate([rows, np.repeat(rows, extra.shape[1])])
        columns_all = np.concatenate([columns, extra.reshape(-1)])
        keep = columns_all >= 0
        keys = np.unique(rows_all[keep] * np.int64(self._detectors) + columns_all[keep])
        return keys // self._detectors, keys % self._detectors

    def strongest(
        self, rows: "np.ndarray", columns: "np.ndarray", levels: "np.ndarray"
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """Пары (набор, детектор), оставшиеся после порога μ_c: σ самых сильных детекторов набора (§7.1.12)."""
        rows = np.asarray(rows, dtype=np.int64)
        columns = np.asarray(columns, dtype=np.int64)
        if self._layers is not None and rows.size and self._saturation > 0:
            return self._strongest_quota(rows, columns, np.asarray(levels, dtype=np.float64))
        if self._colour is not None and rows.size and (self._saturation > 0 or self._share > 0.0):
            return self._strongest_colour(rows, columns, np.asarray(levels, dtype=np.float64))
        if self._share > 0.0 and rows.size:
            return self._strongest_share(rows, columns, np.asarray(levels, dtype=np.float64))
        if self._saturation == 0 or rows.size == 0:
            return rows, columns
        levels = np.asarray(levels)
        if np.any(np.diff(rows) < 0):
            order = np.argsort(rows, kind="stable")
            rows, columns, levels = rows[order], columns[order], levels[order]
        if _strongest_compiled is not None:
            # Быстрый путь: скомпилированный цикл по наборам, без плотной матрицы. Ответ тот же; пары, где номера
            # детекторов набора идут не по возрастанию, он не берёт — их разбирает прямой путь ниже.
            starts = np.searchsorted(rows, np.arange(int(rows[-1]) + 2))
            kept = np.zeros(rows.shape[0], dtype=np.bool_)
            if _strongest_compiled(starts, np.ascontiguousarray(columns), np.ascontiguousarray(levels), self._saturation, kept):
                return rows[kept], columns[kept]
        return self._strongest_dense(rows, columns, levels)

    def _strongest_quota(
        self, rows: "np.ndarray", columns: "np.ndarray", levels: "np.ndarray"
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """План v8.6: σ сильнейших с квотой по слоям — в каждом наборе сначала не больше ⌈σ / слоёв⌉ сильнейших каждого
        слоя (по силе), затем остаток σ — по силе из остальных; при равных — меньший номер детектора."""
        layer = self._layers[columns]
        count = int(self._layers.max()) + 1 if self._layers.size else 1
        cap = -(-self._saturation // max(1, count))
        order = np.lexsort((columns, -levels, layer, rows))
        group = rows[order] * np.int64(count) + layer[order]
        starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
        rank = np.arange(order.size, dtype=np.int64) - np.repeat(starts, np.diff(np.r_[starts, order.size]))
        first = np.zeros(rows.size, dtype=bool)
        first[order] = rank < cap
        order2 = np.lexsort((columns, -levels, ~first, rows))
        sorted_rows = rows[order2]
        sets = int(sorted_rows[-1]) + 1
        bounds = np.searchsorted(sorted_rows, np.arange(sets + 1))
        within = np.arange(sorted_rows.size, dtype=np.int64) - bounds[sorted_rows]
        kept = np.zeros(rows.size, dtype=bool)
        kept[order2[within < self._saturation]] = True
        return rows[kept], columns[kept]

    def _strongest_colour(
        self, rows: "np.ndarray", columns: "np.ndarray", levels: "np.ndarray"
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """Colour merge (§6.5, §3.3.2): в каждом наборе — σ детекторов, сначала самых широких слоёв (меньший цвет), внутри
        слоя — сильнейшие; при равных — меньший номер детектора. σ — saturation_limit или доля share от сработавших."""
        colour = self._colour[columns]
        order = np.lexsort((columns, -levels, colour, rows))
        sorted_rows = rows[order]
        sets = int(sorted_rows[-1]) + 1
        starts = np.searchsorted(sorted_rows, np.arange(sets + 1))
        counts = np.diff(starts)
        if self._share > 0.0:
            take = np.maximum(1, np.rint(self._share * counts)).astype(np.int64)
            if self._saturation > 0:
                take = np.minimum(take, self._saturation)
        else:
            take = np.full(sets, self._saturation, dtype=np.int64)
        rank = np.arange(sorted_rows.size, dtype=np.int64) - starts[sorted_rows]
        kept = np.zeros(rows.size, dtype=bool)
        kept[order[rank < take[sorted_rows]]] = True
        return rows[kept], columns[kept]

    def _strongest_share(
        self, rows: "np.ndarray", columns: "np.ndarray", levels: "np.ndarray"
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """σ долей: в каждом наборе — round(share · сработавших) сильнейших (не меньше 1; не больше saturation_limit,
        если он задан). Разметка автора 2026-11-20: σ — середина «плато» уровней; по всем окнам она около половины
        сработавших (§7.1.12, пример статьи — 30 из 80). При равных уровнях раньше идёт меньший номер детектора."""
        order = np.lexsort((columns, -levels, rows))
        sorted_rows = rows[order]
        sets = int(sorted_rows[-1]) + 1
        starts = np.searchsorted(sorted_rows, np.arange(sets + 1))
        counts = np.diff(starts)
        take = np.maximum(1, np.rint(self._share * counts)).astype(np.int64)
        if self._saturation > 0:
            take = np.minimum(take, self._saturation)
        rank = np.arange(sorted_rows.size, dtype=np.int64) - starts[sorted_rows]
        kept = np.zeros(rows.size, dtype=bool)
        kept[order[rank < take[sorted_rows]]] = True
        return rows[kept], columns[kept]

    def _strongest_dense(
        self, rows: "np.ndarray", columns: "np.ndarray", levels: "np.ndarray"
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """Прямой путь отбора σ сильнейших: плотная матрица «наборы × детекторы». Пары идут по наборам."""
        levels = np.asarray(levels, dtype=np.float64)
        # Наборы, где детекторов не больше σ, остаются целиком. В остальных — σ самых сильных; при равной силе
        # раньше идёт детектор с меньшим номером. Вместо сортировки всех пар — порог внутри набора: в плотной
        # матрице «наборы × детекторы» находится σ-й по силе уровень, проходят уровни выше него и столько равных
        # ему (по возрастанию номера детектора), сколько не хватает до σ.
        sets = int(rows[-1]) + 1
        starts = np.searchsorted(rows, np.arange(sets + 1))
        step = max(1, self.SELECT_CELLS // self._detectors)
        kept_rows: list["np.ndarray"] = []
        kept_columns: list["np.ndarray"] = []
        for first in range(0, sets, step):
            last = min(sets, first + step)
            entries = slice(int(starts[first]), int(starts[last]))
            if entries.stop == entries.start:
                continue
            strength = np.full((last - first, self._detectors), -np.inf, dtype=np.float64)
            strength[rows[entries] - first, columns[entries]] = levels[entries]
            if self._saturation >= self._detectors:
                keep = np.isfinite(strength)
            else:
                border = np.partition(strength, self._detectors - self._saturation, axis=1)[
                    :, self._detectors - self._saturation
                ][:, None]
                above = strength > border
                equal = (strength == border) & np.isfinite(strength)
                missing = self._saturation - above.sum(axis=1, keepdims=True)
                keep = above | (equal & (np.cumsum(equal, axis=1) <= missing))
            found_rows, found_columns = np.nonzero(keep)
            kept_rows.append(found_rows.astype(np.int64) + first)
            kept_columns.append(found_columns.astype(np.int64))
        if not kept_rows:
            return rows[:0], columns[:0]
        return np.concatenate(kept_rows), np.concatenate(kept_columns)

    def detector_bits(self, length: int) -> "np.ndarray":
        """Бит каждого детектора в коде длины length: случайная перестановка, пока детекторов не больше длины.
        Топографические биты — по биту на сетку (детекторы × сетки): ключ ячейки по модулю длины."""
        if self._topography is not None:
            return (self._topography % np.uint64(int(length))).astype(np.int64)
        rng = np.random.default_rng(self._seed)
        parts = []
        left = self._detectors
        while left > 0:
            parts.append(rng.permutation(int(length))[:left])
            left -= int(length)
        return np.concatenate(parts).astype(np.int64)

    def _density(self, rows: "np.ndarray", columns: "np.ndarray", count: int, length: int) -> float:
        """Средняя плотность кодов наборов 0…count−1 при длине кода length: доля единичных бит.

        Единицы считаются по битовой карте «наборы × биты», кусками по DENSITY_CELLS ячеек: пара (набор, бит) ставит
        единицу, два детектора набора с общим битом дают одну. То же число, что у прямого счёта (_density_sorted),
        без сортировки всех пар.
        """
        length = int(length)
        rows, place = self._pairs(np.asarray(rows, dtype=np.int64), self.detector_bits(length)[columns])
        if rows.size and np.any(np.diff(rows) < 0):
            order = np.argsort(rows, kind="stable")
            rows, place = rows[order], place[order]
        sets = int(rows[-1]) + 1 if rows.size else 0
        step = max(1, self.DENSITY_CELLS // length)
        starts = np.searchsorted(rows, np.arange(0, sets + step, step))
        ones = 0
        for chunk, first in enumerate(range(0, sets, step)):
            entries = slice(int(starts[chunk]), int(starts[chunk + 1]))
            if entries.stop == entries.start:
                continue
            seen = np.zeros(min(step, sets - first) * length, dtype=np.bool_)
            seen[(rows[entries] - first) * length + place[entries]] = True
            ones += int(np.count_nonzero(seen))
        return float(ones) / (max(1, int(count)) * length)

    def _density_sorted(self, rows: "np.ndarray", columns: "np.ndarray", count: int, length: int) -> float:
        """Прямой счёт плотности: различные пары (набор, бит) сортировкой. Мерка для быстрого счёта _density."""
        rows, place = self._pairs(np.asarray(rows, dtype=np.int64), self.detector_bits(length)[columns])
        keys = np.unique(rows * int(length) + place)
        return float(keys.shape[0]) / (max(1, int(count)) * int(length))

    @staticmethod
    def _pairs(rows: "np.ndarray", place: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
        """Пары (набор, бит): один бит на детектор — как есть; топографические — по паре на каждую сетку."""
        if place.ndim == 1:
            return rows, place
        return np.repeat(rows, place.shape[1]), place.reshape(-1)

    def fit_length(self, rows: "np.ndarray", columns: "np.ndarray", count: int) -> int:
        """Наименьшая длина кода, при которой средняя плотность кодов наборов 0…count−1 не выше заданной."""
        low, high = self._min_steps, self._max_steps
        if self._density(rows, columns, count, low * self.LENGTH_STEP) <= self._target:
            high = low
        elif self._density(rows, columns, count, high * self.LENGTH_STEP) > self._target:
            low = high
        else:
            # Плотность убывает с длиной кода: ищется наименьшая длина с плотностью не выше заданной.
            while high - low > 1:
                middle = (low + high) // 2
                if self._density(rows, columns, count, middle * self.LENGTH_STEP) > self._target:
                    low = middle
                else:
                    high = middle
        return high * self.LENGTH_STEP

    def encode(
        self,
        rows: "np.ndarray",
        columns: "np.ndarray",
        levels: "np.ndarray",
        count: int,
        *,
        calibration: "np.ndarray | None" = None,
        length: int | None = None,
    ) -> tuple["np.ndarray", int]:
        """Упакованные коды наборов 0…count−1 и длина кода.

        calibration — булева маска наборов (по числу наборов), по которым подбирается длина кода: обучающие окна;
        None — все наборы. Выбор идёт по маске, а не по порядку строк, чтобы проверочные окна не влияли на код.
        length — длина, уже подобранная по обучающим окнам (обученный уровень, новая картинка): тогда она не
        подбирается.
        """
        fired = np.bincount(np.asarray(rows, dtype=np.int64), minlength=int(count))
        rows, columns = self.with_ancestors(*self.strongest(rows, columns, levels))
        if length is not None:
            sample = int(count)
            length = int(length)
        elif calibration is None:
            sample = int(count)
            length = self.fit_length(rows, columns, sample)
        else:
            mask = np.asarray(calibration, dtype=bool)
            if mask.shape != (int(count),):
                raise ValueError("calibration: нужна булева маска по числу наборов")
            sample = max(1, int(mask.sum()))
            inside = mask[rows]
            length = self.fit_length(rows[inside], columns[inside], sample)
        bits = self.detector_bits(length)
        packed = np.zeros((int(count), length // 8), dtype=np.uint8)
        order = np.argsort(rows, kind="stable")
        rows, columns = rows[order], columns[order]
        starts = np.searchsorted(rows, np.arange(0, int(count) + self.CHUNK_ROWS, self.CHUNK_ROWS))
        ones = 0
        for chunk, first in enumerate(range(0, int(count), self.CHUNK_ROWS)):
            size = min(self.CHUNK_ROWS, int(count) - first)
            matrix = np.zeros((size, length), dtype=np.uint8)
            entries = slice(int(starts[chunk]), int(starts[chunk + 1]))
            chunk_rows, chunk_bits = self._pairs(rows[entries] - first, bits[columns[entries]])
            matrix[chunk_rows, chunk_bits] = 1
            ones += int(matrix.sum())
            packed[first : first + size] = np.packbits(matrix, axis=1)
        kept = np.bincount(rows, minlength=int(count))
        LOGGER.event(
            "activation.set_embedding.size",
            section=NEURAL_ANALOGY,
            data={
                "name": self._name,
                "calibration_sets": sample,
                "calibration": "обучающие окна" if calibration is not None else "все наборы",
                "target_density": self._target,
                "code_length": length,
                "density": ones / (max(1, int(count)) * length),
                "min_length": self._min_steps * self.LENGTH_STEP,
                "max_length": self._max_steps * self.LENGTH_STEP,
                "note": "§6.6, шаг 3: размер выходного слоя — по среднему числу активных детекторов; длина кода "
                "наименьшая, при которой средняя плотность кодов не выше заданной (§2)",
            },
        )
        LOGGER.event(
            "activation.set_embedding",
            section=MORPHOLOGICAL_EMBEDDINGS,
            data={
                "name": self._name,
                "sets": int(count),
                "detectors": self._detectors,
                "fired_per_set": float(fired.mean()) if fired.size else 0.0,
                "kept_per_set": float(kept.mean()) if kept.size else 0.0,
                "saturation_limit": self._saturation,
                "bits_per_set": ones / max(1, int(count)),
                "empty_sets": float(np.mean(kept == 0)) if kept.size else 0.0,
                "distinct_codes": int(np.unique(packed.view(np.dtype((np.void, packed.shape[1]))).reshape(-1)).shape[0]),
                "note": "эмбеддинг набора — биты σ самых сильных сработавших детекторов, один случайный бит на "
                "детектор (§6.3, §6.5, §7.1.12); 0 — без ограничения σ",
            },
        )
        return packed, length
