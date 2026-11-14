"""Эмбеддинги наборов стимулов из сработавших детекторов (§6.5, §6.6, §7.1.12).

HierarchyDetection.active_sets отвечает, какие детекторы и с каким уровнем сработали на каждый набор. Здесь
из них собирается выходной код:

    - у детектора один случайный бит выходного кода (§6.3);
    - в код идут самые сильные детекторы набора, не больше σ (динамический порог μ_c, §7.1.12);
    - код набора — объединение их бит (§6.5; при одном бите на детектор colour merge с бюджетом σ и есть
      выбор σ сильнейших);
    - длина кода — наименьшая, при которой средняя плотность кодов не выше заданной (§6.6, шаг 3; §2).
"""
from __future__ import annotations

import numpy as np

from damp.article_refs import MORPHOLOGICAL_EMBEDDINGS, NEURAL_ANALOGY
from damp.logging import LOGGER


class DetectorEmbeddings:
    """Коды наборов по сработавшим детекторам: выбор σ сильнейших, длина под плотность, один бит на детектор."""

    # Длины перебираются кратными этому числу: код хранится упакованным по 8 бит в байт.
    LENGTH_STEP = 8
    # Строк за одну сборку бит: матрица «наборы × биты» целиком не создаётся.
    CHUNK_ROWS = 32768
    # Ячеек плотной матрицы «наборы × детекторы» за один проход отбора самых сильных детекторов.
    SELECT_CELLS = 8_000_000

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
    ) -> None:
        if detectors <= 0:
            raise ValueError("detectors must be positive")
        if saturation_limit < 0:
            raise ValueError("saturation_limit must be non-negative (0 — без ограничения)")
        if not 0.0 < target_density < 1.0:
            raise ValueError("target_density must be in (0, 1)")
        if min_length <= 0 or max_length < min_length:
            raise ValueError("code length bounds must satisfy 0 < min_length <= max_length")
        self._detectors = int(detectors)
        self._saturation = int(saturation_limit)
        self._target = float(target_density)
        self._min_steps = -(-int(min_length) // self.LENGTH_STEP)
        self._max_steps = max(self._min_steps, int(max_length) // self.LENGTH_STEP)
        self._seed = int(seed)
        self._name = str(name)

    def strongest(
        self, rows: "np.ndarray", columns: "np.ndarray", levels: "np.ndarray"
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """Пары (набор, детектор), оставшиеся после порога μ_c: σ самых сильных детекторов набора (§7.1.12)."""
        rows = np.asarray(rows, dtype=np.int64)
        columns = np.asarray(columns, dtype=np.int64)
        if self._saturation == 0 or rows.size == 0:
            return rows, columns
        levels = np.asarray(levels, dtype=np.float64)
        if np.any(np.diff(rows) < 0):
            order = np.argsort(rows, kind="stable")
            rows, columns, levels = rows[order], columns[order], levels[order]
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
        """Бит каждого детектора в коде длины length: случайная перестановка, пока детекторов не больше длины."""
        rng = np.random.default_rng(self._seed)
        parts = []
        left = self._detectors
        while left > 0:
            parts.append(rng.permutation(int(length))[:left])
            left -= int(length)
        return np.concatenate(parts).astype(np.int64)

    def _density(self, rows: "np.ndarray", columns: "np.ndarray", count: int, length: int) -> float:
        bits = self.detector_bits(length)
        keys = np.unique(rows * int(length) + bits[columns])
        return float(keys.shape[0]) / (max(1, int(count)) * int(length))

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
        calibration: int | None = None,
    ) -> tuple["np.ndarray", int]:
        """Упакованные коды наборов 0…count−1 и длина кода; длина подбирается по первым calibration наборам."""
        fired = np.bincount(np.asarray(rows, dtype=np.int64), minlength=int(count))
        rows, columns = self.strongest(rows, columns, levels)
        sample = int(count) if calibration is None else max(1, min(int(calibration), int(count)))
        inside = rows < sample
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
            matrix[rows[entries] - first, bits[columns[entries]]] = 1
            ones += int(matrix.sum())
            packed[first : first + size] = np.packbits(matrix, axis=1)
        kept = np.bincount(rows, minlength=int(count))
        LOGGER.event(
            "activation.set_embedding.size",
            section=NEURAL_ANALOGY,
            data={
                "name": self._name,
                "calibration_sets": sample,
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
