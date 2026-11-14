"""Код пары «что × где»: детекторы-произведения над детекторами двух координат (расширение статьи).

Код набора стимулов по §2.2.1 — OR их кодов. Если у стимула две координаты («что» и «где») и каждая
кодируется своими детекторами (§4.6), то OR набора теряет связь между ними: набор {(a₁, p₁), (a₂, p₂)}
даёт тот же код, что {(a₁, p₂), (a₂, p₁)}. Детектор-произведение эту связь сохраняет: он срабатывает,
только когда у одного стимула активны и детектор координаты «что», и детектор координаты «где».
Каждому детектору-произведению назначается один случайный бит кода, как широкому детектору в §4.2.1.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from damp.article_refs import CONJUNCTION, NEURAL_ANALOGY, OUTSIDE_ARTICLE
from damp.encoding.damp_encoder import Encoder
from damp.logging import LOGGER


class ConjunctionEncoder:
    """Код набора стимулов «что × где» детекторами-произведениями (расширение статьи; OR набора — §2.2.1).

    what, where — кодеры одной координаты каждый; из них берутся только детекторы и их слои (§4.1–§4.3):
    какой детектор активен для значения. Детектор-произведение — пара (детектор «что», детектор «где»)
    из слоёв с одним номером (cross_layers=True — из любых двух слоёв). Срабатывают все пары активных
    детекторов стимула, как срабатывают все широкие детекторы, в поле которых попал стимул (§4.1).
    """

    # Сколько элементов (стимулов) обрабатывается за один проход и сколько байт занимает плотный блок кодов.
    BLOCK_ELEMENTS = 40_000
    BLOCK_BYTES = 64 * 1024 * 1024
    LENGTH_STEP = 8
    FIT_ROUNDS = 12
    FIT_TOLERANCE = 0.008

    def __init__(
        self,
        what: Encoder,
        where: Encoder,
        *,
        random_seed: int,
        cross_layers: bool = False,
        nested: bool = False,
        name: str = "conjunction",
    ) -> None:
        if len(what.dimensions) != 1 or len(where.dimensions) != 1:
            raise ValueError("what and where encoders must have exactly one dimension each")
        self._what = what
        self._where = where
        self._seed = int(random_seed)
        # True — у двумерных координат срабатывают только вложенные круги (Encoder.active_detectors, nested).
        self._nested = bool(nested)
        self._active = 0
        self._name = str(name)
        self._what_layers = self._layer_ranges(what)
        self._where_layers = self._layer_ranges(where)
        if cross_layers:
            self._pairs = [(a, b) for a in range(len(self._what_layers)) for b in range(len(self._where_layers))]
        else:
            if len(self._what_layers) != len(self._where_layers):
                raise ValueError("what and where must have the same number of layers unless cross_layers is set")
            self._pairs = [(n, n) for n in range(len(self._what_layers))]
        self._detectors = sum(self._what_layers[a][1] * self._where_layers[b][1] for a, b in self._pairs)
        self._length = 0
        self._tables: list[np.ndarray] = []
        LOGGER.event(
            "conjunction.init",
            section=OUTSIDE_ARTICLE,
            data={
                "name": self._name,
                "what": what.dimensions[0].title,
                "where": where.dimensions[0].title,
                "what_layers": [count for _, count in self._what_layers],
                "where_layers": [count for _, count in self._where_layers],
                "layer_pairs": len(self._pairs),
                "cross_layers": bool(cross_layers),
                "nested": self._nested,
                "detectors": self._detectors,
                "note": "детектор-произведение = детектор «что» × детектор «где»; один случайный бит на детектор "
                "(как в §4.2.1); статья такого детектора не описывает",
            },
        )

    @staticmethod
    def _layer_ranges(encoder: Encoder) -> list[tuple[int, int]]:
        """Первый номер и число детекторов каждого слоя (детекторы слоя идут в кодере подряд)."""
        layers = encoder.detector_layer_indices()
        ranges = []
        for layer in range(int(layers.max()) + 1):
            members = np.flatnonzero(layers == layer)
            ranges.append((int(members[0]), int(members.size)))
        return ranges

    @property
    def code_length(self) -> int:
        return self._length

    @property
    def detector_count(self) -> int:
        return self._detectors

    def set_length(self, length: int) -> None:
        """Длина кода и случайные биты детекторов-произведений (по таблице на пару слоёв)."""
        length = max(self.LENGTH_STEP, int(math.ceil(length / self.LENGTH_STEP)) * self.LENGTH_STEP)
        self._length = length
        self._tables = [
            np.random.default_rng([self._seed, number]).integers(
                0, length, size=(self._what_layers[a][1], self._where_layers[b][1]), dtype=np.int32
            )
            for number, (a, b) in enumerate(self._pairs)
        ]

    def _padded(self, encoder: Encoder, ranges: Sequence[tuple[int, int]], values: "np.ndarray") -> list["np.ndarray"]:
        """Активные детекторы значений по слоям: (значения × k) номеров внутри слоя, −1 — пусто."""
        values = np.asarray(values, dtype=np.float64)
        count = values.shape[0]
        rows, detectors = encoder.active_detectors(0, values, nested=self._nested)
        layer_of = encoder.detector_layer_indices()[detectors]
        result = []
        for layer, (first, _) in enumerate(ranges):
            mine = layer_of == layer
            layer_rows, layer_ids = rows[mine], detectors[mine] - first
            order = np.argsort(layer_rows, kind="stable")
            layer_rows, layer_ids = layer_rows[order], layer_ids[order]
            per_row = np.bincount(layer_rows, minlength=count)
            width = max(1, int(per_row.max()) if count else 1)
            starts = np.concatenate([[0], np.cumsum(per_row)[:-1]]) if count else np.zeros(0, dtype=np.int64)
            padded = np.full((count, width), -1, dtype=np.int32)
            padded[layer_rows, np.arange(layer_rows.shape[0]) - starts[layer_rows]] = layer_ids
            result.append(padded)
        return result

    def _bits(self, what_pad, where_pad, what_index: "np.ndarray", where_index: "np.ndarray") -> "np.ndarray":
        """Биты детекторов-произведений каждого элемента: (элементы × K), −1 — пусто."""
        parts = []
        for (a, b), table in zip(self._pairs, self._tables):
            left, right = what_pad[a][what_index], where_pad[b][where_index]
            bits = table[np.maximum(left, 0)[:, :, None], np.maximum(right, 0)[:, None, :]]
            valid = (left >= 0)[:, :, None] & (right >= 0)[:, None, :]
            parts.append(np.where(valid, bits, -1).reshape(left.shape[0], -1))
        return np.concatenate(parts, axis=1)

    def _encode(self, what_pad, where_pad, what_index, where_index, groups, group_count: int) -> "np.ndarray":
        row_bytes = -(-self._length // 8)
        packed = np.zeros((group_count, row_bytes), dtype=np.uint8)
        order = np.argsort(groups, kind="stable")
        groups, what_index, where_index = groups[order], what_index[order], where_index[order]
        max_groups = max(1, self.BLOCK_BYTES // max(1, self._length))
        start, total = 0, groups.shape[0]
        self._active = 0
        while start < total:
            stop = min(total, start + self.BLOCK_ELEMENTS)
            stop = min(stop, int(np.searchsorted(groups, groups[start] + max_groups, side="left")))
            # Блок заканчивается на границе набора: код набора — OR всех его элементов (§2.2.1).
            stop = max(stop, start + 1)
            stop = int(np.searchsorted(groups, groups[stop - 1], side="right"))
            first, last = int(groups[start]), int(groups[stop - 1]) + 1
            dense = np.zeros((last - first, self._length), dtype=np.uint8)
            bits = self._bits(what_pad, where_pad, what_index[start:stop], where_index[start:stop])
            rows = np.repeat(groups[start:stop] - first, bits.shape[1])
            flat = bits.reshape(-1)
            keep = flat >= 0
            self._active += int(np.count_nonzero(keep))
            dense[rows[keep], flat[keep]] = 1
            packed[first:last] = np.packbits(dense, axis=1)
            start = stop
        return packed

    @staticmethod
    def _density(packed: "np.ndarray", length: int) -> float:
        if packed.shape[0] == 0:
            return 0.0
        return float(np.unpackbits(packed, axis=1, count=length).mean())

    def fit_length(
        self,
        what_values: "np.ndarray",
        what_index: "np.ndarray",
        where_values: "np.ndarray",
        where_index: "np.ndarray",
        groups: "np.ndarray",
        *,
        sample_groups: int,
        target_density: float,
        min_length: int,
        max_length: int,
    ) -> float:
        """Длина кода под заданную среднюю плотность кодов наборов (§6.6): по первым sample_groups наборам."""
        groups = np.asarray(groups, dtype=np.int64)
        take = groups < int(sample_groups)
        if not np.any(take):
            raise ValueError("no elements in the calibration sets")
        what_pad = self._padded(self._what, self._what_layers, what_values)
        where_pad = self._padded(self._where, self._where_layers, where_values)
        wi, pi, g = np.asarray(what_index)[take], np.asarray(where_index)[take], groups[take]
        sets = int(g.max()) + 1
        length, density, previous = float(max(min_length, 1024)), 0.0, 0
        for _ in range(self.FIT_ROUNDS):
            self.set_length(int(min(max_length, max(min_length, length))))
            density = self._density(self._encode(what_pad, where_pad, wi, pi, g, sets), self._length)
            # Стоп: плотность попала в допуск или длина перестала меняться (упёрлась в границу).
            if abs(density - target_density) < self.FIT_TOLERANCE or self._length == previous:
                break
            previous = self._length
            # Случайные биты: плотность = 1 − exp(−детекторов / длина), отсюда поправка длины.
            length = self._length * math.log(1.0 - min(density, 0.999)) / math.log(1.0 - target_density)
        LOGGER.event(
            "conjunction.code_size",
            section=NEURAL_ANALOGY,
            data={
                "name": self._name,
                "code_length": self._length,
                "density": density,
                "target_density": target_density,
                "calibration_sets": sets,
                "detectors": self._detectors,
                "note": "длина кода подобрана под среднюю плотность кодов наборов (§6.6, шаг 3)",
            },
        )
        return density

    def encode_sets(
        self,
        what_values: "np.ndarray",
        what_index: "np.ndarray",
        where_values: "np.ndarray",
        where_index: "np.ndarray",
        groups: "np.ndarray",
        group_count: int,
    ) -> "np.ndarray":
        """Коды наборов стимулов упакованными битами (наборы × ⌈L/8⌉): OR кодов элементов набора (§2.2.1).

        Элемент i — стимул (what_values[what_index[i]], where_values[where_index[i]]) набора groups[i].
        """
        if self._length <= 0:
            raise ValueError("code length is not set: call fit_length or set_length first")
        groups = np.asarray(groups, dtype=np.int64)
        what_pad = self._padded(self._what, self._what_layers, what_values)
        where_pad = self._padded(self._where, self._where_layers, where_values)
        packed = self._encode(
            what_pad,
            where_pad,
            np.asarray(what_index, dtype=np.int64),
            np.asarray(where_index, dtype=np.int64),
            groups,
            int(group_count),
        )
        sample = packed[: min(packed.shape[0], 2000)]
        LOGGER.event(
            "conjunction.encode.sets",
            section=CONJUNCTION,
            data={
                "name": self._name,
                "sets": int(group_count),
                "elements": int(groups.shape[0]),
                "elements_per_set": float(groups.shape[0]) / max(1, int(group_count)),
                "detectors_per_element": float(self._active) / max(1, int(groups.shape[0])),
                "code_length": self._length,
                "density": self._density(sample, self._length),
                "note": "код набора — OR битов детекторов-произведений его элементов",
            },
        )
        return packed
