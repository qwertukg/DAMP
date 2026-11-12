"""Коды точек раскладки компактными массивами.

Раскладка большого числа кодов (§5.1.1: N точек — N кодов) не может держать по объекту Python на точку:
BitArray занимает ~11 КБ, 5 млн точек — десятки гигабайт. Таблица хранит то же самое массивами:
биты кодов упакованы по 8 в байт (§2.2: разреженные битовые векторы), известная компонента стимула (hue)
и подписи — по числу на точку. Объекты BitArray, строки «0101…» и точки LayoutPoint строятся из таблицы
по требованию и дают те же значения, что и раньше.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from damp.article_refs import SPARSE_BIT_VECTORS
from damp.encoding.bitarray import BitArray
from damp.logging import LOGGER


class LayoutCodeTable:
    """N кодов длины L: упакованные биты (N × ⌈L/8⌉, uint8), hue (float64), подписи и число единиц."""

    # Строк за одну распаковку бит (N × L байт не создаются целиком).
    CHUNK_ROWS = 16384

    def __init__(
        self,
        *,
        packed: "np.ndarray",
        code_length: int,
        hues: "np.ndarray",
        label_index: "np.ndarray",
        label_values: Sequence[str],
        ones: "np.ndarray | None" = None,
    ) -> None:
        if code_length <= 0:
            raise ValueError("code_length must be positive")
        packed = np.ascontiguousarray(packed, dtype=np.uint8)
        if packed.ndim != 2 or packed.shape[1] != -(-code_length // 8):
            raise ValueError("packed must be N × ceil(code_length / 8) bytes")
        count = int(packed.shape[0])
        hues = np.ascontiguousarray(hues, dtype=np.float64)
        label_index = np.ascontiguousarray(label_index, dtype=np.int32)
        if hues.shape != (count,) or label_index.shape != (count,):
            raise ValueError("hues and label_index must have one value per code")
        self._packed = packed
        self._code_length = int(code_length)
        self._hues = hues
        self._label_index = label_index
        self._label_values = tuple(str(value) for value in label_values)
        if ones is None:
            ones = self._count_ones(packed)
        self._ones = np.ascontiguousarray(ones, dtype=np.int32)
        if self._ones.shape != (count,):
            raise ValueError("ones must have one value per code")

    # ------------------------------------------------------------------ построение
    @staticmethod
    def _count_ones(packed: "np.ndarray") -> "np.ndarray":
        """Число единиц каждого кода (§2.2.4)."""
        if packed.shape[0] == 0:
            return np.zeros(0, dtype=np.int32)
        if hasattr(np, "bitwise_count"):
            return np.bitwise_count(packed).sum(axis=1, dtype=np.int64).astype(np.int32)
        return np.unpackbits(packed, axis=1).sum(axis=1, dtype=np.int64).astype(np.int32)

    @staticmethod
    def pack_bits(bits: "np.ndarray") -> "np.ndarray":
        """Матрица бит 0/1 (k × L) → упакованные байты (k × ⌈L/8⌉); хвост последнего байта — нули."""
        bits = np.asarray(bits)
        if bits.ndim != 2:
            raise ValueError("bits must be a k × L matrix")
        return np.packbits(bits != 0, axis=1)

    @classmethod
    def from_bits(
        cls,
        bits: "np.ndarray",
        *,
        hues: Sequence[float],
        labels: Sequence[object],
    ) -> "LayoutCodeTable":
        """Таблица из матрицы бит (N × L) и значений на точку; подписи — строки."""
        bits = np.asarray(bits)
        label_values: list[str] = []
        positions: dict[str, int] = {}
        label_index = np.empty(len(labels), dtype=np.int32)
        for row, label in enumerate(labels):
            text = str(label)
            slot = positions.get(text)
            if slot is None:
                slot = positions[text] = len(label_values)
                label_values.append(text)
            label_index[row] = slot
        return cls(
            packed=cls.pack_bits(bits),
            code_length=int(bits.shape[1]),
            hues=np.asarray(hues, dtype=np.float64),
            label_index=label_index,
            label_values=label_values,
        )

    @classmethod
    def from_codes(
        cls,
        codes: Sequence[BitArray],
        *,
        hues: Sequence[float],
        labels: Sequence[object],
    ) -> "LayoutCodeTable":
        """Таблица из кодов BitArray одной длины (точки в том же порядке)."""
        if not codes:
            raise ValueError("codes must be non-empty")
        code_length = len(codes[0])
        bits = np.zeros((len(codes), code_length), dtype=np.uint8)
        for row, code in enumerate(codes):
            if len(code) != code_length:
                raise ValueError("all codes must have the same length")
            bits[row] = np.frombuffer(code._bits, dtype=np.uint8, count=code_length)
        return cls.from_bits(bits, hues=hues, labels=labels)

    @classmethod
    def concatenate(cls, tables: Sequence["LayoutCodeTable"]) -> "LayoutCodeTable":
        """Одна таблица из нескольких с одной длиной кода; подписи объединяются по значению."""
        if not tables:
            raise ValueError("tables must be non-empty")
        code_length = tables[0].code_length
        label_values: list[str] = []
        positions: dict[str, int] = {}
        indices: list["np.ndarray"] = []
        for table in tables:
            if table.code_length != code_length:
                raise ValueError("all tables must have the same code length")
            mapping = np.empty(len(table._label_values), dtype=np.int32)
            for slot, text in enumerate(table._label_values):
                target = positions.get(text)
                if target is None:
                    target = positions[text] = len(label_values)
                    label_values.append(text)
                mapping[slot] = target
            indices.append(mapping[table._label_index] if table.count else np.zeros(0, dtype=np.int32))
        return cls(
            packed=np.concatenate([table._packed for table in tables], axis=0),
            code_length=code_length,
            hues=np.concatenate([table._hues for table in tables]),
            label_index=np.concatenate(indices),
            label_values=label_values,
            ones=np.concatenate([table._ones for table in tables]),
        )

    def take(self, order: "np.ndarray") -> "LayoutCodeTable":
        """Таблица с точками в порядке order."""
        order = np.asarray(order, dtype=np.int64)
        return LayoutCodeTable(
            packed=self._packed[order],
            code_length=self._code_length,
            hues=self._hues[order],
            label_index=self._label_index[order],
            label_values=self._label_values,
            ones=self._ones[order],
        )

    def hue_group_order(self) -> "np.ndarray":
        """Перестановка точек «по группам hue»: группы — в порядке первого появления hue, внутри — как были.

        Тот же порядок давал словарь hue → коды, собранный обходом записей (порядок точек раскладки).
        """
        if self.count == 0:
            return np.zeros(0, dtype=np.int64)
        _, first, inverse = np.unique(self._hues, return_index=True, return_inverse=True)
        # Ранг группы — место её первого появления среди групп.
        rank = np.empty(first.shape[0], dtype=np.int64)
        rank[np.argsort(first, kind="stable")] = np.arange(first.shape[0], dtype=np.int64)
        return np.argsort(rank[inverse.reshape(-1)], kind="stable")

    def grouped_by_hue(self) -> "LayoutCodeTable":
        """Таблица с точками, сгруппированными по hue (см. hue_group_order)."""
        order = self.hue_group_order()
        if np.array_equal(order, np.arange(self.count, dtype=np.int64)):
            return self
        return self.take(order)

    # ------------------------------------------------------------------ значения
    @property
    def count(self) -> int:
        return int(self._packed.shape[0])

    def __len__(self) -> int:
        return self.count

    @property
    def code_length(self) -> int:
        return self._code_length

    @property
    def packed(self) -> "np.ndarray":
        """Биты кодов по 8 в байт, старший бит байта — меньший номер бита кода (N × ⌈L/8⌉, uint8)."""
        return self._packed

    @property
    def hues(self) -> "np.ndarray":
        return self._hues

    @property
    def ones(self) -> "np.ndarray":
        """Число единиц каждого кода (int32)."""
        return self._ones

    @property
    def label_values(self) -> tuple[str, ...]:
        return self._label_values

    @property
    def label_index(self) -> "np.ndarray":
        return self._label_index

    def label(self, index: int) -> str:
        return self._label_values[int(self._label_index[index])]

    def labels(self) -> list[str]:
        values = self._label_values
        return [values[slot] for slot in self._label_index.tolist()]

    def hue_count(self) -> int:
        """Число различных значений hue."""
        return int(np.unique(self._hues).shape[0]) if self.count else 0

    def bits(self, start: int = 0, stop: int | None = None) -> "np.ndarray":
        """Биты кодов точек [start, stop) матрицей uint8 (строки × L)."""
        stop = self.count if stop is None else stop
        return np.unpackbits(self._packed[start:stop], axis=1, count=self._code_length)

    def bit_rows(self, indices: "np.ndarray") -> "np.ndarray":
        """Биты кодов точек indices матрицей uint8 (len(indices) × L)."""
        return np.unpackbits(self._packed[np.asarray(indices, dtype=np.int64)], axis=1, count=self._code_length)

    def bit_chunks(self) -> Iterable[tuple[int, "np.ndarray"]]:
        """Пары (номер первой точки, биты порции точек) по CHUNK_ROWS точек."""
        for start in range(0, self.count, self.CHUNK_ROWS):
            yield start, self.bits(start, min(self.count, start + self.CHUNK_ROWS))

    def words64(self) -> "np.ndarray":
        """Коды 64-битными словами (N × ⌈L/64⌉, uint64) для подсчёта общих бит через popcount (§2.2.4)."""
        words = -(-self._code_length // 64)
        if self._packed.shape[1] == words * 8:
            return self._packed.view(np.uint64)
        padded = np.zeros((self.count, words * 8), dtype=np.uint8)
        padded[:, : self._packed.shape[1]] = self._packed
        return padded.view(np.uint64)

    def values(self, start: int = 0, stop: int | None = None) -> list[str]:
        """Коды точек [start, stop) строками «0101…» — как BitArray.to01()."""
        stop = self.count if stop is None else stop
        length = self._code_length
        result: list[str] = []
        for first in range(start, stop, self.CHUNK_ROWS):
            last = min(stop, first + self.CHUNK_ROWS)
            text = (self.bits(first, last) + 48).tobytes().decode("ascii")
            result.extend(text[offset : offset + length] for offset in range(0, (last - first) * length, length))
        return result

    def code(self, index: int) -> BitArray:
        """Код точки объектом BitArray."""
        row = self.bits(index, index + 1)[0]
        return BitArray.from_bits(row.tobytes(), np.flatnonzero(row).tolist())

    def codes(self, start: int = 0, stop: int | None = None) -> list[BitArray]:
        """Коды точек [start, stop) объектами BitArray."""
        stop = self.count if stop is None else stop
        result: list[BitArray] = []
        for first in range(start, stop, self.CHUNK_ROWS):
            block = self.bits(first, min(stop, first + self.CHUNK_ROWS))
            result.extend(
                BitArray.from_bits(block[row].tobytes(), np.flatnonzero(block[row]).tolist())
                for row in range(block.shape[0])
            )
        return result

    def log_summary(self) -> None:
        """Сводка кодов раскладки: длина и число единиц (§2.2)."""
        ones = self._ones
        LOGGER.event(
            "layout.codes",
            section=SPARSE_BIT_VECTORS,
            data={
                "code_length": self._code_length if self.count else 0,
                "ones_min": int(ones.min()) if self.count else 0,
                "ones_max": int(ones.max()) if self.count else 0,
                "ones_avg": (int(ones.sum(dtype=np.int64)) / self.count) if self.count else 0,
            },
        )
