from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Dict, List, Mapping, Sequence

import numpy as np

from damp.article_refs import (
    OUTSIDE_ARTICLE,
)
from damp.encoding.bitarray import BitArray
from damp.layout.code_table import LayoutCodeTable
from damp.logging import LOGGER


@dataclass(frozen=True)
class LayoutEncodedCode:
    code: BitArray
    label: str


@dataclass(frozen=True)
class LayoutCodeRecord:
    index: int
    label: str
    hue: float
    code: BitArray

    @property
    def ones(self) -> int:
        return self.code.count()

    def as_payload(self, *, y: int | None, x: int | None) -> Dict[str, object]:
        return {
            "index": self.index,
            "y": y,
            "x": x,
            "label": self.label,
            "value": self.code.to01(),
            "hue": self.hue,
            "ones": self.ones,
        }


class LayoutJsonWriter:
    """Файл раскладки JSON потоком, без словаря на каждую точку (вне статьи: формат хранения).

    Текст файла — тот же, что даёт json.dump(payload, indent=2, ensure_ascii=True) для словаря с ключами
    верхнего уровня и списком точек под ключом POINTS_KEY: {index, y, x, label, value, hue, ones}.
    Точки пишутся порциями прямо из таблицы кодов, поэтому память не зависит от их числа.
    """

    POINTS_KEY = "layout"
    # Точек в одной порции записи.
    CHUNK_POINTS = 8192
    # Файл больше этого размера не читается целиком: ключи верхнего уровня берутся из его начала и конца.
    LARGE_FILE_BYTES = 64 * 1024 * 1024
    # Сколько байт начала и конца большого файла просматривается в поисках списка точек.
    PROBE_BYTES = 4 * 1024 * 1024
    _POINTS_OPEN = b'\n  "layout": ['
    _POINTS_CLOSE = b"\n  ]"

    @staticmethod
    def _value_text(value: object) -> str:
        """Значение ключа верхнего уровня — как его пишет json.dump с отступом 2 на первом уровне."""
        return json.dumps(value, ensure_ascii=True, indent=2).replace("\n", "\n  ")

    @staticmethod
    def _number_texts(values: "np.ndarray | None", count: int) -> list[str]:
        if values is None:
            return ["null"] * count
        return [str(value) for value in np.asarray(values).tolist()]

    @classmethod
    def _point_texts(
        cls,
        table: LayoutCodeTable,
        start: int,
        stop: int,
        y: "np.ndarray | None",
        x: "np.ndarray | None",
    ) -> list[str]:
        count = stop - start
        ys = cls._number_texts(None if y is None else y[start:stop], count)
        xs = cls._number_texts(None if x is None else x[start:stop], count)
        values = table.values(start, stop)
        label_texts = [json.dumps(text, ensure_ascii=True) for text in table.label_values]
        labels = [label_texts[slot] for slot in table.label_index[start:stop].tolist()]
        # json пишет float через float.__repr__; нечисла и бесконечности — словами NaN, Infinity.
        hue_values = table.hues[start:stop]
        if bool(np.isfinite(hue_values).all()):
            hues = [repr(value) for value in hue_values.tolist()]
        else:
            hues = [json.dumps(value) for value in hue_values.tolist()]
        ones = table.ones[start:stop].tolist()
        return [
            f'{{\n      "index": {start + row},\n      "y": {ys[row]},\n      "x": {xs[row]},\n'
            f'      "label": {labels[row]},\n      "value": "{values[row]}",\n      "hue": {hues[row]},\n'
            f'      "ones": {ones[row]}\n    }}'
            for row in range(count)
        ]

    @classmethod
    def write(
        cls,
        path: str,
        payload: Mapping[str, object],
        table: LayoutCodeTable,
        *,
        y: "np.ndarray | None" = None,
        x: "np.ndarray | None" = None,
    ) -> None:
        """Пишет payload; на месте ключа POINTS_KEY (его значение игнорируется) — список точек table.

        y, x — ячейки точек (None — точки ещё не расставлены, пишется null).
        """
        if cls.POINTS_KEY not in payload:
            raise ValueError("payload must contain the points key")
        with open(path, "w", encoding="utf-8") as fp:
            fp.write("{")
            first_key = True
            for key, value in payload.items():
                fp.write("\n  " if first_key else ",\n  ")
                first_key = False
                fp.write(json.dumps(str(key), ensure_ascii=True) + ": ")
                if key != cls.POINTS_KEY:
                    fp.write(cls._value_text(value))
                    continue
                if table.count == 0:
                    fp.write("[]")
                    continue
                fp.write("[\n    ")
                for start in range(0, table.count, cls.CHUNK_POINTS):
                    stop = min(table.count, start + cls.CHUNK_POINTS)
                    if start:
                        fp.write(",\n    ")
                    fp.write(",\n    ".join(cls._point_texts(table, start, stop, y, x)))
                fp.write("\n  ]")
            fp.write("\n}")

    @classmethod
    def read_header(cls, path: str) -> dict[str, object] | None:
        """Ключи верхнего уровня файла раскладки по порядку; значение списка точек заменено на None.

        Небольшой файл читается целиком; у большого берутся начало (до списка точек) и конец (после него):
        список точек на гигабайты ради нескольких ключей не разбирается.
        """
        size = os.path.getsize(path)
        if size > cls.LARGE_FILE_BYTES:
            with open(path, "rb") as fp:
                head = fp.read(cls.PROBE_BYTES)
                fp.seek(max(0, size - cls.PROBE_BYTES))
                tail = fp.read()
            start = head.find(cls._POINTS_OPEN)
            end = tail.rfind(cls._POINTS_CLOSE)
            if start >= 0 and end >= 0:
                text = head[:start] + b'\n  "layout": null' + tail[end + len(cls._POINTS_CLOSE) :]
                try:
                    payload = json.loads(text.decode("utf-8"))
                except ValueError:
                    payload = None
                if isinstance(payload, dict):
                    return payload
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if not isinstance(payload, dict):
            return None
        if cls.POINTS_KEY in payload:
            payload[cls.POINTS_KEY] = None
        return payload


class LayoutPayloadBuilder:
    """Коды для раскладки в порядке поступления: таблицей массивов, без объекта на каждый код."""

    def __init__(
        self,
        *,
        similarity: str | None,
        lambda_threshold: float | None,
        eta: float | None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self._chunks: List[LayoutCodeTable] = []
        self._count = 0
        self._table: LayoutCodeTable | None = None
        self._similarity = similarity
        self._lambda_threshold = lambda_threshold
        self._eta = eta
        self._metadata: Dict[str, object] = dict(metadata or {})

    def __len__(self) -> int:
        return self._count

    @property
    def records(self) -> Sequence[LayoutCodeRecord]:
        table = self.table()
        labels = table.labels()
        return tuple(
            LayoutCodeRecord(index=index, label=labels[index], hue=float(table.hues[index]), code=code)
            for index, code in enumerate(table.codes())
        )

    def add_code(self, *, label: int | str, hue: float, code: BitArray) -> LayoutCodeRecord:
        bits = np.frombuffer(code._bits, dtype=np.uint8, count=len(code)).reshape(1, -1)
        self.add_codes(label=label, hues=[float(hue)], bits=bits)
        return LayoutCodeRecord(index=self._count - 1, label=str(label), hue=float(hue), code=code)

    def add_codes(self, *, label: int | str, hues: Sequence[float], bits: "np.ndarray") -> None:
        """Коды одного стимула (например, все углопозиции цифры): матрица бит k × L и hue каждого кода."""
        bits = np.asarray(bits)
        if bits.ndim != 2 or bits.shape[0] != len(hues):
            raise ValueError("bits must be a k × L matrix with one hue per row")
        if bits.shape[0] == 0:
            return
        self._chunks.append(LayoutCodeTable.from_bits(bits, hues=hues, labels=[str(label)] * bits.shape[0]))
        self._count += int(bits.shape[0])
        self._table = None

    def add_packed(self, *, label: int | str, hues: Sequence[float], packed: "np.ndarray", code_length: int) -> None:
        """Коды одного стимула упакованными битами (k × ⌈L/8⌉, uint8) и hue каждого кода."""
        if len(hues) == 0:
            return
        self.add_table(
            LayoutCodeTable(
                packed=packed,
                code_length=code_length,
                hues=np.asarray(hues, dtype=np.float64),
                label_index=np.zeros(len(hues), dtype=np.int32),
                label_values=(str(label),),
            )
        )

    def add_table(self, table: LayoutCodeTable) -> None:
        """Готовая таблица кодов (точки в порядке поступления)."""
        if table.count == 0:
            return
        self._chunks.append(table)
        self._count += table.count
        self._table = None

    def table(self) -> LayoutCodeTable:
        """Все коды в порядке поступления."""
        if self._table is None:
            if not self._chunks:
                # Кодов нет: пустая таблица (длина кода условная), список точек файла — пустой.
                return LayoutCodeTable(
                    packed=np.zeros((0, 1), dtype=np.uint8),
                    code_length=1,
                    hues=np.zeros(0, dtype=np.float64),
                    label_index=np.zeros(0, dtype=np.int32),
                    label_values=(),
                )
            self._table = self._chunks[0] if len(self._chunks) == 1 else LayoutCodeTable.concatenate(self._chunks)
            self._chunks = [self._table]
        return self._table

    def code_table(self) -> LayoutCodeTable:
        """Коды в порядке точек раскладки: группы hue в порядке первого появления (как codes_by_hue)."""
        return self.table().grouped_by_hue()

    def codes_by_hue(self) -> Dict[float, List[LayoutEncodedCode]]:
        grouped: Dict[float, List[LayoutEncodedCode]] = {}
        if self._count == 0:
            return grouped
        table = self.table()
        labels = table.labels()
        for index, code in enumerate(table.codes()):
            grouped.setdefault(float(table.hues[index]), []).append(LayoutEncodedCode(code=code, label=labels[index]))
        return grouped

    def _header(self) -> Dict[str, object]:
        return {
            **self._metadata,
            "width": None,
            "height": None,
            "points": self._count,
            "similarity": self._similarity,
            "lambda_threshold": self._lambda_threshold,
            "eta": self._eta,
            "pair_radius": None,
            LayoutJsonWriter.POINTS_KEY: None,
        }

    def base_payload(self) -> Mapping[str, object]:
        payload = self._header()
        payload[LayoutJsonWriter.POINTS_KEY] = [record.as_payload(y=None, x=None) for record in self.records]
        return payload

    # Оценка размера записи точки в файле раскладки сверх строки кода: ключи, подпись, hue, ячейка.
    POINT_OVERHEAD_BYTES = 170

    def save_base(self, path: str, *, max_points_bytes: int | None = None) -> None:
        """Записать файл раскладки до самой раскладки: заголовок (подпись кодера, параметры) и коды точек.

        max_points_bytes — предел размера списка точек: если он больше (такой файл как кеш раскладки всё равно
        не читается), пишется только заголовок, а точки с ячейками запишет раскладка (Layout.save_json):
        список на гигабайты не пишется дважды.
        """
        if not path:
            raise ValueError("path must be provided for layout payload export")
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        table = self.table()
        estimated = table.count * (table.code_length + self.POINT_OVERHEAD_BYTES)
        points_written = max_points_bytes is None or estimated <= max_points_bytes
        if not points_written:
            table = table.take(np.zeros(0, dtype=np.int64))
        LayoutJsonWriter.write(path, self._header(), table)
        LOGGER.event(
            "layout.payload.base",
            section=OUTSIDE_ARTICLE,
            data={
                "path": path,
                "points": self._count,
                "points_written": points_written,
                "similarity": self._similarity,
                "lambda_threshold": self._lambda_threshold,
                "eta": self._eta,
            },
        )
