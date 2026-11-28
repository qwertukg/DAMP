"""Строки большого массива на диске (вне статьи, техника счёта): коды окон уровня после отбора образцов карты.

Коды всех окон уровня нужны отбору образцов карты (§5) и срабатываниям (§6.5), а между ними и после — только их число и
длины. Срабатывания читают коды частями по порядку, поэтому после отбора образцов коды уходят во временный файл и
читаются из него кусками (RowsOnDisk). На 40 000 картинках коды штрихов уровня 0 — 4.4 ГБ ОЗУ, на 60 000 — 6.6 ГБ
(разбор 2026-11-26, docs/SPEEDUP.md). Числа те же: чтение отдаёт те же байты.

Каждое чтение открывает отображение файла и сразу закрывает его: прочитанные страницы не остаются в рабочем наборе
процесса, их держит кеш файлов системы и отдаёт по первому требованию. Файл удаляется, когда объект больше не нужен
(weakref.finalize), и в любом случае — при выходе из процесса.

Малые массивы на диск не уходят (порог RowsOnDisk.MIN_BYTES, по умолчанию 1 ГиБ): у прогонов на сотнях и тысячах
картинок путь прежний. DAMP_SPILL_BYTES — порог в байтах (0 — на диск всё, для сверки «до бита»), DAMP_SPILL_DIR — папка
файлов (по умолчанию — «spill» рядом с папкой кеша шагов CACHE_DIR: в самой папке кеша файлы считались бы в его предел).
"""

from __future__ import annotations

import os
import tempfile
import time
import weakref
from pathlib import Path

import numpy as np


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


class RowsOnDisk:
    """Массив строк (строки × …) во временном файле: shape, dtype, ndim, len и чтение по номерам строк — срез, номер
    или массив номеров (как у numpy, с копией в ОЗУ). np.asarray(rows) читает весь массив."""

    on_disk = True
    MIN_BYTES = 1 << 30
    PREFIX = "damp-rows-"
    STALE_SECONDS = 24 * 3600  # файлы упавших прогонов: старше суток — удаляются при первом сбросе на диск
    _swept: set[str] = set()

    def __init__(self, array: "np.ndarray", directory: "str | os.PathLike | None" = None) -> None:
        array = np.ascontiguousarray(array)
        self.shape = tuple(int(size) for size in array.shape)
        self.dtype = array.dtype
        folder = Path(directory) if directory is not None else self.folder()
        folder.mkdir(parents=True, exist_ok=True)
        self._sweep(folder)
        handle, path = tempfile.mkstemp(prefix=self.PREFIX, suffix=".bin", dir=str(folder))
        try:
            with os.fdopen(handle, "wb") as file:
                array.tofile(file)
        except BaseException:
            _remove(path)
            raise
        self.path = path
        self._finalizer = weakref.finalize(self, _remove, path)

    # ------------------------------------------------------------------ куда и когда
    @staticmethod
    def folder() -> Path:
        found = os.environ.get("DAMP_SPILL_DIR")
        if found:
            return Path(found)
        from damp import settings

        return Path(settings.CACHE_DIR).parent / "spill"

    @classmethod
    def threshold(cls) -> int:
        found = os.environ.get("DAMP_SPILL_BYTES")
        return int(found) if found not in (None, "") else int(cls.MIN_BYTES)

    @classmethod
    def maybe(cls, array: "np.ndarray") -> "np.ndarray | RowsOnDisk":
        """array на диск, если он не меньше порога (threshold); иначе — как есть."""
        if isinstance(array, RowsOnDisk) or not isinstance(array, np.ndarray) or array.ndim < 1:
            return array
        return cls(array) if int(array.nbytes) >= cls.threshold() else array

    @classmethod
    def _sweep(cls, folder: Path) -> None:
        key = str(folder.resolve())
        if key in cls._swept:
            return
        cls._swept.add(key)
        now = time.time()
        for path in folder.glob(cls.PREFIX + "*.bin"):
            try:
                if now - path.stat().st_mtime > cls.STALE_SECONDS:
                    path.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------ как массив
    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def nbytes(self) -> int:
        return self.size * int(self.dtype.itemsize)

    def __len__(self) -> int:
        return self.shape[0]

    def _empty(self) -> "np.ndarray":
        return np.empty(self.shape, dtype=self.dtype)

    def __getitem__(self, key) -> "np.ndarray":  # noqa: ANN001
        if self.size == 0:
            return np.array(self._empty()[key])
        mapped = np.memmap(self.path, dtype=self.dtype, mode="r", shape=self.shape)
        try:
            return np.array(mapped[key])  # копия в ОЗУ: отображение закрывается, как только на него нет ссылок
        finally:
            del mapped

    def __array__(self, dtype=None, copy=None) -> "np.ndarray":  # noqa: ANN001
        found = self[:]
        return found if dtype is None else found.astype(dtype, copy=False)

    def release(self) -> None:
        """Удалить файл сразу (дальше читать нельзя)."""
        self._finalizer()

    def __repr__(self) -> str:
        return f"RowsOnDisk(shape={self.shape}, dtype={self.dtype}, path={self.path!r})"
