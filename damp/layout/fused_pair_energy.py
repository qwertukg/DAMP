"""Энергии пар (§5.5.1, §5.5.2) и точек (§5.7) одним слитым ядром на слагаемое — те же числа float32.

Что ускоряется. Исходный проход TorchPairEnergyEngine на каждую порцию пар строит матрицу слагаемых
[пары, точки] полусотней поэлементных операций torch: общие биты кодов (матричное произведение), нормировка
сходства, порог λ (§5.4), расстояния, окрестность, вес, произведения. Каждая операция — отдельный проход
по памяти устройства, поэтому время шага — это число проходов, а не арифметика (§5.11, §5.12).

Математика та же, меняется только способ счёта:

  1. Сходство кодов от расстановки точек не зависит и предвычисляется (§5.9.5). Пока помещается
     в память — таблицей самих значений sim_λ(a, v) float32 [точки × точки] (SimilarityTable): её считает
     функция сходства исходного прохода, при смене λ (§7.1.7) она строится заново. Иначе — таблицей числа
     общих бит кодов (§2.2.4) uint8 [точки × точки] или [классы одинаковых кодов]² (CommonBitsTable):
     целые числа, от λ и η не зависят; sim_λ тогда считает ядро. Если не помещается и она — общие биты
     считаются на лету: по окну ячеек — ядром popcount(a & v) по упакованным 64-битным словам кодов
     (одна выборка слов кода точки окна и два запуска ядра вместо десятка операций над словами),
     по всему V — произведением кодов, как в исходном проходе. Число общих бит — целое, все способы
     дают одно и то же число.
  2. Слагаемое считает одно ядро на элемент (пара, точка): sim = common·n_a^−½·n_v^−½ и порог λ
     (жёсткий или сигмоида η) — если на входе общие биты; окрестность ‖v − c‖ ≤ r, вес (гаусс,
     1/(d + ε) либо d) и оба слагаемых φ_c: s1·w1 + s2·w2 и φ_s: s2·w1 + s1·w2 (§5.5). Операции и их
     порядок — те же, что у исходного прохода, с округлением каждой отдельно (без слияния умножения
     и сложения в FMA), поэтому каждое слагаемое побитово то же.
  3. Суммирует та же редукция sum(dim=1). Сумма float32 зависит от порядка сложения; у редукции CUDA
     порядок внутри строки определяют ширина строки, выравнивание её начала (16 байт) и — при малом
     числе строк — число строк тензора. Поэтому полные порции пар исходного прохода (у всех одна форма)
     ставятся в один тензор так, чтобы каждая начиналась со строки, кратной 4 (между порциями — пустые
     пары), а остаток суммируется отдельным тензором той же формы, что в исходном проходе.

Проверки. Объединение порций сверяется с суммами отдельных тензоров порций при первой встрече каждой
ширины строки; весь расчёт сверяется с исходным проходом движком (TorchPairEnergyEngine.pair_energies).
При любом расхождении слитый расчёт отключается, энергии считает исходный проход.
"""
from __future__ import annotations

import math
import os
import time
from typing import Any

import numpy as np

from damp.article_refs import GPU_IMPLEMENTATION, OPTIM_SIM_MATRIX, OUTSIDE_ARTICLE
from damp.logging import LOGGER


class KernelDiskCache:
    """Кеш скомпилированных ядер jiterator на диске (инфраструктура, вне статьи).

    torch компилирует ядро при первом вызове (≈ 0.3–0.6 с) и кладёт результат в каталог кеша, чтобы
    следующий запуск программы взял готовое. Файл пишется под временным именем «…_tmp_<pid>» и затем
    переименовывается; на Windows torch переименовывает ещё открытый файл, переименование не удаётся,
    и каждый запуск компилирует все ядра заново. publish довершает переименование для ядер этого модуля
    (имена начинаются с PREFIX) и убирает устаревшие временные файлы прошлых запусков.
    """

    PREFIX = "damp_"
    _MARK = "_tmp_"
    # Временные файлы других процессов моложе этого возраста не трогаются: процесс может ещё работать.
    STALE_SECONDS = 600.0

    @classmethod
    def directory(cls) -> str | None:
        """Каталог кеша ядер torch — те же правила, что в torch (jit_utils): переменная окружения или TEMP."""
        if os.environ.get("USE_PYTORCH_KERNEL_CACHE") == "0":
            return None
        explicit = os.environ.get("PYTORCH_KERNEL_CACHE_PATH")
        if explicit:
            return explicit
        temp = os.environ.get("TEMP")
        return None if not temp else os.path.join(temp, "torch", "kernels")

    @classmethod
    def publish(cls) -> int:
        """Переименовать временные файлы ядер этого процесса в постоянные; вернуть число опубликованных."""
        if os.name != "nt":
            return 0
        directory = cls.directory()
        if directory is None or not os.path.isdir(directory):
            return 0
        own = f"{cls._MARK}{os.getpid()}"
        published = 0
        removed = 0
        now = time.time()
        try:
            names = os.listdir(directory)
        except OSError:
            return 0
        for name in names:
            if not name.startswith(cls.PREFIX) or cls._MARK not in name:
                continue
            source = os.path.join(directory, name)
            target = os.path.join(directory, name[: name.rindex(cls._MARK)])
            try:
                if name.endswith(own):
                    if os.path.exists(target):
                        os.remove(source)
                        removed += 1
                    else:
                        os.replace(source, target)
                        published += 1
                elif os.path.exists(target) and now - os.path.getmtime(source) > cls.STALE_SECONDS:
                    os.remove(source)
                    removed += 1
            except OSError:
                continue
        if published or removed:
            LOGGER.event(
                "layout.fused.kernel_cache",
                section=OUTSIDE_ARTICLE,
                data={
                    "directory": directory,
                    "published": published,
                    "removed": removed,
                    "note": "скомпилированные ядра сохранены для следующих запусков",
                },
            )
        return published


class _Kernel:
    """Ядро jiterator: после первого вызова с новым набором типов входов публикует файл кеша на диске."""

    def __init__(self, function: Any) -> None:
        self._function = function
        self._signatures: set[tuple] = set()

    def __call__(self, *tensors: Any, **scalars: Any) -> Any:
        result = self._function(*tensors, **scalars)
        signature = tuple(tensor.dtype for tensor in tensors)
        if signature not in self._signatures:
            # Под каждый набор типов входов torch компилирует своё ядро.
            self._signatures.add(signature)
            KernelDiskCache.publish()
        return result


class CommonBitsTable:
    """Число общих бит кодов (§2.2.4) для всех пар классов одинаковых кодов — таблица на устройстве (§5.9.5).

    Класс — множество точек с одним и тем же кодом. Пока таблица [точки]² помещается в память, класс —
    сама точка (без лишней выборки номера класса); иначе таблица строится по различным кодам. Последние
    строка и столбец — нули: «точки нет» (пустая ячейка окна, пустая вторая ячейка пары).
    """

    # Предел элементов (строки × классы) на блок матричного произведения при построении.
    BLOCK_ELEMENTS = 32_000_000

    def __init__(self, torch_mod: Any, table: Any, class_of_points: Any | None, point_count: int) -> None:
        self._torch = torch_mod
        self._table = table
        self._classes = int(table.shape[0]) - 1
        self._point_count = int(point_count)
        self._flat = table.view(-1)
        # Номер класса точки; у «точки нет» (номер N) — нулевая строка таблицы.
        self._class_of_points = class_of_points
        if class_of_points is None:
            self._class_ext = None
            self._class_ext_np = None
            self._rows_view = table[:, : self._point_count]
        else:
            class_ext = torch_mod.empty(self._point_count + 1, dtype=class_of_points.dtype, device=table.device)
            class_ext[: self._point_count] = class_of_points
            class_ext[self._point_count] = self._classes
            self._class_ext = class_ext
            self._class_ext_np = class_ext.cpu().numpy().astype(np.int64)
            self._rows_view = None

    @classmethod
    def build(cls, engine: Any, *, max_bytes: int) -> "CommonBitsTable | None":
        """Таблица общих бит; None — не помещается в max_bytes даже по классам одинаковых кодов."""
        torch = engine.torch_mod
        device = engine.torch_device
        count = engine.point_count
        if not engine.has_float_codes:
            LOGGER.event(
                "layout.fused.table.skipped",
                section=OPTIM_SIM_MATRIX,
                data={
                    "points": count,
                    "note": "точек слишком много для таблицы общих бит: коды на устройстве — 64-битными словами, "
                    "общие биты считаются на лету (§5.9.5)",
                },
            )
            return None
        codes = engine.codes
        max_ones = int(engine.ones.max().item()) if count else 0
        dtype = torch.uint8 if max_ones <= 255 else torch.int16
        item_bytes = 1 if max_ones <= 255 else 2
        class_of_points = None
        class_codes = codes
        classes = count
        if (count + 1) ** 2 * item_bytes > max_bytes:
            # Точек слишком много: таблица по различным кодам (у одинаковых кодов строки таблицы равны).
            packed = np.packbits(codes.to(torch.uint8).cpu().numpy(), axis=1)
            _, first, inverse = np.unique(packed, axis=0, return_index=True, return_inverse=True)
            classes = int(first.size)
            if (classes + 1) ** 2 * item_bytes > max_bytes:
                LOGGER.event(
                    "layout.fused.table.skipped",
                    section=OPTIM_SIM_MATRIX,
                    data={
                        "points": count,
                        "classes": classes,
                        "table_mb": (classes + 1) ** 2 * item_bytes / 1e6,
                        "max_mb": max_bytes / 1e6,
                        "note": "таблица общих бит не помещается в память: общие биты считаются на лету (§5.9.5)",
                    },
                )
                return None
            index_dtype = torch.int32 if (classes + 1) ** 2 < 2**31 else torch.int64
            class_of_points = torch.as_tensor(inverse.reshape(-1).astype(np.int64), device=device).to(index_dtype)
            class_codes = codes[torch.as_tensor(first.astype(np.int64), device=device)]
        with torch.no_grad():
            table = torch.zeros((classes + 1, classes + 1), dtype=dtype, device=device)
            block = max(1, cls.BLOCK_ELEMENTS // max(1, classes))
            transposed = class_codes.T
            for start in range(0, classes, block):
                stop = min(classes, start + block)
                # Сумма произведений бит — целое не больше длины кода: в float32 точна при любом порядке.
                table[start:stop, :classes] = (class_codes[start:stop] @ transposed).to(dtype)
        LOGGER.event(
            "layout.fused.table.build",
            section=OPTIM_SIM_MATRIX,
            data={
                "points": count,
                "classes": classes,
                "by_code_classes": class_of_points is not None,
                "table_mb": (classes + 1) ** 2 * item_bytes / 1e6,
                "dtype": "uint8" if item_bytes == 1 else "int16",
                "note": "число общих бит кодов всех пар (§2.2.4) предвычислено один раз (§5.9.5); "
                "от λ и η не зависит",
            },
        )
        return cls(torch, table, class_of_points, count)

    @property
    def index_dtype(self) -> Any:
        """Тип линейного индекса таблицы: int32, пока (классы + 1)² < 2³¹."""
        return self._torch.int32 if (self._classes + 1) ** 2 < 2**31 else self._torch.int64

    @property
    def stride(self) -> int:
        return self._classes + 1

    @property
    def flat(self) -> Any:
        return self._flat

    def row_offsets(self, points_ext: "np.ndarray") -> "np.ndarray":
        """Начало строки таблицы для точек (номер N — «точки нет»): класс · (классы + 1), на хосте."""
        if self._class_ext_np is None:
            return points_ext.astype(np.int64) * self.stride
        return self._class_ext_np[points_ext] * self.stride

    def rows(self, points_ext: Any) -> Any:
        """Общие биты точек points_ext со всеми точками V: [точки пар, N]."""
        torch = self._torch
        if self._class_ext is None:
            return torch.index_select(self._rows_view, 0, points_ext)
        by_class = torch.index_select(self._table, 0, torch.index_select(self._class_ext, 0, points_ext))
        return torch.index_select(by_class, 1, self._class_of_points)

    def columns(self, points_ext: Any) -> Any:
        """Столбцы таблицы для точек окна (номер N — пустая ячейка)."""
        if self._class_ext is None:
            return points_ext
        return self._torch.index_select(self._class_ext, 0, points_ext)


class SimilarityTable:
    """sim_λ(a, v) всех пар точек (§5.4) — таблица float32 (N + 1) × (N + 1) на устройстве (§5.9.5).

    Значения считает функция движка similarity_rows — те же операции float32 над теми же числами, что
    в исходном проходе, поэтому элемент таблицы побитово равен сходству, которое проход считает на лету.
    Последние строка и столбец — нули: «точки нет» (пустая ячейка окна, пустая вторая ячейка пары).
    Таблица зависит от λ и η: при их смене (§7.1.7) строится заново.
    """

    # Предел элементов (строки × точки) на блок расчёта сходства при построении.
    BLOCK_ELEMENTS = 16_000_000

    def __init__(self, engine: Any, *, lambda_threshold: float, eta: float | None) -> None:
        torch = engine.torch_mod
        device = engine.torch_device
        count = engine.point_count
        self._torch = torch
        self._lambda = float(lambda_threshold)
        self._eta = None if eta is None else float(eta)
        self._stride = count + 1
        with torch.no_grad():
            table = torch.zeros((count + 1, count + 1), dtype=torch.float32, device=device)
            block = max(1, self.BLOCK_ELEMENTS // max(1, count))
            for start in range(0, count, block):
                stop = min(count, start + block)
                rows = torch.arange(start, stop, device=device)
                table[start:stop, :count] = engine.similarity_rows(rows, lambda_threshold=lambda_threshold, eta=eta)
        self._table = table
        self._rows_view = table[:, :count]
        self._flat = table.view(-1)
        LOGGER.event(
            "layout.fused.similarity_table.build",
            section=OPTIM_SIM_MATRIX,
            data={
                "points": count,
                "lambda_threshold": self._lambda,
                "eta": 0.0 if self._eta is None else self._eta,
                "eta_mode": "hard_cutoff" if self._eta is None else "sigmoid",
                "table_mb": self.bytes_needed(count) / 1e6,
                "note": "sim_λ всех пар точек предвычислена (§5.9.5) теми же операциями, что в проходе по V (§5.4)",
            },
        )

    @staticmethod
    def bytes_needed(count: int) -> int:
        return 4 * (count + 1) ** 2

    def matches(self, lambda_threshold: float, eta: float | None) -> bool:
        return self._lambda == float(lambda_threshold) and self._eta == (None if eta is None else float(eta))

    @property
    def stride(self) -> int:
        return self._stride

    @property
    def flat(self) -> Any:
        return self._flat

    def rows(self, points_ext: Any) -> Any:
        """sim_λ точек points_ext (N — «точки нет») со всеми точками V: [точки пар, N]."""
        return self._torch.index_select(self._rows_view, 0, points_ext)


class FusedPairEnergy:
    """φ_c, φ_s пар (§5.5) и энергии точек (§5.7) слитым ядром: те же числа, что у TorchPairEnergyEngine."""

    # Координаты ячейки (y, x) упакованы в одно число y·PACK + x (точно в float32, пока y, x < PACK).
    PACK = 4096
    # Смещения ячеек окна упакованы как (dy + BIAS)·PACK + (dx + BIAS).
    OFFSET_BIAS = 2048
    # Полные порции исходного прохода объединяются в один тензор редукции, если строк в порции не меньше
    # стольких: при меньшем числе строк порядок сложения внутри строки зависит от числа строк тензора.
    # Для каждой ширины строки объединение сверяется с суммами отдельных порций и при расхождении
    # для этой ширины не применяется.
    MERGE_MIN_ROWS = 16
    # Порция в общем тензоре начинается со строки, кратной стольким: начало строки выровнено на 16 байт
    # так же, как в отдельном тензоре порции (редукция читает float32 четвёрками).
    ROW_ALIGN = 4
    # Память устройства на элемент блока (пара × точка × 2 суммы) в самом затратном случае — расчёт по окну
    # с таблицей по классам кодов: номера ячеек и точек, индексы таблицы int64, общие биты, слагаемые.
    BLOCK_BYTES_PER_ELEMENT = 32
    # То же при подсчёте общих бит на лету: к элементу добавляются слова кода точки окна (64 байта на точку).
    PACKED_BLOCK_BYTES_PER_ELEMENT = 72
    # Слов кода на один запуск ядра popcount (у ядра не больше 8 входов: слова точки пары и точки окна).
    POPCOUNT_WORDS = 4
    # Наименьший блок: меньше не имеет смысла — время блока определяет число запусков ядер.
    MIN_BLOCK_ELEMENTS = 4_000_000
    # Строки таблицы пар шага (одна загрузка int32): точки a и b, их ячейки, нормировки (биты float32),
    # начала строк таблицы общих бит, ячейки числами float32 (биты). Входы ядра слагаемых — только float32:
    # если хоть один вход другого типа, torch приводит типы на каждом элементе, и ядро работает втрое медленнее.
    _POINT_A, _POINT_B, _CELL_A, _CELL_B, _NORM_A, _NORM_B, _ROW_A, _ROW_B, _CELL_AF, _CELL_BF = range(10)

    _KERNEL_CACHE: dict[Any, Any] = {}

    def __init__(
        self,
        engine: Any,
        *,
        block_elements: int,
        table_max_bytes: int,
        table_memory_fraction: float,
        similarity_table: bool = True,
    ) -> None:
        if block_elements <= 0 or table_max_bytes < 0 or not 0.0 < table_memory_fraction <= 1.0:
            raise ValueError("block_elements must be positive, table_max_bytes >= 0, table_memory_fraction in (0, 1]")
        torch = engine.torch_mod
        self._engine = engine
        self._torch = torch
        self._device = engine.torch_device
        self._count = engine.point_count
        self._height, self._width = engine.grid_shape
        self._similarity = engine.similarity
        self._block_elements = int(block_elements)
        # Суммы участков общего тензора равны суммам отдельных тензоров (иначе участки копируются).
        self._merge_ok = True
        self._merge_checked: set[tuple[bool, int, int]] = set()
        # Ширины строк, для которых общая редукция полных порций разошлась с редукциями отдельных порций.
        self._merge_refused: set[tuple[bool, int, int]] = set()
        self._offset_cache: dict[int, tuple[Any, Any, Any]] = {}
        count = self._count
        with torch.no_grad():
            ones = engine.ones
            norm = torch.zeros(count + 1, dtype=torch.float32, device=self._device)
            if self._similarity == "cosine":
                # Тот же множитель n^−½, что в TorchPairEnergyEngine._sim_lambda_from_common (нулевой код — 0).
                norm[:count] = torch.where(
                    ones > 0, torch.rsqrt(torch.clamp(ones, min=1.0)), torch.zeros_like(ones)
                )
            else:
                norm[:count] = ones
            # Нормировка сходства точки; у «точки нет» (номер N) — 0, её сходство с любой точкой — 0.
            self._norm = norm
            self._norm_row = norm[:count].unsqueeze(0)
            # Те же значения на хосте: нормировки точек пары уходят на устройство вместе с таблицей пар шага.
            self._norm_np = norm.cpu().numpy()
        max_bytes = int(table_max_bytes)
        free_bytes = engine.free_memory_bytes()
        if free_bytes is not None:
            max_bytes = min(max_bytes, int(free_bytes * table_memory_fraction))
        # Уровень 1: таблица значений sim_λ (строится при первом расчёте под его λ и η); уровень 2: таблица
        # общих бит; уровень 3: общие биты на лету.
        self._sim_enabled = (
            bool(similarity_table)
            and SimilarityTable.bytes_needed(count) <= max_bytes
            and (count + 1) ** 2 < 2**31
        )
        self._sim: SimilarityTable | None = None
        self._table = (
            CommonBitsTable.build(engine, max_bytes=max_bytes) if max_bytes > 0 and not self._sim_enabled else None
        )
        self._int32_index = self._table is None or self._table.index_dtype == torch.int32
        # Блок слитого ядра держит на устройстве до BLOCK_BYTES_PER_ELEMENT байт на элемент: блоки
        # не должны занимать больше трети памяти, оставшейся после таблицы сходства.
        reserve = SimilarityTable.bytes_needed(count) if self._sim_enabled else 0
        on_the_fly = not self._sim_enabled and self._table is None
        if on_the_fly:
            # Упакованные коды точек и они же по ячейкам сетки (слова кода и нормировка точки ячейки).
            reserve = (count + 1 + self._height * self._width + 1) * (-(-engine.code_length // 64) * 8 + 4)
        # Упакованные коды точек для подсчёта общих бит на лету (строятся при первом расчёте по окну).
        self._packed: Any = None
        element_bytes = self.PACKED_BLOCK_BYTES_PER_ELEMENT if on_the_fly else self.BLOCK_BYTES_PER_ELEMENT
        free_bytes = engine.free_memory_bytes()
        if free_bytes is not None:
            budget = max(0, free_bytes - reserve) // (3 * element_bytes)
            self._block_elements = int(max(self.MIN_BLOCK_ELEMENTS, min(self._block_elements, budget)))
        LOGGER.event(
            "layout.fused.energy.init",
            section=GPU_IMPLEMENTATION,
            data={
                "points": count,
                "grid": (self._height, self._width),
                "similarity": self._similarity,
                "source": "similarity_table" if self._sim_enabled else "common_bits_table" if self._table is not None
                else "common_bits_on_the_fly",
                "block_elements": self._block_elements,
                "note": "слагаемые φ_c, φ_s (§5.5) считает одно ядро на элемент (пара, точка); суммы — та же "
                "редукция float32, что у прохода по всему V или по окну ячеек (§5.12)",
            },
        )

    # ------------------------------------------------------------------ применимость и таблицы сходства
    @classmethod
    def supported(cls, engine: Any) -> bool:
        """Слитое ядро компилируется только для CUDA; координаты ячеек должны помещаться в упаковку."""
        if engine.torch_device.type != "cuda":
            return False
        height, width = engine.grid_shape
        if height >= cls.PACK or width >= cls.PACK or engine.point_count >= 2**31 - 2:
            return False
        try:
            from torch.cuda import jiterator  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return engine.similarity in ("cosine", "jaccard")

    @property
    def uses_table(self) -> bool:
        return self._sim_enabled or self._table is not None

    @property
    def on_the_fly(self) -> bool:
        """Общие биты кодов считаются на лету по словам (таблиц сходства нет)."""
        return not self._sim_enabled and self._table is None

    def window_energies_device(
        self,
        point_a: Any,
        point_b: Any,
        cell_a: Any,
        cell_b: Any,
        grid_points: Any,
        *,
        offsets: tuple["np.ndarray", "np.ndarray"],
        local_radius: float | None,
        kernel: str,
        euclidean: bool,
        distance_eps: float,
        kernel_width: float,
        lambda_threshold: float,
        eta: float | None,
    ) -> tuple[Any, Any]:
        """φ_c, φ_s пар по ячейкам окна окрестности — входы и выходы на устройстве (§5.5.2, §5.9.4, §5.12).

        point_a, point_b — точки пар (N — второй точки нет), cell_a, cell_b — линейные номера их ячеек,
        grid_points — точка каждой ячейки сетки (int32; N — пусто, последний элемент — «за сеткой»),
        offsets — смещения ячеек окна (dy, dx) от ячейки центра пары. Слагаемые — те же, что у pair_energies
        (то же ядро, общие биты на лету); суммы берёт редукция блока строк, поэтому они равны исходному
        проходу с точностью сложения float32, а не побитово.
        """
        if not self.on_the_fly:
            raise RuntimeError("device window energies need the on-the-fly common bits source")
        if kernel not in ("gauss", "inverse"):
            raise ValueError("kernel must be 'gauss' or 'inverse'")
        torch = self._torch
        device = self._device
        rows = int(point_a.numel())
        window = self._sampled_offsets(
            np.asarray(offsets[0], dtype=np.int64).reshape(-1), np.asarray(offsets[1], dtype=np.int64).reshape(-1)
        )
        cells = int(window[0].numel())
        if kernel == "gauss":
            weight, scale = "gauss", 2.0 * kernel_width * kernel_width
        else:
            weight, scale = "inverse", float(distance_eps)
        terms_kernel = self._terms_kernel(
            similarity=self._similarity, hard=eta is None, weight=weight, euclidean=bool(euclidean), window=True
        )
        scalars = {
            "lam": float(lambda_threshold),
            "neg_eta": 0.0 if eta is None else -float(eta),
            "local_radius": 0.0 if local_radius is None else float(local_radius),
            "kernel_scale": float(scale),
        }
        with torch.no_grad():
            # Таблица пар шага — в том же виде, что у pair_energies: точки, упакованные ячейки, нормировки.
            table = torch.empty((10, rows), dtype=torch.int32, device=device)
            floats = table.view(torch.float32)
            for side, (point, cell) in enumerate(((point_a, cell_a), (point_b, cell_b))):
                row_y = torch.div(cell, self._width, rounding_mode="floor")
                packed = row_y * self.PACK + (cell - row_y * self._width)
                table[self._POINT_A + side] = point
                table[self._CELL_A + side] = packed
                floats[self._CELL_AF + side] = packed.to(torch.float32)
                floats[self._NORM_A + side] = torch.index_select(self._norm, 0, point)
            cell_codes = (
                torch.index_select(self._packed_codes(), 0, grid_points),
                torch.index_select(self._norm, 0, grid_points),
            )
            energy_c = torch.empty(rows, dtype=torch.float32, device=device)
            energy_s = torch.empty(rows, dtype=torch.float32, device=device)
            block = max(1, self._block_elements // max(1, cells * 2))
            for first in range(0, rows, block):
                last = min(rows, first + block)
                value_a, value_b, norm_v = self._common_window(table, None, grid_points, first, last, window, cell_codes)
                terms_c, terms_s = terms_kernel(
                    value_a, value_b,
                    floats[self._NORM_A, first:last].unsqueeze(1), floats[self._NORM_B, first:last].unsqueeze(1),
                    norm_v, window[2],
                    floats[self._CELL_AF, first:last].unsqueeze(1), floats[self._CELL_BF, first:last].unsqueeze(1),
                    **scalars,
                )
                del value_a, value_b
                energy_c[first:last] = terms_c.sum(dim=1)
                energy_s[first:last] = terms_s.sum(dim=1)
                del terms_c, terms_s
        return energy_c, energy_s

    def _similarity_table(self, lambda_threshold: float, eta: float | None) -> "SimilarityTable | None":
        """Таблица sim_λ для этих λ и η; None — уровень выключен (не помещается в память)."""
        if not self._sim_enabled:
            return None
        if self._sim is None or not self._sim.matches(lambda_threshold, eta):
            # Прежняя таблица освобождается до построения новой: вдвоём они могут не поместиться.
            self._sim = None
            self._torch.cuda.empty_cache()
            self._sim = SimilarityTable(self._engine, lambda_threshold=lambda_threshold, eta=eta)
        return self._sim

    # ------------------------------------------------------------------ ядра
    @classmethod
    def _terms_kernel(cls, *, similarity: str | None, hard: bool, weight: str, euclidean: bool, window: bool) -> Any:
        """Ядро слагаемых (§5.5): выходы — слагаемое φ_c: s1·w1 + s2·w2 и слагаемое φ_s: s2·w1 + s1·w2.

        Входы — ячейки пары и точки v и либо готовые sim_λ точек пары с точкой v (similarity = None, таблица
        SimilarityTable), либо общие биты и нормировки (тогда sim_λ считает ядро: мера similarity, порог hard).
        В теле нет знака «больше»: разбор шаблона jiterator ищет по нему конец заголовка функции.
        """
        if similarity is None:
            hard = False
        key = ("terms", similarity, hard, weight, euclidean, window)
        cached = cls._KERNEL_CACHE.get(key)
        if cached is not None:
            return cached
        from torch.cuda.jiterator import _create_multi_output_jit_fn

        if similarity is None:
            sim = ""
        elif similarity == "cosine":
            sim = """
  float sim_a = __fmul_rn(__fmul_rn((float)ca, (float)ra), (float)rc);
  float sim_b = __fmul_rn(__fmul_rn((float)cb, (float)rb), (float)rc);
"""
        else:
            sim = """
  float den_a = __fsub_rn(__fadd_rn((float)ra, (float)rc), (float)ca);
  float den_b = __fsub_rn(__fadd_rn((float)rb, (float)rc), (float)cb);
  bool pos_a = (0.0f < den_a) && (0.0f < (float)ra) && (0.0f < (float)rc);
  bool pos_b = (0.0f < den_b) && (0.0f < (float)rb) && (0.0f < (float)rc);
  float sim_a = pos_a ? __fdiv_rn((float)ca, den_a) : 0.0f;
  float sim_b = pos_b ? __fdiv_rn((float)cb, den_b) : 0.0f;
"""
        if similarity is None:
            threshold = """
  float s1 = (float)ca;
  float s2 = (float)cb;
"""
        elif hard:
            threshold = """
  float lam_f = (float)lam;
  bool cut_a = (0.0f < lam_f) ? (sim_a < lam_f) : (sim_a <= 0.0f);
  bool cut_b = (0.0f < lam_f) ? (sim_b < lam_f) : (sim_b <= 0.0f);
  float s1 = cut_a ? 0.0f : sim_a;
  float s2 = cut_b ? 0.0f : sim_b;
"""
        else:
            threshold = """
  float lam_f = (float)lam;
  float neg_eta_f = (float)neg_eta;
  float s1 = __fdiv_rn(sim_a, __fadd_rn(1.0f, ::exp(__fmul_rn(neg_eta_f, __fsub_rn(sim_a, lam_f)))));
  float s2 = __fdiv_rn(sim_b, __fadd_rn(1.0f, ::exp(__fmul_rn(neg_eta_f, __fsub_rn(sim_b, lam_f)))));
"""
        # 1/PACK = 2⁻¹²: умножения и вычитания распаковки точны.
        unpack = """
  float pos_y = floorf((float)pos * 0.000244140625f);
  float pos_x = (float)pos - pos_y * 4096.0f;
  float pay = floorf((float)pa * 0.000244140625f);
  float pax = (float)pa - pay * 4096.0f;
  float pby = floorf((float)pb * 0.000244140625f);
  float pbx = (float)pb - pby * 4096.0f;
"""
        if window:
            # Точка окна стоит в своей ячейке: её позиция — ячейка ⌊c⌋ + смещение (§5.5.2).
            unpack += """
  float py = floorf((pay + pby) * 0.5f) + (pos_y - 2048.0f);
  float px = floorf((pax + pbx) * 0.5f) + (pos_x - 2048.0f);
"""
        else:
            unpack += """
  float py = pos_y;
  float px = pos_x;
"""
        distances = """
  float dy1 = pay - py;
  float dx1 = pax - px;
  float dy2 = pby - py;
  float dx2 = pbx - px;
  float d1 = __fadd_rn(__fmul_rn(dy1, dy1), __fmul_rn(dx1, dx1));
  float d2 = __fadd_rn(__fmul_rn(dy2, dy2), __fmul_rn(dx2, dx2));
  bool valid = (d1 != 0.0f) && (d2 != 0.0f);
"""
        neighbourhood = """
  float aby = pay - pby;
  float abx = pax - pbx;
  float base = ceilf(sqrtf(__fadd_rn(__fmul_rn(aby, aby), __fmul_rn(abx, abx)))) + 1.0f;
  float lr = (float)local_radius;
  float radius = (base < lr) ? lr : base;
  float rsq = __fmul_rn(radius, radius);
  float dyc = py - (pay + pby) * 0.5f;
  float dxc = px - (pax + pbx) * 0.5f;
  valid = valid && (__fadd_rn(__fmul_rn(dyc, dyc), __fmul_rn(dxc, dxc)) <= rsq);
"""
        metric = ("sqrtf(d1)", "sqrtf(d2)") if euclidean else ("d1", "d2")
        if weight == "gauss":
            weights = neighbourhood + """
  float sig2 = __fmul_rn((float)kernel_scale, rsq);
  float w1 = ::exp(__fdiv_rn(-d1, sig2));
  float w2 = ::exp(__fdiv_rn(-d2, sig2));
"""
        elif weight == "inverse":
            weights = neighbourhood + f"""
  float w1 = __fdiv_rn(1.0f, __fadd_rn({metric[0]}, (float)kernel_scale));
  float w2 = __fdiv_rn(1.0f, __fadd_rn({metric[1]}, (float)kernel_scale));
"""
        else:
            weights = f"""
  float w1 = {metric[0]};
  float w2 = {metric[1]};
"""
        tail = """
  float a1 = valid ? s1 : 0.0f;
  float a2 = valid ? s2 : 0.0f;
  tc = __fadd_rn(__fmul_rn(a1, w1), __fmul_rn(a2, w2));
  ts = __fadd_rn(__fmul_rn(a2, w1), __fmul_rn(a1, w2));
}
"""
        inputs = "T ca, T cb, T pos, T pa, T pb, " if similarity is None else "T ca, T cb, T ra, T rb, T rc, T pos, T pa, T pb, "
        source = (
            "template <typename T> void damp_terms(" + inputs
            + "T lam, T neg_eta, T local_radius, T kernel_scale, T& tc, T& ts) {"
            + sim
            + threshold
            + unpack
            + distances
            + weights
            + tail
        )
        kernel = _Kernel(
            _create_multi_output_jit_fn(source, num_outputs=2, lam=0.0, neg_eta=0.0, local_radius=0.0, kernel_scale=1.0)
        )
        cls._KERNEL_CACHE[key] = kernel
        return kernel

    @classmethod
    def _cell_kernel(cls) -> Any:
        """Ячейка окна пары: линейный номер ячейки сетки или номер «за сеткой» (height·width)."""
        cached = cls._KERNEL_CACHE.get("cell")
        if cached is None:
            from torch.cuda.jiterator import _create_jit_fn

            source = """
template <typename T> T damp_cell(T pa, T pb, T oy, T ox, T height, T width) {
  T pay = pa / 4096;
  T pax = pa - pay * 4096;
  T pby = pb / 4096;
  T pbx = pb - pby * 4096;
  T y = (pay + pby) / 2 + oy;
  T x = (pax + pbx) / 2 + ox;
  bool inside = (0 <= y) && (y < height) && (0 <= x) && (x < width);
  return inside ? y * width + x : height * width;
}
"""
            cached = _Kernel(_create_jit_fn(source, height=1, width=1))
            cls._KERNEL_CACHE["cell"] = cached
        return cached

    @classmethod
    def _popcount_kernel(cls) -> Any:
        """Число общих бит POPCOUNT_WORDS слов кода: Σ popcount(a_j & v_j) (§2.2.4); все входы int64."""
        cached = cls._KERNEL_CACHE.get("popcount")
        if cached is None:
            from torch.cuda.jiterator import _create_jit_fn

            source = """
template <typename T> T damp_popcount(T a0, T a1, T a2, T a3, T v0, T v1, T v2, T v3) {
  return __popcll(a0 & v0) + __popcll(a1 & v1) + __popcll(a2 & v2) + __popcll(a3 & v3);
}
"""
            cached = _Kernel(_create_jit_fn(source))
            cls._KERNEL_CACHE["popcount"] = cached
        return cached

    def _packed_codes(self) -> Any:
        """Коды точек 64-битными словами: (N + 1) × слова int64; строка N — нули («точки нет»).

        Число слов дополнено нулевыми до кратного POPCOUNT_WORDS.
        """
        if self._packed is None:
            torch = self._torch
            words = self._engine.packed_codes()
            width = -(-int(words.shape[1]) // self.POPCOUNT_WORDS) * self.POPCOUNT_WORDS
            packed = torch.zeros((self._count + 1, width), dtype=torch.int64, device=self._device)
            packed[: self._count, : int(words.shape[1])] = words
            self._packed = packed
            LOGGER.event(
                "layout.fused.packed_codes",
                section=OPTIM_SIM_MATRIX,
                data={
                    "points": self._count,
                    "words": width,
                    "memory_mb": (self._count + 1) * width * 8 / 1e6,
                    "note": "таблицы сходства не помещаются в память: общие биты кодов (§2.2.4) считаются на лету "
                    "как popcount(a & v) по упакованным словам (§5.9.5, §5.12)",
                },
            )
        return self._packed

    def _window_offsets(self, reach: int) -> tuple[Any, Any, Any]:
        """Смещения ячеек окна (2R+1)² построчно: dy, dx (int32, 1×K) и упакованные (float32, 1×K)."""
        cached = self._offset_cache.get(reach)
        if cached is None:
            torch = self._torch
            side = 2 * reach + 1
            offsets = torch.arange(-reach, reach + 1, device=self._device, dtype=torch.int32)
            dy = offsets.repeat_interleave(side).unsqueeze(0)
            dx = offsets.repeat(side).unsqueeze(0)
            packed = ((dy + self.OFFSET_BIAS) * self.PACK + (dx + self.OFFSET_BIAS)).to(torch.float32)
            cached = (dy, dx, packed)
            if len(self._offset_cache) > 64:
                self._offset_cache.clear()
            self._offset_cache[reach] = cached
        return cached

    # ------------------------------------------------------------------ расчёт
    def pair_energies(
        self,
        idx_a: "np.ndarray",
        idx_b: "np.ndarray",
        pos_a: "np.ndarray",
        pos_b: "np.ndarray",
        *,
        positions: "np.ndarray",
        grid: "np.ndarray",
        neighbourhood: bool,
        local_radius: float | None,
        kernel: str,
        euclidean: bool,
        distance_eps: float,
        kernel_width: float,
        lambda_threshold: float,
        eta: float | None,
        need_swapped: bool = True,
        eval_indices: Sequence[int] | None = None,
        sample_offsets: tuple[Any, Any] | None = None,
    ) -> tuple["np.ndarray", "np.ndarray", str] | None:
        """(φ_c, φ_s, вид расчёта) — те же числа, что у TorchPairEnergyEngine.pair_energies; None — неприменимо.

        eval_indices — подмножество B точек (§5.9.4): расчёт «по всем точкам», но столбцы — только точки B.
        sample_offsets — выборка ячеек окна (смещения dy, dx от ячейки центра пары): расчёт по окну ячеек,
        но ячейки — только выбранные (§5.9.4).

        positions — ячейки (y, x) всех точек N×2, grid — сетка с номером точки или −1 (массивы хоста): нужная
        расчёту часть расстановки уходит на устройство вместе с парами шага одной загрузкой — ячейки точек
        для расчёта по всему V, сетка для расчёта по окну ячеек. Вид расчёта («dense» — по всем точкам V,
        «window» — по окну ячеек) нужен сверке с исходным проходом.
        """
        torch = self._torch
        count = self._count
        pairs = int(len(idx_a))
        if pairs == 0:
            return None
        if neighbourhood and kernel not in ("gauss", "inverse"):
            return None
        idx_a = np.asarray(idx_a, dtype=np.int64)
        idx_b = np.asarray(idx_b, dtype=np.int64)
        pos_a = np.asarray(pos_a, dtype=np.int64)
        pos_b = np.asarray(pos_b, dtype=np.int64)
        engine = self._engine
        subset = None if eval_indices is None else np.asarray(eval_indices, dtype=np.int64)
        sample_dy = sample_dx = None
        if neighbourhood and sample_offsets is not None:
            if subset is not None:
                return None
            sample_dy = np.asarray(sample_offsets[0], dtype=np.int64).reshape(-1)
            sample_dx = np.asarray(sample_offsets[1], dtype=np.int64).reshape(-1)
            if sample_dy.size == 0 or sample_dy.size != sample_dx.size:
                return None
            if max(int(np.abs(sample_dy).max()), int(np.abs(sample_dx).max())) >= self.OFFSET_BIAS:
                return None
        if subset is not None and subset.size == 0:
            return None
        reach = 0
        dense = True
        if neighbourhood:
            # Радиус окрестности r ≥ ‖p1 − p2‖ (+1) и полуширина окна — как в neighbourhood_geometry (§5.5.2).
            reach = engine.window_reach(pos_a, pos_b, local_radius)
            if sample_dy is not None:
                dense = False
            else:
                dense = subset is not None or (2 * reach + 1) ** 2 >= count
                if not dense and reach >= self.OFFSET_BIAS:
                    return None
        if dense and subset is None and not self._sim_enabled and self._table is None and not engine.has_float_codes:
            # По всем точкам V общие биты дают произведения кодов float32 — их на устройстве нет.
            return None
        if dense:
            width = count if subset is None else int(subset.size)
            chunk = engine.dense_chunk_rows() if subset is None else max(1, engine.CHUNK_ELEMENTS // width)
        else:
            width = (2 * reach + 1) ** 2 if sample_dy is None else int(sample_dy.size)
            chunk = engine.local_chunk_rows()
        align = self.ROW_ALIGN
        full = pairs // chunk
        tail = pairs - full * chunk
        stride = -(-chunk // align) * align
        # Место пары в общем тензоре: порция j занимает строки с j·stride (кратно ROW_ALIGN); остаток — после.
        rows_total = full * stride + tail
        rows_padded = -(-rows_total // align) * align
        # Одна загрузка int32: таблица пар шага и за ней расстановка — упакованные ячейки точек (биты float32)
        # для расчёта по всему V или сетка с номером точки (N — пусто, последний элемент — «за сеткой»).
        table_size = 10 * rows_padded
        extra_size = width if dense else self._height * self._width + 1
        buffer = np.zeros(table_size + extra_size, dtype=np.int32)
        if dense and subset is None:
            buffer[table_size:].view(np.float32)[:] = positions[:, 0] * self.PACK + positions[:, 1]
        elif dense:
            buffer[table_size:].view(np.float32)[:] = positions[subset, 0] * self.PACK + positions[subset, 1]
        else:
            cells_grid = buffer[table_size:-1].reshape(self._height, self._width)
            cells_grid[:] = grid
            cells_grid[cells_grid < 0] = count
            buffer[-1] = count
        table = buffer[:table_size].reshape(10, rows_padded)
        norms = table[self._NORM_A : self._NORM_B + 1].view(np.float32)
        cells = table[self._CELL_AF : self._CELL_BF + 1].view(np.float32)
        point_b = np.where(idx_b < 0, count, idx_b)
        cell_a = pos_a[:, 0] * self.PACK + pos_a[:, 1]
        cell_b = pos_b[:, 0] * self.PACK + pos_b[:, 1]
        if stride == chunk:
            slot = None
            if rows_padded != pairs:
                # Пустые пары в конце: точек нет, слагаемые — нули.
                table[self._POINT_A : self._POINT_B + 1, pairs:] = count
            table[self._POINT_A, :pairs] = idx_a
            table[self._POINT_B, :pairs] = point_b
            table[self._CELL_A, :pairs] = cell_a
            table[self._CELL_B, :pairs] = cell_b
            cells[0, :pairs] = cell_a
            cells[1, :pairs] = cell_b
            norms[0, :pairs] = self._norm_np[idx_a]
            norms[1, :pairs] = self._norm_np[point_b]
        else:
            index = np.arange(pairs, dtype=np.int64)
            slot = (index // chunk) * stride + index % chunk
            # Пустые пары между порциями: точек нет, слагаемые — нули.
            table[self._POINT_A : self._POINT_B + 1] = count
            table[self._POINT_A, slot] = idx_a
            table[self._POINT_B, slot] = point_b
            table[self._CELL_A, slot] = cell_a
            table[self._CELL_B, slot] = cell_b
            cells[0, slot] = cell_a
            cells[1, slot] = cell_b
            norms[0, slot] = self._norm_np[idx_a]
            norms[1, slot] = self._norm_np[point_b]
        # По подмножеству B общие биты считаются на лету по словам кодов (столбцов мало, таблицы не нужны).
        sim_table = self._similarity_table(lambda_threshold, eta) if subset is None else None
        offsets_np = None
        if not dense and sim_table is not None:
            # Начало строки таблицы sim_λ: точка · (N + 1); (N + 1)² помещается в int32.
            np.multiply(table[self._POINT_A : self._POINT_B + 1], sim_table.stride, out=table[self._ROW_A : self._ROW_B + 1])
        elif not dense and self._table is not None:
            offsets_np = self._table.row_offsets(table[self._POINT_A : self._POINT_B + 1])
            if self._int32_index:
                table[self._ROW_A : self._ROW_B + 1] = offsets_np
        hard = eta is None
        if not neighbourhood:
            weight, scale = "distance", 0.0
        elif kernel == "gauss":
            weight, scale = "gauss", 2.0 * kernel_width * kernel_width
        else:
            weight, scale = "inverse", float(distance_eps)
        terms_kernel = self._terms_kernel(
            similarity=None if sim_table is not None else self._similarity,
            hard=hard, weight=weight, euclidean=bool(euclidean), window=not dense,
        )
        scalars = {
            "lam": float(lambda_threshold),
            "neg_eta": 0.0 if eta is None else -float(eta),
            "local_radius": 0.0 if local_radius is None else float(local_radius),
            "kernel_scale": float(scale),
        }
        # Исходный проход по всему V считает φ_s всегда; по окну ячеек — только когда она нужна.
        halves = 2 if (need_swapped or dense) else 1
        merge_key = (dense, width, chunk)
        merge = (
            self._merge_ok and chunk >= self.MERGE_MIN_ROWS and full > 0 and merge_key not in self._merge_refused
        )
        check = self._merge_ok and merge_key not in self._merge_checked
        # Блок — целое число порций: общие биты блока берутся одной выборкой, слагаемые считает один запуск ядра.
        block_pieces = max(1, self._block_elements // max(1, stride * width * 2))
        # Строки [0, merged_end) — полные порции; при объединении их суммы берутся одной редукцией блока.
        merged_end = (full - 1) * stride + chunk if merge else 0
        take = [np.empty(rows_padded, dtype=np.int64) for _ in range(halves)]
        cursor = 0
        outputs: list[Any] = []
        with torch.no_grad():
            device = self._device
            device_buffer = torch.as_tensor(buffer, device=device)
            rows_table = device_buffer[:table_size].view(10, rows_padded)
            rows_norm = rows_table.view(torch.float32)
            if offsets_np is None or self._int32_index:
                row_offsets = rows_table[self._ROW_A : self._ROW_B + 1]
            else:
                row_offsets = torch.as_tensor(offsets_np, device=device)
            window: tuple[Any, Any, Any] | None = None
            subset_words = subset_norm = None
            cell_codes: tuple[Any, Any] | None = None
            if dense:
                grid_points = None
                position = device_buffer[table_size:].view(torch.float32).unsqueeze(0)
                if subset is not None:
                    subset_device = torch.as_tensor(subset, device=device)
                    subset_words = torch.index_select(self._packed_codes(), 0, subset_device)
                    subset_norm = torch.index_select(self._norm, 0, subset_device).unsqueeze(0)
            else:
                grid_points = device_buffer[table_size:]
                window = self._window_offsets(reach) if sample_dy is None else self._sampled_offsets(sample_dy, sample_dx)
                position = window[2]
                if sim_table is None and self._table is None:
                    # Общие биты на лету: слова кода и нормировка точки каждой ячейки сетки берутся одной
                    # выборкой на шаг. Окно пары читает их по ячейкам: соседние ячейки окна — соседние строки
                    # памяти (выборка слов по номерам точек окна разбросана по всей таблице кодов).
                    cell_codes = (
                        torch.index_select(self._packed_codes(), 0, grid_points),
                        torch.index_select(self._norm, 0, grid_points),
                    )
            for first in range(0, rows_padded, block_pieces * stride):
                last = min(rows_padded, first + block_pieces * stride)
                cell_a = rows_norm[self._CELL_AF, first:last].unsqueeze(1)
                cell_b = rows_norm[self._CELL_BF, first:last].unsqueeze(1)
                if sim_table is not None:
                    # Готовые sim_λ из таблицы: ядру остаются окрестность, веса и произведения.
                    if dense:
                        value_a, value_b = self._similarity_dense(sim_table, rows_table, first, last)
                    else:
                        value_a, value_b = self._similarity_window(
                            sim_table, rows_table, row_offsets, grid_points, first, last, window
                        )
                    terms = terms_kernel(value_a, value_b, position, cell_a, cell_b, **scalars)[:halves]
                else:
                    if dense and subset is None:
                        value_a, value_b = self._common_dense(rows_table, first, last)
                        norm_v = self._norm_row
                    elif dense:
                        value_a, value_b = self._common_subset(rows_table, first, last, subset_words)
                        norm_v = subset_norm
                    else:
                        value_a, value_b, norm_v = self._common_window(
                            rows_table, row_offsets, grid_points, first, last, window, cell_codes
                        )
                    terms = terms_kernel(
                        value_a, value_b,
                        rows_norm[self._NORM_A, first:last].unsqueeze(1), rows_norm[self._NORM_B, first:last].unsqueeze(1),
                        norm_v, position, cell_a, cell_b,
                        **scalars,
                    )[:halves]
                del value_a, value_b
                # Порции блока: (начало, конец) в строках блока; объединяемые — полные порции до merged_end.
                pieces = [
                    (start - first, min(start + chunk, rows_total, last) - first)
                    for start in range(first, min(last, rows_total), stride)
                ]
                merged_pieces = [piece for piece in pieces if piece[1] + first <= merged_end]
                separate_pieces = pieces[len(merged_pieces) :]
                for half, half_terms in enumerate(terms):
                    if merged_pieces:
                        # Одна редукция на все полные порции блока.
                        block_sums = half_terms.sum(dim=1)
                        if check and not self._sums_match(half_terms, block_sums, merged_pieces, whole=True):
                            self._merge_refused.add(merge_key)
                            return self._without_merge(
                                idx_a, idx_b, pos_a, pos_b, width=width, chunk=chunk, positions=positions, grid=grid,
                                neighbourhood=neighbourhood, local_radius=local_radius, kernel=kernel,
                                euclidean=euclidean, distance_eps=distance_eps, kernel_width=kernel_width,
                                lambda_threshold=lambda_threshold, eta=eta, need_swapped=need_swapped,
                                eval_indices=eval_indices, sample_offsets=sample_offsets,
                            )
                        outputs.append(block_sums)
                        merged_rows = merged_pieces[-1][1]
                        take[half][first : first + merged_rows] = cursor + np.arange(merged_rows, dtype=np.int64)
                        cursor += int(half_terms.shape[0])
                    for lo, hi in separate_pieces:
                        piece = half_terms[lo:hi]
                        if self._merge_ok:
                            # Участок общего тензора: та же форма и то же выравнивание начала, что у отдельного.
                            sums = piece.sum(dim=1)
                            if check and not self._sums_match(half_terms, sums, [(lo, hi)], whole=False):
                                self._merge_ok = False
                                return self._without_merge(
                                    idx_a, idx_b, pos_a, pos_b, width=width, chunk=chunk, positions=positions, grid=grid,
                                    neighbourhood=neighbourhood, local_radius=local_radius, kernel=kernel,
                                    euclidean=euclidean, distance_eps=distance_eps, kernel_width=kernel_width,
                                    lambda_threshold=lambda_threshold, eta=eta, need_swapped=need_swapped,
                                    eval_indices=eval_indices, sample_offsets=sample_offsets,
                                )
                        else:
                            # Отдельный тензор порции — как в исходном проходе.
                            sums = piece.clone().sum(dim=1)
                        outputs.append(sums)
                        take[half][first + lo : first + hi] = cursor + np.arange(hi - lo, dtype=np.int64)
                        cursor += hi - lo
                del terms
            if check:
                self._merge_checked.add(merge_key)
            values = (torch.cat(outputs) if len(outputs) > 1 else outputs[0]).cpu().numpy()
        # Пара берёт сумму своей строки; суммы пустых пар отбрасываются.
        place = slice(0, pairs) if slot is None else slot
        result_c = values[take[0][place]]
        if halves == 2:
            result_s = values[take[1][place]]
        else:
            # Путь по окну ячеек при need_swapped = False возвращает нули, как исходный проход.
            result_s = np.zeros(pairs, dtype=np.float32)
        return result_c, result_s, "dense" if dense else "window"

    def _without_merge(self, idx_a: Any, idx_b: Any, pos_a: Any, pos_b: Any, *, width: int, chunk: int, **arguments: Any) -> Any:
        """Суммы общего тензора разошлись с суммами отдельных порций: расчёт повторяется без объединения.

        Для этой ширины строки полные порции дальше суммируются по отдельности (как в исходном проходе);
        если разошлась сумма участка общего тензора, участки дальше копируются в отдельные тензоры.
        """
        LOGGER.event(
            "layout.fused.energy.merge_disabled",
            section=GPU_IMPLEMENTATION,
            data={
                "width": width,
                "chunk_rows": chunk,
                "separate_tensors": not self._merge_ok,
                "note": "сумма строки в общем тензоре отличается от суммы в тензоре порции: "
                "порции этой ширины суммируются по отдельности, как в исходном проходе (§5.12)",
            },
            force=True,
        )
        return self.pair_energies(idx_a, idx_b, pos_a, pos_b, **arguments)

    def _similarity_dense(self, sim_table: SimilarityTable, rows_table: Any, first: int, last: int) -> tuple[Any, Any]:
        """sim_λ точек a и b строк блока со всеми точками V (§5.4): строки таблицы."""
        rows = last - first
        values = sim_table.rows(rows_table[self._POINT_A : self._POINT_B + 1, first:last].reshape(-1))
        return values[:rows], values[rows:]

    def _sampled_offsets(self, sample_dy: "np.ndarray", sample_dx: "np.ndarray") -> tuple[Any, Any, Any]:
        """Смещения выборки ячеек окна (§5.9.4) в том же виде, что _window_offsets: dy, dx и упакованные."""
        torch = self._torch
        dy = torch.as_tensor(sample_dy.astype(np.int32), device=self._device).unsqueeze(0)
        dx = torch.as_tensor(sample_dx.astype(np.int32), device=self._device).unsqueeze(0)
        packed = ((dy + self.OFFSET_BIAS) * self.PACK + (dx + self.OFFSET_BIAS)).to(torch.float32)
        return dy, dx, packed

    def _popcount_common(self, pair_words: Any, other_words: tuple[Any, ...]) -> Any:
        """Общие биты кодов (§2.2.4): Σ popcount(a_j & v_j) по словам; pair_words — [строки, слова],
        other_words — по тензору на слово, формы транслируются со столбцом [строки, 1]. float32."""
        popcount = self._popcount_kernel()
        step = self.POPCOUNT_WORDS
        count = None
        for word in range(0, len(other_words), step):
            part = popcount(
                *(pair_words[:, word + offset : word + offset + 1] for offset in range(step)),
                *(other_words[word + offset] for offset in range(step)),
            )
            count = part if count is None else count.add_(part)
        return count.to(self._torch.float32)

    def _common_subset(self, rows_table: Any, first: int, last: int, subset_words: Any) -> tuple[Any, Any]:
        """Общие биты точек a и b строк блока с точками подмножества B (§5.9.4): слова кодов, ядро popcount."""
        torch = self._torch
        rows = last - first
        width = int(subset_words.shape[1])
        pair_words = torch.index_select(
            self._packed_codes(), 0, rows_table[self._POINT_A : self._POINT_B + 1, first:last].reshape(-1)
        ).view(2, rows, width)
        columns = tuple(subset_words[:, word].unsqueeze(0) for word in range(width))
        return self._popcount_common(pair_words[0], columns), self._popcount_common(pair_words[1], columns)

    def _window_points(
        self, rows_table: Any, grid_points: Any, first: int, last: int, window: tuple[Any, Any, Any]
    ) -> Any:
        """Точки ячеек окна пар блока (N — пустая ячейка или ячейка за сеткой): [строки · K] (§5.5.2)."""
        return self._torch.index_select(grid_points, 0, self._window_cells(rows_table, first, last, window))

    def _window_cells(self, rows_table: Any, first: int, last: int, window: tuple[Any, Any, Any]) -> Any:
        """Линейные номера ячеек окна пар блока (height·width — ячейка за сеткой): [строки · K]."""
        dy, dx, _ = window
        return self._cell_kernel()(
            rows_table[self._CELL_A, first:last].unsqueeze(1), rows_table[self._CELL_B, first:last].unsqueeze(1),
            dy, dx, height=self._height, width=self._width,
        ).view(-1)

    def _similarity_window(
        self,
        sim_table: SimilarityTable,
        rows_table: Any,
        row_offsets: Any,
        grid_points: Any,
        first: int,
        last: int,
        window: tuple[Any, Any, Any],
    ) -> tuple[Any, Any]:
        """sim_λ точек a и b с точками окна ячеек (§5.5.2): элементы таблицы."""
        torch = self._torch
        rows = last - first
        point = self._window_points(rows_table, grid_points, first, last, window).view(1, rows, -1)
        # Линейный индекс таблицы: начало строки точки пары + точка окна — для a и b сразу.
        index = point + row_offsets[:, first:last].unsqueeze(2)
        values = torch.index_select(sim_table.flat, 0, index.view(-1)).view(2, rows, -1)
        return values[0], values[1]

    def _common_dense(self, rows_table: Any, first: int, last: int) -> tuple[Any, Any]:
        """Общие биты точек a и b строк блока со всеми точками V (§2.2.4): таблица или произведение кодов."""
        torch = self._torch
        rows = last - first
        points = rows_table[self._POINT_A : self._POINT_B + 1, first:last].reshape(-1)
        if self._table is not None:
            # Числа общих бит из таблицы (uint8) — во float32: целые, преобразование точное.
            common = self._table.rows(points).to(torch.float32)
        else:
            codes = self._engine.codes
            # «Точки нет» (номер N) берёт код точки N − 1: её нормировка 0, сходство обнуляется в ядре.
            common = codes[torch.clamp(points, max=self._count - 1)] @ codes.T
        return common[:rows], common[rows:]

    def _common_window(
        self, rows_table: Any, row_offsets: Any, grid_points: Any, first: int, last: int,
        window: tuple[Any, Any, Any],
        cell_codes: tuple[Any, Any] | None = None,
    ) -> tuple[Any, Any, Any]:
        """Общие биты точек a и b с точками окна ячеек и нормировки точек окна (§5.5.2, §5.12).

        cell_codes — слова кода и нормировка точки каждой ячейки сетки (общие биты на лету).
        """
        torch = self._torch
        rows = last - first
        if cell_codes is not None and self._table is None:
            # Слова кода точки окна — по номеру её ячейки (у пустой ячейки и ячейки за сеткой — нули).
            cell_words, cell_norm = cell_codes
            lin = self._window_cells(rows_table, first, last, window)
            cells = int(lin.numel()) // rows
            width = int(cell_words.shape[1])
            norm_v = torch.index_select(cell_norm, 0, lin).view(rows, cells)
            window_words = torch.index_select(cell_words, 0, lin).view(rows, cells, width)
            pair_words = torch.index_select(
                self._packed_codes(), 0, rows_table[self._POINT_A : self._POINT_B + 1, first:last].reshape(-1)
            ).view(2, rows, width)
            columns = tuple(window_words[:, :, word] for word in range(width))
            return (
                self._popcount_common(pair_words[0], columns), self._popcount_common(pair_words[1], columns), norm_v
            )
        point = self._window_points(rows_table, grid_points, first, last, window)
        cells = int(point.numel()) // rows
        norm_v = torch.index_select(self._norm, 0, point).view(rows, cells)
        if self._table is not None:
            column = self._table.columns(point).view(1, rows, cells)
            # Линейный индекс таблицы: начало строки точки пары + столбец точки окна — для a и b сразу.
            index = column + row_offsets[:, first:last].unsqueeze(2)
            common = torch.index_select(self._table.flat, 0, index.view(-1)).to(torch.float32).view(2, rows, cells)
            return common[0], common[1], norm_v
        # Общие биты на лету: слова кода каждой точки окна берутся одной выборкой (у пустой ячейки — нули)
        # и сравниваются со словами кодов точек a и b ядром popcount, по POPCOUNT_WORDS слов за запуск.
        packed = self._packed_codes()
        width = int(packed.shape[1])
        window_words = torch.index_select(packed, 0, point).view(rows, cells, width)
        pair_words = torch.index_select(
            packed, 0, rows_table[self._POINT_A : self._POINT_B + 1, first:last].reshape(-1)
        ).view(2, rows, width)
        columns = tuple(window_words[:, :, word] for word in range(width))
        return self._popcount_common(pair_words[0], columns), self._popcount_common(pair_words[1], columns), norm_v

    def _sums_match(self, terms: Any, sums: Any, pieces: list[tuple[int, int]], *, whole: bool) -> bool:
        """Суммы строк общего тензора равны суммам отдельных тензоров порций (как в исходном проходе).

        whole = True: sums — редукция всего тензора блока; whole = False: sums — редукция участка одной порции.
        """
        torch = self._torch
        for lo, hi in pieces:
            separate = terms[lo:hi].clone().sum(dim=1)
            if not bool(torch.equal(separate, sums[lo:hi] if whole else sums)):
                return False
        return True
