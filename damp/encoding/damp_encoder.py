from __future__ import annotations

from dataclasses import dataclass
import colorsys
import hashlib
import math
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np

from damp.article_refs import (
    COLOR_ENCODING,
    CONCATENATION,
    CYCLIC_COORDS,
    INTEGER_ENCODING,
    LAID_OUT_STRUCTURE,
    LONG_RANGE_IMPORTANCE,
    MULTI_DIM_CODES,
    ONE_DIM_CODES,
    OUTSIDE_ARTICLE,
    SHORT_LONG_ORDER,
    SPARSE_BIT_VECTORS,
    WIDE_DETECTORS,
)
from damp.encoding.bitarray import BitArray
from damp.logging import LOGGER

# Сдвиг каждого слоя на свою долю шага: слои не выстраиваются в начале
# координат, и плотность кода не зависит от положения стимула (Angle-Encoder).
GOLDEN_RATIO_FRACTION = (math.sqrt(5.0) - 1.0) / 2.0
# §2: «bit vectors work well when no more than 25% of bits are set to 1».
_MAX_SPARSE_DENSITY = 0.25


@dataclass(frozen=True)
class Detectors:
    """Один слой широких детекторов: число и перекрытие соседей (§4.1)."""

    count: int
    overlap: float

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("detector count must be positive")
        if not (0.0 <= self.overlap < 1.0):
            raise ValueError("overlap must be in [0.0, 1.0)")
        LOGGER.event(
            "detectors.init",
            section=WIDE_DETECTORS,
            data={"count": self.count, "overlap": self.overlap},
        )


def geometric_detector_layers(
    *,
    layers_count: int,
    widest_detectors: int,
    narrowest_detectors: int,
    overlap: float,
) -> tuple[Detectors, ...]:
    """Слои от самого широкого к самому узкому с геометрическим ростом (§4.2, §4.5).

    Число детекторов растёт геометрически от ``widest_detectors`` до
    ``narrowest_detectors``. Округление может дать соседним слоям одинаковое
    число, поэтому каждый слой получает хотя бы на один детектор больше
    предыдущего, а затем ограничивается так, чтобы самый узкий слой сохранил
    точное число. Так устроен Angle-Encoder.
    """
    if layers_count < 2:
        raise ValueError("at least two detector layers are required")
    if widest_detectors <= 0 or narrowest_detectors <= 0:
        raise ValueError("detector counts must be positive")
    last = layers_count - 1
    if narrowest_detectors - widest_detectors < last:
        raise ValueError(
            f"narrowest layer must have at least {last} more detectors than the widest"
        )
    ratio = (narrowest_detectors / widest_detectors) ** (1.0 / last)
    counts: list[int] = []
    for layer in range(layers_count):
        if layer == 0:
            counts.append(int(widest_detectors))
        elif layer == last:
            counts.append(int(narrowest_detectors))
        else:
            count = int(math.floor(widest_detectors * ratio**layer + 0.5))
            counts.append(max(count, counts[layer - 1] + 1))
    for layer in range(last - 1, 0, -1):
        counts[layer] = min(counts[layer], counts[layer + 1] - 1)
    LOGGER.event(
        "detectors.geometric_layers",
        section=SHORT_LONG_ORDER,
        data={
            "layers_count": layers_count,
            "widest": widest_detectors,
            "narrowest": narrowest_detectors,
            "overlap": overlap,
            "counts": counts,
            "ratio": ratio,
        },
    )
    return tuple(Detectors(count, overlap) for count in counts)


class Dimension:
    """Одна координата стимула со своим сегментом кода (§2.2.3, §4.2)."""

    def __init__(
        self,
        title: str,
        size: Tuple[float, float],
        detector_layers: Sequence[Detectors],
        closed: bool,
        code_length: int,
        edge_detectors: bool = False,
    ) -> None:
        self.title = str(title)
        if len(size) != 2:
            raise ValueError("size must contain exactly two values")
        self.min_value = float(size[0])
        self.max_value = float(size[1])
        if self.max_value <= self.min_value:
            raise ValueError("size must be an increasing range")
        if not detector_layers:
            raise ValueError("detector_layers must be non-empty")
        if code_length <= 0:
            raise ValueError("code_length must be positive")
        if code_length < len(detector_layers):
            raise ValueError("code_length must give every detector layer at least one bit (colour band, §3.2)")
        self.detector_layers = list(detector_layers)
        self.closed = bool(closed)
        # Только для отрезка: True — детекторы заходят за концы (схема Angle-Encoder), False — ровно count
        # детекторов слоя внутри отрезка (§4.2.1 не требует детекторов за краями).
        self.edge_detectors = bool(edge_detectors) and not self.closed
        self.code_length = int(code_length)
        self.period = self.max_value - self.min_value
        layers = [(layer.count, layer.overlap) for layer in self.detector_layers]
        LOGGER.event(
            "dimension.init",
            section=ONE_DIM_CODES,
            data={
                "title": self.title,
                "min_value": self.min_value,
                "max_value": self.max_value,
                "closed": self.closed,
                "period": self.period,
                "code_length": self.code_length,
                "edge_detectors": self.edge_detectors,
                "layer_specs": layers,
            },
        )
        if self.closed:
            LOGGER.event(
                "dimension.closed",
                section=CYCLIC_COORDS,
                data={"title": self.title, "period": self.period},
            )

    def layer_band(self, layer_index: int) -> tuple[int, int]:
        """Цветовая полоса слоя в сегменте: начало и длина (§3.2, §4.4.2, §5.14).

        Цвет бита — его важность при colour merge (§3.2). Широкие детекторы отвечают за дальний
        порядок и красную часть «спектра», узкие — за ближний и фиолетовую (§3.3.1, §4.2.1, §4.4.2);
        младший бит красный, старший фиолетовый (§5.14). Поэтому слой с индексом 0 (самый широкий)
        получает младшие биты сегмента, самый узкий — старшие. Полосы равной длины: при этом
        ожидаемое число случайных совпадений бит Σ 1/b_l минимально при заданной длине сегмента.
        """
        layers = len(self.detector_layers)
        if not 0 <= layer_index < layers:
            raise ValueError("layer_index out of range")
        base, remainder = divmod(self.code_length, layers)
        # Остаток отдаётся самым узким слоям: у них больше всего детекторов.
        extra_from = layers - remainder
        start = layer_index * base + max(0, layer_index - extra_from)
        length = base + (1 if layer_index >= extra_from else 0)
        return start, length


class ClosedDimension(Dimension):
    """Циклическая координата: соседние значения через границу периода близки (§4.4)."""

    def __init__(
        self,
        title: str,
        size: Tuple[float, float],
        detector_layers: Sequence[Detectors],
        *,
        code_length: int,
    ) -> None:
        super().__init__(
            title=title,
            size=size,
            detector_layers=detector_layers,
            closed=True,
            code_length=code_length,
        )


class OpenedDimension(Dimension):
    """Отрезок (§4.2). edge_detectors=True — детекторы заходят за концы, чтобы края покрывались как
    середина (схема Angle-Encoder); False — в слое ровно count детекторов внутри отрезка."""

    def __init__(
        self,
        title: str,
        size: Tuple[float, float],
        detector_layers: Sequence[Detectors],
        *,
        code_length: int,
        edge_detectors: bool = True,
    ) -> None:
        super().__init__(
            title=title,
            size=size,
            detector_layers=detector_layers,
            closed=False,
            code_length=code_length,
            edge_detectors=edge_detectors,
        )


@dataclass(frozen=True)
class DetectorWindow:
    """Рецептивное поле одного широкого детектора и его случайный бит (§4.1, §4.2.1)."""

    detector_index: int
    dimension_index: int
    layer_index: int
    center: float
    half_width: float
    closed: bool
    span_min: float
    span_max: float
    period: float
    bit: int
    local_bit: int

    def is_active(self, value: float) -> bool:
        if self.closed:
            if self.half_width >= self.period / 2.0:
                return True
            diff = abs(value - self.center) % self.period
            return min(diff, self.period - diff) <= self.half_width
        return abs(value - self.center) <= self.half_width


@dataclass(frozen=True)
class _DimensionIndex:
    """Векторизованные параметры детекторов одной координаты."""

    offset: int
    code_length: int
    closed: bool
    period: float
    min_value: float
    max_value: float
    centers: "np.ndarray"
    half_widths: "np.ndarray"
    bits: "np.ndarray"

    def active_mask(self, values: "np.ndarray") -> "np.ndarray":
        """Маска (значения × детекторы): детектор активен, если значение в его рецептивном поле (§4.1)."""
        values = np.asarray(values, dtype=np.float64)[:, None]
        if self.closed:
            diff = np.abs(values - self.centers[None, :]) % self.period
            distance = np.minimum(diff, self.period - diff)
            return (distance <= self.half_widths[None, :]) | (self.half_widths[None, :] >= self.period / 2.0)
        return np.abs(values - self.centers[None, :]) <= self.half_widths[None, :]

    def active_detector_bits(self, value: float) -> "np.ndarray":
        """Биты всех активных детекторов (с повторами: несколько детекторов могут делить бит)."""
        return self.bits[self.active_mask(np.asarray([value]))[0]]

    def active_bits(self, value: float) -> tuple["np.ndarray", "np.ndarray"]:
        unique, hits = np.unique(self.active_detector_bits(value), return_counts=True)
        return unique, hits


class Encoder:
    """Широкие детекторы со случайными битами, как в Angle-Encoder (§4.1–§4.6).

    §2.2.3 оставляет выбор: коды координат можно склеить (конкатенация — у каждой координаты свой
    сегмент, длина кода растёт) или слить в общий код той же длины (shared_code=True: биты всех
    координат объединяются операцией OR, §2.2.1; растут плотность и вероятность коллизий). Каждому
    детектору назначается один случайный бит цветовой полосы своего слоя (§3.2, §4.2.1); несколько
    детекторов, в том числе разных координат, могут разделять бит — такие коллизии допускаются
    статьёй и логируются как паразитные биты.
    """

    def __init__(
        self,
        *dimensions: Dimension,
        random_seed: Optional[int] = None,
        log_every: Optional[int] = None,
        layer_phase: bool = True,
        shared_code: bool = False,
    ) -> None:
        if not dimensions:
            raise ValueError("at least one dimension is required")
        self.shared_code = bool(shared_code)
        if self.shared_code and len({dim.code_length for dim in dimensions}) != 1:
            raise ValueError("shared code: all dimensions must have the same code_length (§2.2.3)")
        self.dimensions = list(dimensions)
        self.random_seed = random_seed
        self._rng = random.Random(random_seed)
        self.log_every = int(log_every) if log_every is not None else 0
        self._log_counter = 0
        self._layer_phase = bool(layer_phase)
        self.detectors: List[DetectorWindow] = []
        self._detectors_by_dimension: List[List[DetectorWindow]] = []
        self._segments: list[tuple[int, int]] = []
        self._indices: list[_DimensionIndex] = []
        offset = 0
        for dim_index, dim in enumerate(self.dimensions):
            detectors_for_dim: List[DetectorWindow] = []
            for layer_index, layer in enumerate(dim.detector_layers):
                windows = self._build_layer_detectors(
                    dim=dim,
                    dim_index=dim_index,
                    layer_index=layer_index,
                    layer=layer,
                    start_index=len(self.detectors) + len(detectors_for_dim),
                    offset=offset,
                )
                detectors_for_dim.extend(windows)
            self.detectors.extend(detectors_for_dim)
            self._detectors_by_dimension.append(detectors_for_dim)
            self._segments.append((offset, dim.code_length))
            self._indices.append(
                _DimensionIndex(
                    offset=offset,
                    code_length=dim.code_length,
                    closed=dim.closed,
                    period=dim.period,
                    min_value=dim.min_value,
                    max_value=dim.max_value,
                    centers=np.asarray([w.center for w in detectors_for_dim], dtype=np.float64),
                    half_widths=np.asarray(
                        [w.half_width for w in detectors_for_dim], dtype=np.float64
                    ),
                    bits=np.asarray([w.bit for w in detectors_for_dim], dtype=np.int64),
                )
            )
            self._log_dimension_bits(dim_index, dim, detectors_for_dim)
            if not self.shared_code:
                offset += dim.code_length
        # Общий код — длина одной координаты (все пишут в одни и те же биты); конкатенация — сумма сегментов.
        self.code_length = self.dimensions[0].code_length if self.shared_code else offset
        dim_summaries = []
        for dim in self.dimensions:
            status = "closed" if dim.closed else "open"
            layers = [(layer.count, layer.overlap) for layer in dim.detector_layers]
            dim_summaries.append(
                f"{dim.title}:{status} range=({dim.min_value},{dim.max_value}) "
                f"bits={dim.code_length} layers={layers}"
            )
        LOGGER.event(
            "encoder.init",
            section=MULTI_DIM_CODES,
            data={
                "dimensions": len(self.dimensions),
                "random_seed": random_seed,
                "layer_phase": self._layer_phase,
                "log_every": self.log_every,
                "dims": dim_summaries,
            },
        )
        expected_density = self.expected_density()
        LOGGER.event(
            "encoder.code_length",
            section=CONCATENATION,
            data={
                "mode": "merge" if self.shared_code else "concatenation",
                "code_length": self.code_length,
                "segments": self._segments,
                "expected_density": expected_density,
                "note": "§2.2.3: слияние сохраняет длину, но повышает плотность и коллизии (OR, §2.2.1); "
                "конкатенация сохраняет коды координат, но удлиняет код",
            },
        )
        if expected_density > _MAX_SPARSE_DENSITY:
            LOGGER.event(
                "encoder.density.warning",
                section=SPARSE_BIT_VECTORS,
                data={
                    "expected_density": expected_density,
                    "max_density": _MAX_SPARSE_DENSITY,
                    "note": "битовые векторы хорошо работают при плотности не выше 25% (§2); при высокой "
                    "насыщенности вместо OR нужен colour merge (§2.2.1, §3.3)",
                },
            )
        LOGGER.event(
            "encoder.detectors",
            section=WIDE_DETECTORS,
            data={"detectors": len(self.detectors)},
        )

    # ------------------------------------------------------------------ geometry
    def _build_layer_detectors(
        self,
        *,
        dim: Dimension,
        dim_index: int,
        layer_index: int,
        layer: Detectors,
        start_index: int,
        offset: int,
    ) -> List[DetectorWindow]:
        if dim.period <= 0:
            raise ValueError("dimension size must be greater than zero")
        count = layer.count
        overlap = layer.overlap
        phase = (layer_index * GOLDEN_RATIO_FRACTION) % 1.0 if self._layer_phase else 0.0
        step = dim.period / count
        width = step / (1.0 - overlap)
        if dim.closed:
            width = min(width, dim.period)
            centers = [
                dim.min_value + ((index + phase) * step) % dim.period
                for index in range(count)
            ]
        elif dim.edge_detectors:
            # Как в LinearEncoder: детекторы, заходящие за концы отрезка,
            # сохраняются, чтобы края покрывались так же плотно, как середина.
            half_steps = width / 2.0 / step
            first = int(math.floor(-half_steps - phase)) + 1
            last = int(math.ceil(count + half_steps - phase)) - 1
            centers = [dim.min_value + (index + phase) * step for index in range(first, last + 1)]
        else:
            # Ровно count детекторов внутри отрезка, по центру своих шагов: поле шире шага (перекрытие),
            # поэтому концы отрезка покрыты при любом перекрытии; сдвиг фазы слоя здесь не применяется —
            # при сдвиге больше полуширины поля начало или конец отрезка остались бы непокрытыми.
            phase = 0.5
            centers = [dim.min_value + (index + phase) * step for index in range(count)]
        half_width = width / 2.0
        LOGGER.event(
            "encoder.layer",
            section=INTEGER_ENCODING,
            data={
                "dimension": dim.title,
                "dimension_index": dim_index,
                "layer_index": layer_index,
                "count": count,
                "detectors": len(centers),
                "overlap": overlap,
                "width": width,
                "step": step,
                "half_width": half_width,
                "phase": phase,
                "center_min": min(centers),
                "center_max": max(centers),
                "closed": dim.closed,
            },
        )
        band_start, band_length = dim.layer_band(layer_index)
        LOGGER.event(
            "encoder.colour_band",
            section=COLOR_ENCODING,
            data={
                "dimension": dim.title,
                "layer_index": layer_index,
                "colour_rank": layer_index,
                "band_start": band_start,
                "band_length": band_length,
                "detectors": len(centers),
                "note": "§3.2: цвет бита — важность; широкий слой — красные младшие биты, "
                "узкий — фиолетовые старшие (§4.4.2, §5.14); бит детектора случаен внутри полосы (§4.2.1)",
            },
        )
        windows: List[DetectorWindow] = []
        # Один случайный бит на детектор (§4.2.1): биты полосы выдаются случайной перестановкой, поэтому
        # детекторы слоя не совпадают, пока их не больше длины полосы; дальше перестановка повторяется
        # и совпадения далёких детекторов допускаются (коллизии §4.2.1).
        bit_pool: list[int] = []
        for position, center in enumerate(centers):
            if not bit_pool:
                bit_pool = list(range(band_start, band_start + band_length))
                self._rng.shuffle(bit_pool)
            local_bit = bit_pool.pop()
            windows.append(
                DetectorWindow(
                    detector_index=start_index + position,
                    dimension_index=dim_index,
                    layer_index=layer_index,
                    center=center,
                    half_width=half_width,
                    closed=dim.closed,
                    span_min=dim.min_value,
                    span_max=dim.max_value,
                    period=dim.period,
                    bit=offset + local_bit,
                    local_bit=local_bit,
                )
            )
        return windows

    def _log_dimension_bits(
        self,
        dim_index: int,
        dim: Dimension,
        windows: Sequence[DetectorWindow],
    ) -> None:
        usage = [0] * dim.code_length
        for window in windows:
            usage[window.local_bit] += 1
        used_bits = sum(1 for value in usage if value > 0)
        shared_bits = sum(1 for value in usage if value > 1)
        LOGGER.event(
            "encoder.random_bits",
            section=INTEGER_ENCODING,
            data={
                "dimension": dim.title,
                "dimension_index": dim_index,
                "detectors": len(windows),
                "code_length": dim.code_length,
                "used_bits": used_bits,
                "shared_bits": shared_bits,
                "max_detectors_per_bit": max(usage) if usage else 0,
                "expected_density": self.expected_dimension_density(dim),
                "random_seed": self.random_seed,
            },
        )

    # ---------------------------------------------------------------- properties
    @staticmethod
    def expected_dimension_density(dim: Dimension) -> float:
        """Оценка плотности сегмента (§4.2.1): в полосе слоя b_l бит и ≈ 1/(1 − overlap) активных детекторов."""
        expected_bits = 0.0
        for layer_index, layer in enumerate(dim.detector_layers):
            _, band_length = dim.layer_band(layer_index)
            active_detectors = 1.0 / (1.0 - layer.overlap)
            expected_bits += band_length * (1.0 - (1.0 - 1.0 / band_length) ** active_detectors)
        return expected_bits / dim.code_length

    def expected_density(self) -> float:
        """Ожидаемая плотность кода (§2, §4.2.1): бит не установлен, если его не поставил ни один слой ни одной
        координаты; в общем коде полосы разных координат накладываются друг на друга."""
        unset = np.ones(self.code_length, dtype=np.float64)
        for (offset, _), dim in zip(self._segments, self.dimensions):
            for layer_index, layer in enumerate(dim.detector_layers):
                start, length = dim.layer_band(layer_index)
                active_detectors = 1.0 / (1.0 - layer.overlap)
                unset[offset + start: offset + start + length] *= (1.0 - 1.0 / length) ** active_detectors
        return float(1.0 - unset.mean())

    @property
    def signature(self) -> str:
        """Отпечаток конфигурации кодера: геометрия слоёв и назначенные биты."""
        digest = hashlib.sha256()
        digest.update(f"shared={self.shared_code}|".encode())
        for dim in self.dimensions:
            digest.update(
                f"{dim.title}|{dim.min_value}|{dim.max_value}|{dim.closed}|{dim.code_length}|".encode()
            )
            for layer in dim.detector_layers:
                digest.update(f"{layer.count}:{layer.overlap};".encode())
        for window in self.detectors:
            digest.update(f"{window.center:.9f},{window.half_width:.9f},{window.bit};".encode())
        return digest.hexdigest()

    def segment(self, dimension: int | str) -> tuple[int, int]:
        """Начало и длина сегмента кода координаты (§2.2.3)."""
        if isinstance(dimension, str):
            for index, dim in enumerate(self.dimensions):
                if dim.title == dimension:
                    return self._segments[index]
            raise KeyError(f"unknown dimension title: {dimension}")
        return self._segments[int(dimension)]

    # ----------------------------------------------------------------- encoding
    def long_range_order(
        self,
        dimension: int | str,
        *,
        samples: int = 25,
        step: float | None = None,
        tolerance: float = 0.02,
    ) -> tuple[list[tuple[float, float]], bool]:
        """Профиль сходства кодов вдоль одной координаты (§4.5, §5.13.1).

        Остальные координаты фиксируются в середине диапазона, по выбранной
        берётся равномерная сетка из `samples` значений, для каждой пары
        считается косинусное сходство кодов, и оно усредняется по расстоянию.
        Сходство считается по битам сегмента этой координаты. Возвращает профиль
        (расстояние, среднее сходство) и признак дальнего порядка: сходство
        убывает монотонно (с допуском `tolerance`) и без плато на четвертях диапазона.
        Плоский или немонотонный хвост означает, что коды не несут топологию
        координаты на больших расстояниях и раскладка не сможет её воспроизвести.
        """
        index = dimension if isinstance(dimension, int) else [d.title for d in self.dimensions].index(dimension)
        dim = self.dimensions[index]
        span = dim.period if dim.closed else dim.max_value - dim.min_value
        if step is not None:
            # Шаг сетки стимулов: профиль считается на тех же расстояниях, что встречаются в данных.
            if step <= 0.0:
                raise ValueError("step must be positive")
            samples = int(round(span / step)) + (0 if dim.closed else 1)
        if samples < 3:
            raise ValueError("samples must be at least 3")
        if dim.closed:
            grid = [dim.min_value + span * k / samples for k in range(samples)]
        else:
            grid = [dim.min_value + span * k / (samples - 1) for k in range(samples)]
        codes = []
        for value in grid:
            # Сравниваются только биты детекторов этой координаты: биты остальных координат
            # постоянны и лишь поднимают пол сходства, скрывая плоский хвост профиля (и в общем коде тоже).
            codes.append(set(self.dimension_bits(index, value)))
        step = span / samples if dim.closed else span / (samples - 1)
        buckets: dict[int, list[float]] = {}
        for i in range(len(grid)):
            for j in range(i + 1, len(grid)):
                distance = abs(grid[i] - grid[j])
                if dim.closed:
                    distance = min(distance, span - distance)
                bucket = int(round(distance / step))
                if bucket <= 0:
                    continue
                denom = math.sqrt(len(codes[i]) * len(codes[j]))
                buckets.setdefault(bucket, []).append(len(codes[i] & codes[j]) / denom if denom else 0.0)
        profile = [(bucket * step, sum(v) / len(v)) for bucket, v in sorted(buckets.items())]
        monotone = all(
            profile[k + 1][1] <= profile[k][1] + tolerance for k in range(len(profile) - 1)
        )
        # Плато: на средней половине диапазона расстояний сходство должно упасть хотя бы на
        # 15% общего падения, иначе коды не различают позиции на средних расстояниях и
        # раскладка не воспроизведёт их порядок (§5.13.1).
        total_drop = profile[0][1] - profile[-1][1]
        max_distance = profile[-1][0]
        middle = [v for d, v in profile if 0.25 * max_distance - 1e-9 <= d <= 0.75 * max_distance + 1e-9]
        plateau = bool(total_drop > 0.0 and len(middle) >= 2 and (middle[0] - middle[-1]) < 0.15 * total_drop)
        # §4.5: сходство и на ближней, и на дальней шкале — профиль должен убывать с расстоянием
        # (постоянный профиль, в том числе нулевой, порядка не несёт) и не обнуляться на дальнем конце.
        decreasing = total_drop > 0.0
        far_similarity = profile[-1][1] > 0.0
        ordered = monotone and decreasing and far_similarity and not plateau
        LOGGER.event(
            "encoder.long_range_order",
            section=LONG_RANGE_IMPORTANCE,
            data={
                "dimension": dim.title,
                "samples": samples,
                "monotone": monotone,
                "decreasing": decreasing,
                "far_similarity": far_similarity,
                "plateau": plateau,
                "ordered": ordered,
                "profile": [(round(d, 3), round(v, 3)) for d, v in profile],
                "note": "сходство должно убывать с расстоянием на всём диапазоне (§4.5), "
                "иначе раскладка не воспроизведёт дальний порядок координаты (§5.13.1)",
            },
        )
        if not ordered:
            LOGGER.event(
                "encoder.long_range_order.warning",
                section=LONG_RANGE_IMPORTANCE,
                data={
                    "dimension": dim.title,
                    "monotone": monotone,
                    "decreasing": decreasing,
                    "far_similarity": far_similarity,
                    "plateau": plateau,
                    "note": "у координаты нет дальнего порядка (§4.5): измените перекрытие или число слоёв",
                },
            )
        return profile, ordered

    def encode_indices(
        self, *values: float
    ) -> tuple[tuple[float, ...], tuple[int, ...], int]:
        """Нормализованные значения, активные биты и число паразитных битов."""
        normalized, hits = self._hit_matrix([values])
        active = np.flatnonzero(hits[0])
        # OR кодов координат (§2.2.1); бит, который поставили несколько детекторов, — паразитный (§4.2.1).
        shared_bits = int(np.count_nonzero(hits[0] > 1))
        return tuple(float(v) for v in normalized[0]), tuple(int(bit) for bit in active), shared_bits

    def _hit_matrix(self, rows: Sequence[Sequence[float]]) -> tuple["np.ndarray", "np.ndarray"]:
        """Для набора стимулов: нормализованные значения (k × dims) и число детекторов на бит (k × длина кода).

        Все координаты считаются векторно: маска активных детекторов каждой координаты (§4.1),
        их биты суммируются в общую матрицу — это OR кодов координат (§2.2.1) с подсчётом коллизий (§4.2.1).
        """
        matrix = np.asarray(rows, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.dimensions):
            raise ValueError("number of values must match number of dimensions")
        count = matrix.shape[0]
        normalized = np.empty_like(matrix)
        flat_hits: list[np.ndarray] = []
        for column, (dim, index) in enumerate(zip(self.dimensions, self._indices)):
            normalized[:, column] = self._normalize_values(dim, matrix[:, column])
            if count == 0:
                continue
            mask = index.active_mask(normalized[:, column])
            row_ids, detector_ids = np.nonzero(mask)
            flat_hits.append(row_ids.astype(np.int64) * self.code_length + index.bits[detector_ids])
        if count == 0:
            return normalized, np.zeros((0, self.code_length), dtype=np.int32)
        flat = np.concatenate(flat_hits) if flat_hits else np.zeros(0, dtype=np.int64)
        hits = np.bincount(flat, minlength=count * self.code_length).astype(np.int32)
        return normalized, hits.reshape(count, self.code_length)

    @staticmethod
    def _normalize_values(dim: "Dimension", values: "np.ndarray") -> "np.ndarray":
        values = np.asarray(values, dtype=np.float64)
        if dim.closed:
            return ((values - dim.min_value) % dim.period) + dim.min_value
        outside = (values < dim.min_value) | (values > dim.max_value)
        if np.any(outside):
            v = float(values[outside][0])
            raise ValueError(
                f"value {v} is outside the open dimension range "
                f"[{dim.min_value}, {dim.max_value}]"
            )
        return values

    @classmethod
    def _normalize_value(cls, dim: "Dimension", value: float) -> float:
        return float(cls._normalize_values(dim, np.asarray([float(value)]))[0])

    @staticmethod
    def _bit_arrays_from_hits(hits: "np.ndarray") -> list[BitArray]:
        bit_matrix = (hits > 0).astype(np.uint8)
        return [
            BitArray.from_bits(bit_matrix[i].tobytes(), np.flatnonzero(bit_matrix[i]).tolist())
            for i in range(bit_matrix.shape[0])
        ]

    def dimension_bits(self, dimension: int | str, value: float) -> tuple[int, ...]:
        """Биты, которые ставят детекторы одной координаты при данном значении (её собственный код)."""
        index = dimension if isinstance(dimension, int) else [d.title for d in self.dimensions].index(dimension)
        dim = self.dimensions[index]
        bits, _ = self._indices[index].active_bits(self._normalize_value(dim, value))
        return tuple(int(bit) for bit in bits)

    def encode(
        self,
        *values: float,
        log_image: object | None = None,
        log_label: int | None = None,
        log_measurements: Sequence[tuple[float, int, int]] | None = None,
    ) -> Tuple[Tuple[float, ...], BitArray]:
        results = self.encode_batch(
            [values], log_image=log_image, log_label=log_label, log_measurements=log_measurements
        )
        return results[0]

    def encode_batch(
        self,
        rows: Sequence[Sequence[float]],
        *,
        log_image: object | None = None,
        log_label: int | None = None,
        log_measurements: Sequence[tuple[float, int, int]] | None = None,
    ) -> list[Tuple[Tuple[float, ...], BitArray]]:
        """Коды набора стимулов одним векторным проходом (все углопозиции цифры разом).

        Результат побитово тот же, что у encode для каждого стимула по отдельности; коды считаются
        матрицей (стимулы × биты), поэтому стоимость почти не зависит от числа стимулов.
        """
        rows = [tuple(float(v) for v in row) for row in rows]
        if not rows:
            return []
        LOGGER.event(
            "encoder.encode.input",
            section=MULTI_DIM_CODES,
            data={"stimuli": len(rows), "values": rows[0], "dimensions": len(self.dimensions)},
        )
        normalized, hits = self._hit_matrix(rows)
        codes = self._bit_arrays_from_hits(hits)
        active_counts = np.count_nonzero(hits > 0, axis=1)
        shared_counts = np.count_nonzero(hits > 1, axis=1)
        LOGGER.event(
            "encoder.encode.normalized",
            section=CYCLIC_COORDS,
            data={"stimuli": len(rows), "values": tuple(float(v) for v in normalized[0])},
        )
        self._log_counter += len(rows)
        code_visuals = []
        code_bits: list[int] | None = None
        patch_meta: dict[str, float | int | str] = {}
        if self.log_every > 0 and self._log_counter % self.log_every < len(rows):
            code_bits = list(codes[0])
            code_visuals.append(LOGGER.visual_bar_chart("encoding/code_bits", code_bits))
        if log_image is not None or log_measurements:
            patch_meta = self._log_stimulus_image(
                codes[0], code_bits, log_image=log_image, log_label=log_label, log_measurements=log_measurements
            )
        LOGGER.event(
            "encoder.encode.code",
            section=SPARSE_BIT_VECTORS,
            data={
                "stimuli": len(rows),
                "code_length": self.code_length,
                "active_bits": int(active_counts[0]),
                "active_bits_mean": float(active_counts.mean()),
                "density": float(active_counts.mean()) / self.code_length,
                "shared_bits": int(shared_counts[0]),
                "shared_bits_mean": float(shared_counts.mean()),
                "code": codes[0],
                "label": int(log_label) if log_label is not None else None,
                "measurements": len(log_measurements) if log_measurements else 0,
                **patch_meta,
            },
            visuals=code_visuals if code_visuals else None,
        )
        return [
            (tuple(float(v) for v in normalized[i]), codes[i]) for i in range(len(rows))
        ]

    def _log_stimulus_image(
        self,
        code: BitArray,
        code_bits: list[int] | None,
        *,
        log_image: object | None,
        log_label: int | None,
        log_measurements: Sequence[tuple[float, int, int]] | None,
    ) -> dict[str, float | int | str]:
        """Картинка стимула с углопозициями и код первого стимула — в Rerun (вне статьи)."""
        image_visuals = []
        image_array = None
        patch_meta: dict[str, float | int | str] = {}
        image_path = "encoding/image"
        if log_label is not None:
            image_path = f"label_{int(log_label)}/image"
        if log_image is not None:
            image_array = np.asarray(log_image)
            image_visuals.append(LOGGER.visual_image(image_path, image_array))
            if code_bits is None:
                code_bits = list(code)
            image_visuals.append(LOGGER.visual_bar_chart(f"{image_path}/code_bits", code_bits))
        if log_measurements:
            positions, colors, patch_meta, grid = self._patch_points(
                log_measurements,
                image_array.shape if image_array is not None else None,
                grid_override=self._infer_patch_grid(),
            )
            image_visuals.append(
                LOGGER.visual_points2d(image_path, positions, colors=colors, radii=grid["radius"])
            )
            if grid["mins"]:
                image_visuals.append(
                    LOGGER.visual_boxes2d(image_path, grid["mins"], grid["sizes"], colors=grid["colors"])
                )
        if image_visuals:
            LOGGER.event(
                "encoder.encode.image",
                section=OUTSIDE_ARTICLE,
                data={
                    "label": int(log_label) if log_label is not None else None,
                    "measurements": len(log_measurements) if log_measurements else 0,
                    **patch_meta,
                },
                visuals=image_visuals,
            )
        return patch_meta

    # ------------------------------------------------------------- pseudo colour
    @staticmethod
    def color_from_code(
        code: BitArray,
        *,
        min_hue: float,
        max_hue: float,
    ) -> tuple[int, int, int]:
        """Сумма псевдоцветов единичных битов: младший бит красный, старший фиолетовый (§5.14)."""
        if max_hue < min_hue:
            raise ValueError("max_hue must be >= min_hue")
        active = code.active_indices()
        if not active:
            return (0, 0, 0)
        denominator = max(1, len(code) - 1)
        red = green = blue = 0.0
        for index in active:
            hue_degrees = min_hue + (max_hue - min_hue) * index / denominator
            r, g, b = colorsys.hsv_to_rgb((hue_degrees % 360.0) / 360.0, 1.0, 1.0)
            red += r
            green += g
            blue += b
        scale = 255.0 / max(red, green, blue)
        color = (
            int(round(red * scale)),
            int(round(green * scale)),
            int(round(blue * scale)),
        )
        LOGGER.event(
            "encoder.code.color",
            section=LAID_OUT_STRUCTURE,
            data={
                "code_length": len(code),
                "active_bits": len(active),
                "min_hue": min_hue,
                "max_hue": max_hue,
                "color": color,
            },
        )
        return color

    # ------------------------------------------------------------- visual helpers
    @staticmethod
    def _angle_to_rgb(angle_deg: float) -> tuple[int, int, int]:
        hue = (angle_deg % 360.0) / 360.0
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return int(r * 255), int(g * 255), int(b * 255)

    def _patch_points(
        self,
        measurements: Sequence[tuple[float, int, int]],
        image_shape: Sequence[int] | None,
        *,
        grid_override: tuple[int, int] | None = None,
    ) -> tuple[
        list[tuple[float, float]],
        list[tuple[int, int, int]],
        dict[str, float | int | str],
        dict[str, object],
    ]:
        if grid_override:
            grid_x, grid_y = grid_override
        else:
            max_x = max(int(x) for _, x, _ in measurements)
            max_y = max(int(y) for _, _, y in measurements)
            grid_x = max_x + 1
            grid_y = max_y + 1
        step_x: float | None = None
        step_y: float | None = None
        offset_x = 0.0
        offset_y = 0.0
        patch_edge: float | None = None
        positions: list[tuple[float, float]] = []
        if image_shape and len(image_shape) >= 2 and grid_x > 0 and grid_y > 0:
            height = float(image_shape[0])
            width = float(image_shape[1])
            if height > 0 and width > 0:
                step_x = width / grid_x
                step_y = height / grid_y
        if step_x is not None and step_y is not None:
            patch_edge = min(step_x, step_y)
            offset_x = max(0.0, (step_x - patch_edge) * grid_x / 2.0)
            offset_y = max(0.0, (step_y - patch_edge) * grid_y / 2.0)
            for _, x, y in measurements:
                positions.append(
                    (
                        offset_x + (float(x) + 0.5) * patch_edge,
                        offset_y + (float(y) + 0.5) * patch_edge,
                    )
                )
        else:
            patch_edge = 1.0
            for _, x, y in measurements:
                positions.append((float(x) + 0.5, float(y) + 0.5))
        colors = [self._angle_to_rgb(angle) for angle, _, _ in measurements]
        grid_mins: list[tuple[float, float]] = []
        grid_sizes: list[tuple[float, float]] = []
        grid_colors: list[tuple[int, int, int, int]] = []
        if patch_edge is not None:
            for y in range(grid_y):
                for x in range(grid_x):
                    grid_mins.append((offset_x + x * patch_edge, offset_y + y * patch_edge))
                    grid_sizes.append((patch_edge, patch_edge))
                    grid_colors.append((255, 255, 255, 60))
        meta: dict[str, float | int | str] = {
            "patch_grid_x": grid_x,
            "patch_grid_y": grid_y,
            "patch_edge": patch_edge if patch_edge is not None else "None",
            "patch_offset_x": offset_x,
            "patch_offset_y": offset_y,
        }
        grid = {
            "mins": grid_mins,
            "sizes": grid_sizes,
            "colors": grid_colors,
            "radius": patch_edge / 2.0 if patch_edge is not None else 0.5,
        }
        return positions, colors, meta, grid

    def _infer_patch_grid(self) -> tuple[int, int] | None:
        x_dim = None
        y_dim = None
        for dim in self.dimensions:
            title = getattr(dim, "title", "")
            if str(title).lower() == "x":
                x_dim = dim
            elif str(title).lower() == "y":
                y_dim = dim
        if x_dim is None or y_dim is None:
            return None
        grid_x = int(round(x_dim.max_value)) - int(round(x_dim.min_value)) + 1
        grid_y = int(round(y_dim.max_value)) - int(round(y_dim.min_value)) + 1
        if grid_x <= 0 or grid_y <= 0:
            return None
        return grid_x, grid_y


@dataclass(frozen=True)
class EncoderCodeColorizer:
    """Псевдоцвет кода или его компоненты по §5.14 (ср. «corner component» на рис. 16)."""

    min_hue: float
    max_hue: float
    start_bit: int = 0
    bit_length: int | None = None

    def __post_init__(self) -> None:
        if self.max_hue < self.min_hue:
            raise ValueError("max_hue must be >= min_hue")
        if self.start_bit < 0:
            raise ValueError("start_bit must be non-negative")
        if self.bit_length is not None and self.bit_length <= 0:
            raise ValueError("bit_length must be positive when set")

    @classmethod
    def for_dimension(
        cls,
        encoder: Encoder,
        dimension: int | str,
        *,
        min_hue: float,
        max_hue: float,
    ) -> "EncoderCodeColorizer":
        start, length = encoder.segment(dimension)
        colorizer = cls(min_hue=min_hue, max_hue=max_hue, start_bit=start, bit_length=length)
        LOGGER.event(
            "encoder.colorizer.dimension",
            section=LAID_OUT_STRUCTURE,
            data={
                "dimension": dimension,
                "start_bit": start,
                "bit_length": length,
                "min_hue": min_hue,
                "max_hue": max_hue,
            },
        )
        return colorizer

    def color(self, code: BitArray) -> tuple[int, int, int]:
        stop = len(code) if self.bit_length is None else self.start_bit + self.bit_length
        if self.start_bit >= len(code) or stop > len(code):
            raise ValueError("color component is outside Encoder code")
        if self.start_bit == 0 and stop == len(code):
            selected = code
        else:
            selected = BitArray.from_indices(
                stop - self.start_bit,
                (
                    index - self.start_bit
                    for index in code.active_indices()
                    if self.start_bit <= index < stop
                ),
            )
        return Encoder.color_from_code(selected, min_hue=self.min_hue, max_hue=self.max_hue)
