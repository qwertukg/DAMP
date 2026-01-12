from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Optional, Sequence, Tuple

from damp.article_refs import (
    CYCLIC_COORDS,
    ENCODING_SYSTEM,
    GEOMETRIC_METHOD,
    MULTI_DIM_CODES,
    ONE_DIM_CODES,
    SPARSE_BIT_VECTORS,
    WIDE_DETECTORS,
)
from damp.encoding.bitarray import BitArray
from damp.logging import log_event


@dataclass(frozen=True)
class Detectors:
    count: int
    overlap: float

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("detector count must be positive")
        if not (0.0 <= self.overlap <= 1.0):
            raise ValueError("overlap must be in [0.0, 1.0]")
        log_event(
            "detectors.init",
            section=WIDE_DETECTORS,
            data={"count": self.count, "overlap": self.overlap},
        )


class Dimension:
    def __init__(
        self,
        title: str,
        size: Tuple[float, float],
        detector_layers: Sequence[Detectors],
        closed: bool,
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
        self.detector_layers = list(detector_layers)
        self.closed = bool(closed)
        self.period = self.max_value - self.min_value
        layers = [(layer.count, layer.overlap) for layer in self.detector_layers]
        log_event(
            "dimension.init",
            section=ONE_DIM_CODES,
            data={
                "title": self.title,
                "min_value": self.min_value,
                "max_value": self.max_value,
                "closed": self.closed,
                "period": self.period,
                "layers": layers,
            },
        )
        if self.closed:
            log_event(
                "dimension.closed",
                section=CYCLIC_COORDS,
                data={"title": self.title, "period": self.period},
            )


class ClosedDimension(Dimension):
    def __init__(self, title: str, size: Tuple[float, float], detector_layers: Sequence[Detectors]) -> None:
        super().__init__(title=title, size=size, detector_layers=detector_layers, closed=True)


class OpenedDimension(Dimension):
    def __init__(self, title: str, size: Tuple[float, float], detector_layers: Sequence[Detectors]) -> None:
        super().__init__(title=title, size=size, detector_layers=detector_layers, closed=False)


@dataclass
class DetectorWindow:
    detector_index: int
    dimension_index: int
    layer_index: int
    center: float
    half_width: float
    closed: bool
    span_min: float
    span_max: float
    period: float

    def is_active(self, value: float) -> bool:
        if self.closed:
            if self.half_width >= self.period / 2.0:
                return True
            diff = abs(value - self.center) % self.period
            return min(diff, self.period - diff) <= self.half_width
        return (self.center - self.half_width) <= value <= (self.center + self.half_width)


class Encoder:
    def __init__(
        self,
        *dimensions: Dimension,
        random_bit: bool = False,
        random_seed: Optional[int] = None,
        log_every: Optional[int] = None,
    ) -> None:
        if not dimensions:
            raise ValueError("at least one dimension is required")
        self.dimensions = list(dimensions)
        self._rng = random.Random(random_seed)
        self.log_every = int(log_every) if log_every is not None else 0
        self._log_state: object | None = None
        self.detectors: List[DetectorWindow] = []
        self._detectors_by_dimension: List[List[DetectorWindow]] = []
        next_index = 0
        for dim_index, dim in enumerate(self.dimensions):
            detectors_for_dim: List[DetectorWindow] = []
            for layer_index, layer in enumerate(dim.detector_layers):
                windows = self._build_layer_detectors(
                    dim=dim,
                    dim_index=dim_index,
                    layer_index=layer_index,
                    layer=layer,
                    start_index=next_index,
                )
                detectors_for_dim.extend(windows)
                self.detectors.extend(windows)
                next_index += len(windows)
            self._detectors_by_dimension.append(detectors_for_dim)
        self.code_length = len(self.detectors)
        self._bit_map = list(range(self.code_length))
        self._random_bit = False
        if random_bit:
            self.random_bit = True
        dim_summaries = []
        for dim in self.dimensions:
            status = "closed" if dim.closed else "open"
            layers = [(layer.count, layer.overlap) for layer in dim.detector_layers]
            dim_summaries.append(
                f"{dim.title}:{status} range=({dim.min_value},{dim.max_value}) layers={layers}"
            )
        log_event(
            "encoder.init",
            section=ENCODING_SYSTEM,
            data={
                "dimensions": len(self.dimensions),
                "random_bit": self._random_bit,
                "random_seed": random_seed,
                "log_every": self.log_every,
                "dims": dim_summaries,
            },
        )
        log_event(
            "encoder.code_length",
            section=SPARSE_BIT_VECTORS,
            data={"code_length": self.code_length},
        )
        log_event(
            "encoder.detectors",
            section=WIDE_DETECTORS,
            data={"detectors": len(self.detectors)},
        )

    @property
    def random_bit(self) -> bool:
        return self._random_bit

    @random_bit.setter
    def random_bit(self, value: bool) -> None:
        value = bool(value)
        if value == self._random_bit:
            return
        self._random_bit = value
        if value:
            self._bit_map = [self._rng.randrange(self.code_length) for _ in range(self.code_length)]
        else:
            self._bit_map = list(range(self.code_length))
        log_event(
            "encoder.random_bit",
            section=ENCODING_SYSTEM,
            data={"random_bit": self._random_bit},
        )

    @property
    def randomBit(self) -> bool:
        return self.random_bit

    @randomBit.setter
    def randomBit(self, value: bool) -> None:
        self.random_bit = value

    def encode(
        self,
        *values: float,
        log_image: object | None = None,
        log_label: int | None = None,
        log_measurements: Sequence[tuple[float, int, int]] | None = None,
    ) -> Tuple[Tuple[float, ...], BitArray]:
        if len(values) != len(self.dimensions):
            raise ValueError("number of values must match number of dimensions")
        log_event(
            "encoder.encode.input",
            section=MULTI_DIM_CODES,
            data={"values": values},
        )
        bit_array = BitArray(self.code_length)
        normalized_values: List[float] = []
        for dim_index, (value, dim) in enumerate(zip(values, self.dimensions)):
            v = float(value)
            if dim.closed:
                if dim.period <= 0:
                    raise ValueError("closed dimension period must be positive")
                v = ((v - dim.min_value) % dim.period) + dim.min_value
            else:
                if not (dim.min_value <= v <= dim.max_value):
                    raise ValueError(
                        f"value {v} is outside the open dimension range "
                        f"[{dim.min_value}, {dim.max_value}]"
                    )
            normalized_values.append(v)
            for detector in self._detectors_by_dimension[dim_index]:
                if detector.is_active(v):
                    bit_array.set(self._bit_map[detector.detector_index], 1)
        log_event(
            "encoder.encode.normalized",
            section=CYCLIC_COORDS,
            data={"values": normalized_values},
        )
        log_event(
            "encoder.encode.code",
            section=SPARSE_BIT_VECTORS,
            data={
                "code_length": self.code_length,
                "active_bits": bit_array.count(),
                "code": bit_array,
            },
        )
        if self.log_every > 0 and log_image is not None:
            from damp.encoding import visualize_encoding as _viz

            if self._log_state is None:
                self._log_state = _viz.LogState()
            label_value = int(log_label) if log_label is not None else None
            _viz.log_encoding(
                self,
                normalized_values,
                bit_array,
                log_image,
                label_value,
                log_every=self.log_every,
                state=self._log_state,
                measurements=log_measurements,
            )
        return tuple(normalized_values), bit_array

    def _build_layer_detectors(
        self,
        dim: Dimension,
        dim_index: int,
        layer_index: int,
        layer: Detectors,
        start_index: int,
    ) -> List[DetectorWindow]:
        if dim.period <= 0:
            raise ValueError("dimension size must be greater than zero")
        count = layer.count
        overlap = layer.overlap
        if dim.closed:
            step = dim.period / count
            if overlap >= 1.0:
                width = dim.period
            else:
                width = step / (1.0 - overlap)
            if width > dim.period:
                width = dim.period
            centers = [dim.min_value + (i + 0.5) * step for i in range(count)]
        else:
            if count == 1:
                width = dim.period
                step = 0.0
            else:
                width = dim.period / (1.0 + (count - 1) * (1.0 - overlap))
                step = width * (1.0 - overlap)
            centers = [dim.min_value + width / 2.0 + i * step for i in range(count)]
        half_width = width / 2.0
        center_min = centers[0] if centers else None
        center_max = centers[-1] if centers else None
        log_event(
            "encoder.layer",
            section=GEOMETRIC_METHOD,
            data={
                "dimension": dim.title,
                "dimension_index": dim_index,
                "layer_index": layer_index,
                "count": count,
                "overlap": overlap,
                "width": width,
                "step": step,
                "half_width": half_width,
                "center_min": center_min,
                "center_max": center_max,
                "closed": dim.closed,
            },
        )
        windows = []
        for offset, center in enumerate(centers):
            windows.append(
                DetectorWindow(
                    detector_index=start_index + offset,
                    dimension_index=dim_index,
                    layer_index=layer_index,
                    center=center,
                    half_width=half_width,
                    closed=dim.closed,
                    span_min=dim.min_value,
                    span_max=dim.max_value,
                    period=dim.period,
                )
            )
        return windows
