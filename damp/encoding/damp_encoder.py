from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, List, Optional, Sequence, Tuple

LOG_ENABLED = True


def _log(message: str) -> None:
    if LOG_ENABLED:
        print(f"[encode] {message}")


@dataclass(frozen=True)
class Detectors:
    count: int
    overlap: float

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("detector count must be positive")
        if not (0.0 <= self.overlap <= 1.0):
            raise ValueError("overlap must be in [0.0, 1.0]")


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


class BitArray:
    def __init__(self, length: int, fill: int = 0) -> None:
        if length <= 0:
            raise ValueError("bit array length must be positive")
        if fill not in (0, 1):
            raise ValueError("fill must be 0 or 1")
        self._data = bytearray([fill] * length)

    def set(self, index: int, value: int = 1) -> None:
        self._data[index] = 1 if value else 0

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> int:
        return self._data[index]

    def __iter__(self):
        return iter(self._data)

    def count(self) -> int:
        return sum(self._data)

    def to01(self) -> str:
        return "".join("1" if bit else "0" for bit in self._data)

    def __repr__(self) -> str:
        return f"BitArray(len={len(self._data)}, ones={self.count()})"


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
        _log(
            f"encoder init dims={len(self.dimensions)} code_length={self.code_length} "
            f"random_bit={self._random_bit} dims={dim_summaries}"
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
