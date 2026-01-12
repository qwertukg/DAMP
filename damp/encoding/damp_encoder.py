from __future__ import annotations

from dataclasses import dataclass
import colorsys
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np

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
from damp.logging import LOGGER


@dataclass(frozen=True)
class Detectors:
    count: int
    overlap: float

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("detector count must be positive")
        if not (0.0 <= self.overlap <= 1.0):
            raise ValueError("overlap must be in [0.0, 1.0]")
        LOGGER.event(
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
        LOGGER.event(
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
            LOGGER.event(
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
        self._log_counter = 0
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
        LOGGER.event(
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
        LOGGER.event(
            "encoder.code_length",
            section=SPARSE_BIT_VECTORS,
            data={"code_length": self.code_length},
        )
        LOGGER.event(
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
        LOGGER.event(
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
        LOGGER.event(
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
        LOGGER.event(
            "encoder.encode.normalized",
            section=CYCLIC_COORDS,
            data={"values": normalized_values},
        )
        self._log_counter += 1
        visuals = []
        patch_meta: dict[str, float | int | str] = {}
        if self.log_every > 0 and self._log_counter % self.log_every == 0:
            visuals.append(
                LOGGER.visual_bar_chart(
                    "encoding/code_bits",
                    list(bit_array),
                )
            )
            image_array = None
            image_path = "encoding/image"
            if log_label is not None:
                image_path = f"label_{int(log_label)}/image"
            if log_image is not None:
                image_array = np.asarray(log_image)
                visuals.append(
                    LOGGER.visual_image(
                        image_path,
                        image_array,
                    )
                )
            if log_measurements:
                positions, colors, patch_meta, grid = self._patch_points(
                    log_measurements,
                    image_array.shape if image_array is not None else None,
                    grid_override=self._infer_patch_grid(),
                )
                visuals.append(
                    LOGGER.visual_points2d(
                        image_path,
                        positions,
                        colors=colors,
                        radii=grid["radius"],
                    )
                )
                if grid["mins"]:
                    visuals.append(
                        LOGGER.visual_boxes2d(
                            image_path,
                            grid["mins"],
                            grid["sizes"],
                            colors=grid["colors"],
                        )
                    )
        LOGGER.event(
            "encoder.encode.code",
            section=SPARSE_BIT_VECTORS,
            data={
                "code_length": self.code_length,
                "active_bits": bit_array.count(),
                "code": bit_array,
                "label": int(log_label) if log_label is not None else None,
                "measurements": len(log_measurements) if log_measurements else 0,
                **patch_meta,
            },
            visuals=visuals if visuals else None,
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
        LOGGER.event(
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
                    grid_mins.append(
                        (
                            offset_x + x * patch_edge,
                            offset_y + y * patch_edge,
                        )
                    )
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
