from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import math
from typing import Sequence

from damp.article_refs import CYCLIC_COORDS, MULTI_DIM_CODES
from damp.encoding.bitarray import BitArray
from damp.encoding.damp_encoder import (
    ClosedDimension,
    Detectors,
    Encoder,
    EncoderCodeColorizer,
    OpenedDimension,
)
from damp.encoding.joint_encoder import JointDetectorLayer, JointEncoder
from damp.layout.pinwheel_topology import ContinuousPinwheelTopology
from damp.logging import LOGGER


@dataclass(frozen=True)
class OrientationEncoding:
    center_index: int
    center_x: float
    center_y: float
    position_x: float
    position_y: float
    source_angle: float
    encoded_angle: float
    chirality: int
    phase: float
    radius: float


class RetinotopicOrientationEncoder:
    """Общий Encoder позиции и осевой ориентации для MNIST и pinwheel-теста."""

    def __init__(
        self,
        topology: ContinuousPinwheelTopology,
        spatial_encoder: JointEncoder,
        angle_encoder: Encoder,
        *,
        angle_minimum: float,
        angle_maximum: float,
        color_minimum_hue: float,
        color_maximum_hue: float,
        color_angle_start: int,
        color_angle_length: int,
        cache_size: int = 0,
    ) -> None:
        self.topology = topology
        self._spatial_encoder = spatial_encoder
        self._angle_encoder = angle_encoder
        self._angle_minimum = float(angle_minimum)
        self._angle_maximum = float(angle_maximum)
        self._angle_period = self._angle_maximum - self._angle_minimum
        if self._angle_period <= 0.0:
            raise ValueError("angle range must be increasing")
        self._spatial_code_length = spatial_encoder.code_length
        self.code_length = self._spatial_code_length + angle_encoder.code_length
        self.colorizer = EncoderCodeColorizer(
            min_hue=color_minimum_hue,
            max_hue=color_maximum_hue,
            start_bit=self._spatial_code_length + color_angle_start,
            bit_length=color_angle_length,
        )
        self._center_lookup = {
            self._coordinate_key(chart.center_x, chart.center_y): index
            for index, chart in enumerate(topology.charts)
        }
        self._cache_size = max(0, int(cache_size))
        self._code_cache: OrderedDict[
            tuple[int, float], tuple[OrientationEncoding, BitArray]
        ] = OrderedDict()
        LOGGER.event(
            "orientation.encoder.init",
            section=MULTI_DIM_CODES,
            data={
                "positions": len(topology.charts),
                "spatial_bits": spatial_encoder.code_length,
                "angular_bits": angle_encoder.code_length,
                "code_length": self.code_length,
                "angle_period": self._angle_period,
                "orientation_not_direction": True,
                "cache_size": self._cache_size,
            },
        )

    @property
    def spatial_code_length(self) -> int:
        return self._spatial_code_length

    @property
    def angular_code_length(self) -> int:
        return self._angle_encoder.code_length

    @staticmethod
    def _coordinate_key(x: float, y: float) -> tuple[float, float]:
        return round(float(x), 12), round(float(y), 12)

    def center_index(self, x: float, y: float) -> int:
        key = self._coordinate_key(x, y)
        index = self._center_lookup.get(key)
        if index is None:
            raise ValueError(f"unknown retinotopic position: ({x}, {y})")
        return index

    def describe(self, angle: float, *, center_index: int) -> OrientationEncoding:
        chart = self.topology.charts[center_index]
        source_angle = self._angle_minimum + (
            (float(angle) - self._angle_minimum) % self._angle_period
        )
        polar_angle = 2.0 * math.pi * (
            source_angle - self._angle_minimum
        ) / self._angle_period
        position_x = chart.center_x + chart.radius * math.cos(polar_angle)
        position_y = chart.center_y + chart.radius * math.sin(polar_angle)
        encoded_angle, phase = self.topology.encode_angle(
            source_angle,
            center_index=center_index,
            position_x=position_x,
            position_y=position_y,
        )
        return OrientationEncoding(
            center_index=center_index,
            center_x=chart.center_x,
            center_y=chart.center_y,
            position_x=position_x,
            position_y=position_y,
            source_angle=source_angle,
            encoded_angle=encoded_angle,
            chirality=chart.chirality,
            phase=phase,
            radius=chart.radius,
        )

    def encode_descriptor(self, descriptor: OrientationEncoding) -> BitArray:
        _, spatial_indices = self._spatial_encoder.encode_indices(
            descriptor.position_x, descriptor.position_y
        )
        _, angular_indices = self._angle_encoder.encode_indices(
            descriptor.encoded_angle
        )
        code = BitArray.from_indices(
            self.code_length,
            (
                *spatial_indices,
                *(self._spatial_code_length + index for index in angular_indices),
            ),
        )
        LOGGER.event_lazy(
            "orientation.encoder.code",
            section=MULTI_DIM_CODES,
            data=lambda: {
                "position": descriptor.center_index,
                "source_angle": descriptor.source_angle,
                "encoded_angle": descriptor.encoded_angle,
                "chirality": descriptor.chirality,
                "phase": descriptor.phase,
                "active_spatial_bits": len(spatial_indices),
                "active_angular_bits": len(angular_indices),
                "active_bits": code.count(),
            },
        )
        return code

    def encode_at(self, angle: float, *, center_index: int) -> tuple[OrientationEncoding, BitArray]:
        source_angle = self._angle_minimum + (
            (float(angle) - self._angle_minimum) % self._angle_period
        )
        cache_key = (int(center_index), source_angle)
        cached = self._code_cache.get(cache_key)
        if cached is not None:
            self._code_cache.move_to_end(cache_key)
            return cached
        descriptor = self.describe(angle, center_index=center_index)
        result = descriptor, self.encode_descriptor(descriptor)
        if self._cache_size > 0:
            self._code_cache[cache_key] = result
            if len(self._code_cache) > self._cache_size:
                self._code_cache.popitem(last=False)
        return result

    def encode_sample(self, sample: object) -> BitArray:
        angle = float(getattr(sample, "angle"))
        center_index = int(getattr(sample, "position_index"))
        return self.encode_at(angle, center_index=center_index)[1]

    def encode(self, angle: float, x: float, y: float, **_: object) -> tuple[tuple[float, ...], BitArray]:
        descriptor, code = self.encode_at(
            angle,
            center_index=self.center_index(x, y),
        )
        LOGGER.event(
            "orientation.encoder.input",
            section=CYCLIC_COORDS,
            data={
                "angle": angle,
                "x": x,
                "y": y,
                "encoded_angle": descriptor.encoded_angle,
            },
        )
        return (descriptor.encoded_angle, float(x), float(y)), code

    def encode_many(
        self, measurements: Sequence[tuple[float, int, int]]
    ) -> list[tuple[tuple[float, ...], BitArray]]:
        encoded: list[tuple[tuple[float, ...], BitArray]] = []
        for angle, x, y in measurements:
            encoded.append(self.encode(float(angle), float(x), float(y)))
        LOGGER.event(
            "orientation.encoder.batch",
            section=MULTI_DIM_CODES,
            data={
                "measurements": len(measurements),
                "cache_entries": len(self._code_cache),
            },
        )
        return encoded

    def active_component_counts(self, descriptor: OrientationEncoding) -> tuple[int, int]:
        _, spatial = self._spatial_encoder.encode_indices(
            descriptor.position_x, descriptor.position_y
        )
        _, angular = self._angle_encoder.encode_indices(descriptor.encoded_angle)
        return len(spatial), len(angular)

    def active_sample_component_counts(self, sample: object) -> tuple[int, int]:
        descriptor = self.describe(
            float(getattr(sample, "angle")),
            center_index=int(getattr(sample, "position_index")),
        )
        return self.active_component_counts(descriptor)


class RetinotopicOrientationEncoderFactory:
    """Создаёт одинаковый ориентационный Encoder для приложения и тестов."""

    @staticmethod
    def create(
        centers: Sequence[tuple[float, float]],
        *,
        spatial_bounds: tuple[tuple[float, float], tuple[float, float]],
        spatial_layers: Sequence[tuple[tuple[int, ...], tuple[float, ...]]],
        angle_layers: Sequence[tuple[int, float]],
        angle_minimum: float,
        angle_maximum: float,
        radius_factor: float,
        isolated_radius: float,
        color_minimum_hue: float,
        color_maximum_hue: float,
        random_seed: int,
        log_every: int,
        cache_size: int = 0,
        topology: ContinuousPinwheelTopology | None = None,
    ) -> RetinotopicOrientationEncoder:
        if not centers:
            raise ValueError("centers must not be empty")
        if not spatial_layers:
            raise ValueError("spatial_layers must not be empty")
        if not angle_layers:
            raise ValueError("angle_layers must not be empty")
        resolved_topology = topology or ContinuousPinwheelTopology(
            centers,
            angle_period=angle_maximum - angle_minimum,
            radius_factor=radius_factor,
            isolated_radius=isolated_radius,
        )
        x_bounds, y_bounds = spatial_bounds
        spatial_encoder = JointEncoder(
            OpenedDimension("x", x_bounds, [Detectors(1, 0.0)]),
            OpenedDimension("y", y_bounds, [Detectors(1, 0.0)]),
            joint_layers=[
                JointDetectorLayer(counts=counts, overlaps=overlaps)
                for counts, overlaps in spatial_layers
            ],
            random_bit=False,
            random_seed=random_seed,
            log_every=log_every,
        )
        angle_encoder = Encoder(
            ClosedDimension(
                "orientation",
                (angle_minimum, angle_maximum),
                [Detectors(count, overlap) for count, overlap in angle_layers],
            ),
            random_bit=False,
            random_seed=random_seed,
            log_every=log_every,
        )
        color_angle_start = sum(count for count, _ in angle_layers[:-1])
        return RetinotopicOrientationEncoder(
            resolved_topology,
            spatial_encoder,
            angle_encoder,
            angle_minimum=angle_minimum,
            angle_maximum=angle_maximum,
            color_minimum_hue=color_minimum_hue,
            color_maximum_hue=color_maximum_hue,
            color_angle_start=color_angle_start,
            color_angle_length=angle_layers[-1][0],
            cache_size=cache_size,
        )
