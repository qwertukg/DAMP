from __future__ import annotations

import colorsys
from dataclasses import dataclass
import math
from typing import Sequence

from damp.activation.activation import ActivationEngine, ActivationResult, Detector
from damp.article_refs import DETECTOR_HIERARCHY, STIMULUS_DETECTION
from damp.encoding.bitarray import BitArray
from damp.encoding.frequency_pruner import SparseCodeFrequencyPruner
from damp.layout.damp_layout import Layout
from damp.logging import LOGGER


@dataclass(frozen=True)
class DetectorSpace:
    """Один последовательный уровень Layout → detector hierarchy (§6.2, §6.5)."""

    index: int
    engine: ActivationEngine
    input_pruner: SparseCodeFrequencyPruner | None = None

    def activate(self, stimuli: Sequence[BitArray]) -> ActivationResult:
        if not stimuli:
            raise ValueError("detector space stimuli must be non-empty")
        resolved_stimuli = (
            tuple(stimuli)
            if self.input_pruner is None
            else self.input_pruner.transform_many(stimuli)
        )
        result = self.engine.activate(resolved_stimuli)
        LOGGER.event(
            "activation.detector_space",
            section=DETECTOR_HIERARCHY,
            data={
                "space": self.index,
                "input_codes": len(stimuli),
                "input_bits": sum(code.count() for code in stimuli),
                "resolved_input_bits": sum(
                    code.count() for code in resolved_stimuli
                ),
                "active_detectors": len(result.active_detectors),
                "output_bits": result.embedding.count(),
            },
        )
        return result


@dataclass(frozen=True)
class DetectorSpaceCascadeResult:
    levels: tuple[ActivationResult, ...]

    @property
    def embedding(self) -> BitArray:
        if not self.levels:
            raise RuntimeError("detector space cascade has no levels")
        return self.levels[-1].embedding


class DetectorSpaceCascade:
    """Передаёт структурный эмбеддинг уровня N как стимул уровня N+1 (§6.2, §6.6)."""

    def __init__(self, spaces: Sequence[DetectorSpace]) -> None:
        if not spaces:
            raise ValueError("at least one detector space is required")
        indices = [space.index for space in spaces]
        if indices != list(range(1, len(spaces) + 1)):
            raise ValueError("detector spaces must be numbered consecutively from one")
        self._spaces = tuple(spaces)
        LOGGER.event(
            "activation.detector_space.cascade.init",
            section=DETECTOR_HIERARCHY,
            data={"spaces": len(self._spaces)},
        )

    @property
    def spaces(self) -> tuple[DetectorSpace, ...]:
        return self._spaces

    def activate(self, stimuli: Sequence[BitArray]) -> DetectorSpaceCascadeResult:
        current = tuple(stimuli)
        levels: list[ActivationResult] = []
        for space in self._spaces:
            result = space.activate(current)
            levels.append(result)
            current = (result.embedding,)
        cascade_result = DetectorSpaceCascadeResult(tuple(levels))
        LOGGER.event(
            "activation.detector_space.cascade",
            section=STIMULUS_DETECTION,
            data={
                "spaces": len(levels),
                "input_codes": len(stimuli),
                "output_bits": cascade_result.embedding.count(),
            },
        )
        return cascade_result


class StructuralEmbeddingHueResolver:
    """Восстанавливает фазу кода из топологии рецептивных полей (§6.2, §6.5)."""

    def __init__(
        self,
        *,
        minimum: float,
        maximum: float,
        bit_phases: Sequence[float | None] | None = None,
    ) -> None:
        if maximum <= minimum:
            raise ValueError("maximum hue must be greater than minimum hue")
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._period = self._maximum - self._minimum
        self._bit_phases = (
            None
            if bit_phases is None
            else tuple(
                None if value is None else float(value) % self._period
                for value in bit_phases
            )
        )

    @classmethod
    def from_detector_space(
        cls,
        *,
        layout: Layout,
        detectors: Sequence[Detector],
        minimum: float,
        maximum: float,
    ) -> "StructuralEmbeddingHueResolver":
        topology = _DetectorOutputTopology(
            layout=layout,
            detectors=detectors,
            period=float(maximum) - float(minimum),
        )
        phases = topology.bit_phases()
        LOGGER.event(
            "activation.detector_space.output_topology",
            section=DETECTOR_HIERARCHY,
            data={
                "detectors": len(detectors),
                "output_bits": len(phases),
                "mapped_bits": sum(value is not None for value in phases),
                "layout_points": len(layout.positions()),
            },
        )
        return cls(
            minimum=minimum,
            maximum=maximum,
            bit_phases=phases,
        )

    def resolve(self, code: BitArray) -> float:
        active = code.active_indices()
        if not active:
            return self._minimum
        phases = self._active_phases(active, len(code))
        sin_sum = sum(math.sin(2.0 * math.pi * phase / self._period) for phase in phases)
        cos_sum = sum(math.cos(2.0 * math.pi * phase / self._period) for phase in phases)
        phase = math.atan2(sin_sum, cos_sum) % (2.0 * math.pi)
        return self._minimum + (
            phase / (2.0 * math.pi)
        ) * (self._maximum - self._minimum)

    def _active_phases(self, active: Sequence[int], code_length: int) -> tuple[float, ...]:
        if self._bit_phases is not None:
            mapped = tuple(
                self._bit_phases[index]
                for index in active
                if index < len(self._bit_phases)
                and self._bit_phases[index] is not None
            )
            if mapped:
                return tuple(float(value) for value in mapped)
        denominator = max(1, code_length)
        return tuple(self._period * index / denominator for index in active)


class OrientationPreferenceHueResolver:
    """Определяет предпочтительную ориентацию по ответам на Encoder-пробы (§6.5)."""

    def __init__(
        self,
        *,
        probes: Sequence[tuple[float, BitArray]],
        orientation_period: float,
        minimum: float,
        maximum: float,
        temperature: float,
    ) -> None:
        if not probes:
            raise ValueError("orientation probes must be non-empty")
        if orientation_period <= 0.0 or maximum <= minimum or temperature <= 0.0:
            raise ValueError("invalid orientation preference parameters")
        lengths = {len(code) for _, code in probes}
        if len(lengths) != 1:
            raise ValueError("orientation probe code lengths must match")
        self._probes = tuple(
            (
                float(angle) % float(orientation_period),
                frozenset(code.active_indices()),
            )
            for angle, code in probes
        )
        self._code_length = lengths.pop()
        self._orientation_period = float(orientation_period)
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._temperature = float(temperature)
        LOGGER.event(
            "activation.detector_space.orientation_preference.init",
            section=STIMULUS_DETECTION,
            data={
                "probes": len(self._probes),
                "orientation_period": self._orientation_period,
                "temperature": self._temperature,
                "code_length": self._code_length,
            },
        )

    def resolve(self, code: BitArray) -> float:
        if len(code) != self._code_length:
            raise ValueError("code length must match orientation probes")
        active = frozenset(code.active_indices())
        if not active:
            return self._minimum
        similarities = [self._jaccard(active, probe) for _, probe in self._probes]
        maximum_similarity = max(similarities, default=0.0)
        weights = [
            math.exp((value - maximum_similarity) / self._temperature)
            for value in similarities
        ]
        sin_sum = 0.0
        cos_sum = 0.0
        for (angle, _), weight in zip(self._probes, weights):
            phase = 2.0 * math.pi * angle / self._orientation_period
            sin_sum += math.sin(phase) * weight
            cos_sum += math.cos(phase) * weight
        if abs(sin_sum) <= 1e-12 and abs(cos_sum) <= 1e-12:
            best = similarities.index(maximum_similarity)
            phase = 2.0 * math.pi * self._probes[best][0] / self._orientation_period
        else:
            phase = math.atan2(sin_sum, cos_sum) % (2.0 * math.pi)
        return self._minimum + (
            phase / (2.0 * math.pi)
        ) * (self._maximum - self._minimum)

    @staticmethod
    def _jaccard(left: frozenset[int], right: frozenset[int]) -> float:
        common = len(left & right)
        union = len(left) + len(right) - common
        return 0.0 if union <= 0 else common / union


class CorticalOrientationPreferenceMap:
    """Карта предпочтений D2 из перекрывающихся рецептивных полей (§6.3–§6.5)."""

    def __init__(
        self,
        *,
        orientation_period: float,
        hue_minimum: float,
        hue_maximum: float,
        kernel_sigma_factor: float,
        minimum_selectivity: float,
    ) -> None:
        if orientation_period <= 0.0 or hue_maximum <= hue_minimum:
            raise ValueError("invalid orientation map ranges")
        if kernel_sigma_factor <= 0.0 or not 0.0 <= minimum_selectivity <= 1.0:
            raise ValueError("invalid orientation map smoothing parameters")
        self._orientation_period = float(orientation_period)
        self._hue_minimum = float(hue_minimum)
        self._hue_maximum = float(hue_maximum)
        self._kernel_sigma_factor = float(kernel_sigma_factor)
        self._minimum_selectivity = float(minimum_selectivity)

    def calibrate_and_render(
        self,
        *,
        engine: ActivationEngine,
        probes: Sequence[tuple[float, BitArray]],
    ) -> tuple[tuple[tuple[float, BitArray], ...], "object"]:
        """Возвращает C2-пробы и поле фазы, не используя угол как цвет точки."""
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("NumPy is required for orientation preference map") from exc
        if not probes:
            raise ValueError("orientation probes must be non-empty")
        detectors = engine.detectors
        detector_indices = {id(detector): index for index, detector in enumerate(detectors)}
        sin_tuning = np.zeros(len(detectors), dtype=np.float64)
        cos_tuning = np.zeros(len(detectors), dtype=np.float64)
        response_sum = np.zeros(len(detectors), dtype=np.float64)
        resolved: list[tuple[float, BitArray]] = []
        for angle, code in probes:
            result = engine.activate((code,))
            if result.embedding.count() > 0:
                resolved.append((float(angle), result.embedding))
            phase = 2.0 * math.pi * (float(angle) % self._orientation_period)
            phase /= self._orientation_period
            for activated in result.active_detectors:
                index = detector_indices.get(id(activated.detector))
                if index is None:
                    continue
                weight = max(0.0, float(activated.activation_level))
                sin_tuning[index] += weight * math.sin(phase)
                cos_tuning[index] += weight * math.cos(phase)
                response_sum[index] += weight
        magnitude = np.hypot(sin_tuning, cos_tuning)
        selectivity = np.divide(
            magnitude,
            response_sum,
            out=np.zeros_like(magnitude),
            where=response_sum > 0.0,
        )
        height = engine.layout.height
        width = engine.layout.width
        field_sin = np.zeros((height, width), dtype=np.float64)
        field_cos = np.zeros((height, width), dtype=np.float64)
        field_weight = np.zeros((height, width), dtype=np.float64)
        for index, detector in enumerate(detectors):
            if response_sum[index] <= 0.0 or selectivity[index] < self._minimum_selectivity:
                continue
            center_y, center_x = detector.center
            major = max(float(detector.radius), 1e-6)
            minor = max(
                float(
                    major
                    if detector.semi_minor_radius is None
                    else detector.semi_minor_radius
                ),
                1e-6,
            )
            support = major
            y_min = max(0, int(math.floor(center_y - support)))
            y_max = min(height - 1, int(math.ceil(center_y + support)))
            x_min = max(0, int(math.floor(center_x - support)))
            x_max = min(width - 1, int(math.ceil(center_x + support)))
            cos_axis = math.cos(float(detector.orientation))
            sin_axis = math.sin(float(detector.orientation))
            sigma = self._kernel_sigma_factor
            for y in range(y_min, y_max + 1):
                dy = float(y) - center_y
                for x in range(x_min, x_max + 1):
                    dx = float(x) - center_x
                    along = dx * cos_axis + dy * sin_axis
                    across = -dx * sin_axis + dy * cos_axis
                    normalized_sq = (along / major) ** 2 + (across / minor) ** 2
                    if normalized_sq > 1.0:
                        continue
                    weight = float(selectivity[index]) * math.exp(
                        -0.5 * normalized_sq / (sigma * sigma)
                    )
                    field_sin[y, x] += weight * sin_tuning[index] / magnitude[index]
                    field_cos[y, x] += weight * cos_tuning[index] / magnitude[index]
                    field_weight[y, x] += weight
        image = np.zeros((height, width, 3), dtype=np.uint8)
        covered = field_weight > 0.0
        phases = np.mod(np.arctan2(field_sin, field_cos), 2.0 * math.pi)
        hues = self._hue_minimum + phases / (2.0 * math.pi) * (
            self._hue_maximum - self._hue_minimum
        )
        for y, x in zip(*np.nonzero(covered)):
            red, green, blue = colorsys.hsv_to_rgb(
                float(hues[y, x] % 360.0) / 360.0,
                1.0,
                1.0,
            )
            image[y, x] = (
                int(round(red * 255.0)),
                int(round(green * 255.0)),
                int(round(blue * 255.0)),
            )
        LOGGER.event(
            "activation.detector_space.orientation_map",
            section=DETECTOR_HIERARCHY,
            data={
                "space": 2,
                "detectors": len(detectors),
                "tuned_detectors": int(np.count_nonzero(response_sum)),
                "selective_detectors": int(
                    np.count_nonzero(selectivity >= self._minimum_selectivity)
                ),
                "covered_pixels": int(np.count_nonzero(covered)),
                "width": width,
                "height": height,
            },
            visuals=[
                LOGGER.visual_image(
                    "activation/detector_space_2/orientation_preference",
                    image,
                    static=True,
                )
            ],
        )
        return tuple(resolved), image


class StructuralEmbeddingColorizer:
    """Показывает вычисленную из топологии фазу без потери насыщенности (§3, §6.2)."""

    def __init__(
        self,
        resolver: StructuralEmbeddingHueResolver | OrientationPreferenceHueResolver,
    ) -> None:
        self._resolver = resolver

    def color(self, code: BitArray) -> tuple[int, int, int]:
        return self._color_hue(self._resolver.resolve(code), active_bits=code.count())

    def color_hue(self, hue: float) -> tuple[int, int, int]:
        return self._color_hue(hue, active_bits=None)

    @staticmethod
    def _color_hue(
        hue: float,
        *,
        active_bits: int | None,
    ) -> tuple[int, int, int]:
        hue = float(hue) % 360.0
        red, green, blue = colorsys.hsv_to_rgb(hue / 360.0, 1.0, 1.0)
        color = (
            int(round(red * 255.0)),
            int(round(green * 255.0)),
            int(round(blue * 255.0)),
        )
        LOGGER.event(
            "activation.detector_space.color",
            section=DETECTOR_HIERARCHY,
            data={
                "hue": hue,
                "active_bits": -1 if active_bits is None else active_bits,
                "saturation": 1.0,
                "value": 1.0,
            },
        )
        return color


class _DetectorOutputTopology:
    """Проецирует координаты центров детекторов на их выходные биты (§6.2–6.5)."""

    def __init__(
        self,
        *,
        layout: Layout,
        detectors: Sequence[Detector],
        period: float,
    ) -> None:
        if not detectors:
            raise ValueError("detectors must be non-empty")
        if period <= 0.0:
            raise ValueError("period must be positive")
        self._positions = tuple(layout.positions())
        if not self._positions:
            raise ValueError("layout positions must be non-empty")
        self._detectors = tuple(detectors)
        self._period = float(period)
        self._center_y = sum(float(y) for y, _ in self._positions) / len(
            self._positions
        )
        self._center_x = sum(float(x) for _, x in self._positions) / len(
            self._positions
        )

    def bit_phases(self) -> tuple[float | None, ...]:
        output_lengths = {len(detector.output_code) for detector in self._detectors}
        if len(output_lengths) != 1:
            raise ValueError("detector output code lengths must match")
        output_length = output_lengths.pop()
        phase_vectors: list[list[tuple[float, float]]] = [
            [] for _ in range(output_length)
        ]
        for detector in self._detectors:
            phase = self._detector_phase(detector)
            weight = max(1e-12, float(detector.energy))
            for bit in detector.output_code.active_indices():
                phase_vectors[bit].append((phase, weight))
        return tuple(
            self._circular_mean(values) if values else None
            for values in phase_vectors
        )

    def _detector_phase(self, detector: Detector) -> float:
        center_y, center_x = detector.center
        angle = math.atan2(
            float(center_y) - self._center_y,
            float(center_x) - self._center_x,
        )
        return (angle % (2.0 * math.pi)) * self._period / (2.0 * math.pi)

    def _circular_mean(self, values: Sequence[tuple[float, float]]) -> float:
        sin_sum = 0.0
        cos_sum = 0.0
        for phase, weight in values:
            angle = 2.0 * math.pi * float(phase) / self._period
            sin_sum += math.sin(angle) * float(weight)
            cos_sum += math.cos(angle) * float(weight)
        if abs(sin_sum) <= 1e-12 and abs(cos_sum) <= 1e-12:
            return float(values[0][0]) % self._period
        return (
            math.atan2(sin_sum, cos_sum) % (2.0 * math.pi)
        ) * self._period / (2.0 * math.pi)
