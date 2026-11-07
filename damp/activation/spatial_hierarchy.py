from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from damp.activation.activation import ActivationResult
from damp.activation.detector_space import DetectorSpace
from damp.article_refs import DETECTOR_HIERARCHY, STIMULUS_DETECTION
from damp.encoding.bitarray import BitArray
from damp.encoding.spatial_composition import (
    LocalOrientationWindowEncoder, SpatialCompositionEncoder, SpatialFeature,
)
from damp.logging import LOGGER


@dataclass(frozen=True)
class SpatialHierarchyResult:
    local_codes: tuple[SpatialFeature, ...]
    parts: tuple[SpatialFeature, ...]
    composition: BitArray
    activation: ActivationResult

    @property
    def embedding(self) -> BitArray:
        return self.activation.embedding


class SpatialDetectorHierarchy:
    """Углопозиции → локальные детали → пространственные композиции (§6.2, §6.6)."""

    def __init__(
        self, *, window_encoder: LocalOrientationWindowEncoder,
        composition_encoder: SpatialCompositionEncoder,
        spaces: Sequence[DetectorSpace],
    ) -> None:
        if tuple(space.index for space in spaces) != (1, 2, 3):
            raise ValueError("spatial hierarchy requires primitive, part and composition spaces")
        self.window_encoder = window_encoder
        self.composition_encoder = composition_encoder
        self.spaces = tuple(spaces)

    @staticmethod
    def encode_local(
        measurements: Sequence[tuple[float, int, int]],
        window_encoder: LocalOrientationWindowEncoder,
        space: DetectorSpace,
    ) -> tuple[SpatialFeature, ...]:
        features = tuple(
            SpatialFeature(window.x, window.y, space.activate(window.codes).embedding)
            for window in window_encoder.encode(measurements)
        )
        LOGGER.event(
            "hierarchy.local.encode", section=DETECTOR_HIERARCHY,
            data={"level": space.index, "windows": len(features),
                  "nonempty": sum(feature.code.count() > 0 for feature in features)},
        )
        return features

    @staticmethod
    def recognize_parts(
        local_codes: Sequence[SpatialFeature], space: DetectorSpace,
    ) -> tuple[SpatialFeature, ...]:
        parts = tuple(
            SpatialFeature(feature.x, feature.y, space.activate((feature.code,)).embedding)
            for feature in local_codes
        )
        LOGGER.event(
            "hierarchy.parts.encode", section=DETECTOR_HIERARCHY,
            data={"level": space.index, "parts": len(parts),
                  "nonempty": sum(part.code.count() > 0 for part in parts)},
        )
        return parts

    def activate_measurements(
        self, measurements: Sequence[tuple[float, int, int]],
    ) -> SpatialHierarchyResult:
        local_codes = self.encode_local(measurements, self.window_encoder, self.spaces[0])
        parts = self.recognize_parts(local_codes, self.spaces[1])
        composition = self.composition_encoder.build(parts)
        activation = self.spaces[2].activate((composition,))
        result = SpatialHierarchyResult(local_codes, parts, composition, activation)
        LOGGER.event(
            "hierarchy.image.encode", section=STIMULUS_DETECTION,
            data={"levels": len(self.spaces), "measurements": len(measurements),
                  "local_windows": len(local_codes), "parts": len(parts),
                  "composition_bits": composition.count(),
                  "embedding_bits": result.embedding.count()},
        )
        return result
