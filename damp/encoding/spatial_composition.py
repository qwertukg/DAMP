from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from damp.article_refs import MULTI_DIM_CODES, CODE_SPACE_ACTIVATION
from damp.encoding.bitarray import BitArray
from damp.encoding.joint_encoder import JointEncoder
from damp.encoding.orientation_encoder import RetinotopicOrientationEncoder
from damp.encoding.frequency_pruner import SparseCodeFrequencyPruner
from damp.logging import LOGGER


@dataclass(frozen=True)
class SpatialFeature:
    x: int
    y: int
    code: BitArray


@dataclass(frozen=True)
class OrientationWindow:
    x: int
    y: int
    codes: tuple[BitArray, ...]
    measurements: tuple[tuple[float, int, int], ...]


class LocalOrientationWindowEncoder:
    """Отбирает локальные стимулы, сохраняя абсолютные координаты изображения (§4.3, §6.1)."""

    def __init__(
        self, encoder: RetinotopicOrientationEncoder, *, grid_size: int,
        window_size: int, stride: int,
    ) -> None:
        if window_size <= 0 or window_size % 2 == 0 or window_size > grid_size:
            raise ValueError("window_size must be odd and fit inside the image grid")
        if not 0 < stride <= window_size:
            raise ValueError("stride must cover the image without gaps")
        self.encoder = encoder
        self.grid_size = grid_size
        self.window_size = window_size
        self.stride = stride
        radius = window_size // 2
        centers = list(range(radius, grid_size - radius, stride))
        if centers[-1] != grid_size - radius - 1:
            centers.append(grid_size - radius - 1)
        self.centers = tuple(centers)
        LOGGER.event(
            "hierarchy.windows.init", section=CODE_SPACE_ACTIVATION,
            data={"grid_size": grid_size, "window_size": window_size,
                  "stride": stride, "centers": self.centers},
        )

    def encode(self, measurements: Sequence[tuple[float, int, int]]) -> tuple[OrientationWindow, ...]:
        radius = self.window_size // 2
        windows: list[OrientationWindow] = []
        for y in self.centers:
            for x in self.centers:
                local = [
                    (angle, px, py)
                    for angle, px, py in measurements
                    if abs(px - x) <= radius and abs(py - y) <= radius
                ]
                if not local:
                    continue
                codes = tuple(code for _, code in self.encoder.encode_many(local))
                windows.append(OrientationWindow(x, y, codes, tuple(local)))
                LOGGER.event(
                    "hierarchy.window.encode", section=MULTI_DIM_CODES,
                    data={"x": x, "y": y, "radius": radius,
                          "grid_size": self.grid_size,
                          "measurements": local, "stimuli": len(codes)},
                )
        return tuple(windows)


class SpatialCompositionEncoder:
    """Кортеж кодов деталей в пространственных областях (§2.2.3, §4.6).

    Каждому биту позиционного Encoder соответствует отдельный блок кода детали.
    Объединение происходит внутри блоков: связь «деталь — место» не теряется.
    Перекрытие позиционных детекторов обеспечивает сходство соседних положений.
    """

    def __init__(
        self, position_encoder: JointEncoder, *, feature_bits: int,
        feature_pruner: SparseCodeFrequencyPruner | None = None,
    ) -> None:
        if feature_bits <= 0:
            raise ValueError("feature_bits must be positive")
        self.position_encoder = position_encoder
        self.feature_bits = feature_bits
        self.feature_pruner = feature_pruner
        self.code_length = position_encoder.code_length * feature_bits
        LOGGER.event(
            "hierarchy.composition.init", section=MULTI_DIM_CODES,
            data={"position_blocks": position_encoder.code_length,
                  "feature_bits": feature_bits, "code_length": self.code_length},
        )

    def build(self, features: Sequence[SpatialFeature]) -> BitArray:
        active: set[int] = set()
        for feature in features:
            if len(feature.code) != self.feature_bits:
                raise ValueError("feature code length does not match the spatial tuple")
            _, position_bits = self.position_encoder.encode_indices(feature.x, feature.y)
            feature_code = (
                feature.code if self.feature_pruner is None
                else self.feature_pruner.transform(feature.code)
            )
            active.update(
                position * self.feature_bits + bit
                for position in position_bits for bit in feature_code.active_indices()
            )
            LOGGER.event(
                "hierarchy.composition.binding", section=MULTI_DIM_CODES,
                data={"x": feature.x, "y": feature.y, "position_bits": position_bits,
                      "feature_active_bits": feature.code.count()},
            )
        code = BitArray.from_indices(self.code_length, active)
        LOGGER.event(
            "hierarchy.composition.encode", section=MULTI_DIM_CODES,
            data={"features": len(features), "active_bits": code.count(),
                  "code_length": len(code), "density": code.count() / len(code)},
        )
        return code
