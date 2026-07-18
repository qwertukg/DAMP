from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import random
from typing import Sequence, Tuple

from damp.article_refs import CYCLIC_COORDS, MULTI_DIM_CODES, SPARSE_BIT_VECTORS, WIDE_DETECTORS
from damp.encoding.bitarray import BitArray
from damp.encoding.damp_encoder import DetectorWindow, Detectors, Dimension, Encoder
from damp.logging import LOGGER


@dataclass(frozen=True)
class JointDetectorLayer:
    counts: tuple[int, ...]
    overlaps: tuple[float, ...]

    def validate(self, dimension_count: int) -> None:
        if len(self.counts) != dimension_count or len(self.overlaps) != dimension_count:
            raise ValueError("joint layer must define count and overlap for every dimension")
        for count, overlap in zip(self.counts, self.overlaps):
            Detectors(count, overlap)


@dataclass(frozen=True)
class JointDetectorWindow:
    detector_index: int
    receptive_fields: tuple[DetectorWindow, ...]

    def is_active(self, values: Sequence[float]) -> bool:
        return all(
            receptive_field.is_active(value)
            for receptive_field, value in zip(self.receptive_fields, values)
        )


class JointEncoder(Encoder):
    """Wide detectors whose receptive fields jointly cover an n-D space (§4.6)."""

    def __init__(
        self,
        *dimensions: Dimension,
        joint_layers: Sequence[JointDetectorLayer],
        random_bit: bool = False,
        random_seed: int | None = None,
        log_every: int | None = None,
    ) -> None:
        super().__init__(*dimensions, random_bit=False, random_seed=random_seed, log_every=0)
        if not joint_layers:
            raise ValueError("joint_layers must be non-empty")
        self._joint_detectors: list[JointDetectorWindow] = []
        for layer_index, layer in enumerate(joint_layers):
            layer.validate(len(self.dimensions))
            fields_by_dimension: list[list[DetectorWindow]] = []
            for dimension_index, (dimension, count, overlap) in enumerate(
                zip(self.dimensions, layer.counts, layer.overlaps)
            ):
                fields_by_dimension.append(
                    self._build_layer_detectors(
                        dim=dimension,
                        dim_index=dimension_index,
                        layer_index=layer_index,
                        layer=Detectors(count, overlap),
                        start_index=0,
                    )
                )
            for receptive_fields in product(*fields_by_dimension):
                self._joint_detectors.append(
                    JointDetectorWindow(
                        detector_index=len(self._joint_detectors),
                        receptive_fields=tuple(receptive_fields),
                    )
                )
        self.code_length = len(self._joint_detectors)
        self._rng = random.Random(random_seed)
        self._bit_map = (
            [self._rng.randrange(self.code_length) for _ in range(self.code_length)]
            if random_bit
            else list(range(self.code_length))
        )
        self.log_every = int(log_every) if log_every is not None else 0
        self._log_counter = 0
        LOGGER.event(
            "encoder.joint.init",
            section=MULTI_DIM_CODES,
            data={
                "dimensions": len(self.dimensions),
                "layers": len(joint_layers),
                "detectors": self.code_length,
                "random_bit": random_bit,
                "random_seed": random_seed,
            },
        )

    def encode(self, *values: float, **_: object) -> Tuple[Tuple[float, ...], BitArray]:
        if len(values) != len(self.dimensions):
            raise ValueError("number of values must match number of dimensions")
        LOGGER.event(
            "encoder.joint.input",
            section=MULTI_DIM_CODES,
            data={"values": values},
        )
        normalized_values: list[float] = []
        for value, dimension in zip(values, self.dimensions):
            normalized = float(value)
            if dimension.closed:
                normalized = (
                    (normalized - dimension.min_value) % dimension.period
                ) + dimension.min_value
            elif not dimension.min_value <= normalized <= dimension.max_value:
                raise ValueError(
                    f"value {normalized} is outside the open dimension range "
                    f"[{dimension.min_value}, {dimension.max_value}]"
                )
            normalized_values.append(normalized)
        LOGGER.event(
            "encoder.joint.normalized",
            section=CYCLIC_COORDS,
            data={"values": normalized_values},
        )
        code = BitArray(self.code_length)
        for detector in self._joint_detectors:
            if detector.is_active(normalized_values):
                code.set(self._bit_map[detector.detector_index], 1)
        self._log_counter += 1
        LOGGER.event(
            "encoder.joint.code",
            section=SPARSE_BIT_VECTORS,
            data={
                "code_length": self.code_length,
                "active_bits": code.count(),
                "code": code,
                "joint_detectors": len(self._joint_detectors),
            },
        )
        LOGGER.event(
            "encoder.joint.detectors",
            section=WIDE_DETECTORS,
            data={"active": code.count(), "total": self.code_length},
        )
        return tuple(normalized_values), code
