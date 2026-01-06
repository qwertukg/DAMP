from __future__ import annotations

import math
import random
from typing import Literal, Sequence

from .bit_array import BitArray
from .layer_config import LayerConfig

BitAssignment = Literal["random", "detector_index"]
CoordinateMode = Literal["cyclic", "linear"]


class Encoder:
    """Encode scalar values with layered detectors that overlap by a fraction of span."""

    def __init__(
        self,
        code_bits: int,
        layers: Sequence[LayerConfig],
        *,
        bits_per_detector: int,
        seed: int,
        bit_assignment: BitAssignment = "detector_index",
        coordinate_mode: CoordinateMode = "cyclic",
        value_range: tuple[float, float] | None = None,
        cycle_length: float = 360.0,
    ) -> None:
        if code_bits <= 0:
            raise ValueError("code_bits must be positive")
        if bits_per_detector <= 0:
            raise ValueError("bits_per_detector must be positive")
        if bits_per_detector > code_bits:
            raise ValueError("bits_per_detector must be <= code_bits")

        self.code_bits = code_bits
        self.bits_per_detector = bits_per_detector
        self.layers = list(layers)
        if not self.layers:
            raise ValueError("layers must be non-empty")
        if bit_assignment not in ("random", "detector_index"):
            raise ValueError("bit_assignment must be 'random' or 'detector_index'")
        if coordinate_mode not in ("cyclic", "linear"):
            raise ValueError("coordinate_mode must be 'cyclic' or 'linear'")
        if cycle_length <= 0:
            raise ValueError("cycle_length must be positive")
        if coordinate_mode == "linear":
            if value_range is None:
                raise ValueError("value_range must be set for linear mode")
            if value_range[1] <= value_range[0]:
                raise ValueError("value_range must have max > min")
        self.bit_assignment = bit_assignment
        self._coordinate_mode = coordinate_mode
        self._value_range = value_range
        self._cycle_length = cycle_length

        self._rng = random.Random(seed)
        self._layer_bits = self._init_layer_bits()

    def encode_sparse(self, value: float) -> list[int]:
        """Return indices of active bits for the given coordinate value."""
        bits: set[int] = set()
        for layer, detector_bits in zip(self.layers, self._layer_bits):
            for detector_index in self._active_detectors(value, layer):
                bits.update(detector_bits[detector_index])
        return sorted(bits)

    def encode_dense(self, value: float) -> BitArray:
        """Return a BitArray with only the active bits set."""
        bits = BitArray(self.code_bits)
        for idx in self.encode_sparse(value):
            bits.set(idx)
        return bits

    def _init_layer_bits(self) -> list[list[list[int]]]:
        layers_bits: list[list[list[int]]] = []
        layer_offset = 0
        for layer in self.layers:
            if layer.detectors <= 0:
                raise ValueError("detectors must be positive")
            if layer.overlap < 0:
                raise ValueError("overlap must be >= 0")

            layer_bits: list[list[int]] = []
            for detector_index in range(layer.detectors):
                if self.bit_assignment == "random":
                    bits = self._rng.sample(
                        range(self.code_bits), self.bits_per_detector
                    )
                else:
                    base = layer_offset + detector_index * self.bits_per_detector
                    if base + self.bits_per_detector > self.code_bits:
                        raise ValueError(
                            "code_bits too small for detector_index assignment"
                        )
                    bits = [base + offset for offset in range(self.bits_per_detector)]
                layer_bits.append(bits)
            layers_bits.append(layer_bits)
            if self.bit_assignment == "detector_index":
                layer_offset += layer.detectors * self.bits_per_detector
        return layers_bits

    def _active_detectors(self, value: float, layer: LayerConfig) -> list[int]:
        detectors = layer.detectors
        overlap = layer.overlap
        span = 1.0 + overlap
        if span <= 0.0:
            return []
        if span >= detectors:
            return list(range(detectors))

        if self._coordinate_mode == "cyclic":
            normalized = value % self._cycle_length
            position = normalized / self._cycle_length * detectors
            max_idx = math.floor(position + 1e-9)
            start_idx = math.floor(position - span + 1e-12) + 1
            return [(idx % detectors) for idx in range(start_idx, max_idx + 1)]

        min_value, max_value = self._value_range if self._value_range else (0.0, 1.0)
        normalized = (value - min_value) / (max_value - min_value)
        position = normalized * detectors
        max_idx = math.floor(position + 1e-9)
        start_idx = math.floor(position - span + 1e-12) + 1
        if max_idx < 0 or start_idx >= detectors:
            return []
        start_idx = max(start_idx, 0)
        max_idx = min(max_idx, detectors - 1)
        return list(range(start_idx, max_idx + 1))
