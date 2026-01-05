from __future__ import annotations

import math
import random
from typing import Literal, Sequence

from BitArray import BitArray
from LayerConfig import LayerConfig

BitAssignment = Literal["random", "detector_index"]


class Encoder:
    """Encode angles with layered detectors that overlap by a fraction of their arc length."""

    def __init__(
        self,
        code_bits: int,
        layers: Sequence[LayerConfig],
        *,
        bits_per_detector: int,
        seed: int,
        bit_assignment: BitAssignment = "random",
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
        self.bit_assignment = bit_assignment

        self._rng = random.Random(seed)
        self._layer_bits = self._init_layer_bits()

    def encode_sparse(self, angle: float) -> list[int]:
        """Return indices of active bits for the given angle (degrees)."""
        bits: set[int] = set()
        for layer, detector_bits in zip(self.layers, self._layer_bits):
            for detector_index in self._active_detectors(angle, layer):
                bits.update(detector_bits[detector_index])
        return sorted(bits)

    def encode_dense(self, angle: float) -> BitArray:
        """Return a BitArray with only the active bits set."""
        bits = BitArray(self.code_bits)
        for idx in self.encode_sparse(angle):
            bits.set(idx)
        return bits

    def encode(
        self, start: float, end: float, step: float
    ) -> list[tuple[BitArray, float, float]]:
        """Return (code, angle, hue) tuples for angles in the inclusive range [start, end]."""
        if step <= 0:
            raise ValueError("step must be positive")

        angle = start
        epsilon = step * 1e-9
        codes: list[tuple[BitArray, float, float]] = []
        while angle <= end + epsilon:
            code = self.encode_dense(angle)
            hue = angle % 360.0
            codes.append((code, angle, hue))
            angle += step
        return codes

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

    @staticmethod
    def _active_detectors(angle: float, layer: LayerConfig) -> list[int]:
        detectors = layer.detectors
        overlap = layer.overlap
        normalized = angle % 360.0
        position = normalized / 360.0 * detectors
        span = 1.0 + overlap
        if span <= 0.0:
            return []
        if span >= detectors:
            return list(range(detectors))
        # Detectors cover [j, j + span) in detector units; collect all j that cover position.
        max_idx = math.floor(position + 1e-9)
        start_idx = math.floor(position - span + 1e-12) + 1
        return [(idx % detectors) for idx in range(start_idx, max_idx + 1)]
