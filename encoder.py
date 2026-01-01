from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence


@dataclass(frozen=True)
class LayerConfig:
    detectors: int
    window: int


class BitArray:
    def __init__(self, size: int, fill: int = 0) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if fill not in (0, 1):
            raise ValueError("fill must be 0 or 1")
        self._bits = bytearray([fill] * size)

    def set(self, idx: int, value: int = 1) -> None:
        self._bits[idx] = 1 if value else 0

    def count(self) -> int:
        return sum(self._bits)

    def common(self, other: "BitArray") -> int:
        if len(self) != len(other):
            raise ValueError("BitArray sizes must match")
        return sum(a & b for a, b in zip(self._bits, other._bits))

    def to01(self) -> str:
        return "".join("1" if b else "0" for b in self._bits)

    def __len__(self) -> int:
        return len(self._bits)


class Encoder:
    """Encode angles with layered wide detectors using a boundary-closed sliding window."""

    def __init__(
        self,
        code_bits: int,
        layers: Sequence[LayerConfig],
        *,
        bits_per_detector: int,
        seed: int,
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

        self._rng = random.Random(seed)
        self._layer_bits = self._init_layer_bits()

    def encode_sparse(self, angle: float) -> list[int]:
        """Return indices of active bits for the given angle (degrees)."""
        bits: set[int] = set()
        for layer, detector_bits in zip(self.layers, self._layer_bits):
            start = self._start_index(angle, layer.detectors)
            for offset in range(layer.window):
                detector_index = (start + offset) % layer.detectors
                bits.update(detector_bits[detector_index])
        return sorted(bits)

    def encode_dense(self, angle: float) -> BitArray:
        """Return a BitArray with only the active bits set."""
        bits = BitArray(self.code_bits)
        for idx in self.encode_sparse(angle):
            bits.set(idx)
        return bits

    def print_codes(
        self,
        start: float,
        end: float,
        step: float,
    ) -> None:
        """Print 0/1 codes with density and overlap for angles in the inclusive range [start, end]."""
        if step <= 0:
            raise ValueError("step must be positive")

        angle = start
        epsilon = step * 1e-9
        prev_code: BitArray | None = None
        while angle <= end + epsilon:
            code = self.encode_dense(angle)
            ones = code.count()
            density = (ones / len(code)) * 100
            if prev_code is None or ones == 0:
                common_pct = 0.0
            else:
                common = code.common(prev_code)
                common_pct = (common / ones) * 100
            code_str = code.to01()
            print(
                f"{angle:7.3f}: {code_str} density={density:.2f}% common={common_pct:.2f}%"
            )
            prev_code = code
            angle += step

    def _init_layer_bits(self) -> list[list[list[int]]]:
        layers_bits: list[list[list[int]]] = []
        for layer in self.layers:
            if layer.detectors <= 0:
                raise ValueError("detectors must be positive")
            if layer.window <= 0:
                raise ValueError("window must be positive")
            if layer.window > layer.detectors:
                raise ValueError("window must be <= detectors")

            layer_bits: list[list[int]] = []
            for _ in range(layer.detectors):
                bits = self._rng.sample(range(self.code_bits), self.bits_per_detector)
                layer_bits.append(bits)
            layers_bits.append(layer_bits)
        return layers_bits

    @staticmethod
    def _start_index(angle: float, detectors: int) -> int:
        normalized = angle % 360.0
        position = normalized / 360.0 * detectors
        return int(position + 1e-9) % detectors
