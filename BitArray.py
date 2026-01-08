from __future__ import annotations
from dataclasses import dataclass


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


@dataclass(frozen=True)
class LayoutPoint:
    code: BitArray
    angle: float
    hue: float
    ones: int