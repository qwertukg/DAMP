from __future__ import annotations

from dataclasses import dataclass

from .bit_array import BitArray


@dataclass(frozen=True)
class LayoutPoint:
    code: BitArray
    angle: float
    hue: float
    ones: int
