from __future__ import annotations

from dataclasses import dataclass

from BitArray import BitArray


@dataclass(frozen=True)
class LayoutPoint:
    code: BitArray
    angle: float
    hue: float
    ones: int
