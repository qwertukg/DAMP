from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerConfig:
    detectors: int
    overlap: float
