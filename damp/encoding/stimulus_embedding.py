from __future__ import annotations

from typing import Sequence

from damp.article_refs import COLOR_MERGE
from damp.encoding.bitarray import BitArray
from damp.logging import LOGGER


class StimulusEmbeddingBuilder:
    """Colour merge набора первичных кодов с бюджетом насыщенности (§3.3)."""

    def __init__(self, *, code_length: int, spatial_priority_bits: int, max_density: float) -> None:
        if code_length <= 0:
            raise ValueError("code_length must be positive")
        if not 0 <= spatial_priority_bits <= code_length:
            raise ValueError("spatial_priority_bits must fit code_length")
        if not 0.0 < max_density <= 1.0:
            raise ValueError("max_density must be in (0, 1]")
        self._code_length = int(code_length)
        self._spatial_priority_bits = int(spatial_priority_bits)
        self._saturation_limit = max(1, int(self._code_length * max_density))

    @property
    def saturation_limit(self) -> int:
        return self._saturation_limit

    def build(self, codes: Sequence[BitArray]) -> BitArray:
        frequencies = [0] * self._code_length
        for code in codes:
            if len(code) != self._code_length:
                raise ValueError("all stimulus codes must have the configured length")
            for index, bit in enumerate(code):
                if bit:
                    frequencies[index] += 1
        active = [index for index, frequency in enumerate(frequencies) if frequency > 0]
        active.sort(
            key=lambda index: (
                index >= self._spatial_priority_bits,
                -frequencies[index],
                index,
            )
        )
        selected = active[: self._saturation_limit]
        embedding = BitArray(self._code_length)
        for index in selected:
            embedding.set(index)
        LOGGER.event(
            "encoder.stimulus.color_merge",
            section=COLOR_MERGE,
            data={
                "stimuli": len(codes),
                "union_bits": len(active),
                "selected_bits": len(selected),
                "saturation_limit": self._saturation_limit,
                "density": len(selected) / self._code_length,
                "spatial_priority_bits": self._spatial_priority_bits,
            },
        )
        return embedding
