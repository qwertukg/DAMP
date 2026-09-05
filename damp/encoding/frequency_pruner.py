from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from damp.article_refs import CLUSTER_PRUNING
from damp.encoding.bitarray import BitArray
from damp.logging import LOGGER


@dataclass(frozen=True)
class SparseCodeFrequencyPruner:
    """Удаляет общие биты, создающие паразитную similarity (§8.3.7)."""

    code_length: int
    kept_indices: frozenset[int]

    def __post_init__(self) -> None:
        if self.code_length <= 0:
            raise ValueError("code_length must be positive")
        if not self.kept_indices:
            raise ValueError("kept_indices must be non-empty")
        if min(self.kept_indices) < 0 or max(self.kept_indices) >= self.code_length:
            raise ValueError("kept index is outside code length")

    @classmethod
    def fit(
        cls,
        codes: Sequence[BitArray],
        *,
        max_document_frequency: float,
    ) -> "SparseCodeFrequencyPruner":
        if not codes:
            raise ValueError("codes must be non-empty")
        if not 0.0 < max_document_frequency < 1.0:
            raise ValueError("max_document_frequency must be in (0, 1)")
        lengths = {len(code) for code in codes}
        if len(lengths) != 1:
            raise ValueError("code lengths must match")
        code_length = lengths.pop()
        frequencies = [0] * code_length
        input_ones = 0
        for code in codes:
            active = tuple(code.active_indices())
            input_ones += len(active)
            for index in active:
                frequencies[index] += 1
        maximum_count = int(math.floor(len(codes) * max_document_frequency))
        kept = frozenset(
            index
            for index, frequency in enumerate(frequencies)
            if 0 < frequency <= maximum_count
        )
        if not kept:
            raise RuntimeError("frequency pruning removed every active bit")
        pruner = cls(code_length=code_length, kept_indices=kept)
        output_ones = sum(
            1
            for code in codes
            for index in code.active_indices()
            if index in kept
        )
        used_bits = sum(frequency > 0 for frequency in frequencies)
        LOGGER.event(
            "encoding.frequency_pruner.fit",
            section=CLUSTER_PRUNING,
            data={
                "codes": len(codes),
                "code_length": code_length,
                "used_bits": used_bits,
                "kept_bits": len(kept),
                "removed_bits": used_bits - len(kept),
                "max_document_frequency": max_document_frequency,
                "input_ones_avg": input_ones / len(codes),
                "output_ones_avg": output_ones / len(codes),
            },
        )
        return pruner

    @classmethod
    def from_kept_codes(
        cls,
        codes: Sequence[BitArray],
    ) -> "SparseCodeFrequencyPruner":
        if not codes:
            raise ValueError("codes must be non-empty")
        lengths = {len(code) for code in codes}
        if len(lengths) != 1:
            raise ValueError("code lengths must match")
        code_length = lengths.pop()
        kept = frozenset(
            index
            for code in codes
            for index in code.active_indices()
        )
        pruner = cls(code_length=code_length, kept_indices=kept)
        LOGGER.event(
            "encoding.frequency_pruner.restore",
            section=CLUSTER_PRUNING,
            data={
                "codes": len(codes),
                "code_length": code_length,
                "kept_bits": len(kept),
            },
        )
        return pruner

    def transform(self, code: BitArray) -> BitArray:
        if len(code) != self.code_length:
            raise ValueError("code length must match frequency pruner")
        result = BitArray.from_indices(
            self.code_length,
            (
                index
                for index in code.active_indices()
                if index in self.kept_indices
            ),
        )
        LOGGER.event(
            "encoding.frequency_pruner.transform",
            section=CLUSTER_PRUNING,
            data={
                "input_bits": code.count(),
                "output_bits": result.count(),
                "kept_bits": len(self.kept_indices),
            },
        )
        return result

    def transform_many(
        self,
        codes: Sequence[BitArray],
    ) -> tuple[BitArray, ...]:
        return tuple(self.transform(code) for code in codes)
