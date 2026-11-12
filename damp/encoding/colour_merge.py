"""Colour merge (§3.3): слияние кодов без пересыщения."""
from __future__ import annotations

import numpy as np

from damp.article_refs import COLOR_MERGE
from damp.logging import LOGGER


class ColourMerge:
    """Отбор битов по «цвету» при слиянии многих кодов (§3.3).

    Цвет бита — его номер в коде: младшие биты красные (широкие слои детекторов, дальний порядок), старшие
    фиолетовые (узкие слои, ближний порядок) (§3.2, §3.3.1, §4.4.2, §5.14). Если единиц в OR кодов не больше
    бюджета насыщенности t, результат — сам OR (§3.3); иначе биты отбираются с одного конца «спектра»
    (§3.3.2): keep="short" сохраняет коротковолновые (старшие) биты — смысл остаётся различимым,
    keep="long" — длинноволновые (младшие) — описание осмысленно как целое.
    """

    KEEP_MODES = ("short", "long")

    def __init__(self, saturation_limit: int, *, keep: str = "short") -> None:
        if saturation_limit <= 0:
            raise ValueError("saturation_limit must be positive")
        if keep not in self.KEEP_MODES:
            raise ValueError(f"keep must be one of {self.KEEP_MODES}")
        self._limit = int(saturation_limit)
        self._keep = keep
        LOGGER.event(
            "colour_merge.init",
            section=COLOR_MERGE,
            data={
                "saturation_limit": self._limit,
                "keep": self._keep,
                "note": "§3.3: |a ∨ b| ≤ t — обычный OR; иначе биты отбираются с одного конца спектра (§3.3.2)",
            },
        )

    @property
    def saturation_limit(self) -> int:
        return self._limit

    @property
    def keep(self) -> str:
        return self._keep

    def merge(self, bits: "np.ndarray") -> tuple["np.ndarray", int]:
        """Код после слияния и число отброшенных битов; вход — OR сливаемых кодов (0/1, длина кода)."""
        bits = np.asarray(bits)
        if bits.ndim != 1:
            raise ValueError("bits must be a one-dimensional 0/1 vector")
        active = np.flatnonzero(bits)
        dropped = max(0, int(active.size) - self._limit)
        if dropped == 0:
            merged = (bits != 0).astype(np.uint8)
        else:
            kept = active[-self._limit :] if self._keep == "short" else active[: self._limit]
            merged = np.zeros(bits.shape[0], dtype=np.uint8)
            merged[kept] = 1
        LOGGER.event(
            "colour_merge.merge",
            section=COLOR_MERGE,
            data={
                "bits_in": int(active.size),
                "bits_out": int(active.size) - dropped,
                "dropped": dropped,
                "saturation_limit": self._limit,
                "keep": self._keep,
                "filtered": dropped > 0,
            },
        )
        return merged, dropped
