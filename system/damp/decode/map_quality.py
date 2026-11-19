"""Мера «собрались ли классы» на карте раскладки: доля соседних точек с одной подписью (вне статьи; ср. рис. 16,
§5.13.2)."""

from __future__ import annotations

import numpy as np

from damp.article_refs import MAPPING_CONTINUITY
from damp.layout.damp_layout import Layout
from damp.logging import LOGGER


class LabelMapQuality:
    """Согласие подписей соседних точек на карте раскладки (диагностика; ср. рис. 16, §5.13.2).

    Подпись в раскладке не участвует: это известное свойство стимула (класс картинки), по которому видно,
    собрала ли раскладка похожие коды рядом. Мера — доля пар соседних занятых ячеек (по горизонтали и
    вертикали) с одинаковой подписью; для сравнения даётся её ожидание при случайной расстановке, Σ p_c².
    """

    def measure(self, layout: Layout) -> dict[str, float]:
        cells = layout.cells_array()
        labels = np.asarray(layout.code_table.label_index, dtype=np.int64)
        grid = np.full((int(layout.height), int(layout.width)), -1, dtype=np.int64)
        grid[cells[:, 0], cells[:, 1]] = labels
        pairs = same = 0
        for first, second in ((grid[:, :-1], grid[:, 1:]), (grid[:-1, :], grid[1:, :])):
            occupied = (first >= 0) & (second >= 0)
            pairs += int(np.count_nonzero(occupied))
            same += int(np.count_nonzero(occupied & (first == second)))
        shares = np.bincount(labels).astype(np.float64) / max(1, labels.shape[0])
        quality = {
            "neighbour_pairs": float(pairs),
            "same_label": same / pairs if pairs else 0.0,
            "same_label_random": float(np.sum(shares * shares)),
        }
        LOGGER.event(
            "decode.map_quality",
            section=MAPPING_CONTINUITY,
            data={
                "points": int(labels.shape[0]),
                "labels": int(np.count_nonzero(shares)),
                "neighbour_pairs": pairs,
                "same_label": quality["same_label"],
                "same_label_random": quality["same_label_random"],
                "note": "доля пар соседних занятых ячеек с одинаковой подписью; same_label_random — её ожидание "
                "при случайной расстановке (подпись в раскладке не участвует)",
            },
        )
        return quality
