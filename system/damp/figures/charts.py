"""График узнавания по уровням цикла (мерка «собрались ли классы», вне статьи)."""
from __future__ import annotations

import math
import os
from typing import Sequence

import numpy as np

from damp.article_refs import OUTSIDE_ARTICLE
from damp.figures.style import FigureStyle
from damp.logging import LOGGER


class AccuracyChart:
    """Столбики узнавания по уровням: отклик уровня, отклик по сильнейшему, склейка уровней; пунктир — голосование."""

    def __init__(self, style: FigureStyle) -> None:
        self.style = style

    @staticmethod
    def _units_per_pixel(axis) -> tuple[float, float]:  # noqa: ANN001
        origin = axis.transData.transform((0.0, 0.0))
        one = axis.transData.transform((1.0, 1.0))
        return 1.0 / abs(one[0] - origin[0]), 1.0 / abs(one[1] - origin[1])

    @staticmethod
    def _arc(cx: float, cy: float, rx: float, ry: float, first: float, last: float) -> list[tuple[float, float]]:
        return [(cx + rx * math.cos(math.radians(a)), cy + ry * math.sin(math.radians(a))) for a in np.linspace(first, last, 8)]

    def _bar(self, axis, x: float, height: float, width: float, colour: str, radius: float = 4.0) -> None:  # noqa: ANN001
        """Столбик от нуля со скруглённым верхом (радиус в пикселях)."""
        from matplotlib.patches import Polygon

        if height <= 0:
            return
        per_x, per_y = self._units_per_pixel(axis)
        rx, ry = min(radius * per_x, width / 2.0), min(radius * per_y, height)
        points = [(x, 0.0), (x, height - ry)] + self._arc(x + rx, height - ry, rx, ry, 180, 90)
        points += self._arc(x + width - rx, height - ry, rx, ry, 90, 0) + [(x + width, 0.0)]
        axis.add_patch(Polygon(points, closed=True, color=colour, linewidth=0))

    def save(self, levels: Sequence[dict], vote: float, path: str, *, title: str) -> str:
        """levels — по уровню: number, seen, alone, peak, glued (доли верных ответов); vote — голосование."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        style = self.style
        hex_ = style.hex
        series = [
            ("Отклик уровня", hex_(style.blue), [100 * level["alone"] for level in levels]),
            ("Отклик по сильнейшему", hex_(style.orange), [100 * level["peak"] for level in levels]),
            ("Склейка уровней 1…k", hex_(style.aqua), [100 * level["glued"] for level in levels]),
        ]
        surface, ink, second, muted, grid = (hex_(c) for c in (style.surface, style.ink, style.second, style.muted, style.grid))
        figure, axis = plt.subplots(figsize=(9.0, 4.6), dpi=110, facecolor=surface)
        axis.set_facecolor(surface)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(grid)
        axis.tick_params(colors=muted, length=0)
        figure.subplots_adjust(left=0.07, right=0.89, top=0.80, bottom=0.14)
        axis.set_xlim(-0.5, len(levels) - 0.5)
        # Ось до 100 %: на 100 проверочных цифрах уровни доходят до 85–90 %, при прежней оси до 80 % столбцы обрезались.
        axis.set_ylim(0, 100)
        width, gap = 0.22, 0.03
        for number, (name, colour, values) in enumerate(series):
            for index, value in enumerate(values):
                left = index + (number - 1) * (width + gap) - width / 2
                self._bar(axis, left, value, width, colour)
                axis.text(left + width / 2, value - 2.0, f"{value:.1f}", ha="center", va="top", color="#ffffff", fontsize=9)
            axis.add_patch(Rectangle((0, 0), 0, 0, color=colour, label=name))
        axis.axhline(100 * vote, color=second, linewidth=1, linestyle=(0, (4, 3)))
        axis.text(-0.47, 100 * vote + 1.0, f"голосование верхних деталей — {100 * vote:.1f} %", color=second, fontsize=10, va="bottom")
        axis.axhline(10, color=muted, linewidth=1, linestyle=(0, (1, 3)))
        axis.text(len(levels) - 0.47, 10, "наугад\n10 %", color=muted, fontsize=10, va="center", clip_on=False)
        axis.set_xticks(range(len(levels)))
        axis.set_xticklabels([f"уровень {level['number']}\nокно видит {100 * level['seen']:.0f} % цифры" for level in levels], color=second)
        axis.set_yticks([0, 20, 40, 60, 80, 100])
        axis.set_yticklabels(["0", "20", "40", "60", "80", "100 %"])
        axis.grid(axis="y", color=grid, linewidth=1)
        axis.set_axisbelow(True)
        axis.legend(loc="lower left", frameon=False, ncols=3, labelcolor=second, handlelength=1.0, handleheight=1.0, bbox_to_anchor=(-0.01, 1.0))
        axis.set_title(title, color=ink, fontsize=13, loc="left", pad=34)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        figure.savefig(path, facecolor=surface)
        plt.close(figure)
        LOGGER.event(
            "figure.save",
            section=OUTSIDE_ARTICLE,
            data={"path": path, "levels": [dict(level) for level in levels], "vote": vote, "note": "узнавание по уровням цикла"},
        )
        return path
