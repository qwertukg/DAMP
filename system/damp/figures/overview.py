"""Обзорная полоса пайплайна: по миниатюре на этап и цикл стрелкой возврата с уровнями внутри (вне статьи)."""
from __future__ import annotations

import os
from typing import Sequence

from PIL import Image, ImageDraw

from damp.article_refs import NEURAL_ANALOGY
from damp.figures.style import Colour, FigureStyle
from damp.logging import LOGGER


class PipelineOverview:
    """Этапы 1–5 и 7 в ряд; этап 6 — петля от эмбеддинга обратно к коду, внутри — карты уровней."""

    def __init__(self, style: FigureStyle, *, side: int = 180, gap: int = 30, margin: int = 20, thumb: int = 150) -> None:
        self.style, self.side, self.gap, self.margin, self.thumb = style, int(side), int(gap), int(margin), int(thumb)

    def _arrow(self, draw: ImageDraw.ImageDraw, x: float, y: float, colour: Colour) -> None:
        draw.line((x - 8, y, x + 7, y), fill=colour, width=2)
        draw.line((x + 2, y - 5, x + 7, y), fill=colour, width=2)
        draw.line((x + 2, y + 5, x + 7, y), fill=colour, width=2)

    def save(
        self,
        tiles: Sequence[tuple[str, str, Image.Image]],
        levels: Sequence[tuple[str, str, Image.Image]],
        path: str,
        *,
        loop_from: int,
        loop_to: int,
        loop_title: str,
        loop_notes: tuple[Sequence[str], Sequence[str]],
    ) -> str:
        """tiles — этапы в ряд (заголовок, подпись, картинка); levels — миниатюры уровней цикла.

        Петля идёт от плитки loop_from вниз и обратно вверх к плитке loop_to.
        """
        style, side, gap, margin, thumb = self.style, self.side, self.gap, self.margin, self.thumb
        width = len(tiles) * side + (len(tiles) - 1) * gap + 2 * margin
        row_top = margin + 48
        loop_top = row_top + side
        thumbs_top = loop_top + 52
        loop_bottom = thumbs_top + thumb + 54
        strip = Image.new("RGB", (width, loop_bottom + margin + 4), style.background)
        draw = ImageDraw.Draw(strip)
        lefts = [margin + number * (side + gap) for number in range(len(tiles))]
        for number, (title, note, picture) in enumerate(tiles):
            left = lefts[number]
            draw.text((left, margin - 4), title, fill=style.ink, font=style.font(18, bold=True))
            draw.text((left, margin + 20), note, fill=style.second, font=style.font(14))
            resample = Image.NEAREST if picture.width < side else Image.LANCZOS
            strip.paste(picture.resize((side, side), resample), (left, row_top))
            draw.rectangle((left - 1, row_top - 1, left + side, row_top + side), outline=style.grid)
            if number:
                self._arrow(draw, left - gap // 2, row_top + side // 2, style.muted)
        start, finish = lefts[loop_from] + side // 2, lefts[loop_to] + side // 2
        draw.line((start, loop_top + 2, start, loop_bottom, finish, loop_bottom, finish, loop_top + 6), fill=style.green, width=3, joint="curve")
        draw.polygon((finish, loop_top + 2, finish - 8, loop_top + 16, finish + 8, loop_top + 16), fill=style.green)
        inner_left, inner_right = finish + 18, start - 18
        draw.text((inner_left, loop_top + 12), loop_title, fill=style.ink, font=style.font(18, bold=True))
        step = (inner_right - inner_left - len(levels) * thumb) // max(1, len(levels) - 1)
        for number, (title, note, picture) in enumerate(levels):
            left = inner_left + number * (thumb + step)
            strip.paste(picture.resize((thumb, thumb), Image.LANCZOS), (left, thumbs_top))
            draw.rectangle((left - 1, thumbs_top - 1, left + thumb, thumbs_top + thumb), outline=style.grid)
            draw.text((left, thumbs_top + thumb + 6), title, fill=style.ink, font=style.font(13, bold=True))
            draw.text((left, thumbs_top + thumb + 24), note, fill=style.second, font=style.font(13))
            if number:
                self._arrow(draw, left - step // 2, thumbs_top + thumb // 2, style.green)
        before, after = loop_notes
        for line, text in enumerate(before):
            draw.text((finish - 14 - draw.textlength(text, font=style.font(14)), loop_top + 62 + line * 19), text, fill=style.second, font=style.font(14))
        for line, text in enumerate(after):
            draw.text((start + 16, loop_top + 62 + line * 19), text, fill=style.second, font=style.font(14))
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        strip.save(path, optimize=True)
        LOGGER.event(
            "figure.save",
            section=NEURAL_ANALOGY,
            data={"path": path, "stages": len(tiles), "levels": len(levels), "note": "пайплайн с циклом: проход повторяется уровень за уровнем (§6.6)"},
        )
        return path
