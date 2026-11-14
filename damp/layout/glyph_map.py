"""Карта раскладки с содержимым точек: в ячейке каждой точки нарисован её стимул (вне статьи, наглядность).

§5.14 красит точку раскладки псевдоцветом её кода. Для кодов наборов (кусков, картинок) этот цвет почти
одинаков у всех точек, и карта ничего не показывает. Здесь в ячейку точки рисуется то, что она кодирует:
по карте видно, что соседние точки — действительно похожие стимулы (§5.13.2).
"""
from __future__ import annotations

import os
from typing import Callable

import numpy as np

from damp.article_refs import OUTSIDE_ARTICLE
from damp.layout.damp_layout import Layout
from damp.logging import LOGGER

# Рисует стимул строки row в квадрат (left, top, size) на картинке canvas (PIL.Image) пером draw (ImageDraw).
TilePainter = Callable[[object, object, int, int, int, int], None]


class LayoutGlyphMap:
    """Раскладка, в ячейках которой нарисованы стимулы её точек.

    packed — коды стимулов (строки × ⌈L/8⌉) в том порядке, в котором их знает вызывающий; точке раскладки
    сопоставляется строка с тем же кодом (одинаковые коды раздаются точкам по порядку строк).
    """

    HEADER = 60
    MAX_SIDE = 9000
    FONTS = ("arial.ttf", "DejaVuSans.ttf")

    def __init__(self, layout: Layout, packed: "np.ndarray") -> None:
        packed = np.ascontiguousarray(packed, dtype=np.uint8)
        table = np.ascontiguousarray(layout.code_table.packed)
        if packed.ndim != 2 or packed.shape[1] != table.shape[1]:
            raise ValueError("codes must have the same packed width as the layout codes")
        waiting: dict[bytes, list[int]] = {}
        for row in range(packed.shape[0] - 1, -1, -1):
            waiting.setdefault(packed[row].tobytes(), []).append(row)
        rows = np.full(table.shape[0], -1, dtype=np.int64)
        for point in range(table.shape[0]):
            candidates = waiting.get(table[point].tobytes())
            if candidates:
                rows[point] = candidates.pop() if len(candidates) > 1 else candidates[0]
        self.rows = rows
        self._cells = layout.cells_array()
        self._height = int(layout.height)
        self._width = int(layout.width)

    def tile_size(self, wanted: int) -> int:
        """Размер ячейки в пикселях: не больше wanted и так, чтобы картинка не вышла за MAX_SIDE."""
        return max(8, min(int(wanted), self.MAX_SIDE // max(self._height, self._width)))

    def save(self, path: str, painter: TilePainter, *, tile: int, title: str = "") -> None:
        """Картинка карты: painter рисует стимул каждой точки в её ячейке; title — подпись сверху."""
        from PIL import Image, ImageDraw, ImageFont

        canvas = Image.new("RGB", (self._width * tile, self._height * tile + self.HEADER), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        if title:
            for name in self.FONTS:
                try:
                    draw.text((10, 12), title, font=ImageFont.truetype(name, 30), fill=(255, 255, 255))
                    break
                except OSError:
                    continue
        painted = 0
        for point, row in enumerate(self.rows):
            if row < 0:
                continue
            left = int(self._cells[point, 1]) * tile
            top = int(self._cells[point, 0]) * tile + self.HEADER
            painter(canvas, draw, int(row), left, top, tile)
            painted += 1
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        canvas.save(path)
        LOGGER.event(
            "layout.glyph_map.save",
            section=OUTSIDE_ARTICLE,
            data={
                "image": path,
                "points": int(self.rows.shape[0]),
                "painted": painted,
                "tile": tile,
                "width": canvas.width,
                "height": canvas.height,
                "note": "в ячейке точки нарисован её стимул: видно, что рядом лежат похожие (§5.13.2)",
            },
        )
