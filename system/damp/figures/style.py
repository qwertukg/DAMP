"""Общий стиль рисунков проекта (вне статьи): палитра, шрифты, лист с подписями, вид цифры со штрихами.

Значения палитры и шрифтов задаёт main.py (FIGURE_*); здесь только способы рисования.
"""
from __future__ import annotations

import colorsys
import math
import os
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from damp import settings
from damp.article_refs import OUTSIDE_ARTICLE
from damp.logging import LOGGER

Colour = tuple[int, int, int]


@dataclass(frozen=True)
class FigureStyle:
    """Палитра и шрифты рисунков: фон, подписи трёх уровней яркости, ряды данных, клетка нулевого бита."""

    background: Colour
    surface: Colour
    ink: Colour
    second: Colour
    muted: Colour
    grid: Colour
    off: Colour
    blue: Colour
    orange: Colour
    aqua: Colour
    green: Colour
    fonts: Sequence[str]
    bold_fonts: Sequence[str]
    _cache: dict = field(default_factory=dict, compare=False, repr=False)

    def font(self, size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
        key = (int(size), bool(bold))
        if key not in self._cache:
            loaded = None
            for name in self.bold_fonts if bold else self.fonts:
                try:
                    loaded = ImageFont.truetype(name, int(size))
                    break
                except OSError:
                    continue
            self._cache[key] = loaded or ImageFont.load_default()
        return self._cache[key]

    @staticmethod
    def hex(colour: Colour) -> str:
        return "#" + "".join(f"{int(value):02x}" for value in colour)

    @staticmethod
    def blend(colour: Colour, base: Colour, share: float) -> Colour:
        """Цвет colour, приглушённый к base: share = 1 — сам цвет, 0 — base."""
        return tuple(int(round(b + (c - b) * float(share))) for c, b in zip(colour, base))


def figure_style() -> FigureStyle:
    """Стиль рисунков проекта из настроек FIGURE_* (вне статьи)."""
    return FigureStyle(
        background=settings.FIGURE_BACKGROUND,
        surface=settings.FIGURE_SURFACE,
        ink=settings.FIGURE_INK,
        second=settings.FIGURE_SECOND,
        muted=settings.FIGURE_MUTED,
        grid=settings.FIGURE_GRID,
        off=settings.FIGURE_OFF,
        blue=settings.FIGURE_BLUE,
        orange=settings.FIGURE_ORANGE,
        aqua=settings.FIGURE_AQUA,
        green=settings.FIGURE_GREEN,
        fonts=settings.FIGURE_FONTS,
        bold_fonts=settings.FIGURE_BOLD_FONTS,
    )


def wrap(draw: ImageDraw.ImageDraw, text: str, width: float, font: ImageFont.FreeTypeFont) -> list[str]:
    """Перенос строки по словам под ширину width пикселей."""
    lines, line = [], ""
    for word in text.split():
        probe = (line + " " + word).strip()
        if draw.textlength(probe, font=font) <= width or not line:
            line = probe
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return lines


def plural(count: int, one: str, few: str, many: str) -> str:
    """Форма слова при числе: 1 деталь, 2 детали, 5 деталей."""
    tail, teen = int(count) % 10, int(count) % 100
    if tail == 1 and teen != 11:
        return one
    if tail in (2, 3, 4) and teen not in (12, 13, 14):
        return few
    return many


def hue_colour(degrees: float, *, saturation: float = 0.9, value: float = 1.0) -> Colour:
    """Цвет направления по кругу: 0° — красный, 120° — зелёный, 240° — синий."""
    red, green, blue = colorsys.hsv_to_rgb((float(degrees) % 360.0) / 360.0, saturation, value)
    return int(red * 255), int(green * 255), int(blue * 255)


class Sheet:
    """Лист рисунка: заголовок, панели с заголовком и подписью в несколько строк."""

    def __init__(self, style: FigureStyle, width: int, head: str, *, margin: int = 20, height: int = 6000) -> None:
        self.style, self.width, self.margin = style, int(width), int(margin)
        self.image = Image.new("RGB", (self.width, int(height)), style.background)
        self.draw = ImageDraw.Draw(self.image)
        self.draw.text((margin, margin - 4), head, fill=style.ink, font=style.font(26, bold=True))
        self.top = margin + 44

    def caption(self, left: int, top: int, title: str, text: str, width: int) -> int:
        """Заголовок панели и подпись под ним; возвращает верх места под картинку панели."""
        self.draw.text((left, top), title, fill=self.style.ink, font=self.style.font(19, bold=True))
        line = top + 27
        if text:
            for row in wrap(self.draw, text, width, self.style.font(16)):
                self.draw.text((left, line), row, fill=self.style.second, font=self.style.font(16))
                line += 21
        return line + 6

    def text(self, left: int, top: int, text: str, width: int, *, size: int = 16, colour: Colour | None = None) -> int:
        for row in wrap(self.draw, text, width, self.style.font(size)):
            self.draw.text((left, top), row, fill=colour or self.style.second, font=self.style.font(size))
            top += size + 5
        return top

    def paste(self, picture: Image.Image, left: int, top: int) -> None:
        self.image.paste(picture, (int(left), int(top)))

    def grid(self, panels: Sequence[tuple[str, str, Image.Image]], *, columns: int, top: int | None = None, gap: int = 20) -> int:
        """Панели сеткой по columns в ряд: в каждом ряду подписи выровнены, картинки начинаются с одной высоты."""
        top = self.top if top is None else int(top)
        for first in range(0, len(panels), columns):
            row = panels[first : first + columns]
            lefts, left = [], self.margin
            for _, _, picture in row:
                lefts.append(left)
                left += picture.width + gap
            start = max(self.caption(x, top, title, text, picture.width) for x, (title, text, picture) in zip(lefts, row))
            for x, (_, _, picture) in zip(lefts, row):
                self.paste(picture, x, start)
            top = start + max(picture.height for _, _, picture in row) + gap + 6
        return top

    def save(self, path: str, bottom: int, *, note: str = "", section: str = OUTSIDE_ARTICLE, data: dict | None = None) -> str:
        """Лист до высоты bottom — в файл; запись в лог с сутью рисунка."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        image = self.image.crop((0, 0, self.width, int(bottom) + self.margin))
        image.save(path, optimize=True)
        LOGGER.event(
            "figure.save",
            section=section,
            data={"path": path, "width": image.width, "height": image.height, "note": note, **(data or {})},
        )
        return path


class DigitView:
    """Вид на картинку цифры в клетках сетки углопозиций: фон — сама цифра, поверх штрихи, точки и круги.

    Клетка сетки — окно Собеля; side_cells клеток на сторону картинки. Штрих рисуется стрелкой вдоль края:
    угол углопозиции — направление перепада яркости (поперёк линии), отсчитан с осью Y вверх.
    """

    def __init__(
        self,
        style: FigureStyle,
        size: int,
        image: np.ndarray,
        side_cells: float,
        scale: float,
        *,
        centre: tuple[float, float] | None = None,
        backdrop: float = 0.35,
    ) -> None:
        self.style, self.size, self.scale = style, int(size), float(scale)
        self.cy, self.cx = centre if centre is not None else ((side_cells - 1) / 2.0, (side_cells - 1) / 2.0)
        self.image = Image.new("RGB", (self.size, self.size), style.background)
        digit = Image.fromarray((np.asarray(image, dtype=np.float64) * backdrop).astype(np.uint8))
        pixels = max(1, int(round(side_cells * self.scale)))
        self.image.paste(
            digit.resize((pixels, pixels), Image.BILINEAR).convert("RGB"),
            (int(round(self.px(-0.5))), int(round(self.py(-0.5)))),
        )
        self.draw = ImageDraw.Draw(self.image)

    def px(self, x: float) -> float:
        return self.size / 2.0 + (float(x) - self.cx) * self.scale

    def py(self, y: float) -> float:
        return self.size / 2.0 + (float(y) - self.cy) * self.scale

    def stick(self, y: float, x: float, angle: float, *, dim: float = 1.0) -> None:
        StickPainter.draw(self.draw, self.px(x), self.py(y), self.scale, angle, dim=dim)

    def circle(self, y: float, x: float, radius: float, colour: Colour, width: int = 2) -> None:
        r = float(radius) * self.scale
        self.draw.ellipse((self.px(x) - r, self.py(y) - r, self.px(x) + r, self.py(y) + r), outline=colour, width=width)

    def dot(self, y: float, x: float, pixels: float, fill: Colour | None, outline: Colour | None = None) -> None:
        box = (self.px(x) - pixels, self.py(y) - pixels, self.px(x) + pixels, self.py(y) + pixels)
        self.draw.ellipse(box, fill=fill, outline=outline, width=2 if outline else 1)


def axial_sticks() -> bool:
    """Штрихи по середине линии — без стрелки (MNIST_STROKE_DETECTORS и MNIST_STROKE_MIDLINE): угол 0…180°,
    0° и 180° — одна линия. Цвет такого штриха идёт по кругу за 180° (как на дашборде), наконечника нет."""
    return bool(settings.MNIST_STROKE_DETECTORS) and bool(settings.MNIST_STROKE_MIDLINE)


def stick_hue(angle: float) -> float:
    """Тон цвета штриха по его углу: у штриха без стрелки круг цветов проходится за 180°."""
    return 2.0 * float(angle) if axial_sticks() else float(angle)


class StickPainter:
    """Штрих отрезком вдоль линии длиной почти в клетку, цвет — направление. У штриха края — стрелка (направление
    перепада яркости), у штриха середины линии — отрезок без наконечника (axial_sticks)."""

    @staticmethod
    def draw(draw: ImageDraw.ImageDraw, px: float, py: float, unit: float, angle: float, *, dim: float = 1.0) -> None:
        along = math.radians(float(angle) + 90.0)
        half = unit * 0.45
        ux, uy = math.cos(along) * half, -math.sin(along) * half
        colour = tuple(int(channel * dim) for channel in hue_colour(stick_hue(angle)))
        width = 2 if unit >= 5 else 1
        draw.line((px - ux, py - uy, px + ux, py + uy), fill=colour, width=width)
        if half >= 4 and not axial_sticks():
            for turn in (150.0, -150.0):
                side = along + math.radians(turn)
                draw.line(
                    (px + ux + math.cos(side) * half * 0.7, py + uy - math.sin(side) * half * 0.7, px + ux, py + uy),
                    fill=colour,
                    width=width,
                )


def bits_raster(bits: np.ndarray, columns: int, cell: int, colours: Colour | Sequence[Colour], off: Colour, *, outline: np.ndarray | None = None, ring: Colour = (255, 255, 255)) -> Image.Image:
    """Биты клетками по columns в строке: единица — свой цвет (или общий), ноль — тёмная клетка; outline — обвести."""
    bits = np.asarray(bits).astype(bool)
    rows = int(math.ceil(bits.shape[0] / columns))
    picture = Image.new("RGB", (columns * cell, rows * cell), (0, 0, 0))
    draw = ImageDraw.Draw(picture)
    single = isinstance(colours, tuple) and len(colours) == 3 and isinstance(colours[0], (int, np.integer))
    for index, bit in enumerate(bits):
        left, top = (index % columns) * cell, (index // columns) * cell
        colour = (colours if single else colours[index]) if bit else off
        draw.rectangle((left, top, left + cell - 2, top + cell - 2), fill=colour)
        if outline is not None and outline[index]:
            draw.rectangle((left - 1, top - 1, left + cell - 1, top + cell - 1), outline=ring, width=1)
    return picture


def crop_cells(image: Image.Image, width: int, height: int, *, tile: int, top_skip: int = 0, centre: tuple[int, int] | None = None) -> Image.Image:
    """Вырезка из карты с рисунками в ячейках, по границам ячеек; по умолчанию — из середины."""
    if centre is None:
        cx = (image.width // 2) // tile * tile
        cy = ((image.height - top_skip) // 2) // tile * tile + top_skip
    else:
        cx, cy = centre
    left, top = cx - (width // 2) // tile * tile, cy - (height // 2) // tile * tile
    return image.crop((left, top, left + width, top + height))
