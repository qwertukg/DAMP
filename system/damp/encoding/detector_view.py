"""Первичные детекторы кодера картинкой в общем стиле рисунков проекта.

Каждая координата рисуется так, как устроены её детекторы (§4.1–§4.4):

- замкнутая координата (угол, §4.4) — кольца: слой — кольцо, самый широкий слой внутри, самый узкий снаружи;
  детектор — дуга своего рецептивного поля; соседние детекторы слоя, поля которых перекрываются, разнесены
  по дорожкам кольца, поэтому перекрытие видно как наложение дуг (§4.2.1);
- отрезок (§4.2) — полосы по слоям сверху вниз, тоже с дорожками перекрытия;
- двумерная позиция (§4.3) — круги на плоскости, по слою на панель.

Цвет детектора — псевдоцвет его бита: младший бит красный, старший фиолетовый (§3.2, §4.4.2, §5.14), поэтому
широкие слои (младшие биты) — красная часть спектра, узкие — фиолетовая. Сработавшие для стимула детекторы
яркие, остальные приглушены. Код стимула — клетки бит: единица в цвете своего бита, бит, который поставили
детекторы разных координат, обведён (паразитный, §4.2.1).
"""
from __future__ import annotations

import colorsys
import math
import os
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from damp.article_refs import CONJUNCTION, INTEGER_ENCODING, OUTSIDE_ARTICLE, TWO_DIM_CODES
from damp.encoding.damp_encoder import DetectorCircle, DetectorWindow, Dimension, Encoder, PlaneDimension
from damp.figures.style import Colour, FigureStyle, Sheet, StickPainter
from damp.logging import LOGGER


class DetectorLayerView:
    """Картинка детекторов кодера по слоям с перекрытием и цветом бита (§4.1, §4.2.1, §4.3, §3.2)."""

    def __init__(
        self,
        encoder: Encoder,
        *,
        style: FigureStyle,
        min_hue: float,
        max_hue: float,
        inactive_alpha: float,
        panel: int,
        code_columns: int,
        code_cell: int,
    ) -> None:
        if max_hue < min_hue:
            raise ValueError("max_hue must be >= min_hue")
        if not 0.0 <= inactive_alpha <= 1.0:
            raise ValueError("inactive_alpha must be in [0, 1]")
        self._encoder = encoder
        self._style = style
        self._min_hue = float(min_hue)
        self._max_hue = float(max_hue)
        self._inactive_alpha = float(inactive_alpha)
        self._panel = int(panel)
        self._code_columns = int(code_columns)
        self._code_cell = int(code_cell)

    # ------------------------------------------------------------------ панели координат
    def ring(self, dimension: int, value: float | None) -> Image.Image:
        """Замкнутая координата кольцами: слой — кольцо, детектор — дуга поля, луч — значение стимула."""
        dim = self._encoder.dimensions[dimension]
        if not dim.closed:
            raise ValueError("ring draws a closed dimension")
        size = self._panel
        style = self._style
        picture = Image.new("RGB", (size, size), style.background)
        draw = ImageDraw.Draw(picture)
        centre = size / 2.0
        inner, outer, gap = 0.17 * size, 0.47 * size, 2.0
        lanes = [self.layer_lanes(dimension, index) for index in range(len(dim.detector_layers))]
        thick = (outer - inner - gap * (len(lanes) - 1)) / float(sum(lanes))
        windows = self._windows(dimension)
        radius = inner
        active_total = 0
        for layer_index, layer_lanes in enumerate(lanes):
            layer_windows = sorted((w for w in windows if w.layer_index == layer_index), key=lambda w: w.center)
            drawn: list[tuple[bool, float, float, float, Colour]] = []
            for position, window in enumerate(layer_windows):
                active = value is not None and window.is_active(float(value))
                active_total += int(active)
                lane_radius = radius + (position % layer_lanes + 0.5) * thick
                drawn.append((active, lane_radius, window.center, window.half_width, self.bit_colour(window.bit)))
            # Сработавшие поверх приглушённых.
            for active, lane_radius, centre_value, half_width, colour in sorted(drawn, key=lambda item: item[0]):
                fill = colour if active or value is None else style.blend(colour, style.background, self._inactive_alpha)
                box = (centre - lane_radius, centre - lane_radius, centre + lane_radius, centre + lane_radius)
                width = max(1, int(round(thick)) - 1)
                if 2.0 * half_width >= dim.period:
                    draw.ellipse(box, outline=fill, width=width)
                    continue
                middle = self._degrees(dim, centre_value)
                span = 360.0 * half_width / dim.period
                # Угол отсчитан против часовой стрелки от оси X; в картинке ось Y вниз, отсюда знак.
                draw.arc(box, -(middle + span), -(middle - span), fill=fill, width=width)
            radius += layer_lanes * thick + gap
        if value is not None:
            angle = math.radians(self._degrees(dim, float(value)))
            draw.line(
                (centre, centre, centre + outer * math.cos(angle), centre - outer * math.sin(angle)),
                fill=style.ink,
                width=2,
            )
            label = f"{float(value):.0f}°" if dim.period == 360.0 else f"{float(value):.3g}"
            font = style.font(22, bold=True)
            draw.text((centre - draw.textlength(label, font=font) / 2, centre - 13), label, fill=style.ink, font=font)
        LOGGER.event(
            "encoder.detectors.view.ring",
            section=INTEGER_ENCODING,
            data={
                "dimension": dim.title,
                "layers": len(lanes),
                "lanes": lanes,
                "detectors": len(windows),
                "active": active_total if value is not None else None,
                "note": "замкнутая координата (§4.4): слой — кольцо, детектор — дуга поля, перекрытие — дорожки",
            },
        )
        return picture

    def segments(self, dimension: int, value: float | None, width: int) -> Image.Image:
        """Отрезок полосами по слоям: детектор — полоса поля, пунктир — значение стимула (§4.2)."""
        dim = self._encoder.dimensions[dimension]
        style = self._style
        lane_height, layer_gap, axis = 4, 5, 24
        lanes = [self.layer_lanes(dimension, index) for index in range(len(dim.detector_layers))]
        height = sum(lanes) * lane_height + layer_gap * len(lanes) + axis
        picture = Image.new("RGB", (int(width), int(height)), style.background)
        draw = ImageDraw.Draw(picture)
        span = float(dim.max_value - dim.min_value)

        def x_of(v: float) -> float:
            return (float(v) - dim.min_value) / span * (width - 1)

        windows = self._windows(dimension)
        top = 0
        for layer_index, layer_lanes in enumerate(lanes):
            layer_windows = sorted((w for w in windows if w.layer_index == layer_index), key=lambda w: w.center)
            drawn = []
            for position, window in enumerate(layer_windows):
                active = value is not None and window.is_active(float(value))
                lane_top = top + (position % layer_lanes) * lane_height
                drawn.append((active, lane_top, window, self.bit_colour(window.bit)))
            for active, lane_top, window, colour in sorted(drawn, key=lambda item: item[0]):
                fill = colour if active or value is None else style.blend(colour, style.background, self._inactive_alpha)
                for start, end in self.field_segments(window, dim):
                    draw.rectangle((x_of(start), lane_top, max(x_of(start) + 1, x_of(end)), lane_top + lane_height - 2), fill=fill)
            top += layer_lanes * lane_height + layer_gap
        if value is not None:
            x = x_of(float(value))
            for y in range(0, top, 6):
                draw.line((x, y, x, min(top, y + 3)), fill=style.ink, width=1)
        draw.line((0, top + 1, width - 1, top + 1), fill=style.grid, width=1)
        tick = 5.0 if span > 10 else 1.0
        first = math.ceil(dim.min_value / tick) * tick
        for mark in np.arange(first, dim.max_value + 1e-9, tick):
            x = x_of(mark)
            draw.line((x, top + 1, x, top + 5), fill=style.muted, width=1)
            draw.text((x - 6, top + 7), f"{mark:g}", fill=style.muted, font=style.font(13))
        return picture

    def plane(self, dimension: int, layer: int, points: np.ndarray, angles: Sequence[float] | None = None) -> Image.Image:
        """Двумерная позиция кругами одного слоя (§4.3); точки — позиции стимулов, сработавшие круги залиты."""
        dim = self._encoder.dimensions[dimension]
        if not isinstance(dim, PlaneDimension):
            raise ValueError("plane draws a two-dimensional position")
        size = self._panel
        style = self._style
        picture = Image.new("RGB", (size, size), style.background)
        draw = ImageDraw.Draw(picture, "RGBA")
        # Вид вмещает плоскость и круги всех слоёв целиком: круги у края выходят за плоскость.
        reach = max((w.radius for w in self._windows(dimension)), default=0.0)
        margin = 6.0
        scale = (size - 2 * margin) / (dim.side + 2.0 * reach)

        def px(x: float) -> float:
            return margin + (float(x) - dim.x_min + reach) * scale

        def py(y: float) -> float:
            return margin + (float(y) - dim.y_min + reach) * scale

        draw.rectangle((px(dim.x_min), py(dim.y_min), px(dim.x_max), py(dim.y_max)), outline=style.grid, width=1)
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        circles = [w for w in self._windows(dimension) if w.layer_index == int(layer)]
        active = [any(circle.is_active(point) for point in points) for circle in circles]
        for circle, on in sorted(zip(circles, active), key=lambda item: item[1]):
            colour = self.bit_colour(circle.bit)
            r = circle.radius * scale
            box = (px(circle.center_x) - r, py(circle.center_y) - r, px(circle.center_x) + r, py(circle.center_y) + r)
            if on:
                draw.ellipse(box, fill=colour + (70,), outline=colour + (255,), width=2)
            else:
                draw.ellipse(box, outline=style.blend(colour, style.background, self._inactive_alpha) + (255,), width=1)
        unit = max(6.0, scale * 0.9)
        for index, (y, x) in enumerate(points):
            if angles is not None:
                StickPainter.draw(draw, px(x), py(y), unit, float(angles[index]))
            else:
                draw.ellipse((px(x) - 3, py(y) - 3, px(x) + 3, py(y) + 3), fill=style.ink)
        LOGGER.event(
            "encoder.detectors.view.plane",
            section=TWO_DIM_CODES,
            data={
                "dimension": dim.title,
                "layer": int(layer),
                "circles": len(circles),
                "radius": circles[0].radius if circles else 0.0,
                "points": int(points.shape[0]),
                "active": int(sum(active)),
                "note": "двумерная позиция (§4.3): круги слоя; сработал круг, в который попала хоть одна точка",
            },
        )
        return picture

    def code(self, values: Sequence[object] | None, *, bits: np.ndarray | None = None) -> tuple[Image.Image, dict]:
        """Код стимула клетками: единица — цвет своего бита, общий бит разных координат обведён (§2.2.1, §4.2.1).

        values — значение каждой координаты (у двумерной — пара либо набор пар), bits — готовый набор единиц.
        """
        encoder = self._encoder
        counts = np.zeros(encoder.code_length, dtype=np.int64)
        per_dimension: dict[str, int] = {}
        if bits is None:
            for index, value in enumerate(values or ()):
                dim = encoder.dimensions[index]
                points = np.asarray(value, dtype=np.float64).reshape(-1, 2) if isinstance(dim, PlaneDimension) else [value]
                own: set[int] = set()
                for point in points:
                    own.update(encoder.dimension_bits(index, point if isinstance(dim, PlaneDimension) else float(point)))
                per_dimension[dim.title] = len(own)
                counts[list(own)] += 1
        else:
            counts[np.asarray(bits, dtype=np.int64)] = 1
        on = counts > 0
        colours = [self.bit_colour(bit) for bit in range(encoder.code_length)]
        from damp.figures.style import bits_raster

        picture = bits_raster(on, self._code_columns, self._code_cell, colours, self._style.off, outline=counts > 1, ring=self._style.ink)
        stats = {
            "length": int(encoder.code_length),
            "ones": int(on.sum()),
            "density": float(on.mean()),
            "parasitic": int((counts > 1).sum()),
            "by_dimension": per_dimension,
        }
        LOGGER.event(
            "encoder.detectors.view.code",
            section=CONJUNCTION,
            data={**stats, "note": "код стимула — OR битов сработавших детекторов всех координат (§2.2.1)"},
        )
        return picture, stats

    # ------------------------------------------------------------------ лист целиком
    def render(self, values: Sequence[float] | None = None) -> np.ndarray:
        """Все координаты кодера и код стимула одним листом; values — по значению на координату или None."""
        encoder = self._encoder
        if values is not None and len(values) != len(encoder.dimensions):
            raise ValueError("values must give one value per encoder dimension")
        style = self._style
        rings, bars, planes = [], [], []
        for index, dim in enumerate(encoder.dimensions):
            value = None if values is None else values[index]
            if isinstance(dim, PlaneDimension):
                for layer in range(len(dim.detector_layers)):
                    points = np.zeros((0, 2)) if value is None else np.asarray(value, dtype=np.float64).reshape(-1, 2)
                    planes.append((f"{dim.title}: слой {layer + 1}", self._layer_text(index, layer), self.plane(index, layer, points)))
            elif dim.closed:
                rings.append((f"{dim.title}: замкнутая координата", self._dimension_text(index, value), self.ring(index, value)))
            else:
                bars.append((index, dim, value))
        code_picture, stats = self.code(values) if values is not None else self.code(None, bits=np.zeros(0, dtype=np.int64))
        width = max(2 * self._panel + 20, self._panel + 20 + code_picture.width) + 40
        mode = "общий код" if encoder.shared_code else "склейка сегментов"
        head = f"Первичные детекторы: {mode}, {encoder.code_length} бит, ожидаемая плотность {encoder.expected_density():.0%}"
        sheet = Sheet(style, width, head)
        first_row = rings + planes
        code_text = (
            f"{stats['ones']} единиц из {stats['length']} ({stats['density']:.0%}); паразитных бит {stats['parasitic']} — обведены"
            if values is not None
            else f"{stats['length']} бит"
        )
        panels = first_row[:1] + [("Код стимула", code_text, code_picture)] + first_row[1:]
        top = sheet.grid(panels, columns=2)
        for index, dim, value in bars:
            picture = self.segments(index, value, width - 40)
            top = sheet.grid([(f"{dim.title}: отрезок", self._dimension_text(index, value), picture)], columns=1, top=top)
        bottom = top
        image = np.asarray(sheet.image.crop((0, 0, width, bottom + 10)))[:, :, :3].copy()
        self._log_view(values, image)
        return image

    def save(self, path: str, values: Sequence[float] | None = None) -> np.ndarray:
        image = self.render(values)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        Image.fromarray(image).save(path, optimize=True)
        LOGGER.event(
            "encoder.detectors.view.save",
            section=OUTSIDE_ARTICLE,
            data={"path": path, "width": int(image.shape[1]), "height": int(image.shape[0])},
        )
        return image

    # ------------------------------------------------------------------ helpers
    def layer_lanes(self, dimension: int, layer_index: int) -> int:
        """Сколько соседних детекторов слоя накрывают одну точку: ширина поля / шаг центров (перекрытие, §4.2.1)."""
        dim = self._encoder.dimensions[dimension]
        layer = dim.detector_layers[layer_index]
        window = next(w for w in self._windows(dimension) if w.layer_index == layer_index)
        width = 2.0 * (window.radius if isinstance(window, DetectorCircle) else window.half_width)
        return max(1, min(int(layer.count), math.ceil(width / (dim.period / layer.count) - 1e-9)))

    @staticmethod
    def field_segments(window: DetectorWindow, dim: Dimension) -> list[tuple[float, float]]:
        """Рецептивное поле на оси координаты; у замкнутой координаты поле переходит через границу периода."""
        start = window.center - window.half_width
        end = window.center + window.half_width
        if not dim.closed:
            return [(max(start, dim.min_value), min(end, dim.max_value))] if end > dim.min_value and start < dim.max_value else []
        if 2.0 * window.half_width >= dim.period:
            return [(dim.min_value, dim.max_value)]
        offset = dim.min_value
        start = (start - offset) % dim.period + offset
        end = start + 2.0 * window.half_width
        if end <= dim.max_value:
            return [(start, end)]
        return [(start, dim.max_value), (dim.min_value, end - dim.period)]

    def bit_colour(self, bit: int) -> Colour:
        """Псевдоцвет бита: младший — красный, старший — фиолетовый (§5.14)."""
        denominator = max(1, self._encoder.code_length - 1)
        hue = self._min_hue + (self._max_hue - self._min_hue) * int(bit) / denominator
        red, green, blue = colorsys.hsv_to_rgb((hue % 360.0) / 360.0, 0.85, 0.95)
        return int(red * 255), int(green * 255), int(blue * 255)

    def _windows(self, dimension: int) -> list:
        return [w for w in self._encoder.detectors if w.dimension_index == dimension]

    @staticmethod
    def _degrees(dim: Dimension, value: float) -> float:
        return (float(value) - dim.min_value) / dim.period * 360.0

    def _dimension_text(self, dimension: int, value: object | None) -> str:
        dim = self._encoder.dimensions[dimension]
        windows = self._windows(dimension)
        counts = [sum(1 for w in windows if w.layer_index == index) for index in range(len(dim.detector_layers))]
        overlaps = sorted({layer.overlap for layer in dim.detector_layers})
        text = (
            f"{len(counts)} слоёв, детекторов {', '.join(str(c) for c in counts)} — всего {len(windows)}; "
            f"перекрытие {', '.join(f'{o:g}' for o in overlaps)}"
        )
        if value is not None:
            active = sum(1 for w in windows if w.is_active(float(value)))
            text += f"; значение {float(value):g}, сработало {active}"
        return text

    def _layer_text(self, dimension: int, layer: int) -> str:
        circles = [w for w in self._windows(dimension) if w.layer_index == layer]
        return f"{len(circles)} кругов радиуса {circles[0].radius:.2g}" if circles else ""

    def _log_view(self, values: Sequence[object] | None, image: np.ndarray) -> None:
        encoder = self._encoder
        LOGGER.event(
            "encoder.detectors.view",
            section=INTEGER_ENCODING,
            data={
                "dimension_layers": [
                    (dim.title, len(dim.detector_layers), sorted({layer.overlap for layer in dim.detector_layers}))
                    for dim in encoder.dimensions
                ],
                "detectors": len(encoder.detectors),
                "code_mode": "merge" if encoder.shared_code else "concatenation",
                "code_length": encoder.code_length,
                "stimulus": None if values is None else [str(v) for v in values],
                "note": "слои детекторов с перекрывающимися полями (§4.1, §4.2.1); цвет детектора — цвет бита (§3.2)",
            },
            visuals=[LOGGER.visual_image("encoder/detectors", image)],
        )
