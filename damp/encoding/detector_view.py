"""Визуализация первичных широких детекторов кодера: слои, рецептивные поля и их перекрытие.

Для каждой координаты (угол, X, Y) рисуется панель: по вертикали слои от самого широкого к самому
узкому (§4.2.1, §4.5), по горизонтали — диапазон значений координаты. Каждый детектор — полоса своего
рецептивного поля (§4.1); соседние детекторы слоя, поля которых перекрываются, разнесены по дорожкам,
поэтому перекрытие видно как наложение полос по горизонтали. Цвет полосы — псевдоцвет бита детектора
(младший бит красный, старший фиолетовый, §5.14): широкие слои — красная часть «спектра», узкие —
фиолетовая (§3.2, §4.4.2). Детекторы, активные для примера стимула, выделены. Нижняя панель показывает
биты кода: какие биты занимают детекторы каждой координаты и какие биты ставит пример стимула (§2.2.1,
§2.2.3); бит, который ставят детекторы разных координат, отмечен как паразитный (§4.2.1).
"""
from __future__ import annotations

import colorsys
import math
import os
from typing import Sequence

import numpy as np

from damp.article_refs import INTEGER_ENCODING, OUTSIDE_ARTICLE
from damp.encoding.damp_encoder import DetectorWindow, Dimension, Encoder
from damp.logging import LOGGER


class DetectorLayerView:
    """Картинка детекторов кодера по слоям с перекрытием и цветовыми полосами (§4.1, §4.2.1, §3.2)."""

    _LANE_FILL = 0.82  # доля высоты дорожки, занятая полосой детектора (зазор между дорожками)

    def __init__(
        self,
        encoder: Encoder,
        *,
        min_hue: float,
        max_hue: float,
        inactive_alpha: float,
        dpi: int,
        width_inches: float,
    ) -> None:
        if max_hue < min_hue:
            raise ValueError("max_hue must be >= min_hue")
        if not 0.0 <= inactive_alpha <= 1.0:
            raise ValueError("inactive_alpha must be in [0, 1]")
        self._encoder = encoder
        self._min_hue = float(min_hue)
        self._max_hue = float(max_hue)
        self._inactive_alpha = float(inactive_alpha)
        self._dpi = int(dpi)
        self._width_inches = float(width_inches)

    # ------------------------------------------------------------------ public
    def render(self, values: Sequence[float] | None = None) -> np.ndarray:
        """RGB-картинка детекторов; `values` — пример стимула (по значению на координату) или None."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        encoder = self._encoder
        if values is not None and len(values) != len(encoder.dimensions):
            raise ValueError("values must give one value per encoder dimension")
        # Высота панели — по числу дорожек: чем сильнее перекрытие, тем больше дорожек в слое.
        height_ratios = [
            max(3.0, 0.5 * sum(self._layer_lanes(dim, index) for index in range(len(dim.detector_layers))))
            for dim in encoder.dimensions
        ] + [len(encoder.dimensions) + 1.0]
        height_inches = 0.3 * sum(height_ratios) + 1.5
        figure = Figure(figsize=(self._width_inches, height_inches), dpi=self._dpi)
        grid = figure.add_gridspec(len(height_ratios), 1, height_ratios=height_ratios, hspace=0.45)
        figure.subplots_adjust(
            left=0.17, right=0.99, top=1.0 - 0.75 / height_inches, bottom=0.45 / height_inches
        )
        mode = "общий код" if encoder.shared_code else "конкатенация сегментов"
        figure.suptitle(
            f"Первичные детекторы: {mode}, {encoder.code_length} бит, ожидаемая плотность "
            f"{encoder.expected_density():.0%}"
            + ("" if values is None else f"; стимул {self._format_values(values)}"),
            fontsize=12,
            y=1.0 - 0.25 / height_inches,
        )
        for dim_index, dim in enumerate(encoder.dimensions):
            axis = figure.add_subplot(grid[dim_index, 0])
            value = None if values is None else float(values[dim_index])
            self._draw_dimension(axis, dim_index, dim, value)
        self._draw_bits(figure.add_subplot(grid[len(encoder.dimensions), 0]), values)
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        image = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
        self._log_view(values, image)
        return image

    def save(self, path: str, values: Sequence[float] | None = None) -> np.ndarray:
        from PIL import Image

        image = self.render(values)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        Image.fromarray(image).save(path)
        LOGGER.event(
            "encoder.detectors.view.save",
            section=OUTSIDE_ARTICLE,
            data={"path": path, "width": int(image.shape[1]), "height": int(image.shape[0])},
        )
        return image

    # ----------------------------------------------------------------- drawing
    def _draw_dimension(self, axis, dim_index: int, dim: Dimension, value: float | None) -> None:
        windows = [w for w in self._encoder.detectors if w.dimension_index == dim_index]
        layers = len(dim.detector_layers)
        for layer_index, layer in enumerate(dim.detector_layers):
            layer_windows = sorted((w for w in windows if w.layer_index == layer_index), key=lambda w: w.center)
            step = dim.period / layer.count
            lanes = self._layer_lanes(dim, layer_index)
            row_top = layers - 1 - layer_index  # самый широкий слой сверху
            if layer_index > 0:
                axis.axhline(row_top + 1.0, color="0.75", linewidth=0.5)
            lane_height = 1.0 / lanes
            active = 0
            for position, window in enumerate(layer_windows):
                is_active = value is not None and window.is_active(value)
                active += int(is_active)
                lane = position % lanes
                bottom = row_top + lane * lane_height + lane_height * (1.0 - self._LANE_FILL) / 2.0
                colour = self._bit_colour(window.bit)
                alpha = 1.0 if (value is None or is_active) else self._inactive_alpha
                for start, end in self._field_segments(window, dim):
                    axis.add_patch(self._rectangle(
                        (start, bottom), end - start, lane_height * self._LANE_FILL,
                        facecolor=colour, alpha=alpha,
                        edgecolor="black" if is_active else "none", linewidth=0.8,
                    ))
            band_start, band_length = dim.layer_band(layer_index)
            LOGGER.event(
                "encoder.detectors.view.layer",
                section=INTEGER_ENCODING,
                data={
                    "dimension": dim.title,
                    "layer": layer_index,
                    "detectors": len(layer_windows),
                    "width": 2.0 * layer_windows[0].half_width if layer_windows else 0.0,
                    "step": step,
                    "overlap": layer.overlap,
                    "lanes": lanes,
                    "active": active if value is not None else None,
                    "band": (band_start, band_length),
                },
            )
        if value is not None:
            axis.axvline(value, color="black", linestyle="--", linewidth=1.0)
        axis.set_xlim(dim.min_value, dim.max_value)
        axis.set_ylim(0.0, layers)
        axis.set_yticks([layers - 1 - index + 0.5 for index in range(layers)])
        axis.set_yticklabels([self._layer_label(dim, index, windows) for index in range(layers)], fontsize=7)
        overlaps = sorted({layer.overlap for layer in dim.detector_layers})
        kind = "замкнутая, период " + f"{dim.period:g}" if dim.closed else "отрезок"
        axis.set_title(
            f"{dim.title}: {kind}, {layers} слоёв, {len(windows)} детекторов, перекрытие "
            + ", ".join(f"{o:g}" for o in overlaps),
            fontsize=10, loc="left",
        )
        axis.tick_params(axis="x", labelsize=8)

    def _draw_bits(self, axis, values: Sequence[float] | None) -> None:
        encoder = self._encoder
        rows = len(encoder.dimensions) + 1
        stimulus_bits: dict[int, int] = {}
        if values is not None:
            for dim_index, value in enumerate(values):
                for bit in encoder.dimension_bits(dim_index, value):
                    stimulus_bits[bit] = stimulus_bits.get(bit, 0) + 1
        for dim_index, dim in enumerate(encoder.dimensions):
            row = rows - 1 - dim_index
            used = {w.bit for w in encoder.detectors if w.dimension_index == dim_index}
            for bit in used:
                axis.add_patch(self._rectangle((bit, row + 0.1), 1.0, 0.8,
                                               facecolor=self._bit_colour(bit), edgecolor="none"))
        if values is not None:
            for bit, count in stimulus_bits.items():
                axis.add_patch(self._rectangle(
                    (bit, 0.1), 1.0, 0.8, facecolor=self._bit_colour(bit),
                    edgecolor="black" if count > 1 else "none", linewidth=1.2,
                ))
        axis.set_xlim(0, encoder.code_length)
        axis.set_ylim(0, rows)
        labels = ["код стимула" if values is not None else "—"] + [
            f"биты {dim.title}" for dim in reversed(encoder.dimensions)
        ]
        axis.set_yticks([row + 0.5 for row in range(rows)])
        axis.set_yticklabels(labels, fontsize=7)
        parasitic = sum(1 for count in stimulus_bits.values() if count > 1)
        axis.set_title(
            f"Биты кода (младший — красный, старший — фиолетовый, §5.14)"
            + ("" if values is None else f"; у стимула {len(stimulus_bits)} бит, паразитных {parasitic} (обведены)"),
            fontsize=10, loc="left",
        )
        axis.tick_params(axis="x", labelsize=8)

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _rectangle(xy, width, height, **kwargs):
        from matplotlib.patches import Rectangle

        return Rectangle(xy, width, height, **kwargs)

    def _layer_lanes(self, dim: Dimension, layer_index: int) -> int:
        """Сколько соседних детекторов слоя накрывают одну точку: ширина поля / шаг центров (перекрытие, §4.2.1)."""
        layer = dim.detector_layers[layer_index]
        dim_index = self._encoder.dimensions.index(dim)
        window = next(w for w in self._encoder.detectors if w.dimension_index == dim_index and w.layer_index == layer_index)
        return max(1, math.ceil(2.0 * window.half_width / (dim.period / layer.count) - 1e-9))

    @staticmethod
    def _field_segments(window: DetectorWindow, dim: Dimension) -> list[tuple[float, float]]:
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

    def _bit_colour(self, bit: int) -> tuple[float, float, float]:
        denominator = max(1, self._encoder.code_length - 1)
        hue = self._min_hue + (self._max_hue - self._min_hue) * bit / denominator
        return colorsys.hsv_to_rgb((hue % 360.0) / 360.0, 0.85, 0.95)

    @staticmethod
    def _layer_label(dim: Dimension, layer_index: int, windows: Sequence[DetectorWindow]) -> str:
        layer = dim.detector_layers[layer_index]
        layer_windows = [w for w in windows if w.layer_index == layer_index]
        width = 2.0 * layer_windows[0].half_width if layer_windows else 0.0
        start, length = dim.layer_band(layer_index)
        count = f"{len(layer_windows)} дет." if len(layer_windows) == layer.count else (
            f"{len(layer_windows)} дет. ({layer.count}+{len(layer_windows) - layer.count} за краями)"
        )
        return f"L{layer_index}: {count}, ширина {width:.3g}, биты {start}–{start + length - 1}"

    def _format_values(self, values: Sequence[float]) -> str:
        return ", ".join(f"{dim.title}={float(v):g}" for dim, v in zip(self._encoder.dimensions, values))

    def _log_view(self, values: Sequence[float] | None, image: np.ndarray) -> None:
        encoder = self._encoder
        active = None
        if values is not None:
            active = {
                dim.title: sum(1 for w in encoder.detectors if w.dimension_index == index and w.is_active(float(values[index])))
                for index, dim in enumerate(encoder.dimensions)
            }
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
                "stimulus": None if values is None else tuple(float(v) for v in values),
                "active_by_dimension": active,
                "note": "слои детекторов с перекрывающимися полями (§4.1, §4.2.1); цвет полосы — цвет бита (§3.2)",
            },
            visuals=[LOGGER.visual_image("encoder/detectors", image)],
        )
