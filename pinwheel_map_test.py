"""Вариант pinwheel_test с центрами по звёздочкам map.png.

Запуск: python pinwheel_map_test.py
Центры задают входные стимулы; итоговые позиции определяет общий Layout.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from damp.article_refs import LAID_OUT_STRUCTURE, MULTI_DIM_CODES
from damp.encoding.orientation_encoder import RetinotopicOrientationEncoder
from damp.layout.damp_layout import Layout
from damp.logging import LOGGER
from main import (
    PINWHEEL_LONG_ANGLE_LAYERS,
    PINWHEEL_LONG_SPATIAL_JOINT_LAYERS,
    PINWHEEL_MAP_BOUNDS_PX,
    PINWHEEL_MAP_CENTERS_PX,
    PINWHEEL_MAP_LAYOUT_PATH,
    PINWHEEL_MAP_LOG_PATH,
    PINWHEEL_MAP_LONG_LAYOUT_PATH,
    PINWHEEL_MAP_OUTPUT_PATH,
    PINWHEEL_MAP_REPORT_PATH,
    PINWHEEL_MAP_SOURCE_PATH,
    PINWHEEL_SHORT_ANGLE_LAYERS,
    PINWHEEL_SHORT_SPATIAL_JOINT_LAYERS,
    configure_logging,
)
from pinwheel_test import (
    PinwheelExperiment,
    PinwheelQualityEvaluator,
    RetinotopicAngleSurface,
)


class MapAngleSurface(RetinotopicAngleSurface):
    """Переносит отмеченные центры с единым масштабом обеих осей (§4.6)."""

    def __init__(self) -> None:
        left, top, right, bottom = PINWHEEL_MAP_BOUNDS_PX
        scale = right - left
        if scale <= 0 or bottom <= top:
            raise ValueError("Map bounds must have positive width and height")
        if any(not (left <= x < right and top <= y < bottom)
               for x, y in PINWHEEL_MAP_CENTERS_PX):
            raise ValueError("A marked center falls outside the map")
        centers = tuple(
            ((x - left) / scale, (y - top) / scale)
            for x, y in PINWHEEL_MAP_CENTERS_PX
        )
        super().__init__(centers)
        for index, (pixel, center) in enumerate(zip(PINWHEEL_MAP_CENTERS_PX, centers)):
            LOGGER.event(
                "pinwheel.map.center", section=MULTI_DIM_CODES,
                data={"index": index, "pixel_xy": pixel, "center_xy": center,
                      "scale_px": scale, "y_direction": "down"},
            )


class MapQualityEvaluator(PinwheelQualityEvaluator):
    """Сохраняет метрики общего теста и происхождение входных координат."""

    def evaluate(
        self, layout: Layout, encoder: RetinotopicOrientationEncoder,
    ) -> Mapping[str, float | int | bool | str]:
        metrics = super().evaluate(layout, encoder)
        report = {
            "source_image": PINWHEEL_MAP_SOURCE_PATH,
            "map_bounds_px": PINWHEEL_MAP_BOUNDS_PX,
            "centers_px": PINWHEEL_MAP_CENTERS_PX,
            "centers": [(chart.center_x, chart.center_y)
                        for chart in self._surface.topology.charts],
            "coordinate_rule": "x=(pixel_x-left)/width; y=(pixel_y-top)/width",
            "center_measurement": "manually read star centers, pixel approximation",
            "quality": dict(metrics),
            "image": PINWHEEL_MAP_OUTPUT_PATH,
            "layout": PINWHEEL_MAP_LAYOUT_PATH,
        }
        Path(PINWHEEL_MAP_REPORT_PATH).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        LOGGER.event(
            "pinwheel.map.report.save", section=LAID_OUT_STRUCTURE,
            data={"report": PINWHEEL_MAP_REPORT_PATH, "positions": len(report["centers"])},
        )
        return metrics


class MapPinwheelExperiment(PinwheelExperiment):
    """Переиспользует кодирование, обе фазы и сохранение исходного теста."""

    def __init__(self) -> None:
        surface = MapAngleSurface()
        long_encoder = self.build_encoder(
            PINWHEEL_LONG_SPATIAL_JOINT_LAYERS, PINWHEEL_LONG_ANGLE_LAYERS,
            spatial_bounds=surface.spatial_bounds, surface=surface,
        )
        short_encoder = self.build_encoder(
            PINWHEEL_SHORT_SPATIAL_JOINT_LAYERS, PINWHEEL_SHORT_ANGLE_LAYERS,
            spatial_bounds=surface.spatial_bounds, surface=surface,
        )
        super().__init__(
            surface, long_encoder, short_encoder,
            output_path=PINWHEEL_MAP_OUTPUT_PATH,
            layout_path=PINWHEEL_MAP_LAYOUT_PATH,
            long_layout_path=PINWHEEL_MAP_LONG_LAYOUT_PATH,
        )
        self._quality = MapQualityEvaluator(surface)


def main() -> None:
    configure_logging()
    LOGGER.configure_file(
        all_paths=(PINWHEEL_MAP_LOG_PATH,), active_path=PINWHEEL_MAP_LOG_PATH,
        events=None,
    )
    MapPinwheelExperiment().run()


if __name__ == "__main__":
    main()
