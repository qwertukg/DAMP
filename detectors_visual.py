"""Визуализация первичных детекторов угла и позиции: слои и перекрытие рецептивных полей.

Кодер берётся тот же, что в MNIST и pinwheel-тесте (main._build_encoder, §2.2.3, §4.2.1, §4.4).
Картинка сохраняется в DETECTORS_VIEW_OUTPUT_PATH и отправляется в Rerun (RERUN=on).
Все константы — в main.py (DETECTORS_VIEW_*).

Пример:
    python detectors_visual.py
    python detectors_visual.py --angle 120 --x 5 --y 20 --output detectors_120.png
"""
from __future__ import annotations

import argparse

from damp.encoding.detector_view import DetectorLayerView
from main import (
    DETECTORS_VIEW_DPI,
    DETECTORS_VIEW_INACTIVE_ALPHA,
    DETECTORS_VIEW_OUTPUT_PATH,
    DETECTORS_VIEW_STIMULUS,
    DETECTORS_VIEW_WIDTH_INCHES,
    LAYOUT_PSEUDO_COLOR_MAX_HUE,
    LAYOUT_PSEUDO_COLOR_MIN_HUE,
    _build_encoder,
    configure_logging,
)


class DetectorsVisualExperiment:
    """Строит кодер MNIST и рисует его детекторы для примера стимула."""

    def __init__(self, *, output_path: str = DETECTORS_VIEW_OUTPUT_PATH) -> None:
        self._output_path = output_path
        self._encoder = _build_encoder()
        self._view = DetectorLayerView(
            self._encoder,
            min_hue=LAYOUT_PSEUDO_COLOR_MIN_HUE,
            max_hue=LAYOUT_PSEUDO_COLOR_MAX_HUE,
            inactive_alpha=DETECTORS_VIEW_INACTIVE_ALPHA,
            dpi=DETECTORS_VIEW_DPI,
            width_inches=DETECTORS_VIEW_WIDTH_INCHES,
        )

    def run(self, stimulus: tuple[float, float, float] | None = DETECTORS_VIEW_STIMULUS) -> str:
        self._view.save(self._output_path, stimulus)
        return self._output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Первичные детекторы угла и позиции: слои и перекрытие")
    parser.add_argument("--angle", type=float, default=DETECTORS_VIEW_STIMULUS[0])
    parser.add_argument("--x", type=float, default=DETECTORS_VIEW_STIMULUS[1])
    parser.add_argument("--y", type=float, default=DETECTORS_VIEW_STIMULUS[2])
    parser.add_argument("--no-stimulus", action="store_true", help="рисовать все детекторы без выделения")
    parser.add_argument("--output", type=str, default=DETECTORS_VIEW_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configure_logging()
    stimulus = None if args.no_stimulus else (args.angle, args.x, args.y)
    DetectorsVisualExperiment(output_path=args.output).run(stimulus)


if __name__ == "__main__":
    main()
