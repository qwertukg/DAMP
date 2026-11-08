"""Как цифра строится из углопозиций.

    python digit_sketch.py 7
    python digit_sketch.py 7 --seed 42 --split test

Берёт случайный стимул MNIST с заданной цифрой, получает его углопозиции (угол, X, Y) тем же
MnistSobelAngleMap, что и конвейер main.py (§4.4, §4.6), и по каждой строит отрезок длиной в окно 4×4.
В окне Rerun рядом: оригинальный стимул и цифра, собранная только из этих линий.
seed выбора печатается в логе — с ним тот же стимул можно открыть снова.
Константы — в main.py (DIGIT_SKETCH_*), логи — внутри реализаций.
"""
from __future__ import annotations

import argparse

from damp.MnistSobelAngleMap import MnistSobelAngleMap
from damp.angle_position_sketch import AnglePositionSketch, MnistDigitSampler
from main import (
    DIGIT_SKETCH_ENTITY_PATH,
    DIGIT_SKETCH_GRID_COLOR,
    DIGIT_SKETCH_INK_THRESHOLD,
    DIGIT_SKETCH_LINE_LENGTH,
    DIGIT_SKETCH_LINE_RADIUS,
    DIGIT_SKETCH_LOST_COLOR,
    DIGIT_SKETCH_MNIST_ROOT,
    DIGIT_SKETCH_SPLIT,
    MNIST_ANGLE_IN_DEGREES,
    MNIST_ANGLE_MAX,
    MNIST_ANGLE_MIN,
    _build_extractor,
)


class DigitSketchRunner:
    def __init__(self, *, split: str, seed: int | None) -> None:
        # Окно открывается на первом логе, поэтому раскладку панелей задаём до остальных реализаций.
        self.sketch = AnglePositionSketch(
            patch_size=MnistSobelAngleMap.PATCH_SIZE,
            grid_size=MnistSobelAngleMap.GRID_SIZE,
            line_length=DIGIT_SKETCH_LINE_LENGTH,
            line_radius=DIGIT_SKETCH_LINE_RADIUS,
            angle_in_degrees=MNIST_ANGLE_IN_DEGREES,
            angle_range=(MNIST_ANGLE_MIN, MNIST_ANGLE_MAX),
            ink_threshold=DIGIT_SKETCH_INK_THRESHOLD,
            grid_color=DIGIT_SKETCH_GRID_COLOR,
            lost_color=DIGIT_SKETCH_LOST_COLOR,
            entity_path=DIGIT_SKETCH_ENTITY_PATH,
        )
        self.sampler = MnistDigitSampler(root=DIGIT_SKETCH_MNIST_ROOT, split=split, seed=seed)
        self.extractor = _build_extractor()

    def run(self, digit: int) -> None:
        stimulus = self.sampler.sample(digit)
        measurements = self.extractor.extract(stimulus.image, stimulus.label)[stimulus.label]
        self.sketch.show(stimulus.image, stimulus.label, measurements)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Цифра MNIST из линий её углопозиций")
    parser.add_argument("digit", type=int, choices=range(10), help="цифра 0…9")
    parser.add_argument("--seed", type=int, default=None, help="seed выбора стимула; по умолчанию случайный")
    parser.add_argument("--split", choices=["train", "test"], default=DIGIT_SKETCH_SPLIT)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    DigitSketchRunner(split=args.split, seed=args.seed).run(args.digit)


if __name__ == "__main__":
    main()
