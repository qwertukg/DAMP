"""Как цифра строится из углопозиций.

    python digit_sketch.py 7
    python digit_sketch.py 7 --seed 42 --split test
    python digit_sketch.py 7 --patch-size 2
    python digit_sketch.py 7 --upscale
    python digit_sketch.py 7 --upscale --patch-size 3

Берёт случайный стимул MNIST с заданной цифрой, получает его углопозиции (угол, X, Y) тем же
MnistSobelAngleMap, что и конвейер main.py (§4.4, §4.6), и по каждой строит отрезок длиной в окно Собеля
(по умолчанию MNIST_PATCH_SIZE, ключ --patch-size меняет окно только здесь, конвейер не трогает). Окно —
в пикселях картинки, на которой считается Собель: с --upscale это пиксели увеличенной картинки.
--upscale / --no-upscale включает предобработку UpscaledSobel (×8 Lanczos3, контраст, Собель с отсечкой,
по умолчанию MNIST_SOBEL_UPSCALE_ENABLED); увеличенная картинка и отклики gx, gy — во вкладках Rerun.
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
    DIGIT_SKETCH_LINE_RADIUS_SCALE,
    DIGIT_SKETCH_LINE_SCALE,
    DIGIT_SKETCH_LOST_COLOR,
    DIGIT_SKETCH_MNIST_ROOT,
    DIGIT_SKETCH_SPLIT,
    MNIST_ANGLE_IN_DEGREES,
    MNIST_ANGLE_MAX,
    MNIST_ANGLE_MIN,
    MNIST_PATCH_SIZE,
    MNIST_SOBEL_ENTITY_PATH,
    MNIST_SOBEL_UPSCALE,
    MNIST_SOBEL_UPSCALE_ENABLED,
    _build_extractor,
)


class DigitSketchRunner:
    def __init__(
        self,
        *,
        split: str,
        seed: int | None,
        patch_size: int = MNIST_PATCH_SIZE,
        upscale: bool = MNIST_SOBEL_UPSCALE_ENABLED,
    ) -> None:
        # Окно открывается на первом логе, поэтому раскладку панелей задаём до остальных реализаций.
        self.sketch = AnglePositionSketch(
            image_side=MnistSobelAngleMap.EXPECTED_SIDE,
            patch_size=patch_size,
            line_scale=DIGIT_SKETCH_LINE_SCALE,
            line_radius_scale=DIGIT_SKETCH_LINE_RADIUS_SCALE,
            angle_in_degrees=MNIST_ANGLE_IN_DEGREES,
            angle_range=(MNIST_ANGLE_MIN, MNIST_ANGLE_MAX),
            grid_color=DIGIT_SKETCH_GRID_COLOR,
            lost_color=DIGIT_SKETCH_LOST_COLOR,
            entity_path=DIGIT_SKETCH_ENTITY_PATH,
            sobel_path=MNIST_SOBEL_ENTITY_PATH if upscale else None,
            # Рисуем на той картинке, где экстрактор ищет контур: при --upscale — увеличенной ×MNIST_SOBEL_UPSCALE.
            pixel_scale=MNIST_SOBEL_UPSCALE if upscale else 1,
        )
        self.sampler = MnistDigitSampler(root=DIGIT_SKETCH_MNIST_ROOT, split=split, seed=seed)
        self.extractor = _build_extractor(patch_size, upscale=upscale)

    def run(self, digit: int) -> None:
        stimulus = self.sampler.sample(digit)
        measurements = self.extractor.extract(stimulus.image, stimulus.label)[stimulus.label]
        contour = self.extractor.contour_patches(stimulus.image)
        image = self.extractor.work_image(stimulus.image)
        self.sketch.show(image, stimulus.label, measurements, contour)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Цифра MNIST из линий её углопозиций")
    parser.add_argument("digit", type=int, choices=range(10), help="цифра 0…9")
    parser.add_argument("--seed", type=int, default=None, help="seed выбора стимула; по умолчанию случайный")
    parser.add_argument("--split", choices=["train", "test"], default=DIGIT_SKETCH_SPLIT)
    parser.add_argument(
        "--patch-size",
        type=int,
        default=MNIST_PATCH_SIZE,
        help="сторона окна Собеля в пикселях картинки, где считается Собель (с --upscale — увеличенной), "
        "только для этого скрипта; по умолчанию MNIST_PATCH_SIZE из main.py",
    )
    parser.add_argument(
        "--upscale",
        action=argparse.BooleanOptionalAction,
        default=MNIST_SOBEL_UPSCALE_ENABLED,
        help="предобработка ×8 Lanczos3 + контраст + Собель с отсечкой; по умолчанию MNIST_SOBEL_UPSCALE_ENABLED",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    DigitSketchRunner(
        split=args.split, seed=args.seed, patch_size=args.patch_size, upscale=args.upscale
    ).run(args.digit)


if __name__ == "__main__":
    main()
