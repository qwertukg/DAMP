"""Цифры 0…9 из контурных углопозиций — таблица «оригинал | 2×2 в режиме контура».

    python digits_contour_test.py
    python digits_contour_test.py --seed 5 --split test
    python digits_contour_test.py --upscale

По случайному стимулу каждой цифры (seed фиксирован, DIGITS_CONTOUR_SEED) получает углопозиции тем же
MnistSobelAngleMap, что и конвейер main.py (§4.4, §4.6), но всегда в режиме контура, и строит строку
таблицы: оригинал и цифра из линий для каждого окна из DIGITS_CONTOUR_PATCH_SIZES. Красным обведены окна,
через которые проходит контур, но углопозиции нет (пропуски). Таблица — DIGITS_CONTOUR_OUTPUT_PATH и окно
Rerun; в логе по каждой цифре — число углопозиций и пропуски, в конце — сводка. --upscale / --no-upscale
включает предобработку UpscaledSobel (×8 Lanczos3, контраст, Собель с отсечкой; MNIST_SOBEL_UPSCALE_ENABLED).
Константы — в main.py (DIGITS_CONTOUR_*), логи — внутри реализаций.
"""
from __future__ import annotations

import argparse

from damp.MnistSobelAngleMap import MnistSobelAngleMap
from damp.angle_position_sketch import AnglePositionSheet, MnistDigitSampler, SheetCell
from main import (
    DIGIT_SKETCH_GRID_COLOR,
    DIGIT_SKETCH_LINE_SCALE,
    DIGIT_SKETCH_LOST_COLOR,
    DIGIT_SKETCH_MNIST_ROOT,
    DIGIT_SKETCH_SPLIT,
    DIGITS_CONTOUR_DIGITS,
    DIGITS_CONTOUR_ENTITY_PATH,
    DIGITS_CONTOUR_GAP,
    DIGITS_CONTOUR_LINE_WIDTH,
    DIGITS_CONTOUR_OUTPUT_PATH,
    DIGITS_CONTOUR_PATCH_SIZES,
    DIGITS_CONTOUR_SCALE,
    DIGITS_CONTOUR_SEED,
    MNIST_ANGLE_IN_DEGREES,
    MNIST_ANGLE_MAX,
    MNIST_ANGLE_MIN,
    MNIST_SOBEL_ENTITY_PATH,
    MNIST_SOBEL_UPSCALE,
    MNIST_SOBEL_UPSCALE_ENABLED,
    _build_extractor,
    configure_logging,
)


class DigitsContourExperiment:
    def __init__(self, *, seed: int, split: str, upscale: bool = MNIST_SOBEL_UPSCALE_ENABLED) -> None:
        # Окно открывается на первом логе, поэтому раскладку панелей задаём до остальных реализаций.
        self.sheet = AnglePositionSheet(
            image_side=MnistSobelAngleMap.EXPECTED_SIDE,
            patch_sizes=DIGITS_CONTOUR_PATCH_SIZES,
            line_scale=DIGIT_SKETCH_LINE_SCALE,
            angle_in_degrees=MNIST_ANGLE_IN_DEGREES,
            angle_range=(MNIST_ANGLE_MIN, MNIST_ANGLE_MAX),
            scale=DIGITS_CONTOUR_SCALE,
            line_width=DIGITS_CONTOUR_LINE_WIDTH,
            gap=DIGITS_CONTOUR_GAP,
            grid_color=DIGIT_SKETCH_GRID_COLOR,
            lost_color=DIGIT_SKETCH_LOST_COLOR,
            entity_path=DIGITS_CONTOUR_ENTITY_PATH,
            sobel_path=MNIST_SOBEL_ENTITY_PATH if upscale else None,
            # Окна DIGITS_CONTOUR_PATCH_SIZES — в пикселях картинки, где считается Собель (с --upscale — увеличенной).
            pixel_scale=MNIST_SOBEL_UPSCALE if upscale else 1,
        )
        self.sampler = MnistDigitSampler(root=DIGIT_SKETCH_MNIST_ROOT, split=split, seed=seed)
        self.extractors = {
            size: _build_extractor(size, contour_only=True, upscale=upscale) for size in DIGITS_CONTOUR_PATCH_SIZES
        }
        self.upscale = upscale

    def run(self) -> None:
        for digit in DIGITS_CONTOUR_DIGITS:
            stimulus = self.sampler.sample(digit)
            cells = [
                SheetCell(
                    patch_size=size,
                    measurements=extractor.extract(stimulus.image, stimulus.label)[stimulus.label],
                    contour_patches=extractor.contour_patches(stimulus.image),
                )
                for size, extractor in self.extractors.items()
            ]
            # При --upscale в первом столбце — увеличенная картинка, на которой искались контур и градиент.
            first = next(iter(self.extractors.values()))
            display = first.work_image(stimulus.image) if self.upscale else None
            self.sheet.add(stimulus, cells, display_image=display)
        self.sheet.save(DIGITS_CONTOUR_OUTPUT_PATH)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Цифры 0…9 из контурных углопозиций: оригинал | 2×2")
    parser.add_argument("--seed", type=int, default=DIGITS_CONTOUR_SEED, help="seed выбора стимулов")
    parser.add_argument("--split", choices=["train", "test"], default=DIGIT_SKETCH_SPLIT)
    parser.add_argument(
        "--upscale",
        action=argparse.BooleanOptionalAction,
        default=MNIST_SOBEL_UPSCALE_ENABLED,
        help="предобработка ×8 Lanczos3 + контраст + Собель с отсечкой; по умолчанию MNIST_SOBEL_UPSCALE_ENABLED",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    configure_logging()
    DigitsContourExperiment(seed=args.seed, split=args.split, upscale=args.upscale).run()


if __name__ == "__main__":
    main()
