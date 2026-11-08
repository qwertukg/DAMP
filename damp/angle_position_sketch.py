"""Цифра из углопозиций: по каждой тройке (угол, X, Y) — отрезок длиной в окно Собеля.

Углопозиция — стимул кодера Angle || X || Y (§4.4, §4.6): угол среднего градиента Собеля в окне 4×4
и индексы этого окна. Чтобы проверить, какую часть формы цифры несут углопозиции, по каждой рисуется
отрезок через центр окна вдоль края штриха (перпендикулярно градиенту) и показывается рядом с
оригинальным стимулом. Само построение отрезков в статье не описано — это диагностика.
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass
import math
import random
from typing import Sequence

import numpy as np

from damp.article_refs import CYCLIC_COORDS, OUTSIDE_ARTICLE
from damp.logging import LOGGER, DigitSketchBlueprint

AngleTriplet = tuple[float, int, int]


@dataclass(frozen=True)
class MnistStimulus:
    image: np.ndarray
    label: int
    index: int
    split: str


class MnistDigitSampler:
    """Случайный стимул MNIST с заданной цифрой; seed логируется, чтобы выбор можно было повторить."""

    DIGITS = range(10)

    def __init__(self, *, root: str, split: str, seed: int | None = None) -> None:
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        self.root = root
        self.split = split
        self.seed = random.SystemRandom().randrange(2**32) if seed is None else int(seed)
        self._random = random.Random(self.seed)
        self._dataset = None
        LOGGER.event(
            "digit_sketch.sampler.init",
            section=OUTSIDE_ARTICLE,
            data={"root": root, "split": split, "seed": self.seed, "seed_given": seed is not None},
        )

    def _load(self):
        if self._dataset is None:
            from torchvision import transforms
            from torchvision.datasets import MNIST

            self._dataset = MNIST(
                root=self.root,
                train=self.split == "train",
                download=True,
                transform=transforms.ToTensor(),
            )
        return self._dataset

    def sample(self, digit: int) -> MnistStimulus:
        if int(digit) not in self.DIGITS:
            raise ValueError(f"digit must be 0…9, got {digit}")
        dataset = self._load()
        candidates = (dataset.targets == int(digit)).nonzero().flatten().tolist()
        index = int(self._random.choice(candidates))
        image_tensor, label = dataset[index]
        image = image_tensor.squeeze(0).numpy()
        LOGGER.event(
            "digit_sketch.stimulus",
            section=OUTSIDE_ARTICLE,
            data={
                "digit": int(digit),
                "label": int(label),
                "split": self.split,
                "index": index,
                "candidates": len(candidates),
                "seed": self.seed,
                "shape": image.shape,
            },
        )
        return MnistStimulus(image=image, label=int(label), index=index, split=self.split)


@dataclass(frozen=True)
class AnglePositionSegment:
    angle: float
    x: int
    y: int
    start: tuple[float, float]
    end: tuple[float, float]
    color: tuple[int, int, int]


class AnglePositionSketch:
    """Строит цифру из отрезков по углопозициям и показывает её рядом с оригинальным стимулом.

    Ядро KY у MnistSobelAngleMap положительно сверху, поэтому угол отсчитан в осях «y вверх».
    В координатах картинки (y вниз) градиент направлен как (cos a, −sin a), а край штриха —
    перпендикулярно ему, (sin a, cos a). Отрезок длиной line_length проходит через центр окна (X, Y).
    """

    def __init__(
        self,
        *,
        patch_size: int,
        grid_size: int,
        line_length: float,
        line_radius: float,
        angle_in_degrees: bool,
        angle_range: tuple[float, float],
        ink_threshold: float,
        grid_color: Sequence[int],
        lost_color: Sequence[int],
        entity_path: str,
    ) -> None:
        if patch_size <= 0 or grid_size <= 0:
            raise ValueError("patch_size and grid_size must be positive")
        if line_length <= 0.0 or line_radius <= 0.0:
            raise ValueError("line_length and line_radius must be positive")
        angle_min, angle_max = (float(value) for value in angle_range)
        if angle_max <= angle_min:
            raise ValueError("angle_range must be increasing")
        self.patch_size = int(patch_size)
        self.grid_size = int(grid_size)
        self.line_length = float(line_length)
        self.line_radius = float(line_radius)
        self.angle_in_degrees = bool(angle_in_degrees)
        self.angle_min = angle_min
        self.angle_period = angle_max - angle_min
        self.ink_threshold = float(ink_threshold)
        self.grid_color = tuple(int(c) for c in grid_color)
        self.lost_color = tuple(int(c) for c in lost_color)
        self.view = DigitSketchBlueprint(entity_path, side=self.patch_size * self.grid_size)
        LOGGER.use_blueprint(self.view.build)
        LOGGER.event(
            "digit_sketch.init",
            section=OUTSIDE_ARTICLE,
            data={
                "patch_size": self.patch_size,
                "grid_size": self.grid_size,
                "line_length": self.line_length,
                "line_radius": self.line_radius,
                "angle_in_degrees": self.angle_in_degrees,
                "angle_min": self.angle_min,
                "angle_period": self.angle_period,
                "ink_threshold": self.ink_threshold,
                "entity_path": self.view.root,
            },
        )

    def _angle_degrees(self, angle: float) -> float:
        return float(angle) if self.angle_in_degrees else math.degrees(float(angle))

    def _angle_color(self, degrees: float) -> tuple[int, int, int]:
        hue = ((degrees - self.angle_min) % self.angle_period) / self.angle_period
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return int(r * 255), int(g * 255), int(b * 255)

    def segment(self, angle: float, x: int, y: int) -> AnglePositionSegment:
        degrees = self._angle_degrees(angle)
        radians = math.radians(degrees)
        direction_x, direction_y = math.sin(radians), math.cos(radians)
        center_x = (int(x) + 0.5) * self.patch_size
        center_y = (int(y) + 0.5) * self.patch_size
        half = self.line_length / 2.0
        segment = AnglePositionSegment(
            angle=degrees,
            x=int(x),
            y=int(y),
            start=(center_x - half * direction_x, center_y - half * direction_y),
            end=(center_x + half * direction_x, center_y + half * direction_y),
            color=self._angle_color(degrees),
        )
        LOGGER.event(
            "digit_sketch.angle_position",
            section=CYCLIC_COORDS,
            data={
                "angle": degrees,
                "angle_cyclic": (degrees - self.angle_min) % self.angle_period + self.angle_min,
                "x": segment.x,
                "y": segment.y,
            },
        )
        LOGGER.event(
            "digit_sketch.segment",
            section=OUTSIDE_ARTICLE,
            data={
                "x": segment.x,
                "y": segment.y,
                "center": (center_x, center_y),
                "start": tuple(round(v, 3) for v in segment.start),
                "end": tuple(round(v, 3) for v in segment.end),
                "length": self.line_length,
                "color": segment.color,
            },
        )
        return segment

    def build(self, measurements: Sequence[AngleTriplet]) -> list[AnglePositionSegment]:
        return [self.segment(angle, x, y) for angle, x, y in measurements]

    def ink_patches(self, image: np.ndarray) -> list[tuple[int, int]]:
        """Окна (X, Y), в которых есть пиксель штриха ярче ink_threshold."""
        side = self.patch_size
        patches: list[tuple[int, int]] = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                patch = image[y * side:(y + 1) * side, x * side:(x + 1) * side]
                if patch.size and float(patch.max()) >= self.ink_threshold:
                    patches.append((x, y))
        return patches

    def show(self, image, label: int, measurements: Sequence[AngleTriplet]) -> list[AnglePositionSegment]:
        """Логирует оригинальный стимул, сетку окон и цифру из отрезков в окно Rerun."""
        stimulus = np.asarray(image, dtype=np.float32)
        expected = (self.patch_size * self.grid_size,) * 2
        if stimulus.shape != expected:
            raise ValueError(f"stimulus must be {expected}, got {stimulus.shape}")
        if stimulus.max() > 1.5:
            stimulus = stimulus / 255.0
        segments = self.build(measurements)

        with_angle = {(segment.x, segment.y) for segment in segments}
        ink = self.ink_patches(stimulus)
        lost = [patch for patch in ink if patch not in with_angle]
        lost_set = set(lost)
        side = self.patch_size
        grid_mins = [(x * side, y * side) for y in range(self.grid_size) for x in range(self.grid_size)]
        grid_colors = [
            self.lost_color if (x, y) in lost_set else self.grid_color
            for y in range(self.grid_size)
            for x in range(self.grid_size)
        ]
        stimulus_u8 = (np.clip(stimulus, 0.0, 1.0) * 255.0).round().astype(np.uint8)

        visuals = [
            LOGGER.visual_image(self.view.stimulus_path, stimulus_u8),
            LOGGER.visual_boxes2d(
                self.view.grid_path,
                grid_mins,
                [(side, side)] * len(grid_mins),
                colors=grid_colors,
            ),
            LOGGER.visual_line_strips2d(
                self.view.lines_path,
                [[segment.start, segment.end] for segment in segments],
                colors=[segment.color for segment in segments],
                radii=self.line_radius,
            ),
        ]
        LOGGER.event(
            "digit_sketch.show",
            section=OUTSIDE_ARTICLE,
            data={
                "label": int(label),
                "angle_positions": len(measurements),
                "segments": len(segments),
                "ink_patches": len(ink),
                "lost_patches": len(lost),
                "lost": lost,
                "line_length": self.line_length,
                "note": "слева оригинальный стимул, справа цифра из отрезков; красные окна — штрих без углопозиции",
            },
            visuals=visuals,
        )
        return segments
