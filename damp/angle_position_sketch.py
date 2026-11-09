"""Цифра из углопозиций: по каждой тройке (угол, X, Y) — отрезок длиной в окно Собеля.

Углопозиция — стимул кодера Angle || X || Y (§4.4, §4.6): угол градиента Собеля в окне и индексы
этого окна. Чтобы проверить, какую часть формы цифры несут углопозиции, по каждой рисуется отрезок
через центр окна вдоль края штриха (перпендикулярно градиенту) и показывается рядом с оригинальным
стимулом: для одной цифры — в окне Rerun (AnglePositionSketch), для набора цифр и окон — таблицей
в PNG и Rerun (AnglePositionSheet). Само построение отрезков в статье не описано — это диагностика.
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass
import math
import random
from typing import Sequence

import numpy as np

from damp.article_refs import CYCLIC_COORDS, OUTSIDE_ARTICLE
from damp.logging import LOGGER, DigitSketchBlueprint, DigitsSheetBlueprint

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




def stimulus_01(image, side: int) -> np.ndarray:
    """Стимул side × side со значениями в [0, 1]."""
    stimulus = np.asarray(image, dtype=np.float32)
    if stimulus.shape != (side, side):
        raise ValueError(f"stimulus must be {(side, side)}, got {stimulus.shape}")
    if stimulus.max() > 1.5:
        stimulus = stimulus / 255.0
    return np.clip(stimulus, 0.0, 1.0)


class AnglePositionLines:
    """Отрезок по углопозиции (угол, X, Y) в пикселях стимула.

    Ядро KY у MnistSobelAngleMap положительно сверху, поэтому угол отсчитан в осях «y вверх».
    В координатах картинки (y вниз) градиент направлен как (cos a, −sin a), а край штриха —
    перпендикулярно ему, (sin a, cos a). Отрезок длиной line_scale · patch_size проходит через центр
    окна (X, Y); цвет — угол по кругу оттенков, как на рис. 16.
    """

    def __init__(
        self,
        *,
        image_side: int,
        patch_size: int,
        line_scale: float,
        angle_in_degrees: bool,
        angle_range: tuple[float, float],
    ) -> None:
        if not 1 <= patch_size <= image_side:
            raise ValueError(f"patch_size must be 1…{image_side}, got {patch_size}")
        if line_scale <= 0.0:
            raise ValueError("line_scale must be positive")
        angle_min, angle_max = (float(value) for value in angle_range)
        if angle_max <= angle_min:
            raise ValueError("angle_range must be increasing")
        self.image_side = int(image_side)
        self.patch_size = int(patch_size)
        # Остаток у правого и нижнего края в сетку не входит — так же, как у MnistSobelAngleMap.
        self.grid_size = self.image_side // self.patch_size
        self.line_length = float(line_scale) * self.patch_size
        self.angle_in_degrees = bool(angle_in_degrees)
        self.angle_min = angle_min
        self.angle_period = angle_max - angle_min
        LOGGER.event(
            "angle_lines.init",
            section=OUTSIDE_ARTICLE,
            data={
                "patch_size": self.patch_size,
                "grid_size": self.grid_size,
                "dropped_edge_pixels": self.image_side - self.grid_size * self.patch_size,
                "line_length": self.line_length,
                "angle_in_degrees": self.angle_in_degrees,
                "angle_min": self.angle_min,
                "angle_period": self.angle_period,
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
            "angle_lines.angle_position",
            section=CYCLIC_COORDS,
            data={
                "angle": degrees,
                "angle_cyclic": (degrees - self.angle_min) % self.angle_period + self.angle_min,
                "x": segment.x,
                "y": segment.y,
            },
        )
        LOGGER.event(
            "angle_lines.segment",
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

    @staticmethod
    def lost_patches(
        segments: Sequence[AnglePositionSegment], contour_patches: Sequence[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Окна, через которые проходит контур, но углопозиции нет, — пропуски контура."""
        with_angle = {(segment.x, segment.y) for segment in segments}
        return [(int(x), int(y)) for x, y in contour_patches if (int(x), int(y)) not in with_angle]


class AnglePositionSketch:
    """Строит цифру из отрезков по углопозициям и показывает её рядом со стимулом в Rerun.

    pixel_scale — во сколько раз картинка, на которой ищутся контур и градиент, больше стимула
    (UpscaledSobel: 8, без предобработки: 1). Стимул показывается в этом разрешении, а сетка окон и
    отрезки рисуются в его пикселях; patch_size — сторона окна в этих же пикселях (как у
    MnistSobelAngleMap). Длина и толщина отрезка задаются в сторонах окна.
    """

    def __init__(
        self,
        *,
        image_side: int,
        patch_size: int,
        line_scale: float,
        line_radius_scale: float,
        angle_in_degrees: bool,
        angle_range: tuple[float, float],
        grid_color: Sequence[int],
        lost_color: Sequence[int],
        entity_path: str,
        sobel_path: str | None = None,
        pixel_scale: int = 1,
    ) -> None:
        if line_radius_scale <= 0.0:
            raise ValueError("line_radius_scale must be positive")
        if int(pixel_scale) < 1:
            raise ValueError("pixel_scale must be ≥ 1")
        self.pixel_scale = int(pixel_scale)
        side = int(image_side) * self.pixel_scale
        # sobel_path — где UpscaledSobel показывает увеличенную картинку и отклики (None — без предобработки).
        self.view = DigitSketchBlueprint(entity_path, side=side, sobel_root=sobel_path)
        LOGGER.use_blueprint(self.view.build)
        self.lines = AnglePositionLines(
            image_side=side,
            patch_size=int(patch_size),
            line_scale=line_scale,
            angle_in_degrees=angle_in_degrees,
            angle_range=angle_range,
        )
        self.line_radius = float(line_radius_scale) * self.lines.patch_size
        self.grid_color = tuple(int(c) for c in grid_color)
        self.lost_color = tuple(int(c) for c in lost_color)
        LOGGER.event(
            "digit_sketch.init",
            section=OUTSIDE_ARTICLE,
            data={
                "patch_size": self.lines.patch_size,
                "pixel_scale": self.pixel_scale,
                "image_side": side,
                "grid_size": self.lines.grid_size,
                "line_length": self.lines.line_length,
                "line_radius": self.line_radius,
                "entity_path": self.view.root,
            },
        )

    def show(
        self,
        image,
        label: int,
        measurements: Sequence[AngleTriplet],
        contour_patches: Sequence[tuple[int, int]],
    ) -> list[AnglePositionSegment]:
        """Логирует стимул, сетку окон и цифру из отрезков в окно Rerun.

        image — стимул в разрешении pixel_scale (при UpscaledSobel — увеличенная законтрастированная картинка).
        contour_patches — окна, через которые проходит контур цифры; те из них, что остались без
        углопозиции, подсвечиваются: это пропуски контура.
        """
        stimulus = stimulus_01(image, self.lines.image_side)
        segments = self.lines.build(measurements)
        with_angle = {(segment.x, segment.y) for segment in segments}
        contour = [(int(x), int(y)) for x, y in contour_patches]
        lost = self.lines.lost_patches(segments, contour)
        lost_set = set(lost)
        side = self.lines.patch_size
        grid = self.lines.grid_size
        grid_mins = [(x * side, y * side) for y in range(grid) for x in range(grid)]
        grid_colors = [
            self.lost_color if (x, y) in lost_set else self.grid_color
            for y in range(grid)
            for x in range(grid)
        ]
        stimulus_u8 = (stimulus * 255.0).round().astype(np.uint8)

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
                "patches_with_angle": len(with_angle),
                "contour_patches": len(contour),
                "off_contour_patches": len(with_angle - set(contour)),
                "lost_patches": len(lost),
                "lost": lost,
                "line_length": self.lines.line_length,
                "note": "слева оригинальный стимул, справа цифра из отрезков; красные окна — контур без углопозиции",
            },
            visuals=visuals,
        )
        return segments


@dataclass(frozen=True)
class SheetCell:
    """Углопозиции одного стимула при одном размере окна и окна, через которые проходит контур."""

    patch_size: int
    measurements: Sequence[AngleTriplet]
    contour_patches: Sequence[tuple[int, int]]


class AnglePositionSheet:
    """Таблица по набору стимулов: строка — стимул, столбцы — оригинал и цифра из линий для каждого окна.

    Рисуется в PNG и показывается в Rerun. На клетках с линиями — сетка окон, красным обведены окна,
    через которые проходит контур, но углопозиции нет (пропуски); подпись — окно и число углопозиций.
    """

    TEXT_COLOR = (255, 255, 255)
    TEXT_OFFSET = (6, 4)

    def __init__(
        self,
        *,
        image_side: int,
        patch_sizes: Sequence[int],
        line_scale: float,
        angle_in_degrees: bool,
        angle_range: tuple[float, float],
        scale: int,
        line_width: int,
        gap: int,
        grid_color: Sequence[int],
        lost_color: Sequence[int],
        entity_path: str,
        sobel_path: str | None = None,
        pixel_scale: int = 1,
    ) -> None:
        if not patch_sizes:
            raise ValueError("patch_sizes must not be empty")
        if scale <= 0 or line_width <= 0 or gap < 0:
            raise ValueError("scale and line_width must be positive, gap non-negative")
        if int(pixel_scale) < 1:
            raise ValueError("pixel_scale must be ≥ 1")
        self.view = DigitsSheetBlueprint(entity_path, sobel_root=sobel_path)
        LOGGER.use_blueprint(self.view.build)
        self.image_side = int(image_side)
        self.scale = int(scale)
        # Окна и отрезки — в пикселях рабочей картинки (стимул × pixel_scale), клетка таблицы — image_side · scale.
        self.pixel_scale = int(pixel_scale)
        self._draw_scale = self.scale / self.pixel_scale
        self.line_width = int(line_width)
        self.gap = int(gap)
        self.grid_color = self._opaque(grid_color)
        self.lost_color = self._opaque(lost_color)
        self.lines = {
            int(size): AnglePositionLines(
                image_side=self.image_side * self.pixel_scale,
                patch_size=int(size),
                line_scale=line_scale,
                angle_in_degrees=angle_in_degrees,
                angle_range=angle_range,
            )
            for size in patch_sizes
        }
        self._rows: list[np.ndarray] = []
        self._totals = {size: {"angle_positions": 0, "contour_patches": 0, "lost_patches": 0} for size in self.lines}
        LOGGER.event(
            "digits_sheet.init",
            section=OUTSIDE_ARTICLE,
            data={
                "patch_sizes": list(self.lines),
                "pixel_scale": self.pixel_scale,
                "scale": self.scale,
                "line_width": self.line_width,
                "gap": self.gap,
                "entity_path": self.view.root,
            },
        )

    @staticmethod
    def _opaque(color: Sequence[int]) -> tuple[int, int, int]:
        """Цвет RGBA, наложенный на чёрный фон клетки."""
        rgba = [int(c) for c in color]
        alpha = rgba[3] / 255.0 if len(rgba) > 3 else 1.0
        return tuple(int(round(c * alpha)) for c in rgba[:3])

    @property
    def _cell_side(self) -> int:
        return self.image_side * self.scale

    def _stimulus_tile(self, stimulus: MnistStimulus, display_image: np.ndarray | None) -> np.ndarray:
        from PIL import Image, ImageDraw

        image = stimulus.image if display_image is None else display_image
        side = int(np.asarray(image).shape[0])
        pixels = (stimulus_01(image, side) * 255.0).round().astype(np.uint8)
        tile = Image.fromarray(pixels).resize((self._cell_side, self._cell_side), Image.NEAREST).convert("RGB")
        caption = f"{stimulus.label}  #{stimulus.index} {stimulus.split}"
        if side != self.image_side:
            caption += f"  x{side // self.image_side}"
        ImageDraw.Draw(tile).text(self.TEXT_OFFSET, caption, fill=self.TEXT_COLOR)
        return np.asarray(tile)

    def _lines_tile(self, cell: SheetCell) -> tuple[np.ndarray, list[AnglePositionSegment], list[tuple[int, int]]]:
        from PIL import Image, ImageDraw

        lines = self.lines[int(cell.patch_size)]
        segments = lines.build(cell.measurements)
        lost = lines.lost_patches(segments, cell.contour_patches)
        k = self._draw_scale
        step = lines.patch_size * k
        # В мелких клетках (окно в пару пикселей увеличенной картинки) толстая линия превратилась бы в пятно.
        width = max(1, min(self.line_width, round(step / 4)))
        tile = Image.new("RGB", (self._cell_side, self._cell_side))
        draw = ImageDraw.Draw(tile)
        for i in range(lines.grid_size + 1):
            draw.line([(i * step, 0), (i * step, self._cell_side)], fill=self.grid_color)
            draw.line([(0, i * step), (self._cell_side, i * step)], fill=self.grid_color)
        for x, y in lost:
            draw.rectangle(
                [x * step, y * step, (x + 1) * step, (y + 1) * step], outline=self.lost_color, width=min(3, width)
            )
        for segment in segments:
            draw.line(
                [(segment.start[0] * k, segment.start[1] * k), (segment.end[0] * k, segment.end[1] * k)],
                fill=segment.color,
                width=width,
            )
        caption = f"{lines.patch_size}x{lines.patch_size}: {len(segments)}"
        if lost:
            caption += f"  lost {len(lost)}"
        draw.text(self.TEXT_OFFSET, caption, fill=self.TEXT_COLOR)
        return np.asarray(tile), segments, lost

    def _join(self, tiles: Sequence[np.ndarray], axis: int) -> np.ndarray:
        parts: list[np.ndarray] = []
        for index, tile in enumerate(tiles):
            if index and self.gap:
                gap_shape = list(tile.shape)
                gap_shape[axis] = self.gap
                parts.append(np.zeros(gap_shape, dtype=np.uint8))
            parts.append(tile)
        return np.concatenate(parts, axis=axis)

    def add(
        self, stimulus: MnistStimulus, cells: Sequence[SheetCell], display_image: np.ndarray | None = None
    ) -> None:
        """Добавляет строку: стимул и цифра из линий для каждого окна из cells.

        display_image — что показать вместо стимула в первом столбце (увеличенная картинка UpscaledSobel,
        на которой искались контур и градиент); сторона — кратная стимулу.
        """
        tiles = [self._stimulus_tile(stimulus, display_image)]
        for cell in cells:
            tile, segments, lost = self._lines_tile(cell)
            tiles.append(tile)
            totals = self._totals[int(cell.patch_size)]
            totals["angle_positions"] += len(segments)
            totals["contour_patches"] += len(cell.contour_patches)
            totals["lost_patches"] += len(lost)
            LOGGER.event(
                "digits_sheet.cell",
                section=OUTSIDE_ARTICLE,
                data={
                    "label": stimulus.label,
                    "index": stimulus.index,
                    "patch_size": int(cell.patch_size),
                    "angle_positions": len(segments),
                    "contour_patches": len(cell.contour_patches),
                    "lost_patches": len(lost),
                    "lost": lost,
                },
            )
        self._rows.append(self._join(tiles, axis=1))

    def save(self, path: str) -> np.ndarray:
        """Склеивает строки в одну картинку, пишет её в PNG и показывает в Rerun."""
        from PIL import Image

        if not self._rows:
            raise RuntimeError("add() must be called before save()")
        sheet = self._join(self._rows, axis=0)
        Image.fromarray(sheet).save(path)
        rows = len(self._rows)
        summary = {
            f"{size}x{size}": (
                f"{totals['angle_positions'] / rows:.1f} углопозиций на цифру, "
                f"пропусков {totals['lost_patches']} из {totals['contour_patches']} окон контура"
            )
            for size, totals in self._totals.items()
        }
        LOGGER.event(
            "digits_sheet.save",
            section=OUTSIDE_ARTICLE,
            data={"path": path, "rows": rows, "shape": sheet.shape, **summary},
            visuals=[LOGGER.visual_image(self.view.sheet_path, sheet)],
        )
        return sheet
