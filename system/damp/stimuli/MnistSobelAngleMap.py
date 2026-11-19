from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np

from damp.article_refs import CYCLIC_COORDS, OUTSIDE_ARTICLE
from damp.logging import LOGGER
from damp.stimuli.stroke_detectors import StrokeDetectors
from damp.stimuli.stroke_midline import StrokeMidline


AngleTriplet = Tuple[float, int, int]      # (angle, X_patch, Y_patch)
ResultMap = Dict[int, List[AngleTriplet]]  # label -> list of (angle, x, y)


@dataclass(frozen=True)
class UpscaledSobel:
    """Предобработка перед Собелем как в process_image (Rust): увеличение, контраст, Собель.

    Повторяет 8-битную арифметику оригинала: стимул → u8; увеличение ×scale фильтром Lanczos3 (u8);
    контраст по формуле image::imageops::contrast — отклонение от середины × ((100 + contrast) / 100)²,
    отсечка 0…255, дробная часть отбрасывается; Собель в целых числах, крайние пиксели картинки — ноль
    (в оригинале цикл идёт по 1..width-1). Отклики делятся на 255, чтобы остаться в единицах картинки
    [0, 1]: grad_threshold и contour_level не меняют смысла. Оси — как у MnistSobelAngleMap (X вправо,
    Y вверх); в оригинале имена gx и gy переставлены (gx там считается вертикальным ядром).

    Отсечка ±255 из оригинала нужна только чтобы уложить отклик в u8-каналы картинки, поэтому здесь она
    применяется лишь к картинкам gx, gy в Rerun. Углы считаются по полному отклику: после контраста край —
    ступенька, отклик на ней до ±1020, и если обрезать компоненты по отдельности, у любого края с наклоном
    ~15…75° к оси обе упираются в 255 — угол слипается в 45°.
    """

    DISPLAY_CLAMP: ClassVar[int] = 255

    scale: int = 8
    contrast: float = 100.0
    entity_path: str = "sobel"

    def __post_init__(self) -> None:
        if int(self.scale) < 1:
            raise ValueError(f"scale должен быть ≥ 1, получил {self.scale}")
        LOGGER.event(
            "upscaled_sobel.init",
            section=OUTSIDE_ARTICLE,
            data={
                "scale": int(self.scale),
                "contrast": self.contrast,
                "contrast_factor": self.contrast_factor,
                "display_clamp": self.DISPLAY_CLAMP,
                "entity_path": self.root,
            },
        )

    @property
    def root(self) -> str:
        return "/" + self.entity_path.strip("/")

    @property
    def contrast_factor(self) -> float:
        """Во сколько раз растягивается отклонение яркости от середины (как в image::imageops::contrast)."""
        return ((100.0 + self.contrast) / 100.0) ** 2

    @property
    def signature(self) -> str:
        return f"upscale={int(self.scale)},contrast={self.contrast:g},clamp=display"

    def image(self, img01: np.ndarray) -> np.ndarray:
        """Стимул, увеличенный ×scale фильтром Lanczos3 и законтрастированный, uint8."""
        from PIL import Image

        pixels = np.clip(np.rint(np.asarray(img01, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)
        height, width = pixels.shape
        scale = int(self.scale)
        resized = Image.fromarray(pixels).resize((width * scale, height * scale), Image.LANCZOS)
        values = np.asarray(resized, dtype=np.float32)
        contrasted = np.clip(((values / 255.0 - 0.5) * self.contrast_factor + 0.5) * 255.0, 0.0, 255.0)
        return np.floor(contrasted).astype(np.uint8)

    def sobel(self, image_u8: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Отклики (gx, gy) без отсечки в единицах [0, 1] и доля ненулевых, что не влезли бы в ±255 (u8)."""
        values = image_u8.astype(np.int32)
        height, width = values.shape
        gx = np.zeros((height, width), dtype=np.int32)
        gy = np.zeros((height, width), dtype=np.int32)
        inner = (slice(1, height - 1), slice(1, width - 1))
        for dy in range(3):
            for dx in range(3):
                shifted = values[dy:dy + height - 2, dx:dx + width - 2]
                gx[inner] += int(kx[dy, dx]) * shifted
                gy[inner] += int(ky[dy, dx]) * shifted
        limit = self.DISPLAY_CLAMP
        nonzero = int(np.count_nonzero(gx) + np.count_nonzero(gy))
        over = int(np.count_nonzero(np.abs(gx) > limit) + np.count_nonzero(np.abs(gy) > limit))
        saturated = over / nonzero if nonzero else 0.0
        return gx.astype(np.float32) / 255.0, gy.astype(np.float32) / 255.0, saturated

    @classmethod
    def signed_rgb(cls, g: np.ndarray) -> np.ndarray:
        """Как в оригинале: отклик обрезан до ±255, плюс — красный канал, минус — синий (только картинка)."""
        limit = cls.DISPLAY_CLAMP
        values = np.clip(np.rint(g * 255.0), -limit, limit).astype(np.int32)
        rgb = np.zeros(g.shape + (3,), dtype=np.uint8)
        rgb[..., 0] = np.clip(values, 0, 255)
        rgb[..., 2] = np.clip(-values, 0, 255)
        return rgb

    def apply(self, img01: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Увеличенная картинка в [0, 1] и отклики Собеля на ней; картинки — в Rerun."""
        upscaled = self.image(img01)
        gx, gy, saturated = self.sobel(upscaled, kx, ky)
        log = LOGGER.should_log("upscaled_sobel.apply")
        visuals = None
        if log:
            visuals = [
                LOGGER.visual_image(f"{self.root}/image", upscaled),
                LOGGER.visual_image(f"{self.root}/gx", self.signed_rgb(gx)),
                LOGGER.visual_image(f"{self.root}/gy", self.signed_rgb(gy)),
            ]
        LOGGER.event(
            "upscaled_sobel.apply",
            section=OUTSIDE_ARTICLE,
            data={
                "shape": upscaled.shape,
                "scale": int(self.scale),
                "contrast_factor": self.contrast_factor,
                "display_clamp": self.DISPLAY_CLAMP,
                "saturated_fraction": saturated,
                "note": "углы — по полному отклику; ±255 только у картинок gx, gy",
            },
            visuals=visuals,
            force=log,
        )
        return upscaled.astype(np.float32) / 255.0, gx, gy


@dataclass
class MnistSobelAngleMap:
    """
    Берёт MNIST-стимул (28x28) + смысл (label) и возвращает:
        {label: [(angle, X, Y), ...]}
    где X,Y — индексы участка patch_size×patch_size пикселей рабочей картинки (0..grid_size-1),
    angle — угол среднего градиента (по Sobel) на этом участке.

    Участок пропускается, если magnitude(meanGx, meanGy) <= grad_threshold.

    Отклики ядра 3×3 берутся в тех пикселях участка, где ядро целиком лежит внутри участка
    (для 4×4 — центральные 2×2, для 3×3 — центр). Если участок меньше ядра (patch_size < 3), таких
    пикселей нет: тогда отклик берётся в каждом пикселе участка и ядро захватывает соседние пиксели
    картинки (за краем картинки — нули, фон MNIST чёрный).

    Сетка — 28 // patch_size участков от левого верхнего угла. Если 28 не делится на patch_size,
    остаток у правого и нижнего края (для 3 — один пиксель) ни в один участок не попадает.

    contour_only=True — углопозиции только по контуру цифры. Контур — пиксели штриха
    (яркость ≥ contour_level), у которых хотя бы один из 4 соседей — фон. Участок без пикселей
    контура пропускается (залитая середина штриха и фон), угол считается по градиенту Собеля только
    в пикселях контура, поэтому середина штриха в него не подмешивается. Каждый пиксель контура лежит
    ровно в одном участке, поэтому на границах участков контур не рвётся. Участок даёт ровно одну
    углопозицию. Если края в участке смотрят в противоположные стороны (тонкий штрих целиком в участке:
    когерентность |Σg| / Σ|g| ниже two_edge_coherence), их градиенты гасили бы друг друга — тогда
    берётся сильнейший край. Если градиент в пикселях контура всё равно гасится (линия толщиной в пиксель:
    в её середине у Собеля ноль, края лежат в соседних пикселях фона), край ищется в участке вместе
    с кольцом соседних пикселей.

    upscaled — предобработка UpscaledSobel (увеличение, контраст, Собель с отсечкой, как в Rust-коде).
    Рабочая картинка — та, на которой считается Собель: стимул 28×28 или его увеличенная копия
    (28·scale). patch_size и сетка участков — в пикселях рабочей картинки: при ×8 участок 3 — это
    3×3 увеличенных пикселя, сетка 224 // 3 = 74 участка, остаток у правого и нижнего края отбрасывается.

    subcell_position=True (только с contour_only) — место углопозиции дробное: там, где край проходит внутри
    участка (центр масс пикселей контура выбранного края с весом силы градиента; у линии в пиксель — центр масс
    её пикселей по яркости), в единицах участка: целое X — середина участка. Без него X, Y — номер участка, и
    наклонная линия ложится ступеньками по серединам участков (вне статьи).

    stroke_detectors (только с contour_only) — штрихи не по сетке участков, а детекторами над точками края
    (StrokeDetectors: §6.4 на плоскости картинки, доработка): место штриха — центр детектора, в тех же единицах
    участка, дробное; угол — по куску края длиннее шага; сетка участков тогда не используется.
    """

    angle_in_degrees: bool = True
    grad_threshold: float = 1e-3  # порог для пропуска "плоских/чёрных" квадратов
    patch_size: int = 4
    contour_only: bool = False
    contour_level: float = 0.5
    two_edge_coherence: float = 0.5
    upscaled: Optional[UpscaledSobel] = None
    subcell_position: bool = False
    stroke_detectors: Optional[StrokeDetectors] = None
    # Точки середины линии вместо точек двух её краёв (только со stroke_detectors, у которых axial=True).
    stroke_midline: Optional[StrokeMidline] = None

    # ВАЖНО: это не поля dataclass, а константы класса (ClassVar)
    EXPECTED_SIDE: ClassVar[int] = 28
    EXPECTED_SHAPE: ClassVar[Tuple[int, int]] = (EXPECTED_SIDE, EXPECTED_SIDE)
    KERNEL_SIZE: ClassVar[int] = 3
    KX: ClassVar[np.ndarray] = np.array([[-1, 0, 1],
                                         [-2, 0, 2],
                                         [-1, 0, 1]], dtype=np.float32)
    KY: ClassVar[np.ndarray] = np.array([[ 1,  2,  1],
                                         [ 0,  0,  0],
                                         [-1, -2, -1]], dtype=np.float32)

    @classmethod
    def load_image(cls, path: str) -> np.ndarray:
        """Картинка из файла в градациях серого 28×28 со значениями в [0, 1]."""
        from PIL import Image

        target_h, target_w = cls.EXPECTED_SHAPE
        image = Image.open(path).convert("L").resize((target_w, target_h))
        array = np.asarray(image, dtype="float32")
        if array.max() > 1.5:
            array = array / 255.0
        LOGGER.event(
            "sobel_map.image.load",
            section=OUTSIDE_ARTICLE,
            data={
                "path": path,
                "shape": array.shape,
                "max": float(array.max()) if array.size else 0.0,
                "min": float(array.min()) if array.size else 0.0,
            },
        )
        return array

    def __post_init__(self) -> None:
        self.patch_size = int(self.patch_size)
        if not 1 <= self.patch_size <= self.work_side:
            raise ValueError(
                f"patch_size должен быть от 1 до {self.work_side} (сторона рабочей картинки), "
                f"получил {self.patch_size}"
            )
        if not 0.0 < self.contour_level <= 1.0:
            raise ValueError(f"contour_level должен быть в (0, 1], получил {self.contour_level}")
        if not 0.0 <= self.two_edge_coherence <= 1.0:
            raise ValueError(f"two_edge_coherence должен быть в [0, 1], получил {self.two_edge_coherence}")
        LOGGER.event(
            "sobel_map.init",
            section=OUTSIDE_ARTICLE,
            data={
                "angle_in_degrees": self.angle_in_degrees,
                "grad_threshold": self.grad_threshold,
                "patch_size": self.patch_size,
                "grid_size": self.grid_size,
                "covered_side": self.covered_side,
                "work_side": self.work_side,
                "dropped_edge_pixels": self.work_side - self.covered_side,
                "mode": self.stimulus_mode,
                "work_patch": self.work_patch,
                "kernel_inside_patch": self.work_patch >= self.KERNEL_SIZE,
                "sobel_centers": None if self.contour_only else list(self._sobel_centers(self.work_patch)),
            },
        )

    @property
    def stimulus_mode(self) -> str:
        """Как участок превращается в углопозицию; входит в проверку совместимости кеша раскладки."""
        mode = "window"
        if self.contour_only:
            mode = f"contour(level={self.contour_level:g},edge={self.two_edge_coherence:g})"
        if self.subcell_position:
            mode += ",subcell"  # место дробное: старая раскладка с целыми местами не подходит
        if self.contour_only and self.stroke_detectors is not None:
            mode += "," + self.stroke_detectors.signature  # штрихи детекторами над точками края, не по сетке
            if self.stroke_midline is not None:
                mode += "," + self.stroke_midline.signature  # точки — середина линии, а не два её края
        if self.upscaled is not None:
            # patch_size — в пикселях увеличенной картинки (до этой версии было в пикселях стимула).
            mode += f",{self.upscaled.signature},patch=work_px"
        return mode

    @property
    def scale(self) -> int:
        """Во сколько раз увеличена картинка, на которой ищутся контур и градиент."""
        return 1 if self.upscaled is None else int(self.upscaled.scale)

    @property
    def work_side(self) -> int:
        """Сторона рабочей картинки (на которой ищутся контур и градиент): 28 или 28·scale."""
        return self.EXPECTED_SIDE * self.scale

    @property
    def work_patch(self) -> int:
        """Сторона участка в пикселях рабочей картинки — это и есть patch_size."""
        return self.patch_size

    @property
    def grid_size(self) -> int:
        return self.work_side // self.patch_size

    @property
    def covered_side(self) -> int:
        """Сколько пикселей рабочей картинки от левого (верхнего) края покрывают участки сетки."""
        return self.grid_size * self.patch_size

    @property
    def patches_total(self) -> int:
        return self.grid_size * self.grid_size

    def _sobel_centers(self, side: Optional[int] = None) -> range:
        """Пиксели участка со стороной side (по каждой оси), в которых берётся отклик ядра."""
        side = self.patch_size if side is None else int(side)
        margin = self.KERNEL_SIZE // 2
        if side >= self.KERNEL_SIZE:
            return range(margin, side - margin)
        return range(side)

    def _normalized(self, image: Union[np.ndarray, "np.typing.ArrayLike"]) -> Tuple[np.ndarray, bool]:
        img = np.asarray(image, dtype=np.float32)
        if img.shape != self.EXPECTED_SHAPE:
            raise ValueError(
                f"Ожидаю MNIST {self.EXPECTED_SIDE}x{self.EXPECTED_SIDE}, получил {img.shape}"
            )
        # нормализуем, если пришло 0..255
        if img.max() > 1.5:
            return img / 255.0, True
        return img, False

    def _patch_slice(self, bx: int, by: int) -> Tuple[slice, slice]:
        """Участок (X, Y) на рабочей картинке (исходной или увеличенной)."""
        p = self.work_patch
        return slice(by * p, (by + 1) * p), slice(bx * p, (bx + 1) * p)

    def _work_image(self, img: np.ndarray) -> np.ndarray:
        """Картинка, на которой ищется контур: стимул или его увеличенная законтрастированная копия."""
        if self.upscaled is None:
            return img
        return self.upscaled.image(img).astype(np.float32) / 255.0

    def work_image(self, image: Union[np.ndarray, "np.typing.ArrayLike"]) -> np.ndarray:
        """Картинка в [0, 1], на которой ищутся контур и градиент (при upscaled — увеличенная ×scale)."""
        img, _ = self._normalized(image)
        work = self._work_image(img)
        LOGGER.event(
            "sobel_map.work_image",
            section=OUTSIDE_ARTICLE,
            data={"shape": work.shape, "scale": self.scale, "upscale": None if self.upscaled is None else self.upscaled.signature},
        )
        return work

    def contour_mask(self, image: Union[np.ndarray, "np.typing.ArrayLike"]) -> np.ndarray:
        """Пиксели контура на рабочей картинке: штрих (≥ contour_level) рядом с фоном (4 соседа)."""
        img, _ = self._normalized(image)
        return self._contour_mask_of(self._work_image(img))

    def _contour_mask_of(self, work: np.ndarray) -> np.ndarray:
        ink = work >= self.contour_level
        padded = np.pad(ink, 1, constant_values=False)
        background_near = ~padded[:-2, 1:-1] | ~padded[2:, 1:-1] | ~padded[1:-1, :-2] | ~padded[1:-1, 2:]
        return ink & background_near

    def contour_patches(self, image: Union[np.ndarray, "np.typing.ArrayLike"]) -> List[Tuple[int, int]]:
        """Участки (X, Y), через которые проходит контур: в каждом должна быть углопозиция."""
        mask = self.contour_mask(image)
        covered = self.covered_side
        patches = [
            (bx, by)
            for by in range(self.grid_size)
            for bx in range(self.grid_size)
            if mask[self._patch_slice(bx, by)].any()
        ]
        LOGGER.event(
            "sobel_map.contour_patches",
            section=OUTSIDE_ARTICLE,
            data={
                "contour_level": self.contour_level,
                "scale": self.scale,
                "contour_pixels": int(mask.sum()),
                "contour_pixels_outside_grid": int(mask.sum() - mask[:covered, :covered].sum()),
                "contour_patches": len(patches),
            },
        )
        return patches

    def extract(
        self, image: Union[np.ndarray, "np.typing.ArrayLike"], label: int, with_magnitude: bool = False
    ) -> ResultMap:
        """Углопозиции картинки: (угол, x, y); при with_magnitude — (угол, x, y, сила, полудлина поля): сила — модуль
        среднего градиента либо энергия детектора, полудлина поля штриха вдоль края — в участках."""
        img, normalized = self._normalized(image)

        LOGGER.event(
            "sobel_map.patch_grid",
            section=OUTSIDE_ARTICLE,
            data={
                "label": int(label),
                "image_shape": img.shape,
                "patch_size": self.patch_size,
                "grid_size": self.grid_size,
                "patches_total": self.patches_total,
            },
        )
        LOGGER.event(
            "sobel_map.patch_coords",
            section=OUTSIDE_ARTICLE,
            data={
                "x_range": (0, self.grid_size - 1),
                "y_range": (0, self.grid_size - 1),
                "patch_size": self.patch_size,
            },
        )

        by_detectors = self.contour_only and self.stroke_detectors is not None
        if self.upscaled is not None:
            work, gx, gy = self.upscaled.apply(img, self.KX, self.KY)
            if by_detectors:
                out = self._detector_strokes(work, gx, gy)
            elif self.contour_only:
                out = self._contour_angles(work, gx, gy)
            else:
                out = self._window_angles(self._field_mean_vector(gx, gy))
        elif self.contour_only:
            gx, gy = self._sobel_field(img)
            out = self._detector_strokes(img, gx, gy) if by_detectors else self._contour_angles(img, gx, gy)
        else:
            # Нули вокруг картинки: ядру у края нужен сосед. Внутри участка ≥ 3×3 они не используются.
            padded = np.pad(img, self.KERNEL_SIZE // 2)
            p = self.patch_size
            out = self._window_angles(lambda bx, by: self._sobel_mean_vector(padded, bx * p, by * p))

        LOGGER.event(
            "sobel_map.extract",
            section=OUTSIDE_ARTICLE,
            data={
                "label": int(label),
                "normalized": normalized,
                "mode": self.stimulus_mode,
                "patch_size": self.patch_size,
                "grid_size": self.grid_size,
                "patches_total": self.patches_total,
                "patches_kept": len({(item[1], item[2]) for item in out}),
                "angle_positions": len(out),
            },
        )
        if not with_magnitude:
            out = [item[:3] for item in out]
        else:
            # Пятое число — полудлина поля штриха вдоль края, в участках (до самой дальней точки поля детектора, §6.4.3);
            # у штриха по сетке поле — его участок: половина участка в каждую сторону.
            out = [item if len(item) == 5 else (*item, 0.5) for item in out]
        return {int(label): out}

    def _field_mean_vector(self, gx: np.ndarray, gy: np.ndarray) -> Callable[[int, int], Tuple[float, float]]:
        """Средний отклик по пикселям участка, где ядро целиком внутри участка (рабочая картинка)."""
        centers = self._sobel_centers(self.work_patch)
        p = self.work_patch

        def mean_vector(bx: int, by: int) -> Tuple[float, float]:
            rows = slice(by * p + centers.start, by * p + centers.stop)
            cols = slice(bx * p + centers.start, bx * p + centers.stop)
            return (
                float(np.mean(gx[rows, cols], dtype=np.float64)),
                float(np.mean(gy[rows, cols], dtype=np.float64)),
            )

        return mean_vector

    def _window_angles(self, mean_vector: Callable[[int, int], Tuple[float, float]]) -> List[AngleTriplet]:
        """Углопозиция участка — средний градиент Собеля по участку (режим "window")."""
        out: List[AngleTriplet] = []

        for by in range(self.grid_size):
            y0 = by * self.patch_size
            for bx in range(self.grid_size):
                x0 = bx * self.patch_size

                mean_gx, mean_gy = mean_vector(bx, by)

                mag = float(np.hypot(mean_gx, mean_gy))
                kept = mag > self.grad_threshold
                LOGGER.event(
                    "sobel_map.patch.metrics",
                    section=OUTSIDE_ARTICLE,
                    data={
                        "x": bx,
                        "y": by,
                        "x0": x0,
                        "y0": y0,
                        "patch_size": self.patch_size,
                        "mean_gx": mean_gx,
                        "mean_gy": mean_gy,
                        "magnitude": mag,
                        "threshold": self.grad_threshold,
                        "kept": kept,
                    },
                )
                if not kept:
                    continue

                angle = float(np.arctan2(mean_gy, mean_gx))
                if self.angle_in_degrees:
                    angle = float(np.degrees(angle))

                LOGGER.event(
                    "sobel_map.patch.angle",
                    section=CYCLIC_COORDS,
                    data={
                        "x": bx,
                        "y": by,
                        "angle": angle,
                        "magnitude": mag,
                    },
                )

                out.append((angle, bx, by, mag))

        return out

    def _sobel_field(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Отклик ядер KX, KY с центром в каждом пикселе картинки (за краем — нули)."""
        k = self.KERNEL_SIZE
        padded = np.pad(img, k // 2)
        height, width = img.shape
        gx = np.zeros_like(img)
        gy = np.zeros_like(img)
        for ky in range(k):
            for kx in range(k):
                shifted = padded[ky:ky + height, kx:kx + width]
                gx += self.KX[ky, kx] * shifted
                gy += self.KY[ky, kx] * shifted
        return gx, gy

    @staticmethod
    def _opposite_edges(vx: np.ndarray, vy: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Делит градиенты участка на два края по знаку проекции на главную ось (тензор структуры)."""
        axis = 0.5 * np.arctan2(2.0 * np.sum(vx * vy), np.sum(vx * vx - vy * vy))
        side = vx * np.cos(axis) + vy * np.sin(axis) >= 0.0
        return [(vx[side], vy[side]), (vx[~side], vy[~side])]

    def _strongest_edge(self, vx: np.ndarray, vy: np.ndarray) -> Tuple[float, float, float, bool]:
        """Средний градиент края: (mean_gx, mean_gy, когерентность, были ли в участке два края).

        Когерентность |Σg| / Σ|g| ниже two_edge_coherence — края противоположны и гасили бы друг
        друга; тогда градиенты делятся на два края и берётся сильнейший (по Σ|g|).
        """
        if vx.size == 0:
            return 0.0, 0.0, 0.0, False
        total = float(np.sum(np.hypot(vx, vy)))
        coherence = float(np.hypot(vx.sum(), vy.sum())) / total if total > 0.0 else 0.0
        two_edges = coherence < self.two_edge_coherence
        if two_edges:
            vx, vy = max(self._opposite_edges(vx, vy), key=lambda edge: float(np.sum(np.hypot(*edge))))
        return float(vx.mean()), float(vy.mean()), coherence, two_edges

    def _edge_place(
        self, bx: int, by: int, on_contour: np.ndarray, gx: np.ndarray, gy: np.ndarray, work: np.ndarray,
        mean_gx: float, mean_gy: float, ridge: bool,
    ) -> Tuple[float, float]:
        """Где край проходит внутри участка (bx, by), в единицах участка: целое число — середина участка (вне статьи).

        Центр масс пикселей контура, чей градиент смотрит туда же, что выбранный край (вес — сила градиента): при
        двух краях в участке берётся тот, по которому мерился угол. У линии в пиксель (ridge: градиент в контуре
        гасится) — центр масс пикселей линии по яркости. Нет веса — середина участка.
        """
        rows, cols = np.nonzero(on_contour)
        if ridge:
            weights = work[on_contour].astype(np.float64)
        else:
            vx, vy = gx[on_contour].astype(np.float64), gy[on_contour].astype(np.float64)
            weights = np.hypot(vx, vy) * ((vx * mean_gx + vy * mean_gy) > 0.0)
        total = float(weights.sum())
        if total <= 0.0:
            return float(bx), float(by)
        p = float(self.work_patch)
        x = bx + float((cols + 0.5) @ weights) / total / p - 0.5
        y = by + float((rows + 0.5) @ weights) / total / p - 0.5
        return x, y

    def _detector_strokes(self, work: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> List[AngleTriplet]:
        """Углопозиции детекторами над точками края (§6.4 на плоскости картинки, доработка): по одной на детектор.

        Место — центр детектора в единицах участка (целое — середина участка), как у дробного места по сетке:
        остальной пайплайн меряет места в тех же единицах. Угол — направление суммы градиентов по куску края
        длиннее шага; сила — энергия детектора (§6.3).
        """
        p = float(self.work_patch)
        out: List[AngleTriplet] = []
        if self.stroke_midline is not None:
            # Середина линии вместо двух краёв (замысел автора 2026-11-17): точки — гребень расстояния до края линии,
            # угол — 0…180°.
            y, x, vx, vy = self.stroke_midline.points(work)
            found, points = self.stroke_detectors.strokes_of(y, x, vx, vy), int(y.shape[0])
        else:
            mask = self._contour_mask_of(work)
            found, points = self.stroke_detectors.strokes(mask, gx, gy), int(mask.sum())
        for stroke in found:
            angle = float(np.degrees(stroke.angle)) if self.angle_in_degrees else float(stroke.angle)
            out.append((angle, stroke.x / p - 0.5, stroke.y / p - 0.5, stroke.strength, stroke.half_length / p))
        LOGGER.event(
            "sobel_map.stroke_detectors",
            section=OUTSIDE_ARTICLE,
            data={
                "contour_level": self.contour_level,
                "scale": self.scale,
                "contour_pixels": points,
                "angle_positions": len(out),
                "method": self.stroke_detectors.signature,
                "midline": None if self.stroke_midline is None else self.stroke_midline.signature,
            },
        )
        return out

    def stroke_points(self, image: Union[np.ndarray, "np.typing.ArrayLike"]) -> dict[str, np.ndarray]:
        """Штрихи картинки вместе с точками, из которых они собраны, — вход уровня 0 по общему подходу
        (LADDER_LEVEL0_PLACE = "stroke"; доработка): атом — штрих, его элементы — точки его поля (§6.4.3).

        Те же штрихи, что у extract(with_magnitude=True), в том же порядке. Места — в участках, как у штрихов.

        Returns:
            strokes — (S, 5): угол, x, y, сила, полудлина поля (как у extract); point_y, point_x — места точек;
            point_angle — направление точки в тех же единицах, что угол штриха (у середины линии — без стрелки);
            point_strength — сила точки, доля сильнейшей на картинке; entry_stroke, entry_point — пары
            «номер штриха — номер точки его поля», штрихи по возрастанию.

        Raises:
            ValueError: штрихи не детекторами (нужны MNIST_CONTOUR_ONLY и MNIST_STROKE_DETECTORS).
        """
        if not (self.contour_only and self.stroke_detectors is not None):
            raise ValueError("точки штрихов есть только у штрихов-детекторов (MNIST_CONTOUR_ONLY, MNIST_STROKE_DETECTORS)")
        img, _ = self._normalized(image)
        if self.upscaled is not None:
            work, gx, gy = self.upscaled.apply(img, self.KX, self.KY)
        else:
            work = img
            gx, gy = self._sobel_field(img)
        if self.stroke_midline is not None:
            y, x, vx, vy = self.stroke_midline.points(work)
        else:
            y, x, vx, vy = self.stroke_detectors._edge_points(self._contour_mask_of(work), gx, gy)
        found, fields = self.stroke_detectors.strokes_with_fields(y, x, vx, vy)
        p = float(self.work_patch)
        y, x = np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)
        vx, vy = np.asarray(vx, dtype=np.float64), np.asarray(vy, dtype=np.float64)
        magnitude = np.hypot(vx, vy)
        # Направление точки — как угол штриха из одной этой точки: без стрелки — 0…180° (удвоенный угол), иначе — угол вектора.
        raw = np.mod(np.arctan2(vy, vx), np.pi) if self.stroke_detectors.axial else np.arctan2(vy, vx)
        strokes = np.asarray(
            [
                (float(np.degrees(s.angle)) if self.angle_in_degrees else float(s.angle), s.x / p - 0.5, s.y / p - 0.5,
                 s.strength, s.half_length / p)
                for s in found
            ],
            dtype=np.float64,
        ).reshape(-1, 5)
        entry_stroke = np.concatenate([np.full(f.shape[0], i, dtype=np.int64) for i, f in enumerate(fields)]) if fields else np.zeros(0, np.int64)
        entry_point = np.concatenate(fields).astype(np.int64) if fields else np.zeros(0, np.int64)
        return {
            "strokes": strokes,
            "point_y": y / p - 0.5,
            "point_x": x / p - 0.5,
            "point_angle": np.degrees(raw) if self.angle_in_degrees else raw,
            "point_strength": magnitude / float(magnitude.max()) if magnitude.size else magnitude,
            "entry_stroke": entry_stroke,
            "entry_point": entry_point,
        }

    def _contour_angles(self, work: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> List[AngleTriplet]:
        """Углопозиции только по контуру: одна на участок, угол — средний градиент края в пикселях контура."""
        out: List[AngleTriplet] = []
        mask = self._contour_mask_of(work)
        contour_patches = 0
        two_edge_patches = 0
        ridge_patches = 0

        for by in range(self.grid_size):
            for bx in range(self.grid_size):
                window = self._patch_slice(bx, by)
                on_contour = mask[window]
                contour_pixels = int(on_contour.sum())
                if contour_pixels == 0:
                    LOGGER.event(
                        "sobel_map.patch.metrics",
                        section=OUTSIDE_ARTICLE,
                        data={"x": bx, "y": by, "patch_size": self.patch_size, "contour_pixels": 0, "kept": False},
                    )
                    continue
                contour_patches += 1
                mean_gx, mean_gy, coherence, two_edges = self._strongest_edge(
                    gx[window][on_contour].astype(np.float64), gy[window][on_contour].astype(np.float64)
                )
                mag = float(np.hypot(mean_gx, mean_gy))
                ridge = mag <= self.grad_threshold
                if ridge:
                    # Линия в пиксель: у Собеля в её середине ноль, края — в соседних пикселях фона.
                    p = self.work_patch
                    ring = (
                        slice(max(0, by * p - 1), (by + 1) * p + 1),
                        slice(max(0, bx * p - 1), (bx + 1) * p + 1),
                    )
                    mean_gx, mean_gy, coherence, two_edges = self._strongest_edge(
                        gx[ring].ravel().astype(np.float64), gy[ring].ravel().astype(np.float64)
                    )
                    mag = float(np.hypot(mean_gx, mean_gy))
                kept = mag > self.grad_threshold
                two_edge_patches += two_edges
                ridge_patches += ridge
                LOGGER.event(
                    "sobel_map.patch.metrics",
                    section=OUTSIDE_ARTICLE,
                    data={
                        "x": bx,
                        "y": by,
                        "x0": bx * self.patch_size,
                        "y0": by * self.patch_size,
                        "patch_size": self.patch_size,
                        "contour_pixels": contour_pixels,
                        "coherence": coherence,
                        "two_edges": two_edges,
                        "ridge": ridge,
                        "mean_gx": mean_gx,
                        "mean_gy": mean_gy,
                        "magnitude": mag,
                        "threshold": self.grad_threshold,
                        "kept": kept,
                    },
                )
                if not kept:
                    continue
                angle = float(np.arctan2(mean_gy, mean_gx))
                if self.angle_in_degrees:
                    angle = float(np.degrees(angle))
                x, y = bx, by
                if self.subcell_position:
                    x, y = self._edge_place(bx, by, on_contour, gx[window], gy[window], work[window], mean_gx, mean_gy, ridge)
                LOGGER.event(
                    "sobel_map.patch.angle",
                    section=CYCLIC_COORDS,
                    data={"x": bx, "y": by, "place": [x, y], "angle": angle, "magnitude": mag},
                )
                out.append((angle, x, y, mag))

        LOGGER.event(
            "sobel_map.contour",
            section=OUTSIDE_ARTICLE,
            data={
                "contour_level": self.contour_level,
                "two_edge_coherence": self.two_edge_coherence,
                "scale": self.scale,
                "contour_pixels": int(mask.sum()),
                "contour_patches": contour_patches,
                "two_edge_patches": two_edge_patches,
                "ridge_patches": ridge_patches,
                "lost_patches": contour_patches - len(out),
                "angle_positions": len(out),
            },
        )
        return out

    def _sobel_mean_vector(self, padded: np.ndarray, x0: int, y0: int) -> Tuple[float, float]:
        """Средний отклик Собеля по пикселям участка с левым верхним углом (x0, y0)."""
        gx_vals: List[float] = []
        gy_vals: List[float] = []
        centers = self._sobel_centers()
        k = self.KERNEL_SIZE

        # Пиксель (y0+cy, x0+cx) картинки лежит в padded со сдвигом k//2, поэтому его окно k×k
        # начинается в padded[y0+cy, x0+cx]. Для 4×4 это те же valid-окна 3×3 внутри участка.
        for cy in centers:
            for cx in centers:
                window = padded[y0 + cy:y0 + cy + k, x0 + cx:x0 + cx + k]
                gx_vals.append(float(np.sum(window * self.KX)))
                gy_vals.append(float(np.sum(window * self.KY)))

        return float(np.mean(gx_vals)), float(np.mean(gy_vals))
