from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict, List, Tuple, Union

import numpy as np

from damp.article_refs import CYCLIC_COORDS, SLIDING_WINDOW, TWO_DIM_CODES
from damp.logging import LOGGER


AngleTriplet = Tuple[float, int, int]      # (angle, X_patch, Y_patch)
ResultMap = Dict[int, List[AngleTriplet]]  # label -> list of (angle, x, y)


@dataclass
class MnistSobelAngleMap:
    """
    Берёт MNIST-стимул (28x28) + смысл (label) и возвращает:
        {label: [(angle, X, Y), ...]}
    где X,Y — индексы 4x4-участка (0..6),
    angle — угол градиента (по Sobel) для этого участка.

    Квадрат 4x4 пропускается, если magnitude(meanGx, meanGy) <= grad_threshold.
    """

    angle_in_degrees: bool = True
    grad_threshold: float = 1e-3  # порог для пропуска "плоских/чёрных" квадратов

    # ВАЖНО: это не поля dataclass, а константы класса (ClassVar)
    PATCH_SIZE: ClassVar[int] = 4
    GRID_SIZE: ClassVar[int] = 7
    EXPECTED_SIDE: ClassVar[int] = PATCH_SIZE * GRID_SIZE
    EXPECTED_SHAPE: ClassVar[Tuple[int, int]] = (EXPECTED_SIDE, EXPECTED_SIDE)
    PATCHES_TOTAL: ClassVar[int] = GRID_SIZE * GRID_SIZE
    KX: ClassVar[np.ndarray] = np.array([[-1, 0, 1],
                                         [-2, 0, 2],
                                         [-1, 0, 1]], dtype=np.float32)
    KY: ClassVar[np.ndarray] = np.array([[ 1,  2,  1],
                                         [ 0,  0,  0],
                                         [-1, -2, -1]], dtype=np.float32)

    def __post_init__(self) -> None:
        LOGGER.event(
            "sobel_map.init",
            section=CYCLIC_COORDS,
            data={
                "angle_in_degrees": self.angle_in_degrees,
                "grad_threshold": self.grad_threshold,
            },
        )

    def extract(self, image: Union[np.ndarray, "np.typing.ArrayLike"], label: int) -> ResultMap:
        img = np.asarray(image, dtype=np.float32)

        if img.shape != self.EXPECTED_SHAPE:
            raise ValueError(
                f"Ожидаю MNIST {self.EXPECTED_SIDE}x{self.EXPECTED_SIDE}, получил {img.shape}"
            )

        # нормализуем, если пришло 0..255
        normalized = False
        if img.max() > 1.5:
            img = img / 255.0
            normalized = True

        LOGGER.event(
            "sobel_map.patch_grid",
            section=SLIDING_WINDOW,
            data={
                "label": int(label),
                "image_shape": img.shape,
                "patch_size": self.PATCH_SIZE,
                "grid_size": self.GRID_SIZE,
                "patches_total": self.PATCHES_TOTAL,
            },
        )
        LOGGER.event(
            "sobel_map.patch_coords",
            section=TWO_DIM_CODES,
            data={
                "x_range": (0, self.GRID_SIZE - 1),
                "y_range": (0, self.GRID_SIZE - 1),
                "patch_size": self.PATCH_SIZE,
            },
        )

        out: List[AngleTriplet] = []

        for by in range(self.GRID_SIZE):
            y0 = by * self.PATCH_SIZE
            for bx in range(self.GRID_SIZE):
                x0 = bx * self.PATCH_SIZE
                patch = img[y0:y0 + self.PATCH_SIZE, x0:x0 + self.PATCH_SIZE]

                mean_gx, mean_gy = self._sobel_mean_vector_on_4x4(patch)

                mag = float(np.hypot(mean_gx, mean_gy))
                kept = mag > self.grad_threshold
                LOGGER.event(
                    "sobel_map.patch.metrics",
                    section=SLIDING_WINDOW,
                    data={
                        "x": bx,
                        "y": by,
                        "x0": x0,
                        "y0": y0,
                        "patch_size": self.PATCH_SIZE,
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

                out.append((angle, bx, by))

        LOGGER.event(
            "sobel_map.extract",
            section=CYCLIC_COORDS,
            data={
                "label": int(label),
                "normalized": normalized,
                "patch_size": self.PATCH_SIZE,
                "grid_size": self.GRID_SIZE,
                "patches_total": self.PATCHES_TOTAL,
                "patches_kept": len(out),
            },
        )
        return {int(label): out}


    @classmethod
    def _sobel_mean_vector_on_4x4(cls, patch4: np.ndarray) -> Tuple[float, float]:
        if patch4.shape != (4, 4):
            raise ValueError(f"Ожидаю 4x4 patch, получил {patch4.shape}")

        gx_vals: List[float] = []
        gy_vals: List[float] = []

        # valid-окна 3x3: (iy,ix) = 0..1
        for iy in range(2):
            for ix in range(2):
                window = patch4[iy:iy + 3, ix:ix + 3]
                gx_vals.append(float(np.sum(window * cls.KX)))
                gy_vals.append(float(np.sum(window * cls.KY)))

        return float(np.mean(gx_vals)), float(np.mean(gy_vals))
