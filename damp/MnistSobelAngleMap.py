from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict, List, Tuple, Union

import numpy as np

from damp.article_refs import CYCLIC_COORDS
from damp.logging import log_event


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
    KX: ClassVar[np.ndarray] = np.array([[-1, 0, 1],
                                         [-2, 0, 2],
                                         [-1, 0, 1]], dtype=np.float32)
    KY: ClassVar[np.ndarray] = np.array([[ 1,  2,  1],
                                         [ 0,  0,  0],
                                         [-1, -2, -1]], dtype=np.float32)

    def __post_init__(self) -> None:
        log_event(
            "sobel_map.init",
            section=CYCLIC_COORDS,
            data={
                "angle_in_degrees": self.angle_in_degrees,
                "grad_threshold": self.grad_threshold,
            },
        )

    def extract(self, image: Union[np.ndarray, "np.typing.ArrayLike"], label: int) -> ResultMap:
        img = np.asarray(image, dtype=np.float32)

        if img.shape != (28, 28):
            raise ValueError(f"Ожидаю MNIST 28x28, получил {img.shape}")

        # нормализуем, если пришло 0..255
        normalized = False
        if img.max() > 1.5:
            img = img / 255.0
            normalized = True

        out: List[AngleTriplet] = []

        for by in range(7):
            y0 = by * 4
            for bx in range(7):
                x0 = bx * 4
                patch = img[y0:y0 + 4, x0:x0 + 4]

                mean_gx, mean_gy = self._sobel_mean_vector_on_4x4(patch)

                mag = float(np.hypot(mean_gx, mean_gy))
                if mag <= self.grad_threshold:
                    continue

                angle = float(np.arctan2(mean_gy, mean_gx))
                if self.angle_in_degrees:
                    angle = float(np.degrees(angle))

                out.append((angle, bx, by))

        log_event(
            "sobel_map.extract",
            section=CYCLIC_COORDS,
            data={
                "label": int(label),
                "normalized": normalized,
                "patches_total": 49,
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
