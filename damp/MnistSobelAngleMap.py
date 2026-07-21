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
    local_orientation_limit: int | None = None

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
        if self.local_orientation_limit is not None and self.local_orientation_limit <= 0:
            raise ValueError("local_orientation_limit must be positive or None")
        LOGGER.event(
            "sobel_map.init",
            section=CYCLIC_COORDS,
            data={
                "angle_in_degrees": self.angle_in_degrees,
                "grad_threshold": self.grad_threshold,
                "local_orientation_limit": self.local_orientation_limit,
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

        gx_values, gy_values = self._all_patch_vectors(img)
        return self._format_result(
            gx_values,
            gy_values,
            label=int(label),
            normalized=normalized,
            image_shape=img.shape,
        )

    def extract_many(
        self,
        images: Union[np.ndarray, "np.typing.ArrayLike"],
        labels: List[int],
    ) -> List[ResultMap]:
        batch = np.asarray(images, dtype=np.float32)
        if batch.ndim != 3 or batch.shape[1:] != self.EXPECTED_SHAPE:
            raise ValueError(
                f"Ожидаю пакет MNIST Nx{self.EXPECTED_SIDE}x{self.EXPECTED_SIDE}, "
                f"получил {batch.shape}"
            )
        if len(labels) != len(batch):
            raise ValueError("number of labels must match image batch")
        normalized_mask = np.max(batch, axis=(1, 2)) > 1.5
        if np.any(normalized_mask):
            batch = batch.copy()
            batch[normalized_mask] /= 255.0
        gx_values, gy_values = self._all_patch_vectors(batch)
        results = [
            self._format_result(
                gx_values[index],
                gy_values[index],
                label=int(labels[index]),
                normalized=bool(normalized_mask[index]),
                image_shape=self.EXPECTED_SHAPE,
            )
            for index in range(len(batch))
        ]
        LOGGER.event(
            "sobel_map.extract.batch",
            section=SLIDING_WINDOW,
            data={"images": len(batch), "patches_per_image": self.PATCHES_TOTAL},
        )
        return results

    @classmethod
    def _all_patch_vectors(
        cls, images: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        single = images.ndim == 2
        batch = images[None, ...] if single else images
        patches = batch.reshape(
            len(batch), cls.GRID_SIZE, cls.PATCH_SIZE,
            cls.GRID_SIZE, cls.PATCH_SIZE,
        ).transpose(0, 1, 3, 2, 4)
        windows = np.lib.stride_tricks.sliding_window_view(
            patches, (3, 3), axis=(-2, -1)
        )
        leading_shape = windows.shape[:-2]
        gx_products = np.ascontiguousarray(windows * cls.KX).reshape(*leading_shape, 9)
        gy_products = np.ascontiguousarray(windows * cls.KY).reshape(*leading_shape, 9)
        gx = np.sum(gx_products, axis=-1, dtype=np.float32)
        gy = np.sum(gy_products, axis=-1, dtype=np.float32)
        gx = gx.reshape(len(batch), cls.GRID_SIZE, cls.GRID_SIZE, 4)
        gy = gy.reshape(len(batch), cls.GRID_SIZE, cls.GRID_SIZE, 4)
        return (gx[0], gy[0]) if single else (gx, gy)

    def _format_result(
        self,
        gx_values: np.ndarray,
        gy_values: np.ndarray,
        *,
        label: int,
        normalized: bool,
        image_shape: tuple[int, int],
    ) -> ResultMap:

        LOGGER.event(
            "sobel_map.patch_grid",
            section=SLIDING_WINDOW,
            data={
                "label": label,
                "image_shape": image_shape,
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
            for bx in range(self.GRID_SIZE):
                y0 = by * self.PATCH_SIZE
                x0 = bx * self.PATCH_SIZE
                gx_patch = gx_values[by, bx]
                gy_patch = gy_values[by, bx]
                magnitudes = np.hypot(gx_patch, gy_patch)
                if self.local_orientation_limit is None:
                    gx = float(np.mean(gx_patch))
                    gy = float(np.mean(gy_patch))
                    magnitude = float(np.hypot(gx, gy))
                    vectors = (
                        [(magnitude, gx, gy)]
                        if magnitude > self.grad_threshold
                        else []
                    )
                else:
                    order = np.argsort(-magnitudes, kind="stable")
                    vectors = [
                        (
                            float(magnitudes[index]),
                            float(gx_patch[index]),
                            float(gy_patch[index]),
                        )
                        for index in order
                        if magnitudes[index] > self.grad_threshold
                    ][: self.local_orientation_limit]
                mean_gx = float(np.mean([item[1] for item in vectors])) if vectors else 0.0
                mean_gy = float(np.mean([item[2] for item in vectors])) if vectors else 0.0
                mag = max((item[0] for item in vectors), default=0.0)
                kept = bool(vectors)
                LOGGER.event_lazy(
                    "sobel_map.patch.metrics",
                    section=SLIDING_WINDOW,
                    data=lambda: {
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

                for local_magnitude, gx, gy in vectors:
                    angle = float(np.arctan2(gy, gx))
                    if self.angle_in_degrees:
                        angle = float(np.degrees(angle))
                    LOGGER.event_lazy(
                        "sobel_map.patch.angle",
                        section=CYCLIC_COORDS,
                        data=lambda: {
                            "x": bx,
                            "y": by,
                            "angle": angle,
                            "magnitude": local_magnitude,
                        },
                    )
                    out.append((angle, bx, by))

        LOGGER.event(
            "sobel_map.extract",
            section=CYCLIC_COORDS,
            data={
                "label": label,
                "normalized": normalized,
                "patch_size": self.PATCH_SIZE,
                "grid_size": self.GRID_SIZE,
                "patches_total": self.PATCHES_TOTAL,
                "patches_kept": len(out),
            },
        )
        return {label: out}


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

    @classmethod
    def _sobel_vectors_on_4x4(cls, patch4: np.ndarray) -> List[Tuple[float, float, float]]:
        if patch4.shape != (4, 4):
            raise ValueError(f"Ожидаю 4x4 patch, получил {patch4.shape}")
        vectors: List[Tuple[float, float, float]] = []
        for iy in range(2):
            for ix in range(2):
                window = patch4[iy:iy + 3, ix:ix + 3]
                gx = float(np.sum(window * cls.KX))
                gy = float(np.sum(window * cls.KY))
                vectors.append((float(np.hypot(gx, gy)), gx, gy))
        return vectors

    def _selected_vectors(self, patch4: np.ndarray) -> List[Tuple[float, float, float]]:
        if self.local_orientation_limit is None:
            gx, gy = self._sobel_mean_vector_on_4x4(patch4)
            magnitude = float(np.hypot(gx, gy))
            return [(magnitude, gx, gy)] if magnitude > self.grad_threshold else []
        vectors = [
            item
            for item in self._sobel_vectors_on_4x4(patch4)
            if item[0] > self.grad_threshold
        ]
        vectors.sort(key=lambda item: item[0], reverse=True)
        return vectors[: self.local_orientation_limit]
