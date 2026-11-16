"""Этап 1: построитель извлечения углопозиций MNIST и подпись способа стимулов для кеша (вне статьи)."""

from __future__ import annotations
from damp import settings
from damp.stimuli.MnistSobelAngleMap import MnistSobelAngleMap, UpscaledSobel


def _build_extractor(
    patch_size: int | None = None,
    *,
    contour_only: bool | None = None,
    upscale: bool | None = None,
) -> MnistSobelAngleMap:
    """Извлечение углопозиций; None — значение из настроек в момент вызова (MNIST_PATCH_SIZE, MNIST_CONTOUR_ONLY,
    MNIST_SOBEL_UPSCALE_ENABLED). Дробное место штриха внутри участка — MNIST_SUBCELL_POSITION (только по контуру)."""
    patch_size = settings.MNIST_PATCH_SIZE if patch_size is None else patch_size
    contour_only = settings.MNIST_CONTOUR_ONLY if contour_only is None else contour_only
    upscale = settings.MNIST_SOBEL_UPSCALE_ENABLED if upscale is None else upscale
    upscaled = None
    if upscale:
        upscaled = UpscaledSobel(
            scale=settings.MNIST_SOBEL_UPSCALE,
            contrast=settings.MNIST_SOBEL_CONTRAST,
            entity_path=settings.MNIST_SOBEL_ENTITY_PATH,
        )
    return MnistSobelAngleMap(
        angle_in_degrees=settings.MNIST_ANGLE_IN_DEGREES,
        grad_threshold=settings.MNIST_GRAD_THRESHOLD,
        patch_size=patch_size,
        contour_only=contour_only,
        contour_level=settings.MNIST_CONTOUR_LEVEL,
        two_edge_coherence=settings.MNIST_CONTOUR_TWO_EDGE_COHERENCE,
        upscaled=upscaled,
        subcell_position=bool(settings.MNIST_SUBCELL_POSITION) and contour_only,
    )


def _stimulus_metadata(extractor: MnistSobelAngleMap) -> dict[str, object]:
    """Параметры получения стимулов, без совпадения которых кеш раскладки не подходит."""
    return {settings.PATCH_SIZE_KEY: extractor.patch_size, settings.STIMULUS_MODE_KEY: extractor.stimulus_mode}
