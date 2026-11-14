"""Окружение прогона: видеокарта (CUDA, MPS) и частота записей журнала (вне статьи)."""

from __future__ import annotations
import os
from typing import ClassVar

from damp import __version__, settings
from damp.article_refs import OUTSIDE_ARTICLE
from damp.logging import LOGGER, JsonlLog


def configure_logging() -> None:
    """Частота записей и файл журнала JSON Lines с ротацией (LOG_FILE_*); DAMP_LOG_FILE=off — без файла."""
    LOGGER.configure_intervals(settings.LOG_INTERVALS, default_interval=settings.LOG_INTERVAL_DEFAULT)
    wanted = settings.LOG_FILE_ENABLED and os.environ.get("DAMP_LOG_FILE", "on").lower() != "off"
    if wanted and LOGGER.file is None:
        LOGGER.attach_file(
            JsonlLog(
                settings.LOG_FILE_PATH,
                max_bytes=settings.LOG_FILE_MAX_BYTES,
                backups=settings.LOG_FILE_BACKUPS,
                array_limit=settings.LOG_FILE_ARRAY_LIMIT,
            )
        )
        LOGGER.event(
            "logging.file",
            section=OUTSIDE_ARTICLE,
            data={"version": __version__, "path": settings.LOG_FILE_PATH, "run": LOGGER.file.run,
                  "max_bytes": settings.LOG_FILE_MAX_BYTES, "backups": settings.LOG_FILE_BACKUPS},
        )


class GpuEnvironment:
    _applied: ClassVar[bool] = False

    def __init__(self, *, mps_high_watermark: float | None) -> None:
        self._mps_high_watermark = mps_high_watermark

    def apply(self) -> None:
        if GpuEnvironment._applied:
            return
        GpuEnvironment._applied = True
        if self._mps_high_watermark is None:
            return
        target = str(self._mps_high_watermark)
        current = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
        mode = "preserve"
        value = current if current is not None else target
        if current is None:
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = target
            mode = "set"
        LOGGER.event(
            "layout.gpu.mps.watermark",
            section=OUTSIDE_ARTICLE,
            data={
                "mode": mode,
                "value": value,
            },
        )


GPU_ENVIRONMENT = GpuEnvironment(mps_high_watermark=settings.MPS_HIGH_WATERMARK_RATIO)
