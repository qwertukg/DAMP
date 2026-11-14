"""Кеш шагов: что лежит на диске и очистка (вне статьи).

Система сама берёт готовый результат шага с диска, если вход шага, значения настроек, которые читает его код, текст
кода и окружение те же. Каждый шаг — запись сама по себе: штрихи картинок, окна, коды окон, карта памяти, детекторы,
срабатывания, детали, отклики и рисунки карт каждого уровня. Из кеша — то же, что посчитано заново, бит в бит.
Настройки — раздел `[cache]` в `client/config.toml`: `CACHE_ENABLED`, `CACHE_DIR`, `CACHE_MAX_BYTES`,
`CACHE_VERIFY_SHARE` (доля попаданий, которые пересчитываются и сравниваются с кешем).
"""
from __future__ import annotations

from pathlib import Path

from ._system import system
from .parts import Session
from .reports import CacheStep


class Cache:
    """Кеш шагов системы: сводка по шагам и очистка.

    Example:
        >>> damp = Damp()
        >>> for row in damp.cache.summary():
        ...     print(row.step, row.entries, row.bytes)
        >>> damp.cache.clear("layout.memory")     # пересчитать карты памяти при следующем прогоне
        >>> Damp(changes={"CACHE_VERIFY_SHARE": 1.0}).cycle.run(train=100, test=500)   # сверить весь кеш
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    @property
    def path(self) -> Path:
        """Папка кеша (`CACHE_DIR`)."""
        with self._session.active():
            return Path(system.step_cache.root)

    @property
    def enabled(self) -> bool:
        """Включён ли кеш (`CACHE_ENABLED` и переменная окружения `DAMP_CACHE` не равна `off`)."""
        with self._session.active():
            return bool(system.step_cache.enabled)

    def summary(self) -> tuple[CacheStep, ...]:
        """Записи по шагам: сколько, сколько места, давность последнего использования.

        Returns:
            Шаги по алфавиту; пустой кортеж, если кеш пуст.
        """
        with self._session.active():
            rows = system.step_cache.summary()
        return tuple(
            CacheStep(
                step=str(row["step"]), entries=int(row["entries"]), bytes=int(row["bytes"]),
                last_used_seconds=float(row["last_used_seconds"] or 0.0),
            )
            for row in rows
        )

    def size(self) -> int:
        """Место на диске под кешем, байт."""
        with self._session.active():
            return int(system.step_cache.size())

    def clear(self, step: str | None = None) -> int:
        """Удалить записи шага `step` (`None` — весь кеш вместе со штрихами картинок).

        Args:
            step: имя шага из `summary()`, например `"activation.detectors"`.

        Returns:
            Сколько записей удалено.
        """
        with self._session.active():
            return int(system.step_cache.clear(step))
