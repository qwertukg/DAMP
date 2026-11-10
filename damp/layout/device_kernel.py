"""Ядра устройства раскладки (§5.12) на jiterator: кеш скомпилированных ядер и вызов с адресами массивов.

jiterator (torch.cuda) компилирует поэлементное ядро из текста функции. Расчётам раскладки нужен доступ
к массивам по индексу (строки сходства, ячейки точек), поэтому массивы устройства передаются ядру адресами
(скаляры long long), а само ядро остаётся поэлементным по своим входам — номерам строк, отрезков, пар.
"""
from __future__ import annotations

import os
import time
from typing import Any

from damp.article_refs import OUTSIDE_ARTICLE
from damp.logging import LOGGER


class KernelDiskCache:
    """Кеш скомпилированных ядер jiterator на диске (инфраструктура, вне статьи).

    torch компилирует ядро при первом вызове (≈ 0.3–0.6 с) и кладёт результат в каталог кеша, чтобы
    следующий запуск программы взял готовое. Файл пишется под временным именем «…_tmp_<pid>» и затем
    переименовывается; на Windows torch переименовывает ещё открытый файл, переименование не удаётся,
    и каждый запуск компилирует все ядра заново. publish довершает переименование для ядер раскладки
    (имена начинаются с PREFIX) и убирает устаревшие временные файлы прошлых запусков.
    """

    PREFIX = "damp_"
    _MARK = "_tmp_"
    # Временные файлы других процессов моложе этого возраста не трогаются: процесс может ещё работать.
    STALE_SECONDS = 600.0

    @classmethod
    def directory(cls) -> str | None:
        """Каталог кеша ядер torch — те же правила, что в torch: переменная окружения или TEMP."""
        if os.environ.get("USE_PYTORCH_KERNEL_CACHE") == "0":
            return None
        explicit = os.environ.get("PYTORCH_KERNEL_CACHE_PATH")
        if explicit:
            return explicit
        temp = os.environ.get("TEMP")
        return None if not temp else os.path.join(temp, "torch", "kernels")

    @classmethod
    def publish(cls) -> int:
        """Переименовать временные файлы ядер этого процесса в постоянные; вернуть число опубликованных."""
        if os.name != "nt":
            return 0
        directory = cls.directory()
        if directory is None or not os.path.isdir(directory):
            return 0
        own = f"{cls._MARK}{os.getpid()}"
        published = 0
        removed = 0
        now = time.time()
        try:
            names = os.listdir(directory)
        except OSError:
            return 0
        for name in names:
            if not name.startswith(cls.PREFIX) or cls._MARK not in name:
                continue
            source = os.path.join(directory, name)
            target = os.path.join(directory, name[: name.rindex(cls._MARK)])
            try:
                if name.endswith(own):
                    if os.path.exists(target):
                        os.remove(source)
                        removed += 1
                    else:
                        os.replace(source, target)
                        published += 1
                elif os.path.exists(target) and now - os.path.getmtime(source) > cls.STALE_SECONDS:
                    os.remove(source)
                    removed += 1
            except OSError:
                continue
        if published or removed:
            LOGGER.event(
                "layout.kernel.cache",
                section=OUTSIDE_ARTICLE,
                data={
                    "directory": directory,
                    "published": published,
                    "removed": removed,
                    "note": "скомпилированные ядра сохранены для следующих запусков",
                },
            )
        return published


class DeviceKernel:
    """Ядро jiterator; после первого вызова с новым набором типов входов публикует файл кеша на диске.

    Ядро получает массивы устройства адресами (скаляры long long): поэлементное ядро jiterator само умеет
    читать только свои входы, а строкам сходства нужен доступ по индексу — номер элемента строки, номер
    похожей точки, её ячейка. Тензоры, чьи адреса переданы, должны жить до конца расчёта (до выгрузки
    результата на хост).
    """

    def __init__(self, function: Any) -> None:
        self._function = function
        self._signatures: set[tuple] = set()

    def __call__(self, *tensors: Any, **scalars: Any) -> Any:
        result = self._function(*tensors, **scalars)
        signature = tuple(tensor.dtype for tensor in tensors)
        if signature not in self._signatures:
            self._signatures.add(signature)
            KernelDiskCache.publish()
        return result


class StagedTable:
    """Таблица int32 шага в закреплённой памяти хоста и её место на устройстве (инфраструктура, вне статьи).

    Таблицу пар шага хост заполняет прямо в закреплённой памяти (host), на устройство она уходит одной
    асинхронной копией без промежуточного буфера (upload). Пока копия не завершена, память хоста менять
    нельзя: следующий host дожидается её. Ядра, запущенные после upload, читают копию на устройстве;
    порядок операций в потоке устройства гарантирует, что копия выполнена до них.
    """

    def __init__(self, torch_mod: Any, device: Any, *, capacity: int = 1 << 16) -> None:
        self._torch = torch_mod
        self._device = device
        self._capacity = 0
        self._host: Any = None
        self._host_array: Any = None
        self._device_table: Any = None
        self._pending: Any = None
        self._view: Any = None
        self._reserve(int(capacity))

    def _reserve(self, capacity: int) -> None:
        torch = self._torch
        self._settle()
        self._host = torch.empty(capacity, dtype=torch.int32, pin_memory=True)
        self._host_array = self._host.numpy()
        self._device_table = torch.empty(capacity, dtype=torch.int32, device=self._device)
        self._capacity = capacity

    def _settle(self) -> None:
        if self._pending is not None:
            self._pending.synchronize()
            self._pending = None

    @property
    def view(self) -> Any:
        """Массив, который вернул последний host (для проверки, что таблица уже лежит в закреплённой памяти)."""
        return self._view

    def host(self, size: int) -> Any:
        """Массив numpy int32 длины size в закреплённой памяти: заполнить и вызвать upload(size)."""
        if size > self._capacity:
            self._reserve(max(size, 2 * self._capacity))
        else:
            self._settle()
        self._view = self._host_array[:size]
        return self._view

    def upload(self, size: int) -> Any:
        """Копия первых size элементов на устройство (асинхронно); возвращает тензор устройства."""
        torch = self._torch
        target = self._device_table[:size]
        target.copy_(self._host[:size], non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._pending = event
        return target
