"""Точка входа фреймворка: `Damp` — этапы пайплайна как поля, настройки и версия.

`Damp()` подключает системный слой и даёт по полю на каждый из семи этапов (см. описание пакета `damp_api`).
`Settings` — настройки только для чтения.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path

from ._system import system
from .cache import Cache
from .parts import Session
from .reports import Stage
from .stages import STAGES, Coding, Cycle, Detectors, Embeddings, Figures, Layout, Recognition, Stimuli


class Settings(Mapping):
    """Настройки только для чтения: ключ — имя из `client/config.toml` (то же, что в коде) с изменениями своего `Damp`.

    Это `Mapping`: работают `settings["ИМЯ"]`, `in`, `len`, перебор по ключам. Нужны и выводимые значения,
    которых нет в файле (правила в `system/damp/settings.py`).

    Example:
        >>> damp = Damp(changes={"LADDER_TOP_COVERAGE": 0.8})
        >>> damp.settings["LADDER_TOP_COVERAGE"]
        0.8
        >>> damp.settings.changes
        {'LADDER_TOP_COVERAGE': 0.8}
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def __getitem__(self, name: str) -> object:
        """Значение настройки: из файла или из `changes` своего `Damp`.

        Raises:
            KeyError: такой настройки нет ни в файле, ни среди выводимых значений.
        """
        if name not in system.settings.KEYS and name not in system.settings.DERIVED:
            raise KeyError(name)
        with self._session.active():
            return getattr(system.settings, name)

    def __iter__(self) -> Iterator[str]:
        """Ключи файла настроек и выводимые из них значения."""
        return iter(sorted(system.settings.KEYS | system.settings.DERIVED))

    def __len__(self) -> int:
        """Сколько настроек (ключи файла и выводимые значения)."""
        return sum(1 for _ in self)

    @property
    def path(self) -> Path:
        """Файл настроек, из которого прочитаны значения."""
        return Path(system.settings.CONFIG_PATH)

    @property
    def changes(self) -> dict[str, object]:
        """Что изменено поверх файла настроек."""
        return dict(self._session.changes)


class Damp:
    """DAMP для клиентского программиста: этапы 1–7 пайплайна по статье и её продолжению.

    Args:
        config: файл настроек; `None` — `client/config.toml`. Файл читается один раз на процесс: другой файл
            в том же процессе — `RuntimeError`.
        log_file: писать ли журнал в `logs/damp.jsonl` (`False` — не писать, как в тестах).
        changes: значения поверх файла настроек, имя ключа → значение, например `{"LADDER_TOP_COVERAGE": 0.8}`.

    Raises:
        KeyError: в `changes` есть имя, которого нет в настройках.
        RuntimeError: системный слой уже подключён с другим файлом настроек.

    Attributes:
        settings (Settings): настройки только для чтения.
        stimuli (Stimuli): этап 1, стимулы.
        coding (Coding): этап 2, первичное кодирование (начало цикла).
        layout (Layout): этап 3, раскладка.
        detectors (Detectors): этап 4, детекторы и активации.
        embeddings (Embeddings): этап 5, эмбеддинги.
        cycle (Cycle): этап 6, цикл (конец цикла).
        recognition (Recognition): этап 7, распознавание.
        figures (Figures): рисунки README.
        cache (Cache): кеш шагов — сводка и очистка; система берёт готовые шаги из него сама.

    Example:
        >>> damp = Damp()                                        # настройки — client/config.toml
        >>> digits = damp.stimuli.digits(train=100, test=500)    # этап 1
        >>> sticks = damp.coding.sticks(digits)                  # этап 2: элементы уровня 0
        >>> level = damp.cycle.step(sticks)                      # этапы 6 → 2 → 3 → 4 → 5 одного уровня
        >>> damp.recognition.vote(level.details)                 # этап 7
        >>> report = damp.cycle.run(train=100, test=500)         # готовый прогон всего цикла
    """

    def __init__(self, config: str | Path | None = None, *, log_file: bool = True,
                 changes: Mapping[str, object] | None = None) -> None:
        system.connect(config, log_file=log_file)
        self._session = Session(changes)
        self.settings = Settings(self._session)
        self.stimuli = Stimuli(self._session)
        self.coding = Coding(self._session)
        self.layout = Layout(self._session)
        self.detectors = Detectors(self._session)
        self.embeddings = Embeddings(self._session)
        self.cycle = Cycle(
            self._session,
            {"coding": self.coding, "layout": self.layout, "detectors": self.detectors, "embeddings": self.embeddings},
        )
        self.recognition = Recognition(self._session)
        self.figures = Figures(self._session)
        self.cache = Cache(self._session)

    @classmethod
    def from_snapshot(cls, path: str | Path, *, log_file: bool = True, changes: Mapping[str, object] | None = None) -> "Damp":
        """`Damp` с настройками счёта из снимка обученного цикла (`TrainedLadder.save`); настройки машины и страницы —
        из своего файла, `changes` — поверх всего.

        Args:
            path: папка снимка.
            log_file: писать журнал в файл.
            changes: свои значения поверх настроек снимка (например, порт дашборда).

        Returns:
            `Damp`, чьи этапы считают так же, как цикл снимка.

        Example:
            >>> damp = Damp.from_snapshot("data/dashboard/trained-1553")
            >>> ladder = damp.cycle.load("data/dashboard/trained-1553")
        """
        system.connect(None, log_file=log_file)
        values, _ = system.snapshot_settings(str(path))
        return cls(log_file=log_file, changes={**values, **dict(changes or {})})

    def with_settings(self, **changes: object) -> "Damp":
        """Новый `Damp` с другими значениями настроек поверх нынешних; сам `Damp` не меняется.

        Args:
            **changes: имя настройки → новое значение.

        Returns:
            `Damp` с теми же этапами и объединёнными изменениями.

        Raises:
            KeyError: имени нет в настройках.

        Example:
            >>> wide = Damp().with_settings(LADDER_TOP_COVERAGE=0.8)
        """
        return Damp(changes={**self._session.changes, **changes})

    @property
    def version(self) -> str:
        """Версия проекта, например `"0.6.0"` (`system/damp/__init__.py`)."""
        return system.version

    @property
    def stages(self) -> tuple[Stage, ...]:
        """Этапы пайплайна по порядку; у этапа 2 — начало цикла, у этапа 6 — конец."""
        return STAGES
