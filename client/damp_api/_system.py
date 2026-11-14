"""Внутренний модуль: мост в системный слой — единственное место клиентской части, которое обращается к system/damp.

В документацию API не входит; клиентский код его не импортирует.

Остальные модули фреймворка и UI берут отсюда только то, что им нужно, по понятным именам. Системные объекты
наружу фреймворка не выходят: части фреймворка держат их внутри и отдают простые значения (reports.py, parts.py).
"""
from __future__ import annotations

import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
SYSTEM = ROOT / "system"


class System:
    """Подключение системного слоя: путь к пакету, файл настроек, журнал, видеокарта."""

    def __init__(self) -> None:
        self._connected = False
        self._config: Path | None = None
        self._started = False

    # ------------------------------------------------------------------ подключение
    def connect(self, config: str | Path | None = None, *, log_file: bool = True) -> None:
        """Подключить системный слой с файлом настроек config (None — client/config.toml).

        Файл настроек читается один раз при первом подключении; другой файл в том же процессе — ошибка.
        Отдельные значения меняются без нового файла — через Damp(changes=…).
        """
        wanted = None if config is None else Path(config).resolve()
        if self._connected:
            if wanted is not None and wanted != self._config:
                raise RuntimeError(f"системный слой уже подключён с настройками {self._config}; другой файл — в новом процессе")
            return
        if wanted is not None:
            os.environ["DAMP_CONFIG"] = str(wanted)
        if not log_file:
            os.environ["DAMP_LOG_FILE"] = "off"
        os.environ.setdefault("RERUN", "off")
        if str(SYSTEM) not in sys.path:
            sys.path.insert(0, str(SYSTEM))
        from damp import settings

        self._config = Path(settings.CONFIG_PATH).resolve()
        self._connected = True

    def start(self) -> None:
        """Перед работой: частота записей, файл журнала и видеокарта (один раз на процесс)."""
        if not self._started:
            from damp.environment import GPU_ENVIRONMENT, configure_logging

            configure_logging()
            GPU_ENVIRONMENT.apply()
            self._started = True

    @contextmanager
    def session(self, changes: Mapping[str, object]) -> Iterator[None]:
        """Работа системы с настройками файла, поверх которых — changes (имя ключа → значение)."""
        self.start()
        with self.settings.overridden(changes):
            yield

    def check_changes(self, changes: Mapping[str, object]) -> None:
        """Все имена changes — ключи файла настроек; иначе KeyError с подсказкой."""
        self.settings.check_changes(changes)

    # ------------------------------------------------------------------ общее
    @property
    def settings(self) -> ModuleType:
        from damp import settings

        return settings

    @property
    def version(self) -> str:
        from damp import __version__

        return __version__

    # ------------------------------------------------------------------ этап 1: стимулы
    def extractor(self, patch_size: int | None = None, *, contour_only: bool | None = None, upscale: bool | None = None):  # noqa: ANN201
        from damp.stimuli.mnist import _build_extractor

        return _build_extractor(patch_size, contour_only=contour_only, upscale=upscale)

    def digits(self, *, train: int, test: int, split: str, shifts, shift_pictures: int):  # noqa: ANN001, ANN201
        """Цифры MNIST с углопозициями: обучающие, проверочные и их сдвиги одним набором (DigitGroups)."""
        from damp.stimuli.digits import DigitGroups, MnistDigits

        return DigitGroups.load(
            MnistDigits(), train=train, test=test, split=split, shifts=tuple(shifts), shift_pictures=shift_pictures
        )

    def sketch_tools(self) -> ModuleType:
        """Рисунки углопозиций: AnglePositionSketch, AnglePositionSheet, SheetCell, MnistDigitSampler."""
        from damp.stimuli import angle_position_sketch

        return angle_position_sketch

    @property
    def picture_side(self) -> int:
        from damp.stimuli.MnistSobelAngleMap import MnistSobelAngleMap

        return int(MnistSobelAngleMap.EXPECTED_SIDE)

    # ------------------------------------------------------------------ этап 2: кодирование
    def encoder(self):  # noqa: ANN201
        """Первичный кодер (угол, X, Y) — общий код (§2.2.3, §4.2.1, §4.4)."""
        from damp.encoding.mnist_codes import _build_encoder

        return _build_encoder()

    def detector_view(self, encoder, **options):  # noqa: ANN001, ANN201
        """Картинка детекторов кодера в стиле рисунков проекта (§4.1–§4.4)."""
        from damp.encoding.detector_view import DetectorLayerView
        from damp.figures.style import figure_style

        return DetectorLayerView(encoder, style=figure_style(), **options)

    def stick_coder(self):  # noqa: ANN201
        """Коды направлений углопозиций — элементы уровня 0 цикла (§4.4)."""
        from damp.encoding.elements import StickCoder

        return StickCoder()

    # ------------------------------------------------------------------ этапы 3–5: проход без цикла
    def first_level(self, **options):  # noqa: ANN201
        """Раскладка всех углопозиций, детекторы и база эмбеддингов (команда layout)."""
        from damp.activation.first_level import MnistLayoutRunner

        return MnistLayoutRunner(**options)

    def pinwheel(self):  # noqa: ANN201
        from damp.layout.pinwheel import PinwheelExperiment

        return PinwheelExperiment()

    # ------------------------------------------------------------------ этап 6: цикл
    def cycle_step(self, groups, *, map_points: int | None):  # noqa: ANN001, ANN201
        """Проход уровня по тем же правилам, что у команды ladder (§6.6): окна внимания (П.2, П.3), их коды (П.4–П.7),
        память (§5), детекторы и срабатывания (§6.4, §6.5), детали (§6.5, П.1) — по шагу на кубик фреймворка."""
        from damp.cycle.step import CycleStep

        return CycleStep(groups, map_max_points=map_points)

    @property
    def level_failed(self) -> type[Exception]:
        """Ошибка «уровень не сложился»: окон нет, коды слились либо над картой нет детекторов."""
        from damp.cycle.step import LadderLevelDegenerate

        return LadderLevelDegenerate

    def cycle(self, **options):  # noqa: ANN201
        """Цикл уровней целиком (§6.2, §6.6; П.1–П.9)."""
        from damp.cycle.ladder import MnistLadderRunner

        return MnistLadderRunner(**options)

    # ------------------------------------------------------------------ этап 7: распознавание
    def level_recognition(self, labels, train: tuple[int, int]):  # noqa: ANN001, ANN201
        """Узнавание по уровню цикла: отклик картинки и голосование деталей (§2.2.5)."""
        from damp.decode.level_recognition import LevelRecognition

        return LevelRecognition(labels, train)

    def decoder(self, **options):  # noqa: ANN201
        """Распознавание по эмбеддингам первого уровня: нечёткий поиск (§2.2.5)."""
        from damp.decode.first_level import MnistDecodeRunner

        return MnistDecodeRunner(**options)

    # ------------------------------------------------------------------ рисунки
    def readme_figures(self):  # noqa: ANN201
        from damp.figures.readme import ReadmeFigures

        return ReadmeFigures()

    def rerun_screenshots(self):  # noqa: ANN201
        """Скриншоты вкладок цикла в Rerun для README."""
        from damp.figures.rerun_shots import RerunScreenshots

        return RerunScreenshots()

    # ------------------------------------------------------------------ кеш шагов
    @property
    def step_cache(self):  # noqa: ANN201
        """Кеш шагов системы (system/damp/cache.py)."""
        from damp.cache import CACHE

        return CACHE


system = System()
