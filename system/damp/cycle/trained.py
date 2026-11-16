"""Этап 6: обученный цикл и путь одной новой картинки через все уровни (§6.5, §6.6, §9.3; продолжение статьи, П.1–П.5).

Цикл учится как обычно (MnistLadderRunner с keep_levels): по обучающим картинкам строятся окна, карты памяти и
детекторы уровней, а линейный читатель (§9.3) учится на пиках их срабатываний. Новая картинка — рисунок дашборда —
идёт теми же классами, что картинка прогона: углопозиции (§4.4), окна (П.2), коды окон (П.4–П.7), срабатывания
детекторов (§6.5), детали (П.1) — уровень за уровнем. Всё, что подобрано по обучающим (мера и пороги окон, длины
кодов, нормировка), берётся из обученных уровней, а не меряется заново: поэтому у картинки MNIST здесь те же окна,
срабатывания, детали и ответ, что в прогоне (tests/test_trained_ladder.py). Хранение обученных уровней и путь одной
картинки — техника (вне статьи); сами шаги — статьи и продолжения.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from damp import settings
from damp.activation.details import DetailCoder
from damp.article_refs import NEURAL_ANALOGY, STIMULUS_DETECTION
from damp.cycle.keyholes import KeyholeBuilder
from damp.cycle.lateral_inhibition import LateralInhibition
from damp.cycle.step import TrainedLevel
from damp.decode.level_recognition import LevelRecognition
from damp.decode.linear_reader import LinearReader
from damp.encoding.elements import LadderElements, StickCoder
from damp.encoding.keyhole_code import KeyholeCoder
from damp.logging import LOGGER
from damp.stimuli.digits import DigitAtoms, DigitGroups, MnistDigits

SIDE = 28  # сторона картинки MNIST, пикселей


@dataclass(frozen=True)
class SightLevel:
    """Что сделал один уровень с картинкой: окна (кусочки), зажжённые точки карты, сработавшие детекторы, детали.

    Номера окон — внутри картинки (0…K−1); элемент вхождения — углопозиция картинки на уровне 1, окно прошлого уровня
    на следующих (деталь окна — элемент следующего уровня, П.1). Точки карты — в порядке раскладки, детекторы — по
    слоям (слой 0 — самый низкий порог λ_d, самые широкие поля, §7.1.10).
    """

    number: int
    keyhole_centre: "np.ndarray"  # (K, 2) центр окна (y, x), узлы сетки картинки
    keyhole_extent: "np.ndarray"  # (K,) сколько клеток картинки видит окно
    entry_keyhole: "np.ndarray"  # (E,) окно вхождения
    entry_element: "np.ndarray"  # (E,) его элемент
    lit_keyhole: "np.ndarray"  # (L,) окно, зажёгшее точку карты (§6.1)
    lit_point: "np.ndarray"  # (L,) точка карты
    lit_value: "np.ndarray"  # (L,) её активация a_j
    fired_keyhole: "np.ndarray"  # (R,) окно, на котором сработал детектор (§6.5)
    fired_detector: "np.ndarray"  # (R,) детектор
    fired_level: "np.ndarray"  # (R,) уровень активации E(d)
    detail_codes: "np.ndarray"  # (K, ⌈длина/8⌉) код детали каждого окна (§6.5, §7.1.12)
    detail_length: int
    strength: "np.ndarray"  # (K,) уровень активации детали — сильнейший из сработавших детекторов (П.1)
    peak: "np.ndarray"  # (D,) сильнейший уровень каждого детектора по окнам картинки, единичной длины (§9.3)


@dataclass(frozen=True)
class Sight:
    """Путь одной картинки: углопозиции, уровни и ответ линейного читателя (§9.3)."""

    stick_angle: "np.ndarray"  # (n,) угол углопозиции, градусы (§4.4)
    stick_y: "np.ndarray"  # (n,) место на сетке картинки, в клетках (дробное при MNIST_SUBCELL_POSITION)
    stick_x: "np.ndarray"
    stick_strength: "np.ndarray"  # (n,) сила штриха, доля сильнейшего штриха картинки
    levels: tuple[SightLevel, ...]
    probabilities: "np.ndarray"  # (классов,) вероятности классов; нули — ответа нет
    answer: int  # самый вероятный класс; -1 — ни один детектор не сработал
    seconds: float


class TrainedLadder:
    """Обученный цикл (§6.6) и его линейный читатель (§9.3): по ним идёт любая новая картинка 28×28.

    groups — картинки обучения (память строится по ним, ranges["train"] — с видами §7.1.4); levels — обученные уровни
    (TrainedLevel); peaks — пики срабатываний всех картинок прогона по уровням (строки ranges["train"] — обучение
    читателя); decay — штраф L2 читателя.
    """

    def __init__(self, groups: DigitGroups, levels: list[TrainedLevel], peaks: list["np.ndarray"], *, decay: float) -> None:
        if not levels:
            raise ValueError("нет ни одного обученного уровня")
        if len(peaks) != len(levels):
            raise ValueError("пики нужны по каждому обученному уровню")
        self.groups = groups
        self.levels = tuple(levels)
        first, last = groups.ranges["train"]
        features = np.concatenate(peaks, axis=1)
        self.reader = LinearReader(features[first:last], groups.labels[first:last], float(decay))
        self.decay = float(decay)
        self._digits = MnistDigits()
        self._sticks = StickCoder()
        self._coder = KeyholeCoder()

    @classmethod
    def train(cls, count: int, *, map_points: int | None, decay: float, observer=None) -> "TrainedLadder":  # noqa: ANN001
        """Цикл на count обучающих цифрах (с видами LADDER_TRAIN_VIEWS), без проверочных; уровни сохраняются.

        Шаги цикла — из кеша, если такой прогон уже был. Рисунки прогона не пишутся (LADDER_SAVE_MAPS выключен на время
        обучения), рабочие файлы — под DASHBOARD_FILE_PREFIX. observer — наблюдатель прогона (LadderObserver).
        """
        from damp.cycle.ladder import MnistLadderRunner

        started = time.perf_counter()
        prefix = str(settings.DASHBOARD_FILE_PREFIX)
        Path(prefix).parent.mkdir(parents=True, exist_ok=True)
        with settings.overridden({"LADDER_SAVE_MAPS": False}):
            runner = MnistLadderRunner(
                count=int(count), evaluate_limit=0, map_max_points=map_points, shifts=(), file_prefix=prefix,
                observer=observer, keep_levels=True,
            )
            runner.run()
        ladder = cls(runner.groups, runner.levels, runner.level_peaks, decay=decay)
        LOGGER.event(
            "dashboard.ladder",
            section=NEURAL_ANALOGY,
            data={
                "train_digits": int(count),
                "train_pictures": int(runner.groups.train_pictures),
                "levels": len(ladder.levels),
                "map_points": [int(level.memory.points) for level in ladder.levels],
                "detectors": [int(level.detectors.built) for level in ladder.levels],
                "decay": float(decay),
                "reader_iterations": int(ladder.reader.iterations),
                "seconds": time.perf_counter() - started,
                "note": "обученный цикл §6.6 для новых картинок: окна, карты памяти и детекторы уровней по обучающим "
                "картинкам с видами (§7.1.4), линейный читатель (§9.3) на их пиках; новая картинка идёт теми же шагами",
            },
        )
        return ladder

    # ------------------------------------------------------------------ новая картинка
    def see(self, image: "np.ndarray", *, log: bool = False) -> Sight:
        """Картинка 28×28 (uint8 0…255 либо float в [0, 1], белая цифра на чёрном) через все обученные уровни.

        Каждый уровень: окна на элементах (П.2) → коды окон (П.4–П.7) → срабатывания детекторов карты (§6.5) →
        детали — элементы следующего уровня (П.1); затем ответ линейного читателя по пикам уровней (§9.3). Шаги
        считаются заново и в кеш не пишутся; журнал шагов молчит. log — записать итог событием dashboard.see.
        """
        started = time.perf_counter()
        pixels = self._pixels(image)
        levels: list[SightLevel] = []
        peaks: list["np.ndarray"] = []
        from damp.cache import CACHE

        with CACHE.bypassed(), LOGGER.quiet():
            values = self._digits.sticks_of(pixels)
            atoms = DigitAtoms.from_rows(
                [np.column_stack([np.zeros(values.shape[0], dtype=np.float64), values])],
                [-1], [np.clip(np.rint(pixels * 255.0), 0, 255).astype(np.uint8)], [-1],
            )
            elements = self._sticks.encode(atoms) if values.shape[0] else None
            for trained in self.levels:
                if elements is None or elements.count == 0:
                    break
                if trained.number > 1 and settings.LADDER_INHIBITION_RADIUS is not None:
                    # Торможение соседей — как в прогоне (MnistLadderRunner._ladder): оно внутри своей картинки.
                    elements = elements.only(
                        LateralInhibition(radius_share=float(settings.LADDER_INHIBITION_RADIUS), name=f"details{trained.number - 1}")
                        .survivors(elements.picture, elements.y, elements.x, elements.power(), elements.reach())
                    )
                    if elements.count == 0:
                        break
                level, elements = self._level(trained, elements)
                if level is None:
                    break
                levels.append(level)
                peaks.append(level.peak)
        # Уровни, до которых картинка не дошла, молчат: их пики нулевые, как у такой картинки в прогоне.
        for trained in self.levels[len(peaks):]:
            peaks.append(np.zeros(int(trained.detectors.built), dtype=np.float32))
        features = np.concatenate(peaks)[None, :]
        probabilities = self.reader.probabilities(features)[0]
        answer = int(self.reader.answers(features)[0])
        sight = Sight(
            stick_angle=np.asarray(atoms.angle, dtype=np.float64),
            stick_y=np.asarray(atoms.y, dtype=np.float64),
            stick_x=np.asarray(atoms.x, dtype=np.float64),
            stick_strength=np.asarray(atoms.strength, dtype=np.float64),
            levels=tuple(levels),
            probabilities=probabilities,
            answer=answer,
            seconds=time.perf_counter() - started,
        )
        if log:
            LOGGER.event(
                "dashboard.see",
                section=STIMULUS_DETECTION,
                data={
                    "sticks": int(sight.stick_angle.shape[0]),
                    "levels": [
                        {"level": level.number, "keyholes": int(level.keyhole_centre.shape[0]),
                         "points_lit": int(np.unique(level.lit_point).shape[0]), "detectors_fired": int(np.unique(level.fired_detector).shape[0])}
                        for level in sight.levels
                    ],
                    "answer": answer,
                    "probability": float(probabilities.max()) if answer >= 0 else 0.0,
                    "seconds": sight.seconds,
                    "note": "рисунок дашборда через обученные уровни: окна (П.2), их коды (П.4–П.7), срабатывания "
                    "детекторов карт (§6.5), детали (П.1) и ответ линейного читателя (§9.3)",
                },
            )
        return sight

    def _level(self, trained: TrainedLevel, elements: LadderElements) -> tuple[SightLevel | None, LadderElements | None]:
        """Проход обученного уровня над элементами одной картинки — те же шаги, что CycleStep.run, без подбора."""
        pieces, extent = KeyholeBuilder.place(elements, number=trained.number, step=trained.step, thresholds=trained.thresholds)
        if pieces.count == 0:
            return None, None
        codes = self._coder.encode(
            elements, pieces, number=trained.number, unit=trained.field.unit, taught=pieces.count,
            content_length=trained.content_length,
        )
        every = np.arange(codes.count, dtype=np.int64)
        detectors = trained.detectors
        firing = detectors.fire(codes.packed, every, every, codes.count)
        if trained.scale is not None:
            firing = firing.scaled(trained.scale)  # нормировка — делителями обучающих окон (вне статьи)
        details, length = DetailCoder(trained.number).encode(
            firing, detectors.layer_index, detectors.built, np.ones(codes.count, dtype=bool), length=trained.detail_length
        )
        strength = firing.strength()
        _, peak = LevelRecognition.responses(pieces.picture, firing.rows, firing.columns, firing.levels, 1, detectors.built)
        lit_keyhole, lit_point, lit_value = detectors.lit(codes.packed)
        level = SightLevel(
            number=trained.number,
            keyhole_centre=np.asarray(pieces.centre, dtype=np.float64),
            keyhole_extent=np.asarray(extent, dtype=np.float64),
            entry_keyhole=np.asarray(pieces.piece, dtype=np.int64),
            entry_element=np.asarray(pieces.element, dtype=np.int64),
            lit_keyhole=lit_keyhole,
            lit_point=lit_point,
            lit_value=lit_value,
            fired_keyhole=np.asarray(firing.rows, dtype=np.int64),
            fired_detector=np.asarray(firing.columns, dtype=np.int64),
            fired_level=np.asarray(firing.levels, dtype=np.float32),
            detail_codes=np.asarray(details, dtype=np.uint8),
            detail_length=int(length),
            strength=np.asarray(strength, dtype=np.float64),
            peak=np.asarray(peak[0], dtype=np.float32),
        )
        return level, DetailCoder.elements(details, length, pieces.picture, pieces.centre, extent, strength)

    @staticmethod
    def _pixels(image: "np.ndarray") -> "np.ndarray":
        """Пикселы float32 в [0, 1], как у MNIST после ToTensor (uint8 / 255 в float32)."""
        image = np.asarray(image)
        if image.shape != (SIDE, SIDE):
            raise ValueError(f"картинка должна быть {SIDE}×{SIDE}, а не {image.shape}")
        if image.dtype == np.uint8:
            return image.astype(np.float32) / np.float32(255.0)
        return np.clip(image.astype(np.float32), 0.0, 1.0)
