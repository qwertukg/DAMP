"""Этап 6. Цикл уровней — команда ladder (§6.2, §6.6; продолжение статьи П.1–П.9).

Цикл собран из тех же частей, что клиентский фреймворк damp_api: цифры (этап 1, stimuli/digits.py), коды
направлений углопозиций (этап 2, encoding/elements.py), проход уровня CycleStep (этапы 2–5, cycle/step.py) и узнавание
по уровню (этап 7, decode/level_recognition.py). Здесь — порядок уровней, остановка, мерки и рисунки прогона.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from typing import Sequence

import numpy as np

from damp import settings
from damp.article_refs import FUZZY_SEARCH, NEURAL_ANALOGY, OUTSIDE_ARTICLE
from damp.cycle.ladder_trace import LEVEL_PARTS, LadderLevelTrace, LadderObserver, LadderObservers, LadderRunTrace, LadderTopTrace
from damp.cycle.step import CycleStep, LadderLevelDegenerate, LevelPass
from damp.cycle.keyholes import LadderField
from damp.decode.level_recognition import LevelRecognition
from damp.encoding.elements import LadderElements, StickCoder, packed_ones
from damp.environment import GPU_ENVIRONMENT, configure_logging
from damp.layout.code_dictionary import CodeDictionary
from damp.layout.damp_layout import Layout
from damp.layout.glyph_map import LayoutGlyphMap
from damp.layout.level_memory import LevelMemory
from damp.logging import LOGGER
from damp.stimuli.digits import DigitAtoms, DigitGroups, ExampleDigit, MnistDigits

_FROM_SETTINGS = object()  # значение по умолчанию берётся из настроек в момент вызова


@dataclass(frozen=True)
class LadderLevelReport:
    """Итог одного уровня лестницы: поля зрения, карта, детали и мерки «собрались ли классы»."""

    number: int
    field_height: int  # поле зрения на картинке, клеток
    field_width: int
    field_radius: float  # средний радиус поля зрения на картинке, клеток
    fields_per_picture: float
    picture_seen: float  # какую долю углопозиций своей картинки видит поле в среднем
    pictures_without_fields: float  # доля обучающих картинок, у которых на уровне нет ни одной детали
    elements_per_field: float
    fields_with_5_to_7: float  # доля полей, в которых от 5 до 7 элементов
    map_points: int
    map_code_length: int  # длина кода точки карты: код окна либо код взгляда
    map_code_bits: float  # среднее число единиц в коде точки карты
    detectors: int
    detail_code_length: int
    detail_bits: float
    distinct_details: int
    energy_mean: float  # качество раскладки Ē карты уровня (§5.8): средняя нормированная энергия точек
    code_resolution: float  # окон обучающих картинок на один различный код детали (§5.13.2.5, «resolution per code»)
    code_largest: int  # самый большой набор окон обучающих картинок с одним кодом детали (§5.13.2.5, «largest cluster size»)
    detail_empty: float  # доля обучающих окон с пустой деталью: ни один детектор не сработал
    detail_full: float  # доля обучающих окон, у которых сработавших детекторов не меньше σ (код упёрся в предел, §7.1.8.1)
    same_label: float  # доля пар соседних точек карты одного класса
    same_label_random: float
    alone: dict[int, float] | None  # узнавание по отклику картинки на этом уровне (число срабатываний детекторов)
    peak: dict[int, float] | None  # то же по сильнейшему отклику каждого детектора
    glued: dict[int, float] | None  # узнавание по склейке откликов картинки на уровнях 1…number
    shifted: dict[tuple[int, int], float]  # узнавание сдвинутых картинок по отклику этого уровня
    unshifted: float | None  # узнавание тех же картинок без сдвига
    same_details: dict[tuple[int, int], float]  # доля общих кодов деталей у картинки и её сдвига
    same_response: dict[tuple[int, int], float]  # сходство откликов картинки и её сдвига (косинус)


class MnistLadderRunner:
    """Лестница уровней в локальных координатах на MNIST (§6.2, §6.6 статьи и её продолжение, П.1–П.9).

    Уровень 0 — углопозиции цифр. Шаг вверх повторяется, пока поле зрения не увидит картинку целиком:

    1. На каждом элементе уровня стоит окно внимания миниколонки (П.2). Радиус окна находится по кластеру похожих
       соседей элемента (§6.4.1–§6.4.3), в мере «что видит одна деталь уровня»: окно на карте равно полю своей
       детали, поэтому поле на картинке растёт вдвое с лишним за уровень (П.3; так же растут рецептивные поля от
       V1 к V2 и V4).
    2. Содержимое окна записывается в его координатах; абсолютного места на картинке в коде нет ни на одном
       уровне (§7.1.4.1). Всё окно — один код: признак связан с местом детекторами позиции (бит места через OR,
       маска на код признака, PositionBindingEncoder, П.4–П.7).
    3. Различные коды обучающих картинок раскладываются тем же Layout (§5) — карта уровня, общая память всех
       миниколонок; над картой строится иерархия детекторов статьи (§6.4).
    4. Картинки подаются снова: каждое окно сверяется с картой (§6.5), сработавшие детекторы дают эмбеддинг
       окна — код детали следующего уровня (§6.2); её место — центр окна, её уровень активации — самый сильный
       из сработавших детекторов (П.1).

    После каждого уровня меряется, собрались ли классы: согласие классов соседних точек карты и узнавание по
    отклику картинки на этом уровне (вне статьи). Поиск цифры идёт снизу вверх, тем же путём, что обучение.
    """

    def __init__(
        self,
        *,
        count: int,
        evaluate_limit: int | None = None,
        split: str = "test",
        map_max_points: "int | None | object" = _FROM_SETTINGS,
        shifts: Sequence[tuple[int, int]] | None = None,
        file_prefix: str | None = None,
        observer: LadderObserver | None = None,
        train_start: int = 0,
        validation: int = 0,
        pixel_shift: tuple[int, int] | None = None,
        pixel_pictures: int = 0,
        train_numbers: Sequence[int] | None = None,
        validation_numbers: Sequence[int] | None = None,
        test_numbers: Sequence[int] | None = None,
        keep_levels: bool = False,
    ) -> None:
        evaluate_limit = settings.LADDER_EVALUATE_LIMIT if evaluate_limit is None else evaluate_limit
        if count <= 0:
            raise ValueError("count must be positive")
        if evaluate_limit < 0:
            raise ValueError("evaluate_limit must be non-negative")
        # Явные номера MNIST групп (стенд bench: случайная выборка из пулов); число картинок группы — их число.
        self._numbers = {
            "train": None if train_numbers is None else tuple(int(n) for n in train_numbers),
            "validation": None if validation_numbers is None else tuple(int(n) for n in validation_numbers),
            "test": None if test_numbers is None else tuple(int(n) for n in test_numbers),
        }
        if self._numbers["train"] is not None and len(self._numbers["train"]) != int(count):
            raise ValueError("count must equal the number of train_numbers")
        if self._numbers["validation"] is not None:
            validation = len(self._numbers["validation"])
        if self._numbers["test"] is not None:
            evaluate_limit = len(self._numbers["test"])
        configure_logging()
        GPU_ENVIRONMENT.apply()
        self.count = int(count)
        self._evaluate_limit = int(evaluate_limit)
        self._split = split
        # Протокол данных (вне статьи): обучающие с train_start, отложенные validation для выбора настроек, сдвиг
        # отложенных на пиксели для мерки устойчивости к деформации.
        self._train_start = int(train_start)
        self._validation = int(validation)
        self._pixel_shift = None if pixel_shift is None else (int(pixel_shift[0]), int(pixel_shift[1]))
        self._pixel_pictures = int(pixel_pictures)
        # Префикс файлов прогона: карты, рисунки окон и поля зрения пишутся под ним.
        self.FILE_PREFIX = str(settings.LADDER_FILE_PREFIX if file_prefix is None else file_prefix)
        limit = settings.LADDER_MAP_MAX_POINTS if map_max_points is _FROM_SETTINGS else map_max_points
        self._map_max_points = None if limit is None else int(limit)
        self._shifts = tuple((int(dy), int(dx)) for dy, dx in (settings.LADDER_SHIFTS if shifts is None else shifts))
        # Наблюдатели получают итоги каждого прохода цикла (рисунки README, окно Rerun); на счёт они не влияют.
        # С включённым Rerun (RERUN=on) цикл показывается вкладками этапов 1–7 (figures/rerun_cycle.py).
        watchers = [] if observer is None else [observer]
        self._rerun = None
        if LOGGER.visuals_enabled():
            from damp.figures.rerun_cycle import CycleRerunView

            self._rerun = CycleRerunView(glyphs=self._glyph_file)
            watchers.append(self._rerun)
        self._observer = LadderObservers(*watchers) if watchers else LadderObserver()
        self._digits = MnistDigits()
        self.reports: list[LadderLevelReport] = []
        self.top_accuracy: dict[int, float] | None = None
        # Итоги для стенда: группы картинок, детали верхнего уровня и отклики уровней (после run).
        self.groups: DigitGroups | None = None
        self.top_details: LadderElements | None = None
        # Срабатывания детекторов верхнего уровня по окнам (§7.1.12: уровни активности — тоже эмбеддинг).
        self.top_firing: dict | None = None
        self.level_responses: list["np.ndarray"] = []
        # Самый сильный уровень активации каждого детектора по окнам картинки, единичной длины, — по уровням (§9.3).
        self.level_peaks: list["np.ndarray"] = []
        self.example: int | None = None
        # keep_levels — сохранить уровни после обучения (TrainedLevel): по ним идёт новая картинка (дашборд). На счёт
        # прогона не влияет; карта и детекторы уровня остаются в памяти, видеокарта под ними освобождается как обычно.
        self._keep_levels = bool(keep_levels)
        self.levels: list = []  # TrainedLevel по уровням (при keep_levels)
        # Уровень 0 — первичная раскладка штрихов (LADDER_LEVEL0): штрихи идут в уровень 1 с выученными кодами.
        self.primary = None

    def run(self) -> dict[int, float] | None:
        """Вся лестница; возвращает узнавание по склейке кодов картинки от всех уровней (None — без проверки)."""
        # Записи прогона помечаются уровнем цикла (0 — стимулы): шкала «cycle» в Rerun, поле cycle в журнале.
        LOGGER.set_cycle(0)
        if self._rerun is not None:
            from damp.figures.rerun_cycle import CycleBlueprint

            # Прогон цикла — своя запись Rerun (окно и файл .rrd) с вкладками этапов 1–7; вкладки заполняются по ходу счёта.
            LOGGER.use_blueprint(CycleBlueprint.build, cycle=True)
            LOGGER.start_recording("cycle")
            self._rerun.follow(tabs=bool(settings.RERUN_FOLLOW_STAGE))
        try:
            return self._run()
        finally:
            LOGGER.set_cycle(None)
            if self._rerun is not None:
                self._rerun.close()

    def _run(self) -> dict[int, float] | None:
        # Группы картинок: обучающие, проверочные и проверочные со сдвигом (первая картинка, после последней).
        self._observer.stage(0, 1)
        groups = DigitGroups.load(
            self._digits,
            train=self.count,
            test=self._evaluate_limit,
            split=self._split,
            shifts=self._shifts,
            shift_pictures=settings.LADDER_SHIFT_PICTURES,
            train_start=self._train_start,
            validation=self._validation,
            pixel_shift=self._pixel_shift,
            pixel_pictures=self._pixel_pictures,
            train_numbers=self._numbers["train"],
            validation_numbers=self._numbers["validation"],
            test_numbers=self._numbers["test"],
            train_views=settings.LADDER_TRAIN_VIEWS,
        )
        self._groups = groups
        self.groups = groups
        atoms = groups.atoms
        # Пример для рисунков и Rerun — одна обучающая картинка цифры-примера на весь прогон (вне статьи).
        self.example = ExampleDigit.choose(groups)
        sticks = StickCoder()
        self._observer.stage(0, 2)
        elements = sticks.encode(atoms)
        self._observer.start(
            LadderRunTrace(
                images=atoms.images,
                labels=atoms.labels,
                stick_picture=atoms.picture,
                stick_y=atoms.y,
                stick_x=atoms.x,
                stick_angle=atoms.angle,
                ranges={str(key): value for key, value in groups.ranges.items() if isinstance(key, str)},
                angle_encoder=sticks.encoder,
                example=self.example,
                example_index=int(atoms.index[self.example]),
            )
        )
        self.reports = []
        responses: list["np.ndarray"] = []
        self.level_responses = responses
        self.level_peaks = []
        self.levels = []
        self.primary = None
        if settings.LADDER_LEVEL0:
            # Уровень 0 (вариант 3 автора, 2026-11-18): первичная раскладка штрихов с местом от центра цифры; код штриха —
            # его сильнейшие детекторы (§6.5, §7.1.12). С ним, а не с готовым кодом наклона, штрих идёт в уровень 1.
            from damp.cycle.primary import StrokeLevel

            self.primary, elements = StrokeLevel.train(groups, elements, map_max_points=self._map_max_points)
            # Видеопамять уровня 0 — как у уровней (LevelPass.release): освобождается сразу после обучения и при
            # keep_levels тоже; детекторы пересобираются при первой новой картинке (LevelDetectors.detection). Иначе
            # карта уровня 0 (30 тыс. точек на 500 цифрах) держит видеокарту всё обучение уровней 1–3, и раскладка
            # следующей карты упирается в память (прогон 446, 2026-11-19).
            self.primary.release()
        elements, reached = self._ladder(elements, responses)
        if reached is None:
            raise RuntimeError("лестница не построила ни одного уровня")
        self.top_details = elements
        self._class_map(elements, groups.labels, reached)
        self._log_summary()
        self._observer.finish(self.reports, self.top_accuracy)
        self._observer.stage(reached.number, 0)
        return self.reports[-1].glued if self.reports else None

    def _glyph_file(self, name: str) -> str | None:
        """Рисунок карты name («level2», «classes») этого прогона, если он нарисован (LADDER_SAVE_MAPS)."""
        path = f"{self.FILE_PREFIX}-{name}-keyholes-{self.count}.glyphs.png" if name.startswith("level") else f"{self.FILE_PREFIX}-{name}-{self.count}.glyphs.png"
        return path if os.path.exists(path) else None

    @staticmethod
    def _cycle_mark(kind: str, number: int, note: str) -> None:
        """Отметка цикла в журнале: начало прохода уровня (этап 2) и его конец (этап 6) — §6.2, §6.6."""
        LOGGER.event(f"ladder.cycle.{kind}", section=NEURAL_ANALOGY, data={"level": number, "note": note})

    def _stopped(self, number: int, reason: str) -> None:
        """Запись о том, что лестница закончилась раньше заданного числа уровней."""
        LOGGER.event(
            "ladder.stopped",
            section=OUTSIDE_ARTICLE,
            data={
                "level": number,
                "levels_built": len(self.reports),
                "reason": reason,
                "note": "уровень не сложился либо поле зрения уже видит картинку целиком; верхним считается "
                "последний построенный уровень",
            },
        )

    # ------------------------------------------------------------------ части прохода по ходу счёта
    def _part(self, number: int, name: str, values: dict) -> None:
        """Часть прохода — в значения уровня (из них собирается LadderLevelTrace) и наблюдателю сразу, по ходу счёта."""
        self._values.update(values)
        self._observer.part(number, name, values)

    def _keyholes_part(self, keyholes, elements: LadderElements) -> dict:  # noqa: ANN001 — LevelKeyholes
        """Окна уровня (П.2, П.3): картинка, центр и радиус окна, вхождения элементов и сами элементы."""
        pieces = keyholes.pieces
        return {
            "unit": float(keyholes.field.unit),
            "seen": float(keyholes.seen),
            "keyhole_radius": float(keyholes.radius),  # средний радиус окна обучающих картинок, клеток
            "keyhole_picture": pieces.picture,
            "keyhole_centre": pieces.centre,
            "keyhole_extent": np.asarray(keyholes.extent, dtype=np.float64),
            "entry_keyhole": pieces.piece,
            "entry_element": pieces.element,
            "entry_offset": pieces.offsets[pieces.offset],
            "element_picture": elements.picture,
            "element_y": elements.y,
            "element_x": elements.x,
        }

    def _watch(self, number: int):  # noqa: ANN202
        """Смотритель прохода CycleStep.run: каждый готовый шаг — часть уровня, затем начинается следующий этап."""

        def watch(name: str, value: object) -> None:
            if name == "codes":
                self._part(number, "codes", {
                    "keyhole_codes": value.packed, "code_length": int(value.code_length),
                    "place_length": int(value.place_length), "content_length": int(value.content_length),
                    "place_encoder": self._step.coder.place,
                })
                self._observer.stage(number, LEVEL_PARTS["memory"])
            elif name == "memory":
                layout = value.layout
                table = layout.code_table
                self._part(number, "memory", {
                    "map_cells": layout.cells_array(), "map_size": (int(layout.height), int(layout.width)),
                    "map_codes": np.ascontiguousarray(table.packed),
                    "map_labels": np.asarray([int(label) for label in table.labels()], dtype=np.int64),
                    "energies": np.asarray(value.energies, dtype=np.float64),
                    "energy_threshold": float(settings.ACTIVATION_ENERGY_THRESHOLD),
                })
                self._observer.stage(number, LEVEL_PARTS["detectors"])
            elif name == "detectors":
                self._part(number, "detectors", {
                    "detectors": np.asarray(
                        [(d.center[0], d.center[1], d.radius, d.lambda_threshold, d.energy) for d in value.detectors], dtype=np.float64
                    ),
                    "layer_index": np.asarray(value.layer_index),
                    "layer_thresholds": tuple(float(threshold) for threshold in value.layer_thresholds),
                    "activation_lambda": float(value.activation_lambda),
                    "detector_threshold": float(settings.LADDER_DETECTOR_THRESHOLD),
                })
            elif name == "firing":
                self._part(number, "firing", {"fired_keyhole": value.rows, "fired_detector": value.columns, "fired_level": value.levels})
                self._observer.stage(number, LEVEL_PARTS["details"])
            elif name == "details":
                details, length, strength, _ = value
                self._part(number, "details", {"detail_codes": details, "detail_length": int(length), "detail_strength": strength})
                self._observer.stage(number, LEVEL_PARTS["recognition"])

        return watch

    def _ladder(self, elements: LadderElements, responses: list["np.ndarray"]) -> tuple[LadderElements, LadderField | None]:
        """Уровни один за другим (проход — CycleStep), пока поле зрения не увидит картинку целиком: последний уровень
        — первый, в окно которого цифра помещается вся (слово автора 2026-11-18: «редуцируемся, пока всё не влезет»)."""
        groups = self._groups
        atoms = groups.atoms
        step = CycleStep(groups, map_max_points=self._map_max_points)
        self._step = step
        recognition = LevelRecognition(groups.labels, groups.ranges["train"])
        reached: LadderField | None = None
        circles: list[tuple[int, float, float, float]] = []  # окна картинки для рисунка: уровень, центр, радиус
        figure = int(self.example)
        # Рисунок: окна всех уровней вокруг одного места — углопозиции, ближайшей к середине цифры.
        first_stick, last_stick = groups.sticks(figure)
        middle = (float(atoms.y[first_stick:last_stick].mean()), float(atoms.x[first_stick:last_stick].mean()))
        for number in range(1, int(settings.LADDER_CLUSTER_LEVELS) + 1):
            # Пятна (П.3): детали, стоящие ближе доли своего поля и похожие кодом, повторяют друг друга — на
            # следующий уровень идёт самая сильная из них.
            elements = step.thin(elements, number)
            LOGGER.set_cycle(number)
            self._cycle_mark("begin", number, "↻ начало цикла: элементы уровня идут на этап 2 — окна и их коды")
            self._observer.begin(number)
            self._observer.stage(number, LEVEL_PARTS["keyholes"])
            self._values: dict[str, object] = {}
            started = time.perf_counter()
            keyholes = step.keyholes(elements, number)
            self._keyholes_seconds = time.perf_counter() - started
            if keyholes.taught < settings.LADDER_MAP_MIN_POINTS:
                self._stopped(number, f"уровень {number}: окон на обучающих картинках — {keyholes.taught}")
                break
            self._part(number, "keyholes", self._keyholes_part(keyholes, elements))
            self._observer.stage(number, LEVEL_PARTS["codes"])
            try:
                level = step.run(elements, keyholes, watch=self._watch(number))
            except LadderLevelDegenerate as stop:
                self._stopped(number, str(stop))
                break
            elements = self._measure(level, elements, responses, recognition)
            del level  # память видеокарты под детекторами уровня больше не нужна
            reached = keyholes.field
            pieces = keyholes.pieces
            own = np.flatnonzero(pieces.picture == figure)
            if own.size:
                # Одно окно уровня — то, что стоит ближе всего к середине цифры.
                near = own[np.argmin(np.hypot(pieces.centre[own, 0] - middle[0], pieces.centre[own, 1] - middle[1]))]
                circles.append((number, float(pieces.centre[near, 0]), float(pieces.centre[near, 1]), float(keyholes.extent[near])))
            covered = keyholes.seen >= settings.LADDER_TOP_COVERAGE
            self._cycle_mark(
                "end", number,
                "↺ конец цикла: окно видит цифру целиком — уровень верхний" if covered
                else "↺ конец прохода: детали уровня снова идут на этап 2",
            )
            if covered:
                # Следующий уровень был бы лишним: его окно вместит картинку целиком и нового не увидит.
                self._stopped(number, f"уровень {number}: поле видит {keyholes.seen:.0%} углопозиций своей картинки")
                break
        self._cluster_figure(atoms, figure, circles)
        return elements, reached

    def _measure(
        self, level: LevelPass, elements: LadderElements, responses: list["np.ndarray"], recognition: LevelRecognition
    ) -> LadderElements:
        """Мерки, рисунки и журнал прохода; возвращает детали — элементы следующего уровня."""
        keyholes, memory, found = level.keyholes, level.memory, level.detectors
        field, pieces, extent, seen = keyholes.field, keyholes.pieces, keyholes.extent, keyholes.seen
        number, unit = field.number, field.unit
        groups = self._groups
        labels = groups.labels
        train_pictures = groups.train_pictures
        pictures = int(labels.shape[0])
        layout, dictionary = memory.layout, memory.dictionary
        if settings.LADDER_SAVE_MAPS:
            self._export(layout, memory.name, memory.energies)
            self._keyhole_glyphs(layout, dictionary, pieces, memory.rows, field, extent)
        rows, columns, levels = level.firing.rows, level.firing.columns, level.firing.levels
        # Для стенда: уровни активации детекторов по окнам; последний записанный уровень — верхний (§7.1.12).
        self.top_firing = {
            "rows": rows, "columns": columns, "levels": levels, "detectors": int(found.built),
            "picture": np.asarray(pieces.picture),
            # Господство классов в детекторах (§8.2) — по точкам карты, то есть по обучающим окнам.
            "dominance": found.dominance(int(labels.max()) + 1) if settings.BENCH_DOMINANCE else None,
        }
        strength = level.strength
        response, peak = LevelRecognition.responses(pieces.picture, rows, columns, levels, pictures, found.built)
        responses.append(response)
        self.level_peaks.append(peak)
        codes = level.codes
        result = level.elements
        what, what_index = result.what_packed, result.what_index
        per_field = np.bincount(pieces.piece, minlength=pieces.count)[pieces.picture < train_pictures]
        alone = glued = unshifted = strongest = None
        shifted: dict[tuple[int, int], float] = {}
        same_response: dict[tuple[int, int], float] = {}
        ranges = groups.ranges
        # Метки уровня — по проверочным; нет их, но есть отложенные (стенд) — по отложенным.
        checked = "test" if "test" in ranges else "validation" if "validation" in ranges else None
        if checked is not None:
            alone = recognition.by_response(response, ranges[checked])
            strongest = recognition.by_response(peak, ranges[checked])
            # Склейка откликов уровней 1…number: у каждого уровня свой сегмент единичной длины (§2.2.3).
            glued = recognition.by_response(np.concatenate(responses, axis=1), ranges[checked])
            first = ranges[checked][0]
            moved = min(int(settings.LADDER_SHIFT_PICTURES), ranges[checked][1] - first)
            if self._shifts and moved:
                unshifted = recognition.by_response(response, (first, first + moved))[min(settings.LADDER_NEIGHBOURS)]
            for shift in self._shifts:
                if tuple(shift) not in ranges:
                    continue  # сдвиги — проверочных картинок; без них (стенд на отложенных) сдвигов нет
                begin, end = ranges[tuple(shift)]
                shifted[tuple(shift)] = recognition.by_response(response, (begin, end))[min(settings.LADDER_NEIGHBOURS)]
                # Отклики единичной длины: их скалярное произведение — косинус; два пустых отклика совпадают.
                own, other = response[first : first + (end - begin)], response[begin:end]
                silent = ~own.any(axis=1) & ~other.any(axis=1)
                same_response[tuple(shift)] = float(np.mean(np.where(silent, 1.0, np.sum(own * other, axis=1))))
        # Узнавание по уровню — часть прохода (этап 7), затем итог прохода целиком (части уровня, собранные по ходу).
        self._part(number, "recognition", {"response": response, "alone": alone, "peak": strongest, "glued": glued})
        trace_fields = {item.name for item in dataclass_fields(LadderLevelTrace)}
        self._observer.level(
            LadderLevelTrace(
                number=number,
                labels=labels,
                seconds={"keyholes": float(getattr(self, "_keyholes_seconds", 0.0)), **level.seconds},
                **{name: value for name, value in self._values.items() if name in trace_fields},
            )
        )
        quality = memory.quality
        measures = self._detail_measures(level, memory)
        report = LadderLevelReport(
            number=number,
            field_height=field.height,
            field_width=field.width,
            field_radius=float(extent[pieces.picture < train_pictures].mean()),
            fields_per_picture=float(per_field.shape[0]) / max(1, train_pictures),
            picture_seen=seen,
            pictures_without_fields=1.0
            - float(np.unique(pieces.picture[pieces.picture < train_pictures]).shape[0]) / max(1, train_pictures),
            elements_per_field=float(per_field.mean()) if per_field.size else 0.0,
            fields_with_5_to_7=float(np.mean((per_field >= 5) & (per_field <= 7))) if per_field.size else 0.0,
            map_points=dictionary.count,
            map_code_length=int(codes.code_length),
            map_code_bits=float(packed_ones(dictionary.packed).mean()),
            detectors=found.built,
            detail_code_length=level.detail_length,
            detail_bits=float(packed_ones(what)[what_index].mean()),
            distinct_details=int(what.shape[0]),
            **measures,
            same_label=quality["same_label"],
            same_label_random=quality["same_label_random"],
            alone=alone,
            peak=strongest,
            glued=glued,
            shifted=shifted,
            unshifted=unshifted,
            same_details=self._same_details(result),
            same_response=same_response,
        )
        self.reports.append(report)
        LOGGER.event(
            "ladder.level",
            section=NEURAL_ANALOGY,
            data={
                "level": number,
                "fields": "keyholes",
                "field_height": report.field_height,
                "field_width": report.field_width,
                "field_radius": report.field_radius,
                "unit": unit,
                "fields_per_picture": report.fields_per_picture,
                "picture_seen": float(seen),
                "pictures_without_fields": report.pictures_without_fields,
                "elements_per_field": report.elements_per_field,
                "elements_per_field_quartiles": (
                    [float(v) for v in np.percentile(per_field, (10, 50, 90))] if per_field.size else []
                ),
                "fields_with_5_to_7": report.fields_with_5_to_7,
                "stimuli": codes.count,
                "distinct_stimuli": codes.count,
                "map_points": report.map_points,
                "map_code_length": report.map_code_length,
                "map_code_bits": report.map_code_bits,
                "strength_mean": float(strength.mean()) if strength.size else 0.0,
                "map_side": int(max(layout.height, layout.width)),
                "detectors": report.detectors,
                "layer_thresholds": [float(value) for value in found.thresholds],
                "activation_lambda": found.activation_lambda,
                "detail_code_length": report.detail_code_length,
                "detail_bits": report.detail_bits,
                "distinct_details": report.distinct_details,
                "energy_mean": report.energy_mean,
                "energy_radius": int(settings.ACTIVATION_POINT_ENERGY_RADIUS),
                "code_resolution": report.code_resolution,
                "code_largest": report.code_largest,
                "detail_empty": report.detail_empty,
                "detail_full": report.detail_full,
                "level_normalisation": settings.LADDER_LEVEL_NORMALISATION or "none",
                "level_normalisation_note": "способ нормировки уровней активации - доработка автора проекта "
                "(вне статьи): §7.1.12 и §9.3 требуют лишь значений в [0, 1]",
                "code_measures_note": "code_resolution и code_largest - мера §5.13.2.5, перенесённая с кодов точек "
                "карты на коды деталей (вне статьи); пустой код деталей не считается, их доля - detail_empty",
                "same_label": report.same_label,
                "same_label_random": report.same_label_random,
                "accuracy_alone": {str(k): v for k, v in (alone or {}).items()},
                "accuracy_peak": {str(k): v for k, v in (strongest or {}).items()},
                "accuracy_glued": {str(k): v for k, v in (glued or {}).items()},
                "accuracy_unshifted": unshifted,
                "accuracy_shifted": {f"{dy},{dx}": value for (dy, dx), value in shifted.items()},
                "same_details_shifted": {f"{dy},{dx}": value for (dy, dx), value in report.same_details.items()},
                "same_response_shifted": {f"{dy},{dx}": value for (dy, dx), value in same_response.items()},
                "note": "шаг лестницы по циклу §6.6: карта кодов уровня (общая память миниколонок), иерархия "
                "детекторов над ней и эмбеддинги полей зрения — детали следующего уровня (§6.2); стимул карты — код "
                "окна; узнавание и согласие классов — мерки «собрались ли "
                "классы», вне статьи",
            },
        )
        if self._keep_levels:
            self.levels.append(level.trained())
        level.release()
        return result

    def _detail_measures(self, level: LevelPass, memory: LevelMemory) -> dict[str, float | int]:
        """Мерки статьи по уровню: Ē карты (§5.8), разрешение и наибольший набор окон с одним кодом детали
        (§5.13.2.5), доли пустых и упёршихся в σ деталей (§7.1.8.1). Коды деталей — по окнам обучающих картинок.

        Мера §5.13.2.5 («resolution per code», «largest cluster size») в статье относится к кодам точек карты;
        здесь она перенесена на коды деталей (вне статьи). Пустой код (ни один детектор не сработал) в расчёт
        не входит, иначе наибольший «кластер» повторял бы долю пустых деталей: она отдельно — detail_empty."""
        trained = level.keyholes.pieces.picture < self._groups.train_pictures
        what, what_index = level.elements.what_packed, level.elements.what_index
        own = what_index[trained]
        empty_code = ~np.asarray(what).any(axis=1)
        filled = own[~empty_code[own]]  # окна обучающих картинок с непустым кодом
        sets = int(filled.shape[0])
        counts = np.bincount(filled, minlength=int(what.shape[0]))
        distinct = int(np.count_nonzero(counts))
        rows, columns = level.firing.rows, level.firing.columns
        if settings.LADDER_EMBEDDING_LAYERS is not None:
            coded = level.detectors.layer_index[columns] < int(settings.LADDER_EMBEDDING_LAYERS)
            rows = rows[coded]
        fired = np.bincount(rows, minlength=int(level.firing.sets))
        limit = int(settings.LADDER_SATURATION)
        return {
            "energy_mean": float(np.mean(memory.energies)) if len(memory.energies) else 0.0,
            "code_resolution": float(sets) / max(1, distinct),
            "code_largest": int(counts.max()) if counts.size else 0,
            "detail_empty": float(np.mean(empty_code[own])) if own.shape[0] else 0.0,
            "detail_full": float(np.mean(fired[trained] >= limit)) if own.shape[0] and limit > 0 else 0.0,
        }

    def _same_details(self, details: LadderElements) -> dict[tuple[int, int], float]:
        """Доля общих кодов деталей у картинки и её сдвига (мера Жаккара по различным кодам, среднее по картинкам)."""
        result: dict[tuple[int, int], float] = {}
        ranges = self._groups.ranges
        if "test" not in ranges:
            return result
        origin = ranges["test"][0]
        codes = int(details.what_packed.shape[0])
        for shift in self._shifts:
            first, last = ranges[tuple(shift)]
            if last <= first:
                continue
            own = (details.picture >= origin) & (details.picture < origin + (last - first))
            moved = (details.picture >= first) & (details.picture < last)
            left = np.unique((details.picture[own] - origin) * codes + details.what_index[own])
            right = np.unique((details.picture[moved] - first) * codes + details.what_index[moved])
            common = np.bincount(np.intersect1d(left, right) // codes, minlength=last - first)
            total = np.bincount(left // codes, minlength=last - first) + np.bincount(right // codes, minlength=last - first)
            union = total - common
            # Картинка без деталей уровня и её сдвиг без деталей совпадают.
            result[tuple(shift)] = float(np.mean(np.where(union > 0, common / np.maximum(union, 1), 1.0)))
        return result

    # ------------------------------------------------------------------ верх лестницы
    def _class_map(self, details: LadderElements, labels: "np.ndarray", field: LadderField) -> None:
        """Карта верхних деталей и узнавание по ним: класс картинки — голосование её верхних деталей (§2.2.5)."""
        number = field.number
        groups = self._groups
        train_pictures = groups.train_pictures
        codes = details.codes()
        memory_map = self._step.memory
        dictionary, train_rows = memory_map.samples(
            codes,
            details.what_length,
            details.picture,
            train_pictures,
            merge_similarity=settings.LADDER_CLASS_MERGE_SIMILARITY,
            max_points=self._step.map_limit(),
            shuffle_seed=None,
        )
        rows = train_rows[dictionary.rows]
        memory = memory_map.remember(dictionary, rows, labels[details.picture[rows]], "classes")
        table = memory.layout.code_table
        self._observer.top(
            LadderTopTrace(
                level=int(number),
                picture=details.picture,
                y=details.y,
                x=details.x,
                codes=codes,
                code_length=int(details.what_length),
                map_cells=memory.layout.cells_array(),
                map_size=(int(memory.layout.height), int(memory.layout.width)),
                map_labels=np.asarray([int(label) for label in table.labels()], dtype=np.int64),
            )
        )
        if settings.LADDER_SAVE_MAPS:
            self._export(memory.layout, "classes", memory.energies)
            self._detail_glyphs(memory.layout, dictionary, details, rows, field)
        memory.release()
        accuracy: dict[int, float] | None = None
        if "test" in groups.ranges:
            accuracy = LevelRecognition(labels, groups.ranges["train"]).by_vote(
                codes, details.what_length, details.picture, groups.ranges["test"]
            )
        self.top_accuracy = accuracy
        LOGGER.event(
            "ladder.classes",
            section=FUZZY_SEARCH,
            data={
                "level": number,
                "details": int(train_rows.shape[0]),
                "details_per_picture": float(train_rows.shape[0]) / max(1, train_pictures),
                "map_points": dictionary.count,
                "same_label": memory.quality["same_label"],
                "same_label_random": memory.quality["same_label_random"],
                "accuracy_vote": {str(k): v for k, v in (accuracy or {}).items()},
                "note": "карта верхних деталей: собрались ли классы рядом; класс проверочной картинки — "
                "голосование её верхних деталей по самым похожим деталям обучающих (поиск снизу вверх, §2.2.5)",
            },
        )

    def _log_summary(self) -> None:
        """Сводка лестницы одной записью: уровни по порядку."""
        LOGGER.event(
            "ladder.summary",
            section=NEURAL_ANALOGY,
            data={
                "levels": len(self.reports),
                "fields": [f"{report.field_height}x{report.field_width}" for report in self.reports],
                "field_radius": [report.field_radius for report in self.reports],
                "picture_seen": [report.picture_seen for report in self.reports],
                "fields_per_picture": [report.fields_per_picture for report in self.reports],
                "elements_per_field": [report.elements_per_field for report in self.reports],
                "map_points": [report.map_points for report in self.reports],
                "same_label": [report.same_label for report in self.reports],
                "accuracy_alone": [None if report.alone is None else report.alone[min(report.alone)] for report in self.reports],
                "accuracy_glued": [None if report.glued is None else report.glued[min(report.glued)] for report in self.reports],
                "accuracy_top_vote": {str(k): v for k, v in (self.top_accuracy or {}).items()},
                "note": "лестница уровней в локальных координатах: поле зрения, число деталей и мерки «собрались "
                "ли классы» по уровням",
            },
        )

    # ------------------------------------------------------------------ файлы и рисунки
    def _export(self, layout: Layout, name: str, energies: Sequence[float]) -> None:
        """Карта уровня в JSON и картинками (псевдоцвета, угол, энергия); energies — энергии точек карты (§5.7)."""
        from damp.activation.first_level import MnistLayoutRunner

        suffix = "auto" if layout.pair_radius is None else str(layout.pair_radius)
        stem = f"{self.FILE_PREFIX}-{name}-{self.count}-{suffix}"
        layout.save_json(f"{stem}-{settings.LAYOUT_CACHE_TAG}.json")
        MnistLayoutRunner._save_layout_images(
            layout,
            image_path=f"{stem}.png",
            energies=energies,
        )

    @staticmethod
    def _hue_colour(hue: float) -> tuple[int, int, int]:
        import colorsys

        red, green, blue = colorsys.hsv_to_rgb((hue % 360.0) / 360.0, 0.9, 1.0)
        return int(red * 255), int(green * 255), int(blue * 255)

    @classmethod
    def _draw_stick(cls, draw, px: float, py: float, unit: float, angle: float, *, dim: float = 1.0) -> None:  # noqa: ANN001
        """Углопозиция стрелкой в точке (px, py); unit — пикселей на клетку, dim — доля яркости."""
        # Угол углопозиции — направление перепада яркости, оно поперёк линии цифры. Отрезок рисуется
        # вдоль края (поворот на 90°) стрелкой: так глаз видит саму линию.
        along = math.radians(angle + 90.0)
        half = unit * 0.45
        # Угол отсчитан с осью Y вверх (ядро Собеля по Y — верх минус низ), на картинке ось Y вниз:
        # вертикальная составляющая берётся с обратным знаком.
        ux, uy = math.cos(along) * half, -math.sin(along) * half
        colour = tuple(int(channel * dim) for channel in cls._hue_colour(angle))
        width = 2 if unit >= 5 else 1
        draw.line((px - ux, py - uy, px + ux, py + uy), fill=colour, width=width)
        if half >= 4:  # наконечник различим только на достаточно длинной стрелке
            for turn in (150.0, -150.0):
                side = along + math.radians(turn)
                draw.line(
                    (px + ux + math.cos(side) * half * 0.7, py + uy - math.sin(side) * half * 0.7, px + ux, py + uy),
                    fill=colour,
                    width=width,
                )

    def _cluster_figure(self, atoms: DigitAtoms, picture: int, circles: Sequence[tuple[int, float, float, float]]) -> None:
        """Картинка с найденными полями зрения одной цифры: окружность — поле, цвет и номер — уровень."""
        from PIL import Image, ImageDraw

        first, last = self._groups.sticks(picture)
        size = int(settings.LADDER_FIGURE_SIZE)
        low_y = min([0.0] + [cy - r for _, cy, _, r in circles]) - 1.5
        high_y = max([settings.LADDER_PICTURE_GRID - 1.0] + [cy + r for _, cy, _, r in circles]) + 1.5
        low_x = min([0.0] + [cx - r for _, _, cx, r in circles]) - 1.5
        high_x = max([settings.LADDER_PICTURE_GRID - 1.0] + [cx + r for _, _, cx, r in circles]) + 1.5
        scale = size / max(high_y - low_y, high_x - low_x)
        centre_y, centre_x = (low_y + high_y) / 2.0, (low_x + high_x) / 2.0

        def to_px(value: float, centre: float) -> float:
            return size / 2.0 + (value - centre) * scale

        canvas = Image.new("RGB", (size, size), (0, 0, 0))
        upscale = settings.MNIST_SOBEL_UPSCALE if settings.MNIST_SOBEL_UPSCALE_ENABLED else 1
        side = int(round(float(atoms.images.shape[1]) * upscale / settings.MNIST_PATCH_SIZE * scale))
        digit = Image.fromarray((atoms.images[picture].astype(np.float64) * 0.45).astype(np.uint8)).resize((side, side))
        canvas.paste(digit.convert("RGB"), (int(round(to_px(-0.5, centre_x))), int(round(to_px(-0.5, centre_y)))))
        draw = ImageDraw.Draw(canvas)
        for index in range(first, last):
            self._draw_stick(
                draw, to_px(float(atoms.x[index]), centre_x), to_px(float(atoms.y[index]), centre_y), scale,
                float(atoms.angle[index]), dim=0.6,
            )
        levels = sorted({number for number, _, _, _ in circles})
        for number, cy, cx, radius in circles:
            colour = self._hue_colour(300.0 * levels.index(number) / max(1, len(levels) - 1)) if len(levels) > 1 else (255, 90, 90)
            px, py, r = to_px(cx, centre_x), to_px(cy, centre_y), radius * scale
            draw.ellipse((px - r, py - r, px + r, py + r), outline=colour, width=2)
            draw.text((px + r * 0.71 + 2, py - r * 0.71 - 12), str(number), fill=colour)
        path = f"{self.FILE_PREFIX}-fields-{self.count}.png"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        canvas.save(path)
        LOGGER.event(
            "ladder.fields",
            section=OUTSIDE_ARTICLE,
            data={
                "image": path,
                "picture": picture,
                "label": int(atoms.labels[picture]),
                "sticks": last - first,
                "fields_by_level": [sum(1 for number, _, _, _ in circles if number == level) for level in levels],
                "radius_by_level": [
                    float(np.mean([radius for number, _, _, radius in circles if number == level])) for level in levels
                ],
                "note": "окружности — окна внимания уровней вокруг одного места цифры (П.2, П.3); цвет и номер — "
                "уровень; радиус — сколько клеток картинки видит окно",
            },
            visuals=[LOGGER.visual_image("ladder/fields", np.asarray(canvas))] if LOGGER.visuals_enabled() else None,
        )

    def _keyhole_glyphs(
        self,
        layout: Layout,
        dictionary: CodeDictionary,
        pieces,  # noqa: ANN001 — KeyholePieces
        rows: "np.ndarray",
        field: LadderField,
        extent: "np.ndarray",
    ) -> None:
        """Карта уровня рисунками: в ячейке точки — её окно внимания: углопозиции картинки в поле зрения окна."""
        self._field_glyphs(
            layout,
            dictionary,
            pieces.picture[rows],
            pieces.centre[rows, 0],
            pieces.centre[rows, 1],
            field,
            extent[rows],
            path=f"{self.FILE_PREFIX}-level{field.number}-keyholes-{self.count}.glyphs.png",
            title=f"Уровень {field.number}: карта окон внимания, {dictionary.count} точек; ярко — что видит окно, "
            f"тускло — соседние углопозиции; ячейка {field.height}×{field.width} клеток, рамка — класс цифры",
        )

    def _detail_glyphs(
        self, layout: Layout, dictionary: CodeDictionary, details: LadderElements, rows: "np.ndarray", field: LadderField
    ) -> None:
        """Карта верхних деталей рисунками: в ячейке — углопозиции картинки в поле зрения детали, рамка — класс."""
        self._field_glyphs(
            layout,
            dictionary,
            details.picture[rows],
            details.y[rows],
            details.x[rows],
            field,
            details.reach()[rows],
            path=f"{self.FILE_PREFIX}-classes-{self.count}.glyphs.png",
            title=f"Верхние детали уровня {field.number}: карта классов, {dictionary.count} точек; рамка — класс цифры",
        )

    def _field_glyphs(
        self,
        layout: Layout,
        dictionary: CodeDictionary,
        picture: "np.ndarray",
        node_y: "np.ndarray",
        node_x: "np.ndarray",
        field: LadderField,
        sight: "np.ndarray",
        *,
        path: str,
        title: str,
    ) -> None:
        """Карта рисунками: в ячейке точки — углопозиции её картинки вокруг центра окна, рамка — класс.

        sight — сколько клеток от своего центра видит окно каждой точки: эти углопозиции рисуются ярко, остальные
        углопозиции ячейки — тускло. Ячейка у всех точек одна (среднее поле уровня), а окна разного размера.
        """
        atoms = self._groups.atoms
        middle_y, middle_x = field.middle
        reach = max(field.height, field.width) / 2.0 + 0.5

        def paint(canvas, draw, sample: int, left: int, top: int, size: int) -> None:  # noqa: ANN001
            scale = size / (2.0 * reach)
            label = float(atoms.labels[picture[sample]]) * settings.LADDER_CLASS_HUE_STEP
            draw.rectangle((left, top, left + size - 1, top + size - 1), outline=self._hue_colour(label))
            first, last = self._groups.sticks(int(picture[sample]))
            for index in range(first, last):
                dy, dx = atoms.y[index] - node_y[sample], atoms.x[index] - node_x[sample]
                if not field.sees(dy, dx):
                    continue
                # Углопозиция видит свою клетку: она в окне, если её клетка задета кругом окна.
                inside = math.hypot(dy, dx) <= float(sight[sample]) + 1e-9
                self._draw_stick(
                    draw,
                    left + size / 2.0 + (dx - middle_x) * scale,
                    top + size / 2.0 + (dy - middle_y) * scale,
                    scale,
                    float(atoms.angle[index]),
                    dim=1.0 if inside else 0.22,
                )

        from pathlib import Path

        from damp.cache import CACHE

        drawn: list[bool] = []

        def compute() -> dict[str, object]:
            glyphs = LayoutGlyphMap(layout, dictionary.packed)
            glyphs.save(path, paint, tile=glyphs.tile_size(settings.LADDER_VIEW_TILE), title=title)
            drawn.append(True)
            return {"png": np.frombuffer(Path(path).read_bytes(), dtype=np.uint8).copy()}

        # Рисунок — тоже шаг кеша: та же карта и те же окна — картинка берётся с диска.
        inputs = {
            "cells": layout.cells_array(), "width": int(layout.width), "height": int(layout.height), "codes": dictionary.packed,
            "picture": np.asarray(picture), "y": np.asarray(node_y), "x": np.asarray(node_x), "sight": np.asarray(sight),
            "box": field.box, "atom_picture": atoms.picture, "atom_angle": atoms.angle, "atom_y": atoms.y, "atom_x": atoms.x,
            "labels": atoms.labels, "title": title,
        }
        found = CACHE.run("figures.map_glyphs", inputs, compute, modules=("damp.cycle.ladder",))
        if not drawn:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(np.asarray(found["png"], dtype=np.uint8).tobytes())
