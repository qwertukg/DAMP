"""Этап 6: один проход цикла — этапы 2–5 над элементами уровня (§6.2, §6.6; продолжение статьи, П.1–П.7).

Окна внимания (П.2) → код окна (этап 2, П.4–П.7) → память уровня (этап 3, §5) → детекторы и срабатывания
(этап 4, §6.4, §6.5) → детали (этап 5, §6.5, П.1). Детали прохода — элементы следующего уровня.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from damp import settings
from damp.activation.details import DetailCoder
from damp.activation.level_detectors import Firing, LevelDetectors
from damp.cycle.keyholes import KeyholeBuilder, LadderField, LevelKeyholes
from damp.cycle.lateral_inhibition import LateralInhibition
from damp.decode.merge_purity import MergePurity
from damp.encoding.elements import LadderElements
from damp.encoding.position_binding import KeyholeCodes
from damp.encoding.keyhole_code import KeyholeCoder
from damp.layout.level_memory import LevelMemory, MemoryMap
from damp.stimuli.digits import DigitGroups


class LadderLevelDegenerate(RuntimeError):
    """Уровень не сложился: окон нет, коды окон слились либо над картой нет детекторов."""


@dataclass(frozen=True)
class LevelPass:
    """Итог прохода уровня: окна и их коды, память, детекторы, срабатывания и детали."""

    keyholes: LevelKeyholes
    codes: KeyholeCodes
    memory: LevelMemory
    detectors: LevelDetectors
    firing: Firing
    details: "np.ndarray"  # код детали каждого окна
    detail_length: int
    strength: "np.ndarray"  # уровень активации детали (П.1)
    elements: LadderElements  # детали как элементы следующего уровня
    seconds: dict = field(default_factory=dict)  # секунды шагов: коды окон, память, детекторы, срабатывания, детали

    @property
    def number(self) -> int:
        return self.keyholes.number

    def release(self) -> None:
        """Освободить память видеокарты под картой уровня и срабатыванием детекторов."""
        self.detectors.release()
        self.memory.release()

    def trained(self) -> "TrainedLevel":
        """Уровень после обучения — всё, что подобрано по обучающим картинкам; срабатывания и коды всех окон в нём
        не хранятся (они на каждый уровень — сотни мегабайт)."""
        from damp.layout.glyph_map import LayoutGlyphMap

        # Точка карты — образец словаря: строка кода с тем же кодом (раскладка держит точки в своём порядке).
        sample = LayoutGlyphMap(self.memory.layout, self.memory.dictionary.packed).rows
        rows = np.asarray(self.memory.rows, dtype=np.int64)[np.maximum(sample, 0)]
        pieces = self.keyholes.pieces
        return TrainedLevel(
            field=self.keyholes.field,
            step=float(self.keyholes.step),
            thresholds=tuple(float(value) for value in self.keyholes.thresholds),
            content_length=int(self.codes.content_length),
            memory=self.memory,
            detectors=self.detectors,
            detail_length=int(self.detail_length),
            scale=self.firing.scale,
            point_picture=np.where(sample >= 0, pieces.picture[rows], -1),
            point_centre=pieces.centre[rows],
            point_extent=np.asarray(self.keyholes.extent, dtype=np.float64)[rows],
        )


@dataclass(frozen=True)
class TrainedLevel:
    """Уровень цикла после обучения (§6.6): всё, что подобрано по обучающим картинкам, — по нему идёт новая картинка.

    Мера плоскости и пороги слоёв окон (§6.4.1–§6.4.3, П.2), длина сегмента содержания кода окна (П.5, §6.6),
    карта памяти (§5) с детекторами (§6.4), делители нормировки уровней активации (вне статьи) и длина кода детали
    (§6.6, §7.1.12). Каждая точка карты помнит окно обучающей картинки, чей код она хранит: картинку, центр окна и его
    поле, — по ним видно содержимое точки (вне статьи, для показа).
    """

    field: LadderField
    step: float
    thresholds: tuple[float, ...]
    content_length: int
    memory: LevelMemory
    detectors: LevelDetectors
    detail_length: int
    scale: "np.ndarray | None"  # делители нормировки уровней по детекторам; None — без нормировки
    point_picture: "np.ndarray"  # обучающая картинка окна каждой точки карты (порядок точек раскладки)
    point_centre: "np.ndarray"  # центр этого окна (y, x), в клетках сетки картинки (дробный при MNIST_SUBCELL_POSITION)
    point_extent: "np.ndarray"  # сколько клеток картинки видит это окно

    @property
    def number(self) -> int:
        return self.field.number


class CycleStep:
    """Проход цикла над элементами уровня; общие для всех уровней части — кодер места окон и раскраска карт."""

    def __init__(self, groups: DigitGroups, *, map_max_points: int | None) -> None:
        self.groups = groups
        self.map_max_points = None if map_max_points is None else int(map_max_points)
        self.attention = KeyholeBuilder(groups)
        self.coder = KeyholeCoder()
        self.memory = MemoryMap()

    def map_limit(self) -> int:
        """Предел числа точек карты уровня; без предела — заведомо больше числа кодов."""
        return self.map_max_points if self.map_max_points is not None else settings.LADDER_DICTIONARY_MAX

    @staticmethod
    def thin(elements: LadderElements, number: int) -> LadderElements:
        """Элементы уровня number после пятен (П.3): из соседних деталей с похожим кодом остаётся сильнейшая
        (LADDER_INHIBITION_RADIUS, LADDER_INHIBITION_SIMILARITY). Штрихи (уровень 1) и выключенные пятна — те же
        элементы. Один вход для прогона цикла, кубиков фреймворка и пути одной картинки."""
        if int(number) <= 1 or settings.LADDER_INHIBITION_RADIUS is None:
            return elements
        return LateralInhibition.thin(elements, f"details{int(number) - 1}")

    def keyholes(self, elements: LadderElements, number: int) -> LevelKeyholes:
        """Окна внимания уровня number (П.2, П.3)."""
        return self.attention.build(elements, number)

    def encode(self, elements: LadderElements, keyholes: LevelKeyholes) -> KeyholeCodes:
        """Коды окон уровня (П.4–П.7)."""
        pieces = keyholes.pieces
        if pieces.count == 0:
            raise LadderLevelDegenerate(f"уровень {keyholes.number}: ни одного окна внимания")
        taught = int(np.count_nonzero(pieces.picture < self.groups.train_pictures))
        return self.coder.encode(elements, pieces, number=keyholes.number, unit=keyholes.field.unit, taught=taught)

    def remember(self, codes: KeyholeCodes, keyholes: LevelKeyholes, *, merge_similarity: float | None = None) -> LevelMemory:
        """Память уровня из кодов окон обучающих картинок (§5); с пределом карты окна обходятся вразброс.
        merge_similarity — порог слияния кодов к образцу; None — LADDER_MERGE_SIMILARITY (уровень 0 берёт свой)."""
        number = keyholes.number
        picture = keyholes.pieces.picture
        # Коды окон почти все различны, и с пределом карты образцами стали бы окна первых картинок: окна
        # обходятся в случайном порядке, чтобы на карту попали окна всех обучающих картинок.
        dictionary, train_rows = self.memory.samples(
            codes.packed,
            codes.code_length,
            picture,
            self.groups.train_pictures,
            merge_similarity=float(settings.LADDER_MERGE_SIMILARITY if merge_similarity is None else merge_similarity),
            max_points=self.map_limit(),
            shuffle_seed=None if self.map_max_points is None else settings.LADDER_RANDOM_SEED + number,
        )
        if dictionary.count < settings.LADDER_MAP_MIN_POINTS:
            raise LadderLevelDegenerate(
                f"уровень {number}: на карте {dictionary.count} точек из {dictionary.distinct} различных кодов — "
                "коды слились"
            )
        rows = train_rows[dictionary.rows]
        # Карты уровней с номера LADDER_LAYOUT_ARTICLE_FROM — в режиме статьи (§5.5.1: φ = Σ s·d по всей карте, η = ∞, без
        # отжига), даже если нижние раскладываются сглаживанием: ограниченное гауссово ядро сглаживания не тянет друг к
        # другу далёкие куски одного класса, и верх рвётся на пятна (разбор 2026-11-21, docs/CLUSTERS.md). Ключ кеша
        # раскладки видит режим: настройки раскладки входят в него значениями.
        article = 0 < int(settings.LADDER_LAYOUT_ARTICLE_FROM) <= int(number) and bool(settings.LAYOUT_SMOOTH_PINWHEELS)
        # Свои настройки раскладки верхних карт (LADDER_LAYOUT_UPPER_CHANGES с уровня LADDER_LAYOUT_UPPER_FROM): рычаги раскладки
        # верха пробуются, не трогая пинвилы нижних уровней (стратегия v3, docs/CLUSTERS.md).
        upper = dict(settings.LADDER_LAYOUT_UPPER_CHANGES) if 0 < int(settings.LADDER_LAYOUT_UPPER_FROM) <= int(number) else {}
        with settings.overridden({**({"LAYOUT_SMOOTH_PINWHEELS": False} if article else {}), **upper}):
            memory = self.memory.remember(dictionary, rows, self.groups.labels[picture[rows]], f"level{number}")
        # Мерка (вне статьи): одной ли цифры картинки, чьи окна слились в одну точку карты.
        merge = MergePurity(seed=settings.LADDER_RANDOM_SEED + number).measure(
            codes.packed[train_rows], codes.code_length, picture[train_rows], self.groups.labels[picture[train_rows]],
            dictionary, name=f"level{number}",
        )
        return replace(memory, quality={**memory.quality, **merge})

    @staticmethod
    def detect(memory: LevelMemory) -> LevelDetectors:
        """Детекторы над картой уровня (§6.4); нет ни одного — уровень не сложился."""
        try:
            return LevelDetectors.build(memory)
        except RuntimeError as error:
            memory.release()
            raise LadderLevelDegenerate(f"{memory.name}: над картой из {memory.points} точек {error}") from error

    def trained(self, keyholes: LevelKeyholes) -> "np.ndarray":
        """Маска окон обучающих картинок (по маске, а не по порядку строк): по ним подбирается длина кода детали и
        нормировка уровней активации — проверочные окна на обучение не влияют."""
        return np.asarray(keyholes.pieces.picture) < int(self.groups.train_pictures)

    @staticmethod
    def fire(detectors: LevelDetectors, codes: KeyholeCodes, trained: "np.ndarray") -> Firing:
        """Каждое окно сверяется с картой одним кодом (§6.5). LADDER_LEVEL_NORMALISATION = "train_max": уровни
        активации нормируются в [0, 1] по обучающим окнам; trained — маска обучающих окон. Способ нормировки —
        доработка автора проекта (вне статьи): §7.1.12 и §9.3 требуют лишь значений в [0, 1]."""
        every = np.arange(codes.count, dtype=np.int64)
        firing = detectors.fire(codes.packed, every, every, codes.count)
        mode = settings.LADDER_LEVEL_NORMALISATION  # "none" в файле настроек читается как None
        if mode == "train_max":
            firing = firing.normalised(trained, detectors.built)
        elif mode is not None:
            raise ValueError(f"LADDER_LEVEL_NORMALISATION: ожидалось 'train_max' или 'none', получено {mode!r}")
        return firing

    def embed(self, firing: Firing, detectors: LevelDetectors, keyholes: LevelKeyholes) -> tuple["np.ndarray", int, "np.ndarray", LadderElements]:
        """Детали окон (§6.5, П.1): коды, их длина, уровни активации и детали как элементы следующего уровня.
        Длина кода подбирается только по окнам обучающих картинок."""
        coder = DetailCoder(keyholes.number)
        details, length = coder.encode(firing, detectors.layer_index, detectors.built, self.trained(keyholes))
        strength = firing.strength()
        pieces = keyholes.pieces
        elements = coder.elements(details, length, pieces.picture, pieces.centre, keyholes.extent, strength)
        return details, int(length), strength, elements

    def run(self, elements: LadderElements, keyholes: LevelKeyholes, *,
            watch: Callable[[str, object], None] | None = None) -> LevelPass:
        """Этапы 2–5 над окнами уровня.

        watch(имя, итог) — сразу после каждого шага: «codes» (KeyholeCodes), «memory» (LevelMemory), «detectors»
        (LevelDetectors), «firing» (Firing), «details» (коды, длина, уровни активации, элементы). Только смотрит: показ
        прохода по ходу счёта (вне статьи); на счёт не влияет, время показа в секунды шагов не входит.
        """
        seconds: dict[str, float] = {}
        clock = time.perf_counter()

        def lap(name: str, value: object) -> None:
            nonlocal clock
            now = time.perf_counter()
            seconds[name] = now - clock
            if watch is not None:
                watch(name, value)
            clock = time.perf_counter()

        codes = self.encode(elements, keyholes)
        lap("codes", codes)
        memory = self.remember(codes, keyholes)
        lap("memory", memory)
        detectors = self.detect(memory)
        lap("detectors", detectors)
        firing = self.fire(detectors, codes, self.trained(keyholes))
        lap("firing", firing)
        details, length, strength, following = self.embed(firing, detectors, keyholes)
        lap("details", (details, length, strength, following))
        return LevelPass(
            seconds=seconds,
            keyholes=keyholes,
            codes=codes,
            memory=memory,
            detectors=detectors,
            firing=firing,
            details=details,
            detail_length=length,
            strength=strength,
            elements=following,
        )
