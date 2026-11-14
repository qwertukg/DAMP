"""Этап 6: один проход цикла — этапы 2–5 над элементами уровня (§6.2, §6.6; продолжение статьи, П.1–П.7).

Окна внимания (П.2) → код окна (этап 2, П.4–П.7) → память уровня (этап 3, §5) → детекторы и срабатывания
(этап 4, §6.4, §6.5) → детали (этап 5, §6.5, П.1). Детали прохода — элементы следующего уровня.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from damp import settings
from damp.activation.details import DetailCoder
from damp.activation.level_detectors import Firing, LevelDetectors
from damp.cycle.keyholes import KeyholeBuilder, LevelKeyholes
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

    def remember(self, codes: KeyholeCodes, keyholes: LevelKeyholes) -> LevelMemory:
        """Память уровня из кодов окон обучающих картинок (§5); с пределом карты окна обходятся вразброс."""
        number = keyholes.number
        picture = keyholes.pieces.picture
        # Коды окон почти все различны, и с пределом карты образцами стали бы окна первых картинок: окна
        # обходятся в случайном порядке, чтобы на карту попали окна всех обучающих картинок.
        dictionary, train_rows = self.memory.samples(
            codes.packed,
            codes.code_length,
            picture,
            self.groups.train_pictures,
            merge_similarity=float(settings.LADDER_MERGE_SIMILARITY),
            max_points=self.map_limit(),
            shuffle_seed=None if self.map_max_points is None else settings.LADDER_RANDOM_SEED + number,
        )
        if dictionary.count < settings.LADDER_MAP_MIN_POINTS:
            raise LadderLevelDegenerate(
                f"уровень {number}: на карте {dictionary.count} точек из {dictionary.distinct} различных кодов — "
                "коды слились"
            )
        rows = train_rows[dictionary.rows]
        return self.memory.remember(dictionary, rows, self.groups.labels[picture[rows]], f"level{number}")

    @staticmethod
    def detect(memory: LevelMemory) -> LevelDetectors:
        """Детекторы над картой уровня (§6.4); нет ни одного — уровень не сложился."""
        try:
            return LevelDetectors.build(memory)
        except RuntimeError as error:
            memory.release()
            raise LadderLevelDegenerate(f"{memory.name}: над картой из {memory.points} точек {error}") from error

    @staticmethod
    def fire(detectors: LevelDetectors, codes: KeyholeCodes) -> Firing:
        """Каждое окно сверяется с картой одним кодом (§6.5)."""
        every = np.arange(codes.count, dtype=np.int64)
        return detectors.fire(codes.packed, every, every, codes.count)

    @staticmethod
    def embed(firing: Firing, detectors: LevelDetectors, keyholes: LevelKeyholes) -> tuple["np.ndarray", int, "np.ndarray", LadderElements]:
        """Детали окон (§6.5, П.1): коды, их длина, уровни активации и детали как элементы следующего уровня."""
        coder = DetailCoder(keyholes.number)
        details, length = coder.encode(firing, detectors.layer_index, detectors.built)
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
        firing = self.fire(detectors, codes)
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
