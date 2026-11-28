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
        if int(number) <= 1 or LateralInhibition.radius_share(int(number)) is None:
            return elements
        return LateralInhibition.thin(elements, f"details{int(number) - 1}", number=int(number))

    def keyholes(self, elements: LadderElements, number: int) -> LevelKeyholes:
        """Окна внимания уровня number (П.2, П.3)."""
        return self.attention.build(elements, number)

    def encode(self, elements: LadderElements, keyholes: LevelKeyholes) -> KeyholeCodes:
        """Коды окон уровня (П.4–П.7)."""
        pieces = keyholes.pieces
        if pieces.count == 0:
            raise LadderLevelDegenerate(f"уровень {keyholes.number}: ни одного окна внимания")
        taught = int(np.count_nonzero(pieces.picture < self.groups.taught_pictures(keyholes.number)))
        return self.coder.encode(elements, pieces, number=keyholes.number, unit=keyholes.field.unit, taught=taught)

    def remember(self, codes: KeyholeCodes, keyholes: LevelKeyholes, *, merge_similarity: float | None = None) -> LevelMemory:
        """Память уровня из кодов окон обучающих картинок (§5); с пределом карты окна обходятся вразброс.
        merge_similarity — порог слияния кодов к образцу; None — LADDER_MERGE_SIMILARITY (уровень 0 берёт свой)."""
        number = keyholes.number
        picture = keyholes.pieces.picture
        self._clear_device_cache()  # раскладке — чистая видеопамять (см. _clear_device_cache)
        # Коды окон почти все различны, и с пределом карты образцами стали бы окна первых картинок: окна
        # обходятся в случайном порядке, чтобы на карту попали окна всех обучающих картинок.
        dictionary, train_rows = self.memory.samples(
            codes.packed,
            codes.code_length,
            picture,
            self.groups.taught_pictures(number),  # с уровня LADDER_EXTRA_FROM — вместе с добавочными
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
        # Отбор образцов (CodeDictionary.select) считает сходства на видеокарте одним куском в 3.7 ГБ и отдаёт его в запас
        # torch; коды точек раскладки (сотни мегабайт, живут до конца уровня) ложились внутрь этого куска и не давали
        # вернуть его видеокарте — запас в 3.7 ГБ висел весь проход уровня, а таблица раскладки вставала рядом. Запас
        # возвращается до раскладки.
        self._clear_device_cache()
        with settings.overridden({**({"LAYOUT_SMOOTH_PINWHEELS": False} if article else {}), **upper}):
            memory = self.memory.remember(dictionary, rows, self.groups.labels[picture[rows]], f"level{number}")
        self._drop_layout_tables(memory.layout)
        # Мерка (вне статьи): одной ли цифры картинки, чьи окна слились в одну точку карты.
        merge = MergePurity(seed=settings.LADDER_RANDOM_SEED + number).measure(
            codes.packed, codes.code_length, picture[train_rows], self.groups.labels[picture[train_rows]],
            dictionary, name=f"level{number}", rows=train_rows,  # коды — по номерам, без копии всех обучающих окон
        )
        return replace(memory, quality={**memory.quality, **merge})

    @staticmethod
    def on_disk(codes: KeyholeCodes, watch: "Callable[[str, object], None] | None" = None) -> KeyholeCodes:
        """Коды окон после отбора образцов карты — во временный файл, если они большие (вне статьи, техника счёта;
        damp.rows_on_disk). Дальше их читают только срабатывания, кусками по порядку; остальным нужны число и длины.
        На 40 000 картинках так уходит из ОЗУ 4.4 ГБ кодов уровня 0 и 1.3–2.5 ГБ кодов уровней 1–3. watch получает
        «codes_on_disk» с новыми кодами — смотритель отпускает ссылку на массив в ОЗУ. Числа те же."""
        from damp.rows_on_disk import RowsOnDisk

        packed = RowsOnDisk.maybe(codes.packed)
        if packed is codes.packed:
            return codes
        moved = replace(codes, packed=packed)
        if watch is not None:
            watch("codes_on_disk", moved)
        return moved

    @staticmethod
    def _drop_layout_tables(layout) -> None:  # noqa: ANN001
        """Отдать таблицы сходств только что посчитанной раскладки (вне статьи, техника счёта): 4·N² байт на устройстве —
        3.4 ГБ у карты в 29 тыс. точек. Детекторам и срабатываниям они не нужны, а держались до конца прохода уровня
        (2026-11-24: 7.3 из 8 ГБ видеопамяти при срабатываниях на удвоенной карте ур. 3). Таблица строится заново, если
        энергии раскладки понадобятся ещё раз (TorchPairEnergyEngine._fused_engine). Сделано здесь, а не в модулях
        раскладки: их текст входит в ключ кеша карт, и правка там сбросила бы все посчитанные карты."""
        engine = getattr(layout, "_torch_engine", None)
        if engine is None or getattr(engine, "_fused", None) is None:
            return
        engine._fused = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — освобождение памяти не должно ронять проход
            pass

    @staticmethod
    def _clear_device_cache() -> None:
        """Вернуть видеокарте память, которую torch держит про запас после освобождённых тензоров (вне статьи, техника
        счёта). Замер 2026-11-24 на 750 картинках: после построения детекторов ур. 0 у torch оставалось 3.8 ГБ кусками по
        сотне мегабайт (сетки счёта в окне); таблица общих бит раскладки ур. 1 (3.7 ГБ одним куском) в них не помещалась
        и ложилась сверху — 7.9 из 8 ГБ, раскладка шла 307 с вместо ~150 с. Раньше память возвращалась попутно — при
        освобождении срабатываний, — но не тогда, когда отклики уровня взяты из кеша шагов. На результат не влияет."""
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — освобождение памяти не должно ронять проход
            pass

    @classmethod
    def detect(cls, memory: LevelMemory) -> LevelDetectors:
        """Детекторы над картой уровня (§6.4); нет ни одного — уровень не сложился."""
        try:
            found = LevelDetectors.build(memory)
            cls._clear_device_cache()  # срабатываниям — чистая видеопамять
            return found
        except RuntimeError as error:
            memory.release()
            raise LadderLevelDegenerate(f"{memory.name}: над картой из {memory.points} точек {error}") from error

    def trained(self, keyholes: LevelKeyholes) -> "np.ndarray":
        """Маска окон обучающих картинок (по маске, а не по порядку строк): по ним подбирается длина кода детали и
        нормировка уровней активации — проверочные окна на обучение не влияют."""
        return np.asarray(keyholes.pieces.picture) < int(self.groups.taught_pictures(keyholes.number))

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
        # Геометрия детекторов — для топографических бит (v4.11) и предков (v8.2).
        geometry = DetailCoder.geometry(detectors) if coder.needs_geometry() else None
        details, length = coder.encode(firing, detectors.layer_index, detectors.built, self.trained(keyholes), geometry=geometry)
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
        codes = self.on_disk(codes, watch)
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
