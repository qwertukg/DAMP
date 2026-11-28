"""Этап 6: обученный цикл и путь одной новой картинки через все уровни (§6.5, §6.6, §9.3; продолжение статьи, П.1–П.5).

Цикл учится как обычно (MnistLadderRunner с keep_levels): по обучающим картинкам строятся окна, карты памяти и
детекторы уровней, а линейный читатель (§9.3) учится на пиках их срабатываний. Класс читается двумя способами:
голоса памяти (§2.2.5, §7.1.13.3: среднее долей голосов кусочков по уровням и согласие уровней,
decode/memory_votes.py) и линейный читатель (§9.3, мерка для целых цифр). Новая картинка — рисунок дашборда —
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
from damp.cycle.primary import PrimaryLevel, PrimarySight, StrokeLevel
from damp.cycle.step import CycleStep, LadderLevelDegenerate, TrainedLevel
from damp.decode.level_recognition import LevelRecognition
from damp.decode.linear_reader import LinearReader
from damp.decode.memory_votes import LevelVotes, MemoryVerdict, MemoryVotes
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
    stick_span: "np.ndarray"  # (n,) полудлина поля штриха вдоль края, клеток (§6.4.3) — для показа
    levels: tuple[SightLevel, ...]
    probabilities: "np.ndarray"  # (классов,) вероятности классов линейного читателя (§9.3); нули — ответа нет
    answer: int  # самый вероятный класс читателя; -1 — ни один детектор не сработал
    primary: "PrimarySight | None"  # уровень 0: что первичная карта сделала с углопозициями (None — карты нет)
    votes: tuple[LevelVotes, ...]  # голоса памяти по уровням, до которых дошла картинка (§2.2.5, §7.1.13.3)
    verdict: MemoryVerdict  # ответ памяти: цифра по среднему долей голосов уровней и согласие уровней (§7.1.13.3)
    seconds: float


class TrainedLadder:
    """Обученный цикл (§6.6) и его линейный читатель (§9.3): по ним идёт любая новая картинка 28×28.

    groups — картинки обучения (память строится по ним, ranges["train"] — с видами §7.1.4); levels — обученные уровни
    (TrainedLevel); peaks — пики срабатываний всех картинок прогона по уровням (строки ranges["train"] — обучение
    читателя); decay — штраф L2 читателя. memory — голоса памяти (MemoryVotes) с порогами знания уровней: пороги
    меряются здесь же, по обучающим картинкам без своих точек памяти. primary — уровень 0: первичная карта
    углопозиций (PrimaryLevel, только для показа) либо уровень 0 цикла (StrokeLevel, LADDER_LEVEL0: штрихи рисунка
    получают в нём выученные коды); None — без неё.
    """

    def __init__(self, groups: DigitGroups, levels: list[TrainedLevel], peaks: list["np.ndarray"], *, decay: float,
                 primary: "PrimaryLevel | StrokeLevel | None" = None) -> None:
        if not levels:
            raise ValueError("нет ни одного обученного уровня")
        if len(peaks) != len(levels):
            raise ValueError("пики нужны по каждому обученному уровню")
        self.groups = groups
        self.levels = tuple(levels)
        self.primary = primary
        first, last = groups.ranges["train"]
        features = np.concatenate(peaks, axis=1)
        self.reader = LinearReader(features[first:last], groups.labels[first:last], float(decay))
        self.decay = float(decay)
        self._digits = MnistDigits()
        self._sticks = StickCoder()
        self._coder = KeyholeCoder()
        # Голоса памяти (§2.2.5, §7.1.13.3). Картинка точки памяти — номер MNIST её обучающей картинки: у видов одной
        # цифры (§7.1.4) он общий, и мерка порога знания исключает цифру вместе со всеми её видами.
        origin = np.asarray(groups.atoms.index, dtype=np.int64)
        self.memory = MemoryVotes(
            [np.asarray([int(label) for label in level.memory.layout.code_table.labels()], dtype=np.int64) for level in self.levels],
            [origin[np.asarray(level.point_picture, dtype=np.int64)] for level in self.levels],
            self.reader.classes,
        )
        self.memory = self.memory.calibrated(self._knowledge())

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
                observer=observer, keep_levels=True, keep_top_firing=False,
            )
            runner.run()
            primary = runner.primary  # уровень 0 цикла (LADDER_LEVEL0): построен прогоном, штрихи прошли через него
            if primary is None:
                # Без LADDER_LEVEL0 уровень 0 — первичная карта углопозиций (этапы 2–3) только для показа: в счёте
                # уровней не участвует, поэтому строится после цикла; предел точек — тот же, что у карт уровней.
                try:
                    primary = PrimaryLevel.build(runner.groups, max_points=map_points)
                except LadderLevelDegenerate as stop:
                    primary = None
                    LOGGER.event("primary.map.warning", section=NEURAL_ANALOGY, data={"reason": str(stop)})
        ladder = cls(runner.groups, runner.levels, runner.level_peaks, decay=decay, primary=primary)
        LOGGER.event(
            "dashboard.ladder",
            section=NEURAL_ANALOGY,
            data={
                "train_digits": int(count),
                "train_pictures": int(runner.groups.train_pictures),
                "levels": len(ladder.levels),
                "map_points": [int(level.memory.points) for level in ladder.levels],
                "primary_points": 0 if primary is None else int(primary.memory.points),
                "detectors": [int(level.detectors.built) for level in ladder.levels],
                "decay": float(decay),
                "knowledge_thresholds": list(ladder.memory.thresholds or ()),
                "reader_iterations": int(ladder.reader.iterations),
                "seconds": time.perf_counter() - started,
                "note": "обученный цикл §6.6 для новых картинок: окна, карты памяти и детекторы уровней по обучающим "
                "картинкам с видами (§7.1.4), линейный читатель (§9.3) на их пиках; новая картинка идёт теми же шагами",
            },
        )
        return ladder

    def _knowledge(self) -> "np.ndarray":
        """Знание уровней (картинка × уровень) у обучающих картинок без своих точек памяти — мерка порогов знания.

        Берётся не больше RECOGNITION_KNOWLEDGE_PICTURES обучающих картинок, равномерно по обучающим; каждая идёт
        через уровни заново, а точки памяти с её номером MNIST в голосах не считаются.
        """
        first, last = self.groups.ranges["train"]
        chosen = np.unique(np.linspace(first, last - 1, min(int(settings.RECOGNITION_KNOWLEDGE_PICTURES), last - first)).astype(np.int64))
        found = np.zeros((chosen.shape[0], len(self.levels)), dtype=np.float64)
        atoms = self.groups.atoms
        pictures = chosen.tolist()
        # Картинки идут пачкой по уровням (see_many), штрихи — из набора картинок прогона (atom_sticks), а не заново из
        # пикселов: знание каждой картинки то же до бита, что у прохода по одной.
        sights = self.see_many(
            [atoms.images[picture] for picture in pictures],
            owns=[int(atoms.index[picture]) for picture in pictures],
            sticks=[self.atom_sticks(atoms, picture) for picture in pictures],
        )
        for row, sight in enumerate(sights):
            for votes in sight.votes:
                found[row, votes.number - 1] = votes.knowledge
        return found

    # ------------------------------------------------------------------ новая картинка
    def see(self, image: "np.ndarray", *, log: bool = False, sticks: "np.ndarray | None" = None) -> Sight:
        """Картинка 28×28 (uint8 0…255 либо float в [0, 1], белая цифра на чёрном) через все обученные уровни.

        Каждый уровень: окна на элементах (П.2) → коды окон (П.4–П.7) → срабатывания детекторов карты (§6.5) →
        детали — элементы следующего уровня (П.1). Затем класс: голоса памяти по уровням, их среднее и согласие
        уровней (§2.2.5, §7.1.13.3) — и ответ линейного читателя по пикам уровней (§9.3). Шаги
        считаются заново и в кеш не пишутся; журнал шагов молчит. log — записать итог событием dashboard.see.
        sticks — уже разобранные штрихи этой картинки (atom_sticks картинки набора); None — разбор пикселов.
        """
        return self._see(image, log=log, sticks=sticks)

    def _see(self, image: "np.ndarray", *, own: int = -1, log: bool = False, sticks: "np.ndarray | None" = None) -> Sight:
        """Путь картинки; own — номер MNIST обучающей картинки, чьи точки памяти не голосуют (мерка порога знания);
        sticks — её штрихи из набора картинок (atom_sticks), None — разбор пикселов (MnistDigits.sticks_of).
        Тот же путь, что у пачки (see_many): пачка из одной картинки."""
        return self._see_batch([image], [int(own)], [sticks], log=log)[0]

    def see_many(self, images, *, owns=None, sticks=None, chunk: int = 1) -> "list[Sight]":  # noqa: ANN001
        # chunk = 1 (2026-11-23): пачка из 64 картинок ускорения не дала (сверка 1076/1077: 163–188 мс на картинку против 153–178),
        # а временные тензоры активации выросли в 64 раза — на паре без кеша видеопамять переполнилась (7.5 из 8 ГБ, проход
        # порогов шёл больше 10 мин). По одной картинке — тот же ответ до бита.
        """Картинки 28×28 пачкой (вне статьи, техника счёта): тот же путь, что у see, но уровень за уровнем —
        срабатывания и зажжённые точки окон всех картинок пачки одним вызовом активации (fire_and_light), а не вызовом
        на картинку.

        Ответ каждой картинки тот же до бита, что у see: окна, коды, детали и голоса считаются по картинке, а строка
        срабатываний зависит только от своего кода (активация — целые общие биты; уровни и энергия активации — частями
        ровно по HierarchyDetection.ROWS строк). owns — номера MNIST обучающих картинок, чьи точки памяти не голосуют
        (−1 — проверочная); sticks — готовые штрихи (atom_sticks), None — разбор пикселов; chunk — картинок в пачке
        (память видеокарты под активацию растёт с числом окон)."""
        images = list(images)
        owns = [-1] * len(images) if owns is None else [int(value) for value in owns]
        sticks = [None] * len(images) if sticks is None else list(sticks)
        if not len(owns) == len(sticks) == len(images):
            raise ValueError("owns и sticks нужны по каждой картинке")
        step = max(1, int(chunk))
        found: list[Sight] = []
        for start in range(0, len(images), step):
            found.extend(self._see_batch(images[start : start + step], owns[start : start + step], sticks[start : start + step]))
        return found

    @staticmethod
    def _fire_batch(detectors, codes: "list[np.ndarray]") -> list:  # noqa: ANN001
        """Срабатывания и зажжённые точки кодов окон нескольких картинок одним вызовом; результат — по картинке, как
        у fire_and_light на её кодах: (Firing, (окно, точка, активация)), номера окон — внутри картинки."""
        from damp.activation.level_detectors import Firing

        counts = [int(part.shape[0]) for part in codes]
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        firing, (lit_rows, lit_points, lit_values) = detectors.fire_and_light(np.concatenate(codes))
        # Строки ответа активации идут по возрастанию номера кода: кусок картинки — от её первого окна до следующей.
        fired_at, lit_at = np.searchsorted(firing.rows, offsets), np.searchsorted(lit_rows, offsets)
        parts = []
        for number, count in enumerate(counts):
            a, b, c, d = int(fired_at[number]), int(fired_at[number + 1]), int(lit_at[number]), int(lit_at[number + 1])
            own = Firing(sets=count, rows=firing.rows[a:b] - offsets[number], columns=firing.columns[a:b].copy(),
                         levels=firing.levels[a:b].copy())
            parts.append((own, (lit_rows[c:d] - offsets[number], lit_points[c:d].copy(), lit_values[c:d].copy())))
        return parts

    def _see_batch(self, images: list, owns: list, sticks: list, *, log: bool = False) -> "list[Sight]":
        """Путь пачки картинок по уровням (see, see_many): у каждой картинки — свои окна, коды, детали и голоса."""
        started = time.perf_counter()
        size = len(images)
        pixels = [self._pixels(image) for image in images]
        atoms: list = [None] * size
        elements: list = [None] * size
        levels: list[list[SightLevel]] = [[] for _ in range(size)]
        peaks: list[list["np.ndarray"]] = [[] for _ in range(size)]
        primary: list = [None] * size
        from damp.cache import CACHE

        with CACHE.bypassed(), LOGGER.quiet():
            for k in range(size):
                values = (self._digits.sticks_of(pixels[k]) if sticks[k] is None
                          else np.asarray(sticks[k], dtype=np.float64).reshape(-1, 5))
                atoms[k] = DigitAtoms.from_rows(
                    [np.column_stack([np.zeros(values.shape[0], dtype=np.float64), values])],
                    [-1], [np.clip(np.rint(pixels[k] * 255.0), 0, 255).astype(np.uint8)], [-1],
                )
                elements[k] = self._sticks.encode(atoms[k]) if values.shape[0] else None
            if isinstance(self.primary, StrokeLevel):
                # Уровень 0 цикла (LADDER_LEVEL0): штрихи получают выученные коды — как в прогоне (MnistLadderRunner._run).
                ready = {}
                for k in range(size):
                    if elements[k] is None:
                        continue
                    codes = self.primary.embed_prepare(elements[k], pixels=pixels[k])
                    if codes is None:
                        primary[k] = self.primary.empty_sight() if owns[k] < 0 else None
                    else:
                        ready[k] = codes
                if ready:
                    fired = self._fire_batch(self.primary.detectors, [codes.packed for codes in ready.values()])
                    for k, (firing, lit) in zip(list(ready), fired):
                        seen, elements[k] = self.primary.embed_finish(elements[k], ready[k], firing, lit)
                        primary[k] = seen if owns[k] < 0 else None
            elif self.primary is not None:
                for k in range(size):
                    if owns[k] < 0:
                        # Уровень 0 только для показа: штрихи рисунка на первичной карте (мерке порогов знания он не нужен).
                        primary[k] = self.primary.see(atoms[k].angle, atoms[k].y, atoms[k].x)
            alive = [k for k in range(size) if elements[k] is not None]
            for trained in self.levels:
                prepared = {}
                for k in alive:
                    if elements[k].count == 0:
                        continue  # картинка дальше не идёт: элементов нет
                    # Пятна — как в прогоне (MnistLadderRunner._ladder): они внутри своей картинки.
                    elements[k] = CycleStep.thin(elements[k], trained.number)
                    if elements[k].count == 0:
                        continue
                    found = self._level_prepare(trained, elements[k])
                    if found is not None:
                        prepared[k] = found  # нет окон — картинка дальше не идёт
                alive = list(prepared)
                if not alive:
                    break
                fired = self._fire_batch(trained.detectors, [prepared[k][2].packed for k in alive])
                for k, (firing, lit) in zip(alive, fired):
                    level, elements[k] = self._level_finish(trained, elements[k], *prepared[k], firing, lit)
                    levels[k].append(level)
                    peaks[k].append(level.peak)
        seconds = (time.perf_counter() - started) / max(1, size)
        return [self._sight(atoms[k], levels[k], peaks[k], primary[k], owns[k], seconds, log=log) for k in range(size)]

    def _sight(self, atoms: DigitAtoms, levels: list, peaks: list, primary, own: int, seconds: float, *,  # noqa: ANN001
               log: bool) -> Sight:
        """Ответ по пройденным уровням картинки: пики, читатель (§9.3), голоса памяти и вердикт (§2.2.5, §7.1.13.3)."""
        # Уровни, до которых картинка не дошла, молчат: их пики нулевые, как у такой картинки в прогоне.
        for trained in self.levels[len(peaks):]:
            peaks.append(np.zeros(int(trained.detectors.built), dtype=np.float32))
        features = np.concatenate(peaks)[None, :]
        probabilities = self.reader.probabilities(features)[0]
        answer = int(self.reader.answers(features)[0])
        votes = tuple(
            self.memory.level(
                level.number, int(level.keyhole_centre.shape[0]), level.lit_keyhole, level.lit_point, level.lit_value,
                without_picture=own,
            )
            for level in levels
        )
        verdict = self.memory.verdict(votes)
        sight = Sight(
            stick_angle=np.asarray(atoms.angle, dtype=np.float64),
            stick_y=np.asarray(atoms.y, dtype=np.float64),
            stick_x=np.asarray(atoms.x, dtype=np.float64),
            stick_strength=np.asarray(atoms.strength, dtype=np.float64),
            stick_span=np.asarray(atoms.span, dtype=np.float64),
            levels=tuple(levels),
            primary=primary,
            probabilities=probabilities,
            answer=answer,
            votes=votes,
            verdict=verdict,
            seconds=seconds,
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
                    "memory": {"verdict": verdict.kind, "digit": verdict.digit, "share": verdict.share, "agreed": verdict.agreed},
                    "votes": [
                        {"level": level.number, "digit": level.digit, "share": level.share, "knowledge": level.knowledge}
                        for level in votes
                    ],
                    "answer": answer,
                    "probability": float(probabilities.max()) if answer >= 0 else 0.0,
                    "seconds": sight.seconds,
                    "note": "рисунок дашборда через обученные уровни: окна (П.2), их коды (П.4–П.7), срабатывания "
                    "детекторов карт (§6.5), детали (П.1); класс — среднее долей голосов памяти по уровням и согласие "
                    "уровней (§2.2.5, §7.1.13.3) и ответ линейного читателя (§9.3)",
                },
            )
        return sight

    @staticmethod
    def atom_sticks(atoms: DigitAtoms, picture: int) -> "np.ndarray | None":
        """Штрихи картинки набора в виде MnistDigits.sticks_of: строки (угол, x, y, сила, полудлина поля); None —
        полудлин поля в наборе нет (тогда штрихи разбираются из пикселов).

        Набор хранит те же строки, что дал разбор её пикселов (вне статьи, техника счёта): пикселы картинки набора —
        uint8, и uint8 → float32 / 255 → uint8 идёт без потерь, поэтому разбор картинки заново дал бы те же строки;
        сила в наборе уже нормирована внутри картинки (сильнейший штрих — ровно 1.0), и DigitAtoms.from_rows делит её
        ещё раз на 1.0 — те же биты. Сверено до бита на 400 обучающих с видами и 200 проверочных."""
        if atoms.span is None:
            return None
        rows = np.flatnonzero(np.asarray(atoms.picture) == int(picture))
        return np.column_stack([
            np.asarray(atoms.angle, dtype=np.float64)[rows], np.asarray(atoms.x, dtype=np.float64)[rows],
            np.asarray(atoms.y, dtype=np.float64)[rows], np.asarray(atoms.strength, dtype=np.float64)[rows],
            np.asarray(atoms.span, dtype=np.float64)[rows],
        ]).reshape(-1, 5)

    def _level(self, trained: TrainedLevel, elements: LadderElements) -> tuple[SightLevel | None, LadderElements | None]:
        """Проход обученного уровня над элементами одной картинки — те же шаги, что CycleStep.run, без подбора."""
        prepared = self._level_prepare(trained, elements)
        if prepared is None:
            return None, None
        firing, lit = trained.detectors.fire_and_light(prepared[2].packed)
        return self._level_finish(trained, elements, *prepared, firing, lit)

    def _level_prepare(self, trained: TrainedLevel, elements: LadderElements):  # noqa: ANN202
        """Окна уровня на элементах картинки (П.2) и их коды (П.4–П.7): (окна, размеры окон, коды); None — окон нет."""
        pieces, extent = KeyholeBuilder.place(elements, number=trained.number, step=trained.step, thresholds=trained.thresholds)
        if pieces.count == 0:
            return None
        codes = self._coder.encode(
            elements, pieces, number=trained.number, unit=trained.field.unit, taught=pieces.count,
            content_length=trained.content_length,
        )
        return pieces, extent, codes

    def _level_finish(self, trained: TrainedLevel, elements: LadderElements, pieces, extent, codes,  # noqa: ANN001
                      firing, lit) -> tuple[SightLevel, LadderElements]:
        """Уровень картинки по срабатываниям её окон (каждое окно — набор из одного кода; срабатывания и зажжённые
        точки карты — за один проход активации, своей или пачки): детали (П.1), пик уровня и элементы следующего."""
        lit_keyhole, lit_point, lit_value = lit
        detectors = trained.detectors
        if trained.scale is not None:
            firing = firing.scaled(trained.scale)  # нормировка — делителями обучающих окон (вне статьи)
        coder = DetailCoder(trained.number)
        details, length = coder.encode(
            firing, detectors.layer_index, detectors.built, np.ones(codes.count, dtype=bool), length=trained.detail_length,
            geometry=DetailCoder.geometry(detectors) if coder.needs_geometry() else None,  # предки (v8.2) — как при обучении
        )
        strength = firing.strength()
        _, peak = LevelRecognition.responses(pieces.picture, firing.rows, firing.columns, firing.levels, 1, detectors.built)
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
