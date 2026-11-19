"""Стенд честного сравнения DAMP, kNN по пикселям и обычной CNN на одних и тех же картинках (вне статьи).

Цель проекта — догнать и обогнать обычную CNN «при прочих равных». «Прочие равные»:

- картинки — случайная выборка номеров MNIST с закреплённым зерном (BenchSample): обучающие наборы — из пула
  BENCH_TRAIN_POOL, не пересекаются между собой; отложенная выборка validation — из пула BENCH_VALIDATION_POOL;
  проверочные test — из всего t10k. Не сплошные блоки: MNIST train не перемешан по авторам, и соседние блоки по 250
  картинок заметно разной трудности. Одни и те же картинки у всех методов;
- настройки каждого метода (k у DAMP и kNN, конфигурация и ранняя остановка CNN) выбираются **только по средней
  точности на части select отложенной выборки**, итог — по части report, в выборе не участвовавшей;
- проверочные картинки test (MNIST t10k) считаются лишь для выбранных настроек и ни на что не влияют;
- итог по методу — среднее, стандартное отклонение и 95 % доверительный интервал (t-распределение) по наборам.
"""

from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from damp import __version__, settings
from damp.article_refs import OUTSIDE_ARTICLE
from damp.decode.baselines import ConvNetBaseline, PixelNearest
from damp.decode.level_recognition import LevelRecognition
from damp.decode.linear_reader import LinearReader
from damp.logging import LOGGER
from damp.stimuli.digits import DigitAtoms, MnistDigits

METHODS = ("damp", "pixels", "cnn")
AUGMENTS = ("none", "shifts")
_NOT_SUPPORTED = {
    "damp": "не отдельный режим: сдвинутые виды обучающих DAMP задаёт LADDER_TRAIN_VIEWS (§7.1.4), строка DAMP одна",
    "pixels": "не поддержано: kNN по пикселям без аугментации",
}


class Summary:
    """Итог по наборам: среднее, стандартное отклонение (n − 1) и 95 % доверительный интервал (t-распределение)."""

    @staticmethod
    def of(values: Sequence[float]) -> dict[str, object] | None:
        """Сводка чисел по наборам; пусто — None. Одного набора на интервал не хватает: std = 0, интервал — None."""
        numbers = [float(value) for value in values]
        if not numbers:
            return None
        count = len(numbers)
        mean = float(np.mean(numbers))
        std = float(np.std(numbers, ddof=1)) if count > 1 else 0.0
        low = high = None
        if count > 1:
            from scipy import stats

            half = float(stats.t.ppf(0.975, count - 1)) * std / math.sqrt(count)
            low, high = mean - half, mean + half
        return {"mean": mean, "std": std, "ci_low": low, "ci_high": high, "n": count, "per_set": numbers}

    @staticmethod
    def paired(first: Sequence[float], second: Sequence[float]) -> dict[str, object] | None:
        """Сводка разности first - second по наборам (парный t-интервал: методы идут на одних наборах и картинках).

        Длины должны совпасть, иначе None.
        """
        if len(first) != len(second):
            return None
        return Summary.of([float(a) - float(b) for a, b in zip(first, second)])


@dataclass
class Score:
    """Результаты одной настройки метода по наборам (список — по одному числу на набор)."""

    select: list[float] = field(default_factory=list)  # точность на части select: по ней выбирают настройку
    validation: list[float] = field(default_factory=list)  # точность на части report: в выборе не участвовала
    test: list[float | None] = field(default_factory=list)
    plain: list[float] = field(default_factory=list)  # точность на первых картинках validation без сдвига
    shifted: list[float] = field(default_factory=list)  # точность на тех же картинках, сдвинутых
    same_answers: list[float] = field(default_factory=list)  # доля картинок, чей ответ при сдвиге не изменился


@dataclass(frozen=True)
class BenchSample:
    """Номера картинок MNIST стенда (вне статьи): обучающие наборы, отложенная выборка, проверочные.

    Случайная выборка с закреплённым зерном BENCH_SEED из своих пулов, а не сплошные блоки: MNIST train не
    перемешан по авторам, соседние блоки по 250 картинок заметно разной трудности (1-NN по пикселям — от 80 до
    94 %), и сплошной блок давал нечестную мерку. Наборы не пересекаются между собой и с отложенной выборкой.
    Выборка «префиксом»: отложенные — начало одной перестановки пула (500 — начало выборки из 2000), набор s —
    s-й кусок перестановки обучающего пула (тот же при любом числе наборов), проверочные — начало перестановки t10k.
    """

    sets: tuple[tuple[int, ...], ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]

    TEST_POOL = 10_000  # картинок в MNIST t10k
    TRAIN_SIZE = 60_000  # картинок в MNIST train

    @classmethod
    def draw(cls, *, train: int, sets: int, validation: int, test: int) -> "BenchSample":
        low, high = (int(value) for value in settings.BENCH_TRAIN_POOL)
        first, last = (int(value) for value in settings.BENCH_VALIDATION_POOL)
        if not (0 <= low < high <= cls.TRAIN_SIZE and 0 <= first < last <= cls.TRAIN_SIZE):
            raise ValueError("BENCH_TRAIN_POOL и BENCH_VALIDATION_POOL — [начало, конец) номеров MNIST train")
        if max(low, first) < min(high, last):
            raise ValueError("пулы обучающих (BENCH_TRAIN_POOL) и отложенных (BENCH_VALIDATION_POOL) пересекаются")
        if int(train) * int(sets) > high - low:
            raise ValueError(f"в пуле обучающих {high - low} картинок, а нужно {int(train) * int(sets)}")
        if int(validation) > last - first:
            raise ValueError(f"в пуле отложенных {last - first} картинок, а нужно {int(validation)}")
        if int(test) > cls.TEST_POOL:
            raise ValueError(f"в MNIST t10k {cls.TEST_POOL} картинок, а нужно {int(test)}")
        rng = np.random.default_rng(int(settings.BENCH_SEED))
        held = rng.permutation(np.arange(first, last))[: int(validation)]
        taught = rng.permutation(np.arange(low, high))[: int(train) * int(sets)].reshape(int(sets), int(train))
        checked = rng.permutation(cls.TEST_POOL)[: int(test)]
        return cls(
            sets=tuple(tuple(int(value) for value in row) for row in taught),
            validation=tuple(int(value) for value in held),
            test=tuple(int(value) for value in checked),
        )


@dataclass(frozen=True)
class BenchSet:
    """Картинки одного набора: обучающие и общие для всех наборов отложенные, проверочные и сдвинутые."""

    number: int
    train: DigitAtoms
    validation: DigitAtoms
    test: DigitAtoms | None
    shifted: DigitAtoms | None  # первые картинки validation, сдвинутые на BENCH_PIXEL_SHIFT пикселей


class HonestBench:
    """Стенд честного сравнения методов на MNIST (вне статьи). Методы: "damp", "pixels", "cnn".

    Параметры по умолчанию — из раздела [bench] файла настроек (читаются в момент вызова). Итог — словарь той же формы,
    что файл data/bench/bench-<время>.json; сам файл пишет run().
    """

    def __init__(
        self,
        *,
        train: int | None = None,
        sets: int | None = None,
        validation: int | None = None,
        test: int | None = None,
        map_points: int | None = None,
        methods: Sequence[str] | None = None,
        augment: Sequence[str] | None = None,
        save: bool = True,
    ) -> None:
        self.train = int(settings.BENCH_TRAIN if train is None else train)
        self.sets = int(settings.BENCH_SETS if sets is None else sets)
        self.validation = int(settings.BENCH_VALIDATION if validation is None else validation)
        self.test = int(settings.BENCH_TEST if test is None else test)
        self.map_points = int(settings.BENCH_MAP_POINTS if map_points is None else map_points)
        self.methods = tuple(settings.BENCH_METHODS if methods is None else methods)
        # В файле настроек строка "none" читается как None (соглашение файла): здесь это режим «без аугментации».
        modes = settings.BENCH_AUGMENT if augment is None else augment
        self.augment = tuple("none" if mode is None else str(mode) for mode in modes)
        self.save = bool(save)
        if self.train <= 0 or self.sets <= 0:
            raise ValueError("train и sets должны быть положительными")
        if self.validation <= 0:
            raise ValueError("нужна отложенная выборка: validation > 0 (по ней выбираются настройки)")
        if self.test < 0:
            raise ValueError("test не может быть отрицательным")
        unknown = [name for name in self.methods if name not in METHODS]
        if unknown or not self.methods:
            raise ValueError(f"методы стенда — из {METHODS}, а не {unknown or list(self.methods)}")
        unknown = [name for name in self.augment if name not in AUGMENTS]
        if unknown or not self.augment:
            raise ValueError(f"аугментация — из {AUGMENTS}, а не {unknown or list(self.augment)}")
        self.neighbours = tuple(int(k) for k in settings.BENCH_NEIGHBOURS)
        self.pictures = min(int(settings.BENCH_PIXEL_PICTURES), self.validation)
        shift = settings.BENCH_PIXEL_SHIFT
        self.shift = (int(shift[0]), int(shift[1]))
        if self.validation < 2:
            raise ValueError("validation < 2: отложенную выборку нужно разделить на select и report")
        # select — первые картинки отложенной выборки (выбор настройки, ранняя остановка CNN), report — остальные.
        self.select_count = min(self.validation - 1, max(1, int(round(self.validation * float(settings.BENCH_SELECT_SHARE)))))
        self.report_count = self.validation - self.select_count
        self._digits = MnistDigits()
        self._scores: dict[tuple[str, str, str], Score] = {}
        self._levels: list[list[dict[str, object]]] = []  # по наборам: уровни DAMP
        self._sets_done = 0
        self._seconds = {"damp": 0.0, "pixels": 0.0, "cnn": 0.0}
        self._numbers: dict[str, object] = {"train": [], "validation": [], "test": []}
        # Картинки без углопозиций (у DAMP — без ответа, то есть неверные; kNN и CNN видят их как обычные).
        self._strokeless: dict[str, object] = {"train": [], "validation": 0, "test": 0}
        self.sample: BenchSample | None = None
        self._view = None
        self._held: DigitAtoms | None = None
        self._checked: DigitAtoms | None = None
        self._moved: DigitAtoms | None = None

    # ------------------------------------------------------------------ данные
    def _load_common(self) -> None:
        """Выборка номеров стенда; отложенная выборка, проверочные и сдвинутые отложенные — общие для всех наборов."""
        digits = self._digits
        self.sample = BenchSample.draw(train=self.train, sets=self.sets, validation=self.validation, test=self.test)
        self._held = digits.load(train=True, count=self.validation, numbers=self.sample.validation)
        self._checked = digits.load(train=False, count=self.test, numbers=self.sample.test) if self.test else None
        if self.pictures:
            self._moved = digits.load(
                train=True, shift=self.shift, count=self.pictures, numbers=self._held.index[: self.pictures]
            )
        self._numbers["validation"] = self._held.index.tolist()
        self._numbers["test"] = [] if self._checked is None else self._checked.index.tolist()
        self._strokeless["validation"] = self._silent(self._held)
        self._strokeless["test"] = 0 if self._checked is None else self._silent(self._checked)

    @staticmethod
    def _silent(atoms: DigitAtoms) -> int:
        """Сколько картинок группы без углопозиций: DAMP их не видит и ответа не даёт."""
        return int(atoms.pictures - np.unique(atoms.picture).shape[0])

    def _load_set(self, number: int) -> BenchSet:
        """Обучающий набор number из выборки стенда; с отложенной выборкой не пересекается."""
        taught = self._digits.load(train=True, count=self.train, numbers=self.sample.sets[number])
        assert not set(taught.index.tolist()) & set(self._held.index.tolist()), "набор пересекается с отложенной выборкой"
        self._numbers["train"].append(taught.index.tolist())
        self._strokeless["train"].append(self._silent(taught))
        return BenchSet(number, taught, self._held, self._checked, self._moved)

    # ------------------------------------------------------------------ учёт результатов
    @staticmethod
    def mean(values: Sequence[float]) -> float:
        """Среднее для выбора настройки: округлено до 1e-9, чтобы шум float (порядок сложения) не решал ничью —
        при равенстве выигрывает первая настройка по порядку."""
        return round(float(np.mean(values)), 9)

    def _score(self, method: str, augment: str, setting: str) -> Score:
        return self._scores.setdefault((method, augment, setting), Score())

    @staticmethod
    def _accuracy(answers: np.ndarray, truth: np.ndarray) -> float:
        """Доля верных ответов; ответ -1 (нет голосов) — неверный."""
        return float(np.mean((answers == truth) & (answers >= 0))) if truth.shape[0] else 0.0

    def _robust(self, score: Score, plain: np.ndarray, moved: np.ndarray, truth: np.ndarray) -> None:
        """Устойчивость к сдвигу одной настройки: точность до и после и доля совпавших ответов (validation)."""
        score.plain.append(self._accuracy(plain, truth))
        score.shifted.append(self._accuracy(moved, truth))
        score.same_answers.append(float(np.mean(plain == moved)) if truth.shape[0] else 0.0)

    # ------------------------------------------------------------------ DAMP
    def _damp(self, data: BenchSet) -> None:
        """DAMP на наборе: цикл уровней, голосование верхних деталей для каждого k, диагностика уровней."""
        from damp.cycle.ladder import MnistLadderRunner

        started = time.perf_counter()
        pictures = self.pictures
        runner = MnistLadderRunner(
            count=self.train,
            map_max_points=self.map_points,
            shifts=(),
            pixel_shift=self.shift if pictures else None,
            pixel_pictures=pictures,
            train_numbers=data.train.index,
            validation_numbers=data.validation.index,
            test_numbers=None if data.test is None else data.test.index,
            evaluate_limit=0 if data.test is None else data.test.pictures,
        )
        runner.run()
        groups, details = runner.groups, runner.top_details
        recognition = LevelRecognition(groups.labels, groups.ranges["train"], neighbours=self.neighbours)
        codes, length, owner = details.codes(), details.what_length, details.picture
        held = groups.ranges["validation"]
        chosen, rest = self._parts(held)
        by_select = recognition.by_vote(codes, length, owner, chosen)
        by_vote = recognition.by_vote(codes, length, owner, rest)
        by_test = recognition.by_vote(codes, length, owner, groups.ranges["test"]) if "test" in groups.ranges else None
        plain = moved = None
        pixel_key = ("pixel", *self.shift)
        if pictures and pixel_key in groups.ranges:
            plain = recognition.answers(codes, length, owner, (held[0], held[0] + pictures))
            moved = recognition.answers(codes, length, owner, groups.ranges[pixel_key])
        truth = groups.labels[held[0] : held[0] + pictures]
        for k in self.neighbours:
            score = self._score("damp", "none", f"k={k}")
            score.select.append(by_select[k])
            score.validation.append(by_vote[k])
            score.test.append(None if by_test is None else by_test[k])
            if plain is not None:
                self._robust(score, plain[k], moved[k], truth)
        self._damp_within(recognition, codes, length, owner, groups, chosen, rest, truth)
        if settings.BENCH_LEVEL_EMBEDDING and runner.top_firing is not None:
            self._damp_levels(recognition, runner.top_firing, groups, chosen, rest, truth)
        if settings.BENCH_LINEAR_DECAYS and runner.level_peaks:
            self._damp_linear(runner.level_peaks, groups, chosen, rest, truth)
        self._levels.append(self._level_measures(runner, recognition, plain is not None))
        self._seconds["damp"] += time.perf_counter() - started
        del runner, groups, details
        self._free_gpu()

    def _damp_within(self, recognition: LevelRecognition, codes, length, owner, groups, chosen, rest, truth) -> None:  # noqa: ANN001
        """Кандидаты нечёткого поиска §2.2.5 (BENCH_EPSILONS): деталь голосует всеми деталями обучающих в своей
        ε-окрестности; выбор ε — по select вместе с k, итог — по report."""
        levels = tuple(float(value) for value in settings.BENCH_EPSILONS)
        if not levels:
            return
        labels = groups.labels
        parts = {"select": chosen, "report": rest}
        if "test" in groups.ranges:
            parts["test"] = groups.ranges["test"]
        pixel_key = ("pixel", *self.shift)
        if self.pictures and pixel_key in groups.ranges:
            held = groups.ranges["validation"]
            parts["plain"] = (held[0], held[0] + self.pictures)
            parts["moved"] = groups.ranges[pixel_key]
        found = {name: recognition.answers_within(codes, length, owner, part, levels) for name, part in parts.items()}
        for level in levels:
            score = self._score("damp", "none", f"ε={level:g}")
            score.select.append(self._accuracy(found["select"][level], labels[chosen[0] : chosen[1]]))
            score.validation.append(self._accuracy(found["report"][level], labels[rest[0] : rest[1]]))
            test = parts.get("test")
            score.test.append(None if test is None else self._accuracy(found["test"][level], labels[test[0] : test[1]]))
            if "plain" in found:
                self._robust(score, found["plain"][level], found["moved"][level], truth)

    def _damp_levels(self, recognition: LevelRecognition, firing: dict, groups, chosen, rest, truth) -> None:  # noqa: ANN001
        """Кандидаты «уровни k=…» (§7.1.12): окно верхнего уровня — вектор уровней активности детекторов в [0, 1],
        сходство — косинус; выбор k — по select вместе с остальными кандидатами, итог — по report."""
        labels = groups.labels
        parts = {"select": chosen, "report": rest}
        if "test" in groups.ranges:
            parts["test"] = groups.ranges["test"]
        pixel_key = ("pixel", *self.shift)
        if self.pictures and pixel_key in groups.ranges:
            held = groups.ranges["validation"]
            parts["plain"] = (held[0], held[0] + self.pictures)
            parts["moved"] = groups.ranges[pixel_key]
        if firing.get("dominance") is not None:
            # Господство классов в детекторах (§8.2): одна настройка, без k.
            found = {name: recognition.answers_dominance(firing, part) for name, part in parts.items()}
            score = self._score("damp", "none", "господство")
            score.select.append(self._accuracy(found["select"], labels[chosen[0] : chosen[1]]))
            score.validation.append(self._accuracy(found["report"], labels[rest[0] : rest[1]]))
            test = parts.get("test")
            score.test.append(None if test is None else self._accuracy(found["test"], labels[test[0] : test[1]]))
            if "plain" in found:
                self._robust(score, found["plain"], found["moved"], truth)
        titles = {"clip": "уровни", "train_max": "уровни/max"}
        for scale in settings.BENCH_LEVEL_SCALES:
            title = titles[str(scale)]
            found = {name: recognition.answers_levels(firing, part, scale=scale) for name, part in parts.items()}
            for k in self.neighbours:
                score = self._score("damp", "none", f"{title} k={k}")
                score.select.append(self._accuracy(found["select"][k], labels[chosen[0] : chosen[1]]))
                score.validation.append(self._accuracy(found["report"][k], labels[rest[0] : rest[1]]))
                test = parts.get("test")
                score.test.append(None if test is None else self._accuracy(found["test"][k], labels[test[0] : test[1]]))
                if "plain" in found:
                    self._robust(score, found["plain"][k], found["moved"][k], truth)

    def _damp_linear(self, peaks: list, groups, chosen, rest, truth) -> None:  # noqa: ANN001
        """Кандидаты «линейный λ=…» (§9.3): эмбеддинг картинки — самые сильные уровни активации детекторов по её
        окнам на всех уровнях цикла (у каждого уровня свой сегмент единичной длины), класс читает один линейный слой
        с softmax, обученный на обучающих картинках; выбор λ — по select вместе с остальными кандидатами."""
        labels = groups.labels
        parts = {"select": chosen, "report": rest}
        if "test" in groups.ranges:
            parts["test"] = groups.ranges["test"]
        pixel_key = ("pixel", *self.shift)
        if self.pictures and pixel_key in groups.ranges:
            held = groups.ranges["validation"]
            parts["plain"] = (held[0], held[0] + self.pictures)
            parts["moved"] = groups.ranges[pixel_key]
        features = np.concatenate(peaks, axis=1)
        found = LinearReader.answers_for(features, labels, groups.ranges["train"], parts, settings.BENCH_LINEAR_DECAYS)
        for decay, answers in found.items():
            score = self._score("damp", "none", f"линейный λ={decay:g}")
            score.select.append(self._accuracy(answers["select"], labels[chosen[0] : chosen[1]]))
            score.validation.append(self._accuracy(answers["report"], labels[rest[0] : rest[1]]))
            test = parts.get("test")
            score.test.append(None if test is None else self._accuracy(answers["test"], labels[test[0] : test[1]]))
            if "plain" in answers:
                self._robust(score, answers["plain"], answers["moved"], truth)

    def _parts(self, held: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        """Картинки отложенной выборки, занимающей held = (первая, после последней): (select, report)."""
        middle = held[0] + self.select_count
        return (held[0], middle), (middle, held[1])

    def _level_measures(self, runner, recognition: LevelRecognition, shifted: bool) -> list[dict[str, object]]:  # noqa: ANN001
        """Мерки статьи по уровням (из отчётов цикла) и диагностика: узнавание по отклику уровня и склейке на
        validation, косинус откликов картинки и её сдвига."""
        groups = runner.groups
        held = groups.ranges["validation"]
        pictures = self.pictures
        responses = runner.level_responses
        levels: list[dict[str, object]] = []
        for index, report in enumerate(runner.reports):
            rest = self._parts(held)[1]  # диагностика уровней - на части, не участвовавшей в выборе
            alone = recognition.by_response(responses[index], rest)
            glued = recognition.by_response(np.concatenate(responses[: index + 1], axis=1), rest)
            cosine = None
            pixel_key = ("pixel", *self.shift)
            if shifted and pixel_key in groups.ranges:
                own = responses[index][held[0] : held[0] + pictures]
                other = responses[index][groups.ranges[pixel_key][0] : groups.ranges[pixel_key][1]]
                silent = ~own.any(axis=1) & ~other.any(axis=1)  # два пустых отклика совпадают
                cosine = float(np.mean(np.where(silent, 1.0, np.sum(own * other, axis=1))))
            levels.append(
                {
                    "number": int(report.number),
                    "picture_seen": float(report.picture_seen),
                    "field_radius": float(report.field_radius),
                    "map_points": int(report.map_points),
                    "detectors": int(report.detectors),
                    "same_label": float(report.same_label),
                    "energy_mean": float(report.energy_mean),
                    "code_resolution": float(report.code_resolution),
                    "code_largest": int(report.code_largest),
                    "detail_bits": float(report.detail_bits),
                    "detail_empty": float(report.detail_empty),
                    "detail_full": float(report.detail_full),
                    "alone": {int(k): float(v) for k, v in alone.items()},
                    "glued": {int(k): float(v) for k, v in glued.items()},
                    "response_cosine": cosine,
                }
            )
        return levels

    # ------------------------------------------------------------------ kNN по пикселям
    def _pixels(self, data: BenchSet) -> None:
        started = time.perf_counter()
        model = PixelNearest(data.train.images, data.train.labels)
        top = max(self.neighbours)
        held = model.order(data.validation.images, top)
        chosen = self.select_count
        asked = None if data.test is None else model.order(data.test.images, top)
        pictures = self.pictures
        plain = moved = None
        if pictures and data.shifted is not None:
            plain, moved = held[:pictures], model.order(data.shifted.images, top)
        truth = data.validation.labels[:pictures]
        for k in self.neighbours:
            score = self._score("pixels", "none", f"k={k}")
            answers, labels = model.vote(held, k), data.validation.labels
            score.select.append(self._accuracy(answers[:chosen], labels[:chosen]))
            score.validation.append(self._accuracy(answers[chosen:], labels[chosen:]))
            score.test.append(None if asked is None else self._accuracy(model.vote(asked, k), data.test.labels))
            if plain is not None:
                self._robust(score, model.vote(plain, k), model.vote(moved, k), truth)
        self._seconds["pixels"] += time.perf_counter() - started

    # ------------------------------------------------------------------ CNN
    @staticmethod
    def cnn_configs() -> list[dict[str, object]]:
        """Перебор CNN: архитектура × lr × weight decay × dropout (без режима аугментации)."""
        return [
            {"arch": arch, "lr": lr, "weight_decay": decay, "dropout": dropout}
            for arch in settings.BENCH_CNN_ARCHS
            for lr in settings.BENCH_CNN_LR
            for decay in settings.BENCH_CNN_WEIGHT_DECAY
            for dropout in settings.BENCH_CNN_DROPOUT
        ]

    def _cnn(self, data: BenchSet) -> None:
        started = time.perf_counter()
        pictures = self.pictures
        truth = data.validation.labels[:pictures]
        for augment in self.augment:
            for config in self.cnn_configs():
                network = ConvNetBaseline(augment=augment, seed=data.number, **config)
                chosen = self.select_count
                # Ранняя остановка и выбор настроек - по select; итоговая точность - по report (в остановке не участвовала).
                result = network.fit(
                    data.train.images, data.train.labels, data.validation.images[:chosen], data.validation.labels[:chosen],
                    dataset=f"set{data.number}",
                )
                score = self._score("cnn", augment, network.name)
                score.select.append(float(result.validation_accuracy))
                score.validation.append(network.accuracy(data.validation.images[chosen:], data.validation.labels[chosen:]))
                score.test.append(None if data.test is None else network.accuracy(data.test.images, data.test.labels))
                if pictures and data.shifted is not None:
                    plain = network.predict(data.validation.images[:pictures])
                    self._robust(score, plain, network.predict(data.shifted.images), truth)
                del network
        self._seconds["cnn"] += time.perf_counter() - started
        self._free_gpu()

    @staticmethod
    def _free_gpu() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # память видеокарты — забота не стенда
            pass

    # ------------------------------------------------------------------ выбор и итог
    def _row(self, method: str, augment: str) -> dict[str, object]:
        """Строка итога: настройка выбрана по средней точности на части select (ничья - первая по порядку); итоговая
        validation - на части report, которая в выборе не участвовала."""
        row: dict[str, object] = {
            "method": method, "augment": augment, "supported": True, "setting": None, "candidates": {},
            "validation": None, "test": None, "robustness": None, "note": "",
        }
        if augment != "none" and method in _NOT_SUPPORTED:
            row.update(supported=False, note=_NOT_SUPPORTED[method])
            return row
        found = [(setting, score) for (name, mode, setting), score in self._scores.items() if name == method and mode == augment]
        if not found or not found[0][1].select:
            return row
        best_setting, best, top = None, None, None
        for setting, score in found:
            mean = self.mean(score.select)
            if best is None or mean > top:
                best_setting, best, top = setting, score, mean
        row["setting"] = best_setting
        row["candidates"] = {setting: float(np.mean(score.select)) for setting, score in found}
        # select и report по наборам у всех кандидатов — для разбора; test — только у выбранной (не подглядывать).
        row["scores"] = {
            setting: {"select": [float(v) for v in score.select], "report": [float(v) for v in score.validation]}
            for setting, score in found
        }
        row["validation"] = Summary.of(best.validation)
        # Ошибка самой конечной проверочной выборки (общей для наборов): биномиальная, по части report.
        mean = float(np.mean(best.validation))
        row["validation_sampling_se"] = math.sqrt(max(mean * (1.0 - mean), 0.0) / self.report_count)
        if best.test and all(value is not None for value in best.test):
            row["test"] = Summary.of(best.test)  # test — только у выбранной настройки
        if best.same_answers:
            row["robustness"] = {
                "pictures": self.pictures,
                "shift": list(self.shift),
                "same_answers": float(np.mean(best.same_answers)),
                "accuracy_plain": float(np.mean(best.plain)),
                "accuracy_shifted": float(np.mean(best.shifted)),
                "same_answers_per_set": [float(v) for v in best.same_answers],
            }
        return row

    def _aggregate_levels(self) -> list[dict[str, object]]:
        """Уровни DAMP: числа, усреднённые по наборам, у которых такой уровень есть."""
        merged: list[dict[str, object]] = []
        numbers = sorted({int(level["number"]) for levels in self._levels for level in levels})
        for number in numbers:
            own = [level for levels in self._levels for level in levels if level["number"] == number]
            item: dict[str, object] = {"number": number, "sets": len(own)}
            for key in own[0]:
                if key == "number":
                    continue
                values = [level[key] for level in own]
                if isinstance(values[0], dict):
                    item[key] = {k: float(np.mean([v[k] for v in values])) for k in values[0]}
                elif values[0] is None:
                    item[key] = None
                else:
                    kept = [v for v in values if v is not None]
                    item[key] = float(np.mean(kept)) if kept else None
            merged.append(item)
        return merged

    @staticmethod
    def _gap(rows: list[dict[str, object]]) -> dict[str, object] | None:
        """Сколько DAMP до CNN: разность средних (доли) DAMP без аугментации и CNN без аугментации и лучшей CNN."""
        damp = next((row for row in rows if row["method"] == "damp" and row["augment"] == "none"), None)
        nets = [row for row in rows if row["method"] == "cnn" and row["validation"] is not None]
        if damp is None or damp["validation"] is None or not nets:
            return None

        def minus(other: dict[str, object]) -> dict[str, float | None]:
            return {
                part: (None if damp[part] is None or other[part] is None else damp[part]["mean"] - other[part]["mean"])
                for part in ("validation", "test")
            }

        def interval(other: dict[str, object]) -> dict[str, object]:
            """Парный интервал разности по наборам (validation и test): методы идут на одних наборах и картинках."""
            return {
                part: (
                    None if damp[part] is None or other[part] is None
                    else Summary.paired(damp[part]["per_set"], other[part]["per_set"])
                )
                for part in ("validation", "test")
            }

        same = next((row for row in nets if row["augment"] == "none"), None)
        # Лучшая CNN (режим аугментации) — по средней select её выбранной настройки, как и всё остальное: report
        # не участвует ни в каком выборе (при равенстве — первая по порядку строк).
        best = max(nets, key=lambda row: HonestBench.mean([row["candidates"][row["setting"]]]))
        return {
            "damp_minus_cnn_none": None if same is None else minus(same),
            "damp_minus_best_cnn": minus(best),
            "best_cnn": f"{best['augment']}: {best['setting']}",
            "intervals": {
                "damp_minus_cnn_none": None if same is None else interval(same),
                "damp_minus_best_cnn": interval(best),
                "note": "парный t-интервал разности по наборам; validation (report) общая для наборов, её конечный "
                "размер в интервал не входит (см. validation_sampling_se в строках методов)",
            },
        }

    @staticmethod
    def _views() -> tuple[tuple[int, int], ...]:
        """Сдвинутые виды обучающих цифр DAMP (LADDER_TRAIN_VIEWS, §7.1.4) без нулевого сдвига."""
        return tuple((int(dy), int(dx)) for dy, dx in settings.LADDER_TRAIN_VIEWS if (int(dy), int(dx)) != (0, 0))

    def result(self) -> dict[str, object]:
        """Итог стенда по уже посчитанным наборам: строки методов, уровни DAMP, разрыв до CNN, протокол, настройки."""
        rows = [self._row(method, augment) for method in self.methods for augment in self.augment]
        return {
            "version": __version__,
            "created": datetime.now().isoformat(timespec="seconds"),
            "sets_done": self._sets_done,
            "protocol": {
                "train": self.train,
                "sets": self.sets,
                "validation": self.validation,
                "sampling": "случайная выборка номеров MNIST с закреплённым зерном из пулов, не сплошные блоки: "
                "MNIST train не перемешан по авторам",
                "seed": int(settings.BENCH_SEED),
                "train_pool": [int(value) for value in settings.BENCH_TRAIN_POOL],
                "validation_pool": [int(value) for value in settings.BENCH_VALIDATION_POOL],
                "test": self.test,
                "map_points": self.map_points,
                "methods": list(self.methods),
                "augment": list(self.augment),
                "neighbours": list(self.neighbours),
                "pixel_shift": list(self.shift),
                "pixel_pictures": self.pictures,
                "validation_select": self.select_count,
                "validation_report": self.report_count,
                "selection": "настройка и ранняя остановка CNN - по средней точности на select (первые картинки "
                "validation); итоговая validation и разрыв до CNN - на report (остальные), в выборе не участвовавшей; "
                "test - только у выбранных настроек",
                "validation_shared": "validation и test общие для всех наборов: интервал по наборам не включает "
                "ошибку конечного размера выборки",
                "strokeless": self._strokeless,
                "strokeless_note": "картинки без углопозиций не пропускаются: у DAMP они без ответа (неверные), kNN "
                "и CNN видят их как обычные",
                "damp_train_views": [list(view) for view in self._views()],
                "damp_train_pictures": self.train * (1 + len(self._views())),
                "damp_views_note": "DAMP учится на обучающих цифрах и их сдвинутых видах (LADDER_TRAIN_VIEWS, §7.1.4) — "
                "это аугментация сдвигами: честная пара для такого DAMP — CNN со сдвигами (shifts), а не без аугментации"
                if self._views() else "DAMP учится только на обучающих цифрах, без видов",
                "numbers": self._numbers,
            },
            "methods": rows,
            "levels": self._aggregate_levels(),
            "gap": self._gap(rows),
            "seconds": {name: round(value, 1) for name, value in self._seconds.items()},
            "settings": {name: getattr(settings, name) for name in sorted(settings.KEYS)},
        }

    # ------------------------------------------------------------------ Rerun и журнал
    def _open_view(self) -> None:
        """Новая запись Rerun «bench» с раскладкой стенда (после прогона цикла окно возвращается к ней)."""
        if self._view is not None:
            self._view.open()

    def _show(self, final: bool = False) -> None:
        if self._view is not None:
            self._view.show(self.result(), final=final)

    def _log_set(self, data: BenchSet, seconds: float) -> None:
        best: dict[str, object] = {}
        for method in self.methods:
            for augment in self.augment:
                row = self._row(method, augment)
                if row["validation"] is not None:
                    best[f"{method}/{augment}"] = {
                        "setting": row["setting"], "validation_last_set": row["validation"]["per_set"][-1],
                    }
        LOGGER.event(
            "bench.set",
            section=OUTSIDE_ARTICLE,
            data={
                "set": data.number, "sets": self.sets,
                "train_numbers_first": [int(value) for value in data.train.index[:5]],
                "seconds": round(seconds, 1), "best_by_validation": best,
                "note": "итог набора: выбранные по средней select настройки пока предварительные",
            },
        )

    @staticmethod
    def _plain(value: object) -> object:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (set, tuple)):
            return list(value)
        return str(value)

    def _write(self, result: dict[str, object]) -> str:
        folder = Path(str(settings.BENCH_OUTPUT_DIR))
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"bench-{datetime.now():%Y%m%d-%H%M%S}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=self._plain), encoding="utf-8")
        return str(path)

    # ------------------------------------------------------------------ запуск
    def run(self) -> dict[str, object]:
        """Все наборы по порядку: DAMP, kNN по пикселям, CNN; выбор настроек по validation; итог в JSON."""
        started = time.perf_counter()
        plain = "none" in self.augment  # без режима none цифры DAMP и kNN нигде не показываются: не считаем
        for name in ("damp", "pixels"):
            if name in self.methods and any(mode != "none" for mode in self.augment):
                LOGGER.event(
                    f"bench.{name}.augment.warning",
                    section=OUTSIDE_ARTICLE,
                    data={"augment": "shifts", "calculated": plain,
                          "note": _NOT_SUPPORTED[name] + ("" if plain else "; расчёт пропущен: режима none нет")},
                )
        LOGGER.event(
            "bench.start",
            section=OUTSIDE_ARTICLE,
            data={
                "train": self.train, "sets": self.sets, "validation": self.validation, "test": self.test,
                "map_points": self.map_points, "methods": list(self.methods), "augment": list(self.augment),
                "neighbours": list(self.neighbours), "pixel_shift": list(self.shift), "pixel_pictures": self.pictures,
                "cnn_configs": len(self.cnn_configs()),
                "note": "стенд честного сравнения: настройки выбираются по select (часть validation), итог - по report, test - только у выбранных",
            },
        )
        if LOGGER.visuals_enabled():
            from damp.figures.rerun_bench import BenchRerunView

            self._view = BenchRerunView()
            self._view.open()
        self._load_common()
        self._show()
        for number in range(self.sets):
            began = time.perf_counter()
            data = self._load_set(number)
            if "damp" in self.methods and plain:
                self._damp(data)
                self._open_view()  # прогон цикла шёл в своей записи Rerun: стенд возвращается в свою
                self._show()  # статичные таблица и столбцы в новую запись не переносятся: рисуем заново
            if "pixels" in self.methods and plain:
                self._pixels(data)
            if "cnn" in self.methods:
                self._cnn(data)
            self._sets_done = number + 1
            self._log_set(data, time.perf_counter() - began)
            self._show(final=self._sets_done == self.sets)
        result = self.result()
        result["seconds"]["total"] = round(time.perf_counter() - started, 1)
        result["path"] = self._write(result) if self.save else None
        LOGGER.event(
            "bench.result",
            section=OUTSIDE_ARTICLE,
            data={
                "file": result["path"],
                "methods": {
                    f"{row['method']}/{row['augment']}": {
                        "setting": row["setting"],
                        "validation": None if row["validation"] is None else (row["validation"]["mean"], row["validation"]["std"]),
                        "test": None if row["test"] is None else (row["test"]["mean"], row["test"]["std"]),
                        "note": row["note"],
                    }
                    for row in result["methods"]
                },
                "gap": result["gap"],
                "seconds": result["seconds"],
                "note": "итог стенда: среднее и стандартное отклонение по наборам; настройки выбраны по validation",
            },
        )
        return result
