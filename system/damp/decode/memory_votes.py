"""Этап 7: класс из памяти — голоса точек памяти по уровням, их среднее и согласие уровней.

Источники. §2.2.5 (нечёткий поиск): у кода окна в памяти есть похожие коды — точки карты, которые он зажёг (§6.1).
§8.2: стимул каждой точки карты известен, значит, известна и его цифра. §7.1.13.3: «it is possible to estimate the
quality of interpretation by evaluating the activation levels of clusters and the contents of active points. It is
possible to clearly distinguish a situation when a space knows a certain word form exactly, from a situation where
the interpretation of a word was obtained in parts, but the word itself is unknown».

Как читается класс:

    голос кусочка   — цифра его самой активной точки памяти: ближайшее воспоминание (§2.2.5);
    голоса уровня   — доли кусочков уровня по цифрам;
    ответ           — цифра с наибольшей средней долей голосов по уровням, до которых дошёл рисунок: уверенный
                      уровень весит больше неуверенного, и ответ есть всегда, когда зажглась хоть одна точка памяти;
    уверенность     — согласие уровней: все уровни назвали эту цифру; её назвали два верхних; уровни расходятся;
    знание уровня   — средняя по кусочкам наибольшая активация точки памяти (кусочек без зажжённых точек даёт 0):
                      насколько память знает то, что видят окна этого уровня. В ответ не входит — показывается рядом.

Доработка (в статье способа нет). Замер 2026-11-17, 500 проверочных цифр, два цикла (1 000 цифр с картой 10 000 точек;
5 000 цифр с картой 5 000):

    среднее долей по всем уровням        89.0 и 95.0 %   — принято;
    только верхний уровень               87.8 и 93.8 %;
    большинство среди ответов уровней    84.6 и 90.2 %   — два слабых уровня перекрикивают один сильный;
    прежнее правило «знаю целиком / по частям / не знаю» (ответ верхнего уровня, если его знание не ниже порога,
    иначе — уровня ниже с большинством кусочков за одну цифру)   83.4 и 88.0 % от всех: у 6–8 % цифр оно молчало,
    хотя верхний уровень там был прав в 68–77 % случаев;
    линейный читатель (§9.3)             92.4 и 95.6 %.

    Согласие уровней как уверенность: все три уровня за одну цифру — у 44–49 % цифр, верно 97.6–98.2 %; два верхних
    согласны — у 77 %, верно 94.6–97.9 %; расходятся — у 23 %, верхний прав в 65–80 %.

Порог знания уровня — квантиль RECOGNITION_KNOWLEDGE_QUANTILE знания обучающих картинок, каждая — без своих
собственных точек памяти: обучающая картинка находит в памяти саму себя, и без этого мерка была бы завышена.

Зачем. Линейный читатель (§9.3) — отдельный слой с учителем над пиками детекторов всей картинки: он обучен только на
целых цифрах и всегда делит 100 % между десятью классами. На части цифры (перекрестие «X» — середина восьмёрки) его
ответ решает смещение классов, а не детекторы; голоса памяти читают то же, что показывают карты уровней.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from damp import settings
from damp.article_refs import UNKNOWN_WORDS
from damp.logging import LOGGER


@dataclass(frozen=True)
class LevelVotes:
    """Голоса памяти одного уровня (§2.2.5, §7.1.13.3)."""

    number: int  # номер уровня, с единицы
    votes: "np.ndarray"  # (классов,) доля кусочков, чья самая активная точка памяти — эта цифра; нули — голосов нет
    knowledge: float  # знание уровня: средняя по кусочкам наибольшая активация точки памяти
    keyholes: int  # сколько кусочков (окон) у рисунка на уровне
    voted: int  # сколько из них зажгли хотя бы одну точку памяти

    @property
    def digit(self) -> int:
        """Победитель голосов уровня; -1 — никто не голосовал."""
        return int(self.votes.argmax()) if self.voted else -1

    @property
    def share(self) -> float:
        """Доля голосов победителя."""
        return float(self.votes.max()) if self.voted else 0.0


@dataclass(frozen=True, eq=False)
class MemoryVerdict:
    """Ответ памяти (§2.2.5, §7.1.13.3): цифра по среднему долей голосов уровней и согласие уровней."""

    kind: str  # MemoryVotes.AGREE, MOSTLY, SPLIT или UNKNOWN
    digit: int  # цифра; -1 — не знаю
    share: float  # средняя по уровням доля голосов этой цифры
    agreed: int  # сколько уровней назвали эту цифру победителем своих голосов
    votes: "np.ndarray"  # (классов,) средние по уровням доли голосов; нули — голосов нет


class MemoryVotes:
    """Чтение класса из памяти уровней (§2.2.5, §7.1.13.3, §8.2).

    labels — цифра каждой точки карты по уровням; point_picture — обучающая картинка окна каждой точки; thresholds —
    пороги знания по уровням (None — пороги не меряны).
    """

    # Уверенность ответа — согласие уровней: все обученные уровни назвали цифру ответа; её назвали два верхних
    # уровня; уровни расходятся; ни один кусочек не зажёг ни одной точки памяти.
    AGREE, MOSTLY, SPLIT, UNKNOWN = "agree", "mostly", "split", "unknown"

    def __init__(
        self,
        labels: Sequence["np.ndarray"],
        point_picture: Sequence["np.ndarray"],
        classes: int,
        thresholds: Sequence[float] | None = None,
    ) -> None:
        if len(labels) != len(point_picture):
            raise ValueError("цифры и картинки точек нужны по каждому уровню")
        self.labels = tuple(np.asarray(level, dtype=np.int64) for level in labels)
        self.point_picture = tuple(np.asarray(level, dtype=np.int64) for level in point_picture)
        self.classes = int(classes)
        if thresholds is not None and len(thresholds) != len(self.labels):
            raise ValueError("пороги знания нужны по каждому уровню")
        self.thresholds = None if thresholds is None else tuple(float(value) for value in thresholds)

    @property
    def levels(self) -> int:
        return len(self.labels)

    def level(
        self,
        number: int,
        keyholes: int,
        lit_keyhole: "np.ndarray",
        lit_point: "np.ndarray",
        lit_value: "np.ndarray",
        *,
        without_picture: int = -1,
    ) -> LevelVotes:
        """Голоса уровня number (с единицы) по зажжённым точкам его карты (§6.1): окно, точка, активация a_j.

        without_picture — обучающая картинка, чьи точки памяти не считаются (мерка порога знания: картинка без самой
        себя); -1 — считаются все.
        """
        labels, pictures = self.labels[number - 1], self.point_picture[number - 1]
        keyhole = np.asarray(lit_keyhole, dtype=np.int64)
        point = np.asarray(lit_point, dtype=np.int64)
        value = np.asarray(lit_value, dtype=np.float64)
        if without_picture >= 0:
            keep = pictures[point] != int(without_picture)
            keyhole, point, value = keyhole[keep], point[keep], value[keep]
        best = np.zeros(int(keyholes), dtype=np.float64)
        votes = np.zeros(self.classes, dtype=np.float64)
        voted = 0
        if keyhole.size:
            np.maximum.at(best, keyhole, value)
            # Самая активная точка каждого кусочка; при равных активациях — точка с меньшим номером (детерминизм).
            order = np.lexsort((point, -value, keyhole))
            first = np.flatnonzero(np.r_[True, keyhole[order][1:] != keyhole[order][:-1]])
            np.add.at(votes, labels[point[order][first]], 1.0)
            voted = int(first.shape[0])
            votes /= float(voted)
        return LevelVotes(
            number=int(number),
            votes=votes,
            knowledge=float(best.mean()) if keyholes else 0.0,
            keyholes=int(keyholes),
            voted=voted,
        )

    def verdict(self, levels: Sequence[LevelVotes]) -> MemoryVerdict:
        """Ответ по голосам уровней рисунка (от первого уровня; рисунок мог не дойти до верхнего).

        Цифра — с наибольшей средней долей голосов по уровням, где голосовал хоть один кусочек (при равенстве —
        меньшая цифра). Уверенность — согласие уровней с этой цифрой.
        """
        voted = [level for level in levels if level.voted]
        if not voted:
            return MemoryVerdict(self.UNKNOWN, -1, 0.0, 0, np.zeros(self.classes, dtype=np.float64))
        mean = np.sum([level.votes for level in voted], axis=0) / float(len(voted))
        digit = int(mean.argmax())
        winners = {level.number: level.digit for level in voted}
        agreed = sum(1 for winner in winners.values() if winner == digit)
        if agreed == self.levels:
            kind = self.AGREE
        elif winners.get(self.levels) == digit and winners.get(self.levels - 1) == digit:
            kind = self.MOSTLY
        else:
            kind = self.SPLIT
        return MemoryVerdict(kind, digit, float(mean[digit]), int(agreed), mean)

    def calibrated(self, knowledge: "np.ndarray") -> "MemoryVotes":
        """Те же голоса с порогами знания: knowledge — знание уровней (картинка × уровень) обучающих картинок без
        своих точек памяти; порог уровня — квантиль RECOGNITION_KNOWLEDGE_QUANTILE (доработка, вне статьи)."""
        knowledge = np.asarray(knowledge, dtype=np.float64).reshape(-1, self.levels)
        quantile = float(settings.RECOGNITION_KNOWLEDGE_QUANTILE)
        thresholds = [float(np.quantile(knowledge[:, level], quantile)) if knowledge.shape[0] else 0.0 for level in range(self.levels)]
        LOGGER.event(
            "recognition.knowledge",
            section=UNKNOWN_WORDS,
            data={
                "pictures": int(knowledge.shape[0]),
                "quantile": quantile,
                "thresholds": thresholds,
                "median": [float(np.median(knowledge[:, level])) if knowledge.shape[0] else 0.0 for level in range(self.levels)],
                "note": "порог знания уровня — квантиль знания обучающих картинок без своих точек памяти (доработка: "
                "в §7.1.13.3 сказано отличать «знаю точно» от «собрано по частям» по уровням активации и содержимому "
                "активных точек, способ не задан); знание — средняя по кусочкам наибольшая активация точки памяти; "
                "в ответ знание не входит, показывается рядом с голосами уровня",
            },
        )
        return MemoryVotes(self.labels, self.point_picture, self.classes, thresholds)
