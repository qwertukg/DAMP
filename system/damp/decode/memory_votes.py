"""Этап 7: класс из памяти — голоса точек памяти по уровням и «знаю целиком / по частям / не знаю».

Источники. §2.2.5 (нечёткий поиск): у кода окна в памяти есть похожие коды — точки карты, которые он зажёг (§6.1).
§8.2: стимул каждой точки карты известен, значит, известна и его цифра. §7.1.13.3: «it is possible to estimate the
quality of interpretation by evaluating the activation levels of clusters and the contents of active points. It is
possible to clearly distinguish a situation when a space knows a certain word form exactly, from a situation where
the interpretation of a word was obtained in parts, but the word itself is unknown».

Как читается класс:

    голос кусочка   — цифра его самой активной точки памяти: ближайшее воспоминание (§2.2.5);
    голоса уровня   — доли кусочков уровня по цифрам;
    знание уровня   — средняя по кусочкам наибольшая активация точки памяти (кусочек без зажжённых точек даёт 0):
                      насколько память знает то, что видят окна этого уровня;
    вердикт         — сверху вниз. Окно верхнего уровня видит цифру целиком: если верхний уровень знает рисунок
                      (знание не ниже порога), ответ — победитель его голосов, «знаю целиком». Если верх не знает,
                      а на уровне ниже за одну цифру больше RECOGNITION_PARTS_MAJORITY кусочков — «по частям».
                      Иначе — «не знаю».

Доработка (в статье способа нет): порог знания уровня — квантиль RECOGNITION_KNOWLEDGE_QUANTILE знания обучающих
картинок, каждая — без своих собственных точек памяти: обучающая картинка находит в памяти саму себя, и без этого
мерка была бы завышена.

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


@dataclass(frozen=True)
class MemoryVerdict:
    """Ответ памяти (§7.1.13.3): знаю целиком, собрал по частям или не знаю."""

    kind: str  # MemoryVotes.WHOLE, PARTS или UNKNOWN
    digit: int  # цифра; -1 — не знаю
    level: int  # уровень, давший ответ; 0 — ответа нет
    share: float  # доля голосов этой цифры на этом уровне


class MemoryVotes:
    """Чтение класса из памяти уровней (§2.2.5, §7.1.13.3, §8.2).

    labels — цифра каждой точки карты по уровням; point_picture — обучающая картинка окна каждой точки; thresholds —
    пороги знания по уровням (None — порогов нет: верхний уровень «знает», если хоть один кусочек проголосовал).
    """

    WHOLE, PARTS, UNKNOWN = "whole", "parts", "unknown"

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
        """Вердикт по голосам уровней рисунка (от первого уровня; рисунок мог не дойти до верхнего)."""
        seen = {level.number: level for level in levels}
        top = seen.get(self.levels)
        if top is not None and top.voted and top.knowledge >= (0.0 if self.thresholds is None else self.thresholds[-1]):
            return MemoryVerdict(self.WHOLE, top.digit, top.number, top.share)
        majority = float(settings.RECOGNITION_PARTS_MAJORITY)
        for number in range(self.levels - 1, 0, -1):
            part = seen.get(number)
            if part is not None and part.voted and part.share > majority:
                return MemoryVerdict(self.PARTS, part.digit, part.number, part.share)
        return MemoryVerdict(self.UNKNOWN, -1, 0, 0.0)

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
                "активных точек, способ не задан); знание — средняя по кусочкам наибольшая активация точки памяти",
            },
        )
        return MemoryVotes(self.labels, self.point_picture, self.classes, thresholds)
