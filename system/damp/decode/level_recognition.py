"""Этап 7: узнавание картинки по уровню цикла — по отклику детекторов и голосованием деталей (§2.2.5).

Отклик картинки и голосование — мерки «собрались ли классы», вне статьи; поиск похожих — нечёткий поиск §2.2.5.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from damp import settings
from damp.decode.nearest_codes import NearestCodeClassifier, NearestVectorClassifier


class LevelRecognition:
    """Узнавание по уровню цикла: отклик картинки и голосование её деталей по самым похожим обучающим (§2.2.5)."""

    def __init__(self, labels: "np.ndarray", train: tuple[int, int], neighbours: Sequence[int] | None = None) -> None:
        self.labels = np.asarray(labels)
        self.train = (int(train[0]), int(train[1]))
        self.neighbours = tuple(settings.LADDER_NEIGHBOURS if neighbours is None else neighbours)

    @staticmethod
    def responses(
        picture: "np.ndarray", rows: "np.ndarray", columns: "np.ndarray", levels: "np.ndarray", pictures: int, detectors: int
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """Отклик каждой картинки, единичной длины: (сколько раз сработал каждый детектор на её окнах; самый
        сильный уровень активации каждого детектора по её окнам — как верхний пулинг HMAX C2).

        picture — картинка каждого окна; rows, columns, levels — срабатывания (окно, детектор, уровень активации).
        Из кеша шагов, если такие срабатывания уже были.
        """
        from damp.cache import CACHE

        def compute() -> dict[str, object]:
            response = np.zeros((int(pictures), int(detectors)), dtype=np.float32)
            np.add.at(response, (picture[rows], columns), 1.0)
            response /= np.maximum(np.sqrt((response * response).sum(axis=1, keepdims=True)), 1e-12)
            peak = np.zeros((int(pictures), int(detectors)), dtype=np.float32)
            np.maximum.at(peak, (picture[rows], columns), levels.astype(np.float32))
            peak /= np.maximum(np.sqrt((peak * peak).sum(axis=1, keepdims=True)), 1e-12)
            return {"response": response, "peak": peak}

        inputs = {"picture": picture, "rows": rows, "columns": columns, "levels": levels, "pictures": int(pictures),
                  "detectors": int(detectors)}
        found = CACHE.run("recognition.responses", inputs, compute, modules=("damp.decode.level_recognition",))
        return found["response"], found["peak"]

    def by_response(self, response: "np.ndarray", queries: tuple[int, int]) -> dict[int, float]:
        """Доля верных ответов на картинках queries по самым похожим откликам обучающих (k соседей → доля)."""
        first, last = self.train
        classifier = NearestVectorClassifier(response[first:last], self.labels[first:last])
        return classifier.accuracy(
            response[queries[0] : queries[1]], self.labels[queries[0] : queries[1]], neighbours=self.neighbours
        )

    def by_vote(self, codes: "np.ndarray", code_length: int, picture: "np.ndarray", queries: tuple[int, int]) -> dict[int, float]:
        """Доля верных ответов голосованием деталей: каждая деталь картинки queries голосует классами k самых
        похожих деталей обучающих картинок. codes — код каждой детали, picture — её картинка."""
        labels = self.labels
        train_rows = np.flatnonzero(picture < self.train[1])
        first, last = queries
        asked = np.flatnonzero((picture >= first) & (picture < last))
        classifier = NearestCodeClassifier(codes[train_rows], labels[picture[train_rows]], code_length, use_gpu=settings.ACTIVATION_USE_GPU)
        classes = int(labels.max()) + 1
        nearest = labels[picture[train_rows]][classifier.neighbours(codes[asked], max(self.neighbours))]
        accuracy: dict[int, float] = {}
        for k in self.neighbours:
            votes = np.zeros((last - first, classes), dtype=np.int64)
            np.add.at(votes, (np.repeat(picture[asked] - first, k), nearest[:, :k].reshape(-1)), 1)
            answered = votes.sum(axis=1) > 0
            accuracy[int(k)] = float(np.mean((votes.argmax(axis=1) == labels[first:last]) & answered))
        return accuracy
