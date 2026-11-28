"""Этап 7: узнавание картинки по уровню цикла — по отклику детекторов и голосованием деталей (§2.2.5).

Отклик картинки и голосование — мерки «собрались ли классы», вне статьи; поиск похожих — нечёткий поиск §2.2.5.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from damp import settings
from damp.article_refs import FUZZY_SEARCH, OUTSIDE_ARTICLE
from damp.logging import LOGGER
from damp.decode.nearest_codes import NearestCodeClassifier, NearestVectorClassifier


class LevelRecognition:
    """Узнавание по уровню цикла: отклик картинки и голосование её деталей по самым похожим обучающим (§2.2.5)."""

    # Срабатываний за кусок при подсчёте откликов: временные массивы (картинка срабатывания, уровни float32) — по куску,
    # а не по всем срабатываниям уровня (на 40 000 картинках у уровня 3 их 110 млн — 1.3 ГБ копий). Числа те же:
    # счётчик срабатываний — целые числа в float32, сумма не зависит от порядка; максимум — тоже.
    RESPONSE_CHUNK = 1 << 24

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
            peak = np.zeros((int(pictures), int(detectors)), dtype=np.float32)
            chunk = int(LevelRecognition.RESPONSE_CHUNK)
            for start in range(0, int(np.shape(rows)[0]), chunk):
                owner, column = picture[rows[start : start + chunk]], columns[start : start + chunk]
                np.add.at(response, (owner, column), 1.0)
                np.maximum.at(peak, (owner, column), levels[start : start + chunk].astype(np.float32))
                del owner, column
            response /= np.maximum(np.sqrt((response * response).sum(axis=1, keepdims=True)), 1e-12)
            peak /= np.maximum(np.sqrt((peak * peak).sum(axis=1, keepdims=True)), 1e-12)
            return {"response": response, "peak": peak}

        inputs = {"picture": picture, "rows": rows, "columns": columns, "levels": levels, "pictures": int(pictures),
                  "detectors": int(detectors)}
        found = CACHE.run("recognition.responses", inputs, compute, modules=("damp.decode.level_recognition",))
        return found["response"], found["peak"]

    def by_response(self, response: "np.ndarray", queries: tuple[int, int]) -> dict[int, float]:
        """Доля верных ответов на картинках queries по самым похожим откликам обучающих (k соседей → доля).

        Картинка с нулевым откликом (ни один детектор не сработал на её окнах) ответа не получает: она считается
        неверной, а не получает случайный класс; доля — от всех картинок queries. Обучающие картинки с нулевым
        откликом в базу не входят (как пустые детали у by_vote): молчащая цифра не может стать соседом.
        """
        first, last = self.train
        asked = response[queries[0] : queries[1]]
        answered = asked.any(axis=1)
        base = response[first:last]
        base_labels = self.labels[first:last]
        loud = base.any(axis=1)
        if not bool(loud.any()):
            return {int(k): 0.0 for k in self.neighbours}
        classifier = NearestVectorClassifier(base[loud], base_labels[loud])
        labels = self.labels[queries[0] : queries[1]]
        if bool(answered.all()):
            return classifier.accuracy(asked, labels, neighbours=self.neighbours)
        total = max(1, int(asked.shape[0]))
        silent = float(1.0 - answered.mean()) if asked.shape[0] else 0.0
        LOGGER.event(
            "recognition.response",
            section=OUTSIDE_ARTICLE,
            data={"queries": int(asked.shape[0]), "pictures_without_response": silent,
                  "note": "картинка с нулевым откликом ответа не получает и считается неверной"},
        )
        if not bool(answered.any()):
            return {int(k): 0.0 for k in self.neighbours}
        found = classifier.accuracy(asked[answered], labels[answered], neighbours=self.neighbours)
        count = int(answered.sum())
        return {k: round(value * count) / total for k, value in found.items()}

    def _answers(
        self, codes: "np.ndarray", code_length: int, picture: "np.ndarray", queries: tuple[int, int]
    ) -> tuple[dict[int, "np.ndarray"], dict[str, object]]:
        """Ответы голосования деталей на картинках queries: для каждого k — класс каждой картинки (-1 — без голосов)
        и сведения о пустых деталях для журнала."""
        labels = self.labels
        codes = np.asarray(codes)
        picture = np.asarray(picture)
        filled = codes.any(axis=1)
        in_train = picture < self.train[1]
        first, last = queries
        in_queries = (picture >= first) & (picture < last)
        train_rows = np.flatnonzero(in_train & filled)
        asked = np.flatnonzero(in_queries & filled)
        classes = int(labels.max()) + 1
        nearest = np.zeros((asked.shape[0], max(self.neighbours)), dtype=np.int64)
        if train_rows.size and asked.size:
            classifier = NearestCodeClassifier(
                codes[train_rows], labels[picture[train_rows]], code_length, use_gpu=settings.ACTIVATION_USE_GPU
            )
            found = labels[picture[train_rows]][classifier.neighbours(codes[asked], max(self.neighbours))]
            nearest[:, : found.shape[1]] = found
            usable = found.shape[1]
        else:
            usable = 0
        answers: dict[int, "np.ndarray"] = {}
        silent = 0.0
        for k in self.neighbours:
            votes = np.zeros((last - first, classes), dtype=np.int64)
            if usable:
                np.add.at(votes, (np.repeat(picture[asked] - first, min(k, usable)), nearest[:, : min(k, usable)].reshape(-1)), 1)
            answered = votes.sum(axis=1) > 0
            answers[int(k)] = np.where(answered, votes.argmax(axis=1), -1)
            silent = float(1.0 - answered.mean()) if last > first else 0.0
        info = {
            "details_train": int(np.count_nonzero(in_train)),
            "details_train_empty": float(1.0 - filled[in_train].mean()) if in_train.any() else 0.0,
            "details_asked": int(np.count_nonzero(in_queries)),
            "details_asked_empty": float(1.0 - filled[in_queries].mean()) if in_queries.any() else 0.0,
            "pictures_without_votes": silent,
        }
        return answers, info

    def answers_levels(self, firing: dict, queries: tuple[int, int], *, scale: str = "clip") -> dict[int, "np.ndarray"]:
        """Ответы по уровням активности детекторов (§7.1.12: «detector activity levels can also be used as an
        embedding … components are changing smoothly in an interval [0, 1]»): для каждого k — класс каждой картинки
        queries (-1 — ответа нет).

        Эмбеддинг окна — вектор уровней E(d) всех детекторов уровня, обрезанных в [0, 1]; сходство — косинус. Каждое
        окно картинки голосует классами k самых похожих окон обучающих картинок; окно без сработавших детекторов не
        голосует и в базу не входит. firing — rows (окно), columns (детектор), levels, detectors, picture (картинка
        каждого окна). Эмбеддинг — по статье; голос окон — мерка вне статьи.

        scale — как уровни приводятся к [0, 1]: "clip" — min(E, 1); "train_max" — E делится на наибольший уровень
        этого детектора по окнам обучающих картинок и обрезается единицей (§9.3: «Normalised activation levels»;
        способ — наш, проверочные окна на него не влияют).
        """
        rows, columns = np.asarray(firing["rows"]), np.asarray(firing["columns"])
        raw = np.asarray(firing["levels"], dtype=np.float32)
        picture = np.asarray(firing["picture"])
        detectors = int(firing["detectors"])
        if scale == "clip":
            levels = np.minimum(raw, 1.0)
        elif scale == "train_max":
            taught = picture[rows] < self.train[1]
            maxima = np.zeros(detectors, dtype=np.float32)
            np.maximum.at(maxima, columns[taught], raw[taught])
            top = maxima[columns]
            levels = np.minimum(np.where(top > 0, raw / np.where(top > 0, top, 1.0), raw), 1.0).astype(np.float32)
        else:
            raise ValueError(f"scale: 'clip' или 'train_max', а не {scale!r}")
        filled = np.bincount(rows, minlength=picture.shape[0]) > 0
        first, last = queries
        train_rows = np.flatnonzero((picture < self.train[1]) & filled)
        asked = np.flatnonzero((picture >= first) & (picture < last) & filled)
        if not train_rows.size or not asked.size:
            return {int(k): np.full(last - first, -1, dtype=np.int64) for k in self.neighbours}

        def vectors(windows: "np.ndarray") -> "np.ndarray":
            """Плотные векторы уровней только для нужных окон."""
            place = np.full(picture.shape[0], -1, dtype=np.int64)
            place[windows] = np.arange(windows.shape[0])
            keep = place[rows] >= 0
            dense = np.zeros((windows.shape[0], detectors), dtype=np.float32)
            dense[place[rows[keep]], columns[keep]] = levels[keep]
            return dense

        classes = int(self.labels.max()) + 1
        base_labels = self.labels[picture[train_rows]]
        classifier = NearestVectorClassifier(vectors(train_rows), base_labels)
        nearest = base_labels[classifier.neighbours(vectors(asked), max(self.neighbours), use_gpu=settings.ACTIVATION_USE_GPU)]
        answers: dict[int, "np.ndarray"] = {}
        for k in self.neighbours:
            usable = min(int(k), nearest.shape[1])
            votes = np.zeros((last - first, classes), dtype=np.int64)
            np.add.at(votes, (np.repeat(picture[asked] - first, usable), nearest[:, :usable].reshape(-1)), 1)
            answered = votes.sum(axis=1) > 0
            answers[int(k)] = np.where(answered, votes.argmax(axis=1), -1)
        LOGGER.event(
            "recognition.levels",
            section=FUZZY_SEARCH,
            data={
                "queries": int(last - first), "windows_asked": int(asked.size), "base_windows": int(train_rows.size),
                "detectors": int(firing["detectors"]),
                "note": "эмбеддинг окна — уровни активности детекторов в [0, 1] (§7.1.12), сходство — косинус; окно "
                "голосует классами k самых похожих окон обучающих картинок",
            },
        )
        return answers

    def answers_dominance(self, firing: dict, queries: tuple[int, int]) -> "np.ndarray":
        """Ответ по господству классов в детекторах (§8.2): класс каждой картинки queries (-1 — ответа нет).

        firing["dominance"] — профиль каждого детектора: доли цифр точек карты в его поле (LevelDetectors.dominance).
        Окно верхнего уровня голосует суммой профилей сработавших детекторов с весом их уровня активности в [0, 1]
        (§7.1.12) и отдаёт один голос, разделённый по классам; класс картинки — больше всего голосов её окон.
        Профиль — по статье (§8.2); чтение класса по нему — мерка вне статьи.
        """
        rows, columns = np.asarray(firing["rows"]), np.asarray(firing["columns"])
        levels = np.minimum(np.asarray(firing["levels"], dtype=np.float64), 1.0)
        picture = np.asarray(firing["picture"])
        profile = np.asarray(firing["dominance"], dtype=np.float64)
        first, last = queries
        mine = (picture[rows] >= first) & (picture[rows] < last)
        windows = picture.shape[0]
        scores = np.zeros((windows, profile.shape[1]), dtype=np.float64)
        np.add.at(scores, rows[mine], levels[mine, None] * profile[columns[mine]])
        total = scores.sum(axis=1, keepdims=True)
        scores = np.divide(scores, total, out=np.zeros_like(scores), where=total > 0)  # один голос на окно
        votes = np.zeros((last - first, profile.shape[1]), dtype=np.float64)
        asked = np.flatnonzero((picture >= first) & (picture < last))
        np.add.at(votes, picture[asked] - first, scores[asked])
        answered = votes.sum(axis=1) > 0
        return np.where(answered, votes.argmax(axis=1), -1)

    def answers_within(
        self, codes: "np.ndarray", code_length: int, picture: "np.ndarray", queries: tuple[int, int],
        thresholds: Sequence[float],
    ) -> dict[float, "np.ndarray"]:
        """Ответы нечёткого поиска §2.2.5: для каждого ε — класс каждой картинки queries (-1 — ответа нет).

        Каждая непустая деталь картинки находит ВСЕ непустые детали обучающих картинок с косинусом не ниже ε
        (S(v) статьи) и отдаёт один голос, разделённый по классам найденных; класс картинки — больше всего
        голосов. Поиск — по статье; голос деталей — мерка вне статьи (статья не говорит, как получить класс).
        """
        labels = self.labels
        codes, picture = np.asarray(codes), np.asarray(picture)
        levels = tuple(float(value) for value in thresholds)
        filled = codes.any(axis=1)
        first, last = queries
        train_rows = np.flatnonzero((picture < self.train[1]) & filled)
        asked = np.flatnonzero((picture >= first) & (picture < last) & filled)
        if not train_rows.size or not asked.size:
            return {level: np.full(last - first, -1, dtype=np.int64) for level in levels}
        classes = int(labels.max()) + 1
        classifier = NearestCodeClassifier(
            codes[train_rows], labels[picture[train_rows]], code_length, use_gpu=settings.ACTIVATION_USE_GPU
        )
        counts = classifier.within(codes[asked], levels, classes).astype(np.float64)
        answers: dict[float, "np.ndarray"] = {}
        found: dict[str, float] = {}
        for index, level in enumerate(levels):
            share = counts[:, index, :]
            total = share.sum(axis=1, keepdims=True)
            share = np.divide(share, total, out=np.zeros_like(share), where=total > 0)  # один голос на деталь
            votes = np.zeros((last - first, classes), dtype=np.float64)
            np.add.at(votes, picture[asked] - first, share)
            answered = votes.sum(axis=1) > 0
            answers[level] = np.where(answered, votes.argmax(axis=1), -1)
            found[f"{level:g}"] = float(total.mean())
        LOGGER.event(
            "recognition.within",
            section=FUZZY_SEARCH,
            data={
                "queries": int(last - first),
                "details_asked": int(asked.size),
                "neighbourhood_mean": found,
                "pictures_without_answer": {f"{level:g}": float(np.mean(answers[level] < 0)) for level in levels},
                "note": "нечёткий поиск §2.2.5: деталь находит все детали обучающих с косинусом ≥ ε и отдаёт один "
                "голос, разделённый по их классам; класс картинки — больше всего голосов",
            },
        )
        return answers

    def answers(
        self, codes: "np.ndarray", code_length: int, picture: "np.ndarray", queries: tuple[int, int]
    ) -> dict[int, "np.ndarray"]:
        """Ответы голосования деталей (как by_vote): для каждого k — класс каждой картинки queries, -1 — картинка
        без голосов (все её детали пусты). Нужны мерке устойчивости: совпали ли ответы картинки и её сдвига."""
        return self._answers(codes, code_length, picture, queries)[0]

    def by_vote(self, codes: "np.ndarray", code_length: int, picture: "np.ndarray", queries: tuple[int, int]) -> dict[int, float]:
        """Доля верных ответов голосованием деталей: каждая деталь картинки queries голосует классами k самых
        похожих деталей обучающих картинок. codes — код каждой детали, picture — её картинка.

        Пустые детали (нулевой код: ни один детектор не сработал) не голосуют и не входят в базу обучающих;
        картинка без голосов не получает ответа и считается неверной (доля — от всех картинок queries).
        """
        first, last = queries
        answers, info = self._answers(codes, code_length, picture, queries)
        truth = self.labels[first:last]
        accuracy = {k: float(np.mean((found == truth) & (found >= 0))) for k, found in answers.items()}
        LOGGER.event(
            "recognition.vote",
            section=OUTSIDE_ARTICLE,
            data={
                **info,
                "accuracy": {str(k): v for k, v in accuracy.items()},
                "note": "пустые детали (ни один детектор не сработал) не голосуют и не входят в базу обучающих; "
                "картинка без голосов ответа не получает и считается неверной",
            },
        )
        return accuracy
