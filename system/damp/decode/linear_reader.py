"""Этап 7: линейный читатель класса над нормированными уровнями активации иерархии детекторов (§9.3).

§9.3: «Normalised activation levels of a detector hierarchy can be used as an embeddings in ℝⁿ, instead of sparse
binary codes. We can imagine a heterogeneous architecture in which the output of a discrete model is fed as an input
to a neural network». Простейшая такая сеть — один линейный слой с softmax (многоклассовая логистическая регрессия).
Сам читатель — с учителем, вне памяти DAMP: учится только на обучающих картинках, отложенные на него не влияют.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from damp.article_refs import NEURAL_INTEGRATION
from damp.logging import LOGGER


class LinearReader:
    """Линейный слой с softmax над эмбеддингом картинки в ℝⁿ (§9.3); штраф L2 на веса — decay.

    Обучение — L-BFGS (SciPy) по средней перекрёстной энтропии обучающих картинок плюс decay · ‖W‖²; старт с нулей,
    поэтому результат детерминирован. Картинка с нулевым эмбеддингом (ни один детектор не сработал) ответа не
    получает (-1), как в остальных читателях; в обучение такие картинки не входят.
    """

    ITERATIONS = 500

    def __init__(self, features: "np.ndarray", labels: "np.ndarray", decay: float) -> None:
        from scipy.optimize import minimize

        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        loud = features.any(axis=1)
        x, y = features[loud], labels[loud]
        self.classes = int(labels.max()) + 1 if labels.size else 0
        self.decay = float(decay)
        count, width = x.shape
        classes = self.classes
        onehot = np.eye(classes)[y]

        def loss(theta: "np.ndarray") -> tuple[float, "np.ndarray"]:
            weights = theta[: width * classes].reshape(width, classes)
            bias = theta[width * classes :]
            logits = x @ weights + bias
            logits -= logits.max(axis=1, keepdims=True)
            chance = np.exp(logits)
            chance /= chance.sum(axis=1, keepdims=True)
            value = -float(np.mean(np.log(chance[np.arange(count), y] + 1e-300))) + self.decay * float(np.sum(weights * weights))
            error = (chance - onehot) / count
            grad = np.concatenate([(x.T @ error + 2.0 * self.decay * weights).ravel(), error.sum(axis=0)])
            return value, grad

        found = minimize(loss, np.zeros(width * classes + classes), jac=True, method="L-BFGS-B",
                         options={"maxiter": self.ITERATIONS})
        self.weights = found.x[: width * classes].reshape(width, classes)
        self.bias = found.x[width * classes :]
        self.loss = float(found.fun)
        self.iterations = int(found.nit)

    def answers(self, features: "np.ndarray") -> "np.ndarray":
        """Класс каждой картинки; -1 — у картинки нулевой эмбеддинг."""
        features = np.asarray(features, dtype=np.float64)
        guess = (features @ self.weights + self.bias).argmax(axis=1)
        return np.where(features.any(axis=1), guess, -1)

    def probabilities(self, features: "np.ndarray") -> "np.ndarray":
        """Вероятности классов softmax(xW + b) каждой картинки (§9.3); у нулевого эмбеддинга — нули (ответа нет).
        Самый вероятный класс — тот же, что у answers: softmax не меняет порядка логитов."""
        features = np.asarray(features, dtype=np.float64)
        logits = features @ self.weights + self.bias
        logits -= logits.max(axis=1, keepdims=True)
        chance = np.exp(logits)
        chance /= chance.sum(axis=1, keepdims=True)
        return np.where(features.any(axis=1, keepdims=True), chance, 0.0)

    @classmethod
    def answers_for(
        cls, features: "np.ndarray", labels: "np.ndarray", train: tuple[int, int], parts: dict[str, tuple[int, int]],
        decays: Sequence[float],
    ) -> dict[float, dict[str, "np.ndarray"]]:
        """Для каждого decay — ответы на каждой части parts (имя → (первая картинка, после последней)); читатель
        учится на картинках train."""
        found: dict[float, dict[str, "np.ndarray"]] = {}
        first, last = train
        for decay in decays:
            reader = cls(features[first:last], labels[first:last], decay)
            found[float(decay)] = {name: reader.answers(features[begin:end]) for name, (begin, end) in parts.items()}
            LOGGER.event(
                "recognition.linear",
                section=NEURAL_INTEGRATION,
                data={"decay": float(decay), "train_pictures": int(last - first), "features": int(features.shape[1]),
                      "loss": reader.loss, "iterations": reader.iterations,
                      "note": "§9.3: нормированные уровни активации иерархии детекторов — эмбеддинг в ℝⁿ, вход "
                              "нейросети; здесь — один линейный слой с softmax, учится на обучающих картинках"},
            )
        return found
