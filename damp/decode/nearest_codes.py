"""Нечёткий поиск по базе кодов (§2.2.5): класс стимула — по самым похожим кодам базы."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from damp.article_refs import FUZZY_SEARCH
from damp.logging import LOGGER


class NearestCodeClassifier:
    """База кодов с метками; класс нового кода — большинство среди k самых похожих (косинус, §2.2.4, §2.2.5)."""

    BLOCK = 1000

    def __init__(self, packed: "np.ndarray", labels: Sequence[int], code_length: int, *, use_gpu: bool = True) -> None:
        import torch

        packed = np.ascontiguousarray(packed, dtype=np.uint8)
        labels = np.asarray(labels, dtype=np.int64)
        if packed.ndim != 2 or packed.shape[0] != labels.shape[0] or packed.shape[0] == 0:
            raise ValueError("packed codes and labels must be non-empty and of the same length")
        self._packed = packed
        self._labels = labels
        self._length = int(code_length)
        self._torch = torch
        self._device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    def _tensor(self, packed: "np.ndarray"):  # noqa: ANN202
        bits = np.unpackbits(packed, axis=1, count=self._length)
        return self._torch.as_tensor(bits, device=self._device).to(self._torch.float32)

    def neighbours(self, packed: "np.ndarray", count: int) -> "np.ndarray":
        """Номера count самых похожих кодов базы для каждого кода запроса, от самого похожего."""
        torch = self._torch
        count = min(int(count), self._packed.shape[0])
        result = []
        for start in range(0, packed.shape[0], self.BLOCK):
            query = self._tensor(np.ascontiguousarray(packed[start : start + self.BLOCK], dtype=np.uint8))
            query_ones = query.sum(dim=1)
            values, indices = [], []
            # База подаётся на устройство порциями: длинные коды большой базы целиком в память не помещаются.
            for base in range(0, self._packed.shape[0], self.BLOCK):
                block = self._tensor(self._packed[base : base + self.BLOCK])
                cosine = (query @ block.T) / torch.sqrt(
                    torch.clamp(query_ones[:, None] * block.sum(dim=1)[None, :], min=1.0)
                )
                top = cosine.topk(min(count, block.shape[0]), dim=1)
                values.append(top.values.cpu().numpy())
                indices.append(top.indices.cpu().numpy() + base)
            values, indices = np.concatenate(values, axis=1), np.concatenate(indices, axis=1)
            order = np.argsort(-values, axis=1, kind="stable")[:, :count]
            result.append(np.take_along_axis(indices, order, axis=1))
        return np.concatenate(result)

    def accuracy(self, packed: "np.ndarray", labels: Sequence[int], *, neighbours: Sequence[int] = (1,)) -> dict[int, float]:
        """Доля верных ответов при голосовании k ближайших кодов для каждого k из neighbours."""
        labels = np.asarray(labels, dtype=np.int64)
        nearest = self._labels[self.neighbours(packed, max(neighbours))]
        classes = int(max(self._labels.max(), labels.max())) + 1
        result = {}
        for k in neighbours:
            votes = nearest[:, :k]
            predicted = np.asarray([np.bincount(row, minlength=classes).argmax() for row in votes])
            result[int(k)] = float(np.mean(predicted == labels))
        LOGGER.event(
            "decode.nearest_codes",
            section=FUZZY_SEARCH,
            data={
                "base": int(self._packed.shape[0]),
                "queries": int(labels.shape[0]),
                "code_length": self._length,
                "accuracy": {str(k): value for k, value in result.items()},
                "note": "класс — большинство среди k самых похожих кодов базы (косинусное сходство)",
            },
        )
        return result
