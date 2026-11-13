"""Окно внимания: куски картинки в координатах своего окна (расширение статьи, указание её автора).

«Окно внимания должно быть достаточно большим, чтобы в него влезало много интересного, но не слишком
большим, чтобы комбинаторика оставалась приемлемой. То, что внутри окна, будет точкой кодового
пространства». Кусок — элементы картинки, попавшие в круглое окно, записанные от центра окна: один и тот же
рисунок даёт один и тот же кусок в любом месте картинки («что»), а место окна хранится отдельно («где»).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from damp.article_refs import OUTSIDE_ARTICLE
from damp.logging import LOGGER


@dataclass(frozen=True)
class WindowPieces:
    """Куски набора картинок: кусок — (картинка, центр окна), вхождение — элемент картинки в окне куска."""

    offsets: "np.ndarray"  # (D, 2) возможные смещения элемента от центра окна (dy, dx)
    picture: "np.ndarray"  # (P,) номер картинки каждого куска
    centre: "np.ndarray"  # (P, 2) центр окна каждого куска (y, x)
    piece: "np.ndarray"  # (E,) номер куска каждого вхождения, по возрастанию
    element: "np.ndarray"  # (E,) номер элемента входа
    offset: "np.ndarray"  # (E,) номер смещения из offsets

    @property
    def count(self) -> int:
        return int(self.picture.shape[0])


class AttentionWindows:
    """Круглые окна радиуса radius с центрами в узлах решётки шага stride (расширение статьи).

    Элементы картинки стоят в целых узлах (y, x). Окно с центром в узле c содержит элементы на расстоянии
    не больше radius; пустые окна кусков не дают. Шаг 1 — окно в каждом узле (плотная выборка).
    """

    def __init__(self, *, radius: float, stride: int = 1) -> None:
        if radius <= 0:
            raise ValueError("radius must be positive")
        if stride < 1:
            raise ValueError("stride must be at least 1")
        self.radius = float(radius)
        self.stride = int(stride)
        reach = int(math.floor(self.radius))
        self.offsets = np.asarray(
            [
                (dy, dx)
                for dy in range(-reach, reach + 1)
                for dx in range(-reach, reach + 1)
                if dy * dy + dx * dx <= self.radius * self.radius
            ],
            dtype=np.int64,
        )

    def pieces(self, picture: "np.ndarray", y: "np.ndarray", x: "np.ndarray") -> WindowPieces:
        """Куски всех картинок: picture, y, x — номер картинки и целые координаты каждого элемента."""
        picture = np.asarray(picture, dtype=np.int64)
        y = np.asarray(y, dtype=np.int64)
        x = np.asarray(x, dtype=np.int64)
        if not (picture.shape == y.shape == x.shape) or picture.ndim != 1:
            raise ValueError("picture, y and x must be one-dimensional arrays of the same length")
        if picture.size == 0:
            raise ValueError("no elements")
        reach = int(math.floor(self.radius))
        low = int(min(y.min(), x.min())) - reach
        span = int(max(y.max(), x.max())) + reach - low + 1
        numbers = np.arange(picture.shape[0], dtype=np.int64)
        keys, elements, offsets = [], [], []
        for number, (dy, dx) in enumerate(self.offsets):
            cy, cx = y - dy, x - dx
            keep = (cy % self.stride == 0) & (cx % self.stride == 0)
            keys.append((picture[keep] * span + (cy[keep] - low)) * span + (cx[keep] - low))
            elements.append(numbers[keep])
            offsets.append(np.full(int(np.count_nonzero(keep)), number, dtype=np.int64))
        keys, elements, offsets = np.concatenate(keys), np.concatenate(elements), np.concatenate(offsets)
        order = np.argsort(keys, kind="stable")
        keys, elements, offsets = keys[order], elements[order], offsets[order]
        piece_keys, piece = np.unique(keys, return_inverse=True)
        pieces = WindowPieces(
            offsets=self.offsets.astype(np.float64),
            picture=piece_keys // (span * span),
            centre=np.stack([(piece_keys // span) % span + low, piece_keys % span + low], axis=1).astype(np.float64),
            piece=piece.reshape(-1).astype(np.int64),
            element=elements,
            offset=offsets,
        )
        pictures = int(np.unique(picture).shape[0])
        LOGGER.event(
            "attention_window.pieces",
            section=OUTSIDE_ARTICLE,
            data={
                "radius": self.radius,
                "stride": self.stride,
                "offsets": int(self.offsets.shape[0]),
                "pictures": pictures,
                "elements": int(picture.shape[0]),
                "pieces": pieces.count,
                "pieces_per_picture": pieces.count / max(1, pictures),
                "elements_per_piece": float(pieces.piece.shape[0]) / max(1, pieces.count),
                "note": "окно внимания: кусок — элементы в окне, записанные от его центра; место окна хранится "
                "отдельно (указание автора статьи, в статье окна нет)",
            },
        )
        return pieces
