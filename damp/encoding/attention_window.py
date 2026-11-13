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
from damp.encoding.damp_encoder import Encoder
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

    square = False  # окно круглое

    def __init__(self, *, radius: float, stride: int = 1) -> None:
        if radius <= 0:
            raise ValueError("radius must be positive")
        if stride < 1:
            raise ValueError("stride must be at least 1")
        self.radius = float(radius)
        self.stride = int(stride)
        self.extent = self.radius  # половина размера окна в клетках
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


class BlockWindows:
    """Окна-блоки: квадраты со стороной side клеток, уложенные плотно, без перекрытия (расширение статьи).

    Каждый элемент картинки входит ровно в один кусок. Поэтому число кусков картинки, умноженное на среднее
    число элементов в куске, равно числу её элементов: чем больше элементов в куске, тем меньше кусков.
    Блок с номером b по оси занимает клетки [b·side, (b+1)·side), его центр — b·side + (side − 1)/2.
    Координата элемента — центр его клетки; у куска предыдущего уровня — центр его блока, поэтому блоки
    со стороной, кратной стороне предыдущего уровня, собирают его куски целиком.
    """

    square = True  # окно квадратное

    def __init__(self, *, side: int) -> None:
        if side < 1:
            raise ValueError("side must be at least 1")
        self.side = int(side)
        self.extent = self.side / 2.0  # половина размера окна в клетках

    def pieces(self, picture: "np.ndarray", y: "np.ndarray", x: "np.ndarray") -> WindowPieces:
        """Куски всех картинок: picture, y, x — номер картинки и координаты каждого элемента."""
        picture = np.asarray(picture, dtype=np.int64)
        y = np.asarray(y, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        if not (picture.shape == y.shape == x.shape) or picture.ndim != 1:
            raise ValueError("picture, y and x must be one-dimensional arrays of the same length")
        if picture.size == 0:
            raise ValueError("no elements")
        block_y = np.floor(y / self.side).astype(np.int64)
        block_x = np.floor(x / self.side).astype(np.int64)
        low = int(min(block_y.min(), block_x.min()))
        span = int(max(block_y.max(), block_x.max())) - low + 1
        keys = (picture * span + (block_y - low)) * span + (block_x - low)
        order = np.argsort(keys, kind="stable")
        piece_keys, piece = np.unique(keys[order], return_inverse=True)
        piece = piece.reshape(-1).astype(np.int64)
        shift = (self.side - 1) / 2.0
        centre = np.stack(
            [((piece_keys // span) % span + low) * self.side + shift, (piece_keys % span + low) * self.side + shift],
            axis=1,
        ).astype(np.float64)
        shifts = np.stack([y[order] - centre[piece, 0], x[order] - centre[piece, 1]], axis=1)
        offsets, offset = np.unique(shifts, axis=0, return_inverse=True)
        pieces = WindowPieces(
            offsets=offsets.astype(np.float64),
            picture=piece_keys // (span * span),
            centre=centre,
            piece=piece,
            element=order.astype(np.int64),
            offset=offset.reshape(-1).astype(np.int64),
        )
        pictures = int(np.unique(picture).shape[0])
        LOGGER.event(
            "attention_window.blocks",
            section=OUTSIDE_ARTICLE,
            data={
                "side": self.side,
                "offsets": int(offsets.shape[0]),
                "pictures": pictures,
                "elements": int(picture.shape[0]),
                "pieces": pieces.count,
                "pieces_per_picture": pieces.count / max(1, pictures),
                "elements_per_piece": float(picture.shape[0]) / max(1, pieces.count),
                "note": "окна-блоки без перекрытия: элемент входит ровно в один кусок, кусков × элементов в куске "
                "= элементов картинки (в статье окна нет)",
            },
        )
        return pieces


class DetectorWindows:
    """Окна — сработавшие детекторы-круги, наброшенные на плоскость картинки (расширение статьи; круги — §4.3).

    Картинка заранее не режется. Слой перекрывающихся кругов покрывает плоскость картинки, и кусок — элементы
    картинки внутри одного сработавшего круга. Элемент входит во все круги, в которые попал (§4.1), поэтому
    соседние куски имеют общие элементы (§4.4.1: соседние окна дают общие биты).
    """

    square = False  # окно круглое

    def __init__(self, *, plane: Encoder) -> None:
        circles = plane.detectors
        if not circles:
            raise ValueError("plane encoder has no detectors")
        self._plane = plane
        self._centre = np.asarray([(circle.center_y, circle.center_x) for circle in circles], dtype=np.float64)
        self.extent = float(max(circle.radius for circle in circles))  # радиус круга в клетках

    def pieces(self, picture: "np.ndarray", y: "np.ndarray", x: "np.ndarray") -> WindowPieces:
        """Куски всех картинок: picture, y, x — номер картинки и координаты каждого элемента."""
        picture = np.asarray(picture, dtype=np.int64)
        position = np.stack([np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)], axis=1)
        if picture.ndim != 1 or position.shape[0] != picture.shape[0]:
            raise ValueError("picture, y and x must be one-dimensional arrays of the same length")
        if picture.size == 0:
            raise ValueError("no elements")
        values, inverse = np.unique(position, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        rows, ids = self._plane.active_detectors(0, values)
        order = np.argsort(rows, kind="stable")
        rows, ids = rows[order], ids[order]
        # Круги каждого места раздаются всем элементам, стоящим в этом месте.
        counts = np.bincount(rows, minlength=values.shape[0])
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        per_element = counts[inverse]
        element = np.repeat(np.arange(picture.shape[0], dtype=np.int64), per_element)
        within = np.arange(element.shape[0]) - np.repeat(np.cumsum(per_element) - per_element, per_element)
        circle = ids[np.repeat(starts[inverse], per_element) + within]
        total = self._centre.shape[0]
        keys = picture[element] * total + circle
        order = np.argsort(keys, kind="stable")
        keys, element = keys[order], element[order]
        piece_keys, piece = np.unique(keys, return_inverse=True)
        piece = piece.reshape(-1).astype(np.int64)
        centre = self._centre[piece_keys % total]
        offsets, offset = np.unique(np.round(position[element] - centre[piece], 6), axis=0, return_inverse=True)
        pieces = WindowPieces(
            offsets=offsets.astype(np.float64),
            picture=piece_keys // total,
            centre=centre,
            piece=piece,
            element=element,
            offset=offset.reshape(-1).astype(np.int64),
        )
        pictures = int(np.unique(picture).shape[0])
        LOGGER.event(
            "attention_window.detectors",
            section=OUTSIDE_ARTICLE,
            data={
                "radius": self.extent,
                "circles": int(total),
                "offsets": int(offsets.shape[0]),
                "pictures": pictures,
                "elements": int(picture.shape[0]),
                "pieces": pieces.count,
                "pieces_per_picture": pieces.count / max(1, pictures),
                "elements_per_piece": float(element.shape[0]) / max(1, pieces.count),
                "circles_per_element": float(element.shape[0]) / max(1, picture.shape[0]),
                "note": "окна — сработавшие круги наброшенного слоя детекторов: кусок — элементы картинки внутри "
                "одного круга, элемент входит во все круги, в которые попал (в статье окна нет)",
            },
        )
        return pieces
