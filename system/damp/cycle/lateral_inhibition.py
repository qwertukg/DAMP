"""Торможение соседей: из близко стоящих деталей остаётся самая сильная (вне статьи; основание — физиология зрения).

Окна внимания соседних миниколонок сильно перекрываются (продолжение статьи, П.2), поэтому детали, стоящие ближе
своего поля зрения, почти повторяют друг друга. В зрительной коре так же: в пределах модуля около 2 мм поля клеток
перекрываются, а совсем новое место поля зрения начинается только через модуль (Hubel & Wiesel 1974). Повторы
убирает конкуренция: тормозные связи снижают ответ менее активных нейронов относительно более активных на
расстоянии 1–2 мм коры (Rolls 2012). В самой статье есть близкое правило для детекторов над раскладкой: центры не
попадают в поля друг друга, при перекрытии остаётся более плотный (§6.4.5).

Здесь то же для выходов миниколонок уровня: деталь остаётся, если в круге торможения вокруг неё нет уже оставшейся
более сильной детали. Сила детали — её уровень активации (П.1). Радиус круга — доля того, что видит сама деталь.
Отбор идёт по убыванию силы, поэтому не зависит от места картинки в кадре.

Пятна (similarity). Соседние миниколонки «часто распознают похожие элементы», и пятна активации образуют кластеры
(П.2, П.3): несколько соседних окон предъявляют одно и то же понятие. С порогом similarity более сильная деталь гасит
соседку, только если их коды похожи не меньше порога — из пятна остаётся одна деталь, а непохожие соседи остаются
все. Без порога гасится любая соседка (прежнее торможение: оно убирало и разные понятия — уровень терял цифру).
В зрительной коре так же: окружение подавляет ответ сильнее всего, когда в нём тот же признак (iso-orientation
surround suppression, Knierim & van Essen 1992).
"""
from __future__ import annotations

import numpy as np

from damp.article_refs import DETECTOR_INSERTION
from damp.logging import LOGGER


class LateralInhibition:
    """Отбор деталей уровня: в круге торможения остаётся самая сильная."""

    def __init__(self, *, radius_share: float, similarity: float | None = None, name: str = "details") -> None:
        if radius_share <= 0:
            raise ValueError("radius_share must be positive")
        if similarity is not None and not 0.0 < similarity <= 1.0:
            raise ValueError("similarity must be in (0, 1]")
        self._share = float(radius_share)
        self._similarity = None if similarity is None else float(similarity)
        self._name = str(name)

    @classmethod
    def thin(cls, elements, name: str):  # noqa: ANN001, ANN206
        """Элементы уровня после торможения по настройкам (LADDER_INHIBITION_RADIUS, LADDER_INHIBITION_SIMILARITY);
        без торможения — те же элементы. Один вход для прогона цикла и для пути одной картинки."""
        from damp import settings

        if settings.LADDER_INHIBITION_RADIUS is None:
            return elements
        similarity = settings.LADDER_INHIBITION_SIMILARITY
        chooser = cls(radius_share=float(settings.LADDER_INHIBITION_RADIUS),
                      similarity=None if similarity is None else float(similarity), name=name)
        codes = None if similarity is None else (elements.codes(), int(elements.what_length))
        return elements.only(chooser.survivors(elements.picture, elements.y, elements.x, elements.power(), elements.reach(), codes=codes))

    def survivors(
        self,
        picture: "np.ndarray",
        y: "np.ndarray",
        x: "np.ndarray",
        strength: "np.ndarray",
        sight: "np.ndarray",
        *,
        codes: "tuple[np.ndarray, int] | None" = None,
    ) -> "np.ndarray":
        """Маска оставшихся деталей.

        picture — номер картинки каждой детали (по возрастанию), y, x — её место, strength — уровень активации,
        sight — сколько клеток от своего места она видит. Радиус торможения детали — radius_share · sight.
        codes — (упакованные коды деталей, длина кода): нужны при пороге similarity — гасит только похожая соседка.
        """
        if self._similarity is not None and codes is None:
            raise ValueError("codes are required when similarity is set")
        picture = np.asarray(picture, dtype=np.int64)
        position = np.stack([np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)], axis=1)
        strength = np.asarray(strength, dtype=np.float64)
        radius = self._share * np.asarray(sight, dtype=np.float64)
        if not (picture.shape[0] == position.shape[0] == strength.shape[0] == radius.shape[0]) or picture.ndim != 1:
            raise ValueError("picture, y, x, strength and sight must be one-dimensional arrays of the same length")
        if picture.size and np.any(np.diff(picture) < 0):
            raise ValueError("details must be ordered by picture")
        keep = np.zeros(picture.shape[0], dtype=bool)
        pictures, starts = np.unique(picture, return_index=True)
        bounds = np.append(starts, picture.shape[0])
        for first, last in zip(bounds[:-1], bounds[1:]):
            own = position[first:last]
            alike = None
            if self._similarity is not None:
                bits = np.unpackbits(np.asarray(codes[0][first:last], dtype=np.uint8), axis=1, count=int(codes[1])).astype(np.float32)
                ones = bits.sum(axis=1)
                alike = (bits @ bits.T) / np.sqrt(np.maximum(ones[:, None] * ones[None, :], 1.0)) >= self._similarity
            # От сильных к слабым; при равной силе — по месту (сверху вниз, слева направо): порядок не зависит
            # от того, как детали пронумерованы.
            order = np.lexsort((own[:, 1], own[:, 0], -strength[first:last]))
            kept: list[int] = []
            for index in order:
                if strength[first + index] <= 0.0:
                    break  # деталь без единого сработавшего детектора ничего не узнала
                if kept:
                    held = np.asarray(kept)
                    distance = np.hypot(own[held, 0] - own[index, 0], own[held, 1] - own[index, 1])
                    near = distance <= radius[first + index]
                    if np.any(near if alike is None else near & alike[index, held]):
                        continue
                kept.append(int(index))
            keep[first + np.asarray(kept, dtype=np.int64)] = True
        total = max(1, int(picture.shape[0]))
        count = max(1, int(pictures.shape[0]))
        LOGGER.event(
            "activation.lateral_inhibition",
            section=DETECTOR_INSERTION,
            data={
                "name": self._name,
                "details": int(picture.shape[0]),
                "kept": int(np.count_nonzero(keep)),
                "kept_share": float(np.count_nonzero(keep)) / total,
                "silent": float(np.mean(strength <= 0.0)) if strength.size else 0.0,
                "details_per_picture": float(picture.shape[0]) / count,
                "kept_per_picture": float(np.count_nonzero(keep)) / count,
                "radius_share": self._share,
                "similarity": 0.0 if self._similarity is None else self._similarity,
                "radius_mean": float(radius.mean()) if radius.size else 0.0,
                "note": "торможение соседей: деталь остаётся, если в круге radius_share · (что она видит) нет уже "
                "оставшейся более сильной; близко к вставке детекторов §6.4.5 (при перекрытии остаётся более "
                "плотный), применено к выходам миниколонок уровня; основание — конкуренция соседних колонок коры",
            },
        )
        return keep
