"""Этап 6: окна внимания уровня — где стоят и сколько видят (продолжение статьи, П.2, П.3; радиусы — §6.4.1–§6.4.3)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from damp import settings
from damp.article_refs import CONTINUATION_SPATIAL_CODES
from damp.cycle.plane_fields import NeighbourScale, PlaneFieldBuilder
from damp.encoding.keyhole import KeyholePieces
from damp.encoding.elements import LadderElements
from damp.logging import LOGGER
from damp.stimuli.digits import DigitGroups


@dataclass(frozen=True)
class LadderField:
    """Поле зрения одного уровня: шаг мест его элементов и клетки картинки, которые оно видит."""

    number: int
    unit: float  # шаг мест элементов кодера места, в долях радиуса окна
    box: tuple[int, int, int, int]  # клетки, которые в среднем видит поле, относительно центра: верх, низ, лево, право
    inner: tuple[int, int, int, int]  # клетки, которые в среднем видит один его элемент, относительно места элемента

    @property
    def height(self) -> int:
        return self.box[1] - self.box[0] + 1

    @property
    def width(self) -> int:
        return self.box[3] - self.box[2] + 1

    @property
    def radius(self) -> float:
        """Радиус окружности вокруг поля зрения с центром в его середине, клеток."""
        return math.hypot(self.height, self.width) / 2.0

    @property
    def middle(self) -> tuple[float, float]:
        """Середина поля зрения относительно узла (dy, dx)."""
        return (self.box[0] + self.box[1]) / 2.0, (self.box[2] + self.box[3]) / 2.0

    def sees(self, dy: float, dx: float) -> bool:
        return self.box[0] <= dy <= self.box[1] and self.box[2] <= dx <= self.box[3]

    def element_sees(self, dy: float, dx: float) -> bool:
        return self.inner[0] <= dy <= self.inner[1] and self.inner[2] <= dx <= self.inner[3]


@dataclass(frozen=True)
class LevelKeyholes:
    """Окна внимания уровня: куски картинок, сколько клеток видит каждое окно и мера плоскости уровня."""

    field: LadderField
    pieces: KeyholePieces
    extent: "np.ndarray"  # сколько клеток картинки видит каждое окно
    seen: float  # какую долю углопозиций своей картинки видит окно, в среднем по обучающим
    sight: float  # среднее поле окна обучающих картинок, клеток
    step: float  # мера плоскости уровня: что видит одна деталь, но не меньше шага между соседями
    taught: int  # окон на обучающих картинках

    @property
    def number(self) -> int:
        return self.field.number

    @property
    def count(self) -> int:
        return self.pieces.count


class KeyholeBuilder:
    """Окно внимания на каждом элементе уровня (П.2); радиус — по кластеру похожих соседей (§6.4.1–§6.4.3).

    Радиусы построения меряются в том, что видит одна деталь уровня: окно на карте равно полю своей детали,
    и окно следующего уровня охватывает более крупную область картинки (П.3).
    """

    def __init__(self, groups: DigitGroups) -> None:
        self._groups = groups
        self._scale = NeighbourScale(eps_factor=settings.LADDER_CLUSTER_EPS)

    def build(self, elements: LadderElements, number: int) -> LevelKeyholes:
        """Окна уровня number над элементами elements (углопозиции либо детали прошлого уровня); результат — из кеша
        шагов, если такой вход уже был (system/damp/cache.py)."""
        from damp.cache import CACHE

        atoms = self._groups.atoms
        inputs = {
            "picture": elements.picture, "y": elements.y, "x": elements.x, "what": elements.what_packed,
            "what_index": elements.what_index, "what_length": elements.what_length, "extent": elements.extent,
            "number": int(number), "train_pictures": self._groups.train_pictures,
            "atom_picture": atoms.picture, "atom_y": atoms.y, "atom_x": atoms.x,
        }
        found = CACHE.run("cycle.keyholes", inputs, lambda: self._pack(self._build(elements, number)), modules=("damp.cycle.keyholes",))
        return self._unpack(found)

    @staticmethod
    def _pack(keyholes: LevelKeyholes) -> dict[str, object]:
        pieces, field = keyholes.pieces, keyholes.field
        return {
            "offsets": pieces.offsets, "picture": pieces.picture, "centre": pieces.centre, "piece": pieces.piece,
            "element": pieces.element, "offset": pieces.offset, "extent": keyholes.extent,
            "seen": keyholes.seen, "sight": keyholes.sight, "step": keyholes.step, "taught": keyholes.taught,
            "number": field.number, "unit": field.unit, "box": list(field.box), "inner": list(field.inner),
        }

    @staticmethod
    def _unpack(found: dict[str, object]) -> LevelKeyholes:
        pieces = KeyholePieces(
            offsets=found["offsets"], picture=found["picture"], centre=found["centre"],
            piece=found["piece"], element=found["element"], offset=found["offset"],
        )
        field = LadderField(
            number=int(found["number"]), unit=float(found["unit"]), box=tuple(found["box"]), inner=tuple(found["inner"])
        )
        return LevelKeyholes(
            field=field, pieces=pieces, extent=found["extent"], seen=float(found["seen"]), sight=float(found["sight"]),
            step=float(found["step"]), taught=int(found["taught"]),
        )

    def _build(self, elements: LadderElements, number: int) -> LevelKeyholes:
        name = f"level{number}"
        train_pictures = self._groups.train_pictures
        taught = elements.picture < train_pictures
        previous = 0.5 if elements.extent is None else float(elements.reach()[taught].mean())
        neighbour_step, thresholds = self._scale.measure(
            elements.picture[taught],
            elements.y[taught],
            elements.x[taught],
            elements.what_packed,
            elements.what_index[taught],
            elements.what_length,
            tuple(settings.LADDER_CLUSTER_SIMILARITY_QUANTILES),
            name=name,
        )
        # Мера плоскости уровня — то, что видит одна его деталь, но не меньше шага между соседними
        # элементами. В ней заданы окрестность кластера, радиус активации и наименьший радиус окна (§6.4):
        # окно на карте равно полю своей детали. У углопозиций мера — одна клетка, и значения совпадают с
        # настройками детекторов над раскладкой.
        step = max(neighbour_step, float(np.median(elements.reach()[taught])))
        fields = PlaneFieldBuilder(
            lambda_thresholds=thresholds,
            activation_radius=settings.LADDER_CLUSTER_ACTIVATION_RADIUS * step,
            eps=settings.LADDER_CLUSTER_EPS * step,
            min_samples=settings.LADDER_CLUSTER_MIN_SAMPLES,
            min_points=settings.LADDER_CLUSTER_MIN_POINTS,
            radius_min=settings.LADDER_CLUSTER_RADIUS_MIN * step,
            energy_radius=settings.LADDER_CLUSTER_ENERGY_RADIUS * step,
            energy_threshold=settings.ACTIVATION_ENERGY_THRESHOLD,
            dense=True,
            name=name,
        ).build(elements.picture, elements.y, elements.x, elements.what_packed, elements.what_index, elements.what_length)
        pieces = fields.pieces
        taught_fields = int(np.count_nonzero(pieces.picture < train_pictures))
        # Сколько клеток картинки видит окно: его элементы вместе с тем, что видят они сами.
        reach = np.hypot(
            elements.y[pieces.element] - pieces.centre[pieces.piece, 0],
            elements.x[pieces.element] - pieces.centre[pieces.piece, 1],
        ) + elements.reach()[pieces.element]
        extent = np.zeros(pieces.count, dtype=np.float64)
        np.maximum.at(extent, pieces.piece, reach)
        if taught_fields < settings.LADDER_MAP_MIN_POINTS:
            # Окон на обучающих картинках слишком мало, чтобы из них сложилась память: уровень не строится.
            empty = LadderField(number=number, unit=float(settings.LADDER_CLUSTER_PLACE_UNIT), box=(0, 0, 0, 0), inner=(0, 0, 0, 0))
            return LevelKeyholes(field=empty, pieces=pieces, extent=extent, seen=0.0, sight=0.0, step=step, taught=taught_fields)
        half = int(math.ceil(float(extent.mean())))
        inner = int(math.ceil(float(elements.reach().mean())))
        field = LadderField(
            number=number,
            unit=float(settings.LADDER_CLUSTER_PLACE_UNIT),
            box=(-half, half, -half, half),
            inner=(-inner, inner, -inner, inner),
        )
        seen = self._picture_seen(pieces, extent, train_pictures)
        sight = float(extent[pieces.picture < train_pictures].mean())
        LOGGER.event(
            "ladder.keyhole",
            section=CONTINUATION_SPATIAL_CODES,
            data={
                "level": number,
                "neighbour_step": neighbour_step,
                "detail_sight": float(np.median(elements.reach()[taught])),
                "plane_step": step,
                "keyhole_radius": float(fields.radius.mean()) if fields.count else 0.0,
                "keyhole_radius_in_steps": float(fields.radius.mean()) / step if fields.count else 0.0,
                "field_radius": sight,
                "growth": sight / previous,
                "picture_seen": seen,
                "keyholes_per_picture": taught_fields / max(1, train_pictures),
                "note": "окно следующего уровня охватывает более крупную область карты (П.3): радиусы построения "
                "окна (§6.4) меряются в том, что видит одна деталь уровня, — окно на карте равно полю своей "
                "детали; growth — во сколько раз поле на картинке шире, чем у детали предыдущего уровня",
            },
        )
        return LevelKeyholes(field=field, pieces=pieces, extent=extent, seen=seen, sight=sight, step=step, taught=taught_fields)

    def _picture_seen(self, pieces: KeyholePieces, extent: "np.ndarray", pictures: int) -> float:
        """Какую долю углопозиций своей картинки видит окно, в среднем по окнам первых pictures картинок."""
        atoms = self._groups.atoms
        shares: list["np.ndarray"] = []
        starts = np.searchsorted(pieces.picture, np.arange(pictures + 1))
        for picture in range(pictures):
            rows = slice(int(starts[picture]), int(starts[picture + 1]))
            first, last = self._groups.sticks(picture)
            if rows.stop == rows.start or last == first:
                continue
            reach = np.hypot(
                atoms.y[first:last][None, :] - pieces.centre[rows, 0][:, None],
                atoms.x[first:last][None, :] - pieces.centre[rows, 1][:, None],
            )
            # Углопозиция видит свою клетку: она в окне, если её клетка задета кругом окна.
            shares.append((reach <= extent[rows][:, None]).mean(axis=1))
        return float(np.concatenate(shares).mean()) if shares else 0.0
