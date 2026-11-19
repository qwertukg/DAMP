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
    radius: float = 0.0  # средний радиус окна обучающих картинок, клеток (круг, в котором окно собирает элементы)
    thresholds: tuple[float, ...] = ()  # пороги слоёв построения окон — квантили сходства соседей обучающих (§6.4)

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
            "thresholds": [float(value) for value in keyholes.thresholds],
            "radius": float(keyholes.radius),
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
            thresholds=tuple(float(value) for value in found.get("thresholds", ())),
            radius=float(found.get("radius", 0.0)),
        )

    @staticmethod
    def min_points(number: int) -> int:
        """Сколько похожих соседей нужно окну уровня number (§6.4.3): LADDER_CLUSTER_MIN_POINTS — число на все уровни
        либо список по уровням 1, 2, … (последнее значение — на всех уровнях выше). Своё число у уровня 1 — по слову
        автора 2026-11-18: с уровнем 0 штрихи несут выученные коды, и окну уровня 1 может хватать меньше соседей."""
        value = settings.LADDER_CLUSTER_MIN_POINTS
        if isinstance(value, (list, tuple)):
            return int(value[min(max(int(number), 1), len(value)) - 1])
        return int(value)

    @classmethod
    def fields(cls, step: float, thresholds: "tuple[float, ...] | list[float]", name: str, number: int = 1) -> PlaneFieldBuilder:
        """Построитель окон уровня number по его мере плоскости step и порогам слоёв (§6.4.1–§6.4.3): окрестность
        кластера, радиус активации и наименьший радиус окна заданы в мере «что видит одна деталь уровня»."""
        return PlaneFieldBuilder(
            lambda_thresholds=thresholds,
            activation_radius=settings.LADDER_CLUSTER_ACTIVATION_RADIUS * step,
            eps=settings.LADDER_CLUSTER_EPS * step,
            min_samples=settings.LADDER_CLUSTER_MIN_SAMPLES,
            min_points=cls.min_points(number),
            radius_min=settings.LADDER_CLUSTER_RADIUS_MIN * step,
            energy_radius=settings.LADDER_CLUSTER_ENERGY_RADIUS * step,
            energy_threshold=settings.ACTIVATION_ENERGY_THRESHOLD,
            dense=True,
            name=name,
            concepts=settings.LADDER_KEYHOLE_CONCEPTS,  # П.7: окно объединяет около 5–7 понятий
            # Нижний предел радиуса окна — доля того, что видит одна деталь уровня: от неё зависит рост поля за уровень.
            radius_floor=settings.LADDER_KEYHOLE_RADIUS_SHARE * step,
            # Картинка без окон не остаётся: деталей меньше, чем нужно кластеру, — окно видит её целиком (§6.6).
            whole_picture=settings.LADDER_KEYHOLE_WHOLE_PICTURE,
        )

    @staticmethod
    def extent(elements: LadderElements, pieces: KeyholePieces) -> "np.ndarray":
        """Сколько клеток картинки видит каждое окно: его элементы вместе с тем, что видят они сами."""
        reach = np.hypot(
            elements.y[pieces.element] - pieces.centre[pieces.piece, 0],
            elements.x[pieces.element] - pieces.centre[pieces.piece, 1],
        ) + elements.reach()[pieces.element]
        extent = np.zeros(pieces.count, dtype=np.float64)
        np.maximum.at(extent, pieces.piece, reach)
        return extent

    @classmethod
    def place(
        cls, elements: LadderElements, *, number: int, step: float, thresholds: "tuple[float, ...]"
    ) -> tuple[KeyholePieces, "np.ndarray"]:
        """Окна новой картинки на обученном уровне number (П.2): мера плоскости step и пороги слоёв thresholds — те,
        что сняты с обучающих картинок, заново не меряются. Окна картинки строятся независимо от других картинок,
        поэтому у картинки прогона здесь те же окна, что в прогоне. Возвращает куски и поле каждого окна."""
        built = cls.fields(step, thresholds, f"level{number}", number).build(
            elements.picture, elements.y, elements.x, elements.what_packed, elements.what_index, elements.what_length
        )
        return built.pieces, cls.extent(elements, built.pieces)

    def _build(self, elements: LadderElements, number: int) -> LevelKeyholes:
        name = f"level{number}"
        train_pictures = self._groups.train_pictures
        taught = elements.picture < train_pictures
        previous = 0.5 if elements.extent is None else float(elements.reach()[taught].mean())
        # Шаг между соседями и пары соседей меряются по сетке окон Собеля: углопозиции стоят по одной на окно, а
        # элементы всех уровней — на местах углопозиций (окно стоит на своём элементе, П.2). Дробное место
        # (MNIST_SUBCELL_POSITION, вне статьи) уточняет, где край проходит внутри окна, но шаг выборки не меняет:
        # у углопозиций шаг — одна клетка. По дробным местам соседи на косом крае сближаются (медиана 0.68 клетки),
        # и все радиусы построения окон сжимаются. При целом месте округление ничего не меняет.
        neighbour_step, thresholds = self._scale.measure(
            elements.picture[taught],
            np.rint(elements.y[taught]),
            np.rint(elements.x[taught]),
            elements.what_packed,
            elements.what_index[taught],
            elements.what_length,
            tuple(settings.LADDER_CLUSTER_SIMILARITY_QUANTILES),
            name=name,
        )
        thresholds = tuple(float(value) for value in thresholds)
        # Мера плоскости уровня — то, что видит одна его деталь, но не меньше шага между соседними
        # элементами. В ней заданы окрестность кластера, радиус активации и наименьший радиус окна (§6.4):
        # окно на карте равно полю своей детали. У углопозиций мера — одна клетка, и значения совпадают с
        # настройками детекторов над раскладкой.
        step = max(neighbour_step, float(np.median(elements.reach()[taught])))
        fields = self.fields(step, thresholds, name, number).build(
            elements.picture, elements.y, elements.x, elements.what_packed, elements.what_index, elements.what_length
        )
        pieces = fields.pieces
        taught_fields = int(np.count_nonzero(pieces.picture < train_pictures))
        extent = self.extent(elements, pieces)
        if taught_fields < settings.LADDER_MAP_MIN_POINTS:
            # Окон на обучающих картинках слишком мало, чтобы из них сложилась память: уровень не строится.
            empty = LadderField(number=number, unit=float(settings.LADDER_CLUSTER_PLACE_UNIT), box=(0, 0, 0, 0), inner=(0, 0, 0, 0))
            return LevelKeyholes(field=empty, pieces=pieces, extent=extent, seen=0.0, sight=0.0, step=step, taught=taught_fields,
                                 thresholds=thresholds)
        # Масштаб мест — только по обучающим окнам и элементам: отложенные, проверочные и сдвинутые картинки
        # не должны влиять на коды обучающих окон (честная мерка, вне статьи).
        half = int(math.ceil(float(extent[pieces.picture < train_pictures].mean())))
        inner = int(math.ceil(float(elements.reach()[taught].mean())))
        field = LadderField(
            number=number,
            unit=float(settings.LADDER_CLUSTER_PLACE_UNIT),
            box=(-half, half, -half, half),
            inner=(-inner, inner, -inner, inner),
        )
        seen = self._picture_seen(pieces, extent, train_pictures)
        sight = float(extent[pieces.picture < train_pictures].mean())
        radius = float(fields.radius[pieces.picture < train_pictures].mean())
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
        return LevelKeyholes(field=field, pieces=pieces, extent=extent, seen=seen, sight=sight, step=step, taught=taught_fields,
                             thresholds=thresholds, radius=radius)

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
