"""Этапы 2–3 как «уровень 0» обученного цикла: первичный код углопозиции и его раскладка — карта с пинвилами.

Углопозиция — стимул (угол, X, Y) (§4.4, §4.6). Её первичный код — биты широких детекторов трёх координат одним
общим кодом (§2.2.3, §4.2.1): тот же кодер, что у первой раскладки (encoding/mnist_codes.py). Раскладка таких кодов
(§5) ставит рядом углопозиции с близкими углом и местом; на карте получаются пинвилы — участки, где направление
плавно обходит круг (§5.13.2, рис. 16).

В счёте уровней первичная карта не участвует: окно уровня 1 берёт у углопозиции только код направления (StickCoder), а
место записывает само (П.4, П.5). Карта нужна, чтобы видеть первичное кодирование: какие её точки зажигают штрихи
рисунка (§6.1) и какие детекторы над ней срабатывают (§6.5).

Карта строится теми же классами, что карты уровней: образцы — MemoryMap.samples (предел точек, углопозиции обходятся
вразброс; сливаются коды со сходством не ниже LADDER_PRIMARY_MERGE_SIMILARITY), раскладка — тот же Layout
(MemoryMap.remember), детекторы — LevelDetectors.

У штрихов по середине линии направление без стрелки: период угла — 180°, как у StickCoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from damp import settings
from damp.activation.details import DetailCoder
from damp.activation.level_detectors import LevelDetectors
from damp.article_refs import LAID_OUT_STRUCTURE, MORPHOLOGICAL_EMBEDDINGS
from damp.cycle.keyholes import LadderField, LevelKeyholes
from damp.cycle.plane_fields import PlaneFieldBuilder
from damp.cycle.step import CycleStep, LadderLevelDegenerate
from damp.encoding.damp_encoder import Encoder
from damp.encoding.elements import LadderElements, StickCoder, distinct_codes
from damp.encoding.keyhole import KeyholePieces
from damp.encoding.keyhole_code import KeyholeCoder
from damp.encoding.mnist_codes import _build_encoder
from damp.layout.code_table import LayoutCodeTable
from damp.layout.level_memory import LevelMemory, MemoryMap
from damp.logging import LOGGER
from damp.stimuli.digits import DigitGroups


@dataclass(frozen=True)
class PrimarySight:
    """Что первичная карта сделала с углопозициями картинки: зажжённые точки (§6.1) и сработавшие детекторы (§6.5)."""

    lit_stick: "np.ndarray"  # (L,) углопозиция картинки, зажёгшая точку карты
    lit_point: "np.ndarray"  # (L,) точка первичной карты (порядок точек раскладки)
    lit_value: "np.ndarray"  # (L,) её активация a_j
    fired_stick: "np.ndarray"  # (R,) углопозиция, на которой сработал детектор
    fired_detector: "np.ndarray"  # (R,) детектор первичной карты
    fired_level: "np.ndarray"  # (R,) его уровень активации E(d)


class PrimaryLevel:
    """Уровень 0: первичная карта углопозиций обучающих цифр и детекторы над ней (§4.6, §5, §6.4).

    memory — карта образцов первичных кодов; detectors — детекторы над ней; point_* — углопозиция обучающей картинки,
    чей код хранит точка карты (порядок точек раскладки): картинка, угол и место на сетке картинки.
    """

    NAME = "level0"
    # Углопозиций за один проход кодера: матрица «стимулы × биты» целиком не создаётся.
    CHUNK = 65536

    def __init__(
        self,
        encoder: Encoder,
        memory: LevelMemory,
        detectors: LevelDetectors,
        *,
        point_picture: "np.ndarray",
        point_angle: "np.ndarray",
        point_y: "np.ndarray",
        point_x: "np.ndarray",
    ) -> None:
        self.encoder = encoder
        self.memory = memory
        self.detectors = detectors
        self.point_picture = np.asarray(point_picture, dtype=np.int64)
        self.point_angle = np.asarray(point_angle, dtype=np.float64)
        self.point_y = np.asarray(point_y, dtype=np.float64)
        self.point_x = np.asarray(point_x, dtype=np.float64)

    @staticmethod
    def axial() -> bool:
        """Направление без стрелки: штрихи по середине линии (как у StickCoder)."""
        return bool(settings.MNIST_STROKE_DETECTORS) and bool(settings.MNIST_STROKE_MIDLINE)

    @classmethod
    def build_encoder(cls) -> Encoder:
        """Кодер первичного стимула (угол, X, Y) — тот же, что у первой раскладки; у штрихов без стрелки период угла 180°."""
        span = float(settings.MNIST_ANGLE_MAX) - float(settings.MNIST_ANGLE_MIN)
        return _build_encoder(angle_span=span / 2.0 if cls.axial() else None)

    @classmethod
    def codes(cls, encoder: Encoder, angle: "np.ndarray", y: "np.ndarray", x: "np.ndarray") -> "np.ndarray":
        """Упакованные первичные коды углопозиций (§4.6): строка на углопозицию, порядок стимула — угол, X, Y."""
        rows = np.column_stack([np.asarray(angle, dtype=np.float64), np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)])
        if rows.shape[0] == 0:
            return np.zeros((0, -(-int(encoder.code_length) // 8)), dtype=np.uint8)
        return np.concatenate(
            [LayoutCodeTable.pack_bits(encoder.encode_matrix(rows[first : first + cls.CHUNK])) for first in range(0, rows.shape[0], cls.CHUNK)]
        )

    @classmethod
    def build(cls, groups: DigitGroups, *, max_points: int | None) -> "PrimaryLevel":
        """Первичная карта по углопозициям обучающих картинок; max_points — предел точек карты (None — без предела)."""
        atoms = groups.atoms
        taught = int(groups.train_pictures)
        train = np.flatnonzero(np.asarray(atoms.picture) < taught)
        if train.shape[0] == 0:
            raise LadderLevelDegenerate("уровень 0: у обучающих картинок нет углопозиций")
        encoder = cls.build_encoder()
        packed = cls.codes(encoder, atoms.angle[train], atoms.y[train], atoms.x[train])
        picture = np.asarray(atoms.picture)[train]
        mapper = MemoryMap()
        # Как у карт уровней: с пределом карты углопозиции обходятся вразброс, чтобы на карту попали штрихи всех цифр.
        dictionary, order = mapper.samples(
            packed,
            int(encoder.code_length),
            picture,
            taught,
            merge_similarity=float(settings.LADDER_PRIMARY_MERGE_SIMILARITY),
            max_points=int(settings.LADDER_DICTIONARY_MAX if max_points is None else max_points),
            shuffle_seed=None if max_points is None else int(settings.LADDER_RANDOM_SEED),
        )
        if dictionary.count < settings.LADDER_MAP_MIN_POINTS:
            raise LadderLevelDegenerate(f"уровень 0: на карте {dictionary.count} точек из {dictionary.distinct} различных кодов")
        rows = order[dictionary.rows]  # номер углопозиции (среди обучающих) у каждого образца
        memory = mapper.remember(dictionary, rows, np.asarray(groups.labels)[picture[rows]], cls.NAME)
        try:
            detectors = LevelDetectors.build(memory)
        except RuntimeError as error:
            memory.release()
            raise LadderLevelDegenerate(f"уровень 0: над картой из {memory.points} точек {error}") from error
        from damp.layout.glyph_map import LayoutGlyphMap

        # Точка карты — образец словаря: углопозиция с тем же кодом (раскладка держит точки в своём порядке).
        sample = LayoutGlyphMap(memory.layout, memory.dictionary.packed).rows
        stick = train[np.asarray(memory.rows, dtype=np.int64)[np.maximum(sample, 0)]]
        level = cls(
            encoder,
            memory,
            detectors,
            point_picture=np.where(sample >= 0, np.asarray(atoms.picture)[stick], -1),
            point_angle=np.asarray(atoms.angle)[stick],
            point_y=np.asarray(atoms.y)[stick],
            point_x=np.asarray(atoms.x)[stick],
        )
        LOGGER.event(
            "primary.map",
            section=LAID_OUT_STRUCTURE,
            data={
                "sticks": int(train.shape[0]),
                "points": int(memory.points),
                "width": int(memory.layout.width),
                "height": int(memory.layout.height),
                "detectors": int(detectors.built),
                "code_length": int(encoder.code_length),
                "angle_period": float(settings.MNIST_ANGLE_MAX - settings.MNIST_ANGLE_MIN) / (2.0 if cls.axial() else 1.0),
                "max_points": None if max_points is None else int(max_points),
                "note": "уровень 0 — первичная карта: раскладка кодов углопозиций (угол, X, Y) обучающих цифр тем же "
                "Layout, что карты уровней (§5); рядом стоят штрихи с близкими углом и местом — пинвилы (§5.13.2, §5.14). "
                "В счёте уровней не участвует: показывает первичное кодирование (этапы 2–3)",
            },
        )
        return level

    def see(self, angle: "np.ndarray", y: "np.ndarray", x: "np.ndarray") -> PrimarySight:
        """Углопозиции картинки на первичной карте: зажжённые точки (§6.1) и сработавшие детекторы (§6.5)."""
        packed = self.codes(self.encoder, angle, y, x)
        if packed.shape[0] == 0:
            empty, nothing = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
            return PrimarySight(empty, empty, nothing, empty, empty, nothing)
        firing, (lit_stick, lit_point, lit_value) = self.detectors.fire_and_light(packed)
        return PrimarySight(
            lit_stick=np.asarray(lit_stick, dtype=np.int64),
            lit_point=np.asarray(lit_point, dtype=np.int64),
            lit_value=np.asarray(lit_value, dtype=np.float32),
            fired_stick=np.asarray(firing.rows, dtype=np.int64),
            fired_detector=np.asarray(firing.columns, dtype=np.int64),
            fired_level=np.asarray(firing.levels, dtype=np.float32),
        )

    def release(self) -> None:
        """Освободить память видеокарты под первичной картой и её детекторами."""
        self.detectors.release()
        self.memory.release()


@dataclass(frozen=True)
class StrokePoints:
    """Точки, из которых детекторы собрали штрихи (середина линии либо край), — элементы уровня 0 по общему подходу
    (LADDER_LEVEL0_PLACE = "stroke"). Штрих — детектор над точками картинки по §6.4 (доработка: §6.4 на плоскости
    картинки); его поле — точки кластера не дальше шага от точки-центра (§6.4.3). Места — в клетках, как у штрихов.
    """

    picture: "np.ndarray"  # (P,) картинка точки
    y: "np.ndarray"  # (P,) место точки, в клетках
    x: "np.ndarray"
    angle: "np.ndarray"  # (P,) направление точки — в тех же единицах, что угол штриха
    strength: "np.ndarray"  # (P,) сила точки, доля сильнейшей на своей картинке
    entry_stroke: "np.ndarray"  # (E,) номер штриха в порядке штрихов прогона, по возрастанию
    entry_point: "np.ndarray"  # (E,) номер точки его поля

    # Разбор картинки на штрихи и точки: от текста этих модулей зависит результат шага кеша.
    MODULES = (
        "damp.stimuli.MnistSobelAngleMap", "damp.stimuli.stroke_detectors", "damp.stimuli.stroke_midline",
        "damp.stimuli.mnist", "damp.cycle.primary",
    )
    # Допуск сверки пересчитанных штрихов со штрихами прогона, клеток.
    MATCH = 1e-6

    @classmethod
    def _parse(cls, extractor, pictures: "list[np.ndarray]") -> dict[str, "np.ndarray"]:  # noqa: ANN001
        """Штрихи и точки картинок (пиксели float32 в [0, 1]): всё одним набором, номер картинки — номер в списке."""
        keys = ("stroke_picture", "stroke_y", "stroke_x", "point_picture", "point_y", "point_x", "point_angle",
                "point_strength", "entry_stroke", "entry_point")
        parts: dict[str, list] = {key: [] for key in keys}
        strokes = points = 0
        with LOGGER.quiet():
            for number, pixels in enumerate(pictures):
                found = extractor.stroke_points(pixels)
                rows, count = found["strokes"], int(found["point_y"].shape[0])
                parts["stroke_picture"].append(np.full(rows.shape[0], number, dtype=np.int64))
                parts["stroke_y"].append(rows[:, 2])
                parts["stroke_x"].append(rows[:, 1])
                parts["point_picture"].append(np.full(count, number, dtype=np.int64))
                parts["point_y"].append(found["point_y"])
                parts["point_x"].append(found["point_x"])
                parts["point_angle"].append(found["point_angle"])
                parts["point_strength"].append(found["point_strength"])
                parts["entry_stroke"].append(found["entry_stroke"] + strokes)
                parts["entry_point"].append(found["entry_point"] + points)
                strokes, points = strokes + rows.shape[0], points + count
        integer = {"stroke_picture", "point_picture", "entry_stroke", "entry_point"}
        return {
            key: (np.concatenate(values) if values else np.zeros(0)).astype(np.int64 if key in integer else np.float64)
            for key, values in parts.items()
        }

    @classmethod
    def of_images(cls, images: "np.ndarray", sticks: LadderElements) -> "StrokePoints":
        """Точки штрихов всех картинок прогона: images — N × 28 × 28 (0…255), номер картинки — номер строки, как у
        sticks. Разбор — тот же, что у углопозиций (штрихи пересчитываются и сверяются со sticks); из кеша шагов."""
        from damp.cache import CACHE
        from damp.stimuli.mnist import _build_extractor

        images = np.ascontiguousarray(images, dtype=np.uint8)

        def compute() -> dict[str, "np.ndarray"]:
            # Пиксели — как у загрузки MNIST (ToTensor: uint8 / 255 во float32): штрихи выходят те же.
            return cls._parse(_build_extractor(), [image.astype(np.float32) / np.float32(255.0) for image in images])

        return cls._aligned(CACHE.run("stimuli.stroke_points", {"images": images}, compute, modules=cls.MODULES), sticks)

    @classmethod
    def of_pixels(cls, pixels: "np.ndarray", sticks: LadderElements) -> "StrokePoints":
        """Точки штрихов одной новой картинки (пиксели float32 в [0, 1] — те же, по которым найдены sticks)."""
        from damp.stimuli.mnist import _build_extractor

        found = cls._parse(_build_extractor(), [np.asarray(pixels, dtype=np.float32)])
        own = int(sticks.picture[0]) if sticks.count else 0
        for key in ("stroke_picture", "point_picture"):
            found[key] = np.full(found[key].shape[0], own, dtype=np.int64)
        return cls._aligned(found, sticks)

    @classmethod
    def _aligned(cls, found: dict, sticks: LadderElements) -> "StrokePoints":  # noqa: ANN001
        """Сверка пересчитанных штрихов со штрихами прогона: те же места в том же порядке (по картинкам)."""
        picture = np.asarray(sticks.picture, dtype=np.int64)
        same = (
            found["stroke_y"].shape[0] == sticks.count
            and np.array_equal(found["stroke_picture"], picture)
            and np.allclose(found["stroke_y"], sticks.y, atol=cls.MATCH, rtol=0.0)
            and np.allclose(found["stroke_x"], sticks.x, atol=cls.MATCH, rtol=0.0)
        )
        if not same:
            raise ValueError("уровень 0: штрихи, пересчитанные по картинкам, не совпали со штрихами прогона — разбор "
                             "углопозиций и разбор точек разошлись (устаревший кеш углопозиций?)")
        return cls(
            picture=found["point_picture"], y=found["point_y"], x=found["point_x"], angle=found["point_angle"],
            strength=found["point_strength"], entry_stroke=found["entry_stroke"], entry_point=found["entry_point"],
        )


class StrokeLevel:
    """Уровень 0 цикла — первичная раскладка штрихов, с которой начинается цикл (§6.5, §7.1.12; П.4, П.5).

    Тот же проход, что у уровней цикла (CycleStep): окно → код «что и где» → память → детекторы → деталь; кусок
    уровня 0 — один штрих (атом), окно на каждом штрихе. Что лежит в окне — LADDER_LEVEL0_PLACE:

    - "stroke" — общий подход (замысел автора 2026-11-19: «подход на всех уровнях одинаковый, память начинается с
      самих атомов»): элементы окна — точки, из которых детектор собрал штрих (его поле, §6.4.3), центр окна — сам
      штрих (§6.4.2), R — до самой дальней точки поля, место точки — смещение / R теми же кругами места (П.4). Код
      штриха — направления его точек под масками их мест (П.5): форма штриха внутри поля (прямой, изгиб, конец,
      перекрестие). Где штрих стоит в цифре, не записано: это дают окна уровня 1 — места соседей (П.4).
    - "digit" — вариант 3 (автор 2026-11-18): в окне один штрих, центр — центр цифры (середина её штрихов), R — до
      самого дальнего штриха, место штриха — от центра цифры (центр — доработка).

    Сдвиг цифры коды штрихов не меняет. Код детали штриха — сильнейшие детекторы карты уровня 0 (§6.5, §7.1.12): с ним,
    а не с готовым кодом наклона (StickCoder), штрих идёт в окна уровня 1. Место штриха, его поле и сила — прежние.

    Для дашборда — тот же вид, что у PrimaryLevel: memory, detectors, point_* (штрих обучающей картинки каждой точки
    карты) и axial().
    """

    NUMBER = 0
    NAME = "level0"

    def __init__(
        self,
        memory: LevelMemory,
        detectors: LevelDetectors,
        *,
        unit: float,
        content_length: int,
        detail_length: int,
        scale: "np.ndarray | None",
        point_picture: "np.ndarray",
        point_angle: "np.ndarray",
        point_y: "np.ndarray",
        point_x: "np.ndarray",
        place: str = "digit",
    ) -> None:
        self.memory = memory
        self.detectors = detectors
        self.place = str(place)  # что лежит в окне уровня 0 (LADDER_LEVEL0_PLACE на момент обучения)
        self.unit = float(unit)
        self.content_length = int(content_length)
        self.detail_length = int(detail_length)
        self.scale = scale
        self.point_picture = np.asarray(point_picture, dtype=np.int64)
        self.point_angle = np.asarray(point_angle, dtype=np.float64)
        self.point_y = np.asarray(point_y, dtype=np.float64)
        self.point_x = np.asarray(point_x, dtype=np.float64)
        self._coder = KeyholeCoder()

    @staticmethod
    def axial() -> bool:
        """Направление без стрелки: штрихи по середине линии (как у StickCoder)."""
        return PrimaryLevel.axial()

    @staticmethod
    def pieces(sticks: LadderElements) -> KeyholePieces:
        """Окна уровня 0: окно на каждом штрихе, в окне — он один; центр окна — середина штрихов своей картинки,
        смещение штриха — в долях расстояния до самого дальнего штриха картинки (как у окон уровней: место — доля
        радиуса окна, PlaneFieldBuilder.OFFSET_STEP)."""
        picture = np.asarray(sticks.picture, dtype=np.int64)
        count = int(picture.shape[0])
        if count == 0:
            nothing = np.zeros(0, dtype=np.int64)
            return KeyholePieces(offsets=np.zeros((0, 2), dtype=np.float64), picture=nothing, centre=np.zeros((0, 2), dtype=np.float64),
                                 piece=nothing, element=nothing, offset=nothing)
        y, x = np.asarray(sticks.y, dtype=np.float64), np.asarray(sticks.x, dtype=np.float64)
        _, inverse = np.unique(picture, return_inverse=True)
        inverse = inverse.reshape(-1)
        sizes = np.bincount(inverse).astype(np.float64)
        centre_y, centre_x = np.bincount(inverse, y) / sizes, np.bincount(inverse, x) / sizes
        dy, dx = y - centre_y[inverse], x - centre_x[inverse]
        radius = np.zeros(sizes.shape[0], dtype=np.float64)
        np.maximum.at(radius, inverse, np.hypot(dy, dx))
        radius = np.where(radius > 0.0, radius, 1.0)
        step = PlaneFieldBuilder.OFFSET_STEP
        shift = np.round(np.stack([dy, dx], axis=1) / radius[inverse][:, None] / step) * step
        offsets, offset = np.unique(shift, axis=0, return_inverse=True)
        every = np.arange(count, dtype=np.int64)
        return KeyholePieces(
            offsets=offsets.astype(np.float64),
            picture=picture,
            centre=np.stack([centre_y, centre_x], axis=1)[inverse],
            piece=every,
            element=every.copy(),
            offset=offset.reshape(-1).astype(np.int64),
        )

    @staticmethod
    def point_pieces(sticks: LadderElements, points: StrokePoints) -> KeyholePieces:
        """Окна уровня 0 по общему подходу (LADDER_LEVEL0_PLACE = "stroke"): окно на каждом штрихе, в окне — точки его
        поля (§6.4.3); центр окна — сам штрих (его взвешенный центр, §6.4.2), R — до самой дальней точки поля, место
        точки — смещение / R с шагом PlaneFieldBuilder.OFFSET_STEP (как у окон уровней, П.4)."""
        stroke = np.asarray(points.entry_stroke, dtype=np.int64)
        point = np.asarray(points.entry_point, dtype=np.int64)
        dy = np.asarray(points.y, dtype=np.float64)[point] - np.asarray(sticks.y, dtype=np.float64)[stroke]
        dx = np.asarray(points.x, dtype=np.float64)[point] - np.asarray(sticks.x, dtype=np.float64)[stroke]
        radius = np.zeros(sticks.count, dtype=np.float64)
        np.maximum.at(radius, stroke, np.hypot(dy, dx))
        radius = np.where(radius > 0.0, radius, 1.0)
        step = PlaneFieldBuilder.OFFSET_STEP
        shift = np.round(np.stack([dy, dx], axis=1) / radius[stroke][:, None] / step) * step
        offsets, offset = np.unique(shift.reshape(-1, 2), axis=0, return_inverse=True)
        return KeyholePieces(
            offsets=offsets.astype(np.float64),
            picture=np.asarray(sticks.picture, dtype=np.int64),
            centre=np.stack([np.asarray(sticks.y, dtype=np.float64), np.asarray(sticks.x, dtype=np.float64)], axis=1),
            piece=stroke,
            element=point,
            offset=offset.reshape(-1).astype(np.int64),
        )

    @classmethod
    def inputs(
        cls, sticks: LadderElements, place: str, *, images: "np.ndarray | None" = None, pixels: "np.ndarray | None" = None
    ) -> "tuple[LadderElements, KeyholePieces, int]":
        """Элементы окон уровня 0 и сами окна по месту place (LADDER_LEVEL0_PLACE); третье — сколько точек.

        "digit" — штрихи, окно на штрихе с местом от центра цифры (pieces); "stroke" — точки штрихов картинок (images —
        все картинки прогона, pixels — одна новая картинка) и окно-штрих над ними (point_pieces).
        """
        if place == "digit":
            return sticks, cls.pieces(sticks), 0
        if place != "stroke":
            raise ValueError(f"LADDER_LEVEL0_PLACE: stroke либо digit, а не {place!r}")
        if pixels is not None:
            points = StrokePoints.of_pixels(pixels, sticks)
        elif images is not None:
            points = StrokePoints.of_images(images, sticks)
        else:
            raise ValueError("уровню 0 по точкам нужны картинки: images (прогон) либо pixels (одна картинка)")
        elements = StickCoder().points(points.picture, points.y, points.x, points.angle, points.strength)
        return elements, cls.point_pieces(sticks, points), int(points.picture.shape[0])

    @staticmethod
    def learned(sticks: LadderElements, details: "np.ndarray", length: int) -> LadderElements:
        """Штрихи с выученными кодами — элементы уровня 1: место, поле и сила те же, «что» — код детали уровня 0."""
        what, what_index = distinct_codes(details)
        return LadderElements(
            picture=sticks.picture, y=sticks.y, x=sticks.x, what_packed=what, what_length=int(length), what_index=what_index,
            extent=sticks.extent, strength=sticks.strength,
        )

    @classmethod
    def train(cls, groups: DigitGroups, sticks: LadderElements, *, map_max_points: int | None) -> "tuple[StrokeLevel, LadderElements]":
        """Уровень 0 по штрихам всех картинок прогона (память и подбор — только по обучающим) и штрихи с выученными
        кодами — вход уровня 1. Шаги — те же, что у прохода уровня (CycleStep), и из кеша шагов."""
        step = CycleStep(groups, map_max_points=map_max_points)
        place = str(settings.LADDER_LEVEL0_PLACE)
        elements, pieces, point_count = cls.inputs(sticks, place, images=groups.atoms.images)
        trained = np.asarray(pieces.picture) < int(groups.train_pictures)
        taught = int(np.count_nonzero(trained))
        if taught < settings.LADDER_MAP_MIN_POINTS:
            raise LadderLevelDegenerate(f"уровень 0: штрихов на обучающих картинках — {taught}")
        unit = float(settings.LADDER_CLUSTER_PLACE_UNIT)
        codes = step.coder.encode(elements, pieces, number=cls.NUMBER, unit=unit, taught=taught)
        field = LadderField(number=cls.NUMBER, unit=unit, box=(0, 0, 0, 0), inner=(0, 0, 0, 0))
        keyholes = LevelKeyholes(field=field, pieces=pieces, extent=np.asarray(sticks.reach(), dtype=np.float64), seen=0.0, sight=0.0,
                                 step=1.0, taught=taught)
        # Первичные коды соседних штрихов похожи сильнее кодов окон: порог слияния — свой, как у первичной карты.
        memory = step.remember(codes, keyholes, merge_similarity=float(settings.LADDER_PRIMARY_MERGE_SIMILARITY))
        detectors = step.detect(memory)
        firing = step.fire(detectors, codes, trained)
        details, length = DetailCoder(cls.NUMBER).encode(firing, detectors.layer_index, detectors.built, trained)
        from damp.layout.glyph_map import LayoutGlyphMap

        # Точка карты — образец словаря: штрих с тем же кодом (окно уровня 0 — один штрих).
        sample = LayoutGlyphMap(memory.layout, memory.dictionary.packed).rows
        stick = np.asarray(memory.rows, dtype=np.int64)[np.maximum(sample, 0)]
        atoms = groups.atoms
        level = cls(
            memory, detectors, unit=unit, content_length=int(codes.content_length), detail_length=int(length), scale=firing.scale,
            point_picture=np.where(sample >= 0, np.asarray(pieces.picture)[stick], -1),
            point_angle=np.asarray(atoms.angle)[stick], point_y=np.asarray(sticks.y)[stick], point_x=np.asarray(sticks.x)[stick],
            place=place,
        )
        learned = cls.learned(sticks, details, length)
        silent = np.ones(codes.count, dtype=bool)
        silent[np.asarray(firing.rows, dtype=np.int64)] = False
        LOGGER.event(
            "level0.strokes",
            section=MORPHOLOGICAL_EMBEDDINGS,
            data={
                "strokes": int(codes.count),
                "taught": taught,
                "points": int(memory.points),
                "detectors": int(detectors.built),
                "code_length": int(codes.code_length),
                "detail_length": int(length),
                "silent_share": float(silent[trained].mean()) if taught else 0.0,
                "angle_codes": int(sticks.what_packed.shape[0]),
                "learned_codes": int(learned.what_packed.shape[0]),
                "place": place,
                "stroke_points": point_count,
                "points_per_stroke": float(len(pieces.piece) / max(1, codes.count)),
                "note": ("уровень 0 по общему подходу: окно на штрихе, в нём точки его поля (§6.4.3), центр — сам штрих, "
                         "место точки — от центра штриха кругами места (П.4), код — направления точек под масками мест "
                         "(П.5)" if place == "stroke" else
                         "уровень 0 — вариант 3: окно на штрихе с центром в центре цифры, место — от центра цифры "
                         "кругами места (П.4; центр — доработка)")
                + "; память, детекторы (§6.4) и код штриха — его сильнейшие детекторы (§6.5, §7.1.12); с ним штрих "
                "идёт в окна уровня 1",
            },
        )
        return level, learned

    def embed(self, sticks: LadderElements, *, pixels: "np.ndarray | None" = None) -> "tuple[PrimarySight, LadderElements]":
        """Штрихи новой картинки через обученный уровень 0 — те же шаги, что в train, без подбора: зажжённые точки и
        сработавшие детекторы (для показа) и штрихи с выученными кодами — вход уровня 1. pixels — сама картинка
        (float32 в [0, 1], по ней найдены sticks): уровню 0 по точкам (place "stroke") нужны точки её штрихов."""
        elements, pieces, _ = self.inputs(sticks, self.place, pixels=pixels)
        if pieces.count == 0:
            empty, nothing = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
            return PrimarySight(empty, empty, nothing, empty, empty, nothing), sticks
        codes = self._coder.encode(elements, pieces, number=self.NUMBER, unit=self.unit, taught=pieces.count,
                                   content_length=self.content_length)
        firing, (lit_stick, lit_point, lit_value) = self.detectors.fire_and_light(codes.packed)
        if self.scale is not None:
            firing = firing.scaled(self.scale)  # нормировка — делителями обучающих штрихов (вне статьи)
        details, length = DetailCoder(self.NUMBER).encode(
            firing, self.detectors.layer_index, self.detectors.built, np.ones(codes.count, dtype=bool), length=self.detail_length
        )
        sight = PrimarySight(
            lit_stick=np.asarray(lit_stick, dtype=np.int64),
            lit_point=np.asarray(lit_point, dtype=np.int64),
            lit_value=np.asarray(lit_value, dtype=np.float32),
            fired_stick=np.asarray(firing.rows, dtype=np.int64),
            fired_detector=np.asarray(firing.columns, dtype=np.int64),
            fired_level=np.asarray(firing.levels, dtype=np.float32),
        )
        return sight, self.learned(sticks, details, length)

    def release(self) -> None:
        """Освободить память видеокарты под картой уровня 0 и её детекторами."""
        self.detectors.release()
        self.memory.release()
