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
from damp.activation.level_detectors import LevelDetectors
from damp.article_refs import LAID_OUT_STRUCTURE
from damp.cycle.step import LadderLevelDegenerate
from damp.encoding.damp_encoder import Encoder
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
