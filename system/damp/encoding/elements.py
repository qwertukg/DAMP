"""Этап 2, начало цикла: элементы уровня и код «что» углопозиции (детекторы угла, §4.4; деталь — §6.5, П.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from damp import settings
from damp.article_refs import OUTSIDE_ARTICLE
from damp.encoding.damp_encoder import ClosedDimension, Encoder, geometric_detector_layers
from damp.layout.code_table import LayoutCodeTable
from damp.logging import LOGGER

if TYPE_CHECKING:
    from damp.stimuli.digits import DigitAtoms


def packed_ones(packed: "np.ndarray") -> "np.ndarray":
    """Число единиц каждого упакованного кода (§2.2.4)."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(packed).sum(axis=1, dtype=np.int64)
    return np.unpackbits(packed, axis=1).sum(axis=1, dtype=np.int64)


def distinct_codes(packed: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
    """Различные коды и номер кода у каждой строки."""
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    view = packed.view(np.dtype((np.void, packed.shape[1]))).reshape(-1)
    _, first, inverse = np.unique(view, return_index=True, return_inverse=True)
    return np.ascontiguousarray(packed[first]), inverse.reshape(-1).astype(np.int64)


@dataclass(frozen=True)
class LadderElements:
    """Элементы уровня цикла: картинка, узел сетки картинки и код «что» каждого элемента.

    На уровне 0 элемент — углопозиция, «что» — код её направления (§4.4). На следующих элемент — деталь,
    собранная окном внимания предыдущего уровня, «что» — её эмбеддинг (§6.5), узел — центр окна (П.1).
    """

    picture: "np.ndarray"  # номер картинки каждого элемента
    y: "np.ndarray"  # узел сетки картинки
    x: "np.ndarray"
    what_packed: "np.ndarray"  # различные коды «что»
    what_length: int
    what_index: "np.ndarray"  # номер кода «что» у каждого элемента
    extent: "np.ndarray | None" = None  # на сколько клеток от своего места видит элемент (поля по кластерам)
    strength: "np.ndarray | None" = None  # уровень активации элемента: насколько его поле совпало с памятью уровня

    @property
    def count(self) -> int:
        return int(self.picture.shape[0])

    def reach(self) -> "np.ndarray":
        """Сколько клеток от своего места видит каждый элемент; углопозиция видит свою клетку."""
        return np.full(self.count, 0.5, dtype=np.float64) if self.extent is None else self.extent

    def only(self, keep: "np.ndarray") -> "LadderElements":
        """Те же элементы без отброшенных (keep — маска оставшихся); коды «что» остаются общими."""
        return LadderElements(
            picture=self.picture[keep],
            y=self.y[keep],
            x=self.x[keep],
            what_packed=self.what_packed,
            what_length=self.what_length,
            what_index=self.what_index[keep],
            extent=None if self.extent is None else self.extent[keep],
            strength=None if self.strength is None else self.strength[keep],
        )

    def power(self) -> "np.ndarray":
        """Уровень активации каждого элемента; у углопозиций он один на всех."""
        return np.ones(self.count, dtype=np.float64) if self.strength is None else self.strength

    def codes(self) -> "np.ndarray":
        """Код «что» каждого элемента, по строке на элемент."""
        return np.ascontiguousarray(self.what_packed[self.what_index])


class StickCoder:
    """Уровень 0 цикла: элемент — углопозиция, «что» — код её направления детекторами угла (§4.4).

    Место углопозиции на картинке в код не входит: его записывает окно внимания в своих координатах (П.4).
    """

    def __init__(self) -> None:
        self.encoder = Encoder(
            ClosedDimension(
                "Angle",
                (settings.MNIST_ANGLE_MIN, settings.MNIST_ANGLE_MAX),
                geometric_detector_layers(
                    layers_count=settings.LADDER_ANGLE_LAYERS,
                    widest_detectors=settings.LADDER_ANGLE_WIDEST_DETECTORS,
                    narrowest_detectors=settings.LADDER_ANGLE_NARROWEST_DETECTORS,
                    overlap=settings.LADDER_ANGLE_OVERLAP,
                ),
                code_length=settings.LADDER_ANGLE_CODE_LENGTH,
            ),
            random_seed=settings.LADDER_RANDOM_SEED,
            layer_phase=settings.ENCODER_LAYER_PHASE,
        )

    def encode(self, atoms: "DigitAtoms") -> LadderElements:
        """Элементы уровня 0: углопозиции набора и коды их направлений (результат — из кеша шагов, если есть).

        При LADDER_STROKE_STRENGTH у элементов есть сила — сила края штриха, нормированная внутри картинки.
        """
        from damp.cache import CACHE

        def compute() -> dict[str, object]:
            what, index = self._codes(atoms)
            return {"what": what, "index": index}

        found = CACHE.run("coding.sticks", {"angle": atoms.angle}, compute, modules=("damp.encoding.elements",))
        return LadderElements(
            picture=atoms.picture,
            y=atoms.y,
            x=atoms.x,
            what_packed=found["what"],
            what_length=self.encoder.code_length,
            what_index=found["index"],
            # Сила штриха — сила его края (вне статьи): по ней пулинг "strongest" выбирает сильнейший элемент круга места.
            strength=np.asarray(atoms.strength, dtype=np.float64) if settings.LADDER_STROKE_STRENGTH else None,
        )

    def _codes(self, atoms: "DigitAtoms") -> tuple["np.ndarray", "np.ndarray"]:
        """Различные коды направлений и номер кода у каждой углопозиции."""
        encoder = self.encoder
        angles, angle_index = np.unique(atoms.angle, return_inverse=True)
        # Углы кодируются порциями: матрица «углы × биты» целиком не создаётся.
        packed = np.concatenate(
            [
                LayoutCodeTable.pack_bits(encoder.encode_matrix(angles[start : start + 8192].reshape(-1, 1)))
                for start in range(0, angles.shape[0], 8192)
            ]
        )
        what, index = distinct_codes(packed)
        LOGGER.event(
            "ladder.sticks",
            section=OUTSIDE_ARTICLE,
            data={
                "sticks": int(atoms.picture.shape[0]),
                "pictures": atoms.pictures,
                "angles": int(angles.shape[0]),
                "angle_codes": int(what.shape[0]),
                "code_length": encoder.code_length,
                "note": "уровень 0: элемент — углопозиция, «что» — код её направления; место на картинке в код "
                "не входит",
            },
        )
        return what, index[angle_index.reshape(-1)]
