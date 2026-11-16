"""Этап 5: детали уровня — эмбеддинги окон по сработавшим детекторам (§6.5, §6.6, §7.1.12; продолжение, П.1)."""

from __future__ import annotations

import numpy as np

from damp import settings
from damp.activation.level_detectors import Firing
from damp.activation.set_embedding import DetectorEmbeddings
from damp.encoding.elements import LadderElements, distinct_codes


class DetailCoder:
    """Код детали — эмбеддинг окна: не больше LADDER_SATURATION самых сильных детекторов, длина кода под плотность
    LADDER_TARGET_DENSITY (§6.6, §7.1.12). Деталь стоит в центре своего окна, её уровень активации — самый сильный
    из сработавших детекторов (П.1); такие детали — элементы следующего уровня (§6.2)."""

    def __init__(self, number: int) -> None:
        self.number = int(number)

    def encode(
        self, firing: Firing, layer_index: "np.ndarray", detectors: int, trained: "np.ndarray", *, length: int | None = None
    ) -> tuple["np.ndarray", int]:
        """Коды деталей, по строке на окно, и их длина (из кеша шагов, если такие срабатывания уже были).

        trained — маска окон обучающих картинок: длина кода подбирается только по ним (§6.6, шаг 3), проверочные
        окна на неё не влияют; length — длина, уже подобранная по обучающим (обученный уровень, новая картинка)."""
        from damp.cache import CACHE

        rows, columns, levels = firing.rows, firing.columns, firing.levels
        layer_index = np.asarray(layer_index)
        trained = np.asarray(trained, dtype=bool)
        fixed = None if length is None else int(length)

        def compute() -> dict[str, object]:
            coded = slice(None)
            if settings.LADDER_EMBEDDING_LAYERS is not None:
                # В код детали идут только широкие слои иерархии (§3.3.2).
                coded = layer_index[columns] < int(settings.LADDER_EMBEDDING_LAYERS)
            details, length = DetectorEmbeddings(
                detectors=int(detectors),
                saturation_limit=settings.LADDER_SATURATION,
                target_density=settings.LADDER_TARGET_DENSITY,
                min_length=settings.LADDER_CODE_LENGTH_MIN,
                max_length=settings.LADDER_CODE_LENGTH_MAX,
                seed=settings.LADDER_RANDOM_SEED + self.number,
                name=f"details{self.number}",
            ).encode(rows[coded], columns[coded], levels[coded], firing.sets, calibration=trained, length=fixed)
            return {"details": details, "length": int(length)}

        inputs = {
            "rows": rows, "columns": columns, "levels": levels, "sets": int(firing.sets), "layer_index": layer_index,
            "detectors": int(detectors), "number": self.number, "trained": trained, "length": 0 if fixed is None else fixed,
        }
        found = CACHE.run("embeddings.details", inputs, compute, modules=("damp.activation.details", "damp.activation.set_embedding"))
        return found["details"], int(found["length"])

    @staticmethod
    def elements(
        details: "np.ndarray",
        length: int,
        picture: "np.ndarray",
        centre: "np.ndarray",
        extent: "np.ndarray",
        strength: "np.ndarray",
    ) -> LadderElements:
        """Детали как элементы следующего уровня: картинка и центр окна, код, поле и уровень активации."""
        what, what_index = distinct_codes(details)
        return LadderElements(
            picture=picture,
            y=centre[:, 0].copy(),
            x=centre[:, 1].copy(),
            what_packed=what,
            what_length=int(length),
            what_index=what_index,
            extent=extent,
            strength=strength,
        )
