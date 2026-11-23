"""Этап 5: детали уровня — эмбеддинги окон по сработавшим детекторам (§6.5, §6.6, §7.1.12; продолжение, П.1)."""

from __future__ import annotations

import numpy as np

from damp import settings
from damp.activation.level_detectors import Firing
from damp.activation.set_embedding import DetectorEmbeddings
from damp.encoding.elements import LadderElements, distinct_codes


def _mix(number: int, layer: int, grid: int, ix: "np.ndarray", iy: "np.ndarray") -> "np.ndarray":
    """Ключ ячейки (уровень, слой, сетка, x, y) — перемешанное 64-битное число (splitmix64): бит по модулю длины кода
    ложится на код равномерно, как случайный бит статьи, но один на всю ячейку."""
    with np.errstate(over="ignore"):
        value = (np.asarray(ix, dtype=np.int64).astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
                 ^ np.asarray(iy, dtype=np.int64).astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
                 ^ np.uint64((number * 1_000_003 + layer * 10_007 + grid * 101 + 17) & 0xFFFFFFFFFFFFFFFF))
        value ^= value >> np.uint64(30)
        value *= np.uint64(0xBF58476D1CE4E5B9)
        value ^= value >> np.uint64(27)
        value *= np.uint64(0x94D049BB133111EB)
        value ^= value >> np.uint64(31)
    return value


class DetailCoder:
    """Код детали — эмбеддинг окна: не больше LADDER_SATURATION самых сильных детекторов, длина кода под плотность
    LADDER_TARGET_DENSITY (§6.6, §7.1.12). Деталь стоит в центре своего окна, её уровень активации — самый сильный
    из сработавших детекторов (П.1); такие детали — элементы следующего уровня (§6.2)."""

    def __init__(self, number: int) -> None:
        self.number = int(number)

    def density(self) -> float:
        """Плотность кода деталей этого уровня (§6.6, шаг 3): LADDER_TARGET_DENSITY_LEVELS — список по уровням (последнее —
        на всех выше; Б5, задача #21), иначе LADDER_TARGET_DENSITY."""
        levels = settings.LADDER_TARGET_DENSITY_LEVELS
        if levels is None:
            return float(settings.LADDER_TARGET_DENSITY)
        levels = list(levels) if isinstance(levels, (list, tuple)) else [levels]
        return float(levels[min(self.number, len(levels) - 1)])

    def topography(self, centre: "np.ndarray", radius: "np.ndarray", layer_index: "np.ndarray") -> "np.ndarray | None":
        """Ключи ячеек карты для топографических бит (план v4.11): ACTIVATION_DETECTOR_BIT_MODE "topographic" с уровня
        ACTIVATION_DETECTOR_BIT_FROM. Ячейка слоя — ACTIVATION_DETECTOR_BIT_CELL медианных радиусов его детекторов;
        ACTIVATION_DETECTOR_BIT_GRIDS сеток со сдвигом на долю ячейки: детекторы одного слоя в одной ячейке делят бит,
        соседние ячейки разных сеток перекрываются. None — бит случайный, как в статье (§6.3)."""
        start = int(settings.ACTIVATION_DETECTOR_BIT_FROM)
        if settings.ACTIVATION_DETECTOR_BIT_MODE != "topographic" or not 0 < start <= self.number:
            return None
        centre = np.asarray(centre, dtype=np.float64)
        radius = np.asarray(radius, dtype=np.float64)
        layer_index = np.asarray(layer_index, dtype=np.int64)
        grids = max(1, int(settings.ACTIVATION_DETECTOR_BIT_GRIDS))
        keys = np.zeros((centre.shape[0], grids), dtype=np.uint64)
        for layer in np.unique(layer_index):
            inside = layer_index == layer
            cell = max(1e-6, float(settings.ACTIVATION_DETECTOR_BIT_CELL) * float(np.median(radius[inside])))
            for grid in range(grids):
                shift = cell * grid / grids
                ix = np.floor((centre[inside, 0] + shift) / cell).astype(np.int64)
                iy = np.floor((centre[inside, 1] + shift) / cell).astype(np.int64)
                keys[inside, grid] = _mix(self.number, int(layer), grid, ix, iy)
        return keys

    def ancestors_on(self) -> int:
        """Сколько слоёв вверх брать предков (план v8.2): LADDER_EMBEDDING_ANCESTORS с уровня LADDER_EMBEDDING_ANCESTORS_FROM;
        0 — выкл."""
        start = int(settings.LADDER_EMBEDDING_ANCESTORS_FROM)
        return int(settings.LADDER_EMBEDDING_ANCESTORS) if 0 < start <= self.number else 0

    def needs_geometry(self) -> bool:
        """Нужны ли центры и радиусы детекторов: топографические биты (v4.11) или предки (v8.2)."""
        return settings.ACTIVATION_DETECTOR_BIT_MODE == "topographic" or self.ancestors_on() > 0

    @staticmethod
    def geometry(detectors) -> "tuple[np.ndarray, np.ndarray]":  # noqa: ANN001
        """Центры и радиусы детекторов карты уровня (LevelDetectors) в порядке их номеров."""
        found = detectors.detectors
        return (np.asarray([d.center for d in found], dtype=np.float64).reshape(-1, 2),
                np.asarray([d.radius for d in found], dtype=np.float64))

    def ancestors(self, centre: "np.ndarray", radius: "np.ndarray", layer_index: "np.ndarray") -> "np.ndarray | None":
        """Предки детекторов (план v8.2, вне статьи по способу; §6.5, §7.1.10 — слои по λ_d вложены): для детектора слоя L —
        детектор слоя L − 1 (шире), в круг которого попадает его центр, ближайший по центру; если такого нет — ближайший
        детектор слоя L − 1. Цепочка на ancestors_on() слоёв вверх; строка — номера предков (−1 — нет). Бит предка — его
        собственный случайный бит (§6.3): канон не меняется. Разные почерки одной детали зажигают соседние узкие
        детекторы — у них общий широкий предок и общий бит; тонкие биты остаются. None — выкл."""
        steps = self.ancestors_on()
        if steps <= 0:
            return None
        centre = np.asarray(centre, dtype=np.float64).reshape(-1, 2)
        radius = np.asarray(radius, dtype=np.float64)
        layer_index = np.asarray(layer_index, dtype=np.int64)
        count = layer_index.shape[0]
        parent = np.full(count, -1, dtype=np.int64)
        for layer in np.unique(layer_index):
            if layer <= 0:
                continue
            upper = np.flatnonzero(layer_index == layer - 1)
            if upper.size == 0:
                continue
            for d in np.flatnonzero(layer_index == layer):
                distance = np.hypot(centre[upper, 0] - centre[d, 0], centre[upper, 1] - centre[d, 1])
                inside = distance <= radius[upper]
                pool = np.flatnonzero(inside) if inside.any() else np.arange(upper.size)
                parent[d] = int(upper[pool[np.argmin(distance[pool])]])
        chain = np.full((count, steps), -1, dtype=np.int64)
        current = np.arange(count, dtype=np.int64)
        for k in range(steps):
            current = np.where(current >= 0, parent[np.maximum(current, 0)], -1)
            chain[:, k] = current
        return chain

    def encode(
        self, firing: Firing, layer_index: "np.ndarray", detectors: int, trained: "np.ndarray", *, length: int | None = None,
        geometry: "tuple[np.ndarray, np.ndarray] | None" = None,
    ) -> tuple["np.ndarray", int]:
        """Коды деталей, по строке на окно, и их длина (из кеша шагов, если такие срабатывания уже были).

        trained — маска окон обучающих картинок: длина кода подбирается только по ним (§6.6, шаг 3), проверочные
        окна на неё не влияют; length — длина, уже подобранная по обучающим (обученный уровень, новая картинка)."""
        from damp.cache import CACHE

        rows, columns, levels = firing.rows, firing.columns, firing.levels
        layer_index = np.asarray(layer_index)
        trained = np.asarray(trained, dtype=bool)
        fixed = None if length is None else int(length)

        # Отбор σ по цвету (§6.5, §3.3.2) с уровня LADDER_EMBEDDING_COLOUR_FROM: цвет детектора — его слой λ_d.
        colour_from = int(settings.LADDER_EMBEDDING_COLOUR_FROM)
        colour = layer_index if 0 < colour_from <= self.number else None
        topography = None if geometry is None else self.topography(geometry[0], geometry[1], layer_index)
        ancestors = None if geometry is None else self.ancestors(geometry[0], geometry[1], layer_index)
        # Квота σ по слоям (план v8.6) с уровня LADDER_EMBEDDING_LAYER_QUOTA_FROM (0 — нет).
        quota_from = int(settings.LADDER_EMBEDDING_LAYER_QUOTA_FROM)
        quota = layer_index if 0 < quota_from <= self.number else None

        def compute() -> dict[str, object]:
            coded = slice(None)
            if settings.LADDER_EMBEDDING_LAYERS is not None:
                # В код детали идут только широкие слои иерархии (§3.3.2).
                coded = layer_index[columns] < int(settings.LADDER_EMBEDDING_LAYERS)
            details, length = DetectorEmbeddings(
                detectors=int(detectors),
                saturation_limit=settings.LADDER_SATURATION,
                saturation_share=float(settings.ACTIVATION_SATURATION_SHARE),
                target_density=self.density(),
                min_length=settings.LADDER_CODE_LENGTH_MIN,
                max_length=settings.LADDER_CODE_LENGTH_MAX,
                seed=settings.LADDER_RANDOM_SEED + self.number,
                name=f"details{self.number}",
                colour=colour,
                topography=topography,
                ancestors=ancestors,
                layers=quota,
            ).encode(rows[coded], columns[coded], levels[coded], firing.sets, calibration=trained, length=fixed)
            return {"details": details, "length": int(length)}

        inputs = {
            "rows": rows, "columns": columns, "levels": levels, "sets": int(firing.sets), "layer_index": layer_index,
            "detectors": int(detectors), "number": self.number, "trained": trained, "length": 0 if fixed is None else fixed,
            "colour": colour is not None,
            **({} if topography is None else {"topography": topography}),
            **({} if ancestors is None else {"ancestors": ancestors}),
            **({} if quota is None else {"layer_quota": True}),
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
