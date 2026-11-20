"""Этап 4: детекторы над картой памяти уровня и их срабатывание на коды (§6.1, §6.4, §6.5, §7.1.10, §7.1.11)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from damp import settings
from damp.activation import DbscanClusterer, Detector, DetectorHierarchyBuilder, DetectorLayerConfig
from damp.activation.hierarchy_detection import HierarchyDetection
from damp.encoding.bitarray import BitArray
from damp.layout.code_dictionary import PairwiseSimilarity
from damp.layout.damp_layout import Layout

if TYPE_CHECKING:
    from damp.layout.level_memory import LevelMemory


@dataclass(frozen=True)
class Firing:
    """Срабатывания детекторов на наборы кодов (§6.5): тройки (набор, детектор, уровень активации E(d))."""

    sets: int  # сколько наборов предъявлено
    rows: "np.ndarray"  # номер набора
    columns: "np.ndarray"  # номер детектора (по слоям, слой 0 — самый широкий)
    levels: "np.ndarray"  # уровень активации E(d) ≥ μ_d
    scale: "np.ndarray | None" = None  # делители нормировки уровней по детекторам (scaled), None — без нормировки

    def strength(self) -> "np.ndarray":
        """Уровень активации каждого набора — самый сильный из сработавших на нём детекторов (П.1)."""
        strength = np.zeros(int(self.sets), dtype=np.float64)
        np.maximum.at(strength, self.rows, self.levels.astype(np.float64))
        return strength

    def train_maxima(self, trained: "np.ndarray", detectors: int) -> "np.ndarray":
        """Наибольший уровень активации каждого детектора по наборам обучающих картинок (0 — не срабатывал)."""
        trained = np.asarray(trained, dtype=bool)
        if trained.shape != (int(self.sets),):
            raise ValueError("trained: нужна маска по числу наборов")
        inside = trained[self.rows]
        maxima = np.zeros(int(detectors), dtype=np.float64)
        np.maximum.at(maxima, self.columns[inside], self.levels[inside].astype(np.float64))
        return maxima

    def normalised(self, trained: "np.ndarray", detectors: int) -> "Firing":
        """Уровни активации в [0, 1]. ДОРАБОТКА автора проекта (вне статьи): статья (§7.1.12: «detector activity
        levels … in an interval [0, 1]»; §9.3: «Normalised activation levels») требует лишь значений в [0, 1] и
        способа нормировки не задаёт. Здесь уровень делится на наибольший уровень этого детектора по наборам
        обучающих картинок и обрезается единицей. Детектор, ни разу не сработавший на обучающих, — min(E, 1).
        Проверочные картинки на нормировку не влияют: её задают только обучающие.

        После нормировки у каждого детектора, сработавшего на обучающих, максимум по ним ровно 1, поэтому
        «сильнейший» детектор на обучающих окнах выбирается среди многих ничьих (уровень 1).

        Args:
            trained: маска наборов, взятых из обучающих картинок (по числу наборов).
            detectors: сколько всего детекторов (номера столбцов — 0…detectors−1).
        """
        return self.scaled(self.train_maxima(trained, detectors))

    def scaled(self, maxima: "np.ndarray") -> "Firing":
        """Уровни, делённые на maxima своего детектора и обрезанные единицей (normalised, вне статьи); maxima
        запоминаются в scale — так нормируется новая картинка обученного уровня теми же делителями."""
        maxima = np.asarray(maxima, dtype=np.float64)
        scale = maxima[self.columns]
        levels = self.levels.astype(np.float64)
        levels = np.minimum(np.where(scale > 0.0, levels / np.where(scale > 0.0, scale, 1.0), levels), 1.0)
        return Firing(sets=self.sets, rows=self.rows, columns=self.columns, levels=levels.astype(self.levels.dtype), scale=maxima)


def map_hierarchy(
    layout: Layout,
    bits: "np.ndarray",
    name: str,
    *,
    layers: int,
    thresholds: Sequence[float] | None,
    energies: Sequence[float] | None = None,
) -> tuple[list[Detector], "np.ndarray", Sequence[float]]:
    """Иерархия детекторов статьи над разложенной картой (§6.4, §7.1.10): детекторы, пороги слоёв, энергии точек.

    thresholds — пороги слоёв λ_d; None — layers порогов равными шагами между квантилями попарных сходств
    кодов карты, которыми заданы пороги её раскладки (§7.1.7). energies — нормированные энергии точек (§5.7);
    None — посчитать по карте.
    """
    if energies is None:
        energies = layout.point_energies(settings.ACTIVATION_POINT_ENERGY_RADIUS)
    quantiles = (settings.LADDER_LAYOUT_LAMBDA_QUANTILE, settings.LADDER_LAYOUT_LAMBDA_MAX_QUANTILE)
    low, high = PairwiseSimilarity(seed=settings.LADDER_RANDOM_SEED).quantiles(bits, quantiles, name=f"{name}-hierarchy")
    levels = np.linspace(low, max(low, high), int(layers))
    if thresholds is not None:
        levels = np.asarray(sorted(thresholds), dtype=np.float64)
    extent = max(1, min(layout.width, layout.height))
    specs = settings.ACTIVATION_DETECTOR_LAYER_SPECS
    configs = []
    for layer, threshold in enumerate(levels):
        # Радиус активации и минимум точек — у слоя той же относительной глубины из настроек первой раскладки.
        _, radius_factor, min_points = specs[min(len(specs) - 1, layer * len(specs) // len(levels))]
        configs.append(
            DetectorLayerConfig(
                lambda_threshold=float(threshold),
                activation_radius=max(settings.ACTIVATION_DETECTOR_RADIUS_MIN, int(round(extent * float(radius_factor)))),
                min_points=int(min_points),
            )
        )
    detectors = DetectorHierarchyBuilder(
        layout=layout,
        energies=energies,
        layer_configs=tuple(configs),
        radius_min=settings.ACTIVATION_DETECTOR_RADIUS_MIN,
        output_bits=settings.ACTIVATION_OUTPUT_BITS,
        energy_threshold=settings.ACTIVATION_ENERGY_THRESHOLD,
        eta=settings.ACTIVATION_ETA,
        attempts_without_change=settings.ACTIVATION_DETECTOR_ATTEMPTS_WITHOUT_CHANGE,
        seed=settings.ACTIVATION_DETECTOR_SEED,
        clusterer=DbscanClusterer(eps=settings.ACTIVATION_DBSCAN_EPS, min_samples=settings.ACTIVATION_DBSCAN_MIN_SAMPLES),
        fill_cell=float(settings.ACTIVATION_DETECTOR_FILL_CELL),
        radius_mode=str(settings.ACTIVATION_DETECTOR_RADIUS_MODE),
    ).build()
    return detectors, levels, energies


class LevelDetectors:
    """Детекторы над картой памяти уровня (§6.4) и срабатывание на коды (§6.1, §6.5).

    Пороги слоёв λ_d — LADDER_HIERARCHY_THRESHOLDS либо равные шаги между квантилями сходств кодов карты; порог
    активации λ_a — LADDER_ACTIVATION_LAMBDA, квантиль LADDER_ACTIVATION_QUANTILE либо порог нижнего слоя (§7.1.11).
    Детекторы — шаг кеша (system/damp/cache.py), срабатывания считаются заново. Нет ни одного детектора — RuntimeError.
    """

    MODULES = ("damp.activation.level_detectors",)

    def __init__(self, layout: Layout, detectors: Sequence[Detector], thresholds: "np.ndarray", energies: Sequence[float],
                 activation_lambda: float, name: str) -> None:
        self.layout = layout
        self.built = len(detectors)
        self._built = list(detectors)
        self.thresholds = thresholds
        self.energies = energies
        self.activation_lambda = float(activation_lambda)
        self.name = name
        self._detection: HierarchyDetection | None = None
        self._layered: tuple | None = None  # детекторы по слоям: считаются один раз (их спрашивает каждая картинка)

    @classmethod
    def build(cls, memory: "LevelMemory") -> "LevelDetectors":
        """Детекторы над картой памяти memory (из кеша, если такая карта уже была)."""
        from damp.cache import CACHE

        layout, name = memory.layout, memory.name
        bits = memory.dictionary.bits()

        def compute() -> dict[str, object]:
            detectors, thresholds, _ = map_hierarchy(
                layout, bits, name, layers=settings.LADDER_HIERARCHY_LAYERS, thresholds=settings.LADDER_HIERARCHY_THRESHOLDS,
                energies=memory.energies,
            )
            if settings.LADDER_ACTIVATION_LAMBDA is not None:
                activation_lambda = float(settings.LADDER_ACTIVATION_LAMBDA)
            elif settings.LADDER_ACTIVATION_QUANTILE is not None:
                activation_lambda = PairwiseSimilarity(seed=settings.LADDER_RANDOM_SEED).quantiles(
                    bits, (float(settings.LADDER_ACTIVATION_QUANTILE),), name=f"{name}-activation"
                )[0]
            else:
                activation_lambda = float(thresholds[0])
            return {**cls._pack(detectors), "thresholds": np.asarray(thresholds, dtype=np.float64),
                    "activation_lambda": float(activation_lambda)}

        inputs = {
            "map_codes": np.ascontiguousarray(layout.code_table.packed), "map_cells": layout.cells_array(),
            "width": int(layout.width), "height": int(layout.height), "energies": np.asarray(memory.energies, dtype=np.float64),
            "codes": memory.dictionary.packed,
        }
        found = CACHE.run("activation.detectors", inputs, compute, modules=cls.MODULES)
        detectors = cls._unpack(found)
        return cls(layout, detectors, found["thresholds"], memory.energies, float(found["activation_lambda"]), name)

    @staticmethod
    def _pack(detectors: Sequence[Detector]) -> dict[str, object]:
        if not detectors:
            raise RuntimeError("нет ни одного детектора")
        ones = [detector.output_code.active_indices() for detector in detectors]
        return {
            "center": np.asarray([detector.center for detector in detectors], dtype=np.float64).reshape(-1, 2),
            "radius": np.asarray([detector.radius for detector in detectors], dtype=np.float64),
            "lambda": np.asarray([detector.lambda_threshold for detector in detectors], dtype=np.float64),
            "energy": np.asarray([detector.energy for detector in detectors], dtype=np.float64),
            "points": np.asarray([detector.points_count for detector in detectors], dtype=np.int64),
            "output_length": np.asarray([len(detector.output_code) for detector in detectors], dtype=np.int64),
            "output_ones": np.asarray([index for row in ones for index in row], dtype=np.int64),
            "output_starts": np.concatenate([[0], np.cumsum([len(row) for row in ones])]).astype(np.int64),
            "labels": [detector.label for detector in detectors],
        }

    @staticmethod
    def _unpack(found: dict[str, object]) -> list[Detector]:
        center, starts, ones = found["center"].tolist(), found["output_starts"], found["output_ones"]
        return [
            Detector(
                center=(float(center[i][0]), float(center[i][1])),
                radius=float(found["radius"][i]),
                lambda_threshold=float(found["lambda"][i]),
                energy=float(found["energy"][i]),
                points_count=int(found["points"][i]),
                output_code=BitArray.from_indices(int(found["output_length"][i]), ones[starts[i] : starts[i + 1]].tolist()),
                label=found["labels"][i],
            )
            for i in range(int(found["radius"].shape[0]))
        ]

    @property
    def detection(self) -> HierarchyDetection:
        """Срабатывание детекторов на видеокарте (§6.5); строится при первом обращении."""
        if self._detection is None:
            self._detection = HierarchyDetection(
                self.layout,
                self._built,
                self.energies,
                similarity=settings.LAYOUT_SIMILARITY,
                lambda_threshold=self.activation_lambda,
                eta=settings.ACTIVATION_ETA,
                energy_threshold=settings.ACTIVATION_ENERGY_THRESHOLD,
                detector_threshold=settings.LADDER_DETECTOR_THRESHOLD,
                activate_best=settings.LADDER_ACTIVATE_BEST,
                # Энергия активации: одиночные зажжённые точки гаснут (§5.7 на матрице активации, расширение статьи).
                local_energy_threshold=settings.ACTIVATION_LOCAL_ENERGY_THRESHOLD,
                local_energy_radius=settings.ACTIVATION_POINT_ENERGY_RADIUS,
                use_gpu=settings.ACTIVATION_USE_GPU,
                name=self.name,
            )
        return self._detection

    def _layers(self) -> tuple[tuple[Detector, ...], "np.ndarray", tuple[float, ...]]:
        """Порядок по слоям, как у HierarchyDetection: слой — номер порога λ_d по возрастанию, внутри слоя — порядок
        построения. Без видеокарты: срабатывание из кеша её не требует."""
        if self._layered is None or self._layered[0] is not self._built or self._layered[1] != len(self._built):
            thresholds = sorted({float(detector.lambda_threshold) for detector in self._built})
            layer = np.asarray([thresholds.index(float(detector.lambda_threshold)) for detector in self._built], dtype=np.int64)
            order = np.argsort(layer, kind="stable")
            layers = (tuple(self._built[int(index)] for index in order), layer[order], tuple(thresholds))
            self._layered = (self._built, len(self._built), layers)
        detectors, layer, thresholds = self._layered[2]
        return detectors, layer.copy(), thresholds

    @property
    def detectors(self) -> tuple[Detector, ...]:
        """Детекторы по слоям: слой 0 — самый низкий порог λ_d, самые широкие области (§7.1.10)."""
        return self._layers()[0]

    @property
    def layer_index(self) -> "np.ndarray":
        """Слой каждого детектора в порядке detectors."""
        return self._layers()[1]

    @property
    def layer_thresholds(self) -> tuple[float, ...]:
        """Пороги λ_d слоёв по возрастанию."""
        return self._layers()[2]

    def dominance(self, classes: int) -> "np.ndarray":
        """Господство классов в детекторах (§8.2: «The stimulus that generated each point of the code space … is known
        and can be preserved. This information can be used afterwards to determine cluster dominance»).

        Для каждого детектора (порядок detectors, как у срабатываний) — доли цифр точек карты в его поле (§6.5) с
        весом энергии точки; точки ниже порога μ_e не считаются. Подпись точки — цифра обучающей картинки, чей код
        её породил: проверочные картинки на профиль не влияют. Детектор без весомых точек — нулевой профиль.
        """
        cells = self.layout.cells_array().astype(np.float64)
        labels = np.asarray([int(label) for label in self.layout.code_table.labels()], dtype=np.int64)
        energy = np.asarray(self.energies, dtype=np.float64)
        weights = np.where(energy >= float(settings.ACTIVATION_ENERGY_THRESHOLD), energy, 0.0)
        detectors = self.detectors
        centre = np.asarray([detector.center for detector in detectors], dtype=np.float64)
        radius = np.asarray([detector.radius for detector in detectors], dtype=np.float64)
        inside = np.hypot(cells[:, None, 0] - centre[None, :, 0], cells[:, None, 1] - centre[None, :, 1]) <= radius[None, :]
        weighted = np.zeros((cells.shape[0], int(classes)), dtype=np.float64)
        weighted[np.arange(cells.shape[0]), labels] = weights
        profile = inside.T.astype(np.float64) @ weighted
        total = profile.sum(axis=1, keepdims=True)
        return np.divide(profile, total, out=np.zeros_like(profile), where=total > 0)

    def fire(self, codes: "np.ndarray", index: "np.ndarray", group: "np.ndarray", sets: int) -> Firing:
        """Срабатывания на наборы кодов: index — номер кода у каждого вхождения, group — номер набора (§6.5).

        В кеш шагов не идут: считаются за секунду на видеокарте, а на диске заняли бы сотни мегабайт на уровень.
        """
        rows, columns, levels = self.detection.active_sets(codes, index, group, int(sets))
        return Firing(sets=int(sets), rows=rows, columns=columns, levels=levels)

    def fire_and_light(self, codes: "np.ndarray") -> tuple[Firing, tuple["np.ndarray", "np.ndarray", "np.ndarray"]]:
        """Срабатывания на каждый код как на набор из одного кода и зажжённые точки карты — за один проход (§6.1,
        §6.5). То же, что fire(codes, все, все, число кодов) и lit(codes), но активация карты считается один раз."""
        rows, columns, levels, lit_rows, lit_points, lit_values = self.detection.active_with_lit(codes)
        firing = Firing(sets=int(np.asarray(codes).shape[0]), rows=rows, columns=columns, levels=levels)
        return firing, (lit_rows, lit_points, lit_values)

    def lit(self, codes: "np.ndarray") -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        """Точки карты, которые зажигает каждый код (§6.1): тройки (код, точка, активация a_j > 0). Та же пороговая
        активация, из которой складывается уровень детектора (§6.5), — для показа на карте; на срабатывания не влияет."""
        return self.detection.lit(codes)

    def release(self) -> None:
        """Освободить память видеокарты под срабатыванием."""
        self._detection = None
