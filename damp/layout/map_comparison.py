"""Сравнение двух расстановок одних и тех же точек: непрерывность отображения и качество (§5.8, §5.13.2).

Одни и те же точки (коды и известные компоненты стимула) стоят на двух картах — например, на первой раскладке
и на раскладке следующего уровня, где точка получила код своего места на первой карте. Сравнивается:

- непрерывность отображения по известным компонентам стимула (§5.13.2, рис. 13–16): у соседних занятых ячеек
  стимулы должны быть близки; «стенка» — пара соседей, стимулы которых различаются сильно;
- качество Ē (§5.8) и энергии точек (§5.7) исходных кодов на каждой из расстановок;
- соответствие расстановок: остались ли соседи соседями и насколько вторая карта повторяет первую с точностью
  до поворота, отражения, сдвига и масштаба (вне статьи).

Известные компоненты стимула (угол, позиция окна) в раскладке не участвуют — это диагностика, как раскраска
карты по углу (рис. 16).
"""
from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass

import numpy as np

from damp.article_refs import MAPPING_CONTINUITY, OUTSIDE_ARTICLE, QUALITY_ASSESS
from damp.layout.code_table import LayoutCodeTable
from damp.layout.damp_layout import Layout
from damp.logging import LOGGER


@dataclass(frozen=True)
class MapContinuity:
    """Непрерывность карты по известным компонентам стимула соседних занятых ячеек (§5.13.2)."""

    neighbour_pairs: int
    mean_angle_delta: float  # средняя разность углов соседей, градусы
    angle_jump_ratio: float  # доля пар соседей с разностью углов больше порога
    position_jump_ratio: float | None  # доля пар соседей, разнесённых по позиции стимула дальше порога
    wall_ratio: float  # доля «стенок»: скачок угла или позиции
    double_jump_ratio: float | None  # скачок сразу и угла, и позиции
    mean_similarity: float  # среднее сходство кодов соседей
    low_similarity_ratio: float  # доля пар соседей со сходством кодов ниже порога λ
    singularities: int  # квадраты 2 × 2 занятых ячеек, вокруг которых угол делает полный оборот


@dataclass(frozen=True)
class MapEnergy:
    """Качество расстановки (§5.8): Ē = среднее E / E_max; E_max и среднее E — ненормированные (§5.7)."""

    average: float  # Ē
    maximum: float  # E_max
    mean: float  # среднее E(c, r_e) по точкам


@dataclass(frozen=True)
class MapCorrespondence:
    """Насколько вторая расстановка повторяет первую (вне статьи)."""

    # Пары соседей второй карты: расстояние между теми же точками на первой карте, ячейки.
    target_pairs: int
    source_distance_mean: float
    source_distance_median: float
    source_distance_p90: float
    kept_neighbours_ratio: float  # соседи второй карты, бывшие соседями (8 направлений) на первой
    new_neighbours_ratio: float  # соседи второй карты, стоявшие на первой дальше порога разрыва
    # Пары соседей первой карты: что с ними стало на второй.
    source_pairs: int
    stayed_neighbours_ratio: float  # остались соседями (8 направлений)
    torn_neighbours_ratio: float  # разошлись дальше порога разрыва
    # Наилучшее наложение первой карты на вторую поворотом, отражением, сдвигом и масштабом.
    fit_residual: float  # среднеквадратичное отклонение, ячейки
    fit_explained: float  # доля разброса ячеек второй карты, объяснённая наложением (R²)
    fit_rotation: float  # поворот, градусы
    fit_reflection: bool
    fit_scale: float


class StimulusMapMeter:
    """Измеритель карт одного набора точек по известным компонентам стимула (§5.13.2) и по энергии (§5.8).

    Точки заданы таблицей кодов (коды и hue — угол стимула) и, если известны, позициями стимула (например,
    окно X, Y). Расстановка — ячейки (y, x) точек в порядке таблицы.
    """

    # Сколько пар соседей распаковывается за раз при расчёте сходства кодов.
    PAIR_CHUNK = 16384

    def __init__(
        self,
        table: LayoutCodeTable,
        *,
        angle_period: float,
        angle_jump: float,
        similarity: str,
        similarity_floor: float,
        positions: "np.ndarray | None" = None,
        position_jump: float | None = None,
        tear_distance: float = 10.0,
    ) -> None:
        if angle_period <= 0.0 or not 0.0 < angle_jump < angle_period / 2.0:
            raise ValueError("angle_jump must be in (0, angle_period / 2)")
        if similarity not in ("cosine", "jaccard"):
            raise ValueError("similarity must be 'cosine' or 'jaccard'")
        if (positions is None) != (position_jump is None):
            raise ValueError("positions and position_jump are given together")
        if tear_distance <= math.sqrt(2.0):
            raise ValueError("tear_distance must be larger than the distance between diagonal neighbours")
        self._table = table
        self._count = int(table.count)
        self._period = float(angle_period)
        self._angle_jump = float(angle_jump)
        self._similarity = similarity
        self._similarity_floor = float(similarity_floor)
        self._positions = None
        self._position_jump = None
        if positions is not None:
            positions = np.asarray(positions, dtype=np.float64)
            if positions.shape != (self._count, 2):
                raise ValueError("positions must contain one (x, y) pair per point")
            self._positions = positions
            self._position_jump = float(position_jump)
        self._tear_distance = float(tear_distance)
        LOGGER.event(
            "map_meter.init",
            section=MAPPING_CONTINUITY,
            data={
                "points": self._count,
                "angle_period": self._period,
                "angle_jump": self._angle_jump,
                "position_jump": self._position_jump,
                "similarity": self._similarity,
                "similarity_floor": self._similarity_floor,
                "tear_distance": self._tear_distance,
                "note": "§5.13.2: гладкость отображения проверяется по известным компонентам стимула соседних "
                "ячеек (ср. рис. 13–16); «стенка» — соседи, различающиеся по углу или по позиции больше порога",
            },
        )

    # ------------------------------------------------------------------ соседи
    def _grid(self, cells: "np.ndarray", shape: tuple[int, int]) -> "np.ndarray":
        cells = np.asarray(cells, dtype=np.int64)
        if cells.shape != (self._count, 2):
            raise ValueError("cells must contain one (y, x) cell per point")
        height, width = int(shape[0]), int(shape[1])
        grid = np.full((height, width), -1, dtype=np.int64)
        grid[cells[:, 0], cells[:, 1]] = np.arange(self._count, dtype=np.int64)
        if int((grid >= 0).sum()) != self._count:
            raise ValueError("a cell holds two points; the layout matrix is discrete (§5.1.2)")
        return grid

    @staticmethod
    def _neighbour_pairs(grid: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
        """Пары точек в соседних по стороне занятых ячейках: сосед справа и сосед снизу."""
        right = (grid[:, :-1] >= 0) & (grid[:, 1:] >= 0)
        down = (grid[:-1, :] >= 0) & (grid[1:, :] >= 0)
        first = np.concatenate([grid[:, :-1][right], grid[:-1, :][down]])
        second = np.concatenate([grid[:, 1:][right], grid[1:, :][down]])
        return first, second

    def _signed_delta(self, first: "np.ndarray", second: "np.ndarray") -> "np.ndarray":
        half = self._period / 2.0
        return (second - first + half) % self._period - half

    def _pair_similarity(self, first: "np.ndarray", second: "np.ndarray") -> "np.ndarray":
        """Сходство кодов пар точек (§2.2.4)."""
        result = np.zeros(first.shape[0], dtype=np.float64)
        ones = self._table.ones.astype(np.float64)
        for start in range(0, first.shape[0], self.PAIR_CHUNK):
            a = first[start : start + self.PAIR_CHUNK]
            b = second[start : start + self.PAIR_CHUNK]
            common = (self._table.bit_rows(a) & self._table.bit_rows(b)).sum(axis=1, dtype=np.int64).astype(np.float64)
            if self._similarity == "cosine":
                denominator = np.sqrt(ones[a] * ones[b])
            else:
                denominator = ones[a] + ones[b] - common
            result[start : start + a.shape[0]] = np.divide(
                common, denominator, out=np.zeros_like(common), where=denominator > 0
            )
        return result

    def _singularities(self, grid: "np.ndarray") -> int:
        """Квадраты 2 × 2 занятых ячеек, при обходе которых угол набегает на полный период (сингулярность)."""
        corners = (grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1])
        full = (corners[0] >= 0) & (corners[1] >= 0) & (corners[2] >= 0) & (corners[3] >= 0)
        if not full.any():
            return 0
        hues = self._table.hues
        angles = [hues[corner[full]] for corner in corners]
        winding = np.zeros(angles[0].shape[0], dtype=np.float64)
        for k in range(4):
            winding += self._signed_delta(angles[k], angles[(k + 1) % 4])
        return int(np.count_nonzero(np.abs(winding) >= 0.75 * self._period))

    # ------------------------------------------------------------------ измерения
    def continuity(self, cells: "np.ndarray", shape: tuple[int, int], *, name: str) -> MapContinuity:
        """Непрерывность расстановки по углу, позиции стимула и сходству кодов соседних ячеек (§5.13.2)."""
        grid = self._grid(cells, shape)
        first, second = self._neighbour_pairs(grid)
        pairs = int(first.shape[0])
        if pairs == 0:
            raise ValueError("the map has no pairs of neighbouring points")
        hues = self._table.hues
        angle_delta = np.abs(self._signed_delta(hues[first], hues[second]))
        angle_jump = angle_delta > self._angle_jump
        position_ratio: float | None = None
        double_ratio: float | None = None
        wall = angle_jump
        if self._positions is not None:
            offset = self._positions[first] - self._positions[second]
            position_jump = np.hypot(offset[:, 0], offset[:, 1]) > self._position_jump
            position_ratio = float(position_jump.mean())
            double_ratio = float((angle_jump & position_jump).mean())
            wall = angle_jump | position_jump
        similarity = self._pair_similarity(first, second)
        result = MapContinuity(
            neighbour_pairs=pairs,
            mean_angle_delta=float(angle_delta.mean()),
            angle_jump_ratio=float(angle_jump.mean()),
            position_jump_ratio=position_ratio,
            wall_ratio=float(wall.mean()),
            double_jump_ratio=double_ratio,
            mean_similarity=float(similarity.mean()),
            low_similarity_ratio=float((similarity < self._similarity_floor).mean()),
            singularities=self._singularities(grid),
        )
        LOGGER.event(
            "map_meter.continuity",
            section=MAPPING_CONTINUITY,
            data={
                "map": name,
                "neighbour_pairs": result.neighbour_pairs,
                "mean_angle_delta": result.mean_angle_delta,
                "angle_jump_ratio": result.angle_jump_ratio,
                "position_jump_ratio": result.position_jump_ratio,
                "wall_ratio": result.wall_ratio,
                "double_jump_ratio": result.double_jump_ratio,
                "mean_similarity": result.mean_similarity,
                "low_similarity_ratio": result.low_similarity_ratio,
                "singularities": result.singularities,
                "angle_jump": self._angle_jump,
                "position_jump": self._position_jump,
                "similarity_floor": self._similarity_floor,
            },
        )
        return result

    def energy(self, layout: Layout, radius: int, *, name: str) -> MapEnergy:
        """Качество Ē (§5.8) расстановки layout и ненормированные энергии её точек (§5.7).

        Ē нормирована на E_max своей расстановки, поэтому для сравнения двух расстановок одних и тех же кодов
        приводится и среднее E без нормировки.
        """
        if layout.point_count != self._count:
            raise ValueError("layout must hold the points of the meter")
        average = layout.average_local_energy(radius)
        # Те же энергии, по которым посчитана Ē (кеш расстановки), без деления на E_max.
        energies = np.asarray(layout._compute_point_energies(radius), dtype=np.float64)
        result = MapEnergy(average=float(average), maximum=float(energies.max()), mean=float(energies.mean()))
        LOGGER.event(
            "map_meter.energy",
            section=QUALITY_ASSESS,
            data={
                "map": name,
                "radius": radius,
                "average": result.average,
                "energy_max": result.maximum,
                "energy_mean": result.mean,
                "lambda_threshold": layout.lambda_threshold,
                "note": "Ē = среднее E / E_max (§5.8); energy_mean — среднее E(c, r_e) без нормировки (§5.7)",
            },
        )
        return result

    def correspondence(
        self,
        source_cells: "np.ndarray",
        target_cells: "np.ndarray",
        shape: tuple[int, int],
    ) -> MapCorrespondence:
        """Соответствие расстановок: судьба пар соседей и наилучшее наложение первой карты на вторую."""
        source = np.asarray(source_cells, dtype=np.float64)
        target = np.asarray(target_cells, dtype=np.float64)
        adjacent = math.sqrt(2.0) + 1e-9
        t_first, t_second = self._neighbour_pairs(self._grid(target_cells, shape))
        s_first, s_second = self._neighbour_pairs(self._grid(source_cells, shape))
        if t_first.size == 0 or s_first.size == 0:
            raise ValueError("both maps need pairs of neighbouring points")
        back = np.hypot(*(source[t_first] - source[t_second]).T)
        forward = np.hypot(*(target[s_first] - target[s_second]).T)
        residual, explained, rotation, reflection, scale = self._fit(source, target)
        result = MapCorrespondence(
            target_pairs=int(t_first.size),
            source_distance_mean=float(back.mean()),
            source_distance_median=float(np.median(back)),
            source_distance_p90=float(np.quantile(back, 0.9)),
            kept_neighbours_ratio=float((back <= adjacent).mean()),
            new_neighbours_ratio=float((back > self._tear_distance).mean()),
            source_pairs=int(s_first.size),
            stayed_neighbours_ratio=float((forward <= adjacent).mean()),
            torn_neighbours_ratio=float((forward > self._tear_distance).mean()),
            fit_residual=residual,
            fit_explained=explained,
            fit_rotation=rotation,
            fit_reflection=reflection,
            fit_scale=scale,
        )
        LOGGER.event(
            "map_meter.correspondence",
            section=OUTSIDE_ARTICLE,
            data={
                "target_pairs": result.target_pairs,
                "source_distance_mean": result.source_distance_mean,
                "source_distance_median": result.source_distance_median,
                "source_distance_p90": result.source_distance_p90,
                "kept_neighbours_ratio": result.kept_neighbours_ratio,
                "new_neighbours_ratio": result.new_neighbours_ratio,
                "source_pairs": result.source_pairs,
                "stayed_neighbours_ratio": result.stayed_neighbours_ratio,
                "torn_neighbours_ratio": result.torn_neighbours_ratio,
                "fit_residual": result.fit_residual,
                "fit_explained": result.fit_explained,
                "fit_rotation": result.fit_rotation,
                "fit_reflection": result.fit_reflection,
                "fit_scale": result.fit_scale,
                "tear_distance": self._tear_distance,
                "note": "source_distance — расстояние на первой карте между соседями второй; kept — были соседями, "
                "new — стояли дальше tear_distance; stayed / torn — то же для соседей первой карты на второй; "
                "fit — наложение первой карты на вторую поворотом, отражением, сдвигом и масштабом",
            },
        )
        return result

    @staticmethod
    def _fit(source: "np.ndarray", target: "np.ndarray") -> tuple[float, float, float, bool, float]:
        """Наилучшее по квадратам наложение source на target: поворот или отражение, масштаб и сдвиг."""
        a = source - source.mean(axis=0)
        b = target - target.mean(axis=0)
        spread = float((b * b).sum())
        norm = float((a * a).sum())
        if spread <= 0.0 or norm <= 0.0:
            return 0.0, 0.0, 0.0, False, 1.0
        u, singular, vt = np.linalg.svd(a.T @ b)
        rotation = u @ vt
        scale = float(singular.sum()) / norm
        fitted = scale * (a @ rotation)
        error = float(((fitted - b) ** 2).sum())
        reflection = bool(np.linalg.det(rotation) < 0.0)
        # Угол поворота: для отражения — угол его поворотной части (после отражения второй оси).
        matrix = np.diag([1.0, -1.0]) @ rotation if reflection else rotation
        angle = math.degrees(math.atan2(matrix[0, 1], matrix[0, 0]))
        return math.sqrt(error / source.shape[0]), 1.0 - error / spread, angle, reflection, scale

    # ------------------------------------------------------------------ карты
    def source_position_image(
        self,
        source_cells: "np.ndarray",
        cells: "np.ndarray",
        shape: tuple[int, int],
        *,
        name: str,
    ) -> "np.ndarray":
        """Карта расстановки cells, раскрашенная местом точки на первой карте (source_cells).

        Цвет точки — её место на первой карте: тон — направление от центра первой карты, яркость — расстояние
        до него. Первая карта в этой раскраске — цветовой круг; вторая карта повторяет его (с поворотом или
        отражением), если воспроизводит первую, а разрывы видны как резкие смены цвета.
        """
        source = np.asarray(source_cells, dtype=np.float64)
        cells = np.asarray(cells, dtype=np.int64)
        if source.shape != (self._count, 2) or cells.shape != (self._count, 2):
            raise ValueError("cells must contain one (y, x) cell per point")
        offset = source - source.mean(axis=0)
        radius = np.hypot(offset[:, 0], offset[:, 1])
        longest = float(radius.max()) if radius.size else 0.0
        tone = (np.arctan2(offset[:, 0], offset[:, 1]) / (2.0 * math.pi)) % 1.0
        value = 0.35 + 0.65 * (radius / longest if longest > 0.0 else np.zeros_like(radius))
        colors = np.zeros((self._count, 3), dtype=np.uint8)
        for index in range(self._count):
            red, green, blue = colorsys.hsv_to_rgb(float(tone[index]), 1.0, float(value[index]))
            colors[index] = (int(red * 255), int(green * 255), int(blue * 255))
        image = np.zeros((int(shape[0]), int(shape[1]), 3), dtype=np.uint8)
        image[cells[:, 0], cells[:, 1]] = colors
        LOGGER.event(
            "map_meter.source_position_image",
            section=OUTSIDE_ARTICLE,
            data={
                "map": name,
                "height": int(shape[0]),
                "width": int(shape[1]),
                "points": self._count,
                "coloring": "source_map_position",
            },
            visuals=[LOGGER.visual_image(f"map_meter/{name}/source_position", image, static=True)],
        )
        return image


__all__ = ["MapContinuity", "MapCorrespondence", "MapEnergy", "StimulusMapMeter"]
