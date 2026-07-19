from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from damp.article_refs import CYCLIC_COORDS, MAPPING_CONTINUITY
from damp.logging import LOGGER


@dataclass(frozen=True)
class PinwheelChart:
    """Локальная карта одной сингулярности осевой ориентации."""

    center_x: float
    center_y: float
    chirality: int
    phase: float
    radius: float


class ContinuousPinwheelTopology:
    """Непрерывное поле ориентаций с произвольными центрами пинвилов.

    Ориентация является осевой величиной с периодом 180 градусов. Поэтому
    половина суммы полярных аргументов остаётся непрерывной при пересечении
    ветви ``atan2``: скачок 360 градусов превращается в полный период 180.
    """

    def __init__(
        self,
        centers: Sequence[tuple[float, float]],
        *,
        angle_period: float,
        radius_factor: float,
        isolated_radius: float,
    ) -> None:
        if angle_period <= 0.0:
            raise ValueError("angle_period must be positive")
        if not 0.0 < radius_factor <= 0.5:
            raise ValueError("radius_factor must be in (0, 0.5]")
        if isolated_radius <= 0.0:
            raise ValueError("isolated_radius must be positive")
        normalized = tuple((float(x), float(y)) for x, y in centers)
        if not normalized:
            raise ValueError("at least one pinwheel center is required")
        if len(set(normalized)) != len(normalized):
            raise ValueError("pinwheel centers must be unique")

        self._centers = normalized
        self._period = float(angle_period)
        self._edges = self._gabriel_edges(normalized)
        self._chiralities = self._solve_chiralities(len(normalized), self._edges)
        phases, phase_rms, phase_max = self._solve_phases(
            normalized,
            self._chiralities,
            self._edges,
            self._period,
        )
        radii = self._radii(normalized, radius_factor, isolated_radius)
        self._charts = tuple(
            PinwheelChart(
                center_x=center[0],
                center_y=center[1],
                chirality=self._chiralities[index],
                phase=phases[index],
                radius=radii[index],
            )
            for index, center in enumerate(normalized)
        )
        LOGGER.event(
            "pinwheel.topology.solve",
            section=MAPPING_CONTINUITY,
            data={
                "centers": len(self._centers),
                "edges": len(self._edges),
                "chiralities": self._chiralities,
                "phases": [chart.phase for chart in self._charts],
                "phase_boundary_rms": phase_rms,
                "phase_boundary_max": phase_max,
                "radii": radii,
                "field": "graph_boundary_angular_synchronization",
            },
        )

    @property
    def charts(self) -> tuple[PinwheelChart, ...]:
        return self._charts

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        return self._edges

    @staticmethod
    def _distance_sq(
        first: tuple[float, float], second: tuple[float, float]
    ) -> float:
        return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2

    @classmethod
    def _gabriel_edges(
        cls, centers: Sequence[tuple[float, float]]
    ) -> tuple[tuple[int, int], ...]:
        """Строит локальный планарный граф без зависимости от сетки центров."""
        if len(centers) == 1:
            return ()
        edges: list[tuple[int, int]] = []
        tolerance = 1e-12
        for first in range(len(centers)):
            for second in range(first + 1, len(centers)):
                edge_length_sq = cls._distance_sq(centers[first], centers[second])
                blocked = False
                for other in range(len(centers)):
                    if other == first or other == second:
                        continue
                    # Точка внутри окружности с ребром first-second как диаметром.
                    if (
                        cls._distance_sq(centers[first], centers[other])
                        + cls._distance_sq(centers[second], centers[other])
                        <= edge_length_sq + tolerance
                    ):
                        blocked = True
                        break
                if not blocked:
                    edges.append((first, second))
        return tuple(edges)

    @staticmethod
    def _solve_chiralities(
        count: int, edges: Sequence[tuple[int, int]]
    ) -> tuple[int, ...]:
        """Максимизирует чередование хиральности на графе соседства."""
        if count == 1:
            return (1,)
        adjacency: list[set[int]] = [set() for _ in range(count)]
        for first, second in edges:
            adjacency[first].add(second)
            adjacency[second].add(first)

        charges = [0] * count
        for root in range(count):
            if charges[root] != 0:
                continue
            charges[root] = 1
            queue = [root]
            while queue:
                current = queue.pop(0)
                for neighbour in sorted(adjacency[current]):
                    if charges[neighbour] == 0:
                        charges[neighbour] = -charges[current]
                        queue.append(neighbour)

        # Нечётные циклы нельзя раскрасить строго двумя цветами. Локальные
        # перевороты дают детерминированный максимум числа разных соседей.
        changed = True
        while changed:
            changed = False
            for index, neighbours in enumerate(adjacency):
                same = sum(charges[index] == charges[item] for item in neighbours)
                opposite = len(neighbours) - same
                if same > opposite:
                    charges[index] *= -1
                    changed = True
        return tuple(charges)

    @classmethod
    def _radii(
        cls,
        centers: Sequence[tuple[float, float]],
        radius_factor: float,
        isolated_radius: float,
    ) -> tuple[float, ...]:
        if len(centers) == 1:
            return (isolated_radius,)
        return tuple(
            radius_factor
            * math.sqrt(
                min(
                    cls._distance_sq(center, other)
                    for other_index, other in enumerate(centers)
                    if other_index != index
                )
            )
            for index, center in enumerate(centers)
        )

    @classmethod
    def _solve_phases(
        cls,
        centers: Sequence[tuple[float, float]],
        chiralities: Sequence[int],
        edges: Sequence[tuple[int, int]],
        period: float,
    ) -> tuple[tuple[float, ...], float, float]:
        if len(centers) == 1:
            return (0.0,), 0.0, 0.0
        adjacency: list[list[tuple[int, float, float]]] = [
            [] for _ in centers
        ]
        constraints: list[tuple[int, int, float]] = []
        for first, second in edges:
            first_x, first_y = centers[first]
            second_x, second_y = centers[second]
            midpoint_x = (first_x + second_x) / 2.0
            midpoint_y = (first_y + second_y) / 2.0
            first_angle = (
                math.degrees(math.atan2(midpoint_y - first_y, midpoint_x - first_x))
                / 2.0
            ) % period
            second_angle = (
                math.degrees(math.atan2(midpoint_y - second_y, midpoint_x - second_x))
                / 2.0
            ) % period
            # desired: phase[second] = phase[first] + delta (mod period)
            delta = (
                chiralities[first] * first_angle
                - chiralities[second] * second_angle
            ) % period
            distance = math.dist(centers[first], centers[second])
            weight = 1.0 / max(distance, 1e-12)
            adjacency[first].append((second, delta, weight))
            adjacency[second].append((first, (-delta) % period, weight))
            constraints.append((first, second, delta))

        phases: list[float | None] = [None] * len(centers)
        roots: set[int] = set()
        for root in range(len(centers)):
            if phases[root] is not None:
                continue
            roots.add(root)
            phases[root] = 0.0
            queue = [root]
            while queue:
                current = queue.pop(0)
                assert phases[current] is not None
                for neighbour, delta, _ in adjacency[current]:
                    if phases[neighbour] is None:
                        phases[neighbour] = (phases[current] + delta) % period
                        queue.append(neighbour)

        resolved = [0.0 if phase is None else phase for phase in phases]
        for _ in range(200):
            max_change = 0.0
            for index in range(len(centers)):
                if index in roots or not adjacency[index]:
                    continue
                sine = 0.0
                cosine = 0.0
                for neighbour, delta_to_neighbour, weight in adjacency[index]:
                    desired = (resolved[neighbour] - delta_to_neighbour) % period
                    doubled = 2.0 * math.pi * desired / period
                    sine += weight * math.sin(doubled)
                    cosine += weight * math.cos(doubled)
                updated = (math.atan2(sine, cosine) * period / (2.0 * math.pi)) % period
                change = abs(updated - resolved[index]) % period
                max_change = max(max_change, min(change, period - change))
                resolved[index] = updated
            if max_change < 1e-9:
                break

        residuals: list[float] = []
        for first, second, delta in constraints:
            residual = abs((resolved[second] - resolved[first] - delta) % period)
            residuals.append(min(residual, period - residual))
        rms = (
            math.sqrt(sum(value * value for value in residuals) / len(residuals))
            if residuals
            else 0.0
        )
        return tuple(resolved), rms, max(residuals, default=0.0)

    def orientation_at(self, x: float, y: float) -> float:
        phase = 0.0
        for index, (center_x, center_y) in enumerate(self._centers):
            dx = float(x) - center_x
            dy = float(y) - center_y
            if abs(dx) <= 1e-15 and abs(dy) <= 1e-15:
                raise ValueError("orientation is undefined at a pinwheel center")
            phase += 0.5 * self._chiralities[index] * math.degrees(math.atan2(dy, dx))
        orientation = phase % self._period
        LOGGER.event(
            "pinwheel.topology.orientation",
            section=CYCLIC_COORDS,
            data={"x": x, "y": y, "orientation": orientation},
        )
        return orientation

    def encode_angle(
        self,
        angle: float,
        *,
        center_index: int,
        position_x: float,
        position_y: float,
    ) -> tuple[float, float]:
        chart = self._charts[center_index]
        encoded = (chart.chirality * float(angle) + chart.phase) % self._period
        return encoded, chart.phase
