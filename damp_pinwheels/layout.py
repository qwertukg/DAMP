from __future__ import annotations

import colorsys
import logging
import math
import random
from typing import Iterable, Sequence

from .bit_array import BitArray
from .layout_point import LayoutPoint

_UNSET = object()
_logger = logging.getLogger(__name__)


class Layout:
    """Discrete layout of code vectors on a 2D grid using the DAMP algorithm."""

    def __init__(
        self,
        codes: Sequence[tuple[BitArray, float, float]],
        *,
        grid_size: int | None = None,
        empty_ratio: float = 0.15,
        similarity: str = "jaccard",
        lambda_threshold: float = 0.65,
        eta: float | None = 10.0,
        precompute_similarity: bool = True,
        max_precompute: int = 3000,
        seed: int = 0,
    ) -> None:
        if not codes:
            raise ValueError("codes must be non-empty")
        if empty_ratio < 0:
            raise ValueError("empty_ratio must be >= 0")
        if similarity not in ("cosine", "jaccard"):
            raise ValueError("similarity must be 'cosine' or 'jaccard'")
        if not 0 <= lambda_threshold <= 1:
            raise ValueError("lambda_threshold must be in [0, 1]")
        if max_precompute <= 0:
            raise ValueError("max_precompute must be positive")

        self._points = [
            LayoutPoint(code=code, angle=angle, hue=hue, ones=code.count())
            for code, angle, hue in codes
        ]
        self._point_count = len(self._points)
        self._similarity = similarity
        self._lambda = lambda_threshold
        self._eta = eta
        self._rng = random.Random(seed)

        if grid_size is None:
            target = len(self._points) * (1.0 + empty_ratio)
            grid_size = math.ceil(math.sqrt(target))
        if grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if grid_size * grid_size < len(self._points):
            raise ValueError("grid_size too small for number of points")

        self.height = grid_size
        self.width = grid_size
        self.grid: list[list[int | None]] = [
            [None for _ in range(self.width)] for _ in range(self.height)
        ]
        self._positions: list[tuple[int, int]] = [(-1, -1)] * len(self._points)
        self._pos_y = [0] * len(self._points)
        self._pos_x = [0] * len(self._points)
        self._place_points()

        self._distance_eps = 1e-6
        self.last_steps = 0
        self._sim_base: list[list[float]] | None = None
        self._sim_indices: list[list[int]] | None = None
        if precompute_similarity and self._point_count <= max_precompute:
            self._build_similarity_cache()
            self._rebuild_similarity_indices()

    def run(
        self,
        *,
        steps: int,
        pairs_per_step: int,
        pair_radius: int | None,
        mode: str = "long",
        local_radius: int | None = None,
        min_swap_ratio: float = 0.01,
        energy_radius: int | None = None,
        energy_check_every: int = 10,
        energy_delta: float = 1e-3,
        energy_patience: int = 3,
        log_every: int | None = None,
        log_prefix: str = "layout",
        snapshot_path: str | None = None,
        snapshot_every: int | None = None,
    ) -> int:
        if steps <= 0:
            raise ValueError("steps must be positive")
        if pairs_per_step <= 0:
            raise ValueError("pairs_per_step must be positive")
        if pair_radius is not None and pair_radius <= 0:
            raise ValueError("pair_radius must be positive when set")
        if mode not in ("long", "short"):
            raise ValueError("mode must be 'long' or 'short'")
        if local_radius is not None and local_radius <= 0:
            raise ValueError("local_radius must be positive when set")
        if not 0 <= min_swap_ratio <= 1:
            raise ValueError("min_swap_ratio must be in [0, 1]")
        if energy_radius is not None and energy_radius <= 0:
            raise ValueError("energy_radius must be positive when set")
        if energy_check_every <= 0:
            raise ValueError("energy_check_every must be positive")
        if energy_delta < 0:
            raise ValueError("energy_delta must be >= 0")
        if energy_patience <= 0:
            raise ValueError("energy_patience must be positive")
        if log_every is not None and log_every <= 0:
            raise ValueError("log_every must be positive when set")
        if snapshot_every is not None and snapshot_every <= 0:
            raise ValueError("snapshot_every must be positive when set")

        total_swaps = 0
        min_swaps = 0
        if min_swap_ratio > 0:
            min_swaps = max(1, int(pairs_per_step * min_swap_ratio))
        steps_executed = 0
        prev_energy: float | None = None
        stable_checks = 0

        for step in range(steps):
            swaps = self.step(
                pairs_per_step=pairs_per_step,
                pair_radius=pair_radius,
                mode=mode,
                local_radius=local_radius,
            )
            total_swaps += swaps
            steps_executed = step + 1
            if log_every is not None and (step + 1) % log_every == 0:
                _logger.info(
                    "%s %s step %d/%d swaps=%d total_swaps=%d",
                    log_prefix,
                    mode,
                    step + 1,
                    steps,
                    swaps,
                    total_swaps,
                )
            if (
                snapshot_path is not None
                and snapshot_every is not None
                and (step + 1) % snapshot_every == 0
            ):
                self.write_ppm(snapshot_path)
            if energy_radius is not None and mode == "long":
                if step % energy_check_every == 0:
                    energy = self.average_local_energy(energy_radius)
                    if log_every is not None and (step + 1) % log_every == 0:
                        _logger.info(
                            "%s %s step %d energy=%.6f",
                            log_prefix,
                            mode,
                            step + 1,
                            energy,
                        )
                    if prev_energy is not None:
                        if abs(energy - prev_energy) <= energy_delta:
                            stable_checks += 1
                        else:
                            stable_checks = 0
                        if stable_checks >= energy_patience:
                            if log_every is not None:
                                _logger.info(
                                    "%s %s early stop at step %d (energy stable)",
                                    log_prefix,
                                    mode,
                                    step + 1,
                                )
                            break
                    prev_energy = energy
            if min_swaps and swaps < min_swaps:
                if log_every is not None:
                    _logger.info(
                        "%s %s early stop at step %d (low swaps: %d)",
                        log_prefix,
                        mode,
                        step + 1,
                        swaps,
                    )
                break

        self.last_steps = steps_executed
        return total_swaps

    def set_similarity_params(
        self,
        *,
        lambda_threshold: float | object = _UNSET,
        eta: float | None | object = _UNSET,
    ) -> None:
        if lambda_threshold is not _UNSET:
            if not 0 <= lambda_threshold <= 1:
                raise ValueError("lambda_threshold must be in [0, 1]")
            self._lambda = lambda_threshold
        if eta is not _UNSET:
            self._eta = eta
        if self._sim_base is not None:
            self._rebuild_similarity_indices()

    def step(
        self,
        *,
        pairs_per_step: int,
        pair_radius: int | None,
        mode: str,
        local_radius: int | None,
    ) -> int:
        swaps: list[tuple[int, int, int, int]] = []
        for _ in range(pairs_per_step):
            y1, x1, y2, x2 = self._sample_pair(pair_radius)
            if self._should_swap(y1, x1, y2, x2, mode=mode, local_radius=local_radius):
                swaps.append((y1, x1, y2, x2))
        return self._apply_swaps(swaps)

    def positions(self) -> list[tuple[int, int]]:
        return list(self._positions)

    def colors_rgb(self) -> list[tuple[int, int, int]]:
        return [self._hue_to_rgb(point.hue) for point in self._points]

    def render_rgb_grid(self, *, empty_color: tuple[int, int, int] = (0, 0, 0)) -> list[
        list[tuple[int, int, int]]
    ]:
        grid = [[empty_color for _ in range(self.width)] for _ in range(self.height)]
        for idx, (y, x) in enumerate(self._positions):
            grid[y][x] = self._hue_to_rgb(self._points[idx].hue)
        return grid

    def write_ppm(
        self,
        path: str,
        *,
        empty_color: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        grid = self.render_rgb_grid(empty_color=empty_color)
        with open(path, "wb") as handle:
            handle.write(f"P6\n{self.width} {self.height}\n255\n".encode("ascii"))
            for row in grid:
                for r, g, b in row:
                    handle.write(bytes((r, g, b)))

    def average_local_energy(self, radius: int) -> float:
        if radius <= 0:
            raise ValueError("radius must be positive")
        if self._point_count == 0:
            return 0.0

        radius_sq = radius * radius
        energies: list[float] = []
        pos_y = self._pos_y
        pos_x = self._pos_x
        sim_lambda = self._sim_lambda_idx

        for idx in range(self._point_count):
            cy = pos_y[idx]
            cx = pos_x[idx]
            energy = 0.0
            for j in range(self._point_count):
                if j == idx:
                    continue
                dy = cy - pos_y[j]
                dx = cx - pos_x[j]
                dist_sq = dy * dy + dx * dx
                if dist_sq > radius_sq:
                    continue
                sim = sim_lambda(idx, j)
                if sim <= 0.0:
                    continue
                energy += sim / (dist_sq + self._distance_eps)
            energies.append(energy)

        emax = max(energies)
        if emax <= 0.0:
            return 0.0
        return sum(e / emax for e in energies) / len(energies)

    def _place_points(self) -> None:
        cells = [(y, x) for y in range(self.height) for x in range(self.width)]
        self._rng.shuffle(cells)
        for idx, (y, x) in enumerate(cells[: len(self._points)]):
            self.grid[y][x] = idx
            self._positions[idx] = (y, x)
            self._pos_y[idx] = y
            self._pos_x[idx] = x

    def _sample_pair(self, radius: int | None) -> tuple[int, int, int, int]:
        idx_a = self._rng.randrange(self._point_count)
        y1, x1 = self._positions[idx_a]

        if radius is None:
            while True:
                y2 = self._rng.randrange(self.height)
                x2 = self._rng.randrange(self.width)
                if (y2, x2) != (y1, x1):
                    return y1, x1, y2, x2

        radius_sq = radius * radius
        for _ in range(50):
            dy = self._rng.randint(-radius, radius)
            dx = self._rng.randint(-radius, radius)
            if dy * dy + dx * dx > radius_sq:
                continue
            y2 = y1 + dy
            x2 = x1 + dx
            if 0 <= y2 < self.height and 0 <= x2 < self.width:
                if (y2, x2) != (y1, x1):
                    return y1, x1, y2, x2

        while True:
            y2 = self._rng.randrange(self.height)
            x2 = self._rng.randrange(self.width)
            if (y2, x2) != (y1, x1):
                return y1, x1, y2, x2

    def _should_swap(
        self,
        y1: int,
        x1: int,
        y2: int,
        x2: int,
        *,
        mode: str,
        local_radius: int | None,
    ) -> bool:
        idx_a = self.grid[y1][x1]
        idx_b = self.grid[y2][x2]

        if idx_a is None and idx_b is None:
            return False
        if idx_a is None:
            idx_a, idx_b = idx_b, idx_a
            y1, x1, y2, x2 = y2, x2, y1, x1

        if mode == "long":
            energy_c, energy_s = self._pair_energy_long(
                idx_a, (y1, x1), idx_b, (y2, x2)
            )
            return energy_s < energy_c

        radius = local_radius
        if radius is None:
            dist_sq = (y1 - y2) ** 2 + (x1 - x2) ** 2
            radius = max(1, int(math.ceil(math.sqrt(dist_sq))))
        energy_c, energy_s = self._pair_energy_short(
            idx_a, (y1, x1), idx_b, (y2, x2), radius=radius
        )
        return energy_s > energy_c

    def _pair_energy_long(
        self,
        idx_a: int,
        pos_a: tuple[int, int],
        idx_b: int | None,
        pos_b: tuple[int, int],
    ) -> tuple[float, float]:
        pos_a_y, pos_a_x = pos_a
        pos_b_y, pos_b_x = pos_b
        pos_y = self._pos_y
        pos_x = self._pos_x
        sim_lambda = self._sim_lambda_idx
        energy_c = 0.0
        energy_s = 0.0
        if self._sim_indices is not None:
            candidates = self._collect_candidates(idx_a, idx_b)
            for idx in candidates:
                if idx == idx_a or idx == idx_b:
                    continue
                s1 = sim_lambda(idx, idx_a)
                s2 = sim_lambda(idx, idx_b) if idx_b is not None else 0.0
                if s1 == 0.0 and s2 == 0.0:
                    continue
                dy1 = pos_a_y - pos_y[idx]
                dx1 = pos_a_x - pos_x[idx]
                dy2 = pos_b_y - pos_y[idx]
                dx2 = pos_b_x - pos_x[idx]
                d1 = dy1 * dy1 + dx1 * dx1
                d2 = dy2 * dy2 + dx2 * dx2
                energy_c += s1 * d1 + s2 * d2
                energy_s += s2 * d1 + s1 * d2
            return energy_c, energy_s

        for idx in range(self._point_count):
            if idx == idx_a or idx == idx_b:
                continue
            s1 = sim_lambda(idx, idx_a)
            s2 = sim_lambda(idx, idx_b) if idx_b is not None else 0.0
            if s1 == 0.0 and s2 == 0.0:
                continue
            dy1 = pos_a_y - pos_y[idx]
            dx1 = pos_a_x - pos_x[idx]
            dy2 = pos_b_y - pos_y[idx]
            dx2 = pos_b_x - pos_x[idx]
            d1 = dy1 * dy1 + dx1 * dx1
            d2 = dy2 * dy2 + dx2 * dx2
            energy_c += s1 * d1 + s2 * d2
            energy_s += s2 * d1 + s1 * d2
        return energy_c, energy_s

    def _pair_energy_short(
        self,
        idx_a: int,
        pos_a: tuple[int, int],
        idx_b: int | None,
        pos_b: tuple[int, int],
        *,
        radius: int,
    ) -> tuple[float, float]:
        cy = (pos_a[0] + pos_b[0]) / 2.0
        cx = (pos_a[1] + pos_b[1]) / 2.0
        radius_sq = radius * radius

        pos_y = self._pos_y
        pos_x = self._pos_x
        sim_lambda = self._sim_lambda_idx
        energy_c = 0.0
        energy_s = 0.0
        if self._sim_indices is not None:
            candidates = self._collect_candidates(idx_a, idx_b)
            for idx in candidates:
                if idx == idx_a or idx == idx_b:
                    continue
                dyc = pos_y[idx] - cy
                dxc = pos_x[idx] - cx
                if dyc * dyc + dxc * dxc > radius_sq:
                    continue
                s1 = sim_lambda(idx, idx_a)
                s2 = sim_lambda(idx, idx_b) if idx_b is not None else 0.0
                if s1 == 0.0 and s2 == 0.0:
                    continue
                dy1 = pos_a[0] - pos_y[idx]
                dx1 = pos_a[1] - pos_x[idx]
                dy2 = pos_b[0] - pos_y[idx]
                dx2 = pos_b[1] - pos_x[idx]
                d1 = dy1 * dy1 + dx1 * dx1
                d2 = dy2 * dy2 + dx2 * dx2
                energy_c += s1 / (d1 + self._distance_eps) + s2 / (
                    d2 + self._distance_eps
                )
                energy_s += s2 / (d1 + self._distance_eps) + s1 / (
                    d2 + self._distance_eps
                )
            return energy_c, energy_s

        for idx in range(self._point_count):
            if idx == idx_a or idx == idx_b:
                continue
            dyc = pos_y[idx] - cy
            dxc = pos_x[idx] - cx
            if dyc * dyc + dxc * dxc > radius_sq:
                continue
            s1 = sim_lambda(idx, idx_a)
            s2 = sim_lambda(idx, idx_b) if idx_b is not None else 0.0
            if s1 == 0.0 and s2 == 0.0:
                continue
            dy1 = pos_a[0] - pos_y[idx]
            dx1 = pos_a[1] - pos_x[idx]
            dy2 = pos_b[0] - pos_y[idx]
            dx2 = pos_b[1] - pos_x[idx]
            d1 = dy1 * dy1 + dx1 * dx1
            d2 = dy2 * dy2 + dx2 * dx2
            energy_c += s1 / (d1 + self._distance_eps) + s2 / (
                d2 + self._distance_eps
            )
            energy_s += s2 / (d1 + self._distance_eps) + s1 / (
                d2 + self._distance_eps
            )
        return energy_c, energy_s

    def _apply_swaps(self, swaps: Iterable[tuple[int, int, int, int]]) -> int:
        used: set[tuple[int, int]] = set()
        applied = 0
        for y1, x1, y2, x2 in swaps:
            if (y1, x1) in used or (y2, x2) in used:
                continue
            idx_a = self.grid[y1][x1]
            idx_b = self.grid[y2][x2]
            self.grid[y1][x1], self.grid[y2][x2] = idx_b, idx_a
            if idx_a is not None:
                self._positions[idx_a] = (y2, x2)
                self._pos_y[idx_a] = y2
                self._pos_x[idx_a] = x2
            if idx_b is not None:
                self._positions[idx_b] = (y1, x1)
                self._pos_y[idx_b] = y1
                self._pos_x[idx_b] = x1
            used.add((y1, x1))
            used.add((y2, x2))
            applied += 1
        return applied

    def _build_similarity_cache(self) -> None:
        sim_base: list[list[float]] = [
            [0.0] * self._point_count for _ in range(self._point_count)
        ]
        for i in range(self._point_count):
            sim_base[i][i] = 1.0
            for j in range(i + 1, self._point_count):
                sim = self._similarity_base(self._points[i], self._points[j])
                sim_base[i][j] = sim
                sim_base[j][i] = sim
        self._sim_base = sim_base
        _logger.info("similarity cache built for %d points", self._point_count)

    def _rebuild_similarity_indices(self) -> None:
        if self._sim_base is None:
            self._sim_indices = None
            return
        indices: list[list[int]] = []
        threshold = self._lambda
        for i in range(self._point_count):
            row = self._sim_base[i]
            indices.append([j for j, sim in enumerate(row) if sim >= threshold])
        self._sim_indices = indices
        _logger.info(
            "similarity index built with lambda=%.3f", self._lambda
        )

    def _collect_candidates(self, idx_a: int, idx_b: int | None) -> list[int]:
        if self._sim_indices is None:
            return list(range(self._point_count))
        if idx_b is None:
            return self._sim_indices[idx_a]
        merged = set(self._sim_indices[idx_a])
        merged.update(self._sim_indices[idx_b])
        return list(merged)

    def _sim_lambda_idx(self, idx_a: int, idx_b: int | None) -> float:
        if idx_b is None:
            return 0.0
        if self._sim_base is not None:
            sim = self._sim_base[idx_a][idx_b]
        else:
            sim = self._similarity_base(self._points[idx_a], self._points[idx_b])
        if sim <= 0.0:
            return 0.0
        if self._eta is None:
            return sim if sim >= self._lambda else 0.0
        return sim * (1.0 / (1.0 + math.exp(-self._eta * (sim - self._lambda))))

    def _similarity_base(self, a: LayoutPoint, b: LayoutPoint) -> float:
        if a.ones == 0 or b.ones == 0:
            code_sim = 0.0
        else:
            common = a.code.common(b.code)
            if self._similarity == "cosine":
                denom = math.sqrt(a.ones * b.ones)
                code_sim = 0.0 if denom == 0 else common / denom
            else:
                union = a.ones + b.ones - common
                code_sim = 0.0 if union == 0 else common / union
        return code_sim

    @staticmethod
    def _hue_to_rgb(hue: float) -> tuple[int, int, int]:
        r, g, b = colorsys.hsv_to_rgb((hue % 360.0) / 360.0, 1.0, 1.0)
        return int(r * 255), int(g * 255), int(b * 255)
