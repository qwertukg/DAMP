"""Шаг дальней фазы большой раскладки целиком на устройстве (§5.3, §5.5, §5.6, §5.10, §5.12).

Шаг раскладки на хосте — выбор пар, их энергии, отбор обменов, применение обменов — на миллионе пар за шаг
тратит на массивы numpy столько же времени, сколько устройство на суммы энергий. Здесь весь шаг идёт
на устройстве: расстановка (ячейка каждой точки и точка каждой ячейки) живёт там же, на хост
возвращается только число применённых обменов. Действия шага — те же, что на хосте:

1. пары (§5.3): первая точка — случайная, вторая ячейка — случайная ячейка круга радиуса r вокруг первой;
   пара, которой за ATTEMPTS попыток не нашлось ячейки в круге и в сетке, в шаг не входит;
2. энергии пар φ_c, φ_s (§5.5): слитое ядро по ячейкам окна окрестности — те же слагаемые, что на хосте
   (FusedPairEnergy.window_energies_device);
3. отбор (§5.6): выгодные обмены (Δ = φ_s − φ_c > 0) принимаются; при температуре T > 0 невыгодный
   принимается с вероятностью exp(Δ / (T · rms|Δ|)) (отжиг — расширение статьи);
4. обмены одного шага не делят ячеек (§5.10): обмен применяется, если он первый по порядку пар шага
   для обеих своих ячеек.

Случайные числа берёт генератор устройства, поэтому расстановка отличается от хода на хосте при том же
seed; это режим больших раскладок, где хода на хосте для сравнения нет.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from damp.article_refs import GPU_IMPLEMENTATION, POINT_EXCHANGE
from damp.logging import LOGGER


class DeviceLongStep:
    """Расстановка на устройстве и шаг дальней фазы над ней."""

    # Попыток выбрать вторую ячейку пары в круге радиуса r (доля удач за попытку — не меньше π/16 у угла сетки).
    ATTEMPTS = 6
    # Сверка энергий с исходным проходом движка — на стольких первых парах шага.
    VERIFY_PAIRS = 4096
    VERIFY_RTOL = 2e-4
    VERIFY_ATOL = 1e-5

    def __init__(self, engine: Any, fused: Any, *, seed: int) -> None:
        torch = engine.torch_mod
        self._engine = engine
        self._fused = fused
        self._torch = torch
        self._device = engine.torch_device
        self._count = engine.point_count
        self._height, self._width = engine.grid_shape
        self._generator = torch.Generator(device=self._device)
        self._generator.manual_seed(int(seed))
        # Ячейка каждой точки (линейный номер y·width + x) и точка каждой ячейки (N — пусто; последний
        # элемент — «за сеткой»).
        self._cell: Any = None
        self._grid: Any = None
        self.version = -1
        self._steps = 0
        LOGGER.event(
            "layout.device_step.init",
            section=GPU_IMPLEMENTATION,
            data={
                "points": self._count,
                "grid": (self._height, self._width),
                "attempts": self.ATTEMPTS,
                "note": "шаг дальней фазы целиком на устройстве: выбор пар (§5.3), энергии (§5.5), отбор (§5.6) "
                "и применение обменов (§5.10); на хост возвращается число обменов",
            },
        )

    # ------------------------------------------------------------------ расстановка
    def upload(self, positions: "np.ndarray", version: int) -> None:
        """Расстановка с хоста: ячейки точек N × 2 (y, x)."""
        torch = self._torch
        linear = positions[:, 0].astype(np.int64) * self._width + positions[:, 1].astype(np.int64)
        self._cell = torch.as_tensor(linear, device=self._device)
        grid = torch.full((self._height * self._width + 1,), self._count, dtype=torch.int32, device=self._device)
        grid[self._cell] = torch.arange(self._count, dtype=torch.int32, device=self._device)
        self._grid = grid
        self.version = int(version)

    def download(self) -> "np.ndarray":
        """Расстановка на хост: ячейки точек N × 2 (y, x), int64."""
        linear = self._cell.cpu().numpy()
        return np.stack([linear // self._width, linear % self._width], axis=1)

    # ------------------------------------------------------------------ шаг
    def _sample_pairs(self, pairs: int, radius: int) -> tuple[Any, Any, Any, Any, Any]:
        """Пары шага (§5.3): точка a, её ячейка, вторая ячейка и смещение (dy, dx) до неё."""
        torch = self._torch
        device = self._device
        generator = self._generator
        height, width = self._height, self._width
        point_a = torch.randint(0, self._count, (pairs,), device=device, generator=generator)
        cell_a = self._cell[point_a]
        y1 = cell_a // width
        x1 = cell_a - y1 * width
        dy = torch.zeros(pairs, dtype=torch.int64, device=device)
        dx = torch.zeros(pairs, dtype=torch.int64, device=device)
        found = torch.zeros(pairs, dtype=torch.bool, device=device)
        radius_sq = radius * radius
        for _ in range(self.ATTEMPTS):
            step = torch.randint(-radius, radius + 1, (2, pairs), device=device, generator=generator)
            distance_sq = step[0] * step[0] + step[1] * step[1]
            y2 = y1 + step[0]
            x2 = x1 + step[1]
            good = (distance_sq <= radius_sq) & (distance_sq != 0) & (y2 >= 0) & (y2 < height) & (x2 >= 0) & (x2 < width)
            good &= ~found
            dy = torch.where(good, step[0], dy)
            dx = torch.where(good, step[1], dx)
            found |= good
        keep = torch.nonzero(found).view(-1)
        point_a, cell_a, dy, dx = point_a[keep], cell_a[keep], dy[keep], dx[keep]
        cell_b = cell_a + dy * width + dx
        return point_a, cell_a, cell_b, dy, dx

    def step(
        self,
        *,
        pairs: int,
        radius: int,
        local_radius: int,
        offsets: Callable[[int], tuple["np.ndarray", "np.ndarray"]],
        temperature: float,
        energy_arguments: dict[str, Any],
        step_index: int | None,
        reference: Callable[..., tuple["np.ndarray", "np.ndarray"]] | None = None,
    ) -> tuple[int, float]:
        """Один шаг: число применённых обменов и сумма их длин ‖p1 − p2‖.

        offsets(reach) — смещения ячеек окна окрестности полуширины reach (все ячейки круга или их выборка,
        §5.9.4); reference — исходный проход движка для сверки энергий (None — без сверки).
        """
        torch = self._torch
        width = self._width
        count = self._count
        with torch.no_grad():
            point_a, cell_a, cell_b, dy, dx = self._sample_pairs(int(pairs), int(radius))
            rows = int(point_a.numel())
            if rows == 0:
                return 0, 0.0
            point_b = self._grid[cell_b].to(torch.int64)
            # Радиус окрестности r ≥ ‖p1 − p2‖ + 1 и полуширина окна — как в neighbourhood_geometry (§5.5.2).
            distance = torch.sqrt((dy * dy + dx * dx).to(torch.float32))
            radius_max = max(float(torch.ceil(distance.max()).item()) + 1.0, float(np.float32(local_radius)))
            reach = int(math.ceil(radius_max)) + 1
            window = offsets(reach)
            energy_c, energy_s = self._fused.window_energies_device(
                point_a, point_b, cell_a, cell_b, self._grid, offsets=window, local_radius=local_radius,
                **energy_arguments,
            )
            if reference is not None:
                self._verify(point_a, point_b, cell_a, cell_b, energy_c, energy_s, window, reach, reference)
            # Окрестность (и gauss-дальняя): обмен выгоден при φ_s > φ_c (§5.6).
            delta = (energy_s - energy_c).to(torch.float64)
            accept = delta > 0.0
            favourable = accept
            if temperature > 0.0:
                rms = float(torch.sqrt(torch.mean(delta * delta)).item()) or 1.0
                draws = torch.rand(rows, dtype=torch.float64, device=self._device, generator=self._generator)
                accept = accept | ((delta < 0.0) & (draws < torch.exp(delta / (temperature * rms))))
            chosen = torch.nonzero(accept).view(-1)
            swaps = int(chosen.numel())
            applied = 0
            length = 0.0
            if swaps:
                first_cell = cell_a[chosen]
                second_cell = cell_b[chosen]
                order = torch.arange(swaps, dtype=torch.int64, device=self._device)
                # Первый по порядку обмен каждой ячейки; обмен применяется, если он первый для обеих своих ячеек.
                earliest = torch.full((self._height * width,), swaps, dtype=torch.int64, device=self._device)
                earliest.scatter_reduce_(
                    0, torch.cat([first_cell, second_cell]), torch.cat([order, order]), reduce="amin", include_self=True
                )
                kept = (earliest[first_cell] == order) & (earliest[second_cell] == order)
                kept_rows = chosen[kept]
                first_cell = first_cell[kept]
                second_cell = second_cell[kept]
                moved_a = point_a[kept_rows]
                moved_b = point_b[kept_rows]
                self._grid[first_cell] = moved_b.to(torch.int32)
                self._grid[second_cell] = moved_a.to(torch.int32)
                self._cell[moved_a] = second_cell
                present = moved_b < count
                self._cell[moved_b[present]] = first_cell[present]
                applied = int(kept_rows.numel())
                length = float(distance[kept_rows].to(torch.float64).sum().item())
            self._steps += 1
            if LOGGER.should_log("layout.device_step"):
                LOGGER.event(
                    "layout.device_step",
                    section=POINT_EXCHANGE,
                    data={
                        "step": 0 if step_index is None else step_index,
                        "pairs": rows,
                        "pairs_requested": int(pairs),
                        "pair_radius": int(radius),
                        "window_reach": reach,
                        "window_cells": int(window[0].size),
                        "favourable": int(favourable.sum().item()),
                        "accepted": swaps,
                        "applied": applied,
                        "temperature": float(temperature),
                        "note": "пары (§5.3), энергии окрестности (§5.5), отбор (§5.6) и обмены без общих ячеек "
                        "(§5.10) — на устройстве",
                    },
                    force=True,
                )
        return applied, length

    def _verify(
        self,
        point_a: Any,
        point_b: Any,
        cell_a: Any,
        cell_b: Any,
        energy_c: Any,
        energy_s: Any,
        window: tuple["np.ndarray", "np.ndarray"],
        reach: int,
        reference: Callable[..., tuple["np.ndarray", "np.ndarray"]],
    ) -> None:
        """Энергии первых пар шага против исходного прохода движка (те же слагаемые, другая редукция)."""
        rows = min(int(point_a.numel()), self.VERIFY_PAIRS)
        width = self._width
        idx_a = point_a[:rows].cpu().numpy()
        idx_b = point_b[:rows].cpu().numpy()
        idx_b = np.where(idx_b >= self._count, -1, idx_b)
        first = cell_a[:rows].cpu().numpy()
        second = cell_b[:rows].cpu().numpy()
        pos_a = np.stack([first // width, first % width], axis=1)
        pos_b = np.stack([second // width, second % width], axis=1)
        expected_c, expected_s = reference(idx_a, idx_b, pos_a, pos_b, window, reach)
        actual_c = energy_c[:rows].cpu().numpy()
        actual_s = energy_s[:rows].cpu().numpy()
        matched = bool(
            np.allclose(actual_c, expected_c, rtol=self.VERIFY_RTOL, atol=self.VERIFY_ATOL)
            and np.allclose(actual_s, expected_s, rtol=self.VERIFY_RTOL, atol=self.VERIFY_ATOL)
        )
        LOGGER.event(
            "layout.device_step.verify",
            section=GPU_IMPLEMENTATION,
            data={
                "pairs": rows,
                "window_cells": int(window[0].size),
                "matched": matched,
                "max_difference": float(
                    max(np.abs(actual_c - expected_c).max(), np.abs(actual_s - expected_s).max())
                ),
                "note": "φ_c, φ_s шага на устройстве равны исходному проходу движка с точностью сложения float32 "
                "(§5.5, §5.12)",
            },
            force=True,
        )
        if not matched:
            raise RuntimeError("device step energies differ from the reference pass")
