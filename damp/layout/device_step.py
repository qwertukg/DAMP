"""Шаг большой раскладки целиком на устройстве (§5.3, §5.5, §5.6, §5.10, §5.12).

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

DeviceLongStep — шаг дальней фазы с гауссовым ядром (режим сглаживания). DeviceShortStep — шаг ближней фазы
(§5.5.2) режима статьи: первая точка — с вероятностью P(p) ∝ −ln Ê(p, r) (§5.9.3), ранняя отсечка пар
по сходству (§5.9.1), энергии окрестности — ядром WindowEnergyKernel по ячейкам окна.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from damp.article_refs import GPU_IMPLEMENTATION, OPTIM_FIRST_POINT, POINT_EXCHANGE
from damp.layout.window_energy_kernel import WindowEnergyKernel
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
        point_a = self._first_points(pairs)
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

    def _first_points(self, pairs: int) -> Any:
        """Первые точки пар шага (§5.3): случайные точки V."""
        return self._torch.randint(0, self._count, (pairs,), device=self._device, generator=self._generator)

    def _apply(self, point_a: Any, point_b: Any, cell_a: Any, cell_b: Any, chosen: Any, distance: Any) -> tuple[int, float]:
        """Применить принятые обмены (строки chosen пар шага): число применённых и сумма их длин.

        Обмены одного шага не делят ячеек (§5.10): обмен применяется, если он первый по порядку пар шага
        для обеих своих ячеек.
        """
        torch = self._torch
        swaps = int(chosen.numel())
        if not swaps:
            return 0, 0.0
        first_cell = cell_a[chosen]
        second_cell = cell_b[chosen]
        order = torch.arange(swaps, dtype=torch.int64, device=self._device)
        # Первый по порядку обмен каждой ячейки; обмен применяется, если он первый для обеих своих ячеек.
        earliest = torch.full((self._height * self._width,), swaps, dtype=torch.int64, device=self._device)
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
        present = moved_b < self._count
        self._cell[moved_b[present]] = first_cell[present]
        self._moved(first_cell, second_cell, moved_a, moved_b)
        return int(kept_rows.numel()), float(distance[kept_rows].to(torch.float64).sum().item())

    def _moved(self, first_cell: Any, second_cell: Any, moved_a: Any, moved_b: Any) -> None:
        """Обмены применены: в ячейках first_cell теперь точки moved_b (N — пусто), в second_cell — moved_a."""

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
            applied, length = self._apply(point_a, point_b, cell_a, cell_b, chosen, distance)
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

class DeviceShortStep(DeviceLongStep):
    """Расстановка на устройстве и шаг ближней фазы режима статьи над ней (§5.5.2, §5.9.1, §5.9.3)."""

    # Ранняя отсечка по сходству (§5.9.1): пары выбираются, пока не набрано нужное число прошедших отсечку
    # или не исчерпано ATTEMPT_FACTOR · (число пар) попыток; недобранное берётся без отсечки — как на хосте.
    ATTEMPT_FACTOR = 5
    # Запас очередной порции попыток над ожидаемым числом (по доле прошедших отсечку в прошлой порции).
    BATCH_MARGIN = 1.1
    # Наименьшая доля прошедших отсечку, на которую рассчитывается порция попыток.
    RATE_FLOOR = 0.05

    # Повторных попыток выбрать вторую ячейку для пар, чья ячейка круга оказалась за сеткой.
    EDGE_ATTEMPTS = 2

    def __init__(self, engine: Any, kernel: WindowEnergyKernel, *, seed: int) -> None:
        super().__init__(engine, None, seed=seed)
        self._kernel = kernel
        # Накопленные веса первых точек P(p) ∝ −ln Ê (§5.9.3); None — первая точка равновероятна.
        self._weights_cdf: Any = None
        # Смещения ячеек круга радиуса r (0 < dy² + dx² ≤ r²) по радиусам и доля пар, прошедших отсечку
        # по сходству на прошлом шаге (по ней выбирается число попыток следующего).
        self._disc_tables: dict[int, Any] = {}
        self._pass_rate = 1.0

    @property
    def kernel(self) -> WindowEnergyKernel:
        return self._kernel

    # ------------------------------------------------------------------ расстановка
    def upload(self, positions: "np.ndarray", version: int) -> None:
        """Расстановка с хоста и коды точек по ячейкам сетки для ядра окна."""
        super().upload(positions, version)
        self._kernel.load(self._grid)

    def _moved(self, first_cell: Any, second_cell: Any, moved_a: Any, moved_b: Any) -> None:
        torch = self._torch
        self._kernel.update(torch.cat([first_cell, second_cell]), torch.cat([moved_b, moved_a]))

    # ------------------------------------------------------------------ энергии точек и веса первой точки
    def point_energies(self, radius: float, *, distance_eps: float, lambda_threshold: float) -> Any:
        """E(c, r) всех точек (§5.7) — тензор устройства float64."""
        return self._kernel.point_energies(
            self._cell, float(radius), distance_eps=distance_eps, lambda_threshold=lambda_threshold
        )

    def refresh_weights(
        self, radius: float, *, energy_eps: float, distance_eps: float, lambda_threshold: float, step_index: int | None
    ) -> None:
        """Веса первой точки P(p) ∝ −ln Ê(p, r) по энергиям всех точек (§5.9.3)."""
        torch = self._torch
        with torch.no_grad():
            energies = self.point_energies(radius, distance_eps=distance_eps, lambda_threshold=lambda_threshold)
            emax = float(energies.max().item()) if self._count else 0.0
            if emax > 0.0:
                weights = torch.clamp(-torch.log(torch.clamp(energies / emax, min=float(energy_eps))), min=0.0)
            else:
                weights = torch.zeros_like(energies)
            total = float(weights.sum().item())
            self._weights_cdf = torch.cumsum(weights, dim=0) if total > 0.0 else None
            mean_energy = float(energies.mean().item()) / emax if emax > 0.0 else 0.0
        LOGGER.event(
            "layout.short.energy_weights",
            section=OPTIM_FIRST_POINT,
            data={
                "energy_radius": radius,
                "points": self._count,
                "step": 0 if step_index is None else step_index,
                "weighted": self._weights_cdf is not None,
                "energy_max": emax,
                "energy_mean_normalized": mean_energy,
                "note": "энергии всех точек на устройстве; первая точка пары — с вероятностью P(p) ∝ −ln Ê (§5.9.3)",
            },
            force=True,
        )

    def _first_points(self, pairs: int) -> Any:
        """Первые точки пар (§5.9.3): по накопленным весам −ln Ê; без весов — равновероятно."""
        torch = self._torch
        cdf = self._weights_cdf
        if cdf is None:
            return super()._first_points(pairs)
        draws = torch.rand(pairs, dtype=torch.float64, device=self._device, generator=self._generator) * cdf[-1]
        return torch.clamp(torch.searchsorted(cdf, draws, right=True), max=self._count - 1)

    def _disc_table(self, radius: int) -> Any:
        """Смещения (dy, dx) всех ячеек круга радиуса r вокруг ячейки, кроме неё самой: тензор K × 2."""
        table = self._disc_tables.get(radius)
        if table is None:
            span = np.arange(-radius, radius + 1, dtype=np.int64)
            dy, dx = np.meshgrid(span, span, indexing="ij")
            distance_sq = dy * dy + dx * dx
            inside = (distance_sq <= radius * radius) & (distance_sq != 0)
            table = self._torch.as_tensor(np.stack([dy[inside], dx[inside]], axis=1), device=self._device)
            if len(self._disc_tables) > 64:
                self._disc_tables.clear()
            self._disc_tables[radius] = table
        return table

    def _sample_pairs(self, pairs: int, radius: int) -> tuple[Any, Any, Any, Any, Any]:
        """Пары шага (§5.3, §5.9.3): точка a (по весам −ln Ê), её ячейка, вторая ячейка и смещение (dy, dx).

        Вторая ячейка — случайная ячейка круга радиуса r вокруг первой: смещение берётся равновероятно
        из таблицы всех смещений круга одним запросом к генератору (то же распределение, что у попыток
        «случайная ячейка квадрата, пока не попадёт в круг»). Повторные попытки нужны только парам, чья
        ячейка оказалась за сеткой; пара без ячейки в сетке в шаг не входит.
        """
        torch = self._torch
        device = self._device
        generator = self._generator
        height, width = self._height, self._width
        table = self._disc_table(int(radius))
        choices = int(table.shape[0])
        point_a = self._first_points(pairs)
        cell_a = self._cell[point_a]
        y1 = cell_a // width
        x1 = cell_a - y1 * width
        step = table[torch.randint(0, choices, (pairs,), device=device, generator=generator)]
        y2 = y1 + step[:, 0]
        x2 = x1 + step[:, 1]
        found = (y2 >= 0) & (y2 < height) & (x2 >= 0) & (x2 < width)
        for _ in range(self.EDGE_ATTEMPTS):
            retry = table[torch.randint(0, choices, (pairs,), device=device, generator=generator)]
            y2 = y1 + retry[:, 0]
            x2 = x1 + retry[:, 1]
            good = (y2 >= 0) & (y2 < height) & (x2 >= 0) & (x2 < width) & ~found
            step = torch.where(good.unsqueeze(1), retry, step)
            found |= good
        keep = torch.nonzero(found).view(-1)
        point_a, cell_a, step = point_a[keep], cell_a[keep], step[keep]
        dy = step[:, 0]
        dx = step[:, 1]
        return point_a, cell_a, cell_a + dy * width + dx, dy, dx

    # ------------------------------------------------------------------ шаг
    def _cut_pairs(self, wanted: int, radius: int, similarity_cutoff: float | None) -> tuple[Any, ...]:
        """Пары шага (§5.3, §5.9.1, §5.9.3): точки a и b (N — вторая ячейка пуста), их ячейки, смещение (dy, dx),
        число пар, отброшенных отсечкой, и число попыток.

        Пара из двух точек с sim(a, b) < отсечки в шаг не входит (§5.9.1: «only one of the points should be
        zero» — пара с пустой второй ячейкой остаётся). Пары выбираются порциями, пока не набрано wanted
        прошедших отсечку или не исчерпано ATTEMPT_FACTOR · wanted попыток; недобранное берётся без отсечки.
        """
        torch = self._torch
        count = self._count
        parts: list[tuple[Any, ...]] = []
        collected = 0
        attempts = 0
        rejected = 0

        def take(batch: int, cutoff: float | None) -> int:
            nonlocal rejected
            point_a, cell_a, cell_b, dy, dx = self._sample_pairs(batch, radius)
            point_b = self._grid[cell_b].to(torch.int64)
            values = (point_a, point_b, cell_a, cell_b, dy, dx)
            if cutoff is not None and int(point_a.numel()):
                similarity = self._kernel.pair_similarity(point_a, point_b)
                keep = torch.nonzero((point_b == count) | (similarity >= float(cutoff))).view(-1)
                rejected += int(point_a.numel()) - int(keep.numel())
                values = tuple(value[keep] for value in values)
            parts.append(values)
            return int(values[0].numel())

        if similarity_cutoff is None:
            collected += take(wanted, None)
            attempts = wanted
        else:
            limit = max(wanted * self.ATTEMPT_FACTOR, wanted)
            # Первая порция — по доле прошедших отсечку на прошлом шаге: обычно хватает одной порции.
            rate = self._pass_rate
            while collected < wanted and attempts < limit:
                missing = wanted - collected
                batch = min(int(math.ceil(missing / max(rate, self.RATE_FLOOR) * self.BATCH_MARGIN)) + 16, limit - attempts)
                passed = take(batch, similarity_cutoff)
                attempts += batch
                collected += passed
                rate = passed / float(batch)
                self._pass_rate = max(rate, self.RATE_FLOOR)
            if collected < wanted:
                # Не добрано за все попытки: остаток — пары без отсечки, как в дальней фазе.
                collected += take(wanted - collected, None)
        point_a, point_b, cell_a, cell_b, dy, dx = (
            torch.cat([part[index] for part in parts])[:wanted] for index in range(6)
        )
        return point_a, point_b, cell_a, cell_b, dy, dx, rejected, attempts

    def short_step(
        self,
        *,
        pairs: int,
        radius: int,
        local_radius: int,
        similarity_cutoff: float | None,
        euclidean: bool,
        distance_eps: float,
        lambda_threshold: float,
        step_index: int | None,
        reference: Callable[..., tuple["np.ndarray", "np.ndarray"]] | None = None,
    ) -> tuple[int, float]:
        """Один шаг ближней фазы: число применённых обменов и сумма их длин ‖p1 − p2‖.

        reference(idx_a, idx_b, pos_a, pos_b, reach) — исходный проход движка для сверки энергий первых пар.
        """
        torch = self._torch
        with torch.no_grad():
            wanted = int(pairs)
            point_a, point_b, cell_a, cell_b, dy, dx, rejected, attempts = self._cut_pairs(
                wanted, int(radius), similarity_cutoff
            )
            rows = int(point_a.numel())
            if rows == 0:
                return 0, 0.0
            distance = torch.sqrt((dy * dy + dx * dx).to(torch.float32))
            reach = WindowEnergyKernel.reach(float(distance.max().item()), local_radius)
            energy_c, energy_s = self._kernel.pair_energies(
                point_a, point_b, cell_a, cell_b,
                reach=reach, local_radius=local_radius, euclidean=euclidean,
                distance_eps=distance_eps, lambda_threshold=lambda_threshold,
            )
            if reference is not None:
                self._verify_short(point_a, point_b, cell_a, cell_b, energy_c, energy_s, reach, reference)
            # Окрестность (§5.5.2): обмен выгоден при φ_s > φ_c (§5.6).
            chosen = torch.nonzero(energy_s > energy_c).view(-1)
            favourable = int(chosen.numel())
            applied, length = self._apply(point_a, point_b, cell_a, cell_b, chosen, distance)
            self._steps += 1
            if LOGGER.should_log("layout.device_step"):
                LOGGER.event(
                    "layout.device_step",
                    section=POINT_EXCHANGE,
                    data={
                        "step": 0 if step_index is None else step_index,
                        "mode": "short",
                        "pairs": rows,
                        "pairs_requested": wanted,
                        "rejected_similarity": rejected,
                        "attempts": attempts,
                        "pair_radius": int(radius),
                        "local_radius": int(local_radius),
                        "window_reach": reach,
                        "favourable": favourable,
                        "applied": applied,
                        "lambda_threshold": float(lambda_threshold),
                        "note": "пары (§5.3, §5.9.1, §5.9.3), энергии окрестности (§5.5.2), отбор (§5.6) и обмены "
                        "без общих ячеек (§5.10) — на устройстве",
                    },
                    force=True,
                )
        return applied, length

    def _verify_short(
        self,
        point_a: Any,
        point_b: Any,
        cell_a: Any,
        cell_b: Any,
        energy_c: Any,
        energy_s: Any,
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
        expected_c, expected_s = reference(idx_a, idx_b, pos_a, pos_b, reach)
        matched = self._kernel.verify(
            energy_c[:rows].cpu().numpy(), energy_s[:rows].cpu().numpy(), expected_c, expected_s, kind="pairs"
        )
        if not matched:
            raise RuntimeError("device short step energies differ from the reference pass")

