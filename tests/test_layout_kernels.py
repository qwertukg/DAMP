"""Ядра шага раскладки не меняют её результат: те же знаки φ_c − φ_s, те же суммы float32, те же пары и обмены.

Проверяется каждое звено ускорения по отдельности и раскладка целиком:
  * скомпилированная выборка пар (§5.3, §5.9.1, §5.9.3) — те же пары и состояние генератора, что у numpy;
  * знак φ_c − φ_s ядром по строкам сходства (§5.5.1, §5.5.2, §5.6) — тот же, что у исходного прохода;
  * суммы пар «у нуля разности» по малым участкам редукции и репликой редукции на хосте — побитово те же;
  * нижняя граница потенциала точки (§5.5.1) — не снимает ни одной пары, которой обмен выгоден;
  * энергии точек (§5.7) ядром — побитово те же, что у прохода по окну ячеек.
"""
from __future__ import annotations

import os
import random
import unittest

os.environ.setdefault("RERUN", "off")

import numpy as np  # noqa: E402

from damp.encoding.bitarray import BitArray  # noqa: E402
from damp.layout.damp_layout import AdaptiveLayoutConfig, Layout, ShortLayoutOptimConfig  # noqa: E402
from damp.layout.pair_sampler import CompiledPairSampler  # noqa: E402
from damp.layout.pair_sign_kernel import PairSignKernel  # noqa: E402
from damp.layout.payload import LayoutEncodedCode  # noqa: E402
from damp.layout.potential_bounds import PotentialBounds  # noqa: E402
from damp.layout.reduction_replica import ReductionReplica  # noqa: E402
from damp.logging import LOGGER  # noqa: E402

LOGGER.configure_intervals({}, default_interval=10**9)

_CODE_LENGTH = 128
_GROUPS = 9


def _clustered_codes(points: int, seed: int) -> dict[float, list[LayoutEncodedCode]]:
    """Коды из нескольких групп: внутри группы сходство выше порога, между группами — ниже; есть повторы кодов."""
    rng = random.Random(seed)
    prototypes = [rng.sample(range(_CODE_LENGTH), 24) for _ in range(_GROUPS)]
    codes: dict[float, list[LayoutEncodedCode]] = {}
    for index in range(points):
        group = index % _GROUPS
        bits = list(prototypes[group])
        for _ in range(rng.randrange(0, 5)):
            bits[rng.randrange(len(bits))] = rng.randrange(_CODE_LENGTH)
        codes.setdefault(float(group), []).append(
            LayoutEncodedCode(code=BitArray.from_indices(_CODE_LENGTH, sorted(set(bits))), label=str(index))
        )
    return codes


def _layout(points: int = 900, *, seed: int = 3, **kwargs) -> Layout:
    params = dict(
        similarity="cosine", lambda_threshold=0.6, eta=None, use_gpu=True, seed=seed,
        long_distance="euclidean", short_distance="squared",
    )
    params.update(kwargs)
    return Layout(_clustered_codes(points, seed), **params)


def _occupied(layout: Layout, count: int, radius: int) -> np.ndarray:
    """Пары шага в виде строк (a, b или −1, ячейка a, ячейка b), как их получает расчёт энергий."""
    layout._sync_arrays()
    cells = layout._sample_pairs_long_numpy(count, radius)
    return CompiledPairSampler.occupied_pairs(np.ascontiguousarray(cells, dtype=np.int64), layout._grid_np)


class _GpuCase(unittest.TestCase):
    def _gpu_layout(self, **kwargs) -> Layout:
        layout = _layout(**kwargs)
        engine = layout._torch_engine
        if engine is None or not PairSignKernel.supported(engine):
            self.skipTest("нужны torch с CUDA и jiterator")
        return layout

    def _ordered(self, layout: Layout, steps: int = 60, pairs: int = 300) -> None:
        """Несколько шагов дальней фазы: частично упорядоченная раскладка."""
        for step in range(steps):
            layout.step(pairs_per_step=pairs, pair_radius=layout.width // 2, mode="long", local_radius=None,
                        step_index=step)
        layout._sync_arrays()
        layout._ensure_torch_positions()

    def _dense(self, layout: Layout, occupied: np.ndarray, *, neighbourhood: bool, local_radius, euclidean: bool):
        return layout._torch_engine.pair_energies(
            occupied[:, 0], occupied[:, 1], occupied[:, 2:4], occupied[:, 4:6],
            neighbourhood=neighbourhood, local_radius=local_radius, kernel="inverse", euclidean=euclidean,
            distance_eps=layout._distance_eps, kernel_width=1.0, lambda_threshold=layout._lambda, eta=None,
        )


class CompiledSamplerTests(unittest.TestCase):
    """§5.3, §5.9.1, §5.9.3: скомпилированная выборка даёт те же пары и то же состояние генератора, что numpy."""

    def setUp(self) -> None:
        self.layout = _layout(points=700, use_gpu=False)
        if self.layout._compiled_sampler is None:
            self.skipTest("numba недоступна")
        self.layout._sync_arrays()

    def test_long_pairs_match_numpy(self) -> None:
        layout = self.layout
        sampler = layout._compiled_sampler
        for count, radius in ((1, 1), (57, 1), (300, 2), (300, layout.width // 2), (200, None)):
            clone = sampler.clone_generator(layout._np_rng)
            expected = layout._sample_pairs_long_numpy(count, radius)
            got = sampler.long_pairs(clone, layout._pos_np, count, radius)
            self.assertTrue(np.array_equal(expected, got), f"count={count} radius={radius}")
            self.assertEqual(clone.bit_generator.state, layout._np_rng.bit_generator.state)

    def test_nearby_cells_in_tile_and_single_cell_tile_match_numpy(self) -> None:
        layout = self.layout
        sampler = layout._compiled_sampler
        points = np.arange(0, 400, 3)
        y1 = layout._pos_np[points, 0].copy()
        x1 = layout._pos_np[points, 1].copy()
        single = (int(y1[0]), int(y1[0]), int(x1[0]), int(x1[0]))
        for radius, bounds in ((1, None), (3, (0, layout.height // 2, 0, layout.width - 1)), (2, (4, 9, 4, 9)),
                               (1, single), (None, (0, 5, 0, 5))):
            clone = sampler.clone_generator(layout._np_rng)
            expected = layout._sample_nearby_np(y1, x1, radius, bounds)
            got = sampler.nearby(clone, y1, x1, radius, bounds)
            self.assertTrue(np.array_equal(expected[0], got[0]) and np.array_equal(expected[1], got[1]),
                            f"radius={radius} bounds={bounds}")
            self.assertEqual(clone.bit_generator.state, layout._np_rng.bit_generator.state)

    def test_short_tile_pairs_match_numpy_with_and_without_weights(self) -> None:
        layout = self.layout
        sampler = layout._compiled_sampler
        rng = np.random.default_rng(11)
        candidates = np.flatnonzero(layout._pos_np[:, 0] < layout.height // 2)
        bounds = (0, layout.height // 2 - 1, 0, layout.width - 1)
        for weighted, cutoff, radius, count in ((True, 0.6, 3, 80), (False, 0.6, 2, 50), (True, None, 4, 60),
                                                (True, 0.95, 1, 40), (True, 0.6, 3, 600), (False, 0.7, 2, 450)):
            weights = rng.random(candidates.size) if weighted else None
            total = float(weights.sum()) if weighted else 0.0
            probabilities = None if weights is None else weights / total
            clone = sampler.clone_generator(layout._np_rng)
            expected = layout._short_tile_pairs_numpy(count, candidates, probabilities, radius, cutoff, bounds)
            got = sampler.short_tile_pairs(
                clone, candidates, weights, total, pos=layout._pos_np, grid=layout._grid_np,
                packed_codes=layout._packed_codes, ones=layout._ones_float64, cosine=True,
                similarity_cutoff=cutoff, count=count, radius=radius, bounds=bounds,
            )
            self.assertTrue(np.array_equal(expected[0], got[0]), f"weighted={weighted} cutoff={cutoff}")
            self.assertEqual(expected[1:], got[1:])
            self.assertEqual(clone.bit_generator.state, layout._np_rng.bit_generator.state)

    def test_tile_groups_equal_searchsorted_selection(self) -> None:
        layout = self.layout
        y_ends = np.asarray([3, 11, layout.height - 1], dtype=np.int64)
        x_ends = np.asarray([0, 7, layout.width - 1], dtype=np.int64)
        order, starts = CompiledPairSampler.tile_groups(layout._pos_np, y_ends, x_ends, 3, 9)
        tile = (np.searchsorted(y_ends, layout._pos_np[:, 0], side="left") * 3
                + np.searchsorted(x_ends, layout._pos_np[:, 1], side="left"))
        for index in range(9):
            self.assertTrue(np.array_equal(order[starts[index] : starts[index + 1]], np.flatnonzero(tile == index)))


class PairSignKernelTests(_GpuCase):
    """§5.5.1, §5.5.2, §5.6: решения ядра совпадают с исходным проходом, суммы «у нуля разности» — побитово."""

    def _kernel(self, layout: Layout):
        sparse = layout._sparse_long_engine()
        self.assertIsNotNone(sparse)
        kernel = layout._pair_sign_kernel(sparse)
        self.assertIsNotNone(kernel)
        return sparse, kernel

    def test_long_status_agrees_with_dense_pass(self) -> None:
        layout = self._gpu_layout()
        sparse, kernel = self._kernel(layout)
        for ordered in (False, True):
            if ordered:
                self._ordered(layout)
            layout._ensure_torch_positions()
            for radius in (layout.width // 2, 3):
                occupied = _occupied(layout, 400, radius)
                table, widest = kernel.pair_table(occupied[:, 0], occupied[:, 1], occupied[:, 2:4], occupied[:, 4:6],
                                                  layout._code_class)
                status = kernel.status_from_table(table, occupied.shape[0], widest, kind="long", euclidean=True)
                dense_c, dense_s = self._dense(layout, occupied, neighbourhood=False, local_radius=None, euclidean=True)
                sign = np.sign(dense_c.astype(np.float64) - dense_s.astype(np.float64))
                self.assertTrue(np.all(sign[status == 1] > 0))
                self.assertTrue(np.all(sign[status == -1] < 0))
                self.assertTrue(np.all(sign[status == PairSignKernel.EQUAL] == 0))
                self.assertGreater(int((status != 0).sum()), occupied.shape[0] // 2)

    def test_neighbourhood_status_agrees_with_window_pass(self) -> None:
        layout = self._gpu_layout()
        sparse, kernel = self._kernel(layout)
        self._ordered(layout)
        for euclidean, radius in ((False, 3), (True, 2)):
            occupied = _occupied(layout, 400, 2)
            table, widest = kernel.pair_table(occupied[:, 0], occupied[:, 1], occupied[:, 2:4], occupied[:, 4:6],
                                              layout._code_class)
            status = kernel.status_from_table(
                table, occupied.shape[0], widest, kind="neighbourhood", euclidean=euclidean,
                local_radius=float(radius), distance_eps=layout._distance_eps,
            )
            dense_c, dense_s = self._dense(layout, occupied, neighbourhood=True, local_radius=radius, euclidean=euclidean)
            sign = np.sign(dense_c.astype(np.float64) - dense_s.astype(np.float64))
            self.assertTrue(np.all(sign[status == 1] > 0))
            self.assertTrue(np.all(sign[status == -1] < 0))
            self.assertTrue(np.all(sign[status == PairSignKernel.EQUAL] == 0))

    def test_reference_sums_and_replica_equal_dense_pass_bitwise(self) -> None:
        layout = self._gpu_layout(points=1100)
        engine = layout._torch_engine
        engine.CHUNK_ELEMENTS = 1100 * 37  # несколько порций плотного прохода, последняя — неполная
        engine.LOCAL_CHUNK_ELEMENTS = 1100 * 23
        sparse, kernel = self._kernel(layout)
        replica = layout._reduction_replica(sparse)
        self._ordered(layout)
        # Радиус окрестности 7 даёт окно 17 × 17 = 289 ячеек: редукция читает строку четвёрками (реплика применима);
        # радиус 3 — окно 81 ячейка: порядок сложения у редукции другой, такие пары считает устройство.
        cases = (("long", True, None, layout.width // 2), ("long", False, None, 3),
                 ("neighbourhood", False, 7, 2), ("neighbourhood", True, 7, 2), ("neighbourhood", False, 3, 2))
        replica_cases = 0
        for kind, euclidean, local_radius, pair_radius in cases:
            occupied = _occupied(layout, 500, pair_radius)
            idx_a, idx_b, pos_a, pos_b = occupied[:, 0], occupied[:, 1], occupied[:, 2:4], occupied[:, 4:6]
            dense_c, dense_s = self._dense(layout, occupied, neighbourhood=kind == "neighbourhood",
                                           local_radius=local_radius, euclidean=euclidean)
            pending = np.arange(0, occupied.shape[0], 7, dtype=np.int64)
            sums = kernel.reference_sums(idx_a, idx_b, pos_a, pos_b, pending, kind=kind, euclidean=euclidean,
                                         local_radius=local_radius, distance_eps=layout._distance_eps)
            self.assertIsNotNone(sums, f"{kind}: малые участки редукции неприменимы")
            self.assertTrue(np.array_equal(sums[0], dense_c[pending]) and np.array_equal(sums[1], dense_s[pending]),
                            f"{kind} euclidean={euclidean}: участки редукции")
            if replica is not None:
                geometry = kernel.reference_geometry(pos_a, pos_b, kind=kind, local_radius=local_radius)
                host = replica.pair_sums(idx_a, idx_b, pos_a, pos_b, pending, layout._pos_np, kind=kind,
                                         euclidean=euclidean, geometry=geometry, local_radius=local_radius,
                                         distance_eps=layout._distance_eps)
                if ReductionReplica.shape(geometry[0], geometry[1]) is None:
                    self.assertIsNone(host, "форма порции не воспроизводится: реплика должна отказаться")
                else:
                    self.assertIsNotNone(host, f"{kind}: реплика редукции неприменима")
                    self.assertTrue(
                        np.array_equal(host[0], dense_c[pending]) and np.array_equal(host[1], dense_s[pending]),
                        f"{kind} euclidean={euclidean}: реплика редукции",
                    )
                    replica_cases += 1
            self.assertGreater(float(np.abs(dense_c).max()), 0.0)
        if replica is not None:
            self.assertGreaterEqual(replica_cases, 3, "реплика должна была посчитать дальнюю фазу и широкое окно")

    def test_identical_codes_give_exactly_equal_sums(self) -> None:
        layout = self._gpu_layout()
        classes = layout._code_class
        pairs = []
        seen: dict[int, int] = {}
        for point, code in enumerate(classes.tolist()):
            if code in seen:
                pairs.append((seen[code], point))
            seen[code] = point
        if not pairs:
            self.skipTest("в наборе нет повторяющихся кодов")
        layout._sync_arrays()
        layout._ensure_torch_positions()
        idx_a = np.asarray([a for a, _ in pairs], dtype=np.int64)
        idx_b = np.asarray([b for _, b in pairs], dtype=np.int64)
        occupied = np.concatenate([idx_a[:, None], idx_b[:, None], layout._pos_np[idx_a], layout._pos_np[idx_b]], axis=1)
        dense_c, dense_s = self._dense(layout, occupied, neighbourhood=False, local_radius=None, euclidean=True)
        self.assertTrue(np.array_equal(dense_c, dense_s), "у точек с одинаковыми кодами φ_c = φ_s точно")
        sparse, kernel = self._kernel(layout)
        table, widest = kernel.pair_table(idx_a, idx_b, occupied[:, 2:4], occupied[:, 4:6], classes)
        status = kernel.status_from_table(table, len(pairs), widest, kind="long", euclidean=True)
        self.assertTrue(np.all(status == PairSignKernel.EQUAL))

    def test_point_energies_equal_window_pass_bitwise(self) -> None:
        layout = self._gpu_layout(points=1100)
        engine = layout._torch_engine
        engine.LOCAL_CHUNK_ELEMENTS = 1100 * 23
        sparse, kernel = self._kernel(layout)
        self._ordered(layout)
        for radius in (1, 3):
            window = engine.point_energies(radius, lambda_threshold=layout._lambda, eta=None,
                                           distance_eps=layout._distance_eps)
            got = kernel.point_energies(layout._pos_np, float(radius), distance_eps=layout._distance_eps)
            self.assertIsNotNone(got)
            self.assertTrue(np.array_equal(got, window), f"radius={radius}")
            self.assertGreater(float(window.max()), 0.0)


class ReductionReplicaTests(_GpuCase):
    """§5.12: порядок сложения редукции float32 устройства воспроизводится на хосте побитово."""

    def test_row_sums_equal_device_reduction(self) -> None:
        import torch

        from damp.layout import reduction_replica as module

        layout = self._gpu_layout(points=200)
        device = layout._torch_engine.torch_device
        rng = np.random.default_rng(5)
        checked = 0
        for width, rows in ((2401, 40), (441, 33), (8693, 64), (8691, 20), (20000, 5), (1369, 16)):
            shape = ReductionReplica.shape(width, rows)
            if shape is None:
                continue
            block_width, warps, step = shape
            data = np.zeros((rows, width), dtype=np.float32)
            for row in range(rows):
                places = rng.choice(width, size=min(width, 300), replace=False)
                data[row, places] = np.exp(rng.random(places.size) * 4.0).astype(np.float32)
            data[0, : min(width, 9)] = 1.5
            data[1, max(0, width - 9) :] = 2.5
            expected = torch.as_tensor(data, device=device).sum(dim=1).cpu().numpy()
            for row in range(rows):
                places = np.flatnonzero(data[row])
                got = module._row_sum(places, data[row, places], width, (row * width) % 4, block_width, warps, step)
                self.assertEqual(got, expected[row], f"width={width} rows={rows} row={row}")
                checked += 1
        self.assertGreater(checked, 100)


class PotentialBoundsTests(_GpuCase):
    """§5.5.1: граница потенциала снимает только пары с φ_c < φ_s; суммы после обменов равны построенным заново."""

    def _bounds(self, layout: Layout) -> tuple[PairSignKernel, PotentialBounds]:
        sparse = layout._sparse_long_engine()
        kernel = layout._pair_sign_kernel(sparse)
        engine = layout._torch_engine
        if not PotentialBounds.supported(engine, (layout.height, layout.width)):
            self.skipTest("граница потенциала недоступна")
        bounds = PotentialBounds(engine, sparse.rows, grid=4, segment=16, similarity="cosine",
                                 grid_shape=(layout.height, layout.width))
        return kernel, bounds

    def test_bounded_pairs_are_unfavourable_in_dense_pass(self) -> None:
        layout = self._gpu_layout(points=1500)
        kernel, bounds = self._bounds(layout)
        self._ordered(layout, steps=150, pairs=400)
        bounds.rebuild()
        bounded_total = 0
        for radius in (layout.width // 2, layout.width // 4, 3):
            occupied = _occupied(layout, 600, radius)
            table, widest = kernel.pair_table(occupied[:, 0], occupied[:, 1], occupied[:, 2:4], occupied[:, 4:6],
                                              layout._code_class)
            status = kernel.status_from_table(table, occupied.shape[0], widest, kind="long", euclidean=True,
                                              bounds=bounds)
            dense_c, dense_s = self._dense(layout, occupied, neighbourhood=False, local_radius=None, euclidean=True)
            bounded = status == PairSignKernel.BOUNDED
            bounded_total += int(bounded.sum())
            self.assertTrue(np.all(dense_c[bounded] < dense_s[bounded]), "граница сняла пару с φ_c ≥ φ_s")
            sign = np.sign(dense_c.astype(np.float64) - dense_s.astype(np.float64))
            self.assertTrue(np.all(sign[status == 1] > 0) and np.all(sign[status == -1] < 0))
        self.assertGreater(bounded_total, 0, "в упорядоченной раскладке граница должна снимать пары")

    def test_moves_keep_sums_equal_to_rebuild(self) -> None:
        layout = self._gpu_layout(points=1500)
        kernel, bounds = self._bounds(layout)
        self._ordered(layout, steps=40)
        bounds.rebuild()
        layout._bounds = bounds
        layout._bounds_on = True
        layout._bounds_move_limit = 10**9
        layout._bounds_enabled = False  # расчёт шага границу не трогает: проверяется только учёт обменов
        for step in range(25):
            layout.step(pairs_per_step=300, pair_radius=layout.width // 3, mode="long", local_radius=None,
                        step_index=step)
        layout._ensure_torch_positions()
        self.assertTrue(bounds.valid)
        incremental = bounds.snapshot()
        bounds.rebuild()
        rebuilt = bounds.snapshot()
        self.assertTrue(np.array_equal(incremental["groups"], rebuilt["groups"]), "массы и моменты групп")
        scale = float(np.abs(rebuilt["energy"]).max())
        self.assertLess(float(np.abs(incremental["energy"] - rebuilt["energy"]).max()), 1e-5 * scale)
        self.assertTrue(np.array_equal(incremental["residual"], rebuilt["residual"]))


class LayoutIdentityTests(_GpuCase):
    """§5.5–§5.7, §7.1.7: раскладка с ядрами, границей и репликой совпадает с раскладкой по блокам строк torch."""

    def _run(self, **kwargs) -> tuple[list[tuple[int, int]], list[float], Layout]:
        layout = self._gpu_layout(points=1500, sparse_verify_every=11, sampler_verify_every=7, **kwargs)
        engine = layout._torch_engine
        engine.CHUNK_ELEMENTS = 1500 * 41
        layout._BOUNDS_MIN_ENTRIES = 0  # граница включается и на малом наборе
        adaptive = AdaptiveLayoutConfig(
            start_radius=layout.width // 2, end_radius=1, swap_ratio_trigger=0.6, lambda_step=0.05,
            lambda_max=0.75, radius_schedule="swaps", radius_factor=0.5, swap_window=3,
        )
        layout.run(steps=260, pairs_per_step=150, pair_radius=layout.width // 2, mode="long",
                   min_swap_ratio=0.0, min_swap_window=None, log_every=None, log_visuals=False,
                   energy_radius=3, energy_every=20, adaptive_params=adaptive)
        layout.run(steps=50, pairs_per_step=150, pair_radius=3, mode="short", local_radius=3,
                   min_swap_ratio=0.0, min_swap_window=None, log_every=None, log_visuals=False,
                   short_optim=ShortLayoutOptimConfig(energy_radius=3, energy_recalc_every=10, partitions=4,
                                                      similarity_cutoff=0.6))
        return layout.positions(), layout._compute_point_energies(3), layout

    def test_layout_identical_with_and_without_kernels(self) -> None:
        fast_positions, fast_energies, fast = self._run()
        plain_positions, plain_energies, plain = self._run(
            pair_kernel=False, compiled_sampler=False, potential_bounds=False, reduction_replica=False
        )
        self.assertIsNone(plain._pair_kernel)
        self.assertIsNotNone(fast._pair_kernel)
        self.assertGreater(fast._bounded_pairs, 0, "граница потенциала должна была работать")
        self.assertEqual(fast_positions, plain_positions)
        self.assertEqual(fast_energies, plain_energies)


if __name__ == "__main__":
    unittest.main()
