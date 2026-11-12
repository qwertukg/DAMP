"""Ускорение раскладки не меняет её результат: те же числа, те же обмены, тот же поток случайных чисел.

Дальняя энергия §5.5.1, окрестность §5.5.2 и энергии точек §5.7 по ненулевым sim_λ сверяются с исходными
проходами движка побитово; выборка пар §5.3 — с прежним порядком запросов к генератору.
"""
from __future__ import annotations

import os
import random
import unittest

os.environ.setdefault("RERUN", "off")

import numpy as np  # noqa: E402

from damp.encoding.bitarray import BitArray  # noqa: E402
from damp.layout.damp_layout import AdaptiveLayoutConfig, Layout, ShortLayoutOptimConfig  # noqa: E402
from damp.layout.payload import LayoutEncodedCode  # noqa: E402
from damp.layout.sparse_long_energy import SimilarityRows, SparseLongEnergy  # noqa: E402
from damp.logging import LOGGER  # noqa: E402

LOGGER.configure_intervals({}, default_interval=10**9)

_CODE_LENGTH = 96
_GROUPS = 12


def _clustered_codes(points: int, seed: int) -> dict[float, list[LayoutEncodedCode]]:
    """Коды из нескольких групп: внутри группы сходство выше порога, между группами — ниже."""
    rng = random.Random(seed)
    prototypes = [rng.sample(range(_CODE_LENGTH), 20) for _ in range(_GROUPS)]
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


def _layout(points: int = 400, *, seed: int = 5, use_gpu: bool = True, **kwargs) -> Layout:
    params = dict(
        similarity="cosine", lambda_threshold=0.6, eta=None, use_gpu=use_gpu, seed=seed,
        long_distance="euclidean", short_distance="squared",
    )
    params.update(kwargs)
    return Layout(_clustered_codes(points, seed), **params)


def _pairs(layout: Layout, count: int, radius: int):
    """Пары шага в том же виде, в каком их получает движок: (точки a, точки b или −1, ячейки a, ячейки b)."""
    layout._sync_arrays()
    cells = layout._sample_pairs_long_np(count, radius)
    idx_a = layout._grid_np[cells[:, 0], cells[:, 1]]
    idx_b = layout._grid_np[cells[:, 2], cells[:, 3]]
    keep = (idx_a >= 0) | (idx_b >= 0)
    cells, idx_a, idx_b = cells[keep], idx_a[keep], idx_b[keep]
    flip = idx_a < 0
    idx_a, idx_b = np.where(flip, idx_b, idx_a), np.where(flip, idx_a, idx_b)
    pos_a = np.where(flip[:, None], cells[:, 2:4], cells[:, 0:2])
    pos_b = np.where(flip[:, None], cells[:, 0:2], cells[:, 2:4])
    return idx_a, idx_b, pos_a, pos_b


def _reference_nearby(layout: Layout, rng, y1, x1, radius, bounds):
    """Прежний выбор второй ячейки пары (§5.3): обход по маске, по два запроса смещений на попытку."""
    count = int(y1.size)
    if bounds is None:
        y_lo = np.zeros(count, dtype=np.int64)
        y_hi = np.full(count, layout.height - 1, dtype=np.int64)
        x_lo = np.zeros(count, dtype=np.int64)
        x_hi = np.full(count, layout.width - 1, dtype=np.int64)
    else:
        y_lo = np.full(count, bounds[0], dtype=np.int64)
        y_hi = np.full(count, bounds[1], dtype=np.int64)
        x_lo = np.full(count, bounds[2], dtype=np.int64)
        x_hi = np.full(count, bounds[3], dtype=np.int64)
        single = (y_lo == y_hi) & (x_lo == x_hi) & (y_lo == y1) & (x_lo == x1)
        y_lo[single] = 0
        y_hi[single] = layout.height - 1
        x_lo[single] = 0
        x_hi[single] = layout.width - 1
    y2 = np.empty(count, dtype=np.int64)
    x2 = np.empty(count, dtype=np.int64)
    pending = np.ones(count, dtype=bool)
    if radius is not None:
        for _ in range(50):
            idx = np.flatnonzero(pending)
            if idx.size == 0:
                break
            dy = rng.integers(-radius, radius + 1, idx.size)
            dx = rng.integers(-radius, radius + 1, idx.size)
            cy = y1[idx] + dy
            cx = x1[idx] + dx
            ok = (
                (dy * dy + dx * dx <= radius * radius)
                & (cy >= y_lo[idx]) & (cy <= y_hi[idx])
                & (cx >= x_lo[idx]) & (cx <= x_hi[idx])
                & ~((cy == y1[idx]) & (cx == x1[idx]))
            )
            hit = idx[ok]
            y2[hit] = cy[ok]
            x2[hit] = cx[ok]
            pending[hit] = False
    while True:
        idx = np.flatnonzero(pending)
        if idx.size == 0:
            break
        cy = rng.integers(y_lo[idx], y_hi[idx] + 1)
        cx = rng.integers(x_lo[idx], x_hi[idx] + 1)
        ok = ~((cy == y1[idx]) & (cx == x1[idx]))
        hit = idx[ok]
        y2[hit] = cy[ok]
        x2[hit] = cx[ok]
        pending[hit] = False
    return y2, x2


class PairSamplingStreamTests(unittest.TestCase):
    """§5.3, §5.9.1: выборка пар выдаёт те же ячейки и оставляет генератор в том же состоянии, что прежняя."""

    def test_nearby_cells_and_generator_state_match_reference(self) -> None:
        layout = _layout(points=300, use_gpu=False)
        layout._sync_arrays()
        cases = [
            (700, 3, None),
            (700, 1, None),
            (300, 40, None),
            (60, 2, (3, 9, 2, 8)),
            (200, 1, (0, 4, 0, 5)),
            (40, None, None),
            (40, None, (2, 6, 2, 6)),
            (1, 1, None),
            (0, 2, None),
        ]
        for count, radius, bounds in cases:
            seed = 100 + count + (radius or 0)
            first = np.random.default_rng(seed).integers(0, layout._point_count, count)
            y1 = layout._pos_np[first, 0]
            x1 = layout._pos_np[first, 1]
            layout._np_rng = np.random.default_rng(seed + 1)
            reference_rng = np.random.default_rng(seed + 1)
            got = layout._sample_nearby_np(y1, x1, radius, bounds)
            want = _reference_nearby(layout, reference_rng, y1, x1, radius, bounds)
            self.assertTrue(np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1]), (count, radius, bounds))
            self.assertEqual(layout._np_rng.bit_generator.state, reference_rng.bit_generator.state)

    def test_single_cell_tile_uses_whole_grid_like_reference(self) -> None:
        layout = _layout(points=120, use_gpu=False)
        layout._sync_arrays()
        y, x = (int(v) for v in layout._pos_np[7])
        y1 = np.array([y, y, int(layout._pos_np[8, 0])])
        x1 = np.array([x, x, int(layout._pos_np[8, 1])])
        for radius in (1, 2, None):
            layout._np_rng = np.random.default_rng(3)
            reference_rng = np.random.default_rng(3)
            got = layout._sample_nearby_np(y1, x1, radius, (y, y, x, x))
            want = _reference_nearby(layout, reference_rng, y1, x1, radius, (y, y, x, x))
            self.assertTrue(np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1]))
            self.assertEqual(layout._np_rng.bit_generator.state, reference_rng.bit_generator.state)

    def test_one_request_of_offsets_equals_two_requests(self) -> None:
        self.assertTrue(Layout._merged_offset_draws())

    def test_packed_similarity_equals_bit_products(self) -> None:
        """§2.2.4: число общих бит через popcount упакованных кодов — то же, что сумма произведений бит."""
        layout = _layout(points=200, use_gpu=False)
        rng = np.random.default_rng(1)
        idx_a = rng.integers(0, layout._point_count, 500)
        idx_b = rng.integers(0, layout._point_count, 500)
        packed = layout._pair_similarity_np(idx_a, idx_b)
        layout._packed_codes = None
        plain = layout._pair_similarity_np(idx_a, idx_b)
        self.assertTrue(np.array_equal(packed, plain))


class LayoutArraysTests(unittest.TestCase):
    """§5.1.2: массивы позиций и сетки повторяют списки раскладки после обменов и после правки списков."""

    def _assert_consistent(self, layout: Layout) -> None:
        layout._sync_arrays()
        for index, (y, x) in enumerate(layout.positions()):
            self.assertEqual((int(layout._pos_np[index, 0]), int(layout._pos_np[index, 1])), (y, x))
        grid = layout.grid
        for y in range(layout.height):
            for x in range(layout.width):
                cell = grid[y][x]
                self.assertEqual(int(layout._grid_np[y, x]), -1 if cell is None else cell)

    def test_swaps_update_lists_and_arrays_together(self) -> None:
        layout = _layout(points=150, use_gpu=False)
        layout._sync_arrays()
        self.assertFalse(layout._arrays_stale)
        cells = layout._sample_pairs_long_np(60, 4)
        applied = layout._apply_swaps([tuple(int(v) for v in row) for row in cells], mode="long")
        self.assertGreater(applied, 0)
        self.assertFalse(layout._arrays_stale, "обмены меняют массивы вместе со списками")
        pos_before = layout._pos_np.copy()
        self._assert_consistent(layout)
        self.assertTrue(np.array_equal(pos_before, layout._pos_np))

    def test_external_list_edit_marks_arrays_stale(self) -> None:
        layout = _layout(points=50, use_gpu=False)
        layout._sync_arrays()
        (y1, x1), (y2, x2) = layout._positions[0], layout._positions[1]
        self.assertTrue(layout._arrays_stale, "обращение к списку помечает массивы устаревшими")
        layout.grid[y1][x1], layout.grid[y2][x2] = 1, 0
        layout._positions[0], layout._positions[1] = (y2, x2), (y1, x1)
        layout._pos_y[0], layout._pos_y[1] = y2, y1
        layout._pos_x[0], layout._pos_x[1] = x2, x1
        self._assert_consistent(layout)


class SparseEnergyTests(unittest.TestCase):
    """§5.5.1, §5.5.2, §5.7: энергии по ненулевым sim_λ побитово равны исходным проходам движка."""

    def _gpu_layout(self, **kwargs) -> Layout:
        layout = _layout(**kwargs)
        if layout._torch_engine is None:
            self.skipTest("torch с GPU недоступен")
        return layout

    def _sparse(self, layout: Layout, **kwargs) -> SparseLongEnergy:
        rows = SimilarityRows.build(layout._torch_engine, lambda_threshold=layout._lambda, max_edges=10**8)
        self.assertIsNotNone(rows)
        params = dict(buffer_bytes=32 * 1024 * 1024, entry_budget=200_000, screen_min_chunks=1)
        params.update(kwargs)
        return SparseLongEnergy(layout._torch_engine, rows, **params)

    def test_rows_hold_exactly_nonzero_sim_lambda_and_follow_lambda(self) -> None:
        import torch

        layout = self._gpu_layout(points=300)
        engine = layout._torch_engine
        rows = SimilarityRows.build(engine, lambda_threshold=0.6, max_edges=10**8)
        all_points = torch.arange(engine.point_count, device=engine.torch_device)

        def dense_rows(threshold: float):
            sim = engine.similarity_rows(all_points, lambda_threshold=threshold, eta=None).clone()
            sim.fill_diagonal_(0.0)
            return sim.cpu().numpy()

        def stored(rows_object: SimilarityRows):
            dense = np.zeros((engine.point_count, engine.point_count), dtype=np.float32)
            ptr = rows_object.row_ptr_np
            col = rows_object.col.cpu().numpy()
            val = rows_object.val.cpu().numpy()
            for point in range(engine.point_count):
                dense[point, col[ptr[point] : ptr[point + 1]]] = val[ptr[point] : ptr[point + 1]]
            return dense

        self.assertTrue(np.array_equal(stored(rows), dense_rows(0.6)))
        self.assertGreater(rows.edges, 0)
        self.assertTrue(rows.prune(0.75))
        self.assertTrue(np.array_equal(stored(rows), dense_rows(0.75)), "прореживание = строки при новом пороге")
        self.assertFalse(rows.prune(0.7), "при снижении λ строки строятся заново")
        self.assertIsNone(SimilarityRows.build(engine, lambda_threshold=0.6, max_edges=10))

    def test_long_energies_equal_dense_pass_bitwise(self) -> None:
        for points, chunk_elements in ((400, 4_000_000), (400, 400 * 37), (901, 901 * 16)):
            layout = self._gpu_layout(points=points)
            engine = layout._torch_engine
            engine.CHUNK_ELEMENTS = chunk_elements  # несколько порций плотного прохода, последняя — неполная
            sparse = self._sparse(layout)
            for radius, euclidean in ((layout.width // 2, True), (2, True), (5, False)):
                idx_a, idx_b, pos_a, pos_b = _pairs(layout, 230, radius)
                layout._ensure_torch_positions()
                dense_c, dense_s = engine.pair_energies(
                    idx_a, idx_b, pos_a, pos_b, neighbourhood=False, local_radius=None, kernel="inverse",
                    euclidean=euclidean, distance_eps=layout._distance_eps, kernel_width=1.0,
                    lambda_threshold=layout._lambda, eta=None,
                )
                exact_c, exact_s = sparse.pair_energies(idx_a, idx_b, pos_a, pos_b, euclidean=euclidean, signs_only=False)
                self.assertTrue(np.array_equal(exact_c, dense_c) and np.array_equal(exact_s, dense_s),
                                f"points={points} chunk={chunk_elements} radius={radius}")
                self.assertGreater(float(np.abs(dense_c).max()), 0.0)
                signs_c, signs_s = sparse.pair_energies(idx_a, idx_b, pos_a, pos_b, euclidean=euclidean, signs_only=True)
                self.assertTrue(np.array_equal(np.sign(signs_c - signs_s), np.sign(dense_c - dense_s)))
                # несколько шагов, чтобы сверить и частично упорядоченную раскладку
                for step in range(5):
                    layout.step(pairs_per_step=120, pair_radius=radius, mode="long", local_radius=None, step_index=step)

    def test_neighbourhood_and_point_energies_equal_window_pass_bitwise(self) -> None:
        for chunk_elements in (16_000_000, 700 * 23):
            layout = self._gpu_layout(points=700)
            engine = layout._torch_engine
            engine.LOCAL_CHUNK_ELEMENTS = chunk_elements
            sparse = self._sparse(layout)
            for step in range(30):
                layout.step(pairs_per_step=150, pair_radius=6, mode="long", local_radius=None, step_index=step)
            layout._ensure_torch_positions()
            engine.CHUNK_ELEMENTS = 700 * 37  # несколько порций и у прохода по всему V
            # Радиус 14 даёт окно 31 × 31 ≥ |V|: исходный проход идёт по всем точкам, а не по окну ячеек.
            cases = (("inverse", False, 3), ("inverse", True, 2), ("gauss", False, 3), ("inverse", False, 14))
            for kernel, euclidean, radius in cases:
                idx_a, idx_b, pos_a, pos_b = _pairs(layout, 180, 2)
                window_c, window_s = engine.pair_energies(
                    idx_a, idx_b, pos_a, pos_b, neighbourhood=True, local_radius=radius, kernel=kernel,
                    euclidean=euclidean, distance_eps=layout._distance_eps, kernel_width=1.0,
                    lambda_threshold=layout._lambda, eta=None,
                )
                common = dict(local_radius=radius, kernel=kernel, euclidean=euclidean,
                              distance_eps=layout._distance_eps, kernel_width=1.0)
                exact = sparse.neighbourhood_energies(idx_a, idx_b, pos_a, pos_b, signs_only=False, **common)
                self.assertIsNotNone(exact)
                self.assertTrue(np.array_equal(exact[0], window_c) and np.array_equal(exact[1], window_s),
                                f"{kernel} euclidean={euclidean} chunk={chunk_elements}")
                self.assertGreater(float(np.abs(window_c).max()), 0.0)
                signs = sparse.neighbourhood_energies(idx_a, idx_b, pos_a, pos_b, signs_only=True, **common)
                self.assertTrue(np.array_equal(np.sign(signs[0] - signs[1]), np.sign(window_c - window_s)))
            # энергии точек §5.7: вторая ячейка пуста, нужна только φ_c
            window = engine.point_energies(3, lambda_threshold=layout._lambda, eta=None, distance_eps=layout._distance_eps)
            layout._sync_arrays()
            indices = np.arange(layout._point_count)
            points = sparse.neighbourhood_energies(
                indices, np.full(layout._point_count, -1), layout._pos_np, layout._pos_np,
                local_radius=3.0, kernel="inverse", euclidean=True, distance_eps=layout._distance_eps,
                kernel_width=1.0, signs_only=False, need_swapped=False,
            )
            self.assertIsNotNone(points)
            self.assertTrue(np.array_equal(points[0], window))

    def test_window_bit_counts_and_chunked_sums_equal_reference(self) -> None:
        import torch

        layout = self._gpu_layout(points=500)
        engine = layout._torch_engine
        device = engine.torch_device
        generator = torch.Generator(device="cpu").manual_seed(0)
        points = torch.randint(0, engine.point_count, (90,), generator=generator).to(device)
        window = torch.randint(0, engine.point_count, (90, 25), generator=generator).to(device)
        full = engine._window_common_bits(points, window)
        engine.WINDOW_FULL_PRODUCT_RATIO = 0  # подсчёт общих бит по упакованным словам кода
        packed = engine._window_common_bits(points, window)
        self.assertTrue(torch.equal(full, packed))
        terms = torch.rand((233, 49), generator=generator).to(device)
        for chunk in (16, 7, 233, 500):
            want = torch.cat([terms[start : start + chunk].clone().sum(dim=1) for start in range(0, 233, chunk)])
            self.assertTrue(torch.equal(engine._sum_rows_by_chunks(terms, chunk), want), f"chunk={chunk}")

    def test_layout_is_identical_with_and_without_sparse_energies(self) -> None:
        """§5.5–§5.7, §7.1.7: раскладка с расчётом по строкам сходства совпадает с исходной шаг в шаг."""

        def run(sparse_long: bool) -> tuple[list[tuple[int, int]], list[float]]:
            layout = self._gpu_layout(points=600, sparse_long=sparse_long, sparse_verify_every=7)
            adaptive = AdaptiveLayoutConfig(
                start_radius=layout.width // 2, end_radius=1, swap_ratio_trigger=0.7, lambda_step=0.05,
                lambda_max=0.75, radius_schedule="swaps", radius_factor=0.5, swap_window=3,
            )
            layout.run(steps=120, pairs_per_step=60, pair_radius=layout.width // 2, mode="long",
                       min_swap_ratio=0.0, min_swap_window=None, log_every=None, log_visuals=False,
                       energy_radius=3, energy_every=20, adaptive_params=adaptive)
            layout.run(steps=40, pairs_per_step=60, pair_radius=3, mode="short", local_radius=3,
                       min_swap_ratio=0.0, min_swap_window=None, log_every=None, log_visuals=False,
                       short_optim=ShortLayoutOptimConfig(energy_radius=3, energy_recalc_every=10, partitions=4,
                                                          similarity_cutoff=0.6))
            self.assertEqual(layout._sparse_long is not None, sparse_long)
            return layout.positions(), layout._compute_point_energies(3)

        sparse_positions, sparse_energies = run(True)
        dense_positions, dense_energies = run(False)
        self.assertEqual(sparse_positions, dense_positions)
        self.assertEqual(sparse_energies, dense_energies)


if __name__ == "__main__":
    unittest.main()
