"""Большие раскладки в режиме статьи (жёсткий порог §7.1.7): затравка по уровням, ближняя фаза на устройстве.

Ядро окна WindowEnergyKernel даёт те же φ_c, φ_s (§5.5.2) и энергии точек (§5.7), что исходный проход
движка, с точностью сложения float32. Шаг ближней фазы на устройстве (DeviceShortStep) держит расстановку
согласованной и повышает энергии точек. Поиск родителя по опорным точкам находит почти тех же родителей,
что перебор. Настройка включает режим только выше границы числа точек.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("RERUN", "off")

import numpy as np  # noqa: E402

import main  # noqa: E402
from damp.layout.damp_layout import AdaptiveLayoutConfig, Layout, ShortLayoutOptimConfig  # noqa: E402
from damp.layout.seeded import SeededPlacement  # noqa: E402
from damp.layout.window_energy_kernel import WindowEnergyKernel  # noqa: E402
from damp.logging import LOGGER  # noqa: E402
from tests.test_layout_large import _builder, _records  # noqa: E402

LOGGER.configure_intervals({}, default_interval=10**9)

_ARTICLE_PARAMS = dict(
    similarity="cosine", lambda_threshold=0.6, eta=None, use_gpu=True, seed=5,
    long_distance="euclidean", short_distance="squared", long_kernel="distance",
)


def _article_layout(points: int, seed: int, **options) -> Layout:
    return Layout(_builder(_records(points, seed)).code_table(), empty_ratio=0.4, **_ARTICLE_PARAMS, **options)


def _device_tables(layout: Layout):
    """Расстановка раскладки тензорами устройства: ячейка каждой точки и точка каждой ячейки (N — пусто)."""
    engine = layout._torch_engine
    torch = engine.torch_mod
    device = engine.torch_device
    cells = layout.cells_array()
    linear = cells[:, 0] * layout.width + cells[:, 1]
    grid_points = np.full(layout.height * layout.width + 1, layout.point_count, dtype=np.int32)
    grid_points[linear] = np.arange(layout.point_count, dtype=np.int32)
    return torch.as_tensor(linear, device=device), torch.as_tensor(grid_points, device=device)


class WindowEnergyKernelTests(unittest.TestCase):
    """§5.5.2, §5.7, §5.12: суммы ядра по ячейкам окна равны исходному проходу движка."""

    def _layout(self) -> Layout:
        layout = _article_layout(3000, 61)
        engine = layout._torch_engine
        if engine is None or not WindowEnergyKernel.supported(engine):
            self.skipTest("ядро окна недоступно (нужен torch с CUDA)")
        layout._ensure_torch_positions()
        return layout

    @staticmethod
    def _pairs(layout: Layout, radius: int, count: int, seed: int):
        rng = np.random.default_rng(seed)
        cells = layout.cells_array()
        grid = np.full((layout.height, layout.width), -1, dtype=np.int64)
        grid[cells[:, 0], cells[:, 1]] = np.arange(layout.point_count)
        idx_a = rng.integers(0, layout.point_count, count)
        offset = rng.integers(-radius, radius + 1, size=(count, 2))
        pos_a = cells[idx_a]
        pos_b = pos_a + offset
        distance_sq = (offset * offset).sum(axis=1)
        keep = (
            (distance_sq > 0) & (distance_sq <= radius * radius)
            & (pos_b[:, 0] >= 0) & (pos_b[:, 0] < layout.height) & (pos_b[:, 1] >= 0) & (pos_b[:, 1] < layout.width)
        )
        idx_a, pos_a, pos_b = idx_a[keep], pos_a[keep], pos_b[keep]
        return idx_a, grid[pos_b[:, 0], pos_b[:, 1]], pos_a, pos_b

    def test_pair_energies_match_reference(self) -> None:
        layout = self._layout()
        engine = layout._torch_engine
        torch = engine.torch_mod
        device = engine.torch_device
        kernel = WindowEnergyKernel(engine)
        _, grid_points = _device_tables(layout)
        kernel.load(grid_points)
        count = layout.point_count
        for radius, local_radius, euclidean in ((3, 2, False), (7, 9, False), (10, 4, True), (1, 5, False)):
            idx_a, idx_b, pos_a, pos_b = self._pairs(layout, radius, 1500, radius)
            self.assertTrue(bool((idx_b < 0).any()), "нужны пары с пустой второй ячейкой")
            reach = engine.window_reach(pos_a, pos_b, local_radius)
            expected_c, expected_s = engine._pair_energies_reference(
                idx_a, idx_b, pos_a, pos_b, neighbourhood=True, local_radius=float(local_radius), kernel="inverse",
                euclidean=euclidean, distance_eps=1e-6, kernel_width=1.0, lambda_threshold=0.6, eta=None,
                window_reach=reach,
            )
            energy_c, energy_s = kernel.pair_energies(
                torch.as_tensor(idx_a, device=device),
                torch.as_tensor(np.where(idx_b < 0, count, idx_b), device=device),
                torch.as_tensor(pos_a[:, 0] * layout.width + pos_a[:, 1], device=device),
                torch.as_tensor(pos_b[:, 0] * layout.width + pos_b[:, 1], device=device),
                reach=reach, local_radius=float(local_radius), euclidean=euclidean,
                distance_eps=1e-6, lambda_threshold=0.6,
            )
            energy_c = energy_c.cpu().numpy()
            energy_s = energy_s.cpu().numpy()
            self.assertGreater(float(expected_c.max()), 0.0)
            self.assertTrue(kernel.verify(energy_c, energy_s, expected_c, expected_s, kind="pairs"), (radius, euclidean))
            # Знак φ_s − φ_c (решение об обмене, §5.6) тот же везде, где разность не в пределах ошибки сложения.
            delta = expected_s.astype(np.float64) - expected_c
            clear = np.abs(delta) > 1e-4 * (np.abs(expected_s) + np.abs(expected_c))
            self.assertTrue(np.array_equal(np.sign(energy_s - energy_c)[clear], np.sign(delta)[clear]))

    def test_point_energies_and_pair_similarity_match_reference(self) -> None:
        layout = self._layout()
        engine = layout._torch_engine
        torch = engine.torch_mod
        kernel = WindowEnergyKernel(engine)
        cell_of_point, grid_points = _device_tables(layout)
        kernel.load(grid_points)
        for radius in (3, 6):
            expected = engine.point_energies(radius, lambda_threshold=0.6, eta=None, distance_eps=1e-6)
            actual = kernel.point_energies(cell_of_point, float(radius), distance_eps=1e-6, lambda_threshold=0.6)
            self.assertTrue(np.allclose(actual.cpu().numpy(), expected, rtol=2e-4, atol=1e-5), radius)
        idx_a, idx_b, _, _ = self._pairs(layout, 4, 800, 9)
        device = engine.torch_device
        similarity = kernel.pair_similarity(
            torch.as_tensor(idx_a, device=device),
            torch.as_tensor(np.where(idx_b < 0, layout.point_count, idx_b), device=device),
        ).cpu().numpy()
        present = idx_b >= 0
        expected = layout._pair_similarity_np(idx_a[present], idx_b[present])
        self.assertTrue(np.allclose(similarity[present], expected, atol=1e-6))
        self.assertEqual(float(np.abs(similarity[~present]).max()), 0.0)

    def test_update_follows_swaps(self) -> None:
        """Коды по ячейкам после update — те же, что после полной загрузки новой расстановки."""
        layout = self._layout()
        engine = layout._torch_engine
        torch = engine.torch_mod
        device = engine.torch_device
        kernel = WindowEnergyKernel(engine)
        cell_of_point, grid_points = _device_tables(layout)
        kernel.load(grid_points)
        count = layout.point_count
        # Обмен первых 200 точек с последними 200 и перенос ещё 50 точек в пустые ячейки.
        cells = cell_of_point.clone()
        first = torch.arange(0, 200, device=device)
        second = torch.arange(count - 200, count, device=device)
        cells[first], cells[second] = cell_of_point[second], cell_of_point[first]
        empty = torch.nonzero(grid_points[:-1] == count).view(-1)[:50]
        movers = torch.arange(300, 350, device=device)
        freed = cells[movers].clone()
        cells[movers] = empty
        changed = torch.cat([cells[first], cells[second], empty, freed])
        occupants = torch.cat([first, second, movers, torch.full_like(freed, count)])
        kernel.update(changed, occupants)
        updated = kernel.point_energies(cells, 4.0, distance_eps=1e-6, lambda_threshold=0.6).cpu().numpy()
        fresh = WindowEnergyKernel(engine)
        new_grid = torch.full_like(grid_points, count)
        new_grid[cells] = torch.arange(count, dtype=new_grid.dtype, device=device)
        fresh.load(new_grid)
        expected = fresh.point_energies(cells, 4.0, distance_eps=1e-6, lambda_threshold=0.6).cpu().numpy()
        self.assertTrue(np.array_equal(updated, expected))


class DeviceShortStepTests(unittest.TestCase):
    """§5.3, §5.5.2, §5.6, §5.9.1, §5.9.3, §5.10: шаг ближней фазы на устройстве."""

    def _layout(self, **options) -> Layout:
        layout = _article_layout(
            2500, 62, device_short_step=True, neighbourhood_radius_min=3, enforce_pair_radius=True, **options
        )
        engine = layout._torch_engine
        if engine is None or not WindowEnergyKernel.supported(engine):
            self.skipTest("ядро окна недоступно (нужен torch с CUDA)")
        return layout

    @staticmethod
    def _run_short(layout: Layout, steps: int = 150) -> None:
        adaptive = AdaptiveLayoutConfig(
            start_radius=6, end_radius=1, swap_ratio_trigger=0.01, lambda_step=None,
            radius_schedule="swaps", radius_factor=0.5, swap_window=5,
        )
        config = ShortLayoutOptimConfig(
            energy_radius=4, energy_recalc_every=25, energy_eps=1e-6, use_weighted_first_point=True,
            similarity_cutoff=0.6, partitions=4,
        )
        layout.run(
            steps=steps, pairs_per_step=250, pair_radius=6, mode="short", local_radius=8,
            min_swap_ratio=0.0, adaptive_params=adaptive, short_optim=config,
        )

    def test_steps_keep_layout_consistent_and_raise_energy(self) -> None:
        layout = self._layout()
        before = sum(layout._compute_point_energies(3))
        self._run_short(layout)
        # Сверка энергий первого шага с исходным проходом прошла: шаг на устройстве не отключён.
        self.assertTrue(layout._device_short_enabled)
        self.assertIsNotNone(layout._device_short)
        cells = layout.cells_array()
        self.assertEqual(len({(int(y), int(x)) for y, x in cells}), layout.point_count)
        grid = layout.grid
        for index, (y, x) in enumerate(layout.positions()):
            self.assertEqual(grid[y][x], index)
        self.assertEqual(sum(1 for row in grid for cell in row if cell is not None), layout.point_count)
        after = sum(layout._compute_point_energies(3))
        self.assertGreater(after, before * 1.5)
        # Энергии точек ядра окна — те же, что у исходного прохода движка по той же расстановке.
        host = _article_layout(2500, 62)
        host.place_cells(cells)
        expected = np.asarray(host._compute_point_energies(3))
        self.assertTrue(np.allclose(np.asarray(layout._compute_point_energies(3)), expected, rtol=2e-4, atol=1e-5))

    def test_steps_are_deterministic(self) -> None:
        first = self._layout()
        self._run_short(first, steps=60)
        second = self._layout()
        self._run_short(second, steps=60)
        self.assertEqual(first.positions(), second.positions())

    def test_neighbourhood_radius_follows_pair(self) -> None:
        """§5.5.2: r ≥ ‖p1 − p2‖ — в больших раскладках берётся наименьший радиус, не ниже заданного."""
        layout = self._layout()
        self.assertEqual(layout._short_neighbourhood_radius(31, 20), 3)
        self.assertEqual(layout._short_neighbourhood_radius(2, 20), 2)
        plain = _article_layout(300, 63)
        self.assertEqual(plain._short_neighbourhood_radius(31, 20), 31)


class PivotParentSearchTests(unittest.TestCase):
    """Поиск родителя по спискам опорных точек: те же или почти такие же похожие родители, что у перебора."""

    class _SmallPivots(SeededPlacement):
        PIVOT_MIN_PARENTS = 100
        PIVOTS = 96
        PROBES = 12
        LIST_MAX = 512

    def test_pivot_search_finds_similar_parents(self) -> None:
        table = _builder(_records(6000, 64)).code_table()
        try:
            import torch

            if not torch.cuda.is_available():
                self.skipTest("нужен torch с CUDA")
        except ImportError:
            self.skipTest("нужен torch с CUDA")
        placement = self._SmallPivots(table, seed_count=1500, seed=3, similarity="cosine")
        parents = placement.seed_indices
        matched = placement._match_device(parents)
        self.assertIsNotNone(matched)
        found = matched[0][:, 0]
        exact = placement._match_host(parents)[0][:, 0]
        words = table.words64()
        ones = table.ones.astype(np.float64)

        def similarity(choice: "np.ndarray") -> "np.ndarray":
            common = np.bitwise_count(words & words[parents[choice]]).sum(axis=1)
            return common / np.sqrt(ones * ones[parents[choice]])

        best = similarity(exact)
        got = similarity(found)
        self.assertTrue(bool((got <= best + 1e-9).all()))
        # Родитель — среди кандидатов из списков ближайших опорных точек: почти всегда тот же по сходству.
        self.assertGreater(float((got >= best - 1e-9).mean()), 0.9)
        self.assertGreater(float(got.mean()), 0.98 * float(best.mean()))


class ArticlePlacementTests(unittest.TestCase):
    """Расстановка от затравки в режиме статьи: различные коды, место в блоке по похожим родителям."""

    def _placement(self):
        records = _records(1200, 66)
        # Повторы кодов: вторая половина точек повторяет коды первой.
        records = records + [(label, hue, code) for label, hue, code in records[:600]]
        table = _builder(records).code_table()
        placement = SeededPlacement(table, seed_count=200, seed=7, similarity="cosine")
        seed_layout = Layout(placement.seed_table(), empty_ratio=0.4, **_ARTICLE_PARAMS)
        return table, placement, seed_layout.cells_array(), (seed_layout.height, seed_layout.width)

    def test_distinct_codes_give_same_parents(self) -> None:
        """Родители, найденные по различным кодам, — те же, что при поиске для каждой точки."""
        table, placement, seed_cells, seed_shape = self._placement()
        representatives, code_of_point = placement._distinct_codes()
        self.assertLess(representatives.size, table.count)
        words = table.words64()
        self.assertTrue(bool((words == words[representatives[code_of_point]]).all()))
        self.assertEqual(np.unique(words, axis=0).shape[0], representatives.size)
        parent_seeds, nearest, similarity = placement.similar_parents(seed_cells, seed_shape, top=4)
        parent_points = placement.seed_indices[parent_seeds]
        every_point, every_similarity = placement._match_host(parent_points, 4)
        ordinary = np.ones(table.count, dtype=bool)
        ordinary[parent_points] = False
        self.assertTrue(np.allclose(similarity[ordinary], every_similarity[ordinary], atol=1e-6))
        self.assertTrue(np.array_equal(nearest[parent_points, 0], np.arange(parent_points.size)))

    def test_neighbour_targets_keep_points_near_parent(self) -> None:
        table, placement, seed_cells, seed_shape = self._placement()
        size = 50
        cells = placement.cells(seed_cells, seed_shape, (size, size), neighbour_threshold=0.6)
        self.assertEqual(cells.shape, (table.count, 2))
        self.assertEqual(len({(int(y), int(x)) for y, x in cells}), table.count)
        self.assertTrue(bool(((cells >= 0) & (cells < size)).all()))
        reference = SeededPlacement(table, seed_count=200, seed=7, similarity="cosine")
        parent_seeds, parent = reference.parents(seed_cells, seed_shape)
        scale = size / seed_shape[0]
        centre = (seed_cells[parent_seeds[parent]] + 0.5) * scale

        def parent_distance(placed: "np.ndarray") -> float:
            return float(np.median(np.hypot(placed[:, 0] + 0.5 - centre[:, 0], placed[:, 1] + 0.5 - centre[:, 1])))

        # Точка сдвинута к другим похожим родителям, но остаётся в окрестности блока своего родителя: не дальше
        # двух расстояний, которые даёт случайное место в блоке (затравка здесь не разложена — сдвиги наибольшие).
        plain = SeededPlacement(table, seed_count=200, seed=7, similarity="cosine").cells(
            seed_cells, seed_shape, (size, size)
        )
        self.assertLess(parent_distance(cells), 2.0 * parent_distance(plain) + scale)

    def test_match_cells_on_device_is_bijection_and_keeps_order(self) -> None:
        rng = np.random.default_rng(8)
        count = 5000
        target_y = rng.random(count) * 90.0
        target_x = rng.random(count) * 90.0
        cells = rng.permutation(100 * 100)[:count]
        cell_y, cell_x = cells // 100, cells % 100
        match = SeededPlacement.match_cells(target_y, target_x, cell_y, cell_x, on_device=True)
        self.assertEqual(sorted(match.tolist()), list(range(count)))
        # Порядок сохраняется: точки с меньшей целевой координатой получают ячейки с меньшей координатой.
        self.assertGreater(float(np.corrcoef(target_y, cell_y[match])[0, 1]), 0.95)
        self.assertGreater(float(np.corrcoef(target_x, cell_x[match])[0, 1]), 0.95)
        host = SeededPlacement.match_cells(target_y, target_x, cell_y, cell_x)
        self.assertEqual(sorted(host.tolist()), list(range(count)))


class BaseFileTests(unittest.TestCase):
    """Файл раскладки большого набора до раскладки — только заголовок; итоговый файл тот же."""

    def test_header_only_base_gives_same_layout_file(self) -> None:
        builder = _builder(_records(300, 67))
        layout = Layout(builder.code_table(), empty_ratio=0.4, **_ARTICLE_PARAMS)
        with tempfile.TemporaryDirectory() as directory:
            full = os.path.join(directory, "full.json")
            header = os.path.join(directory, "header.json")
            builder.save_base(full)
            builder.save_base(header, max_points_bytes=1)
            self.assertLess(os.path.getsize(header), os.path.getsize(full) // 10)
            # Набор, чей список точек укладывается в предел, пишется целиком — как без предела.
            whole = os.path.join(directory, "whole.json")
            builder.save_base(whole, max_points_bytes=10**9)
            with open(full, "rb") as left, open(whole, "rb") as right:
                self.assertEqual(left.read(), right.read())
            layout.save_json(full)
            layout.save_json(header)
            with open(full, "rb") as left, open(header, "rb") as right:
                self.assertEqual(left.read(), right.read())


class EncodedCodesFileTests(unittest.TestCase):
    """Файл кодов набора: те же углопозиции и коды, что дало кодирование; другой кодер — другой файл."""

    class _Digits:
        """Набор «цифр»: изображения 28 × 28 с отрезками (как тензоры MNIST 1 × 28 × 28) и метки."""

        def __init__(self, count: int) -> None:
            import torch

            rng = np.random.default_rng(11)
            self._items = []
            for index in range(count):
                image = np.zeros((28, 28), dtype=np.float32)
                row = int(rng.integers(6, 22))
                column = int(rng.integers(6, 22))
                image[row, 4:24] = 1.0
                image[4:24, column] = 1.0
                self._items.append((torch.as_tensor(image).unsqueeze(0), index % 10))

        def __iter__(self):
            return iter(self._items)

    def test_cached_codes_equal_encoded_codes(self) -> None:
        encoder = main._build_encoder()
        extractor = main._build_extractor()
        digits = self._Digits(40)
        with tempfile.TemporaryDirectory() as directory, mock.patch.multiple(
            main, ENCODING_CODES_CACHE_MIN_DIGITS=10, ENCODING_PARALLEL_MIN_DIGITS=10**9
        ):
            store = main.EncodedCodesFile(directory, encoder=encoder, split="train", label=None)
            first = main.StimulusCodeCache(digits, encoder, extractor, label=None, store=store)
            encoded = first.take(30)
            self.assertTrue(os.path.exists(store.path))
            self.assertGreater(sum(item.count for item in encoded), 0)
            # Повторный запуск: коды берутся из файла, набор не кодируется.
            second = main.StimulusCodeCache(digits, encoder, extractor, label=None, store=store)
            with mock.patch.object(main.StimulusCodeCache, "encode_digit", side_effect=AssertionError("encoded")):
                cached = second.take(30)
            self.assertEqual(len(cached), len(encoded))
            for left, right in zip(encoded, cached):
                self.assertEqual(left.label, right.label)
                self.assertEqual(left.angles, right.angles)
                self.assertTrue(np.array_equal(left.packed, right.packed))
                self.assertEqual(left.code_length, right.code_length)
            # Следующие цифры набора кодируются после взятых из файла — те же, что без файла.
            more = second.take(40)
            plain = main.StimulusCodeCache(digits, encoder, extractor, label=None).take(40)
            self.assertEqual(len(more), 40)
            for left, right in zip(plain, more):
                self.assertEqual(left.label, right.label)
                self.assertEqual(left.angles, right.angles)
                self.assertTrue(np.array_equal(left.packed, right.packed))
            # Файл дополнен до 40 цифр; меньший запрос берёт начало файла.
            third = main.StimulusCodeCache(digits, encoder, extractor, label=None, store=store)
            with mock.patch.object(main.StimulusCodeCache, "encode_digit", side_effect=AssertionError("encoded")):
                self.assertEqual(len(third.take(35)), 35)
            # Другая константа кодирования — другой отпечаток: прежний файл не подходит.
            with mock.patch.object(main, "MNIST_GRAD_THRESHOLD", main.MNIST_GRAD_THRESHOLD * 2.0):
                other = main.EncodedCodesFile(directory, encoder=encoder, split="train", label=None)
            self.assertNotEqual(other.path, store.path)
            self.assertIsNone(other.load(10))
            # Малый набор файл не читает и не пишет.
            small_store = main.EncodedCodesFile(directory, encoder=encoder, split="test", label=None)
            main.StimulusCodeCache(digits, encoder, extractor, label=None, store=small_store).take(5)
            self.assertFalse(os.path.exists(small_store.path))


class ArticleLargeTunerTests(unittest.TestCase):
    """§5.9.4, §5.10, §7.1.7: настройка больших раскладок режима статьи и уровни затравки."""

    _ARTICLE = dict(LAYOUT_SMOOTH_PINWHEELS=False, LAYOUT_PAIRS_POINT_FRACTION=0.1)

    def test_article_large_layout_parameters(self) -> None:
        with mock.patch.multiple(main, **self._ARTICLE):
            tuner = main.LayoutParameterTuner()
            small = tuner.tune(main.LAYOUT_LONG_FULL_SPACE_MAX_POINTS)
            large = tuner.tune(1_000_000)
        self.assertIsNone(small.seed_points)
        self.assertFalse(small.device_short_step)
        self.assertIsNone(small.neighbourhood_radius_min)
        self.assertEqual(large.seed_points, main.LAYOUT_SEED_POINTS)
        self.assertTrue(large.device_short_step)
        self.assertFalse(large.device_step)
        self.assertTrue(large.enforce_pair_radius)
        self.assertIsNone(large.long_subset_size)
        self.assertIsNone(large.neighbourhood_sample_side)
        self.assertEqual(large.neighbourhood_radius_min, main.LAYOUT_TUNE_LARGE_SHORT_NEIGHBOURHOOD_MIN)
        # §5.10: пар за шаг — не больше доли точек.
        self.assertEqual(large.short_pairs_per_step, 100_000)
        self.assertLessEqual(large.short_local_radius, main.LAYOUT_TUNE_LARGE_SHORT_RADIUS_MAX)

    def test_smooth_large_layout_parameters_keep_device_long_step(self) -> None:
        with mock.patch.multiple(main, LAYOUT_SMOOTH_PINWHEELS=True, LAYOUT_PAIRS_POINT_FRACTION=None):
            large = main.LayoutParameterTuner().tune(1_000_000)
        self.assertFalse(large.device_short_step)
        self.assertIsNone(large.neighbourhood_radius_min)
        self.assertEqual(large.device_step, bool(main.LAYOUT_DEVICE_STEP))

    def test_seed_levels(self) -> None:
        seed = 30000
        with mock.patch.multiple(main, LAYOUT_SMOOTH_PINWHEELS=False, LAYOUT_SEED_LEVEL_RATIO_MAX=32):
            # До 32 точек на точку затравки — один уровень.
            self.assertEqual(main._seed_level_points(882_607, seed), seed)
            self.assertEqual(main._seed_level_points(seed * 32, seed), seed)
        with mock.patch.multiple(main, LAYOUT_SMOOTH_PINWHEELS=False, LAYOUT_SEED_LEVEL_RATIO_MAX=16):
            # 29 точек на точку затравки при пределе 16 — два уровня по √29.
            middle = main._seed_level_points(882_607, seed)
            self.assertAlmostEqual(882_607 / middle, middle / seed, delta=0.01 * middle / seed)
        with mock.patch.multiple(main, LAYOUT_SMOOTH_PINWHEELS=False, LAYOUT_SEED_LEVEL_RATIO_MAX=32):
            # 5.3 млн точек: два уровня с равным отношением √177 ≈ 13.3.
            middle = main._seed_level_points(5_302_618, seed)
            self.assertAlmostEqual(5_302_618 / middle, middle / seed, delta=0.01 * middle / seed)
            self.assertEqual(main._seed_level_points(middle, seed), seed)
            self.assertEqual(main._seed_level_points(seed - 1, seed), seed)
        with mock.patch.multiple(main, LAYOUT_SMOOTH_PINWHEELS=True):
            self.assertEqual(main._seed_level_points(5_302_618, seed), seed)

    def test_seeded_article_pipeline(self) -> None:
        """Раскладка от затравки в режиме статьи: два уровня, точки расставлены, соседи — из одной группы."""
        table = _builder(_records(9000, 65)).code_table()
        article = dict(
            LAYOUT_SMOOTH_PINWHEELS=False, LAYOUT_PAIRS_POINT_FRACTION=0.1, LAYOUT_ETA=None,
            LAYOUT_LONG_KERNEL="distance", LAYOUT_ADAPTIVE_RADIUS_SCHEDULE="swaps",
            LAYOUT_SHORT_ADAPTIVE_RADIUS_SCHEDULE="swaps", LAYOUT_ADAPTIVE_SWAP_TRIGGER=0.01,
            LAYOUT_ADAPTIVE_LAMBDA_STEP=0.05, LAYOUT_ANNEAL_TEMPERATURE=0.0,
            LAYOUT_ADAPTIVE_RADIUS_START_FACTOR=0.5, LAYOUT_LAMBDA_THRESHOLD=0.6,
            LAYOUT_LONG_FULL_SPACE_MAX_POINTS=1500, LAYOUT_SEED_POINTS=600, LAYOUT_SEED_LEVEL_RATIO_MAX=4,
        )
        with mock.patch.multiple(main, **article):
            import dataclasses

            original_tune = main.LayoutParameterTuner.tune

            def short_tune(self, total_points):  # noqa: ANN001, ANN202
                params = original_tune(self, total_points)
                return dataclasses.replace(params, long_steps=400, short_steps=150)

            with mock.patch.object(main.LayoutParameterTuner, "tune", short_tune):
                params = main.LayoutParameterTuner().tune(table.count)
                self.assertEqual(params.seed_points, 600)
                # 9000 / 600 = 15 при пределе 4 точки на родителя: уровни 600 → 2324 → 9000.
                self.assertGreater(main._seed_level_points(table.count, 600), 600)
                layout = main._run_layout(table, params, count=0, total_codes=table.count, use_cache=False)
        cells = layout.cells_array()
        self.assertEqual(len({(int(y), int(x)) for y, x in cells}), table.count)
        grid = np.full((layout.height, layout.width), -1, dtype=np.int64)
        grid[cells[:, 0], cells[:, 1]] = np.arange(table.count)
        left, right = grid[:, :-1], grid[:, 1:]
        both = (left >= 0) & (right >= 0)
        groups = table.hues
        self.assertGreater(float((groups[left[both]] == groups[right[both]]).mean()), 0.6)
        # Порог λ большой раскладки продолжает расписание затравки (§7.1.7), а не начинается заново.
        self.assertGreaterEqual(layout.lambda_threshold, 0.6)


if __name__ == "__main__":
    unittest.main()
