"""Слитый расчёт энергий и шаг раскладки на массивах не меняют результат раскладки.

φ_c, φ_s слитого ядра (§5.5.1, §5.5.2, §5.7) сверяются с исходным проходом движка побитово — для сигмоиды
и жёсткого порога (§5.4), ядер гаусса, 1/d и d, расчёта по всему V и по окну ячеек, всех источников
сходства (таблица sim_λ, таблица общих бит по точкам и по классам кодов, общие биты на лету, §5.9.5).
Раскладка с отжигом (расширение статьи) и без него совпадает со слитым расчётом и без него шаг в шаг.
"""
from __future__ import annotations

import os
import random
import unittest

os.environ.setdefault("RERUN", "off")

import numpy as np  # noqa: E402

from damp.encoding.bitarray import BitArray  # noqa: E402
from damp.encoding.damp_encoder import EncoderCodeColorizer  # noqa: E402
from damp.layout.damp_layout import AdaptiveLayoutConfig, Layout, ShortLayoutOptimConfig, _UniformDraws  # noqa: E402
from damp.layout.payload import LayoutEncodedCode  # noqa: E402
from damp.logging import LOGGER  # noqa: E402

LOGGER.configure_intervals({}, default_interval=10**9)

_CODE_LENGTH = 96
_GROUPS = 12


def _clustered_codes(points: int, seed: int) -> dict[float, list[LayoutEncodedCode]]:
    """Коды из нескольких групп с повторами: внутри группы сходство высокое, между группами — низкое."""
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


def _layout(points: int = 400, *, seed: int = 5, **kwargs) -> Layout:
    params = dict(
        similarity="cosine", lambda_threshold=0.6, eta=10.0, use_gpu=True, seed=seed,
        long_distance="euclidean", short_distance="squared", long_kernel="gauss", long_kernel_width=1.0,
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


def _bits(values: "np.ndarray") -> "np.ndarray":
    return np.ascontiguousarray(values).view(np.int32)


class FusedEnergyTests(unittest.TestCase):
    """§5.5, §5.7, §5.12: слитое ядро даёт побитово те же φ_c, φ_s, что исходный проход движка."""

    def _engine_layout(self, **kwargs) -> Layout:
        layout = _layout(**kwargs)
        engine = layout._torch_engine
        if engine is None or not engine.fused_active:
            self.skipTest("слитое ядро недоступно (нужен torch с CUDA)")
        return layout

    def _configure(self, layout: Layout, *, source: str) -> None:
        """Источник сходства слитого расчёта: таблица sim_λ, таблица общих бит (по точкам, по классам), на лету."""
        points = layout._point_count
        table_bytes = {
            "similarity": 1 << 30,
            "common": 1 << 30,
            # Таблица по точкам не помещается на байт: строится по классам одинаковых кодов.
            "classes": (points + 1) ** 2 - 1,
            "product": 0,
        }[source]
        layout._torch_engine.configure_fused(
            enabled=True, block_elements=3_000_000, table_max_bytes=table_bytes, table_memory_fraction=1.0,
            verify_every=0, similarity_table=source == "similarity",
        )

    def _assert_fused_equals_reference(self, layout: Layout, label: str) -> None:
        engine = layout._torch_engine
        layout._sync_arrays()
        host = {"positions": layout._pos_np, "grid": layout._grid_np}
        distance_eps = layout._distance_eps
        cases = (
            # (окрестность, ядро, евклидово, радиус окрестности, радиус пар, ширина гаусса)
            (True, "gauss", False, layout.width, layout.width, 1.0),
            (True, "gauss", False, 6, 6, 1.0),
            (True, "gauss", False, 4, 3, 0.7),
            (True, "inverse", False, 3, 3, 1.0),
            (True, "inverse", True, None, 2, 1.0),
            (False, "inverse", True, None, layout.width // 2, 1.0),
            (False, "inverse", False, None, 4, 1.0),
        )
        for eta in (10.0, None):
            for neighbourhood, kernel, euclidean, local_radius, pair_radius, width in cases:
                for count in (330, 37, 1):
                    idx_a, idx_b, pos_a, pos_b = _pairs(layout, count, pair_radius)
                    if idx_a.size == 0:
                        continue
                    arguments = dict(
                        neighbourhood=neighbourhood, local_radius=local_radius, kernel=kernel, euclidean=euclidean,
                        distance_eps=distance_eps, kernel_width=width, lambda_threshold=0.6, eta=eta,
                    )
                    fused = engine._fused_engine().pair_energies(idx_a, idx_b, pos_a, pos_b, **host, **arguments)
                    self.assertIsNotNone(fused, label)
                    engine.update_positions(layout._pos_np[:, 0], layout._pos_np[:, 1])
                    reference = engine._pair_energies_reference(idx_a, idx_b, pos_a, pos_b, **arguments)
                    case = f"{label} eta={eta} nb={neighbourhood} {kernel} euclid={euclidean} r={local_radius} pairs={count}"
                    self.assertTrue(np.array_equal(_bits(fused[0]), _bits(reference[0])), case)
                    self.assertTrue(np.array_equal(_bits(fused[1]), _bits(reference[1])), case)
                    self.assertGreater(float(np.abs(reference[0]).max()), 0.0, case)
        # Энергии точек §5.7: вторая ячейка пуста, φ_s не нужна (в пути по окну ячеек — нули).
        indices = np.arange(layout._point_count)
        absent = np.full(layout._point_count, -1)
        arguments = dict(
            neighbourhood=True, local_radius=3.0, kernel="inverse", euclidean=True, distance_eps=distance_eps,
            kernel_width=1.0, lambda_threshold=0.6, eta=10.0, need_swapped=False,
        )
        fused = engine._fused_engine().pair_energies(indices, absent, layout._pos_np, layout._pos_np, **host, **arguments)
        reference = engine._pair_energies_reference(indices, absent, layout._pos_np, layout._pos_np, **arguments)
        self.assertTrue(np.array_equal(_bits(fused[0]), _bits(reference[0])), f"{label} point energies")
        self.assertTrue(np.array_equal(_bits(fused[1]), _bits(reference[1])), f"{label} point energies swapped")

    def test_fused_energies_equal_reference_pass_bitwise_for_every_similarity_source(self) -> None:
        for source in ("similarity", "common", "classes", "product"):
            layout = self._engine_layout(points=700)
            self._configure(layout, source=source)
            for step in range(12):
                layout.step(pairs_per_step=150, pair_radius=6, mode="long", local_radius=None, step_index=step)
            self._assert_fused_equals_reference(layout, source)

    def test_fused_energies_equal_reference_for_split_and_unaligned_chunks(self) -> None:
        """Порции исходного прохода: много порций с остатком; число строк порции не кратно 4; порции короче 32 строк."""
        for dense_rows, local_rows in ((64, 96), (41, 67), (16, 23)):
            layout = self._engine_layout(points=700)
            engine = layout._torch_engine
            engine.CHUNK_ELEMENTS = layout._point_count * dense_rows
            engine.LOCAL_CHUNK_ELEMENTS = layout._point_count * local_rows
            self._configure(layout, source="similarity")
            self._assert_fused_equals_reference(layout, f"chunks {dense_rows}/{local_rows}")
            self.assertTrue(engine._fused_engine()._merge_ok, "объединение порций не должно отключаться")

    def test_jaccard_similarity_matches_reference(self) -> None:
        for source in ("similarity", "common", "product"):
            layout = self._engine_layout(points=500, similarity="jaccard", lambda_threshold=0.4)
            self._configure(layout, source=source)
            engine = layout._torch_engine
            layout._sync_arrays()
            host = {"positions": layout._pos_np, "grid": layout._grid_np}
            engine.update_positions(layout._pos_np[:, 0], layout._pos_np[:, 1])
            for eta in (10.0, None):
                for local_radius, pair_radius in ((layout.width, layout.width), (4, 3)):
                    idx_a, idx_b, pos_a, pos_b = _pairs(layout, 200, pair_radius)
                    arguments = dict(
                        neighbourhood=True, local_radius=local_radius, kernel="gauss", euclidean=False,
                        distance_eps=layout._distance_eps, kernel_width=1.0, lambda_threshold=0.4, eta=eta,
                    )
                    fused = engine._fused_engine().pair_energies(idx_a, idx_b, pos_a, pos_b, **host, **arguments)
                    reference = engine._pair_energies_reference(idx_a, idx_b, pos_a, pos_b, **arguments)
                    self.assertTrue(np.array_equal(_bits(fused[0]), _bits(reference[0])), f"{source} eta={eta}")
                    self.assertTrue(np.array_equal(_bits(fused[1]), _bits(reference[1])), f"{source} eta={eta}")

    def test_engine_verifies_fused_energies_and_returns_reference_numbers(self) -> None:
        layout = self._engine_layout(points=400)
        engine = layout._torch_engine
        layout._ensure_torch_positions()
        idx_a, idx_b, pos_a, pos_b = _pairs(layout, 150, 4)
        arguments = dict(
            neighbourhood=True, local_radius=4, kernel="gauss", euclidean=False, distance_eps=layout._distance_eps,
            kernel_width=1.0, lambda_threshold=0.6, eta=10.0,
        )
        first = engine.pair_energies(idx_a, idx_b, pos_a, pos_b, **arguments)   # сверка: исходный проход
        second = engine.pair_energies(idx_a, idx_b, pos_a, pos_b, **arguments)  # слитое ядро
        self.assertTrue(engine.fused_active)
        self.assertTrue(np.array_equal(_bits(first[0]), _bits(second[0])) and np.array_equal(_bits(first[1]), _bits(second[1])))

    def _run_smooth(self, fused: bool) -> tuple[list[tuple[int, int]], list[float], Layout]:
        """Раскладка в режиме сглаживания: гауссово ядро, сигмоида, отжиг, геометрическое убывание радиуса."""
        layout = _layout(points=600, fused_energy=fused, fused_verify_every=7)
        if layout._torch_engine is None:
            self.skipTest("torch с GPU недоступен")
        adaptive = AdaptiveLayoutConfig(
            start_radius=layout.width, end_radius=1, swap_ratio_trigger=0.0, lambda_step=None, lambda_max=0.8,
            radius_schedule="geometric", radius_factor=0.5, swap_window=3, temperature=0.1,
            temperature_radius_max=int(layout.width * 0.4), temperature_radius_min=3,
        )
        layout.run(steps=150, pairs_per_step=120, pair_radius=adaptive.start_radius, mode="long",
                   min_swap_ratio=0.0, min_swap_window=None, log_every=None, log_visuals=False,
                   energy_radius=3, energy_every=25, adaptive_params=adaptive)
        layout.run(steps=30, pairs_per_step=80, pair_radius=3, mode="short", local_radius=3,
                   min_swap_ratio=0.0, min_swap_window=None, log_every=None, log_visuals=False,
                   short_optim=ShortLayoutOptimConfig(energy_radius=3, energy_recalc_every=10, partitions=4,
                                                      similarity_cutoff=0.6))
        return layout.positions(), layout._compute_point_energies(3), layout

    def test_smooth_layout_is_identical_with_and_without_fused_energies(self) -> None:
        """§5.5–§5.7: раскладка со слитым ядром совпадает с раскладкой исходными проходами шаг в шаг."""
        fused_positions, fused_energies, fused_layout = self._run_smooth(True)
        if not fused_layout._torch_engine.fused_active:
            self.skipTest("слитое ядро недоступно (нужен torch с CUDA)")
        plain_positions, plain_energies, plain_layout = self._run_smooth(False)
        self.assertFalse(plain_layout._torch_engine.fused_active)
        self.assertEqual(fused_positions, plain_positions)
        self.assertEqual(fused_energies, plain_energies)
        self.assertEqual(fused_layout._long_swap_radius_sum, plain_layout._long_swap_radius_sum)
        self.assertEqual(fused_layout._long_swap_count, plain_layout._long_swap_count)
        self.assertEqual(fused_layout._rng.getstate(), plain_layout._rng.getstate())
        self.assertEqual(fused_layout._np_rng.bit_generator.state, plain_layout._np_rng.bit_generator.state)

    def test_strict_layout_is_identical_with_and_without_fused_energies(self) -> None:
        """Режим статьи (§5.5.1, §7.1.7) без строк сходства: проход по всему V слитым ядром и операциями torch."""

        def run(fused: bool) -> tuple[list[tuple[int, int]], list[float]]:
            layout = _layout(points=500, eta=None, long_kernel="distance", sparse_long=False, fused_energy=fused,
                             fused_verify_every=5)
            if layout._torch_engine is None:
                self.skipTest("torch с GPU недоступен")
            adaptive = AdaptiveLayoutConfig(
                start_radius=layout.width // 2, end_radius=1, swap_ratio_trigger=0.7, lambda_step=0.05,
                lambda_max=0.75, radius_schedule="swaps", radius_factor=0.5, swap_window=3,
            )
            layout.run(steps=80, pairs_per_step=60, pair_radius=layout.width // 2, mode="long",
                       min_swap_ratio=0.0, min_swap_window=None, log_every=None, log_visuals=False,
                       energy_radius=3, energy_every=20, adaptive_params=adaptive)
            layout.run(steps=30, pairs_per_step=60, pair_radius=3, mode="short", local_radius=3,
                       min_swap_ratio=0.0, min_swap_window=None, log_every=None, log_visuals=False,
                       short_optim=ShortLayoutOptimConfig(energy_radius=3, energy_recalc_every=10, partitions=4,
                                                          similarity_cutoff=0.6))
            return layout.positions(), layout._compute_point_energies(3)

        self.assertEqual(run(True), run(False))


class ArrayStepTests(unittest.TestCase):
    """§5.6, §5.10: шаг на массивах применяет те же обмены, что обход пар по порядку; списки догоняют массивы."""

    def test_first_come_swaps_equal_sequential_rule(self) -> None:
        rng = np.random.default_rng(7)
        for cells, count in ((40, 60), (400, 300), (5000, 700), (9, 30), (100000, 500), (50, 1)):
            for _ in range(20):
                first = rng.integers(0, cells, count)
                second = rng.integers(0, cells, count)
                used: set[int] = set()
                expected = []
                for index, (a, b) in enumerate(zip(first.tolist(), second.tolist())):
                    if a in used or b in used:
                        continue
                    used.add(a)
                    used.add(b)
                    expected.append(index)
                self.assertEqual(Layout._first_come_swaps(first, second).tolist(), expected)

    def test_array_swaps_equal_list_swaps(self) -> None:
        """_apply_swaps_arrays меняет расстановку так же, как _apply_tracked_swaps, включая учёт радиусов обменов."""
        by_arrays = _layout(points=300, use_gpu=False)
        by_lists = _layout(points=300, use_gpu=False)
        by_arrays._sync_arrays()
        for round_index in range(6):
            cells = by_arrays._sample_pairs_long_np(180, 5)
            listed = [((int(y1), int(x1), int(y2), int(x2)), 1.0) for y1, x1, y2, x2 in cells]
            applied_arrays = by_arrays._apply_swaps_arrays(cells, mode="long")
            applied_lists = by_lists._apply_tracked_swaps(listed, mode="long")
            self.assertEqual(applied_arrays, applied_lists, round_index)
            self.assertTrue(by_arrays._lists_stale)
            self.assertEqual(by_arrays.positions(), by_lists.positions())
            self.assertEqual(by_arrays.grid, by_lists.grid)
            self.assertEqual(by_arrays._row_occupied, by_lists._row_occupied)
            self.assertEqual(by_arrays._pos_y, by_lists._pos_y)
            self.assertEqual(by_arrays._pos_x, by_lists._pos_x)
            self.assertEqual(by_arrays._long_swap_radius_sum, by_lists._long_swap_radius_sum)
            self.assertEqual(by_arrays._long_swap_count, by_lists._long_swap_count)
            by_arrays._sync_arrays()

    def test_lists_are_rebuilt_in_place_after_array_swaps(self) -> None:
        layout = _layout(points=200, use_gpu=False)
        grid = layout.grid
        positions = layout._positions
        layout._sync_arrays()
        cells = layout._sample_pairs_long_np(90, 4)
        self.assertGreater(layout._apply_swaps_arrays(cells, mode="short"), 0)
        self.assertIs(layout.grid, grid, "объект списка сетки тот же")
        self.assertIs(layout._positions, positions, "объект списка позиций тот же")
        for index, (y, x) in enumerate(positions):
            self.assertEqual(grid[y][x], index)
        occupied = sum(1 for row in grid for cell in row if cell is not None)
        self.assertEqual(occupied, layout._point_count)

    def test_bulk_uniform_draws_equal_single_draws(self) -> None:
        """Отбор обменов при отжиге: числа генератора пачкой — те же числа и то же состояние генератора."""
        for count in (1, 5, 511, 512, 700, 5000):
            one_by_one = random.Random(11)
            in_bulk = random.Random(11)
            for _ in range(3):
                expected = np.array([one_by_one.random() for _ in range(count)])
                self.assertTrue(np.array_equal(_UniformDraws.take(in_bulk, count), expected), count)
            self.assertEqual(one_by_one.getstate(), in_bulk.getstate())
        self.assertTrue(_UniformDraws._verified())


class EncodingHelpersTests(unittest.TestCase):
    """Строка битов и псевдоцвет кода (§5.14) — те же, что при побитовом обходе."""

    def test_to01_equals_bitwise_join(self) -> None:
        rng = random.Random(3)
        for length in (1, 7, 64, 513):
            code = BitArray.from_indices(length, rng.sample(range(length), rng.randrange(0, length + 1)))
            self.assertEqual(code.to01(), "".join("1" if bit else "0" for bit in code))

    def test_code_color_equals_sum_of_bit_colors(self) -> None:
        import colorsys

        rng = random.Random(4)
        colorizer = EncoderCodeColorizer(min_hue=0.0, max_hue=300.0)
        for _ in range(20):
            length = 128
            code = BitArray.from_indices(length, rng.sample(range(length), 25))
            red = green = blue = 0.0
            for index in code.active_indices():
                hue = 0.0 + (300.0 - 0.0) * index / (length - 1)
                r, g, b = colorsys.hsv_to_rgb((hue % 360.0) / 360.0, 1.0, 1.0)
                red += r
                green += g
                blue += b
            scale = 255.0 / max(red, green, blue)
            expected = (int(round(red * scale)), int(round(green * scale)), int(round(blue * scale)))
            self.assertEqual(colorizer.color(code), expected)


if __name__ == "__main__":
    unittest.main()
