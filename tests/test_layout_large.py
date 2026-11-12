"""Большие раскладки: коды массивами, потоковый файл раскладки, энергия по подмножеству B и по выборке ячеек.

Компактный путь данных не меняет результат: таблица кодов даёт те же точки, цвета (§5.14) и тот же файл
раскладки, что объекты на каждую точку. Слитое ядро по подмножеству B и по выборке ячеек окрестности (§5.9.4)
даёт побитово те же φ_c, φ_s, что исходный проход движка. Режим больших раскладок включается настройкой
только выше границы числа точек.
"""
from __future__ import annotations

import io
import json
import os
import random
import tempfile
import unittest

os.environ.setdefault("RERUN", "off")

import numpy as np  # noqa: E402

import main  # noqa: E402
from damp.encoding.bitarray import BitArray  # noqa: E402
from damp.encoding.damp_encoder import EncoderCodeColorizer  # noqa: E402
from damp.layout.code_table import LayoutCodeTable  # noqa: E402
from damp.layout.damp_layout import AdaptiveLayoutConfig, Layout  # noqa: E402
from damp.layout.payload import LayoutEncodedCode, LayoutJsonWriter, LayoutPayloadBuilder  # noqa: E402
from damp.layout.seeded import SeededPlacement  # noqa: E402
from damp.logging import LOGGER  # noqa: E402

LOGGER.configure_intervals({}, default_interval=10**9)

_CODE_LENGTH = 96
_GROUPS = 12


def _records(points: int, seed: int) -> list[tuple[str, float, BitArray]]:
    """Коды в порядке поступления: (подпись, hue, код); hue чередуются, чтобы группировка меняла порядок."""
    rng = random.Random(seed)
    prototypes = [rng.sample(range(_CODE_LENGTH), 20) for _ in range(_GROUPS)]
    records = []
    for index in range(points):
        group = (index * 5) % _GROUPS
        bits = list(prototypes[group])
        for _ in range(rng.randrange(0, 5)):
            bits[rng.randrange(len(bits))] = rng.randrange(_CODE_LENGTH)
        records.append((str(index % 10), float(group) * 7.5, BitArray.from_indices(_CODE_LENGTH, sorted(set(bits)))))
    return records


def _builder(records) -> LayoutPayloadBuilder:
    builder = LayoutPayloadBuilder(
        similarity=None, lambda_threshold=None, eta=None, metadata={"encoder": "тест", "patch": {"size": 9, "mode": "w"}}
    )
    for label, hue, code in records:
        builder.add_code(label=label, hue=hue, code=code)
    return builder


def _mapping(records) -> dict[float, list[LayoutEncodedCode]]:
    grouped: dict[float, list[LayoutEncodedCode]] = {}
    for label, hue, code in records:
        grouped.setdefault(hue, []).append(LayoutEncodedCode(code=code, label=label))
    return grouped


_LAYOUT_PARAMS = dict(
    similarity="cosine", lambda_threshold=0.6, eta=10.0, use_gpu=True, seed=5,
    long_distance="euclidean", short_distance="squared", long_kernel="gauss", long_kernel_width=1.0,
)


def _bits(values: "np.ndarray") -> "np.ndarray":
    return np.ascontiguousarray(values).view(np.int32)


class CodeTableTests(unittest.TestCase):
    """Таблица кодов хранит те же коды, подписи и порядок точек, что объекты на каждую точку."""

    def test_table_matches_objects(self) -> None:
        records = _records(300, 3)
        builder = _builder(records)
        table = builder.table()
        self.assertEqual(table.count, len(records))
        self.assertEqual(table.values(), [code.to01() for _, _, code in records])
        self.assertEqual(table.labels(), [label for label, _, _ in records])
        self.assertEqual(table.hues.tolist(), [hue for _, hue, _ in records])
        self.assertEqual(table.ones.tolist(), [code.count() for _, _, code in records])
        self.assertEqual([code.to01() for code in table.codes()], table.values())
        # Слова по 64 бита — те же, что давала упаковка матрицы бит раскладки.
        bits = np.stack([np.frombuffer(code._bits, dtype=np.uint8) for _, _, code in records])
        padded = np.zeros((bits.shape[0], 128), dtype=np.uint8)
        padded[:, :_CODE_LENGTH] = bits
        self.assertTrue(np.array_equal(table.words64(), np.packbits(padded, axis=1).view(np.uint64)))

    def test_hue_group_order_matches_dictionary_grouping(self) -> None:
        records = _records(257, 4)
        builder = _builder(records)
        grouped = builder.codes_by_hue()
        expected = [(entry.label, hue, entry.code.to01()) for hue, entries in grouped.items() for entry in entries]
        table = builder.code_table()
        actual = list(zip(table.labels(), table.hues.tolist(), table.values()))
        self.assertEqual(actual, expected)
        self.assertEqual(table.hue_count(), len(grouped))

    def test_packed_chunks_equal_single_codes(self) -> None:
        records = _records(120, 6)
        single = _builder(records).table()
        chunked = LayoutPayloadBuilder(similarity=None, lambda_threshold=None, eta=None)
        for start in range(0, len(records), 7):
            part = records[start : start + 7]
            for label in sorted({label for label, _, _ in part}):
                rows = [(hue, code) for item_label, hue, code in part if item_label == label]
                bits = np.stack([np.frombuffer(code._bits, dtype=np.uint8) for _, code in rows])
                chunked.add_packed(
                    label=label, hues=[hue for hue, _ in rows], packed=LayoutCodeTable.pack_bits(bits),
                    code_length=_CODE_LENGTH,
                )
        self.assertEqual(sorted(chunked.table().values()), sorted(single.values()))
        self.assertEqual(len(chunked), len(records))


class LayoutJsonTests(unittest.TestCase):
    """Файл раскладки, записанный потоком, — тот же текст, что json.dump словаря с точками."""

    def _reference_text(self, payload: dict) -> str:
        buffer = io.StringIO()
        json.dump(payload, buffer, ensure_ascii=True, indent=2)
        return buffer.getvalue()

    def test_base_file_matches_json_dump(self) -> None:
        builder = _builder(_records(90, 7))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "base.json")
            builder.save_base(path)
            with open(path, "r", encoding="utf-8") as fp:
                text = fp.read()
        self.assertEqual(text, self._reference_text(dict(builder.base_payload())))

    def test_layout_file_matches_json_dump_and_keeps_metadata(self) -> None:
        records = _records(140, 8)
        builder = _builder(records)
        layout = Layout(builder.code_table(), **_LAYOUT_PARAMS)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "layout.json")
            builder.save_base(path)
            layout.save_json(path)
            with open(path, "r", encoding="utf-8") as fp:
                text = fp.read()
            saved = json.loads(text)
            # Большой файл читается по началу и концу: ключи верхнего уровня — те же.
            limit = LayoutJsonWriter.LARGE_FILE_BYTES
            LayoutJsonWriter.LARGE_FILE_BYTES = 0
            try:
                header = LayoutJsonWriter.read_header(path)
            finally:
                LayoutJsonWriter.LARGE_FILE_BYTES = limit
        table = layout.code_table
        positions = layout.positions()
        expected = dict(builder.base_payload())
        expected.update(
            {
                "width": layout.width, "height": layout.height, "points": table.count, "similarity": "cosine",
                "lambda_threshold": 0.6, "eta": 10.0, "pair_radius": None,
                "layout": [
                    {
                        "index": index, "y": positions[index][0], "x": positions[index][1],
                        "label": table.label(index), "value": table.values(index, index + 1)[0],
                        "hue": float(table.hues[index]), "ones": int(table.ones[index]),
                    }
                    for index in range(table.count)
                ],
            }
        )
        self.assertEqual(text, self._reference_text(expected))
        self.assertEqual(saved["encoder"], "тест")
        self.assertEqual(list(header), list(saved))
        self.assertEqual({key: value for key, value in header.items() if key != "layout"},
                         {key: value for key, value in saved.items() if key != "layout"})
        self.assertIsNone(header["layout"])


class ColorTests(unittest.TestCase):
    """§5.14: цвета кодов массивом — те же числа, что у расчёта по одному коду."""

    def test_vector_colors_equal_scalar_colors(self) -> None:
        rng = np.random.default_rng(11)
        bits = (rng.random((500, 64)) < 0.2).astype(np.uint8)
        bits[7] = 0
        bits[8] = 1
        codes = [BitArray.from_bits(row.tobytes(), np.flatnonzero(row).tolist()) for row in bits]
        for colorizer in (
            EncoderCodeColorizer(min_hue=0.0, max_hue=300.0),
            EncoderCodeColorizer(min_hue=20.0, max_hue=250.0, start_bit=10, bit_length=30),
        ):
            expected = [colorizer.color(code) for code in codes]
            self.assertEqual([tuple(row) for row in colorizer.colors(bits).tolist()], expected)

    def test_layout_colors_from_table_equal_object_colors(self) -> None:
        records = _records(200, 9)
        layout = Layout(_builder(records).code_table(), component_period=45.0, **_LAYOUT_PARAMS)
        self.assertEqual(layout.colors_rgb(), [layout.colorizer.color(point.code) for point in layout._points])
        legacy = Layout(_mapping(records), component_period=45.0, **_LAYOUT_PARAMS)
        self.assertEqual(layout.colors_rgb(), legacy.colors_rgb())
        for period in (None, 45.0):
            self.assertEqual(layout.component_colors_rgb(period=period), legacy.component_colors_rgb(period=period))


class TableLayoutTests(unittest.TestCase):
    """Раскладка от таблицы кодов — та же, что от отображения hue → коды."""

    def test_same_layout_from_table_and_mapping(self) -> None:
        records = _records(400, 12)
        from_table = Layout(_builder(records).code_table(), **_LAYOUT_PARAMS)
        from_mapping = Layout(_mapping(records), **_LAYOUT_PARAMS)
        self.assertEqual(from_table.positions(), from_mapping.positions())
        self.assertEqual(from_table.labels(), from_mapping.labels())
        self.assertEqual(from_table.values(), from_mapping.values())
        self.assertEqual(from_table.hues(), from_mapping.hues())
        for layout in (from_table, from_mapping):
            adaptive = AdaptiveLayoutConfig(
                start_radius=layout.width, end_radius=1, swap_ratio_trigger=0.0, lambda_step=None,
                radius_schedule="geometric", swap_window=3, temperature=0.1,
                temperature_radius_max=10, temperature_radius_min=3,
            )
            layout.run(steps=40, pairs_per_step=300, pair_radius=layout.width, mode="long",
                       min_swap_ratio=0.0, adaptive_params=adaptive)
        self.assertEqual(from_table.positions(), from_mapping.positions())
        self.assertEqual(from_table.point_energies(3), from_mapping.point_energies(3))
        self.assertTrue(np.array_equal(from_table.render_image(log=False), from_mapping.render_image(log=False)))

    def test_initial_placement_matches_shuffled_cells(self) -> None:
        records = _records(150, 13)
        layout = Layout(_builder(records).code_table(), **_LAYOUT_PARAMS)
        cells = [(y, x) for y in range(layout.height) for x in range(layout.width)]
        random.Random(5).shuffle(cells)
        self.assertEqual(layout.positions(), cells[: len(records)])
        grid = layout.grid
        for index, (y, x) in enumerate(layout.positions()):
            self.assertEqual(grid[y][x], index)


class LargeEnergyTests(unittest.TestCase):
    """§5.9.4: слитое ядро по подмножеству B и по выборке ячеек окна побитово равно исходному проходу."""

    def _layout(self, **kwargs) -> Layout:
        layout = Layout(_builder(_records(500, 21)).code_table(), **{**_LAYOUT_PARAMS, **kwargs})
        engine = layout._torch_engine
        if engine is None or not engine.fused_active:
            self.skipTest("слитое ядро недоступно (нужен torch с CUDA)")
        engine.configure_fused(
            enabled=True, block_elements=3_000_000, table_max_bytes=0, table_memory_fraction=1.0,
            verify_every=0, similarity_table=False,
        )
        return layout

    def _pairs(self, layout: Layout, count: int, radius: int):
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

    def _compare(self, layout: Layout, label: str, **extra) -> None:
        engine = layout._torch_engine
        layout._sync_arrays()
        host = {"positions": layout._pos_np, "grid": layout._grid_np}
        engine.update_positions(layout._pos_np[:, 0], layout._pos_np[:, 1])
        for eta in (10.0, None):
            for kernel, local_radius, pair_radius in (("gauss", 9, 9), ("gauss", 5, 4), ("inverse", 6, 3)):
                for count in (700, 41):
                    idx_a, idx_b, pos_a, pos_b = self._pairs(layout, count, pair_radius)
                    arguments = dict(
                        neighbourhood=True, local_radius=local_radius, kernel=kernel, euclidean=False,
                        distance_eps=layout._distance_eps, kernel_width=1.0, lambda_threshold=0.6, eta=eta, **extra,
                    )
                    fused = engine._fused_engine().pair_energies(idx_a, idx_b, pos_a, pos_b, **host, **arguments)
                    self.assertIsNotNone(fused, label)
                    reference = engine._pair_energies_reference(idx_a, idx_b, pos_a, pos_b, **arguments)
                    case = f"{label} eta={eta} {kernel} r={local_radius} pairs={count}"
                    self.assertTrue(np.array_equal(_bits(fused[0]), _bits(reference[0])), case)
                    self.assertTrue(np.array_equal(_bits(fused[1]), _bits(reference[1])), case)
                    self.assertGreater(float(np.abs(reference[0]).max()), 0.0, case)

    def test_subset_energies_equal_reference(self) -> None:
        layout = self._layout()
        subset = random.Random(1).sample(range(layout.point_count), 128)
        self._compare(layout, "subset", eval_indices=subset)
        # Дальняя энергия статьи (§5.5.1) по подмножеству B.
        engine = layout._torch_engine
        idx_a, idx_b, pos_a, pos_b = self._pairs(layout, 300, layout.width // 2)
        arguments = dict(
            neighbourhood=False, local_radius=None, kernel="inverse", euclidean=True, distance_eps=1e-6,
            kernel_width=1.0, lambda_threshold=0.6, eta=None, eval_indices=subset,
        )
        fused = engine._fused_engine().pair_energies(
            idx_a, idx_b, pos_a, pos_b, positions=layout._pos_np, grid=layout._grid_np, **arguments
        )
        reference = engine._pair_energies_reference(idx_a, idx_b, pos_a, pos_b, **arguments)
        self.assertTrue(np.array_equal(_bits(fused[0]), _bits(reference[0])))
        self.assertTrue(np.array_equal(_bits(fused[1]), _bits(reference[1])))

    def test_sampled_window_energies_equal_reference(self) -> None:
        layout = self._layout()
        rng = np.random.default_rng(2)
        offsets = (rng.integers(-11, 12, size=90), rng.integers(-11, 12, size=90))
        self._compare(layout, "sample", sample_offsets=offsets)

    def test_packed_codes_engine_equals_float_codes_engine(self) -> None:
        """Без кодов float32 на устройстве (большое N) строки кодов распаковываются из 64-битных слов."""
        from damp.layout.pair_energy_gpu import TorchPairEnergyEngine

        layout = self._layout()
        engine = layout._torch_engine
        limit = TorchPairEnergyEngine.FLOAT_CODES_MAX_BYTES
        TorchPairEnergyEngine.FLOAT_CODES_MAX_BYTES = 0
        try:
            packed_layout = self._layout()
        finally:
            TorchPairEnergyEngine.FLOAT_CODES_MAX_BYTES = limit
        packed_engine = packed_layout._torch_engine
        self.assertTrue(engine.has_float_codes)
        self.assertFalse(packed_engine.has_float_codes)
        indices = engine.torch_mod.arange(layout.point_count, device=engine.torch_device)
        self.assertTrue(bool(engine.torch_mod.equal(packed_engine._code_rows(indices), engine.codes)))
        subset = random.Random(3).sample(range(layout.point_count), 96)
        self._compare(packed_layout, "packed subset", eval_indices=subset)
        rng = np.random.default_rng(4)
        self._compare(packed_layout, "packed sample",
                      sample_offsets=(rng.integers(-9, 10, size=70), rng.integers(-9, 10, size=70)))
        # Точные окна и энергии точек без кодов float32 — те же числа.
        packed_layout._sync_arrays()
        self.assertEqual(packed_layout.point_energies(3), layout.point_energies(3))

    def test_engine_verification_on_part_of_pairs(self) -> None:
        from damp.layout.pair_energy_gpu import TorchPairEnergyEngine

        layout = self._layout()
        engine = layout._torch_engine
        engine.configure_fused(
            enabled=True, block_elements=3_000_000, table_max_bytes=0, table_memory_fraction=1.0,
            verify_every=1, similarity_table=False,
        )
        idx_a, idx_b, pos_a, pos_b = self._pairs(layout, 900, 6)
        arguments = dict(
            neighbourhood=True, local_radius=6, kernel="gauss", euclidean=False, distance_eps=1e-6,
            kernel_width=1.0, lambda_threshold=0.6, eta=10.0, positions=layout._pos_np, grid=layout._grid_np,
        )
        full = engine.pair_energies(idx_a, idx_b, pos_a, pos_b, **arguments)
        limit = TorchPairEnergyEngine.VERIFY_MAX_PAIRS
        TorchPairEnergyEngine.VERIFY_MAX_PAIRS = 128
        try:
            partial = engine.pair_energies(idx_a, idx_b, pos_a, pos_b, **arguments)
        finally:
            TorchPairEnergyEngine.VERIFY_MAX_PAIRS = limit
        self.assertTrue(engine.fused_active)
        self.assertTrue(np.array_equal(_bits(full[0]), _bits(partial[0])))
        self.assertTrue(np.array_equal(_bits(full[1]), _bits(partial[1])))


class NeighbourhoodSampleTests(unittest.TestCase):
    """§5.9.4: выборка ячеек окрестности — по ячейке из слоя, каждая ячейка окна равновероятна."""

    def test_stratified_sample_is_uniform_over_window(self) -> None:
        layout = Layout(_builder(_records(300, 31)).code_table(), neighbourhood_sample_side=4, **_LAYOUT_PARAMS)
        pos_a = np.array([[10, 10], [3, 4]], dtype=np.int64)
        pos_b = np.array([[12, 9], [5, 8]], dtype=np.int64)
        local_radius = 9
        reach = 10  # радиус окрестности max(⌈‖a − b‖⌉ + 1, 9) = 9, полуширина окна ⌈9⌉ + 1
        side = 2 * reach + 1
        stride = side // 4
        hits = np.zeros((side, side), dtype=np.int64)
        draws = 4000
        for _ in range(draws):
            dy, dx = layout._neighbourhood_sample(pos_a, pos_b, local_radius, 0)
            inside = (np.abs(dy) <= reach) & (np.abs(dx) <= reach)
            # В каждом слое — ровно одна ячейка: выборка покрывает окно без повторов.
            self.assertEqual(len(set(zip(dy.tolist(), dx.tolist()))), dy.size)
            self.assertLess(int(np.abs(dy).max()), reach + stride)
            hits[dy[inside] + reach, dx[inside] + reach] += 1
        expected = draws / stride**2
        self.assertLess(float(np.abs(hits - expected).max()), 6.0 * expected**0.5)
        # Среднее число ячеек выборки внутри окна — доля 1/t² от его ячеек.
        self.assertLess(abs(int(hits.sum()) / draws - side * side / stride**2), 0.02 * side * side / stride**2)

    def test_small_window_is_whole_disc(self) -> None:
        """Малое окно не прореживается: берутся все ячейки, которые могут попасть в окрестность (круг)."""
        layout = Layout(_builder(_records(300, 32)).code_table(), neighbourhood_sample_side=12, **_LAYOUT_PARAMS)
        engine = layout._torch_engine
        if engine is None or not engine.fused_active:
            self.skipTest("слитое ядро недоступно (нужен torch с CUDA)")
        self.assertIsNone(Layout(_mapping(_records(50, 33)), **_LAYOUT_PARAMS)._neighbourhood_sample(
            np.array([[10, 10]]), np.array([[11, 10]]), 99, 0))
        layout._sync_arrays()
        cells = layout._sample_pairs_long_np(400, 6)
        idx_a = layout._grid_np[cells[:, 0], cells[:, 1]]
        idx_b = layout._grid_np[cells[:, 2], cells[:, 3]]
        keep = idx_a >= 0
        idx_a, idx_b, pos_a, pos_b = idx_a[keep], idx_b[keep], cells[keep, 0:2], cells[keep, 2:4]
        offsets = layout._neighbourhood_sample(pos_a, pos_b, 6, 0)
        reach = engine.window_reach(pos_a, pos_b, 6)
        self.assertLess(offsets[0].size, (2 * reach + 1) ** 2)
        self.assertEqual(len(set(zip(offsets[0].tolist(), offsets[1].tolist()))), offsets[0].size)
        arguments = dict(
            neighbourhood=True, local_radius=6, kernel="gauss", euclidean=False, distance_eps=1e-6,
            kernel_width=1.0, lambda_threshold=0.6, eta=10.0,
        )
        engine.update_positions(layout._pos_np[:, 0], layout._pos_np[:, 1])
        whole = engine._pair_energies_reference(idx_a, idx_b, pos_a, pos_b, **arguments)
        disc = engine._pair_energies_reference(idx_a, idx_b, pos_a, pos_b, sample_offsets=offsets, **arguments)
        # Те же слагаемые (ячейки вне круга дают нули); порядок сложения float32 другой.
        self.assertTrue(np.allclose(whole[0], disc[0], rtol=1e-5, atol=1e-6))
        self.assertTrue(np.allclose(whole[1], disc[1], rtol=1e-5, atol=1e-6))

    def test_sampled_layout_run_is_deterministic(self) -> None:
        def run() -> list[tuple[int, int]]:
            layout = Layout(
                _builder(_records(600, 34)).code_table(), neighbourhood_sample_side=3,
                long_subset_size=64, long_subset_refresh=1, long_subset_min_neighbours=20, **_LAYOUT_PARAMS,
            )
            adaptive = AdaptiveLayoutConfig(
                start_radius=layout.width, end_radius=1, swap_ratio_trigger=0.0, lambda_step=None,
                radius_schedule="geometric", swap_window=3, temperature=0.1,
                temperature_radius_max=10, temperature_radius_min=3,
            )
            layout.run(steps=60, pairs_per_step=400, pair_radius=layout.width, mode="long",
                       min_swap_ratio=0.0, adaptive_params=adaptive)
            return layout.positions()

        first = run()
        self.assertEqual(first, run())
        self.assertEqual(len(set(first)), len(first))


class SeededPlacementTests(unittest.TestCase):
    """Большая раскладка от затравки: точки встают возле самых похожих точек раскладки подмножества."""

    def test_match_cells_is_bijection_and_keeps_order(self) -> None:
        rng = np.random.default_rng(41)
        side = 40
        cell_y, cell_x = np.divmod(rng.permutation(side * side)[:1200], side)
        target_y = rng.random(1200) * side
        target_x = rng.random(1200) * side
        match = SeededPlacement.match_cells(target_y, target_x, cell_y, cell_x)
        self.assertEqual(sorted(match.tolist()), list(range(1200)))
        shift = np.hypot(cell_y[match] + 0.5 - target_y, cell_x[match] + 0.5 - target_x)
        self.assertLess(float(shift.mean()), 3.0)
        # Цели совпадают с центрами ячеек: каждая точка получает свою ячейку.
        exact = SeededPlacement.match_cells(cell_y + 0.5, cell_x + 0.5, cell_y, cell_x)
        self.assertTrue(np.array_equal(exact, np.arange(1200)))

    def test_small_seed_islands_are_not_parents(self) -> None:
        occupied = np.zeros((12, 12), dtype=bool)
        occupied[1:8, 1:8] = True
        occupied[10, 10] = occupied[11, 11] = True
        occupied[10, 2] = True
        labels = SeededPlacement.components(occupied)
        self.assertEqual(len(np.unique(labels[occupied])), 3)
        self.assertEqual(int(labels[10, 10]), int(labels[11, 11]))
        table = _builder(_records(400, 46)).code_table()
        placement = SeededPlacement(table, seed_count=52, seed=3)
        placement.PARENT_COMPONENT_FRACTION = 0.1
        seed_cells = np.argwhere(occupied)
        mask = placement._parent_mask(seed_cells, (12, 12))
        self.assertEqual(int(mask.sum()), 49)
        self.assertFalse(bool(mask[-3:].any()))

    def test_cells_follow_seed_layout(self) -> None:
        records = _records(900, 42)
        table = _builder(records).code_table()
        placement = SeededPlacement(table, seed_count=150, seed=5)
        self.assertEqual(placement.seed_count, 150)
        self.assertTrue(np.array_equal(placement.seed_indices, np.sort(placement.seed_indices)))
        seed_layout = Layout(placement.seed_table(), **_LAYOUT_PARAMS)
        seed_cells = seed_layout.cells_array()
        seed_shape = (seed_layout.height, seed_layout.width)
        shape = (37, 37)
        cells = placement.cells(seed_cells, seed_shape, shape)
        self.assertEqual(cells.shape, (900, 2))
        self.assertEqual(len({(int(y), int(x)) for y, x in cells}), 900)
        self.assertTrue(bool(((cells >= 0) & (cells < 37)).all()))
        # Родитель — самая похожая точка затравки: проверка на процессоре даёт тех же родителей.
        parent_seeds, parent = placement.parents(seed_cells, seed_shape)
        host = placement._match_host(placement.seed_indices[parent_seeds])[0][:, 0]
        host[placement.seed_indices[parent_seeds]] = np.arange(parent_seeds.size)
        words = table.words64()
        ones = table.ones.astype(np.float64)

        def similarity(points: "np.ndarray", parents: "np.ndarray") -> "np.ndarray":
            common = np.bitwise_count(words[points] & words[parents]).sum(axis=1)
            return common / np.sqrt(ones[points] * ones[parents])

        everyone = np.arange(900)
        parent_points = placement.seed_indices[parent_seeds]
        self.assertTrue(np.allclose(similarity(everyone, parent_points[parent]), similarity(everyone, parent_points[host])))
        # Точка стоит рядом с блоком своего родителя.
        scale = 37 / seed_shape[0]
        centre = (seed_cells[parent_seeds[parent]] + 0.5) * scale
        distance = np.hypot(cells[:, 0] + 0.5 - centre[:, 0], cells[:, 1] + 0.5 - centre[:, 1])
        self.assertLess(float(np.median(distance)), 3.0 * scale)

    def test_layout_from_initial_cells(self) -> None:
        records = _records(300, 43)
        table = _builder(records).code_table()
        reference = Layout(table, **_LAYOUT_PARAMS)
        cells = reference.cells_array()[::-1].copy()
        layout = Layout(table, initial_cells=cells, **_LAYOUT_PARAMS)
        self.assertEqual(layout.positions(), [tuple(cell) for cell in cells.tolist()])
        self.assertTrue(np.array_equal(layout.cells_array(), cells))
        layout.place_cells(reference.cells_array())
        self.assertEqual(layout.positions(), reference.positions())
        self.assertEqual(layout.point_energies(3), reference.point_energies(3))
        with self.assertRaises(ValueError):
            layout.place_cells(np.zeros((300, 2), dtype=np.int64))
        with self.assertRaises(ValueError):
            layout.place_cells(cells[:10])

    def test_seeded_pipeline_layout(self) -> None:
        """_run_layout от затравки: все точки расставлены, порядок соседей не хуже затравки."""
        records = _records(2400, 44)
        table = _builder(records).code_table()
        params = main.LayoutParameterTuner().tune(table.count)
        import dataclasses

        params = dataclasses.replace(
            params, seed_points=600, long_steps=300, short_steps=20, enforce_pair_radius=True,
            neighbourhood_sample_side=12,
        )
        layout = main._run_layout(table, params, count=0, total_codes=table.count, use_cache=False)
        cells = layout.cells_array()
        self.assertEqual(len({(int(y), int(x)) for y, x in cells}), table.count)
        grid = np.full((layout.height, layout.width), -1, dtype=np.int64)
        grid[cells[:, 0], cells[:, 1]] = np.arange(table.count)
        left, right = grid[:, :-1], grid[:, 1:]
        both = (left >= 0) & (right >= 0)
        groups = table.hues
        same = float((groups[left[both]] == groups[right[both]]).mean())
        # Коды одной группы (одного hue) похожи: соседи по раскладке — чаще всего из одной группы.
        self.assertGreater(same, 0.6)


class DeviceStepTests(unittest.TestCase):
    """§5.3, §5.5, §5.6, §5.10: шаг дальней фазы на устройстве — те же энергии и согласованная расстановка."""

    def _layout(self, *, device_step: bool = True) -> Layout:
        layout = Layout(
            _builder(_records(1500, 51)).code_table(), neighbourhood_sample_side=12, enforce_pair_radius=True,
            device_step=device_step, fused_table_max_bytes=0, fused_similarity_table=False, **_LAYOUT_PARAMS,
        )
        engine = layout._torch_engine
        if engine is None or not engine.fused_active:
            self.skipTest("слитое ядро недоступно (нужен torch с CUDA)")
        return layout

    @staticmethod
    def _same_group_neighbours(layout: Layout) -> float:
        cells = layout.cells_array()
        grid = np.full((layout.height, layout.width), -1, dtype=np.int64)
        grid[cells[:, 0], cells[:, 1]] = np.arange(layout.point_count)
        left, right = grid[:, :-1], grid[:, 1:]
        both = (left >= 0) & (right >= 0)
        hues = layout.code_table.hues
        return float((hues[left[both]] == hues[right[both]]).mean())

    def _run_long(self, layout: Layout) -> None:
        adaptive = AdaptiveLayoutConfig(
            start_radius=layout.width, end_radius=1, swap_ratio_trigger=0.0, lambda_step=None,
            radius_schedule="geometric", swap_window=3, temperature=0.1,
            temperature_radius_max=10, temperature_radius_min=3,
        )
        layout.run(steps=250, pairs_per_step=600, pair_radius=layout.width, mode="long",
                   min_swap_ratio=0.0, adaptive_params=adaptive)

    def test_device_window_energies_match_reference(self) -> None:
        layout = self._layout()
        engine = layout._torch_engine
        fused = engine.fused_on_the_fly()
        self.assertIsNotNone(fused)
        torch = engine.torch_mod
        device = engine.torch_device
        layout._sync_arrays()
        count = layout.point_count
        cells = layout._sample_pairs_long_np(700, 6)
        idx_a = layout._grid_np[cells[:, 0], cells[:, 1]]
        idx_b = layout._grid_np[cells[:, 2], cells[:, 3]]
        keep = idx_a >= 0
        idx_a, idx_b, pos_a, pos_b = idx_a[keep], idx_b[keep], cells[keep, 0:2], cells[keep, 2:4]
        reach = engine.window_reach(pos_a, pos_b, 6)
        rng = np.random.default_rng(6)
        grid_points = np.full(layout.height * layout.width + 1, count, dtype=np.int32)
        grid_points[layout._pos_np[:, 0] * layout.width + layout._pos_np[:, 1]] = np.arange(count)
        engine.update_positions(layout._pos_np[:, 0], layout._pos_np[:, 1])
        for kernel, eta, offsets in (
            ("gauss", 10.0, layout._disc_offsets(reach)),
            ("gauss", None, (rng.integers(-reach, reach + 1, 60), rng.integers(-reach, reach + 1, 60))),
            ("inverse", 10.0, layout._disc_offsets(reach)),
        ):
            arguments = dict(
                kernel=kernel, euclidean=False, distance_eps=1e-6, kernel_width=1.0, lambda_threshold=0.6, eta=eta,
            )
            energy_c, energy_s = fused.window_energies_device(
                torch.as_tensor(idx_a, device=device),
                torch.as_tensor(np.where(idx_b < 0, count, idx_b), device=device),
                torch.as_tensor(pos_a[:, 0] * layout.width + pos_a[:, 1], device=device),
                torch.as_tensor(pos_b[:, 0] * layout.width + pos_b[:, 1], device=device),
                torch.as_tensor(grid_points, device=device),
                offsets=offsets, local_radius=6, **arguments,
            )
            reference = engine._pair_energies_reference(
                idx_a, idx_b, pos_a, pos_b, neighbourhood=True, local_radius=6,
                sample_offsets=offsets, window_reach=reach, **arguments,
            )
            self.assertTrue(np.allclose(energy_c.cpu().numpy(), reference[0], rtol=1e-4, atol=1e-5), kernel)
            self.assertTrue(np.allclose(energy_s.cpu().numpy(), reference[1], rtol=1e-4, atol=1e-5), kernel)
            self.assertGreater(float(np.abs(reference[0]).max()), 0.0)

    def test_device_steps_keep_layout_consistent_and_ordered(self) -> None:
        layout = self._layout()
        self._run_long(layout)
        # Сверка энергий первого шага прошла: шаг на устройстве не отключён.
        self.assertTrue(layout._device_step_enabled)
        self.assertIsNotNone(layout._device_state)
        self.assertGreater(layout._long_swap_count, 0)
        cells = layout.cells_array()
        self.assertEqual(len({(int(y), int(x)) for y, x in cells}), layout.point_count)
        grid = layout.grid
        for index, (y, x) in enumerate(layout.positions()):
            self.assertEqual(grid[y][x], index)
        self.assertEqual(sum(1 for row in grid for cell in row if cell is not None), layout.point_count)
        host = self._layout(device_step=False)
        self._run_long(host)
        self.assertIsNone(host._device_state)
        device_order = self._same_group_neighbours(layout)
        host_order = self._same_group_neighbours(host)
        self.assertGreater(device_order, 0.5)
        self.assertGreater(device_order, host_order - 0.1)
        # Шаги на хосте после шагов на устройстве видят ту же расстановку.
        before = layout.positions()
        layout.step(pairs_per_step=50, pair_radius=None, mode="long", local_radius=None)
        after = layout.positions()
        self.assertEqual(len(set(after)), len(after))
        self.assertGreater(sum(1 for a, b in zip(before, after) if a == b), len(before) // 2)

    def test_device_steps_are_deterministic(self) -> None:
        first = self._layout()
        self._run_long(first)
        second = self._layout()
        self._run_long(second)
        self.assertEqual(first.positions(), second.positions())


class PairRadiusTests(unittest.TestCase):
    """§5.3: в больших раскладках пары, вторая ячейка которых дальше радиуса выбора, в шаг не входят."""

    def test_pairs_beyond_radius_are_dropped(self) -> None:
        layout = Layout(_builder(_records(200, 45)).code_table(), enforce_pair_radius=True, **_LAYOUT_PARAMS)
        pairs = np.array([[0, 0, 0, 3], [5, 5, 9, 9], [2, 2, 5, 6], [7, 1, 7, 2]], dtype=np.int64)
        kept = layout._pairs_within_radius(pairs, 5, mode="long", step_index=0)
        self.assertEqual(kept.tolist(), [[0, 0, 0, 3], [2, 2, 5, 6], [7, 1, 7, 2]])
        near = pairs[[0, 3]]
        self.assertIs(layout._pairs_within_radius(near, 5, mode="long", step_index=0), near)


class TunerTests(unittest.TestCase):
    """§7.1.7: настройка до границы числа точек прежняя; выше — режим больших раскладок (сглаживание)."""

    def test_regime_boundary(self) -> None:
        tuner = main.LayoutParameterTuner()
        small = tuner.tune(main.LAYOUT_LONG_FULL_SPACE_MAX_POINTS)
        self.assertIsNone(small.neighbourhood_sample_side)
        self.assertIsNone(small.long_subset_size)
        self.assertLessEqual(small.long_pairs_per_step, main.LAYOUT_TUNE_LONG_PAIRS_MAX)
        self.assertLessEqual(small.short_pairs_per_step, main.LAYOUT_TUNE_SHORT_PAIRS_MAX)
        large = tuner.tune(1_000_000)
        if not main.LAYOUT_SMOOTH_PINWHEELS:
            self.assertIsNone(large.neighbourhood_sample_side)
            return
        self.assertEqual(large.neighbourhood_sample_side, main.LAYOUT_NEIGHBOURHOOD_SAMPLE_SIDE)
        self.assertTrue(large.enforce_pair_radius)
        self.assertFalse(small.enforce_pair_radius)
        self.assertIsNone(small.seed_points)
        self.assertEqual(large.seed_points, main.LAYOUT_SEED_POINTS)
        if main.LAYOUT_SEED_POINTS is None:
            self.assertEqual(large.long_subset_size, main.LAYOUT_TUNE_LARGE_LONG_SUBSET)
        else:
            # Глобальный порядок — из раскладки затравки: подмножество B дальней фазе не нужно.
            self.assertIsNone(large.long_subset_size)
        self.assertEqual(large.long_pairs_per_step, int(1_000_000 * main.LAYOUT_TUNE_LARGE_LONG_PAIR_FACTOR))
        self.assertEqual(large.short_pairs_per_step, int(1_000_000 * main.LAYOUT_TUNE_LARGE_SHORT_PAIR_FACTOR))
        self.assertLessEqual(large.short_local_radius, main.LAYOUT_TUNE_LARGE_SHORT_RADIUS_MAX)
        self.assertLessEqual(large.short_energy_radius, main.LAYOUT_TUNE_LARGE_SHORT_ENERGY_RADIUS_MAX)


class EncodingTests(unittest.TestCase):
    """§4.6: коды матрицей бит и коды рабочего процесса — те же, что объектами в основном процессе."""

    def test_encode_matrix_equals_encode_batch(self) -> None:
        encoder = main._build_encoder()
        rows = [(float(angle), float(x), float(y)) for angle, x, y in ((10.0, 3.0, 4.0), (200.5, 20.0, 11.0), (359.0, 0.0, 27.0))]
        bits = encoder.encode_matrix(rows)
        batch = encoder.encode_batch(rows)
        self.assertEqual(bits.shape, (3, encoder.code_length))
        for row, (_, code) in zip(bits, batch):
            self.assertEqual((row + 48).tobytes().decode("ascii"), code.to01())

    def test_worker_codes_equal_main_process_codes(self) -> None:
        encoder = main._build_encoder()
        extractor = main._build_extractor()
        rng = np.random.default_rng(5)
        image = np.zeros((28, 28), dtype=np.float32)
        image[6:22, 9:12] = 1.0
        image[6:9, 9:20] = 1.0
        image += rng.random((28, 28), dtype=np.float32) * 0.01
        direct = main.StimulusCodeCache.encode_digit(encoder, extractor, image, 7)
        # Кодер и извлечение рабочего процесса заданы заранее: журнал этого процесса не выключается.
        main.DigitEncodingWorker._encoder = main._build_encoder()
        main.DigitEncodingWorker._extractor = main._build_extractor()
        try:
            label, angles, packed = main.DigitEncodingWorker.encode([(image, 7)])[0]
        finally:
            main.DigitEncodingWorker._encoder = None
            main.DigitEncodingWorker._extractor = None
        self.assertGreater(direct.count, 0)
        self.assertEqual((label, angles), (direct.label, direct.angles))
        self.assertTrue(np.array_equal(packed, direct.packed))
        self.assertEqual(len(direct.codes), direct.count)


if __name__ == "__main__":
    unittest.main()
