"""Второй уровень, вариант «код на углопозицию»: код места точки (§4.3, §6.2) и сравнение двух карт (§5.8, §5.13.2)."""
from __future__ import annotations

import os
import tempfile
import types
import unittest

os.environ.setdefault("RERUN", "off")

import numpy as np  # noqa: E402

from damp.activation.plane_detection import BestMatchActivation, PlaneDetection, PlaneDetectorSpace  # noqa: E402
from damp.activation.point_codes import LayoutPointCodes, ThresholdActivation  # noqa: E402
from damp.encoding.damp_encoder import Encoder, PlaneDimension, plane_detector_layers  # noqa: E402
from damp.layout.code_table import LayoutCodeTable  # noqa: E402
from damp.layout.damp_layout import Layout  # noqa: E402
from damp.layout.map_comparison import StimulusMapMeter  # noqa: E402
from damp.logging import LOGGER  # noqa: E402
import main  # noqa: E402

LOGGER.configure_intervals({}, default_interval=10**9)


def _random_codes(count: int, length: int, ones: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bits = np.zeros((count, length), dtype=np.uint8)
    for row in range(count):
        bits[row, rng.choice(length, size=ones, replace=False)] = 1
    return bits


def _cosine(bits: np.ndarray, row: int) -> np.ndarray:
    values = bits.astype(np.float64)
    ones = values.sum(axis=1)
    return (values @ values[row]) / np.sqrt(ones * ones[row])


class ThresholdActivationTests(unittest.TestCase):
    def setUp(self) -> None:
        bits = _random_codes(40, 96, 24, seed=11)
        bits[7] = bits[3]  # две точки раскладки с одним кодом
        self.bits = bits
        self.table = LayoutCodeTable.from_bits(bits, hues=[0.0] * 40, labels=["a"] * 40)

    def _activation(self, lambda_threshold: float, use_gpu: bool = False) -> ThresholdActivation:
        return ThresholdActivation(
            self.table, similarity="cosine", lambda_threshold=lambda_threshold, use_gpu=use_gpu
        )

    def test_stimulus_lights_every_point_at_or_above_the_threshold(self) -> None:
        similarity = _cosine(self.bits, 15)
        threshold = float(np.sort(similarity)[-6])  # шесть самых похожих точек, считая саму точку 15
        match = self._activation(threshold).lit_points(self.table.packed[15:16])
        self.assertEqual(match.points.tolist(), np.flatnonzero(similarity >= threshold - 1e-9).tolist())
        self.assertGreater(match.points.size, 1)
        self.assertEqual((match.exact, match.silent), (1, 0))

    def test_threshold_one_lights_only_points_with_the_same_code(self) -> None:
        match = self._activation(1.0).lit_points(self.table.packed[3:4])
        self.assertEqual(match.points.tolist(), [3, 7])

    def test_set_of_stimuli_lights_the_union(self) -> None:
        activation = self._activation(0.45)
        united = activation.lit_points(self.table.packed[[2, 21]]).points.tolist()
        separate = sorted(
            set(activation.lit_points(self.table.packed[2:3]).points.tolist())
            | set(activation.lit_points(self.table.packed[21:22]).points.tolist())
        )
        self.assertEqual(united, separate)

    def test_stimulus_below_threshold_lights_nothing(self) -> None:
        stimulus = np.zeros(96, dtype=np.uint8)
        stimulus[np.flatnonzero(self.bits[15])[:2]] = 1
        match = self._activation(0.9).lit_points(LayoutCodeTable.pack_bits(stimulus[None, :]))
        self.assertEqual(match.points.size, 0)
        self.assertEqual(match.silent, 1)

    def test_device_and_host_agree(self) -> None:
        stimuli = LayoutCodeTable.pack_bits(_random_codes(9, 96, 24, seed=23))
        host = self._activation(0.3, use_gpu=False).lit_points(stimuli)
        device = self._activation(0.3, use_gpu=True).lit_points(stimuli)
        self.assertEqual(host.points.tolist(), device.points.tolist())

    def test_zero_threshold_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._activation(0.0)


class LayoutPointCodesTests(unittest.TestCase):
    def setUp(self) -> None:
        bits = _random_codes(60, 96, 12, seed=31)
        bits[9] = bits[4]  # код встречается на карте дважды
        self.bits = bits
        hues = [float(row * 6 % 360) for row in range(60)]
        labels = [str(row % 10) for row in range(60)]
        table = LayoutCodeTable.from_bits(bits, hues=hues, labels=labels)
        self.layout = Layout(table, empty_ratio=0.5, similarity="cosine", use_gpu=False, seed=0)
        self.encoder = Encoder(
            PlaneDimension(
                "Map",
                PlaneDetectorSpace.plane_size(self.layout),
                plane_detector_layers(layers_count=4, widest_radius=0.85, narrowest_radius=0.12, overlap=0.7),
                code_length=400,
            ),
            random_seed=0,
        )
        self.detection = PlaneDetection(
            BestMatchActivation(self.layout.code_table, similarity="cosine", lambda_threshold=0.5, use_gpu=False),
            PlaneDetectorSpace(self.layout, self.encoder),
        )
        self.codes = LayoutPointCodes(self.detection, self.layout.code_table)

    def _cell_code(self, point: int) -> np.ndarray:
        cells = self.layout.cells_array()
        return self.encoder.encode_matrix([(cells[point, 0] + 0.5, cells[point, 1] + 0.5)])[0]

    def test_point_with_a_unique_code_gets_the_code_of_its_cell(self) -> None:
        bits = self.codes.table().bits()
        for point in (0, 17, 42, 59):
            np.testing.assert_array_equal(bits[point], self._cell_code(point))
        self.assertEqual(int(self.codes.lit_points[17]), 1)

    def test_points_with_one_code_get_the_union_of_their_places(self) -> None:
        bits = self.codes.table().bits()
        united = self._cell_code(4) | self._cell_code(9)
        np.testing.assert_array_equal(bits[4], united)
        np.testing.assert_array_equal(bits[9], united)
        self.assertEqual(self.codes.lit_points[[4, 9]].tolist(), [2, 2])

    def test_table_keeps_the_points_of_the_first_layout(self) -> None:
        source = self.layout.code_table
        table = self.codes.table()
        self.assertEqual(table.count, source.count)
        self.assertEqual(table.code_length, self.encoder.code_length)
        np.testing.assert_array_equal(table.hues, source.hues)
        self.assertEqual(table.labels(), source.labels())
        np.testing.assert_array_equal(table.ones, table.bits().sum(axis=1))

    def test_codes_carry_the_topology_of_the_map(self) -> None:
        summary = self.codes.log_summary(
            self.layout.cells_array(), similarity="cosine", lambda_threshold=0.65, seed=0
        )
        self.assertEqual(summary.codes, 60)
        self.assertEqual(summary.source_unique_codes, 59)
        self.assertEqual(summary.unique_codes, 59)
        self.assertEqual(summary.lit_points_max, 2)
        self.assertTrue(0.0 < summary.density_mean <= 0.25)
        self.assertGreater(summary.profile[0][1], summary.profile[-1][1])

    def test_similarity_of_place_codes_decreases_with_distance_on_the_map(self) -> None:
        # Карта 30 × 30 из 600 точек: профиль сходства кодов мест убывает на всём диапазоне расстояний (§4.5).
        bits = _random_codes(600, 128, 16, seed=61)
        table = LayoutCodeTable.from_bits(bits, hues=[0.0] * 600, labels=["a"] * 600)
        layout = Layout(
            table, empty_ratio=0.5, similarity="cosine", use_gpu=False, seed=0, precompute_similarity=False
        )
        encoder = Encoder(
            PlaneDimension(
                "Map",
                PlaneDetectorSpace.plane_size(layout),
                plane_detector_layers(layers_count=6, widest_radius=0.85, narrowest_radius=0.06, overlap=0.7),
                code_length=1200,
            ),
            random_seed=0,
        )
        detection = PlaneDetection(
            BestMatchActivation(layout.code_table, similarity="cosine", lambda_threshold=0.5, use_gpu=False),
            PlaneDetectorSpace(layout, encoder),
        )
        summary = LayoutPointCodes(detection, layout.code_table).log_summary(
            layout.cells_array(), similarity="cosine", lambda_threshold=0.65, seed=0
        )
        self.assertTrue(summary.ordered)
        self.assertGreater(summary.profile[0][1], 0.65)
        self.assertLess(summary.profile[-1][1], 0.3)
        self.assertTrue(0.0 < summary.pairs_above_lambda < 0.2)

    def test_threshold_rule_unites_places_of_similar_codes(self) -> None:
        table = self.layout.code_table
        similarity = _cosine(self.bits, 20)
        threshold = float(np.sort(similarity)[-4])
        detection = PlaneDetection(
            ThresholdActivation(table, similarity="cosine", lambda_threshold=threshold, use_gpu=False),
            PlaneDetectorSpace(self.layout, self.encoder),
        )
        codes = LayoutPointCodes(detection, table)
        united = np.zeros(self.encoder.code_length, dtype=np.uint8)
        for point in np.flatnonzero(similarity >= threshold - 1e-9).tolist():
            united |= self._cell_code(point)
        np.testing.assert_array_equal(codes.table().bits()[20], united)
        self.assertGreaterEqual(int(codes.lit_points[20]), 4)


class StimulusMapMeterTests(unittest.TestCase):
    def setUp(self) -> None:
        bits = _random_codes(80, 96, 12, seed=41)
        rng = np.random.default_rng(7)
        hues = rng.integers(0, 360, size=80).astype(np.float64)
        table = LayoutCodeTable.from_bits(bits, hues=hues, labels=["a"] * 80).grouped_by_hue()
        self.layout = Layout(table, empty_ratio=0.2, similarity="cosine", use_gpu=False, seed=0)
        self.table = self.layout.code_table
        self.shape = (self.layout.height, self.layout.width)

    def _meter(self, table: LayoutCodeTable | None = None, **kwargs) -> StimulusMapMeter:  # noqa: ANN003
        return StimulusMapMeter(
            self.table if table is None else table,
            angle_period=360.0,
            angle_jump=30.0,
            similarity="cosine",
            similarity_floor=0.65,
            **kwargs,
        )

    def test_angle_metrics_match_pinwheel_quality(self) -> None:
        from pinwheel_test import PinwheelQualityEvaluator

        expected = PinwheelQualityEvaluator(period=360.0, smooth_delta=30.0).evaluate(
            self.layout, self.table.codes()
        )
        measured = self._meter().continuity(self.layout.cells_array(), self.shape, name="test")
        self.assertAlmostEqual(measured.mean_angle_delta, expected.mean_neighbour_delta, places=9)
        self.assertAlmostEqual(measured.angle_jump_ratio, expected.sharp_transition_ratio, places=12)
        self.assertEqual(measured.singularities, expected.singularities)
        # Без позиций стимула стенка — скачок угла.
        self.assertEqual(measured.wall_ratio, measured.angle_jump_ratio)
        self.assertIsNone(measured.position_jump_ratio)

    def test_wall_between_two_regions_is_counted_by_angle_and_by_position(self) -> None:
        # Карта 4 × 6 без пустых ячеек: слева угол 10°, справа 100°; окно стимула меняется по строкам.
        height, width = 4, 6
        cells = np.array([(y, x) for y in range(height) for x in range(width)], dtype=np.int64)
        hues = np.where(cells[:, 1] < 3, 10.0, 100.0)
        positions = np.column_stack([np.zeros(height * width), np.where(cells[:, 0] < 2, 0.0, 8.0)])
        bits = _random_codes(height * width, 96, 12, seed=3)
        table = LayoutCodeTable.from_bits(bits, hues=hues, labels=["a"] * (height * width))
        meter = self._meter(table, positions=positions, position_jump=3.0)
        measured = meter.continuity(cells, (height, width), name="regions")
        pairs = height * (width - 1) + (height - 1) * width
        self.assertEqual(measured.neighbour_pairs, pairs)
        self.assertAlmostEqual(measured.angle_jump_ratio, height / pairs)  # вертикальная граница
        self.assertAlmostEqual(measured.position_jump_ratio, width / pairs)  # горизонтальная граница
        self.assertAlmostEqual(measured.wall_ratio, (height + width) / pairs)
        self.assertEqual(measured.double_jump_ratio, 0.0)
        self.assertAlmostEqual(measured.mean_angle_delta, 90.0 * height / pairs)

    def test_neighbour_similarity_uses_the_codes_of_the_points(self) -> None:
        cells = np.array([(0, 0), (0, 1), (0, 2)], dtype=np.int64)
        bits = np.zeros((3, 16), dtype=np.uint8)
        bits[0, :8] = 1
        bits[1, :8] = 1  # тот же код, что у точки 0
        bits[2, 8:] = 1  # ни одного общего бита с точкой 1
        table = LayoutCodeTable.from_bits(bits, hues=[0.0, 0.0, 0.0], labels=["a"] * 3)
        measured = self._meter(table).continuity(cells, (1, 3), name="codes")
        self.assertAlmostEqual(measured.mean_similarity, 0.5)
        self.assertAlmostEqual(measured.low_similarity_ratio, 0.5)

    def test_same_map_and_mirrored_map_correspond_exactly(self) -> None:
        cells = self.layout.cells_array()
        meter = self._meter()
        same = meter.correspondence(cells, cells, self.shape)
        self.assertEqual(same.kept_neighbours_ratio, 1.0)
        self.assertEqual(same.stayed_neighbours_ratio, 1.0)
        self.assertEqual(same.new_neighbours_ratio, 0.0)
        self.assertAlmostEqual(same.fit_explained, 1.0)
        self.assertAlmostEqual(same.fit_residual, 0.0, places=6)
        self.assertFalse(same.fit_reflection)
        mirrored = cells.copy()
        mirrored[:, 1] = self.shape[1] - 1 - mirrored[:, 1]
        mirror = meter.correspondence(cells, mirrored, self.shape)
        self.assertEqual(mirror.kept_neighbours_ratio, 1.0)
        self.assertAlmostEqual(mirror.fit_explained, 1.0)
        self.assertTrue(mirror.fit_reflection)
        self.assertAlmostEqual(mirror.fit_scale, 1.0)

    def test_shuffled_map_does_not_correspond(self) -> None:
        cells = self.layout.cells_array()
        shuffled = cells[np.random.default_rng(5).permutation(cells.shape[0])]
        result = self._meter().correspondence(cells, shuffled, self.shape)
        self.assertLess(result.kept_neighbours_ratio, 0.5)
        self.assertLess(result.fit_explained, 0.5)

    def test_energy_is_the_layout_quality_with_its_scale(self) -> None:
        measured = self._meter().energy(self.layout, 3, name="test")
        self.assertAlmostEqual(measured.average, self.layout.average_local_energy(3), places=12)
        self.assertGreater(measured.maximum, 0.0)
        # Ē = среднее E / E_max (§5.8).
        self.assertAlmostEqual(measured.mean / measured.maximum, measured.average, places=9)

    def test_source_position_image_paints_the_points_of_the_map(self) -> None:
        cells = self.layout.cells_array()
        image = self._meter().source_position_image(cells, cells, self.shape, name="test")
        self.assertEqual(image.shape, (self.shape[0], self.shape[1], 3))
        occupied = image.any(axis=2)
        self.assertEqual(int(occupied.sum()), cells.shape[0])
        self.assertTrue(occupied[cells[:, 0], cells[:, 1]].all())


class PointPlaneBuildTests(unittest.TestCase):
    """Константы SECOND_LEVEL_POINT_* (main.py): код одной точки по плотности и дальнему порядку (§2, §4.5)."""

    def test_place_code_has_target_density_and_long_range_order_on_maps_of_the_experiment(self) -> None:
        for side in (115, 199):  # карты 100 и 300 цифр MNIST
            encoder = main._build_point_plane_encoder(types.SimpleNamespace(height=side, width=side))
            self.assertEqual(encoder.code_length, main.SECOND_LEVEL_POINT_CODE_LENGTH)
            _, ordered = encoder.long_range_order("Map")
            self.assertTrue(ordered, f"нет дальнего порядка на карте {side}")
            # Длина кода подобрана под плотность 25 % (§2): код места внутри карты — около четверти бит.
            places = [(side * fy + 0.5, side * fx + 0.5) for fy, fx in ((0.5, 0.5), (0.3, 0.6), (0.7, 0.35))]
            density = float(encoder.encode_matrix(places).sum(axis=1).mean()) / encoder.code_length
            self.assertGreater(density, 0.22)
            self.assertLess(density, 0.30)

    def test_threshold_rule_has_its_own_code_with_long_range_order(self) -> None:
        for side in (115, 199):
            plane = types.SimpleNamespace(height=side, width=side)
            encoder = main._build_point_plane_encoder(plane, "threshold")
            self.assertEqual(encoder.code_length, main.SECOND_LEVEL_POINT_THRESHOLD_CODE_LENGTH)
            self.assertEqual(len(encoder.dimensions[0].detector_layers), main.SECOND_LEVEL_POINT_THRESHOLD_LAYERS)
            _, ordered = encoder.long_range_order("Map")
            self.assertTrue(ordered, f"нет дальнего порядка на карте {side}")
        with self.assertRaises(ValueError):
            main._build_point_plane_encoder(types.SimpleNamespace(height=115, width=115), "all")

    def test_narrowest_circle_is_set_in_cells_of_the_map(self) -> None:
        for side in (115, 199):
            encoder = main._build_point_plane_encoder(types.SimpleNamespace(height=side, width=side))
            plane = encoder.dimensions[0]
            _, radius, _, _ = plane.layer_geometry(plane.detector_layers[-1])
            self.assertAlmostEqual(radius, main.SECOND_LEVEL_POINT_RADIUS_MIN_CELLS, delta=0.02)

    def test_detection_follows_the_activation_rule(self) -> None:
        bits = _random_codes(60, 96, 12, seed=31)
        bits[9] = bits[4]  # код встречается на карте дважды
        table = LayoutCodeTable.from_bits(bits, hues=[0.0] * 60, labels=["a"] * 60)
        layout = Layout(table, empty_ratio=0.5, similarity="cosine", use_gpu=False, seed=0)
        self.assertEqual(main.SECOND_LEVEL_POINT_ACTIVATION, "best")  # правило, выбранное для второго уровня
        best = main._build_point_plane_detection(layout)
        self.assertIsInstance(best._activation, BestMatchActivation)
        self.assertEqual(best.code_length, main.SECOND_LEVEL_POINT_CODE_LENGTH)
        # Правило §6.5: стимул точки зажигает набор точек, у кода набора своя длина.
        threshold = main._build_point_plane_detection(layout, "threshold")
        self.assertIsInstance(threshold._activation, ThresholdActivation)
        self.assertEqual(threshold.code_length, main.SECOND_LEVEL_POINT_THRESHOLD_CODE_LENGTH)
        with self.assertRaises(ValueError):
            main._build_point_plane_detection(layout, "all")
        with self.assertRaises(ValueError):
            main.MnistSecondLevelPointRunner(count=1, activation="all")
        # Код точки с правилом «самые похожие» — код её ячейки в кодере варианта.
        codes = LayoutPointCodes(best, layout.code_table)
        cells = layout.cells_array()
        expected = best.space.encoder.encode_matrix([(cells[5, 0] + 0.5, cells[5, 1] + 0.5)])[0]
        np.testing.assert_array_equal(codes.table().bits()[5], expected)
        # Код, стоящий в двух местах, — OR кодов обоих мест: биты не отбираются и не отбрасываются.
        both = best.space.encoder.encode_matrix(
            [(cells[4, 0] + 0.5, cells[4, 1] + 0.5), (cells[9, 0] + 0.5, cells[9, 1] + 0.5)]
        )
        np.testing.assert_array_equal(codes.table().bits()[4], both[0] | both[1])
        np.testing.assert_array_equal(codes.table().bits()[9], both[0] | both[1])


class SecondLevelPointComparisonTests(unittest.TestCase):
    """Сравнение карт прогона: вторая расстановка тех же точек против первой (§5.8, §5.13.2)."""

    def test_comparison_of_a_layout_with_itself_changes_nothing(self) -> None:
        bits = _random_codes(120, 96, 12, seed=51)
        rng = np.random.default_rng(9)
        hues = rng.integers(0, 360, size=120).astype(np.float64)
        table = LayoutCodeTable.from_bits(bits, hues=hues, labels=["a"] * 120).grouped_by_hue()
        first = Layout(table, empty_ratio=0.5, similarity="cosine", use_gpu=False, seed=0)
        detection = main._build_point_plane_detection(first)
        codes = LayoutPointCodes(detection, first.code_table)
        summary = codes.log_summary(
            first.cells_array(), similarity="cosine", lambda_threshold=main.LAYOUT_LAMBDA_THRESHOLD
        )
        # Вторая раскладка не запускается: коды мест ставятся в те же ячейки, что точки первой.
        second = Layout(
            codes.table(),
            empty_ratio=0.5,
            similarity="cosine",
            use_gpu=False,
            seed=0,
            initial_cells=first.cells_array(),
        )
        runner = main.MnistSecondLevelPointRunner.__new__(main.MnistSecondLevelPointRunner)
        runner.count = 0
        runner._activation = "best"
        runner._primary = types.SimpleNamespace(colorizer=main._build_colorizer())
        runner._params = main.LayoutParameterTuner().tune(table.count)
        windows = np.column_stack([rng.integers(0, 24, size=120), rng.integers(0, 24, size=120)]).astype(np.float64)
        runner._stimulus_positions = lambda _table: windows
        with tempfile.TemporaryDirectory() as directory:
            report = runner._compare(first, second, summary, os.path.join(directory, "second.png"))
            self.assertTrue(os.path.exists(os.path.join(directory, "second.source.png")))
            self.assertTrue(os.path.exists(os.path.join(directory, "second.source-first.png")))
        self.assertEqual(report.first, report.second)
        self.assertAlmostEqual(report.first_energy.average, report.second_energy.average, places=12)
        self.assertAlmostEqual(report.first_energy.mean, report.second_energy.mean, places=9)
        self.assertEqual(report.correspondence.kept_neighbours_ratio, 1.0)
        self.assertAlmostEqual(report.correspondence.fit_explained, 1.0)
        self.assertIsNotNone(report.first.position_jump_ratio)


if __name__ == "__main__":
    unittest.main()
