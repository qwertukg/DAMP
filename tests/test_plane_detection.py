"""Детекторы-круги двумерной позиции (§4.3) и коды следующего уровня над раскладкой (§6.2, §6.5)."""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("RERUN", "off")

import numpy as np  # noqa: E402

from damp.activation.plane_detection import (  # noqa: E402
    BestMatchActivation,
    PlaneDetection,
    PlaneDetectorSpace,
    StimulusSetCodes,
)
from damp.encoding.bitarray import BitArray  # noqa: E402
from damp.encoding.colour_merge import ColourMerge  # noqa: E402
from damp.encoding.damp_encoder import (  # noqa: E402
    DetectorCircle,
    Detectors,
    Encoder,
    OpenedDimension,
    PlaneDimension,
    geometric_detector_layers,
    plane_detector_layers,
)
from damp.layout.code_table import LayoutCodeTable  # noqa: E402
from damp.layout.damp_layout import Layout  # noqa: E402
from damp.logging import LOGGER  # noqa: E402
import main  # noqa: E402

LOGGER.configure_intervals({}, default_interval=10**9)


def _plane_encoder(height: float = 40.0, width: float = 40.0, *, code_length: int = 600, seed: int = 0) -> Encoder:
    layers = plane_detector_layers(layers_count=5, widest_radius=0.85, narrowest_radius=0.05, overlap=0.7)
    return Encoder(
        PlaneDimension("Map", ((0.0, height), (0.0, width)), layers, code_length=code_length),
        random_seed=seed,
    )


class PlaneDimensionTests(unittest.TestCase):
    def test_layers_follow_radius_range_and_grow_geometrically(self) -> None:
        layers = plane_detector_layers(layers_count=5, widest_radius=0.85, narrowest_radius=0.05, overlap=0.7)
        counts = [layer.count for layer in layers]
        # r = side / (2 · count · (1 − overlap)): 0.85 стороны — 2 круга по стороне, 0.05 — 33.
        self.assertEqual(counts[0], 2)
        self.assertEqual(counts[-1], 33)
        self.assertEqual(counts, sorted(set(counts)))
        self.assertTrue(all(layer.overlap == 0.7 for layer in layers))

    def test_circles_fill_the_plane_like_segments_fill_a_line(self) -> None:
        encoder = _plane_encoder()
        plane = encoder.dimensions[0]
        for layer_index, layer in enumerate(plane.detector_layers):
            step, radius, rows, cols = plane.layer_geometry(layer)
            circles = [w for w in encoder.detectors if w.layer_index == layer_index]
            self.assertEqual(len(circles), rows * cols)
            self.assertTrue(all(isinstance(circle, DetectorCircle) for circle in circles))
            # Тот же закон, что у одномерного детектора (§4.2): диаметр = шаг / (1 − перекрытие).
            self.assertAlmostEqual(2.0 * radius, step / (1.0 - layer.overlap))
            self.assertAlmostEqual(circles[0].center_y, 0.5 * step)
            self.assertAlmostEqual(circles[0].center_x, 0.5 * step)
            # Бит детектора лежит в цветовой полосе своего слоя (§3.2).
            start, length = plane.layer_band(layer_index)
            self.assertTrue(all(start <= circle.bit < start + length for circle in circles))

    def test_active_circles_match_exhaustive_check(self) -> None:
        encoder = _plane_encoder(40.0, 28.0)
        index = encoder._indices[0]
        rng = np.random.default_rng(3)
        positions = np.column_stack([rng.uniform(0.0, 40.0, 500), rng.uniform(0.0, 28.0, 500)])
        positions = np.vstack([positions, [[0.0, 0.0], [40.0, 28.0], [0.0, 28.0], [20.0, 14.0]]])
        rows, detectors = index.active_pairs(positions)
        expected_rows, expected_detectors = np.nonzero(index.active_mask(positions))
        self.assertEqual(
            sorted(zip(rows.tolist(), detectors.tolist())),
            sorted(zip(expected_rows.tolist(), expected_detectors.tolist())),
        )
        for row in (0, 250, 503):
            active = {int(d) for r, d in zip(rows.tolist(), detectors.tolist()) if r == row}
            by_circle = {
                position
                for position, circle in enumerate(encoder.detectors)
                if circle.is_active(positions[row])
            }
            self.assertEqual(active, by_circle)

    def test_every_position_is_covered_in_every_layer(self) -> None:
        encoder = _plane_encoder()
        layers = encoder.detector_layer_indices()
        index = encoder._indices[0]
        grid = np.array([[y + 0.5, x + 0.5] for y in range(40) for x in range(40)], dtype=np.float64)
        rows, detectors = index.active_pairs(grid)
        covered = np.zeros((grid.shape[0], len(encoder.dimensions[0].detector_layers)), dtype=bool)
        covered[rows, layers[detectors]] = True
        self.assertTrue(covered.all())

    def test_code_similarity_decreases_with_distance_on_the_plane(self) -> None:
        encoder = _plane_encoder(code_length=2000)
        profile, ordered = encoder.long_range_order("Map")
        self.assertTrue(ordered)
        self.assertGreater(profile[0][1], profile[len(profile) // 2][1])
        self.assertGreater(profile[-1][1], 0.0)

    def test_set_code_is_or_of_position_codes(self) -> None:
        encoder = _plane_encoder()
        rng = np.random.default_rng(5)
        positions = rng.uniform(0.0, 40.0, size=(37, 2))
        matrix = encoder.encode_matrix(positions)
        bits, active = encoder.encode_set(positions)
        np.testing.assert_array_equal(bits, matrix.max(axis=0))
        index = encoder._indices[0]
        _, detectors = index.active_pairs(positions)
        np.testing.assert_array_equal(active, np.unique(detectors))

    def test_position_outside_the_plane_is_rejected(self) -> None:
        encoder = _plane_encoder()
        with self.assertRaises(ValueError):
            encoder.encode_matrix([(41.0, 3.0)])

    def test_plane_dimension_combines_with_line_dimensions(self) -> None:
        line_layers = geometric_detector_layers(layers_count=3, widest_detectors=2, narrowest_detectors=8, overlap=0.5)
        plane_layers = (Detectors(2, 0.7), Detectors(6, 0.7))
        encoder = Encoder(
            OpenedDimension("T", (0.0, 10.0), line_layers, code_length=60, edge_detectors=False),
            PlaneDimension("Map", ((0.0, 20.0), (0.0, 20.0)), plane_layers, code_length=120),
            random_seed=1,
        )
        self.assertEqual(encoder.code_length, 180)
        _, code = encoder.encode(4.0, 7.5, 12.5)
        line_bits = set(encoder.dimension_bits("T", 4.0))
        plane_bits = set(encoder.dimension_bits("Map", (7.5, 12.5)))
        self.assertEqual(set(code.active_indices()), line_bits | plane_bits)
        self.assertTrue(all(bit < 60 for bit in line_bits))
        self.assertTrue(all(bit >= 60 for bit in plane_bits))

    def test_line_encoder_signature_is_unchanged(self) -> None:
        # Отпечаток кодера MNIST входит в кеш раскладки и кодов: двумерная координата его не меняет.
        self.assertEqual(
            main._build_encoder().signature,
            "b8a53e99d91a8fb5bd56ba817513fd8b544bd2d5ef17b0d701815d9fe3832d44",
        )


class ColourMergeTests(unittest.TestCase):
    def test_merge_is_plain_or_within_budget(self) -> None:
        bits = np.zeros(32, dtype=np.uint8)
        bits[[1, 5, 30]] = 1
        merged, dropped = ColourMerge(3).merge(bits)
        np.testing.assert_array_equal(merged, bits)
        self.assertEqual(dropped, 0)

    def test_merge_keeps_one_end_of_the_spectrum(self) -> None:
        bits = np.zeros(32, dtype=np.uint8)
        bits[[1, 5, 9, 20, 30]] = 1
        short, dropped = ColourMerge(2, keep="short").merge(bits)
        self.assertEqual(np.flatnonzero(short).tolist(), [20, 30])
        self.assertEqual(dropped, 3)
        long, _ = ColourMerge(2, keep="long").merge(bits)
        self.assertEqual(np.flatnonzero(long).tolist(), [1, 5])


def _random_codes(count: int, length: int, ones: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bits = np.zeros((count, length), dtype=np.uint8)
    for row in range(count):
        bits[row, rng.choice(length, size=ones, replace=False)] = 1
    return bits


class BestMatchActivationTests(unittest.TestCase):
    def setUp(self) -> None:
        bits = _random_codes(40, 96, 12, seed=11)
        bits[7] = bits[3]  # две точки раскладки с одним кодом
        self.bits = bits
        self.table = LayoutCodeTable.from_bits(bits, hues=[0.0] * 40, labels=["a"] * 40)

    def _activation(self, lambda_threshold: float = 0.5, use_gpu: bool = False) -> BestMatchActivation:
        return BestMatchActivation(
            self.table, similarity="cosine", lambda_threshold=lambda_threshold, use_gpu=use_gpu
        )

    def test_known_code_lights_all_points_with_that_code(self) -> None:
        match = self._activation().lit_points(LayoutCodeTable.pack_bits(self.bits[[3, 20]]))
        self.assertEqual(match.points.tolist(), [3, 7, 20])
        self.assertEqual((match.exact, match.nearest, match.silent), (2, 0, 0))

    def test_new_code_lights_the_most_similar_points(self) -> None:
        stimulus = self.bits[15].copy()
        active = np.flatnonzero(stimulus)
        stimulus[active[0]] = 0  # 11 общих бит из 12: сходство с точкой 15 наибольшее
        match = self._activation().lit_points(LayoutCodeTable.pack_bits(stimulus[None, :]))
        self.assertEqual(match.points.tolist(), [15])
        self.assertEqual((match.exact, match.nearest, match.silent), (0, 1, 0))
        self.assertAlmostEqual(match.weakest, 11 / np.sqrt(11 * 12), places=5)

    def test_stimulus_below_threshold_lights_nothing(self) -> None:
        stimulus = np.zeros(96, dtype=np.uint8)
        stimulus[np.flatnonzero(self.bits[15])[:2]] = 1
        stimulus[np.flatnonzero(self.bits.sum(axis=0) == 0)[:10]] = 1
        match = self._activation(lambda_threshold=0.9).lit_points(LayoutCodeTable.pack_bits(stimulus[None, :]))
        self.assertEqual(match.points.size, 0)
        self.assertEqual(match.silent, 1)

    def test_device_and_host_backends_agree(self) -> None:
        stimuli = _random_codes(9, 96, 12, seed=23)
        host = self._activation(lambda_threshold=0.0, use_gpu=False).lit_points(LayoutCodeTable.pack_bits(stimuli))
        device = self._activation(lambda_threshold=0.0, use_gpu=True).lit_points(LayoutCodeTable.pack_bits(stimuli))
        self.assertEqual(host.points.tolist(), device.points.tolist())

    def test_bitarray_stimuli_are_packed_like_the_table(self) -> None:
        activation = self._activation()
        codes = [BitArray.from_indices(96, np.flatnonzero(self.bits[row]).tolist()) for row in (3, 20)]
        np.testing.assert_array_equal(activation.pack(codes), LayoutCodeTable.pack_bits(self.bits[[3, 20]]))


class PlaneDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bits = _random_codes(60, 96, 12, seed=31)
        table = LayoutCodeTable.from_bits(self.bits, hues=[0.0] * 60, labels=["a"] * 60)
        self.layout = Layout(table, empty_ratio=0.5, similarity="cosine", use_gpu=False, seed=0)
        self.encoder = Encoder(
            PlaneDimension(
                "Map",
                PlaneDetectorSpace.plane_size(self.layout),
                plane_detector_layers(layers_count=4, widest_radius=0.85, narrowest_radius=0.08, overlap=0.7),
                code_length=400,
            ),
            random_seed=0,
        )
        self.space = PlaneDetectorSpace(self.layout, self.encoder)
        self.detection = PlaneDetection(
            BestMatchActivation(self.layout.code_table, similarity="cosine", lambda_threshold=0.5, use_gpu=False),
            self.space,
        )

    def test_code_of_one_point_is_the_code_of_its_cell(self) -> None:
        cells = self.layout.cells_array()
        code = self.space.detect(np.array([17]))
        expected = self.encoder.encode_matrix([(cells[17, 0] + 0.5, cells[17, 1] + 0.5)])[0]
        np.testing.assert_array_equal(code.bits, expected)

    def test_code_of_a_stimulus_set_unites_circles_of_all_lit_points(self) -> None:
        table = self.layout.code_table
        rows = [4, 9, 33]
        result = self.detection.encode_packed(table.packed[rows])
        self.assertEqual(result.lit_points.tolist(), rows)
        united = np.zeros(self.encoder.code_length, dtype=np.uint8)
        for row in rows:
            united |= self.space.detect(np.array([row])).bits
        np.testing.assert_array_equal(result.bits, united)
        self.assertEqual(result.embedding.count(), int(united.sum()))

    def test_near_points_share_more_circles_than_far_points(self) -> None:
        cells = self.layout.cells_array().astype(np.float64)
        distances = np.hypot(cells[:, 0] - cells[0, 0], cells[:, 1] - cells[0, 1])
        order = np.argsort(distances)
        near, far = int(order[1]), int(order[-1])
        base = self.space.detect(np.array([0])).bits.astype(np.int64)
        near_common = int((base & self.space.detect(np.array([near])).bits).sum())
        far_common = int((base & self.space.detect(np.array([far])).bits).sum())
        self.assertGreater(near_common, far_common)

    def test_colour_merge_limits_saturation(self) -> None:
        space = PlaneDetectorSpace(self.layout, self.encoder, merge=ColourMerge(20, keep="short"))
        code = space.detect(np.arange(60))
        self.assertEqual(int(code.bits.sum()), 20)
        self.assertGreater(code.dropped_bits, 0)

    def test_stimulus_set_codes_become_a_layout_table(self) -> None:
        table = self.layout.code_table
        codes = StimulusSetCodes(self.detection.code_length)
        for label, rows in ((3, [0, 1, 2]), (7, [30, 31]), (3, [5, 6])):
            codes.add(label, self.detection.encode_packed(table.packed[rows], label=label))
        second = codes.table(hue_step=36.0)
        self.assertEqual(second.count, 3)
        self.assertEqual(second.labels(), ["3", "7", "3"])
        self.assertEqual(second.hues.tolist(), [108.0, 252.0, 108.0])
        np.testing.assert_array_equal(second.bits(), codes.bits())
        self.assertEqual([code.count() for code in codes.embeddings()], second.ones.tolist())
        stats = codes.log_summary(self.space, similarity="cosine")
        self.assertEqual(stats["sample"], 3)


if __name__ == "__main__":
    unittest.main()
