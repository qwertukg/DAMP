"""Лестница уровней в локальных координатах: окна внимания, коды окон и взглядов, детектирование наборов, эмбеддинги."""

from __future__ import annotations

import os
import sys
import unittest


os.environ.setdefault("RERUN", "off")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "system"))

import numpy as np  # noqa: E402

from damp.activation import ActivationEngine, Detector  # noqa: E402
from damp.activation.hierarchy_detection import HierarchyDetection  # noqa: E402
from damp.activation.set_embedding import DetectorEmbeddings  # noqa: E402
from damp.decode.nearest_codes import NearestVectorClassifier  # noqa: E402
from damp.encoding.bitarray import BitArray  # noqa: E402
from damp.encoding.damp_encoder import Encoder, PlaneDimension, plane_detector_layers  # noqa: E402
from damp.encoding.position_binding import PositionBindingEncoder  # noqa: E402
from tests.test_article_invariants import _clustered_layout  # noqa: E402


def _place_encoder() -> Encoder:
    return Encoder(
        PlaneDimension(
            "Place",
            ((-4.0, 4.0), (-4.0, 4.0)),
            plane_detector_layers(layers_count=3, widest_radius=0.25, narrowest_radius=0.09375, overlap=0.7),
            code_length=256,
            layer_bands=False,
        ),
        random_seed=0,
    )


class SetDetectionTest(unittest.TestCase):
    @staticmethod
    def _detectors() -> list[Detector]:
        bit = BitArray.from_indices(16, [0])
        return [
            Detector(center=(2.5, 2.5), radius=3.0, lambda_threshold=0.5, energy=6.0, points_count=16, output_code=bit),
            Detector(center=(8.5, 8.5), radius=3.0, lambda_threshold=0.5, energy=6.0, points_count=16, output_code=bit),
            Detector(center=(2.5, 2.5), radius=1.6, lambda_threshold=0.8, energy=2.0, points_count=4, output_code=bit),
            Detector(center=(8.5, 8.5), radius=1.6, lambda_threshold=0.8, energy=2.0, points_count=4, output_code=bit),
        ]

    @staticmethod
    def _stimuli() -> list[BitArray]:
        return [
            BitArray.from_indices(64, [0, 1, 2, 3, 4, 5, 6, 7]),
            BitArray.from_indices(64, [40, 41, 42, 43, 44, 45, 46, 47]),
            BitArray.from_indices(64, [0, 1, 2, 3, 40, 41, 42, 43]),
            BitArray.from_indices(64, [56, 57, 58, 59, 60, 61, 62, 63]),
        ]

    def test_a_set_gives_what_the_activation_engine_gives_for_its_stimuli(self) -> None:
        layout = _clustered_layout()
        detectors = self._detectors()
        energies = [1.0 if point % 5 else 0.05 for point in range(32)]
        settings = {"lambda_threshold": 0.5, "eta": None, "energy_threshold": 0.12, "detector_threshold": 0.5}
        engine = ActivationEngine(layout, detectors, saturation_limit=4, energy_map=energies, use_gpu=False, **settings)
        detection = HierarchyDetection(layout, detectors, energies, similarity="cosine", use_gpu=False, **settings)
        stimuli = self._stimuli()
        packed = np.packbits(np.asarray([list(code) for code in stimuli], dtype=np.uint8), axis=1)
        sets = [[0, 1], [2], [3], [0, 2, 3], [1, 1]]
        index = np.asarray([code for members in sets for code in members], dtype=np.int64)
        groups = np.asarray([number for number, members in enumerate(sets) for _ in members], dtype=np.int64)
        rows, found, levels = detection.active_sets(packed, index, groups, len(sets))
        for number, members in enumerate(sets):
            _, active = engine.detect([stimuli[code] for code in members])
            expected = {id(item.detector): float(item.activation_level) for item in active}
            mine = {
                id(detection.detectors[int(detector)]): float(level)
                for detector, level in zip(found[rows == number], levels[rows == number])
            }
            self.assertEqual(set(mine), set(expected), f"set {number}")
            for key, level in expected.items():
                self.assertAlmostEqual(mine[key], level, places=5)

    def test_sets_of_one_stimulus_match_the_single_code_detection(self) -> None:
        layout = _clustered_layout()
        detectors = self._detectors()
        energies = [1.0] * 32
        detection = HierarchyDetection(
            layout, detectors, energies, similarity="cosine", lambda_threshold=0.5, eta=None, energy_threshold=0.12,
            detector_threshold=0.5, use_gpu=False,
        )
        packed = np.packbits(np.asarray([list(code) for code in self._stimuli()], dtype=np.uint8), axis=1)
        single_rows, single_found = detection.active(packed)
        rows, found, _ = detection.active_sets(packed, np.arange(4), np.arange(4), 4)
        self.assertEqual(sorted(zip(rows.tolist(), found.tolist())), sorted(zip(single_rows.tolist(), single_found.tolist())))

    def test_an_empty_set_fires_nothing(self) -> None:
        layout = _clustered_layout()
        detection = HierarchyDetection(
            layout, self._detectors(), [1.0] * 32, similarity="cosine", lambda_threshold=0.5, eta=None,
            energy_threshold=0.12, detector_threshold=0.5, use_gpu=False,
        )
        packed = np.packbits(np.asarray([list(code) for code in self._stimuli()], dtype=np.uint8), axis=1)
        rows, _, _ = detection.active_sets(packed, np.asarray([0, 1]), np.asarray([0, 2]), 3)
        self.assertNotIn(1, rows.tolist())
        self.assertIn(0, rows.tolist())
        self.assertIn(2, rows.tolist())


class DetectorEmbeddingsTest(unittest.TestCase):
    @staticmethod
    def _embeddings(saturation: int = 2, density: float = 0.25) -> DetectorEmbeddings:
        return DetectorEmbeddings(
            detectors=40, saturation_limit=saturation, target_density=density, min_length=8, max_length=4096, seed=0
        )

    def test_keeps_the_strongest_detectors_of_each_set(self) -> None:
        rows = np.asarray([0, 0, 0, 1, 1, 2])
        columns = np.asarray([5, 6, 7, 1, 2, 9])
        levels = np.asarray([0.6, 0.9, 0.7, 1.5, 0.8, 0.5])
        kept_rows, kept_columns = self._embeddings(2).strongest(rows, columns, levels)
        kept = sorted(zip(kept_rows.tolist(), kept_columns.tolist()))
        self.assertEqual(kept, [(0, 6), (0, 7), (1, 1), (1, 2), (2, 9)])
        all_rows, all_columns = self._embeddings(0).strongest(rows, columns, levels)
        self.assertEqual(all_rows.shape[0], 6)

    def test_length_is_the_smallest_under_the_target_density(self) -> None:
        rng = np.random.default_rng(1)
        rows = np.repeat(np.arange(50), 10)
        columns = np.concatenate([rng.choice(40, 10, replace=False) for _ in range(50)])
        levels = np.ones(rows.shape[0])
        embeddings = self._embeddings(0, density=0.25)
        packed, length = embeddings.encode(rows, columns, levels, 50)
        self.assertEqual(packed.shape, (50, length // 8))
        density = float(np.unpackbits(packed, axis=1, count=length).mean())
        self.assertLessEqual(density, 0.25)
        self.assertGreater(embeddings._density(rows, columns, 50, length - 8), 0.25)

    def test_equal_sets_of_detectors_give_equal_codes(self) -> None:
        rows = np.asarray([0, 0, 1, 1, 3])
        columns = np.asarray([4, 9, 9, 4, 11])
        packed, length = self._embeddings(0).encode(rows, columns, np.ones(5), 4)
        self.assertTrue(np.array_equal(packed[0], packed[1]))
        self.assertFalse(np.array_equal(packed[0], packed[3]))
        self.assertEqual(int(packed[2].sum()), 0)  # набор без сработавших детекторов — пустой код
        again, same_length = self._embeddings(0).encode(rows, columns, np.ones(5), 4)
        self.assertEqual(length, same_length)
        self.assertTrue(np.array_equal(packed, again))

    def test_every_detector_has_its_own_bit_while_they_fit(self) -> None:
        bits = self._embeddings().detector_bits(64)
        self.assertEqual(bits.shape[0], 40)
        self.assertEqual(np.unique(bits).shape[0], 40)
        self.assertEqual(self._embeddings().detector_bits(16).shape[0], 40)


class NearestVectorClassifierTest(unittest.TestCase):
    def test_class_of_the_most_similar_vector(self) -> None:
        base = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        classifier = NearestVectorClassifier(base, [3, 5, 7])
        queries = np.asarray([[0.9, 0.1, 0.0], [0.0, 2.0, 0.5], [0.1, 0.0, 4.0], [0.0, 0.2, 0.1]])
        accuracy = classifier.accuracy(queries, [3, 5, 7, 7], neighbours=(1,))
        self.assertAlmostEqual(accuracy[1], 0.75)


if __name__ == "__main__":
    unittest.main()


class PlaneFieldsTest(unittest.TestCase):
    """Поля зрения по кластерам (§6.4): кластер похожих соседей, взвешенный центр, радиус, вставка."""

    @staticmethod
    def _builder(thresholds=0.9, min_points: int = 5):
        from damp.cycle.plane_fields import PlaneFieldBuilder

        return PlaneFieldBuilder(
            lambda_thresholds=thresholds, activation_radius=6.0, eps=1.5, min_samples=3, min_points=min_points,
            radius_min=1.0, energy_radius=3.0, energy_threshold=0.12,
        )

    @staticmethod
    def _codes(kinds) -> np.ndarray:
        bits = np.zeros((len(kinds), 32), dtype=np.float32)
        for row, kind in enumerate(kinds):
            bits[row, kind * 8 : kind * 8 + 8] = 1
        return bits

    def test_a_row_of_alike_neighbours_becomes_one_field(self) -> None:
        positions = np.asarray([[0.0, float(x)] for x in range(7)])
        fields = self._builder().picture_fields(positions, self._codes([0] * 7))
        self.assertEqual(len(fields), 1)
        centre_y, centre_x, radius, points = fields[0]
        self.assertAlmostEqual(centre_y, 0.0)
        self.assertAlmostEqual(centre_x, 3.0)
        # Радиус — наименьший, вмещающий не меньше пяти точек кластера: заполнение у отрезка падает с радиусом.
        self.assertAlmostEqual(radius, 2.0)
        self.assertEqual(points, 5)

    def test_unlike_neighbours_are_not_gathered(self) -> None:
        positions = np.asarray([[0.0, float(x)] for x in range(7)])
        self.assertEqual(self._builder().picture_fields(positions, self._codes([0, 1, 2, 3, 0, 1, 2])), [])

    def test_too_few_points_give_no_field(self) -> None:
        positions = np.asarray([[0.0, float(x)] for x in range(4)])
        self.assertEqual(self._builder().picture_fields(positions, self._codes([0] * 4)), [])

    def test_centres_do_not_fall_into_each_other(self) -> None:
        rng = np.random.default_rng(0)
        positions = np.asarray([[float(y), float(x)] for y in range(6) for x in range(12)])
        fields = self._builder(min_points=4).picture_fields(positions, self._codes(rng.integers(0, 2, positions.shape[0])))
        self.assertGreater(len(fields), 1)
        for first in range(len(fields)):
            for second in range(first + 1, len(fields)):
                gap = np.hypot(fields[first][0] - fields[second][0], fields[first][1] - fields[second][1])
                self.assertGreater(gap, max(fields[first][2], fields[second][2]))

    def test_fields_move_with_the_picture(self) -> None:
        """Поля привязаны к содержимому картинки, а не к кадру: сдвиг картинки сдвигает поля и не меняет куски."""
        rng = np.random.default_rng(1)
        y = np.concatenate([np.zeros(7), np.arange(1, 7), np.full(6, 6.0)])
        x = np.concatenate([np.arange(7.0), np.full(6, 6.0), np.arange(5.0, -1.0, -1.0)])
        kinds = [0] * 7 + [1] * 6 + [2] * 6
        packed = np.packbits(self._codes(kinds).astype(np.uint8), axis=1)
        index = np.arange(len(kinds))
        picture = np.zeros(len(kinds), dtype=np.int64)
        builder = self._builder(thresholds=(0.9, 0.0), min_points=4)
        own = builder.build(picture, y, x, packed, index, 32)
        moved = builder.build(picture, y + 3.25, x - 7.5, packed, index, 32)
        self.assertGreater(own.count, 0)
        self.assertEqual(own.count, moved.count)
        self.assertTrue(np.allclose(own.radius, moved.radius))
        self.assertTrue(np.allclose(own.pieces.centre + (3.25, -7.5), moved.pieces.centre))
        self.assertTrue(np.array_equal(own.pieces.element, moved.pieces.element))
        self.assertTrue(np.allclose(own.pieces.offsets[own.pieces.offset], moved.pieces.offsets[moved.pieces.offset]))
        # Место элемента — в долях радиуса поля: всегда внутри единичного круга.
        self.assertLessEqual(float(np.abs(own.pieces.offsets).max()), 1.0 + 1e-9)
        del rng

    def test_lower_threshold_layers_gather_what_a_strict_layer_leaves(self) -> None:
        positions = np.asarray([[0.0, float(x)] for x in range(10)])
        # Код точки делит с соседом 6 бит из 8, через одного — 4: при строгом пороге похожих на центр точек нет,
        # при мягком центр зажигает по два соседа с каждой стороны — пять точек, как раз на поле.
        bits = np.zeros((10, 32), dtype=np.float32)
        for row in range(10):
            bits[row, row * 2 : row * 2 + 8] = 1
        self.assertEqual(self._builder(thresholds=0.9).picture_fields(positions, bits), [])
        self.assertGreater(len(self._builder(thresholds=(0.9, 0.4)).picture_fields(positions, bits)), 0)


def _binding_place() -> Encoder:
    """Детекторы позиции окна: два слоя кругов, место накрывают около четырёх кругов."""
    return Encoder(
        PlaneDimension(
            "Place",
            ((-4.0, 4.0), (-4.0, 4.0)),
            plane_detector_layers(layers_count=2, widest_radius=0.1875, narrowest_radius=0.09375, overlap=0.4),
            code_length=256,
            layer_bands=False,
        ),
        random_seed=0,
    )


class PositionBindingTest(unittest.TestCase):
    """Код окна внимания: бит места детектора позиции (OR) и его маска на код признака (XOR)."""

    OFFSETS = np.asarray([(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)], dtype=np.float64)
    LENGTH = 64

    def _encoder(self, **changes) -> PositionBindingEncoder:  # noqa: ANN003
        settings = dict(
            pooling="max", mask="address", mask_bits=6, target_density=0.2, saturation=1.0, max_length=1024, seed=3
        )
        settings.update(changes)
        return PositionBindingEncoder(_binding_place(), **settings)

    def _features(self, rows: list[list[int]]) -> np.ndarray:
        bits = np.zeros((len(rows), self.LENGTH), dtype=np.uint8)
        for row, ones in enumerate(rows):
            bits[row, ones] = 1
        return np.packbits(bits, axis=1)

    def _random_features(self, count: int, ones: int = 8, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return self._features([sorted(rng.permutation(self.LENGTH)[:ones].tolist()) for _ in range(count)])

    def _encode(self, encoder, packed, keyholes, *, unit: float = 1.0, offsets=None, strength=None):  # noqa: ANN001, ANN202
        """keyholes — список окон, окно — список пар (номер признака, номер места из OFFSETS)."""
        what = np.asarray([feature for keyhole in keyholes for feature, _ in keyhole], dtype=np.int64)
        place = np.asarray([spot for keyhole in keyholes for _, spot in keyhole], dtype=np.int64)
        groups = np.repeat(np.arange(len(keyholes)), [len(keyhole) for keyhole in keyholes])
        return encoder.encode(
            packed,
            self.LENGTH,
            what,
            self.OFFSETS if offsets is None else offsets,
            place,
            groups,
            len(keyholes),
            unit=unit,
            strength=strength,
        )

    @staticmethod
    def _segments(codes) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN001
        bits = codes.bits()
        return bits[:, : codes.place_length], bits[:, codes.place_length :]

    def _active(self, encoder, spot: int) -> np.ndarray:  # noqa: ANN001
        _, detectors = encoder.place.active_detectors(0, self.OFFSETS[[spot]])
        return detectors

    def test_content_is_the_feature_code_moved_by_the_address_mask_of_every_active_detector(self) -> None:
        encoder = self._encoder()
        ones = [1, 5, 9, 20, 33, 47, 60]
        codes = self._encode(encoder, self._features([ones]), [[(0, 4)]])
        place, content = self._segments(codes)
        detectors = self._active(encoder, 4)
        self.assertGreater(detectors.size, 1)
        # Позиция: биты сработавших детекторов места.
        self.assertEqual(set(np.flatnonzero(place[0])), set(encoder.place.detector_bits()[detectors].tolist()))
        # Содержание: у каждого детектора — перестановка битов признака, число единиц сохраняется.
        masks = encoder.address_masks(codes.content_length)
        expected: set[int] = set()
        for detector in detectors:
            moved = {bit ^ int(masks[detector]) for bit in ones}
            self.assertEqual(len(moved), len(ones))
            expected |= moved
        self.assertEqual(set(np.flatnonzero(content[0]).tolist()), expected)

    def test_one_layer_of_place_circles_is_the_widest(self) -> None:
        """П.7: один слой кругов места — только самые широкие круги (несколько крупных областей окна)."""
        from damp.encoding.damp_encoder import plane_detector_layers

        one = plane_detector_layers(layers_count=1, widest_radius=0.1875, narrowest_radius=0.09375, overlap=0.4)
        two = plane_detector_layers(layers_count=2, widest_radius=0.1875, narrowest_radius=0.09375, overlap=0.4)
        self.assertEqual(len(one), 1)
        self.assertEqual(one[0].count, two[0].count)

    def test_concepts_keep_the_strongest_codes_and_every_place_bit(self) -> None:
        """П.7: окно объединяет K самых сильных понятий — коды остальных отброшены целиком, биты места — у всех."""
        packed = self._random_features(3, seed=5)
        strength = np.asarray([0.2, 0.9, 0.5])  # понятие 0 — самое слабое
        full = self._encode(self._encoder(pooling="strongest"), packed, [[(0, 0), (1, 4), (2, 8)]], strength=strength)
        two = self._encode(
            self._encoder(pooling="strongest", concepts=2), packed, [[(0, 0), (1, 4), (2, 8)]], strength=strength
        )
        without = self._encode(self._encoder(pooling="strongest"), packed, [[(1, 4), (2, 8)]], strength=strength[1:])
        place_full, _ = self._segments(full)
        place_two, content_two = self._segments(two)
        _, content_without = self._segments(without)
        np.testing.assert_array_equal(place_two, place_full)  # место отмечено и у отброшенного понятия
        np.testing.assert_array_equal(content_two, content_without)  # его код в содержание не вошёл
        with self.assertRaises(ValueError):
            self._encoder(concepts=0)

    def test_same_feature_in_two_places_differs_in_place_bits_and_in_content(self) -> None:
        encoder = self._encoder()
        # Одна и та же вертикальная линия в левом верхнем и в правом нижнем углу окна.
        codes = self._encode(encoder, self._random_features(1), [[(0, 0)], [(0, 8)]])
        place, content = self._segments(codes)
        self.assertFalse(np.array_equal(place[0], place[1]))
        self.assertFalse(np.array_equal(content[0], content[1]))

    def test_swapping_features_between_places_changes_the_code(self) -> None:
        encoder = self._encoder()
        packed = self._random_features(2)
        codes = self._encode(encoder, packed, [[(0, 1), (1, 7)], [(1, 1), (0, 7)]])
        place, content = self._segments(codes)
        # Занятые места те же, и OR кодов признаков с кодами мест совпал бы: связь держит только маска.
        self.assertTrue(np.array_equal(place[0], place[1]))
        self.assertFalse(np.array_equal(content[0], content[1]))

    def test_code_changes_smoothly_with_the_place_of_a_feature(self) -> None:
        encoder = self._encoder()
        codes = self._encode(encoder, self._random_features(1), [[(0, spot)] for spot in range(9)]).bits().astype(np.float64)
        common = codes @ codes.T
        ones = codes.sum(axis=1)
        jaccard = common / (ones[:, None] + ones[None, :] - common)
        distance = np.hypot(
            self.OFFSETS[:, None, 0] - self.OFFSETS[None, :, 0], self.OFFSETS[:, None, 1] - self.OFFSETS[None, :, 1]
        )
        near = jaccard[np.isclose(distance, 1.0)].mean()
        far = jaccard[distance >= 2.0].mean()
        self.assertGreater(near, far + 0.1)
        self.assertLess(near, 1.0)

    def test_offsets_are_measured_in_steps_of_the_element_grid(self) -> None:
        packed = self._random_features(3)
        keyholes = [[(0, 0), (1, 4), (2, 5)], [(2, 8)]]
        own = self._encode(self._encoder(), packed, keyholes)
        scaled = self._encode(self._encoder(), packed, keyholes, unit=4.0, offsets=self.OFFSETS * 4.0)
        self.assertTrue(np.array_equal(own.packed, scaled.packed))

    def test_pooling_selects_what_a_detector_takes_from_its_area(self) -> None:
        first, second = [1, 5, 9, 20, 33, 47], [5, 9, 12, 33, 50, 61]
        packed = self._features([first, second])
        keyholes = [[(0, 4), (1, 4)], [(0, 4)], [(1, 4)]]
        # Максимум по битам: область детектора отдаёт объединение кодов признаков.
        _, content = self._segments(self._encode(self._encoder(pooling="max"), packed, keyholes))
        self.assertTrue(np.array_equal(content[0], content[1] | content[2]))
        # Минимум по битам: остаются общие биты признаков.
        encoder = self._encoder(pooling="min")
        codes = self._encode(encoder, packed, keyholes)
        masks = encoder.address_masks(codes.content_length)
        shared = sorted(set(first) & set(second))
        expected = {bit ^ int(masks[detector]) for detector in self._active(encoder, 4) for bit in shared}
        self.assertEqual(set(np.flatnonzero(self._segments(codes)[1][0]).tolist()), expected)
        # Самый сильный: область отдаёт код одного признака — с наибольшим уровнем активации.
        strength = np.asarray([0.2, 0.9, 1.0, 1.0])
        _, content = self._segments(self._encode(self._encoder(pooling="strongest"), packed, keyholes, strength=strength))
        self.assertTrue(np.array_equal(content[0], content[2]))
        with self.assertRaises(ValueError):
            self._encode(self._encoder(pooling="strongest"), packed, keyholes)

    def test_literal_xor_with_a_mask_vector_is_the_symmetric_difference(self) -> None:
        encoder = self._encoder(mask="bits", mask_bits=6)
        ones = [1, 5, 9, 20, 33, 47, 60]
        codes = self._encode(encoder, self._features([ones]), [[(0, 4)]])
        self.assertEqual(codes.content_length, self.LENGTH)
        masks = encoder.bit_masks(self.LENGTH)
        expected: set[int] = set()
        for detector in self._active(encoder, 4):
            mask = set(masks[detector].tolist())
            self.assertEqual(len(mask), 6)
            changed = set(ones) ^ mask
            # Число единиц после XOR: k + m − 2·(общих бит кода и маски).
            self.assertEqual(len(changed), len(ones) + 6 - 2 * len(set(ones) & mask))
            expected |= changed
        self.assertEqual(set(np.flatnonzero(self._segments(codes)[1][0]).tolist()), expected)

    def test_colour_merge_keeps_the_budget_and_drops_narrow_layers_first(self) -> None:
        packed = self._random_features(6, ones=10)
        keyholes = [[(feature, spot) for feature, spot in zip(range(6), (0, 2, 3, 4, 6, 8))]]
        free = self._encode(self._encoder(), packed, keyholes)
        encoder = self._encoder(saturation=0.05)
        tight = self._encode(encoder, packed, keyholes)
        self.assertEqual(free.content_length, tight.content_length)
        budget = int(0.05 * tight.content_length)
        kept = self._segments(tight)[1][0]
        full = self._segments(free)[1][0]
        self.assertGreater(int(full.sum()), budget)
        self.assertEqual(int(kept.sum()), budget)
        self.assertEqual(int(np.count_nonzero(kept & ~full)), 0)
        # Биты широкого слоя детекторов позиции: их больше бюджета, значит, остались только они.
        masks = encoder.address_masks(tight.content_length)
        layers = encoder.place.detector_layer_indices()
        bits = np.unpackbits(packed, axis=1, count=self.LENGTH)
        wide: set[int] = set()
        for feature, spot in keyholes[0]:
            for detector in self._active(encoder, spot):
                if layers[detector] == 0:
                    wide |= {int(bit) ^ int(masks[detector]) for bit in np.flatnonzero(bits[feature])}
        self.assertGreaterEqual(len(wide), budget)
        self.assertTrue(set(np.flatnonzero(kept).tolist()) <= wide)
        # Место не урезается: биты позиции — отдельный сегмент.
        self.assertTrue(np.array_equal(self._segments(tight)[0], self._segments(free)[0]))

    def test_codes_are_the_same_on_the_device_and_on_the_host(self) -> None:
        """Сборка кодов на видеокарте (раскраска, слияние цветов, упаковка) даёт те же байты, что на процессоре."""
        rng = np.random.default_rng(8)
        packed = self._random_features(30, ones=12, seed=2)
        keyholes = [
            [(int(rng.integers(0, 30)), int(spot)) for spot in rng.permutation(9)[: int(rng.integers(1, 9))]]
            for _ in range(200)
        ]
        strength = rng.random(sum(len(keyhole) for keyhole in keyholes)) + 0.1
        for pooling in ("max", "strongest", "min"):
            for saturation in (1.0, 0.06):
                settings = dict(pooling=pooling, saturation=saturation, target_density=0.15)
                host = self._encode(self._encoder(use_gpu=False, **settings), packed, keyholes, strength=strength)
                device = self._encode(self._encoder(use_gpu=True, **settings), packed, keyholes, strength=strength)
                self.assertEqual(host.code_length, device.code_length)
                self.assertTrue(np.array_equal(host.packed, device.packed), f"{pooling}, σ = {saturation}")

    def test_content_length_grows_until_codes_are_sparse_and_then_stays(self) -> None:
        encoder = self._encoder(target_density=0.1)
        packed = self._random_features(4)
        codes = self._encode(encoder, packed, [[(0, 0), (1, 4), (2, 8), (3, 2)]] * 3)
        self.assertGreater(codes.content_length, self.LENGTH)
        self.assertEqual(codes.content_length & (codes.content_length - 1), 0)
        self.assertLessEqual(self._segments(codes)[1].mean(), 0.1)
        # Длина подобрана первым вызовом: тем же кодером новые окна кодируются в то же пространство.
        again = self._encode(encoder, packed, [[(3, 5)]])
        self.assertEqual(again.code_length, codes.code_length)
        with self.assertRaises(ValueError):
            self._encoder(pooling="mean")
        with self.assertRaises(ValueError):
            self._encoder(mask="or")


class DenseKeyholesTest(unittest.TestCase):
    """Окна внимания миниколонок: окно на каждом элементе, радиус по кластеру его соседей (П.2, §6.4.3)."""

    @staticmethod
    def _builder(scale: float = 1.0, thresholds=0.9, min_points: int = 5):
        from damp.cycle.plane_fields import PlaneFieldBuilder

        return PlaneFieldBuilder(
            lambda_thresholds=thresholds, activation_radius=6.0 * scale, eps=1.5 * scale, min_samples=3,
            min_points=min_points, radius_min=1.0 * scale, energy_radius=3.0 * scale, energy_threshold=0.12, dense=True,
        )

    @staticmethod
    def _alike(count: int) -> np.ndarray:
        bits = np.zeros((count, 32), dtype=np.float32)
        bits[:, :8] = 1
        return bits

    def test_concepts_cap_the_keyhole_at_the_kth_nearest_element(self) -> None:
        """П.7: окно объединяет не больше concepts понятий — радиус не дальше concepts-го ближайшего элемента."""
        from damp.cycle.plane_fields import PlaneFieldBuilder

        positions = np.asarray([[0.0, float(x)] for x in range(9)])
        builder = PlaneFieldBuilder(
            lambda_thresholds=0.9, activation_radius=6.0, eps=1.5, min_samples=3, min_points=5, radius_min=1.0,
            energy_radius=3.0, energy_threshold=0.12, dense=True, concepts=5,
        )
        radius = {x: r for _, x, r, _ in builder.picture_fields(positions, self._alike(9))}
        # Середина ряда: 5-й ближайший (с самим элементом) — на расстоянии 2, как и без ограничения; у конца ряда
        # окно было 4, а теперь — до 5-го ближайшего, тоже 4 (0, 1, 2, 3, 4).
        self.assertAlmostEqual(radius[4.0], 2.0)
        self.assertAlmostEqual(radius[0.0], 4.0)
        narrow = PlaneFieldBuilder(
            lambda_thresholds=0.9, activation_radius=6.0, eps=1.5, min_samples=3, min_points=3, radius_min=1.0,
            energy_radius=3.0, energy_threshold=0.12, dense=True, concepts=3,
        )
        radius = {x: r for _, x, r, _ in narrow.picture_fields(positions, self._alike(9))}
        self.assertAlmostEqual(radius[4.0], 1.0)  # 3 элемента: сам и два соседа
        self.assertLessEqual(radius[0.0], 2.0)  # у конца ряда: сам, 1, 2

    def test_every_element_gets_a_keyhole_centred_on_itself(self) -> None:
        positions = np.asarray([[0.0, float(x)] for x in range(9)])
        fields = self._builder().picture_fields(positions, self._alike(9))
        # Без вытеснения: окон столько же, сколько элементов, и стоят они на самих элементах.
        self.assertEqual(len(fields), 9)
        self.assertEqual(sorted((y, x) for y, x, _, _ in fields), [(0.0, float(x)) for x in range(9)])
        radius = {x: r for _, x, r, _ in fields}
        # В середине ряда наименьший круг с пятью соседями — радиус 2; у конца ряда он шире.
        self.assertAlmostEqual(radius[4.0], 2.0)
        self.assertAlmostEqual(radius[0.0], 4.0)

    def test_neighbouring_keyholes_overlap(self) -> None:
        positions = np.asarray([[0.0, float(x)] for x in range(9)])
        fields = sorted(self._builder().picture_fields(positions, self._alike(9)), key=lambda field: field[1])
        for first, second in zip(fields[:-1], fields[1:]):
            self.assertLess(second[1] - first[1], first[2] + second[2])

    def test_keyhole_grows_with_the_measure_of_the_plane(self) -> None:
        """Радиусы построения в мере «что видит деталь»: окно на карте не меньше поля своей детали."""
        positions = np.asarray([[0.0, float(x)] for x in range(25)])
        small = {x: r for _, x, r, _ in self._builder(scale=1.0).picture_fields(positions, self._alike(25))}
        large = {x: r for _, x, r, _ in self._builder(scale=3.0).picture_fields(positions, self._alike(25))}
        self.assertAlmostEqual(small[12.0], 2.0)
        self.assertAlmostEqual(large[12.0], 3.0)
        self.assertGreaterEqual(min(large.values()), 3.0)

    def test_keyholes_follow_the_picture_when_it_moves(self) -> None:
        y = np.concatenate([np.zeros(7), np.arange(1, 7), np.full(6, 6.0)])
        x = np.concatenate([np.arange(7.0), np.full(6, 6.0), np.arange(5.0, -1.0, -1.0)])
        bits = self._alike(19)
        packed = np.packbits(bits.astype(np.uint8), axis=1)
        index = np.arange(19)
        picture = np.zeros(19, dtype=np.int64)
        builder = self._builder(thresholds=(0.9, 0.0))
        own = builder.build(picture, y, x, packed, index, 32)
        moved = builder.build(picture, y + 3.25, x - 7.5, packed, index, 32)
        self.assertEqual(own.count, 19)
        self.assertTrue(np.allclose(own.radius, moved.radius))
        self.assertTrue(np.allclose(own.pieces.centre + (3.25, -7.5), moved.pieces.centre))
        self.assertTrue(np.array_equal(own.pieces.element, moved.pieces.element))
        self.assertTrue(np.allclose(own.pieces.offsets[own.pieces.offset], moved.pieces.offsets[moved.pieces.offset]))

    def test_keyhole_holds_the_element_that_sets_its_radius(self) -> None:
        """§6.4.3: радиус окна — расстояние до самого дальнего вошедшего соседа, и этот сосед — в окне. Центр и радиус
        хранятся с 6 знаками; на дробных местах (MNIST_SUBCELL_POSITION) округление не должно выбрасывать его."""
        rng = np.random.default_rng(7)
        count = 60
        y, x = rng.uniform(0.0, 9.0, count), rng.uniform(0.0, 9.0, count)
        packed = np.packbits(self._alike(1).astype(np.uint8), axis=1)
        built = self._builder().build(np.zeros(count, dtype=np.int64), y, x, packed, np.zeros(count, dtype=np.int64), 32)
        pieces = built.pieces
        reach = np.hypot(y[pieces.element] - pieces.centre[pieces.piece, 0], x[pieces.element] - pieces.centre[pieces.piece, 1])
        farthest = np.zeros(pieces.count)
        np.maximum.at(farthest, pieces.piece, reach)
        bounded = built.radius > 1.0 + 1e-6  # радиус задан соседом, а не наименьшим радиусом поля
        self.assertGreater(int(bounded.sum()), 10)
        np.testing.assert_allclose(farthest[bounded], built.radius[bounded], atol=2e-6)


class MatrixClustersTest(unittest.TestCase):
    """Кластеры всех центров разом (умножением матриц) совпадают с обходом от каждого центра по отдельности."""

    RADIUS, EPS, MIN_SAMPLES, MIN_POINTS, RADIUS_MIN = 6.0, 1.5, 3, 5, 1.0

    @staticmethod
    def _plain_cluster(near: np.ndarray, seed: int, min_samples: int) -> np.ndarray:
        """Обход DBSCAN от одного центра — прямая запись §6.4.1, по ней сверяется быстрый счёт."""
        core = near.sum(axis=1) >= min_samples
        start = seed
        if not core[seed]:
            around = np.flatnonzero(near[seed] & core)
            if around.size == 0:
                return np.zeros(near.shape[0], dtype=bool)
            start = int(around[0])
        member = np.zeros(near.shape[0], dtype=bool)
        member[start] = True
        pending = [start]
        while pending:
            point = pending.pop()
            if not core[point]:
                continue
            fresh = near[point] & ~member
            member |= near[point]
            pending.extend(np.flatnonzero(fresh).tolist())
        member[seed] = True
        return member

    def _plain_keyholes(self, positions: np.ndarray, bits: np.ndarray, threshold: float) -> list:
        """Окна одного слоя, посчитанные по одному центру за раз."""
        import math

        count = positions.shape[0]
        ones = bits.sum(axis=1)
        similarity = (bits @ bits.T) / np.sqrt(np.maximum(ones[:, None] * ones[None, :], 1.0))
        distance = np.hypot(positions[:, None, 0] - positions[None, :, 0], positions[:, None, 1] - positions[None, :, 1])
        activation = np.where(similarity >= threshold, similarity, 0.0).astype(np.float64)
        np.fill_diagonal(activation, np.where(ones > 0, 1.0, 0.0))
        around = (distance <= 3.0) & ~np.eye(count, dtype=bool)
        energy = np.where(around, activation / np.maximum(distance, self.RADIUS_MIN), 0.0).sum(axis=1)
        energy = energy / energy.max() if energy.max() > 0 else np.zeros(count)
        alive = energy >= 0.12
        keyholes = []
        for centre in np.flatnonzero(alive):
            disc = np.flatnonzero((distance[centre] <= self.RADIUS) & alive & (activation[centre] > 0))
            seed = int(np.searchsorted(disc, centre))
            if seed >= disc.size or disc[seed] != centre:
                continue
            cluster = disc[self._plain_cluster(distance[np.ix_(disc, disc)] <= self.EPS, seed, self.MIN_SAMPLES)]
            if cluster.size < self.MIN_POINTS:
                continue
            reach = np.sort(distance[centre, cluster])
            inside = np.arange(1, reach.size + 1, dtype=np.float64)
            radii = np.maximum(self.RADIUS_MIN, reach)
            allowed = (radii <= self.RADIUS) & (inside >= self.MIN_POINTS)
            if not allowed.any():
                continue
            radius = float(radii[int(np.argmax(np.where(allowed, inside / (math.pi * radii * radii), -np.inf)))])
            points = int(np.count_nonzero((distance[centre] <= radius) & alive))
            if points < self.MIN_POINTS:
                continue
            keyholes.append((round(float(positions[centre, 0]), 6), round(float(positions[centre, 1]), 6), round(radius, 6), points))
        return keyholes

    def test_keyholes_match_the_search_from_every_centre(self) -> None:
        from damp.cycle.plane_fields import PlaneFieldBuilder

        rng = np.random.default_rng(11)
        checked = 0
        for trial in range(12):
            # Точки на клетках, как углопозиции: куски линий и случайная россыпь; коды четырёх видов с общими битами.
            cells = rng.permutation(14 * 14)[: int(rng.integers(20, 90))]
            positions = np.stack([cells // 14, cells % 14], axis=1).astype(np.float64)
            bits = np.zeros((positions.shape[0], 40), dtype=np.float32)
            for row, kind in enumerate(rng.integers(0, 4, positions.shape[0])):
                bits[row, kind * 6 : kind * 6 + 12] = 1
            for threshold in (0.9, 0.4, 0.0):
                builder = PlaneFieldBuilder(
                    lambda_thresholds=threshold, activation_radius=self.RADIUS, eps=self.EPS,
                    min_samples=self.MIN_SAMPLES, min_points=self.MIN_POINTS, radius_min=self.RADIUS_MIN,
                    energy_radius=3.0, energy_threshold=0.12, dense=True,
                )
                expected = sorted(self._plain_keyholes(positions, bits, threshold))
                self.assertEqual(sorted(builder.picture_fields(positions, bits)), expected, f"trial {trial}, λ {threshold}")
                checked += len(expected)
        self.assertGreater(checked, 300)


class SameResultFasterTest(unittest.TestCase):
    """Ускоренные счёты дают тот же результат, что прямые: кластер на сетке, отбор сильнейших, таблица общих бит."""

    def test_grid_cluster_matches_the_reference_search(self) -> None:
        from damp.activation import DbscanClusterer

        rng = np.random.default_rng(3)
        checked = noise = 0
        for eps, min_samples in ((1.5, 3), (1.5, 5), (1.0, 2), (2.0, 4), (2.5, 3)):
            clusterer = DbscanClusterer(eps=eps, min_samples=min_samples)
            for _ in range(40):
                side = int(rng.integers(6, 16))
                cells = rng.permutation(side * side)[: int(rng.integers(3, side * side * 0.6))]
                ys, xs = (cells // side).astype(np.int64) + 7, (cells % side).astype(np.int64) - 3
                positions = list(zip(ys.tolist(), xs.tolist()))
                for seed in (positions[0], positions[len(positions) // 2], (0, 0)):
                    expected = set(clusterer._cluster_of_python(positions, seed))
                    mask = clusterer.cluster_mask(ys, xs, seed)
                    self.assertEqual({position for position, kept in zip(positions, mask.tolist()) if kept}, expected)
                    checked += 1
                    noise += not expected
        self.assertGreater(checked, 500)
        self.assertGreater(noise, 0)
        self.assertLess(noise, checked)

    def test_strongest_detectors_match_full_sorting(self) -> None:
        rng = np.random.default_rng(4)
        for saturation in (1, 3, 7, 40, 60):
            embeddings = DetectorEmbeddings(
                detectors=40, saturation_limit=saturation, target_density=0.25, min_length=8, max_length=4096, seed=0
            )
            fired = rng.random((300, 40)) < rng.random((300, 1))
            rows, columns = np.nonzero(fired)
            # Уровни с малым числом разных значений: равных много, проверяется и правило при равенстве.
            levels = rng.integers(1, 5, rows.shape[0]).astype(np.float64) / 4.0
            order = np.lexsort((columns, -levels, rows))
            first = np.searchsorted(rows[order], rows[order], side="left")
            keep = np.arange(rows.shape[0]) - first < saturation
            expected = sorted(zip(rows[order][keep].tolist(), columns[order][keep].tolist()))
            shuffled = rng.permutation(rows.shape[0])
            kept_rows, kept_columns = embeddings.strongest(rows[shuffled], columns[shuffled], levels[shuffled])
            self.assertEqual(sorted(zip(kept_rows.tolist(), kept_columns.tolist())), expected, f"σ = {saturation}")

    def test_hierarchy_is_the_same_with_and_without_the_table_of_common_bits(self) -> None:
        from damp.activation import DbscanClusterer, DetectorHierarchyBuilder, DetectorLayerConfig

        def build(table_points: int):
            layout = _clustered_layout()
            builder = DetectorHierarchyBuilder(
                layout=layout,
                energies=layout.point_energies(3),
                layer_configs=[
                    DetectorLayerConfig(lambda_threshold=0.4, activation_radius=6, min_points=4),
                    DetectorLayerConfig(lambda_threshold=0.6, activation_radius=4, min_points=3),
                ],
                radius_min=1,
                output_bits=256,
                energy_threshold=0.05,
                eta=None,
                attempts_without_change=60,
                seed=1,
                clusterer=DbscanClusterer(eps=1.5, min_samples=3),
            )
            builder.COMMON_TABLE_POINTS = table_points
            detectors = builder.build()
            return [(d.center, d.radius, d.lambda_threshold, d.energy, d.points_count) for d in detectors], builder

        with_table, fast = build(20000)
        without_table, plain = build(0)
        self.assertIsNotNone(fast._common)
        self.assertIsNone(plain._common)
        # Без таблицы кластеры считаются по одному центру; с таблицей и видеокартой — разом для всех центров слоя.
        self.assertEqual(plain._layer_clusters, {})
        import torch

        if torch.cuda.is_available():
            self.assertEqual(len(fast._layer_clusters), 2)
        self.assertGreater(len(with_table), 1)
        self.assertEqual(with_table, without_table)


class LateralInhibitionTest(unittest.TestCase):
    """Торможение соседей: из близко стоящих деталей остаётся самая сильная."""

    @staticmethod
    def _keep(y, x, strength, sight, share: float = 0.5, picture=None):
        from damp.cycle.lateral_inhibition import LateralInhibition

        count = len(y)
        picture = np.zeros(count, dtype=np.int64) if picture is None else np.asarray(picture)
        return LateralInhibition(radius_share=share).survivors(
            picture, np.asarray(y, dtype=float), np.asarray(x, dtype=float), np.asarray(strength, dtype=float),
            np.full(count, float(sight)),
        )

    def test_the_strongest_of_close_details_stays(self) -> None:
        # Пять деталей в ряд через клетку, круг торможения — 2 клетки (половина от того, что видит деталь).
        keep = self._keep([0] * 5, range(5), [0.3, 0.9, 0.5, 0.4, 0.8], sight=4.0)
        # Сильнейшая (место 1) остаётся и гасит соседей до места 3; место 4 дальше круга и остаётся.
        self.assertEqual(keep.tolist(), [False, True, False, False, True])

    def test_details_farther_than_the_circle_all_stay(self) -> None:
        keep = self._keep([0] * 4, [0, 3, 6, 9], [0.2, 0.9, 0.4, 0.6], sight=4.0)
        self.assertTrue(keep.all())

    def test_silent_details_are_dropped(self) -> None:
        keep = self._keep([0, 0, 0], [0, 10, 20], [0.0, 0.5, 0.0], sight=2.0)
        self.assertEqual(keep.tolist(), [False, True, False])

    def test_pictures_do_not_inhibit_each_other(self) -> None:
        keep = self._keep([0, 0, 0, 0], [0, 1, 0, 1], [0.9, 0.5, 0.4, 0.8], sight=4.0, picture=[0, 0, 1, 1])
        self.assertEqual(keep.tolist(), [True, False, False, True])

    def test_choice_does_not_depend_on_the_place_of_the_picture(self) -> None:
        rng = np.random.default_rng(5)
        y, x = rng.integers(0, 12, 40).astype(float), rng.integers(0, 12, 40).astype(float)
        strength = rng.random(40) + 0.1
        own = self._keep(y, x, strength, sight=5.0)
        moved = self._keep(y + 3.25, x - 7.5, strength, sight=5.0)
        self.assertTrue(np.array_equal(own, moved))
        self.assertLess(int(own.sum()), 40)
        # Оставшиеся детали не попадают в круги друг друга.
        kept = np.flatnonzero(own)
        gaps = np.hypot(y[kept][:, None] - y[kept][None, :], x[kept][:, None] - x[kept][None, :])
        np.fill_diagonal(gaps, np.inf)
        self.assertGreater(float(gaps.min()), 2.5)
