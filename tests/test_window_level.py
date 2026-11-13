"""Уровень «окно внимания» (расширение статьи): куски в координатах окна, детекторы-произведения, словарь образцов."""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("RERUN", "off")

import numpy as np  # noqa: E402

from damp.decode.nearest_codes import NearestCodeClassifier  # noqa: E402
from damp.encoding.attention_window import AttentionWindows, BlockWindows  # noqa: E402
from damp.encoding.conjunction import ConjunctionEncoder  # noqa: E402
from damp.encoding.damp_encoder import (  # noqa: E402
    ClosedDimension,
    Encoder,
    PlaneDimension,
    geometric_detector_layers,
    halving_detector_layers,
    plane_detector_layers,
)
from damp.layout.code_dictionary import LeaderDictionary, PairwiseSimilarity  # noqa: E402
from damp.logging import LOGGER  # noqa: E402

LOGGER.mute()


def _angle_encoder(layers: int = 3) -> Encoder:
    return Encoder(
        ClosedDimension(
            "Angle",
            (0.0, 360.0),
            geometric_detector_layers(layers_count=layers, widest_detectors=4, narrowest_detectors=24, overlap=0.6),
            code_length=64,
        ),
        random_seed=0,
    )


def _plane_encoder(layers: int = 3, side: float = 5.0) -> Encoder:
    return Encoder(
        PlaneDimension(
            "Offset",
            ((-side / 2, side / 2), (-side / 2, side / 2)),
            plane_detector_layers(layers_count=layers, widest_radius=0.5, narrowest_radius=0.16, overlap=0.6),
            code_length=64,
            layer_bands=False,
        ),
        random_seed=0,
    )


class HalvingLayersTest(unittest.TestCase):
    def test_counts_double(self) -> None:
        layers = halving_detector_layers(layers_count=5, widest_detectors=2, overlap=0.6)
        self.assertEqual([layer.count for layer in layers], [2, 4, 8, 16, 32])
        self.assertTrue(all(layer.overlap == 0.6 for layer in layers))

    def test_every_circle_lies_inside_a_circle_of_the_previous_layer(self) -> None:
        """Решётки слоёв совпадают: круг слоя целиком внутри одного из кругов предыдущего слоя."""
        encoder = Encoder(
            PlaneDimension(
                "Map",
                ((0.0, 64.0), (0.0, 64.0)),
                halving_detector_layers(layers_count=4, widest_detectors=2, overlap=0.6),
                code_length=64,
                layer_bands=False,
            ),
            random_seed=0,
        )
        by_layer: dict[int, list] = {}
        for circle in encoder.detectors:
            by_layer.setdefault(circle.layer_index, []).append(circle)
        for layer in range(1, 4):
            for circle in by_layer[layer]:
                inside = any(
                    np.hypot(circle.center_y - parent.center_y, circle.center_x - parent.center_x) + circle.radius
                    <= parent.radius + 1e-9
                    for parent in by_layer[layer - 1]
                )
                self.assertTrue(inside)


class NestedCirclesTest(unittest.TestCase):
    def test_nested_circles_lie_inside_a_kept_wider_circle(self) -> None:
        encoder = _plane_encoder(layers=4, side=40.0)
        values = np.asarray([[1.5, -7.0], [0.0, 0.0], [-12.25, 9.5]])
        rows, everything = encoder.active_detectors(0, values)
        nested_rows, nested = encoder.active_detectors(0, values, nested=True)
        for row in range(values.shape[0]):
            all_ids = set(everything[rows == row].tolist())
            kept = [encoder.detectors[i] for i in nested[nested_rows == row]]
            self.assertTrue({c.detector_index for c in kept} <= all_ids)
            widest = min(c.layer_index for c in kept)
            for circle in kept:
                if circle.layer_index == widest:
                    continue
                self.assertTrue(
                    any(
                        np.hypot(circle.center_y - parent.center_y, circle.center_x - parent.center_x) + circle.radius
                        <= parent.radius + 1e-9
                        for parent in kept
                        if parent.layer_index < circle.layer_index
                    )
                )

    def test_halving_layers_keep_every_circle(self) -> None:
        """При слоях «вдвое» решётки совпадают: вложены все круги, правило ничего не отбрасывает."""
        encoder = Encoder(
            PlaneDimension(
                "Map",
                ((0.0, 64.0), (0.0, 64.0)),
                halving_detector_layers(layers_count=4, widest_detectors=2, overlap=0.6),
                code_length=64,
                layer_bands=False,
            ),
            random_seed=0,
        )
        values = np.asarray([[10.3, 50.1], [33.0, 31.9], [63.0, 2.0]])
        _, everything = encoder.active_detectors(0, values)
        _, nested = encoder.active_detectors(0, values, nested=True)
        self.assertEqual(sorted(everything.tolist()), sorted(nested.tolist()))


class EncoderActiveDetectorsTest(unittest.TestCase):
    def test_matches_set_code_of_single_value(self) -> None:
        encoder = _plane_encoder()
        value = np.asarray([[0.4, -1.1]])
        rows, detectors = encoder.active_detectors(0, value)
        _, active = encoder.encode_set(value)
        self.assertTrue(np.all(rows == 0))
        self.assertEqual(sorted(detectors.tolist()), sorted(active.tolist()))

    def test_title_and_unknown_title(self) -> None:
        encoder = _angle_encoder()
        by_index = encoder.active_detectors(0, np.asarray([10.0, 200.0]))
        by_title = encoder.active_detectors("Angle", np.asarray([10.0, 200.0]))
        self.assertTrue(np.array_equal(by_index[1], by_title[1]))
        with self.assertRaises(KeyError):
            encoder.active_detectors("Nope", np.asarray([1.0]))


class ConjunctionEncoderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.encoder = ConjunctionEncoder(_angle_encoder(), _plane_encoder(), random_seed=0)
        self.encoder.set_length(1024)
        self.angles = np.asarray([20.0, 200.0])
        self.places = np.asarray([[-1.5, -1.5], [1.5, 1.5]])

    def _code(self, pairs) -> np.ndarray:
        what = np.asarray([a for a, _ in pairs])
        where = np.asarray([p for _, p in pairs])
        return self.encoder.encode_sets(self.angles, what, self.places, where, np.zeros(len(pairs), dtype=np.int64), 1)[0]

    def test_set_code_does_not_depend_on_order(self) -> None:
        self.assertTrue(np.array_equal(self._code([(0, 0), (1, 1)]), self._code([(1, 1), (0, 0)])))

    def test_binding_is_kept(self) -> None:
        """Набор {(a₁, p₁), (a₂, p₂)} отличается от {(a₁, p₂), (a₂, p₁)}: связь «что — где» не теряется."""
        straight = np.unpackbits(self._code([(0, 0), (1, 1)]))
        swapped = np.unpackbits(self._code([(0, 1), (1, 0)]))
        common = int(np.count_nonzero(straight & swapped))
        self.assertLess(common, 0.5 * int(np.count_nonzero(straight)))

    def test_set_code_is_union_of_element_codes(self) -> None:
        both = np.unpackbits(self._code([(0, 0), (1, 1)]))
        first = np.unpackbits(self._code([(0, 0)]))
        second = np.unpackbits(self._code([(1, 1)]))
        self.assertTrue(np.array_equal(both, first | second))

    def test_groups_are_encoded_separately(self) -> None:
        packed = self.encoder.encode_sets(
            self.angles, np.asarray([0, 1]), self.places, np.asarray([0, 1]), np.asarray([0, 1]), 2
        )
        self.assertTrue(np.array_equal(packed[0], self._code([(0, 0)])))
        self.assertTrue(np.array_equal(packed[1], self._code([(1, 1)])))

    def test_length_is_fitted_to_density(self) -> None:
        rng = np.random.default_rng(0)
        angles = np.arange(0.0, 360.0, 5.0)
        places = np.stack(np.meshgrid(np.arange(-2.0, 2.1), np.arange(-2.0, 2.1), indexing="ij"), axis=-1).reshape(-1, 2)
        groups = np.repeat(np.arange(200), 6)
        what = rng.integers(0, angles.shape[0], groups.shape[0])
        where = rng.integers(0, places.shape[0], groups.shape[0])
        density = self.encoder.fit_length(
            angles, what, places, where, groups, sample_groups=200, target_density=0.25, min_length=32, max_length=65536
        )
        self.assertAlmostEqual(density, 0.25, delta=0.02)
        packed = self.encoder.encode_sets(angles, what, places, where, groups, 200)
        self.assertAlmostEqual(float(np.unpackbits(packed, axis=1, count=self.encoder.code_length).mean()), density, delta=1e-9)

    def test_layers_must_match_without_cross(self) -> None:
        with self.assertRaises(ValueError):
            ConjunctionEncoder(_angle_encoder(3), _plane_encoder(4), random_seed=0)
        crossed = ConjunctionEncoder(_angle_encoder(3), _plane_encoder(4), random_seed=0, cross_layers=True)
        self.assertGreater(crossed.detector_count, 0)


class AttentionWindowsTest(unittest.TestCase):
    def test_piece_does_not_depend_on_place_of_the_window(self) -> None:
        """Тот же рисунок в другом месте картинки даёт те же куски, сдвинутые целиком."""
        windows = AttentionWindows(radius=1.5)
        y = np.asarray([3, 3, 4])
        x = np.asarray([3, 4, 4])
        first = windows.pieces(np.zeros(3, dtype=np.int64), y, x)
        second = windows.pieces(np.zeros(3, dtype=np.int64), y + 7, x + 5)
        self.assertEqual(first.count, second.count)
        self.assertTrue(np.array_equal(first.centre + np.asarray([7.0, 5.0]), second.centre))
        self.assertTrue(np.array_equal(first.piece, second.piece))
        self.assertTrue(np.array_equal(first.offset, second.offset))

    def test_offsets_point_from_centre_to_element(self) -> None:
        windows = AttentionWindows(radius=2.5)
        y, x = np.asarray([10, 12]), np.asarray([10, 11])
        pieces = windows.pieces(np.zeros(2, dtype=np.int64), y, x)
        element = np.stack([y, x], axis=1).astype(np.float64)[pieces.element]
        self.assertTrue(np.array_equal(pieces.centre[pieces.piece] + pieces.offsets[pieces.offset], element))
        self.assertTrue(np.all(np.hypot(pieces.offsets[:, 0], pieces.offsets[:, 1]) <= 2.5))

    def test_pictures_do_not_mix_and_stride_thins_windows(self) -> None:
        y, x = np.asarray([5, 5]), np.asarray([5, 5])
        dense = AttentionWindows(radius=1.5).pieces(np.asarray([0, 1]), y, x)
        self.assertEqual(dense.count, 18)  # 9 окон вокруг элемента в каждой из двух картинок
        self.assertEqual(set(dense.picture.tolist()), {0, 1})
        sparse = AttentionWindows(radius=1.5, stride=2).pieces(np.asarray([0, 1]), y, x)
        self.assertLess(sparse.count, dense.count)
        self.assertTrue(np.all(sparse.centre % 2 == 0))


class BlockWindowsTest(unittest.TestCase):
    def test_every_element_is_in_exactly_one_piece(self) -> None:
        rng = np.random.default_rng(1)
        picture = np.sort(rng.integers(0, 3, 200))
        y, x = rng.integers(0, 24, 200), rng.integers(0, 24, 200)
        pieces = BlockWindows(side=3).pieces(picture, y, x)
        self.assertEqual(sorted(pieces.element.tolist()), list(range(200)))
        self.assertTrue(np.all(np.diff(pieces.piece) >= 0))
        # Элемент лежит в своём блоке: смещение от центра не больше (side − 1)/2 по каждой оси.
        self.assertTrue(np.all(np.abs(pieces.offsets) <= 1.0))
        centre = pieces.centre[pieces.piece]
        shift = pieces.offsets[pieces.offset]
        self.assertTrue(np.array_equal(centre[:, 0] + shift[:, 0], y[pieces.element].astype(np.float64)))
        self.assertTrue(np.array_equal(centre[:, 1] + shift[:, 1], x[pieces.element].astype(np.float64)))
        self.assertTrue(np.array_equal(pieces.picture[pieces.piece], picture[pieces.element]))

    def test_larger_blocks_collect_whole_smaller_blocks(self) -> None:
        y, x = np.asarray([0, 2, 3, 5, 6]), np.asarray([0, 2, 3, 5, 6])
        small = BlockWindows(side=3).pieces(np.zeros(5, dtype=np.int64), y, x)
        self.assertEqual(small.centre.tolist(), [[1.0, 1.0], [4.0, 4.0], [7.0, 7.0]])
        large = BlockWindows(side=6).pieces(small.picture, small.centre[:, 0], small.centre[:, 1])
        self.assertEqual(large.centre.tolist(), [[2.5, 2.5], [8.5, 8.5]])
        self.assertEqual(large.piece.tolist(), [0, 0, 1])
        self.assertEqual(sorted(large.offsets.tolist()), [[-1.5, -1.5], [1.5, 1.5]])


class LeaderDictionaryTest(unittest.TestCase):
    def test_similar_codes_merge_and_distinct_stay(self) -> None:
        length = 64
        base = np.zeros((3, length), dtype=np.uint8)
        base[0, :16] = 1
        base[1, 16:32] = 1
        base[2, 32:48] = 1
        near = base[0].copy()
        near[15] = 0
        near[60] = 1  # сходство с первым образцом 15/16
        codes = np.stack([base[0], base[0], near, base[1], base[2], base[1]])
        dictionary = LeaderDictionary(similarity_threshold=0.7, max_size=10, use_gpu=False).select(
            np.packbits(codes, axis=1), length
        )
        self.assertEqual(dictionary.distinct, 4)
        self.assertEqual(dictionary.count, 3)
        self.assertEqual(sorted(dictionary.rows.tolist()), [0, 3, 4])
        self.assertTrue(np.array_equal(dictionary.bits()[np.argsort(dictionary.rows)], base))

    def test_size_is_capped(self) -> None:
        codes = np.eye(32, dtype=np.uint8)
        dictionary = LeaderDictionary(similarity_threshold=0.9, max_size=5, use_gpu=False).select(np.packbits(codes, axis=1), 32)
        self.assertEqual(dictionary.count, 5)

    def test_bank_parts_do_not_change_the_dictionary(self) -> None:
        rng = np.random.default_rng(3)
        prototypes = (rng.random((40, 96)) < 0.25).astype(np.uint8)
        codes = prototypes[rng.integers(0, 40, 600)].copy()
        flips = rng.integers(0, 96, (600, 3))
        codes[np.arange(600)[:, None], flips] ^= 1
        packed = np.packbits(codes, axis=1)

        class SmallParts(LeaderDictionary):
            BLOCK = 16
            BANK_PART = 7

        whole = LeaderDictionary(similarity_threshold=0.7, max_size=500, use_gpu=False).select(packed, 96)
        parts = SmallParts(similarity_threshold=0.7, max_size=500, use_gpu=False).select(packed, 96)
        self.assertGreater(whole.count, 7)
        self.assertEqual(sorted(whole.rows.tolist()), sorted(parts.rows.tolist()))

    def test_pairwise_quantiles_are_ordered(self) -> None:
        rng = np.random.default_rng(0)
        bits = (rng.random((60, 128)) < 0.25).astype(np.uint8)
        low, high = PairwiseSimilarity(seed=0).quantiles(bits, (0.5, 0.9), name="test")
        self.assertLess(low, high)
        self.assertGreater(low, 0.0)


class NearestCodeClassifierTest(unittest.TestCase):
    def test_classes_are_found_by_the_most_similar_codes(self) -> None:
        rng = np.random.default_rng(0)
        prototypes = (rng.random((3, 256)) < 0.25).astype(np.uint8)
        base, labels = [], []
        for label, prototype in enumerate(prototypes):
            for _ in range(6):
                code = prototype.copy()
                code[rng.integers(0, 256, 8)] ^= 1
                base.append(code)
                labels.append(label)
        classifier = NearestCodeClassifier(np.packbits(np.asarray(base), axis=1), labels, 256, use_gpu=False)
        accuracy = classifier.accuracy(np.packbits(prototypes, axis=1), [0, 1, 2], neighbours=(1, 5))
        self.assertEqual(accuracy, {1: 1.0, 5: 1.0})


if __name__ == "__main__":
    unittest.main()
