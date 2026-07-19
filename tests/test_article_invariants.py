from __future__ import annotations

import math
import unittest

from damp.activation import ActivationEngine, Detector
from damp.decode import DecoderConfig, EmbeddingDatabase, LabeledEmbedding
from damp.encoding.bitarray import BitArray
from damp.encoding.damp_encoder import ClosedDimension, Detectors, Encoder, EncoderCodeColorizer
from damp.encoding.orientation_encoder import RetinotopicOrientationEncoder
from damp.layout.damp_layout import Layout
from pinwheel_test import (
    PinwheelExperiment,
    PinwheelPositionGenerator,
    PinwheelQualityEvaluator,
    RetinotopicAngleSurface,
)
from main import (
    LAYOUT_PSEUDO_COLOR_MAX_HUE,
    LAYOUT_PSEUDO_COLOR_MIN_HUE,
    PINWHEEL_ANGLE_COUNT,
    PINWHEEL_COUNT,
    PINWHEEL_LAMBDA_THRESHOLD,
    PINWHEEL_POSITION_GENERATOR_CANDIDATES,
    PINWHEEL_POSITION_GENERATOR_MARGIN,
    PINWHEEL_POSITION_MAX,
    PINWHEEL_POSITION_MIN,
    PINWHEEL_SHORT_ANGLE_LAYERS,
    PINWHEEL_SHORT_SPATIAL_JOINT_LAYERS,
    _build_encoder,
)


class ArticleInvariantTests(unittest.TestCase):
    @staticmethod
    def _single_bit(length: int, index: int) -> BitArray:
        code = BitArray(length)
        code.set(index)
        return code

    @staticmethod
    def _shared_boundary(
        surface: RetinotopicAngleSurface,
        first_position: int,
        second_position: int,
    ) -> list[tuple[object, object]]:
        by_position = {
            position: {
                (round(sample.position_x, 12), round(sample.position_y, 12)): sample
                for sample in surface.samples
                if sample.position_index == position
            }
            for position in (first_position, second_position)
        }
        shared = sorted(
            set(by_position[first_position]) & set(by_position[second_position])
        )
        return [
            (
                by_position[first_position][coordinate],
                by_position[second_position][coordinate],
            )
            for coordinate in shared
        ]

    def test_closed_coordinate_is_continuous_at_boundary(self) -> None:
        encoder = Encoder(
            ClosedDimension("angle", (0.0, 360.0), [Detectors(36, 0.5)]),
            random_bit=True,
            random_seed=0,
        )
        _, first = encoder.encode(0.0)
        _, last = encoder.encode(359.0)
        self.assertGreater(first.common(last), 0)

    def test_main_uses_shared_axial_orientation_encoder(self) -> None:
        encoder = _build_encoder()
        self.assertIsInstance(encoder, RetinotopicOrientationEncoder)
        values_zero, code_zero = encoder.encode(0.0, 3.0, 3.0)
        values_half_turn, code_half_turn = encoder.encode(180.0, 3.0, 3.0)
        self.assertTrue(math.isclose(values_zero[0], values_half_turn[0]))
        self.assertEqual(code_zero.to01(), code_half_turn.to01())
        descriptor = encoder.describe(45.0, center_index=24)
        spatial_bits, angular_bits = encoder.active_component_counts(descriptor)
        self.assertGreater(spatial_bits, angular_bits)

    def test_point_energy_uses_euclidean_distance(self) -> None:
        code = self._single_bit(8, 0)
        layout = Layout(
            {0.0: [code, code, code]},
            grid_size=3,
            empty_ratio=0.0,
            lambda_threshold=0.0,
            eta=None,
            use_gpu=False,
            colorizer=EncoderCodeColorizer(
                LAYOUT_PSEUDO_COLOR_MIN_HUE,
                LAYOUT_PSEUDO_COLOR_MAX_HUE,
            ),
        )
        layout.grid = [[None] * 3 for _ in range(3)]
        layout._row_occupied = [set() for _ in range(3)]
        for index, x in enumerate((0, 1, 2)):
            layout.grid[0][x] = index
            layout._positions[index] = (0, x)
            layout._pos_y[index] = 0
            layout._pos_x[index] = x
            layout._row_occupied[0].add(x)
        energy = layout._compute_point_energies(3)[0]
        self.assertTrue(math.isclose(energy, 1.5, rel_tol=2e-6))

    def test_detector_activation_ratio_is_not_clipped(self) -> None:
        stimulus = self._single_bit(8, 0)
        layout = Layout(
            {0.0: [stimulus]},
            grid_size=1,
            empty_ratio=0.0,
            lambda_threshold=0.0,
            eta=None,
            use_gpu=False,
            colorizer=EncoderCodeColorizer(
                LAYOUT_PSEUDO_COLOR_MIN_HUE,
                LAYOUT_PSEUDO_COLOR_MAX_HUE,
            ),
        )
        output = self._single_bit(4, 0)
        detector = Detector(
            center=(0.0, 0.0),
            radius=1.0,
            lambda_threshold=0.5,
            energy=0.1,
            points_count=1,
            output_code=output,
        )
        engine = ActivationEngine(
            layout,
            [detector],
            lambda_threshold=0.0,
            eta=None,
            energy_threshold=0.0,
            detector_threshold=0.0,
            saturation_limit=4,
            energy_map=[1.0],
            use_gpu=False,
        )
        result = engine.activate([stimulus])
        self.assertGreater(result.active_detectors[0].activation_level, 1.0)

    def test_binary_j2_equals_jaccard(self) -> None:
        stored = self._single_bit(4, 0)
        query = BitArray(4)
        query.set(0)
        query.set(1)
        database = EmbeddingDatabase(
            [LabeledEmbedding(label=0, embedding=stored)],
            DecoderConfig(
                similarity="j2",
                temperature=1.0,
                min_confidence=0.0,
                expected_classes=(0,),
                knn_top_k=1,
            ),
        )
        self.assertEqual(database._similarity_value(query, 2, stored, 1), 0.5)

    def test_pseudocolour_uses_bit_order(self) -> None:
        red = Encoder.color_from_code(
            self._single_bit(8, 0),
            min_hue=LAYOUT_PSEUDO_COLOR_MIN_HUE,
            max_hue=LAYOUT_PSEUDO_COLOR_MAX_HUE,
        )
        violet = Encoder.color_from_code(
            self._single_bit(8, 7),
            min_hue=LAYOUT_PSEUDO_COLOR_MIN_HUE,
            max_hue=LAYOUT_PSEUDO_COLOR_MAX_HUE,
        )
        self.assertGreater(red[0], red[2])
        self.assertGreater(violet[2], violet[1])

    def test_energy_stability_stops_both_layout_phases(self) -> None:
        for mode in ("long", "short"):
            with self.subTest(mode=mode):
                code = self._single_bit(8, 0)
                layout = Layout(
                    {0.0: [code]},
                    grid_size=1,
                    empty_ratio=0.0,
                    lambda_threshold=0.0,
                    eta=None,
                    use_gpu=False,
                    colorizer=EncoderCodeColorizer(
                        LAYOUT_PSEUDO_COLOR_MIN_HUE,
                        LAYOUT_PSEUDO_COLOR_MAX_HUE,
                    ),
                )
                layout.step = lambda **_: 0  # type: ignore[method-assign]
                layout.average_local_energy = (  # type: ignore[method-assign]
                    lambda *_, **__: 1.0
                )
                layout.run(
                    steps=10,
                    pairs_per_step=1,
                    pair_radius=1,
                    mode=mode,
                    local_radius=1 if mode == "short" else None,
                    min_swap_ratio=0.0,
                    energy_radius=1,
                    energy_stability_window=2,
                    energy_stability_delta=0.001,
                    energy_stability_every=1,
                    log_visuals=False,
                )
                self.assertEqual(layout.last_steps, 3)

    def test_pinwheel_surface_is_nine_positions_times_angles(self) -> None:
        surface = RetinotopicAngleSurface()
        self.assertEqual(len(surface.samples), PINWHEEL_COUNT * PINWHEEL_ANGLE_COUNT)
        self.assertEqual(len({sample.position_index for sample in surface.samples}), PINWHEEL_COUNT)
        self.assertEqual(len({sample.angle for sample in surface.samples}), PINWHEEL_ANGLE_COUNT)
        self.assertEqual(len({sample.radius for sample in surface.samples}), 1)

    def test_pinwheel_winding_components_use_diagonal_connectivity(self) -> None:
        self.assertEqual(
            PinwheelQualityEvaluator._winding_component_count(
                {(0, 0), (1, 1), (4, 4)}
            ),
            2,
        )

    def test_pinwheel_color_uses_only_angular_encoder_bits(self) -> None:
        surface = RetinotopicAngleSurface()
        encoder = PinwheelExperiment.build_encoder(
            PINWHEEL_SHORT_SPATIAL_JOINT_LAYERS,
            PINWHEEL_SHORT_ANGLE_LAYERS,
        )
        first, same_encoded_angle_other_position = self._shared_boundary(surface, 0, 1)[0]
        first_code = encoder.encode_sample(first)
        other_code = encoder.encode_sample(same_encoded_angle_other_position)
        self.assertEqual(first_code.to01(), other_code.to01())
        self.assertTrue(
            math.isclose(
                first.encoded_angle,
                same_encoded_angle_other_position.encoded_angle,
                abs_tol=1e-9,
            )
        )
        self.assertEqual(
            encoder.colorizer.color(first_code),
            encoder.colorizer.color(other_code),
        )
        colors = {
            encoder.colorizer.color(encoder.encode_sample(sample))
            for sample in surface.samples
        }
        self.assertGreaterEqual(len(colors), PINWHEEL_ANGLE_COUNT)

    def test_pinwheel_input_phases_match_shared_boundaries(self) -> None:
        surface = RetinotopicAngleSurface()
        horizontal = self._shared_boundary(surface, 0, 1)
        vertical = self._shared_boundary(surface, 0, 3)
        self.assertGreaterEqual(len(horizontal), 1)
        self.assertGreaterEqual(len(vertical), 1)
        for first, second in horizontal + vertical:
            self.assertTrue(
                math.isclose(first.encoded_angle, second.encoded_angle, abs_tol=1e-9)
            )
            self.assertEqual(first.radius_index, second.radius_index)
            self.assertTrue(math.isclose(first.position_x, second.position_x))
            self.assertTrue(math.isclose(first.position_y, second.position_y))
        horizontal_left, horizontal_right = horizontal[0]
        self.assertIn(horizontal_left.chirality, (-1, 1))
        self.assertIn(horizontal_right.chirality, (-1, 1))

        self.assertTrue(
            all(
                -1e-12 <= sample.position_x <= 1.0 + 1e-12
                and -1e-12 <= sample.position_y <= 1.0 + 1e-12
                for sample in surface.samples
            )
        )

    def test_pinwheel_field_is_continuous_for_arbitrary_centers(self) -> None:
        center_sets = (
            ((0.19, 0.22), (0.81, 0.76)),
            ((0.14, 0.18), (0.73, 0.21), (0.43, 0.84)),
            ((0.12, 0.31), (0.66, 0.13), (0.88, 0.69), (0.37, 0.82)),
        )
        for centers in center_sets:
            with self.subTest(centers=len(centers)):
                surface = RetinotopicAngleSurface(centers)
                self.assertEqual(
                    len(surface.samples), len(centers) * PINWHEEL_ANGLE_COUNT
                )
                for position in range(len(centers)):
                    samples = surface.samples[
                        position * PINWHEEL_ANGLE_COUNT :
                        (position + 1) * PINWHEEL_ANGLE_COUNT
                    ]
                    deltas = [
                        PinwheelQualityEvaluator._angle_delta(
                            samples[index].encoded_angle,
                            samples[(index + 1) % len(samples)].encoded_angle,
                        )
                        for index in range(len(samples))
                    ]
                    self.assertLess(max(deltas), 15.0)

    def test_pinwheel_position_generator_is_seeded_and_non_symmetric(self) -> None:
        generator = PinwheelPositionGenerator(
            minimum=PINWHEEL_POSITION_MIN,
            maximum=PINWHEEL_POSITION_MAX,
            margin=PINWHEEL_POSITION_GENERATOR_MARGIN,
            candidates=PINWHEEL_POSITION_GENERATOR_CANDIDATES,
        )
        first = generator.generate(4, seed=17)
        second = generator.generate(4, seed=17)
        different = generator.generate(4, seed=18)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertEqual(len(set(first)), 4)
        self.assertTrue(
            all(
                PINWHEEL_POSITION_MIN < x < PINWHEEL_POSITION_MAX
                and PINWHEEL_POSITION_MIN < y < PINWHEEL_POSITION_MAX
                for x, y in first
            )
        )

    def test_pinwheel_position_bits_outweigh_angle_bits(self) -> None:
        surface = RetinotopicAngleSurface()
        encoder = PinwheelExperiment.build_encoder(
            PINWHEEL_SHORT_SPATIAL_JOINT_LAYERS,
            PINWHEEL_SHORT_ANGLE_LAYERS,
            spatial_bounds=surface.spatial_bounds,
        )
        for sample in surface.samples[::PINWHEEL_ANGLE_COUNT // 8]:
            spatial_bits, angular_bits = encoder.active_sample_component_counts(sample)
            self.assertGreater(spatial_bits, angular_bits)

    def test_pinwheel_codes_are_continuous_across_shared_boundaries(self) -> None:
        surface = RetinotopicAngleSurface()
        encoder = PinwheelExperiment.build_encoder(
            PINWHEEL_SHORT_SPATIAL_JOINT_LAYERS,
            PINWHEEL_SHORT_ANGLE_LAYERS,
        )

        def cosine(first_sample: object, second_sample: object) -> float:
            first = encoder.encode_sample(first_sample)
            second = encoder.encode_sample(second_sample)
            return first.common(second) / math.sqrt(first.count() * second.count())

        horizontal_pair = self._shared_boundary(surface, 0, 1)[0]
        vertical_pair = self._shared_boundary(surface, 0, 3)[0]
        horizontal_boundary = cosine(*horizontal_pair)
        vertical_boundary = cosine(*vertical_pair)
        far_same_angle = cosine(
            surface.samples[0],
            surface.samples[8 * PINWHEEL_ANGLE_COUNT],
        )

        self.assertGreater(horizontal_boundary, PINWHEEL_LAMBDA_THRESHOLD)
        self.assertGreater(vertical_boundary, PINWHEEL_LAMBDA_THRESHOLD)
        self.assertGreater(horizontal_boundary, far_same_angle)
        self.assertGreater(vertical_boundary, far_same_angle)


if __name__ == "__main__":
    unittest.main()
