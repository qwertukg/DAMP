from __future__ import annotations

import math
import os
import random
import tempfile
import unittest
from unittest.mock import patch

from damp.activation import (
    ActivationEngine,
    Detector,
    DetectorHierarchyScaleResolver,
    DetectorLayerConfigFactory,
    EmbeddingSaturationResolver,
)
from damp.decode import (
    DecoderConfig,
    DecoderParameterScaleResolver,
    EmbeddingDatabase,
    LabeledEmbedding,
)
from damp.encoding.bitarray import BitArray
from damp.encoding.damp_encoder import ClosedDimension, Detectors, Encoder, EncoderCodeColorizer
from damp.encoding.orientation_encoder import RetinotopicOrientationEncoder
from damp.encoding.stimulus_embedding import StimulusEmbeddingBuilder
from damp.layout.damp_layout import (
    ContinuityLayoutConfig,
    Layout,
    _WeightedIndexSampler,
)
from damp.layout.payload import LayoutPayloadBuilder
from damp.logging import DampLogger, RerunDashboard
from damp.MnistSobelAngleMap import MnistSobelAngleMap
from pinwheel_test import (
    PinwheelExperiment,
    PinwheelPositionGenerator,
    PinwheelQualityEvaluator,
    RetinotopicAngleSurface,
)
from main import (
    ACTIVATION_MAX_EMBEDDING_DENSITY,
    ACTIVATION_DETECTOR_LAYER_SPECS,
    ACTIVATION_OUTPUT_BITS,
    ACTIVATION_SATURATION_LIMIT,
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
    def test_stimulus_colour_merge_prioritizes_position_and_density(self) -> None:
        first = BitArray(8)
        second = BitArray(8)
        for index in (0, 1, 6):
            first.set(index)
        for index in (2, 3, 7):
            second.set(index)
        builder = StimulusEmbeddingBuilder(
            code_length=8,
            spatial_priority_bits=4,
            max_density=0.5,
        )
        embedding = builder.build((first, second))
        self.assertEqual(embedding.count(), 4)
        self.assertTrue(all(embedding[index] for index in range(4)))

    def test_sobel_extractor_preserves_multiple_local_orientations(self) -> None:
        image = __import__("numpy").zeros((28, 28), dtype="float32")
        image[0:4, 1:3] = 1.0
        mean_extractor = MnistSobelAngleMap(grad_threshold=0.01)
        local_extractor = MnistSobelAngleMap(
            grad_threshold=0.01,
            local_orientation_limit=4,
        )
        mean_count = len(mean_extractor.extract(image, 0)[0])
        local_count = len(local_extractor.extract(image, 0)[0])
        self.assertGreater(local_count, mean_count)

    def test_elliptical_detector_coverage_uses_minor_axis(self) -> None:
        code = self._single_bit(8, 0)
        layout = Layout(
            {0.0: [code, code]},
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
        for index, position in enumerate(((1, 1), (2, 1))):
            y, x = position
            layout.grid[y][x] = index
            layout._positions[index] = position
            layout._pos_y[index] = y
            layout._pos_x[index] = x
            layout._row_occupied[y].add(x)
        detector = Detector(
            center=(1.0, 1.0),
            radius=2.0,
            semi_minor_radius=0.5,
            orientation=0.0,
            lambda_threshold=0.5,
            energy=1.0,
            points_count=1,
            output_code=self._single_bit(4, 0),
        )
        engine = ActivationEngine(
            layout,
            [detector],
            lambda_threshold=0.0,
            eta=None,
            energy_threshold=0.0,
            detector_threshold=0.0,
            saturation_limit=4,
            energy_map=[1.0, 1.0],
            use_gpu=False,
        )
        self.assertEqual(len(engine._coverage[0]), 1)

    def test_decoder_parameters_scale_with_embedding_database(self) -> None:
        resolver = DecoderParameterScaleResolver(
            reference_embeddings=100,
            base_top_k=5,
            base_temperature=0.05,
            max_temperature=0.10,
        )
        small = resolver.resolve(100)
        medium = resolver.resolve(200)
        large = resolver.resolve(1000)
        self.assertEqual((small.knn_top_k, small.temperature), (5, 0.05))
        self.assertEqual((medium.knn_top_k, medium.temperature), (5, 0.05))
        self.assertEqual((large.knn_top_k, large.temperature), (5, 0.05))

    def test_embedding_saturation_scales_but_respects_density_limit(self) -> None:
        resolver = EmbeddingSaturationResolver(
            output_bits=2048,
            max_density=0.25,
            base_active_bits=125,
        )
        self.assertEqual(resolver.resolve(1.0), 125)
        self.assertEqual(resolver.resolve(72 / 51), 105)
        self.assertEqual(resolver.resolve(162 / 51), 70)
        self.assertEqual(resolver.resolve(0.01), 512)

    def test_detector_hierarchy_scales_with_layout_extent(self) -> None:
        class SmallLayout:
            width = 51
            height = 51

        class LargeLayout:
            width = 162
            height = 162

        resolver = DetectorHierarchyScaleResolver(
            reference_extent=51,
            base_radius_max=24,
            base_max_detectors=256,
        )
        small = resolver.resolve(SmallLayout())
        large = resolver.resolve(LargeLayout())
        self.assertEqual((small.radius_max, small.max_detectors), (24, 256))
        self.assertEqual((large.radius_max, large.max_detectors), (76, 814))
        self.assertGreater(large.factor, small.factor)
        layers = DetectorLayerConfigFactory(
            radius_min=1,
            radius_max=large.radius_max,
            min_points=3,
        ).build(LargeLayout(), ACTIVATION_DETECTOR_LAYER_SPECS)
        self.assertEqual(
            [layer.activation_radius for layer in layers],
            [57, 47, 39, 32, 26, 19],
        )

    def test_embedding_density_does_not_exceed_configured_limit(self) -> None:
        self.assertLessEqual(
            ACTIVATION_SATURATION_LIMIT / ACTIVATION_OUTPUT_BITS,
            ACTIVATION_MAX_EMBEDDING_DENSITY,
        )

    def test_file_logs_are_filtered_and_only_active_log_is_replaced(self) -> None:
        logger = DampLogger(spawn=False)
        with tempfile.TemporaryDirectory() as directory:
            layout_log = os.path.join(directory, "layout.log")
            evaluate_log = os.path.join(directory, "evaluate.log")
            with open(layout_log, "w", encoding="utf-8") as stream:
                stream.write("previous layout\n")
            logger.configure_file(
                all_paths=(layout_log, evaluate_log),
                active_path=evaluate_log,
                events={"keep"},
            )
            with patch("damp.logging.rr.is_enabled", return_value=False):
                logger.event("drop", section="6.5")
                logger.event("keep", section="6.5", data={"accuracy": 0.9})
            with open(layout_log, encoding="utf-8") as stream:
                self.assertEqual(stream.read(), "previous layout\n")
            with open(evaluate_log, encoding="utf-8") as stream:
                content = stream.read()
            self.assertIn("keep accuracy=0.9", content)
            self.assertNotIn("drop", content)

    def test_logger_uses_completion_title_in_console_and_rerun_text(self) -> None:
        logger = DampLogger(spawn=False)
        with (
            patch.object(logger, "_console_log") as console_log,
            patch("damp.logging.rr.is_enabled", return_value=False),
        ):
            logger.event(
                "pipeline.test.completed",
                title="ТЕСТИРОВАНИЕ ЗАВЕРШЕНО",
                section="6.5",
                data={"elapsed_seconds": 1.25, "average_accuracy": 0.9},
                force=True,
            )
        message = console_log.call_args.args[0]
        self.assertTrue(message.startswith("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО"))
        self.assertIn("elapsed_seconds=1.25", message)
        self.assertIn("average_accuracy=0.9", message)

    def test_detector_layers_resolve_to_mnist_feature_scales(self) -> None:
        class LayoutExtent:
            width = 51
            height = 51

        layers = DetectorLayerConfigFactory(
            radius_min=1,
            radius_max=24,
            min_points=3,
        ).build(
            LayoutExtent(),
            (
                (0.50, 0.35, 3.50),
                (0.60, 0.29, 3.20),
                (0.70, 0.24, 2.65),
                (0.75, 0.20, 2.40),
                (0.80, 0.16, 2.00),
                (0.85, 0.12, 1.30),
            ),
        )
        self.assertEqual(
            [layer.activation_radius for layer in layers],
            [18, 15, 12, 10, 8, 6],
        )
        self.assertEqual(
            [layer.min_points for layer in layers],
            [63, 48, 32, 24, 16, 8],
        )
        self.assertTrue(
            all(
                left.activation_radius > right.activation_radius
                for left, right in zip(layers, layers[1:])
            )
        )

    def test_rerun_event_filter_keeps_only_dashboard_events(self) -> None:
        logger = DampLogger(spawn=False)
        logger._initialized = True
        logger.configure_rerun(
            events={"decoder.evaluation.summary"},
            dashboard_profile=None,
        )
        with (
            patch("damp.logging.rr.is_enabled", return_value=True),
            patch("damp.logging.rr.reset_time"),
            patch("damp.logging.rr.set_time"),
            patch("damp.logging.rr.log") as log,
        ):
            logger.event("decoder.internal", section="6.5", data={"value": 1})
            self.assertEqual(log.call_count, 0)
            logger.event(
                "decoder.evaluation.summary",
                section="6.5",
                data={"accuracy": 0.9},
            )
            self.assertGreater(log.call_count, 0)

    def test_rerun_dashboards_are_explicit(self) -> None:
        self.assertIsNotNone(RerunDashboard.build("layout"))
        self.assertIsNotNone(RerunDashboard.build("evaluate"))

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

    def test_joint_encoder_index_preserves_exhaustive_detector_code(self) -> None:
        encoder = _build_encoder()
        spatial_encoder = encoder._spatial_encoder
        for values in ((0.0, 0.0), (3.5, 2.75), (6.0, 6.0)):
            _, indexed_code = spatial_encoder.encode(*values)
            exhaustive_code = BitArray(spatial_encoder.code_length)
            for detector in spatial_encoder._joint_detectors:
                if detector.is_active(values):
                    exhaustive_code.set(
                        spatial_encoder._bit_map[detector.detector_index],
                        1,
                    )
            self.assertEqual(indexed_code.to01(), exhaustive_code.to01())

    def test_sparse_integer_similarity_matches_binary_cosine(self) -> None:
        first = BitArray(16)
        second = BitArray(16)
        for index in (1, 4, 9):
            first.set(index)
        for index in (1, 7, 9, 12):
            second.set(index)
        layout = Layout(
            {0.0: [first, second]},
            grid_size=2,
            empty_ratio=0.0,
            similarity="cosine",
            lambda_threshold=0.0,
            eta=None,
            precompute_similarity=False,
            use_gpu=False,
            colorizer=EncoderCodeColorizer(0.0, 300.0),
        )
        expected = 2.0 / math.sqrt(3.0 * 4.0)
        self.assertTrue(math.isclose(layout._similarity_value(0, 1), expected))

    def test_cached_weight_sampler_preserves_random_choices_sequence(self) -> None:
        population = tuple(range(7))
        weights = (0.5, 1.5, 0.25, 4.0, 2.0, 0.75, 3.0)
        reference_rng = random.Random(19)
        indexed_rng = random.Random(19)
        expected = [
            reference_rng.choices(population, weights=weights, k=1)[0]
            for _ in range(100)
        ]
        sampler = _WeightedIndexSampler(population, weights)
        actual = [sampler.sample(indexed_rng) for _ in range(100)]
        self.assertEqual(actual, expected)

    def test_layout_checkpoint_restores_rng_and_next_swaps(self) -> None:
        codes = {
            float(index): [self._single_bit(24, index)]
            for index in range(12)
        }

        def build() -> Layout:
            return Layout(
                codes,
                grid_size=5,
                empty_ratio=0.0,
                similarity="cosine",
                lambda_threshold=0.0,
                eta=None,
                seed=7,
                use_gpu=False,
                colorizer=EncoderCodeColorizer(0.0, 300.0),
            )

        original = build()
        original.run(
            steps=3,
            pairs_per_step=20,
            pair_radius=3,
            mode="long",
            min_swap_ratio=0.0,
            log_visuals=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "phase.json")
            original.save_json(path)
            restored = build()
            self.assertTrue(restored.restore_positions_json(path))
            for layout in (original, restored):
                layout.run(
                    steps=2,
                    pairs_per_step=20,
                    pair_radius=3,
                    mode="long",
                    min_swap_ratio=0.0,
                    log_visuals=False,
                )
            self.assertEqual(original.positions(), restored.positions())

    def test_continuity_swap_gain_matches_affected_edge_energy(self) -> None:
        codes: dict[float, list[BitArray]] = {}
        for index in range(10):
            code = BitArray(16)
            code.set(index)
            code.set((index + 1) % 10)
            codes[float(index * 18)] = [code]
        layout = Layout(
            codes,
            grid_size=4,
            empty_ratio=0.0,
            similarity="cosine",
            lambda_threshold=0.0,
            eta=10.0,
            seed=5,
            use_gpu=False,
            colorizer=EncoderCodeColorizer(0.0, 180.0),
        )
        config = ContinuityLayoutConfig(
            sweeps=1,
            defect_points=4,
            candidates_per_point=4,
            cyclic_period=180.0,
            cyclic_sharp_threshold=15.0,
        )

        def reference(first: tuple[int, int], second: tuple[int, int]) -> float:
            first_index = layout.grid[first[0]][first[1]]
            second_index = layout.grid[second[0]][second[1]]
            assert first_index is not None and second_index is not None
            edges = set()
            for point in (first, second):
                for neighbour in layout._continuity_neighbours(point):
                    edges.add(tuple(sorted((point, neighbour))))

            def after(position: tuple[int, int]) -> int | None:
                if position == first:
                    return second_index
                if position == second:
                    return first_index
                return layout.grid[position[0]][position[1]]

            before = 0.0
            after_total = 0.0
            for start, end in edges:
                start_before = layout.grid[start[0]][start[1]]
                end_before = layout.grid[end[0]][end[1]]
                if start_before is not None and end_before is not None:
                    before += layout._continuity_pair_score(
                        start_before, end_before, config
                    )
                start_after = after(start)
                end_after = after(end)
                if start_after is not None and end_after is not None:
                    after_total += layout._continuity_pair_score(
                        start_after, end_after, config
                    )
            return after_total - before

        positions = layout.positions()
        for first, second in zip(positions[:4], positions[-4:]):
            self.assertTrue(
                math.isclose(
                    layout._continuity_swap_gain(first, second, config),
                    reference(first, second),
                    abs_tol=1e-12,
                )
            )

    def test_encoded_payload_round_trip_preserves_codes(self) -> None:
        payload = LayoutPayloadBuilder(
            similarity=None,
            lambda_threshold=None,
            eta=None,
        )
        payload.add_code(label=3, hue=42.0, code=self._single_bit(12, 5))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "encoded.json")
            payload.save_base(path)
            restored = LayoutPayloadBuilder.load_base(path)
        self.assertEqual(len(restored.records), 1)
        self.assertEqual(restored.records[0].label, "3")
        self.assertEqual(restored.records[0].hue, 42.0)
        self.assertEqual(restored.records[0].code.to01(), payload.records[0].code.to01())

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
