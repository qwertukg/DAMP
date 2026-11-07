from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import main
from damp.activation.activation import ActivationResult, EmbeddingSaturationResolver
from damp.activation.detector_space import DetectorSpace
from damp.activation.spatial_hierarchy import SpatialDetectorHierarchy
from damp.encoding.bitarray import BitArray
from damp.encoding.spatial_composition import SpatialFeature
from damp.layout.pinwheel_topology import ContinuousPinwheelTopology
from damp.logging import LOGGER


class SpatialHierarchyTests(unittest.TestCase):
    def setUp(self) -> None:
        log_patch = patch.object(LOGGER, "event")
        log_patch.start()
        self.addCleanup(log_patch.stop)

    def test_mnist_opposite_directions_are_distinct_at_every_position(self) -> None:
        encoder = main._build_encoder()
        for y in range(main.MNIST_POSITION_GRID_SIZE):
            for x in range(main.MNIST_POSITION_GRID_SIZE):
                with self.subTest(x=x, y=y):
                    _, first = encoder.encode(30.0, x, y)
                    _, opposite = encoder.encode(210.0, x, y)
                    self.assertNotEqual(first.to01(), opposite.to01())
                    _, wrapped = encoder.encode(390.0, x, y)
                    _, negative = encoder.encode(-150.0, x, y)
                    self.assertEqual(first.to01(), wrapped.to01())
                    self.assertEqual(opposite.to01(), negative.to01())

    def test_mnist_direction_codes_remain_similar_across_full_turn_boundary(self) -> None:
        encoder = main._build_encoder()
        _, zero = encoder.encode(0.0, 3, 3)
        _, near = encoder.encode(359.0, 3, 3)
        _, opposite = encoder.encode(180.0, 3, 3)
        self.assertGreater(zero.common(near), zero.common(opposite))

    def test_angular_layers_cover_full_circle_at_four_scales(self) -> None:
        encoder = main._build_encoder()
        layers = encoder._angle_encoder._layer_indices_by_dimension[0]
        self.assertEqual([len(layer.windows) for layer in layers], [360, 72, 12, 4])
        for layer, width in zip(layers, (4 / 3, 20 / 3, 40.0, 120.0)):
            self.assertAlmostEqual(layer.windows[0].half_width * 2, width)
            spacing = layer.windows[1].center - layer.windows[0].center
            self.assertAlmostEqual((width - spacing) / width, 0.25)
            for angle in (0.0, 0.5, 45.0, 90.0, 179.5, 270.0, 359.5):
                with self.subTest(width=width, angle=angle):
                    active = sum(d.is_active(angle) for d in layer.windows)
                    self.assertGreaterEqual(active, 1)
                    self.assertLessEqual(active, 2)
        _, code = encoder.encode(90.0, 1, 1)
        self.assertEqual(encoder.angular_code_length, 448)
        self.assertEqual(len(code), 1477)
        self.assertEqual(sum(bit >= encoder.spatial_code_length for bit in code.active_indices()), 8)

    def test_angular_color_keeps_finest_layer_resolution(self) -> None:
        encoder = main._build_encoder()
        self.assertEqual(encoder.colorizer.start_bit, encoder.spatial_code_length)
        self.assertEqual(encoder.colorizer.bit_length, 360)
        _, first = encoder.encode(90.0, 1, 1)
        _, near = encoder.encode(100.0, 1, 1)
        self.assertNotEqual(encoder.colorizer.color(first), encoder.colorizer.color(near))

    def test_topology_scales_polar_angles_to_configured_period(self) -> None:
        for period in (180.0, 360.0):
            with self.subTest(period=period):
                topology = ContinuousPinwheelTopology(
                    [(0.0, 0.0)], angle_period=period,
                    radius_factor=0.5, isolated_radius=0.5,
                )
                self.assertAlmostEqual(topology.orientation_at(0.0, 1.0), period / 4)
                self.assertAlmostEqual(topology.orientation_at(-1.0, 0.0), period / 2)
                upper = topology.orientation_at(-1.0, 1e-9)
                lower = topology.orientation_at(-1.0, -1e-9)
                self.assertAlmostEqual(upper, lower, places=6)

    def test_windows_preserve_absolute_positions_and_ignore_distant_stimuli(self) -> None:
        encoder = main._build_window_encoder(main._build_encoder())
        original = encoder.encode([(0.0, 0, 0), (90.0, 1, 1)])
        translated = encoder.encode([(0.0, 2, 2), (90.0, 3, 3)])
        first = next(window for window in original if (window.x, window.y) == (1, 1))
        second = next(window for window in translated if (window.x, window.y) == (3, 3))
        self.assertEqual(first.measurements, ((0.0, 0, 0), (90.0, 1, 1)))
        self.assertEqual(second.measurements, ((0.0, 2, 2), (90.0, 3, 3)))
        for first_code, second_code in zip(first.codes, second.codes):
            self.assertNotEqual(first_code.to01(), second_code.to01())
        extra = encoder.encode([(0.0, 0, 0), (90.0, 1, 1), (45.0, 6, 6)])
        first_extra = next(window for window in extra if (window.x, window.y) == (1, 1))
        self.assertEqual(first.measurements, first_extra.measurements)

    def test_overlapping_windows_keep_the_same_absolute_stimulus_code(self) -> None:
        orientation_encoder = main._build_encoder()
        encoder = main._build_window_encoder(orientation_encoder)
        measurement = (45.0, 2, 2)
        windows = encoder.encode([measurement])
        self.assertEqual(len(windows), 4)
        _, expected = orientation_encoder.encode(*measurement)
        for window in windows:
            self.assertEqual(window.measurements, (measurement,))
            self.assertEqual(window.codes[0].to01(), expected.to01())

    def test_windows_cover_absolute_image_boundaries(self) -> None:
        encoder = main._build_window_encoder(main._build_encoder())
        measurements = [(0.0, x, y) for y in range(7) for x in range(7)]
        windows = encoder.encode(measurements)
        self.assertEqual(
            {measurement for window in windows for measurement in window.measurements},
            set(measurements),
        )

    def test_composition_binds_each_feature_to_its_own_position(self) -> None:
        encoder = main._build_composition_encoder()
        a = BitArray.from_indices(main.ACTIVATION_OUTPUT_BITS, (3, 7))
        b = BitArray.from_indices(main.ACTIVATION_OUTPUT_BITS, (13, 17))
        features = [SpatialFeature(1, 1, a), SpatialFeature(5, 5, b)]
        swapped = [SpatialFeature(1, 1, b), SpatialFeature(5, 5, a)]
        code = encoder.build(features)
        self.assertNotEqual(code.to01(), encoder.build(swapped).to01())
        self.assertEqual(code.to01(), encoder.build(list(reversed(features))).to01())
        vertical = encoder.build([SpatialFeature(1, 1, a), SpatialFeature(1, 5, a)])
        horizontal = encoder.build([SpatialFeature(1, 1, a), SpatialFeature(5, 1, a)])
        self.assertNotEqual(vertical.to01(), horizontal.to01())

    def test_neighbouring_positions_keep_feature_similarity(self) -> None:
        encoder = main._build_composition_encoder()
        feature = BitArray.from_indices(main.ACTIVATION_OUTPUT_BITS, (3, 7))
        first = encoder.build([SpatialFeature(1, 1, feature)])
        near = encoder.build([SpatialFeature(2, 1, feature)])
        far = encoder.build([SpatialFeature(5, 5, feature)])
        self.assertGreater(first.common(near), first.common(far))

    def test_every_local_window_reaches_level_two_separately(self) -> None:
        window_encoder = main._build_window_encoder(main._build_encoder())
        engines = [Mock() for _ in range(main.DETECTOR_SPACE_COUNT)]
        for index, engine in enumerate(engines):
            code = BitArray.from_indices(main.ACTIVATION_OUTPUT_BITS, (index,))
            engine.activate.return_value = ActivationResult([], [], [], code)
        model = SpatialDetectorHierarchy(
            window_encoder=window_encoder,
            composition_encoder=main._build_composition_encoder(),
            spaces=[DetectorSpace(index, engine) for index, engine in enumerate(engines, start=1)],
        )
        result = model.activate_measurements([(0.0, 1, 1), (90.0, 5, 5)])
        self.assertEqual(len(result.local_codes), 2)
        self.assertEqual(engines[0].activate.call_count, 2)
        self.assertEqual(engines[1].activate.call_count, 2)
        self.assertEqual(engines[2].activate.call_count, 1)
        self.assertEqual([(part.x, part.y) for part in result.parts], [(1, 1), (5, 5)])
        for call in engines[1].activate.call_args_list:
            self.assertEqual(len(call.args[0]), 1)
        self.assertIs(engines[2].activate.call_args.args[0][0], result.composition)
        self.assertIs(result.embedding, engines[2].activate.return_value.embedding)

    def test_training_rejects_test_split(self) -> None:
        pipeline = main.SpatialHierarchyPipeline(
            train_count=100, sample_count=100, encoder=main._build_encoder(),
            extractor=Mock(), checkpoint_store=Mock(), final_embedding_store=Mock(),
        )
        with self.assertRaisesRegex(ValueError, "training split"):
            pipeline._samples(Mock(train=False), None)

    def test_manifest_rejects_different_spatial_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            primary = str(Path(directory) / "level1.json")
            config = main.SpatialHierarchyPipeline.config()
            config["stride"] += 1
            Path(primary + main.HIERARCHY_MANIFEST_SUFFIX).write_text(
                json.dumps({"config": config, "split": "train"}), encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "параметры иерархии изменились"):
                main.SpatialHierarchyPipeline.load_manifest(primary)

    def test_decoder_single_and_batch_use_same_spatial_path(self) -> None:
        from damp.decode.mnist_decoder import MnistDecoder

        model = Mock()
        code = BitArray.from_indices(main.ACTIVATION_OUTPUT_BITS, (7, 11))
        model.activate_measurements.return_value.activation = ActivationResult([], [], [], code)
        model.activate_measurements.return_value.embedding = code
        model.spaces = (Mock(), Mock(), Mock())
        extractor = Mock()
        measurements = [(90.0, 1, 1)]
        extractor.extract.return_value = {-1: measurements}
        extractor.extract_many.return_value = [{-1: measurements}]
        database = Mock(code_length=main.ACTIVATION_OUTPUT_BITS, expected_classes=tuple(range(10)))
        decoder = MnistDecoder(
            encoder=Mock(), extractor=extractor, activation_engine=None,
            spatial_hierarchy=model, embedding_db=database,
            evaluation_confidence_bins=10, evaluation_error_examples=16,
            stimulus_embedding_builder=Mock(), structural_activation_interval=100,
        )
        image = np.zeros((28, 28), dtype=np.float32)
        single = decoder.embed_image(image).embedding
        batch = decoder.embed_batch([image])[0]
        self.assertEqual(single.to01(), batch.to01())
        self.assertEqual(model.activate_measurements.call_count, 2)
        for call in model.activate_measurements.call_args_list:
            self.assertEqual(call.args[0], measurements)

    def test_small_composition_space_keeps_only_a_fraction_of_detectors(self) -> None:
        resolver = EmbeddingSaturationResolver(
            output_bits=main.ACTIVATION_OUTPUT_BITS,
            max_density=main.ACTIVATION_MAX_EMBEDDING_DENSITY,
            base_active_bits=main.ACTIVATION_TARGET_ACTIVE_BITS,
        )
        budget = resolver.resolve(
            0.25, detector_count=41, active_fraction=main.HIERARCHY_COMPOSITION_ACTIVE_FRACTION,
        )
        self.assertEqual(budget, 11)
        self.assertLess(budget, 41)
        self.assertEqual(resolver.resolve(0.25), 250)


if __name__ == "__main__":
    unittest.main()
