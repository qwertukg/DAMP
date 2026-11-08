"""Проверяемые инварианты статьи DAMP для Encoder, Layout и детекторов."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("RERUN", "off")

from damp.activation import (  # noqa: E402
    ActivationEngine,
    DbscanClusterer,
    Detector,
    DetectorHierarchyBuilder,
    DetectorLayerConfig,
)
from damp.encoding.bitarray import BitArray  # noqa: E402
from damp.encoding.damp_encoder import (  # noqa: E402
    ClosedDimension,
    Detectors,
    Encoder,
    EncoderCodeColorizer,
    OpenedDimension,
    geometric_detector_layers,
)
from damp.layout.damp_layout import Layout  # noqa: E402
from damp.layout.payload import LayoutEncodedCode  # noqa: E402
from damp.logging import LOGGER  # noqa: E402
import main  # noqa: E402

LOGGER.configure_intervals({}, default_interval=10**9)


def _cosine(a: BitArray, b: BitArray) -> float:
    return a.common(b) / math.sqrt(a.count() * b.count())


def _angle_encoder(seed: int = 0, *, layers: int = 12, narrowest: int = 90, length: int = 128) -> Encoder:
    return Encoder(
        ClosedDimension(
            "Angle",
            (0.0, 360.0),
            geometric_detector_layers(
                layers_count=layers, widest_detectors=2, narrowest_detectors=narrowest, overlap=0.4
            ),
            code_length=length,
        ),
        random_seed=seed,
    )


class EncoderInvariantTests(unittest.TestCase):
    """§2.2.3, §4.1, §4.2.1, §4.4, §4.5: случайные биты, конкатенация, замкнутость."""

    def test_each_detector_gets_one_random_bit_inside_its_segment(self) -> None:
        encoder = Encoder(
            ClosedDimension("Angle", (0.0, 360.0), [Detectors(4, 0.4), Detectors(8, 0.4)], code_length=32),
            OpenedDimension("X", (0.0, 6.0), [Detectors(2, 0.4), Detectors(7, 0.4)], code_length=16),
            random_seed=3,
        )
        self.assertEqual(encoder.code_length, 48)
        self.assertEqual(encoder.segment("Angle"), (0, 32))
        self.assertEqual(encoder.segment("X"), (32, 16))
        for window in encoder.detectors:
            start, length = encoder.segment(window.dimension_index)
            self.assertTrue(start <= window.bit < start + length)
            self.assertEqual(window.bit, start + window.local_bit)
        bits = [window.bit for window in encoder.detectors if window.dimension_index == 0]
        self.assertNotEqual(bits, sorted(bits), "биты назначаются случайно, а не по порядку")

    def test_random_bits_are_reproducible_by_seed_and_differ_between_seeds(self) -> None:
        first = _angle_encoder(seed=7)
        second = _angle_encoder(seed=7)
        third = _angle_encoder(seed=8)
        self.assertEqual([w.bit for w in first.detectors], [w.bit for w in second.detectors])
        self.assertNotEqual([w.bit for w in first.detectors], [w.bit for w in third.detectors])
        self.assertEqual(first.encode(42.0)[1].to01(), second.encode(42.0)[1].to01())

    def test_bit_collisions_are_allowed_and_measured_density_matches_estimate(self) -> None:
        encoder = _angle_encoder(seed=0, layers=22, narrowest=180)
        codes = [encoder.encode(float(angle))[1] for angle in range(360)]
        measured = sum(code.count() for code in codes) / (360 * encoder.code_length)
        self.assertGreater(len(encoder.detectors), encoder.code_length, "детекторов больше, чем битов")
        self.assertAlmostEqual(measured, encoder.expected_density(), delta=0.05)
        self.assertLessEqual(measured, 0.25, "плотность не выше 25% (§2)")
        _, _, shared = encoder.encode_indices(10.0)
        self.assertGreaterEqual(shared, 0)

    def test_geometric_layers_grow_from_widest_to_narrowest(self) -> None:
        layers = geometric_detector_layers(layers_count=6, widest_detectors=2, narrowest_detectors=90, overlap=0.4)
        counts = [layer.count for layer in layers]
        self.assertEqual(counts[0], 2)
        self.assertEqual(counts[-1], 90)
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(len(set(counts)), len(counts), "у каждого слоя своё число детекторов")
        with self.assertRaises(ValueError):
            geometric_detector_layers(layers_count=5, widest_detectors=2, narrowest_detectors=4, overlap=0.4)

    def test_closed_dimension_is_continuous_across_the_period_boundary(self) -> None:
        encoder = _angle_encoder(seed=1)
        near_zero = encoder.encode(0.5)[1]
        near_full = encoder.encode(359.5)[1]
        opposite = encoder.encode(180.0)[1]
        self.assertGreater(_cosine(near_zero, near_full), 0.9)
        self.assertGreater(_cosine(near_zero, near_full), _cosine(near_zero, opposite))
        self.assertEqual(encoder.encode(370.0)[1].to01(), encoder.encode(10.0)[1].to01())

    def test_similarity_decreases_with_angle_distance_but_keeps_long_range_overlap(self) -> None:
        encoder = _angle_encoder(seed=0, layers=22, narrowest=180)
        codes = [encoder.encode(float(angle))[1] for angle in range(360)]

        def mean_similarity(delta: int) -> float:
            return sum(_cosine(codes[a], codes[(a + delta) % 360]) for a in range(360)) / 360

        profile = [mean_similarity(delta) for delta in (1, 5, 20, 60, 180)]
        self.assertEqual(profile, sorted(profile, reverse=True))
        self.assertGreater(profile[-1], 0.0, "самые широкие слои дают дальний порядок (§4.5)")

    def test_long_range_order_diagnostic_distinguishes_ordered_and_flat_codes(self) -> None:
        """§4.5, §5.13.1: диагностика дальнего порядка должна видеть плоский хвост профиля сходства."""
        from damp.encoding.damp_encoder import OpenedDimension, geometric_detector_layers

        ordered = Encoder(
            OpenedDimension("X", (0.0, 6.0), geometric_detector_layers(layers_count=7, widest_detectors=2,
                                                                       narrowest_detectors=10, overlap=0.8),
                            code_length=256),
            random_seed=0, log_every=10 ** 9, layer_phase=True,
        )
        profile, ok = ordered.long_range_order("X", step=1.0)
        self.assertTrue(ok, profile)
        self.assertEqual([round(d) for d, _ in profile], [1, 2, 3, 4, 5, 6])
        self.assertTrue(all(profile[k + 1][1] < profile[k][1] for k in range(len(profile) - 1)), profile)
        flat = Encoder(
            OpenedDimension("X", (0.0, 6.0), geometric_detector_layers(layers_count=2, widest_detectors=12,
                                                                       narrowest_detectors=13, overlap=0.0),
                            code_length=256),
            random_seed=0, log_every=10 ** 9, layer_phase=True,
        )
        _, flat_ok = flat.long_range_order("X", step=1.0)
        self.assertFalse(flat_ok)

    def test_open_dimension_covers_its_ends_as_densely_as_the_middle(self) -> None:
        encoder = Encoder(
            OpenedDimension(
                "X",
                (0.0, 6.0),
                geometric_detector_layers(layers_count=6, widest_detectors=2, narrowest_detectors=7, overlap=0.4),
                code_length=64,
            ),
            random_seed=0,
        )
        counts = [encoder.encode(float(value))[1].count() for value in range(7)]
        middle = counts[3]
        self.assertGreaterEqual(counts[0], 0.6 * middle)
        self.assertGreaterEqual(counts[6], 0.6 * middle)
        with self.assertRaises(ValueError):
            encoder.encode(6.5)

    def test_pseudo_colour_is_the_sum_of_bit_colours(self) -> None:
        colorizer = EncoderCodeColorizer(min_hue=0.0, max_hue=300.0)
        red_only = BitArray.from_indices(8, [0])
        self.assertEqual(colorizer.color(red_only), (255, 0, 0))
        mixed = BitArray.from_indices(8, [0, 7])
        self.assertEqual(colorizer.color(mixed), colorizer.color(BitArray.from_indices(8, [7, 0])))
        self.assertNotEqual(colorizer.color(mixed), (255, 0, 0))
        self.assertEqual(colorizer.color(BitArray(8)), (0, 0, 0))

    def test_colorizer_can_select_one_dimension_segment(self) -> None:
        encoder = Encoder(
            ClosedDimension("Angle", (0.0, 360.0), [Detectors(4, 0.4)], code_length=16),
            OpenedDimension("X", (0.0, 6.0), [Detectors(3, 0.4)], code_length=16),
            random_seed=0,
        )
        colorizer = EncoderCodeColorizer.for_dimension(encoder, "Angle", min_hue=0.0, max_hue=300.0)
        self.assertEqual((colorizer.start_bit, colorizer.bit_length), (0, 16))
        same_angle_a = encoder.encode(10.0, 0.0)[1]
        same_angle_b = encoder.encode(10.0, 6.0)[1]
        self.assertEqual(colorizer.color(same_angle_a), colorizer.color(same_angle_b))


class LayoutInvariantTests(unittest.TestCase):
    """§5.4, §5.7, §5.14, §7.1.6: порог λ, энергия точки, псевдоцвета и запас ячеек."""

    @staticmethod
    def _tiny_layout(**kwargs) -> Layout:
        codes = {
            0.0: [LayoutEncodedCode(code=BitArray.from_indices(8, [0, 1, 2]), label="a")],
            1.0: [LayoutEncodedCode(code=BitArray.from_indices(8, [0, 1, 3]), label="b")],
            2.0: [LayoutEncodedCode(code=BitArray.from_indices(8, [5, 6, 7]), label="c")],
        }
        params = dict(similarity="cosine", lambda_threshold=0.3, eta=None, use_gpu=False, seed=0)
        params.update(kwargs)
        return Layout(codes, **params)

    def test_eta_zero_is_rejected_because_it_disables_the_threshold(self) -> None:
        with self.assertRaises(ValueError):
            self._tiny_layout(eta=0.0)
        with self.assertRaises(ValueError):
            self._tiny_layout(eta=-1.0)
        layout = self._tiny_layout(eta=None)
        with self.assertRaises(ValueError):
            layout.set_similarity_params(eta=0.0)
        layout.set_similarity_params(eta=10.0)

    def test_sigmoid_threshold_matches_article_formula(self) -> None:
        layout = self._tiny_layout(eta=10.0, lambda_threshold=0.5)
        sim = layout._similarity_value(0, 1)
        expected = sim / (1.0 + math.exp(-10.0 * (sim - 0.5)))
        self.assertAlmostEqual(layout._sim_lambda_idx(0, 1), expected)
        hard = self._tiny_layout(eta=None, lambda_threshold=0.9)
        self.assertEqual(hard._sim_lambda_idx(0, 1), 0.0)
        self.assertEqual(hard._sim_lambda_idx(0, 2), 0.0)

    def test_point_energy_uses_euclidean_distance_and_is_normalised(self) -> None:
        layout = self._tiny_layout(grid_size=3)
        layout.grid = [[None] * 3 for _ in range(3)]
        layout._row_occupied = [set() for _ in range(3)]
        positions = [(0, 0), (0, 2), (2, 2)]
        for index, (y, x) in enumerate(positions):
            layout.grid[y][x] = index
            layout._row_occupied[y].add(x)
        layout._positions = positions
        layout._pos_y = [p[0] for p in positions]
        layout._pos_x = [p[1] for p in positions]
        energies = layout.point_energies(3)
        sim_ab = layout._sim_lambda_idx(0, 1)
        sim_bc = layout._sim_lambda_idx(1, 2)
        sim_ac = layout._sim_lambda_idx(0, 2)
        eps = layout._distance_eps
        raw = [
            sim_ab / (2.0 + eps) + sim_ac / (math.sqrt(8.0) + eps),
            sim_ab / (2.0 + eps) + sim_bc / (2.0 + eps),
            sim_bc / (2.0 + eps) + sim_ac / (math.sqrt(8.0) + eps),
        ]
        emax = max(raw)
        for value, expected in zip(energies, raw):
            self.assertAlmostEqual(value, expected / emax, places=9)
        self.assertEqual(max(energies), 1.0)

    def test_layout_colours_come_from_codes_not_from_stimulus_labels(self) -> None:
        codes = {
            0.0: [
                LayoutEncodedCode(code=BitArray.from_indices(8, [0]), label="same-hue-a"),
                LayoutEncodedCode(code=BitArray.from_indices(8, [7]), label="same-hue-b"),
            ]
        }
        layout = Layout(codes, use_gpu=False, lambda_threshold=0.0, eta=None)
        colours = layout.colors_rgb()
        self.assertNotEqual(colours[0], colours[1])
        self.assertEqual(colours[0], (255, 0, 0))
        component = layout.component_colors_rgb(period=360.0)
        self.assertEqual(component[0], component[1])

    def test_default_empty_ratio_matches_space_filling_margin(self) -> None:
        self.assertEqual(main.LAYOUT_EMPTY_RATIO, 0.15)
        layout = Layout(
            {float(i): [BitArray.from_indices(8, [i % 8])] for i in range(20)},
            empty_ratio=0.15,
            use_gpu=False,
            lambda_threshold=0.0,
            eta=None,
        )
        self.assertEqual(layout.width, math.ceil(math.sqrt(20 * 1.15)))

    def test_short_pair_sampling_escapes_single_cell_partition(self) -> None:
        layout = self._tiny_layout(grid_size=3)
        y1, x1 = layout._positions[0]
        y2, x2 = layout._sample_nearby_position(y1, x1, 1, bounds=(y1, y1, x1, x1))
        self.assertNotEqual((y2, x2), (y1, x1))
        self.assertTrue(0 <= y2 < layout.height and 0 <= x2 < layout.width)

    def test_gaussian_long_kernel_weights_neighbours_by_distance(self) -> None:
        """Расширение: ядро gauss даёт φ = Σ s·exp(−2d²/r²), ядро inverse — формулу §5.5.2."""
        layout = self._tiny_layout(grid_size=5, long_kernel="gauss", long_kernel_radius_min=2)
        layout.grid = [[None] * 5 for _ in range(5)]
        layout._row_occupied = [set() for _ in range(5)]
        positions = [(0, 0), (1, 1), (0, 4)]  # a, b (похож на a), c (не похож ни на кого)
        for index, (y, x) in enumerate(positions):
            layout.grid[y][x] = index
            layout._positions[index] = (y, x)
            layout._pos_y[index] = y
            layout._pos_x[index] = x
            layout._row_occupied[y].add(x)
        radius = 4
        s_ba = layout._sim_lambda_idx(1, 0)
        self.assertGreater(s_ba, 0.0)
        self.assertEqual(layout._sim_lambda_idx(1, 2), 0.0)
        energy_c, energy_s = layout._pair_energy_short(0, (0, 0), 2, (0, 4), radius=radius, kernel="gauss")
        d_ba, d_bc = 2.0, 10.0  # квадраты расстояний b→a и b→c
        self.assertAlmostEqual(energy_c, s_ba * math.exp(-2.0 * d_ba / radius ** 2))
        self.assertAlmostEqual(energy_s, s_ba * math.exp(-2.0 * d_bc / radius ** 2))
        inv_c, inv_s = layout._pair_energy_short(0, (0, 0), 2, (0, 4), radius=radius, kernel="inverse")
        self.assertAlmostEqual(inv_c, s_ba / (d_ba + layout._distance_eps))
        self.assertAlmostEqual(inv_s, s_ba / (d_bc + layout._distance_eps))
        with self.assertRaises(ValueError):
            self._tiny_layout(long_kernel="flat")
        # дальняя фаза с гауссовым ядром выполняется на CPU без ошибок
        layout.run(steps=2, pairs_per_step=4, pair_radius=2, mode="long", min_swap_ratio=0.0,
                   min_swap_window=None, log_every=None, energy_stability_window=None, log_visuals=False)

    def test_adaptive_lambda_never_exceeds_lambda_max(self) -> None:
        from damp.layout.damp_layout import AdaptiveLayoutConfig, _AdaptiveLayoutState

        state = _AdaptiveLayoutState(
            AdaptiveLayoutConfig(start_radius=4, end_radius=1, swap_ratio_trigger=1.0,
                                 lambda_step=0.3, lambda_max=0.7),
            total_steps=10,
        )
        current = 0.5
        for step in range(5):
            new_lambda, _ = state.update_after_step(step=step, swaps=0, current_lambda=current)
            if new_lambda is not None:
                current = new_lambda
        self.assertAlmostEqual(current, 0.7)

    def test_main_layout_threshold_is_enabled(self) -> None:
        self.assertIsNotNone(main.LAYOUT_ETA)
        self.assertGreater(main.LAYOUT_ETA, 0.0)
        self.assertGreater(main.LAYOUT_LAMBDA_THRESHOLD, 0.0)

    def test_smoothing_flag_switches_long_phase_between_extension_and_article(self) -> None:
        """LAYOUT_SMOOTH_PINWHEELS = False возвращает дальнюю фазу §5.5.1, §5.6, §7.1.7."""
        import re
        import sys
        import types

        source = Path(main.__file__).read_text(encoding="utf-8")
        pattern = re.compile(r"^LAYOUT_SMOOTH_PINWHEELS = (True|False)$", re.MULTILINE)
        self.assertEqual(len(pattern.findall(source)), 1)

        def load(flag: bool) -> types.ModuleType:
            name = f"main_smooth_{flag}"
            module = types.ModuleType(name)
            module.__file__ = main.__file__
            sys.modules[name] = module  # dataclass ищет модуль класса в sys.modules
            try:
                exec(compile(pattern.sub(f"LAYOUT_SMOOTH_PINWHEELS = {flag}", source), main.__file__, "exec"),
                     module.__dict__)
            finally:
                del sys.modules[name]
            return module

        article = load(False)
        self.assertEqual(article.LAYOUT_LONG_KERNEL, "distance")
        self.assertEqual(article.LAYOUT_ADAPTIVE_RADIUS_START_FACTOR, 0.5)
        self.assertFalse(article.LAYOUT_ADAPTIVE_RADIUS_GEOMETRIC)
        self.assertEqual(article.LAYOUT_ANNEAL_TEMPERATURE, 0.0)
        self.assertEqual(article.LAYOUT_ADAPTIVE_SWAP_TRIGGER, 0.01)
        self.assertIsNotNone(article.LAYOUT_ADAPTIVE_LAMBDA_STEP)
        self.assertGreater(article.LAYOUT_ADAPTIVE_LAMBDA_MAX, article.LAYOUT_LAMBDA_THRESHOLD)
        smooth = load(True)
        self.assertEqual(smooth.LAYOUT_LONG_KERNEL, "gauss")
        self.assertGreater(smooth.LAYOUT_ANNEAL_TEMPERATURE, 0.0)
        self.assertTrue(smooth.LAYOUT_ADAPTIVE_RADIUS_GEOMETRIC)
        self.assertNotEqual(article.LAYOUT_CACHE_TAG, smooth.LAYOUT_CACHE_TAG)


def _clustered_layout(seed: int = 0) -> Layout:
    """Две компактные группы похожих кодов на сетке 12×12 для тестов детекторов."""
    import random

    rng = random.Random(seed)
    codes: dict[float, list[LayoutEncodedCode]] = {}
    index = 0
    base_a = [0, 1, 2, 3, 4, 5, 6, 7]
    base_b = [40, 41, 42, 43, 44, 45, 46, 47]
    for group, base in ((0.0, base_a), (1.0, base_b)):
        bucket: list[LayoutEncodedCode] = []
        for _ in range(16):
            bits = list(base)
            bits[rng.randrange(8)] = 20 + rng.randrange(8)
            bucket.append(LayoutEncodedCode(code=BitArray.from_indices(64, bits), label=f"g{group}-{index}"))
            index += 1
        codes[group] = bucket
    layout = Layout(codes, grid_size=12, similarity="cosine", lambda_threshold=0.5, eta=None, use_gpu=False, seed=seed)
    layout.grid = [[None] * 12 for _ in range(12)]
    layout._row_occupied = [set() for _ in range(12)]
    positions: list[tuple[int, int]] = []
    for point_index in range(32):
        if point_index < 16:
            y, x = 1 + point_index // 4, 1 + point_index % 4
        else:
            y, x = 7 + (point_index - 16) // 4, 7 + (point_index - 16) % 4
        positions.append((y, x))
        layout.grid[y][x] = point_index
        layout._row_occupied[y].add(x)
    layout._positions = positions
    layout._pos_y = [p[0] for p in positions]
    layout._pos_x = [p[1] for p in positions]
    return layout


class DetectorInvariantTests(unittest.TestCase):
    """§6.3–§6.5: случайные центры и биты, DBSCAN, вставка, уровень активации."""

    def test_dbscan_returns_the_cluster_of_the_seed_and_rejects_noise(self) -> None:
        clusterer = DbscanClusterer(eps=1.5, min_samples=3)
        block = [(y, x) for y in range(3) for x in range(3)]
        far_noise = [(10, 10)]
        cluster = clusterer.cluster_of(block + far_noise, (1, 1))
        self.assertEqual(sorted(cluster), block)
        self.assertEqual(clusterer.cluster_of(block + far_noise, (10, 10)), [])
        self.assertEqual(clusterer.cluster_of(block, (5, 5)), [])

    def _build(self, seed: int, *, layout: Layout | None = None) -> list[Detector]:
        layout = layout or _clustered_layout()
        energies = layout.point_energies(3)
        builder = DetectorHierarchyBuilder(
            layout=layout,
            energies=energies,
            layer_configs=[DetectorLayerConfig(lambda_threshold=0.5, activation_radius=4, min_points=3)],
            radius_min=1,
            output_bits=256,
            energy_threshold=0.05,
            eta=None,
            attempts_without_change=60,
            seed=seed,
            clusterer=DbscanClusterer(eps=1.5, min_samples=3),
        )
        return builder.build()

    def test_detectors_cover_both_clusters_with_random_single_output_bits(self) -> None:
        detectors = self._build(seed=0)
        self.assertGreaterEqual(len(detectors), 2)
        for detector in detectors:
            self.assertEqual(detector.output_code.count(), 1, "§6.3: один случайный бит")
            self.assertEqual(len(detector.output_code), 256)
            self.assertGreater(detector.energy, 0.0)
            self.assertGreaterEqual(detector.points_count, 3)
        bits = sorted(detector.output_code.active_indices()[0] for detector in detectors)
        self.assertNotEqual(bits, list(range(len(bits))), "биты не выдаются по порядку")
        centres_y = sorted(detector.center[0] for detector in detectors)
        self.assertLess(centres_y[0], 6.0)
        self.assertGreater(centres_y[-1], 6.0)

    def test_detector_centres_do_not_fall_inside_each_other(self) -> None:
        detectors = self._build(seed=1)
        for first in detectors:
            for second in detectors:
                if first is second:
                    continue
                distance = math.hypot(first.center[0] - second.center[0], first.center[1] - second.center[1])
                self.assertGreater(distance, first.radius)
                self.assertGreater(distance, second.radius)

    def test_detector_construction_is_stochastic_and_seeded(self) -> None:
        first = [(d.center, d.output_code.active_indices()) for d in self._build(seed=0)]
        repeat = [(d.center, d.output_code.active_indices()) for d in self._build(seed=0)]
        other = [(d.output_code.active_indices()) for d in self._build(seed=5)]
        self.assertEqual(first, repeat)
        self.assertNotEqual([bit for _, bit in first], other)

    def test_activation_level_is_a_ratio_without_clipping(self) -> None:
        layout = _clustered_layout()
        energies = layout.point_energies(3)
        detectors = self._build(seed=0, layout=layout)
        engine = ActivationEngine(
            layout=layout,
            detectors=detectors,
            lambda_threshold=0.5,
            eta=None,
            energy_threshold=0.05,
            detector_threshold=0.5,
            saturation_limit=50,
            energy_map=energies,
            use_gpu=False,
        )
        weak_detector = Detector(
            center=detectors[0].center,
            radius=detectors[0].radius,
            lambda_threshold=0.5,
            energy=detectors[0].energy / 4.0,
            points_count=detectors[0].points_count,
            output_code=BitArray.from_indices(256, [3]),
            label="weak",
        )
        engine_weak = ActivationEngine(
            layout=layout,
            detectors=[weak_detector],
            lambda_threshold=0.5,
            eta=None,
            energy_threshold=0.05,
            detector_threshold=0.5,
            saturation_limit=50,
            energy_map=energies,
            use_gpu=False,
        )
        covered = range(16, 32) if detectors[0].center[0] > 6.0 else range(16)
        stimuli = [layout._points[index].code for index in covered]
        levels = [item.activation_level for item in engine_weak.activate(stimuli).active_detectors]
        self.assertTrue(levels and levels[0] > 1.0, "уровень активации — отношение энергий без ограничения (§6.5)")
        result = engine.activate(stimuli)
        self.assertGreater(result.embedding.count(), 0)
        self.assertLessEqual(result.embedding.count(), 50)

    def test_colour_merge_prefers_higher_lambda_detectors_when_budget_is_short(self) -> None:
        layout = _clustered_layout()
        energies = layout.point_energies(3)
        base = self._build(seed=0, layout=layout)[0]
        # Оба детектора попадают в D_mu (sigma = 2 детектора), но бюджет битов
        # вмещает код только одного: по §3.3.2 сохраняется коротковолновый (λ=0.85).
        red = Detector(center=base.center, radius=base.radius, lambda_threshold=0.5, energy=base.energy,
                       points_count=base.points_count, output_code=BitArray.from_indices(256, [10, 11]), label="red")
        violet = Detector(center=base.center, radius=base.radius, lambda_threshold=0.85, energy=base.energy,
                          points_count=base.points_count, output_code=BitArray.from_indices(256, [20, 21]), label="violet")
        engine = ActivationEngine(
            layout=layout,
            detectors=[red, violet],
            lambda_threshold=0.5,
            eta=None,
            energy_threshold=0.05,
            detector_threshold=0.1,
            saturation_limit=2,
            energy_map=energies,
            use_gpu=False,
        )
        stimulus_index = 16 if base.center[0] > 6.0 else 0
        result = engine.activate([layout._points[stimulus_index].code])
        self.assertEqual(result.embedding.active_indices(), (20, 21))


class PinwheelConfigurationTests(unittest.TestCase):
    """§2.2.3, §4.4: по 360 углов в каждой позиции кодируются как стимулы MNIST."""

    def test_pinwheel_encoder_matches_mnist_segments_and_detector_assignment(self) -> None:
        from pinwheel_test import PinwheelEncoderFactory

        self.assertEqual(main.PINWHEEL_ANGLE_COUNT, 360)
        encoder = PinwheelEncoderFactory.create()
        reference = main._build_encoder()
        self.assertEqual(encoder.signature, reference.signature)
        self.assertEqual([dim.title for dim in encoder.dimensions], ["Angle", "X", "Y"])
        self.assertEqual(encoder.code_length, main.MNIST_ANGLE_CODE_LENGTH + 2 * main.MNIST_POSITION_CODE_LENGTH)
        for dimension in ("Angle", "X", "Y"):
            self.assertEqual(encoder.segment(dimension), reference.segment(dimension))

    def test_all_angles_use_every_position_and_exact_mnist_codes(self) -> None:
        from pinwheel_test import PinwheelExperiment

        experiment = PinwheelExperiment()
        reference = main._build_encoder()
        codes = experiment.encode_codes()
        self.assertEqual(sorted(codes), [float(angle) for angle in range(360)])
        positions = len(main.PINWHEEL_POSITIONS)
        self.assertGreaterEqual(positions, 2)
        self.assertEqual(len(set(main.PINWHEEL_POSITIONS)), positions)
        self.assertEqual(sum(len(bucket) for bucket in codes.values()), 360 * positions)
        position_segments = {
            position: {dimension: set() for dimension in ("X", "Y")}
            for position in main.PINWHEEL_POSITIONS
        }
        for angle, bucket in codes.items():
            self.assertEqual(len(bucket), len(main.PINWHEEL_POSITIONS))
            for (x, y), entry in zip(main.PINWHEEL_POSITIONS, bucket):
                _, expected = reference.encode(angle, x, y)
                self.assertEqual(entry.code, expected, f"код ({angle}, {x}, {y}) отличается от MNIST")
                bits = entry.code.to01()
                for dimension, segments in position_segments[(x, y)].items():
                    offset, length = reference.segment(dimension)
                    segments.add(bits[offset:offset + length])
        for position, dimensions in position_segments.items():
            for dimension, segments in dimensions.items():
                self.assertEqual(len(segments), 1, f"сегмент {dimension} позиции {position} должен быть постоянным")
                self.assertIn("1", next(iter(segments)))
        self.assertNotEqual(codes[0.0][0].code, codes[0.0][1].code)

    def test_export_replaces_codes_and_labels_from_previous_experiment(self) -> None:
        from pinwheel_test import PinwheelExperiment

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layout.json"
            experiment = PinwheelExperiment(
                output_path=str(Path(directory) / "codes.png"),
                component_output_path=str(Path(directory) / "angle.png"),
                energy_output_path=str(Path(directory) / "energy.png"),
                layout_path=str(path),
            )
            codes = experiment.encode_codes()
            layout = Layout(codes, use_gpu=False, precompute_similarity=False, parallel_workers=1)
            path.write_text(json.dumps({
                "layout": [{"index": 0, "label": "obsolete", "hue": -1.0,
                            "value": "1" * 4096, "ones": 4096}],
                "encoder_signature": "obsolete",
            }), encoding="utf-8")
            experiment._save(layout)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["stimulus_positions"],
                [{"x": x, "y": y} for x, y in main.PINWHEEL_POSITIONS],
            )
            expected = [(angle, entry) for angle, bucket in codes.items() for entry in bucket]
            self.assertEqual(payload["points"], len(expected))
            self.assertEqual(len(payload["layout"]), len(expected))
            for point, (angle, entry) in zip(payload["layout"], expected):
                self.assertEqual(point["value"], entry.code.to01())
                self.assertEqual(point["label"], entry.label)
                self.assertEqual(point["hue"], angle)
                self.assertEqual(point["ones"], entry.code.count())
                self.assertIsNotNone(point["x"])
                self.assertIsNotNone(point["y"])


if __name__ == "__main__":
    unittest.main()
