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

    def test_shared_code_merges_dimension_codes_without_concatenation(self) -> None:
        """§2.2.3, §2.2.1: общий код той же длины — OR кодов координат; коллизии между координатами учитываются."""
        layers = geometric_detector_layers(layers_count=4, widest_detectors=2, narrowest_detectors=20, overlap=0.5)
        encoder = Encoder(
            ClosedDimension("Angle", (0.0, 360.0), layers, code_length=64),
            OpenedDimension("X", (0.0, 6.0), layers, code_length=64),
            random_seed=0,
            shared_code=True,
        )
        self.assertEqual(encoder.code_length, 64)
        self.assertEqual(encoder.segment("Angle"), (0, 64))
        self.assertEqual(encoder.segment("X"), (0, 64))
        self.assertTrue(all(0 <= window.bit < 64 for window in encoder.detectors))
        measured = []
        for angle in range(0, 360, 15):
            for x in (0.0, 2.5, 6.0):
                _, indices, parasitic = encoder.encode_indices(float(angle), x)
                union = set(encoder.dimension_bits("Angle", angle)) | set(encoder.dimension_bits("X", x))
                self.assertEqual(set(indices), union, "код — OR кодов координат (§2.2.1)")
                self.assertEqual(len(indices), len(set(indices)))
                measured.append(len(indices) / 64)
        self.assertAlmostEqual(sum(measured) / len(measured), encoder.expected_density(), delta=0.06)
        with self.assertRaises(ValueError):
            Encoder(ClosedDimension("Angle", (0.0, 360.0), layers, code_length=64),
                    OpenedDimension("X", (0.0, 6.0), layers, code_length=32), shared_code=True)

    def test_detector_view_shows_layers_overlap_and_wraps_closed_fields(self) -> None:
        """§4.1, §4.2.1: картинка детекторов — слои, дорожки перекрытия, поле через границу периода (§4.4)."""
        from damp.encoding.detector_view import DetectorLayerView

        angle_layers = geometric_detector_layers(layers_count=3, widest_detectors=2, narrowest_detectors=12, overlap=0.4)
        x_layers = geometric_detector_layers(layers_count=3, widest_detectors=2, narrowest_detectors=8, overlap=0.8)
        angle = ClosedDimension("Angle", (0.0, 360.0), angle_layers, code_length=48)
        x = OpenedDimension("X", (0.0, 6.0), x_layers, code_length=48)
        encoder = Encoder(angle, x, random_seed=0, shared_code=True)
        view = DetectorLayerView(encoder, min_hue=0.0, max_hue=300.0, inactive_alpha=0.3, dpi=40, width_inches=8.0)
        image = view.render((10.0, 3.0))
        self.assertEqual(image.ndim, 3)
        self.assertEqual(image.shape[2], 3)
        self.assertGreater(int(image.std()), 0)
        self.assertEqual(view._layer_lanes(angle, 0), 2, "перекрытие 0.4: точку накрывают два соседних детектора")
        self.assertEqual(view._layer_lanes(x, 2), 5, "перекрытие 0.8: точку накрывают пять соседних детекторов")
        wrapped = next(w for w in encoder.detectors if w.dimension_index == 0 and w.center - w.half_width < 0.0)
        segments = view._field_segments(wrapped, angle)
        self.assertEqual(len(segments), 2, "поле у границы периода рисуется двумя кусками")
        self.assertAlmostEqual(sum(end - start for start, end in segments), 2.0 * wrapped.half_width)

    def test_bit_colour_follows_receptive_field_width(self) -> None:
        """§3.2, §4.4.2, §5.14: широкие детекторы — красные младшие биты, узкие — фиолетовые старшие."""
        layers = geometric_detector_layers(layers_count=5, widest_detectors=2, narrowest_detectors=40, overlap=0.4)
        dim = ClosedDimension("Angle", (0.0, 360.0), layers, code_length=53)
        encoder = Encoder(dim, random_seed=0)
        bands = [dim.layer_band(index) for index in range(len(layers))]
        self.assertEqual(bands[0][0], 0)
        self.assertEqual(sum(length for _, length in bands), 53)
        for (start, length), (next_start, _) in zip(bands, bands[1:]):
            self.assertEqual(start + length, next_start, "полосы идут подряд от широкого слоя к узкому")
        for window in encoder.detectors:
            start, length = bands[window.layer_index]
            self.assertTrue(start <= window.local_bit < start + length, window)
        widest = [w.local_bit for w in encoder.detectors if w.layer_index == 0]
        self.assertEqual(len(widest), len(set(widest)), "в полосе хватает бит: детекторы слоя не совпадают")
        with self.assertRaises(ValueError):
            ClosedDimension("Angle", (0.0, 360.0), layers, code_length=4)

    def test_random_bits_are_reproducible_by_seed_and_differ_between_seeds(self) -> None:
        first = _angle_encoder(seed=7)
        second = _angle_encoder(seed=7)
        third = _angle_encoder(seed=8)
        self.assertEqual([w.bit for w in first.detectors], [w.bit for w in second.detectors])
        self.assertNotEqual([w.bit for w in first.detectors], [w.bit for w in third.detectors])
        self.assertEqual(first.encode(42.0)[1].to01(), second.encode(42.0)[1].to01())

    def test_bit_collisions_are_allowed_and_measured_density_matches_estimate(self) -> None:
        encoder = _angle_encoder(seed=0, layers=22, narrowest=180, length=256)  # угол MNIST из main.py
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

    def test_open_dimension_without_edge_detectors_keeps_exact_counts_and_covers_segment(self) -> None:
        """§4.2.1: в слое ровно count детекторов внутри отрезка, соседние перекрываются, концы покрыты."""
        layers = geometric_detector_layers(layers_count=5, widest_detectors=2, narrowest_detectors=14, overlap=0.4)
        encoder = Encoder(
            OpenedDimension("X", (0.0, 28.0), layers, code_length=128, edge_detectors=False),
            random_seed=0,
        )
        for layer_index, layer in enumerate(layers):
            windows = [w for w in encoder.detectors if w.layer_index == layer_index]
            self.assertEqual(len(windows), layer.count)
            self.assertTrue(all(0.0 < w.center < 28.0 for w in windows))
            for value in [k * 0.25 for k in range(0, 113)]:
                self.assertGreaterEqual(sum(1 for w in windows if w.is_active(value)), 1, (layer_index, value))
            centers = sorted(w.center for w in windows)
            if len(centers) > 1:
                self.assertLess(centers[1] - centers[0], 2.0 * windows[0].half_width, "соседние поля перекрываются")

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
                   min_swap_window=None, log_every=None, log_visuals=False)

    def test_adaptive_lambda_never_exceeds_lambda_max(self) -> None:
        from damp.layout.damp_layout import AdaptiveLayoutConfig, _AdaptiveLayoutState

        state = _AdaptiveLayoutState(
            AdaptiveLayoutConfig(start_radius=4, end_radius=1, swap_ratio_trigger=1.0,
                                 lambda_step=0.3, lambda_max=0.7),
            total_steps=10,
        )
        current = 0.5
        for step in range(5):
            new_lambda = state.update_after_step(step=step, swaps=0, current_lambda=current).new_lambda
            if new_lambda is not None:
                current = new_lambda
        self.assertAlmostEqual(current, 0.7)

    def test_swap_schedule_changes_radius_and_lambda_only_on_one_percent_trigger(self) -> None:
        """§7.1.7: r и λ меняются, когда обмены за единицу времени падают до 1 % от начального значения этапа."""
        from damp.layout.damp_layout import AdaptiveLayoutConfig, _AdaptiveLayoutState

        config = AdaptiveLayoutConfig(start_radius=40, end_radius=1, swap_ratio_trigger=0.01,
                                      lambda_step=0.05, lambda_max=0.8, radius_schedule="swaps",
                                      radius_factor=0.5, swap_window=2)
        state = _AdaptiveLayoutState(config, total_steps=1000)
        current = 0.65
        # Первое окно задаёт начальное значение этапа (200 обменов), r постоянен, пока обмены не упали до 1 %.
        for step, swaps in enumerate([200, 200, 150, 50, 10, 4]):
            update = state.update_after_step(step=step, swaps=swaps, current_lambda=current)
            self.assertFalse(update.triggered)
            self.assertEqual(state.radius_for_step(step + 1), 40)
        update = state.update_after_step(step=6, swaps=0, current_lambda=current)  # среднее окна 2 = 1 % от 200
        self.assertTrue(update.triggered)
        self.assertAlmostEqual(update.new_lambda, 0.70)
        self.assertEqual(state.radius_for_step(500), 20)
        current = update.new_lambda
        # Новый этап: начальное значение берётся заново, старые 200 обменов не учитываются.
        update = state.update_after_step(step=7, swaps=6, current_lambda=current)
        self.assertIsNone(update.swap_ratio)
        update = state.update_after_step(step=8, swaps=6, current_lambda=current)
        self.assertAlmostEqual(update.stage_initial, 6.0)
        self.assertFalse(update.triggered)
        self.assertFalse(state.schedule_complete(current))
        # Обмены упали ниже порога остановки §5.6 до конца расписания: следующий этап вместо остановки.
        for step in range(9, 40):
            current = state.force_next_stage(step=step, current_lambda=current).new_lambda or current
        self.assertEqual(state.radius_for_step(1000), 1)
        self.assertAlmostEqual(current, 0.8)
        self.assertTrue(state.schedule_complete(current))

    def test_step_schedules_are_marked_as_extension_and_validated(self) -> None:
        from damp.layout.damp_layout import AdaptiveLayoutConfig, _AdaptiveLayoutState

        with self.assertRaises(ValueError):
            AdaptiveLayoutConfig(start_radius=4, radius_schedule="cosine")
        geometric = _AdaptiveLayoutState(
            AdaptiveLayoutConfig(start_radius=16, end_radius=1, swap_ratio_trigger=0.0,
                                 radius_schedule="geometric"),
            total_steps=5,
        )
        self.assertEqual([geometric.radius_for_step(s) for s in range(5)], [16, 8, 4, 2, 1])
        self.assertTrue(geometric.schedule_complete(0.65))

    def test_far_energies_follow_lambda_after_adaptive_raise(self) -> None:
        """§5.4, §7.1.7: после подъёма λ дальние энергии считаются с новым порогом, а не из кеша строк sim_λ."""
        from damp.layout.damp_layout import _AdaptiveUpdate

        layout = self._tiny_layout(grid_size=3)  # sim(a, b) = 2/3 > λ = 0.3, c не похож ни на кого
        (ya, xa), (yc, xc) = layout._positions[0], layout._positions[2]
        pair = [(ya, xa, yc, xc, 0, 2)]  # пара (a, c): в φ входит слагаемое s(a, b)·d
        before = layout._calculate_far_energies(pair, advance_step=False)
        self.assertGreater(max(abs(v) for v in before[0]), 0.0)
        layout._apply_adaptive_lambda(
            _AdaptiveUpdate(new_lambda=0.8, swap_ratio=0.0, swap_rate=0.0, stage_initial=1.0,
                            triggered=True, radius=1),
            step=1,
        )
        after = layout._calculate_far_energies(pair, advance_step=False)
        # При λ = 0.8 > 2/3 и жёстком пороге все sim_λ нулевые: энергии пары обнуляются.
        self.assertEqual(after[0], (0.0, 0.0))

    def test_layout_stops_when_quality_reaches_target(self) -> None:
        """§5.6: шаги повторяются, пока качество раскладки Ē (§5.8) не достигнет заданного значения."""
        layout = self._tiny_layout(grid_size=5)
        layout.run(steps=50, pairs_per_step=4, pair_radius=2, mode="long", min_swap_ratio=0.0,
                   min_swap_window=None, log_every=None, log_visuals=False,
                   energy_radius=3, energy_every=1, energy_target=1e-9)
        self.assertLess(layout.last_steps, 50)
        self.assertGreaterEqual(layout._last_energy_value, 1e-9)
        with self.assertRaises(ValueError):
            layout.run(steps=1, pairs_per_step=1, pair_radius=1, mode="long", energy_target=0.5)
        with self.assertRaises(ValueError):  # r_e для Ē задаётся явно (§5.7)
            layout.run(steps=1, pairs_per_step=1, pair_radius=1, mode="long", energy_every=1)

    def test_saved_json_keeps_code_and_position_of_the_same_point(self) -> None:
        """§5.14: в JSON раскладки код и координаты одной записи принадлежат одной точке."""
        layout = _clustered_layout()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layout.json"
            # Базовый файл с другим порядком записей (как payload MNIST в порядке стимулов).
            shuffled = list(reversed(range(layout._point_count)))
            path.write_text(json.dumps({
                "encoder_signature": "kept",
                "layout": [{"index": k, "value": layout._values[i], "label": layout._labels[i], "hue": -1.0}
                           for k, i in enumerate(shuffled)],
            }), encoding="utf-8")
            layout.save_json(str(path))
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["encoder_signature"], "kept")
        for item in payload["layout"]:
            index = int(item["index"])
            self.assertEqual(item["value"], layout._values[index])
            self.assertEqual((item["y"], item["x"]), layout._positions[index])
            self.assertEqual(item["hue"], layout._points[index].hue)

    def test_average_energy_is_normalised_by_maximum_over_whole_space(self) -> None:
        """§5.7, §5.8: Ē = (1/|V+|) Σ E/E_max, E_max = max по всему V, без выборки."""
        layout = _clustered_layout()
        raw = layout._compute_point_energies(3)
        self.assertEqual(len(raw), layout._point_count)
        expected = sum(value / max(raw) for value in raw) / len(raw)
        self.assertAlmostEqual(layout.average_local_energy(3), expected, places=6)
        normalised = layout.point_energies(3)
        self.assertAlmostEqual(max(normalised), 1.0)

    def test_first_point_weights_use_energies_of_all_points_and_are_periodic(self) -> None:
        """§5.9.3: P(p) ∝ −ln Ê(p, r) по энергиям всех точек, веса пересчитываются периодически."""
        from damp.layout.damp_layout import ShortLayoutOptimConfig

        layout = _clustered_layout()
        config = ShortLayoutOptimConfig(energy_radius=3, energy_recalc_every=10, energy_eps=1e-6)
        layout._ensure_short_energy_weights(step_index=0, config=config)
        energies = layout.point_energies(3)
        expected = [max(0.0, -math.log(max(1e-6, value))) for value in energies]
        for got, want in zip(layout._short_energy_weights, expected):
            self.assertAlmostEqual(got, want, places=6)
        cap = -math.log(1e-6)
        self.assertLess(sum(1 for w in layout._short_energy_weights if w >= cap - 1e-9), layout._point_count)
        weights_before = layout._short_energy_weights
        (y1, x1), (y2, x2) = layout._positions[0], layout._positions[20]
        layout._apply_swaps([(y1, x1, y2, x2)], mode="short")
        self.assertIs(layout._short_energy_weights, weights_before, "обмен не сбрасывает веса")
        layout._ensure_short_energy_weights(step_index=5, config=config)
        self.assertIs(layout._short_energy_weights, weights_before, "пересчёт только раз в energy_recalc_every")
        layout._ensure_short_energy_weights(step_index=10, config=config)
        self.assertIsNot(layout._short_energy_weights, weights_before)

    def test_main_layout_threshold_is_enabled(self) -> None:
        # §5.4: η = 0 выключает порог; η = None — жёсткий порог η = ∞ (§7.1.7).
        self.assertTrue(main.LAYOUT_ETA is None or main.LAYOUT_ETA > 0.0)
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
        self.assertEqual(article.LAYOUT_ADAPTIVE_RADIUS_SCHEDULE, "swaps")
        self.assertEqual(article.LAYOUT_SHORT_ADAPTIVE_RADIUS_SCHEDULE, "swaps")
        self.assertEqual(article.LAYOUT_ANNEAL_TEMPERATURE, 0.0)
        self.assertEqual(article.LAYOUT_ADAPTIVE_SWAP_TRIGGER, 0.01)
        self.assertIsNone(article.LAYOUT_ETA)
        self.assertEqual(article.LAYOUT_LAMBDA_THRESHOLD, 0.65)
        self.assertAlmostEqual(article.LAYOUT_ADAPTIVE_LAMBDA_STEP, 0.05)
        self.assertAlmostEqual(article.LAYOUT_ADAPTIVE_LAMBDA_MAX, 0.8)
        smooth = load(True)
        self.assertEqual(smooth.LAYOUT_LONG_KERNEL, "gauss")
        self.assertGreater(smooth.LAYOUT_ANNEAL_TEMPERATURE, 0.0)
        self.assertEqual(smooth.LAYOUT_ADAPTIVE_RADIUS_SCHEDULE, "geometric")
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


class RerunViewerTests(unittest.TestCase):
    """Окно Rerun всегда показывает ход раскладки: логгер делает свою раскладку панелей активной."""

    def test_logger_activates_damp_blueprint_right_after_spawning_viewer(self) -> None:
        import inspect
        from unittest import mock

        import rerun.blueprint as rrb

        from damp import logging as damp_logging

        blueprint = damp_logging.RerunViewerBlueprint.build()
        self.assertIsInstance(blueprint, rrb.Blueprint)
        default_path = inspect.signature(Layout.run).parameters["log_path"].default
        self.assertEqual(damp_logging.RerunViewerBlueprint.LAYOUT_MAP, f"/{default_path}/angle",
                         "главная панель — ход раскладки по углу, туда его пишет Layout")
        self.assertEqual(damp_logging.RerunViewerBlueprint.LAYOUT_CODE_MAP, f"/{default_path}/image")
        logger = damp_logging.DampLogger(spawn=True)
        with mock.patch.object(damp_logging.rr, "init"), \
                mock.patch.object(damp_logging.rr, "is_enabled", return_value=True), \
                mock.patch.object(damp_logging.rr, "spawn") as spawn, \
                mock.patch.object(damp_logging.rr, "send_blueprint") as send:
            logger._ensure_rerun()
            logger._ensure_rerun()
        spawn.assert_called_once()
        send.assert_called_once()
        self.assertTrue(send.call_args.kwargs["make_active"])


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

    def test_detector_energy_counts_whole_receptive_field_activated_by_centre_code(self) -> None:
        """§6.3, §6.4.5: n_d = |A_μ|, e_d = Σ a·e по A_λd(c_d, r_d) — всё поле r_d вокруг c_d, код из центра."""
        layout = _clustered_layout()
        energies = layout.point_energies(3)
        positions = layout.positions()
        for detector in self._build(seed=0, layout=layout):
            cy, cx = detector.center
            field = [i for i, (y, x) in enumerate(positions) if math.hypot(y - cy, x - cx) <= detector.radius]
            centre = min(field, key=lambda i: (math.hypot(positions[i][0] - cy, positions[i][1] - cx), i))
            a_mu = [i for i in field if energies[i] >= 0.05]
            e_d = 0.0
            for i in a_mu:
                similarity = layout._similarity_value(centre, i)
                e_d += (similarity if similarity >= detector.lambda_threshold else 0.0) * energies[i]
            self.assertEqual(detector.points_count, len(a_mu))
            self.assertAlmostEqual(detector.energy, e_d, places=9)
            engine = ActivationEngine(layout=layout, detectors=[detector], lambda_threshold=detector.lambda_threshold,
                                      eta=None, energy_threshold=0.05, detector_threshold=0.0, saturation_limit=8,
                                      energy_map=energies, use_gpu=False)
            level = engine.activate([layout._points[centre].code]).active_detectors[0].activation_level
            self.assertAlmostEqual(level, 1.0, places=6, msg="код из центра активирует детектор ровно на e_d (§6.5)")

    def test_jaccard2_is_the_article_formula_for_bit_vectors(self) -> None:
        """§2.2.4.2: J2 = Σ a·b / Σ max(a², b²); для битов это |a∧b| / |a∨b| = J."""
        import types

        from damp.decode.mnist_decoder import EmbeddingDatabase

        a = BitArray.from_indices(16, [0, 1, 2, 3, 4, 5])
        b = BitArray.from_indices(16, [0, 1, 2, 3, 8, 9])
        bits_a, bits_b = [int(c) for c in a.to01()], [int(c) for c in b.to01()]
        article_j2 = sum(x * y for x, y in zip(bits_a, bits_b)) / sum(max(x * x, y * y) for x, y in zip(bits_a, bits_b))
        for name in ("jaccard", "jaccard2"):
            database = types.SimpleNamespace(_config=types.SimpleNamespace(similarity=name))
            got = EmbeddingDatabase._similarity_value(database, a, a.count(), b, b.count())
            self.assertAlmostEqual(got, article_j2)
        from damp.activation.activation import _similarity_name

        self.assertEqual(_similarity_name("jaccard2"), "jaccard")

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
        if main.ENCODER_SHARED_CODE:
            self.assertEqual(encoder.code_length, main.ENCODER_SHARED_CODE_LENGTH)
        else:
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
                active = set(entry.code.active_indices())
                for dimension, own in position_segments[(x, y)].items():
                    value = x if dimension == "X" else y
                    bits = frozenset(reference.dimension_bits(dimension, value))
                    self.assertTrue(bits <= active, f"биты {dimension} позиции {(x, y)} входят в код")
                    own.add(bits)
        for position, dimensions in position_segments.items():
            for dimension, own in dimensions.items():
                self.assertEqual(len(own), 1, f"код {dimension} позиции {position} должен быть постоянным")
                self.assertTrue(next(iter(own)))
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


def _random_layout(seed: int = 3, *, points: int = 40, grid_size: int = 9, use_gpu: bool = False, **kwargs) -> Layout:
    """Случайные разреженные коды на сетке с пустыми ячейками — для сверки GPU-энергий с CPU-формулами."""
    import random as _random

    rng = _random.Random(seed)
    codes: dict[float, list] = {}
    for index in range(points):
        bits = rng.sample(range(64), 12)
        codes.setdefault(float(index % 7), []).append(
            LayoutEncodedCode(code=BitArray.from_indices(64, bits), label=str(index))
        )
    params = dict(similarity="cosine", lambda_threshold=0.2, eta=None, use_gpu=use_gpu, seed=seed, grid_size=grid_size)
    params.update(kwargs)
    return Layout(codes, **params)


class GpuPairEnergyTests(unittest.TestCase):
    """§5.5, §5.7, §5.9.4, §5.12: энергии пар и точек на GPU равны CPU-формулам; выбор пар векторный."""

    def _gpu_layout(self, **kwargs) -> Layout:
        layout = _random_layout(use_gpu=True, **kwargs)
        if layout._torch_engine is None:
            self.skipTest("torch с GPU недоступен")
        return layout

    def _pairs(self, layout: Layout, count: int = 60):
        import numpy as np

        layout._sync_arrays()
        rng = np.random.default_rng(11)
        rows = []
        while len(rows) < count:
            y1, x1 = int(rng.integers(0, layout.height)), int(rng.integers(0, layout.width))
            y2, x2 = int(rng.integers(0, layout.height)), int(rng.integers(0, layout.width))
            if (y1, x1) == (y2, x2) or (layout.grid[y1][x1] is None and layout.grid[y2][x2] is None):
                continue
            if layout.grid[y1][x1] is None:
                y1, x1, y2, x2 = y2, x2, y1, x1
            rows.append((y1, x1, y2, x2))
        return rows

    def test_neighbourhood_energies_match_cpu_for_both_kernels(self) -> None:
        import numpy as np

        for kernel, distance in (("gauss", "squared"), ("inverse", "squared"), ("inverse", "euclidean")):
            layout = self._gpu_layout(long_kernel="gauss", short_distance=distance, long_kernel_width=1.0)
            pairs = self._pairs(layout)
            idx_a = np.array([layout.grid[y1][x1] for y1, x1, _, _ in pairs])
            idx_b = np.array([-1 if layout.grid[y2][x2] is None else layout.grid[y2][x2] for _, _, y2, x2 in pairs])
            pos_a = np.array([(y1, x1) for y1, x1, _, _ in pairs], dtype=float)
            pos_b = np.array([(y2, x2) for _, _, y2, x2 in pairs], dtype=float)
            layout._ensure_torch_positions()
            radius = 3
            energy_c, energy_s = layout._torch_engine.pair_energies(
                idx_a, idx_b, pos_a, pos_b, neighbourhood=True, local_radius=radius, kernel=kernel,
                euclidean=(distance == "euclidean"), distance_eps=layout._distance_eps,
                kernel_width=layout._long_kernel_width, lambda_threshold=layout._lambda, eta=layout._eta,
            )
            for i, (y1, x1, y2, x2) in enumerate(pairs):
                r = layout._short_pair_radius(y1, x1, y2, x2, radius)
                cpu_c, cpu_s = layout._pair_energy_short(
                    int(idx_a[i]), (y1, x1), None if idx_b[i] < 0 else int(idx_b[i]), (y2, x2), radius=r, kernel=kernel
                )
                self.assertAlmostEqual(float(energy_c[i]), cpu_c, places=4, msg=f"{kernel}/{distance} phi_c pair {i}")
                self.assertAlmostEqual(float(energy_s[i]), cpu_s, places=4, msg=f"{kernel}/{distance} phi_s pair {i}")

    def test_far_energies_match_cpu_formula_over_subset(self) -> None:
        """§5.5.1 по подмножеству B ⊆ V (§5.9.4): φ_c = Σ_{v∈B} s1·d1 + s2·d2 с евклидовым d."""
        import numpy as np

        layout = self._gpu_layout(long_distance="euclidean")
        pairs = self._pairs(layout, count=30)
        subset = list(range(0, layout._point_count, 2))
        layout._ensure_torch_positions()
        idx_a = np.array([layout.grid[y1][x1] for y1, x1, _, _ in pairs])
        idx_b = np.array([-1 if layout.grid[y2][x2] is None else layout.grid[y2][x2] for _, _, y2, x2 in pairs])
        energy_c, energy_s = layout._torch_engine.pair_energies(
            idx_a, idx_b, np.array([p[:2] for p in pairs], dtype=float), np.array([p[2:] for p in pairs], dtype=float),
            neighbourhood=False, local_radius=None, kernel="inverse", euclidean=True,
            distance_eps=layout._distance_eps, kernel_width=1.0, lambda_threshold=layout._lambda, eta=layout._eta,
            eval_indices=subset,
        )
        for i, (y1, x1, y2, x2) in enumerate(pairs):
            expected_c = expected_s = 0.0
            for v in subset:
                if v == idx_a[i] or v == idx_b[i]:
                    continue
                s1 = layout._sim_lambda_idx(int(idx_a[i]), v)
                s2 = 0.0 if idx_b[i] < 0 else layout._sim_lambda_idx(int(idx_b[i]), v)
                vy, vx = layout._positions[v]
                d1 = math.hypot(y1 - vy, x1 - vx)
                d2 = math.hypot(y2 - vy, x2 - vx)
                expected_c += s1 * d1 + s2 * d2
                expected_s += s2 * d1 + s1 * d2
            self.assertAlmostEqual(float(energy_c[i]), expected_c, places=3)
            self.assertAlmostEqual(float(energy_s[i]), expected_s, places=3)

    def test_point_energies_on_gpu_match_cpu(self) -> None:
        for points, grid_size in ((40, 9), (300, 24)):  # малое окно → полный проход; большое V → окно ячеек
            layout = self._gpu_layout(points=points, grid_size=grid_size)
            gpu = layout._compute_point_energies(3)
            cpu = layout._point_energies_cpu(3)
            self.assertEqual(len(gpu), len(cpu))
            for got, want in zip(gpu, cpu):
                self.assertAlmostEqual(got, want, places=4)

    def test_local_window_energies_match_full_pass_and_cpu(self) -> None:
        """Окно ячеек вокруг центра пары даёт те же φ_c, φ_s, что проход по всему V и CPU (§5.5.2)."""
        import numpy as np

        layout = self._gpu_layout(points=300, grid_size=24, long_kernel="gauss", long_kernel_width=1.0)
        pairs = self._pairs(layout, count=80)
        idx_a = np.array([layout.grid[y1][x1] for y1, x1, _, _ in pairs])
        idx_b = np.array([-1 if layout.grid[y2][x2] is None else layout.grid[y2][x2] for _, _, y2, x2 in pairs])
        pos_a = np.array([(y1, x1) for y1, x1, _, _ in pairs], dtype=float)
        pos_b = np.array([(y2, x2) for _, _, y2, x2 in pairs], dtype=float)
        layout._ensure_torch_positions()
        engine = layout._torch_engine
        for kernel in ("gauss", "inverse"):
            common = dict(neighbourhood=True, local_radius=2, kernel=kernel, euclidean=False,
                          distance_eps=layout._distance_eps, kernel_width=1.0,
                          lambda_threshold=layout._lambda, eta=layout._eta)
            local_c, local_s = engine.pair_energies(idx_a, idx_b, pos_a, pos_b, **common)
            # полный проход — тот же вызов с явным списком всех точек (окно не используется)
            full_c, full_s = engine.pair_energies(
                idx_a, idx_b, pos_a, pos_b, eval_indices=list(range(layout._point_count)), **common
            )
            for i, (y1, x1, y2, x2) in enumerate(pairs):
                r = layout._short_pair_radius(y1, x1, y2, x2, 2)
                cpu_c, cpu_s = layout._pair_energy_short(
                    int(idx_a[i]), (y1, x1), None if idx_b[i] < 0 else int(idx_b[i]), (y2, x2), radius=r, kernel=kernel
                )
                self.assertAlmostEqual(float(local_c[i]), cpu_c, places=4, msg=f"{kernel} local phi_c {i}")
                self.assertAlmostEqual(float(local_s[i]), cpu_s, places=4, msg=f"{kernel} local phi_s {i}")
                self.assertAlmostEqual(float(full_c[i]), cpu_c, places=4, msg=f"{kernel} full phi_c {i}")
                self.assertAlmostEqual(float(full_s[i]), cpu_s, places=4, msg=f"{kernel} full phi_s {i}")

    def test_vectorised_pair_sampling_respects_radius_bounds_and_cutoff(self) -> None:
        import numpy as np

        layout = _random_layout(points=30, grid_size=8)
        layout._sync_arrays()
        pairs = layout._sample_pairs_long_np(200, 2)
        self.assertEqual(pairs.shape, (200, 4))
        self.assertTrue(np.all((pairs[:, 0] != pairs[:, 2]) | (pairs[:, 1] != pairs[:, 3])), "ячейки пары различны")
        self.assertTrue(np.all(pairs >= 0) and np.all(pairs[:, [0, 2]] < layout.height) and np.all(pairs[:, [1, 3]] < layout.width))
        dist_sq = (pairs[:, 0] - pairs[:, 2]) ** 2 + (pairs[:, 1] - pairs[:, 3]) ** 2
        self.assertGreater(float(np.mean(dist_sq <= 4)), 0.9, "вторая точка почти всегда в радиусе r (§5.3)")
        # плитка из одной ячейки с первой точкой расширяется до всей сетки (§5.10)
        y1 = np.array([layout._positions[0][0]])
        x1 = np.array([layout._positions[0][1]])
        y2, x2 = layout._sample_nearby_np(y1, x1, 1, bounds=(int(y1[0]), int(y1[0]), int(x1[0]), int(x1[0])))
        self.assertNotEqual((int(y2[0]), int(x2[0])), (int(y1[0]), int(x1[0])))
        # отсечка по сходству (§5.9.1): при cutoff > 1 все пары с занятой второй ячейкой отбрасываются,
        # недобор восполняется парами дальней фазы
        strict = layout._sample_pairs_short_tile_np(
            40, pair_radius=None, similarity_cutoff=1.01, weighted_first=False, step_index=0
        )
        self.assertEqual(strict.shape, (40, 4))
        loose = layout._sample_pairs_short_tile_np(
            40, pair_radius=2, similarity_cutoff=None, weighted_first=False, step_index=0
        )
        self.assertEqual(loose.shape, (40, 4))
        partitioned = layout._sample_pairs_short_np(
            45, pair_radius=2, similarity_cutoff=None, weighted_first=True, step_index=0, partitions=4
        )
        self.assertEqual(partitioned.shape, (45, 4))


if __name__ == "__main__":
    unittest.main()
