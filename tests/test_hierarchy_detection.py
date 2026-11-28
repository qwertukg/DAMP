"""Пакетное детектирование иерархией детекторов (§6.5) совпадает с ActivationEngine на одном стимуле."""

from __future__ import annotations

import os
import sys
import unittest


os.environ.setdefault("RERUN", "off")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "system"))

import numpy as np  # noqa: E402

from damp.activation import ActivationEngine, Detector  # noqa: E402
from damp.activation.hierarchy_detection import HierarchyDetection  # noqa: E402
from damp.encoding.bitarray import BitArray  # noqa: E402
from tests.test_article_invariants import _clustered_layout  # noqa: E402


class HierarchyDetectionTest(unittest.TestCase):
    @staticmethod
    def _detectors() -> list[Detector]:
        bit = BitArray.from_indices(16, [0])
        return [
            # Слой с более высоким порогом стоит первым: порядок слоёв HierarchyDetection выстраивает сам.
            Detector(center=(8.5, 8.5), radius=1.6, lambda_threshold=0.8, energy=2.0, points_count=4, output_code=bit),
            Detector(center=(2.5, 2.5), radius=3.0, lambda_threshold=0.5, energy=6.0, points_count=16, output_code=bit),
            Detector(center=(8.5, 8.5), radius=3.0, lambda_threshold=0.5, energy=6.0, points_count=16, output_code=bit),
            Detector(center=(2.5, 2.5), radius=1.6, lambda_threshold=0.8, energy=2.0, points_count=4, output_code=bit),
        ]

    def test_matches_the_activation_engine_on_each_stimulus(self) -> None:
        layout = _clustered_layout()
        detectors = self._detectors()
        energies = [1.0 if point % 5 else 0.05 for point in range(32)]  # часть точек ниже порога энергии
        settings = {"lambda_threshold": 0.5, "eta": None, "energy_threshold": 0.12, "detector_threshold": 0.5}
        engine = ActivationEngine(layout, detectors, saturation_limit=4, energy_map=energies, use_gpu=False, **settings)
        detection = HierarchyDetection(layout, detectors, energies, similarity="cosine", use_gpu=False, **settings)
        self.assertEqual(detection.layer_index.tolist(), [0, 0, 1, 1])
        self.assertEqual([d.lambda_threshold for d in detection.detectors], [0.5, 0.5, 0.8, 0.8])
        stimuli = [
            BitArray.from_indices(64, [0, 1, 2, 3, 4, 5, 6, 7]),
            BitArray.from_indices(64, [40, 41, 42, 43, 44, 45, 46, 47]),
            BitArray.from_indices(64, [0, 1, 2, 3, 40, 41, 42, 43]),
            BitArray.from_indices(64, [56, 57, 58, 59, 60, 61, 62, 63]),
        ]
        packed = np.packbits(np.asarray([list(code) for code in stimuli], dtype=np.uint8), axis=1)
        rows, found = detection.active(packed)
        for number, code in enumerate(stimuli):
            _, active = engine.detect([code])
            expected = {id(item.detector) for item in active}
            mine = {id(detection.detectors[int(index)]) for index in found[rows == number]}
            self.assertEqual(mine, expected, f"stimulus {number}")
        self.assertEqual(found[rows == 3].size, 0)  # непохожий код ничего не зажигает

    def test_best_point_lights_even_below_the_threshold(self) -> None:
        layout = _clustered_layout()
        detectors = self._detectors()
        energies = [1.0] * 32
        settings = {"similarity": "cosine", "eta": None, "energy_threshold": 0.12, "detector_threshold": 0.05, "use_gpu": False}
        far = np.packbits(np.asarray([list(BitArray.from_indices(64, [0, 1, 2, 3, 50, 51, 52, 53]))], dtype=np.uint8), axis=1)
        strict = HierarchyDetection(layout, detectors, energies, lambda_threshold=0.9, **settings)
        self.assertEqual(strict.active(far)[1].size, 0)  # сходство с любой точкой ниже 0.9: ничего не зажглось
        lenient = HierarchyDetection(layout, detectors, energies, lambda_threshold=0.9, activate_best=True, **settings)
        self.assertGreater(lenient.active(far)[1].size, 0)  # самая похожая точка зажжена и подняла детектор

    def _local_detection(self, layout, threshold: float) -> HierarchyDetection:  # noqa: ANN001
        return HierarchyDetection(
            layout, self._detectors(), [1.0] * 32, similarity="cosine", lambda_threshold=0.5, eta=None,
            energy_threshold=0.12, detector_threshold=0.5, local_energy_threshold=threshold, local_energy_radius=3.0,
            use_gpu=False,
        )

    def test_local_energy_silences_a_lonely_point_and_keeps_the_patch(self) -> None:
        """Энергия активации (§5.7 на матрице активации, расширение статьи): одиночная точка гаснет, пятно остаётся."""
        import torch

        layout = _clustered_layout()
        cells = layout.cells_array().astype(np.float64)
        distance = np.hypot(cells[:, 0] - cells[0, 0], cells[:, 1] - cells[0, 1])
        patch = np.flatnonzero(distance <= 2.0)
        lonely = int(np.argmax(distance))
        self.assertGreater(distance[lonely], 5.0)  # вокруг одиночной точки в радиусе 3 нет точек пятна
        self.assertGreaterEqual(patch.size, 5)
        activation = np.zeros((1, cells.shape[0]), dtype=np.float32)
        activation[0, patch] = 0.8
        activation[0, lonely] = 0.8
        kept = self._local_detection(layout, 0.2)._local(torch.as_tensor(activation)).numpy()[0]
        self.assertEqual(kept[lonely], 0.0)
        self.assertGreater(kept[0], 0.0)  # середина пятна горит
        self.assertGreater(np.count_nonzero(kept[patch]), patch.size // 2)
        off = self._local_detection(layout, 0.0)._local(torch.as_tensor(activation)).numpy()[0]
        np.testing.assert_array_equal(off, activation[0])  # 0 — выключено: активация как в статье

    def test_local_energy_does_not_depend_on_the_batch(self) -> None:
        layout = _clustered_layout()
        detection = self._local_detection(layout, 0.2)
        stimuli = [
            BitArray.from_indices(64, [0, 1, 2, 3, 4, 5, 6, 7]),
            BitArray.from_indices(64, [40, 41, 42, 43, 44, 45, 46, 47]),
            BitArray.from_indices(64, [0, 1, 2, 3, 40, 41, 42, 43]),
        ]
        packed = np.packbits(np.asarray([list(code) for code in stimuli], dtype=np.uint8), axis=1)
        rows, found = detection.active(packed)
        for number in range(len(stimuli)):
            alone_rows, alone = detection.active(packed[number : number + 1])
            self.assertEqual(sorted(alone.tolist()), sorted(found[rows == number].tolist()), f"stimulus {number}")
        sets_rows, sets_found, _ = detection.active_sets(packed, np.arange(3), np.arange(3), 3)
        self.assertEqual(sorted(zip(sets_rows.tolist(), sets_found.tolist())), sorted(zip(rows.tolist(), found.tolist())))

    def test_int8_codes_give_the_same_firing(self) -> None:
        """Коды int8 на видеокарте (ACTIVATION_INT8_CODES) дают те же срабатывания и зажжённые точки до бита, что float32:
        число общих бит — целое у обоих произведений. Карта — 501 точка с кодом в 99 бит (ни то, ни другое не кратно 8),
        пачки кодов короче и длиннее 17 строк."""
        import torch

        if not torch.cuda.is_available() or not hasattr(torch, "_int_mm"):
            self.skipTest("нужны CUDA и torch._int_mm")
        from damp.layout.damp_layout import Layout
        from tests.test_layout_int8_codes import _LAYOUT_PARAMS, _table

        layout = Layout(_table(), **_LAYOUT_PARAMS)
        points, side = 501, int(layout.width)
        rng = np.random.default_rng(7)
        bit = BitArray.from_indices(16, [0])
        detectors = [
            Detector(center=(float(rng.uniform(2, side - 2)), float(rng.uniform(2, side - 2))), radius=float(rng.uniform(2.0, 5.0)),
                     lambda_threshold=(0.3, 0.5)[number % 2], energy=float(rng.uniform(1.0, 4.0)), points_count=8, output_code=bit)
            for number in range(24)
        ]
        energies = [1.0 if point % 7 else 0.05 for point in range(points)]
        common = dict(similarity="cosine", lambda_threshold=0.3, eta=None, energy_threshold=0.12, detector_threshold=0.2,
                      activate_best=True, local_energy_threshold=0.2, local_energy_radius=3.0)
        plain = HierarchyDetection(layout, detectors, energies, **common)
        fast = HierarchyDetection(layout, detectors, energies, int8_codes=True, **common)
        self.assertFalse(plain._int8)
        self.assertTrue(fast._int8)
        self.assertTrue(bool(torch.equal(plain._ones, fast._ones)))
        bits = np.asarray(layout.code_table.bits(), dtype=np.uint8)
        fired = 0
        for count in (1, 5, 16, 17, 40, 300):
            codes = bits[rng.integers(0, points, size=count)].copy()
            flips = rng.integers(0, codes.shape[1], size=(count, 3))
            codes[np.arange(count)[:, None], flips] ^= 1  # коды окон — похожие на точки карты, но не равные им
            packed = np.packbits(codes, axis=1)
            for method in ("active", "lit", "active_with_lit"):
                for expected, found in zip(getattr(plain, method)(packed), getattr(fast, method)(packed)):
                    self.assertTrue(np.array_equal(expected, found), f"{method}, кодов {count}")
            groups = np.sort(rng.integers(0, max(1, count // 3), size=count))
            index = rng.permutation(count)
            expected = plain.active_sets(packed, index, groups, max(1, count // 3))
            found = fast.active_sets(packed, index, groups, max(1, count // 3))
            for left, right in zip(expected, found):
                self.assertTrue(np.array_equal(left, right), f"active_sets, кодов {count}")
            fired += int(expected[0].size)
        self.assertGreater(fired, 0)  # сверка не пустая: детекторы срабатывали

    def test_int8_codes_fall_back_to_float32_when_the_check_fails(self) -> None:
        import torch

        if not torch.cuda.is_available() or not hasattr(torch, "_int_mm"):
            self.skipTest("нужны CUDA и torch._int_mm")
        layout = _clustered_layout()
        settings = {"similarity": "cosine", "lambda_threshold": 0.5, "eta": None, "energy_threshold": 0.12, "detector_threshold": 0.5}
        check = HierarchyDetection._int8_matches
        HierarchyDetection._int8_matches = lambda self, packed: False
        try:
            detection = HierarchyDetection(layout, self._detectors(), [1.0] * 32, int8_codes=True, **settings)
        finally:
            HierarchyDetection._int8_matches = check
        self.assertFalse(detection._int8)
        self.assertEqual(detection._points.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
