"""Пакетное детектирование иерархией детекторов (§6.5) совпадает с ActivationEngine на одном стимуле."""
from __future__ import annotations

import os
import sys
import unittest

os.environ.setdefault("RERUN", "off")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from damp.activation import ActivationEngine, Detector  # noqa: E402
from damp.activation.hierarchy_detection import DetectedCodes, HierarchyDetection  # noqa: E402
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

    def test_detected_codes_return_the_detectors_of_requested_codes(self) -> None:
        rows = np.asarray([2, 0, 2, 0, 0])
        detectors = np.asarray([5, 1, 3, 4, 0])
        what = DetectedCodes("Shape", rows, detectors, np.asarray([0, 0, 0, 1, 1, 1]), count=3)
        self.assertEqual(what.dimensions[0].title, "Shape")
        asked, found = what.active_detectors(0, np.asarray([2.0, 1.0, 0.0]))
        self.assertEqual(sorted(found[asked == 0].tolist()), [3, 5])
        self.assertEqual(found[asked == 1].size, 0)
        self.assertEqual(sorted(found[asked == 2].tolist()), [0, 1, 4])
        self.assertEqual(what.detector_layer_indices().tolist(), [0, 0, 0, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
