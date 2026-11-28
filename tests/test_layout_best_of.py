"""«Лучшая из K» раскладок карты уровня (MemoryMap.lay_out, LADDER_LAYOUT_BEST_OF): попытки — раскладки уровня с гладким
стартом, выбор — по мерке без меток (LADDER_LAYOUT_BEST_OF_MEASURE)."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "client"), os.path.join(ROOT, "system")]
os.environ.setdefault("RERUN", "off")

from damp import settings  # noqa: E402
from damp.layout.level_memory import MemoryMap  # noqa: E402
from damp.logging import LOGGER  # noqa: E402

LOGGER.configure_intervals({}, default_interval=10**9)

_SMOOTH = {"LADDER_LAYOUT_SMOOTH_START_FROM": 4, "LADDER_LAYOUT_SMOOTH_START_PERPLEXITY": 10, "LADDER_LAYOUT_SMOOTH_START_RADIUS": 0.1,
           "LAYOUT_LOG_VISUALS": False}


def _codes(points: int = 160, length: int = 256, groups: int = 8, seed: int = 3) -> tuple["np.ndarray", "np.ndarray"]:
    """Коды из нескольких групп: внутри группы сходство высокое, между группами — низкое; все коды различны."""
    rng = np.random.default_rng(seed)
    prototypes = rng.random((groups, length)) < 0.2
    labels = np.arange(points) % groups
    bits = prototypes[labels].copy()
    flips = rng.random((points, length)) < 0.05
    bits ^= flips
    bits[np.arange(points), rng.integers(0, length, size=points)] = True
    return bits.astype(np.uint8), labels.astype(np.int64)


def _cells(bits: "np.ndarray", labels: "np.ndarray", changes: dict) -> tuple["np.ndarray", "np.ndarray"]:
    """Клетки точек и коды в порядке точек карты после раскладки уровня 4 с настройками changes."""
    with settings.overridden({**_SMOOTH, **changes}):
        layout = MemoryMap().lay_out(bits, labels, "level4")
    cells = np.asarray(layout.cells_array(), dtype=np.int64).copy()
    codes = np.asarray(layout.code_table.bits(), dtype=np.uint8).copy()
    layout.release_device()
    return cells, codes


class BestOfTests(unittest.TestCase):
    def test_near_distance_equals_a_direct_count(self) -> None:
        bits, _ = _codes(points=40)
        rng = np.random.default_rng(5)
        cells = np.stack([rng.permutation(49)[:40] // 7, rng.permutation(49)[:40] % 7], axis=1)
        codes = bits.astype(np.float64)
        expected = 0.0
        for point in range(40):
            sim = (codes @ codes[point]) / np.sqrt(codes.sum(axis=1) * codes[point].sum())
            sim[point] = -np.inf
            nearest = np.argsort(-sim, kind="stable")[:10]
            expected += float(np.sqrt(((cells[nearest] - cells[point]) ** 2).sum(axis=1)).sum())
        self.assertAlmostEqual(MemoryMap.near_distance(bits, cells), expected / 400, places=9)

    def test_best_of_takes_the_smooth_layout_with_the_least_measure(self) -> None:
        bits, labels = _codes()
        base = int(settings.LAYOUT_RANDOM_SEED)
        singles = [_cells(bits, labels, {"LAYOUT_RANDOM_SEED": base + 1009 * attempt}) for attempt in range(3)]
        # Попытка 0 — раскладка без выбора: то же зерно, тот же гладкий старт.
        plain, _ = _cells(bits, labels, {})
        self.assertTrue(np.array_equal(plain, singles[0][0]))
        self.assertFalse(np.array_equal(singles[0][0], singles[1][0]))  # зёрна дают разные раскладки — выбирать есть из чего
        for measure in ("near", "far"):
            chosen, codes = _cells(bits, labels, {"LADDER_LAYOUT_BEST_OF": 3, "LADDER_LAYOUT_BEST_OF_FROM": 4,
                                                  "LADDER_LAYOUT_BEST_OF_MEASURE": measure})
            if measure == "near":
                values = [MemoryMap.near_distance(codes_k, cells_k) for cells_k, codes_k in singles]
                expected = singles[int(np.argmin(values))][0]
            else:
                expected = None
                # Порог λ у попыток один (расписание уровня): раскладка с наименьшей φ среди трёх.
                with settings.overridden(_SMOOTH):
                    layout = MemoryMap().lay_out(bits, labels, "level4")
                    threshold = float(layout.lambda_threshold)
                    layout.release_device()
                values = [MemoryMap.far_energy(codes_k, cells_k, threshold) for cells_k, codes_k in singles]
                expected = singles[int(np.argmin(values))][0]
            self.assertTrue(np.array_equal(chosen, expected), measure)

    def test_measure_is_part_of_the_level_key_only_where_best_of_acts(self) -> None:
        with settings.overridden({"LADDER_LAYOUT_BEST_OF": 3, "LADDER_LAYOUT_BEST_OF_FROM": 4, "LADDER_LAYOUT_BEST_OF_MEASURE": "near"}):
            self.assertIsNone(MemoryMap.level_settings(3)["LADDER_LAYOUT_BEST_OF_MEASURE"])
            self.assertEqual(MemoryMap.level_settings(4)["LADDER_LAYOUT_BEST_OF_MEASURE"], "near")
            self.assertEqual(MemoryMap.level_settings(3)["LADDER_LAYOUT_BEST_OF"], 1)


if __name__ == "__main__":
    unittest.main()
