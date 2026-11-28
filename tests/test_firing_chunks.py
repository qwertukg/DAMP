"""Firing: strength, train_maxima и scaled кусками (Firing.CHUNK) — те же числа до бита, что одним куском."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "client"), os.path.join(ROOT, "system")]
os.environ.setdefault("RERUN", "off")
os.environ.setdefault("DAMP_RERUN_VIEWER", "off")

from damp.activation.level_detectors import Firing  # noqa: E402


class FiringChunksTests(unittest.TestCase):
    def test_chunks_give_the_same_numbers(self) -> None:
        rng = np.random.default_rng(4)
        sets, detectors, count = 300, 45, 5000
        rows = np.sort(rng.integers(0, sets, count)).astype(np.int64)
        columns = rng.integers(0, 40, count).astype(np.int64)
        trained = rng.random(sets) < 0.7
        untaught = np.flatnonzero(~trained[rows])
        columns[untaught[::3]] = 40 + np.arange(untaught[::3].size) % 5  # детекторы 40–44 — только на проверочных
        firing = Firing(sets=sets, rows=rows, columns=columns, levels=(rng.random(count) * 3.0).astype(np.float32))
        trained_columns = np.zeros(detectors, dtype=bool)
        trained_columns[firing.columns[trained[firing.rows]]] = True
        self.assertFalse(trained_columns.all())  # есть детекторы, не сработавшие на обучающих: делитель 0
        found = {}
        for chunk in (7, 1 << 22):
            Firing.CHUNK = chunk
            try:
                maxima = firing.train_maxima(trained, detectors)
                found[chunk] = (firing.strength(), maxima, firing.scaled(maxima).levels, firing.normalised(trained, detectors).levels)
            finally:
                Firing.CHUNK = 1 << 22
        for small, whole in zip(found[7], found[1 << 22]):
            self.assertEqual(small.dtype, whole.dtype)
            self.assertTrue(np.array_equal(small, whole))
        self.assertEqual(found[7][2].dtype, np.float32)
        self.assertLessEqual(float(found[7][2].max()), 1.0)

    def test_empty_firing(self) -> None:
        firing = Firing(sets=3, rows=np.zeros(0, dtype=np.int64), columns=np.zeros(0, dtype=np.int64), levels=np.zeros(0, dtype=np.float32))
        self.assertTrue(np.array_equal(firing.strength(), np.zeros(3)))
        self.assertEqual(firing.normalised(np.ones(3, dtype=bool), 5).levels.shape, (0,))


if __name__ == "__main__":
    unittest.main()
