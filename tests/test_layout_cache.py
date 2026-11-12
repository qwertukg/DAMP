"""Кеш раскладки (вне статьи): готовая раскладка читается, если она считалась с тем же плановым радиусом пар."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

os.environ.setdefault("RERUN", "off")

import numpy as np  # noqa: E402

from damp.layout.payload import LayoutPayloadBuilder  # noqa: E402
from damp.logging import LOGGER  # noqa: E402
import main  # noqa: E402

LOGGER.configure_intervals({}, default_interval=10**9)

POINTS = 30
CODE_LENGTH = 96


class LayoutCacheRadiusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = main.LayoutParameterTuner().tune(POINTS)
        self.expected = main._expected_pair_radius(POINTS, self.params)

    def _saved(self, directory: str, *, planned: int | None, final: int) -> str:
        rng = np.random.default_rng(4)
        bits = np.zeros((POINTS, CODE_LENGTH), dtype=np.uint8)
        for row in range(POINTS):
            bits[row, rng.choice(CODE_LENGTH, size=12, replace=False)] = 1
        payload = LayoutPayloadBuilder(similarity=None, lambda_threshold=None, eta=None, metadata={})
        payload.add_packed(label=1, hues=[0.0] * POINTS, packed=np.packbits(bits, axis=1), code_length=CODE_LENGTH)
        if planned is not None:
            payload.add_metadata({main.PLANNED_PAIR_RADIUS_KEY: planned})
        path = os.path.join(directory, "layout.json")
        payload.save_base(path)
        layout = main._new_layout(payload.code_table(), self.params, None)
        layout.pair_radius = final
        layout.save_json(path)
        return path

    def _load(self, path: str):
        return main.LayoutCacheFile.load(path, self.params, expected_code_length=CODE_LENGTH)

    def test_layout_planned_with_expected_radius_is_reused_whatever_its_final_radius(self) -> None:
        # Большая раскладка режима статьи заканчивается на радиусе меньше планового.
        with tempfile.TemporaryDirectory() as directory:
            path = self._saved(directory, planned=self.expected, final=max(1, self.expected - 1) + 5)
            with open(path, "r", encoding="utf-8") as fp:
                self.assertEqual(json.load(fp)[main.PLANNED_PAIR_RADIUS_KEY], self.expected)
            layout = self._load(path)
            self.assertIsNotNone(layout)
            self.assertEqual(layout.point_count, POINTS)

    def test_layout_planned_with_another_radius_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._saved(directory, planned=self.expected + 3, final=self.expected)
            self.assertIsNone(self._load(path))

    def test_file_without_planned_radius_is_checked_by_final_radius(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNotNone(self._load(self._saved(directory, planned=None, final=self.expected)))
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(self._load(self._saved(directory, planned=None, final=self.expected + 2)))


if __name__ == "__main__":
    unittest.main()
