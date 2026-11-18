"""Пятна (system/damp/cycle/lateral_inhibition.py; П.2, П.3): из соседних деталей с похожим кодом на следующий уровень
идёт одна, сильнейшая; непохожие соседи остаются."""

from __future__ import annotations

import os
import sys
import unittest


os.environ.setdefault("RERUN", "off")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "system"))

import numpy as np  # noqa: E402

from damp import settings  # noqa: E402
from damp.cycle.lateral_inhibition import LateralInhibition  # noqa: E402
from damp.encoding.elements import LadderElements  # noqa: E402

LENGTH = 40


def _code(*ones: int) -> "np.ndarray":
    bits = np.zeros(LENGTH, dtype=np.uint8)
    bits[list(ones)] = 1
    return np.packbits(bits)


# Понятие «А» и почти то же «А′» (9 общих бит из 10: сходство 0.9), и другое понятие «Б».
A = _code(*range(0, 10))
A_NEAR = _code(*range(1, 11))
B = _code(*range(20, 30))


def _keep(codes, x, strength, *, similarity=0.8, share: float = 1.0, sight: float = 4.0, picture=None) -> list:
    count = len(x)
    picture = np.zeros(count, dtype=np.int64) if picture is None else np.asarray(picture)
    chooser = LateralInhibition(radius_share=share, similarity=similarity)
    return chooser.survivors(
        picture, np.zeros(count), np.asarray(x, dtype=float), np.asarray(strength, dtype=float), np.full(count, sight),
        codes=None if similarity is None else (np.stack(codes), LENGTH),
    ).tolist()


class DetailSpotsTest(unittest.TestCase):
    def test_similar_neighbour_goes_and_a_different_one_stays(self) -> None:
        """Сильная деталь гасит соседку с похожим кодом (одно пятно), а соседка с другим понятием остаётся."""
        self.assertEqual(_keep([A, A_NEAR, B], [0, 1, 2], [0.9, 0.5, 0.4]), [True, False, True])
        # Прежнее торможение (без порога сходства) гасило обеих — вместе с другим понятием.
        self.assertEqual(_keep([A, A_NEAR, B], [0, 1, 2], [0.9, 0.5, 0.4], similarity=None), [True, False, False])

    def test_one_detail_is_left_of_a_spot(self) -> None:
        """Пятно — несколько соседних окон с одним понятием: остаётся самая сильная деталь."""
        self.assertEqual(_keep([A, A_NEAR, A, A_NEAR, A], [0, 1, 2, 3, 4], [0.3, 0.6, 0.9, 0.5, 0.4], sight=6.0), [False, False, True, False, False])

    def test_same_concept_far_apart_stays_twice(self) -> None:
        """Одно понятие в двух местах картинки, дальше круга торможения, — две детали (П.6: признак в разных местах)."""
        self.assertEqual(_keep([A, A, B], [0, 9, 1], [0.9, 0.8, 0.7]), [True, True, True])

    def test_pictures_do_not_touch_each_other(self) -> None:
        self.assertEqual(_keep([A, A, A, A], [0, 1, 0, 1], [0.9, 0.5, 0.4, 0.8], picture=[0, 0, 1, 1]), [True, False, False, True])

    def test_threshold_decides_what_is_one_concept(self) -> None:
        self.assertEqual(_keep([A, A_NEAR], [0, 1], [0.9, 0.5], similarity=0.95), [True, True])  # 0.9 < 0.95 — разные понятия
        self.assertEqual(_keep([A, A_NEAR], [0, 1], [0.9, 0.5], similarity=0.85), [True, False])

    def test_codes_are_needed_with_the_threshold(self) -> None:
        with self.assertRaises(ValueError):
            LateralInhibition(radius_share=1.0, similarity=0.8).survivors(
                np.zeros(2, dtype=np.int64), np.zeros(2), np.arange(2.0), np.ones(2), np.full(2, 4.0)
            )
        with self.assertRaises(ValueError):
            LateralInhibition(radius_share=1.0, similarity=1.5)

    def test_thin_follows_the_settings(self) -> None:
        """LateralInhibition.thin — один вход для прогона цикла и пути одной картинки: читает настройки."""
        what = np.stack([A, A_NEAR, B])
        elements = LadderElements(
            picture=np.zeros(3, dtype=np.int64), y=np.zeros(3), x=np.arange(3.0), what_packed=what, what_length=LENGTH,
            what_index=np.arange(3), extent=np.full(3, 4.0), strength=np.array([0.9, 0.5, 0.4]),
        )
        with settings.overridden({"LADDER_INHIBITION_RADIUS": None}):
            self.assertIs(LateralInhibition.thin(elements, "test"), elements)
        with settings.overridden({"LADDER_INHIBITION_RADIUS": 1.0, "LADDER_INHIBITION_SIMILARITY": 0.8}):
            self.assertEqual(LateralInhibition.thin(elements, "test").x.tolist(), [0.0, 2.0])
        with settings.overridden({"LADDER_INHIBITION_RADIUS": 1.0, "LADDER_INHIBITION_SIMILARITY": None}):
            self.assertEqual(LateralInhibition.thin(elements, "test").x.tolist(), [0.0])


if __name__ == "__main__":
    unittest.main()
