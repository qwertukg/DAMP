"""Чистота слияния (system/damp/decode/merge_purity.py; мерка, вне статьи): одной ли цифры картинки, чьи окна слились в
одну точку карты памяти. Идеал автора (2026-11-18): на последнем уровне остаётся 10 классов."""

from __future__ import annotations

import os
import sys
import unittest


os.environ.setdefault("RERUN", "off")
os.environ.setdefault("DAMP_LOG_FILE", "off")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "system"))

import numpy as np  # noqa: E402

from damp.decode.merge_purity import MergePurity  # noqa: E402
from damp.layout.code_dictionary import CodeDictionary  # noqa: E402

LENGTH = 32


def _code(*ones: int) -> "np.ndarray":
    bits = np.zeros(LENGTH, dtype=np.uint8)
    bits[list(ones)] = 1
    return np.packbits(bits)


# Три образца карты: «А», «Б», «В» — без общих бит.
A, B, C = _code(*range(0, 8)), _code(*range(8, 16)), _code(*range(16, 24))
A_NEAR = _code(*range(1, 9))  # 7 общих бит с «А» из 8
B_NEAR = _code(*range(9, 17))


def _dictionary() -> CodeDictionary:
    return CodeDictionary(packed=np.stack([A, B, C]), rows=np.arange(3), code_length=LENGTH, distinct=3)


class MergePurityTest(unittest.TestCase):
    def test_keyhole_belongs_to_its_most_similar_point(self) -> None:
        nearest = MergePurity(use_gpu=False).nearest(np.stack([A, A_NEAR, B_NEAR, C]), LENGTH, _dictionary())
        self.assertEqual(nearest.tolist(), [0, 0, 1, 2])

    def test_purity_counts_pictures_of_shared_points(self) -> None:
        # Точка «А»: окна картинок 0, 1 (цифра 3) и 2 (цифра 5) — общая, главная цифра 3: две картинки из трёх.
        # Точка «Б»: окна картинок 3 и 4, обе цифры 7 — общая и чистая. Точка «В»: два окна одной картинки 5.
        codes = np.stack([A, A_NEAR, A, B, B_NEAR, C, C])
        picture = np.array([0, 1, 2, 3, 4, 5, 5])
        labels = np.array([3, 3, 5, 7, 7, 1, 1])
        found = MergePurity(use_gpu=False).measure(codes, LENGTH, picture, labels, _dictionary(), name="test")
        self.assertAlmostEqual(found["merge_pictures_per_point"], (3 + 2 + 1) / 3)
        self.assertAlmostEqual(found["merge_shared_points"], 2 / 3)
        self.assertAlmostEqual(found["merge_purity"], (2 + 2) / (3 + 2))
        self.assertAlmostEqual(found["merge_pure_points"], 1 / 2)
        self.assertAlmostEqual(found["merge_digits_per_point"], (2 + 1) / 2)

    def test_several_keyholes_of_one_picture_count_once(self) -> None:
        # Пять окон картинки 0 и одно окно картинки 1 в одной точке: это две картинки, а не шесть окон.
        codes = np.stack([A, A, A, A_NEAR, A, A])
        picture = np.array([0, 0, 0, 0, 0, 1])
        labels = np.array([4, 4, 4, 4, 4, 9])
        found = MergePurity(use_gpu=False).measure(codes, LENGTH, picture, labels, _dictionary(), name="test")
        self.assertAlmostEqual(found["merge_purity"], 1 / 2)
        self.assertAlmostEqual(found["merge_pure_points"], 0.0)

    def test_nothing_is_mixed_when_no_point_is_shared(self) -> None:
        found = MergePurity(use_gpu=False).measure(
            np.stack([A, B, C]), LENGTH, np.array([0, 1, 2]), np.array([1, 2, 3]), _dictionary(), name="test"
        )
        self.assertEqual((found["merge_shared_points"], found["merge_purity"], found["merge_pure_points"]), (0.0, 1.0, 1.0))

    def test_ten_pure_points_are_the_ideal(self) -> None:
        """Идеал: по точке на цифру, в каждой — картинки только этой цифры; случайная чистота заметно ниже."""
        samples = np.stack([_code(3 * digit, 3 * digit + 1, 3 * digit + 2) for digit in range(10)])
        dictionary = CodeDictionary(packed=samples, rows=np.arange(10), code_length=LENGTH, distinct=10)
        labels = np.repeat(np.arange(10), 6)
        found = MergePurity(use_gpu=False).measure(samples[labels], LENGTH, np.arange(60), labels, dictionary, name="test")
        self.assertEqual((found["merge_shared_points"], found["merge_purity"], found["merge_pure_points"]), (1.0, 1.0, 1.0))
        self.assertAlmostEqual(found["merge_pictures_per_point"], 6.0)
        self.assertLess(found["merge_purity_random"], 0.6)


if __name__ == "__main__":
    unittest.main()
