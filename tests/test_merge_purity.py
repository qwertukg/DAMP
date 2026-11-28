"""Чистота слияния (system/damp/decode/merge_purity.py; мерка, вне статьи): одной ли цифры картинки, чьи окна слились в
одну точку карты памяти. Идеал автора (2026-11-18): на последнем уровне остаётся 10 классов."""

from __future__ import annotations

import os
import sys
import unittest


os.environ.setdefault("RERUN", "off")
os.environ.setdefault("DAMP_LOG_FILE", "off")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "system"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "client"))

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


    def test_int8_codes_find_the_same_nearest_points_as_float32(self) -> None:
        """Мерка на видеокарте с кодами int8 (ACTIVATION_INT8_CODES) даёт те же ближайшие точки, что float32: число общих
        бит — то же целое, косинус — та же формула. Код в 299 бит и карта в 501 точку — ни то, ни другое не кратно 8;
        частей кодов — короче и длиннее 17 строк."""
        import torch

        from damp import settings

        if not torch.cuda.is_available() or not hasattr(torch, "_int_mm"):
            self.skipTest("нужны CUDA и torch._int_mm")
        rng = np.random.default_rng(3)
        length, points = 299, 501
        prototypes = rng.random((20, length)) < 0.2
        bank = prototypes[np.arange(points) % 20] ^ (rng.random((points, length)) < 0.03)
        dictionary = CodeDictionary(packed=np.packbits(bank, axis=1), rows=np.arange(points), code_length=length, distinct=points)
        for count in (5, 2048 + 40):
            codes = prototypes[rng.integers(0, 20, size=count)] ^ (rng.random((count, length)) < 0.05)
            packed = np.packbits(codes, axis=1)
            measure = MergePurity(use_gpu=True)
            with settings.overridden({"ACTIVATION_INT8_CODES": False}):
                plain = measure.nearest(packed, length, dictionary)
            with settings.overridden({"ACTIVATION_INT8_CODES": True}):
                fast = measure._nearest_int8(torch, torch.device("cuda"), packed, length, dictionary)
                used = measure.nearest(packed, length, dictionary)
            self.assertIsNotNone(fast, f"кодов {count}: сверка с float32 не прошла")
            self.assertTrue(np.array_equal(plain, fast), f"кодов {count}")
            self.assertTrue(np.array_equal(plain, used), f"кодов {count}")


    def test_rows_without_copy_give_the_same_measure(self) -> None:
        """Коды по номерам (rows) — то же, что копия этих кодов: на процессоре, на видеокарте float32 и int8."""
        import torch

        from damp import settings

        rng = np.random.default_rng(11)
        length, points = 301, 300
        prototypes = rng.random((10, length)) < 0.2
        bank = prototypes[np.arange(points) % 10] ^ (rng.random((points, length)) < 0.03)
        dictionary = CodeDictionary(packed=np.packbits(bank, axis=1), rows=np.arange(points), code_length=length, distinct=points)
        codes = np.packbits(prototypes[rng.integers(0, 10, size=5000)] ^ (rng.random((5000, length)) < 0.05), axis=1)
        rows = rng.permutation(5000)[:3100]
        picture = rng.integers(0, 400, size=rows.shape[0])
        labels = rng.integers(0, 10, size=rows.shape[0])
        devices = [False] + ([True] if torch.cuda.is_available() else [])
        for use_gpu in devices:
            for int8 in (False, True):
                with settings.overridden({"ACTIVATION_INT8_CODES": int8}):
                    measure = MergePurity(seed=3, use_gpu=use_gpu)
                    copied = measure.measure(codes[rows], length, picture, labels, dictionary, name="test")
                    indexed = measure.measure(codes, length, picture, labels, dictionary, name="test", rows=rows)
                    self.assertEqual(copied, indexed, f"видеокарта {use_gpu}, int8 {int8}")


if __name__ == "__main__":
    unittest.main()
