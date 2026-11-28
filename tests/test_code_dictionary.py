"""Отбор образцов карты (LeaderDictionary.select): обход без копии кодов, группы одинаковых кодов по хешу, коды int8 —
тот же словарь, что прежний счёт (копия в порядке обхода, np.unique по строкам, float32)."""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "client"), os.path.join(ROOT, "system")]
os.environ.setdefault("RERUN", "off")

from damp import settings  # noqa: E402
from damp.layout.code_dictionary import LeaderDictionary  # noqa: E402


def _codes(count: int = 3000, length: int = 203, seed: int = 4) -> "np.ndarray":
    """Коды с повторами и близкими кодами: группы вокруг прототипов, часть кодов — точные копии других."""
    rng = np.random.default_rng(seed)
    prototypes = rng.random((40, length)) < 0.2
    bits = prototypes[rng.integers(0, 40, size=count)] ^ (rng.random((count, length)) < 0.04)
    repeat = rng.integers(0, count, size=count // 5)
    bits[rng.integers(0, count, size=count // 5)] = bits[repeat]  # точные повторы
    return np.packbits(bits, axis=1)


def _reference(packed: "np.ndarray", length: int, order: "np.ndarray", threshold: float, size: int):  # noqa: ANN202
    """Прежний счёт: копия кодов в порядке обхода, группы — np.unique по строкам, банк float32 на процессоре."""
    copy = np.ascontiguousarray(packed[order])
    with settings.overridden({"LAYOUT_INT8_CODES": False}):
        return LeaderDictionary(similarity_threshold=threshold, max_size=size, use_gpu=False).select(copy, length)


class LeaderDictionaryTests(unittest.TestCase):
    def _check(self, found, expected, label: str) -> None:  # noqa: ANN001
        self.assertTrue(np.array_equal(found.rows, expected.rows), label)
        self.assertTrue(np.array_equal(found.packed, expected.packed), label)
        self.assertEqual(found.distinct, expected.distinct, label)

    def test_order_without_copy_gives_the_same_dictionary(self) -> None:
        packed, length = _codes(), 203
        order = np.random.default_rng(1).permutation(packed.shape[0])
        for threshold, size in ((0.9, 10**6), (0.6, 10**6), (0.9, 700)):
            expected = _reference(packed, length, order, threshold, size)
            with settings.overridden({"LAYOUT_INT8_CODES": False}):
                found = LeaderDictionary(similarity_threshold=threshold, max_size=size, use_gpu=False).select(packed, length, order=order)
            self._check(found, expected, f"порог {threshold}, предел {size}")

    def test_hash_collision_falls_back_to_exact_rows(self) -> None:
        packed, length = _codes(seed=6), 203
        order = np.random.default_rng(2).permutation(packed.shape[0])
        expected = _reference(packed, length, order, 0.9, 10**6)
        groups = LeaderDictionary._groups
        LeaderDictionary._groups = classmethod(lambda cls, packed, order, attempt=0: None)
        try:
            with settings.overridden({"LAYOUT_INT8_CODES": False}):
                found = LeaderDictionary(similarity_threshold=0.9, max_size=10**6, use_gpu=False).select(packed, length, order=order)
        finally:
            LeaderDictionary._groups = groups
        self._check(found, expected, "прежний путь")

    def test_groups_equal_unique_rows(self) -> None:
        packed = _codes(seed=8)
        order = np.random.default_rng(3).permutation(packed.shape[0])
        copy = np.ascontiguousarray(packed[order])
        _, first, counts = np.unique(copy.view(np.dtype((np.void, copy.shape[1]))).reshape(-1), return_index=True, return_counts=True)
        found = LeaderDictionary._groups(packed, order)
        self.assertIsNotNone(found)
        expected = sorted(zip(first.tolist(), counts.tolist()))
        self.assertEqual(sorted(zip(found[0].tolist(), found[1].tolist())), expected)

    def test_codes_differing_in_top_word_bits_are_not_one_group(self) -> None:
        """Коды, различные только старшими битами нескольких 64-битных слов (бит 56 слова в порядке packbits): у прежнего
        хеша без перемешивания разности 2⁶³ сокращались, группы не сходились — и отбор шёл сортировкой копии всех кодов."""
        rng = np.random.default_rng(12)
        length = 1024
        base = (rng.random((300, length)) < 0.1).astype(np.uint8)
        top = np.arange(56, length, 64)  # старший бит каждого слова
        base[:, top] = 0
        variants = [base]
        for pair in ((0, 1), (2, 5), (3, 4, 6, 7), (1, 2, 3, 4, 5, 6, 7, 8)):  # чётное число старших бит: 2⁶³·2k ≡ 0
            flipped = base.copy()
            flipped[:, top[list(pair)]] = 1
            variants.append(flipped)
        packed = np.packbits(np.vstack(variants + [base[:50]]), axis=1)  # плюс точные повторы
        order = rng.permutation(packed.shape[0])
        copy = np.ascontiguousarray(packed[order])
        _, first, counts = np.unique(copy.view(np.dtype((np.void, copy.shape[1]))).reshape(-1), return_index=True, return_counts=True)
        for attempt in range(LeaderDictionary.HASH_ATTEMPTS):
            found = LeaderDictionary._groups(packed, order, attempt)
            self.assertIsNotNone(found, f"попытка {attempt}")
            self.assertEqual(sorted(zip(found[0].tolist(), found[1].tolist())), sorted(zip(first.tolist(), counts.tolist())))

    def test_mix_is_a_bijection_on_samples(self) -> None:
        rng = np.random.default_rng(13)
        words = np.unique(np.concatenate([rng.integers(0, 2**63, 200000, dtype=np.uint64) * np.uint64(2),
                                          np.uint64(1) << np.arange(64, dtype=np.uint64), np.zeros(1, dtype=np.uint64)]))
        with np.errstate(over="ignore"):
            self.assertEqual(np.unique(LeaderDictionary._mix(words)).shape[0], words.shape[0])

    def test_int8_codes_on_the_device_give_the_same_dictionary(self) -> None:
        import torch

        if not torch.cuda.is_available() or not hasattr(torch, "_int_mm"):
            self.skipTest("нужны CUDA и torch._int_mm")
        packed, length = _codes(count=2500, length=203), 203  # длина кода не кратна 8, блоков больше одного
        order = np.random.default_rng(5).permutation(packed.shape[0])
        for threshold, size in ((0.9, 10**6), (0.7, 900)):
            expected = _reference(packed, length, order, threshold, size)
            with settings.overridden({"LAYOUT_INT8_CODES": True}):
                found = LeaderDictionary(similarity_threshold=threshold, max_size=size, use_gpu=True).select(packed, length, order=order)
            with settings.overridden({"LAYOUT_INT8_CODES": False}):
                plain = LeaderDictionary(similarity_threshold=threshold, max_size=size, use_gpu=True).select(packed, length, order=order)
            self._check(found, expected, f"int8, порог {threshold}, предел {size}")
            self._check(plain, expected, f"float32 на видеокарте, порог {threshold}, предел {size}")


if __name__ == "__main__":
    unittest.main()
