"""Быстрые пути сборки деталей (system/damp/activation/set_embedding.py) дают тот же ответ, что прямые (вне статьи):
отбор σ сильнейших детекторов — скомпилированный цикл против плотной матрицы, плотность кодов — битовая карта против
сортировки пар."""

from __future__ import annotations

import os
import sys
import unittest


os.environ.setdefault("RERUN", "off")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "system"))

import numpy as np  # noqa: E402

from damp.activation import set_embedding  # noqa: E402
from damp.activation.set_embedding import DetectorEmbeddings  # noqa: E402


def _coder(detectors: int, saturation: int, *, density: float = 0.25) -> DetectorEmbeddings:
    return DetectorEmbeddings(detectors=detectors, saturation_limit=saturation, target_density=density, min_length=16,
                              max_length=2048, seed=7, name="fast-test")


def _firings(seed: int, sets: int, detectors: int, *, most: int, ties: bool = True) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Срабатывания: у набора от 0 до most детекторов по возрастанию номера; уровни — с повторами, если ties."""
    rng = np.random.default_rng(seed)
    rows, columns = [], []
    for index in range(sets):
        fired = np.sort(rng.choice(detectors, int(rng.integers(0, most + 1)), replace=False))
        rows.append(np.full(fired.shape[0], index, dtype=np.int64))
        columns.append(fired.astype(np.int64))
    rows, columns = np.concatenate(rows), np.concatenate(columns)
    levels = rng.random(rows.shape[0]).astype(np.float32)
    if ties:
        levels = np.round(levels, 1)  # много равных уровней: проверяется порядок при равной силе
    return rows, columns, levels


def _direct(coder: DetectorEmbeddings, rows: "np.ndarray", columns: "np.ndarray", levels: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
    """Прямой путь отбора: как strongest, но всегда по плотной матрице."""
    rows, columns, levels = np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64), np.asarray(levels)
    if np.any(np.diff(rows) < 0):
        order = np.argsort(rows, kind="stable")
        rows, columns, levels = rows[order], columns[order], levels[order]
    return coder._strongest_dense(rows, columns, levels)


class StrongestFastPathTest(unittest.TestCase):
    def setUp(self) -> None:
        if set_embedding._strongest_compiled is None:
            self.skipTest("numba не установлена: быстрого пути отбора нет, работает прямой")

    def test_same_pairs_as_the_dense_matrix(self) -> None:
        """σ сильнейших детекторов набора: скомпилированный цикл и плотная матрица дают те же пары в том же порядке."""
        for seed, (sets, detectors, most, saturation) in enumerate([(400, 60, 40, 5), (300, 200, 120, 50), (200, 30, 30, 1), (150, 500, 20, 50)]):
            with self.subTest(sets=sets, detectors=detectors, saturation=saturation):
                rows, columns, levels = _firings(seed, sets, detectors, most=most)
                coder = _coder(detectors, saturation)
                fast_rows, fast_columns = coder.strongest(rows, columns, levels)
                slow_rows, slow_columns = _direct(coder, rows, columns, levels)
                self.assertTrue(np.array_equal(fast_rows, slow_rows))
                self.assertTrue(np.array_equal(fast_columns, slow_columns))
                self.assertLessEqual(int(np.bincount(fast_rows, minlength=sets).max()), saturation)

    def test_equal_levels_keep_the_lower_detector(self) -> None:
        """При равной силе в код идёт детектор с меньшим номером — как у плотной матрицы."""
        rows = np.zeros(6, dtype=np.int64)
        columns = np.array([1, 3, 4, 7, 8, 9], dtype=np.int64)
        levels = np.array([0.5, 0.9, 0.5, 0.5, 0.9, 0.5], dtype=np.float32)
        kept_rows, kept_columns = _coder(10, 4).strongest(rows, columns, levels)
        self.assertEqual(kept_columns.tolist(), [1, 3, 4, 8])  # оба 0.9 и два первых из четырёх 0.5
        self.assertTrue(np.array_equal(kept_columns, _direct(_coder(10, 4), rows, columns, levels)[1]))

    def test_saturation_not_below_the_detectors_keeps_everything(self) -> None:
        rows, columns, levels = _firings(5, 100, 40, most=40)
        coder = _coder(40, 40)
        fast, slow = coder.strongest(rows, columns, levels), _direct(coder, rows, columns, levels)
        self.assertTrue(np.array_equal(fast[0], rows) and np.array_equal(fast[1], columns))
        self.assertTrue(np.array_equal(fast[0], slow[0]) and np.array_equal(fast[1], slow[1]))

    def test_pairs_out_of_order_go_the_direct_way(self) -> None:
        """Пары не по порядку (наборы вперемешку, детекторы набора не по возрастанию) — ответ всё равно тот же."""
        rows, columns, levels = _firings(9, 200, 80, most=60)
        mixed = np.random.default_rng(1).permutation(rows.shape[0])
        coder = _coder(80, 7)
        fast = coder.strongest(rows[mixed], columns[mixed], levels[mixed])
        slow = _direct(coder, rows, columns, levels)
        self.assertTrue(np.array_equal(fast[0], slow[0]) and np.array_equal(fast[1], slow[1]))

    def test_levels_of_either_precision_give_the_same_pairs(self) -> None:
        rows, columns, levels = _firings(3, 200, 90, most=70)
        coder = _coder(90, 9)
        single, double = coder.strongest(rows, columns, levels), coder.strongest(rows, columns, levels.astype(np.float64))
        self.assertTrue(np.array_equal(single[0], double[0]) and np.array_equal(single[1], double[1]))


class DensityFastPathTest(unittest.TestCase):
    def test_bitmap_counts_the_same_bits_as_sorting(self) -> None:
        """Плотность кодов по битовой карте — то же число, что прямой счёт сортировкой пар (набор, бит)."""
        rows, columns, levels = _firings(11, 600, 300, most=120, ties=False)
        coder = _coder(300, 50)
        kept_rows, kept_columns = coder.strongest(rows, columns, levels)
        for length in (16, 64, 128, 256, 1024, 2048):
            with self.subTest(length=length):
                self.assertEqual(coder._density(kept_rows, kept_columns, 600, length), coder._density_sorted(kept_rows, kept_columns, 600, length))

    def test_bitmap_in_small_chunks_is_the_same(self) -> None:
        """Карта считается кусками: размер куска на ответ не влияет; пустые наборы и наборы вне пар учтены в делителе."""
        rows, columns, levels = _firings(12, 500, 100, most=30)
        coder = _coder(100, 10)
        kept_rows, kept_columns = coder.strongest(rows, columns, levels)
        whole = coder._density(kept_rows, kept_columns, 700, 64)
        coder.DENSITY_CELLS = 64 * 3  # по три набора на кусок
        self.assertEqual(coder._density(kept_rows, kept_columns, 700, 64), whole)
        self.assertEqual(whole, coder._density_sorted(kept_rows, kept_columns, 700, 64))

    def test_pairs_out_of_order_and_empty_input(self) -> None:
        rows, columns, levels = _firings(13, 300, 100, most=40)
        coder = _coder(100, 0)  # без отбора: пары идут как есть
        mixed = np.random.default_rng(2).permutation(rows.shape[0])
        self.assertEqual(coder._density(rows[mixed], columns[mixed], 300, 128), coder._density_sorted(rows, columns, 300, 128))
        self.assertEqual(coder._density(rows[:0], columns[:0], 10, 64), 0.0)

    def test_code_length_and_codes_are_those_of_the_direct_way(self) -> None:
        """Вся сборка: длина кода и сами коды — те же, что с прямыми путями отбора и счёта плотности."""
        rows, columns, levels = _firings(14, 800, 250, most=150)
        mask = np.random.default_rng(3).random(800) < 0.7
        fast_codes, fast_length = _coder(250, 50).encode(rows, columns, levels, 800, calibration=mask)

        class Direct(DetectorEmbeddings):
            def strongest(self, rows, columns, levels):  # noqa: ANN001, ANN201
                return _direct(self, rows, columns, levels)

            _density = DetectorEmbeddings._density_sorted

        direct = Direct(detectors=250, saturation_limit=50, target_density=0.25, min_length=16, max_length=2048, seed=7, name="direct")
        slow_codes, slow_length = direct.encode(rows, columns, levels, 800, calibration=mask)
        self.assertEqual(fast_length, slow_length)
        self.assertTrue(np.array_equal(fast_codes, slow_codes))


if __name__ == "__main__":
    unittest.main()
