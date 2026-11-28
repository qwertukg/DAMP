"""Коды окон на диске (damp.rows_on_disk) и две правки пиков ОЗУ — те же числа до бита, что прежде:
чтение RowsOnDisk = массив в ОЗУ; срабатывания по кодам на диске = по кодам в ОЗУ; хеш ключа кеша без копии = прежний
хеш через tobytes(); проверка порядка кусками (set_embedding._descends) = np.any(np.diff(x) < 0); отклики картинок
кусками срабатываний (LevelRecognition.responses) = одним проходом."""
from __future__ import annotations

import gc
import hashlib
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "client"), os.path.join(ROOT, "system")]
os.environ.setdefault("RERUN", "off")
os.environ.setdefault("DAMP_RERUN_VIEWER", "off")
os.environ.setdefault("DAMP_CACHE", "off")

import numpy as np  # noqa: E402

from damp import cache as cache_module  # noqa: E402
from damp.activation import set_embedding  # noqa: E402
from damp.activation.hierarchy_detection import HierarchyDetection  # noqa: E402
from damp.rows_on_disk import RowsOnDisk  # noqa: E402


class _Folder(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.mkdtemp(prefix="damp-rows-test-")
        self._env = {name: os.environ.get(name) for name in ("DAMP_SPILL_DIR", "DAMP_SPILL_BYTES")}
        os.environ["DAMP_SPILL_DIR"] = self.folder

    def tearDown(self) -> None:
        for name, value in self._env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        gc.collect()
        shutil.rmtree(self.folder, ignore_errors=True)

    def files(self) -> list[str]:
        return [name for name in os.listdir(self.folder) if name.startswith(RowsOnDisk.PREFIX)]


class RowsOnDiskTests(_Folder):
    def test_reads_the_same_bytes(self) -> None:
        rng = np.random.default_rng(1)
        array = rng.integers(0, 256, size=(1000, 37), dtype=np.uint8)
        rows = RowsOnDisk(array)
        self.assertEqual((rows.shape, rows.dtype, rows.ndim, len(rows), rows.nbytes), (array.shape, array.dtype, 2, 1000, array.nbytes))
        self.assertTrue(rows.on_disk)
        keys = [slice(None), slice(10, 20), slice(990, 2000), slice(0, 0), 7, -1, np.arange(100, 400),
                np.array([5, 3, 3, 999, 0]), np.array([], dtype=np.int64), np.unique(rng.integers(0, 1000, 300))]
        for key in keys:
            found, expected = rows[key], array[key]
            self.assertIs(type(found), np.ndarray)
            self.assertEqual(found.dtype, expected.dtype)
            self.assertTrue(np.array_equal(found, expected))
            self.assertTrue(found.flags.c_contiguous)
        self.assertTrue(np.array_equal(np.asarray(rows), array))
        self.assertTrue(np.array_equal(np.ascontiguousarray(rows), array))
        self.assertEqual(np.unpackbits(rows[3]).shape, (37 * 8,))

    def test_other_dtypes_and_empty(self) -> None:
        for array in (np.arange(60, dtype=np.int64).reshape(20, 3), np.zeros((0, 5), dtype=np.uint8),
                      np.linspace(0, 1, 9, dtype=np.float32)):
            rows = RowsOnDisk(array)
            self.assertTrue(np.array_equal(rows[:], array))
            self.assertEqual(rows[:].dtype, array.dtype)

    def test_file_goes_away(self) -> None:
        rows = RowsOnDisk(np.ones((10, 4), dtype=np.uint8))
        path = rows.path
        self.assertTrue(os.path.exists(path))
        _ = rows[2:5]
        rows.release()
        self.assertFalse(os.path.exists(path))
        other = RowsOnDisk(np.ones((10, 4), dtype=np.uint8))
        path = other.path
        del other
        gc.collect()
        self.assertFalse(os.path.exists(path))
        self.assertEqual(self.files(), [])

    def test_threshold(self) -> None:
        small = np.ones((10, 4), dtype=np.uint8)
        os.environ.pop("DAMP_SPILL_BYTES", None)
        self.assertIs(RowsOnDisk.maybe(small), small)  # порог по умолчанию — 1 ГиБ
        os.environ["DAMP_SPILL_BYTES"] = "41"
        self.assertIs(RowsOnDisk.maybe(small), small)  # 40 байт < 41
        os.environ["DAMP_SPILL_BYTES"] = "40"
        moved = RowsOnDisk.maybe(small)
        self.assertIsInstance(moved, RowsOnDisk)
        self.assertIs(RowsOnDisk.maybe(moved), moved)
        self.assertEqual(os.path.dirname(moved.path), self.folder)


class FiringFromDiskTests(_Folder):
    @staticmethod
    def _detection() -> HierarchyDetection:
        from tests.test_article_invariants import _clustered_layout
        from tests.test_hierarchy_detection import HierarchyDetectionTest

        return HierarchyDetection(
            _clustered_layout(), HierarchyDetectionTest._detectors(), [1.0 if point % 5 else 0.05 for point in range(32)],
            similarity="cosine", lambda_threshold=0.5, eta=None, energy_threshold=0.12, detector_threshold=0.3, use_gpu=False,
        )

    def test_same_triples(self) -> None:
        detection = self._detection()
        detection.SET_CELLS = 64 * 32  # по 64 вхождения за проход: много проходов
        rng = np.random.default_rng(7)
        bits = (rng.random((500, 64)) < 0.15).astype(np.uint8)
        bits[::3, :8] = 1  # часть кодов похожа на точки карты: детекторы срабатывают
        packed = np.packbits(bits, axis=1)
        every = np.arange(packed.shape[0], dtype=np.int64)
        # Наборы по одному коду (как у прохода уровня) и наборы из нескольких кодов вразброс (номера не подряд).
        index = rng.integers(0, packed.shape[0], 900)
        groups = np.sort(rng.integers(0, 300, 900))
        for args in ((every, every, packed.shape[0]), (index, groups, 300)):
            in_memory = detection.active_sets(packed, *args)
            on_disk = detection.active_sets(RowsOnDisk(packed), *args)
            self.assertGreater(in_memory[0].size, 0)
            for mine, theirs in zip(on_disk, in_memory):
                self.assertEqual(mine.dtype, theirs.dtype)
                self.assertTrue(np.array_equal(mine, theirs))


class StepOnDiskTests(_Folder):
    def test_codes_move_and_watch_hears(self) -> None:
        from damp.cycle.step import CycleStep
        from damp.encoding.position_binding import KeyholeCodes

        packed = np.random.default_rng(3).integers(0, 256, size=(50, 6), dtype=np.uint8)
        codes = KeyholeCodes(packed=packed, code_length=45, place_length=13, content_length=32)
        heard = []
        os.environ.pop("DAMP_SPILL_BYTES", None)
        self.assertIs(CycleStep.on_disk(codes, lambda name, value: heard.append(name)), codes)
        self.assertEqual(heard, [])
        os.environ["DAMP_SPILL_BYTES"] = "0"
        moved = CycleStep.on_disk(codes, lambda name, value: heard.append((name, value)))
        self.assertIsInstance(moved.packed, RowsOnDisk)
        self.assertEqual((moved.count, moved.code_length, moved.place_length, moved.content_length), (50, 45, 13, 32))
        self.assertEqual(heard, [("codes_on_disk", moved)])
        self.assertTrue(np.array_equal(np.asarray(moved.packed), packed))
        self.assertTrue(np.array_equal(moved.bits(), codes.bits()))


class FingerprintTests(unittest.TestCase):
    @staticmethod
    def _old(value: np.ndarray) -> str:
        array = np.ascontiguousarray(value)
        parts = (b"array", str(array.dtype).encode(), str(array.shape).encode(), array.tobytes())
        hasher = hashlib.blake2b(digest_size=20)
        for part in parts:
            hasher.update(len(part).to_bytes(8, "little"))
            hasher.update(part)
        return hasher.hexdigest()

    def test_same_digest_as_tobytes(self) -> None:
        rng = np.random.default_rng(5)
        base = rng.integers(0, 1000, size=(40, 30))
        values = [
            base.astype(np.uint8), base, base.astype(np.int32), base.astype(np.float32), base.astype(np.float16),
            base > 500, base.T, base[::3, 1::2], np.zeros((0, 7), dtype=np.uint8), np.array(3.5), np.array([], dtype=np.int64),
            np.array(["2026-11-26", "2026-11-27"], dtype="datetime64[D]"),
            np.zeros(3, dtype=[("a", np.int32), ("b", np.float64)]), base.astype(np.complex64), np.array(["ab", "c"]),
        ]
        for value in values:
            self.assertEqual(cache_module.fingerprint(value), self._old(value), f"{value.dtype} {value.shape}")


class ResponsesTests(unittest.TestCase):
    @staticmethod
    def _old(picture, rows, columns, levels, pictures, detectors):  # noqa: ANN001, ANN205
        response = np.zeros((int(pictures), int(detectors)), dtype=np.float32)
        np.add.at(response, (picture[rows], columns), 1.0)
        response /= np.maximum(np.sqrt((response * response).sum(axis=1, keepdims=True)), 1e-12)
        peak = np.zeros((int(pictures), int(detectors)), dtype=np.float32)
        np.maximum.at(peak, (picture[rows], columns), levels.astype(np.float32))
        peak /= np.maximum(np.sqrt((peak * peak).sum(axis=1, keepdims=True)), 1e-12)
        return response, peak

    def test_same_as_one_pass(self) -> None:
        from damp.decode.level_recognition import LevelRecognition

        old_chunk = LevelRecognition.RESPONSE_CHUNK
        rng = np.random.default_rng(11)
        try:
            for windows, pictures, detectors, fired, kind in ((900, 40, 25, 5000, np.float32), (300, 12, 7, 0, np.float32),
                                                             (2000, 70, 60, 30000, np.float64)):
                picture = np.sort(rng.integers(0, pictures - 2, windows))  # две последние картинки — без окон
                rows = np.sort(rng.integers(0, windows, fired)).astype(np.int64)
                columns = rng.integers(0, detectors, fired).astype(np.int64)
                levels = (rng.random(fired) * 3).astype(kind)
                expected = self._old(picture, rows, columns, levels, pictures, detectors)
                for chunk in (1, 7, 1000, 1 << 24):
                    LevelRecognition.RESPONSE_CHUNK = chunk
                    found = LevelRecognition.responses(picture, rows, columns, levels, pictures, detectors)
                    for mine, theirs in zip(found, expected):
                        self.assertEqual(mine.dtype, theirs.dtype)
                        self.assertTrue(np.array_equal(mine, theirs), f"{chunk} {fired}")
        finally:
            LevelRecognition.RESPONSE_CHUNK = old_chunk


class DescendsTests(unittest.TestCase):
    def test_same_as_diff(self) -> None:
        old = set_embedding.ORDER_CHUNK
        rng = np.random.default_rng(9)
        try:
            for chunk in (1, 2, 7, 1 << 22):
                set_embedding.ORDER_CHUNK = chunk
                cases = [np.zeros(0, dtype=np.int64), np.array([4]), np.array([3, 3, 3]), np.array([1, 2, 2, 5]),
                         np.array([2, 1]), np.sort(rng.integers(0, 50, 100))]
                for position in (1, 6, 7, 8, 13, 14, 99):  # спад на границах кусков и внутри
                    broken = np.sort(rng.integers(0, 50, 100))
                    broken[position] = broken[position - 1] - 1
                    cases.append(broken)
                cases += [rng.integers(0, 5, 64) for _ in range(20)]
                for values in cases:
                    expected = bool(np.any(np.diff(values) < 0))
                    self.assertEqual(set_embedding._descends(values), expected, f"{chunk}: {values.tolist()}")
        finally:
            set_embedding.ORDER_CHUNK = old


if __name__ == "__main__":
    unittest.main()
