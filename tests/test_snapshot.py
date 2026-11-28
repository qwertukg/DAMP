"""Снимок обученного цикла (system/damp/snapshot.py, TrainedLadder.save / load): без обучения — тот же цикл до бита.

GraphSnapshot — граф объектов без pickle: значения, общие ссылки и циклы, массивы, генераторы случайных чисел; чужие
объекты не пишутся. TrainedLadder: снимок малого цикла, загруженный заново, ведёт картинку по уровням так же, как
исходный, цифра в цифру; сверка ответов при загрузке видит совпадение.
"""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np


class Foreign:
    """Объект не из пакета damp: в снимок не идёт."""

    def __init__(self) -> None:
        self.value = 1


class GraphSnapshotTest(unittest.TestCase):
    """Запись и чтение графа: значения, общие ссылки, циклы, массивы, генераторы, отказ чужим объектам."""

    def _round_trip(self, root: object, **options) -> object:  # noqa: ANN003
        from damp.snapshot import GraphSnapshot

        with tempfile.TemporaryDirectory() as folder:
            GraphSnapshot(**options).save(root, folder)
            return GraphSnapshot(**options).load(folder)

    def test_values_shared_references_and_cycles(self) -> None:
        from damp.cycle.trained import SightLevel
        from damp.encoding.bitarray import BitArray

        array = np.arange(12, dtype=np.float32).reshape(3, 4)
        numbers = list(range(200))  # длинный числовой список уходит в npz
        loop: list = ["петля"]
        loop.append(loop)
        bits = BitArray.from_indices(40, [1, 5, 39])
        level = SightLevel(
            number=2, keyhole_centre=np.ones((2, 2)), keyhole_extent=np.ones(2), entry_keyhole=np.arange(3),
            entry_element=np.arange(3), lit_keyhole=np.arange(2), lit_point=np.arange(2), lit_value=np.ones(2, np.float32),
            fired_keyhole=np.arange(1), fired_detector=np.arange(1), fired_level=np.ones(1, np.float32),
            detail_codes=np.zeros((2, 4), np.uint8), detail_length=32, strength=np.ones(2), peak=np.zeros(5, np.float32),
        )
        root = {
            "array": array, "same": array, "numbers": numbers, "twice": [numbers, numbers], "loop": loop,
            (1, "ключ"): 2.5, "set": set(range(100)), "small_set": {3, 1, 2}, "bytes": bytearray(b"\x01\x02" * 3000),
            "scalar": np.float32(1.25), "nan": float("nan"), "big": 2**70, "none": None, "flag": True,
            "bits": bits, "level": level, "level_again": level,
        }
        found = self._round_trip(root)
        self.assertTrue(np.array_equal(found["array"], array))
        self.assertEqual(found["array"].dtype, np.float32)
        self.assertIs(found["array"], found["same"])
        self.assertEqual(found["numbers"], numbers)
        self.assertIs(found["twice"][0], found["twice"][1])
        self.assertIs(found["loop"][1], found["loop"])
        self.assertEqual(found[(1, "ключ")], 2.5)
        self.assertEqual(found["set"], set(range(100)))
        self.assertEqual(found["small_set"], {1, 2, 3})
        self.assertEqual(found["bytes"], root["bytes"])
        self.assertIsInstance(found["bytes"], bytearray)
        self.assertEqual(found["scalar"].dtype, np.float32)
        self.assertEqual(float(found["scalar"]), 1.25)
        self.assertTrue(np.isnan(found["nan"]))
        self.assertEqual(found["big"], 2**70)
        self.assertIsNone(found["none"])
        self.assertIs(found["flag"], True)
        self.assertEqual(found["bits"].to_indices() if hasattr(found["bits"], "to_indices") else bytes(found["bits"]._bits),
                         bits.to_indices() if hasattr(bits, "to_indices") else bytes(bits._bits))
        self.assertIs(found["level"], found["level_again"])
        self.assertEqual(found["level"].detail_length, 32)
        self.assertTrue(np.array_equal(found["level"].entry_keyhole, level.entry_keyhole))

    def test_random_generators_continue_the_same(self) -> None:
        python, numpy_generator, legacy = random.Random(5), np.random.default_rng(7), np.random.RandomState(9)
        python.random(), numpy_generator.random(3), legacy.rand(2)
        found = self._round_trip({"python": python, "numpy": numpy_generator, "legacy": legacy})
        self.assertEqual(found["python"].random(), python.random())
        self.assertTrue(np.array_equal(found["numpy"].random(5), numpy_generator.random(5)))
        self.assertTrue(np.array_equal(found["legacy"].rand(5), legacy.rand(5)))

    def test_tensor_keeps_values_and_type(self) -> None:
        import torch

        tensor = torch.arange(10, dtype=torch.int64).reshape(2, 5)
        found = self._round_trip({"tensor": tensor, "device": torch.device("cpu")})
        self.assertTrue(torch.equal(found["tensor"], tensor))
        self.assertEqual(found["tensor"].dtype, torch.int64)
        self.assertEqual(str(found["device"]), "cpu")

    def test_foreign_object_and_code_are_refused(self) -> None:
        from damp.snapshot import SnapshotError

        with self.assertRaises(SnapshotError) as caught:
            self._round_trip({"inner": [Foreign()]})
        self.assertIn("root['inner'][0]", str(caught.exception))
        with self.assertRaises(SnapshotError):
            self._round_trip({"code": len})

    def test_transient_attributes_are_empty_and_rebuilt(self) -> None:
        from damp.encoding.bitarray import BitArray

        bits = BitArray.from_indices(16, [2, 3])
        found = self._round_trip(
            bits, transient={"BitArray": {"_active"}}, rebuild={"BitArray": lambda value: setattr(value, "rebuilt", True)}
        )
        self.assertIsNone(found._active)
        self.assertTrue(found.rebuilt)
        self.assertEqual(bytes(found._bits), bytes(bits._bits))


class LadderSnapshotTest(unittest.TestCase):
    """Снимок малого обученного цикла: загруженный цикл ведёт картинки так же, как исходный, и сверка это видит."""

    TRAIN = 30

    @classmethod
    def setUpClass(cls) -> None:
        from torchvision.datasets import MNIST

        from damp import settings
        from damp.cycle.ladder import MnistLadderRunner
        from damp.cycle.trained import TrainedLadder

        cls.changes = {"LADDER_SAVE_MAPS": False, "LADDER_TRAIN_VIEWS": ()}
        cls.folder = tempfile.TemporaryDirectory()
        with settings.overridden(cls.changes):
            runner = MnistLadderRunner(
                count=cls.TRAIN, evaluate_limit=0, map_max_points=None, shifts=(),
                file_prefix=str(Path(cls.folder.name) / "ladder"), keep_levels=True, keep_top_firing=False,
            )
            runner.run()
            cls.ladder = TrainedLadder(runner.groups, runner.levels, runner.level_peaks, decay=1e-7, primary=runner.primary)
            cls.images = MNIST(root="./data", train=False, download=True).data[200:206].numpy()
            cls.before = [cls.ladder.see(image) for image in cls.images]
            values = {name: getattr(settings, name) for name in settings.KEYS}
            cls.path = Path(cls.folder.name) / "snapshot"
            cls.meta = cls.ladder.save(cls.path, values=values, label="тест", control=8)
            cls.after = [cls.ladder.see(image) for image in cls.images]  # тот же цикл после записи (видеокарта заново)
            cls.loaded, cls.loaded_meta = TrainedLadder.load(cls.path)
            cls.same = cls.loaded.check(cls.loaded_meta["control"])
            cls.found = [cls.loaded.see(image) for image in cls.images]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.folder.cleanup()

    def _assert_same(self, first, second) -> None:  # noqa: ANN001
        self.assertEqual(first.answer, second.answer)
        self.assertTrue(np.array_equal(first.probabilities, second.probabilities))
        self.assertEqual((first.verdict.kind, first.verdict.digit), (second.verdict.kind, second.verdict.digit))
        self.assertEqual(len(first.levels), len(second.levels))
        for a, b in zip(first.levels, second.levels):
            for name in ("keyhole_centre", "entry_keyhole", "lit_point", "lit_value", "fired_detector", "fired_level",
                         "detail_codes", "strength", "peak"):
                self.assertTrue(np.array_equal(getattr(a, name), getattr(b, name)), f"уровень {a.number}: {name}")

    def test_loaded_ladder_sees_the_same(self) -> None:
        for before, found in zip(self.before, self.found):
            self._assert_same(before, found)

    def test_saving_does_not_change_the_ladder(self) -> None:
        for before, after in zip(self.before, self.after):
            self._assert_same(before, after)

    def test_control_matches_on_load(self) -> None:
        self.assertEqual(self.meta["control"]["pictures"], 8)
        self.assertEqual(self.same, 8)

    def test_meta_keeps_settings_without_local_ones(self) -> None:
        from damp.cycle.trained import TrainedLadder

        self.assertEqual(self.loaded_meta["label"], "тест")
        self.assertIn("ACTIVATION_SATURATION_LIMIT", self.loaded_meta["settings"])
        changes = TrainedLadder.snapshot_changes(self.loaded_meta)
        self.assertIn("LADDER_CLUSTER_LEVELS", changes)
        self.assertFalse([name for name in changes if name.startswith(("DASHBOARD_", "LOG_", "CACHE_"))])

    def test_not_a_snapshot_folder_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "чужой.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                self.ladder.save(folder, values={}, control=0)
            with self.assertRaises(FileNotFoundError):
                type(self.ladder).load(folder)


if __name__ == "__main__":
    unittest.main()
