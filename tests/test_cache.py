"""Кеш шагов: из кеша — то же, что посчитано заново, бит в бит; шаги независимы; сверка ловит порчу; предел размера.

Цикл с кешем сверяется с циклом без кеша на малом наборе: все массивы прохода каждого уровня (коды окон, карта,
детекторы, срабатывания, детали, отклики) и точность совпадают.
"""
import dataclasses
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("RERUN", "off")

import numpy as np  # noqa: E402

from damp import settings  # noqa: E402
from damp.cache import CACHE, StepCache, fingerprint  # noqa: E402


@contextmanager
def cache_in(folder: str, **changes: object):
    """Кеш включён в папке folder (тесты по умолчанию работают без кеша)."""
    before = os.environ.get("DAMP_CACHE")
    os.environ["DAMP_CACHE"] = "on"
    try:
        with settings.overridden({"CACHE_ENABLED": True, "CACHE_DIR": folder, **changes}):
            yield
    finally:
        if before is None:
            os.environ.pop("DAMP_CACHE", None)
        else:
            os.environ["DAMP_CACHE"] = before


class StepCacheTest(unittest.TestCase):
    MODULES = ("damp.decode.level_recognition",)  # читает LADDER_NEIGHBOURS — им проверяется ключ по настройкам

    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.calls = 0

    def tearDown(self) -> None:
        self.folder.cleanup()

    def compute(self) -> dict[str, object]:
        self.calls += 1
        return {"codes": np.arange(12, dtype=np.uint8).reshape(3, 4), "length": 32, "share": 0.1 + 0.2, "box": [-1, 1]}

    def run_step(self, values: np.ndarray, cache: StepCache = CACHE) -> dict[str, object]:
        return cache.run("test.step", {"values": values, "number": 1}, self.compute, modules=self.MODULES)

    def test_hit_returns_same_bits(self) -> None:
        with cache_in(self.folder.name):
            first = self.run_step(np.ones(5))
            second = self.run_step(np.ones(5))
        self.assertEqual(self.calls, 1)
        self.assertTrue(StepCache.same(first, second))
        self.assertEqual(second["codes"].dtype, np.uint8)
        self.assertEqual(second["share"], 0.1 + 0.2)  # число с плавающей точкой — бит в бит через JSON

    def test_key_follows_input_settings_and_disable(self) -> None:
        with cache_in(self.folder.name):
            self.run_step(np.ones(5))
            self.run_step(np.zeros(5))  # другой вход
            with settings.overridden({"LADDER_NEIGHBOURS": (1, 3)}):
                self.run_step(np.ones(5))  # другая настройка, которую читает код шага
            with settings.overridden({"LOG_INTERVAL_INIT": 7}):
                self.run_step(np.ones(5))  # настройка журнала результат не меняет — попадание
        self.assertEqual(self.calls, 3)
        self.run_step(np.ones(5))  # кеш выключен (так в тестах по умолчанию) — счёт
        self.assertEqual(self.calls, 4)

    def test_verification_catches_spoiled_entry(self) -> None:
        with cache_in(self.folder.name):
            self.run_step(np.ones(5))
            entry = next(Path(self.folder.name, "steps", "test.step").glob("*.json"))
            data = json.loads(entry.read_text(encoding="utf-8"))
            data["outputs"]["length"]["value"] = 33  # порча записи
            entry.write_text(json.dumps(data), encoding="utf-8")
            with settings.overridden({"CACHE_VERIFY_SHARE": 1.0}):
                fresh = self.run_step(np.ones(5))
            again = self.run_step(np.ones(5))
        self.assertEqual(fresh["length"], 32)
        self.assertEqual(again["length"], 32)  # запись заменена свежим результатом
        self.assertEqual(self.calls, 2)

    def test_size_limit_evicts_old_entries(self) -> None:
        with cache_in(self.folder.name, CACHE_MAX_BYTES=1):
            cache = StepCache()
            for value in range(3):
                self.run_step(np.full(5, value), cache)
            self.assertLessEqual(len(cache.entries()), 1)
            # Массивы без записей удалены: остались только те, на которые ссылается уцелевшая запись.
            self.assertLessEqual(len(list(Path(self.folder.name, "blobs").glob("*/*.npy"))), len(cache.entries()))

    def test_fingerprint_sees_dtype_and_shape(self) -> None:
        self.assertNotEqual(fingerprint(np.zeros(4, dtype=np.int32)), fingerprint(np.zeros(4, dtype=np.int64)))
        self.assertNotEqual(fingerprint(np.zeros((2, 2))), fingerprint(np.zeros(4)))
        self.assertEqual(fingerprint((1, 2)), fingerprint((1, 2)))


class CycleCacheTest(unittest.TestCase):
    """Цикл с кешем = цикл без кеша, цифра в цифру; детекторы по карте из кеша = детекторы по исходной карте."""

    TRAIN, TEST = 30, 20

    @staticmethod
    def run_cycle(folder: str):  # noqa: ANN205
        from damp.cycle.ladder import MnistLadderRunner
        from damp.cycle.ladder_trace import LadderObserver

        class Traces(LadderObserver):
            def __init__(self) -> None:
                self.levels = []

            def level(self, trace) -> None:  # noqa: ANN001
                self.levels.append(trace)

        traces = Traces()
        runner = MnistLadderRunner(
            count=CycleCacheTest.TRAIN, evaluate_limit=CycleCacheTest.TEST, map_max_points=None, shifts=(),
            file_prefix=str(Path(folder) / "ladder"), observer=traces,
        )
        runner.run()
        return runner, traces.levels

    def assert_same_cycle(self, expected, actual) -> None:  # noqa: ANN001
        (runner_a, levels_a), (runner_b, levels_b) = expected, actual
        self.assertEqual(runner_a.top_accuracy, runner_b.top_accuracy)
        self.assertEqual([dataclasses.asdict(r) for r in runner_a.reports], [dataclasses.asdict(r) for r in runner_b.reports])
        self.assertEqual(len(levels_a), len(levels_b))
        for left, right in zip(levels_a, levels_b):
            for field in dataclasses.fields(left):
                a, b = getattr(left, field.name), getattr(right, field.name)
                if isinstance(a, np.ndarray):
                    with self.subTest(level=left.number, field=field.name):
                        self.assertEqual(a.dtype, b.dtype)
                        self.assertTrue(np.array_equal(a, b), field.name)
                elif isinstance(a, (int, float, str, tuple)):
                    with self.subTest(level=left.number, field=field.name):
                        self.assertEqual(a, b)

    def test_cached_cycle_repeats_direct_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as folder, settings.overridden({"LADDER_SAVE_MAPS": True}):
            direct = self.run_cycle(folder)  # без кеша
            cache = str(Path(folder) / "cache")
            with cache_in(cache):
                filled = self.run_cycle(folder)  # кеш наполняется
                entries = len(CACHE.entries())
                repeated = self.run_cycle(folder)  # всё из кеша
                self.assertEqual(len(CACHE.entries()), entries)  # повтор не добавил записей — все шаги попали
                # Детекторы по карте, восстановленной из кеша: стираются только они — карта берётся из кеша.
                CACHE.clear("activation.detectors")
                restored = self.run_cycle(folder)
                self.assertEqual(len(CACHE.entries()), entries)  # те же детекторы → срабатывания и далее попали
        self.assert_same_cycle(direct, filled)
        self.assert_same_cycle(direct, repeated)
        self.assert_same_cycle(direct, restored)


if __name__ == "__main__":
    unittest.main()
