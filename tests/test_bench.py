"""Стенд честного сравнения: протокол, выбор настроек только по validation, интервалы, итоговый JSON (вне статьи).

Тесты считают на крошечных числах (20 обучающих, 2 набора, одна конфигурация CNN, несколько эпох). Запуск — из корня
репозитория: данные MNIST берутся из ./data.
"""
import dataclasses
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("RERUN", "off")

from damp import settings  # noqa: E402
from damp.decode.bench import BenchSample, BenchSet, HonestBench, Score, Summary  # noqa: E402
from damp.figures.rerun_bench import BenchBlueprint, BenchTable  # noqa: E402

SMALL = {
    "BENCH_CNN_ARCHS": ("lenet5",),
    "BENCH_CNN_LR": (0.001,),
    "BENCH_CNN_WEIGHT_DECAY": (0.0,),
    "BENCH_CNN_DROPOUT": (0.0,),
    "BENCH_CNN_EPOCHS": 3,
    "BENCH_CNN_PATIENCE": 2,
    "BENCH_NEIGHBOURS": (1, 3),
    "LADDER_SAVE_MAPS": False,
    # Виды обучающих (§7.1.4) проверяют тесты протокола и кубиков; стенду они здесь не нужны, а цикл с ними впятеро дольше.
    "LADDER_TRAIN_VIEWS": (),
}


class SummaryTest(unittest.TestCase):
    def test_mean_std_and_t_interval(self) -> None:
        from scipy import stats

        values = [0.80, 0.84, 0.90, 0.86]
        summary = Summary.of(values)
        mean, std = float(np.mean(values)), float(np.std(values, ddof=1))
        half = float(stats.t.ppf(0.975, 3)) * std / 2.0
        self.assertAlmostEqual(summary["mean"], mean)
        self.assertAlmostEqual(summary["std"], std)
        self.assertAlmostEqual(summary["ci_low"], mean - half)
        self.assertAlmostEqual(summary["ci_high"], mean + half)
        self.assertEqual(summary["per_set"], values)

    def test_single_set_has_no_interval(self) -> None:
        summary = Summary.of([0.9])
        self.assertEqual((summary["mean"], summary["std"], summary["ci_low"], summary["n"]), (0.9, 0.0, None, 1))
        self.assertIsNone(Summary.of([]))

    def test_paired_difference_is_interval_of_per_set_differences(self) -> None:
        paired = Summary.paired([0.9, 0.8, 0.7], [0.8, 0.8, 0.5])
        expected = Summary.of([0.1, 0.0, 0.2])
        self.assertAlmostEqual(paired["mean"], expected["mean"])
        self.assertAlmostEqual(paired["ci_low"], expected["ci_low"])
        self.assertIsNone(Summary.paired([0.9], [0.8, 0.7]))


class SettingsFileAugmentTest(unittest.TestCase):
    """Режимы аугментации из файла настроек: строка "none" там читается как None — стенд понимает её как режим."""

    def test_none_from_settings_file_is_plain_mode(self) -> None:
        with settings.overridden({"BENCH_AUGMENT": ["none", "shifts"]}):
            self.assertIsNone(settings.BENCH_AUGMENT[0])  # соглашение файла настроек: "none" → None
            bench = HonestBench(train=20, sets=2, validation=20, methods=("pixels",))
        self.assertEqual(bench.augment, ("none", "shifts"))


class SelectionTest(unittest.TestCase):
    """Настройка выбирается только по средней validation: test на выбор не влияет."""

    def _bench(self, tests: tuple[list[float], list[float]]) -> HonestBench:
        bench = HonestBench(train=20, sets=2, validation=20, methods=("pixels",), augment=("none",))
        bench._scores[("pixels", "none", "k=1")] = Score(select=[0.90, 0.80], validation=[0.1, 0.1], test=list(tests[0]))
        bench._scores[("pixels", "none", "k=3")] = Score(select=[0.70, 0.60], validation=[0.9, 0.9], test=list(tests[1]))
        bench._sets_done = 2
        return bench

    def test_choice_follows_validation_not_test(self) -> None:
        for tests in (([0.1, 0.1], [0.9, 0.9]), ([0.9, 0.9], [0.1, 0.1]), ([0.0, 0.0], [1.0, 1.0])):
            with self.subTest(tests=tests):
                row = self._bench(tests)._row("pixels", "none")
                self.assertEqual(row["setting"], "k=1")  # validation 0.85 против 0.65, что бы ни показывал test
                self.assertAlmostEqual(row["test"]["mean"], float(np.mean(tests[0])))  # test выбранной, а не лучший test
                self.assertAlmostEqual(row["candidates"]["k=3"], 0.65)
                self.assertAlmostEqual(row["validation"]["mean"], 0.1)  # итог - по report, а не по select

    def test_report_part_does_not_choose(self) -> None:
        """Подмена точности на report не меняет выбор: он идёт только по select."""
        for report in ([0.0, 0.0], [1.0, 1.0]):
            bench = HonestBench(train=20, sets=2, validation=20, methods=("pixels",), augment=("none",))
            bench._scores[("pixels", "none", "k=1")] = Score(select=[0.9, 0.9], validation=list(report), test=[None, None])
            bench._scores[("pixels", "none", "k=3")] = Score(select=[0.5, 0.5], validation=[1.0 - r for r in report], test=[None, None])
            self.assertEqual(bench._row("pixels", "none")["setting"], "k=1")

    def test_validation_is_split_into_select_and_report(self) -> None:
        bench = HonestBench(train=20, sets=2, validation=20, methods=("pixels",), augment=("none",))
        self.assertEqual((bench.select_count, bench.report_count), (10, 10))
        self.assertEqual(bench._parts((100, 120)), ((100, 110), (110, 120)))
        with self.assertRaises(ValueError):
            HonestBench(validation=1)

    def test_gap_has_paired_interval(self) -> None:
        rows = [
            {"method": "damp", "augment": "none", "setting": "k=1", "candidates": {"k=1": 0.8},
             "validation": Summary.of([0.8, 0.82, 0.78]), "test": None},
            {"method": "cnn", "augment": "none", "setting": "lenet5", "candidates": {"lenet5": 0.9},
             "validation": Summary.of([0.9, 0.91, 0.92]), "test": None},
        ]
        gap = HonestBench._gap(rows)
        self.assertAlmostEqual(gap["damp_minus_best_cnn"]["validation"], 0.8 - 0.91, places=6)
        span = gap["intervals"]["damp_minus_best_cnn"]["validation"]
        self.assertEqual(span["n"], 3)
        self.assertIsNotNone(span["ci_low"])

    def test_best_cnn_is_chosen_by_select_not_by_report(self) -> None:
        """Режим аугментации лучшей CNN выбирается по select; report (validation) в выборе не участвует."""
        rows = [
            {"method": "damp", "augment": "none", "setting": "k=1", "candidates": {"k=1": 0.8},
             "validation": Summary.of([0.6, 0.6]), "test": None},
            {"method": "cnn", "augment": "none", "setting": "a", "candidates": {"a": 0.95},  # лучше по select
             "validation": Summary.of([0.7, 0.7]), "test": None},
            {"method": "cnn", "augment": "shifts", "setting": "b", "candidates": {"b": 0.90},  # лучше по report
             "validation": Summary.of([0.9, 0.9]), "test": None},
        ]
        gap = HonestBench._gap(rows)
        self.assertEqual(gap["best_cnn"], "none: a")
        self.assertAlmostEqual(gap["damp_minus_best_cnn"]["validation"], 0.6 - 0.7, places=6)

    def test_augment_without_none_skips_damp_and_pixels(self) -> None:
        with settings.overridden({**SMALL}):
            bench = HonestBench(train=20, sets=1, validation=20, methods=("damp", "pixels", "cnn"), augment=("shifts",), save=False)
            bench.run()
        self.assertEqual(bench._seconds["damp"], 0.0)
        self.assertEqual(bench._seconds["pixels"], 0.0)
        self.assertFalse(any(key[0] in ("damp", "pixels") for key in bench._scores))

    def test_float_noise_does_not_break_a_tie(self) -> None:
        """Равные по смыслу средние складываются в float по-разному; ничья всё равно у первой настройки."""
        bench = HonestBench(train=20, sets=3, validation=20, methods=("pixels",), augment=("none",))
        bench._scores[("pixels", "none", "k=1")] = Score(select=[0.3, 0.2, 0.1], validation=[0.5, 0.6, 0.7], test=[None] * 3)
        bench._scores[("pixels", "none", "k=3")] = Score(select=[0.1, 0.2, 0.3], validation=[0.9, 0.9, 0.9], test=[None] * 3)
        self.assertLess(np.mean([0.3, 0.2, 0.1]), np.mean([0.1, 0.2, 0.3]))  # шум float есть
        row = bench._row("pixels", "none")
        self.assertEqual(row["setting"], "k=1")
        # select и report всех кандидатов — для разбора; test в них нет.
        self.assertEqual(row["scores"]["k=3"], {"select": [0.1, 0.2, 0.3], "report": [0.9, 0.9, 0.9]})
        self.assertEqual(set(row["scores"]), {"k=1", "k=3"})

    def test_tie_goes_to_first_setting(self) -> None:
        bench = HonestBench(train=20, sets=2, validation=20, methods=("pixels",), augment=("none",))
        bench._scores[("pixels", "none", "k=1")] = Score(select=[0.8, 0.8], validation=[0.8, 0.8], test=[0.0, 0.0])
        bench._scores[("pixels", "none", "k=3")] = Score(select=[0.8, 0.8], validation=[0.8, 0.8], test=[1.0, 1.0])
        self.assertEqual(bench._row("pixels", "none")["setting"], "k=1")

    def test_unsupported_augment_is_a_dash(self) -> None:
        bench = HonestBench(train=20, sets=2, validation=20, methods=("damp", "pixels", "cnn"), augment=("none", "shifts"))
        for method in ("damp", "pixels"):
            row = bench._row(method, "shifts")
            self.assertFalse(row["supported"])
            self.assertIsNone(row["validation"])
        self.assertIn("LADDER_TRAIN_VIEWS", bench._row("damp", "shifts")["note"])  # сдвиги DAMP — виды обучающих (§7.1.4)
        self.assertTrue(bench._row("cnn", "shifts")["supported"])

    def test_bad_arguments(self) -> None:
        with self.assertRaises(ValueError):
            HonestBench(methods=("magic",))
        with self.assertRaises(ValueError):
            HonestBench(augment=("rotate",))
        with self.assertRaises(ValueError):
            HonestBench(validation=0)


class TestLabelsDoNotChooseTest(unittest.TestCase):
    """На одних и тех же наборах подмена меток test меняет только число test, а не выбранную настройку."""

    def test_swapped_test_labels_keep_choice(self) -> None:
        with settings.overridden({**SMALL, "BENCH_NEIGHBOURS": (1, 3, 5)}):
            outcomes = []
            for swap in (False, True):
                bench = HonestBench(train=20, sets=2, validation=20, test=20, methods=("pixels", "cnn"), augment=("none",))
                bench._load_common()
                for number in range(2):
                    data = bench._load_set(number)
                    if swap:
                        labels = (data.test.labels + 3) % 10
                        data = dataclasses.replace(data, test=dataclasses.replace(data.test, labels=labels))
                    bench._pixels(data)
                    bench._cnn(data)
                bench._sets_done = 2
                outcomes.append({(m, a): bench._row(m, a) for m in ("pixels", "cnn") for a in ("none",)})
        for key in outcomes[0]:
            with self.subTest(row=key):
                self.assertEqual(outcomes[0][key]["setting"], outcomes[1][key]["setting"])
                self.assertEqual(outcomes[0][key]["candidates"], outcomes[1][key]["candidates"])
                self.assertEqual(outcomes[0][key]["validation"]["per_set"], outcomes[1][key]["validation"]["per_set"])
                self.assertNotEqual(outcomes[0][key]["test"]["per_set"], outcomes[1][key]["test"]["per_set"])


class ProtocolTest(unittest.TestCase):
    def test_sets_are_disjoint_from_each_other_and_validation(self) -> None:
        with settings.overridden({**SMALL}):
            bench = HonestBench(train=20, sets=3, validation=20, methods=("pixels",), augment=("none",))
            bench._load_common()
            seen: set[int] = set()
            for number in range(3):
                data = bench._load_set(number)
                numbers = set(data.train.index.tolist())
                np.testing.assert_array_equal(data.train.index, bench.sample.sets[number])  # ровно номера выборки
                self.assertFalse(numbers & seen)
                self.assertFalse(numbers & set(bench._held.index.tolist()))
                seen |= numbers
            self.assertEqual(len(seen), 60)
            np.testing.assert_array_equal(bench._held.index, bench.sample.validation)

    def test_overlapping_or_small_pools_are_refused(self) -> None:
        cases = (
            {"BENCH_TRAIN_POOL": [0, 100], "BENCH_VALIDATION_POOL": [50, 200]},  # пулы пересекаются
            {"BENCH_TRAIN_POOL": [0, 30], "BENCH_VALIDATION_POOL": [50000, 60000]},  # 2 набора по 20 не помещаются
            {"BENCH_TRAIN_POOL": [0, 50000], "BENCH_VALIDATION_POOL": [59990, 60000]},  # 20 отложенных из 10
        )
        for changes in cases:
            with self.subTest(changes=changes), settings.overridden({**SMALL, **changes}):
                bench = HonestBench(train=20, sets=2, validation=20, methods=("pixels",), augment=("none",))
                with self.assertRaises(ValueError):
                    bench._load_common()


class SampleTest(unittest.TestCase):
    """Выборка номеров стенда: случайная (не сплошной блок), воспроизводимая по зерну, устроена «префиксом»."""

    def draw(self, **changes: object) -> BenchSample:
        options = {"train": 20, "sets": 3, "validation": 40, "test": 30}
        options.update({key: changes.pop(key) for key in list(changes) if key in options})
        with settings.overridden(changes):
            return BenchSample.draw(**options)

    def test_seeded_random_and_inside_the_pools(self) -> None:
        sample = self.draw()
        self.assertEqual(sample, self.draw())  # то же зерно — те же картинки
        self.assertNotEqual(sample, self.draw(BENCH_SEED=1))
        held = np.asarray(sample.validation)
        low, high = settings.BENCH_VALIDATION_POOL
        self.assertTrue(np.all((held >= low) & (held < high)))
        self.assertFalse(np.all(np.diff(held) == 1))  # не сплошной блок
        taught = np.asarray(sample.sets)
        low, high = settings.BENCH_TRAIN_POOL
        self.assertTrue(np.all((taught >= low) & (taught < high)))
        self.assertEqual(len(set(taught.reshape(-1).tolist())), taught.size)  # наборы не пересекаются
        self.assertTrue(all(0 <= number < BenchSample.TEST_POOL for number in sample.test))

    def test_prefix_property(self) -> None:
        sample = self.draw()
        self.assertEqual(self.draw(validation=10).validation, sample.validation[:10])
        self.assertEqual(self.draw(sets=1).sets[0], sample.sets[0])
        self.assertEqual(self.draw(test=5).test, sample.test[:5])


class TableTest(unittest.TestCase):
    def test_markdown_ends_with_total_and_blueprint_builds(self) -> None:
        with settings.overridden({**SMALL}):
            bench = HonestBench(train=20, sets=2, validation=20, methods=("pixels", "cnn"), augment=("none", "shifts"))
            bench._scores[("pixels", "none", "k=1")] = Score(select=[0.7, 0.8], validation=[0.7, 0.8], test=[None, None])
            bench._scores[("cnn", "none", "lenet5")] = Score(select=[0.8, 0.9], validation=[0.8, 0.9], test=[None, None])
            bench._scores[("cnn", "shifts", "lenet5")] = Score(select=[0.85, 0.9], validation=[0.85, 0.9], test=[None, None])
            bench._sets_done = 2
            text = BenchTable.markdown(bench.result(), final=True)
        lines = text.strip().splitlines()
        self.assertTrue(lines[-1].startswith("| **Итого**"), lines[-1])
        self.assertIn("kNN по пикселям", text)
        self.assertIsNotNone(BenchBlueprint.build())


class BenchRunTest(unittest.TestCase):
    """Стенд целиком на крошечных числах: DAMP, kNN по пикселям, CNN; итог и JSON."""

    def test_full_run_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as folder, settings.overridden({**SMALL, "BENCH_OUTPUT_DIR": folder}):
            bench = HonestBench(
                train=20, sets=2, validation=20, test=20, map_points=None, methods=("damp", "pixels", "cnn"),
                augment=("none", "shifts"),
            )
            result = bench.run()
            files = list(Path(folder).glob("bench-*.json"))
            self.assertEqual(len(files), 1)
            saved = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual(Path(result["path"]).name, files[0].name)
        self.assertEqual(saved["protocol"]["sets"], 2)
        self.assertEqual(len(saved["protocol"]["numbers"]["train"]), 2)
        self.assertEqual(len(saved["protocol"]["numbers"]["validation"]), 20)
        self.assertEqual(len(saved["protocol"]["numbers"]["test"]), 20)
        self.assertIn("BENCH_SETS", saved["settings"])
        self.assertEqual(saved["version"].count("."), 2)
        rows = {(row["method"], row["augment"]): row for row in saved["methods"]}
        self.assertEqual(set(rows), {("damp", "none"), ("damp", "shifts"), ("pixels", "none"), ("pixels", "shifts"),
                                     ("cnn", "none"), ("cnn", "shifts")})
        self.assertFalse(rows[("damp", "shifts")]["supported"])
        self.assertIn("LADDER_TRAIN_VIEWS", rows[("damp", "shifts")]["note"])  # виды DAMP — настройка, не режим стенда
        views = [list(view) for view in saved["settings"]["LADDER_TRAIN_VIEWS"] if tuple(view) != (0, 0)]
        self.assertEqual(saved["protocol"]["damp_train_views"], views)
        self.assertEqual(saved["protocol"]["damp_train_pictures"], 20 * (1 + len(views)))
        for key in (("damp", "none"), ("pixels", "none"), ("cnn", "none"), ("cnn", "shifts")):
            with self.subTest(row=key):
                row = rows[key]
                self.assertTrue(row["supported"])
                self.assertEqual(len(row["validation"]["per_set"]), 2)
                self.assertEqual(len(row["test"]["per_set"]), 2)  # test — у выбранной настройки
                self.assertIn(row["setting"], row["candidates"])
                best = max(row["candidates"].values())
                self.assertAlmostEqual(row["candidates"][row["setting"]], best)  # выбор — по select
                self.assertIn("validation_sampling_se", row)
                self.assertIsNotNone(row["robustness"])
        candidates = set(rows[("damp", "none")]["candidates"])
        self.assertTrue({"k=1", "k=3"} <= candidates)  # битовые коды деталей
        if settings.BENCH_LEVEL_EMBEDDING:  # уровни активности как эмбеддинг (§7.1.12)
            titles = {"clip": "уровни", "train_max": "уровни/max"}
            expected = {f"{titles[scale]} k={k}" for scale in settings.BENCH_LEVEL_SCALES for k in (1, 3)}
            self.assertTrue(expected <= candidates)
        # Линейный читатель над уровнями активации (§9.3).
        self.assertTrue({f"линейный λ={float(decay):g}" for decay in settings.BENCH_LINEAR_DECAYS} <= candidates)
        self.assertTrue(saved["levels"])
        self.assertIsNotNone(saved["levels"][0]["response_cosine"])
        self.assertIsNotNone(saved["gap"])
        self.assertIn("damp_minus_best_cnn", saved["gap"])
        self.assertIn("intervals", saved["gap"])
        self.assertEqual((saved["protocol"]["validation_select"], saved["protocol"]["validation_report"]), (10, 10))
        self.assertIn("strokeless", saved["protocol"])
        self.assertEqual(saved["protocol"]["seed"], settings.BENCH_SEED)
        self.assertEqual(saved["protocol"]["numbers"]["validation"], list(bench.sample.validation))
        self.assertEqual(saved["protocol"]["numbers"]["train"], [list(row) for row in bench.sample.sets])


if __name__ == "__main__":
    unittest.main()
