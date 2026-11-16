"""Клиентский фреймворк damp_api: этапы по видению автора, настройки, тот же результат, что у системы напрямую,
и правило «к системе — только через клиентскую часть»."""
import ast
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("RERUN", "off")

import numpy as np  # noqa: E402

from damp_api import Code, Damp, Stimulus  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class ClientApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.damp = Damp(log_file=False)

    def test_stages_follow_author_pipeline(self) -> None:
        stages = self.damp.stages
        self.assertEqual([stage.number for stage in stages], [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(
            [stage.name for stage in stages],
            ["stimuli", "coding", "layout", "detectors", "embeddings", "cycle", "recognition"],
        )
        self.assertEqual({stage.number: stage.cycle for stage in stages if stage.cycle}, {2: "начало", 6: "конец"})
        for stage in stages:
            with self.subTest(stage=stage.name):
                part = getattr(self.damp, stage.name)
                self.assertEqual(part.about, stage)
                self.assertTrue((ROOT / stage.code).is_dir(), stage.code)

    def test_settings_are_read_only_config_names(self) -> None:
        settings = self.damp.settings
        self.assertEqual(settings.path.name, "config.toml")
        self.assertGreater(len(settings), 300)
        self.assertIn("LADDER_TOP_COVERAGE", settings)
        self.assertNotIn("ladder_top_coverage", settings)
        with self.assertRaises(TypeError):
            settings["LADDER_TOP_COVERAGE"] = 1.0  # type: ignore[index]
        self.assertEqual(self.damp.version.count("."), 2)

    def test_encode_matches_system_encoder(self) -> None:
        from damp.encoding.mnist_codes import _build_encoder

        code = self.damp.coding.encode(45, 14, 14)
        _, bits = _build_encoder().encode(45.0, 14.0, 14.0)
        self.assertIsInstance(code, Code)
        self.assertEqual(code.ones, tuple(int(i) for i in bits.active_indices()))
        self.assertGreater(code.density, 0.0)
        self.assertLess(code.density, 0.5)

    def test_stimuli_match_system_extractor(self) -> None:
        from damp.stimuli.mnist import _build_extractor

        image = np.zeros((28, 28), dtype=np.float32)
        image[6:22, 12:16] = 1.0
        stimuli = self.damp.stimuli.of_image(image * 255, label=1)
        direct = _build_extractor().extract(image, label=1).get(1, [])
        self.assertTrue(stimuli)
        self.assertTrue(all(isinstance(item, Stimulus) for item in stimuli))
        self.assertEqual([(s.angle, s.x, s.y) for s in stimuli], [(float(a), int(round(x)), int(round(y))) for a, x, y in direct])

    def test_detectors_picture_is_saved(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            target = str(Path(folder) / "detectors.png")
            self.assertEqual(self.damp.coding.detectors_picture(path=target), target)
            self.assertGreater(Path(target).stat().st_size, 0)


class SettingsChangesTest(unittest.TestCase):
    """Свои значения настроек поверх файла: действуют только внутри вызовов своего Damp."""

    def test_changes_apply_to_own_damp_only(self) -> None:
        from damp import settings

        base = Damp(log_file=False)
        tuned = base.with_settings(LADDER_TOP_COVERAGE=0.5, ENCODER_RANDOM_SEED=7)
        self.assertEqual(tuned.settings["LADDER_TOP_COVERAGE"], 0.5)
        self.assertEqual(tuned.settings["LADDER_RANDOM_SEED"], 7)  # выводимое значение следует за источником
        self.assertEqual(tuned.settings.changes, {"LADDER_TOP_COVERAGE": 0.5, "ENCODER_RANDOM_SEED": 7})
        self.assertEqual(base.settings["LADDER_TOP_COVERAGE"], settings.LADDER_TOP_COVERAGE)
        self.assertNotEqual(settings.LADDER_TOP_COVERAGE, 0.5)  # после вызова файл снова в силе
        self.assertEqual(settings.LADDER_RANDOM_SEED, settings.ENCODER_RANDOM_SEED)

    def test_unknown_and_derived_names_are_refused(self) -> None:
        with self.assertRaises(KeyError):
            Damp(log_file=False, changes={"NO_SUCH_SETTING": 1})
        with self.assertRaises(KeyError):
            Damp(log_file=False, changes={"LADDER_RANDOM_SEED": 1})  # выводится из ENCODER_RANDOM_SEED


class CycleBlocksTest(unittest.TestCase):
    """Цикл, собранный кубиками фреймворка, — тот же, что готовый прогон системы, цифра в цифру."""

    TRAIN, TEST = 30, 20

    def test_blocks_repeat_ready_cycle(self) -> None:
        from damp import settings
        from damp.cycle.ladder import MnistLadderRunner
        from damp.cycle.ladder_trace import LadderObserver

        class Traces(LadderObserver):
            def __init__(self) -> None:
                self.levels = []

            def level(self, trace) -> None:  # noqa: ANN001
                self.levels.append(trace)

        damp = Damp(log_file=False, changes={"LADDER_SAVE_MAPS": False})
        traces = Traces()
        with tempfile.TemporaryDirectory() as folder, settings.overridden({"LADDER_SAVE_MAPS": False}):
            runner = MnistLadderRunner(
                count=self.TRAIN, evaluate_limit=self.TEST, map_max_points=None, shifts=(),
                file_prefix=str(Path(folder) / "ladder"), observer=traces,
            )
            runner.run()
        self.assertTrue(traces.levels)

        digits = damp.stimuli.digits(train=self.TRAIN, test=self.TEST)
        # Виды обучающих (§7.1.4) у кубиков — те же, что у готового прогона (LADDER_TRAIN_VIEWS).
        self.assertEqual(digits._groups.ranges, runner.groups.ranges)
        elements = damp.coding.sticks(digits)
        firings = []
        for trace in traces.levels:
            with self.subTest(level=trace.number):
                level = damp.cycle.step(elements)
                self.assertEqual(level.number, trace.number)
                self.assertTrue(np.array_equal(level.codes._codes.packed, trace.keyhole_codes))
                self.assertTrue(np.array_equal(level.memory._memory.layout.code_table.packed, trace.map_codes))
                self.assertEqual(level.detectors.count, trace.detectors.shape[0])
                self.assertTrue(np.array_equal(level.firing._firing.columns, trace.fired_detector))
                self.assertTrue(np.array_equal(level.details._elements.codes(), trace.detail_codes))
                elements = level.details
                firings.append(level.firing)
        self.assertEqual(damp.recognition.vote(elements), runner.top_accuracy)
        # §9.3: линейный читатель из кубиков — тот же, что над пиками уровней готового прогона.
        from damp.decode.linear_reader import LinearReader

        groups = runner.groups
        first, last = groups.ranges["test"]
        expected = LinearReader.answers_for(
            np.concatenate(runner.level_peaks, axis=1), groups.labels, groups.ranges["train"], {"test": (first, last)}, (1e-7,)
        )[1e-7]["test"]
        truth = groups.labels[first:last]
        self.assertEqual(damp.recognition.linear(firings, decays=(1e-7,)),
                         {1e-7: float(np.mean((expected == truth) & (expected >= 0)))})


class TrainedLadderPartTest(unittest.TestCase):
    """Кубик `cycle.trained`: часть `TrainedLadder` отдаёт простые значения, согласованные с картами уровней, — и их же
    шлёт страница дашборда (client/dashboard/server.py). Совпадение пути картинки с прогоном цифра в цифру — в
    tests/test_trained_ladder.py."""

    def test_part_maps_sight_and_dashboard_answers(self) -> None:
        import json

        from torchvision.datasets import MNIST

        from dashboard.server import DashboardData

        with tempfile.TemporaryDirectory() as folder:
            damp = Damp(log_file=False, changes={"LADDER_TRAIN_VIEWS": (), "DASHBOARD_FILE_PREFIX": str(Path(folder) / "ladder")})
            ladder = damp.cycle.trained(30)
        self.assertGreaterEqual(ladder.levels, 1)
        self.assertEqual(ladder.train_pictures, 30)
        maps, sticks = ladder.maps(), ladder.sticks()
        self.assertEqual(len(maps), ladder.levels)
        self.assertEqual(int(np.asarray(sticks.labels).shape[0]), 30)
        for level in maps:
            points, detectors = np.asarray(level.labels).shape[0], np.asarray(level.layer).shape[0]
            self.assertEqual(np.asarray(level.cells).shape, (points, 2))
            self.assertEqual(np.asarray(level.detectors).shape, (detectors, 5))
            self.assertTrue(np.all(np.asarray(level.point_picture) < 30))
            self.assertEqual(len(level.layer_thresholds), int(np.asarray(level.layer).max()) + 1)
        image = MNIST(root="./data", train=False, download=True).data[0].numpy()
        sight = ladder.see(image)
        self.assertIn(sight.digit, range(-1, 10))
        self.assertAlmostEqual(sum(sight.probabilities), 1.0 if sight.digit >= 0 else 0.0, places=6)
        for level, built in zip(sight.levels, maps):
            keyholes = np.asarray(level.keyhole_extent).shape[0]
            self.assertTrue(np.all(np.asarray(level.lit_point) < np.asarray(built.labels).shape[0]))
            self.assertTrue(np.all(np.asarray(level.fired_detector) < np.asarray(built.layer).shape[0]))
            self.assertTrue(np.all(np.asarray(level.fired_keyhole) < keyholes))
        self.assertTrue(json.dumps(DashboardData.model(ladder, maps, sticks, damp.settings)))
        answer = DashboardData.sight(sight, maps, 40)
        self.assertEqual(len(answer["levels"]), len(sight.levels))
        self.assertEqual(len(answer["probabilities"]), 10)


class DigitGroupNamesTest(unittest.TestCase):
    """Имена групп картинок: у разных сдвигов на пиксели разные имена."""

    def test_pixel_shifts_have_distinct_names(self) -> None:
        from types import SimpleNamespace

        from damp_api.parts import Digits

        ranges = {"train": (0, 5), "validation": (5, 8), "test": (8, 9), (1, 0): (9, 10),
                  ("pixel", 1, 1): (10, 11), ("pixel", 1, 2): (11, 12)}
        digits = Digits(None, SimpleNamespace(ranges=ranges))
        names = digits.groups
        self.assertEqual(len(names), len(ranges))
        self.assertEqual(names["pixel:1,1"], (10, 11))
        self.assertEqual(names["pixel:1,2"], (11, 12))
        self.assertEqual(names["1,0"], (9, 10))
        self.assertEqual(names["validation"], (5, 8))


class BenchApiTest(unittest.TestCase):
    """Стенд честного сравнения через фреймворк: тот же итог, что у системы, и простые записи на выходе."""

    def test_bench_through_api(self) -> None:
        from damp_api import BenchMethod, BenchReport, BenchScore

        changes = {
            "BENCH_CNN_ARCHS": ("lenet5",), "BENCH_CNN_LR": (0.001,), "BENCH_CNN_WEIGHT_DECAY": (0.0,),
            "BENCH_CNN_DROPOUT": (0.0,), "BENCH_CNN_EPOCHS": 2, "BENCH_CNN_PATIENCE": 1, "BENCH_NEIGHBOURS": (1, 3),
            "LADDER_SAVE_MAPS": False,
        }
        with tempfile.TemporaryDirectory() as folder:
            damp = Damp(log_file=False, changes={**changes, "BENCH_OUTPUT_DIR": folder})
            report = damp.recognition.bench(
                train=20, sets=2, validation=20, test=0, methods=("pixels", "cnn"), augment=("none", "shifts")
            )
            self.assertTrue(Path(report.path).is_file())
        self.assertIsInstance(report, BenchReport)
        self.assertEqual((report.train, report.sets, report.validation, report.test), (20, 2, 20, 0))
        rows = {(row.method, row.augment): row for row in report.methods}
        self.assertEqual(set(rows), {("pixels", "none"), ("pixels", "shifts"), ("cnn", "none"), ("cnn", "shifts")})
        self.assertIsInstance(rows[("cnn", "none")], BenchMethod)
        self.assertIsInstance(rows[("cnn", "none")].validation, BenchScore)
        self.assertEqual(len(rows[("cnn", "none")].validation.per_set), 2)
        self.assertFalse(rows[("pixels", "shifts")].supported)
        self.assertIsNone(rows[("cnn", "none")].test)  # test=0: проверочные не считались
        self.assertEqual(report.levels, ())
        self.assertIsNone(report.gap)  # DAMP не считался

    def test_bench_refuses_unknown_method(self) -> None:
        with self.assertRaises(ValueError):
            Damp(log_file=False).recognition.bench(methods=("magic",))


class ReadmeApiLinksTest(unittest.TestCase):
    """Ссылки справочника API в README ведут на строку своей функции: код сдвинулся — ссылки обновить."""

    def test_links_point_to_their_functions(self) -> None:
        import re

        text = (ROOT / "README.md").read_text(encoding="utf-8")
        links = re.findall(r"\[`([^`]+)`\]\((client/damp_api/\w+\.py)#L(\d+)\)", text)
        self.assertGreater(len(links), 30)
        for shown, path, line in links:
            target = (ROOT / path).read_text(encoding="utf-8").splitlines()[int(line) - 1].strip()
            name = re.match(r"(?:detectors\.)?(\w+)", shown).group(1)
            expected = {"Damp": "def __init__", "settings": "def __getitem__"}.get(name)
            with self.subTest(link=shown):
                self.assertTrue(target.startswith(("def ", "class ")), f"{path}#L{line}: {target}")
                self.assertIn(expected or name, target, f"{path}#L{line}: {target}")


class SystemOnlyThroughClientTest(unittest.TestCase):
    """В клиентской части к системному пакету damp обращается только мост damp_api/_system.py."""

    def test_only_bridge_imports_system(self) -> None:
        bridge = ROOT / "client" / "damp_api" / "_system.py"
        for path in (ROOT / "client").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            modules = [
                name
                for node in ast.walk(tree)
                for name in (
                    [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
                    else [alias.name for alias in node.names] if isinstance(node, ast.Import)
                    else []
                )
            ]
            touches = [name for name in modules if name == "damp" or name.startswith("damp.")]
            with self.subTest(file=str(path.relative_to(ROOT))):
                if path == bridge:
                    self.assertTrue(touches)
                else:
                    self.assertEqual(touches, [], "к системе — только через damp_api/_system.py")


if __name__ == "__main__":
    unittest.main()
