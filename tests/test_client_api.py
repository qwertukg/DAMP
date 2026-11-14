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
        elements = damp.coding.sticks(digits)
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
        self.assertEqual(damp.recognition.vote(elements), runner.top_accuracy)


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
