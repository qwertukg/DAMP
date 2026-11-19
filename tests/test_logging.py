"""Журнал JSON Lines: одна запись на строку, полные данные, краткие длинные массивы, ротация по размеру."""
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("RERUN", "off")

import numpy as np  # noqa: E402

from damp.logging import JsonlLog  # noqa: E402


class JsonlLogTest(unittest.TestCase):
    def test_records_are_json_lines_with_full_data(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "logs" / "damp.jsonl"
            log = JsonlLog(str(path), max_bytes=10**6, backups=2, array_limit=4)
            log.write("ladder.level", "6.5", {"level": 1, "shift": {(3, 5): 1.0}, "short": np.arange(3), "long": np.arange(10)}, "ladder.level level=1 [6.5]")
            log.write("encoder.density.warning", "2", None, "encoder.density.warning [2]")
            log.close()
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["event"] for r in records], ["ladder.level", "encoder.density.warning"])
        self.assertEqual([r["level"] for r in records], ["info", "warning"])
        first = records[0]
        self.assertEqual(first["section"], "6.5")
        self.assertEqual(first["data"]["level"], 1)
        self.assertEqual(first["data"]["shift"], {"(3, 5)": 1.0})
        self.assertEqual(first["data"]["short"], [0, 1, 2])
        self.assertEqual(first["data"]["long"], {"shape": [10], "dtype": first["data"]["long"]["dtype"], "min": 0.0, "max": 9.0})
        self.assertEqual(first["run"], log.run)
        self.assertTrue(first["msg"].endswith("[6.5]"))

    def test_rotation_keeps_the_newest_records(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "damp.jsonl"
            log = JsonlLog(str(path), max_bytes=400, backups=2, array_limit=10)
            for number in range(40):
                log.write("figure.save", "вне статьи", {"number": number}, f"figure.save number={number}")
            files = sorted(p.name for p in Path(folder).iterdir())
            last = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
            log.close()
        self.assertEqual(files, ["damp.jsonl", "damp.jsonl.1", "damp.jsonl.2"])
        self.assertEqual(last["data"]["number"], 39)

    def test_busy_file_is_not_rotated_and_records_keep_coming(self) -> None:
        """Файл держит другой процесс (пары прогонов подбора): перенос на Windows падает — запись идёт в тот же файл,
        без ошибок на каждой записи, перенос — после паузы."""
        from unittest import mock

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "damp.jsonl"
            log = JsonlLog(str(path), max_bytes=200, backups=2, array_limit=10)
            handler = log._logger.handlers[0]
            with mock.patch.object(handler, "rotate", side_effect=PermissionError("занят")), \
                    mock.patch.object(handler, "handleError") as failed:
                for number in range(20):
                    log.write("figure.save", "вне статьи", {"number": number}, f"figure.save number={number}")
            lines = path.read_text(encoding="utf-8").splitlines()
            log.close()
        failed.assert_not_called()
        self.assertEqual(json.loads(lines[-1])["data"]["number"], 19)

    def test_log_opens_later_when_the_file_is_busy_at_start(self) -> None:
        """В миг старта файл переносит другой процесс: журнал не падает, а открывает файл при следующей записи."""
        from unittest import mock

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "damp.jsonl"
            log = JsonlLog(str(path), max_bytes=10_000, backups=2, array_limit=10)
            handler = log._logger.handlers[0]
            real_open = handler._open
            with mock.patch.object(handler, "_open", side_effect=[PermissionError("занят"), real_open()]):
                log.write("figure.save", "вне статьи", {"number": 1}, "первая — файл занят")
                log.write("figure.save", "вне статьи", {"number": 2}, "вторая — файл свободен")
            lines = path.read_text(encoding="utf-8").splitlines()
            log.close()
        self.assertEqual([json.loads(line)["data"]["number"] for line in lines], [2])


class LoggerNotMutedTest(unittest.TestCase):
    """Тесты не глушат журнал: прогоны тестов видны в окне Rerun (RERUN=on), подписки хода счёта работают."""

    def test_logger_is_not_muted(self) -> None:
        from damp.logging import LOGGER

        self.assertFalse(LOGGER._muted, "какой-то модуль тестов вызвал LOGGER.mute(): журнал и Rerun молчат весь прогон")


class CycleBlueprintReturnTest(unittest.TestCase):
    """Пока идёт цикл — его раскладка; первое событие вне цикла возвращает окно к базовой раскладке.

    Базовая — последняя, заданная use_blueprint без cycle=True (по умолчанию — ход раскладки).
    """

    def test_returns_to_layout_progress_outside_cycle(self) -> None:
        from unittest import mock

        from damp.logging import LOGGER, RerunViewerBlueprint

        def builder() -> None:
            return None

        previous, saved = LOGGER.cycle, (LOGGER._base_blueprint, LOGGER._blueprint_builder, LOGGER._cycle_blueprint)
        LOGGER._base_blueprint = RerunViewerBlueprint.build  # базовая раскладка по умолчанию, что бы ни задали другие тесты
        with mock.patch.object(LOGGER, "_send_blueprint"):
            try:
                LOGGER.use_blueprint(builder, cycle=True)
                LOGGER.set_cycle(2)
                LOGGER.event("test.cycle.inside", section="вне статьи")
                self.assertIs(LOGGER._blueprint_builder, builder)
                LOGGER.set_cycle(None)
                LOGGER.event("test.cycle.outside", section="вне статьи")
                self.assertEqual(LOGGER._blueprint_builder, RerunViewerBlueprint.build)
            finally:
                LOGGER.set_cycle(previous)
                LOGGER._base_blueprint, LOGGER._blueprint_builder, LOGGER._cycle_blueprint = saved

    def test_returns_to_last_base_blueprint(self) -> None:
        from unittest import mock

        from damp.logging import LOGGER, RerunViewerBlueprint

        def base() -> None:
            return None

        def cycle_layout() -> None:
            return None

        previous, saved = LOGGER.cycle, (LOGGER._base_blueprint, LOGGER._blueprint_builder, LOGGER._cycle_blueprint)
        with mock.patch.object(LOGGER, "_send_blueprint"):
            try:
                LOGGER.use_blueprint(base)
                LOGGER.use_blueprint(cycle_layout, cycle=True)
                self.assertIs(LOGGER._base_blueprint, base, "раскладка цикла не должна стать базовой")
                LOGGER.set_cycle(1)
                LOGGER.event("test.cycle.inside", section="вне статьи")
                self.assertIs(LOGGER._blueprint_builder, cycle_layout)
                LOGGER.set_cycle(None)
                LOGGER.event("test.cycle.outside", section="вне статьи")
                self.assertIs(LOGGER._blueprint_builder, base)
            finally:
                LOGGER.set_cycle(previous)
                LOGGER._base_blueprint, LOGGER._blueprint_builder, LOGGER._cycle_blueprint = saved


if __name__ == "__main__":
    unittest.main()
