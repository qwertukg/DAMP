"""Типы событий не зависят от формата лога: одни и те же события в разном оформлении и публичные логи loghub.

Правил под формат в конвейере нет: заголовок строки и переменные места находятся по статистике (вне статьи),
тип события — отклик пространства на код строки целиком (§6.5, §7.1.10).
"""
from __future__ import annotations

import collections
import csv
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("RERUN", "off")

import numpy as np  # noqa: E402

from damp.logging import LOGGER  # noqa: E402
import log_insights  # noqa: E402

LOGGER.configure_intervals({}, default_interval=10**9)

LOGHUB = Path(__file__).resolve().parent.parent / "data" / "loghub"


def grouped_exactly(found: list[int], expected: list[str]) -> float:
    """Доля строк, сгруппированных точно: множество строк типа равно множеству строк шаблона."""
    by_found: dict[int, set[int]] = collections.defaultdict(set)
    by_expected: dict[str, set[int]] = collections.defaultdict(set)
    for index, (kind, template) in enumerate(zip(found, expected)):
        by_found[kind].add(index)
        by_expected[template].add(index)
    exact = sum(len(rows) for rows in by_found.values() if rows == by_expected[expected[next(iter(rows))]])
    return exact / len(found)


def line_types(path: str) -> tuple[list[int], list[int]]:
    """Тип каждой строки файла со словами и номера этих строк."""
    corpus, space = log_insights.LogInsightsExperiment(reference=[path], target=[]).build_space()
    types = log_insights.EventTypes(corpus, space)
    return types.line_types().tolist(), [line.number for line in corpus.lines]


class SameEventsInDifferentFormatsTests(unittest.TestCase):
    """Одни и те же события в пяти оформлениях строки дают одни и те же типы."""

    USERS = ("alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi")
    # (компонент, сообщение): {user} — переменное слово, {n} — число.
    EVENTS = (
        ("auth", "Accepted publickey for {user} from 10.0.0.{n} port {n}"),
        ("auth", "Invalid user {user} from 10.0.0.{n}"),
        ("auth", "Disconnected after too many authentication failures for {user}"),
        ("auth", "session opened for user {user} by uid {n}"),
        ("db", "connection pool exhausted waiting {n}ms for free slot"),
        ("db", "slow query took {n}ms and returned {n} rows"),
        ("db", "replication lag is {n} seconds behind primary"),
        ("cache", "evicted {n} entries because of memory pressure"),
        ("cache", "warmup finished loaded {n} keys"),
        ("http", "request finished with status {n} sent {n} bytes"),
        ("http", "upstream timed out while connecting to backend"),
        ("scheduler", "job {n} postponed because worker queue is full"),
    )
    FORMATS = {
        "plain": "{component}: {message}",
        "syslog": "Nov 12 10:{minute:02d}:{second:02d} web01 app[{pid}]: {component}: {message}",
        "asctime": "[Thu Nov 12 10:{minute:02d}:{second:02d} 2026] [info] {component} {message}",
        "iso": "2026-11-12 10:{minute:02d}:{second:02d},{pid} INFO {component} - {message}",
        "json": '{{"ts": "2026-11-12T10:{minute:02d}:{second:02d}Z", "level": "info", "logger": "{component}", "msg": "{message}"}}',
    }
    REPEATS = 40

    @classmethod
    def _lines(cls, layout: str) -> tuple[list[str], list[str]]:
        rng = np.random.default_rng(0)
        lines, templates = [], []
        for step in range(cls.REPEATS):
            for index in rng.permutation(len(cls.EVENTS)):
                component, message = cls.EVENTS[int(index)]
                text = message.replace("{user}", cls.USERS[int(rng.integers(len(cls.USERS)))])
                while "{n}" in text:
                    text = text.replace("{n}", str(int(rng.integers(1, 5000))), 1)
                lines.append(layout.format(component=component, message=text, minute=step % 60,
                                           second=int(rng.integers(60)), pid=int(rng.integers(100, 999))))
                templates.append(f"{component}:{message}")
        return lines, templates

    def test_types_match_events_in_every_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for name, layout in self.FORMATS.items():
                lines, templates = self._lines(layout)
                path = os.path.join(directory, f"{name}.log")
                Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
                found, numbers = line_types(path)
                expected = [templates[number - 1] for number in numbers]
                with self.subTest(format=name):
                    self.assertEqual(len(found), len(lines))
                    self.assertEqual(len(set(found)), len(self.EVENTS))
                    self.assertEqual(grouped_exactly(found, expected), 1.0)


@unittest.skipUnless(LOGHUB.is_dir(), "нет data/loghub: наборы loghub скачиваются отдельно (docs/LOG_INSIGHTS.md)")
class LoghubTests(unittest.TestCase):
    """Публичные логи пяти форматов с разметкой шаблонов (loghub, по 2000 строк); разметка — только для оценки."""

    # Набор → наименьшая доля точно сгруппированных строк (измерено: 1.00, 1.00, 0.96, 0.68, 0.68).
    EXPECTED = {"Apache": 0.95, "HDFS": 0.95, "Zookeeper": 0.9, "Linux": 0.6, "OpenSSH": 0.6}

    def test_types_follow_templates(self) -> None:
        for name, minimum in self.EXPECTED.items():
            log = LOGHUB / f"{name}_2k.log"
            structured = LOGHUB / f"{name}_2k.log_structured.csv"
            if not (log.is_file() and structured.is_file()):
                continue
            with open(structured, encoding="utf-8", errors="replace") as handle:
                templates = {int(row["LineId"]): row["EventId"] for row in csv.DictReader(handle)}
            found, numbers = line_types(str(log))
            with self.subTest(dataset=name):
                self.assertGreaterEqual(grouped_exactly(found, [templates[number] for number in numbers]), minimum)


if __name__ == "__main__":
    unittest.main()
