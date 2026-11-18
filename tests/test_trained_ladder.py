"""Обученный цикл (system/damp/cycle/trained.py): новая картинка идёт через уровни так же, как картинка прогона.

Цифры проверочной группы прогона подаются в TrainedLadder.see как рисунки — пикселами из MNIST — и на каждом уровне
сверяются с прогоном цифра в цифру: окна, их вхождения, срабатывания детекторов, детали, пики, вероятности классов.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class TrainedLadderTest(unittest.TestCase):
    """Один рисунок через обученные уровни = та же картинка в прогоне MnistLadderRunner (§6.5, §6.6, §9.3; П.1–П.5)."""

    TRAIN, TEST = 30, 6

    @classmethod
    def setUpClass(cls) -> None:
        from torchvision.datasets import MNIST

        from damp import settings
        from damp.cycle.ladder import MnistLadderRunner
        from damp.cycle.ladder_trace import LadderObserver
        from damp.cycle.trained import TrainedLadder

        class Traces(LadderObserver):
            def __init__(self) -> None:
                self.levels = []

            def level(self, trace) -> None:  # noqa: ANN001
                self.levels.append(trace)

        cls.traces = Traces()
        changes = {"LADDER_SAVE_MAPS": False, "LADDER_TRAIN_VIEWS": ()}
        with tempfile.TemporaryDirectory() as folder, settings.overridden(changes):
            cls.runner = MnistLadderRunner(
                count=cls.TRAIN, evaluate_limit=cls.TEST, map_max_points=None, shifts=(),
                file_prefix=str(Path(folder) / "ladder"), observer=cls.traces, keep_levels=True,
            )
            cls.runner.run()
            cls.ladder = TrainedLadder(cls.runner.groups, cls.runner.levels, cls.runner.level_peaks, decay=1e-7)
            groups = cls.runner.groups
            first, last = groups.ranges["test"]
            mnist = MNIST(root="./data", train=False, download=True)
            cls.pictures = list(range(first, last))
            cls.sights = {p: cls.ladder.see(mnist.data[int(groups.atoms.index[p])].numpy()) for p in cls.pictures}

    def test_levels_are_kept_one_per_trained_level(self) -> None:
        self.assertTrue(self.traces.levels)
        self.assertEqual(len(self.runner.levels), len(self.traces.levels))
        self.assertEqual(len(self.runner.levels), len(self.runner.level_peaks))
        for trained, trace in zip(self.runner.levels, self.traces.levels):
            self.assertEqual(trained.number, trace.number)
            self.assertEqual(trained.point_picture.shape[0], trace.map_cells.shape[0])
            # Каждая точка карты взята из окна обучающей картинки.
            self.assertTrue(np.all((trained.point_picture >= 0) & (trained.point_picture < self.runner.groups.train_pictures)))

    def test_sticks_are_those_of_the_run(self) -> None:
        groups = self.runner.groups
        atoms = groups.atoms
        for picture, sight in self.sights.items():
            with self.subTest(picture=picture):
                first, last = groups.sticks(picture)
                self.assertTrue(np.array_equal(sight.stick_angle, atoms.angle[first:last]))
                self.assertTrue(np.array_equal(sight.stick_y, atoms.y[first:last]))
                self.assertTrue(np.array_equal(sight.stick_x, atoms.x[first:last]))
                self.assertTrue(np.array_equal(sight.stick_strength, atoms.strength[first:last]))

    def test_every_level_repeats_the_run(self) -> None:
        groups = self.runner.groups
        for picture, sight in self.sights.items():
            for index, trace in enumerate(self.traces.levels):
                rows = np.flatnonzero(trace.keyhole_picture == picture)
                with self.subTest(picture=picture, level=trace.number):
                    if rows.size == 0:
                        # Окон нет — картинка дальше не идёт, как в прогоне.
                        self.assertEqual(len(sight.levels), index)
                        break
                    level = sight.levels[index]
                    base = int(rows[0])
                    self.assertEqual(level.number, trace.number)
                    self.assertTrue(np.array_equal(level.keyhole_centre, trace.keyhole_centre[rows]))
                    self.assertTrue(np.array_equal(level.keyhole_extent, trace.keyhole_extent[rows]))
                    entries = np.isin(trace.entry_keyhole, rows)
                    self.assertTrue(np.array_equal(level.entry_keyhole, trace.entry_keyhole[entries] - base))
                    # Элементы уровня — штрихи либо детали прошлого уровня после пятен (П.3): в пути одной картинки
                    # они считаются от её первого элемента.
                    shift = int(np.flatnonzero(trace.element_picture == picture)[0])
                    if index == 0:
                        self.assertEqual(shift, groups.sticks(picture)[0])
                    self.assertTrue(np.array_equal(level.entry_element, trace.entry_element[entries] - shift))
                    fired = np.isin(trace.fired_keyhole, rows)
                    self.assertTrue(np.array_equal(level.fired_keyhole, trace.fired_keyhole[fired] - base))
                    self.assertTrue(np.array_equal(level.fired_detector, trace.fired_detector[fired]))
                    self.assertTrue(np.array_equal(level.fired_level, trace.fired_level[fired]))
                    self.assertTrue(np.array_equal(level.detail_codes, trace.detail_codes[rows]))
                    self.assertEqual(level.detail_length, trace.detail_length)
                    self.assertTrue(np.array_equal(level.strength, trace.detail_strength[rows]))
                    self.assertTrue(np.array_equal(level.peak, self.runner.level_peaks[index][picture]))

    def test_reader_answer_is_that_of_the_run(self) -> None:
        features = np.concatenate(self.runner.level_peaks, axis=1)
        for picture, sight in self.sights.items():
            with self.subTest(picture=picture):
                expected = self.ladder.reader.probabilities(features[picture : picture + 1])[0]
                self.assertTrue(np.array_equal(sight.probabilities, expected))
                self.assertEqual(sight.answer, int(self.ladder.reader.answers(features[picture : picture + 1])[0]))

    def test_lit_points_match_the_fired_detectors(self) -> None:
        """Зажжённые точки — показ (§6.1): у окна со сработавшим детектором есть зажжённые точки, номера — точки карты."""
        for sight in self.sights.values():
            for level, trained in zip(sight.levels, self.runner.levels):
                with self.subTest(level=level.number):
                    points = trained.point_picture.shape[0]
                    self.assertTrue(np.all((level.lit_point >= 0) & (level.lit_point < points)))
                    self.assertTrue(np.all(level.lit_value > 0))
                    self.assertTrue(set(np.unique(level.fired_keyhole).tolist()) <= set(np.unique(level.lit_keyhole).tolist()))

    def test_memory_votes_read_the_lit_points(self) -> None:
        """Голоса памяти (§2.2.5, §7.1.13.3): по уровню на каждый пройденный, из зажжённых точек; ответ — их среднее."""
        memory = self.ladder.memory
        self.assertEqual(len(memory.thresholds), len(self.runner.levels))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in memory.thresholds))
        for sight in self.sights.values():
            self.assertEqual([votes.number for votes in sight.votes], [level.number for level in sight.levels])
            for level, votes in zip(sight.levels, sight.votes):
                with self.subTest(level=level.number):
                    expected = memory.level(level.number, level.keyhole_centre.shape[0], level.lit_keyhole, level.lit_point, level.lit_value)
                    self.assertTrue(np.array_equal(votes.votes, expected.votes))
                    self.assertEqual(votes.knowledge, expected.knowledge)
                    self.assertEqual(votes.keyholes, level.keyhole_centre.shape[0])
                    self.assertEqual(votes.voted, np.unique(level.lit_keyhole).shape[0])
            expected = memory.verdict(sight.votes)
            self.assertEqual((sight.verdict.kind, sight.verdict.digit, sight.verdict.share, sight.verdict.agreed),
                             (expected.kind, expected.digit, expected.share, expected.agreed))
            self.assertTrue(np.array_equal(sight.verdict.votes, expected.votes))
            self.assertIn(sight.verdict.kind, (memory.AGREE, memory.MOSTLY, memory.SPLIT))
            self.assertEqual(sight.verdict.digit, int(np.mean([votes.votes for votes in sight.votes], axis=0).argmax()))

    def test_training_picture_knows_itself_unless_left_out(self) -> None:
        """Обучающая картинка находит в памяти саму себя; мерка порога знания считает её без своих точек памяти."""
        atoms = self.runner.groups.atoms
        first, _ = self.runner.groups.ranges["train"]
        higher = 0
        for picture in range(first, first + 5):
            whole = self.ladder._see(atoms.images[picture])
            left_out = self.ladder._see(atoms.images[picture], own=int(atoms.index[picture]))
            for with_own, without in zip(whole.votes, left_out.votes):
                self.assertGreaterEqual(with_own.knowledge, without.knowledge)
                higher += int(with_own.knowledge > without.knowledge)
        self.assertGreater(higher, 0)

    def test_empty_picture_has_no_answer(self) -> None:
        sight = self.ladder.see(np.zeros((28, 28), dtype=np.uint8))
        self.assertEqual(sight.levels, ())
        self.assertEqual(sight.answer, -1)
        self.assertFalse(sight.probabilities.any())
        self.assertEqual(sight.votes, ())
        self.assertEqual((sight.verdict.kind, sight.verdict.digit), (self.ladder.memory.UNKNOWN, -1))


if __name__ == "__main__":
    unittest.main()
