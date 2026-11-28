"""Добавочные картинки верхних уровней (LADDER_EXTRA_PICTURES, LADDER_EXTRA_FROM; вне статьи, расширение автора).

Добавочные картинки стоят в наборе сразу за обучающими. Ниже уровня LADDER_EXTRA_FROM они только предъявляются (как
проверочные): в память уровня не входят. С уровня LADDER_EXTRA_FROM они — обучающие уровня: карта уровня раскладывается
и на их окнах. Проверочные картинки в память не входят никогда. Выключенная настройка набор картинок не меняет.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class ExtraPicturesGroupsTest(unittest.TestCase):
    """Набор картинок: группа «extra» и число обучающих картинок уровня."""

    TRAIN, EXTRA, TEST = 20, 10, 5

    def test_extra_group_follows_train(self) -> None:
        from damp.stimuli.digits import DigitGroups, MnistDigits

        plain = DigitGroups.load(MnistDigits(), train=self.TRAIN, test=self.TEST, train_views=())
        more = DigitGroups.load(MnistDigits(), train=self.TRAIN, test=self.TEST, train_views=(), extra=self.EXTRA)
        self.assertNotIn("extra", plain.ranges)
        self.assertEqual(more.ranges["train"], (0, self.TRAIN))
        self.assertEqual(more.ranges["extra"], (self.TRAIN, self.TRAIN + self.EXTRA))
        self.assertEqual(more.ranges["test"], (self.TRAIN + self.EXTRA, self.TRAIN + self.EXTRA + self.TEST))
        index, before = np.asarray(more.atoms.index), np.asarray(plain.atoms.index)
        # Обучающие и проверочные — те же картинки, что без добавочных; добавочные — следующие по порядку MNIST train.
        self.assertTrue(np.array_equal(index[: self.TRAIN], before[: self.TRAIN]))
        self.assertTrue(np.array_equal(index[self.TRAIN + self.EXTRA :], before[self.TRAIN :]))
        extra = index[self.TRAIN : self.TRAIN + self.EXTRA]
        self.assertGreater(int(extra.min()), int(index[: self.TRAIN].max()))
        self.assertEqual(np.unique(extra).size, self.EXTRA)
        self.assertEqual(more.train_pictures, self.TRAIN)

    def test_taught_pictures_by_level(self) -> None:
        from damp import settings
        from damp.stimuli.digits import DigitGroups, MnistDigits

        plain = DigitGroups.load(MnistDigits(), train=self.TRAIN, test=self.TEST, train_views=())
        more = DigitGroups.load(MnistDigits(), train=self.TRAIN, test=self.TEST, train_views=(), extra=self.EXTRA)
        with settings.overridden({"LADDER_EXTRA_PICTURES": self.EXTRA, "LADDER_EXTRA_FROM": 0}):
            self.assertEqual([more.taught_pictures(n) for n in (0, 1, 3, 9)], [self.TRAIN] * 4)
        with settings.overridden({"LADDER_EXTRA_PICTURES": self.EXTRA, "LADDER_EXTRA_FROM": 3}):
            self.assertEqual([more.taught_pictures(n) for n in (0, 1, 2)], [self.TRAIN] * 3)
            self.assertEqual([more.taught_pictures(n) for n in (3, 4)], [self.TRAIN + self.EXTRA] * 2)
            # Нет группы «extra» — обучающие те же на всех уровнях.
            self.assertEqual([plain.taught_pictures(n) for n in (1, 3, 4)], [self.TRAIN] * 3)


    def test_taught_pictures_by_level_list(self) -> None:
        """LADDER_EXTRA_PICTURES списком по уровням: на уровне обучающие — первые столько добавочных картинок."""
        from damp import settings
        from damp.stimuli.digits import DigitGroups, MnistDigits

        more = DigitGroups.load(MnistDigits(), train=self.TRAIN, test=self.TEST, train_views=(), extra=self.EXTRA)
        with settings.overridden({"LADDER_EXTRA_PICTURES": [0, 0, 4, 10], "LADDER_EXTRA_FROM": 0}):
            self.assertEqual(DigitGroups.extra_wanted(), 10)
            self.assertEqual([more.taught_pictures(n) for n in (0, 1, 2, 3, 4, 5)],
                             [self.TRAIN, self.TRAIN, self.TRAIN, self.TRAIN + 4, self.TRAIN + 10, self.TRAIN + 10])
        with settings.overridden({"LADDER_EXTRA_PICTURES": [0, 50]}):
            self.assertEqual(more.taught_pictures(2), self.TRAIN + self.EXTRA)  # не больше, чем добавочных в наборе
        with settings.overridden({"LADDER_EXTRA_PICTURES": 10, "LADDER_EXTRA_FROM": 0}):
            self.assertEqual(DigitGroups.extra_wanted(), 0)
        with settings.overridden({"LADDER_EXTRA_PICTURES": 10, "LADDER_EXTRA_FROM": 3}):
            self.assertEqual(DigitGroups.extra_wanted(), 10)
            self.assertEqual([more.taught_pictures(n) for n in (2, 3)], [self.TRAIN, self.TRAIN + 10])


class ExtraPicturesByLevelLadderTest(unittest.TestCase):
    """Прогон со списком по уровням: чем выше уровень, тем больше добавочных картинок в его памяти."""

    TRAIN, TEST, BY_LEVEL = 30, 2, [0, 8, 20]

    @classmethod
    def setUpClass(cls) -> None:
        from damp import settings
        from damp.cycle.ladder import MnistLadderRunner

        changes = {"LADDER_SAVE_MAPS": False, "LADDER_TRAIN_VIEWS": (), "LADDER_EXTRA_PICTURES": cls.BY_LEVEL, "LADDER_EXTRA_FROM": 0}
        with tempfile.TemporaryDirectory() as folder, settings.overridden(changes):
            cls.runner = MnistLadderRunner(
                count=cls.TRAIN, evaluate_limit=cls.TEST, map_max_points=None, shifts=(),
                file_prefix=str(Path(folder) / "ladder"), keep_levels=True,
            )
            cls.runner.run()

    def test_each_level_takes_its_own_number_of_extra_pictures(self) -> None:
        self.assertEqual(self.runner.groups.ranges["extra"], (self.TRAIN, self.TRAIN + max(self.BY_LEVEL)))
        if len(self.runner.levels) < 3:
            self.skipTest("прогон построил меньше 3 уровней")
        for level in self.runner.levels:
            with self.subTest(level=level.number):
                picture = np.asarray(level.point_picture)
                picture = picture[picture >= 0]
                allowed = self.TRAIN + self.BY_LEVEL[min(int(level.number), len(self.BY_LEVEL)) - 1]
                self.assertTrue(np.all(picture < allowed))
                if allowed > self.TRAIN:
                    self.assertTrue(np.any(picture >= self.TRAIN))


class ExtraPicturesLadderTest(unittest.TestCase):
    """Прогон: память уровней ниже LADDER_EXTRA_FROM — только из обучающих, с него — ещё и из добавочных."""

    TRAIN, EXTRA, TEST, FROM = 30, 20, 4, 2

    @classmethod
    def setUpClass(cls) -> None:
        from damp import settings
        from damp.cycle.ladder import MnistLadderRunner
        from damp.cycle.trained import TrainedLadder

        changes = {"LADDER_SAVE_MAPS": False, "LADDER_TRAIN_VIEWS": (), "LADDER_EXTRA_PICTURES": cls.EXTRA,
                   "LADDER_EXTRA_FROM": cls.FROM}
        with tempfile.TemporaryDirectory() as folder, settings.overridden(changes):
            cls.runner = MnistLadderRunner(
                count=cls.TRAIN, evaluate_limit=cls.TEST, map_max_points=None, shifts=(),
                file_prefix=str(Path(folder) / "ladder"), keep_levels=True,
            )
            cls.runner.run()
            cls.ladder = TrainedLadder(cls.runner.groups, cls.runner.levels, cls.runner.level_peaks, decay=1e-7,
                                       primary=cls.runner.primary)
            cls.taught = {int(level.number): cls.runner.groups.taught_pictures(int(level.number)) for level in cls.runner.levels}

    def test_extra_group_is_in_the_run(self) -> None:
        ranges = self.runner.groups.ranges
        self.assertEqual(ranges["extra"], (self.TRAIN, self.TRAIN + self.EXTRA))
        self.assertEqual(ranges["test"][0], self.TRAIN + self.EXTRA)

    def test_memory_takes_extra_pictures_only_from_the_level(self) -> None:
        if len(self.runner.levels) < self.FROM:
            self.skipTest(f"прогон построил меньше {self.FROM} уровней")
        for level in self.runner.levels:
            with self.subTest(level=level.number):
                picture = np.asarray(level.point_picture)
                picture = picture[picture >= 0]
                self.assertTrue(picture.size)
                if level.number < self.FROM:
                    self.assertEqual(self.taught[int(level.number)], self.TRAIN)
                    self.assertTrue(np.all(picture < self.TRAIN))
                else:
                    self.assertEqual(self.taught[int(level.number)], self.TRAIN + self.EXTRA)
                    self.assertTrue(np.all(picture < self.TRAIN + self.EXTRA))  # проверочные в память не входят
                    self.assertTrue(np.any(picture >= self.TRAIN))  # добавочные — на карте

    def test_reader_learns_on_train_pictures_and_new_picture_passes(self) -> None:
        from torchvision.datasets import MNIST

        groups = self.runner.groups
        self.assertEqual(groups.ranges["train"], (0, self.TRAIN))
        first, _ = groups.ranges["test"]
        mnist = MNIST(root="./data", train=False, download=True)
        sight = self.ladder.see(mnist.data[int(groups.atoms.index[first])].numpy())
        self.assertLessEqual(len(sight.levels), len(self.runner.levels))
        self.assertEqual(sight.probabilities.shape[0], np.unique(groups.labels[: self.TRAIN]).size)


if __name__ == "__main__":
    unittest.main()
