"""Уровень 0 обученного цикла (system/damp/cycle/primary.py): первичная карта углопозиций — раскладка кодов «угол и
место» (§4.6, §5) — и её отклик на штрихи картинки (§6.1, §6.5)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path


os.environ.setdefault("RERUN", "off")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "system"))

import numpy as np  # noqa: E402

from damp import settings  # noqa: E402


class PrimaryEncoderTest(unittest.TestCase):
    """Первичный код углопозиции: угол и место (§4.4, §4.6); у штрихов без стрелки период угла — 180°."""

    def _encoder(self, midline: bool):  # noqa: ANN202
        from damp.cycle.primary import PrimaryLevel

        with settings.overridden({"MNIST_STROKE_DETECTORS": True, "MNIST_STROKE_MIDLINE": midline}):
            return PrimaryLevel, PrimaryLevel.build_encoder()

    @staticmethod
    def _alike(first: "np.ndarray", second: "np.ndarray") -> float:
        a, b = np.unpackbits(first), np.unpackbits(second)
        return float((a & b).sum()) / max(1.0, float((a | b).sum()))

    def test_midline_angle_repeats_after_half_a_circle(self) -> None:
        level, encoder = self._encoder(True)
        codes = level.codes(encoder, np.array([1.0, 179.0, 91.0]), np.array([10.0, 10.0, 10.0]), np.array([12.0, 12.0, 12.0]))
        self.assertGreater(self._alike(codes[0], codes[1]), self._alike(codes[0], codes[2]))  # 1° и 179° — почти одна линия
        level, encoder = self._encoder(False)
        codes = level.codes(encoder, np.array([1.0, 179.0, 359.0]), np.array([10.0, 10.0, 10.0]), np.array([12.0, 12.0, 12.0]))
        self.assertGreater(self._alike(codes[0], codes[2]), self._alike(codes[0], codes[1]))  # у края 1° близок к 359°, не к 179°

    def test_code_keeps_both_angle_and_place(self) -> None:
        """Сходство первичных кодов убывает и с разницей угла, и с расстоянием на картинке (§4.5, §4.6)."""
        level, encoder = self._encoder(True)
        codes = level.codes(encoder, np.array([40.0, 40.0, 40.0, 100.0]), np.array([10.0, 10.5, 20.0, 10.0]), np.array([12.0, 12.5, 4.0, 12.0]))
        near, far, turned = self._alike(codes[0], codes[1]), self._alike(codes[0], codes[2]), self._alike(codes[0], codes[3])
        self.assertGreater(near, far)
        self.assertGreater(near, turned)

    def test_no_sticks_no_codes(self) -> None:
        level, encoder = self._encoder(True)
        empty = level.codes(encoder, np.zeros(0), np.zeros(0), np.zeros(0))
        self.assertEqual(empty.shape, (0, -(-encoder.code_length // 8)))


class PrimaryLevelTest(unittest.TestCase):
    """Первичная карта маленького цикла: точки — углопозиции обучающих цифр, соседи похожи углом и местом."""

    TRAIN, LIMIT = 30, 600

    @classmethod
    def setUpClass(cls) -> None:
        from torchvision.datasets import MNIST

        from damp.cycle.ladder import MnistLadderRunner
        from damp.cycle.primary import PrimaryLevel
        from damp.cycle.trained import TrainedLadder

        changes = {"LADDER_SAVE_MAPS": False, "LADDER_TRAIN_VIEWS": ()}
        with tempfile.TemporaryDirectory() as folder, settings.overridden(changes):
            cls.runner = MnistLadderRunner(
                count=cls.TRAIN, evaluate_limit=0, map_max_points=None, shifts=(), file_prefix=str(Path(folder) / "ladder"), keep_levels=True,
            )
            cls.runner.run()
            cls.primary = PrimaryLevel.build(cls.runner.groups, max_points=cls.LIMIT)
            cls.ladder = TrainedLadder(cls.runner.groups, cls.runner.levels, cls.runner.level_peaks, decay=1e-7, primary=cls.primary)
            mnist = MNIST(root="./data", train=False, download=True)
            cls.sight = cls.ladder.see(mnist.data[3].numpy())
            cls.empty = cls.ladder.see(np.zeros((28, 28), dtype=np.uint8))

    def test_points_are_sticks_of_training_pictures(self) -> None:
        primary, atoms = self.primary, self.runner.groups.atoms
        points = int(primary.memory.points)
        self.assertGreater(points, 100)
        self.assertLessEqual(points, self.LIMIT)
        for values in (primary.point_picture, primary.point_angle, primary.point_y, primary.point_x):
            self.assertEqual(values.shape, (points,))
        self.assertTrue(np.all((primary.point_picture >= 0) & (primary.point_picture < self.runner.groups.train_pictures)))
        # Угол и место точки — те, что у углопозиции её обучающей картинки.
        known = {(int(p), round(float(a), 4), round(float(y), 4), round(float(x), 4)) for p, a, y, x in zip(atoms.picture, atoms.angle, atoms.y, atoms.x)}
        for index in range(0, points, max(1, points // 50)):
            key = (int(primary.point_picture[index]), round(float(primary.point_angle[index]), 4), round(float(primary.point_y[index]), 4),
                   round(float(primary.point_x[index]), 4))
            self.assertIn(key, known)

    def test_every_point_has_its_own_cell(self) -> None:
        layout = self.primary.memory.layout
        cells = np.asarray(layout.cells_array())
        self.assertEqual(cells.shape, (int(self.primary.memory.points), 2))
        self.assertEqual(np.unique(cells[:, 0] * int(layout.width) + cells[:, 1]).shape[0], cells.shape[0])

    def test_neighbours_on_the_map_are_alike_in_angle_and_place(self) -> None:
        """Раскладка первичных кодов (§5): соседние точки карты ближе по углу и по месту, чем случайные — пинвилы."""
        primary = self.primary
        layout = primary.memory.layout
        cells = np.asarray(layout.cells_array())
        grid = -np.ones((int(layout.height), int(layout.width)), dtype=np.int64)
        grid[cells[:, 0], cells[:, 1]] = np.arange(cells.shape[0])
        first = np.concatenate([grid[:, :-1].ravel(), grid[:-1, :].ravel()])
        second = np.concatenate([grid[:, 1:].ravel(), grid[1:, :].ravel()])
        keep = (first >= 0) & (second >= 0)
        first, second = first[keep], second[keep]
        period = 180.0 if primary.axial() else 360.0

        def angle_gap(a: "np.ndarray", b: "np.ndarray") -> float:
            gap = np.abs(primary.point_angle[a] - primary.point_angle[b]) % period
            return float(np.minimum(gap, period - gap).mean())

        def place_gap(a: "np.ndarray", b: "np.ndarray") -> float:
            return float(np.hypot(primary.point_y[a] - primary.point_y[b], primary.point_x[a] - primary.point_x[b]).mean())

        rng = np.random.default_rng(0)
        any_first, any_second = rng.integers(0, cells.shape[0], 5000), rng.integers(0, cells.shape[0], 5000)
        self.assertLess(angle_gap(first, second), 0.5 * angle_gap(any_first, any_second))
        self.assertLess(place_gap(first, second), 0.5 * place_gap(any_first, any_second))

    def test_picture_lights_points_like_its_sticks(self) -> None:
        """Штрихи картинки зажигают точки первичной карты (§6.1): самая активная точка штриха близка ему углом и местом."""
        primary, sight = self.primary, self.sight
        found = sight.primary
        sticks, points = sight.stick_angle.shape[0], int(primary.memory.points)
        self.assertGreater(sticks, 10)
        self.assertTrue(np.all((found.lit_stick >= 0) & (found.lit_stick < sticks)))
        self.assertTrue(np.all((found.lit_point >= 0) & (found.lit_point < points)))
        self.assertTrue(np.all(found.lit_value > 0))
        self.assertTrue(np.all((found.fired_detector >= 0) & (found.fired_detector < int(primary.detectors.built))))
        self.assertTrue(set(np.unique(found.fired_stick).tolist()) <= set(np.unique(found.lit_stick).tolist()))
        order = np.lexsort((-found.lit_value, found.lit_stick))
        firsts = np.flatnonzero(np.r_[True, found.lit_stick[order][1:] != found.lit_stick[order][:-1]])
        stick, point = found.lit_stick[order][firsts], found.lit_point[order][firsts]
        period = 180.0 if primary.axial() else 360.0
        gap = np.abs(sight.stick_angle[stick] - primary.point_angle[point]) % period
        gap = np.minimum(gap, period - gap)
        away = np.hypot(sight.stick_y[stick] - primary.point_y[point], sight.stick_x[stick] - primary.point_x[point])
        self.assertLess(float(np.median(gap)), 20.0)
        self.assertLess(float(np.median(away)), 4.0)

    def test_empty_picture_lights_nothing(self) -> None:
        found = self.empty.primary
        self.assertTrue(found is None or (found.lit_point.shape[0] == 0 and found.fired_detector.shape[0] == 0))

    def test_ladder_without_primary_map_still_sees(self) -> None:
        from damp.cycle.trained import TrainedLadder

        plain = TrainedLadder(self.runner.groups, self.runner.levels, self.runner.level_peaks, decay=1e-7)
        from torchvision.datasets import MNIST

        sight = plain.see(MNIST(root="./data", train=False, download=True).data[3].numpy())
        self.assertIsNone(sight.primary)
        self.assertEqual(sight.answer, self.sight.answer)
        self.assertEqual(sight.verdict.digit, self.sight.verdict.digit)


if __name__ == "__main__":
    unittest.main()
