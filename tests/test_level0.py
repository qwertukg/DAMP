"""Уровень 0 цикла — первичная раскладка штрихов (system/damp/cycle/primary.py, StrokeLevel): окно на каждом штрихе.

LADDER_LEVEL0_PLACE = "stroke" — общий подход (автор 2026-11-19): в окне точки поля штриха (§6.4.3), центр — сам
штрих, место точки — от центра штриха кругами места (П.4). "digit" — вариант 3 (2026-11-18): в окне один штрих,
место — от центра цифры."""

from __future__ import annotations

import unittest

import numpy as np

from damp.cycle.plane_fields import PlaneFieldBuilder
from damp.cycle.primary import StrokeLevel, StrokePoints
from damp.encoding.elements import LadderElements
from damp.encoding.keyhole_code import KeyholeCoder


def strokes(picture, y, x, angle_code) -> LadderElements:  # noqa: ANN001
    """Штрихи с кодами наклона angle_code (номер кода у каждого штриха) — коды из восьми бит по одному на наклон."""
    what = np.packbits(np.eye(8, dtype=np.uint8), axis=1)
    return LadderElements(
        picture=np.asarray(picture, dtype=np.int64), y=np.asarray(y, dtype=np.float64), x=np.asarray(x, dtype=np.float64),
        what_packed=what, what_length=8, what_index=np.asarray(angle_code, dtype=np.int64),
        strength=np.ones(len(picture), dtype=np.float64),
    )


class StrokeLevelPiecesTest(unittest.TestCase):
    """Окна уровня 0: по окну на штрих, центр — середина штрихов картинки, место — доля радиуса до дальнего штриха."""

    def setUp(self) -> None:
        self.sticks = strokes([0, 0, 0, 0, 1, 1], [2, 2, 6, 6, 10, 14], [2, 6, 2, 6, 3, 3], [0, 1, 2, 3, 4, 5])
        self.pieces = StrokeLevel.pieces(self.sticks)

    def test_one_keyhole_per_stroke_with_the_stroke_alone(self) -> None:
        self.assertEqual(self.pieces.count, self.sticks.count)
        self.assertTrue(np.array_equal(self.pieces.piece, np.arange(self.sticks.count)))
        self.assertTrue(np.array_equal(self.pieces.element, np.arange(self.sticks.count)))
        self.assertTrue(np.array_equal(self.pieces.picture, self.sticks.picture))

    def test_centre_is_the_middle_of_the_picture_strokes(self) -> None:
        self.assertTrue(np.allclose(self.pieces.centre[:4], [[4.0, 4.0]] * 4))
        self.assertTrue(np.allclose(self.pieces.centre[4:], [[12.0, 3.0]] * 2))

    def test_place_is_a_share_of_the_farthest_stroke(self) -> None:
        shift = self.pieces.offsets[self.pieces.offset]
        step = PlaneFieldBuilder.OFFSET_STEP
        half = np.round(np.sqrt(0.5) / step) * step  # (±2, ±2) от центра при радиусе 2√2
        self.assertTrue(np.allclose(np.abs(shift[:4]), half))
        self.assertTrue(np.allclose(shift[4:], [[-1.0, 0.0], [1.0, 0.0]]))
        self.assertLessEqual(float(np.abs(shift).max()), 1.0)

    def test_moving_the_digit_does_not_change_the_codes(self) -> None:
        """Место — от центра цифры: сдвиг всей цифры коды штрихов не меняет (в отличие от первичной карты «угол + X + Y»)."""
        moved = strokes([0, 0, 0, 0], [5, 5, 9, 9], [7, 11, 7, 11], [0, 1, 2, 3])
        first = self.sticks.only(self.sticks.picture == 0)
        coder = KeyholeCoder()
        codes = [coder.encode(sticks, StrokeLevel.pieces(sticks), number=0, unit=0.5, taught=4, content_length=64).packed
                 for sticks in (first, moved)]
        self.assertTrue(np.array_equal(codes[0], codes[1]))

    def test_learned_strokes_keep_place_and_strength(self) -> None:
        details = np.packbits(np.eye(16, dtype=np.uint8)[: self.sticks.count], axis=1)
        learned = StrokeLevel.learned(self.sticks, details, 16)
        self.assertTrue(np.array_equal(learned.y, self.sticks.y))
        self.assertTrue(np.array_equal(learned.x, self.sticks.x))
        self.assertTrue(np.array_equal(learned.power(), self.sticks.power()))
        self.assertTrue(np.array_equal(learned.codes(), details))
        self.assertEqual(learned.what_length, 16)


def stroke_points(stroke, picture, y, x, angle) -> StrokePoints:  # noqa: ANN001
    """Точки штрихов: stroke — номер штриха каждой точки (по возрастанию), остальное — сама точка."""
    count = len(picture)
    return StrokePoints(
        picture=np.asarray(picture, dtype=np.int64), y=np.asarray(y, dtype=np.float64), x=np.asarray(x, dtype=np.float64),
        angle=np.asarray(angle, dtype=np.float64), strength=np.ones(count, dtype=np.float64),
        entry_stroke=np.asarray(stroke, dtype=np.int64), entry_point=np.arange(count, dtype=np.int64),
    )


class StrokePointPiecesTest(unittest.TestCase):
    """Уровень 0 по общему подходу: окно на штрихе, в нём точки его поля, центр — сам штрих, место — доля R (П.4)."""

    def setUp(self) -> None:
        # Два штриха на картинке 0 и один на картинке 1; у каждого — точки его поля.
        self.sticks = strokes([0, 0, 1], [2.0, 2.0, 5.0], [2.0, 6.0, 3.0], [0, 1, 2])
        self.points = stroke_points(
            [0, 0, 0, 1, 1, 1, 1, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [2.0, 2.0, 2.0, 2.0, 2.4, 1.6, 2.0, 5.0, 5.0, 5.0],
            [1.6, 2.0, 2.4, 5.6, 6.0, 6.0, 6.4, 2.7, 3.0, 3.3],
            [90.0, 90.0, 90.0, 0.0, 0.0, 0.0, 0.0, 90.0, 90.0, 90.0],
        )
        self.pieces = StrokeLevel.point_pieces(self.sticks, self.points)

    def test_one_keyhole_per_stroke_with_its_field_points(self) -> None:
        self.assertEqual(self.pieces.count, self.sticks.count)
        self.assertTrue(np.array_equal(self.pieces.piece, self.points.entry_stroke))
        self.assertTrue(np.array_equal(self.pieces.element, self.points.entry_point))
        self.assertTrue(np.array_equal(self.pieces.picture, self.sticks.picture))
        self.assertTrue(np.allclose(self.pieces.centre, np.stack([self.sticks.y, self.sticks.x], axis=1)))

    def test_place_is_a_share_of_the_farthest_field_point(self) -> None:
        shift = self.pieces.offsets[self.pieces.offset]
        self.assertTrue(np.allclose(shift[:3], [[0.0, -1.0], [0.0, 0.0], [0.0, 1.0]]))  # точки вдоль штриха, R = 0.4
        self.assertTrue(np.allclose(np.abs(shift).max(axis=1)[[0, 2, 3, 4, 5, 6]], 1.0))
        self.assertLessEqual(float(np.abs(shift).max()), 1.0)

    def test_moving_the_digit_does_not_change_the_codes(self) -> None:
        """Место точки — от центра своего штриха: сдвиг всей цифры коды штрихов не меняет."""
        coder = KeyholeCoder()

        def codes(dy: float, dx: float) -> np.ndarray:
            sticks = strokes([0, 0, 1], self.sticks.y + dy, self.sticks.x + dx, [0, 1, 2])
            points = stroke_points(self.points.entry_stroke, self.points.picture, self.points.y + dy, self.points.x + dx,
                                   self.points.angle)
            elements = LadderElements(
                picture=points.picture, y=points.y, x=points.x, what_packed=np.packbits(np.eye(8, dtype=np.uint8), axis=1),
                what_length=8, what_index=(points.angle // 90).astype(np.int64), strength=points.strength,
            )
            return coder.encode(elements, StrokeLevel.point_pieces(sticks, points), number=0, unit=0.5, taught=3,
                                content_length=64).packed

        self.assertTrue(np.array_equal(codes(0.0, 0.0), codes(3.0, -2.0)))

    def test_shape_inside_the_stroke_changes_the_code(self) -> None:
        """Прямой штрих и уголок с теми же направлениями точек — разные коды: код знает форму штриха внутри поля."""
        coder = KeyholeCoder()
        sticks = strokes([0, 1], [2.0, 2.0], [2.0, 2.0], [0, 0])
        straight = [(2.0, 1.6), (2.0, 2.0), (2.0, 2.4)]
        corner = [(2.4, 2.0), (2.0, 2.0), (2.0, 2.4)]
        places = straight + corner
        points = stroke_points([0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [p[0] for p in places], [p[1] for p in places], [90.0] * 6)
        elements = LadderElements(
            picture=points.picture, y=points.y, x=points.x, what_packed=np.packbits(np.eye(8, dtype=np.uint8), axis=1),
            what_length=8, what_index=np.zeros(6, dtype=np.int64), strength=points.strength,
        )
        codes = coder.encode(elements, StrokeLevel.point_pieces(sticks, points), number=0, unit=0.5, taught=2, content_length=64).packed
        self.assertFalse(np.array_equal(codes[0], codes[1]))


class StrokePointsOfPictureTest(unittest.TestCase):
    """Точки штрихов настоящей цифры: те же штрихи, что у разбора углопозиций, у каждого — точки его поля (§6.4.3)."""

    def test_points_belong_to_the_same_strokes(self) -> None:
        from damp import settings
        from damp.stimuli.digits import DigitAtoms, MnistDigits
        from damp.encoding.elements import StickCoder

        if not (settings.MNIST_CONTOUR_ONLY and settings.MNIST_STROKE_DETECTORS):
            self.skipTest("точки штрихов есть только у штрихов-детекторов")
        digits = MnistDigits()
        pixels = np.zeros((28, 28), dtype=np.float32)
        pixels[6:22, 12:15] = 1.0  # вертикальная палочка
        pixels[6:9, 8:20] = 1.0  # и перекладина сверху
        values = digits.sticks_of(pixels)
        atoms = DigitAtoms.from_rows([np.column_stack([np.zeros(values.shape[0]), values])], [0],
                                     [np.clip(np.rint(pixels * 255.0), 0, 255).astype(np.uint8)], [0])
        sticks = StickCoder().encode(atoms)
        points = StrokePoints.of_pixels(pixels, sticks)
        self.assertGreater(sticks.count, 3)
        per_stroke = np.bincount(points.entry_stroke, minlength=sticks.count)
        self.assertTrue(np.all(per_stroke >= int(settings.MNIST_STROKE_MIN_POINTS)))
        # Точка поля — не дальше шага от точки-центра детектора, а центр — их взвешенное среднее: от центра — меньше двух
        # шагов. Места — в клетках; клетка — MNIST_PATCH_SIZE пикселей рабочей картинки, шаг — в тех же пикселях.
        step = float(settings.MNIST_STROKE_STEP) / float(settings.MNIST_PATCH_SIZE)
        reach = np.hypot(points.y[points.entry_point] - sticks.y[points.entry_stroke],
                         points.x[points.entry_point] - sticks.x[points.entry_stroke])
        self.assertLess(float(reach.max()), 2.0 * step)


if __name__ == "__main__":
    unittest.main()
