"""Штрихи по середине линии (вне статьи; замысел автора 2026-11-17): середина вместо двух краёв, толщина не важна,
угол без стрелки, на перекрестии — два штриха разного направления."""

from __future__ import annotations

import os
import sys
import unittest


os.environ.setdefault("RERUN", "off")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "system"))

import numpy as np  # noqa: E402

from damp import settings  # noqa: E402
from damp.encoding.elements import StickCoder  # noqa: E402
from damp.stimuli.stroke_detectors import StrokeDetectors  # noqa: E402
from damp.stimuli.stroke_midline import StrokeMidline  # noqa: E402

SIDE = 224  # сторона рабочей картинки: 28 пикселей MNIST × 8


def _line(angle: float, half_width: float, *, centre: tuple[float, float] = (112.0, 112.0), half_length: float = 80.0) -> "np.ndarray":
    """Прямая линия заданной полутолщины на чёрном поле; angle — направление линии, градусы, ось Y вверх."""
    rows, cols = np.mgrid[0:SIDE, 0:SIDE].astype(np.float64)
    dx, dy = cols + 0.5 - centre[1], -(rows + 0.5 - centre[0])
    along_x, along_y = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    along, across = dx * along_x + dy * along_y, -dx * along_y + dy * along_x
    return ((np.abs(across) <= half_width) & (np.abs(along) <= half_length)).astype(np.float64)


def _midline(*, bridge: float = 0.0) -> StrokeMidline:
    """Середина линии; bridge > 0 — с соединением разрывов (мостики не длиннее bridge пикселей)."""
    return StrokeMidline(level=0.5, sigma=2.0, strength_share=0.3, blob=0.6, tip=0.5, weak_share=0.1, bridge=bridge)


def _pieces(y: "np.ndarray", x: "np.ndarray") -> int:
    """Сколько отдельных кусков у середины: точки не дальше двух пикселей — один кусок."""
    from scipy import ndimage

    raster = np.zeros((SIDE, SIDE), dtype=bool)
    raster[np.floor(y).astype(int), np.floor(x).astype(int)] = True
    return int(ndimage.label(ndimage.binary_dilation(raster, iterations=1), structure=np.ones((3, 3)))[1])


def _ring(centre: tuple[float, float], radius: float, half_width: float, *, cut: float = 0.0) -> "np.ndarray":
    """Кольцо заданной полутолщины; cut — вырез справа шириной во столько градусов (разомкнутое кольцо)."""
    rows, cols = np.mgrid[0:SIDE, 0:SIDE].astype(np.float64)
    dy, dx = rows + 0.5 - centre[0], cols + 0.5 - centre[1]
    ink = np.abs(np.hypot(dy, dx) - radius) <= half_width
    if cut > 0.0:
        ink &= np.abs(np.degrees(np.arctan2(-dy, dx))) > cut / 2.0
    return ink.astype(np.float64)


def _loops(y: "np.ndarray", x: "np.ndarray") -> int:
    """Сколько замкнутых петель у середины: области фона, со всех сторон окружённые линией (не мельче 500 пикселей)."""
    from scipy import ndimage

    raster = np.zeros((SIDE, SIDE), dtype=bool)
    raster[np.floor(y).astype(int), np.floor(x).astype(int)] = True
    raster = ndimage.binary_dilation(raster, structure=np.ones((3, 3)))
    holes, count = ndimage.label(~raster)
    sizes = ndimage.sum(~raster, holes, range(1, count + 1))
    return sum(1 for index, size in enumerate(sizes, start=1) if index != holes[0, 0] and size >= 500)


def _detectors() -> StrokeDetectors:
    return StrokeDetectors(step=6.0, similarity=0.7, angle_span=2.25, min_points=3, dense_share=0.8, smoothing=2,
                           grad_threshold=0.05, axial=True)


def _strokes(work: "np.ndarray") -> list:
    return _detectors().strokes_of(*_midline().points(work))


def _apart(first: float, second: float) -> float:
    """Разность направлений без стрелки, градусы, 0…90."""
    gap = abs(first - second) % 180.0
    return min(gap, 180.0 - gap)


class StrokeMidlineTest(unittest.TestCase):
    def test_points_lie_on_the_middle_of_the_line(self) -> None:
        """Середина линии — точки, равноудалённые от её краёв; направление точки — поперёк линии."""
        y, x, vx, vy = _midline().points(_line(0.0, 8.0))
        self.assertGreater(y.shape[0], 100)
        np.testing.assert_allclose(y, 112.0, atol=1.0)  # горизонтальная линия по строке 112
        self.assertGreater(x.max() - x.min(), 120.0)  # и вдоль почти всей её длины
        across = np.degrees(np.arctan2(vy, vx)) % 180.0
        # Поперёк горизонтальной линии — вертикаль; у самых концов линии направление отклоняется на несколько градусов.
        self.assertTrue(np.all(np.abs(across - 90.0) < 5.0))
        self.assertLess(float(np.median(np.abs(across - 90.0))), 0.5)

    def test_thickness_does_not_move_the_strokes(self) -> None:
        """Мозгу не важна толщина штриха: тонкая и жирная линия дают штрихи в тех же местах с тем же углом."""
        thin, thick = _strokes(_line(30.0, 4.0)), _strokes(_line(30.0, 14.0))
        self.assertGreaterEqual(len(thin), 15)
        # Жирная линия короче по середине на свою полутолщину с каждого конца: её штрихи — среди штрихов тонкой.
        self.assertLessEqual(len(thick), len(thin))
        self.assertGreaterEqual(len(thick), len(thin) - 6)
        for stroke in thick:
            nearest = min(thin, key=lambda other: np.hypot(other.x - stroke.x, other.y - stroke.y))
            self.assertLess(np.hypot(nearest.x - stroke.x, nearest.y - stroke.y), 6.0)  # в пределах шага штрихов
            self.assertLess(_apart(np.degrees(stroke.angle), np.degrees(nearest.angle)), 2.0)

    def test_angle_has_no_arrow(self) -> None:
        """Угол штриха середины — направление поперёк линии, 0…180°, с точностью 1–2° (требование автора)."""
        for direction in (0.0, 20.0, 45.0, 90.0, 135.0, 170.0):
            strokes = _strokes(_line(direction, 7.0))
            self.assertGreaterEqual(len(strokes), 15, f"линия {direction}°")
            for stroke in strokes:
                angle = float(np.degrees(stroke.angle))
                self.assertGreaterEqual(angle, 0.0)
                self.assertLess(angle, 180.0)
                self.assertLess(_apart(angle, direction + 90.0), 2.0, f"линия {direction}°")

    def test_strokes_stand_a_step_apart(self) -> None:
        """Центры штрихов одной линии — не ближе шага и без пропусков: от одного до двух шагов между соседями (§6.4.5)."""
        strokes = sorted(_strokes(_line(0.0, 6.0)), key=lambda stroke: stroke.x)
        gaps = np.diff([stroke.x for stroke in strokes])
        self.assertTrue(np.all(gaps >= 6.0 - 1e-3))
        # У конца линии центр крайнего штриха сдвинут к её середине: зазор чуть больше двух шагов.
        self.assertTrue(np.all(gaps <= 12.0 + 1.0))
        self.assertTrue(np.all(gaps[1:-1] <= 6.0 + 1.0))  # вдали от концов — ровно через шаг

    def test_crossing_keeps_both_lines(self) -> None:
        """Перекрестие: у места пересечения стоят штрихи обеих линий — им мешают только штрихи похожего направления."""
        cross = np.maximum(_line(0.0, 6.0), _line(90.0, 6.0))
        strokes = _strokes(cross)
        near = [stroke for stroke in strokes if np.hypot(stroke.x - 112.0, stroke.y - 112.0) < 30.0]
        horizontal = [stroke for stroke in near if _apart(np.degrees(stroke.angle), 90.0) < 10.0]
        vertical = [stroke for stroke in near if _apart(np.degrees(stroke.angle), 0.0) < 10.0]
        self.assertGreaterEqual(len(horizontal), 2)
        self.assertGreaterEqual(len(vertical), 2)

    def test_shift_of_the_picture_shifts_the_strokes(self) -> None:
        """Штрихи стоят в координатах самой линии: сдвиг картинки сдвигает их все вместе."""
        base = _strokes(_line(60.0, 7.0))
        moved = _strokes(_line(60.0, 7.0, centre=(112.0 + 8.0, 112.0 + 16.0)))
        self.assertEqual(len(base), len(moved))
        first = sorted((stroke.x, stroke.y, stroke.angle) for stroke in base)
        second = sorted((stroke.x - 16.0, stroke.y - 8.0, stroke.angle) for stroke in moved)
        np.testing.assert_allclose(np.asarray(first), np.asarray(second), atol=1e-6)

    def test_bridges_join_a_crossing_at_its_centre(self) -> None:
        """Перекрестие: гребень рвётся на четыре куска, мостики сводят их в общий центр — середина снова одна."""
        cross = np.maximum(_line(0.0, 7.0), _line(90.0, 7.0))
        torn = _midline().points(cross)
        self.assertGreaterEqual(_pieces(torn[0], torn[1]), 3)
        y, x, vx, vy = _midline(bridge=48.0).points(cross)
        self.assertEqual(_pieces(y, x), 1)
        added = y.shape[0] - torn[0].shape[0]
        self.assertGreater(added, 0)
        near = np.hypot(y - 112.0, x - 112.0) < 20.0  # новые точки — у центра перекрестия, в пределах толщины линии
        self.assertGreaterEqual(int(near.sum()), added)

    def test_bridges_join_a_side_branch(self) -> None:
        """Линия упирается в линию сбоку: ветка соединяется с основной линией в центре стыка."""
        branch = np.maximum(_line(0.0, 7.0), _line(90.0, 7.0, centre=(72.0, 112.0), half_length=40.0))
        self.assertGreaterEqual(_pieces(*_midline().points(branch)[:2]), 2)
        self.assertEqual(_pieces(*_midline(bridge=48.0).points(branch)[:2]), 1)

    def test_bridges_add_nothing_to_a_whole_line(self) -> None:
        """У целой линии соединять нечего: «веер» слабых точек у её концов в середину не идёт."""
        for direction in (0.0, 30.0, 90.0):
            line = _line(direction, 7.0)
            plain, bridged = _midline().points(line), _midline(bridge=48.0).points(line)
            self.assertEqual(_pieces(plain[0], plain[1]), 1)
            self.assertEqual(bridged[0].shape[0], plain[0].shape[0])

    def test_bridges_close_both_loops_of_an_eight(self) -> None:
        """Разрыв внутри петли: оба его конца — в одном куске (петля замкнута с другой стороны), и всё же он соединяется."""
        for upper, lower in (((62.0, 112.0), 30.0), ((134.0, 112.0), 40.0)), (((72.0, 112.0), 38.0), ((152.0, 112.0), 38.0)):
            eight = np.maximum(_ring(upper[0], upper[1], 8.0), _ring(lower[0], lower[1], 8.0))
            self.assertLess(_loops(*_midline().points(eight)[:2]), 2)
            y, x, vx, vy = _midline(bridge=48.0).points(eight)
            self.assertEqual(_loops(y, x), 2)
            self.assertTrue(np.all(np.isfinite(vx)) and np.all(np.hypot(vx, vy) > 0.0))

    def test_bridges_leave_an_open_ring_open(self) -> None:
        """Между концами линии нет чернил — они не соединяются: разомкнутое кольцо остаётся разомкнутым."""
        for cut in (15.0, 40.0):
            ring = _ring((112.0, 112.0), 60.0, 8.0, cut=cut)
            plain, bridged = _midline().points(ring), _midline(bridge=48.0).points(ring)
            self.assertEqual(bridged[0].shape[0], plain[0].shape[0])
            self.assertEqual(_loops(bridged[0], bridged[1]), 0)
        whole = _midline(bridge=48.0).points(_ring((112.0, 112.0), 60.0, 8.0))
        self.assertEqual(_loops(whole[0], whole[1]), 1)

    def test_even_width_line_keeps_one_point_across(self) -> None:
        """Линия чётной ширины: вершина расстояния — между двумя пикселями; из двух равных берётся один, без разрывов."""
        y, x, _, _ = _midline().points(_line(0.0, 8.0))  # строки 104…119: вершина между строками 111 и 112
        columns = np.floor(x).astype(int)
        inner = (columns >= 50) & (columns < 174)
        self.assertEqual(np.unique(columns[inner]).shape[0], 124)  # в каждом столбце есть точка…
        self.assertEqual(int(inner.sum()), 124)  # …и ровно одна
        self.assertEqual(np.unique(np.floor(y[inner]).astype(int)).shape[0], 1)  # все — в одной строке

    def test_every_point_has_a_direction(self) -> None:
        """У каждой точки середины, и у точек мостиков тоже, есть направление: без него цифра осталась бы без штрихов."""
        cross = np.maximum(_line(20.0, 9.0), _line(110.0, 5.0))
        y, x, vx, vy = _midline(bridge=48.0).points(cross)
        self.assertTrue(np.all(np.isfinite(vx)) and np.all(np.isfinite(vy)))
        self.assertTrue(np.all(np.hypot(vx, vy) > 0.0))
        strokes = _detectors().strokes_of(y, x, vx, vy)
        self.assertGreaterEqual(len(strokes), 20)
        # У центра перекрестия стоят штрихи обеих линий.
        near = [stroke for stroke in strokes if np.hypot(stroke.x - 112.0, stroke.y - 112.0) < 14.0]
        self.assertTrue(any(_apart(np.degrees(stroke.angle), 110.0) < 15.0 for stroke in near))
        self.assertTrue(any(_apart(np.degrees(stroke.angle), 200.0) < 15.0 for stroke in near))

    def test_empty_picture_has_no_points(self) -> None:
        y, x, vx, vy = _midline().points(np.zeros((SIDE, SIDE)))
        self.assertEqual(y.shape[0], 0)
        self.assertEqual(_detectors().strokes_of(y, x, vx, vy), [])


class AxialAngleCodeTest(unittest.TestCase):
    """Круг кода угла у штрихов середины — 180°: 0° и 180° — одна линия."""

    @staticmethod
    def _code(coder: StickCoder, angle: float) -> "np.ndarray":
        return np.asarray(coder.encoder.encode_matrix(np.asarray([[angle]], dtype=np.float64)))[0].astype(bool)

    @staticmethod
    def _alike(first: "np.ndarray", second: "np.ndarray") -> float:
        return float((first & second).sum()) / float(np.sqrt(first.sum() * second.sum()))

    def test_period_is_half_a_circle_for_midline_strokes(self) -> None:
        with settings.overridden({"MNIST_STROKE_DETECTORS": True, "MNIST_STROKE_MIDLINE": True}):
            coder = StickCoder()
            near = self._alike(self._code(coder, 1.0), self._code(coder, 179.0))  # 2° друг от друга через 0°
            far = self._alike(self._code(coder, 1.0), self._code(coder, 91.0))  # перпендикулярные линии
        # Широкие слои детекторов угла общие у далёких углов: сходство далёких направлений — около 0.25, не ноль.
        self.assertGreater(near, 0.6)
        self.assertLess(far, 0.35)

    def test_period_is_a_full_circle_for_edge_strokes(self) -> None:
        with settings.overridden({"MNIST_STROKE_DETECTORS": True, "MNIST_STROKE_MIDLINE": False}):
            coder = StickCoder()
            opposite = self._alike(self._code(coder, 1.0), self._code(coder, 181.0))  # встречные края линии
        self.assertLess(opposite, 0.35)


if __name__ == "__main__":
    unittest.main()
