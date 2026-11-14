"""Рисунки проекта: картинка детекторов, наблюдатель прогона цикла, приёмы общего стиля."""
import os
import unittest

os.environ.setdefault("RERUN", "off")

import numpy as np

from damp.encoding.damp_encoder import (
    ClosedDimension,
    Encoder,
    PlaneDimension,
    geometric_detector_layers,
    plane_detector_layers,
)
from damp.encoding.detector_view import DetectorLayerView
from damp.figures.recorder import LadderRecorder
from damp.figures.style import bits_raster, crop_cells, plural, wrap
from damp.cycle.ladder_trace import LadderObserver
from damp.figures.style import figure_style


class DetectorViewTest(unittest.TestCase):
    """§4.3, §4.4, §2.2.1: кольца угла, круги места и код стимула."""

    def setUp(self) -> None:
        self.style = figure_style()

    def view(self, encoder: Encoder) -> DetectorLayerView:
        return DetectorLayerView(
            encoder, style=self.style, min_hue=0.0, max_hue=300.0, inactive_alpha=0.3, panel=120, code_columns=16, code_cell=6
        )

    def test_ring_marks_only_active_arcs(self) -> None:
        layers = geometric_detector_layers(layers_count=3, widest_detectors=2, narrowest_detectors=12, overlap=0.5)
        encoder = Encoder(ClosedDimension("Angle", (0.0, 360.0), layers, code_length=48), random_seed=0)
        view = self.view(encoder)
        picture = view.ring(0, 90.0)
        self.assertEqual(picture.size, (120, 120))
        _, stats = view.code((90.0,))
        active = sum(1 for w in encoder.detectors if w.is_active(90.0))
        self.assertEqual(stats["by_dimension"]["Angle"], len(set(encoder.dimension_bits(0, 90.0))))
        self.assertLessEqual(stats["ones"], active, "биты — по одному на сработавший детектор, совпадения допускаются")

    def test_place_bits_are_or_of_circles_hit_by_points(self) -> None:
        """Биты места набора точек — OR битов кругов, в которые попала хоть одна точка (§2.2.1, §4.3)."""
        layers = plane_detector_layers(layers_count=2, widest_radius=0.25, narrowest_radius=0.1, overlap=0.4)
        encoder = Encoder(PlaneDimension("Place", ((-4.0, 4.0), (-4.0, 4.0)), layers, code_length=64, layer_bands=False), random_seed=0)
        view = self.view(encoder)
        points = np.asarray([[0.0, 0.0], [1.5, -2.0], [-3.0, 3.0]])
        picture = view.plane(0, 1, points)
        self.assertEqual(picture.size, (120, 120))
        _, stats = view.code([points])
        expected = set()
        for point in points:
            expected.update(encoder.dimension_bits(0, point))
        self.assertEqual(stats["ones"], len(expected))


class RecorderTest(unittest.TestCase):
    def test_default_observer_does_nothing(self) -> None:
        observer = LadderObserver()
        self.assertIsNone(observer.start(None))
        self.assertIsNone(observer.level(None))
        self.assertIsNone(observer.top(None))

    def test_recorder_keeps_levels_and_finds_them_by_number(self) -> None:
        class Level:
            def __init__(self, number: int) -> None:
                self.number = number
                self.keyhole_picture = np.zeros(3)
                self.map_cells = np.zeros((2, 2))
                self.detectors = np.zeros((4, 5))
                self.fired_keyhole = np.zeros(5)

        recorder = LadderRecorder()
        recorder.start(object())
        recorder.level(Level(1))
        recorder.level(Level(2))
        self.assertEqual(recorder.level_trace(2).number, 2)
        with self.assertRaises(KeyError):
            recorder.level_trace(3)


class StyleTest(unittest.TestCase):
    def test_plural_forms(self) -> None:
        self.assertEqual([plural(n, "деталь", "детали", "деталей") for n in (1, 2, 5, 11, 21, 22, 112)],
                         ["деталь", "детали", "деталей", "деталей", "деталь", "детали", "деталей"])

    def test_bits_raster_size_and_colours(self) -> None:
        picture = bits_raster(np.asarray([1, 0, 1, 1, 0]), 2, 5, (255, 0, 0), (10, 10, 10))
        self.assertEqual(picture.size, (10, 15))
        self.assertEqual(picture.getpixel((1, 1)), (255, 0, 0))
        self.assertEqual(picture.getpixel((6, 1)), (10, 10, 10))

    def test_crop_cells_keeps_cell_grid(self) -> None:
        from PIL import Image

        image = Image.new("RGB", (100, 130))
        crop = crop_cells(image, 40, 20, tile=10, top_skip=30)
        self.assertEqual(crop.size, (40, 20))

    def test_wrap_keeps_long_word_on_its_own_line(self) -> None:
        from PIL import Image, ImageDraw

        draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
        font = figure_style().font(14)
        lines = wrap(draw, "одно очень-очень-длинное-слово ещё", 40, font)
        self.assertGreaterEqual(len(lines), 2)
        self.assertTrue(all(line for line in lines))


if __name__ == "__main__":
    unittest.main()
