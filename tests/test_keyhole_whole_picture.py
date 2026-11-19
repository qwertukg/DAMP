"""Окна картинки без кластера (system/damp/cycle/plane_fields.py; верх цикла, §6.6): когда деталей на картинке меньше,
чем нужно кластеру (§6.4.3), окно каждого её элемента — вся картинка: детали не теряются, а объединяются дальше."""

from __future__ import annotations

import os
import sys
import unittest


os.environ.setdefault("RERUN", "off")
os.environ.setdefault("DAMP_LOG_FILE", "off")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "system"))

import numpy as np  # noqa: E402

from damp.cycle.plane_fields import PlaneFieldBuilder  # noqa: E402

LENGTH = 32


def _builder(*, whole: bool, min_points: int = 8, dense: bool = True, concepts=None, floor: float = 1.5) -> PlaneFieldBuilder:
    return PlaneFieldBuilder(
        lambda_thresholds=(0.9, 0.5), activation_radius=6.0, eps=1.5, min_samples=3, min_points=min_points,
        radius_min=1.0, energy_radius=3.0, energy_threshold=0.12, dense=dense, concepts=concepts, radius_floor=floor,
        whole_picture=whole,
    )


def _alike(count: int) -> "np.ndarray":
    bits = np.zeros((count, LENGTH), dtype=np.float32)
    bits[:, :8] = 1
    return bits


def _different(count: int) -> "np.ndarray":
    bits = np.zeros((count, LENGTH), dtype=np.float32)
    for row in range(count):
        bits[row, 3 * row:3 * row + 3] = 1  # до десяти разных кодов по три бита в 32 битах
    return bits


class WholePictureKeyholesTest(unittest.TestCase):
    def test_small_picture_gets_a_keyhole_on_every_element(self) -> None:
        positions = np.asarray([[0.0, 0.0], [0.0, 3.0], [4.0, 0.0]])
        fields = _builder(whole=True).picture_fields(positions, _different(3))
        self.assertEqual([(y, x) for y, x, _, _ in fields], [(0.0, 0.0), (0.0, 3.0), (4.0, 0.0)])
        # Радиус — до самого дальнего элемента картинки: окно видит её целиком.
        self.assertEqual([radius for _, _, radius, _ in fields], [4.0, 5.0, 5.0])
        self.assertEqual([points for _, _, _, points in fields], [3, 3, 3])

    def test_without_the_rule_a_small_picture_has_no_keyholes(self) -> None:
        positions = np.asarray([[0.0, 0.0], [0.0, 3.0], [4.0, 0.0]])
        self.assertEqual(_builder(whole=False).picture_fields(positions, _different(3)), [])

    def test_single_element_sees_itself_with_the_floor_radius(self) -> None:
        fields = _builder(whole=True, floor=2.5).picture_fields(np.asarray([[7.0, 9.0]]), _alike(1))
        self.assertEqual(fields, [(7.0, 9.0, 2.5, 1)])

    def test_element_with_an_empty_code_gets_no_keyhole(self) -> None:
        # Деталь, на окне которой не сработал ни один детектор (пустой код), молчит: окна на ней нет, но в окна
        # соседних деталей она входит как место.
        positions = np.asarray([[0.0, 0.0], [0.0, 3.0], [4.0, 0.0]])
        bits = _different(3)
        bits[1] = 0
        fields = _builder(whole=True).picture_fields(positions, bits)
        self.assertEqual([(y, x) for y, x, _, _ in fields], [(0.0, 0.0), (4.0, 0.0)])
        self.assertEqual([points for _, _, _, points in fields], [3, 3])
        bits[:] = 0
        self.assertEqual(_builder(whole=True).picture_fields(positions, bits), [])

    def test_picture_with_clusters_is_not_touched(self) -> None:
        # Девять похожих соседей в ряд: окна — по кластеру (§6.4.3), с правилом и без него одни и те же.
        positions = np.stack([np.zeros(9), np.arange(9, dtype=np.float64)], axis=1)
        with_rule = _builder(whole=True, min_points=5).picture_fields(positions, _alike(9))
        without = _builder(whole=False, min_points=5).picture_fields(positions, _alike(9))
        self.assertTrue(without)
        self.assertEqual(with_rule, without)

    def test_picture_with_enough_elements_but_no_cluster_is_seen_whole(self) -> None:
        # Девять элементов далеко друг от друга: кластера нет ни у одного — окно каждого видит всю картинку.
        positions = np.stack([np.zeros(9), 10.0 * np.arange(9, dtype=np.float64)], axis=1)
        self.assertEqual(_builder(whole=False).picture_fields(positions, _different(9)), [])
        fields = _builder(whole=True).picture_fields(positions, _different(9))
        self.assertEqual(len(fields), 9)
        self.assertEqual({points for _, _, _, points in fields}, {9})

    def test_rule_is_for_dense_keyholes_only(self) -> None:
        positions = np.asarray([[0.0, 0.0], [0.0, 3.0], [4.0, 0.0]])
        self.assertEqual(_builder(whole=True, dense=False).picture_fields(positions, _different(3)), [])

    def test_build_keeps_every_element_of_small_pictures(self) -> None:
        # Две картинки: в первой три детали, во второй одна. Каждая деталь получает окно, в окне — все детали картинки.
        picture = np.asarray([0, 0, 0, 1])
        y = np.asarray([0.0, 0.0, 4.0, 2.0])
        x = np.asarray([0.0, 3.0, 0.0, 2.0])
        packed = np.packbits(_different(4).astype(np.uint8), axis=1)
        built = _builder(whole=True).build(picture, y, x, packed, np.arange(4), LENGTH)
        pieces = built.pieces
        self.assertEqual(pieces.count, 4)
        self.assertEqual(pieces.picture.tolist(), [0, 0, 0, 1])
        self.assertEqual(np.bincount(pieces.piece, minlength=4).tolist(), [3, 3, 3, 1])
        self.assertEqual(built.covered, 1.0)
        # Элементы окна — только своей картинки.
        self.assertTrue(np.array_equal(picture[pieces.element], pieces.picture[pieces.piece]))


if __name__ == "__main__":
    unittest.main()
