"""Протокол данных и вход цикла: отложенная выборка, сдвиги, сила штриха, пример-пятёрка (вне статьи)."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

os.environ.setdefault("RERUN", "off")

from damp import settings  # noqa: E402
from damp.encoding.elements import StickCoder  # noqa: E402
from damp.logging import LOGGER  # noqa: E402
from damp.stimuli.digits import DigitGroups, ExampleDigit, MnistDigits, PictureSticks  # noqa: E402
from damp.stimuli.mnist import _build_extractor  # noqa: E402
from tests.test_cache import cache_in  # noqa: E402


class StrokeStrengthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.digits = MnistDigits()
        cls.atoms = cls.digits.load(train=True, count=20)

    def test_default_extract_keeps_the_old_format_and_magnitude_is_optional(self) -> None:
        image = self.atoms.images[0].astype(np.float32) / 255.0
        for upscale in (True, False):
            for contour in (True, False):
                with self.subTest(upscale=upscale, contour=contour):
                    extractor = _build_extractor(contour_only=contour, upscale=upscale)
                    plain = extractor.extract(image, label=0)[0]
                    rich = extractor.extract(image, label=0, with_magnitude=True)[0]
                    self.assertTrue(plain)
                    self.assertTrue(all(len(item) == 3 for item in plain))
                    self.assertTrue(all(len(item) == 4 and item[3] > 0.0 for item in rich))
                    self.assertEqual(plain, [item[:3] for item in rich])

    def test_strength_is_normalised_inside_each_picture(self) -> None:
        atoms = self.atoms
        self.assertEqual(atoms.strength.shape, atoms.angle.shape)
        self.assertTrue(np.all(atoms.strength > 0.0) and np.all(atoms.strength <= 1.0))
        for picture in range(atoms.pictures):
            own = atoms.strength[atoms.picture == picture]
            self.assertAlmostEqual(float(own.max()), 1.0, places=12)
        self.assertGreater(np.unique(atoms.strength).shape[0], atoms.pictures)  # у разных штрихов разная сила

    def test_pictures_carry_their_mnist_numbers(self) -> None:
        self.assertEqual(self.atoms.index.dtype, np.int64)
        self.assertEqual(self.atoms.index.shape, self.atoms.labels.shape)
        self.assertTrue(np.all(np.diff(self.atoms.index) > 0))
        later = self.digits.load(train=True, count=3, start=int(self.atoms.index[5]))
        np.testing.assert_array_equal(later.index, self.atoms.index[5:8])
        np.testing.assert_array_equal(later.strength, self.atoms.strength[self.atoms.picture.searchsorted(5) :][: later.angle.shape[0]])

    def test_stick_coder_gives_the_strength_only_when_enabled(self) -> None:
        for enabled in (True, False):
            with self.subTest(enabled=enabled), settings.overridden({"LADDER_STROKE_STRENGTH": enabled}):
                elements = StickCoder().encode(self.atoms)
                if enabled:
                    np.testing.assert_array_equal(elements.power(), self.atoms.strength)
                else:
                    self.assertIsNone(elements.strength)
                    np.testing.assert_array_equal(elements.power(), np.ones(elements.count))

    def test_strongest_pooling_takes_the_strongest_element(self) -> None:
        # Пулинг "strongest" упорядочивает вхождения области по убыванию силы: head берёт сильнейшее.
        strength = np.asarray([0.2, 0.9, 0.5, 0.7])
        entry = np.arange(4)
        pair = np.zeros(4, dtype=np.int64)
        order = np.lexsort((entry, -strength[entry], pair))
        self.assertEqual(int(entry[order][0]), 1)

    def test_old_cache_files_of_three_columns_are_not_read(self) -> None:
        with tempfile.TemporaryDirectory() as folder, cache_in(folder):
            raw = [f"{self.digits_raw()}/train-images-idx3-ubyte", f"{self.digits_raw()}/train-labels-idx1-ubyte"]
            store = PictureSticks("train", raw)
            store.path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(store.path, index=np.asarray([0]), starts=np.asarray([0, 2]), values=np.zeros((2, 3)))
            self.assertIsNone(PictureSticks("train", raw).get(0))
            store.put(0, np.zeros((2, PictureSticks.COLUMNS)))
            store.save()
            self.assertEqual(PictureSticks("train", raw).get(0).shape, (2, 4))

    @staticmethod
    def digits_raw() -> str:
        from torchvision.datasets import MNIST

        return MNIST(root="./data", train=True, download=True).raw_folder

    def test_cached_strokes_equal_fresh_ones(self) -> None:
        with tempfile.TemporaryDirectory() as folder, cache_in(folder):
            fresh = MnistDigits().load(train=True, count=6)
            again = MnistDigits().load(train=True, count=6)  # из кеша штрихов
        for name in ("picture", "angle", "y", "x", "strength", "index"):
            np.testing.assert_array_equal(getattr(fresh, name), getattr(again, name))


class DataProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.digits = MnistDigits()
        cls.groups = DigitGroups.load(
            cls.digits, train=12, test=8, shifts=[(1, 1)], shift_pictures=4, validation=10, validation_start=59000,
            pixel_shift=(2, 2), pixel_pictures=5,
        )

    def test_groups_go_in_order_and_cover_all_pictures(self) -> None:
        ranges = self.groups.ranges
        self.assertEqual(ranges["train"], (0, 12))
        self.assertEqual(ranges["validation"], (12, 22))
        self.assertEqual(ranges["test"], (22, 30))
        self.assertEqual(ranges[(1, 1)], (30, 34))
        self.assertEqual(ranges[("pixel", 2, 2)], (34, 39))
        self.assertEqual(self.groups.pictures, 39)
        self.assertEqual(self.groups.shifts, ((1, 1),))
        self.assertEqual(self.groups.pixel_shifts, ((2, 2),))

    def test_held_out_pictures_do_not_meet_the_taught_ones(self) -> None:
        index = self.groups.atoms.index
        taught, held = set(index[:12].tolist()), set(index[12:22].tolist())
        self.assertFalse(taught & held)
        self.assertTrue(min(held) >= 59000)
        with self.assertRaises(AssertionError):
            DigitGroups.load(self.digits, train=12, test=0, validation=5, validation_start=0)

    def test_default_start_of_the_held_out_set_comes_from_settings(self) -> None:
        with settings.overridden({"BENCH_VALIDATION_POOL": [58000, 60000]}):
            groups = DigitGroups.load(self.digits, train=5, test=0, validation=3)
        self.assertTrue(int(groups.atoms.index[5]) >= 58000)
        self.assertNotIn("test", groups.ranges)

    def test_explicit_numbers_are_taken_exactly_in_order(self) -> None:
        taught, held, checked = [7, 3, 11, 5], [59003, 59001, 50000], [9, 2]
        groups = DigitGroups.load(
            self.digits, train=0, test=0, train_numbers=taught, validation_numbers=held, test_numbers=checked,
            pixel_shift=(1, 1), pixel_pictures=2,
        )
        self.assertEqual(groups.ranges["train"], (0, 4))
        self.assertEqual(groups.ranges["validation"], (4, 7))
        self.assertEqual(groups.ranges["test"], (7, 9))
        self.assertEqual(groups.ranges[("pixel", 1, 1)], (9, 11))
        self.assertEqual(groups.atoms.index.tolist(), taught + held + checked + held[:2])
        with self.assertRaises(AssertionError):  # отложенные не могут совпасть с обучающими
            DigitGroups.load(self.digits, train=0, test=0, train_numbers=[1, 2], validation_numbers=[2, 3])

    def test_train_views_are_shifted_copies_of_the_taught_pictures(self) -> None:
        # §7.1.4: несколько разбиений одной цифры — её сдвинутые виды; они тоже обучающие и идут сразу за ними.
        taught, held = [7, 3], [59003]
        groups = DigitGroups.load(
            self.digits, train=0, test=0, train_numbers=taught, validation_numbers=held, train_views=[(0, 0), (0, 1), (-1, 0)],
        )
        self.assertEqual(groups.ranges["train"], (0, 6))  # (0, 0) — это сами обучающие, отдельным видом не берётся
        self.assertEqual(groups.ranges["views"], (2, 6))
        self.assertEqual((groups.original_pictures, groups.train_pictures), (2, 6))
        self.assertEqual(groups.ranges["validation"], (6, 7))
        # Пример — только среди исходных цифр: у вида номер MNIST тот же, а картинка сдвинута.
        with settings.overridden({"LADDER_FIGURE_DIGIT": int(groups.labels[0]), "LADDER_FIGURE_PICTURE": "random"}):
            for _ in range(10):
                self.assertLess(ExampleDigit.choose(groups), 2)
        self.assertEqual(groups.atoms.index.tolist(), taught * 3 + held)
        np.testing.assert_array_equal(groups.labels[2:4], groups.labels[:2])
        atoms = groups.atoms
        plain, right = atoms.only(0, 2), atoms.only(2, 4)
        np.testing.assert_array_equal(right.images, np.stack([MnistDigits.shift_pixels(image, 0, 1) for image in plain.images]))
        self.assertGreater(float(right.x.mean()), float(plain.x.mean()))
        same = DigitGroups.load(self.digits, train=0, test=0, train_numbers=taught, validation_numbers=held)
        self.assertEqual(same.ranges["train"], (0, 2))

    def test_without_the_new_options_the_groups_are_the_old_ones(self) -> None:
        groups = DigitGroups.load(self.digits, train=6, test=4, shifts=[(1, 1)], shift_pictures=2)
        self.assertEqual(list(groups.ranges), ["train", "test", (1, 1)])
        self.assertEqual(groups.ranges["test"], (6, 10))

    def test_train_start_moves_the_taught_pictures(self) -> None:
        groups = DigitGroups.load(self.digits, train=4, test=0, train_start=10)
        self.assertTrue(int(groups.atoms.index[0]) >= 10)

    def test_pixel_shift_changes_the_strokes_and_cell_shift_does_not(self) -> None:
        atoms = self.groups.atoms
        held = atoms.only(12, 17)
        pixel = atoms.only(*self.groups.ranges[("pixel", 2, 2)])
        np.testing.assert_array_equal(pixel.index, held.index)
        np.testing.assert_array_equal(pixel.labels, held.labels)
        self.assertFalse(np.array_equal(pixel.images, held.images))
        same = pixel.x.shape == held.x.shape and np.array_equal(pixel.x, held.x) and np.array_equal(pixel.y, held.y)
        self.assertFalse(same and np.array_equal(pixel.angle, held.angle))
        # Сдвиг на клетки — те же штрихи (углы, силы), другие места.
        test = atoms.only(*self.groups.ranges["test"])
        cells = atoms.only(*self.groups.ranges[(1, 1)])
        first = np.flatnonzero(test.picture < 4)
        np.testing.assert_array_equal(cells.angle, test.angle[first])
        np.testing.assert_array_equal(cells.strength, test.strength[first])
        np.testing.assert_array_equal(cells.x, test.x[first] + 1)
        np.testing.assert_array_equal(cells.y, test.y[first] + 1)

    def test_shift_pixels_fills_with_zeros(self) -> None:
        image = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
        moved = MnistDigits.shift_pixels(image, 1, -2)
        np.testing.assert_array_equal(moved[0], np.zeros(4))
        np.testing.assert_array_equal(moved[1:, :2], image[:3, 2:])
        np.testing.assert_array_equal(moved[:, 2:], np.zeros((4, 2)))
        np.testing.assert_array_equal(MnistDigits.shift_pixels(image, 0, 0), image)
        np.testing.assert_array_equal(MnistDigits.shift_pixels(image, 9, 0), np.zeros((4, 4)))

    def test_shifted_pictures_stay_out_of_the_stroke_cache(self) -> None:
        with tempfile.TemporaryDirectory() as folder, cache_in(folder):
            self.digits.load(train=True, count=3, shift=(2, 2))
            self.assertFalse(list(Path(folder).glob("pictures/*.npz")))


class ExampleDigitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.groups = DigitGroups.load(MnistDigits(), train=60, test=10, validation=10, validation_start=59000)

    def choose(self, **changes: object) -> tuple[int, mock.MagicMock]:
        with settings.overridden(changes), mock.patch.object(LOGGER, "event") as event:
            return ExampleDigit.choose(self.groups), event

    def test_random_example_is_a_taught_five_and_is_logged(self) -> None:
        seen = set()
        for _ in range(12):
            picture, event = self.choose(LADDER_FIGURE_PICTURE="random")
            self.assertLess(picture, self.groups.train_pictures)
            self.assertEqual(int(self.groups.labels[picture]), 5)
            data = event.call_args.kwargs["data"]
            self.assertEqual(event.call_args.args[0], "ladder.example")
            self.assertEqual(data["picture"], picture)
            self.assertEqual(data["mnist_index"], int(self.groups.atoms.index[picture]))
            self.assertEqual(data["label"], 5)
            seen.add(picture)
        self.assertGreater(len(seen), 1)  # каждый раз — другая пятёрка

    def test_number_picks_the_five_by_count(self) -> None:
        fives = np.flatnonzero(self.groups.labels[: self.groups.train_pictures] == 5)
        for number in (0, 2):
            picture, _ = self.choose(LADDER_FIGURE_PICTURE=number)
            self.assertEqual(picture, int(fives[number]))

    def test_unknown_word_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.choose(LADDER_FIGURE_PICTURE="always")


class RunnerProtocolTest(unittest.TestCase):
    """Прогон цикла по отложенной выборке: пример прогона, группы, метки по validation, итоги для стенда."""

    def test_runner_uses_the_held_out_set_and_one_example_per_run(self) -> None:
        from damp.cycle.ladder import MnistLadderRunner
        from damp.cycle.ladder_trace import LadderObserver

        class Catch(LadderObserver):
            trace = None

            def start(self, trace) -> None:  # noqa: ANN001
                Catch.trace = trace

        # Без видов обучающих (§7.1.4): их проверяет test_train_views_are_shifted_copies_of_the_taught_pictures.
        with tempfile.TemporaryDirectory() as folder, settings.overridden({"LADDER_SAVE_MAPS": False, "LADDER_TRAIN_VIEWS": ()}):
            runner = MnistLadderRunner(
                count=40, evaluate_limit=0, map_max_points=None, shifts=(), file_prefix=str(Path(folder) / "ladder"),
                observer=Catch(), validation=15, pixel_shift=(1, 1), pixel_pictures=5,
            )
            runner.run()
        groups, trace = runner.groups, Catch.trace
        self.assertNotIn("test", groups.ranges)
        self.assertEqual(groups.ranges["validation"], (groups.train_pictures, groups.train_pictures + 15))
        self.assertEqual(groups.pixel_shifts, ((1, 1),))
        self.assertEqual(int(groups.labels[runner.example]), 5)
        self.assertLess(runner.example, groups.train_pictures)
        self.assertEqual(trace.example, runner.example)
        self.assertEqual(trace.example_index, int(groups.atoms.index[runner.example]))
        self.assertTrue(runner.reports and len(runner.level_responses) == len(runner.reports))
        self.assertIsNotNone(runner.top_details)
        self.assertTrue(all(report.alone is not None for report in runner.reports))  # метки — по отложенной выборке


if __name__ == "__main__":
    unittest.main()
