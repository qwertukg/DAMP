"""Эталоны: kNN по пикселям и CNN (вне статьи)."""
import os
import unittest

import numpy as np

os.environ.setdefault("RERUN", "off")

from damp.decode.baselines import ConvNetBaseline, PixelNearest, ShiftAugment  # noqa: E402
from damp.stimuli.digits import MnistDigits  # noqa: E402


class PixelNearestTest(unittest.TestCase):
    def test_matches_direct_computation(self) -> None:
        rng = np.random.default_rng(1)
        train = rng.random((40, 28, 28)).astype(np.float32)
        labels = rng.integers(0, 4, 40)
        query = rng.random((12, 28, 28)).astype(np.float32)
        model = PixelNearest(train, labels)
        for k in (1, 3, 5, 9):
            expected = []
            for picture in query:
                distance = ((train.reshape(40, -1).astype(np.float64) - picture.reshape(-1)) ** 2).sum(axis=1)
                near = labels[np.argsort(distance, kind="stable")[:k]]
                expected.append(int(np.bincount(near, minlength=10).argmax()))
            np.testing.assert_array_equal(model.predict(query, k), expected)

    def test_tie_goes_to_lower_class_and_bytes_are_scaled(self) -> None:
        train = np.zeros((2, 28, 28), dtype=np.uint8)
        train[1] += 255
        model = PixelNearest(train, np.array([7, 3]))
        query = np.full((1, 28, 28), 127.5)
        self.assertEqual(int(model.predict(query, 2)[0]), 3)  # 1-1, меньший номер
        self.assertEqual(model.accuracy(train, np.array([7, 3]), ks=[1]), {1: 1.0})


class ShiftAugmentTest(unittest.TestCase):
    def test_shifts_by_whole_pixels_within_range(self) -> None:
        import torch

        rng = np.random.default_rng(2)
        pixels = rng.random((16, 28, 28)).astype(np.float32)
        augment = ShiftAugment(2)
        batch = torch.as_tensor(pixels)[:, None]
        dy, dx = augment.draw(16, torch.Generator().manual_seed(5))
        self.assertTrue(int(dy.abs().max()) <= 2 and int(dx.abs().max()) <= 2)
        self.assertGreater(len({(int(a), int(b)) for a, b in zip(dy, dx)}), 3)
        moved = augment.apply(batch, dy, dx)[:, 0].numpy()
        for i in range(16):
            np.testing.assert_array_equal(moved[i], MnistDigits.shift_pixels(pixels[i], int(dy[i]), int(dx[i])))

    def test_zero_pixels_is_identity(self) -> None:
        import torch

        batch = torch.rand(3, 1, 28, 28)
        self.assertTrue(torch.equal(ShiftAugment(0)(batch), batch))


class ConvNetBaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        digits = MnistDigits()
        train = digits.load(train=True, count=100)
        check = digits.load(train=True, count=100, start=59000)
        cls.train = (train.images, train.labels)
        cls.check = (check.images, check.labels)

    def fit(self, arch: str = "lenet5", augment: str = "none", seed: int = 0) -> ConvNetBaseline:
        model = ConvNetBaseline(arch, augment=augment, seed=seed)
        model.fit(*self.train, *self.check, dataset="test", epochs=60, patience=100)
        return model

    def test_learns_better_than_chance_and_is_deterministic(self) -> None:
        first, second = self.fit(), self.fit()
        self.assertGreater(first.result.validation_accuracy, 0.5)
        self.assertEqual(first.result.history, second.result.history)
        np.testing.assert_array_equal(first.predict(self.check[0]), second.predict(self.check[0]))
        self.assertAlmostEqual(first.accuracy(*self.check), first.result.validation_accuracy)

    def test_wide_with_shifts_runs(self) -> None:
        model = self.fit("wide", "shifts")
        self.assertGreater(model.result.validation_accuracy, 0.3)
        self.assertEqual(model.predict(self.check[0]).shape, (100,))


if __name__ == "__main__":
    unittest.main()
