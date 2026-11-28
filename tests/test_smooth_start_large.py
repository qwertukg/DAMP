"""Гладкий старт больших карт (smooth_start.sparse_affinities, embed_blocked): тот же счёт, что плотный, без матриц N × N."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "client"), os.path.join(ROOT, "system")]
os.environ.setdefault("RERUN", "off")

from damp.layout import smooth_start  # noqa: E402


def _codes(points: int = 400, length: int = 512, groups: int = 8, seed: int = 1) -> tuple["np.ndarray", "np.ndarray"]:
    rng = np.random.default_rng(seed)
    prototypes = rng.random((groups, length)) < 0.2
    labels = np.arange(points) % groups
    return (prototypes[labels] ^ (rng.random((points, length)) < 0.05)).astype(np.uint8), labels


def _own_share(xy: "np.ndarray", labels: "np.ndarray") -> float:
    """Доля точек своей группы среди 10 ближайших на плоскости."""
    distance = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2).astype(np.float64)
    np.fill_diagonal(distance, np.inf)
    return float((labels[np.argsort(distance, axis=1)[:, :10]] == labels[:, None]).mean())


class LargeSmoothStartTests(unittest.TestCase):
    def test_sparse_affinities_equal_dense_ones(self) -> None:
        bits, _ = _codes()
        dense = smooth_start.affinities(bits, 10.0)
        rows, cols, values = smooth_start.sparse_affinities(bits, 10.0, use_gpu=False)
        rebuilt = np.zeros_like(dense)
        rebuilt[rows, cols] = values
        # Разница — только на равных сходствах у границы «3 · perplexity ближайших»: плотный счёт берёт все равные.
        self.assertLess(float(np.abs(dense - rebuilt).max()), 0.01 * float(dense.max()))
        self.assertTrue(np.allclose(rebuilt, rebuilt.T))
        self.assertAlmostEqual(float(values.sum()), 1.0, places=4)

    def test_blocked_gradient_equals_the_dense_gradient(self) -> None:
        import torch

        bits, _ = _codes()
        dense = smooth_start.affinities(bits, 10.0)
        rows, cols, values = smooth_start.sparse_affinities(bits, 10.0, use_gpu=False)
        rng = np.random.default_rng(2)
        y = torch.as_tensor(rng.normal(size=(bits.shape[0], 2)).astype(np.float32))
        target = torch.as_tensor(dense).clamp(min=1e-12)
        square = (y * y).sum(dim=1)
        near = 1.0 / (1.0 + (square[:, None] + square[None, :] - 2.0 * (y @ y.T)).clamp(min=0.0))
        near.fill_diagonal_(0.0)
        q = (near / near.sum()).clamp(min=1e-12)
        for factor in (12.0, 1.0):
            pull = (target * factor - q) * near
            expected = 4.0 * (pull.sum(dim=1, keepdim=True) * y - pull @ y)
            for block in (64, 128, 4096):  # блок не кратен числу точек, кратен, больше карты
                found = smooth_start.blocked_gradient(y, torch.as_tensor(rows), torch.as_tensor(cols),
                                                      torch.as_tensor(values).clamp(min=1e-12), factor, block)
                self.assertLess(float((expected - found).abs().max()), 0.01 * float(expected.abs().max()), f"усиление {factor}, блок {block}")

    def test_edge_rows_sum_every_edge_once(self) -> None:
        import torch

        rng = np.random.default_rng(3)
        edge_i = rng.integers(0, 50, size=2000)  # рёбра не по порядку; у точек 50–59 рёбер нет
        edge_i[rng.choice(2000, 300, replace=False)] = 7  # «хаб»: строк таблицы у точки 7 — около десяти
        values = torch.as_tensor(rng.normal(size=(edge_i.size, 2)))
        first, second = smooth_start.edge_rows(edge_i, 60)
        zero = torch.zeros((1, 2), dtype=values.dtype)
        pulled = torch.cat([values, zero])[torch.as_tensor(first)].sum(dim=1)
        found = torch.cat([pulled, zero])[torch.as_tensor(second)].sum(dim=1)
        expected = torch.zeros((60, 2), dtype=values.dtype).index_add_(0, torch.as_tensor(edge_i), values)
        self.assertTrue(torch.allclose(found, expected))
        self.assertEqual(float(found[50:].abs().max()), 0.0)

    def test_blocked_descent_repeats_to_the_bit(self) -> None:
        import torch

        gpu = torch.cuda.is_available()  # на видеокарте index_add_ складывал притяжение в разном порядке — разные раскладки
        bits, _ = _codes()
        edges = smooth_start.sparse_affinities(bits, 10.0, use_gpu=gpu)
        found = [smooth_start.embed_blocked(edges, bits.shape[0], 5, use_gpu=gpu, iterations=300, explore=100) for _ in range(2)]
        self.assertTrue(np.array_equal(found[0], found[1]))

    def test_cells_of_a_map_above_the_limit_keep_code_neighbours_together(self) -> None:
        bits, labels = _codes(points=300)
        limit = smooth_start.LIMIT
        dense = smooth_start.smooth_cells(bits, 20, 20, perplexity=10.0, seed=0, use_gpu=False)
        smooth_start.LIMIT = 100  # карта «больше предела»: счёт без матриц N × N
        try:
            blocked = smooth_start.smooth_cells(bits, 20, 20, perplexity=10.0, seed=0, use_gpu=False)
        finally:
            smooth_start.LIMIT = limit
        for cells in (dense, blocked):
            self.assertEqual(len({(int(y), int(x)) for y, x in cells}), bits.shape[0])  # по одной точке в клетке
            self.assertTrue(((cells >= 0) & (cells < 20)).all())
        # На сетке 20 × 20 у групп по 37 точек доля своих среди 10 ближайших — около 0.8 (соседи на границах пятен — чужие):
        # счёт без матриц N × N даёт то же, что плотный (замер: 0.80 против 0.80 на трёх зёрнах).
        self.assertGreater(_own_share(blocked.astype(np.float64), labels), 0.7)
        self.assertGreater(_own_share(blocked.astype(np.float64), labels), _own_share(dense.astype(np.float64), labels) - 0.05)


if __name__ == "__main__":
    unittest.main()
