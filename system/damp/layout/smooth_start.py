"""Гладкий старт раскладки (план v19.15; вне статьи — в статье точки ставят в случайные клетки, §7.1.6; слово автора
2026-11-23: «делай гладкий старт»).

Со случайного старта класс кристаллизуется сразу в нескольких местах карты (§5.13.1: «several crystallisation centres»), и
обмены по одной точке потом не перевозят готовый кусок. Здесь стартовая расстановка берётся из сходства кодов, без меток:
точки расставляются на плоскости так, чтобы у каждой рядом оказались её ближайшие по коду (t-SNE: у точки около
perplexity «своих» соседей, своя ширина близости), затем плоскость сажается на сетку карты. Дальше раскладку доводит обычный
движок с малого радиуса — он наводит местный порядок, не разрывая собранные классы.

Замер (docs/CONVERGENCE.md §2.11; те же коды верхней карты, 6 зёрен, опора — 36 раскладок со случайного старта):
М1u 0.848 против 0.762, честная сходимость 0.700 против 0.659.

Счёт детерминирован: начальные координаты — от зерна, дальше одни и те же действия над одними и теми же числами.
"""

from __future__ import annotations

import sys

import numpy as np

LIMIT = 8000  # точек: матрицы N × N (float32) — до ~1.3 ГБ видеопамяти на 8000 точек; карты крупнее раскладываются как обычно


def affinities(bits: "np.ndarray", perplexity: float, block: int = 1024) -> "np.ndarray":
    """Сродства точек P (N × N, сумма 1): у каждой точки своя ширина близости, подобранная так, чтобы «своих» соседей у неё
    было около perplexity (энтропия строки = ln perplexity) среди её 3 · perplexity ближайших; расстояние — квадрат
    косинусного (1 − cos) кодов."""
    x = np.asarray(bits, dtype=np.float32)
    x = x / np.maximum(np.sqrt(x.sum(axis=1, keepdims=True)), 1e-6)
    count = x.shape[0]
    found = np.zeros((count, count), dtype=np.float32)
    target = float(np.log(perplexity))
    neighbours = max(1, min(count - 1, int(round(3.0 * float(perplexity)))))
    for start in range(0, count, block):
        rows = np.arange(start, min(start + block, count))
        distance = np.clip(1.0 - (x[rows] @ x.T).astype(np.float64), 0.0, None) ** 2
        distance[np.arange(rows.shape[0]), rows] = np.inf  # сама себе точка не сосед
        # Сродство — только к neighbours ближайшим (3 · perplexity): с остальными оно ровно ноль. Плотное сродство склеивает
        # классы слабыми дальними связями (замер: М1u стартовой расстановки 0.834 против 0.872 у разрежённого).
        edge = np.partition(distance, neighbours - 1, axis=1)[:, neighbours - 1 : neighbours]
        distance = np.where(distance <= edge, distance, np.inf)
        nearest = distance.min(axis=1, keepdims=True)
        beta = np.ones(rows.shape[0], dtype=np.float64)
        low = np.zeros(rows.shape[0], dtype=np.float64)
        high = np.full(rows.shape[0], np.inf, dtype=np.float64)
        for _ in range(60):  # деление пополам по ширине близости
            p = np.exp(-(distance - nearest) * beta[:, None])
            p /= p.sum(axis=1, keepdims=True)
            entropy = -(p * np.log(np.maximum(p, 1e-300))).sum(axis=1)
            wide = entropy > target
            low = np.where(wide, beta, low)
            high = np.where(wide, high, beta)
            beta = np.where(wide, np.where(np.isinf(high), beta * 2.0, (beta + high) / 2.0), (beta + low) / 2.0)
        found[rows] = p.astype(np.float32)
    return (found + found.T) / np.float32(2.0 * count)


def embed(p: "np.ndarray", seed: int, *, use_gpu: bool, iterations: int = 1000, explore: int = 250,
          exaggeration: float = 12.0) -> "np.ndarray":
    """Координаты точек на плоскости (N × 2) по сродствам P: спуск по расхождению Кульбака — Лейблера между P и сродством на
    плоскости 1 / (1 + d²) (t-SNE, точный градиент); первые explore шагов притяжение усилено в exaggeration раз."""
    import torch

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    count = int(p.shape[0])
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    y = (torch.randn((count, 2), generator=generator, dtype=torch.float32) * 1e-4).to(device)
    target = torch.as_tensor(p, dtype=torch.float32, device=device).clamp_(min=1e-12)
    rate = max(count / exaggeration / 4.0, 50.0)
    update = torch.zeros_like(y)
    gains = torch.ones_like(y)
    with torch.no_grad():
        for step in range(int(iterations)):
            early = step < int(explore)
            square = (y * y).sum(dim=1)
            near = 1.0 / (1.0 + (square[:, None] + square[None, :] - 2.0 * (y @ y.T)).clamp_(min=0.0))
            near.fill_diagonal_(0.0)
            q = (near / near.sum()).clamp_(min=1e-12)
            pull = ((target * exaggeration if early else target) - q) * near
            gradient = 4.0 * (pull.sum(dim=1, keepdim=True) * y - pull @ y)
            gains = torch.where(update * gradient < 0.0, gains + 0.2, gains * 0.8).clamp_(min=0.01)
            update = (0.5 if early else 0.8) * update - rate * gains * gradient
            y = y + update
    return y.cpu().numpy().astype(np.float64)


def snap(xy: "np.ndarray", height: int, width: int) -> "np.ndarray":
    """Точки плоскости → клетки сетки height × width, по одной точке в клетке: рекурсивное деление пополам (точки делятся
    пропорционально числу клеток половин — пустые клетки расходятся по карте равномерно, соседство сохраняется)."""
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 100000))
    cells = np.zeros((xy.shape[0], 2), dtype=np.int64)

    def assign(index: "np.ndarray", top: int, bottom: int, left: int, right: int) -> None:
        count = index.shape[0]
        if count == 0:
            return
        capacity = (bottom - top) * (right - left)
        if capacity == 1:
            cells[index[0]] = (top, left)
            return
        if (bottom - top) >= (right - left):
            middle = (top + bottom) // 2
            first = (middle - top) * (right - left)
            order = index[np.argsort(xy[index, 0], kind="stable")]
            share = int(min(max(round(count * first / capacity), count - (capacity - first)), min(count, first)))
            assign(order[:share], top, middle, left, right)
            assign(order[share:], middle, bottom, left, right)
        else:
            middle = (left + right) // 2
            first = (bottom - top) * (middle - left)
            order = index[np.argsort(xy[index, 1], kind="stable")]
            share = int(min(max(round(count * first / capacity), count - (capacity - first)), min(count, first)))
            assign(order[:share], top, bottom, left, middle)
            assign(order[share:], top, bottom, middle, right)

    try:
        assign(np.arange(xy.shape[0]), 0, int(height), 0, int(width))
    finally:
        sys.setrecursionlimit(limit)
    return cells


def smooth_cells(bits: "np.ndarray", height: int, width: int, *, perplexity: float, seed: int, use_gpu: bool) -> "np.ndarray":
    """Стартовые клетки точек карты height × width по их кодам bits (N × длина, 0 / 1): сродства → плоскость → сетка."""
    count = int(np.asarray(bits).shape[0])
    if count > height * width:
        raise ValueError("точек больше, чем клеток карты")
    xy = embed(affinities(bits, min(float(perplexity), max(2.0, (count - 1) / 3.0))), seed, use_gpu=use_gpu)
    return snap(xy, height, width)
