"""Гладкий старт раскладки (план v19.15; вне статьи — в статье точки ставят в случайные клетки, §7.1.6; слово автора
2026-11-23: «делай гладкий старт»).

Со случайного старта класс кристаллизуется сразу в нескольких местах карты (§5.13.1: «several crystallisation centres»), и
обмены по одной точке потом не перевозят готовый кусок. Здесь стартовая расстановка берётся из сходства кодов, без меток:
точки расставляются на плоскости так, чтобы у каждой рядом оказались её ближайшие по коду (t-SNE: у точки около
perplexity «своих» соседей, своя ширина близости), затем плоскость сажается на сетку карты. Дальше раскладку доводит обычный
движок с малого радиуса — он наводит местный порядок, не разрывая собранные классы.

Замер (docs/CONVERGENCE.md §2.11; те же коды верхней карты, 6 зёрен, опора — 36 раскладок со случайного старта):
М1u 0.848 против 0.762, честная сходимость 0.700 против 0.659.

Счёт детерминирован: начальные координаты — от зерна, дальше одни и те же действия над одними и теми же числами (и на
видеокарте: притяжение больших карт складывается в постоянном порядке — edge_rows, не атомарным index_add_).
"""

from __future__ import annotations

import sys

import numpy as np

LIMIT = 8000  # точек: плотный счёт — матрицы N × N (float32), до ~1.3 ГБ видеопамяти на 8000 точек
# Карты крупнее LIMIT — тот же счёт без матриц N × N (sparse_affinities, embed_blocked): сродства рёбрами, отталкивание блоками
# строк; память O(блок × N). Карты крупнее LARGE_LIMIT раскладываются со случайного старта, как обычно.
LARGE_LIMIT = 70000
BLOCK_CELLS = 64_000_000  # пар «строка блока × точка» на блок отталкивания (float32, несколько временных — около 1 ГБ)
ROW_WIDTH = 32  # рёбер в строке таблицы притяжения (edge_rows): у точки ~30 своих соседей и столько же пришедших к ней


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


def sparse_affinities(bits: "np.ndarray", perplexity: float, *, use_gpu: bool, block: int = 1024, chunk: int = 16384):  # noqa: ANN201
    """Сродства рёбрами для больших карт: (строки, столбцы, значения) — те же сродства, что у affinities, без матрицы N × N.
    У точки neighbours = 3 · perplexity ближайших по коду (при равенстве сходств — первые по порядку выборки), ширина
    близости — тем же делением пополам, симметризация — (P + Pᵀ) / 2N сложением рёбер обоих направлений."""
    import torch

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    codes = torch.as_tensor(np.ascontiguousarray(bits, dtype=np.uint8), device=device)
    count = int(codes.shape[0])
    neighbours = max(1, min(count - 1, int(round(3.0 * float(perplexity)))))
    target = float(np.log(perplexity))
    norms = torch.sqrt(codes.sum(dim=1, dtype=torch.float32).clamp_(min=1e-12))
    rows_all, cols_all, values_all = [], [], []
    with torch.no_grad():
        for start in range(0, count, block):
            stop = min(start + block, count)
            left = codes[start:stop].to(torch.float32)
            cosine = torch.empty((stop - start, count), dtype=torch.float32, device=device)
            for first in range(0, count, chunk):  # коды — uint8 на устройстве, во float32 — кусками столбцов
                last = min(first + chunk, count)
                cosine[:, first:last] = left @ codes[first:last].to(torch.float32).T
            cosine /= norms[start:stop, None] * norms[None, :]
            cosine[torch.arange(stop - start, device=device), torch.arange(start, stop, device=device)] = -2.0  # сама себе не сосед
            near, index = torch.topk(cosine, neighbours, dim=1)
            distance = np.clip(1.0 - near.cpu().numpy().astype(np.float64), 0.0, None) ** 2
            nearest = distance.min(axis=1, keepdims=True)
            beta = np.ones(stop - start, dtype=np.float64)
            low = np.zeros(stop - start, dtype=np.float64)
            high = np.full(stop - start, np.inf, dtype=np.float64)
            for _ in range(60):  # деление пополам по ширине близости
                p = np.exp(-(distance - nearest) * beta[:, None])
                p /= p.sum(axis=1, keepdims=True)
                entropy = -(p * np.log(np.maximum(p, 1e-300))).sum(axis=1)
                wide = entropy > target
                low = np.where(wide, beta, low)
                high = np.where(wide, high, beta)
                beta = np.where(wide, np.where(np.isinf(high), beta * 2.0, (beta + high) / 2.0), (beta + low) / 2.0)
            rows_all.append(np.repeat(np.arange(start, stop, dtype=np.int64), neighbours))
            cols_all.append(index.cpu().numpy().astype(np.int64).reshape(-1))
            values_all.append(p.astype(np.float32).reshape(-1))
    rows, cols, values = np.concatenate(rows_all), np.concatenate(cols_all), np.concatenate(values_all)
    both = torch.sparse_coo_tensor(
        torch.as_tensor(np.stack([np.concatenate([rows, cols]), np.concatenate([cols, rows])])),
        torch.as_tensor(np.concatenate([values, values])), (count, count)).coalesce()
    index = both.indices().numpy()
    return index[0], index[1], (both.values().numpy() / np.float32(2.0 * count)).astype(np.float32)


def edge_rows(edge_i: "np.ndarray", count: int, width: int = ROW_WIDTH) -> tuple["np.ndarray", "np.ndarray"]:
    """Таблицы сложения притяжения по рёбрам в постоянном порядке. Рёбра каждой точки лежат подряд в строках шириной width
    (first: строка × место → номер ребра; пустое место — номер len(edge_i), нулевое ребро), строки точки — в second
    (точка × k → номер строки; пусто — номер числа строк, нулевая строка). Сумма — сбором и сложением вдоль оси: порядок
    сложения один и тот же при каждом запуске, в отличие от index_add_ на видеокарте (атомарное сложение float32 идёт в
    разном порядке, и за 1000 шагов спуска разница последних знаков вырастает в другую раскладку)."""
    edge_i = np.asarray(edge_i, dtype=np.int64)
    edges = int(edge_i.shape[0])
    order = np.argsort(edge_i, kind="stable")
    owner = edge_i[order]
    counts = np.bincount(edge_i, minlength=count)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    slot = np.arange(edges, dtype=np.int64) - starts[owner]
    lines = (counts + width - 1) // width
    line_start = np.concatenate([[0], np.cumsum(lines)[:-1]]).astype(np.int64)
    total = int(lines.sum())
    first = np.full((total, int(width)), edges, dtype=np.int64)
    first[line_start[owner] + slot // width, slot % width] = order
    deepest = int(lines.max()) if count else 0
    second = np.full((count, max(1, deepest)), total, dtype=np.int64)
    for k in range(deepest):
        has = lines > k
        second[has, k] = line_start[has] + k
    return first, second


def blocked_gradient(y, edge_i, edge_j, edge_p, exaggeration: float, block: int, rows=None):  # noqa: ANN001, ANN201
    """Градиент расхождения Кульбака — Лейблера без матриц N × N — тот же, что в embed: 4 · [Σ_j P_ij · k · near_ij ·
    (y_i − y_j) − (1 / Z) · Σ_j near_ij² · (y_i − y_j)], near = 1 / (1 + d²), Z = Σ near по всем парам. Притяжение — по рёбрам
    сродств (rows — таблицы edge_rows на устройстве y; нет — строятся здесь), отталкивание и Z — блоками строк."""
    import torch

    count = int(y.shape[0])
    square = (y * y).sum(dim=1)
    repulsion = torch.zeros_like(y)
    total = torch.zeros((), dtype=torch.float64, device=y.device)
    for start in range(0, count, block):
        stop = min(start + block, count)
        part = y[start:stop]
        near = 1.0 / (1.0 + (square[start:stop, None] + square[None, :] - 2.0 * (part @ y.T)).clamp_(min=0.0))
        near[torch.arange(stop - start, device=y.device), torch.arange(start, stop, device=y.device)] = 0.0
        total += near.sum(dtype=torch.float64)
        near *= near
        repulsion[start:stop] = near.sum(dim=1, keepdim=True) * part - near @ y
    difference = y[edge_i] - y[edge_j]
    weight = edge_p * exaggeration / (1.0 + (difference * difference).sum(dim=1))
    if rows is None:
        rows = tuple(torch.as_tensor(table, device=y.device) for table in edge_rows(edge_i.cpu().numpy(), count))
    first, second = rows
    zero = torch.zeros((1, y.shape[1]), dtype=y.dtype, device=y.device)
    pulled = torch.cat([weight[:, None] * difference, zero])[first].sum(dim=1)
    attraction = torch.cat([pulled, zero])[second].sum(dim=1)
    return 4.0 * (attraction - repulsion / total.to(torch.float32))


def embed_blocked(edges, count: int, seed: int, *, use_gpu: bool, iterations: int = 1000, explore: int = 250,  # noqa: ANN001, ANN201
                  exaggeration: float = 12.0) -> "np.ndarray":
    """Координаты точек больших карт (N × 2): тот же спуск, что embed (начало от зерна, скорость, усиление, инерция), градиент
    — blocked_gradient. Блок строк — по BLOCK_CELLS пар «строка × точка»."""
    import torch

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    y = (torch.randn((count, 2), generator=generator, dtype=torch.float32) * 1e-4).to(device)
    edge_i = torch.as_tensor(edges[0], device=device)
    edge_j = torch.as_tensor(edges[1], device=device)
    edge_p = torch.as_tensor(edges[2], device=device).clamp_(min=1e-12)
    rows = tuple(torch.as_tensor(table, device=device) for table in edge_rows(edges[0], count))
    block = max(128, min(4096, BLOCK_CELLS // max(1, count)))
    rate = max(count / exaggeration / 4.0, 50.0)
    update = torch.zeros_like(y)
    gains = torch.ones_like(y)
    with torch.no_grad():
        for step in range(int(iterations)):
            early = step < int(explore)
            gradient = blocked_gradient(y, edge_i, edge_j, edge_p, exaggeration if early else 1.0, block, rows)
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
    width_of_near = min(float(perplexity), max(2.0, (count - 1) / 3.0))
    if count > LIMIT:
        xy = embed_blocked(sparse_affinities(bits, width_of_near, use_gpu=use_gpu), count, seed, use_gpu=use_gpu)
    else:
        xy = embed(affinities(bits, width_of_near), seed, use_gpu=use_gpu)
    return snap(xy, height, width)
