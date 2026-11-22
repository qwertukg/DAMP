"""Проход «лучший ответ» после раскладки карты (план v4.15, вне статьи по способу, по статье по мере §5.2, рис. 10).

Заблудшая точка — та, чьё поле притяжения в своей ячейке намного слабее, чем в лучшей ячейке карты. Поле точки i в ячейке c:
F_i(c) = Σ_j (s_ij − λ) · exp(−d(c, p_j)² / 2σ²) по всем точкам j ≠ i (s — косинус кодов, λ — порог раскладки): похожие
тянут, непохожие отталкивают, как в энергии раскладки §5.2, но в радиусе σ кластера, а не круга вокруг пары.
Обмен i (ячейка p) с жителем k ячейки q (или переход в пустую q) принимается, если растёт общая энергия
E = Σ_{a<b} (s_ab − λ)·g(d_ab):  ΔE = F_i(q) − F_i(p) + F_k(p) − F_k(q) − 2(s_ik − λ)(1 − g(d_pq)) > 0.
Метки цифр не используются.
"""

from __future__ import annotations

import numpy as np


def rescue_cells(
    bits: "np.ndarray",
    cells: "np.ndarray",
    height: int,
    width: int,
    threshold: float,
    *,
    passes: int,
    share: float,
    sigma: float,
    targets: int = 8,
) -> tuple["np.ndarray", list[dict]]:
    """Новые ячейки точек (N × 2) и сводка по проходам. bits — коды точек в порядке ячеек cells."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    codes = torch.as_tensor(np.asarray(bits, dtype=np.float32), device=device)
    norms = codes.sum(dim=1).clamp_min(1e-9).sqrt()
    count = int(codes.shape[0])
    weights = torch.empty((count, count), dtype=torch.float32, device=device)
    step = 1024
    for start in range(0, count, step):
        block = codes[start:start + step] @ codes.T
        weights[start:start + step] = block / (norms[start:start + step, None] * norms[None, :]) - float(threshold)
    weights.fill_diagonal_(0.0)
    del codes

    cells = np.asarray(cells, dtype=np.int64).copy()
    grid_y, grid_x = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32),
                                    torch.arange(width, device=device, dtype=torch.float32), indexing="ij")
    grid = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=1)  # G × 2
    occupant = np.full(height * width, -1, dtype=np.int64)
    occupant[cells[:, 0] * width + cells[:, 1]] = np.arange(count)
    scale = -0.5 / float(sigma) ** 2

    def kernel_rows(points: "np.ndarray") -> "torch.Tensor":
        at = torch.as_tensor(cells[points], device=device, dtype=torch.float32)
        return torch.exp(((at[:, None, :] - grid[None, :, :]) ** 2).sum(dim=2) * scale)  # len × G

    report = []
    limit = max(1, int(round(float(share) * count)))
    for number in range(int(passes)):
        kernel = kernel_rows(np.arange(count))  # N × G: вклад каждой точки в поле каждой ячейки
        field = weights @ kernel  # N × G
        own = field[torch.arange(count, device=device), torch.as_tensor(cells[:, 0] * width + cells[:, 1], device=device)]
        best = field.max(dim=1).values
        gaps = (best - own).cpu().numpy()
        order = [int(i) for i in np.argsort(-gaps, kind="stable")[:limit] if gaps[i] > 0]
        del field
        moved, gain = 0, 0.0
        for i in order:
            row = weights[i] @ kernel  # G: поле точки i при нынешних местах
            p = int(cells[i, 0] * width + cells[i, 1])
            top = torch.topk(row, min(int(targets), row.shape[0])).indices.cpu().numpy()
            choice, choice_gain = None, 0.0
            for q in top:
                q = int(q)
                if q == p:
                    continue
                k = int(occupant[q])
                delta = float(row[q] - row[p])
                if k >= 0:
                    at_k = weights[k] @ kernel[:, [p, q]]  # поле жителя k только в двух ячейках
                    dpq = float(((cells[i] - cells[k]) ** 2).sum())
                    pair = float(weights[i, k])
                    delta += float(at_k[0] - at_k[1]) - 2.0 * pair * (1.0 - float(np.exp(dpq * scale)))
                if delta > choice_gain:
                    choice, choice_gain = q, delta
            if choice is None:
                continue
            k = int(occupant[choice])
            old = cells[i].copy()
            cells[i] = (choice // width, choice % width)
            occupant[choice] = i
            occupant[p] = k
            changed = [i]
            if k >= 0:
                cells[k] = old
                changed.append(k)
            kernel[changed] = kernel_rows(np.asarray(changed))
            moved += 1
            gain += choice_gain
        report.append({"pass": number + 1, "candidates": len(order), "moved": moved, "gain": round(gain, 3)})
        if moved == 0:
            break
    del weights
    if device == "cuda":
        torch.cuda.empty_cache()
    return cells, report
