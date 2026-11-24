"""Прототип раскладки обменами на сетке (как движок DAMP: пары в радиусе r(t), обмен при выигрыше цели), в котором можно
менять вес пары и ядро расстояния. Ищем рецепт, с которым обмены дают целые классы, как эталон (t-SNE).

python proto.py <папка> <зёрна кодов> <уровень> <зёрна раскладки> <варианты через запятую: вес:ядро[:шаги]> <файл итогов>
Веса: dense (как сейчас: s·сигм(10(s−λ0.82))), knnK (K ближайших, «или»), mknnK (взаимные), pK (сродства t-SNE, перплексия K),
snnK (общие соседи), hardQ (жёсткий порог, квантиль Q %).  Ядра: gauss (σ = r(t), не меньше 3 — как сейчас), lin (d, статья),
log (log(1 + d²) — t-SNE на сетке), logS (log(1 + d²/S²)), cauchyS (−1/(1 + d²/S²)).
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from top_stand import m1_by_digit, map_purity

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def affinities(sim, perplexity):
    """Сродства t-SNE: у каждой точки своя ширина по перплексии; симметричные."""
    dist = np.clip(1.0 - sim, 0.0, None) ** 2
    n = dist.shape[0]
    beta = np.ones(n)
    lo, hi = np.zeros(n), np.full(n, np.inf)
    target = np.log(perplexity)
    np.fill_diagonal(dist, np.inf)
    base = dist.min(axis=1, keepdims=True)
    for _ in range(60):
        p = np.exp(-(dist - base) * beta[:, None])
        p[np.arange(n), np.arange(n)] = 0.0
        total = p.sum(axis=1)
        p /= total[:, None]
        entropy = -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1)
        high = entropy > target
        lo = np.where(high, beta, lo)
        hi = np.where(high, hi, beta)
        beta = np.where(high, np.where(np.isinf(hi), beta * 2.0, (beta + hi) / 2.0), (beta + lo) / 2.0)
    return (p + p.T) / 2.0


def weights(sim, kind):
    n = sim.shape[0]
    s = sim.copy()
    np.fill_diagonal(s, 0.0)
    upper = s[np.triu_indices(n, 1)]
    if kind == "dense":
        lam = np.quantile(upper, 0.82)
        w = s / (1.0 + np.exp(-10.0 * (s - lam)))
    elif kind.startswith("hard"):
        lam = np.quantile(upper, float(kind[4:]) / 100.0)
        w = np.where(s >= lam, s, 0.0)
    elif kind.startswith("p"):
        w = affinities(sim, float(kind[1:]))
    else:
        k = int("".join(ch for ch in kind if ch.isdigit()))
        near = np.argpartition(-s, k, axis=1)[:, :k]
        m = np.zeros((n, n), dtype=bool)
        m[np.arange(n)[:, None], near] = True
        if kind.startswith("mknn"):
            w = (m & m.T).astype(np.float64)
        elif kind.startswith("knn"):
            w = (m | m.T).astype(np.float64)
        elif kind.startswith("snn"):
            w = (m.astype(np.float32) @ m.astype(np.float32).T) / k
        else:
            raise ValueError(kind)
    np.fill_diagonal(w, 0.0)
    return w / w.sum() * n  # общий масштаб не важен; для температуры — на точку


def cost(kernel, d2, radius):
    if kernel in ("gauss", "gauss1"):
        sigma = max(float(radius), 3.0 if kernel == "gauss" else 1.0)
        return -torch.exp(-d2 / (2.0 * sigma * sigma))
    if kernel == "lin":
        return torch.sqrt(d2)
    if kernel.startswith("log"):
        scale = float(kernel[3:]) if len(kernel) > 3 else 1.0
        return torch.log1p(d2 / (scale * scale))
    if kernel.startswith("cauchy"):
        scale = float(kernel[6:])
        return -1.0 / (1.0 + d2 / (scale * scale))
    raise ValueError(kernel)


def lay_out(w, height, width, kernel, seed, steps=2500, pairs=4000, init=None, radius_share=1.0, w2=None):
    """w2 — вторая матрица весов: вес плавно переходит от w к w2 по ходу раскладки (смена разрешения, §5.13.2.4)."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    n, cells = w.shape[0], height * width
    W1 = torch.as_tensor(w, dtype=torch.float32, device=DEV)
    W2 = None if w2 is None else torch.as_tensor(w2, dtype=torch.float32, device=DEV)
    W = W1
    if init is None:
        order = torch.randperm(cells, generator=gen, device=DEV)
        cell_of = order[:n].clone()                               # клетка каждой точки
    else:
        cell_of = torch.as_tensor(init[:, 0] * width + init[:, 1], dtype=torch.long, device=DEV)
    point_at = torch.full((cells,), -1, dtype=torch.long, device=DEV)
    point_at[cell_of] = torch.arange(n, device=DEV)
    ys = (torch.arange(cells, device=DEV) // width).float()
    xs = (torch.arange(cells, device=DEV) % width).float()
    start_radius = max(1.0, float(max(height, width)) * float(radius_share))
    for step in range(steps):
        radius = max(1.0, start_radius * (1.0 / start_radius) ** (step / max(1, steps - 1)))
        if W2 is not None and step % 50 == 0:
            alpha = min(1.0, (step / max(1, steps - 1)) / 0.7)      # к 70 % шагов — только вторая матрица
            W = (1.0 - alpha) * W1 + alpha * W2
        a = torch.randperm(n, generator=gen, device=DEV)[: min(pairs, n)]
        ca = cell_of[a]
        r = int(round(radius))
        dy = torch.randint(-r, r + 1, (a.shape[0],), generator=gen, device=DEV)
        dx = torch.randint(-r, r + 1, (a.shape[0],), generator=gen, device=DEV)
        ty = torch.clamp((ca // width) + dy, 0, height - 1)
        tx = torch.clamp((ca % width) + dx, 0, width - 1)
        cc = ty * width + tx
        keep = cc != ca
        a, ca, cc = a[keep], ca[keep], cc[keep]
        b = point_at[cc]
        py, px = ys[cell_of], xs[cell_of]                          # позиции всех точек
        d2a = (ys[ca][:, None] - py[None, :]) ** 2 + (xs[ca][:, None] - px[None, :]) ** 2
        d2c = (ys[cc][:, None] - py[None, :]) ** 2 + (xs[cc][:, None] - px[None, :]) ** 2
        gain = cost(kernel, d2c, radius) - cost(kernel, d2a, radius)   # изменение цены связи при переезде a → c
        wa = W[a].clone()
        has_b = b >= 0
        wb = torch.zeros_like(wa)
        wb[has_b] = W[b[has_b]]
        rows = torch.arange(a.shape[0], device=DEV)
        wa[rows[has_b], b[has_b]] = 0.0                              # связь a–b при обмене не меняется
        wb[rows, a] = 0.0
        delta = ((wa - wb) * gain).sum(dim=1)
        good = torch.nonzero(delta < -1e-9).squeeze(1)
        if good.numel() == 0:
            continue
        # без пересечений: клетка участвует не больше чем в одном обмене шага (первый по порядку)
        ga, gca, gcc, gb = a[good].cpu().numpy(), ca[good].cpu().numpy(), cc[good].cpu().numpy(), b[good].cpu().numpy()
        both = np.concatenate([gca, gcc])
        _, first = np.unique(both, return_index=True)
        is_first = np.zeros(both.shape[0], dtype=bool)
        is_first[first] = True
        ok = is_first[: ga.shape[0]] & is_first[ga.shape[0]:]
        ga, gca, gcc, gb = (torch.as_tensor(v[ok], device=DEV) for v in (ga, gca, gcc, gb))
        cell_of[ga] = gcc
        point_at[gcc] = ga
        point_at[gca] = gb
        moved = gb >= 0
        cell_of[gb[moved]] = gca[moved]
    out = cell_of.cpu().numpy()
    return np.stack([out // width, out % width], axis=1)


def snap(xy, height, width):
    """Точки плоскости → клетки сетки без повторов (рекурсивное деление пополам)."""
    sys.setrecursionlimit(100000)
    cells = np.zeros((xy.shape[0], 2), dtype=np.int64)

    def assign(idx, y0, y1, x0, x1):
        n = idx.shape[0]
        if n == 0:
            return
        if (y1 - y0) * (x1 - x0) == 1:
            cells[idx[0]] = (y0, x0)
            return
        capacity = (y1 - y0) * (x1 - x0)
        if (y1 - y0) >= (x1 - x0):
            mid = (y0 + y1) // 2
            cap1 = (mid - y0) * (x1 - x0)
            order = idx[np.argsort(xy[idx, 0], kind="stable")]
            n1 = int(min(max(round(n * cap1 / capacity), n - (capacity - cap1)), min(n, cap1)))
            assign(order[:n1], y0, mid, x0, x1); assign(order[n1:], mid, y1, x0, x1)
        else:
            mid = (x0 + x1) // 2
            cap1 = (y1 - y0) * (mid - x0)
            order = idx[np.argsort(xy[idx, 1], kind="stable")]
            n1 = int(min(max(round(n * cap1 / capacity), n - (capacity - cap1)), min(n, cap1)))
            assign(order[:n1], y0, y1, x0, mid); assign(order[n1:], y0, y1, mid, x1)

    assign(np.arange(xy.shape[0]), 0, int(height), 0, int(width))
    return cells


def coarse(x, sim, height, width, size, kind, seed, steps=1500):
    """От крупного к мелкому: группы похожих точек (Уорд по кодам, без меток) раскладываются как целые на грубой сетке,
    затем точки садятся в клетки вокруг места своей группы."""
    from scipy.cluster.hierarchy import fcluster, linkage
    n = x.shape[0]
    groups = fcluster(linkage(x, method="ward"), t=max(2, n // size), criterion="maxclust") - 1
    count = int(groups.max()) + 1
    member = np.zeros((count, n), dtype=np.float64)
    member[groups, np.arange(n)] = 1.0
    sizes = member.sum(axis=1)
    mean = (member @ sim @ member.T) / np.outer(sizes, sizes)        # среднее сходство групп
    np.fill_diagonal(mean, 0.0)
    w = weights(mean, kind)
    side = int(np.ceil(np.sqrt(count / 0.85)))
    place = lay_out(w, side, side, "gauss1", seed, steps=steps)
    rng = np.random.default_rng(seed)
    xy = (place[groups] + 0.5 + rng.uniform(-0.45, 0.45, size=(n, 2))) * np.array([height / side, width / side])
    return snap(xy, height, width), count


def main():
    folder, seeds, level = sys.argv[1], [int(v) for v in sys.argv[2].split(",")], int(sys.argv[3])
    layout_seeds = [int(v) for v in sys.argv[4].split(",")]
    variants, out = sys.argv[5].split(","), sys.argv[6]
    for seed in seeds:
        z = np.load(f"{folder}/dump-seed{seed}.npz")
        bits = np.unpackbits(z[f"L{level}_bits"], axis=1)[:, : int(z[f"L{level}_length"])].astype(np.float32)
        labels, picture = z[f"L{level}_labels"], z[f"L{level}_picture"]
        height, width = [int(v) for v in z[f"L{level}_size"]]
        x = bits / np.sqrt(bits.sum(axis=1, keepdims=True))
        sim = (x @ x.T).astype(np.float64)
        cache = {}
        def get(kind):
            if kind not in cache:
                cache[kind] = weights(sim, kind)
            return cache[kind]

        for name in variants:
            for s in layout_seeds:
                start = time.time()
                cells = None
                for number, stage in enumerate(name.split(">")):    # этапы: вес[+вес2]:ядро[:шаги[:доля радиуса]]
                    part = stage.split(":")
                    if part[0].startswith("ml"):                    # ml<размер группы>:<вес групп>
                        cells, _ = coarse(x, sim, height, width, int(part[0][2:]), part[1], 1000 * seed + s)
                        continue
                    kinds, kernel = part[0].split("+"), part[1]
                    steps = int(part[2]) if len(part) > 2 else 2500
                    share = float(part[3]) if len(part) > 3 else (1.0 if number == 0 else 0.5)
                    cells = lay_out(get(kinds[0]), height, width, kernel, 1000 * seed + s + 77 * number, steps=steps, init=cells,
                                    radius_share=share, w2=get(kinds[1]) if len(kinds) > 1 else None)
                m1, patches = m1_by_digit(cells, labels, height, width)
                row = {"seed": seed, "level": level, "variant": name, "layout_seed": s, "m1u": float(np.mean([m1.get(d, 0.0) for d in range(10)])),
                       "m1": [round(m1.get(d, 0.0), 3) for d in range(10)], "patches": patches, "honest": map_purity(cells, labels, picture),
                       "seconds": round(time.time() - start, 1)}
                open(out, "a", encoding="utf-8").write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"{name} зерно {seed} раскладка {s}: М1u {row['m1u']:.3f} честная {row['honest']:.3f} пятен {patches} ({row['seconds']} с)", flush=True)


if __name__ == "__main__":
    main()
