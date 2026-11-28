# -*- coding: utf-8 -*-
"""Карты всех уровней (0–5) тремя способами на одних и тех же кодах: наша раскладка, t-SNE, UMAP (оба — без меток, посажены на
ту же сетку). Уровень 0 — цвет по углу штриха (пинвил, как на картах стенда), уровни 1–5 — цвет по цифре.

    python tools/tuning/top/three_maps.py embed <выгрузка.npz> <tsne|umap>     # клетки способа → <выгрузка>.<способ>.npz
    python tools/tuning/top/three_maps.py draw <выгрузка.npz> <рисунок.png> [подпись]

Системный Python со scikit-learn и umap-learn, только процессор.
"""
import colorsys
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")
sys.setrecursionlimit(100000)
EIGHT = np.ones((3, 3), bool)
COMPACT = "--compact" in sys.argv
DIGIT = [None] * 10


def digit_colours():
    from PIL import Image
    return [tuple(int(c) for c in Image.new("HSV", (1, 1), (int(d * 25.5), 220, 255)).convert("RGB").getpixel((0, 0))) for d in range(10)]


def snap(xy, height, width):
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
            assign(order[:n1], y0, mid, x0, x1)
            assign(order[n1:], mid, y1, x0, x1)
        else:
            mid = (x0 + x1) // 2
            cap1 = (y1 - y0) * (mid - x0)
            order = idx[np.argsort(xy[idx, 1], kind="stable")]
            n1 = int(min(max(round(n * cap1 / capacity), n - (capacity - cap1)), min(n, cap1)))
            assign(order[:n1], y0, y1, x0, mid)
            assign(order[n1:], y0, y1, mid, x1)

    assign(np.arange(xy.shape[0]), 0, int(height), 0, int(width))
    return cells


def compact(xy, height, width):
    """Точки плоскости → сплошное пятно клеток в центре сетки (пустые клетки остаются снаружи, как у нашей раскладки):
    берутся N клеток, ближайших к центру, и точки сопоставляются им делением пополам — клетки и точки режутся по одной оси
    на равные половины, порядок вдоль оси сохраняется."""
    count = xy.shape[0]
    yy, xx = np.meshgrid(np.arange(int(height)), np.arange(int(width)), indexing="ij")
    far = (yy - (height - 1) / 2.0) ** 2 + (xx - (width - 1) / 2.0) ** 2
    order = np.argsort(far.ravel(), kind="stable")[:count]
    target = np.stack([yy.ravel()[order], xx.ravel()[order]], axis=1)
    cells = np.zeros((count, 2), dtype=np.int64)

    def match(points, places):
        if points.size == 1:
            cells[points[0]] = target[places[0]]
            return
        extent = target[places].max(axis=0) - target[places].min(axis=0)
        axis = 0 if extent[0] >= extent[1] else 1
        places = places[np.lexsort((target[places, 1 - axis], target[places, axis]))]
        points = points[np.lexsort((xy[points, 1 - axis], xy[points, axis]))]
        half = points.size // 2
        match(points[:half], places[:half])
        match(points[half:], places[half:])

    match(np.arange(count), np.arange(count))
    return cells


def m1u(cells, labels, height, width):
    grid = np.full((int(height), int(width)), -1, dtype=np.int64)
    grid[cells[:, 0], cells[:, 1]] = labels
    empty = grid < 0
    out, count = [], 0
    for d in range(10):
        mask = grid == d
        if not mask.any():
            continue
        parts, _ = ndimage.label(mask | (empty & ndimage.binary_dilation(mask, structure=EIGHT)), structure=EIGHT)
        sizes = np.bincount(parts[mask])[1:]
        out.append(float(sizes.max() / mask.sum()))
        count += int((sizes >= 5).sum())
    return float(np.mean(out)), count


def purity(cells, labels, picture, k=10, sample=4000):
    rng = np.random.default_rng(0)
    queries = np.arange(cells.shape[0]) if cells.shape[0] <= sample else np.sort(rng.choice(cells.shape[0], sample, replace=False))
    tree = cKDTree(cells.astype(np.float64))
    _, near = tree.query(cells[queries].astype(np.float64), k=min(k * 6, cells.shape[0]))
    pure = np.zeros(queries.shape[0])
    for row, q in enumerate(queries):
        cand = near[row]
        cand = cand[picture[cand] != picture[q]][:k]
        pure[row] = (labels[cand] == labels[q]).mean() if cand.size else 0.0
    return float(pure.mean())


def smoothness(cells, angle, period, height, width):
    """Гладкость карты углов: средняя разность угла у соседних по стороне занятых клеток, в градусах (меньше — глаже)."""
    grid = np.full((int(height), int(width)), np.nan)
    grid[cells[:, 0], cells[:, 1]] = angle
    total, count = 0.0, 0
    for a, b in ((grid[:, :-1], grid[:, 1:]), (grid[:-1, :], grid[1:, :])):
        both = ~np.isnan(a) & ~np.isnan(b)
        diff = np.abs(a[both] - b[both]) % period
        diff = np.minimum(diff, period - diff)
        total += float(diff.sum())
        count += int(both.sum())
    return total / max(1, count)


def levels_of(z):
    return sorted(int(k[1:].split("_")[0]) for k in z.files if k.endswith("_bits"))


def embed(path, method):
    z = np.load(path)
    saved = {}
    for n in levels_of(z):
        began = time.time()
        bits = np.unpackbits(z[f"L{n}_bits"], axis=1)[:, : int(z[f"L{n}_length"])].astype(np.float32)
        height, width = [int(v) for v in z[f"L{n}_size"]]
        if method == "tsne":
            from sklearn.manifold import TSNE
            xy = TSNE(n_components=2, metric="cosine", perplexity=30, init="random", random_state=0, n_jobs=6).fit_transform(bits)
        else:
            import umap
            xy = umap.UMAP(n_components=2, metric="cosine", n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(bits)
        saved[f"L{n}_cells"] = snap(xy, height, width)
        np.savez_compressed(path[:-4] + f".{method}.npz", **saved)
        print(f"{method} ур. {n}: точек {bits.shape[0]} ({time.time() - began:.0f} с)", flush=True)


def draw(path, out, caption):
    from PIL import Image, ImageDraw, ImageFont
    colours = digit_colours()
    z = np.load(path)
    found = {"наша раскладка": {f"L{n}_cells": z[f"L{n}_cells"] for n in levels_of(z)}}
    for method, title in (("tsne", "t-SNE"), ("umap", "UMAP")):
        extra = path[:-4] + f".{method}.npz"
        if os.path.exists(extra):
            found[title] = dict(np.load(extra))
    levels = levels_of(z)
    side, pad, top, gap = 330, 16, 92, 60
    try:
        big, mid, small = ImageFont.truetype("arialbd.ttf", 22), ImageFont.truetype("arialbd.ttf", 15), ImageFont.truetype("arial.ttf", 13)
    except OSError:
        big = mid = small = ImageFont.load_default()
    sheet = Image.new("RGB", (pad + len(levels) * (side + pad), top + len(found) * (side + gap) + 46), "black")
    pen = ImageDraw.Draw(sheet)
    pen.text((pad, 10), caption, fill="white", font=big)
    x = pad
    for d in range(10):
        pen.rectangle([x, 46, x + 16, 62], fill=colours[d])
        pen.text((x + 22, 46), str(d), fill="white", font=mid)
        x += 50
    period = float(z["L0_period"]) if "L0_period" in z.files else 180.0
    bar = 180
    for i in range(bar):
        r, g, b = colorsys.hls_to_rgb(i / bar, 0.60, 0.95)
        pen.line((x + 60 + i, 46, x + 60 + i, 62), fill=(int(r * 255), int(g * 255), int(b * 255)))
    pen.text((x + 30, 46), "0°", fill="white", font=small)
    pen.text((x + 66 + bar, 46), f"{period:.0f}° — угол штриха (уровень 0)", fill="white", font=small)
    report = {}
    for line, (title, cells_of) in enumerate(found.items()):
        for column, n in enumerate(levels):
            if f"L{n}_cells" not in cells_of:
                continue
            cells = np.asarray(cells_of[f"L{n}_cells"], dtype=np.int64)
            labels, picture = z[f"L{n}_labels"], z[f"L{n}_picture"]
            height, width = [int(v) for v in z[f"L{n}_size"]]
            if COMPACT and title != "наша раскладка":
                cells = compact(cells.astype(np.float64), height, width)  # без пустых клеток внутри
            left, y = pad + column * (side + pad), top + line * (side + gap)
            grid = np.zeros((height, width, 3), dtype=np.uint8)
            if n == 0:
                angle = z["L0_angle"]
                fills = np.asarray([[int(c * 255) for c in colorsys.hls_to_rgb((float(a) / period) % 1.0, 0.60, 0.95)] for a in angle], dtype=np.uint8)
                note = f"гладкость {smoothness(cells, angle, period, height, width):.1f}°   честная {purity(cells, labels, picture):.3f}"
                report[f"{title} ур. 0"] = note
            else:
                fills = np.asarray(colours, dtype=np.uint8)[labels]
                value, patches = m1u(cells, labels, height, width)
                honest = purity(cells, labels, picture)
                note = f"М1u {value:.3f}   честная {honest:.3f}   пятен {patches}"
                report[f"{title} ур. {n}"] = {"m1u": round(value, 3), "honest": round(honest, 3), "patches": patches}
            grid[cells[:, 0], cells[:, 1]] = fills
            scale = max(1, side // width)
            image = Image.fromarray(grid).resize((width * scale, height * scale), Image.NEAREST)
            sheet.paste(image, (left + (side - image.width) // 2, y))
            pen.text((left, y - 20), f"ур. {n} · {title} · точек {labels.shape[0]}", fill="white", font=mid)
            pen.text((left, y + side + 4), note, fill=(210, 210, 210), font=small)
    sheet.save(out)
    json.dump(report, open(out[:-4] + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for key, value in report.items():
        print(key, value)
    print("рисунок:", out)


if __name__ == "__main__":
    if sys.argv[1] == "embed":
        embed(sys.argv[2], sys.argv[3])
    else:
        rest = [a for a in sys.argv[4:] if a != "--compact"]
        draw(sys.argv[2], sys.argv[3], rest[0] if rest else "Карты уровней 0–5 на одних и тех же кодах: наша раскладка, t-SNE, UMAP")
