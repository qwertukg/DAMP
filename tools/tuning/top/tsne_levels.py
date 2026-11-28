"""Карты всех уровней: наша раскладка и t-SNE на тех же кодах (без меток), посаженный на ту же сетку. Цвет — цифра.
python tsne_levels.py <папка выгрузки> <зерно> <файл картинки>  (только процессор)"""
import json
import sys
import time
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.spatial import cKDTree
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")
sys.setrecursionlimit(100000)
EIGHT = np.ones((3, 3), bool)
COLOURS = [tuple(int(c) for c in Image.new("HSV", (1, 1), (int(d * 25.5), 220, 255)).convert("RGB").getpixel((0, 0))) for d in range(10)]


def m1_by_digit(cells, labels, height, width):
    grid = np.full((int(height), int(width)), -1, dtype=np.int64)
    grid[cells[:, 0], cells[:, 1]] = labels
    empty = grid < 0
    out, count = [], 0
    for d in range(10):
        mask = grid == d
        parts, _ = ndimage.label(mask | (empty & ndimage.binary_dilation(mask, structure=EIGHT)), structure=EIGHT)
        sizes = np.bincount(parts[mask])[1:]
        out.append(float(sizes.max() / mask.sum()))
        count += int((sizes >= 5).sum())
    return out, count


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


def snap(xy, height, width):
    """Точки плоскости → клетки сетки без повторов: рекурсивное деление пополам (быстро, соседство сохраняется)."""
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


def panel(cells, labels, height, width, side):
    scale = max(1, side // int(width))
    grid = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    grid[cells[:, 0], cells[:, 1]] = np.asarray(COLOURS, dtype=np.uint8)[labels]
    image = Image.fromarray(grid).resize((int(width) * scale, int(height) * scale), Image.NEAREST)
    return image


def main():
    folder, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    z = np.load(f"{folder}/dump-seed{seed}.npz")
    levels = sorted(int(k[1:].split("_")[0]) for k in z.files if k.endswith("_bits"))
    found, saved = {}, {}
    for n in levels:
        start = time.time()
        bits = np.unpackbits(z[f"L{n}_bits"], axis=1)[:, : int(z[f"L{n}_length"])].astype(np.float32)
        labels, picture, cells = z[f"L{n}_labels"], z[f"L{n}_picture"], z[f"L{n}_cells"]
        height, width = [int(v) for v in z[f"L{n}_size"]]
        xy = TSNE(n_components=2, metric="cosine", perplexity=30, init="random", random_state=0, n_jobs=6).fit_transform(bits)
        ref = snap(xy, height, width)
        row = {}
        for name, c in (("ours", cells), ("tsne", ref)):
            m1, patches = m1_by_digit(c, labels, height, width)
            row[name] = {"m1u": float(np.mean(m1)), "m1": [round(v, 3) for v in m1], "patches": patches, "honest": purity(c, labels, picture)}
        row["points"], row["size"] = int(labels.shape[0]), [height, width]
        found[n] = row
        saved[f"L{n}_tsne_cells"], saved[f"L{n}_tsne_xy"] = ref, xy
        print(f"ур. {n}: точек {row['points']} | наша М1u {row['ours']['m1u']:.3f} честная {row['ours']['honest']:.3f} пятен {row['ours']['patches']} | "
              f"t-SNE М1u {row['tsne']['m1u']:.3f} честная {row['tsne']['honest']:.3f} пятен {row['tsne']['patches']} ({time.time() - start:.0f} с)", flush=True)
        np.savez_compressed(f"{folder}/tsne-levels-seed{seed}.npz", **saved)
        json.dump(found, open(f"{folder}/tsne-levels-seed{seed}.json", "w", encoding="utf-8"), ensure_ascii=False)
    # рисунок: два ряда (наша / t-SNE) × уровни
    side, pad, top, gap = 390, 18, 96, 58
    try:
        big, mid, small = ImageFont.truetype("arialbd.ttf", 22), ImageFont.truetype("arialbd.ttf", 16), ImageFont.truetype("arial.ttf", 14)
    except OSError:
        big = mid = small = ImageFont.load_default()
    sheet = Image.new("RGB", (pad + len(levels) * (side + pad), top + 2 * (side + gap) + 40), "black")
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 12), f"Карты уровней 1–{levels[-1]} на одних и тех же кодах (опора, зерно {seed}): наша раскладка и t-SNE. Цвет — цифра.", fill="white", font=big)
    x = pad
    for d in range(10):
        draw.rectangle([x, 48, x + 18, 66], fill=COLOURS[d])
        draw.text((x + 24, 48), str(d), fill="white", font=mid)
        x += 56
    for column, n in enumerate(levels):
        labels = z[f"L{n}_labels"]
        height, width = found[n]["size"]
        left = pad + column * (side + pad)
        for line, (name, title, c) in enumerate((("ours", "наша раскладка", z[f"L{n}_cells"]), ("tsne", "t-SNE", saved[f"L{n}_tsne_cells"]))):
            y = top + line * (side + gap)
            r = found[n][name]
            draw.text((left, y - 22), f"ур. {n} · {title} · точек {found[n]['points']}", fill="white", font=mid)
            image = panel(c, labels, height, width, side)
            sheet.paste(image, (left + (side - image.width) // 2, y))
            draw.text((left, y + side + 4), f"М1u {r['m1u']:.3f}   честная {r['honest']:.3f}   пятен {r['patches']}", fill=(210, 210, 210), font=small)
    sheet.save(out)
    print("рисунок:", out)


if __name__ == "__main__":
    main()
