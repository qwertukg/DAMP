# -*- coding: utf-8 -*-
"""Гладкая расстановка точек уровня (t-SNE по кодам, без меток), посаженная на сетку карты: клетки → data/top-stand/tsne-L<уровень>-seed<зерно>.npz.
Системный Python со scikit-learn, только процессор.  python tools/tuning/top/tsne_cells.py [зёрна=1,3,0,2,4,5] [уровень=5] [запуски=0,1]"""
import os
import sys
import warnings

import numpy as np
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")
sys.setrecursionlimit(100000)
FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "top-stand")


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


seeds = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "1,3,0,2,4,5").split(",")]
level = int(sys.argv[2]) if len(sys.argv) > 2 else 5
states = [int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "0,1").split(",")]
for seed in seeds:
    z = np.load(os.path.join(FOLDER, f"dump-seed{seed}.npz"))
    bits = np.unpackbits(z[f"L{level}_bits"], axis=1)[:, : int(z[f"L{level}_length"])].astype(np.float32)
    height, width = [int(v) for v in z[f"L{level}_size"]]
    saved = {}
    method = sys.argv[4] if len(sys.argv) > 4 else "tsne"  # tsne | umap15 | umap30
    for state in states:
        if method == "tsne":
            xy = TSNE(n_components=2, metric="cosine", perplexity=30, init="random", random_state=state, n_jobs=4).fit_transform(bits)
        else:
            import umap
            xy = umap.UMAP(n_components=2, metric="cosine", n_neighbors=int(method[4:]), min_dist=0.3, random_state=state).fit_transform(bits)
        saved[f"cells{state}"] = snap(xy, height, width)
    np.savez_compressed(os.path.join(FOLDER, f"{method}-L{level}-seed{seed}.npz"), **saved)
    print(f"зерно {seed}: клетки гладкой расстановки ({method}) записаны ({len(states)} запуска)", flush=True)
