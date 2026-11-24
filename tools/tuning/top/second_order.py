# -*- coding: utf-8 -*-
"""Гладкий старт верха по коду или по общим соседям (второй порядок, §6.2) — без стенда, на выгрузке (CPU).

Для кодов верхнего уровня выгрузки: t-SNE (scikit-learn, косинус, перплексия P) по битам кода и по вектору «мои k ближайших
по коду» (сходство двух точек — доля общих соседей), посадка на сетку карты делением пополам, М1u и честная. Без доводки
движком — сравнение самих стартов. Несколько запусков на вариант.

    python tools/tuning/top/second_order.py <выгрузка.npz> [перплексия=10] [k=10,15,20] [запусков=6]
Системный Python со scikit-learn.
"""
import os
import sys
import warnings

import numpy as np
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")
sys.setrecursionlimit(100000)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from three_maps import m1u, purity, snap  # noqa: E402

z = np.load(sys.argv[1])
perplexity = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
ks = [int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "10,15,20").split(",")]
runs = int(sys.argv[4]) if len(sys.argv) > 4 else 6
top = max(int(k[1:].split("_")[0]) for k in z.files if k.endswith("_bits"))
bits = np.unpackbits(z[f"L{top}_bits"], axis=1)[:, : int(z[f"L{top}_length"])].astype(np.float32)
labels, picture = z[f"L{top}_labels"], z[f"L{top}_picture"]
height, width = [int(v) for v in z[f"L{top}_size"]]
x = bits / np.maximum(np.linalg.norm(bits, axis=1, keepdims=True), 1e-6)
sim = x @ x.T
np.fill_diagonal(sim, -1)
variants = {"код (опора)": bits}
for k in ks:
    near = np.argsort(-sim, axis=1)[:, :k]
    member = np.zeros_like(bits)
    member[np.repeat(np.arange(bits.shape[0]), k), near.ravel()] = 1.0
    member[np.arange(bits.shape[0]), np.arange(bits.shape[0])] = 1.0
    variants[f"общие соседи k {k}"] = member
print(f"ур. {top}: точек {bits.shape[0]}, карта {height}×{width}; перплексия {perplexity}, запусков {runs}")
for name, data in variants.items():
    out = []
    for state in range(runs):
        xy = TSNE(n_components=2, metric="cosine", perplexity=perplexity, init="random", random_state=state, n_jobs=2).fit_transform(data)
        cells = snap(xy, height, width)
        out.append((m1u(cells, labels, height, width)[0], purity(cells, labels, picture)))
    out = np.array(out)
    print(f"{name:20s} М1u {out[:, 0].mean():.3f} ± {out[:, 0].std(ddof=1):.3f}   честная {out[:, 1].mean():.3f}")
