# -*- coding: utf-8 -*-
"""Почему чище код не даёт целее класс: по цифрам — М1 на картах ур. 4 и 5; в пространстве кодов ур. 5 — на сколько кусков
распадается класс (связные части графа «точка — её 5 ближайших своего класса среди 15 ближайших всех»), доля класса в
главной части; и перенос: лежат ли точки ур. 4 одной картинки в одном пятне карты ур. 4. Процессор."""
import os
import sys

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.setrecursionlimit(100000)
EIGHT = np.ones((3, 3), bool)


def patches(cells, labels, height, width):
    grid = np.full((height, width), -1, dtype=np.int64)
    grid[cells[:, 0], cells[:, 1]] = labels
    empty = grid < 0
    patch = np.zeros(labels.shape[0], dtype=np.int64)
    m1 = np.zeros(10)
    offset = 0
    for d in range(10):
        mask = grid == d
        if not mask.any():
            continue
        parts, count = ndimage.label(mask | (empty & ndimage.binary_dilation(mask, structure=EIGHT)), structure=EIGHT)
        own = labels == d
        patch[own] = parts[cells[own, 0], cells[own, 1]] + offset
        offset += count + 1
        sizes = np.bincount(parts[mask])[1:]
        m1[d] = sizes.max() / mask.sum()
    return patch, m1


for path in sys.argv[1:]:
    z = np.load(path)
    print(f"\n=== {os.path.basename(path)}")
    found = {}
    top = max(int(k[1:].split("_")[0]) for k in z.files if k.endswith("_bits"))
    for n in (top - 1, top):
        labels, picture, cells = z[f"L{n}_labels"], z[f"L{n}_picture"], z[f"L{n}_cells"].astype(np.int64)
        height, width = [int(v) for v in z[f"L{n}_size"]]
        patch, m1 = patches(cells, labels, height, width)
        found[n] = (labels, picture, patch)
        print(f"ур. {n} карта, М1 по цифрам:            " + " ".join(f"{v:.2f}" for v in m1) + f"  среднее {m1.mean():.3f}")
        # картинка: доля её точек в самом большом её пятне; доля картинок класса, чьё главное пятно — главное пятно класса
        whole, home = np.zeros(10), np.zeros(10)
        for d in range(10):
            own = labels == d
            main = np.bincount(patch[own]).argmax()
            ids = np.unique(picture[own])
            part = [np.bincount(patch[own & (picture == p)]).max() / (own & (picture == p)).sum() for p in ids]
            where = [np.bincount(patch[own & (picture == p)]).argmax() == main for p in ids]
            whole[d], home[d] = np.mean(part), np.mean(where)
        print("   картинка в одном пятне (доля точек): " + " ".join(f"{v:.2f}" for v in whole) + f"  среднее {whole.mean():.3f}")
        print("   картинок в главном пятне класса:     " + " ".join(f"{v:.2f}" for v in home) + f"  среднее {home.mean():.3f}")
    # пространство кодов верхнего уровня
    bits = np.unpackbits(z[f"L{top}_bits"], axis=1)[:, : int(z[f"L{top}_length"])].astype(np.float32)
    labels, picture = z[f"L{top}_labels"], z[f"L{top}_picture"]
    x = bits / np.maximum(np.sqrt(bits.sum(axis=1, keepdims=True)), 1e-6)
    sim = x @ x.T
    np.fill_diagonal(sim, -1)
    near = np.argsort(-sim, axis=1)[:, :15]
    rows = np.repeat(np.arange(x.shape[0]), 15)
    same = labels[rows] == labels[near.ravel()]
    graph = coo_matrix((np.ones(same.sum()), (rows[same], near.ravel()[same])), shape=(x.shape[0],) * 2)
    _, part = connected_components(graph, directed=False)
    main_share, parts = np.zeros(10), np.zeros(10)
    for d in range(10):
        sizes = np.bincount(part[labels == d])
        sizes = sizes[sizes > 0]
        main_share[d], parts[d] = sizes.max() / sizes.sum(), (sizes >= 5).sum()
    print(f"ур. {top} коды: доля класса в главной связной части: " + " ".join(f"{v:.2f}" for v in main_share) + f"  среднее {main_share.mean():.3f}")
    print(f"ур. {top} коды: связных частей класса (≥ 5 точек):   " + " ".join(f"{int(v):4d}" for v in parts) + f"  всего {int(parts.sum())}")
