# -*- coding: utf-8 -*-
"""Сводка выгрузки по уровням 3–5: точек, точек на картинку, М1u карты, честная чистота карты и кода, чистота детекторов
(13 сильнейших у точки: доля своего класса), сходство «свой класс / чужой». Несколько выгрузок — рядом. Процессор."""
import os
import sys

import numpy as np

sys.setrecursionlimit(100000)
sys.path.insert(0, os.path.join("tools", "tuning", "top"))
from three_maps import m1u, purity  # noqa: E402

K = 10


def code_purity(x, labels, picture):
    hit = np.zeros(x.shape[0])
    for start in range(0, x.shape[0], 512):
        rows = np.arange(start, min(start + 512, x.shape[0]))
        sim = x[rows] @ x.T
        sim[picture[rows][:, None] == picture[None, :]] = -1.0
        near = np.argpartition(-sim, K, axis=1)[:, :K]
        hit[rows] = (labels[near] == labels[rows][:, None]).mean(axis=1)
    return np.array([hit[labels == d].mean() for d in range(10)])


def detectors(z, n, keep=13):
    if f"L{n}_fire_row" not in z.files:
        return None
    labels = z[f"L{n}_labels"]
    row, det, level = z[f"L{n}_fire_row"], z[f"L{n}_fire_det"], z[f"L{n}_fire_level"]
    count = z[f"L{n}_det_radius"].shape[0]
    order = np.lexsort((-level, row))
    row, det = row[order], det[order]
    rank = np.arange(row.shape[0]) - np.searchsorted(row, row)
    use = rank < keep
    table = np.zeros((count, 10))
    np.add.at(table, (det[use], labels[row[use]]), 1.0)
    share = table / np.maximum(table.sum(axis=1, keepdims=True), 1)
    own = share[det[use], labels[row[use]]]
    return count, row.shape[0] / labels.shape[0], np.array([own[labels[row[use]] == d].mean() for d in range(10)])


for path in sys.argv[1:]:
    z = np.load(path)
    print(f"\n=== {os.path.basename(path)}")
    for n in (3, 4, 5):
        if f"L{n}_bits" not in z.files:
            continue
        bits = np.unpackbits(z[f"L{n}_bits"], axis=1)[:, : int(z[f"L{n}_length"])].astype(np.float32)
        labels, picture, cells = z[f"L{n}_labels"], z[f"L{n}_picture"], z[f"L{n}_cells"].astype(np.int64)
        height, width = [int(v) for v in z[f"L{n}_size"]]
        x = bits / np.maximum(np.sqrt(bits.sum(axis=1, keepdims=True)), 1e-6)
        value, patches = m1u(cells, labels, height, width)
        code = code_purity(x, labels, picture)
        print(f"ур. {n}: точек {labels.shape[0]} ({labels.shape[0] / np.unique(picture).size:.1f} на картинку, картинок {np.unique(picture).size}), карта {height}×{width}, "
              f"единиц в коде {bits.sum(axis=1).mean():.0f}; М1u карты {value:.3f}, пятен {patches}, честная карты {purity(cells, labels, picture):.3f}, чистота кода {code.mean():.3f}")
        print("   чистота кода по цифрам:      " + " ".join(f"{v:.2f}" for v in code))
        found = detectors(z, n)
        if found:
            count, per, own = found
            print(f"   детекторов {count}, срабатываний на точку {per:.1f}; своего класса в 13 сильнейших: " + " ".join(f"{v:.2f}" for v in own) + f"  среднее {own.mean():.3f}")
