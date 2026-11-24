# -*- coding: utf-8 -*-
"""Сколько класса в «мешке детекторов» картинки при разном σ: для каждой картинки — набор детекторов карты уровня N,
зажжённых её точками (у точки оставлены σ сильнейших), без мест. Мерка: ближайшая по набору картинка — своего класса?
(исключая саму картинку; косинус; наборы — двоичные и со счётом). Процессор."""
import sys

import numpy as np

z = np.load(sys.argv[1])
print("ур.  σ      детекторов/картинку  ближайшая своя (двоичн.)  свои из 5 (двоичн.)  ближайшая своя (счёт)  свои из 5 (счёт)   по цифрам, свои из 5 (счёт)")
for n in (4, 5):
    labels, picture = z[f"L{n}_labels"], z[f"L{n}_picture"]
    row, det, level = z[f"L{n}_fire_row"], z[f"L{n}_fire_det"], z[f"L{n}_fire_level"]
    count = z[f"L{n}_det_radius"].shape[0]
    order = np.lexsort((-level, row))
    row, det, level = row[order], det[order], level[order]
    rank = np.arange(row.shape[0]) - np.searchsorted(row, row)
    ids = np.unique(picture)
    index = {p: i for i, p in enumerate(ids)}
    column = np.array([index[p] for p in picture])
    label_of = np.array([labels[picture == p][0] for p in ids])
    for keep in (1, 2, 3, 4, 6, 9, 13, 20, 10 ** 6):
        use = rank < keep
        bag = np.zeros((ids.size, count), dtype=np.float32)
        np.add.at(bag, (column[row[use]], det[use]), 1.0)
        out = []
        for kind in ("binary", "count"):
            x = (bag > 0).astype(np.float32) if kind == "binary" else bag
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)
            sim = x @ x.T
            np.fill_diagonal(sim, -1)
            near = np.argsort(-sim, axis=1)[:, :5]
            hit = label_of[near] == label_of[:, None]
            out.append((hit[:, 0].mean(), hit.mean(), [hit[label_of == d].mean() for d in range(10)]))
        name = "все" if keep > 1000 else str(keep)
        print(f"{n:3d}  {name:5s}  {(bag > 0).sum(axis=1).mean():19.1f}  {out[0][0]:24.3f}  {out[0][1]:19.3f}  {out[1][0]:21.3f}  {out[1][1]:16.3f}   "
              + " ".join(f"{v:.2f}" for v in out[1][2]))
