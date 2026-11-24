# -*- coding: utf-8 -*-
"""Честная чистота идеальной карты при разном числе точек: 10 классов — 10 компактных пятен (центры на решётке 4 × 3 +
малый шум → посадка на сетку делением пополам). Мерка та же, что на стенде (свои среди 10 ближайших с других картинок)."""
import os
import sys

import numpy as np

sys.setrecursionlimit(100000)
sys.path.insert(0, os.path.join("tools", "tuning", "top"))
from three_maps import m1u, purity, snap  # noqa: E402

rng = np.random.default_rng(0)
centres = np.array([[y, x] for y in range(3) for x in range(4)][:10], dtype=float)
for points, per_picture in ((1444, 6), (240, 1)):
    side = int(np.ceil(np.sqrt(points / 0.85)))
    out = []
    for trial in range(5):
        labels = np.repeat(np.arange(10), points // 10 + 1)[:points]
        rng.shuffle(labels)
        picture = np.arange(points) // per_picture
        xy = centres[labels] + rng.normal(0, 0.18, (points, 2))
        cells = snap(xy, side, side)
        out.append((m1u(cells, labels, side, side)[0], purity(cells, labels, picture)))
    out = np.array(out)
    print(f"точек {points} (по {per_picture} на картинку), сетка {side}×{side}: М1u {out[:, 0].mean():.3f}, честная идеальной карты {out[:, 1].mean():.3f}")
