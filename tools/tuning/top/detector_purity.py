# -*- coding: utf-8 -*-
"""То же, но у каждой точки оставлены только S сильнейших детекторов (S = 3, 6, 13, все). Плюс: сколько детекторов из
зажжённых точкой лежат рядом с её собственной клеткой (свой участок карты) и какова чистота «своих» и «дальних»."""
import sys

import numpy as np

z = np.load(sys.argv[1])
for n in [k for k in (2, 3, 4, 5) if f"L{k}_fire_row" in z.files]:
    labels, cells = z[f"L{n}_labels"], z[f"L{n}_cells"].astype(np.float64)
    row, det, level = z[f"L{n}_fire_row"], z[f"L{n}_fire_det"], z[f"L{n}_fire_level"]
    radius, centre = z[f"L{n}_det_radius"], z[f"L{n}_det_centre"]
    count = radius.shape[0]
    print(f"\nур. {n}: точек {labels.shape[0]}, детекторов {count}; центр детектора в пределах {centre.min():.1f}…{centre.max():.1f}")
    order = np.lexsort((-level, row))
    row, det, level = row[order], det[order], level[order]
    rank = np.arange(row.shape[0]) - np.searchsorted(row, row)          # место детектора у своей точки: 0 — сильнейший
    far = np.sqrt(((cells[row] - centre[det]) ** 2).sum(axis=1)) / radius[det]   # расстояние точки до центра детектора в его радиусах
    print("  отбор     срабатываний/точку  чистота детектора  своего класса: " + " ".join(f"{d:4d}" for d in range(10)) + "  среднее  доля «своих» (точка внутри 1.5 радиуса)")
    for keep in (1, 3, 6, 13, 10 ** 6):
        use = rank < keep
        table = np.zeros((count, 10))
        np.add.at(table, (det[use], labels[row[use]]), 1.0)
        total = table.sum(axis=1)
        share = table / np.maximum(total[:, None], 1)
        pure = np.average(share.max(axis=1)[total > 0], weights=total[total > 0])
        own = share[det[use], labels[row[use]]]
        by = [own[labels[row[use]] == d].mean() for d in range(10)]
        name = "все" if keep > 1000 else f"{keep} сильн."
        print(f"  {name:9s} {use.sum() / labels.shape[0]:18.1f}  {pure:17.3f}                 " + " ".join(f"{v:4.2f}" for v in by) + f"  {np.mean(by):7.3f}  {(far[use] <= 1.5).mean():.2f}")
    near = far <= 1.5
    print(f"  расстояние до центра детектора (в радиусах): медиана {np.median(far):.1f}; у сильнейшего {np.median(far[rank == 0]):.1f}; "
          f"уровень срабатывания: свои {level[near].mean():.2f}, дальние {level[~near].mean():.2f}")
