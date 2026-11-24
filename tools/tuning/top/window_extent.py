# -*- coding: utf-8 -*-
"""Размер окна уровня против размера цифры: для окон ур. N — радиус окна и смещения элементов (центров точек уровня N−1 той
же картинки) от центра окна в долях радиуса; сколько разных мест (шаг 0.125 радиуса) и сколько узких кругов места (при K
кругах по стороне плоскости −1…1) занимают элементы окна.

    python tools/tuning/top/window_extent.py <выгрузка.npz> [радиусы «4:28.9,5:58.0»] [узких кругов «4:8,5:8»]

Радиус окна — из журнала (`plane_fields.build`, radius_mean): на ур. 4–5 все окна стоят на нижнем пределе радиуса. Без него
берётся extent выгрузки — «сколько клеток видит окно» (элементы вместе с тем, что видят они сами): это не радиус окна."""
import sys

import numpy as np

z = np.load(sys.argv[1])
radii = {int(k): float(v) for k, v in (pair.split(":") for pair in sys.argv[2].split(","))} if len(sys.argv) > 2 else {}
circles = {int(k): int(v) for k, v in (pair.split(":") for pair in sys.argv[3].split(","))} if len(sys.argv) > 3 else {}
print("ур.  окон  радиус окна  полуразмах цифры  элементов в окне  смещение/радиус: медиана, 90 %  разных мест на окно  узких кругов поперёк цифры (медиана)")
for n in (2, 3, 4, 5):
    centre, extent, picture = z[f"L{n}_centre"], z[f"L{n}_extent"], z[f"L{n}_picture"]
    low_centre, low_picture = z[f"L{n - 1}_centre"], z[f"L{n - 1}_picture"]
    by = {p: low_centre[low_picture == p] for p in np.unique(low_picture)}
    rng = np.random.default_rng(0)
    rows = rng.choice(centre.shape[0], min(600, centre.shape[0]), replace=False)
    inside, shift_all, places, across, half = [], [], [], [], []
    k = circles.get(n, 16)
    for r in rows:
        own = by.get(picture[r])
        if own is None or own.shape[0] == 0:
            continue
        radius = radii.get(n, float(extent[r]))
        reach = np.hypot(own[:, 0] - centre[r, 0], own[:, 1] - centre[r, 1])
        seen = own[reach <= radius + 1e-6]
        if seen.shape[0] == 0:
            continue
        shift = (seen - centre[r]) / radius
        inside.append(seen.shape[0])
        shift_all.append(np.hypot(shift[:, 0], shift[:, 1]))
        places.append(np.unique(np.round(shift / 0.125), axis=0).shape[0])
        across.append(max(np.ptp(shift[:, 0]), np.ptp(shift[:, 1])) / (2.0 / k))
        half.append(max(np.ptp(own[:, 0]), np.ptp(own[:, 1])) / 2)
    s = np.concatenate(shift_all)
    shown = f"{radii[n]:.1f}" if n in radii else f"{np.median(extent):.1f}*"
    print(f"{n:3d}  {centre.shape[0]:5d}  {shown:>11s}  {np.median(half):16.2f}  {np.mean(inside):16.1f}  {np.median(s):.2f}, {np.quantile(s, 0.9):.2f}"
          f"                      {np.mean(places):6.1f}               {np.median(across):.1f} (кругов по стороне {k})")
print("* — extent выгрузки, не радиус окна")
