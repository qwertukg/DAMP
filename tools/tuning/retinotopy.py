# -*- coding: utf-8 -*-
"""Карта уровня 0 как ретинотопическая карта: место штриха (ретинотопия) и угол (пинвилы) — рядом, по прогонам.

    retinotopy.py <выход png> <прогон> ...

Место штриха — относительно центра своей цифры (среднее мест штрихов той же картинки на карте). Цвет «ретинотопии» —
направление места от центра (оттенок) и расстояние (яркость: у центра светлее). Мерки: средняя разница мест соседних
точек карты, клеток (меньше — ретинотопия глаже) и топографическая связь — корреляция Спирмена между расстоянием по
карте и расстоянием мест у случайных пар (ближе к 1 — дальний порядок мест на карте сохранён).
"""
import colorsys
import io
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1]
runs = [int(v) for v in sys.argv[2:]]
log = {json.loads(l)["run"]: json.loads(l) for l in io.open(os.path.join(HERE, "tune2_log.jsonl"), encoding="utf-8")}


def place_of(data):  # noqa: ANN001, ANN201
    y, x, picture = data["y"].astype(float), data["x"].astype(float), data["picture"].astype(int)
    count = picture.max() + 1
    sy, sx, n = np.zeros(count), np.zeros(count), np.zeros(count)
    ok = picture >= 0
    np.add.at(sy, picture[ok], y[ok]); np.add.at(sx, picture[ok], x[ok]); np.add.at(n, picture[ok], 1)
    p = np.maximum(picture, 0)
    return y - sy[p] / np.maximum(n[p], 1), x - sx[p] / np.maximum(n[p], 1)


def measures(cells, dy, dx, angle, period):  # noqa: ANN001, ANN201
    height, width = cells[:, 0].max() + 1, cells[:, 1].max() + 1
    index = np.full((height, width), -1)
    index[cells[:, 0], cells[:, 1]] = np.arange(cells.shape[0])
    place_diff, angle_diff = [], []
    for a, b in ((index[:, :-1], index[:, 1:]), (index[:-1, :], index[1:, :])):
        held = (a >= 0) & (b >= 0)
        i, j = a[held], b[held]
        place_diff.append(np.hypot(dy[i] - dy[j], dx[i] - dx[j]))
        d = np.abs(angle[i] - angle[j]) % period
        angle_diff.append(np.minimum(d, period - d))
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, cells.shape[0], 20000), rng.integers(0, cells.shape[0], 20000)
    map_d = np.hypot(*(cells[i] - cells[j]).T)
    place_d = np.hypot(dy[i] - dy[j], dx[i] - dx[j])
    rank = lambda v: np.argsort(np.argsort(v))  # noqa: E731
    spearman = float(np.corrcoef(rank(map_d), rank(place_d))[0, 1])
    return float(np.mean(np.concatenate(place_diff))), float(np.mean(np.concatenate(angle_diff))), spearman


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(len(runs), 3, figsize=(15, 5 * len(runs)), dpi=110)
axes = np.atleast_2d(axes)
fig.patch.set_facecolor("#fcfcfb")
rows = []
for row, run in enumerate(runs):
    data = np.load(f"data/maps/tune2-{run}-level0.npz")
    cells, angle, period = data["cells"].astype(int), data["angle"].astype(float), float(data["period"])
    dy, dx = place_of(data)
    place_step, angle_step, spearman = measures(cells, dy, dx, angle, period)
    rows.append((run, place_step, angle_step, spearman))
    reach = np.quantile(np.hypot(dy, dx), 0.98)
    direction = (np.degrees(np.arctan2(-dy, dx)) % 360) / 360
    radius = np.clip(np.hypot(dy, dx) / reach, 0, 1)
    retino = np.array([colorsys.hls_to_rgb(h, 0.85 - 0.45 * r, 0.9) for h, r in zip(direction, radius)])
    pinwheel = np.array([colorsys.hls_to_rgb((a / period) % 1.0, 0.6, 0.95) for a in angle])
    label = log.get(run, {}).get("label", "").replace("250 цифр: ", "")
    for col, (colours, title) in enumerate([
            (retino, f"{run}: ретинотопия — где штрих в цифре\n(оттенок — сторона от центра, темнее — дальше)"),
            (pinwheel, f"пинвилы — угол штриха\nсоседи различаются на {angle_step:.1f}°"),
            (None, f"место соседей различается на {place_step:.2f} клетки\nсвязь «карта ↔ место» (Спирмен) {spearman:.2f}")]):
        ax = axes[row, col]
        if colours is None:
            ax.scatter(dx, dy, c=retino, s=2, linewidths=0)
            ax.set_aspect("equal"); ax.invert_yaxis()
            ax.set_xlabel("место по горизонтали от центра цифры"); ax.set_ylabel("по вертикали (вниз +)")
            ax.set_title(title + "\n(те же цвета на плоскости цифры)", loc="left", fontsize=9.5)
            continue
        ax.scatter(cells[:, 1], cells[:, 0], c=colours, s=4.2, marker="s", linewidths=0)
        ax.set_xlim(-1, cells[:, 1].max() + 1); ax.set_ylim(cells[:, 0].max() + 1, -1); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, loc="left", fontsize=9.5)
    axes[row, 0].text(0, -0.04, label, transform=axes[row, 0].transAxes, fontsize=8.5, color="#57534e", va="top")
fig.suptitle("Карта уровня 0 как ретинотопическая карта: место плавно, угол — пинвилами поверх", x=0.04, ha="left",
             fontsize=14, fontweight="bold")
fig.savefig(out, bbox_inches="tight", facecolor="#fcfcfb")
print(out)
for r in rows:
    print(r)
