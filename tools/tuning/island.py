# -*- coding: utf-8 -*-
"""Что лежит в островке карты уровня 0: углопозиции (угол и место штриха относительно центра цифры), цифры, картинки.

    island.py <data/maps/tune2-N-level0.npz> <выход png>

Островок — точки ниже трещины: трещина — пустые клетки внутри карты (сверху и снизу в том же столбце есть точки),
собранные в длинную горизонтальную полосу; граница островка — прямая по трещине, продолженная на всю ширину.
"""
import colorsys
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

data = np.load(sys.argv[1])
out = sys.argv[2]
cells = data["cells"].astype(int)
angle = data["angle"].astype(float)
y, x = data["y"].astype(float), data["x"].astype(float)
labels, picture = data["labels"].astype(int), data["picture"].astype(int)
height, width, period = int(data["height"]), int(data["width"]), float(data["period"])

occupied = np.zeros((height, width), dtype=bool)
occupied[cells[:, 0], cells[:, 1]] = True
# Пустые клетки внутри карты по столбцу: выше и ниже в том же столбце есть точки.
inner = np.zeros_like(occupied)
for col in range(width):
    rows = np.flatnonzero(occupied[:, col])
    if rows.size >= 2:
        inner[rows.min():rows.max() + 1, col] = ~occupied[rows.min():rows.max() + 1, col]
parts, count = ndimage.label(inner, structure=np.ones((3, 3)))
sizes = ndimage.sum(inner, parts, index=np.arange(1, count + 1)) if count else np.zeros(0)
spans = [np.ptp(np.flatnonzero(np.any(parts == k + 1, axis=0))) + 1 for k in range(count)]
crack_id = int(np.argmax(spans)) + 1  # самая длинная по ширине полоса пустых клеток — трещина
crack = np.argwhere(parts == crack_id)
fit = np.polyfit(crack[:, 1].astype(float), crack[:, 0].astype(float), 1)  # строка трещины = a·столбец + b
boundary = np.polyval(fit, cells[:, 1].astype(float))
lower = crack[:, 0].mean() > height / 2
island = cells[:, 0] > boundary if lower else cells[:, 0] < boundary
print(f"карта {height}×{width}, точек {cells.shape[0]}; трещина: {crack.shape[0]} пустых клеток, столбцы "
      f"{crack[:, 1].min()}–{crack[:, 1].max()}, строка ≈{crack[:, 0].mean():.1f}; в островке точек {island.sum()}")

# Место штриха относительно центра своей цифры (центр — среднее мест штрихов той же картинки на карте).
cy = np.zeros(picture.max() + 1); cx = np.zeros(picture.max() + 1); n = np.zeros(picture.max() + 1)
valid = picture >= 0
np.add.at(cy, picture[valid], y[valid]); np.add.at(cx, picture[valid], x[valid]); np.add.at(n, picture[valid], 1)
dy = y - cy[np.maximum(picture, 0)] / np.maximum(n[np.maximum(picture, 0)], 1)
dx = x - cx[np.maximum(picture, 0)] / np.maximum(n[np.maximum(picture, 0)], 1)


def colour(a: float) -> tuple:
    return colorsys.hls_to_rgb((a / period) % 1.0, 0.6, 0.95)


SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#8a8984"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig = plt.figure(figsize=(16, 9.6), dpi=130)
fig.patch.set_facecolor(SURFACE)
grid = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.38, wspace=0.28)

# 1. Карта уровня 0 в цветах пинвила, островок обведён, остальное притушено.
ax = fig.add_subplot(grid[0, 0:2])
rgb = np.array([colour(a) for a in angle])
shade = np.where(island[:, None], rgb, 0.35 * rgb + 0.65)
ax.scatter(cells[:, 1], cells[:, 0], c=shade, s=4.2, marker="s", linewidths=0)
ax.scatter(crack[:, 1], crack[:, 0], c="#111111", s=4.2, marker="s", linewidths=0)
xs = np.arange(width)
ax.plot(xs, np.polyval(fit, xs), color="#111111", lw=0.8, ls="--")
ax.set_xlim(-1, width); ax.set_ylim(height, -1); ax.set_aspect("equal"); ax.axis("off")
ax.set_title(f"Карта уровня 0: островок ярко, остальное бледно;\n"
             f"чёрное — трещина ({crack.shape[0]} пустых клеток), пунктир — граница",
             loc="left", fontsize=10.5, color=INK)

# 2. Где на цифре эти штрихи: место относительно центра цифры (клетки сетки картинки), отрезок — угол.
ax = fig.add_subplot(grid[0, 2:4])
ax.scatter(dx[~island], dy[~island], s=2, color="#d8d7d2", linewidths=0, label="остальная карта")
pick = np.flatnonzero(island)
for i in pick:
    t = np.radians(angle[i])
    ax.plot([dx[i] - 0.35 * np.cos(t), dx[i] + 0.35 * np.cos(t)], [dy[i] + 0.35 * np.sin(t), dy[i] - 0.35 * np.sin(t)],
            color=colour(angle[i]), lw=1.4, solid_capstyle="round")
ax.set_aspect("equal"); ax.invert_yaxis()
ax.axhline(0, color=MUTED, lw=0.6); ax.axvline(0, color=MUTED, lw=0.6)
ax.set_xlabel("место по горизонтали от центра цифры, клеток"); ax.set_ylabel("по вертикали, клеток (вниз — +)")
ax.set_title("Углопозиции островка: место штриха от центра цифры,\n"
             "отрезок — его угол (цвет — как на карте); серое — вся карта",
             loc="left", fontsize=10.5, color=INK)

# 3. Углы: островок против остальной карты.
ax = fig.add_subplot(grid[1, 0])
bins = np.linspace(0, period, 19)
h_in = np.histogram(angle[island], bins)[0] / max(1, island.sum())
h_out = np.histogram(angle[~island], bins)[0] / max(1, (~island).sum())
mid = 0.5 * (bins[:-1] + bins[1:])
ax.bar(mid, h_out, width=period / 18 * 0.9, color="#d8d7d2", label="остальная карта")
ax.bar(mid, h_in, width=period / 18 * 0.45, color=[colour(m) for m in mid], label="островок")
ax.set_xlabel("угол штриха, °"); ax.set_ylabel("доля точек"); ax.legend(frameon=False, fontsize=8.5)
ax.set_title("Углы", loc="left", fontsize=10.5)

# 4. Место: расстояние от центра цифры и направление.
ax = fig.add_subplot(grid[1, 1])
r_in, r_out = np.hypot(dx[island], dy[island]), np.hypot(dx[~island], dy[~island])
rb = np.linspace(0, max(r_in.max(), r_out.max()), 16)
ax.hist(r_out, rb, density=True, color="#d8d7d2", label="остальная карта")
ax.hist(r_in, rb, density=True, histtype="step", color="#2a78d6", lw=2, label="островок")
ax.set_xlabel("расстояние штриха от центра цифры, клеток"); ax.legend(frameon=False, fontsize=8.5)
ax.set_title("Место: далеко ли от центра", loc="left", fontsize=10.5)

# 5. Сторона цифры: где штрих относительно центра (8 секторов).
ax = fig.add_subplot(grid[1, 2])
names = ["справа", "справа-снизу", "снизу", "слева-снизу", "слева", "слева-сверху", "сверху", "справа-сверху"]
sector = ((np.degrees(np.arctan2(dy, dx)) + 22.5) % 360 // 45).astype(int)
s_in = np.bincount(sector[island], minlength=8) / max(1, island.sum())
s_out = np.bincount(sector[~island], minlength=8) / max(1, (~island).sum())
pos = np.arange(8)
ax.barh(pos + 0.2, s_out, height=0.38, color="#d8d7d2", label="остальная карта")
ax.barh(pos - 0.2, s_in, height=0.38, color="#2a78d6", label="островок")
ax.set_yticks(pos, names); ax.invert_yaxis(); ax.legend(frameon=False, fontsize=8.5)
ax.set_title("Сторона цифры", loc="left", fontsize=10.5)

# 6. Цифры.
ax = fig.add_subplot(grid[1, 3])
d_in = np.bincount(labels[island], minlength=10) / max(1, island.sum())
d_out = np.bincount(labels[~island], minlength=10) / max(1, (~island).sum())
pos = np.arange(10)
ax.bar(pos - 0.2, d_out, width=0.38, color="#d8d7d2", label="остальная карта")
ax.bar(pos + 0.2, d_in, width=0.38, color="#2a78d6", label="островок")
ax.set_xticks(pos); ax.set_xlabel("цифра"); ax.legend(frameon=False, fontsize=8.5)
ax.set_title(f"Цифры (картинок в островке: {np.unique(picture[island]).size} из {np.unique(picture).size})",
             loc="left", fontsize=10.5)

fig.suptitle(f"Островок карты уровня 0: {island.sum()} точек из {cells.shape[0]}", x=0.06, ha="left",
             fontsize=14, fontweight="bold", color=INK)
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print(out)
print("углы, медиана: островок %.0f°, остальное %.0f°" % (np.median(angle[island]), np.median(angle[~island])))
print("расстояние от центра, медиана: островок %.2f, остальное %.2f" % (np.median(r_in), np.median(r_out)))
print("стороны (островок):", {names[k]: round(float(s_in[k]), 3) for k in range(8)})
print("стороны (остальное):", {names[k]: round(float(s_out[k]), 3) for k in range(8)})
print("цифры (островок):", {k: round(float(d_in[k]), 3) for k in range(10)})
