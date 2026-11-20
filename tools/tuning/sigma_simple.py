# -*- coding: utf-8 -*-
"""Инфографика простыми словами: что такое σ (ACTIVATION_SATURATION_LIMIT)."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = sys.argv[1]
INK, SOFT, MUTED, PAPER = "#1f1d1a", "#57534e", "#a8a29e", "#fbfaf7"
LIT, LIT2, OFF, CARD = "#e8590c", "#fcc419", "#d6d3d1", "#ffffff"
rng = np.random.default_rng(3)

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(17, 9.2), dpi=130)
fig.patch.set_facecolor(PAPER)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 170); ax.set_ylim(0, 92); ax.axis("off")

ax.text(4, 86, "Что такое σ", fontsize=30, fontweight="bold", color=INK, va="center")
ax.text(4, 80, "σ — сколько самых ярких «лампочек» берём, чтобы описать кусок цифры", fontsize=16, color=SOFT, va="center")

cards = [(4, "1", "Окно смотрит\nна кусок цифры"), (37, "2", "Память узнаёт:\nзагораются похожие места"),
         (70, "3", "Над памятью — лампочки\n(детекторы), каждая горит\nсо своей силой"),
         (103, "4", "Берём σ самых ярких,\nостальные выключаем"), (136, "5", "Каждая взятая лампочка\nзажигает свой бит —\nвот и код куска")]
for x0, number, text in cards:
    ax.add_patch(FancyBboxPatch((x0, 22), 30, 52, boxstyle="round,pad=0.4,rounding_size=2.2", fc=CARD, ec="#e7e5e4", lw=1.2))
    ax.add_patch(Circle((x0 + 4.2, 69.5), 2.6, color=INK))
    ax.text(x0 + 4.2, 69.5, number, color="white", fontsize=13, fontweight="bold", ha="center", va="center")
    ax.text(x0 + 15, 30.5, text, fontsize=11.5, color=INK, ha="center", va="center", linespacing=1.35)
for x0 in (34.6, 67.6, 100.6, 133.6):
    ax.add_patch(FancyArrowPatch((x0, 50), (x0 + 2.8, 50), arrowstyle="-|>", mutation_scale=18, color=MUTED, lw=1.6))

# 1. Пятёрка и окно.
ax.plot([21, 12], [64, 64], color=INK, lw=7, solid_capstyle="round")
ax.plot([12, 12.4], [64, 56.3], color=INK, lw=7, solid_capstyle="round")
arc = np.radians(np.linspace(130, -140, 160))
ax.plot(16 + 5.6 * np.cos(arc), 52 + 5.6 * np.sin(arc), color=INK, lw=7, solid_capstyle="round")
ax.add_patch(Circle((18.5, 52), 6.2, fill=False, ec=LIT, lw=3))
ax.text(26.2, 59.5, "окно", color=LIT, fontsize=11, fontweight="bold")

# 2. Карта памяти: серые точки, тёплое пятно.
gx, gy = np.meshgrid(np.arange(40, 64.1, 1.6), np.arange(40, 64.1, 1.6))
gx, gy = gx.ravel(), gy.ravel()
glow = np.exp(-((gx - 50) ** 2 + (gy - 53) ** 2) / 18) + 0.6 * np.exp(-((gx - 57) ** 2 + (gy - 46) ** 2) / 8)
colours = [LIT if g > 0.55 else (LIT2 if g > 0.2 else OFF) for g in glow]
ax.scatter(gx, gy, s=34, c=colours, marker="s", linewidths=0)
ax.text(52, 37.2, "карта памяти уровня", color=MUTED, fontsize=10, ha="center")

# 3. Лампочки над картой: круги разной яркости.
lamps = [(81, 57, 0.95), (88, 55, 0.80), (84, 50, 0.65), (92, 49, 0.55), (78, 47, 0.45), (86, 44, 0.40),
         (94, 58, 0.30), (79, 62, 0.25), (95, 43, 0.20)]
ax.scatter(*np.meshgrid(np.arange(74, 98.1, 2.0), np.arange(40, 66.1, 2.0)), s=6, color=OFF, linewidths=0)
for x, y, power in lamps:
    ax.add_patch(Circle((x, y), 2.9, color=LIT, alpha=0.12 + 0.75 * power, lw=0))
    ax.add_patch(Circle((x, y), 2.9, fill=False, ec=LIT, lw=0.8 + 2.5 * power))
ax.text(86, 37.2, "чем ярче — тем сильнее похоже", color=MUTED, fontsize=10, ha="center")

# 4. Отбор σ самых ярких: лампочки в ряд по яркости.
powers = sorted((p for _, _, p in lamps), reverse=True)
sigma = 4
for i, power in enumerate(powers):
    x = 106.5 + i * 2.85
    keep = i < sigma
    ax.add_patch(Rectangle((x - 1.0, 41), 2.0, 17 * power, color=LIT if keep else OFF))
    if not keep:
        ax.plot([x - 1.0, x + 1.0], [42 + 20 * power + 1.2, 42 + 20 * power + 1.2], color=MUTED, lw=0)
ax.plot([105.4, 105.4, 106.5 + (sigma - 1) * 2.85 + 1.1, 106.5 + (sigma - 1) * 2.85 + 1.1], [58.5, 60, 60, 58.5], color=INK, lw=1.6)
ax.text(106.5 + (sigma - 1) * 2.85 / 2, 61.6, f"σ = {sigma}", color=INK, fontsize=13, fontweight="bold", ha="center")
ax.text(120.5, 39.5, "по яркости, слева — самые яркие", color=MUTED, fontsize=10, ha="center")

# 5. Код: полоса бит, σ единиц.
bits = np.zeros(48, dtype=int)
on = rng.choice(48, size=sigma, replace=False)
bits[on] = 1
for i, bit in enumerate(bits):
    row, col = divmod(i, 8)
    ax.add_patch(Rectangle((140.5 + col * 2.7, 61 - row * 3.4), 2.3, 2.9, color=LIT if bit else "#efede9"))
word = "единица" if sigma % 10 == 1 and sigma % 100 != 11 else ("единицы" if sigma % 10 in (2, 3, 4) and sigma % 100 not in (12, 13, 14) else "единиц")
ax.text(151, 39.5, f"{sigma} {word} из {bits.size} бит —\nкод куска цифры", color=MUTED, fontsize=10, ha="center", linespacing=1.3)

# Низ: что меняет σ — два примера.
ax.add_patch(FancyBboxPatch((4, 3), 162, 14.5, boxstyle="round,pad=0.4,rounding_size=2", fc="#f5f3ef", ec="#e7e5e4"))
ax.text(8, 13.6, "Мало σ", fontsize=14, fontweight="bold", color=INK)
ax.text(8, 7.6, "в коде только самое главное:\nгрубо, но разные почерки одной цифры\nчаще дают одинаковый код", fontsize=11, color=SOFT, linespacing=1.3, va="center")
ax.text(62, 13.6, "Много σ", fontsize=14, fontweight="bold", color=INK)
ax.text(62, 7.6, "код подробный: видно больше оттенков,\nно попадают и слабые, почти случайные\nлампочки — коды «шумят»", fontsize=11, color=SOFT, linespacing=1.3, va="center")
ax.text(117, 13.6, "У нас сейчас", fontsize=14, fontweight="bold", color=LIT)
ax.text(117, 7.6, "на кусок загорается 80–90 лампочек,\nберём σ = 15; в статье — около 50\n(§7.1.12)", fontsize=11, color=SOFT, linespacing=1.3, va="center")

fig.savefig(OUT, facecolor=PAPER)
print(OUT)
