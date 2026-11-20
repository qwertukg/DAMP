# -*- coding: utf-8 -*-
"""Что меняет ширина гауссова ядра (σ = ширина · r): колокол, карта уровня 0 пинвилами, пинвилы крупно, карта верха.

    compare_width.py <выход png> <прогон:ширина> ...   (например 566:0.5 561:1.0 568:1.5)

Карта уровня 0 — из data/maps/tune2-N-level0.npz (стенд пишет её каждым прогоном), карта верха — вырезка из листа
карт data/maps/tune2-N.png (четвёртая панель). Гладкость пинвилов — средняя разница углов соседних точек (по кругу 180°).
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
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1]
runs = [(int(a.split(":")[0]), float(a.split(":")[1])) for a in sys.argv[2:]]
log = {}
for line in io.open(os.path.join(HERE, "tune2_log.jsonl"), encoding="utf-8"):
    entry = json.loads(line)
    log[entry["run"]] = entry
INK, MUTED, SURFACE = "#0b0b0b", "#8a8984", "#fcfcfb"
COLOURS = ["#2196f3", "#4caf50", "#e8590c", "#9c36b5"]


def hue(angle: float, period: float) -> tuple:
    return colorsys.hls_to_rgb((angle / period) % 1.0, 0.6, 0.95)


def smoothness(cells: np.ndarray, angle: np.ndarray, period: float, height: int, width: int) -> float:
    grid = np.full((height, width), np.nan)
    grid[cells[:, 0], cells[:, 1]] = angle
    diffs = []
    for a, b in ((grid[:, :-1], grid[:, 1:]), (grid[:-1, :], grid[1:, :])):
        held = ~np.isnan(a) & ~np.isnan(b)
        d = np.abs(a[held] - b[held]) % period
        diffs.append(np.minimum(d, period - d))
    return float(np.mean(np.concatenate(diffs)))


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig = plt.figure(figsize=(18, 4.6 * len(runs) + 1.2), dpi=110)
fig.patch.set_facecolor(SURFACE)
gs = fig.add_gridspec(len(runs), 4, wspace=0.12, hspace=0.32, width_ratios=[1.2, 1, 1, 1])
for row, ((run, width), colour) in enumerate(zip(runs, COLOURS)):
    data = np.load(f"data/maps/tune2-{run}-level0.npz")
    cells, angle = data["cells"].astype(int), data["angle"].astype(float)
    height, side_w, period = int(data["height"]), int(data["width"]), float(data["period"])
    entry = log[run]
    smooth = smoothness(cells, angle, period, height, side_w)

    ax = fig.add_subplot(gs[row, 0])
    r = 14
    d = np.linspace(0, 20, 300)
    for other_width, other_colour in zip([w for _, w in runs], COLOURS):
        w_ = np.where(d <= r, np.exp(-d ** 2 / (2 * (other_width * r) ** 2)), 0.0)
        ax.plot(d, w_, color=other_colour, lw=3.0 if other_width == width else 1.0, alpha=1.0 if other_width == width else 0.35)
    edge = np.exp(-1 / (2 * width ** 2))
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("расстояние d, клеток (окрестность r = 14)")
    ax.set_ylabel("вес соседа w(d)")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.set_title(f"ширина {width:g}: σ = {width:g}·r, вес на краю {edge:.2f}", loc="left", fontsize=11, color=colour,
                 fontweight="bold")

    ax = fig.add_subplot(gs[row, 1])
    ax.scatter(cells[:, 1], cells[:, 0], c=[hue(a, period) for a in angle], s=4.2, marker="s", linewidths=0)
    ax.set_xlim(-1, side_w); ax.set_ylim(height, -1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"уровень 0 (прогон {run})\nсходимость {entry['level0']['honest']:.3f}", loc="left", fontsize=10)

    ax = fig.add_subplot(gs[row, 2])
    cy, cx = height // 2, side_w // 2
    view = 18
    near = (np.abs(cells[:, 0] - cy) <= view) & (np.abs(cells[:, 1] - cx) <= view)
    ax.scatter(cells[near, 1], cells[near, 0], c=[hue(a, period) for a in angle[near]], s=48, marker="s", linewidths=0)
    ax.set_xlim(cx - view, cx + view); ax.set_ylim(cy + view, cy - view); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"пинвилы крупно: соседи различаются\nв среднем на {smooth:.1f}°", loc="left", fontsize=10)

    ax = fig.add_subplot(gs[row, 3])
    sheet = Image.open(f"data/maps/tune2-{run}.png").convert("RGB")
    panels = (sheet.width - 20) // 320
    left = 20 + (panels - 1) * 320
    ax.imshow(np.asarray(sheet.crop((left, 74, left + 300, 374))))
    ax.axis("off")
    ax.set_title(f"верх: сходимость {entry['top_honest']:.3f}\nпятен {entry.get('top_patches')}, в главном {entry.get('top_in_main', 0):.2f}",
                 loc="left", fontsize=10)

fig.suptitle("Что меняет ширина гауссова ядра (вариант 561, зерно 0): уже колокол — уже «круг внимания» точки при раскладке",
             x=0.04, ha="left", fontsize=14, fontweight="bold", color=INK)
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print(out)
for run, width in runs:
    entry = log[run]
    data = np.load(f"data/maps/tune2-{run}-level0.npz")
    print(run, width, "ур.0", entry["level0"]["honest"], "гладкость", round(smoothness(data["cells"].astype(int), data["angle"].astype(float),
          float(data["period"]), int(data["height"]), int(data["width"])), 2), "верх", entry["top_honest"], "пятен", entry.get("top_patches"))
