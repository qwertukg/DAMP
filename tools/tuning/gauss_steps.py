# -*- coding: utf-8 -*-
"""σ гауссова ядра дальней фазы раскладки (расширение статьи) — по шагам на настоящей карте уровня 0 из кеша прогона.

    gauss_steps.py <changes.json> <обучающих цифр> <какая пятёрка> <выход png>

Дальняя фаза (§5.5.1, у нас ядро "gauss"): пара точек a, b меняется местами, если после обмена их сходство с соседями,
взвешенное колоколом w(d) = exp(−d² / 2σ²), больше. σ = LAYOUT_LONG_KERNEL_WIDTH · r(t), где r(t) — радиус выбора пар:
в начале фазы он почти во всю карту, к концу сжимается до нескольких клеток. В статье вместо колокола — расстояние:
φ = Σ s·d по всей карте (минимизация), без границы.
"""
import colorsys
import json
import os
import sys

sys.path[:0] = [os.path.join(os.getcwd(), "client"), os.path.join(os.getcwd(), "system")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from damp_api import Damp  # noqa: E402

INK, MUTED, SURFACE, ACCENT = "#0b0b0b", "#8a8984", "#fcfcfb", "#d6451b"
BELL = ["#2196f3", "#4caf50", "#e8590c"]


def hue(angle: float, period: float) -> tuple:
    return colorsys.hls_to_rgb((angle / period) % 1.0, 0.6, 0.95)


def main() -> None:
    changes = json.load(open(sys.argv[1], encoding="utf-8"))["changes"]
    count, which, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    damp = Damp(changes={**changes, "DASHBOARD_FILE_PREFIX": "data/tune/ladder"})
    ladder = damp.cycle.trained(count)
    width = float(damp.settings["LAYOUT_LONG_KERNEL_WIDTH"])
    run = ladder._ladder
    groups = run.groups
    primary = run.primary
    layout = primary.memory.layout
    cells = np.asarray(layout.cells_array(), dtype=np.float64)
    angle = np.asarray(primary.point_angle, dtype=np.float64)
    picture_of = np.asarray(primary.point_picture, dtype=np.int64)
    period = 180.0
    first, last = groups.ranges["train"]
    fives = [p for p in range(first, last) if int(groups.labels[p]) == 5]
    picture = int(fives[which])
    mnist = int(groups.atoms.index[picture])
    mine = np.flatnonzero(picture_of == picture)
    centre_of_map = cells.mean(axis=0)
    p = int(mine[np.argmin(np.hypot(*(cells[mine] - centre_of_map).T))])
    similarity = layout._pair_similarity_np(np.full(cells.shape[0], p, dtype=np.int64), np.arange(cells.shape[0], dtype=np.int64))
    similarity[p] = np.nan
    distance = np.hypot(cells[:, 0] - cells[p, 0], cells[:, 1] - cells[p, 1])
    side = int(max(layout.height, layout.width))
    radii = [int(0.5 * side), max(3, int(0.12 * side)), 3]  # начало, середина, конец дальней фазы

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(18, 10.5), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1], hspace=0.42, wspace=0.6)

    # 1. Карта уровня 0 и точка p.
    ax = fig.add_subplot(gs[0, 0:2])
    ax.scatter(cells[:, 1], cells[:, 0], c=[hue(a, period) for a in angle], s=5, marker="s", linewidths=0)
    ax.plot(cells[p, 1], cells[p, 0], "o", ms=11, mfc="none", mec=INK, mew=2.2)
    ax.set_xlim(-1, layout.width); ax.set_ylim(layout.height, -1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"1. Карта уровня 0 (цвет — угол штриха); кружок — точка p:\nштрих пятёрки MNIST train №{mnist}",
                 loc="left", fontsize=10.5, color=INK)

    # 2. Сходство кода p со всеми точками.
    ax = fig.add_subplot(gs[0, 2:4])
    shown = np.nan_to_num(similarity, nan=0.0)
    ax.scatter(cells[:, 1], cells[:, 0], c=shown, cmap="YlOrBr", vmin=0, vmax=np.nanquantile(similarity, 0.995), s=5, marker="s",
               linewidths=0)
    ax.plot(cells[p, 1], cells[p, 0], "o", ms=11, mfc="none", mec=INK, mew=2.2)
    ax.set_xlim(-1, layout.width); ax.set_ylim(layout.height, -1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("2. Насколько коды точек похожи на код p (s)\nтемнее — похожее; похожие уже собрались рядом",
                 loc="left", fontsize=10.5, color=INK)

    # 3. Колокол при трёх радиусах: что «чувствует» p — s · w(d) внутри окрестности r.
    inner = gs[0, 4:6].subgridspec(1, 3, wspace=0.08)
    for k, (r, colour) in enumerate(zip(radii, BELL)):
        ax = fig.add_subplot(inner[0, k])
        sigma = width * r
        weight = np.where(distance <= r, np.exp(-distance ** 2 / (2 * sigma ** 2)), 0.0)
        felt = shown * weight
        view = max(r + 2, 6)
        near = distance <= view * 1.42
        ax.scatter(cells[near, 1], cells[near, 0], c=felt[near], cmap="Greys", vmin=0, vmax=max(1e-9, np.quantile(felt[near], 0.99)),
                   s=max(2, 900 / view ** 1.3), marker="s", linewidths=0)
        ax.add_patch(Circle((cells[p, 1], cells[p, 0]), r, fill=False, color=colour, lw=2))
        ax.set_xlim(cells[p, 1] - view, cells[p, 1] + view); ax.set_ylim(cells[p, 0] + view, cells[p, 0] - view)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        stage = ["начало фазы", "середина", "конец"][k]
        ax.set_title(f"{stage}\nr = {r}, σ = {sigma:.0f}", fontsize=9.5, color=colour)
    fig.text(0.665, 0.905, "3. Что «чувствует» p: сходство × колокол (s · w)\nвнутри окрестности r(t); r сжимается по ходу фазы",
             fontsize=10.5, color=INK, va="bottom")

    # 4. Колокол и ядро статьи.
    ax = fig.add_subplot(gs[1, 0:3])
    d = np.linspace(0, radii[0] * 1.15, 400)
    for r, colour in zip(radii, BELL):
        sigma = width * r
        w = np.where(d <= r, np.exp(-d ** 2 / (2 * sigma ** 2)), 0.0)
        ax.plot(d, w, color=colour, lw=2.5, label=f"колокол, r = {r}, σ = {sigma:.0f}")
    ax.plot(d, d / d.max(), color=MUTED, lw=2, ls="--", label="статья: штраф растёт с расстоянием (φ = Σ s·d), без границы")
    ax.set_xlabel("расстояние d до точки p, клеток"); ax.set_ylabel("вес соседа w(d)")
    ax.legend(frameon=False, fontsize=9)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.set_title("4. Колокол w(d) = exp(−d² / 2σ²), σ = ширина · r: далёкие точки не тянут (у статьи — тянут все)",
                 loc="left", fontsize=10.5)

    # 5. Пинвилы крупно: кусок карты уровня 0.
    ax = fig.add_subplot(gs[1, 3:6])
    view = 16
    near = (np.abs(cells[:, 0] - cells[p, 0]) <= view) & (np.abs(cells[:, 1] - cells[p, 1]) <= view)
    ax.scatter(cells[near, 1], cells[near, 0], c=[hue(a, period) for a in angle[near]], s=60, marker="s", linewidths=0)
    ax.plot(cells[p, 1], cells[p, 0], "o", ms=11, mfc="none", mec=INK, mew=2.2)
    ax.set_xlim(cells[p, 1] - view, cells[p, 1] + view); ax.set_ylim(cells[p, 0] + view, cells[p, 0] - view)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("5. Итог: пинвилы крупно — углы плавно идут по кругу вокруг центров;\nсоседние пинвилы закручены в разные стороны",
                 loc="left", fontsize=10.5)
    fig.suptitle(f"σ гауссова ядра: как точка выбирает соседей при раскладке (дальняя фаза, ширина {width:g})", x=0.06, ha="left",
                 fontsize=15, fontweight="bold", color=INK)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(out, f"точка {p}", f"MNIST №{mnist}", f"радиусы {radii}")


if __name__ == "__main__":
    main()
