# -*- coding: utf-8 -*-
"""Что такое σ (ACTIVATION_SATURATION_LIMIT): путь одного окна пятёрки до кода детали, по шагам, на настоящих данных.

    sigma_steps.py <файл с изменениями JSON (как у стенда)> <обучающих цифр> <уровень> <номер пятёрки среди обучающих> <выход png>

Шаги (§6.1, §6.5, §7.1.12): окно на цифре → код окна зажигает карту уровня → срабатывают детекторы, у каждого свой
уровень E(d) → берутся σ сильнейших → у каждого взятого детектора свой случайный бит → код детали (эмбеддинг окна).
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

GOT: dict = {}


def capture() -> None:
    from damp.cycle.step import CycleStep

    fire, embed = CycleStep.fire, CycleStep.embed

    def fire_seen(detectors, codes, trained):  # noqa: ANN001, ANN202
        firing = fire(detectors, codes, trained)
        GOT.setdefault("codes", []).append((codes, detectors, firing))
        return firing

    def embed_seen(self, firing, detectors, keyholes):  # noqa: ANN001, ANN202
        result = embed(self, firing, detectors, keyholes)
        known = GOT.setdefault("embed", {}).get(int(keyholes.number))
        if known is None or int(keyholes.pieces.picture.shape[0]) > int(known[0].pieces.picture.shape[0]):
            GOT["embed"][int(keyholes.number)] = (keyholes, result)
        return result

    CycleStep.fire = staticmethod(fire_seen)
    CycleStep.embed = embed_seen
    from damp.activation.details import DetailCoder

    encode = DetailCoder.encode

    def encode_seen(self, *args, **kwargs):  # noqa: ANN001, ANN202
        result = encode(self, *args, **kwargs)
        # Обучение идёт первым и по всем окнам; потом цикл пропускает по одной картинке (порог знания) — их не берём.
        if int(self.number) == 0 and ("details0" not in GOT or np.asarray(result[0]).shape[0] > np.asarray(GOT["details0"][0]).shape[0]):
            GOT["details0"] = result
        return result

    DetailCoder.encode = encode_seen


def hue(angle: float, period: float) -> tuple:
    return colorsys.hls_to_rgb((angle / period) % 1.0, 0.55, 0.95)


def segment_sse(sx, sy, sxx, sxy, syy, a, b):  # noqa: ANN001, ANN201
    """Сумма квадратов остатков прямой по точкам [a, b) через префиксные суммы (a, b — массивы одной формы)."""
    n = (b - a).astype(np.float64)
    x, y = sx[b] - sx[a], sy[b] - sy[a]
    xx, xy, yy = sxx[b] - sxx[a], sxy[b] - sxy[a], syy[b] - syy[a]
    vx = xx - x * x / np.maximum(n, 1)
    cov = xy - x * y / np.maximum(n, 1)
    vy = yy - y * y / np.maximum(n, 1)
    return np.where(n >= 2, vy - np.where(vx > 1e-12, cov * cov / np.maximum(vx, 1e-12), 0.0), 0.0)


def breaks(levels: np.ndarray) -> tuple[int, int]:
    y = np.sort(levels)[::-1].astype(np.float64)
    n = y.size
    x = np.arange(n, dtype=np.float64)
    pad = lambda v: np.concatenate([[0.0], np.cumsum(v)])  # noqa: E731
    sx, sy, sxx, sxy, syy = pad(x), pad(y), pad(x * x), pad(x * y), pad(y * y)
    best, found = np.inf, (1, n - 1)
    for b1 in range(1, max(2, n // 3 + 1)):
        b2 = np.arange(b1 + 2, n, dtype=np.int64)
        if b2.size == 0:
            continue
        zero = np.zeros_like(b2)
        total = (segment_sse(sx, sy, sxx, sxy, syy, zero, np.full_like(b2, b1))
                 + segment_sse(sx, sy, sxx, sxy, syy, np.full_like(b2, b1), b2)
                 + segment_sse(sx, sy, sxx, sxy, syy, b2, np.full_like(b2, n)))
        k = int(np.argmin(total))
        if total[k] < best:
            best, found = float(total[k]), (b1, int(b2[k]))
    return found


def main() -> None:
    changes = json.load(open(sys.argv[1], encoding="utf-8"))["changes"]
    count, which, prefix = int(sys.argv[2]), int(sys.argv[4]), sys.argv[5]
    wanted = [int(v) for v in sys.argv[3].split(",")]
    capture()
    damp = Damp(changes={**changes, "DASHBOARD_FILE_PREFIX": "data/tune/ladder"})
    ladder = damp.cycle.trained(count)
    sigma = int(damp.settings["ACTIVATION_SATURATION_LIMIT"])
    share = float(damp.settings["ACTIVATION_SATURATION_SHARE"])
    run = ladder._ladder
    groups = run.groups
    first, last = groups.ranges["train"]
    fives = [p for p in range(first, last) if int(groups.labels[p]) == 5]
    picture = int(fives[which])
    for level_number in wanted:
        draw(run, groups, picture, level_number, sigma, f"{prefix}-level{level_number}.png", share)


def draw(run, groups, picture, level_number, sigma, out, share=0.0) -> None:  # noqa: ANN001
    from damp.activation.set_embedding import DetectorEmbeddings

    mnist = int(groups.atoms.index[picture])
    s0, s1 = int(groups.starts[picture]), int(groups.starts[picture + 1])
    sy, sx, sa = (np.asarray(groups.atoms.y[s0:s1], float), np.asarray(groups.atoms.x[s0:s1], float),
                  np.asarray(groups.atoms.angle[s0:s1], float))
    middle = np.array([sy.mean(), sx.mean()])
    if level_number == 0:
        # Уровень 0: окно на каждом штрихе, держит его одного; окна идут в порядке штрихов (StrokeLevel.pieces).
        primary = run.primary
        memory, detectors = primary.memory, primary.detectors
        atoms = int(np.asarray(groups.atoms.y).shape[0])
        codes, _, firing = next(item for item in GOT["codes"] if item[1] is detectors or int(item[2].sets) == atoms)
        details, length = GOT["details0"]
        window = s0 + int(np.argmin(np.hypot(sy - middle[0], sx - middle[1])))
        cy, cx, radius = float(groups.atoms.y[window]), float(groups.atoms.x[window]), 1.1
        window_note = "окно держит один штрих; место — от центра цифры"
    else:
        level = next(item for item in run.levels if int(item.number) == level_number)
        memory, detectors = level.memory, level.detectors
        keyholes, (details, length, strength, _) = GOT["embed"][level_number]
        codes, _, firing = next(item for item in GOT["codes"] if item[1] is detectors or int(item[2].sets) == int(keyholes.pieces.picture.shape[0]))
        mine = np.flatnonzero(np.asarray(keyholes.pieces.picture) == picture)
        centre = np.asarray(keyholes.pieces.centre, dtype=np.float64)
        window = int(mine[np.argmin(np.hypot(centre[mine, 0] - middle[0], centre[mine, 1] - middle[1]))])
        cy, cx = (float(v) for v in centre[window])
        radius = float(np.asarray(keyholes.extent, dtype=np.float64)[window])
        window_note = "код окна = место + содержание (П.4–П.5)"
    # Срабатывания окна: детектор и его уровень E(d) (нормированный, как в цикле); σ сильнейших — тем же отбором.
    rows, columns, levels = np.asarray(firing.rows), np.asarray(firing.columns), np.asarray(firing.levels, dtype=np.float64)
    pick = rows == window
    fired, fired_level = columns[pick], levels[pick]
    _, kept_columns = DetectorEmbeddings(detectors=int(detectors.built), saturation_limit=sigma, target_density=0.1,
                                         min_length=64, max_length=8192, seed=0, name="sigma", saturation_share=share).strongest(
        np.zeros(fired.size, dtype=np.int64), fired, fired_level)
    kept = np.isin(fired, kept_columns)
    taken = int(kept.sum())
    sigma_label = f"σ = {share:.0%} сработавших → {taken}" if share > 0 else f"σ = {sigma}"
    sigma = taken if share > 0 else sigma
    order = np.argsort(-fired_level, kind="stable")
    lit_rows, lit_points, lit_values = detectors.lit(np.asarray(codes.packed)[window:window + 1])
    layout = memory.layout
    cells = np.asarray(layout.cells_array(), dtype=np.float64)
    table = [(d.center[0], d.center[1], d.radius) for d in detectors.detectors]
    layer = np.asarray(detectors.layer_index)
    code_bits = np.unpackbits(np.asarray(details)[window], count=int(length))
    period = 180.0

    INK, MUTED, SURFACE, KEEP, DROP = "#0b0b0b", "#8a8984", "#fcfcfb", "#d6451b", "#9aa3ad"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(18, 10.5), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.2, 1], hspace=0.42, wspace=0.55)

    # Шаг 1: окно на цифре.
    ax = fig.add_subplot(gs[0, 0:2])
    ax.imshow(np.asarray(groups.atoms.images[picture]), cmap="gray_r", extent=[-0.94, 23.94, 23.94, -0.94], alpha=0.35)
    for y, x, a in zip(sy, sx, sa):
        t = np.radians(a)
        ax.plot([x - 0.45 * np.cos(t), x + 0.45 * np.cos(t)], [y + 0.45 * np.sin(t), y - 0.45 * np.sin(t)], color=hue(a, period), lw=2)
    ax.add_patch(Circle((cx, cy), radius, fill=False, color=KEEP, lw=2))
    ax.plot(cx, cy, "+", color=KEEP, ms=10)
    ax.set_xlim(-1, 24); ax.set_ylim(24, -1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"1. Окно уровня {level_number} на пятёрке (MNIST train №{mnist})\n{window_note}",
                 loc="left", fontsize=10.5, color=INK)

    # Шаг 2: код окна зажигает карту.
    ax = fig.add_subplot(gs[0, 2:4])
    ax.scatter(cells[:, 1], cells[:, 0], s=3, color="#e4e3de", marker="s", linewidths=0)
    value = np.asarray(lit_values, dtype=np.float64)
    points = np.asarray(lit_points, dtype=np.int64)
    ax.scatter(cells[points, 1], cells[points, 0], s=4, c=value, cmap="YlOrBr", vmin=0, vmax=max(1e-9, value.max()), marker="s",
               linewidths=0)
    ax.set_xlim(-1, layout.width); ax.set_ylim(layout.height, -1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"2. Код окна зажигает карту уровня {level_number} (§6.1)\n"
                 f"горит {points.size} точек из {cells.shape[0]}: сходство кода ≥ λ_a", loc="left", fontsize=10.5, color=INK)

    # Шаг 3: сработавшие детекторы на карте.
    ax = fig.add_subplot(gs[0, 4:6])
    ax.scatter(cells[:, 1], cells[:, 0], s=22, color="#eeeeea", marker="s", linewidths=0)
    ax.scatter(cells[points, 1], cells[points, 0], s=22, c=value, cmap="YlOrBr", vmin=0, vmax=max(1e-9, value.max()), marker="s",
               linewidths=0)
    for column, lvl, k in sorted(zip(fired, fired_level, kept), key=lambda item: item[2]):
        yy, xx, rr = table[int(column)]
        ax.add_patch(Circle((xx, yy), rr, fill=False, color=KEEP if k else DROP, lw=0.6 + 2.2 * lvl, alpha=0.95 if k else 0.6))
    # Крупно: область, где горят точки и стоят сработавшие детекторы.
    spots = [table[int(c)] for c in fired]
    ys = np.concatenate([cells[points, 0], [a - r for a, _, r in spots], [a + r for a, _, r in spots]])
    xs = np.concatenate([cells[points, 1], [b - r for _, b, r in spots], [b + r for _, b, r in spots]])
    pad = 3.0
    ax.set_xlim(xs.min() - pad, xs.max() + pad); ax.set_ylim(ys.max() + pad, ys.min() - pad); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"3. Крупно: сработали {fired.size} детекторов (§6.5); толщина — уровень E(d)\n"
                 f"красные — {int(kept.sum())} сильнейших ({sigma_label}), серые — отброшены", loc="left", fontsize=10.5, color=INK)

    # Шаг 4: уровни по убыванию и порог σ.
    ax = fig.add_subplot(gs[1, 0:3])
    ranked = fired_level[order]
    colours = [KEEP if k else DROP for k in kept[order]]
    ax.bar(np.arange(1, ranked.size + 1), ranked, color=colours, width=0.8)
    ax.axvline(sigma + 0.5, color=INK, lw=1.2, ls="--")
    ax.text(sigma + 0.8, ranked.max() * 0.95, f"{sigma_label}:\nв код идут {sigma} самых сильных", fontsize=9.5, va="top")
    for i, column in enumerate(fired[order][: min(ranked.size, 40)]):
        ax.text(i + 1, ranked[i] + 0.01, str(int(layer[int(column)])), ha="center", va="bottom", fontsize=7, color=MUTED)
    # Отрезки, как в разметке автора (2026-11-20): три прямые по уровням — лидеры (синий), плато (зелёный), хвост (синий);
    # на плато засечки четвертей и кружок в середине — место σ «середина плато».
    if ranked.size >= 6:
        b1, b2 = breaks(ranked)
        xs = np.arange(1, ranked.size + 1, dtype=np.float64)
        for start, stop, colour in ((0, b1, "#2196f3"), (b1, b2, "#4caf50"), (b2, ranked.size, "#2196f3")):
            if stop - start >= 2:
                slope, shift = np.polyfit(xs[start:stop], ranked[start:stop], 1)
                ends = np.array([xs[start] - 0.4, xs[stop - 1] + 0.4])
                ax.plot(ends, slope * ends + shift + 0.015, color=colour, lw=3, solid_capstyle="round", zorder=5)
            elif stop - start == 1:
                ax.plot([xs[start] - 0.4, xs[start] + 0.4], [ranked[start] + 0.015] * 2, color=colour, lw=3, zorder=5)
        slope, shift = np.polyfit(xs[b1:b2], ranked[b1:b2], 1)
        for share in (0.25, 0.5, 0.75):
            x = xs[b1] + share * (xs[b2 - 1] - xs[b1])
            y = slope * x + shift + 0.015
            if share == 0.5:
                ax.plot([x], [y], "o", ms=9, mfc="none", mec="#4caf50", mew=2.2, zorder=6)
                ax.annotate(f"середина плато ≈ {x:.0f}", (x, y), xytext=(8, 14), textcoords="offset points", fontsize=9.5,
                            color="#2e7d32", fontweight="bold")
            else:
                ax.plot([x, x], [y - 0.025, y + 0.025], color="#4caf50", lw=2.2, zorder=6)
        ax.text(0.99, 0.97, f"лидеры: 1–{b1}   плато: {b1 + 1}–{b2}   хвост: {b2 + 1}–{ranked.size}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK)
    ax.set_xlabel("сработавшие детекторы окна, по убыванию уровня (цифра над столбцом — слой: 0 широкий … 4 узкий)")
    ax.set_ylabel("уровень E(d), нормированный")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("4. Отбор σ сильнейших (§7.1.12: «μ_c… |D| ∼ 50»; у нас σ — сколько детекторов взять)", loc="left", fontsize=10.5)

    # Шаг 5: код детали.
    ax = fig.add_subplot(gs[1, 3:6])
    side = int(np.ceil(np.sqrt(code_bits.size)))
    board = np.full(side * side, np.nan)
    board[: code_bits.size] = code_bits
    ax.imshow(board.reshape(side, side), cmap=matplotlib.colors.ListedColormap(["#ecebe6", KEEP]), vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    ones = int(code_bits.sum())
    ax.set_title(f"5. Код детали окна: {code_bits.size} бит, единиц {ones}\n"
                 f"у каждого взятого детектора свой случайный бит (§6.3); "
                 f"{sigma - ones if ones < sigma else 0} совпали", loc="left", fontsize=10.5)
    fig.suptitle(f"Что такое σ: путь окна пятёрки до кода детали, уровень {level_number} ({sigma_label})", x=0.06, ha="left",
                 fontsize=15, fontweight="bold", color=INK)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(out, f"MNIST №{mnist}", f"окно {window}", f"сработало {fired.size}", f"взято {int(kept.sum())}", f"единиц {ones}")


if __name__ == "__main__":
    main()
