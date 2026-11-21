# -*- coding: utf-8 -*-
"""Разбор: почему цифра собралась (или нет) в одно пятно на картах уровней — на обученной лестнице прогона стенда.

    python tools/tuning/class_study.py [зерно=0] [цифра=6] [папка=data/maps/class-study]

Опора — tools/tuning/base.json + облегчённые настройки стенда + зерно (0 — прогон 648), обучение из кеша. По каждому
уровню и цифре: М1 (доля в главном пятне), честная чистота k ближайших по коду (соседи с других картинок), граф раскладки
G_λ — пары кодов со сходством ≥ λ (раскладка тянет друг к другу только их, §5.4): плотность связей внутри цифры, доля
связей наружу («утечка»), число связных кусков. Для верхнего уровня — те же мерки по месту и по содержанию кода окна
отдельно и детекторы уровня ниже, на которых срабатывают в основном окна этой цифры. Итог — summary.json и рисунки.
Автор 2026-11-21: «в 648 шестёрка целиком лежит в одном пятне — очень внимательно исследуй этот случай, дай отчёт».
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sigma_levels as SL  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage, sparse  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
DIGIT = int(sys.argv[2]) if len(sys.argv) > 2 else 6
FOLDER = sys.argv[3] if len(sys.argv) > 3 else "data/maps/class-study"
EIGHT = np.ones((3, 3), bool)
DARK, INK, MUTED = "#0d0e12", "#e8e8ec", "#8a8c96"
DIGIT_RGB = [tuple(v / 255 for v in Image.new("HSV", (1, 1), (int(d * 25.5), 220, 255)).convert("RGB").getpixel((0, 0))) for d in range(10)]
SAMPLE = 3000  # точек на уровне для попарных сходств (верхний уровень — все)


def m1_by_digit(cells, labels, height, width) -> dict:
    grid = np.full((int(height), int(width)), -1, dtype=np.int64)
    grid[cells[:, 0], cells[:, 1]] = labels
    empty = grid < 0
    out = {}
    for d in range(10):
        mask = grid == d
        if not mask.any():
            continue
        joined = mask | (empty & ndimage.binary_dilation(mask, structure=EIGHT))
        parts, _ = ndimage.label(joined, structure=EIGHT)
        sizes = np.bincount(parts[mask])[1:]
        out[d] = {"points": int(mask.sum()), "main": float(sizes.max() / mask.sum()), "patches": int((sizes >= 5).sum())}
    return out


def cosine(bits: "np.ndarray") -> "np.ndarray":
    x = bits.astype(np.float32)
    norm = np.sqrt(x.sum(axis=1, keepdims=True))
    x /= np.maximum(norm, 1e-6)
    return x @ x.T


def graph_stats(sim, labels, pictures, lam, k=10) -> dict:
    """Мерки по цифрам: честная чистота k ближайших, связи G_λ внутри и наружу, куски, средние сходства."""
    n = sim.shape[0]
    honest = sim.copy()
    honest[pictures[:, None] == pictures[None, :]] = -np.inf
    top = np.argpartition(-honest, kth=min(k, n - 1) - 1, axis=1)[:, :k]
    same_top = labels[top] == labels[:, None]
    edge = (sim >= lam) & (pictures[:, None] != pictures[None, :])
    np.fill_diagonal(edge, False)
    out = {}
    for d in range(10):
        mine = labels == d
        m = int(mine.sum())
        if m < 2:
            continue
        inner = edge[np.ix_(mine, mine)]
        outer = edge[np.ix_(mine, ~mine)]
        pairs = m * (m - 1)
        n_comp, comp = connected_components(sparse.csr_matrix(inner), directed=False)
        sizes = np.bincount(comp)
        isolated = int((inner.sum(axis=1) == 0).sum())
        within = sim[np.ix_(mine, mine)][~np.eye(m, dtype=bool)]
        between = sim[np.ix_(mine, ~mine)]
        out[d] = {
            "points": m,
            "knn_purity": float(same_top[mine].mean()),
            "edge_density_in": float(inner.sum() / max(1, pairs)),
            "edges_in_per_point": float(inner.sum(axis=1).mean()),
            "edges_out_per_point": float(outer.sum(axis=1).mean()),
            "leak": float(outer.sum() / max(1, outer.sum() + inner.sum())),
            "components": int((sizes >= 5).sum()),
            "largest_component": float(sizes.max() / m),
            "isolated": isolated,
            "sim_in": float(within.mean()),
            "sim_out": float(between.mean()),
        }
    return out


def codes_of(L, detectors):
    """Коды окон уровня L из перехвата: по объекту детекторов либо по числу окон."""
    windows = int(SL.GOT["embed"][L][0].pieces.picture.shape[0]) if L in SL.GOT["embed"] else -1
    return next(item for item in SL.GOT["codes"] if item[1] is detectors or int(item[2].sets) == windows)


def main() -> None:
    os.makedirs(FOLDER, exist_ok=True)
    base = json.load(open(os.path.join(HERE, "base.json"), encoding="utf-8"))
    light = {"LAYOUT_LOG_VISUALS": False, "ENCODING_WORKERS": 4, "ACTIVATION_PARALLEL_WORKERS": 2, "LADDER_CLASS_MAP": False}
    changes = {**base, **light, "DASHBOARD_FILE_PREFIX": "data/tune/ladder"}
    if SEED:
        changes["ENCODER_RANDOM_SEED"] = SEED
    SL.capture()
    damp = SL.Damp(changes=changes)
    ladder = damp.cycle.trained(250)
    run = ladder._ladder
    groups = run.groups
    labels_all = np.asarray(groups.labels)
    first, last = groups.ranges["train"]
    maps = {int(m.number): m for m in ladder.maps()}
    levels = {int(item.number): item for item in run.levels}
    top_level = max(maps)
    rng = np.random.default_rng(0)
    summary = {"seed": SEED, "digit": DIGIT, "pictures": {}, "levels": {}}
    for d in range(10):
        summary["pictures"][d] = int(((labels_all == d) & (np.arange(labels_all.size) >= first) & (np.arange(labels_all.size) < last)).sum())
    sims_top = None
    for L in sorted(maps):
        lm = maps[L]
        level = levels[L]
        table = level.memory.layout.code_table
        bits = np.asarray(table.bits(), dtype=np.uint8)
        labels = np.asarray(lm.labels, dtype=np.int64)
        pictures = np.asarray(lm.point_picture, dtype=np.int64)
        cells = np.asarray(lm.cells, dtype=np.int64)
        layout = level.memory.layout
        lam_final = float(layout.lambda_threshold)
        pick = np.arange(labels.size) if labels.size <= SAMPLE or L == top_level else np.sort(rng.choice(labels.size, SAMPLE, replace=False))
        sim = cosine(bits[pick])
        tri = sim[np.triu_indices(pick.size, 1)]
        lam_start = float(np.quantile(tri, float(damp.settings["LADDER_LAYOUT_LAMBDA_QUANTILE"])))
        entry = {
            "points": int(labels.size), "code_length": int(bits.shape[1]), "lambda_start": lam_start, "lambda_final": lam_final,
            "m1": m1_by_digit(cells, labels, lm.height, lm.width),
            "graph_start": graph_stats(sim, labels[pick], pictures[pick], lam_start),
            "graph_final": graph_stats(sim, labels[pick], pictures[pick], lam_final),
        }
        if L == top_level:
            sims_top = (sim, labels[pick], pictures[pick], lam_start, lam_final)
            codes, _, _ = codes_of(L, level.detectors)
            place_len = int(codes.place_length)
            for name, part in (("place", bits[:, :place_len]), ("content", bits[:, place_len:])):
                s = cosine(part)
                q = float(np.quantile(s[np.triu_indices(s.shape[0], 1)], float(damp.settings["LADDER_LAYOUT_LAMBDA_QUANTILE"])))
                entry[f"graph_{name}"] = graph_stats(s, labels, pictures, q)
            entry["place_length"] = place_len
            # одна картинка — сколько точек верха и все ли в одном пятне
            per_picture = {}
            for p in np.unique(pictures[labels == DIGIT]):
                per_picture[int(p)] = int((pictures == p).sum())
            entry["digit_pictures"] = per_picture
            entry["cells"] = cells.tolist()
            entry["labels"] = labels.tolist()
            entry["pictures_of_points"] = pictures.tolist()
            entry["height"], entry["width"] = int(lm.height), int(lm.width)
        summary["levels"][L] = entry
        g = entry["graph_start"].get(DIGIT, {})
        print(f"уровень {L}: точек {labels.size}, λ {lam_start:.3f}→{lam_final:.3f}; цифра {DIGIT}: М1 {entry['m1'].get(DIGIT, {}).get('main', 0):.2f}, "
              f"чистота kNN {g.get('knn_purity', 0):.2f}, утечка {g.get('leak', 0):.2f}, кусков {g.get('components')}")
    # детекторы уровня ниже верхнего, на которых срабатывают в основном окна этой цифры
    below = top_level - 1
    keyholes, _ = SL.GOT["embed"][below]
    codes_b, det_b, firing_b = codes_of(below, levels[below].detectors)
    win_label = labels_all[np.asarray(keyholes.pieces.picture)]
    rows, cols = np.asarray(firing_b.rows), np.asarray(firing_b.columns)
    fired_label = win_label[rows]
    built = int(det_b.built)
    counts = np.zeros((built, 10), dtype=np.int64)
    np.add.at(counts, (cols, fired_label), 1)
    total = counts.sum(axis=1)
    share = counts[:, DIGIT] / np.maximum(total, 1)
    specific = np.flatnonzero((share >= 0.6) & (total >= 5))
    digit_windows = np.flatnonzero(win_label == DIGIT)
    hit = np.isin(rows, digit_windows) & np.isin(cols, specific)
    covered = np.unique(rows[hit]).size / max(1, digit_windows.size)
    spec_all = {}
    for d in range(10):
        sh = counts[:, d] / np.maximum(total, 1)
        sp = np.flatnonzero((sh >= 0.6) & (total >= 5))
        dw = np.flatnonzero(win_label == d)
        h = np.isin(rows, dw) & np.isin(cols, sp)
        spec_all[d] = {"specific_detectors": int(sp.size), "windows_covered": float(np.unique(rows[h]).size / max(1, dw.size))}
    summary["below"] = {"level": below, "detectors": built, "specific": specific.tolist(), "windows_covered": float(covered),
                        "by_digit": spec_all}
    print(f"уровень {below}: детекторов {built}, «своих» у цифры {DIGIT} (≥60 % срабатываний): {specific.size}; окна цифры с ними: {covered:.0%}")
    json.dump(summary, open(os.path.join(FOLDER, f"summary-seed{SEED}-digit{DIGIT}.json"), "w", encoding="utf-8"), ensure_ascii=False)
    figures(summary, maps, sims_top, levels, below, specific, groups, first, last)


def figures(summary, maps, sims_top, levels, below, specific, groups, first, last) -> None:
    def save(fig, name):
        path = os.path.join(FOLDER, f"{name}-seed{SEED}.png")
        fig.savefig(path, facecolor=DARK, dpi=110)
        plt.close(fig)
        print(path)

    def style(ax):
        ax.tick_params(colors=MUTED, labelsize=9)
        for s in ax.spines.values():
            s.set_color("#444")
        ax.set_facecolor(DARK)

    Ls = sorted(maps)
    # 1. карты уровней: цифра — своим цветом, остальные серым
    fig, axes = plt.subplots(1, len(Ls), figsize=(5 * len(Ls), 5.6))
    fig.patch.set_facecolor(DARK)
    for ax, L in zip(np.atleast_1d(axes), Ls):
        lm = maps[L]
        cells, labels = np.asarray(lm.cells), np.asarray(lm.labels)
        mine = labels == DIGIT
        s = 1.5 if cells.shape[0] > 6000 else 4
        ax.scatter(cells[~mine, 1], cells[~mine, 0], s=s, color="#3a3d48", marker="s", linewidths=0)
        ax.scatter(cells[mine, 1], cells[mine, 0], s=s * 1.6, color=DIGIT_RGB[DIGIT], marker="s", linewidths=0)
        m = summary["levels"][L]["m1"].get(DIGIT, {})
        ax.set_title(f"уровень {L}: «{DIGIT}» в главном пятне {m.get('main', 0):.0%}, пятен {m.get('patches')}", color=INK, fontsize=11)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off"); ax.set_facecolor(DARK)
    save(fig, f"digit{DIGIT}-levels")
    # 2. таблица по цифрам и уровням: М1, чистота kNN, утечка
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    fig.patch.set_facecolor(DARK)
    for ax, (key, title, source) in zip(axes, (("main", "М1: доля в главном пятне", "m1"), ("knn_purity", "чистота 10 ближайших по коду (честная)", "graph_start"),
                                               ("leak", "утечка: доля связей G_λ к чужим цифрам", "graph_start"))):
        table = np.array([[summary["levels"][L][source].get(d, {}).get(key, np.nan) for d in range(10)] for L in Ls])
        im = ax.imshow(table, cmap="viridis_r" if key == "leak" else "viridis", vmin=0, vmax=1, aspect="auto")
        for i in range(table.shape[0]):
            for j in range(10):
                ax.text(j, i, f"{table[i, j]:.2f}", ha="center", va="center", color="white" if table[i, j] < 0.6 or key == "leak" else "black", fontsize=8)
        ax.set_xticks(range(10)); ax.set_yticks(range(len(Ls))); ax.set_yticklabels([f"ур. {L}" for L in Ls])
        ax.set_title(title, color=INK, fontsize=11)
        style(ax)
    save(fig, "table")
    # 3. сходства на верхнем уровне: своя цифра против чужих, рядом λ
    sim, labels, pictures, lam0, lam1 = sims_top
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.4), sharex=True)
    fig.patch.set_facecolor(DARK)
    other = pictures[:, None] != pictures[None, :]
    for d, ax in enumerate(axes.reshape(-1)):
        mine = labels == d
        inner = sim[np.ix_(mine, mine)][other[np.ix_(mine, mine)]]
        outer = sim[np.ix_(mine, ~mine)].reshape(-1)
        bins = np.linspace(0, 1, 60)
        ax.hist(outer, bins=bins, color="#6c7385", density=True, alpha=0.8)
        ax.hist(inner, bins=bins, color=DIGIT_RGB[d], density=True, alpha=0.7)
        ax.axvline(lam0, color="white", lw=1.2, ls="--")
        g = summary["levels"][max(maps)]["graph_start"].get(d, {})
        m = summary["levels"][max(maps)]["m1"].get(d, {})
        ax.set_title(f"«{d}»: М1 {m.get('main', 0):.2f}, утечка {g.get('leak', 0):.2f}", color=INK, fontsize=10)
        style(ax)
    save(fig, "similarity-top")
    # 4. все картинки цифры среди обучающих и сколько у каждой точек верха
    pics = [p for p in range(first, last) if int(groups.labels[p]) == DIGIT]
    per = summary["levels"][max(maps)]["digit_pictures"]
    cols = 10
    rows_n = int(np.ceil(len(pics) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 1.6, rows_n * 1.9))
    fig.patch.set_facecolor(DARK)
    for ax in np.atleast_1d(axes).reshape(-1):
        ax.axis("off")
    for ax, p in zip(np.atleast_1d(axes).reshape(-1), pics):
        ax.imshow(np.asarray(groups.atoms.images[p]), cmap="gray")
        ax.set_title(f"№{int(groups.atoms.index[p])}: {per.get(p, 0)}", color=INK, fontsize=9)
    save(fig, f"digit{DIGIT}-pictures")
    # 5. карта уровня ниже: детекторы, где срабатывают в основном окна цифры
    lm = maps[below]
    cells = np.asarray(lm.cells)
    det = np.asarray(lm.detectors)
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(DARK)
    ax.scatter(cells[:, 1], cells[:, 0], s=3, c=[DIGIT_RGB[int(d)] for d in lm.labels], marker="s", linewidths=0, alpha=0.35)
    for j in specific:
        y, x, r = det[j, 0], det[j, 1], det[j, 2]
        ax.add_patch(plt.Circle((x, y), r, fill=False, color="white", lw=1.5))
    ax.set_title(f"уровень {below}: {len(specific)} детекторов «{DIGIT}» (≥ 60 % срабатываний)", color=INK, fontsize=11)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off"); ax.set_facecolor(DARK)
    save(fig, f"digit{DIGIT}-detectors-below")


if __name__ == "__main__":
    main()
