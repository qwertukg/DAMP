# -*- coding: utf-8 -*-
"""Своя схема на каждый уровень цикла — на одной обучающей четвёрке, на настоящих данных лучшей опоры.

    python tools/tuning/level_diagram.py [номер четвёрки среди обучающих=0] [папка=data/maps/level-diagrams]

Опора — tools/tuning/base.json (принятое в подборе v2) + зерно 0 + облегчённые настройки стенда (как в tune_cycle.py):
обучение берётся из кеша прогона 632. На уровне: элементы на цифре → окно → код окна → карта (память) → детекторы →
зажглись точки → сработали детекторы → отбор σ → код детали → детали уровня на цифре; условия — ромбы.
Автор 2026-11-21: «сделай свою диаграмму на каждый уровень. на цифре 4, на лучшем прогоне текущего состояния».
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sigma_levels as SL  # noqa: E402  (перехват срабатываний и эмбеддингов, путь к системе)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

WHICH = int(sys.argv[1]) if len(sys.argv) > 1 else 0
FOLDER = sys.argv[2] if len(sys.argv) > 2 else "data/maps/level-diagrams"
PANELS = os.path.join(FOLDER, "panels")
DIGIT = 4
DARK, INK, MUTED, KEEP, DROP = "#0d0e12", "#e8e8ec", "#8a8c96", "#ff6a2a", "#7d8590"
LAYER_COLOURS = ["#ff5050", "#ffc13b", "#42d17a", "#3fa9ff", "#b07bff", "#ff7bd5", "#9be15d"]


def panel(name: str, draw_fn, size=(5.2, 5.2)) -> str:
    """Отдельная картинка подшага (тёмный фон, без рамок осей)."""
    fig = plt.figure(figsize=size, dpi=150)
    fig.patch.set_facecolor(DARK)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_facecolor(DARK)
    draw_fn(ax)
    path = os.path.join(PANELS, f"{name}.png")
    fig.savefig(path, facecolor=DARK)
    plt.close(fig)
    return path


def digit_axes(ax, image) -> None:
    ax.imshow(np.asarray(image), cmap="gray", extent=[-0.94, 23.94, 23.94, -0.94], alpha=0.55)
    ax.set_xlim(-1, 24); ax.set_ylim(24, -1); ax.set_aspect("equal"); ax.axis("off")


def sticks(ax, sy, sx, sa, faint=False) -> None:
    for y, x, a in zip(sy, sx, sa):
        t = np.radians(a)
        ax.plot([x - 0.45 * np.cos(t), x + 0.45 * np.cos(t)], [y + 0.45 * np.sin(t), y - 0.45 * np.sin(t)],
                color=SL.hue(a, 180.0), lw=1.2 if faint else 2.4, alpha=0.45 if faint else 1.0)


def main() -> None:
    from damp import settings

    os.makedirs(PANELS, exist_ok=True)
    base = json.load(open(os.path.join(HERE, "base.json"), encoding="utf-8"))
    light = {"LAYOUT_LOG_VISUALS": False, "ENCODING_WORKERS": 4, "ACTIVATION_PARALLEL_WORKERS": 2, "LADDER_CLASS_MAP": False}
    SL.capture()
    damp = SL.Damp(changes={**base, **light, "DASHBOARD_FILE_PREFIX": "data/tune/ladder"})
    ladder = damp.cycle.trained(250)
    sigma = int(damp.settings["ACTIVATION_SATURATION_LIMIT"])
    run = ladder._ladder
    groups = run.groups
    first, last = groups.ranges["train"]
    fours = [p for p in range(first, last) if int(groups.labels[p]) == DIGIT]
    picture = int(fours[WHICH])
    mnist = int(groups.atoms.index[picture])
    s0, s1 = int(groups.starts[picture]), int(groups.starts[picture + 1])
    sy, sx, sa = (np.asarray(v[s0:s1], float) for v in (groups.atoms.y, groups.atoms.x, groups.atoms.angle))
    image = groups.atoms.images[picture]
    middle = np.array([sy.mean(), sx.mean()])
    maps = {int(m.number): m for m in ladder.maps()}
    primary = ladder.primary_map()
    place_lengths = list(settings.LADDER_BINDING_PLACE_CODE_LENGTH) if hasattr(settings, "LADDER_BINDING_PLACE_CODE_LENGTH") else [176, 1792, 1536, 1088]
    place_bits = base.get("LADDER_BINDING_PLACE_BITS", [39, 1])
    summary = []
    for L in range(0, 1 + len(run.levels)):
        if L == 0:
            prim = run.primary
            memory, detectors = prim.memory, prim.detectors
            atoms = int(np.asarray(groups.atoms.y).shape[0])
            codes, _, firing = next(item for item in SL.GOT["codes"] if item[1] is detectors or int(item[2].sets) == atoms)
            details, length = SL.GOT["details0"]
            window = s0 + int(np.argmin(np.hypot(sy - middle[0], sx - middle[1])))
            cy, cx, radius = float(groups.atoms.y[window]), float(groups.atoms.x[window]), 1.1
            mine_centres = np.c_[sy, sx]
            members = np.array([window - s0])
            elements = np.c_[sy, sx]
            cells = np.asarray(primary.cells, dtype=np.float64)
            colours = [SL.hue(a, float(primary.angle_period)) for a in primary.angle]
        else:
            level = next(item for item in run.levels if int(item.number) == L)
            memory, detectors = level.memory, level.detectors
            keyholes, (details, length, strength, _) = SL.GOT["embed"][L]
            codes, _, firing = next(item for item in SL.GOT["codes"] if item[1] is detectors or int(item[2].sets) == int(keyholes.pieces.picture.shape[0]))
            mine = np.flatnonzero(np.asarray(keyholes.pieces.picture) == picture)
            centre = np.asarray(keyholes.pieces.centre, dtype=np.float64)
            window = int(mine[np.argmin(np.hypot(centre[mine, 0] - middle[0], centre[mine, 1] - middle[1]))])
            cy, cx = (float(v) for v in centre[window])
            radius = float(np.asarray(keyholes.extent, dtype=np.float64)[window])
            mine_centres = centre[mine]
            if L == 1:
                elements = np.c_[sy, sx]
            else:
                prev_keyholes, _ = SL.GOT["embed"][L - 1]
                pm = np.flatnonzero(np.asarray(prev_keyholes.pieces.picture) == picture)
                elements = np.asarray(prev_keyholes.pieces.centre, dtype=np.float64)[pm]
            members = np.flatnonzero(np.hypot(elements[:, 0] - cy, elements[:, 1] - cx) <= radius)
            lm = maps[L]
            cells = np.asarray(lm.cells, dtype=np.float64)
            palette = [tuple(int(c) for c in Image.new("HSV", (1, 1), (int(d * 25.5), 220, 255)).convert("RGB").getpixel((0, 0))) for d in range(10)]
            colours = [tuple(v / 255 for v in palette[int(d)]) for d in lm.labels]
        rows, cols_, levels = np.asarray(firing.rows), np.asarray(firing.columns), np.asarray(firing.levels, dtype=np.float64)
        pick = rows == window
        fired, fired_level = cols_[pick], levels[pick]
        from damp.activation.set_embedding import DetectorEmbeddings

        _, kept_columns = DetectorEmbeddings(detectors=int(detectors.built), saturation_limit=sigma, target_density=0.1, min_length=64,
                                             max_length=8192, seed=0, name="sigma").strongest(np.zeros(fired.size, dtype=np.int64), fired, fired_level)
        kept = np.isin(fired, kept_columns)
        order = np.argsort(-fired_level, kind="stable")
        lit_rows, lit_points, lit_values = detectors.lit(np.asarray(codes.packed)[window:window + 1])
        layout = memory.layout
        mcells = np.asarray(layout.cells_array(), dtype=np.float64)
        table = [(d.center[0], d.center[1], d.radius) for d in detectors.detectors]
        layer = np.asarray(detectors.layer_index)
        code_bits = np.unpackbits(np.asarray(details)[window], count=int(length))
        window_bits = np.unpackbits(np.asarray(codes.packed)[window], count=int(codes.code_length))
        pl = int(place_lengths[min(L, len(place_lengths) - 1)]) * int(place_bits[min(L, len(place_bits) - 1)] if isinstance(place_bits, list) else place_bits)
        value = np.asarray(lit_values, dtype=np.float64)
        points = np.asarray(lit_points, dtype=np.int64)
        tag = f"L{L}"

        # 1. элементы уровня на цифре
        def p1(ax):
            digit_axes(ax, image)
            sticks(ax, sy, sx, sa, faint=L > 1)
            if L >= 2:
                ax.scatter(elements[:, 1], elements[:, 0], s=36, color="#4fd1c5", edgecolors="white", linewidths=0.5, zorder=5)
        f1 = panel(f"{tag}-1-elements", p1)

        # 2. окно на цифре: круг и элементы внутри
        def p2(ax):
            digit_axes(ax, image)
            sticks(ax, sy, sx, sa, faint=True)
            if L >= 2:
                ax.scatter(elements[:, 1], elements[:, 0], s=22, color="#4fd1c5", alpha=0.5, zorder=4)
            ax.scatter(elements[members, 1], elements[members, 0], s=60, color=KEEP, edgecolors="white", linewidths=0.7, zorder=6)
            ax.add_patch(Circle((cx, cy), radius, fill=False, color="#ffd34d", lw=2.5))
            ax.plot(cx, cy, "+", color="#ffd34d", ms=14, mew=2.5)
        f2 = panel(f"{tag}-2-window", p2)

        # 3. код окна: место (оранжевые) + содержание (синие)
        def p3(ax):
            n = window_bits.size
            side = int(np.ceil(np.sqrt(n)))
            board = np.zeros((side * side, 3)) + 0.12
            on = np.flatnonzero(window_bits)
            board[on[on < pl]] = (1.0, 0.45, 0.15)
            board[on[on >= pl]] = (0.3, 0.6, 1.0)
            board[n:] = 0.05
            ax.imshow(board.reshape(side, side, 3), interpolation="nearest")
            ax.axis("off")
        f3 = panel(f"{tag}-3-window-code", p3)

        # 4. карта уровня (память): L0 — угол штриха, выше — цифра
        def p4(ax):
            ax.scatter(cells[:, 1], cells[:, 0], s=3 if cells.shape[0] > 6000 else 6, c=colours, marker="s", linewidths=0)
            ax.set_xlim(-1, layout.width); ax.set_ylim(layout.height, -1); ax.set_aspect("equal"); ax.axis("off")
        f4 = panel(f"{tag}-4-map", p4)

        # 5. детекторы над картой: круги по слоям
        def p5(ax):
            ax.scatter(mcells[:, 1], mcells[:, 0], s=2, color="#3a3d48", marker="s", linewidths=0)
            for (yy, xx, rr), k in zip(table, layer):
                ax.add_patch(Circle((xx, yy), rr, fill=False, color=LAYER_COLOURS[int(k) % len(LAYER_COLOURS)], lw=0.5, alpha=0.8))
            ax.set_xlim(-1, layout.width); ax.set_ylim(layout.height, -1); ax.set_aspect("equal"); ax.axis("off")
        f5 = panel(f"{tag}-5-detectors", p5)

        # 6. код окна зажёг точки карты
        def p6(ax):
            ax.scatter(mcells[:, 1], mcells[:, 0], s=3, color="#30333d", marker="s", linewidths=0)
            ax.scatter(mcells[points, 1], mcells[points, 0], s=5, c=value, cmap="YlOrBr", vmin=0, vmax=max(1e-9, value.max()), marker="s", linewidths=0)
            ax.set_xlim(-1, layout.width); ax.set_ylim(layout.height, -1); ax.set_aspect("equal"); ax.axis("off")
        f6 = panel(f"{tag}-6-lit", p6)

        # 7. сработавшие детекторы, крупно: красные — σ сильнейших
        def p7(ax):
            ax.scatter(mcells[:, 1], mcells[:, 0], s=20, color="#262932", marker="s", linewidths=0)
            ax.scatter(mcells[points, 1], mcells[points, 0], s=20, c=value, cmap="YlOrBr", vmin=0, vmax=max(1e-9, value.max()), marker="s", linewidths=0)
            for column, lvl, k in sorted(zip(fired, fired_level, kept), key=lambda item: item[2]):
                yy, xx, rr = table[int(column)]
                ax.add_patch(Circle((xx, yy), rr, fill=False, color=KEEP if k else DROP, lw=0.6 + 2.4 * lvl, alpha=0.95 if k else 0.55))
            spots = [table[int(c)] for c in fired] or [(cy, cx, 1)]
            ys = np.concatenate([mcells[points, 0], [a - r for a, _, r in spots], [a + r for a, _, r in spots]])
            xs = np.concatenate([mcells[points, 1], [b - r for _, b, r in spots], [b + r for _, b, r in spots]])
            ax.set_xlim(xs.min() - 3, xs.max() + 3); ax.set_ylim(ys.max() + 3, ys.min() - 3); ax.set_aspect("equal"); ax.axis("off")
        f7 = panel(f"{tag}-7-fired", p7)

        # 8. уровни E(d) по убыванию, порог σ
        def p8(ax):
            ranked = fired_level[order]
            ax.bar(np.arange(1, ranked.size + 1), ranked, color=[KEEP if k else DROP for k in kept[order]], width=0.8)
            ax.axvline(sigma + 0.5, color=INK, lw=1.4, ls="--")
            ax.text(sigma + 1, max(ranked.max(), 1e-3) * 0.97, f"σ = {sigma}", color=INK, fontsize=12, va="top")
            ax.tick_params(colors=MUTED, labelsize=9)
            for s in ax.spines.values():
                s.set_color("#444")
            ax.set_facecolor(DARK)
        f8 = panel(f"{tag}-8-sigma", p8, size=(7.0, 4.6))

        # 9. код детали
        def p9(ax):
            side = int(np.ceil(np.sqrt(code_bits.size)))
            board = np.zeros((side * side, 3)) + 0.12
            board[np.flatnonzero(code_bits)] = (1.0, 0.42, 0.16)
            board[code_bits.size:] = 0.05
            ax.imshow(board.reshape(side, side, 3), interpolation="nearest")
            ax.axis("off")
        f9 = panel(f"{tag}-9-detail-code", p9)

        # 10. все детали уровня на цифре; деталь этого окна — жёлтым
        def p10(ax):
            digit_axes(ax, image)
            sticks(ax, sy, sx, sa, faint=True)
            ax.scatter(mine_centres[:, 1], mine_centres[:, 0], s=48, color="#4fd1c5", edgecolors="white", linewidths=0.6, zorder=5)
            ax.scatter([cx], [cy], s=150, color="#ffd34d", edgecolors="black", linewidths=1.0, zorder=6)
        f10 = panel(f"{tag}-10-details", p10)

        sees = float(np.mean(np.hypot(sy - cy, sx - cx) <= radius)) if L > 0 else 1.0 / max(1, sy.size)
        summary.append({"L": L, "files": [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10], "members": int(members.size), "elements": int(elements.shape[0]),
                        "radius": radius, "points": int(mcells.shape[0]), "detectors": len(table), "layers": int(layer.max()) + 1 if layer.size else 0,
                        "lit": int(points.size), "fired": int(fired.size), "kept": int(kept.sum()), "code": int(code_bits.size), "ones": int(code_bits.sum()),
                        "window_bits": int(window_bits.size), "place": int(pl), "window_ones": int(window_bits.sum()), "sees": sees,
                        "details": int(mine_centres.shape[0]), "mnist": mnist})
        print(f"уровень {L}: окно {window}, элементов в окне {members.size}, зажглось {points.size}, сработало {fired.size}, взято {int(kept.sum())}")
    json.dump(summary, open(os.path.join(FOLDER, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    compose(summary)
    precycle(groups, picture, image, sy, sx, sa, summary)
    recognition(ladder, image, maps, primary, summary)
    master(summary)


# ---------------------------------------------------------------- сборка схем уровней
def compose(summary) -> None:
    def F(size, bold=False):
        return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)

    F_T, F_SUB, F_CELL, F_SMALL = F(110, True), F(46), F(40, True), F(34)
    names = {0: "штрихов", 1: "штрихов (деталей уровня 0)", 2: "деталей уровня 1", 3: "деталей уровня 2", 4: "деталей уровня 3", 5: "деталей уровня 4"}
    last = max(item["L"] for item in summary)
    for item in summary:
        L = item["L"]
        W, H = 7200, 5400
        sheet = Image.new("RGB", (W, H), (13, 14, 18))
        d = ImageDraw.Draw(sheet)

        def wrap(s, fnt, width):
            out, line = [], ""
            for word in s.split(" "):
                probe = (line + " " + word).strip()
                if d.textlength(probe, font=fnt) <= width:
                    line = probe
                else:
                    out.append(line)
                    line = word
            out.append(line)
            return out

        def arrow(p, q, colour=(205, 207, 215)):
            import math

            d.line([p, q], fill=colour, width=10)
            a = math.atan2(q[1] - p[1], q[0] - p[0])
            h = 36
            d.polygon([q, (q[0] - h * math.cos(a) + h * 0.6 * math.sin(a), q[1] - h * math.sin(a) - h * 0.6 * math.cos(a)),
                       (q[0] - h * math.cos(a) - h * 0.6 * math.sin(a), q[1] - h * math.sin(a) + h * 0.6 * math.cos(a))], fill=colour)

        def cell(x, y, w, h, label, path=None, note=None):
            d.rounded_rectangle((x, y, x + w, y + h), radius=26, fill=(36, 39, 50), outline=(95, 100, 115), width=2)
            yy = y + 18
            for line in wrap(label, F_CELL, w - 40):
                d.text((x + 20, yy), line, font=F_CELL, fill=(240, 240, 245))
                yy += 50
            if note:
                for line in wrap(note, F_SMALL, w - 40):
                    d.text((x + 20, yy), line, font=F_SMALL, fill=(170, 172, 182))
                    yy += 42
            if path:
                im = Image.open(path).convert("RGB")
                aw, ah = w - 40, y + h - yy - 24
                k = min(aw / im.width, ah / im.height)
                im = im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)
                sheet.paste(im, (int(x + (w - im.width) / 2), int(yy + 8 + (ah - im.height) / 2)))

        def diamond(cx, cy, w, h, s):
            d.polygon([(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)], fill=(56, 46, 16), outline=(245, 205, 70), width=3)
            lines = wrap(s, F_SMALL, w * 0.6)
            y0 = cy - len(lines) * 22
            for line in lines:
                d.text((cx - d.textlength(line, font=F_SMALL) / 2, y0), line, font=F_SMALL, fill=(255, 240, 200))
                y0 += 44

        f = item["files"]
        d.text((120, 60), f"Уровень {L}: виток цикла на четвёрке (MNIST train №{item['mnist']})", font=F_T, fill=(255, 255, 255))
        sub = (f"лучшая опора: перекрытие угла 0.85 + код окна до 8192, зерно 0 (прогон 632) · "
               + (f"окно — один штрих, место от центра цифры" if L == 0 else
                  f"окно — кластер похожих соседних {names[L]}: в окне {item['members']} из {item['elements']}, радиус {item['radius']:.1f} клетки, видит ~{100 * item['sees']:.0f} % штрихов цифры")
               + f" · карта {item['points']} точек · {item['detectors']} детекторов в {item['layers']} слоях · σ = 10")
        y = 210
        for line in wrap(sub, F_SUB, W - 240):
            d.text((120, y), line, font=F_SUB, fill=(185, 188, 198))
            y += 60
        top = y + 50
        cw, ch, gx = 1180, 1480, 220
        xs = [120 + i * (cw + gx) for i in range(5)]
        r1, r2, r3 = top, top + ch + 160, top + 2 * (ch + 160)
        # Строка 1: элементы → окно → ◇ → код окна → карта
        cell(xs[0], r1, cw, ch, "1. Элементы уровня на цифре", f[0], "штрихи" if L <= 1 else f"детали уровня {L - 1} — центры их окон")
        cell(xs[1], r1, cw, ch, "2. Окно на цифре", f[1], f"в окне {item['members']} элементов (оранжевые)")
        diamond(xs[2] + cw / 2, r1 + ch / 2, cw - 40, ch * 0.55, "окно = один штрих" if L == 0 else "кластер ≥ N похожих соседей? нет — окно на всю картинку")
        cell(xs[3], r1, cw, ch, "3. Код окна", f[2], f"{item['window_bits']} бит, единиц {item['window_ones']}: место (оранж.) {item['place']} бит + содержание (синие)")
        cell(xs[4], r1, cw, ch, "4. Первая подача: карта уровня (память)", f[3], "цвет — угол штриха" if L == 0 else "цвет — цифра картинки точки")
        for i in range(4):
            arrow((xs[i] + cw + 10, r1 + ch / 2), (xs[i + 1] - 10, r1 + ch / 2))
        # Строка 2 (справа налево): детекторы → зажглись → ◇ → сработали
        cell(xs[4], r2, cw, ch, "5. Детекторы над картой", f[4], f"{item['detectors']} детекторов, цвет — слой")
        cell(xs[3], r2, cw, ch, "6. Вторая подача: код окна зажёг точки", f[5], f"горит {item['lit']} точек из {item['points']}")
        diamond(xs[2] + cw / 2, r2 + ch / 2, cw - 40, ch * 0.55, "сходство ≥ λ_a и энергия активации ≥ 0.5?")
        cell(xs[1], r2, cw, ch, "7. Сработали детекторы (крупно)", f[6], f"{item['fired']} сработало; красные — {item['kept']} сильнейших")
        cell(xs[0], r2, cw, ch, "8. Отбор σ сильнейших по E(d)", f[7], "красные идут в код")
        arrow((xs[4] + cw / 2, r1 + ch + 10), (xs[4] + cw / 2, r2 - 10))
        for i in (4, 3, 2, 1):
            arrow((xs[i] - 10, r2 + ch / 2), (xs[i - 1] + cw + 10, r2 + ch / 2))
        # Строка 3: код детали → детали на цифре → дальше
        cell(xs[0], r3, cw, ch, "9. Код детали", f[8], f"{item['code']} бит, единиц {item['ones']}: по биту от сильнейших")
        cell(xs[1], r3, cw, ch, "10. Детали уровня на цифре", f[9], f"{item['details']} деталей; жёлтая — деталь этого окна")
        arrow((xs[0] + cw / 2, r2 + ch + 10), (xs[0] + cw / 2, r3 - 10))
        arrow((xs[0] + cw + 10, r3 + ch / 2), (xs[1] - 10, r3 + ch / 2))
        nxt = "→ пятна: из похожих соседних деталей остаётся сильнейшая → детали — элементы окон уровня " + str(L + 1) if L < last else \
            "→ окно видит цифру целиком — обучение закончено; дальше пороги знания и распознавание"
        diamond(xs[2] + cw / 2, r3 + ch / 2, cw - 40, ch * 0.55, "рядом похожая деталь (≥ 0.8)? оставить сильнейшую")
        arrow((xs[1] + cw + 10, r3 + ch / 2), (xs[2] + 20, r3 + ch / 2))
        y = r3 + 120
        F_NEXT = F(64, True)
        for line in wrap(nxt, F_NEXT, 2 * cw + gx):
            d.text((xs[3], y), line, font=F_NEXT, fill=(70, 205, 140))
            y += 80
        arrow((xs[2] + cw - 20, r3 + ch / 2), (xs[3] - 10, r3 + ch / 2), (70, 205, 140))
        out = os.path.join(FOLDER, f"level{L}-four.png")
        sheet.save(out, optimize=True)
        print(out)


# ---------------------------------------------------------------- общая сборка сетки подшагов (тот же стиль, что у уровней)
def grid_sheet(title, subtitle, rows, out, tail=None):
    """rows — строки подшагов: ("cell", заголовок, картинка или None, пояснение) или ("if", вопрос); змейкой: чётные строки
    слева направо, нечётные — справа налево; между строками — стрелка вниз."""
    import math

    def F(size, bold=False):
        return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)

    F_T, F_SUB, F_CELL, F_SMALL = F(110, True), F(46), F(40, True), F(34)
    cols = max(len(r) for r in rows)
    cw, ch, gx, gy = 1180, 1480, 220, 160
    W = 120 * 2 + cols * cw + (cols - 1) * gx
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))

    def wrap(s, fnt, width):
        out_, line = [], ""
        for word in s.split(" "):
            candidate = (line + " " + word).strip()
            if probe.textlength(candidate, font=fnt) <= width:
                line = candidate
            else:
                out_.append(line)
                line = word
        out_.append(line)
        return out_

    head = wrap(subtitle, F_SUB, W - 240)
    top = 210 + 60 * len(head) + 50
    H = top + len(rows) * (ch + gy) + (260 if tail else 60)
    sheet = Image.new("RGB", (W, H), (13, 14, 18))
    d = ImageDraw.Draw(sheet)
    d.text((120, 60), title, font=F_T, fill=(255, 255, 255))
    y = 210
    for line in head:
        d.text((120, y), line, font=F_SUB, fill=(185, 188, 198))
        y += 60

    def arrow(p, q, colour=(205, 207, 215)):
        d.line([p, q], fill=colour, width=10)
        a = math.atan2(q[1] - p[1], q[0] - p[0])
        h = 36
        d.polygon([q, (q[0] - h * math.cos(a) + h * 0.6 * math.sin(a), q[1] - h * math.sin(a) - h * 0.6 * math.cos(a)),
                   (q[0] - h * math.cos(a) - h * 0.6 * math.sin(a), q[1] - h * math.sin(a) + h * 0.6 * math.cos(a))], fill=colour)

    positions = []
    for r, row in enumerate(rows):
        ry = top + r * (ch + gy)
        order = list(range(len(row))) if r % 2 == 0 else list(range(cols - len(row), cols))[::-1]
        line_pos = []
        for k, item in zip(order, row):
            x = 120 + k * (cw + gx)
            if item[0] == "if":
                cx, cy, w, h = x + cw / 2, ry + ch / 2, cw - 40, ch * 0.55
                d.polygon([(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)], fill=(56, 46, 16), outline=(245, 205, 70), width=3)
                lines = wrap(item[1], F_SMALL, w * 0.6)
                y0 = cy - len(lines) * 22
                for line in lines:
                    d.text((cx - d.textlength(line, font=F_SMALL) / 2, y0), line, font=F_SMALL, fill=(255, 240, 200))
                    y0 += 44
            else:
                _, label, path, note = item
                d.rounded_rectangle((x, ry, x + cw, ry + ch), radius=26, fill=(36, 39, 50), outline=(95, 100, 115), width=2)
                yy = ry + 18
                for line in wrap(label, F_CELL, cw - 40):
                    d.text((x + 20, yy), line, font=F_CELL, fill=(240, 240, 245))
                    yy += 50
                for line in wrap(note or "", F_SMALL, cw - 40):
                    d.text((x + 20, yy), line, font=F_SMALL, fill=(170, 172, 182))
                    yy += 42
                if path:
                    im = Image.open(path).convert("RGB")
                    aw, ah = cw - 40, ry + ch - yy - 24
                    k2 = min(aw / im.width, ah / im.height)
                    im = im.resize((max(1, int(im.width * k2)), max(1, int(im.height * k2))), Image.LANCZOS)
                    sheet.paste(im, (int(x + (cw - im.width) / 2), int(yy + 8 + (ah - im.height) / 2)))
            line_pos.append(x)
        for i in range(1, len(line_pos)):
            a, b = line_pos[i - 1], line_pos[i]
            if b > a:
                arrow((a + cw + 10, ry + ch / 2), (b - 10, ry + ch / 2))
            else:
                arrow((a - 10, ry + ch / 2), (b + cw + 10, ry + ch / 2))
        if positions:
            px = positions[-1][-1]
            arrow((px + cw / 2, ry - gy + 10), (px + cw / 2, ry - 10))
        positions.append(line_pos)
    if tail:
        fN = F(64, True)
        y = top + len(rows) * (ch + gy) - 40
        for line in wrap(tail, fN, W - 240):
            d.text((120, y), line, font=fN, fill=(70, 205, 140))
            y += 80
    sheet.save(out, optimize=True)
    print(out)
    return out


def bits_board(ax, bits, bands):
    """Код картинкой: единица — цветом полосы своего слоя (широкий — красный … узкий — фиолетовый)."""
    import colorsys

    n = bits.size
    side = int(np.ceil(np.sqrt(n)))
    board = np.zeros((side * side, 3)) + 0.12
    for i in np.flatnonzero(bits):
        band = min(bands - 1, int(i * bands / n))
        board[i] = colorsys.hsv_to_rgb(0.83 * band / max(1, bands - 1), 0.85, 1.0)
    board[n:] = 0.05
    ax.imshow(board.reshape(side, side, 3), interpolation="nearest")
    ax.axis("off")


def precycle(groups, picture, image, sy, sx, sa, summary) -> None:
    """Шаги 1–3 на четвёрке: картинка → рабочая → точки середины линии → штрихи → код угла → место → код окна уровня 0."""
    from damp.encoding.elements import StickCoder
    from damp.stimuli.mnist import _build_extractor

    img = np.asarray(image, dtype=np.uint8)
    extractor = _build_extractor()
    work = extractor.work_image(img)
    midline = extractor.stroke_midline
    py, px, vx, vy = midline.points(work) if midline is not None else (np.zeros(0),) * 4
    centre = np.array([sy.mean(), sx.mean()])
    pick = int(np.argmin(np.hypot(sy - centre[0], sx - centre[1])))
    angle = float(sa[pick])
    radius = float(np.hypot(sy - centre[0], sx - centre[1]).max())
    angle_bits = np.asarray(StickCoder().encoder.encode_matrix([[angle]]), dtype=np.uint8).reshape(-1)
    l0 = summary[0]

    def p_raw(ax):
        ax.imshow(img, cmap="gray", interpolation="nearest"); ax.axis("off")

    def p_work(ax):
        ax.imshow(work, cmap="gray"); ax.axis("off")

    def p_mid(ax):
        ax.imshow(work, cmap="gray", alpha=0.35)
        direction = (np.degrees(np.arctan2(vy, vx)) + 90.0) % 180.0
        ax.scatter(px, py, s=6, c=[SL.hue(a, 180.0) for a in direction], linewidths=0)
        ax.axis("off")

    def p_sticks(ax):
        digit_axes(ax, image); sticks(ax, sy, sx, sa)

    def p_one(ax):
        digit_axes(ax, image); sticks(ax, sy, sx, sa, faint=True)
        sticks(ax, sy[pick:pick + 1], sx[pick:pick + 1], sa[pick:pick + 1])
        ax.add_patch(Circle((sx[pick], sy[pick]), 0.9, fill=False, color="#ffd34d", lw=2.5))

    def p_angle(ax):
        t = np.linspace(0, np.pi, 200)
        ax.plot(np.cos(t), np.sin(t), color="#666", lw=2)
        for a in range(0, 180, 15):
            rr = np.radians(a)
            ax.plot([0.92 * np.cos(rr), np.cos(rr)], [0.92 * np.sin(rr), np.sin(rr)], color="#888", lw=1)
        rr = np.radians(angle)
        ax.plot([0, np.cos(rr)], [0, np.sin(rr)], color=SL.hue(angle, 180.0), lw=5)
        ax.text(0, -0.18, f"{angle:.0f}°", color=INK, ha="center", fontsize=22)
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-0.35, 1.15); ax.set_aspect("equal"); ax.axis("off")

    def p_angle_code(ax):
        bits_board(ax, angle_bits, 10)

    def p_place(ax):
        digit_axes(ax, image); sticks(ax, sy, sx, sa, faint=True)
        ax.add_patch(Circle((centre[1], centre[0]), radius, fill=False, color="#888", lw=1.5, ls="--"))
        ax.annotate("", xy=(sx[pick], sy[pick]), xytext=(centre[1], centre[0]), arrowprops=dict(arrowstyle="->", color="#ffd34d", lw=2.5))
        ax.plot(centre[1], centre[0], "o", color="white", ms=7)

    files = [panel("pre-1-raw", p_raw), panel("pre-2-work", p_work), panel("pre-3-midline", p_mid), panel("pre-4-sticks", p_sticks),
             panel("pre-5-one", p_one), panel("pre-6-angle", p_angle), panel("pre-7-angle-code", p_angle_code), panel("pre-8-place", p_place)]
    rows = [
        [("cell", "1. Картинка MNIST 28×28", files[0], f"обучающая четвёрка №{l0['mnist']}"),
         ("cell", "2. Рабочая картинка 224×224", files[1], "увеличение ×8, контраст"),
         ("cell", "3. Точки середины линии", files[2], f"{len(px)} точек: гребень расстояния до края; цвет — направление"),
         ("if", "гребень: сила ≥ порога, линия, а не пятно, не кончик?"),
         ("cell", "4. Штрихи", files[3], f"{sy.size} штрихов: детектор над соседними точками → угол 0…180° и место")],
        [("cell", "5. Один штрих (окно уровня 0)", files[4], "ближайший к центру цифры"),
         ("cell", "6. Угол штриха", files[5], f"{angle:.0f}°, период 180° (линия без стрелки)"),
         ("cell", "7. Код угла: 10 слоёв, бит на детектор", files[6], f"256 бит, единиц {int(angle_bits.sum())}; цвет — слой (широкий красный … узкий фиолетовый)"),
         ("cell", "8. Место — от центра цифры", files[7], "вектор от центра к штриху в долях радиуса цифры → круги места (10 слоёв)"),
         ("cell", "9. Код окна уровня 0", l0["files"][2], f"место ×39 копий ({l0['place']} бит, оранж.) + содержание — угол под масками (синие)")],
    ]
    grid_sheet(f"Шаги 1–3: картинка → штрихи → код (четвёрка MNIST train №{l0['mnist']})",
               "лучшая опора подбора: перекрытие угла 0.85, код окна до 8192, 4 уровня, зерно 0 · всё на настоящих данных этой картинки · "
               "С(§4.2, §4.4 — код угла; §6.4 — детектор по точкам), Д — штрихи по середине линии, место от центра цифры (П.4)",
               rows, os.path.join(FOLDER, "pre-cycle-four.png"), tail="→ код окна каждого штриха — на уровень 0 (начало цикла)")


def recognition(ladder, image, maps, primary, summary) -> None:
    """Распознавание этой четвёрки: штрихи → что зажглось на каждом уровне → голоса уровней и пороги знания → ответы."""
    sight = ladder.see(np.asarray(image, dtype=np.uint8), log=False)
    l0 = summary[0]

    def p_raw(ax):
        ax.imshow(np.asarray(image), cmap="gray", interpolation="nearest"); ax.axis("off")

    def p_sticks(ax):
        digit_axes(ax, image)
        sticks(ax, np.asarray(sight.stick_y, float), np.asarray(sight.stick_x, float), np.asarray(sight.stick_angle, float))

    f_raw, f_sticks = panel("rec-1-raw", p_raw), panel("rec-2-sticks", p_sticks)
    lit = []
    if sight.primary is not None:
        cells0 = np.asarray(primary.cells, dtype=np.float64)
        pts0 = np.asarray(sight.primary.lit_point, dtype=np.int64)

        def p0(ax, cells0=cells0, pts0=pts0):
            ax.scatter(cells0[:, 1], cells0[:, 0], s=2, color="#30333d", marker="s", linewidths=0)
            ax.scatter(cells0[pts0, 1], cells0[pts0, 0], s=3, color="#ffb347", marker="s", linewidths=0)
            ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
        lit.append((0, panel("rec-L0-lit", p0), int(np.unique(pts0).size), int(cells0.shape[0])))
    for lv in sight.levels:
        n = int(lv.number)
        if n not in maps:
            continue
        cells = np.asarray(maps[n].cells, dtype=np.float64)
        pts = np.asarray(lv.lit_point, dtype=np.int64)

        def pl(ax, cells=cells, pts=pts):
            ax.scatter(cells[:, 1], cells[:, 0], s=3, color="#30333d", marker="s", linewidths=0)
            ax.scatter(cells[pts, 1], cells[pts, 0], s=5, color="#ffb347", marker="s", linewidths=0)
            ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
        lit.append((n, panel(f"rec-L{n}-lit", pl), int(np.unique(pts).size), int(cells.shape[0])))
    votes = list(sight.votes)

    def p_votes(ax):
        ax.axis("off")
        k = max(1, len(votes))
        for i, v in enumerate(votes):
            sub = ax.inset_axes([0.02, 1 - (i + 1) / k + 0.03, 0.96, 1 / k - 0.06])
            vals = np.asarray(v.votes, dtype=float)
            sub.bar(range(10), vals, color=["#ff6a2a" if j == v.digit else "#7d8590" for j in range(10)])
            sub.set_xticks(range(10)); sub.tick_params(colors=MUTED, labelsize=8); sub.set_yticks([])
            sub.set_facecolor(DARK)
            for s in sub.spines.values():
                s.set_color("#444")
            sub.text(0.99, 0.9, f"ур. {v.number}: «{v.digit}» {100 * v.share:.0f} %, {'знает' if v.knows else 'не знает'}",
                     transform=sub.transAxes, ha="right", va="top", color=INK, fontsize=9)
    f_votes = panel("rec-votes", p_votes, size=(5.2, 6.4))

    def p_memory(ax):
        vals = np.asarray(sight.memory_votes, dtype=float) if len(sight.memory_votes) else np.zeros(10)
        ax.bar(range(10), vals, color=["#ff6a2a" if j == sight.memory_digit else "#7d8590" for j in range(10)])
        ax.set_xticks(range(10)); ax.tick_params(colors=MUTED); ax.set_facecolor(DARK)
        ax.text(0.02, 0.97, f"память: «{sight.memory_digit}» ({sight.verdict})", transform=ax.transAxes, color=INK, fontsize=14, va="top")
    f_memory = panel("rec-memory", p_memory)

    def p_reader(ax):
        vals = np.asarray(sight.probabilities, dtype=float)
        ax.bar(range(10), vals, color=["#42d17a" if j == sight.digit else "#7d8590" for j in range(10)])
        ax.set_xticks(range(10)); ax.tick_params(colors=MUTED); ax.set_facecolor(DARK)
        ax.text(0.02, 0.97, f"читатель: «{sight.digit}» {100 * vals.max():.0f} %", transform=ax.transAxes, color=INK, fontsize=14, va="top")
    f_reader = panel("rec-reader", p_reader)
    flat = [("cell", "1. Картинка", f_raw, f"четвёрка №{l0['mnist']} (обучающая; так же идёт любая новая)"),
            ("cell", "2. Штрихи", f_sticks, f"{np.asarray(sight.stick_y).size} штрихов → код угла и места")]
    flat += [("cell", f"Уровень {n}: зажглись точки", path, f"{count} из {total} точек; сработали детекторы → детали → окна уровня {n + 1}")
             for n, path, count, total in lit]
    flat += [("cell", "Голоса уровней и пороги знания", f_votes, "у уровня — доли голосов за цифры; «знает», если доля ≥ порога"),
             ("if", "уровни согласны? целиком / по частям / не знаю"),
             ("cell", "Ответ памяти", f_memory, "среднее долей голосов уровней (§7.1.13.3)"),
             ("cell", "Ответ читателя", f_reader, "линейный слой над уровнями активации (§9.3)")]
    rows = [flat[i:i + 5] for i in range(0, len(flat), 5)]
    grid_sheet(f"Распознавание: четвёрка №{l0['mnist']} через все уровни",
               "те же карты и детекторы, что при обучении: картинка не меняет их — только зажигает · пороги знания — по 100 обучающим "
               "картинкам, прошедшим этот же путь", rows, os.path.join(FOLDER, "recognition-four.png"),
               tail=f"→ ответ памяти «{sight.memory_digit}» ({sight.verdict}), читателя «{sight.digit}»")


def master(summary) -> None:
    """Общая схема: шаги 1–3 → витки уровней 0…L (цикл) → распознавание, стрелками; каждая часть — своя схема."""
    import math

    parts = [os.path.join(FOLDER, "pre-cycle-four.png")] + [os.path.join(FOLDER, f"level{item['L']}-four.png") for item in summary] + \
        [os.path.join(FOLDER, "recognition-four.png")]
    names = ["Шаги 1–3"] + [f"Уровень {item['L']}" for item in summary] + ["Распознавание"]
    cols, tw = 3, 3000
    thumbs = []
    for path in parts:
        im = Image.open(path).convert("RGB")
        k = tw / im.width
        thumbs.append(im.resize((tw, int(im.height * k)), Image.LANCZOS))
    th = max(t.height for t in thumbs)
    gx, gy, pad, top = 360, 420, 160, 420
    rows = math.ceil(len(thumbs) / cols)
    W, H = pad * 2 + cols * tw + (cols - 1) * gx, top + rows * (th + gy) + 200
    sheet = Image.new("RGB", (W, H), (13, 14, 18))
    d = ImageDraw.Draw(sheet)
    fT, fN = ImageFont.truetype("arialbd.ttf", 150), ImageFont.truetype("arialbd.ttf", 80)
    d.text((pad, 80), "DAMP на одной четвёрке: полная схема по частям", font=fT, fill=(255, 255, 255))
    d.text((pad, 270), "каждая часть — своя подробная схема (те же файлы по отдельности в data/maps/level-diagrams/); зелёная рамка — "
                       "витки цикла по уровням; стрелки — порядок", font=ImageFont.truetype("arial.ttf", 56), fill=(185, 188, 198))
    boxes = []
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        c = c if r % 2 == 0 else cols - 1 - c
        x, y = pad + c * (tw + gx), top + r * (th + gy)
        sheet.paste(t, (x, y + 90))
        d.text((x, y), names[i], font=fN, fill=(70, 205, 140) if names[i].startswith("Уровень") else (255, 255, 255))
        boxes.append((x, y + 90, x + tw, y + 90 + t.height))
    for (x0, y0, x1, y1), name in zip(boxes, names):
        if name.startswith("Уровень"):
            d.rounded_rectangle((x0 - 30, y0 - 130, x1 + 30, y1 + 30), radius=40, outline=(70, 205, 140), width=6)

    def arrow(p, q, colour=(220, 220, 228)):
        d.line([p, q], fill=colour, width=16)
        a = math.atan2(q[1] - p[1], q[0] - p[0])
        h = 60
        d.polygon([q, (q[0] - h * math.cos(a) + h * 0.6 * math.sin(a), q[1] - h * math.sin(a) - h * 0.6 * math.cos(a)),
                   (q[0] - h * math.cos(a) - h * 0.6 * math.sin(a), q[1] - h * math.sin(a) + h * 0.6 * math.cos(a))], fill=colour)

    for (a0, a1, a2, a3), (b0, b1, b2, b3) in zip(boxes, boxes[1:]):
        if abs(a1 - b1) < 10:
            if b0 > a0:
                arrow((a2 + 40, (a1 + a3) / 2), (b0 - 40, (b1 + b3) / 2))
            else:
                arrow((a0 - 40, (a1 + a3) / 2), (b2 + 40, (b1 + b3) / 2))
        else:
            arrow(((a0 + a2) / 2, a3 + 40), ((b0 + b2) / 2, b1 - 150))
    out = os.path.join(FOLDER, "full-four.png")
    sheet.save(out, optimize=True)
    print(out, sheet.size)


if __name__ == "__main__":
    main()
