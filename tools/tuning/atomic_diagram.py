# -*- coding: utf-8 -*-
"""Атомарная схема DAMP на одной четвёрке: каждая клетка — одна операция кода на настоящих данных, петли и условия.

    python tools/tuning/atomic_diagram.py [зерно=2] [папка=data/maps/atomic]

Опора — tools/tuning/base.json (принятое в подборе v2) + облегчённые настройки стенда + зерно (2 — прогон 649): обучение
из кеша стенда. Промежуточные шаги считаются здесь тем же кодом системы (середина линии, штрихи, коды угла и места,
маски, раскладка с остановками, детекторы, срабатывание, пятна, распознавание) и сверяются с итогом прогона.
Холст квадратный; читать с левого верхнего угла слева направо, ряд за рядом вниз. Оранжевая рамка — петля «для каждого»,
жёлтая — условие, зелёная линия — возврат на следующий уровень.
Автор 2026-11-21: «большую детальную диаграмму, состоящую из множества атомарных картинок с циклами и условиями. всё на
низком уровне»; «вверху слева начинаем и вправо вниз идём. холст квадратный».
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sigma_levels as SL  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Wedge  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 2
FOLDER = sys.argv[2] if len(sys.argv) > 2 else "data/maps/atomic"
PANELS = os.path.join(FOLDER, "panels")
DIGIT = 4
DARK, INK, MUTED, KEEP, DROP = "#0d0e12", "#e8e8ec", "#8a8c96", "#ff6a2a", "#5d6470"
GOOD, BAD = "#42d17a", "#ff4d4d"
LAYER_COLOURS = ["#ff5050", "#ffc13b", "#42d17a", "#3fa9ff", "#b07bff", "#ff7bd5", "#9be15d", "#ffd34d", "#4fd1c5", "#c0c4ff"]
DIGIT_RGB = [tuple(v / 255 for v in Image.new("HSV", (1, 1), (int(d * 25.5), 220, 255)).convert("RGB").getpixel((0, 0))) for d in range(10)]
FLOW: list = []  # клетки по порядку: ("cell"| "if", заголовок, картинка, пояснение) и разметка ("section", ...), ("loop", ...)


# ---------------------------------------------------------------- рисование клеток
def panel(name: str, draw_fn, size=(5.0, 4.2)) -> str:
    fig = plt.figure(figsize=size, dpi=110)
    fig.patch.set_facecolor(DARK)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_facecolor(DARK)
    draw_fn(ax)
    path = os.path.join(PANELS, f"{name}.png")
    fig.savefig(path, facecolor=DARK)
    plt.close(fig)
    return path


def chart_axes(ax) -> None:
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color("#444")
    ax.set_facecolor(DARK)


def cell(title, path, note="") -> None:
    FLOW.append(("cell", title, path, note))


def cond(question, path, note="") -> None:
    FLOW.append(("if", question, path, note))


def section(name, colour) -> None:
    FLOW.append(("section", name, colour))


def loop(label, count, back="") -> None:
    """Следующие count клеток — тело петли «для каждого …»; back — подпись стрелки возврата."""
    FLOW.append(("loop", label, count, back))


def mark(name) -> None:
    FLOW.append(("mark", name))


def board(ax, bits, colours=None, side=None, base=0.13) -> None:
    """Код картинкой: клетка — бит; единица — своим цветом (по умолчанию оранжевым)."""
    bits = np.asarray(bits).reshape(-1)
    n = bits.size
    side = side or int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / side))
    img = np.zeros((rows * side, 3)) + base
    img[n:] = 0.04
    on = np.flatnonzero(bits)
    if colours is None:
        img[on] = (1.0, 0.42, 0.16)
    else:
        img[on] = np.asarray(colours, dtype=float)[on] if np.asarray(colours).shape[0] == n else colours
    ax.imshow(img.reshape(rows, side, 3), interpolation="nearest")
    ax.axis("off")


def main() -> None:
    from damp import settings

    os.makedirs(PANELS, exist_ok=True)
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
    first, last = groups.ranges["train"]
    fours = [p for p in range(first, last) if int(groups.labels[p]) == DIGIT]
    picture = int(fours[0])
    mnist = int(groups.atoms.index[picture])
    s0, s1 = int(groups.starts[picture]), int(groups.starts[picture + 1])
    sy, sx, sa = (np.asarray(v[s0:s1], float) for v in (groups.atoms.y, groups.atoms.x, groups.atoms.angle))
    image = np.asarray(groups.atoms.images[picture], dtype=np.uint8)
    ctx = dict(settings=settings, ladder=ladder, run=run, groups=groups, picture=picture, mnist=mnist, s0=s0, s1=s1,
               sy=sy, sx=sx, sa=sa, image=image, base=base)
    print(f"четвёрка: обучающая №{picture}, MNIST train №{mnist}, штрихов {sy.size}")
    # Damp держит свои правки настроек только внутри своих вызовов: промежуточные шаги считаются здесь же — с настройками прогона.
    run_settings = {key: value for key, value in changes.items() if key != "DASHBOARD_FILE_PREFIX"}
    with settings.overridden(run_settings):
        stage_picture(ctx)
        stage_strokes(ctx)
        stage_angle(ctx)
        stage_place(ctx)
        stage_window(ctx)
        stage_layout(ctx)
        stage_detectors(ctx)
        stage_activation(ctx)
        stage_spots(ctx)
        stage_levels(ctx)
    stage_recognition(ctx)
    json.dump([list(map(str, item)) for item in FLOW], open(os.path.join(FOLDER, "flow.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=0)
    compose(ctx)


# ---------------------------------------------------------------- этап 1: картинка → середина линии
def stage_picture(ctx) -> None:
    from scipy import ndimage
    from damp.stimuli.mnist import _build_extractor

    image = ctx["image"]
    extractor = _build_extractor()
    work = extractor.work_image(image)
    mid = extractor.stroke_midline
    ctx["extractor"], ctx["work"] = extractor, work
    ink = work >= float(mid.level)
    distance = ndimage.distance_transform_edt(ink)
    sigma = float(mid.sigma)
    smooth = ndimage.gaussian_filter(distance, sigma)
    dxx = ndimage.gaussian_filter(distance, sigma, order=(0, 2))
    dyy = ndimage.gaussian_filter(distance, sigma, order=(2, 0))
    dxy = ndimage.gaussian_filter(distance, sigma, order=(1, 1))
    mean, root = (dxx + dyy) / 2.0, np.sqrt(((dxx - dyy) / 2.0) ** 2 + dxy ** 2)
    across, along = mean - root, mean + root
    ax_, ay_ = dxy, across - dxx
    bx_, by_ = across - dyy, dxy
    second = (bx_ * bx_ + by_ * by_) > (ax_ * ax_ + ay_ * ay_)
    nx, ny = np.where(second, bx_, ax_), np.where(second, by_, ay_)
    norm = np.hypot(nx, ny)
    norm[norm == 0.0] = 1.0
    nx, ny = nx / norm, ny / norm
    flip = np.where(np.abs(nx) >= np.abs(ny), nx < 0.0, ny < 0.0)
    nx, ny = np.where(flip, -nx, nx), np.where(flip, -ny, ny)
    rows, cols = np.mgrid[0: distance.shape[0], 0: distance.shape[1]].astype(np.float64)
    tilted = smooth + mid.TIE * (rows + 0.618 * cols)
    ahead = ndimage.map_coordinates(tilted, [rows + ny, cols + nx], order=1, mode="nearest")
    behind = ndimage.map_coordinates(tilted, [rows - ny, cols - nx], order=1, mode="nearest")
    strength = np.maximum(-across, 0.0)
    top = float(strength.max())
    slope_x = ndimage.gaussian_filter(distance, sigma, order=(0, 1))
    slope_y = ndimage.gaussian_filter(distance, sigma, order=(1, 0))
    slope_along = np.abs(slope_x * (-ny) + slope_y * nx)
    crest = (tilted >= ahead) & (tilted > behind) & (smooth >= float(mid.depth))
    c_strong = crest & (strength >= float(mid.strength_share) * top)
    c_line = c_strong & (np.abs(along) <= float(mid.blob) * strength)
    ridge = c_line & (slope_along <= float(mid.tip) * strength * sigma)
    weak = crest & ~ridge & (strength >= float(mid.weak_share) * top)
    paths = mid._bridges(ridge, weak, smooth, ink) if mid.bridge > 0 and ridge.any() else []
    py, px, vx, vy = mid.points(work)
    ctx["mid"] = (py, px, vx, vy)
    direction = (np.degrees(np.arctan2(-ny, nx)) + 90.0) % 180.0
    ys, xs = np.nonzero(ink)
    box = (max(0, ys.min() - 8), min(work.shape[0], ys.max() + 9), max(0, xs.min() - 8), min(work.shape[1], xs.max() + 9))
    ctx["box"] = box

    def crop(ax):
        ax.set_xlim(box[2], box[3]); ax.set_ylim(box[1], box[0]); ax.axis("off"); ax.set_aspect("equal")

    def heat(values, cmap, mask=None):
        def draw(ax):
            v = np.where(ink if mask is None else mask, values, np.nan)
            ax.imshow(v, cmap=cmap, interpolation="nearest")
            crop(ax)
        return draw

    def passfail(passed, failed, extra=None):
        def draw(ax):
            ax.imshow(np.where(ink, 0.18, 0.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            yy, xx = np.nonzero(failed)
            ax.scatter(xx, yy, s=4, color=DROP, linewidths=0)
            yy, xx = np.nonzero(passed)
            ax.scatter(xx, yy, s=5, color=KEEP, linewidths=0)
            if extra:
                extra(ax)
            crop(ax)
        return draw

    def raw(ax):
        ax.imshow(image, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
        for k in range(29):
            ax.axhline(k - 0.5, color="#30333d", lw=0.4); ax.axvline(k - 0.5, color="#30333d", lw=0.4)
        ax.axis("off")

    section("1. Картинка → середина линии", (90, 170, 255))
    cell("1.1 Пиксели MNIST 28×28", panel("1-01-raw", raw), f"обучающая четвёрка, MNIST train №{ctx['mnist']}; яркость 0…255")
    cell("1.2 Увеличение ×8 → 224×224", panel("1-02-work", lambda ax: (ax.imshow(work, cmap="gray"), crop(ax))),
         f"рабочая картинка {work.shape[0]}×{work.shape[1]}, яркость 0…1")
    cond("1.3 Пиксель ≥ порога чернил?", panel("1-03-ink", lambda ax: (ax.imshow(ink, cmap="gray", interpolation="nearest"), crop(ax))),
         f"порог {mid.level:g}: да — чернила ({int(ink.sum())} пикс.), нет — фон")
    cell("1.4 Расстояние до края", panel("1-04-distance", heat(distance, "magma")), f"у каждого пикселя чернил; до {distance.max():.1f} пикс.")
    cell("1.5 Сглаживание", panel("1-05-smooth", heat(smooth, "magma")), f"гаусс σ = {sigma:g} пикс.")
    cell("1.6 Изгиб поперёк линии", panel("1-06-strength", heat(strength, "inferno")),
         "вторые производные (гессиан): сила гребня = −(наименьший изгиб)")
    cell("1.7 Направление поперёк линии", panel("1-07-direction", lambda ax: (ax.imshow(
        np.where(ink[..., None], plt.get_cmap("hsv")(direction / 180.0)[..., :3] * (0.25 + 0.75 * np.clip(strength / top, 0, 1))[..., None], 0)),
        crop(ax))), "собственный вектор гессиана; цвет — угол 0…180°, яркость — сила")
    cond("1.8 Вершина поперёк линии?", panel("1-08-crest", passfail(crest, ink & ~crest)),
         f"выше соседа спереди и сзади поперёк линии и глубже {mid.depth:g} пикс.: да — {int(crest.sum())}")
    cond("1.9 Сила ≥ доли от наибольшей?", panel("1-09-strong", passfail(c_strong, crest & ~c_strong)),
         f"доля {mid.strength_share:g}: да — {int(c_strong.sum())}, нет — {int((crest & ~c_strong).sum())}")
    cond("1.10 Линия, а не пятно?", panel("1-10-line", passfail(c_line, c_strong & ~c_line)),
         f"изгиб вдоль ≤ {mid.blob:g} изгиба поперёк: да — {int(c_line.sum())}, нет — {int((c_strong & ~c_line).sum())}")
    cond("1.11 Не кончик линии?", panel("1-11-tip", passfail(ridge, c_line & ~ridge)),
         f"наклон вдоль ≤ {mid.tip:g}·сила·σ: да — {int(ridge.sum())} точек гребня")
    cell("1.12 Слабые точки гребня", panel("1-12-weak", passfail(ridge, weak)),
         f"оранжевые — гребень, серые — слабые (сила ≥ {mid.weak_share:g} наибольшей): {int(weak.sum())}")

    def bridges(ax):
        passfail(ridge, weak)(ax)
        for path in paths:
            ax.plot(path[:, 1], path[:, 0], color=GOOD, lw=2.5)
    cond("1.13 Разрыв ≤ длины мостика?", panel("1-13-bridges", bridges),
         f"да — мостик по слабым точкам (зелёный), до {mid.bridge:g} пикс.: мостиков {len(paths)}")

    def final(ax):
        ax.imshow(work, cmap="gray", alpha=0.3)
        d = (np.degrees(np.arctan2(vy, vx)) + 90.0) % 180.0
        ax.scatter(px - 0.5, py - 0.5, s=5, c=[SL.hue(a, 180.0) for a in d], linewidths=0)
        crop(ax)
    cell("1.14 Точки середины линии", panel("1-14-midline", final), f"{py.size} точек; у каждой — направление без стрелки (цвет)")


# ---------------------------------------------------------------- этап 1: точки → штрихи (детекторы на картинке)
def stage_strokes(ctx) -> None:
    py, px, vx, vy = ctx["mid"]
    sd = ctx["extractor"].stroke_detectors
    box = ctx["box"]
    y, x = np.asarray(py, float), np.asarray(px, float)
    magnitude = np.hypot(vx, vy)
    ux, uy = vx / magnitude, vy / magnitude
    distance = np.hypot(y[:, None] - y[None, :], x[:, None] - x[None, :])
    cosine = np.abs(ux[:, None] * ux[None, :] + uy[:, None] * uy[None, :]) if sd.axial else ux[:, None] * ux[None, :] + uy[:, None] * uy[None, :]
    reach = float(sd.step) * float(sd.angle_span)
    member = sd._clusters(distance, cosine)
    field = member & (distance <= float(sd.step))
    points = field.sum(axis=1)
    energy = magnitude / float(magnitude.max())
    weight = np.where(field, cosine * energy[None, :], 0.0)
    total = weight.sum(axis=1)
    valid = (points >= int(sd.min_points)) & (total > 0.0)
    safe = np.where(total > 0.0, total, 1.0)
    cy, cx = (weight @ y) / safe, (weight @ x) / safe
    span = member.astype(np.float64)
    double_x, double_y = (vx * vx - vy * vy) / magnitude, 2.0 * vx * vy / magnitude
    angle = np.mod(0.5 * np.arctan2(span @ double_y, span @ double_x), np.pi)
    chosen = sd._insert(y, x, cy, cx, points, valid, angle)
    p = float(ctx["extractor"].work_patch)
    got_y, got_x = cy[chosen] / p - 0.5, cx[chosen] / p - 0.5
    same = len(chosen) == ctx["sy"].size and np.allclose(np.sort(got_y), np.sort(ctx["sy"]), atol=1e-6)
    print(f"штрихи: пересчитано {len(chosen)}, в прогоне {ctx['sy'].size}, совпадают: {same}")
    # Пример — штрих ближе всех к центру цифры: он же окно уровня 0 дальше.
    centre = np.array([ctx["sy"].mean(), ctx["sx"].mean()])
    k = int(np.argmin(np.hypot(got_y - centre[0], got_x - centre[1])))
    c = int(chosen[k])
    ctx["example_stroke"] = int(np.argmin(np.hypot(ctx["sy"] - got_y[k], ctx["sx"] - got_x[k])))
    active = (distance[c] <= reach) & (cosine[c] >= float(sd.similarity))
    allowed = (distance <= reach) & (cosine >= float(sd.similarity))
    near = distance <= sd.NEIGHBOUR
    np.fill_diagonal(near, False)
    wave = np.full(y.size, -1)
    wave[c] = 0
    frontier, step = np.array([c]), 0
    while frontier.size:
        step += 1
        nxt = np.flatnonzero(near[frontier].any(axis=0) & allowed[c] & (wave < 0))
        wave[nxt] = step
        frontier = nxt
    zoom = 1.6 * reach

    def around(ax):
        ax.set_xlim(x[c] - zoom, x[c] + zoom); ax.set_ylim(y[c] + zoom, y[c] - zoom); ax.set_aspect("equal"); ax.axis("off")

    def base_pts(ax, alpha=0.35):
        ax.imshow(ctx["work"], cmap="gray", alpha=0.25)
        ax.scatter(x - 0.5, y - 0.5, s=9, color="#5d6470", alpha=alpha, linewidths=0)

    def p_act(ax):
        base_pts(ax)
        ax.scatter(x[active] - 0.5, y[active] - 0.5, s=16, color=KEEP, linewidths=0)
        ax.add_patch(Circle((x[c] - 0.5, y[c] - 0.5), reach, fill=False, color="#ffd34d", lw=1.5, ls="--"))
        ax.plot(x[c] - 0.5, y[c] - 0.5, "*", color="#ffd34d", ms=16)
        around(ax)

    def p_wave(ax):
        base_pts(ax)
        grown = wave >= 0
        ax.scatter(x[grown] - 0.5, y[grown] - 0.5, s=18, c=wave[grown], cmap="viridis", linewidths=0)
        ax.plot(x[c] - 0.5, y[c] - 0.5, "*", color="#ffd34d", ms=16)
        around(ax)

    def p_field(ax):
        base_pts(ax)
        ax.scatter(x[field[c]] - 0.5, y[field[c]] - 0.5, s=20, color=KEEP, linewidths=0)
        ax.add_patch(Circle((x[c] - 0.5, y[c] - 0.5), float(sd.step), fill=False, color="#ffd34d", lw=2))
        around(ax)

    def p_centre(ax):
        base_pts(ax)
        ax.scatter(x[field[c]] - 0.5, y[field[c]] - 0.5, s=20, color=KEEP, alpha=0.5, linewidths=0)
        t = angle[c]
        L = 1.2 * float(sd.step)
        ax.plot([cx[c] - 0.5 - L * np.cos(t), cx[c] - 0.5 + L * np.cos(t)], [cy[c] - 0.5 + L * np.sin(t), cy[c] - 0.5 - L * np.sin(t)],
                color=SL.hue(np.degrees(t), 180.0), lw=4)
        ax.plot(cx[c] - 0.5, cy[c] - 0.5, "o", color="white", ms=6)
        around(ax)

    def crop(ax):
        ax.set_xlim(box[2], box[3]); ax.set_ylim(box[1], box[0]); ax.axis("off"); ax.set_aspect("equal")

    candidates = np.flatnonzero(valid)
    dense = points[candidates] >= float(sd.dense_share) * float(np.median(points[candidates]))

    def p_cands(ax):
        ax.imshow(ctx["work"], cmap="gray", alpha=0.25)
        ax.scatter(cx[candidates[dense]] - 0.5, cy[candidates[dense]] - 0.5, s=8, color=KEEP, linewidths=0)
        ax.scatter(cx[candidates[~dense]] - 0.5, cy[candidates[~dense]] - 0.5, s=8, color="#3fa9ff", linewidths=0)
        crop(ax)

    def p_insert(ax):
        ax.imshow(ctx["work"], cmap="gray", alpha=0.25)
        skipped = np.setdiff1d(candidates, chosen)
        ax.scatter(cx[skipped] - 0.5, cy[skipped] - 0.5, s=5, color=DROP, linewidths=0)
        for i in chosen:
            ax.add_patch(Circle((cx[i] - 0.5, cy[i] - 0.5), float(sd.step), fill=False, color=GOOD, lw=0.8, alpha=0.8))
        ax.scatter(cx[chosen] - 0.5, cy[chosen] - 0.5, s=14, color=GOOD, linewidths=0)
        crop(ax)

    def p_sticks(ax):
        ax.imshow(ctx["work"], cmap="gray", alpha=0.3)
        for i in chosen:
            t = angle[i]
            L = 0.9 * float(sd.step)
            ax.plot([cx[i] - 0.5 - L * np.cos(t), cx[i] - 0.5 + L * np.cos(t)], [cy[i] - 0.5 + L * np.sin(t), cy[i] - 0.5 - L * np.sin(t)],
                    color=SL.hue(np.degrees(t), 180.0), lw=2.6)
        crop(ax)

    section("2. Точки → штрихи (детекторы над точками, §6.4 на картинке)", (90, 170, 255))
    loop("для каждой точки — центр детектора c", 5, "следующая точка")
    cell("2.1 Активация A(c)", panel("2-01-activation", p_act),
         f"точки ближе {reach:.1f} пикс. с похожим направлением (|cos| ≥ {sd.similarity:g}): {int(active.sum())}")
    cell("2.2 Кластер растёт по соседям", panel("2-02-cluster", p_wave),
         f"шаг — соседи ≤ {sd.NEIGHBOUR:g} пикс. внутри активации; волн {int(wave.max())}, точек {int((wave >= 0).sum())}")
    cell("2.3 Поле детектора", panel("2-03-field", p_field), f"точки кластера ближе шага {sd.step:g} пикс.: {int(points[c])}")
    cond(f"2.4 В поле ≥ {sd.min_points} точек?", panel("2-04-enough", p_field),
         f"да — кандидат; нет — точки мало. Кандидатов {int(valid.sum())} из {y.size}")
    cell("2.5 Центр и угол", panel("2-05-centre", p_centre),
         f"центр — среднее с весом сходство × сила; угол — по удвоенным углам кластера: {np.degrees(angle[c]):.0f}°")
    cell("2.6 Порядок кандидатов", panel("2-06-candidates", p_cands),
         f"сначала плотные (оранж., ≥ {sd.dense_share:g} медианы точек), потом редкие (синие)")
    loop("для каждого кандидата по порядку", 1, "следующий кандидат")
    cond("2.7 Рядом (< шага) уже взят штрих с похожим углом?", panel("2-07-insert", p_insert),
         f"да — пропустить (серые); нет — взять (зелёные круги): {len(chosen)}")
    cell("2.8 Штрихи цифры", panel("2-08-sticks", p_sticks), f"{len(chosen)} штрихов: место (дробное) и угол 0…180°")


# ---------------------------------------------------------------- этап 2: код угла
def stage_angle(ctx) -> None:
    from damp.encoding.elements import StickCoder

    i = ctx["example_stroke"]
    sy, sx, sa, image = ctx["sy"], ctx["sx"], ctx["sa"], ctx["image"]
    a = float(sa[i])
    enc = StickCoder().encoder
    layers = np.asarray(enc.detector_layer_indices())
    bits = np.asarray(enc.detector_bits())
    centres = np.asarray([d.center for d in enc.detectors], float)
    halves = np.asarray([d.half_width for d in enc.detectors], float)
    _, ids = enc.active_detectors(0, np.array([a]))
    ids = np.asarray(ids)
    code = np.asarray(enc.encode_matrix([[a]]), dtype=np.uint8).reshape(-1)
    length = code.size
    mine = np.zeros(length, np.uint8)
    mine[bits[ids]] = 1
    print(f"код угла: пересчитан == кодер: {bool((mine == code).all())}, единиц {int(code.sum())}")
    ctx["angle_code"], ctx["angle"] = code, a

    def p_one(ax):
        ax.imshow(image, cmap="gray", extent=[-0.94, 23.94, 23.94, -0.94], alpha=0.55)
        for yy, xx, aa, bright in zip(sy, sx, sa, np.arange(sy.size) == i):
            t = np.radians(aa)
            ax.plot([xx - 0.45 * np.cos(t), xx + 0.45 * np.cos(t)], [yy + 0.45 * np.sin(t), yy - 0.45 * np.sin(t)],
                    color=SL.hue(aa, 180.0), lw=3 if bright else 1.2, alpha=1 if bright else 0.4)
        ax.add_patch(Circle((sx[i], sy[i]), 1.0, fill=False, color="#ffd34d", lw=2.5))
        ax.set_xlim(-1, 24); ax.set_ylim(24, -1); ax.set_aspect("equal"); ax.axis("off")

    section("3. Код угла штриха (§4.2, §4.4)", (255, 160, 70))
    cell("3.1 Один штрих — окно уровня 0", panel("3-01-stroke", p_one), f"ближайший к центру цифры; угол {a:.0f}° (период 180°)")
    count = int(layers.max()) + 1
    loop("для каждого слоя детекторов угла k (широкий → узкий)", count, "следующий слой")
    band_lo = [int(bits[layers == k].min()) for k in range(count)]
    band_hi = [int(bits[layers == k].max()) for k in range(count)]
    for k in range(count):
        mine_k = np.flatnonzero(layers == k)
        on_k = np.intersect1d(ids, mine_k)

        def p_layer(ax, mine_k=mine_k, on_k=on_k, k=k):
            ax.add_patch(Wedge((0, 0), 1.0, 0, 180, width=0.02, color="#555"))
            for j in mine_k:
                r = 0.55 + 0.4 * ((j - mine_k[0]) % 3) / 2.0
                lo, hi = centres[j] - halves[j], centres[j] + halves[j]
                hit = j in on_k
                ax.add_patch(Wedge((0, 0), r + 0.03, lo, hi, width=0.06 if hit else 0.025,
                                   color=KEEP if hit else LAYER_COLOURS[k % len(LAYER_COLOURS)], alpha=1.0 if hit else 0.35))
            t = np.radians(a)
            ax.plot([0, 1.05 * np.cos(t)], [0, 1.05 * np.sin(t)], color="white", lw=2.5)
            # полоса кода слоя: биты слоя, сработавший — оранжевый
            lo_b, hi_b = band_lo[k], band_hi[k]
            strip = np.zeros(hi_b - lo_b + 1)
            strip[bits[on_k] - lo_b] = 1
            sub = ax.inset_axes([0.05, -0.02, 0.9, 0.12])
            sub.imshow(np.where(strip[None, :] > 0, 1.0, 0.15), cmap="Oranges", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
            sub.axis("off")
            ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.35, 1.1); ax.set_aspect("equal"); ax.axis("off")
        width = float(2 * halves[mine_k].mean())
        cond(f"3.{k + 2} Слой {k}: угол внутри дуги детектора?", panel(f"3-{k + 2:02d}-layer{k}", p_layer),
             f"{mine_k.size} детекторов по {width:.0f}°; да — {on_k.size}: бит {', '.join(str(int(b)) for b in bits[on_k][:4])} (полоса {band_lo[k]}…{band_hi[k]})")
    colours = np.zeros((length, 3))
    for k in range(count):
        colours[band_lo[k]: band_hi[k] + 1] = plt.get_cmap("hsv")(0.83 * k / max(1, count - 1))[:3]
    cell("3.12 Код угла = OR битов", panel("3-12-angle-code", lambda ax: board(ax, code, colours)),
         f"{length} бит, единиц {int(code.sum())}; цвет — слой (широкий красный … узкий фиолетовый)")


# ---------------------------------------------------------------- этап 2: код места уровня 0
def stage_place(ctx) -> None:
    from damp.cycle.plane_fields import PlaneFieldBuilder
    from damp.encoding.keyhole_code import KeyholeCoder

    settings = ctx["settings"]
    sy, sx, image, i = ctx["sy"], ctx["sx"], ctx["image"], ctx["example_stroke"]
    cy, cx = sy.mean(), sx.mean()
    R = float(np.hypot(sy - cy, sx - cx).max())
    step = PlaneFieldBuilder.OFFSET_STEP
    raw = np.array([sy[i] - cy, sx[i] - cx]) / R
    shift = np.round(raw / step) * step
    unit = float(settings.LADDER_CLUSTER_PLACE_UNIT)
    value = shift / unit
    coder = KeyholeCoder()
    place = coder.place(0)
    rows, ids = place.active_detectors(0, value[None, :])
    ids = np.asarray(ids)
    dets = place.detectors
    layers = np.asarray(place.detector_layer_indices())
    copies = coder.place_bits(0)
    ctx.update(place_value=value, place_ids=ids, keyhole_coder=coder)

    def p_offset(ax):
        ax.imshow(image, cmap="gray", extent=[-0.94, 23.94, 23.94, -0.94], alpha=0.55)
        ax.add_patch(Circle((cx, cy), R, fill=False, color="#888", lw=1.5, ls="--"))
        ax.annotate("", xy=(sx[i], sy[i]), xytext=(cx, cy), arrowprops=dict(arrowstyle="->", color="#ffd34d", lw=2.5))
        ax.plot(cx, cy, "o", color="white", ms=7)
        ax.set_xlim(-1, 24); ax.set_ylim(24, -1); ax.set_aspect("equal"); ax.axis("off")

    def p_round(ax):
        g = np.arange(-1, 1 + 1e-9, step)
        gy_, gx_ = np.meshgrid(g, g)
        ax.scatter(gx_, gy_, s=4, color="#444")
        ax.add_patch(Circle((0, 0), 1, fill=False, color="#888", ls="--"))
        ax.plot(raw[1], raw[0], "o", color="#ffd34d", ms=8)
        ax.plot(shift[1], shift[0], "s", color=KEEP, ms=9)
        ax.annotate("", xy=(shift[1], shift[0]), xytext=(raw[1], raw[0]), arrowprops=dict(arrowstyle="->", color=INK))
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(1.1, -1.1); ax.set_aspect("equal"); ax.axis("off")

    section("4. Место штриха — от центра цифры, круги места (§4.3, П.4)", (255, 160, 70))
    cell("4.1 Смещение от центра цифры", panel("4-01-offset", p_offset),
         f"центр — середина штрихов; R — до дальнего штриха = {R:.1f} кл.; смещение / R = ({raw[0]:+.2f}, {raw[1]:+.2f})")
    cell("4.2 Округление до шага решётки", panel("4-02-round", p_round),
         f"шаг {step:g} → ({shift[0]:+.3f}, {shift[1]:+.3f}); в плоскость мест — / {unit:g}")
    count = int(layers.max()) + 1
    loop("для каждого слоя кругов места k (широкий → узкий)", count, "следующий слой")
    extent = float(settings.LADDER_PLACE_EXTENT)
    for k in range(count):
        mine_k = np.flatnonzero(layers == k)
        on_k = np.intersect1d(ids, mine_k)

        def p_layer(ax, mine_k=mine_k, on_k=on_k, k=k):
            for j in mine_k:
                d = dets[j]
                hit = j in on_k
                ax.add_patch(Circle((d.center_x, d.center_y), d.radius, fill=hit, color=KEEP if hit else LAYER_COLOURS[k % len(LAYER_COLOURS)],
                                    alpha=0.35 if hit else 0.25, lw=0.6))
                if hit:
                    ax.add_patch(Circle((d.center_x, d.center_y), d.radius, fill=False, color=KEEP, lw=2))
            ax.add_patch(Circle((0, 0), 1 / unit, fill=False, color="#888", ls="--", lw=1))
            ax.plot(value[1], value[0], "o", color="white", ms=7)
            ax.set_xlim(-extent, extent); ax.set_ylim(extent, -extent); ax.set_aspect("equal"); ax.axis("off")
        side = int(round(np.sqrt(mine_k.size)))
        cond(f"4.{k + 3} Слой {k}: точка внутри круга?", panel(f"4-{k + 3:02d}-place{k}", p_layer),
             f"{mine_k.size} кругов (~{side}×{side}), радиус {dets[mine_k[0]].radius:.2f}; да — {on_k.size}")
    code = np.zeros(int(place.code_length), np.uint8)
    code[np.asarray(place.detector_bits())[ids]] = 1
    ctx["place_code"] = code
    cell("4.13 Код места", panel("4-13-place-code", lambda ax: board(ax, code)),
         f"{code.size} бит, единиц {int(code.sum())}; у окна уровня 0 каждый круг даёт {copies} бит (копии)")


# ---------------------------------------------------------------- этап 2: код окна уровня 0
def stage_window(ctx) -> None:
    settings = ctx["settings"]
    coder = ctx["keyhole_coder"]
    enc = coder.encoder(0)
    ids = ctx["place_ids"]
    codes, _, firing = next(item for item in SL.GOT["codes"] if int(item[2].sets) == int(np.asarray(ctx["groups"].atoms.y).shape[0]))
    window = ctx["s0"] + ctx["example_stroke"]
    real = np.unpackbits(np.asarray(codes.packed)[window], count=int(codes.code_length))
    place_len, content_len = int(codes.place_length), int(codes.content_length)
    place_bits = np.zeros(place_len, np.uint8)
    place_bits[np.asarray(enc._place_bits)[ids].reshape(-1)] = 1
    masks = enc.address_masks(content_len)
    angle_on = np.flatnonzero(ctx["angle_code"])
    layers = np.asarray(enc.place.detector_layer_indices())
    content = np.full(content_len, -1)
    for d in sorted(ids, key=lambda j: layers[j]):
        content[angle_on ^ int(masks[d])] = layers[d]
    mine = np.concatenate([place_bits, (content >= 0).astype(np.uint8)])
    real_place, real_content = real[:place_len], real[place_len:]
    dropped = (content >= 0) & (real_content == 0)
    print(f"код окна: место совпало {bool((place_bits == real_place).all())}, содержание лишнее {int(((content < 0) & (real_content > 0)).sum())}, "
          f"слито {int(dropped.sum())}")
    ctx["window_codes"] = (codes, firing, window)
    d0 = int(sorted(ids, key=lambda j: layers[j])[0])

    def p_place(ax):
        board(ax, place_bits)

    def p_mask(ax):
        one = np.zeros(content_len, np.uint8)
        one[angle_on ^ int(masks[d0])] = 1
        top = ax.inset_axes([0.0, 0.62, 1.0, 0.36])
        board(top, ctx["angle_code"], side=64)
        bottom = ax.inset_axes([0.0, 0.0, 1.0, 0.58])
        board(bottom, one, side=int(np.ceil(np.sqrt(content_len)) * 2))
        ax.axis("off")

    def p_content(ax):
        colours = np.zeros((content_len, 3))
        on = content >= 0
        colours[on] = [plt.get_cmap("hsv")(0.83 * c / 9.0)[:3] for c in content[on]]
        board(ax, on.astype(np.uint8), colours)

    def p_merge(ax):
        colours = np.zeros((content_len, 3))
        colours[(content >= 0) & ~dropped] = (1.0, 0.42, 0.16)
        colours[dropped] = (0.36, 0.39, 0.44)
        board(ax, (content >= 0).astype(np.uint8), colours)

    def p_window(ax):
        colours = np.zeros((real.size, 3))
        colours[:place_len] = (1.0, 0.45, 0.15)
        colours[place_len:] = (0.3, 0.6, 1.0)
        board(ax, real, colours)

    section("5. Код окна: место (OR) + содержание под маской места (П.4–П.5)", (255, 160, 70))
    cell("5.1 Биты места", panel("5-01-place-bits", p_place), f"{place_len} бит: {len(ids)} кругов × {coder.place_bits(0)} бит, единиц {int(place_bits.sum())}")
    loop("для каждого сработавшего круга места d", 1, "следующий круг")
    cell("5.2 Код угла под маской круга", panel("5-02-mask", p_mask),
         f"«address»: бит i → i ⊕ m_d (перестановка); вверху код угла, внизу — куда легли его {angle_on.size} единиц")
    cell("5.3 Содержание = OR по кругам", panel("5-03-content", p_content), f"{content_len} бит; цвет — слой круга, давшего бит")
    cond("5.4 Единиц больше бюджета?", panel("5-04-merge", p_merge),
         f"насыщение {settings.LADDER_BINDING_SATURATION:g}: да — слить цвета, первыми уходят узкие слои (серые: {int(dropped.sum())})")
    cell("5.5 Код окна уровня 0", panel("5-05-window", p_window),
         f"{real.size} бит, единиц {int(real.sum())}: место (оранж.) + содержание (синие)")


# ---------------------------------------------------------------- этап 3: раскладка (первая подача)
def stage_layout(ctx) -> None:
    import damp.layout.damp_layout as DL
    from damp.layout.level_memory import MemoryMap

    run = ctx["run"]
    prim = run.primary
    memory = prim.memory
    layout0 = memory.layout
    # коды и цифры точек в порядке раскладки: раскладка заново сохраняет этот порядок (группы по цвету уже собраны)
    bits = np.asarray(layout0.code_table.bits(), dtype=np.uint8)
    labels = np.asarray([int(v) for v in layout0.code_table.labels()], dtype=np.int64)
    primary = ctx["ladder"].primary_map()
    angle_of_row = np.asarray(primary.angle, float)
    codes, _, _ = ctx["window_codes"]
    print(f"раскладка уровня 0: точек {bits.shape[0]}, окон {codes.count}")
    # Раскладка заново тем же кодом с остановками: каждый шаг записывается, ячейки — на шагах 0, 1, 2, 4, 8, …
    record, snaps = [], {}
    original = DL.Layout.step

    def step(self, *args, **kwargs):
        index = len(record)
        mode = kwargs.get("mode")
        if index == 0:
            snaps[("start", 0)] = self.cells_array().copy()
        if mode == "short" and not any(r[0] == "short" for r in record):
            snaps[("short", index)] = self.cells_array().copy()
            try:
                ctx["pair_examples"] = pair_examples(self, kwargs.get("local_radius") or 3)
            except Exception as error:  # noqa: BLE001  пример пары — для показа; без него схема всё равно собирается
                print(f"пример пары не снят: {error!r}")
                ctx["pair_examples"] = {}
        swaps = original(self, *args, **kwargs)
        record.append((mode, kwargs.get("pair_radius"), int(swaps), float(self._lambda)))
        if (index + 1) & index == 0:
            snaps[(mode, index + 1)] = self.cells_array().copy()
        return swaps

    def pair_examples(layout, radius):
        rng = np.random.default_rng(0)
        cells = layout.cells_array()
        found = {}
        for _ in range(4000):
            a = int(rng.integers(cells.shape[0]))
            y1, x1 = (int(v) for v in cells[a])
            y2, x2 = y1 + int(rng.integers(-2, 3)), x1 + int(rng.integers(-2, 3))
            if not (0 <= y2 < layout.height and 0 <= x2 < layout.width) or (y1, x1) == (y2, x2):
                continue
            ok, gain = layout._should_swap(y1, x1, y2, x2, mode="short", local_radius=radius)
            if ok not in found:
                idx_a, idx_b = layout.grid[y1][x1], layout.grid[y2][x2]
                if idx_b is None:
                    continue
                ec, es = layout._pair_energy_short(idx_a, (y1, x1), idx_b, (y2, x2), radius=max(radius, 3))
                window = [(yy, xx, layout.grid[yy][xx]) for yy in range(y1 - radius - 2, y1 + radius + 3) for xx in range(x1 - radius - 2, x1 + radius + 3)
                          if 0 <= yy < layout.height and 0 <= xx < layout.width]
                found[ok] = dict(a=(y1, x1), b=(y2, x2), ia=idx_a, ib=idx_b, before=ec, after=es, window=window, radius=radius)
            if len(found) == 2:
                break
        return found

    DL.Layout.step = step
    try:
        fresh = MemoryMap().lay_out(bits, labels, "atomic-level0")
    finally:
        DL.Layout.step = original
    final = fresh.cells_array()
    hue = np.asarray([SL.hue(a, 180.0) for a in angle_of_row[: bits.shape[0]]])
    W, H = int(fresh.width), int(fresh.height)

    def p_snap(cells):
        def draw(ax):
            ax.scatter(cells[:, 1], cells[:, 0], s=1.2, c=hue, marker="s", linewidths=0)
            ax.set_xlim(-1, W); ax.set_ylim(H, -1); ax.set_aspect("equal"); ax.axis("off")
        return draw

    # квантили сходств — порог λ (§7.1.7)
    rng = np.random.default_rng(0)
    sample = bits[rng.choice(bits.shape[0], size=min(1500, bits.shape[0]), replace=False)].astype(np.float64)
    norm = np.sqrt(sample.sum(axis=1))
    sims = (sample @ sample.T) / np.outer(norm, norm)
    sims = sims[np.triu_indices_from(sims, 1)]
    lam0 = float(record[0][3]) if record else float(fresh.lambda_threshold)

    def p_hist(ax):
        ax.hist(sims, bins=80, color="#6c7385")
        ax.axvline(lam0, color=KEEP, lw=2)
        ax.text(lam0, ax.get_ylim()[1] * 0.9, f" λ = {lam0:.2f}", color=KEEP, fontsize=11)
        chart_axes(ax)

    def p_swaps(ax):
        steps = np.arange(len(record))
        ratio = np.asarray([r[2] for r in record], float)
        ax.plot(steps, ratio, color="#6c7385", lw=0.8)
        rad = np.asarray([r[1] or 0 for r in record], float)
        ax2 = ax.twinx()
        ax2.plot(steps, rad, color=KEEP, lw=1.5)
        ax2.tick_params(colors=KEEP, labelsize=8)
        short0 = next((k for k, r in enumerate(record) if r[0] == "short"), None)
        if short0 is not None:
            ax.axvline(short0, color=GOOD, lw=1.2, ls="--")
        chart_axes(ax)

    def p_pair(example, ok):
        def draw(ax):
            if not example:
                ax.axis("off"); return
            for yy, xx, idx in example["window"]:
                colour = hue[idx] if idx is not None else (0.1, 0.1, 0.12)
                ax.add_patch(plt.Rectangle((xx - 0.5, yy - 0.5), 1, 1, color=colour))
            for (yy, xx), c in ((example["a"], "white"), (example["b"], "#ffd34d")):
                ax.add_patch(plt.Rectangle((xx - 0.5, yy - 0.5), 1, 1, fill=False, color=c, lw=3))
            ax.annotate("", xy=(example["b"][1], example["b"][0]), xytext=(example["a"][1], example["a"][0]),
                        arrowprops=dict(arrowstyle="<->", color="white", lw=2))
            ys = [w[0] for w in example["window"]]; xs = [w[1] for w in example["window"]]
            ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6); ax.set_ylim(max(ys) + 0.6, min(ys) - 0.6); ax.set_aspect("equal"); ax.axis("off")
        return draw

    long_snaps = sorted((k for k in snaps if k[0] == "long"), key=lambda k: k[1])
    short_snaps = sorted((k for k in snaps if k[0] == "short"), key=lambda k: k[1])
    pick_long = [long_snaps[j] for j in sorted({0, len(long_snaps) // 3, 2 * len(long_snaps) // 3, len(long_snaps) - 1}) if long_snaps]
    examples = ctx.get("pair_examples", {})
    section("6. Первая подача: раскладка кодов окон на карту (§5)", (70, 205, 140))
    cond("6.1 Код окна совпал с уже взятым образцом?", panel("6-01-dictionary", p_hist),
         f"сходство ≥ {ctx['settings'].LADDER_PRIMARY_MERGE_SIMILARITY:g} — слить; образцов {bits.shape[0]} из {codes.count} окон. "
         f"Гистограмма сходств пар: λ — квантиль")
    cell("6.2 Случайная расстановка", panel("6-02-start", p_snap(snaps[("start", 0)])),
         f"сетка {W}×{H}, пустых клеток ~{ctx['settings'].LAYOUT_EMPTY_RATIO:.0%}; цвет — угол штриха")
    loop("дальняя фаза: для каждого шага — пары точек в радиусе r", 2 + len(pick_long), "следующий шаг")
    cell("6.3 Пара точек", panel("6-03-pair", p_pair(examples.get(True), True)),
         "первая — с весом по энергии (§5.9.3), вторая — в радиусе r")
    ex_t, ex_f = examples.get(True), examples.get(False)
    cond("6.4 Энергия после обмена лучше?", panel("6-04-swap", p_pair(ex_f, False)),
         (f"да — обменять (пример слева: {ex_t['before']:.2f} → {ex_t['after']:.2f}); " if ex_t else "")
         + (f"нет — оставить (здесь {ex_f['before']:.2f} → {ex_f['after']:.2f})" if ex_f else ""))
    for k in pick_long:
        r = record[k[1] - 1] if k[1] > 0 else record[0]
        cell(f"6.{5 + pick_long.index(k)} После {k[1]} шагов", panel(f"6-05-long{k[1]}", p_snap(snaps[k])), f"радиус пар {r[1]}, λ {r[3]:.2f}, обменов на шаге {r[2]}")
    n = 5 + len(pick_long)
    cond(f"6.{n} Обменов меньше порога?", panel("6-swaps", p_swaps),
         "да — радиус меньше (оранж.) и λ выше; радиус дошёл до наименьшего — ближняя фаза (зелёная черта)")
    if short_snaps:
        cell(f"6.{n + 1} Начало ближней фазы", panel("6-short", p_snap(snaps[short_snaps[0]])), "пары соседей, энергия по окрестности (§5.5.2)")
    cell(f"6.{n + 2} Карта уровня 0", panel("6-final", p_snap(np.asarray(primary.cells, float))),
         f"из прогона: {np.asarray(primary.cells).shape[0]} точек; пинвилы — углы по кругу")
    ctx["layout_record"] = record


# ---------------------------------------------------------------- этап 4: детекторы над картой
def stage_detectors(ctx) -> None:
    from scipy import ndimage

    settings = ctx["settings"]
    prim = ctx["run"].primary
    memory, detectors = prim.memory, prim.detectors
    cells = np.asarray(memory.layout.cells_array(), float)
    energies = np.asarray(memory.energies, float)
    bits = np.asarray(memory.dictionary.bits()).astype(np.float64)
    table = detectors.detectors
    layer = np.asarray(detectors.layer_index)
    specs = list(settings.ACTIVATION_DETECTOR_LAYER_SPECS)
    side = float(max(memory.layout.width, memory.layout.height))
    mu_e = float(settings.ACTIVATION_ENERGY_THRESHOLD)
    top = energies.max() if energies.size else 1.0
    # пример — самый крупный детектор слоя 0
    first = [j for j in range(len(table)) if layer[j] == 0]
    d = max(first, key=lambda j: table[j].radius)
    centre = np.asarray(table[d].center, float)
    c = int(np.argmin(np.hypot(cells[:, 0] - centre[0], cells[:, 1] - centre[1])))
    lam, share, minimum = specs[0]
    if getattr(detectors, "layer_thresholds", None) is not None and len(detectors.layer_thresholds):
        lam = float(detectors.layer_thresholds[0])  # λ_d слоя в прогоне — квантиль сходств кодов карты (§7.1.7)
    r_a = share * side
    norm = np.sqrt(bits.sum(axis=1))
    sim = (bits @ bits[c]) / (norm * norm[c] + 1e-12)
    near = np.hypot(cells[:, 0] - cells[c, 0], cells[:, 1] - cells[c, 1]) <= r_a
    act = near & (sim >= lam)
    # DBSCAN на клетках: сосед — ближе eps, ядро — ≥ min_samples точек активации рядом; кластер — от центра
    eps, mins = float(settings.ACTIVATION_DBSCAN_EPS), int(settings.ACTIVATION_DBSCAN_MIN_SAMPLES)
    pts = np.flatnonzero(act)
    P = cells[pts]
    dd = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1]) <= eps
    core = dd.sum(axis=1) >= mins
    start = int(np.flatnonzero(pts == c)[0]) if c in pts else int(np.argmin(np.hypot(P[:, 0] - cells[c, 0], P[:, 1] - cells[c, 1])))
    seen = np.zeros(pts.size, bool)
    seen[start] = True
    queue = [start]
    while queue:
        q = queue.pop()
        if not core[q]:
            continue
        for j in np.flatnonzero(dd[q] & ~seen):
            seen[j] = True
            queue.append(j)
    cluster = pts[seen]
    fill_cell = float(settings.ACTIVATION_DETECTOR_FILL_CELL)
    dist = np.hypot(cells[cluster, 0] - centre[0], cells[cluster, 1] - centre[1])
    radii = np.arange(1, int(np.ceil(dist.max() if dist.size else 1)) + 1)
    counts = np.asarray([(dist <= r).sum() for r in radii])
    fill = counts / (np.pi * (radii + fill_cell) ** 2)
    ok = counts >= minimum
    W, H = memory.layout.width, memory.layout.height
    primary = ctx["ladder"].primary_map()
    hue = [SL.hue(a, 180.0) for a in primary.angle]

    def zoom(ax, r=None):
        r = r or max(8.0, r_a * 0.7)
        ax.set_xlim(centre[1] - r, centre[1] + r); ax.set_ylim(centre[0] + r, centre[0] - r); ax.set_aspect("equal"); ax.axis("off")

    def p_energy(ax):
        ax.scatter(cells[:, 1], cells[:, 0], s=1.3, c=energies, cmap="magma", marker="s", linewidths=0)
        ax.set_xlim(-1, W); ax.set_ylim(H, -1); ax.set_aspect("equal"); ax.axis("off")

    def p_centres(ax):
        good = energies >= mu_e * top
        ax.scatter(cells[~good, 1], cells[~good, 0], s=1.3, color="#2a2d36", marker="s", linewidths=0)
        ax.scatter(cells[good, 1], cells[good, 0], s=1.3, color=KEEP, marker="s", linewidths=0)
        ax.set_xlim(-1, W); ax.set_ylim(H, -1); ax.set_aspect("equal"); ax.axis("off")

    def p_act(ax):
        ax.scatter(cells[:, 1], cells[:, 0], s=14, c=hue, marker="s", linewidths=0, alpha=0.25)
        ax.scatter(cells[act, 1], cells[act, 0], s=14, color=KEEP, marker="s", linewidths=0)
        ax.add_patch(Circle((cells[c, 1], cells[c, 0]), r_a, fill=False, color="#ffd34d", ls="--", lw=1.5))
        ax.plot(cells[c, 1], cells[c, 0], "*", color="#ffd34d", ms=14)
        zoom(ax, r_a * 1.1)

    def p_db(ax):
        ax.scatter(cells[act, 1], cells[act, 0], s=14, color=DROP, marker="s", linewidths=0)
        ax.scatter(cells[cluster, 1], cells[cluster, 0], s=14, color=KEEP, marker="s", linewidths=0)
        ax.plot(cells[c, 1], cells[c, 0], "*", color="#ffd34d", ms=14)
        zoom(ax, r_a * 1.1)

    def p_fill(ax):
        ax.plot(radii, fill, color="#6c7385", lw=1.5)
        ax.scatter(radii[ok], fill[ok], s=18, color=KEEP, zorder=3)
        ax.axvline(table[d].radius, color=GOOD, lw=1.5, ls="--")
        ax.set_xlabel("радиус, клеток", color=MUTED, fontsize=9)
        chart_axes(ax)

    def p_layer(k):
        def draw(ax):
            ax.scatter(cells[:, 1], cells[:, 0], s=1.0, c=hue, marker="s", linewidths=0, alpha=0.35)
            for j in np.flatnonzero(layer == k):
                t = table[j]
                ax.add_patch(Circle((t.center[1], t.center[0]), t.radius, fill=False, color=LAYER_COLOURS[k], lw=0.6))
            ax.set_xlim(-1, W); ax.set_ylim(H, -1); ax.set_aspect("equal"); ax.axis("off")
        return draw

    section("7. Детекторы над картой: слои иерархии (§6.3–§6.4, §7.1.10)", (176, 123, 255))
    cell("7.1 Энергия точек Ê", panel("7-01-energy", p_energy), "Σ сходство / расстояние по соседям (§5.7); светлее — выше")
    loop(f"для каждого слоя k = 0…{len(specs) - 1}: пока {settings.ACTIVATION_DETECTOR_ATTEMPTS_WITHOUT_CHANGE} попыток подряд без нового", 6, "следующая попытка")
    cond("7.2 Энергия центра ≥ μ_e?", panel("7-02-centres", p_centres), f"μ_e = {mu_e:g} от наибольшей: оранжевые могут быть центром")
    cell("7.3 Активация вокруг центра", panel("7-03-activation", p_act),
         f"слой 0: ближе r_a = {share:g}·сторона = {r_a:.0f} кл. и сходство ≥ λ_d = {lam:.2f}: {int(act.sum())} точек")
    cell("7.4 Кластер DBSCAN", panel("7-04-dbscan", p_db), f"eps {eps:g}, ядро ≥ {mins}: связный с центром кусок — {cluster.size} точек")
    cell("7.5 Радиус — наибольшее заполнение", panel("7-05-fill", p_fill),
         f"f = n / π(r + {fill_cell:g})², n ≥ {int(minimum)}: зелёная черта — радиус детектора прогона {table[d].radius:.1f}")
    cond("7.6 Центр не в поле другого детектора?", panel("7-06-overlap", p_layer(0)), "да — детектор слоя; нет — попытка не засчитана")
    cell("7.7 Детекторы слоя 0", panel("7-07-layer0", p_layer(0)), f"{int((layer == 0).sum())} детекторов")
    for k in range(1, len(specs)):
        cell(f"7.{7 + k} Слой {k}", panel(f"7-{7 + k:02d}-layer{k}", p_layer(k)),
             f"λ_d {float(detectors.layer_thresholds[k]) if getattr(detectors, 'layer_thresholds', None) is not None else specs[k][0]:.2f}, "
             f"r_a {specs[k][1]:g}·сторона, ≥ {int(specs[k][2])} точек: {int((layer == k).sum())} детекторов")


# ---------------------------------------------------------------- этап 5: вторая подача — срабатывание и код детали
def stage_activation(ctx) -> None:
    from damp.activation.set_embedding import DetectorEmbeddings

    settings = ctx["settings"]
    prim = ctx["run"].primary
    memory, detectors = prim.memory, prim.detectors
    codes, firing, window = ctx["window_codes"]
    cells = np.asarray(memory.layout.cells_array(), float)
    W, H = memory.layout.width, memory.layout.height
    bits = np.asarray(memory.dictionary.bits()).astype(np.float64)
    wb = np.unpackbits(np.asarray(codes.packed)[window], count=int(codes.code_length)).astype(np.float64)
    sim = (bits @ wb) / (np.sqrt(bits.sum(axis=1)) * np.sqrt(wb.sum()) + 1e-12)
    lam = float(settings.ACTIVATION_LAMBDA_THRESHOLD)
    _, lit_points, lit_values = detectors.lit(np.asarray(codes.packed)[window:window + 1])
    lit_points, lit_values = np.asarray(lit_points, int), np.asarray(lit_values, float)
    rows, cols, levels = np.asarray(firing.rows), np.asarray(firing.columns), np.asarray(firing.levels, float)
    pick = rows == window
    fired, fired_level = cols[pick], levels[pick]
    sigma = int(settings.ACTIVATION_SATURATION_LIMIT)
    _, kept_columns = DetectorEmbeddings(detectors=int(detectors.built), saturation_limit=sigma, target_density=0.1, min_length=64,
                                         max_length=8192, seed=0, name="sigma").strongest(np.zeros(fired.size, dtype=np.int64), fired, fired_level)
    kept = np.isin(fired, kept_columns)
    details, length = SL.GOT["details0"]
    code = np.unpackbits(np.asarray(details)[window], count=int(length))
    table = detectors.detectors
    hue = [SL.hue(a, 180.0) for a in ctx["ladder"].primary_map().angle]
    best = int(fired[np.argmax(fired_level)]) if fired.size else 0
    t = table[best]
    inside = np.hypot(cells[:, 0] - t.center[0], cells[:, 1] - t.center[1]) <= t.radius
    lit_mask = np.zeros(cells.shape[0], bool)
    lit_mask[lit_points] = True
    ctx["fired0"] = (fired, fired_level, kept)

    def full(ax):
        ax.set_xlim(-1, W); ax.set_ylim(H, -1); ax.set_aspect("equal"); ax.axis("off")

    def p_sim(ax):
        ax.scatter(cells[:, 1], cells[:, 0], s=1.3, c=sim, cmap="viridis", marker="s", linewidths=0)
        full(ax)

    def p_lit(ax):
        ax.scatter(cells[:, 1], cells[:, 0], s=1.3, color="#2a2d36", marker="s", linewidths=0)
        strong = sim >= lam
        ax.scatter(cells[strong, 1], cells[strong, 0], s=1.3, color=DROP, marker="s", linewidths=0)
        ax.scatter(cells[lit_points, 1], cells[lit_points, 0], s=2.0, color=KEEP, marker="s", linewidths=0)
        full(ax)

    def p_one(ax):
        r = t.radius * 2.2
        ax.scatter(cells[:, 1], cells[:, 0], s=40, c=hue, marker="s", linewidths=0, alpha=0.25)
        ax.scatter(cells[lit_mask, 1], cells[lit_mask, 0], s=40, color=KEEP, marker="s", linewidths=0)
        ax.add_patch(Circle((t.center[1], t.center[0]), t.radius, fill=False, color="#ffd34d", lw=2.5))
        ax.set_xlim(t.center[1] - r, t.center[1] + r); ax.set_ylim(t.center[0] + r, t.center[0] - r); ax.set_aspect("equal"); ax.axis("off")

    def p_fired(ax):
        ax.scatter(cells[:, 1], cells[:, 0], s=1.0, c=hue, marker="s", linewidths=0, alpha=0.3)
        for column, lvl, k in sorted(zip(fired, fired_level, kept), key=lambda item: item[2]):
            tt = table[int(column)]
            ax.add_patch(Circle((tt.center[1], tt.center[0]), tt.radius, fill=False, color=KEEP if k else DROP, lw=0.8 + 2 * lvl))
        full(ax)

    order = np.argsort(-fired_level, kind="stable")

    def p_sigma(ax):
        ranked = fired_level[order]
        ax.bar(np.arange(1, ranked.size + 1), ranked, color=[KEEP if k else DROP for k in kept[order]], width=0.8)
        ax.axvline(sigma + 0.5, color=INK, lw=1.2, ls="--")
        ax.text(sigma + 1, max(ranked.max(), 1e-3) * 0.95, f"σ = {sigma}", color=INK, fontsize=11, va="top")
        chart_axes(ax)

    section("8. Вторая подача: окно зажигает карту → детекторы → код детали (§6.1, §6.5)", (176, 123, 255))
    loop("для каждого окна уровня (здесь — штрих четвёрки)", 7, "следующее окно")
    cell("8.1 Сходство кода окна с точками", panel("8-01-similarity", p_sim), "косинус кода окна с кодом каждой точки карты")
    cond("8.2 Сходство ≥ λ_a и энергия активации ≥ μ_a?", panel("8-02-lit", p_lit),
         f"λ_a = {lam:g} (красные), μ_a = {settings.ACTIVATION_LOCAL_ENERGY_THRESHOLD:g}: зажглись {lit_points.size} (оранж.)")
    loop("для каждого детектора d", 2, "следующий детектор")
    cell("8.3 Уровень детектора E(d)", panel("8-03-detector", p_one), "Σ активация × энергия точек в круге / e_d; нормировка по обучающим окнам")
    cond(f"8.4 E(d) ≥ μ_d = {settings.ACTIVATION_DETECTOR_THRESHOLD:g}?", panel("8-04-fired", p_fired),
         f"да — сработал: {fired.size} детекторов (красные — σ сильнейших)")
    cond(f"8.5 Детектор среди σ = {sigma} сильнейших?", panel("8-05-sigma", p_sigma), f"да — его бит в код: взято {int(kept.sum())}")
    cell("8.6 Код детали (эмбеддинг)", panel("8-06-detail", lambda ax: board(ax, code)),
         f"{code.size} бит, единиц {int(code.sum())}: случайные биты взятых детекторов, плотность ~0.1")
    cell("8.7 Детали всех штрихов", panel("8-07-details", lambda ax: draw_details(ax, ctx, 0)), "у каждого окна уровня — своя деталь (код)")


def draw_details(ax, ctx, level, highlight=None):
    image = ctx["image"]
    ax.imshow(image, cmap="gray", extent=[-0.94, 23.94, 23.94, -0.94], alpha=0.55)
    sy, sx, sa = ctx["sy"], ctx["sx"], ctx["sa"]
    for yy, xx, aa in zip(sy, sx, sa):
        t = np.radians(aa)
        ax.plot([xx - 0.45 * np.cos(t), xx + 0.45 * np.cos(t)], [yy + 0.45 * np.sin(t), yy - 0.45 * np.sin(t)],
                color=SL.hue(aa, 180.0), lw=2.2 if level == 0 else 1.0, alpha=1 if level == 0 else 0.4)
    ax.set_xlim(-1, 24); ax.set_ylim(24, -1); ax.set_aspect("equal"); ax.axis("off")


# ---------------------------------------------------------------- этап 6: пятна и окна следующего уровня
def stage_spots(ctx) -> None:
    settings = ctx["settings"]
    picture = ctx["picture"]
    keyholes1, _ = SL.GOT["embed"][1]
    pieces = keyholes1.pieces
    mine = np.flatnonzero(np.asarray(pieces.picture) == picture)
    centre = np.asarray(pieces.centre, float)
    extent = np.asarray(keyholes1.extent, float)
    middle = np.array([ctx["sy"].mean(), ctx["sx"].mean()])
    w = int(mine[np.argmin(np.hypot(centre[mine, 0] - middle[0], centre[mine, 1] - middle[1]))])
    cy, cx, radius = float(centre[w, 0]), float(centre[w, 1]), float(extent[w])
    members = np.flatnonzero(np.hypot(ctx["sy"] - cy, ctx["sx"] - cx) <= radius)
    run = ctx["run"]
    level1 = next(item for item in run.levels if int(item.number) == 1)
    elements0 = getattr(level1, "elements", None)
    image = ctx["image"]

    def p_windows(ax):
        draw_details(ax, ctx, 1)
        ax.scatter(centre[mine, 1], centre[mine, 0], s=30, color="#4fd1c5", edgecolors="white", linewidths=0.5, zorder=5)
        ax.add_patch(Circle((cx, cy), radius, fill=False, color="#ffd34d", lw=2.5))
        ax.scatter(ctx["sx"][members], ctx["sy"][members], s=50, color=KEEP, zorder=6)

    neighbours = ctx["settings"].LADDER_CLUSTER_MIN_POINTS
    n1 = neighbours[0] if isinstance(neighbours, (list, tuple)) else neighbours
    section("9. Пятна и окна следующего уровня (§6.4.3, П.7)", (70, 205, 140))
    cond(f"9.1 Рядом похожая деталь сильнее (сходство ≥ {settings.LADDER_INHIBITION_SIMILARITY:g})?",
         panel("9-01-inhibition", lambda ax: draw_details(ax, ctx, 0)),
         f"да — погаснуть (торможение соседей, радиус {settings.LADDER_INHIBITION_RADIUS:g}); нет — деталь идёт выше")
    loop("для каждой оставшейся детали — центр окна уровня 1", 2, "следующий центр")
    cond(f"9.2 В круге ≥ {n1} похожих деталей?", panel("9-02-window", p_windows),
         f"растить радиус, пока не наберётся; окно уровня 1: радиус {radius:.1f} кл., в нём {members.size} штрихов")
    cond(f"9.3 Окно похоже на взятое (≥ {settings.LADDER_MERGE_SIMILARITY:g})?", panel("9-03-windows", p_windows),
         f"да — слить с образцом; окон уровня 1 на четвёрке: {mine.size}")
    mark("level_back_from")
    cond(f"9.4 Окно видит ≥ {settings.LADDER_TOP_COVERAGE:.0%} цифры?", panel("9-04-coverage", p_windows),
         "нет — детали становятся элементами, следующий уровень: назад к коду окна (5); да — обучение закончено")


# ---------------------------------------------------------------- уровни 1…4 на этой четвёрке
def stage_levels(ctx) -> None:
    from damp.activation.set_embedding import DetectorEmbeddings

    run, ladder, picture = ctx["run"], ctx["ladder"], ctx["picture"]
    maps = {int(m.number): m for m in ladder.maps()}
    middle = np.array([ctx["sy"].mean(), ctx["sx"].mean()])
    section("10. Тот же виток на уровнях 1…4 (зелёная линия — возврат)", (70, 205, 140))
    count = len(run.levels)
    loop("для каждого уровня L = 1…" + str(count), 3 * count, "следующий уровень")
    for level in run.levels:
        L = int(level.number)
        keyholes, (details, length, strength, _) = SL.GOT["embed"][L]
        codes, _, firing = next(item for item in SL.GOT["codes"] if item[1] is level.detectors or int(item[2].sets) == int(keyholes.pieces.picture.shape[0]))
        mine = np.flatnonzero(np.asarray(keyholes.pieces.picture) == picture)
        centre = np.asarray(keyholes.pieces.centre, float)
        w = int(mine[np.argmin(np.hypot(centre[mine, 0] - middle[0], centre[mine, 1] - middle[1]))])
        radius = float(np.asarray(keyholes.extent, float)[w])
        lm = maps[L]
        cells = np.asarray(lm.cells, float)
        colours = [DIGIT_RGB[int(d)] for d in lm.labels]
        _, lit_points, _ = level.detectors.lit(np.asarray(codes.packed)[w:w + 1])
        lit_points = np.asarray(lit_points, int)
        rows, cols = np.asarray(firing.rows), np.asarray(firing.columns)
        fired = cols[rows == w]
        table = level.detectors.detectors
        code = np.unpackbits(np.asarray(details)[w], count=int(length))

        def p_win(ax, cy=centre[w, 0], cx=centre[w, 1], radius=radius, mine=mine):
            draw_details(ax, ctx, 1)
            ax.scatter(centre[mine, 1], centre[mine, 0], s=26, color="#4fd1c5", edgecolors="white", linewidths=0.4, zorder=5)
            ax.add_patch(Circle((cx, cy), radius, fill=False, color="#ffd34d", lw=2.5))
            ax.set_xlim(min(-1, cx - radius), max(24, cx + radius)); ax.set_ylim(max(24, cy + radius), min(-1, cy - radius))

        def p_map(ax, cells=cells, colours=colours, lit_points=lit_points, fired=fired, table=table):
            ax.scatter(cells[:, 1], cells[:, 0], s=2.0 if cells.shape[0] > 5000 else 5, c=colours, marker="s", linewidths=0, alpha=0.45)
            ax.scatter(cells[lit_points, 1], cells[lit_points, 0], s=6, color="white", marker="s", linewidths=0)
            for column in fired:
                t = table[int(column)]
                ax.add_patch(Circle((t.center[1], t.center[0]), t.radius, fill=False, color=KEEP, lw=1.2))
            ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")

        cell(f"10.{L}a Окно уровня {L}", panel(f"10-L{L}-window", p_win), f"радиус {radius:.1f} кл.; окон на четвёрке {mine.size}")
        cell(f"10.{L}b Карта уровня {L} зажглась", panel(f"10-L{L}-map", p_map),
             f"{cells.shape[0]} точек, цвет — цифра; белые — зажглись ({np.unique(lit_points).size}), круги — сработали ({fired.size})")
        cell(f"10.{L}c Код детали уровня {L}", panel(f"10-L{L}-code", lambda ax, code=code: board(ax, code)),
             f"{code.size} бит, единиц {int(code.sum())}")
    mark("level_back_to_end")


# ---------------------------------------------------------------- этап 7: распознавание
def stage_recognition(ctx) -> None:
    ladder = ctx["ladder"]
    sight = ladder.see(ctx["image"], log=False)
    votes = list(sight.votes)
    section("11. Распознавание: та же четвёрка как новая картинка (§7.1.13)", (255, 205, 70))
    for v in votes:
        def p_vote(ax, v=v):
            vals = np.asarray(v.votes, float)
            ax.bar(range(10), vals, color=[DIGIT_RGB[j] if j == v.digit else "#5d6470" for j in range(10)])
            ax.set_xticks(range(10))
            chart_axes(ax)
        cond(f"11.{votes.index(v) + 1} Уровень {v.number}: доля голосов ≥ порога знания?", panel(f"11-vote{v.number}", p_vote),
             f"«{v.digit}» {100 * v.share:.0f} %: {'да — знает' if v.knows else 'нет — не знает'}")

    def p_memory(ax):
        vals = np.asarray(sight.memory_votes, float) if len(sight.memory_votes) else np.zeros(10)
        ax.bar(range(10), vals, color=[DIGIT_RGB[j] if j == sight.memory_digit else "#5d6470" for j in range(10)])
        ax.set_xticks(range(10)); chart_axes(ax)

    def p_reader(ax):
        vals = np.asarray(sight.probabilities, float)
        ax.bar(range(10), vals, color=[GOOD if j == sight.digit else "#5d6470" for j in range(10)])
        ax.set_xticks(range(10)); chart_axes(ax)
    cond(f"11.{len(votes) + 1} Уровни согласны?", panel("11-memory", p_memory), f"память: «{sight.memory_digit}» ({sight.verdict}) — целиком / по частям / не знаю")
    cell(f"11.{len(votes) + 2} Ответ читателя", panel("11-reader", p_reader), f"линейный слой над уровнями активации (§9.3): «{sight.digit}» {100 * max(sight.probabilities):.0f} %")


# ---------------------------------------------------------------- сборка квадратного холста
def compose(ctx) -> None:
    def F(size, bold=False):
        return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)

    F_T, F_SUB, F_CELL, F_NOTE, F_SEC, F_LOOP = F(120, True), F(44), F(30, True), F(24), F(30, True), F(26, True)
    cells = [item for item in FLOW if item[0] in ("cell", "if")]
    # раскладка по рядам: петля не рвётся на два ряда
    best = None
    for cols in range(8, 16):
        pos, r, c, i = [], 0, 0, 0
        for item in FLOW:
            if item[0] == "loop" and c + item[2] > cols and c > 0 and item[2] <= cols:
                r, c = r + 1, 0
            if item[0] in ("cell", "if"):
                if c >= cols:
                    r, c = r + 1, 0
                pos.append((r, c))
                c += 1
        rows_n = r + 1
        cw, ch, gx, gy = 660, 820, 120, 210
        W = 2 * 320 + cols * cw + (cols - 1) * gx
        H = 560 + rows_n * (ch + gy) + 150
        score = abs(math.log(W / H))
        if best is None or score < best[0]:
            best = (score, cols, pos, W, H)
    _, cols, pos, W, H = best
    cw, ch, gx, gy = 660, 820, 120, 210
    S = max(W, H)
    ox, oy = 320 + (S - W) // 2, 560 + (S - H) // 2
    sheet = Image.new("RGB", (S, S), (13, 14, 18))
    d = ImageDraw.Draw(sheet)
    probe = d

    def wrap(s, fnt, width):
        out, line = [], ""
        for word in str(s).split(" "):
            cand = (line + " " + word).strip()
            if probe.textlength(cand, font=fnt) <= width:
                line = cand
            else:
                out.append(line)
                line = word
        out.append(line)
        return out

    def xy(r, c):
        return ox + c * (cw + gx), oy + r * (ch + gy)

    def arrow(points, colour=(200, 202, 212), width=6, head=22):
        d.line(points, fill=colour, width=width, joint="curve")
        (x0, y0), (x1, y1) = points[-2], points[-1]
        a = math.atan2(y1 - y0, x1 - x0)
        d.polygon([(x1, y1), (x1 - head * math.cos(a) + head * 0.6 * math.sin(a), y1 - head * math.sin(a) - head * 0.6 * math.cos(a)),
                   (x1 - head * math.cos(a) - head * 0.6 * math.sin(a), y1 - head * math.sin(a) + head * 0.6 * math.cos(a))], fill=colour)

    d.text((320, 90), f"DAMP по атомам: одна четвёрка (MNIST train №{ctx['mnist']}) от пикселей до ответа", font=F_T, fill=(255, 255, 255))
    sub = ("каждая клетка — одна операция кода на настоящих данных (опора подбора v2: перекрытие угла 0.85, код окна до 8192, 4 уровня, "
           f"зерно {SEED}) · читать слева направо, ряд за рядом вниз · жёлтая рамка — условие (да / нет), оранжевая — петля «для каждого», "
           "зелёная линия — возврат на следующий уровень")
    y = 260
    for line in wrap(sub, F_SUB, S - 640):
        d.text((320, y), line, font=F_SUB, fill=(185, 188, 198))
        y += 58
    # петли и разделы
    k = 0
    loops, sections, marks = [], {}, {}
    for item in FLOW:
        if item[0] == "loop":
            loops.append((k, item[2], item[1], item[3]))
        elif item[0] == "section":
            sections[k] = (item[1], item[2])
        elif item[0] == "mark":
            marks[item[1]] = k - 1
        else:
            k += 1
    for start, count, label, back in loops:
        span = [pos[j] for j in range(start, min(start + count, len(pos)))]
        if not span:
            continue
        r0 = span[0][0]
        segs = {}
        for r, c in span:
            segs.setdefault(r, []).append(c)
        for r, cs in segs.items():
            x0, y0 = xy(r, min(cs))
            x1, _ = xy(r, max(cs))
            x1 += cw
            d.rounded_rectangle((x0 - 30, y0 - 62, x1 + 30, y0 + ch + 70), radius=28, outline=(255, 140, 50), width=4)
            if r == r0:
                d.text((x0 - 10, y0 - 54), "петля: " + label, font=F_LOOP, fill=(255, 160, 80))
            yb = y0 + ch + 40
            arrow([(x1 - 40, yb), (x0 + 40, yb), (x0 + 40, y0 + ch + 8)], colour=(255, 140, 50), width=5, head=18)
            if back:
                d.text((x0 + 70, yb - 34), "← " + back, font=F_NOTE, fill=(255, 160, 80))
    for i, (r, c) in enumerate(pos):
        kind, title, path, note = cells[i]
        x, y0 = xy(r, c)
        if i in sections:
            name, colour = sections[i]
            tw = d.textlength(name, font=F_SEC)
            d.rounded_rectangle((x - 4, y0 - 120, x + tw + 30, y0 - 70), radius=12, fill=tuple(int(v * 0.35) for v in colour), outline=colour, width=2)
            d.text((x + 12, y0 - 115), name, font=F_SEC, fill=(255, 255, 255))
        is_if = kind == "if"
        d.rounded_rectangle((x, y0, x + cw, y0 + ch), radius=20, fill=(36, 39, 50), outline=(245, 205, 70) if is_if else (95, 100, 115), width=3 if is_if else 2)
        ty = y0 + 14
        tx = x + 18
        if is_if:
            dx_, dy_ = 26, 20
            d.polygon([(tx + dx_, ty + 2), (tx + 2 * dx_, ty + dy_ + 2), (tx + dx_, ty + 2 * dy_ + 2), (tx, ty + dy_ + 2)], fill=(245, 205, 70))
            tx += 2 * dx_ + 12
        for line in wrap(title, F_CELL, x + cw - 18 - tx)[:3]:
            d.text((tx, ty), line, font=F_CELL, fill=(255, 240, 200) if is_if else (240, 240, 245))
            ty += 38
        notes = wrap(note, F_NOTE, cw - 36)[:4]
        ny = y0 + ch - 14 - 30 * len(notes)
        for line in notes:
            d.text((x + 18, ny), line, font=F_NOTE, fill=(170, 172, 182))
            ny += 30
        if path and os.path.exists(path):
            im = Image.open(path).convert("RGB")
            aw, ah = cw - 30, (y0 + ch - 20 - 30 * len(notes)) - (ty + 10)
            k2 = min(aw / im.width, ah / im.height)
            im = im.resize((max(1, int(im.width * k2)), max(1, int(im.height * k2))), Image.LANCZOS)
            sheet.paste(im, (int(x + (cw - im.width) / 2), int(ty + 6 + (ah - im.height) / 2)))
        if is_if:
            d.text((x + cw - 70, y0 + ch // 2 - 40), "да", font=F_LOOP, fill=(245, 205, 70))
        # стрелка к следующей клетке
        if i + 1 < len(pos):
            r2, c2 = pos[i + 1]
            if r2 == r:
                arrow([(x + cw + 8, y0 + ch // 2), (x + cw + gx - 8, y0 + ch // 2)])
            else:
                x2, y2 = xy(r2, c2)
                mid = y0 + ch + gy // 2 + 40
                arrow([(x + cw // 2, y0 + ch + 8), (x + cw // 2, mid), (x2 + cw // 2, mid), (x2 + cw // 2, y2 - 130)], colour=(150, 152, 165), width=5)
    # возврат на следующий уровень: от условия «окно видит всю цифру?» к коду окна (раздел 5)
    if "level_back_from" in marks:
        src = marks["level_back_from"] + 1
        dst = next(i for i in sections if sections[i][0].startswith("5."))
        (r1, c1), (r2, c2) = pos[src], pos[dst]
        x1, y1 = xy(r1, c1)
        x2, y2 = xy(r2, c2)
        lane = ox - 200
        over = y2 - 155  # промежуток над рядом цели: ниже возвратов петель ряда выше, выше подписей разделов
        enter = x2 + int(cw * 0.75)
        arrow([(x1, y1 + ch // 2 + 60), (x1 - gx // 2, y1 + ch // 2 + 60), (x1 - gx // 2, y1 - gy + 30), (lane, y1 - gy + 30), (lane, over),
               (enter, over), (enter, y2 - 8)], colour=(70, 205, 140), width=8, head=26)
        d.text((lane + 14, (y1 + y2) // 2), "нет: следующий уровень", font=F_LOOP, fill=(70, 205, 140))
    out = os.path.join(FOLDER, f"atomic-four-seed{SEED}.png")
    sheet.save(out, optimize=True)
    small = sheet.resize((S // 4, S // 4), Image.LANCZOS)
    small.save(os.path.join(FOLDER, f"atomic-four-seed{SEED}-preview.png"))
    print(out, sheet.size, "клеток", len(pos), "колонок", cols)


if __name__ == "__main__":
    main()
