# -*- coding: utf-8 -*-
"""Где у окон «плато» уровней срабатывания: лидеры → плато → хвост, по всем окнам каждого уровня (из кеша прогона).

    plateau.py <changes.json> <обучающих цифр> <выход png> [окон на уровень = 800]

Для каждого окна уровни E(d) сработавших детекторов сортируются по убыванию и приближаются тремя прямыми (два излома
b1 и b2, наименьшая сумма квадратов): [0, b1) — лидеры, [b1, b2) — плато, [b2, n) — хвост. Середина плато (b1 + b2) / 2 —
место σ по разметке автора (2026-11-20). По уровню — медианы: сработало n, b1, b2, середина, доля середины от n.
"""
import json
import os
import sys

sys.path[:0] = [os.path.join(os.getcwd(), "client"), os.path.join(os.getcwd(), "system")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from damp_api import Damp  # noqa: E402

FIRED: list = []


def capture() -> None:
    from damp.cycle.step import CycleStep

    fire = CycleStep.fire

    def fire_seen(detectors, codes, trained):  # noqa: ANN001, ANN202
        firing = fire(detectors, codes, trained)
        FIRED.append((detectors, firing))
        return firing

    CycleStep.fire = staticmethod(fire_seen)


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
    count, out = int(sys.argv[2]), sys.argv[3]
    per_level = int(sys.argv[4]) if len(sys.argv) > 4 else 800
    capture()
    damp = Damp(changes={**changes, "DASHBOARD_FILE_PREFIX": "data/tune/ladder"})
    ladder = damp.cycle.trained(count)
    run = ladder._ladder
    sigma = int(damp.settings["ACTIVATION_SATURATION_LIMIT"])
    named = [(0, run.primary.detectors)] + [(int(level.number), level.detectors) for level in run.levels]
    rng = np.random.default_rng(0)
    rows_out, curves = [], {}
    for number, detectors in named:
        candidates = [f for d, f in FIRED if d is detectors]
        firing = max(candidates, key=lambda f: int(f.sets))
        rows, levels = np.asarray(firing.rows), np.asarray(firing.levels, dtype=np.float64)
        order = np.argsort(rows, kind="stable")
        rows, levels = rows[order], levels[order]
        starts = np.searchsorted(rows, np.arange(int(firing.sets) + 1))
        counts = np.diff(starts)
        windows = np.flatnonzero(counts >= 6)
        pick = rng.choice(windows, size=min(per_level, windows.size), replace=False)
        stats, grid = [], []
        for w in pick:
            lv = levels[starts[w]:starts[w + 1]]
            b1, b2 = breaks(lv)
            n = lv.size
            mid = 0.5 * (b1 + b2)
            y = np.sort(lv)[::-1]
            stats.append((n, b1, b2, mid, mid / n, float(y[min(n - 1, int(round(mid)))]), float(y[0]),
                          float(y[min(n - 1, sigma - 1)]) if n >= sigma else float("nan")))
            grid.append(np.interp(np.linspace(0, 1, 101), np.linspace(0, 1, n), y))
        s = np.asarray(stats)
        curves[number] = np.median(np.asarray(grid), axis=0)
        row = {"level": number, "windows": int(pick.size), "fired": float(np.median(s[:, 0])), "b1": float(np.median(s[:, 1])),
               "b2": float(np.median(s[:, 2])), "mid": float(np.median(s[:, 3])), "share": float(np.median(s[:, 4])),
               "share_p25": float(np.quantile(s[:, 4], 0.25)), "share_p75": float(np.quantile(s[:, 4], 0.75)),
               "level_at_mid": float(np.median(s[:, 5])), "top": float(np.median(s[:, 6])),
               "sigma_inside_plateau": float(np.mean((s[:, 1] < sigma) & (sigma < s[:, 2])))}
        rows_out.append(row)
        print(json.dumps(row, ensure_ascii=False))
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=130)
    for (number, curve), colour in zip(sorted(curves.items()), ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#6250d6"]):
        r = next(item for item in rows_out if item["level"] == number)
        ax.plot(np.linspace(0, 1, 101), curve, color=colour, lw=2, label=f"уровень {number}: сработало ≈{r['fired']:.0f}, середина плато ≈{r['mid']:.0f} ({100 * r['share']:.0f} %)")
        ax.axvline(r["share"], color=colour, lw=1, ls="--")
    ax.set_xlabel("место детектора в окне по убыванию уровня, доля от сработавших")
    ax.set_ylabel("уровень E(d), нормированный (медиана по окнам)")
    ax.legend(frameon=False, fontsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"Кривая уровней срабатывания по всем окнам; пунктир — середина плато (σ сейчас {sigma})", loc="left", fontsize=10.5)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
