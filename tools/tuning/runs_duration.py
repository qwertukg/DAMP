# -*- coding: utf-8 -*-
"""График длительности всех прогонов из docs/runs.jsonl (с 84-го — раньше журнала нет).

    runs_duration.py <репозиторий> <выходной png>

Время — весь прогон (метка времени стенда); где проверочных нет или метки нет — время обучения цикла.
Цвет и форма точки — число обучающих цифр.
"""
import io
import json
import os
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

REPO = sys.argv[1]
OUT = sys.argv[2]

rows = [json.loads(line) for line in io.open(os.path.join(REPO, "docs", "runs.jsonl"), encoding="utf-8")]
rows.sort(key=lambda r: r["run"])

# Цвета — по порядку первого появления серии (фиксированный порядок палитры), форма — вторая подсказка.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e0"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["D", "s", "o", "^", "v"]

order = []
for r in rows:
    if r["train"] not in order:
        order.append(r["train"])
style = {count: (SLOTS[i], MARKERS[i]) for i, count in enumerate(order)}


def duration(r: dict) -> tuple[float, bool]:
    if r.get("minutes"):
        return float(r["minutes"]), True
    return float(r.get("train_minutes") or 0.0), not r.get("tests")


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(13, 6.2), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

for count in order:
    group = [r for r in rows if r["train"] == count]
    color, marker = style[count]
    full = [(r["run"], duration(r)[0]) for r in group if duration(r)[1]]
    part = [(r["run"], duration(r)[0]) for r in group if not duration(r)[1]]
    values = [duration(r)[0] for r in group]
    tests = max((r.get("tests") or 0) for r in group)
    with_tests = sum(1 for r in group if r.get("tests"))
    n = len(group)
    word = "прогон" if n % 10 == 1 and n % 100 != 11 else ("прогона" if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14) else "прогонов")
    if not with_tests:
        extra = " без проверочных"
    elif with_tests == n:
        extra = f" + {tests} проверочных"
    else:
        extra = f" (у {with_tests} из них + {tests} проверочных)"
    label = f"{count} обучающих{extra} — {n} {word}"
    if full:
        ax.scatter(*zip(*full), s=34, c=color, marker=marker, edgecolors=SURFACE, linewidths=0.8, zorder=3, label=label)
    if part:
        ax.scatter(*zip(*part), s=34, facecolors="none", edgecolors=color, marker=marker, linewidths=1.2, zorder=3)

# Подписи медиан прямо у серий.
notes = {
    1000: (92, 150, "1000 обучающих:\n61 и 118 мин"),
}
for count in order:
    group = [r for r in rows if r["train"] == count]
    values = [duration(r)[0] for r in group]
    median = statistics.median(values)
    first, last = group[0]["run"], group[-1]["run"]
    if count == 1000:
        text, x, y, ha = f"{count} обучающих: {values[0]:.0f} и {values[1]:.0f} мин", last + 6, max(values), "left"
    elif count == 500:
        text, x, y, ha = f"500 обучающих: медиана {median:.1f} мин", 112, 14.0, "left"
    elif count == 200:
        text, x, y, ha = f"200 обучающих: медиана {median:.1f} мин", 300, 2.05, "left"
    elif count == 100:
        text, x, y, ha = f"100: медиана {median:.1f} мин", first - 4, 0.5, "left"
    else:
        text, x, y, ha = f"250: медиана {median:.1f} мин", first - 6, 8.5, "left"
    ax.text(x, y, text, color=INK, fontsize=10, ha=ha, va="center", zorder=4,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE, edgecolor="none", alpha=0.85))

ax.set_yscale("log")
ticks = [0.5, 1, 2, 5, 10, 20, 50, 100, 200]
ax.yaxis.set_major_locator(FixedLocator(ticks))
ax.yaxis.set_minor_locator(NullLocator())
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_ylim(0.35, 220)
ax.set_xlim(rows[0]["run"] - 6, rows[-1]["run"] + 6)
ax.set_xlabel("номер прогона", color=INK2)
ax.set_ylabel("минут на прогон (лог. шкала)", color=INK2)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.grid(axis="x", visible=False)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.tick_params(colors=INK2, length=0)

total = sum(duration(r)[0] for r in rows) / 60.0
days = f"{rows[0]['date'][:10]} — {rows[-1]['date'][:10]}"
fig.suptitle(f"Длительность прогонов: {len(rows)} прогонов, №{rows[0]['run']}–{rows[-1]['run']}, всего {total:.0f} ч счёта",
             x=0.06, y=0.985, ha="left", fontsize=14, color=INK, fontweight="bold")
ax.set_title(f"{days}. Точка — прогон. Полная — весь прогон; пустая — только обучение (метки времени нет).\n"
             f"Для серии без проверочных время = обучение. Зависания в кеше (505, 506) не учтены — в журнал пишется время счёта.",
             loc="left", fontsize=9, color=INK2, pad=10)
legend = ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK, markerscale=1.1)
fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, facecolor=SURFACE)
print(OUT)
for count in order:
    values = [duration(r)[0] for r in rows if r["train"] == count]
    print(count, len(values), f"медиана {statistics.median(values):.1f}", f"мин {min(values):.1f}", f"макс {max(values):.1f}", f"сумма {sum(values) / 60:.1f} ч")
