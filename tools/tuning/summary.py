# -*- coding: utf-8 -*-
"""Сводная таблица серии подбора на N обучающих цифрах (по журналу стенда tune2_log.jsonl) и лучший вариант.

    summary.py [обучающих цифр=250] [с прогона=496]

Лучший — по честной сходимости верха; при равенстве до 0.005 — по доле точек в главном пятне своей цифры.
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 250
FIRST = int(sys.argv[2]) if len(sys.argv) > 2 else 496


def short(value) -> str:  # noqa: ANN001
    if isinstance(value, list):
        return "/".join(str(v) for v in value)
    if isinstance(value, bool):
        return "да" if value else "нет"
    return str(value)


def place_heavy(entry: dict) -> bool:
    """Код, где место весомее угла: на уровне 0 не меньше 30 бит места на круг (около 52 % единиц и больше)."""
    bits = entry["changes"].get("LADDER_BINDING_PLACE_BITS", 1)
    first = bits[0] if isinstance(bits, list) else bits
    return int(first) >= 30


rows = []
for line in io.open(os.path.join(HERE, "tune2_log.jsonl"), encoding="utf-8"):
    entry = json.loads(line)
    if entry["run"] >= FIRST and entry.get("count") == COUNT:
        rows.append(entry)
rows.sort(key=lambda e: e["run"])
if not rows:
    print("прогонов нет")
    sys.exit()
# Лучший — среди кодов, где место весомее угла (так задано автором); прочие — ориентиры.
candidates = [e for e in rows if place_heavy(e)] or rows
# Карта уровня 0 должна быть цельной (автор 2026-11-20: «гладкость первичной карты очень важна!»): с трещиной или
# островком (больше 10 пустых клеток внутри карты) вариант лучшим не считается.
candidates = [e for e in candidates if (e.get("level0") or {}).get("inner_empty", 0) <= 10] or candidates
best = max(candidates, key=lambda e: (round(e["top_honest"], 2), e.get("top_in_main", 0)))
print(f"| № | Что изменено | Окно видит, ур. 1 / 2 / 3 | Сходимость по уровням | **Сходимость верха** | Пятен | В главном пятне | Одиночек | Точек верха | Ур. 0: пинвилы, ° / дыр / ретинотопия |")
print("|---|---|---|---|---|---|---|---|---|---|")
for e in rows:
    funnel = [row for row in e.get("funnel", []) if int(row.get("level", 1)) >= 1]
    sees = " / ".join(f"{100 * row['window_sees']:.0f}" for row in funnel) + " %"
    honest = " / ".join(f"{v:.2f}" for v in e["honest"])
    mark = "**" if e is best else ""
    label = e["label"].replace("250 цифр: ", "") + ("" if place_heavy(e) else " — ориентир, место не весомее")
    print(f"| {mark}{e['run']}{mark} | {label} | {sees} | {honest} | {mark}{e['top_honest']:.3f}{mark} | {e.get('top_patches', '—')} | "
          f"{e.get('top_in_main', 0):.2f} | " + (f"{100 * e['top_lonely']:.1f} %" if "top_lonely" in e else "—") + f" | {e['points'][-1]} | " + (f"{e['level0']['smooth_deg']} / {e['level0']['inner_empty']} / {e['level0'].get('retino_spearman', '—')}" if (e.get('level0') or {}).get('smooth_deg') is not None else "—") + " |")
print(f"| **Итого** | {len(rows)} прогонов на {COUNT} цифрах | | | лучший — **№{best['run']}**: {best['top_honest']:.3f} | "
      f"{best.get('top_patches', '—')} | {best.get('top_in_main', 0):.2f} | | | |")
print()
print(f"ЛУЧШИЙ №{best['run']}: {best['label']}")
print("  изменения: " + ", ".join(f"{k} = {short(v)}" for k, v in best["changes"].items()))
