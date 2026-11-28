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
V2_FIRST = 574  # план подбора v2 (2026-11-20): с этого прогона — мерки М1–М7 и пары зёрен
BASE = {"MNIST_ANGLE_OVERLAP": 0.85, "LADDER_BINDING_LENGTH_MAX": 8192, "LADDER_TOP_COVERAGE": 0.99}  # с 2026-11-21 (646–649): 4 уровня


def v2_summary(rows: list) -> None:
    """Сводка плана v2: прогоны парами зёрен (ENCODER_RANDOM_SEED 0 и 2) — одна точка оси. Главные мерки — М1 (в главном
    пятне своей цифры, верх) и М2 (радиальность классов, верх), у уровня 0 — М7 (гладкость, дыры, полоса, ретинотопия) и
    М3 (нити); честная сходимость — страж (не ниже 0.61). Правило приёма: оба зерна не хуже опоры, среднее лучше > 0.02,
    на уровне 0 нет трещин (пустых внутри ≤ 10) ни на одном зерне."""
    def zero(e: dict, key: str, default="—"):  # noqa: ANN202
        return (e.get("level0") or {}).get(key, default)

    print("| № | Что изменено | Зерно | **М1 верха** / пятен | **М2 верха** | Честная (страж) | Ур. 0: пинвилы, ° / дыр / полоса / ретинотопия | Ур. 0: нитей / мусора | Ур. 0, код угла: штрихов на код / наиб. группа | Ē ур. 0 | Читатель / память, % |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for e in rows:
        seed = e["changes"].get("ENCODER_RANDOM_SEED", 0)
        print(f"| {e['run']} | {e['label'].replace('250 цифр: ', '')} | {seed} | {e.get('top_m1_main', 0):.3f} / {e.get('top_m1_patches', '—')} | "
              f"{e.get('top_m2', 0):.3f} | {e['top_honest']:.3f} | {zero(e, 'smooth_deg')} / {zero(e, 'inner_empty')} / {zero(e, 'longest_gap')} / "
              f"{zero(e, 'retino_spearman')} | {zero(e, 'm3_threads', 0):.3f} / {zero(e, 'm3_rubbish', 0):.3f} | {zero(e, 'm6_angle_per_code')} / {zero(e, 'm6_angle_largest')} | "
              f"{zero(e, 'energy', 0):.3f} | " + (f"{100 * e['accuracy200']:.0f} / {100 * (e['memory200'].get('right_share') or 0):.0f}" if e.get('test_count') else "—") + " |")
    groups = {}
    for e in rows:
        key = json.dumps({k: v for k, v in e["changes"].items() if k != "ENCODER_RANDOM_SEED"}, sort_keys=True, ensure_ascii=False)
        groups.setdefault(key, []).append(e)
    print()
    print("| Точка (правки без зерна) | Прогоны | М1 среднее (мин) | М2 среднее (мин) | Честная среднее (мин) | Пинвилы, ° | Дыр ур. 0, макс | Ретинотопия, мин | Читатель / память, % |")
    print("|---|---|---|---|---|---|---|---|---|")
    scored = []
    for key, items in groups.items():
        m1 = [e.get("top_m1_main", 0) for e in items]
        m2 = [e.get("top_m2", 0) for e in items]
        honest = [e["top_honest"] for e in items]
        smooth = [zero(e, "smooth_deg", 0) for e in items]
        holes = [zero(e, "inner_empty", 0) for e in items]
        retino = [zero(e, "retino_spearman", 0) for e in items]
        scored.append((sum(m1) / len(m1), key, items))
        print(f"| {key} | {', '.join(str(e['run']) for e in items)} | {sum(m1) / len(m1):.3f} ({min(m1):.3f}) | {sum(m2) / len(m2):.3f} ({min(m2):.3f}) | "
              f"{sum(honest) / len(honest):.3f} ({min(honest):.3f}) | {sum(smooth) / len(smooth):.2f} | {max(holes)} | {min(retino)} | " + (
                  f"{100 * sum(e['accuracy200'] for e in tested) / len(tested):.0f} / {100 * sum(e['memory200'].get('right_share') or 0 for e in tested) / len(tested):.0f}"
                  if (tested := [e for e in items if e.get('test_count')]) else "—") + " |")
    # Парное сравнение (анализ 2026-11-21): кандидат и опора на одних и тех же зёрнах — общий сдвиг зерна сокращается,
    # шум разности на зерно 0.026–0.033 против ~0.06 без пары. Опора — принятая точка BASE (её значения по каждому зерну).
    base_by_seed = {}
    for e in rows:
        if {k: v for k, v in e["changes"].items() if k != "ENCODER_RANDOM_SEED"} == BASE:
            base_by_seed.setdefault(e["changes"].get("ENCODER_RANDOM_SEED", 0), e)
    print()
    print(f"Парно к опоре {json.dumps(BASE, ensure_ascii=False)} (зёрна опоры: {sorted(base_by_seed)}):")
    print("| Точка | Зёрна | Δ М1 по зёрнам | **Δ М1 среднее** | Δ М1² среднее | Δ честной среднее | Δ гладкости ур. 0, ° |")
    print("|---|---|---|---|---|---|---|")
    for key, items in groups.items():
        paired = [(e, base_by_seed[e["changes"].get("ENCODER_RANDOM_SEED", 0)]) for e in items if e["changes"].get("ENCODER_RANDOM_SEED", 0) in base_by_seed]
        if not paired or json.loads(key) == BASE:
            continue
        d1 = [e.get("top_m1_main", 0) - b.get("top_m1_main", 0) for e, b in paired]
        d2 = [e["top_m1_sq"] - b["top_m1_sq"] for e, b in paired if "top_m1_sq" in e and "top_m1_sq" in b]
        dh = [e["top_honest"] - b["top_honest"] for e, b in paired]
        ds = [zero(e, "smooth_deg", 0) - zero(b, "smooth_deg", 0) for e, b in paired]
        seeds = [e["changes"].get("ENCODER_RANDOM_SEED", 0) for e, _ in paired]
        print(f"| {key} | {', '.join(map(str, seeds))} | {' / '.join(f'{v:+.3f}' for v in d1)} | **{sum(d1) / len(d1):+.3f}** | "
              + (f"{sum(d2) / len(d2):+.3f}" if d2 else "—") + f" | {sum(dh) / len(dh):+.3f} | {sum(ds) / len(ds):+.2f} |")
    complete = [item for item in scored if len(item[2]) >= 2]
    fine = [item for item in complete if min(e["top_honest"] for e in item[2]) >= 0.61 and max(zero(e, "inner_empty", 0) for e in item[2]) <= 10]
    if fine or complete:
        top = max(fine or complete, key=lambda item: item[0])
        print(f"| **Итого** | {len(rows)} прогонов, {len(groups)} точек | лучшая по М1 (пара, страж и цельный уровень 0): {top[1]} — {top[0]:.3f} | | | | | |")


if FIRST >= V2_FIRST:
    v2_summary(rows)
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
