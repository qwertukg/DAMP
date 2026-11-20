# -*- coding: utf-8 -*-
"""Таблица результатов прогонов (слово автора 2026-11-18: «храни таблицу результатов; в неё будем добавлять другие
прогоны и расширять новыми параметрами»).

Хранится в репозитории: docs/runs.jsonl — сжатые записи прогонов (источник), docs/RUNS.md — таблица для чтения. Записи
берутся из журнала стенда подбора (tune2_log.jsonl во временной папке) и сливаются с docs/runs.jsonl по номеру прогона,
поэтому история не теряется, даже если временная папка пропадёт.

Столбцы — формат автора (уровень за уровнем через «/»): параметры, радиус окна, какую долю цифры видит окно, деталей в
окне, точек карты, сходимость по уровням и верха, читатель, память, ответ на «X», время. Настройки, которые меняются не во
всех прогонах, получают свой столбец сами (см. AXES); прочие изменения — в столбце «Другие изменения».

    runs_table.py            — пересобрать таблицу
    runs_table.py --from N   — взять из журнала стенда прогоны с номера N (по умолчанию 84: текущее устройство цикла)
"""
import io
import json
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "c:/Users/danii/AppData/Local/Temp/claude/90c048c3c2430e10/50fe34b5-999a-4743-bfe4-e4327ebbf623/scratchpad/damp-second-level"
STORE = os.path.join(REPO, "docs", "runs.jsonl")
TABLE = os.path.join(REPO, "docs", "RUNS.md")
FIRST = int(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 84

# Настройки-оси: своим столбцом, в этом порядке. Новая ось — строка сюда (и её значение по умолчанию из конфига).
AXES = [
    ("LADDER_CLUSTER_MIN_POINTS", "Соседей"),
    ("LADDER_KEYHOLE_RADIUS_SHARE", "Предел"),
    ("LADDER_LEVEL0", "Уровень 0"),
    ("LADDER_LEVEL0_PLACE", "Уровень 0: в окне"),
    ("LADDER_KEYHOLE_WHOLE_PICTURE", "Окно на всю картинку"),
    ("ACTIVATION_SATURATION_LIMIT", "σ — детекторов в эмбеддинге"),
    ("LADDER_TARGET_DENSITY", "Плотность эмбеддинга"),
    ("LADDER_BINDING_PLACE_BITS", "Бит места на круг"),
    ("LADDER_BINDING_POOLING", "Пулинг круга места"),
    ("LADDER_BINDING_SATURATION", "Насыщенность содержания"),
    ("LADDER_LAYOUT_LAMBDA_QUANTILE", "Квантиль λ карты, низ"),
    ("LADDER_LAYOUT_LAMBDA_MAX_QUANTILE", "Квантиль λ карты, верх"),
    ("LADDER_HIERARCHY_THRESHOLDS", "Пороги слоёв детекторов"),
    ("LAYOUT_SHORT_MIN_SWAP_RATIO", "Ближняя фаза: порог обменов"),
    ("LAYOUT_TUNE_SHORT_RADIUS_FACTOR", "Ближняя фаза: радиус пары, доля сетки"),
    ("LAYOUT_TUNE_SHORT_ENERGY_RADIUS_FACTOR", "Ближняя фаза: радиус энергии, доля сетки"),
    ("LAYOUT_TUNE_SHORT_STEPS_MIN", "Ближняя фаза: шагов не меньше"),
    ("LAYOUT_TUNE_SHORT_STEPS_MAX", "Ближняя фаза: шагов не больше"),
    ("LAYOUT_SHORT_LOCAL_RADIUS", "Ближняя фаза: радиус окрестности"),
    ("LAYOUT_SHORT_ENERGY_RADIUS", "Ближняя фаза: радиус энергии"),
    ("LAYOUT_MIN_SWAP_RATIO", "Дальняя фаза: порог обменов"),
    ("LAYOUT_ADAPTIVE_SWAP_TRIGGER", "Ступень λ и r: доля обменов"),
    ("LAYOUT_ADAPTIVE_LAMBDA_STEP", "Ступень λ: шаг"),
    ("ACTIVATION_LOCAL_ENERGY_THRESHOLD", "Порог энергии активации"),
    ("ENCODER_RANDOM_SEED", "Зерно"),
    ("LAYOUT_LONG_WEIGHTED_FIRST", "Дальняя фаза: первая точка по −ln Ê"),
    ("ACTIVATION_DETECTOR_FILL_CELL", "Радиус детектора: поправка на клетку"),
    ("ACTIVATION_DETECTOR_RADIUS_MODE", "Радиус детектора: способ"),
    ("ACTIVATION_SATURATION_SHARE", "σ долей от сработавших"),
    ("LAYOUT_LONG_KERNEL_WIDTH", "Ширина гауссова ядра"),
    ("LADDER_BINDING_PLACE_NARROWEST", "Код места: самых узких кругов по стороне"),
    ("LADDER_INHIBITION_SIMILARITY", "Пятна: порог"),
    ("LADDER_TOP_COVERAGE", "Остановка: охват"),
    ("LADDER_CLUSTER_LEVELS", "Уровней не больше"),
    ("LADDER_TRAIN_DIGITS", "Цифры обучения"),
]
# Значения по умолчанию у прогонов, которые их не меняли: что тогда стояло в конфиге (уровня 0 до прогона 109 не было,
# окно на всю картинку стояло с 2026-11-17). Новые прогоны задают все оси явно.
DEFAULTS = {"LADDER_CLUSTER_MIN_POINTS": 14, "LADDER_KEYHOLE_RADIUS_SHARE": 1.15, "LADDER_LEVEL0": False,
            "LADDER_KEYHOLE_WHOLE_PICTURE": True, "ACTIVATION_SATURATION_LIMIT": 50, "LADDER_TARGET_DENSITY": 0.25,
            "LADDER_LEVEL0_PLACE": "digit",  # до 2026-11-19 уровень 0 был только вариантом 3 (центр цифры)
            # Оси, добавленные 2026-11-20: значения конфига у прогонов, которые их не меняли.
            "LADDER_BINDING_PLACE_BITS": 1, "LADDER_BINDING_POOLING": "strongest", "LADDER_BINDING_SATURATION": 0.25,
            "LADDER_LAYOUT_LAMBDA_QUANTILE": 0.82, "LADDER_LAYOUT_LAMBDA_MAX_QUANTILE": 0.982, "LADDER_HIERARCHY_THRESHOLDS": "none",
            "LAYOUT_SHORT_MIN_SWAP_RATIO": 0.005, "LAYOUT_TUNE_SHORT_RADIUS_FACTOR": 0.08, "LAYOUT_TUNE_SHORT_ENERGY_RADIUS_FACTOR": 0.06,
            "LAYOUT_TUNE_SHORT_STEPS_MIN": 160, "LAYOUT_TUNE_SHORT_STEPS_MAX": 900, "LADDER_INHIBITION_SIMILARITY": 0.8,
            "LADDER_TOP_COVERAGE": 0.9, "LADDER_CLUSTER_LEVELS": 12, "LADDER_TRAIN_DIGITS": "all",
            "LAYOUT_SHORT_LOCAL_RADIUS": 7, "LAYOUT_SHORT_ENERGY_RADIUS": 7, "LAYOUT_MIN_SWAP_RATIO": 0.0001,
            "LAYOUT_ADAPTIVE_SWAP_TRIGGER": 0.0, "LAYOUT_ADAPTIVE_LAMBDA_STEP": "none", "ACTIVATION_LOCAL_ENERGY_THRESHOLD": 0.5,
            "ENCODER_RANDOM_SEED": 0, "LAYOUT_LONG_WEIGHTED_FIRST": False,
            "ACTIVATION_DETECTOR_FILL_CELL": 0.0, "ACTIVATION_DETECTOR_RADIUS_MODE": "fill",
            "ACTIVATION_SATURATION_SHARE": 0.0, "LAYOUT_LONG_KERNEL_WIDTH": 1.0, "LADDER_BINDING_PLACE_NARROWEST": 16}
# С прогона 448 (0.13.1, 2026-11-20) стенд меняет только окна: остальные оси — из конфига на тот день, явно в записи.
CURRENT = {"LADDER_LEVEL0": True, "LADDER_LEVEL0_PLACE": "digit", "LADDER_KEYHOLE_WHOLE_PICTURE": False,
           "ACTIVATION_SATURATION_LIMIT": 50, "LADDER_TARGET_DENSITY": 0.217,
           "LADDER_BINDING_PLACE_CODE_LENGTH": [176, 1792, 1536, 1088]}


def shown(value) -> str:  # noqa: ANN001
    """Значение оси в таблице: да / нет для переключателей."""
    if isinstance(value, list):
        return " / ".join(str(v) for v in value)
    names = {"stroke": "точки штриха", "digit": "штрих, центр цифры"}
    if isinstance(value, str) and value in names:
        return names[value]
    return "да" if value is True else "нет" if value is False else str(value)


def compact(entry: dict, date: str) -> dict:
    """Сжатая запись прогона: только то, что идёт в таблицу."""
    # Строка уровня 0 (окно на штрихе) в таблицу не идёт: столбцы — уровни 1…верх, как у прежних прогонов.
    funnel = [row for row in entry.get("funnel", []) if "failed" not in row and int(row.get("level", 1)) >= 1]
    memory = entry.get("memory200", {})
    x = entry.get("x") or {}
    return {
        "run": entry["run"],
        "date": entry.get("date") or date,
        "label": entry.get("label", ""),
        "train": entry.get("count"),
        "tests": entry.get("test_count", 200),
        "changes": ({**CURRENT, **entry.get("changes", {})} if entry["run"] >= 448 else entry.get("changes", {})),
        "levels": entry.get("levels"),
        "radius": [row.get("window_radius") for row in funnel],
        "sees": [row.get("window_sees") for row in funnel],
        "inside": [row.get("elements_per_window") for row in funnel],
        "points": [row.get("points") for row in funnel],
        "honest": entry.get("honest", []),
        "patches": entry.get("top_patches"),
        "in_main": entry.get("top_in_main"),
        "lonely": entry.get("top_lonely"),
        "reader": entry.get("accuracy200"),
        "memory": memory.get("right_share"),
        "x": {"memory": x.get("memory_digit"), "reader": x.get("reader_digit")} if x else None,
        "train_minutes": round(entry.get("train_seconds", 0.0) / 60.0, 1),
        "minutes": entry.get("minutes"),
    }


def load() -> dict:
    runs = {}
    if os.path.exists(STORE):
        for line in io.open(STORE, encoding="utf-8"):
            if line.strip():
                item = json.loads(line)
                runs[item["run"]] = item
    log = os.path.join(HERE, "tune2_log.jsonl")
    if os.path.exists(log):
        for line in io.open(log, encoding="utf-8"):
            entry = json.loads(line)
            if entry["run"] < FIRST:
                continue
            stamp = os.path.join(HERE, f"series-total-{entry['run']}.txt")
            date = datetime.fromtimestamp(os.path.getmtime(stamp)).strftime("%Y-%m-%d %H:%M") if os.path.exists(stamp) else ""
            item = compact(entry, date)
            if os.path.exists(stamp) and item.get("minutes") is None:
                try:
                    item["minutes"] = float(open(stamp, encoding="utf-8").read().strip())
                except ValueError:
                    pass
            runs[item["run"]] = {**runs.get(item["run"], {}), **{k: v for k, v in item.items() if v not in (None, "")}}
    return runs


def joined(values, form) -> str:  # noqa: ANN001
    return " / ".join(form(value) for value in values if value is not None) or "—"


def table(rows: list) -> list:
    best_top = max(row["honest"][-1] for row in rows if row["honest"])
    best_reader = max(row["reader"] or 0 for row in rows)
    best_memory = max(row["memory"] or 0 for row in rows)
    for row in rows:
        # Без проверочных картинок (стенд с tests = 0) читатель — 0: «по обоим» тогда — одна сходимость.
        row["_both"] = (row["honest"][-1] / best_top if row["honest"] and best_top else 0) + ((row["reader"] or 0) / best_reader if best_reader else 0)

    def bold(text: str, flag: bool) -> str:
        return f"**{text}**" if flag else text

    head = ["№", "Дата"] + [name for _, name in AXES] + [
        "Другие изменения", "Уровней", "Радиус окна, клеток", "Окно видит, %", "Деталей в окне", "Точек карты, тыс.",
        "Сходимость по уровням", "Сходимость верха", "Пятен верха / в главном пятне", "Одиночек верха", "Читатель", "Память", "«X»: память / читатель", "Прогон, мин"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in rows:
        changes = row.get("changes", {})
        level0 = changes.get("LADDER_LEVEL0", DEFAULTS["LADDER_LEVEL0"])
        axes = ["—" if key == "LADDER_LEVEL0_PLACE" and not level0 else shown(changes.get(key, DEFAULTS.get(key, "—")))
                for key, _ in AXES]
        others = ", ".join(f"`{key} = {value}`" for key, value in changes.items() if key not in dict(AXES)) or "—"
        top = row["honest"][-1] if row["honest"] else 0.0
        x = row.get("x") or {}
        cells = [str(row["run"]), row.get("date", "")[:16]] + axes + [
            others, str(row.get("levels", "—")),
            joined(row["radius"], lambda v: f"{v:.1f}"),
            joined(row["sees"], lambda v: f"{100 * v:.0f}"),
            joined(row["inside"], lambda v: f"{v:.1f}"),
            joined(row["points"], lambda v: f"{v / 1000:.1f}"),
            joined(row["honest"], lambda v: f"{v:.2f}"),
            bold(f"{top:.3f}", top == best_top),
            (f"{row['patches']} / {row['in_main']:.2f}" if row.get("patches") is not None else "—"),
            (f"{100 * row['lonely']:.1f} %" if row.get("lonely") is not None else "—"),
            ("—" if not row.get("tests") else bold(f"{100 * (row['reader'] or 0):.1f} %", row["reader"] == best_reader)),
            ("—" if not row.get("tests") else bold(f"{100 * (row['memory'] or 0):.1f} %", row["memory"] == best_memory)),
            f"«{x.get('memory', '—')}» / «{x.get('reader', '—')}»" if x else "—",
            str(row.get("minutes") or row.get("train_minutes") or "—"),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    def name(row: dict) -> str:
        changes = row.get("changes", {})
        return f"№{row['run']} (" + ", ".join(f"{label.lower()} {shown(changes.get(key, DEFAULTS.get(key)))}" for key, label in AXES) + ")"

    leaders = (f"лучшая сходимость — {name(max(rows, key=lambda r: r['honest'][-1] if r['honest'] else 0))}; "
               f"лучший читатель — {name(max(rows, key=lambda r: r['reader'] or 0))}; "
               f"лучшая память — {name(max(rows, key=lambda r: r['memory'] or 0))}; "
               f"по обоим — {name(max(rows, key=lambda r: r['_both']))}")
    lines.append("| **Итого** | | " + " | ".join("" for _ in AXES) + f" | | | | | | | | {leaders} | | | | | | | |")
    return lines


def main() -> None:
    runs = load()
    with io.open(STORE, "w", encoding="utf-8", newline="\n") as stream:
        for number in sorted(runs):
            stream.write(json.dumps(runs[number], ensure_ascii=False) + "\n")
    out = [
        "# Таблица прогонов",
        "",
        "Результаты прогонов цикла на стенде подбора, накопительно: каждый новый прогон дописывается строкой, новый",
        "подбираемый параметр — столбцом. Источник — [runs.jsonl](runs.jsonl) (по записи на прогон); таблица собирается из него.",
        "",
        "**Как мерено.** Обучение — первые N цифр MNIST train (без видов), предел карты снят; проверка — картинки MNIST t10k,",
        "не участвовавшие в обучении. Сходимость — доля соседних точек карты одной цифры среди пар точек с **разных**",
        "обучающих картинок (случайная карта ≈ 0.10). Читатель — линейный слой над уровнями активации (§9.3); память — ответ",
        "по голосам кусочков (§2.2.5, §7.1.13.3). «X» — ответ на нарисованный крест (пример автора). Шум: сходимость около",
        "±0.02 на 500 цифрах, точность на 1000 проверочных — около ±1.5 п.п.",
        "",
        "**Столбцы.** Значения по уровням идут через «/», от первого к верхнему. Радиус окна — средний, в клетках сетки",
        "штрихов; «окно видит» — какую долю штрихов своей цифры накрывает окно; деталей в окне — элементов предыдущего",
        "уровня (на уровне 1 — штрихов). Жирным — лучшее значение в разделе. «По обоим» — сходимость верха / лучшая +",
        "читатель / лучший. Подбор окон 2026-11-18 (поиск по образцу Хука — Дживса, остановлен на локальном оптимуме)",
        "выставил в конфиг 14 соседей и предел 1.15; с уровнем 0 (2026-11-18…19) — [10, 14] и 1.15 (прогон 266). С 0.13.1",
        "(2026-11-20) длина места окна своя у уровня — `LADDER_BINDING_PLACE_CODE_LENGTH`; прогоны 448–457 — подбор окон",
        "уровней 2–3 под долю цифры (уровень 2 — около трети, уровень 3 — вся цифра) на 100 цифрах: шум там больше —",
        "сходимость около ±0.04, точность на 100 проверочных около ±4–7 п.п.",
        "",
    ]
    for train in sorted({row.get("train") for row in runs.values()}, reverse=True):
        rows = [runs[number] for number in sorted(runs) if runs[number].get("train") == train]
        tests = sorted({row.get("tests") for row in rows})
        out += [f"## {train} обучающих цифр", "", f"Проверочных картинок: {', '.join(map(str, tests))}. Прогонов: {len(rows)}.", ""]
        out += table(rows)
        out += [""]
    text = "\n".join(out)
    io.open(TABLE, "w", encoding="utf-8", newline="").write(text.replace("\n", "\r\n"))
    print(f"таблица: {TABLE}; прогонов {len(runs)}; источник {STORE}")
    analysis(runs)


STAND = os.path.join(REPO, "docs", "runs-stand.jsonl")
ANALYSIS = os.path.join(REPO, "docs", "runs-analysis.csv")


def analysis(runs: dict) -> None:
    """Таблица для анализа (автор 2026-11-20: «данные по всем параметрам, которые мы подбираем, собирай в таблицу — они
    потом понадобятся для анализа»): по строке на прогон — все подбираемые параметры (столбцы AXES и любые другие
    изменённые настройки, действующее значение) и все мерки, какие есть. Полные записи стенда (уровень 0, детекторы по
    слоям, доля места, свечение, энергия) копируются в docs/runs-stand.jsonl — временная папка стенда может пропасть."""
    import csv

    stand = {}
    if os.path.exists(STAND):
        for line in io.open(STAND, encoding="utf-8"):
            if line.strip():
                item = json.loads(line)
                stand[item["run"]] = item
    log = os.path.join(HERE, "tune2_log.jsonl")
    if os.path.exists(log):
        for line in io.open(log, encoding="utf-8"):
            entry = json.loads(line)
            stand[entry["run"]] = {k: v for k, v in entry.items() if k not in ("table", "image")}
    with io.open(STAND, "w", encoding="utf-8", newline="\n") as stream:
        for number in sorted(stand):
            stream.write(json.dumps(stand[number], ensure_ascii=False) + "\n")

    def flat(value):  # noqa: ANN001, ANN202
        if isinstance(value, (list, tuple)):
            return " / ".join(str(v) for v in value)
        if isinstance(value, bool):
            return int(value)
        return value

    others = sorted({key for row in runs.values() for key in row.get("changes", {}) if key not in dict(AXES)})
    params = [key for key, _ in AXES] + others
    rows = []
    for number in sorted(runs):
        row = runs[number]
        changes = row.get("changes", {})
        full = stand.get(number, {})
        record = {"run": number, "date": row.get("date", ""), "train": row.get("train"), "tests": row.get("tests"),
                  "label": row.get("label", "")}
        for key in params:
            record[key] = flat(changes.get(key, DEFAULTS.get(key, "")))
        honest = row.get("honest", []) or []
        record.update({
            "levels": row.get("levels"), "top_honest": honest[-1] if honest else "",
            **{f"honest_L{i + 1}": v for i, v in enumerate(honest)},
            **{f"points_L{i + 1}": v for i, v in enumerate(row.get("points", []) or [])},
            **{f"sees_L{i + 1}": v for i, v in enumerate(row.get("sees", []) or [])},
            **{f"radius_L{i + 1}": v for i, v in enumerate(row.get("radius", []) or [])},
            "top_patches": row.get("patches", ""), "top_in_main": row.get("in_main", ""), "top_lonely": row.get("lonely", ""),
            "reader": row.get("reader", ""), "memory": row.get("memory", ""),
            "minutes": row.get("minutes") or row.get("train_minutes", ""),
        })
        zero = full.get("level0") or {}
        share = zero.get("place_share") or {}
        record.update({
            "L0_honest": zero.get("honest", ""), "L0_patches": zero.get("patches", ""), "L0_lonely": zero.get("lonely", ""),
            "L0_place_share_mean": share.get("mean", ""), "L0_place_share_min": share.get("min", ""),
            "L0_angle_heavier_points": share.get("angle_heavier", ""),
            "L0_smooth_deg": zero.get("smooth_deg", ""), "L0_inner_empty": zero.get("inner_empty", ""),
            "L0_longest_gap": zero.get("longest_gap", ""),
            "L0_retino_step": zero.get("retino_step", ""), "L0_retino_spearman": zero.get("retino_spearman", ""),
        })
        for i, shape in enumerate(full.get("detector_shapes") or []):
            record[f"detectors_by_layer_L{i + 1}"] = flat(shape.get("count", []))
            record[f"detector_radius_by_layer_L{i + 1}"] = flat(shape.get("radius", []))
        for name in ("glow45", "glow60", "energy", "detectors"):
            values = full.get(name)
            if isinstance(values, list):
                for i, v in enumerate(values):
                    record[f"{name}_L{i + 1}"] = v.get(name) if isinstance(v, dict) else v
        rows.append(record)
    columns = []
    for record in rows:
        for key in record:
            if key not in columns:
                columns.append(key)
    with io.open(ANALYSIS, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter=";")
        writer.writeheader()
        for record in rows:
            writer.writerow(record)
    print(f"для анализа: {ANALYSIS} ({len(rows)} прогонов × {len(columns)} столбцов); полные записи стенда: {STAND}")


main()
