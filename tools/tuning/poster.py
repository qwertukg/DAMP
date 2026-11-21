# -*- coding: utf-8 -*-
"""Большая диаграмма «путь картинки через DAMP по шагам» из сохранённых рисунков: README (docs/stage*.png), Rerun
(docs/rerun-*.png), дашборд, рисунки по шагам на пятёрке №182 (data/maps) и карты подбора (data/maps/tune2-*.png).

    python tools/tuning/poster.py [выход=data/maps/pipeline-poster.png]

Запускать из корня репозитория. Автор 2026-11-21: «нарисуй полную диаграмму с примерами в виде вырезок из всего, что у
нас есть… по шагам в деталях… очень большую и детальную, чтобы я мог увеличивать».
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

OUT = sys.argv[1] if len(sys.argv) > 1 else "data/maps/pipeline-poster.png"
W = 7200                      # ширина листа
PAD = 80                      # поля
TEXT_W = 1900                 # колонка текста шага
GAP = 50
IMG_H = 1500                  # высота строки вырезок
BG = (14, 14, 18)
CARD = (26, 27, 33)
CYCLE = (70, 200, 140)
FONT = "arial.ttf"
BOLD = "arialbd.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(BOLD if bold else FONT, size)


F_H1, F_H2, F_STEP, F_TITLE, F_BODY, F_CAP, F_NOTE = font(150, True), font(64), font(120, True), font(66, True), font(42), font(30), font(38)


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt, width: int) -> list[str]:
    lines = []
    for paragraph in text.split("\n"):
        words, line = paragraph.split(" "), ""
        for word in words:
            probe = (line + " " + word).strip()
            if draw.textlength(probe, font=fnt) <= width:
                line = probe
            else:
                if line:
                    lines.append(line)
                line = word
        lines.append(line)
    return lines


def text_height(draw, text, fnt, width, spacing=1.32) -> int:
    return int(len(wrap(draw, text, fnt, width)) * fnt.size * spacing)


def draw_text(draw, xy, text, fnt, width, fill=(225, 225, 230), spacing=1.32) -> int:
    x, y = xy
    for line in wrap(draw, text, fnt, width):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += int(fnt.size * spacing)
    return y


def load(spec):
    """spec: (путь, рамка вырезки или None, подпись)."""
    path, box, caption = spec
    image = Image.open(path).convert("RGB")
    if box:
        image = image.crop(box)
    return image, caption, path


def layout_images(specs, avail_w, target_h):
    """Раскладка вырезок по строкам: высота строки target_h (широкие сжимаются по ширине)."""
    rows, row, row_w = [], [], 0
    for spec in specs:
        image, caption, path = load(spec)
        scale = min(target_h / image.height, avail_w / image.width)
        w, h = int(image.width * scale), int(image.height * scale)
        if row and row_w + w > avail_w:
            rows.append(row)
            row, row_w = [], 0
        row.append((image.resize((w, h), Image.LANCZOS), caption, path))
        row_w += w + GAP
    if row:
        rows.append(row)
    return rows


def rows_height(rows) -> int:
    return sum(max(img.height for img, _, _ in row) + 110 for row in rows)


def paste_rows(sheet, draw, rows, x0, y0) -> int:
    y = y0
    for row in rows:
        x, top = x0, max(img.height for img, _, _ in row)
        for img, caption, path in row:
            draw.rectangle((x - 6, y - 6, x + img.width + 5, y + img.height + 5), outline=(70, 72, 82), width=3)
            sheet.paste(img, (x, y))
            draw.text((x, y + img.height + 14), caption, font=F_CAP, fill=(200, 200, 205))
            draw.text((x, y + img.height + 52), path, font=F_CAP, fill=(120, 122, 130))
            x += img.width + GAP
        y += top + 110
    return y


# Вырезки карт подбора 576: панель уровня n — x от 20 + n·320, ширина 300; заголовки с y = 30, карта до y = 374.
def tune_panel(n: int, run: int = 576):
    left = 20 + n * 320
    return (f"data/maps/tune2-{run}.png", (left, 30, left + 300, 380), f"карта уровня {n} (прогон {run}){' — цвет: угол штриха' if n == 0 else ' — цвет: цифра'}")


STEPS = [
    {"n": "1", "stage": "этап 1", "title": "Берутся 250 обучающих картинок MNIST",
     "text": "Первые 250 картинок MNIST train; каждая 28×28 увеличивается в 8 раз — рабочая картинка 224×224. Проверочные — 100 картинок t10k, в обучении их нет.\n"
             "С(§10.1.4: «работаем над MNIST»)",
     "images": [("docs/stage1-digits.png", None, "цифры 0…9 и их штрихи"), ("docs/rerun-1-stimuli.png", None, "Rerun, вкладка «1. Стимулы»")]},
    {"n": "2", "stage": "этап 1", "title": "На картинке находятся штрихи",
     "text": "Д. Детекторы над точками середины линии (гребень расстояния до края): соседние точки с похожим направлением дают штрих — короткий отрезок с углом 0…180° и местом. Около 53 штрихов на цифру, 13.3 тыс. на 250 цифр. Дальше сырая картинка не нужна.\n"
             "С(§6.4: детектор по точкам контура; замысел автора — одна середина линии вместо двух краёв)",
     "images": [("docs/stage1-sticks.png", None, "пятёрка: картинка → точки середины линии → 54 штриха")]},
    {"n": "3", "stage": "этап 2", "title": "Каждому штриху — код угла",
     "text": "С. 10 слоёв детекторов угла: от 2 широких до 180 узких, перекрытие 0.85 (подбор v2). Сработавший детектор ставит один случайный бит в цветовой полосе своего слоя: широкий слой — красные младшие биты, узкий — фиолетовые старшие. Код угла — 256 бит, около 56 единиц (22 %, не больше 25 % по §2).\n"
             "С(§3.2, §4.2, §4.4, §4.5)",
     "images": [("docs/stage2-detectors.png", None, "угол и место кодируются одинаково: слоями, по биту на детектор")]},
    {"cycle": "begin"},
    {"n": "4", "stage": "этап 2", "title": "Окна и их коды",
     "text": "Уровень 0: окно на каждом штрихе держит его одного; место штриха — от центра цифры, кругами места (10 слоёв, 2…16 кругов по стороне, перекрытие 0.52), 39 бит на круг — место чуть весомее угла.\n"
             "Уровни 1–3: окно — кластер похожих соседних деталей прошлого уровня той же картинки (10 / 3 / 14 соседей, нижний предел радиуса 1.15 / 0.5 / 2.0 поля детали); нет кластера — окно на всю картинку. Место элемента — от центра окна.\n"
             "Код окна = биты места всех элементов через OR + содержание: код элемента под XOR-маской своего места, пулинг «сильнейший»; содержание до 4096 бит, сверх 25 % — слиянием цветов.\n"
             "Д С(П.3–П.5; §6.4.3)",
     "images": [("docs/stage6-keyholes.png", None, "окна уровней 1–3 вокруг одного места пятёрки"),
                ("docs/stage6-step-level0.png", (0, 0, 900, 560), "уровень 0: окно и код окна (место + содержание)"),
                ("docs/rerun-2-coding.png", None, "Rerun, «2. Кодирование»: биты места и содержания")]},
    {"n": "5", "stage": "этап 3", "title": "Первая подача: раскладка — карта, память уровня",
     "text": "С. Коды окон всех обучающих картинок раскладываются по квадратной сетке (15 % пустых клеток). Дальняя фаза: обмены пар точек, гауссов колокол сжимается от размера карты до 3 клеток, первая точка пары — по −ln Ê; затем ближняя фаза. Похожие коды ложатся рядом: на уровне 0 — пинвилы углов, место плавно (ретинотопия 0.92); выше — детали одной цифры собираются в пятна.\n"
             "Перед раскладкой окна с косинусом ≥ 0.8 сливаются к одному образцу (уровень 0 — только одинаковые). λ раскладки — квантиль сходств кодов карты (0.82).\n"
             "С(§5.2–§5.9, §7.1.6–§7.1.7, §2.2.4)",
     "images": [tune_panel(0), ("docs/stage3-layout-level0.png", None, "уровень 0: в ячейке — штрих точки (README)"),
                ("data/maps/gauss-steps-561.png", None, "как точка выбирает соседей: колокол сжимается"),
                ("data/maps/retinotopy-570-561-555.png", None, "уровень 0 как ретинотопическая карта"),
                ("docs/stage3-layout.png", None, "уровень 1: в ячейке — что видит окно точки"),
                ("docs/stage3-pinwheel.png", None, "тест пинвила: 4 позиции × 360 углов"),
                ("docs/rerun-3-layout.png", None, "Rerun, «3. Раскладка»: карта по ходу, цвет — цифра")]},
    {"n": "6", "stage": "этап 4", "title": "Над картой строятся детекторы",
     "text": "С. 5 слоёв детекторов, порог каждого слоя — квантиль сходств карты. Вокруг центров с высокой энергией — кластеры DBSCAN (eps 1.5, мин. 3 точки); радиус детектора — по наибольшему заполнению (поправка на клетку 0.5); центры не перекрываются; слой полон после 400 попыток без изменений. 500–700 детекторов на карту, у каждого — свой случайный бит.\n"
             "С(§6.4, §7.1.10)",
     "images": [("docs/stage4-detectors.png", None, "детекторы над картой уровня 1 по слоям"),
                ("docs/rerun-4-detectors.png", None, "Rerun, «4. Детекторы»: слои и сработавшие")]},
    {"n": "7", "stage": "этапы 4–5", "title": "Вторая подача: активация → эмбеддинг (деталь уровня)",
     "text": "С. Код того же окна той же картинки сравнивается со всеми точками готовой карты: зажигаются похожие не ниже λ_a; одиночные зажжённые на отшибе гаснут (энергия активации ≥ 0.5, Д). Детектор срабатывает по энергии зажжённых точек в своём круге: E(d) = Σ a·e / e_d, нормировка по обучающим окнам (Д). В код детали идут σ = 10 сильнейших сработавших — по одному биту; длина ≈ σ / 0.1 = 100 бит.\n"
             "С(§6.1, §6.5, §6.6, §7.1.12)",
     "images": [("docs/stage4-activation.png", None, "окно → зажглись точки памяти → сработали детекторы"),
                ("data/maps/sigma561-five182-level0.png", None, "σ по шагам на пятёрке №182, уровень 0"),
                ("docs/stage5-embedding.png", None, "эмбеддинг: сильнейшие детекторы → код детали"),
                ("docs/rerun-5-embeddings.png", None, "Rerun, «5. Эмбеддинги»")]},
    {"n": "8", "stage": "этап 6", "title": "Пятна и переход на следующий уровень",
     "text": "Д. Из соседних похожих деталей одной картинки (ближе доли своего поля, код похож ≥ 0.8) на следующий уровень идёт сильнейшая. Детали уходят на шаг 4 следующего уровня. Цикл останавливается, когда окно уровня видит цифру почти целиком (≥ 90 % штрихов) — сейчас после уровня 3.\n"
             "Д С(П.2, П.3: пятна; П.7: пока сочетания не станут кодами целых понятий)",
     "images": [("docs/rerun-6-cycle.png", None, "Rerun, «6. Цикл»: окна уровней вокруг одного места"),
                ("docs/stage6-step-level1.png", None, "уровень 1: окно → код → память → деталь")]},
    {"cycle": "end"},
    {"levels": True},
    {"n": "9", "stage": "этап 7", "title": "Пороги знания и читатель",
     "text": "Д. 100 обучающих картинок проходят заново целиком: штрихи → все витки по готовым картам (карты не меняются, детекторы только срабатывают). По ним — пороги памяти «знаю целиком / по частям / не знаю» и линейный читатель над уровнями активации.\n"
             "Д С(§2.2.5, §7.1.13.3; В — читатель: §9.3 только упоминает)",
     "images": [("docs/stage7-fives.png", None, "проверочные пятёрки: верхние детали и их голоса")]},
    {"n": "10", "stage": "этап 7", "title": "Проверка: распознавание 100 проверочных картинок",
     "text": "Картинка проходит шаги 2–3 и все витки по готовым картам; память (голоса кусочков) и читатель дают ответ. Сейчас (пара 596/597): читатель 92 %, память 85 %.\n"
             "Классы наверху собраны слабо: М1 ≈ 0.31 — в главном пятне своей цифры 31 % точек верха, пятен около 100 вместо 10; честная сходимость верха 0.63. Цель автора — 10 кластеров наверху.",
     "images": [("docs/stage7-top-map.png", None, "карта верхних деталей (README)"), ("docs/stage7-accuracy.png", None, "узнавание по уровням (README, 0.13)"),
                ("docs/rerun-7-recognition.png", None, "Rerun, «7. Распознавание»"), ("docs/dashboard.png", None, "дашборд: рисуем цифру — кусочки и кластеры всех уровней")]},
]

LEVELS = [
    (0, "1 штрих", "2 %", "13 305", "0.31"),
    (1, "~11 деталей уровня 0 — обрывки контура", "21 %", "10 448", "0.38"),
    (2, "~5 деталей уровня 1 — части цифры", "35 %", "8 129", "0.34"),
    (3, "~27 деталей уровня 2 — цифра целиком", "98 %", "4 487", "0.65"),
]


def main() -> None:
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    avail = W - 2 * PAD - TEXT_W - GAP - 120
    blocks = []
    for step in STEPS:
        if "cycle" in step:
            blocks.append(("cycle", step["cycle"], 260))
        elif "levels" in step:
            rows_per_level = []
            for n, *_ in LEVELS:
                specs = [(f"docs/stage6-step-level{n}.png", None, f"уровень {n}: шаг на пятёрке (README)"),
                         (f"data/maps/seg561-five182-level{n}.png", None, f"уровень {n}: путь окна пятёрки №182 до кода детали"), tune_panel(n)]
                rows_per_level.append(layout_images(specs, avail, 1350))
            extra = layout_images([("docs/stage6-level-maps.png", None, "карты памяти уровней (вырезки из середины)"),
                                   ("docs/stage7-class-clusters.png", None, "собираются ли цифры в пятна по уровням")], avail, 1300)
            height = 330 + sum(rows_height(r) + 60 for r in rows_per_level) + rows_height(extra) + 80
            blocks.append(("levels", (rows_per_level, extra), height))
        else:
            rows = layout_images(step["images"], avail, IMG_H)
            th = text_height(probe, step["text"], F_BODY, TEXT_W - 60) + 360
            blocks.append(("step", (step, rows), max(rows_height(rows), th) + 120))
    overview = layout_images([("docs/pipeline-overview.png", None, "обзор пайплайна (README)")], W - 2 * PAD, 1500)
    footer = layout_images([("docs/structure.png", None, "устройство проекта"), ("docs/rerun-0-overview.png", None, "Rerun, «Обзор»: сводка прогона по уровням")], W - 2 * PAD, 1400)
    head_h = 560 + rows_height(overview) + 60
    total = head_h + sum(h for *_, h in blocks) + rows_height(footer) + 300
    sheet = Image.new("RGB", (W, total), BG)
    draw = ImageDraw.Draw(sheet)

    # Шапка.
    draw.text((PAD, PAD), "DAMP — путь картинки через сеть, по шагам", font=F_H1, fill=(255, 255, 255))
    draw_text(draw, (PAD, PAD + 200), "2026-11-21 · конфигурация 561 + перекрытие кода угла 0.85 · 250 обучающих цифр MNIST, 100 проверочных · "
              "вырезки: рисунки README (docs/stage*.png), Rerun (docs/rerun-*.png), дашборд, рисунки по шагам на пятёрке №182 (data/maps), "
              "карты подбора (data/maps/tune2-576.png). Рисунки сняты на разных версиях 0.13.x — числа на них могут отличаться; числа в тексте — текущие.",
              F_H2, W - 2 * PAD, fill=(190, 190, 200))
    draw.text((PAD, PAD + 420), "Метки: С — по статье · Д — наше расширение · В — вне статьи · С(…) — что говорит статья или продолжение (П.)",
              font=F_NOTE, fill=(150, 200, 255))
    y = paste_rows(sheet, draw, overview, PAD, head_h - rows_height(overview) - 20)

    cycle_top = None
    for kind, payload, height in blocks:
        if kind == "cycle":
            colour = CYCLE
            draw.rectangle((PAD, y + 40, W - PAD, y + height - 40), fill=(20, 52, 40))
            if payload == "begin":
                cycle_top = y + height - 40
                draw.polygon([(PAD + 60, y + 90), (PAD + 60, y + 210), (PAD + 170, y + 150)], fill=colour)
                draw.text((PAD + 220, y + 90), "НАЧАЛО ЦИКЛА (этап 2) — шаги 4–8 повторяются на каждом уровне: 0 → 1 → 2 → 3",
                          font=F_TITLE, fill=colour)
                draw.text((PAD + 220, y + 175), "на уровне 0 элементы — штрихи; на уровне k — детали уровня k − 1 тех же картинок",
                          font=F_NOTE, fill=(170, 220, 195))
            else:
                draw.polygon([(PAD + 170, y + 90), (PAD + 170, y + 210), (PAD + 60, y + 150)], fill=colour)
                draw.text((PAD + 220, y + 90), "КОНЕЦ ЦИКЛА (этап 6) — детали уровня идут на шаг 4 следующего уровня",
                          font=F_TITLE, fill=colour)
                draw.text((PAD + 220, y + 175), "витков 4: уровни 0, 1, 2, 3; на уровне 3 окно видит 98 % цифры — стоп",
                          font=F_NOTE, fill=(170, 220, 195))
                if cycle_top is not None:
                    # Стрелка возврата вдоль левого края: от конца цикла к началу.
                    draw.rectangle((PAD + 10, cycle_top, PAD + 34, y + 60), fill=colour)
                    draw.polygon([(PAD - 8, cycle_top + 70), (PAD + 52, cycle_top + 70), (PAD + 22, cycle_top)], fill=colour)
            y += height
            continue
        if kind == "levels":
            rows_per_level, extra = payload
            draw.rectangle((PAD + 60, y + 20, W - PAD, y + height - 20), fill=CARD)
            draw.text((PAD + 120, y + 60), "Витки цикла на пятёрке: уровни 0 → 3", font=F_H1, fill=(255, 255, 255))
            draw.text((PAD + 120, y + 230), "одна и та же картинка, на каждом уровне окно больше: штрих → обрывок контура → часть цифры → цифра целиком",
                      font=F_NOTE, fill=(190, 190, 200))
            yy = y + 330
            for (n, inside, sees, points, honest), rows in zip(LEVELS, rows_per_level):
                draw.text((PAD + 120, yy), f"Уровень {n}", font=F_TITLE, fill=CYCLE)
                draw_text(draw, (PAD + 120, yy + 100), f"в окне: {inside}\nокно видит {sees} цифры\nточек карты: {points}\nчестная сходимость: {honest}",
                          F_BODY, TEXT_W - 120)
                yy = paste_rows(sheet, draw, rows, PAD + TEXT_W + GAP + 60, yy) + 60
            paste_rows(sheet, draw, extra, PAD + TEXT_W + GAP + 60, yy)
            draw_text(draw, (PAD + 120, yy), "Карты всех уровней и сбор цифр в пятна — по уровням снизу вверх.", F_BODY, TEXT_W - 120)
            y += height
            continue
        step, rows = payload
        inside_cycle = step["n"] in ("4", "5", "6", "7", "8")
        left = PAD + (60 if inside_cycle else 0)
        draw.rectangle((left, y + 20, W - PAD, y + height - 20), fill=CARD)
        if inside_cycle:
            draw.rectangle((PAD + 10, y, PAD + 34, y + height), fill=CYCLE)
        draw.ellipse((left + 40, y + 60, left + 240, y + 260), fill=(60, 120, 220))
        number = step["n"]
        draw.text((left + 140 - probe.textlength(number, font=F_STEP) / 2, y + 90), number, font=F_STEP, fill=(255, 255, 255))
        draw.text((left + 280, y + 70), step["stage"], font=F_NOTE, fill=(150, 200, 255))
        draw_text(draw, (left + 280, y + 120), step["title"], F_TITLE, TEXT_W - 300, fill=(255, 255, 255), spacing=1.15)
        draw_text(draw, (left + 50, y + 330), step["text"], F_BODY, TEXT_W - 60)
        paste_rows(sheet, draw, rows, PAD + TEXT_W + GAP + 60, y + 70)
        y += height

    draw.text((PAD, y + 40), "Устройство проекта и сводка прогона", font=F_TITLE, fill=(255, 255, 255))
    paste_rows(sheet, draw, footer, PAD, y + 160)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    sheet.save(OUT, optimize=True)
    print(f"{OUT}: {sheet.width}×{sheet.height}, {os.path.getsize(OUT) / 2**20:.1f} МБ")


main()
