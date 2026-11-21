# -*- coding: utf-8 -*-
"""Квадратная блок-схема «путь картинки через DAMP»: шаги, цикл по уровням, условия и внутренние петли; у шагов —
вырезки деталей из сохранённых рисунков (README docs/stage*.png, дашборд, рисунки по шагам data/maps, карты подбора).

    python tools/tuning/diagram.py [выход=data/maps/pipeline-diagram.png]

Автор 2026-11-21: «сделай не в виде ленты… а в виде большой детальной квадратной картинки с диаграммой (с циклами и
условиями), где шаги сопровождаются вырезками деталей из того, что есть, а не целыми картинками».
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

OUT = sys.argv[1] if len(sys.argv) > 1 else "data/maps/pipeline-diagram.png"
S = 9000
BG, CARD, EDGE = (13, 14, 18), (27, 29, 36), (80, 84, 96)
BLUE, GREEN, ORANGE, YELLOW, GREY = (70, 140, 240), (70, 205, 140), (240, 140, 60), (245, 205, 70), (200, 202, 210)


def F(size, bold=False):
    return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)


F_T, F_SUB, F_NUM, F_HEAD, F_BODY, F_SMALL, F_CAP = F(140, True), F(52), F(84, True), F(54, True), F(34), F(30), F(24)
sheet = Image.new("RGB", (S, S), BG)
d = ImageDraw.Draw(sheet)


def wrap(text, fnt, width):
    out = []
    for para in text.split("\n"):
        line = ""
        for word in para.split(" "):
            probe = (line + " " + word).strip()
            if d.textlength(probe, font=fnt) <= width:
                line = probe
            else:
                out.append(line)
                line = word
        out.append(line)
    return out


def text(x, y, s, fnt, width, fill=GREY, gap=1.3):
    for line in wrap(s, fnt, width):
        d.text((x, y), line, font=fnt, fill=fill)
        y += int(fnt.size * gap)
    return y


def crop(path, box):
    im = Image.open(path).convert("RGB")
    return im.crop(box) if box else im


def images(x, y, w, h, specs, row_h=None):
    """Вырезки в прямоугольнике (x, y, w, h): по строкам, высота строки подбирается, чтобы всё влезло."""
    items = [(crop(p, b), cap) for p, b, cap in specs]
    n = len(items)
    best = None
    # Все разбиения на строки подряд (≤ 2^(n−1)): у каждой строки своя высота — во всю ширину; если по высоте не
    # влезает — все строки сжимаются одинаково. Берётся разбиение с наибольшей площадью картинок.
    for mask in range(1 << (n - 1)):
        groups, current = [], [items[0]]
        for i in range(1, n):
            if mask >> (i - 1) & 1:
                groups.append(current)
                current = []
            current.append(items[i])
        groups.append(current)
        heights = [(w - 30 * (len(g) - 1)) / sum(im.width / im.height for im, _ in g) for g in groups]
        if row_h:
            heights = [min(hh, row_h) for hh in heights]
        need = sum(heights) + 46 * len(groups)
        k = min(1.0, (h - 46 * len(groups)) / max(1.0, sum(heights)))
        heights = [hh * k for hh in heights]
        area = sum(hh * hh * sum(im.width / im.height for im, _ in g) for hh, g in zip(heights, groups))
        if best is None or area > best[0]:
            best = (area, groups, heights)
    _, groups, heights = best
    yy = y
    for g, rh in zip(groups, heights):
        xx = x
        for im, cap in g:
            iw = max(1, int(im.width * rh / im.height))
            im = im.resize((iw, max(1, int(rh))), Image.LANCZOS)
            d.rectangle((xx - 4, yy - 4, xx + iw + 3, yy + int(rh) + 3), outline=EDGE, width=3)
            sheet.paste(im, (xx, yy))
            d.text((xx, yy + int(rh) + 8), cap, font=F_CAP, fill=(160, 162, 172))
            xx += iw + 30
        yy += int(rh) + 46
    return yy


def node(x, y, w, h, num, title, stage, body, colour=BLUE):
    d.rounded_rectangle((x, y, x + w, y + h), radius=40, fill=CARD, outline=colour, width=6)
    d.ellipse((x + 30, y + 30, x + 170, y + 170), fill=colour)
    d.text((x + 100 - d.textlength(num, font=F_NUM) / 2, y + 52), num, font=F_NUM, fill=(255, 255, 255))
    d.text((x + 200, y + 34), stage, font=F_SMALL, fill=(150, 200, 255))
    yy = text(x + 200, y + 76, title, F_HEAD, w - 240, fill=(255, 255, 255), gap=1.15)
    return text(x + 40, max(yy, y + 190) + 14, body, F_BODY, w - 80)


def arrow(points, colour=GREY, width=10, head=40, label=None, at=0, lfill=None):
    d.line(points, fill=colour, width=width, joint="curve")
    (x1, y1), (x2, y2) = points[-2], points[-1]
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    tip = (x2, y2)
    left = (x2 - head * math.cos(a) + head * 0.6 * math.sin(a), y2 - head * math.sin(a) - head * 0.6 * math.cos(a))
    right = (x2 - head * math.cos(a) - head * 0.6 * math.sin(a), y2 - head * math.sin(a) + head * 0.6 * math.cos(a))
    d.polygon([tip, left, right], fill=colour)
    if label:
        lx, ly = points[at]
        d.text((lx + 16, ly + 10), label, font=F_SMALL, fill=lfill or colour)


def diamond(cx, cy, w, h, s, fill=(52, 44, 18), edge=YELLOW, fnt=None):
    fnt = fnt or F_SMALL
    d.polygon([(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)], fill=fill, outline=edge, width=5)
    lines = wrap(s, fnt, w * 0.62)
    y0 = cy - len(lines) * fnt.size * 0.62
    for line in lines:
        d.text((cx - d.textlength(line, font=fnt) / 2, y0), line, font=fnt, fill=(255, 240, 200))
        y0 += fnt.size * 1.25


def chain(x, y, w, items, loop=None, exit_=None):
    """Мини-схема внутри узла: шаги (◇ — условие) слева направо; loop — подпись возврата (петля), exit_ — выход."""
    n = len(items)
    bw = (w - 60 * (n - 1)) / n
    bh = 150
    centres = []
    for i, it in enumerate(items):
        bx = x + i * (bw + 60)
        if it.startswith("◇"):
            diamond(bx + bw / 2, y + bh / 2, bw, bh + 40, it[1:].strip(), fnt=F_CAP)
        else:
            d.rounded_rectangle((bx, y, bx + bw, y + bh), radius=20, fill=(38, 42, 54), outline=(110, 116, 132), width=3)
            lines = wrap(it, F_CAP, bw - 24)
            yy = y + bh / 2 - len(lines) * 15
            for line in lines:
                d.text((bx + bw / 2 - d.textlength(line, font=F_CAP) / 2, yy), line, font=F_CAP, fill=(230, 230, 236))
                yy += 30
        centres.append(bx + bw / 2)
        if i:
            arrow([(bx - 58, y + bh / 2), (bx - 4, y + bh / 2)], width=6, head=22)
    bottom = y + bh + 20
    if loop:
        arrow([(centres[-1], bottom), (centres[-1], bottom + 70), (centres[0], bottom + 70), (centres[0], bottom + 4)], colour=ORANGE, width=6, head=22)
        d.text((centres[0] + 30, bottom + 80), "петля: " + loop, font=F_CAP, fill=ORANGE)
        bottom += 130
    if exit_:
        d.text((x, bottom + 4), "→ " + exit_, font=F_CAP, fill=GREEN)
        bottom += 44
    return bottom + 20


ST1, ST2 = "docs/stage1-sticks.png", "docs/stage2-detectors.png"
ACT, EMB, DET = "docs/stage4-activation.png", "docs/stage5-embedding.png", "docs/stage4-detectors.png"
SIG, GAU, DASH, STEP1 = "data/maps/sigma561-five182-level1.png", "data/maps/gauss-steps-561.png", "docs/dashboard.png", "docs/stage6-step-level1.png"
TUNE = "data/maps/tune2-576.png"


def tune(n):
    return (TUNE, (20 + n * 320, 74, 320 + n * 320, 374), f"карта уровня {n}")


# ---------------------------------------------------------------- шапка
d.text((150, 60), "DAMP — путь картинки через сеть: цикл, условия, детали", font=F_T, fill=(255, 255, 255))
text(150, 240, "250 обучающих цифр MNIST, 100 проверочных · конфигурация 561 + перекрытие угла 0.85 + код окна до 8192 (2026-11-21) · "
     "вырезки из рисунков README, дашборда, рисунков по шагам на пятёрке №182 и карт подбора; числа в тексте — текущие · "
     "С — по статье, Д — наше расширение, В — вне статьи; жёлтый ромб — условие, оранжевая стрелка назад — петля (повтор)", F_SUB, S - 300, fill=(185, 188, 198))

# ---------------------------------------------------------------- шаги 1–3
x0, w0 = 150, 2350
y = node(x0, 460, w0, 1300, "1", "Картинки MNIST", "этап 1",
         "Первые 250 картинок MNIST train; 28×28 → рабочая 224×224 (×8). Проверочные — 100 из t10k, в обучении их нет. С(§10.1.4)")
images(x0 + 40, y + 20, w0 - 80, 1760 - y - 60, [("docs/stage1-digits.png", (10, 130, 1320, 330), "цифры 0…4 — их штрихи"),
                                               (ST1, (80, 250, 420, 530), "пятёрка 28×28 → 224×224")])
y = node(x0, 1840, w0, 1300, "2", "Штрихи", "этап 1",
         "Д. Детекторы над точками середины линии (гребень расстояния до края): штрих — отрезок с углом 0…180° и местом; ~53 на цифру, "
         "13.3 тыс. на 250. Дальше картинка не нужна. С(§6.4: детектор по точкам контура)")
images(x0 + 40, y + 20, w0 - 80, 3140 - y - 60, [(ST1, (520, 260, 840, 520), "300 точек середины линии"),
                                               (ST1, (960, 260, 1280, 520), "54 штриха, цвет — наклон"),
                                               (DASH, (330, 90, 500, 260), "дашборд: штрихи рисунка")])
y = node(x0, 3220, w0, 1650, "3", "Код угла штриха", "этап 2",
         "С. 10 слоёв детекторов угла: 2 широких … 180 узких, перекрытие 0.85; сработавший ставит один случайный бит в полосе своего "
         "слоя (широкий — красные младшие биты, узкий — фиолетовые). 256 бит, ~56 единиц (22 % ≤ 25 %, §2). С(§3.2, §4.2, §4.4)")
images(x0 + 40, y + 20, w0 - 80, 4870 - y - 60, [(ST2, (20, 170, 320, 490), "угол 105°: 10 слоёв кольцами"),
                                               (ST2, (330, 170, 940, 490), "слои 1…10: горят 2–4 детектора"),
                                               (ST2, (955, 175, 1345, 280), "код угла: бит на детектор")])

# ---------------------------------------------------------------- рамка цикла
fx, fy, fw, fh = 2680, 430, 6170, 6420
d.rounded_rectangle((fx, fy, fx + fw, fy + fh), radius=60, outline=GREEN, width=12)
d.rectangle((fx + 80, fy - 40, fx + 4400, fy + 40), fill=BG)
d.text((fx + 100, fy - 36), "ЦИКЛ (этапы 2 → 6): шаги 4–8 повторяются на уровнях 0 → 1 → 2 → 3", font=F_HEAD, fill=GREEN)

# Узлы цикла.
N4 = (3350, 600, 2400, 2150)
N5 = (5950, 600, 2750, 2150)
N6 = (5950, 2930, 2750, 1720)
N7 = (5950, 4830, 2750, 1900)
N8 = (3350, 4830, 2400, 1900)
CEN = (3350, 2930, 2400, 1720)

x, y, w, h = N4
yy = node(x, y, w, h, "4", "Окна и их коды", "этап 2 ← начало витка",
          "Д. Уровень 0: окно — один штрих, место от центра цифры кругами места (10 слоёв, 2…16 кругов, 39 бит на круг). "
          "Уровни 1–3: окно — кластер похожих соседних деталей прошлого уровня той же картинки (10 / 3 / 14), место — от центра окна. "
          "Код окна = место всех элементов (OR) + содержание: код элемента под XOR-маской места, пулинг «сильнейший», до 8192 бит. С(П.3–П.5)", GREEN)
yy = chain(x + 40, yy + 10, w - 80, ["детали уровня k − 1 картинки", "◇ кластер ≥ N похожих соседей?", "окно = кластер", "нет — окно на всю картинку"])
images(x + 40, yy, w - 80, y + h - yy - 40, [(ST2, (20, 610, 330, 900), "место штриха от центра цифры"),
                                            (ST2, (330, 620, 940, 930), "круги места: 10 слоёв"),
                                            (STEP1, (92, 218, 402, 483), "окно уровня 1 на пятёрке"),
                                            (STEP1, (535, 132, 805, 523), "код окна: место + содержание")])

x, y, w, h = N5
yy = node(x, y, w, h, "5", "Первая подача: раскладка — карта (память уровня)", "этап 3",
          "С. Коды окон всех обучающих картинок — по квадратной сетке (15 % пустых). Похожие ложатся рядом: на уровне 0 — пинвилы углов и "
          "плавное место (ретинотопия 0.92), выше — пятна деталей одной цифры. Перед раскладкой окна с косинусом ≥ 0.8 сливаются; λ — квантиль 0.82. "
          "С(§5.2–§5.9, §7.1.6–7.1.7)", GREEN)
yy = chain(x + 40, yy + 10, w - 80, ["пара точек: первая — по −ln Ê", "◇ энергия после обмена выше?", "обменять", "колокол сжимается 62 → 3 клетки"],
           loop="дальняя фаза ~2 300 шагов × 4 656 пар, пока доля обменов > порога", exit_="ближняя фаза 20 шагов → карта готова")
images(x + 40, yy, w - 80, y + h - yy - 40, [tune(0), (GAU, (1290, 120, 1840, 430), "колокол: начало · середина · конец"),
                                            (GAU, (130, 700, 900, 1160), "вес соседа w(d)"),
                                            ("docs/stage3-layout-level0.png", (20, 160, 310, 305), "ур. 0: штрих в ячейке"),
                                            ("docs/stage3-layout.png", (18, 150, 480, 360), "ур. 1: что видит окно точки")])

x, y, w, h = N6
yy = node(x, y, w, h, "6", "Детекторы над картой", "этап 4",
          "С. 5 слоёв, порог слоя — квантиль сходств карты; кластер DBSCAN (eps 1.5, мин. 3) вокруг центра с энергией ≥ μ_e; радиус — по "
          "наибольшему заполнению (+0.5 клетки); центры не перекрываются. 500–700 детекторов, у каждого свой бит. С(§6.4, §7.1.10)", GREEN)
yy = chain(x + 40, yy + 10, w - 80, ["случайный центр, энергия ≥ μ_e", "кластер DBSCAN", "радиус по заполнению", "◇ центр не перекрыт?", "детектор в слой"],
           loop="пока 400 попыток подряд без нового детектора; затем следующий слой (5 слоёв)")
images(x + 40, yy, w - 80, y + h - yy - 40, [(DET, (15, 130, 1000, 340), "слои 1…5: детекторы мельчают"),
                                            (SIG, (1353, 131, 1808, 581), "сработавшие на окне пятёрки")])

x, y, w, h = N7
yy = node(x, y, w, h, "7", "Вторая подача: активация → деталь уровня", "этапы 4–5",
          "С. Код того же окна сравнивается со всеми точками карты; детекторы срабатывают по энергии зажжённых точек в круге: "
          "E(d) = Σ a·e / e_d (нормировка по обучающим окнам — Д). В код детали — по биту от σ = 10 сильнейших; длина ≈ σ / 0.1 = 100 бит. "
          "С(§6.1, §6.5, §6.6, §7.1.12)", GREEN)
yy = chain(x + 40, yy + 10, w - 80, ["◇ сходство с точкой ≥ λ_a?", "◇ энергия активации ≥ 0.5? (Д)", "уровни E(d) детекторов", "σ = 10 сильнейших → код детали"])
images(x + 40, yy, w - 80, y + h - yy - 40, [(ACT, (480, 190, 860, 560), "зажглись точки памяти"),
                                            (ACT, (920, 190, 1300, 560), "сработали детекторы"),
                                            (SIG, (81, 707, 1010, 1162), "σ сильнейших из сработавших"),
                                            (SIG, (1242, 697, 1626, 1111), "код детали: 10 бит"),
                                            (EMB, (550, 290, 820, 500), "детали на цифре")])

x, y, w, h = N8
yy = node(x, y, w, h, "8", "Пятна и переход вверх", "этап 6 ← конец витка",
          "Д. Из соседних похожих деталей одной картинки (ближе доли поля, код похож ≥ 0.8) остаётся сильнейшая; детали — элементы окон "
          "следующего уровня. С(П.2, П.3: пятна; П.7: пока сочетания не станут кодами целых понятий)", GREEN)
yy = chain(x + 40, yy + 10, w - 80, ["детали уровня k картинки", "◇ рядом похожая (≥ 0.8)?", "оставить сильнейшую"])
images(x + 40, yy, w - 80, y + h - yy - 40, [("docs/stage6-keyholes.png", (60, 60, 660, 660), "окна уровней 1–3 вокруг одного места"),
                                            (EMB, (465, 155, 870, 205), "код детали"),
                                            (DASH, (1180, 330, 1450, 590), "дашборд: кластеры уровня 2")])

x, y, w, h = CEN
d.rounded_rectangle((x, y, x + w, y + h), radius=40, fill=(20, 36, 30), outline=GREEN, width=4)
d.text((x + 40, y + 30), "Витки цикла: окно растёт", font=F_HEAD, fill=GREEN)
text(x + 40, y + 110, "уровень 0: 1 штрих, 2 % цифры, 13.3 тыс. точек · уровень 1: ~11 штрихов, 21 %, 10.4 тыс. · "
     "уровень 2: ~5 деталей, 35 %, 8.1 тыс. · уровень 3: ~27 деталей, 98 %, 4.5 тыс.", F_SMALL, w - 80)
yy = images(x + 40, y + 300, w - 80, 760, [tune(0), tune(1), tune(2), tune(3)])
images(x + 40, yy + 10, w - 80, y + h - yy - 60, [("docs/stage7-class-clusters.png", (10, 735, 1686, 915), "уровень 3: где лежит каждая цифра (синим)")])

# Стрелки цикла: 4 → 5 → 6 → 7 → 8 → ◇ → (нет) 4 / (да) выход.
arrow([(N4[0] + N4[2], 1650), (N5[0] - 6, 1650)], GREEN, 14, 50)
arrow([(N5[0] + N5[2] / 2, N5[1] + N5[3]), (N6[0] + N6[2] / 2, N6[1] - 6)], GREEN, 14, 50)
arrow([(N6[0] + N6[2] / 2, N6[1] + N6[3]), (N7[0] + N7[2] / 2, N7[1] - 6)], GREEN, 14, 50)
arrow([(N7[0], 5780), (N8[0] + N8[2] + 6, 5780)], GREEN, 14, 50)
DX, DY = 3010, 5780
diamond(DX, DY, 560, 640, "окно уровня видит ≥ 90 % цифры?", fnt=F_SMALL)
arrow([(N8[0], DY), (DX + 284, DY)], GREEN, 14, 50)
arrow([(DX, DY - 320), (DX, 1650), (N4[0] - 6, 1650)], GREEN, 14, 50, label="нет: уровень k + 1 — детали на шаг 4", at=1, lfill=GREEN)
d.text((DX - 250, DY - 900), "нет", font=F_HEAD, fill=GREEN)

# Вход в цикл из шага 3 и выход в распознавание.
arrow([(x0 + w0, 4100), (2830, 4100), (2830, 900), (N4[0] - 6, 900)], BLUE, 14, 50)
d.text((2560, 840), "штрихи с кодом угла → уровень 0", font=F_SMALL, fill=BLUE)
arrow([(DX, DY + 320), (DX, 7130)], YELLOW, 14, 50)
d.text((DX + 30, 6900), "да (уровень 3: 98 %) — обучение закончено", font=F_SMALL, fill=YELLOW)

# ---------------------------------------------------------------- шаги 9–10 и итог
N9 = (150, 7150, 3100, 1700)
N10 = (3400, 7150, 3300, 1700)
x, y, w, h = N9
yy = node(x, y, w, h, "9", "Пороги знания и читатель", "этап 7",
          "Д. 100 обучающих картинок заново целиком: шаги 2–3 и все витки по готовым картам (карты не меняются, детекторы только срабатывают) → "
          "пороги «знаю целиком / по частям / не знаю» и линейный читатель над уровнями E(d). С(§2.2.5, §7.1.13.3; В — читатель, §9.3)", YELLOW)
images(x + 40, yy + 10, w - 80, y + h - yy - 50, [("docs/stage7-fives.png", (12, 138, 1150, 555), "проверочные пятёрки: верхние детали, голоса, ближайшая обучающая")])
x, y, w, h = N10
yy = node(x, y, w, h, "10", "Распознавание: память и читатель", "этап 7",
          "Картинка → штрихи → все витки по готовым картам; память — голоса кусочков, читатель — линейный слой. "
          "Дашборд: рисуешь цифру — видишь кусочки и кластеры всех уровней.", YELLOW)
images(x + 40, yy + 10, w - 80, y + h - yy - 50, [(DASH, (15, 60, 305, 590), "рисунок"), (DASH, (330, 55, 710, 595), "уровень 0"),
                                                  (DASH, (720, 55, 1110, 595), "уровень 1"), (DASH, (1520, 55, 1905, 595), "уровень 3")])
arrow([(N9[0] + N9[2], 8000), (N10[0] - 6, 8000)], YELLOW, 12, 44)
rx, ry, rw, rh = 6900, 7150, 1950, 1700
d.rounded_rectangle((rx, ry, rx + rw, ry + rh), radius=40, fill=(40, 34, 18), outline=YELLOW, width=6)
d.text((rx + 40, ry + 40), "Сейчас", font=F_HEAD, fill=YELLOW)
text(rx + 40, ry + 130, "читатель 92 % · память 85 % (100 проверочных)\n"
     "сбор классов наверху: М1 ≈ 0.31 — в главном пятне своей цифры 31 % точек; пятен ~100 вместо 10\n"
     "честная сходимость верха 0.63 · гладкость уровня 0 10.0° · ретинотопия 0.92\n"
     "цель автора — 10 кластеров наверху\n"
     "лучшая находка подбора: плотность эмбеддинга 0.07 — М1 +0.067 (на краю стража)", F_BODY, rw - 80, fill=(240, 230, 200))

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
sheet.save(OUT, optimize=True)
print(f"{OUT}: {S}×{S}, {os.path.getsize(OUT) / 2**20:.1f} МБ")
