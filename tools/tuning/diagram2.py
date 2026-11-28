# -*- coding: utf-8 -*-
"""Подробная квадратная блок-схема DAMP: каждый шаг разбит на подшаги-блоки (у блока — одна отдельная картинка,
вырезанная из составных рисунков и снимков), условия — ромбы, петли — оранжевые стрелки назад.

    python tools/tuning/diagram2.py [выход=data/maps/pipeline-diagram-detailed.png]

Автор 2026-11-21: «можешь порезать в ней скриншоты на отдельные картинки, из которых они состоят? и разделить блоки
в диаграмме на отдельные кусочки».
"""
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

OUT = sys.argv[1] if len(sys.argv) > 1 else "data/maps/pipeline-diagram-detailed.png"
S = 10800
BG, CARD, CELL, EDGE = (13, 14, 18), (25, 27, 34), (36, 39, 50), (95, 100, 115)
BLUE, GREEN, ORANGE, YELLOW, GREY = (70, 140, 240), (70, 205, 140), (240, 140, 60), (245, 205, 70), (205, 207, 215)


def F(size, bold=False):
    return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)


F_T, F_SUB, F_NUM, F_HEAD, F_BODY, F_CELL, F_SMALL = F(150, True), F(54), F(80, True), F(56, True), F(34), F(30, True), F(28)
sheet = Image.new("RGB", (S, S), BG)
d = ImageDraw.Draw(sheet)
_cache = {}


def wrap(s, fnt, width):
    out = []
    for para in s.split("\n"):
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


def text(x, y, s, fnt, width, fill=GREY, gap=1.28, center=False):
    for line in wrap(s, fnt, width):
        dx = (width - d.textlength(line, font=fnt)) / 2 if center else 0
        d.text((x + dx, y), line, font=fnt, fill=fill)
        y += int(fnt.size * gap)
    return y


def crop(path, box):
    if path not in _cache:
        _cache[path] = Image.open(path).convert("RGB")
    im = _cache[path]
    return im.crop(box) if box else im


def arrow(points, colour=GREY, width=8, head=30):
    d.line(points, fill=colour, width=width, joint="curve")
    (x1, y1), (x2, y2) = points[-2], points[-1]
    a = math.atan2(y2 - y1, x2 - x1)
    left = (x2 - head * math.cos(a) + head * 0.6 * math.sin(a), y2 - head * math.sin(a) - head * 0.6 * math.cos(a))
    right = (x2 - head * math.cos(a) - head * 0.6 * math.sin(a), y2 - head * math.sin(a) + head * 0.6 * math.cos(a))
    d.polygon([(x2, y2), left, right], fill=colour)


def diamond(cx, cy, w, h, s, fnt):
    d.polygon([(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)], fill=(56, 46, 16), outline=YELLOW, width=3)
    lines = wrap(s, fnt, w * 0.6)
    y0 = cy - len(lines) * fnt.size * 0.64
    for line in lines:
        d.text((cx - d.textlength(line, font=fnt) / 2, y0), line, font=fnt, fill=(255, 240, 200))
        y0 += fnt.size * 1.28


def cell(x, y, w, h, item):
    """Подшаг: ('box', подпись, (файл, рамка)) — блок с картинкой; ('text', подпись) — блок-действие; ('if', вопрос)."""
    kind = item[0]
    if kind == "if":
        diamond(x + w / 2, y + h / 2, min(w, h * 1.5) - 10, min(h, w / 1.2) - 10, item[1], F_SMALL)
        return
    d.rounded_rectangle((x, y, x + w, y + h), radius=24, fill=CELL, outline=EDGE, width=2)
    if kind == "text":
        lines = wrap(item[1], F_CELL, w - 40)
        yy = y + h / 2 - len(lines) * F_CELL.size * 0.64
        for line in lines:
            d.text((x + w / 2 - d.textlength(line, font=F_CELL) / 2, yy), line, font=F_CELL, fill=(235, 235, 240))
            yy += F_CELL.size * 1.28
        return
    _, label, (path, box) = item
    ty = text(x + 18, y + 14, label, F_CELL, w - 36, fill=(240, 240, 245), gap=1.2)
    im = crop(path, box)
    aw, ah = w - 36, y + h - ty - 22
    k = min(aw / im.width, ah / im.height)
    iw, ih = max(1, int(im.width * k)), max(1, int(im.height * k))
    im = im.resize((iw, ih), Image.LANCZOS)
    px, py = int(x + (w - iw) / 2), int(ty + 8 + (ah - ih) / 2)
    sheet.paste(im, (px, py))
    d.rectangle((px - 2, py - 2, px + iw + 1, py + ih + 1), outline=(70, 74, 86), width=2)


def flow(x, y, w, h, rows, loops=(), colour=GREY):
    """Подшаги по строкам (rows — список строк, строка — список подшагов); стрелки по порядку, переход строки — стрелкой
    вниз-влево; loops — (строка, от, к, подпись): оранжевая петля под строкой от подшага «от» назад к «к»."""
    gap_x, gap_y = 70, 150
    rh = (h - gap_y * (len(rows) - 1)) / len(rows)
    centres = []
    for r, row in enumerate(rows):
        cw = (w - gap_x * (len(row) - 1)) / len(row)
        ry = y + r * (rh + gap_y)
        line = []
        for c, item in enumerate(row):
            cx = x + c * (cw + gap_x)
            cell(cx, ry, cw, rh - (60 if any(lp[0] == r for lp in loops) else 0), item)
            line.append((cx, ry, cw))
            if c:
                px = line[c - 1][0] + line[c - 1][2]
                arrow([(px + 6, ry + rh / 2 - 30), (cx - 6, ry + rh / 2 - 30)], colour)
        centres.append(line)
        if r:
            lx, ly, lw = centres[r - 1][-1]
            fx, fy, fw = line[0]
            mid = ly + rh + gap_y / 2
            arrow([(lx + lw / 2, ly + rh - 4), (lx + lw / 2, mid), (fx + fw / 2, mid), (fx + fw / 2, fy - 6)], colour)
    for r, a, b, label in loops:
        line = centres[r]
        ry = y + r * (rh + gap_y)
        ax, bx = line[a][0] + line[a][2] / 2, line[b][0] + line[b][2] / 2
        bottom = ry + rh - 60
        arrow([(ax, bottom + 4), (ax, bottom + 44), (bx, bottom + 44), (bx, bottom + 8)], ORANGE, 7, 26)
        d.text((bx + 20, bottom + 50), "петля: " + label, font=F_SMALL, fill=ORANGE)


def node(x, y, w, h, num, title, stage, body, colour):
    d.rounded_rectangle((x, y, x + w, y + h), radius=44, fill=None, outline=colour, width=3)
    d.ellipse((x + 30, y + 30, x + 170, y + 170), fill=colour)
    d.text((x + 100 - d.textlength(num, font=F_NUM) / 2, y + 52), num, font=F_NUM, fill=(255, 255, 255))
    d.text((x + 200, y + 34), stage, font=F_SMALL, fill=(150, 200, 255))
    yy = text(x + 200, y + 74, title, F_HEAD, w - 240, fill=(255, 255, 255), gap=1.12)
    return text(x + 40, max(yy, y + 186) + 10, body, F_BODY, w - 80) + 30


# ---------------------------------------------------------------- вырезки (рамки в пикселях исходных рисунков)
ST1, ST2, DIG = "docs/stage1-sticks.png", "docs/stage2-detectors.png", "docs/stage1-digits.png"
ACT, DET, STEP1 = "docs/stage4-activation.png", "docs/stage4-detectors.png", "docs/stage6-step-level1.png"
SIG, GAU, DASH, FIV, TUNE = "data/maps/sigma561-five182-level1.png", "data/maps/gauss-steps-561.png", "docs/dashboard.png", "docs/stage7-fives.png", "data/maps/tune2-576.png"
ANG_X = [397, 518, 640, 762, 884]                       # центры колец слоёв угла и кругов места по столбцам


def ang(layer):                                          # кольцо слоя угла 1…10
    c, r = (layer - 1) % 5, (layer - 1) // 5
    return (ST2, (ANG_X[c] - 62, 180 + r * 155, ANG_X[c] + 62, 330 + r * 155))


def place(layer):                                        # круги места слоя 1…10
    c, r = (layer - 1) % 5, (layer - 1) // 5
    return (ST2, (ANG_X[c] - 62, 622 + r * 155, ANG_X[c] + 62, 772 + r * 155))


def det(layer):                                          # детекторы слоя 1…5 над картой уровня 1
    return (DET, (15 + (layer - 1) * 197, 130, 205 + (layer - 1) * 197, 335))


def tune(n):
    return (TUNE, (20 + n * 320, 74, 320 + n * 320, 374))


def dash_five(n):                                        # штрихи / окна уровня n на рисунке в дашборде
    return (DASH, (330 + n * 400, 90, 500 + n * 400, 260))


def dash_map(n):                                         # карта уровня n с кластерами в дашборде
    return (DASH, (372 + n * 400, 300, 662 + n * 400, 588))


def cluster_digit(k):                                    # уровень 3: где лежит цифра k (синим)
    return ("docs/stage7-class-clusters.png", (15 + k * 166, 750, 90 + k * 166, 828))


# ---------------------------------------------------------------- шапка
d.text((150, 60), "DAMP — путь картинки через сеть: подробно, по подшагам", font=F_T, fill=(255, 255, 255))
text(150, 250, "250 обучающих цифр MNIST, 100 проверочных · конфигурация 561 + перекрытие угла 0.85 + код окна до 8192 (2026-11-21) · "
     "каждый блок — подшаг с одной картинкой, вырезанной из рисунков README, дашборда, рисунков по шагам на пятёрке №182 и карт подбора · "
     "жёлтый ромб — условие, оранжевая стрелка назад — петля · С — по статье, Д — наше расширение, В — вне статьи", F_SUB, S - 300, fill=(185, 188, 198))

# ---------------------------------------------------------------- шаги 1–3 (слева)
X0, W0 = 150, 2700
y = node(X0, 480, W0, 1500, "1", "Картинки MNIST", "этап 1", "Первые 250 картинок MNIST train; проверочные — 100 из t10k. С(§10.1.4)", BLUE)
flow(X0 + 40, y, W0 - 80, 1980 - y - 40, [[("box", "MNIST 28×28 → 224×224 (×8)", (ST1, (80, 250, 420, 530))),
                                         ("box", "пятёрка: её штрихи", (DIG, (40, 470, 230, 660))),
                                         ("box", "ноль", (DIG, (60, 150, 220, 320))), ("box", "четвёрка", (DIG, (1080, 150, 1240, 330)))]], colour=BLUE)
y = node(X0, 2060, W0, 1500, "2", "Штрихи", "этап 1",
         "Д. Штрих — детектор над точками середины линии: угол 0…180° + место; ~53 на цифру. С(§6.4)", BLUE)
flow(X0 + 40, y, W0 - 80, 3560 - y - 40, [[("box", "гребень расстояния до края: точки середины линии", (ST1, (520, 260, 840, 520))),
                                         ("box", "соседние точки с похожим направлением → штрих", (ST1, (960, 260, 1280, 520))),
                                         ("box", "дашборд: штрихи рисунка", dash_five(0))]], colour=BLUE)
y = node(X0, 3640, W0, 2200, "3", "Код угла штриха", "этап 2",
         "С. 10 слоёв детекторов угла, перекрытие 0.85; сработавший ставит случайный бит в полосе своего слоя; 256 бит, 22 %. С(§3.2, §4.2, §4.4)", BLUE)
flow(X0 + 40, y, W0 - 80, 5840 - y - 40, [[("box", "угол штриха (105°): 10 слоёв кольцами", (ST2, (20, 170, 320, 490))),
                                          ("box", "слой 1: 2 детектора", ang(1)), ("box", "слой 3: 5", ang(3))],
                                         [("box", "слой 6: 24", ang(6)), ("box", "слой 10: 180", ang(10)),
                                          ("box", "код угла: бит на сработавший детектор", (ST2, (955, 175, 1345, 280)))]], colour=BLUE)

# ---------------------------------------------------------------- цикл
FX, FY, FW, FH = 3050, 470, 7600, 7350
d.rounded_rectangle((FX, FY, FX + FW, FY + FH), radius=70, outline=GREEN, width=5)
d.rectangle((FX + 90, FY - 46, FX + 5000, FY + 46), fill=BG)
d.text((FX + 110, FY - 40), "ЦИКЛ (этапы 2 → 6): шаги 4–8 повторяются на уровнях 0 → 1 → 2 → 3", font=F_HEAD, fill=GREEN)
N4 = (3700, 650, 3250, 2450)
N5 = (7150, 650, 3350, 2450)
N6 = (7150, 3300, 3350, 1950)
N7 = (7150, 5450, 3350, 2250)
N8 = (3700, 5450, 3250, 2250)
CEN = (3700, 3300, 3250, 1950)

x, y0, w, h = N4
y = node(x, y0, w, h, "4", "Окна и их коды", "этап 2 ← начало витка",
         "Д. Код окна = место всех элементов (OR) + содержание (код элемента под XOR-маской места), до 8192 бит. С(П.3–П.5)", GREEN)
flow(x + 40, y, w - 80, y0 + h - y - 40, [
    [("box", "место штриха — от центра цифры", (ST2, (20, 610, 330, 900))), ("box", "круги места: слой 1 (2×2)", place(1)),
     ("box", "слой 5 (6×6)", place(5)), ("box", "слой 10 (16×16)", place(10)), ("box", "код места", (ST2, (955, 620, 1345, 700)))],
    [("box", "уровень 0: код окна = место + угол", (ST2, (15, 1020, 535, 1160))), ("if", "уровень ≥ 1: кластер ≥ N похожих соседей?"),
     ("text", "нет — окно на всю картинку"), ("box", "окно уровня 1 на пятёрке", (STEP1, (92, 218, 402, 483))),
     ("box", "место (оранж.) + содержание (синие)", (STEP1, (535, 132, 805, 523)))]], colour=GREEN)

x, y0, w, h = N5
y = node(x, y0, w, h, "5", "Первая подача: раскладка → карта", "этап 3",
         "С. Коды окон всех обучающих картинок — по сетке (15 % пустых); похожее ложится рядом; окна с косинусом ≥ 0.8 сливаются. С(§5.2–§5.9, §7.1.6)", GREEN)
flow(x + 40, y, w - 80, y0 + h - y - 40, [
    [("text", "пара точек: первая — по −ln Ê"), ("if", "энергия после обмена выше?"), ("text", "обменять места"),
     ("box", "колокол в начале: r = 62", (GAU, (1335, 225, 1500, 435))), ("box", "в середине: r = 14", (GAU, (1502, 225, 1667, 435))),
     ("box", "в конце: r = 3", (GAU, (1668, 225, 1832, 435)))],
    [("box", "вес соседа w(d) от расстояния", (GAU, (140, 720, 900, 1160))), ("box", "итог, уровень 0: пинвилы углов", tune(0)),
     ("box", "крупно: углы плавно по кругу", (GAU, (1233, 725, 1608, 1100))), ("box", "ур. 0: штрих в ячейке", ("docs/stage3-layout-level0.png", (20, 160, 310, 305))),
     ("box", "ур. 1: что видит окно точки", ("docs/stage3-layout.png", (18, 150, 480, 360)))]],
    loops=[(0, 2, 0, "дальняя фаза ~2 300 шагов × 4 656 пар; колокол сжимается; затем ближняя фаза 20 шагов")], colour=GREEN)

x, y0, w, h = N6
y = node(x, y0, w, h, "6", "Детекторы над картой", "этап 4",
         "С. 5 слоёв; кластер DBSCAN вокруг центра с энергией ≥ μ_e; радиус по заполнению; центры не перекрываются. С(§6.4, §7.1.10)", GREEN)
flow(x + 40, y, w - 80, y0 + h - y - 40, [
    [("text", "случайный центр, энергия ≥ μ_e"), ("text", "кластер DBSCAN"), ("text", "радиус по заполнению"),
     ("if", "центр не перекрыт?"), ("text", "детектор в слой")],
    [("box", "слой 1 (λ 0.44)", det(1)), ("box", "слой 2", det(2)), ("box", "слой 3", det(3)), ("box", "слой 4", det(4)),
     ("box", "слой 5 (λ 0.56)", det(5)), ("box", "сработавшие на окне", (SIG, (1353, 131, 1808, 581)))]],
    loops=[(0, 4, 0, "пока 400 попыток подряд без нового детектора; затем следующий слой")], colour=GREEN)

x, y0, w, h = N7
y = node(x, y0, w, h, "7", "Вторая подача: активация → деталь", "этапы 4–5",
         "С. Код того же окна сравнивается с картой; детекторы срабатывают по энергии зажжённых точек; в код детали — σ = 10 сильнейших. С(§6.1, §6.5, §7.1.12)", GREEN)
flow(x + 40, y, w - 80, y0 + h - y - 40, [
    [("box", "окно на цифре — стимул", (ACT, (90, 230, 400, 480))), ("if", "сходство с точкой ≥ λ_a?"),
     ("box", "зажглись точки памяти", (ACT, (480, 190, 860, 560))), ("if", "энергия активации ≥ 0.5? (Д)"),
     ("box", "сработали детекторы", (ACT, (920, 190, 1300, 560)))],
    [("box", "уровни E(d) по убыванию", (SIG, (81, 707, 1010, 1162))), ("text", "σ = 10 сильнейших: по случайному биту"),
     ("box", "код детали (100 бит)", (SIG, (1242, 697, 1626, 1111))), ("box", "детали на цифре", (STEP1, (552, 862, 805, 1058)))]], colour=GREEN)

x, y0, w, h = N8
y = node(x, y0, w, h, "8", "Пятна и переход вверх", "этап 6 ← конец витка",
         "Д. Из соседних похожих деталей одной картинки остаётся сильнейшая; детали — элементы окон следующего уровня. С(П.2, П.3, П.7)", GREEN)
flow(x + 40, y, w - 80, y0 + h - y - 40, [
    [("box", "детали уровня k картинки", (STEP1, (552, 862, 805, 1058))), ("if", "рядом похожая деталь (≥ 0.8)?"),
     ("text", "оставить сильнейшую")],
    [("box", "окна уровней 1–3 вокруг одного места", ("docs/stage6-keyholes.png", (60, 60, 660, 660))),
     ("box", "дашборд: окна уровня 1", dash_five(1)), ("box", "окна уровня 2", dash_five(2)), ("box", "окна уровня 3", dash_five(3))]], colour=GREEN)

x, y0, w, h = CEN
d.rounded_rectangle((x, y0, x + w, y0 + h), radius=44, fill=None, outline=GREEN, width=3)
d.text((x + 40, y0 + 30), "Витки цикла: окно растёт", font=F_HEAD, fill=GREEN)
text(x + 40, y0 + 110, "ур. 0: 1 штрих, 2 % цифры · ур. 1: ~11 штрихов, 21 % · ур. 2: ~5 деталей, 35 % · ур. 3: ~27 деталей, 98 %", F_BODY, w - 80)
flow(x + 40, y0 + 230, w - 80, h - 270, [
    [("box", "уровень 0: 13.3 тыс.", tune(0)), ("box", "уровень 1: 10.4 тыс.", tune(1)), ("box", "уровень 2: 8.1 тыс.", tune(2)),
     ("box", "уровень 3: 4.5 тыс.", tune(3))],
    [("box", f"ур. 3: «{k}»", cluster_digit(k)) for k in range(10)]], colour=GREEN)

# Стрелки цикла и ромб «стоп».
arrow([(N4[0] + N4[2], 1900), (N5[0] - 6, 1900)], GREEN, 16, 54)
arrow([(N5[0] + N5[2] / 2, N5[1] + N5[3]), (N6[0] + N6[2] / 2, N6[1] - 6)], GREEN, 16, 54)
arrow([(N6[0] + N6[2] / 2, N6[1] + N6[3]), (N7[0] + N7[2] / 2, N7[1] - 6)], GREEN, 16, 54)
arrow([(N7[0], 6600), (N8[0] + N8[2] + 6, 6600)], GREEN, 16, 54)
DX, DY = 3370, 6600
diamond(DX, DY, 560, 700, "окно уровня видит ≥ 90 % цифры?", F_SMALL)
arrow([(N8[0], DY), (DX + 284, DY)], GREEN, 16, 54)
arrow([(DX, DY - 352), (DX, 1900), (N4[0] - 6, 1900)], GREEN, 16, 54)
d.text((DX - 260, 3900), "нет:", font=F_HEAD, fill=GREEN)
text(DX - 300, 3980, "уровень k + 1 — детали на шаг 4", F_SMALL, 260, fill=GREEN)
arrow([(X0 + W0, 4700), (3200, 4700), (3200, 1050), (N4[0] - 6, 1050)], BLUE, 16, 54)
d.text((2900, 980), "штрихи → уровень 0", font=F_SMALL, fill=BLUE)
arrow([(DX, DY + 352), (DX, 8050)], YELLOW, 16, 54)
d.text((DX + 30, 7860), "да (ур. 3: 98 %) — обучение закончено", font=F_SMALL, fill=YELLOW)

# ---------------------------------------------------------------- шаги 9–10 и итог
N9 = (150, 8100, 4300, 2550)
N10 = (4650, 8100, 4100, 2550)
x, y0, w, h = N9
y = node(x, y0, w, h, "9", "Пороги знания и читатель", "этап 7",
         "Д. 100 обучающих картинок заново целиком по готовым картам → пороги «знаю целиком / по частям / не знаю» и линейный читатель. "
         "С(§2.2.5, §7.1.13.3; В — читатель, §9.3)", YELLOW)
flow(x + 40, y, w - 80, y0 + h - y - 40, [
    [("box", "проверочная пятёрка", (FIV, (12, 345, 200, 540))), ("box", "верхние детали и их голоса", (FIV, (205, 345, 390, 540))),
     ("box", "голоса по цифрам: ответ «2» — ошибка", (FIV, (425, 345, 745, 565))),
     ("box", "ближайшие обучающие по уровням", (FIV, (745, 345, 1135, 470)))]], colour=YELLOW)
x, y0, w, h = N10
y = node(x, y0, w, h, "10", "Распознавание: рисунок → уровни → ответ", "этап 7",
         "Картинка → штрихи → все витки по готовым картам; память — голоса кусочков, читатель — линейный слой.", YELLOW)
flow(x + 40, y, w - 80, y0 + h - y - 40, [
    [("box", "рисунок", (DASH, (20, 85, 300, 365))), ("box", "ур. 0: штрихи", dash_five(0)), ("box", "ур. 0: зажглись кластеры", dash_map(0)),
     ("box", "ур. 1: кусочки", dash_five(1)), ("box", "ур. 1: кластеры", dash_map(1))],
    [("box", "ур. 2: кусочки", dash_five(2)), ("box", "ур. 2: кластеры", dash_map(2)), ("box", "ур. 3: кусочки", dash_five(3)),
     ("box", "ур. 3: кластеры", dash_map(3)), ("box", "ответ памяти", (DASH, (15, 520, 305, 600)))]], colour=YELLOW)
arrow([(N9[0] + N9[2], 9350), (N10[0] - 6, 9350)], YELLOW, 14, 48)
rx, ry, rw, rh = 8950, 8100, 1700, 2550
d.rounded_rectangle((rx, ry, rx + rw, ry + rh), radius=44, fill=None, outline=YELLOW, width=3)
d.text((rx + 40, ry + 40), "Сейчас", font=F_HEAD, fill=YELLOW)
text(rx + 40, ry + 140, "читатель 92 %, память 85 % (100 проверочных)\n\n"
     "сбор классов наверху: М1 ≈ 0.31 — в главном пятне своей цифры 31 % точек; пятен ~100 вместо 10\n\n"
     "честная сходимость верха 0.64 · гладкость ур. 0 10.0° · ретинотопия 0.92\n\n"
     "цель — 10 кластеров наверху\n\n"
     "лучшая находка подбора: плотность эмбеддинга 0.07 — М1 +0.05 по 4 зёрнам (ниже стража честной)", F_BODY, rw - 80, fill=(240, 230, 200))

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
sheet.save(OUT, optimize=True)
BLOCKS = {"1-kartinki": (X0, 480, W0, 1500), "2-shtrihi": (X0, 2060, W0, 1500), "3-kod-ugla": (X0, 3640, W0, 2200),
          "4-okna-i-kody": N4, "5-raskladka": N5, "6-detektory": N6, "7-aktivaciya-detal": N7, "8-pyatna": N8,
          "vitki-cikla": CEN, "9-porogi-znaniya": N9, "10-raspoznavanie": N10, "cikl-celikom": (FX - 40, FY - 80, FW + 80, FH + 120)}
folder = os.path.join(os.path.dirname(OUT) or ".", "diagram-blocks")
os.makedirs(folder, exist_ok=True)
for name, (bx, by, bw, bh) in BLOCKS.items():
    sheet.crop((int(bx) - 20, int(by) - 20, int(bx + bw) + 20, int(by + bh) + 20)).save(os.path.join(folder, f"{name}.png"), optimize=True)
print(f"блоки по отдельности: {folder} ({len(BLOCKS)} файлов)")
print(f"{OUT}: {S}×{S}, {os.path.getsize(OUT) / 2**20:.1f} МБ")
